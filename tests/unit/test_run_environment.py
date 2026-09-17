from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes.fake_run_port import FakeCommandResult, FakeRunPort  # noqa: E402

from tower_rl.application.run_environment import (  # noqa: E402
    BRIDGE_EVENT_DIVERGENCE,
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.domain.episode import (  # noqa: E402
    ActionOutcome,
    DecisionEvent,
    TerminationOutcome,
)
from tower_rl.domain.run_actions import WAIT, upgrade_action  # noqa: E402
from tower_rl.domain.run_state import RunStateBuilder  # noqa: E402
from tower_rl.ports.run_port import RunPortError  # noqa: E402


def _environment(**port_kwargs: object) -> tuple[InstrumentedRunEnvironment, FakeRunPort]:
    port = FakeRunPort(**port_kwargs)  # type: ignore[arg-type]
    environment = InstrumentedRunEnvironment(
        port=port,
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )
    return environment, port


def test_reset_returns_a_valid_active_state() -> None:
    environment, _ = _environment()

    state = environment.reset()

    assert state.valid and state.lifecycle == "active"
    assert state.wave == 1
    assert environment.state is state


def test_reset_surfaces_an_instance_that_will_not_start() -> None:
    environment, _ = _environment(refuse_to_start=True)

    with pytest.raises(RunPortError, match="refused to start"):
        environment.reset()


def test_a_masked_action_is_refused_without_reaching_the_game() -> None:
    environment, port = _environment()
    environment.reset()
    before = port.advances

    transition = environment.step(upgrade_action("utility", 0))  # never offered

    assert transition.outcome is ActionOutcome.UNAVAILABLE
    assert transition.invalid_reasons == ("action is masked",)
    assert not transition.admissible
    assert port.advances == before, "a masked action must not advance the game"


def test_a_purchase_is_confirmed_and_immediately_re_decided() -> None:
    environment, _ = _environment()
    state = environment.reset()
    target = next(row for row in state.rows if row.available)

    transition = environment.step(target.action)

    assert transition.outcome is ActionOutcome.EXECUTED
    assert transition.events == (DecisionEvent.PURCHASE_SETTLED,)
    # The decision problem changed, so no game time is advanced first.
    assert transition.requested_game_ms == 0
    assert transition.next_state is not None
    after = next(row for row in transition.next_state.rows if row.action == target.action)
    assert after.level == 1
    assert transition.next_state.cash_log < state.cash_log
    assert transition.admissible


def test_waiting_advances_until_something_actionable_changes() -> None:
    environment, port = _environment()
    environment.reset()

    transition = environment.step(WAIT)

    assert transition.outcome is ActionOutcome.WAITED
    assert transition.events != ()
    assert transition.requested_game_ms > 0
    assert transition.next_state is not None
    assert transition.invalid_reasons == ()
    # One decision is one bridge command now, not one command per slice.
    assert port.advances == 1


def test_a_decision_costs_no_read_beyond_its_own_advance() -> None:
    """The advance returns the settled state, so re-reading would be a second trip.

    The bridge already paused the world and let the pause land before answering.
    Waiting for the next stream tick would pay a round trip per decision for a
    state that can only be later than the one the result describes.
    """
    environment, port = _environment()
    environment.reset()
    reads = port.reads

    for _ in range(3):
        transition = environment.step(WAIT)
        assert transition.next_state is not None

    assert port.advances == 3
    assert port.reads == reads, "an advance already carries the state it settled at"


def test_a_purchase_costs_no_read_beyond_its_own_command_result() -> None:
    """A confirmed purchase carries its settled state too, the same mechanism.

    `buy_upgrade`'s result binds the state the bridge sent immediately before
    it, exactly as `advance_until_event`'s does. Re-reading afterwards would
    pay a ~250 ms stream-tick wait for a state the result already describes.
    """
    environment, port = _environment()
    state = environment.reset()
    target = next(row for row in state.rows if row.available)
    reads = port.reads

    transition = environment.step(target.action)

    assert transition.outcome is ActionOutcome.EXECUTED
    assert transition.next_state is not None
    assert port.reads == reads, "a purchase already carries the state it settled at"


def test_a_lying_bridge_reason_is_recorded_as_a_divergence() -> None:
    """The host predicate is authoritative; a disagreement is made visible."""
    environment, port = _environment(
        damage_per_second=0.0, cash_per_second=0.0, seconds_per_wave=10_000.0
    )
    environment.reset()
    honest = port.advance_until_event

    def claims_a_wave_that_never_came(**kwargs: object) -> FakeCommandResult:
        result = honest(**kwargs)  # type: ignore[arg-type]
        return FakeCommandResult(
            result.outcome, "event:wave_changed", result.frames, result.game_ms,
            state=result.state,
        )

    port.advance_until_event = claims_a_wave_that_never_came  # type: ignore[method-assign]

    transition = environment.step(WAIT)

    assert transition.next_state is not None
    assert transition.next_state.wave == transition.state.wave
    assert BRIDGE_EVENT_DIVERGENCE in transition.invalid_reasons
    assert not transition.admissible
    assert environment.summarize(TerminationOutcome.OPERATOR_STOP).invalid_transitions == 1


def test_an_unconfirmed_advance_fails_the_episode_rather_than_passing_as_a_wait() -> None:
    environment, port = _environment()
    environment.reset()

    def cannot_say_how_far_it_got(**_kwargs: object) -> FakeCommandResult:
        return FakeCommandResult("ambiguous", "no_frame_rendered")

    port.advance_until_event = cannot_say_how_far_it_got  # type: ignore[method-assign]

    transition = environment.step(WAIT)

    assert transition.outcome is ActionOutcome.AMBIGUOUS
    assert transition.termination is TerminationOutcome.ACTION_PIPELINE_FAILED
    assert transition.truncated and not transition.admissible
    assert any("no_frame_rendered" in reason for reason in transition.invalid_reasons)


def test_an_advance_cut_short_of_its_budget_is_counted_not_hidden() -> None:
    """The bridge has its own wall ceiling: stopping early is slowness, not an event."""
    environment, port = _environment()
    environment.reset()

    def stopped_on_the_wall_ceiling(**_kwargs: object) -> FakeCommandResult:
        return FakeCommandResult(
            "confirmed", "budget_exhausted", frames=3, game_ms=50.0,
            state=port.read_state(),
        )

    port.advance_until_event = stopped_on_the_wall_ceiling  # type: ignore[method-assign]

    transition = environment.step(WAIT)

    assert transition.admissible, "a short advance is still a genuine transition"
    assert environment.summarize(TerminationOutcome.OPERATOR_STOP).advances_cut_short == 1


def test_the_summary_reports_what_advancing_cost_the_game_clock() -> None:
    """Game seconds over wall seconds is the speed-up the design is judged on."""
    environment, _ = _environment(damage_per_second=4.0)
    environment.reset()

    for _ in range(50):
        transition = environment.step(WAIT)
        if transition.terminated:
            break
    else:
        pytest.fail("the tower never died")

    summary = environment.summarize(transition.termination)
    assert summary.frames > 0
    assert summary.game_ms > 0.0
    # The game's own clock, beside the budgeted game time it is meant to equal.
    assert summary.round_ms == pytest.approx(summary.game_ms)
    # Wall time inside advances, which the report subtracts from total wall time
    # to show what the decision boundaries cost. The double invents its own
    # figure, so only that it is carried through is testable here.
    assert summary.advance_wall_seconds > 0.0


def test_reward_is_wave_progress_only() -> None:
    environment, _ = _environment(cash_per_second=0.0, seconds_per_wave=0.5)
    environment.reset()

    rewards = []
    for _ in range(8):
        transition = environment.step(WAIT)
        rewards.append(transition.reward)
        if transition.terminated or transition.truncated:
            break

    assert sum(rewards) > 0, "waves advanced, so return must be positive"
    assert all(reward >= 0 for reward in rewards)
    # A purchase on its own earns nothing; only surviving into a new wave does.
    environment, _ = _environment(seconds_per_wave=10_000.0)
    state = environment.reset()
    target = next(row for row in state.rows if row.available)
    assert environment.step(target.action).reward == 0.0


def test_a_dead_tower_terminates_as_game_over() -> None:
    environment, _ = _environment(damage_per_second=4.0)
    environment.reset()

    for _ in range(50):
        transition = environment.step(WAIT)
        if transition.terminated:
            break
    else:
        pytest.fail("the tower never died")

    assert transition.termination is TerminationOutcome.GAME_OVER
    assert transition.next_state is not None and transition.next_state.terminal
    summary = environment.summarize(transition.termination)
    assert summary.valid and summary.final_wave >= 1
    assert summary.decisions > 0


def test_episode_summary_counts_purchases_and_invalid_transitions() -> None:
    environment, _ = _environment()
    state = environment.reset()
    target = next(row for row in state.rows if row.available)

    environment.step(target.action)
    environment.step(upgrade_action("utility", 0))  # masked, hence invalid

    summary = environment.summarize(TerminationOutcome.OPERATOR_STOP)

    assert summary.purchases == 1
    assert summary.decisions == 2
    assert summary.invalid_transitions == 1
    assert not summary.valid, "only a game over counts as a valid episode"


def test_a_stalled_run_truncates_rather_than_running_forever() -> None:
    environment, port = _environment(damage_per_second=0.0, seconds_per_wave=10_000.0)
    environment.cadence = CadenceConfig(
        max_quiet_game_ms=500, max_episode_wall_seconds=0.0
    )
    environment.reset()

    transition = environment.step(WAIT)

    assert transition.truncated
    assert transition.termination is TerminationOutcome.MAX_EPISODE_DURATION
    assert not transition.admissible
    assert port.active, "truncation is an environment decision, not a game over"


def test_the_death_boundary_is_settled_by_advancing_not_by_reading_again() -> None:
    """M1B-E008: health goes negative a moment before game-over flips.

    Between decisions the bridge holds the world paused, so a second read returns
    the identical reading however many times it is asked. Only another frame of
    game time can settle the boundary, so the recovery advances minimally.
    """
    from dataclasses import replace

    environment, port = _environment()
    environment.reset()
    original = port.read_state

    def frozen_read():
        # A paused world: the same contradictory instant, read as often as liked.
        reading = original()
        return None if reading is None else replace(
            reading, health=-2.0, lifecycle="active", terminal=False
        )

    port.read_state = frozen_read  # type: ignore[method-assign]
    advances = port.advances
    state = environment._read_state()

    assert state is not None and state.valid, state.invalid_reasons
    assert port.advances == advances + 1, "the world has to move for the boundary to settle"
    assert environment._tally.recovered_transients == 1


def test_the_recovery_advance_asks_for_one_frame_and_no_more() -> None:
    """Recovering costs the smallest amount of game time the protocol carries."""
    from dataclasses import replace

    environment, port = _environment()
    environment.cadence = CadenceConfig(max_quiet_game_ms=1000, frame_game_ms=100.0)
    environment.reset()
    original = port.advance_until_event
    asked: list[int] = []

    def recording_advance(**kwargs):
        asked.append(kwargs["budget_game_ms"])
        return original(**kwargs)

    port.advance_until_event = recording_advance  # type: ignore[method-assign]
    reading = port.read_state()
    assert reading is not None
    environment._build_state(replace(reading, health=-2.0, lifecycle="active", terminal=False))

    assert asked == [100], "one frame's worth of game time, not a decision's worth"


def test_a_persistently_contradictory_state_is_still_invalid() -> None:
    """A boundary that survives another frame is a real failure, not a transient."""
    from dataclasses import replace

    environment, port = _environment()
    environment.reset()
    original = port.advance_until_event

    def contradictory_advance(**kwargs):
        result = original(**kwargs)
        assert result.state is not None
        return replace(
            result,
            state=replace(result.state, health=-2.0, lifecycle="active", terminal=False),
        )

    port.advance_until_event = contradictory_advance  # type: ignore[method-assign]
    reading = port.read_state()
    assert reading is not None
    state = environment._build_state(
        replace(reading, health=-2.0, lifecycle="active", terminal=False)
    )

    assert state is not None and not state.valid
    assert "negative health in an active run" in state.invalid_reasons
    assert environment._tally.recovered_transients == 0, "nothing admissible was recovered"


def test_a_recovery_advance_that_is_not_confirmed_fails_the_episode_visibly() -> None:
    """An unrecoverable boundary is a pipeline failure, never a silent wait."""
    from dataclasses import replace

    environment, port = _environment()
    environment.reset()
    original = port.advance_until_event
    calls = {"count": 0}

    def failing_recovery(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            result = original(**kwargs)
            assert result.state is not None
            return replace(
                result,
                state=replace(result.state, health=-2.0, lifecycle="active", terminal=False),
            )
        return FakeCommandResult("ambiguous", "no_frame_rendered")

    port.advance_until_event = failing_recovery  # type: ignore[method-assign]
    transition = environment.step(WAIT)

    assert transition.termination is TerminationOutcome.ACTION_PIPELINE_FAILED
    assert transition.outcome is ActionOutcome.AMBIGUOUS
    assert not transition.admissible
    assert any("death boundary did not settle" in reason for reason in transition.invalid_reasons)
    assert environment._tally.recovered_transients == 0


def test_the_summary_reports_the_speed_the_run_actually_executed_at() -> None:
    """M1B-E009: the terminal state reports zero, because the game stops time."""
    environment, port = _environment(damage_per_second=4.0)
    state = environment.reset()
    running_speed = state.game_speed

    for _ in range(50):
        transition = environment.step(WAIT)
        if transition.terminated:
            break
    else:
        pytest.fail("the tower never died")

    assert transition.next_state is not None and transition.next_state.terminal
    summary = environment.summarize(transition.termination)
    assert summary.game_speed == running_speed
    assert summary.game_speed > 0.0, "a speed of zero says nothing about the run"
