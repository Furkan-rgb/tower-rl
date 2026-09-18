from __future__ import annotations

import math

import pytest
from fakes.fake_run_port import FakeCommandResult, FakeRunPort

from tower_rl.environment.episode import (
    ActionOutcome,
    DecisionEvent,
    DecisionView,
    TerminationOutcome,
)
from tower_rl.environment.run_actions import WAIT, upgrade_action
from tower_rl.environment.run_environment import (
    ADVANCE_TRUNCATED_BY_WALL,
    BRIDGE_EVENT_DIVERGENCE,
    GAME_TIME_DEFLATED,
    GAME_TIME_INFLATED,
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.environment.run_port import RunPortError
from tower_rl.environment.run_state import RunStateBuilder


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
    """A loop stopped on a reading its settled state does not corroborate.

    It spent neither its budget nor ended on an event the settled state still
    shows. The observation the agent receives is that settled state, so the
    transition is genuine - but a rise in the count says the loop and the state
    it reports are drifting apart, so it is counted (M1B-E032).
    """
    environment, port = _environment()
    environment.reset()

    def stopped_short_of_the_budget(**_kwargs: object) -> FakeCommandResult:
        return FakeCommandResult(
            "confirmed", "budget_exhausted", frames=3, game_ms=50.0,
            state=port.read_state(),
        )

    port.advance_until_event = stopped_short_of_the_budget  # type: ignore[method-assign]

    transition = environment.step(WAIT)

    assert transition.admissible, "a short advance is still a genuine transition"
    assert environment.summarize(TerminationOutcome.OPERATOR_STOP).advances_cut_short == 1


def test_an_advance_truncated_by_wall_time_fails_the_episode_by_name() -> None:
    """No advance may be truncated by wall time.

    The bridge reports that case under its own reason rather than as a spent
    budget, because the settled state cannot tell the two apart. An advance cut
    off by how long the host took is load-dependent, so the episode it belongs
    to is not comparable with one that was not, and is failed by name rather
    than absorbed into `advances_cut_short`.
    """
    environment, port = _environment()
    environment.reset()

    def stopped_on_the_wall_ceiling(**_kwargs: object) -> FakeCommandResult:
        return FakeCommandResult(
            "confirmed", "wall_ceiling", frames=3, game_ms=50.0,
            state=port.read_state(),
        )

    port.advance_until_event = stopped_on_the_wall_ceiling  # type: ignore[method-assign]

    transition = environment.step(WAIT)

    assert transition.termination is TerminationOutcome.OBSERVATION_INVALID
    assert not transition.admissible
    assert ADVANCE_TRUNCATED_BY_WALL in transition.invalid_reasons
    summary = environment.summarize(transition.termination)
    assert not summary.valid, "a truncated episode may not count towards the curve"
    assert ADVANCE_TRUNCATED_BY_WALL in summary.termination_detail
    assert summary.advances_cut_short == 0, "the two conditions are not the same thing"


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


def test_the_summary_records_the_wave_the_episode_actually_started_at() -> None:
    """A fresh run starts at wave 1; anything higher continued a leftover run."""
    environment, _ = _environment()
    environment.reset()

    summary = environment.summarize(TerminationOutcome.OPERATOR_STOP)

    assert summary.starting_wave == 1


def test_a_leftover_run_is_recorded_rather_than_started_fresh() -> None:
    """`begin_episode` refreshes a frozen leftover run but still continues it.

    Nothing here recovers from the contamination; the point is only that it
    stays visible in the record.
    """
    environment, _ = _environment(starting_wave=3)
    environment.reset()

    summary = environment.summarize(TerminationOutcome.OPERATOR_STOP)

    assert summary.starting_wave == 3


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
    assert environment.summarize(TerminationOutcome.OPERATOR_STOP).recovered_transients == 1


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


def test_waiting_all_the_way_to_death_stays_valid_at_the_default_damage_rate() -> None:
    """The fake's stopping predicate must read the health it actually transmits.

    A wait-only episode at the default damage rate lands the health-change
    threshold squarely on the rounding boundary: the double once decided on its
    internal float and reported `event:health_changed` for a delta the
    transmitted reading put just under the threshold, so every such episode ended
    with zero valid transitions and the divergence check - correctly - fired.
    """
    environment, _ = _environment()
    environment.reset()

    for _ in range(200):
        transition = environment.step(WAIT)
        assert transition.invalid_reasons == ()
        assert transition.admissible
        if transition.terminated:
            break
    else:
        pytest.fail("the tower never died")

    summary = environment.summarize(transition.termination)
    assert summary.valid and summary.invalid_transitions == 0


def test_a_world_that_outruns_its_budget_fails_the_episode_by_name() -> None:
    """M1B-E023: a faster world must not be allowed to report a flattering wave.

    The bridge asks each frame to be worth `frame_game_ms` of game time and
    reports the game's own round clock beside that budget. When the round clock
    runs away from it the world is simulating time the record never asked for,
    and every wave it reaches is measured in a different unit from the runs it
    would be compared with. The episode is failed and counted, never rescaled.
    """
    environment, _ = _environment(world_time_scale=1.5)
    environment.reset()

    for _ in range(20):
        transition = environment.step(WAIT)
        if transition.termination is not None:
            break
    else:
        pytest.fail("the inflated clock was never noticed")

    assert transition.termination is TerminationOutcome.OBSERVATION_INVALID
    assert not transition.admissible
    assert any(GAME_TIME_INFLATED in reason for reason in transition.invalid_reasons)
    summary = environment.summarize(transition.termination)
    assert summary.invalid_transitions == 1
    assert not summary.valid, "an inflated episode may not count towards the curve"
    assert any(GAME_TIME_INFLATED in reason for reason in summary.termination_detail)


def test_a_world_that_keeps_to_its_budget_is_not_accused_of_running_fast() -> None:
    environment, _ = _environment()
    environment.reset()

    for _ in range(20):
        transition = environment.step(WAIT)
        if transition.terminated:
            break
        assert transition.admissible
        assert not any(GAME_TIME_INFLATED in reason for reason in transition.invalid_reasons)
    assert environment.summarize(TerminationOutcome.OPERATOR_STOP).invalid_transitions == 0


def test_a_world_that_undercredits_its_budget_fails_the_episode_by_a_distinct_name() -> None:
    """A ratio sitting well below 1.0 is a fidelity failure in the other direction.

    A 150ms-step arm measured a sustained 0.987 where healthy runs pooled
    1.007-1.014, and the guard could not see it because it only ever looked
    upward. This is the mirror of `test_a_world_that_outruns_its_budget_...`,
    named distinctly so a report says which direction the clock disagreed.
    """
    environment, _ = _environment(world_time_scale=0.95)
    environment.reset()

    for _ in range(20):
        transition = environment.step(WAIT)
        if transition.termination is not None:
            break
    else:
        pytest.fail("the deflated clock was never noticed")

    assert transition.termination is TerminationOutcome.OBSERVATION_INVALID
    assert not transition.admissible
    assert any(GAME_TIME_DEFLATED in reason for reason in transition.invalid_reasons)
    assert not any(GAME_TIME_INFLATED in reason for reason in transition.invalid_reasons)
    summary = environment.summarize(transition.termination)
    assert summary.invalid_transitions == 1
    assert not summary.valid, "a deflated episode may not count towards the curve"
    assert any(GAME_TIME_DEFLATED in reason for reason in summary.termination_detail)


def test_a_world_that_keeps_to_its_budget_is_not_accused_of_running_slow() -> None:
    environment, _ = _environment()
    environment.reset()

    for _ in range(20):
        transition = environment.step(WAIT)
        if transition.terminated:
            break
        assert transition.admissible
        assert not any(GAME_TIME_DEFLATED in reason for reason in transition.invalid_reasons)
    assert environment.summarize(TerminationOutcome.OPERATOR_STOP).invalid_transitions == 0


def test_the_final_advance_of_a_healthy_run_does_not_trip_the_lower_bound() -> None:
    """The round clock resets with the round, so the last advance reports none.

    That legitimate zero must not be mistaken for the world under-simulating
    time: `MIN_RATIO_EVIDENCE_GAME_MS` is what keeps one advance's worth of
    round time - lost to the reset, not to a defect - from swamping an
    episode's worth already accumulated at 1x.
    """
    environment, _ = _environment(round_clock_resets_on_death=True)
    environment.reset()

    for _ in range(200):
        transition = environment.step(WAIT)
        if transition.termination is not None:
            break
    else:
        pytest.fail("the tower never died")

    assert not any(GAME_TIME_DEFLATED in reason for reason in transition.invalid_reasons)
    summary = environment.summarize(transition.termination)
    assert not any(GAME_TIME_DEFLATED in reason for reason in summary.termination_detail)


def test_the_wall_ceiling_outranks_an_event_the_settled_state_shows() -> None:
    """A truncated advance is refused even when the run ended inside it.

    The bridge gives the ceiling precedence over every event reason, so a stall
    is never excused by the world happening to do something while the host was
    slow. The host still has to see the run end - the episode must terminate -
    but it terminates by the invariant's name, not as a clean game over.
    """
    from dataclasses import replace as replace_reading

    environment, port = _environment()
    environment.reset()
    reading = port.read_state()
    assert reading is not None
    ended = replace_reading(
        reading, lifecycle="terminal", terminal=True, round_active=False,
        health=0.0, game_speed=0.0,
    )

    def truncated_on_an_ended_run(**_kwargs: object) -> FakeCommandResult:
        return FakeCommandResult(
            "confirmed", "wall_ceiling", frames=3, game_ms=50.0, state=ended,
        )

    port.advance_until_event = truncated_on_an_ended_run  # type: ignore[method-assign]

    transition = environment.step(WAIT)

    assert DecisionEvent.RUN_ENDED in transition.events, "the host must still see the end"
    assert transition.termination is TerminationOutcome.OBSERVATION_INVALID
    assert transition.truncated and not transition.terminated
    assert not transition.admissible
    assert ADVANCE_TRUNCATED_BY_WALL in transition.invalid_reasons
    # The ceiling makes no claim about events, so it is not a disagreement.
    assert BRIDGE_EVENT_DIVERGENCE not in transition.invalid_reasons


def test_a_truncated_death_boundary_settle_fails_the_episode_too() -> None:
    """The one-frame recovery is held to the same invariant as any advance.

    A 15 s stall rendering a single frame is exactly the distortion the ceiling
    exists to hear, so the settled observation it returns is refused rather than
    recovered.
    """
    from dataclasses import replace as replace_reading

    environment, port = _environment()
    environment.reset()
    original = port.advance_until_event
    calls = {"count": 0}

    def stalling_recovery(**kwargs: object) -> FakeCommandResult:
        calls["count"] += 1
        result = original(**kwargs)  # type: ignore[arg-type]
        assert result.state is not None
        if calls["count"] == 1:
            # The contradictory instant that asks for the recovery advance.
            return replace_reading(
                result,
                state=replace_reading(
                    result.state, health=-2.0, lifecycle="active", terminal=False
                ),
            )
        return replace_reading(result, reason="wall_ceiling")

    port.advance_until_event = stalling_recovery  # type: ignore[method-assign]

    transition = environment.step(WAIT)

    assert calls["count"] == 2, "the boundary still has to be settled by advancing"
    assert transition.termination is TerminationOutcome.OBSERVATION_INVALID
    assert not transition.admissible
    assert transition.next_state is not None
    assert ADVANCE_TRUNCATED_BY_WALL in transition.next_state.invalid_reasons
    assert environment._tally.recovered_transients == 0
    summary = environment.summarize(transition.termination)
    assert not summary.valid
    assert any(ADVANCE_TRUNCATED_BY_WALL in detail for detail in summary.termination_detail)


def test_an_episode_records_one_row_per_wave_it_entered() -> None:
    """Per-wave rows are the instrument a final wave cannot be (issue #24).

    This run crosses two wave boundaries, so it holds three rows: two whole
    waves and the fragment it died in, which is labelled rather than dropped.
    """
    environment, _ = _environment(damage_per_second=0.3, seconds_per_wave=6.0)
    environment.reset()
    boundaries = {}

    for _ in range(400):
        transition = environment.step(WAIT)
        after = transition.next_state
        if after is not None and after.wave != transition.state.wave:
            boundaries[after.wave] = after
        if transition.terminated:
            break
    else:
        pytest.fail("the tower never died")

    summary = environment.summarize(transition.termination)
    assert [wave.wave for wave in summary.waves] == [1, 2, 3]
    # Only the wave the episode died in is a fragment.
    assert [wave.completed for wave in summary.waves] == [True, True, False]
    # Measured round time and decisions are partitioned, never double counted.
    assert sum(wave.game_ms for wave in summary.waves) == pytest.approx(summary.round_ms)
    assert sum(wave.decisions for wave in summary.waves) == summary.decisions
    assert all(wave.game_ms > 0.0 and wave.decisions > 0 for wave in summary.waves)
    # The boundary state is the state the wave began at, as the observation
    # carries it - cash log-scaled, health as a fraction.
    for wave in summary.waves[1:]:
        entered = boundaries[wave.wave]
        assert wave.health_fraction == entered.health_fraction
        assert wave.cash_log == entered.cash_log
    assert summary.waves[0].health_fraction == 1.0


def test_an_advance_crossing_a_wave_boundary_is_charged_to_the_wave_it_started_in() -> None:
    """The bridge reports one round-clock delta per advance and cannot split it.

    So the whole delta goes to the wave that was current when the advance began.
    Stated, deterministic, and worth at most one advance either side of a boundary.
    """
    environment, port = _environment(damage_per_second=0.3, seconds_per_wave=6.0)
    environment.reset()
    crossing_round_ms = 0.0
    before_crossing = 0.0
    original = port.advance_until_event
    seen: list[float] = []

    def recording_advance(**kwargs: object) -> FakeCommandResult:
        result = original(**kwargs)  # type: ignore[arg-type]
        seen.append(result.round_ms)
        return result

    port.advance_until_event = recording_advance  # type: ignore[method-assign]

    while True:
        transition = environment.step(WAIT)
        if transition.next_state is not None and transition.next_state.wave == 2:
            crossing_round_ms = seen[-1]
            break
        before_crossing += seen[-1]
        if transition.terminated:
            pytest.fail("the run died before it changed wave")

    summary = environment.summarize(TerminationOutcome.OPERATOR_STOP)
    assert crossing_round_ms > 0.0
    first = next(wave for wave in summary.waves if wave.wave == 1)
    assert first.game_ms == pytest.approx(before_crossing + crossing_round_ms)
    assert next(wave for wave in summary.waves if wave.wave == 2).game_ms == 0.0


def test_a_decision_stream_observer_hears_one_view_per_step() -> None:
    """The seam a spectator watches: one view per decision, and no more."""
    environment, _ = _environment()
    seen: list[DecisionView] = []
    environment.on_decision = seen.append
    environment.reset()

    for _ in range(3):
        environment.step(WAIT)

    assert len(seen) == 3
    assert [view.decision for view in seen] == [1, 2, 3]
    assert {view.episode for view in seen} == {1}
    assert [view.action for view in seen] == ["wait", "wait", "wait"]
    assert not any(view.done for view in seen)


def test_a_view_says_what_its_own_transition_said() -> None:
    """The panel and the record must not be able to disagree."""
    environment, _ = _environment()
    seen: list[DecisionView] = []
    environment.on_decision = seen.append
    state = environment.reset()
    target = next(row for row in state.rows if row.available)

    transition = environment.step(target.action)
    view = seen[-1]

    assert transition.next_state is not None
    assert view.wave == transition.next_state.wave
    assert view.health_fraction == transition.next_state.health_fraction
    assert view.cash == pytest.approx(math.expm1(transition.next_state.cash_log))
    assert view.reward == transition.reward
    assert view.action == str(target.action)
    assert view.termination is transition.termination


def test_the_last_view_of_an_episode_is_the_one_marked_done() -> None:
    """A watcher learns the tower died from the stream, not from a summary."""
    environment, _ = _environment(damage_per_second=2.0, max_health=1.0)
    seen: list[DecisionView] = []
    environment.on_decision = seen.append
    environment.reset()

    while not seen or not seen[-1].done:
        environment.step(WAIT)

    assert [view.done for view in seen] == [False] * (len(seen) - 1) + [True]
    assert seen[-1].termination is TerminationOutcome.GAME_OVER


def test_episodes_are_numbered_across_resets_and_decisions_are_not() -> None:
    environment, _ = _environment()
    seen: list[DecisionView] = []
    environment.on_decision = seen.append

    environment.reset()
    environment.step(WAIT)
    environment.reset()
    environment.step(WAIT)

    assert [(view.episode, view.decision) for view in seen] == [(1, 1), (2, 1)]


def test_an_environment_with_no_observer_builds_no_views() -> None:
    """The default costs an attribute test per decision and nothing else."""
    environment, _ = _environment()
    environment.reset()

    transition = environment.step(WAIT)

    assert environment.on_decision is None
    assert transition.admissible
