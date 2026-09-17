from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes.fake_run_port import FakeRunPort  # noqa: E402

from tower_rl.application.run_environment import (  # noqa: E402
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
        cadence=CadenceConfig(slice_game_ms=250, max_quiet_game_ms=1000),
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
    environment, _ = _environment()
    environment.reset()

    transition = environment.step(WAIT)

    assert transition.outcome is ActionOutcome.WAITED
    assert transition.events != ()
    assert transition.requested_game_ms > 0
    assert transition.next_state is not None


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
        slice_game_ms=250, max_quiet_game_ms=500, max_episode_wall_seconds=0.0
    )
    environment.reset()

    transition = environment.step(WAIT)

    assert transition.truncated
    assert transition.termination is TerminationOutcome.MAX_EPISODE_DURATION
    assert not transition.admissible
    assert port.active, "truncation is an environment decision, not a game over"


def test_the_death_boundary_transient_is_recovered_not_discarded() -> None:
    """M1B-E008: health goes negative a moment before game-over flips."""
    from dataclasses import replace

    environment, port = _environment()
    environment.reset()
    original = port.read_state

    calls = {"count": 0}

    def flaky_read():
        reading = original()
        calls["count"] += 1
        if calls["count"] == 1 and reading is not None:
            # Active, but health already negative: the inconsistent instant.
            return replace(reading, health=-2.0, lifecycle="active", terminal=False)
        return reading

    port.read_state = flaky_read  # type: ignore[method-assign]
    state = environment._read_state()

    assert state is not None and state.valid, state.invalid_reasons
    assert environment._tally.recovered_transients == 1


def test_a_persistently_contradictory_state_is_still_invalid() -> None:
    from dataclasses import replace

    environment, port = _environment()
    environment.reset()
    original = port.read_state

    def always_contradictory():
        reading = original()
        return None if reading is None else replace(
            reading, health=-2.0, lifecycle="active", terminal=False
        )

    port.read_state = always_contradictory  # type: ignore[method-assign]
    state = environment._read_state()

    assert state is not None and not state.valid
    assert "negative health in an active run" in state.invalid_reasons
