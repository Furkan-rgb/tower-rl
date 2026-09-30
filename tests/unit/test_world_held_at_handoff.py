"""The agent decides only while the world stands still (`#95`, ADR 0015).

Every path that hands control back to the agent - `reset` of a fresh, a
post-game-over and a retired run, a `WAIT`, a purchase that advances nothing, a
purchase followed by a choice-point span, the death-boundary recovery - is
checked here to hand over a held world, and a port that hands one over running
is refused at the handoff by name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.episode import TerminationOutcome
from tower_rl.environment.run_actions import WAIT, upgrade_action
from tower_rl.environment.run_environment import (
    WORLD_NOT_HELD,
    CadenceConfig,
    DecisionCadence,
    InstrumentedRunEnvironment,
)
from tower_rl.environment.run_port import RunPortError
from tower_rl.environment.run_state import RunState, RunStateBuilder


def _environment(
    decision_cadence: DecisionCadence = DecisionCadence.CHOICE_POINTS, **port_kwargs: object
) -> tuple[InstrumentedRunEnvironment, FakeRunPort]:
    port = FakeRunPort(**port_kwargs)  # type: ignore[arg-type]
    environment = InstrumentedRunEnvironment(
        port=port,
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
        decision_cadence=decision_cadence,
    )
    return environment, port


def _handed_over_held(port: FakeRunPort, state: RunState | None) -> None:
    assert state is not None
    assert state.lifecycle == "active" and not state.terminal
    assert port.world_held
    assert WORLD_NOT_HELD not in state.invalid_reasons


def _fresh_reset(environment: InstrumentedRunEnvironment, port: FakeRunPort) -> RunState | None:
    return environment.reset()


def _reset_after_game_over(
    environment: InstrumentedRunEnvironment, port: FakeRunPort
) -> RunState | None:
    port.damage_per_second = 2.0
    environment.reset()
    while environment.step(WAIT).termination is None:
        pass
    port.damage_per_second = 0.25
    return environment.reset()


def _reset_after_retirement(
    environment: InstrumentedRunEnvironment, port: FakeRunPort
) -> RunState | None:
    port.continues_live_runs = True
    environment.reset()
    environment.step(WAIT)
    state = environment.reset()
    assert environment.summarize(TerminationOutcome.OPERATOR_STOP).retired_run_wave > 0
    return state


def _wait(environment: InstrumentedRunEnvironment, port: FakeRunPort) -> RunState | None:
    environment.reset()
    return environment.step(WAIT).next_state


def _purchase_that_advances_nothing(
    environment: InstrumentedRunEnvironment, port: FakeRunPort
) -> RunState | None:
    environment.reset()
    advances = port.advances
    transition = environment.step(upgrade_action("attack", 0))
    assert port.advances == advances, "the default fake re-decides straight after a purchase"
    return transition.next_state


def _purchase_then_a_choice_point_span(
    environment: InstrumentedRunEnvironment, port: FakeRunPort
) -> RunState | None:
    # One slot at 5, the run opens holding exactly that, cash trickles back: the
    # purchase leaves nothing to choose, so the environment advances to the next
    # choice point before handing over.
    port.offered = {"attack": 1, "defense": 0, "utility": 0}
    port.start_cash = 5.0
    port.cash_per_second = 0.5
    port.damage_per_second = 0.0
    environment.reset()
    advances = port.advances
    transition = environment.step(upgrade_action("attack", 0))
    assert port.advances > advances
    return transition.next_state


def _death_boundary_recovery(
    environment: InstrumentedRunEnvironment, port: FakeRunPort
) -> RunState | None:
    environment.reset()
    original = port.read_state
    port.read_state = lambda: (  # type: ignore[method-assign]
        None if (reading := original()) is None
        else replace(reading, health=-2.0, lifecycle="active", terminal=False)
    )
    state = environment._read_state()
    port.read_state = original  # type: ignore[method-assign]
    assert environment._tally.recovered_transients == 1
    return state


PATHS: dict[str, Callable[[InstrumentedRunEnvironment, FakeRunPort], RunState | None]] = {
    "fresh reset": _fresh_reset,
    "reset after game over": _reset_after_game_over,
    "reset after a retired run": _reset_after_retirement,
    "wait": _wait,
    "purchase advancing nothing": _purchase_that_advances_nothing,
    "purchase then choice-point span": _purchase_then_a_choice_point_span,
    "death-boundary recovery": _death_boundary_recovery,
}


@pytest.mark.parametrize("path", PATHS, ids=list(PATHS))
def test_every_handoff_path_hands_over_a_held_world(path: str) -> None:
    environment, port = _environment()

    state = PATHS[path](environment, port)

    _handed_over_held(port, state)


def test_a_round_handed_over_running_is_refused_at_reset() -> None:
    environment, _ = _environment(DecisionCadence.EVERY_SLICE, holds_at_round_start=False)

    with pytest.raises(RunPortError, match="WORLD_NOT_HELD"):
        environment.reset()


def test_an_advance_that_leaves_the_world_running_is_refused_by_name() -> None:
    environment, _ = _environment(holds_after_advance=False)
    environment.reset()

    transition = environment.step(WAIT)

    assert transition.termination is TerminationOutcome.OBSERVATION_INVALID
    assert transition.truncated and not transition.admissible
    summary = environment.summarize(transition.termination)
    assert any(WORLD_NOT_HELD in reason for reason in summary.termination_detail)


def test_a_purchase_on_a_running_world_is_refused_too() -> None:
    """A purchase does not move the hold, so it hands over whatever it inherited."""
    environment, port = _environment()
    environment.reset()
    port.world_held = False

    transition = environment.step(upgrade_action("attack", 0))

    assert transition.termination is TerminationOutcome.OBSERVATION_INVALID
    assert transition.next_state is not None
    assert WORLD_NOT_HELD in transition.next_state.invalid_reasons


def test_a_run_that_ended_is_not_asked_to_be_held() -> None:
    environment, _ = _environment(damage_per_second=2.0)
    environment.reset()

    while (transition := environment.step(WAIT)).termination is None:
        pass

    assert transition.termination is TerminationOutcome.GAME_OVER
