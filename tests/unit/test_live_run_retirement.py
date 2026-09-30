"""An episode that ends without the game's own death never hands its run on (`#95`).

The port's `begin_episode` continues a live run rather than starting one, as the
real adapter does, so these tests run on a fake that does the same
(`continues_live_runs`). Before the fix, the episode after an invalid end opened
mid-game on the run that had just failed.
"""

from __future__ import annotations

import threading

import pytest
from fakes.fake_run_port import DEVICE_REAL_ROWS, FakeCommandResult, FakeRunPort

from tower_rl.environment import run_environment
from tower_rl.environment.episode import ActionOutcome, TerminationOutcome
from tower_rl.environment.run_actions import WAIT, action_index, upgrade_action
from tower_rl.environment.run_environment import (
    RETIREMENT_WALL_CEILING_SECONDS,
    STALLED_REASON_PREFIX,
    CadenceConfig,
    InstrumentedRunEnvironment,
    RetirementAbandoned,
    RetirementFailed,
    UpgradeAvailability,
)
from tower_rl.environment.run_port import RunPortError
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.actor import Actor, ActorConfig
from tower_rl.learning.policies import CheapestFirstPolicy
from tower_rl.learning.r2d2_replay import R2D2Replay

WORKSHOP_LEVEL = 5


def _environment(**port_kwargs: object) -> tuple[InstrumentedRunEnvironment, FakeRunPort]:
    """An environment playing every real row at a Workshop level, as M3 runs are."""
    settings: dict[str, object] = {
        "continues_live_runs": True,
        "real_rows": dict(DEVICE_REAL_ROWS),
        "offered": {"attack": 4, "defense": 2, "utility": 0},
        "damage_per_second": 0.1,
    }
    settings.update(port_kwargs)
    port = FakeRunPort(**settings)  # type: ignore[arg-type]
    environment = InstrumentedRunEnvironment(
        port=port,
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
        upgrade_availability=UpgradeAvailability.ALL,
        workshop_level=WORKSHOP_LEVEL,
    )
    return environment, port


def _cut_mid_game(environment: InstrumentedRunEnvironment, port: FakeRunPort) -> int:
    """Play into the run, then end the episode on a mask-legal rejection.

    Returns the wave the run was left live at.
    """
    state = environment.reset()
    while state.wave < 3:
        transition = environment.step(WAIT)
        assert transition.termination is None and transition.next_state is not None
        state = transition.next_state
    action = upgrade_action("attack", 0)
    assert state.action_mask[action_index(action)]
    port.slots[("attack", 0)].cost = port.cash + 1_000.0

    transition = environment.step(action)

    assert transition.termination is TerminationOutcome.MASK_LEGAL_REJECTED
    assert port.active, "the cut leaves the game's run live"
    return state.wave


def test_an_invalid_end_mid_game_is_followed_by_a_fresh_run_not_the_cut_one() -> None:
    environment, port = _environment()
    cut_wave = _cut_mid_game(environment, port)

    state = environment.reset()

    assert state.wave == 1
    summary = environment.summarize(TerminationOutcome.OPERATOR_STOP)
    assert summary.starting_wave == 1
    assert summary.retired_run_wave == cut_wave
    assert summary.retirement_wall_seconds >= 0.0
    assert port.episodes == 2 and port.wave == 1


def test_the_fresh_run_has_its_settings_written_and_checked_again() -> None:
    environment, port = _environment()
    _cut_mid_game(environment, port)
    port.workshop_commands.clear()

    environment.reset()

    # The Workshop is written before the fresh round and read back after it.
    assert [command[0] for command in port.workshop_commands] == ["set", "begin", "read"]
    held = {row.name: row.after for row in port.workshop_levels()}
    assert held["Damage"] == WORKSHOP_LEVEL
    # A fresh round locks every row the image does not offer again (a continued
    # run would not), so every real row open says the unlock was issued anew.
    assert all(
        port.slots[(family, index)].unlocked
        for family, rows in DEVICE_REAL_ROWS.items()
        for index in range(rows)
    )
    summary = environment.summarize(TerminationOutcome.OPERATOR_STOP)
    assert not summary.upgrade_setup_drifted


def test_the_invalid_end_keeps_its_own_reason_and_is_a_truncation() -> None:
    """Retiring the run afterwards changes nothing about how the cut was classified."""
    environment, port = _environment()
    environment.reset()
    port.slots[("attack", 0)].cost = port.cash + 1_000.0

    transition = environment.step(upgrade_action("attack", 0))
    summary = environment.summarize(transition.termination or TerminationOutcome.OPERATOR_STOP)
    environment.reset()

    assert transition.outcome is ActionOutcome.UNAVAILABLE
    assert transition.termination is TerminationOutcome.MASK_LEGAL_REJECTED
    assert transition.truncated and not transition.terminated
    assert summary.termination is TerminationOutcome.MASK_LEGAL_REJECTED


def test_nothing_of_the_retired_run_reaches_replay_or_the_next_episode_record() -> None:
    environment, port = _environment()
    _cut_mid_game(environment, port)
    advances_before = port.advances
    replay = R2D2Replay(capacity=256, seed=0)
    actor = Actor(
        environment=environment,
        policy=CheapestFirstPolicy(),
        config=ActorConfig(),
        replay=replay,
    )

    result = actor.run_episode()

    summary = result.summary
    assert summary.termination is TerminationOutcome.GAME_OVER
    assert summary.starting_wave == 1 and summary.waves[0].wave == 1
    retirement_advances = port.advances - advances_before - summary.advances
    assert retirement_advances > 0, "the cut run was played out by the reset"
    assert len(replay) == result.sequences_accepted > 0
    assert {item.metadata.episode_id for item in replay._items} == {summary.episode_id}


def test_an_episode_ended_by_the_game_retires_nothing() -> None:
    environment, port = _environment(damage_per_second=2.0)
    environment.reset()
    while True:
        transition = environment.step(WAIT)
        if transition.termination is not None:
            break
    assert transition.termination is TerminationOutcome.GAME_OVER
    advances_before = port.advances

    environment.reset()

    summary = environment.summarize(TerminationOutcome.OPERATOR_STOP)
    assert summary.retired_run_wave == 0 and summary.retirement_wall_seconds == 0.0
    assert port.advances - advances_before == summary.advances, "nothing was played out"


def test_a_retirement_that_stops_moving_the_game_clock_fails_the_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung pipeline during retirement is a failed episode start, not a spin."""
    environment, port = _environment()
    _cut_mid_game(environment, port)
    environment.cadence = CadenceConfig(max_quiet_game_ms=1000, stall_window_wall_seconds=10.0)
    calls = {"count": 0}

    def frozen(**_: object) -> FakeCommandResult:
        calls["count"] += 1
        return FakeCommandResult(
            "confirmed", "budget_exhausted", state=port._observe()
        )

    monkeypatch.setattr(port, "advance_until_event", frozen)
    clock = [1_000.0]

    def ticking() -> float:
        clock[0] += 1.0
        return clock[0]

    monkeypatch.setattr(run_environment.time, "monotonic", ticking)

    with pytest.raises(RunPortError, match=STALLED_REASON_PREFIX):
        environment.reset()
    assert calls["count"] > 1


def test_a_stale_first_retirement_advance_reads_again_and_carries_on() -> None:
    """A world running free since its round began can outrun the sequence just read."""
    environment, port = _environment()
    cut_wave = _cut_mid_game(environment, port)
    real_advance = port.advance_until_event
    calls = {"count": 0}

    def outrun_once(**kwargs: object) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            port.sequence += 1  # the world streamed a newer observation
        return real_advance(**kwargs)  # type: ignore[arg-type]

    port.advance_until_event = outrun_once  # type: ignore[method-assign]

    state = environment.reset()

    assert state.wave == 1
    assert environment.summarize(TerminationOutcome.OPERATOR_STOP).retired_run_wave == cut_wave


def _a_clock_that_ticks_a_second_a_read(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1_000.0]

    def ticking() -> float:
        clock[0] += 1.0
        return clock[0]

    monkeypatch.setattr(run_environment.time, "monotonic", ticking)


def test_frames_rendering_on_a_frozen_round_clock_is_not_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `GAME_TIME_DEFLATED` signature: the bridge credits game time, the game's clock stands.

    Progress is the round clock, so this retirement fails at the stall window
    rather than going on for as long as the bridge keeps crediting frames.
    """
    environment, port = _environment()
    cut_wave = _cut_mid_game(environment, port)
    environment.cadence = CadenceConfig(max_quiet_game_ms=1000, stall_window_wall_seconds=10.0)
    calls = {"count": 0}

    def frames_but_no_round_clock(**_: object) -> FakeCommandResult:
        calls["count"] += 1
        return FakeCommandResult(
            "confirmed",
            "budget_exhausted",
            frames=100,
            game_ms=10_000.0,
            round_ms=0.0,
            state=port._observe(),
        )

    monkeypatch.setattr(port, "advance_until_event", frames_but_no_round_clock)
    _a_clock_that_ticks_a_second_a_read(monkeypatch)

    with pytest.raises(RetirementFailed, match=STALLED_REASON_PREFIX) as failed:
        environment.reset()

    assert 1 < calls["count"] < 20, "it failed at the bound, not after the bridge gave up"
    assert failed.value.retired_run_wave == cut_wave
    assert failed.value.retirement_wall_seconds > 10.0
    assert f"retired_run_wave {cut_wave}" in str(failed.value)


def test_a_retirement_that_keeps_progressing_still_ends_at_the_wall_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that will not die is bounded by wall time whatever its clock does."""
    environment, port = _environment()
    cut_wave = _cut_mid_game(environment, port)
    calls = {"count": 0}

    def progressing_forever(**_: object) -> FakeCommandResult:
        calls["count"] += 1
        return FakeCommandResult(
            "confirmed",
            "budget_exhausted",
            frames=100,
            game_ms=10_000.0,
            round_ms=10_000.0,
            state=port._observe(),
        )

    monkeypatch.setattr(port, "advance_until_event", progressing_forever)
    _a_clock_that_ticks_a_second_a_read(monkeypatch)

    with pytest.raises(RetirementFailed, match="took over") as failed:
        environment.reset()

    assert failed.value.retired_run_wave == cut_wave
    assert failed.value.retirement_wall_seconds > RETIREMENT_WALL_CEILING_SECONDS
    assert calls["count"] > 10, "the stall window, which it was not, did not end it"


def test_a_stop_set_during_a_retirement_abandons_it_between_advances() -> None:
    environment, port = _environment()
    _cut_mid_game(environment, port)
    stop = threading.Event()
    environment.stop_requested = stop.is_set
    real_advance = port.advance_until_event

    def stop_during_the_first_advance(**kwargs: object) -> object:
        stop.set()
        return real_advance(**kwargs)  # type: ignore[arg-type]

    port.advance_until_event = stop_during_the_first_advance  # type: ignore[method-assign]
    advances_before = port.advances

    with pytest.raises(RetirementAbandoned):
        environment.reset()

    assert port.advances == advances_before + 1
    assert port.active, "nothing ended the run, so the next reset retires it"
