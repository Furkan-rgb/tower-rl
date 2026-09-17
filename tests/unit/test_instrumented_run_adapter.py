from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tower_rl.infrastructure.instrumented_bridge import (
    BridgeCommandResult,
    BridgeObservation,
    BridgeRunUnavailable,
    CommandOutcome,
    UpgradeInventoryEntry,
)
from tower_rl.infrastructure.instrumented_run_adapter import (
    GAME_SPEED,
    InstrumentedRunAdapter,
)
from tower_rl.ports.run_port import RunPortError


def _observation(sequence: int = 1, *, terminal: bool = False, speed: float = 64.0):
    return BridgeObservation(
        sequence=sequence,
        lifecycle="terminal" if terminal else "active",
        wave=3,
        cash=100.0,
        health=0.0 if terminal else 4.0,
        max_health=5.0,
        terminal=terminal,
        round_active=not terminal,
        game_speed=speed,
        play_time=12.0,
        upgrades=(
            UpgradeInventoryEntry("attack", 0, 10.0, 1, 50, True, False, False),
        ),
    )


@dataclass
class FakeClient:
    states: list[object] = field(default_factory=list)
    sent: list[dict[str, object]] = field(default_factory=list)
    outcome: str = "confirmed"
    default_speed: float = 64.0

    def read_state(self) -> object:
        if self.states:
            return self.states.pop(0)
        return _observation(speed=self.default_speed)

    def send_command(self, message: dict[str, object]) -> BridgeCommandResult:
        self.sent.append(message)
        return BridgeCommandResult(
            request_id=str(message["request_id"]),
            outcome=CommandOutcome(self.outcome),
            reason="ok",
            observation_sequence=1,
        )


def _adapter(client: FakeClient, **kwargs: object) -> InstrumentedRunAdapter:
    return InstrumentedRunAdapter(client=client, **kwargs)  # type: ignore[arg-type]


def test_an_unavailable_run_reads_as_no_state_not_as_invented_values() -> None:
    client = FakeClient(states=[BridgeRunUnavailable(4, "no_initialized_run")])

    assert _adapter(client).read_state() is None


def _actions(client: FakeClient) -> list[object]:
    return [message.get("action") for message in client.sent if message["kind"] == "lifecycle"]


def test_an_episode_starts_through_the_games_own_control_and_reads_no_pixel() -> None:
    """M1B-E022: the boundary presses `start_round`; nothing classifies a screen."""
    client = FakeClient(
        states=[
            BridgeRunUnavailable(1, "no_initialized_run"),
            BridgeRunUnavailable(1, "no_initialized_run"),
            _observation(),
        ]
    )

    _adapter(client).begin_episode()

    assert _actions(client) == ["start_round"]


def test_a_terminal_run_is_sent_home_before_the_round_is_started() -> None:
    """`BattlePanel` only exists at home, so a finished run is closed first."""
    client = FakeClient(
        states=[_observation(terminal=True), _observation(terminal=True), _observation()]
    )

    _adapter(client).begin_episode()

    assert _actions(client) == ["go_home", "start_round"]


def test_a_control_the_game_does_not_honour_fails_the_boundary() -> None:
    client = FakeClient(
        states=[
            BridgeRunUnavailable(1, "no_initialized_run"),
            BridgeRunUnavailable(1, "no_initialized_run"),
        ],
        outcome="ambiguous",
    )

    with pytest.raises(RunPortError, match="did not honour start_round"):
        _adapter(client, episode_start_timeout=0.2).begin_episode()


def test_one_advance_carries_the_whole_cadence_to_the_bridge() -> None:
    """M1B-E016: the loop belongs in the bridge, one round trip per decision.

    Asking for a slice at a time cost a round trip per slice and about eight of
    them per decision. The three fields pinned here are the whole contract: how
    long the bridge may run, what a frame is worth, and the health move it must
    interrupt for.
    """
    client = FakeClient(states=[_observation(speed=1.0)])
    adapter = _adapter(client)
    adapter.read_state()

    adapter.advance_until_event(
        expected_sequence=1,
        budget_game_ms=2000,
        frame_game_ms=1000 / 60,
        health_change_fraction=0.05,
    )

    message = client.sent[-1]
    assert message["kind"] == "advance"
    assert message["budget_game_ms"] == 2000
    assert message["frame_game_ms"] == 1000 / 60
    assert message["health_change_fraction"] == 0.05
    assert message["expected_observation_sequence"] == 1


def test_release_leaves_the_game_running_for_the_next_session() -> None:
    client = FakeClient(states=[_observation(speed=64.0)])
    adapter = _adapter(client)

    adapter.release()

    assert client.sent[-1]["kind"] == "lifecycle"
    assert client.sent[-1]["action"] == "unpause"


def test_the_game_speed_multiplier_is_pinned_at_one() -> None:
    """The multiplier is not a speed-up mechanism, so the adapter holds it at 1x."""
    assert GAME_SPEED == 1.0
    # The clone comes back from the boundary running at 8x; the adapter must
    # put it back to 1x before the episode starts.
    client = FakeClient(
        states=[BridgeRunUnavailable(1, "no_initialized_run")], default_speed=8.0
    )

    _adapter(client).begin_episode()

    speed_command = next(message for message in client.sent if message["kind"] == "set_speed")
    assert speed_command["value"] == 1.0


def test_purchases_bind_the_state_they_were_decided_from() -> None:
    client = FakeClient()
    adapter = _adapter(client)

    adapter.buy_upgrade("defense", 1, expected_sequence=42)

    message = client.sent[-1]
    assert message["kind"] == "buy_upgrade"
    assert message["family"] == "defense" and message["index"] == 1
    assert message["expected_observation_sequence"] == 42


def test_a_refused_speed_pin_is_an_explicit_failure() -> None:
    client = FakeClient(
        states=[BridgeRunUnavailable(1, "no_initialized_run"), _observation(speed=1.5)],
        outcome="rejected",
        default_speed=1.5,
    )

    with pytest.raises(RunPortError, match="refused the pinned 1x speed"):
        _adapter(client).begin_episode()


def test_a_frozen_leftover_run_is_resumed_before_an_episode_begins() -> None:
    """An episode that ended host-side leaves the bridge holding the world still.

    The cached reading of a paused run still says "active", so without this the
    next episode would start on a view of a world that had stopped moving, and
    on a sequence frozen at whatever the previous episode last saw.
    """
    client = FakeClient(states=[_observation(sequence=9)])

    _adapter(client).begin_episode()

    resume = client.sent[0]
    assert resume["kind"] == "lifecycle" and resume["action"] == "unpause"
    assert resume["expected_observation_sequence"] == 9


def test_a_run_that_will_not_resume_refuses_to_start_an_episode() -> None:
    client = FakeClient(states=[_observation(sequence=9)], outcome="ambiguous")

    with pytest.raises(RunPortError, match="could not be resumed"):
        _adapter(client).begin_episode()


def test_a_finished_run_is_not_asked_to_resume() -> None:
    """The bridge never pauses a run that ended, and its `unpause` waits for an
    active run, so asking would only stall the boundary."""
    client = FakeClient(
        states=[_observation(terminal=True), _observation(terminal=True), _observation()]
    )

    _adapter(client).begin_episode()

    assert "unpause" not in _actions(client)
