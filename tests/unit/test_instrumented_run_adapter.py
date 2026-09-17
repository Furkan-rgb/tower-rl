from __future__ import annotations

import io
from dataclasses import dataclass, field

import pytest
from PIL import Image

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
from tower_rl.infrastructure.visual_profile import Screen
from tower_rl.ports.android import CapturedFrame, InputReceipt, ScreenPoint
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


def _png(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1080, 1920), colour).save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class FakeDevice:
    screen: Screen = Screen.HOME
    taps: list[ScreenPoint] = field(default_factory=list)

    def screenshot(self) -> CapturedFrame:
        return CapturedFrame(
            frame_id="frame", captured_at_monotonic=0.0, width=1080, height=1920,
            png_bytes=_png((0, 0, 0)),
        )

    def tap(self, point: ScreenPoint) -> InputReceipt:
        self.taps.append(point)
        return InputReceipt(event_id="tap", accepted_at_monotonic=0.0)


def _adapter(client: FakeClient, device: FakeDevice, **kwargs: object) -> InstrumentedRunAdapter:
    return InstrumentedRunAdapter(
        client=client, device=device, settle_seconds=0.0, **kwargs  # type: ignore[arg-type]
    )


def test_an_unavailable_run_reads_as_no_state_not_as_invented_values() -> None:
    client = FakeClient(states=[BridgeRunUnavailable(4, "no_initialized_run")])

    assert _adapter(client, FakeDevice()).read_state() is None


def test_a_tap_is_refused_unless_the_screen_is_positively_classified(monkeypatch) -> None:
    client = FakeClient(
        states=[
            BridgeRunUnavailable(1, "no_initialized_run"),
            BridgeRunUnavailable(1, "no_initialized_run"),
        ]
    )
    device = FakeDevice()
    # The real classifier sees a blank frame here, which is deliberately UNKNOWN.
    adapter = _adapter(client, device, episode_start_timeout=0.2)

    with pytest.raises(RunPortError, match="refusing to tap"):
        adapter.begin_episode()
    assert device.taps == [], "no tap may be sent from an unclassified screen"


def test_a_classified_home_screen_permits_exactly_the_battle_tap(monkeypatch) -> None:
    import tower_rl.infrastructure.instrumented_run_adapter as module

    monkeypatch.setattr(module, "classify", lambda _image: Screen.HOME)
    client = FakeClient(
        states=[
            BridgeRunUnavailable(1, "no_initialized_run"),
            BridgeRunUnavailable(1, "no_initialized_run"),
            _observation(),
        ]
    )
    device = FakeDevice()

    _adapter(client, device).begin_episode()

    assert len(device.taps) == 1
    assert device.taps[0] == module.BATTLE_BUTTON


def test_a_terminal_run_taps_retry_from_the_result_screen(monkeypatch) -> None:
    import tower_rl.infrastructure.instrumented_run_adapter as module

    monkeypatch.setattr(module, "classify", lambda _image: Screen.RESULT)
    client = FakeClient(
        states=[_observation(terminal=True), _observation(terminal=True), _observation()]
    )
    device = FakeDevice()

    _adapter(client, device).begin_episode()

    assert device.taps == [module.RETRY_BUTTON]


def test_one_advance_carries_the_whole_cadence_to_the_bridge() -> None:
    """M1B-E016: the loop belongs in the bridge, one round trip per decision.

    Asking for a slice at a time cost a round trip per slice and about eight of
    them per decision. The three fields pinned here are the whole contract: how
    long the bridge may run, what a frame is worth, and the health move it must
    interrupt for.
    """
    client = FakeClient(states=[_observation(speed=1.0)])
    adapter = _adapter(client, FakeDevice())
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
    adapter = _adapter(client, FakeDevice())

    adapter.release()

    assert client.sent[-1]["kind"] == "lifecycle"
    assert client.sent[-1]["action"] == "unpause"


def test_the_game_speed_multiplier_is_pinned_at_one(monkeypatch) -> None:
    """The multiplier is not a speed-up mechanism, so the adapter holds it at 1x."""
    import tower_rl.infrastructure.instrumented_run_adapter as module

    monkeypatch.setattr(module, "classify", lambda _image: Screen.HOME)
    assert GAME_SPEED == 1.0
    # The clone comes back from the boundary running at 8x; the adapter must
    # put it back to 1x before the episode starts.
    client = FakeClient(
        states=[BridgeRunUnavailable(1, "no_initialized_run")], default_speed=8.0
    )

    _adapter(client, FakeDevice()).begin_episode()

    speed_command = next(message for message in client.sent if message["kind"] == "set_speed")
    assert speed_command["value"] == 1.0


def test_purchases_bind_the_state_they_were_decided_from() -> None:
    client = FakeClient()
    adapter = _adapter(client, FakeDevice())

    adapter.buy_upgrade("defense", 1, expected_sequence=42)

    message = client.sent[-1]
    assert message["kind"] == "buy_upgrade"
    assert message["family"] == "defense" and message["index"] == 1
    assert message["expected_observation_sequence"] == 42


def test_a_refused_speed_pin_is_an_explicit_failure(monkeypatch) -> None:
    import tower_rl.infrastructure.instrumented_run_adapter as module

    monkeypatch.setattr(module, "classify", lambda _image: Screen.HOME)
    client = FakeClient(
        states=[BridgeRunUnavailable(1, "no_initialized_run"), _observation(speed=1.5)],
        outcome="rejected",
        default_speed=1.5,
    )

    with pytest.raises(RunPortError, match="refused the pinned 1x speed"):
        _adapter(client, FakeDevice()).begin_episode()


def test_a_frozen_leftover_run_is_resumed_before_an_episode_begins() -> None:
    """An episode that ended host-side leaves the bridge holding the world still.

    The cached reading of a paused run still says "active", so without this the
    next episode would start on a view of a world that had stopped moving, and
    on a sequence frozen at whatever the previous episode last saw.
    """
    client = FakeClient(states=[_observation(sequence=9)])

    _adapter(client, FakeDevice()).begin_episode()

    resume = client.sent[0]
    assert resume["kind"] == "lifecycle" and resume["action"] == "unpause"
    assert resume["expected_observation_sequence"] == 9


def test_a_run_that_will_not_resume_refuses_to_start_an_episode() -> None:
    client = FakeClient(states=[_observation(sequence=9)], outcome="ambiguous")

    with pytest.raises(RunPortError, match="could not be resumed"):
        _adapter(client, FakeDevice()).begin_episode()


def test_a_finished_run_is_not_asked_to_resume() -> None:
    """The bridge never pauses a run that ended, and its `unpause` waits for an
    active run, so asking would only stall the retry flow."""
    import tower_rl.infrastructure.instrumented_run_adapter as module

    client = FakeClient(
        states=[_observation(terminal=True), _observation(terminal=True), _observation()]
    )
    device = FakeDevice()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "classify", lambda _image: Screen.RESULT)
        _adapter(client, device).begin_episode()

    assert all(message["kind"] != "lifecycle" for message in client.sent)
