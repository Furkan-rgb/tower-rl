"""The adapter that drives one real instrumented instance through `RunPort`.

It owns the two things the environment must not know about: the wire protocol,
and the fact that starting an episode still requires one tap.  That tap is the
only screen interaction in the training loop and it is gated on a positive
classification, because `M1-E005` showed what an ungated coordinate tap can
reach.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

from PIL import Image

from tower_rl.infrastructure.adb_probe import ScreenKind, classify_frame
from tower_rl.infrastructure.instrumented_bridge import (
    BridgeCommandResult,
    BridgeObservation,
    BridgeRunUnavailable,
    InstrumentedBridgeClient,
)
from tower_rl.ports.android import ScreenPoint
from tower_rl.ports.run_port import RunPortError

#: Above this speed the world moves faster than a host round trip, so the game is
#: paused between decisions and advanced in bounded slices instead. Below it,
#: free running is cheaper. Measured in `M1B-E003`.
PAUSE_STEPPING_SPEED = 16.0

#: The only two coordinates this loop may ever touch, each gated on a positive
#: screen classification immediately before use.
BATTLE_BUTTON = ScreenPoint(540 / 1080, 1553 / 1920)
RETRY_BUTTON = ScreenPoint(300 / 1080, 1417 / 1920)


class TapTarget:
    """A tap is only permitted from the screen that owns that control."""

    def __init__(self, point: ScreenPoint, permitted: ScreenKind) -> None:
        self.point = point
        self.permitted = permitted


START_FROM_HOME = TapTarget(BATTLE_BUTTON, ScreenKind.HOME)
RETRY_FROM_RESULT = TapTarget(RETRY_BUTTON, ScreenKind.RESULT)


@dataclass
class InstrumentedRunAdapter:
    """One rooted clone, presented to the environment as a semantic run port."""

    client: InstrumentedBridgeClient
    device: object  # AdbDevice-shaped: screenshot() and tap() only
    requested_speed: float = 64.0
    episode_start_timeout: float = 90.0
    settle_seconds: float = 3.0
    _last_speed: float = field(default=0.0, init=False)

    # -- reading -----------------------------------------------------------

    def read_state(self) -> BridgeObservation | None:
        state = self.client.read_state()
        if isinstance(state, BridgeRunUnavailable):
            return None
        self._last_speed = state.game_speed
        return state

    # -- lifecycle ---------------------------------------------------------

    def begin_episode(self) -> None:
        """Bring the instance into an active run, tapping only when classified."""
        deadline = time.monotonic() + self.episode_start_timeout
        while time.monotonic() < deadline:
            state = self.client.read_state()
            if isinstance(state, BridgeObservation) and not state.terminal:
                self._apply_speed(state)
                return
            # The result panel appears a moment after the bridge reports terminal,
            # so settle before looking, or the classification races the animation.
            time.sleep(self.settle_seconds)
            target = START_FROM_HOME if state is None or isinstance(
                state, BridgeRunUnavailable
            ) else RETRY_FROM_RESULT
            self._gated_tap(target)
            self._await_active(deadline)
            state = self.client.read_state()
            if isinstance(state, BridgeObservation) and not state.terminal:
                self._apply_speed(state)
                return
        raise RunPortError("the instance did not reach an active run in time")

    def _await_active(self, deadline: float) -> None:
        while time.monotonic() < deadline:
            state = self.client.read_state()
            if isinstance(state, BridgeObservation) and not state.terminal:
                return
            time.sleep(0.5)

    def _gated_tap(self, target: TapTarget) -> None:
        """Refuse to tap unless the screen is positively the expected one."""
        frame = self.device.screenshot()  # type: ignore[attr-defined]
        screen = classify_frame(Image.open(io.BytesIO(frame.png_bytes)).convert("RGB"))
        if screen is not target.permitted:
            raise RunPortError(
                f"refusing to tap: expected {target.permitted.value}, saw {screen.value}"
            )
        self.device.tap(target.point)  # type: ignore[attr-defined]

    # -- commands ----------------------------------------------------------

    def buy_upgrade(
        self, family: str, slot: int, *, expected_sequence: int
    ) -> BridgeCommandResult:
        return self.client.send_command(
            {
                "type": "command",
                "protocol_version": 1,
                "request_id": self._request_id("buy"),
                "expected_observation_sequence": expected_sequence,
                "kind": "buy_upgrade",
                "family": family,
                "index": slot,
            }
        )

    def advance(self, *, expected_sequence: int, game_ms: int) -> BridgeCommandResult:
        """Advance game time, pausing between decisions when the world is fast."""
        if self._last_speed >= PAUSE_STEPPING_SPEED:
            # Above the threshold a host round trip costs more game time than the
            # slice itself, so deliberation must not happen while the world runs.
            return self.client.send_command(
                {
                    "type": "command",
                    "protocol_version": 1,
                    "request_id": self._request_id("step"),
                    "expected_observation_sequence": expected_sequence,
                    "kind": "step",
                    "game_ms": game_ms,
                }
            )
        return self.client.send_command(
            {
                "type": "command",
                "protocol_version": 1,
                "request_id": self._request_id("wait"),
                "expected_observation_sequence": expected_sequence,
                "kind": "wait",
            }
        )

    # -- speed -------------------------------------------------------------

    def _apply_speed(self, state: BridgeObservation) -> None:
        """Request the configured training speed, and record what was applied."""
        if abs(state.game_speed - self.requested_speed) < 0.01:
            self._last_speed = state.game_speed
            return
        result = self.client.send_command(
            {
                "type": "command",
                "protocol_version": 1,
                "request_id": self._request_id("speed"),
                "expected_observation_sequence": state.sequence,
                "kind": "set_speed",
                "value": self.requested_speed,
            }
        )
        if result.outcome != "confirmed":
            raise RunPortError(f"the game refused the training speed: {result.reason}")
        applied = self.client.read_state()
        if isinstance(applied, BridgeObservation):
            self._last_speed = applied.game_speed

    def _request_id(self, kind: str) -> str:
        return f"{kind}-{time.monotonic_ns() % 1_000_000_000}"
