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

from tower_rl.infrastructure.instrumented_bridge import (
    BridgeCommandResult,
    BridgeObservation,
    BridgeRunUnavailable,
    InstrumentedBridgeClient,
)
from tower_rl.infrastructure.visual_profile import (
    RESULT_SETTLE_SECONDS,
    Screen,
    classify,
    describe_drift,
)
from tower_rl.ports.android import ScreenPoint
from tower_rl.ports.run_port import RunPortError

#: Pausing between decisions is disabled by default. The reasoning that it should
#: switch on above 16x was sound about decision density and wrong about cost:
#: every slice pays a host round trip and a wall-clock floor, and `M1B-E006`
#: measured the same policy at 64x reaching wave 10 in 14.6 s free-running against
#: wave 3 in 273 s stepped - roughly nineteen times the throughput, and better
#: play. Set a finite threshold only if decision density is shown to bind.
PAUSE_STEPPING_SPEED = float("inf")

#: The only two coordinates this loop may ever touch, each gated on a positive
#: screen classification immediately before use.
BATTLE_BUTTON = ScreenPoint(540 / 1080, 1553 / 1920)
RETRY_BUTTON = ScreenPoint(300 / 1080, 1417 / 1920)


class TapTarget:
    """A tap is only permitted from the screen that owns that control."""

    def __init__(self, point: ScreenPoint, permitted: Screen) -> None:
        self.point = point
        self.permitted = permitted


START_FROM_HOME = TapTarget(BATTLE_BUTTON, Screen.HOME)
RETRY_FROM_RESULT = TapTarget(RETRY_BUTTON, Screen.RESULT)


@dataclass
class InstrumentedRunAdapter:
    """One rooted clone, presented to the environment as a semantic run port."""

    client: InstrumentedBridgeClient
    device: object  # AdbDevice-shaped: screenshot() and tap() only
    requested_speed: float = 64.0
    #: Pause between decisions above this speed. Configurable because the right
    #: value is an empirical question: pausing protects decision density, but
    #: each slice costs a round trip, and M1B-E006 measures which dominates.
    pause_stepping_speed: float = PAUSE_STEPPING_SPEED
    episode_start_timeout: float = 120.0
    #: The result panel animates in; classifying earlier sees a transition, not a
    #: screen, and tapping across a transition is the M1-E005 failure.
    settle_seconds: float = RESULT_SETTLE_SECONDS
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
        image = Image.open(io.BytesIO(frame.png_bytes)).convert("RGB")
        screen = classify(image)
        if screen is not target.permitted:
            drift = describe_drift(image).get(target.permitted.value, ())
            raise RunPortError(
                f"refusing to tap: expected {target.permitted.value}, saw {screen.value}; "
                f"anchors disagreeing: {'; '.join(drift) or 'none'}"
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
        if self._last_speed >= self.pause_stepping_speed:
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

    def release(self) -> None:
        """Leave the game running, whatever mode this adapter used.

        A paused game outlives the client that paused it: the next session then
        advances nothing and every episode times out. `M1B-E006` saw exactly that
        cascade, so releasing is part of shutting down rather than an optimisation.
        """
        state = self.client.read_state()
        if isinstance(state, BridgeObservation):
            self.client.send_command(
                {
                    "type": "command",
                    "protocol_version": 1,
                    "request_id": self._request_id("unpause"),
                    "expected_observation_sequence": state.sequence,
                    "kind": "lifecycle",
                    "action": "unpause",
                }
            )

    def _request_id(self, kind: str) -> str:
        return f"{kind}-{time.monotonic_ns() % 1_000_000_000}"
