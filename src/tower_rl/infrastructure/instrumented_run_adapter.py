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
from dataclasses import dataclass

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

#: The game's own speed multiplier, pinned at 1x. It is not a speed-up mechanism
#: for this project: a faster game clock makes every rendered frame worth more
#: game time, which coarsens the agent's decisions in exact proportion to the
#: speed gained (`M1B-E012`). Speed comes from stepping frames faster instead,
#: with a fixed amount of game time per frame, so the multiplier no longer buys
#: anything and is held at 1 so that nothing else silently depends on it.
GAME_SPEED = 1.0

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
    episode_start_timeout: float = 120.0
    #: The result panel animates in; classifying earlier sees a transition, not a
    #: screen, and tapping across a transition is the M1-E005 failure.
    settle_seconds: float = RESULT_SETTLE_SECONDS

    # -- reading -----------------------------------------------------------

    def read_state(self) -> BridgeObservation | None:
        state = self.client.read_state()
        if isinstance(state, BridgeRunUnavailable):
            return None
        return state

    # -- lifecycle ---------------------------------------------------------

    def begin_episode(self) -> None:
        """Bring the instance into an active run, tapping only when classified."""
        deadline = time.monotonic() + self.episode_start_timeout
        while time.monotonic() < deadline:
            state = self.client.read_state()
            if isinstance(state, BridgeObservation) and not state.terminal:
                self._pin_game_speed(state)
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
                self._pin_game_speed(state)
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

    def advance_until_event(
        self,
        *,
        expected_sequence: int,
        budget_game_ms: int,
        frame_game_ms: float,
        health_change_fraction: float,
    ) -> BridgeCommandResult:
        """Step frames in the bridge until an event or the budget, in one trip.

        The host used to ask for one short slice at a time and re-read the state
        after each, which cost a round trip per slice and about eight of them per
        decision. The bridge now runs that loop itself, so the frame rather than
        the round trip governs what a decision costs.

        The result carries the settled observation the bridge took after its own
        pause had landed, so the caller needs no further read to see where the
        world stopped.
        """
        return self.client.send_command(
            {
                "type": "command",
                "protocol_version": 1,
                "request_id": self._request_id("advance"),
                "expected_observation_sequence": expected_sequence,
                "kind": "advance",
                "budget_game_ms": budget_game_ms,
                "frame_game_ms": frame_game_ms,
                "health_change_fraction": health_change_fraction,
            }
        )

    # -- speed -------------------------------------------------------------

    def _pin_game_speed(self, state: BridgeObservation) -> None:
        """Hold the game's own multiplier at 1x; it is a pin, not a setting."""
        if abs(state.game_speed - GAME_SPEED) < 0.01:
            return
        result = self.client.send_command(
            {
                "type": "command",
                "protocol_version": 1,
                "request_id": self._request_id("speed"),
                "expected_observation_sequence": state.sequence,
                "kind": "set_speed",
                "value": GAME_SPEED,
            }
        )
        if result.outcome != "confirmed":
            raise RunPortError(f"the game refused the pinned 1x speed: {result.reason}")

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
