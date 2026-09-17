"""The adapter that drives one real instrumented instance through `RunPort`.

It owns the one thing the environment must not know about: the wire protocol.
Nothing in this loop reads a pixel. An episode boundary presses the game's own
controls - `go_home` to close a finished run, then `start_round`, which is the
home screen's own BATTLE control - and reads the game's own `round_active` and
`game_over` to see whether the round really started.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from tower_rl.infrastructure.instrumented_bridge import (
    BridgeCommandResult,
    BridgeObservation,
    BridgeRunUnavailable,
    InstrumentedBridgeClient,
)
from tower_rl.ports.run_port import RunPortError

#: The game's own speed multiplier, pinned at 1x. It is not a speed-up mechanism
#: for this project: a faster game clock makes every rendered frame worth more
#: game time, which coarsens the agent's decisions in exact proportion to the
#: speed gained (`M1B-E012`). Speed comes from stepping frames faster instead,
#: with a fixed amount of game time per frame, so the multiplier no longer buys
#: anything and is held at 1 so that nothing else silently depends on it.
GAME_SPEED = 1.0


@dataclass
class InstrumentedRunAdapter:
    """One rooted clone, presented to the environment as a semantic run port."""

    client: InstrumentedBridgeClient
    episode_start_timeout: float = 120.0
    #: Whether this episode still owes the pin it can only apply once the world
    #: has moved. Set when an episode begins, cleared by the first advance.
    _pin_after_first_advance: bool = field(default=False, init=False)

    # -- reading -----------------------------------------------------------

    def read_state(self) -> BridgeObservation | None:
        state = self.client.read_state()
        if isinstance(state, BridgeRunUnavailable):
            return None
        return state

    # -- lifecycle ---------------------------------------------------------

    def begin_episode(self) -> None:
        """Bring the instance into an active run through the game's own controls.

        A finished run is closed first: the round is started by the home
        screen's BATTLE control, which only exists while the home screen is up,
        so a terminal run has to be sent home before it can be asked to start
        one. Each press is confirmed by the game's own run state, and a press
        the game does not honour raises rather than being assumed.
        """
        self._resume_a_frozen_run()
        deadline = time.monotonic() + self.episode_start_timeout
        while time.monotonic() < deadline:
            state = self.client.read_state()
            if isinstance(state, BridgeObservation) and not state.terminal:
                self._begin_pinned(state.sequence)
                return
            if isinstance(state, BridgeObservation):
                self._press("go_home", state.sequence)
                state = self.client.read_state()
            self._press("start_round", state.sequence)
            self._await_active(deadline)
            state = self.client.read_state()
            if isinstance(state, BridgeObservation) and not state.terminal:
                self._begin_pinned(state.sequence)
                return
        raise RunPortError("the instance did not reach an active run in time")

    def _press(self, action: str, expected_sequence: int) -> None:
        """Press one of the game's own controls and require its own confirmation."""
        result = self.client.send_command(
            {
                "type": "command",
                "protocol_version": 1,
                "request_id": self._request_id(action),
                "expected_observation_sequence": expected_sequence,
                "kind": "lifecycle",
                "action": action,
            }
        )
        if result.outcome != "confirmed":
            raise RunPortError(f"the game did not honour {action}: {result.reason}")

    def _resume_a_frozen_run(self) -> None:
        """Never start an episode on a reading of a world that is standing still.

        An episode can end host-side - a truncation, an invalid observation, a
        pipeline failure - while the run itself is still going. The bridge then
        holds that run paused and streams no new state, so `begin_episode` would
        see a cached active reading, return at once, and hand the policy a run it
        has already been playing as though it had just begun.

        Unpausing is what makes the bridge stream again; the game's own state
        after the command is therefore current. A run that is already over is
        never the standing-still case - the bridge holds the sequence only for a
        pause its settled state confirms, so a terminal reading is always a
        reading it is still refreshing - and its own `unpause` waits for an
        active run. So a finished run is left alone for `begin_episode` to close
        and restart through the game's own controls:
        asking it to resume would only stall the boundary on a lifecycle timeout.
        """
        state = self.client.read_state()
        if not isinstance(state, BridgeObservation) or state.terminal:
            return
        result = self.client.send_command(
            {
                "type": "command",
                "protocol_version": 1,
                "request_id": self._request_id("unpause"),
                "expected_observation_sequence": state.sequence,
                "kind": "lifecycle",
                "action": "unpause",
            }
        )
        if result.outcome != "confirmed":
            raise RunPortError(
                f"the previous run could not be resumed to start an episode: {result.reason}"
            )

    def _await_active(self, deadline: float) -> None:
        while time.monotonic() < deadline:
            state = self.client.read_state()
            if isinstance(state, BridgeObservation) and not state.terminal:
                return
            time.sleep(0.5)

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
        result = self.client.send_command(
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
        # An advance is the first thing in an episode that unpauses the world,
        # and that is the moment a speed the boundary pin never saw takes hold.
        # The result carries the settled observation the bridge paused on, so
        # the pin is bound to a sequence that still stands; an advance that
        # carries no state leaves the debt for the next one.
        if self._pin_after_first_advance and result.state is not None:
            self._pin_after_first_advance = False
            self._pin_game_speed(result.state.sequence)
        return result

    # -- speed -------------------------------------------------------------

    def _begin_pinned(self, sequence: int) -> None:
        """Pin the speed for a starting episode, and owe the pin one more time.

        Starting a round is not the last moment the multiplier can change: the
        world is standing still when an episode begins, and whatever the game
        holds while it is still takes effect when it next moves. So the pin is
        applied here and again after the episode's first advance.
        """
        self._pin_game_speed(sequence)
        self._pin_after_first_advance = True

    def _pin_game_speed(self, sequence: int) -> None:
        """Hold the game's own multiplier at 1x; it is a pin, not a setting.

        Applied unconditionally. It used to be skipped when the observed
        `game_speed` already read 1x, which made the pin depend on a field that
        cannot witness it: every observation the host sees is taken from a world
        the bridge has paused, and the field reads 0.0 there whatever the
        unpaused world runs at (M1B-E009). A precondition read from a field that
        cannot report the truth is a pin that silently never fires.
        """
        result = self.client.send_command(
            {
                "type": "command",
                "protocol_version": 1,
                "request_id": self._request_id("speed"),
                "expected_observation_sequence": sequence,
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
