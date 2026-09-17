"""The adapter that drives one real instrumented instance through `RunPort`.

It owns the one thing the environment must not know about: the wire protocol.
Nothing in this loop reads a pixel. An episode boundary presses the game's own
controls - `go_home` to close a finished run, then `start_round`, which is the
home screen's own BATTLE control - and reads the game's own `round_active` and
`game_over` to see whether the round really started.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from tower_rl.infrastructure.instrumented_bridge import (
    BridgeCommandResult,
    BridgeObservation,
    BridgeRunUnavailable,
    BridgeStaleObservationError,
    InstrumentedBridgeClient,
)
from tower_rl.ports.run_port import RunPortError

#: The game's own speed multiplier, pinned at 1x. It is not a speed-up mechanism
#: for this project: a faster game clock makes every rendered frame worth more
#: game time, which coarsens the agent's decisions in exact proportion to the
#: speed gained (`M1B-E012`). Speed comes from stepping frames faster instead,
#: with a fixed amount of game time per frame, so the multiplier no longer buys
#: anything and is held at 1 so that nothing else silently depends on it. This
#: is the declared target `_pin_game_speed` steps the game's own control onto,
#: and the speed the budgeted game time in `run_environment` assumes.
GAME_SPEED = 1.0


@dataclass
class InstrumentedRunAdapter:
    """One rooted clone, presented to the environment as a semantic run port."""

    client: InstrumentedBridgeClient
    episode_start_timeout: float = 120.0
    #: Whether a round is being played. While it is, the environment holds the
    #: observation sequence the next command must bind, so the adapter issues
    #: nothing of its own initiative: see `_command_between_rounds`.
    _round_in_progress: bool = field(default=False, init=False)

    # -- reading -----------------------------------------------------------

    def read_state(self) -> BridgeObservation | None:
        state = self._latest_state()
        if isinstance(state, BridgeRunUnavailable):
            return None
        return state

    def _latest_state(self) -> BridgeObservation | BridgeRunUnavailable:
        """The bridge's freshest state, with a lost sequence reported as a failure.

        A stream whose sequence moved backwards is the same kind of event a
        refused command is: this episode cannot be trusted, and saying so as a
        port failure costs the episode rather than the process.
        """
        try:
            return self.client.read_state()
        except BridgeStaleObservationError as stale:
            raise RunPortError(f"the bridge stream lost its sequence: {stale}") from stale

    # -- lifecycle ---------------------------------------------------------

    def begin_episode(self) -> None:
        """Bring the instance into an active run through the game's own controls.

        A finished run is closed first: the round is started by the home
        screen's BATTLE control, which only exists while the home screen is up,
        so a terminal run has to be sent home before it can be asked to start
        one. Each press is confirmed by the game's own run state, and a press
        the game does not honour raises rather than being assumed.
        """
        self._round_in_progress = False
        self._resume_a_frozen_run()
        deadline = time.monotonic() + self.episode_start_timeout
        while time.monotonic() < deadline:
            state = self._latest_state()
            if isinstance(state, BridgeObservation) and not state.terminal:
                self._start_round(state.sequence)
                return
            if isinstance(state, BridgeObservation):
                self._press("go_home", state.sequence)
                state = self._latest_state()
            self._press("start_round", state.sequence)
            self._await_active(deadline)
            state = self._latest_state()
            if isinstance(state, BridgeObservation) and not state.terminal:
                self._start_round(state.sequence)
                return
        raise RunPortError("the instance did not reach an active run in time")

    def _start_round(self, sequence: int) -> None:
        """Pin the speed for the episode about to begin, then hand the round over.

        Once this returns the environment owns the observation sequence every
        further command must bind, which is why the pin happens here and nowhere
        later. Its effect is not taken on trust: `application/run_environment.py`
        fails any episode whose round clock outruns the game time its advances
        budgeted, which is what actually verifies the multiplier (M1B-E023).
        """
        self._pin_game_speed(sequence)
        self._round_in_progress = True

    def _press(self, action: str, expected_sequence: int) -> None:
        """Press one of the game's own controls and require its own confirmation."""
        result = self._command_between_rounds(
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
        state = self.read_state()
        if not isinstance(state, BridgeObservation) or state.terminal:
            return
        result = self._command_between_rounds(
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
            state = self._latest_state()
            if isinstance(state, BridgeObservation) and not state.terminal:
                return
            time.sleep(0.5)

    # -- commands ----------------------------------------------------------

    def buy_upgrade(
        self, family: str, slot: int, *, expected_sequence: int
    ) -> BridgeCommandResult:
        return self._send(
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
        return self._send(
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

    # -- issuing commands --------------------------------------------------

    def _send(self, command: Mapping[str, object]) -> BridgeCommandResult:
        """Send one command and report a refused sequence as a port failure.

        The bridge refuses a command that does not bind the observation it last
        sent. That is an ordinary port failure - this episode cannot say what
        the world did - and it is reported as one, so the episode is classified
        and counted like any other and the next one starts from a fresh
        observation. Left as an `InstrumentedBridgeError` it escaped the
        environment entirely and killed the process, which on an unattended run
        is the difference between losing an episode and losing the night.
        """
        try:
            return self.client.send_command(command)
        except BridgeStaleObservationError as stale:
            raise RunPortError(f"the bridge refused a stale command: {stale}") from stale

    def _command_between_rounds(self, command: Mapping[str, object]) -> BridgeCommandResult:
        """Send a command of the adapter's own initiative, only between rounds.

        Every command consumes an observation sequence. While a round is being
        played the environment holds the sequence the next command must bind and
        learns the new one from the result it gets back, so a command it never
        asked for strands that expectation and the next advance is refused as
        stale. A post-advance speed pin did exactly that (M1B-E024). The
        adapter's own commands therefore belong to the episode boundary, and
        this refuses to issue one anywhere else rather than leaving the next
        such command to rediscover the hazard.
        """
        if self._round_in_progress:
            raise RunPortError(
                f"{command.get('kind')} may not be issued while a round is in progress"
            )
        return self._send(command)

    # -- speed -------------------------------------------------------------

    def _pin_game_speed(self, sequence: int) -> None:
        """Hold the game's own multiplier at 1x by pressing the game's own control.

        Writing `gameSpeed` does not hold it. The field is the rate the world is
        running at now, and the game restores its own remembered speed whenever
        it unpauses, so a value written while the bridge holds the world still
        is overwritten before the world next moves. The bridge confirmed that
        write by reading back the slot it had just written, which is how a
        confirmed 1x pin came to sit beside a world running at this account's
        1.5x ceiling (`M1B-E025`). It is the same lesson the round-start control
        taught: press what the game presses, do not write what it reads.

        `SpeedChangeMax` and `SpeedChangeDown` are the game's own speed buttons
        and they do take. Pressing to the ceiling first makes the landing
        deterministic whatever the world was left at, and this account's ladder
        puts 1x exactly one step below its 1.5x ceiling. Exactly one step down
        is taken and never more: below 1x the ladder reaches 0, which is the
        game's paused state, and a world standing still credits no game time and
        ends no episode.

        The effect is still not taken on trust. `application/run_environment.py`
        fails any episode whose round clock outruns the game time its advances
        budgeted, which is what actually witnesses the multiplier (`M1B-E023`).
        """
        self._press("speed_max", sequence)
        state = self._latest_state()
        if not isinstance(state, BridgeObservation):
            raise RunPortError("the instance stopped reporting a run while the speed was pinned")
        self._press("speed_down", state.sequence)

    def release(self) -> None:
        """Leave the game running, whatever mode this adapter used.

        A paused game outlives the client that paused it: the next session then
        advances nothing and every episode times out. `M1B-E006` saw exactly that
        cascade, so releasing is part of shutting down rather than an optimisation.
        """
        # Releasing ends whatever round the host was playing, so the unpause it
        # sends is a boundary command like any other.
        self._round_in_progress = False
        state = self.read_state()
        if isinstance(state, BridgeObservation):
            self._command_between_rounds(
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
