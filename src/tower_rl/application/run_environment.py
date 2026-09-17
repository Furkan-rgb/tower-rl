"""The instrumented Tier-1 run environment.

Owns decision cadence, transition assembly, reward, and episode classification.
It knows nothing about Android, sockets, taps, or pausing: those live behind
`RunPort`.  It emits no transition until the next state has passed validation, so
an environment failure can never be mistaken for an ordinary `WAIT`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from tower_rl.domain.episode import (
    ActionOutcome,
    DecisionEvent,
    EpisodeSummary,
    RunTransition,
    TerminationOutcome,
    wave_progress_reward,
)
from tower_rl.domain.run_actions import RunActionId, action_index
from tower_rl.domain.run_state import RunState, RunStateBuilder, validate_transition
from tower_rl.ports.run_port import RunPort, RunPortError


@dataclass(frozen=True)
class CadenceConfig:
    """Event-triggered cadence, expressed in game time (see solution.md 7.3)."""

    #: One advance slice. Small enough to notice an event promptly, large enough
    #: that a quiet stretch does not cost a round trip per frame.
    slice_game_ms: int = 250
    #: Backstop: ask for a decision even when nothing else changed.
    max_quiet_game_ms: int = 2000
    #: A health move worth interrupting for, as a fraction of maximum health.
    health_change_fraction: float = 0.05
    #: Refuse to run forever if the game stops producing terminal states.
    max_episode_wall_seconds: float = 900.0


#: The one inconsistency the bridge can legitimately show. Health and the round
#: flag are read separately, so at the instant of death health goes negative a
#: moment before the game flips game-over (M1B-E008). The settled state arrives
#: on the next read, so this is recovered rather than excluded.
DEATH_BOUNDARY_TRANSIENT = "negative health in an active run"


@dataclass
class _EpisodeTally:
    decisions: int = 0
    purchases: int = 0
    invalid_transitions: int = 0
    recovered_transients: int = 0
    started_at: float = 0.0
    peak_wave: int = 0


@dataclass
class InstrumentedRunEnvironment:
    """One instrumented game instance presented as a semantic environment."""

    port: RunPort
    builder: RunStateBuilder
    cadence: CadenceConfig = field(default_factory=CadenceConfig)
    _state: RunState | None = field(default=None, init=False)
    _episode_id: str = field(default="", init=False)
    _tally: _EpisodeTally = field(default_factory=_EpisodeTally, init=False)
    _last_reasons: tuple[str, ...] = field(default=(), init=False)

    # -- episode lifecycle -------------------------------------------------

    def reset(self) -> RunState:
        """Begin an episode and return its first valid active state."""
        self.port.begin_episode()
        state = self._read_state()
        if state is None or state.lifecycle != "active":
            raise RunPortError("the instance did not reach an active run")
        self._state = state
        self._episode_id = uuid.uuid4().hex
        self._tally = _EpisodeTally(started_at=time.monotonic(), peak_wave=state.wave)
        self._last_reasons = ()
        return state

    @property
    def state(self) -> RunState:
        if self._state is None:
            raise RunPortError("the environment has no episode in progress")
        return self._state

    def summarize(self, termination: TerminationOutcome) -> EpisodeSummary:
        """Close out the episode record for the outcome the caller observed."""
        state = self.state
        return EpisodeSummary(
            episode_id=self._episode_id,
            profile_id=state.profile_id,
            final_wave=self._tally.peak_wave,
            decisions=self._tally.decisions,
            purchases=self._tally.purchases,
            termination=termination,
            elapsed_wall_seconds=round(time.monotonic() - self._tally.started_at, 3),
            game_speed=state.game_speed,
            invalid_transitions=self._tally.invalid_transitions,
            termination_detail=self._last_reasons,
        )

    # -- stepping ----------------------------------------------------------

    def step(self, action: RunActionId) -> RunTransition:
        """Execute one semantic decision and advance to the next decision point."""
        state = self.state
        started = time.monotonic()
        self._tally.decisions += 1
        mask = state.action_mask

        if not mask[action_index(action)]:
            # A masked action is refused by the environment rather than sent. The
            # game would reject it anyway; refusing here keeps the failure
            # attributable to the policy instead of to the device.
            return self._finish(
                state, None, action, ActionOutcome.UNAVAILABLE, started, 0, (),
                ("action is masked",),
            )

        outcome = ActionOutcome.WAITED
        if not action.is_wait:
            assert action.family is not None and action.slot is not None
            result = self.port.buy_upgrade(
                action.family.value, action.slot, expected_sequence=state.source_sequence
            )
            outcome = _purchase_outcome(result.outcome)
            if outcome is ActionOutcome.EXECUTED:
                self._tally.purchases += 1
            if outcome in (ActionOutcome.AMBIGUOUS, ActionOutcome.FAILED):
                # An unconfirmed purchase leaves the game in a state the record
                # cannot describe, so the episode is classified, not continued.
                after = self._read_state()
                return self._finish(
                    state, after, action, outcome, started, 0, (),
                    (f"purchase was not confirmed: {result.reason}",),
                )

        next_state, events, requested_ms, reasons = self._advance_to_decision(state, action)
        return self._finish(
            state, next_state, action, outcome, started, requested_ms, events, reasons
        )

    # -- internals ---------------------------------------------------------

    def _advance_to_decision(
        self, state: RunState, action: RunActionId
    ) -> tuple[RunState | None, tuple[DecisionEvent, ...], int, tuple[str, ...]]:
        """Advance game time until something actionable changes."""
        events: list[DecisionEvent] = []
        requested_ms = 0
        latest = state

        if not action.is_wait:
            # A confirmed purchase already changed the decision problem: cash fell
            # and that slot's price rose. Re-decide immediately rather than
            # advancing the world first.
            after = self._read_state()
            if after is None:
                return None, (DecisionEvent.RUN_ENDED,), 0, ("run ended during the purchase",)
            events.append(DecisionEvent.PURCHASE_SETTLED)
            return after, tuple(events), 0, validate_transition(state, after)

        while requested_ms < self.cadence.max_quiet_game_ms:
            if time.monotonic() - self._tally.started_at > self.cadence.max_episode_wall_seconds:
                return (
                    latest,
                    (DecisionEvent.SLICE_ELAPSED,),
                    requested_ms,
                    ("episode exceeded its wall-clock limit",),
                )
            self.port.advance(
                expected_sequence=latest.source_sequence, game_ms=self.cadence.slice_game_ms
            )
            requested_ms += self.cadence.slice_game_ms
            observed = self._read_state()
            if observed is None:
                return None, (DecisionEvent.RUN_ENDED,), requested_ms, ()
            latest = observed
            events = list(self._events_between(state, latest))
            if events:
                break
        else:
            events = [DecisionEvent.SLICE_ELAPSED]

        return (
            latest,
            tuple(events or [DecisionEvent.SLICE_ELAPSED]),
            requested_ms,
            validate_transition(state, latest),
        )

    def _events_between(self, before: RunState, after: RunState) -> tuple[DecisionEvent, ...]:
        events: list[DecisionEvent] = []
        if after.terminal or after.lifecycle != "active":
            events.append(DecisionEvent.RUN_ENDED)
        if after.wave != before.wave:
            events.append(DecisionEvent.WAVE_CHANGED)
        gained = zip(before.action_mask, after.action_mask, strict=True)
        if any(now and not was for was, now in gained):
            events.append(DecisionEvent.NEWLY_AFFORDABLE)
        health_move = abs(after.health_fraction - before.health_fraction)
        if health_move >= self.cadence.health_change_fraction:
            events.append(DecisionEvent.HEALTH_CHANGED)
        return tuple(events)

    def _read_state(self) -> RunState | None:
        reading = self.port.read_state()
        if reading is None:
            return None
        state = self.builder.build(reading, captured_at_monotonic=time.monotonic())
        if tuple(state.invalid_reasons) == (DEATH_BOUNDARY_TRANSIENT,):
            # Read once more rather than discarding an otherwise complete episode.
            # One retry only: a state that stays contradictory is a real failure.
            settled = self.port.read_state()
            if settled is not None:
                self._tally.recovered_transients += 1
                state = self.builder.build(settled, captured_at_monotonic=time.monotonic())
        return state

    def _finish(
        self,
        state: RunState,
        next_state: RunState | None,
        action: RunActionId,
        outcome: ActionOutcome,
        started: float,
        requested_ms: int,
        events: tuple[DecisionEvent, ...],
        reasons: tuple[str, ...],
    ) -> RunTransition:
        termination = _classify(next_state, outcome, reasons)
        if termination is not None:
            detail = list(reasons)
            if next_state is not None and not next_state.valid:
                detail.extend(f"state: {reason}" for reason in next_state.invalid_reasons)
            self._last_reasons = tuple(detail)
        if next_state is not None:
            self._state = next_state
            self._tally.peak_wave = max(self._tally.peak_wave, next_state.wave)
        transition = RunTransition(
            state=state,
            next_state=next_state,
            action=action,
            action_mask=state.action_mask,
            outcome=outcome,
            reward=wave_progress_reward(state, next_state),
            terminated=termination is TerminationOutcome.GAME_OVER,
            truncated=termination is not None and termination is not TerminationOutcome.GAME_OVER,
            termination=termination,
            events=events,
            elapsed_wall_seconds=round(time.monotonic() - started, 4),
            requested_game_ms=requested_ms,
            invalid_reasons=reasons,
        )
        if not transition.admissible:
            self._tally.invalid_transitions += 1
        return transition


def _purchase_outcome(outcome: str) -> ActionOutcome:
    return {
        "confirmed": ActionOutcome.EXECUTED,
        "rejected": ActionOutcome.UNAVAILABLE,
        "ambiguous": ActionOutcome.AMBIGUOUS,
    }.get(outcome, ActionOutcome.FAILED)


def _classify(
    next_state: RunState | None, outcome: ActionOutcome, reasons: tuple[str, ...]
) -> TerminationOutcome | None:
    """Give every stopping condition its own name, never a shared 'failed'."""
    if outcome in (ActionOutcome.AMBIGUOUS, ActionOutcome.FAILED):
        return TerminationOutcome.ACTION_PIPELINE_FAILED
    if any("wall-clock limit" in reason for reason in reasons):
        return TerminationOutcome.MAX_EPISODE_DURATION
    if next_state is None:
        return TerminationOutcome.UI_STATE_LOST
    if not next_state.valid or reasons:
        return TerminationOutcome.OBSERVATION_INVALID
    if next_state.terminal:
        return TerminationOutcome.GAME_OVER
    return None
