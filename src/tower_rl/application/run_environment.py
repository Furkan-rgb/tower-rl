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
from tower_rl.domain.run_state import (
    ExactRunReadingLike,
    RunState,
    RunStateBuilder,
    validate_transition,
)
from tower_rl.ports.run_port import AdvanceResultLike, RunPort, RunPortError


@dataclass(frozen=True)
class CadenceConfig:
    """Event-triggered cadence, expressed in game time (see solution.md 9.2c).

    Every field here is game time or a game quantity. Wall clock appears once,
    as a hang deadline, because how long a run takes in real seconds is a
    property of the host rather than of the decision problem.
    """

    #: What one rendered frame is worth. This is the floor on decision
    #: granularity and it is fixed, so the same game moments are offered to the
    #: policy however fast the host renders.
    frame_game_ms: float = 1000.0 / 60.0
    #: Backstop: ask for a decision even when nothing else changed. Also the
    #: budget one advance may spend before returning.
    max_quiet_game_ms: int = 2000
    #: A health move worth interrupting for, as a fraction of maximum health.
    health_change_fraction: float = 0.05
    #: Refuse to run forever if the game stops producing terminal states.
    max_episode_wall_seconds: float = 900.0


#: The one inconsistency the bridge can legitimately show. Health and the round
#: flag are read separately, so at the instant of death health goes negative a
#: moment before the game flips game-over (M1B-E008). The two agree again once
#: the game has processed another frame, so this is recovered rather than
#: excluded - by advancing the world minimally, never by looking again: between
#: decisions the bridge holds the world paused, and a frozen world answers a
#: second read with the identical reading.
DEATH_BOUNDARY_TRANSIENT = "negative health in an active run"

#: The smallest advance the wire protocol will carry, mirroring
#: `MIN_ADVANCE_BUDGET_GAME_MS` in `infrastructure/instrumented_bridge.py`. The
#: death-boundary recovery asks for one frame of game time, or for this floor
#: when a frame is worth less than the protocol allows.
MIN_ADVANCE_GAME_MS = 10

#: The bridge stops advancing at the first decision condition it sees, but
#: `_events_between` below remains the only definition of what a decision
#: condition *is*. When the two disagree the transition is recorded as invalid
#: rather than the host predicate being relaxed to match: a silent drift between
#: the two would change the decision problem without anything saying so.
BRIDGE_EVENT_DIVERGENCE = "the bridge and the host disagree about the decision event"

#: What the bridge prefixes a reason with when it stopped on an event rather
#: than on the budget, and the reason it gives when it stopped on neither.
_BRIDGE_EVENT_PREFIX = "event:"
_BRIDGE_BUDGET_REASON = "budget_exhausted"


class _DeathBoundaryUnresolved(RunPortError):
    """The minimal advance that should have settled the death boundary failed.

    It is a `RunPortError` because that is what it is - the port could not move
    the world - but `step` catches it and classifies the episode, exactly as it
    classifies any other advance that could not say what it did.
    """

    def __init__(self, outcome: ActionOutcome, reason: str) -> None:
        super().__init__(reason)
        self.outcome = outcome
        self.reason = reason


@dataclass
class _EpisodeTally:
    decisions: int = 0
    purchases: int = 0
    invalid_transitions: int = 0
    recovered_transients: int = 0
    started_at: float = 0.0
    peak_wave: int = 0
    #: What advancing this episode actually cost the game clock. Wall seconds
    #: divided into game seconds is the speed-up, which is the number the
    #: stepping design is judged on.
    frames: int = 0
    game_ms: float = 0.0
    #: The game's own per-round clock across those same advances, and the wall
    #: time they took. Wall time minus advance wall time is the per-decision
    #: boundary cost.
    round_ms: float = 0.0
    advance_wall_micros: int = 0
    #: Advances the bridge ended early, on its own wall-clock ceiling, without
    #: either spending the budget or finding an event.
    advances_cut_short: int = 0
    #: The speed the run was seen executing at while it was still running. The
    #: final state is always terminal and the game has stopped time by then, so
    #: sampling there reports zero for every episode (M1B-E009).
    active_game_speed: float = 0.0


@dataclass(frozen=True)
class _Advance:
    """What advancing to the next decision produced."""

    state: RunState | None
    events: tuple[DecisionEvent, ...]
    requested_game_ms: int
    reasons: tuple[str, ...]
    #: Set only when the port itself failed, in which case it replaces the
    #: action's own outcome so the episode is classified as a pipeline failure
    #: rather than as an ordinary wait.
    failure: ActionOutcome | None = None


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
        self._tally = _EpisodeTally(
            started_at=time.monotonic(),
            peak_wave=state.wave,
            active_game_speed=state.game_speed,
        )
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
            game_speed=self._tally.active_game_speed,
            frames=self._tally.frames,
            game_ms=round(self._tally.game_ms, 3),
            round_ms=round(self._tally.round_ms, 3),
            advance_wall_seconds=round(self._tally.advance_wall_micros / 1_000_000, 3),
            advances_cut_short=self._tally.advances_cut_short,
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
        purchase_result: AdvanceResultLike | None = None
        try:
            if not action.is_wait:
                assert action.family is not None and action.slot is not None
                purchase_result = self.port.buy_upgrade(
                    action.family.value, action.slot, expected_sequence=state.source_sequence
                )
                outcome = _purchase_outcome(purchase_result.outcome)
                if outcome is ActionOutcome.EXECUTED:
                    self._tally.purchases += 1
                if outcome in (ActionOutcome.AMBIGUOUS, ActionOutcome.FAILED):
                    # An unconfirmed purchase leaves the game in a state the record
                    # cannot describe, so the episode is classified, not continued.
                    after = self._read_state()
                    return self._finish(
                        state, after, action, outcome, started, 0, (),
                        (f"purchase was not confirmed: {purchase_result.reason}",),
                    )

            advanced = self._advance_to_decision(state, action, purchase_result)
        except _DeathBoundaryUnresolved as unresolved:
            # A death boundary the world would not settle is a pipeline failure,
            # not a wait: nothing here can say what the game did next.
            return self._finish(
                state, None, action, unresolved.outcome, started, 0, (), (unresolved.reason,)
            )
        return self._finish(
            state,
            advanced.state,
            action,
            advanced.failure or outcome,
            started,
            advanced.requested_game_ms,
            advanced.events,
            advanced.reasons,
        )

    # -- internals ---------------------------------------------------------

    def _advance_to_decision(
        self,
        state: RunState,
        action: RunActionId,
        purchase_result: AdvanceResultLike | None = None,
    ) -> _Advance:
        """Advance game time until something actionable changes."""
        if not action.is_wait:
            # A confirmed purchase already changed the decision problem: cash fell
            # and that slot's price rose. Re-decide immediately rather than
            # advancing the world first. The command result carries the state
            # the bridge sent immediately before it, the same mechanism an
            # advance uses, so reading again would cost a second round trip and
            # could only show a later state than the one the result describes.
            assert purchase_result is not None
            after = self._build_state(purchase_result.state)
            if after is None:
                reason = ("run ended during the purchase",)
                return _Advance(None, (DecisionEvent.RUN_ENDED,), 0, reason)
            return _Advance(
                after,
                (DecisionEvent.PURCHASE_SETTLED,),
                0,
                validate_transition(state, after),
            )

        if time.monotonic() - self._tally.started_at > self.cadence.max_episode_wall_seconds:
            return _Advance(
                state,
                (DecisionEvent.SLICE_ELAPSED,),
                0,
                ("episode exceeded its wall-clock limit",),
            )

        budget = self.cadence.max_quiet_game_ms
        result = self.port.advance_until_event(
            expected_sequence=state.source_sequence,
            budget_game_ms=budget,
            frame_game_ms=self.cadence.frame_game_ms,
            health_change_fraction=self.cadence.health_change_fraction,
        )
        self._tally.frames += result.frames
        self._tally.game_ms += result.game_ms
        self._tally.round_ms += result.round_ms
        self._tally.advance_wall_micros += result.wall_micros
        if result.reason == _BRIDGE_BUDGET_REASON and result.game_ms < budget:
            # The bridge stopped on its own wall-clock ceiling rather than on the
            # budget. The transition is genuine, so it is counted rather than
            # rejected - but counted, because it is the difference between a
            # speed-up and a stall.
            self._tally.advances_cut_short += 1
        if result.outcome != "confirmed":
            # An advance that cannot say how far it got leaves the record unable
            # to describe what happened, exactly as an unconfirmed purchase does.
            # It used to be ignored, which quietly attributed a bridge failure to
            # the policy's WAIT.
            return _Advance(
                self._read_state(),
                (),
                budget,
                (f"advance was not confirmed: {result.reason}",),
                failure=_advance_failure(result.outcome),
            )

        # The advance already carries the state the world settled at when it
        # stopped. Reading again would cost a second round trip per decision and
        # could only show a later state than the one the result describes.
        observed = self._build_state(result.state)
        if observed is None:
            return _Advance(
                None,
                (DecisionEvent.RUN_ENDED,),
                budget,
                self._divergence(result.reason, (DecisionEvent.RUN_ENDED,)),
            )
        events = self._events_between(state, observed)
        return _Advance(
            observed,
            events or (DecisionEvent.SLICE_ELAPSED,),
            budget,
            validate_transition(state, observed) + self._divergence(result.reason, events),
        )

    def _divergence(
        self, bridge_reason: str, events: tuple[DecisionEvent, ...]
    ) -> tuple[str, ...]:
        """Compare the bridge's stopping reason with this environment's predicate.

        The predicate is never softened to agree with the bridge. If the two drift
        apart the decisions the agent is offered have changed, so the transition is
        marked invalid and counted, which is what makes the drift visible.
        """
        claims_event = bridge_reason.startswith(_BRIDGE_EVENT_PREFIX)
        if claims_event != bool(events):
            return (BRIDGE_EVENT_DIVERGENCE,)
        return ()

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
        return self._build_state(self.port.read_state())

    def _build_state(self, reading: ExactRunReadingLike | None) -> RunState | None:
        if reading is None:
            return None
        state = self.builder.build(reading, captured_at_monotonic=time.monotonic())
        if tuple(state.invalid_reasons) == (DEATH_BOUNDARY_TRANSIENT,):
            return self._settle_death_boundary(state)
        return state

    def _settle_death_boundary(self, state: RunState) -> RunState:
        """Let the game take one more frame so the death boundary can resolve.

        The world is paused while the host decides, and a frozen world cannot
        resolve an inconsistency by being observed again: the same reading comes
        back. What settles the boundary is the game processing another frame, so
        the recovery asks for the smallest advance there is - a single frame's
        worth of game time - and takes the settled observation it returns.

        One attempt only. A state that is still contradictory after the world has
        moved on is a real failure, and it keeps its reasons so the episode is
        classified invalid rather than quietly accepted.
        """
        result = self.port.advance_until_event(
            expected_sequence=state.source_sequence,
            budget_game_ms=max(MIN_ADVANCE_GAME_MS, int(self.cadence.frame_game_ms)),
            frame_game_ms=self.cadence.frame_game_ms,
            health_change_fraction=self.cadence.health_change_fraction,
        )
        # This frame really was stepped, so it is charged to the episode like any
        # other: the speed-up is measured from what the game clock actually cost.
        self._tally.frames += result.frames
        self._tally.game_ms += result.game_ms
        self._tally.round_ms += result.round_ms
        self._tally.advance_wall_micros += result.wall_micros
        if result.outcome != "confirmed" or result.state is None:
            # The same failure an unconfirmed advance is: the record cannot say
            # what the world did, so the episode is classified, never continued.
            raise _DeathBoundaryUnresolved(
                _advance_failure(result.outcome),
                f"the death boundary did not settle: {result.reason}",
            )
        settled = self.builder.build(result.state, captured_at_monotonic=time.monotonic())
        if settled.valid:
            self._tally.recovered_transients += 1
        return settled

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
            if next_state.lifecycle == "active":
                self._tally.active_game_speed = next_state.game_speed
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


def _advance_failure(outcome: str) -> ActionOutcome:
    return ActionOutcome.AMBIGUOUS if outcome == "ambiguous" else ActionOutcome.FAILED


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
