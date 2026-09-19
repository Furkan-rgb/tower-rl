"""The instrumented Tier-1 run environment.

Owns decision cadence, transition assembly, reward, and episode classification.
It knows nothing about Android, sockets, taps, or pausing: those live behind
`RunPort`.  It emits no transition until the next state has passed validation, so
an environment failure can never be mistaken for an ordinary `WAIT`.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum

from tower_rl.environment.decision_time import (
    BRIDGE_ROUND_TRIP,
    OBSERVATION_DECODE,
    DecisionTimeProfile,
)
from tower_rl.environment.episode import (
    ActionOutcome,
    DecisionEvent,
    DecisionView,
    EpisodeSummary,
    RunTransition,
    TerminationOutcome,
    WaveRecord,
    wave_progress_reward,
)
from tower_rl.environment.run_actions import WAIT, RunActionId, action_index
from tower_rl.environment.run_port import AdvanceResultLike, RunPort, RunPortError
from tower_rl.environment.run_state import (
    ExactRunReadingLike,
    RunState,
    RunStateBuilder,
    validate_transition,
)


class DecisionCadence(StrEnum):
    """When the environment asks the policy for a decision (ADR 0009).

    Not the same thing as `CadenceConfig`, which says when the *world* stops
    advancing. This says which of those stops the policy is actually asked
    about. The two are independent: every stop is a normal cadence stop,
    checked against the host predicate and charged to the episode's tallies
    either way.
    """

    #: The contract. A decision is asked for only at a choice point - a state
    #: whose legal set contains at least one purchase. A slice whose only legal
    #: action is WAIT is answered by the environment and advanced through, and
    #: its reward and game time accrue to the surrounding decision.
    CHOICE_POINTS = "choice-points"
    #: Run 1's protocol: every cadence slice is a decision, including the 68%
    #: of them that offered nothing but WAIT (M2-E004). Kept for one purpose -
    #: reproducing run 1 and replaying its checkpoints under the protocol they
    #: were collected under - and for nothing else.
    EVERY_SLICE = "every-slice"


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
#: `MIN_ADVANCE_BUDGET_GAME_MS` in `simulation/instrumented_bridge.py`. The
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
#: than on the budget, the reason it gives when it stopped on neither, and the
#: reason it gives when its own wall-time ceiling cut the advance off.
_BRIDGE_EVENT_PREFIX = "event:"
_BRIDGE_BUDGET_REASON = "budget_exhausted"
_BRIDGE_WALL_CEILING_REASON = "wall_ceiling"


#: The most the game's own round clock may read per millisecond of game time the
#: advances budgeted. `captureDeltaTime` makes one rendered frame worth exactly
#: `frame_game_ms` however fast the game's own multiplier runs, so the two clocks
#: should agree; measured across six known-good episodes they did, at 1.069 of
#: round clock per budgeted millisecond (1.011 taken over whole episodes, the
#: difference being the settle frames' uncounted round time). The same six
#: episodes with the world left at this account's 1.5 speed ceiling measured
#: 1.625, an inflation of 1.520. This ceiling sits between the two and nearer the
#: good value than the bad one: ordinary variation around 1.069 passes, and
#: nothing running at 1.5x can (M1B-E023).
MAX_ROUND_CLOCK_RATIO = 1.25

#: The least the game's own round clock may read per millisecond of game time
#: the advances budgeted. Healthy runs pooled 1.007-1.014x across episodes,
#: with no single episode measured above 1.0140; a 150ms-step arm that
#: under-credited simulated time measured 0.987x, below every one of those
#: healthy measurements. 1.0 would be the natural floor - the round clock
#: reading less than the budgeted game time means less of the world was
#: simulated than was asked for - but it leaves no room at all for the
#: ordinary float noise a sum of many small deltas carries, and a run
#: legitimately agreeing at 1.0 must not fail on that noise alone. 0.99 keeps
#: that room, clears the lowest pooled healthy ratio by 0.017, and still sits
#: clearly above the deflated arm's 0.987 - sustained deflation, not noise,
#: is what crosses it.
MIN_ROUND_CLOCK_RATIO = 0.99

#: One advance is too short a window to judge a clock by, so the ratio is taken
#: over the episode so far and only once a full backstop budget of game time has
#: been spent. A world running at 1.5x trips the upper bound inside the first
#: few decisions, which is the point: a faster world must not be allowed to
#: finish an episode and report a flattering wave. This threshold alone would
#: not save the lower bound from the advance that ends a run - its round time
#: legitimately reads zero while its game time does not, and that alone can
#: pull an otherwise-healthy episode's pooled ratio under the floor - which is
#: why that one advance is exempted from the lower bound explicitly rather than
#: relied on to wash out statistically.
MIN_RATIO_EVIDENCE_GAME_MS = 2000.0

#: The episode ran in a world that simulated more time than it was asked for,
#: so nothing it reports is comparable with anything measured at 1x.
GAME_TIME_INFLATED = "the game simulated more time than the advance budgeted"

#: The episode ran in a world that simulated less time than it was asked for -
#: as real a fidelity failure as inflation, just in the other direction, and
#: named distinctly so a report says which way the clock disagreed.
GAME_TIME_DEFLATED = "the game simulated less time than the advance budgeted"

#: An advance ended because the bridge ran out of wall time, not because the
#: world did anything. No advance may be truncated that way: how long the host
#: took to render is not part of the decision problem, so an episode containing
#: one was measured under a different problem from every episode that was not,
#: and is failed by name rather than counted. The ceiling is 15 s against a
#: full advance of about 0.33 s at the measured 16.2 ms frame time (M1B-E032),
#: a 46x margin, so this costs nothing until actor scaling changes that - which
#: is exactly when it must be heard rather than absorbed.
ADVANCE_TRUNCATED_BY_WALL = "an advance was cut off by the bridge's wall-time ceiling"


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
class _WaveTally:
    """One wave index's share of an episode, accumulated while it is current."""

    wave: int
    #: The state the wave began at, taken from the transition that entered it.
    health_fraction: float
    cash_log: float
    #: The game's own round clock across the advances made while this wave was
    #: current - measured time, not budget.
    game_ms: float = 0.0
    decisions: int = 0
    #: Advances made while this wave was current, decided or not.
    advances: int = 0
    completed: bool = False


@dataclass
class _EpisodeTally:
    decisions: int = 0
    #: Advances made inside this episode's decision spans, decided or not.
    #: Equal to `decisions` under `every-slice`.
    advances: int = 0
    purchases: int = 0
    invalid_transitions: int = 0
    recovered_transients: int = 0
    started_at: float = 0.0
    peak_wave: int = 0
    #: The wave observed in this episode's first state. A fresh run starts at 1;
    #: anything else means the episode continued a leftover run.
    starting_wave: int = 0
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
    #: Advances the bridge stopped mid-loop on a reading the settled snapshot
    #: then did not corroborate: it spent neither its budget nor found an event
    #: the settled state still shows. The transition is genuine - the settled
    #: state is what the agent observes - so it is counted, not rejected. Not
    #: the wall-time ceiling, which is `ADVANCE_TRUNCATED_BY_WALL` and fails the
    #: episode, and not a frame budget, of which there is none (M1B-E032).
    advances_cut_short: int = 0
    #: The speed the run was seen executing at while it was still running. The
    #: final state is always terminal and the game has stopped time by then, so
    #: sampling there reports zero for every episode (M1B-E009).
    active_game_speed: float = 0.0
    #: One accumulator per wave index the episode entered, in order. The last is
    #: the current wave; every advance and every decision is charged to it.
    waves: list[_WaveTally] = field(default_factory=list)

    def enter_wave(self, state: RunState) -> None:
        """Open the accumulator for the wave `state` is in, closing the previous."""
        if self.waves:
            self.waves[-1].completed = True
        self.waves.append(
            _WaveTally(
                wave=state.wave,
                health_fraction=state.health_fraction,
                cash_log=state.cash_log,
            )
        )

    def charge_advance(self, round_ms: float) -> None:
        """Charge an advance's measured round time to the wave it started in.

        An advance that crosses a wave boundary is charged whole to the wave
        that was current when it began: the bridge reports one round-clock delta
        per advance and cannot say how it split, so the attribution is stated
        rather than guessed. It is deterministic, and at most one advance's
        worth of game time sits on either side of each boundary.
        """
        if self.waves:
            self.waves[-1].game_ms += round_ms

    def charge_decision(self) -> None:
        if self.waves:
            self.waves[-1].decisions += 1

    def charge_span_advance(self) -> None:
        """Count one advance of a decision span against the episode and its wave.

        Charged to the wave that was current when the advance began, exactly as
        its round time is, and counted only for advances a transition covers -
        the death-boundary settle is a recovery, not a slice of the decision
        problem, and is already counted as a recovered transient.
        """
        self.advances += 1
        if self.waves:
            self.waves[-1].advances += 1


@dataclass(frozen=True)
class _Advance:
    """What advancing to the next decision produced.

    One advance, or a whole span of them once forced WAIT slices have been
    folded together by `_advance_to_choice_point`. The span reads as one
    advance on purpose: the transition it becomes covers the span.
    """

    state: RunState | None
    events: tuple[DecisionEvent, ...]
    requested_game_ms: int
    reasons: tuple[str, ...]
    #: Set only when the port itself failed, in which case it replaces the
    #: action's own outcome so the episode is classified as a pipeline failure
    #: rather than as an ordinary wait.
    failure: ActionOutcome | None = None
    #: Measured round-clock game time across this advance, and how many
    #: advances it covers.
    round_ms: float = 0.0
    advances: int = 1

    def followed_by(self, step: _Advance) -> _Advance:
        """This span extended by the advance that continued it."""
        return _Advance(
            state=step.state,
            events=_merged(self.events, step.events),
            requested_game_ms=self.requested_game_ms + step.requested_game_ms,
            reasons=self.reasons + step.reasons,
            failure=step.failure,
            round_ms=self.round_ms + step.round_ms,
            advances=self.advances + step.advances,
        )


@dataclass
class InstrumentedRunEnvironment:
    """One instrumented game instance presented as a semantic environment."""

    port: RunPort
    builder: RunStateBuilder
    cadence: CadenceConfig = field(default_factory=CadenceConfig)
    #: Which cadence stops the policy is asked about. `CHOICE_POINTS` is the
    #: contract; `EVERY_SLICE` exists to reproduce run 1 (ADR 0009).
    decision_cadence: DecisionCadence = DecisionCadence.CHOICE_POINTS
    #: Where this instance's decision time goes. One profile per instance,
    #: mutated only by the actor thread that drives it (see
    #: `environment/decision_time.py`); a run publishes snapshots of it.
    profile: DecisionTimeProfile = field(default_factory=DecisionTimeProfile)
    #: An observer of the decision stream, called once per `step` with what that
    #: decision did. `None` - the default, and what every collecting and
    #: evaluating path leaves it as - costs one attribute test per decision and
    #: builds nothing. It exists for `scripts/spectate.py`, which is a human
    #: watching one run; it is not a logging hook, and nothing it is handed is
    #: a record of anything (see `DecisionView`).
    on_decision: Callable[[DecisionView], None] | None = None
    _state: RunState | None = field(default=None, init=False)
    _episode_id: str = field(default="", init=False)
    #: Episodes begun on this environment, which only the view above reports.
    _episodes: int = field(default=0, init=False)
    _tally: _EpisodeTally = field(default_factory=_EpisodeTally, init=False)
    _last_reasons: tuple[str, ...] = field(default=(), init=False)

    # -- episode lifecycle -------------------------------------------------

    def reset(self) -> RunState:
        """Begin an episode and return the first state the policy is asked about.

        Under `CHOICE_POINTS` that is the run's first choice point, which a
        fresh run need not open on: the environment advances through the
        opening slices exactly as it does inside `step`, so the first
        observation of an episode is the same kind of state as every later one.
        """
        with self.profile.span(BRIDGE_ROUND_TRIP):
            self.port.begin_episode()
        state = self._read_state()
        if state is None or state.lifecycle != "active":
            raise RunPortError("the instance did not reach an active run")
        self._state = state
        self._episode_id = uuid.uuid4().hex
        self._episodes += 1
        self._tally = _EpisodeTally(
            started_at=time.monotonic(),
            peak_wave=state.wave,
            starting_wave=state.wave,
            active_game_speed=state.game_speed,
        )
        self._tally.enter_wave(state)
        self._last_reasons = ()
        self._state = self._first_choice_point(state)
        return self._state

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
            advances=self._tally.advances,
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
            recovered_transients=self._tally.recovered_transients,
            starting_wave=self._tally.starting_wave,
            waves=tuple(
                WaveRecord(
                    wave=wave.wave,
                    completed=wave.completed,
                    game_ms=round(wave.game_ms, 3),
                    decisions=wave.decisions,
                    advances=wave.advances,
                    health_fraction=wave.health_fraction,
                    cash_log=wave.cash_log,
                )
                for wave in self._tally.waves
            ),
        )

    # -- stepping ----------------------------------------------------------

    def step(self, action: RunActionId) -> RunTransition:
        """Execute one semantic decision and advance to the next one.

        Under `CHOICE_POINTS` the next decision is the next choice point, so
        the transition this returns may cover several advances. Its reward is
        the wave progress across the whole span and its `game_ms` the measured
        game time of it; what the environment withheld is the decision, never
        an advance's accounting.
        """
        state = self.state
        started = time.monotonic()
        self._tally.decisions += 1
        self._tally.charge_decision()
        mask = state.action_mask

        if not mask[action_index(action)]:
            # A masked action is refused by the environment rather than sent. The
            # game would reject it anyway; refusing here keeps the failure
            # attributable to the policy instead of to the device.
            return self._finish(
                state, None, action, ActionOutcome.UNAVAILABLE, started, 0, (),
                ("action is masked",), advances=0, game_ms=0.0,
            )

        outcome = ActionOutcome.WAITED
        purchase_result: AdvanceResultLike | None = None
        try:
            if not action.is_wait:
                assert action.family is not None and action.slot is not None
                with self.profile.span(BRIDGE_ROUND_TRIP):
                    purchase_result = self.port.buy_upgrade(
                        action.family.value,
                        action.slot,
                        expected_sequence=state.source_sequence,
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
                        advances=0, game_ms=0.0,
                    )

            advanced = self._advance_to_choice_point(state, action, purchase_result)
        except _DeathBoundaryUnresolved as unresolved:
            # A death boundary the world would not settle is a pipeline failure,
            # not a wait: nothing here can say what the game did next.
            return self._finish(
                state, None, action, unresolved.outcome, started, 0, (), (unresolved.reason,),
                advances=0, game_ms=0.0,
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
            advances=advanced.advances,
            game_ms=advanced.round_ms,
        )

    # -- internals ---------------------------------------------------------

    def _advance_to_choice_point(
        self,
        state: RunState,
        action: RunActionId,
        purchase_result: AdvanceResultLike | None = None,
    ) -> _Advance:
        """Advance until the policy has something to choose, or the run ends.

        Under `EVERY_SLICE` this is one advance and nothing else, which is what
        run 1 collected. Under `CHOICE_POINTS` a settled state offering only
        WAIT is not a decision - the policy has one answer available - so the
        environment takes that answer itself and advances again. Every advance
        in the span is an ordinary one: same cadence, same divergence check,
        same wave and episode tallies. The span stops at a choice point, at the
        end of the run, or at the first thing that makes the transition
        inadmissible, because a span may not be built across a state the record
        cannot describe.
        """
        span = self._advance(state, action, purchase_result)
        if self.decision_cadence is DecisionCadence.EVERY_SLICE:
            return span
        while self._span_continues(span):
            assert span.state is not None
            self._enter(span.state)
            span = span.followed_by(self._advance(span.state, WAIT))
        return span

    def _advance(
        self,
        state: RunState,
        action: RunActionId,
        purchase_result: AdvanceResultLike | None = None,
    ) -> _Advance:
        """One advance of a decision span, counted against the episode."""
        self._tally.charge_span_advance()
        return self._advance_to_decision(state, action, purchase_result)

    def _span_continues(self, span: _Advance) -> bool:
        """Whether the environment may withhold the decision and advance again."""
        return (
            span.state is not None
            and span.failure is None
            and not span.reasons
            and span.state.valid
            and span.state.lifecycle == "active"
            and not span.state.terminal
            and not span.state.is_choice_point
        )

    def _first_choice_point(self, state: RunState) -> RunState:
        """Advance a freshly begun run to the first state worth deciding at.

        The same loop `step` runs, from the state `begin_episode` left. It
        fails rather than returns when the run ends or goes invalid on the way:
        an episode whose first observation is not a state a policy can act in
        never began, exactly as one that never reached an active run did not.
        """
        if self.decision_cadence is DecisionCadence.EVERY_SLICE:
            return state
        span = _Advance(state, (), 0, (), advances=0)
        while self._span_continues(span):
            assert span.state is not None
            self._enter(span.state)
            span = span.followed_by(self._advance(span.state, WAIT))
        if (
            span.reasons
            or span.state is None
            or not span.state.valid
            or span.state.lifecycle != "active"
        ):
            detail = "; ".join(span.reasons) or "the run ended"
            raise RunPortError(f"the run reached no choice point: {detail}")
        self._enter(span.state)
        return span.state

    def _enter(self, state: RunState) -> None:
        """Make one settled state current for the episode's tallies."""
        self._tally.peak_wave = max(self._tally.peak_wave, state.wave)
        if self._tally.waves and state.wave != self._tally.waves[-1].wave:
            self._tally.enter_wave(state)
        if state.lifecycle == "active":
            self._tally.active_game_speed = state.game_speed

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
        with self.profile.span(BRIDGE_ROUND_TRIP):
            result = self.port.advance_until_event(
                expected_sequence=state.source_sequence,
                budget_game_ms=budget,
                frame_game_ms=self.cadence.frame_game_ms,
                health_change_fraction=self.cadence.health_change_fraction,
            )
        self._tally.frames += result.frames
        self._tally.game_ms += result.game_ms
        self._tally.round_ms += result.round_ms
        self._tally.charge_advance(result.round_ms)
        self._tally.advance_wall_micros += result.wall_micros
        if result.reason == _BRIDGE_BUDGET_REASON and result.game_ms < budget:
            # The loop stopped on a mid-loop reading that the settled snapshot
            # does not corroborate: no event survives to the settled state and
            # the budget was not spent. The transition is genuine - the settled
            # state is what the agent observes - so it is counted rather than
            # rejected.
            self._tally.advances_cut_short += 1
        truncated = (
            (ADVANCE_TRUNCATED_BY_WALL,)
            if result.reason == _BRIDGE_WALL_CEILING_REASON
            else ()
        )
        if result.outcome != "confirmed":
            # An advance that cannot say how far it got leaves the record unable
            # to describe what happened, exactly as an unconfirmed purchase does.
            # It used to be ignored, which quietly attributed a bridge failure to
            # the policy's WAIT. Whether it also ended the run is unknown, so the
            # lower bound stays in play here exactly as it always has.
            clock_fidelity = self._round_clock_fidelity(advance_ended_run=False)
            return _Advance(
                self._read_state(),
                (),
                budget,
                (f"advance was not confirmed: {result.reason}",) + clock_fidelity + truncated,
                failure=_advance_failure(result.outcome),
                round_ms=result.round_ms,
            )

        # The advance already carries the state the world settled at when it
        # stopped. Reading again would cost a second round trip per decision and
        # could only show a later state than the one the result describes.
        observed = self._build_state(result.state)
        # The round clock resets with the round, so the one advance that ends a
        # run - whether the state vanished outright or settled terminal - always
        # reports less round time than the game time it spent getting there.
        # That is the fidelity check's one known-legitimate zero, so this
        # advance alone is exempted from the lower bound (see
        # `_round_clock_fidelity`); the upper bound stays in force.
        ended = observed is None or observed.terminal or observed.lifecycle != "active"
        clock_fidelity = self._round_clock_fidelity(advance_ended_run=ended)
        if observed is None:
            return _Advance(
                None,
                (DecisionEvent.RUN_ENDED,),
                budget,
                self._divergence(result.reason, (DecisionEvent.RUN_ENDED,))
                + clock_fidelity
                + truncated,
                round_ms=result.round_ms,
            )
        events = self._events_between(state, observed)
        return _Advance(
            observed,
            events or (DecisionEvent.SLICE_ELAPSED,),
            budget,
            validate_transition(state, observed)
            + self._divergence(result.reason, events)
            + clock_fidelity
            + truncated,
            round_ms=result.round_ms,
        )

    def _round_clock_fidelity(self, *, advance_ended_run: bool) -> tuple[str, ...]:
        """Refuse an episode whose world ran faster or slower than budgeted.

        The bridge reports both clocks per advance: the game time it budgeted
        (frames times `frame_game_ms`) and the game's own round clock across the
        same frames. They are supposed to be the same time measured twice. When
        the round clock runs away above the budget the world is simulating more
        time per frame than it was told to - the shape a speed multiplier left
        applied has; when it falls away below, the world is simulating less -
        the shape a step that starves the game of frames has. Either way every
        wave and decision count the episode goes on to report is measured in a
        different unit from the runs it will be compared with. The episode is
        failed by name rather than compensated for: scaling the frame's worth to
        match would hide the wrong assumption and keep the numbers incomparable.

        The advance that ends a run is exempt from the lower bound alone: the
        round clock resets with the round, so that one advance legitimately
        reports none of it while still having spent game time reaching the end,
        which pulls the ratio down on every death whether or not the world was
        deflated. The upper bound is never exempt - an ending advance cannot
        report more round time than it budgeted, only less or none.
        """
        if self._tally.game_ms < MIN_RATIO_EVIDENCE_GAME_MS:
            return ()
        ratio = self._tally.round_ms / self._tally.game_ms
        if ratio > MAX_ROUND_CLOCK_RATIO:
            return (f"{GAME_TIME_INFLATED}: round clock ran {ratio:.3f}x the budgeted game time",)
        if ratio < MIN_ROUND_CLOCK_RATIO and not advance_ended_run:
            return (f"{GAME_TIME_DEFLATED}: round clock ran {ratio:.3f}x the budgeted game time",)
        return ()

    def _divergence(
        self, bridge_reason: str, events: tuple[DecisionEvent, ...]
    ) -> tuple[str, ...]:
        """Compare the bridge's stopping reason with this environment's predicate.

        The predicate is never softened to agree with the bridge. If the two drift
        apart the decisions the agent is offered have changed, so the transition is
        marked invalid and counted, which is what makes the drift visible.
        """
        if bridge_reason == _BRIDGE_WALL_CEILING_REASON:
            # The wall ceiling outranks every event reason the bridge could
            # give, so it says nothing about what the settled state shows and
            # there is nothing here to compare. The truncation is already
            # reported by `ADVANCE_TRUNCATED_BY_WALL`; calling it a disagreement
            # as well would name a drift that did not happen.
            return ()
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
        with self.profile.span(BRIDGE_ROUND_TRIP):
            reading = self.port.read_state()
        return self._build_state(reading)

    def _build_state(self, reading: ExactRunReadingLike | None) -> RunState | None:
        if reading is None:
            return None
        with self.profile.span(OBSERVATION_DECODE):
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
        with self.profile.span(BRIDGE_ROUND_TRIP):
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
        self._tally.charge_advance(result.round_ms)
        self._tally.advance_wall_micros += result.wall_micros
        if result.outcome != "confirmed" or result.state is None:
            # The same failure an unconfirmed advance is: the record cannot say
            # what the world did, so the episode is classified, never continued.
            raise _DeathBoundaryUnresolved(
                _advance_failure(result.outcome),
                f"the death boundary did not settle: {result.reason}",
            )
        with self.profile.span(OBSERVATION_DECODE):
            settled = self.builder.build(result.state, captured_at_monotonic=time.monotonic())
        if result.reason == _BRIDGE_WALL_CEILING_REASON:
            # A stall long enough to hit the wall ceiling while rendering a
            # single frame is exactly what the invariant exists to hear, so this
            # advance is refused like any other truncated one. The reason rides
            # on the state because that is what this recovery returns - the
            # transition reasons are assembled by the caller, which never sees
            # this advance - and an invalid state makes the transition
            # inadmissible and the episode `OBSERVATION_INVALID` just the same.
            return replace(
                settled,
                valid=False,
                invalid_reasons=settled.invalid_reasons + (ADVANCE_TRUNCATED_BY_WALL,),
            )
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
        *,
        advances: int,
        game_ms: float,
    ) -> RunTransition:
        termination = _classify(next_state, outcome, reasons)
        if termination is not None:
            detail = list(reasons)
            if next_state is not None and not next_state.valid:
                detail.extend(f"state: {reason}" for reason in next_state.invalid_reasons)
            self._last_reasons = tuple(detail)
        if next_state is not None:
            self._state = next_state
            self._enter(next_state)
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
            advances=advances,
            game_ms=round(game_ms, 3),
            invalid_reasons=reasons,
        )
        if not transition.admissible:
            self._tally.invalid_transitions += 1
        if self.on_decision is not None:
            self.on_decision(self._view(transition))
        return transition

    def _view(self, transition: RunTransition) -> DecisionView:
        """The decision just taken, as a spectator reads it.

        Read from the transition and from nothing else, so what a panel shows
        and what the episode record holds cannot drift apart. The state quoted
        is the one the decision produced; when the port produced none there is
        nothing later to quote, so the state it was taken in stands - that
        transition is ending the episode anyway.
        """
        shown = transition.next_state or transition.state
        action = transition.action
        return DecisionView(
            episode=self._episodes,
            decision=self._tally.decisions,
            wave=shown.wave,
            cash=math.expm1(shown.cash_log),
            health_fraction=shown.health_fraction,
            action="wait" if action.is_wait else str(action),
            reward=transition.reward,
            game_ms=transition.game_ms,
            done=transition.termination is not None,
            termination=transition.termination,
        )


def _merged(
    seen: tuple[DecisionEvent, ...], found: tuple[DecisionEvent, ...]
) -> tuple[DecisionEvent, ...]:
    """The events of a span so far, extended by one advance's, without repeats.

    A span that crossed three waves stopped on `wave_changed` three times; the
    transition says the span changed wave, in the order the conditions were
    first met, and the count of advances says how often.
    """
    return seen + tuple(event for event in found if event not in seen)


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
