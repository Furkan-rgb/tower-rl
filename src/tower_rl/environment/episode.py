"""Episode outcomes and transitions for an instrumented Tier-1 run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from tower_rl.environment.run_actions import RunActionId
from tower_rl.environment.run_state import RunState

REWARD_SCHEMA_VERSION = "reward-v1"


class TerminationOutcome(StrEnum):
    """Why an episode stopped. Only `GAME_OVER` is a normal terminal transition."""

    GAME_OVER = "game_over"
    OPERATOR_STOP = "operator_stop"
    MAX_EPISODE_DURATION = "max_episode_duration"
    OBSERVATION_INVALID = "observation_invalid"
    ACTION_PIPELINE_FAILED = "action_pipeline_failed"
    UI_STATE_LOST = "ui_state_lost"
    DEVICE_FAILED = "device_failed"
    BASELINE_DRIFT = "baseline_drift"
    RECOVERY_FAILED = "recovery_failed"


#: Outcomes that describe a complete, genuine game episode. Evaluation accepts
#: only these; everything else is an environment or infrastructure failure and is
#: excluded from model-quality measurement.
VALID_TERMINATIONS: frozenset[TerminationOutcome] = frozenset({TerminationOutcome.GAME_OVER})


class ActionOutcome(StrEnum):
    """What the game did with a requested action."""

    EXECUTED = "executed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    WAITED = "waited"
    INVALID_OBSERVATION = "invalid_observation"


class DecisionEvent(StrEnum):
    """Why the environment stopped advancing and asked for a new decision."""

    WAVE_CHANGED = "wave_changed"
    NEWLY_AFFORDABLE = "newly_affordable"
    HEALTH_CHANGED = "health_changed"
    PURCHASE_SETTLED = "purchase_settled"
    SLICE_ELAPSED = "slice_elapsed"
    RUN_ENDED = "run_ended"


@dataclass(frozen=True)
class PurchaseView:
    """What one upgrade decision actually bought, as a human reads it.

    A slot index alone says nothing to a watcher: `attack:3` is not an upgrade,
    it is a coordinate. This is what the decision did to that row - the level it
    now stands at, and what reaching it cost in earned cash - which is what a
    panel needs beside the row's name to say what was bought.

    `cost` is raw cash, backed out of the row's `cost_log` exactly as
    `hud_readings` backs out a magnitude, and it is the price the row carried in
    the state the decision was taken from: after the purchase the row quotes the
    *next* level's price, which is not what was paid.
    """

    #: Which row, as the action names it: `attack:3`. The game's own name for
    #: that row is not here - the slot labels are read from the bridge by the
    #: session that watches, and naming a row is that session's job.
    action: str
    #: The level the row stands at in the state the decision produced.
    level_after: int
    max_level: int
    #: Earned cash paid, in the unit the player reads.
    cost: float


@dataclass(frozen=True)
class DecisionView:
    """One decision as a human watching the run sees it.

    The decision stream is the only seam a spectator needs: a view is emitted
    once per `InstrumentedRunEnvironment.step`, carries what that decision did,
    and is thrown away by everything that does not want it. It is deliberately
    *not* a `RunTransition` - a panel that held transitions would hold two
    whole observations per decision and would have to know `observation-v2` to
    read them - and deliberately not named `DecisionEvent`, which above is the
    cadence condition the environment stopped advancing on.

    Nothing here is a second source of truth. The durable record of an episode
    is still `EpisodeSummary`; this is a live view of the same decisions.
    """

    #: Episodes begun on this environment, counting from one.
    episode: int
    #: Decisions taken in this episode, counting from one.
    decision: int
    #: The state the decision produced, or the state it was taken in when the
    #: port produced none - which is itself a failing episode about to end.
    wave: int
    #: Earned cash, back out of the observation's log scale. `observation-v2`
    #: carries `cash_log`, and a human reads cash.
    cash: float
    health_fraction: float
    #: Every live reading in the unit the player reads it in, under the game's
    #: own `Main` field name. This is what lets a watcher hold the panel beside
    #: the HUD and check the two agree, which is how `observation-v2` is
    #: verified against the game rather than against itself.
    hud: Mapping[str, float]
    #: `wait`, or the upgrade slot bought, as `attack:3`.
    action: str
    reward: float
    #: Measured game time the decision covers: how long the agent held this
    #: choice before it was asked again. One cadence slice under `every-slice`;
    #: under `choice-points` it is the whole span of forced WAIT slices the
    #: environment advanced through.
    game_ms: float
    #: Whether this decision ended the episode, however it ended.
    done: bool
    #: Why it ended, when it did.
    termination: TerminationOutcome | None
    #: What the decision bought, when it bought something. `None` for a wait,
    #: and for a decision whose slot the state no longer carries.
    purchase: PurchaseView | None = None


def wave_progress_reward(state: RunState, next_state: RunState | None) -> float:
    """V1 reward: genuine wave progress only.

    Accumulated return therefore equals waves survived, which is the authoritative
    objective. Purchases, currency, damage and action frequency are deliberately
    unrewarded, so the policy cannot farm a proxy that does not end in a higher
    final wave.
    """
    if next_state is None:
        return 0.0
    return float(next_state.wave - state.wave)


@dataclass(frozen=True)
class RunTransition:
    """One validated environment transition, ready for replay."""

    state: RunState
    next_state: RunState | None
    action: RunActionId
    action_mask: tuple[bool, ...]
    outcome: ActionOutcome
    reward: float
    terminated: bool
    truncated: bool
    termination: TerminationOutcome | None
    #: Every cadence condition the span stopped on, in the order the span met
    #: them. One advance's events when the span is one advance, which is what
    #: `every-slice` always produces.
    events: tuple[DecisionEvent, ...]
    elapsed_wall_seconds: float
    #: Game time the environment *requested*, not measured: this build exposes no
    #: live in-run clock (see M1B-E003), so the honest record is the request.
    requested_game_ms: int
    #: Times the world was advanced for this one decision: one for an ordinary
    #: wait, more when forced WAIT slices were advanced through to reach the
    #: next choice point (ADR 0009), and zero when the decision moved nothing -
    #: a masked action, an unconfirmed purchase, or a confirmed purchase whose
    #: settled state was already worth deciding at. Never more than the port
    #: was actually asked to advance.
    advances: int = 0
    #: Measured game time across the same span: the game's own round clock,
    #: summed over its advances. Beside `requested_game_ms`, which is a budget.
    game_ms: float = 0.0
    invalid_reasons: tuple[str, ...] = ()
    reward_schema_version: str = REWARD_SCHEMA_VERSION

    @property
    def admissible(self) -> bool:
        """Whether this transition may enter replay."""
        return (
            self.next_state is not None
            and self.next_state.valid
            and self.state.valid
            and not self.invalid_reasons
        )


@dataclass(frozen=True)
class WaveRecord:
    """What one episode did while one wave index was current.

    Per-wave rows exist because a final wave is a blunt instrument: almost all
    of its between-episode variance comes from how many waves an episode
    survives, not from what any one wave is like. How long wave k took and what
    it cost is where a change in how the game is driven shows up
    (`experiment/wave_statistics.py`).
    """

    wave: int
    #: False for the wave the episode ended in, whose duration and decisions are
    #: a fragment of a wave rather than a wave. True once a later wave began.
    completed: bool
    #: Measured game time spent in this wave: the game's own round clock across
    #: the advances made while it was current, never a budget. An advance that
    #: crosses a wave boundary is charged whole to the wave that was current
    #: when it started.
    game_ms: float
    #: Decisions the environment asked for while this wave was current. Under
    #: `choice-points` these are choice points, not cadence slices.
    decisions: int
    #: Internal advances made while this wave was current, decided or not. The
    #: two counts are equal under `every-slice` and diverge once forced WAIT
    #: slices are advanced through (ADR 0009).
    advances: int
    #: The state at the *start* of this wave, as the observation carries it:
    #: cash exists only log-scaled in `observation-v2` and is recorded as such.
    health_fraction: float
    cash_log: float


@dataclass(frozen=True)
class EpisodeSummary:
    """The immutable record of one attempted episode."""

    episode_id: str
    profile_id: str
    final_wave: int
    #: Decisions the policy was asked for. Under `choice-points` that is the
    #: choice points this episode reached, which is a different unit from the
    #: cadence slices run 1 counted - hence `advances` beside it.
    decisions: int
    purchases: int
    termination: TerminationOutcome
    elapsed_wall_seconds: float
    game_speed: float
    invalid_transitions: int
    #: What advancing this episode cost the game clock: rendered frames and the
    #: game time they were worth. Game seconds over wall seconds is the speed-up.
    frames: int = 0
    game_ms: float = 0.0
    #: Internal advances the episode made, decided or not. Equal to `decisions`
    #: under `every-slice`; under `choice-points` the difference between them is
    #: what the agent was no longer asked about (ADR 0009). Both units are kept
    #: so a run collected under either cadence can be read in the other's terms.
    advances: int = 0
    #: The game's own per-round clock across the same advances. Beside
    #: `game_ms` it is what makes the intended 1:1 mapping between budgeted and
    #: passed game time checkable instead of assumed.
    round_ms: float = 0.0
    #: Wall time spent inside advances alone. The rest of `elapsed_wall_seconds`
    #: is decision-boundary overhead, which is only readable as the difference.
    advance_wall_seconds: float = 0.0
    #: Advances the bridge stopped mid-loop on a reading its own settled
    #: snapshot then did not corroborate: neither the game-time budget spent nor
    #: an event the settled state still shows. Benign - the settled state is
    #: what the agent observes - but counted, because a rise in it says the
    #: loop and the state it reports are drifting apart. Not the wall-time
    #: ceiling, which fails the episode by name instead (M1B-E032).
    advances_cut_short: int = 0
    #: Why the episode ended the way it did. An outcome without its reason cannot
    #: be diagnosed later, and a rate without reasons cannot be fixed at all.
    termination_detail: tuple[str, ...] = ()
    #: Death-boundary transients the environment recovered from by advancing a
    #: single frame, rather than by excluding the episode (M1B-E008).
    recovered_transients: int = 0
    #: The wave observed in this episode's first state. A fresh run always
    #: starts at 1; anything higher means the episode continued a leftover run
    #: instead of starting one, which is contamination that must stay visible
    #: rather than be silently recovered from (`begin_episode` refreshes a
    #: frozen leftover run but still continues it).
    starting_wave: int = 0
    #: One row per wave index this episode entered, in order. Empty only for a
    #: summary assembled without the environment's per-wave tally.
    waves: tuple[WaveRecord, ...] = ()

    @property
    def valid(self) -> bool:
        return self.termination in VALID_TERMINATIONS
