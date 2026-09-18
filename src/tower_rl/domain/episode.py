"""Episode outcomes and transitions for an instrumented Tier-1 run."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tower_rl.domain.run_actions import RunActionId
from tower_rl.domain.run_state import RunState

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
    events: tuple[DecisionEvent, ...]
    elapsed_wall_seconds: float
    #: Game time the environment *requested*, not measured: this build exposes no
    #: live in-run clock (see M1B-E003), so the honest record is the request.
    requested_game_ms: int
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
class EpisodeSummary:
    """The immutable record of one attempted episode."""

    episode_id: str
    profile_id: str
    final_wave: int
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

    @property
    def valid(self) -> bool:
        return self.termination in VALID_TERMINATIONS
