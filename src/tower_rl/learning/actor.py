"""The actor: drives one environment with one policy and emits replay sequences.

Sequences are fixed length with a burn-in prefix and a configurable stride, so
the learner always receives contiguous history and warms its stacked window on
the prefix rather than on zeros.  Every episode contributes both the step that
began it and the step that ended it: its front is padded by one burn-in of
filler, the last window is aligned to its end, and an episode shorter than one
window is padded further rather than dropped.  The actor never
decides whether a transition is admissible; the environment classifies it and
replay refuses what is not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tower_rl.environment.decision_time import (
    OBSERVATION_DECODE,
    POLICY_FORWARD,
    DecisionTimeProfile,
)
from tower_rl.environment.episode import REWARD_SCHEMA_VERSION, EpisodeSummary, TerminationOutcome
from tower_rl.environment.features import StateFeatures, encode_state
from tower_rl.environment.run_actions import ACTION_SCHEMA_VERSION, WAIT, action_at, action_index
from tower_rl.environment.run_environment import InstrumentedRunEnvironment
from tower_rl.environment.run_state import OBSERVATION_SCHEMA_VERSION
from tower_rl.learning.policies import Policy
from tower_rl.learning.replay import (
    PrioritizedSequenceReplay,
    ReplaySequence,
    ReplayStep,
    SequenceMetadata,
)

#: `WAIT` is index 0 of `run-action-v1`, asked rather than assumed.
WAIT_ACTION_INDEX = action_index(WAIT)


@dataclass(frozen=True)
class ActorConfig:
    """Sequence shape and exploration for one actor."""

    actor_id: str = "actor-0"
    #: `docs/solution.md` 9.4: 80 stored decisions, 40 of burn-in, a 40 step
    #: learning unroll. Halving these doubles the fraction of steps that the
    #: n-step tail leaves without a target, which is why the documented geometry
    #: is the default rather than a cheaper one.
    sequence_length: int = 80
    burn_in: int = 40
    #: Distance between the starts of consecutive sequences. A stride shorter than
    #: the learning window overlaps them, which is what R2D2 does.
    stride: int = 40
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("a sequence needs at least two steps")
        if not 0 <= self.burn_in < self.sequence_length:
            raise ValueError("burn-in must leave at least one learning step")
        if self.stride < 1:
            raise ValueError("stride must be positive")


@dataclass
class EpisodeResult:
    """What one episode produced, for both training and reporting."""

    summary: EpisodeSummary
    sequences_offered: int
    sequences_accepted: int
    total_reward: float
    #: Decisions that chose `WAIT` (action index 0). Counted here because this
    #: is where the actions are taken; the episode summary knows what the game
    #: did, not what the policy asked for. Beside `summary.purchases` it is what
    #: makes a degenerate policy - one that waits out every episode - visible
    #: while it is happening rather than only in the final wave.
    wait_decisions: int = 0
    #: The policy's ez-greedy options this episode (`StackedDqnBackbone`): how
    #: many started and the most decisions one ran for. 0 for a policy without
    #: them, and for stacked-dqn with ez-greedy off.
    options_started: int = 0
    longest_option: int = 0
    #: What a scripted policy says about its own episode, for the episode
    #: record (`TurtlePolicy.episode_detail`: the wave it switched build in).
    #: Empty for a policy that says nothing.
    policy_detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Actor:
    """One environment, one policy, emitting sequences into replay."""

    environment: InstrumentedRunEnvironment
    policy: Policy
    config: ActorConfig = field(default_factory=ActorConfig)
    replay: PrioritizedSequenceReplay | None = None
    model_version: int = 0
    #: Where this actor's decision time goes, accumulated on its own thread and
    #: never shared with another actor. Most of a decision is spent inside the
    #: environment, so a training run points the environment's profile at this
    #: one (`TrainingRun.__post_init__`) and the two charge the same buckets.
    profile: DecisionTimeProfile = field(default_factory=DecisionTimeProfile)
    #: Called on this actor's thread before every decision's forward pass, so
    #: whatever it does to the policy lands between two decisions and never
    #: inside one. A training run refreshes the policy's parameters here
    #: (`TrainingRun._before_decision`); None leaves the policy alone.
    before_decision: Callable[[], None] | None = None

    def run_episode(self) -> EpisodeResult:
        """Play one episode to its classified end and emit its sequences."""
        state = self.environment.reset()
        carried_state = self.policy.initial_state()
        steps: list[ReplayStep] = []
        total_reward = 0.0
        termination = TerminationOutcome.OPERATOR_STOP

        # No decision cap: the environment's own liveness guard (`STALLED`,
        # `run_environment.STALL_WINDOW_WALL_SECONDS`) is what ends an episode
        # that stops making progress, so this loop runs until the environment
        # says the episode is over rather than until some fixed count of
        # decisions is spent (`#88`).
        while True:
            with self.profile.span(OBSERVATION_DECODE):
                features = encode_state(state)
            if not any(features.mask):
                # No action is available, which means the run is already over.
                termination = TerminationOutcome.GAME_OVER
                break
            if self.before_decision is not None:
                self.before_decision()
            with self.profile.span(POLICY_FORWARD):
                action_index, carried_state = self.policy.act(
                    features, carried_state, epsilon=self.config.epsilon
                )
            transition = self.environment.step(action_at(action_index))
            total_reward += transition.reward
            steps.append(
                ReplayStep(
                    features=features,
                    action_index=action_index,
                    reward=transition.reward,
                    done=transition.terminated,
                    admissible=transition.admissible,
                    game_ms=transition.game_ms,
                )
            )
            if transition.termination is not None:
                termination = transition.termination
                break
            if transition.next_state is not None:
                state = transition.next_state

        summary = self.environment.summarize(termination)
        offered, accepted = self._emit(steps, summary)
        return EpisodeResult(
            summary,
            offered,
            accepted,
            total_reward,
            wait_decisions=sum(1 for step in steps if step.action_index == WAIT_ACTION_INDEX),
            # Read off the policy rather than returned by `act`, so the policy
            # interface every other arm implements is unchanged; its
            # `initial_state` above started the counts for this episode.
            options_started=int(getattr(self.policy, "options_started", 0)),
            longest_option=int(getattr(self.policy, "longest_option", 0)),
            policy_detail=dict(getattr(self.policy, "episode_detail", {})),
        )

    def _emit(self, steps: list[ReplayStep], summary: EpisodeSummary) -> tuple[int, int]:
        if self.replay is None:
            return 0, 0
        metadata = SequenceMetadata(
            episode_id=summary.episode_id,
            actor_id=self.config.actor_id,
            profile_id=summary.profile_id,
            observation_schema=OBSERVATION_SCHEMA_VERSION,
            action_schema=ACTION_SCHEMA_VERSION,
            reward_schema=REWARD_SCHEMA_VERSION,
            model_version=self.model_version,
            epsilon=self.config.epsilon,
            game_speed=summary.game_speed,
        )
        offered = accepted = 0
        # One acquisition for the whole episode: several actors write into the
        # one buffer while the learner samples it, and replay leaves that
        # discipline to its callers (see `PrioritizedSequenceReplay.lock`).
        # Uncontended for a single actor, which is the fleet of one.
        with self.profile.acquiring(self.replay.lock):
            for window in self._windows(steps):
                offered += 1
                # Replay is the authority on admissibility; a window containing a
                # classified failure is refused there and counted, not dropped here.
                sequence = ReplaySequence(metadata, window, self.config.burn_in)
                if self.replay.add(sequence):
                    accepted += 1
        return offered, accepted

    def _windows(self, steps: list[ReplayStep]) -> list[tuple[ReplayStep, ...]]:
        """Cut one episode into learning windows, first and terminal steps included.

        Striding over the episode alone loses both of its ends. A window's
        burn-in prefix is never a target, so window 0's first `burn_in` steps -
        the opening decisions of every episode - were never learned (#92). And
        striding emits whole windows only, so the step that ends the episode
        reached replay only when the episode length happened to be a multiple
        of the stride, and an episode shorter than one window not at all.
        Termination is the whole of the negative signal under `reward-v1`, so
        that loss is silent and severe.

        Two rules fix both. Every episode is left-padded with `burn_in` filler
        steps, so window 0's burn-in is filler and the first decision is its
        first learning step; an episode still too short for one window is
        padded further, up to a full window, so its real steps land at the
        end. The filler is the zero history the episode really started from
        (`_left_padded`). And the last window is aligned to the end of the
        episode, overlapping its predecessor where it must; overlap only
        duplicates experience, whereas a missing step is never learned at all.
        """
        length, stride = self.config.sequence_length, self.config.stride
        if not steps:
            # The run was already over when the episode opened; there is no
            # decision to learn from, padding included.
            return []
        padded = self._left_padded(steps, max(length, len(steps) + self.config.burn_in))
        starts = list(range(0, len(padded) - length + 1, stride))
        if starts[-1] + length < len(padded):
            starts.append(len(padded) - length)
        return [padded[start : start + length] for start in starts]

    @staticmethod
    def _left_padded(steps: list[ReplayStep], length: int) -> tuple[ReplayStep, ...]:
        """Fill the front of an episode up to `length` steps with steps that never train.

        The filler carries zeroed features, so the history window a stacked
        backbone builds from the prefix is the zero state it already starts an
        episode from, and it borrows the first real step's action mask so the
        masked Q-values stay finite. It is flagged as padding, and the learner
        excludes flagged steps from both the loss and the TD errors that set
        priorities.
        """
        first = steps[0]
        filler = ReplayStep(
            features=StateFeatures(
                scalars=(0.0,) * len(first.features.scalars),
                rows=(0.0,) * len(first.features.rows),
                mask=first.features.mask,
            ),
            action_index=first.action_index,
            reward=0.0,
            done=False,
            admissible=True,
            padding=True,
            # No time passes in filler, so it discounts nothing.
            game_ms=0.0,
        )
        return (filler,) * (length - len(steps)) + tuple(steps)
