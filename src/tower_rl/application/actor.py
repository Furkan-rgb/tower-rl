"""The actor: drives one environment with one policy and emits replay sequences.

Sequences are fixed length with a burn-in prefix and a configurable stride, so a
recurrent learner always receives contiguous history.  The actor never decides
whether a transition is admissible; the environment classifies it and replay
refuses what is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tower_rl.application.policies import Policy, describe
from tower_rl.application.replay import (
    PrioritizedSequenceReplay,
    ReplaySequence,
    ReplayStep,
    SequenceMetadata,
)
from tower_rl.application.run_environment import InstrumentedRunEnvironment
from tower_rl.domain.episode import REWARD_SCHEMA_VERSION, EpisodeSummary, TerminationOutcome
from tower_rl.domain.features import encode_state
from tower_rl.domain.run_actions import ACTION_SCHEMA_VERSION, action_at
from tower_rl.domain.run_state import OBSERVATION_SCHEMA_VERSION


@dataclass(frozen=True)
class ActorConfig:
    """Sequence shape and exploration for one actor."""

    actor_id: str = "actor-0"
    sequence_length: int = 40
    burn_in: int = 20
    #: Distance between the starts of consecutive sequences. A stride shorter than
    #: the learning window overlaps them, which is what R2D2 does.
    stride: int = 20
    epsilon: float = 0.0
    #: Refuses to spin forever if the environment never terminates an episode.
    max_decisions_per_episode: int = 20_000

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


@dataclass
class Actor:
    """One environment, one policy, emitting sequences into replay."""

    environment: InstrumentedRunEnvironment
    policy: Policy
    config: ActorConfig = field(default_factory=ActorConfig)
    replay: PrioritizedSequenceReplay | None = None
    model_version: int = 0

    def run_episode(self) -> EpisodeResult:
        """Play one episode to its classified end and emit its sequences."""
        state = self.environment.reset()
        recurrent = self.policy.initial_state()
        steps: list[ReplayStep] = []
        total_reward = 0.0
        termination = TerminationOutcome.OPERATOR_STOP

        for _ in range(self.config.max_decisions_per_episode):
            features = encode_state(state)
            if not any(features.mask):
                # No action is available, which means the run is already over.
                termination = TerminationOutcome.GAME_OVER
                break
            action_index, recurrent = self.policy.act(
                features, recurrent, epsilon=self.config.epsilon
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
                )
            )
            if transition.termination is not None:
                termination = transition.termination
                break
            if transition.next_state is not None:
                state = transition.next_state
        else:
            termination = TerminationOutcome.MAX_EPISODE_DURATION

        summary = self.environment.summarize(termination)
        offered, accepted = self._emit(steps, summary)
        return EpisodeResult(summary, offered, accepted, total_reward)

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
        length, stride = self.config.sequence_length, self.config.stride
        for start in range(0, max(len(steps) - length + 1, 0), stride):
            window = tuple(steps[start : start + length])
            offered += 1
            # Replay is the authority on admissibility; a window containing a
            # classified failure is refused there and counted, not dropped here.
            if self.replay.add(ReplaySequence(metadata, window, self.config.burn_in)):
                accepted += 1
        return offered, accepted


def describe_actor(actor: Actor) -> str:
    return f"{actor.config.actor_id}:{describe(actor.policy)}"
