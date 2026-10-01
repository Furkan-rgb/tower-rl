"""The actor: drives one environment with one policy and emits its episodes to replay.

An episode goes to replay whole, as the steps of one stream (`_emit_stream`,
`_emit_items`), with the policy's recurrent state beside it. DreamerV3 keeps
its latent at every step; R2D2 keeps its LSTM state before the decisions at
0, 40, 80, ... (`R2D2Backbone.replay_entry`). R2D2's policy is told each
transition's game time after it (`R2D2Backbone.after_transition`), since its
next input is that transition's reward; that holds in evaluation too. The actor
never decides whether a transition is admissible; the environment classifies it
and the stream is cut at the first one that is not.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy

from tower_rl.environment.decision_time import (
    OBSERVATION_DECODE,
    POLICY_FORWARD,
    DecisionTimeProfile,
)
from tower_rl.environment.episode import REWARD_SCHEMA_VERSION, EpisodeSummary, TerminationOutcome
from tower_rl.environment.features import StateFeatures, encode_state
from tower_rl.environment.run_actions import (
    ACTION_SCHEMA_VERSION,
    RUN_ACTIONS,
    WAIT,
    action_at,
    action_index,
)
from tower_rl.environment.run_environment import InstrumentedRunEnvironment
from tower_rl.environment.run_state import OBSERVATION_SCHEMA_VERSION
from tower_rl.learning.dreamer_replay import DreamerReplay, episode_steps
from tower_rl.learning.policies import Policy
from tower_rl.learning.r2d2 import R2D2Backbone
from tower_rl.learning.r2d2_replay import (
    R2D2_SEQUENCE_PERIOD,
    R2D2Replay,
    item_layout,
    state_count,
)
from tower_rl.learning.replay import ReplayRejected, SequenceMetadata
from tower_rl.learning.step_arrays import StepArrays

#: `WAIT` is index 0 of `run-action-v1`, asked rather than assumed.
WAIT_ACTION_INDEX = action_index(WAIT)


@dataclass(frozen=True)
class Decision:
    """One decision as the actor took it: what it saw, what it did, what followed."""

    features: StateFeatures
    action_index: int
    reward: float
    done: bool
    admissible: bool
    #: The game time the transition spanned, in ms: 0 for a purchase. Required,
    #: because a default of 0 would silently mean "no discount" to a learner
    #: that discounts by game time, at every call site that forgot it.
    game_ms: float = field(kw_only=True)

    def __post_init__(self) -> None:
        if not 0 <= self.action_index < len(RUN_ACTIONS):
            raise ReplayRejected(f"action index {self.action_index} is outside the schema")
        if not math.isfinite(self.game_ms) or self.game_ms < 0.0:
            raise ReplayRejected(f"game time {self.game_ms} ms is not finite and non-negative")


@dataclass(frozen=True)
class ActorConfig:
    """One actor's name and the rate it explores at."""

    actor_id: str = "actor-0"
    epsilon: float = 0.0


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


@dataclass
class Actor:
    """One environment, one policy, emitting episodes into replay."""

    environment: InstrumentedRunEnvironment
    policy: Policy
    config: ActorConfig = field(default_factory=ActorConfig)
    #: DreamerV3's step replay stores the policy's latent at every step, read
    #: off it by `replay_entry`, and the episode's final observation; R2D2's
    #: item replay stores its (h, c) every 40 decisions, and the same. None
    #: plays the episode without storing it.
    replay: DreamerReplay | R2D2Replay | None = None
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
        """Play one episode to its classified end and emit it to replay."""
        state = self.environment.reset()
        carried_state = self.policy.initial_state()
        steps: list[Decision] = []
        latents: list[tuple[Any, Any]] = []
        #: R2D2's (h, c) before the decisions at 0, 40, 80, ...
        states: list[numpy.ndarray] = []
        final: StateFeatures | None = None
        storing_latents = isinstance(self.replay, DreamerReplay)
        recurrent = isinstance(self.policy, R2D2Backbone)
        streaming = storing_latents or isinstance(self.replay, R2D2Replay)
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
                final = features
                break
            if self.before_decision is not None:
                self.before_decision()
            with self.profile.span(POLICY_FORWARD):
                if recurrent and len(steps) % R2D2_SEQUENCE_PERIOD == 0:
                    states.append(self.policy.replay_entry(carried_state))  # type: ignore[attr-defined]
                action_index, carried_state = self.policy.act(
                    features, carried_state, epsilon=self.config.epsilon
                )
                if storing_latents:
                    latents.append(self.policy.replay_entry(carried_state))  # type: ignore[attr-defined]
            transition = self.environment.step(action_at(action_index))
            if recurrent:
                carried_state = self.policy.after_transition(  # type: ignore[attr-defined]
                    carried_state, transition.game_ms
                )
            total_reward += transition.reward
            steps.append(
                Decision(
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
                if streaming and transition.next_state is not None:
                    with self.profile.span(OBSERVATION_DECODE):
                        final = encode_state(transition.next_state)
                break
            if transition.next_state is not None:
                state = transition.next_state

        summary = self.environment.summarize(termination)
        if isinstance(self.replay, DreamerReplay):
            offered, accepted = self._emit_stream(self.replay, steps, latents, final, summary)
        elif isinstance(self.replay, R2D2Replay):
            offered, accepted = self._emit_items(self.replay, steps, states, final, summary)
        else:
            offered = accepted = 0
        return EpisodeResult(
            summary,
            offered,
            accepted,
            total_reward,
            wait_decisions=sum(1 for step in steps if step.action_index == WAIT_ACTION_INDEX),
        )

    def _metadata(self, summary: EpisodeSummary) -> SequenceMetadata:
        return SequenceMetadata(
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

    def _emit_stream(
        self,
        replay: DreamerReplay,
        steps: list[Decision],
        latents: list[tuple[Any, Any]],
        final: StateFeatures | None,
        summary: EpisodeSummary,
    ) -> tuple[int, int]:
        """Append the episode to this actor's stream in Dreamer's layout.

        Step t is observation t with the action taken at it and the reward,
        termination and game time of the transition into it. The last step is
        the final observation - the terminal one when the run died - with no
        action. The stream stops at the first inadmissible transition: its
        own observation was the admissible one before it led to, and is the
        episode's last; what it led to may not be valid (`DreamerReplay`).
        The final observation's latent is never read: the step after it
        starts an episode and resets the model, so zeros stand in for it
        until the learner writes its posterior back.
        """
        if not steps:
            # The run was already over when the episode opened.
            return 0, 0
        observations = _kept_observations(replay, steps, final)
        if not observations:
            return 1, 0
        count = len(observations)
        into: list[Decision | None] = [None, *steps[: count - 1]]
        blank = (numpy.zeros_like(latents[0][0]), numpy.zeros_like(latents[0][1]))
        entries = [*latents[:count], *[blank] * (count - len(latents))]
        episode = episode_steps(
            scalars=[o.scalars for o in observations],
            rows=[o.rows for o in observations],
            mask=[o.mask for o in observations],
            action=[steps[t].action_index if t < count - 1 else 0 for t in range(count)],
            reward=[0.0 if s is None else s.reward for s in into],
            terminal=[False if s is None else s.done for s in into],
            game_ms=[0.0 if s is None else s.game_ms for s in into],
            deter=[deter for deter, _ in entries],
            stoch=[stoch for _, stoch in entries],
        )
        with self.profile.acquiring(replay.lock):
            accepted = replay.add(self.config.actor_id, self._metadata(summary), episode)
        return 1, int(accepted)

    def _emit_items(
        self,
        replay: R2D2Replay,
        steps: list[Decision],
        states: list[numpy.ndarray],
        final: StateFeatures | None,
        summary: EpisodeSummary,
    ) -> tuple[int, int]:
        """Insert the episode's R2D2 items whole, as it ends (ADR 0014, ADR 0017).

        The steps are `_emit_stream`'s, in its layout and cut at the same
        place, and each item keeps the (h, c) the policy held before its first
        decision. A stream of one step is one item of that step, as Acme's
        TRUNCATE writes an episode of one step (structured.py 345-358). Returns
        the items offered and inserted: the replay's unit, and what the learner
        is credited for (`TrainingRun`).
        """
        if not steps:
            # The run was already over when the episode opened.
            return 0, 0
        observations = _kept_observations(replay, steps, final)
        count = len(observations)
        if not count:
            # The first transition was inadmissible: nothing survived the cut.
            return 1, 0
        into: list[Decision | None] = [None, *steps[: count - 1]]
        episode = StepArrays(
            scalars=numpy.asarray([o.scalars for o in observations], numpy.float32),
            rows=numpy.asarray([o.rows for o in observations], numpy.float32).reshape(count, -1),
            mask=numpy.asarray([o.mask for o in observations], numpy.bool_),
            action=numpy.asarray(
                [steps[t].action_index if t < count - 1 else 0 for t in range(count)], numpy.int64
            ),
            reward=numpy.asarray([0.0 if s is None else s.reward for s in into], numpy.float32),
            terminal=numpy.asarray([s is not None and s.done for s in into], numpy.bool_),
            game_ms=numpy.asarray([0.0 if s is None else s.game_ms for s in into], numpy.float32),
        )
        grid = numpy.stack(states[: state_count(count)])
        items = len(item_layout(count)[0])
        with self.profile.acquiring(replay.lock):
            accepted = replay.add(self._metadata(summary), episode, grid)
        return items, items if accepted else 0


def _kept_observations(
    replay: DreamerReplay | R2D2Replay, steps: list[Decision], final: StateFeatures | None
) -> list[StateFeatures]:
    """An episode's observations as a stream replay keeps them.

    Every observation, and the final one when the environment gave it; or,
    at the first inadmissible transition, up to observation `kept`: the
    admissible transition `kept - 1`'s valid next state, and the stream's
    last. With no transition before it there is nothing to keep.
    """
    kept = next((i for i, step in enumerate(steps) if not step.admissible), None)
    if kept is None:
        observations = [step.features for step in steps]
        if final is not None:
            observations.append(final)
        return observations
    replay.stats.reject("inadmissible_transition")
    return [step.features for step in steps[: kept + 1]] if kept else []
