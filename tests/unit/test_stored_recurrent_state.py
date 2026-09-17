"""Burn-in starts from the state the actor stored, not from zeros.

R2D2 section 2.3 measures three ways of initialising the recurrent state of a
replayed sequence and finds stored state plus burn-in materially better than zero
state plus burn-in. The recurrent arm used to do the latter - the variant its own
source paper argues against - which would have handicapped exactly the arm the
benchmark is about. These are the regression tests for that, at both ends: the
actor stores the state, and the learner burns in from it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes.fake_run_port import FakeRunPort  # noqa: E402

from tower_rl.application.actor import Actor, ActorConfig  # noqa: E402
from tower_rl.application.replay import (  # noqa: E402
    PrioritizedSequenceReplay,
    ReplaySequence,
    ReplayStep,
    SequenceMetadata,
)
from tower_rl.application.run_environment import (  # noqa: E402
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.domain.features import (  # noqa: E402
    ROW_COUNT,
    ROW_WIDTH,
    SCALAR_COUNT,
    StateFeatures,
)
from tower_rl.domain.run_actions import RUN_ACTIONS  # noqa: E402
from tower_rl.domain.run_state import RunStateBuilder  # noqa: E402
from tower_rl.learning.backbone import collate  # noqa: E402
from tower_rl.learning.network import NetworkConfig  # noqa: E402
from tower_rl.learning.recurrent_q import RecurrentQBackbone, RecurrentQConfig  # noqa: E402
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig  # noqa: E402

PROFILE = "fake-profile-v1"
SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)
LENGTH, BURN_IN, STRIDE = 4, 2, 2


@dataclass
class CountingPolicy:
    """Carries the number of steps taken so far as its whole recurrent state.

    An integer is enough: what is under test is *which* carried state each window
    is stored with, and an integer names the step it was entered with.
    """

    steps_taken: int = field(default=0, init=False)

    def initial_state(self) -> int:
        return 0

    def stored_recurrent_state(self, state: int) -> int:
        return state

    def act(self, features: StateFeatures, state: int, *, epsilon: float) -> tuple[int, int]:
        self.steps_taken += 1
        return 0, state + 1


def _environment(**port_kwargs: object) -> InstrumentedRunEnvironment:
    return InstrumentedRunEnvironment(
        port=FakeRunPort(**port_kwargs),  # type: ignore[arg-type]
        builder=RunStateBuilder(profile_id=PROFILE),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )


def _played(policy: object, *, sequence_length: int) -> list[ReplaySequence]:
    replay = PrioritizedSequenceReplay(capacity=256, seed=0)
    actor = Actor(
        environment=_environment(damage_per_second=1.0),
        policy=policy,  # type: ignore[arg-type]
        config=ActorConfig(
            sequence_length=sequence_length, burn_in=BURN_IN, stride=STRIDE
        ),
        replay=replay,
    )
    actor.run_episode()
    return list(replay._items)


def test_an_actor_stores_the_state_the_policy_held_at_each_window_start() -> None:
    policy = CountingPolicy()
    stored = _played(policy, sequence_length=LENGTH)

    assert stored, "the episode must have produced windows"
    for position, sequence in enumerate(stored[:-1]):
        # Windows before the end-aligned one start on the stride.
        assert sequence.recurrent_state == position * STRIDE
    assert stored[-1].recurrent_state == policy.steps_taken - LENGTH


def test_a_left_padded_window_stores_the_state_the_episode_began_from() -> None:
    """Padding stands in for before the episode began; so does the initial state."""
    policy = CountingPolicy()
    # Longer than the episode the fake run produces, so one padded window is emitted.
    (sequence,) = _played(policy, sequence_length=512)

    assert sequence.steps[0].padding
    assert sequence.recurrent_state == CountingPolicy().initial_state()


# -- the learner's end -----------------------------------------------------


def _features(seed: float) -> StateFeatures:
    return StateFeatures(
        scalars=(seed,) * SCALAR_COUNT,
        rows=(seed,) * (ROW_COUNT * ROW_WIDTH),
        mask=tuple(index < 3 for index in range(len(RUN_ACTIONS))),
    )


def _sequence(state: object) -> ReplaySequence:
    steps = tuple(
        ReplayStep(
            features=_features(0.1 * index),
            action_index=index % 3,
            reward=1.0,
            done=index == 7,
            admissible=True,
        )
        for index in range(8)
    )
    return ReplaySequence(
        SequenceMetadata(
            episode_id="e", actor_id="a", profile_id="p",
            observation_schema="observation-v1", action_schema="run-action-v1",
            reward_schema="reward-v1", model_version=0, epsilon=0.0, game_speed=8.0,
        ),
        steps,
        BURN_IN,
        recurrent_state=state,
    )


def _recurrent() -> RecurrentQBackbone:
    return RecurrentQBackbone(config=RecurrentQConfig(seed=0), network_config=SMALL)


def _state(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    full = torch.full((1, 1, SMALL.core_hidden), value)
    return full, full.clone()


def _loss(state: object) -> float:
    backbone = _recurrent()
    return backbone.learn(collate((_sequence(state),), (1.0,))).loss


def test_burn_in_starts_from_the_stored_state_rather_than_from_zeros() -> None:
    """A state far from zero must change what burn-in reconstructs, and so the loss.

    Two identically seeded backbones see the same sequence; only the stored state
    differs. If burn-in ignored it and started from zeros, as it used to, the two
    losses would be identical.
    """
    from_zeros = _loss(None)
    from_stored = _loss(_state(1.0))

    assert from_stored != pytest.approx(from_zeros)
    # The difference is the state, not nondeterminism: the same state twice gives
    # the same answer, and a zeroed stored state is the zero-state variant itself.
    assert _loss(_state(1.0)) == pytest.approx(from_stored)
    assert _loss(_state(0.0)) == pytest.approx(from_zeros)


def test_a_batch_may_not_mix_stored_and_missing_recurrent_states() -> None:
    with pytest.raises(ValueError, match="mix sequences"):
        collate((_sequence(_state(1.0)), _sequence(None)), (1.0, 1.0))


def test_a_stored_state_is_detached_and_on_the_cpu() -> None:
    backbone = _recurrent()
    hidden, cell = backbone.stored_recurrent_state(backbone.initial_state())

    assert not hidden.requires_grad and not cell.requires_grad
    assert hidden.device.type == "cpu" and cell.device.type == "cpu"


# -- the stacked arm -------------------------------------------------------


def test_the_stacked_backbone_stores_no_recurrent_state() -> None:
    """It has none: its history is the stored scalars, rebuilt from burn-in."""
    backbone = StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=BURN_IN + 1), network_config=SMALL
    )

    assert backbone.stored_recurrent_state(backbone.initial_state()) is None

    first = backbone.learn(collate((_sequence(None),), (1.0,))).loss
    again = StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=BURN_IN + 1), network_config=SMALL
    ).learn(collate((_sequence(None),), (1.0,))).loss
    assert first == pytest.approx(again)


def test_an_actor_driving_the_stacked_backbone_stores_nothing_per_window() -> None:
    backbone = StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=2), network_config=SMALL
    )
    stored = _played(backbone, sequence_length=LENGTH)

    assert stored
    assert all(sequence.recurrent_state is None for sequence in stored)
