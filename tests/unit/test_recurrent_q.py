from __future__ import annotations

import pytest
import torch

from tower_rl.application.replay import (
    PrioritizedSequenceReplay,
    ReplaySequence,
    ReplayStep,
    SequenceMetadata,
)
from tower_rl.domain.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.domain.run_actions import RUN_ACTIONS
from tower_rl.learning.backbone import collate
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.recurrent_q import (
    RecurrentQBackbone,
    RecurrentQConfig,
    parameters_are_equal,
)

ACTIONS = len(RUN_ACTIONS)
SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)


def _features(*, valid: tuple[int, ...] = (0, 1, 2), seed: float = 0.5) -> StateFeatures:
    mask = tuple(index in valid for index in range(ACTIONS))
    return StateFeatures(
        scalars=tuple([seed] * SCALAR_COUNT),
        rows=tuple([seed] * (ROW_COUNT * ROW_WIDTH)),
        mask=mask,
    )


def _metadata() -> SequenceMetadata:
    return SequenceMetadata(
        episode_id="e", actor_id="a", profile_id="p",
        observation_schema="observation-v1", action_schema="run-action-v1",
        reward_schema="reward-v1", model_version=0, epsilon=0.0, game_speed=8.0,
    )


def _sequence(length: int = 8, burn_in: int = 2, *, reward: float = 1.0) -> ReplaySequence:
    steps = tuple(
        ReplayStep(
            features=_features(seed=0.1 * index),
            action_index=index % 3,
            reward=reward,
            done=index == length - 1,
            admissible=True,
        )
        for index in range(length)
    )
    return ReplaySequence(_metadata(), steps, burn_in)


def _backbone(**config: object) -> RecurrentQBackbone:
    return RecurrentQBackbone(
        config=RecurrentQConfig(seed=0, **config),  # type: ignore[arg-type]
        network_config=SMALL,
    )


def test_acting_only_ever_returns_a_valid_action() -> None:
    backbone = _backbone()
    features = _features(valid=(0, 7, 33))
    state = backbone.initial_state()

    for epsilon in (0.0, 1.0):
        for _ in range(20):
            action, state = backbone.act(features, state, epsilon=epsilon)
            assert action in (0, 7, 33)


def test_acting_on_a_state_with_no_valid_action_is_refused() -> None:
    backbone = _backbone()
    features = _features(valid=())

    with pytest.raises(ValueError, match="no action is available"):
        backbone.act(features, backbone.initial_state(), epsilon=0.0)


def test_one_learning_step_changes_weights_and_reports_errors() -> None:
    backbone = _backbone()
    batch = collate((_sequence(), _sequence()), (1.0, 1.0))
    before = [parameter.clone() for parameter in backbone.online.parameters()]

    metrics = backbone.learn(batch)

    assert backbone.model_version == 1
    assert metrics.loss >= 0.0
    assert metrics.gradient_norm > 0.0
    assert len(metrics.td_errors) == 2
    after = list(backbone.online.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_td_errors_feed_replay_priorities_directly() -> None:
    backbone = _backbone()
    replay = PrioritizedSequenceReplay(capacity=4, seed=0)
    replay.add(_sequence())
    replay.add(_sequence())
    indices, sequences, weights = replay.sample(2)

    metrics = backbone.learn(collate(sequences, weights))
    replay.update_priorities(indices, metrics.td_errors)

    assert all(priority > 0 for priority in replay._priorities)


def test_the_target_network_only_moves_on_its_interval() -> None:
    backbone = _backbone(target_update_interval=3)
    batch = collate((_sequence(),), (1.0,))

    backbone.learn(batch)
    assert not parameters_are_equal(backbone.online, backbone.target)

    backbone.learn(batch)
    backbone.learn(batch)
    assert parameters_are_equal(backbone.online, backbone.target)


def test_burn_in_is_not_trained_on() -> None:
    """Only the learning window contributes TD errors."""
    backbone = _backbone()
    batch = collate((_sequence(length=8, burn_in=3),), (1.0,))

    metrics = backbone.learn(batch)

    assert len(metrics.td_errors[0]) == 5, "eight steps minus three burn-in"


def test_state_round_trips_exactly() -> None:
    backbone = _backbone()
    backbone.learn(collate((_sequence(),), (1.0,)))
    saved = backbone.state_dict()

    restored = _backbone()
    restored.load_state_dict(saved)

    assert restored.model_version == backbone.model_version
    assert parameters_are_equal(restored.online, backbone.online)
    assert parameters_are_equal(restored.target, backbone.target)
