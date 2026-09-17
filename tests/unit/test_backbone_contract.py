"""What every backbone must do identically, whatever algorithm it runs.

The benchmark's premise is that several algorithms are addressed through one
interface, over one environment, one action space and one replay. Anything a
backbone is free to vary lives inside `learn`; everything asserted here is the
part that must not vary, or the comparison stops being a comparison.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from tower_rl.application.replay import ReplaySequence, ReplayStep, SequenceMetadata
from tower_rl.domain.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.domain.run_actions import RUN_ACTIONS
from tower_rl.learning.backbone import Backbone, collate
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.recurrent_q import RecurrentQBackbone, RecurrentQConfig
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig

ACTIONS = len(RUN_ACTIONS)
SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)
BURN_IN = 2

#: Every backbone under comparison. A new one is added here and must pass
#: unchanged; if it cannot, it is not comparable to the others.
BACKBONES: dict[str, Callable[[], Backbone]] = {
    "recurrent-q": lambda: RecurrentQBackbone(
        config=RecurrentQConfig(seed=0), network_config=SMALL
    ),
    "stacked-dqn": lambda: StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=BURN_IN + 1), network_config=SMALL
    ),
}


@pytest.fixture(params=sorted(BACKBONES), name="backbone")
def _backbone(request: pytest.FixtureRequest) -> Backbone:
    return BACKBONES[request.param]()


def _features(*, valid: tuple[int, ...] = (0, 1, 2), seed: float = 0.5) -> StateFeatures:
    return StateFeatures(
        scalars=tuple([seed] * SCALAR_COUNT),
        rows=tuple([seed] * (ROW_COUNT * ROW_WIDTH)),
        mask=tuple(index in valid for index in range(ACTIONS)),
    )


def _zero_recurrent_state() -> tuple[torch.Tensor, torch.Tensor]:
    """An explicit zero state, valid across backbones: the stacked arm ignores it."""
    full = torch.zeros(1, 1, SMALL.core_hidden)
    return full, full.clone()


def _sequence(
    *, length: int = 8, padding: int = 0, padded_reward: float = 0.0
) -> ReplaySequence:
    steps = tuple(
        ReplayStep(
            features=_features(seed=0.1 * index),
            action_index=index % 3,
            reward=padded_reward if index < padding else 1.0,
            done=index == length - 1,
            admissible=True,
            padding=index < padding,
        )
        for index in range(length)
    )
    return ReplaySequence(
        SequenceMetadata(
            episode_id="e", actor_id="a", profile_id="p",
            observation_schema="observation-v1", action_schema="run-action-v1",
            reward_schema="reward-v1", model_version=0, epsilon=0.0, game_speed=8.0,
        ),
        steps,
        BURN_IN,
        recurrent_state=_zero_recurrent_state(),
    )


def test_greedy_acting_stays_inside_the_mask(backbone: Backbone) -> None:
    state = backbone.initial_state()

    for _ in range(5):
        action, state = backbone.act(_features(valid=(0, 7, 33)), state, epsilon=0.0)
        assert action in (0, 7, 33)


def test_exploring_acting_stays_inside_the_mask(backbone: Backbone) -> None:
    """Exploration draws from the valid set, never from the whole space."""
    state = backbone.initial_state()

    for _ in range(50):
        action, state = backbone.act(_features(valid=(0, 41)), state, epsilon=1.0)
        assert action in (0, 41)


def test_a_state_with_no_valid_action_is_refused_rather_than_guessed(
    backbone: Backbone,
) -> None:
    with pytest.raises(ValueError, match="no action"):
        backbone.act(_features(valid=()), backbone.initial_state(), epsilon=0.0)


def test_learning_advances_the_model_version_and_reports_its_errors(
    backbone: Backbone,
) -> None:
    batch = collate((_sequence(), _sequence()), (1.0, 1.0))
    before = backbone.model_version

    metrics = backbone.learn(batch)

    assert backbone.model_version == before + 1
    assert metrics.loss >= 0.0
    # One TD error per sequence per learnable step: replay prioritises on these.
    assert len(metrics.td_errors) == 2
    assert all(len(row) == 8 - BURN_IN for row in metrics.td_errors)


def test_padding_is_neither_trained_on_nor_prioritised(
    backbone: Backbone, request: pytest.FixtureRequest
) -> None:
    """A short episode is padded to fill a window; the filler must do nothing.

    Padded steps carry no experience, so they may not enter the loss and may not
    contribute a TD error - a zero there would drag the mean term of the
    sequence's priority down and make an early death look duller than it is.
    """
    other = BACKBONES[request.node.callspec.params["backbone"]]()

    quiet = backbone.learn(collate((_sequence(padding=3),), (1.0,)))
    loud = other.learn(collate((_sequence(padding=3, padded_reward=999.0),), (1.0,)))

    assert quiet.loss == loud.loss, "padding cannot move the loss"
    real_learning_steps = 8 - max(BURN_IN, 3)
    assert len(quiet.td_errors[0]) == real_learning_steps


def test_burn_in_never_contributes_to_the_loss(backbone: Backbone) -> None:
    metrics = backbone.learn(collate((_sequence(length=8),), (1.0,)))

    assert len(metrics.td_errors[0]) == 8 - BURN_IN


def test_state_round_trips_exactly(backbone: Backbone, request: pytest.FixtureRequest) -> None:
    backbone.learn(collate((_sequence(),), (1.0,)))
    restored = BACKBONES[request.node.callspec.params["backbone"]]()

    restored.load_state_dict(backbone.state_dict())

    assert restored.model_version == backbone.model_version
    features = _features(valid=(0, 5, 9))
    assert (
        restored.act(features, restored.initial_state(), epsilon=0.0)[0]
        == backbone.act(features, backbone.initial_state(), epsilon=0.0)[0]
    )


def test_every_backbone_states_where_its_parameters_live(backbone: Backbone) -> None:
    assert isinstance(backbone.device, torch.device)
