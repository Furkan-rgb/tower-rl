from __future__ import annotations

import pytest
import torch

from tower_rl.application.replay import ReplaySequence, ReplayStep, SequenceMetadata
from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.backbone import collate, parameters_are_equal
from tower_rl.learning.network import NetworkConfig, StackedPolicyNetwork
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig

ACTIONS = len(RUN_ACTIONS)
SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)


def _features(*, valid: tuple[int, ...] = (0, 1, 2), seed: float = 0.5) -> StateFeatures:
    mask = tuple(index in valid for index in range(ACTIONS))
    return StateFeatures(
        scalars=tuple([seed] * SCALAR_COUNT),
        rows=tuple([seed] * (ROW_COUNT * ROW_WIDTH)),
        mask=mask,
    )


def _sequence(*, length: int = 8, burn_in: int = 4) -> ReplaySequence:
    steps = tuple(
        ReplayStep(
            features=_features(seed=0.1 * index),
            action_index=index % 3,
            reward=1.0,
            done=index == length - 1,
            admissible=True,
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
        burn_in,
    )


def _backbone(**overrides: object) -> StackedDqnBackbone:
    settings: dict[str, object] = {"seed": 0, "history_length": 4}
    settings.update(overrides)
    return StackedDqnBackbone(
        config=StackedDqnConfig(**settings),  # type: ignore[arg-type]
        network_config=SMALL,
    )


def test_the_window_carries_the_previous_steps_not_the_current_one() -> None:
    network = StackedPolicyNetwork(SMALL, history_length=3)
    scalars = torch.arange(12, dtype=torch.float32).view(1, 3, SCALAR_COUNT)
    state = network.initial_state(1)

    stacked = network.stack(scalars, state)

    assert stacked.shape == (1, 3, SCALAR_COUNT * 3)
    # The first step has no history at all, so its window is padded with zeros
    # and ends with the step itself.
    assert torch.equal(stacked[0, 0, -SCALAR_COUNT:], scalars[0, 0])
    assert torch.equal(stacked[0, 0, :-SCALAR_COUNT], torch.zeros(SCALAR_COUNT * 2))
    # By the third step the window is full and ordered oldest to newest.
    assert torch.equal(stacked[0, 2], scalars[0].flatten())


def test_carrying_state_across_calls_matches_one_long_call() -> None:
    """Acting step by step must see the same window as a batched forward pass."""
    network = StackedPolicyNetwork(SMALL, history_length=3)
    scalars = torch.randn(1, 4, SCALAR_COUNT)
    rows = torch.randn(1, 4, ROW_COUNT, ROW_WIDTH)
    mask = torch.ones(1, 4, ACTIONS, dtype=torch.bool)

    whole, _ = network(scalars, rows, mask)

    state = network.initial_state(1)
    stepwise = []
    for index in range(4):
        q, state = network(
            scalars[:, index : index + 1], rows[:, index : index + 1],
            mask[:, index : index + 1], state,
        )
        stepwise.append(q)

    assert torch.allclose(whole, torch.cat(stepwise, dim=1), atol=1e-6)


def test_history_of_one_is_the_no_history_ablation() -> None:
    network = StackedPolicyNetwork(SMALL, history_length=1)
    scalars = torch.randn(2, 3, SCALAR_COUNT)
    state = network.initial_state(2)

    assert torch.equal(network.stack(scalars, state), scalars)
    assert network.initial_state(2).shape == (2, 0, SCALAR_COUNT)


def test_learning_runs_and_only_the_window_contributes_errors() -> None:
    backbone = _backbone()
    batch = collate((_sequence(length=8, burn_in=4),), (1.0,))

    metrics = backbone.learn(batch)

    assert len(metrics.td_errors[0]) == 4, "eight steps minus four burn-in"
    assert metrics.weighted_loss >= 0.0
    assert backbone.model_version == 1


def test_a_burn_in_too_short_to_fill_the_window_is_refused() -> None:
    """Silently padding here would train on history acting never sees."""
    backbone = _backbone(history_length=6)
    batch = collate((_sequence(length=8, burn_in=2),), (1.0,))

    with pytest.raises(ValueError, match="cannot fill a window"):
        backbone.learn(batch)


def test_the_target_follows_the_online_network_without_ever_matching_it() -> None:
    backbone = _backbone()
    before = [parameter.clone() for parameter in backbone.target.parameters()]

    backbone.learn(collate((_sequence(),), (1.0,)))

    assert not parameters_are_equal(backbone.online, backbone.target), "EMA, not a copy"
    moved = any(
        not torch.equal(old, new)
        for old, new in zip(before, backbone.target.parameters(), strict=True)
    )
    assert moved, "an EMA target must move on every step"


def test_acting_respects_the_mask_and_threads_its_window() -> None:
    backbone = _backbone()
    state = backbone.initial_state()

    action, state = backbone.act(_features(valid=(0, 5)), state, epsilon=0.0)

    assert action in (0, 5)
    assert state.shape == (1, 3, SCALAR_COUNT)


def test_state_round_trips_exactly() -> None:
    backbone = _backbone()
    backbone.learn(collate((_sequence(),), (1.0,)))
    saved = backbone.state_dict()

    restored = _backbone()
    restored.load_state_dict(saved)

    assert restored.model_version == backbone.model_version
    assert parameters_are_equal(restored.online, backbone.online)
    assert parameters_are_equal(restored.target, backbone.target)
