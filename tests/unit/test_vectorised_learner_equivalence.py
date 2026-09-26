"""The vectorised learner step computes exactly what the step-by-step one did.

`n_step_targets` used to build every step's return in a Python loop over time
and offset, and `collate` used to build nested Python lists and one tensor per
field. Both were rewritten for speed alone (board item #72). The originals are
kept here, and only here, as the oracles the rewrites are held to: equal to the
bit on random inputs, so the change cannot have moved a target, a batch or a
gradient step.
"""

from __future__ import annotations

import random

import pytest
import torch

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.backbone import SequenceBatch, collate
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.replay import (
    PrioritizedSequenceReplay,
    ReplaySequence,
    ReplayStep,
    SequenceMetadata,
)
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig
from tower_rl.learning.value_learning import (
    evaluated_next_values,
    n_step_targets,
    value_fit_correlation,
)

ACTIONS = len(RUN_ACTIONS)
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
    ),
]


def _loop_n_step_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    online_q: torch.Tensor,
    target_q: torch.Tensor,
    mask: torch.Tensor,
    *,
    discount: float,
    n_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The step-by-step implementation `n_step_targets` replaced, verbatim."""
    batch, time = rewards.shape
    evaluated = evaluated_next_values(online_q, target_q, mask)

    targets = torch.zeros_like(rewards)
    learnable = torch.zeros_like(rewards)
    terminal = dones.to(rewards.dtype)
    for step in range(time):
        accumulated = torch.zeros(batch, device=rewards.device)
        alive = torch.ones(batch, device=rewards.device)
        factor = 1.0
        ended_inside_window = torch.zeros(batch, device=rewards.device)
        for offset in range(n_step):
            index = step + offset
            if index >= time:
                break
            accumulated = accumulated + alive * factor * rewards[:, index]
            factor *= discount
            alive = alive * (1.0 - terminal[:, index])
            ended_inside_window = torch.maximum(ended_inside_window, terminal[:, index])
        bootstrap = step + n_step
        if bootstrap < time:
            accumulated = accumulated + alive * factor * evaluated[:, bootstrap]
            learnable[:, step] = 1.0
        else:
            learnable[:, step] = ended_inside_window
        targets[:, step] = accumulated
    return targets, learnable


def _list_collate(
    sequences: tuple[ReplaySequence, ...],
    weights: tuple[float, ...],
    *,
    device: torch.device | None = None,
) -> SequenceBatch:
    """The list-building `collate` the packed one replaced, verbatim."""
    length = len(sequences[0].steps)
    burn_in = sequences[0].burn_in

    def _features(step_features: StateFeatures) -> tuple[list[float], list[float], list[bool]]:
        return list(step_features.scalars), list(step_features.rows), list(step_features.mask)

    scalars, rows, masks, actions, rewards, dones, padding, game_ms = (
        [], [], [], [], [], [], [], []
    )
    for sequence in sequences:
        collected = [_features(step.features) for step in sequence.steps]
        scalars.append([item[0] for item in collected])
        rows.append([item[1] for item in collected])
        masks.append([item[2] for item in collected])
        actions.append([step.action_index for step in sequence.steps])
        rewards.append([step.reward for step in sequence.steps])
        dones.append([step.done for step in sequence.steps])
        padding.append([step.padding for step in sequence.steps])
        game_ms.append([step.game_ms for step in sequence.steps])

    row_tensor = torch.tensor(rows, dtype=torch.float32, device=device)
    return SequenceBatch(
        scalars=torch.tensor(scalars, dtype=torch.float32, device=device),
        rows=row_tensor.view(len(sequences), length, ROW_COUNT, ROW_WIDTH),
        mask=torch.tensor(masks, dtype=torch.bool, device=device),
        actions=torch.tensor(actions, dtype=torch.int64, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        dones=torch.tensor(dones, dtype=torch.bool, device=device),
        padding=torch.tensor(padding, dtype=torch.bool, device=device),
        game_ms=torch.tensor(game_ms, dtype=torch.float32, device=device),
        weights=torch.tensor(weights, dtype=torch.float32, device=device),
        burn_in=burn_in,
    )


def _random_target_inputs(
    seed: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    """One random batch as `StackedDqnBackbone.learn` hands it to the target.

    Varies everything the target depends on: n below, at and past the window
    length, the discount, episode ends from none to most steps, front padding
    as the actor writes it (zero reward, never done), masked actions valued at
    minus infinity as the network values them, and states with no valid action.
    """
    generator = torch.Generator().manual_seed(seed)
    draw = random.Random(seed)
    batch = draw.randint(1, 6)
    time = draw.randint(1, 40)
    n_step = draw.randint(1, 15)
    discount = draw.uniform(0.5, 0.9999)

    rewards = torch.randn(batch, time, generator=generator) * draw.choice((0.1, 1.0, 50.0))
    rewards = torch.where(torch.rand(batch, time, generator=generator) < 0.3, 0.0, rewards)
    dones = torch.rand(batch, time, generator=generator) < draw.choice((0.0, 0.05, 0.3, 0.9))
    mask = torch.rand(batch, time, ACTIONS, generator=generator) < 0.3
    mask[..., 0] = torch.rand(batch, time, generator=generator) < 0.9
    online_q = torch.randn(batch, time, ACTIONS, generator=generator) * 10.0
    target_q = torch.randn(batch, time, ACTIONS, generator=generator) * 10.0
    online_q = online_q.masked_fill(~mask, float("-inf"))
    target_q = target_q.masked_fill(~mask, float("-inf"))

    padding = torch.zeros(batch, time, dtype=torch.bool)
    for row in range(batch):
        padding[row, : draw.randint(0, time - 1)] = True
    rewards = rewards.masked_fill(padding, 0.0)
    dones = dones & ~padding

    return (
        rewards.to(device),
        dones.to(device),
        online_q.to(device),
        target_q.to(device),
        mask.to(device),
        discount,
        n_step,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_vectorised_n_step_targets_equal_the_step_by_step_loop(device: str) -> None:
    for seed in range(300):
        rewards, dones, online_q, target_q, mask, discount, n_step = _random_target_inputs(
            seed, device
        )

        # The per-decision discount as `StackedDqnConfig` hands it over when
        # --discount-per-game-second is off: one constant d per transition.
        discounts = StackedDqnConfig(discount=discount).transition_discounts(
            torch.zeros_like(rewards)
        )
        targets, learnable = n_step_targets(
            rewards, dones, online_q, target_q, mask, discounts=discounts, n_step=n_step
        )
        expected_targets, expected_learnable = _loop_n_step_targets(
            rewards, dones, online_q, target_q, mask, discount=discount, n_step=n_step
        )

        assert targets.shape == expected_targets.shape, f"seed {seed}"
        assert targets.dtype == expected_targets.dtype, f"seed {seed}"
        torch.testing.assert_close(
            targets, expected_targets, rtol=0.0, atol=0.0, msg=f"seed {seed}"
        )
        torch.testing.assert_close(
            learnable, expected_learnable, rtol=0.0, atol=0.0, msg=f"seed {seed}"
        )


_METADATA = SequenceMetadata(
    episode_id="e", actor_id="a", profile_id="p",
    observation_schema="observation-v1", action_schema="run-action-v1",
    reward_schema="reward-v1", model_version=0, epsilon=0.0, game_speed=8.0,
)
LENGTH = 12
BURN_IN = 3


def _random_sequence(draw: random.Random) -> ReplaySequence:
    padded = draw.randint(0, LENGTH - BURN_IN - 1)
    end = draw.choice((None, LENGTH - 1, draw.randint(padded, LENGTH - 1)))
    steps = []
    for index in range(LENGTH):
        mask = tuple(action == 0 or draw.random() < 0.3 for action in range(ACTIONS))
        steps.append(
            ReplayStep(
                features=StateFeatures(
                    scalars=tuple(draw.gauss(0.0, 3.0) for _ in range(SCALAR_COUNT)),
                    rows=tuple(draw.random() for _ in range(ROW_COUNT * ROW_WIDTH)),
                    mask=mask,
                ),
                # Taken from the valid set, as an actor takes it.
                action_index=draw.choice([action for action, valid in enumerate(mask) if valid]),
                reward=0.0 if index < padded else draw.gauss(0.0, 2.0),
                done=index == end,
                admissible=True,
                game_ms=1000.0,
                padding=index < padded,
            )
        )
    return ReplaySequence(_METADATA, tuple(steps), BURN_IN)


def _seeded_replay(seed: int) -> PrioritizedSequenceReplay:
    draw = random.Random(seed)
    replay = PrioritizedSequenceReplay(capacity=24, seed=seed)
    for _ in range(30):
        replay.add(_random_sequence(draw))
    indices, _, _ = replay.sample(8)
    replay.update_priorities(
        indices, tuple((draw.uniform(0.0, 5.0),) for _ in indices)
    )
    return replay


def test_a_weight_count_that_differs_from_the_sequence_count_is_refused() -> None:
    """Packing would otherwise drop surplus weights without a word."""
    replay = _seeded_replay(0)
    _, sequences, weights = replay.sample(4)

    for wrong in (weights + (1.0,), weights[:-1]):
        with pytest.raises(ValueError, match="exactly one weight"):
            collate(sequences, wrong)


def _assert_same_batch(batch: SequenceBatch, expected: SequenceBatch) -> None:
    assert batch.burn_in == expected.burn_in
    for name in (
        "scalars", "rows", "mask", "actions", "rewards", "dones", "padding", "game_ms", "weights"
    ):
        value, reference = getattr(batch, name), getattr(expected, name)
        assert value.dtype == reference.dtype, name
        assert value.shape == reference.shape, name
        assert value.device == reference.device, name
        assert torch.equal(value, reference), name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(20))
def test_a_seeded_sample_collates_to_the_same_tensors_as_before(seed: int, device: str) -> None:
    """Same replay, same seed, same sample: the packed batch is the listed one."""
    replay, twin = _seeded_replay(seed), _seeded_replay(seed)

    indices, sequences, weights = replay.sample(8)
    twin_indices, twin_sequences, twin_weights = twin.sample(8)
    assert (indices, weights) == (twin_indices, twin_weights)

    _assert_same_batch(
        collate(sequences, weights, device=torch.device(device)),
        _list_collate(twin_sequences, twin_weights, device=torch.device(device)),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_steps_on_a_packed_batch_equal_those_on_a_listed_one(device: str) -> None:
    """End to end: metrics, priorities and parameters after three steps agree exactly."""
    small = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

    def backbone() -> StackedDqnBackbone:
        return StackedDqnBackbone(
            config=StackedDqnConfig(seed=0, history_length=BURN_IN + 1, n_step=4),
            network_config=small,
            device=torch.device(device),
        )

    packed, listed = backbone(), backbone()
    replay = _seeded_replay(7)
    for _ in range(3):
        _, sequences, weights = replay.sample(8)
        ours = packed.learn(collate(sequences, weights, device=packed.device))
        theirs = listed.learn(_list_collate(sequences, weights, device=listed.device))
        assert ours == theirs

    for name, value in packed.online.state_dict().items():
        assert torch.equal(value, listed.online.state_dict()[name]), name
    for name, value in packed.target.state_dict().items():
        assert torch.equal(value, listed.target.state_dict()[name]), name


# -- per-decision discount after board #81 ----------------------------------
#
# `n_step_targets` and `value_fit_correlation` took one scalar discount before
# the game-time discount made it a tensor of per-transition ds. Their scalar
# versions are frozen here as the oracle the flag-off path is held to, to the
# bit: with --discount-per-game-second off nothing about a target may move.


def _scalar_n_step_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    online_q: torch.Tensor,
    target_q: torch.Tensor,
    mask: torch.Tensor,
    *,
    discount: float,
    n_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`n_step_targets` with one scalar discount, as it was before #81, verbatim."""
    batch, time = rewards.shape
    evaluated = evaluated_next_values(online_q, target_q, mask)

    terminal = dones.to(rewards.dtype)
    beyond = rewards.new_zeros(batch, n_step)
    padded_rewards = torch.cat((rewards, beyond), dim=1)
    padded_terminal = torch.cat((terminal, beyond), dim=1)

    accumulated = torch.zeros_like(rewards)
    alive = torch.ones_like(rewards)
    ended_inside_window = torch.zeros_like(rewards)
    factor = 1.0
    for offset in range(n_step):
        window_rewards = padded_rewards[:, offset : offset + time]
        window_terminal = padded_terminal[:, offset : offset + time]
        accumulated = accumulated + alive * factor * window_rewards
        factor *= discount
        alive = alive * (1.0 - window_terminal)
        ended_inside_window = torch.maximum(ended_inside_window, window_terminal)

    bootstrapped = max(time - n_step, 0)
    head = accumulated[:, :bootstrapped] + (
        alive[:, :bootstrapped] * factor * evaluated[:, n_step:]
    )
    targets = torch.cat((head, accumulated[:, bootstrapped:]), dim=1)
    learnable = torch.cat(
        (torch.ones_like(head), ended_inside_window[:, bootstrapped:]), dim=1
    )
    return targets, learnable


def _scalar_value_fit_correlation(
    online_q: torch.Tensor,
    mask: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    real: torch.Tensor,
    *,
    discount: float,
) -> float | None:
    """`value_fit_correlation` with one scalar discount, as it was before #81, verbatim."""
    values = torch.where(mask, online_q, torch.full_like(online_q, float("-inf"))).amax(dim=-1)
    time = rewards.shape[1]
    terminal = dones.to(rewards.dtype)
    returns = torch.zeros_like(rewards)
    ends_inside = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], device=rewards.device)
    ended = torch.zeros_like(running)
    for step in reversed(range(time)):
        running = rewards[:, step] + discount * (1.0 - terminal[:, step]) * running
        ended = torch.maximum(terminal[:, step], ended)
        returns[:, step] = running
        ends_inside[:, step] = ended

    keep = (ends_inside > 0) & (real > 0) & torch.isfinite(values)
    predicted = values[keep]
    realised = returns[keep]
    if predicted.numel() < 2:
        return None
    predicted = predicted - predicted.mean()
    realised = realised - realised.mean()
    spread = predicted.norm() * realised.norm()
    if float(spread.item()) <= 0.0:
        return None
    return float((predicted @ realised / spread).item())


@pytest.mark.parametrize("device", DEVICES)
def test_with_the_game_time_discount_off_targets_and_value_fit_are_unchanged(
    device: str,
) -> None:
    """T1: flag off, every target and every value fit equals the scalar one, bit for bit.

    The random batches vary n, the discount, episode ends and front padding,
    so windows run past the end both with and without a terminal inside. The
    game time is random too: per decision it must be ignored entirely.
    """
    for seed in range(300):
        rewards, dones, online_q, target_q, mask, discount, n_step = _random_target_inputs(
            seed, device
        )
        config = StackedDqnConfig(discount=discount)
        game_ms = torch.rand(rewards.shape, generator=torch.Generator().manual_seed(seed))
        discounts = config.transition_discounts((game_ms * 20_000.0).to(device))
        assert not config.books_reward_at_span_end

        targets, learnable = n_step_targets(
            rewards, dones, online_q, target_q, mask, discounts=discounts, n_step=n_step
        )
        expected_targets, expected_learnable = _scalar_n_step_targets(
            rewards, dones, online_q, target_q, mask, discount=discount, n_step=n_step
        )
        assert targets.dtype == expected_targets.dtype, f"seed {seed}"
        torch.testing.assert_close(
            targets, expected_targets, rtol=0.0, atol=0.0, msg=f"seed {seed}"
        )
        torch.testing.assert_close(
            learnable, expected_learnable, rtol=0.0, atol=0.0, msg=f"seed {seed}"
        )

        real = (torch.rand(rewards.shape, generator=torch.Generator().manual_seed(seed)) < 0.9)
        real = real.to(rewards.dtype).to(device)
        fit = value_fit_correlation(
            online_q, mask, rewards, dones, real, discounts=discounts
        )
        expected_fit = _scalar_value_fit_correlation(
            online_q, mask, rewards, dones, real, discount=discount
        )
        assert fit == expected_fit, f"seed {seed}"
