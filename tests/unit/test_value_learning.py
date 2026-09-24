from __future__ import annotations

import pytest
import torch

from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.value_learning import n_step_targets, value_fit_correlation

ACTIONS = len(RUN_ACTIONS)
DISCOUNT = 0.997


def test_a_positive_reward_stream_produces_positive_targets() -> None:
    rewards = torch.ones(1, 4)
    dones = torch.zeros(1, 4, dtype=torch.bool)
    q = torch.zeros(1, 4, ACTIONS)
    mask = torch.ones(1, 4, ACTIONS, dtype=torch.bool)

    targets, learnable = n_step_targets(
        rewards, dones, q, q, mask, discount=DISCOUNT, n_step=2
    )

    # Two steps of reward 1 at discount 0.997 for the steps that can bootstrap.
    assert targets[0, 0] == pytest.approx(1.0 + 0.997)
    assert learnable[0, 0] == 1.0
    # The final steps have no bootstrap state and did not terminate, so they are
    # excluded rather than trained on a truncated return.
    assert learnable[0, 3] == 0.0


def test_termination_stops_the_return_per_sequence_not_per_batch() -> None:
    rewards = torch.ones(2, 4)
    dones = torch.tensor([[False, True, False, False], [False, False, False, False]])
    q = torch.full((2, 4, ACTIONS), 5.0)
    mask = torch.ones(2, 4, ACTIONS, dtype=torch.bool)

    targets, learnable = n_step_targets(
        rewards, dones, q, q, mask, discount=DISCOUNT, n_step=3
    )

    # The first sequence ends at index 1, so its return is two rewards and no
    # bootstrap; the second keeps accumulating and bootstraps.
    assert targets[0, 0] == pytest.approx(1.0 + 0.997)
    assert targets[1, 0] > targets[0, 0]
    assert learnable[0, 0] == 1.0


def _completed_episode(
    values: list[float], rewards: list[float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One sequence that ends inside the window, with the values it predicted."""
    length = len(values)
    q = torch.zeros(1, length, ACTIONS)
    # The value of a state is the best masked action, so put it there.
    q[0, :, 0] = torch.tensor(values)
    mask = torch.zeros(1, length, ACTIONS, dtype=torch.bool)
    mask[0, :, 0] = True
    dones = torch.zeros(1, length, dtype=torch.bool)
    dones[0, -1] = True
    return q, mask, torch.tensor([rewards]), dones, torch.ones(1, length)


def test_value_fit_is_one_when_the_values_are_the_realised_return() -> None:
    """The falsifier a flat curve needs: a learner whose values track the return."""
    discount = 0.99
    rewards = [1.0, 0.0, 2.0, 1.0]
    realised = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + discount * running
        realised.append(running)
    realised.reverse()
    # The last step ends the episode, so its return is its reward alone.
    realised[-1] = rewards[-1]
    q, mask, reward_tensor, dones, real = _completed_episode(realised, rewards)

    fit = value_fit_correlation(q, mask, reward_tensor, dones, real, discount=discount)

    assert fit == pytest.approx(1.0, abs=1e-5)


def test_value_fit_is_negative_when_the_values_run_the_other_way() -> None:
    rewards = [1.0, 0.0, 2.0, 1.0]
    q, mask, reward_tensor, dones, real = _completed_episode([0.0, 1.0, 2.0, 3.0], rewards)

    fit = value_fit_correlation(q, mask, reward_tensor, dones, real, discount=0.99)

    assert fit is not None and fit < 0.0


def test_value_fit_is_withheld_where_the_return_was_never_realised() -> None:
    """A truncated return is a different quantity, not a noisier one."""
    rewards = torch.ones(1, 4)
    dones = torch.zeros(1, 4, dtype=torch.bool)
    q = torch.zeros(1, 4, ACTIONS)
    mask = torch.ones(1, 4, ACTIONS, dtype=torch.bool)

    assert (
        value_fit_correlation(q, mask, rewards, dones, torch.ones(1, 4), discount=0.99)
        is None
    )


def test_value_fit_is_withheld_rather_than_claimed_for_a_constant_prediction() -> None:
    rewards = [1.0, 0.0, 2.0, 1.0]
    q, mask, reward_tensor, dones, real = _completed_episode([3.0] * 4, rewards)

    assert (
        value_fit_correlation(q, mask, reward_tensor, dones, real, discount=0.99) is None
    )


def test_padding_is_not_a_prediction_the_value_fit_is_scored_on() -> None:
    rewards = [0.0, 0.0, 2.0, 1.0]
    q, mask, reward_tensor, dones, real = _completed_episode([9.0, 9.0, 2.0, 1.0], rewards)
    real[0, :2] = 0.0

    fit = value_fit_correlation(q, mask, reward_tensor, dones, real, discount=0.99)

    # Only the two real steps are correlated, and they agree exactly.
    assert fit == pytest.approx(1.0, abs=1e-5)


def _handcrafted(
    dones_at: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Twelve steps rewarding 1, 2, 3, ..., with a flat bootstrap value of 100."""
    time = 12
    rewards = torch.arange(1, time + 1, dtype=torch.float32).view(1, time)
    dones = torch.zeros(1, time, dtype=torch.bool)
    for index in dones_at:
        dones[0, index] = True
    q = torch.full((1, time, ACTIONS), 100.0)
    mask = torch.ones(1, time, ACTIONS, dtype=torch.bool)
    return rewards, dones, q, mask


def _expected(start: int, n: int, dones_at: tuple[int, ...]) -> float:
    """The n-step return written out by hand: stop at an end, else bootstrap."""
    total = 0.0
    for offset in range(n):
        index = start + offset
        total += DISCOUNT**offset * (index + 1)
        if index in dones_at:
            return total
    return total + DISCOUNT**n * 100.0


@pytest.mark.parametrize("n", [3, 10])
@pytest.mark.parametrize("dones_at", [(), (5,)])
def test_n_step_targets_for_the_two_ends_of_the_anneal(
    n: int, dones_at: tuple[int, ...]
) -> None:
    """The anneal's n = 10 and n = 3, with and without an episode end inside."""
    rewards, dones, q, mask = _handcrafted(dones_at)

    targets, learnable = n_step_targets(
        rewards, dones, q, q, mask, discount=DISCOUNT, n_step=n
    )

    for step in range(12):
        bootstraps = step + n < 12
        ends_inside = any(step <= end < step + n for end in dones_at)
        assert learnable[0, step] == float(bootstraps or ends_inside), step
        if bootstraps or ends_inside:
            assert targets[0, step] == pytest.approx(_expected(step, n, dones_at)), step


def test_the_same_sequence_can_be_targeted_at_a_different_n_on_each_call() -> None:
    """Replay stores raw steps, so an annealed n is just a different argument."""
    rewards, dones, q, mask = _handcrafted((5,))

    long, _ = n_step_targets(rewards, dones, q, q, mask, discount=DISCOUNT, n_step=10)
    short, _ = n_step_targets(rewards, dones, q, q, mask, discount=DISCOUNT, n_step=3)
    again, _ = n_step_targets(rewards, dones, q, q, mask, discount=DISCOUNT, n_step=10)

    assert torch.equal(long, again)
    assert short[0, 0] == pytest.approx(_expected(0, 3, (5,)))
    assert long[0, 0] == pytest.approx(_expected(0, 10, (5,)))
