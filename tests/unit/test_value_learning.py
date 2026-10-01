from __future__ import annotations

import math

import pytest
import torch

from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.value_learning import (
    game_time_discounts,
    survival_rewards,
    value_fit_correlation,
)

ACTIONS = len(RUN_ACTIONS)
DISCOUNT = 0.997


def constant(rewards: torch.Tensor, discount: float) -> torch.Tensor:
    """The same d for every transition: the per-decision discount."""
    return torch.full_like(rewards, discount, dtype=torch.float64)


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

    fit = value_fit_correlation(
        q, mask, reward_tensor, dones, real, discounts=constant(reward_tensor, discount)
    )

    assert fit == pytest.approx(1.0, abs=1e-5)


def test_value_fit_is_negative_when_the_values_run_the_other_way() -> None:
    rewards = [1.0, 0.0, 2.0, 1.0]
    q, mask, reward_tensor, dones, real = _completed_episode([0.0, 1.0, 2.0, 3.0], rewards)

    fit = value_fit_correlation(
        q, mask, reward_tensor, dones, real, discounts=constant(reward_tensor, 0.99)
    )

    assert fit is not None and fit < 0.0


def test_value_fit_is_withheld_where_the_return_was_never_realised() -> None:
    """A truncated return is a different quantity, not a noisier one."""
    rewards = torch.ones(1, 4)
    dones = torch.zeros(1, 4, dtype=torch.bool)
    q = torch.zeros(1, 4, ACTIONS)
    mask = torch.ones(1, 4, ACTIONS, dtype=torch.bool)

    assert (
        value_fit_correlation(
            q, mask, rewards, dones, torch.ones(1, 4), discounts=constant(rewards, 0.99)
        )
        is None
    )


def test_value_fit_is_withheld_rather_than_claimed_for_a_constant_prediction() -> None:
    rewards = [1.0, 0.0, 2.0, 1.0]
    q, mask, reward_tensor, dones, real = _completed_episode([3.0] * 4, rewards)

    assert (
        value_fit_correlation(
            q, mask, reward_tensor, dones, real, discounts=constant(reward_tensor, 0.99)
        )
        is None
    )


def test_padding_is_not_a_prediction_the_value_fit_is_scored_on() -> None:
    rewards = [0.0, 0.0, 2.0, 1.0]
    q, mask, reward_tensor, dones, real = _completed_episode([9.0, 9.0, 2.0, 1.0], rewards)
    real[0, :2] = 0.0

    fit = value_fit_correlation(
        q, mask, reward_tensor, dones, real, discounts=constant(reward_tensor, 0.99)
    )

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


# -- discounting by game time and the survival reward (ADR 0013) -------------

GAMMA_S = 0.997
BETA = -math.log(GAMMA_S)


def test_a_purchase_takes_no_game_time_and_costs_no_discount() -> None:
    discounts = game_time_discounts(GAMMA_S, torch.zeros(1, 3))

    assert torch.equal(discounts, torch.ones(1, 3, dtype=torch.float64))
    assert torch.equal(survival_rewards(discounts), torch.zeros(1, 3, dtype=torch.float64))


def test_a_span_is_discounted_by_the_game_seconds_it_took() -> None:
    discounts = game_time_discounts(GAMMA_S, torch.tensor([[2000.0, 0.0, 5000.0]]))

    assert discounts[0].tolist() == pytest.approx([GAMMA_S**2, 1.0, GAMMA_S**5])


@pytest.mark.parametrize("cut", [[35.0], [10.0, 0.0, 0.0, 25.0], [5.0, 5.0, 5.0, 20.0]])
def test_survival_reward_depends_on_the_time_survived_not_how_it_was_cut(
    cut: list[float],
) -> None:
    """A 35 s window earns (1 - g^35)/(beta*35) however split, each span valued at its start."""
    discounts = game_time_discounts(GAMMA_S, torch.tensor([cut]) * 1000.0)
    rewards = survival_rewards(discounts)[0].to(torch.float64)
    # The spans' rewards, each valued at its own start: the return from the window's first step.
    starts = torch.cumprod(torch.cat((torch.ones(1, dtype=torch.float64), discounts[0, :-1])), 0)

    assert (rewards * starts).sum().item() == pytest.approx(
        (1 - GAMMA_S**35) / (BETA * 35), rel=1e-6
    )
