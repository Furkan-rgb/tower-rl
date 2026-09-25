from __future__ import annotations

import pytest
import torch

from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.stacked_dqn import StackedDqnConfig
from tower_rl.learning.value_learning import n_step_targets, value_fit_correlation

ACTIONS = len(RUN_ACTIONS)
DISCOUNT = 0.997


def constant(rewards: torch.Tensor, discount: float) -> torch.Tensor:
    """The same d for every transition: the per-decision discount."""
    return torch.full_like(rewards, discount, dtype=torch.float64)


def test_a_positive_reward_stream_produces_positive_targets() -> None:
    rewards = torch.ones(1, 4)
    dones = torch.zeros(1, 4, dtype=torch.bool)
    q = torch.zeros(1, 4, ACTIONS)
    mask = torch.ones(1, 4, ACTIONS, dtype=torch.bool)

    targets, learnable = n_step_targets(
        rewards, dones, q, q, mask, discounts=constant(rewards, DISCOUNT), n_step=2
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
        rewards, dones, q, q, mask, discounts=constant(rewards, DISCOUNT), n_step=3
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


@pytest.mark.parametrize("n", [3, 10])
@pytest.mark.parametrize("dones_at", [(), (5,)])
def test_n_step_targets_for_the_two_ends_of_the_anneal(
    n: int, dones_at: tuple[int, ...]
) -> None:
    """The anneal's n = 10 and n = 3, with and without an episode end inside."""
    rewards, dones, q, mask = _handcrafted(dones_at)

    targets, learnable = n_step_targets(
        rewards, dones, q, q, mask, discounts=constant(rewards, DISCOUNT), n_step=n
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

    discounts = constant(rewards, DISCOUNT)
    long, _ = n_step_targets(rewards, dones, q, q, mask, discounts=discounts, n_step=10)
    short, _ = n_step_targets(rewards, dones, q, q, mask, discounts=discounts, n_step=3)
    again, _ = n_step_targets(rewards, dones, q, q, mask, discounts=discounts, n_step=10)

    assert torch.equal(long, again)
    assert short[0, 0] == pytest.approx(_expected(0, 3, (5,)))
    assert long[0, 0] == pytest.approx(_expected(0, 10, (5,)))


# -- discounting by game time (board #81) -----------------------------------
#
# Under --discount-per-game-second a transition's d is gamma_s ** seconds, the
# reward it carries is valued at its start as d * r (booked at the end of its
# span), and the target composes both. These are worked by hand.

GAMMA_S = 0.997
PER_SECOND = StackedDqnConfig(discount_per_game_second=GAMMA_S)
BOOTSTRAP = 10.0


def _timed(
    seconds: list[float], rewards: list[float], dones_at: tuple[int, ...] = ()
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One sequence under the game-time discount, as `learn` hands it over."""
    time = len(seconds)
    discounts = PER_SECOND.transition_discounts(torch.tensor([seconds]) * 1000.0)
    assert PER_SECOND.books_reward_at_span_end
    step_rewards = torch.tensor([rewards]) * discounts.float()
    dones = torch.zeros(1, time, dtype=torch.bool)
    for index in dones_at:
        dones[0, index] = True
    q = torch.full((1, time, ACTIONS), BOOTSTRAP)
    mask = torch.ones(1, time, ACTIONS, dtype=torch.bool)
    return step_rewards, dones, q, mask, discounts


def test_a_reward_is_booked_at_the_end_of_its_span_and_the_bootstrap_at_the_window_end() -> None:
    """T2: spans of 2, 0 and 5 s; the reward rides the second one."""
    step_rewards, dones, q, mask, discounts = _timed([2.0, 0.0, 5.0, 1.0], [0.0, 1.0, 0.0, 0.0])

    targets, learnable = n_step_targets(
        step_rewards, dones, q, q, mask, discounts=discounts, n_step=3
    )

    # gamma^2 to reach the reward's span, gamma^0 for where inside it the
    # reward is booked; the bootstrap is discounted by all 7 s of the window.
    expected = GAMMA_S**2 * GAMMA_S**0 * 1.0 + GAMMA_S**7 * BOOTSTRAP
    assert learnable[0, 0] == 1.0
    assert targets[0, 0].item() == pytest.approx(expected, rel=1e-6)


def test_a_purchase_takes_no_game_time_and_costs_no_discount() -> None:
    """T3: d = 1 at 0 ms, so the bootstrap is discounted by the timed steps only."""
    assert torch.equal(
        PER_SECOND.transition_discounts(torch.zeros(1, 3)), torch.ones(1, 3, dtype=torch.float64)
    )
    seconds = [0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 1.0]
    step_rewards, dones, q, mask, discounts = _timed(seconds, [0.0] * 7)

    targets, _ = n_step_targets(step_rewards, dones, q, q, mask, discounts=discounts, n_step=6)

    assert targets[0, 0].item() == pytest.approx(GAMMA_S**7 * BOOTSTRAP, rel=1e-6)


def test_a_terminal_inside_the_window_ends_the_timed_return() -> None:
    """T4: no bootstrap, and nothing after the terminal counts."""
    step_rewards, dones, q, mask, discounts = _timed(
        [2.0, 3.0, 1.0, 1.0, 1.0], [1.0, 1.0, 5.0, 5.0, 5.0], dones_at=(1,)
    )

    targets, learnable = n_step_targets(
        step_rewards, dones, q, q, mask, discounts=discounts, n_step=3
    )

    expected = GAMMA_S**2 * 1.0 + GAMMA_S**2 * (GAMMA_S**3 * 1.0)
    assert learnable[0, 0] == 1.0
    assert targets[0, 0].item() == pytest.approx(expected, rel=1e-6)


def test_a_window_past_the_end_is_learnable_only_if_it_terminated() -> None:
    """T5, and padding at 0 ms changes no real step's target."""
    open_rewards, open_dones, q, mask, discounts = _timed([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    _, open_learnable = n_step_targets(
        open_rewards, open_dones, q, q, mask, discounts=discounts, n_step=3
    )
    assert open_learnable.tolist() == [[0.0, 0.0, 0.0]]

    ended_rewards, ended_dones, q, mask, discounts = _timed(
        [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], dones_at=(2,)
    )
    ended, ended_learnable = n_step_targets(
        ended_rewards, ended_dones, q, q, mask, discounts=discounts, n_step=3
    )
    assert ended_learnable.tolist() == [[1.0, 1.0, 1.0]]
    assert ended[0, 1].item() == pytest.approx(GAMMA_S**2 + GAMMA_S**2 * GAMMA_S**3, rel=1e-6)

    # The same episode behind two steps of front padding, as the actor pads a
    # short one: zero reward, never done, and no game time.
    padded_rewards, padded_dones, q, mask, discounts = _timed(
        [0.0, 0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0, 1.0], dones_at=(4,)
    )
    padded, padded_learnable = n_step_targets(
        padded_rewards, padded_dones, q, q, mask, discounts=discounts, n_step=3
    )
    assert torch.equal(padded[:, 2:], ended)
    assert torch.equal(padded_learnable[:, 2:], ended_learnable)
