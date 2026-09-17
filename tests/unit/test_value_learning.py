from __future__ import annotations

import pytest
import torch

from tower_rl.domain.run_actions import RUN_ACTIONS
from tower_rl.learning.value_learning import n_step_targets

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
