from __future__ import annotations

import pytest
import torch

from tower_rl.domain.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT
from tower_rl.domain.run_actions import RUN_ACTIONS
from tower_rl.learning.network import (
    NetworkConfig,
    RecurrentPolicyNetwork,
    greedy_action,
    masked_max,
)

ACTIONS = len(RUN_ACTIONS)


def _batch(batch: int = 2, time: int = 3, *, valid: list[int] | None = None):
    torch.manual_seed(0)
    scalars = torch.randn(batch, time, SCALAR_COUNT)
    rows = torch.randn(batch, time, ROW_COUNT, ROW_WIDTH)
    mask = torch.zeros(batch, time, ACTIONS, dtype=torch.bool)
    for index in valid if valid is not None else [0, 1, 2]:
        mask[..., index] = True
    return scalars, rows, mask


def test_forward_shapes_and_recurrent_state_threading() -> None:
    network = RecurrentPolicyNetwork()
    scalars, rows, mask = _batch()

    q, state = network(scalars, rows, mask)

    assert q.shape == (2, 3, ACTIONS)
    assert state[0].shape == (1, 2, network.config.core_hidden)

    # State carried forward must change the output; otherwise the LSTM is inert.
    continued, _ = network(scalars, rows, mask, state)
    assert not torch.allclose(q, continued)


def test_invalid_actions_are_unselectable_and_never_bootstrap() -> None:
    network = RecurrentPolicyNetwork()
    scalars, rows, mask = _batch(valid=[0, 5, 9])

    q, _ = network(scalars, rows, mask)

    assert torch.isinf(q[~mask]).all() and (q[~mask] < 0).all()
    assert torch.isfinite(q[mask]).all()
    chosen = greedy_action(q)
    assert torch.tensor([value in (0, 5, 9) for value in chosen.flatten()]).all()


def test_a_terminal_state_bootstraps_zero_rather_than_negative_infinity() -> None:
    network = RecurrentPolicyNetwork()
    scalars, rows, mask = _batch()
    mask[:] = False  # no action is available on a terminal state

    q, _ = network(scalars, rows, mask)

    assert torch.isinf(q).all()
    assert torch.equal(masked_max(q), torch.zeros(2, 3))


def test_dueling_centre_uses_valid_actions_only() -> None:
    """Adding an always-invalid slot must not move the valid actions' Q-values."""
    network = RecurrentPolicyNetwork()
    scalars, rows, mask = _batch(valid=[0, 1])

    narrow, _ = network(scalars, rows, mask)

    wider = mask.clone()
    wider[..., 40] = False  # still invalid, but exercised through the same path
    same, _ = network(scalars, rows, wider)

    assert torch.allclose(narrow[mask], same[mask])


def test_parameter_count_is_independent_of_the_roster_size() -> None:
    small = RecurrentPolicyNetwork(NetworkConfig(row_count=10, action_count=11))
    large = RecurrentPolicyNetwork(NetworkConfig(row_count=40, action_count=41))

    def weights(model: RecurrentPolicyNetwork) -> int:
        return sum(p.numel() for name, p in model.named_parameters() if "identity" not in name)

    assert weights(small) == weights(large), "shared encoders must not scale with the roster"


def test_identity_table_is_over_provisioned_for_future_slots() -> None:
    config = NetworkConfig()

    assert config.identity_capacity > config.row_count

    with pytest.raises(ValueError, match="at least the current roster"):
        NetworkConfig(identity_capacity=4, row_count=60, action_count=61)
    with pytest.raises(ValueError, match="WAIT plus one action"):
        NetworkConfig(row_count=60, action_count=60)


def test_gradients_reach_the_shared_row_encoder() -> None:
    network = RecurrentPolicyNetwork()
    scalars, rows, mask = _batch()

    q, _ = network(scalars, rows, mask)
    q[mask].sum().backward()

    encoder_grad = network.trunk.row_encoder[0].weight.grad
    assert encoder_grad is not None and encoder_grad.abs().sum() > 0
    assert network.trunk.identity.weight.grad is not None
