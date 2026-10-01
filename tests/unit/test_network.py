"""`TowerTrunk`, the encoder R2D2's network is built on (`learning/network.py`)."""

from __future__ import annotations

import pytest
import torch

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT
from tower_rl.learning.network import NetworkConfig, TowerTrunk


def _inputs(batch: int = 2, time: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(batch, time, SCALAR_COUNT, generator=generator),
        torch.randn(batch, time, ROW_COUNT, ROW_WIDTH, generator=generator),
    )


def test_every_row_is_encoded_through_shared_weights() -> None:
    trunk = TowerTrunk(NetworkConfig())
    scalars, rows = _inputs()

    encoded_rows, encoded_scalars = trunk.encode(scalars, rows)

    hidden = trunk.config.hidden
    assert encoded_rows.shape == (2, 3, ROW_COUNT, hidden)
    assert encoded_scalars.shape == (2, 3, hidden)
    # Two rows with the same features differ only by their learned identity.
    same = rows.clone()
    same[..., 1, :] = same[..., 0, :]
    first, second = trunk.encode(scalars, same)[0][..., :2, :].unbind(dim=-2)
    assert not torch.allclose(first, second)


def test_parameter_count_is_independent_of_the_roster_size() -> None:
    small = TowerTrunk(NetworkConfig(row_count=10, action_count=11))
    large = TowerTrunk(NetworkConfig(row_count=40, action_count=41))

    def weights(model: TowerTrunk) -> int:
        return sum(p.numel() for name, p in model.named_parameters() if "identity" not in name)

    assert weights(small) == weights(large), "shared encoders must not scale with the roster"


def test_identity_table_is_over_provisioned_for_future_slots() -> None:
    config = NetworkConfig()

    assert config.identity_capacity > config.row_count

    with pytest.raises(ValueError, match="at least the current roster"):
        NetworkConfig(identity_capacity=4, row_count=60, action_count=61)
    with pytest.raises(ValueError, match="WAIT plus one action"):
        NetworkConfig(row_count=60, action_count=60)


def test_gradients_reach_the_shared_row_encoder_and_the_identity_table() -> None:
    trunk = TowerTrunk(NetworkConfig())
    scalars, rows = _inputs()

    encoded_rows, encoded_scalars = trunk.encode(scalars, rows)
    (encoded_rows.sum() + encoded_scalars.sum()).backward()

    encoder_grad = trunk.row_encoder[0].weight.grad
    assert encoder_grad is not None and encoder_grad.abs().sum() > 0
    identity_grad = trunk.identity.weight.grad
    assert identity_grad is not None and identity_grad[: ROW_COUNT].abs().sum() > 0
    assert identity_grad[ROW_COUNT:].abs().sum() == 0, "only the slots in use are trained"
    scalar_grad = trunk.scalar_encoder[0].weight.grad
    assert scalar_grad is not None and scalar_grad.abs().sum() > 0
