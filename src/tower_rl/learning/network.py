"""The V1 policy networks described in `solution.md` 9.3.

The upgrade rows are scored by one shared encoder and one shared scorer, so the
parameter count does not depend on how many upgrades exist and a slot is valued
by what it is rather than by a weight vector bound to its index.  A learned
identity embedding supplies per-slot specificity, and its table is deliberately
larger than the current roster so a slot that becomes available later occupies an
unused row instead of forcing a reshape.

`TowerTrunk` is the shared encoder; R2D2's network (`learning/r2d2.py`) is built on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT
from tower_rl.environment.run_actions import RUN_ACTIONS


@dataclass(frozen=True)
class NetworkConfig:
    """Compact by default: sample quality and throughput dominate, not capacity."""

    scalar_count: int = SCALAR_COUNT
    row_count: int = ROW_COUNT
    row_width: int = ROW_WIDTH
    action_count: int = len(RUN_ACTIONS)
    #: Larger than the live roster on purpose, so a later slot needs no reshape.
    identity_capacity: int = 96
    identity_dim: int = 16
    hidden: int = 128

    def __post_init__(self) -> None:
        if self.identity_capacity < self.row_count:
            raise ValueError("the identity table must cover at least the current roster")
        if self.action_count != self.row_count + 1:
            raise ValueError("the action space is WAIT plus one action per upgrade row")


class TowerTrunk(nn.Module):
    """Encodes one state: every upgrade row through shared weights, and the run scalars."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.identity = nn.Embedding(config.identity_capacity, config.identity_dim)
        self.row_encoder = nn.Sequential(
            nn.Linear(config.row_width + config.identity_dim, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(config.scalar_count, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.SiLU(),
        )

    def encode(self, scalars: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        """Return every row encoded by the shared weights, and the encoded scalars.

        `scalars` is `[batch, time, scalar_count]` and `rows` is
        `[batch, time, row_count, row_width]`.
        """
        batch, time = rows.shape[0], rows.shape[1]
        cfg = self.config
        identities = torch.arange(cfg.row_count, device=rows.device)
        identity = self.identity(identities).expand(batch, time, cfg.row_count, cfg.identity_dim)
        encoded_rows = self.row_encoder(torch.cat((rows, identity), dim=-1))
        return encoded_rows, self.scalar_encoder(scalars)


def dueling_masked_q(value: Tensor, advantages: Tensor, mask: Tensor) -> Tensor:
    """Combine value and advantage, centring over valid actions only.

    Subtracting the mean over *all* actions would let the many permanently invalid
    slots drag the centre around, which shifts the Q-values of the handful of
    actions that are actually available. Invalid actions are then driven to
    negative infinity so they can never be selected and can never back up value
    through a target maximum.
    """
    valid = mask.to(advantages.dtype)
    count = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
    centre = (advantages * valid).sum(dim=-1, keepdim=True) / count
    q = value + advantages - centre
    return q.masked_fill(~mask, float("-inf"))

