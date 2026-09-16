"""The V1 policy network described in `solution.md` 9.3.

The upgrade rows are scored by one shared encoder and one shared scorer, so the
parameter count does not depend on how many upgrades exist and a slot is valued
by what it is rather than by a weight vector bound to its index.  A learned
identity embedding supplies per-slot specificity, and its table is deliberately
larger than the current roster so a slot that becomes available later occupies an
unused row instead of forcing a reshape.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tower_rl.domain.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT
from tower_rl.domain.run_actions import RUN_ACTIONS

RecurrentState = tuple[Tensor, Tensor]


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
    recurrent_hidden: int = 128

    def __post_init__(self) -> None:
        if self.identity_capacity < self.row_count:
            raise ValueError("the identity table must cover at least the current roster")
        if self.action_count != self.row_count + 1:
            raise ValueError("the action space is WAIT plus one action per upgrade row")


class TowerPolicyNetwork(nn.Module):
    """Dueling recurrent Q-network over a variable-length upgrade roster."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        cfg = self.config

        self.identity = nn.Embedding(cfg.identity_capacity, cfg.identity_dim)
        self.row_encoder = nn.Sequential(
            nn.Linear(cfg.row_width + cfg.identity_dim, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(cfg.scalar_count, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.SiLU(),
        )
        # Mean and max pooling together: the mean says what the roster looks like
        # overall, the max says whether any single slot is compelling right now.
        self.core = nn.LSTM(cfg.hidden * 3, cfg.recurrent_hidden, batch_first=True)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.recurrent_hidden, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 1),
        )
        self.wait_advantage = nn.Sequential(
            nn.Linear(cfg.recurrent_hidden, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 1),
        )
        self.row_advantage = nn.Sequential(
            nn.Linear(cfg.hidden + cfg.recurrent_hidden, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 1),
        )

    def initial_state(self, batch: int, device: torch.device | None = None) -> RecurrentState:
        zeros = torch.zeros(1, batch, self.config.recurrent_hidden, device=device)
        return zeros, zeros.clone()

    def forward(
        self,
        scalars: Tensor,
        rows: Tensor,
        mask: Tensor,
        state: RecurrentState | None = None,
    ) -> tuple[Tensor, RecurrentState]:
        """Return masked Q-values for `[batch, time, action]` and the next state.

        `rows` is `[batch, time, row_count, row_width]` and `mask` is
        `[batch, time, action_count]` with index 0 always `WAIT`.
        """
        batch, time = scalars.shape[0], scalars.shape[1]
        cfg = self.config

        identities = torch.arange(cfg.row_count, device=rows.device)
        identity = self.identity(identities).expand(batch, time, cfg.row_count, cfg.identity_dim)
        encoded_rows = self.row_encoder(torch.cat((rows, identity), dim=-1))

        pooled = torch.cat(
            (encoded_rows.mean(dim=2), encoded_rows.amax(dim=2), self.scalar_encoder(scalars)),
            dim=-1,
        )
        recurrent, next_state = self.core(pooled, state)

        value = self.value_head(recurrent)
        wait = self.wait_advantage(recurrent)
        expanded = recurrent.unsqueeze(2).expand(batch, time, cfg.row_count, cfg.recurrent_hidden)
        rows_advantage = self.row_advantage(torch.cat((encoded_rows, expanded), dim=-1)).squeeze(-1)
        advantages = torch.cat((wait, rows_advantage), dim=-1)

        return _dueling_masked_q(value, advantages, mask), next_state


def _dueling_masked_q(value: Tensor, advantages: Tensor, mask: Tensor) -> Tensor:
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


def greedy_action(q_values: Tensor) -> Tensor:
    """Pick the best valid action. Invalid actions are already `-inf`."""
    return q_values.argmax(dim=-1)


def masked_max(q_values: Tensor) -> Tensor:
    """The bootstrapped maximum over valid actions, or zero when none exist.

    A terminal state has no valid action at all; its bootstrap must contribute
    nothing rather than negative infinity, which would poison the target.
    """
    best = q_values.max(dim=-1).values
    return torch.where(torch.isfinite(best), best, torch.zeros_like(best))
