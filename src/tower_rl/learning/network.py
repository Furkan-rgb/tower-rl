"""The V1 policy networks described in `solution.md` 9.3.

The upgrade rows are scored by one shared encoder and one shared scorer, so the
parameter count does not depend on how many upgrades exist and a slot is valued
by what it is rather than by a weight vector bound to its index.  A learned
identity embedding supplies per-slot specificity, and its table is deliberately
larger than the current roster so a slot that becomes available later occupies an
unused row instead of forcing a reshape.

Two networks are built from the same encoder and the same dueling heads, and
differ only in how they carry information across time: `RecurrentPolicyNetwork`
threads an LSTM state, `StackedPolicyNetwork` concatenates the last `k` run
scalar vectors.  Sharing everything else is what makes a comparison between them
a comparison of that one choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tower_rl.domain.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT
from tower_rl.domain.run_actions import RUN_ACTIONS

RecurrentState = tuple[Tensor, Tensor]
#: The stacked agent's carried state: the last `history_length - 1` scalar
#: vectors, shaped `[batch, history_length - 1, scalars]`.
StackedState = Tensor


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
    #: Width of whatever carries information across time, recurrent or not.
    core_hidden: int = 128

    def __post_init__(self) -> None:
        if self.identity_capacity < self.row_count:
            raise ValueError("the identity table must cover at least the current roster")
        if self.action_count != self.row_count + 1:
            raise ValueError("the action space is WAIT plus one action per upgrade row")


class TowerTrunk(nn.Module):
    """Encodes one state: every upgrade row through shared weights, then pooled."""

    def __init__(self, config: NetworkConfig, *, history_length: int = 1) -> None:
        super().__init__()
        if history_length < 1:
            raise ValueError("history length must be at least one step")
        self.config = config
        self.history_length = history_length
        self.identity = nn.Embedding(config.identity_capacity, config.identity_dim)
        self.row_encoder = nn.Sequential(
            nn.Linear(config.row_width + config.identity_dim, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(config.scalar_count * history_length, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.SiLU(),
        )

    @property
    def output_width(self) -> int:
        return self.config.hidden * 3

    def forward(self, scalars: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        """Return the encoded rows and the pooled state summary.

        `scalars` is `[batch, time, scalar_count * history_length]` and `rows` is
        `[batch, time, row_count, row_width]`.
        """
        batch, time = rows.shape[0], rows.shape[1]
        cfg = self.config
        identities = torch.arange(cfg.row_count, device=rows.device)
        identity = self.identity(identities).expand(batch, time, cfg.row_count, cfg.identity_dim)
        encoded_rows = self.row_encoder(torch.cat((rows, identity), dim=-1))
        # Mean and max pooling together: the mean says what the roster looks like
        # overall, the max says whether any single slot is compelling right now.
        pooled = torch.cat(
            (encoded_rows.mean(dim=2), encoded_rows.amax(dim=2), self.scalar_encoder(scalars)),
            dim=-1,
        )
        return encoded_rows, pooled


class DuelingHeads(nn.Module):
    """Turns a core summary and the encoded rows into masked Q-values."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.value_head = nn.Sequential(
            nn.Linear(config.core_hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 1),
        )
        self.wait_advantage = nn.Sequential(
            nn.Linear(config.core_hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 1),
        )
        self.row_advantage = nn.Sequential(
            nn.Linear(config.hidden + config.core_hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 1),
        )

    def forward(self, core: Tensor, encoded_rows: Tensor, mask: Tensor) -> Tensor:
        batch, time = core.shape[0], core.shape[1]
        cfg = self.config
        value = self.value_head(core)
        wait = self.wait_advantage(core)
        expanded = core.unsqueeze(2).expand(batch, time, cfg.row_count, cfg.core_hidden)
        rows_advantage = self.row_advantage(torch.cat((encoded_rows, expanded), dim=-1)).squeeze(-1)
        return _dueling_masked_q(value, torch.cat((wait, rows_advantage), dim=-1), mask)


class RecurrentPolicyNetwork(nn.Module):
    """Dueling recurrent Q-network over a variable-length upgrade roster."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.trunk = TowerTrunk(self.config)
        self.core = nn.LSTM(self.trunk.output_width, self.config.core_hidden, batch_first=True)
        self.heads = DuelingHeads(self.config)

    def initial_state(self, batch: int, device: torch.device | None = None) -> RecurrentState:
        zeros = torch.zeros(1, batch, self.config.core_hidden, device=device)
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
        encoded_rows, pooled = self.trunk(scalars, rows)
        core, next_state = self.core(pooled, state)
        return self.heads(core, encoded_rows, mask), next_state


class StackedPolicyNetwork(nn.Module):
    """Dueling feed-forward Q-network over a stacked window of run scalars.

    `docs/rl-candidates.md` 3.1 argues that this problem is much closer to fully
    observed than to partially observed, so a short window of recent scalars may
    carry everything recurrence would. Upgrade rows are supplied for the current
    step only: they already describe the build, and stacking them would multiply
    the input width for no information gain.
    """

    def __init__(self, config: NetworkConfig | None = None, *, history_length: int = 8) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.history_length = history_length
        self.trunk = TowerTrunk(self.config, history_length=history_length)
        self.core = nn.Sequential(
            nn.Linear(self.trunk.output_width, self.config.core_hidden),
            nn.LayerNorm(self.config.core_hidden),
            nn.SiLU(),
            nn.Linear(self.config.core_hidden, self.config.core_hidden),
            nn.SiLU(),
        )
        self.heads = DuelingHeads(self.config)

    def initial_state(self, batch: int, device: torch.device | None = None) -> StackedState:
        return torch.zeros(
            batch, self.history_length - 1, self.config.scalar_count, device=device
        )

    def carry(self, scalars: Tensor, state: StackedState) -> StackedState:
        """The state that follows `scalars`, without computing anything else.

        Used to walk the burn-in prefix of a stored sequence: for this agent
        burn-in fills the window rather than warming a hidden state, and the two
        are the same requirement.
        """
        if self.history_length == 1:
            return state
        return torch.cat((state, scalars), dim=1)[:, -(self.history_length - 1) :]

    def stack(self, scalars: Tensor, state: StackedState) -> Tensor:
        """Window each step with the `history_length - 1` scalar vectors before it."""
        if self.history_length == 1:
            return scalars
        time = scalars.shape[1]
        padded = torch.cat((state, scalars), dim=1)
        windows = [padded[:, offset : offset + time] for offset in range(self.history_length)]
        return torch.cat(windows, dim=-1)

    def forward(
        self,
        scalars: Tensor,
        rows: Tensor,
        mask: Tensor,
        state: StackedState | None = None,
    ) -> tuple[Tensor, StackedState]:
        """Return masked Q-values and the window state that follows this input."""
        if state is None:
            state = self.initial_state(scalars.shape[0], scalars.device)
        encoded_rows, pooled = self.trunk(self.stack(scalars, state), rows)
        return self.heads(self.core(pooled), encoded_rows, mask), self.carry(scalars, state)


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
