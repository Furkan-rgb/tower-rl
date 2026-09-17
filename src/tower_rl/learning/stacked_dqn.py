"""The rank-1 candidate from `docs/rl-candidates.md` 3.1.

Masked data-efficient DQN on a stacked history: the same masked dueling double-Q
learning as the recurrent backbone, over the same replay sequences and the same
targets, but carrying time in a window of recent run scalars instead of in an
LSTM state.  That single difference is the hypothesis - section 2.3 of the
candidate study argues this problem is much closer to fully observed than the
recurrent skeleton assumes, and this backbone is how that gets tested rather than
asserted.

Three things depart from the recurrent backbone's operating point, all from the
Atari 100k literature the study cites: an EMA target rather than a periodic hard
copy, decoupled weight decay, and a replay ratio the training loop supplies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch

from tower_rl.domain.features import ROW_COUNT, ROW_WIDTH, StateFeatures
from tower_rl.learning.backbone import LearnMetrics, SequenceBatch
from tower_rl.learning.network import NetworkConfig, StackedPolicyNetwork, StackedState
from tower_rl.learning.value_learning import n_step_targets, weighted_sequence_loss


@dataclass(frozen=True)
class StackedDqnConfig:
    #: How many run-scalar vectors the window holds, the current one included.
    #: The candidate study treats this as the tuned knob in the range 4 to 16 and
    #: names k = 1 as the ablation that settles whether history is needed at all.
    history_length: int = 8
    discount: float = 0.997
    n_step: int = 5
    learning_rate: float = 1e-4
    #: Decoupled weight decay, hence AdamW rather than Adam.
    weight_decay: float = 1e-5
    #: A target that follows the online network smoothly. At this replay ratio a
    #: periodic hard copy moves the target in large infrequent jumps, which is
    #: what the data-efficient recipe replaces.
    target_ema_decay: float = 0.995
    gradient_clip: float = 10.0
    huber_delta: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.history_length < 1:
            raise ValueError("history length must be at least one step")
        if not 0.0 < self.discount < 1.0:
            raise ValueError("discount must be within (0, 1)")
        if self.n_step < 1:
            raise ValueError("n-step must be positive")
        if not 0.0 < self.target_ema_decay < 1.0:
            raise ValueError("target EMA decay must be within (0, 1)")


@dataclass
class StackedDqnBackbone:
    """Masked, dueling, feed-forward Double Q-learning on a stacked history."""

    config: StackedDqnConfig = field(default_factory=StackedDqnConfig)
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    online: StackedPolicyNetwork = field(init=False)
    target: StackedPolicyNetwork = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    _steps: int = field(default=0, init=False)
    _random: random.Random = field(init=False)

    def __post_init__(self) -> None:
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        history = self.config.history_length
        self.online = StackedPolicyNetwork(self.network_config, history_length=history)
        self.target = StackedPolicyNetwork(self.network_config, history_length=history)
        self.online = self.online.to(self.device)
        self.target = self.target.to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._random = random.Random(self.config.seed)

    # -- acting ------------------------------------------------------------

    @property
    def model_version(self) -> int:
        return self._steps

    def initial_state(self) -> StackedState:
        return self.online.initial_state(1, self.device)

    def act(
        self, features: StateFeatures, state: StackedState | None, *, epsilon: float
    ) -> tuple[int, StackedState]:
        """Choose among valid actions only, exploring within the mask."""
        valid = [index for index, allowed in enumerate(features.mask) if allowed]
        if not valid:
            raise ValueError("no action is available in this state")
        scalars = torch.tensor(
            [[list(features.scalars)]], dtype=torch.float32, device=self.device
        )
        rows = torch.tensor(
            [[list(features.rows)]], dtype=torch.float32, device=self.device
        ).view(1, 1, ROW_COUNT, ROW_WIDTH)
        mask = torch.tensor([[list(features.mask)]], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q, next_state = self.online(scalars, rows, mask, state)
        # Exploration still respects the mask: an epsilon action is drawn from the
        # valid set, never from the whole space, so exploration cannot waste a
        # step on something the game would refuse anyway.
        if epsilon > 0.0 and self._random.random() < epsilon:
            return self._random.choice(valid), next_state
        return int(q[0, 0].argmax().item()), next_state

    # -- learning ----------------------------------------------------------

    def learn(self, batch: SequenceBatch) -> LearnMetrics:
        """One optimisation step over a batch of equal-length sequences."""
        burn_in = batch.burn_in
        if burn_in < self.config.history_length - 1:
            raise ValueError(
                f"burn-in of {burn_in} cannot fill a window of "
                f"{self.config.history_length}; the first learning steps would be "
                "trained on padded history that acting never sees mid-episode"
            )

        # Burn-in fills the window here rather than warming a hidden state. It
        # costs no forward pass: the window is the stored scalars themselves.
        history = self.online.carry(
            batch.scalars[:, :burn_in],
            self.online.initial_state(batch.batch_size, self.device),
        )

        scalars = batch.scalars[:, burn_in:]
        rows = batch.rows[:, burn_in:]
        mask = batch.mask[:, burn_in:]
        actions = batch.actions[:, burn_in:]
        rewards = batch.rewards[:, burn_in:]
        dones = batch.dones[:, burn_in:]

        online_q, _ = self.online(scalars, rows, mask, history)
        chosen = online_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            target_q, _ = self.target(scalars, rows, mask, history)
            targets, learnable = n_step_targets(
                rewards,
                dones,
                online_q.detach(),
                target_q,
                mask,
                discount=self.config.discount,
                n_step=self.config.n_step,
            )

        errors = (targets - chosen) * learnable
        loss = weighted_sequence_loss(
            chosen, targets, learnable, batch.weights, huber_delta=self.config.huber_delta
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self._steps += 1
        self._update_target()

        absolute = errors.abs().detach()
        return LearnMetrics(
            loss=float(loss.detach().item()),
            mean_absolute_td_error=float(absolute.mean().item()),
            gradient_norm=float(gradient_norm.item()),
            td_errors=tuple(tuple(row.tolist()) for row in absolute.cpu()),
        )

    def _update_target(self) -> None:
        """Move the target a little way towards the online network, every step."""
        decay = self.config.target_ema_decay
        with torch.no_grad():
            for target, online in zip(
                self.target.parameters(), self.online.parameters(), strict=True
            ):
                target.mul_(decay).add_(online, alpha=1.0 - decay)
            # Buffers - the LayerNorm statistics here - are copied rather than
            # averaged; they are not learned parameters and averaging them would
            # mix two normalisations.
            for target_buffer, online_buffer in zip(
                self.target.buffers(), self.online.buffers(), strict=True
            ):
                target_buffer.copy_(online_buffer)

    # -- persistence -------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self._steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._steps = int(state["steps"])
