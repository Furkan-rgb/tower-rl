"""The recurrent value-based backbone: the contract baseline for the comparison.

Double Q-learning, n-step returns, Huber loss, a target network, and burn-in
before the learning window, per `solution.md` 9.4.  `docs/rl-candidates.md`
argues that R2D2's published hyperparameters target a budget four orders of
magnitude larger than ours, so the defaults here come from the data-efficient
setting; the skeleton is R2D2's, the operating point is not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from tower_rl.domain.features import ROW_COUNT, ROW_WIDTH, StateFeatures
from tower_rl.learning.backbone import LearnMetrics, SequenceBatch
from tower_rl.learning.network import (
    NetworkConfig,
    RecurrentState,
    TowerPolicyNetwork,
)


@dataclass(frozen=True)
class RecurrentQConfig:
    discount: float = 0.997
    n_step: int = 5
    learning_rate: float = 1e-4
    #: How often the target network copies the online one, in optimisation steps.
    target_update_interval: int = 200
    gradient_clip: float = 10.0
    huber_delta: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.discount < 1.0:
            raise ValueError("discount must be within (0, 1)")
        if self.n_step < 1:
            raise ValueError("n-step must be positive")
        if self.target_update_interval < 1:
            raise ValueError("target update interval must be positive")


@dataclass
class RecurrentQBackbone:
    """Masked, dueling, recurrent Double Q-learning."""

    config: RecurrentQConfig = field(default_factory=RecurrentQConfig)
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    online: TowerPolicyNetwork = field(init=False)
    target: TowerPolicyNetwork = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    _steps: int = field(default=0, init=False)
    _random: random.Random = field(init=False)

    def __post_init__(self) -> None:
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.online = TowerPolicyNetwork(self.network_config).to(self.device)
        self.target = TowerPolicyNetwork(self.network_config).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.config.learning_rate
        )
        self._random = random.Random(self.config.seed)

    # -- acting ------------------------------------------------------------

    @property
    def model_version(self) -> int:
        return self._steps

    def initial_state(self) -> RecurrentState:
        return self.online.initial_state(1, self.device)

    def act(
        self, features: StateFeatures, state: RecurrentState | None, *, epsilon: float
    ) -> tuple[int, RecurrentState]:
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
        state = self.online.initial_state(batch.batch_size, self.device)
        target_state = self.target.initial_state(batch.batch_size, self.device)

        if burn_in:
            # Burn-in reconstructs the recurrent state without training on it, so
            # a stored state that has drifted since collection cannot bias the
            # learning window.
            with torch.no_grad():
                _, state = self.online(
                    batch.scalars[:, :burn_in], batch.rows[:, :burn_in],
                    batch.mask[:, :burn_in], state,
                )
                _, target_state = self.target(
                    batch.scalars[:, :burn_in], batch.rows[:, :burn_in],
                    batch.mask[:, :burn_in], target_state,
                )

        scalars = batch.scalars[:, burn_in:]
        rows = batch.rows[:, burn_in:]
        mask = batch.mask[:, burn_in:]
        actions = batch.actions[:, burn_in:]
        rewards = batch.rewards[:, burn_in:]
        dones = batch.dones[:, burn_in:]

        online_q, _ = self.online(scalars, rows, mask, state)
        chosen = online_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            target_q, _ = self.target(scalars, rows, mask, target_state)
            targets, learnable = self._n_step_targets(
                rewards, dones, online_q.detach(), target_q, mask
            )

        errors = (targets - chosen) * learnable
        loss_per_step = torch.nn.functional.huber_loss(
            chosen, targets, reduction="none", delta=self.config.huber_delta
        )
        # Steps whose n-step window runs past the end of the sequence have no
        # well-defined target. They are excluded rather than trained on a
        # truncated return, which would bias their value downwards.
        counted = learnable.sum(dim=1).clamp(min=1.0)
        loss = ((loss_per_step * learnable).sum(dim=1) / counted * batch.weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self._steps += 1
        if self._steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        absolute = errors.abs().detach()
        return LearnMetrics(
            loss=float(loss.detach().item()),
            mean_absolute_td_error=float(absolute.mean().item()),
            gradient_norm=float(gradient_norm.item()),
            td_errors=tuple(tuple(row.tolist()) for row in absolute.cpu()),
        )

    def _n_step_targets(
        self, rewards: Tensor, dones: Tensor, online_q: Tensor, target_q: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Double Q n-step targets, and which steps have a well-defined one.

        The bootstrap comes from the state `n` steps ahead inside the same
        sequence, so a step whose window runs past the end is not learnable here.
        Termination is per sequence element, never collapsed across the batch.
        """
        batch, time = rewards.shape
        discount, n_step = self.config.discount, self.config.n_step

        # Double Q: the online network chooses the action, the target network
        # values it. That split is what stops one optimistic network from
        # bootstrapping its own overestimate.
        best = online_q.argmax(dim=-1, keepdim=True)
        evaluated = target_q.gather(-1, best).squeeze(-1)
        # A state with no valid action is terminal and must contribute zero, not
        # the negative infinity the mask would otherwise carry through.
        evaluated = torch.where(mask.any(dim=-1), evaluated, torch.zeros_like(evaluated))
        evaluated = torch.where(
            torch.isfinite(evaluated), evaluated, torch.zeros_like(evaluated)
        )

        targets = torch.zeros_like(rewards)
        learnable = torch.zeros_like(rewards)
        terminal = dones.to(rewards.dtype)
        for step in range(time):
            accumulated = torch.zeros(batch, device=rewards.device)
            alive = torch.ones(batch, device=rewards.device)
            factor = 1.0
            ended_inside_window = torch.zeros(batch, device=rewards.device)
            for offset in range(n_step):
                index = step + offset
                if index >= time:
                    break
                accumulated = accumulated + alive * factor * rewards[:, index]
                factor *= discount
                alive = alive * (1.0 - terminal[:, index])
                ended_inside_window = torch.maximum(ended_inside_window, terminal[:, index])
            bootstrap = step + n_step
            if bootstrap < time:
                accumulated = accumulated + alive * factor * evaluated[:, bootstrap]
                learnable[:, step] = 1.0
            else:
                # Without a bootstrap state the return is only complete if the
                # episode actually ended inside the window.
                learnable[:, step] = ended_inside_window
            targets[:, step] = accumulated
        return targets, learnable

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


def parameters_are_equal(left: nn.Module, right: nn.Module) -> bool:
    """Whether two modules hold identical weights, used by resume verification."""
    left_state, right_state = left.state_dict(), right.state_dict()
    if left_state.keys() != right_state.keys():
        return False
    return all(torch.equal(left_state[key], right_state[key]) for key in left_state)
