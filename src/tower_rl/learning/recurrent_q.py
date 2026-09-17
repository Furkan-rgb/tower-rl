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
from torch import nn

from tower_rl.domain.features import ROW_COUNT, ROW_WIDTH, StateFeatures
from tower_rl.learning.backbone import LearnMetrics, SequenceBatch
from tower_rl.learning.network import (
    NetworkConfig,
    RecurrentPolicyNetwork,
    RecurrentState,
)
from tower_rl.learning.value_learning import (
    n_step_targets,
    real_step_td_errors,
    weighted_sequence_loss,
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

    online: RecurrentPolicyNetwork = field(init=False)
    target: RecurrentPolicyNetwork = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    _steps: int = field(default=0, init=False)
    _random: random.Random = field(init=False)

    def __post_init__(self) -> None:
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.online = RecurrentPolicyNetwork(self.network_config).to(self.device)
        self.target = RecurrentPolicyNetwork(self.network_config).to(self.device)
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

    def stored_recurrent_state(self, state: RecurrentState) -> RecurrentState:
        """The LSTM state as replay keeps it: detached, on CPU, its own storage.

        Replay outlives both the autograd graph that produced the state and, on a
        GPU run, the memory it lived in, so a stored state must own neither.
        """
        hidden, cell = state
        return hidden.detach().cpu().clone(), cell.detach().cpu().clone()

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
        stored = self._stored_state(batch)
        state = stored
        target_state = stored

        if burn_in:
            # Burn-in reconstructs the recurrent state without training on it, so
            # the staleness of the stored state - it was produced by older
            # parameters - is corrected before the learning window, by each
            # network through its own weights. R2D2 section 2.3 measures this
            # combination against burning in from zeros and prefers it; zero
            # state plus burn-in is the variant that paper argues against.
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
        real = (~batch.padding[:, burn_in:]).to(rewards.dtype)

        online_q, _ = self.online(scalars, rows, mask, state)
        chosen = online_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            target_q, _ = self.target(scalars, rows, mask, target_state)
            targets, learnable = n_step_targets(
                rewards,
                dones,
                online_q.detach(),
                target_q,
                mask,
                discount=self.config.discount,
                n_step=self.config.n_step,
            )
            # Padding is filler that fills a window for a short episode. It is
            # never a target, so it leaves the loss and the priorities alone.
            learnable = learnable * real

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
        if self._steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        absolute = errors.abs().detach()
        return LearnMetrics(
            loss=float(loss.detach().item()),
            mean_absolute_td_error=float(
                (absolute.sum() / real.sum().clamp(min=1.0)).item()
            ),
            gradient_norm=float(gradient_norm.item()),
            td_errors=real_step_td_errors(absolute, real),
        )

    def _stored_state(self, batch: SequenceBatch) -> RecurrentState:
        """The state burn-in starts from: what the actor stored, per sequence.

        Batched here rather than in collation because the shape is this network's
        own: an LSTM state is `[layers, batch, hidden]`, so sequences join along
        dimension one. This backbone always stores a state (`stored_recurrent_state`
        never returns `None`), so a batch that carries none was not produced by
        this backbone's own actor. Silently starting from zeros there would
        quietly reintroduce the zero-state burn-in R2D2 section 2.3 argues
        against, so it is refused rather than guessed; a caller that legitimately
        wants zero-state burn-in must pass an explicit zeroed state.
        """
        if not batch.recurrent_states:
            raise ValueError(
                "recurrent batch carries no stored recurrent state: every "
                "sequence's `recurrent_state` was None"
            )
        hidden = torch.cat([state[0] for state in batch.recurrent_states], dim=1)
        cell = torch.cat([state[1] for state in batch.recurrent_states], dim=1)
        return hidden.to(self.device), cell.to(self.device)

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
