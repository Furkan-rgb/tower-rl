"""R2D2: a recurrent, dueling, double-Q learner on stored-state replay.

A port of R2D2 as published (Kapturowski et al. 2019, section 2.3, section 3,
Appendix, Table 2); where the paper is silent, as Acme builds it
(google-deepmind/acme at 4949d3ce: `agents/jax/r2d2/{learning,networks}.py`,
`jax/networks/{atari,embedding,duelling}.py`), with rlax's targets
(`nonlinear_bellman.py`, `multistep.py`, `transforms.py`) and haiku's
initialisers (`basic.py`, `recurrent.py`). It learns from `R2D2Replay`'s items.

What it follows:

- **Network.** A torso, then an LSTM of 512 fed the torso's output, the
  previous action one-hot and tanh of the previous reward (P Appendix; Acme
  `OAREmbedding`, embedding.py 47-58), then a dueling head of two MLPs of 512
  reading the LSTM output (Acme `DuellingMLP(A, [512])`). Initialised as
  haiku does (`_haiku_initialise`).
- **Learn step.** Both networks start from the item's stored state and are
  unrolled over its 40 burn-in steps, then over its trace (learning.py 86-116).
  Each of the 80 trace steps is a target: 5-step double Q, the online network
  choosing and the target network valuing (learning.py 121-143), through the
  value rescaling h (P section 2.3, `signed_hyperbolic`) and rlax's shortened
  returns at the item's end (`transformed_n_step_targets`). The loss is 0.5
  times the sum of squared TD errors over the trace, weighted by the item's
  importance-sampling weight and averaged over the batch; no Huber loss and
  no clipping (learning.py 144-151). Adam at the paper's lr 1e-4 and epsilon
  1e-3 (P Table 2). The target is a hard copy every 2,500 learner steps
  (learning.py 184-185).
- **Acting.** The LSTM state, the previous action and the previous reward are
  carried across decisions (`R2D2State`); an episode starts from zeros. The
  greedy action is the argmax of Q; with probability epsilon it is uniform.

What differs, each forced:

- **The torso** (the observation is a vector and 60 rows, not an image): the
  shared row encoder of `TowerTrunk` stands for the convolution's shared
  weights, and its 60 encoded rows, flattened in order - row i is action
  i + 1 - with the encoded scalars, stand for the convolution's flattened
  output. Then Acme's torso MLP: Linear 512, LayerNorm, ReLU
  (atari.py `DeepAtariTorso(hidden_sizes=[512], use_layer_norm=True)`).
- **The action mask** (docs/environment-contract.md): advantages are centred
  over the valid actions and invalid actions are -inf (`dueling_masked_q`);
  the greedy action, the double-Q argmax and an exploratory action are over
  the valid ones; a state with none has value 0.
- **Pad steps and the episode's last step are not targets**, where Acme
  trains on its zero pads: a pad step's mask is empty, so its Q is undefined,
  and no transition leaves the last step. An item's n-step return stops at
  its episode's last step as rlax's stops at a sequence's end, bootstrapping
  from that step's value: 0 when the run died, since its mask is empty.
- **The discount and the reward** (ADR 0013): each transition's discount is
  0.999 ** (its game-seconds), times 1 - done, inside rlax's per-step
  discounts; the reward is the survival reward (1 - 0.999 ** t) * V_REF,
  learned and fed to the LSTM, in place of the stored wave reward.
- **"Step" is a decision** (ADR 0009): n, the burn-in, the trace and the
  target period count decisions.

One choice where the paper and Acme differ: the burn-in is unrolled with no
gradient, as the paper describes it (section 3: burn-in only produces the
start state), where Acme differentiates the online unroll through it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy
import torch
from torch import Tensor, nn
from torch.nn import functional

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, StateFeatures
from tower_rl.learning.backbone import LearnMetrics, SequenceBatch
from tower_rl.learning.network import NetworkConfig, TowerTrunk, dueling_masked_q
from tower_rl.learning.r2d2_replay import R2D2_STATE_SIZE
from tower_rl.learning.value_learning import (
    evaluated_next_values,
    game_time_discounts,
    real_step_td_errors,
    survival_rewards,
    value_fit_correlation,
)

#: The name a run, its checkpoints and `--backbone` file this backbone under.
R2D2 = "r2d2"
#: Acme `DeepAtariTorso(hidden_sizes=[512])`'s MLP.
TORSO_SIZE = 512
#: Acme `DuellingMLP(num_actions, hidden_sizes=[512])`: each stream's hidden layer.
HEAD_HIDDEN = 512
#: P section 2.3's h(x) = sign(x)(sqrt(|x| + 1) - 1) + eps x; rlax's default eps.
VALUE_RESCALING_EPSILON = 1e-3


@dataclass(frozen=True)
class R2D2Config:
    """Every R2D2 learning setting, at the paper's values (Table 2)."""

    #: The task, not the algorithm (ADR 0013): discount per game-second, in
    #: place of the published 0.997 per step. Not a code default; the
    #: protocol's is 0.999, and the run sets it.
    discount_per_game_second: float
    #: P section 2.3, Acme `bootstrap_n`.
    n_step: int = 5
    #: P Table 2. Acme's config default is 1e-3; its own example runs 1e-4.
    learning_rate: float = 1e-4
    #: P Table 2. Acme leaves optax's 1e-8.
    adam_epsilon: float = 1e-3
    #: P Table 2, Acme `target_update_period`: learner steps between hard copies.
    target_update_period: int = 2_500
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.discount_per_game_second < 1.0:
            raise ValueError("discount per game-second must be within (0, 1)")
        if self.n_step < 1:
            raise ValueError("n-step must be positive")
        if self.target_update_period < 1:
            raise ValueError("the target update period must be positive")


# -- value rescaling (rlax transforms.py 60-84) --------------------------------


def signed_hyperbolic(x: Tensor, eps: float = VALUE_RESCALING_EPSILON) -> Tensor:
    """h(x) = sign(x)(sqrt(|x| + 1) - 1) + eps x."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def signed_parabolic(x: Tensor, eps: float = VALUE_RESCALING_EPSILON) -> Tensor:
    """h's exact inverse: sign(x)(z^2 - 1), z = (sqrt(1 + 4 eps (eps + 1 + |x|)) - 1) / 2 eps."""
    z = torch.sqrt(1.0 + 4.0 * eps * (eps + 1.0 + torch.abs(x))) / 2.0 / eps - 1.0 / 2.0 / eps
    return torch.sign(x) * (torch.square(z) - 1.0)


def transformed_n_step_targets(
    values: Tensor, rewards: Tensor, discounts: Tensor, last: Tensor, n: int
) -> Tensor:
    """rlax `transformed_n_step_returns`: h(n-step return with bootstrap h^-1(v)), per trace step.

    `values` [B, T + 1] are the rescaled bootstrap values v_0 .. v_T of the
    trace's steps; `rewards` and `discounts` [B, T] those of the transition
    *out of* each of its first T steps; `last` [B] each item's last real step.
    For step t: sum over k < n of (prod over j < k of d_{t+j}) r_{t+k}, plus
    (prod over j < n of d_{t+j}) h^-1(v_{min(t+n, last)}). Past `last` a
    transition has reward 0 and discount 1, as a pad step's does and as rlax
    pads a sequence's end (multistep.py 160-178), so a return running past
    the end is shortened to bootstrap from the last step's value.
    """
    batch, time = rewards.shape
    beyond = rewards.new_zeros(batch, n)
    padded_rewards = torch.cat((rewards, beyond), dim=1)
    padded_discounts = torch.cat((discounts, torch.ones_like(beyond)), dim=1)
    steps = torch.arange(time, device=rewards.device)
    bootstrap = torch.minimum(steps[None] + n, last.clamp(min=0)[:, None])
    returns = signed_parabolic(values.gather(1, bootstrap))
    for offset in reversed(range(n)):
        returns = (
            padded_rewards[:, offset : offset + time]
            + padded_discounts[:, offset : offset + time] * returns
        )
    return signed_hyperbolic(returns)


def trace_targets(
    online_q: Tensor,
    target_q: Tensor,
    mask: Tensor,
    game_ms: Tensor,
    dones: Tensor,
    padding: Tensor,
    *,
    discount_per_game_second: float,
    n: int,
) -> tuple[Tensor, Tensor]:
    """A trace's rescaled n-step double-Q targets, and which of its steps are targets.

    The trace is [B, T + 1] steps; the targets are for its first T.

    The inputs are the trace's steps in the replay's step layout: `game_ms`
    and `dones` describe the transition *into* each step, so the transition
    out of step t is step t + 1's. Its discount is gamma_s ** seconds times
    1 - done and its reward the survival reward (ADR 0013). The online
    network chooses the bootstrap action among the valid ones and the target
    network values it; a state with no valid action is worth 0
    (`evaluated_next_values`). Step t is a target iff step t + 1 is real, so a
    pad step and the episode's last step - an empty mask, an undefined Q -
    never are. Returns (targets [B, T] in `online_q`'s dtype, valid [B, T]).
    """
    discounts_into = game_time_discounts(discount_per_game_second, game_ms[:, 1:])
    rewards = survival_rewards(discounts_into)
    discounts = discounts_into * (~dones[:, 1:])
    # Items are right-padded, so the real steps are a prefix.
    last = (~padding).sum(dim=1) - 1
    values = evaluated_next_values(online_q, target_q, mask).to(torch.float64)
    targets = transformed_n_step_targets(values, rewards, discounts, last, n)
    return targets.to(online_q.dtype), ~padding[:, 1:]


# -- network -------------------------------------------------------------------


class R2D2Network(nn.Module):
    """Torso, LSTM over (torso, previous action, previous reward), masked dueling head."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        cfg = self.config
        self.trunk = TowerTrunk(cfg)
        self.torso = nn.Sequential(
            nn.Linear(cfg.row_count * cfg.hidden + cfg.hidden, TORSO_SIZE),
            nn.LayerNorm(TORSO_SIZE),
            nn.ReLU(),
        )
        self.core = nn.LSTM(TORSO_SIZE + cfg.action_count + 1, R2D2_STATE_SIZE, batch_first=True)
        self.value = nn.Sequential(
            nn.Linear(R2D2_STATE_SIZE, HEAD_HIDDEN), nn.ReLU(), nn.Linear(HEAD_HIDDEN, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(R2D2_STATE_SIZE, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, cfg.action_count),
        )
        _haiku_initialise(self)

    def forward(
        self,
        scalars: Tensor,
        rows: Tensor,
        mask: Tensor,
        previous_actions: Tensor,
        previous_rewards: Tensor,
        state: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Masked Q [B, T, A] from state (h, c), each [1, B, 512], and the state after.

        `scalars` [B, T, scalars], `rows` [B, T, rows, width], `mask` [B, T, A];
        `previous_actions` [B, T] the action taken at the step before and
        `previous_rewards` [B, T] the reward of the transition into the step.
        """
        encoded_rows, encoded_scalars = self.trunk.encode(scalars, rows)
        torso = self.torso(torch.cat((encoded_rows.flatten(2), encoded_scalars), dim=-1))
        embedded = torch.cat(
            (
                torso,
                functional.one_hot(previous_actions, self.config.action_count).to(torso.dtype),
                torch.tanh(previous_rewards).unsqueeze(-1).to(torso.dtype),
            ),
            dim=-1,
        )
        core, (h, c) = self.core(embedded, state)
        return dueling_masked_q(self.value(core), self.advantage(core), mask), (h, c)


def _haiku_initialise(network: R2D2Network) -> None:
    """haiku's defaults: every Linear truncated normal (2 std) at 1 / sqrt(fan_in), zero bias.

    haiku's LSTM is one Linear over [x, h] with one bias, and adds 1 to the
    forget gate's pre-activation (recurrent.py 339-343). Here its two weight
    matrices take that Linear's fan-in, `bias_hh` is zero and frozen, and the
    forget slice of `bias_ih` starts at 1: the same function and the same
    gradients, since Adam's update does not see a constant offset.
    """
    for module in network.modules():
        if isinstance(module, nn.Linear):
            _truncated_normal(module.weight, module.in_features)
            nn.init.zeros_(module.bias)
    core = network.core
    lstm = dict(core.named_parameters())
    fan_in = core.input_size + core.hidden_size
    _truncated_normal(lstm["weight_ih_l0"], fan_in)
    _truncated_normal(lstm["weight_hh_l0"], fan_in)
    with torch.no_grad():
        lstm["bias_ih_l0"].zero_()
        # torch's gate order is input, forget, cell, output.
        lstm["bias_ih_l0"][core.hidden_size : 2 * core.hidden_size] = 1.0
        lstm["bias_hh_l0"].zero_()
    lstm["bias_hh_l0"].requires_grad_(False)


def _truncated_normal(weight: Tensor, fan_in: int) -> None:
    std = 1.0 / math.sqrt(fan_in)
    nn.init.trunc_normal_(weight, std=std, a=-2.0 * std, b=2.0 * std)


# -- acting state ----------------------------------------------------------------


@dataclass(frozen=True)
class R2D2State:
    """What acting carries from one decision to the next.

    `h` and `c` [1, 1, 512] are the LSTM state before the next decision's
    forward pass; `previous_action` the action taken at the last decision;
    `previous_reward` the survival reward of the transition since. That
    reward is None from `act` until `after_transition` supplies it, because
    only the environment knows the game time the transition took.
    """

    h: Tensor
    c: Tensor
    previous_action: int
    previous_reward: float | None


# -- backbone --------------------------------------------------------------------


@dataclass
class R2D2Backbone:
    """R2D2 (module docstring), addressed through the `Backbone` protocol."""

    config: R2D2Config
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    online: R2D2Network = field(init=False)
    target: R2D2Network = field(init=False)
    optimizer: torch.optim.Adam = field(init=False)
    _steps: int = field(default=0, init=False)
    #: The stream every exploratory draw comes from; `acting_copy` reseeds it per actor.
    _random: random.Random = field(init=False)

    def __post_init__(self) -> None:
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.online = R2D2Network(self.network_config).to(self.device)
        self.target = R2D2Network(self.network_config).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            [parameter for parameter in self.online.parameters() if parameter.requires_grad],
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )
        self._random = random.Random(self.config.seed)

    @property
    def model_version(self) -> int:
        return self._steps

    # -- acting ------------------------------------------------------------------

    def initial_state(self) -> R2D2State:
        """The zero LSTM state, action 0 and reward 0: Acme's episode start."""
        zeros = torch.zeros(1, 1, R2D2_STATE_SIZE, device=self.device)
        return R2D2State(zeros, zeros.clone(), 0, 0.0)

    def act(
        self, features: StateFeatures, state: R2D2State, *, epsilon: float
    ) -> tuple[int, R2D2State]:
        """Epsilon-greedy over the valid actions; the state after this decision's forward pass."""
        valid = [index for index, allowed in enumerate(features.mask) if allowed]
        if not valid:
            raise ValueError("no action is available in this state")
        if state.previous_reward is None:
            raise ValueError("the last transition's game time was never given (`after_transition`)")
        device = self.device
        scalars = torch.tensor([[features.scalars]], dtype=torch.float32, device=device)
        rows = torch.tensor([[features.rows]], dtype=torch.float32, device=device)
        mask = torch.tensor([[features.mask]], dtype=torch.bool, device=device)
        with torch.no_grad():
            q, (h, c) = self.online(
                scalars,
                rows.view(1, 1, ROW_COUNT, ROW_WIDTH),
                mask,
                torch.tensor([[state.previous_action]], device=device),
                torch.tensor([[state.previous_reward]], dtype=torch.float32, device=device),
                (state.h, state.c),
            )
        if epsilon > 0.0 and self._random.random() < epsilon:
            action = self._random.choice(valid)
        else:
            action = int(q[0, 0].argmax().item())
        return action, R2D2State(h, c, action, None)

    def after_transition(self, state: R2D2State, game_ms: float) -> R2D2State:
        """The state with the survival reward of a transition of `game_ms` as its previous reward.

        Computed as the learner computes it from the same step's game time.
        """
        discount = game_time_discounts(
            self.config.discount_per_game_second, torch.tensor([game_ms])
        )
        reward = float(survival_rewards(discount).to(torch.float32).item())
        return R2D2State(state.h, state.c, state.previous_action, reward)

    def replay_entry(self, state: R2D2State) -> numpy.ndarray:
        """The LSTM state as replay stores it: float32 [2, 512], (h, c).

        An actor takes it before the decisions at 0, 40, 80, ... of an
        episode, for `R2D2Replay.add`'s `states`.
        """
        return torch.stack((state.h[0, 0], state.c[0, 0])).float().cpu().numpy()

    # -- learning ----------------------------------------------------------------

    def learn(self, batch: SequenceBatch) -> LearnMetrics:
        """One optimisation step on a batch of `R2D2Replay` items (module docstring)."""
        if batch.context is None or batch.previous_actions is None:
            raise ValueError("R2D2 learns from R2D2Replay items, with stored states")
        burn_in = batch.burn_in
        # Step layout: `game_ms`, `dones` and the reward of a step describe the
        # transition into it, so the transition out of trace step t is t + 1's.
        discounts_into = game_time_discounts(self.config.discount_per_game_second, batch.game_ms)
        rewards_into = survival_rewards(discounts_into)
        inputs = (
            batch.scalars,
            batch.rows,
            batch.mask,
            batch.previous_actions,
            rewards_into.to(torch.float32),
        )
        start = (batch.context[0].unsqueeze(0), batch.context[1].unsqueeze(0))

        with torch.no_grad():
            target_q, _ = self.target(*inputs, start)
            target_q = target_q[:, burn_in:]
            online_start = start
            if burn_in:
                _, online_start = self.online(*(x[:, :burn_in] for x in inputs), start)
        online_q, _ = self.online(*(x[:, burn_in:] for x in inputs), online_start)

        mask = batch.mask[:, burn_in:]
        trace = online_q.shape[1] - 1
        with torch.no_grad():
            targets, valid = trace_targets(
                online_q.detach(),
                target_q,
                mask,
                batch.game_ms[:, burn_in:],
                batch.dones[:, burn_in:],
                batch.padding[:, burn_in:],
                discount_per_game_second=self.config.discount_per_game_second,
                n=self.config.n_step,
            )

        taken = online_q[:, :trace].gather(-1, batch.actions[:, burn_in:-1].unsqueeze(-1))
        chosen = torch.where(valid, taken.squeeze(-1), torch.zeros_like(targets))
        errors = torch.where(valid, targets - chosen, torch.zeros_like(targets))
        loss = (0.5 * errors.square().sum(dim=1) * batch.weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = torch.nn.utils.get_total_norm(
            [p.grad for p in self.online.parameters() if p.grad is not None]
        )
        self.optimizer.step()
        self._steps += 1
        if self._steps % self.config.target_update_period == 0:
            self.target.load_state_dict(self.online.state_dict())

        absolute = errors.detach().abs()
        real = valid.to(absolute.dtype)
        with torch.no_grad():
            fit = value_fit_correlation(
                signed_parabolic(online_q.detach()[:, :trace]),
                mask[:, :trace],
                rewards_into[:, burn_in + 1 :].to(torch.float32),
                batch.dones[:, burn_in + 1 :],
                real,
                discounts=discounts_into[:, burn_in + 1 :],
            )
        taken_values = signed_parabolic(chosen.detach()[valid])
        return LearnMetrics(
            weighted_loss=float(loss.detach().item()),
            unweighted_mean_absolute_td_error=float(
                (absolute.sum() / real.sum().clamp(min=1.0)).item()
            ),
            gradient_norm=float(gradient_norm.item()),
            td_errors=real_step_td_errors(absolute, real),
            value_fit_correlation=fit,
            taken_q_max=float(taken_values.max().item()) if taken_values.numel() else None,
        )

    # -- persistence -------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Everything a resume needs: both networks, Adam's state and the learner step."""
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

    def network_state_dict(self) -> dict[str, Any]:
        """What acting reads: the online network and the step it is from; no optimizer state."""
        return {"online": self.online.state_dict(), "steps": self._steps}

    def load_network_state_dict(self, state: dict[str, Any]) -> None:
        """Load `network_state_dict` into an acting copy."""
        self.online.load_state_dict(state["online"])
        self._steps = int(state["steps"])
