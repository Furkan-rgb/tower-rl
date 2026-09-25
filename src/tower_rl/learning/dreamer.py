"""DreamerV3 (Hafner et al. 2023, arXiv:2301.04104) as a backbone.

A world model - an RSSM over the encoded observation - learned from replayed
sequences, and an actor and critic learned in the world model's imagination.
It is a port of the official code (danijar/dreamerv3 at e3f02248: `agent.py`,
`rssm.py`, `embodied/jax/*`) at its `size12m` preset, in PyTorch and in
float32. Every value, and every place this differs from the official code, is
in `docs/solution.md` 9.4c; the differences are the ones this project's replay,
action mask and acting path require.

It is one `Backbone` like any other: replay, the actor, `TrainingRun`, the
checkpoint format and the evaluator are the ones every backbone uses. Replay
stores a step's own action and the reward and termination of the transition
out of it; Dreamer's layout carries the action, reward and termination of the
transition *into* a step. `learn` converts between the two, and that shift is
where the three layout rules come from:

- a window's first step has no known reward or termination, so its reward and
  continue losses are masked;
- a window that ends an episode has no terminal observation (replay does not
  store one), so a phantom step is predicted from the last state and action by
  the prior alone, and trains only the reward and continue heads;
- a window starts from the zero state (the official `replay_context: 0` path),
  as does the first real step after a short episode's front padding, which is
  the episode's first step.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS, WAIT, action_index
from tower_rl.learning.backbone import LearnMetrics, SequenceBatch
from tower_rl.learning.dreamer_math import (
    LaProp,
    PercentileNormaliser,
    categorical_kl,
    lambda_return,
    masked_entropy,
    masked_policy_log_probs,
    sample_index,
    symlog,
    twohot_bins,
    twohot_loss,
    twohot_mean,
    unimix_probs,
)
from tower_rl.learning.value_learning import real_step_td_errors, value_fit_correlation

#: The name a run, its checkpoints and `--backbone` file this backbone under.
DREAMERV3 = "dreamerv3"
ACTIONS = len(RUN_ACTIONS)
#: The one action every active run allows, so a decoded mask always keeps it.
WAIT_INDEX = action_index(WAIT)
ROWS_WIDTH = ROW_COUNT * ROW_WIDTH


@dataclass(frozen=True)
class DreamerConfig:
    """Every DreamerV3 setting, at the official values (`configs.yaml`).

    `size12m` for the network sizes; the batch and the train ratio of the
    `atari100k` preset, the official low-data, one-environment setting.
    `docs/solution.md` 9.4c cites each and records every departure.
    """

    # size12m (configs.yaml `size12m`).
    deter: int = 2048
    hidden: int = 256
    classes: int = 16
    units: int = 256
    stoch: int = 32
    blocks: int = 8
    # Layer counts: rssm imglayers/obslayers/dynlayers; enc, dec, heads.
    prior_layers: int = 2
    posterior_layers: int = 1
    dynamics_layers: int = 1
    encoder_layers: int = 3
    decoder_layers: int = 3
    reward_layers: int = 1
    continue_layers: int = 1
    actor_layers: int = 3
    critic_layers: int = 3
    bins: int = 255
    latent_unimix: float = 0.01
    #: The paper's 1%; the code lists it but its categorical head never applies it.
    actor_unimix: float = 0.01
    actor_outscale: float = 0.01
    free_nats: float = 1.0
    # Loop geometry (configs.yaml batch_size, batch_length; atari100k train_ratio).
    batch_size: int = 16
    batch_length: int = 64
    #: Replayed steps per collected decision.
    train_ratio: float = 256.0
    imagination_horizon: int = 15
    horizon: int = 333
    return_lambda: float = 0.95
    entropy_scale: float = 3e-4
    slow_critic_rate: float = 0.02
    slow_regulariser: float = 1.0
    return_norm_rate: float = 0.01
    return_norm_limit: float = 1.0
    return_norm_low: float = 5.0
    return_norm_high: float = 95.0
    # loss_scales; `rec` applies to every decoded key.
    reconstruction_scale: float = 1.0
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    dynamics_scale: float = 1.0
    representation_scale: float = 0.1
    policy_scale: float = 1.0
    value_scale: float = 1.0
    replay_value_scale: float = 0.3
    # opt: LaProp with AGC and a linear warm-up.
    learning_rate: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-20
    agc: float = 0.3
    agc_floor: float = 1e-3
    warmup: int = 1000
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.deter % self.blocks:
            raise ValueError("the deterministic state must split evenly into blocks")
        if self.hidden < 1 or self.units < 1:
            raise ValueError("hidden and units must be positive")
        if self.bins % 2 == 0:
            raise ValueError("the twohot bins are symmetric about a centre bin")

    @property
    def gradient_steps_per_decision(self) -> float:
        """The train ratio as this project counts it: one step replays a whole batch."""
        return self.train_ratio / (self.batch_size * self.batch_length)

    @property
    def discount(self) -> float:
        return 1.0 - 1.0 / self.horizon


# -- layers ------------------------------------------------------------------


def _initialise(weight: Tensor, fan_in: int, outscale: float) -> None:
    """`nets.Initializer('trunc_normal', 'in')` times the layer's outscale."""
    std = 1.1368 * math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        nn.init.trunc_normal_(weight, std=std, a=-2.0 * std, b=2.0 * std)
        weight.mul_(outscale)


def _linear(inputs: int, outputs: int, *, outscale: float = 1.0) -> nn.Linear:
    layer = nn.Linear(inputs, outputs)
    _initialise(layer.weight, inputs, outscale)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(inputs: int, units: int, layers: int) -> nn.Sequential:
    """`nets.MLP`: linear, RMS norm, SiLU, per layer."""
    modules: list[nn.Module] = []
    for _ in range(layers):
        modules += [_linear(inputs, units), nn.RMSNorm(units, eps=1e-4), nn.SiLU()]
        inputs = units
    return nn.Sequential(*modules)


def _head(inputs: int, units: int, layers: int, outputs: int, outscale: float) -> nn.Sequential:
    """`MLPHead`: an MLP and one output layer."""
    return nn.Sequential(_mlp(inputs, units, layers), _linear(units, outputs, outscale=outscale))


class _BlockLinear(nn.Module):
    """`nets.BlockLinear` on input already split into its groups: [..., g, i] -> [..., g, o]."""

    def __init__(self, inputs: int, outputs: int, blocks: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(blocks, inputs // blocks, outputs // blocks))
        self.bias = nn.Parameter(torch.zeros(blocks, outputs // blocks))
        # `Initializer.compute_fans` counts the whole input as the fan-in.
        _initialise(self.weight, inputs, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...gi,gio->...go", x, self.weight) + self.bias


class _Dynamics(nn.Module):
    """The RSSM of `rssm.py`: block GRU core, prior and posterior."""

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        c = config
        self.config = c
        self.deter_in = _mlp(c.deter, c.hidden, 1)
        self.stoch_in = _mlp(c.stoch * c.classes, c.hidden, 1)
        self.action_in = _mlp(ACTIONS, c.hidden, 1)
        group = c.deter // c.blocks + 3 * c.hidden
        self.hidden_layers = nn.ModuleList(
            [_BlockLinear(group * c.blocks, c.deter, c.blocks) for _ in range(c.dynamics_layers)]
        )
        self.hidden_norms = nn.ModuleList(
            [nn.RMSNorm(c.deter, eps=1e-4) for _ in range(c.dynamics_layers)]
        )
        self.gru = _BlockLinear(c.deter, 3 * c.deter, c.blocks)
        self.prior = _head(c.deter, c.hidden, c.prior_layers, c.stoch * c.classes, 1.0)
        self.posterior = _head(
            c.deter + c.units, c.hidden, c.posterior_layers, c.stoch * c.classes, 1.0
        )

    def core(self, deter: Tensor, stoch: Tensor, action: Tensor) -> Tensor:
        """`RSSM._core`: the next deterministic state from the last state and action."""
        c = self.config
        mixed = torch.cat(
            (self.deter_in(deter), self.stoch_in(stoch), self.action_in(action)), -1
        )
        groups = deter.reshape(*deter.shape[:-1], c.blocks, c.deter // c.blocks)
        x = torch.cat((groups, mixed.unsqueeze(-2).expand(*mixed.shape[:-1], c.blocks, -1)), -1)
        for layer, norm in zip(self.hidden_layers, self.hidden_norms, strict=True):
            x = functional.silu(norm(layer(x).flatten(-2)))
            x = x.reshape(*x.shape[:-1], c.blocks, c.deter // c.blocks)
        reset, candidate, update = self.gru(x).chunk(3, -1)
        reset = torch.sigmoid(reset.flatten(-2))
        candidate = torch.tanh(reset * candidate.flatten(-2))
        update = torch.sigmoid(update.flatten(-2) - 1.0)
        return update * candidate + (1.0 - update) * deter

    def prior_logits(self, deter: Tensor) -> Tensor:
        logits: Tensor = self.prior(deter)
        return logits.reshape(*logits.shape[:-1], self.config.stoch, self.config.classes)

    def posterior_logits(self, deter: Tensor, tokens: Tensor) -> Tensor:
        logits: Tensor = self.posterior(torch.cat((deter, tokens), -1))
        return logits.reshape(*logits.shape[:-1], self.config.stoch, self.config.classes)


class WorldModel(nn.Module):
    """Encoder, RSSM, decoder, reward and continue heads: everything the model loss trains."""

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        c = config
        feature = c.deter + c.stoch * c.classes
        self.encoder = _mlp(SCALAR_COUNT + ROWS_WIDTH + ACTIONS, c.units, c.encoder_layers)
        self.dynamics = _Dynamics(c)
        self.decoder = _mlp(feature, c.units, c.decoder_layers)
        self.decode_scalars = _linear(c.units, SCALAR_COUNT)
        self.decode_rows = _linear(c.units, ROWS_WIDTH)
        self.decode_mask = _linear(c.units, ACTIONS)
        self.reward = _head(feature, c.units, c.reward_layers, c.bins, 0.0)
        self.cont = _head(feature, c.units, c.continue_layers, 1, 1.0)

    def encode(self, scalars: Tensor, rows: Tensor, mask: Tensor) -> Tensor:
        """Symlog the float keys and concatenate, as `DictConcat` does; the mask goes in as 0/1."""
        flat = torch.cat(
            (symlog(scalars), symlog(rows.flatten(-2)), mask.to(scalars.dtype)), -1
        )
        tokens: Tensor = self.encoder(flat)
        return tokens

    def decoded_mask(self, feature: Tensor) -> Tensor:
        """The mask the model believes a state has, WAIT always included."""
        logits: Tensor = self.decode_mask(self.decoder(feature))
        valid = logits > 0.0
        valid[..., WAIT_INDEX] = True
        return valid


# -- the backbone --------------------------------------------------------------


@dataclass
class DreamerBackbone:
    """DreamerV3: a world model, and an actor and critic trained in its imagination."""

    config: DreamerConfig = field(default_factory=DreamerConfig)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    world_model: WorldModel = field(init=False)
    actor: nn.Sequential = field(init=False)
    critic: nn.Sequential = field(init=False)
    #: The critic's exponential moving average, which regularises it.
    slow_critic: nn.Sequential = field(init=False)
    optimizer: LaProp = field(init=False)
    return_normaliser: PercentileNormaliser = field(init=False)
    _bins: Tensor = field(init=False)
    _steps: int = field(default=0, init=False)
    #: The stream every acting draw comes from; `acting_copy` reseeds it per actor.
    _random: random.Random = field(init=False)

    def __post_init__(self) -> None:
        c = self.config
        if c.seed is not None:
            torch.manual_seed(c.seed)
        feature = c.deter + c.stoch * c.classes
        self.world_model = WorldModel(c).to(self.device)
        self.actor = _head(feature, c.units, c.actor_layers, ACTIONS, c.actor_outscale).to(
            self.device
        )
        self.critic = _head(feature, c.units, c.critic_layers, c.bins, 0.0).to(self.device)
        self.slow_critic = _head(feature, c.units, c.critic_layers, c.bins, 0.0).to(self.device)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        self.slow_critic.requires_grad_(False)
        self.optimizer = LaProp(
            [
                *self.world_model.parameters(),
                *self.actor.parameters(),
                *self.critic.parameters(),
            ],
            lr=c.learning_rate,
            beta1=c.beta1,
            beta2=c.beta2,
            eps=c.epsilon,
            agc=c.agc,
            agc_floor=c.agc_floor,
            warmup=c.warmup,
        )
        self.return_normaliser = PercentileNormaliser(
            rate=c.return_norm_rate,
            limit=c.return_norm_limit,
            low_percentile=c.return_norm_low,
            high_percentile=c.return_norm_high,
            device=self.device,
        )
        self._bins = twohot_bins(c.bins).to(self.device)
        self._random = random.Random(c.seed)

    @property
    def model_version(self) -> int:
        return self._steps

    # -- acting ----------------------------------------------------------------

    def initial_state(self) -> tuple[Tensor, Tensor, Tensor]:
        """The zero state and no previous action: what `is_first` resets to."""
        c = self.config
        return (
            torch.zeros(1, c.deter, device=self.device),
            torch.zeros(1, c.stoch * c.classes, device=self.device),
            torch.zeros(1, ACTIONS, device=self.device),
        )

    def act(
        self, features: StateFeatures, state: tuple[Tensor, Tensor, Tensor], *, epsilon: float
    ) -> tuple[int, tuple[Tensor, Tensor, Tensor]]:
        """Sample the policy over the valid actions, as the official agent always does.

        `epsilon` is ignored: DreamerV3 adds no exploration noise; its policy's
        own entropy is its exploration, in training and in evaluation alike.
        Every draw - the latent and the action - comes from `_random`, so each
        acting copy samples from a stream of its own.
        """
        if not any(features.mask):
            raise ValueError("no action is available in this state")
        c = self.config
        deter, stoch, previous = state
        scalars = torch.tensor([features.scalars], dtype=torch.float32, device=self.device)
        rows = torch.tensor([features.rows], dtype=torch.float32, device=self.device)
        mask = torch.tensor([features.mask], dtype=torch.bool, device=self.device)
        uniform = torch.tensor(
            [self._random.random() for _ in range(c.stoch + 1)], device=self.device
        )
        with torch.no_grad():
            tokens = self.world_model.encode(scalars, rows.view(1, ROW_COUNT, ROW_WIDTH), mask)
            deter = self.world_model.dynamics.core(deter, stoch, previous)
            logits = self.world_model.dynamics.posterior_logits(deter, tokens)
            stoch = self._latent(logits, uniform[None, : c.stoch])
            log_probs = masked_policy_log_probs(
                self.actor(torch.cat((deter, stoch), -1)), mask, c.actor_unimix
            )
            action = int(sample_index(log_probs.exp(), uniform[c.stoch :]).item())
        chosen = functional.one_hot(torch.tensor([action], device=self.device), ACTIONS)
        return action, (deter, stoch, chosen.to(torch.float32))

    def _latent(self, logits: Tensor, uniform: Tensor) -> Tensor:
        """A straight-through one-hot sample of the stochastic state, flattened."""
        probs = unimix_probs(logits, self.config.latent_unimix)
        onehot = functional.one_hot(sample_index(probs, uniform), self.config.classes)
        return (onehot.to(probs.dtype) + (probs - probs.detach())).flatten(-2)

    # -- learning --------------------------------------------------------------

    def learn(self, batch: SequenceBatch) -> LearnMetrics:
        """One update of world model, actor and critic on one batch: `Agent.train`."""
        c = self.config
        if batch.burn_in != 0:
            raise ValueError("DreamerV3 starts every window from the zero state; burn-in is 0")
        length = int(batch.scalars.shape[1])
        if (batch.batch_size, length) != (c.batch_size, c.batch_length):
            raise ValueError(
                f"DreamerV3 trains on {c.batch_size} x {c.batch_length} batches, "
                f"not {batch.batch_size} x {length}"
            )
        world = self.world_model
        dynamics = world.dynamics
        size, device = batch.batch_size, self.device
        real = ~batch.padding
        ends = batch.dones[:, -1] & real[:, -1]

        # Dreamer's layout, one step longer than the window: step t carries the
        # action, reward and termination of the transition into t, and step
        # `length` is the phantom after the window's last step.
        zero = torch.zeros(size, 1, device=device)
        previous = torch.cat(
            (
                torch.zeros(size, 1, ACTIONS, device=device),
                functional.one_hot(batch.actions, ACTIONS).to(torch.float32),
            ),
            1,
        )
        reward_in = torch.cat((zero, batch.rewards), 1)
        terminal_in = torch.cat((zero, batch.dones.to(torch.float32)), 1)
        # is_first: the window's first step, and the first real step after padding.
        reset = torch.cat(
            (torch.ones(size, 1, dtype=torch.bool, device=device), batch.padding[:, :-1]), 1
        )

        # -- world model --
        tokens = world.encode(batch.scalars, batch.rows, batch.mask)
        deter = torch.zeros(size, c.deter, device=device)
        stoch = torch.zeros(size, c.stoch * c.classes, device=device)
        uniform = torch.rand(length + 1, size, c.stoch, device=device)
        deters, stochs, posteriors = [], [], []
        for step in range(length):
            keep = (~reset[:, step]).to(torch.float32)[:, None]
            deter = dynamics.core(deter * keep, stoch * keep, previous[:, step] * keep)
            logits = dynamics.posterior_logits(deter, tokens[:, step])
            posteriors.append(unimix_probs(logits, c.latent_unimix))
            stoch = self._latent(logits, uniform[step])
            deters.append(deter)
            stochs.append(stoch)
        # The phantom: the prior's prediction of the state the last action led to.
        deter = dynamics.core(deter, stoch, previous[:, length])
        deters.append(deter)
        stochs.append(self._latent(dynamics.prior_logits(deter), uniform[length]))
        deter_all = torch.stack(deters, 1)
        stoch_all = torch.stack(stochs, 1)
        feature = torch.cat((deter_all, stoch_all), -1)

        posterior = torch.stack(posteriors, 1)
        prior = unimix_probs(dynamics.prior_logits(deter_all[:, :length]), c.latent_unimix)
        dynamics_loss = categorical_kl(posterior.detach(), prior).sum(-1).clamp(min=c.free_nats)
        representation_loss = (
            categorical_kl(posterior, prior.detach()).sum(-1).clamp(min=c.free_nats)
        )
        decoded = world.decoder(feature[:, :length])
        scalars_loss = (
            (world.decode_scalars(decoded) - symlog(batch.scalars)).square().sum(-1)
        )
        rows_loss = (world.decode_rows(decoded) - symlog(batch.rows.flatten(-2))).square().sum(-1)
        mask_loss = functional.binary_cross_entropy_with_logits(
            world.decode_mask(decoded), batch.mask.to(torch.float32), reduction="none"
        ).sum(-1)
        reward_loss = twohot_loss(world.reward(feature), reward_in, self._bins)
        continue_target = (1.0 - terminal_in) * c.discount
        continue_loss = functional.binary_cross_entropy_with_logits(
            world.cont(feature).squeeze(-1), continue_target, reduction="none"
        )
        observed = real.to(torch.float32)
        # Reward and continue: not at the window's first step, whose stored
        # reward and termination belong to a step outside the window; at the
        # phantom only where it is a terminal. An episode's first step after
        # padding is trained on the filler's reward 0 and no termination, which
        # is the official `is_first` target.
        transition = torch.cat(
            (zero, observed[:, 1:], ends.to(torch.float32)[:, None]), 1
        )

        # -- imagination, from every real posterior state --
        starts = size * length
        starting = real.reshape(starts).to(torch.float32)
        with torch.no_grad():
            imagined, actions, masks = self._imagine(
                deter_all[:, :length].reshape(starts, -1),
                stoch_all[:, :length].reshape(starts, -1),
            )
            imagined_reward = twohot_mean(world.reward(imagined), self._bins)
            imagined_continue = torch.sigmoid(world.cont(imagined)).squeeze(-1)
            slow_value = twohot_mean(self.slow_critic(imagined), self._bins)
        log_probs = masked_policy_log_probs(
            self.actor(imagined[:, :-1]), masks, c.actor_unimix
        )
        value_logits = self.critic(imagined)
        value = twohot_mean(value_logits, self._bins).detach()
        # contdisc: the continue head already carries the discount.
        returns = lambda_return(
            torch.zeros_like(imagined_continue),
            1.0 - imagined_continue,
            imagined_reward,
            value,
            1.0,
            c.return_lambda,
        )
        self.return_normaliser.update(returns[real.reshape(starts)])
        _, scale = self.return_normaliser.stats()
        advantage = (returns - value[:, :-1]) / scale
        weight = torch.cumprod(imagined_continue, 1)[:, :-1]
        chosen = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = masked_entropy(log_probs, masks)
        policy_loss = (weight * -(chosen * advantage + c.entropy_scale * entropy)).mean(1)
        value_loss = (
            weight
            * (
                twohot_loss(value_logits[:, :-1], returns, self._bins)
                + c.slow_regulariser
                * twohot_loss(value_logits[:, :-1], slow_value[:, :-1], self._bins)
            )
        ).mean(1)

        # -- the critic on replayed states, bootstrapped by imagined returns --
        replay_logits = self.critic(feature)
        replay_value = twohot_mean(replay_logits, self._bins).detach()
        with torch.no_grad():
            replay_slow = twohot_mean(self.slow_critic(feature), self._bins)
        boot = torch.cat((returns[:, 0].reshape(size, length), replay_value[:, length:]), 1)
        # The phantom is always last; a window that does not end its episode is
        # cut after its own last step instead, which then has no target.
        last = torch.zeros(size, length + 1, dtype=torch.bool, device=device)
        last[:, length] = True
        last[:, length - 1] = ~ends
        replay_returns = lambda_return(
            last, terminal_in, reward_in, boot, c.discount, c.return_lambda
        )
        replay_weight = (real & ~last[:, :length]).to(torch.float32)
        replay_loss = twohot_loss(
            replay_logits[:, :length], replay_returns, self._bins
        ) + c.slow_regulariser * twohot_loss(
            replay_logits[:, :length], replay_slow[:, :length], self._bins
        )

        loss = (
            c.dynamics_scale * _mean(dynamics_loss, observed)
            + c.representation_scale * _mean(representation_loss, observed)
            + c.reconstruction_scale
            * (
                _mean(scalars_loss, observed)
                + _mean(rows_loss, observed)
                + _mean(mask_loss, observed)
            )
            + c.reward_scale * _mean(reward_loss, transition)
            + c.continue_scale * _mean(continue_loss, transition)
            + c.policy_scale * _mean(policy_loss, starting)
            + c.value_scale * _mean(value_loss, starting)
            + c.replay_value_scale * _mean(replay_loss, replay_weight)
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = torch.nn.utils.get_total_norm(
            [
                parameter.grad
                for group in self.optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
        )
        self.optimizer.step()
        self._update_slow_critic()
        self._steps += 1

        errors = (replay_returns - replay_value[:, :length]).abs() * replay_weight
        td_errors = tuple(
            sequence or (0.0,) for sequence in real_step_td_errors(errors, replay_weight)
        )
        with torch.no_grad():
            fit = value_fit_correlation(
                replay_value[:, :length, None].expand(-1, -1, ACTIONS),
                batch.mask,
                batch.rewards,
                batch.dones,
                observed,
                discount=c.discount,
            )
        return LearnMetrics(
            weighted_loss=float(loss.detach().item()),
            unweighted_mean_absolute_td_error=float(
                (errors.sum() / replay_weight.sum().clamp(min=1.0)).item()
            ),
            gradient_norm=float(gradient_norm.item()),
            td_errors=td_errors,
            value_fit_correlation=fit,
        )

    def _imagine(self, deter: Tensor, stoch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Roll the prior forward under the policy: `RSSM.imagine` with `policyfn`.

        Returns the imagined features [n, horizon + 1, f], the actions taken
        from each but the last [n, horizon] and the decoded masks they were
        sampled under [n, horizon]. Called without gradients: nothing the
        official loss differentiates flows through the imagined states.
        """
        c = self.config
        count = deter.shape[0]
        features = [torch.cat((deter, stoch), -1)]
        actions, masks = [], []
        for _ in range(c.imagination_horizon):
            mask = self.world_model.decoded_mask(features[-1])
            probs = masked_policy_log_probs(
                self.actor(features[-1]), mask, c.actor_unimix
            ).exp()
            action = sample_index(probs, torch.rand(count, device=self.device))
            deter = self.world_model.dynamics.core(
                deter, stoch, functional.one_hot(action, ACTIONS).to(torch.float32)
            )
            stoch = self._latent(
                self.world_model.dynamics.prior_logits(deter),
                torch.rand(count, c.stoch, device=self.device),
            )
            features.append(torch.cat((deter, stoch), -1))
            actions.append(action)
            masks.append(mask)
        return torch.stack(features, 1), torch.stack(actions, 1), torch.stack(masks, 1)

    def _update_slow_critic(self) -> None:
        rate = self.config.slow_critic_rate
        with torch.no_grad():
            for slow, source in zip(
                self.slow_critic.parameters(), self.critic.parameters(), strict=True
            ):
                slow.mul_(1.0 - rate).add_(source, alpha=rate)

    # -- persistence -------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "slow_critic": self.slow_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "return_normaliser": self.return_normaliser.state_dict(),
            "steps": self._steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.world_model.load_state_dict(state["world_model"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.slow_critic.load_state_dict(state["slow_critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.return_normaliser.load_state_dict(state["return_normaliser"])
        self._steps = int(state["steps"])


def _mean(values: Tensor, weight: Tensor) -> Tensor:
    """The mean over the entries a weight keeps: padding contributes nothing."""
    return (values * weight).sum() / weight.sum().clamp(min=1.0)
