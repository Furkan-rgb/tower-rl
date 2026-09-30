"""DreamerV3 (Hafner et al. 2023, arXiv:2301.04104) as a backbone.

A world model - an RSSM over the encoded observation - learned from replayed
sequences, and an actor and critic learned in the world model's imagination.
It is a port of the official code (danijar/dreamerv3 at e3f02248: `agent.py`,
`rssm.py`, `embodied/jax/*`) at its `size12m` preset, in PyTorch. On CUDA it
learns in bfloat16 compute, the official `compute_dtype`, with float32
parameters, optimiser state, distributions and losses, and its recurrent steps
compiled; on the CPU, in float32 and eager. Every value, and every place this
differs from the official code, is
in `docs/solution.md` 9.4c; the differences are the ones this project's replay,
action mask and acting path require.

It is one `Backbone` like any other: the actor, `TrainingRun`, the checkpoint
format and the evaluator are the ones every backbone uses. Its replay is its
own, `DreamerReplay`, the official step replay: windows start at every step,
run across episodes, and start from the latent stored for the step before
them, which the learner writes back after every update; an episode's last
step is its final observation. So a window is read exactly as the official
`Agent.train` reads one, reward and continue losses at its first step
included.
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
from tower_rl.learning.dreamer_replay import DREAMER_REPLAY_CONTEXT
from tower_rl.learning.value_learning import (
    game_time_discounts,
    real_step_td_errors,
    survival_rewards,
    value_fit_correlation,
)

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
    #: 0: `configs.yaml` lists `policy.unimix: 0.01`, but the discrete policy
    #: head `heads.py` `Head.categorical` (104-113) builds `outs.Categorical(logits)`
    #: without it, so the official actor mixes in nothing. A checkpoint from
    #: before recorded 0.01 and still acts with it.
    actor_unimix: float = 0.0
    #: True: the action mask is a boolean key, which the official code treats
    #: as discrete with two classes (`elements/space.py` 15-16, 42-43): the
    #: encoder takes it one-hot (`nets.py` 488-493) and the decoder predicts
    #: it with a two-class categorical head per action (`rssm.py` 299-300),
    #: read by its argmax. False is the 0/1 input and binary head a checkpoint
    #: from before recorded no key for; it still acts that way.
    mask_one_hot: bool = True
    actor_outscale: float = 0.01
    free_nats: float = 1.0
    # Loop geometry (configs.yaml batch_size, batch_length; train_ratio per
    # Table 2 (Proprio/Visual Control, 500K-1M-step budget, 12M model), which
    # matches the code's crafter preset: run.steps 1.1e6, envs 1, train_ratio 512).
    batch_size: int = 16
    batch_length: int = 64
    #: Replayed steps per collected decision.
    train_ratio: float = 512.0
    imagination_horizon: int = 15
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
    # The task, not the algorithm (ADR 0013): held identical across learners.
    #: Discount per second of game time, in place of the official per-step
    #: `horizon: 333`. A transition that spans t game-seconds is discounted by
    #: this ** t, which the continue target carries (`contdisc`); a purchase
    #: spans none and is not discounted (docs/solution.md 9.4c, 9.4d). Not a
    #: code default: the protocol's is 0.999, and the run sets it. None only
    #: rebuilds, to act, a policy from a checkpoint from before it; `learn`
    #: refuses it.
    discount_per_game_second: float | None = None
    #: Learn from game time survived, (1 - d) * V_REF, instead of the wave
    #: reward: `value_learning.survival_rewards` (docs/solution.md 9.4e).
    survival_time_reward: bool = False

    def __post_init__(self) -> None:
        if self.discount_per_game_second is not None and not (
            0.0 < self.discount_per_game_second < 1.0
        ):
            raise ValueError("discount per game-second must be within (0, 1)")
        if self.survival_time_reward and self.discount_per_game_second is None:
            raise ValueError("the survival-time reward needs a discount per game-second")
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


class _RMSNorm(nn.RMSNorm):
    """`nets.Norm('rms')`: computed in float32 and returned in the input's dtype."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).to(x.dtype)


def _mlp(inputs: int, units: int, layers: int) -> nn.Sequential:
    """`nets.MLP`: linear, RMS norm, SiLU, per layer."""
    modules: list[nn.Module] = []
    for _ in range(layers):
        modules += [_linear(inputs, units), _RMSNorm(units, eps=1e-4), nn.SiLU()]
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
        # The bias in the product's dtype, as `nets.BlockLinear` casts it.
        product = torch.einsum("...gi,gio->...go", x, self.weight)
        return product + self.bias.to(product.dtype)


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
            [_RMSNorm(c.deter, eps=1e-4) for _ in range(c.dynamics_layers)]
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

    # Logits leave in float32, as `outs.Categorical` takes them.
    def prior_logits(self, deter: Tensor) -> Tensor:
        logits: Tensor = self.prior(deter).float()
        return logits.reshape(*logits.shape[:-1], self.config.stoch, self.config.classes)

    def posterior_logits(self, deter: Tensor, tokens: Tensor) -> Tensor:
        logits: Tensor = self.posterior(torch.cat((deter, tokens), -1)).float()
        return logits.reshape(*logits.shape[:-1], self.config.stoch, self.config.classes)


class WorldModel(nn.Module):
    """Encoder, RSSM, decoder, reward and continue heads: everything the model loss trains."""

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        c = config
        feature = c.deter + c.stoch * c.classes
        self.mask_classes = 2 if c.mask_one_hot else 1
        self.encoder = _mlp(
            SCALAR_COUNT + ROWS_WIDTH + ACTIONS * self.mask_classes, c.units, c.encoder_layers
        )
        self.dynamics = _Dynamics(c)
        self.decoder = _mlp(feature, c.units, c.decoder_layers)
        self.decode_scalars = _linear(c.units, SCALAR_COUNT)
        self.decode_rows = _linear(c.units, ROWS_WIDTH)
        self.decode_mask = _linear(c.units, ACTIONS * self.mask_classes)
        self.reward = _head(feature, c.units, c.reward_layers, c.bins, 0.0)
        self.cont = _head(feature, c.units, c.continue_layers, 1, 1.0)

    def encode(self, scalars: Tensor, rows: Tensor, mask: Tensor) -> Tensor:
        """Symlog the float keys and concatenate, as `DictConcat` does; the mask one-hot."""
        if self.mask_classes == 2:
            encoded = functional.one_hot(mask.long(), 2).flatten(-2).to(scalars.dtype)
        else:
            encoded = mask.to(scalars.dtype)
        flat = torch.cat((symlog(scalars), symlog(rows.flatten(-2)), encoded), -1)
        tokens: Tensor = self.encoder(flat)
        return tokens

    def mask_logits(self, decoded: Tensor) -> Tensor:
        """The decoder's mask logits, float32: [..., ACTIONS, 2], or [..., ACTIONS] if 0/1."""
        logits: Tensor = self.decode_mask(decoded).float()
        if self.mask_classes == 2:
            return logits.reshape(*logits.shape[:-1], ACTIONS, 2)
        return logits

    def mask_loss(self, logits: Tensor, mask: Tensor) -> Tensor:
        """The mask's reconstruction loss, summed over actions as `outs.Agg` sums a key."""
        if self.mask_classes == 2:
            chosen = logits.log_softmax(-1).gather(-1, mask.long().unsqueeze(-1))
            return -chosen.squeeze(-1).sum(-1)
        return functional.binary_cross_entropy_with_logits(
            logits, mask.to(torch.float32), reduction="none"
        ).sum(-1)

    def valid_actions(self, logits: Tensor) -> Tensor:
        """The decoded mask read as a mask: its argmax (logit > 0 if 0/1), WAIT always valid."""
        valid = logits.argmax(-1).bool() if self.mask_classes == 2 else logits > 0.0
        valid[..., WAIT_INDEX] = True
        return valid

    def decoded_mask(self, feature: Tensor) -> Tensor:
        """The mask the model believes a state has, WAIT always included."""
        return self.valid_actions(self.mask_logits(self.decoder(feature)))


@dataclass(frozen=True)
class _LossParts:
    """What `learn` reads off one loss besides the loss itself, over the trained steps."""

    replay_returns: Tensor  # [B, T - 1]
    replay_value: Tensor  # [B, T]
    replay_weight: Tensor  # [B, T - 1]
    reward_in: Tensor
    terminal_in: Tensor
    discount_in: Tensor
    reset: Tensor
    last: Tensor
    #: The posterior latents, written back into replay.
    deter: Tensor
    stoch: Tensor
    checks: dict[str, Tensor]


# -- the backbone --------------------------------------------------------------


@dataclass
class DreamerBackbone:
    """DreamerV3: a world model, and an actor and critic trained in its imagination."""

    config: DreamerConfig = field(default_factory=DreamerConfig)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    #: Learn in bfloat16 compute, the official `compute_dtype`: parameters,
    #: optimiser state, distributions, losses and the return normaliser stay
    #: float32. None: on CUDA only.
    mixed_precision: bool | None = None
    #: Compile the two recurrent loops of `learn` and replay them as CUDA
    #: graphs (`_compiled_observe`, `_compiled_imagine_rollout`). None: on CUDA only.
    compiled: bool | None = None

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
        on_cuda = self.device.type == "cuda"
        if self.mixed_precision is None:
            self.mixed_precision = on_cuda
        if self.compiled is None:
            self.compiled = on_cuda
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

    def replay_entry(self, state: tuple[Tensor, Tensor, Tensor]) -> tuple[Any, Any]:
        """The latent `act` just reached, as replay stores it: the official `dyn` entry.

        `deter` as float32 [deter] and the stochastic sample as its class
        indices [stoch], which hold the one-hot exactly (`DreamerReplay`).
        """
        c = self.config
        deter, stoch, _ = state
        indices = stoch[0].detach().view(c.stoch, c.classes).argmax(-1)
        return (
            deter[0].detach().float().cpu().numpy(),
            indices.to(torch.int8).cpu().numpy(),
        )

    # -- learning --------------------------------------------------------------

    def learn(self, batch: SequenceBatch) -> LearnMetrics:
        """One update of world model, actor and critic on one batch: `Agent.train`.

        The batch is `DreamerReplay`'s: `length` = context + batch length steps
        in Dreamer's layout, the window starting from the stored latent of its
        context step (`_apply_replay_context`).
        """
        c = self.config
        if batch.context is None or batch.first is None or batch.last is None:
            raise ValueError("DreamerV3 learns from its step replay, whose windows carry latents")
        if batch.burn_in != DREAMER_REPLAY_CONTEXT:
            raise ValueError(f"DreamerV3's replay context is {DREAMER_REPLAY_CONTEXT} step")
        if c.discount_per_game_second is None:
            raise ValueError("DreamerV3 learns under the game-time discount, and has none")
        length = int(batch.scalars.shape[1]) - batch.burn_in
        if (batch.batch_size, length) != (c.batch_size, c.batch_length):
            raise ValueError(
                f"DreamerV3 trains on {c.batch_size} x {c.batch_length} batches, "
                f"not {batch.batch_size} x {length}"
            )
        if self.compiled:
            # A new iteration for the CUDA graphs: the last update's graph
            # outputs are no longer read.
            torch.compiler.cudagraph_mark_step_begin()  # type: ignore[no-untyped-call]
        # Each transition's own d and the reward it carries, valued at its
        # start: exactly stacked-dqn's (`StackedDqnBackbone.learn`), for the
        # transition into each step. The survival-time reward replaces the
        # wave change; the wave change, booked where its span ends, is d * r.
        discounts = game_time_discounts(c.discount_per_game_second, batch.game_ms)
        if c.survival_time_reward:
            step_rewards = survival_rewards(discounts).to(batch.rewards.dtype)
        else:
            step_rewards = batch.rewards * discounts.to(batch.rewards.dtype)
        discounts = discounts.to(batch.rewards.dtype)
        with torch.autocast(self.device.type, torch.bfloat16, enabled=bool(self.mixed_precision)):
            loss, parts = self._loss(batch, discounts, step_rewards)

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

        returns, value, weight = parts.replay_returns, parts.replay_value, parts.replay_weight
        errors = (returns - value[:, :-1]).abs() * weight
        td_errors = tuple(sequence or (0.0,) for sequence in real_step_td_errors(errors, weight))
        with torch.no_grad():
            # The realised return from each step on, in replay's own layout:
            # the transition out of step t is the one into t + 1, and a step
            # followed by `is_first` ends its chain. The value is V, so every
            # action reads it.
            ends = parts.terminal_in[:, 1:].bool() | parts.reset[:, 1:]
            fit = value_fit_correlation(
                value[:, :-1, None].expand(-1, -1, ACTIONS),
                torch.ones(*value[:, :-1].shape, ACTIONS, dtype=torch.bool, device=self.device),
                parts.reward_in[:, 1:],
                ends,
                (~parts.last[:, :-1]).to(torch.float32),
                discounts=parts.discount_in[:, 1:],
            )
            measured = {
                name: value
                for name, value in zip(
                    parts.checks, torch.stack(list(parts.checks.values())).tolist(), strict=True
                )
                if not math.isnan(value)
            }
            latents = (
                parts.deter.detach().float().cpu(),
                parts.stoch.detach()
                .view(*parts.stoch.shape[:2], c.stoch, c.classes)
                .argmax(-1)
                .to(torch.int8)
                .cpu(),
            )
        # A ratio of two means, reported only where the batch has game time.
        if measured["dreamer_true_dt_seconds"] > 0.0:
            measured["dreamer_implied_to_true_dt"] = (
                measured["dreamer_implied_dt_seconds"] / measured["dreamer_true_dt_seconds"]
            )
        return LearnMetrics(
            weighted_loss=float(loss.detach().item()),
            unweighted_mean_absolute_td_error=float(
                (errors.sum() / weight.sum().clamp(min=1.0)).item()
            ),
            gradient_norm=float(gradient_norm.item()),
            td_errors=td_errors,
            value_fit_correlation=fit,
            diagnostics=measured,
            latents=latents,
        )

    def _loss(
        self, batch: SequenceBatch, discounts: Tensor, step_rewards: Tensor
    ) -> tuple[Tensor, _LossParts]:
        """The whole loss of one update: `Agent.loss` after `_apply_replay_context`.

        `discounts` and `step_rewards` are each stored transition's d and
        reward, for the transition into each step. Every head's output is
        taken to float32 before it enters a distribution or a loss, as the
        official `outs.py` does, so under bfloat16 compute only the networks'
        own layers run in bfloat16. Every loss is a mean over all its entries,
        as the official `v.mean()` is.
        """
        c = self.config
        world = self.world_model
        dynamics = world.dynamics
        context, size = batch.burn_in, batch.batch_size
        length = c.batch_length
        assert batch.context is not None and batch.first is not None and batch.last is not None
        # `_apply_replay_context`: the carry is the context step's stored
        # latent; the window is every later step, each after the action taken
        # at the step before it.
        deter0 = batch.context[0].to(torch.float32)
        stoch0 = functional.one_hot(batch.context[1], c.classes).flatten(-2).to(torch.float32)
        previous = functional.one_hot(batch.actions[:, context - 1 : -1], ACTIONS).to(
            torch.float32
        )
        scalars, rows, mask = (
            batch.scalars[:, context:],
            batch.rows[:, context:],
            batch.mask[:, context:],
        )
        reward_in = step_rewards[:, context:]
        terminal_in = batch.dones[:, context:].to(torch.float32)
        discount_in = discounts[:, context:]
        seconds_in = batch.game_ms[:, context:] / 1000.0
        reset = batch.first[:, context:]
        last = batch.last[:, context:]

        # -- world model --
        tokens = world.encode(scalars, rows, mask)
        uniform = torch.rand(length, size, c.stoch, device=self.device)
        observe = _compiled_observe if self.compiled else DreamerBackbone._observe
        deter_all, stoch_all, posterior = observe(
            self, tokens, previous, reset, uniform, deter0, stoch0
        )
        feature = torch.cat((deter_all, stoch_all), -1)

        prior = unimix_probs(dynamics.prior_logits(deter_all), c.latent_unimix)
        dynamics_loss = categorical_kl(posterior.detach(), prior).sum(-1).clamp(min=c.free_nats)
        representation_loss = (
            categorical_kl(posterior, prior.detach()).sum(-1).clamp(min=c.free_nats)
        )
        decoded = world.decoder(feature)
        scalars_loss = (world.decode_scalars(decoded).float() - symlog(scalars)).square().sum(-1)
        rows_loss = (
            (world.decode_rows(decoded).float() - symlog(rows.flatten(-2))).square().sum(-1)
        )
        mask_logits = world.mask_logits(decoded)
        mask_loss = world.mask_loss(mask_logits, mask)
        reward_loss = twohot_loss(world.reward(feature).float(), reward_in, self._bins)
        # contdisc, per transition: the continue head predicts (1 - terminal)
        # * gamma_s ** t, 1 for a purchase, where the official target is
        # (1 - terminal) * (1 - 1 / horizon).
        continue_target = (1.0 - terminal_in) * discount_in
        continue_logits = world.cont(feature).float().squeeze(-1)
        continue_loss = functional.binary_cross_entropy_with_logits(
            continue_logits, continue_target, reduction="none"
        )

        # -- imagination, from every posterior state (`imag_last: 0`) --
        starts = size * length
        with torch.no_grad():
            imagined, actions, masks = self._imagine(
                deter_all.reshape(starts, -1), stoch_all.reshape(starts, -1)
            )
            imagined_reward = twohot_mean(world.reward(imagined).float(), self._bins)
            imagined_continue = torch.sigmoid(world.cont(imagined).float()).squeeze(-1)
            slow_value = twohot_mean(self.slow_critic(imagined).float(), self._bins)
        log_probs = masked_policy_log_probs(
            self.actor(imagined[:, :-1]).float(), masks, c.actor_unimix
        )
        value_logits = self.critic(imagined).float()
        value = twohot_mean(value_logits, self._bins).detach()
        # contdisc: the continue head already carries the discount, so the
        # return applies none of its own on top of it.
        returns = lambda_return(
            torch.zeros_like(imagined_continue),
            1.0 - imagined_continue,
            imagined_reward,
            value,
            1.0,
            c.return_lambda,
        )
        self.return_normaliser.update(returns)
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

        # -- `repl_loss`: the critic on replayed states, bootstrapped by imagined returns --
        replay_logits = self.critic(feature).float()
        replay_value = twohot_mean(replay_logits, self._bins).detach()
        with torch.no_grad():
            replay_slow = twohot_mean(self.slow_critic(feature).float(), self._bins)
        boot = returns[:, 0].reshape(size, length)
        # The stored transitions' own d, where the official code has its
        # constant `disc`; `terminal_in` zeroes the bootstrap past a death.
        replay_returns = lambda_return(
            last, terminal_in, reward_in, boot, discount_in, c.return_lambda
        )
        replay_weight = (~last[:, :-1]).to(torch.float32)
        replay_loss = replay_weight * (
            twohot_loss(replay_logits[:, :-1], replay_returns, self._bins)
            + c.slow_regulariser
            * twohot_loss(replay_logits[:, :-1], replay_slow[:, :-1], self._bins)
        )

        loss = (
            c.dynamics_scale * dynamics_loss.mean()
            + c.representation_scale * representation_loss.mean()
            + c.reconstruction_scale * (scalars_loss.mean() + rows_loss.mean() + mask_loss.mean())
            + c.reward_scale * reward_loss.mean()
            + c.continue_scale * continue_loss.mean()
            + c.policy_scale * policy_loss.mean()
            + c.value_scale * value_loss.mean()
            + c.replay_value_scale * replay_loss.mean()
        )
        with torch.no_grad():
            checks = self._checks(
                continue_logits, continue_target, seconds_in, terminal_in,
                (~reset).to(torch.float32), mask_logits, mask,
            )
        return loss, _LossParts(
            replay_returns=replay_returns,
            replay_value=replay_value,
            replay_weight=replay_weight,
            reward_in=reward_in,
            terminal_in=terminal_in,
            discount_in=discount_in,
            reset=reset,
            last=last,
            deter=deter_all,
            stoch=stoch_all,
            checks=checks,
        )

    def _checks(
        self,
        continue_logits: Tensor,
        continue_target: Tensor,
        seconds_in: Tensor,
        terminal_in: Tensor,
        transition: Tensor,
        mask_logits: Tensor,
        mask: Tensor,
    ) -> dict[str, Tensor]:
        """Whether the continue head has learned game time, and the decoded mask the true one.

        From tensors the loss already holds. The continue checks read every
        transition but the one into an episode's first step (`transition`),
        which is no transition at all.
        The continue head carries d = gamma_s ** t, so on a real transition
        that did not end the episode it implies a game time log(c) / log
        gamma_s, read against the one stored. The decoded mask is what
        imagination samples under: a false valid lets the actor be credited
        for an action the game would refuse.
        """
        per_second = self.config.discount_per_game_second
        assert per_second is not None  # `learn` refuses a config without one
        live = transition * (1.0 - terminal_in)
        implied = functional.logsigmoid(continue_logits) / math.log(per_second)
        # Per transition, where a relative error is defined: at least 1 s of
        # game time. NaN, which `learn` leaves unreported, where none is.
        timed = live * (seconds_in >= 1.0).to(live.dtype)
        relative = (implied - seconds_in).abs() / seconds_in.clamp(min=1.0)
        relative_error = torch.where(
            timed.sum() > 0, _mean(relative, timed), torch.full_like(timed.sum(), math.nan)
        )
        decoded = self.world_model.valid_actions(mask_logits)
        return {
            "dreamer_implied_dt_seconds": _mean(implied, live),
            "dreamer_true_dt_seconds": _mean(seconds_in, live),
            "dreamer_implied_dt_relative_error": relative_error,
            "dreamer_predicted_continue": _mean(torch.sigmoid(continue_logits), transition),
            "dreamer_true_continue": _mean(continue_target, transition),
            "dreamer_mask_false_valid_rate": (
                # 1 - precision: of the entries imagination may sample, the
                # share the game would refuse. Not over every invalid entry,
                # which rows locked all run would dilute.
                (decoded & ~mask).sum() / decoded.sum().clamp(min=1)
            ).float(),
            "dreamer_mask_false_invalid_rate": (
                (~decoded & mask).sum() / mask.sum().clamp(min=1)
            ).float(),
        }

    def _observe(
        self,
        tokens: Tensor,
        previous: Tensor,
        reset: Tensor,
        uniform: Tensor,
        deter: Tensor,
        stoch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The posterior states over the window from the context's latent: `RSSM.observe`.

        Every draw comes in as `uniform` [T, B, stoch], from torch's stream.
        `deter` and `stoch` [B, *] are the carry the window starts from; a step
        marked `reset` starts from zeros instead, as `is_first` does.

        Returns the deterministic and stochastic states [B, T, *] and the
        posterior probabilities [B, T, stoch, classes].
        """
        c = self.config
        dynamics = self.world_model.dynamics
        deters, stochs, posteriors = [], [], []
        for step in range(tokens.shape[1]):
            keep = (~reset[:, step]).to(torch.float32)[:, None]
            deter = dynamics.core(deter * keep, stoch * keep, previous[:, step] * keep)
            logits = dynamics.posterior_logits(deter, tokens[:, step])
            posteriors.append(unimix_probs(logits, c.latent_unimix))
            stoch = self._latent(logits, uniform[step])
            deters.append(deter)
            stochs.append(stoch)
        return torch.stack(deters, 1), torch.stack(stochs, 1), torch.stack(posteriors, 1)

    def _imagine(self, deter: Tensor, stoch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Roll the prior forward under the policy: `RSSM.imagine` with `policyfn`.

        Returns the imagined features [n, horizon + 1, f], the actions taken
        from each but the last [n, horizon] and the decoded masks they were
        sampled under [n, horizon]. Called without gradients: nothing the
        official loss differentiates flows through the imagined states.
        """
        c = self.config
        count = deter.shape[0]
        # Every step's draws up front, in the order a step takes them: the
        # action's, then the latent's.
        draws = [
            (torch.rand(count, device=self.device), torch.rand(count, c.stoch, device=self.device))
            for _ in range(c.imagination_horizon)
        ]
        action_uniform = torch.stack([action for action, _ in draws])
        latent_uniform = torch.stack([latent for _, latent in draws])
        rollout = _compiled_imagine_rollout if self.compiled else DreamerBackbone._imagine_rollout
        return rollout(self, deter, stoch, action_uniform, latent_uniform)

    def _imagine_rollout(
        self, deter: Tensor, stoch: Tensor, action_uniform: Tensor, latent_uniform: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """`_imagine` given its uniforms [horizon, n] and [horizon, n, stoch]."""
        c = self.config
        features = [torch.cat((deter, stoch), -1)]
        actions, masks = [], []
        for step in range(c.imagination_horizon):
            mask = self.world_model.decoded_mask(features[-1])
            probs = masked_policy_log_probs(
                self.actor(features[-1]).float(), mask, c.actor_unimix
            ).exp()
            action = sample_index(probs, action_uniform[step])
            deter = self.world_model.dynamics.core(
                deter, stoch, functional.one_hot(action, ACTIONS).to(torch.float32)
            )
            logits = self.world_model.dynamics.prior_logits(deter)
            stoch = self._latent(logits, latent_uniform[step])
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

    def network_state_dict(self) -> dict[str, Any]:
        """Every network and the return scale, as `state_dict` holds them, without the optimizer."""
        state = self.state_dict()
        del state["optimizer"]
        return state

    def load_network_state_dict(self, state: dict[str, Any]) -> None:
        self.world_model.load_state_dict(state["world_model"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.slow_critic.load_state_dict(state["slow_critic"])
        self.return_normaliser.load_state_dict(state["return_normaliser"])
        self._steps = int(state["steps"])

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.world_model.load_state_dict(state["world_model"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.slow_critic.load_state_dict(state["slow_critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.return_normaliser.load_state_dict(state["return_normaliser"])
        self._steps = int(state["steps"])


# The two recurrent loops of `learn`, compiled whole on their first call (on
# CUDA, by default) and replayed as CUDA graphs ("reduce-overhead"). At batch 16
# the learner is bound by launching thousands of small kernels, not by
# arithmetic; fused and graphed, the loops launch a few. Every draw is made
# outside them, from torch's stream, so compiling changes no sample. Functions
# are compiled, not modules, so every state_dict key stays the module's own.
# The first call compiles for minutes; inductor's on-disk cache makes a later
# process's first call take seconds.
_compiled_observe = torch.compile(
    DreamerBackbone._observe, fullgraph=True, dynamic=False, mode="reduce-overhead"
)
_compiled_imagine_rollout = torch.compile(
    DreamerBackbone._imagine_rollout, fullgraph=True, dynamic=False, mode="reduce-overhead"
)


def _mean(values: Tensor, weight: Tensor) -> Tensor:
    """The mean over the entries a weight keeps: padding contributes nothing."""
    return (values * weight).sum() / weight.sum().clamp(min=1.0)
