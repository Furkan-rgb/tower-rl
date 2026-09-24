"""The rank-1 candidate from `docs/rl-candidates.md` 3.1.

Masked dueling double-Q learning over replay sequences, carrying time in a
window of recent run scalars rather than in a hidden state: section 2.3 of the
candidate study argues this problem is much closer to fully observed than
partially observed, and this is the backbone the project runs on.

Three things come from the Atari 100k literature the study cites: an EMA target
rather than a periodic hard copy, decoupled weight decay, and a replay ratio the
training loop supplies.

The rest of BBF (Schwarzer et al. 2023, arXiv:2305.19452) is configuration of
this same backbone rather than a second one: the discount anneal, the weight
decay mask and Adam epsilon, no gradient clipping, acting with the target, and
shrink-and-perturb resets. Every one of them defaults to run 4's recipe;
`docs/solution.md` "BBF recipe" compares each value with the official code.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, StateFeatures
from tower_rl.learning.backbone import LearnMetrics, SequenceBatch
from tower_rl.learning.network import NetworkConfig, StackedPolicyNetwork, StackedState
from tower_rl.learning.value_learning import (
    n_step_targets,
    real_step_td_errors,
    value_fit_correlation,
    weighted_sequence_loss,
)


@dataclass(frozen=True)
class StackedDqnConfig:
    #: How many run-scalar vectors the window holds, the current one included.
    #: The candidate study treats this as the tuned knob in the range 4 to 16 and
    #: names k = 1 as the ablation that settles whether history is needed at all.
    history_length: int = 8
    #: 1/(1 - discount) is the horizon in decisions. An episode here is about
    #: 121 decisions, so 0.997 (horizon 333) is effectively undiscounted and
    #: leaves the return dominated by noise far past anything the state predicts.
    discount: float = 0.99
    #: About 21.7 decisions pass per wave, and the whole reward is the wave
    #: change, so a short n-step needs several bootstrap hops to carry one wave
    #: back to the decisions that earned it.
    n_step: int = 10
    #: Where the n-step anneal ends, or None to hold `n_step` fixed. BBF
    #: (Schwarzer et al. 2023, arXiv:2305.19452) starts long for fast early
    #: credit propagation and shortens it as the value estimate becomes worth
    #: bootstrapping from; here `n_step` is where it starts.
    n_step_final: int | None = None
    #: Gradient steps the anneal from `n_step` to `n_step_final` takes, after
    #: which the final n holds. Counted in the learner's own steps, which a
    #: checkpoint carries, so a resumed run continues the schedule.
    n_step_anneal_steps: int = 0
    #: Where a discount anneal starts, or None to hold `discount` fixed. BBF
    #: anneals from 0.97 to 0.997 alongside n, over the same steps, exponentially
    #: in 1 - discount (`spr_agent.py` `exponential_decay_scheduler`, reverse);
    #: `discount` is where it ends. Needs the n-step anneal's length.
    discount_initial: float | None = None
    learning_rate: float = 1e-4
    #: Decoupled weight decay, hence AdamW rather than Adam.
    weight_decay: float = 1e-5
    #: Whether weight decay reaches one-dimensional parameters - biases and
    #: normalisation gains - as well as matrices. BBF's optimizer masks them out
    #: (`create_scaling_optimizer`, `x.ndim != 1`); run 4 decayed everything.
    weight_decay_on_vectors: bool = True
    #: Adam's epsilon: PyTorch's default, or BBF's 1.5e-4.
    adam_eps: float = 1e-8
    #: Gradient steps between shrink-and-perturb resets, or 0 for none (BBF).
    #: A reset moves the trunk halfway to a fresh initialisation, replaces the
    #: core and the heads with fresh ones, does the same to the target from an
    #: initialisation of its own, and restarts the n-step and discount anneals.
    reset_every_steps: int = 0
    #: The gradient steps the run has budgeted. No reset is taken unless more
    #: than a full `reset_every_steps` of the budget is left after it to recover
    #: in, so none falls in the final interval: BBF's `no_resets_after`.
    no_resets_after_steps: int = 0
    #: A target that follows the online network smoothly. At this replay ratio a
    #: periodic hard copy moves the target in large infrequent jumps, which is
    #: what the data-efficient recipe replaces.
    target_ema_decay: float = 0.995
    #: The global gradient norm a step is clipped to, or None for no clipping,
    #: as BBF's optimizer has none. The norm is measured and reported either way.
    gradient_clip: float | None = 10.0
    #: Act with the EMA target network rather than the online one: BBF's
    #: `target_action_selection=True`. It is the network whose weights reach
    #: the actors, a checkpoint's evaluation and the arm, since `act` is the one
    #: path all three choose actions through.
    act_with_target: bool = False
    huber_delta: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.history_length < 1:
            raise ValueError("history length must be at least one step")
        if not 0.0 < self.discount < 1.0:
            raise ValueError("discount must be within (0, 1)")
        if self.n_step < 1:
            raise ValueError("n-step must be positive")
        if (self.n_step_final is None) != (self.n_step_anneal_steps == 0):
            raise ValueError("an n-step anneal needs both a final n and a length in steps")
        if self.n_step_final is not None and self.n_step_final < 1:
            raise ValueError("the final n-step must be positive")
        if self.n_step_anneal_steps < 0:
            raise ValueError("the n-step anneal cannot take a negative number of steps")
        if not 0.0 < self.target_ema_decay < 1.0:
            raise ValueError("target EMA decay must be within (0, 1)")
        if self.discount_initial is not None:
            if not 0.0 < self.discount_initial < 1.0:
                raise ValueError("the initial discount must be within (0, 1)")
            if self.n_step_anneal_steps == 0:
                raise ValueError("a discount anneal runs over the n-step anneal's steps")
        if self.gradient_clip is not None and self.gradient_clip <= 0.0:
            raise ValueError("a gradient clip must be positive; None is no clipping")
        if self.adam_eps <= 0.0:
            raise ValueError("Adam's epsilon must be positive")
        if self.reset_every_steps < 0:
            raise ValueError("a reset interval cannot be negative")
        if self.reset_every_steps and self.no_resets_after_steps < 1:
            raise ValueError("resets need the run's budget in gradient steps")

    def n_step_at(self, gradient_steps: int) -> int:
        """The n the target is built with this many gradient steps into a cycle.

        Exponential interpolation, as in BBF: n0 * (n1 / n0) ** (t / T), rounded,
        with t held at T once the anneal is over. Fixed at `n_step` without one.
        A cycle is the whole run unless resets are on; each reset starts one.
        """
        if self.n_step_final is None:
            return self.n_step
        ratio = self.n_step_final / self.n_step
        return round(self.n_step * float(ratio ** self._anneal(gradient_steps)))

    def discount_at(self, gradient_steps: int) -> float:
        """The discount this many gradient steps into a cycle.

        The same exponential schedule as n, applied to 1 - discount, which is
        what BBF's `exponential_decay_scheduler(reverse=True)` computes.
        """
        if self.discount_initial is None:
            return self.discount
        start, end = 1.0 - self.discount_initial, 1.0 - self.discount
        return 1.0 - start * float((end / start) ** self._anneal(gradient_steps))

    def resets_after(self, gradient_steps: int) -> bool:
        """Whether a reset follows the step that brought the count to this.

        Every `reset_every_steps`, unless one interval of the budget or less
        would be left to recover in. BBF skips on `next_reset > no_resets_after
        + reset_offset`, counting environment steps with `reset_offset=1`, so its
        100k run takes resets near 36k, 76k and 116k gradient steps and skips the
        fourth. The strict bound here gives the same outcome: no reset in the
        final interval.
        """
        every = self.reset_every_steps
        return (
            every > 0
            and gradient_steps > 0
            and gradient_steps % every == 0
            and gradient_steps + every < self.no_resets_after_steps
        )

    def _anneal(self, gradient_steps: int) -> float:
        return min(gradient_steps, self.n_step_anneal_steps) / self.n_step_anneal_steps


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
    #: Gradient steps since the last reset, or since the start without one: the
    #: position the n-step and discount anneals are read at.
    _cycle_steps: int = field(default=0, init=False)
    #: Resets taken so far.
    _resets: int = field(default=0, init=False)
    #: What every reset's fresh initialisations are seeded from, so a resumed
    #: run resets to exactly the weights the uninterrupted one would have.
    _reset_seed: int = field(default=0, init=False)
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
        self.optimizer = _adamw(self.online, self.config)
        self._random = random.Random(self.config.seed)
        self._reset_seed = (
            self.config.seed
            if self.config.seed is not None
            else random.SystemRandom().getrandbits(62)
        )

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

        acting = self.target if self.config.act_with_target else self.online
        with torch.no_grad():
            q, next_state = acting(scalars, rows, mask, state)
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
        real = (~batch.padding[:, burn_in:]).to(rewards.dtype)

        online_q, _ = self.online(scalars, rows, mask, history)
        chosen = online_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Read before this step is counted, so the first step of a cycle is t = 0.
        discount = self.config.discount_at(self._cycle_steps)
        with torch.no_grad():
            target_q, _ = self.target(scalars, rows, mask, history)
            targets, learnable = n_step_targets(
                rewards,
                dones,
                online_q.detach(),
                target_q,
                mask,
                discount=discount,
                n_step=self.config.n_step_at(self._cycle_steps),
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
        if self.config.gradient_clip is None:
            gradient_norm = torch.nn.utils.get_total_norm(
                [p.grad for p in self.online.parameters() if p.grad is not None]
            )
        else:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.online.parameters(), self.config.gradient_clip
            )
        self.optimizer.step()
        self._steps += 1
        self._cycle_steps += 1
        self._update_target()
        if self.config.resets_after(self._steps):
            self._reset()

        absolute = errors.abs().detach()
        with torch.no_grad():
            fit = value_fit_correlation(
                online_q.detach(),
                mask,
                rewards,
                dones,
                real,
                discount=discount,
            )
        return LearnMetrics(
            weighted_loss=float(loss.detach().item()),
            unweighted_mean_absolute_td_error=float(
                (absolute.sum() / real.sum().clamp(min=1.0)).item()
            ),
            gradient_norm=float(gradient_norm.item()),
            td_errors=real_step_td_errors(absolute, real),
            value_fit_correlation=fit,
        )

    def _update_target(self) -> None:
        """Move the target a little way towards the online network, every step."""
        decay = self.config.target_ema_decay
        with torch.no_grad():
            for target, online in zip(
                self.target.parameters(), self.online.parameters(), strict=True
            ):
                target.mul_(decay).add_(online, alpha=1.0 - decay)
            # This network registers no buffers: `LayerNorm` here is affine with
            # learned weight and bias, and neither it nor the trunk keeps running
            # statistics. The copy is therefore over an empty list and exists to
            # stay correct if a module that does keep state is ever added - such
            # state must be copied rather than averaged, since averaging two
            # normalisations is not a normalisation.
            for target_buffer, online_buffer in zip(
                self.target.buffers(), self.online.buffers(), strict=True
            ):
                target_buffer.copy_(online_buffer)

    def _reset(self) -> None:
        """BBF's shrink-and-perturb reset (`spr_agent.py` `jit_reset`).

        The trunk is BBF's encoder: each parameter becomes 0.5 old + 0.5 fresh.
        The core and the heads are its head: they are replaced by a fresh
        initialisation. The target is treated the same way from a fresh
        initialisation of its own. The whole Adam state is cleared, trunk
        included. BBF's optimizer state is an optax chain of masked states, so
        `copy_params`' `keys_to_copy` never matches inside it and the fresh state
        replaces all of it. The anneals then start their next cycle.
        """
        self._resets += 1
        fresh_online = self._fresh_network("online")
        fresh_target = self._fresh_network("target")
        with torch.no_grad():
            for network, fresh in ((self.online, fresh_online), (self.target, fresh_target)):
                for old, new in zip(
                    network.trunk.parameters(), fresh.trunk.parameters(), strict=True
                ):
                    old.copy_(old * SHRINK_FACTOR + new * PERTURB_FACTOR)
                for old, new in zip(
                    _parameters_of(network.core, network.heads),
                    _parameters_of(fresh.core, fresh.heads),
                    strict=True,
                ):
                    old.copy_(new)
        self.optimizer.state.clear()
        self._cycle_steps = 0

    def _fresh_network(self, role: str) -> StackedPolicyNetwork:
        """A new initialisation for this reset, seeded by run, reset and role.

        Built under a forked generator, so a reset neither reads nor moves the
        global one, and on the CPU, so the draw is the same on every device.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(reset_seed(self._reset_seed, self._resets, role))
            fresh = StackedPolicyNetwork(
                self.network_config, history_length=self.config.history_length
            )
        return fresh.to(self.device)

    # -- persistence -------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self._steps,
            "cycle_steps": self._cycle_steps,
            "resets": self._resets,
            "reset_seed": self._reset_seed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        # Every state this project has ever written carries the optimizer, so a
        # state that does not is a truncated file rather than an old one. It
        # raises here on the missing key: resuming on fresh moments instead
        # would change how the next steps are taken without saying so.
        self.optimizer.load_state_dict(state["optimizer"])
        self._steps = int(state["steps"])
        # A state written before resets existed never reset, so its cycle is
        # the whole run: the anneals it trained under read the total count.
        self._cycle_steps = int(state.get("cycle_steps", self._steps))
        self._resets = int(state.get("resets", 0))
        self._reset_seed = int(state.get("reset_seed", self._reset_seed))


#: BBF's shrink-and-perturb weights (`BBF.gin` `shrink_factor=0.5`,
#: `perturb_factor=0.5`; the paper's "move them 50% towards the random target").
SHRINK_FACTOR = 0.5
PERTURB_FACTOR = 0.5


def reset_seed(base: int, reset: int, role: str) -> int:
    """The seed of one fresh initialisation: this run's, this reset's, this network's.

    Online and target draw different initialisations, as BBF's `jit_reset`
    splits its key in two.
    """
    digest = hashlib.sha256(f"{base}:{reset}:{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _parameters_of(*modules: nn.Module) -> list[nn.Parameter]:
    return [parameter for module in modules for parameter in module.parameters()]


def _adamw(network: nn.Module, config: StackedDqnConfig) -> torch.optim.AdamW:
    """AdamW over the online network, with or without decay on vectors.

    Without it the parameters are split in two groups by rank, BBF's mask:
    matrices decay, one-dimensional parameters do not. With it there is the
    one group every run before BBF was built with.
    """
    if config.weight_decay_on_vectors:
        return torch.optim.AdamW(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_eps,
        )
    parameters = list(network.parameters())
    groups: list[dict[str, Any]] = [
        {"params": [p for p in parameters if p.ndim != 1]},
        {"params": [p for p in parameters if p.ndim == 1], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.adam_eps,
    )
