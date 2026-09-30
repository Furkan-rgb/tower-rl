"""The rank-1 candidate from `docs/rl-candidates.md` 3.1.

Masked dueling double-Q learning over replay sequences, carrying time in a
window of recent run scalars rather than in a hidden state: section 2.3 of the
candidate study argues this problem is much closer to fully observed than
partially observed, and this is the backbone the project runs on.

Four things come from the Atari 100k literature the study cites: an EMA target
rather than a periodic hard copy, decoupled weight decay, a replay ratio the
training loop supplies, and SR-SPR's periodic resets (`StackedDqnBackbone._reset`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, StateFeatures
from tower_rl.environment.run_actions import WAIT, action_index
from tower_rl.learning.backbone import LearnMetrics, SequenceBatch
from tower_rl.learning.exploration import zeta_duration
from tower_rl.learning.network import NetworkConfig, StackedPolicyNetwork, StackedState
from tower_rl.learning.value_learning import (
    game_time_discounts,
    n_step_targets,
    real_step_td_errors,
    survival_rewards,
    value_fit_correlation,
    weighted_sequence_loss,
)

#: What a running ez-greedy option takes while its own action is masked: the
#: environment's no-op, always legal in an active run (docs/solution.md 7.2).
WAIT_INDEX = action_index(WAIT)

#: A reset's shrink-and-perturb of the trunk: trunk <- SHRINK * old + PERTURB *
#: fresh. SR-SPR's 0.8 / 0.2 (`SR_SPR.gin` in the BBF code,
#: github.com/google-research/google-research/tree/master/bigger_better_faster),
#: not BBF's 0.5 / 0.5, which BBF takes because its 4x-wide network needs more
#: regularisation; this one is 197k parameters (docs/solution.md 9.4).
RESET_SHRINK = 0.8
RESET_PERTURB = 0.2
#: What a reset re-initialises outright, by `StackedPolicyNetwork` attribute:
#: the core and the dueling heads, BBF's projection and head
#: (`reset_projection`, `reset_head`). The trunk, BBF's encoder, is only
#: shrunk and perturbed. The optimizer state of these parameters is cleared
#: with them, as Nikishin et al. 2022 (arXiv 2205.07802) reset optimizer
#: statistics with the layers they reset; the trunk's moments are kept.
RESET_MODULES = ("core", "heads")


@dataclass(frozen=True)
class StackedDqnConfig:
    #: How many run-scalar vectors the window holds, the current one included.
    #: The candidate study treats this as the tuned knob in the range 4 to 16 and
    #: names k = 1 as the ablation that settles whether history is needed at all.
    history_length: int = 8
    #: 1/(1 - discount) is the horizon in decisions: 100 at 0.99. What the
    #: per-decision discount was calibrated against was the ~121-decision
    #: episode of baseline v1; the game-time discount below replaced it for the
    #: runs since, whose horizon does not move with how many choice points a
    #: policy makes.
    discount: float = 0.99
    #: Discount per second of game time instead of per decision, or None for
    #: `discount` per decision. A transition that spans t game-seconds is then
    #: discounted by this ** t: a purchase takes no game time and costs no
    #: discount, and the horizon is fixed in waves rather than in however many
    #: choice points a policy makes (docs/solution.md 9.4d). When set,
    #: `discount` is not read. Not a code default: the protocol's discount
    #: horizon is 0.999, a horizon of 1,000 game-seconds or about 28.5 waves,
    #: a task parameter held identical across learners from M3-P015 on (ADR
    #: 0013); M3-P003 to M3-P014 ran at 0.997 or 0.999 as a tuned choice.
    discount_per_game_second: float | None = None
    #: Replace the wave reward, in the learner only, with game time survived in
    #: waves, integrated exactly under the game-time discount and scaled so
    #: the return is bounded by `V_REF` (docs/solution.md 9.4e, ADR 0013).
    #: Dying later in a wave then scores higher, which
    #: the wave reward cannot express. Needs `discount_per_game_second`.
    survival_time_reward: bool = False
    #: About 21.7 decisions pass per wave, and the whole reward is the wave
    #: change (under the wave reward), so a short n-step needs several
    #: bootstrap hops to carry one wave back to the decisions that earned it.
    n_step: int = 10
    #: Where the n-step anneal ends, or None to hold `n_step` fixed. Starting
    #: long gives fast early credit propagation; shortening it as the value
    #: estimate becomes worth bootstrapping from trades that for less variance.
    #: `n_step` is where the anneal starts.
    n_step_final: int | None = None
    #: Gradient steps the anneal from `n_step` to `n_step_final` takes, after
    #: which the final n holds. Counted in the learner's own steps, which a
    #: checkpoint carries, so a resumed run continues the schedule.
    n_step_anneal_steps: int = 0
    learning_rate: float = 1e-4
    #: Decoupled weight decay, hence AdamW rather than Adam.
    weight_decay: float = 1e-5
    #: AdamW's epsilon. Rainbow, DER, SPR and BBF use 1.5e-4, but that value
    #: pairs with rewards clipped to +-1; this run's rewards are ~1/35 per
    #: game-second, so 1.5e-4 dominates rather than bounds the second moment:
    #: at M3-P010 300k, 41% of parameters had sqrt(v_hat) below it, and its
    #: mean update per step was 0.044x the learning rate against M3-P009's
    #: 0.102x at 1e-8 (M3-P010, docs/experiments.md). Reverted to torch's
    #: default 1e-8. A resume keeps the epsilon its optimizer state holds.
    adam_epsilon: float = 1e-8
    #: A target that follows the online network smoothly. At this replay ratio a
    #: periodic hard copy moves the target in large infrequent jumps, which is
    #: what the data-efficient recipe replaces.
    target_ema_decay: float = 0.995
    #: Explore with ez-greedy (docs/solution.md 9.5): an exploratory action,
    #: once drawn, is repeated for a zeta-distributed number of decisions
    #: rather than for one. Acting only; off acts exactly as before.
    ez_greedy: bool = False
    #: Gradient steps between SR-SPR-style resets (`StackedDqnBackbone._reset`),
    #: or 0 for none, which is every run before M3-P014. Counted in the
    #: learner's own steps, which a checkpoint carries.
    reset_every_steps: int = 0
    #: The last step a reset may happen at: BBF's `no_resets_after`, so the
    #: network is never reset with less than one interval left to recover in.
    #: The training script derives it from the budget.
    last_reset_step: int = 0
    gradient_clip: float = 10.0
    huber_delta: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.history_length < 1:
            raise ValueError("history length must be at least one step")
        if not 0.0 < self.discount < 1.0:
            raise ValueError("discount must be within (0, 1)")
        if self.discount_per_game_second is not None and not (
            0.0 < self.discount_per_game_second < 1.0
        ):
            raise ValueError("discount per game-second must be within (0, 1)")
        if self.survival_time_reward and self.discount_per_game_second is None:
            raise ValueError("the survival-time reward needs a discount per game-second")
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
        if self.reset_every_steps < 0:
            raise ValueError("the reset interval cannot be negative")
        if self.last_reset_step < 0:
            raise ValueError("the last reset step cannot be negative")

    @property
    def books_reward_at_span_end(self) -> bool:
        """Whether a reward is valued at the end of its span rather than its start.

        Only under the game-time discount, where a span has a length to be
        discounted over: the wave change is booked where the span ends, so the
        reward a transition carries, valued at its start, is d * r. Per decision
        a transition has no length and the reward is r, as it always was.
        """
        return self.discount_per_game_second is not None

    def transition_discounts(self, game_ms: torch.Tensor) -> torch.Tensor:
        """Each transition's own discount d, from the game time it spanned.

        float64, so a constant per-decision d multiplies up to exactly the
        scalar powers the per-decision target has always used.
        """
        if self.discount_per_game_second is None:
            return torch.full_like(game_ms, self.discount, dtype=torch.float64)
        return game_time_discounts(self.discount_per_game_second, game_ms)

    def n_step_at(self, gradient_steps: int) -> int:
        """The n the target is built with after this many gradient steps.

        Exponential interpolation: n0 * (n1 / n0) ** (t / T), rounded, with t
        held at T once the anneal is over. Fixed at `n_step` without one.
        """
        if self.n_step_final is None:
            return self.n_step
        progress = min(gradient_steps, self.n_step_anneal_steps) / self.n_step_anneal_steps
        return round(self.n_step * float((self.n_step_final / self.n_step) ** progress))


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
    #: The step of the latest reset, 0 before any, and how many resets there
    #: have been. The n-step anneal is counted from the former (BBF restarts it
    #: after every reset); the latter seeds the next fresh network.
    _steps_at_reset: int = field(default=0, init=False)
    resets: int = field(default=0, init=False)
    _random: random.Random = field(init=False)
    #: The ez-greedy option running in this episode: its action, and how many
    #: more decisions it takes after the ones already taken (0 when none runs).
    _option_action: int = field(default=0, init=False)
    _option_remaining: int = field(default=0, init=False)
    _option_length: int = field(default=0, init=False)
    #: This episode's ez-greedy options, for the collected-episode record: how
    #: many started (n = 1 included) and the most decisions one ran for before
    #: it ended or the episode did. Both stay 0 with ez-greedy off.
    options_started: int = field(default=0, init=False)
    longest_option: int = field(default=0, init=False)

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
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        self._random = random.Random(self.config.seed)

    # -- acting ------------------------------------------------------------

    @property
    def model_version(self) -> int:
        return self._steps

    def initial_state(self) -> StackedState:
        # An episode boundary ends any option, and its counts start over.
        self._option_remaining = self._option_length = 0
        self.options_started = self.longest_option = 0
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

        # The forward pass runs even inside an ez-greedy option: the window
        # must advance by this decision whatever acts on it.
        with torch.no_grad():
            q, next_state = self.online(scalars, rows, mask, state)
        if self.config.ez_greedy:
            return self._ez_greedy_action(q, valid, epsilon), next_state
        # Exploration still respects the mask: an epsilon action is drawn from the
        # valid set, never from the whole space, so exploration cannot waste a
        # step on something the game would refuse anyway.
        if epsilon > 0.0 and self._random.random() < epsilon:
            return self._random.choice(valid), next_state
        return int(q[0, 0].argmax().item()), next_state

    def _ez_greedy_action(self, q: torch.Tensor, valid: list[int], epsilon: float) -> int:
        """Algorithm 1 of Dabney et al. 2021, counted in decisions.

        No coin is flipped while an option runs. A new option draws its length
        n, then its action from the valid set, and this decision is the first
        of its n - so n = 1 is one epsilon-greedy step. A running option whose
        action is masked takes WAIT for that decision and still counts it down,
        resuming the action once it is legal again. At epsilon 0 nothing
        explores, a running option included.
        """
        if epsilon > 0.0 and self._option_remaining > 0:
            self._option_remaining -= 1
            self._option_length += 1
            self.longest_option = max(self.longest_option, self._option_length)
            if self._option_action in valid:
                return self._option_action
            if WAIT_INDEX not in valid:
                raise ValueError("WAIT is not available in this state")
            return WAIT_INDEX
        if epsilon > 0.0 and self._random.random() < epsilon:
            self._option_remaining = zeta_duration(self._random) - 1
            self._option_action = self._random.choice(valid)
            self._option_length = 1
            self.options_started += 1
            self.longest_option = max(self.longest_option, 1)
            return self._option_action
        return int(q[0, 0].argmax().item())

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
        discounts = self.config.transition_discounts(batch.game_ms[:, burn_in:])
        # The reward each transition carries, valued at its start: the wave
        # change, or under the survival-time reward the game time the span
        # survived, which replaces it here and nowhere else (solution.md 9.4e).
        if self.config.survival_time_reward:
            step_rewards = survival_rewards(discounts).to(rewards.dtype)
        else:
            step_rewards = (
                rewards * discounts.to(rewards.dtype)
                if self.config.books_reward_at_span_end
                else rewards
            )

        online_q, _ = self.online(scalars, rows, mask, history)
        chosen = online_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            target_q, _ = self.target(scalars, rows, mask, history)
            targets, learnable = n_step_targets(
                step_rewards,
                dones,
                online_q.detach(),
                target_q,
                mask,
                discounts=discounts,
                # Taken before this step is counted, so the first step after
                # the start or a reset is t = 0.
                n_step=self.config.n_step_at(self._steps - self._steps_at_reset),
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
        self._update_target()
        every = self.config.reset_every_steps
        if every and self._steps % every == 0 and self._steps <= self.config.last_reset_step:
            self._reset()

        absolute = errors.abs().detach()
        with torch.no_grad():
            fit = value_fit_correlation(
                online_q.detach(),
                mask,
                step_rewards,
                dones,
                real,
                discounts=discounts,
            )
        taken = chosen.detach()[real > 0]
        return LearnMetrics(
            weighted_loss=float(loss.detach().item()),
            unweighted_mean_absolute_td_error=float(
                (absolute.sum() / real.sum().clamp(min=1.0)).item()
            ),
            gradient_norm=float(gradient_norm.item()),
            td_errors=real_step_td_errors(absolute, real),
            value_fit_correlation=fit,
            taken_q_max=float(taken.max().item()) if taken.numel() else None,
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
        """SR-SPR's reset, as the BBF code performs it (docs/solution.md 9.4).

        Against a fresh network: the core and heads become its parameters, the
        trunk moves a fifth of the way to them, the target becomes the online
        network, the reset parameters' AdamW moments are dropped and the trunk's
        kept, and the n-step anneal starts again from here.
        """
        fresh = self._fresh_network()
        reset: list[torch.nn.Parameter] = []
        with torch.no_grad():
            for (name, online), replacement in zip(
                self.online.named_parameters(), fresh.parameters(), strict=True
            ):
                if name.split(".")[0] in RESET_MODULES:
                    online.copy_(replacement)
                    reset.append(online)
                else:
                    online.mul_(RESET_SHRINK).add_(replacement, alpha=RESET_PERTURB)
        self.target.load_state_dict(self.online.state_dict())
        for parameter in reset:
            # AdamW starts a parameter with no state over at step 0: zero
            # moments and bias correction from the beginning.
            self.optimizer.state.pop(parameter, None)
        self.resets += 1
        self._steps_at_reset = self._steps

    def _fresh_network(self) -> StackedPolicyNetwork:
        """A newly initialised network, seeded by the run seed and the reset count.

        Built on the CPU inside a forked RNG, so a run's resets are reproducible
        from its seed and draw nothing from the stream the rest of the run uses.
        ``fork_rng(devices=[])`` only saves and restores the CPU generator, so
        only the CPU generator is reseeded here (``torch.default_generator``,
        not ``torch.manual_seed``, which would also reseed every CUDA
        generator and is not undone on exit from the fork).

        With ``seed`` set to ``None`` (only tests do this; ``--seed`` defaults
        to 0), the CPU generator is never reseeded here, so every reset draws
        the same fresh network from wherever the ambient CPU RNG stream is at
        that point in the run.
        """
        with torch.random.fork_rng(devices=[]):
            if self.config.seed is not None:
                seed = random.Random(f"{self.config.seed}/reset/{self.resets}").getrandbits(63)
                torch.default_generator.manual_seed(seed)
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
            "steps_at_reset": self._steps_at_reset,
            "resets": self.resets,
        }

    def network_state_dict(self) -> dict[str, Any]:
        """What acting reads: the online network and its step; no target, no optimizer."""
        return {"online": self.online.state_dict(), "steps": self._steps}

    def load_network_state_dict(self, state: dict[str, Any]) -> None:
        self.online.load_state_dict(state["online"])
        self._steps = int(state["steps"])

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        # Every state this project has ever written carries the optimizer, so a
        # state that does not is a truncated file rather than an old one. It
        # raises here on the missing key: resuming on fresh moments instead
        # would change how the next steps are taken without saying so.
        self.optimizer.load_state_dict(state["optimizer"])
        self._steps = int(state["steps"])
        # Absent from every state written before resets existed, none of which
        # was ever reset: its anneal counts from step 0 and no reset has seeded.
        self._steps_at_reset = int(state.get("steps_at_reset", 0))
        self.resets = int(state.get("resets", 0))
