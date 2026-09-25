"""The rank-1 candidate from `docs/rl-candidates.md` 3.1.

Masked dueling double-Q learning over replay sequences, carrying time in a
window of recent run scalars rather than in a hidden state: section 2.3 of the
candidate study argues this problem is much closer to fully observed than
partially observed, and this is the backbone the project runs on.

Three things come from the Atari 100k literature the study cites: an EMA target
rather than a periodic hard copy, decoupled weight decay, and a replay ratio the
training loop supplies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import torch

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, StateFeatures
from tower_rl.learning.backbone import LearnMetrics, SequenceBatch
from tower_rl.learning.network import NetworkConfig, StackedPolicyNetwork, StackedState
from tower_rl.learning.value_learning import (
    n_step_targets,
    real_step_td_errors,
    value_fit_correlation,
    weighted_sequence_loss,
)

#: One wave in game-seconds: the unit the survival-time reward is paid in.
#: Waves are clock-driven, and every completed wave 2..19 of the M3-P003
#: evaluation lasted 34.88-35.20 s (docs/solution.md 9.4e). It only sets the
#: scale - a positive scale on the whole reward leaves the optimum unchanged.
WAVE_SECONDS = 35.0


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
    #: Discount per second of game time instead of per decision, or None for
    #: `discount` per decision. A transition that spans t game-seconds is then
    #: discounted by this ** t: a purchase takes no game time and costs no
    #: discount, and the horizon is fixed in waves rather than in however many
    #: choice points a policy makes (docs/solution.md 9.4d). When set,
    #: `discount` is not read.
    discount_per_game_second: float | None = None
    #: Replace the wave reward, in the learner only, with game time survived in
    #: waves, integrated exactly under the game-time discount
    #: (docs/solution.md 9.4e). Dying later in a wave then scores higher, which
    #: the wave reward cannot express. Needs `discount_per_game_second`.
    survival_time_reward: bool = False
    #: About 21.7 decisions pass per wave, and the whole reward is the wave
    #: change, so a short n-step needs several bootstrap hops to carry one wave
    #: back to the decisions that earned it.
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
        seconds = game_ms.to(torch.float64) / 1000.0
        return torch.pow(self.discount_per_game_second, seconds)

    def survival_rewards(self, discounts: torch.Tensor) -> torch.Tensor:
        """Game time each transition survived, in waves, valued at its start.

        A reward of 1/WAVE_SECONDS per game-second, integrated over a span of
        t seconds under gamma_s ** t: (1 - d) / (beta * WAVE_SECONDS), with
        beta = -ln gamma_s (Bradtke & Duff 1995, Eq. 12). A span of no game
        time - a purchase, or padding - earns exactly 0. float64, as
        `discounts` is.
        """
        if self.discount_per_game_second is None:
            raise ValueError("the survival-time reward needs a discount per game-second")
        beta = -math.log(self.discount_per_game_second)
        return (1.0 - discounts) / (beta * WAVE_SECONDS)

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
        real = (~batch.padding[:, burn_in:]).to(rewards.dtype)
        discounts = self.config.transition_discounts(batch.game_ms[:, burn_in:])
        # The reward each transition carries, valued at its start: the wave
        # change, or under the survival-time reward the game time the span
        # survived, which replaces it here and nowhere else (solution.md 9.4e).
        if self.config.survival_time_reward:
            step_rewards = self.config.survival_rewards(discounts).to(rewards.dtype)
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
                # Taken before this step is counted, so the first step is t = 0.
                n_step=self.config.n_step_at(self._steps),
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
        # Every state this project has ever written carries the optimizer, so a
        # state that does not is a truncated file rather than an old one. It
        # raises here on the missing key: resuming on fresh moments instead
        # would change how the next steps are taken without saying so.
        self.optimizer.load_state_dict(state["optimizer"])
        self._steps = int(state["steps"])
