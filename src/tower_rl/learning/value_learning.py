"""What every value-based backbone in the comparison shares.

The task's side of the target (ADR 0013): the game-time discount and the
survival-time reward are the same for every learner, and the double-Q
bootstrap's masked argmax is one definition.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

#: One wave in game-seconds: the unit the survival-time reward's bound,
#: `V_REF`, is expressed in. Waves are clock-driven: every completed wave
#: 2..19 of the M3-P003 evaluation lasted 34.88-35.20 s (docs/solution.md
#: 9.4e), and waves 2-74 of the M3-P014 replay dump 35.0 s (ADR 0013).
WAVE_SECONDS = 35.0

#: The survival-time reward's maximum return, whatever the discount: an
#: immortal policy's return, about 9.51 waves, at the 0.997 per game-second
#: every survival-time run before M3-P015 used. Only at 0.997 is a return in
#: waves; at any other discount it is in units of that 0.997 horizon, the
#: return a policy surviving for ever would have had there. The reward is
#: scaled to it so the value
#: scale - and with it the |TD| and gradient-norm monitors - does not move
#: with the discount horizon (ADR 0013, Reward scaling). At 0.997 the scaled reward is the
#: unscaled one of those runs.
V_REF = 1.0 / (WAVE_SECONDS * -math.log(0.997))


def game_time_discounts(per_game_second: float, game_ms: Tensor) -> Tensor:
    """Each transition's own discount d = gamma_s ** t, t the game-seconds it spanned.

    A purchase, and padding, span no game time and get exactly 1. float64, so
    a product of many ds is exact to far below the value's resolution. Every
    learner that discounts by game time takes its d from here (ADR 0013).
    """
    return torch.pow(per_game_second, game_ms.to(torch.float64) / 1000.0)


def survival_rewards(discounts: Tensor) -> Tensor:
    """Game time each transition survived, valued at its start: (1 - d) * V_REF.

    A constant reward per game-second, integrated over a span of t seconds
    under gamma_s ** t (Bradtke & Duff 1995, Eq. 12), scaled so an immortal
    policy's return is `V_REF`: a return is V_REF * (1 - gamma_s ** T),
    bounded by V_REF whatever gamma_s is (ADR 0013). At gamma_s 0.997 this
    is the (1 - d) / (beta * WAVE_SECONDS), beta = -ln gamma_s, of every
    survival-time run before M3-P015. A span of no game time - a purchase,
    or padding - earns exactly 0. The survival-time reward of every learner
    that takes `--survival-time-reward` (docs/solution.md 9.4e).
    """
    return (1.0 - discounts) * V_REF


def evaluated_next_values(online_q: Tensor, target_q: Tensor, mask: Tensor) -> Tensor:
    """Double Q: the online network chooses, the target network values.

    That split is what stops one optimistic network from bootstrapping its own
    overestimate. A state with no valid action is terminal and must contribute
    zero rather than the negative infinity the mask would otherwise carry.
    """
    best = online_q.argmax(dim=-1, keepdim=True)
    evaluated = target_q.gather(-1, best).squeeze(-1)
    evaluated = torch.where(mask.any(dim=-1), evaluated, torch.zeros_like(evaluated))
    return torch.where(torch.isfinite(evaluated), evaluated, torch.zeros_like(evaluated))


def real_step_td_errors(absolute: Tensor, real: Tensor) -> tuple[tuple[float, ...], ...]:
    """Per-sequence absolute TD errors over real steps only.

    Padding is filler, not a step the learner was wrong about. Reporting a zero
    for it would drag the mean term of a sequence's priority down and make a
    padded sequence look duller than it is, which is exactly the sequence - a
    short episode, an early death - that prioritization should be surfacing.
    """
    kept = real.detach().cpu().bool()
    return tuple(
        tuple(row[flags].tolist())
        for row, flags in zip(absolute.detach().cpu(), kept, strict=True)
    )


def value_fit_correlation(
    online_q: Tensor,
    mask: Tensor,
    rewards: Tensor,
    dones: Tensor,
    real: Tensor,
    *,
    discounts: Tensor,
) -> float | None:
    """Correlation between V(s_t) and the return the episode actually realised.

    This is the falsifier a flat learning curve cannot distinguish without: a
    learner whose values track the realised return is learning slowly, and one
    whose values are uncorrelated with it is broken, and the two look identical
    in the final-wave distribution.

    The return is realised rather than bootstrapped, so only steps whose episode
    terminates inside the stored sequence have one: for any later step the tail
    of the return is missing and a truncated return would be a different
    quantity. Padded steps are not experience and are excluded like everywhere
    else. `None` when the batch holds fewer than two such steps, or when either
    side is constant across them - a correlation is undefined there, and zero
    would be a claim.

    `discounts` [B, T] are the same per-transition ds the target is built
    with, and `rewards` the same rewards, so the realised return is the
    quantity the values are trained towards rather than a different one.
    """
    values = torch.where(mask, online_q, torch.full_like(online_q, float("-inf"))).amax(dim=-1)
    time = rewards.shape[1]
    terminal = dones.to(rewards.dtype)
    step_discounts = discounts.to(rewards.dtype)
    returns = torch.zeros_like(rewards)
    ends_inside = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], device=rewards.device)
    ended = torch.zeros_like(running)
    for step in reversed(range(time)):
        running = (
            rewards[:, step] + step_discounts[:, step] * (1.0 - terminal[:, step]) * running
        )
        ended = torch.maximum(terminal[:, step], ended)
        returns[:, step] = running
        ends_inside[:, step] = ended

    keep = (ends_inside > 0) & (real > 0) & torch.isfinite(values)
    predicted = values[keep]
    realised = returns[keep]
    if predicted.numel() < 2:
        return None
    predicted = predicted - predicted.mean()
    realised = realised - realised.mean()
    spread = predicted.norm() * realised.norm()
    if float(spread.item()) <= 0.0:
        return None
    return float((predicted @ realised / spread).item())
