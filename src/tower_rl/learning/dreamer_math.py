"""The numerical parts of DreamerV3 that carry no network of their own.

Each function is a port of the official code (danijar/dreamerv3 at e3f02248)
and names where it came from, so a reader can check the port line by line:
`embodied/jax/nets.py` (symlog, symexp), `embodied/jax/heads.py` and
`embodied/jax/outs.py` (the symexp twohot output, categoricals with unimix),
`dreamerv3/agent.py` (`lambda_return`), `embodied/jax/utils.py` (`Normalize`
with `impl=perc`) and `embodied/jax/opt.py` (AGC, `scale_by_rms`,
`scale_by_momentum`: together LaProp). The one thing here the official code
does not have is the action mask; `docs/solution.md` 9.4d records why.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def twohot_bins(count: int = 255) -> Tensor:
    """The bins of the symexp twohot output: `heads.py` `Head.symexp_twohot`.

    Exponentially spaced in the raw value, symmetric about zero, built from one
    half so the two halves are exact negatives of each other.
    """
    if count % 2 == 1:
        half = symexp(torch.linspace(-20.0, 0.0, (count - 1) // 2 + 1))
        return torch.cat((half, -half[:-1].flip(0)))
    half = symexp(torch.linspace(-20.0, 0.0, count // 2))
    return torch.cat((half, -half.flip(0)))


def twohot_mean(logits: Tensor, bins: Tensor) -> Tensor:
    """The expected value under a twohot output: `outs.py` `TwoHot.pred`.

    Summed symmetrically outward from the centre bin, as the official code does,
    so uniform probabilities over symmetric bins predict exactly zero.
    """
    probs = torch.softmax(logits, -1)
    count = logits.shape[-1]
    if count % 2 == 1:
        middle = (count - 1) // 2
        low = (probs[..., :middle] * bins[:middle]).flip(-1)
        high = probs[..., middle + 1 :] * bins[middle + 1 :]
        centre = probs[..., middle] * bins[middle]
        return centre + (low + high).sum(-1)
    low = (probs[..., : count // 2] * bins[: count // 2]).flip(-1)
    high = probs[..., count // 2 :] * bins[count // 2 :]
    return (low + high).sum(-1)


def twohot_loss(logits: Tensor, target: Tensor, bins: Tensor) -> Tensor:
    """Cross-entropy against the twohot encoding of `target`: `outs.py` `TwoHot.loss`.

    The target is encoded in raw space against the symexp-spaced bins, as the
    current official code does (older code and the paper encode in symlog
    space). Values outside the outermost bins encode onto that bin alone.
    """
    target = target.detach()
    count = bins.shape[0]
    below = (bins <= target[..., None]).sum(-1) - 1
    above = count - (bins > target[..., None]).sum(-1)
    below = below.clamp(0, count - 1)
    above = above.clamp(0, count - 1)
    equal = below == above
    to_below = torch.where(equal, 1.0, (bins[below] - target).abs())
    to_above = torch.where(equal, 1.0, (bins[above] - target).abs())
    total = to_below + to_above
    weight_below = to_above / total
    weight_above = to_below / total
    encoded = torch.nn.functional.one_hot(below, count) * weight_below[..., None]
    encoded = encoded + torch.nn.functional.one_hot(above, count) * weight_above[..., None]
    return -(encoded * torch.log_softmax(logits, -1)).sum(-1)


def unimix_probs(logits: Tensor, unimix: float) -> Tensor:
    """Softmax mixed with the uniform distribution: `outs.py` `Categorical`."""
    probs = torch.softmax(logits, -1)
    return (1.0 - unimix) * probs + unimix / logits.shape[-1]


def categorical_kl(left: Tensor, right: Tensor) -> Tensor:
    """KL(left || right) between two batches of normalised probabilities, over the last axis."""
    return (left * (left.log() - right.log())).sum(-1)


def masked_policy_log_probs(logits: Tensor, mask: Tensor, unimix: float) -> Tensor:
    """Log-probabilities of the actor's categorical, restricted to valid actions.

    The official categorical mixes 1% uniform over every action; here the mix
    is uniform over the valid ones, and an invalid action has probability
    exactly zero (log-probability minus infinity), so it can never be sampled.
    """
    masked = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked, -1)
    valid = mask.to(probs.dtype)
    probs = (1.0 - unimix) * probs + unimix * valid / valid.sum(-1, keepdim=True)
    # The logarithm is taken of valid entries only: log(0) would carry an
    # infinite gradient that `where` multiplies by zero into NaN.
    safe = torch.where(mask, probs, torch.ones_like(probs)).log()
    return safe.masked_fill(~mask, float("-inf"))


def masked_entropy(log_probs: Tensor, mask: Tensor) -> Tensor:
    """Entropy over the valid actions of a masked categorical."""
    terms = torch.where(mask, log_probs.exp() * log_probs, torch.zeros_like(log_probs))
    return -terms.sum(-1)


def sample_index(probs: Tensor, uniform: Tensor) -> Tensor:
    """Inverse-CDF draw of one index per distribution from uniforms in [0, 1).

    The caller supplies the uniforms, so the stream they come from - the
    acting copy's own `random.Random`, or torch's - is the caller's choice. A
    zero-probability entry leaves the running sum exactly unchanged and so is
    never the first entry above the draw; the clamp to the last entry with
    positive probability covers a draw that rounds up to the total.
    """
    cdf = probs.cumsum(-1)
    point = (uniform * cdf[..., -1]).unsqueeze(-1)
    index = torch.searchsorted(cdf.contiguous(), point.contiguous(), right=True).squeeze(-1)
    positions = torch.arange(probs.shape[-1], device=probs.device)
    last = torch.where(probs > 0, positions, torch.zeros_like(positions)).amax(-1)
    return torch.minimum(index, last)


def lambda_return(
    last: Tensor,
    term: Tensor,
    reward: Tensor,
    boot: Tensor,
    discount: float,
    lam: float,
) -> Tensor:
    """TD(lambda) returns over [batch, time], one fewer step than the inputs.

    `agent.py` `lambda_return`, step for step. `last` cuts the recursion (the
    return at the step before it bootstraps entirely from `boot`); `term` zeroes
    the bootstrap. Reward and flags at step t belong to the transition into t.
    The official signature also takes a `val` it only checks the shape of; it
    is left out here.
    """
    live = (1.0 - term.to(reward.dtype))[:, 1:] * discount
    cont = (1.0 - last.to(reward.dtype))[:, 1:] * lam
    interm = reward[:, 1:] + (1.0 - cont) * live * boot[:, 1:]
    returns = [boot[:, -1]]
    for step in reversed(range(live.shape[1])):
        returns.append(interm[:, step] + live[:, step] * cont[:, step] * returns[-1])
    return torch.stack(list(reversed(returns))[:-1], 1)


class PercentileNormaliser:
    """The return normaliser: `utils.py` `Normalize(impl='perc', debias=False)`.

    Exponential moving averages of the 5th and 95th percentiles of the returns;
    the scale is their distance, floored at `limit` so small returns are never
    scaled up.
    """

    def __init__(
        self,
        *,
        rate: float = 0.01,
        limit: float = 1.0,
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
        device: torch.device | None = None,
    ) -> None:
        self.rate = rate
        self.limit = limit
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.low = torch.zeros((), device=device)
        self.high = torch.zeros((), device=device)

    def update(self, values: Tensor) -> None:
        values = values.detach().float().flatten()
        low = torch.quantile(values, self.low_percentile / 100.0)
        high = torch.quantile(values, self.high_percentile / 100.0)
        self.low = (1.0 - self.rate) * self.low + self.rate * low
        self.high = (1.0 - self.rate) * self.high + self.rate * high

    def stats(self) -> tuple[Tensor, Tensor]:
        """Offset and scale."""
        return self.low, torch.clamp(self.high - self.low, min=self.limit)

    def state_dict(self) -> dict[str, Tensor]:
        return {"low": self.low.clone(), "high": self.high.clone()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.low = state["low"].to(self.low.device).clone()
        self.high = state["high"].to(self.high.device).clone()


class LaProp(torch.optim.Optimizer):
    """DreamerV3's optimiser chain: AGC, then RMS scaling, then momentum.

    `agent.py` `_make_opt` with `opt.py`: `clip_by_agc(0.3, pmin=1e-3)`, then
    `scale_by_rms(beta2, eps)` with bias correction, then
    `scale_by_momentum(beta1)` with bias correction, then a learning rate that
    ramps linearly from zero over `warmup` updates. Normalising the gradient
    before the momentum, rather than after as Adam does, is what makes it
    LaProp (Ziyin et al. 2020).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        agc: float,
        agc_floor: float,
        warmup: int,
    ) -> None:
        defaults: dict[str, Any] = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "agc": agc,
            "agc_floor": agc_floor,
            "warmup": warmup,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        if closure is not None:
            raise ValueError("LaProp takes no closure")
        for group in self.param_groups:
            beta1, beta2, eps = group["beta1"], group["beta2"], group["eps"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["nu"] = torch.zeros_like(parameter)
                    state["mu"] = torch.zeros_like(parameter)
                # optax counts schedule steps from zero, so the first update
                # is taken at a learning rate of exactly zero.
                count = state["step"]
                warmup = group["warmup"]
                rate = group["lr"] * (min(count, warmup) / warmup if warmup else 1.0)
                update = parameter.grad
                if group["agc"]:
                    upper = group["agc"] * torch.clamp(
                        torch.linalg.vector_norm(parameter), min=group["agc_floor"]
                    )
                    ratio = torch.linalg.vector_norm(update) / upper
                    update = update / torch.clamp(ratio, min=1.0)
                step = count + 1
                nu, mu = state["nu"], state["mu"]
                nu.mul_(beta2).add_(update * update, alpha=1.0 - beta2)
                update = update / ((nu / (1.0 - beta2**step)).sqrt() + eps)
                mu.mul_(beta1).add_(update, alpha=1.0 - beta1)
                parameter.add_(mu / (1.0 - beta1**step), alpha=-rate)
                state["step"] = step
