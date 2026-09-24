"""Comparing arms without fooling ourselves.

Two traps do most of the damage in a benchmark like this one, and neither is an
algorithm problem.  The first is temporal confounding: running arm A for an hour
and then arm B for an hour confounds the arm with whatever drifted on the host,
the device or the account in between, so arms are interleaved in small blocks
instead.  The second is reading a difference out of noise, so differences are
reported as an interval rather than a point, and the interval comes from the data
by resampling rather than from an assumed distribution.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

BOOTSTRAP_ITERATIONS = 10_000

#: The proportion trimmed from each tail by `iqm`. A quarter off each end is
#: what makes the remainder the middle half.
IQM_TRIM = 0.25


@dataclass(frozen=True)
class Difference:
    """One pairwise comparison, reported as a range rather than a verdict."""

    left: str
    right: str
    left_mean: float
    right_mean: float
    difference: float
    low: float
    high: float
    effect_size: float
    left_episodes: int
    right_episodes: int

    @property
    def separated(self) -> bool:
        """Whether the interval excludes zero.

        This is the only claim the data supports. An interval containing zero
        means the arms are indistinguishable at this sample size, which is not
        the same as being equal.
        """
        return self.low > 0.0 or self.high < 0.0

    def describe(self) -> str:
        verdict = "separated" if self.separated else "indistinguishable"
        return (
            f"{self.left} {self.left_mean:.2f} vs {self.right} {self.right_mean:.2f}: "
            f"difference {self.difference:+.2f} "
            f"[{self.low:+.2f}, {self.high:+.2f}] d={self.effect_size:+.2f} "
            f"n={self.left_episodes}/{self.right_episodes} — {verdict}"
        )


def bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Resample the difference in means and return it with a percentile interval.

    Resampling makes no assumption about the shape of the final-wave distribution,
    which is bounded, discrete and skewed, and therefore not well served by a
    normal approximation at these sample sizes.
    """
    if len(left) < 2 or len(right) < 2:
        raise ValueError("each arm needs at least two episodes to be compared")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")

    generator = random.Random(seed)
    observed = statistics.fmean(left) - statistics.fmean(right)
    differences = []
    for _ in range(iterations):
        resampled_left = statistics.fmean(generator.choices(list(left), k=len(left)))
        resampled_right = statistics.fmean(generator.choices(list(right), k=len(right)))
        differences.append(resampled_left - resampled_right)
    differences.sort()
    tail = (1.0 - confidence) / 2.0
    low = differences[int(tail * iterations)]
    high = differences[min(int((1.0 - tail) * iterations), iterations - 1)]
    return observed, low, high


def iqm(values: Sequence[float]) -> float:
    """The interquartile mean: the middle half by the trim rule of Agarwal et al. 2021.

    "The middle half" is the intent, not the arithmetic. The rule is the one
    `rliable` computes, which is `scipy.stats.trim_mean(x, 0.25)`: drop
    `int(n * 0.25)` values from each end and average the rest. That is exactly
    half the sample only when `n` is a multiple of four - at `n = 14` it keeps 8
    of 14, which is 57% - because a fractional value cannot be dropped. The
    convention is followed rather than improved on, so a number here and a
    number from `rliable` are the same number.

    `rliable` itself is deliberately not a dependency of this project: the
    estimator is four lines, and its interval comes from `stratified_bootstrap`
    below rather than from a second bootstrap implementation.

    A final wave is bounded, discrete and skewed, and a handful of very long
    episodes move its mean a long way. The IQM discards those tails without
    discarding the shape of the bulk the way a median does.
    """
    if not values:
        raise ValueError("an interquartile mean needs at least one value")
    ordered = sorted(values)
    # Trimmed by count, not by interpolated quantile, which is what
    # `scipy.stats.trim_mean(x, 0.25)` does and therefore what `rliable` does.
    cut = int(len(ordered) * IQM_TRIM)
    middle = ordered[cut : len(ordered) - cut]
    return statistics.fmean(middle)


def _ordered_strata(strata: Mapping[str, Sequence[float]]) -> list[list[float]]:
    """Each stratum's values, in a stable order, refusing what cannot be resampled."""
    if not strata:
        raise ValueError("a stratified bootstrap needs at least one stratum")
    if any(not values for values in strata.values()):
        empty = sorted(name for name, values in strata.items() if not values)
        raise ValueError(f"strata with no values cannot be resampled: {empty}")
    return [list(strata[name]) for name in sorted(strata)]


def _resample_pool(ordered: list[list[float]], generator: random.Random) -> list[float]:
    """One resample: every stratum redrawn to its own size, with replacement, then pooled."""
    pooled: list[float] = []
    for values in ordered:
        pooled.extend(generator.choices(values, k=len(values)))
    return pooled


def _percentile_interval(estimates: list[float], confidence: float) -> tuple[float, float]:
    """The bounds of the percentile interval over a bootstrap's estimates."""
    estimates.sort()
    count = len(estimates)
    tail = (1.0 - confidence) / 2.0
    low = estimates[int(tail * count)]
    high = estimates[min(int((1.0 - tail) * count), count - 1)]
    return low, high


def stratified_bootstrap(
    strata: Mapping[str, Sequence[float]],
    statistic: Callable[[Sequence[float]], float] = iqm,
    *,
    resamples: int = BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """A statistic over pooled strata, with a percentile interval that respects them.

    Every resample redraws each stratum to its own size, with replacement,
    before pooling - so a stratum that contributed twenty episodes contributes
    twenty to every resample. This is the stratified bootstrap of Agarwal et al.
    2021, with the strata being whatever unit the samples are not exchangeable
    across: here the actor an episode was collected on, and later the training
    seed a checkpoint came from. Resampling the pool flat instead would let one
    actor's episodes crowd out another's and would report an interval narrower
    or wider than the design earns.

    Returns the point estimate over the observed pool and the bounds of the
    percentile interval, in that order.
    """
    if resamples < 1:
        raise ValueError("a bootstrap needs at least one resample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")

    ordered = _ordered_strata(strata)
    observed = statistic([value for values in ordered for value in values])
    generator = random.Random(seed)
    estimates = [statistic(_resample_pool(ordered, generator)) for _ in range(resamples)]
    low, high = _percentile_interval(estimates, confidence)
    return observed, low, high


def stratified_bootstrap_difference(
    left: Mapping[str, Sequence[float]],
    right: Mapping[str, Sequence[float]],
    statistic: Callable[[Sequence[float]], float] = iqm,
    *,
    resamples: int = BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """The difference between two arms' statistics, with a stratified interval.

    This is the paired form of `stratified_bootstrap`, and it is what a rule
    written about "the pairwise interval of the IQM difference" names. Each arm
    is resampled within its own strata - actors here - independently of the
    other, because the two arms were collected as separate fleets and no episode
    of one pairs with an episode of the other; the difference of the two
    resampled statistics is the quantity whose percentile interval is returned.

    Taking the difference inside the resample rather than differencing two
    marginal intervals is the sharper test: two intervals that overlap can still
    have a difference that excludes zero, which is why the marginal intervals
    are reported beside this and never instead of it.

    Returns the difference over the observed pools and the bounds of the
    percentile interval, in that order.
    """
    if resamples < 1:
        raise ValueError("a bootstrap needs at least one resample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")

    ordered_left = _ordered_strata(left)
    ordered_right = _ordered_strata(right)
    observed = statistic(
        [value for values in ordered_left for value in values]
    ) - statistic([value for values in ordered_right for value in values])
    generator = random.Random(seed)
    differences = [
        statistic(_resample_pool(ordered_left, generator))
        - statistic(_resample_pool(ordered_right, generator))
        for _ in range(resamples)
    ]
    low, high = _percentile_interval(differences, confidence)
    return observed, low, high


def cohens_d(left: Sequence[float], right: Sequence[float]) -> float:
    """Standardised effect size, pooled. Reported beside the raw difference."""
    if len(left) < 2 or len(right) < 2:
        raise ValueError("each arm needs at least two episodes")
    left_variance = statistics.variance(left)
    right_variance = statistics.variance(right)
    pooled = ((len(left) - 1) * left_variance + (len(right) - 1) * right_variance) / (
        len(left) + len(right) - 2
    )
    if pooled <= 0.0:
        return 0.0
    return (statistics.fmean(left) - statistics.fmean(right)) / math.sqrt(pooled)


def compare(
    arms: Mapping[str, Sequence[float]], *, confidence: float = 0.95, seed: int | None = 0
) -> tuple[Difference, ...]:
    """Every pairwise comparison, in a stable order."""
    names = sorted(arms)
    results = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            observed, low, high = bootstrap_difference(
                arms[left], arms[right], confidence=confidence, seed=seed
            )
            results.append(
                Difference(
                    left=left,
                    right=right,
                    left_mean=statistics.fmean(arms[left]),
                    right_mean=statistics.fmean(arms[right]),
                    difference=observed,
                    low=low,
                    high=high,
                    effect_size=cohens_d(arms[left], arms[right]),
                    left_episodes=len(arms[left]),
                    right_episodes=len(arms[right]),
                )
            )
    return tuple(results)
