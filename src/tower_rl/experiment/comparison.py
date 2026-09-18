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


def interleave_schedule(
    arms: tuple[str, ...], episodes_per_arm: int, *, block: int = 5, seed: int | None = None
) -> tuple[str, ...]:
    """Order episodes so drift lands on every arm equally.

    Arms take turns in small blocks rather than running to completion one after
    another, and the order within each round is shuffled so no arm is
    systematically first. A block larger than one amortises the cost of switching
    configuration, which on this environment means restarting at a new speed.
    """
    if not arms:
        raise ValueError("a comparison needs at least one arm")
    if episodes_per_arm < 1 or block < 1:
        raise ValueError("episodes per arm and block size must be positive")

    generator = random.Random(seed)
    schedule: list[str] = []
    remaining = dict.fromkeys(arms, episodes_per_arm)
    while any(count > 0 for count in remaining.values()):
        order = [arm for arm in arms if remaining[arm] > 0]
        generator.shuffle(order)
        for arm in order:
            take = min(block, remaining[arm])
            schedule.extend([arm] * take)
            remaining[arm] -= take
    return tuple(schedule)


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
    """The interquartile mean: the mean of the middle 50% of the sample.

    This is the aggregate recommended by Agarwal et al. 2021, *Deep
    Reinforcement Learning at the Edge of the Statistical Precipice*, and this
    is the same estimator their `rliable` library computes - a symmetric 25%
    trimmed mean. `rliable` itself is deliberately not a dependency of this
    project: the estimator is four lines, and its interval comes from
    `stratified_bootstrap` below rather than from a second bootstrap
    implementation.

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
    if not strata:
        raise ValueError("a stratified bootstrap needs at least one stratum")
    if any(not values for values in strata.values()):
        empty = sorted(name for name, values in strata.items() if not values)
        raise ValueError(f"strata with no values cannot be resampled: {empty}")
    if resamples < 1:
        raise ValueError("a bootstrap needs at least one resample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")

    ordered = [list(strata[name]) for name in sorted(strata)]
    observed = statistic([value for values in ordered for value in values])
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        pooled: list[float] = []
        for values in ordered:
            pooled.extend(generator.choices(values, k=len(values)))
        estimates.append(statistic(pooled))
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    low = estimates[int(tail * resamples)]
    high = estimates[min(int((1.0 - tail) * resamples), resamples - 1)]
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


def required_episodes(
    standard_deviation: float, difference: float, *, power: float = 0.8
) -> int:
    """Episodes per arm to detect a difference, at five percent significance.

    Used to decide a budget before running, and to state afterwards what the
    sample could and could not have detected.
    """
    if standard_deviation <= 0 or difference <= 0:
        raise ValueError("standard deviation and difference must be positive")
    z_power = {0.8: 0.8416, 0.9: 1.2816}.get(power)
    if z_power is None:
        raise ValueError("only 80 and 90 percent power are tabulated")
    return math.ceil(2 * (1.96 + z_power) ** 2 * standard_deviation**2 / difference**2)
