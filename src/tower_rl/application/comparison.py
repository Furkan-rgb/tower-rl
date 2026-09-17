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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

BOOTSTRAP_ITERATIONS = 10_000


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
