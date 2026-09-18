"""Per-wave-index comparison of two arms, for questions a final wave cannot answer.

The question this exists for is whether a change to how the emulator is driven —
raising the guest frame rate from 60 Hz to 120 Hz — changes how the *game*
behaves. The game is not deterministic, so replaying identical actions and
diffing trajectories is invalid: identical action sequences already diverge by
the thirteenth decision. Equivalence therefore has to be distributional.

Mean final wave is the obvious distributional statistic and the wrong one. Its
pooled standard deviation is about 2.27 waves, so detecting half a wave at 80%
power needs roughly 324 episodes per arm — a budget the device does not have.
The statistics here are per wave *index* instead: how long wave k took, how many
decisions it cost, what health and cash the run held when wave k began. Those
carry a fraction of the between-episode variance, because the enormous variance
in a final wave comes from *how many* waves an episode survives, not from what
any one wave is like. A timing or physics change moves them; the episode length
lottery does not.

Reporting rules, deliberately structural rather than conventional:

* every comparison is an interval, never a verdict;
* every comparison carries `detectable_difference`, the smallest difference the
  sample could have found at 80% power, so a result can never be read as "no
  difference" when it only means "not distinguishable at this n";
* a wave index that too few episodes reached is reported as underpowered by
  name, never dropped in silence.

The bootstrap, Cohen's d and the power constants come from `comparison.py`;
nothing statistical is re-derived here.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tower_rl.experiment.comparison import (
    BOOTSTRAP_ITERATIONS,
    Difference,
    bootstrap_difference,
    cohens_d,
)

#: The same two normal deviates `comparison.required_episodes` uses: a two-sided
#: five percent test, and the tabulated powers. `detectable_difference` below is
#: the exact algebraic inverse of that function, and a test pins the two together.
Z_SIGNIFICANCE = 1.96
Z_BY_POWER: dict[float, float] = {0.8: 0.8416, 0.9: 1.2816}

#: The per-wave quantities this module compares, in report order. Each names a
#: field of `WaveObservation`; a field left `None` by the capture is reported as
#: uncaptured rather than treated as absent data.
PER_WAVE_STATISTICS: tuple[str, ...] = ("game_ms", "decisions", "health_fraction", "cash_log")

#: The whole-episode statistics, kept for the record so a report can state what
#: the blunt instrument could and could not see beside what the sharp one saw.
PER_EPISODE_STATISTICS: tuple[str, ...] = ("final_wave", "decisions")


@dataclass(frozen=True)
class WaveObservation:
    """What one episode did at one wave index.

    `completed` is false for the wave the episode died in: its duration and
    decision count are a fragment of a wave, and averaging fragments with whole
    waves would report a difference in how far the arms got as a difference in
    what a wave costs. Incomplete waves are excluded from the per-wave
    statistics; the final wave statistic is where episode length belongs.

    A quantity the environment does not capture is `None`, not zero.
    """

    wave: int
    completed: bool = True
    #: Game time spent inside this wave, on the game's own round clock.
    game_ms: float | None = None
    #: Decisions the environment asked for while this wave was current.
    decisions: int | None = None
    #: Health fraction observed in the first state of this wave.
    health_fraction: float | None = None
    #: Log-scaled cash observed in the first state of this wave, as the state
    #: schema carries it; raw cash is not part of `observation-v1`.
    cash_log: float | None = None


@dataclass(frozen=True)
class WavePoint:
    """One statistic at one wave index, with what it could not have detected."""

    wave: int
    difference: Difference
    detectable_difference: float

    def describe(self) -> str:
        return (
            f"  wave {self.wave:>3}: {self.difference.describe()} "
            f"(could detect >= {self.detectable_difference:.3f})"
        )


@dataclass(frozen=True)
class UnderpoweredWave:
    """A wave index too few episodes reached to compare. Named, not dropped."""

    wave: int
    left_episodes: int
    right_episodes: int


@dataclass(frozen=True)
class WaveStatisticComparison:
    """One per-wave statistic across every wave index both arms reached."""

    statistic: str
    points: tuple[WavePoint, ...]
    underpowered: tuple[UnderpoweredWave, ...]
    #: Coverage-weighted mean of the per-wave standardised effects. Wave indices
    #: share episodes, so this summarises the per-wave picture; it is not an
    #: effect measured on independent samples.
    pooled_effect_size: float | None
    #: The smallest standardised effect a single wave index could have shown at
    #: 80% power, at the widest per-wave sample. Pooling cannot beat this,
    #: because the wave indices are the same episodes seen again.
    detectable_effect_size: float | None
    #: Set when the capture does not record this quantity at all.
    uncaptured: bool = False

    @property
    def separated_waves(self) -> tuple[int, ...]:
        return tuple(point.wave for point in self.points if point.difference.separated)


@dataclass(frozen=True)
class EpisodeStatisticComparison:
    """One whole-episode statistic, with what it could not have detected."""

    statistic: str
    difference: Difference
    detectable_difference: float


@dataclass(frozen=True)
class EquivalenceAnalysis:
    """Everything two arms' episode records support saying about each other."""

    left: str
    right: str
    confidence: float
    power: float
    per_wave: tuple[WaveStatisticComparison, ...]
    per_episode: tuple[EpisodeStatisticComparison, ...]
    left_episodes: int
    right_episodes: int


def detectable_difference(
    standard_deviation: float, left_episodes: int, right_episodes: int, *, power: float = 0.8
) -> float:
    """The smallest difference this sample could have found, in the statistic's units.

    The exact inverse of `comparison.required_episodes`: that answers "how many
    episodes for this difference", this answers "what difference at these
    episodes". Every result in this module carries it, so a non-significant
    interval always arrives with the size of the effect it could have missed.
    """
    if standard_deviation < 0:
        raise ValueError("standard deviation cannot be negative")
    if left_episodes < 2 or right_episodes < 2:
        raise ValueError("each arm needs at least two episodes")
    z_power = Z_BY_POWER.get(power)
    if z_power is None:
        raise ValueError("only 80 and 90 percent power are tabulated")
    return (
        (Z_SIGNIFICANCE + z_power)
        * standard_deviation
        * math.sqrt(1.0 / left_episodes + 1.0 / right_episodes)
    )


def _pooled_standard_deviation(left: Sequence[float], right: Sequence[float]) -> float:
    pooled = ((len(left) - 1) * statistics.variance(left) + (len(right) - 1) * statistics.variance(
        right
    )) / (len(left) + len(right) - 2)
    return math.sqrt(max(pooled, 0.0))


def _difference(
    name: str,
    left_name: str,
    right_name: str,
    left: Sequence[float],
    right: Sequence[float],
    *,
    confidence: float,
    seed: int | None,
    iterations: int,
) -> tuple[Difference, float]:
    observed, low, high = bootstrap_difference(
        left, right, iterations=iterations, confidence=confidence, seed=seed
    )
    difference = Difference(
        left=f"{left_name}.{name}",
        right=f"{right_name}.{name}",
        left_mean=statistics.fmean(left),
        right_mean=statistics.fmean(right),
        difference=observed,
        low=low,
        high=high,
        effect_size=cohens_d(left, right),
        left_episodes=len(left),
        right_episodes=len(right),
    )
    return difference, _pooled_standard_deviation(left, right)


# ---------------------------------------------------------------------------
# Reading arms
# ---------------------------------------------------------------------------


def wave_observations(record: Mapping[str, Any]) -> tuple[WaveObservation, ...]:
    """The per-wave rows of one episode record, empty when none were captured.

    The durable episode record (`evaluator.episode_record`) carries `"waves"`,
    one row per wave index the episode entered, so this returns nothing only for
    records written before the environment captured them. An empty result is
    reported as uncaptured, which is a different statement from "the arms did
    not differ".
    """
    rows = record.get("waves") or ()
    return tuple(
        WaveObservation(
            wave=int(row["wave"]),
            completed=bool(row.get("completed", True)),
            game_ms=_optional_float(row.get("game_ms")),
            decisions=_optional_int(row.get("decisions")),
            health_fraction=_optional_float(row.get("health_fraction")),
            cash_log=_optional_float(row.get("cash_log")),
        )
        for row in rows
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def episode_records(report: Mapping[str, Any], arm: str | None = None) -> list[dict[str, Any]]:
    """The valid episode records of a report JSON, from either report shape.

    `run_actors.py` writes one arm per file (`evaluator.to_record`); a
    `compare_arms.py` report nests arms under `arms`. Both carry the same
    per-episode records, so a caller should not have to know which it holds.
    """
    if "arms" in report:
        arms = report["arms"]
        if arm is None:
            if len(arms) != 1:
                raise ValueError(f"report holds arms {sorted(arms)}; name the one to read")
            arm = next(iter(arms))
        if arm not in arms:
            raise ValueError(f"report has no arm {arm!r}; it holds {sorted(arms)}")
        records = arms[arm].get("episodes", [])
    else:
        records = report.get("episodes", [])
    return [dict(record) for record in records if record.get("valid", True)]


# ---------------------------------------------------------------------------
# Comparing arms
# ---------------------------------------------------------------------------


def _wave_values(
    records: Sequence[Mapping[str, Any]], statistic: str
) -> tuple[dict[int, list[float]], bool]:
    """Values by wave index, and whether the capture recorded this quantity at all."""
    by_wave: dict[int, list[float]] = {}
    captured = False
    for record in records:
        for observation in wave_observations(record):
            value = getattr(observation, statistic)
            if value is None:
                continue
            captured = True
            if not observation.completed:
                # A wave the episode died in is a fragment, not a wave.
                continue
            by_wave.setdefault(observation.wave, []).append(float(value))
    return by_wave, captured


def _compare_wave_statistic(
    statistic: str,
    left_name: str,
    right_name: str,
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    *,
    confidence: float,
    power: float,
    seed: int | None,
    iterations: int,
) -> WaveStatisticComparison:
    left_by_wave, left_captured = _wave_values(left_records, statistic)
    right_by_wave, right_captured = _wave_values(right_records, statistic)
    if not (left_captured and right_captured):
        return WaveStatisticComparison(
            statistic=statistic,
            points=(),
            underpowered=(),
            pooled_effect_size=None,
            detectable_effect_size=None,
            uncaptured=True,
        )

    points: list[WavePoint] = []
    underpowered: list[UnderpoweredWave] = []
    for wave in sorted(set(left_by_wave) | set(right_by_wave)):
        left = left_by_wave.get(wave, [])
        right = right_by_wave.get(wave, [])
        # Deep wave indices are reached by fewer episodes, so a comparison there
        # runs out of sample before it runs out of waves. Say which, rather than
        # letting the report end at whatever depth happened to be comparable.
        if len(left) < 2 or len(right) < 2:
            underpowered.append(UnderpoweredWave(wave, len(left), len(right)))
            continue
        difference, spread = _difference(
            statistic,
            left_name,
            right_name,
            left,
            right,
            confidence=confidence,
            seed=seed,
            iterations=iterations,
        )
        points.append(
            WavePoint(
                wave=wave,
                difference=difference,
                detectable_difference=detectable_difference(
                    spread, len(left), len(right), power=power
                ),
            )
        )

    pooled: float | None = None
    detectable_effect: float | None = None
    if points:
        weights = [
            float(min(point.difference.left_episodes, point.difference.right_episodes))
            for point in points
        ]
        pooled = sum(
            weight * point.difference.effect_size
            for weight, point in zip(weights, points, strict=True)
        ) / sum(weights)
        widest = max(points, key=lambda point: min(point.difference.left_episodes,
                                                   point.difference.right_episodes))
        detectable_effect = detectable_difference(
            1.0,
            widest.difference.left_episodes,
            widest.difference.right_episodes,
            power=power,
        )
    return WaveStatisticComparison(
        statistic=statistic,
        points=tuple(points),
        underpowered=tuple(underpowered),
        pooled_effect_size=pooled,
        detectable_effect_size=detectable_effect,
    )


def _compare_episode_statistic(
    statistic: str,
    left_name: str,
    right_name: str,
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    *,
    confidence: float,
    power: float,
    seed: int | None,
    iterations: int,
) -> EpisodeStatisticComparison | None:
    left = [float(record[statistic]) for record in left_records if statistic in record]
    right = [float(record[statistic]) for record in right_records if statistic in record]
    if len(left) < 2 or len(right) < 2:
        return None
    difference, spread = _difference(
        statistic,
        left_name,
        right_name,
        left,
        right,
        confidence=confidence,
        seed=seed,
        iterations=iterations,
    )
    return EpisodeStatisticComparison(
        statistic=statistic,
        difference=difference,
        detectable_difference=detectable_difference(spread, len(left), len(right), power=power),
    )


def analyse(
    left: str,
    left_records: Sequence[Mapping[str, Any]],
    right: str,
    right_records: Sequence[Mapping[str, Any]],
    *,
    confidence: float = 0.95,
    power: float = 0.8,
    seed: int | None = 0,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> EquivalenceAnalysis:
    """Compare two arms per wave index and per episode, as intervals."""
    if len(left_records) < 2 or len(right_records) < 2:
        raise ValueError("each arm needs at least two valid episodes to be compared")
    per_wave = tuple(
        _compare_wave_statistic(
            statistic,
            left,
            right,
            left_records,
            right_records,
            confidence=confidence,
            power=power,
            seed=seed,
            iterations=iterations,
        )
        for statistic in PER_WAVE_STATISTICS
    )
    per_episode = [
        _compare_episode_statistic(
            statistic,
            left,
            right,
            left_records,
            right_records,
            confidence=confidence,
            power=power,
            seed=seed,
            iterations=iterations,
        )
        for statistic in PER_EPISODE_STATISTICS
    ]
    return EquivalenceAnalysis(
        left=left,
        right=right,
        confidence=confidence,
        power=power,
        per_wave=per_wave,
        per_episode=tuple(item for item in per_episode if item is not None),
        left_episodes=len(left_records),
        right_episodes=len(right_records),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(analysis: EquivalenceAnalysis) -> str:
    """The textual report, in `compare_arms.py`'s style: intervals, never verdicts."""
    lines = [
        f"{analysis.left} vs {analysis.right}: "
        f"{analysis.left_episodes}/{analysis.right_episodes} valid episodes, "
        f"{analysis.confidence:.0%} intervals, detectable at {analysis.power:.0%} power",
        "per wave index",
    ]
    for comparison in analysis.per_wave:
        if comparison.uncaptured:
            lines.append(f" {comparison.statistic}: not captured by this run's episode records")
            continue
        separated = comparison.separated_waves
        pooled = comparison.pooled_effect_size or 0.0
        floor = comparison.detectable_effect_size or 0.0
        lines.append(
            f" {comparison.statistic}: {len(comparison.points)} wave indices compared, "
            f"separated at {list(separated) or 'none'}, "
            f"pooled d={pooled:+.3f} (a single wave index could detect d >= {floor:.3f})"
        )
        lines.extend(point.describe() for point in comparison.points)
        for sparse in comparison.underpowered:
            lines.append(
                f"  wave {sparse.wave:>3}: not compared — "
                f"{sparse.left_episodes}/{sparse.right_episodes} episodes reached it"
            )
    lines.append("per episode")
    for episode in analysis.per_episode:
        lines.append(
            f" {episode.statistic}: {episode.difference.describe()} "
            f"(could detect >= {episode.detectable_difference:.3f})"
        )
    return "\n".join(lines)


def to_record(analysis: EquivalenceAnalysis) -> dict[str, Any]:
    """The same result as JSON, for the durable experiment log."""
    return {
        "left": analysis.left,
        "right": analysis.right,
        "confidence": analysis.confidence,
        "power": analysis.power,
        "left_episodes": analysis.left_episodes,
        "right_episodes": analysis.right_episodes,
        "per_wave": [
            {
                "statistic": comparison.statistic,
                "uncaptured": comparison.uncaptured,
                "pooled_effect_size": comparison.pooled_effect_size,
                "detectable_effect_size": comparison.detectable_effect_size,
                "separated_waves": list(comparison.separated_waves),
                "waves": [_difference_record(point.difference, point.detectable_difference)
                          | {"wave": point.wave} for point in comparison.points],
                "underpowered_waves": [
                    {
                        "wave": sparse.wave,
                        "left_episodes": sparse.left_episodes,
                        "right_episodes": sparse.right_episodes,
                    }
                    for sparse in comparison.underpowered
                ],
            }
            for comparison in analysis.per_wave
        ],
        "per_episode": [
            {"statistic": episode.statistic}
            | _difference_record(episode.difference, episode.detectable_difference)
            for episode in analysis.per_episode
        ],
    }


def _difference_record(difference: Difference, detectable: float) -> dict[str, Any]:
    return {
        "left_mean": round(difference.left_mean, 4),
        "right_mean": round(difference.right_mean, 4),
        "difference": round(difference.difference, 4),
        "interval": [round(difference.low, 4), round(difference.high, 4)],
        "effect_size": round(difference.effect_size, 4),
        "left_episodes": difference.left_episodes,
        "right_episodes": difference.right_episodes,
        "separated": difference.separated,
        # Never absent: a non-significant interval without this reads as "no
        # difference", which is the one thing it does not mean.
        "detectable_difference": round(detectable, 4),
    }


def analyse_reports(
    left_path: str,
    right_path: str,
    *,
    left_arm: str | None = None,
    right_arm: str | None = None,
) -> str:
    """Compare two report JSONs by path and render the result.

    The documented one-liner, since wiring a subcommand into the device runner
    buys nothing this does not:

        uv run python -c "from tower_rl.experiment.wave_statistics import \\
            analyse_reports; print(analyse_reports('a.json', 'b.json'))"
    """
    import json
    from pathlib import Path

    left = json.loads(Path(left_path).read_text())
    right = json.loads(Path(right_path).read_text())
    return render(
        analyse(
            left_arm or Path(left_path).stem,
            episode_records(left, left_arm),
            right_arm or Path(right_path).stem,
            episode_records(right, right_arm),
        )
    )
