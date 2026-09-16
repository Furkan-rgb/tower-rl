"""Exploration-free evaluation that reports a distribution, never a best run.

`best` must mean the strongest checkpoint under a repeatable multi-episode
protocol.  A single high wave is noise: the scripted policy's own final wave has
a measured standard deviation of 1.22 waves (`M1B-E006`), so this module
deliberately makes it hard to report a headline number without the spread.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from tower_rl.application.actor import Actor, ActorConfig
from tower_rl.application.policies import Policy, describe
from tower_rl.application.run_environment import InstrumentedRunEnvironment
from tower_rl.domain.episode import EpisodeSummary


@dataclass(frozen=True)
class WaveDistribution:
    """The shape of an arm's performance, not just its centre."""

    episodes: int
    mean: float
    median: float
    stdev: float
    minimum: int
    maximum: int
    lower_quartile: float

    @classmethod
    def of(cls, waves: list[int]) -> WaveDistribution:
        if not waves:
            raise ValueError("a distribution needs at least one episode")
        ordered = sorted(waves)
        return cls(
            episodes=len(waves),
            mean=statistics.fmean(waves),
            median=statistics.median(waves),
            # One episode has no spread; reporting zero would imply certainty.
            stdev=statistics.stdev(waves) if len(waves) > 1 else float("nan"),
            minimum=ordered[0],
            maximum=ordered[-1],
            lower_quartile=(
                statistics.quantiles(waves, n=4)[0] if len(waves) > 3 else float(ordered[0])
            ),
        )


@dataclass(frozen=True)
class EvaluationReport:
    """An immutable result tied to exactly what produced it."""

    policy: str
    profile_id: str
    model_version: int
    game_speed: float
    valid_episodes: int
    invalid_episodes: int
    distribution: WaveDistribution
    invalid_by_reason: dict[str, int] = field(default_factory=dict)
    #: The validator text behind each invalid episode. An outcome without its
    #: reason cannot be diagnosed later (M1B-E007).
    invalid_detail: dict[str, int] = field(default_factory=dict)
    total_decisions: int = 0
    total_wall_seconds: float = 0.0

    @property
    def invalid_rate(self) -> float:
        attempted = self.valid_episodes + self.invalid_episodes
        return 0.0 if attempted == 0 else self.invalid_episodes / attempted

    def summary_line(self) -> str:
        """Deliberately puts the spread next to the mean."""
        spread = self.distribution
        return (
            f"{self.policy}: mean {spread.mean:.2f} median {spread.median:.1f} "
            f"sd {spread.stdev:.2f} range {spread.minimum}-{spread.maximum} "
            f"over {spread.episodes} valid episodes, invalid rate {self.invalid_rate:.1%}"
        )


def evaluate(
    environment: InstrumentedRunEnvironment,
    policy: Policy,
    *,
    episodes: int,
    profile_id: str,
    model_version: int = 0,
    actor_config: ActorConfig | None = None,
) -> EvaluationReport:
    """Run exploration-free episodes and report their distribution.

    Evaluation never writes to replay and never explores: epsilon is forced to
    zero here rather than trusted from configuration, because an evaluation that
    quietly explored would overstate nothing and understate everything.
    """
    base = actor_config or ActorConfig()
    config = ActorConfig(
        actor_id=base.actor_id,
        sequence_length=base.sequence_length,
        burn_in=base.burn_in,
        stride=base.stride,
        epsilon=0.0,
        max_decisions_per_episode=base.max_decisions_per_episode,
    )
    actor = Actor(environment=environment, policy=policy, config=config, replay=None)

    valid: list[EpisodeSummary] = []
    invalid: list[EpisodeSummary] = []
    decisions = 0
    wall = 0.0
    speed = 0.0
    for _ in range(episodes):
        result = actor.run_episode()
        summary = result.summary
        decisions += summary.decisions
        wall += summary.elapsed_wall_seconds
        speed = summary.game_speed
        (valid if summary.valid else invalid).append(summary)

    if not valid:
        raise ValueError("no valid episode was produced; the arm cannot be scored")

    reasons: dict[str, int] = {}
    detail: dict[str, int] = {}
    for summary in invalid:
        key = summary.termination.value
        reasons[key] = reasons.get(key, 0) + 1
        for text in summary.termination_detail or ("no detail recorded",):
            detail[text] = detail.get(text, 0) + 1

    return EvaluationReport(
        policy=describe(policy),
        profile_id=profile_id,
        model_version=model_version,
        game_speed=speed,
        valid_episodes=len(valid),
        invalid_episodes=len(invalid),
        distribution=WaveDistribution.of([summary.final_wave for summary in valid]),
        invalid_by_reason=reasons,
        invalid_detail=detail,
        total_decisions=decisions,
        total_wall_seconds=round(wall, 2),
    )


def to_record(report: EvaluationReport) -> dict[str, Any]:
    """A flat record for the durable experiment log."""
    spread = report.distribution
    return {
        "policy": report.policy,
        "profile_id": report.profile_id,
        "model_version": report.model_version,
        "game_speed": report.game_speed,
        "valid_episodes": report.valid_episodes,
        "invalid_episodes": report.invalid_episodes,
        "invalid_rate": round(report.invalid_rate, 4),
        "invalid_by_reason": report.invalid_by_reason,
        "invalid_detail": report.invalid_detail,
        "mean_final_wave": round(spread.mean, 3),
        "median_final_wave": spread.median,
        "stdev_final_wave": round(spread.stdev, 3) if spread.stdev == spread.stdev else None,
        "lower_quartile_final_wave": spread.lower_quartile,
        "minimum_final_wave": spread.minimum,
        "maximum_final_wave": spread.maximum,
        "total_decisions": report.total_decisions,
        "total_wall_seconds": report.total_wall_seconds,
    }
