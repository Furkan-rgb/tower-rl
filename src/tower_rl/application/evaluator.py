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
    #: Decisions taken in the valid episodes alone. Density is a property of the
    #: episodes it is reported over, so an invalid episode's decisions must not
    #: be divided by a count that excludes the episode itself.
    decisions_in_valid_episodes: int = 0
    total_wall_seconds: float = 0.0
    #: Frames the bridge stepped, summed over every attempted episode.
    total_frames: int = 0
    #: Frames times `frame_game_ms`: the game time the advances *instructed* the
    #: world to be worth, not a measurement of what it delivered. Kept for
    #: diagnosis, as the denominator of the acceptance ratio below.
    total_budgeted_game_seconds: float = 0.0
    #: The game's own per-round clock over the same advances, and the only
    #: measurement of game time here. Against `total_budgeted_game_seconds` it
    #: shows whether the budgeted game time was really delivered; the two should
    #: agree to within rounding.
    total_round_seconds: float = 0.0
    #: Wall seconds spent inside advances. `total_wall_seconds` minus this is
    #: what the decision boundaries themselves cost.
    total_advance_wall_seconds: float = 0.0
    #: Advances the bridge stopped mid-loop without spending the budget and
    #: without an event the settled snapshot corroborates (M1B-E032). Not the
    #: wall-time ceiling, which fails the episode by name instead.
    advances_cut_short: int = 0
    #: Every attempted episode, valid and invalid alike, in the order they ran.
    #: A statistical comparison needs the per-episode samples, not just the
    #: aggregates above; this is what feeds bootstrap intervals and Cohen's d
    #: (`comparison.py`) without a throwaway observer wrapper.
    episodes: tuple[EpisodeSummary, ...] = ()
    #: Episodes whose first state was already past wave 1: the episode
    #: continued a leftover run instead of starting fresh. Not excluded, only
    #: counted, so contamination stays visible in an unattended run.
    episodes_not_started_fresh: int = 0

    @property
    def invalid_rate(self) -> float:
        attempted = self.valid_episodes + self.invalid_episodes
        return 0.0 if attempted == 0 else self.invalid_episodes / attempted

    @property
    def decisions_per_episode(self) -> float:
        """Mean decisions in a valid episode; the 1x reference of 89.3 is one.

        Over valid episodes only, numerator and denominator alike. An invalid
        episode is an environment failure, and folding its decisions into a mean
        of the episodes that survived would inflate the density of exactly the
        arms that failed most.
        """
        if self.valid_episodes == 0:
            return 0.0
        return self.decisions_in_valid_episodes / self.valid_episodes

    @property
    def decisions_per_wave(self) -> float:
        """Density per wave, which is what speed used to erode (M1B-E012)."""
        mean_wave = self.distribution.mean
        if mean_wave <= 0:
            return 0.0
        return self.decisions_per_episode / mean_wave

    @property
    def speedup(self) -> float:
        """Measured game seconds bought per wall second; the point of stepping frames.

        Measured on the game's own round clock, not on frames times
        `frame_game_ms`: the latter is what the advances asked for, and asking is
        not evidence that the world simulated it.
        """
        if self.total_wall_seconds <= 0:
            return 0.0
        return self.total_round_seconds / self.total_wall_seconds

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
    attempted: list[EpisodeSummary] = []
    decisions = 0
    wall = 0.0
    frames = 0
    game_ms = 0.0
    round_ms = 0.0
    advance_wall = 0.0
    cut_short = 0
    speed = 0.0
    for _ in range(episodes):
        result = actor.run_episode()
        summary = result.summary
        decisions += summary.decisions
        wall += summary.elapsed_wall_seconds
        frames += summary.frames
        game_ms += summary.game_ms
        round_ms += summary.round_ms
        advance_wall += summary.advance_wall_seconds
        cut_short += summary.advances_cut_short
        speed = summary.game_speed
        attempted.append(summary)
        (valid if summary.valid else invalid).append(summary)

    reasons: dict[str, int] = {}
    detail: dict[str, int] = {}
    for summary in invalid:
        key = summary.termination.value
        reasons[key] = reasons.get(key, 0) + 1
        for text in summary.termination_detail or ("no detail recorded",):
            detail[text] = detail.get(text, 0) + 1

    if not valid:
        # An arm that scores nothing still knows why every episode failed, and
        # that is the only thing it has to say. Carrying the reasons into the
        # error keeps a wholly failed arm from reporting silence.
        why = ", ".join(f"{text} (x{count})" for text, count in sorted(detail.items()))
        raise ValueError(
            "no valid episode was produced; the arm cannot be scored. "
            f"{len(invalid)} invalid episodes: {why or 'no detail recorded'}"
        )

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
        decisions_in_valid_episodes=sum(summary.decisions for summary in valid),
        total_wall_seconds=round(wall, 2),
        total_frames=frames,
        total_budgeted_game_seconds=round(game_ms / 1000.0, 3),
        total_round_seconds=round(round_ms / 1000.0, 3),
        total_advance_wall_seconds=round(advance_wall, 3),
        advances_cut_short=cut_short,
        episodes=tuple(attempted),
        episodes_not_started_fresh=sum(1 for summary in attempted if summary.starting_wave > 1),
    )


def episode_record(index: int, summary: EpisodeSummary) -> dict[str, Any]:
    """One episode's shape for the durable log; valid and invalid alike.

    This is what `comparison.py`'s bootstrap intervals, Cohen's d, and
    `required_episodes` consume — per-episode samples, not the aggregates
    above. Shared between `evaluator.py` and any caller that collects
    `EpisodeSummary` itself (`compare_arms.py`), so the shape is not
    duplicated per call site.
    """
    return {
        "episode_index": index,
        "valid": summary.valid,
        "final_wave": summary.final_wave,
        "decisions": summary.decisions,
        "purchases": summary.purchases,
        "frames": summary.frames,
        "budgeted_game_ms": summary.game_ms,
        "round_ms": summary.round_ms,
        "advance_wall_seconds": summary.advance_wall_seconds,
        "elapsed_wall_seconds": summary.elapsed_wall_seconds,
        # The reasons the episode was invalid; empty for a valid episode. Today
        # this is the same tuple as `termination_detail` because every invalid
        # transition here also terminates the episode, but the two are kept as
        # separate keys because they answer different questions: this one asks
        # why the episode is excluded from scoring, the other how it ended.
        "invalid_reasons": summary.termination_detail if not summary.valid else (),
        "termination_detail": summary.termination_detail,
        "advances_cut_short": summary.advances_cut_short,
        "recovered_transients": summary.recovered_transients,
        "starting_wave": summary.starting_wave,
    }


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
        "decisions_in_valid_episodes": report.decisions_in_valid_episodes,
        "total_wall_seconds": report.total_wall_seconds,
        "total_advance_wall_seconds": report.total_advance_wall_seconds,
        "decisions_per_episode": round(report.decisions_per_episode, 3),
        "decisions_per_wave": round(report.decisions_per_wave, 3),
        "total_frames": report.total_frames,
        "total_budgeted_game_seconds": report.total_budgeted_game_seconds,
        "total_round_seconds": report.total_round_seconds,
        "advances_cut_short": report.advances_cut_short,
        "speedup": round(report.speedup, 3),
        "episodes_not_started_fresh": report.episodes_not_started_fresh,
        "episodes": [
            episode_record(index, summary) for index, summary in enumerate(report.episodes)
        ],
    }
