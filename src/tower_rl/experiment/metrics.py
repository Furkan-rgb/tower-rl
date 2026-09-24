"""What a run's measurements aggregate to, as plain numbers and plain records.

Pure functions over what training and evaluation already measured: nothing here
times anything, reads a device, or decides anything about the run. One shape per
scope - the run, one collection window, one actor, one evaluation - so a health
problem can be placed on the budget axis rather than only read off a total.

MLflow metrics are scalars, so anything a number cannot carry (a by-reason
mapping, a detail string) stays in the JSON report the same functions produce.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from tower_rl.environment.decision_time import (
    BUCKETS,
    EMPTY_BREAKDOWN,
    DecisionTimeBreakdown,
)
from tower_rl.environment.episode import EpisodeSummary
from tower_rl.experiment.run_identity import SCRIPTED_REFERENCE
from tower_rl.learning.evaluator import EvaluationReport, episode_record
from tower_rl.learning.training import (
    ActorProgress,
    CollectedEpisode,
    CollectionWindow,
    EpisodeHealth,
    SelectionPeriod,
    TrainingProgressReport,
    episode_health,
)


@dataclass(frozen=True)
class LearningCurvePoint:
    """One exploration-free measurement of an arm, placed on its budget.

    This is the artefact the run is read from: whether the number is going up,
    how far into the budget it got there, and which checkpoint on disk produced
    it. Everything here is either a cost already paid or a measurement already
    taken; nothing is inferred.
    """

    #: Where on the budget this point sits, and what it cost to get here.
    decisions: int
    episodes: int
    wall_seconds: float
    #: Optimisation steps applied to the weights that were evaluated.
    model_version: int
    #: The evaluation itself, at epsilon 0, never written to replay.
    mean_final_wave: float
    #: None for a single-episode sample, which has no spread to report.
    stdev_final_wave: float | None
    final_waves: list[int]
    valid_episodes: int
    invalid_episodes: int
    invalid_by_reason: dict[str, int]
    #: The scripted floor is the bar; the difference is spelled out rather than
    #: left to the reader to subtract.
    versus_scripted_reference: float
    #: The checkpoint holding exactly the weights this point scored.
    checkpoint_fingerprint: str
    checkpoint_path: str
    #: The learner's health at this point, as the diagnostics name it. The loss
    #: carries the importance-sampling weights and the TD error does not, which
    #: is why they are never reported under one name.
    weighted_loss: float | None
    unweighted_mean_absolute_td_error: float | None
    gradient_norm: float | None
    #: Predicted value against realised discounted return. Near zero with the
    #: other signals healthy means the learner is not learning the return.
    value_fit_correlation: float | None
    #: What the collecting policy did over the last window of episodes. The
    #: random baseline buys 18.7 upgrades per episode.
    collection_wait_fraction: float | None
    collection_purchases_per_episode: float | None
    #: True for the single pre-registered exploration-free evaluation of the
    #: final checkpoint, which is the headline number against the scripted
    #: reference. Every other point is incidental.
    pre_registered_final: bool = False

    def line(self) -> str:
        spread = "n/a" if self.stdev_final_wave is None else f"{self.stdev_final_wave:.2f}"
        return (
            f"decisions {self.decisions} wall {self.wall_seconds:.0f}s "
            f"mean final wave {self.mean_final_wave:.2f} sd {spread} "
            f"vs scripted {SCRIPTED_REFERENCE}: {self.versus_scripted_reference:+.2f} "
            f"({self.valid_episodes} valid, {self.invalid_episodes} invalid)"
        )


def curve_metrics(
    point: LearningCurvePoint,
    evaluation: EvaluationReport,
) -> dict[str, float]:
    """What one curve point is worth tracking for, keyed by nothing but itself.

    Every number here is already measured; none is instrumented for tracking.
    The learner health signals are the ones `learn` returns anyway - the
    weighted loss, the unweighted TD error magnitude, the gradient norm and the
    value fit - averaged over the last hundred steps.
    """
    waves = sum(point.final_waves)
    metrics: dict[str, float] = {
        "eval_mean_final_wave": point.mean_final_wave,
        "eval_valid_episodes": float(point.valid_episodes),
        "eval_invalid_episodes": float(point.invalid_episodes),
        "versus_scripted_reference": point.versus_scripted_reference,
        "episodes": float(point.episodes),
        "optimisation_steps": float(point.model_version),
        "wall_seconds": point.wall_seconds,
    }
    if point.stdev_final_wave is not None:
        metrics["eval_stdev_final_wave"] = point.stdev_final_wave
    if waves:
        # Device cost per wave reached, measured exploration-free: the density
        # the budget is actually spent at.
        metrics["eval_decisions_per_wave"] = evaluation.decisions_in_valid_episodes / waves
    health = {
        # Weighted and unweighted are spelled out in the key itself: reading one
        # as the other is what made the first run look like it was learning.
        "learner_weighted_loss_with_is_weights": point.weighted_loss,
        "learner_unweighted_mean_absolute_td_error": point.unweighted_mean_absolute_td_error,
        "learner_gradient_norm": point.gradient_norm,
        "learner_value_fit_correlation": point.value_fit_correlation,
        "collection_wait_fraction": point.collection_wait_fraction,
        "collection_purchases_per_episode": point.collection_purchases_per_episode,
    }
    metrics.update({key: value for key, value in health.items() if value is not None})
    return metrics


def health_metrics(health: EpisodeHealth, *, prefix: str) -> dict[str, float]:
    """The numeric fields of `EpisodeHealth`, keyed for MLflow.

    MLflow metrics are scalars, so `invalid_by_reason` and `invalid_detail` stay
    in the JSON report only; everything a health problem is *located* by - not
    just named - travels to the tracker too.
    """
    metrics = {
        f"{prefix}valid_episodes": float(health.valid_episodes),
        f"{prefix}invalid_episodes": float(health.invalid_episodes),
        f"{prefix}advances_cut_short": float(health.advances_cut_short),
        f"{prefix}pin_restarts": float(health.pin_restarts),
        f"{prefix}episodes_not_started_fresh": float(health.episodes_not_started_fresh),
        f"{prefix}bridge_event_divergence": float(health.bridge_event_divergence),
        f"{prefix}stale_or_duplicate": float(health.stale_or_duplicate),
        f"{prefix}game_time_inflated": float(health.game_time_inflated),
    }
    if health.round_budgeted_ratio is not None:
        metrics[f"{prefix}round_budgeted_ratio"] = health.round_budgeted_ratio
    if health.worst_round_budgeted_ratio is not None:
        metrics[f"{prefix}worst_round_budgeted_ratio"] = health.worst_round_budgeted_ratio
    return metrics


def episode_metrics(
    episode: CollectedEpisode,
    *,
    actor_index: int,
    epsilon: float,
    cumulative_game_ms: float,
) -> dict[str, float]:
    """One collected episode, as the tracked unit it is.

    The collection window beside this is the smoothed view - a hundred episodes
    averaged - and it closes about once an hour, which is far too coarse to see
    where a run turned. This is the raw series under it: one point per episode,
    keyed by the decisions spent when that episode ended, so it lines up with
    the checkpoint points on the same axis.

    `episode_valid` is a number rather than a flag because a metric store holds
    numbers; read as a rate over a window it is the run's honesty, which is
    exactly what an unattended overnight run has to be readable on.
    """
    summary = episode.summary
    return {
        "episode_final_wave": float(summary.final_wave),
        "episode_decisions": float(summary.decisions),
        # The game's own round clock, not frames times `frame_game_ms`: the
        # latter is what the advances asked for, which is not evidence.
        "episode_game_ms": float(summary.round_ms),
        # The game time spent by the end of this episode, a statistic beside
        # the decisions the series is keyed by.
        "episode_game_seconds_cumulative": cumulative_game_ms / 1000.0,
        "episode_wait_fraction": episode.wait_fraction,
        "episode_purchases": float(summary.purchases),
        "episode_valid": 1.0 if summary.valid else 0.0,
        # Which instance played it, as a number: a fleet's episodes are one
        # series, and an actor that starts trailing the others is invisible in
        # the aggregate until it withdraws.
        "episode_actor": float(actor_index),
        "episode_epsilon": epsilon,
    }


def learner_metrics(report: TrainingProgressReport) -> dict[str, float]:
    """What the learner has been doing lately, on the episode's key.

    These are the learner's own trailing summaries over its last hundred
    optimisation steps - it already keeps them - sampled here rather than
    computed. Until now they reached the store only through a curve point, and
    a run with no mid-run evaluation therefore reported them exactly once, at
    the end, which is not a curve.
    """
    # A quantity the learner has not measured yet is absent rather than zero: a
    # zero loss before the first optimisation step would read as a solved
    # problem, and a zero correlation as a learner predicting nothing.
    measured: dict[str, float | None] = {
        "learner_optimisation_steps": float(report.optimisation_steps),
        # The game time spent at this point of the decision axis.
        "learner_game_seconds": report.game_seconds,
        "learner_importance_beta": report.importance_beta,
        "learner_weighted_loss": report.mean_recent_weighted_loss,
        "learner_unweighted_mean_absolute_td_error": (
            report.mean_recent_unweighted_absolute_td_error
        ),
        "learner_gradient_norm": report.mean_recent_gradient_norm,
        "learner_value_fit_correlation": report.mean_recent_value_fit_correlation,
    }
    return {name: value for name, value in measured.items() if value is not None}


def window_metrics(
    window: CollectionWindow, *, actor_index: Mapping[str, int]
) -> dict[str, float]:
    """One point of the collection curve, which is what the run is read from.

    The keys, all on the window's `decisions_at_end`:

    - `collection_mean_final_wave`, `collection_versus_scripted_reference`,
      `collection_episodes`, `collection_stdev_final_wave`,
      `collection_standard_error` - the pooled window over every actor;
    - `collection_window_wait_fraction`,
      `collection_window_purchases_per_episode` - what the policy did in it;
    - `collection_window_mean_final_wave_actor{i}` - the same window for actor
      `i` of the fleet alone, absent for an actor with no valid episode in it.
      Under an exploration ladder the actors sit at rates two orders of
      magnitude apart and the pooled mean above is nobody's performance;
    - `collection_window_near_greedy_mean_final_wave` and
      `collection_window_near_greedy_episodes` - the window pooled over the
      near-greedy actors only, which is the series a readout of what the policy
      itself reaches cites. Identical to the pooled series under a uniform
      schedule, where every actor is near-greedy;
    - `collection_window_*` health counters, from `health_metrics`.

    `actor_index` places each actor id on the series index it reports under -
    the fleet's own order, the same one `episode_actor` uses.
    """
    metrics = {
        "collection_mean_final_wave": window.mean_final_wave,
        "collection_versus_scripted_reference": window.mean_final_wave - SCRIPTED_REFERENCE,
        "collection_episodes": float(window.episodes),
        "collection_window_wait_fraction": window.wait_fraction,
        "collection_window_purchases_per_episode": window.purchases_per_episode,
        "collection_window_near_greedy_episodes": float(window.near_greedy_episodes),
    }
    if window.near_greedy_mean_final_wave is not None:
        metrics["collection_window_near_greedy_mean_final_wave"] = (
            window.near_greedy_mean_final_wave
        )
    for actor_id, mean in window.mean_final_wave_by_actor.items():
        index = actor_index.get(actor_id)
        if index is None:
            # An id the fleet does not know has no series to go on; the pooled
            # numbers above already carry its episodes.
            continue
        metrics[f"collection_window_mean_final_wave_actor{index}"] = mean
    if window.stdev_final_wave is not None and window.standard_error is not None:
        metrics["collection_stdev_final_wave"] = window.stdev_final_wave
        metrics["collection_standard_error"] = window.standard_error
    # The window's own health, so a problem can be placed on the budget axis
    # rather than only read off the run's total.
    metrics.update(health_metrics(window.health, prefix="collection_window_"))
    return metrics


def selection_period_metrics(period: SelectionPeriod) -> dict[str, float]:
    """One closed selection period: the series the arm is chosen on.

    The collection window beside it is cut in episodes and smooths the curve;
    this is cut in decisions, where a numbered checkpoint is written, so a point
    here belongs to a checkpoint on disk. Keyed on the decisions at the close,
    like every other series in the store. A period no near-greedy actor
    finished a valid episode in carries no mean at all rather than a zero,
    which would read as a policy that reached wave nothing.
    """
    metrics = {
        "selection_period": float(period.index),
        "selection_period_near_greedy_episodes": float(period.near_greedy_episodes),
    }
    if period.mean_final_wave is not None:
        metrics["selection_period_near_greedy_mean_final_wave"] = period.mean_final_wave
    if period.best_mean_final_wave is not None:
        metrics["selection_period_best_near_greedy_mean_final_wave"] = (
            period.best_mean_final_wave
        )
    return metrics


def selection_period_line(period: SelectionPeriod) -> str:
    mean = "n/a" if period.mean_final_wave is None else f"{period.mean_final_wave:.2f}"
    best = (
        "n/a"
        if period.best_mean_final_wave is None
        else f"{period.best_mean_final_wave:.2f}"
    )
    return (
        f"period {period.index} at {period.decisions_at_end} decisions: "
        f"near-greedy mean final wave {mean} over {period.near_greedy_episodes} "
        f"episodes, best {best}"
    )


def window_line(window: CollectionWindow) -> str:
    error = "n/a" if window.standard_error is None else f"{window.standard_error:.2f}"
    health = window.health
    return (
        f"window {window.index} decisions {window.decisions_at_end} "
        f"mean final wave {window.mean_final_wave:.2f} se {error} "
        f"over {window.episodes} collected episodes, "
        f"wait {window.wait_fraction:.1%} purchases/episode "
        f"{window.purchases_per_episode:.1f} "
        f"(invalid {health.invalid_episodes} cut_short {health.advances_cut_short} "
        f"divergence {health.bridge_event_divergence})"
    )


def collected_episode_records(report: TrainingProgressReport) -> list[dict[str, object]]:
    """Every collected episode's record, reusing the evaluator's shape.

    `episode_record` is what `run_episodes.py` and `compare_arms.py` already
    serialise per-episode records with; this is that same shape, plus the actor
    id, since a fleet's episodes are one series and a health problem must be
    traceable back to the instance that produced it.
    """
    return [
        {**episode_record(index, episode.summary), "actor_id": episode.actor_id}
        for index, episode in enumerate(report.collected)
    ]


def health_counters(summaries: Sequence[EpisodeSummary]) -> dict[str, object]:
    """`EpisodeHealth`, as a plain dict for the JSON report.

    One shape shared by the whole run, each actor and each collection window
    (`training.episode_health`); a fleet is watched by the same counters at
    every one of those scopes.
    """
    return asdict(episode_health(summaries))


#: How often the decision-time decomposition is emitted, in seconds of the
#: run's wall clock, checked at episode boundaries. Deliberately a time cadence
#: rather than the collection window: a window is a hundred episodes and closes
#: about once an hour, and the question this measurement exists to answer - is
#: the host idle on its emulators or contended in Python - has to be readable
#: from a few minutes of steady state, not from a whole run.
DECISION_TIME_INTERVAL_SECONDS = 30.0


def pooled(breakdowns: list[DecisionTimeBreakdown]) -> DecisionTimeBreakdown:
    """Every actor's decomposition added into one, the fleet's.

    Aggregation, so it belongs here rather than beside the profile the actors
    write: nothing in the loop pools anything, only a report does.
    """
    total = EMPTY_BREAKDOWN
    for item in breakdowns:
        total = total + item
    return total


def fleet_decision_time(report: TrainingProgressReport) -> dict[str, DecisionTimeBreakdown]:
    """Each actor's cumulative time decomposition, as it last published it.

    Read on an actor's thread while it holds the run's progress lock, which is
    the same lock every actor publishes its own snapshot under.
    """
    return {
        actor_id: progress.decision_time or EMPTY_BREAKDOWN
        for actor_id, progress in report.actors.items()
    }


def decision_time_metrics(fleet: DecisionTimeBreakdown, actors: int) -> dict[str, float]:
    """The decomposition as MLflow scalars, in milliseconds per decision.

    `busy_fraction` is the headline: the share of an actor thread's wall time
    that was executing Python at all. A fleet idle on its emulators sits low
    and flat as actors are added; a fleet contending for the interpreter does
    not.
    """
    per_decision = 1000.0 / fleet.decisions if fleet.decisions else 0.0
    metrics = {
        "decision_wall_ms": round(fleet.elapsed_seconds * per_decision, 3),
        "decision_cpu_ms": round(fleet.cpu_seconds * per_decision, 3),
        "decision_busy_fraction": round(fleet.busy_fraction, 4),
        "decision_accounting_error_ms": round(fleet.accounting_error_seconds * 1000, 6),
        "decisions_per_hour_per_actor": (
            round(fleet.decisions_per_hour, 1) if actors else 0.0
        ),
    }
    for name in BUCKETS:
        bucket = fleet.buckets[name]
        metrics[f"decision_wall_ms_{name}"] = round(bucket.wall_seconds * per_decision, 3)
        metrics[f"decision_cpu_ms_{name}"] = round(bucket.cpu_seconds * per_decision, 3)
    return metrics


def decision_time_line(name: str, fleet: DecisionTimeBreakdown, actors: int) -> str:
    per_decision = 1000.0 / fleet.decisions if fleet.decisions else 0.0
    parts = " ".join(
        f"{label} {fleet.buckets[key].wall_seconds * per_decision:.1f}"
        f"/{fleet.buckets[key].cpu_seconds * per_decision:.1f}"
        for key, label in (
            ("bridge_round_trip", "bridge"),
            ("observation_decode", "observe"),
            ("policy_forward", "policy"),
            ("learner_step", "learn"),
            ("blocked", "blocked"),
            ("residual", "residual"),
        )
    )
    return (
        f"[{name}] decision time over {fleet.decisions} decisions on {actors} actors: "
        f"total {fleet.elapsed_seconds * per_decision:.1f}ms wall "
        f"{fleet.cpu_seconds * per_decision:.1f}ms cpu, "
        f"busy {fleet.busy_fraction:.1%}, "
        f"{fleet.decisions_per_hour:.0f} decisions/hour per actor; "
        f"wall/cpu ms per decision: {parts}"
    )


def per_hour(count: float, wall_seconds: float) -> float:
    """A rate over the fleet's wall clock, which is the device time it cost.

    N actors collecting for an hour bought one hour of device time however many
    of them were alive for it, so the fleet's own clock is the denominator.
    """
    return round(count / wall_seconds * 3600, 1) if wall_seconds > 0 else 0.0


def actor_summary(
    progress: ActorProgress,
    report: TrainingProgressReport,
) -> dict[str, object]:
    """What one actor of the fleet contributed, beside the aggregate.

    `episodes` counts every attempt including the ones the port never delivered
    a summary for; the health counters below are pooled over the summaries that
    were delivered, which is one episode fewer whenever the port refused one.
    """
    summaries = [episode.summary for episode in report.episodes_of(progress.actor_id)]
    return {
        # The health counters first, so the identity and attempt-counting keys
        # below - which count every attempt, not only the ones with a summary -
        # are what wins where the two would otherwise collide on "episodes".
        **health_counters(summaries),
        "actor_id": progress.actor_id,
        "episodes": progress.episodes,
        "decisions": progress.decisions,
        # The game time this actor's instance played, a statistic.
        "game_seconds": round(progress.game_ms / 1000.0, 3),
        "failed_episodes": progress.failed_episodes,
        # Set only for an actor whose instance failed every episode the limit
        # allows; the rest of the fleet kept collecting without it.
        "withdrawn": progress.withdrawn,
        "episodes_per_hour": per_hour(progress.episodes, report.wall_seconds),
        "decisions_per_hour": per_hour(progress.decisions, report.wall_seconds),
        "game_seconds_per_hour": per_hour(progress.game_ms / 1000.0, report.wall_seconds),
        # This actor's own wall time, decomposed. `decisions_per_hour` above is
        # taken over the fleet's clock; the one inside this record is taken over
        # the actor's own collecting time, which is what a per-actor rate means.
        "decision_time": (
            progress.decision_time.as_record()
            if progress.decision_time is not None
            else EMPTY_BREAKDOWN.as_record()
        ),
    }


__all__ = [
    "DECISION_TIME_INTERVAL_SECONDS",
    "LearningCurvePoint",
    "actor_summary",
    "collected_episode_records",
    "curve_metrics",
    "decision_time_line",
    "decision_time_metrics",
    "fleet_decision_time",
    "health_counters",
    "health_metrics",
    "per_hour",
    "pooled",
    "selection_period_line",
    "selection_period_metrics",
    "window_line",
    "window_metrics",
]
