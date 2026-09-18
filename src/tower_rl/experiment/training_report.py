"""What one training run is read from afterwards.

The run itself (`learning/training.py`) collects and learns; this is the
record it leaves behind - the collection curve, the learning curve and the
checkpoint each of its points names, where the fleet's decision time went, and
the summary JSON that carries all of it beside the floors it is read against.
Every number here is one training or evaluation already produced; nothing is
measured again.

The writing of checkpoints belongs to the learner (`learning/checkpoint.py`);
this module decides *when* one is written and *which measurement* it stands
behind.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tower_rl.environment.decision_time import EMPTY_BREAKDOWN, DecisionTimeBreakdown
from tower_rl.experiment.metrics import (
    DECISION_TIME_INTERVAL_SECONDS,
    LearningCurvePoint,
    actor_summary,
    collected_episode_records,
    curve_metrics,
    decision_time_line,
    decision_time_metrics,
    fleet_decision_time,
    health_metrics,
    per_hour,
    pooled,
    window_line,
    window_metrics,
)
from tower_rl.experiment.run_identity import REFERENCE_FINAL_WAVES, SCRIPTED_REFERENCE
from tower_rl.experiment.tracking import TrackedRun
from tower_rl.learning.backbone import Backbone
from tower_rl.learning.checkpoint import (
    CheckpointIdentity,
    TrainingProgress,
    write_checkpoint,
)
from tower_rl.learning.evaluator import EvaluationReport, to_record
from tower_rl.learning.replay import PrioritizedSequenceReplay
from tower_rl.learning.training import (
    CollectionWindow,
    TrainingProgressReport,
    TrainingRun,
    action_distribution,
    collection_windows,
    episode_health,
)


@dataclass
class TrainingReport:
    """The record of one training run, kept as the run proceeds."""

    name: str
    run_dir: Path
    backbone: Backbone
    replay: PrioritizedSequenceReplay
    training: TrainingRun
    identity: CheckpointIdentity
    resolved: dict[str, object]
    #: The monotonic origin every curve point's wall clock is measured from.
    started: float
    #: Where this run records itself. `NoExperimentTracker` hands out a handle
    #: that keeps nothing, so the code below has no tracked and untracked paths.
    run: TrackedRun
    learning_curve: list[LearningCurvePoint] = field(default_factory=list)
    #: The collection curve: every window of collected episodes that has closed.
    #: This is the series the run is read from; the learning curve holds the one
    #: pre-registered evaluation.
    collection_curve: list[CollectionWindow] = field(default_factory=list)
    #: The weight digest of the checkpoint last written, which is what a curve
    #: point names when it says which checkpoint it corresponds to.
    last_checkpoint_fingerprint: str = ""
    #: The pre-registered final point, set by `record_point` when it records
    #: one: the headline number, and the only point the run is judged on. The
    #: report records evaluations; it does not run them.
    final_point: LearningCurvePoint | None = None
    #: The decision-time decomposition, one record per emission interval. Each
    #: record is a delta, so it describes that interval alone rather than the
    #: run's average, which is what makes a short measurement readable.
    decision_time_curve: list[dict[str, object]] = field(default_factory=list)
    #: Each actor's cumulative decomposition at the last emission, which the
    #: next one is measured against.
    decision_time_baseline: dict[str, DecisionTimeBreakdown] = field(default_factory=dict)
    decision_time_emitted: float = 0.0

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoints" / "latest.pt"

    def checkpoint(self, report: TrainingProgressReport) -> None:
        """The resume point, overwritten in place as the run proceeds."""
        self.last_checkpoint_fingerprint = self._write(report, self.checkpoint_path)

    def _write(self, report: TrainingProgressReport, path: Path) -> str:
        """Write one checkpoint and return the digest of the weights in it."""
        return write_checkpoint(
            path,
            identity=self.identity,
            progress=TrainingProgress(
                optimisation_steps=report.optimisation_steps,
                environment_decisions=report.decisions,
                episodes=report.episodes,
                # Read from the run rather than re-evaluated from its
                # schedules: the exploration rate and the importance exponent
                # are the run's to publish, and a resume has to restore what was
                # actually used.
                epsilon=report.epsilon,
                importance_beta=report.importance_beta,
            ),
            backbone_state=self.backbone.state_dict(),
            resolved_config=self.resolved,
            replay_provenance={**self.replay.snapshot(), "restored": False},
        )

    def record_point(
        self, evaluation: EvaluationReport, *, pre_registered_final: bool = False
    ) -> LearningCurvePoint:
        """Place one evaluation on the curve, against the checkpoint it scored.

        Each point gets its own checkpoint file rather than sharing the resume
        point, which is overwritten as the run proceeds: the strongest model of a
        run is the one a point names, and a fingerprint pointing at a file that
        has since moved on would name nothing.
        """
        progress = self.training.report
        window = self.training.config.collection_window_episodes
        recent = action_distribution(progress.collected[-window:])
        path = self.run_dir / "checkpoints" / f"decisions-{progress.decisions:07d}.pt"
        digest = self._write(progress, path)
        spread = evaluation.distribution
        point = LearningCurvePoint(
            decisions=progress.decisions,
            episodes=progress.episodes,
            wall_seconds=round(time.monotonic() - self.started, 1),
            model_version=evaluation.model_version,
            mean_final_wave=round(spread.mean, 3),
            # NaN is how `WaveDistribution` says a single episode has no spread.
            stdev_final_wave=round(spread.stdev, 3) if spread.stdev == spread.stdev else None,
            final_waves=[
                summary.final_wave for summary in evaluation.episodes if summary.valid
            ],
            valid_episodes=evaluation.valid_episodes,
            invalid_episodes=evaluation.invalid_episodes,
            invalid_by_reason=dict(evaluation.invalid_by_reason),
            versus_scripted_reference=round(spread.mean - SCRIPTED_REFERENCE, 3),
            checkpoint_fingerprint=digest,
            checkpoint_path=str(path),
            weighted_loss=progress.mean_recent_weighted_loss,
            unweighted_mean_absolute_td_error=(
                progress.mean_recent_unweighted_absolute_td_error
            ),
            gradient_norm=progress.mean_recent_gradient_norm,
            value_fit_correlation=progress.mean_recent_value_fit_correlation,
            collection_wait_fraction=None if recent is None else recent.wait_fraction,
            collection_purchases_per_episode=(
                None if recent is None else recent.purchases_per_episode
            ),
            pre_registered_final=pre_registered_final,
        )
        self.learning_curve.append(point)
        if pre_registered_final:
            # The headline is named here, where the point is made, rather than
            # by whoever asked for the evaluation.
            self.final_point = point
        # Keyed by decisions consumed, because that is the budget unit the
        # comparison equalises on; the checkpoint goes up under the fingerprint
        # the point names, so a tracked point resolves to an exact file.
        self.run.log_metrics(curve_metrics(point, evaluation), decisions=progress.decisions)
        self.run.log_artifact(path, directory=f"checkpoints/{digest}")
        return point

    def record_collection_windows(self) -> None:
        """Emit every window of collected episodes that has closed since the last call.

        Called per episode, and cheap: a closed window is never recomputed into a
        second point, and the series is what both the report and the tracked run
        carry the curve as.
        """
        windows = collection_windows(
            self.training.report.collected,
            size=self.training.config.collection_window_episodes,
        )
        for window in windows[len(self.collection_curve) :]:
            self.collection_curve.append(window)
            self.run.log_metrics(window_metrics(window), decisions=window.decisions_at_end)
            print(f"[{self.name}] collection: {window_line(window)}", flush=True)

    def record_decision_time(self, *, final: bool = False) -> None:
        """Emit where the fleet's decision time went since the last emission.

        Called per episode on the collecting thread, under the run's progress
        lock, and cheap: it reads snapshots the actors have already published
        and does no timing of its own. `final` flushes the tail so a short run
        still reports the interval it ended in.
        """
        now = time.monotonic()
        if not final and now - self.decision_time_emitted < DECISION_TIME_INTERVAL_SECONDS:
            return
        report = self.training.report
        current = fleet_decision_time(report)
        interval = {
            actor_id: breakdown.since(
                self.decision_time_baseline.get(actor_id, EMPTY_BREAKDOWN)
            )
            for actor_id, breakdown in current.items()
        }
        fleet = pooled(list(interval.values()))
        if fleet.decisions < 1:
            # Nothing was collected in this interval; an empty decomposition
            # would divide by zero and say nothing.
            return
        self.decision_time_baseline = current
        self.decision_time_emitted = now
        self.decision_time_curve.append(
            {
                "index": len(self.decision_time_curve),
                "decisions_at_end": report.decisions,
                "episodes_at_end": report.episodes,
                "collection_windows_closed": len(self.collection_curve),
                "actors": {
                    actor_id: breakdown.as_record() for actor_id, breakdown in interval.items()
                },
                "fleet": fleet.as_record(),
            }
        )
        self.run.log_metrics(
            decision_time_metrics(fleet, len(interval)), decisions=report.decisions
        )
        print(decision_time_line(self.name, fleet, len(interval)), flush=True)

    def summary(self) -> dict[str, object]:
        report = self.training.report
        # Flush the interval the run ended in, so a short measurement is not
        # lost for having finished between emissions.
        self.record_decision_time(final=True)
        cumulative = fleet_decision_time(report)
        distribution = action_distribution(report.collected)
        # Pooled once over every collected episode and reused for the "health"
        # key, the legacy by-reason mapping, and the MLflow metrics below: one
        # count of the run's honesty, not three.
        health = episode_health(report.episode_summaries)
        self.run.log_metrics(health_metrics(health, prefix="health_"), decisions=report.decisions)
        return {
            "backbone": self.name,
            "run_id": self.identity.run_id,
            "resolved_config": self.resolved,
            "decisions": report.decisions,
            "episodes": report.episodes,
            "valid_episodes": report.valid_episodes,
            "optimisation_steps": report.optimisation_steps,
            "mean_recent_weighted_loss": report.mean_recent_weighted_loss,
            "sequences_accepted": report.sequences_accepted,
            "wall_seconds": report.wall_seconds,
            "final_waves": report.final_waves,
            # The collection curve first: it is what the run is read from, and
            # the learning curve below it holds the pre-registered evaluation of
            # the final checkpoint, which is the headline against the floors.
            "collection_curve": [asdict(window) for window in self.collection_curve],
            "collection_window_episodes": self.training.config.collection_window_episodes,
            "final_evaluation": (
                asdict(self.final_point) if self.final_point is not None else None
            ),
            "learning_curve": [asdict(point) for point in self.learning_curve],
            "mean_recent_unweighted_absolute_td_error": (
                report.mean_recent_unweighted_absolute_td_error
            ),
            "mean_recent_gradient_norm": report.mean_recent_gradient_norm,
            "mean_recent_value_fit_correlation": report.mean_recent_value_fit_correlation,
            "action_distribution": (
                asdict(distribution) if distribution is not None else None
            ),
            "reference_final_waves": REFERENCE_FINAL_WAVES,
            # The fleet: what each actor contributed, and the aggregate rate the
            # run was actually collected at.
            "actors": [
                actor_summary(progress, report) for progress in report.actors.values()
            ],
            "actors_withdrawn": sum(
                1 for progress in report.actors.values() if progress.withdrawn is not None
            ),
            "health": asdict(health),
            # Where the fleet's wall time went: one record per emission
            # interval, and the run's totals per actor and pooled.
            "decision_time_curve": self.decision_time_curve,
            "decision_time": {
                "interval_seconds": DECISION_TIME_INTERVAL_SECONDS,
                "actors": {
                    actor_id: breakdown.as_record()
                    for actor_id, breakdown in cumulative.items()
                },
                "fleet": pooled(list(cumulative.values())).as_record(),
            },
            "episodes_per_hour": per_hour(report.episodes, report.wall_seconds),
            "decisions_per_hour": per_hour(report.decisions, report.wall_seconds),
            # Kept for compatibility with the report's earlier shape; identical
            # to `health["invalid_by_reason"]`, which is where it is now pooled.
            "invalid_episodes_by_reason": health.invalid_by_reason,
            "failed_episodes": report.failed_episodes,
            "episode_failures": report.episode_failures,
            "evaluation_failures": report.evaluation_failures,
            "checkpoints_written": report.checkpoints_written,
            "checkpoint_path": str(self.checkpoint_path),
            # Every collected episode's own record - what certifies the run
            # stayed honest for its whole span, not only in aggregate.
            "collected_episodes": collected_episode_records(report),
            "evaluations": [to_record(item) for item in report.evaluations],
            "replay": self.replay.snapshot(),
        }


__all__ = ["TrainingReport"]
