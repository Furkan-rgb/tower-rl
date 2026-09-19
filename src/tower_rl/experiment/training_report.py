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
    checkpoint_period_line,
    checkpoint_period_metrics,
    collected_episode_records,
    curve_metrics,
    decision_time_line,
    decision_time_metrics,
    episode_metrics,
    fleet_decision_time,
    health_metrics,
    learner_metrics,
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
    #: What the parent segment had already spent, when this run resumed one.
    #: Zero for a run that started from scratch. This is the one source of that
    #: offset: both tracked series are placed on the whole run's budget from it
    #: - the episode axis below starts at it, and the collection windows are cut
    #: from it - and the rates at the end of `summary` subtract it so they
    #: describe this segment rather than the parent's decisions over this
    #: segment's clock.
    resumed_decisions: int = 0
    resumed_episodes: int = 0
    #: What the parent segment had already spent of the budget, which is the
    #: same offset in the unit the budget is counted in.
    resumed_game_ms: float = 0.0
    #: How far the episode series has been reported: collected episodes already
    #: sent to the tracker, and the decisions spent by the end of the last of
    #: them. The episode is the tracked unit, so both are carried rather than
    #: recomputed - an episode joins the series when it ends, and is never
    #: reported twice. `decisions_logged` tracks
    #: `TrainingProgressReport.decisions` exactly, so it begins at the resume
    #: point rather than at zero; `episodes_logged` begins at zero either way,
    #: because the episode list itself is not restored - only the decisions
    #: behind it. A running sum rather than a prefix sum over `collected`, which
    #: would re-add the whole list once per episode over the thousands a
    #: multi-hour run collects.
    episodes_logged: int = field(init=False, default=0)
    decisions_logged: int = field(init=False, default=0)
    #: The same running sum in game time, so each episode's point carries the
    #: budget position it ended at as well as the decisions axis it is keyed on.
    game_ms_logged: float = field(init=False, default=0.0)
    #: How many closed checkpoint periods have been reported. The periods
    #: themselves belong to the run - it is the run that decides whether it is
    #: still improving - and this is only how far the record has followed it.
    periods_logged: int = field(init=False, default=0)
    #: The tracking run this report's checkpoints name as their own, so a resume
    #: from one of them can continue that series. None when untracked.
    tracking_run_id: str | None = None

    def __post_init__(self) -> None:
        self.decisions_logged = self.resumed_decisions
        self.game_ms_logged = self.resumed_game_ms

    @property
    def near_greedy_actor_ids(self) -> frozenset[str]:
        """The actors whose episodes read as the policy's performance, not search.

        The run's own answer: it is the run that holds the fleet's order and the
        exploration schedule read against it, and a second definition here could
        cut the collection curve's near-greedy series over one set of actors
        while the run stopped itself on another.
        """
        return self.training.near_greedy_actor_ids

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoints" / "latest.pt"

    def checkpoint(self, report: TrainingProgressReport) -> None:
        """The resume point, overwritten in place as the run proceeds."""
        self.last_checkpoint_fingerprint = self._write(report, self.checkpoint_path)

    def numbered_checkpoint(self, report: TrainingProgressReport) -> None:
        """One candidate model of the run, named by the game time behind it.

        Beside `latest.pt` rather than instead of it: the resume point is
        overwritten as the run proceeds and therefore names no particular model,
        while these are the arms a later evaluation chooses among. The name
        carries the game seconds actually spent when it was written - the
        counter lands past its period, not on it, because an episode is played
        to its classified end - so a file says what it cost rather than what it
        was aimed at. The `gs` prefix on the number is what tells a file of this
        run from a `checkpoint-<decisions>.pt` of run 1, whose number counts
        something else entirely.
        """
        path = (
            self.run_dir
            / "checkpoints"
            / f"checkpoint-gs{int(report.game_seconds):07d}.pt"
        )
        digest = self._write(report, path)
        # Under the same tracked run as every metric this report logs, and under
        # the weight digest a later reading names it by: a candidate the run's
        # own page could not reach would have to be found by hand, months later,
        # from a path in a JSON file.
        self.run.log_artifact(path, directory=f"checkpoints/{digest}")
        print(f"[{self.name}] checkpoint {path.name}", flush=True)

    def _write(self, report: TrainingProgressReport, path: Path) -> str:
        """Write one checkpoint and return the digest of the weights in it."""
        return write_checkpoint(
            path,
            identity=self.identity,
            progress=TrainingProgress(
                optimisation_steps=report.optimisation_steps,
                environment_decisions=report.decisions,
                environment_game_ms=report.game_ms,
                episodes=report.episodes,
                # Read from the run rather than re-evaluated from its
                # schedules: the schedule position and the importance exponent
                # are the run's to publish, and a resume has to restore what was
                # actually used. Under a ladder the exploration figure is
                # informational - the actors were at rates of their own, and the
                # per-episode series is where those are read.
                epsilon=report.epsilon,
                importance_beta=report.importance_beta,
                # The early-stopping tracker, so a resume continues the one
                # near-greedy curve the run is judged on instead of starting
                # its plateau count over.
                checkpoint_periods_closed=report.plateau.periods_closed,
                best_period_near_greedy_mean=report.plateau.best_mean_final_wave,
                periods_without_improvement=report.plateau.periods_without_improvement,
            ),
            backbone_state=self.backbone.state_dict(),
            resolved_config=self.resolved,
            replay_provenance={**self.replay.snapshot(), "restored": False},
            tracking_run_id=self.tracking_run_id,
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

    def record_episodes(self) -> None:
        """Report every collected episode that has not been reported yet.

        The episode is the tracked unit. The collection curve beside this is the
        smoothed view and closes about once an hour, which is far too coarse to
        show where a run turned; this is the series under it, one point per
        episode, keyed by the decisions spent when that episode ended so it
        shares an axis with the checkpoints. The learner's own trailing
        summaries go up on the same key, because a run with no mid-run
        evaluation would otherwise report them exactly once, at the end.

        Called per episode, on the collecting thread, under the run's progress
        lock. It reads what has already been measured and measures nothing.
        """
        report = self.training.report
        actors = self.training.actor_index
        exploration = self.training.config.exploration
        learner = learner_metrics(report)
        for episode in report.collected[self.episodes_logged :]:
            # The decisions at the end of this episode, not the run's current
            # total: the two differ whenever more than one episode has arrived
            # since the last call, and a point on the wrong key is a point on
            # the wrong part of the curve.
            self.decisions_logged += episode.summary.decisions
            self.game_ms_logged += episode.summary.round_ms
            self.episodes_logged += 1
            index = actors.get(episode.actor_id, -1)
            self.run.log_metrics(
                {
                    **episode_metrics(
                        episode,
                        actor_index=index,
                        # Where this episode left the budget, on the axis the
                        # run is actually spent against. The step below stays
                        # decisions, which is monotone and comparable with
                        # every series already recorded.
                        cumulative_game_ms=self.game_ms_logged,
                        # The rate the actor that played this episode was
                        # exploring at, where its episode ended - not the run's
                        # published one, which under a ladder is some other
                        # actor's rung entirely.
                        epsilon=(
                            report.epsilon
                            if index < 0
                            else exploration.epsilon_for(index, self.decisions_logged)
                        ),
                    ),
                    **learner,
                },
                decisions=self.decisions_logged,
            )

    def record_collection_windows(self) -> None:
        """Emit every window of collected episodes that has closed since the last call.

        Called per episode, and cheap: a closed window is never recomputed into a
        second point, and the series is what both the report and the tracked run
        carry the curve as.
        """
        windows = collection_windows(
            self.training.report.collected,
            size=self.training.config.collection_window_episodes,
            # This segment's episodes, on the whole run's budget: the curve of a
            # run trained in two sittings is one series, and a window keyed from
            # zero would land underneath the parent's own points.
            spent_before=self.resumed_decisions,
            near_greedy_actor_ids=self.near_greedy_actor_ids,
        )
        for window in windows[len(self.collection_curve) :]:
            self.collection_curve.append(window)
            self.run.log_metrics(
                window_metrics(window, actor_index=self.training.actor_index),
                decisions=window.decisions_at_end,
            )
            print(f"[{self.name}] collection: {window_line(window)}", flush=True)

    def record_checkpoint_periods(self) -> None:
        """Report every checkpoint period that has closed since the last call.

        The period is the interval a numbered checkpoint is written at the end
        of, and its near-greedy mean final wave is what the run judges itself
        on: the same number, on the same key, that decided whether the run went
        on collecting. Keyed by the decisions spent at the crossing, with the
        budget position beside it as a metric, exactly as every other series
        here is.

        Called per episode, on the collecting thread, under the run's progress
        lock. It reads what the run already closed and measures nothing.
        """
        periods = self.training.report.checkpoint_periods
        for period in periods[self.periods_logged :]:
            self.periods_logged += 1
            self.run.log_metrics(
                checkpoint_period_metrics(period), decisions=period.decisions_at_end
            )
            print(f"[{self.name}] {checkpoint_period_line(period)}", flush=True)

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

    def _early_stopping(self) -> dict[str, object]:
        """Whether the run stopped itself, and on what.

        Recorded whether or not early stopping was on: a run that spent its
        whole budget says so with `early_stopped` false and the thresholds it
        was judged under beside it, so two runs are comparable without knowing
        which of them had the flag.
        """
        config = self.training.config
        plateau = self.training.report.plateau
        periods = self.training.report.checkpoint_periods
        return {
            "patience_periods": config.early_stop_patience_periods,
            "min_improvement": config.early_stop_min_improvement,
            "early_stopped": self.training.stopped_early,
            # The period the run stopped at, and the two means the decision was
            # made on: the best the curve had reached, and what the period that
            # closed the run was worth.
            "stopped_at_period": plateau.stopped_at_period,
            "best_period_near_greedy_mean_final_wave": plateau.best_mean_final_wave,
            "closing_period_near_greedy_mean_final_wave": (
                periods[-1].mean_final_wave if periods else None
            ),
            "periods_closed": plateau.periods_closed,
            "periods_without_improvement": plateau.periods_without_improvement,
            # False for a fresh run, and for a resume from a checkpoint that
            # recorded no tracker: that run's plateau count starts over, and
            # saying so is what keeps a fresh baseline from reading as a
            # continued one.
            "tracker_restored_from_parent": plateau.restored,
        }

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
            # The budget position, and how far past the budget the last episode
            # of each actor carried it: the budget is accounted at episode
            # granularity, so two arms equalised on it were equalised to within
            # this much.
            "game_seconds": round(report.game_seconds, 3),
            "budget_overshoot_game_ms": round(
                self.training.budget_overshoot_game_ms, 3
            ),
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
            # The periods the run judged itself on, and what it decided. A run
            # that stopped early spent less than its budget, so a reading of
            # the curve has to be able to see that it stopped and why.
            "checkpoint_periods": [
                asdict(period) for period in report.checkpoint_periods
            ],
            "early_stopping": self._early_stopping(),
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
            # This segment's own throughput. The counters above are the whole
            # run's and come back restored on a resume, but the wall clock is
            # this sitting's alone and is deliberately not restored - dividing
            # one by the other would report the parent's decisions against this
            # segment's hours and read as several times the rate the device
            # collects at.
            "episodes_per_hour": per_hour(
                report.episodes - self.resumed_episodes, report.wall_seconds
            ),
            "decisions_per_hour": per_hour(
                report.decisions - self.resumed_decisions, report.wall_seconds
            ),
            # The comparable throughput: game seconds bought per wall hour is
            # what the device sells, and it does not move with how often the
            # environment happened to ask for a decision.
            "game_seconds_per_hour": per_hour(
                report.game_seconds - self.resumed_game_ms / 1000.0,
                report.wall_seconds,
            ),
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
