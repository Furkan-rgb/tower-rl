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

import json
import shutil
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from tower_rl.environment.decision_time import EMPTY_BREAKDOWN, DecisionTimeBreakdown
from tower_rl.experiment.metrics import (
    DECISION_TIME_INTERVAL_SECONDS,
    LearningCurvePoint,
    actor_summary,
    collected_episode_record,
    collected_episode_records,
    curve_metrics,
    decision_time_line,
    decision_time_metrics,
    episode_metrics,
    fleet_decision_time,
    health_metrics,
    invalid_episode_line,
    learner_metrics,
    per_hour,
    pooled,
    selection_period_line,
    selection_period_metrics,
    window_line,
    window_metrics,
)
from tower_rl.experiment.run_identity import reference_final_waves, scripted_reference
from tower_rl.experiment.tracking import TrackedRun
from tower_rl.learning.backbone import Backbone
from tower_rl.learning.checkpoint import (
    CheckpointIdentity,
    TrainingProgress,
    capture_rng_state,
    write_checkpoint,
)
from tower_rl.learning.evaluator import EvaluationReport, to_record
from tower_rl.learning.learner import IDLE_LOAD, LearnerLoad
from tower_rl.learning.replay import PrioritizedSequenceReplay, ReplayImage
from tower_rl.learning.training import (
    CollectionWindow,
    TrainingProgressReport,
    TrainingRun,
    action_distribution,
    collection_windows,
    episode_health,
)

#: Where in a run's directory its replay buffer is saved, beside
#: `checkpoints/`: `<run_dir>/replay/`. Each save is a dump of its own inside
#: it (`replay_dump_name`), and `latest.pt` names the one it was written with
#: (`Checkpoint.paired_replay`); every other is deleted once that `latest.pt` is
#: in place. A run from before format 5 saved one dump, as it ended, as
#: `replay/` itself.
REPLAY_DIRECTORY = "replay"
#: Where that one end-of-run dump waited while a resumed segment's replaced it.
#: Left behind only by a process of that era killed mid-save; a resume refuses
#: to guess past it. Nothing writes it any more.
REPLAY_BACKUP_DIRECTORY = "replay.previous"


def replay_dump_name(decisions: int) -> str:
    """A replay dump's directory inside `replay/`, from the decisions it was saved at.

    Like a numbered checkpoint's name, it only says which save it is; the
    decisions a resume checks are read out of the dump's metadata.
    """
    return f"d{decisions:07d}"


def non_finite_tensors(state: Any, name: str = "") -> list[str]:
    """The keys of every tensor in a nested state dict holding a NaN or an infinity."""
    if isinstance(state, torch.Tensor):
        is_float = state.is_floating_point() or state.is_complex()
        return [name] if is_float and not bool(torch.isfinite(state).all()) else []
    if isinstance(state, Mapping):
        items: Iterable[tuple[Any, Any]] = state.items()
    elif isinstance(state, (list, tuple)):
        items = enumerate(state)
    else:
        return []
    return [
        broken
        for key, value in items
        for broken in non_finite_tensors(value, f"{name}.{key}" if name else str(key))
    ]


def numbered_checkpoint_name(decisions: int) -> str:
    """A numbered checkpoint's file name, from the decisions behind it.

    The name only says which checkpoint of a run it is; nothing reads the number
    back out of it - a checkpoint's progress is in the file.
    """
    return f"checkpoint-d{decisions:07d}.pt"


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
    #: The learner thread's cumulative load at the last emission, likewise.
    learner_baseline: LearnerLoad = IDLE_LOAD
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
    #: The same running sum in game time, reported beside the decisions axis
    #: each episode's point is keyed on.
    game_ms_logged: float = field(init=False, default=0.0)
    #: How many closed selection periods have been reported. The periods
    #: themselves belong to the run - it is the run that decides whether it is
    #: still improving - and this is only how far the record has followed it.
    periods_logged: int = field(init=False, default=0)
    #: The tracking run this report's checkpoints name as their own, so a resume
    #: from one of them can continue that series. None when untracked.
    tracking_run_id: str | None = None
    #: The saved buffer this run's replay was loaded from on resume, or None
    #: when it started empty. Recorded in every checkpoint's replay provenance.
    replay_restored_from: str | None = None
    #: Resume-point saves whose replay could not be written. While one fails
    #: `latest.pt` does not advance (it is only written after its replay), so a
    #: run that keeps failing has a resume point that falls further behind.
    failed_resume_saves: int = field(init=False, default=0)
    #: The decisions of the resume point in place: the last save that
    #: succeeded, or where this segment resumed (0 for a run that has written
    #: none). What a failed save reports the run to be behind by.
    resume_point_decisions: int = field(init=False, default=0)
    #: Where each collected episode's record is appended as it is reported,
    #: one JSON line each, flushed as it is written: the records survive a
    #: kill that the summary, written only as the run ends, does not. None
    #: writes none.
    episode_stream: Path | None = None

    def __post_init__(self) -> None:
        self.resume_point_decisions = self.resumed_decisions
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
        """The resume point, rewritten as the run proceeds: `latest.pt` and its replay.

        Called with the run's progress lock held (`TrainingRun.checkpoint`), on
        the thread of the actor whose episode triggered it, so nothing is
        counted or learned until it returns. That actor's emulator idles for the
        save, and any other actor that finishes an episode meanwhile waits for
        the lock too; the buffer's own lock is taken only to capture it. The
        conversion of the sequences to arrays (`ReplayImage.write`) is pure
        Python and holds the GIL, so the actors still mid-episode are slowed
        while it runs, not left undisturbed.
        """
        with self.replay.lock:
            image = self.replay.image()
        self._write_resume_point(report, image)

    def save_resume_point(self, *, after_failure: bool = False) -> None:
        """Write the resume point once more as the run ends, with the fleet held still.

        However the run ends short of a hard kill: the actors stop at their
        next lock, so nothing is counted or checkpointed after it.

        `after_failure` is the run ending on an exception or a forced interrupt
        (a second SIGINT), which may have struck mid-update. Then a backbone
        holding a non-finite weight or optimizer moment writes nothing: the
        last periodic resume point is a better one than a broken one written
        over it.
        """
        with self.training.held_still():
            report = self.training.report
            if after_failure:
                broken = non_finite_tensors(self.backbone.state_dict())
                if broken:
                    print(
                        f"[{self.name}] resume point not written: non-finite values "
                        f"in {', '.join(broken[:5])}; the last periodic latest.pt stands",
                        flush=True,
                    )
                    return
            self._write_resume_point(report, self.replay.image())

    def _write_resume_point(self, report: TrainingProgressReport, image: ReplayImage) -> None:
        """Write the buffer, then `latest.pt` naming it, then delete every other dump.

        That order is what lets a kill at any moment leave a pair a resume can
        restore: the new dump is complete (`ReplayImage.write` renames it into
        place) before the `latest.pt` that names it replaces the old one
        (`checkpoint.save`, atomic with its checksum), and the old dump is
        deleted only after that. Until the rename, the old `latest.pt` and the
        dump it names are both still there; after it, the new ones are. Both
        name the one decision count, in the dump's metadata and in the
        checkpoint's progress, and a resume refuses a pair where they differ.

        A failed replay save moves nothing: it is counted and reported loudly,
        not raised, and the previous pair stands - a `latest.pt` without its
        replay would be a resume point that loses the buffer. The cost is that
        while saves keep failing the resume point freezes at the last good
        pair; the count and the line say how far behind it is. Only one dump is
        kept, whatever the number of numbered checkpoints: the resume point is
        `latest.pt` alone.
        """
        replays = self.run_dir / REPLAY_DIRECTORY
        dump = replays / replay_dump_name(report.decisions)
        # A second save at the same count - the run's last, straight after a
        # periodic one - is a dump of its own, not a rewrite of the one the
        # current `latest.pt` names.
        repeat = 0
        while dump.exists():
            repeat += 1
            dump = replays / f"{replay_dump_name(report.decisions)}-{repeat}"
        started = time.monotonic()
        try:
            size = image.write(
                dump, run={"decisions": report.decisions, "identity": asdict(self.identity)}
            )
        except Exception as failure:  # noqa: BLE001 - best effort; see above
            self.failed_resume_saves += 1
            print(
                f"[{self.name}] !!! RESUME POINT NOT SAVED (failure "
                f"{self.failed_resume_saves}): replay not saved ({failure}); "
                f"latest.pt stays at {self.resume_point_decisions} decisions, now "
                f"{report.decisions - self.resume_point_decisions} decisions behind "
                "the run, and will not advance until a save succeeds",
                flush=True,
            )
            self.run.log_metrics(
                {"health_failed_resume_saves": float(self.failed_resume_saves)},
                decisions=report.decisions,
            )
            return
        self.last_checkpoint_fingerprint = self._write(
            report, self.checkpoint_path, paired_replay=dump.relative_to(self.run_dir).as_posix()
        )
        self.resume_point_decisions = report.decisions
        for stale in replays.iterdir():
            if stale == dump:
                continue
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                # The files of a dump from before format 5, saved as `replay/`.
                stale.unlink()
        print(
            f"[{self.name}] resume point at {report.decisions} decisions: latest.pt and "
            f"{len(image.sequences)} replay sequences ({size / 1e9:.2f} GB) in "
            f"{time.monotonic() - started:.1f} s",
            flush=True,
        )

    def numbered_checkpoint(self, report: TrainingProgressReport) -> None:
        """One candidate model of the run, named by the decisions behind it.

        Beside `latest.pt` rather than instead of it: the resume point is
        overwritten as the run proceeds and therefore names no particular model,
        while these are the arms a later evaluation chooses among. The name
        carries the decisions actually spent when it was written - the counter
        lands past its period, not on it, because an episode is played to its
        classified end.
        """
        path = self.run_dir / "checkpoints" / numbered_checkpoint_name(report.decisions)
        digest = self._write(report, path)
        # Under the same tracked run as every metric this report logs, and under
        # the weight digest a later reading names it by: a candidate the run's
        # own page could not reach would have to be found by hand, months later,
        # from a path in a JSON file.
        self.run.log_artifact(path, directory=f"checkpoints/{digest}")
        print(f"[{self.name}] checkpoint {path.name}", flush=True)

    def _write(
        self, report: TrainingProgressReport, path: Path, *, paired_replay: str | None = None
    ) -> str:
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
                # Read with the learner held still, as every save is, so the
                # steps owed and the steps counted are of the same moment.
                learner_debt_steps=self.training.learner_thread.counted_debt_steps(),
            ),
            backbone_state=self.backbone.state_dict(),
            resolved_config=self.resolved,
            replay_provenance={
                **self.replay.snapshot(),
                "restored": self.replay_restored_from is not None,
                "restored_from": self.replay_restored_from,
            },
            tracking_run_id=self.tracking_run_id,
            paired_replay=paired_replay,
            rng_state=capture_rng_state(),
        )

    def record_point(
        self, evaluation: EvaluationReport, *, pre_registered_final: bool = False
    ) -> LearningCurvePoint:
        """Place one evaluation on the curve, against the checkpoint it scored.

        Each point gets its own checkpoint file rather than sharing the resume
        point, which is overwritten as the run proceeds: the strongest model of a
        run is the one a point names, and a fingerprint pointing at a file that
        has since moved on would name nothing. The file is named by the model
        version as well as the decisions: the learner steps on its own thread
        (ADR 0017), so the final evaluation can score weights that a periodic
        point at the same decision count did not.
        """
        progress = self.training.report
        window = self.training.config.collection_window_episodes
        recent = action_distribution(progress.collected[-window:])
        name = f"decisions-{progress.decisions:07d}-v{evaluation.model_version}.pt"
        path = self.run_dir / "checkpoints" / name
        digest = self._write(progress, path)
        spread = evaluation.distribution
        floor = scripted_reference(self.identity.workshop_level)
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
            versus_scripted_reference=(
                None if floor is None else round(spread.mean - floor, 3)
            ),
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

        Each episode's record is also appended to `episode_stream`, and an
        invalid episode is printed with its first reason, both as it is reported
        here: what a run ending abnormally would otherwise take with it.

        Called per episode, on the collecting thread, under the run's progress
        lock. It reads what has already been measured and measures nothing.
        """
        report = self.training.report
        actors = self.training.actor_index
        exploration = self.training.config.exploration
        learner = learner_metrics(report)
        start = self.episodes_logged
        for offset, episode in enumerate(report.collected[start:]):
            index = start + offset
            if self.episode_stream is not None:
                with self.episode_stream.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(collected_episode_record(index, episode), default=str) + "\n"
                    )
            if not episode.summary.valid:
                print(f"[{self.name}] {invalid_episode_line(index, episode)}", flush=True)
            # The decisions at the end of this episode, not the run's current
            # total: the two differ whenever more than one episode has arrived
            # since the last call, and a point on the wrong key is a point on
            # the wrong part of the curve.
            self.decisions_logged += episode.summary.decisions
            self.game_ms_logged += episode.summary.round_ms
            self.episodes_logged += 1
            actor = actors.get(episode.actor_id, -1)
            self.run.log_metrics(
                {
                    **episode_metrics(
                        episode,
                        actor_index=actor,
                        # Game time spent by this episode's end, a statistic;
                        # the step below is decisions, the progress axis.
                        cumulative_game_ms=self.game_ms_logged,
                        # The rate the actor that played this episode was
                        # exploring at, where its episode ended - not the run's
                        # published one, which under a ladder is some other
                        # actor's rung entirely.
                        epsilon=(
                            report.epsilon
                            if actor < 0
                            else exploration.epsilon_for(actor, self.decisions_logged)
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
                window_metrics(
                    window,
                    actor_index=self.training.actor_index,
                    scripted_floor=scripted_reference(self.identity.workshop_level),
                ),
                decisions=window.decisions_at_end,
            )
            print(f"[{self.name}] collection: {window_line(window)}", flush=True)

    def record_selection_periods(self) -> None:
        """Report every selection period that has closed since the last call.

        A numbered checkpoint is written where a period closes, and the period's
        near-greedy mean final wave is what the arm is chosen on and what the
        run judges itself on. Keyed by the decisions spent at the close.

        Called per episode, on the collecting thread, under the run's progress
        lock. It reads what the run already closed and measures nothing.
        """
        periods = self.training.report.selection_periods
        for period in periods[self.periods_logged :]:
            self.periods_logged += 1
            self.run.log_metrics(
                selection_period_metrics(period), decisions=period.decisions_at_end
            )
            print(f"[{self.name}] {selection_period_line(period)}", flush=True)

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
        learner_now = self.training.learner_load()
        learner = learner_now.since(self.learner_baseline)
        self.decision_time_baseline = current
        self.learner_baseline = learner_now
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
                "learner": learner.as_record(),
            }
        )
        self.run.log_metrics(
            decision_time_metrics(fleet, len(interval), learner), decisions=report.decisions
        )
        print(decision_time_line(self.name, fleet, len(interval), learner), flush=True)

    def _early_stopping(self) -> dict[str, object]:
        """Whether the run stopped itself, and on what.

        Recorded whether or not early stopping was on: a run that spent its
        whole budget says so with `early_stopped` false and the thresholds it
        was judged under beside it, so two runs are comparable without knowing
        which of them had the flag.
        """
        config = self.training.config
        plateau = self.training.report.plateau
        periods = self.training.report.selection_periods
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
            # Every kill bar the run reached, and whether it stopped the run
            # there; `early_stopped` above is true for either kind of stop.
            "kill_bar_checks": [
                asdict(check) for check in self.training.report.kill_bar_checks
            ],
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
        self.run.log_metrics(
            {
                **health_metrics(health, prefix="health_"),
                "health_failed_resume_saves": float(self.failed_resume_saves),
            },
            decisions=report.decisions,
        )
        return {
            "backbone": self.name,
            "run_id": self.identity.run_id,
            "resolved_config": self.resolved,
            # The budget position; the game time beside it is a statistic.
            "decisions": report.decisions,
            "game_seconds": round(report.game_seconds, 3),
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
            # The periods the arm is chosen on and the run judged itself on,
            # and what it decided. A run that stopped early spent less than its
            # budget, so a reading of the curve has to see that it stopped and
            # why. The periods are this segment's: a resume does not restore
            # the parent's.
            "selection_periods": [
                asdict(period) for period in report.selection_periods
            ],
            "early_stopping": self._early_stopping(),
            "final_evaluation": (
                asdict(self.final_point) if self.final_point is not None else None
            ),
            # Why there is no final evaluation, when it was skipped on purpose:
            # a run stopped on a kill bar is not evaluated here, nor one the
            # operator stopped - before the evaluation or during it.
            "final_evaluation_skipped": (
                "kill_bar"
                if self.training.killed_by is not None
                else "interrupted"
                if self.training.interrupted
                else None
            ),
            # Stopped from outside (SIGINT) short of its budget: every actor
            # abandoned the episode it was in, and the resume point was written
            # as the run ended, as for any other stop.
            "interrupted": self.training.interrupted,
            "learning_curve": [asdict(point) for point in self.learning_curve],
            "mean_recent_unweighted_absolute_td_error": (
                report.mean_recent_unweighted_absolute_td_error
            ),
            "mean_recent_gradient_norm": report.mean_recent_gradient_norm,
            "mean_recent_value_fit_correlation": report.mean_recent_value_fit_correlation,
            "action_distribution": (
                asdict(distribution) if distribution is not None else None
            ),
            "reference_final_waves": reference_final_waves(self.identity.workshop_level),
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
            # Game seconds per wall hour: what the device sells, a statistic.
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
            "failed_resume_saves": self.failed_resume_saves,
            "checkpoint_path": str(self.checkpoint_path),
            # Every collected episode's own record - what certifies the run
            # stayed honest for its whole span, not only in aggregate.
            "collected_episodes": collected_episode_records(report),
            "evaluations": [to_record(item) for item in report.evaluations],
            "replay": self.replay.snapshot(),
        }
