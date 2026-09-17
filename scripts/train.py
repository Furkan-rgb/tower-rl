#!/usr/bin/env python3
"""Train one or more backbones against the clone under one equal budget.

Private device runner for the instrumented-training profile. Everything it
produces - checkpoints, reports, replay metadata - is written outside the
repository, and every tap it can make is gated inside the adapter on a positive
screen classification.

Naming several backbones interleaves them on the one device in decision blocks,
for the same reason `compare_arms.py` interleaves its episodes: training one arm
to completion and then the next would confound the backbone with whatever
drifted on the host, the device or the account in between.

    TOWER_BRIDGE_BUILD_DIR=... uv run --extra tracking python scripts/train.py \\
        --backbone recurrent-q --backbone stacked-dqn \\
        --budget-decisions 20000

The run records itself to the local MLflow store under `~/.local/state/tower-rl`;
`--extra tracking` is what puts MLflow in the environment. Pass `--no-track` to
run without recording, which leaves nothing to compare the run against later.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from run_episodes import (  # noqa: E402
    add_cadence_arguments,
    cadence_from,
    compatibility,
)

from tower_rl.application.actor import Actor, ActorConfig  # noqa: E402
from tower_rl.application.comparison import interleave_schedule  # noqa: E402
from tower_rl.application.evaluator import EvaluationReport, evaluate, to_record  # noqa: E402
from tower_rl.application.replay import PrioritizedSequenceReplay  # noqa: E402
from tower_rl.application.run_environment import InstrumentedRunEnvironment  # noqa: E402
from tower_rl.application.training import (  # noqa: E402
    CollectionWindow,
    TrainingConfig,
    TrainingProgressReport,
    TrainingRun,
    action_distribution,
    collection_windows,
)
from tower_rl.domain.episode import REWARD_SCHEMA_VERSION  # noqa: E402
from tower_rl.domain.run_actions import ACTION_SCHEMA_VERSION  # noqa: E402
from tower_rl.domain.run_state import OBSERVATION_SCHEMA_VERSION, RunStateBuilder  # noqa: E402
from tower_rl.infrastructure.instrumented_bridge import InstrumentedBridgeClient  # noqa: E402
from tower_rl.infrastructure.instrumented_run_adapter import InstrumentedRunAdapter  # noqa: E402
from tower_rl.learning.backbone import Backbone  # noqa: E402
from tower_rl.learning.checkpoint import (  # noqa: E402
    Checkpoint,
    CheckpointIdentity,
    TrainingProgress,
    fingerprint,
    save,
    write_manifest,
)
from tower_rl.learning.network import NetworkConfig  # noqa: E402
from tower_rl.learning.recurrent_q import RecurrentQBackbone, RecurrentQConfig  # noqa: E402
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig  # noqa: E402
from tower_rl.ports.experiment_tracker import (  # noqa: E402
    ExperimentTracker,
    NoExperimentTracker,
    TrackedRun,
)
from tower_rl.ports.run_port import RunPortError  # noqa: E402

#: Every backbone in the comparison, addressed identically. Adding one here is
#: all it takes to put it under the same protocol as the others.
BACKBONES = ("recurrent-q", "stacked-dqn")

#: The measured floors a learning curve has to be read against, carried in every
#: report so the curve is legible without a second document. Mean final wave over
#: 23 valid episodes per arm, `M1B-E021` at commit 86fcf3c.
REFERENCE_FINAL_WAVES: dict[str, object] = {
    "metric": "mean final wave",
    "episodes_per_arm": 23,
    "source": "M1B-E021 at commit 86fcf3c",
    "scripted": 5.57,
    "random": 5.35,
    "wait": 1.87,
}

#: The floor a learned arm has to clear to mean anything.
SCRIPTED_REFERENCE = 5.57


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


def window_metrics(window: CollectionWindow) -> dict[str, float]:
    """One point of the collection curve, which is what the run is read from."""
    metrics = {
        "collection_mean_final_wave": window.mean_final_wave,
        "collection_versus_scripted_reference": window.mean_final_wave - SCRIPTED_REFERENCE,
        "collection_episodes": float(window.episodes),
        "collection_window_wait_fraction": window.wait_fraction,
        "collection_window_purchases_per_episode": window.purchases_per_episode,
    }
    if window.stdev_final_wave is not None and window.standard_error is not None:
        metrics["collection_stdev_final_wave"] = window.stdev_final_wave
        metrics["collection_standard_error"] = window.standard_error
    return metrics


def window_line(window: CollectionWindow) -> str:
    error = "n/a" if window.standard_error is None else f"{window.standard_error:.2f}"
    return (
        f"window {window.index} decisions {window.decisions_at_end} "
        f"mean final wave {window.mean_final_wave:.2f} se {error} "
        f"over {window.episodes} collected episodes, "
        f"wait {window.wait_fraction:.1%} purchases/episode "
        f"{window.purchases_per_episode:.1f}"
    )


def invalid_episodes_by_reason(report: TrainingProgressReport) -> dict[str, int]:
    """Why the collected episodes that were not scored ended, counted by name.

    A failed episode is an ordinary event that training survives, but survival
    without a record would hide a device that is failing steadily, so the reasons
    are carried in the report beside the waves.
    """
    counts: dict[str, int] = {}
    for summary in report.episode_summaries:
        if not summary.valid:
            counts[summary.termination.value] = counts.get(summary.termination.value, 0) + 1
    return counts


def source_revision() -> str:
    """Bind every artifact to the code that produced it."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


@dataclass
class Arm:
    """One backbone under training, with everything that belongs only to it."""

    name: str
    run_dir: Path
    backbone: Backbone
    replay: PrioritizedSequenceReplay
    training: TrainingRun
    identity: CheckpointIdentity
    resolved: dict[str, object]
    #: The monotonic origin every curve point's wall clock is measured from.
    started: float
    #: Where this arm records itself. `NoExperimentTracker` hands out a handle
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
    #: Runs one exploration-free evaluation and records it on the curve. Held
    #: here as well as on the training run because the pre-registered final
    #: evaluation is taken by the session, after the budget is spent, rather
    #: than on a period.
    evaluation: Callable[[bool], EvaluationReport] | None = None
    #: The point that evaluation produced, which is the headline number.
    final_point: LearningCurvePoint | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoints" / "latest.pt"

    def checkpoint(self, report: TrainingProgressReport) -> None:
        """The resume point, overwritten in place as the run proceeds."""
        self.last_checkpoint_fingerprint = self._write(report, self.checkpoint_path)

    def _write(self, report: TrainingProgressReport, path: Path) -> str:
        """Write one checkpoint atomically and return its weight digest."""
        config = self.training.config
        digest = fingerprint(self.backbone.state_dict())
        save(
            Checkpoint(
                identity=self.identity,
                progress=TrainingProgress(
                    optimisation_steps=report.optimisation_steps,
                    environment_decisions=report.decisions,
                    episodes=report.episodes,
                    epsilon=config.epsilon(report.decisions),
                    importance_beta=config.beta(report.decisions),
                ),
                backbone_state=self.backbone.state_dict(),
                resolved_config=self.resolved,
                replay_provenance={**self.replay.snapshot(), "restored": False},
            ),
            path,
        )
        return digest

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

    def summary(self) -> dict[str, object]:
        report = self.training.report
        distribution = action_distribution(report.collected)
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
            "invalid_episodes_by_reason": invalid_episodes_by_reason(report),
            "failed_episodes": report.failed_episodes,
            "episode_failures": report.episode_failures,
            "evaluation_failures": report.evaluation_failures,
            "checkpoints_written": report.checkpoints_written,
            "checkpoint_path": str(self.checkpoint_path),
            "evaluations": [to_record(item) for item in report.evaluations],
            "replay": self.replay.snapshot(),
        }


def build_backbone(
    name: str, arguments: argparse.Namespace, device: torch.device
) -> tuple[Backbone, RecurrentQConfig | StackedDqnConfig]:
    """The backbone and the learner settings it was fixed with.

    The settings come back alongside it because they are part of what the arm
    is configured by - n-step, discount, learning rate - and a run that does not
    record them cannot be compared with the next one.
    """
    if name == "recurrent-q":
        recurrent = RecurrentQConfig(
            seed=arguments.seed,
            n_step=arguments.n_step,
            discount=arguments.discount,
            learning_rate=arguments.learning_rate,
        )
        return (
            RecurrentQBackbone(
                config=recurrent,
                network_config=NetworkConfig(),
                device=device,
            ),
            recurrent,
        )
    stacked = StackedDqnConfig(
        seed=arguments.seed,
        history_length=arguments.history_length,
        n_step=arguments.n_step,
        discount=arguments.discount,
        learning_rate=arguments.learning_rate,
        # The EMA target belongs to this backbone alone; the recurrent arm
        # copies its target on a period instead.
        target_ema_decay=arguments.target_ema_decay,
    )
    return (
        StackedDqnBackbone(
            config=stacked,
            network_config=NetworkConfig(),
            device=device,
        ),
        stacked,
    )


def build_arm(
    name: str,
    arguments: argparse.Namespace,
    *,
    environment: InstrumentedRunEnvironment,
    device: torch.device,
    profile_id: str,
    parent: Path,
    revision: str,
    started: float,
    tracker: ExperimentTracker,
    tags: dict[str, str],
) -> Arm:
    run_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = parent / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    backbone, learner = build_backbone(name, arguments, device)
    replay = PrioritizedSequenceReplay(
        capacity=arguments.replay_capacity,
        alpha=arguments.priority_alpha,
        seed=arguments.seed,
    )
    config = TrainingConfig(
        budget_decisions=arguments.budget_decisions,
        warmup_sequences=arguments.warmup_sequences,
        batch_size=arguments.batch_size,
        gradient_steps_per_decision=arguments.gradient_steps_per_decision,
        epsilon_start=arguments.epsilon_start,
        epsilon_end=arguments.epsilon_end,
        epsilon_anneal_decisions=arguments.epsilon_anneal_decisions,
        collection_window_episodes=arguments.collection_window_episodes,
        evaluate_every_episodes=arguments.evaluate_every_episodes,
        checkpoint_every_episodes=arguments.checkpoint_every_episodes,
    )
    stride = max(1, arguments.sequence_length // 2)
    burn_in = burn_in_for(name, arguments)
    actor = Actor(
        environment=environment,
        policy=backbone,
        config=ActorConfig(
            actor_id=f"{arguments.serial}:{name}",
            sequence_length=arguments.sequence_length,
            burn_in=burn_in,
            stride=stride,
        ),
        replay=replay,
    )
    cadence = environment.cadence
    resolved: dict[str, object] = {
        "backbone": name,
        "budget_decisions": arguments.budget_decisions,
        "seed": arguments.seed,
        "batch_size": arguments.batch_size,
        "warmup_sequences": config.warmup_sequences,
        "gradient_steps_per_decision": arguments.gradient_steps_per_decision,
        "sequence_length": arguments.sequence_length,
        # The burn-in this arm was built with, which is not the same number for
        # both arms: see `burn_in_for`.
        "burn_in": burn_in,
        "stride": stride,
        "history_length": arguments.history_length if name == "stacked-dqn" else None,
        "n_step": learner.n_step,
        "discount": learner.discount,
        "learning_rate": learner.learning_rate,
        "target_ema_decay": (
            arguments.target_ema_decay if name == "stacked-dqn" else None
        ),
        "epsilon_start": config.epsilon_start,
        "epsilon_end": config.epsilon_end,
        "epsilon_anneal_decisions": config.epsilon_anneal_decisions,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "priority_alpha": arguments.priority_alpha,
        "replay_capacity": arguments.replay_capacity,
        "collection_window_episodes": config.collection_window_episodes,
        "evaluate_every_episodes": arguments.evaluate_every_episodes,
        "evaluation_episodes": arguments.evaluation_episodes,
        "checkpoint_every_episodes": arguments.checkpoint_every_episodes,
        # The cadence the environment was actually built with, not what was
        # asked for on the command line.
        "frame_game_ms": cadence.frame_game_ms,
        "max_quiet_game_ms": cadence.max_quiet_game_ms,
        "health_change_fraction": cadence.health_change_fraction,
        "block_decisions": arguments.block_decisions,
        "device": str(device),
    }
    # The floors the curve is read against travel with the run, so a comparison
    # opened months later is self-contained.
    params: dict[str, object] = dict(resolved)
    params.update(
        {f"reference_{key}": value for key, value in REFERENCE_FINAL_WAVES.items()}
    )
    run = tracker.start_run(
        name=run_id, params=params, tags={**tags, "backbone": name, "run_id": run_id}
    )
    print(f"[{name}] tracking run {run.run_id}", flush=True)
    arm = Arm(
        name=name,
        run_dir=run_dir,
        backbone=backbone,
        replay=replay,
        training=TrainingRun(
            actor=actor,
            replay=replay,
            backbone=backbone,
            config=config,
        ),
        started=started,
        run=run,
        identity=CheckpointIdentity(
            run_id=run_id,
            backbone=name,
            profile_id=profile_id,
            observation_schema=OBSERVATION_SCHEMA_VERSION,
            action_schema=ACTION_SCHEMA_VERSION,
            reward_schema=REWARD_SCHEMA_VERSION,
            source_revision=revision,
        ),
        resolved=resolved,
    )

    def run_evaluation(pre_registered_final: bool = False) -> EvaluationReport:
        # Exploration-free, never written to replay; the evaluator enforces both.
        report = evaluate(
            environment,
            backbone,
            episodes=arguments.evaluation_episodes,
            profile_id=profile_id,
            model_version=backbone.model_version,
        )
        point = arm.record_point(report, pre_registered_final=pre_registered_final)
        if pre_registered_final:
            arm.final_point = point
        print(f"[{name}] {report.summary_line()}", flush=True)
        print(f"[{name}] curve: {point.line()}", flush=True)
        return report

    def on_episode(report: TrainingProgressReport) -> None:
        arm.record_collection_windows()
        print(
            f"[{name}] episode {report.episodes} decisions {report.decisions}/"
            f"{config.budget_decisions} steps {report.optimisation_steps}",
            flush=True,
        )

    arm.evaluation = run_evaluation
    # The periodic hook takes no argument and is off by default: mid-run
    # evaluation buys points too noisy to read at the price of device time.
    arm.training.evaluate = run_evaluation
    arm.training.checkpoint = arm.checkpoint
    arm.training.on_episode = on_episode
    manifest = run_dir / "manifest.json"
    write_manifest(manifest, {"run_id": run_id, **resolved})
    run.log_artifact(manifest)
    return arm


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Everything the run is configured by, validated before a device is touched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        action="append",
        choices=BACKBONES,
        help="repeat to interleave several arms under one equal budget",
    )
    parser.add_argument("--budget-decisions", type=int, default=20_000, help="per arm")
    parser.add_argument(
        "--block-decisions",
        type=int,
        default=2_000,
        help="decisions before handing the device to the next arm; lands on an episode",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-steps-per-decision",
        type=float,
        default=0.25,
        help="the replay ratio; 0.25 is about 126 transitions replayed per generated",
    )
    parser.add_argument("--warmup-sequences", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument(
        "--burn-in",
        type=int,
        default=40,
        help="recurrent-q only: how much history warms the LSTM state before the unroll",
    )
    parser.add_argument(
        "--stacked-burn-in",
        type=int,
        default=7,
        help=(
            "stacked-dqn only: exactly history-length - 1, which is what fills the "
            "window; anything longer discards learnable steps for nothing"
        ),
    )
    # Read by stacked-dqn only; the recurrent backbone carries time in its state.
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--target-ema-decay",
        type=float,
        default=0.995,
        help="stacked-dqn only; the recurrent arm copies its target on a period",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-anneal-decisions",
        type=int,
        default=10_000,
        help="decisions to anneal exploration over; held at the end value afterwards",
    )
    parser.add_argument(
        "--priority-alpha",
        type=float,
        default=0.0,
        help=(
            "prioritized replay exponent; 0 samples uniformly and makes every "
            "importance-sampling weight exactly one, so the loss is readable"
        ),
    )
    parser.add_argument(
        "--collection-window-episodes",
        type=int,
        default=100,
        help="episodes per point of the collection curve the run is read from",
    )
    # Mid-run exploration-free evaluation is off: the curve is read from the
    # collection episodes, and 5-episode points cost device time for a standard
    # error no improvement worth having could clear. A positive period turns the
    # periodic hook back on deliberately.
    parser.add_argument("--evaluate-every-episodes", type=int, default=0)
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=30,
        help="the pre-registered exploration-free evaluation of the final checkpoint",
    )
    parser.add_argument("--checkpoint-every-episodes", type=int, default=25)
    parser.add_argument("--serial", default="emulator-5556")
    parser.add_argument("--port", type=int, default=47652)
    add_cadence_arguments(parser)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path.home() / ".local/state/tower-rl/runs",
        help="outside the repository; checkpoints and reports are never committed",
    )
    parser.add_argument(
        "--experiment",
        default="tower-rl-training",
        help="the MLflow experiment runs of this session land in",
    )
    # Tracking is on by default and a missing MLflow is an error rather than a
    # silent fallback: an untracked run is exactly the outcome this exists to
    # prevent, and device time is too expensive to spend on a run that leaves
    # nothing comparable behind. `--no-track` is the deliberate way out.
    parser.add_argument(
        "--no-track",
        dest="track",
        action="store_false",
        help="run without recording to MLflow; the run leaves no tracked history",
    )
    arguments = parser.parse_args(argv)

    arguments.backbone = list(dict.fromkeys(arguments.backbone or ["recurrent-q"]))
    if arguments.serial == "emulator-5554":
        raise SystemExit("refusing to train against the canonical evaluation AVD")
    if (
        "stacked-dqn" in arguments.backbone
        and arguments.stacked_burn_in < arguments.history_length - 1
    ):
        # Checked here rather than at the first optimisation step, which is an
        # hour of collection later.
        raise SystemExit(
            f"burn-in {arguments.stacked_burn_in} cannot fill a window of "
            f"{arguments.history_length}"
        )
    for name in arguments.backbone:
        if burn_in_for(name, arguments) >= arguments.sequence_length:
            raise SystemExit(
                f"burn-in {burn_in_for(name, arguments)} leaves {name} no learning steps "
                f"in a sequence of {arguments.sequence_length}"
            )
    return arguments


def burn_in_for(name: str, arguments: argparse.Namespace) -> int:
    """How much of a sequence this backbone burns in before it learns.

    Per arm, because the number means two different things. The recurrent arm
    burns in to reconstruct an LSTM state that was produced by older parameters,
    and needs enough steps to do it. The stacked arm has no state to
    reconstruct: its burn-in only fills the history window, so
    `history_length - 1` steps fill it exactly and every further step is a
    learnable step thrown away. A single global flag would therefore have to be
    wrong for one of the two arms.
    """
    return int(arguments.stacked_burn_in if name == "stacked-dqn" else arguments.burn_in)


def tracking_uri(arguments: argparse.Namespace) -> str:
    """Where runs are recorded: beside the run state, never in the repository.

    SQLite rather than a directory of files because MLflow 3 refuses the
    filesystem backend, and local either way: nothing leaves this machine.
    """
    override = os.environ.get("MLFLOW_TRACKING_URI")
    if override:
        return override
    return f"sqlite:///{arguments.run_dir.parent / 'mlflow.db'}"


def artifact_root(arguments: argparse.Namespace) -> str:
    """Where tracked files land, beside the store and outside the repository."""
    return str(arguments.run_dir.parent / "mlartifacts")


def build_tracker(arguments: argparse.Namespace) -> ExperimentTracker:
    """The tracker the session records itself through.

    The MLflow adapter is imported here and nowhere else, so a machine without
    MLflow installed can still run everything that does not ask to be tracked.
    """
    if not arguments.track:
        return NoExperimentTracker()
    try:
        from tower_rl.infrastructure.mlflow_tracker import MlflowExperimentTracker
    except ImportError as missing:
        raise SystemExit(
            f"tracking is on but MLflow is not installed ({missing}). "
            "Install it with `uv sync --extra tracking`, or pass --no-track "
            "to run untracked."
        ) from missing
    arguments.run_dir.parent.mkdir(parents=True, exist_ok=True)
    return MlflowExperimentTracker(
        tracking_uri=tracking_uri(arguments),
        experiment=arguments.experiment,
        artifact_root=artifact_root(arguments),
    )


def train_session(
    arguments: argparse.Namespace,
    environment: InstrumentedRunEnvironment,
    *,
    profile_id: str,
    revision: str,
    device: torch.device,
    tracker: ExperimentTracker | None = None,
    bridge_version: str | None = None,
) -> dict[str, object]:
    """Train every named arm to its budget and return the session report.

    The environment is a parameter rather than something built here, so the one
    place a bridge to the real device is opened is `main`. Nothing else decides
    what the arms are talking to.
    """
    names = list(arguments.backbone)
    started = time.monotonic()
    session = arguments.run_dir / f"session-{time.strftime('%Y%m%d-%H%M%S')}"
    recorder = NoExperimentTracker() if tracker is None else tracker
    tags = {
        "source_revision": revision,
        "profile_id": profile_id,
        "session": session.name,
        "device_serial": arguments.serial,
        # One actor per arm: the arms take turns on the one instance rather
        # than collecting in parallel.
        "actors": "1",
    }
    if bridge_version is not None:
        tags["bridge_version"] = bridge_version
    arms = {
        name: build_arm(
            name,
            arguments,
            environment=environment,
            device=device,
            profile_id=profile_id,
            parent=session,
            revision=revision,
            started=started,
            tracker=recorder,
            tags=tags,
        )
        for name in names
    }

    blocks_per_arm = -(-arguments.budget_decisions // arguments.block_decisions)
    schedule = interleave_schedule(tuple(names), blocks_per_arm, block=1, seed=arguments.seed)
    try:
        for name in schedule:
            arm = arms[name]
            if arm.training.finished:
                continue
            arm.training.advance(arguments.block_decisions)

        for arm in arms.values():
            arm.checkpoint(arm.training.report)
            # The one pre-registered measurement of the run: exploration-free,
            # on the final weights, sized so its standard error can resolve a
            # real difference against the scripted floor. Taken after the budget
            # is spent, so it costs no decisions and cannot be chosen after the
            # fact from a series of mid-run points.
            if arm.evaluation is None:
                continue
            try:
                arm.evaluation(True)
            except (RunPortError, ValueError) as failure:
                # Losing the headline measurement must not lose the run: the
                # collection curve and the checkpoints are already on disk.
                arm.training.report.evaluation_failures.append(str(failure))
                print(f"[{arm.name}] final evaluation failed: {failure}", flush=True)

        summaries = [arm.summary() for arm in arms.values()]
        report: dict[str, object] = {
            "session": str(session),
            "profile_id": profile_id,
            "source_revision": revision,
            "budget_decisions_per_arm": arguments.budget_decisions,
            "block_decisions": arguments.block_decisions,
            "wall_seconds": round(time.monotonic() - started, 1),
            # Repeated at the top of the report as well as inside each arm: the
            # curve is meaningless without the floors it is read against.
            "reference_final_waves": REFERENCE_FINAL_WAVES,
            "arms": summaries,
        }
        session.mkdir(parents=True, exist_ok=True)
        (session / "summary.json").write_text(json.dumps(report, indent=2, default=str))
        for arm, summary in zip(arms.values(), summaries, strict=True):
            # The arm's summary holds its learning curve and the per-episode
            # evaluation records, so it is what a tracked run is read from.
            path = arm.run_dir / "summary.json"
            path.write_text(json.dumps(summary, indent=2, default=str))
            arm.run.log_artifact(path)
        return report
    finally:
        # A run that ended badly is still a run that has to be closed, or it
        # would sit open in the store forever.
        for arm in arms.values():
            arm.run.finish()


def main() -> int:
    arguments = parse_arguments()

    # Built before the device is touched: a session that cannot be recorded
    # should fail now rather than an hour into collection.
    tracker = build_tracker(arguments)
    print(f"tracking: {tracker.tracking_uri} experiment {arguments.experiment}", flush=True)
    if arguments.track:
        print(
            "open the UI with: uv run --extra tracking mlflow ui "
            f"--backend-store-uri {tracker.tracking_uri}",
            flush=True,
        )
        print(f"artifacts: {artifact_root(arguments)}", flush=True)

    build_dir = Path(
        os.environ.get("TOWER_BRIDGE_BUILD_DIR")
        or Path("/tmp/tower-bridge-live.latest").read_text().strip()
    )
    expected = compatibility(build_dir)
    client = InstrumentedBridgeClient(
        "127.0.0.1",
        arguments.port,
        expected_compatibility=expected,
        connect_timeout=5.0,
        read_timeout=120.0,
        heartbeat_timeout=60.0,
    )
    client.connect()
    adapter = InstrumentedRunAdapter(client=client)
    environment = InstrumentedRunEnvironment(
        port=adapter,
        builder=RunStateBuilder(profile_id=expected.profile_id),
        cadence=cadence_from(arguments),
    )

    try:
        report = train_session(
            arguments,
            environment,
            profile_id=expected.profile_id,
            revision=source_revision(),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            tracker=tracker,
            bridge_version=expected.bridge_version,
        )
    finally:
        adapter.release()
        client.close()

    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
