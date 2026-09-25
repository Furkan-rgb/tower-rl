#!/usr/bin/env python3
"""Train the stacked-DQN backbone against the clone under one fixed budget.

Private device runner for the instrumented-training profile. Everything it
produces - checkpoints, reports, replay metadata - is written outside the
repository. Nothing in this path reads a pixel or touches the screen: the round
boundary lives in the bridge, and an action is a semantic upgrade purchase the
bridge performs in the game, so there is no screen classification and no tap for
one to gate.

Progress has one unit, decisions (`learning/training.py`). The budget is
cumulative decisions across the fleet (`--budget-decisions`), independent of the
game's speed; the run stops after the episode that crosses it, so it overshoots
by at most one episode per actor. Game time and wall time are reported beside
it as statistics.

    uv run --extra tracking python scripts/train.py \\
        --budget-decisions 60000

`--actors N` collects on N emulator instances at once, one actor thread each,
into the one replay buffer and the one learner, so the budget is spent about N
times faster: collection measured linearly to four instances on this host,
8,850 decisions an hour at N=1 and 27,781 at N=4 with fidelity intact at every N
(M1B-E028). A single actor is the default and keeps addressing `--serial` and
`--port`, which is the instance the operator brought up by hand. A fleet owns
its instances instead: it takes them from `CloneInstance` by index exactly as
`run_actors.py` does, brings each one up only after the previous one is ready -
four simultaneous cold boots is the one thing the fleet measurement broke on -
and tears them all down when the run ends.

    uv run --extra tracking python scripts/train.py \\
        --actors 4 --budget-decisions 60000

The decision axis is cut into selection periods (`--selection-period-decisions`,
default 15,000). Where a period closes the run writes a numbered checkpoint,
`checkpoint-d<decisions>.pt`, and takes the mean final wave of the near-greedy
actors' valid episodes that ended inside it; the summary lists them as
`selection_periods`, and the arm is chosen from them by hand by the rule in
`docs/solution.md` 9.2b. `--checkpoint-every-decisions` writes extra numbered
checkpoints between them, for a learning curve; they are never the arm.

`--early-stop-patience-periods N` lets the run stop before its budget is spent:
a curve that has not improved on its best period by
`--early-stop-min-improvement` waves for N selection periods in a row has
stopped learning, so the run ends after writing that period's checkpoint and
the summary records what it stopped on. The default, 0, spends the whole
budget.

`--kill-bar AT:START:MIN` (repeatable) pre-registers a floor on the decision
axis: when the fleet's cumulative decisions first reach AT, the near-greedy
actors' valid episodes that ended in (START, AT] must average at least MIN
waves, or the run stops there and the summary records which bar stopped it. A
window with no such episode measures nothing and does not stop the run.

`--resume <checkpoint.pt>` continues a run that has already spent part of its
budget. The weights, the optimizer moments, the decision and game-time counters
and every schedule and cadence derived from them come back from the file; the
replay buffer does not, so the run re-warms it under the loaded policy before
learning restarts. `--budget-decisions` stays the whole run's total.

    uv run --extra tracking python scripts/train.py \\
        --resume state/runs/<session>/<run>/checkpoints/latest.pt \\
        --budget-decisions 120000

The run records itself to the local MLflow store under `state/`;
`--extra tracking` is what puts MLflow in the environment. Pass `--no-track` to
run without recording, which leaves nothing to compare the run against later.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from run_episodes import (  # noqa: E402
    add_cadence_arguments,
    add_upgrade_availability_argument,
    decision_cadence_from,
    open_environment,
    upgrade_availability_from,
)

from tower_rl.console_timestamp import timestamped_print as print  # noqa: E402
from tower_rl.environment.project_state import state_directory  # noqa: E402
from tower_rl.environment.run_environment import InstrumentedRunEnvironment  # noqa: E402
from tower_rl.environment.run_port import RunPortError  # noqa: E402
from tower_rl.experiment.run_identity import (  # noqa: E402
    REFERENCE_FINAL_WAVES,
    RunIdentity,
    checkpoint_identity,
    dreamer_resolved_config,
    resolved_config,
    source_revision,
    tracked_params,
)
from tower_rl.experiment.tracking import (  # noqa: E402
    ExperimentTracker,
    NoExperimentTracker,
    artifact_root,
    tracking_uri,
)
from tower_rl.experiment.training_report import TrainingReport  # noqa: E402
from tower_rl.learning.actor import Actor, ActorConfig  # noqa: E402
from tower_rl.learning.backbone import Backbone  # noqa: E402
from tower_rl.learning.checkpoint import (  # noqa: E402
    DECISION_BUDGET_FORMAT_VERSION,
    CheckpointError,
    ResumeState,
    resume_state,
    write_manifest,
)
from tower_rl.learning.dreamer import DREAMERV3, DreamerBackbone, DreamerConfig  # noqa: E402
from tower_rl.learning.evaluator import EvaluationReport, evaluate  # noqa: E402
from tower_rl.learning.exploration import (  # noqa: E402
    EXPLORATION_OPTIONS,
    LADDER,
    ExplorationSchedule,
)
from tower_rl.learning.network import NetworkConfig  # noqa: E402
from tower_rl.learning.replay import PrioritizedSequenceReplay  # noqa: E402
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig  # noqa: E402
from tower_rl.learning.training import (  # noqa: E402
    ActorProgress,
    KillBar,
    NearGreedyPlateau,
    TrainingConfig,
    TrainingProgressReport,
    TrainingRun,
)
from tower_rl.simulation.bridge import (  # noqa: E402
    bridge_build_directory,
    compatibility,
    deploy_bridge,
)
from tower_rl.simulation.bring_up import (  # noqa: E402
    bring_up,
    require_game_activity,
    require_offline,
)
from tower_rl.simulation.fleet import (  # noqa: E402
    bring_up_fleet,
    prepare_pinned_snapshot,
    tear_down_fleet,
)
from tower_rl.simulation.frame_rate import raise_frame_rate  # noqa: E402
from tower_rl.simulation.instance import (  # noqa: E402
    GUEST_FRAME_RATE_HZ,
    MAX_GUEST_FRAME_RATE_HZ,
    CloneInstance,
)
from tower_rl.simulation.instrumented_bridge import (  # noqa: E402
    BridgeCompatibility,
    InstrumentedBridgeClient,
)
from tower_rl.simulation.instrumented_run_adapter import InstrumentedRunAdapter  # noqa: E402

#: The default backbone, and the one every run before DreamerV3 trained.
BACKBONE = "stacked-dqn"
#: What `--backbone` chooses from.
BACKBONES = (BACKBONE, DREAMERV3)

#: What a uniform schedule anneals to when `--epsilon-end` is not given. Held
#: here rather than as the flag's default so that a value the ladder would
#: ignore can be told from one that was never given at all.
DEFAULT_EPSILON_END = 0.05

#: The per-decision discount when neither `--discount` nor
#: `--discount-per-game-second` is given, held here for the same reason.
DEFAULT_DISCOUNT = 0.99


@dataclass(frozen=True)
class ActorInstance:
    """One emulator instance an actor collects on, and what it is called.

    Identity and environment travel together because the report is per actor:
    an aggregate that cannot name the instance a failure came from cannot say
    which emulator to look at.
    """

    serial: str
    environment: InstrumentedRunEnvironment


def kill_bar(text: str) -> KillBar:
    """One `--kill-bar AT:START:MIN`, as the run's config holds it."""
    try:
        at, start, minimum = text.split(":")
        return KillBar(
            at_decisions=int(at),
            window_start_decisions=int(start),
            min_mean_final_wave=float(minimum),
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not AT:START:MIN with 0 <= START < AT ({error})"
        ) from error


def build_backbone(
    arguments: argparse.Namespace, device: torch.device
) -> tuple[Backbone, StackedDqnConfig, NetworkConfig]:
    """The backbone and the two settings objects it was fixed with.

    The settings come back alongside it because they are part of what the arm
    is configured by - n-step, discount, learning rate, the width of the network
    - and a run that does not record them cannot be compared with the next one,
    nor can one of its checkpoints be rebuilt into the policy that wrote it.
    """
    stacked = StackedDqnConfig(
        seed=arguments.seed,
        history_length=arguments.history_length,
        n_step=arguments.n_step,
        n_step_final=arguments.n_step_final,
        n_step_anneal_steps=arguments.n_step_anneal_steps,
        discount=arguments.discount,
        discount_per_game_second=arguments.discount_per_game_second,
        survival_time_reward=arguments.survival_time_reward,
        learning_rate=arguments.learning_rate,
        target_ema_decay=arguments.target_ema_decay,
    )
    network = NetworkConfig()
    return (
        StackedDqnBackbone(
            config=stacked,
            network_config=network,
            device=device,
        ),
        stacked,
        network,
    )


def build_arm(
    name: str,
    arguments: argparse.Namespace,
    *,
    instances: Sequence[ActorInstance],
    device: torch.device,
    profile_id: str,
    parent: Path,
    revision: str,
    started: float,
    tracker: ExperimentTracker,
    tags: dict[str, str],
    resume: ResumeState | None = None,
) -> tuple[TrainingReport, Callable[[bool], EvaluationReport]]:
    # Identity first: the run id every artefact is filed under, and the
    # compatibility key its checkpoints are written with, derived from it in the
    # one place that knows which schemas this code is. A resumed segment gets a
    # run id and a directory of its own - it must not overwrite the resume point
    # it was started from - and says which checkpoint it continues instead.
    # The cadence is the run's, not an instance's: one argument fixes it for
    # every actor, for the identity its checkpoints are keyed on and for the
    # snapshot it records. Reading it back off an instance would let a fleet
    # whose instances somehow disagreed name one of them and say nothing.
    decision_cadence = decision_cadence_from(arguments)
    # The availability is the run's in exactly the same way: one argument fixes
    # what every actor may buy, for the identity its checkpoints are keyed on
    # and for the snapshot it records (ADR 0011).
    upgrade_availability = upgrade_availability_from(arguments)
    identity = RunIdentity.started_now(
        name,
        profile_id=profile_id,
        source_revision=revision,
        decision_cadence=decision_cadence,
        upgrade_availability=upgrade_availability,
    )
    run_id = identity.run_id
    run_dir = parent / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if arguments.backbone == DREAMERV3:
        backbone: Backbone = DreamerBackbone(
            config=DreamerConfig(seed=arguments.seed), device=device
        )
        # Only for `resolved_config`'s signature: `dreamer_resolved_config`
        # records every stacked-dqn setting of a DreamerV3 run as None.
        learner, network = StackedDqnConfig(), NetworkConfig()
    else:
        backbone, learner, network = build_backbone(arguments, device)
    if resume is not None:
        # The weights, the target network and the optimizer moments together:
        # they are one `state_dict`, and a resume that took only the weights
        # would restart Adam's moments silently mid-run.
        backbone.load_state_dict(dict(resume.backbone_state))
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
        exploration=ExplorationSchedule.for_option(
            arguments.exploration,
            actors=len(instances),
            epsilon_start=arguments.epsilon_start,
            epsilon_end=arguments.epsilon_end,
            anneal_decisions=arguments.epsilon_anneal_decisions,
        ),
        collection_window_episodes=arguments.collection_window_episodes,
        evaluate_every_episodes=arguments.evaluate_every_episodes,
        checkpoint_every_episodes=arguments.checkpoint_every_episodes,
        checkpoint_every_decisions=arguments.checkpoint_every_decisions,
        selection_period_decisions=arguments.selection_period_decisions,
        early_stop_patience_periods=arguments.early_stop_patience_periods,
        early_stop_min_improvement=arguments.early_stop_min_improvement,
        parameter_sync_episodes=arguments.parameter_sync_episodes,
        kill_bars=tuple(arguments.kill_bars),
    )
    stride = max(1, arguments.sequence_length // 2)
    burn_in = int(arguments.stacked_burn_in)
    # One actor per instance, all of them writing into the one buffer above and
    # all of them learned from by the one backbone. `TrainingRun` gives each its
    # own copy of that backbone to act from and refreshes it on the configured
    # cadence; the network passed here is only what those copies are made of.
    actors = [
        Actor(
            environment=instance.environment,
            policy=backbone,
            config=ActorConfig(
                actor_id=f"{instance.serial}:{name}",
                sequence_length=arguments.sequence_length,
                burn_in=burn_in,
                stride=stride,
            ),
            replay=replay,
        )
        for instance in instances
    ]
    resolved = resolved_config(
        name,
        arguments,
        actor_ids=[actor.config.actor_id for actor in actors],
        config=config,
        learner=learner,
        network=network,
        cadence=instances[0].environment.cadence,
        decision_cadence=decision_cadence,
        upgrade_availability=upgrade_availability,
        burn_in=burn_in,
        stride=stride,
        device=device,
        parent_checkpoint=None if resume is None else resume.parent_checkpoint,
    )
    if isinstance(backbone, DreamerBackbone):
        resolved = dreamer_resolved_config(resolved, backbone.config)
    if resume is not None and resume.tracking_run_id is not None:
        # The same run, not a second one beside it: the curve of a run trained
        # in two sittings is one series, on the one decision axis both
        # segments' points are keyed by. Its params were fixed when the first segment
        # started and are not restated here.
        run = tracker.open_run(resume.tracking_run_id)
    else:
        run = tracker.start_run(
            name=run_id,
            params=tracked_params(resolved),
            tags={**tags, "backbone": name, "run_id": run_id},
        )
    print(f"[{name}] tracking run {run.run_id}", flush=True)
    # What a checkpoint of this run names as the series it belongs to. An
    # untracked run has no such series, and must not write the untracked
    # handle's placeholder into a file a later resume would try to attach to.
    tracked_run_id = None if isinstance(tracker, NoExperimentTracker) else run.run_id
    progress = TrainingProgressReport()
    if resume is not None:
        # The counters the budget, the schedules and the cadences are all read
        # from. Epsilon and beta are not restored beside them: `TrainingRun`
        # derives both from this counter, so they land where a run that had
        # never stopped would have them.
        progress = TrainingProgressReport(
            decisions=resume.decisions,
            game_ms=resume.game_ms,
            episodes=resume.episodes,
            optimisation_steps=resume.optimisation_steps,
            # The early-stopping tracker, so a run trained in two sittings is
            # judged on one near-greedy curve. A checkpoint written before
            # early stopping existed records none, and the tracker then starts
            # fresh - said out loud below, because a fresh baseline that read
            # as a continued one would let a plateaued run go on collecting.
            plateau=NearGreedyPlateau(
                periods_closed=resume.periods_closed or 0,
                best_mean_final_wave=resume.best_period_near_greedy_mean,
                periods_without_improvement=resume.periods_without_improvement,
                restored=resume.periods_closed is not None,
            ),
        )
    arm = TrainingReport(
        name=name,
        run_dir=run_dir,
        backbone=backbone,
        replay=replay,
        training=TrainingRun(
            actors=actors,
            replay=replay,
            backbone=backbone,
            config=config,
            report=progress,
        ),
        started=started,
        run=run,
        identity=checkpoint_identity(identity),
        resolved=resolved,
        # What the parent had spent, which is where this segment's series start
        # on the run's budget and what its own throughput is measured net of.
        resumed_decisions=progress.decisions,
        resumed_episodes=progress.episodes,
        resumed_game_ms=progress.game_ms,
        tracking_run_id=tracked_run_id,
    )

    def run_evaluation(pre_registered_final: bool = False) -> EvaluationReport:
        # Exploration-free, never written to replay; the evaluator enforces both.
        # It borrows the first instance, so it may only run while the fleet is
        # not collecting: the one pre-registered evaluation is taken after the
        # budget is spent, and mid-run evaluation is refused for a fleet.
        report = evaluate(
            instances[0].environment,
            backbone,
            episodes=arguments.evaluation_episodes,
            profile_id=profile_id,
            model_version=backbone.model_version,
        )
        point = arm.record_point(report, pre_registered_final=pre_registered_final)
        print(f"[{name}] {report.summary_line()}", flush=True)
        print(f"[{name}] curve: {point.line()}", flush=True)
        return report

    def on_withdrawal(progress: ActorProgress) -> None:
        # One instance stopping is invisible in the aggregate - the fleet simply
        # collects slower - so the actor and what killed it are named as it
        # happens, for an operator reading a log hours later.
        print(
            f"[{name}] actor {progress.actor_id} withdrawn after "
            f"{progress.consecutive_failures} failed episodes: {progress.withdrawn}",
            flush=True,
        )

    # Learning is refused until the buffer holds `--warmup-sequences` again:
    # replay is not persisted, so a resumed run re-warms it under the loaded
    # policy at the epsilon its schedule has reached. That is the ordinary
    # warm-up rule, not a mode, but where its boundary fell is the one thing a
    # reading of the resumed segment cannot recover afterwards, so the first
    # optimisation step past the resume point is reported when it happens.
    warmed = resume is None

    def on_episode(report: TrainingProgressReport) -> None:
        nonlocal warmed
        if (
            not warmed
            and resume is not None
            and report.optimisation_steps > resume.optimisation_steps
        ):
            warmed = True
            arm.run.log_metrics(
                {"warmup_finished_decisions": float(report.decisions)},
                decisions=report.decisions,
            )
            print(
                f"[{name}] replay re-warmed; learning restarted at "
                f"{report.decisions} decisions",
                flush=True,
            )
        # The episode first: it is the tracked unit, and the window below it is
        # the smoothed view of the same series.
        arm.record_episodes()
        arm.record_collection_windows()
        # The periods the run judges itself on, after the windows: a crossing
        # is closed inside the same hook, and this is the record of it.
        arm.record_selection_periods()
        arm.record_decision_time()
        print(
            f"[{name}] episode {report.episodes} decisions "
            f"{report.decisions}/{config.budget_decisions} game seconds "
            f"{report.game_seconds:.0f} steps {report.optimisation_steps}",
            flush=True,
        )

    # The periodic hook takes no argument and is off by default: mid-run
    # evaluation buys points too noisy to read at the price of device time.
    arm.training.evaluate = run_evaluation
    arm.training.checkpoint = arm.checkpoint
    # The candidates a post-hoc selection chooses among, beside the resume point.
    arm.training.numbered_checkpoint = arm.numbered_checkpoint
    arm.training.on_episode = on_episode
    arm.training.on_withdrawal = on_withdrawal
    manifest = run_dir / "manifest.json"
    write_manifest(manifest, {"run_id": run_id, **resolved})
    run.log_artifact(manifest)
    if resume is not None:
        # Where this segment picked the budget up, on the same axis every other
        # point of the run is keyed by. The parent itself is named in
        # `resolved_config`, which travels in the manifest above, in every
        # checkpoint this segment writes, and in its summary.
        run.log_metrics(
            {
                "resumed_from_decisions": float(resume.decisions),
                "resumed_from_game_seconds": resume.game_ms / 1000.0,
            },
            decisions=resume.decisions,
        )
        print(
            f"[{name}] resuming {resume.parent_checkpoint} at "
            f"{resume.decisions} decisions of {config.budget_decisions}",
            flush=True,
        )
        if config.early_stop_patience_periods and not progress.plateau.restored:
            print(
                f"[{name}] the parent checkpoint records no early-stopping "
                "tracker, so the plateau count starts fresh from this segment's "
                "first selection period",
                flush=True,
            )
    # The evaluation comes back beside the report rather than on it: the report
    # records what an evaluation produced, and the session decides when the one
    # pre-registered evaluation is taken.
    return arm, run_evaluation


#: Replay sequences before DreamerV3's first update. The official loop trains
#: once replay holds one batch of steps (16 x 64 = 1,024; `embodied/run/train.py`);
#: at a stride of 32, less the window each episode's edge costs, that is about 25.
DREAMER_WARMUP_SEQUENCES = 25

#: Flags only stacked-dqn reads. Given with `--backbone dreamerv3` they would be
#: silently unused, so they are refused.
STACKED_ONLY_FLAGS = (
    "history_length",
    "n_step",
    "n_step_final",
    "n_step_anneal_steps",
    "discount",
    "discount_per_game_second",
    "survival_time_reward",
    "learning_rate",
    "target_ema_decay",
    # An epsilon anneal: DreamerV3 adds no exploration noise to anneal.
    "epsilon_anneal_decisions",
)


def dreamer_loop_settings() -> dict[str, object]:
    """The training-loop settings DreamerV3 fixes, by argument name (solution.md 9.4c)."""
    config = DreamerConfig()
    return {
        "sequence_length": config.batch_length,
        # Every window starts from the zero state (`replay_context: 0`).
        "stacked_burn_in": 0,
        "batch_size": config.batch_size,
        "gradient_steps_per_decision": config.gradient_steps_per_decision,
        "warmup_sequences": DREAMER_WARMUP_SEQUENCES,
        # It samples its own policy and adds no exploration noise.
        "exploration": "uniform",
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        # Uniform replay, as the official loop samples.
        "priority_alpha": 0.0,
    }


def settle_dreamer_settings(
    parser: argparse.ArgumentParser, argv: list[str] | None, arguments: argparse.Namespace
) -> None:
    """Under `--backbone dreamerv3`, fix its loop settings and refuse any flag against them.

    A flag that repeats a fixed value is accepted; one that contradicts it, or
    one only stacked-dqn reads, is refused rather than silently overridden.
    """
    if arguments.backbone != DREAMERV3:
        return
    fixed = dreamer_loop_settings()
    # Parsed again with nothing defaulted, so a flag that was given can be told
    # from one that was left alone.
    parser.set_defaults(**{dest: None for dest in (*fixed, *STACKED_ONLY_FLAGS)})
    given = vars(parser.parse_args(argv))
    for dest in STACKED_ONLY_FLAGS:
        if given[dest] is not None:
            raise SystemExit(
                f"--{dest.replace('_', '-')} is a stacked-dqn setting; "
                "--backbone dreamerv3 does not read it"
            )
    for dest, value in fixed.items():
        if given[dest] is not None and given[dest] != value:
            raise SystemExit(
                f"--{dest.replace('_', '-')} {given[dest]} contradicts DreamerV3's "
                f"fixed {value} (docs/solution.md 9.4c)"
            )
        setattr(arguments, dest, value)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Everything the run is configured by, validated before a device is touched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-decisions",
        type=int,
        required=True,
        help="the whole run's budget: cumulative decisions across the fleet",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "a checkpoint to continue a run's budget from: the weights, the "
            "optimizer, the decision and game-time counters and every schedule "
            "and cadence derived from them come back, and the replay buffer is "
            "re-warmed under the loaded policy. --budget-decisions stays the "
            "whole run's total, so a checkpoint at or past it is refused"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backbone",
        choices=BACKBONES,
        default=BACKBONE,
        help=(
            "the learner: stacked-dqn, or dreamerv3 at its published settings "
            "(docs/solution.md 9.4c), which fix the sequence, batch, replay "
            "ratio and exploration flags and refuse a value that contradicts them"
        ),
    )
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
        "--stacked-burn-in",
        type=int,
        default=7,
        help=(
            "exactly history-length - 1, which is what fills the window; anything "
            "longer discards learnable steps for nothing"
        ),
    )
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument(
        "--n-step",
        type=int,
        default=10,
        help="the n-step return, or where it starts under --n-step-final",
    )
    parser.add_argument(
        "--n-step-final",
        type=int,
        default=None,
        help=(
            "anneal n exponentially from --n-step to this over "
            "--n-step-anneal-steps gradient steps, then hold it; unset "
            "holds --n-step fixed"
        ),
    )
    parser.add_argument(
        "--n-step-anneal-steps",
        type=int,
        default=0,
        help="gradient steps the n-step anneal takes; needs --n-step-final",
    )
    parser.add_argument(
        "--discount",
        type=float,
        # Unset rather than 0.99, so a value given beside
        # --discount-per-game-second can be told from the default.
        default=None,
        help=f"the discount per decision (default {DEFAULT_DISCOUNT})",
    )
    parser.add_argument(
        "--discount-per-game-second",
        type=float,
        default=None,
        help=(
            "discount by game time instead of per decision: a transition that "
            "spans t game-seconds is discounted by this ** t, so a purchase "
            "costs no discount (docs/solution.md 9.4d); unset discounts per "
            "decision, and --discount may not be given with it"
        ),
    )
    parser.add_argument(
        "--survival-time-reward",
        action="store_true",
        help=(
            "learn from game time survived, in waves, instead of the wave reward, "
            "so dying later in a wave scores higher (docs/solution.md 9.4e); "
            "needs --discount-per-game-second"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--target-ema-decay",
        type=float,
        default=0.995,
        help="how slowly the target network follows the online one",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument(
        "--epsilon-end",
        type=float,
        # Unset rather than 0.05, so a value the ladder would ignore can be told
        # from the default it resolves to below.
        default=None,
        help=(
            "the rate every actor of a uniform schedule anneals to (default "
            "0.05); under --exploration ladder each actor anneals to its own "
            "rung instead and this may not be given"
        ),
    )
    parser.add_argument(
        "--exploration",
        choices=EXPLORATION_OPTIONS,
        default="uniform",
        help=(
            "uniform anneals every actor to --epsilon-end, which is what every "
            "run so far collected under; ladder anneals actor i of N to the "
            "Ape-X rate 0.4 ** (1 + 7 i / (N - 1)) instead, so one fleet "
            "searches and reports at once. --epsilon-end is the uniform "
            "schedule's floor only and is ignored under ladder, where each "
            "actor has a floor of its own"
        ),
    )
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
    parser.add_argument(
        "--checkpoint-every-decisions",
        type=int,
        default=0,
        help=(
            "decisions between extra numbered checkpoints, written beside "
            "latest.pt as checkpoint-d<decisions>.pt for a learning curve; 0 "
            "writes them only where a selection period closes"
        ),
    )
    parser.add_argument(
        "--selection-period-decisions",
        type=int,
        default=15_000,
        help=(
            "decisions per selection period. A numbered checkpoint is written "
            "where each closes; the arm is the checkpoint of the period with "
            "the highest near-greedy mean final wave from period 2 on, and "
            "early stopping counts these periods"
        ),
    )
    parser.add_argument(
        "--early-stop-patience-periods",
        type=int,
        default=0,
        help=(
            "stop the run when the near-greedy collection curve has not "
            "improved for this many selection periods in a row, after writing "
            "that period's checkpoint; 0 spends the whole budget"
        ),
    )
    parser.add_argument(
        "--early-stop-min-improvement",
        type=float,
        default=0.2,
        help=(
            "waves a selection period must add to the best period mean so far "
            "to count as an improvement; 0.2 is about the standard error of a "
            "hundred-episode window, so anything inside it is noise"
        ),
    )
    parser.add_argument(
        "--kill-bar",
        dest="kill_bars",
        type=kill_bar,
        action="append",
        default=[],
        metavar="AT:START:MIN",
        help=(
            "stop the run if, when the fleet first reaches AT decisions, the "
            "near-greedy actors' valid episodes that ended in (START, AT] "
            "average fewer than MIN waves; repeatable, off by default"
        ),
    )
    parser.add_argument(
        "--parameter-sync-episodes",
        type=int,
        default=1,
        help=(
            "episodes one actor plays between refreshes of the copy of the "
            "network it acts from; 1 starts every episode from the learner's "
            "current parameters, which is what a single actor has always done"
        ),
    )
    parser.add_argument("--serial", default="emulator-5556")
    parser.add_argument("--port", type=int, default=47652)
    parser.add_argument(
        "--actors",
        type=int,
        default=1,
        help=(
            "emulator instances collecting at once, one actor each, into the one "
            "replay and the one learner; above 1 the fleet brings its own "
            "instances up by index and tears them down again"
        ),
    )
    # Read on the fleet path only: a single actor collects on the instance the
    # operator brought up, and nothing here starts or stops it.
    # `-gpu host` is the standing training renderer (M1B-E052 and the handoff's
    # renderer decision): equivalent to lavapipe on game-time ratio,
    # decisions/wave and mean wave, and the one the measured runs were taken
    # under. It cannot snapshot a Vulkan app, so a fleet under it cold-starts
    # every instance instead of restoring the pinned snapshot; `bring_up` says
    # so by name and takes the cold path.
    parser.add_argument("--renderer", default="host", help="fleet bring-up only")
    parser.add_argument(
        "--cores", type=int, default=4, help="emulator cores per instance; fleet only"
    )
    parser.add_argument(
        "--frame-rate-hz",
        type=int,
        default=GUEST_FRAME_RATE_HZ,
        help=(
            "guest frame rate the fleet is raised to after bring-up, as "
            f"run_actors.py does; default {GUEST_FRAME_RATE_HZ}, the fleet "
            "operating rate. Read on the fleet path only: a single actor "
            "collects on the instance the operator brought up and owns its "
            "own rate, and nothing here raises it."
        ),
    )
    add_cadence_arguments(parser)
    add_upgrade_availability_argument(parser)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=state_directory() / "runs",
        help=(
            "the project's git-ignored state directory; checkpoints and "
            "reports are never committed"
        ),
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

    if arguments.discount is not None and arguments.discount_per_game_second is not None:
        # Two definitions of one discount: neither may be silently unused.
        raise SystemExit(
            "--discount and --discount-per-game-second each define the discount; "
            "give one or the other"
        )
    if arguments.discount is None:
        arguments.discount = DEFAULT_DISCOUNT
    if arguments.epsilon_end is None:
        arguments.epsilon_end = DEFAULT_EPSILON_END
    elif arguments.exploration == LADDER:
        # The ladder replaces the end of the anneal per actor, so a value given
        # here would be silently unused - and the one thing an exploration
        # setting may not be is silently unused.
        raise SystemExit(
            "--epsilon-end is the uniform schedule's floor and is ignored under "
            "--exploration ladder, where every actor anneals to its own rung"
        )
    if arguments.serial == "emulator-5554":
        raise SystemExit("refusing to train against the canonical evaluation AVD")
    if arguments.actors < 1:
        raise SystemExit("a run needs at least one actor")
    if arguments.actors > 1:
        first = CloneInstance(index=0)
        if arguments.serial != first.serial or arguments.port != first.bridge_host_port:
            # A fleet derives every instance from `CloneInstance`, the same
            # source `run_actors.py` uses, so a serial or port named by hand
            # would address one instance and be ignored for the rest.
            raise SystemExit(
                "a fleet takes its instances from CloneInstance by index; "
                "--serial and --port configure a single actor only"
            )
        if arguments.evaluate_every_episodes:
            # Evaluation borrows an instance, and every instance is collecting.
            # The pre-registered final evaluation is unaffected: it is taken
            # after the budget is spent, with the fleet stopped.
            raise SystemExit(
                "mid-run evaluation needs an instance to itself and cannot run "
                "while a fleet is collecting; leave --evaluate-every-episodes at 0"
            )
    if not 1 <= arguments.frame_rate_hz <= MAX_GUEST_FRAME_RATE_HZ:
        # The same measured ceiling `run_actors.py` holds its rate to: above it
        # the guest reports a rate it is not delivering, so `confirm_frame_rate`
        # would pass on an instance collecting at some other rate entirely.
        raise SystemExit(
            f"--frame-rate-hz {arguments.frame_rate_hz} is outside "
            f"1..{MAX_GUEST_FRAME_RATE_HZ}; no measured fps supports a guest "
            "rate above that"
        )
    if arguments.budget_decisions < 1:
        raise SystemExit("--budget-decisions must be positive")
    if arguments.checkpoint_every_decisions < 0:
        raise SystemExit("--checkpoint-every-decisions cannot be negative")
    if arguments.selection_period_decisions < 1:
        raise SystemExit("--selection-period-decisions must be positive")
    if arguments.early_stop_patience_periods < 0:
        raise SystemExit("--early-stop-patience-periods cannot be negative")
    if arguments.early_stop_min_improvement < 0:
        raise SystemExit("--early-stop-min-improvement cannot be negative")
    if (arguments.n_step_final is None) != (arguments.n_step_anneal_steps == 0):
        raise SystemExit(
            "--n-step-final and --n-step-anneal-steps configure one anneal and "
            "are given together or not at all"
        )
    if arguments.n_step_anneal_steps < 0:
        raise SystemExit("--n-step-anneal-steps cannot be negative")
    settle_dreamer_settings(parser, argv, arguments)
    if arguments.survival_time_reward and arguments.discount_per_game_second is None:
        # The reward is integrated under the game-time discount; per decision
        # a span has no length to integrate over. After the DreamerV3 check, so
        # that backbone is told it does not read the flag at all.
        raise SystemExit("--survival-time-reward needs --discount-per-game-second")
    if (
        arguments.backbone == BACKBONE
        and arguments.stacked_burn_in < arguments.history_length - 1
    ):
        # Checked here rather than at the first optimisation step, which is an
        # hour of collection later.
        raise SystemExit(
            f"burn-in {arguments.stacked_burn_in} cannot fill a window of "
            f"{arguments.history_length}"
        )
    if arguments.stacked_burn_in >= arguments.sequence_length:
        raise SystemExit(
            f"burn-in {arguments.stacked_burn_in} leaves no learning steps "
            f"in a sequence of {arguments.sequence_length}"
        )
    return arguments


def resume_point(
    arguments: argparse.Namespace, *, profile_id: str, revision: str
) -> ResumeState | None:
    """The checkpoint this run continues, read and checked before a device is touched.

    Three refusals, all here rather than an hour into collection. The identity
    is the checkpoint's own: a file from another arm, another device profile or
    another observation, action or reward schema is not experience this run can
    go on from, and `CheckpointIdentity.incompatibilities` names which. The
    format is the second: a file from before the decision budget (format below
    4) counted its selection periods in game time, so it is for evaluation
    only. The budget is the third: `--budget-decisions` is the whole run's total, not
    this segment's, so a checkpoint at 50,123 of 120,000 continues to 120,000
    and a larger budget extends the run - but a checkpoint that has already
    spent the budget is nothing this run can add to.
    """
    if arguments.resume is None:
        return None
    # Built exactly as the fresh-run path builds it, from the profile the bridge
    # reported and the schema versions this code is; the run id and the source
    # revision in it are deliberately not compared.
    expected = checkpoint_identity(
        RunIdentity.started_now(
            arguments.backbone,
            profile_id=profile_id,
            source_revision=revision,
            # The cadence this run will collect under: a checkpoint collected
            # under the other one is not experience it can continue (ADR 0009).
            decision_cadence=decision_cadence_from(arguments),
            # A checkpoint collected on other rows is not experience this run
            # can continue either (ADR 0011).
            upgrade_availability=upgrade_availability_from(arguments),
        )
    )
    try:
        state = resume_state(arguments.resume, expected=expected)
    except CheckpointError as failure:
        raise SystemExit(f"--resume {arguments.resume}: {failure}") from failure
    if state.format_version < DECISION_BUDGET_FORMAT_VERSION:
        raise SystemExit(
            f"--resume {arguments.resume} is a format {state.format_version} "
            "checkpoint from the game-time budget era: it is for evaluation "
            "only and cannot be resumed under --budget-decisions"
        )
    if arguments.backbone == BACKBONE and "discount" in state.resolved_config:
        # The discount defines the target. Resuming under another one would
        # train one set of weights towards two value scales without a word.
        # A file that recorded no settings at all has nothing to compare; one
        # from before the game-time discount has no per-second key, which
        # reads as None - per decision, which is what it was trained under.
        recorded = (
            state.resolved_config.get("discount"),
            state.resolved_config.get("discount_per_game_second"),
            # Likewise the reward: a file from before the survival-time reward
            # has no key, which reads as off - the wave reward it learned from.
            state.resolved_config.get("survival_time_reward", False),
        )
        requested = (
            None if arguments.discount_per_game_second is not None else arguments.discount,
            arguments.discount_per_game_second,
            arguments.survival_time_reward,
        )
        if recorded != requested:
            raise SystemExit(
                f"--resume {arguments.resume} was trained with discount {recorded[0]}, "
                f"discount per game-second {recorded[1]} and survival-time reward "
                f"{recorded[2]}; this run asks for {requested[0]}, {requested[1]} "
                f"and {requested[2]}, a different target"
            )
    if state.decisions >= arguments.budget_decisions:
        raise SystemExit(
            f"--resume {arguments.resume} is already at {state.decisions} "
            f"decisions, which --budget-decisions {arguments.budget_decisions} "
            "does not extend; raise the budget above it to continue the run"
        )
    return state


def build_tracker(arguments: argparse.Namespace) -> ExperimentTracker:
    """The tracker the session records itself through.

    The MLflow adapter is imported here and nowhere else, so a machine without
    MLflow installed can still run everything that does not ask to be tracked.
    """
    if not arguments.track:
        return NoExperimentTracker()
    try:
        from tower_rl.experiment.mlflow_tracking import MlflowExperimentTracker
    except ImportError as missing:
        raise SystemExit(
            f"tracking is on but MLflow is not installed ({missing}). "
            "Install it with `uv sync --extra tracking`, or pass --no-track "
            "to run untracked."
        ) from missing
    arguments.run_dir.parent.mkdir(parents=True, exist_ok=True)
    return MlflowExperimentTracker(
        tracking_uri=tracking_uri(arguments.run_dir),
        experiment=arguments.experiment,
        artifact_root=artifact_root(arguments.run_dir),
    )


def train_session(
    arguments: argparse.Namespace,
    instances: Sequence[ActorInstance],
    *,
    profile_id: str,
    revision: str,
    device: torch.device,
    tracker: ExperimentTracker | None = None,
    bridge_version: str | None = None,
    bring_up_failures: Sequence[str] = (),
    resume: ResumeState | None = None,
) -> dict[str, object]:
    """Train the arm to its budget and return the session report.

    The instances are a parameter rather than something built here, so the one
    place a bridge to a real device is opened is `main`. Nothing else decides
    what the arm is talking to.
    """
    started = time.monotonic()
    session = arguments.run_dir / f"session-{time.strftime('%Y%m%d-%H%M%S')}"
    recorder = NoExperimentTracker() if tracker is None else tracker
    tags = {
        "source_revision": revision,
        "profile_id": profile_id,
        "session": session.name,
        "device_serial": ",".join(instance.serial for instance in instances),
        # The fleet the arm collected with, as it actually came up: an instance
        # that failed its bring-up is not one of these.
        "actors": str(len(instances)),
    }
    if bridge_version is not None:
        tags["bridge_version"] = bridge_version
    arm, run_evaluation = build_arm(
        arguments.backbone,
        arguments,
        instances=instances,
        device=device,
        profile_id=profile_id,
        parent=session,
        revision=revision,
        started=started,
        tracker=recorder,
        tags=tags,
        resume=resume,
    )

    try:
        # To the whole run's budget: a resumed run starts with part of it spent.
        arm.training.run()

        killed = arm.training.killed_by
        if killed is not None:
            bar = killed.bar
            print(
                f"[{arm.name}] stopped at {killed.decisions} decisions on the kill "
                f"bar at {bar.at_decisions}: the near-greedy mean final wave over "
                f"({bar.window_start_decisions}, {bar.at_decisions}] was "
                f"{killed.mean_final_wave:.2f} over {killed.near_greedy_episodes} "
                f"episodes, below {bar.min_mean_final_wave}",
                flush=True,
            )
        elif arm.training.stopped_early:
            plateau = arm.training.report.plateau
            # An early stop is the run's own decision and is invisible in the
            # counters alone - a run that stopped at 30,000 of 60,000
            # decisions looks like one that was interrupted.
            print(
                f"[{arm.name}] stopped early at selection period "
                f"{plateau.stopped_at_period}: the near-greedy curve did not "
                f"improve on {plateau.best_mean_final_wave:.2f} waves for "
                f"{arguments.early_stop_patience_periods} periods",
                flush=True,
            )

        arm.checkpoint(arm.training.report)
        # The one pre-registered measurement of the run: exploration-free, on
        # the final weights, sized so its standard error can resolve a real
        # difference against the scripted floor. Taken after the budget is
        # spent, so it costs none of the budget and cannot be chosen after
        # the fact from a series of mid-run points.
        # Skipped for a run stopped on a kill bar: whether a killed run's arm
        # is evaluated is its pre-registration's decision (solution.md 9.2b),
        # and this evaluation costs hours of device time (2.28 h in run 3).
        # The summary records the skip. A plateau stop still evaluates.
        if killed is not None:
            print(f"[{arm.name}] final evaluation skipped: stopped on a kill bar", flush=True)
        else:
            try:
                run_evaluation(True)
            except (RunPortError, ValueError) as failure:
                # Losing the headline measurement must not lose the run: the
                # collection curve and the checkpoints are already on disk.
                arm.training.report.evaluation_failures.append(str(failure))
                print(f"[{arm.name}] final evaluation failed: {failure}", flush=True)

        summary = arm.summary()
        report: dict[str, object] = {
            "session": str(session),
            "profile_id": profile_id,
            "source_revision": revision,
            "budget_decisions": arguments.budget_decisions,
            "actors": len(instances),
            "actor_serials": [instance.serial for instance in instances],
            # Instances that never came up at all, which cost the fleet an actor
            # before a single episode was collected.
            "bring_up_failures": list(bring_up_failures),
            "wall_seconds": round(time.monotonic() - started, 1),
            # Repeated at the top of the report as well as inside each arm: the
            # curve is meaningless without the floors it is read against.
            "reference_final_waves": REFERENCE_FINAL_WAVES,
            # One arm. The session used to carry a list of them, from a
            # comparison of several backbones that was retired: this project
            # trains one backbone and compares it against the non-learned floors
            # afterwards, through `report_arms.py`, not inside a session.
            "arm": summary,
        }
        session.mkdir(parents=True, exist_ok=True)
        (session / "summary.json").write_text(json.dumps(report, indent=2, default=str))
        # The arm's summary holds its learning curve and the per-episode
        # evaluation records, so it is what a tracked run is read from.
        path = arm.run_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str))
        arm.run.log_artifact(path)
        return report
    finally:
        # A run that ended badly is still a run that has to be closed, or it
        # would sit open in the store forever.
        arm.run.finish()


def connect(
    serial: str,
    port: int,
    arguments: argparse.Namespace,
    expected: BridgeCompatibility,
    opened: list[tuple[InstrumentedRunAdapter, InstrumentedBridgeClient]],
) -> ActorInstance:
    """Open one instance's bridge and present it as an environment to train on.

    Every client it opens is appended to `opened`, which is what the caller
    releases and closes afterwards: a fleet that half connected must still put
    down every bridge it picked up.
    """
    client, adapter, environment = open_environment(port, expected, arguments)
    opened.append((adapter, client))
    return ActorInstance(serial=serial, environment=environment)


def main() -> int:
    arguments = parse_arguments()
    revision = source_revision()

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
        print(f"artifacts: {artifact_root(arguments.run_dir)}", flush=True)

    expected = compatibility(bridge_build_directory())
    # Read once the profile the checkpoint is checked against is known, and
    # still before anything is brought up: a resume that cannot be honoured
    # must fail now, not an hour into collection.
    resume = resume_point(arguments, profile_id=expected.profile_id, revision=revision)
    if (
        resume is not None
        and resume.tracking_run_id is not None
        # There is nothing to attach to under `--no-track`: the parent's run id
        # names a run in a store this session is not recording into, and saying
        # it was resumed would be a claim about a curve nothing is writing.
        and not isinstance(tracker, NoExperimentTracker)
    ):
        # The parent run is read back here, where a missing one costs nothing,
        # rather than at `build_arm` - which runs with the fleet already up.
        # The handle is discarded; the arm opens its own.
        tracker.open_run(resume.tracking_run_id)
        print(f"resuming tracked run {resume.tracking_run_id}", flush=True)

    opened: list[tuple[InstrumentedRunAdapter, InstrumentedBridgeClient]] = []
    started: list[CloneInstance] = []

    def open_instance(instance: CloneInstance) -> ActorInstance:
        """Bring one instance of the fleet up ready and offline, and connect."""
        # Registered before bring-up is attempted, not after it succeeds: a
        # bring-up that fails partway (or an emulator that comes up but never
        # reaches offline) must still be torn down, so anything that might
        # have started an emulator process has to be in `started` before that
        # attempt, not only once it is known to have worked.
        started.append(instance)
        bring_up(
            instance,
            arguments.renderer,
            deploy=deploy_bridge,
            read_only=True,
            cores=arguments.cores,
            frame_rate_hz=arguments.frame_rate_hz,
        )
        # By interface, per instance, before anything is collected on it.
        require_offline(instance)
        # Every instance boots at the stock 60 Hz per-uid game override; this
        # actor raises its own instance the moment its own bring-up returns,
        # exactly where `run_actors.collect_episodes` does it, and immediately
        # before anything is measured. A failed raise here fails this instance
        # by name, the same as a failed bring-up: `bring_up_fleet`'s existing
        # per-instance failure handling applies.
        require_game_activity(instance)
        raise_frame_rate(instance, arguments.frame_rate_hz)
        return connect(instance.serial, instance.bridge_host_port, arguments, expected, opened)

    failures: list[str] = []
    try:
        if arguments.actors == 1:
            # Exactly as before there were fleets: the instance the operator
            # brought up, addressed by --serial and --port, and left running.
            # --frame-rate-hz is not applied here: the operator owns this
            # instance's rate along with everything else about how it was
            # brought up, so nothing on this path raises it.
            instances = [connect(arguments.serial, arguments.port, arguments, expected, opened)]
        else:
            # A read-only fleet cannot save a snapshot, so the pinned one is
            # prepared once before any actor starts, exactly as the scripted
            # fleet does it.
            prepare_pinned_snapshot(arguments.renderer, arguments.cores)
            instances, failures = bring_up_fleet(
                [CloneInstance(index=index) for index in range(arguments.actors)],
                open_instance,
            )
        report = train_session(
            arguments,
            instances,
            profile_id=expected.profile_id,
            revision=revision,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            tracker=tracker,
            bridge_version=expected.bridge_version,
            bring_up_failures=failures,
            resume=resume,
        )
    finally:
        tear_down_fleet(opened, started)

    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
