#!/usr/bin/env python3
"""Train the stacked-DQN backbone against the clone under one fixed budget.

Private device runner for the instrumented-training profile. Everything it
produces - checkpoints, reports, replay metadata - is written outside the
repository, and every tap it can make is gated inside the adapter on a positive
screen classification.

Collection runs in decision blocks (`--block-decisions`), each landing on an
episode boundary, until the budget is spent.

    TOWER_BRIDGE_BUILD_DIR=... uv run --extra tracking python scripts/train.py \\
        --budget-decisions 20000

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

    TOWER_BRIDGE_BUILD_DIR=... uv run --extra tracking python scripts/train.py \\
        --actors 4 --budget-decisions 100000

The run records itself to the local MLflow store under `~/.local/state/tower-rl`;
`--extra tracking` is what puts MLflow in the environment. Pass `--no-track` to
run without recording, which leaves nothing to compare the run against later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from clone_session import CloneInstance, bring_up, require_offline  # noqa: E402
from run_actors import (  # noqa: E402
    deploy_bridge,
    prepare_pinned_snapshot,
    tear_down_instance,
)
from run_episodes import (  # noqa: E402
    add_cadence_arguments,
    cadence_from,
    compatibility,
)

from tower_rl.application.actor import Actor, ActorConfig  # noqa: E402
from tower_rl.application.evaluator import EvaluationReport, evaluate  # noqa: E402
from tower_rl.application.replay import PrioritizedSequenceReplay  # noqa: E402
from tower_rl.application.run_environment import InstrumentedRunEnvironment  # noqa: E402
from tower_rl.application.training import (  # noqa: E402
    ActorProgress,
    TrainingConfig,
    TrainingProgressReport,
    TrainingRun,
)
from tower_rl.domain.episode import REWARD_SCHEMA_VERSION  # noqa: E402
from tower_rl.domain.run_actions import ACTION_SCHEMA_VERSION  # noqa: E402
from tower_rl.domain.run_state import OBSERVATION_SCHEMA_VERSION, RunStateBuilder  # noqa: E402
from tower_rl.experiment.run_identity import (  # noqa: E402
    REFERENCE_FINAL_WAVES,
    new_run_id,
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
from tower_rl.infrastructure.instrumented_bridge import (  # noqa: E402
    BridgeCompatibility,
    InstrumentedBridgeClient,
)
from tower_rl.infrastructure.instrumented_run_adapter import InstrumentedRunAdapter  # noqa: E402
from tower_rl.learning.backbone import Backbone  # noqa: E402
from tower_rl.learning.checkpoint import CheckpointIdentity, write_manifest  # noqa: E402
from tower_rl.learning.network import NetworkConfig  # noqa: E402
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig  # noqa: E402
from tower_rl.ports.run_port import RunPortError  # noqa: E402

#: The one backbone this project trains.
BACKBONE = "stacked-dqn"


@dataclass(frozen=True)
class ActorInstance:
    """One emulator instance an actor collects on, and what it is called.

    Identity and environment travel together because the report is per actor:
    an aggregate that cannot name the instance a failure came from cannot say
    which emulator to look at.
    """

    serial: str
    environment: InstrumentedRunEnvironment


def build_backbone(
    arguments: argparse.Namespace, device: torch.device
) -> tuple[Backbone, StackedDqnConfig]:
    """The backbone and the learner settings it was fixed with.

    The settings come back alongside it because they are part of what the arm
    is configured by - n-step, discount, learning rate - and a run that does not
    record them cannot be compared with the next one.
    """
    stacked = StackedDqnConfig(
        seed=arguments.seed,
        history_length=arguments.history_length,
        n_step=arguments.n_step,
        discount=arguments.discount,
        learning_rate=arguments.learning_rate,
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
    instances: Sequence[ActorInstance],
    device: torch.device,
    profile_id: str,
    parent: Path,
    revision: str,
    started: float,
    tracker: ExperimentTracker,
    tags: dict[str, str],
) -> TrainingReport:
    run_id = new_run_id(name)
    run_dir = parent / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    backbone, learner = build_backbone(arguments, device)
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
        parameter_sync_episodes=arguments.parameter_sync_episodes,
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
        cadence=instances[0].environment.cadence,
        burn_in=burn_in,
        stride=stride,
        device=device,
    )
    run = tracker.start_run(
        name=run_id,
        params=tracked_params(resolved),
        tags={**tags, "backbone": name, "run_id": run_id},
    )
    print(f"[{name}] tracking run {run.run_id}", flush=True)
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
        if pre_registered_final:
            arm.final_point = point
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

    def on_episode(report: TrainingProgressReport) -> None:
        arm.record_collection_windows()
        arm.record_decision_time()
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
    arm.training.on_withdrawal = on_withdrawal
    manifest = run_dir / "manifest.json"
    write_manifest(manifest, {"run_id": run_id, **resolved})
    run.log_artifact(manifest)
    return arm


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Everything the run is configured by, validated before a device is touched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-decisions", type=int, default=20_000)
    parser.add_argument(
        "--block-decisions",
        type=int,
        default=2_000,
        help="decisions collected before the loop checks the budget; lands on an episode",
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
        "--stacked-burn-in",
        type=int,
        default=7,
        help=(
            "exactly history-length - 1, which is what fills the window; anything "
            "longer discards learnable steps for nothing"
        ),
    )
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--target-ema-decay",
        type=float,
        default=0.995,
        help="how slowly the target network follows the online one",
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
    parser.add_argument("--renderer", default="lavapipe", help="fleet bring-up only")
    parser.add_argument(
        "--cores", type=int, default=4, help="emulator cores per instance; fleet only"
    )
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
    if arguments.stacked_burn_in < arguments.history_length - 1:
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
    arm = build_arm(
        BACKBONE,
        arguments,
        instances=instances,
        device=device,
        profile_id=profile_id,
        parent=session,
        revision=revision,
        started=started,
        tracker=recorder,
        tags=tags,
    )

    blocks = -(-arguments.budget_decisions // arguments.block_decisions)
    try:
        for _ in range(blocks):
            if arm.training.finished:
                break
            arm.training.advance(arguments.block_decisions)

        arm.checkpoint(arm.training.report)
        # The one pre-registered measurement of the run: exploration-free, on
        # the final weights, sized so its standard error can resolve a real
        # difference against the scripted floor. Taken after the budget is
        # spent, so it costs no decisions and cannot be chosen after the fact
        # from a series of mid-run points.
        if arm.evaluation is not None:
            try:
                arm.evaluation(True)
            except (RunPortError, ValueError) as failure:
                # Losing the headline measurement must not lose the run: the
                # collection curve and the checkpoints are already on disk.
                arm.training.report.evaluation_failures.append(str(failure))
                print(f"[{arm.name}] final evaluation failed: {failure}", flush=True)

        summaries = [arm.summary()]
        report: dict[str, object] = {
            "session": str(session),
            "profile_id": profile_id,
            "source_revision": revision,
            "budget_decisions_per_arm": arguments.budget_decisions,
            "block_decisions": arguments.block_decisions,
            "actors": len(instances),
            "actor_serials": [instance.serial for instance in instances],
            # Instances that never came up at all, which cost the fleet an actor
            # before a single episode was collected.
            "bring_up_failures": list(bring_up_failures),
            "wall_seconds": round(time.monotonic() - started, 1),
            # Repeated at the top of the report as well as inside each arm: the
            # curve is meaningless without the floors it is read against.
            "reference_final_waves": REFERENCE_FINAL_WAVES,
            "arms": summaries,
        }
        session.mkdir(parents=True, exist_ok=True)
        (session / "summary.json").write_text(json.dumps(report, indent=2, default=str))
        # The arm's summary holds its learning curve and the per-episode
        # evaluation records, so it is what a tracked run is read from.
        path = arm.run_dir / "summary.json"
        path.write_text(json.dumps(summaries[0], indent=2, default=str))
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
    client = InstrumentedBridgeClient(
        "127.0.0.1",
        port,
        expected_compatibility=expected,
        connect_timeout=5.0,
        read_timeout=120.0,
        heartbeat_timeout=60.0,
    )
    client.connect()
    adapter = InstrumentedRunAdapter(client=client)
    opened.append((adapter, client))
    return ActorInstance(
        serial=serial,
        environment=InstrumentedRunEnvironment(
            port=adapter,
            builder=RunStateBuilder(profile_id=expected.profile_id),
            cadence=cadence_from(arguments),
        ),
    )


def bring_up_fleet(
    instances: Sequence[CloneInstance],
    open_instance: Callable[[CloneInstance], ActorInstance],
) -> tuple[list[ActorInstance], list[str]]:
    """Bring each instance up only after the previous one's bring-up concludes.

    This is `run_actors.stagger_bring_up`'s rule with nothing left to gate.
    Four emulators cold-booting at the same instant pushed host load to 10.71
    and left the last of them unable to reach home inside its timeout, while
    steady-state collection uses 563% of 3,200% available CPU (M1B-E028): the
    contention is entirely in the boot, so a bring-up must not begin until the
    previous one has concluded. There it is an event each actor waits on,
    because collection starts as soon as an instance is ready; here training
    starts only once the fleet is up, so sequencing the bring-ups is the same
    rule and needs no gate at all.

    A bring-up that fails costs that actor and not the fleet, exactly as a
    failed bring-up there still releases the next actor: it is reported, and the
    next instance is brought up regardless.
    """
    ready: list[ActorInstance] = []
    failures: list[str] = []
    for instance in instances:
        try:
            ready.append(open_instance(instance))
        except Exception as error:  # noqa: BLE001 - one actor's failure, not the fleet's
            failures.append(f"{instance.serial}: {type(error).__name__}: {error}")
            print(f"{instance.serial}: bring-up failed: {error}", flush=True)
            continue
        print(f"{instance.serial}: ready", flush=True)
    if not ready:
        raise SystemExit(f"no instance of the fleet came up: {'; '.join(failures)}")
    return ready, failures


def tear_down_fleet(
    opened: Sequence[tuple[InstrumentedRunAdapter, InstrumentedBridgeClient]],
    started: Sequence[CloneInstance],
    tear_down: Callable[[CloneInstance], None] = tear_down_instance,
) -> None:
    """Put down every bridge and every instance this run brought up.

    Leaving an emulator running is a safety failure rather than an
    inconvenience, so each step here is independent and best-effort: releasing
    reads the bridge, and on a client that had stopped answering that read
    raised inside the caller's `finally`, skipping every remaining release and
    every teardown and leaving four emulators running with the overlay mounted.
    A failure is reported and the next instance is put down anyway.
    """
    for adapter, client in opened:
        try:
            adapter.release()
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            print(f"bridge on port {client.port}: release failed: {error}", flush=True)
        finally:
            client.close()
    for instance in started:
        # Only instances this run brought up are torn down, and one that
        # refuses to clean up must not leave the others running.
        try:
            tear_down(instance)
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            print(f"{instance.serial}: teardown failed: {error}", flush=True)


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
        print(f"artifacts: {artifact_root(arguments.run_dir)}", flush=True)

    build_dir = Path(
        os.environ.get("TOWER_BRIDGE_BUILD_DIR")
        or Path("/tmp/tower-bridge-live.latest").read_text().strip()
    )
    expected = compatibility(build_dir)
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
        )
        # By interface, per instance, before anything is collected on it.
        require_offline(instance)
        return connect(instance.serial, instance.bridge_host_port, arguments, expected, opened)

    failures: list[str] = []
    try:
        if arguments.actors == 1:
            # Exactly as before there were fleets: the instance the operator
            # brought up, addressed by --serial and --port, and left running.
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
            revision=source_revision(),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            tracker=tracker,
            bridge_version=expected.bridge_version,
            bring_up_failures=failures,
        )
    finally:
        tear_down_fleet(opened, started)

    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
