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

    TOWER_BRIDGE_BUILD_DIR=... uv run python scripts/train.py \\
        --backbone recurrent-q --backbone stacked-dqn \\
        --budget-decisions 20000 --speed 64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from run_episodes import compatibility  # noqa: E402

from tower_rl.application.actor import Actor, ActorConfig  # noqa: E402
from tower_rl.application.comparison import interleave_schedule  # noqa: E402
from tower_rl.application.evaluator import EvaluationReport, evaluate, to_record  # noqa: E402
from tower_rl.application.replay import PrioritizedSequenceReplay  # noqa: E402
from tower_rl.application.run_environment import (  # noqa: E402
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.application.training import (  # noqa: E402
    TrainingConfig,
    TrainingProgressReport,
    TrainingRun,
)
from tower_rl.domain.episode import REWARD_SCHEMA_VERSION  # noqa: E402
from tower_rl.domain.run_actions import ACTION_SCHEMA_VERSION  # noqa: E402
from tower_rl.domain.run_state import OBSERVATION_SCHEMA_VERSION, RunStateBuilder  # noqa: E402
from tower_rl.infrastructure.adb_device import AdbDevice  # noqa: E402
from tower_rl.infrastructure.instrumented_bridge import InstrumentedBridgeClient  # noqa: E402
from tower_rl.infrastructure.instrumented_run_adapter import InstrumentedRunAdapter  # noqa: E402
from tower_rl.learning.backbone import Backbone  # noqa: E402
from tower_rl.learning.checkpoint import (  # noqa: E402
    Checkpoint,
    CheckpointIdentity,
    TrainingProgress,
    save,
    write_manifest,
)
from tower_rl.learning.network import NetworkConfig  # noqa: E402
from tower_rl.learning.recurrent_q import RecurrentQBackbone, RecurrentQConfig  # noqa: E402
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig  # noqa: E402

#: Every backbone in the comparison, addressed identically. Adding one here is
#: all it takes to put it under the same protocol as the others.
BACKBONES = ("recurrent-q", "stacked-dqn")


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

    def checkpoint(self, report: TrainingProgressReport) -> None:
        config = self.training.config
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
            self.run_dir / "checkpoints" / "latest.pt",
        )

    def summary(self) -> dict[str, object]:
        report = self.training.report
        return {
            "backbone": self.name,
            "run_id": self.identity.run_id,
            "resolved_config": self.resolved,
            "decisions": report.decisions,
            "episodes": report.episodes,
            "valid_episodes": report.valid_episodes,
            "optimisation_steps": report.optimisation_steps,
            "sequences_accepted": report.sequences_accepted,
            "wall_seconds": report.wall_seconds,
            "final_waves": report.final_waves,
            "evaluations": [to_record(item) for item in report.evaluations],
            "replay": self.replay.snapshot(),
        }


def build_backbone(
    name: str, arguments: argparse.Namespace, device: torch.device
) -> Backbone:
    if name == "recurrent-q":
        return RecurrentQBackbone(
            config=RecurrentQConfig(seed=arguments.seed),
            network_config=NetworkConfig(),
            device=device,
        )
    return StackedDqnBackbone(
        config=StackedDqnConfig(seed=arguments.seed, history_length=arguments.history_length),
        network_config=NetworkConfig(),
        device=device,
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
) -> Arm:
    run_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = parent / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    backbone = build_backbone(name, arguments, device)
    replay = PrioritizedSequenceReplay(capacity=arguments.replay_capacity, seed=arguments.seed)
    actor = Actor(
        environment=environment,
        policy=backbone,
        config=ActorConfig(
            actor_id=f"{arguments.serial}:{name}",
            sequence_length=arguments.sequence_length,
            burn_in=arguments.burn_in,
            stride=max(1, arguments.sequence_length // 2),
        ),
        replay=replay,
    )
    config = TrainingConfig(
        budget_decisions=arguments.budget_decisions,
        batch_size=arguments.batch_size,
        gradient_steps_per_decision=arguments.gradient_steps_per_decision,
        evaluate_every_episodes=arguments.evaluate_every_episodes,
        checkpoint_every_episodes=arguments.checkpoint_every_episodes,
    )
    resolved: dict[str, object] = {
        "backbone": name,
        "budget_decisions": arguments.budget_decisions,
        "speed": arguments.speed,
        "seed": arguments.seed,
        "batch_size": arguments.batch_size,
        "gradient_steps_per_decision": arguments.gradient_steps_per_decision,
        "sequence_length": arguments.sequence_length,
        "burn_in": arguments.burn_in,
        "history_length": arguments.history_length if name == "stacked-dqn" else None,
        "slice_game_ms": arguments.slice_ms,
        "block_decisions": arguments.block_decisions,
        "device": str(device),
    }
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

    def run_evaluation() -> EvaluationReport:
        # Exploration-free, never written to replay; the evaluator enforces both.
        report = evaluate(
            environment,
            backbone,
            episodes=arguments.evaluation_episodes,
            profile_id=profile_id,
            model_version=backbone.model_version,
        )
        print(f"[{name}] {report.summary_line()}", flush=True)
        return report

    arm.training.evaluate = run_evaluation
    arm.training.checkpoint = arm.checkpoint
    arm.training.on_episode = lambda report: print(
        f"[{name}] episode {report.episodes} decisions {report.decisions}/"
        f"{config.budget_decisions} steps {report.optimisation_steps}",
        flush=True,
    )
    write_manifest(run_dir / "manifest.json", {"run_id": run_id, **resolved})
    return arm


def main() -> int:
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
    parser.add_argument("--speed", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-steps-per-decision", type=float, default=0.5)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--burn-in", type=int, default=20)
    # Read by stacked-dqn only; the recurrent backbone carries time in its state.
    parser.add_argument("--history-length", type=int, default=8)
    # Evaluation costs device time at the same rate as training, so its period
    # is long and its sample is the 23 episodes M1B-E008 sized for one wave.
    parser.add_argument("--evaluate-every-episodes", type=int, default=100)
    parser.add_argument("--evaluation-episodes", type=int, default=23)
    parser.add_argument("--checkpoint-every-episodes", type=int, default=25)
    parser.add_argument("--serial", default="emulator-5556")
    parser.add_argument("--port", type=int, default=47652)
    parser.add_argument("--slice-ms", type=int, default=250)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path.home() / ".local/state/tower-rl/runs",
        help="outside the repository; checkpoints and reports are never committed",
    )
    arguments = parser.parse_args()

    names = list(dict.fromkeys(arguments.backbone or ["recurrent-q"]))
    if arguments.serial == "emulator-5554":
        raise SystemExit("refusing to train against the canonical evaluation AVD")
    if "stacked-dqn" in names and arguments.burn_in < arguments.history_length - 1:
        # Checked here rather than at the first optimisation step, which is an
        # hour of collection later.
        raise SystemExit(
            f"burn-in {arguments.burn_in} cannot fill a window of {arguments.history_length}"
        )

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
    adapter = InstrumentedRunAdapter(
        client=client, device=AdbDevice(arguments.serial), requested_speed=arguments.speed
    )
    environment = InstrumentedRunEnvironment(
        port=adapter,
        builder=RunStateBuilder(profile_id=expected.profile_id),
        cadence=CadenceConfig(
            slice_game_ms=arguments.slice_ms,
            max_quiet_game_ms=arguments.slice_ms * 8,
            max_episode_wall_seconds=600.0,
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session = arguments.run_dir / f"session-{time.strftime('%Y%m%d-%H%M%S')}"
    revision = source_revision()
    arms = {
        name: build_arm(
            name,
            arguments,
            environment=environment,
            device=device,
            profile_id=expected.profile_id,
            parent=session,
            revision=revision,
        )
        for name in names
    }

    blocks_per_arm = -(-arguments.budget_decisions // arguments.block_decisions)
    schedule = interleave_schedule(tuple(names), blocks_per_arm, block=1, seed=arguments.seed)
    started = time.monotonic()
    try:
        for name in schedule:
            arm = arms[name]
            if arm.training.finished:
                continue
            arm.training.advance(arguments.block_decisions)
    finally:
        adapter.release()
        client.close()

    for arm in arms.values():
        arm.checkpoint(arm.training.report)

    report = {
        "session": str(session),
        "profile_id": expected.profile_id,
        "source_revision": revision,
        "speed": arguments.speed,
        "budget_decisions_per_arm": arguments.budget_decisions,
        "block_decisions": arguments.block_decisions,
        "wall_seconds": round(time.monotonic() - started, 1),
        "arms": [arm.summary() for arm in arms.values()],
    }
    session.mkdir(parents=True, exist_ok=True)
    (session / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    for arm in arms.values():
        (arm.run_dir / "summary.json").write_text(
            json.dumps(arm.summary(), indent=2, default=str)
        )
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
