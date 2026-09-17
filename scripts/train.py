#!/usr/bin/env python3
"""Train one backbone against the clone under a decision budget.

Private device runner for the instrumented-training profile. Everything it
produces - checkpoints, reports, replay metadata - is written outside the
repository, and every tap it can make is gated inside the adapter on a positive
screen classification.

    TOWER_BRIDGE_BUILD_DIR=... uv run python scripts/train.py \\
        --backbone recurrent-q --budget-decisions 20000 --speed 64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from run_episodes import compatibility  # noqa: E402

from tower_rl.application.actor import Actor, ActorConfig  # noqa: E402
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
from tower_rl.learning.checkpoint import (  # noqa: E402
    Checkpoint,
    CheckpointIdentity,
    TrainingProgress,
    save,
    write_manifest,
)
from tower_rl.learning.network import NetworkConfig  # noqa: E402
from tower_rl.learning.recurrent_q import RecurrentQBackbone, RecurrentQConfig  # noqa: E402

BACKBONES = {"recurrent-q": RecurrentQBackbone}


def source_revision() -> str:
    """Bind every artifact to the code that produced it."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="recurrent-q")
    parser.add_argument("--budget-decisions", type=int, default=20_000)
    parser.add_argument("--speed", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-steps-per-decision", type=float, default=0.5)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--burn-in", type=int, default=20)
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

    if arguments.serial == "emulator-5554":
        raise SystemExit("refusing to train against the canonical evaluation AVD")

    run_id = f"{arguments.backbone}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = arguments.run_dir / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

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
    backbone = BACKBONES[arguments.backbone](
        config=RecurrentQConfig(seed=arguments.seed),
        network_config=NetworkConfig(),
        device=device,
    )
    replay = PrioritizedSequenceReplay(capacity=arguments.replay_capacity, seed=arguments.seed)
    actor = Actor(
        environment=environment,
        policy=backbone,
        config=ActorConfig(
            actor_id=f"{arguments.serial}:{arguments.backbone}",
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
    identity = CheckpointIdentity(
        run_id=run_id,
        backbone=arguments.backbone,
        profile_id=expected.profile_id,
        observation_schema=OBSERVATION_SCHEMA_VERSION,
        action_schema=ACTION_SCHEMA_VERSION,
        reward_schema=REWARD_SCHEMA_VERSION,
        source_revision=source_revision(),
    )
    resolved = {
        "backbone": arguments.backbone,
        "budget_decisions": arguments.budget_decisions,
        "speed": arguments.speed,
        "seed": arguments.seed,
        "batch_size": arguments.batch_size,
        "gradient_steps_per_decision": arguments.gradient_steps_per_decision,
        "sequence_length": arguments.sequence_length,
        "burn_in": arguments.burn_in,
        "slice_game_ms": arguments.slice_ms,
        "device": str(device),
    }
    write_manifest(run_dir / "manifest.json", {"run_id": run_id, **resolved})

    def run_evaluation() -> EvaluationReport:
        # Exploration-free, never written to replay; the evaluator enforces both.
        report = evaluate(
            environment,
            backbone,
            episodes=arguments.evaluation_episodes,
            profile_id=expected.profile_id,
            model_version=backbone.model_version,
        )
        print(report.summary_line(), flush=True)
        return report

    def write_checkpoint(report: TrainingProgressReport) -> None:
        save(
            Checkpoint(
                identity=identity,
                progress=TrainingProgress(
                    optimisation_steps=report.optimisation_steps,
                    environment_decisions=report.decisions,
                    episodes=report.episodes,
                    epsilon=config.epsilon(report.decisions),
                    importance_beta=config.beta(report.decisions),
                ),
                backbone_state=backbone.state_dict(),
                resolved_config=resolved,
                replay_provenance={**replay.snapshot(), "restored": False},
            ),
            run_dir / "checkpoints" / "latest.pt",
        )

    training = TrainingRun(
        actor=actor,
        replay=replay,
        backbone=backbone,
        config=config,
        evaluate=run_evaluation,
        checkpoint=write_checkpoint,
        on_episode=lambda report: print(
            f"episode {report.episodes} decisions {report.decisions}/"
            f"{config.budget_decisions} steps {report.optimisation_steps}",
            flush=True,
        ),
    )

    try:
        report = training.run()
    finally:
        adapter.release()
        client.close()

    write_checkpoint(report)
    summary = {
        "run_id": run_id,
        "resolved_config": resolved,
        "decisions": report.decisions,
        "episodes": report.episodes,
        "valid_episodes": report.valid_episodes,
        "optimisation_steps": report.optimisation_steps,
        "sequences_accepted": report.sequences_accepted,
        "wall_seconds": report.wall_seconds,
        "final_waves": report.final_waves,
        "evaluations": [to_record(item) for item in report.evaluations],
        "replay": replay.snapshot(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({key: summary[key] for key in
                      ("run_id", "decisions", "episodes", "optimisation_steps")}, indent=2))
    print(f"run directory: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
