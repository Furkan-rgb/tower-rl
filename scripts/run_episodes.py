#!/usr/bin/env python3
"""Run N episodes of one policy against the private instrumented clone.

Private device runner for the instrumented-training profile. It wires the real
bridge adapter to the environment and evaluator and writes its report outside the
repository. It never touches the canonical evaluation AVD, and it never taps:
since the round boundary moved into the bridge, nothing in this path reads a
pixel or touches the screen.

    TOWER_BRIDGE_BUILD_DIR=... ./scripts/run_episodes.py --episodes 50

`--policy` names the arm: one of the non-learned floors, or `checkpoint:<path>`
for a checkpoint a training run left behind, which is rebuilt into the backbone
that wrote it and played greedily. Every record says which it was.

    TOWER_BRIDGE_BUILD_DIR=... ./scripts/run_episodes.py --episodes 50 \\
        --policy checkpoint:~/.local/state/tower-rl/runs/.../checkpoint-0100000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# One torch thread per actor process. A fleet runs N of these at once against N
# emulators on one host, and torch sizes its intra-op pool from the core count,
# so N processes each claiming every core thrash instead of collecting. Acting
# is a single-sample forward pass through a small network and has nothing to
# gain from a pool anyway. Set before torch is first imported, because OpenMP
# and MKL read them when their runtimes initialise; `tests/conftest.py` does the
# same thing for the same reason.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

from tower_rl.environment.run_environment import (  # noqa: E402
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.environment.run_state import RunStateBuilder  # noqa: E402
from tower_rl.learning.actor import ActorConfig  # noqa: E402
from tower_rl.learning.checkpoint import CheckpointError, identity_hash  # noqa: E402
from tower_rl.learning.evaluator import EvaluationReport, evaluate, to_record  # noqa: E402
from tower_rl.learning.policies import (  # noqa: E402
    CheapestFirstPolicy,
    Policy,
    RandomPolicy,
    WaitOnlyPolicy,
    checkpoint_policy,
)
from tower_rl.simulation.bridge import compatibility  # noqa: E402
from tower_rl.simulation.instrumented_bridge import (  # noqa: E402
    InstrumentedBridgeClient,
)
from tower_rl.simulation.instrumented_run_adapter import (  # noqa: E402
    InstrumentedRunAdapter,
)

torch.set_num_threads(1)

POLICIES = {
    "scripted": CheapestFirstPolicy,
    "random": RandomPolicy,
    "wait": WaitOnlyPolicy,
}

#: How a checkpoint is named as an arm, beside the names above.
CHECKPOINT_SELECTOR = "checkpoint:"


def policy_from(selector: str) -> tuple[Policy, dict[str, object]]:
    """The arm this run plays, and the identity every record of it carries.

    A selector is one of the non-learned floors by name, or `checkpoint:<path>`.
    The two are the same kind of thing to everything downstream - the actor, the
    evaluator, the per-episode records - which is the point: a checkpoint is
    measured by exactly the protocol its floors are.

    The identity travels with the record rather than being inferred from the
    directory a file happens to sit in. `run_actors.py` writes one file per
    actor and a selection reads many of those directories at once, so a record
    that could not name its own arm could only be attributed by convention.
    """
    if selector in POLICIES:
        return POLICIES[selector](), {"name": selector}
    if not selector.startswith(CHECKPOINT_SELECTOR):
        raise SystemExit(
            f"unknown policy {selector!r}; choose from {sorted(POLICIES)} "
            f"or {CHECKPOINT_SELECTOR}<path>"
        )
    path = Path(selector[len(CHECKPOINT_SELECTOR) :]).expanduser()
    try:
        policy, identity = checkpoint_policy(path)
    except (CheckpointError, ValueError) as failure:
        raise SystemExit(f"cannot play {path} as an arm: {failure}") from failure
    return policy, {
        # Named for the file, which is named for the decisions behind it, so an
        # actor id and a report line say which checkpoint of the run this is.
        "name": path.stem,
        "checkpoint_path": str(path.resolve()),
        # The path is where the file is today; this is what it holds.
        "checkpoint_identity": identity_hash(identity),
        "run_id": identity.run_id,
    }


def actor_record(
    report: EvaluationReport,
    identity: Mapping[str, object],
    *,
    frame_game_ms: float,
    max_quiet_game_ms: int,
    wall_seconds: float,
) -> dict[str, Any]:
    """One actor's durable record: the episodes, and which arm produced them.

    `run_actors.py` writes one of these per instance and a later selection reads
    whole directories of them, so the arm has to be inside the record. The
    evaluator's own `policy` field names the class that acted, which is the same
    class for every checkpoint of every run.
    """
    record = to_record(report)
    record["policy_identity"] = dict(identity)
    record["frame_game_ms"] = frame_game_ms
    record["max_quiet_game_ms"] = max_quiet_game_ms
    record["wall_seconds"] = round(wall_seconds, 1)
    record["episodes_per_hour"] = (
        round(report.valid_episodes / wall_seconds * 3600, 1) if wall_seconds > 0 else 0.0
    )
    return record


def add_cadence_arguments(parser: argparse.ArgumentParser) -> None:
    """The cadence is game time throughout; the only wall clock is a hang deadline.

    There is no speed argument. The game's own multiplier is pinned at 1x inside
    the adapter, and speed comes from the bridge stepping frames, so a speed knob
    here could only reintroduce the coarsening it was removed for (M1B-E012).
    """
    parser.add_argument(
        "--frame-game-ms",
        type=float,
        default=1000.0 / 60.0,
        help="game time one rendered frame is worth; the floor on decision granularity",
    )
    parser.add_argument(
        "--max-quiet-game-ms",
        type=int,
        default=2000,
        help="game time one advance may spend before returning a decision anyway",
    )
    parser.add_argument(
        "--max-episode-wall-seconds",
        type=float,
        default=600.0,
        help="hang deadline in wall seconds; wall time is not bounded by game time",
    )


def cadence_from(arguments: argparse.Namespace) -> CadenceConfig:
    return CadenceConfig(
        frame_game_ms=arguments.frame_game_ms,
        max_quiet_game_ms=arguments.max_quiet_game_ms,
        max_episode_wall_seconds=arguments.max_episode_wall_seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--policy",
        default="scripted",
        help=(
            f"one of {sorted(POLICIES)}, or {CHECKPOINT_SELECTOR}<path> to play a "
            "checkpoint a training run left behind"
        ),
    )
    parser.add_argument("--serial", default="emulator-5556")
    parser.add_argument("--port", type=int, default=47652)
    add_cadence_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("/tmp/tower-rl-episodes.json"))
    arguments = parser.parse_args()

    if arguments.serial == "emulator-5554":
        raise SystemExit("refusing to run against the canonical evaluation AVD")

    # Before the device is touched: a checkpoint that cannot be rebuilt should
    # fail now, not after an emulator has been brought up for it.
    policy, identity = policy_from(arguments.policy)

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
    handshake = client.connect()
    print(
        f"bridge {handshake.bridge_version} profile {handshake.compatibility.profile_id} "
        f"speed {handshake.game_speed}",
        flush=True,
    )

    adapter = InstrumentedRunAdapter(client=client)
    environment = InstrumentedRunEnvironment(
        port=adapter,
        builder=RunStateBuilder(profile_id=expected.profile_id),
        cadence=cadence_from(arguments),
    )

    started = time.monotonic()
    try:
        report = evaluate(
            environment,
            policy,
            episodes=arguments.episodes,
            profile_id=expected.profile_id,
            actor_config=ActorConfig(actor_id=f"{arguments.serial}:{identity['name']}"),
        )
    finally:
        adapter.release()
        client.close()

    record = actor_record(
        report,
        identity,
        frame_game_ms=arguments.frame_game_ms,
        max_quiet_game_ms=arguments.max_quiet_game_ms,
        wall_seconds=time.monotonic() - started,
    )
    arguments.output.write_text(json.dumps(record, indent=2))
    print(report.summary_line(), flush=True)
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
