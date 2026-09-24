#!/usr/bin/env python3
"""Run N episodes of one policy against the private instrumented clone.

Private device runner for the instrumented-training profile. It wires the real
bridge adapter to the environment and evaluator and writes its report outside the
repository. It never touches the canonical evaluation AVD, and it never taps:
since the round boundary moved into the bridge, nothing in this path reads a
pixel or touches the screen.

    ./scripts/run_episodes.py --episodes 50

`--policy` names the arm: one of the non-learned floors, or `checkpoint:<path>`
for a checkpoint a training run left behind, which is rebuilt into the backbone
that wrote it and played greedily. Every record says which it was.

    ./scripts/run_episodes.py --episodes 50 \\
        --policy checkpoint:state/runs/.../checkpoint-d0015000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
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

from tower_rl.console_timestamp import timestamped_print as print  # noqa: E402
from tower_rl.environment.project_state import state_directory  # noqa: E402
from tower_rl.environment.run_environment import (  # noqa: E402
    CadenceConfig,
    DecisionCadence,
    InstrumentedRunEnvironment,
    UpgradeAvailability,
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
from tower_rl.simulation.bridge import bridge_build_directory, compatibility  # noqa: E402
from tower_rl.simulation.instrumented_bridge import (  # noqa: E402
    BridgeCompatibility,
    InstrumentedBridgeClient,
    UpgradeSlotLabel,
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

def policy_from(
    selector: str,
    *,
    decision_cadence: DecisionCadence,
    upgrade_availability: UpgradeAvailability,
) -> tuple[Policy, dict[str, object]]:
    """The arm this run plays, and the identity every record of it carries.

    A selector is one of the non-learned floors by name, or `checkpoint:<path>`.
    The two are the same kind of thing to everything downstream - the actor, the
    evaluator, the per-episode records - which is the point: a checkpoint is
    measured by exactly the protocol its floors are.

    Which is also why the protocol this session will run is passed in: a floor
    plays whatever it is given, but a checkpoint learned one cadence and one set
    of purchasable rows, and playing it under the others measures something the
    run it came from never posed. `checkpoint_policy` refuses that by name.

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
        policy, identity = checkpoint_policy(
            path,
            decision_cadence=str(decision_cadence),
            upgrade_availability=str(upgrade_availability),
        )
    except (CheckpointError, ValueError) as failure:
        raise SystemExit(f"cannot play {path} as an arm: {failure}") from failure
    return policy, {
        # Named for the file, which is named for the game time behind it, so an
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
    decision_cadence: DecisionCadence,
    upgrade_availability: UpgradeAvailability,
    wall_seconds: float,
    labels: Sequence[UpgradeSlotLabel] = (),
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
    # Which protocol these episodes were played under. Decisions mean different
    # things under the two, so a record that could not say which cadence
    # produced it could not be compared with anything (ADR 0009).
    record["decision_cadence"] = str(decision_cadence)
    # Which upgrade rows these episodes could buy from. A record collected on
    # the image's six rows and one collected on every real row are measurements
    # of two different decision problems (ADR 0011).
    record["upgrade_availability"] = str(upgrade_availability)
    record["wall_seconds"] = round(wall_seconds, 1)
    # What the game calls each slot the actions address, so the human reading
    # this record afterwards can tell what `attack:3` was. Never an input: the
    # policy addresses a slot by index, and a renamed row must not change what a
    # checkpoint means.
    record["upgrade_rows"] = [
        {"family": label.family, "index": label.index, "name": label.name,
         "description": label.description}
        for label in labels
        if label.name
    ]
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
        # 100 ms, adopted in M1B-E018 and shown equivalent and 2.6-3.0x faster under M2 in M2-S001.
        default=100.0,
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
    parser.add_argument(
        "--decision-cadence",
        choices=[cadence.value for cadence in DecisionCadence],
        default=DecisionCadence.CHOICE_POINTS.value,
        help="which cadence stops the policy is asked about: choice-points is "
        "the contract, every-slice reproduces run 1's protocol (ADR 0009)",
    )


def add_upgrade_availability_argument(parser: argparse.ArgumentParser) -> None:
    """Which upgrade rows the run is played with (ADR 0011).

    Separate from the cadence arguments because it is a separate thing: the
    cadence says when the policy is asked, this says what it may buy. `image` is
    the default and is what every baseline so far was measured under.
    """
    parser.add_argument(
        "--upgrade-availability",
        choices=[availability.value for availability in UpgradeAvailability],
        default=UpgradeAvailability.IMAGE.value,
        help="which upgrade rows are purchasable: image is what the profile "
        "image offers, all reopens every real row at each round start (ADR 0011)",
    )


def upgrade_availability_from(arguments: argparse.Namespace) -> UpgradeAvailability:
    """The availability this invocation collects or plays under."""
    return UpgradeAvailability(arguments.upgrade_availability)


def cadence_from(arguments: argparse.Namespace) -> CadenceConfig:
    return CadenceConfig(
        frame_game_ms=arguments.frame_game_ms,
        max_quiet_game_ms=arguments.max_quiet_game_ms,
        max_episode_wall_seconds=arguments.max_episode_wall_seconds,
    )


def decision_cadence_from(arguments: argparse.Namespace) -> DecisionCadence:
    """The named cadence policy this invocation collects or plays under."""
    return DecisionCadence(arguments.decision_cadence)


def open_environment(
    port: int, expected: BridgeCompatibility, arguments: argparse.Namespace
) -> tuple[InstrumentedBridgeClient, InstrumentedRunAdapter, InstrumentedRunEnvironment]:
    """Connect the bridge and build the adapter and environment on top of it.

    The wiring every collection entry point (`run_episodes.py`, `train.py`,
    `spectate.py`) needs identically: the same client timeouts, then the
    adapter, then the environment built from this invocation's cadence and
    upgrade-availability arguments. The caller keeps its own connect-time
    side effects (printing the handshake, reading slot labels, wrapping the
    result) and its own `release()`/`close()` in `finally`.
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
    environment = InstrumentedRunEnvironment(
        port=adapter,
        builder=RunStateBuilder(profile_id=expected.profile_id),
        cadence=cadence_from(arguments),
        decision_cadence=decision_cadence_from(arguments),
        upgrade_availability=upgrade_availability_from(arguments),
    )
    return client, adapter, environment


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
    add_upgrade_availability_argument(parser)
    parser.add_argument(
        "--output", type=Path, default=state_directory() / "records" / "episodes.json"
    )
    arguments = parser.parse_args()

    if arguments.serial == "emulator-5554":
        raise SystemExit("refusing to run against the canonical evaluation AVD")

    # Before the device is touched: a checkpoint that cannot be rebuilt should
    # fail now, not after an emulator has been brought up for it.
    policy, identity = policy_from(
        arguments.policy,
        decision_cadence=decision_cadence_from(arguments),
        upgrade_availability=upgrade_availability_from(arguments),
    )
    expected = compatibility(bridge_build_directory())

    client, adapter, environment = open_environment(arguments.port, expected, arguments)
    handshake = client.handshake
    print(
        f"bridge {handshake.bridge_version} profile {handshake.compatibility.profile_id} "
        f"speed {handshake.game_speed}",
        flush=True,
    )
    # Before the first round: a command of the adapter's own initiative belongs
    # to the episode boundary, and these are constant for the build.
    labels = adapter.slot_labels()

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
        decision_cadence=decision_cadence_from(arguments),
        upgrade_availability=upgrade_availability_from(arguments),
        wall_seconds=time.monotonic() - started,
        labels=labels,
    )
    arguments.output.write_text(json.dumps(record, indent=2))
    print(report.summary_line(), flush=True)
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
