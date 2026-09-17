#!/usr/bin/env python3
"""Run N episodes of one policy against the private instrumented clone.

Private device runner for the instrumented-training profile. It wires the real
bridge adapter to the environment and evaluator and writes its report outside the
repository. It never touches the canonical evaluation AVD, and it never taps:
since the round boundary moved into the bridge, nothing in this path reads a
pixel or touches the screen.

    TOWER_BRIDGE_BUILD_DIR=... ./scripts/run_episodes.py --episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tower_rl.application.actor import ActorConfig  # noqa: E402
from tower_rl.application.evaluator import evaluate, to_record  # noqa: E402
from tower_rl.application.policies import (  # noqa: E402
    CheapestFirstPolicy,
    RandomPolicy,
    WaitOnlyPolicy,
)
from tower_rl.application.run_environment import (  # noqa: E402
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.domain.run_state import RunStateBuilder  # noqa: E402
from tower_rl.infrastructure.instrumented_bridge import (  # noqa: E402
    BridgeCompatibility,
    InstrumentedBridgeClient,
)
from tower_rl.infrastructure.instrumented_run_adapter import (  # noqa: E402
    InstrumentedRunAdapter,
)

POLICIES = {
    "scripted": CheapestFirstPolicy,
    "random": RandomPolicy,
    "wait": WaitOnlyPolicy,
}


def compatibility(build_dir: Path) -> BridgeCompatibility:
    """Read the private build's configured identity; never hard-coded here."""
    cache = {
        key: value
        for line in (build_dir / "CMakeCache.txt").read_text().splitlines()
        if ":STRING=" in line
        for key, value in [line.split(":STRING=", 1)]
    }
    return BridgeCompatibility(
        package_version=cache["TOWER_BRIDGE_PACKAGE_VERSION"],
        package_version_code=int(cache["TOWER_BRIDGE_PACKAGE_VERSION_CODE"]),
        official_signer_sha256=cache["TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256"],
        original_libunity_sha256=cache["TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256"],
        libil2cpp_sha256=cache["TOWER_BRIDGE_LIBIL2CPP_SHA256"],
        unity_version=cache["TOWER_BRIDGE_UNITY_VERSION"],
        il2cpp_metadata_version=int(cache["TOWER_BRIDGE_METADATA_VERSION"]),
        bridge_version=cache["TOWER_BRIDGE_VERSION"],
        profile_id=cache["TOWER_BRIDGE_PROFILE_ID"],
    )


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
    parser.add_argument("--policy", choices=sorted(POLICIES), default="scripted")
    parser.add_argument("--serial", default="emulator-5556")
    parser.add_argument("--port", type=int, default=47652)
    add_cadence_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("/tmp/tower-rl-episodes.json"))
    arguments = parser.parse_args()

    if arguments.serial == "emulator-5554":
        raise SystemExit("refusing to run against the canonical evaluation AVD")

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
            POLICIES[arguments.policy](),
            episodes=arguments.episodes,
            profile_id=expected.profile_id,
            actor_config=ActorConfig(actor_id=f"{arguments.serial}:{arguments.policy}"),
        )
    finally:
        adapter.release()
        client.close()

    record = to_record(report)
    record["frame_game_ms"] = arguments.frame_game_ms
    record["max_quiet_game_ms"] = arguments.max_quiet_game_ms
    record["wall_seconds"] = round(time.monotonic() - started, 1)
    record["episodes_per_hour"] = (
        round(report.valid_episodes / (time.monotonic() - started) * 3600, 1)
    )
    arguments.output.write_text(json.dumps(record, indent=2))
    print(report.summary_line(), flush=True)
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
