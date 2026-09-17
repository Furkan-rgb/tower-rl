#!/usr/bin/env python3
"""Compare several arms on the clone, interleaved, reporting intervals.

Private device runner for the instrumented-training profile. Arms take turns in
small blocks rather than running one after another, because running arm A for an
hour and then arm B confounds the arm with whatever drifted in between. Results
are reported as bootstrap intervals rather than verdicts.

    TOWER_BRIDGE_BUILD_DIR=... uv run python scripts/compare_arms.py \\
        --arm scripted@64 --arm scripted@16 --episodes 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_episodes import POLICIES, compatibility  # noqa: E402

from tower_rl.application.actor import Actor, ActorConfig  # noqa: E402
from tower_rl.application.comparison import (  # noqa: E402
    compare,
    interleave_schedule,
    required_episodes,
)
from tower_rl.application.evaluator import WaveDistribution  # noqa: E402
from tower_rl.application.run_environment import (  # noqa: E402
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.domain.run_state import RunStateBuilder  # noqa: E402
from tower_rl.infrastructure.adb_device import AdbDevice  # noqa: E402
from tower_rl.infrastructure.instrumented_bridge import InstrumentedBridgeClient  # noqa: E402
from tower_rl.infrastructure.instrumented_run_adapter import InstrumentedRunAdapter  # noqa: E402


@dataclass(frozen=True)
class Arm:
    """One configuration under comparison."""

    name: str
    policy: str
    speed: float

    @classmethod
    def parse(cls, text: str) -> Arm:
        """Accept `policy@speed`, for example `scripted@64`."""
        policy, _, speed = text.partition("@")
        if policy not in POLICIES:
            raise SystemExit(f"unknown policy {policy!r}; choose from {sorted(POLICIES)}")
        if not speed:
            raise SystemExit(f"arm {text!r} must name a speed, for example scripted@64")
        return cls(name=text, policy=policy, speed=float(speed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="policy@speed")
    parser.add_argument("--episodes", type=int, default=25, help="episodes per arm")
    parser.add_argument("--block", type=int, default=5, help="episodes before switching arm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--serial", default="emulator-5556")
    parser.add_argument("--port", type=int, default=47652)
    parser.add_argument("--slice-ms", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("/tmp/tower-rl-comparison.json"))
    arguments = parser.parse_args()

    if arguments.serial == "emulator-5554":
        raise SystemExit("refusing to run against the canonical evaluation AVD")

    arms = {arm.name: arm for arm in (Arm.parse(text) for text in arguments.arm)}
    if len(arms) < 2:
        raise SystemExit("a comparison needs at least two distinct arms")

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
    adapter = InstrumentedRunAdapter(client=client, device=AdbDevice(arguments.serial))
    environment = InstrumentedRunEnvironment(
        port=adapter,
        builder=RunStateBuilder(profile_id=expected.profile_id),
        cadence=CadenceConfig(
            slice_game_ms=arguments.slice_ms,
            max_quiet_game_ms=arguments.slice_ms * 8,
            max_episode_wall_seconds=600.0,
        ),
    )
    actors = {
        name: Actor(
            environment=environment,
            policy=POLICIES[arm.policy](),
            config=ActorConfig(actor_id=f"{arguments.serial}:{name}"),
        )
        for name, arm in arms.items()
    }

    schedule = interleave_schedule(
        tuple(arms), arguments.episodes, block=arguments.block, seed=arguments.seed
    )
    waves: dict[str, list[float]] = defaultdict(list)
    invalid: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    started = time.monotonic()

    try:
        for index, name in enumerate(schedule, start=1):
            # The adapter applies the requested speed when the episode begins, so
            # setting it here is what makes an arm switch actually take effect.
            adapter.requested_speed = arms[name].speed
            summary = actors[name].run_episode().summary
            if summary.valid:
                waves[name].append(summary.final_wave)
            else:
                for reason in summary.termination_detail or (summary.termination.value,):
                    invalid[name][reason] += 1
            if index % 10 == 0:
                print(f"{index}/{len(schedule)} episodes", flush=True)
    finally:
        adapter.release()
        client.close()

    scored = {name: values for name, values in waves.items() if len(values) >= 2}
    report: dict[str, object] = {
        "profile_id": expected.profile_id,
        "episodes_requested_per_arm": arguments.episodes,
        "block": arguments.block,
        "seed": arguments.seed,
        "wall_seconds": round(time.monotonic() - started, 1),
        "arms": {
            name: {
                "policy": arms[name].policy,
                "speed": arms[name].speed,
                "valid_episodes": len(waves[name]),
                "invalid_detail": dict(invalid[name]),
                **_distribution(waves[name]),
            }
            for name in arms
        },
    }
    if len(scored) >= 2:
        differences = compare(scored, seed=arguments.seed)
        report["differences"] = [
            {
                "left": item.left,
                "right": item.right,
                "difference": round(item.difference, 3),
                "interval": [round(item.low, 3), round(item.high, 3)],
                "effect_size": round(item.effect_size, 3),
                "separated": item.separated,
            }
            for item in differences
        ]
        for item in differences:
            print(item.describe(), flush=True)
        # State what this sample could not have detected, not only what it did.
        spread = max(_spread(values) for values in scored.values())
        report["detectable_difference_at_this_n"] = {
            "standard_deviation": round(spread, 3),
            "episodes_for_one_wave": required_episodes(spread, 1.0) if spread > 0 else None,
        }

    arguments.output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0


def _distribution(values: list[float]) -> dict[str, object]:
    if len(values) < 1:
        return {"mean_final_wave": None}
    spread = WaveDistribution.of([int(value) for value in values])
    return {
        "mean_final_wave": round(spread.mean, 3),
        "median_final_wave": spread.median,
        "stdev_final_wave": round(spread.stdev, 3) if spread.stdev == spread.stdev else None,
        "minimum_final_wave": spread.minimum,
        "maximum_final_wave": spread.maximum,
    }


def _spread(values: list[float]) -> float:
    spread = WaveDistribution.of([int(value) for value in values]).stdev
    return spread if spread == spread else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
