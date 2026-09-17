#!/usr/bin/env python3
"""Run N actors concurrently, one per clone instance, and report throughput.

Throughput is the bottleneck: one actor collects roughly 70-90 valid episodes an
hour, and every downstream budget — the comparison floor, equal-budget training
of several backbones — is priced in device hours. This runner exists to measure
what N actors buy, so N is chosen from a measurement rather than from optimism.

Each actor is a separate `run_episodes.py` process against its own emulator
instance, its own bridge deployment and its own forwarded host port, so the
actors are concurrent in the only sense that matters here: N games stepping at
once. This script owns their lifecycle and nothing else; the episode loop,
validity classification and per-episode records all stay in the single-actor
runner and the evaluator.

Instances share the one clone AVD through `-read-only` (see `clone_session.py`),
so N actors cost N overlays rather than N copies of a multi-gigabyte image.

Every actor comes up the same way `clone_session.py up` does: it restores the
snapshot pinned to the bridge this fleet deploys, so the fleet opens no network
window at all and every actor starts from identical account state. The snapshot
is prepared once, before the fleet, because a `-read-only` instance cannot save
one. `--cold` skips it and cold-starts every actor.

Measured on device: scaling is linear to at least 4 actors (fidelity intact at
every N; several `-read-only` instances do restore the one snapshot
concurrently with no corruption). The one defect the same run found is in
bring-up, not collection: four emulators cold-booting at the same instant
pushed host load to 10.71 and total CPU to 1,835%, and the last of the four
never left `main_unavailable` inside its cold-launch timeout while its peers
each reached home alone in 60-90s. Steady state uses only 563% of 3,200%
available CPU, so the contention is entirely in the simultaneous boot.
`stagger_bring_up` is the fix: each instance's bring-up begins only once the
previous instance has signalled ready, so boots do not pile up, while episode
collection afterwards is exactly as concurrent as before.

    TOWER_BRIDGE_BUILD_DIR=... uv run python scripts/run_actors.py \\
        --actors 2 --episodes 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clone_session import (  # noqa: E402
    CloneInstance,
    bridge_key,
    bring_up,
    keyed_snapshot_name,
    kill_emulator,
    require_offline,
    snapshot_exists,
)
from run_episodes import POLICIES, add_cadence_arguments  # noqa: E402

from tower_rl.application.run_environment import BRIDGE_EVENT_DIVERGENCE  # noqa: E402

#: A rejected command the bridge reports by name; the environment carries the
#: name through into the episode's termination detail.
STALE_OR_DUPLICATE = "stale_or_duplicate"

SCRIPTS = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ActorOutcome:
    """What one actor produced, including the reason it produced nothing."""

    index: int
    serial: str
    wall_seconds: float
    record: dict[str, Any] | None = None
    failure: str | None = None
    #: Teardown is reported separately: an actor that collected its episodes and
    #: then failed to clean up is a device-state problem, not a lost measurement.
    teardown_failure: str | None = None


def run_actor(
    instance: CloneInstance,
    collect: Callable[[CloneInstance], dict[str, Any]],
    tear_down: Callable[[CloneInstance], None],
) -> ActorOutcome:
    """Collect one actor's episodes, always tearing its instance down.

    This never raises. One emulator crashing, one bridge handshake failing or one
    episode hanging is that actor's outcome alone; the other actors keep running
    and the aggregate reports the failure rather than losing it.
    """
    started = time.monotonic()
    record: dict[str, Any] | None = None
    failure: str | None = None
    teardown_failure: str | None = None
    try:
        record = collect(instance)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        try:
            tear_down(instance)
        except Exception as error:
            teardown_failure = f"{type(error).__name__}: {error}"
    return ActorOutcome(
        index=instance.index,
        serial=instance.serial,
        wall_seconds=round(time.monotonic() - started, 1),
        record=record,
        failure=failure,
        teardown_failure=teardown_failure,
    )


def run_fleet(
    instances: list[CloneInstance],
    collect: Callable[[CloneInstance], dict[str, Any]],
    tear_down: Callable[[CloneInstance], None],
) -> list[ActorOutcome]:
    """Run every actor at once and return their outcomes in instance order."""
    with ThreadPoolExecutor(max_workers=len(instances)) as pool:
        futures = [pool.submit(run_actor, instance, collect, tear_down) for instance in instances]
        return [future.result() for future in futures]


def health_counters(record: dict[str, Any]) -> dict[str, int]:
    """The health counters already tracked per episode, summed for one actor.

    `advances_cut_short` and `episodes_not_started_fresh` are counted by the
    evaluator; the two bridge-level reasons are counted from the per-episode
    termination detail, which is where the environment records them.
    """
    episodes = record.get("episodes", ())
    detail = [text for episode in episodes for text in episode.get("termination_detail", ())]
    return {
        "bridge_event_divergence": sum(1 for text in detail if BRIDGE_EVENT_DIVERGENCE in text),
        "stale_or_duplicate": sum(1 for text in detail if STALE_OR_DUPLICATE in text),
        "advances_cut_short": int(record.get("advances_cut_short", 0)),
        "episodes_not_started_fresh": int(record.get("episodes_not_started_fresh", 0)),
    }


def aggregate(outcomes: list[ActorOutcome], wall_seconds: float) -> dict[str, Any]:
    """Aggregate valid episodes per hour, and what each actor contributed.

    The rate is over the fleet's wall clock, not the sum of the actors' clocks:
    N actors running for an hour bought one hour of device time, however long
    each of them individually stayed alive.
    """
    actors: list[dict[str, Any]] = []
    health = {
        "bridge_event_divergence": 0,
        "stale_or_duplicate": 0,
        "advances_cut_short": 0,
        "episodes_not_started_fresh": 0,
    }
    valid = 0
    invalid = 0
    for outcome in outcomes:
        entry: dict[str, Any] = {
            "index": outcome.index,
            "serial": outcome.serial,
            "wall_seconds": outcome.wall_seconds,
            "failure": outcome.failure,
            "teardown_failure": outcome.teardown_failure,
        }
        record = outcome.record
        if record is not None:
            counters = health_counters(record)
            for name, count in counters.items():
                health[name] += count
            valid += int(record["valid_episodes"])
            invalid += int(record["invalid_episodes"])
            entry.update(
                {
                    "valid_episodes": record["valid_episodes"],
                    "invalid_episodes": record["invalid_episodes"],
                    "invalid_rate": record["invalid_rate"],
                    "mean_final_wave": record["mean_final_wave"],
                    "valid_episodes_per_hour": (
                        round(int(record["valid_episodes"]) / outcome.wall_seconds * 3600, 1)
                        if outcome.wall_seconds > 0
                        else 0.0
                    ),
                    **counters,
                }
            )
        actors.append(entry)

    attempted = valid + invalid
    return {
        "actors_requested": len(outcomes),
        "actors_reporting": sum(1 for outcome in outcomes if outcome.record is not None),
        "actors_failed": sum(1 for outcome in outcomes if outcome.failure is not None),
        "wall_seconds": round(wall_seconds, 1),
        "valid_episodes": valid,
        "invalid_episodes": invalid,
        "invalid_rate": round(invalid / attempted, 4) if attempted else 0.0,
        "valid_episodes_per_hour": (
            round(valid / wall_seconds * 3600, 1) if wall_seconds > 0 else 0.0
        ),
        "health": health,
        "actors": actors,
    }


class ActorFailure(RuntimeError):
    """A step of one actor's lifecycle failed; only that actor is affected."""


def run_bridge(command: str, instance: CloneInstance) -> None:
    """Deploy or clean up the instrumented bridge on one instance.

    What the script prints is the device-safety evidence itself: the `libunity.so`
    hash it re-verified against the original, the package identity, the number of
    live mounts, and whether the bridge artifacts are gone. Cleanup that reports
    nothing is indistinguishable from cleanup that verified nothing, so the output
    is relayed to the operator's log rather than kept for an error path that a
    successful cleanup never takes. Every line is tagged, since N actors clean up
    at once and an unattributed identity report verifies no particular instance.
    """
    result = subprocess.run(
        [
            str(SCRIPTS / "instrumented_bridge.sh"),
            command,
            instance.serial,
            str(instance.bridge_host_port),
        ],
        capture_output=True,
        text=True,
    )
    for marker, text in (("", result.stdout), ("error: ", result.stderr)):
        for line in text.splitlines():
            print(f"{instance.serial} {command}: {marker}{line}", flush=True)
    if result.returncode != 0:
        raise ActorFailure(
            f"instrumented_bridge.sh {command} failed on {instance.serial}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def deploy_bridge(instance: CloneInstance) -> None:
    """The bridge deployment step of the cold path, tagged into the fleet's log."""
    run_bridge("deploy", instance)


def collect_episodes(
    instance: CloneInstance,
    arguments: argparse.Namespace,
    *,
    signal_ready: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Bring one instance up ready and offline, and run its episodes.

    Bring-up is the same decision every instance makes: restore the snapshot for
    the bridge this fleet deploys if the AVD holds one, and otherwise cold-start,
    deploy and relaunch through the one online window. Each actor's instance is
    `-read-only`, so it writes to its own overlay and saves no snapshot; the
    pinned one is prepared once, before the fleet, by `prepare_pinned_snapshot`.

    `signal_ready` is called once bring-up concludes, success or failure alike
    (see `stagger_bring_up`): a fleet run uses it to let the next actor's
    bring-up begin, and a failed bring-up must release that next actor just as
    surely as a successful one, or one dead actor would stall the rest of the
    fleet from ever starting.
    """
    try:
        bring_up(
            instance,
            arguments.renderer,
            deploy=deploy_bridge,
            read_only=True,
            cores=arguments.cores,
            force_cold=arguments.cold,
        )
        # By interface, per instance, immediately before anything is measured.
        require_offline(instance)
    finally:
        signal_ready()

    output = Path(arguments.output_directory) / f"{instance.serial}.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_episodes.py"),
            "--episodes", str(arguments.episodes),
            "--policy", arguments.policy,
            "--serial", instance.serial,
            "--port", str(instance.bridge_host_port),
            "--frame-game-ms", str(arguments.frame_game_ms),
            "--max-quiet-game-ms", str(arguments.max_quiet_game_ms),
            "--max-episode-wall-seconds", str(arguments.max_episode_wall_seconds),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ActorFailure(
            f"run_episodes.py failed on {instance.serial}: "
            f"{result.stderr.strip()[-500:] or result.stdout.strip()[-500:]}"
        )
    record: dict[str, Any] = json.loads(output.read_text())
    return record


#: A backstop against a readiness signal that never arrives, not the
#: sequencing mechanism itself. `collect_episodes` always signals in a
#: `finally`, and bring_up's own internal waits (`wait_for_boot`,
#: `wait_until_ready`) are already bounded at 300s, so in the ordinary case —
#: including a failed bring-up — this is never reached; it exists only so a
#: bug that skipped the signal could not stall the rest of the fleet forever.
BRING_UP_STAGGER_BACKSTOP = 360.0


def stagger_bring_up(
    instances: list[CloneInstance], arguments: argparse.Namespace
) -> Callable[[CloneInstance], dict[str, Any]]:
    """Sequence each instance's bring-up after the previous instance's readiness.

    This is the fix for the one failure a fleet of 4 hit on device: booting all
    four emulators at the same instant peaked host load at 10.71 and total CPU
    at 1,835%, and the last of the four never left `main_unavailable` within its
    cold-launch timeout while its peers each reached home alone in 60-90s.
    Steady-state collection uses only 563% of 3,200% available CPU, so the
    contention is entirely in the simultaneous boot, not in running — which is
    why only bring-up is gated here. Once an instance is up, its episode
    collection runs exactly as concurrently as it always has.

    Sequencing on readiness rather than a fixed sleep means instance i+1 starts
    its bring-up the moment instance i's bring-up actually concludes, not after
    a guessed duration — whether instance i succeeded or failed, since one dead
    actor must not block the rest of the fleet from starting.
    """
    gates = [threading.Event() for _ in instances]

    def collect(instance: CloneInstance) -> dict[str, Any]:
        if instance.index > 0:
            previous = gates[instance.index - 1]
            if not previous.wait(BRING_UP_STAGGER_BACKSTOP):
                print(
                    f"{instance.serial}: instance {instance.index - 1} never signalled "
                    f"ready within {BRING_UP_STAGGER_BACKSTOP:.0f}s; starting anyway",
                    flush=True,
                )
        return collect_episodes(instance, arguments, signal_ready=gates[instance.index].set)

    return collect


def tear_down_instance(instance: CloneInstance) -> None:
    """Remove the bridge and stop the emulator, whatever the actor did.

    Leaving an emulator running is a device-safety failure, so killing it happens
    even when the bridge refuses to clean up, and the bridge's failure is what is
    reported afterwards.
    """
    bridge_error: Exception | None = None
    try:
        run_bridge("cleanup", instance)
    except Exception as error:
        bridge_error = error
    finally:
        kill_emulator(instance)
    if bridge_error is not None:
        raise bridge_error


def prepare_pinned_snapshot(renderer: str, cores: int) -> str:
    """Make sure the fleet has a snapshot to restore, by taking the cold path once.

    A `-read-only` actor cannot save a snapshot, so without this every actor
    would cold-start and the fleet would open N network windows instead of none.
    The preparation runs alone on index 0, writable, before any actor starts —
    two instances must not write the one AVD at the same time — and its instance
    is torn down again whether it succeeded or not.
    """
    instance = CloneInstance()
    name = keyed_snapshot_name(bridge_key())
    if snapshot_exists(instance, name):
        return name
    print(f"no snapshot for the current bridge; preparing {name} once", flush=True)
    try:
        bring_up(instance, renderer, deploy=deploy_bridge, cores=cores)
    finally:
        tear_down_instance(instance)
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=2, help="concurrent instances to run")
    parser.add_argument("--episodes", type=int, default=20, help="episodes per actor")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="scripted")
    parser.add_argument("--renderer", default="lavapipe")
    parser.add_argument("--cores", type=int, default=4, help="emulator cores per instance")
    parser.add_argument(
        "--cold",
        action="store_true",
        help="ignore the pinned snapshot and cold-start every actor",
    )
    add_cadence_arguments(parser)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("/tmp/tower-rl-actors"),
        help="where each actor's own episode record is written",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/tower-rl-actors.json"))
    arguments = parser.parse_args()

    if arguments.actors < 1:
        raise SystemExit("a fleet needs at least one actor")
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    instances = [CloneInstance(index=index) for index in range(arguments.actors)]
    if not arguments.cold:
        prepare_pinned_snapshot(arguments.renderer, arguments.cores)

    started = time.monotonic()
    outcomes = run_fleet(
        instances,
        stagger_bring_up(instances, arguments),
        tear_down_instance,
    )
    report = aggregate(outcomes, time.monotonic() - started)
    report["policy"] = arguments.policy
    report["episodes_per_actor"] = arguments.episodes
    report["frame_game_ms"] = arguments.frame_game_ms
    report["cores_per_instance"] = arguments.cores

    arguments.output.write_text(json.dumps(report, indent=2))
    for actor in report["actors"]:
        print(
            f"{actor['serial']}: "
            + (f"failed: {actor['failure']}" if actor["failure"] else
               f"{actor['valid_episodes']} valid, {actor['valid_episodes_per_hour']}/hour"),
            flush=True,
        )
    print(
        f"aggregate: {report['valid_episodes']} valid episodes from "
        f"{report['actors_reporting']}/{report['actors_requested']} actors, "
        f"{report['valid_episodes_per_hour']} valid episodes/hour",
        flush=True,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["actors_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
