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
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_episodes import POLICIES, add_cadence_arguments  # noqa: E402

from tower_rl.environment.run_environment import BRIDGE_EVENT_DIVERGENCE  # noqa: E402
from tower_rl.simulation.bridge import ActorFailure, deploy_bridge  # noqa: E402
from tower_rl.simulation.bring_up import (  # noqa: E402
    bring_up,
    require_game_activity,
    require_offline,
)
from tower_rl.simulation.fleet import (  # noqa: E402
    prepare_pinned_snapshot,
    stagger_bring_up,
    tear_down_instance,
)
from tower_rl.simulation.frame_rate import raise_frame_rate  # noqa: E402
from tower_rl.simulation.instance import (  # noqa: E402
    GUEST_FRAME_RATE_HZ,
    MAX_GUEST_FRAME_RATE_HZ,
    CloneInstance,
)

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
                    # Which arm this actor collected for, when a fleet holds two.
                    "frame_rate_hz": record.get("frame_rate_hz"),
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


def frame_rates(text: str, actors: int) -> list[int]:
    """The guest rate each instance runs at, one per instance index.

    One value is every instance's rate. A comma-separated list is one rate per
    index, which is what lets a behavioural comparison of two rates be a single
    fleet: both levers are per emulator, so instances at different rates run in
    the same window against the same host, and the arms are interleaved rather
    than sequential. A list of the wrong length is refused by name — a
    comparison that silently ran six of its seven instances at one rate would
    look exactly like one that ran seven.
    """
    values = [part.strip() for part in text.split(",")]
    try:
        rates = [int(value) for value in values]
    except ValueError:
        raise SystemExit(f"--frame-rate-hz must be whole numbers of Hz, not {text!r}") from None
    if len(rates) == 1:
        rates = rates * actors
    if len(rates) != actors:
        raise SystemExit(
            f"--frame-rate-hz lists {len(values)} rates for {actors} actors; "
            "give one rate for the whole fleet or exactly one per instance"
        )
    for rate in rates:
        # The same measured ceiling the module constant is held to: above it the
        # guest reports a rate it is not delivering, so the confirmation would
        # pass on an instance collecting at some other rate entirely.
        if not 1 <= rate <= MAX_GUEST_FRAME_RATE_HZ:
            raise SystemExit(
                f"--frame-rate-hz {rate} is outside 1..{MAX_GUEST_FRAME_RATE_HZ}; "
                "no measured fps supports a guest rate above that"
            )
    return rates


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

    Nothing else waits for the fleet. Every instance boots at the stock 60 Hz and
    this actor raises its *own* instance the moment its own bring-up returns,
    then starts collecting while its peers are still booting. The fleet-wide
    rendezvous that used to sit here came from the reading that a raised peer
    killed a booting one, which `M1B-E043` refuted, and it was not free: it
    parked a ready instance idle for as long as the remaining boots took — up to
    N x 360s — and the guest's Google Play spent that window installing the
    update it had downloaded, over a game no step was watching. Two 7-actor runs
    lost actors exactly there.
    """
    # This instance's own rate, not the fleet's: a comparison of two rates runs
    # them side by side in one fleet, so the rate belongs to the index.
    frame_rate_hz = arguments.frame_rates[instance.index]
    try:
        bring_up(
            instance,
            arguments.renderer,
            deploy=deploy_bridge,
            read_only=True,
            cores=arguments.cores,
            force_cold=arguments.cold,
            frame_rate_hz=frame_rate_hz,
        )
    finally:
        signal_ready()

    # By interface, per instance, immediately before anything is measured:
    # `run_episodes.py` makes no offline check of its own, so this is the last
    # reading that can still precede an episode. `cold_bring_up` has already
    # verified the same thing at the end of bring-up.
    require_offline(instance)
    # Nothing has watched the game since bring-up returned, and a game with no
    # activity has no surface, so SurfaceFlinger publishes no applied frame rate
    # for its uid at all — which is how this arrived twice on device, as
    # `applied frame rate absent` from the raise below. It is reported here by
    # name instead: `M1B-E049` measured that a relaunch this side of the network
    # cut sits at the OFFLINE modal, so the instance is lost, not recoverable.
    require_game_activity(instance)
    raise_frame_rate(instance, frame_rate_hz)

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
    # The arm this actor belongs to, written back into the durable record: the
    # analysis groups episodes by the rate they were collected at, and a record
    # that does not carry its own rate can only be attributed by the directory
    # it happens to sit in.
    record["frame_rate_hz"] = frame_rate_hz
    output.write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=2, help="concurrent instances to run")
    parser.add_argument("--episodes", type=int, default=20, help="episodes per actor")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="scripted")
    parser.add_argument("--renderer", default="lavapipe")
    parser.add_argument(
        "--frame-rate-hz",
        default=str(GUEST_FRAME_RATE_HZ),
        help="guest frame rate for the whole fleet, or one rate per instance "
        "index as a comma-separated list (60,120,60,...); default "
        f"{GUEST_FRAME_RATE_HZ}, the fleet operating rate",
    )
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
    arguments.frame_rates = frame_rates(arguments.frame_rate_hz, arguments.actors)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    instances = [CloneInstance(index=index) for index in range(arguments.actors)]
    if not arguments.cold:
        prepare_pinned_snapshot(arguments.renderer, arguments.cores)

    started = time.monotonic()
    outcomes = run_fleet(
        instances,
        stagger_bring_up(
            instances,
            lambda instance, signal_ready: collect_episodes(
                instance, arguments, signal_ready=signal_ready
            ),
        ),
        tear_down_instance,
    )
    report = aggregate(outcomes, time.monotonic() - started)
    report["policy"] = arguments.policy
    report["episodes_per_actor"] = arguments.episodes
    report["frame_game_ms"] = arguments.frame_game_ms
    report["cores_per_instance"] = arguments.cores
    # Per instance index, because one fleet may hold two arms; each actor's
    # entry and each actor's own record carry the rate it collected at.
    report["frame_rates_hz"] = arguments.frame_rates

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
