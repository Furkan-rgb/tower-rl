#!/usr/bin/env python3
"""Bring a disposable clone instance up offline, and snapshot it already running.

The game cannot cold-launch without a network: it stops at a Firebase
online-status check and an OFFLINE modal, and never reaches the battle home
screen (`M1B-E010`). It plays fine once the network is cut. So the only online
window is application startup, and this script exists to make that window short,
verified, and identical every time rather than a sequence typed by hand.

`snapshot` removes the window entirely: an emulator snapshot taken while the game
is up and idle with the radios already down restores into an already-started,
already-offline game. A restore also skips whatever intro a cold boot walks
through, and — the reason that matters most for the benchmark — it starts from
byte-identical account state, so the progression this account accumulates
between runs cannot drift between two arms of a comparison the way it did in the
frame-size sweep (`M1B-E018`).

A snapshot carries the bridge that was deployed when it was taken, and a stale
bridge does not answer the current client, so each snapshot is named for the
bridge inside it: `tower_clone_home_offline_<key>`, where the key is a hash of
`libtower_bridge.so` in the private build directory. `up` is therefore the normal
way an instance comes up:

    uv run python scripts/clone_session.py up

It restores that snapshot when the AVD holds one for the bridge we are about to
deploy, verifies it (offline by interface, game process alive, the bridge's own
readiness reading) and connects. Otherwise — no snapshot, a snapshot for another
bridge, or a restore that does not verify — it takes the cold path once, with its
one online window, and saves the snapshot the next `up` restores. `up --cold`
forces the cold path.

Between arms of a comparison, restore the pinned state deliberately so each arm
starts from the same account:

    uv run python scripts/clone_session.py restore     # the current bridge's snapshot

Several instances can run at once from the one clone AVD. `--read-only` gives
each instance its own writable overlay over the untouched base image, so N
actors need N overlays rather than N copies of a multi-gigabyte AVD. The
instance is addressed by `--index`, a top-level flag that precedes the
subcommand: index 0 is today's `emulator-5556`, and each further index takes
the next even console port. A read-only instance cannot save a snapshot; take
snapshots on index 0 without `--read-only`, run alone.

The emulator refuses to share one AVD unless *every* instance holding it is
`-read-only` — including index 0. Bringing up a second instance while index 0
is running writable is refused outright, so index 0 has to be started
`--read-only` too before any further index can attach. `launch_emulator` checks
this itself for any index above 0 and raises before touching the emulator at
all, because the emulator's own refusal of the second instance says nothing
about the first one being the cause.

UNVERIFIED ON THIS HOST: an attempt to add a second instance failed here
because index 0 was running writable, which is the constraint above; whether
several `-read-only` instances can then share the AVD is still unverified and
is checked by the next device stage, not by this file.

Nothing here ever taps, and nothing here reads a pixel. Readiness — "the game has
finished starting up, so it is safe to cut the network and to snapshot" — is the
bridge's own reading: it reports `main_unavailable` while the game is still
starting and `no_initialized_run` once it is up and idle at home. Screenshot
classification was the previous oracle, and it is renderer-dependent and
intermittently wrong, and it is gone: the review path is a human watching a
checkpoint play, which needs no classifier.

Readiness therefore needs the bridge deployed first, and `instrumented_bridge.sh
deploy` refuses to run against an online instance, so the cold path's order is
`start` (the
instance up and offline, the game not launched), then `deploy`, then `launch` —
the one short online window, which also covers deploy's own cold launch.

    uv run python scripts/clone_session.py start
    ./scripts/instrumented_bridge.sh deploy emulator-5556
    uv run python scripts/clone_session.py launch
    uv run python scripts/clone_session.py snapshot
    uv run python scripts/clone_session.py restore

Adding a second instance to share the AVD needs index 0 read-only too, and
`--index` precedes the subcommand it applies to:

    uv run python scripts/clone_session.py start --read-only
    uv run python scripts/clone_session.py --index 1 up --read-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tower_rl.simulation.bridge import ActorFailure, deploy_bridge  # noqa: E402
from tower_rl.simulation.bring_up import (  # noqa: E402
    bridge_key,
    bring_up,
    game_pid,
    keyed_snapshot_name,
    launch_game_at_home,
    restore,
    routable_interfaces,
    save_snapshot,
    snapshot_exists,
    start,
    why_not_ready,
)
from tower_rl.simulation.frame_rate import raise_frame_rate  # noqa: E402
from tower_rl.simulation.instance import (  # noqa: E402
    CLONE_AVD,
    CloneError,
    CloneInstance,
)


def report(instance: CloneInstance) -> None:
    routable = routable_interfaces(instance)
    reason = why_not_ready(instance)
    print(f"instance:  {instance.serial} ({instance.avd})")
    print(f"game pid:  {game_pid(instance) or 'not running'}")
    print(f"ready:     {reason or 'yes: up and idle, the bridge can drive it'}")
    print(f"network:   {'; '.join(routable) if routable else 'offline'}")
    try:
        name = keyed_snapshot_name(bridge_key())
    except CloneError as error:
        print(f"snapshot:  unknown: {error}")
        return
    held = "saved" if snapshot_exists(instance, name) else "not saved: the next bring-up is cold"
    print(f"snapshot:  {name} ({held})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="instance index; 0 is emulator-5556")
    parser.add_argument("--avd", default=CLONE_AVD, help="clone AVD; the canonical AVD is refused")
    sub = parser.add_subparsers(dest="command", required=True)
    started = sub.add_parser("start", help="cold start: the instance up and offline")
    started.add_argument("--renderer", default="lavapipe")
    restored = sub.add_parser("restore", help="launch from a snapshot without connecting")
    restored.add_argument("name", nargs="?", help="default: the snapshot for the current bridge")
    restored.add_argument("--renderer", default="lavapipe")
    up = sub.add_parser(
        "up", help="restore the snapshot for the current bridge, or cold-start and save one"
    )
    up.add_argument("--renderer", default="lavapipe")
    up.add_argument("--cold", action="store_true", help="skip any snapshot and take the cold path")
    for launching in (started, restored, up):
        launching.add_argument(
            "--read-only",
            action="store_true",
            help="share the AVD with other instances; cannot save a snapshot",
        )
        launching.add_argument("--cores", type=int, default=8)
    sub.add_parser(
        "launch", help="launch the game online, wait for the bridge, then cut the radios"
    )
    sub.add_parser("verify", help="report game process, readiness and network state")
    saved = sub.add_parser("snapshot", help="save a snapshot of the running, offline game")
    saved.add_argument("name", nargs="?", help="default: the snapshot for the current bridge")
    arguments = parser.parse_args()

    try:
        instance = CloneInstance(index=arguments.index, avd=arguments.avd)
        if arguments.command == "start":
            start(
                instance,
                arguments.renderer,
                read_only=arguments.read_only,
                cores=arguments.cores,
            )
        elif arguments.command == "launch":
            launch_game_at_home(instance)
        elif arguments.command == "up":
            bring_up(
                instance,
                arguments.renderer,
                deploy=deploy_bridge,
                read_only=arguments.read_only,
                cores=arguments.cores,
                force_cold=arguments.cold,
            )
            # A single instance is a fleet of one: it is up, so it may be
            # raised — but only a `--read-only` instance, which writes to a
            # throwaway overlay. `cmd game set` is GameManagerService state that
            # outlives the process, so raising a writable instance would leave
            # the clone AVD modified and a later `snapshot` would bake the
            # override into the state every future run restores.
            if arguments.read_only:
                raise_frame_rate(instance)
            else:
                print(
                    f"{instance.serial}: writable, so the guest stays at the stock rate; "
                    "raise it on a --read-only instance",
                    flush=True,
                )
        elif arguments.command == "snapshot":
            save_snapshot(instance, arguments.name or keyed_snapshot_name(bridge_key()))
        elif arguments.command == "restore":
            restore(
                instance,
                arguments.name or keyed_snapshot_name(bridge_key()),
                renderer=arguments.renderer,
                read_only=arguments.read_only,
                cores=arguments.cores,
            )
        else:
            report(instance)
    # Two names, not one, and deliberately not one type. `deploy_bridge` is
    # shared with the fleet and raises `ActorFailure`, which also covers a
    # `run_episodes.py` process that failed — not a statement about the clone's
    # state, so it is not a `CloneError` and must not be made one. What this
    # command owes its caller either way is the reason by name and a non-zero
    # status, which is what a traceback out of `main` stopped giving it.
    except (CloneError, ActorFailure) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
