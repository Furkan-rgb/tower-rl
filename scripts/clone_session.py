#!/usr/bin/env python3
"""Bring a disposable clone instance up offline, and snapshot it already running.

The game cannot cold-launch without a network: it stops at a Firebase
online-status check and an OFFLINE modal, and never reaches the battle home
screen (`M1B-E010`). It plays fine once the network is cut. So the only online
window is application startup, and this script exists to make that window short,
verified, and identical every time rather than a sequence typed by hand.

`snapshot` removes the window entirely: an emulator snapshot taken while the game
is up and idle with the radios already down restores into an already-started,
already-offline game.

Several instances can run at once from the one clone AVD. `--read-only` gives
each instance its own writable overlay over the untouched base image, so N
actors need N overlays rather than N copies of a multi-gigabyte AVD. The
instance is addressed by `--index`: index 0 is today's `emulator-5556`, and each
further index takes the next even console port. A read-only instance cannot save
a snapshot; take snapshots on index 0 without `--read-only`.

UNVERIFIED ON THIS HOST: no `-read-only` instance has been launched here yet.
Everything below index 0 is the same sequence that has run for a year; the
shared-AVD mechanism itself is verified by the next device stage, not by this
file.

Nothing here ever taps, and nothing here reads a pixel. Readiness — "the game has
finished starting up, so it is safe to cut the network and to snapshot" — is the
bridge's own reading: it reports `main_unavailable` while the game is still
starting and `no_initialized_run` once it is up and idle at home. Screenshot
classification was the previous oracle, and it is renderer-dependent and
intermittently wrong; `visual_profile` stays for the review path, where a human
watches a checkpoint play and the picture is the point.

Readiness therefore needs the bridge deployed first, and `instrumented_bridge.sh
deploy` refuses to run against an online instance, so the order is `start` (the
instance up and offline, the game not launched), then `deploy`, then `launch` —
the one short online window, which also covers deploy's own cold launch.

    uv run python scripts/clone_session.py start
    uv run python scripts/clone_session.py --index 1 --read-only start
    ./scripts/instrumented_bridge.sh deploy emulator-5556
    uv run python scripts/clone_session.py launch
    uv run python scripts/clone_session.py snapshot tower_clone_home_offline
    uv run python scripts/clone_session.py restore tower_clone_home_offline
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_episodes import compatibility  # noqa: E402

from tower_rl.doctor import find_android_tool  # noqa: E402
from tower_rl.infrastructure.instrumented_bridge import (  # noqa: E402
    BridgeCompatibility,
    BridgeObservation,
    BridgeRunUnavailable,
    InstrumentedBridgeClient,
    InstrumentedBridgeError,
)

PACKAGE = "com.TechTreeGames.TheTower"
#: The disposable rooted clone. The canonical evaluation AVD is never touched.
CLONE_AVD = "tower_rl_instrumented_api36"
#: The canonical evaluation AVD: never launched by automation, at any index.
CANONICAL_AVD = "tower_rl_api36_play_x86_64"
#: Emulator console ports are even and two apart, and the adb serial is the
#: console port. Index 0 is the port every existing invocation already uses.
FIRST_CONSOLE_PORT = 5556
#: The bridge's fixed device port, mirrored from `instrumented_bridge.sh`. Every
#: instance listens on the same device port, so each forwards its own host port.
BRIDGE_DEVICE_PORT = 47651
FIRST_BRIDGE_HOST_PORT = 47652
#: The bridge reports this reason once the game is up and holding no run, which
#: is the home screen; while the game is still starting — the splash, or the
#: Firebase OFFLINE modal — it reports `main_unavailable` instead. See
#: `native/tower_bridge/README.md`.
GAME_IS_IDLE = "no_initialized_run"


class CloneError(RuntimeError):
    """The clone is not in the state this step requires."""


@dataclass(frozen=True)
class CloneInstance:
    """One emulator instance of the clone, identified by index.

    Instance identity is one concept — AVD, console port, adb serial — so it is
    derived in one place rather than restated by every caller.
    """

    index: int = 0
    avd: str = CLONE_AVD

    def __post_init__(self) -> None:
        if self.index < 0:
            raise CloneError(f"instance index must not be negative: {self.index}")
        if self.avd == CANONICAL_AVD:
            raise CloneError(f"refusing to touch the canonical evaluation AVD {CANONICAL_AVD}")

    @property
    def console_port(self) -> int:
        return FIRST_CONSOLE_PORT + 2 * self.index

    @property
    def serial(self) -> str:
        return f"emulator-{self.console_port}"

    @property
    def bridge_host_port(self) -> int:
        """The host side of this instance's forward to the bridge's device port."""
        return FIRST_BRIDGE_HOST_PORT + self.index


def adb(instance: CloneInstance, *args: str, timeout: float = 30.0) -> str:
    binary = find_android_tool("adb")
    if binary is None:
        raise CloneError("adb not found")
    result = subprocess.run(
        [str(binary), "-s", instance.serial, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def routable_interfaces(instance: CloneInstance) -> list[str]:
    """Interfaces other than loopback holding an IPv4 address.

    This is the offline check. `airplane_mode_on` is not: it reads 1 while the
    wifi radio is still up with a route, which is how every run before
    `M1B-E010` executed online while reporting itself offline.
    """
    lines = adb(instance, "shell", "ip", "-o", "-4", "addr", "show").splitlines()
    return [line.strip() for line in lines if line.strip() and " lo " not in line]


def require_offline(instance: CloneInstance) -> None:
    routable = routable_interfaces(instance)
    if routable:
        raise CloneError(f"{instance.serial} is online: {'; '.join(routable)}")


def set_radios(instance: CloneInstance, enabled: bool, *, settle: float = 25.0) -> None:
    """Turn the radios on or off and wait for the interface to follow."""
    state = "enable" if enabled else "disable"
    adb(instance, "shell", "svc", "wifi", state)
    adb(instance, "shell", "svc", "data", state)
    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        if bool(routable_interfaces(instance)) is enabled:
            return
        time.sleep(2.0)
    raise CloneError(f"radios did not turn {state} within {settle:.0f}s")


def game_pid(instance: CloneInstance) -> str:
    return adb(instance, "shell", "pidof", PACKAGE)


def expected_compatibility() -> BridgeCompatibility:
    """The private build's identity, which every bridge handshake is checked against."""
    configured = os.environ.get("TOWER_BRIDGE_BUILD_DIR")
    live = Path("/tmp/tower-bridge-live.latest")
    if not configured and not live.is_file():
        raise CloneError(
            "the private bridge build directory is unknown: set TOWER_BRIDGE_BUILD_DIR"
        )
    try:
        return compatibility(Path(configured or live.read_text().strip()))
    except (OSError, KeyError, ValueError) as error:
        raise CloneError(f"cannot read the private bridge build identity: {error}") from error


def read_bridge_state(instance: CloneInstance) -> BridgeObservation | BridgeRunUnavailable:
    """Ask the deployed bridge what the game is doing, over this instance's port.

    The forward is re-established here rather than assumed: it lives on the host
    side of adb and does not survive the emulator it pointed at, so a restored
    snapshot has none even though the bridge inside it is listening.
    """
    adb(instance, "forward", f"tcp:{instance.bridge_host_port}", f"tcp:{BRIDGE_DEVICE_PORT}")
    client = InstrumentedBridgeClient(
        "127.0.0.1",
        instance.bridge_host_port,
        expected_compatibility=expected_compatibility(),
        connect_timeout=5.0,
        read_timeout=30.0,
        heartbeat_timeout=30.0,
    )
    try:
        client.connect()
        return client.read_state()
    except InstrumentedBridgeError as error:
        raise CloneError(f"the bridge is not answering on {instance.serial}: {error}") from error
    finally:
        client.close()


def why_not_ready(instance: CloneInstance) -> str | None:
    """Why the game cannot be driven yet, or None once it can.

    Ready is not a picture: the process is alive, the bridge completes its
    handshake, and the state it reports is a coherent idle one. The bridge
    answers from the splash and from the OFFLINE modal too, so a handshake alone
    proves nothing about startup — what separates them is `main_unavailable`,
    which is the bridge saying `Main` does not exist yet.
    """
    if not game_pid(instance):
        return "the game is not running"
    try:
        state = read_bridge_state(instance)
    except CloneError as error:
        return str(error)
    if isinstance(state, BridgeRunUnavailable):
        if state.reason == GAME_IS_IDLE:
            return None
        return f"the game is still starting: {state.reason}"
    if state.round_active:
        return f"a round is already running at wave {state.wave}"
    return None


def wait_until_ready(instance: CloneInstance, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "nothing observed yet"
    while time.monotonic() < deadline:
        reason = why_not_ready(instance)
        if reason is None:
            return
        if reason != last:
            print(f"  {instance.serial}: {reason}", flush=True)
            last = reason
        time.sleep(5.0)
    raise CloneError(f"{instance.serial} never became ready: {last}")


def wait_for_boot(instance: CloneInstance, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if adb(instance, "shell", "getprop", "sys.boot_completed", timeout=10.0) == "1":
            return
        time.sleep(5.0)
    raise CloneError(f"{instance.serial} did not finish booting")


def emulator_command(
    instance: CloneInstance,
    *,
    binary: str,
    renderer: str,
    snapshot: str | None,
    read_only: bool,
    cores: int,
) -> list[str]:
    """The exact invocation for one instance.

    `-read-only` is what lets several instances share the one AVD: the base
    image stays untouched and each instance writes to its own overlay. It is
    also incompatible with saving a snapshot, which is why it is a choice and
    not the default.
    """
    command = [
        binary, f"@{instance.avd}",
        "-gpu", renderer, "-no-audio", "-no-boot-anim", "-no-window",
        "-cores", str(cores), "-port", str(instance.console_port), "-no-snapshot-save",
    ]
    if read_only:
        command.append("-read-only")
    command += ["-snapshot", snapshot] if snapshot else ["-no-snapshot-load"]
    return command


def launch_emulator(
    instance: CloneInstance,
    renderer: str,
    snapshot: str | None,
    *,
    read_only: bool = False,
    cores: int = 8,
) -> None:
    binary = find_android_tool("emulator")
    if binary is None:
        raise CloneError("emulator not found")
    command = emulator_command(
        instance,
        binary=str(binary),
        renderer=renderer,
        snapshot=snapshot,
        read_only=read_only,
        cores=cores,
    )
    detail = [renderer]
    if snapshot:
        detail.append(f"snapshot {snapshot}")
    if read_only:
        detail.append("read-only")
    print(
        f"launching {instance.avd} on {instance.serial} ({', '.join(detail)})",
        flush=True,
    )
    subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    wait_for_boot(instance, timeout=300.0)


def kill_emulator(instance: CloneInstance) -> None:
    """Never leave an instance running when a run ends."""
    adb(instance, "emu", "kill", timeout=60.0)


def start(
    instance: CloneInstance, renderer: str, *, read_only: bool = False, cores: int = 8
) -> None:
    """Cold start: the instance up and offline, with the game not yet launched.

    The game is launched afterwards by `launch_game_at_home`, because readiness is
    now the bridge's own reading and the bridge has to be deployed first — and
    `instrumented_bridge.sh deploy` refuses to run against an online instance.
    """
    launch_emulator(instance, renderer, snapshot=None, read_only=read_only, cores=cores)
    set_radios(instance, False)
    require_offline(instance)
    print(f"{instance.serial} is up and offline; the game is not launched yet", flush=True)


def launch_game_at_home(instance: CloneInstance) -> None:
    """Cold-launch the game past its network check and leave it idle at home, offline.

    The game blocks on a Firebase check and an OFFLINE modal, so a cold launch
    needs a network however briefly. Anything that cold-launches the game on an
    offline clone — `instrumented_bridge.sh deploy` above all — must come back
    through here or it stays on that modal.

    Home is established without reading a pixel, by asking the bridge, so this
    requires the bridge already deployed and its overlay mounted: it runs after
    `deploy`, never before it. The radios come down only once the bridge says the
    game is up, and the result is verified by interface.
    """
    print("enabling radios for the startup check only", flush=True)
    set_radios(instance, True)
    adb(instance, "shell", "am", "force-stop", PACKAGE)
    adb(instance, "shell", "monkey", "-p", PACKAGE, "-c", "android.intent.category.LAUNCHER", "1")
    wait_until_ready(instance, timeout=300.0)
    print("the game is up and idle; cutting the network", flush=True)
    set_radios(instance, False)
    require_offline(instance)
    reason = why_not_ready(instance)
    if reason is not None:
        raise CloneError(f"the game did not survive the network being cut: {reason}")
    print(f"{instance.serial} is at home and offline", flush=True)


def restore(
    instance: CloneInstance, snapshot: str, *, read_only: bool = False, cores: int = 8
) -> None:
    """The point of the snapshot: never connect at all."""
    launch_emulator(instance, "lavapipe", snapshot=snapshot, read_only=read_only, cores=cores)
    require_offline(instance)
    # The claim a snapshot makes is that the game is already started, so that is
    # what is checked. The bridge is deliberately not required here: a snapshot
    # may have been taken before any bridge was deployed into it, and the fleet
    # deploys and asserts bridge readiness immediately afterwards anyway.
    if not game_pid(instance):
        raise CloneError(f"restored snapshot has no running game: {snapshot}")
    print(
        f"restored {snapshot} on {instance.serial}: game running, offline, never connected",
        flush=True,
    )


def save_snapshot(instance: CloneInstance, name: str) -> None:
    """Refuse to capture a state that is not the one worth restoring."""
    require_offline(instance)
    reason = why_not_ready(instance)
    if reason is not None:
        raise CloneError(f"refusing to snapshot: {reason}")
    saved = adb(instance, "emu", "avd", "snapshot", "save", name, timeout=300.0)
    print(saved or "saved", flush=True)
    print(f"snapshot {name}: game running, idle at home, offline", flush=True)


def report(instance: CloneInstance) -> None:
    routable = routable_interfaces(instance)
    reason = why_not_ready(instance)
    print(f"instance:  {instance.serial} ({instance.avd})")
    print(f"game pid:  {game_pid(instance) or 'not running'}")
    print(f"ready:     {reason or 'yes: up and idle, the bridge can drive it'}")
    print(f"network:   {'; '.join(routable) if routable else 'offline'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="instance index; 0 is emulator-5556")
    parser.add_argument("--avd", default=CLONE_AVD, help="clone AVD; the canonical AVD is refused")
    sub = parser.add_subparsers(dest="command", required=True)
    started = sub.add_parser("start", help="cold start: the instance up and offline")
    started.add_argument("--renderer", default="lavapipe")
    restored = sub.add_parser("restore", help="launch from a snapshot without connecting")
    restored.add_argument("name")
    for launching in (started, restored):
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
    saved.add_argument("name")
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
        elif arguments.command == "snapshot":
            save_snapshot(instance, arguments.name)
        elif arguments.command == "restore":
            restore(
                instance,
                arguments.name,
                read_only=arguments.read_only,
                cores=arguments.cores,
            )
        else:
            report(instance)
    except CloneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
