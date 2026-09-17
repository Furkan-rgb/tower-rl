#!/usr/bin/env python3
"""Bring a disposable clone instance up offline, and snapshot it already running.

The game cannot cold-launch without a network: it stops at a Firebase
online-status check and an OFFLINE modal, and never reaches the battle home
screen (`M1B-E010`). It plays fine once the network is cut. So the only online
window is application startup, and this script exists to make that window short,
verified, and identical every time rather than a sequence typed by hand.

`snapshot` removes the window entirely: an emulator snapshot taken while the game
is at home with the radios already down restores into an already-started,
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

Nothing here ever taps. It observes the screen and changes radio state only.

    uv run python scripts/clone_session.py start
    uv run python scripts/clone_session.py --index 1 --read-only start
    uv run python scripts/clone_session.py snapshot tower_clone_home_offline
    uv run python scripts/clone_session.py restore tower_clone_home_offline
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from tower_rl.doctor import find_android_tool  # noqa: E402
from tower_rl.infrastructure.adb_device import AdbDevice  # noqa: E402
from tower_rl.infrastructure.visual_profile import Screen, classify  # noqa: E402

PACKAGE = "com.TechTreeGames.TheTower"
#: The disposable rooted clone. The canonical evaluation AVD is never touched.
CLONE_AVD = "tower_rl_instrumented_api36"
#: The canonical evaluation AVD: never launched by automation, at any index.
CANONICAL_AVD = "tower_rl_api36_play_x86_64"
#: Emulator console ports are even and two apart, and the adb serial is the
#: console port. Index 0 is the port every existing invocation already uses.
FIRST_CONSOLE_PORT = 5556


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


def screen(instance: CloneInstance) -> Screen:
    frame = AdbDevice(instance.serial).screenshot()
    return classify(Image.open(io.BytesIO(frame.png_bytes)).convert("RGB"))


def wait_for_screen(instance: CloneInstance, expected: Screen, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        current = screen(instance)
        if current is not last:
            print(f"  {instance.serial} screen: {current.value}", flush=True)
            last = current
        if current is expected:
            return
        time.sleep(5.0)
    seen = last.value if last else "nothing"
    raise CloneError(f"never reached {expected.value}; last saw {seen}")


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
    """Cold start: online only long enough to get the game past its check."""
    launch_emulator(instance, renderer, snapshot=None, read_only=read_only, cores=cores)
    launch_game_at_home(instance)


def launch_game_at_home(instance: CloneInstance) -> None:
    """Cold-launch the game past its network check and leave it at home, offline.

    The game blocks on a Firebase check and an OFFLINE modal, so a cold launch
    needs a network however briefly. Anything that cold-launches the game on an
    offline clone — a start, or `instrumented_bridge.sh deploy` — must come back
    through here or it lands on that modal.
    """
    print("enabling radios for the startup check only", flush=True)
    set_radios(instance, True)
    adb(instance, "shell", "am", "force-stop", PACKAGE)
    adb(instance, "shell", "monkey", "-p", PACKAGE, "-c", "android.intent.category.LAUNCHER", "1")
    wait_for_screen(instance, Screen.HOME, timeout=300.0)
    print("reached home; cutting the network", flush=True)
    set_radios(instance, False)
    require_offline(instance)
    if screen(instance) is not Screen.HOME:
        raise CloneError("the game left home when the network was cut")
    print(f"{instance.serial} is at home and offline", flush=True)


def restore(
    instance: CloneInstance, snapshot: str, *, read_only: bool = False, cores: int = 8
) -> None:
    """The point of the snapshot: never connect at all."""
    launch_emulator(instance, "lavapipe", snapshot=snapshot, read_only=read_only, cores=cores)
    require_offline(instance)
    current = screen(instance)
    if current is not Screen.HOME:
        raise CloneError(f"restored snapshot is not at home: {current.value}")
    print(
        f"restored {snapshot} on {instance.serial}: at home, offline, never connected",
        flush=True,
    )


def save_snapshot(instance: CloneInstance, name: str) -> None:
    """Refuse to capture a state that is not the one worth restoring."""
    require_offline(instance)
    current = screen(instance)
    if current is not Screen.HOME:
        raise CloneError(f"refusing to snapshot: screen is {current.value}, not home")
    if not adb(instance, "shell", "pidof", PACKAGE):
        raise CloneError("refusing to snapshot: the game is not running")
    saved = adb(instance, "emu", "avd", "snapshot", "save", name, timeout=300.0)
    print(saved or "saved", flush=True)
    print(f"snapshot {name}: game running, at home, offline", flush=True)


def report(instance: CloneInstance) -> None:
    routable = routable_interfaces(instance)
    print(f"instance:  {instance.serial} ({instance.avd})")
    print(f"screen:    {screen(instance).value}")
    print(f"game pid:  {adb(instance, 'shell', 'pidof', PACKAGE) or 'not running'}")
    print(f"network:   {'; '.join(routable) if routable else 'offline'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="instance index; 0 is emulator-5556")
    parser.add_argument("--avd", default=CLONE_AVD, help="clone AVD; the canonical AVD is refused")
    sub = parser.add_subparsers(dest="command", required=True)
    started = sub.add_parser("start", help="cold start, online only for the startup check")
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
    sub.add_parser("verify", help="report screen, game process and network state")
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
