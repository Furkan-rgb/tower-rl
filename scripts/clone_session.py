#!/usr/bin/env python3
"""Bring the disposable clone up offline, and snapshot it already running.

The game cannot cold-launch without a network: it stops at a Firebase
online-status check and an OFFLINE modal, and never reaches the battle home
screen (`M1B-E010`). It plays fine once the network is cut. So the only online
window is application startup, and this script exists to make that window short,
verified, and identical every time rather than a sequence typed by hand.

`snapshot` removes the window entirely: an emulator snapshot taken while the game
is at home with the radios already down restores into an already-started,
already-offline game.

Nothing here ever taps. It observes the screen and changes radio state only.

    uv run python scripts/clone_session.py start
    uv run python scripts/clone_session.py snapshot tower_clone_home_offline
    uv run python scripts/clone_session.py restore tower_clone_home_offline
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from tower_rl.doctor import find_android_tool  # noqa: E402
from tower_rl.infrastructure.adb_device import AdbDevice  # noqa: E402
from tower_rl.infrastructure.visual_profile import Screen, classify  # noqa: E402

PACKAGE = "com.TechTreeGames.TheTower"
#: The disposable rooted clone. The canonical evaluation AVD is never touched.
SERIAL = "emulator-5556"
AVD = "tower_rl_instrumented_api36"
PORT = 5556


class CloneError(RuntimeError):
    """The clone is not in the state this step requires."""


def adb(*args: str, timeout: float = 30.0) -> str:
    binary = find_android_tool("adb")
    if binary is None:
        raise CloneError("adb not found")
    result = subprocess.run(
        [str(binary), "-s", SERIAL, *args], capture_output=True, text=True, timeout=timeout
    )
    return result.stdout.strip()


def routable_interfaces() -> list[str]:
    """Interfaces other than loopback holding an IPv4 address.

    This is the offline check. `airplane_mode_on` is not: it reads 1 while the
    wifi radio is still up with a route, which is how every run before
    `M1B-E010` executed online while reporting itself offline.
    """
    lines = adb("shell", "ip", "-o", "-4", "addr", "show").splitlines()
    return [line.strip() for line in lines if line.strip() and " lo " not in line]


def require_offline() -> None:
    routable = routable_interfaces()
    if routable:
        raise CloneError(f"device is online: {'; '.join(routable)}")


def set_radios(enabled: bool, *, settle: float = 25.0) -> None:
    """Turn the radios on or off and wait for the interface to follow."""
    state = "enable" if enabled else "disable"
    adb("shell", "svc", "wifi", state)
    adb("shell", "svc", "data", state)
    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        if bool(routable_interfaces()) is enabled:
            return
        time.sleep(2.0)
    raise CloneError(f"radios did not turn {state} within {settle:.0f}s")


def screen() -> Screen:
    frame = AdbDevice(SERIAL).screenshot()
    return classify(Image.open(io.BytesIO(frame.png_bytes)).convert("RGB"))


def wait_for_screen(expected: Screen, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        current = screen()
        if current is not last:
            print(f"  screen: {current.value}", flush=True)
            last = current
        if current is expected:
            return
        time.sleep(5.0)
    seen = last.value if last else "nothing"
    raise CloneError(f"never reached {expected.value}; last saw {seen}")


def wait_for_boot(*, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if adb("shell", "getprop", "sys.boot_completed", timeout=10.0) == "1":
            return
        time.sleep(5.0)
    raise CloneError("the emulator did not finish booting")


def launch_emulator(renderer: str, snapshot: str | None) -> None:
    binary = find_android_tool("emulator")
    if binary is None:
        raise CloneError("emulator not found")
    command = [
        str(binary), f"@{AVD}",
        "-gpu", renderer, "-no-audio", "-no-boot-anim", "-no-window",
        "-cores", "8", "-port", str(PORT), "-no-snapshot-save",
    ]
    command += ["-snapshot", snapshot] if snapshot else ["-no-snapshot-load"]
    print(f"launching {AVD} ({renderer}{', snapshot ' + snapshot if snapshot else ''})", flush=True)
    subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    wait_for_boot(timeout=300.0)


def start(renderer: str) -> None:
    """Cold start: online only long enough to get the game past its check."""
    launch_emulator(renderer, snapshot=None)
    print("enabling radios for the startup check only", flush=True)
    set_radios(True)
    adb("shell", "am", "force-stop", PACKAGE)
    adb("shell", "monkey", "-p", PACKAGE, "-c", "android.intent.category.LAUNCHER", "1")
    wait_for_screen(Screen.HOME, timeout=300.0)
    print("reached home; cutting the network", flush=True)
    set_radios(False)
    require_offline()
    if screen() is not Screen.HOME:
        raise CloneError("the game left home when the network was cut")
    print("clone is at home and offline", flush=True)


def restore(snapshot: str) -> None:
    """The point of the snapshot: never connect at all."""
    launch_emulator("lavapipe", snapshot=snapshot)
    require_offline()
    current = screen()
    if current is not Screen.HOME:
        raise CloneError(f"restored snapshot is not at home: {current.value}")
    print(f"restored {snapshot}: at home, offline, never connected", flush=True)


def save_snapshot(name: str) -> None:
    """Refuse to capture a state that is not the one worth restoring."""
    require_offline()
    current = screen()
    if current is not Screen.HOME:
        raise CloneError(f"refusing to snapshot: screen is {current.value}, not home")
    if not adb("shell", "pidof", PACKAGE):
        raise CloneError("refusing to snapshot: the game is not running")
    print(adb("emu", "avd", "snapshot", "save", name, timeout=300.0) or "saved", flush=True)
    print(f"snapshot {name}: game running, at home, offline", flush=True)


def report() -> None:
    routable = routable_interfaces()
    print(f"screen:    {screen().value}")
    print(f"game pid:  {adb('shell', 'pidof', PACKAGE) or 'not running'}")
    print(f"network:   {'; '.join(routable) if routable else 'offline'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    started = sub.add_parser("start", help="cold start, online only for the startup check")
    started.add_argument("--renderer", default="lavapipe")
    sub.add_parser("verify", help="report screen, game process and network state")
    saved = sub.add_parser("snapshot", help="save a snapshot of the running, offline game")
    saved.add_argument("name")
    restored = sub.add_parser("restore", help="launch from a snapshot without connecting")
    restored.add_argument("name")
    arguments = parser.parse_args()

    try:
        if arguments.command == "start":
            start(arguments.renderer)
        elif arguments.command == "snapshot":
            save_snapshot(arguments.name)
        elif arguments.command == "restore":
            restore(arguments.name)
        else:
            report()
    except CloneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
