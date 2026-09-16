"""Ordinary-input Android device adapter used by the M1 controller."""

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

from tower_rl.doctor import find_android_tool
from tower_rl.ports.android import CapturedFrame, DeviceHealth, InputReceipt, ScreenPoint

GAME_PACKAGE = "com.TechTreeGames.TheTower"


class AdbDeviceError(RuntimeError):
    """Raised when a bounded ordinary ADB operation fails."""


class AdbDevice:
    """One stable serial with no game-specific strategy."""

    def __init__(self, serial: str, adb: Path | None = None) -> None:
        self.serial = serial
        self.adb = adb or find_android_tool("adb")
        if self.adb is None:
            raise AdbDeviceError("adb not found")
        self._frame_counter = 0

    def _run(self, *args: str, timeout: float = 20.0) -> str:
        for attempt in range(3):
            result = subprocess.run(
                [str(self.adb), "-s", self.serial, *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout
            detail = result.stderr.strip() or result.stdout.strip() or "unknown adb error"
            transient = any(
                marker in detail.lower()
                for marker in ("not found", "offline", "closed", "no devices")
            )
            if not transient or attempt == 2:
                raise AdbDeviceError(f"adb {' '.join(args)} failed: {detail}")
            time.sleep(1.0)
        raise AssertionError("unreachable")

    def start(self) -> None:
        self.launch_app()

    def stop(self) -> None:
        self._run("shell", "am", "force-stop", GAME_PACKAGE)

    def health(self) -> DeviceHealth:
        state = self._run("get-state").strip()
        booted = self._run("shell", "getprop", "sys.boot_completed").strip() == "1"
        foreground = self._foreground_package()
        return DeviceHealth(
            serial=self.serial,
            connected=state == "device",
            boot_completed=booted,
            foreground_package=foreground,
            renderer_profile=None,
            failure_reason=None if state == "device" and booted else "device not ready",
        )

    def _foreground_package(self) -> str | None:
        output = self._run("shell", "dumpsys", "window")
        for line in output.splitlines():
            if "mCurrentFocus=" in line:
                value = line.rsplit(" ", 1)[-1]
                return value.split("/", 1)[0] if "/" in value else None
        return None

    def screenshot(self) -> CapturedFrame:
        result = subprocess.run(
            [str(self.adb), "-s", self.serial, "exec-out", "screencap", "-p"],
            capture_output=True,
            timeout=20,
            check=False,
        )
        if result.returncode or not result.stdout:
            detail = result.stderr.decode(errors="replace").strip() or "empty screencap"
            raise AdbDeviceError(f"screencap failed: {detail}")
        self._frame_counter += 1
        digest = hashlib.sha256(result.stdout).hexdigest()[:16]
        return CapturedFrame(
            frame_id=f"{self.serial}-{self._frame_counter:08d}-{digest}",
            captured_at_monotonic=time.monotonic(),
            width=1080,
            height=1920,
            png_bytes=result.stdout,
        )

    def tap(self, point: ScreenPoint) -> InputReceipt:
        x, y = point.pixels(1080, 1920)
        event_id = f"tap-{time.monotonic_ns()}"
        self._run("shell", "input", "tap", str(x), str(y))
        return InputReceipt(event_id=event_id, accepted_at_monotonic=time.monotonic())

    def app_foreground(self) -> bool:
        return self._foreground_package() == GAME_PACKAGE

    def launch_app(self) -> None:
        self._run(
            "shell",
            "am",
            "start",
            "-W",
            "-n",
            f"{GAME_PACKAGE}/com.unity3d.player.UnityPlayerActivity",
            timeout=30.0,
        )

    def restore_baseline(self, baseline_id: str) -> None:
        self._run("emu", "avd", "snapshot", "load", baseline_id, timeout=45.0)
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            result = subprocess.run(
                [str(self.adb), "-s", self.serial, "get-state"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip() == "device":
                return
            time.sleep(1.0)
        raise AdbDeviceError(f"device did not reconnect after restoring snapshot: {self.serial}")
