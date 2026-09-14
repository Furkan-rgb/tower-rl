"""Concrete ADB adapter for the M0 baseline/navigation probe."""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from PIL import Image

from tower_rl.doctor import find_android_tool

GAME_PACKAGE = "com.TechTreeGames.TheTower"
EXPECTED_SIZE = (1080, 1920)


class ProbeError(RuntimeError):
    """Raised when a probe operation cannot establish a trusted result."""


class ScreenKind(StrEnum):
    HOME = "battle_home_tier_1"
    ACTIVE_RUN = "tier_1_active_run"
    WAVE_INFO = "tier_1_wave_info_modal"
    RESULT = "tier_1_result"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProbeReport:
    """Evidence returned by one validated frame probe."""

    screen: ScreenKind
    foreground_package: str | None
    airplane_mode: bool
    route: str
    frame_sha256: str
    width: int
    height: int
    valid: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NavigationReport:
    """Evidence for the bounded Home -> run -> result -> Home flow."""

    initial: ProbeReport
    active_run: ProbeReport
    result: ProbeReport
    final: ProbeReport
    valid: bool
    failure: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _close(
    pixel: tuple[int, int, int], expected: tuple[int, int, int], tolerance: int = 8
) -> bool:
    return all(
        abs(actual - wanted) <= tolerance
        for actual, wanted in zip(pixel, expected, strict=True)
    )


def classify_frame(image: Image.Image) -> ScreenKind:
    """Classify only the known profile states; return UNKNOWN on drift."""

    image = image.convert("RGB")
    if image.size != EXPECTED_SIZE:
        return ScreenKind.UNKNOWN

    def pixel(x: int, y: int) -> tuple[int, int, int]:
        return cast(tuple[int, int, int], image.getpixel((x, y)))

    home_header = pixel(10, 10)
    home_body = pixel(10, 200)
    home_panel = pixel(540, 1000)
    if (
        _close(home_header, (54, 49, 118))
        and _close(home_body, (28, 24, 53))
        and _close(home_panel, (54, 49, 118))
    ):
        return ScreenKind.HOME

    top_left = pixel(10, 10)
    center = pixel(540, 1000)
    if sum(top_left) < 70 and sum(pixel(540, 180)) >= 200:
        return ScreenKind.WAVE_INFO
    if sum(top_left) < 70 and sum(center) >= 70:
        return ScreenKind.RESULT
    if sum(top_left) < 70 and sum(center) < 70:
        return ScreenKind.ACTIVE_RUN
    return ScreenKind.UNKNOWN


class AndroidProbe:
    """ADB-backed infrastructure adapter for one already-provisioned device."""

    def __init__(self, serial: str, adb: Path | None = None) -> None:
        self.serial = serial
        self.adb = adb or find_android_tool("adb")
        if self.adb is None:
            raise ProbeError("adb not found")

    def _run(self, *arguments: str, timeout: float = 15.0) -> str:
        result = subprocess.run(
            [str(self.adb), "-s", self.serial, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown adb error"
            raise ProbeError(f"adb {' '.join(arguments)} failed: {detail}")
        return result.stdout

    def _foreground_package(self) -> str | None:
        output = self._run("shell", "dumpsys", "window")
        match = re.search(r"mCurrentFocus=.*?\s([\w.]+)/", output)
        return match.group(1) if match else None

    def _airplane_mode(self) -> bool:
        return self._run("shell", "settings", "get", "global", "airplane_mode_on").strip() == "1"

    def _route(self) -> str:
        return self._run("shell", "ip", "route").strip()

    def capture(self, frame_path: Path | None = None) -> tuple[Image.Image, str]:
        result = subprocess.run(
            [str(self.adb), "-s", self.serial, "exec-out", "screencap", "-p"],
            check=False,
            capture_output=True,
            timeout=20,
        )
        if result.returncode != 0 or not result.stdout:
            detail = result.stderr.decode(errors="replace").strip() or "empty screencap"
            raise ProbeError(f"screencap failed: {detail}")
        digest = hashlib.sha256(result.stdout).hexdigest()
        if frame_path is not None:
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame_path.write_bytes(result.stdout)
        try:
            image = Image.open(io.BytesIO(result.stdout))
            image.load()
        except Exception as error:  # Pillow has several decode exception types.
            raise ProbeError(f"invalid PNG screencap: {error}") from error
        return image, digest

    def report(self, frame_path: Path | None = None) -> ProbeReport:
        image, digest = self.capture(frame_path)
        screen = classify_frame(image)
        foreground = self._foreground_package()
        airplane = self._airplane_mode()
        route = self._route()
        reasons: list[str] = []
        if image.size != EXPECTED_SIZE:
            reasons.append(f"unexpected frame size {image.size[0]}x{image.size[1]}")
        if foreground != GAME_PACKAGE:
            reasons.append(f"unexpected foreground package {foreground!r}")
        if not airplane:
            reasons.append("airplane mode is disabled")
        if route:
            reasons.append(f"external route present: {route}")
        if screen is ScreenKind.UNKNOWN:
            reasons.append("screen did not match the pinned visual profile")
        return ProbeReport(
            screen=screen,
            foreground_package=foreground,
            airplane_mode=airplane,
            route=route,
            frame_sha256=digest,
            width=image.width,
            height=image.height,
            valid=not reasons,
            reasons=tuple(reasons),
        )

    def tap(self, x: int, y: int) -> None:
        self._run("shell", "input", "tap", str(x), str(y))

    def restore_snapshot(self, name: str) -> None:
        """Restore a named local AVD snapshot through the emulator console."""

        self._run("emu", "avd", "snapshot", "load", name, timeout=30.0)
        time.sleep(2.0)

    def wait_for(self, expected: ScreenKind, timeout: float = 12.0) -> ProbeReport:
        deadline = time.monotonic() + timeout
        last: ProbeReport | None = None
        while time.monotonic() < deadline:
            last = self.report()
            if last.valid and last.screen is expected:
                return last
            if expected is ScreenKind.RESULT and last.valid and last.screen is ScreenKind.WAVE_INFO:
                self.tap(1005, 250)
            time.sleep(0.5)
        if last is None:
            raise ProbeError(f"timed out waiting for {expected.value}")
        raise ProbeError(
            f"timed out waiting for {expected.value}; last={last.screen.value}; "
            f"reasons={'; '.join(last.reasons)}"
        )

    def navigate_home_to_tier1_and_back(self) -> NavigationReport:
        """Run the bounded no-upgrade navigation smoke sequence."""

        initial = self.report()
        if not initial.valid or initial.screen is not ScreenKind.HOME:
            raise ProbeError(f"initial baseline invalid: {initial.to_dict()}")
        self.tap(540, 1550)
        active = self.wait_for(ScreenKind.ACTIVE_RUN)
        self.tap(1015, 68)
        time.sleep(0.5)
        self.tap(960, 270)
        time.sleep(0.5)
        self.tap(720, 1095)
        result = self.wait_for(ScreenKind.RESULT)
        self.tap(780, 1420)
        final = self.wait_for(ScreenKind.HOME)
        return NavigationReport(initial, active, result, final, True, None)


class AdbProbe(AndroidProbe):
    """Explicit adapter name used by the application composition root."""


__all__ = [
    "AdbProbe",
    "AndroidProbe",
    "EXPECTED_SIZE",
    "GAME_PACKAGE",
    "NavigationReport",
    "ProbeError",
    "ProbeReport",
    "ScreenKind",
    "classify_frame",
]
