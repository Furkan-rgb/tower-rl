"""Application-facing port for ordinary Android control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ScreenPoint:
    """Normalized UI point; learned policies never construct this type."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ValueError("screen coordinates must be normalized to [0, 1]")

    def pixels(self, width: int, height: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ValueError("screen dimensions must be positive")
        return round(self.x * (width - 1)), round(self.y * (height - 1))


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: str
    captured_at_monotonic: float
    width: int
    height: int
    png_bytes: bytes
    path: Path | None = None


@dataclass(frozen=True)
class InputReceipt:
    event_id: str
    accepted_at_monotonic: float


@dataclass(frozen=True)
class DeviceHealth:
    serial: str
    connected: bool
    boot_completed: bool
    foreground_package: str | None
    renderer_profile: str | None
    failure_reason: str | None = None


class AndroidDevice(Protocol):
    """Ordinary-input device boundary used by controller and environment."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def health(self) -> DeviceHealth: ...

    def screenshot(self) -> CapturedFrame: ...

    def tap(self, point: ScreenPoint) -> InputReceipt: ...

    def app_foreground(self) -> bool: ...

    def launch_app(self) -> None: ...

    def restore_baseline(self, baseline_id: str) -> None: ...
