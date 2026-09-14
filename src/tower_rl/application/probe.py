"""Application use case for the M0 baseline/navigation probe."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ProbeReportLike(Protocol):
    def to_dict(self) -> dict[str, object]: ...


class ProbePort(Protocol):
    """Port required by the probe use case; infrastructure supplies it."""

    def report(self, frame_path: Path | None = None) -> ProbeReportLike: ...

    def navigate_home_to_tier1_and_back(self) -> ProbeReportLike: ...

    def restore_snapshot(self, name: str) -> None: ...


class ProbeService:
    """Coordinates probing without knowing how Android transport is implemented."""

    def __init__(self, probe: ProbePort) -> None:
        self._probe = probe

    def execute(
        self, *, navigate: bool, frame_path: Path | None, restore_snapshot: str | None
    ) -> dict[str, object]:
        if navigate:
            report = self._probe.navigate_home_to_tier1_and_back()
            payload = report.to_dict()
            if restore_snapshot:
                self._probe.restore_snapshot(restore_snapshot)
                payload["restored_snapshot"] = restore_snapshot
            return payload
        report = self._probe.report(frame_path)
        return report.to_dict()
