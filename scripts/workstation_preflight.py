#!/usr/bin/env python3
"""Read-only workstation and Android-tool inventory for Tower-RL.

SDK and tool discovery is not repeated here: it comes from `tower_rl.doctor`,
which is the tested authority. A private copy drifted once already and reported
no SDK on a workstation where the doctor found one.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tower_rl.doctor import find_android_tool, sdk_roots  # noqa: E402

TOOLS = ("adb", "emulator", "sdkmanager", "apkanalyzer")


@dataclass(frozen=True)
class Host:
    system: str
    release: str
    machine: str
    python: str
    cpu_count: int | None
    memory_bytes: int | None
    free_storage_bytes: int


@dataclass(frozen=True)
class WorkstationReport:
    """Everything the inventory found, in one typed shape."""

    host: Host
    android_sdk_root: str | None
    tools: dict[str, str | None]
    avds: list[str] = field(default_factory=list)
    #: Present only when the emulator could not be listed, which is a finding
    #: rather than an empty inventory.
    avd_error: str | None = None
    system_images: list[str] = field(default_factory=list)
    device: dict[str, str] | None = None

    @property
    def ready(self) -> bool:
        return all(self.tools[name] for name in ("adb", "emulator"))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _run(command: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, "", str(error)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        code, output, _ = _run(["sysctl", "-n", "hw.memsize"])
        if code == 0 and output.isdigit():
            return int(output)
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) * 1024
    return None


def build_report(serial: str | None = None) -> WorkstationReport:
    root = next((candidate for candidate in sdk_roots() if candidate.is_dir()), None)
    tools = {name: find_android_tool(name) for name in TOOLS}
    emulator = tools["emulator"]
    avds: list[str] = []
    avd_error: str | None = None
    if emulator is not None:
        code, output, error = _run([str(emulator), "-list-avds"])
        if code == 0:
            avds = output.splitlines()
        else:
            avd_error = error or f"emulator -list-avds exited {code}"

    system_images: list[str] = []
    if root is not None and (root / "system-images").is_dir():
        image_root = root / "system-images"
        system_images = [
            str(path.parent.relative_to(image_root))
            for path in sorted(image_root.rglob("source.properties"))
        ]

    device: dict[str, str] | None = None
    if serial:
        device = {"serial": serial, "state": "unavailable"}
        adb = tools["adb"]
        if adb is not None:
            code, output, error = _run([str(adb), "-s", serial, "get-state"])
            device["state"] = output if code == 0 else "unavailable"
            if error:
                device["error"] = error

    return WorkstationReport(
        host=Host(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            python=platform.python_version(),
            cpu_count=os.cpu_count(),
            memory_bytes=_memory_bytes(),
            free_storage_bytes=shutil.disk_usage(Path.cwd()).free,
        ),
        android_sdk_root=str(root) if root else None,
        tools={name: str(path) if path else None for name, path in tools.items()},
        avds=avds,
        avd_error=avd_error,
        system_images=system_images,
        device=device,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="Optional ADB serial to check without changing state.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()
    report = build_report(args.serial)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        host = report.host
        print(f"Host: {host.system} {host.machine} ({host.cpu_count} CPUs)")
        print(f"Android SDK: {report.android_sdk_root or 'not found'}")
        print("Tools:")
        for name, path in report.tools.items():
            print(f"  {name}: {path or 'not found'}")
        print(f"AVDs: {', '.join(report.avds) or report.avd_error or 'none detected'}")
        print(f"System images: {len(report.system_images)}")
        if report.device is not None:
            print(f"Device {report.device['serial']}: {report.device['state']}")
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
