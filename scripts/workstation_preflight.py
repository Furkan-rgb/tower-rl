#!/usr/bin/env python3
"""Read-only workstation and Android-tool inventory for Tower-RL."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def _command(name: str) -> str | None:
    path = shutil.which(name)
    return str(Path(path).resolve()) if path else None


def _run(command: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, "", str(error)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _sdk_root() -> str | None:
    configured = os.environ.get("ANDROID_SDK_ROOT") or os.environ.get("ANDROID_HOME")
    candidates = [configured] if configured else []
    candidates.extend(
        [
            str(Path.home() / "Library/Android/sdk"),
            str(Path.home() / "Android/Sdk"),
            "/opt/homebrew/share/android-commandlinetools",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return str(Path(candidate).resolve())
    return None


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


def _tool_report(sdk_root: str | None) -> dict[str, object]:
    tools: dict[str, str | None] = {
        name: _command(name) for name in ("adb", "emulator", "sdkmanager", "avdmanager")
    }
    if sdk_root:
        roots = {
            "adb": Path(sdk_root) / "platform-tools/adb",
            "emulator": Path(sdk_root) / "emulator/emulator",
            "sdkmanager": Path(sdk_root) / "cmdline-tools/latest/bin/sdkmanager",
            "avdmanager": Path(sdk_root) / "cmdline-tools/latest/bin/avdmanager",
        }
        for name, candidate in roots.items():
            if tools[name] is None and candidate.is_file():
                tools[name] = str(candidate.resolve())
    return tools


def build_report(serial: str | None = None) -> dict[str, object]:
    sdk_root = _sdk_root()
    tools = _tool_report(sdk_root)
    report: dict[str, object] = {
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": _memory_bytes(),
            "free_storage_bytes": shutil.disk_usage(Path.cwd()).free,
        },
        "android_sdk_root": sdk_root,
        "tools": tools,
        "avds": [],
        "system_images": [],
    }
    emulator = tools["emulator"]
    if isinstance(emulator, str):
        code, output, error = _run([emulator, "-list-avds"])
        report["avds"] = output.splitlines() if code == 0 else {"error": error}
    if sdk_root:
        image_root = Path(sdk_root) / "system-images"
        if image_root.is_dir():
            report["system_images"] = [
                str(path.relative_to(image_root))
                for path in sorted(image_root.rglob("source.properties"))
            ]
    if serial:
        adb = tools["adb"]
        device: dict[str, object] = {"serial": serial}
        if isinstance(adb, str):
            code, output, error = _run([adb, "-s", serial, "get-state"])
            device["state"] = output if code == 0 else "unavailable"
            if error:
                device["error"] = error
        report["device"] = device
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="Optional ADB serial to check without changing state.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()
    report = build_report(args.serial)
    tools = report["tools"]
    if not isinstance(tools, dict):
        raise RuntimeError("internal error: malformed tool report")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        host = report["host"]
        print(f"Host: {host['system']} {host['machine']} ({host['cpu_count']} CPUs)")
        print(f"Android SDK: {report['android_sdk_root'] or 'not found'}")
        print("Tools:")
        for name, path in tools.items():
            print(f"  {name}: {path or 'not found'}")
        print(f"AVDs: {', '.join(report['avds']) or 'none detected'}")
        print(f"System images: {len(report['system_images'])}")
        if "device" in report:
            print(f"Device {args.serial}: {report['device']['state']}")
    return 0 if all(tools[name] for name in ("adb", "emulator")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
