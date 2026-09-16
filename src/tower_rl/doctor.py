"""Read-only host, package, and Android-device diagnostics."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from tower_rl.xapk import XapkInspectionError, inspect_xapk


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sdk_roots() -> tuple[Path, ...]:
    configured = [
        Path(value)
        for variable in ("ANDROID_HOME", "ANDROID_SDK_ROOT")
        if (value := os.environ.get(variable))
    ]
    conventional = [
        Path.home() / "Library/Android/sdk",
        Path.home() / "Android/Sdk",
        # Where the Linux workstation bootstrap installs it. Without this an
        # unattended run cannot find adb unless a shell happens to export the
        # SDK on PATH, which a background process does not inherit.
        Path.home() / ".local/share/android-sdk",
        Path("/opt/homebrew/share/android-commandlinetools"),
    ]
    roots: list[Path] = []
    for root in configured + conventional:
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def find_android_tool(name: str) -> Path | None:
    direct = shutil.which(name)
    if direct:
        return Path(direct).resolve()
    relative_candidates = {
        "adb": ("platform-tools/adb",),
        "emulator": ("emulator/emulator",),
        "apkanalyzer": ("cmdline-tools/latest/bin/apkanalyzer",),
        "sdkmanager": ("cmdline-tools/latest/bin/sdkmanager",),
    }
    for root in _sdk_roots():
        for relative in relative_candidates.get(name, ()):
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    return None


def _run(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)


def check_host() -> CheckResult:
    usage = shutil.disk_usage(Path.cwd())
    details: dict[str, object] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "free_storage_bytes": usage.free,
    }
    if platform.system() == "Darwin":
        virtualization = _run(["sysctl", "-n", "kern.hv_support"])
        details["hardware_virtualization"] = virtualization.stdout.strip() == "1"
    status = CheckStatus.PASS if usage.free >= 20 * 1024**3 else CheckStatus.WARN
    return CheckResult(
        name="host",
        status=status,
        message="host characterized" if status is CheckStatus.PASS else "less than 20 GiB free",
        details=details,
    )


def check_android_tools() -> CheckResult:
    resolved = {name: find_android_tool(name) for name in ("adb", "emulator", "apkanalyzer")}
    missing = [name for name, path in resolved.items() if path is None]
    if missing:
        return CheckResult(
            name="android_tools",
            status=CheckStatus.FAIL,
            message=f"missing Android tools: {', '.join(missing)}",
            details={name: str(path) if path else None for name, path in resolved.items()},
        )
    return CheckResult(
        name="android_tools",
        status=CheckStatus.PASS,
        message="required Android tools found",
        details={name: str(path) for name, path in resolved.items()},
    )


def check_xapk(path: Path) -> CheckResult:
    analyzer = find_android_tool("apkanalyzer")
    if analyzer is None:
        return CheckResult(
            name="xapk",
            status=CheckStatus.FAIL,
            message="cannot inspect XAPK without apkanalyzer",
            details={"path": str(path)},
        )
    try:
        metadata = inspect_xapk(path, analyzer)
    except XapkInspectionError as error:
        return CheckResult(
            name="xapk",
            status=CheckStatus.FAIL,
            message=str(error),
            details={"path": str(path)},
        )
    return CheckResult(
        name="xapk",
        status=CheckStatus.PASS,
        message=(
            f"{metadata.package_name} {metadata.version_name}; "
            f"{len(metadata.apks)} APKs; ABIs={','.join(metadata.native_abis)}"
        ),
        details=metadata.to_dict(),
    )


def check_device(serial: str | None) -> CheckResult:
    adb = find_android_tool("adb")
    if adb is None:
        return CheckResult("device", CheckStatus.FAIL, "adb not found", {})
    devices = _run([str(adb), "devices", "-l"])
    connected = [
        line.split()[0]
        for line in devices.stdout.splitlines()[1:]
        if line.strip() and " device " in f" {line} "
    ]
    if serial is None:
        status = CheckStatus.PASS if connected else CheckStatus.WARN
        return CheckResult(
            "device",
            status,
            f"{len(connected)} Android device(s) connected",
            {"connected_serials": connected},
        )
    if serial not in connected:
        return CheckResult(
            "device",
            CheckStatus.FAIL,
            f"configured device is not connected: {serial}",
            {"connected_serials": connected},
        )
    properties: dict[str, object] = {"serial": serial}
    for key in ("ro.product.cpu.abi", "ro.build.version.sdk", "sys.boot_completed"):
        result = _run([str(adb), "-s", serial, "shell", "getprop", key])
        properties[key] = result.stdout.strip()
    game_package = _run(
        [str(adb), "-s", serial, "shell", "pm", "path", "com.TechTreeGames.TheTower"]
    )
    properties["game_package_installed"] = game_package.stdout.startswith("package:")
    billing = _run(
        [
            str(adb),
            "-s",
            serial,
            "shell",
            "cmd",
            "package",
            "query-services",
            "--brief",
            "-a",
            "com.android.vending.billing.InAppBillingService.BIND",
        ]
    )
    billing_available = (
        "No services found" not in billing.stdout and "services found" in billing.stdout
    )
    properties["play_billing_service_available"] = billing_available
    if not billing_available:
        return CheckResult(
            "device",
            CheckStatus.FAIL,
            "device image exposes no Google Play billing service",
            properties,
        )
    if not properties["game_package_installed"]:
        return CheckResult(
            "device",
            CheckStatus.FAIL,
            "The Tower package is not installed on the configured device",
            properties,
        )
    return CheckResult("device", CheckStatus.PASS, f"device ready: {serial}", properties)


def run_doctor(xapk: Path, serial: str | None) -> list[CheckResult]:
    return [check_host(), check_android_tools(), check_xapk(xapk), check_device(serial)]


def render_json(results: list[CheckResult]) -> str:
    payload = {
        "ok": not any(result.status is CheckStatus.FAIL for result in results),
        "checks": [result.to_dict() for result in results],
    }
    return json.dumps(payload, indent=2, sort_keys=True)
