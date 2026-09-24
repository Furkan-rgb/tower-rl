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

from tower_rl.simulation.android_sdk import find_android_tool, sdk_roots
from tower_rl.xapk import XapkInspectionError, inspect_xapk

#: The tools a training host needs discoverable on `PATH` or under a standard
#: SDK layout. `sdkmanager` is here beside the three the fleet touches at
#: runtime because it is what installs a missing system image.
ANDROID_TOOLS = ("adb", "emulator", "apkanalyzer", "sdkmanager")


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




def _run(command: list[str], timeout: float = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)


def _memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        memory = _run(["sysctl", "-n", "hw.memsize"])
        if memory.returncode == 0 and memory.stdout.strip().isdigit():
            return int(memory.stdout.strip())
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) * 1024
    return None


def check_host() -> CheckResult:
    usage = shutil.disk_usage(Path.cwd())
    details: dict[str, object] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
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
    resolved = {name: find_android_tool(name) for name in ANDROID_TOOLS}
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


def check_sdk_inventory() -> CheckResult:
    """The SDK root, its AVDs and its installed system images.

    Separate from `check_android_tools`: a host can have every tool on `PATH`
    and still have no AVD or system image to boot one from, which is a
    different finding from a missing binary.
    """
    root = next((candidate for candidate in sdk_roots() if candidate.is_dir()), None)
    avds: list[str] = []
    avd_error: str | None = None
    emulator = find_android_tool("emulator")
    if emulator is not None:
        listing = _run([str(emulator), "-list-avds"])
        if listing.returncode == 0:
            avds = listing.stdout.splitlines()
        else:
            avd_error = listing.stderr.strip() or f"emulator -list-avds exited {listing.returncode}"
    system_images: list[str] = []
    if root is not None and (root / "system-images").is_dir():
        image_root = root / "system-images"
        system_images = [
            str(path.parent.relative_to(image_root))
            for path in sorted(image_root.rglob("source.properties"))
        ]
    details: dict[str, object] = {
        "android_sdk_root": str(root) if root else None,
        "avds": avds,
        "avd_error": avd_error,
        "system_images": system_images,
    }
    if root is None:
        return CheckResult("sdk_inventory", CheckStatus.WARN, "no Android SDK root found", details)
    return CheckResult(
        "sdk_inventory",
        CheckStatus.PASS,
        f"{len(avds)} AVD(s), {len(system_images)} system image(s)",
        details,
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


def run_doctor(xapk: Path | None, serial: str | None) -> list[CheckResult]:
    """Every check this project can run without a device-changing action.

    `xapk` is optional: the reference package is not required to characterize
    a host, which is what makes this the one command a fresh checkout runs
    first, before `local/*.xapk` is even in place.
    """
    checks = [check_host(), check_android_tools(), check_sdk_inventory()]
    if xapk is not None:
        checks.append(check_xapk(xapk))
    checks.append(check_device(serial))
    return checks


def render_json(results: list[CheckResult]) -> str:
    payload = {
        "ok": not any(result.status is CheckStatus.FAIL for result in results),
        "checks": [result.to_dict() for result in results],
    }
    return json.dumps(payload, indent=2, sort_keys=True)
