"""Where adb and the emulator are on this host.

Tool discovery, not a host diagnostic: `doctor` reports on an SDK, but every
step that reaches an instance has to find the binary first, and the simulation
may not read back from the module that checks the host. So the two functions
that answer "where is it" live at the bottom of the simulation and `doctor`
reads them from here.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def sdk_roots() -> tuple[Path, ...]:
    """Where an Android SDK may live, configured roots first."""
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
    for root in sdk_roots():
        for relative in relative_candidates.get(name, ()):
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    return None
