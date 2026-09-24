#!/usr/bin/env python3
"""Write the private build configuration from the installed bridge's own cache.

The private build configuration has never existed as a file: the values were
passed on a `cmake` command line by hand and survive only in the
`CMakeCache.txt` installed beside the deployed bridge, which is why a rebuild
could not be checked against the configuration it was supposed to reproduce.
Extracting them into `state/bridge/config/profile.cmake` makes the
`cmake -C state/bridge/config/profile.cmake` recipe in `docs/setup.md` work,
and nothing is committed by doing it: `state/` is ignored in full.

`docs/setup.md` documents rerunning this at any time to refresh
`profile.cmake` from whatever bridge is currently installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tower_rl.environment.project_state import state_directory  # noqa: E402

#: The cache entries that are the private build configuration: what game build
#: this bridge is for and what it must find there. Exactly the values
#: `native/tower_bridge/README.md` calls private — the package and profile
#: identifiers, the official signer, and the two library digests. The rest of
#: the cache (Unity version, metadata version, bridge version, and every
#: toolchain entry CMake writes for itself) is either public in `CMakeLists.txt`
#: or belongs to the build tree it came from, and copying it forward would
#: pin a new build to an old NDK.
PRIVATE_CACHE_ENTRIES = (
    "TOWER_BRIDGE_PACKAGE_VERSION",
    "TOWER_BRIDGE_PACKAGE_VERSION_CODE",
    "TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256",
    "TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256",
    "TOWER_BRIDGE_LIBIL2CPP_SHA256",
    "TOWER_BRIDGE_PROFILE_ID",
)


class MigrationRefused(Exception):
    """The write was not attempted, and the reason is in the message."""


def cache_entries(cache: Path) -> dict[str, tuple[str, str]]:
    """Each `NAME:TYPE=value` line of a `CMakeCache.txt`, by name.

    A cache line is `NAME:TYPE=value`; comments and blank lines are not, and
    `NAME-ADVANCED:INTERNAL=1` entries are CMake's own bookkeeping about the
    entries rather than entries themselves.
    """
    found: dict[str, tuple[str, str]] = {}
    for line in cache.read_text().splitlines():
        if line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        declaration, value = line.split("=", 1)
        name, _, kind = declaration.partition(":")
        if name and kind:
            found[name] = (kind, value)
    return found


def write_private_build_configuration(bridge: Path) -> Path | None:
    """Write `config/profile.cmake` from the installed build's own cache.

    Returns the file written, or `None` when this host has no installed bridge
    to read a configuration out of — a checkout that never deployed one has
    nothing to recover and is not an error.

    Every named entry must be present. A configuration missing one of them would
    produce a bridge that compiles and then answers a compatibility error
    instead of a handshake, and the entry it is missing is named rather than
    left to a failed deployment to discover.
    """
    cache = bridge / "current" / "CMakeCache.txt"
    if not cache.is_file():
        return None
    entries = cache_entries(cache)
    missing = [name for name in PRIVATE_CACHE_ENTRIES if name not in entries]
    if missing:
        raise MigrationRefused(
            f"the installed build's cache {cache} records no {', '.join(missing)}; "
            "the private build configuration cannot be written from it"
        )
    lines = [
        "# The private build configuration of the installed bridge, extracted from",
        f"# {cache} by scripts/migrate_state.py. Never committed.",
        "# Configure a rebuild with: cmake -C state/bridge/config/profile.cmake ...",
        "",
    ]
    for name in PRIVATE_CACHE_ENTRIES:
        kind, value = entries[name]
        lines.append(f'set({name} "{value}" CACHE {kind} "")')
    profile = bridge / "config" / "profile.cmake"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("\n".join(lines) + "\n")
    return profile


def main() -> int:
    destination = state_directory()
    try:
        profile = write_private_build_configuration(destination / "bridge")
    except MigrationRefused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 1
    if profile is None:
        print("no installed bridge: no private build configuration to write")
    else:
        print(f"private build configuration written to {profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
