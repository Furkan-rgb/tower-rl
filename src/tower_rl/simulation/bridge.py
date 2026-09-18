"""Getting the instrumented bridge onto one instance, and its build identity.

`instrumented_bridge.sh` owns bridge deployment and its device-safety checks —
the `libunity.so` overlay mount, the refusal of the canonical AVD, the refusal
to deploy against an online instance. Nothing here reimplements any of that;
this is the one Python call into that script, the artifact that call deploys,
and the identity of the build it comes from, which every handshake is checked
against.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from tower_rl.simulation.instance import CloneError, CloneInstance
from tower_rl.simulation.instrumented_bridge import BridgeCompatibility

#: The script that owns bridge deployment. It is a shell script beside the
#: entry points rather than package data because it is also run by hand, and
#: because the device-safety checks in it are meant to be read.
BRIDGE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "instrumented_bridge.sh"

#: Where this host keeps the bridge builds it can deploy. The artifact is never
#: committed and is far too large to be, but it also cannot live in a build
#: directory under `/tmp`: a session's scratchpad is gone after a reboot, and a
#: pointer into one is how `/tmp/tower-bridge-live.latest` came to name a
#: directory that no longer existed. One directory per bridge, named for the
#: SHA-256 of the `libtower_bridge.so` inside it, with `current` a symlink to
#: the one that is deployed — so `ls -l` shows which bridge this host installs
#: and what its digest is without opening anything. A module-level path, so a
#: test can point it at a directory of its own.
BRIDGE_STATE_DIRECTORY = (
    Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    / "tower-rl"
    / "bridge"
)



def installed_bridge_directory() -> Path:
    """The installed bridge this host deploys, resolved through `current`.

    Two things are checked here and nowhere else, because a bridge that is not
    the one we think it is fails much later and much less clearly — at a
    handshake, or as a digest read back off a device that no longer matches
    anything on the host. The pointer must resolve to a directory that exists,
    and the artifact inside it must hash to the name of that directory, which is
    what makes the layout self-verifying: the name is the claim, the bytes are
    the evidence, and an installed build that was truncated, half-copied or
    overwritten in place cannot pass as the digest it is filed under.
    """
    current = BRIDGE_STATE_DIRECTORY / "current"
    if not current.is_symlink() and not current.exists():
        raise CloneError(
            f"no bridge is installed: {current} does not exist "
            "(install one under its own SHA-256 and point `current` at it)"
        )
    resolved = current.resolve()
    if not resolved.is_dir():
        raise CloneError(
            f"the installed-bridge pointer {current} dangles: it names {resolved}, "
            "which is not a directory"
        )
    binary = resolved / "libtower_bridge.so"
    try:
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    except OSError as error:
        raise CloneError(f"cannot read the installed bridge {binary}: {error}") from error
    if digest != resolved.name:
        raise CloneError(
            f"the installed bridge {binary} hashes to {digest}, not to the "
            f"{resolved.name} its directory name records"
        )
    return resolved


def bridge_build_directory() -> Path:
    """Where the bridge this host deploys is: an override, or what is installed.

    `TOWER_BRIDGE_BUILD_DIR` still wins, because a bridge is developed by
    building it and deploying it straight out of the build tree. Everything
    else — every fleet run, every `clone_session.py up` — takes the installed
    one, which survives a reboot and says what it is.
    """
    configured = os.environ.get("TOWER_BRIDGE_BUILD_DIR")
    if configured:
        return Path(configured)
    return installed_bridge_directory()


def compatibility(build_dir: Path) -> BridgeCompatibility:
    """Read the private build's configured identity; never hard-coded here."""
    cache = {
        key: value
        for line in (build_dir / "CMakeCache.txt").read_text().splitlines()
        if ":STRING=" in line
        for key, value in [line.split(":STRING=", 1)]
    }
    return BridgeCompatibility(
        package_version=cache["TOWER_BRIDGE_PACKAGE_VERSION"],
        package_version_code=int(cache["TOWER_BRIDGE_PACKAGE_VERSION_CODE"]),
        official_signer_sha256=cache["TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256"],
        original_libunity_sha256=cache["TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256"],
        libil2cpp_sha256=cache["TOWER_BRIDGE_LIBIL2CPP_SHA256"],
        unity_version=cache["TOWER_BRIDGE_UNITY_VERSION"],
        il2cpp_metadata_version=int(cache["TOWER_BRIDGE_METADATA_VERSION"]),
        bridge_version=cache["TOWER_BRIDGE_VERSION"],
        profile_id=cache["TOWER_BRIDGE_PROFILE_ID"],
    )


class ActorFailure(RuntimeError):
    """A step of one actor's lifecycle failed; only that actor is affected."""


def run_bridge(command: str, instance: CloneInstance) -> None:
    """Deploy or clean up the instrumented bridge on one instance.

    What the script prints is the device-safety evidence itself: the `libunity.so`
    hash it re-verified against the original, the package identity, the number of
    live mounts, and whether the bridge artifacts are gone. Cleanup that reports
    nothing is indistinguishable from cleanup that verified nothing, so the output
    is relayed to the operator's log rather than kept for an error path that a
    successful cleanup never takes. Every line is tagged, since N actors clean up
    at once and an unattributed identity report verifies no particular instance.

    `BRIDGE_SCRIPT` is derived from this file's own location, so an installed or
    relocated package can point it at nothing. That is checked here rather than
    left to `subprocess`, whose `FileNotFoundError` names a path and no reason.
    """
    if not BRIDGE_SCRIPT.is_file():
        raise ActorFailure(
            f"the bridge deployment script is missing: {BRIDGE_SCRIPT} "
            "(it is derived from the package location and must sit in scripts/)"
        )
    result = subprocess.run(
        [
            str(BRIDGE_SCRIPT),
            command,
            instance.serial,
            str(instance.bridge_host_port),
        ],
        capture_output=True,
        text=True,
    )
    for marker, text in (("", result.stdout), ("error: ", result.stderr)):
        for line in text.splitlines():
            print(f"{instance.serial} {command}: {marker}{line}", flush=True)
    if result.returncode != 0:
        raise ActorFailure(
            f"instrumented_bridge.sh {command} failed on {instance.serial}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def deploy_bridge(instance: CloneInstance) -> None:
    """The bridge deployment step of the cold path, tagged into the fleet's log."""
    run_bridge("deploy", instance)
