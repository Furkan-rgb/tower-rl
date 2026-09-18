"""Getting the instrumented bridge onto one instance, and its build identity.

`instrumented_bridge.sh` owns bridge deployment and its device-safety checks —
the `libunity.so` overlay mount, the refusal of the canonical AVD, the refusal
to deploy against an online instance. Nothing here reimplements any of that;
this is the one Python call into that script, and the identity of the build it
deploys, which every handshake is checked against.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tower_rl.simulation.instance import CloneInstance
from tower_rl.simulation.instrumented_bridge import BridgeCompatibility

#: The script that owns bridge deployment. It is a shell script beside the
#: entry points rather than package data because it is also run by hand, and
#: because the device-safety checks in it are meant to be read.
BRIDGE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "instrumented_bridge.sh"



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
    """
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
