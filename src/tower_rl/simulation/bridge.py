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
import re
import subprocess
from pathlib import Path

from tower_rl.environment.project_state import state_directory
from tower_rl.simulation.instance import PACKAGE, CloneError, CloneInstance, adb
from tower_rl.simulation.instrumented_bridge import BridgeCompatibility

#: The script that owns bridge deployment. It is a shell script beside the
#: entry points rather than package data because it is also run by hand, and
#: because the device-safety checks in it are meant to be read.
BRIDGE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "instrumented_bridge.sh"

#: Where `instrumented_bridge.sh deploy` leaves the bridge on the device. It is
#: the app-private copy that the mounted overlay loads, so it is the one file
#: whose digest says which bridge the game is actually running.
DEPLOYED_BRIDGE_PATH = f"/data/user/0/{PACKAGE}/files/libtower_bridge.so"

#: A read-back is a digest or it is nothing. `sha256sum` writes its own failures
#: to the same stream (`sha256sum: ...: No such file or directory`), and a
#: sampler that took the first word of whatever came back read that text as a
#: reading once already (`M1B-E047`).
DIGEST_PATTERN = re.compile("[0-9a-f]{64}")

#: Where this project keeps the bridge builds it can deploy: `state/bridge/`,
#: inside the checkout and git-ignored. The artifact is never committed and is
#: far too large to be, but it also cannot live in a build directory under
#: `/tmp`: a session's scratchpad is gone after a reboot, and a pointer into one
#: is how `/tmp/tower-bridge-live.latest` came to name a directory that no
#: longer existed. One directory per bridge, named for the SHA-256 of the
#: `libtower_bridge.so` inside it, with `current` a symlink to the one that is
#: deployed — so `ls -l` shows which bridge this project installs and what its
#: digest is without opening anything. `config/` beside them holds the private
#: build configuration, which is also never committed. A module-level path, so a
#: test can point it at a directory of its own.
BRIDGE_STATE_DIRECTORY = state_directory() / "bridge"


def artifact_digest(binary: Path) -> str:
    """The SHA-256 of one bridge artifact on this host."""
    try:
        return hashlib.sha256(binary.read_bytes()).hexdigest()
    except OSError as error:
        raise CloneError(f"cannot read the bridge artifact {binary}: {error}") from error


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
    digest = artifact_digest(binary)
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

    The override is therefore optional, and no command in this repository's
    documentation sets it: a runner with no `TOWER_BRIDGE_BUILD_DIR` in its
    environment finds the installed bridge, which is what an ordinary run
    wants. Set it only to run against a bridge you are building.
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


def deployed_bridge_read_back(instance: CloneInstance) -> str:
    """Whatever the device says about the bridge deployed on this instance.

    One form, and the only one: `adb shell "su -c 'sha256sum <path>'"`. The
    deployed bridge is app-private — mode 0555 under the package's own `files`
    directory, below a `/data/user/0/<package>` that only the app and root may
    traverse — so the read needs root. Which form of root is not a detail:
    `adb shell su 0 sha256sum <path>` returned nothing on every instance of two
    seven-actor fleets (`M1B-E046`, `M1B-E053`), and both had to record the
    deployed bridge as unconfirmed; the `su -c` form answered on all seven
    (`M1B-E048`) and is the form `instrumented_bridge.sh` has always used for
    its own `libunity.so` read-backs. Plain `adb shell sha256sum` answered once
    (`M1B-E056`) because that instance's adbd happened to be running as root,
    which is a property of how the emulator came up rather than of the check.

    Anything that is not a digest is no reading at all, so what came back is
    returned as it is and judged by the caller: the device's own words are what
    an operator needs when root is refused, and are not a digest either way.
    """
    output = adb(instance, "shell", f"su -c 'sha256sum {DEPLOYED_BRIDGE_PATH}'")
    return output.replace("\r", "").strip()


def confirm_deployed_bridge(instance: CloneInstance, expected: str) -> str:
    """Read the deployed bridge back and hold it to the artifact this host sent.

    The bridge on the device is what produced every observation a run records,
    so a run that cannot name it has no evidence about which bridge it measured.
    That is why all three outcomes below are failures rather than warnings: a
    read-back that said nothing, one that said something other than a digest,
    and a digest that disagrees are equally unable to say the device is running
    the artifact this host deployed.
    """
    read_back = deployed_bridge_read_back(instance)
    words = read_back.split()
    digest = words[0] if words else ""
    if not read_back:
        raise ActorFailure(
            f"{instance.serial}: no digest came back for {DEPLOYED_BRIDGE_PATH}, "
            f"so the deployed bridge cannot be confirmed as {expected}"
        )
    if not DIGEST_PATTERN.fullmatch(digest):
        raise ActorFailure(
            f"{instance.serial}: the read-back of {DEPLOYED_BRIDGE_PATH} is not a digest, "
            f"so the deployed bridge cannot be confirmed as {expected}; the device said: "
            f"{read_back.splitlines()[0][:200]}"
        )
    if digest != expected:
        raise ActorFailure(
            f"{instance.serial}: the deployed bridge is {digest}, "
            f"not the {expected} this host deployed"
        )
    print(f"{instance.serial} deploy: deployed bridge confirmed {digest}", flush=True)
    return digest


def deploy_bridge(instance: CloneInstance) -> None:
    """The bridge deployment step of the cold path, tagged into the fleet's log.

    Deployment and its confirmation are one step, not two a caller may take
    separately: the CLI and the fleet both reach the device through here, so
    there is one read-back and no path on which a bridge is deployed and never
    read back.

    The artifact is resolved and hashed first, which is also what runs the
    installed-bridge checks — a dangling `current`, or an artifact that does not
    hash to the directory name it is filed under. A bad install is then refused
    before anything is pushed, rather than after it is mounted over the game's
    own `libunity.so` and has to be cleaned up.
    """
    expected = artifact_digest(bridge_build_directory() / "libtower_bridge.so")
    run_bridge("deploy", instance)
    confirm_deployed_bridge(instance, expected)
