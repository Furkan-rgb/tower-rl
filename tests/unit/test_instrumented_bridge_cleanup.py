"""What `instrumented_bridge.sh` reports when a post-condition or a reading fails.

Mostly `cleanup`, which is the step that decides whether an instance was left as
it was found; `verify` appears once, for the one thing it does decide.

Cleanup is the one step that decides whether an instance was left as it was
found, and its readbacks are only worth having if a caller can act on them.
Everything the script reaches for is one `adb` binary, so a stub `adb` under a
stub SDK root is the whole device here: it answers the readings from files the
test sets, and records every invocation so the ones that come *after* a failed
check can be shown to have run anyway.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
BRIDGE = REPOSITORY / "scripts" / "instrumented_bridge.sh"

ORIGINAL_LIBUNITY_SHA256 = "ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040"
DEPLOYED_LIBUNITY_SHA256 = "a" * 64

CMAKE_CACHE = f"""# This is the CMakeCache file.
TOWER_BRIDGE_PACKAGE_VERSION:STRING=29.0.3
TOWER_BRIDGE_PACKAGE_VERSION_CODE:STRING=1199
TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256:STRING={ORIGINAL_LIBUNITY_SHA256}
"""

ADB_STUB = """#!/usr/bin/env bash
set -uo pipefail
state="$TOWER_STUB_STATE"
echo "$*" >> "$state/calls"
reading() { cat "$state/$1"; }
case "$*" in
  *"emu avd name"*) echo "tower_rl_instrumented_api36" ;;
  *"shell pm path"*) echo "package:/data/app/~~ab==/com.TechTreeGames.TheTower-1/base.apk" ;;
  *"sha256sum"*) echo "$(reading libunity_sha256)  /lib/arm64/libunity.so" ;;
  *"am force-stop"*) ;;
  *"cmd game reset"*) ;;
  *"dumpsys package"*)
    echo "    userId=10123"
    echo "    versionName=$(reading version_name)"
    echo "    versionCode=$(reading version_code) minSdk=24 targetSdk=35"
    echo "    installerPackageName=$(reading installer)"
    ;;
  *"dumpsys SurfaceFlinger"*)
    echo "  GameModeFrameRateOverride {10123, $(reading override) 60}"
    ;;
  *umount*) ;;
  *"grep -c libunity.so"*) reading mounts ;;
  *"rm -f"*) ;;
  *"forward --remove"*) ;;
  *"bridge_artifacts: none"*)
    [ -e "$state/artifacts_survive" ] || echo "bridge_artifacts: none"
    ;;
  *"test ! -e"*)
    [ -e "$state/artifacts_survive" ] || echo "bridge_artifacts: removed"
    ;;
  *"shell ip -o -4 addr show"*) echo "1: lo    inet 127.0.0.1/8 scope host lo" ;;
  *) echo "stub adb: unexpected $*" >&2; exit 1 ;;
esac
"""


@dataclass(frozen=True)
class Device:
    """A stub instance, and the readings cleanup will take from it."""

    environment: dict[str, str]
    state: Path

    def reads(self, **readings: str) -> None:
        for name, value in readings.items():
            (self.state / name).write_text(value)

    @property
    def calls(self) -> str:
        return (self.state / "calls").read_text()

    def run(self, command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(BRIDGE), command, "emulator-5556"],
            env=self.environment, capture_output=True, text=True, timeout=60, check=False,
        )

    def cleanup(self) -> subprocess.CompletedProcess[str]:
        return self.run("cleanup")

    def verify(self) -> subprocess.CompletedProcess[str]:
        return self.run("verify")


@pytest.fixture
def device(tmp_path: Path) -> Device:
    """A device that reads back exactly as a finished cleanup should."""
    platform_tools = tmp_path / "sdk" / "platform-tools"
    platform_tools.mkdir(parents=True)
    adb = platform_tools / "adb"
    adb.write_text(ADB_STUB)
    adb.chmod(0o755)

    build = tmp_path / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(CMAKE_CACHE)

    state = tmp_path / "state"
    state.mkdir()
    (state / "calls").write_text("")

    environment = dict(os.environ)
    environment.update(
        ANDROID_SDK_ROOT=str(tmp_path / "sdk"),
        TOWER_BRIDGE_BUILD_DIR=str(build),
        TOWER_STUB_STATE=str(state),
    )
    device = Device(environment=environment, state=state)
    device.reads(
        libunity_sha256=ORIGINAL_LIBUNITY_SHA256,
        version_name="29.0.3",
        version_code="1199",
        installer="com.android.vending",
        override="0",
        mounts="0",
    )
    return device


def test_a_cleanup_whose_checks_all_pass_exits_zero(device: Device) -> None:
    result = device.cleanup()

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"libunity_sha256: {ORIGINAL_LIBUNITY_SHA256}" in result.stdout
    assert "versionCode=1199" in result.stdout
    assert "installerPackageName=com.android.vending" in result.stdout
    assert "libunity_mounts: 0" in result.stdout
    assert "game_frame_rate_override: reset" in result.stdout
    assert "bridge_artifacts: removed" in result.stdout
    assert "cleanup_checks: all passed" in result.stdout


def test_an_overlay_that_will_not_unmount_fails_after_every_step_has_run(device: Device) -> None:
    # The overlay is still mounted and the target still reads as the bridge's
    # view of it, which is what `unmount_overlay` gives up on.
    device.reads(mounts="1", libunity_sha256=DEPLOYED_LIBUNITY_SHA256)

    result = device.cleanup()

    assert result.returncode != 0
    assert "the overlay is still mounted" in result.stderr
    assert f"libunity_sha256 on emulator-5556 is {DEPLOYED_LIBUNITY_SHA256}" in result.stderr
    assert "1 libunity.so mount(s) survive" in result.stderr
    # Everything after the failed unmount still ran: the artifacts were removed,
    # the forward was withdrawn, and identity was read back and reported.
    assert "rm -f" in device.calls
    assert "forward --remove" in device.calls
    assert "libunity_mounts: 1" in result.stdout
    assert "bridge_artifacts: removed" in result.stdout
    assert "cleanup_checks: 3 failed on emulator-5556" in result.stderr


def test_a_frame_rate_override_that_did_not_reset_fails_the_cleanup(device: Device) -> None:
    device.reads(override="120")

    result = device.cleanup()

    assert result.returncode != 0
    assert "game_frame_rate_override: NOT-reset, still 120" in result.stderr
    assert "the game frame-rate override on emulator-5556 is still 120" in result.stderr
    # The rest of the cleanup is unaffected: only the exit status changed.
    assert "bridge_artifacts: removed" in result.stdout
    assert "cleanup_checks: 1 failed on emulator-5556" in result.stderr


def test_a_surviving_bridge_artifact_fails_the_cleanup(device: Device) -> None:
    (device.state / "artifacts_survive").touch()

    result = device.cleanup()

    assert result.returncode != 0
    assert "bridge artifacts survive on emulator-5556" in result.stderr


def test_an_installer_that_is_not_play_fails_the_cleanup(device: Device) -> None:
    device.reads(installer="com.example.sideload")

    result = device.cleanup()

    assert result.returncode != 0
    assert "installerPackageName is com.example.sideload, not com.android.vending" in result.stderr


def test_verify_fails_when_a_reading_comes_back_empty(device: Device) -> None:
    """A blank reading is not a confirmation, and must not be reported as one."""
    assert device.verify().returncode == 0
    device.reads(version_code="")

    result = device.verify()

    assert result.returncode != 0
    assert "versionCode read back empty on emulator-5556" in result.stderr
    assert "verify_checks: 1 failed on emulator-5556" in result.stderr
