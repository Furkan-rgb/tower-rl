"""Writing the private build configuration from the installed bridge's cache."""

from __future__ import annotations

import subprocess
from pathlib import Path

import migrate_state
import pytest
from migrate_state import MigrationRefused, write_private_build_configuration

#: A cache as CMake writes one: the private entries among the toolchain's own,
#: with comments, a blank line and an `-ADVANCED` bookkeeping entry in it.
CMAKE_CACHE = """\
# This is the CMakeCache file.
//Official package version
TOWER_BRIDGE_PACKAGE_VERSION:STRING=29.0.3
TOWER_BRIDGE_PACKAGE_VERSION_CODE:STRING=1199
TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256:STRING=aa11
TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256:STRING=ffc1
TOWER_BRIDGE_LIBIL2CPP_SHA256:STRING=bb22
TOWER_BRIDGE_UNITY_VERSION:STRING=6000.3.15f1
TOWER_BRIDGE_METADATA_VERSION:STRING=39
TOWER_BRIDGE_PROFILE_ID:STRING=instrumented-training-v1
TOWER_BRIDGE_DIAGNOSTICS:BOOL=OFF

CMAKE_ANDROID_NDK:PATH=/home/somebody/.local/share/android-sdk/ndk/29.0.14206865
CMAKE_ANDROID_NDK-ADVANCED:INTERNAL=1
"""


def test_the_private_build_configuration_is_written_from_the_installed_cache(
    tmp_path: Path,
) -> None:
    """The `-C` file `docs/setup.md` configures a rebuild with, ready to use.

    Exactly the private entries, as `set(... CACHE <TYPE> "")` lines: the values
    that say which game build this bridge is for. Not the toolchain's own
    entries, which would pin a new build to the NDK path of an old one.
    """
    bridge = tmp_path / "state" / "bridge"
    (bridge / "abc123").mkdir(parents=True)
    (bridge / "abc123" / "CMakeCache.txt").write_text(CMAKE_CACHE)
    (bridge / "current").symlink_to("abc123")

    profile = write_private_build_configuration(bridge)

    assert profile == bridge / "config" / "profile.cmake"
    written = profile.read_text()
    assert 'set(TOWER_BRIDGE_PACKAGE_VERSION "29.0.3" CACHE STRING "")' in written
    assert 'set(TOWER_BRIDGE_PACKAGE_VERSION_CODE "1199" CACHE STRING "")' in written
    assert 'set(TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256 "aa11" CACHE STRING "")' in written
    assert 'set(TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256 "ffc1" CACHE STRING "")' in written
    assert 'set(TOWER_BRIDGE_LIBIL2CPP_SHA256 "bb22" CACHE STRING "")' in written
    assert 'set(TOWER_BRIDGE_PROFILE_ID "instrumented-training-v1" CACHE STRING "")' in written
    assert "CMAKE_ANDROID_NDK" not in written
    assert "TOWER_BRIDGE_UNITY_VERSION" not in written, "public, and in CMakeLists.txt"


def test_a_cache_missing_a_private_entry_is_refused_by_name(tmp_path: Path) -> None:
    """A bridge built without one of these compiles and then refuses a handshake."""
    bridge = tmp_path / "state" / "bridge"
    (bridge / "abc123").mkdir(parents=True)
    (bridge / "abc123" / "CMakeCache.txt").write_text(
        CMAKE_CACHE.replace("TOWER_BRIDGE_PROFILE_ID:STRING=instrumented-training-v1\n", "")
    )
    (bridge / "current").symlink_to("abc123")

    with pytest.raises(MigrationRefused, match="TOWER_BRIDGE_PROFILE_ID"):
        write_private_build_configuration(bridge)


def test_a_host_with_no_installed_bridge_has_no_configuration_to_recover(
    tmp_path: Path,
) -> None:
    assert write_private_build_configuration(tmp_path / "state" / "bridge") is None


def test_the_private_build_configuration_is_git_ignored(tmp_path: Path) -> None:
    """It is a secret in a public repository: `state/` must cover it, not luck."""
    repository = Path(__file__).resolve().parents[2]
    ignored = subprocess.run(
        ["git", "check-ignore", "state/bridge/config/profile.cmake", "state/logs"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored.returncode == 0
    assert ignored.stdout.split() == ["state/bridge/config/profile.cmake", "state/logs"]


def test_the_command_writes_the_configuration_and_prints_where(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`main` is the write step alone now: no tree to move, no emulator to check."""
    destination = tmp_path / "repo" / "state"
    bridge = destination / "bridge"
    (bridge / "abc123").mkdir(parents=True)
    (bridge / "abc123" / "CMakeCache.txt").write_text(CMAKE_CACHE)
    (bridge / "current").symlink_to("abc123")
    monkeypatch.setattr(migrate_state, "state_directory", lambda: destination)

    assert migrate_state.main() == 0

    printed = capsys.readouterr().out
    profile = bridge / "config" / "profile.cmake"
    assert f"private build configuration written to {profile}" in printed
    assert "TOWER_BRIDGE_PROFILE_ID" in profile.read_text()


def test_the_command_reports_nothing_to_write_with_no_installed_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(migrate_state, "state_directory", lambda: tmp_path / "state")

    assert migrate_state.main() == 0
    assert "no installed bridge" in capsys.readouterr().out
