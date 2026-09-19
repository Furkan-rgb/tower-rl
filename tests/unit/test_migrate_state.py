"""The one-shot move of a host's former state tree into the project.

Every case is played against `tmp_path` trees, never the real one: the move is
destructive by design — it renames a tree and removes what it came from — and a
test that touched the host's own state would be indistinguishable from running
the migration for real.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import migrate_state
import pytest
from migrate_state import (
    MigrationRefused,
    migrate,
    running_emulators,
    write_private_build_configuration,
)

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


def former_tree(root: Path) -> Path:
    """A state tree shaped like the one this host kept under the home directory."""
    source = root / "tower-rl"
    (source / "bridge" / "abc123").mkdir(parents=True)
    (source / "bridge" / "abc123" / "libtower_bridge.so").write_bytes(b"bridge")
    (source / "bridge" / "abc123" / "CMakeCache.txt").write_text(CMAKE_CACHE)
    (source / "bridge" / "current").symlink_to(source / "bridge" / "abc123")
    (source / "runs" / "session-1").mkdir(parents=True)
    (source / "runs" / "session-1" / "report.json").write_text("{}")
    (source / "mlflow.db").write_text("store")
    return source


def test_every_entry_moves_and_the_old_location_is_gone(tmp_path: Path) -> None:
    """One place to look for state afterwards, which is the point of the move."""
    source = former_tree(tmp_path / "home")
    destination = tmp_path / "repo" / "state"

    moved = migrate(source, destination)

    assert not source.exists()
    assert (destination / "runs" / "session-1" / "report.json").read_text() == "{}"
    assert (destination / "mlflow.db").read_text() == "store"
    assert (destination / "bridge" / "abc123" / "CMakeCache.txt").read_text() == CMAKE_CACHE
    assert [(old.name, new.name) for old, new in moved] == [
        ("bridge", "bridge"),
        ("mlflow.db", "mlflow.db"),
        ("runs", "runs"),
    ]


def test_the_current_bridge_symlink_comes_back_relative(tmp_path: Path) -> None:
    """It named an absolute path outside the project; it names its sibling now."""
    source = former_tree(tmp_path / "home")
    destination = tmp_path / "repo" / "state"

    migrate(source, destination)

    current = destination / "bridge" / "current"
    assert os.readlink(current) == "abc123"
    assert current.resolve() == destination / "bridge" / "abc123"


def test_a_state_directory_that_already_has_content_is_refused(tmp_path: Path) -> None:
    """Two trees are never merged into each other: this move happens once."""
    source = former_tree(tmp_path / "home")
    destination = tmp_path / "repo" / "state"
    (destination / "runs").mkdir(parents=True)

    with pytest.raises(MigrationRefused, match="already has content"):
        migrate(source, destination)

    assert (source / "mlflow.db").exists(), "nothing moved before the refusal"


def test_an_empty_state_directory_is_not_content(tmp_path: Path) -> None:
    """A checkout where something already made `state/` is still a first run."""
    source = former_tree(tmp_path / "home")
    destination = tmp_path / "repo" / "state"
    destination.mkdir(parents=True)

    assert migrate(source, destination)


def test_a_missing_former_location_is_nothing_to_do(tmp_path: Path) -> None:
    """Running it twice is harmless; the second run has nothing to move."""
    assert migrate(tmp_path / "absent", tmp_path / "state") == []
    assert not (tmp_path / "state").exists()


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


def test_a_running_emulator_is_read_from_proc_and_not_from_a_command_line(
    tmp_path: Path,
) -> None:
    """A live instance holds the bridge and the run directory open; refuse then.

    The reading is `/proc/<pid>/exe`, so a process that merely names the
    emulator in its arguments is not one, and a directory that is not a pid is
    not looked at at all.
    """
    proc = tmp_path / "proc"
    for pid, executable in (("41", "/usr/bin/bash"), ("42", "/opt/sdk/qemu-system-x86_64")):
        (proc / pid).mkdir(parents=True)
        (proc / pid / "exe").symlink_to(executable)
    (proc / "self").mkdir()

    assert running_emulators(proc) == ["42 /opt/sdk/qemu-system-x86_64"]
    assert running_emulators(tmp_path / "absent") == []


def test_the_command_refuses_while_an_emulator_is_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def moved(source: Path, destination: Path) -> list[tuple[Path, Path]]:
        raise AssertionError("the tree must not be touched while an emulator runs")

    monkeypatch.setattr(migrate_state, "running_emulators", lambda: ["42 qemu-system-x86_64"])
    monkeypatch.setattr(migrate_state, "migrate", moved)

    assert migrate_state.main() == 1
    assert "refusing to move project state" in capsys.readouterr().err


def test_the_mapping_is_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """What moved where, so the operator can see it rather than infer it."""
    source = former_tree(tmp_path / "home")
    destination = tmp_path / "repo" / "state"
    monkeypatch.setattr(migrate_state, "running_emulators", lambda: [])
    monkeypatch.setattr(migrate_state, "FORMER_STATE_DIRECTORY", source)
    monkeypatch.setattr(migrate_state, "state_directory", lambda: destination)

    assert migrate_state.main() == 0

    printed = capsys.readouterr().out
    assert f"{source / 'runs'} -> {destination / 'runs'}" in printed
    assert f"{source} is gone" in printed
    profile = destination / "bridge" / "config" / "profile.cmake"
    assert f"private build configuration written to {profile}" in printed
    assert "TOWER_BRIDGE_PROFILE_ID" in profile.read_text()
