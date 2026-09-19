"""The one-shot move of a host's former state tree into the project.

Every case is played against `tmp_path` trees, never the real one: the move is
destructive by design — it renames a tree and removes what it came from — and a
test that touched the host's own state would be indistinguishable from running
the migration for real.
"""

from __future__ import annotations

import os
from pathlib import Path

import migrate_state
import pytest
from migrate_state import MigrationRefused, migrate, running_emulators


def former_tree(root: Path) -> Path:
    """A state tree shaped like the one this host kept under the home directory."""
    source = root / "tower-rl"
    (source / "bridge" / "abc123").mkdir(parents=True)
    (source / "bridge" / "abc123" / "libtower_bridge.so").write_bytes(b"bridge")
    (source / "bridge" / "config").mkdir()
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
    assert (destination / "bridge" / "config").is_dir()
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
