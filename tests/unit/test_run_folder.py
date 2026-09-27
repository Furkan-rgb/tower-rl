"""One folder per training run, resumed in place as segments (board #90).

Everything a run writes - manifest, checkpoints, replay, each segment's summary
and log, the stage logs pointed there and the evaluations of its checkpoints -
lives under `state/runs/<run name>/`. These tests train against the fake port,
so no device is involved. Where an evaluation of a checkpoint is filed is the
checkpoint arm's concern and is tested in `test_checkpoint_arm.py`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_train_entry_point import (
    RESUME_PERIOD,
    latest_checkpoint,
    numbered,
    resume_from,
    session,
)

from tower_rl.experiment.run_folder import run_ids
from tower_rl.experiment.training_report import REPLAY_BACKUP_DIRECTORY, REPLAY_DIRECTORY
from tower_rl.learning.checkpoint import load
from tower_rl.learning.replay import PrioritizedSequenceReplay, read_replay_metadata


def manifest_of(folder: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((folder / "manifest.json").read_text())
    return manifest


def resumed(run_dir: Path, checkpoint: Path, budget: int, **flags: str) -> dict[str, Any]:
    """A second segment from `checkpoint`, as `main` runs it: parsed, checked, trained."""
    return session(
        run_dir,
        budget=str(budget),
        settings={
            "--checkpoint-every-decisions": str(RESUME_PERIOD),
            "--resume": str(checkpoint),
            **flags,
        },
        resume=resume_from(run_dir, checkpoint, budget),
    )


def test_a_fresh_run_keeps_everything_in_one_folder_named_for_the_backbone_and_the_time(
    tmp_path: Path,
) -> None:
    report = numbered(tmp_path, 200)
    folder = Path(report["run_folder"])

    assert folder.parent == tmp_path
    assert folder.name.startswith("stacked-dqn-") and folder.name.endswith("Z")
    assert report["segment"] == 1
    assert (folder / "checkpoints" / "latest.pt").exists()
    assert sorted(folder.glob("checkpoints/checkpoint-d*.pt"))
    assert (folder / REPLAY_DIRECTORY).is_dir()
    summary = json.loads((folder / "segments" / "1" / "summary.json").read_text())
    assert summary["arm"]["run_id"] == report["arm"]["run_id"]
    log = (folder / "segments" / "1" / "train.log").read_text()
    assert f"run folder {folder}, segment 1" in log
    assert "[stacked-dqn] episode 1 decisions" in log
    manifest = manifest_of(folder)
    (segment,) = manifest["segments"]
    assert segment["segment"] == 1
    assert segment["run_id"] == report["arm"]["run_id"]
    assert segment["parent_checkpoint"] is None
    assert segment["resumed_from_decisions"] == 0
    # The upgrade setup of #89, in the one manifest.
    assert manifest["upgrade_setup_digest"]
    assert run_ids(folder) == [report["arm"]["run_id"]]
    # Nothing outside the folder: no session directory beside it.
    assert list(tmp_path.iterdir()) == [folder]


def test_a_named_run_is_written_where_its_name_says(tmp_path: Path) -> None:
    report = numbered(tmp_path, 200, **{"--run-name": "seed-1"})

    assert Path(report["run_folder"]) == tmp_path / "seed-1"


def test_a_folder_that_holds_a_run_is_not_started_again(tmp_path: Path) -> None:
    numbered(tmp_path, 200, **{"--run-name": "seed-1"})

    with pytest.raises(SystemExit, match="already holds a run"):
        numbered(tmp_path, 200, **{"--run-name": "seed-1"})


def test_a_resume_from_latest_continues_in_the_same_folder_as_a_second_segment(
    tmp_path: Path,
) -> None:
    first = numbered(tmp_path, 200)
    folder = Path(first["run_folder"])
    latest = latest_checkpoint(first)
    first_decisions = first["arm"]["decisions"]
    first_numbered = sorted(folder.glob("checkpoints/checkpoint-d*.pt"))

    second = resumed(tmp_path, latest, 400)

    assert Path(second["run_folder"]) == folder
    assert second["segment"] == 2
    assert list(tmp_path.iterdir()) == [folder]
    # Both segments' own summary and log, the first left as it was.
    first_summary = json.loads((folder / "segments" / "1" / "summary.json").read_text())
    assert first_summary["arm"]["decisions"] == first_decisions
    second_summary = json.loads((folder / "segments" / "2" / "summary.json").read_text())
    assert second_summary["arm"]["decisions"] >= 400
    assert "segment 2" in (folder / "segments" / "2" / "train.log").read_text()
    # One manifest, listing both, the second naming what it continued.
    manifest = manifest_of(folder)
    assert [entry["segment"] for entry in manifest["segments"]] == [1, 2]
    assert manifest["segments"][1]["run_id"] == second["arm"]["run_id"]
    assert manifest["segments"][1]["parent_checkpoint"].startswith(str(latest))
    assert manifest["segments"][1]["resumed_from_decisions"] == first_decisions
    assert manifest["run_id"] == second["arm"]["run_id"]
    assert manifest["upgrade_setup_digest"]
    assert run_ids(folder) == [first["arm"]["run_id"], second["arm"]["run_id"]]
    # The first segment's candidates are still there beside the second's.
    later = sorted(folder.glob("checkpoints/checkpoint-d*.pt"))
    assert set(first_numbered) < set(later)
    # The replay the second segment reloaded is replaced by the one it ended with,
    # at the decision count of the latest.pt beside it.
    restored = second["arm"]["resolved_config"]["replay_restored_from"]
    assert restored == str(folder / REPLAY_DIRECTORY)
    decisions = load(latest).progress.environment_decisions
    assert decisions == second["arm"]["decisions"]
    assert read_replay_metadata(folder / REPLAY_DIRECTORY)["run"]["decisions"] == decisions


def test_a_resume_from_a_numbered_checkpoint_branches_into_a_new_folder(
    tmp_path: Path,
) -> None:
    first = numbered(tmp_path / "runs", 300)
    folder = Path(first["run_folder"])
    earliest = sorted(folder.glob("checkpoints/checkpoint-d*.pt"))[0]
    # A numbered checkpoint is not the point the saved replay describes.
    shutil.rmtree(folder / REPLAY_DIRECTORY)
    before = manifest_of(folder)

    branch = resumed(tmp_path / "runs", earliest, 400, **{"--run-name": "branch"})

    assert Path(branch["run_folder"]) == tmp_path / "runs" / "branch"
    assert manifest_of(folder) == before
    (segment,) = manifest_of(tmp_path / "runs" / "branch")["segments"]
    assert segment["parent_checkpoint"].startswith(str(earliest))


def test_a_resume_from_a_numbered_checkpoint_does_not_write_into_its_folder(
    tmp_path: Path,
) -> None:
    first = numbered(tmp_path, 300)
    folder = Path(first["run_folder"])
    earliest = sorted(folder.glob("checkpoints/checkpoint-d*.pt"))[0]
    shutil.rmtree(folder / REPLAY_DIRECTORY)

    with pytest.raises(SystemExit, match="already holds a run"):
        resumed(tmp_path, earliest, 400, **{"--run-name": folder.name})


def test_a_checkpoint_of_the_old_layout_still_resumes_into_a_new_folder(
    tmp_path: Path,
) -> None:
    """`state/runs/session-*/<run id>/`, as every run before #90 left it."""
    first = numbered(tmp_path / "written", 200)
    written = Path(first["run_folder"])
    # Rebuilt as the old layout: the folder named for its run id inside a
    # session directory, a manifest with no segments, the summary beside it.
    old = tmp_path / "runs" / "session-20260920-120000" / first["arm"]["run_id"]
    old.parent.mkdir(parents=True)
    shutil.move(written, old)
    shutil.rmtree(old / "segments")
    manifest = manifest_of(old)
    del manifest["segments"]
    (old / "manifest.json").write_text(json.dumps(manifest))
    (old / "summary.json").write_text(json.dumps(first["arm"]))
    untouched = {
        path.relative_to(old): path.read_bytes()
        for path in old.rglob("*")
        if path.is_file()
    }
    latest = old / "checkpoints" / "latest.pt"

    second = resumed(tmp_path / "runs", latest, 400)

    folder = Path(second["run_folder"])
    assert folder.parent == tmp_path / "runs"
    assert folder.name.startswith("stacked-dqn-")
    assert second["arm"]["decisions"] >= 400
    # The saved replay beside the old checkpoint was reloaded, not moved.
    assert second["arm"]["resolved_config"]["replay_restored_from"] == str(old / REPLAY_DIRECTORY)
    (segment,) = manifest_of(folder)["segments"]
    assert segment["parent_checkpoint"].startswith(str(latest))
    assert segment["resumed_from_decisions"] == first["arm"]["decisions"]
    # And the old folder is exactly as it was.
    assert {
        path.relative_to(old): path.read_bytes()
        for path in old.rglob("*")
        if path.is_file()
    } == untouched
    assert run_ids(old) == [first["arm"]["run_id"]]


def dump_bytes(dump: Path) -> dict[Path, bytes]:
    return {path.relative_to(dump): path.read_bytes() for path in dump.rglob("*") if path.is_file()}


def test_a_replay_save_that_fails_leaves_the_earlier_dump_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = numbered(tmp_path, 200)
    folder = Path(first["run_folder"])
    dump = folder / REPLAY_DIRECTORY
    before = dump_bytes(dump)

    def fails(self: PrioritizedSequenceReplay, directory: Path, **_: Any) -> int:
        directory.mkdir()
        (directory / "partial").write_bytes(b"half a dump")
        raise OSError("disk full")

    monkeypatch.setattr(PrioritizedSequenceReplay, "save_to", fails)
    resumed(tmp_path, latest_checkpoint(first), 400)

    assert dump_bytes(dump) == before
    assert not (folder / REPLAY_BACKUP_DIRECTORY).exists()


def test_a_dump_left_aside_by_an_interrupted_save_refuses_the_resume(tmp_path: Path) -> None:
    first = numbered(tmp_path, 200)
    folder = Path(first["run_folder"])
    shutil.copytree(folder / REPLAY_DIRECTORY, folder / REPLAY_BACKUP_DIRECTORY)

    with pytest.raises(SystemExit, match="interrupted save"):
        resumed(tmp_path, latest_checkpoint(first), 400)
