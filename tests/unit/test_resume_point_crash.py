"""A run killed at any moment of writing its resume point leaves a pair that resumes.

Each case runs a short training session in a child process and SIGKILLs it at
one step of one of its resume points - the replay dump half written, the dump
in place but `latest.pt` not yet, the checkpoint's checksum between its
renames, the old dump half deleted or wholly deleted - and then resumes from
what is left: the `latest.pt` there and the replay it names must load together
at one decision count, the earlier save's before `latest.pt` is replaced and
the killed save's after. The resume that follows then saves again, and that
save must leave only the pair it wrote: whatever the kill left in `replay/`
is removed.

The killed save is the second of a fresh run, or the first of a segment
resumed from a folder as an older run left it: a format-4 `latest.pt` beside
the dump of its own decision count, or `M3-P015`'s single dump as `replay/`
itself. A kill is what the OOM killer does; nothing in the process gets to
clean up.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_train_entry_point import (
    RESUME_PERIOD,
    latest_checkpoint,
    numbered,
    resume_from,
    session,
)

from tower_rl.experiment.training_report import REPLAY_DIRECTORY
from tower_rl.learning.checkpoint import load, save
from tower_rl.learning.replay import (
    REPLAY_DUMP_METADATA,
    PrioritizedSequenceReplay,
    read_replay_metadata,
)

ROOT = Path(__file__).resolve().parents[2]

#: The child: a session that kills itself at one step of its `kill_save`th
#: resume point, printing each resume point's decision count as it starts one.
#: With a checkpoint given it is a second segment resumed from it, else a fresh
#: run.
CHILD = """
import os, signal, sys
from pathlib import Path

root = Path(sys.argv[1])
for path in (root / "src", root / "scripts", root / "tests", root / "tests" / "unit"):
    sys.path.insert(0, str(path))

import shutil
import train
from test_train_entry_point import RESUME_PERIOD, numbered, resume_from, session
from tower_rl.experiment import training_report
from tower_rl.learning import checkpoint, replay

kill_at, run_dir, kill_save, resume = sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
saves = 0
renamed = False


def killing(point):
    return saves == kill_save and point == kill_at


def kill(point):
    if killing(point):
        os.kill(os.getpid(), signal.SIGKILL)


write_resume_point = training_report.TrainingReport._write_resume_point


def counted(self, report, image):
    global saves
    saves += 1
    print("save", saves, report.decisions, flush=True)
    write_resume_point(self, report, image)


training_report.TrainingReport._write_resume_point = counted

write_rows = replay._write_rows
rows_written = 0


def rows(*arguments):
    global rows_written
    rows_written += 1
    write_rows(*arguments)
    if rows_written % 3 == 0:
        kill("mid-dump")


replay._write_rows = rows

write_checkpoint = training_report.write_checkpoint


def checkpoint_written(path, **given):
    if path.name == "latest.pt":
        kill("dump-in-place")
    return write_checkpoint(path, **given)


training_report.write_checkpoint = checkpoint_written

replace = os.replace


def replaced(source, target):
    global renamed
    name = Path(target).name
    if name == "latest.pt":
        kill("sidecar-names-both")
        renamed = True
    elif name == "latest.pt.sha256" and renamed:
        renamed = False
        kill("checkpoint-in-place")
    replace(source, target)


checkpoint.os.replace = replaced

rmtree = shutil.rmtree


def removed(path, *arguments, **given):
    # `shutil` is one module: the replay writer clears a stale `.partial` too.
    if Path(path).name.endswith(".partial"):
        rmtree(path, *arguments, **given)
        return
    if killing("mid-old-dump-deletion"):
        # What a kill part way through `rmtree` leaves: some of the files gone.
        for stale in sorted(Path(path).iterdir())[:3]:
            stale.unlink()
        kill("mid-old-dump-deletion")
    rmtree(path, *arguments, **given)
    kill("old-dump-deleted")


training_report.shutil.rmtree = removed

unlink = Path.unlink


def unlinked(self, *arguments, **given):
    unlink(self, *arguments, **given)
    # A dump saved as `replay/` itself is deleted file by file.
    if self.parent.name == "replay":
        kill("mid-old-dump-deletion")


Path.unlink = unlinked

if resume == "-":
    numbered(run_dir, 400)
else:
    session(
        run_dir,
        budget="400",
        settings={"--checkpoint-every-decisions": str(RESUME_PERIOD), "--resume": resume},
        resume=resume_from(run_dir, Path(resume), 400),
    )
"""


def as_format_4(latest: Path) -> None:
    """The `latest.pt` of a run before format 5: it names no replay, and has no streams.

    The dump it was written with is still `replay/d<its decisions>/`, which is
    where a resume from a format-4 checkpoint looks for one at its own count.
    """
    save(replace(load(latest), paired_replay=None, rng_state=None, format_version=4), latest)


def as_single_dump(latest: Path) -> None:
    """`M3-P015`'s layout: one dump, in dump format 1, saved as `replay/` itself."""
    folder = latest.parent.parent
    checkpoint = load(latest)
    assert checkpoint.paired_replay is not None
    replays = folder / REPLAY_DIRECTORY
    staged = shutil.move(folder / checkpoint.paired_replay, folder / "staged")
    replays.rmdir()
    Path(staged).rename(replays)
    metadata = read_replay_metadata(replays)
    del metadata["sampler_state"]
    metadata["format_version"] = 1
    (replays / REPLAY_DUMP_METADATA).write_text(json.dumps(metadata))
    as_format_4(latest)


#: What each case starts from: nothing (a fresh run, killed at its second
#: save), or the folder of a run of an older layout, killed at the first save
#: of the segment that resumes it.
FRESH, FORMAT_4, SINGLE_DUMP = "fresh", "format-4", "single-dump"

#: Each kill, and which save's pair survives it: the one before the killed
#: save ("earlier") or the killed save's own ("killed").
FRESH_KILLS = [
    ("mid-dump", "earlier"),
    ("dump-in-place", "earlier"),
    ("sidecar-names-both", "earlier"),
    ("checkpoint-in-place", "killed"),
    ("mid-old-dump-deletion", "killed"),
    ("old-dump-deleted", "killed"),
]
#: `M3-P015`'s dump is files rather than a directory, so it has no single
#: moment of "deleted"; the two older layouts are killed the other five ways.
RESUMED_KILLS = [kill for kill in FRESH_KILLS if kill[0] != "old-dump-deleted"]


def run_killed_child(
    runs: Path, kill_at: str, kill_save: int, resume: Path | None
) -> dict[int, int]:
    """Run the child until it kills itself; each resume point's decisions, by number."""
    killed = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            str(ROOT),
            kill_at,
            str(runs),
            str(kill_save),
            "-" if resume is None else str(resume),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=110,
        check=False,
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr[-2000:]
    return {
        int(number): int(decisions)
        for _, number, decisions in (
            line.split() for line in killed.stdout.splitlines() if line.startswith("save ")
        )
    }


@pytest.mark.parametrize(
    ("layout", "kill_at", "survivor"),
    [(FRESH, kill, survivor) for kill, survivor in FRESH_KILLS]
    + [(FORMAT_4, kill, survivor) for kill, survivor in RESUMED_KILLS]
    + [(SINGLE_DUMP, kill, survivor) for kill, survivor in RESUMED_KILLS],
)
def test_a_run_killed_while_writing_its_resume_point_resumes_from_a_matching_pair(
    tmp_path: Path, layout: str, kill_at: str, survivor: str
) -> None:
    runs = tmp_path / "runs"
    earlier_dump: Path | None = None
    if layout == FRESH:
        saves = run_killed_child(runs, kill_at, 2, None)
        assert set(saves) == {1, 2}
        expected = saves[1 if survivor == "earlier" else 2]
    else:
        first = numbered(runs, 200)
        latest = latest_checkpoint(first)
        earlier_dump = latest.parent.parent / str(load(latest).paired_replay)
        {FORMAT_4: as_format_4, SINGLE_DUMP: as_single_dump}[layout](latest)
        parent = load(latest).progress.environment_decisions
        saves = run_killed_child(runs, kill_at, 1, latest)
        assert set(saves) == {1}
        expected = parent if survivor == "earlier" else saves[1]

    (latest,) = runs.glob("*/checkpoints/latest.pt")
    folder = latest.parent.parent
    replays = folder / REPLAY_DIRECTORY
    assert load(latest).progress.environment_decisions == expected
    paired = load(latest).paired_replay
    if paired is None:
        # The older layout, its `latest.pt` not yet replaced: the dump it was
        # written with is still where a resume looks for it.
        assert layout != FRESH and survivor == "earlier"
        dump = replays if layout == SINGLE_DUMP else earlier_dump
    else:
        dump = folder / paired
    assert dump is not None
    assert read_replay_metadata(dump)["run"]["decisions"] == expected
    buffer = PrioritizedSequenceReplay(capacity=64)
    buffer.load_from(dump)
    assert len(buffer) > 0

    resume = resume_from(runs, latest, 800)
    assert resume.decisions == expected
    assert resume.replay_dump == dump

    # What the kill left in `replay/` beside the dump the pair names.
    left = [
        entry
        for entry in replays.iterdir()
        if (entry.is_dir() if dump == replays else entry != dump)
    ]
    if kill_at == "mid-old-dump-deletion" and layout != SINGLE_DUMP:
        # The old dump, part deleted: fewer files than the complete one.
        (partly,) = left
        assert 0 < len(list(partly.iterdir())) < len(list(dump.iterdir()))
    if kill_at in ("dump-in-place", "sidecar-names-both", "checkpoint-in-place"):
        assert left, "the kill was to leave a stray dump"

    # The resumed run saves again, and its save removes whatever was left.
    resumed = session(
        runs,
        budget="800",
        settings={"--checkpoint-every-decisions": str(RESUME_PERIOD), "--resume": str(latest)},
        resume=resume,
    )
    assert resumed["arm"]["decisions"] >= 800
    paired = load(latest).paired_replay
    assert paired is not None
    assert load(latest).progress.environment_decisions > expected
    assert list(replays.iterdir()) == [folder / paired]
