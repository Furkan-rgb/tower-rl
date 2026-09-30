"""A run killed at any moment of writing its resume point leaves a pair that resumes.

Each case runs a short training session in a child process and SIGKILLs it at
one step of its second resume point - the replay dump half written, the dump in
place but `latest.pt` not yet, the checkpoint's checksum between its renames,
the old dump half deleted - and then resumes from what is left: the
`latest.pt` there and the replay it names must load together at one decision
count, the first save's before `latest.pt` is replaced and the second's after.
A kill is what the OOM killer does; nothing in the process gets to clean up.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from test_train_entry_point import resume_from

from tower_rl.learning.checkpoint import load
from tower_rl.learning.replay import PrioritizedSequenceReplay, read_replay_metadata

ROOT = Path(__file__).resolve().parents[2]

#: The child: a session that kills itself at `KILL_AT` during its second
#: resume point, printing each resume point's decision count as it starts one.
CHILD = """
import os, signal, sys
from pathlib import Path

root = Path(sys.argv[1])
for path in (root / "src", root / "scripts", root / "tests", root / "tests" / "unit"):
    sys.path.insert(0, str(path))

import shutil
import train
from test_train_entry_point import numbered
from tower_rl.experiment import training_report
from tower_rl.learning import checkpoint, replay

kill_at, run_dir = sys.argv[2], Path(sys.argv[3])
saves = 0
renamed = False


def kill(point):
    if saves == 2 and point == kill_at:
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
    rmtree(path, *arguments, **given)
    # `shutil` is one module: the replay writer clears a stale `.partial` too.
    if not Path(path).name.endswith(".partial"):
        kill("old-dump-deleted")


training_report.shutil.rmtree = removed

numbered(run_dir, 400)
"""


@pytest.mark.parametrize(
    ("kill_at", "survivor"),
    [
        ("mid-dump", 1),
        ("dump-in-place", 1),
        ("sidecar-names-both", 1),
        ("checkpoint-in-place", 2),
        ("old-dump-deleted", 2),
    ],
)
def test_a_run_killed_while_writing_its_resume_point_resumes_from_a_matching_pair(
    tmp_path: Path, kill_at: str, survivor: int
) -> None:
    killed = subprocess.run(
        [sys.executable, "-c", CHILD, str(ROOT), kill_at, str(tmp_path / "runs")],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=110,
        check=False,
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr[-2000:]
    saves = {
        int(number): int(decisions)
        for _, number, decisions in (
            line.split() for line in killed.stdout.splitlines() if line.startswith("save ")
        )
    }
    assert set(saves) == {1, 2}

    (latest,) = (tmp_path / "runs").glob("*/checkpoints/latest.pt")
    decisions = load(latest).progress.environment_decisions
    assert decisions == saves[survivor]
    folder = latest.parent.parent
    paired = load(latest).paired_replay
    assert paired is not None
    dump = folder / paired
    assert read_replay_metadata(dump)["run"]["decisions"] == decisions
    buffer = PrioritizedSequenceReplay(capacity=64)
    buffer.load_from(dump)
    assert len(buffer) > 0

    resume = resume_from(tmp_path / "runs", latest, 800)
    assert resume.decisions == decisions
    assert resume.replay_dump == dump
