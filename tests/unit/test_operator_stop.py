"""One SIGINT at any moment of a run ends it the way a kill bar does.

Each case runs a training session in a child process, which sends itself
SIGINT at one moment - while the replay is still warming, part way through an
episode, part way through a gradient step on the learner thread, part way
through writing a periodic resume point, or during the final evaluation - and
then carries on as the signal leaves it. The signal is sent to the process, as
`run_stage.sh` sends it to the stage's process group, from whichever thread
reached the moment: the learner's for a gradient step, the main thread's for
the final evaluation, and an actor's otherwise.

The child must exit normally, having written what a stopped run owes: the
segment's summary with every collected episode's record in it, the same
records one per line in `episodes.jsonl`, and a resume point whose
`latest.pt` and replay dump load together at the decisions the summary
reports. A second SIGINT is the operator refusing to wait: the child exits on
`KeyboardInterrupt`, and the resume point it leaves still loads as a pair.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tower_rl.learning.checkpoint import load
from tower_rl.learning.replay import PrioritizedSequenceReplay, read_replay_metadata

ROOT = Path(__file__).resolve().parents[2]

#: The child: a session that sends itself SIGINT at one moment, `times` times.
CHILD = """
import os, signal, sys, threading
from pathlib import Path

root = Path(sys.argv[1])
for path in (root / "src", root / "scripts", root / "tests", root / "tests" / "unit"):
    sys.path.insert(0, str(path))

import train
from test_train_entry_point import session
from tower_rl.environment.run_environment import InstrumentedRunEnvironment
from tower_rl.learning import training
from tower_rl.learning.actor import Actor
from tower_rl.learning.replay import ReplayImage

moment, run_dir, times = sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
calls = {}
sent = False
# Set once the main thread has handled a SIGINT: two sent back to back would
# be one pending signal, which is not what an operator pressing twice sends.
handled = threading.Event()
on_sigint = train.OperatorStop._on_sigint


def handling(self, *arguments):
    handled.set()
    return on_sigint(self, *arguments)


train.OperatorStop._on_sigint = handling


def at(name, call):
    # SIGINT the process the `call`th time `name` is reached, once.
    global sent
    calls[name] = calls.get(name, 0) + 1
    if name == moment and calls[name] == call and not sent:
        sent = True
        for _ in range(times):
            handled.clear()
            os.kill(os.getpid(), signal.SIGINT)
            if threading.current_thread() is not threading.main_thread():
                assert handled.wait(30), "the main thread never handled the SIGINT"


def wrap(owner, attribute, name, call):
    original = getattr(owner, attribute)

    def wrapped(*arguments, **given):
        at(name, call)
        return original(*arguments, **given)

    setattr(owner, attribute, wrapped)


# The third episode the fleet of two starts - so after one has ended - and
# before the buffer is warm.
wrap(Actor, "run_episode", "warm-up", 3)
# Part way through an episode, between two of its steps: the fifth step after
# the fleet has started its third episode, so one has already ended.
step = InstrumentedRunEnvironment.step


def stepped(self, *arguments, **given):
    if calls.get("warm-up", 0) >= 3:
        at("mid-episode", 5)
    return step(self, *arguments, **given)


InstrumentedRunEnvironment.step = stepped
# Inside a gradient step, on the learner thread (ADR 0017).
wrap(training.Learner, "learn", "mid-learn", 3)
# While a periodic resume point's replay is being written.
wrap(ReplayImage, "write", "mid-save", 1)
# The final evaluation, which runs on the main thread.
wrap(train, "evaluate", "final-evaluation", 1)

settings = {"--evaluate-every-episodes": "0"}
budget = "200" if moment == "final-evaluation" else "1000000"
actors = 1 if moment == "final-evaluation" else 2
if moment == "warm-up":
    settings["--warmup-sequences"] = "60"
report = session(run_dir, budget=budget, actors=actors, settings=settings)
print("finished", report["arm"]["decisions"], flush=True)
"""


def interrupted_child(runs: Path, moment: str, times: int = 1) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", CHILD, str(ROOT), moment, str(runs), str(times)],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=170,
        check=False,
    )


def assert_a_loadable_pair(runs: Path) -> int:
    """The resume point's decisions, once its `latest.pt` and dump agree on them."""
    (latest,) = runs.glob("*/checkpoints/latest.pt")
    checkpoint = load(latest)
    assert checkpoint.paired_replay is not None
    dump = latest.parent.parent / checkpoint.paired_replay
    decisions = checkpoint.progress.environment_decisions
    assert read_replay_metadata(dump)["run"]["decisions"] == decisions
    buffer = PrioritizedSequenceReplay(capacity=64)
    buffer.load_from(dump)
    return decisions


@pytest.mark.parametrize(
    "moment", ["warm-up", "mid-episode", "mid-learn", "mid-save", "final-evaluation"]
)
def test_one_sigint_at_any_moment_ends_the_run_with_its_summary_records_and_pair(
    tmp_path: Path, moment: str
) -> None:
    runs = tmp_path / "runs"
    child = interrupted_child(runs, moment)
    assert child.returncode == 0, child.stderr[-3000:]
    assert "KeyboardInterrupt" not in child.stderr

    (summary_path,) = runs.glob("*/segments/1/summary.json")
    arm: dict[str, Any] = json.loads(summary_path.read_text())["arm"]
    assert arm["interrupted"] is True
    assert arm["final_evaluation_skipped"] == "interrupted"
    assert arm["final_evaluation"] is None
    records = arm["collected_episodes"]
    assert records, "the run collected before it was stopped"
    streamed = [
        json.loads(line)
        for line in (summary_path.parent / "episodes.jsonl").read_text().splitlines()
    ]
    assert streamed == records

    assert assert_a_loadable_pair(runs) == arm["decisions"]
    log = (summary_path.parent / "train.log").read_text()
    if moment == "final-evaluation":
        assert "final evaluation abandoned: SIGINT" in log
        assert arm["decisions"] >= 200
    else:
        assert "stopped by SIGINT" in log
        assert "final evaluation skipped: stopped by SIGINT" in log
    if moment == "warm-up":
        assert arm["optimisation_steps"] == 0
    if moment == "mid-save":
        # The save in flight finished, and the run's end wrote the pair again.
        assert log.count("resume point at") >= 2


def test_a_second_sigint_exits_at_once_and_leaves_a_loadable_pair(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    child = interrupted_child(runs, "mid-learn", times=2)
    assert child.returncode != 0
    assert "KeyboardInterrupt" in child.stderr
    assert not list(runs.glob("*/segments/1/summary.json"))
    assert_a_loadable_pair(runs)
    # What had been collected is on disk even without the summary.
    (stream,) = runs.glob("*/segments/1/episodes.jsonl")
    assert stream.read_text().splitlines()
