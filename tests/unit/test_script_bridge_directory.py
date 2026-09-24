"""Where the device runners find the bridge they expect to be talking to.

Each of them used to read `/tmp/tower-bridge-live.latest` itself, so the rule
about where a bridge lives was written three times in three scripts and a
dangling pointer surfaced as whatever `compatibility` happened to raise on a
missing directory. It is one function in the simulation package now
(`bridge_build_directory`), and these tests hold the scripts to it.

Nothing here starts an emulator: the scripts are stopped at the first thing they
do with the directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import compare_arms
import pytest
import run_episodes
import train

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

#: The pointer file no script may read for itself any more.
POINTER = "/tmp/tower-bridge-live.latest"


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (run_episodes, ["run_episodes.py"]),
        (compare_arms, ["compare_arms.py", "--arm", "scripted", "--arm", "random"]),
        (train, ["train.py", "--budget-decisions", "1000", "--no-track"]),
    ],
    ids=["run_episodes", "compare_arms", "train"],
)
def test_the_build_directory_comes_from_the_simulation_function(
    module: Any, argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Patching the one function moves every runner's idea of where the bridge is."""
    installed = tmp_path / "installed-bridge"
    seen: list[Path] = []

    def refuse(build_dir: Path) -> None:
        # The first thing every runner does with the directory, and far enough
        # to prove where it came from: nothing has been connected to yet.
        seen.append(build_dir)
        raise SystemExit("stopped before the device")

    monkeypatch.setattr(module, "bridge_build_directory", lambda: installed)
    monkeypatch.setattr(module, "compatibility", refuse)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="stopped before the device"):
        module.main()

    assert seen == [installed]


def test_no_script_reads_the_pointer_file_for_itself() -> None:
    """One rule about where a bridge lives, in one place."""
    offenders = sorted(
        path.name
        for path in SCRIPTS.rglob("*")
        if path.is_file() and POINTER in path.read_text(errors="ignore")
    )

    assert offenders == []
