"""An episode's steps as numpy arrays, and how a replay dump streams them.

Shared by the two replays that store steps once, flat, per episode:
DreamerV3's (`dreamer_replay.py`), which adds its latents to each step, and
R2D2's (`r2d2_replay.py`), which reads them as they are. A step holds its
observation, the action taken at it, and the reward, termination and game time
of the transition *into* it. An episode's last step is its final observation -
the environment's terminal one when the run died - with no action taken.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy

from tower_rl.learning.replay import ReplayDumpError, ReplayRejected


@dataclass(frozen=True)
class StepArrays:
    """One episode's steps, one row per step."""

    scalars: numpy.ndarray  # float32 [n, SCALAR_COUNT]
    rows: numpy.ndarray  # float32 [n, ROW_COUNT * ROW_WIDTH]
    mask: numpy.ndarray  # bool [n, actions]
    #: The action taken at the step; 0 at the last step, where none is.
    action: numpy.ndarray  # int64 [n]
    #: The transition into the step: 0, False and 0 at the first.
    reward: numpy.ndarray  # float32 [n]
    terminal: numpy.ndarray  # bool [n]
    game_ms: numpy.ndarray  # float32 [n]

    def __post_init__(self) -> None:
        count = len(self.action)
        if count < 1:
            raise ReplayRejected("an episode needs at least one step")
        if any(len(getattr(self, name.name)) != count for name in fields(self)):
            raise ReplayRejected("every field of an episode needs one row per step")

    def __len__(self) -> int:
        return len(self.action)


def write_rows(path: Path, total: int, parts: Sequence[numpy.ndarray]) -> None:
    """Arrays of one dtype and row shape as one `.npy` of `total` rows, written a part at a time.

    No concatenated copy is built in memory; the file is synced and dropped
    from the page cache, so a dump of gigabytes does not sit in memory as dirty
    pages beside the buffer it was written from.
    """
    if not parts:
        numpy.save(path, numpy.zeros((0,), numpy.float32))
        return
    reference = parts[0]
    with path.open("wb") as stream:
        numpy.lib.format.write_array_header_1_0(
            stream,
            {
                "descr": reference.dtype.str,
                "fortran_order": False,
                "shape": (total, *reference.shape[1:]),
            },
        )
        for part in parts:
            stream.write(numpy.ascontiguousarray(part).tobytes())
        stream.flush()
        os.fsync(stream.fileno())
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def read_rows(directory: Path, name: str) -> Any:
    """One array of a dump, memory-mapped."""
    try:
        return numpy.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ReplayDumpError(f"replay dump array {name} is unreadable: {error}") from error
