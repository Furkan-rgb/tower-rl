"""The run folder: the one directory a training run and everything about it lives in.

```text
state/runs/<run name>/
  manifest.json          the run's configuration and upgrade setup, and `segments`
  checkpoints/           latest.pt and the numbered checkpoints, every segment's
  replay/d<decisions>/   the buffer saved with latest.pt, which names it
  segments/<n>/          one per sitting: summary.json, train.log and episodes.jsonl
  logs/                  run_stage.sh stage logs, pointed here by --log-directory
  evaluations/<name>/    per-actor records of a checkpoint of this run played greedily
```

The folder is named before anything runs, so a supervisor's log can be pointed
into it at launch. A resume from the folder's own `latest.pt` continues inside
it as the next segment; anything else a resume could start from - a numbered
checkpoint, or a run written before this layout (`state/runs/session-*/<run>/`) -
starts a new folder that names that checkpoint as its parent, so no earlier
evidence is overwritten.

A segment is one sitting of `train.py`. Its run id is the one its checkpoints
and its tracked-run name carry; the folder's name is the operator's, and the
manifest's `segments` list is what ties the two together.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
CHECKPOINTS = "checkpoints"
LATEST_CHECKPOINT = "latest.pt"
SEGMENTS = "segments"
EVALUATIONS = "evaluations"
#: A segment's own files, inside `segments/<n>/`.
SEGMENT_SUMMARY = "summary.json"
SEGMENT_LOG = "train.log"
#: Each collected episode's record, one JSON line per episode, appended as the
#: episode ends: what a segment killed before it could write its summary keeps.
SEGMENT_EPISODES = "episodes.jsonl"


def utc_stamp(now: datetime | None = None) -> str:
    """A UTC time as a name part: `20260927T134501Z`."""
    return (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")


def new_run_name(backbone: str, now: datetime | None = None) -> str:
    """The default name of a new run's folder: the learner and when it started."""
    return f"{backbone}-{utc_stamp(now)}"


def run_folder_of(checkpoint: Path) -> Path:
    """The run folder a checkpoint was written into: `<run>/checkpoints/<file>`."""
    return checkpoint.resolve().parent.parent


def recorded_segments(run_folder: Path) -> list[dict[str, Any]]:
    """The segments a run folder's manifest lists; none for a folder of the old layout."""
    manifest = run_folder / MANIFEST
    if not manifest.exists():
        return []
    segments = json.loads(manifest.read_text()).get("segments", [])
    return list(segments)


def run_ids(run_folder: Path) -> list[str]:
    """The run ids a run folder's checkpoints carry: one per segment.

    A folder of the old layout (`session-*/<run id>/`) was one segment, named
    for its run id.
    """
    segments = recorded_segments(run_folder)
    if not segments:
        return [run_folder.name]
    return [str(segment["run_id"]) for segment in segments]


def continues_in_place(run_folder: Path, checkpoint: Path) -> bool:
    """Whether a resume from `checkpoint` is the next segment of `run_folder`.

    Only the folder's own `latest.pt`, and only in a folder of this layout: a
    numbered checkpoint is a branch from an earlier point, and continuing it in
    place would overwrite the numbered checkpoints written after it.
    """
    return (
        checkpoint.resolve() == run_folder.resolve() / CHECKPOINTS / LATEST_CHECKPOINT
        and bool(recorded_segments(run_folder))
    )


def segment_directory(run_folder: Path, segment: int) -> Path:
    """Where one segment's summary and log are written: `segments/<n>/`."""
    return run_folder / SEGMENTS / str(segment)


def evaluation_directory(checkpoint: Path, name: str) -> Path:
    """Where an evaluation of `checkpoint` goes by default: `<its run>/evaluations/<name>/`."""
    return run_folder_of(checkpoint) / EVALUATIONS / name
