"""Atomic checkpoint writing and verified resume.

A checkpoint is written to a temporary sibling, flushed, checksummed and then
renamed, so a crash mid-write cannot destroy the previous known-good resume
point.  Resume is verified rather than assumed: the payload is checksummed on
read and the restored backbone must reproduce the saved fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from tower_rl.environment.run_environment import DecisionCadence

#: Version 2 added `tracking_run_id`, so a resumed run can carry on recording
#: into the run its parent was recorded under instead of starting a second
#: series. Everything else a resume needs was already in version 1 - the
#: optimizer moments and the target network travel inside `backbone_state`, and
#: the decision counter inside `progress` - which is why version 1 is still read
#: rather than refused: the first M2 run's checkpoints are resumable.
#: Version 3 added `environment_game_ms`: the budget is cumulative game time
#: across the fleet, so the counter a resume continues the budget from is game
#: time rather than decisions. Versions 1 and 2 are still read - their weights,
#: optimizer moments and decision counter are all still there, and every
#: evaluation path wants exactly those - but they record no game time, so a
#: resume that continues a game-time budget refuses them by name rather than
#: reading their spent budget as zero.
CHECKPOINT_FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)
#: The first format that records game time. Named rather than compared against
#: the current version, so a later format does not silently make version 3
#: files unresumable.
GAME_TIME_FORMAT_VERSION = 3


class CheckpointError(RuntimeError):
    """A checkpoint could not be written, read, or verified."""


@dataclass(frozen=True)
class CheckpointIdentity:
    """What a checkpoint is, and what it may legitimately be resumed into."""

    run_id: str
    backbone: str
    profile_id: str
    observation_schema: str
    action_schema: str
    reward_schema: str
    source_revision: str
    #: Which cadence the experience behind these weights was collected under
    #: (ADR 0009). Absent from every checkpoint written before choice points
    #: existed, and those are exactly the every-slice ones - so the default is
    #: the honest reading of a file that does not say, not a convenience.
    decision_cadence: str = DecisionCadence.EVERY_SLICE

    def incompatibilities(self, other: CheckpointIdentity) -> tuple[str, ...]:
        """Differences that make a resume unsafe. The run id may legitimately differ."""
        reasons = []
        for field_name in (
            "backbone",
            "profile_id",
            "observation_schema",
            "action_schema",
            "reward_schema",
            # A policy that learned to act at every slice did not learn the
            # problem a choice-point run poses it, so the weights are not
            # experience either run can continue or be measured against.
            "decision_cadence",
        ):
            mine, theirs = getattr(self, field_name), getattr(other, field_name)
            if mine != theirs:
                reasons.append(f"{field_name} differs: {mine!r} vs {theirs!r}")
        return tuple(reasons)


def identity_hash(identity: CheckpointIdentity) -> str:
    """A short stable token naming one checkpoint identity.

    What a measurement cites when it says which run and which schemas the
    episodes behind it came from. The run id alone would not do: two runs of the
    same code and profile are legitimately interchangeable for a resume, and a
    record that only carried a path would stop meaning anything the moment the
    file was copied.

    `decision_cadence` is deliberately not hashed. This token names a run, and a
    run collects under one cadence from beginning to end, so the field can never
    distinguish two identities that share a run id - while hashing it would
    re-key every checkpoint and record written before the field existed, and the
    tokens already cited in selection records and reports would stop resolving.
    Using a checkpoint under the wrong cadence is refused by
    `incompatibilities`, which says which field differs; that is the instrument
    for the refusal, and this is the instrument for naming the run.
    """
    payload = json.dumps(_hashed_fields(identity), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hashed_fields(identity: CheckpointIdentity) -> dict[str, Any]:
    fields = asdict(identity)
    del fields["decision_cadence"]
    return fields


@dataclass(frozen=True)
class TrainingProgress:
    """The counters a restart resumes from, and what the run last acted at.

    The counters are what a resume reads: the budget position
    (`environment_game_ms`), and the decisions, episodes and optimisation steps
    behind it. `epsilon` and `importance_beta` are not restored from here and
    must not be - epsilon is a function of `environment_decisions` and beta a
    function of the budget position, and a run derives both again, so a stored
    value would silently outrank a changed anneal horizon or a changed budget.
    They are recorded because a checkpoint should say what the run was actually
    acting and sampling at when it was written.

    `environment_decisions` stays beside the game time rather than being
    replaced by it: the replay ratio and the exploration anneal are both
    counted in decisions by design, so a resumed run needs the decision counter
    to put itself back on those schedules.
    """

    optimisation_steps: int = 0
    environment_decisions: int = 0
    #: The budget position: measured game time across every actor. Zero in a
    #: version 1 or 2 file, which was written when the budget was decisions.
    environment_game_ms: float = 0.0
    episodes: int = 0
    epsilon: float = 0.0
    importance_beta: float = 0.4


@dataclass(frozen=True)
class Checkpoint:
    """One complete, resumable training state."""

    identity: CheckpointIdentity
    progress: TrainingProgress
    backbone_state: Mapping[str, Any]
    resolved_config: Mapping[str, Any] = field(default_factory=dict)
    replay_provenance: Mapping[str, Any] = field(default_factory=dict)
    #: The tracking run this checkpoint's training was recorded under, so a
    #: resume continues that one series rather than opening a second curve
    #: beside it. None when the run was not tracked, and absent from every
    #: version 1 checkpoint.
    tracking_run_id: str | None = None
    format_version: int = CHECKPOINT_FORMAT_VERSION


def fingerprint(state: Mapping[str, Any]) -> str:
    """A stable digest of tensor contents, used to verify a resume really matched."""
    digest = hashlib.sha256()
    # Keys are not all strings: a stepped optimizer's state is keyed by integer
    # parameter index, so both the ordering and the digest go through `repr`.
    for key in sorted(state, key=repr):
        value = state[key]
        digest.update(repr(key).encode("utf-8"))
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, Mapping):
            digest.update(fingerprint(value).encode("utf-8"))
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def save(checkpoint: Checkpoint, path: Path) -> str:
    """Write atomically and return the payload checksum.

    The temporary file is a sibling so the rename stays on one filesystem, which
    is what makes it atomic. A failed write leaves the previous file untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": checkpoint.format_version,
        "identity": asdict(checkpoint.identity),
        "progress": asdict(checkpoint.progress),
        "backbone_state": checkpoint.backbone_state,
        "resolved_config": dict(checkpoint.resolved_config),
        "replay_provenance": dict(checkpoint.replay_provenance),
        "tracking_run_id": checkpoint.tracking_run_id,
        "backbone_fingerprint": fingerprint(checkpoint.backbone_state),
    }

    handle, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checksum = _file_checksum(temporary)
        temporary.with_suffix(".sha256").write_text(checksum)
        os.replace(temporary, path)
        os.replace(temporary.with_suffix(".sha256"), _checksum_path(path))
    except Exception as error:  # noqa: BLE001 - re-raised as a checkpoint failure
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(".sha256").unlink(missing_ok=True)
        raise CheckpointError(f"could not write checkpoint {path}: {error}") from error
    return checksum


def write_checkpoint(
    path: Path,
    *,
    identity: CheckpointIdentity,
    progress: TrainingProgress,
    backbone_state: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    replay_provenance: Mapping[str, Any],
    tracking_run_id: str | None = None,
) -> str:
    """Assemble one checkpoint, write it atomically, return its weight digest.

    The digest is of the weights, not of the file: it is what a measurement names
    when it says which parameters produced it, and it survives the file being
    overwritten or copied elsewhere.
    """
    save(
        Checkpoint(
            identity=identity,
            progress=progress,
            backbone_state=backbone_state,
            resolved_config=resolved_config,
            replay_provenance=replay_provenance,
            tracking_run_id=tracking_run_id,
        ),
        path,
    )
    return fingerprint(backbone_state)


def load(path: Path, *, expected: CheckpointIdentity | None = None) -> Checkpoint:
    """Read, checksum, and optionally require compatibility with a running job."""
    if not path.exists():
        raise CheckpointError(f"no checkpoint at {path}")
    recorded = _checksum_path(path)
    if recorded.exists():
        actual = _file_checksum(path)
        if actual != recorded.read_text().strip():
            raise CheckpointError(f"checkpoint {path} failed its checksum")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
        raise CheckpointError(
            f"checkpoint format {payload.get('format_version')} is not supported"
        )
    identity = CheckpointIdentity(**payload["identity"])
    if expected is not None:
        reasons = identity.incompatibilities(expected)
        if reasons:
            raise CheckpointError(f"incompatible checkpoint: {'; '.join(reasons)}")

    state = payload["backbone_state"]
    if fingerprint(state) != payload["backbone_fingerprint"]:
        raise CheckpointError("checkpoint weights do not match their recorded fingerprint")

    return Checkpoint(
        identity=identity,
        progress=TrainingProgress(**payload["progress"]),
        backbone_state=state,
        resolved_config=payload.get("resolved_config", {}),
        replay_provenance=payload.get("replay_provenance", {}),
        tracking_run_id=payload.get("tracking_run_id"),
        format_version=int(payload["format_version"]),
    )


@dataclass(frozen=True)
class ResumeState:
    """Where a resumed run carries on from, read out of its parent checkpoint.

    Read alongside `load` rather than instead of it: this is the same payload,
    narrowed to what a second segment of one run has to continue - the weights
    and optimizer moments to go on learning from, the decision counter every
    schedule and every cadence is derived from, and the tracking run its curve
    belongs on. The replay buffer is deliberately absent: it is not persisted,
    and the run re-warms it under the loaded policy.
    """

    #: The parent, as a measurement cites it: the file it was read from and the
    #: identity hash of the run that wrote it. The path alone would stop meaning
    #: anything the moment the file moved.
    parent_checkpoint: str
    #: The exploration schedule's position. Epsilon is a function of it, so it
    #: is derived again rather than restored - a stored epsilon would silently
    #: outrank a changed anneal horizon.
    decisions: int
    #: The budget position: what the parent had already spent of the run's
    #: cumulative game time. Zero in a version 1 or 2 file, which recorded
    #: none - see `format_version`.
    game_ms: float
    episodes: int
    optimisation_steps: int
    #: The format the parent was written in. Carried so the caller that knows
    #: what the resume is for can refuse a file that cannot answer it: a file
    #: below version 3 says nothing about game time, and a run budgeted in game
    #: seconds would read its whole spent budget as zero.
    format_version: int
    #: None when the parent was untracked, and for every version 1 checkpoint.
    tracking_run_id: str | None
    backbone_state: Mapping[str, Any]


def resume_state(path: Path, *, expected: CheckpointIdentity | None = None) -> ResumeState:
    """Read one checkpoint as the point a run continues from."""
    checkpoint = load(path, expected=expected)
    return ResumeState(
        parent_checkpoint=f"{path}@{identity_hash(checkpoint.identity)}",
        decisions=checkpoint.progress.environment_decisions,
        game_ms=checkpoint.progress.environment_game_ms,
        episodes=checkpoint.progress.episodes,
        optimisation_steps=checkpoint.progress.optimisation_steps,
        format_version=checkpoint.format_version,
        tracking_run_id=checkpoint.tracking_run_id,
        backbone_state=checkpoint.backbone_state,
    )


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a run manifest atomically, for the same reason checkpoints are."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception as error:  # noqa: BLE001 - re-raised as a checkpoint failure
        temporary.unlink(missing_ok=True)
        raise CheckpointError(f"could not write manifest {path}: {error}") from error


def _checksum_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
