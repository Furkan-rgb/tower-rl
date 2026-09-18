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

CHECKPOINT_FORMAT_VERSION = 1


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

    def incompatibilities(self, other: CheckpointIdentity) -> tuple[str, ...]:
        """Differences that make a resume unsafe. The run id may legitimately differ."""
        reasons = []
        for field_name in (
            "backbone",
            "profile_id",
            "observation_schema",
            "action_schema",
            "reward_schema",
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
    """
    payload = json.dumps(asdict(identity), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class TrainingProgress:
    """Counters and schedules that must survive a restart to resume honestly."""

    optimisation_steps: int = 0
    environment_decisions: int = 0
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
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
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
