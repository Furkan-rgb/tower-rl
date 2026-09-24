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

from tower_rl.environment.run_environment import DecisionCadence, UpgradeAvailability

#: Version 2 added `tracking_run_id`, so a resumed run can carry on recording
#: into the run its parent was recorded under instead of starting a second
#: series. Everything else a resume needs was already in version 1 - the
#: optimizer moments and the target network travel inside `backbone_state`, and
#: the decision counter inside `progress` - which is why version 1 is still read
#: rather than refused: the first M2 run's checkpoints are resumable.
#: Version 3 added `environment_game_ms`, from when the budget was game time.
#: The budget is decisions again, which every version records, so all three are
#: read and resumable; a version 1 or 2 file reports zero game time.
#: Early stopping added three counters to `progress` without a fourth version:
#: they are optional fields with defaults, so a format 3 file written before
#: them still loads and a reader that does not know them still reads everything
#: else. Absence is meaningful rather than silent - see
#: `TrainingProgress.checkpoint_periods_closed`.
CHECKPOINT_FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)


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
    #: Which upgrade rows the experience behind these weights was collected with
    #: (ADR 0011). Absent from every checkpoint written before upgrade
    #: availability existed, and those were all collected on what the profile
    #: image offers - so the default is the honest reading of a file that does
    #: not say.
    upgrade_availability: str = UpgradeAvailability.IMAGE

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
            # Nor did a policy that learned on six purchasable rows learn the
            # problem a fully unlocked run poses: the legal set is a different
            # one, so the weights and the baselines are not interchangeable
            # (ADR 0011).
            "upgrade_availability",
        ):
            # Read as the strings they are declared as. Two of these are
            # written from `StrEnum` members, and a reason quoting
            # `<UpgradeAvailability.IMAGE: 'image'>` at an operator names the
            # type rather than the value they chose.
            mine = str(getattr(self, field_name))
            theirs = str(getattr(other, field_name))
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

    `decision_cadence` and `upgrade_availability` are deliberately not hashed.
    This token names a run, and a run collects under one cadence and one
    availability from beginning to end, so neither field can distinguish two
    identities that share a run id - while hashing either would re-key every
    checkpoint and record written before it existed, and the tokens already
    cited in selection records and reports would stop resolving. Using a
    checkpoint under the wrong cadence or the wrong availability is refused by
    `incompatibilities`, which says which field differs; that is the instrument
    for the refusal, and this is the instrument for naming the run.
    """
    payload = json.dumps(_hashed_fields(identity), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hashed_fields(identity: CheckpointIdentity) -> dict[str, Any]:
    fields = asdict(identity)
    del fields["decision_cadence"]
    del fields["upgrade_availability"]
    return fields


@dataclass(frozen=True)
class TrainingProgress:
    """The counters a restart resumes from, and what the run last acted at.

    The counters are what a resume reads: the budget position
    (`environment_decisions`), and the game time, episodes and optimisation
    steps beside it. `epsilon` and `importance_beta` are not restored from here
    and must not be - both are functions of `environment_decisions`, and a run
    derives them again, so a stored value would silently outrank a changed
    anneal horizon or a changed budget. They are recorded because a checkpoint
    should say what the run was actually acting and sampling at when it was
    written.
    """

    optimisation_steps: int = 0
    #: The budget position: decisions across every actor.
    environment_decisions: int = 0
    #: Measured game time across every actor, a statistic. Zero in a version 1
    #: or 2 file, which did not record it.
    environment_game_ms: float = 0.0
    episodes: int = 0
    epsilon: float = 0.0
    importance_beta: float = 0.4
    #: The early-stopping tracker, so a run trained in two sittings is judged on
    #: one near-greedy curve rather than starting its plateau count over at
    #: every resume: the selection periods closed so far, the best period mean
    #: and how many periods in a row have failed to improve on it. Optional
    #: within format 3 - `checkpoint_periods_closed` is None in a file written
    #: before early stopping existed, which is how a resume tells a tracker it
    #: can continue from one it has to start fresh. The name is kept from when
    #: periods were cut at checkpoint crossings, because it is a key in files
    #: already written; a file from then counted game-second periods.
    checkpoint_periods_closed: int | None = None
    best_period_near_greedy_mean: float | None = None
    periods_without_improvement: int | None = None


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
    #: The budget position, and the exploration schedule's. Epsilon is a
    #: function of it, so it is derived again rather than restored - a stored
    #: epsilon would silently outrank a changed anneal horizon.
    decisions: int
    #: The game time the parent had played, a statistic. Zero in a version 1
    #: or 2 file, which recorded none.
    game_ms: float
    episodes: int
    optimisation_steps: int
    #: None when the parent was untracked, and for every version 1 checkpoint.
    tracking_run_id: str | None
    backbone_state: Mapping[str, Any]
    #: The early-stopping tracker as the parent recorded it. `periods_closed` is
    #: None for a checkpoint written before early stopping existed: the resumed
    #: run then starts the tracker fresh and says so, rather than reading
    #: absence as a run that had closed no period.
    periods_closed: int | None = None
    best_period_near_greedy_mean: float | None = None
    periods_without_improvement: int = 0


def resume_state(path: Path, *, expected: CheckpointIdentity | None = None) -> ResumeState:
    """Read one checkpoint as the point a run continues from."""
    checkpoint = load(path, expected=expected)
    return ResumeState(
        parent_checkpoint=f"{path}@{identity_hash(checkpoint.identity)}",
        decisions=checkpoint.progress.environment_decisions,
        game_ms=checkpoint.progress.environment_game_ms,
        episodes=checkpoint.progress.episodes,
        optimisation_steps=checkpoint.progress.optimisation_steps,
        tracking_run_id=checkpoint.tracking_run_id,
        backbone_state=checkpoint.backbone_state,
        periods_closed=checkpoint.progress.checkpoint_periods_closed,
        best_period_near_greedy_mean=checkpoint.progress.best_period_near_greedy_mean,
        periods_without_improvement=checkpoint.progress.periods_without_improvement or 0,
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
