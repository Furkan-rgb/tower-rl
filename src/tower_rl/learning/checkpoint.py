"""Atomic checkpoint writing and verified resume.

A checkpoint is written to a temporary sibling, flushed, checksummed and then
renamed, so a crash mid-write cannot destroy the previous known-good resume
point - including the moment between the file's rename and its checksum
sidecar's (`save`).  Resume is verified rather than assumed: the payload is
checksummed on read and the restored backbone must reproduce the saved
fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy
import torch

from tower_rl.environment.run_environment import DecisionCadence, UpgradeAvailability

#: Version 2 added `tracking_run_id`, so a resumed run can carry on recording
#: into the run its parent was recorded under instead of starting a second
#: series. Everything else a resume needs was already in version 1 - the
#: optimizer moments and the target network travel inside `backbone_state`, and
#: the decision counter inside `progress` - which is why version 1 is still read
#: rather than refused by `load`.
#: Version 3 added `environment_game_ms`, from when the budget was game time.
#: Early stopping added three counters to `progress` within version 3: they are
#: optional fields with defaults, so a format 3 file written before them still
#: loads. Absence is meaningful rather than silent - see
#: `TrainingProgress.checkpoint_periods_closed`.
#: Version 4 is the decision-budget era (`#68`): the payload is unchanged, but
#: its selection-period counters are counted in decisions. Versions 1 to 3 still
#: load, for evaluation; `scripts/train.py` refuses to resume them, because
#: their plateau counters were counted over other periods.
#: Version 5 added `paired_replay`, the replay dump written with a `latest.pt`
#: as one resume point, and `rng_state`, the process's random streams. Both are
#: None in a version 4 file, which still resumes: with the end-of-run dump found
#: by its decision count, and with random streams starting from the seed.
#: Version 6 is the learner-thread era (ADR 0017): `progress.learner_debt_steps`
#: records the gradient steps the learner still owed, which a resume pays. A
#: file from before it was trained under the actor-thread learner, whose steps
#: were taken by whichever actor ended an episode; versions 1 to 5 still load,
#: for evaluation and selection, and `scripts/train.py` refuses to resume them,
#: because continuing would make the run a mixed one.
CHECKPOINT_FORMAT_VERSION = 6
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3, 4, 5, 6)
#: The first format a run may resume from. Named rather than compared against
#: the current version, so a later format does not make version 4 unresumable.
DECISION_BUDGET_FORMAT_VERSION = 4
#: The first format written by a run trained on the learner thread.
LEARNER_THREAD_FORMAT_VERSION = 6


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
    #: The Workshop runway profile's level the experience was collected on
    #: (ADR 0012). Absent from every checkpoint written before the profile
    #: existed, and those were all collected on the bare account - so 0 is the
    #: honest reading of a file that does not say.
    workshop_level: int = 0
    #: The digest of the upgrade setup the game actually held for the run's
    #: first episode (`environment/upgrade_setup.py`): which rows were
    #: purchasable in-run and the Workshop level each stood at. None until the
    #: run's first episode has been played, and in every checkpoint written
    #: before the setup was recorded; a None on either side is not a refusal,
    #: because it says nothing either way.
    upgrade_setup_digest: str | None = None

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
            # Nor did one that learned on another Workshop setup: the tower it
            # played is a different tower, and baseline v2 results are not
            # comparable with v1's (ADR 0012).
            "workshop_level",
        ):
            # Read as the strings they are declared as. Two of these are
            # written from `StrEnum` members, and a reason quoting
            # `<UpgradeAvailability.IMAGE: 'image'>` at an operator names the
            # type rather than the value they chose.
            mine = str(getattr(self, field_name))
            theirs = str(getattr(other, field_name))
            if mine != theirs:
                reasons.append(f"{field_name} differs: {mine!r} vs {theirs!r}")
        # Nor did one whose game held another upgrade setup than the one asked
        # for under the same names. Only when both sides say which they were.
        if (
            self.upgrade_setup_digest is not None
            and other.upgrade_setup_digest is not None
            and self.upgrade_setup_digest != other.upgrade_setup_digest
        ):
            reasons.append(
                "upgrade_setup_digest differs: "
                f"{self.upgrade_setup_digest[:12]!r} vs {other.upgrade_setup_digest[:12]!r}"
            )
        return tuple(reasons)


def identity_hash(identity: CheckpointIdentity) -> str:
    """A short stable token naming one checkpoint identity.

    What a measurement cites when it says which run and which schemas the
    episodes behind it came from. The run id alone would not do: two runs of the
    same code and profile are legitimately interchangeable for a resume, and a
    record that only carried a path would stop meaning anything the moment the
    file was copied.

    `decision_cadence`, `upgrade_availability`, `workshop_level` and
    `upgrade_setup_digest` are deliberately not hashed. This token names a
    run, and a run collects under one cadence, one availability, one Workshop
    level and one upgrade setup from beginning to end, so none of the four can
    distinguish two identities that share a run id - while hashing any of them
    would re-key every checkpoint and record written before it existed, and the
    tokens already cited in selection records and reports would stop
    resolving. The setup digest is also unknown until a run's first episode,
    so hashing it would give one run two names. Using a checkpoint under the
    wrong cadence, availability, Workshop level or setup is refused by
    `incompatibilities`, which
    says which field differs; that is the instrument for the refusal, and this
    is the instrument for naming the run.
    """
    payload = json.dumps(_hashed_fields(identity), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hashed_fields(identity: CheckpointIdentity) -> dict[str, Any]:
    fields = asdict(identity)
    del fields["decision_cadence"]
    del fields["upgrade_availability"]
    del fields["workshop_level"]
    del fields["upgrade_setup_digest"]
    return fields


@dataclass(frozen=True)
class TrainingProgress:
    """The counters a restart resumes from, and what the run last acted at.

    The counters are what a resume reads: the budget position
    (`environment_decisions`), and the game time, episodes and optimisation
    steps beside it. `epsilon` and `importance_beta` are not restored from here
    and must not be: epsilon is a function of `environment_decisions`, which a
    run derives again, so a stored value would silently outrank a changed
    anneal horizon or a changed budget; beta is replay's fixed exponent (it was
    annealed over the budget before board #85). They are recorded because a
    checkpoint should say what the run was actually acting and sampling at when
    it was written.
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
    #: Gradient steps the learner still owed on the episodes counted so far
    #: (`LearnerThread.counted_debt_steps`). Zero in a file before format 6,
    #: which has no such debt and cannot be resumed.
    learner_debt_steps: float = 0.0


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
    #: The replay dump this checkpoint was written with as one resume point,
    #: relative to its run folder (`replay/d0425078`): the buffer at this
    #: checkpoint's decision count, which a resume from it reloads. None for a
    #: checkpoint written without one - every numbered checkpoint, and every
    #: file before version 5.
    paired_replay: str | None = None
    #: The process's random streams as the checkpoint was written
    #: (`capture_rng_state`), restored by a resume so it draws on rather than
    #: starting each stream again from the seed. None before version 5.
    rng_state: Mapping[str, Any] | None = None
    format_version: int = CHECKPOINT_FORMAT_VERSION


def capture_rng_state() -> dict[str, Any]:
    """The process-wide random streams: Python's, NumPy's, torch's CPU and CUDA ones.

    The streams a component keeps for itself are not here: replay's sampler
    travels in its dump, and each acting copy's exploration stream is seeded
    again from its actor's identity when the fleet is built.
    """
    return {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Put back the streams `capture_rng_state` took.

    CUDA's only onto the same number of devices it was taken from: a resume on
    another host draws CUDA randomness from its own seed, which is no worse than
    a resume before these were saved at all.
    """
    random.setstate(state["python"])
    numpy.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda = state["torch_cuda"]
    if cuda and torch.cuda.is_available() and torch.cuda.device_count() == len(cuda):
        torch.cuda.set_rng_state_all(cuda)


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

    The file and its checksum sidecar are two renames, and a process killed
    between them would leave a checkpoint its sidecar refuses. So the sidecar
    is first replaced by one naming both the file in place and the new one,
    then the file, then the sidecar by one naming the new file alone: at every
    moment the file on disk is one its sidecar names (`load` accepts any line).
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
        "paired_replay": checkpoint.paired_replay,
        "rng_state": checkpoint.rng_state,
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
        sidecar = temporary.with_suffix(".sha256")
        if path.exists():
            _write_synced(sidecar, f"{_file_checksum(path)}\n{checksum}")
            os.replace(sidecar, _checksum_path(path))
        os.replace(temporary, path)
        _write_synced(sidecar, checksum)
        os.replace(sidecar, _checksum_path(path))
        _sync_directory(path.parent)
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
    paired_replay: str | None = None,
    rng_state: Mapping[str, Any] | None = None,
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
            paired_replay=paired_replay,
            rng_state=rng_state,
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
        # Two lines only while `save` is between its renames; see there.
        if actual not in recorded.read_text().split():
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
        paired_replay=payload.get("paired_replay"),
        rng_state=payload.get("rng_state"),
        format_version=int(payload["format_version"]),
    )


@dataclass(frozen=True)
class ResumeState:
    """Where a resumed run carries on from, read out of its parent checkpoint.

    Read alongside `load` rather than instead of it: this is the same payload,
    narrowed to what a second segment of one run has to continue - the weights
    and optimizer moments to go on learning from, the decision counter every
    schedule and every cadence is derived from, and the tracking run its curve
    belongs on. The replay buffer is not in the checkpoint: it is saved beside
    it, `paired_replay` names the dump a `latest.pt` was written with, and
    `replay_dump` is the dump the caller found matches this checkpoint.
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
    #: The format the parent was written in, so the caller can refuse a file
    #: from before the decision budget.
    format_version: int
    #: The early-stopping tracker as the parent recorded it. `periods_closed` is
    #: None for a checkpoint written before early stopping existed: the resumed
    #: run then starts the tracker fresh and says so, rather than reading
    #: absence as a run that had closed no period.
    periods_closed: int | None = None
    best_period_near_greedy_mean: float | None = None
    periods_without_improvement: int = 0
    #: The settings the parent was written with, so a resume can refuse one
    #: it could not continue under - a different discount is a different target.
    resolved_config: Mapping[str, Any] = field(default_factory=dict)
    #: The parent run's saved replay buffer, when it was saved at exactly this
    #: checkpoint's decision count; None when there is none and the run
    #: re-warms replay. Set by the caller that checked it, not read from here.
    replay_dump: Path | None = None
    #: The parent's upgrade setup digest, which the resumed run's first episode
    #: must reproduce; None for a parent written before it was recorded.
    upgrade_setup_digest: str | None = None
    #: `Checkpoint.paired_replay`: the dump the parent wrote with this file, relative
    #: to its run folder; None for a numbered checkpoint and before version 5.
    paired_replay: str | None = None
    #: `Checkpoint.rng_state`, restored once the run is built; None before version 5.
    rng_state: Mapping[str, Any] | None = None
    #: `TrainingProgress.learner_debt_steps`: what the learner still owed the parent.
    learner_debt_steps: float = 0.0


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
        format_version=checkpoint.format_version,
        periods_closed=checkpoint.progress.checkpoint_periods_closed,
        best_period_near_greedy_mean=checkpoint.progress.best_period_near_greedy_mean,
        periods_without_improvement=checkpoint.progress.periods_without_improvement or 0,
        resolved_config=checkpoint.resolved_config,
        upgrade_setup_digest=checkpoint.identity.upgrade_setup_digest,
        paired_replay=checkpoint.paired_replay,
        rng_state=checkpoint.rng_state,
        learner_debt_steps=checkpoint.progress.learner_debt_steps,
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


def _write_synced(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(directory: Path) -> None:
    """Make the renames inside `directory` durable."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
