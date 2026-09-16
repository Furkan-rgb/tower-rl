from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tower_rl.domain.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.domain.run_actions import RUN_ACTIONS
from tower_rl.learning.checkpoint import (
    Checkpoint,
    CheckpointError,
    CheckpointIdentity,
    TrainingProgress,
    fingerprint,
    load,
    save,
    write_manifest,
)
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.recurrent_q import (
    RecurrentQBackbone,
    RecurrentQConfig,
    parameters_are_equal,
)

SMALL = NetworkConfig(hidden=16, recurrent_hidden=16, identity_dim=4)


def _identity(**overrides: str) -> CheckpointIdentity:
    base = {
        "run_id": "run-1",
        "backbone": "recurrent-q",
        "profile_id": "profile-v1",
        "observation_schema": "observation-v1",
        "action_schema": "run-action-v1",
        "reward_schema": "reward-v1",
        "source_revision": "abc123",
    }
    base.update(overrides)
    return CheckpointIdentity(**base)  # type: ignore[arg-type]


def _backbone() -> RecurrentQBackbone:
    return RecurrentQBackbone(config=RecurrentQConfig(seed=0), network_config=SMALL)


def _features() -> StateFeatures:
    return StateFeatures(
        scalars=tuple([0.3] * SCALAR_COUNT),
        rows=tuple([0.2] * (ROW_COUNT * ROW_WIDTH)),
        mask=tuple(index in (0, 4, 11) for index in range(len(RUN_ACTIONS))),
    )


def test_resume_reproduces_identical_behaviour(tmp_path: Path) -> None:
    original = _backbone()
    path = tmp_path / "latest.pt"
    save(
        Checkpoint(
            identity=_identity(),
            progress=TrainingProgress(optimisation_steps=7, environment_decisions=900),
            backbone_state=original.state_dict(),
        ),
        path,
    )

    restored_backbone = _backbone()
    checkpoint = load(path)
    restored_backbone.load_state_dict(dict(checkpoint.backbone_state))

    assert checkpoint.progress.optimisation_steps == 7
    assert checkpoint.progress.environment_decisions == 900
    assert parameters_are_equal(restored_backbone.online, original.online)
    assert parameters_are_equal(restored_backbone.target, original.target)

    # The real test of a resume is that it decides the same way.
    features = _features()
    first, _ = original.act(features, original.initial_state(), epsilon=0.0)
    second, _ = restored_backbone.act(features, restored_backbone.initial_state(), epsilon=0.0)
    assert first == second


def test_a_failed_write_leaves_the_previous_checkpoint_intact(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"
    good = _backbone()
    save(Checkpoint(_identity(), TrainingProgress(optimisation_steps=1), good.state_dict()), path)
    before = path.read_bytes()

    class Unsaveable:
        def __reduce__(self) -> tuple[object, ...]:
            raise RuntimeError("cannot serialize")

    with pytest.raises(CheckpointError, match="could not write checkpoint"):
        save(Checkpoint(_identity(), TrainingProgress(), {"broken": Unsaveable()}), path)

    assert path.read_bytes() == before
    assert load(path).progress.optimisation_steps == 1
    assert not list(tmp_path.glob("*.partial")), "no temporary file may be left behind"


def test_corrupted_payloads_are_refused_not_resumed(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"
    save(Checkpoint(_identity(), TrainingProgress(), _backbone().state_dict()), path)

    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(CheckpointError, match="failed its checksum"):
        load(path)


def test_incompatible_checkpoints_cannot_be_resumed_into_a_running_job(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"
    save(Checkpoint(_identity(), TrainingProgress(), _backbone().state_dict()), path)

    with pytest.raises(CheckpointError, match="profile_id differs"):
        load(path, expected=_identity(profile_id="profile-v2"))
    with pytest.raises(CheckpointError, match="observation_schema differs"):
        load(path, expected=_identity(observation_schema="observation-v2"))
    with pytest.raises(CheckpointError, match="backbone differs"):
        load(path, expected=_identity(backbone="dreamer"))

    # A different run id is ordinary: resuming starts a new run.
    assert load(path, expected=_identity(run_id="run-2")).identity.run_id == "run-1"


def test_a_missing_checkpoint_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="no checkpoint"):
        load(tmp_path / "absent.pt")


def test_fingerprint_detects_a_single_changed_weight() -> None:
    backbone = _backbone()
    state = backbone.state_dict()
    before = fingerprint(state)

    with torch.no_grad():
        next(iter(backbone.online.parameters())).add_(1e-3)

    assert fingerprint(backbone.state_dict()) != before


def test_replay_provenance_and_config_survive_the_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"
    save(
        Checkpoint(
            identity=_identity(),
            progress=TrainingProgress(),
            backbone_state=_backbone().state_dict(),
            resolved_config={"n_step": 5, "speed": 64.0},
            replay_provenance={"sequences": 120, "restored": False},
        ),
        path,
    )

    checkpoint = load(path)

    assert checkpoint.resolved_config["n_step"] == 5
    # A checkpoint must state whether training resumed with a restored buffer.
    assert checkpoint.replay_provenance["restored"] is False


def test_manifests_are_written_atomically(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    write_manifest(path, {"run_id": "run-1", "speed": 64.0})

    assert "run-1" in path.read_text()
    assert not list(tmp_path.glob("*.partial"))
