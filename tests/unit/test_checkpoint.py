from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.backbone import parameters_are_equal
from tower_rl.learning.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    Checkpoint,
    CheckpointError,
    CheckpointIdentity,
    TrainingProgress,
    fingerprint,
    identity_hash,
    load,
    resume_state,
    save,
    write_manifest,
)
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig

SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)


def _identity(**overrides: str) -> CheckpointIdentity:
    base = {
        "run_id": "run-1",
        "backbone": "stacked-dqn",
        "profile_id": "profile-v1",
        "observation_schema": "observation-v1",
        "action_schema": "run-action-v1",
        "reward_schema": "reward-v1",
        "source_revision": "abc123",
    }
    base.update(overrides)
    return CheckpointIdentity(**base)  # type: ignore[arg-type]


def _backbone() -> StackedDqnBackbone:
    return StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=2), network_config=SMALL
    )


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


def test_a_resume_state_names_its_parent_and_the_position_it_continues_from(
    tmp_path: Path,
) -> None:
    """What a second segment of one run reads out of the checkpoint it is given."""
    path = tmp_path / "latest.pt"
    backbone = _backbone()
    save(
        Checkpoint(
            identity=_identity(),
            progress=TrainingProgress(
                optimisation_steps=31,
                environment_decisions=900,
                environment_game_ms=1_800_000.0,
                episodes=12,
            ),
            backbone_state=backbone.state_dict(),
            tracking_run_id="mlflow-run-1",
        ),
        path,
    )

    state = resume_state(path)

    # The budget position, which is what the resumed run's budget check reads.
    assert state.decisions == 900 and state.episodes == 12
    # Game time travels beside it as a statistic.
    assert state.game_ms == 1_800_000.0
    assert load(path).format_version == state.format_version == CHECKPOINT_FORMAT_VERSION == 4
    assert state.optimisation_steps == 31
    assert state.tracking_run_id == "mlflow-run-1"
    assert "optimizer" in state.backbone_state, "the moments travel with the weights"
    # Identified by what it is as well as by where it is: a path alone stops
    # meaning anything the moment the file is copied.
    assert state.parent_checkpoint == f"{path}@{identity_hash(_identity())}"
    # Epsilon and beta are deliberately not among them: both are functions of
    # the decision counter, and a run derives them from it again.
    assert not hasattr(state, "epsilon")


def test_the_earlier_checkpoint_format_is_still_read(tmp_path: Path) -> None:
    """An old file still loads, for evaluation.

    Version 1 carried no tracking run id and neither it nor version 2 records
    game time, which reads back as zero. Reading is `load`'s; whether a run may
    continue from it is `scripts/train.py`'s, which refuses anything before
    the decision budget (format 4).
    """
    path = tmp_path / "legacy.pt"
    save(
        Checkpoint(
            identity=_identity(),
            progress=TrainingProgress(environment_decisions=200_174),
            backbone_state=_backbone().state_dict(),
            format_version=1,
        ),
        path,
    )

    state = resume_state(path)

    assert state.decisions == 200_174
    assert state.tracking_run_id is None
    # It says nothing about game time, and says so as zero rather than as a
    # guess.
    assert state.game_ms == 0.0
    assert load(path).format_version == state.format_version == 1

    # A version this code does not know is still refused rather than guessed at.
    future = tmp_path / "future.pt"
    torch.save({"format_version": 5, "identity": {}}, future)
    with pytest.raises(CheckpointError, match="format 5 is not supported"):
        load(future)


def test_a_checkpoint_that_names_no_cadence_is_read_as_run_1_s(tmp_path: Path) -> None:
    """Every file written before choice points existed is every-slice experience.

    Its identity dict has no cadence key at all, so the default is the honest
    reading of what it holds rather than a convenience - and a choice-point run
    is refused it by name (ADR 0009).
    """
    backbone = _backbone().state_dict()
    identity = asdict(_identity())
    del identity["decision_cadence"]
    path = tmp_path / "run-1.pt"
    torch.save(
        {
            "format_version": 1,
            "identity": identity,
            "progress": asdict(TrainingProgress(environment_decisions=200_174)),
            "backbone_state": backbone,
            "backbone_fingerprint": fingerprint(backbone),
        },
        path,
    )

    loaded = load(path)

    assert loaded.identity.decision_cadence == "every-slice"
    with pytest.raises(CheckpointError, match="decision_cadence differs"):
        load(path, expected=_identity(decision_cadence="choice-points"))


def test_manifests_are_written_atomically(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    write_manifest(path, {"run_id": "run-1", "speed": 64.0})

    assert "run-1" in path.read_text()
    assert not list(tmp_path.glob("*.partial"))


def test_fingerprint_handles_the_integer_keys_a_stepped_optimizer_uses() -> None:
    """A real optimizer state is keyed by parameter index, not by name."""
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.sum().backward()
    optimizer.step()
    state = optimizer.state_dict()

    assert any(isinstance(key, int) for key in state["state"])
    digest = fingerprint(state)

    assert digest == fingerprint(state), "the digest must be stable"
    assert len(digest) == 64


def test_run_1_still_hashes_to_the_token_its_selection_record_cites() -> None:
    """The identity token names a run, so adding a field must not re-key it.

    `8344a482eede` is what run 1's own `selection.json` cites for
    `stacked-dqn-20260918-215839-e3c6ba`, written before the decision cadence
    was part of an identity. A token that changed under it would strand every
    record and report that already quotes one, and it could not tell two
    identities apart anyway: a run collects under one cadence throughout. The
    cross-cadence refusal is `incompatibilities`, not this.
    """
    run_one = CheckpointIdentity(
        run_id="stacked-dqn-20260918-215839-e3c6ba",
        backbone="stacked-dqn",
        profile_id="tower-play-29.0.3-rooted-readonly-v1",
        observation_schema="observation-v1",
        action_schema="run-action-v1",
        reward_schema="reward-v1",
        source_revision="3c494b7",
    )

    assert identity_hash(run_one) == "8344a482eede"
    assert identity_hash(replace(run_one, decision_cadence="choice-points")) == "8344a482eede"
    assert identity_hash(replace(run_one, profile_id="other")) != "8344a482eede"
