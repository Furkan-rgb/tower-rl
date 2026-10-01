"""What a run is, and what it records itself as having been configured with.

`experiment/run_identity.py` resolves the identity every artefact of a run is
bound to: the run id, the revision that produced it, and the flat snapshot of
everything the run was actually fixed with. The snapshot is checked through the
run it is taken from, because what matters is not that the function copies its
arguments but that the settings the run was built with are the ones it records.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import train
from test_train_entry_point import PROFILE, R2D2_SMOKE_REFRESH, arguments, environment, reduced_r2d2

from tower_rl.environment.episode import REWARD_SCHEMA_VERSION
from tower_rl.environment.run_actions import ACTION_SCHEMA_VERSION
from tower_rl.environment.run_environment import DecisionCadence, UpgradeAvailability
from tower_rl.environment.run_state import OBSERVATION_SCHEMA_VERSION
from tower_rl.experiment.run_identity import (
    REFERENCE_FINAL_WAVES,
    RunIdentity,
    checkpoint_identity,
    new_run_id,
    source_revision,
    tracked_params,
)
from tower_rl.learning.checkpoint import identity_hash

BACKBONE = "r2d2"


def _arm(run_dir: Path, **overrides: str | None) -> Any:
    with reduced_r2d2():
        arm, _ = train.build_arm(
            BACKBONE,
            arguments(run_dir, **overrides),
            instances=[train.ActorInstance(serial="fake-0", environment=environment())],
            device=torch.device("cpu"),
            profile_id=PROFILE,
            run_dir=run_dir / "run",
            segment=1,
            revision="test",
            started=0.0,
            tracker=train.NoExperimentTracker(),
            tags={},
        )
    return arm


def test_every_flag_reaches_the_thing_it_configures(tmp_path: Path) -> None:
    """A flag that reaches nothing is worse than no flag: it looks like a knob.

    R2D2 reads the task's discount, the seed and the collection window; every
    other loop setting is fixed (ADR 0018), so the run must record those.
    """
    arm = _arm(
        tmp_path,
        **{
            "--discount-per-game-second": "0.9",
            "--seed": "5",
            "--collection-window-episodes": "5",
        },
    )

    learner = arm.backbone.config
    assert learner.discount_per_game_second == 0.9 and learner.seed == 5
    config = arm.training.config
    assert config.collection_window_episodes == 5
    assert config.warmup_sequences == 2
    assert config.batch_size == 2  # the smoke size, `reduced_r2d2`
    assert config.exploration.epsilon_start == 0.4 and config.exploration.anneal_decisions == 0
    assert config.exploration.floors != ()
    assert config.parameter_sync_decisions == R2D2_SMOKE_REFRESH
    # And the run records what it was actually built with.
    resolved = arm.resolved
    assert resolved["discount_per_game_second"] == 0.9
    assert resolved["r2d2_discount_per_game_second"] == 0.9
    assert resolved["epsilon_anneal_decisions"] == 0
    assert resolved["parameter_sync_decisions"] == R2D2_SMOKE_REFRESH


def test_the_run_records_the_burn_in_r2d2_is_fixed_to(tmp_path: Path) -> None:
    """The record of a run says the burn-in its stored states were taken with."""
    assert _arm(tmp_path / "r2d2").resolved["burn_in"] == train.R2D2_BURN_IN


def test_a_run_id_names_its_backbone_and_collides_with_nothing() -> None:
    """Two runs started in the same second are still two runs."""
    first, second = new_run_id(BACKBONE), new_run_id(BACKBONE)

    assert first.startswith(f"{BACKBONE}-") and first != second


def test_a_checkpoint_key_is_derived_from_the_run_rather_than_assembled_by_a_script() -> None:
    """The schemas belong to the code, so the script never names them.

    `scripts/train.py` used to build the checkpoint identity out of three schema
    constants of its own, which is a compatibility decision taken at the command
    line. It passes a profile id and gets both identities back instead.
    """
    identity = RunIdentity.started_now(
        BACKBONE, profile_id="fake-profile-v1", source_revision="abc1234"
    )
    key = checkpoint_identity(identity)

    assert key.run_id == identity.run_id and key.backbone == BACKBONE
    assert key.profile_id == "fake-profile-v1" and key.source_revision == "abc1234"
    assert key.observation_schema == OBSERVATION_SCHEMA_VERSION
    assert key.action_schema == ACTION_SCHEMA_VERSION
    assert key.reward_schema == REWARD_SCHEMA_VERSION


def test_a_later_run_may_resume_an_earlier_one_s_checkpoint() -> None:
    """The run id is deliberately not part of the compatibility key.

    Resume exists precisely so a new run can continue an old one's weights; what
    may not differ is the arm, the device profile and all three schemas.
    """
    first = checkpoint_identity(
        RunIdentity.started_now(BACKBONE, profile_id="p", source_revision="abc")
    )
    second = checkpoint_identity(
        RunIdentity.started_now(BACKBONE, profile_id="p", source_revision="def")
    )
    other_arm = checkpoint_identity(
        RunIdentity.started_now("other", profile_id="p", source_revision="abc")
    )

    assert first.run_id != second.run_id
    assert first.incompatibilities(second) == ()
    assert other_arm.incompatibilities(first) != ()


def test_the_arm_is_filed_under_the_identity_it_was_built_with(tmp_path: Path) -> None:
    """What the script composes is what the checkpoints are written under."""
    arm = _arm(tmp_path)

    manifest = json.loads((arm.run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == arm.identity.run_id
    assert manifest["segments"][0]["run_id"] == arm.identity.run_id
    assert arm.identity.profile_id == PROFILE
    assert arm.identity.observation_schema == OBSERVATION_SCHEMA_VERSION


def test_every_artifact_is_bound_to_the_code_that_produced_it() -> None:
    """A revision that cannot be resolved says so rather than being omitted."""
    revision = source_revision()

    assert revision and " " not in revision


def test_the_floors_the_curve_is_read_against_travel_with_the_run() -> None:
    """A comparison opened months later must be self-contained."""
    params = tracked_params({"seed": 0})

    assert params["seed"] == 0
    assert params["reference_scripted"] == REFERENCE_FINAL_WAVES["scripted"]
    assert params["reference_source"] == REFERENCE_FINAL_WAVES["source"]


def test_a_run_1_checkpoint_refuses_to_be_resumed_under_choice_points() -> None:
    """The two cadences pose different decision problems (ADR 0009).

    A checkpoint collected at every slice is not experience a choice-point run
    can continue, and the refusal names the difference rather than leaving the
    two to be compared as if they were the same arm.
    """
    run_one = checkpoint_identity(
        RunIdentity.started_now(
            BACKBONE,
            profile_id="p",
            source_revision="abc",
            decision_cadence=DecisionCadence.EVERY_SLICE,
        )
    )
    today = checkpoint_identity(
        RunIdentity.started_now(BACKBONE, profile_id="p", source_revision="abc")
    )

    reasons = run_one.incompatibilities(today)

    assert any("decision_cadence differs" in reason for reason in reasons)
    assert run_one.incompatibilities(run_one) == ()
    assert today.decision_cadence == DecisionCadence.CHOICE_POINTS


def test_a_checkpoint_from_the_previous_observation_schema_is_refused_by_name() -> None:
    """`observation-v2` shows the policy a different world; v1 weights are not it.

    The refusal is by name rather than by tensor width on purpose: two schemas
    of the same width would still mean different things, and a shape mismatch
    surfaces as a torch error nobody can attribute.
    """
    today = checkpoint_identity(
        RunIdentity.started_now(BACKBONE, profile_id="p", source_revision="abc")
    )
    v1 = replace(today, observation_schema="observation-v1")

    reasons = today.incompatibilities(v1)

    assert any("observation_schema differs" in reason for reason in reasons)
    assert today.observation_schema == OBSERVATION_SCHEMA_VERSION == "observation-v2"


def test_the_resolved_configuration_says_which_cadence_collected_the_run(
    tmp_path: Path,
) -> None:
    """A record that cannot say which protocol produced it compares with nothing.

    Read from the environment the arm was built with, like every other cadence
    setting in the snapshot, rather than from the command line.
    """
    arm = _arm(tmp_path)

    assert arm.resolved["decision_cadence"] == "choice-points"
    assert arm.identity.decision_cadence == DecisionCadence.CHOICE_POINTS


def test_an_arm_that_could_buy_every_row_refuses_to_resume_one_that_could_not() -> None:
    """`image` and `all` are different decision problems (ADR 0011).

    Six purchasable rows and every real row are not the same legal set, so the
    weights collected under one are not experience the other can continue, and
    the baselines measured under one do not read against the other. The identity
    *hash* is deliberately unchanged by the field: a run has one availability
    from beginning to end, so the token still names the run, and every token
    already cited in a record still resolves.
    """
    image = checkpoint_identity(
        RunIdentity.started_now(BACKBONE, profile_id="p", source_revision="abc")
    )
    unlocked = checkpoint_identity(
        RunIdentity.started_now(
            BACKBONE,
            profile_id="p",
            source_revision="abc",
            upgrade_availability=UpgradeAvailability.ALL,
        )
    )

    reasons = image.incompatibilities(unlocked)

    assert any("upgrade_availability differs" in reason for reason in reasons)
    assert unlocked.incompatibilities(unlocked) == ()
    assert image.upgrade_availability == "image", "the image is what a run says nothing about"
    assert unlocked.profile_id == image.profile_id, (
        "availability is applied at the round start; the image is still profile v1"
    )
    assert identity_hash(replace(image, upgrade_availability="all")) == identity_hash(image)


def test_the_resolved_configuration_says_which_rows_the_run_could_buy(
    tmp_path: Path,
) -> None:
    """The snapshot a curve is read against has to say what was purchasable."""
    arm = _arm(tmp_path)

    assert arm.resolved["upgrade_availability"] == "image"
    assert arm.identity.upgrade_availability == UpgradeAvailability.IMAGE

    unlocked = _arm(tmp_path / "all", **{"--upgrade-availability": "all"})

    assert unlocked.resolved["upgrade_availability"] == "all"
    assert unlocked.identity.upgrade_availability == UpgradeAvailability.ALL
