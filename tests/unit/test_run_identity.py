"""What a run is, and what it records itself as having been configured with.

`experiment/run_identity.py` resolves the identity every artefact of a run is
bound to: the run id, the revision that produced it, and the flat snapshot of
everything the run was actually fixed with. The snapshot is checked through the
run it is taken from, because what matters is not that the function copies its
arguments but that the settings the run was built with are the ones it records.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import train
from test_train_entry_point import PROFILE, SMALL_NETWORK, arguments, environment

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


def _arm(run_dir: Path, **overrides: str | None) -> Any:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        arm, _ = train.build_arm(
            train.BACKBONE,
            arguments(run_dir, **overrides),
            instances=[train.ActorInstance(serial="fake-0", environment=environment())],
            device=torch.device("cpu"),
            profile_id=PROFILE,
            parent=run_dir,
            revision="test",
            started=0.0,
            tracker=train.NoExperimentTracker(),
            tags={},
        )
    return arm


def test_every_flag_reaches_the_thing_it_configures(tmp_path: Path) -> None:
    """A flag that reaches nothing is worse than no flag: it looks like a knob."""
    arm = _arm(
        tmp_path,
        **{
            "--n-step": "3",
            "--discount": "0.9",
            "--learning-rate": "0.002",
            "--target-ema-decay": "0.9",
            "--warmup-sequences": "7",
            "--epsilon-start": "0.8",
            "--epsilon-end": "0.02",
            "--epsilon-anneal-decisions": "77",
            "--collection-window-episodes": "5",
            "--gradient-steps-per-decision": "0.25",
            "--batch-size": "4",
            "--parameter-sync-episodes": "4",
        },
    )

    learner = arm.backbone.config
    assert (learner.n_step, learner.discount, learner.learning_rate) == (3, 0.9, 0.002)
    assert learner.target_ema_decay == 0.9
    config = arm.training.config
    assert config.warmup_sequences == 7
    assert (config.exploration.epsilon_start, config.exploration.epsilon_end) == (0.8, 0.02)
    assert config.exploration.anneal_decisions == 77
    assert config.exploration.option == "uniform" and config.exploration.floors == ()
    assert config.collection_window_episodes == 5
    assert (config.batch_size, config.gradient_steps_per_decision) == (4, 0.25)
    assert config.parameter_sync_episodes == 4
    # And the run records what it was actually built with.
    resolved = arm.resolved
    assert resolved["n_step"] == 3 and resolved["discount"] == 0.9
    assert resolved["epsilon_anneal_decisions"] == 77
    assert resolved["target_ema_decay"] == 0.9
    assert resolved["parameter_sync_episodes"] == 4


def test_the_burn_in_the_arm_is_built_with_is_the_one_that_fills_the_window(
    tmp_path: Path,
) -> None:
    """Burn-in only fills the history window; anything longer throws steps away."""
    stacked = _arm(tmp_path / "stacked", **{"--stacked-burn-in": "3"})

    assert stacked.training.actors[0].config.burn_in == 3
    assert stacked.resolved["burn_in"] == 3


def test_a_run_id_names_its_backbone_and_collides_with_nothing() -> None:
    """Two runs started in the same second are still two runs."""
    first, second = new_run_id(train.BACKBONE), new_run_id(train.BACKBONE)

    assert first.startswith(f"{train.BACKBONE}-") and first != second


def test_a_checkpoint_key_is_derived_from_the_run_rather_than_assembled_by_a_script() -> None:
    """The schemas belong to the code, so the script never names them.

    `scripts/train.py` used to build the checkpoint identity out of three schema
    constants of its own, which is a compatibility decision taken at the command
    line. It passes a profile id and gets both identities back instead.
    """
    identity = RunIdentity.started_now(
        train.BACKBONE, profile_id="fake-profile-v1", source_revision="abc1234"
    )
    key = checkpoint_identity(identity)

    assert key.run_id == identity.run_id and key.backbone == train.BACKBONE
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
        RunIdentity.started_now(train.BACKBONE, profile_id="p", source_revision="abc")
    )
    second = checkpoint_identity(
        RunIdentity.started_now(train.BACKBONE, profile_id="p", source_revision="def")
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

    assert arm.identity.run_id == arm.run_dir.name
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
            train.BACKBONE,
            profile_id="p",
            source_revision="abc",
            decision_cadence=DecisionCadence.EVERY_SLICE,
        )
    )
    today = checkpoint_identity(
        RunIdentity.started_now(train.BACKBONE, profile_id="p", source_revision="abc")
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
        RunIdentity.started_now(train.BACKBONE, profile_id="p", source_revision="abc")
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
        RunIdentity.started_now(train.BACKBONE, profile_id="p", source_revision="abc")
    )
    unlocked = checkpoint_identity(
        RunIdentity.started_now(
            train.BACKBONE,
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


def test_the_resolved_configuration_says_whether_exploration_was_ez_greedy(
    tmp_path: Path,
) -> None:
    """False is every run before board #83."""
    assert _arm(tmp_path / "off").resolved["ez_greedy"] is False
    arm = _arm(tmp_path / "on", **{"--ez-greedy": None})
    assert arm.backbone.config.ez_greedy is True
    assert arm.resolved["ez_greedy"] is True
