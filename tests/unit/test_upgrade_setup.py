"""The upgrade setup a run was played on, recorded from what the game read back.

Every episode records which in-run rows were purchasable at its first
observation and the Workshop level each stood at after the round began, with a
digest of that record. The run's first setup goes into its manifest and its
checkpoints; a later episode on another setup is invalid by name, and a run
that continues or plays a checkpoint on another setup is refused.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fakes.fake_run_port import DEVICE_REAL_ROWS, FakeRunPort
from test_train_entry_point import latest_checkpoint, numbered, resume_from, session

from tower_rl.environment.episode import TerminationOutcome
from tower_rl.environment.run_actions import WAIT
from tower_rl.environment.run_environment import (
    CadenceConfig,
    DecisionCadence,
    InstrumentedRunEnvironment,
    UpgradeAvailability,
)
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.environment.upgrade_setup import (
    UPGRADE_SETUP_DRIFT,
    UpgradeSetup,
    UpgradeSetupReference,
    UpgradeSetupRefused,
    UpgradeSetupRow,
    setup_digest,
)
from tower_rl.environment.workshop import WORKSHOP_OFF
from tower_rl.learning.checkpoint import CheckpointIdentity, identity_hash, load
from tower_rl.learning.evaluator import episode_record, evaluate
from tower_rl.learning.policies import Policy, WaitOnlyPolicy


def _environment(
    availability: UpgradeAvailability = UpgradeAvailability.IMAGE,
    level: int = WORKSHOP_OFF,
    reference: UpgradeSetupReference | None = None,
    **port: Any,
) -> tuple[InstrumentedRunEnvironment, FakeRunPort]:
    fake = FakeRunPort(**{"real_rows": dict(DEVICE_REAL_ROWS), **port})
    environment = InstrumentedRunEnvironment(
        port=fake,
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
        decision_cadence=DecisionCadence.CHOICE_POINTS,
        upgrade_availability=availability,
        workshop_level=level,
    )
    if reference is not None:
        environment.setup_reference = reference
    return environment, fake


def _setup(environment: InstrumentedRunEnvironment) -> UpgradeSetup:
    environment.reset()
    setup = environment.summarize(TerminationOutcome.OPERATOR_STOP).upgrade_setup
    assert setup is not None
    return setup


def test_the_setup_is_every_named_row_read_off_the_first_observation() -> None:
    environment, port = _environment()
    first = environment.reset()
    setup = environment.summarize(TerminationOutcome.OPERATOR_STOP).upgrade_setup

    assert setup is not None
    named = [label for label in port.slot_labels() if label.name]
    assert len(setup.rows) == len(named) == sum(DEVICE_REAL_ROWS.values())
    unlocked = {str(row.action): row.unlocked for row in first.rows}
    for row in setup.rows:
        assert row.name == f"{row.family} {row.index}"
        assert row.available_in_run == unlocked[f"{row.family}:{row.index}"]
    # The fake's image offers four attack rows and two defense rows.
    assert sum(row.available_in_run for row in setup.rows) == 6
    # No Workshop profile: nothing was read, and every row records 0.
    assert setup.workshop_read_back is False
    assert {row.workshop_level for row in setup.rows} == {0}


def test_the_workshop_levels_are_the_ones_read_back_not_the_ones_asked_for() -> None:
    held = _setup(_environment(level=5)[0])
    reverted = _setup(_environment(level=5, workshop_reverts_at_round_start=True)[0])

    assert held.workshop_read_back and reverted.workshop_read_back
    damage = next(row for row in held.rows if (row.family, row.index) == ("attack", 0))
    assert damage.workshop_level == 5
    # The game put the levels back at the round start; the record says so.
    assert {row.workshop_level for row in reverted.rows} <= {0, None}
    assert held.digest != reverted.digest


def test_image_and_all_are_different_setups() -> None:
    image = _setup(_environment(UpgradeAvailability.IMAGE)[0])
    every = _setup(_environment(UpgradeAvailability.ALL)[0])

    assert all(row.available_in_run for row in every.rows)
    assert not all(row.available_in_run for row in image.rows)
    assert image.digest != every.digest


def test_the_digest_is_stable_and_canonical() -> None:
    one = _setup(_environment()[0])
    two = _setup(_environment()[0])
    canonical = json.dumps(one.to_record(), sort_keys=True, separators=(",", ":"))

    assert one == two and one.digest == two.digest
    assert one.digest == hashlib.sha256(canonical.encode()).hexdigest()
    assert setup_digest(json.loads(json.dumps(one.to_record(), indent=4))) == one.digest
    # The rows are held in (family, index) order whatever order the game named them.
    labels = list(reversed(_environment()[1].slot_labels()))
    environment, port = _environment()
    port.slot_labels = lambda: tuple(labels)  # type: ignore[method-assign]
    assert _setup(environment).digest == one.digest


def test_an_episode_record_carries_only_the_digest() -> None:
    environment, _ = _environment()
    environment.reset()
    summary = environment.summarize(TerminationOutcome.OPERATOR_STOP)
    record = episode_record(0, summary)

    assert summary.upgrade_setup is not None
    assert record["upgrade_setup_digest"] == summary.upgrade_setup.digest
    assert "upgrade_setup" not in record
    json.dumps(record)


def test_an_episode_on_another_setup_than_the_runs_first_is_invalid_by_name() -> None:
    """Two instances of one run, one of them offering every row: its episodes drift."""
    reference = UpgradeSetupReference()
    first, _ = _environment(UpgradeAvailability.IMAGE, reference=reference)
    other, _ = _environment(UpgradeAvailability.ALL, reference=reference)

    first.reset()
    state = other.reset()
    transition = other.step(WAIT)
    summary = other.summarize(transition.termination or TerminationOutcome.OPERATOR_STOP)

    assert not state.valid
    assert any(reason.startswith(UPGRADE_SETUP_DRIFT) for reason in state.invalid_reasons)
    assert not transition.admissible
    assert transition.termination is TerminationOutcome.OBSERVATION_INVALID
    assert not summary.valid
    assert any(UPGRADE_SETUP_DRIFT in text for text in summary.termination_detail)
    # The drifted episode's record carries the setup it was played on.
    assert summary.upgrade_setup_drifted and summary.upgrade_setup is not None
    record = episode_record(0, summary)
    assert record["upgrade_setup"] == summary.upgrade_setup.to_record()
    assert setup_digest(record["upgrade_setup"]) == record["upgrade_setup_digest"]
    # The run's own setup is still the first episode's, and it stays valid.
    assert reference.first is not None
    assert first.reset().valid


def test_evaluating_under_another_setup_than_the_checkpoints_is_refused() -> None:
    expected = _setup(_environment(UpgradeAvailability.ALL)[0]).digest
    environment, _ = _environment(
        UpgradeAvailability.IMAGE,
        reference=UpgradeSetupReference(expected=expected, expected_from="run-a/latest.pt"),
    )
    policy: Policy = WaitOnlyPolicy()

    with pytest.raises(UpgradeSetupRefused, match="run-a/latest.pt"):
        evaluate(environment, policy, episodes=1, profile_id="fake-profile-v1")


def test_evaluating_under_the_checkpoints_setup_proceeds() -> None:
    expected = _setup(_environment()[0]).digest
    environment, _ = _environment(reference=UpgradeSetupReference(expected=expected))

    assert environment.reset().valid


def test_the_identity_checks_the_digest_only_when_both_sides_have_one() -> None:
    base = CheckpointIdentity(
        run_id="a", backbone="r2d2", profile_id="p", observation_schema="o",
        action_schema="a", reward_schema="r", source_revision="s",
    )
    one = replace(base, upgrade_setup_digest="1" * 64)
    two = replace(base, upgrade_setup_digest="2" * 64)

    assert one.incompatibilities(one) == ()
    assert base.incompatibilities(one) == () and one.incompatibilities(base) == ()
    (reason,) = one.incompatibilities(two)
    assert reason.startswith("upgrade_setup_digest differs")
    # Not hashed: a run's name does not change once its first episode is played.
    assert identity_hash(base) == identity_hash(one)


def test_a_run_records_its_first_setup_in_the_manifest_and_checkpoints(
    tmp_path: Path,
) -> None:
    report = numbered(tmp_path, 200)
    run_dir = Path(report["run_folder"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    digests = {episode["upgrade_setup_digest"] for episode in report["arm"]["collected_episodes"]}

    assert digests == {manifest["upgrade_setup_digest"]}
    assert setup_digest(manifest["upgrade_setup"]) == manifest["upgrade_setup_digest"]
    assert load(latest_checkpoint(report)).identity.upgrade_setup_digest == (
        manifest["upgrade_setup_digest"]
    )


def test_resuming_under_another_setup_is_refused(tmp_path: Path) -> None:
    first = numbered(tmp_path / "first", 200)
    resume = resume_from(tmp_path / "second", latest_checkpoint(first), 400)
    assert resume.upgrade_setup_digest is not None

    with pytest.raises(UpgradeSetupRefused, match="latest.pt"):
        # The same flags, but a game that names one row fewer.
        session(
            tmp_path / "second",
            budget="400",
            resume=resume,
            real_rows={"attack": 19, "defense": 20, "utility": 20},
        )


def test_the_setup_row_is_what_the_record_names() -> None:
    row = UpgradeSetupRow("attack", 0, "Damage", True, 5)
    setup = UpgradeSetup(rows=(row,), workshop_read_back=True)

    assert setup.to_record() == {
        "workshop_read_back": True,
        "rows": [
            {
                "family": "attack", "index": 0, "name": "Damage",
                "available_in_run": True, "workshop_level": 5,
            }
        ],
    }


def test_exactly_one_concurrent_admit_is_told_it_pinned() -> None:
    """A fleet's actors reach their first round start together; one records it."""
    setup = _setup(_environment()[0])
    pinned_with: list[UpgradeSetup] = []
    reference = UpgradeSetupReference(on_pinned=pinned_with.append)
    threads = 16
    start = threading.Barrier(threads)
    results: list[tuple[bool, str | None]] = []

    def admit() -> None:
        start.wait()
        results.append(reference.admit(setup))

    workers = [threading.Thread(target=admit) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert len(results) == threads
    assert sum(pinned for pinned, _ in results) == 1
    assert all(drift is None for _, drift in results)
    assert pinned_with == [setup]
