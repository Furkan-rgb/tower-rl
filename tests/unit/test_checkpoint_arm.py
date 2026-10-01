"""A checkpoint played as an arm, on the same terms as the non-learned floors.

No emulator and no bridge: the checkpoint is written by hand, rebuilt through
the selector the runners take, and asked for actions against the fake port. What
is under test is that the rebuilt policy is the same policy - greedy, and
choosing what the backbone that wrote it would choose - and that every record it
produces says which checkpoint it was.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest
import run_actors
import run_episodes
import torch
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.features import StateFeatures
from tower_rl.environment.project_state import state_directory
from tower_rl.environment.run_environment import (
    CadenceConfig,
    DecisionCadence,
    InstrumentedRunEnvironment,
    UpgradeAvailability,
)
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.checkpoint import (
    CheckpointError,
    CheckpointIdentity,
    TrainingProgress,
    identity_hash,
    load,
    write_checkpoint,
)
from tower_rl.learning.evaluator import evaluate
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.policies import CheapestFirstPolicy, TurtlePolicy, checkpoint_policy
from tower_rl.learning.r2d2 import R2D2Backbone, R2D2Config
from tower_rl.simulation.instrumented_bridge import UpgradeSlotLabel

PROFILE = "fake-profile-v1"

#: Narrow enough to build and load in milliseconds, and deliberately not the
#: default width: a rebuild that silently used the defaults would load the wrong
#: shape and fail here rather than in an hour of device time.
NETWORK = NetworkConfig(hidden=16, identity_dim=4)

LEARNER = R2D2Config(discount_per_game_second=0.999, seed=7)

#: The protocol these fixtures' checkpoint was collected under, which is what a
#: session has to be playing for it to be playable at all. `identity()` leaves
#: both at their defaults, so an every-slice checkpoint on the image's rows.
PLAYED: dict[str, str | int] = {
    "decision_cadence": DecisionCadence.EVERY_SLICE.value,
    "upgrade_availability": UpgradeAvailability.IMAGE.value,
    "workshop_level": 0,
}


def identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        run_id="r2d2-20260101-000000-abcdef",
        backbone="r2d2",
        profile_id=PROFILE,
        observation_schema="observation-v1",
        action_schema="action-v1",
        reward_schema="reward-v1",
        source_revision="test",
    )


def resolved() -> dict[str, object]:
    """The snapshot `experiment/run_identity.resolved_config` writes."""
    return {
        "backbone": "r2d2",
        "network_identity_capacity": NETWORK.identity_capacity,
        "network_identity_dim": NETWORK.identity_dim,
        "network_hidden": NETWORK.hidden,
        **{f"r2d2_{field.name}": getattr(LEARNER, field.name) for field in fields(R2D2Config)},
    }


def trained_checkpoint(directory: Path, decisions: int = 300) -> tuple[Path, R2D2Backbone]:
    """One checkpoint, and the backbone whose weights are in it."""
    backbone = R2D2Backbone(
        config=LEARNER, network_config=NETWORK, device=torch.device("cpu")
    )
    path = directory / f"checkpoint-{decisions:07d}.pt"
    write_checkpoint(
        path,
        identity=identity(),
        progress=TrainingProgress(environment_decisions=decisions),
        backbone_state=backbone.state_dict(),
        resolved_config=resolved(),
        replay_provenance={},
    )
    return path, backbone


def features(seed: int) -> StateFeatures:
    """One arbitrary but valid state, of the shape the network was built for."""
    generator = torch.Generator().manual_seed(seed)
    scalars = torch.rand(NETWORK.scalar_count, generator=generator).tolist()
    rows = torch.rand(NETWORK.row_count * NETWORK.row_width, generator=generator).tolist()
    return StateFeatures(
        scalars=tuple(scalars),
        rows=tuple(rows),
        mask=tuple([True] * NETWORK.action_count),
    )


def environment() -> InstrumentedRunEnvironment:
    return InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=2.0),
        builder=RunStateBuilder(profile_id=PROFILE),
        cadence=CadenceConfig(frame_game_ms=100.0, max_quiet_game_ms=4000),
    )


def test_a_rebuilt_checkpoint_chooses_what_the_backbone_that_wrote_it_chooses(
    tmp_path: Path,
) -> None:
    """The weights are the arm; a rebuild that acted differently would be a new one."""
    path, original = trained_checkpoint(tmp_path)

    rebuilt, restored = checkpoint_policy(path, **PLAYED)

    assert restored == identity()
    for seed in range(8):
        state = features(seed)
        expected, _ = original.act(state, original.initial_state(), epsilon=0.0)
        actual, _ = rebuilt.act(state, rebuilt.initial_state(), epsilon=0.0)
        assert actual == expected


def test_a_rebuilt_checkpoint_acts_greedily(tmp_path: Path) -> None:
    """Greedy is the argmax of its own values, taken the same way every time."""
    path, _ = trained_checkpoint(tmp_path)
    rebuilt, _ = checkpoint_policy(path, **PLAYED)

    for seed in range(4):
        state = features(seed)
        scalars = torch.tensor([[list(state.scalars)]], dtype=torch.float32)
        rows = torch.tensor([[list(state.rows)]], dtype=torch.float32).view(
            1, 1, NETWORK.row_count, NETWORK.row_width
        )
        mask = torch.tensor([[list(state.mask)]], dtype=torch.bool)
        start = rebuilt.initial_state()
        with torch.no_grad():
            values, _ = rebuilt.online(
                scalars,
                rows,
                mask,
                torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, 1),
                (start.h, start.c),
            )
        argmax = int(values[0, 0].argmax().item())

        chosen = {rebuilt.act(state, rebuilt.initial_state(), epsilon=0.0)[0] for _ in range(5)}
        assert chosen == {argmax}


def test_a_checkpoint_is_selected_by_path_beside_the_named_floors(tmp_path: Path) -> None:
    path, _ = trained_checkpoint(tmp_path)

    scripted, scripted_identity = run_episodes.policy_from(
        "scripted",
        decision_cadence=DecisionCadence.EVERY_SLICE,
        upgrade_availability=UpgradeAvailability.IMAGE,
        workshop_level=0,
    )
    _, checkpoint_identity = run_episodes.policy_from(
        f"checkpoint:{path}",
        decision_cadence=DecisionCadence.EVERY_SLICE,
        upgrade_availability=UpgradeAvailability.IMAGE,
        workshop_level=0,
    )

    assert isinstance(scripted, CheapestFirstPolicy)
    assert scripted_identity == {"name": "scripted"}
    assert checkpoint_identity == {
        "name": "checkpoint-0000300",
        "checkpoint_path": str(path.resolve()),
        "checkpoint_identity": identity_hash(identity()),
        "run_id": identity().run_id,
        "upgrade_setup_digest": None,
    }


def test_an_arm_that_is_neither_a_name_nor_a_checkpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown policy"):
        run_episodes.policy_from(
            "greedy",
            decision_cadence=DecisionCadence.EVERY_SLICE,
            upgrade_availability=UpgradeAvailability.IMAGE,
            workshop_level=0,
        )
    with pytest.raises(SystemExit, match="no checkpoint at"):
        run_episodes.policy_from(
            f"checkpoint:{tmp_path / 'absent.pt'}",
            decision_cadence=DecisionCadence.EVERY_SLICE,
            upgrade_availability=UpgradeAvailability.IMAGE,
            workshop_level=0,
        )
    # The fleet checks the same thing before it starts N emulators for it.
    with pytest.raises(SystemExit, match="unknown policy"):
        run_actors.checkpoint_arm("greedy")
    with pytest.raises(SystemExit, match="no checkpoint at"):
        run_actors.checkpoint_arm(f"checkpoint:{tmp_path / 'absent.pt'}")
    assert run_actors.checkpoint_arm("scripted") is None


def test_an_actor_record_names_the_checkpoint_that_produced_it(tmp_path: Path) -> None:
    """The arm is in the record, not in the directory the record sits in."""
    path, _ = trained_checkpoint(tmp_path)
    policy, arm = run_episodes.policy_from(
        f"checkpoint:{path}",
        decision_cadence=DecisionCadence.EVERY_SLICE,
        upgrade_availability=UpgradeAvailability.IMAGE,
        workshop_level=0,
    )

    report = evaluate(environment(), policy, episodes=2, profile_id=PROFILE)
    record = run_episodes.actor_record(
        report, arm, frame_game_ms=100.0, max_quiet_game_ms=4000,
        decision_cadence=DecisionCadence.CHOICE_POINTS,
        upgrade_availability=UpgradeAvailability.IMAGE, workshop_level=0, wall_seconds=12.0,
    )

    assert record["policy_identity"] == arm
    assert record["policy_identity"]["checkpoint_identity"] == identity_hash(identity())
    # Still the shape every other arm is written in.
    assert record["upgrade_availability"] == "image"
    assert record["valid_episodes"] == 2
    assert len(record["episodes"]) == 2


def test_an_evaluation_file_carries_the_upgrade_setup_once() -> None:
    """At the file level; each episode below it carries only the digest."""
    played = environment()
    report = evaluate(played, CheapestFirstPolicy(), episodes=2, profile_id=PROFILE)
    setup = played.setup_reference.first
    record = run_episodes.actor_record(
        report, {"name": "scripted"}, frame_game_ms=100.0, max_quiet_game_ms=4000,
        decision_cadence=DecisionCadence.CHOICE_POINTS,
        upgrade_availability=UpgradeAvailability.IMAGE, workshop_level=0, wall_seconds=1.0,
        upgrade_setup=setup,
    )

    assert setup is not None
    assert record["upgrade_setup"] == setup.to_record()
    assert record["upgrade_setup_digest"] == setup.digest
    assert {episode["upgrade_setup_digest"] for episode in record["episodes"]} == {setup.digest}
    assert all("upgrade_setup" not in episode for episode in record["episodes"])


def test_the_fleet_report_carries_the_arm_its_actors_played(tmp_path: Path) -> None:
    """Resolved by the actors: the fleet process never loads the checkpoint."""
    path, _ = trained_checkpoint(tmp_path)
    _, arm = run_episodes.policy_from(
        f"checkpoint:{path}",
        decision_cadence=DecisionCadence.EVERY_SLICE,
        upgrade_availability=UpgradeAvailability.IMAGE,
        workshop_level=0,
    )
    report = evaluate(environment(), CheapestFirstPolicy(), episodes=1, profile_id=PROFILE)
    record = run_episodes.actor_record(
        report, arm, frame_game_ms=100.0, max_quiet_game_ms=4000,
        decision_cadence=DecisionCadence.CHOICE_POINTS,
        upgrade_availability=UpgradeAvailability.IMAGE, workshop_level=0, wall_seconds=9.0,
    )

    aggregated = run_actors.aggregate(
        [run_actors.ActorOutcome(index=0, serial="emulator-5556", wall_seconds=9.0, record=record)],
        wall_seconds=9.0,
    )

    assert aggregated["actors"][0]["policy_identity"] == arm
    assert aggregated["policy_identity"] == arm


def test_a_checkpoint_refuses_to_be_played_under_other_rows_or_another_cadence(
    tmp_path: Path,
) -> None:
    """The evaluation half of the refusal, not just the resume half.

    Neither setting is in the weights, so neither would fail to load: a
    checkpoint trained on the image's six purchasable rows would quietly play a
    fully unlocked run and report a wave nothing can be compared with. The
    refusal names both values, which is what makes it actionable rather than a
    puzzle (ADR 0009, ADR 0011).
    """
    path, _ = trained_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="upgrade_availability differs") as unlocked:
        checkpoint_policy(
            path,
            decision_cadence=DecisionCadence.EVERY_SLICE.value,
            upgrade_availability=UpgradeAvailability.ALL.value,
            workshop_level=0,
        )
    assert "'image' vs 'all'" in str(unlocked.value)

    with pytest.raises(ValueError, match="decision_cadence differs"):
        checkpoint_policy(
            path,
            decision_cadence=DecisionCadence.CHOICE_POINTS.value,
            upgrade_availability=UpgradeAvailability.IMAGE.value,
            workshop_level=0,
        )

    # And the matching protocol plays, so the refusal is a refusal and not a
    # path that never worked.
    rebuilt, restored = checkpoint_policy(path, **PLAYED)
    assert restored == identity()
    assert rebuilt.online.training is False


def test_a_checkpoint_refuses_another_workshop_level_to_play_or_resume(tmp_path: Path) -> None:
    """A baseline v1 checkpoint is not a v2 one, in either direction (ADR 0012).

    The level is not in the weights, so nothing would fail to load: the refusal
    is the only thing between a v1 checkpoint and a v2 run's numbers.
    """
    path, _ = trained_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="workshop_level differs"):
        checkpoint_policy(
            path,
            decision_cadence=DecisionCadence.EVERY_SLICE.value,
            upgrade_availability=UpgradeAvailability.IMAGE.value,
            workshop_level=5,
        )
    with pytest.raises(CheckpointError, match="workshop_level differs"):
        load(path, expected=replace(identity(), workshop_level=5))
    assert load(path, expected=identity()).identity == identity()


def test_an_arm_the_session_cannot_play_is_refused_before_the_episodes(
    tmp_path: Path,
) -> None:
    """The selector is resolved before the first episode, and refuses there.

    In an actor's own process, which is the only place a checkpoint is loaded:
    `run_actors.checkpoint_arm` deliberately never loads one, so a fleet still
    meets a mismatched arm once per actor rather than once before bring-up.
    What this pins is that the refusal reaches the arm-selection path at all,
    rather than surfacing as a played episode.
    """
    path, _ = trained_checkpoint(tmp_path)

    with pytest.raises(SystemExit, match="upgrade_availability differs"):
        run_episodes.policy_from(
            f"checkpoint:{path}",
            decision_cadence=DecisionCadence.EVERY_SLICE,
            upgrade_availability=UpgradeAvailability.ALL,
            workshop_level=0,
        )


# -- where an evaluation of a checkpoint is filed (board #90) -----------------


def fleet_arguments(**given: Any) -> argparse.Namespace:
    settings: dict[str, Any] = {
        "policy": "scripted",
        "evaluation_name": None,
        "output_directory": None,
        "output": None,
    }
    settings.update(given)
    return argparse.Namespace(**settings)


def test_an_evaluation_of_a_checkpoint_is_filed_with_its_run_by_default(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "runs" / "seed-1" / "checkpoints" / "checkpoint-d0015000.pt"
    named = fleet_arguments(policy=f"checkpoint:{checkpoint}", evaluation_name="set-a")

    run_actors.settle_outputs(named)

    evaluation = tmp_path / "runs" / "seed-1" / "evaluations" / "set-a"
    assert named.output_directory == evaluation
    assert named.output == evaluation / "fleet.json"

    unnamed = fleet_arguments(policy=f"checkpoint:{checkpoint}")
    run_actors.settle_outputs(unnamed)
    assert unnamed.output_directory.parent == tmp_path / "runs" / "seed-1" / "evaluations"
    assert unnamed.output_directory.name.startswith("checkpoint-d0015000-")

    one = argparse.Namespace(
        policy=f"checkpoint:{checkpoint}",
        evaluation_name="set-a",
        output=None,
        serial="emulator-5556",
    )
    run_episodes.settle_output(one)
    assert one.output == evaluation / "emulator-5556.json"


def test_a_floor_and_an_explicit_output_keep_the_records_directory(tmp_path: Path) -> None:
    """What `test_state_directory` checks of every other writing default."""
    floor = fleet_arguments()
    run_actors.settle_outputs(floor)
    assert floor.output_directory == state_directory() / "records" / "actors"
    assert floor.output == state_directory() / "records" / "actors.json"

    checkpoint = tmp_path / "run" / "checkpoints" / "latest.pt"
    explicit = fleet_arguments(
        policy=f"checkpoint:{checkpoint}", output_directory=tmp_path / "eval-arm"
    )
    run_actors.settle_outputs(explicit)
    assert explicit.output_directory == tmp_path / "eval-arm"

    with pytest.raises(SystemExit, match="--output-directory replaces"):
        run_actors.settle_outputs(
            fleet_arguments(
                policy=f"checkpoint:{checkpoint}",
                evaluation_name="set-a",
                output_directory=tmp_path / "eval-arm",
            )
        )
    with pytest.raises(SystemExit, match="evaluation of a checkpoint policy"):
        run_actors.settle_outputs(fleet_arguments(evaluation_name="set-a"))


def test_a_floor_played_alone_is_recorded_in_the_records_directory() -> None:
    one = argparse.Namespace(
        policy="scripted", evaluation_name=None, output=None, serial="emulator-5556"
    )
    run_episodes.settle_output(one)
    assert one.output == state_directory() / "records" / "episodes.json"


def test_turtle_is_an_arm_bound_to_the_games_row_names() -> None:
    """A selector for a policy that buys by name, and the binding both runners share."""
    policy, identity = run_episodes.policy_from(
        "turtle",
        decision_cadence=DecisionCadence.CHOICE_POINTS,
        upgrade_availability=UpgradeAvailability.ALL,
        workshop_level=5,
    )
    assert isinstance(policy, TurtlePolicy)
    assert identity == {"name": "turtle"}

    labels = [
        UpgradeSlotLabel(family, index, name, "")
        for family, index, name in (
            ("defense", 3, "Defense Absolute"),
            ("defense", 4, "Thorn Damage"),
        )
    ]
    run_episodes.bind_row_names(policy, labels)
    assert policy.rows is not None and policy.rows["Defense Absolute"] == 24

    # Loudly, on a row the game does not name.
    with pytest.raises(ValueError, match="Thorn Damage"):
        run_episodes.bind_row_names(TurtlePolicy(), labels[:-1])
    # And a policy that addresses slots by index is left as it was.
    run_episodes.bind_row_names(CheapestFirstPolicy(), labels)
