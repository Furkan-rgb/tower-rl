"""A checkpoint played as an arm, on the same terms as the non-learned floors.

No emulator and no bridge: the checkpoint is written by hand, rebuilt through
the selector the runners take, and asked for actions against the fake port. What
is under test is that the rebuilt policy is the same policy - greedy, and
choosing what the backbone that wrote it would choose - and that every record it
produces says which checkpoint it was.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import run_actors
import run_episodes
import torch
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.features import StateFeatures
from tower_rl.environment.run_environment import (
    CadenceConfig,
    DecisionCadence,
    InstrumentedRunEnvironment,
    UpgradeAvailability,
)
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.checkpoint import (
    CheckpointIdentity,
    TrainingProgress,
    identity_hash,
    write_checkpoint,
)
from tower_rl.learning.evaluator import evaluate
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.policies import CheapestFirstPolicy, checkpoint_policy
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig

PROFILE = "fake-profile-v1"

#: Narrow enough to build and load in milliseconds, and deliberately not the
#: default width: a rebuild that silently used the defaults would load the wrong
#: shape and fail here rather than in an hour of device time.
NETWORK = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

LEARNER = StackedDqnConfig(history_length=4, n_step=3, seed=7)


def identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        run_id="stacked-dqn-20260101-000000-abcdef",
        backbone="stacked-dqn",
        profile_id=PROFILE,
        observation_schema="observation-v1",
        action_schema="action-v1",
        reward_schema="reward-v1",
        source_revision="test",
    )


def resolved() -> dict[str, object]:
    """The snapshot `experiment/run_identity.resolved_config` writes."""
    return {
        "backbone": "stacked-dqn",
        "history_length": LEARNER.history_length,
        "n_step": LEARNER.n_step,
        "discount": LEARNER.discount,
        "learning_rate": LEARNER.learning_rate,
        "target_ema_decay": LEARNER.target_ema_decay,
        "network_identity_capacity": NETWORK.identity_capacity,
        "network_identity_dim": NETWORK.identity_dim,
        "network_hidden": NETWORK.hidden,
        "network_core_hidden": NETWORK.core_hidden,
    }


def trained_checkpoint(directory: Path, decisions: int = 300) -> tuple[Path, StackedDqnBackbone]:
    """One checkpoint, and the backbone whose weights are in it."""
    backbone = StackedDqnBackbone(
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

    rebuilt, restored = checkpoint_policy(path)

    assert restored == identity()
    for seed in range(8):
        state = features(seed)
        expected, _ = original.act(state, original.initial_state(), epsilon=0.0)
        actual, _ = rebuilt.act(state, rebuilt.initial_state(), epsilon=0.0)
        assert actual == expected


def test_a_rebuilt_checkpoint_acts_greedily(tmp_path: Path) -> None:
    """Greedy is the argmax of its own values, taken the same way every time."""
    path, _ = trained_checkpoint(tmp_path)
    rebuilt, _ = checkpoint_policy(path)

    for seed in range(4):
        state = features(seed)
        scalars = torch.tensor([[list(state.scalars)]], dtype=torch.float32)
        rows = torch.tensor([[list(state.rows)]], dtype=torch.float32).view(
            1, 1, NETWORK.row_count, NETWORK.row_width
        )
        mask = torch.tensor([[list(state.mask)]], dtype=torch.bool)
        with torch.no_grad():
            values, _ = rebuilt.online(scalars, rows, mask, rebuilt.initial_state())
        argmax = int(values[0, 0].argmax().item())

        chosen = {rebuilt.act(state, rebuilt.initial_state(), epsilon=0.0)[0] for _ in range(5)}
        assert chosen == {argmax}


def test_a_checkpoint_is_selected_by_path_beside_the_named_floors(tmp_path: Path) -> None:
    path, _ = trained_checkpoint(tmp_path)

    scripted, scripted_identity = run_episodes.policy_from("scripted")
    _, checkpoint_identity = run_episodes.policy_from(f"checkpoint:{path}")

    assert isinstance(scripted, CheapestFirstPolicy)
    assert scripted_identity == {"name": "scripted"}
    assert checkpoint_identity == {
        "name": "checkpoint-0000300",
        "checkpoint_path": str(path.resolve()),
        "checkpoint_identity": identity_hash(identity()),
        "run_id": identity().run_id,
    }


def test_an_arm_that_is_neither_a_name_nor_a_checkpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown policy"):
        run_episodes.policy_from("greedy")
    with pytest.raises(SystemExit, match="no checkpoint at"):
        run_episodes.policy_from(f"checkpoint:{tmp_path / 'absent.pt'}")
    # The fleet checks the same thing before it starts N emulators for it.
    with pytest.raises(SystemExit, match="unknown policy"):
        run_actors.checkpoint_arm("greedy")
    with pytest.raises(SystemExit, match="no checkpoint at"):
        run_actors.checkpoint_arm(f"checkpoint:{tmp_path / 'absent.pt'}")
    assert run_actors.checkpoint_arm("scripted") is None


def test_an_actor_record_names_the_checkpoint_that_produced_it(tmp_path: Path) -> None:
    """The arm is in the record, not in the directory the record sits in."""
    path, _ = trained_checkpoint(tmp_path)
    policy, arm = run_episodes.policy_from(f"checkpoint:{path}")

    report = evaluate(environment(), policy, episodes=2, profile_id=PROFILE)
    record = run_episodes.actor_record(
        report, arm, frame_game_ms=100.0, max_quiet_game_ms=4000,
        decision_cadence=DecisionCadence.CHOICE_POINTS,
        upgrade_availability=UpgradeAvailability.IMAGE, wall_seconds=12.0,
    )

    assert record["policy_identity"] == arm
    assert record["policy_identity"]["checkpoint_identity"] == identity_hash(identity())
    # Still the shape every other arm is written in.
    assert record["upgrade_availability"] == "image"
    assert record["valid_episodes"] == 2
    assert len(record["episodes"]) == 2


def test_the_fleet_report_carries_the_arm_its_actors_played(tmp_path: Path) -> None:
    """Resolved by the actors: the fleet process never loads the checkpoint."""
    path, _ = trained_checkpoint(tmp_path)
    _, arm = run_episodes.policy_from(f"checkpoint:{path}")
    report = evaluate(environment(), CheapestFirstPolicy(), episodes=1, profile_id=PROFILE)
    record = run_episodes.actor_record(
        report, arm, frame_game_ms=100.0, max_quiet_game_ms=4000,
        decision_cadence=DecisionCadence.CHOICE_POINTS,
        upgrade_availability=UpgradeAvailability.IMAGE, wall_seconds=9.0,
    )

    aggregated = run_actors.aggregate(
        [run_actors.ActorOutcome(index=0, serial="emulator-5556", wall_seconds=9.0, record=record)],
        wall_seconds=9.0,
    )

    assert aggregated["actors"][0]["policy_identity"] == arm
    assert aggregated["policy_identity"] == arm
