"""The training entry point, end to end against the fake port.

No emulator, no adb, no bridge: `train_session` takes the environment it trains
against, so the double never reaches a path a device run can take. What is under
test is the thing the developer actually starts - argument parsing, arm
construction, the interleaved block schedule, periodic evaluation and
checkpointing, and the learning curve the run is read from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import train  # noqa: E402
from fakes.fake_run_port import FakeRunPort  # noqa: E402

from tower_rl.application.actor import ActorConfig  # noqa: E402
from tower_rl.application.evaluator import evaluate  # noqa: E402
from tower_rl.application.run_environment import (  # noqa: E402
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.domain.episode import TerminationOutcome  # noqa: E402
from tower_rl.domain.features import StateFeatures  # noqa: E402
from tower_rl.domain.run_state import RunStateBuilder  # noqa: E402
from tower_rl.learning.checkpoint import fingerprint, load  # noqa: E402
from tower_rl.learning.network import NetworkConfig  # noqa: E402

#: Tensors this small spend their time handing work between threads rather than
#: computing: one thread runs the whole file about fifteen times faster.
torch.set_num_threads(1)

PROFILE = "fake-profile-v1"

#: A network narrow enough that the entry point can be exercised in seconds. At
#: production width every decision is a CPU forward pass and dominates the run;
#: what is under test here is the plumbing around the learner, not its capacity,
#: which the backbone contract suite covers.
SMALL_NETWORK = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

#: A whole learning curve point, as the report must carry it.
POINT_KEYS = {
    "decisions",
    "episodes",
    "wall_seconds",
    "model_version",
    "mean_final_wave",
    "stdev_final_wave",
    "final_waves",
    "valid_episodes",
    "invalid_episodes",
    "invalid_by_reason",
    "versus_scripted_reference",
    "checkpoint_fingerprint",
    "checkpoint_path",
}


def arguments(run_dir: Path, *backbones: str, **overrides: str) -> argparse.Namespace:
    """The real parser, so the entry point's own defaults and checks are used."""
    argv = []
    for name in backbones:
        argv += ["--backbone", name]
    settings = {
        "--budget-decisions": "150",
        "--block-decisions": "50",
        "--batch-size": "2",
        "--gradient-steps-per-decision": "0.2",
        "--sequence-length": "6",
        "--burn-in": "3",
        "--history-length": "4",
        "--replay-capacity": "64",
        "--evaluate-every-episodes": "1",
        "--evaluation-episodes": "2",
        "--checkpoint-every-episodes": "2",
        "--serial": "fake-0",
        "--max-quiet-game-ms": "4000",
        "--run-dir": str(run_dir),
    }
    settings.update(overrides)
    for flag, value in settings.items():
        argv += [flag, value]
    return train.parse_arguments(argv)


def environment(**overrides: Any) -> InstrumentedRunEnvironment:
    settings: dict[str, Any] = {"damage_per_second": 2.0}
    settings.update(overrides)
    return InstrumentedRunEnvironment(
        port=FakeRunPort(**settings),
        builder=RunStateBuilder(profile_id=PROFILE),
        # `frame_game_ms` is the standing 100 ms of M1B-E018.
        cadence=CadenceConfig(frame_game_ms=100.0, max_quiet_game_ms=4000),
    )


def session(
    run_dir: Path, *backbones: str, budget: str = "120", **fake: Any
) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        return train.train_session(
            arguments(run_dir, *backbones, **{"--budget-decisions": budget}),
            environment(**fake),
            profile_id=PROFILE,
            revision="test",
            device=torch.device("cpu"),
        )


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One interleaved session over both backbones, reused by several checks."""
    run_dir = tmp_path_factory.mktemp("runs")
    return session(run_dir, "recurrent-q", "stacked-dqn")


def test_both_backbones_train_under_one_interleaved_budget(trained: dict[str, Any]) -> None:
    arms = trained["arms"]

    assert [arm["backbone"] for arm in arms] == ["recurrent-q", "stacked-dqn"]
    for arm in arms:
        assert arm["decisions"] >= 120, "every arm spends the same budget"
        assert arm["episodes"] > 0
        assert arm["optimisation_steps"] > 0
        assert arm["sequences_accepted"] > 0
        assert arm["failed_episodes"] == 0


def test_the_report_carries_a_well_formed_learning_curve(trained: dict[str, Any]) -> None:
    for arm in trained["arms"]:
        curve = arm["learning_curve"]

        assert curve, "a run with evaluations must produce curve points"
        assert all(set(point) == POINT_KEYS for point in curve)
        decisions = [point["decisions"] for point in curve]
        assert decisions == sorted(decisions), "a curve is placed on the budget in order"
        for point in curve:
            assert point["wall_seconds"] >= 0.0
            assert point["valid_episodes"] + point["invalid_episodes"] == 2
            assert len(point["final_waves"]) == point["valid_episodes"]
            assert point["mean_final_wave"] > 0
            # Two evaluation episodes have a spread; it is reported beside the mean.
            assert point["stdev_final_wave"] is not None
            assert point["versus_scripted_reference"] == pytest.approx(
                point["mean_final_wave"] - train.SCRIPTED_REFERENCE, abs=1e-3
            )


def test_the_curve_is_readable_against_the_measured_baselines(trained: dict[str, Any]) -> None:
    """The comparison has to be in the artefact, not in another document."""
    for reference in [trained["reference_final_waves"]] + [
        arm["reference_final_waves"] for arm in trained["arms"]
    ]:
        assert reference["scripted"] == 5.57
        assert reference["random"] == 5.35
        assert reference["wait"] == 1.87
        assert reference["source"]


def test_each_point_names_a_checkpoint_that_holds_the_weights_it_scored(
    trained: dict[str, Any],
) -> None:
    for arm in trained["arms"]:
        curve = arm["learning_curve"]
        for point in curve:
            # The file still holds the weights the point scored: a point writes
            # its own checkpoint rather than sharing the overwritten resume one.
            stored = load(Path(point["checkpoint_path"]))
            assert fingerprint(stored.backbone_state) == point["checkpoint_fingerprint"]
            assert stored.progress.environment_decisions == point["decisions"]
        # The fingerprint moves as the model learns, or the curve could not be
        # attributed to anything.
        versions = {point["model_version"] for point in curve}
        digests = {point["checkpoint_fingerprint"] for point in curve}
        assert len(digests) == len(versions)


def test_everything_the_run_writes_lands_outside_the_repository(
    trained: dict[str, Any],
) -> None:
    repository = Path(train.__file__).resolve().parents[1]
    session_dir = Path(trained["session"])
    summary = json.loads((session_dir / "summary.json").read_text())

    assert repository not in session_dir.parents
    assert summary["arms"][0]["learning_curve"] == trained["arms"][0]["learning_curve"]
    for arm in trained["arms"]:
        run_dir = Path(arm["checkpoint_path"]).parents[1]
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "manifest.json").exists()
        # The checkpoint round-trips: written atomically, checksummed on read.
        checkpoint = load(Path(arm["checkpoint_path"]))
        assert checkpoint.identity.backbone == arm["backbone"]
        assert checkpoint.progress.environment_decisions == arm["decisions"]


def test_evaluation_runs_without_exploration(tmp_path: Path) -> None:
    """The entry point's evaluation goes through `evaluate`, which forces zero."""
    seen: list[float] = []

    class RecordingPolicy:
        def initial_state(self) -> None:
            return None

        def stored_recurrent_state(self, state: None) -> None:
            return None

        def act(
            self, features: StateFeatures, state: None, *, epsilon: float
        ) -> tuple[int, None]:
            seen.append(epsilon)
            return next(index for index, allowed in enumerate(features.mask) if allowed), None

    report = evaluate(
        environment(),
        RecordingPolicy(),
        episodes=2,
        profile_id=PROFILE,
        # Exploration asked for and refused: evaluation is measurement.
        actor_config=ActorConfig(epsilon=1.0),
    )

    assert seen and set(seen) == {0.0}
    assert report.valid_episodes == 2


def test_an_episode_the_port_refuses_does_not_abort_the_session(tmp_path: Path) -> None:
    report = session(tmp_path, "recurrent-q", refuse_episodes=frozenset({2, 3}))

    # Episode ordinals are consumed by evaluation episodes too, so one refusal
    # lands on collection and one on an evaluation. Neither may end the session.
    arm = report["arms"][0]
    assert arm["failed_episodes"] >= 1
    assert len(arm["evaluation_failures"]) >= 1
    assert arm["failed_episodes"] + len(arm["evaluation_failures"]) == 2
    # The budget is still spent and the curve still produced.
    assert arm["decisions"] >= 120
    assert arm["learning_curve"]


def test_an_ambiguous_advance_is_classified_and_the_session_continues(
    tmp_path: Path,
) -> None:
    # Ordinal 1 is the first collected episode; evaluation episodes take the
    # ordinals after it.
    report = session(tmp_path, "recurrent-q", ambiguous_advance_episodes=frozenset({1}))

    arm = report["arms"][0]
    assert arm["failed_episodes"] == 0, "the port answered; the episode did not"
    assert arm["episodes"] > 1 and arm["decisions"] >= 120
    # The pipeline failure is an invalid episode, counted rather than fatal.
    assert arm["valid_episodes"] < arm["episodes"]
    assert arm["invalid_episodes_by_reason"] == {
        TerminationOutcome.ACTION_PIPELINE_FAILED.value: 1
    }
