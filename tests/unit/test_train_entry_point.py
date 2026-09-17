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
    "weighted_loss",
    "unweighted_mean_absolute_td_error",
    "gradient_norm",
    "value_fit_correlation",
    "collection_wait_fraction",
    "collection_purchases_per_episode",
    "pre_registered_final",
}

#: One point of the collection curve, as the report must carry it.
WINDOW_KEYS = {
    "index",
    "episodes",
    "decisions",
    "decisions_at_end",
    "mean_final_wave",
    "stdev_final_wave",
    "standard_error",
    "wait_fraction",
    "purchases_per_episode",
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
        "--warmup-sequences": "2",
        "--sequence-length": "6",
        "--burn-in": "3",
        # Per arm: the stacked backbone needs exactly `history-length - 1`.
        "--stacked-burn-in": "3",
        "--history-length": "4",
        "--replay-capacity": "64",
        "--evaluate-every-episodes": "1",
        "--evaluation-episodes": "2",
        # A run this short would never close a hundred-episode window.
        "--collection-window-episodes": "2",
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


#: Long enough that every arm plays more than one episode, which is what closes
#: a window of the collection curve.
INTERLEAVED_BUDGET = "300"


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One interleaved session over both backbones, reused by several checks."""
    run_dir = tmp_path_factory.mktemp("runs")
    return session(run_dir, "recurrent-q", "stacked-dqn", budget=INTERLEAVED_BUDGET)


def test_both_backbones_train_under_one_interleaved_budget(trained: dict[str, Any]) -> None:
    arms = trained["arms"]

    assert [arm["backbone"] for arm in arms] == ["recurrent-q", "stacked-dqn"]
    for arm in arms:
        assert arm["decisions"] >= int(INTERLEAVED_BUDGET), "every arm spends the budget"
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


def test_the_regime_the_run_is_pinned_to_is_what_the_defaults_say(tmp_path: Path) -> None:
    """The settings of the second training run, where the developer reads them.

    Pinned as a test because every one of them was chosen against a measured
    failure of the first run; a silent drift back would cost another run of
    device time to discover.
    """
    defaults = train.parse_arguments(["--run-dir", str(tmp_path)])

    assert defaults.gradient_steps_per_decision == 0.25
    assert defaults.batch_size == 8
    assert defaults.warmup_sequences == 100
    assert defaults.sequence_length == 80
    assert defaults.stacked_burn_in == defaults.history_length - 1 == 7
    assert defaults.burn_in == 40, "the recurrent arm reconstructs a state, not a window"
    assert defaults.n_step == 10
    assert defaults.discount == 0.99
    assert defaults.learning_rate == 1e-4
    assert defaults.target_ema_decay == 0.995
    assert (defaults.epsilon_start, defaults.epsilon_end) == (1.0, 0.05)
    assert defaults.epsilon_anneal_decisions == 10_000
    assert defaults.priority_alpha == 0.0, "importance weights of exactly one"
    assert defaults.replay_capacity == 4096
    assert defaults.collection_window_episodes == 100
    assert defaults.evaluate_every_episodes == 0, "no frequent mid-run evaluation"
    assert defaults.evaluation_episodes == 30


def _arm(run_dir: Path, name: str, **overrides: str) -> Any:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        return train.build_arm(
            name,
            arguments(run_dir, name, **overrides),
            environment=environment(),
            device=torch.device("cpu"),
            profile_id=PROFILE,
            parent=run_dir,
            revision="test",
            started=0.0,
            tracker=train.NoExperimentTracker(),
            tags={},
        )


def test_every_flag_reaches_the_thing_it_configures(tmp_path: Path) -> None:
    """A flag that reaches nothing is worse than no flag: it looks like a knob."""
    arm = _arm(
        tmp_path,
        "stacked-dqn",
        **{
            "--n-step": "3",
            "--discount": "0.9",
            "--learning-rate": "0.002",
            "--target-ema-decay": "0.9",
            "--warmup-sequences": "7",
            "--epsilon-start": "0.8",
            "--epsilon-end": "0.02",
            "--epsilon-anneal-decisions": "77",
            "--priority-alpha": "0.3",
            "--collection-window-episodes": "5",
            "--gradient-steps-per-decision": "0.25",
            "--batch-size": "4",
        },
    )

    learner = arm.backbone.config
    assert (learner.n_step, learner.discount, learner.learning_rate) == (3, 0.9, 0.002)
    assert learner.target_ema_decay == 0.9
    assert arm.replay.alpha == 0.3
    config = arm.training.config
    assert config.warmup_sequences == 7
    assert (config.epsilon_start, config.epsilon_end) == (0.8, 0.02)
    assert config.epsilon_anneal_decisions == 77
    assert config.collection_window_episodes == 5
    assert (config.batch_size, config.gradient_steps_per_decision) == (4, 0.25)
    # And the run records what it was actually built with.
    resolved = arm.resolved
    assert resolved["n_step"] == 3 and resolved["discount"] == 0.9
    assert resolved["priority_alpha"] == 0.3
    assert resolved["epsilon_anneal_decisions"] == 77
    assert resolved["target_ema_decay"] == 0.9


def test_the_two_backbones_burn_in_differently(tmp_path: Path) -> None:
    """Burn-in means two different things, so one number cannot serve both arms.

    The stacked arm's burn-in only fills its history window; anything past
    `history_length - 1` throws learnable steps away. The recurrent arm's burn-in
    reconstructs a stored LSTM state and needs the length it was tuned with.
    """
    stacked = _arm(tmp_path / "stacked", "stacked-dqn", **{"--stacked-burn-in": "3"})
    recurrent = _arm(tmp_path / "recurrent", "recurrent-q", **{"--burn-in": "4"})

    assert stacked.training.actor.config.burn_in == 3
    assert recurrent.training.actor.config.burn_in == 4
    assert stacked.resolved["burn_in"] == 3
    assert recurrent.resolved["burn_in"] == 4


def test_a_stacked_burn_in_too_short_for_the_window_is_refused(tmp_path: Path) -> None:
    """Checked before the device is touched, not an hour into collection."""
    with pytest.raises(SystemExit, match="cannot fill a window"):
        arguments(tmp_path, "stacked-dqn", **{"--stacked-burn-in": "2"})


def test_the_report_carries_the_collection_curve(trained: dict[str, Any]) -> None:
    """The series the run is read from: collected episodes, in closed windows."""
    for arm in trained["arms"]:
        curve = arm["collection_curve"]

        assert curve, "a run of several episodes closes at least one window"
        assert all(set(window) == WINDOW_KEYS for window in curve)
        assert [window["index"] for window in curve] == list(range(len(curve)))
        assert arm["collection_window_episodes"] == 2
        placements = [window["decisions_at_end"] for window in curve]
        assert placements == sorted(placements)
        for window in curve:
            assert window["episodes"] == 2
            assert window["mean_final_wave"] > 0
            assert window["standard_error"] is not None
            assert 0.0 <= window["wait_fraction"] <= 1.0
            assert window["purchases_per_episode"] >= 0.0
        # The windows do not overlap, so their episodes sum to what was scored
        # without double counting; a trailing partial window is not a point.
        scored = sum(window["episodes"] for window in curve)
        assert scored <= arm["valid_episodes"] < scored + 2


def test_the_run_ends_on_one_pre_registered_exploration_free_evaluation(
    trained: dict[str, Any],
) -> None:
    for arm in trained["arms"]:
        final = arm["final_evaluation"]

        assert final is not None and final["pre_registered_final"] is True
        # It scores the final weights, after the budget was spent.
        assert final["decisions"] == arm["decisions"]
        assert final["model_version"] == arm["optimisation_steps"]
        assert final["versus_scripted_reference"] == pytest.approx(
            final["mean_final_wave"] - train.SCRIPTED_REFERENCE, abs=1e-3
        )
        # Exactly one, and it is the last point on the curve.
        headline = [
            point for point in arm["learning_curve"] if point["pre_registered_final"]
        ]
        assert headline == [final] == [arm["learning_curve"][-1]]


def test_the_learner_diagnostics_travel_with_every_point(trained: dict[str, Any]) -> None:
    """Without them a flat curve cannot be told from a broken learner."""
    for arm in trained["arms"]:
        point = arm["final_evaluation"]

        assert point["weighted_loss"] is not None
        assert point["unweighted_mean_absolute_td_error"] is not None
        # Two names, because they are two quantities: the loss carries the
        # importance-sampling weights and moves with the beta schedule.
        assert point["weighted_loss"] != point["unweighted_mean_absolute_td_error"]
        assert point["gradient_norm"] is not None
        fit = point["value_fit_correlation"]
        assert fit is None or -1.0 <= fit <= 1.0
        assert 0.0 <= point["collection_wait_fraction"] <= 1.0
        assert point["collection_purchases_per_episode"] >= 0.0
        distribution = arm["action_distribution"]
        assert distribution["episodes"] == arm["episodes"] - arm["failed_episodes"]
        assert distribution["decisions"] == arm["decisions"]
