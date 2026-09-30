"""The training entry point, end to end against the fake port.

No emulator, no adb, no bridge: `train_session` takes the instances it trains
against, so the double never reaches a path a device run can take. What is under
test is the thing the developer actually starts - argument parsing, arm
construction, the interleaved block schedule, periodic evaluation and
checkpointing, the learning curve the run is read from, and the fleet: a session
on several instances, its per-actor account, and its staggered bring-up.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy
import pytest
import torch
import train
from fakes.fake_run_port import FakeRunPort
from fakes.recording_tracker import RecordingTracker

from tower_rl.environment.episode import TerminationOutcome
from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.environment.run_environment import (
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.experiment.metrics import per_hour
from tower_rl.experiment.tracking import NoExperimentTracker
from tower_rl.experiment.training_report import (
    REPLAY_DIRECTORY,
    non_finite_tensors,
    numbered_checkpoint_name,
)
from tower_rl.learning.actor import Actor, ActorConfig
from tower_rl.learning.checkpoint import (
    Checkpoint,
    identity_hash,
    load,
    restore_rng_state,
    save,
)
from tower_rl.learning.evaluator import evaluate
from tower_rl.learning.exploration import ape_x_floors
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.policies import Policy, checkpoint_policy
from tower_rl.learning.replay import (
    R2D2_PRIORITY_EXPONENT,
    REPLAY_DUMP_METADATA,
    PrioritizedSequenceReplay,
    ReplayImage,
    ReplayStep,
    read_replay_metadata,
)
from tower_rl.learning.stacked_dqn import StackedDqnBackbone
from tower_rl.learning.training import TrainingRun
from tower_rl.learning.value_learning import V_REF, n_step_targets
from tower_rl.simulation.instance import CloneInstance

#: Tensors this small spend their time handing work between threads rather than
#: computing: one thread runs the whole file about fifteen times faster.

PROFILE = "fake-profile-v1"

#: A network narrow enough that the entry point can be exercised in seconds. At
#: production width every decision is a CPU forward pass and dominates the run;
#: what is under test here is the plumbing around the learner, not its capacity,
#: which the backbone contract suite covers.
SMALL_NETWORK = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

def arguments(run_dir: Path, **overrides: str | None) -> argparse.Namespace:
    """The real parser, so the entry point's own defaults and checks are used.

    A value of None gives the flag alone, as a switch is given.
    """
    argv: list[str] = []
    settings = {
        "--budget-decisions": "200",
        "--batch-size": "2",
        "--gradient-steps-per-decision": "0.2",
        "--warmup-sequences": "2",
        "--sequence-length": "6",
        # Exactly `history-length - 1`, which is what fills the window.
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
        argv += [flag] if value is None else [flag, value]
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


def fleet(count: int = 1, **fake: Any) -> list[train.ActorInstance]:
    """`count` independent fake instances, named as a fleet's instances are."""
    return [
        train.ActorInstance(serial=f"fake-{index}", environment=environment(**fake))
        for index in range(count)
    ]


def session(
    run_dir: Path,
    budget: str = "200",
    actors: int = 1,
    settings: dict[str, str | None] | None = None,
    tracker: Any = None,
    resume: Any = None,
    **fake: Any,
) -> dict[str, Any]:
    overrides = {"--budget-decisions": budget, **(settings or {})}
    if actors > 1:
        overrides.update(
            {
                "--actors": str(actors),
                # A fleet addresses its instances through `CloneInstance`, so
                # the single-actor serial may not be named beside it.
                "--serial": CloneInstance(index=0).serial,
                # Mid-run evaluation would need an instance to itself.
                "--evaluate-every-episodes": "0",
            }
        )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        return train.train_session(
            arguments(run_dir, **overrides),
            fleet(actors, **fake),
            profile_id=PROFILE,
            revision="test",
            device=torch.device("cpu"),
            tracker=tracker,
            resume=resume,
        )


#: Long enough that the arm plays more than one episode, which is what closes
#: a window of the collection curve.
TRAINING_BUDGET = "300"


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One short session, reused by several checks."""
    run_dir = tmp_path_factory.mktemp("runs")
    return session(run_dir, budget=TRAINING_BUDGET)


def test_the_backbone_trains_under_one_budget(trained: dict[str, Any]) -> None:
    arm = trained["arm"]

    assert arm["backbone"] == "stacked-dqn"
    assert arm["decisions"] >= int(TRAINING_BUDGET), "the arm spends the budget"
    assert arm["episodes"] > 0
    assert arm["optimisation_steps"] > 0
    assert arm["sequences_accepted"] > 0
    assert arm["failed_episodes"] == 0


def test_evaluation_runs_without_exploration(tmp_path: Path) -> None:
    """The entry point's evaluation goes through `evaluate`, which forces zero."""
    seen: list[float] = []

    class RecordingPolicy:
        def initial_state(self) -> None:
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
    report = session(tmp_path, refuse_episodes=frozenset({2, 3}))

    # Episode ordinals are consumed by evaluation episodes too, so one refusal
    # lands on collection and one on an evaluation. Neither may end the session.
    arm = report["arm"]
    assert arm["failed_episodes"] >= 1
    assert len(arm["evaluation_failures"]) >= 1
    assert arm["failed_episodes"] + len(arm["evaluation_failures"]) == 2
    # The budget is still spent and the curve still produced.
    assert arm["decisions"] >= 200
    assert arm["learning_curve"]


def test_an_ambiguous_advance_is_classified_and_the_session_continues(
    tmp_path: Path,
) -> None:
    # Ordinal 1 is the first collected episode; evaluation episodes take the
    # ordinals after it.
    report = session(tmp_path, ambiguous_advance_episodes=frozenset({1}))

    arm = report["arm"]
    assert arm["failed_episodes"] == 0, "the port answered; the episode did not"
    assert arm["episodes"] > 1 and arm["decisions"] >= 200
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
    defaults = train.parse_arguments(["--budget-decisions", "1000", "--run-dir", str(tmp_path)])

    # M3-P009's replay ratio, reverted from M3-P010's 0.114 (board #85,
    # solution.md 9.4).
    assert defaults.gradient_steps_per_decision == 1.0
    assert defaults.batch_size == 8
    assert defaults.warmup_sequences == 100
    assert defaults.sequence_length == 80
    assert defaults.stacked_burn_in == defaults.history_length - 1 == 7
    assert defaults.n_step == 10
    assert defaults.discount == 0.99
    assert defaults.learning_rate == 1e-4
    assert defaults.target_ema_decay == 0.995
    assert (defaults.epsilon_start, defaults.epsilon_end) == (1.0, 0.05)
    assert defaults.epsilon_anneal_decisions == 10_000
    assert defaults.exploration == "uniform", "run 1's schedule is the default"
    # M3-P009's known-good capacity, reverted from M3-P010's 25,000 (board #85).
    assert defaults.replay_capacity == 4096
    assert defaults.collection_window_episodes == 100
    assert defaults.evaluate_every_episodes == 0, "no frequent mid-run evaluation"
    assert defaults.evaluation_episodes == 30
    # Ape-X's refresh of every 400 frames, 100 agent steps: a fleet's actors act
    # from copies of the network, refreshed inside an episode as well.
    assert defaults.parameter_sync_decisions == 100


def test_a_stacked_burn_in_too_short_for_the_window_is_refused(tmp_path: Path) -> None:
    """Checked before the device is touched, not an hour into collection."""
    with pytest.raises(SystemExit, match="cannot fill a window"):
        arguments(tmp_path, **{"--stacked-burn-in": "2"})


# -- --backbone dreamerv3 ------------------------------------------------------
#
# Its fixed loop settings, and one session at a size that trains in seconds, as
# this suite shrinks stacked-dqn's network; the published sizes are what
# `DreamerConfig()` holds and what the parse tests read without the patch.

#: Batch 2 x 6 at a train ratio of 3: 0.25 gradient steps per decision (a size
#: chosen to be small and fast, not to match the published train ratio).
SMALL_DREAMER: dict[str, Any] = dict(
    deter=16, hidden=8, classes=4, units=8, stoch=4, blocks=2,
    batch_size=2, batch_length=6, train_ratio=3.0,
)


#: The protocol's discount horizon (ADR 0013), which a DreamerV3 run must give.
DREAMER_DISCOUNT = ("--discount-per-game-second", "0.999")


def dreamer_arguments(run_dir: Path, *flags: str) -> argparse.Namespace:
    return train.parse_arguments(
        [
            "--budget-decisions", "1000", "--run-dir", str(run_dir), "--backbone", "dreamerv3",
            *DREAMER_DISCOUNT, *flags,
        ]
    )


def dreamer_session(run_dir: Path, *flags: str) -> dict[str, Any]:
    """One DreamerV3 session at a size that trains in seconds, on the fake fleet."""
    from tower_rl.learning.dreamer import DreamerConfig

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            train, "DreamerConfig", lambda **given: DreamerConfig(**{**SMALL_DREAMER, **given})
        )
        patch.setattr(train, "DREAMER_WARMUP_SEQUENCES", 2)
        patch.setattr(train, "DREAMER_REPLAY_CAPACITY", 64)
        parsed = dreamer_arguments(
            run_dir,
            "--budget-decisions", "200",
            "--evaluate-every-episodes", "1",
            "--evaluation-episodes", "2",
            "--collection-window-episodes", "2",
            "--checkpoint-every-episodes", "2",
            "--serial", "fake-0",
            "--max-quiet-game-ms", "4000",
            *flags,
        )
        report: dict[str, Any] = train.train_session(
            parsed, fleet(1), profile_id=PROFILE, revision="test", device=torch.device("cpu")
        )
    return report


def test_dreamerv3_fixes_its_published_loop_settings(tmp_path: Path) -> None:
    parsed = dreamer_arguments(tmp_path)
    assert parsed.backbone == "dreamerv3"
    assert (parsed.sequence_length, parsed.stacked_burn_in) == (64, 0)
    assert parsed.batch_size == 16
    assert parsed.gradient_steps_per_decision == 0.5
    # One batch of items, and the official replay size in items (steps).
    assert parsed.warmup_sequences == 16 * 64
    assert parsed.replay_capacity == 5_000_000
    assert parsed.exploration == "uniform"
    assert (parsed.epsilon_start, parsed.epsilon_end) == (0.0, 0.0)


def test_the_default_backbone_is_stacked_dqn(tmp_path: Path) -> None:
    parsed = train.parse_arguments(["--budget-decisions", "1000", "--run-dir", str(tmp_path)])
    assert parsed.backbone == "stacked-dqn"


def test_a_flag_that_repeats_a_dreamerv3_value_is_accepted(tmp_path: Path) -> None:
    parsed = dreamer_arguments(tmp_path, "--batch-size", "16", "--sequence-length", "64")
    assert parsed.batch_size == 16


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--batch-size", "8"),
        ("--sequence-length", "80"),
        ("--stacked-burn-in", "7"),
        ("--gradient-steps-per-decision", "1.0"),
        ("--warmup-sequences", "100"),
        ("--epsilon-start", "1.0"),
        ("--epsilon-end", "0.05"),
        ("--exploration", "ladder"),
    ],
)
def test_a_flag_that_contradicts_a_dreamerv3_value_is_refused(
    tmp_path: Path, flag: str, value: str
) -> None:
    with pytest.raises(SystemExit, match="contradicts DreamerV3|ladder"):
        dreamer_arguments(tmp_path, flag, value)


@pytest.mark.parametrize(
    "flags",
    [
        ("--history-length", "8"),
        ("--n-step", "3"),
        ("--discount", "0.99"),
        ("--learning-rate", "1e-4"),
        ("--target-ema-decay", "0.995"),
        ("--n-step-final", "3", "--n-step-anneal-steps", "100"),
        ("--epsilon-anneal-decisions", "8000"),
    ],
)
def test_a_stacked_dqn_flag_is_refused_under_dreamerv3(
    tmp_path: Path, flags: tuple[str, ...]
) -> None:
    with pytest.raises(SystemExit, match="stacked-dqn setting"):
        dreamer_arguments(tmp_path, *flags)


def test_a_dreamerv3_session_trains_and_its_checkpoint_plays_per_instance_streams(
    tmp_path: Path,
) -> None:
    from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT
    from tower_rl.environment.run_actions import RUN_ACTIONS
    from tower_rl.learning.dreamer import DreamerBackbone, DreamerConfig

    report = dreamer_session(tmp_path)
    arm = report["arm"]
    assert arm["backbone"] == "dreamerv3"
    assert arm["decisions"] >= 200 and arm["optimisation_steps"] > 0
    assert arm["failed_episodes"] == 0
    resolved = arm["resolved_config"]
    assert (resolved["sequence_length"], resolved["burn_in"], resolved["stride"]) == (6, 0, 3)
    assert resolved["dreamer_deter"] == 16 and resolved["dreamer_train_ratio"] == 3.0
    # On the CPU the learner computes in float32; on CUDA, in bfloat16.
    assert resolved["dreamer_compute_dtype"] == "float32"
    # Uniform replay, as the official loop samples; not an option (board #85).
    assert (resolved["priority_alpha"], resolved["importance_beta"]) == (0.0, 0.0)
    for stacked in ("history_length", "n_step", "discount", "learning_rate", "network_hidden"):
        assert resolved[stacked] is None, stacked
    # The task's discount and reward, under the keys stacked-dqn records them under.
    assert resolved["discount_per_game_second"] == 0.999
    assert resolved["survival_time_reward"] is False
    assert resolved["survival_reward_bound"] is None
    assert "dreamer_horizon" not in resolved

    latest = Path(report["run_folder"]) / "checkpoints" / "latest.pt"
    policy, identity = checkpoint_policy(
        latest,
        decision_cadence=resolved["decision_cadence"],
        upgrade_availability=resolved["upgrade_availability"],
        workshop_level=0,
    )
    assert identity.backbone == "dreamerv3"
    assert isinstance(policy, DreamerBackbone)
    assert policy.config == DreamerConfig(
        **SMALL_DREAMER, seed=0, discount_per_game_second=0.999
    )
    assert policy.model_version == arm["optimisation_steps"]
    assert not any(parameter.requires_grad for parameter in policy.actor.parameters())

    # Each evaluating instance samples the checkpoint's policy from a stream of
    # its own, seeded by its serial as `run_episodes.py` does; the same serial
    # draws the same stream.
    def sampled(serial: str) -> list[int]:
        instance, _ = checkpoint_policy(
            latest,
            decision_cadence=resolved["decision_cadence"],
            upgrade_availability=resolved["upgrade_availability"],
            workshop_level=0,
            sampling_seed=serial,
        )
        every = tuple(True for _ in range(len(RUN_ACTIONS)))
        state = instance.initial_state()
        chosen = []
        for index in range(40):
            features = StateFeatures(
                scalars=tuple([0.01 * index] * SCALAR_COUNT),
                rows=tuple([0.01 * index] * (ROW_COUNT * ROW_WIDTH)),
                mask=every,
            )
            action, state = instance.act(features, state, epsilon=0.0)
            chosen.append(action)
        return chosen

    assert sampled("emulator-5556") == sampled("emulator-5556")
    assert sampled("emulator-5556") != sampled("emulator-5558")


#: A checkpoint cadence and a selection period short enough that a test budget
#: crosses each several times, and deliberately not multiples of each other:
#: the two are independent, as run 5b's 5,000 and 15,000 are.
CHECKPOINT_CADENCE = 70
SELECTION_PERIOD = 100


def numbered_checkpoints(report: dict[str, Any]) -> tuple[dict[str, Any], Path, list[int]]:
    """The arm, its checkpoint directory, and the decisions each file names."""
    arm = report["arm"]
    directory = Path(report["run_folder"]) / "checkpoints"
    files = directory.glob("checkpoint-d*.pt")
    return arm, directory, sorted(int(path.stem.removeprefix("checkpoint-d")) for path in files)


def test_a_numbered_checkpoint_is_written_at_every_crossing_of_the_cadence_or_a_period(
    tmp_path: Path,
) -> None:
    """One checkpoint per episode that crosses a multiple of either, named by decisions.

    The counter lands past a multiple rather than on it - an episode is played
    to its classified end - and one episode crossing both, or several multiples
    at once, is one checkpoint. The expected files are recomputed here from the
    run's own episode series rather than assumed to be one per multiple.
    """
    report = session(
        tmp_path,
        budget="600",
        settings={
            "--checkpoint-every-decisions": str(CHECKPOINT_CADENCE),
            "--selection-period-decisions": str(SELECTION_PERIOD),
        },
    )
    arm, directory, written = numbered_checkpoints(report)

    spent = 0
    expected: list[int] = []
    for episode in arm["collected_episodes"]:
        before, spent = spent, spent + int(episode["decisions"])
        every = (CHECKPOINT_CADENCE, SELECTION_PERIOD)
        crossed = (spent // step > before // step for step in every)
        if any(crossed):
            expected.append(spent)

    assert len(written) >= 2, "a 600-decision budget crosses both several times"
    assert written == expected
    # Every selection period closes on a checkpoint, so whichever one the arm
    # rule picks is a file on disk.
    closes = {period["decisions_at_end"] for period in arm["selection_periods"]}
    assert closes and closes <= set(written)
    # Beside the resume point, which is overwritten and names no one model.
    assert (directory / "latest.pt").exists()


def test_a_numbered_checkpoint_carries_the_run_it_came_from(tmp_path: Path) -> None:
    """Each one is resumable and says which run, and which decisions, made it."""
    report = session(
        tmp_path,
        budget="300",
        settings={"--checkpoint-every-decisions": str(CHECKPOINT_CADENCE)},
    )
    arm, directory, written = numbered_checkpoints(report)
    assert written, "the budget crosses the cadence at least once"

    for decisions in written:
        checkpoint = load(directory / numbered_checkpoint_name(decisions))
        assert checkpoint.identity.run_id == arm["run_id"]
        assert checkpoint.identity.backbone == "stacked-dqn"
        assert checkpoint.identity.profile_id == PROFILE
        # The name is the decisions the progress in the file records, not an
        # aspiration; a reader still goes by the file, never by the name.
        assert checkpoint.progress.environment_decisions == decisions
        # Game time travels with it as a statistic.
        assert checkpoint.progress.environment_game_ms > 0
        # The width of the network, so the checkpoint can be rebuilt into the
        # policy that wrote it without being told what shape it is.
        assert checkpoint.resolved_config["network_hidden"] == SMALL_NETWORK.hidden


def test_by_default_numbered_checkpoints_are_written_only_where_periods_close(
    tmp_path: Path,
) -> None:
    """No cadence by default, and a run shorter than one period writes none."""
    defaults = train.parse_arguments(["--budget-decisions", "1000", "--run-dir", str(tmp_path)])
    assert defaults.checkpoint_every_decisions == 0
    assert defaults.selection_period_decisions == 15_000

    _, directory, written = numbered_checkpoints(session(tmp_path, budget="200"))

    assert written == []
    assert (directory / "latest.pt").exists()


def test_early_stopping_is_off_by_default_and_resolved_with_its_threshold(
    tmp_path: Path,
) -> None:
    """Every run measured so far spent its whole budget; that stays the default."""
    defaults = train.parse_arguments(["--budget-decisions", "1000", "--run-dir", str(tmp_path)])

    assert defaults.early_stop_patience_periods == 0
    assert defaults.early_stop_min_improvement == 0.2


def test_the_n_step_anneal_and_kill_bars_are_off_by_default(tmp_path: Path) -> None:
    defaults = train.parse_arguments(["--budget-decisions", "1000", "--run-dir", str(tmp_path)])

    assert defaults.n_step == 10
    assert defaults.n_step_final is None and defaults.n_step_anneal_steps == 0
    assert defaults.kill_bars == []


def test_a_half_configured_n_step_anneal_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="together"):
        arguments(tmp_path, **{"--n-step-final": "3"})
    with pytest.raises(SystemExit, match="together"):
        arguments(tmp_path, **{"--n-step-anneal-steps": "100"})


def test_a_kill_bar_is_parsed_as_at_start_minimum(tmp_path: Path) -> None:
    parsed = train.parse_arguments(
        [
            "--budget-decisions", "1000", "--run-dir", str(tmp_path),
            "--kill-bar", "12000:8000:8.6",
            "--kill-bar", "26262:8000:10.2",
        ]
    )

    assert [
        (bar.at_decisions, bar.window_start_decisions, bar.min_mean_final_wave)
        for bar in parsed.kill_bars
    ] == [(12000, 8000, 8.6), (26262, 8000, 10.2)]
    for malformed in ("12000:8000", "8000:12000:8.6", "a:b:c"):
        with pytest.raises(SystemExit):
            train.parse_arguments(
                ["--budget-decisions", "1000", "--run-dir", str(tmp_path), "--kill-bar", malformed]
            )


def test_a_run_below_its_kill_bar_stops_and_says_which_bar(tmp_path: Path) -> None:
    """The stop is the run's own and is recorded like the plateau stop."""
    report = session(
        tmp_path,
        budget=TRAINING_BUDGET,
        settings={
            "--kill-bar": "100:0:1000",
            "--n-step-final": "3",
            "--n-step-anneal-steps": "5",
        },
    )

    arm = report["arm"]
    stopping = arm["early_stopping"]
    assert stopping["early_stopped"]
    [check] = stopping["kill_bar_checks"]
    assert check["stopped"] and check["bar"]["at_decisions"] == 100
    assert check["mean_final_wave"] < 1000
    assert arm["decisions"] < int(TRAINING_BUDGET)
    # A killed run skips the final evaluation and says so.
    assert arm["final_evaluation"] is None
    assert arm["final_evaluation_skipped"] == "kill_bar"
    resolved = arm["resolved_config"]
    assert resolved["kill_bars"] == [[100, 0, 1000.0]]
    assert (resolved["n_step"], resolved["n_step_final"], resolved["n_step_anneal_steps"]) == (
        10,
        3,
        5,
    )


def test_a_run_not_stopped_on_a_kill_bar_still_takes_its_final_evaluation(
    trained: dict[str, Any],
) -> None:
    arm = trained["arm"]

    assert arm["final_evaluation"] is not None
    assert arm["final_evaluation_skipped"] is None


def test_a_plateau_stop_still_takes_its_final_evaluation(tmp_path: Path) -> None:
    """Only a kill bar skips it: a plateau stop still evaluates."""
    report = session(
        tmp_path,
        budget="1000",
        settings={
            "--selection-period-decisions": "50",
            "--early-stop-patience-periods": "1",
            "--early-stop-min-improvement": "1000",
        },
    )

    arm = report["arm"]
    assert arm["early_stopping"]["stopped_at_period"] is not None
    assert arm["final_evaluation"] is not None
    assert arm["final_evaluation_skipped"] is None


def test_a_negative_cadence_or_an_empty_selection_period_is_refused(tmp_path: Path) -> None:
    """Checked in the parser, before any device is touched."""
    with pytest.raises(SystemExit, match="cannot be negative"):
        arguments(tmp_path, **{"--checkpoint-every-decisions": "-1"})
    with pytest.raises(SystemExit, match="--selection-period-decisions"):
        arguments(tmp_path, **{"--selection-period-decisions": "0"})
    with pytest.raises(SystemExit, match="--budget-decisions"):
        arguments(tmp_path, **{"--budget-decisions": "0"})


def test_the_budget_is_the_only_progress_flag_and_it_is_in_decisions(tmp_path: Path) -> None:
    """The game-time flags are gone, not deprecated beside their replacements."""
    for retired in (
        "--budget-game-seconds",
        "--block-game-seconds",
        "--checkpoint-every-game-seconds",
    ):
        with pytest.raises(SystemExit):
            train.parse_arguments(
                ["--budget-decisions", "1000", "--run-dir", str(tmp_path), retired, "100"]
            )


#: Two fake instances, which is the fleet arrangement a device run takes: every
#: block of collection uses every actor.
FLEET_BUDGET = "300"


@pytest.fixture(scope="module")
def fleet_trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    run_dir = tmp_path_factory.mktemp("fleet")
    return session(run_dir, budget=FLEET_BUDGET, actors=2)


def test_a_fleet_trains_the_backbone_under_one_budget(
    fleet_trained: dict[str, Any],
) -> None:
    """The arm still works, and spends its budget across both instances."""
    assert fleet_trained["actors"] == 2
    assert fleet_trained["actor_serials"] == ["fake-0", "fake-1"]
    assert fleet_trained["bring_up_failures"] == []

    arm = fleet_trained["arm"]
    assert arm["decisions"] >= int(FLEET_BUDGET)
    assert arm["optimisation_steps"] > 0 and arm["sequences_accepted"] > 0
    assert arm["resolved_config"]["actors"] == 2
    assert arm["resolved_config"]["actor_ids"] == [
        f"fake-0:{arm['backbone']}",
        f"fake-1:{arm['backbone']}",
    ]
    # The pre-registered evaluation still lands, taken with the fleet stopped.
    assert arm["final_evaluation"]["pre_registered_final"] is True


def test_one_dead_instance_does_not_end_a_fleet_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One emulator refusing every episode costs an actor, not the run."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        report = train.train_session(
            arguments(
                tmp_path,
                **{
                    "--budget-decisions": "40",
                    "--actors": "2",
                    "--serial": CloneInstance(index=0).serial,
                    "--evaluate-every-episodes": "0",
                },
            ),
            [
                train.ActorInstance(serial="fake-0", environment=environment()),
                train.ActorInstance(
                    serial="fake-1", environment=environment(refuse_to_start=True)
                ),
            ],
            profile_id=PROFILE,
            revision="test",
            device=torch.device("cpu"),
        )

    arm = report["arm"]
    dead = next(actor for actor in arm["actors"] if actor["actor_id"] == "fake-1:stacked-dqn")
    alive = next(actor for actor in arm["actors"] if actor["actor_id"] == "fake-0:stacked-dqn")
    assert dead["withdrawn"] is not None and dead["failed_episodes"] > 0
    assert arm["actors_withdrawn"] == 1
    assert alive["decisions"] == arm["decisions"] > 0
    assert arm["game_seconds"] == pytest.approx(alive["game_seconds"])
    assert arm["final_evaluation"] is not None, "the run was still measured"
    # A withdrawal is invisible in the aggregate, so it is announced when it
    # happens, naming the instance that left and what took it out.
    announcement = next(
        line for line in capsys.readouterr().out.splitlines() if "withdrawn" in line
    )
    assert "fake-1:stacked-dqn" in announcement and dead["withdrawn"] in announcement


def test_a_single_actor_run_records_exactly_one_actor(tmp_path: Path) -> None:
    """The default, and the configuration the in-flight run is reproducible from."""
    parsed = train.parse_arguments(["--budget-decisions", "1000", "--run-dir", str(tmp_path)])
    assert parsed.actors == 1

    report = session(tmp_path)

    assert report["actors"] == 1 and report["actor_serials"] == ["fake-0"]
    arm = report["arm"]
    assert arm["resolved_config"]["actors"] == 1
    assert [actor["actor_id"] for actor in arm["actors"]] == ["fake-0:stacked-dqn"]
    assert arm["actors"][0]["decisions"] == arm["decisions"]


def test_a_fleet_refuses_an_instance_named_by_hand(tmp_path: Path) -> None:
    """--serial and --port configure one actor; a fleet is addressed by index."""
    with pytest.raises(SystemExit, match="CloneInstance"):
        arguments(tmp_path, **{"--actors": "2", "--serial": "fake-0"})


def test_a_fleet_refuses_mid_run_evaluation(tmp_path: Path) -> None:
    """Evaluation borrows an instance, and every instance is collecting."""
    with pytest.raises(SystemExit, match="instance to itself"):
        arguments(
            tmp_path,
            **{
                "--actors": "2",
                "--serial": CloneInstance(index=0).serial,
                "--evaluate-every-episodes": "1",
            },
        )


def test_a_run_needs_at_least_one_actor(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="at least one actor"):
        arguments(tmp_path, **{"--actors": "0"})


def test_an_instance_whose_bring_up_fails_is_still_torn_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bring-up that fails partway must not leave its emulator running.

    `main`'s `open_instance` used to append to `started` only once bring-up had
    succeeded, so an instance whose bring-up raised was never in the list
    `tear_down_fleet` sweeps: the emulator it may already have launched
    survived a clean `exit 0`. Registration now happens before the attempt, so
    this exercises `main` end to end - against fakes, never a device - to prove
    the failed instance is still handed to teardown.
    """
    monkeypatch.setenv("TOWER_BRIDGE_BUILD_DIR", str(tmp_path))
    expected = SimpleNamespace(profile_id=PROFILE, bridge_version="v1")
    monkeypatch.setattr(train, "compatibility", lambda build_dir: expected)
    monkeypatch.setattr(train, "prepare_pinned_snapshot", lambda renderer, cores: "snap")
    monkeypatch.setattr(train, "require_offline", lambda instance: None)
    monkeypatch.setattr(train, "require_game_activity", lambda instance: None)
    monkeypatch.setattr(train, "raise_frame_rate", lambda instance, frame_rate_hz: None)
    monkeypatch.setattr(
        train,
        "connect",
        lambda serial, port, arguments, expected, opened: train.ActorInstance(
            serial=serial, environment=environment()
        ),
    )

    def fake_bring_up(
        instance: CloneInstance,
        renderer: str,
        *,
        deploy: Any,
        read_only: bool,
        cores: int,
        frame_rate_hz: int,
    ) -> str:
        if instance.index == 1:
            raise RuntimeError("cold boot never reached home")
        return "cold"

    monkeypatch.setattr(train, "bring_up", fake_bring_up)

    torn: list[str] = []

    def fake_tear_down(instance: CloneInstance) -> None:
        torn.append(instance.serial)

    # `tear_down_fleet`'s teardown callable is bound as a default parameter at
    # definition time, exactly as `main` calls it with none supplied, so the
    # spy has to replace that default rather than pass an explicit argument.
    monkeypatch.setattr(train.tear_down_fleet, "__defaults__", (fake_tear_down,))

    captured: dict[str, Any] = {}

    def fake_train_session(
        arguments: argparse.Namespace, instances: list[train.ActorInstance], **kwargs: Any
    ) -> dict[str, Any]:
        captured["instances"] = list(instances)
        captured["bring_up_failures"] = kwargs["bring_up_failures"]
        return {}

    monkeypatch.setattr(train, "train_session", fake_train_session)
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--budget-decisions", "1000", "--actors", "2", "--no-track"]
    )

    exit_code = train.main()

    assert exit_code == 0
    # Both instances are torn down, including the one whose bring-up failed.
    assert torn == ["emulator-5556", "emulator-5558"]
    assert captured["bring_up_failures"] and "emulator-5558" in captured["bring_up_failures"][0]
    assert [item.serial for item in captured["instances"]] == ["emulator-5556"]


def test_a_fleet_raises_every_instance_after_its_own_bring_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every instance is raised off the stock 60 Hz right after its bring-up.

    `run_actors.collect_episodes` raises an instance's rate itself, the moment
    its own bring-up returns, in the order bring-up -> offline -> game activity
    -> raised. `train.py`'s fleet path is the same composition inside
    `open_instance`, so this pins the call order per instance rather than only
    the fact that the calls happen.
    """
    monkeypatch.setenv("TOWER_BRIDGE_BUILD_DIR", str(tmp_path))
    expected = SimpleNamespace(profile_id=PROFILE, bridge_version="v1")
    monkeypatch.setattr(train, "compatibility", lambda build_dir: expected)
    monkeypatch.setattr(train, "prepare_pinned_snapshot", lambda renderer, cores: "snap")
    monkeypatch.setattr(
        train,
        "connect",
        lambda serial, port, arguments, expected, opened: train.ActorInstance(
            serial=serial, environment=environment()
        ),
    )
    monkeypatch.setattr(train.tear_down_fleet, "__defaults__", (lambda instance: None,))
    monkeypatch.setattr(
        train,
        "train_session",
        lambda arguments, instances, **kwargs: {"instances": list(instances)},
    )

    calls: list[tuple[str, int]] = []

    def fake_bring_up(
        instance: CloneInstance,
        renderer: str,
        *,
        deploy: Any,
        read_only: bool,
        cores: int,
        frame_rate_hz: int,
    ) -> str:
        assert frame_rate_hz == 90, "the parsed --frame-rate-hz threads into bring-up"
        calls.append(("bring_up", instance.index))
        return "cold"

    def fake_require_offline(instance: CloneInstance) -> None:
        calls.append(("require_offline", instance.index))

    def fake_require_game_activity(instance: CloneInstance) -> None:
        calls.append(("require_game_activity", instance.index))

    def fake_raise_frame_rate(instance: CloneInstance, frame_rate_hz: int) -> None:
        assert frame_rate_hz == 90
        calls.append(("raise_frame_rate", instance.index))

    monkeypatch.setattr(train, "bring_up", fake_bring_up)
    monkeypatch.setattr(train, "require_offline", fake_require_offline)
    monkeypatch.setattr(train, "require_game_activity", fake_require_game_activity)
    monkeypatch.setattr(train, "raise_frame_rate", fake_raise_frame_rate)
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--budget-decisions", "1000", "--actors", "2",
         "--frame-rate-hz", "90", "--no-track"]
    )

    exit_code = train.main()

    assert exit_code == 0
    # Bring-up, offline, game activity, then the raise - per instance, before
    # the next instance's bring-up begins.
    assert calls == [
        ("bring_up", 0),
        ("require_offline", 0),
        ("require_game_activity", 0),
        ("raise_frame_rate", 0),
        ("bring_up", 1),
        ("require_offline", 1),
        ("require_game_activity", 1),
        ("raise_frame_rate", 1),
    ]


def test_a_single_actor_run_raises_no_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's own instance owns its rate; the connect path never raises it.

    `--actors 1` addresses the instance the operator already brought up by
    hand, so nothing on this path may call `require_game_activity` or
    `raise_frame_rate` - those belong to the fleet's own bring-up.
    """
    monkeypatch.setenv("TOWER_BRIDGE_BUILD_DIR", str(tmp_path))
    expected = SimpleNamespace(profile_id=PROFILE, bridge_version="v1")
    monkeypatch.setattr(train, "compatibility", lambda build_dir: expected)
    monkeypatch.setattr(
        train,
        "connect",
        lambda serial, port, arguments, expected, opened: train.ActorInstance(
            serial=serial, environment=environment()
        ),
    )
    monkeypatch.setattr(
        train,
        "train_session",
        lambda arguments, instances, **kwargs: {"instances": list(instances)},
    )

    activity_calls: list[CloneInstance] = []
    raise_calls: list[CloneInstance] = []
    monkeypatch.setattr(
        train, "require_game_activity", lambda instance: activity_calls.append(instance)
    )
    monkeypatch.setattr(
        train, "raise_frame_rate", lambda instance, frame_rate_hz: raise_calls.append(instance)
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--budget-decisions", "1000", "--no-track"])

    exit_code = train.main()

    assert exit_code == 0
    assert activity_calls == []
    assert raise_calls == []


def test_frame_rate_hz_is_bounded(tmp_path: Path) -> None:
    """The same ceiling `run_actors.py` holds its rate to, refused by name."""
    with pytest.raises(SystemExit, match="frame-rate-hz"):
        arguments(tmp_path, **{"--frame-rate-hz": "301"})
    with pytest.raises(SystemExit, match="frame-rate-hz"):
        arguments(tmp_path, **{"--frame-rate-hz": "0"})


def test_resolved_config_carries_the_frame_rate(tmp_path: Path) -> None:
    """A run's identity carries the rate it actually trained at."""
    report = session(tmp_path, settings={"--frame-rate-hz": "90"})
    assert report["arm"]["resolved_config"]["frame_rate_hz"] == 90


# -- resume ----------------------------------------------------------------
#
# A run trained in two sittings is one run: the second segment continues the
# first's budget, its schedules, its checkpoint cadence and its tracked curve,
# and - resumed from the `latest.pt` the first wrote as it ended - the replay
# buffer the first saved beside it. With no saved buffer it re-warms under the
# loaded policy, which is the ordinary warm-up rule applied again.

#: The checkpoint cadence of a resumed run, in decisions.
RESUME_PERIOD = 100


def numbered(run_dir: Path, budget: int, **overrides: str | None) -> dict[str, Any]:
    """One segment of a run, leaving a numbered checkpoint on every crossing."""
    return session(
        run_dir,
        budget=str(budget),
        settings={"--checkpoint-every-decisions": str(RESUME_PERIOD), **overrides},
    )


def latest_checkpoint(report: dict[str, Any]) -> Path:
    """The resume point of a finished segment, which is what a resume is given."""
    return Path(report["run_folder"]) / "checkpoints" / "latest.pt"


def resume_from(
    run_dir: Path, checkpoint: Path, budget: int, profile_id: str = PROFILE, **flags: str
) -> Any:
    """The parsed `--resume` of a second segment, as `main` reads it.

    The profile is the one the bridge reported, which is what the checkpoint's
    identity is checked against before anything is brought up. `flags` are the
    segment's own, a `--run-name` among them.
    """
    return train.resume_point(
        arguments(
            run_dir,
            **{"--budget-decisions": str(budget), "--resume": str(checkpoint), **flags},
        ),
        profile_id=profile_id,
        revision="test",
    )


def resumed_arm(
    run_dir: Path, checkpoint: Path, budget: int, **overrides: str
) -> tuple[Any, Any]:
    """`build_arm` alone, so what a resume restored can be read before a decision.

    Everything under test here is settled at construction - the counters, the
    schedule position the run publishes from them, the optimizer moments - and
    a single collected episode would move all of it.
    """
    settings = arguments(
        run_dir,
        **{
            "--budget-decisions": str(budget),
            "--resume": str(checkpoint),
            "--checkpoint-every-decisions": str(RESUME_PERIOD),
            **overrides,
        },
    )
    resume = train.resume_point(settings, profile_id=PROFILE, revision="test")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        arm, _ = train.build_arm(
            train.BACKBONE,
            settings,
            instances=fleet(1),
            device=torch.device("cpu"),
            profile_id=PROFILE,
            run_dir=settings.run_folder,
            segment=1,
            revision="test",
            started=0.0,
            tracker=NoExperimentTracker(),
            tags={},
            resume=resume,
        )
    return arm, resume


def test_the_exploration_a_run_collected_under_is_on_its_record(
    tmp_path: Path,
) -> None:
    """A curve read months later cannot be told from a uniform one without it."""
    uniform = session(tmp_path / "uniform", actors=2)["arm"]["resolved_config"]
    assert uniform["exploration"] == "uniform"
    assert uniform["exploration_epsilon_floors"] == []

    ladder = session(
        tmp_path / "ladder", actors=2, settings={"--exploration": "ladder"}
    )["arm"]["resolved_config"]
    assert ladder["exploration"] == "ladder"
    assert ladder["exploration_epsilon_floors"] == pytest.approx(list(ape_x_floors(2)))
    # The ladder is the fleet's exploration, not its identity: everything a
    # checkpoint is compatibility-checked on is untouched by it.
    assert ladder["parent_checkpoint"] is None
    assert {key: ladder[key] for key in ("backbone", "actors", "actor_ids")} == {
        key: uniform[key] for key in ("backbone", "actors", "actor_ids")
    }


def test_an_epsilon_end_passed_with_the_ladder_is_refused(tmp_path: Path) -> None:
    """The ladder replaces the end of the anneal, so the flag would go unused."""
    with pytest.raises(SystemExit, match="--epsilon-end"):
        arguments(tmp_path, **{"--exploration": "ladder", "--epsilon-end": "0.001"})
    # The equals form is the same flag and is refused the same way: what is read
    # is the parsed value, not the shape of the argument vector.
    with pytest.raises(SystemExit, match="--epsilon-end"):
        train.parse_arguments(
            ["--budget-decisions", "1000", "--run-dir", str(tmp_path),
             "--exploration", "ladder", "--epsilon-end=0.05"]
        )

    # Either alone is ordinary, and an unset flag resolves to the uniform floor.
    ladder = arguments(tmp_path, **{"--exploration": "ladder"})
    assert (ladder.exploration, ladder.epsilon_end) == ("ladder", 0.05)
    assert arguments(tmp_path, **{"--epsilon-end": "0.001"}).epsilon_end == 0.001


def test_a_run_can_be_resumed_onto_the_ladder(tmp_path: Path) -> None:
    """A second sitting may explore differently from the one it continues.

    Exploration is not part of what a checkpoint is refused for, so a segment
    collected uniformly can be continued under the ladder: what the resume
    restores is the weights and the counters, and what the ladder changes is
    only how the fleet collects from here on.
    """
    first = numbered(tmp_path / "first", 300)
    checkpoint = latest_checkpoint(first)

    arm, resume = resumed_arm(
        tmp_path / "second", checkpoint, budget=600, **{"--exploration": "ladder"}
    )

    exploration = arm.training.config.exploration
    assert exploration.option == "ladder"
    assert exploration.floors == pytest.approx(list(ape_x_floors(1)))
    assert arm.resolved["exploration"] == "ladder"
    # The identity the resume was accepted on is the parent's, unchanged.
    assert resume.parent_checkpoint == arm.resolved["parent_checkpoint"]
    assert arm.training.report.decisions == resume.decisions > 0


def test_a_resume_restores_the_counters_schedules_and_optimizer(tmp_path: Path) -> None:
    """The state a second sitting continues from, before it collects anything."""
    first = numbered(tmp_path / "first", 300)
    checkpoint = latest_checkpoint(first)
    parent = load(checkpoint)
    spent = parent.progress.environment_decisions
    spent_game_ms = parent.progress.environment_game_ms

    arm, resume = resumed_arm(tmp_path / "second", checkpoint, budget=600)

    progress = arm.training.report
    config = arm.training.config
    # The budget position, which the budget, the checkpoint cadence and the
    # exploration schedule are all read from, and the game time beside it.
    assert progress.decisions == spent == first["arm"]["decisions"]
    assert progress.game_ms == spent_game_ms > 0
    assert progress.game_seconds == pytest.approx(first["arm"]["game_seconds"])
    assert progress.episodes == parent.progress.episodes
    assert progress.optimisation_steps == parent.progress.optimisation_steps
    assert not arm.training.finished, "the budget was raised, so there is more to spend"
    # Exploration is derived from the counter rather than restored, so it is
    # where a run that never stopped would have it - and not at the start of
    # its schedule.
    assert (
        progress.epsilon
        == config.exploration.epsilon_for(0, spent)
        != config.exploration.epsilon_for(0, 0)
    )
    # The optimizer's moments come back with the weights: one state dict, and a
    # resume that took only the weights would restart Adam silently mid-run.
    moments = parent.backbone_state["optimizer"]["state"]
    assert moments, "the first segment took optimisation steps"
    restored = arm.backbone.state_dict()["optimizer"]["state"]
    for key, state in moments.items():
        assert torch.equal(state["exp_avg"], restored[key]["exp_avg"])
        assert torch.equal(state["exp_avg_sq"], restored[key]["exp_avg_sq"])
        # Adam's own step count per parameter: it is what the bias correction
        # divides by, so moments restored under a step count of zero would be
        # scaled as if the run had just started.
        assert float(state["step"]) == float(restored[key]["step"]) > 0
    # The parent is named by file and by identity hash, not by path alone.
    assert arm.resolved["parent_checkpoint"] == resume.parent_checkpoint
    assert str(checkpoint) in resume.parent_checkpoint
    assert identity_hash(parent.identity) in resume.parent_checkpoint
    # The episode series is keyed on decisions and carries the budget position
    # beside it, so both continue where the parent left off rather than
    # restarting at zero.
    assert arm.decisions_logged == spent
    assert arm.game_ms_logged == spent_game_ms


def test_a_resumed_run_spends_the_rest_of_the_budget(tmp_path: Path) -> None:
    """The budget is the run's total, and the second segment finishes it."""
    first = numbered(tmp_path / "first", 200)
    spent = first["arm"]["decisions"]

    second = session(
        tmp_path / "second",
        budget="400",
        settings={"--checkpoint-every-decisions": str(RESUME_PERIOD)},
        resume=resume_from(tmp_path / "second", latest_checkpoint(first), 400),
    )

    arm = second["arm"]
    assert arm["decisions"] >= 400, "the run continues to the whole budget"
    assert arm["resolved_config"]["budget_decisions"] == 400
    assert arm["resolved_config"]["parent_checkpoint"] is not None
    # The second segment collected what was left, not the whole budget again.
    collected = sum(int(episode["decisions"]) for episode in arm["collected_episodes"])
    assert collected == arm["decisions"] - spent
    # The numbered cadence carries on rather than restarting: every file this
    # segment wrote is past the resume point, and none of them answers a
    # multiple the first segment already answered.
    _, _, written = numbered_checkpoints(second)
    assert written and min(written) > spent
    assert min(written) // RESUME_PERIOD > spent // RESUME_PERIOD
    # Throughput is this sitting's: the counters above are the whole run's and
    # came back restored, but the wall clock is this segment's alone, so the
    # parent's decisions must not be charged to it.
    assert arm["decisions_per_hour"] == per_hour(arm["decisions"] - spent, arm["wall_seconds"])
    assert arm["episodes_per_hour"] == per_hour(
        arm["episodes"] - first["arm"]["episodes"], arm["wall_seconds"]
    )
    assert arm["decisions_per_hour"] < per_hour(arm["decisions"], arm["wall_seconds"])


def test_a_checkpoint_that_has_already_spent_the_budget_is_refused(tmp_path: Path) -> None:
    """`--budget-decisions` is the run's total, so it must be raised to extend it."""
    first = numbered(tmp_path / "first", 200)
    spent = int(first["arm"]["decisions"])
    checkpoint = latest_checkpoint(first)

    with pytest.raises(SystemExit, match="does not extend"):
        resume_from(tmp_path / "second", checkpoint, spent)

    # One decision more is a run with something left to spend.
    state = resume_from(tmp_path / "second", checkpoint, spent + 1)
    assert state.decisions == spent


def test_a_missing_resume_point_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no checkpoint"):
        resume_from(tmp_path, tmp_path / "absent.pt", 400)


def test_a_checkpoint_from_another_profile_is_refused_before_bring_up(
    tmp_path: Path,
) -> None:
    """Identity is checked, not assumed: the episodes behind a checkpoint from
    another device profile or another schema are not experience this run can go
    on from, and the refusal has to come before an emulator is started."""
    first = numbered(tmp_path / "first", 50)

    with pytest.raises(SystemExit, match="profile_id differs"):
        resume_from(
            tmp_path / "second",
            latest_checkpoint(first),
            100,
            profile_id="some-other-profile-v1",
        )


def legacy_checkpoint(tmp_path: Path, format_version: int) -> tuple[Checkpoint, Path]:
    """A real parent rewritten in an older format under a game-time-era name."""
    first = numbered(tmp_path / "first", 200)
    parent = load(latest_checkpoint(first))
    legacy = tmp_path / "checkpoint-gs0000800.pt"
    save(
        Checkpoint(
            identity=parent.identity,
            progress=replace(
                parent.progress,
                environment_game_ms=parent.progress.environment_game_ms
                if format_version >= 3
                else 0.0,
            ),
            backbone_state=parent.backbone_state,
            resolved_config=parent.resolved_config,
            format_version=format_version,
        ),
        legacy,
    )
    return parent, legacy


@pytest.mark.parametrize("format_version", [1, 2, 3])
def test_a_game_time_era_checkpoint_is_refused_for_resume_by_name(
    tmp_path: Path, format_version: int
) -> None:
    """Its selection-period counters were counted in game time; it evaluates only."""
    _, legacy = legacy_checkpoint(tmp_path, format_version)

    with pytest.raises(SystemExit, match="game-time budget era") as refusal:
        resume_from(tmp_path / "second", legacy, 400)

    assert f"format {format_version} checkpoint" in str(refusal.value)
    assert "evaluation only" in str(refusal.value)


@pytest.mark.parametrize("format_version", [4, 5])
def test_a_checkpoint_from_the_actor_thread_learner_is_refused_for_resume(
    tmp_path: Path, format_version: int
) -> None:
    """Continuing it on the learner thread would make one run of two learners (ADR 0017)."""
    _, legacy = legacy_checkpoint(tmp_path, format_version)

    with pytest.raises(SystemExit, match="actor-thread learner") as refusal:
        resume_from(tmp_path / "second", legacy, 400)

    assert f"format {format_version} checkpoint" in str(refusal.value)
    assert "mixed run" in str(refusal.value)


@pytest.mark.parametrize("format_version", [3, 5])
def test_an_older_checkpoint_still_loads_for_evaluation(
    tmp_path: Path, format_version: int
) -> None:
    """The refusal is the resume's alone: the file rebuilds into a policy."""
    parent, legacy = legacy_checkpoint(tmp_path, format_version)

    assert load(legacy).format_version == format_version
    _, identity = checkpoint_policy(
        legacy,
        decision_cadence=parent.identity.decision_cadence.value,
        upgrade_availability=parent.identity.upgrade_availability.value,
        workshop_level=0,
    )

    assert identity == parent.identity, "rebuilt from the file's own identity"


def test_a_checkpoint_without_optimizer_state_is_a_truncated_file(
    tmp_path: Path,
) -> None:
    """Every checkpoint this project has written carries the optimizer.

    One that does not is truncated, not old, so it raises on the missing key
    rather than resuming on fresh moments - which would change how the next
    steps are taken without saying so.
    """
    first = numbered(tmp_path / "first", 50)
    parent = load(latest_checkpoint(first))
    truncated = tmp_path / "truncated.pt"
    save(
        Checkpoint(
            identity=parent.identity,
            progress=parent.progress,
            backbone_state={
                key: value
                for key, value in parent.backbone_state.items()
                if key != "optimizer"
            },
        ),
        truncated,
    )

    with pytest.raises(KeyError, match="optimizer"):
        resumed_arm(tmp_path / "second", truncated, budget=400)


def test_a_tracked_run_is_continued_rather_than_started_again(tmp_path: Path) -> None:
    """One curve, one run id: the second segment logs onto the first's series."""
    tracker = RecordingTracker()
    first = session(
        tmp_path / "first",
        budget="200",
        tracker=tracker,
        settings={"--checkpoint-every-decisions": str(RESUME_PERIOD)},
    )
    spent = first["arm"]["decisions"]
    checkpoint = latest_checkpoint(first)
    assert load(checkpoint).tracking_run_id == tracker.runs[0].run_id
    logged_first = len(tracker.runs[0].points)

    second = session(
        tmp_path / "second",
        budget="400",
        tracker=tracker,
        settings={"--checkpoint-every-decisions": str(RESUME_PERIOD)},
        resume=resume_from(tmp_path / "second", checkpoint, 400),
    )

    assert len(tracker.runs) == 1, "no second run was opened beside the first"
    assert second["arm"]["run_id"] != first["arm"]["run_id"], "its artefacts are its own"
    run = tracker.runs[0]
    assert "open_run" in run.calls
    # The episode series is one series on one axis: every point the second
    # segment placed is past the budget position the first ended at, and the
    # first of them is exactly that position plus the episode that produced it.
    episodes = [
        point for point in run.points[logged_first:] if "episode_final_wave" in point.metrics
    ]
    assert episodes, "the second segment collected episodes"
    assert episodes[0].decisions == spent + int(episodes[0].metrics["episode_decisions"])
    assert [point.decisions for point in episodes] == sorted(
        point.decisions for point in episodes
    )
    # Where the segment picked up, and where learning restarted on the buffer
    # it reloaded, both on the same axis.
    resumed = [point for point in run.points if "resumed_from_decisions" in point.metrics]
    assert [point.decisions for point in resumed] == [spent]
    warmed = [point for point in run.points if "warmup_finished_decisions" in point.metrics]
    assert len(warmed) == 1 and warmed[0].decisions > spent


def test_an_untracked_resume_does_not_announce_a_tracked_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--no-track` records nothing, so there is no parent series to continue."""
    tracker = RecordingTracker()
    first = session(tmp_path / "first", budget="200", tracker=tracker)
    checkpoint = latest_checkpoint(first)
    assert load(checkpoint).tracking_run_id == tracker.runs[0].run_id

    monkeypatch.setenv("TOWER_BRIDGE_BUILD_DIR", str(tmp_path))
    monkeypatch.setattr(
        train,
        "compatibility",
        lambda build_dir: SimpleNamespace(profile_id=PROFILE, bridge_version="v1"),
    )
    monkeypatch.setattr(
        train,
        "connect",
        lambda serial, port, arguments, expected, opened: train.ActorInstance(
            serial=serial, environment=environment()
        ),
    )
    monkeypatch.setattr(train, "train_session", lambda arguments, instances, **kwargs: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--no-track",
            "--resume",
            str(checkpoint),
            "--budget-decisions",
            "100000",
            # The parent's, so the replay it saved is reloaded rather than refused
            # and the resume is not refused for a changed loop setting.
            "--replay-capacity",
            "64",
            "--gradient-steps-per-decision",
            "0.2",
            "--run-dir",
            str(tmp_path / "second"),
        ],
    )

    assert train.main() == 0
    assert "resuming tracked run" not in capsys.readouterr().out


def test_a_run_split_in_two_covers_the_budget_the_whole_run_does(tmp_path: Path) -> None:
    """The end-to-end claim: 300 in one sitting, or 150 and 150, is one run.

    The two are not decision-for-decision identical and cannot be. Episode
    length here depends on what the policy does, and the second segment samples
    the buffer it reloaded from its own seed, so its episodes are not the
    episodes a single sitting would have played and its counter lands past the
    period's multiples in different places. What is
    the same is what the budget bought: the whole budget spent, and one
    numbered checkpoint per crossing of the period, in order, continuing
    through the resume rather than restarting at it.
    """

    def crossings(report: dict[str, Any]) -> list[int]:
        """Which multiples of the period this segment's files answered."""
        _, _, written = numbered_checkpoints(report)
        return [decisions // RESUME_PERIOD for decisions in written]

    whole = numbered(tmp_path / "whole", 300)
    once = crossings(whole)

    first = numbered(tmp_path / "first", 150)
    second = session(
        tmp_path / "second",
        budget="300",
        settings={"--checkpoint-every-decisions": str(RESUME_PERIOD)},
        resume=resume_from(tmp_path / "second", latest_checkpoint(first), 300),
    )
    split = crossings(first) + crossings(second)

    assert whole["arm"]["decisions"] >= 300, "one sitting spends the budget"
    assert second["arm"]["decisions"] >= 300, "two sittings spend the same budget"
    # The collection curve is one series across the two sittings: every window
    # is keyed by a position on the whole run's budget, so none of the second
    # segment's points falls at or below where the first segment stopped.
    resumed_at = first["arm"]["decisions"]
    early_windows = [window["decisions_at_end"] for window in first["arm"]["collection_curve"]]
    late_windows = [window["decisions_at_end"] for window in second["arm"]["collection_curve"]]
    assert early_windows and late_windows, "both sittings closed a window"
    assert min(late_windows) > resumed_at
    windows = early_windows + late_windows
    assert windows == sorted(windows) and len(set(windows)) == len(windows)
    # Both arrangements answer each crossing once, in order, and the split run
    # answers none of them twice across its two segments - the cadence went on
    # through the resume instead of starting again from it.
    for answered in (once, split):
        assert answered == sorted(set(answered)) and answered
    assert min(crossings(second)) > max(crossings(first))


# -- replay saved with every latest.pt, reloaded on resume (board #97) --------


def saved_replay(run_dir: Path) -> Path:
    """The dump the one run directory's `latest.pt` under `run_dir` names.

    Found as a resume finds it, and the only one kept: every other is deleted
    once that `latest.pt` is in place.
    """
    (latest,) = run_dir.glob("*/checkpoints/latest.pt")
    folder = latest.parent.parent
    paired = load(latest).paired_replay
    assert paired is not None, "every latest.pt is written with its replay"
    dump = folder / paired
    assert list((folder / REPLAY_DIRECTORY).iterdir()) == [dump]
    return dump


def paired_decisions(dump: Path) -> tuple[int, int]:
    """The decision counts a dump and the `latest.pt` naming it each record."""
    checkpoint = load(dump.parent.parent / "checkpoints" / "latest.pt")
    return (
        read_replay_metadata(dump)["run"]["decisions"],
        checkpoint.progress.environment_decisions,
    )


def test_a_run_saves_its_replay_beside_the_latest_checkpoint_it_ends_with(
    tmp_path: Path,
) -> None:
    first = numbered(tmp_path, 200)
    dump = saved_replay(tmp_path)

    assert dump.parent.parent == latest_checkpoint(first).parent.parent
    metadata = read_replay_metadata(dump)
    checkpoint = load(latest_checkpoint(first))
    assert metadata["run"]["decisions"] == checkpoint.progress.environment_decisions
    assert metadata["run"]["identity"] == asdict(checkpoint.identity)
    assert metadata["sequences"] == first["arm"]["replay"]["sequences"] > 0
    assert (metadata["capacity"], metadata["alpha"]) == (64, R2D2_PRIORITY_EXPONENT)


def test_every_latest_checkpoint_a_run_writes_is_written_with_its_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not only the last: a kill between two of them leaves a matching pair."""
    pairs: list[tuple[int, int]] = []
    write = train.TrainingReport._write_resume_point

    def recorded(self: Any, report: Any, image: ReplayImage) -> None:
        write(self, report, image)
        pairs.append(paired_decisions(saved_replay(self.run_dir.parent)))

    monkeypatch.setattr(train.TrainingReport, "_write_resume_point", recorded)
    first = numbered(tmp_path, 300)

    # Every other episode, every numbered checkpoint, and the run's end.
    assert len(pairs) >= len(numbered_checkpoints(first)[2]) + 1
    assert all(dump == checkpoint for dump, checkpoint in pairs)
    assert pairs[-1][1] == first["arm"]["decisions"]


def test_a_resume_from_that_checkpoint_reloads_the_replay_and_says_so(tmp_path: Path) -> None:
    first = numbered(tmp_path / "first", 200)
    dump = saved_replay(tmp_path / "first")
    expected = PrioritizedSequenceReplay(capacity=64)
    expected.load_from(dump)

    arm, resume = resumed_arm(tmp_path / "second", latest_checkpoint(first), budget=400)

    assert resume.replay_dump == dump
    assert list(arm.replay._items) == list(expected._items)
    assert list(arm.replay._priorities) == list(expected._priorities)
    # The sampler goes on from where it stood, not from the seed.
    assert arm.replay._random.getstate() == expected._random.getstate()
    assert arm.resolved["replay_restored_from"] == str(dump)
    manifest = json.loads((arm.run_dir / "manifest.json").read_text())
    assert manifest["replay_restored_from"] == str(dump)
    # Warm already: the gate reads what the buffer holds, not how it got it.
    assert arm.training._warm()


def test_a_resume_puts_back_the_random_streams_the_checkpoint_was_written_with(
    tmp_path: Path,
) -> None:
    first = numbered(tmp_path / "first", 200)
    saved = load(latest_checkpoint(first)).rng_state
    assert saved is not None

    resumed_arm(tmp_path / "second", latest_checkpoint(first), budget=400)
    drawn = (random.random(), numpy.random.random(), torch.rand(3))
    restore_rng_state(saved)
    expected = (random.random(), numpy.random.random(), torch.rand(3))

    assert drawn[:2] == expected[:2]
    assert torch.equal(drawn[2], expected[2])


def test_a_resume_with_no_saved_replay_re_warms_as_before(tmp_path: Path) -> None:
    first = numbered(tmp_path / "first", 200)
    # Moved aside whole, as an operator re-warms on purpose.
    shutil.rmtree(saved_replay(tmp_path / "first").parent)

    arm, resume = resumed_arm(tmp_path / "second", latest_checkpoint(first), budget=400)

    assert resume.replay_dump is None
    assert len(arm.replay) == 0
    assert arm.resolved["replay_restored_from"] is None


def test_a_latest_checkpoint_whose_replay_is_gone_is_refused(tmp_path: Path) -> None:
    """`replay/` there without the dump `latest.pt` names is a broken pair, not a re-warm."""
    first = numbered(tmp_path / "first", 200)
    shutil.rmtree(saved_replay(tmp_path / "first"))

    with pytest.raises(SystemExit, match="the resume point is broken"):
        resume_from(tmp_path / "second", latest_checkpoint(first), 400)


def test_a_latest_checkpoint_and_a_replay_at_another_count_are_refused(tmp_path: Path) -> None:
    first = numbered(tmp_path / "first", 200)
    dump = saved_replay(tmp_path / "first")
    metadata = read_replay_metadata(dump)
    metadata["run"]["decisions"] -= 1
    (dump / REPLAY_DUMP_METADATA).write_text(json.dumps(metadata))

    with pytest.raises(SystemExit, match="mismatched pair"):
        resume_from(tmp_path / "second", latest_checkpoint(first), 400)


def test_a_numbered_checkpoint_reloads_the_replay_saved_with_it(tmp_path: Path) -> None:
    """The pair written at a numbered checkpoint is as good a resume point until replaced."""
    first = numbered(tmp_path / "first", 200)
    dump = saved_replay(tmp_path / "first")
    decisions = read_replay_metadata(dump)["run"]["decisions"]
    _, directory, _ = numbered_checkpoints(first)
    # The run's last latest.pt is its end, past any numbered one; standing in for
    # a run killed straight after one, the numbered file at that count is its own.
    at_the_dump = directory / numbered_checkpoint_name(decisions)
    save(replace(load(latest_checkpoint(first)), paired_replay=None), at_the_dump)

    assert resume_from(tmp_path / "second", at_the_dump, 400).replay_dump == dump


def test_a_resume_from_an_earlier_checkpoint_is_refused_while_the_replay_is_there(
    tmp_path: Path,
) -> None:
    """Replay from the run's end must not be mixed into a resume from its middle."""
    first = numbered(tmp_path / "first", 300)
    _, directory, written = numbered_checkpoints(first)
    earlier = directory / numbered_checkpoint_name(written[0])

    with pytest.raises(SystemExit, match="decisions, not this checkpoint's"):
        resume_from(tmp_path / "second", earlier, 600)

    # Moved aside by hand, it is the ordinary resume with an empty buffer.
    replays = saved_replay(tmp_path / "first").parent
    replays.rename(replays.with_name("replay-set-aside"))
    assert resume_from(tmp_path / "second", earlier, 600).replay_dump is None


def test_a_saved_replay_of_another_capacity_is_refused_before_bring_up(tmp_path: Path) -> None:
    first = numbered(tmp_path / "first", 200)
    with pytest.raises(SystemExit, match="capacity 64"):
        train.resume_point(
            arguments(
                tmp_path / "second",
                **{
                    "--budget-decisions": "400",
                    "--resume": str(latest_checkpoint(first)),
                    "--replay-capacity": "128",
                },
            ),
            profile_id=PROFILE,
            revision="test",
        )


def test_a_resumed_run_learns_from_the_reloaded_replay_without_re_warming(
    tmp_path: Path,
) -> None:
    """End to end: a warm-up the new segment alone could not reach is already met."""
    first = numbered(tmp_path / "first", 200)
    parent_steps = first["arm"]["optimisation_steps"]
    budget = first["arm"]["decisions"] + 60
    # More sequences than 60 decisions can collect, but fewer than were saved.
    settings = {"--warmup-sequences": "50"}

    second = session(
        tmp_path / "second",
        budget=str(budget),
        settings=settings,
        resume=resume_from(tmp_path / "second", latest_checkpoint(first), budget),
    )
    assert second["arm"]["optimisation_steps"] > parent_steps

    shutil.rmtree(saved_replay(tmp_path / "first").parent)
    third = session(
        tmp_path / "third",
        budget=str(budget),
        settings=settings,
        resume=resume_from(tmp_path / "third", latest_checkpoint(first), budget),
    )
    assert third["arm"]["optimisation_steps"] == parent_steps, "re-warming, it could not learn"


def interrupted_session(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException | None = None,
) -> None:
    """A session whose run fails partway, after collecting some of its budget."""

    def run_then_fail(self: TrainingRun) -> Any:
        self.advance(100)
        raise failure or RuntimeError("the run failed")

    monkeypatch.setattr(TrainingRun, "run", run_then_fail)
    session(run_dir, budget="400")


@pytest.mark.parametrize(
    "failure", [RuntimeError("the run failed"), KeyboardInterrupt()], ids=["error", "interrupt"]
)
def test_a_run_that_fails_still_writes_its_resume_point_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    with pytest.raises(type(failure)):
        interrupted_session(tmp_path, monkeypatch, failure)

    dump_decisions, decisions = paired_decisions(saved_replay(tmp_path))
    assert 100 <= decisions < 400
    assert dump_decisions == decisions


def test_an_interrupt_while_the_fleet_collects_leaves_a_matching_resume_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C reaches the main thread while the actor threads are mid-collection.

    They are not joined by the interrupt, so once the resume point is written
    they must stop rather than go on counting and checkpointing over it.
    """
    collecting: list[threading.Thread] = []

    def interrupted(self: TrainingRun) -> Any:
        # The fleet runs off the main thread, as the pool's actors do when the
        # interrupt leaves the main thread's join.
        # A daemon, so a regression that never halts cannot hang the suite.
        fleet = threading.Thread(
            target=self.advance, args=(self.config.budget_decisions,), daemon=True
        )
        collecting.append(fleet)
        fleet.start()
        deadline = time.monotonic() + 60
        while (
            self.report.decisions < 100
            and fleet.is_alive()
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert fleet.is_alive(), "the fleet stopped before it was interrupted"
        assert self.report.decisions >= 100, "the fleet collected too slowly to interrupt"
        raise KeyboardInterrupt

    monkeypatch.setattr(TrainingRun, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        # A budget far past what the test waits for, so only the halt stops it;
        # two actors, and a checkpoint every other episode to overwrite with.
        session(tmp_path, budget="1000000", actors=2)
    (fleet,) = collecting
    fleet.join(timeout=60)
    assert not fleet.is_alive(), "the actors stopped at the halt"

    dump_decisions, decisions = paired_decisions(saved_replay(tmp_path))
    assert dump_decisions == decisions


def test_a_run_stopped_mid_collection_resumes_from_the_steps_its_weights_took(
    tmp_path: Path,
) -> None:
    """The learner thread is quiesced at the stop's resume point (ADR 0014, 0017).

    The run's stop is set from outside while the fleet collects and the learner
    steps beside it. The resume point it leaves counts exactly the steps its
    weights took, and a second segment picks both up and learns on.
    """
    advance = TrainingRun.advance

    def stopped_part_way(self: TrainingRun) -> Any:
        def stop_after_some_learning() -> None:
            deadline = time.monotonic() + 60
            while self.report.decisions < 150 and time.monotonic() < deadline:
                time.sleep(0.002)
            self.stop.set()

        threading.Thread(target=stop_after_some_learning, daemon=True).start()
        return advance(self, self.config.budget_decisions)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(TrainingRun, "run", stopped_part_way)
        first = session(tmp_path / "first", budget="1000000", actors=2)
    assert first["arm"]["interrupted"] is True
    parent = load(latest_checkpoint(first))
    steps = parent.progress.optimisation_steps
    assert steps == first["arm"]["optimisation_steps"] > 0

    arm, _ = resumed_arm(tmp_path / "second", latest_checkpoint(first), budget=1_000_000)
    assert arm.backbone.model_version == steps, "the weights took the steps counted"

    budget = parent.progress.environment_decisions + 100
    second = session(
        tmp_path / "third",
        budget=str(budget),
        resume=resume_from(tmp_path / "third", latest_checkpoint(first), budget),
    )
    assert second["arm"]["optimisation_steps"] > steps


def test_a_resume_pays_the_debt_its_parent_still_owed(tmp_path: Path) -> None:
    """The steps a resumed run takes are the ones an uninterrupted run's rule gives it.

    A parent that ended owing a part of a step (the ratio times its decisions,
    less the whole steps taken) writes that debt into its checkpoint (ADR 0017).
    The resume carries it, so over the two segments the steps stay the ratio of
    the decisions and the fraction is not dropped at the seam.
    """
    ratio = 0.2
    # Ends at 103 decisions, 73 of them counted warm: 14.6 steps, 0.6 of one owed.
    first = numbered(tmp_path / "first", 100)
    checkpoint = latest_checkpoint(first)
    parent = load(checkpoint)
    owed = parent.progress.learner_debt_steps
    assert 0 < owed < 1, "the parent ended part-way through a step"

    arm, resume = resumed_arm(tmp_path / "second", checkpoint, budget=400)
    assert resume.learner_debt_steps == owed
    assert arm.training.learner_thread.counted_debt_steps() == pytest.approx(owed)

    second = session(
        tmp_path / "third",
        budget="400",
        resume=resume_from(tmp_path / "third", checkpoint, 400),
    )

    spent = second["arm"]["decisions"] - parent.progress.environment_decisions
    taken = second["arm"]["optimisation_steps"] - parent.progress.optimisation_steps
    earned = owed + ratio * spent
    assert taken <= earned + 1e-9 < taken + 1, "the segment ended owing under one step"
    assert taken != int(ratio * spent), "without the parent's fraction it would take one fewer"


def test_a_resume_that_rewarms_its_buffer_owes_nothing(tmp_path: Path) -> None:
    """A debt owed on the parent's buffer is not carried onto an empty one (ADR 0017)."""
    first = numbered(tmp_path / "first", 100)
    checkpoint = latest_checkpoint(first)
    assert load(checkpoint).progress.learner_debt_steps > 0
    shutil.rmtree(checkpoint.parent.parent / "replay")

    arm, _ = resumed_arm(tmp_path / "second", checkpoint, budget=400)

    assert arm.training.learner_thread.counted_debt_steps() == 0


def test_a_run_that_fails_with_non_finite_weights_leaves_the_periodic_resume_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure mid-update must not overwrite the last good pair with a broken one."""

    def run_then_break(self: TrainingRun) -> Any:
        self.advance(100)
        # In place, as a diverged update would leave it.
        next(iter(self.backbone.state_dict()["online"].values())).fill_(float("nan"))
        raise RuntimeError("the run failed")

    monkeypatch.setattr(TrainingRun, "run", run_then_break)
    with pytest.raises(RuntimeError, match="the run failed"):
        session(tmp_path, budget="400")

    (checkpoint,) = tmp_path.glob("*/checkpoints/latest.pt")
    assert not non_finite_tensors(dict(load(checkpoint).backbone_state))
    # The periodic pair, still whole.
    dump_decisions, decisions = paired_decisions(saved_replay(tmp_path))
    assert dump_decisions == decisions


def refuse_to_write(self: ReplayImage, directory: Path, **kwargs: Any) -> int:
    raise OSError("disk full")


def test_a_failed_replay_save_neither_masks_the_error_nor_moves_the_resume_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `latest.pt` without its replay would be a resume point that loses the buffer."""
    monkeypatch.setattr(ReplayImage, "write", refuse_to_write)
    with pytest.raises(RuntimeError, match="the run failed"):
        interrupted_session(tmp_path / "failed", monkeypatch)
    assert not list((tmp_path / "failed").glob("*/checkpoints/latest.pt"))

    # A run that spends its budget is not failed by it either.
    monkeypatch.undo()
    monkeypatch.setattr(ReplayImage, "write", refuse_to_write)
    report = session(tmp_path / "finished", budget="200")
    assert report["arm"]["decisions"] >= 200
    assert not list((tmp_path / "finished").glob(f"*/{REPLAY_DIRECTORY}/*"))


# -- discounting by game time (board #81) ------------------------------------


def test_the_game_time_discount_is_off_by_default(trained: dict[str, Any]) -> None:
    """Off, the run discounts per decision exactly as every run before it."""
    resolved = trained["arm"]["resolved_config"]
    assert resolved["discount_per_game_second"] is None
    assert resolved["discount"] == 0.99


def test_the_two_discounts_are_refused_together(tmp_path: Path) -> None:
    """T8: each defines the discount, so one of them would go silently unused."""
    with pytest.raises(SystemExit, match="one or the other"):
        arguments(tmp_path, **{"--discount": "0.99", "--discount-per-game-second": "0.997"})


def test_dreamerv3_takes_the_game_time_discount_and_refuses_to_run_without_it(
    tmp_path: Path,
) -> None:
    """ADR 0013: the discount is the task's, and DreamerV3 has no per-step one of its own."""
    assert dreamer_arguments(tmp_path).discount_per_game_second == 0.999
    with pytest.raises(SystemExit, match="needs --discount-per-game-second"):
        train.parse_arguments(
            ["--budget-decisions", "1000", "--run-dir", str(tmp_path), "--backbone", "dreamerv3"]
        )


def test_a_game_time_run_records_its_discount_and_its_checkpoint_plays(tmp_path: Path) -> None:
    """T8: the flag is in the run's identity, and the checkpoint rebuilds with it."""
    report = session(tmp_path, settings={"--discount-per-game-second": "0.997"})
    resolved = report["arm"]["resolved_config"]
    assert resolved["discount_per_game_second"] == 0.997
    assert resolved["discount"] is None, "the per-decision discount played no part"

    policy, _ = checkpoint_policy(
        latest_checkpoint(report),
        decision_cadence=resolved["decision_cadence"],
        upgrade_availability=resolved["upgrade_availability"],
        workshop_level=0,
    )
    assert isinstance(policy, StackedDqnBackbone)
    assert policy.config.discount_per_game_second == 0.997

    # It resumes under the same discount, and only under it.
    resumed = train.resume_point(
        arguments(
            tmp_path / "second",
            **{
                "--budget-decisions": "400",
                "--resume": str(latest_checkpoint(report)),
                "--discount-per-game-second": "0.997",
            },
        ),
        profile_id=PROFILE,
        revision="test",
    )
    assert resumed is not None and resumed.decisions > 0
    with pytest.raises(SystemExit, match="a different target"):
        resume_from(tmp_path / "third", latest_checkpoint(report), 400)


def test_a_checkpoint_from_before_the_game_time_discount_still_plays(tmp_path: Path) -> None:
    """T8: a resolved config without the key rebuilds, discounting per decision."""
    parent_path = latest_checkpoint(numbered(tmp_path / "first", 50))
    parent = load(parent_path)
    older = tmp_path / "older.pt"
    settings = dict(parent.resolved_config)
    del settings["discount_per_game_second"]
    save(replace(parent, resolved_config=settings), older)

    policy, _ = checkpoint_policy(
        older,
        decision_cadence=parent.identity.decision_cadence.value,
        upgrade_availability=parent.identity.upgrade_availability.value,
        workshop_level=0,
    )
    assert isinstance(policy, StackedDqnBackbone)
    assert policy.config.discount_per_game_second is None
    assert policy.config.discount == settings["discount"]
    # And it resumes under the per-decision default it was trained with.
    assert resume_from(tmp_path / "second", older, 400).decisions > 0


@pytest.mark.parametrize(
    "flags",
    [
        {"--discount-per-game-second": "0.997"},
        {"--discount": "0.9"},
    ],
)
def test_a_resume_under_another_discount_is_refused(
    tmp_path: Path, flags: dict[str, str]
) -> None:
    """A different discount is a different target: one set of weights, two scales."""
    checkpoint = latest_checkpoint(numbered(tmp_path / "first", 50))

    with pytest.raises(SystemExit, match="a different target"):
        train.resume_point(
            arguments(
                tmp_path / "second",
                **{"--budget-decisions": "400", "--resume": str(checkpoint), **flags},
            ),
            profile_id=PROFILE,
            revision="test",
        )


# -- the survival-time reward (board #82) ------------------------------------

SURVIVAL: dict[str, str | None] = {
    "--discount-per-game-second": "0.997",
    "--survival-time-reward": None,
}


def test_the_survival_time_reward_is_off_by_default(trained: dict[str, Any]) -> None:
    """Off, the run learns from the wave reward exactly as every run before it."""
    assert trained["arm"]["resolved_config"]["survival_time_reward"] is False


def test_the_survival_time_reward_is_refused_without_the_game_time_discount(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="needs --discount-per-game-second"):
        arguments(tmp_path, **{"--survival-time-reward": None})


def test_dreamerv3_takes_the_survival_time_reward(tmp_path: Path) -> None:
    """The reward is the task's (ADR 0013): the flag means what it means for stacked-dqn."""
    assert dreamer_arguments(tmp_path, "--survival-time-reward").survival_time_reward


def dreamer_resume(run_dir: Path, checkpoint: Path, *flags: str) -> Any:
    """A second `dreamer_session` segment's `--resume` of this checkpoint, under these flags."""
    from tower_rl.learning.dreamer import DreamerConfig

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            train, "DreamerConfig", lambda **given: DreamerConfig(**{**SMALL_DREAMER, **given})
        )
        patch.setattr(train, "DREAMER_REPLAY_CAPACITY", 64)
        return train.resume_point(
            dreamer_arguments(
                run_dir, "--budget-decisions", "400",
                "--resume", str(checkpoint), *flags,
            ),
            profile_id=PROFILE,
            revision="test",
        )


def test_a_dreamerv3_run_writes_and_reloads_the_same_resume_pair(tmp_path: Path) -> None:
    """Its step replay - latents, final observations, the queue - is saved with every latest.pt."""
    checkpoint = latest_checkpoint(dreamer_session(tmp_path / "first"))
    dump = saved_replay(tmp_path / "first")

    dump_decisions, decisions = paired_decisions(dump)
    assert dump_decisions == decisions
    assert dreamer_resume(tmp_path / "second", checkpoint).replay_dump == dump
    from tower_rl.learning.dreamer_replay import DreamerReplay

    restored = DreamerReplay(capacity=64, length=6 + 1)
    restored.load_from(dump)
    assert len(restored) > 0 and restored.steps_held > len(restored)


def test_a_dreamerv3_checkpoint_from_before_the_step_replay_plays_but_does_not_resume(
    tmp_path: Path,
) -> None:
    """Format 6: zero-start windows and no stored latents. Resuming it would mix two replays."""
    checkpoint = latest_checkpoint(dreamer_session(tmp_path / "first"))
    parent = load(checkpoint)
    settings = {**parent.resolved_config, "dreamer_actor_unimix": 0.01}
    older = tmp_path / "older.pt"
    save(replace(parent, format_version=6, resolved_config=settings), older)
    with pytest.raises(SystemExit, match="mixed run"):
        dreamer_resume(tmp_path / "second", older)
    # It still plays, with the actor unimix it was trained with.
    policy, _ = checkpoint_policy(
        older,
        decision_cadence=settings["decision_cadence"],
        upgrade_availability=settings["upgrade_availability"],
        workshop_level=0,
    )
    assert policy.config.actor_unimix == 0.01


def test_a_dreamerv3_run_resumes_only_under_its_own_discount_and_reward(
    tmp_path: Path,
) -> None:
    checkpoint = latest_checkpoint(dreamer_session(tmp_path / "first"))

    assert dreamer_resume(tmp_path / "second", checkpoint).decisions > 0
    with pytest.raises(SystemExit, match="a different target"):
        dreamer_resume(tmp_path / "third", checkpoint, "--discount-per-game-second", "0.997")
    with pytest.raises(SystemExit, match="a different target"):
        dreamer_resume(tmp_path / "fourth", checkpoint, "--survival-time-reward")

    # A checkpoint from before the game-time discount: a per-step horizon.
    parent = load(checkpoint)
    settings = {
        key: value
        for key, value in parent.resolved_config.items()
        if key not in ("dreamer_discount_per_game_second", "dreamer_survival_time_reward")
    }
    settings.update(
        dreamer_horizon=333,
        discount_per_game_second=None,
        survival_time_reward=None,
        survival_reward_bound=None,
    )
    older = tmp_path / "older.pt"
    save(replace(parent, resolved_config=settings), older)
    with pytest.raises(SystemExit, match="per-step horizon 333"):
        dreamer_resume(tmp_path / "fifth", older)
    # It still plays: acting reads no discount.
    policy, _ = checkpoint_policy(
        older,
        decision_cadence=settings["decision_cadence"],
        upgrade_availability=settings["upgrade_availability"],
        workshop_level=0,
    )
    assert policy.config.discount_per_game_second is None


def survival_resume(run_dir: Path, checkpoint: Path, **flags: str | None) -> Any:
    """A second segment's `--resume` of this checkpoint, under these flags."""
    return train.resume_point(
        arguments(
            run_dir, **{"--budget-decisions": "400", "--resume": str(checkpoint), **flags}
        ),
        profile_id=PROFILE,
        revision="test",
    )


def test_a_survival_time_run_records_its_reward_and_resumes_only_under_it(
    tmp_path: Path,
) -> None:
    """The key is in the run's identity, and a resume must ask for the same reward."""
    report = session(tmp_path, settings=SURVIVAL)
    assert report["arm"]["resolved_config"]["survival_time_reward"] is True
    assert report["arm"]["resolved_config"]["survival_reward_bound"] == V_REF
    checkpoint = latest_checkpoint(report)

    resumed = survival_resume(tmp_path / "second", checkpoint, **SURVIVAL)
    assert resumed is not None and resumed.decisions > 0
    with pytest.raises(SystemExit, match="a different target"):
        survival_resume(
            tmp_path / "third", checkpoint, **{"--discount-per-game-second": "0.997"}
        )


def test_a_wave_reward_checkpoint_is_not_resumed_under_the_survival_time_reward(
    tmp_path: Path,
) -> None:
    """The reward defines the target as the discount does: one set of weights, two scales."""
    checkpoint = latest_checkpoint(
        numbered(tmp_path / "first", 50, **{"--discount-per-game-second": "0.997"})
    )

    with pytest.raises(SystemExit, match="a different target"):
        survival_resume(tmp_path / "second", checkpoint, **SURVIVAL)


def test_a_checkpoint_from_before_the_survival_time_reward_resumes_with_it_off(
    tmp_path: Path,
) -> None:
    """A resolved config without the key reads as the wave reward it learned from."""
    parent = load(latest_checkpoint(numbered(tmp_path / "first", 50)))
    older = tmp_path / "older.pt"
    settings = dict(parent.resolved_config)
    del settings["survival_time_reward"]
    save(replace(parent, resolved_config=settings), older)

    assert resume_from(tmp_path / "second", older, 400).decisions > 0


def test_a_scaled_survival_run_at_0_999_resumes_under_the_same_flags(tmp_path: Path) -> None:
    """M3-P015's own resume path: a scaled 0.999 checkpoint continues at 0.999."""
    flags = {**SURVIVAL, "--discount-per-game-second": "0.999"}
    report = numbered(tmp_path / "first", 50, **flags)
    resolved = report["arm"]["resolved_config"]
    assert resolved["discount_per_game_second"] == 0.999
    assert resolved["survival_reward_bound"] == V_REF

    resumed = survival_resume(tmp_path / "second", latest_checkpoint(report), **flags)
    assert resumed is not None and resumed.decisions > 0


def _before_the_reward_bound(tmp_path: Path, per_second: str) -> Path:
    """A survival-time checkpoint as a run before ADR 0013 wrote it: no bound key."""
    flags = {**SURVIVAL, "--discount-per-game-second": per_second}
    parent = load(latest_checkpoint(numbered(tmp_path / "first", 50, **flags)))
    older = tmp_path / "older.pt"
    settings = dict(parent.resolved_config)
    del settings["survival_reward_bound"]
    save(replace(parent, resolved_config=settings), older)
    return older


def test_an_unscaled_survival_checkpoint_at_0_997_resumes_under_the_scaled_reward(
    tmp_path: Path,
) -> None:
    """At 0.997 the scaled reward is the unscaled one: nothing is mixed."""
    older = _before_the_reward_bound(tmp_path, "0.997")

    assert survival_resume(tmp_path / "second", older, **SURVIVAL).decisions > 0


def test_an_unscaled_survival_checkpoint_at_0_999_is_not_resumed_under_the_scaled_reward(
    tmp_path: Path,
) -> None:
    """Its value scale was 28.6, the scaled reward's is V_REF: two scales, one network."""
    older = _before_the_reward_bound(tmp_path, "0.999")
    flags = {**SURVIVAL, "--discount-per-game-second": "0.999"}

    with pytest.raises(SystemExit, match="a different target"):
        survival_resume(tmp_path / "second", older, **flags)


# -- ez-greedy (board #83) ---------------------------------------------------


def test_ez_greedy_is_off_by_default(trained: dict[str, Any]) -> None:
    """Off, the run explores one decision at a time as every run before it."""
    assert trained["arm"]["resolved_config"]["ez_greedy"] is False


def test_ez_greedy_is_refused_under_dreamerv3(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="stacked-dqn setting"):
        dreamer_arguments(tmp_path, "--ez-greedy")


def test_an_ez_greedy_run_records_its_options_and_resumes_only_under_it(
    tmp_path: Path,
) -> None:
    """The flag is in the run's identity, its episodes count their options, and a
    resume must ask for the same exploration."""
    report = numbered(tmp_path / "first", 50, **{"--ez-greedy": None})
    arm = report["arm"]
    assert arm["resolved_config"]["ez_greedy"] is True
    records = arm["collected_episodes"]
    # The anneal starts at epsilon 1, so every early decision is exploratory.
    assert sum(record["options_started"] for record in records) > 0
    assert all(
        record["longest_option"] >= (1 if record["options_started"] else 0)
        for record in records
    )
    checkpoint = latest_checkpoint(report)

    resumed = survival_resume(tmp_path / "second", checkpoint, **{"--ez-greedy": None})
    assert resumed is not None and resumed.decisions > 0
    with pytest.raises(SystemExit, match="a different exploration"):
        survival_resume(tmp_path / "third", checkpoint)


def test_a_one_step_checkpoint_is_not_resumed_under_ez_greedy(tmp_path: Path) -> None:
    checkpoint = latest_checkpoint(numbered(tmp_path / "first", 50))

    with pytest.raises(SystemExit, match="a different exploration"):
        survival_resume(tmp_path / "second", checkpoint, **{"--ez-greedy": None})


def test_a_checkpoint_from_before_ez_greedy_resumes_with_it_off(tmp_path: Path) -> None:
    """A resolved config without the key reads as the exploration it collected under."""
    parent = load(latest_checkpoint(numbered(tmp_path / "first", 50)))
    older = tmp_path / "older.pt"
    settings = dict(parent.resolved_config)
    del settings["ez_greedy"]
    save(replace(parent, resolved_config=settings), older)

    assert resume_from(tmp_path / "second", older, 400).decisions > 0


# -- prioritized replay (board #85) ------------------------------------------


def test_stacked_dqn_samples_by_r2d2_priorities_and_records_them(
    trained: dict[str, Any],
) -> None:
    resolved = trained["arm"]["resolved_config"]
    assert (resolved["priority_alpha"], resolved["importance_beta"]) == (0.9, 0.6)


@pytest.mark.parametrize("backbone", ["stacked-dqn", "dreamerv3"])
def test_no_flag_can_change_how_replay_is_sampled(
    tmp_path: Path, backbone: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prioritization cannot silently be off again: there is no option for it."""
    with pytest.raises(SystemExit):
        train.parse_arguments(
            [
                "--budget-decisions", "1000", "--run-dir", str(tmp_path),
                "--backbone", backbone, "--priority-alpha", "0",
            ]
        )
    assert "unrecognized arguments: --priority-alpha" in capsys.readouterr().err


def test_a_checkpoint_sampled_uniformly_is_not_resumed_under_prioritized_replay(
    tmp_path: Path,
) -> None:
    """Every stacked-dqn file before #85 recorded alpha 0 and a beta anneal."""
    parent = load(latest_checkpoint(numbered(tmp_path / "first", 50)))
    older = tmp_path / "uniform.pt"
    settings = dict(parent.resolved_config)
    del settings["importance_beta"]
    settings.update(priority_alpha=0.0, beta_start=0.4, beta_end=1.0)
    save(replace(parent, resolved_config=settings), older)

    with pytest.raises(SystemExit, match="a different replay"):
        resume_from(tmp_path / "second", older, 400)


# -- The replay ratio, the buffer and what a resume keeps (boards #92, #93, #85)


def test_the_default_replay_ratio_replays_about_560_transitions_per_decision(
    tmp_path: Path,
) -> None:
    """Gradient steps per decision x sequences per step x learnable steps each.

    The learnable steps are counted by the target itself: a window of 80 with
    burn-in 7 leaves 73 steps, and the last n of them have no bootstrap state
    inside the window. The default reverted to M3-P009's 1.0 gradient steps
    per decision (board #85); at that ratio a batch of 8 replays about 560
    transitions per decision at the final n of 3.
    """
    defaults = train.parse_arguments(["--budget-decisions", "1000", "--run-dir", str(tmp_path)])
    unroll = defaults.sequence_length - defaults.stacked_burn_in
    rewards = torch.zeros(1, unroll)
    q = torch.zeros(1, unroll, len(RUN_ACTIONS))
    mask = torch.ones(1, unroll, len(RUN_ACTIONS), dtype=torch.bool)

    def replayed_per_decision(n_step: int) -> float:
        _, learnable = n_step_targets(
            rewards,
            rewards.bool(),
            q,
            q,
            mask,
            discounts=torch.full((1, unroll), 0.9, dtype=torch.float64),
            n_step=n_step,
        )
        return float(
            defaults.gradient_steps_per_decision * defaults.batch_size * learnable.sum().item()
        )

    # At the final n of the anneal the recipe runs, and while n is at its start.
    assert replayed_per_decision(3) == pytest.approx(560.0, abs=1.0)
    assert replayed_per_decision(defaults.n_step) == pytest.approx(504.0, abs=1.0)


def test_the_default_buffer_holds_about_170k_decisions(tmp_path: Path) -> None:
    """M3-P009's known-good capacity, deliberately short of the whole budget.

    25,000 windows would have held the whole ~1M-decision budget, but kept the
    early heavily-explored data forever and cost ~19 GiB with swap already
    full, with no evidence for it beyond covering the budget (M3-P010,
    docs/experiments.md, board #85). 4096 windows at baseline v2's
    550-decision episodes evict long before the budget is spent.
    """
    defaults = train.parse_arguments(["--budget-decisions", "1000", "--run-dir", str(tmp_path)])
    actor = Actor(
        environment=cast(InstrumentedRunEnvironment, None),
        policy=cast(Policy, None),
        config=ActorConfig(
            sequence_length=defaults.sequence_length,
            burn_in=defaults.stacked_burn_in,
            # What `build_arm` strides by.
            stride=max(1, defaults.sequence_length // 2),
        ),
    )
    step = ReplayStep(
        features=StateFeatures(
            scalars=(0.0,) * SCALAR_COUNT,
            rows=(0.0,) * (ROW_COUNT * ROW_WIDTH),
            mask=(True,) * len(RUN_ACTIONS),
        ),
        action_index=0,
        reward=0.0,
        done=False,
        admissible=True,
        game_ms=0.0,
    )
    decisions = 550
    windows = len(actor._windows([step] * decisions))

    assert windows == 13
    covered_decisions = defaults.replay_capacity * decisions / windows
    assert covered_decisions == pytest.approx(173_292, abs=1000)
    assert covered_decisions < 200_000, "capacity is deliberately short of the ~1M budget"


def test_a_resume_under_another_loop_setting_is_refused(tmp_path: Path) -> None:
    """None of them is in the identity, and a changed default would move them silently."""
    checkpoint = latest_checkpoint(session(tmp_path / "first"))

    with pytest.raises(
        SystemExit, match=r"--gradient-steps-per-decision 0\.2 \(this run asks for 0\.3\)"
    ):
        train.resume_point(
            arguments(
                tmp_path / "second",
                **{
                    "--budget-decisions": "1000",
                    "--resume": str(checkpoint),
                    "--gradient-steps-per-decision": "0.3",
                },
            ),
            profile_id=PROFILE,
            revision="test",
        )


def test_a_parameter_sync_of_one_episode_reads_as_a_cadence_of_zero() -> None:
    """Every checkpoint before the cadence in decisions refreshed once per episode."""
    assert train.recorded_loop_settings({"parameter_sync_episodes": 1}) == {
        "parameter_sync_decisions": 0
    }
    assert train.recorded_loop_settings({}) == {}


def test_a_parameter_sync_of_several_episodes_is_refused_as_having_no_equivalent() -> None:
    with pytest.raises(SystemExit, match="--parameter-sync-episodes 3, which has no equivalent"):
        train.recorded_loop_settings({"parameter_sync_episodes": 3})


def test_resets_stop_one_interval_before_the_steps_the_budget_buys(tmp_path: Path) -> None:
    """BBF's `no_resets_after`: 1M decisions at 1.0 and 100k gives 9 resets."""
    resetting = arguments(
        tmp_path,
        **{
            "--budget-decisions": "1000000",
            "--gradient-steps-per-decision": "1.0",
            "--reset-every-steps": "100000",
        },
    )
    _, learner, _ = train.build_backbone(resetting, torch.device("cpu"))

    assert (learner.reset_every_steps, learner.last_reset_step) == (100_000, 900_000)
    assert train.last_reset_step(arguments(tmp_path)) == 0


def test_the_learner_resets_are_logged_as_they_happen(tmp_path: Path) -> None:
    """A dip in the curve can be put against the reset that caused it."""
    tracker = RecordingTracker()
    summary = session(
        tmp_path,
        budget="200",
        tracker=tracker,
        settings={"--reset-every-steps": "5"},
    )

    logged = [
        point.metrics["learner_resets"]
        for point in tracker.runs[0].points
        if "learner_resets" in point.metrics
    ]
    assert logged, "the run was long enough to reset"
    # Once per change: an episode can span more than one reset.
    assert logged == sorted(set(logged)) and logged[0] >= 1.0
    resolved = summary["arm"]["resolved_config"]
    # 200 decisions at 0.2 buy 40 steps, and the last interval is left alone.
    assert (resolved["reset_every_steps"], resolved["last_reset_step"]) == (5, 35)
    # The learner steps on its own thread (ADR 0017), so the last reset may be
    # taken in the block's closing drain, after the last episode's hook: it is
    # logged when the block ends.
    assert logged[-1] == 7.0


def test_a_resume_is_held_to_its_reset_interval() -> None:
    assert train.recorded_loop_settings({"reset_every_steps": 100_000}) == {
        "reset_every_steps": 100_000
    }
    # A DreamerV3 run records it as None, and has nothing to compare.
    assert train.recorded_loop_settings({"reset_every_steps": None}) == {}


def test_a_checkpoint_from_before_the_adam_epsilon_resumes_at_its_own(tmp_path: Path) -> None:
    """Its optimizer state carries 1e-8, torch restores it, and the record says so.

    Rewritten as a pre-change checkpoint is: the optimizer at torch's epsilon,
    no `adam_epsilon` in its settings, and a parameter lag of one episode.
    """
    checkpoint = latest_checkpoint(session(tmp_path / "first"))
    old = load(checkpoint)
    state = dict(old.backbone_state)
    optimizer = dict(state["optimizer"])
    optimizer["param_groups"] = [{**group, "eps": 1e-8} for group in optimizer["param_groups"]]
    state["optimizer"] = optimizer
    settings = {
        key: value
        for key, value in old.resolved_config.items()
        if key not in ("adam_epsilon", "parameter_sync_decisions")
    }
    settings["parameter_sync_episodes"] = 1
    save(replace(old, backbone_state=state, resolved_config=settings), checkpoint)

    with pytest.raises(SystemExit, match="--parameter-sync-decisions 0"):
        resumed_arm(tmp_path / "refused", checkpoint, 1000)

    arm, _ = resumed_arm(
        tmp_path / "second", checkpoint, 1000, **{"--parameter-sync-decisions": "0"}
    )

    assert arm.backbone.optimizer.param_groups[0]["eps"] == 1e-8
    assert arm.resolved["adam_epsilon"] == 1e-8
    assert arm.training.config.parameter_sync_decisions == 0


def test_dreamerv3_is_refreshed_only_between_episodes(tmp_path: Path) -> None:
    """Its latent is state the parameters produced; a swap inside an episode would split it."""
    assert dreamer_arguments(tmp_path).parameter_sync_decisions == 0
    with pytest.raises(SystemExit, match="contradicts DreamerV3"):
        dreamer_arguments(tmp_path, "--parameter-sync-decisions", "100")
