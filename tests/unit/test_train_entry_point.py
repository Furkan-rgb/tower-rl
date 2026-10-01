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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from tower_rl.learning.actor import ActorConfig
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
from tower_rl.learning.policies import checkpoint_policy
from tower_rl.learning.r2d2 import R2D2Backbone, R2D2Config
from tower_rl.learning.r2d2_replay import R2D2Replay, R2D2ReplayImage
from tower_rl.learning.replay import (
    R2D2_PRIORITY_EXPONENT,
    REPLAY_DUMP_METADATA,
    read_replay_metadata,
)
from tower_rl.learning.training import TrainingRun
from tower_rl.learning.value_learning import V_REF
from tower_rl.simulation.instance import CloneInstance

PROFILE = "fake-profile-v1"

#: A network narrow enough that the entry point can be exercised in seconds. At
#: production width every decision is a CPU forward pass and dominates the run;
#: what is under test here is the plumbing around the learner, not its capacity,
#: which the backbone contract suite covers.
SMALL_NETWORK = NetworkConfig(hidden=16, identity_dim=4)

#: R2D2 reduced for the CPU: a target copy every 3 steps, a refresh every 10
#: decisions, two items to warm, a batch of two, room for 64 items. Test-only.
R2D2_SMOKE_TARGET_PERIOD = 3
R2D2_SMOKE_REFRESH = 10
R2D2_SMOKE_CAPACITY = 64

#: The task's discount and reward, which an R2D2 run must give (ADR 0013).
R2D2_TASK: dict[str, str | None] = {
    "--discount-per-game-second": "0.999",
    "--survival-time-reward": None,
}


@contextmanager
def reduced_r2d2(warmup_items: int = 2) -> Iterator[None]:
    """The entry point's R2D2 constants at the smoke size, for parsing and for building.

    The published sizes are fixed settings the parser refuses to contradict, so
    the reduction is made where they are read: the parser and `build_arm` see
    the same small buffer, batch and refresh, and a resume sees its parent's.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        patch.setattr(
            train,
            "R2D2Config",
            lambda **given: R2D2Config(target_update_period=R2D2_SMOKE_TARGET_PERIOD, **given),
        )
        patch.setattr(train, "R2D2_MIN_REPLAY_ITEMS", warmup_items)
        patch.setattr(train, "R2D2_BATCH_SIZE", 2)
        patch.setattr(train, "R2D2_REPLAY_CAPACITY", R2D2_SMOKE_CAPACITY)
        patch.setattr(train, "ACTOR_REFRESH_DECISIONS", R2D2_SMOKE_REFRESH)
        yield


#: What `main` is given beside the budget: the arm and the task's discount and reward.
R2D2_FLAGS = [
    "--backbone", "r2d2", "--discount-per-game-second", "0.999", "--survival-time-reward",
]


def arguments(
    run_dir: Path, *, warmup_items: int = 2, **overrides: str | None
) -> argparse.Namespace:
    """The real parser at the R2D2 smoke size, so the entry point's own checks are used.

    A value of None gives the flag alone, as a switch is given.
    """
    argv = ["--backbone", "r2d2"]
    settings: dict[str, str | None] = {
        **R2D2_TASK,
        "--budget-decisions": "200",
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
    with reduced_r2d2(warmup_items):
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
    warmup_items: int = 2,
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
    parsed = arguments(run_dir, warmup_items=warmup_items, **overrides)
    with reduced_r2d2(warmup_items):
        return train.train_session(
            parsed,
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

    assert arm["backbone"] == "r2d2"
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


# -- --backbone dreamerv3 ------------------------------------------------------
#
# Its fixed loop settings, and one session at a size that trains in seconds, as
# this suite shrinks R2D2's network; the published sizes are what
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
        patch.setattr(train, "DREAMER_WARMUP_ITEMS", 2)
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
    assert (parsed.sequence_length, parsed.burn_in) == (64, 0)
    assert parsed.batch_size == 16
    assert parsed.gradient_steps_per_decision == 0.5
    # One batch of items, and the official replay size in items (steps).
    assert parsed.warmup_sequences == 16 * 64
    assert parsed.replay_capacity == 5_000_000
    assert parsed.exploration == "uniform"
    assert (parsed.epsilon_start, parsed.epsilon_end) == (0.0, 0.0)


def test_the_backbone_has_no_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The backbone is the arm: a run that does not name one is refused, not guessed."""
    with pytest.raises(SystemExit):
        train.parse_arguments(["--budget-decisions", "1000", "--run-dir", str(tmp_path)])
    assert "--backbone" in capsys.readouterr().err


def test_a_flag_that_repeats_a_dreamerv3_value_is_accepted(tmp_path: Path) -> None:
    parsed = dreamer_arguments(tmp_path, "--batch-size", "16", "--sequence-length", "64")
    assert parsed.batch_size == 16


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--batch-size", "8"),
        ("--sequence-length", "80"),
        ("--burn-in", "7"),
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


#: The flags of the deleted stacked-dqn backbone (cb2f324 is the last commit that
#: has it): gone from the parser, so a command line that still gives one fails
#: loudly under either backbone rather than being read as something else.
RETIRED_STACKED_DQN_FLAGS = [
    ("--history-length", "8"),
    ("--stacked-burn-in", "7"),
    ("--n-step", "3"),
    ("--discount", "0.99"),
    ("--learning-rate", "1e-4"),
    ("--target-ema-decay", "0.995"),
    ("--n-step-final", "3"),
    ("--n-step-anneal-steps", "100"),
    ("--reset-every-steps", "1000"),
    ("--ez-greedy",),
    ("--priority-alpha", "0"),
]


@pytest.mark.parametrize("flags", RETIRED_STACKED_DQN_FLAGS)
@pytest.mark.parametrize("backbone", ["dreamerv3", "r2d2"])
def test_a_retired_stacked_dqn_flag_is_refused_as_unrecognized(
    tmp_path: Path, backbone: str, flags: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        train.parse_arguments(
            [
                "--budget-decisions", "1000", "--run-dir", str(tmp_path),
                "--backbone", backbone, *DREAMER_DISCOUNT, "--survival-time-reward", *flags,
            ]
        )
    assert f"unrecognized arguments: {flags[0]}" in capsys.readouterr().err


def test_stacked_dqn_is_not_a_backbone(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        train.parse_arguments(
            ["--budget-decisions", "1000", "--run-dir", str(tmp_path), "--backbone", "stacked-dqn"]
        )
    assert "invalid choice: 'stacked-dqn'" in capsys.readouterr().err


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
    assert (resolved["sequence_length"], resolved["burn_in"]) == (6, 0)
    assert resolved["dreamer_deter"] == 16 and resolved["dreamer_train_ratio"] == 3.0
    # On the CPU the learner computes in float32; on CUDA, in bfloat16.
    assert resolved["dreamer_compute_dtype"] == "float32"
    # Uniform replay, as the official loop samples; not an option (board #85).
    assert (resolved["priority_alpha"], resolved["importance_beta"]) == (0.0, 0.0)
    assert resolved["network_hidden"] is None
    # The task's discount and reward, under the keys every backbone records them under.
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
        assert checkpoint.identity.backbone == "r2d2"
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
    defaults = r2d2_arguments(tmp_path)
    assert defaults.checkpoint_every_decisions == 0
    assert defaults.selection_period_decisions == 15_000

    _, directory, written = numbered_checkpoints(session(tmp_path, budget="200"))

    assert written == []
    assert (directory / "latest.pt").exists()


def test_early_stopping_is_off_by_default_and_resolved_with_its_threshold(
    tmp_path: Path,
) -> None:
    """Every run measured so far spent its whole budget; that stays the default."""
    defaults = r2d2_arguments(tmp_path)

    assert defaults.early_stop_patience_periods == 0
    assert defaults.early_stop_min_improvement == 0.2


def test_kill_bars_are_off_by_default(tmp_path: Path) -> None:
    assert arguments(tmp_path).kill_bars == []


def test_a_kill_bar_is_parsed_as_at_start_minimum(tmp_path: Path) -> None:
    parsed = r2d2_arguments(
        tmp_path, "--kill-bar", "12000:8000:8.6", "--kill-bar", "26262:8000:10.2"
    )

    assert [
        (bar.at_decisions, bar.window_start_decisions, bar.min_mean_final_wave)
        for bar in parsed.kill_bars
    ] == [(12000, 8000, 8.6), (26262, 8000, 10.2)]
    for malformed in ("12000:8000", "8000:12000:8.6", "a:b:c"):
        with pytest.raises(SystemExit):
            r2d2_arguments(tmp_path, "--kill-bar", malformed)


def test_a_run_below_its_kill_bar_stops_and_says_which_bar(tmp_path: Path) -> None:
    """The stop is the run's own and is recorded like the plateau stop.

    Two actors: a bar reads the near-greedy ones, and a lone actor's ladder is
    its base rate of 0.4.
    """
    report = session(
        tmp_path,
        budget=TRAINING_BUDGET,
        actors=2,
        settings={"--kill-bar": "100:0:1000"},
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
        actors=2,
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
            r2d2_arguments(tmp_path, retired, "100")


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
    parsed = arguments(
        tmp_path,
        **{
            "--budget-decisions": "40",
            "--actors": "2",
            "--serial": CloneInstance(index=0).serial,
            "--evaluate-every-episodes": "0",
        },
    )
    with reduced_r2d2():
        report = train.train_session(
            parsed,
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
    dead = next(actor for actor in arm["actors"] if actor["actor_id"] == "fake-1:r2d2")
    alive = next(actor for actor in arm["actors"] if actor["actor_id"] == "fake-0:r2d2")
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
    assert "fake-1:r2d2" in announcement and dead["withdrawn"] in announcement


def test_a_single_actor_run_records_exactly_one_actor(tmp_path: Path) -> None:
    """The default, and the configuration the in-flight run is reproducible from."""
    parsed = r2d2_arguments(tmp_path)
    assert parsed.actors == 1

    report = session(tmp_path)

    assert report["actors"] == 1 and report["actor_serials"] == ["fake-0"]
    arm = report["arm"]
    assert arm["resolved_config"]["actors"] == 1
    assert [actor["actor_id"] for actor in arm["actors"]] == ["fake-0:r2d2"]
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
        sys,
        "argv",
        ["train.py", "--budget-decisions", "1000", "--actors", "2", "--no-track", *R2D2_FLAGS],
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
         "--frame-rate-hz", "90", "--no-track", *R2D2_FLAGS]
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
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--budget-decisions", "1000", "--no-track", *R2D2_FLAGS]
    )

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
    with reduced_r2d2():
        arm, _ = train.build_arm(
            "r2d2",
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
    resolved = session(tmp_path, actors=2)["arm"]["resolved_config"]

    assert resolved["exploration"] == "ladder"
    assert resolved["exploration_epsilon_floors"] == pytest.approx(list(ape_x_floors(2)))
    # Fixed from the first decision: nothing anneals.
    assert resolved["epsilon_anneal_decisions"] == 0
    assert resolved["epsilon_start"] == 0.4


def test_an_epsilon_end_passed_with_the_ladder_is_refused(tmp_path: Path) -> None:
    """The ladder replaces the end of the anneal, so the flag would go unused."""
    with pytest.raises(SystemExit, match="--epsilon-end"):
        arguments(tmp_path, **{"--exploration": "ladder", "--epsilon-end": "0.001"})
    # The equals form is the same flag and is refused the same way: what is read
    # is the parsed value, not the shape of the argument vector.
    with pytest.raises(SystemExit, match="--epsilon-end"):
        r2d2_arguments(tmp_path, "--exploration", "ladder", "--epsilon-end=0.05")

    # The ladder alone is what R2D2 fixes, so repeating it is ordinary.
    assert arguments(tmp_path, **{"--exploration": "ladder"}).exploration == "ladder"


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
    # Exploration is derived from the counter rather than restored; R2D2's is
    # the fixed ladder, so it is where the run began it.
    assert progress.epsilon == config.exploration.epsilon_for(0, spent) == ape_x_floors(1)[0]
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
            *R2D2_FLAGS,
            "--run-dir",
            str(tmp_path / "second"),
        ],
    )

    # At the parent's sizes, so the replay it saved is reloaded rather than refused.
    with reduced_r2d2():
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
    assert metadata["inserted"] == first["arm"]["replay"]["sequences"] > 0
    assert metadata["capacity"] == R2D2_SMOKE_CAPACITY
    assert metadata["alpha"] == R2D2_PRIORITY_EXPONENT


def test_every_latest_checkpoint_a_run_writes_is_written_with_its_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not only the last: a kill between two of them leaves a matching pair."""
    pairs: list[tuple[int, int]] = []
    write = train.TrainingReport._write_resume_point

    def recorded(self: Any, report: Any, image: R2D2ReplayImage) -> None:
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
    expected = R2D2Replay(capacity=R2D2_SMOKE_CAPACITY)
    expected.load_from(dump)

    arm, resume = resumed_arm(tmp_path / "second", latest_checkpoint(first), budget=400)

    assert resume.replay_dump == dump
    assert len(arm.replay) == len(expected) > 0
    for column in ("_episode", "_offset", "_length", "_priority"):
        assert numpy.array_equal(getattr(arm.replay, column), getattr(expected, column)), column
    # The sampler goes on from where it stood, not from the seed.
    assert arm.replay._random.bit_generator.state == expected._random.bit_generator.state
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
    # R2D2 fixes its buffer size, so a changed one is a changed constant, read
    # into the parsed arguments as the parser reads it.
    asking = arguments(
        tmp_path / "second",
        **{"--budget-decisions": "400", "--resume": str(latest_checkpoint(first))},
    )
    asking.replay_capacity = 2 * R2D2_SMOKE_CAPACITY
    with pytest.raises(SystemExit, match=f"capacity {R2D2_SMOKE_CAPACITY}"):
        train.resume_point(asking, profile_id=PROFILE, revision="test")


def test_a_resumed_run_learns_from_the_reloaded_replay_without_re_warming(
    tmp_path: Path,
) -> None:
    """End to end: a warm-up the new segment alone could not reach is already met."""
    first = numbered(tmp_path / "first", 200)
    parent_steps = first["arm"]["optimisation_steps"]
    budget = first["arm"]["decisions"] + 60
    # More items than 60 decisions can collect, but no more than were saved.
    warmup = {"warmup_items": 5}
    assert first["arm"]["replay"]["sequences"] >= 5

    second = session(
        tmp_path / "second",
        budget=str(budget),
        resume=resume_from(tmp_path / "second", latest_checkpoint(first), budget),
        **warmup,
    )
    assert second["arm"]["optimisation_steps"] > parent_steps

    shutil.rmtree(saved_replay(tmp_path / "first").parent)
    third = session(
        tmp_path / "third",
        budget=str(budget),
        resume=resume_from(tmp_path / "third", latest_checkpoint(first), budget),
        **warmup,
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


#: Steps a stopped parent still owes. R2D2's credits are whole items of
#: `gradient_steps_per_item` steps each, so a run that spends its budget drains
#: to no debt; a part of one is what an interrupted parent leaves, written here.
OWED_STEPS = 3.0


def owing_checkpoint(tmp_path: Path) -> Path:
    """A parent's `latest.pt` as a stopped run writes it: steps still owed on its buffer."""
    checkpoint = latest_checkpoint(numbered(tmp_path / "first", 100))
    parent = load(checkpoint)
    assert parent.progress.learner_debt_steps == 0, "a run that spends its budget drains"
    owing = replace(
        parent, progress=replace(parent.progress, learner_debt_steps=OWED_STEPS)
    )
    save(owing, checkpoint)
    return checkpoint


def test_a_resume_pays_the_debt_its_parent_still_owed(tmp_path: Path) -> None:
    """The steps a resumed run takes are the ones an uninterrupted run's rule gives it.

    The debt a parent ended owing is written into its checkpoint (ADR 0017).
    The resume carries it, so over the two segments the steps stay the credits'
    worth and what was owed is not dropped at the seam.
    """
    checkpoint = owing_checkpoint(tmp_path)
    parent = load(checkpoint)

    arm, resume = resumed_arm(tmp_path / "second", checkpoint, budget=400)
    assert resume.learner_debt_steps == OWED_STEPS
    assert arm.training.learner_thread.counted_debt_steps() == pytest.approx(OWED_STEPS)

    second = session(
        tmp_path / "third",
        budget="400",
        resume=resume_from(tmp_path / "third", checkpoint, 400),
    )

    taken = second["arm"]["optimisation_steps"] - parent.progress.optimisation_steps
    per_item = int(second["arm"]["resolved_config"]["gradient_steps_per_item"])
    # What is owed plus whole items' worth, all paid: without the carried debt
    # the segment would take a multiple of the items' steps alone.
    assert taken > OWED_STEPS
    assert taken % per_item == OWED_STEPS % per_item != 0


def test_a_resume_that_rewarms_its_buffer_owes_nothing(tmp_path: Path) -> None:
    """A debt owed on the parent's buffer is not carried onto an empty one (ADR 0017)."""
    checkpoint = owing_checkpoint(tmp_path)
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


def refuse_to_write(self: R2D2ReplayImage, directory: Path, **kwargs: Any) -> int:
    raise OSError("disk full")


def test_a_failed_replay_save_neither_masks_the_error_nor_moves_the_resume_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `latest.pt` without its replay would be a resume point that loses the buffer."""
    monkeypatch.setattr(R2D2ReplayImage, "write", refuse_to_write)
    with pytest.raises(RuntimeError, match="the run failed"):
        interrupted_session(tmp_path / "failed", monkeypatch)
    assert not list((tmp_path / "failed").glob("*/checkpoints/latest.pt"))

    # A run that spends its budget is not failed by it either.
    monkeypatch.undo()
    monkeypatch.setattr(R2D2ReplayImage, "write", refuse_to_write)
    report = session(tmp_path / "finished", budget="200")
    assert report["arm"]["decisions"] >= 200
    assert not list((tmp_path / "finished").glob(f"*/{REPLAY_DIRECTORY}/*"))


# -- discounting by game time (board #81) ------------------------------------


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

    policy, _ = checkpoint_policy(
        latest_checkpoint(report),
        decision_cadence=resolved["decision_cadence"],
        upgrade_availability=resolved["upgrade_availability"],
        workshop_level=0,
    )
    assert isinstance(policy, R2D2Backbone)
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


def test_a_resume_under_another_discount_is_refused(tmp_path: Path) -> None:
    """A different discount is a different target: one set of weights, two scales."""
    checkpoint = latest_checkpoint(numbered(tmp_path / "first", 50))

    with pytest.raises(SystemExit, match="a different target"):
        train.resume_point(
            arguments(
                tmp_path / "second",
                **{
                    "--budget-decisions": "400",
                    "--resume": str(checkpoint),
                    "--discount-per-game-second": "0.997",
                },
            ),
            profile_id=PROFILE,
            revision="test",
        )


# -- the survival-time reward (board #82) ------------------------------------

SURVIVAL: dict[str, str | None] = {
    "--discount-per-game-second": "0.997",
    "--survival-time-reward": None,
}


def test_the_survival_time_reward_is_refused_without_the_game_time_discount(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="needs --discount-per-game-second"):
        train.parse_arguments(
            [
                "--budget-decisions", "1000", "--run-dir", str(tmp_path),
                "--backbone", "dreamerv3", "--survival-time-reward",
            ]
        )


def test_dreamerv3_takes_the_survival_time_reward(tmp_path: Path) -> None:
    """The reward is the task's (ADR 0013): the flag means what it means for R2D2."""
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
    """Format 6: zero-start windows and no stored latents. Resuming it would mix two replays.

    It recorded an actor unimix of 0.01 and no mask encoding, and its network
    took the mask as 0/1.
    """
    from tower_rl.learning.dreamer import DreamerBackbone

    checkpoint = latest_checkpoint(dreamer_session(tmp_path / "first"))
    parent = load(checkpoint)
    settings = {**parent.resolved_config, "dreamer_actor_unimix": 0.01}
    del settings["dreamer_mask_one_hot"]
    playing = dict(
        decision_cadence=settings["decision_cadence"],
        upgrade_availability=settings["upgrade_availability"],
        workshop_level=0,
    )
    current, _ = checkpoint_policy(checkpoint, **playing)
    state = DreamerBackbone(config=replace(current.config, mask_one_hot=False)).state_dict()
    older = tmp_path / "older.pt"
    save(
        replace(parent, format_version=6, resolved_config=settings, backbone_state=state),
        older,
    )
    with pytest.raises(SystemExit, match="mixed run"):
        dreamer_resume(tmp_path / "second", older)
    # It still plays, with the actor unimix and the mask encoding it was trained with.
    policy, _ = checkpoint_policy(older, **playing)
    assert policy.config.actor_unimix == 0.01
    assert policy.config.mask_one_hot is False
    features = StateFeatures(
        scalars=(0.5,) * SCALAR_COUNT,
        rows=(0.5,) * (ROW_COUNT * ROW_WIDTH),
        mask=tuple(index < 3 for index in range(len(RUN_ACTIONS))),
    )
    action, _ = policy.act(features, policy.initial_state(), epsilon=0.0)
    assert features.mask[action]


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
    """The key is in the run's identity, and a resume must ask for the same target."""
    report = session(tmp_path, settings=SURVIVAL)
    assert report["arm"]["resolved_config"]["survival_time_reward"] is True
    assert report["arm"]["resolved_config"]["survival_reward_bound"] == V_REF
    checkpoint = latest_checkpoint(report)

    resumed = survival_resume(tmp_path / "second", checkpoint, **SURVIVAL)
    assert resumed is not None and resumed.decisions > 0
    with pytest.raises(SystemExit, match="a different target"):
        survival_resume(
            tmp_path / "third", checkpoint, **{"--discount-per-game-second": "0.999"}
        )


def wave_reward_checkpoint(tmp_path: Path, *, forget_the_key: bool) -> Path:
    """An R2D2 checkpoint rewritten as a wave-reward run left it (DreamerV3's option).

    R2D2 learns the survival reward only, so the file is rewritten: either
    saying the reward was off, or, as a file from before the survival-time
    reward, saying nothing about it.
    """
    parent = load(latest_checkpoint(numbered(tmp_path / "first", 50)))
    settings = {
        **parent.resolved_config,
        "survival_time_reward": False,
        "survival_reward_bound": None,
    }
    if forget_the_key:
        del settings["survival_time_reward"]
    older = tmp_path / "wave.pt"
    save(replace(parent, resolved_config=settings), older)
    return older


@pytest.mark.parametrize("forget_the_key", [False, True], ids=["off", "unrecorded"])
def test_a_wave_reward_checkpoint_is_not_resumed_under_the_survival_time_reward(
    tmp_path: Path, forget_the_key: bool
) -> None:
    """The reward defines the target as the discount does: one set of weights, two scales.

    A file with no key reads as the wave reward it learned from.
    """
    checkpoint = wave_reward_checkpoint(tmp_path, forget_the_key=forget_the_key)

    with pytest.raises(SystemExit, match="a different target"):
        survival_resume(tmp_path / "second", checkpoint)


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


# -- prioritized replay (board #85) ------------------------------------------


def test_r2d2_samples_by_priority_and_records_the_exponents(trained: dict[str, Any]) -> None:
    resolved = trained["arm"]["resolved_config"]
    assert (resolved["priority_alpha"], resolved["importance_beta"]) == (0.9, 0.6)


def test_a_resume_under_another_loop_setting_is_refused(tmp_path: Path) -> None:
    """None of them is in the identity, and a changed default would move them silently."""
    parent = load(latest_checkpoint(session(tmp_path / "first")))
    longer = tmp_path / "longer-refresh.pt"
    save(
        replace(
            parent,
            resolved_config={
                **parent.resolved_config,
                "parameter_sync_decisions": 2 * R2D2_SMOKE_REFRESH,
            },
        ),
        longer,
    )

    with pytest.raises(
        SystemExit,
        match=rf"--parameter-sync-decisions {2 * R2D2_SMOKE_REFRESH} "
        rf"\(this run asks for {R2D2_SMOKE_REFRESH}\)",
    ):
        resume_from(tmp_path / "second", longer, 1000)

    # From before the cadence was in decisions: a refresh at every episode start.
    episodes = {
        key: value
        for key, value in parent.resolved_config.items()
        if key != "parameter_sync_decisions"
    }
    older = tmp_path / "per-episode-refresh.pt"
    save(replace(parent, resolved_config={**episodes, "parameter_sync_episodes": 1}), older)
    with pytest.raises(SystemExit, match="--parameter-sync-decisions 0"):
        resume_from(tmp_path / "third", older, 1000)


def test_a_parameter_sync_of_one_episode_reads_as_a_cadence_of_zero() -> None:
    """Every checkpoint before the cadence in decisions refreshed once per episode."""
    assert train.recorded_loop_settings({"parameter_sync_episodes": 1}) == {
        "parameter_sync_decisions": 0
    }
    assert train.recorded_loop_settings({}) == {}


def test_a_parameter_sync_of_several_episodes_is_refused_as_having_no_equivalent() -> None:
    with pytest.raises(SystemExit, match="--parameter-sync-episodes 3, which has no equivalent"):
        train.recorded_loop_settings({"parameter_sync_episodes": 3})


def test_dreamerv3_acts_on_the_last_finished_step_at_every_decision(tmp_path: Path) -> None:
    """The official agent swaps parameters at the next policy call after each step."""
    assert dreamer_arguments(tmp_path).parameter_sync_decisions == 1
    with pytest.raises(SystemExit, match="contradicts DreamerV3"):
        dreamer_arguments(tmp_path, "--parameter-sync-decisions", "100")


# -- R2D2 --------------------------------------------------------------------

def r2d2_arguments(run_dir: Path, *flags: str) -> argparse.Namespace:
    """The parser at R2D2's published sizes, which `arguments` reduces."""
    return train.parse_arguments(
        [
            "--budget-decisions", "1000", "--run-dir", str(run_dir), "--backbone", "r2d2",
            "--discount-per-game-second", "0.999", "--survival-time-reward", *flags,
        ]
    )


def test_r2d2_fixes_its_published_loop_settings(tmp_path: Path) -> None:
    parsed = r2d2_arguments(tmp_path)
    assert (parsed.sequence_length, parsed.burn_in) == (80, 40)
    assert (parsed.batch_size, parsed.warmup_sequences) == (64, 1_250)
    assert parsed.replay_capacity == 100_000
    assert parsed.exploration == "ladder" and parsed.epsilon_anneal_decisions == 0
    assert parsed.parameter_sync_decisions == 400
    # Not flags any more: the backbone's own published values.
    config = R2D2Config(discount_per_game_second=0.999)
    assert (config.n_step, config.learning_rate) == (5, 1e-4)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--batch-size", "32"),
        ("--sequence-length", "64"),
        ("--exploration", "uniform"),
        ("--epsilon-anneal-decisions", "8000"),
        ("--parameter-sync-decisions", "100"),
        ("--warmup-sequences", "100"),
    ],
)
def test_a_flag_that_contradicts_an_r2d2_value_is_refused(
    tmp_path: Path, flag: str, value: str
) -> None:
    with pytest.raises(SystemExit, match="contradicts R2D2"):
        r2d2_arguments(tmp_path, flag, value)


@pytest.mark.parametrize("flag", ["--epsilon-end", "--gradient-steps-per-decision"])
def test_a_flag_r2d2_does_not_read_is_refused(tmp_path: Path, flag: str) -> None:
    """Its ladder has no floor to anneal to, and its replay ratio is per item."""
    with pytest.raises(SystemExit, match=f"{flag} is not read by --backbone r2d2"):
        r2d2_arguments(tmp_path, flag, "0.5")


@pytest.mark.parametrize(
    ("task", "needs"),
    [
        (("--discount-per-game-second", "0.999"), "--backbone r2d2 learns the survival reward"),
        (("--survival-time-reward",), "--backbone r2d2 discounts by game time"),
    ],
)
def test_r2d2_needs_the_game_time_discount_and_the_survival_reward(
    tmp_path: Path, task: tuple[str, ...], needs: str
) -> None:
    with pytest.raises(SystemExit, match=needs):
        train.parse_arguments(
            ["--budget-decisions", "1000", "--run-dir", str(tmp_path), "--backbone", "r2d2", *task]
        )


def r2d2_session(
    run_dir: Path, budget: str, resume: Any = None, loads: list[int] | None = None
) -> dict[str, Any]:
    """One R2D2 session through the entry point, counting the acting copies' refreshes."""
    from tower_rl.learning.learner import Learner

    publish_to = Learner.publish_to

    def counted(learner: Learner, acting: Any) -> int:
        if loads is not None:
            loads.append(learner.published)
        return publish_to(learner, acting)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Learner, "publish_to", counted)
        return session(
            run_dir,
            budget=budget,
            settings={"--evaluation-episodes": "1", "--evaluate-every-episodes": "0"},
            resume=(
                None
                if resume is None
                else train.resume_point(
                    arguments(
                        run_dir,
                        **{"--budget-decisions": budget, "--resume": str(resume)},
                    ),
                    profile_id=PROFILE,
                    revision="test",
                )
            ),
        )


def test_an_r2d2_session_learns_copies_its_target_refreshes_checkpoints_and_resumes(
    tmp_path: Path,
) -> None:
    """The CPU smoke run of `--backbone r2d2`, at test-only reduced settings (ADR 0018)."""
    loads: list[int] = []
    first = r2d2_session(tmp_path / "first", "200", loads=loads)
    arm = first["arm"]
    assert arm["backbone"] == "r2d2" and arm["failed_episodes"] == 0
    steps = arm["optimisation_steps"]
    # Learning: five steps per item inserted, whole items only, the warming
    # episode's included and nothing before it.
    assert steps >= 2 * R2D2_SMOKE_TARGET_PERIOD
    assert steps % 5 == 0 and steps <= 5 * arm["sequences_accepted"]
    # Refreshes inside episodes, counted across them: the first load at the
    # first episode's start, then one per 10 decisions, never one per episode.
    assert len(loads) >= 2
    assert len(loads) <= 1 + arm["decisions"] // R2D2_SMOKE_REFRESH

    resolved = arm["resolved_config"]
    assert resolved["r2d2_target_update_period"] == R2D2_SMOKE_TARGET_PERIOD
    assert resolved["gradient_steps_per_item"] == 5.0
    assert resolved["learner_debt_bound_items"] == 125
    assert resolved["gradient_steps_per_decision"] is None
    assert resolved["exploration_epsilon_floors"] == [0.4]

    # A checkpoint: format 8, the target a copy of an online network at a
    # multiple of the period, so no longer the target it was initialised to.
    checkpoint = latest_checkpoint(first)
    saved = load(checkpoint)
    assert saved.format_version == 8 and saved.identity.backbone == "r2d2"
    state = saved.backbone_state
    fresh = R2D2Backbone(
        R2D2Config(discount_per_game_second=0.999, seed=resolved["seed"]),
        network_config=SMALL_NETWORK,
    ).state_dict()
    assert any(
        not torch.equal(state["target"][key], fresh["target"][key]) for key in fresh["target"]
    )
    # It plays as a policy.
    policy, _ = checkpoint_policy(
        checkpoint,
        decision_cadence=resolved["decision_cadence"],
        upgrade_availability=resolved["upgrade_availability"],
        workshop_level=0,
    )
    assert isinstance(policy, R2D2Backbone)

    # The resume pair: the dump holds items with their stored states.
    dump = saved_replay(tmp_path / "first")
    dump_decisions, decisions = paired_decisions(dump)
    assert dump_decisions == decisions
    restored = R2D2Replay(capacity=64)
    restored.load_from(dump)
    assert len(restored) > 0

    second = r2d2_session(tmp_path / "second", "400", resume=checkpoint)
    resumed = second["arm"]
    assert resumed["resolved_config"]["replay_restored_from"] == str(dump)
    assert resumed["decisions"] >= 400
    assert resumed["optimisation_steps"] > steps
