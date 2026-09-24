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
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import train
from fakes.fake_run_port import FakeRunPort
from fakes.recording_tracker import RecordingTracker

from tower_rl.environment.episode import TerminationOutcome
from tower_rl.environment.features import StateFeatures
from tower_rl.environment.run_environment import (
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.experiment.metrics import per_hour
from tower_rl.experiment.tracking import NoExperimentTracker
from tower_rl.experiment.training_report import numbered_checkpoint_name
from tower_rl.learning.actor import ActorConfig
from tower_rl.learning.checkpoint import Checkpoint, identity_hash, load, save
from tower_rl.learning.evaluator import evaluate
from tower_rl.learning.exploration import ape_x_floors
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.policies import checkpoint_policy
from tower_rl.simulation.instance import CloneInstance

#: Tensors this small spend their time handing work between threads rather than
#: computing: one thread runs the whole file about fifteen times faster.

PROFILE = "fake-profile-v1"

#: A network narrow enough that the entry point can be exercised in seconds. At
#: production width every decision is a CPU forward pass and dominates the run;
#: what is under test here is the plumbing around the learner, not its capacity,
#: which the backbone contract suite covers.
SMALL_NETWORK = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

def arguments(run_dir: Path, **overrides: str) -> argparse.Namespace:
    """The real parser, so the entry point's own defaults and checks are used."""
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
    settings: dict[str, str] | None = None,
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

    assert defaults.gradient_steps_per_decision == 0.25
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
    assert defaults.priority_alpha == 0.0, "importance weights of exactly one"
    assert defaults.replay_capacity == 4096
    assert defaults.collection_window_episodes == 100
    assert defaults.evaluate_every_episodes == 0, "no frequent mid-run evaluation"
    assert defaults.evaluation_episodes == 30
    # One episode of parameter lag: a fleet's actors act from copies of the
    # network, and refreshing every episode is what a single actor acting from
    # the learner itself has always done.
    assert defaults.parameter_sync_episodes == 1


def test_a_stacked_burn_in_too_short_for_the_window_is_refused(tmp_path: Path) -> None:
    """Checked before the device is touched, not an hour into collection."""
    with pytest.raises(SystemExit, match="cannot fill a window"):
        arguments(tmp_path, **{"--stacked-burn-in": "2"})


#: A checkpoint cadence and a selection period short enough that a test budget
#: crosses each several times, and deliberately not multiples of each other:
#: the two are independent, as run 5b's 5,000 and 15,000 are.
CHECKPOINT_CADENCE = 70
SELECTION_PERIOD = 100


def numbered_checkpoints(report: dict[str, Any]) -> tuple[dict[str, Any], Path, list[int]]:
    """The arm, its checkpoint directory, and the decisions each file names."""
    arm = report["arm"]
    directory = Path(report["session"]) / arm["run_id"] / "checkpoints"
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
# first's budget, its schedules, its checkpoint cadence and its tracked curve.
# What it does not continue is the replay buffer, which is not persisted - it
# re-warms under the loaded policy, which is the ordinary warm-up rule applied
# again from the resume point.

#: The checkpoint cadence of a resumed run, in decisions.
RESUME_PERIOD = 100


def numbered(run_dir: Path, budget: int, **overrides: str) -> dict[str, Any]:
    """One segment of a run, leaving a numbered checkpoint on every crossing."""
    return session(
        run_dir,
        budget=str(budget),
        settings={"--checkpoint-every-decisions": str(RESUME_PERIOD), **overrides},
    )


def latest_checkpoint(report: dict[str, Any]) -> Path:
    """The resume point of a finished segment, which is what a resume is given."""
    return Path(report["session"]) / report["arm"]["run_id"] / "checkpoints" / "latest.pt"


def resume_from(
    run_dir: Path, checkpoint: Path, budget: int, profile_id: str = PROFILE
) -> Any:
    """The parsed `--resume` of a second segment, as `main` reads it.

    The profile is the one the bridge reported, which is what the checkpoint's
    identity is checked against before anything is brought up.
    """
    return train.resume_point(
        arguments(
            run_dir,
            **{"--budget-decisions": str(budget), "--resume": str(checkpoint)},
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
            parent=run_dir / "resumed-session",
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
    # Exploration and the importance exponent are derived from the counter
    # rather than restored, so they are where a run that never stopped would
    # have them - and not at the start of their schedules.
    assert (
        progress.epsilon
        == config.exploration.epsilon_for(0, spent)
        != config.exploration.epsilon_for(0, 0)
    )
    assert progress.importance_beta == config.beta(spent) != config.beta(0)
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


def test_a_game_time_era_checkpoint_still_loads_for_evaluation(tmp_path: Path) -> None:
    """The refusal is the resume's alone: the file rebuilds into a policy."""
    parent, legacy = legacy_checkpoint(tmp_path, 3)

    assert load(legacy).format_version == 3
    _, identity = checkpoint_policy(
        legacy,
        decision_cadence=parent.identity.decision_cadence.value,
        upgrade_availability=parent.identity.upgrade_availability.value,
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
    # Where the segment picked up, and where its re-warmed buffer let learning
    # restart, both on the same axis.
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
            "--run-dir",
            str(tmp_path / "second"),
        ],
    )

    assert train.main() == 0
    assert "resuming tracked run" not in capsys.readouterr().out


def test_a_run_split_in_two_covers_the_budget_the_whole_run_does(tmp_path: Path) -> None:
    """The end-to-end claim: 300 in one sitting, or 150 and 150, is one run.

    The two are not decision-for-decision identical and cannot be. Episode
    length here depends on what the policy does, and the second segment learns
    from a buffer it re-warmed rather than from the one the first ended with,
    so its episodes are not the episodes a single sitting would have played and
    its counter lands past the period's multiples in different places. What is
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


# --- The BBF recipe: one name, resolved to explicit values on the record -----


def test_the_bbf_recipe_resolves_to_bbfs_values(tmp_path: Path) -> None:
    parsed = train.parse_arguments(
        ["--budget-decisions", "1000", "--run-dir", str(tmp_path), "--recipe", "bbf"]
    )

    assert parsed.recipe == "bbf"
    assert parsed.network_width == 4
    assert parsed.gradient_steps_per_decision == 2.0
    assert (parsed.n_step, parsed.n_step_final, parsed.n_step_anneal_steps) == (10, 3, 10_000)
    assert (parsed.discount_initial, parsed.discount) == (0.97, 0.997)
    assert (parsed.learning_rate, parsed.weight_decay, parsed.adam_eps) == (1e-4, 0.1, 1.5e-4)
    assert parsed.weight_decay_on_vectors is False
    assert parsed.target_ema_decay == 0.995
    assert parsed.reset_every_steps == 40_000
    assert (parsed.exploration, parsed.epsilon_end) == ("uniform", 0.0)
    assert parsed.early_stop_patience_periods == 0


def test_without_a_recipe_the_learner_is_the_one_run_4_trained(tmp_path: Path) -> None:
    defaults = train.parse_arguments(["--budget-decisions", "1000", "--run-dir", str(tmp_path)])

    assert defaults.recipe is None
    assert defaults.network_width == 1
    assert defaults.discount_initial is None
    assert (defaults.weight_decay, defaults.weight_decay_on_vectors) == (1e-5, True)
    assert defaults.adam_eps == 1e-8
    assert defaults.reset_every_steps == 0


def test_the_bbf_recipe_refuses_a_ladder_and_the_plateau_rule(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="epsilon 0"):
        arguments(tmp_path, **{"--recipe": "bbf", "--exploration": "ladder"})
    with pytest.raises(SystemExit, match="plateau"):
        arguments(tmp_path, **{"--recipe": "bbf", "--early-stop-patience-periods": "2"})


def test_a_bbf_session_resets_records_its_recipe_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the entry point with the fake port, with resets short enough to happen."""
    monkeypatch.setitem(train.RECIPES["bbf"], "reset_every_steps", 5)
    monkeypatch.setitem(train.RECIPES["bbf"], "n_step_anneal_steps", 4)

    report = session(tmp_path / "first", budget=TRAINING_BUDGET, settings={"--recipe": "bbf"})

    resolved = report["arm"]["resolved_config"]
    assert resolved["recipe"] == "bbf"
    assert (resolved["network_hidden"], resolved["network_core_hidden"]) == (
        SMALL_NETWORK.hidden * 4,
        SMALL_NETWORK.core_hidden * 4,
    )
    assert (resolved["discount_initial"], resolved["discount"]) == (0.97, 0.997)
    assert (resolved["weight_decay"], resolved["weight_decay_on_vectors"]) == (0.1, False)
    assert resolved["adam_eps"] == 1.5e-4
    assert resolved["reset_every_steps"] == 5
    # The decision budget at the replay ratio the test harness runs at.
    assert resolved["no_resets_after_steps"] == int(int(TRAINING_BUDGET) * 0.2)
    assert (resolved["exploration"], resolved["epsilon_end"]) == ("uniform", 0.0)

    checkpoint = latest_checkpoint(report)
    written = load(checkpoint)
    state = written.backbone_state
    assert state["steps"] > 5 and state["resets"] >= 1
    assert state["cycle_steps"] == state["steps"] - 5 * state["resets"]
    # An evaluation rebuilds the policy from the record alone, optimizer included.
    policy, _ = checkpoint_policy(
        checkpoint,
        decision_cadence=str(written.identity.decision_cadence),
        upgrade_availability=str(written.identity.upgrade_availability),
    )
    assert policy.network_config.hidden == SMALL_NETWORK.hidden * 4

    arm, _ = resumed_arm(tmp_path / "second", checkpoint, budget=600, **{"--recipe": "bbf"})

    resumed = arm.training.backbone.state_dict()
    assert (resumed["steps"], resumed["cycle_steps"], resumed["resets"]) == (
        state["steps"],
        state["cycle_steps"],
        state["resets"],
    )
    assert resumed["reset_seed"] == state["reset_seed"]
