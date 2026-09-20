"""The two stages of the evaluation protocol, over directories on disk.

Neither script here touches a device: both read what `run_actors.py` already
left behind. The synthetic directories below are that shape exactly - one JSON
record per actor, each naming the arm that played it - so what is under test is
the selection and the reporting, not the collection.

The last test is the whole chain end to end against the fake port: train, take
the numbered checkpoints, play each one, select among them, report the selection
against the floors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import report_arms
import run_episodes
import select_checkpoint
import torch
import train
from fakes.fake_run_port import FakeRunPort
from fakes.recording_tracker import RecordedRun

from tower_rl.environment.run_environment import (
    CadenceConfig,
    DecisionCadence,
    InstrumentedRunEnvironment,
    UpgradeAvailability,
)
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.checkpoint import (
    Checkpoint,
    CheckpointIdentity,
    TrainingProgress,
    save,
)
from tower_rl.learning.evaluator import evaluate
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.policies import CheapestFirstPolicy, RandomPolicy

PROFILE = "fake-profile-v1"

SMALL_NETWORK = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)


# ---------------------------------------------------------------------------
# Synthetic evaluation directories, in the shape run_actors.py writes
# ---------------------------------------------------------------------------


def episode(final_wave: int, decisions: int) -> dict[str, Any]:
    """One valid episode record, with the per-wave rows the comparison reads."""
    return {
        "episode_index": 0,
        "valid": True,
        "final_wave": final_wave,
        "decisions": decisions,
        "purchases": decisions // 2,
        "frames": decisions * 6,
        "budgeted_game_ms": decisions * 100.0,
        "round_ms": decisions * 100.0,
        "advance_wall_seconds": decisions * 0.01,
        "elapsed_wall_seconds": decisions * 0.02,
        "invalid_reasons": (),
        "termination_detail": (),
        "advances_cut_short": 0,
        "recovered_transients": 0,
        "starting_wave": 1,
        "waves": [
            {
                "wave": wave,
                "completed": wave < final_wave,
                "game_ms": 1000.0 + wave * 50.0,
                "decisions": max(1, decisions // final_wave),
                "health_fraction": max(0.0, 1.0 - wave * 0.05),
                "cash_log": 3.0 + wave * 0.1,
            }
            for wave in range(1, final_wave + 1)
        ],
    }


def actor_record(waves: list[int], identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": "StackedDqnBackbone",
        "policy_identity": identity,
        "valid_episodes": len(waves),
        "invalid_episodes": 0,
        "episodes": [episode(wave, 30 + wave * 3) for wave in waves],
    }


def evaluation_directory(
    root: Path, name: str, per_actor: dict[str, list[int]], identity: dict[str, Any]
) -> Path:
    """One arm's evaluation: one record per actor, all naming the same arm."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    for actor, waves in per_actor.items():
        (directory / f"{actor}.json").write_text(
            json.dumps(actor_record(waves, identity), indent=2)
        )
    return directory


#: The run every synthetic checkpoint below belongs to. A run directory is named
#: for its run id, which is what ties a record to the run that produced it.
RUN_ID = "stacked-dqn-20260101-000000-abcdef"


def checkpoint_identity(
    path: Path, *, run_id: str = RUN_ID, digest: str = "abc123def456"
) -> dict[str, Any]:
    """What `run_episodes.policy_from` writes into every record of a checkpoint."""
    return {
        "name": path.stem,
        "checkpoint_path": str(path),
        "checkpoint_identity": digest,
        "run_id": run_id,
    }


#: Decisions a synthetic checkpoint records per game second of its name. The
#: two units are related by how a run happened to play, so the selection reads
#: the decisions out of the file rather than deriving them from the name; this
#: is only what these fixtures happen to have played at.
DECISIONS_PER_GAME_SECOND = 2


def write_checkpoint_file(path: Path, game_seconds: int) -> None:
    """A real, readable checkpoint: the selection reads the decisions out of it."""
    save(
        Checkpoint(
            identity=CheckpointIdentity(
                run_id=RUN_ID,
                backbone="stacked-dqn",
                profile_id=PROFILE,
                observation_schema="observation-v1",
                action_schema="run-action-v1",
                reward_schema="reward-v1",
                source_revision="test",
            ),
            progress=TrainingProgress(
                environment_decisions=game_seconds * DECISIONS_PER_GAME_SECOND,
                environment_game_ms=game_seconds * 1000.0,
            ),
            backbone_state={"weight": torch.ones(1)},
        ),
        path,
    )


def run_with_checkpoints(
    root: Path, game_seconds: list[int], *, run_id: str = RUN_ID
) -> tuple[Path, list[Path]]:
    """A run directory holding numbered checkpoints, named by their game seconds."""
    run = root / run_id
    (run / "checkpoints").mkdir(parents=True)
    paths = []
    for spent in game_seconds:
        path = run / "checkpoints" / f"checkpoint-gs{spent:07d}.pt"
        write_checkpoint_file(path, spent)
        paths.append(path)
    return run, paths


def invoke(module: Any, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    return int(module.main())


# ---------------------------------------------------------------------------
# Stage one: selection
# ---------------------------------------------------------------------------


def test_the_selection_is_the_checkpoint_with_the_highest_interquartile_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run, checkpoints = run_with_checkpoints(tmp_path, [100, 200, 300])
    weak, best, late = checkpoints
    directories = [
        evaluation_directory(
            tmp_path / "evals",
            "weak",
            {"emulator-5556": [4, 5, 4, 5], "emulator-5558": [5, 4, 5, 4]},
            checkpoint_identity(weak),
        ),
        evaluation_directory(
            tmp_path / "evals",
            "best",
            {"emulator-5556": [11, 12, 11, 12], "emulator-5558": [12, 11, 12, 11]},
            checkpoint_identity(best),
        ),
        evaluation_directory(
            tmp_path / "evals",
            "late",
            {"emulator-5556": [7, 8, 7, 8], "emulator-5558": [8, 7, 8, 7]},
            checkpoint_identity(late),
        ),
    ]
    output = tmp_path / "selection.json"

    code = invoke(
        select_checkpoint,
        [str(run), *[str(item) for item in directories], "--resamples", "200",
         "--output", str(output)],
        monkeypatch,
    )

    assert code == 0
    printed = capsys.readouterr().out
    # Every candidate is in the table, on both reported statistics.
    for path in checkpoints:
        assert printed.count(path.name) >= 2
    assert "final_wave" in printed and "decisions" in printed
    assert f"selected {best}" in printed

    report = json.loads(output.read_text())
    assert report["selection"]["checkpoint"] == str(best)
    assert report["selection"]["selected_on"] == "final_wave"
    assert report["selection"]["game_seconds"] == 200
    # The decisions behind it travel beside the budget position, read out of
    # the file rather than off its name.
    assert report["selection"]["decisions"] == 200 * DECISIONS_PER_GAME_SECOND
    assert len(report["candidates"]) == 3
    # The table is in the order the run produced them, not the command line's.
    assert [item["game_seconds"] for item in report["candidates"]] == [100, 200, 300]
    # Three separated arms: nothing contests the winner at this sample.
    assert report["selection_contested_by"] == []


def test_a_selection_among_checkpoints_it_cannot_separate_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The selection is still made; a difference the sample cannot resolve is not a finding."""
    run, checkpoints = run_with_checkpoints(tmp_path, [100, 200])
    first, second = checkpoints
    directories = [
        evaluation_directory(
            tmp_path / "evals",
            "first",
            {"emulator-5556": [6, 7, 6, 7], "emulator-5558": [7, 6, 7, 6]},
            checkpoint_identity(first),
        ),
        evaluation_directory(
            tmp_path / "evals",
            "second",
            {"emulator-5556": [6, 7, 7, 7], "emulator-5558": [7, 6, 7, 7]},
            checkpoint_identity(second),
        ),
    ]

    invoke(
        select_checkpoint,
        [str(run), *[str(item) for item in directories], "--resamples", "200",
         "--output", str(tmp_path / "selection.json")],
        monkeypatch,
    )

    printed = capsys.readouterr().out
    assert "could not separate" in printed


def test_an_evaluation_of_another_runs_checkpoint_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = run_with_checkpoints(tmp_path, [100])
    other = tmp_path / "elsewhere" / "checkpoint-gs0900000.pt"
    other.parent.mkdir(parents=True)
    write_checkpoint_file(other, 900_000)
    directory = evaluation_directory(
        tmp_path / "evals", "foreign", {"a": [5, 6]}, checkpoint_identity(other)
    )

    with pytest.raises(SystemExit, match="not a numbered checkpoint this run left"):
        invoke(select_checkpoint, [str(run), str(directory)], monkeypatch)


def test_two_runs_at_the_same_period_do_not_borrow_each_others_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file names collide; the run id is what says whose checkpoint it is.

    Two runs trained at the same `--checkpoint-every-game-seconds` leave files
    called exactly the same thing. Identifying a candidate by its file name
    would accept the other run's evaluation here, and the model reported as this
    run's work would be a model it never produced.
    """
    mine, my_checkpoints = run_with_checkpoints(tmp_path / "a", [100, 200])
    theirs, their_checkpoints = run_with_checkpoints(
        tmp_path / "b", [100, 200], run_id="stacked-dqn-20260202-000000-fedcba"
    )
    # The same names, in two runs.
    assert [path.name for path in my_checkpoints] == [path.name for path in their_checkpoints]

    borrowed = evaluation_directory(
        tmp_path / "evals",
        "borrowed",
        {"emulator-5556": [9, 9, 9, 9]},
        checkpoint_identity(
            their_checkpoints[1], run_id=theirs.name, digest="ffffffffffff"
        ),
    )

    with pytest.raises(SystemExit, match="played a checkpoint of run"):
        invoke(select_checkpoint, [str(mine), str(borrowed)], monkeypatch)


def test_candidates_that_disagree_about_their_run_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One selection is made over one run's checkpoints, or over nothing."""
    run, checkpoints = run_with_checkpoints(tmp_path, [100, 200])
    directories = [
        evaluation_directory(
            tmp_path / "evals", "first", {"a": [5, 6]}, checkpoint_identity(checkpoints[0])
        ),
        evaluation_directory(
            tmp_path / "evals",
            "second",
            {"a": [7, 8]},
            # The same run id, a different identity: different schemas or a
            # different source revision behind the same run directory.
            checkpoint_identity(checkpoints[1], digest="0123456789ab"),
        ),
    ]

    with pytest.raises(SystemExit, match="do not agree on what produced them"):
        invoke(select_checkpoint, [str(run), *[str(item) for item in directories]], monkeypatch)


def test_candidates_are_ordered_by_game_seconds_not_by_how_their_names_sort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The zero padding only orders correctly while every name is the same width.

    A ten-million-game-second run is about three days of collection on the
    fleet, not a hypothetical, and at that point the widths mix:
    `checkpoint-gs10000000` sorts before `checkpoint-gs9000000` as text and
    after it as a number.
    """
    run, checkpoints = run_with_checkpoints(tmp_path, [9_000_000, 10_000_000])
    nine, ten = checkpoints
    assert sorted(path.name for path in checkpoints)[0] == ten.name, (
        "the premise: as text, the later checkpoint sorts first"
    )
    directories = [
        evaluation_directory(
            tmp_path / "evals", "ten", {"a": [8, 9, 8, 9]}, checkpoint_identity(ten)
        ),
        evaluation_directory(
            tmp_path / "evals", "nine", {"a": [4, 5, 4, 5]}, checkpoint_identity(nine)
        ),
    ]
    output = tmp_path / "selection.json"

    invoke(
        select_checkpoint,
        [str(run), *[str(item) for item in directories], "--resamples", "200",
         "--output", str(output)],
        monkeypatch,
    )

    report = json.loads(output.read_text())
    assert [item["game_seconds"] for item in report["candidates"]] == [
        9_000_000,
        10_000_000,
    ]
    # The printed table reads as a curve, in the order the run produced them.
    printed = capsys.readouterr().out
    assert printed.index(nine.name) < printed.index(ten.name)


def test_an_evaluation_of_a_non_checkpoint_arm_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = run_with_checkpoints(tmp_path, [100])
    directory = evaluation_directory(
        tmp_path / "evals", "scripted", {"a": [5, 6]}, {"name": "scripted"}
    )

    with pytest.raises(SystemExit, match="did not play a checkpoint"):
        invoke(select_checkpoint, [str(run), str(directory)], monkeypatch)


def test_a_run_without_numbered_checkpoints_has_nothing_to_choose_among(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)

    with pytest.raises(SystemExit, match="--checkpoint-every-game-seconds"):
        invoke(select_checkpoint, [str(run), str(tmp_path)], monkeypatch)


def test_a_directory_that_mixes_two_arms_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One directory holds one arm; pooling two would report a policy that never played."""
    run, checkpoints = run_with_checkpoints(tmp_path, [100, 200])
    directory = evaluation_directory(
        tmp_path / "evals", "mixed", {"a": [5, 6]}, checkpoint_identity(checkpoints[0])
    )
    (directory / "b.json").write_text(
        json.dumps(actor_record([7, 8], checkpoint_identity(checkpoints[1])))
    )

    with pytest.raises(SystemExit, match="mixes arms"):
        invoke(select_checkpoint, [str(run), str(directory)], monkeypatch)


# ---------------------------------------------------------------------------
# Stage two: reporting the selection against the floors
# ---------------------------------------------------------------------------


def arm_directories(tmp_path: Path) -> dict[str, Path]:
    return {
        "random": evaluation_directory(
            tmp_path / "arms",
            "random",
            {"emulator-5556": [3, 4, 3, 4, 5], "emulator-5558": [4, 3, 4, 3, 4]},
            {"name": "random"},
        ),
        "scripted": evaluation_directory(
            tmp_path / "arms",
            "scripted",
            {"emulator-5556": [5, 6, 5, 6, 6], "emulator-5558": [6, 5, 6, 5, 6]},
            {"name": "scripted"},
        ),
        "stacked-dqn": evaluation_directory(
            tmp_path / "arms",
            "stacked-dqn",
            {"emulator-5556": [10, 11, 10, 12, 11], "emulator-5558": [11, 10, 12, 11, 10]},
            # The full identity a checkpoint arm's records carry, which is what
            # `--selection` is checked against.
            checkpoint_identity(
                Path(f"/runs/{RUN_ID}/checkpoints/checkpoint-0000300.pt")
            ),
        ),
    }


def test_the_report_prints_every_section_for_every_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arms = arm_directories(tmp_path)
    output = tmp_path / "arms.json"

    code = invoke(
        report_arms,
        [
            *[f"{name}={path}" for name, path in arms.items()],
            "--resamples", "200",
            "--output-directory", str(tmp_path / "pooled"),
            "--output", str(output),
        ],
        monkeypatch,
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "interquartile mean, stratified by actor" in printed
    assert "pairwise difference in final wave IQM, stratified by actor" in printed
    assert "secondary, on the mean" in printed
    assert "per wave index" in printed
    for name in arms:
        assert name in printed

    report = json.loads(output.read_text())
    assert sorted(report["arms"]) == ["random", "scripted", "stacked-dqn"]
    for entry in report["arms"].values():
        assert entry["valid_episodes"] == 10
        assert entry["final_wave"]["low"] <= entry["final_wave"]["iqm"]
        assert entry["final_wave"]["iqm"] <= entry["final_wave"]["high"]
    # The learned arm is the highest of the three, and is separated from both.
    assert report["arms"]["stacked-dqn"]["final_wave"]["low"] > (
        report["arms"]["scripted"]["final_wave"]["high"]
    )
    assert len(report["differences"]) == 3
    # The pre-registered statistic is the IQM difference; the mean difference is
    # kept beside it as secondary. Three arms this far apart separate on both.
    assert all(item["iqm_separated"] for item in report["differences"])
    assert all(item["separated"] for item in report["differences"])
    for item in report["differences"]:
        low, high = item["iqm_interval"]
        assert low <= item["iqm_difference"] <= high
    assert len(report["per_wave"]) == 3
    # The pooled files the per-wave comparison was run over are kept beside it.
    assert sorted(p.name for p in (tmp_path / "pooled").glob("*.json")) == [
        "random.json",
        "scripted.json",
        "stacked-dqn.json",
    ]


def test_an_arm_name_that_is_not_a_name_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name becomes a file under --output-directory and a metric key."""
    arms = arm_directories(tmp_path)
    pooled = tmp_path / "pooled"

    for bad in ("../escape", "two words", "stacked/dqn", "arm:one"):
        with pytest.raises(SystemExit, match="may hold only letters"):
            invoke(
                report_arms,
                [f"random={arms['random']}", f"{bad}={arms['scripted']}",
                 "--output-directory", str(pooled)],
                monkeypatch,
            )
    assert not pooled.exists(), "refused before a directory was made for it"

    # The names the protocol actually uses are all accepted.
    for good in ("random", "scripted", "stacked-dqn", "stacked_dqn", "arm2"):
        assert report_arms.ARM_NAME.fullmatch(good)


def selection_file(tmp_path: Path, **overrides: Any) -> Path:
    """A selection.json of the shape `select_checkpoint.py` writes."""
    selection = {
        "run_id": RUN_ID,
        "checkpoint": f"/runs/{RUN_ID}/checkpoints/checkpoint-0000300.pt",
        "decisions": 300,
        "checkpoint_identity": "abc123def456",
        "selected_on": "final_wave",
        "iqm": 10.5,
        "interval": [9.0, 12.0],
        "selected_at": "2026-09-18T00:00:00+00:00",
    }
    selection.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection, indent=2))
    return path


def test_the_report_accepts_a_set_b_that_played_the_selected_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The learned arm here carries exactly the identity set A chose."""
    arms = arm_directories(tmp_path)
    output = tmp_path / "arms.json"

    code = invoke(
        report_arms,
        [
            *[f"{name}={path}" for name, path in arms.items()],
            "--selection", str(selection_file(tmp_path)),
            "--resamples", "200",
            "--output-directory", str(tmp_path / "pooled"),
            "--output", str(output),
        ],
        monkeypatch,
    )

    assert code == 0
    assert json.loads(output.read_text())["selection"]["decisions"] == 300


def test_a_set_b_that_played_another_model_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale directory, or the right run's wrong checkpoint, is the hazard."""
    arms = arm_directories(tmp_path)
    common = [
        *[f"{name}={path}" for name, path in arms.items()],
        "--resamples", "200",
        "--output-directory", str(tmp_path / "pooled"),
        "--output", str(tmp_path / "arms.json"),
    ]

    # Another run entirely: the identity hash does not match.
    other_run = selection_file(
        tmp_path / "a", run_id="stacked-dqn-20260202-000000-fedcba",
        checkpoint_identity="ffffffffffff",
    )
    with pytest.raises(SystemExit, match="played a checkpoint of another run"):
        invoke(report_arms, [*common, "--selection", str(other_run)], monkeypatch)

    # The right run, a different checkpoint of it: the hash cannot tell those
    # apart, so the name of the checkpoint is what does.
    other_checkpoint = selection_file(
        tmp_path / "b",
        checkpoint=f"/runs/{RUN_ID}/checkpoints/checkpoint-0000100.pt",
        decisions=100,
    )
    with pytest.raises(SystemExit, match="of the same run"):
        invoke(report_arms, [*common, "--selection", str(other_checkpoint)], monkeypatch)


def test_a_report_with_no_checkpoint_arm_cannot_satisfy_a_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arms = arm_directories(tmp_path)

    with pytest.raises(SystemExit, match="no arm here played a checkpoint"):
        invoke(
            report_arms,
            [
                f"random={arms['random']}", f"scripted={arms['scripted']}",
                "--selection", str(selection_file(tmp_path)),
                "--resamples", "200",
                "--output-directory", str(tmp_path / "pooled"),
                "--output", str(tmp_path / "arms.json"),
            ],
            monkeypatch,
        )


def test_arms_are_named_and_a_report_needs_two_of_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arms = arm_directories(tmp_path)
    with pytest.raises(SystemExit, match="expected name=<directory>"):
        invoke(report_arms, [str(arms["random"])], monkeypatch)
    with pytest.raises(SystemExit, match="at least two arms"):
        invoke(report_arms, [f"random={arms['random']}"], monkeypatch)
    with pytest.raises(SystemExit, match="named twice"):
        invoke(
            report_arms,
            [f"random={arms['random']}", f"random={arms['scripted']}"],
            monkeypatch,
        )
    with pytest.raises(SystemExit, match="no directory at"):
        invoke(
            report_arms,
            [f"random={arms['random']}", f"scripted={tmp_path / 'absent'}"],
            monkeypatch,
        )


# ---------------------------------------------------------------------------
# The whole protocol, against the fake port
# ---------------------------------------------------------------------------


def environment() -> InstrumentedRunEnvironment:
    return InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=2.0),
        builder=RunStateBuilder(profile_id=PROFILE),
        cadence=CadenceConfig(frame_game_ms=100.0, max_quiet_game_ms=4000),
    )


def play(selector: str, directory: Path, *, actors: int = 2, episodes: int = 2) -> Path:
    """What `run_actors.py` does, minus the emulators: one record per actor.

    The policy is resolved through the runner's own selector and the episodes go
    through the evaluator, so this is the collection path a device run takes with
    the device removed - the fake port never reaches replay, only evaluation.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(actors):
        policy, identity = run_episodes.policy_from(selector)
        report = evaluate(environment(), policy, episodes=episodes, profile_id=PROFILE)
        record = run_episodes.actor_record(
            report,
            identity,
            frame_game_ms=100.0,
            max_quiet_game_ms=4000,
            decision_cadence=DecisionCadence.CHOICE_POINTS,
            upgrade_availability=UpgradeAvailability.IMAGE,
            wall_seconds=60.0,
        )
        (directory / f"fake-{index}.json").write_text(json.dumps(record, indent=2))
    return directory


def test_train_then_select_then_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Train, evaluate each numbered checkpoint, select one, report it on a fresh set."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        session = train.train_session(
            train.parse_arguments(
                [
                    "--budget-game-seconds", "1200",
                    "--block-game-seconds", "200",
                    "--checkpoint-every-game-seconds", "400",
                    "--batch-size", "2",
                    "--gradient-steps-per-decision", "0.2",
                    "--warmup-sequences", "2",
                    "--sequence-length", "6",
                    "--stacked-burn-in", "3",
                    "--history-length", "4",
                    "--replay-capacity", "64",
                    "--evaluate-every-episodes", "0",
                    "--evaluation-episodes", "1",
                    "--collection-window-episodes", "2",
                    "--serial", "fake-0",
                    "--max-quiet-game-ms", "4000",
                    "--run-dir", str(tmp_path / "runs"),
                ]
            ),
            [train.ActorInstance(serial="fake-0", environment=environment())],
            profile_id=PROFILE,
            revision="test",
            device=torch.device("cpu"),
        )

    run = Path(session["session"]) / session["arm"]["run_id"]
    checkpoints = sorted((run / "checkpoints").glob("checkpoint-gs*.pt"))
    assert len(checkpoints) >= 2, "the budget crosses the period more than once"

    # Set A: every candidate, each in its own directory.
    set_a = [
        play(f"checkpoint:{path}", tmp_path / "set-a" / path.stem) for path in checkpoints
    ]
    selection = tmp_path / "selection.json"
    assert (
        invoke(
            select_checkpoint,
            [str(run), *[str(item) for item in set_a], "--resamples", "200",
             "--output", str(selection)],
            monkeypatch,
        )
        == 0
    )
    chosen = Path(json.loads(selection.read_text())["selection"]["checkpoint"])
    assert chosen in checkpoints
    # The same decision, written beside the run for the next command to read.
    beside = json.loads((run / "selection.json").read_text())
    assert Path(beside["checkpoint"]) == chosen
    assert beside["run_id"] == run.name

    # Set B: the selection, played again into a directory of its own, beside the
    # floors it has to be read against.
    arms = {
        "random": play("random", tmp_path / "set-b" / "random"),
        "scripted": play("scripted", tmp_path / "set-b" / "scripted"),
        "stacked-dqn": play(f"checkpoint:{chosen}", tmp_path / "set-b" / "stacked-dqn"),
    }
    report = tmp_path / "arms.json"
    assert (
        invoke(
            report_arms,
            [
                *[f"{name}={path}" for name, path in arms.items()],
                "--resamples", "200",
                "--output-directory", str(tmp_path / "pooled"),
                "--output", str(report),
            ],
            monkeypatch,
        )
        == 0
    )

    printed = capsys.readouterr().out
    assert "selected" in printed and "per wave index" in printed
    scored = json.loads(report.read_text())
    assert sorted(scored["arms"]) == ["random", "scripted", "stacked-dqn"]
    # The learned arm carries the checkpoint it was; the floors carry their name.
    assert scored["arms"]["stacked-dqn"]["policy_identity"]["name"] == chosen.stem
    assert scored["arms"]["scripted"]["policy_identity"] == {"name": "scripted"}
    assert len(scored["differences"]) == 3


# ---------------------------------------------------------------------------
# Results taken after the run, on the run they are about
# ---------------------------------------------------------------------------


def attached(module: Any, monkeypatch: pytest.MonkeyPatch) -> RecordedRun:
    """A handle on an existing tracked run, recorded instead of sent anywhere."""
    recorded = RecordedRun(name="stacked-dqn-under-test", params={}, tags={})
    monkeypatch.setattr(
        module,
        "open_tracked_run",
        lambda run_id, *, run_dir, experiment: recorded,
    )
    return recorded


def test_the_greedy_curve_is_logged_onto_the_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On both axes: the run's decision axis, and the budget axis in game seconds."""
    run, checkpoints = run_with_checkpoints(tmp_path, [100, 200])
    weak, best = checkpoints
    directories = [
        evaluation_directory(
            tmp_path / "evals",
            "weak",
            {"emulator-5556": [4, 5, 4, 5], "emulator-5558": [5, 4, 5, 4]},
            checkpoint_identity(weak),
        ),
        evaluation_directory(
            tmp_path / "evals",
            "best",
            {"emulator-5556": [11, 12, 11, 12], "emulator-5558": [12, 11, 12, 11]},
            checkpoint_identity(best),
        ),
    ]
    recorded = attached(select_checkpoint, monkeypatch)

    invoke(
        select_checkpoint,
        [str(run), *[str(item) for item in directories], "--resamples", "200",
         "--mlflow-run", "abc123", "--output", str(tmp_path / "selection.json")],
        monkeypatch,
    )

    # Each candidate twice: once on the decisions the store's step axis is in,
    # and once on the game seconds the run was actually budgeted in.
    on_decisions = [
        point for point in recorded.points if "greedy_final_wave_iqm" in point.metrics
    ]
    on_game_seconds = [
        point
        for point in recorded.points
        if "greedy_final_wave_iqm_by_game_seconds" in point.metrics
    ]
    assert [point.decisions for point in on_decisions] == [
        100 * DECISIONS_PER_GAME_SECOND,
        200 * DECISIONS_PER_GAME_SECOND,
    ]
    assert [point.decisions for point in on_game_seconds] == [100, 200]
    for point in on_decisions:
        assert set(point.metrics) == {
            "greedy_final_wave_iqm",
            "greedy_final_wave_ci_low",
            "greedy_final_wave_ci_high",
        }
        assert (
            point.metrics["greedy_final_wave_ci_low"]
            <= point.metrics["greedy_final_wave_iqm"]
            <= point.metrics["greedy_final_wave_ci_high"]
        )
    # The same numbers on both axes, under their own keys.
    assert [point.metrics["greedy_final_wave_iqm"] for point in on_decisions] == [
        point.metrics["greedy_final_wave_iqm_by_game_seconds"]
        for point in on_game_seconds
    ]
    assert on_decisions[1].metrics["greedy_final_wave_iqm"] > (
        on_decisions[0].metrics["greedy_final_wave_iqm"]
    )


def test_the_set_b_results_are_logged_onto_the_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arms = arm_directories(tmp_path)
    recorded = attached(report_arms, monkeypatch)

    invoke(
        report_arms,
        [
            *[f"{name}={path}" for name, path in arms.items()],
            "--resamples", "200",
            "--mlflow-run", "abc123",
            "--output-directory", str(tmp_path / "pooled"),
            "--output", str(tmp_path / "arms.json"),
        ],
        monkeypatch,
    )

    logged = {key: value for point in recorded.points for key, value in point.metrics.items()}
    for name in arms:
        low = logged[f"report_{name}_final_wave_ci_low"]
        iqm = logged[f"report_{name}_final_wave_iqm"]
        high = logged[f"report_{name}_final_wave_ci_high"]
        assert low <= iqm <= high
    # The pre-registered statistic, per pair, beside the per-arm ones: this is
    # what the decision rule is read off.
    for pair in ("random_minus_scripted", "random_minus_stacked-dqn",
                 "scripted_minus_stacked-dqn"):
        low = logged[f"report_{pair}_final_wave_iqm_ci_low"]
        difference = logged[f"report_{pair}_final_wave_iqm_diff"]
        high = logged[f"report_{pair}_final_wave_iqm_ci_high"]
        assert low <= difference <= high
    # The model is the stronger arm, so both of its differences are negative.
    assert logged["report_scripted_minus_stacked-dqn_final_wave_iqm_diff"] < 0.0
    assert logged["report_random_minus_stacked-dqn_final_wave_iqm_diff"] < 0.0
    # One measurement about a finished run, not a point on its budget.
    assert {point.decisions for point in recorded.points} == {0}
    assert logged["report_stacked-dqn_final_wave_iqm"] > logged["report_scripted_final_wave_iqm"]


def test_nothing_is_tracked_unless_a_run_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scripts read directories; recording is an extra the caller asks for."""
    arms = arm_directories(tmp_path)

    def refuse(run_id: str, *, run_dir: Path, experiment: str) -> None:
        raise AssertionError("no run was named; nothing may be opened")

    monkeypatch.setattr(report_arms, "open_tracked_run", refuse)

    assert (
        invoke(
            report_arms,
            [
                *[f"{name}={path}" for name, path in arms.items()],
                "--resamples", "200",
                "--output-directory", str(tmp_path / "pooled"),
                "--output", str(tmp_path / "arms.json"),
            ],
            monkeypatch,
        )
        == 0
    )


def test_the_floors_go_through_the_same_selector_as_a_checkpoint(tmp_path: Path) -> None:
    """One protocol: the floors and a checkpoint are played by identical machinery."""
    for selector, expected in (("random", RandomPolicy), ("scripted", CheapestFirstPolicy)):
        policy, identity = run_episodes.policy_from(selector)
        assert isinstance(policy, expected)
        assert identity == {"name": selector}
