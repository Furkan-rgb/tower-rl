"""What a training session records about itself, and what it costs when it cannot.

No tracking server, no MLflow install and no device: the session under test runs
against the fake port and reports through a recording double, so what is checked
is the calls training makes rather than what MLflow does with them. The one test
that exercises MLflow itself skips when the optional extra is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import train
from fakes.recording_tracker import RecordedRun, RecordingTracker
from test_import_contracts import imported_modules
from test_train_entry_point import (
    PROFILE,
    SMALL_NETWORK,
    arguments,
    fleet,
    session,
)

from tower_rl.environment.project_state import repository_root, state_directory
from tower_rl.experiment.run_identity import SCRIPTED_REFERENCE
from tower_rl.experiment.tracking import (
    ExperimentTracker,
    NoExperimentTracker,
    TrackedRun,
    artifact_root,
    tracking_uri,
)
from tower_rl.learning.exploration import ape_x_floors

REPOSITORY = Path(train.__file__).resolve().parents[1]


def tracked_session(
    run_dir: Path,
    tracker: RecordingTracker,
    **overrides: str,
) -> dict[str, Any]:
    settings = {"--budget-game-seconds": "600", **overrides}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        return train.train_session(
            arguments(run_dir, **settings),
            fleet(),
            profile_id=PROFILE,
            revision="test-revision",
            device=torch.device("cpu"),
            tracker=tracker,
            bridge_version="bridge-test",
        )


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> RecordedRun:
    """One short tracked session, read by several checks."""
    tracker = RecordingTracker()
    tracked_session(tmp_path_factory.mktemp("tracked"), tracker)
    assert len(tracker.runs) == 1, "one tracked run per arm"
    return tracker.runs[0]


def test_the_default_tracker_satisfies_the_port_and_keeps_nothing() -> None:
    tracker = NoExperimentTracker()

    assert isinstance(tracker, ExperimentTracker)
    run = tracker.start_run(name="arm", params={"seed": 0}, tags={"backbone": "x"})
    assert isinstance(run, TrackedRun)
    # Every call the session makes is accepted and answers nothing.
    run.log_metrics({"eval_mean_final_wave": 1.0}, decisions=10)
    run.log_artifact(Path("/nowhere/checkpoint.pt"), directory="checkpoints/abc")
    run.finish()
    assert run.run_id == "untracked"
    # Attaching to an existing run - what a post-hoc selection does - answers
    # the same handle, so a script that was given no run id takes no other path.
    assert tracker.open_run("whatever").run_id == "untracked"


def test_training_needs_no_tracker_at_all(tmp_path: Path) -> None:
    """The untracked path is the same path, not a second one."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        report = train.train_session(
            arguments(tmp_path, **{"--budget-game-seconds": "600"}),
            fleet(),
            profile_id=PROFILE,
            revision="test",
            device=torch.device("cpu"),
        )

    assert report["arm"]["game_seconds"] >= 600
    assert report["arm"]["decisions"] > 0
    assert report["arm"]["learning_curve"]


def test_only_the_adapter_knows_which_tracker_it_is() -> None:
    """Learning and application code may not import MLflow, directly or not."""
    adapter = Path("src/tower_rl/experiment/mlflow_tracking.py")
    sources = list((REPOSITORY / "src").rglob("*.py"))
    importers = {
        path.relative_to(REPOSITORY)
        for path in sources
        if any(name.split(".")[0] == "mlflow" for name in imported_modules(path))
    }

    assert importers == {adapter}
    # One other module may name MLflow without importing it: the store policy
    # beside the port, which decides the URI the adapter is handed.
    assert {path.relative_to(REPOSITORY) for path in sources if "mlflow" in path.read_text()} == {
        adapter,
        Path("src/tower_rl/experiment/tracking.py"),
    }


def test_an_absent_mlflow_is_refused_rather_than_silently_untracked(
    tmp_path: Path,
) -> None:
    """A tracked run that cannot be tracked must not start; --no-track can."""
    tracked = arguments(tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "mlflow", None)
        patch.setitem(sys.modules, "mlflow.tracking", None)
        patch.delitem(
            sys.modules, "tower_rl.experiment.mlflow_tracking", raising=False
        )
        with pytest.raises(SystemExit) as refusal:
            train.build_tracker(tracked)

        assert "--no-track" in str(refusal.value)
        untracked = train.parse_arguments(
            ["--no-track", "--run-dir", str(tmp_path)]
        )
        assert isinstance(train.build_tracker(untracked), NoExperimentTracker)


def test_the_session_reports_in_the_order_a_run_happens(recorded: RecordedRun) -> None:
    calls = recorded.calls

    # The run is opened, its manifest goes up, then every curve point reports a
    # measurement and the checkpoint behind it, and the summary closes the run.
    assert calls[:3] == ["start_run", "log_artifact", "log_metrics"]
    assert calls[-2:] == ["log_artifact", "finish"]
    assert calls.count("log_metrics") == len(recorded.points) >= 2
    # Each evaluation point contributes a measurement and the checkpoint it
    # scored; a collection-curve point is a measurement and nothing else,
    # because it scores episodes that were collected rather than weights.
    evaluations = [
        point for point in recorded.points if "eval_mean_final_wave" in point.metrics
    ]
    assert calls.count("log_artifact") == len(evaluations) + 2


def test_metrics_are_keyed_by_decisions_consumed(recorded: RecordedRun) -> None:
    """The step stays decisions: one monotone axis every series shares.

    The budget is game time, and the game time each point sits at travels as a
    metric beside it (`learner_game_seconds`, `episode_game_seconds_cumulative`)
    - so the same curves can be read in the unit the run was spent in without
    the store holding two step axes.
    """
    steps = [point.decisions for point in recorded.points]

    assert steps == sorted(steps) and steps[0] > 0
    learner = [
        point for point in recorded.points if "learner_game_seconds" in point.metrics
    ]
    assert learner, "the budget position is readable beside the decision axis"
    assert learner[-1].metrics["learner_game_seconds"] >= 600
    evaluations = [
        point for point in recorded.points if "eval_mean_final_wave" in point.metrics
    ]
    windows = [
        point
        for point in recorded.points
        if "collection_mean_final_wave" in point.metrics
    ]
    assert windows, "the collection curve is what the run is read from"
    for point in windows:
        assert point.metrics["collection_mean_final_wave"] > 0
        assert point.metrics["collection_episodes"] == pytest.approx(2.0)
        assert 0.0 <= point.metrics["collection_window_wait_fraction"] <= 1.0
        assert point.metrics["collection_window_purchases_per_episode"] >= 0.0
    for point in evaluations:
        assert point.metrics["eval_mean_final_wave"] > 0
        assert point.metrics["eval_valid_episodes"] + point.metrics[
            "eval_invalid_episodes"
        ] == pytest.approx(2.0)
        assert point.metrics["versus_scripted_reference"] == pytest.approx(
            point.metrics["eval_mean_final_wave"] - SCRIPTED_REFERENCE, abs=1e-3
        )
    # The learning-health signals `learn` already returns, once it has run. The
    # weighted loss and the unweighted TD error are separate keys on purpose:
    # reading the first as the second is what made the first run unreadable.
    last = evaluations[-1].metrics
    assert last["learner_weighted_loss_with_is_weights"] >= 0.0
    assert last["learner_unweighted_mean_absolute_td_error"] >= 0.0
    assert last["learner_gradient_norm"] >= 0.0
    assert "collection_wait_fraction" in last
    assert last["collection_purchases_per_episode"] >= 0.0
    assert last["eval_decisions_per_wave"] > 0.0


def test_the_run_carries_its_configuration_and_the_measured_floors(
    recorded: RecordedRun,
) -> None:
    params = recorded.params

    assert params["backbone"] == "stacked-dqn"
    for key in (
        "seed",
        "budget_game_seconds",
        "gradient_steps_per_decision",
        "sequence_length",
        "burn_in",
        "stride",
        "n_step",
        "discount",
        "learning_rate",
        "epsilon_start",
        "epsilon_end",
        "epsilon_anneal_decisions",
        "priority_alpha",
        "collection_window_episodes",
        "beta_start",
        "beta_end",
        "batch_size",
        "warmup_sequences",
        "frame_game_ms",
        "max_quiet_game_ms",
        "health_change_fraction",
        "evaluation_episodes",
    ):
        assert key in params, key
    # A comparison must be readable from the tracked run alone.
    assert params["reference_scripted"] == 5.57
    assert params["reference_random"] == 5.35
    assert params["reference_wait"] == 1.87
    assert params["reference_source"] == train.REFERENCE_FINAL_WAVES["source"]


def test_provenance_travels_with_the_run(recorded: RecordedRun) -> None:
    tags = recorded.tags

    assert tags["source_revision"] == "test-revision"
    assert tags["bridge_version"] == "bridge-test"
    assert tags["profile_id"] == PROFILE
    assert tags["backbone"] == "stacked-dqn"
    assert tags["actors"] == "1"
    assert tags["session"].startswith("session-")
    assert tags["run_id"] == recorded.name


def test_a_tracked_point_resolves_to_the_exact_checkpoint_file(
    recorded: RecordedRun,
) -> None:
    checkpoints = [
        (path, directory)
        for path, directory in recorded.artifacts
        if directory is not None
    ]

    evaluations = [
        point for point in recorded.points if "eval_mean_final_wave" in point.metrics
    ]
    assert len(checkpoints) == len(evaluations)
    for path, directory in checkpoints:
        assert path.exists() and directory is not None
        assert directory == f"checkpoints/{directory.split('/')[1]}"
        # The directory is the weight digest, which is what a curve point names.
        assert len(directory.split("/")[1]) >= 16


def test_tracking_writes_only_into_the_git_ignored_state_directory(
    recorded: RecordedRun, tmp_path: Path
) -> None:
    """The store and its artifacts live beside the runs, under `state/`.

    They used to be held outside the repository altogether, on the reasoning
    that nothing generated may be committed. The location is now inside the
    checkout and git-ignored instead, so a checkout is the whole project — what
    is committed and the state beside it — and the rule that none of it is
    committed is `.gitignore`'s to keep (`test_state_directory.py`).
    """
    for path, _ in recorded.artifacts:
        assert REPOSITORY not in path.resolve().parents

    defaults = train.parse_arguments([])
    store = tracking_uri(defaults.run_dir)
    artifacts = Path(artifact_root(defaults.run_dir))

    assert defaults.run_dir == state_directory() / "runs"
    assert store == f"sqlite:///{state_directory() / 'mlflow.db'}"
    assert artifacts == state_directory() / "mlartifacts"
    # Not REPOSITORY: from a linked worktree, state/ lives under the main
    # checkout repository_root() resolves to, not the worktree's own tree.
    assert state_directory().parent == repository_root()


def test_the_mlflow_adapter_records_what_it_is_given(tmp_path: Path) -> None:
    """The real adapter against a local file store, when the extra is installed."""
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from tower_rl.experiment.mlflow_tracking import MlflowExperimentTracker

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}")
    tracker = MlflowExperimentTracker(
        tracking_uri=uri,
        experiment="test-experiment",
        artifact_root=str(tmp_path / "mlartifacts"),
    )

    run = tracker.start_run(
        name="stacked-dqn-test", params={"seed": 0}, tags={"backbone": "stacked-dqn"}
    )
    run.log_metrics({"eval_mean_final_wave": 6.5}, decisions=120)
    run.log_artifact(artifact, directory="checkpoints/abc")
    run.finish()

    client = MlflowClient(tracking_uri=uri)
    stored = client.get_run(run.run_id)
    assert stored.data.params["seed"] == "0"
    assert stored.data.tags["backbone"] == "stacked-dqn"
    assert stored.info.status == "FINISHED"
    history = client.get_metric_history(run.run_id, "eval_mean_final_wave")
    assert [(item.step, item.value) for item in history] == [(120, 6.5)]
    assert [item.path for item in client.list_artifacts(run.run_id, "checkpoints/abc")] == [
        "checkpoints/abc/summary.json"
    ]
    # Files land under the root the adapter was given, not under the caller's
    # working directory, which for a run started from a checkout is the repository.
    assert (tmp_path / "mlartifacts").is_dir()


def test_the_mlflow_adapter_reopens_a_run_that_already_exists(tmp_path: Path) -> None:
    """What a post-hoc selection does: attach to a finished run and add to it.

    The greedy score of a checkpoint is taken hours after the run that produced
    it, so `open_run` is the only way those numbers reach the page they are
    about. Exercised against a real store rather than a double, because the one
    thing it does beyond handing out a handle - reading the run back so a
    mistyped id is refused rather than silently swallowed - is the store's
    behaviour, not this project's.
    """
    pytest.importorskip("mlflow")
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    from tower_rl.experiment.mlflow_tracking import MlflowExperimentTracker

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MlflowExperimentTracker(
        tracking_uri=uri,
        experiment="reopen-experiment",
        artifact_root=str(tmp_path / "mlartifacts"),
    )
    training = tracker.start_run(name="stacked-dqn-test", params={"seed": 0}, tags={})
    training.log_metrics({"episode_final_wave": 6.0}, decisions=1000)
    training.finish()

    # Hours later, from another process: the run is addressed by its id alone.
    reopened = MlflowExperimentTracker(
        tracking_uri=uri,
        experiment="reopen-experiment",
        artifact_root=str(tmp_path / "mlartifacts"),
    ).open_run(training.run_id)
    reopened.log_metrics({"greedy_final_wave_iqm": 8.5}, decisions=1000)

    assert reopened.run_id == training.run_id
    client = MlflowClient(tracking_uri=uri)
    greedy = client.get_metric_history(training.run_id, "greedy_final_wave_iqm")
    assert [(item.step, item.value) for item in greedy] == [(1000, 8.5)]
    # On the same run and the same axis as what training reported, which is the
    # whole point: the greedy curve sits above the exploring one.
    exploring = client.get_metric_history(training.run_id, "episode_final_wave")
    assert [(item.step, item.value) for item in exploring] == [(1000, 6.0)]
    # Terminated by training and left terminated: attaching is not restarting.
    assert client.get_run(training.run_id).info.status == "FINISHED"

    # A run id that names nothing is refused where it is typed, not by a metric
    # disappearing into a store nobody reads again.
    with pytest.raises(MlflowException):
        MlflowExperimentTracker(
            tracking_uri=uri,
            experiment="reopen-experiment",
            artifact_root=str(tmp_path / "mlartifacts"),
        ).open_run("0123456789abcdef0123456789abcdef")


#: Every key one collected episode reports. An episode is the tracked unit, so
#: a run is readable at this resolution or it is readable only in aggregate.
EPISODE_KEYS = {
    "episode_final_wave",
    "episode_decisions",
    "episode_game_ms",
    "episode_wait_fraction",
    "episode_purchases",
    "episode_valid",
    "episode_actor",
    "episode_epsilon",
}

#: What the learner reports on the same key, so its curve exists between
#: evaluations rather than only at them.
LEARNER_KEYS = {
    "learner_optimisation_steps",
    "learner_importance_beta",
    "learner_weighted_loss",
    "learner_unweighted_mean_absolute_td_error",
    "learner_gradient_norm",
}

#: Long enough for the arm to play about ten episodes against the fake port.
EPISODE_BUDGET = "1200"


@pytest.fixture(scope="module")
def per_episode(tmp_path_factory: pytest.TempPathFactory) -> tuple[RecordedRun, Any]:
    """A run long enough to have an episode series, with its numbered checkpoints."""
    tracker = RecordingTracker()
    report = tracked_session(
        tmp_path_factory.mktemp("episodes"),
        tracker,
        **{
            "--budget-game-seconds": EPISODE_BUDGET,
            # Numbered checkpoints on, so the artifacts and the metrics can be
            # checked to land on the one run.
            "--checkpoint-every-game-seconds": "400",
            # The episode series is the point here; mid-run evaluation only adds
            # episodes that are not collection.
            "--evaluate-every-episodes": "0",
        },
    )
    assert len(tracker.runs) == 1, "one tracked run per arm"
    return tracker.runs[0], report


def test_every_collected_episode_is_one_tracked_point(
    per_episode: tuple[RecordedRun, Any],
) -> None:
    """The episode is the tracked unit, not only the window that smooths it."""
    recorded, report = per_episode
    collected = report["arm"]["collected_episodes"]
    points = [point for point in recorded.points if set(point.metrics) >= EPISODE_KEYS]

    assert len(collected) >= 5, "the budget buys an episode series worth reading"
    assert len(points) == len(collected), "one point per episode, none repeated"
    for point, episode in zip(points, collected, strict=True):
        assert point.metrics["episode_final_wave"] == float(episode["final_wave"])
        assert point.metrics["episode_decisions"] == float(episode["decisions"])
        assert point.metrics["episode_game_ms"] == pytest.approx(episode["round_ms"])
        assert point.metrics["episode_purchases"] == float(episode["purchases"])
        assert point.metrics["episode_valid"] in (0.0, 1.0)
        assert 0.0 <= point.metrics["episode_wait_fraction"] <= 1.0
        # One actor, so index zero; a fleet's episodes are one series and this
        # is what places each of them on an instance.
        assert point.metrics["episode_actor"] == 0.0
        assert 0.0 <= point.metrics["episode_epsilon"] <= 1.0


def test_an_episode_point_sits_where_that_episode_ended(
    per_episode: tuple[RecordedRun, Any],
) -> None:
    """Keyed by the decisions spent at its own end, so it shares the checkpoints' axis."""
    recorded, report = per_episode
    points = [point for point in recorded.points if set(point.metrics) >= EPISODE_KEYS]

    spent = 0
    expected = []
    for episode in report["arm"]["collected_episodes"]:
        spent += int(episode["decisions"])
        expected.append(spent)

    assert [point.decisions for point in points] == expected


def test_the_learner_reports_between_evaluations_and_not_only_at_them(
    per_episode: tuple[RecordedRun, Any],
) -> None:
    """With no mid-run evaluation the learner would otherwise report once, at the end."""
    recorded, _ = per_episode
    learning = [
        point
        for point in recorded.points
        if set(point.metrics) >= EPISODE_KEYS and set(point.metrics) >= LEARNER_KEYS
    ]

    assert len(learning) >= 2, "a curve, not a point"
    for point in learning:
        assert point.metrics["learner_weighted_loss"] >= 0.0
        assert point.metrics["learner_unweighted_mean_absolute_td_error"] >= 0.0
        assert point.metrics["learner_gradient_norm"] >= 0.0
        assert point.metrics["learner_optimisation_steps"] > 0
    steps = [point.metrics["learner_optimisation_steps"] for point in learning]
    assert steps == sorted(steps)


def test_the_numbered_checkpoints_land_on_the_run_that_reported_the_episodes(
    per_episode: tuple[RecordedRun, Any],
) -> None:
    """One run id carries both, or the candidate cannot be found from the curve."""
    recorded, report = per_episode
    run_dir = Path(report["session"]) / report["arm"]["run_id"]
    numbered = sorted((run_dir / "checkpoints").glob("checkpoint-*.pt"))
    uploaded = [path for path, _ in recorded.artifacts if path.name.startswith("checkpoint-")]

    assert numbered, "the budget crosses the checkpoint period"
    assert uploaded == numbered
    # Filed under the digest of the weights in them, which is what a later
    # reading names a model by.
    directories = [directory for path, directory in recorded.artifacts if path in numbered]
    assert directories and all(
        directory is not None and directory.startswith("checkpoints/")
        for directory in directories
    )


def test_a_laddered_fleet_reports_its_collection_windows_per_actor(
    tmp_path: Path,
) -> None:
    """The pooled series alone cannot be read when the actors explore differently.

    Under a ladder the fleet's actors sit at rates that differ by orders of
    magnitude, so the pooled window is nobody's performance. Each actor's own
    mean goes up beside it, and the near-greedy actors are pooled into the one
    series a readout of what the policy itself reaches can cite.
    """
    tracker = RecordingTracker()
    session(
        tmp_path,
        budget="300",
        actors=2,
        settings={"--exploration": "ladder"},
        tracker=tracker,
    )

    points = [
        point
        for point in tracker.runs[0].points
        if "collection_mean_final_wave" in point.metrics
    ]
    assert points, "the collection curve is what the run is read from"
    for point in points:
        metrics = point.metrics
        # Actor 1 of 2 is the bottom of the ladder and the only near-greedy one,
        # so the near-greedy series is its episodes rather than the fleet's.
        assert metrics["collection_window_near_greedy_episodes"] <= metrics[
            "collection_episodes"
        ]
        per_actor = {
            key for key in metrics if key.startswith("collection_window_mean_final_wave_actor")
        }
        assert per_actor <= {
            "collection_window_mean_final_wave_actor0",
            "collection_window_mean_final_wave_actor1",
        }
        assert per_actor, "a closed window holds at least one actor's episodes"
    # Over the run, both actors and the near-greedy series are all reported.
    keys = {key for point in points for key in point.metrics}
    assert "collection_window_mean_final_wave_actor0" in keys
    assert "collection_window_mean_final_wave_actor1" in keys
    assert "collection_window_near_greedy_mean_final_wave" in keys


def test_a_uniform_fleet_s_near_greedy_window_is_its_pooled_window(
    tmp_path: Path,
) -> None:
    """Every actor draws the one rate, so the two series are the same episodes."""
    tracker = RecordingTracker()
    session(tmp_path, budget="300", actors=2, tracker=tracker)

    points = [
        point
        for point in tracker.runs[0].points
        if "collection_mean_final_wave" in point.metrics
    ]
    assert points
    for point in points:
        metrics = point.metrics
        assert metrics["collection_window_near_greedy_episodes"] == pytest.approx(
            metrics["collection_episodes"]
        )
        assert metrics["collection_window_near_greedy_mean_final_wave"] == pytest.approx(
            metrics["collection_mean_final_wave"]
        )


def test_each_episode_logs_the_rate_the_actor_that_played_it_explored_at(
    tmp_path: Path,
) -> None:
    """Under a ladder the run's own published rate is some other actor's rung.

    The anneal is one decision long here, so every actor is on its rung for the
    whole run and the two rungs of a fleet of two are 0.4 and 0.4 ** 8 - far
    enough apart that an episode credited to the wrong actor's rate is obvious.
    """
    rungs = ape_x_floors(2)
    tracker = RecordingTracker()
    session(
        tmp_path,
        budget="300",
        actors=2,
        settings={"--exploration": "ladder", "--epsilon-anneal-decisions": "1"},
        tracker=tracker,
    )

    episodes = [
        point for point in tracker.runs[0].points if "episode_epsilon" in point.metrics
    ]
    assert episodes, "the episode is the tracked unit"
    seen = set()
    for point in episodes:
        actor = int(point.metrics["episode_actor"])
        assert point.metrics["episode_epsilon"] == pytest.approx(rungs[actor])
        seen.add(actor)
    assert seen == {0, 1}, "both rungs of the ladder collected"
