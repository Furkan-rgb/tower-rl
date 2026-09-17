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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import train  # noqa: E402
from fakes.recording_tracker import RecordedRun, RecordingTracker  # noqa: E402
from test_train_entry_point import (  # noqa: E402
    PROFILE,
    SMALL_NETWORK,
    arguments,
    fleet,
)

from tower_rl.ports.experiment_tracker import (  # noqa: E402
    ExperimentTracker,
    NoExperimentTracker,
    TrackedRun,
)

torch.set_num_threads(1)

REPOSITORY = Path(train.__file__).resolve().parents[1]


def tracked_session(run_dir: Path, tracker: RecordingTracker) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        return train.train_session(
            arguments(run_dir, "recurrent-q", **{"--budget-decisions": "120"}),
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


def test_training_needs_no_tracker_at_all(tmp_path: Path) -> None:
    """The untracked path is the same path, not a second one."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        report = train.train_session(
            arguments(tmp_path, "recurrent-q", **{"--budget-decisions": "120"}),
            fleet(),
            profile_id=PROFILE,
            revision="test",
            device=torch.device("cpu"),
        )

    assert report["arms"][0]["decisions"] >= 120
    assert report["arms"][0]["learning_curve"]


def test_only_the_adapter_knows_which_tracker_it_is() -> None:
    """Learning and application code may not import MLflow, directly or not."""
    adapter = Path("src/tower_rl/infrastructure/mlflow_tracker.py")
    importers = {
        path.relative_to(REPOSITORY)
        for path in (REPOSITORY / "src").rglob("*.py")
        if "mlflow" in path.read_text()
    }

    assert importers == {adapter}


def test_an_absent_mlflow_is_refused_rather_than_silently_untracked(
    tmp_path: Path,
) -> None:
    """A tracked run that cannot be tracked must not start; --no-track can."""
    tracked = arguments(tmp_path, "recurrent-q")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "mlflow", None)
        patch.setitem(sys.modules, "mlflow.tracking", None)
        patch.delitem(
            sys.modules, "tower_rl.infrastructure.mlflow_tracker", raising=False
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
    steps = [point.decisions for point in recorded.points]

    assert steps == sorted(steps) and steps[0] > 0
    # A point sits where the budget had actually been spent to, which is past
    # the limit by whatever the last episode needed to reach a classified end.
    assert steps[-1] >= 120
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
            point.metrics["eval_mean_final_wave"] - train.SCRIPTED_REFERENCE, abs=1e-3
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

    assert params["backbone"] == "recurrent-q"
    for key in (
        "seed",
        "budget_decisions",
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
    assert tags["backbone"] == "recurrent-q"
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


def test_tracking_writes_nothing_inside_the_repository(
    recorded: RecordedRun, tmp_path: Path
) -> None:
    for path, _ in recorded.artifacts:
        assert REPOSITORY not in path.resolve().parents

    defaults = train.parse_arguments([])
    store = train.tracking_uri(defaults)
    artifacts = Path(train.artifact_root(defaults))

    assert store == f"sqlite:///{Path.home()}/.local/state/tower-rl/mlflow.db"
    assert REPOSITORY not in Path(store.removeprefix("sqlite:///")).parents
    assert REPOSITORY not in artifacts.parents


def test_the_mlflow_adapter_records_what_it_is_given(tmp_path: Path) -> None:
    """The real adapter against a local file store, when the extra is installed."""
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from tower_rl.infrastructure.mlflow_tracker import MlflowExperimentTracker

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}")
    tracker = MlflowExperimentTracker(
        tracking_uri=uri,
        experiment="test-experiment",
        artifact_root=str(tmp_path / "mlartifacts"),
    )

    run = tracker.start_run(
        name="recurrent-q-test", params={"seed": 0}, tags={"backbone": "recurrent-q"}
    )
    run.log_metrics({"eval_mean_final_wave": 6.5}, decisions=120)
    run.log_artifact(artifact, directory="checkpoints/abc")
    run.finish()

    client = MlflowClient(tracking_uri=uri)
    stored = client.get_run(run.run_id)
    assert stored.data.params["seed"] == "0"
    assert stored.data.tags["backbone"] == "recurrent-q"
    assert stored.info.status == "FINISHED"
    history = client.get_metric_history(run.run_id, "eval_mean_final_wave")
    assert [(item.step, item.value) for item in history] == [(120, 6.5)]
    assert [item.path for item in client.list_artifacts(run.run_id, "checkpoints/abc")] == [
        "checkpoints/abc/summary.json"
    ]
    # Files land under the root the adapter was given, not under the caller's
    # working directory, which for a run started from a checkout is the repository.
    assert (tmp_path / "mlartifacts").is_dir()
