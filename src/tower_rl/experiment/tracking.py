"""The port a training session records itself through.

A run that leaves no trace cannot be compared with the next one, so every
training session reports what it was configured with, what it measured, and
which files it produced.  Where that lands - a local MLflow store, a different
tool later, nowhere at all - is an adapter's business, and nothing that learns
or trains may know which one it is talking to.

A run is addressed by a handle rather than by an id the caller carries around,
because arms of one comparison are interleaved on the device and their runs are
therefore open at the same time.

Where the store itself lives is decided here too (`tracking_uri`,
`artifact_root`): it is a policy about this project's runs - beside the run
state, never inside the repository - rather than anything MLflow decides.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from tower_rl.environment.project_state import state_directory


@runtime_checkable
class TrackedRun(Protocol):
    """One open run: a single arm of a session, from its start to its end."""

    @property
    def run_id(self) -> str:
        """What the developer opens in the UI to find this arm."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], *, decisions: int) -> None:
        """Record measurements at a point on the decision budget.

        Decisions rather than wall time or step count: decisions are the unit the
        comparison protocol equalises on, so a curve keyed by anything else would
        not be comparable across arms.
        """
        ...

    def log_artifact(self, path: Path, *, directory: str | None = None) -> None:
        """Store a file the run produced, optionally under a named directory."""
        ...

    def finish(self) -> None:
        """Close the run, whether it reached its budget or failed."""
        ...


@runtime_checkable
class ExperimentTracker(Protocol):
    """Where training runs are recorded."""

    @property
    def tracking_uri(self) -> str:
        """The store runs land in, printed so the developer can open it."""
        ...

    def start_run(
        self,
        *,
        name: str,
        params: Mapping[str, object],
        tags: Mapping[str, str],
    ) -> TrackedRun:
        """Open a run for one arm, with everything fixed about it up front."""
        ...

    def open_run(self, run_id: str) -> TrackedRun:
        """Attach to a run that already exists, to add results taken after it.

        The exploration-free measurements of a run's checkpoints are taken hours
        or days after the run itself finished, on a device the run no longer
        owns. They belong on the run they are about - the greedy curve above the
        exploring one - and not on a second run nothing links to it.
        """
        ...


class _UntrackedRun:
    """The handle `NoExperimentTracker` hands out; it keeps nothing."""

    @property
    def run_id(self) -> str:
        return "untracked"

    def log_metrics(self, metrics: Mapping[str, float], *, decisions: int) -> None:
        return None

    def log_artifact(self, path: Path, *, directory: str | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


class NoExperimentTracker:
    """The default: training runs exactly as it does without a tracker.

    Kept beside the protocol rather than in infrastructure because it depends on
    nothing, and because it is what lets tests and any run started without a
    store exercise the same code path a tracked run takes.
    """

    @property
    def tracking_uri(self) -> str:
        return "none"

    def start_run(
        self,
        *,
        name: str,
        params: Mapping[str, object],
        tags: Mapping[str, str],
    ) -> TrackedRun:
        return _UntrackedRun()

    def open_run(self, run_id: str) -> TrackedRun:
        return _UntrackedRun()


def open_tracked_run(run_id: str, *, run_dir: Path, experiment: str) -> TrackedRun:
    """A handle on an existing run, for results taken after it finished.

    The MLflow adapter is imported here rather than at module scope, exactly as
    a training session imports it: a machine with no MLflow can still run
    everything that does not ask to record itself, and the `ImportError` is the
    caller's to turn into whatever a command line should say about it.
    """
    from tower_rl.experiment.mlflow_tracking import MlflowExperimentTracker

    tracker = MlflowExperimentTracker(
        tracking_uri=tracking_uri(run_dir),
        experiment=experiment,
        artifact_root=artifact_root(run_dir),
    )
    return tracker.open_run(run_id)


def tracking_uri(run_dir: Path) -> str:
    """Where runs are recorded: beside the run state, never in the repository.

    SQLite rather than a directory of files because MLflow 3 refuses the
    filesystem backend, and local either way: nothing leaves this machine.
    """
    override = os.environ.get("MLFLOW_TRACKING_URI")
    if override:
        return override
    return f"sqlite:///{run_dir.parent / 'mlflow.db'}"


def artifact_root(run_dir: Path) -> str:
    """Where tracked files land, beside the store and outside the repository."""
    return str(run_dir.parent / "mlartifacts")


def add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    """The `--mlflow-run`/`--run-dir`/`--experiment` options a tracked command shares.

    `select_checkpoint.py` and `report_arms.py` both add a set B's or a
    selection's results onto an existing training run's page, so they take
    this by the same three flags rather than each defining its own copy.
    """
    parser.add_argument(
        "--mlflow-run",
        default=None,
        help=(
            "an existing tracked run id to add these results to, so the greedy "
            "curve lands on the page of the training run it is about"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=state_directory() / "runs",
        help="where the tracking store lives; only read with --mlflow-run",
    )
    parser.add_argument(
        "--experiment",
        default="tower-rl-training",
        help="the MLflow experiment the run belongs to; only read with --mlflow-run",
    )


def tracked_run(arguments: argparse.Namespace) -> TrackedRun | None:
    """The run these results are added to, or nothing if none was named."""
    if not arguments.mlflow_run:
        return None
    try:
        return open_tracked_run(
            arguments.mlflow_run,
            run_dir=arguments.run_dir,
            experiment=arguments.experiment,
        )
    except ImportError as missing:
        raise SystemExit(
            f"--mlflow-run needs MLflow installed ({missing}). "
            "Install it with `uv sync --extra tracking`, or drop the flag."
        ) from missing


__all__ = [
    "ExperimentTracker",
    "NoExperimentTracker",
    "TrackedRun",
    "add_tracking_arguments",
    "artifact_root",
    "open_tracked_run",
    "tracked_run",
    "tracking_uri",
]
