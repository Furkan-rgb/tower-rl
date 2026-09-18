"""The `ExperimentTracker` adapter backed by a local MLflow store.

MLflow was chosen because it runs entirely on this machine against a local
SQLite backend - no account, no cloud, nothing to reach for at three in the
morning - while still giving run comparison and model lineage over months of
runs.  It is the only module in the project that imports it: everything else
sees the port, so replacing MLflow is one class.

The store is SQLite rather than a directory of files because MLflow 3 put the
filesystem backend into maintenance mode and refuses it outright.  Artifacts are
still plain files, under an explicitly given root: MLflow's default would put
them in `./mlruns` relative to the working directory, which for a run started
from a checkout is inside the repository.

Runs are driven through `MlflowClient` rather than the module-level
`mlflow.start_run` API on purpose.  Arms of one comparison are interleaved on
the device, so several runs are open at once and a global "active run" would
attribute a block to whichever arm happened to start last.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from mlflow.tracking import MlflowClient

from tower_rl.experiment.tracking import TrackedRun


class MlflowTrackedRun:
    """One MLflow run, addressed by the id its handle carries."""

    def __init__(self, client: MlflowClient, run_id: str) -> None:
        self._client = client
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_metrics(self, metrics: Mapping[str, float], *, decisions: int) -> None:
        for key, value in metrics.items():
            self._client.log_metric(self._run_id, key, float(value), step=decisions)

    def log_artifact(self, path: Path, *, directory: str | None = None) -> None:
        self._client.log_artifact(self._run_id, str(path), artifact_path=directory)

    def finish(self) -> None:
        self._client.set_terminated(self._run_id)


class MlflowExperimentTracker:
    """Records runs into a local MLflow store under one experiment."""

    def __init__(self, *, tracking_uri: str, experiment: str, artifact_root: str) -> None:
        self._tracking_uri = tracking_uri
        self._client = MlflowClient(tracking_uri=tracking_uri)
        existing = self._client.get_experiment_by_name(experiment)
        self._experiment_id = (
            existing.experiment_id if existing is not None
            else self._client.create_experiment(
                experiment, artifact_location=f"{artifact_root}/{experiment}"
            )
        )

    @property
    def tracking_uri(self) -> str:
        return self._tracking_uri

    def start_run(
        self,
        *,
        name: str,
        params: Mapping[str, object],
        tags: Mapping[str, str],
    ) -> TrackedRun:
        run = self._client.create_run(
            self._experiment_id, tags=dict(tags), run_name=name
        )
        run_id = run.info.run_id
        for key, value in params.items():
            self._client.log_param(run_id, key, value)
        return MlflowTrackedRun(self._client, run_id)


__all__ = ["MlflowExperimentTracker", "MlflowTrackedRun"]
