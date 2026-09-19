"""A tracker that records what it was asked to record and nothing else.

This is a TEST DOUBLE. It exists so the calls a training session makes through
`ExperimentTracker` can be checked without a tracking server, an MLflow install
or a file store, which is also what keeps the suite runnable on a machine that
has none of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from tower_rl.experiment.tracking import TrackedRun


@dataclass
class MetricPoint:
    """One `log_metrics` call, with the budget position it was keyed by."""

    decisions: int
    metrics: dict[str, float]


@dataclass
class RecordedRun:
    """Everything one arm reported, in the order it reported it."""

    name: str
    params: dict[str, object]
    tags: dict[str, str]
    points: list[MetricPoint] = field(default_factory=list)
    artifacts: list[tuple[Path, str | None]] = field(default_factory=list)
    finished: bool = False
    #: Method names in call order, which is what the session's shape is read from.
    calls: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return f"recorded-{self.name}"

    def log_metrics(self, metrics: Mapping[str, float], *, decisions: int) -> None:
        self.calls.append("log_metrics")
        self.points.append(MetricPoint(decisions=decisions, metrics=dict(metrics)))

    def log_artifact(self, path: Path, *, directory: str | None = None) -> None:
        self.calls.append("log_artifact")
        self.artifacts.append((path, directory))

    def finish(self) -> None:
        self.calls.append("finish")
        self.finished = True


@dataclass
class RecordingTracker:
    """Hands out a `RecordedRun` per arm and keeps them all."""

    runs: list[RecordedRun] = field(default_factory=list)

    @property
    def tracking_uri(self) -> str:
        return "recording://memory"

    def start_run(
        self,
        *,
        name: str,
        params: Mapping[str, object],
        tags: Mapping[str, str],
    ) -> TrackedRun:
        run = RecordedRun(name=name, params=dict(params), tags=dict(tags))
        run.calls.append("start_run")
        self.runs.append(run)
        return run

    def open_run(self, run_id: str) -> TrackedRun:
        """Attach to a run already recorded here, as a resumed segment does.

        The same `RecordedRun` comes back, so what a resume reports lands on the
        one series the first segment opened - which is the whole point of
        reattaching rather than starting a second run beside it.
        """
        for run in self.runs:
            if run.run_id == run_id:
                run.calls.append("open_run")
                return run
        raise KeyError(f"no recorded run {run_id}")
