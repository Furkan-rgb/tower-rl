"""The port a training session records itself through.

A run that leaves no trace cannot be compared with the next one, so every
training session reports what it was configured with, what it measured, and
which files it produced.  Where that lands - a local MLflow store, a different
tool later, nowhere at all - is an adapter's business, and nothing that learns
or trains may know which one it is talking to.

A run is addressed by a handle rather than by an id the caller carries around,
because arms of one comparison are interleaved on the device and their runs are
therefore open at the same time.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable


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


__all__ = ["ExperimentTracker", "NoExperimentTracker", "TrackedRun"]
