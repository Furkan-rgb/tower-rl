"""Timestamped stdout for stage entry points.

``scripts/run_stage.sh`` captures the stdout/stderr of the entry points below
into a per-run log file. Those entry points write plain ``print()`` output
(none of them use the :mod:`logging` module), so the log has no per-line
time and failure times must otherwise be inferred from the stage's overall
``wall_seconds``. ``timestamped_print`` is a drop-in replacement for
``print`` that prefixes each call with a local ``YYYY-MM-DDTHH:MM:SS``
timestamp.

Note: a single ``print()`` call whose argument already contains embedded
newlines (for example ``print(json.dumps(report, indent=2))``) only gets one
timestamp, on its first line; the remaining lines of that call are not
individually stamped.
"""

from __future__ import annotations

import builtins
from datetime import datetime
from typing import Any

__all__ = ["timestamped_print"]


def timestamped_print(*args: Any, **kwargs: Any) -> None:
    """Print like the builtin, prefixed with a local-time timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    builtins.print(timestamp, *args, **kwargs)
