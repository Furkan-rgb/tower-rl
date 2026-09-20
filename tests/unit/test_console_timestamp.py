"""`timestamped_print`'s one guarantee: every call gets a stamped prefix.

`scripts/run_actors.py`, `scripts/train.py`, `scripts/run_episodes.py` and
`scripts/spectate.py` shadow the builtin `print` with this at import time, so
the stage log `scripts/run_stage.sh` captures their stdout/stderr into has a
per-line time instead of only the stage's overall `wall_seconds`.
"""

from __future__ import annotations

import re

import pytest

from tower_rl.console_timestamp import timestamped_print

TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} ")


def test_timestamped_print_prefixes_the_line(capsys: pytest.CaptureFixture[str]) -> None:
    timestamped_print("collecting", "3 actors")

    written = capsys.readouterr().out
    assert TIMESTAMP_PREFIX.match(written), written
    assert written.rstrip("\n").endswith("collecting 3 actors")
