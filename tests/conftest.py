"""What every test in this suite needs before a single test module is imported.

Two things, both of which have to happen once and early.

The import paths. Tests import the package from `src/`, the entry points from
`scripts/` (they are runnable files, not an installed package), and the test
doubles from `tests/fakes`. Doing that here rather than in each module keeps
one copy of the rule and keeps it independent of which file pytest happens to
import first.

The thread count. Torch sizes its intra-op pool from the core count, so two
pytest runs overlapping on one host oversubscribe every core and thrash: two
orphaned runs once burned 25 CPU-hours and invalidated a measurement (#4).
Nothing in this suite is a throughput benchmark of torch itself - the fleet
tests measure plumbing, not matmul - so one thread per process costs nothing
here and makes concurrent runs additive instead of quadratic. The environment
variables must be set before torch is first imported, because OpenMP and MKL
read them when their runtimes initialise; `set_num_threads` then covers the
torch-side pool for the case where torch was already imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for path in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency in practice
    pass
else:
    torch.set_num_threads(1)
