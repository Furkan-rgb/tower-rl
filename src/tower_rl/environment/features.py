"""The policy's numeric view of run state.

The encoding is part of `observation-v1`: changing it changes what a trained
checkpoint means, so any change needs a new schema version.  The layout is split
the way the network in `solution.md` 9.3 consumes it - a few run scalars, then
one fixed-width row per upgrade slot scored by shared weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.environment.run_state import RunState

#: Run-level features, in order.
SCALAR_FEATURES: tuple[str, ...] = (
    "wave_log",
    "cash_log",
    "health_fraction",
    "max_health_log",
)

#: Per-slot features, in order. Deliberately excludes `tier_unlocked`, which live
#: 29.0.3 reports false for every offered upgrade and therefore carries no signal
#: (M1B-E001); it stays in the raw reading for drift detection.
ROW_FEATURES: tuple[str, ...] = (
    "cost_log",
    "affordability",
    "level_fraction",
    "headroom",
    "unlocked",
    "maxed",
    "available",
)

SCALAR_COUNT = len(SCALAR_FEATURES)
ROW_COUNT = len(RUN_ACTIONS) - 1
ROW_WIDTH = len(ROW_FEATURES)


@dataclass(frozen=True)
class StateFeatures:
    """One encoded state: run scalars, per-slot rows, and the validity mask.

    Rows are flattened in action order, so row `i` describes action index `i + 1`
    and index 0 is always `WAIT`.
    """

    scalars: tuple[float, ...]
    rows: tuple[float, ...]
    mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.scalars) != SCALAR_COUNT:
            raise ValueError(f"expected {SCALAR_COUNT} scalar features")
        if len(self.rows) != ROW_COUNT * ROW_WIDTH:
            raise ValueError(f"expected {ROW_COUNT * ROW_WIDTH} row features")
        if len(self.mask) != len(RUN_ACTIONS):
            raise ValueError(f"expected a mask of {len(RUN_ACTIONS)} actions")


def encode_state(state: RunState) -> StateFeatures:
    """Encode validated run state for the policy."""
    scalars = (
        state.wave_log,
        state.cash_log,
        state.health_fraction,
        state.max_health_log,
    )
    rows: list[float] = []
    for row in state.rows:
        level_fraction = 0.0 if row.max_level <= 0 else min(1.0, row.level / row.max_level)
        rows.extend(
            (
                row.cost_log,
                row.affordability,
                level_fraction,
                row.headroom,
                float(row.unlocked),
                float(row.maxed),
                float(row.available),
            )
        )
    return StateFeatures(scalars=scalars, rows=tuple(rows), mask=state.action_mask)
