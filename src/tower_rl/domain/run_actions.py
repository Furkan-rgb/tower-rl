"""The versioned semantic action space for an instrumented Tier-1 run.

An action identity is the game's own upgrade slot, not a screen coordinate and
not a hand-assigned label.  The game reports every slot it has and flips
availability, so the space is fixed at `WAIT` plus every slot, and what changes
between profiles is the mask rather than the numbering.  That keeps numbering
stable across progression and game updates, which `run-action-v1` requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

ACTION_SCHEMA_VERSION = "run-action-v1"

# The supported 29.0.3 baseline reports twenty slots per family. A build that
# reports a different count is a different action schema and must fail closed
# rather than silently renumber. The bridge reads availability over exactly this
# width while it advances, as `kMaskSlotsPerFamily` in
# `native/tower_bridge/tower_bridge.cpp`; if the two drift apart the bridge stops
# on availability the host has no action for.
SLOTS_PER_FAMILY = 20


class UpgradeFamily(StrEnum):
    ATTACK = "attack"
    DEFENSE = "defense"
    UTILITY = "utility"


@dataclass(frozen=True, order=True)
class RunActionId:
    """One semantic run action: `WAIT`, or one earned-cash upgrade slot."""

    family: UpgradeFamily | None
    slot: int | None

    def __post_init__(self) -> None:
        if (self.family is None) != (self.slot is None):
            raise ValueError("an upgrade action needs both a family and a slot")
        if self.slot is not None and not 0 <= self.slot < SLOTS_PER_FAMILY:
            raise ValueError(f"upgrade slot {self.slot} is outside the supported range")

    @property
    def is_wait(self) -> bool:
        return self.family is None

    def __str__(self) -> str:
        return "WAIT" if self.family is None else f"{self.family.value}:{self.slot}"


WAIT = RunActionId(None, None)


def upgrade_action(family: UpgradeFamily | str, slot: int) -> RunActionId:
    """Return the stable identity of one upgrade slot."""
    return RunActionId(UpgradeFamily(family), slot)


def _build_action_space() -> tuple[RunActionId, ...]:
    actions = [WAIT]
    for family in UpgradeFamily:
        actions.extend(RunActionId(family, slot) for slot in range(SLOTS_PER_FAMILY))
    return tuple(actions)


RUN_ACTIONS: tuple[RunActionId, ...] = _build_action_space()
"""Every action index in `run-action-v1`; index 0 is always `WAIT`."""

_ACTION_INDICES: dict[RunActionId, int] = {
    action: index for index, action in enumerate(RUN_ACTIONS)
}


def action_index(action: RunActionId) -> int:
    """Return the stable integer index of an action within this schema version."""
    try:
        return _ACTION_INDICES[action]
    except KeyError:
        raise ValueError(f"{action} is not part of {ACTION_SCHEMA_VERSION}") from None


def action_at(index: int) -> RunActionId:
    """Return the action identity for a stable schema index."""
    if not 0 <= index < len(RUN_ACTIONS):
        raise ValueError(f"action index {index} is outside {ACTION_SCHEMA_VERSION}")
    return RUN_ACTIONS[index]
