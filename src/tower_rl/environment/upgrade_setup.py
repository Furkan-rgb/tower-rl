"""The upgrade setup an episode was actually played on, as the game read it back.

`upgrade_availability` and `workshop_level` say what a run *asked* for. This is
what the game held once that request had been applied: every in-run row the
game names, whether it was purchasable at the episode's first observation, and
the permanent Workshop level it stood at after the round began. Two runs whose
requests are spelled the same but whose games held different setups - a build
that offers fewer rows, a Workshop write that landed on other rows - are then
told apart by the setup itself rather than by the flags that asked for it.

The digest is what everything else compares: every episode record and the run
manifest carry it, a checkpoint's identity checks it, and an episode played on
a setup other than the run's first is invalid by name (`UPGRADE_SETUP_DRIFT`).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tower_rl.environment.run_port import UpgradeSlotLabelLike, WorkshopRowLike
from tower_rl.environment.run_state import RunState

#: An episode was played on another upgrade setup than the run's first. Its
#: episodes are not the same decision problem, so every state of it is invalid
#: and the episode is classified, loudly, the way a reverted Workshop level is.
UPGRADE_SETUP_DRIFT = "UPGRADE_SETUP_DRIFT"


@dataclass(frozen=True)
class UpgradeSetupRow:
    """One in-run row as the game held it at an episode's start."""

    family: str
    index: int
    #: The game's own label for the row (`slot_labels`).
    name: str
    #: The row's in-run availability flag in the episode's first observation,
    #: read after any unlock the round start applied.
    available_in_run: bool
    #: The permanent Workshop level the game read back for this row once the
    #: round had started. 0 when nothing was read (`workshop_read_back` is
    #: false); None only if a read came back without this row.
    workshop_level: int | None


class UpgradeSetupRefused(RuntimeError):
    """A run was begun on a setup other than the checkpoint it continues or plays.

    Deliberately not a `RunPortError`: the port delivered an episode, and
    retrying the boundary cannot change which setup the game holds. It ends the
    run rather than being counted as one more failed episode.
    """


@dataclass(frozen=True)
class UpgradeSetup:
    """Every in-run row the game names, as it stood at one episode's start."""

    rows: tuple[UpgradeSetupRow, ...]
    #: Whether the Workshop levels were read back from the game. False when the
    #: run writes no Workshop profile: no read is made then, and every row's
    #: level is recorded as 0 - the account as the image holds it.
    workshop_read_back: bool

    @classmethod
    def read_back(
        cls,
        labels: Sequence[UpgradeSlotLabelLike],
        first_state: RunState,
        workshop: Sequence[WorkshopRowLike] | None,
    ) -> UpgradeSetup:
        """The setup from what the game reported: labels, first state, Workshop read."""
        available = {str(row.action): row.unlocked for row in first_state.rows}
        levels = (
            {(row.family, row.index): row.after for row in workshop}
            if workshop is not None
            else None
        )
        rows = sorted(
            (
                UpgradeSetupRow(
                    family=label.family,
                    index=label.index,
                    name=label.name,
                    # A named slot the observation does not carry cannot be
                    # bought, which is what this flag records.
                    available_in_run=available.get(f"{label.family}:{label.index}", False),
                    workshop_level=(
                        0 if levels is None else levels.get((label.family, label.index))
                    ),
                )
                for label in labels
                if label.name
            ),
            key=lambda row: (row.family, row.index),
        )
        return cls(rows=tuple(rows), workshop_read_back=workshop is not None)

    def to_record(self) -> dict[str, Any]:
        """The setup as the JSON an episode record and a manifest carry."""
        return {
            "workshop_read_back": self.workshop_read_back,
            "rows": [
                {
                    "family": row.family,
                    "index": row.index,
                    "name": row.name,
                    "available_in_run": row.available_in_run,
                    "workshop_level": row.workshop_level,
                }
                for row in self.rows
            ],
        }

    @property
    def digest(self) -> str:
        """sha256 of the record as canonical JSON: sorted keys, no whitespace."""
        return setup_digest(self.to_record())


def setup_digest(record: Mapping[str, Any]) -> str:
    """The digest of one setup record, however it was obtained."""
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class UpgradeSetupReference:
    """The setup a run is held to: its first episode's, shared by its whole fleet.

    One per run, handed to every environment of it, so "the run's first" means
    the first episode any actor began rather than each actor's own. `expected`
    is the digest of a checkpoint the run continues or plays, when it has one:
    the run's first setup must then be that one, or the run is refused.
    """

    expected: str | None = None
    #: What `expected` came from, for the refusal to name.
    expected_from: str = "the checkpoint"
    first: UpgradeSetup | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def admit(self, setup: UpgradeSetup) -> str | None:
        """Pin the run's first setup, or say how this one drifted from it.

        Returns None for a setup the run is on, and the drift reason otherwise.
        Raises `UpgradeSetupRefused` when the first setup is not the expected one.
        """
        digest = setup.digest
        with self._lock:
            if self.first is None:
                if self.expected is not None and digest != self.expected:
                    raise UpgradeSetupRefused(
                        f"the game holds upgrade setup {digest[:12]}, not "
                        f"{self.expected[:12]} from {self.expected_from}: the rows "
                        "available in-run or the Workshop levels differ, so the "
                        "checkpoint's experience is not this run's decision problem"
                    )
                self.first = setup
                return None
            first = self.first.digest
        if digest == first:
            return None
        return (
            f"{UPGRADE_SETUP_DRIFT}: this episode's setup {digest[:12]} differs from "
            f"the run's first {first[:12]}"
        )
