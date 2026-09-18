"""Normalized run state handed to a policy, derived from exact bridge readings.

This is the environment side of `observation-v1` for the instrumented profile.
It owns normalization, availability masking, and fail-closed validation; it does
not own transport, game rules, or the learning algorithm.  Scaling choices are
part of the schema version: changing one changes what a trained checkpoint means.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from tower_rl.environment.run_actions import (
    RUN_ACTIONS,
    SLOTS_PER_FAMILY,
    RunActionId,
    UpgradeFamily,
    action_index,
    upgrade_action,
)

OBSERVATION_SCHEMA_VERSION = "observation-v1"

# Clipped so one enormous ratio cannot dominate the encoder while an upgrade is
# trivially affordable; the mask already carries "can I buy it at all".
MAX_AFFORDABILITY_RATIO = 10.0


class UpgradeEntryLike(Protocol):
    """One exact upgrade reading, as any source must present it."""

    @property
    def family(self) -> str: ...
    @property
    def index(self) -> int: ...
    @property
    def cost(self) -> float: ...
    @property
    def level(self) -> int: ...
    @property
    def max_level(self) -> int: ...
    @property
    def unlocked(self) -> bool: ...
    @property
    def maxed(self) -> bool: ...


class ExactRunReadingLike(Protocol):
    """The exact run readings this schema is built from.

    Declared structurally so the domain never imports a transport type: the
    dependency runs inward, and any source presenting these readings will do.
    Every member is read-only, which is both true - the domain never writes back
    to a reading - and necessary: settable attributes are invariant, so a source
    reporting a narrower type than the one declared here would be rejected.
    """

    @property
    def sequence(self) -> int: ...
    @property
    def lifecycle(self) -> str: ...
    @property
    def wave(self) -> int: ...
    @property
    def cash(self) -> float: ...
    @property
    def health(self) -> float: ...
    @property
    def max_health(self) -> float: ...
    @property
    def game_speed(self) -> float: ...
    @property
    def upgrades(self) -> Sequence[UpgradeEntryLike]: ...


def _log_scale(value: float) -> float:
    """Compress an unbounded non-negative quantity into a learnable range."""
    return math.log1p(max(value, 0.0))


@dataclass(frozen=True)
class UpgradeRow:
    """One upgrade slot as the policy sees it."""

    action: RunActionId
    cost_log: float
    affordability: float
    level: int
    max_level: int
    headroom: float
    unlocked: bool
    maxed: bool
    available: bool


@dataclass(frozen=True)
class RunState:
    """One validated, evidence-bearing view of an active or terminal run."""

    source_sequence: int
    captured_at_monotonic: float
    profile_id: str
    lifecycle: str
    wave: int
    wave_log: float
    cash_log: float
    health_fraction: float
    max_health_log: float
    game_speed: float
    rows: tuple[UpgradeRow, ...]
    action_mask: tuple[bool, ...]
    valid: bool
    invalid_reasons: tuple[str, ...] = ()
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    @property
    def terminal(self) -> bool:
        return self.lifecycle == "terminal"

    def available_actions(self) -> tuple[RunActionId, ...]:
        return tuple(
            action for action, allowed in zip(RUN_ACTIONS, self.action_mask, strict=True) if allowed
        )


@dataclass(frozen=True)
class RunStateBuilder:
    """Translates one exact bridge observation into `observation-v1`."""

    profile_id: str
    slots_per_family: int = SLOTS_PER_FAMILY

    def build(
        self, reading: ExactRunReadingLike, *, captured_at_monotonic: float
    ) -> RunState:
        """Build normalized run state, refusing to invent any missing reading."""
        reasons: list[str] = []
        entries = self._index_entries(reading, reasons)
        cash = reading.cash
        max_health = reading.max_health
        wave = reading.wave
        lifecycle = reading.lifecycle
        active = lifecycle == "active"

        if max_health <= 0.0:
            reasons.append("maximum health is not positive")
            health_fraction = 0.0
        else:
            ratio = reading.health / max_health
            health_fraction = min(max(ratio, 0.0), 1.0)
            # The killing blow overkills: the game stores negative tower health
            # after the fatal hit, so a terminal state legitimately reads below
            # zero (M1B-E007). That is a correct reading of a dead tower, not a
            # corrupt one. Negative health while the run is still active would be
            # contradictory, and health above maximum is impossible either way.
            if ratio > 1.0:
                reasons.append("health exceeds maximum")
            elif ratio < 0.0 and active:
                reasons.append("negative health in an active run")
        if cash < 0.0:
            reasons.append("cash is negative")
        if wave < 0:
            reasons.append("wave is negative")

        rows = tuple(self._build_row(entries, action, cash, active) for action in RUN_ACTIONS[1:])
        # WAIT is always available inside a valid active run and never otherwise:
        # waiting on a terminal run is not a decision, it is a stalled actor.
        mask = (active, *(row.available for row in rows))

        return RunState(
            source_sequence=reading.sequence,
            captured_at_monotonic=captured_at_monotonic,
            profile_id=self.profile_id,
            lifecycle=lifecycle,
            wave=wave,
            wave_log=_log_scale(wave),
            cash_log=_log_scale(cash),
            health_fraction=health_fraction,
            max_health_log=_log_scale(max_health),
            game_speed=reading.game_speed,
            rows=rows,
            action_mask=mask,
            valid=not reasons,
            invalid_reasons=tuple(reasons),
        )

    def _index_entries(
        self, reading: ExactRunReadingLike, reasons: list[str]
    ) -> Mapping[RunActionId, UpgradeEntryLike]:
        entries: dict[RunActionId, UpgradeEntryLike] = {}
        for entry in reading.upgrades:
            if entry.index >= self.slots_per_family:
                reasons.append("upgrade inventory is wider than the supported action schema")
                continue
            entries[upgrade_action(entry.family, entry.index)] = entry
        expected = len(UpgradeFamily) * self.slots_per_family
        if len(entries) != expected:
            reasons.append(
                f"upgrade inventory has {len(entries)} slots; exactly {expected} are required"
            )
        return entries

    def _build_row(
        self,
        entries: Mapping[RunActionId, UpgradeEntryLike],
        action: RunActionId,
        cash: float,
        active: bool,
    ) -> UpgradeRow:
        entry = entries.get(action)
        if entry is None:
            # A slot the bridge did not report is masked, never assumed cheap.
            return UpgradeRow(action, 0.0, 0.0, 0, 0, 0.0, False, False, False)
        cost, level, max_level = entry.cost, entry.level, entry.max_level
        unlocked, maxed = entry.unlocked, entry.maxed
        priced = cost > 0.0
        headroom = 0.0 if max_level <= 0 else max(0.0, (max_level - level) / max_level)
        affordability = min(cash / cost, MAX_AFFORDABILITY_RATIO) if priced else 0.0
        available = active and unlocked and not maxed and priced and cost <= cash
        return UpgradeRow(
            action=action,
            cost_log=_log_scale(cost) if priced else 0.0,
            affordability=affordability,
            level=level,
            max_level=max_level,
            headroom=headroom,
            unlocked=unlocked,
            maxed=maxed,
            available=available,
        )



def validate_transition(previous: RunState, current: RunState) -> tuple[str, ...]:
    """Return the reasons one state cannot legitimately follow another."""
    reasons: list[str] = []
    if current.source_sequence <= previous.source_sequence:
        reasons.append("bridge sequence did not advance")
    if current.captured_at_monotonic <= previous.captured_at_monotonic:
        reasons.append("capture time did not advance")
    if current.profile_id != previous.profile_id:
        reasons.append("profile identity changed inside an episode")
    if current.schema_version != previous.schema_version:
        reasons.append("observation schema changed inside an episode")
    if previous.lifecycle == "active" and current.lifecycle == "active":
        if current.wave < previous.wave:
            reasons.append("wave moved backwards inside an active episode")
        for before, after in zip(previous.rows, current.rows, strict=True):
            if after.level < before.level:
                reasons.append(f"{after.action} level moved backwards")
    return tuple(reasons)


def action_is_allowed(state: RunState, action: RunActionId) -> bool:
    """Whether the mask admits this action for this state."""
    return state.action_mask[action_index(action)]
