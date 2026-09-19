"""Normalized run state handed to a policy, derived from exact bridge readings.

This is the environment side of `observation-v2` for the instrumented profile.
It owns normalization, availability masking, and fail-closed validation; it does
not own transport, game rules, or the learning algorithm.  Scaling choices are
part of the schema version: changing one changes what a trained checkpoint means.

`observation-v2` shows the policy what the player sees on the run screen: the
tower's live combat stats, the wave and the threat in it, the economy and the
round clock, beside the upgrade grid v1 already carried.  Each of those is one
field of the game's own `Main`, read whole and rescaled - never summarised -
which is what `LIVE_FIELDS` declares once for the whole system.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from tower_rl.environment.run_actions import (
    RUN_ACTIONS,
    SLOTS_PER_FAMILY,
    RunActionId,
    UpgradeFamily,
    action_index,
    upgrade_action,
)

OBSERVATION_SCHEMA_VERSION = "observation-v2"

#: What `closestEnemyDistance` reads when there is no enemy to be at a distance.
#: The game stores a sentinel rather than an absence, and 138 device snapshots
#: put it at exactly this value in every enemy-free one (board #39). It is read
#: as a distance of zero and an `enemy_present` flag of zero, so the policy never
#: sees a threat ten thousand metres away that does not exist.
NO_ENEMY_DISTANCE = 10_000.0


def _log_scale(value: float) -> float:
    """Compress an unbounded non-negative quantity into a learnable range."""
    return math.log1p(max(value, 0.0))


class LiveTransform(StrEnum):
    """How one live reading is rescaled into a feature.

    Rescaling, never summarising: every transform here is monotone over the
    range the game produces, so nothing the player can read is collapsed away.
    The transform also fixes the field's range invariant - what a legitimate
    reading of it can be - which is what makes a bad reading an attributable
    anomaly rather than a plausible number (the M1B-E017 lesson).
    """

    #: An unbounded non-negative quantity - damage, health, cash, a duration in
    #: seconds - compressed with `log1p`. Finite and non-negative.
    MAGNITUDE = "magnitude"
    #: A field the game stores in percent (`criticalChance` reads 1.0 for 1 %,
    #: board #39), divided by 100. Must land in [0, 1] after scaling.
    PERCENT = "percent"
    #: A small count, passed through raw: its magnitude is already learnable and
    #: its steps mean something. Finite and non-negative.
    COUNT = "count"
    #: A boolean the game stores as 0 or 1.
    FLAG = "flag"
    #: A raw distance, with `NO_ENEMY_DISTANCE` read as absence and a presence
    #: flag beside it. Finite and non-negative.
    DISTANCE = "distance"


@dataclass(frozen=True)
class LiveField:
    """One `Main` field the bridge reads every snapshot, and how v2 scales it.

    `wire` is the game's own field name and also the key the bridge sends it
    under: one name from the game to the tensor, so a reading is traceable back
    to the field it came from without a translation table in between.
    """

    wire: str
    feature: str
    transform: LiveTransform
    #: What the player reads on the HUD, for the contract table and the panel.
    unit: str
    #: Only a `DISTANCE` field has one: the flag that says the distance is real.
    presence_feature: str | None = None

    @property
    def features(self) -> tuple[str, ...]:
        if self.presence_feature is None:
            return (self.feature,)
        return (self.feature, self.presence_feature)


#: Every live reading `observation-v2` carries, in feature order. This is the
#: single declaration of the schema's run scalars beyond the four v1 keeps
#: explicit on `RunState`: the bridge decoder requires exactly these wire names,
#: `features.SCALAR_FEATURES` takes its order from here, and
#: `docs/environment-contract.md` tabulates it. Units are the ones 138 device
#: snapshots observed (board #39).
LIVE_FIELDS: tuple[LiveField, ...] = (
    # -- the tower's live stats, as its own upgrade rows report them --------
    LiveField("damage", "damage_log", LiveTransform.MAGNITUDE, "damage per shot"),
    LiveField("attackSpeed", "attack_speed_log", LiveTransform.MAGNITUDE, "shots per second"),
    LiveField("criticalChance", "critical_chance_fraction", LiveTransform.PERCENT, "percent"),
    LiveField("criticalMult", "critical_mult_log", LiveTransform.MAGNITUDE, "multiplier"),
    LiveField("superCritChance", "super_crit_chance_fraction", LiveTransform.PERCENT, "percent"),
    LiveField("multishotChance", "multishot_chance_fraction", LiveTransform.PERCENT, "percent"),
    LiveField("multishotTargets", "multishot_targets", LiveTransform.COUNT, "targets"),
    LiveField("rapidFireChance", "rapid_fire_chance_fraction", LiveTransform.PERCENT, "percent"),
    LiveField("rapidFireDuration", "rapid_fire_duration_log", LiveTransform.MAGNITUDE, "seconds"),
    LiveField("towerRangeDistance", "tower_range_log", LiveTransform.MAGNITUDE, "metres"),
    LiveField("knockbackChance", "knockback_chance_fraction", LiveTransform.PERCENT, "percent"),
    LiveField("knockbackForce", "knockback_force_log", LiveTransform.MAGNITUDE, "force"),
    LiveField("lifesteal", "lifesteal_fraction", LiveTransform.PERCENT, "percent"),
    LiveField("thornDamage", "thorn_damage_log", LiveTransform.MAGNITUDE, "damage reflected"),
    LiveField("defenseAbs", "defense_absolute_log", LiveTransform.MAGNITUDE, "damage blocked"),
    LiveField("defenseRel", "defense_relative_fraction", LiveTransform.PERCENT, "percent"),
    LiveField("towerHealthRegen", "health_regen_log", LiveTransform.MAGNITUDE, "health per second"),
    LiveField("wallHealth", "wall_health_log", LiveTransform.MAGNITUDE, "health"),
    LiveField("wallRebuild", "wall_rebuild_log", LiveTransform.MAGNITUDE, "seconds"),
    LiveField("orbCount", "orb_count", LiveTransform.COUNT, "orbs"),
    LiveField("orbSpeed", "orb_speed_log", LiveTransform.MAGNITUDE, "revolutions per second"),
    # -- the wave, and the threat in it -------------------------------------
    LiveField("currentWaveBaseHealth", "wave_base_health_log", LiveTransform.MAGNITUDE, "health"),
    LiveField("currentWaveBaseDamage", "wave_base_damage_log", LiveTransform.MAGNITUDE, "damage"),
    LiveField(
        "currentWaveBaseKillCash", "wave_base_kill_cash_log", LiveTransform.MAGNITUDE, "cash"
    ),
    LiveField("enemiesSpawnedThisWave", "enemies_spawned", LiveTransform.COUNT, "enemies"),
    LiveField("enemiesKilledThisWave", "enemies_killed", LiveTransform.COUNT, "enemies"),
    LiveField(
        "estimatedEnemiesToSpawnThisWave", "enemies_expected", LiveTransform.COUNT, "enemies"
    ),
    LiveField(
        "closestEnemyDistance",
        "closest_enemy_distance",
        LiveTransform.DISTANCE,
        "metres",
        presence_feature="enemy_present",
    ),
    LiveField("bossWaveBool", "boss_wave", LiveTransform.FLAG, "boolean"),
    LiveField("bossSpawnedBool", "boss_spawned", LiveTransform.FLAG, "boolean"),
    LiveField("miniBossWaveBool", "mini_boss_wave", LiveTransform.FLAG, "boolean"),
    LiveField("waveTimer", "wave_timer_log", LiveTransform.MAGNITUDE, "seconds"),
    LiveField("waveLengthSeconds", "wave_length_log", LiveTransform.MAGNITUDE, "seconds"),
    LiveField("waveCooldownSeconds", "wave_cooldown_log", LiveTransform.MAGNITUDE, "seconds"),
    # -- the economy, and the round clock -----------------------------------
    LiveField("cashPerWave", "cash_per_wave_log", LiveTransform.MAGNITUDE, "cash per wave"),
    LiveField("cashEarnedThisWave", "cash_earned_this_wave_log", LiveTransform.MAGNITUDE, "cash"),
    LiveField("gameplayTimeThisRound", "round_time_log", LiveTransform.MAGNITUDE, "seconds"),
)

#: The wire names the bridge must send, in one place for the decoder to require.
LIVE_WIRE_NAMES: tuple[str, ...] = tuple(live.wire for live in LIVE_FIELDS)

#: Live feature names in tensor order, presence flags included.
LIVE_FEATURES: tuple[str, ...] = tuple(name for live in LIVE_FIELDS for name in live.features)

#: The prefix an out-of-range reading is reported under, so a record reader can
#: find every one of them and attribute each to the `Main` field it came from.
OUT_OF_RANGE_REASON = "OBSERVATION_OUT_OF_RANGE"


def scale_live_reading(live: LiveField, raw: float) -> tuple[dict[str, float], str | None]:
    """Rescale one raw reading, and say when the reading cannot be legitimate.

    A violation is reported rather than raised: the episode is not lost, the
    transition is made inadmissible by the reason, and the anomaly is attributed
    to the `Main` field it came from in the episode record.
    """
    zeros = dict.fromkeys(live.features, 0.0)
    out_of_range = f"{OUT_OF_RANGE_REASON}:{live.wire}"
    if not math.isfinite(raw):
        return zeros, out_of_range
    match live.transform:
        case LiveTransform.PERCENT:
            fraction = raw / 100.0
            if not 0.0 <= fraction <= 1.0:
                return zeros, out_of_range
            return {live.feature: fraction}, None
        case LiveTransform.FLAG:
            if raw not in (0.0, 1.0):
                return zeros, out_of_range
            return {live.feature: raw}, None
        case LiveTransform.COUNT:
            if raw < 0.0:
                return zeros, out_of_range
            return {live.feature: raw}, None
        case LiveTransform.DISTANCE:
            if raw < 0.0:
                return zeros, out_of_range
            if raw >= NO_ENEMY_DISTANCE:
                return zeros, None
            return {live.feature: raw, str(live.presence_feature): 1.0}, None
        case LiveTransform.MAGNITUDE:
            if raw < 0.0:
                return zeros, out_of_range
            return {live.feature: _log_scale(raw)}, None


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
    @property
    def live(self) -> Mapping[str, float]:
        """Every `LIVE_FIELDS` reading, raw, keyed by the game's own field name."""
        ...


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
    #: Every `LIVE_FEATURES` value, scaled, keyed by feature name. One mapping
    #: rather than thirty-eight fields: the schema is declared once in
    #: `LIVE_FIELDS`, and a field added there needs no second declaration here.
    live: Mapping[str, float]
    rows: tuple[UpgradeRow, ...]
    action_mask: tuple[bool, ...]
    valid: bool
    invalid_reasons: tuple[str, ...] = ()
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    @property
    def terminal(self) -> bool:
        return self.lifecycle == "terminal"

    @property
    def is_choice_point(self) -> bool:
        """Whether this state offers the policy a purchase to choose.

        A state whose only legal action is `WAIT` is not a decision: the policy
        has exactly one answer available and the environment already knows what
        it is. The mask is built from cash and prices in `_build_row`, so this
        is read off the mask rather than recomputed - what is legal and what is
        a choice point cannot drift apart. Index 0 is `WAIT`; every other index
        is a purchase.
        """
        return any(self.action_mask[1:])

    def available_actions(self) -> tuple[RunActionId, ...]:
        return tuple(
            action for action, allowed in zip(RUN_ACTIONS, self.action_mask, strict=True) if allowed
        )


@dataclass(frozen=True)
class RunStateBuilder:
    """Translates one exact bridge observation into `observation-v2`."""

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
        live = self._scale_live(reading, reasons)

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
            live=live,
            rows=rows,
            action_mask=mask,
            valid=not reasons,
            invalid_reasons=tuple(reasons),
        )

    def _scale_live(
        self, reading: ExactRunReadingLike, reasons: list[str]
    ) -> Mapping[str, float]:
        """Scale every live reading, refusing to invent one the source omitted.

        This is where `observation-v2`'s range invariant is enforced: a reading
        that cannot be legitimate is zeroed *and* named, so the transition is
        inadmissible and the anomaly reaches the episode record attributed to
        the `Main` field it came from rather than disappearing into a plausible
        number (the M1B-E017 lesson).
        """
        raw = reading.live
        scaled: dict[str, float] = dict.fromkeys(LIVE_FEATURES, 0.0)
        for live in LIVE_FIELDS:
            if live.wire not in raw:
                # Never a silent zero: the bridge refuses to start without the
                # field, so an absent one here is a source that is not v2.
                reasons.append(f"observation is missing the live reading {live.wire}")
                continue
            values, reason = scale_live_reading(live, float(raw[live.wire]))
            scaled.update(values)
            if reason is not None:
                reasons.append(reason)
        return scaled

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
        # Unclipped in v2. v1 clipped the ratio at ten so one trivially cheap
        # upgrade could not dominate the encoder, which also erased the whole
        # difference between "ten times over" and "a hundred times over" - the
        # difference between an early purchase and a late one. `log1p` keeps that
        # ordering at a learnable magnitude without a ceiling to saturate at.
        affordability = _log_scale(cash / cost) if priced else 0.0
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


def hud_readings(state: RunState) -> dict[str, float]:
    """Every live reading back in the unit the player reads it in.

    The exact inverse of `scale_live_reading`, keyed by the game's own field
    name. It exists so a human can hold the panel beside the game's HUD and see
    the same numbers: a log-scaled damage is not something anybody can check a
    screen against, and a value nobody can check is a value nobody will.
    """
    readings: dict[str, float] = {}
    for live in LIVE_FIELDS:
        scaled = state.live[live.feature]
        match live.transform:
            case LiveTransform.MAGNITUDE:
                readings[live.wire] = math.expm1(scaled)
            case LiveTransform.PERCENT:
                readings[live.wire] = scaled * 100.0
            case LiveTransform.DISTANCE:
                # An absence reads as the game's own sentinel again, so the panel
                # shows "no enemy" rather than an enemy at zero distance.
                present = state.live[str(live.presence_feature)]
                readings[live.wire] = scaled if present else NO_ENEMY_DISTANCE
            case _:
                readings[live.wire] = scaled
    return readings


def action_is_allowed(state: RunState, action: RunActionId) -> bool:
    """Whether the mask admits this action for this state."""
    return state.action_mask[action_index(action)]
