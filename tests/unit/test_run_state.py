from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from tower_rl.environment.run_actions import (
    ACTION_SCHEMA_VERSION,
    RUN_ACTIONS,
    SLOTS_PER_FAMILY,
    WAIT,
    RunActionId,
    UpgradeFamily,
    action_at,
    action_index,
    upgrade_action,
)
from tower_rl.environment.run_state import (
    LIVE_FIELDS,
    LIVE_WIRE_NAMES,
    NO_ENEMY_DISTANCE,
    OUT_OF_RANGE_REASON,
    LiveTransform,
    RunState,
    RunStateBuilder,
    validate_transition,
)
from tower_rl.simulation.instrumented_bridge import BridgeObservation, UpgradeInventoryEntry

BUILDER = RunStateBuilder(profile_id="tower-play-29.0.3-rooted-v1")


def action_is_allowed(state: RunState, action: RunActionId) -> bool:
    """Whether the mask admits this action for this state."""
    return state.action_mask[action_index(action)]


def _entry(
    family: str,
    index: int,
    *,
    cost: float = 10.0,
    level: int = 0,
    max_level: int = 100,
    unlocked: bool = True,
    maxed: bool = False,
) -> UpgradeInventoryEntry:
    return UpgradeInventoryEntry(
        family=family,
        index=index,
        cost=cost,
        level=level,
        max_level=max_level,
        unlocked=unlocked,
        tier_unlocked=False,
        maxed=maxed,
    )


def _reading(
    *,
    sequence: int = 1,
    lifecycle: str = "active",
    wave: int = 3,
    cash: float = 100.0,
    health: float = 4.0,
    max_health: float = 5.0,
    overrides: dict[tuple[str, int], UpgradeInventoryEntry] | None = None,
    live: dict[str, float] | None = None,
) -> BridgeObservation:
    entries = []
    for family in ("attack", "defense", "utility"):
        for index in range(SLOTS_PER_FAMILY):
            override = (overrides or {}).get((family, index))
            entries.append(override if override is not None else _entry(family, index))
    return BridgeObservation(
        sequence=sequence,
        lifecycle=lifecycle,
        wave=wave,
        cash=cash,
        health=health,
        max_health=max_health,
        terminal=lifecycle == "terminal",
        round_active=lifecycle == "active",
        game_speed=1.5,
        play_time=1234.5,
        upgrades=tuple(entries),
        live={**dict.fromkeys(LIVE_WIRE_NAMES, 0.0), **(live or {})},
    )


def test_action_space_is_stable_and_wait_is_first() -> None:
    assert action_index(WAIT) == 0
    assert action_at(0) == WAIT
    assert len(RUN_ACTIONS) == 1 + 3 * SLOTS_PER_FAMILY
    # Numbering must not depend on iteration order anywhere else in the system.
    assert action_index(upgrade_action(UpgradeFamily.ATTACK, 0)) == 1
    assert action_index(upgrade_action(UpgradeFamily.DEFENSE, 0)) == 1 + SLOTS_PER_FAMILY
    assert action_index(upgrade_action(UpgradeFamily.UTILITY, 0)) == 1 + 2 * SLOTS_PER_FAMILY
    assert str(upgrade_action("attack", 2)) == "attack:2"
    assert ACTION_SCHEMA_VERSION == "run-action-v1"


def test_the_slot_width_matches_the_width_the_bridge_scans() -> None:
    """The bridge reads availability over `kMaskSlotsPerFamily` slots per family.

    That constant in `native/tower_bridge/tower_bridge.cpp` and this one are the
    same number in two languages and cannot be shared, so they are pinned here:
    if they drift, the bridge stops advancing on availability the host has no
    action for, or ignores availability the host would have acted on.
    """
    assert SLOTS_PER_FAMILY == 20


def test_actions_outside_the_schema_are_refused() -> None:
    with pytest.raises(ValueError, match="outside the supported range"):
        upgrade_action(UpgradeFamily.ATTACK, SLOTS_PER_FAMILY)
    with pytest.raises(ValueError, match="needs both a family and a slot"):
        RunActionId(UpgradeFamily.ATTACK, None)
    with pytest.raises(ValueError, match="outside"):
        action_at(len(RUN_ACTIONS))


def test_exact_readings_become_normalized_run_state() -> None:
    state = BUILDER.build(_reading(cash=100.0), captured_at_monotonic=10.0)

    assert state.valid and state.invalid_reasons == ()
    assert state.wave == 3
    assert state.wave_log == pytest.approx(math.log1p(3))
    assert state.cash_log == pytest.approx(math.log1p(100.0))
    assert state.health_fraction == pytest.approx(0.8)
    assert state.game_speed == 1.5
    assert len(state.rows) == 3 * SLOTS_PER_FAMILY
    assert state.schema_version == "observation-v2"

    row = state.rows[0]
    assert row.action == upgrade_action("attack", 0)
    assert row.cost_log == pytest.approx(math.log1p(10.0))
    assert row.affordability == pytest.approx(math.log1p(10.0))
    assert row.headroom == pytest.approx(1.0)


def test_every_live_reading_round_trips_through_its_own_transform() -> None:
    """One raw reading per declared field, scaled the way the contract says.

    The raw values are chosen so each transform produces something no other
    transform would, which is what makes this a round trip rather than a check
    that the keys exist.
    """
    raw = {
        "damage": 12.09,  # magnitude
        "criticalChance": 5.0,  # percent on the wire
        "multishotTargets": 2.0,  # small count, raw
        "bossWaveBool": 1.0,  # boolean
        "closestEnemyDistance": 1.5,  # a real distance
        "waveTimer": 34.47,  # seconds
    }
    state = BUILDER.build(_reading(live=raw), captured_at_monotonic=1.0)

    assert state.valid, state.invalid_reasons
    assert state.live["damage_log"] == pytest.approx(math.log1p(12.09))
    assert state.live["critical_chance_fraction"] == pytest.approx(0.05)
    assert state.live["multishot_targets"] == pytest.approx(2.0)
    assert state.live["boss_wave"] == pytest.approx(1.0)
    assert state.live["closest_enemy_distance"] == pytest.approx(1.5)
    assert state.live["enemy_present"] == pytest.approx(1.0)
    assert state.live["wave_timer_log"] == pytest.approx(math.log1p(34.47))
    # Nothing the schema declares may be absent from a valid state.
    assert set(state.live) == {name for live in LIVE_FIELDS for name in live.features}


def test_the_no_enemy_sentinel_is_an_absence_not_a_distance() -> None:
    state = BUILDER.build(
        _reading(live={"closestEnemyDistance": NO_ENEMY_DISTANCE}), captured_at_monotonic=1.0
    )

    assert state.valid, state.invalid_reasons
    assert state.live["closest_enemy_distance"] == pytest.approx(0.0)
    assert state.live["enemy_present"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("wire", "raw"),
    [
        ("damage", -1.0),  # a magnitude cannot be negative
        ("criticalChance", 250.0),  # a fraction cannot exceed one
        ("enemiesSpawnedThisWave", -3.0),  # a count cannot be negative
        ("bossWaveBool", 7.0),  # a boolean is 0 or 1
        ("closestEnemyDistance", -2.0),  # a distance cannot be negative
        ("gameplayTimeThisRound", -0.5),  # a clock cannot run backwards
    ],
)
def test_an_impossible_live_reading_names_its_own_field(wire: str, raw: float) -> None:
    """The M1B-E017 invariant: an anomaly is attributable, never a plausible zero."""
    state = BUILDER.build(_reading(live={wire: raw}), captured_at_monotonic=1.0)

    assert not state.valid
    assert f"{OUT_OF_RANGE_REASON}:{wire}" in state.invalid_reasons
    # Zeroed as well as named, so nothing downstream reads the bad value.
    field = next(live for live in LIVE_FIELDS if live.wire == wire)
    assert all(state.live[name] == 0.0 for name in field.features)


def test_a_reading_the_source_never_sent_is_refused_rather_than_assumed() -> None:
    reading = _reading()
    without = replace(reading, live={k: v for k, v in reading.live.items() if k != "damage"})

    state = BUILDER.build(without, captured_at_monotonic=1.0)

    assert not state.valid
    assert "observation is missing the live reading damage" in state.invalid_reasons


def test_every_declared_field_names_a_transform_and_a_unit() -> None:
    """The contract table is generated from this, so it cannot be half-declared."""
    for live in LIVE_FIELDS:
        assert live.wire and live.feature and live.unit
        assert isinstance(live.transform, LiveTransform)
        assert (live.presence_feature is not None) == (
            live.transform is LiveTransform.DISTANCE
        )
    assert len(set(LIVE_WIRE_NAMES)) == len(LIVE_WIRE_NAMES)


def test_mask_follows_game_owned_availability() -> None:
    overrides = {
        ("attack", 1): _entry("attack", 1, cost=500.0),  # unaffordable
        ("attack", 2): _entry("attack", 2, unlocked=False),  # not offered
        ("attack", 3): _entry("attack", 3, maxed=True),  # nothing left to buy
        ("attack", 4): _entry("attack", 4, cost=0.0),  # unpriced, never free
    }
    state = BUILDER.build(_reading(cash=100.0, overrides=overrides), captured_at_monotonic=1.0)

    assert action_is_allowed(state, WAIT)
    assert action_is_allowed(state, upgrade_action("attack", 0))
    for slot in (1, 2, 3, 4):
        assert not action_is_allowed(state, upgrade_action("attack", slot))


def test_a_terminal_run_offers_no_action_at_all() -> None:
    state = BUILDER.build(_reading(lifecycle="terminal", health=0.0), captured_at_monotonic=1.0)

    assert state.terminal
    assert not any(state.action_mask)
    assert not action_is_allowed(state, WAIT)


def test_affordability_is_unclipped_so_late_wealth_still_reads_as_wealth() -> None:
    """v1 clipped the ratio at ten, which erased every difference above it."""
    overrides = {("attack", 0): _entry("attack", 0, cost=1.0)}
    rich = BUILDER.build(_reading(cash=10_000.0, overrides=overrides), captured_at_monotonic=1.0)
    richer = BUILDER.build(
        _reading(cash=1_000_000.0, overrides=overrides), captured_at_monotonic=1.0
    )

    assert rich.rows[0].affordability == pytest.approx(math.log1p(10_000.0))
    assert richer.rows[0].affordability > rich.rows[0].affordability
    assert rich.rows[0].cost_log == pytest.approx(math.log1p(1.0))


def test_impossible_readings_invalidate_rather_than_normalize() -> None:
    state = BUILDER.build(_reading(health=9.0, max_health=5.0), captured_at_monotonic=1.0)

    assert not state.valid
    assert "health exceeds maximum" in state.invalid_reasons
    # The value is still clamped for downstream safety, never silently trusted.
    assert state.health_fraction == pytest.approx(1.0)


def test_a_short_inventory_is_invalid_and_its_missing_slots_are_masked() -> None:
    truncated = replace(_reading(), upgrades=_reading().upgrades[:10])

    state = BUILDER.build(truncated, captured_at_monotonic=1.0)

    assert not state.valid
    assert any("exactly 60 are required" in reason for reason in state.invalid_reasons)
    assert not action_is_allowed(state, upgrade_action("utility", 0))


def test_transitions_must_move_forward_within_one_profile() -> None:
    first = BUILDER.build(_reading(sequence=1, wave=3), captured_at_monotonic=1.0)
    forward = BUILDER.build(_reading(sequence=2, wave=4), captured_at_monotonic=2.0)

    assert validate_transition(first, forward) == ()

    stale = BUILDER.build(_reading(sequence=1, wave=4), captured_at_monotonic=2.0)
    assert "bridge sequence did not advance" in validate_transition(first, stale)

    backwards = BUILDER.build(_reading(sequence=3, wave=2), captured_at_monotonic=3.0)
    assert "wave moved backwards inside an active episode" in validate_transition(first, backwards)

    other_profile = RunStateBuilder(profile_id="other").build(
        _reading(sequence=4), captured_at_monotonic=4.0
    )
    assert "profile identity changed inside an episode" in validate_transition(first, other_profile)


def test_an_upgrade_level_cannot_move_backwards_inside_an_episode() -> None:
    bought = {("attack", 0): _entry("attack", 0, level=2)}
    before = BUILDER.build(_reading(sequence=1, overrides=bought), captured_at_monotonic=1.0)
    after = BUILDER.build(_reading(sequence=2), captured_at_monotonic=2.0)

    assert "attack:0 level moved backwards" in validate_transition(before, after)


def test_overkill_on_the_killing_blow_is_a_valid_terminal_reading() -> None:
    """M1B-E007: the game stores negative health after the fatal hit."""
    state = BUILDER.build(
        _reading(lifecycle="terminal", health=-37.5, max_health=5.0),
        captured_at_monotonic=1.0,
    )

    assert state.valid, state.invalid_reasons
    assert state.terminal
    assert state.health_fraction == 0.0


def test_negative_health_during_an_active_run_is_contradictory() -> None:
    state = BUILDER.build(
        _reading(lifecycle="active", health=-1.0, max_health=5.0), captured_at_monotonic=1.0
    )

    assert not state.valid
    assert "negative health in an active run" in state.invalid_reasons


def test_health_above_maximum_is_invalid_in_any_lifecycle() -> None:
    for lifecycle in ("active", "terminal"):
        state = BUILDER.build(
            _reading(lifecycle=lifecycle, health=9.0, max_health=5.0),
            captured_at_monotonic=1.0,
        )
        assert not state.valid
        assert "health exceeds maximum" in state.invalid_reasons


def test_only_the_sentinel_itself_is_an_absence() -> None:
    """A distance past the sentinel is a reading this schema cannot account for.

    Treating everything at or above 10000 as "no enemy" would let a field that
    had started reporting something else pass silently as an absence forever,
    which is exactly the silence the range invariant exists to break.
    """
    beyond = BUILDER.build(
        _reading(live={"closestEnemyDistance": NO_ENEMY_DISTANCE + 1.0}),
        captured_at_monotonic=1.0,
    )

    assert not beyond.valid
    assert f"{OUT_OF_RANGE_REASON}:closestEnemyDistance" in beyond.invalid_reasons
    assert beyond.live["closest_enemy_distance"] == 0.0
    assert beyond.live["enemy_present"] == 0.0


def test_the_contract_table_is_the_declaration_it_claims_to_be() -> None:
    """`docs/environment-contract.md` says what the schema is; this proves it does.

    The table is what a reader trusts to know what the agent sees and in what
    unit. A documented transform that has quietly stopped matching the code is
    worse than none: it is a false account of the observation every checkpoint
    was trained on. So the rows are parsed and compared, rather than the
    document being kept current by somebody remembering to.
    """
    how = {
        LiveTransform.MAGNITUDE: "log1p",
        LiveTransform.PERCENT: "divided by 100",
        LiveTransform.COUNT: "raw",
        LiveTransform.FLAG: "raw 0/1",
        LiveTransform.DISTANCE: "raw",
    }
    contract = (
        Path(__file__).resolve().parents[2] / "docs" / "environment-contract.md"
    ).read_text()
    _, _, after = contract.partition("### Live readings")
    table, _, _ = after.partition("\nEvery one of these")
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in table.splitlines()
        # The header names the class rather than a field, so it looks like a row
        # and is not one.
        if line.startswith("| `") and not line.startswith("| `Main` field")
    ]

    assert len(rows) == len(LIVE_FIELDS), "one documented row per declared field"
    for row, live in zip(rows, LIVE_FIELDS, strict=True):
        wire, unit, transform, features = row
        assert wire == f"`{live.wire}`"
        assert unit == live.unit
        # The distance row spells its sentinel handling out after the transform
        # name; every other row is the transform and nothing else.
        assert transform.startswith(how[live.transform])
        assert features == ", ".join(f"`{name}`" for name in live.features)
