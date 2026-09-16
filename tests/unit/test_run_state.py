from __future__ import annotations

import math

import pytest

from tower_rl.domain.run_actions import (
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
from tower_rl.domain.run_state import (
    MAX_AFFORDABILITY_RATIO,
    RunStateBuilder,
    action_is_allowed,
    validate_transition,
)
from tower_rl.infrastructure.instrumented_bridge import BridgeObservation, UpgradeInventoryEntry

BUILDER = RunStateBuilder(profile_id="tower-play-29.0.3-rooted-v1")


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
    assert state.schema_version == "observation-v1"

    row = state.rows[0]
    assert row.action == upgrade_action("attack", 0)
    assert row.cost_log == pytest.approx(math.log1p(10.0))
    assert row.affordability == pytest.approx(10.0)
    assert row.headroom == pytest.approx(1.0)


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
    assert state.available_actions() == ()
    assert not action_is_allowed(state, WAIT)


def test_affordability_is_clipped_but_cost_is_not_lost() -> None:
    overrides = {("attack", 0): _entry("attack", 0, cost=1.0)}
    state = BUILDER.build(_reading(cash=10_000.0, overrides=overrides), captured_at_monotonic=1.0)

    assert state.rows[0].affordability == pytest.approx(MAX_AFFORDABILITY_RATIO)
    assert state.rows[0].cost_log == pytest.approx(math.log1p(1.0))


def test_impossible_readings_invalidate_rather_than_normalize() -> None:
    state = BUILDER.build(_reading(health=9.0, max_health=5.0), captured_at_monotonic=1.0)

    assert not state.valid
    assert "health fraction outside [0, 1]" in state.invalid_reasons
    # The value is still clamped for downstream safety, never silently trusted.
    assert state.health_fraction == pytest.approx(1.0)


def test_a_short_inventory_is_invalid_and_its_missing_slots_are_masked() -> None:
    reading = _reading()
    truncated = BridgeObservation(
        sequence=reading.sequence,
        lifecycle=reading.lifecycle,
        wave=reading.wave,
        cash=reading.cash,
        health=reading.health,
        max_health=reading.max_health,
        terminal=reading.terminal,
        round_active=reading.round_active,
        game_speed=reading.game_speed,
        play_time=reading.play_time,
        upgrades=reading.upgrades[:10],
    )

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
