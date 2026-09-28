"""The scripted turtle build: Thorns/Defense Absolute alternation, and row names."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from tower_rl.environment.features import (
    ROW_COUNT,
    ROW_FEATURES,
    ROW_WIDTH,
    SCALAR_FEATURES,
    StateFeatures,
)
from tower_rl.environment.run_actions import action_index, upgrade_action
from tower_rl.learning.policies import (
    DEFENSE_ABSOLUTE,
    THORN_DAMAGE,
    TurtlePolicy,
    turtle_row_indices,
)

#: The row labels the device reported, copied from `upgrade_rows` in
#: `state/records/m3-p004/eval-arm/emulator-5568.json` (every eval-arm file of
#: that run reports the same 48). Pinned here because `state/` is not in git;
#: `test_the_pinned_labels_match_the_recorded_file` checks the copy wherever the
#: record is present.
RECORDED_LABELS: tuple[tuple[str, int, str], ...] = (
    ("attack", 0, "Damage"),
    ("attack", 1, "Attack Speed"),
    ("attack", 2, "Critical Chance"),
    ("attack", 3, "Critical Factor"),
    ("attack", 4, "Attack Range"),
    ("attack", 5, "Damage / Meter"),
    ("attack", 6, "Multishot Chance"),
    ("attack", 7, "Multishot Targets"),
    ("attack", 8, "Rapid Fire Chance"),
    ("attack", 9, "Rapid Fire Duration"),
    ("attack", 10, "Bounce Shot Chance"),
    ("attack", 11, "Bounce Shot Targets"),
    ("attack", 12, "Bounce Shot Range"),
    ("attack", 13, "Super Crit Chance"),
    ("attack", 14, "Super Crit Mult"),
    ("attack", 15, "Rend Armor Chance"),
    ("attack", 16, "Rend Armor Mult"),
    ("defense", 0, "Health"),
    ("defense", 1, "Health Regen"),
    ("defense", 2, "Defense %"),
    ("defense", 3, "Defense Absolute"),
    ("defense", 4, "Thorn Damage"),
    ("defense", 5, "Lifesteal"),
    ("defense", 6, "Knockback Chance"),
    ("defense", 7, "Knockback Force"),
    ("defense", 8, "Orb Speed"),
    ("defense", 9, "Orbs"),
    ("defense", 10, "Shockwave Size"),
    ("defense", 11, "Shockwave Frequency"),
    ("defense", 12, "Land Mine Chance"),
    ("defense", 13, "Land Mine Damage"),
    ("defense", 14, "Land Mine Radius"),
    ("defense", 15, "Death Defy"),
    ("defense", 16, "Wall Health"),
    ("defense", 17, "Wall Rebuild"),
    ("utility", 0, "Cash Bonus"),
    ("utility", 1, "Cash / Wave"),
    ("utility", 2, "Coins / Kill Bonus"),
    ("utility", 3, "Coins / Wave"),
    ("utility", 4, "Free Attack Upgrade"),
    ("utility", 5, "Free Defense Upgrade"),
    ("utility", 6, "Free Utility Upgrade"),
    ("utility", 7, "Interest / Wave"),
    ("utility", 8, "Recovery Amount"),
    ("utility", 9, "Max Recovery"),
    ("utility", 10, "Package Chance"),
    ("utility", 11, "Enemy Attack Level Skip"),
    ("utility", 12, "Enemy Health Level Skip"),
)

RECORD = Path(__file__).resolve().parents[2] / "state/records/m3-p004/eval-arm/emulator-5568.json"


@dataclass(frozen=True)
class Label:
    family: str
    index: int
    name: str


LABELS = tuple(Label(*row) for row in RECORDED_LABELS)


def _slot(family: str, index: int) -> int:
    return action_index(upgrade_action(family, index))


DAMAGE = _slot("attack", 0)


def _features(
    *,
    wave: int = 10,
    thorns: float = 0.0,
    rows: dict[int, dict[str, float]] | None = None,
) -> StateFeatures:
    """A state whose every row is unlocked, unmaxed, priced 100 and unaffordable
    unless `rows` says otherwise (by action index)."""
    scalars = [0.0] * len(SCALAR_FEATURES)
    for name, value in (
        ("wave_log", math.log1p(wave)),
        ("health_fraction", 1.0),
        ("thorn_damage_log", math.log1p(thorns)),
    ):
        scalars[SCALAR_FEATURES.index(name)] = value
    flat = [0.0] * (ROW_COUNT * ROW_WIDTH)
    mask = [True] + [False] * ROW_COUNT
    for action in range(1, ROW_COUNT + 1):
        row = {"cost_log": math.log1p(100.0), "unlocked": 1.0, "maxed": 0.0, "available": 0.0}
        row.update((rows or {}).get(action, {}))
        for feature, value in row.items():
            flat[(action - 1) * ROW_WIDTH + ROW_FEATURES.index(feature)] = value
        mask[action] = bool(row["available"])
    return StateFeatures(scalars=tuple(scalars), rows=tuple(flat), mask=tuple(mask))


def _policy() -> TurtlePolicy:
    policy = TurtlePolicy()
    policy.bind_row_names(LABELS)
    policy.initial_state()
    return policy


ROWS = turtle_row_indices(LABELS)


def _buyable(*names: str, costs: dict[str, float] | None = None) -> dict[int, dict[str, float]]:
    return {
        ROWS[name]: {"available": 1.0, "cost_log": math.log1p((costs or {}).get(name, 100.0))}
        for name in names
    }


# -- row names ----------------------------------------------------------------


def test_row_names_resolve_to_the_recorded_device_slots() -> None:
    expected = {
        DEFENSE_ABSOLUTE: _slot("defense", 3),
        THORN_DAMAGE: _slot("defense", 4),
    }
    assert expected == ROWS


@pytest.mark.skipif(not RECORD.is_file(), reason="the device record is not in this checkout")
def test_the_pinned_labels_match_the_recorded_file() -> None:
    recorded = json.loads(RECORD.read_text())["upgrade_rows"]
    assert tuple((row["family"], row["index"], row["name"]) for row in recorded) == RECORDED_LABELS


def test_a_missing_row_name_fails_loudly() -> None:
    labels = [label for label in LABELS if label.name != THORN_DAMAGE]
    with pytest.raises(ValueError, match="Thorn Damage"):
        turtle_row_indices(labels)
    with pytest.raises(ValueError, match="Thorn Damage"):
        TurtlePolicy().bind_row_names(labels)


def test_a_row_named_twice_fails_loudly() -> None:
    with pytest.raises(ValueError, match="two rows"):
        turtle_row_indices([*LABELS, Label("defense", 19, THORN_DAMAGE)])


def test_an_unbound_turtle_refuses_to_act() -> None:
    policy = TurtlePolicy()
    policy.initial_state()
    with pytest.raises(RuntimeError, match="row names"):
        policy.act(_features(), None)


# -- alternation below the breakpoint ------------------------------------------


def test_purchases_alternate_thorns_then_defense_absolute_below_the_breakpoint() -> None:
    rows = _buyable(THORN_DAMAGE, DEFENSE_ABSOLUTE)
    policy = _policy()
    state = _features(thorns=10.0, rows=rows)
    assert policy.act(state, None)[0] == ROWS[THORN_DAMAGE]
    assert policy.act(state, None)[0] == ROWS[DEFENSE_ABSOLUTE]
    assert policy.act(state, None)[0] == ROWS[THORN_DAMAGE]
    assert policy.act(state, None)[0] == ROWS[DEFENSE_ABSOLUTE]


def test_an_unaffordable_row_is_waited_for_rather_than_skipped() -> None:
    # Only Defense Absolute is affordable, but it is not Thorns's turn yet.
    rows = _buyable(DEFENSE_ABSOLUTE)
    policy = _policy()
    state = _features(thorns=10.0, rows=rows)
    assert policy.act(state, None)[0] == 0
    assert policy.act(state, None)[0] == 0
    # Thorns becomes affordable; its turn is still owed.
    rows = _buyable(THORN_DAMAGE, DEFENSE_ABSOLUTE)
    assert policy.act(_features(thorns=10.0, rows=rows), None)[0] == ROWS[THORN_DAMAGE]


# -- the handover at the breakpoint --------------------------------------------


def test_only_defense_absolute_is_bought_from_the_breakpoint_on() -> None:
    rows = _buyable(THORN_DAMAGE, DEFENSE_ABSOLUTE)
    policy = _policy()
    assert policy.act(_features(thorns=34.0, rows=rows), None)[0] == ROWS[DEFENSE_ABSOLUTE]
    assert policy.act(_features(thorns=40.0, rows=rows), None)[0] == ROWS[DEFENSE_ABSOLUTE]


def test_a_maxed_thorns_below_the_breakpoint_goes_straight_to_defense_absolute() -> None:
    rows = _buyable(THORN_DAMAGE, DEFENSE_ABSOLUTE)
    rows[ROWS[THORN_DAMAGE]]["maxed"] = 1.0
    rows[ROWS[THORN_DAMAGE]]["available"] = 0.0
    policy = _policy()
    assert policy.act(_features(thorns=10.0, rows=rows), None)[0] == ROWS[DEFENSE_ABSOLUTE]


def test_a_locked_thorns_below_the_breakpoint_goes_straight_to_defense_absolute() -> None:
    rows = _buyable(DEFENSE_ABSOLUTE)
    rows[ROWS[THORN_DAMAGE]] = {"unlocked": 0.0}
    policy = _policy()
    assert policy.act(_features(thorns=10.0, rows=rows), None)[0] == ROWS[DEFENSE_ABSOLUTE]


def test_a_maxed_defense_absolute_waits() -> None:
    rows = _buyable(THORN_DAMAGE)
    rows[ROWS[DEFENSE_ABSOLUTE]] = {"maxed": 1.0, "unlocked": 1.0}
    policy = _policy()
    assert policy.act(_features(thorns=40.0, rows=rows), None)[0] == 0


def test_no_row_besides_thorns_and_defense_absolute_is_ever_bought() -> None:
    rows = {DAMAGE: {"available": 1.0, "cost_log": math.log1p(1.0)}}
    assert _policy().act(_features(thorns=40.0, rows=rows), None)[0] == 0


# -- reset between episodes ----------------------------------------------------


def test_the_turn_resets_to_thorns_first_at_the_start_of_each_episode() -> None:
    rows = _buyable(THORN_DAMAGE, DEFENSE_ABSOLUTE)
    policy = _policy()
    state = _features(thorns=10.0, rows=rows)
    assert policy.act(state, None)[0] == ROWS[THORN_DAMAGE]
    assert policy.act(state, None)[0] == ROWS[DEFENSE_ABSOLUTE]
    policy.initial_state()
    assert policy.act(state, None)[0] == ROWS[THORN_DAMAGE]
