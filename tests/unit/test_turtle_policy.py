"""The scripted turtle-then-blender build: its rule, its switch, and its row names."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.features import (
    ROW_COUNT,
    ROW_FEATURES,
    ROW_WIDTH,
    SCALAR_FEATURES,
    StateFeatures,
)
from tower_rl.environment.run_actions import action_index, upgrade_action
from tower_rl.environment.run_environment import CadenceConfig, InstrumentedRunEnvironment
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.evaluator import evaluate, to_record
from tower_rl.learning.policies import (
    CASH_PER_WAVE,
    DEFENSE_ABSOLUTE,
    DEFENSE_PERCENT,
    HEALTH,
    KNOCKBACK_CHANCE,
    KNOCKBACK_FORCE,
    ORB_SPEED,
    ORBS,
    THORN_DAMAGE,
    CheapestFirstPolicy,
    TurtlePolicy,
    defense_absolute_margin,
    hits_to_kill,
    next_thorn_breakpoint,
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
    health: float = 1.0,
    base_damage: float = 10.0,
    defense_absolute: float = 100.0,
    thorns: float = 60.0,
    rows: dict[int, dict[str, float]] | None = None,
) -> StateFeatures:
    """A state whose every row is unlocked, unmaxed, priced 100 and unaffordable
    unless `rows` says otherwise (by action index)."""
    scalars = [0.0] * len(SCALAR_FEATURES)
    for name, value in (
        ("wave_log", math.log1p(wave)),
        ("health_fraction", health),
        ("wave_base_damage_log", math.log1p(base_damage)),
        ("defense_absolute_log", math.log1p(defense_absolute)),
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


# -- the margin and the breakpoints ------------------------------------------


def test_hits_to_kill_is_ceil_of_one_over_thorns_capped() -> None:
    assert hits_to_kill(0.0) == 50
    assert hits_to_kill(1.0) == 50
    assert hits_to_kill(5.0) == 20
    assert hits_to_kill(10.0) == 10
    assert hits_to_kill(11.0) == 10
    assert hits_to_kill(21.0) == 5
    assert hits_to_kill(26.0) == 4
    assert hits_to_kill(34.0) == 3
    assert hits_to_kill(51.0) == 2
    assert hits_to_kill(100.0) == 1


def test_the_margin_is_the_heat_up_of_every_hit_before_the_last() -> None:
    assert defense_absolute_margin(0.0) == pytest.approx(1.04**49)
    assert defense_absolute_margin(5.0) == pytest.approx(1.04**19)
    assert defense_absolute_margin(51.0) == pytest.approx(1.04)
    assert defense_absolute_margin(100.0) == 1.0


def test_the_next_breakpoint_is_the_first_one_above_the_reading() -> None:
    assert next_thorn_breakpoint(0.0) == 11.0
    assert next_thorn_breakpoint(10.99) == 11.0
    assert next_thorn_breakpoint(11.0) == 21.0
    assert next_thorn_breakpoint(25.0) == 26.0
    assert next_thorn_breakpoint(33.0) == 34.0
    assert next_thorn_breakpoint(50.0) == 51.0
    assert next_thorn_breakpoint(51.0) is None


# -- row names ----------------------------------------------------------------


def test_row_names_resolve_to_the_recorded_device_slots() -> None:
    expected = {
        HEALTH: _slot("defense", 0),
        DEFENSE_PERCENT: _slot("defense", 2),
        DEFENSE_ABSOLUTE: _slot("defense", 3),
        THORN_DAMAGE: _slot("defense", 4),
        KNOCKBACK_CHANCE: _slot("defense", 6),
        KNOCKBACK_FORCE: _slot("defense", 7),
        ORB_SPEED: _slot("defense", 8),
        ORBS: _slot("defense", 9),
        CASH_PER_WAVE: _slot("utility", 1),
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
        turtle_row_indices([*LABELS, Label("defense", 19, ORBS)])


def test_an_unbound_turtle_refuses_to_act() -> None:
    policy = TurtlePolicy()
    policy.initial_state()
    with pytest.raises(RuntimeError, match="row names"):
        policy.act(_features(), None)


# -- the turtle phase ---------------------------------------------------------


def test_defense_absolute_comes_first_when_it_does_not_hold() -> None:
    # thorns 5 % -> 20 hits -> margin 1.04**19 ~ 2.11; 20 < 2.11 * 10.
    state = _features(
        wave=3, defense_absolute=20.0, thorns=5.0,
        rows=_buyable(DEFENSE_ABSOLUTE, CASH_PER_WAVE, THORN_DAMAGE),
    )
    assert _policy().act(state, None)[0] == ROWS[DEFENSE_ABSOLUTE]


def test_an_unaffordable_defense_absolute_is_saved_for() -> None:
    state = _features(
        wave=3, defense_absolute=20.0, thorns=5.0, rows=_buyable(CASH_PER_WAVE, THORN_DAMAGE)
    )
    assert _policy().act(state, None)[0] == 0


def test_cash_per_wave_is_bought_through_wave_eight_once_defense_holds() -> None:
    rows = _buyable(DEFENSE_ABSOLUTE, CASH_PER_WAVE, THORN_DAMAGE)
    holding = {"defense_absolute": 22.0, "thorns": 5.0}  # 22 >= 2.11 * 10
    assert _policy().act(_features(wave=8, rows=rows, **holding), None)[0] == ROWS[CASH_PER_WAVE]
    assert _policy().act(_features(wave=9, rows=rows, **holding), None)[0] == ROWS[THORN_DAMAGE]


def test_thorns_rise_until_the_last_breakpoint() -> None:
    rows = _buyable(THORN_DAMAGE, DEFENSE_PERCENT, HEALTH)
    assert _policy().act(_features(wave=20, thorns=50.0, rows=rows), None)[0] == ROWS[THORN_DAMAGE]
    past = _policy().act(_features(wave=20, thorns=51.0, rows=rows), None)[0]
    assert past in (ROWS[DEFENSE_PERCENT], ROWS[HEALTH])


def test_past_thorns_the_cheaper_of_defense_percent_and_health_is_bought() -> None:
    rows = _buyable(DEFENSE_PERCENT, HEALTH, costs={DEFENSE_PERCENT: 50.0, HEALTH: 80.0})
    assert _policy().act(_features(wave=20, rows=rows), None)[0] == ROWS[DEFENSE_PERCENT]
    rows = _buyable(DEFENSE_PERCENT, HEALTH, costs={DEFENSE_PERCENT: 90.0, HEALTH: 80.0})
    assert _policy().act(_features(wave=20, rows=rows), None)[0] == ROWS[HEALTH]


def test_a_maxed_row_is_passed_over() -> None:
    rows = _buyable(THORN_DAMAGE, HEALTH, costs={HEALTH: 50.0})
    rows[ROWS[THORN_DAMAGE]]["maxed"] = 1.0
    rows[ROWS[THORN_DAMAGE]]["available"] = 0.0
    assert _policy().act(_features(wave=20, thorns=30.0, rows=rows), None)[0] == ROWS[HEALTH]


def test_damage_is_never_bought_however_cheap() -> None:
    rows = {DAMAGE: {"available": 1.0, "cost_log": math.log1p(1.0)}}
    assert _policy().act(_features(wave=20, rows=rows), None)[0] == 0


# -- the switch -----------------------------------------------------------------


def _failing(wave: int, **kwargs: Any) -> StateFeatures:
    return _features(wave=wave, defense_absolute=1.0, thorns=5.0, **kwargs)


def test_defense_failing_at_two_consecutive_wave_starts_switches_to_blender() -> None:
    policy = _policy()
    rows = _buyable(DEFENSE_ABSOLUTE, KNOCKBACK_CHANCE, THORN_DAMAGE)
    assert policy.act(_failing(10, rows=rows), None)[0] == ROWS[DEFENSE_ABSOLUTE]
    assert policy.switch_wave is None
    choice = policy.act(_failing(11, rows=rows), None)[0]
    assert policy.switch_wave == 11
    # Blender buys no more Defense Absolute: Thorns (5 %) comes first.
    assert choice == ROWS[THORN_DAMAGE]
    assert policy.episode_detail == {"switch_wave": 11}


def test_a_wave_where_defense_held_breaks_the_run_of_failures() -> None:
    policy = _policy()
    policy.act(_failing(10), None)
    policy.act(_features(wave=11, defense_absolute=100.0, thorns=5.0), None)
    policy.act(_failing(12), None)
    assert policy.switch_wave is None


def test_failures_must_be_at_consecutive_waves() -> None:
    policy = _policy()
    policy.act(_failing(10), None)
    policy.act(_failing(12), None)
    assert policy.switch_wave is None


def test_only_the_first_decision_of_a_wave_is_its_start() -> None:
    policy = _policy()
    policy.act(_features(wave=10, defense_absolute=100.0, thorns=5.0), None)
    policy.act(_failing(10), None)  # mid-wave: not a wave start
    policy.act(_failing(11), None)
    assert policy.switch_wave is None


def test_health_below_the_threshold_switches_at_once() -> None:
    policy = _policy()
    policy.act(_features(wave=7, health=0.81), None)
    assert policy.switch_wave is None
    policy.act(_features(wave=7, health=0.79), None)
    assert policy.switch_wave == 7


def test_the_switch_is_one_way_and_reset_per_episode() -> None:
    policy = _policy()
    policy.act(_features(wave=7, health=0.5), None)
    policy.act(_features(wave=8, health=1.0), None)
    assert policy.switch_wave == 7
    policy.initial_state()
    assert policy.switch_wave is None
    assert policy.episode_detail == {"switch_wave": None}


# -- the blender phase -------------------------------------------------------------


def _blending() -> TurtlePolicy:
    policy = _policy()
    policy.act(_features(wave=30, health=0.5), None)
    return policy


def test_blender_raises_thorns_to_the_last_breakpoint_first() -> None:
    rows = _buyable(THORN_DAMAGE, KNOCKBACK_CHANCE, costs={KNOCKBACK_CHANCE: 1.0})
    state = _features(wave=30, thorns=40.0, rows=rows)
    assert _blending().act(state, None)[0] == ROWS[THORN_DAMAGE]


def test_blender_buys_the_cheapest_affordable_knockback_or_orb_row() -> None:
    costs = {KNOCKBACK_CHANCE: 90.0, KNOCKBACK_FORCE: 70.0, ORBS: 80.0, ORB_SPEED: 60.0}
    rows = _buyable(KNOCKBACK_CHANCE, KNOCKBACK_FORCE, ORBS, HEALTH, costs=costs)
    # Orb Speed is cheapest but unaffordable, so the cheapest affordable one wins.
    rows[ROWS[ORB_SPEED]] = {"cost_log": math.log1p(60.0)}
    assert _blending().act(_features(wave=30, rows=rows), None)[0] == ROWS[KNOCKBACK_FORCE]


def test_blender_falls_back_to_health_and_defense_percent() -> None:
    rows = _buyable(HEALTH, DEFENSE_PERCENT, DEFENSE_ABSOLUTE, costs={HEALTH: 10.0})
    assert _blending().act(_features(wave=30, rows=rows), None)[0] == ROWS[HEALTH]


# -- the episode record ------------------------------------------------------------


@dataclass
class _SayingPolicy(CheapestFirstPolicy):
    """Any policy with an `episode_detail`: the record carries it per episode."""

    episodes: int = 0

    def initial_state(self) -> None:
        self.episodes += 1

    @property
    def episode_detail(self) -> dict[str, Any]:
        return {"switch_wave": self.episodes}


def test_the_episode_record_carries_what_the_policy_said() -> None:
    environment = InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=1.0),  # type: ignore[arg-type]
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )
    report = evaluate(environment, _SayingPolicy(), episodes=2, profile_id="fake-profile-v1")
    episodes = to_record(report)["episodes"]
    assert [episode["policy_detail"] for episode in episodes] == [
        {"switch_wave": 1},
        {"switch_wave": 2},
    ]


def test_a_policy_that_says_nothing_adds_nothing_to_the_record() -> None:
    environment = InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=1.0),  # type: ignore[arg-type]
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )
    report = evaluate(environment, CheapestFirstPolicy(), episodes=1, profile_id="fake-profile-v1")
    assert "policy_detail" not in to_record(report)["episodes"][0]
