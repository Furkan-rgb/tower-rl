"""The late-game build arms: one shared spread opening, then a build from wave 30."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    EARLY_GAME_ROWS,
    LATE_GAME_BUILDS,
    LATE_GAME_WAVE,
    BindsRowNames,
    CheapestFirstPolicy,
    LateGameBuildPolicy,
    TurtlePolicy,
    WaitOnlyPolicy,
)


@dataclass(frozen=True)
class Label:
    family: str
    index: int
    name: str


#: The game's row labels for the rows the builds name, plus rows they must never
#: buy (family, index, name as `state/records/m3-p004/eval-arm/*.json` reports).
LABELS = tuple(
    Label(*row)
    for row in (
        ("attack", 0, "Damage"),
        ("attack", 1, "Attack Speed"),
        ("attack", 2, "Critical Chance"),
        ("defense", 0, "Health"),
        ("defense", 1, "Health Regen"),
        ("defense", 2, "Defense %"),
        ("defense", 3, "Defense Absolute"),
        ("defense", 4, "Thorn Damage"),
        ("defense", 5, "Lifesteal"),
        ("utility", 0, "Cash Bonus"),
    )
)
ROWS = {label.name: action_index(upgrade_action(label.family, label.index)) for label in LABELS}
OUTSIDE = ("Critical Chance", "Lifesteal", "Cash Bonus")
BEFORE = LATE_GAME_WAVE - 1


def _features(wave: int, costs: dict[str, float]) -> StateFeatures:
    """A state in which exactly the rows named in `costs` are affordable, at that cost."""
    scalars = [0.0] * len(SCALAR_FEATURES)
    scalars[SCALAR_FEATURES.index("wave_log")] = math.log1p(wave)
    flat = [0.0] * (ROW_COUNT * ROW_WIDTH)
    mask = [True] + [False] * ROW_COUNT
    for action in range(1, ROW_COUNT + 1):
        row = {"cost_log": math.log1p(1000.0), "unlocked": 1.0}
        for name, cost in costs.items():
            if ROWS[name] == action:
                row.update(cost_log=math.log1p(cost), available=1.0)
                mask[action] = True
        for feature, value in row.items():
            flat[(action - 1) * ROW_WIDTH + ROW_FEATURES.index(feature)] = value
    return StateFeatures(scalars=tuple(scalars), rows=tuple(flat), mask=tuple(mask))


def _policy(build: str) -> LateGameBuildPolicy:
    policy = LateGameBuildPolicy(build)
    policy.bind_row_names(LABELS)
    policy.initial_state()
    return policy


def _act(policy: LateGameBuildPolicy, wave: int, *affordable: str) -> int:
    """The policy's action in a state where `affordable` rows all cost the same."""
    return policy.act(_features(wave, {name: 100.0 for name in affordable}), None)[0]


# -- the early game --------------------------------------------------------------


@pytest.mark.parametrize("build", LATE_GAME_BUILDS)
def test_the_early_game_buys_the_cheapest_affordable_row_of_the_set(build: str) -> None:
    state = _features(
        BEFORE,
        {"Health": 50.0, "Attack Speed": 20.0, "Thorn Damage": 80.0, "Damage": 30.0},
    )
    assert _policy(build).act(state, None)[0] == ROWS["Attack Speed"]


@pytest.mark.parametrize("build", LATE_GAME_BUILDS)
def test_the_early_game_never_buys_a_row_outside_the_set(build: str) -> None:
    # A row outside the set is cheaper and affordable, and still is not bought.
    cheap = {name: 1.0 for name in OUTSIDE}
    assert (
        _policy(build).act(_features(BEFORE, {**cheap, "Health": 500.0}), None)[0]
        == (ROWS["Health"])
    )
    assert _policy(build).act(_features(BEFORE, cheap), None)[0] == 0


def test_every_row_of_the_set_can_be_bought_in_the_early_game() -> None:
    for name in EARLY_GAME_ROWS:
        assert _act(_policy("split"), BEFORE, name) == ROWS[name]


@pytest.mark.parametrize("build", LATE_GAME_BUILDS)
def test_nothing_affordable_waits_in_both_phases(build: str) -> None:
    assert _act(_policy(build), BEFORE) == 0
    assert _act(_policy(build), LATE_GAME_WAVE) == 0


def test_every_build_plays_the_same_before_the_switch_wave() -> None:
    states = [
        _features(wave, costs)
        for wave in (1, 10, BEFORE)
        for costs in (
            {"Damage": 40.0, "Thorn Damage": 10.0},
            {"Defense Absolute": 5.0, "Health": 9.0},
            {"Defense %": 70.0},
            {},
        )
    ]
    played = {
        build: [_policy(build).act(state, None)[0] for state in states]
        for build in LATE_GAME_BUILDS
    }
    assert len({tuple(actions) for actions in played.values()}) == 1


# -- the late game ---------------------------------------------------------------


@pytest.mark.parametrize("wave", [LATE_GAME_WAVE, LATE_GAME_WAVE + 25])
def test_defense_absolute_buys_only_defense_absolute_from_the_switch(wave: int) -> None:
    policy = _policy("defense-absolute")
    cheaper = {"Damage": 1.0, "Thorn Damage": 2.0, "Defense Absolute": 500.0}
    assert policy.act(_features(wave, cheaper), None)[0] == ROWS["Defense Absolute"]
    assert _act(policy, wave, "Damage", "Thorn Damage", "Health") == 0


@pytest.mark.parametrize("wave", [LATE_GAME_WAVE, LATE_GAME_WAVE + 25])
def test_thorns_buys_only_thorn_damage_from_the_switch(wave: int) -> None:
    policy = _policy("thorns")
    cheaper = {"Damage": 1.0, "Defense Absolute": 2.0, "Thorn Damage": 500.0}
    assert policy.act(_features(wave, cheaper), None)[0] == ROWS["Thorn Damage"]
    assert _act(policy, wave, "Damage", "Defense Absolute", "Health") == 0


def test_the_switch_is_at_the_switch_wave_exactly() -> None:
    state = {"Damage": 1.0, "Thorn Damage": 500.0}
    assert _policy("thorns").act(_features(BEFORE, state), None)[0] == ROWS["Damage"]
    assert _policy("thorns").act(_features(LATE_GAME_WAVE, state), None)[0] == ROWS["Thorn Damage"]


def test_split_alternates_defense_absolute_and_thorns() -> None:
    policy = _policy("split")
    both = ("Defense Absolute", "Thorn Damage", "Damage")
    bought = [_act(policy, LATE_GAME_WAVE, *both) for _ in range(6)]
    assert bought == [ROWS[name] for name in ("Defense Absolute", "Thorn Damage") * 3]


def test_split_waits_for_its_row_rather_than_substituting_the_other() -> None:
    policy = _policy("split")
    assert (
        _act(policy, LATE_GAME_WAVE, "Defense Absolute", "Thorn Damage")
        == (ROWS["Defense Absolute"])
    )
    # Thorns is owed; only Defense Absolute and Damage are affordable.
    assert _act(policy, LATE_GAME_WAVE, "Defense Absolute", "Damage") == 0
    assert _act(policy, LATE_GAME_WAVE, "Defense Absolute", "Damage") == 0
    assert _act(policy, LATE_GAME_WAVE, "Defense Absolute", "Thorn Damage") == ROWS["Thorn Damage"]


def test_split_counts_from_the_switch_not_from_the_opening() -> None:
    policy = _policy("split")
    for _ in range(3):
        assert _act(policy, BEFORE, "Defense Absolute") == ROWS["Defense Absolute"]
    assert (
        _act(policy, LATE_GAME_WAVE, "Defense Absolute", "Thorn Damage")
        == (ROWS["Defense Absolute"])
    )


def test_split_starts_each_episode_even() -> None:
    policy = _policy("split")
    assert (
        _act(policy, LATE_GAME_WAVE, "Defense Absolute", "Thorn Damage")
        == (ROWS["Defense Absolute"])
    )
    policy.initial_state()
    assert (
        _act(policy, LATE_GAME_WAVE, "Defense Absolute", "Thorn Damage")
        == (ROWS["Defense Absolute"])
    )


# -- row names -------------------------------------------------------------------


@pytest.mark.parametrize("missing", EARLY_GAME_ROWS)
def test_a_row_the_game_does_not_name_fails_loudly(missing: str) -> None:
    labels = [label for label in LABELS if label.name != missing]
    with pytest.raises(ValueError, match=missing):
        LateGameBuildPolicy("split").bind_row_names(labels)


def test_an_unbound_build_refuses_to_act() -> None:
    with pytest.raises(RuntimeError, match="row names"):
        LateGameBuildPolicy("split").act(_features(BEFORE, {}), None)


def test_an_unknown_build_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown build"):
        LateGameBuildPolicy("offense")


def test_the_name_binding_protocol_covers_the_turtle_and_the_builds() -> None:
    assert isinstance(TurtlePolicy(), BindsRowNames)
    assert isinstance(LateGameBuildPolicy("thorns"), BindsRowNames)
    assert not isinstance(CheapestFirstPolicy(), BindsRowNames)
    assert not isinstance(WaitOnlyPolicy(), BindsRowNames)
