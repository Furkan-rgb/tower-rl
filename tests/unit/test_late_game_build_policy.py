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
    ORBS_PRICE_CEILING,
    BindsRowNames,
    BlenderBuildPolicy,
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
        ("defense", 6, "Knockback Chance"),
        ("defense", 7, "Knockback Force"),
        ("defense", 8, "Orb Speed"),
        ("defense", 9, "Orbs"),
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
    policy = BlenderBuildPolicy() if build == "blender" else LateGameBuildPolicy(build)
    policy.bind_row_names(LABELS)
    policy.initial_state()
    return policy


def _act(policy: LateGameBuildPolicy, wave: int, *affordable: str) -> int:
    """The policy's action in a state where `affordable` rows all cost the same."""
    return policy.act(_features(wave, {name: 100.0 for name in affordable}), None)[0]


# -- the early game --------------------------------------------------------------


#: Every build, the blender included: all of them share the opening.
EVERY_BUILD = (*LATE_GAME_BUILDS, "blender")
BLENDER_ROWS = ("Orbs", "Orb Speed", "Knockback Chance", "Knockback Force")


@pytest.mark.parametrize("build", EVERY_BUILD)
def test_the_early_game_buys_the_cheapest_affordable_row_of_the_set(build: str) -> None:
    state = _features(
        BEFORE,
        {"Health": 50.0, "Attack Speed": 20.0, "Thorn Damage": 80.0, "Damage": 30.0},
    )
    assert _policy(build).act(state, None)[0] == ROWS["Attack Speed"]


@pytest.mark.parametrize("build", EVERY_BUILD)
def test_the_early_game_never_buys_a_row_outside_the_set(build: str) -> None:
    # A row outside the set is cheaper and affordable, and still is not bought;
    # the blender's own rows are outside it too.
    cheap = {name: 1.0 for name in (*OUTSIDE, *BLENDER_ROWS)}
    assert (
        _policy(build).act(_features(BEFORE, {**cheap, "Health": 500.0}), None)[0]
        == (ROWS["Health"])
    )
    assert _policy(build).act(_features(BEFORE, cheap), None)[0] == 0


def test_every_row_of_the_set_can_be_bought_in_the_early_game() -> None:
    for name in EARLY_GAME_ROWS:
        assert _act(_policy("split"), BEFORE, name) == ROWS[name]


@pytest.mark.parametrize("build", EVERY_BUILD)
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
        build: [_policy(build).act(state, None)[0] for state in states] for build in EVERY_BUILD
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


# -- the blender -----------------------------------------------------------------


def _priced(
    wave: int, cash: float, prices: dict[str, float], maxed: tuple[str, ...] = ()
) -> StateFeatures:
    """A state with those row prices, in which a row is affordable iff it costs at most `cash`."""
    scalars = [0.0] * len(SCALAR_FEATURES)
    scalars[SCALAR_FEATURES.index("wave_log")] = math.log1p(wave)
    flat = [0.0] * (ROW_COUNT * ROW_WIDTH)
    mask = [True] + [False] * ROW_COUNT
    for name, action in ROWS.items():
        price = prices.get(name, 1000.0)
        row = {"cost_log": math.log1p(price), "unlocked": 1.0, "maxed": float(name in maxed)}
        mask[action] = name in prices and name not in maxed and price <= cash
        for feature, value in row.items():
            flat[(action - 1) * ROW_WIDTH + ROW_FEATURES.index(feature)] = value
    return StateFeatures(scalars=tuple(scalars), rows=tuple(flat), mask=tuple(mask))


def _play(
    policy: LateGameBuildPolicy,
    prices: dict[str, list[float]],
    steps: int,
    maxed: tuple[str, ...] = (),
) -> list[str]:
    """What the policy buys over `steps` decisions with cash to spare, each row's
    price moving to the next of its list once bought (the game prices by level)."""
    level = dict.fromkeys(prices, 0)
    bought = []
    for _ in range(steps):
        now = {name: costs[level[name]] for name, costs in prices.items()}
        index = policy.act(_priced(LATE_GAME_WAVE, 1e9, now, maxed), None)[0]
        name = next(name for name, action in ROWS.items() if action == index)
        level[name] += 1
        bought.append(name)
    return bought


def test_the_blender_splits_cash_evenly_between_defense_absolute_and_orbs() -> None:
    # Orbs first on the tie; then Defense Absolute until its spend reaches the
    # Orbs' (100 + 100 + 100 = 300), then the next Orbs level.
    bought = _play(
        _policy("blender"),
        {"Defense Absolute": [100.0] * 20, "Orbs": [300.0, 1250.0, 4000.0], "Damage": [1.0] * 20},
        steps=6,
    )
    assert bought == ["Orbs", *["Defense Absolute"] * 3, "Orbs", "Defense Absolute"]


@pytest.mark.parametrize(
    ("orbs", "maxed"),
    [([300.0], ("Orbs",)), ([ORBS_PRICE_CEILING + 1.0], ())],
    ids=["orbs-maxed", "orbs-above-ceiling"],
)
def test_support_rows_are_blender_spend_alternating_with_defense_absolute(
    orbs: list[float], maxed: tuple[str, ...]
) -> None:
    # Each support purchase stays below the Defense Absolute spend it follows,
    # so the order alternates only if support spend counts on the blender side.
    bought = _play(
        _policy("blender"),
        {
            "Defense Absolute": [100.0] * 20,
            "Orbs": orbs,
            "Orb Speed": [100.0, 150.0],
            "Knockback Chance": [90.0, 200.0],
            "Knockback Force": [95.0, 210.0],
        },
        steps=6,
        maxed=maxed,
    )
    assert bought == [
        "Knockback Chance",
        "Defense Absolute",
        "Knockback Force",
        "Defense Absolute",
        "Orb Speed",
        "Defense Absolute",
    ]


def test_the_blender_saves_for_orbs_rather_than_buying_defense_absolute() -> None:
    policy = _policy("blender")
    # Orbs is owed (nothing spent yet) and costs more than the cash in hand.
    prices = {"Orbs": 300.0, "Defense Absolute": 100.0, "Orb Speed": 15.0, "Thorn Damage": 10.0}
    assert policy.act(_priced(LATE_GAME_WAVE, 250.0, prices), None)[0] == 0
    assert policy.act(_priced(LATE_GAME_WAVE, 300.0, prices), None)[0] == ROWS["Orbs"]
    # Now Defense Absolute is owed, and it is saved for in turn.
    assert policy.act(_priced(LATE_GAME_WAVE, 99.0, prices), None)[0] == 0


def test_after_the_priced_orbs_the_blender_buys_the_cheapest_support_row() -> None:
    above = ORBS_PRICE_CEILING + 1.0
    support = {"Orb Speed": 26.0, "Knockback Chance": 20.0, "Knockback Force": 27.0}
    # An Orbs level dearer than the ceiling is not saved for.
    state = _priced(LATE_GAME_WAVE, 1e9, {"Orbs": above, "Defense Absolute": 100.0, **support})
    assert _policy("blender").act(state, None)[0] == ROWS["Knockback Chance"]
    # Nor is a maxed Orbs; on a price tie the first listed, Orb Speed, goes first.
    tie = dict.fromkeys(support, 20.0)
    state = _priced(LATE_GAME_WAVE, 1e9, {"Orbs": 300.0, **tie}, maxed=("Orbs",))
    assert _policy("blender").act(state, None)[0] == ROWS["Orb Speed"]


def test_the_blender_saves_for_the_second_orbs_level_but_not_the_third() -> None:
    # The second level costs 1,250 and is saved for; the third costs 4,000 and
    # is not, which is what starved both sides in run 1 of #122.
    second = _priced(LATE_GAME_WAVE, 100.0, {"Orbs": 1250.0, "Orb Speed": 15.0})
    assert _policy("blender").act(second, None)[0] == 0
    third = _priced(LATE_GAME_WAVE, 100.0, {"Orbs": 4000.0, "Orb Speed": 15.0})
    assert _policy("blender").act(third, None)[0] == ROWS["Orb Speed"]


def test_a_side_with_nothing_left_leaves_every_purchase_to_the_other() -> None:
    prices = {"Defense Absolute": 100.0, "Orbs": 300.0}
    blender_done = _priced(LATE_GAME_WAVE, 1e9, prices, maxed=BLENDER_ROWS)
    policy = _policy("blender")
    assert [policy.act(blender_done, None)[0] for _ in range(2)] == [ROWS["Defense Absolute"]] * 2
    defense_done = _priced(LATE_GAME_WAVE, 1e9, prices, maxed=("Defense Absolute",))
    policy = _policy("blender")
    assert [policy.act(defense_done, None)[0] for _ in range(2)] == [ROWS["Orbs"]] * 2


def test_the_blender_never_buys_thorns_or_the_opening_rows_late() -> None:
    cheap = {name: 1.0 for name in (*EARLY_GAME_ROWS, *OUTSIDE)}
    del cheap["Defense Absolute"]
    prices = {**cheap, "Defense Absolute": 500.0, "Orbs": 300.0}
    assert _policy("blender").act(_priced(LATE_GAME_WAVE, 250.0, prices), None)[0] == 0


def test_the_blender_counts_spend_from_the_switch_and_per_episode() -> None:
    policy = _policy("blender")
    # Defense Absolute bought in the opening does not count against the blender.
    for _ in range(3):
        assert _act(policy, BEFORE, "Defense Absolute") == ROWS["Defense Absolute"]
    prices = {"Defense Absolute": 100.0, "Orbs": 300.0}
    assert policy.act(_priced(LATE_GAME_WAVE, 1e9, prices), None)[0] == ROWS["Orbs"]
    policy.initial_state()
    assert policy.act(_priced(LATE_GAME_WAVE, 1e9, prices), None)[0] == ROWS["Orbs"]


# -- row names -------------------------------------------------------------------


@pytest.mark.parametrize("missing", EARLY_GAME_ROWS)
def test_a_row_the_game_does_not_name_fails_loudly(missing: str) -> None:
    labels = [label for label in LABELS if label.name != missing]
    with pytest.raises(ValueError, match=missing):
        LateGameBuildPolicy("split").bind_row_names(labels)


@pytest.mark.parametrize("missing", BLENDER_ROWS)
def test_a_blender_row_the_game_does_not_name_fails_loudly(missing: str) -> None:
    labels = [label for label in LABELS if label.name != missing]
    with pytest.raises(ValueError, match=missing):
        BlenderBuildPolicy().bind_row_names(labels)


def test_an_unbound_build_refuses_to_act() -> None:
    with pytest.raises(RuntimeError, match="row names"):
        LateGameBuildPolicy("split").act(_features(BEFORE, {}), None)


def test_an_unknown_build_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown build"):
        LateGameBuildPolicy("offense")


def test_the_name_binding_protocol_covers_the_turtle_and_the_builds() -> None:
    assert isinstance(TurtlePolicy(), BindsRowNames)
    assert isinstance(LateGameBuildPolicy("thorns"), BindsRowNames)
    assert isinstance(BlenderBuildPolicy(), BindsRowNames)
    assert not isinstance(CheapestFirstPolicy(), BindsRowNames)
    assert not isinstance(WaitOnlyPolicy(), BindsRowNames)
