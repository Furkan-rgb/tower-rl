"""What each actor of a fleet explores at, and that `uniform` is what it was.

The ladder is what run 1's evidence asked for: at a uniform 0.05 the run made
about 1.7 exploratory buys an episode and ~2,300 in its whole post-anneal span,
so essentially no alternative build order was ever played (M2-E002/M2-E004).
The uniform schedule stays exactly what that run collected under, which is why
the first test here is a regression against the arithmetic it used.
"""

from __future__ import annotations

import pytest

from tower_rl.learning.exploration import (
    EXPLORATION_OPTIONS,
    NEAR_GREEDY_EPSILON,
    ExplorationSchedule,
    ape_x_floors,
)

#: The ladder the analysis of run 1 priced, for the seven-instance fleet:
#: `0.4 ** (1 + 7 i / 6)`, mean 0.0870.
LADDER_OF_SEVEN = (0.4, 0.1373399, 0.0471556, 0.0161909, 0.0055591, 0.0019087, 0.0006554)


def old_uniform_epsilon(decisions: int, start: float, end: float, horizon: int) -> float:
    """`TrainingConfig.epsilon` exactly as run 1 evaluated it, kept for the test."""
    fraction = min(1.0, decisions / horizon)
    return start + (end - start) * fraction


@pytest.mark.parametrize("start,end,horizon", [(1.0, 0.05, 10_000), (0.8, 0.02, 77)])
@pytest.mark.parametrize("actor", [0, 3, 6])
def test_uniform_is_the_rate_run_one_collected_at(
    start: float, end: float, horizon: int, actor: int
) -> None:
    """Every actor, every point of the budget: the schedule run 1 drew.

    The default path may not move. A run compared against run 1's curve is
    compared against these numbers, and a silent change in them would make the
    two curves measurements of different experiments.
    """
    schedule = ExplorationSchedule.for_option(
        "uniform",
        actors=7,
        epsilon_start=start,
        epsilon_end=end,
        anneal_decisions=horizon,
    )

    assert schedule.floors == (), "a uniform schedule has no per-actor floor"
    for decisions in (0, 1, 999, horizon - 1, horizon, horizon + 1, 200_000):
        expected = old_uniform_epsilon(decisions, start, end, horizon)
        assert schedule.annealed(decisions) == pytest.approx(expected)
        assert schedule.epsilon_for(actor, decisions) == pytest.approx(expected)


def test_every_actor_of_a_uniform_fleet_is_near_greedy() -> None:
    """They share the one rate, so the near-greedy series is the pooled one."""
    schedule = ExplorationSchedule.for_option("uniform", actors=7)

    assert all(schedule.is_near_greedy(index) for index in range(7))


def test_the_ladder_is_ape_x_s() -> None:
    """`eps_i = eps ** (1 + i / (N - 1) * alpha)`, eps 0.4 and alpha 7."""
    floors = ape_x_floors(7)

    assert floors == pytest.approx(LADDER_OF_SEVEN, rel=1e-4)
    assert list(floors) == sorted(floors, reverse=True)
    assert sum(floors) / 7 == pytest.approx(0.0870, abs=5e-4)


def test_a_fleet_of_one_takes_the_base_rate() -> None:
    """`i / (N - 1)` is undefined for one actor; it is taken as zero.

    The paper never considers a fleet of one, so the ladder degenerates to its
    search end rather than to a division by zero - which is also why `uniform`
    and not `ladder` is what a single-actor run defaults to.
    """
    assert ape_x_floors(1) == pytest.approx((0.4,))

    with pytest.raises(ValueError, match="at least one actor"):
        ape_x_floors(0)


def test_the_anneal_is_the_fleet_s_and_the_floor_is_the_actor_s() -> None:
    """The two compose by max, so a continuation past the anneal is the ladder."""
    schedule = ExplorationSchedule.for_option(
        "ladder", actors=7, epsilon_start=1.0, epsilon_end=0.05, anneal_decisions=10_000
    )

    # Inside the anneal every actor is carried by the fleet's rate, which is
    # above all but the top of the ladder.
    assert schedule.epsilon_for(6, 0) == pytest.approx(1.0)
    assert schedule.epsilon_for(0, 5_000) == pytest.approx(0.525)
    # Past it the anneal holds at 0.05 and the ladder is what is left: the top
    # actors above that floor, the bottom ones far below it.
    assert schedule.epsilon_for(0, 200_000) == pytest.approx(0.4)
    assert schedule.epsilon_for(2, 200_000) == pytest.approx(0.05), "annealed rate wins"
    assert schedule.epsilon_for(6, 200_000) == pytest.approx(0.05), "annealed rate wins"
    # A segment resumed past the anneal with no anneal left to run - which is
    # what a continuation of run 1 is - collects on the ladder alone.
    resumed = ExplorationSchedule.for_option(
        "ladder", actors=7, epsilon_start=0.0, epsilon_end=0.0, anneal_decisions=1
    )
    assert [resumed.epsilon_for(index, 200_000) for index in range(7)] == pytest.approx(
        LADDER_OF_SEVEN, rel=1e-4
    )


def test_the_near_greedy_actors_of_a_ladder_are_the_bottom_of_it() -> None:
    """The ones whose episodes are read as performance rather than as search."""
    schedule = ExplorationSchedule.for_option("ladder", actors=7)

    near_greedy = [index for index in range(7) if schedule.is_near_greedy(index)]
    assert near_greedy == [3, 4, 5, 6]
    assert all(schedule.floors[index] <= NEAR_GREEDY_EPSILON for index in near_greedy)


def test_the_schedule_says_which_option_it_is() -> None:
    """What the run records itself as having collected under."""
    assert EXPLORATION_OPTIONS == ("uniform", "ladder")
    assert ExplorationSchedule.for_option("uniform", actors=4).option == "uniform"
    assert ExplorationSchedule.for_option("ladder", actors=4).option == "ladder"

    with pytest.raises(ValueError, match="unknown exploration option"):
        ExplorationSchedule.for_option("greedy", actors=4)
