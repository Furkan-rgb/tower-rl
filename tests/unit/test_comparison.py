from __future__ import annotations

import random

import pytest

from tower_rl.application.comparison import (
    bootstrap_difference,
    cohens_d,
    compare,
    interleave_schedule,
    required_episodes,
)


def test_every_arm_gets_its_full_budget() -> None:
    schedule = interleave_schedule(("a", "b", "c"), episodes_per_arm=10, block=3, seed=1)

    assert len(schedule) == 30
    for arm in ("a", "b", "c"):
        assert schedule.count(arm) == 10


def test_arms_are_interleaved_rather_than_run_to_completion() -> None:
    """Running one arm then the next confounds the arm with whatever drifted."""
    schedule = interleave_schedule(("a", "b"), episodes_per_arm=20, block=5, seed=1)

    first_half = schedule[:20]
    # A sequential schedule would put twenty of one arm in the first half.
    assert 5 <= first_half.count("a") <= 15
    assert 5 <= first_half.count("b") <= 15


def test_no_arm_is_systematically_first() -> None:
    leaders = {interleave_schedule(("a", "b"), 10, block=2, seed=seed)[0] for seed in range(20)}

    assert leaders == {"a", "b"}


def test_a_schedule_needs_arms_and_a_budget() -> None:
    with pytest.raises(ValueError, match="at least one arm"):
        interleave_schedule((), 5)
    with pytest.raises(ValueError, match="must be positive"):
        interleave_schedule(("a",), 0)


def test_a_real_difference_separates_from_zero() -> None:
    generator = random.Random(7)
    left = [round(generator.gauss(12.0, 1.2)) for _ in range(40)]
    right = [round(generator.gauss(9.8, 1.2)) for _ in range(40)]

    difference, low, high = bootstrap_difference(left, right, iterations=2000, seed=3)

    assert difference > 1.5
    assert low > 0.0, "a two-wave difference at n=40 must exclude zero"
    assert high > low


def test_noise_is_not_reported_as_a_difference() -> None:
    generator = random.Random(11)
    left = [round(generator.gauss(9.8, 1.2)) for _ in range(25)]
    right = [round(generator.gauss(9.8, 1.2)) for _ in range(25)]

    _, low, high = bootstrap_difference(left, right, iterations=2000, seed=5)

    assert low < 0.0 < high, "identical arms must be indistinguishable"


def test_comparison_states_separation_rather_than_a_verdict() -> None:
    arms = {"scripted": [10] * 20 + [9] * 5, "random": [2] * 20 + [3] * 5}

    (difference,) = compare(arms, seed=1)

    assert difference.left == "random" and difference.right == "scripted"
    assert difference.separated
    assert "separated" in difference.describe()
    assert difference.left_episodes == 25 and difference.right_episodes == 25


def test_an_interval_containing_zero_is_not_a_claim_of_equality() -> None:
    arms = {"a": [10, 9, 11, 10, 9, 10], "b": [10, 10, 9, 11, 10, 9]}

    (difference,) = compare(arms, seed=1)

    assert not difference.separated
    assert "indistinguishable" in difference.describe()


def test_effect_size_has_a_sign_and_a_scale() -> None:
    assert cohens_d([12] * 10, [10] * 10) == 0.0  # no within-arm variance
    positive = cohens_d([12, 13, 11, 12], [10, 9, 11, 10])
    assert positive > 1.0
    assert cohens_d([10, 9, 11, 10], [12, 13, 11, 12]) == pytest.approx(-positive)


def test_two_episodes_are_the_minimum_comparable_sample() -> None:
    with pytest.raises(ValueError, match="at least two episodes"):
        bootstrap_difference([10], [9, 10])


def test_required_episodes_matches_the_measured_protocol() -> None:
    """M1B-E006/E008: sd 1.26 means about 23 episodes per arm for one wave."""
    assert required_episodes(1.26, 1.0) == pytest.approx(25, abs=3)
    assert required_episodes(1.26, 2.0) < required_episodes(1.26, 1.0)
    assert required_episodes(1.26, 0.5) > 80
