from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from tower_rl.experiment.comparison import (
    bootstrap_difference,
    cohens_d,
    compare,
    interleave_schedule,
    iqm,
    required_episodes,
    stratified_bootstrap,
    stratified_bootstrap_difference,
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


def test_the_interquartile_mean_is_the_middle_half() -> None:
    """A quarter trimmed off each end, by count, as `rliable` trims it."""
    assert iqm([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)
    assert iqm([1, 2, 3, 4]) == pytest.approx(2.5)
    # Too short to trim anything: every value is in the middle half.
    assert iqm([7, 3]) == pytest.approx(5.0)
    assert iqm([9]) == pytest.approx(9.0)


def test_the_trim_follows_scipy_rather_than_an_exact_half() -> None:
    """`int(n * 0.25)` off each end, which is `scipy.stats.trim_mean(x, 0.25)`.

    A sample whose size is not a multiple of four cannot have exactly a quarter
    dropped from each end, and the convention resolves that by dropping fewer,
    not by interpolating a quantile. Pinned at two such sizes because the
    alternative readings differ there and agree at the sizes above: this is what
    keeps a number here and a number from `rliable` the same number.
    """
    # n=7: one off each end, five kept - not the three or four an exact half
    # would keep.
    seven = [1, 2, 3, 4, 5, 6, 7]
    assert iqm(seven) == pytest.approx(sum(seven[1:6]) / 5)
    assert iqm(seven) == pytest.approx(4.0)

    # n=14: three off each end, eight kept, which is 57% of the sample.
    fourteen = list(range(1, 15))
    assert iqm(fourteen) == pytest.approx(sum(fourteen[3:11]) / 8)
    assert iqm(fourteen) == pytest.approx(7.5)


def test_the_interquartile_mean_ignores_the_tails_the_mean_chases() -> None:
    """One runaway episode moves a mean and must not move the IQM."""
    ordinary = [5, 5, 6, 6, 6, 7, 7, 8]
    outlier = [*ordinary[:-1], 400]

    assert iqm(outlier) == pytest.approx(iqm(ordinary))
    assert sum(outlier) / len(outlier) > sum(ordinary) / len(ordinary) + 40


def test_an_empty_sample_has_no_interquartile_mean() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        iqm([])


def test_the_stratified_interval_covers_a_known_centre() -> None:
    """A sample drawn around a known centre must bracket it."""
    generator = random.Random(7)
    strata = {
        f"actor-{index}": [generator.gauss(6.0, 1.2) for _ in range(40)] for index in range(4)
    }

    point, low, high = stratified_bootstrap(strata, resamples=2_000, seed=3)

    assert low < 6.0 < high
    assert low < point < high
    # An interval this wide would cover anything; it has to be informative.
    assert high - low < 1.0


def test_the_stratified_interval_separates_a_known_difference() -> None:
    """Two arms two waves apart, each stratified by actor, must not overlap."""
    generator = random.Random(11)
    weak = {f"a{index}": [generator.gauss(5.0, 1.0) for _ in range(30)] for index in range(3)}
    strong = {f"a{index}": [generator.gauss(7.0, 1.0) for _ in range(30)] for index in range(3)}

    weak_point, weak_low, weak_high = stratified_bootstrap(weak, resamples=2_000, seed=1)
    strong_point, strong_low, strong_high = stratified_bootstrap(strong, resamples=2_000, seed=1)

    assert weak_point == pytest.approx(5.0, abs=0.3)
    assert strong_point == pytest.approx(7.0, abs=0.3)
    assert weak_high < strong_low, "a two-wave difference is visible at this n"


def test_the_bootstrap_resamples_within_strata_rather_than_across_them() -> None:
    """Each stratum keeps its own size in every resample.

    Two strata of different sizes, each internally constant: resampling within
    them can only ever redraw the same values, so the estimate cannot move. A
    flat resample of the pool would mix the strata in varying proportions and
    produce a non-degenerate interval, which is exactly the thing that would
    misstate the precision a design earns.
    """
    strata = {"a": [0.0] * 30, "b": [10.0] * 10}

    point, low, high = stratified_bootstrap(
        strata, statistic=lambda values: sum(values) / len(values), resamples=500, seed=1
    )

    assert point == pytest.approx(2.5)
    assert low == pytest.approx(2.5) and high == pytest.approx(2.5)


def test_a_stratum_may_not_be_empty_and_a_bootstrap_needs_strata() -> None:
    with pytest.raises(ValueError, match="at least one stratum"):
        stratified_bootstrap({})
    with pytest.raises(ValueError, match="no values"):
        stratified_bootstrap({"a": [1.0], "b": []})


def test_the_pairwise_difference_is_exact_when_every_stratum_is_constant() -> None:
    """A hand-checkable gap: the IQM of each pool is arithmetic, and so is their difference.

    Left pools eight 6s and eight 8s; the trim drops four from each end and
    leaves [6, 6, 6, 6, 8, 8, 8, 8], an IQM of 7.0. Right is sixteen 5s, an IQM
    of 5.0. The gap is 2.0, and because resampling within a constant stratum can
    only redraw the same value, every resample reproduces it - so the interval
    is the point, and any leakage between the strata or between the arms would
    show up as an interval that moved.
    """
    left = {"a": [6.0] * 8, "b": [8.0] * 8}
    right = {"a": [5.0] * 8, "b": [5.0] * 8}

    difference, low, high = stratified_bootstrap_difference(
        left, right, resamples=200, seed=1
    )

    assert difference == pytest.approx(2.0)
    assert low == pytest.approx(2.0) and high == pytest.approx(2.0)


def test_the_pairwise_interval_covers_a_known_interquartile_gap() -> None:
    """Two arms two waves apart: the interval brackets the gap and excludes zero."""
    generator = random.Random(13)
    weak = {f"a{index}": [generator.gauss(5.0, 1.0) for _ in range(30)] for index in range(3)}
    strong = {f"a{index}": [generator.gauss(7.0, 1.0) for _ in range(30)] for index in range(3)}

    difference, low, high = stratified_bootstrap_difference(
        strong, weak, resamples=2_000, seed=3
    )

    assert difference == pytest.approx(2.0, abs=0.4)
    assert low < 2.0 < high, "the interval must cover the gap it was drawn around"
    assert low > 0.0, "a two-wave gap at this n must exclude zero"


def test_the_pairwise_interval_does_not_separate_two_samples_of_one_arm() -> None:
    generator = random.Random(17)
    left = {f"a{index}": [generator.gauss(6.0, 1.2) for _ in range(30)] for index in range(3)}
    right = {f"a{index}": [generator.gauss(6.0, 1.2) for _ in range(30)] for index in range(3)}

    _, low, high = stratified_bootstrap_difference(left, right, resamples=2_000, seed=5)

    assert low < 0.0 < high, "identical arms must be indistinguishable"


def test_the_pairwise_difference_resamples_each_arm_within_its_own_strata() -> None:
    """One arm's strata are its own: a lopsided arm keeps its proportions.

    The left arm's two strata differ in size and in value, so a flat resample of
    its pool would mix them in varying proportions and widen the difference. Here
    they are constant, so a stratified resample cannot move - and the mean
    statistic makes the proportions visible: 30 zeros and 10 tens is 2.5, not 5.
    """
    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    left = {"a": [0.0] * 30, "b": [10.0] * 10}
    right = {"only": [1.0] * 12}

    difference, low, high = stratified_bootstrap_difference(
        left, right, mean, resamples=500, seed=1
    )

    assert difference == pytest.approx(1.5)
    assert low == pytest.approx(1.5) and high == pytest.approx(1.5)


def test_the_pairwise_difference_is_determined_by_its_seed() -> None:
    left = {f"a{index}": [4.0, 5.0, 6.0, 7.0, 8.0] for index in range(2)}
    right = {f"a{index}": [3.0, 4.0, 5.0, 6.0, 7.0] for index in range(2)}

    first = stratified_bootstrap_difference(left, right, resamples=400, seed=2)
    again = stratified_bootstrap_difference(left, right, resamples=400, seed=2)
    other = stratified_bootstrap_difference(left, right, resamples=400, seed=9)

    assert first == again
    assert first[0] == other[0], "the observed difference does not depend on the seed"
    assert first[1:] != other[1:], "the interval is a resample and does depend on it"


def test_a_pairwise_difference_needs_strata_with_values_on_both_sides() -> None:
    with pytest.raises(ValueError, match="at least one stratum"):
        stratified_bootstrap_difference({}, {"a": [1.0]})
    with pytest.raises(ValueError, match="no values"):
        stratified_bootstrap_difference({"a": [1.0]}, {"b": []})
