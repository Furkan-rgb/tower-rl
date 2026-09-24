"""The per-wave gate must see what a final wave cannot, and say what it missed."""

from __future__ import annotations

import math
import random

import pytest

from tower_rl.experiment.wave_statistics import (
    analyse,
    episode_records,
    render,
    to_record,
)

# A wave is worth about thirty game seconds and about nine decisions, and an
# episode ends somewhere near wave nine with a spread of a couple of waves: the
# shape the device runs actually produce (M1B-E006, and the pooled sd of 2.27
# behind issue #22's budget).
WAVE_GAME_MS = 30_000.0
WAVE_DECISIONS = 9.0
MEAN_FINAL_WAVE = 9.0
FINAL_WAVE_SD = 2.27
# Fewer iterations than production: the intervals here only need to be stable,
# and a test that bootstraps four statistics across ten wave indices twice over
# should not cost ten thousand resamples each time.
ITERATIONS = 2_000


def episode(
    generator: random.Random, final_wave: int, *, game_ms_scale: float = 1.0
) -> dict[str, object]:
    """One synthetic episode record, in `evaluator.episode_record`'s shape plus waves."""
    waves = []
    health = 1.0
    cash_log = 0.0
    decisions = 0
    for wave in range(1, final_wave + 1):
        wave_decisions = max(1, round(generator.gauss(WAVE_DECISIONS, 0.8)))
        decisions += wave_decisions
        waves.append(
            {
                "wave": wave,
                # The wave the run died in is a fragment of a wave, not a wave.
                "completed": wave < final_wave,
                "game_ms": generator.gauss(WAVE_GAME_MS, WAVE_GAME_MS * 0.03) * game_ms_scale,
                "decisions": wave_decisions,
                "health_fraction": health,
                "cash_log": cash_log,
            }
        )
        health = max(0.0, health - generator.gauss(0.09, 0.02))
        cash_log += generator.gauss(0.4, 0.05)
    return {
        "episode_index": 0,
        "valid": True,
        "final_wave": final_wave,
        "decisions": decisions,
        "waves": waves,
    }


def arm(seed: int, episodes: int = 25, *, game_ms_scale: float = 1.0) -> list[dict[str, object]]:
    generator = random.Random(seed)
    return [
        episode(
            generator,
            max(2, round(generator.gauss(MEAN_FINAL_WAVE, FINAL_WAVE_SD))),
            game_ms_scale=game_ms_scale,
        )
        for _ in range(episodes)
    ]


def paired_arms(episodes: int = 25, *, game_ms_scale: float) -> tuple[list, list]:
    """Two arms with the *same* final waves, differing only inside each wave.

    The same seed gives both arms the same episode lengths and the same
    within-wave noise, so the only difference is the injected one. That is what
    makes the contrast between the per-wave statistic and the final wave
    statistic a property of the statistics rather than of the draw.
    """
    return arm(7, episodes), arm(7, episodes, game_ms_scale=game_ms_scale)


def statistic(analysis, name):
    return next(item for item in analysis.per_wave if item.statistic == name)


def episode_statistic(analysis, name):
    return next(item for item in analysis.per_episode if item.statistic == name)


def test_identical_arms_are_indistinguishable_and_say_by_how_much() -> None:
    """Same distribution, different draws: intervals straddle zero, floors are small."""
    analysis = analyse("A", arm(1), "B", arm(2), iterations=ITERATIONS)

    game_ms = statistic(analysis, "game_ms")
    assert not game_ms.uncaptured
    assert game_ms.separated_waves == ()
    for point in game_ms.points:
        assert point.difference.low < 0.0 < point.difference.high
    # The floor is what makes "indistinguishable" a bounded claim rather than a
    # claim of equality. At the wave every episode reaches, twenty-five episodes
    # an arm resolve under three percent of a wave's duration; the deepest wave
    # indices, reached by a handful of episodes, resolve far less and say so.
    assert game_ms.points[0].detectable_difference < 0.03 * WAVE_GAME_MS
    assert max(point.detectable_difference for point in game_ms.points) < 0.5 * WAVE_GAME_MS
    assert abs(game_ms.pooled_effect_size) < 0.5
    assert game_ms.detectable_effect_size == pytest.approx(0.7926, abs=0.01)


def test_per_wave_detects_a_shift_the_final_wave_cannot_see() -> None:
    """The whole point: ten percent more game time per wave, same final waves.

    A physics or timing change that slows every wave by ten percent leaves the
    final wave distribution untouched here by construction, and mean final wave
    is blind to it at any n. The per-wave duration statistic finds it at
    twenty-five episodes an arm.
    """
    left, right = paired_arms(game_ms_scale=1.10)
    analysis = analyse("60Hz", left, "120Hz", right, iterations=ITERATIONS)

    game_ms = statistic(analysis, "game_ms")
    compared = [point.wave for point in game_ms.points]
    assert game_ms.separated_waves == tuple(compared)
    first = game_ms.points[0]
    assert first.difference.difference == pytest.approx(-0.10 * WAVE_GAME_MS, rel=0.15)
    # The shift is several multiples of what the sample could resolve.
    assert abs(first.difference.difference) > 3 * first.detectable_difference
    assert game_ms.pooled_effect_size < -2.0

    final_wave = episode_statistic(analysis, "final_wave")
    assert not final_wave.difference.separated
    assert final_wave.difference.difference == pytest.approx(0.0, abs=1e-9)
    # And the blunt instrument states its own blindness rather than reporting
    # equality: at this n it could not have found even a whole wave.
    assert final_wave.detectable_difference > 1.0

    text = render(analysis)
    assert "not captured" not in text
    assert "could detect >=" in text
    assert "no difference" not in text


def test_decisions_per_wave_also_carries_far_less_variance_than_a_final_wave() -> None:
    """Both blunt statistics are reported, and both are blunter than a wave index."""
    left, right = paired_arms(game_ms_scale=1.0)
    analysis = analyse("A", left, "B", right, iterations=ITERATIONS)

    per_wave_floor = statistic(analysis, "decisions").points[0].detectable_difference
    per_episode_floor = episode_statistic(analysis, "decisions").detectable_difference
    assert per_wave_floor < per_episode_floor
    assert episode_statistic(analysis, "final_wave").detectable_difference > 1.0


def test_deep_wave_indices_with_too_few_episodes_are_named_not_dropped() -> None:
    """Fewer episodes reach wave twenty; the report must say so, not stop early."""
    generator = random.Random(3)
    left = [episode(generator, wave) for wave in (5, 6, 7, 8, 20)]
    right = [episode(generator, wave) for wave in (5, 6, 7, 9, 21)]
    analysis = analyse("A", left, "B", right, iterations=ITERATIONS)

    game_ms = statistic(analysis, "game_ms")
    compared = {point.wave for point in game_ms.points}
    sparse = {item.wave: (item.left_episodes, item.right_episodes) for item in game_ms.underpowered}
    # Every wave either arm reached is accounted for exactly once.
    assert compared.isdisjoint(sparse)
    assert compared | set(sparse) == set(range(1, 21))
    assert sparse[20] == (0, 1)
    assert "not compared" in render(analysis)


def test_an_uncaptured_quantity_is_reported_as_uncaptured() -> None:
    """A record without wave rows says so; it does not say the arms agree."""
    records = [
        {"valid": True, "final_wave": 9, "decisions": 80},
        {"valid": True, "final_wave": 7, "decisions": 65},
        {"valid": True, "final_wave": 11, "decisions": 96},
    ]
    analysis = analyse("A", records, "B", records, iterations=ITERATIONS)

    assert all(comparison.uncaptured for comparison in analysis.per_wave)
    assert all(comparison.points == () for comparison in analysis.per_wave)
    assert "not captured by this run's episode records" in render(analysis)
    # The whole-episode statistics still work on today's records.
    assert episode_statistic(analysis, "final_wave").difference.difference == 0.0


def test_episode_records_reads_the_run_actors_report_shape() -> None:
    """`run_actors.py` writes one arm per file (`evaluator.to_record`)."""
    flat = {"episodes": [{"valid": True, "final_wave": 9}, {"valid": False, "final_wave": 1}]}

    assert len(episode_records(flat)) == 1


def test_every_rendered_and_recorded_result_carries_its_blind_spot() -> None:
    """The reporting rule, enforced structurally rather than by convention."""
    left, right = paired_arms(game_ms_scale=1.0)
    record = to_record(analyse("A", left, "B", right, iterations=ITERATIONS))

    waves = [wave for entry in record["per_wave"] for wave in entry["waves"]]
    assert waves
    for item in waves + record["per_episode"]:
        # Zero only where the quantity itself never varies — health is full at
        # the start of wave one in every episode — and zero there is the honest
        # floor rather than a missing one.
        assert item["detectable_difference"] >= 0.0
        assert math.isfinite(item["detectable_difference"])
        assert item["interval"][0] <= item["difference"] <= item["interval"][1]


#: The exact per-wave keys `evaluator.episode_record` promises and
#: `wave_observations` reads. Pinned on both sides so a rename cannot quietly
#: drop the analysis back onto the `uncaptured` path.
WAVE_RECORD_KEYS = {
    "wave", "completed", "game_ms", "decisions", "advances", "health_fraction", "cash_log",
}


def _fake_port_arm(seed: int, episodes: int = 6) -> list[dict[str, object]]:
    """One arm of real episode records, played end to end against the fake port."""
    import json

    from fakes.fake_run_port import FakeRunPort

    from tower_rl.environment.run_environment import CadenceConfig, InstrumentedRunEnvironment
    from tower_rl.environment.run_state import RunStateBuilder
    from tower_rl.learning.evaluator import episode_record, evaluate
    from tower_rl.learning.policies import RandomPolicy

    report = evaluate(
        InstrumentedRunEnvironment(
            port=FakeRunPort(damage_per_second=0.3, seconds_per_wave=4.0),
            builder=RunStateBuilder(profile_id="fake-profile-v1"),
            cadence=CadenceConfig(max_quiet_game_ms=1000),
        ),
        RandomPolicy(seed=seed),
        episodes=episodes,
        profile_id="fake-profile-v1",
    )
    # Through JSON, because that is how a device arm reaches the analysis.
    return [
        json.loads(json.dumps(episode_record(index, summary)))
        for index, summary in enumerate(report.episodes)
    ]


def test_the_environments_own_records_reach_the_per_wave_analysis() -> None:
    """The capture (#24) and the analysis (#22) meet here, not by assumption.

    Two arms are played against the fake port, written through JSON exactly as a
    device run writes them, and compared by the real analysis. Every per-wave
    statistic must be captured and produce points; the `uncaptured` path is what
    this test exists to stay off.
    """
    left = _fake_port_arm(seed=1)
    right = _fake_port_arm(seed=2)

    for record in left + right:
        assert record["waves"], "every episode entered at least one wave"
        assert all(set(wave) == WAVE_RECORD_KEYS for wave in record["waves"])

    analysis = analyse(
        "seed-1",
        episode_records({"episodes": left}),
        "seed-2",
        episode_records({"episodes": right}),
        iterations=ITERATIONS,
    )

    assert {comparison.statistic for comparison in analysis.per_wave} == {
        "game_ms",
        "decisions",
        "health_fraction",
        "cash_log",
    }
    for comparison in analysis.per_wave:
        assert not comparison.uncaptured, f"{comparison.statistic} fell back to uncaptured"
        assert comparison.points, f"{comparison.statistic} produced no per-wave point"
        assert comparison.detectable_effect_size is not None
    # And the report renders the per-wave picture rather than an absence of one.
    assert "not captured" not in render(analysis)
