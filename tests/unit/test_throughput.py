from __future__ import annotations

from typing import Any

import pytest

from tower_rl.experiment.throughput import throughput


def _episode(
    *,
    valid: bool = True,
    decisions: int = 100,
    final_wave: int = 20,
    elapsed: float = 25.0,
    advance_wall: float = 17.0,
    round_trip: float = 22.0,
    purchases: int = 10,
    purchase_round_trip: float = 0.6,
) -> dict[str, Any]:
    return {
        "valid": valid,
        "final_wave": final_wave,
        "decisions": decisions,
        "advances": decisions + 5,
        "purchases": purchases,
        "frames": 1800,
        "budgeted_game_ms": 180_000.0,
        "round_ms": 181_800.0,
        "advance_wall_seconds": advance_wall,
        "advance_round_trip_seconds": round_trip,
        "advance_round_trip_cpu_seconds": 0.5,
        "purchase_round_trip_seconds": purchase_round_trip,
        "elapsed_wall_seconds": elapsed,
    }


def _record(episodes: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    valid = sum(1 for episode in episodes if episode["valid"])
    return {
        "episodes": episodes,
        "valid_episodes": valid,
        "invalid_episodes": len(episodes) - valid,
        "invalid_by_reason": {"observation_invalid": len(episodes) - valid},
        "wall_seconds": wall_seconds,
        "total_policy_seconds": 0.01 * sum(e["decisions"] for e in episodes),
    }


def test_a_decision_is_split_into_advance_transport_purchase_policy_and_host() -> None:
    report = throughput({"emulator-5556": _record([_episode(), _episode()], 56.0)})

    split = report["decision_split_ms"]
    assert split["advance"] == pytest.approx(170.0)
    # Round trip 22 s against 17 s of bridge wall over 100 decisions.
    assert split["transport"] == pytest.approx(50.0)
    assert split["purchase"] == pytest.approx(6.0)
    assert split["policy"] == pytest.approx(10.0)
    # 25 s in the episode, less 22 s of advances and 0.6 s of purchases, less policy.
    assert split["other_host"] == pytest.approx(14.0)
    assert sum(split.values()) == pytest.approx(report["readings"]["decision_ms"]["mean"])
    readings = report["readings"]
    assert readings["transport_ms_per_advance"]["mean"] == pytest.approx(5000 / 105, abs=1e-3)
    assert readings["transport_cpu_ms"]["mean"] == pytest.approx(5.0)
    assert report["purchase_round_trip_ms"] == pytest.approx(60.0)
    # Two 25 s episodes in 56 s of actor wall time: 3 s outside each.
    assert report["boundary_seconds_per_episode"] == pytest.approx(3.0)


def test_frames_per_second_and_the_fingerprint_are_read_per_episode() -> None:
    report = throughput({"emulator-5556": _record([_episode(), _episode()], 50.0)})

    actor = report["actors"]["emulator-5556"]
    assert actor["frames_per_advance_second"] == pytest.approx(1800 / 17.0, abs=1e-3)
    assert actor["frames_per_episode_second"] == pytest.approx(72.0)
    assert report["fleet_frames_per_episode_second"] == pytest.approx(72.0)
    fingerprint = report["fingerprint"]
    assert fingerprint["final_wave"]["mean"] == 20.0
    assert fingerprint["final_wave_counts"] == {"20": 2}
    assert fingerprint["decisions_per_wave"]["mean"] == pytest.approx(5.0)
    assert fingerprint["round_clock_ratio"]["mean"] == pytest.approx(180_000 / 181_800, abs=1e-4)


def test_invalid_episodes_are_counted_and_never_measured() -> None:
    slow_and_broken = _episode(valid=False, elapsed=500.0, final_wave=3)
    report = throughput(
        {"emulator-5556": _record([_episode(), slow_and_broken], 530.0)}
    )

    assert report["validity"] == {
        "valid_episodes": 1,
        "invalid_episodes": 1,
        "invalid_by_reason": {"observation_invalid": 1},
        "measured_episodes": 1,
    }
    assert report["readings"]["decision_ms"]["mean"] == pytest.approx(250.0)
    assert report["fingerprint"]["final_wave_counts"] == {"20": 1}


def test_intervals_are_stratified_by_actor_and_repeatable() -> None:
    records = {
        "emulator-5556": _record([_episode(elapsed=24.0), _episode(elapsed=26.0)], 55.0),
        "emulator-5558": _record([_episode(elapsed=30.0), _episode(elapsed=32.0)], 67.0),
    }
    first = throughput(records)
    decision = first["readings"]["decision_ms"]
    assert decision["low"] <= decision["mean"] <= decision["high"]
    assert decision["low"] < decision["high"]
    assert throughput(records) == first
    # Two actors at once: their in-episode frame rates add.
    assert first["fleet_frames_per_episode_second"] == pytest.approx(
        sum(actor["frames_per_episode_second"] for actor in first["actors"].values()), abs=0.01
    )


def test_a_fleet_with_nothing_measurable_reports_only_its_validity() -> None:
    report = throughput(
        {"emulator-5556": _record([_episode(valid=False)], 30.0)}
    )

    assert report["validity"]["measured_episodes"] == 0
    assert "readings" not in report
    assert report["actors"]["emulator-5556"]["measured_episodes"] == 0
