"""Instance identity, aggregation and failure isolation for the actor fleet.

No emulator, no adb, no bridge: the lifecycle steps are injected, so what is
under test is the plumbing that decides which instance an actor addresses, what
the aggregate says, and what happens to the other actors when one dies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from clone_session import (  # noqa: E402
    CANONICAL_AVD,
    CLONE_AVD,
    CloneError,
    CloneInstance,
    emulator_command,
)
from run_actors import (  # noqa: E402
    ActorOutcome,
    aggregate,
    bridge_host_port,
    health_counters,
    run_actor,
    run_fleet,
)

from tower_rl.application.evaluator import (  # noqa: E402
    EvaluationReport,
    WaveDistribution,
    to_record,
)
from tower_rl.domain.episode import EpisodeSummary, TerminationOutcome  # noqa: E402


def summary(
    *, final_wave: int, valid: bool = True, detail: tuple[str, ...] = (), cut_short: int = 0
) -> EpisodeSummary:
    return EpisodeSummary(
        episode_id="e",
        profile_id="p",
        final_wave=final_wave,
        decisions=10,
        purchases=1,
        termination=(
            TerminationOutcome.GAME_OVER if valid else TerminationOutcome.OBSERVATION_INVALID
        ),
        elapsed_wall_seconds=1.0,
        game_speed=1.0,
        invalid_transitions=0 if valid else 1,
        advances_cut_short=cut_short,
        termination_detail=detail,
        starting_wave=1,
    )


def actor_record(summaries: list[EpisodeSummary]) -> dict[str, object]:
    """A real single-actor record, produced by the evaluator's own reporting."""
    valid = [item for item in summaries if item.valid]
    report = EvaluationReport(
        policy="scripted",
        profile_id="p",
        model_version=0,
        game_speed=1.0,
        valid_episodes=len(valid),
        invalid_episodes=len(summaries) - len(valid),
        distribution=WaveDistribution.of([item.final_wave for item in valid]),
        advances_cut_short=sum(item.advances_cut_short for item in summaries),
        episodes=tuple(summaries),
        episodes_not_started_fresh=sum(1 for item in summaries if item.starting_wave > 1),
    )
    return to_record(report)


def test_instance_index_derives_an_even_console_port_and_its_serial() -> None:
    assert (CloneInstance().console_port, CloneInstance().serial) == (5556, "emulator-5556")
    ports = [CloneInstance(index=index).console_port for index in range(4)]
    assert ports == [5556, 5558, 5560, 5562]
    assert all(port % 2 == 0 for port in ports)
    assert CloneInstance(index=3).serial == "emulator-5562"


def test_each_instance_gets_its_own_bridge_host_port() -> None:
    ports = [bridge_host_port(CloneInstance(index=index)) for index in range(3)]
    assert ports == [47652, 47653, 47654]
    assert len(set(ports)) == 3


def test_the_canonical_evaluation_avd_is_refused() -> None:
    with pytest.raises(CloneError, match="canonical"):
        CloneInstance(index=0, avd=CANONICAL_AVD)
    with pytest.raises(CloneError):
        CloneInstance(index=-1)


def test_instances_share_the_clone_avd_read_only_on_their_own_port() -> None:
    command = emulator_command(
        CloneInstance(index=2),
        binary="emulator",
        renderer="lavapipe",
        snapshot=None,
        read_only=True,
        cores=4,
    )
    assert command[1] == f"@{CLONE_AVD}"
    assert "-read-only" in command
    assert command[command.index("-port") + 1] == "5560"
    assert "-no-snapshot-load" in command


def test_a_writable_instance_is_not_launched_read_only() -> None:
    command = emulator_command(
        CloneInstance(),
        binary="emulator",
        renderer="lavapipe",
        snapshot="home_offline",
        read_only=False,
        cores=8,
    )
    assert "-read-only" not in command
    assert command[command.index("-snapshot") + 1] == "home_offline"


def test_health_counters_come_from_the_per_episode_records() -> None:
    record = actor_record(
        [
            summary(final_wave=7, cut_short=2),
            summary(
                final_wave=3,
                valid=False,
                detail=("the bridge and the host disagree about the decision event",),
            ),
            summary(
                final_wave=4,
                valid=False,
                detail=("advance was not confirmed: stale_or_duplicate",),
                cut_short=1,
            ),
        ]
    )
    assert health_counters(record) == {
        "bridge_event_divergence": 1,
        "stale_or_duplicate": 1,
        "advances_cut_short": 3,
        "episodes_not_started_fresh": 0,
    }


def test_the_aggregate_is_valid_episodes_per_fleet_hour() -> None:
    outcomes = [
        ActorOutcome(
            index=0,
            serial="emulator-5556",
            wall_seconds=1800.0,
            record=actor_record([summary(final_wave=5), summary(final_wave=7, cut_short=1)]),
        ),
        ActorOutcome(
            index=1,
            serial="emulator-5558",
            wall_seconds=1800.0,
            record=actor_record(
                [
                    summary(final_wave=6),
                    summary(final_wave=2, valid=False, detail=("advance: stale_or_duplicate",)),
                ]
            ),
        ),
    ]
    report = aggregate(outcomes, wall_seconds=1800.0)

    assert report["valid_episodes"] == 3
    assert report["invalid_episodes"] == 1
    assert report["valid_episodes_per_hour"] == 6.0
    assert report["health"]["advances_cut_short"] == 1
    assert report["health"]["stale_or_duplicate"] == 1
    assert [actor["valid_episodes_per_hour"] for actor in report["actors"]] == [4.0, 2.0]
    assert report["actors_reporting"] == 2
    assert report["actors_failed"] == 0


def test_one_actor_failing_leaves_the_others_intact_and_still_tears_down() -> None:
    instances = [CloneInstance(index=index) for index in range(3)]
    torn_down: list[str] = []

    def collect(instance: CloneInstance) -> dict[str, object]:
        if instance.index == 1:
            raise RuntimeError("bridge handshake failed")
        return actor_record([summary(final_wave=5 + instance.index)])

    def tear_down(instance: CloneInstance) -> None:
        torn_down.append(instance.serial)

    outcomes = run_fleet(instances, collect, tear_down)
    report = aggregate(outcomes, wall_seconds=3600.0)

    assert sorted(torn_down) == ["emulator-5556", "emulator-5558", "emulator-5560"]
    assert [outcome.index for outcome in outcomes] == [0, 1, 2]
    assert report["actors_failed"] == 1
    assert report["actors_reporting"] == 2
    assert report["valid_episodes"] == 2
    assert report["valid_episodes_per_hour"] == 2.0
    failed = report["actors"][1]
    assert "bridge handshake failed" in failed["failure"]
    assert "valid_episodes" not in failed
    assert [actor["failure"] for actor in report["actors"]] == [None, failed["failure"], None]


def test_a_teardown_failure_is_reported_without_losing_the_episodes() -> None:
    def collect(instance: CloneInstance) -> dict[str, object]:
        return actor_record([summary(final_wave=9)])

    def tear_down(instance: CloneInstance) -> None:
        raise RuntimeError("the overlay is still mounted")

    outcome = run_actor(CloneInstance(), collect, tear_down)

    assert outcome.failure is None
    assert outcome.record is not None
    assert "overlay is still mounted" in (outcome.teardown_failure or "")
