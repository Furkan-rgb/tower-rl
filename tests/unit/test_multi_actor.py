"""Instance identity, aggregation and failure isolation for the actor fleet.

No emulator, no adb, no bridge: the lifecycle steps are injected, so what is
under test is the plumbing that decides which instance an actor addresses, what
the aggregate says, and what happens to the other actors when one dies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest
import run_actors
from clone_session import (
    CANONICAL_AVD,
    CLONE_AVD,
    CloneError,
    CloneInstance,
    emulator_command,
)
from run_actors import (
    ActorFailure,
    ActorOutcome,
    aggregate,
    collect_episodes,
    health_counters,
    run_actor,
    run_bridge,
    run_fleet,
    stagger_bring_up,
)

from tower_rl.environment.episode import EpisodeSummary, TerminationOutcome
from tower_rl.learning.evaluator import (
    EvaluationReport,
    WaveDistribution,
    to_record,
)


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
    ports = [CloneInstance(index=index).bridge_host_port for index in range(3)]
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


def bring_up_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, cold: bool = False
) -> tuple[list[str], list[dict[str, object]]]:
    """Record one actor's bring-up and how it asked for it, with the device injected."""
    steps: list[str] = []
    asked: list[dict[str, object]] = []
    instance = CloneInstance()
    output = tmp_path / f"{instance.serial}.json"

    def episode_process(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        steps.append(Path(command[1]).name)
        output.write_text(json.dumps(actor_record([summary(final_wave=4)])))
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_bring_up(target: CloneInstance, renderer: str, **keywords: object) -> str:
        steps.append("bring_up")
        asked.append({"serial": target.serial, "renderer": renderer, **keywords})
        return "restored"

    monkeypatch.setattr(run_actors, "bring_up", fake_bring_up)
    monkeypatch.setattr(run_actors, "require_offline", lambda *_: steps.append("require_offline"))
    monkeypatch.setattr(
        run_actors,
        "require_game_activity",
        lambda *_, **__: bool(steps.append("require_game_activity")),
    )
    monkeypatch.setattr(run_actors, "raise_frame_rate", lambda *_: steps.append("raise_frame_rate"))
    monkeypatch.setattr(run_actors, "run_bridge", lambda command, _: steps.append(command))
    monkeypatch.setattr(run_actors.subprocess, "run", episode_process)

    arguments = argparse.Namespace(
        cold=cold,
        renderer="lavapipe",
        cores=4,
        episodes=2,
        policy="scripted",
        frame_game_ms=100.0,
        max_quiet_game_ms=2000,
        max_episode_wall_seconds=600.0,
        output_directory=tmp_path,
    )
    assert collect_episodes(instance, arguments)["valid_episodes"] == 1
    return steps, asked


def test_an_actor_is_ready_and_verified_offline_before_any_episode_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bring-up is one decision now: restore the pinned snapshot, or cold-start.

    Whichever path it takes, it ends ready and offline, and offline is re-checked
    by interface immediately before the driver starts. The game's activity is
    re-checked there too: nothing has watched it since bring-up returned, and a
    game the guest's Play killed in between has no surface, so the raise that
    follows would fail with no applied rate rather than put the game back.
    """
    steps, asked = bring_up_steps(monkeypatch, tmp_path)

    assert steps == [
        "bring_up",
        "require_offline",
        "require_game_activity",
        "raise_frame_rate",
        "run_episodes.py",
    ]
    assert asked == [
        {
            "serial": "emulator-5556",
            "renderer": "lavapipe",
            "deploy": run_actors.deploy_bridge,
            "read_only": True,
            "cores": 4,
            "force_cold": False,
        }
    ]


def test_an_actor_can_be_made_to_cold_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, asked = bring_up_steps(monkeypatch, tmp_path, cold=True)
    assert asked[0]["force_cold"] is True


def prepare(
    monkeypatch: pytest.MonkeyPatch, *, held: bool
) -> tuple[list[str], list[dict[str, object]]]:
    """Run the pre-fleet preparation against an injected snapshot registry."""
    steps: list[str] = []
    asked: list[dict[str, object]] = []

    def fake_bring_up(target: CloneInstance, renderer: str, **keywords: object) -> str:
        steps.append("bring_up")
        asked.append({"serial": target.serial, **keywords})
        return "cold"

    monkeypatch.setattr(run_actors, "bridge_key", lambda: "abc123")
    monkeypatch.setattr(run_actors, "snapshot_exists", lambda *_: held)
    monkeypatch.setattr(run_actors, "bring_up", fake_bring_up)
    monkeypatch.setattr(
        run_actors, "tear_down_instance", lambda instance: steps.append("tear_down")
    )
    name = run_actors.prepare_pinned_snapshot("lavapipe", 4)
    assert name.endswith("abc123")
    return steps, asked


def test_the_pinned_snapshot_is_prepared_once_on_a_writable_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-only actor cannot save one, so the fleet would otherwise stay cold."""
    steps, asked = prepare(monkeypatch, held=False)

    assert steps == ["bring_up", "tear_down"]
    assert asked == [{"serial": "emulator-5556", "deploy": run_actors.deploy_bridge, "cores": 4}]


def test_nothing_is_prepared_when_the_snapshot_for_this_bridge_is_already_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert prepare(monkeypatch, held=True) == ([], [])


class ConcurrencyRecorder:
    """The highest number of overlapping `enter`/`exit` pairs seen at once."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def exit(self) -> None:
        with self.lock:
            self.current -= 1


def stagger_arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        cold=False,
        renderer="lavapipe",
        cores=4,
        episodes=2,
        policy="scripted",
        frame_game_ms=100.0,
        max_quiet_game_ms=2000,
        max_episode_wall_seconds=600.0,
        output_directory=tmp_path,
    )


def test_bring_ups_are_sequenced_but_collection_still_runs_concurrently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The device defect this fixes: 4 simultaneous cold boots, one never ready.

    Bring-up must never overlap (peak concurrency 1); the run_episodes.py
    subprocess calls that follow bring-up must still run at once, exactly as
    they did before staggering.
    """
    instances = [CloneInstance(index=index) for index in range(3)]
    bring_up_tracker = ConcurrencyRecorder()
    collect_tracker = ConcurrencyRecorder()

    def fake_bring_up(target: CloneInstance, renderer: str, **keywords: object) -> str:
        bring_up_tracker.enter()
        time.sleep(0.02)
        bring_up_tracker.exit()
        return "restored"

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        collect_tracker.enter()
        time.sleep(0.2)
        collect_tracker.exit()
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(actor_record([summary(final_wave=4)])))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run_actors, "bring_up", fake_bring_up)
    monkeypatch.setattr(run_actors, "require_offline", lambda *_: None)
    monkeypatch.setattr(run_actors, "require_game_activity", lambda *_, **__: False)
    monkeypatch.setattr(run_actors, "raise_frame_rate", lambda *_: None)
    monkeypatch.setattr(run_actors.subprocess, "run", fake_run)

    outcomes = run_fleet(
        instances, stagger_bring_up(instances, stagger_arguments(tmp_path)), lambda _: None
    )

    assert bring_up_tracker.peak == 1
    assert collect_tracker.peak == len(instances)
    assert all(outcome.failure is None for outcome in outcomes)


def test_a_slow_boot_does_not_let_the_backstop_overlap_the_next_bring_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The backstop is a per-instance timeout, not a fleet-start deadline.

    On device a 7-instance cold host fleet took ~165s per bring-up against a
    360s backstop measured from fleet start: the window had expired for the
    later instances before their predecessors had even launched, and their
    boots overlapped. Timed from the previous instance's own launch, a boot
    slower than the whole backstop still cannot overlap its successor.
    """
    instances = [CloneInstance(index=index) for index in range(3)]
    monkeypatch.setattr(run_actors, "BRING_UP_STAGGER_BACKSTOP", 0.3)
    bring_up_tracker = ConcurrencyRecorder()

    def fake_bring_up(target: CloneInstance, renderer: str, **keywords: object) -> str:
        bring_up_tracker.enter()
        time.sleep(0.25)
        bring_up_tracker.exit()
        return "restored"

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(actor_record([summary(final_wave=4)])))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run_actors, "bring_up", fake_bring_up)
    monkeypatch.setattr(run_actors, "require_offline", lambda *_: None)
    monkeypatch.setattr(run_actors, "require_game_activity", lambda *_, **__: False)
    monkeypatch.setattr(run_actors, "raise_frame_rate", lambda *_: None)
    monkeypatch.setattr(run_actors.subprocess, "run", fake_run)

    outcomes = run_fleet(
        instances, stagger_bring_up(instances, stagger_arguments(tmp_path)), lambda _: None
    )

    assert bring_up_tracker.peak == 1
    assert all(outcome.failure is None for outcome in outcomes)


def test_an_actor_collects_while_a_peer_is_still_booting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """There is no fleet rendezvous: a ready actor raises and collects at once.

    The barrier this replaces held every ready instance idle until the last
    bring-up concluded — on the refuted reading that a raised peer killed a
    booting one (`M1B-E043`) — and that idle window is when the guest's Play
    installs the update it downloaded and kills the game. So the only thing an
    actor waits for is its own bring-up.
    """
    instances = [CloneInstance(index=index) for index in range(3)]
    events: list[str] = []
    lock = threading.Lock()

    def fake_bring_up(target: CloneInstance, renderer: str, **keywords: object) -> str:
        # The last instance boots slowly, so its peers must not be waiting on it.
        time.sleep(0.4 if target.index == 2 else 0.01)
        with lock:
            events.append(f"up:{target.index}")
        return "restored"

    def fake_raise(target: CloneInstance) -> None:
        with lock:
            events.append(f"raise:{target.index}")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        serial = command[command.index("--serial") + 1]
        with lock:
            events.append(f"episodes:{serial}")
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(actor_record([summary(final_wave=4)])))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run_actors, "bring_up", fake_bring_up)
    monkeypatch.setattr(run_actors, "require_offline", lambda *_: None)
    monkeypatch.setattr(run_actors, "require_game_activity", lambda *_, **__: False)
    monkeypatch.setattr(run_actors, "raise_frame_rate", fake_raise)
    monkeypatch.setattr(run_actors.subprocess, "run", fake_run)

    outcomes = run_fleet(
        instances, stagger_bring_up(instances, stagger_arguments(tmp_path)), lambda _: None
    )

    assert all(outcome.failure is None for outcome in outcomes)
    assert events.index("raise:0") < events.index("up:2")
    assert events.index("episodes:emulator-5556") < events.index("up:2")
    # Each instance is still raised exactly once, and never after its episodes.
    assert sum(1 for event in events if event.startswith("raise:")) == len(instances)
    for index, instance in enumerate(instances):
        assert events.index(f"raise:{index}") < events.index(f"episodes:{instance.serial}")


def test_a_bring_up_failure_does_not_block_the_rest_of_the_fleet_from_starting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instances = [CloneInstance(index=index) for index in range(3)]
    started: list[str] = []
    torn_down: list[str] = []

    def fake_bring_up(target: CloneInstance, renderer: str, **keywords: object) -> str:
        started.append(target.serial)
        if target.index == 1:
            raise RuntimeError("cold boot refused")
        return "restored"

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(actor_record([summary(final_wave=4)])))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run_actors, "bring_up", fake_bring_up)
    monkeypatch.setattr(run_actors, "require_offline", lambda *_: None)
    monkeypatch.setattr(run_actors, "require_game_activity", lambda *_, **__: False)
    monkeypatch.setattr(run_actors, "raise_frame_rate", lambda *_: None)
    monkeypatch.setattr(run_actors.subprocess, "run", fake_run)

    outcomes = run_fleet(
        instances,
        stagger_bring_up(instances, stagger_arguments(tmp_path)),
        lambda instance: torn_down.append(instance.serial),
    )

    assert sorted(started) == ["emulator-5556", "emulator-5558", "emulator-5560"]
    assert sorted(torn_down) == ["emulator-5556", "emulator-5558", "emulator-5560"]
    failed = [outcome for outcome in outcomes if outcome.failure is not None]
    assert len(failed) == 1
    assert failed[0].index == 1
    assert "cold boot refused" in (failed[0].failure or "")


def test_a_game_that_lost_its_activity_before_the_raise_runs_no_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The instance is lost, and the actor says so instead of measuring it.

    `M1B-E049`: after the network is cut there is no relaunch that reaches home,
    so a game the guest's Play killed between bring-up and the first episode
    cannot be put back. What must not happen is an episode collected from it, or
    the bare `applied frame rate absent` the raise would otherwise report.
    """
    ran: list[str] = []

    def refuse(instance: CloneInstance) -> None:
        raise CloneError(f"{instance.serial}: the game has lost its activity")

    monkeypatch.setattr(run_actors, "bring_up", lambda *_, **__: "cold")
    monkeypatch.setattr(run_actors, "require_offline", lambda *_: None)
    monkeypatch.setattr(run_actors, "require_game_activity", refuse)
    monkeypatch.setattr(run_actors, "raise_frame_rate", lambda *_: ran.append("raise"))
    monkeypatch.setattr(
        run_actors.subprocess, "run", lambda *_, **__: ran.append("episodes")
    )

    with pytest.raises(CloneError, match="lost its activity"):
        collect_episodes(CloneInstance(), stagger_arguments(tmp_path))

    assert ran == []


def bridge_output(
    monkeypatch: pytest.MonkeyPatch, *, stdout: str, stderr: str = "", status: int = 0
) -> None:
    def script(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, status, stdout, stderr)

    monkeypatch.setattr(run_actors.subprocess, "run", script)


def test_the_cleanup_identity_report_reaches_the_operators_log(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cleanup verification is a device-safety property, so it must be visible."""
    bridge_output(
        monkeypatch,
        stdout=(
            "libunity_sha256: ffc1f3ef\nversionCode=1199\n"
            "libunity_mounts: 0\nbridge_artifacts: removed\n"
        ),
    )

    run_bridge("cleanup", CloneInstance(index=1))

    printed = capsys.readouterr().out
    assert "emulator-5558 cleanup: libunity_sha256: ffc1f3ef" in printed
    assert "emulator-5558 cleanup: libunity_mounts: 0" in printed
    assert "emulator-5558 cleanup: bridge_artifacts: removed" in printed


def test_a_failing_bridge_step_reports_its_output_as_well_as_failing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bridge_output(
        monkeypatch, stdout="libunity_mounts: 1\n", stderr="warning: still mounted\n", status=1
    )

    with pytest.raises(ActorFailure, match="still mounted"):
        run_bridge("cleanup", CloneInstance())

    printed = capsys.readouterr().out
    assert "emulator-5556 cleanup: libunity_mounts: 1" in printed
    assert "emulator-5556 cleanup: error: warning: still mounted" in printed
