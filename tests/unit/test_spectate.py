"""The watching path: its panel model, its refusal, and its episode loop.

`curses` is not tested here and deliberately so. What is worth holding is the
model — the lines a state of the world produces — and the loop that turns the
environment's decision stream into them; the drawing is the one part a human
looking at the terminal verifies better than an assertion can.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import spectate
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.episode import DecisionView, EpisodeSummary, TerminationOutcome
from tower_rl.environment.project_state import state_directory
from tower_rl.environment.run_environment import CadenceConfig, InstrumentedRunEnvironment
from tower_rl.environment.run_state import (
    LIVE_WIRE_NAMES,
    NO_ENEMY_DISTANCE,
    RunStateBuilder,
)
from tower_rl.learning.policies import Policy, RandomPolicy
from tower_rl.simulation.instance import CloneInstance
from tower_rl.simulation.instrumented_bridge import UpgradeSlotLabel

REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass
class RecordingPanel:
    """A panel that keeps what it was told to draw and presses nothing."""

    drawn: list[list[str]] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    held: float | None = None

    def draw(self, lines: Sequence[str]) -> None:
        self.drawn.append(list(lines))

    def key(self) -> str:
        return self.keys.pop(0) if self.keys else ""

    def hold(self, timeout: float) -> None:
        self.held = timeout


@dataclass
class _InterruptingPolicy:
    """Plays normally, then raises `KeyboardInterrupt` mid-episode, as Ctrl-C does."""

    inner: Policy
    after_episodes: int
    episodes: int = 0
    decisions: int = 0

    def initial_state(self) -> object:
        self.episodes += 1
        self.decisions = 0
        return self.inner.initial_state()

    def act(self, features: object, state: object, *, epsilon: float = 0.0) -> tuple[int, object]:
        self.decisions += 1
        if self.episodes > self.after_episodes and self.decisions > 2:
            raise KeyboardInterrupt
        return self.inner.act(features, state, epsilon=epsilon)  # type: ignore[arg-type]


def _view(**overrides: object) -> DecisionView:
    fields: dict[str, object] = {
        "episode": 1,
        "decision": 1,
        "wave": 3,
        "cash": 1240.0,
        "health_fraction": 0.5,
        # Every live reading at its resting value, with a few set to something a
        # watcher could check against the HUD.
        "hud": {
            **dict.fromkeys(LIVE_WIRE_NAMES, 0.0),
            "damage": 12.09,
            "criticalChance": 5.0,
            "closestEnemyDistance": NO_ENEMY_DISTANCE,
            "enemiesKilledThisWave": 26.0,
            "enemiesSpawnedThisWave": 27.0,
            "estimatedEnemiesToSpawnThisWave": 21.0,
            "waveTimer": 12.5,
            "waveLengthSeconds": 26.0,
            "waveCooldownSeconds": 9.0,
            "gameplayTimeThisRound": 343.0,
        },
        "action": "wait",
        "reward": 0.0,
        "game_ms": 2000.0,
        "done": False,
        "termination": None,
    }
    fields.update(overrides)
    return DecisionView(**fields)  # type: ignore[arg-type]


# -- the panel's model -----------------------------------------------------


def test_the_panel_shows_the_state_the_latest_decision_left() -> None:
    spectator = spectate.Spectator()
    spectator.observe(_view(wave=7, cash=1240.0, health_fraction=0.62, action="attack:2"))

    lines = spectate.panel_lines(
        spectator,
        policy="checkpoint-0100000",
        renderer="lavapipe",
        episodes_requested=1,
        elapsed_seconds=60.0,
    )
    text = "\n".join(lines)

    assert "wave 7" in text
    assert "1,240" in text, "cash is read by a human, not log-scaled"
    assert "62%" in text
    assert "attack:2" in text
    assert "episode 1 of 1" in text
    assert "1.0/min" in text, "one decision in one minute"


def test_the_panel_shows_the_tower_stats_and_the_wave_in_the_games_own_units() -> None:
    """The whole point of the block: a watcher can check it against the HUD."""
    spectator = spectate.Spectator()
    spectator.observe(_view())

    lines = spectate.panel_lines(
        spectator,
        policy="random",
        renderer="lavapipe",
        episodes_requested=1,
        elapsed_seconds=60.0,
    )
    text = "\n".join(lines)

    assert "dmg 12.1" in text, "damage is a HUD number, not a log"
    assert "crit% 5.0" in text, "percent as the game stores it, not a fraction"
    assert "enemies 26/27 of ~21" in text
    assert "nearest none" in text, "the sentinel is an absence, never a distance"
    assert "wave clock 12.5/26+9s" in text
    assert "round 343s" in text


def test_the_panel_names_the_upgrade_rows_a_slot_index_cannot() -> None:
    spectator = spectate.Spectator()
    spectator.observe(_view(action="attack:2"))
    labels = (
        UpgradeSlotLabel("attack", 0, "Damage", "Tower damage"),
        UpgradeSlotLabel("attack", 2, "Critical Chance", "Chance to crit"),
        # The game's own trailing empties: carried on the wire to keep the slot
        # indices aligned, and never shown.
        UpgradeSlotLabel("attack", 19, "", ""),
        UpgradeSlotLabel("defense", 0, "Health", "Tower max health"),
        UpgradeSlotLabel("utility", 1, "Cash / Wave", "Cash each wave"),
    )

    text = "\n".join(
        spectate.panel_lines(
            spectator,
            policy="random",
            renderer="lavapipe",
            episodes_requested=1,
            elapsed_seconds=60.0,
            labels=labels,
        )
    )

    assert "2:Critical Chance" in text
    assert "defense 0:Health" in text
    assert "1:Cash / Wave" in text
    assert "19:" not in text


def test_a_session_record_names_the_rows_the_actions_addressed() -> None:
    labels = (
        UpgradeSlotLabel("attack", 0, "Damage", "Tower damage"),
        UpgradeSlotLabel("attack", 19, "", ""),
    )

    record = spectate.session_record(
        (),
        {"name": "random"},
        frame_rate_hz=60,
        decision_cadence="choice-points",
        wall_seconds=12.0,
        labels=labels,
    )

    assert record["upgrade_rows"] == [
        {"family": "attack", "index": 0, "name": "Damage", "description": "Tower damage"}
    ]


def test_the_panel_says_how_long_the_agent_held_the_decision() -> None:
    """A decision covers every forced-WAIT slice the environment played through.

    Under choice points a watcher sees one line for what used to be dozens, so
    the panel says how much game time that one line stands for (ADR 0009).
    """
    spectator = spectate.Spectator()
    spectator.observe(_view(game_ms=15_000.0))

    lines = spectate.panel_lines(
        spectator,
        policy="random",
        renderer="lavapipe",
        episodes_requested=1,
        elapsed_seconds=60.0,
    )

    assert "held 15.0s" in "\n".join(lines)


def test_the_panel_counts_episodes_and_means_their_final_waves() -> None:
    spectator = spectate.Spectator()
    for wave in (4, 6):
        spectator.observe(_view(wave=wave, done=True, termination=TerminationOutcome.GAME_OVER))

    lines = spectate.panel_lines(
        spectator, policy="random", renderer="lavapipe", episodes_requested=0, elapsed_seconds=10.0
    )
    text = "\n".join(lines)

    assert spectator.episodes_finished == 2
    assert spectator.mean_final_wave == 5.0
    assert "episodes played 2" in text
    assert "mean final wave 5.00" in text
    assert "game_over" in text, "the death the watcher just saw is named"
    assert "episode 1 of unlimited" in text


def test_the_panel_keeps_only_the_last_twenty_actions_newest_first() -> None:
    spectator = spectate.Spectator()
    for decision in range(1, 26):
        spectator.observe(_view(decision=decision, action=f"attack:{decision}"))

    lines = spectate.panel_lines(
        spectator, policy="random", renderer="lavapipe", episodes_requested=1, elapsed_seconds=1.0
    )
    actions = [line.strip() for line in lines if line.startswith("  ")]

    assert len(actions) == spectate.RECENT_ACTIONS
    assert actions[0].endswith("attack:25"), "newest first"
    assert actions[-1].endswith("attack:6")


def test_the_panel_says_the_session_is_over_while_it_holds() -> None:
    """The last thing that happens is the tower dying; it stays on screen."""
    spectator = spectate.Spectator()
    spectator.observe(_view(done=True, termination=TerminationOutcome.GAME_OVER))

    holding = "\n".join(
        spectate.panel_lines(
            spectator,
            policy="random",
            renderer="lavapipe",
            episodes_requested=1,
            elapsed_seconds=1.0,
            holding=True,
        )
    )

    assert "press any key" in holding


def test_the_plain_panel_prints_the_death_on_the_decision_it_happened_on(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--no-panel` is what an unattended session writes; the death must be in it.

    A log that shows health reaching 0 but never names the ended episode makes
    the reader infer the death instead of reading it.
    """
    spectator = spectate.Spectator()
    spectator.observe(_view(wave=8, done=True, termination=TerminationOutcome.GAME_OVER))
    lines = spectate.panel_lines(
        spectator,
        policy="checkpoint-0050123",
        renderer="lavapipe",
        episodes_requested=1,
        elapsed_seconds=10.0,
    )

    spectate.PlainPanel().draw(lines)

    printed = capsys.readouterr().out
    assert "episode 1 ended at wave 8: game_over" in printed
    assert "wave 8" in printed, "the state line is still there"


def test_the_plain_panel_stays_quiet_while_the_episode_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The state and the live readings, and nothing else.

    The death line appears on the death, not before it, and no blank or stray
    line ever reaches the log: three lines a decision, always the same three.
    """
    spectator = spectate.Spectator()
    spectator.observe(_view(wave=3))

    spectate.PlainPanel().draw(
        spectate.panel_lines(
            spectator,
            policy="random",
            renderer="lavapipe",
            episodes_requested=1,
            elapsed_seconds=10.0,
        )
    )

    printed = capsys.readouterr().out
    assert printed.count("\n") == 3, "the state line and the two live-reading lines"
    assert "dmg 12.1" in printed, "a log is what a recording is lined up against"
    assert "wave clock" in printed


def test_the_plain_panel_draws_before_the_first_decision_arrives(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The waiting frame has no death line to find at `DEATH_LINE` either."""
    spectate.PlainPanel().draw(
        spectate.panel_lines(
            spectate.Spectator(),
            policy="random",
            renderer="lavapipe",
            episodes_requested=1,
            elapsed_seconds=0.0,
        )
    )

    printed = capsys.readouterr().out
    assert "waiting for the first decision" in printed
    assert printed.count("\n") == 1


def test_the_panel_draws_before_the_first_decision_arrives() -> None:
    lines = spectate.panel_lines(
        spectate.Spectator(),
        policy="random",
        renderer="lavapipe",
        episodes_requested=1,
        elapsed_seconds=0.0,
    )

    assert "waiting for the first decision" in "\n".join(lines)


# -- the exclusive-device refusal ------------------------------------------


def test_spectating_is_refused_while_an_emulator_is_attached() -> None:
    with pytest.raises(SystemExit, match="refusing to spectate"):
        spectate.refuse_a_shared_host("List of devices attached\nemulator-5556\tdevice\n", [])


def test_spectating_is_refused_while_a_qemu_process_is_running() -> None:
    """adb can be blind to an emulator that has not published its console yet."""
    with pytest.raises(SystemExit, match="refusing to spectate"):
        spectate.refuse_a_shared_host(
            "List of devices attached\n", ["4711 /usr/bin/qemu-system-x86_64"]
        )


def test_an_empty_host_is_not_refused() -> None:
    spectate.refuse_a_shared_host("List of devices attached\n\n", [])


def _fake_proc(root: Path, processes: dict[str, str]) -> Path:
    """A `/proc` with one entry per pid, each `exe` pointing at its binary."""
    proc = root / "proc"
    for pid, executable in processes.items():
        entry = proc / pid
        entry.mkdir(parents=True)
        binary = root / "bin" / executable
        binary.parent.mkdir(exist_ok=True)
        binary.touch()
        (entry / "exe").symlink_to(binary)
    (proc / "self").mkdir(parents=True, exist_ok=True)
    return proc


def test_an_emulator_is_found_by_what_it_is_running_not_by_its_command_line(
    tmp_path: Path,
) -> None:
    """`/proc/<pid>/exe` is the kernel's answer; a command line is a string.

    The distinction is not academic: this script names `qemu-system` in its own
    arguments and in this very test, and a `pgrep -f` would report both.
    """
    proc = _fake_proc(
        tmp_path,
        {
            "101": "qemu-system-x86_64",
            "102": "python3",
            "103": "qemu-system-aarch64",
        },
    )

    found = spectate.running_emulators(proc)

    assert [entry.split()[0] for entry in found] == ["101", "103"]
    assert all("qemu-system" in entry for entry in found)


def test_a_host_with_no_qemu_process_reads_as_free(tmp_path: Path) -> None:
    proc = _fake_proc(tmp_path, {"101": "python3", "102": "emulator"})

    assert spectate.running_emulators(proc) == []


def test_a_process_that_will_not_be_read_is_skipped_rather_than_guessed_at(
    tmp_path: Path,
) -> None:
    """A dangling or unreadable `exe` is somebody else's process, not evidence."""
    proc = tmp_path / "proc"
    (proc / "101").mkdir(parents=True)  # no exe link at all
    (proc / "not-a-pid").mkdir()

    assert spectate.running_emulators(proc) == []


# -- the episode loop ------------------------------------------------------


def _environment() -> tuple[InstrumentedRunEnvironment, FakeRunPort]:
    port = FakeRunPort(damage_per_second=2.0, max_health=2.0)
    environment = InstrumentedRunEnvironment(
        port=port,
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )
    return environment, port


def test_a_session_plays_the_episodes_it_was_asked_for_and_draws_every_decision() -> None:
    """End to end on the fake port: the panel sees exactly the decisions taken."""
    environment, _ = _environment()
    panel = RecordingPanel()
    spectator = spectate.Spectator()
    summaries: list[EpisodeSummary] = []

    spectate.spectate_session(
        environment,
        RandomPolicy(seed=7),
        panel,
        spectator,
        summaries,
        episodes=3,
        policy_name="random",
        renderer="lavapipe",
        actor_id="spectate:random",
    )

    assert len(summaries) == 3
    decisions = sum(summary.decisions for summary in summaries)
    assert spectator.decisions == decisions
    assert len(panel.drawn) == decisions, "one redraw per decision"
    assert spectator.episodes_finished == 3
    assert environment.on_decision is None, "the observer is handed back"

    record = spectate.session_record(
        summaries, {"name": "random"}, frame_rate_hz=60,
        decision_cadence="choice-points", wall_seconds=12.0,
    )
    assert [row["episode_index"] for row in record["episodes"]] == [0, 1, 2]
    assert [row["final_wave"] for row in record["episodes"]] == [
        summary.final_wave for summary in summaries
    ]
    assert record["policy_identity"] == {"name": "random"}
    assert record["frame_rate_hz"] == 60


def test_a_record_names_the_rate_the_session_actually_ran_at() -> None:
    """A session watched at 120 Hz must not be recorded as the 60 Hz default."""
    record = spectate.session_record(
        (), {"name": "random"}, frame_rate_hz=120,
        decision_cadence="choice-points", wall_seconds=1.0,
    )

    assert record["frame_rate_hz"] == 120
    assert spectate.SPECTATE_FRAME_RATE_HZ == 60, "the default is still real time"


def test_q_stops_at_the_next_decision_and_keeps_the_episodes_already_finished() -> None:
    """An episode is minutes long; a stop that waited for one is not a stop."""
    environment, _ = _environment()
    panel = RecordingPanel()
    spectator = spectate.Spectator()
    panel.keys = ["", "", "q"]
    summaries: list[EpisodeSummary] = []

    spectate.spectate_session(
        environment,
        RandomPolicy(seed=3),
        panel,
        spectator,
        summaries,
        episodes=0,
        policy_name="random",
        renderer="lavapipe",
        actor_id="spectate:random",
    )

    assert summaries == [], "the interrupted episode did not finish, so it has no record"
    assert spectator.decisions == 3
    assert environment.on_decision is None


def test_ctrl_c_keeps_every_episode_that_had_already_finished() -> None:
    """A Ctrl-C is not a reason to throw away runs that were played in full.

    The summaries belong to the caller and are appended as each episode ends,
    so an interrupt out of episode three leaves one and two exactly where a `q`
    would have left them - which is what lets `run()` still write the record.
    """
    environment, _ = _environment()
    panel = RecordingPanel()
    spectator = spectate.Spectator()
    summaries: list[EpisodeSummary] = []

    with pytest.raises(KeyboardInterrupt):
        spectate.spectate_session(
            environment,
            _InterruptingPolicy(RandomPolicy(seed=11), after_episodes=2),
            panel,
            spectator,
            summaries,
            episodes=0,
            policy_name="random",
            renderer="lavapipe",
            actor_id="spectate:random",
        )

    assert len(summaries) == 2, "the two finished episodes survive the interrupt"
    assert spectator.episodes_finished == 2
    assert environment.on_decision is None, "the observer is still handed back"

    record = spectate.session_record(
        summaries, {"name": "random"}, frame_rate_hz=60,
        decision_cadence="choice-points", wall_seconds=9.0,
    )
    assert [row["episode_index"] for row in record["episodes"]] == [0, 1]


# -- recording stays on the human-facing path ------------------------------


def _reachable_names(path: Path) -> set[str]:
    """Every name a running line in `path` can reach, docstrings excluded."""
    tree = ast.parse(path.read_text())
    docstrings = {
        node.body[0].value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value not in docstrings
        ):
            names.add(node.value)
    return names


@pytest.mark.parametrize("script", ["train.py", "run_actors.py", "run_episodes.py"])
def test_no_collecting_path_records_the_screen_or_reaches_for_the_panel(script: str) -> None:
    """Recording is for a human watching, and costs device time to produce.

    A training or evaluation run that recorded itself would spend part of every
    actor's host on encoding video nothing reads, and would do it invisibly. The
    rule is one-directional: `spectate.py` composes the collecting path's
    pieces, and none of them may reach back for its.
    """
    names = _reachable_names(REPOSITORY / "scripts" / script)

    assert "screenrecord" not in names
    assert "spectate" not in names
    assert "GuestRecording" not in names


def test_a_recording_and_its_records_land_in_the_projects_state_directory() -> None:
    """`state/recordings/`, not a `recordings/` beside whatever the cwd was.

    An mp4 is far above GitHub's file limit and this repo is public, so the one
    thing that must hold is that both the video and the per-run records land
    under the git-ignored state directory — including the relative `--record`
    filename, which is resolved against it rather than against the cwd.
    """
    assert state_directory() / "recordings" == spectate.RECORDINGS_DIRECTORY

    arguments = spectate.parse_arguments(["--record", "session.mp4"])

    assert arguments.record == state_directory() / "recordings" / "session.mp4"
    assert arguments.output_directory == state_directory() / "recordings" / "records"


# -- the recording's lifecycle ---------------------------------------------


@dataclass
class FakeGuest:
    """A stand-in for `adb`: it answers, it records, and it touches no device."""

    #: Exit statuses to answer successive `screenrecord` invocations with.
    statuses: list[int] = field(default_factory=list)
    #: Called with the recording after each chunk, to act as `finish` would.
    after_chunk: object = None
    calls: list[tuple[str, ...]] = field(default_factory=list)
    chunks: int = 0
    fail_with: Exception | None = None

    def __call__(self, instance: CloneInstance, *args: str, timeout: float = 30.0) -> str:
        self.calls.append(args)
        if args[0] == "pull":
            Path(args[2]).write_text("video")
            return ""
        if args[0] == "shell" and args[1].startswith("screenrecord"):
            if self.fail_with is not None:
                raise self.fail_with
            status = self.statuses[self.chunks] if self.chunks < len(self.statuses) else 0
            self.chunks += 1
            if callable(self.after_chunk):
                self.after_chunk()
            return f"rc={status}"
        return ""

    def screenrecords(self) -> list[tuple[str, ...]]:
        return [call for call in self.calls if call[0] == "shell" and "screenrecord" in call[1]]


def _recording(tmp_path: Path, guest: FakeGuest) -> spectate.GuestRecording:
    return spectate.GuestRecording(CloneInstance(), tmp_path / "session.mp4", run=guest)


def test_chunks_run_back_to_back_and_are_all_pulled_when_the_session_ends(
    tmp_path: Path,
) -> None:
    """The three-minute limit is the Android tool's, so a session is chunks."""
    guest = FakeGuest()
    recording = _recording(tmp_path, guest)
    # Two whole chunks, then a stop between chunks - what `finish` does to the
    # loop when it arrives while no chunk is in flight.
    guest.after_chunk = lambda: recording._stop.set() if guest.chunks == 2 else None

    recording._record()
    pulled = recording.finish()

    assert len(guest.screenrecords()) == 2
    assert [path.name for path in pulled] == ["session-000.mp4", "session-001.mp4"]
    assert ("shell", "rm", "-f", "/sdcard/tower-rl-spectate-000.mp4") in guest.calls
    assert ("shell", "rm", "-f", "/sdcard/tower-rl-spectate-001.mp4") in guest.calls


def test_no_chunk_is_started_once_stopping_has_begun(tmp_path: Path) -> None:
    """A chunk started after `finish` is a file nothing would ever pull."""
    guest = FakeGuest()
    recording = _recording(tmp_path, guest)
    guest.after_chunk = lambda: recording._stop.set()

    recording._record()
    recording.finish()
    started = len(guest.screenrecords())
    recording._record()  # as a live thread would, one more time round the loop

    assert len(guest.screenrecords()) == started == 1


def test_the_stop_flag_is_set_before_the_guest_is_interrupted(tmp_path: Path) -> None:
    """Ordering is the whole of it: interrupt first and the loop starts a chunk."""
    guest = FakeGuest()
    recording = _recording(tmp_path, guest)
    seen: list[bool] = []

    def watch_order(instance: CloneInstance, *args: str, timeout: float = 30.0) -> str:
        if args[:2] == ("shell", "pkill"):
            seen.append(recording._stop.is_set())
        return guest(instance, *args, timeout=timeout)

    recording.run = watch_order
    recording.finish()

    assert seen == [True]


def test_a_chunk_cut_off_mid_write_is_pulled_as_an_ordinary_chunk(tmp_path: Path) -> None:
    """`finish` lands mid-chunk, and the file it interrupted is still whole.

    `screenrecord` finalises what it is writing on SIGINT, so an interrupted
    chunk plays and differs from a whole one only in its duration - measured on
    device, three interrupted chunks all read back as valid MP4 (`M2-E003`).
    Nothing here names it apart.
    """
    guest = FakeGuest()
    recording = _recording(tmp_path, guest)
    guest.after_chunk = lambda: recording._stop.set() if guest.chunks == 2 else None

    recording._record()
    pulled = recording.finish()

    assert [path.name for path in pulled] == ["session-000.mp4", "session-001.mp4"]


def test_a_recording_the_guest_stopped_answering_is_still_pulled(tmp_path: Path) -> None:
    """Ctrl-C mid-chunk: the emulator goes, adb raises, the loop ends quietly.

    A lost recording must not be a lost session, so the failure is reported and
    what reached the guest is still retrieved.
    """
    guest = FakeGuest(fail_with=RuntimeError("device offline"))
    recording = _recording(tmp_path, guest)

    recording._record()
    pulled = recording.finish()

    assert [path.name for path in pulled] == ["session-000.mp4"]
    assert len(guest.screenrecords()) == 1, "the loop stops rather than retrying forever"


# -- the decision track beside the video -----------------------------------


def _track(tmp_path: Path, times: list[float], **overrides: object) -> spectate.DecisionTrack:
    """A track on a clock that reads the given times, one per call."""
    fields: dict[str, object] = {
        "path": tmp_path / "session.decisions.jsonl",
        "anchor": 100.0,
        "clock": lambda: times.pop(0),
    }
    fields.update(overrides)
    return spectate.DecisionTrack(**fields)  # type: ignore[arg-type]


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_decision_is_written_as_one_line_saying_when_it_was_and_what_it_bought(
    tmp_path: Path,
) -> None:
    """The line is the whole seam: a renderer reads it and nothing else."""
    track = _track(
        tmp_path,
        [112.5],
        labels=[UpgradeSlotLabel(family="attack", index=2, name="Critical Chance", description="")],
    )

    track.write(
        _view(episode=2, decision=17, wave=6, cash=1240.0, action="attack:2", game_ms=9800.0)
    )
    track.close()

    (line,) = _lines(track.path)
    assert line["video_s"] == 12.5
    assert line["chunk"] == 0 and line["chunk_s"] == 12.5
    assert line["episode"] == 2 and line["decision"] == 17
    assert line["wave"] == 6 and line["cash"] == 1240.0
    assert line["health_fraction"] == 0.5
    assert line["action"] == "attack:2"
    assert line["label"] == "Critical Chance", "the game's own name for the row, not the index"
    assert line["held_s"] == 9.8
    assert line["ended"] is False and "reason" not in line
    hud = line["hud"]
    assert isinstance(hud, dict)
    assert hud["damage"] == 12.09 and hud["waveTimer"] == 12.5
    assert set(hud) <= set(spectate.PANEL_HUD_WIRES)


def test_a_wait_is_named_as_one_and_an_unknown_slot_keeps_its_index(tmp_path: Path) -> None:
    """A panel that said `attack:2` for everything would say nothing."""
    track = _track(tmp_path, [101.0, 102.0], labels=[])

    track.write(_view(action="wait"))
    track.write(_view(action="defense:1"))
    track.close()

    assert [line["label"] for line in _lines(track.path)] == [spectate.HOLD_LABEL, "defense:1"]


def test_the_death_decision_carries_the_end_and_why(tmp_path: Path) -> None:
    track = _track(tmp_path, [150.0])

    track.write(_view(done=True, termination=TerminationOutcome.GAME_OVER))
    track.close()

    (line,) = _lines(track.path)
    assert line["ended"] is True
    assert line["reason"] == "game_over"


def test_every_reading_the_panel_draws_is_one_the_track_writes() -> None:
    """The rendered panel shows what the watcher saw, or it is a second view.

    Read from `hud_lines`' own source: a reading added to the panel and not to
    `PANEL_HUD_WIRES` would be drawn live and missing from every recording.
    """
    source = ast.parse((REPOSITORY / "scripts" / "spectate.py").read_text())
    (function,) = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "hud_lines"
    ]
    read = {
        node.slice.value
        for node in ast.walk(function)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }

    assert read, "the reader found no HUD lookups at all"
    assert read <= set(spectate.PANEL_HUD_WIRES)


def test_a_track_is_written_for_every_decision_a_session_takes(tmp_path: Path) -> None:
    """The fake port plays the episodes; the track is the record of them."""
    environment, _ = _environment()
    summaries: list[EpisodeSummary] = []
    clock = iter(float(tick) for tick in range(100, 1000))
    track = spectate.DecisionTrack(
        path=tmp_path / "session.decisions.jsonl", anchor=100.0, clock=lambda: next(clock)
    )

    spectate.spectate_session(
        environment,
        RandomPolicy(seed=5),
        RecordingPanel(),
        spectate.Spectator(),
        summaries,
        episodes=2,
        policy_name="random",
        renderer="lavapipe",
        actor_id="spectate:random",
        track=track,
    )
    track.close()

    lines = _lines(track.path)
    assert len(lines) == sum(summary.decisions for summary in summaries)
    assert [line["episode"] for line in lines] == sorted(line["episode"] for line in lines)
    assert [line["video_s"] for line in lines] == sorted(line["video_s"] for line in lines)
    assert sum(1 for line in lines if line["ended"]) == 2, "one ending per episode"


# -- placing a decision in the video ---------------------------------------


def test_a_decision_is_placed_in_the_chunk_that_was_recording_at_the_time() -> None:
    """The chunk and the offset into it: what a seam cannot move."""
    starts = [100.5, 281.0, 462.0]

    assert spectate.video_position(100.6, 100.0, starts) == spectate.VideoPosition(
        video_s=0.6, chunk=0, chunk_s=0.1
    )
    assert spectate.video_position(300.0, 100.0, starts) == spectate.VideoPosition(
        video_s=200.0, chunk=1, chunk_s=19.0
    )
    assert spectate.video_position(470.0, 100.0, starts) == spectate.VideoPosition(
        video_s=370.0, chunk=2, chunk_s=8.0
    )


def test_a_decision_taken_before_the_guest_started_recording_belongs_to_the_first_chunk() -> None:
    """There is no earlier frame for it to be in, so it is not placed before one."""
    position = spectate.video_position(100.2, 100.0, [100.5])

    assert position.chunk == 0
    assert position.video_s == 0.2
    assert position.chunk_s == 0.2


def test_the_anchor_is_taken_when_the_guest_is_asked_to_record(tmp_path: Path) -> None:
    """Everything beside the video is timed from the instant `start` returns.

    The chunk starts are taken on the same clock, one per chunk, which is what
    makes a decision placeable in a session longer than three minutes.
    """
    guest = FakeGuest()
    ticks = iter([50.0, 50.25, 230.0])
    recording = spectate.GuestRecording(
        CloneInstance(), tmp_path / "session.mp4", run=guest, clock=lambda: next(ticks)
    )
    guest.after_chunk = lambda: recording._stop.set() if guest.chunks == 2 else None

    anchor = recording.start()
    if recording._thread is not None:
        recording._thread.join(timeout=5.0)

    assert anchor == 50.0 == recording.anchor
    assert recording.chunk_starts == [50.25, 230.0]
    assert anchor <= recording.chunk_starts[0], "the guest starts after it is asked to"


def test_the_track_lands_beside_the_video_it_belongs_to() -> None:
    video = spectate.RECORDINGS_DIRECTORY / "session.mp4"

    assert spectate.decisions_beside(video).name == "session.decisions.jsonl"
    assert spectate.decisions_beside(video).parent == spectate.RECORDINGS_DIRECTORY


def test_a_record_says_where_the_recording_and_its_track_are() -> None:
    """A record read afterwards is where the two files are lined up from."""
    record = spectate.session_record(
        (), {"name": "random"}, frame_rate_hz=60,
        decision_cadence="choice-points", wall_seconds=1.0,
        recording={"video": "/tmp/session.mp4", "anchor_monotonic": 50.0},
    )

    assert record["recording"] == {"video": "/tmp/session.mp4", "anchor_monotonic": 50.0}
    assert "recording" not in spectate.session_record(
        (), {"name": "random"}, frame_rate_hz=60,
        decision_cadence="choice-points", wall_seconds=1.0,
    ), "a session that recorded nothing says nothing about a recording"
