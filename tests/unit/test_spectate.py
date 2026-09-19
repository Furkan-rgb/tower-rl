"""The watching path: its panel model, its refusal, and its episode loop.

`curses` is not tested here and deliberately so. What is worth holding is the
model — the lines a state of the world produces — and the loop that turns the
environment's decision stream into them; the drawing is the one part a human
looking at the terminal verifies better than an assertion can.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import spectate
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.episode import DecisionView, EpisodeSummary, TerminationOutcome
from tower_rl.environment.run_environment import CadenceConfig, InstrumentedRunEnvironment
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.policies import Policy, RandomPolicy
from tower_rl.simulation.instance import CloneInstance

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
        "action": "wait",
        "reward": 0.0,
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
    """One line a decision: the death line appears on the death, not before it."""
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

    assert capsys.readouterr().out.count("\n") == 1, "no blank or stray second line"


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
        summaries, {"name": "random"}, frame_rate_hz=60, wall_seconds=12.0
    )
    assert [row["episode_index"] for row in record["episodes"]] == [0, 1, 2]
    assert [row["final_wave"] for row in record["episodes"]] == [
        summary.final_wave for summary in summaries
    ]
    assert record["policy_identity"] == {"name": "random"}
    assert record["frame_rate_hz"] == 60


def test_a_record_names_the_rate_the_session_actually_ran_at() -> None:
    """A session watched at 120 Hz must not be recorded as the 60 Hz default."""
    record = spectate.session_record((), {"name": "random"}, frame_rate_hz=120, wall_seconds=1.0)

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
        summaries, {"name": "random"}, frame_rate_hz=60, wall_seconds=9.0
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
