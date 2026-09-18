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

from tower_rl.environment.episode import DecisionView, TerminationOutcome
from tower_rl.environment.run_environment import CadenceConfig, InstrumentedRunEnvironment
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.policies import RandomPolicy

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

    def wait_for_key(self, timeout: float) -> None:
        self.held = timeout


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
        spectator, policy="checkpoint-0100000", episodes_requested=1, elapsed_seconds=60.0
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
        spectator, policy="random", episodes_requested=0, elapsed_seconds=10.0
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
        spectator, policy="random", episodes_requested=1, elapsed_seconds=1.0
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
            episodes_requested=1,
            elapsed_seconds=1.0,
            holding=True,
        )
    )

    assert "press any key" in holding


def test_the_panel_draws_before_the_first_decision_arrives() -> None:
    lines = spectate.panel_lines(
        spectate.Spectator(), policy="random", episodes_requested=1, elapsed_seconds=0.0
    )

    assert "waiting for the first decision" in "\n".join(lines)


# -- the exclusive-device refusal ------------------------------------------


def test_spectating_is_refused_while_an_emulator_is_attached() -> None:
    with pytest.raises(SystemExit, match="refusing to spectate"):
        spectate.refuse_a_shared_host("List of devices attached\nemulator-5556\tdevice\n", "")


def test_spectating_is_refused_while_a_qemu_process_is_running() -> None:
    """adb can be blind to an emulator that has not published its console yet."""
    with pytest.raises(SystemExit, match="refusing to spectate"):
        spectate.refuse_a_shared_host(
            "List of devices attached\n", "4711 qemu-system-x86_64 -avd tower_rl\n"
        )


def test_an_empty_host_is_not_refused() -> None:
    spectate.refuse_a_shared_host("List of devices attached\n\n", "\n")


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

    summaries = spectate.spectate_session(
        environment,
        RandomPolicy(seed=7),
        panel,
        spectator,
        episodes=3,
        policy_name="random",
        actor_id="spectate:random",
    )

    assert len(summaries) == 3
    decisions = sum(summary.decisions for summary in summaries)
    assert spectator.decisions == decisions
    assert len(panel.drawn) == decisions, "one redraw per decision"
    assert spectator.episodes_finished == 3
    assert environment.on_decision is None, "the observer is handed back"

    record = spectate.session_record(summaries, {"name": "random"}, wall_seconds=12.0)
    assert [row["episode_index"] for row in record["episodes"]] == [0, 1, 2]
    assert [row["final_wave"] for row in record["episodes"]] == [
        summary.final_wave for summary in summaries
    ]
    assert record["policy_identity"] == {"name": "random"}
    assert record["frame_rate_hz"] == spectate.SPECTATE_FRAME_RATE_HZ


def test_q_stops_at_the_next_decision_and_keeps_the_episodes_already_finished() -> None:
    """An episode is minutes long; a stop that waited for one is not a stop."""
    environment, _ = _environment()
    panel = RecordingPanel()
    spectator = spectate.Spectator()
    panel.keys = ["", "", "q"]

    summaries = spectate.spectate_session(
        environment,
        RandomPolicy(seed=3),
        panel,
        spectator,
        episodes=0,
        policy_name="random",
        actor_id="spectate:random",
    )

    assert summaries == [], "the interrupted episode did not finish, so it has no record"
    assert spectator.decisions == 3
    assert environment.on_decision is None


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
