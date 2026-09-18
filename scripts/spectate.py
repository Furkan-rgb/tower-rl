#!/usr/bin/env python3
"""Watch one agent play: the game on screen, its decisions beside it.

This is the human-facing path and nothing measures anything here. One clone
instance is brought up *with a window*, the chosen arm plays through the same
environment the fleet uses, and a terminal panel shows what it is doing while it
does it — wave, cash, health, the last twenty actions, how many episodes it has
played and what wave they ended at.

    uv run python scripts/spectate.py --policy checkpoint:<path>

The default session is one episode: the agent plays a single run from wave 1
until the tower dies, the panel holds the final state, and a keypress tears the
instance down. `--episodes N` plays N, and `--episodes 0` plays until `q`.

Two rules this path does not share with any other. It runs at 60 Hz, which is
real time — the fleet's 120 Hz exists to make an advance cheap, and a human
watching wants the game's own speed. And it refuses to start at all while any
emulator is running: a session here must never share the host with a
measurement or a training run, whose throughput is what the host is for.

Everything else is exactly what the fleet does: the canonical AVD is refused by
`CloneInstance`, the instance is `-read-only`, the bridge is deployed with its
digest confirmed, offline is verified by interface, and the instance is torn
down through `tear_down_instance` whatever happened.
"""

from __future__ import annotations

import argparse
import curses
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# One torch thread, for the same reason `run_episodes.py` sets it: acting is one
# small forward pass per decision and a pool buys nothing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from run_episodes import (  # noqa: E402
    CHECKPOINT_SELECTOR,
    POLICIES,
    add_cadence_arguments,
    cadence_from,
    policy_from,
)

from tower_rl.environment.episode import DecisionView, EpisodeSummary  # noqa: E402
from tower_rl.environment.run_environment import InstrumentedRunEnvironment  # noqa: E402
from tower_rl.environment.run_state import RunStateBuilder  # noqa: E402
from tower_rl.learning.actor import Actor, ActorConfig  # noqa: E402
from tower_rl.learning.evaluator import episode_record  # noqa: E402
from tower_rl.learning.policies import Policy  # noqa: E402
from tower_rl.simulation.android_sdk import find_android_tool  # noqa: E402
from tower_rl.simulation.bridge import (  # noqa: E402
    bridge_build_directory,
    compatibility,
    deploy_bridge,
)
from tower_rl.simulation.bring_up import (  # noqa: E402
    bring_up,
    require_game_activity,
    require_offline,
)
from tower_rl.simulation.fleet import tear_down_instance  # noqa: E402
from tower_rl.simulation.frame_rate import raise_frame_rate  # noqa: E402
from tower_rl.simulation.instance import CloneInstance, adb  # noqa: E402
from tower_rl.simulation.instrumented_bridge import InstrumentedBridgeClient  # noqa: E402
from tower_rl.simulation.instrumented_run_adapter import InstrumentedRunAdapter  # noqa: E402

#: The rate a spectated instance runs at. 60 Hz is the guest's stock rate, which
#: is real time: one game second per wall second. The fleet's 120 exists to buy
#: throughput, which is worth nothing to somebody watching.
SPECTATE_FRAME_RATE_HZ = 60

#: How many past actions the panel keeps. Twenty is what fits beside the game
#: window without the panel becoming the thing being read.
RECENT_ACTIONS = 20

#: The Android limit on one `screenrecord`, in seconds. It is a hard limit in
#: the guest tool, so a longer session is recorded as consecutive chunks and the
#: seam between two of them loses about a second while the next one starts.
SCREENRECORD_CHUNK_SECONDS = 180


class SpectateStopped(Exception):
    """The watcher asked to stop. Raised out of the decision stream."""


# -- the panel's model -----------------------------------------------------
#
# Everything a panel shows is derived from the decision stream and from nothing
# else, so the panel is a view of the same decisions the episode records are
# built from rather than a second account of the run. The two functions below
# are the whole model, and they are pure: `curses` draws what they return.


@dataclass
class Spectator:
    """What has been seen so far, as the panel needs it."""

    recent: deque[str] = field(default_factory=lambda: deque(maxlen=RECENT_ACTIONS))
    latest: DecisionView | None = None
    decisions: int = 0
    episodes_finished: int = 0
    final_waves: list[int] = field(default_factory=list)

    def observe(self, view: DecisionView) -> None:
        self.latest = view
        self.decisions += 1
        self.recent.append(f"e{view.episode} d{view.decision} {view.action}")
        if view.done:
            self.episodes_finished += 1
            self.final_waves.append(view.wave)

    @property
    def mean_final_wave(self) -> float:
        """The running mean of the waves finished episodes died at."""
        return statistics.fmean(self.final_waves) if self.final_waves else 0.0


def panel_lines(
    spectator: Spectator,
    *,
    policy: str,
    episodes_requested: int,
    elapsed_seconds: float,
    holding: bool = False,
) -> list[str]:
    """The lines to draw, top to bottom. The whole of the rendering decision.

    `elapsed_seconds` is passed in rather than read from a clock here, so the
    panel has no hidden state and a test can assert every line it produces.
    """
    view = spectator.latest
    of = "unlimited" if episodes_requested == 0 else str(episodes_requested)
    episode = view.episode if view is not None else 0
    per_minute = spectator.decisions / elapsed_seconds * 60 if elapsed_seconds > 0 else 0.0

    lines = [
        f"tower-rl spectate — {policy} — episode {episode} of {of}",
        "",
    ]
    if view is None:
        lines.append("waiting for the first decision")
    else:
        lines.append(
            f"wave {view.wave}   cash {view.cash:,.0f}   "
            f"health {view.health_fraction:.0%}   reward {view.reward:+.0f}"
        )
        lines.append(
            f"episodes played {spectator.episodes_finished}   "
            f"mean final wave {spectator.mean_final_wave:.2f}   "
            f"decisions {spectator.decisions}   {per_minute:.1f}/min"
        )
        if view.done:
            outcome = view.termination.value if view.termination is not None else "unknown"
            lines.append(f"episode {view.episode} ended at wave {view.wave}: {outcome}")
    lines.append("")
    lines.append(f"last {RECENT_ACTIONS} actions, newest first:")
    lines.extend(f"  {action}" for action in reversed(spectator.recent))
    lines.append("")
    lines.append(
        "session over — press any key to tear the instance down"
        if holding
        else "q to stop after this decision"
    )
    return lines


# -- panels ----------------------------------------------------------------


class Panel(Protocol):
    """Somewhere to draw the lines, and a way to hear a keypress.

    Two implementations: the curses panel a human watches, and the plain one
    that prints a line per decision, which is what a log wants and what makes
    an unattended session readable afterwards.
    """

    def draw(self, lines: Sequence[str]) -> None: ...

    def key(self) -> str:
        """The key pressed since the last call, or `""` for none."""
        ...

    def wait_for_key(self, timeout: float) -> None: ...


@dataclass
class PlainPanel:
    """One line per decision on stdout. Never reads a key."""

    def draw(self, lines: Sequence[str]) -> None:
        # The status line and the newest action: a log wants the run's progress,
        # not a redrawn screen.
        print(f"{lines[2]} | {lines[0]}", flush=True)

    def key(self) -> str:
        return ""

    def wait_for_key(self, timeout: float) -> None:
        print(f"holding the final state for {timeout:.0f}s", flush=True)
        time.sleep(timeout)


@dataclass
class CursesPanel:
    """The terminal panel, redrawn once per decision."""

    screen: Any

    def draw(self, lines: Sequence[str]) -> None:
        height, width = self.screen.getmaxyx()
        self.screen.erase()
        for row, line in enumerate(lines[: height - 1]):
            self.screen.addnstr(row, 0, line, width - 1)
        self.screen.refresh()

    def key(self) -> str:
        try:
            pressed = self.screen.getkey()
        except curses.error:  # nothing was pressed
            return ""
        return str(pressed)

    def wait_for_key(self, timeout: float) -> None:
        self.screen.nodelay(False)
        self.screen.getch()


# -- the exclusive-device refusal ------------------------------------------


def refuse_a_shared_host(devices: str, qemu_processes: str) -> None:
    """Refuse to spectate while anything else is on the host. Named, not silent.

    A spectated instance runs windowed at 60 Hz for as long as a human watches
    it, which is exactly the cost a measurement or a training run must not be
    asked to share. Both readings are taken because either can be the only one:
    `adb devices` misses an emulator whose adb has not come up, and a qemu
    process is what the emulator actually is.
    """
    attached = [
        line.split("\t")[0]
        for line in devices.splitlines()[1:]
        if line.strip() and not line.startswith("*")
    ]
    running = [line for line in qemu_processes.splitlines() if line.strip()]
    if not attached and not running:
        return
    raise SystemExit(
        "refusing to spectate while an emulator is running "
        f"(adb: {', '.join(attached) or 'none'}; qemu processes: {len(running)}). "
        "Spectating takes the host to itself: it runs windowed and in real time, "
        "and a measurement or a training run must never share that. Stop the "
        "other instance first."
    )


def host_is_free() -> None:
    """Take both readings the refusal is made from, then make it."""
    binary = find_android_tool("adb")
    devices = ""
    if binary is not None:
        devices = subprocess.run(
            [str(binary), "devices"], capture_output=True, text=True, timeout=30.0
        ).stdout
    qemu = subprocess.run(
        ["pgrep", "-af", "qemu-system"], capture_output=True, text=True, timeout=30.0
    ).stdout
    refuse_a_shared_host(devices, qemu)


# -- recording -------------------------------------------------------------


@dataclass
class GuestRecording:
    """`screenrecord` on the guest, in chunks, pulled to the host at the end.

    The guest tool stops itself after three minutes, which is a limit in the
    tool rather than something a flag lifts, so a longer session is recorded as
    consecutive numbered chunks: `<name>-000.mp4`, `<name>-001.mp4`, and so on.
    About a second is lost at each seam while the next chunk starts. This is a
    human-facing convenience and no measurement depends on it, which is why a
    lossy seam is acceptable here and would not be anywhere else.
    """

    instance: CloneInstance
    destination: Path
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _chunks: list[str] = field(default_factory=list, init=False)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def _record(self) -> None:
        while not self._stop.is_set():
            guest_path = f"/sdcard/tower-rl-spectate-{len(self._chunks):03d}.mp4"
            self._chunks.append(guest_path)
            try:
                adb(
                    self.instance,
                    "shell",
                    "screenrecord",
                    "--time-limit",
                    str(SCREENRECORD_CHUNK_SECONDS),
                    guest_path,
                    timeout=SCREENRECORD_CHUNK_SECONDS + 60.0,
                )
            except Exception as error:  # noqa: BLE001 - a lost recording is not a lost session
                print(f"recording stopped: {error}", flush=True)
                return

    def finish(self) -> list[Path]:
        """Stop the guest cleanly and pull every chunk. Never fatal."""
        self._stop.set()
        pulled: list[Path] = []
        try:
            # SIGINT rather than a kill: the guest tool finalises the file it is
            # writing on an interrupt and leaves an unplayable one on a kill.
            adb(self.instance, "shell", "pkill", "-INT", "screenrecord", timeout=30.0)
            if self._thread is not None:
                self._thread.join(timeout=60.0)
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            for index, guest_path in enumerate(self._chunks):
                local = self.destination.with_name(
                    f"{self.destination.stem}-{index:03d}{self.destination.suffix}"
                )
                adb(self.instance, "pull", guest_path, str(local), timeout=300.0)
                adb(self.instance, "shell", "rm", "-f", guest_path, timeout=30.0)
                if local.exists():
                    pulled.append(local)
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            print(f"could not retrieve the recording: {error}", flush=True)
        return pulled


# -- the session -----------------------------------------------------------


def spectate_session(
    environment: InstrumentedRunEnvironment,
    policy: Policy,
    panel: Panel,
    spectator: Spectator,
    *,
    episodes: int,
    policy_name: str,
    actor_id: str,
    started: float | None = None,
) -> list[EpisodeSummary]:
    """Play episodes, drawing the panel once per decision, until told to stop.

    The environment's decision stream is the whole of what the panel sees, and
    `q` is heard on the same stream: the stop is raised out of the observer, so
    it takes effect at the next decision rather than at the end of an episode
    that may be minutes away. The episode it interrupts has no summary, which is
    correct — it did not finish.
    """
    begun = time.monotonic() if started is None else started
    stopping = False

    def observe(view: DecisionView) -> None:
        nonlocal stopping
        spectator.observe(view)
        panel.draw(
            panel_lines(
                spectator,
                policy=policy_name,
                episodes_requested=episodes,
                elapsed_seconds=time.monotonic() - begun,
            )
        )
        if panel.key().lower() == "q":
            stopping = True
            raise SpectateStopped

    environment.on_decision = observe
    actor = Actor(
        environment=environment,
        policy=policy,
        config=ActorConfig(actor_id=actor_id),
        replay=None,
    )
    summaries: list[EpisodeSummary] = []
    try:
        while episodes == 0 or len(summaries) < episodes:
            try:
                summaries.append(actor.run_episode().summary)
            except SpectateStopped:
                break
            if stopping:
                break
    finally:
        environment.on_decision = None
    return summaries


def session_record(
    summaries: Sequence[EpisodeSummary], identity: dict[str, object], *, wall_seconds: float
) -> dict[str, Any]:
    """The same per-episode rows the fleet writes, for the episodes just played.

    `episode_record` is the evaluator's own row, so a spectated episode and a
    collected one are the same record. The panel is a view of these decisions,
    never a second source for them.
    """
    return {
        "policy_identity": dict(identity),
        "frame_rate_hz": SPECTATE_FRAME_RATE_HZ,
        "wall_seconds": round(wall_seconds, 1),
        "episodes": [episode_record(index, summary) for index, summary in enumerate(summaries)],
    }


def report_lines(spectator: Spectator, summaries: Sequence[EpisodeSummary]) -> list[str]:
    """What the session did, for stdout after the panel is gone."""
    waves = [summary.final_wave for summary in summaries]
    return [
        f"{len(summaries)} episodes, {spectator.decisions} decisions",
        f"final waves: {', '.join(str(wave) for wave in waves) or 'none'}",
        f"mean final wave: {statistics.fmean(waves):.2f}" if waves else "mean final wave: none",
    ]


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default="scripted",
        help=(
            f"the arm to watch: one of {sorted(POLICIES)}, or "
            f"{CHECKPOINT_SELECTOR}<path> for a checkpoint a training run left"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="episodes to play; 0 plays until you press q (default: one run to the death)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=10.0,
        help="how long --no-panel holds the final state before tearing down; "
        "the curses panel waits for a keypress instead",
    )
    parser.add_argument(
        "--frame-rate-hz",
        type=int,
        default=SPECTATE_FRAME_RATE_HZ,
        help=f"guest frame rate; {SPECTATE_FRAME_RATE_HZ} is real time, 120 is the fleet's",
    )
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--renderer", default="host")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument(
        "--no-panel",
        action="store_true",
        help="print one line per decision instead of drawing a terminal panel",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="record the guest screen to this .mp4; a session longer than "
        f"{SCREENRECORD_CHUNK_SECONDS}s is pulled back as numbered chunks",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="where the episode records are written; omitted, none are kept",
    )
    add_cadence_arguments(parser)
    arguments = parser.parse_args(argv)
    if arguments.episodes < 0:
        raise SystemExit("--episodes cannot be negative; 0 means until you press q")
    if arguments.frame_rate_hz not in (60, 120):
        raise SystemExit(
            "--frame-rate-hz is 60 (real time) or 120 (the fleet's rate) for a "
            "spectated session; anything else is a throughput choice and belongs "
            "to run_actors.py"
        )
    return arguments


def run(arguments: argparse.Namespace) -> int:
    """Bring one windowed instance up, watch it play, and always put it down."""
    host_is_free()
    # Before the device is touched: a checkpoint that cannot be rebuilt should
    # fail now, not after an emulator has been brought up for it.
    policy, identity = policy_from(arguments.policy)
    expected = compatibility(bridge_build_directory())
    instance = CloneInstance(index=arguments.instance_index)

    bring_up(
        instance,
        arguments.renderer,
        deploy=deploy_bridge,
        read_only=True,
        cores=arguments.cores,
        frame_rate_hz=arguments.frame_rate_hz,
        windowed=True,
    )
    recording: GuestRecording | None = None
    started = time.monotonic()
    summaries: list[EpisodeSummary] = []
    spectator = Spectator()
    try:
        require_offline(instance)
        require_game_activity(instance)
        raise_frame_rate(instance, arguments.frame_rate_hz)
        if arguments.record is not None:
            recording = GuestRecording(instance, arguments.record)
            recording.start()

        client = InstrumentedBridgeClient(
            "127.0.0.1",
            instance.bridge_host_port,
            expected_compatibility=expected,
            connect_timeout=5.0,
            read_timeout=120.0,
            heartbeat_timeout=60.0,
        )
        client.connect()
        adapter = InstrumentedRunAdapter(client=client)
        environment = InstrumentedRunEnvironment(
            port=adapter,
            builder=RunStateBuilder(profile_id=expected.profile_id),
            cadence=cadence_from(arguments),
        )
        try:
            summaries = watch(environment, policy, spectator, arguments, identity)
        finally:
            adapter.release()
            client.close()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        if recording is not None:
            for path in recording.finish():
                print(f"recording: {path}", flush=True)
        tear_down_instance(instance)

    if arguments.output_directory is not None and summaries:
        arguments.output_directory.mkdir(parents=True, exist_ok=True)
        record = session_record(summaries, identity, wall_seconds=time.monotonic() - started)
        output = arguments.output_directory / f"{instance.serial}.json"
        output.write_text(json.dumps(record, indent=2))
        print(f"episodes: {output}", flush=True)
    for line in report_lines(spectator, summaries):
        print(line, flush=True)
    return 0


def watch(
    environment: InstrumentedRunEnvironment,
    policy: Policy,
    spectator: Spectator,
    arguments: argparse.Namespace,
    identity: dict[str, object],
) -> list[EpisodeSummary]:
    """Run the session inside whichever panel was asked for, and hold at the end.

    The hold is the point of watching one run: the last thing that happens is
    the tower dying, and tearing the window down on the same instant leaves
    nothing to look at. The curses panel waits for a keypress; `--no-panel`,
    which is what an unattended session uses, waits `--hold-seconds`.
    """
    name = str(identity["name"])
    actor_id = f"spectate:{name}"

    def session(panel: Panel) -> list[EpisodeSummary]:
        summaries = spectate_session(
            environment,
            policy,
            panel,
            spectator,
            episodes=arguments.episodes,
            policy_name=name,
            actor_id=actor_id,
        )
        panel.draw(
            panel_lines(
                spectator,
                policy=name,
                episodes_requested=arguments.episodes,
                elapsed_seconds=0.0,
                holding=True,
            )
        )
        panel.wait_for_key(arguments.hold_seconds)
        return summaries

    if arguments.no_panel:
        return session(PlainPanel())

    def inside_curses(screen: Any) -> list[EpisodeSummary]:
        curses.curs_set(0)
        screen.nodelay(True)
        return session(CursesPanel(screen))

    result: list[EpisodeSummary] = curses.wrapper(inside_curses)
    return result


def main() -> int:
    return run(parse_arguments())


if __name__ == "__main__":
    raise SystemExit(main())
