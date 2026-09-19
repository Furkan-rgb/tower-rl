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

The guest renders through `-gpu lavapipe` by default, not the host renderer the
fleet uses: the host renderer glitches the picture, which makes a recording of
it useless. `--record` and the per-run episode JSON both land under
`state/recordings/`, inside the project's git-ignored state directory — an mp4
is far above GitHub's file limit, and this repo is public.

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tower_rl.environment.project_state import state_directory  # noqa: E402

#: Where a spectated session's output lives: recordings and their per-run
#: records in one place under the project's state directory, git-ignored
#: because an mp4 is far above GitHub's file limit and this repo is public.
#: `state_directory()` resolves from the package's own location rather than the
#: cwd, so this lands beside the project whichever directory it is run from.
RECORDINGS_DIRECTORY = state_directory() / "recordings"

# One torch thread, for the same reason `run_episodes.py` sets it: acting is one
# small forward pass per decision and a pool buys nothing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from run_episodes import (  # noqa: E402
    CHECKPOINT_SELECTOR,
    POLICIES,
    add_cadence_arguments,
    cadence_from,
    decision_cadence_from,
    policy_from,
)

from tower_rl.environment.episode import DecisionView, EpisodeSummary  # noqa: E402
from tower_rl.environment.run_environment import InstrumentedRunEnvironment  # noqa: E402
from tower_rl.environment.run_state import (  # noqa: E402
    NO_ENEMY_DISTANCE,
    RunStateBuilder,
)
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
from tower_rl.simulation.instrumented_bridge import (  # noqa: E402
    InstrumentedBridgeClient,
    UpgradeSlotLabel,
)
from tower_rl.simulation.instrumented_run_adapter import InstrumentedRunAdapter  # noqa: E402

#: The rate a spectated instance runs at. 60 Hz is the guest's stock rate, which
#: is real time: one game second per wall second. The fleet's 120 exists to buy
#: throughput, which is worth nothing to somebody watching.
SPECTATE_FRAME_RATE_HZ = 60

#: How many past actions the panel keeps. Twenty is what fits beside the game
#: window without the panel becoming the thing being read.
RECENT_ACTIONS = 20
#: Where `panel_lines` puts the line naming the episode that just ended, and
#: leaves blank while one is running. The slot is always there so both panels
#: can find it at the same index: the plain one prints it to a log, and the
#: curses one draws it.
DEATH_LINE = 4
#: Where `panel_lines` puts the two live-reading lines — the tower stats and the
#: wave block, in the game's own units. Like `DEATH_LINE` the slots are always
#: there, blank before the first decision, so both panels index them the same
#: way: the curses one draws them and the plain one prints them to a log, which
#: is what lets a recording be lined up against what the agent actually saw.
HUD_LINES = slice(DEATH_LINE + 2, DEATH_LINE + 4)

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
    #: The range each live reading took this session, in the game's own unit:
    #: `[min, max, first, last]` per `Main` field. The panel is what a human
    #: checks against the HUD; this is the same readings in a form a record can
    #: be checked against afterwards, and it is the shape the device capture in
    #: `FIELD-SAMPLES.md` (board #39) reports, so the two compare directly.
    live_ranges: dict[str, list[float]] = field(default_factory=dict)

    def observe(self, view: DecisionView) -> None:
        self.latest = view
        self.decisions += 1
        for name, value in view.hud.items():
            seen = self.live_ranges.get(name)
            if seen is None:
                self.live_ranges[name] = [value, value, value, value]
            else:
                seen[0] = min(seen[0], value)
                seen[1] = max(seen[1], value)
                seen[3] = value
        self.recent.append(f"e{view.episode} d{view.decision} {view.action}")
        if view.done:
            self.episodes_finished += 1
            self.final_waves.append(view.wave)

    @property
    def mean_final_wave(self) -> float:
        """The running mean of the waves finished episodes died at."""
        return statistics.fmean(self.final_waves) if self.final_waves else 0.0


#: The tower stats the panel shows, and what the HUD calls each one. Compact on
#: purpose: these are the rows a watcher can find on screen and check against,
#: not every reading `observation-v2` carries.
TOWER_STATS: tuple[tuple[str, str, str], ...] = (
    ("damage", "dmg", "{:.1f}"),
    ("attackSpeed", "aspd", "{:.2f}"),
    ("criticalChance", "crit%", "{:.1f}"),
    ("criticalMult", "critx", "{:.2f}"),
    ("towerRangeDistance", "range", "{:.2f}"),
    ("towerHealthRegen", "regen", "{:.3f}"),
    ("defenseAbs", "defabs", "{:.1f}"),
    ("defenseRel", "def%", "{:.1f}"),
    ("wallHealth", "wall", "{:.1f}"),
)


#: The wave-and-enemy readings the panel's second line is built from, under the
#: game's own `Main` names.
WAVE_BLOCK_WIRES: tuple[str, ...] = (
    "closestEnemyDistance",
    "enemiesKilledThisWave",
    "enemiesSpawnedThisWave",
    "estimatedEnemiesToSpawnThisWave",
    "waveTimer",
    "waveLengthSeconds",
    "waveCooldownSeconds",
    "currentWaveBaseHealth",
    "currentWaveBaseDamage",
    "gameplayTimeThisRound",
    "bossWaveBool",
    "bossSpawnedBool",
    "miniBossWaveBool",
)

#: Every reading the panel puts on screen: the tower stats and the wave block.
#: This is what `DecisionTrack` writes beside a recording, so the rendered panel
#: shows the watcher's own two lines rather than a second selection of readings
#: - `observation-v2` carries far more, and none of the rest is on the panel.
PANEL_HUD_WIRES: tuple[str, ...] = tuple(wire for wire, _, _ in TOWER_STATS) + WAVE_BLOCK_WIRES

#: What a decision that bought nothing is called. The slot labels name what was
#: bought; waiting has no slot, so it is named here.
HOLD_LABEL = "Hold"


def hud_lines(view: DecisionView) -> list[str]:
    """The tower stats and the wave/enemy block, as the HUD shows them.

    Two lines, compact, in the game's own units, so a watcher can hold them
    beside the screen. This is the whole check that `observation-v2` reads what
    the player reads: the agent's view of the game, written where a human can
    see it disagree with the game.
    """
    hud = view.hud
    stats = "  ".join(
        f"{label} {form.format(hud[wire])}" for wire, label, form in TOWER_STATS
    )
    distance = hud["closestEnemyDistance"]
    nearest = "none" if distance >= NO_ENEMY_DISTANCE else f"{distance:.2f}"
    boss = ""
    if hud["bossWaveBool"]:
        boss = "  BOSS" + (" spawned" if hud["bossSpawnedBool"] else "")
    elif hud["miniBossWaveBool"]:
        boss = "  mini-boss"
    wave = (
        f"enemies {hud['enemiesKilledThisWave']:.0f}/{hud['enemiesSpawnedThisWave']:.0f}"
        f" of ~{hud['estimatedEnemiesToSpawnThisWave']:.0f}   nearest {nearest}   "
        f"wave clock {hud['waveTimer']:.1f}/{hud['waveLengthSeconds']:.0f}"
        f"+{hud['waveCooldownSeconds']:.0f}s   base hp {hud['currentWaveBaseHealth']:.1f}"
        f"   base dmg {hud['currentWaveBaseDamage']:.2f}   "
        f"round {hud['gameplayTimeThisRound']:.0f}s{boss}"
    )
    return [stats, wave]


def label_lines(labels: Sequence[UpgradeSlotLabel]) -> list[str]:
    """What each offered upgrade row is called, one line per family.

    The policy sees a numbered slot; a human cannot. These names are the game's
    own, read once per session, and they are here and in the session record for
    exactly that reason - never in the observation tensor, where a renamed row
    would change what a checkpoint means.
    """
    if not labels:
        return []
    lines = []
    for family in ("attack", "defense", "utility"):
        named = [
            f"{label.index}:{label.name}"
            for label in labels
            if label.family == family and label.name
        ]
        lines.append(f"  {family:<8}{' '.join(named)}")
    return ["upgrade rows:", *lines]


def panel_lines(
    spectator: Spectator,
    *,
    policy: str,
    renderer: str,
    episodes_requested: int,
    elapsed_seconds: float,
    labels: Sequence[UpgradeSlotLabel] = (),
    holding: bool = False,
) -> list[str]:
    """The lines to draw, top to bottom. The whole of the rendering decision.

    `elapsed_seconds` is passed in rather than read from a clock here, so the
    panel has no hidden state and a test can assert every line it produces.
    `renderer` is named on the same line for the same reason: what rendered the
    picture is part of what a watcher, or a log read afterwards, needs to know
    it was looking at.

    The first five lines are a fixed layout, because both panels index into it:
    title, blank, state, counters, and `DEATH_LINE` - the episode that just
    ended, or blank.
    """
    view = spectator.latest
    of = "unlimited" if episodes_requested == 0 else str(episodes_requested)
    episode = view.episode if view is not None else 0
    per_minute = spectator.decisions / elapsed_seconds * 60 if elapsed_seconds > 0 else 0.0

    lines = [
        f"tower-rl spectate — {policy} — {renderer} — episode {episode} of {of}",
        "",
    ]
    if view is None:
        lines.append("waiting for the first decision")
        lines.append("")
    else:
        lines.append(
            f"wave {view.wave}   cash {view.cash:,.0f}   "
            f"health {view.health_fraction:.0%}   reward {view.reward:+.0f}   "
            # How long the agent held this decision: under `choice-points` a
            # decision covers every forced-WAIT slice the environment advanced
            # through, so a watcher sees holds of seconds, not of one slice.
            f"held {view.game_ms / 1000:.1f}s"
        )
        lines.append(
            f"episodes played {spectator.episodes_finished}   "
            f"mean final wave {spectator.mean_final_wave:.2f}   "
            f"decisions {spectator.decisions}   {per_minute:.1f}/min"
        )
    # `DEATH_LINE` always exists and is blank unless this very decision ended an
    # episode, so a reader at that index needs no other way to ask.
    ended = ""
    if view is not None and view.done:
        outcome = view.termination.value if view.termination is not None else "unknown"
        ended = f"episode {view.episode} ended at wave {view.wave}: {outcome}"
    lines.append(ended)
    lines.append("")
    # Two slots, always present so both panels can index them, and blank until
    # there is a decision to read them from.
    lines.extend(hud_lines(view) if view is not None else ["", ""])
    lines.append("")
    lines.extend(label_lines(labels))
    if labels:
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

    def hold(self, timeout: float) -> None:
        """Keep the final state on screen for at most `timeout` seconds."""
        ...


@dataclass
class PlainPanel:
    """One line per decision on stdout. Never reads a key."""

    def draw(self, lines: Sequence[str]) -> None:
        # The status line and the newest action: a log wants the run's progress,
        # not a redrawn screen. `panel_lines` puts the ended-episode line at
        # index 4 and leaves that slot blank while the episode is running, so
        # the death reaches a log on the decision it happened on rather than
        # only the watcher of a terminal panel.
        print(f"{lines[2]} | {lines[0]}", flush=True)
        if len(lines) > DEATH_LINE and lines[DEATH_LINE]:
            print(lines[DEATH_LINE], flush=True)
        # The live readings too, in the game's own units. A session watched
        # through this panel is one somebody reads afterwards, and lining a
        # recording up against what the agent saw is the only way
        # `observation-v2` gets checked against the game rather than itself.
        for line in lines[HUD_LINES]:
            if line:
                print(f"  {line}", flush=True)

    def key(self) -> str:
        return ""

    def hold(self, timeout: float) -> None:
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

    def hold(self, timeout: float) -> None:
        """Wait for a key, but not forever: the timeout is honoured either way.

        A session nobody came back to must still put its instance down, and an
        emulator left running because a panel blocked on a key is the same
        device-safety failure as one left running by a skipped teardown.
        """
        self.screen.nodelay(True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.screen.getch() != -1:
                return
            time.sleep(0.05)


# -- the exclusive-device refusal ------------------------------------------


def running_emulators(proc: Path = Path("/proc")) -> list[str]:
    """Every live process whose executable is a `qemu-system-*`, by pid.

    Read from `/proc/<pid>/exe`, which is the kernel's own answer to "what is
    this process running": a symlink to the binary itself. Not `pgrep -f`,
    which matches a *command line* - it would report this script for having the
    word in an argument, report an editor with the emulator's log open, and
    miss a qemu whose argv was rewritten. The link is unreadable for processes
    this user does not own, and those are skipped rather than guessed at: an
    emulator started by somebody else is not one this session can stop anyway.
    """
    found: list[str] = []
    try:
        entries = sorted(proc.iterdir())
    except OSError:  # no procfs to read; the adb reading below still stands
        return found
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            executable = os.readlink(entry / "exe")
        except OSError:  # gone, or not ours to look at
            continue
        if Path(executable).name.startswith("qemu-system"):
            found.append(f"{entry.name} {executable}")
    return found


def refuse_a_shared_host(devices: str, qemu_processes: Sequence[str]) -> None:
    """Refuse to spectate while anything else is on the host. Named, not silent.

    A spectated instance runs windowed at 60 Hz for as long as a human watches
    it, which is exactly the cost a measurement or a training run must not be
    asked to share. Both readings are taken because either can be the only one:
    `adb devices` misses an emulator whose adb has not come up, and a qemu
    process is what an emulator actually is however adb sees it.
    """
    attached = [
        line.split("\t")[0]
        for line in devices.splitlines()[1:]
        if line.strip() and not line.startswith("*")
    ]
    if not attached and not qemu_processes:
        return
    raise SystemExit(
        "refusing to spectate while an emulator is running "
        f"(adb: {', '.join(attached) or 'none'}; "
        f"qemu processes: {', '.join(qemu_processes) or 'none'}). "
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
    refuse_a_shared_host(devices, running_emulators())


# -- recording -------------------------------------------------------------


class RunGuestCommand(Protocol):
    """How a command reaches the guest: `simulation.instance.adb`'s shape."""

    def __call__(self, instance: CloneInstance, *args: str, timeout: float = 30.0) -> str: ...


@dataclass
class GuestRecording:
    """`screenrecord` on the guest, in chunks, pulled to the host at the end.

    The guest tool stops itself after three minutes, which is a limit in the
    tool rather than something a flag lifts, so a longer session is recorded as
    consecutive numbered chunks: `<name>-000.mp4`, `<name>-001.mp4`, and so on.
    About a second is lost at each seam while the next chunk starts. This is a
    human-facing convenience and no measurement depends on it, which is why a
    lossy seam is acceptable here and would not be anywhere else.

    Every chunk is whole. `finish` stops the guest with SIGINT, and on this
    image `screenrecord` answers an interrupt by finalising the file it is
    writing and exiting 0 - three interrupted chunks were pulled and read back
    as valid MP4 with durations (`M2-E003`). So there is no truncated-chunk
    case to name: a chunk this pulls is playable whether it ran its three
    minutes out or was cut short, and only its duration says which.

    The lifecycle is the part worth stating. `finish` sets the stop flag
    *before* it interrupts the guest, and the loop tests that flag before
    starting a chunk, so no chunk can be started after a stop has begun -
    which would leave a file on the guest that nothing afterwards pulls or
    removes. `_chunks` is read only once the recording thread has been joined,
    so the list is never walked while it is being appended to.
    """

    instance: CloneInstance
    destination: Path
    #: How a guest command is run. `adb` is the only implementation; it is a
    #: field so the lifecycle above can be tested without a device.
    run: RunGuestCommand = adb
    #: The clock every time here is read from. A field for the same reason `run`
    #: is one: the timing below is what a decision is placed in the video by,
    #: and it has to be assertable without a device.
    clock: Callable[[], float] = time.monotonic
    #: When the recording began, on `clock`: `None` until `start`. Everything
    #: written beside the video is timed from this instant.
    anchor: float | None = field(default=None, init=False)
    #: When each chunk began, on the same clock, in the order they were started.
    #: Appended by the recording thread and read by the session's thread, which
    #: is safe for a list of floats and is all that is asked of it: a reader
    #: that misses the newest entry by a moment places that decision in the
    #: chunk before, one seam early.
    chunk_starts: list[float] = field(default_factory=list, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    #: The guest path of every chunk started, in order, which is also the order
    #: they are pulled and removed in.
    _chunks: list[str] = field(default_factory=list, init=False)

    def start(self) -> float:
        """Begin recording, and answer the instant everything is timed from.

        The anchor is taken here, immediately before the guest is asked to
        record, because that is the earliest moment the host can name. What the
        guest does next - an adb round trip and a `screenrecord` process start -
        happens after it, so the video begins slightly *later* than the anchor
        says and every `video_s` is early by that much. Measured against a
        session's own seams this is a few hundred milliseconds, under a second;
        the same order as the second already lost at each chunk seam, and the
        reason nothing measured is ever read off this timing.
        """
        self.anchor = self.clock()
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()
        return self.anchor

    def _record(self) -> None:
        """Record chunk after chunk until asked to stop. Never raises."""
        while not self._stop.is_set():
            guest_path = f"/sdcard/tower-rl-spectate-{len(self._chunks):03d}.mp4"
            self._chunks.append(guest_path)
            self.chunk_starts.append(self.clock())
            try:
                self.run(
                    self.instance,
                    "shell",
                    f"screenrecord --time-limit {SCREENRECORD_CHUNK_SECONDS} {guest_path}",
                    timeout=SCREENRECORD_CHUNK_SECONDS + 60.0,
                )
            except Exception as error:  # noqa: BLE001 - a lost recording is not a lost session
                # What reached the guest is still pulled below: a recording of
                # the moment somebody wanted to watch is worth more than none.
                print(f"recording stopped: {error}", flush=True)
                return

    def finish(self) -> list[Path]:
        """Stop the guest cleanly and pull every chunk. Never fatal."""
        # Before the interrupt, always: the loop must see the stop first, or it
        # starts a chunk nothing below knows to pull.
        self._stop.set()
        pulled: list[Path] = []
        try:
            # SIGINT rather than a kill: the guest tool finalises the file it is
            # writing on an interrupt and leaves an unplayable one on a kill.
            self.run(self.instance, "shell", "pkill", "-INT", "screenrecord", timeout=30.0)
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            print(f"could not stop the guest recording: {error}", flush=True)
        if self._thread is not None:
            self._thread.join(timeout=SCREENRECORD_CHUNK_SECONDS + 60.0)
        try:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            for index, guest_path in enumerate(self._chunks):
                local = self.destination.with_name(
                    f"{self.destination.stem}-{index:03d}{self.destination.suffix}"
                )
                self.run(self.instance, "pull", guest_path, str(local), timeout=300.0)
                self.run(self.instance, "shell", "rm", "-f", guest_path, timeout=30.0)
                if local.exists():
                    pulled.append(local)
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            print(f"could not retrieve the recording: {error}", flush=True)
        return pulled


@dataclass(frozen=True)
class VideoPosition:
    """Where a moment lands in a recording: overall, and inside which chunk.

    Both are needed and neither substitutes for the other. `video_s` is time
    since the recording began, which is what a human reads; `chunk`/`chunk_s`
    is where the frame actually is, because a chunk seam loses about a second
    of wall time that the concatenated video does not contain. A renderer
    places a decision from the chunk pair, never from `video_s`.
    """

    video_s: float
    chunk: int
    chunk_s: float


def video_position(now: float, anchor: float, chunk_starts: Sequence[float]) -> VideoPosition:
    """Place `now` in the recording anchored at `anchor`, by chunk.

    The chunk is the last one started at or before `now`; a moment before the
    first chunk started - the window between the anchor and the guest process
    coming up - is placed at the start of chunk zero rather than before it,
    because there is no earlier frame for it to belong to.
    """
    video_s = max(0.0, now - anchor)
    chunk = 0
    began = anchor
    for index, start in enumerate(chunk_starts):
        if start > now:
            break
        chunk, began = index, start
    return VideoPosition(
        video_s=round(video_s, 3),
        chunk=chunk,
        chunk_s=round(max(0.0, now - began), 3),
    )


@dataclass
class DecisionTrack:
    """Every decision of a recorded session, timed against the video.

    One JSON object per line beside the mp4, written as the decision happens
    and flushed, so a session that ended in a crash still says what the agent
    did up to it. This is not a second account of the run - the durable record
    is still `EpisodeSummary` - it is the decision stream with a timestamp
    added, which is the one thing a video needs and an episode record does not
    carry: *when*.

    The timing is the part to be honest about. Times are read from the same
    monotonic clock the recording was anchored on, and the anchor is the moment
    the host asked the guest to record, not the moment the guest captured its
    first frame: the adb round trip and the `screenrecord` process start fall
    between them, so the picture is a few hundred milliseconds - under a second
    - behind what is written here, and the panel leads the video by that much.
    That is the same order as the second lost at each chunk seam. Watching is a
    human-facing path and nothing measured is read off this timing.
    """

    path: Path
    #: The recording's anchor, on `clock`: what every time here is relative to.
    anchor: float
    #: The recording's chunk starts, on the same clock. The live list from
    #: `GuestRecording`, so a chunk begun after this was built still counts.
    chunk_starts: Sequence[float] = ()
    #: What the game calls each upgrade slot, so a line says `Critical Chance`
    #: and not only `attack:2`. Set once the bridge has been asked, which is
    #: after the recording has already started.
    labels: Sequence[UpgradeSlotLabel] = ()
    clock: Callable[[], float] = time.monotonic
    _file: IO[str] | None = field(default=None, init=False)

    def label_for(self, action: str) -> str:
        """The game's name for what this action bought, or `Hold` for a wait."""
        family, _, index = action.partition(":")
        if not index:
            return HOLD_LABEL
        for label in self.labels:
            if label.family == family and str(label.index) == index:
                return label.name or action
        return action

    def write(self, view: DecisionView) -> None:
        """Append one decision, placed in the video."""
        position = video_position(self.clock(), self.anchor, self.chunk_starts)
        line: dict[str, Any] = {
            "video_s": position.video_s,
            "chunk": position.chunk,
            "chunk_s": position.chunk_s,
            "episode": view.episode,
            "decision": view.decision,
            "wave": view.wave,
            "cash": round(view.cash, 1),
            "health_fraction": round(view.health_fraction, 4),
            "action": view.action,
            "label": self.label_for(view.action),
            # Game time, which is what the agent held the choice for; at 60 Hz
            # that is also wall time, and at 120 it is half of it.
            "held_s": round(view.game_ms / 1000, 3),
            "hud": {
                wire: round(float(view.hud[wire]), 4)
                for wire in PANEL_HUD_WIRES
                if wire in view.hud
            },
            "ended": view.done,
        }
        if view.done:
            line["reason"] = view.termination.value if view.termination is not None else "unknown"
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w")
        self._file.write(json.dumps(line) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def decisions_beside(video: Path) -> Path:
    """Where the decision track for `video` lives: beside it, on its stem.

    The chunks are `<stem>-000.mp4`, `<stem>-001.mp4`, and the track is
    `<stem>.decisions.jsonl` - one per session rather than one per chunk,
    because each line already says which chunk it is in.
    """
    return video.with_name(f"{video.stem}.decisions.jsonl")


# -- the session -----------------------------------------------------------


def spectate_session(
    environment: InstrumentedRunEnvironment,
    policy: Policy,
    panel: Panel,
    spectator: Spectator,
    summaries: list[EpisodeSummary],
    *,
    episodes: int,
    policy_name: str,
    actor_id: str,
    renderer: str,
    labels: Sequence[UpgradeSlotLabel] = (),
    track: DecisionTrack | None = None,
) -> None:
    """Play episodes, drawing the panel once per decision, until told to stop.

    The environment's decision stream is the whole of what the panel sees, and
    `q` is heard on the same stream: the stop is raised out of the observer, so
    it takes effect at the next decision rather than at the end of an episode
    that may be minutes away. The episode it interrupts has no summary, which is
    correct — it did not finish.

    `summaries` belongs to the caller and is appended to as each episode ends,
    rather than returned at the end. A Ctrl-C arrives as a `KeyboardInterrupt`
    inside whichever episode was running, and an episode that finished before
    it is a real episode: the caller still holds every one of them, so stopping
    that way keeps exactly what stopping with `q` keeps.
    """
    begun = time.monotonic()

    def observe(view: DecisionView) -> None:
        spectator.observe(view)
        # Written before the panel is drawn: a line the video can be read
        # against is worth more than the redraw that follows it, and the draw
        # is what raises the stop.
        if track is not None:
            track.write(view)
        panel.draw(
            panel_lines(
                spectator,
                policy=policy_name,
                renderer=renderer,
                episodes_requested=episodes,
                elapsed_seconds=time.monotonic() - begun,
                labels=labels,
            )
        )
        if panel.key().lower() == "q":
            raise SpectateStopped

    environment.on_decision = observe
    actor = Actor(
        environment=environment,
        policy=policy,
        config=ActorConfig(actor_id=actor_id),
        replay=None,
    )
    try:
        while episodes == 0 or len(summaries) < episodes:
            try:
                summaries.append(actor.run_episode().summary)
            except SpectateStopped:
                break
    finally:
        environment.on_decision = None


def session_record(
    summaries: Sequence[EpisodeSummary],
    identity: dict[str, object],
    *,
    frame_rate_hz: int,
    decision_cadence: str,
    wall_seconds: float,
    labels: Sequence[UpgradeSlotLabel] = (),
    live_ranges: Mapping[str, Sequence[float]] | None = None,
    recording: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The same per-episode rows the fleet writes, for the episodes just played.

    `episode_record` is the evaluator's own row, so a spectated episode and a
    collected one are the same record. The panel is a view of these decisions,
    never a second source for them.

    The rate recorded is the rate the session ran at, not the default: a record
    that named the constant would say 60 for episodes played at 120, and the
    rate is exactly what makes two records comparable or not.
    """
    return {
        "policy_identity": dict(identity),
        "frame_rate_hz": frame_rate_hz,
        # The protocol these episodes were played under (ADR 0009).
        "decision_cadence": decision_cadence,
        "wall_seconds": round(wall_seconds, 1),
        # What the game calls each slot the actions address. For the human
        # reading this record afterwards: `attack:3` alone does not say which
        # upgrade the agent bought. Never an input to anything - a renamed row
        # must not change what a checkpoint means.
        "upgrade_rows": [
            {"family": label.family, "index": label.index, "name": label.name,
             "description": label.description}
            for label in labels
            if label.name
        ],
        # What each live reading actually read this session, in the game's own
        # unit: `{field: {min, max, first, last}}`. The panel is what a human
        # checks against the HUD; this is what a record can be checked against
        # afterwards, and it is the shape board #39's device capture reports.
        "live_readings": {
            name: dict(zip(("min", "max", "first", "last"), seen, strict=True))
            for name, seen in sorted((live_ranges or {}).items())
        },
        # The recording this session left, if it left one: where the chunks
        # and the decision track are, and the monotonic instant both are timed
        # from. The anchor is here rather than only in the track because it is
        # what says how the two files line up, and a record read afterwards is
        # where that question is asked.
        **({"recording": dict(recording)} if recording is not None else {}),
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
    parser.add_argument(
        "--renderer",
        default="lavapipe",
        help="guest GPU renderer; lavapipe is the default because the host "
        "renderer glitches the picture, which makes a recording of it useless",
    )
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
        help="record the guest screen to this .mp4; a relative filename is "
        f"written under {RECORDINGS_DIRECTORY}, an absolute path elsewhere; a "
        f"session longer than {SCREENRECORD_CHUNK_SECONDS}s is pulled back as "
        "numbered chunks",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=RECORDINGS_DIRECTORY / "records",
        help="where the episode records are written",
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
    if arguments.record is not None and not arguments.record.is_absolute():
        arguments.record = RECORDINGS_DIRECTORY / arguments.record
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
    track: DecisionTrack | None = None
    started = time.monotonic()
    summaries: list[EpisodeSummary] = []
    spectator = Spectator()
    labels: tuple[UpgradeSlotLabel, ...] = ()
    try:
        require_offline(instance)
        require_game_activity(instance)
        raise_frame_rate(instance, arguments.frame_rate_hz)
        if arguments.record is not None:
            recording = GuestRecording(instance, arguments.record)
            # The anchor comes back from `start` rather than being read off the
            # clock again here: what the video is timed from is the instant the
            # guest was asked to record, and only `start` knows it.
            track = DecisionTrack(
                path=decisions_beside(arguments.record),
                anchor=recording.start(),
                chunk_starts=recording.chunk_starts,
            )

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
        # Before the first round, which is the only time a command of the
        # adapter's own initiative may be issued: the labels are constant for
        # the build, so once is all this session needs.
        labels = adapter.slot_labels()
        if track is not None:
            # The rows are read after the recording starts, so the track is
            # told their names now rather than being built with them.
            track.labels = labels
        environment = InstrumentedRunEnvironment(
            port=adapter,
            builder=RunStateBuilder(profile_id=expected.profile_id),
            cadence=cadence_from(arguments),
            decision_cadence=decision_cadence_from(arguments),
        )
        try:
            watch(
                environment, policy, spectator, summaries, arguments, identity, labels, track
            )
        finally:
            adapter.release()
            client.close()
    except KeyboardInterrupt:
        # Every episode that finished before the interrupt is still an episode,
        # and `summaries` is owned here rather than returned, so it holds them.
        print("stopped", flush=True)
    finally:
        if track is not None:
            track.close()
        if recording is not None:
            for path in recording.finish():
                print(f"recording: {path}", flush=True)
            if track is not None and track.path.exists():
                print(f"decisions: {track.path}", flush=True)
        tear_down_instance(instance)

    if arguments.output_directory is not None and summaries:
        arguments.output_directory.mkdir(parents=True, exist_ok=True)
        record = session_record(
            summaries,
            identity,
            frame_rate_hz=arguments.frame_rate_hz,
            decision_cadence=str(decision_cadence_from(arguments)),
            wall_seconds=time.monotonic() - started,
            labels=labels,
            live_ranges=spectator.live_ranges,
            recording=(
                {
                    "video": str(arguments.record),
                    "decisions": str(track.path),
                    "anchor_monotonic": round(track.anchor, 3),
                    # Each chunk's start as an offset from the anchor, which is
                    # what the seams cost: the gap between one chunk's start
                    # and the last is a chunk's length plus the lost second.
                    "chunk_start_offsets_s": [
                        round(start - track.anchor, 3) for start in recording.chunk_starts
                    ],
                }
                if track is not None and recording is not None
                else None
            ),
        )
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
    summaries: list[EpisodeSummary],
    arguments: argparse.Namespace,
    identity: dict[str, object],
    labels: Sequence[UpgradeSlotLabel] = (),
    track: DecisionTrack | None = None,
) -> None:
    """Run the session inside whichever panel was asked for, and hold at the end.

    The hold is the point of watching one run: the last thing that happens is
    the tower dying, and tearing the window down on the same instant leaves
    nothing to look at. The curses panel waits for a key or `--hold-seconds`,
    whichever comes first; `--no-panel`, which is what an unattended session
    uses, waits out `--hold-seconds`.
    """
    name = str(identity["name"])
    actor_id = f"spectate:{name}"

    def session(panel: Panel) -> None:
        spectate_session(
            environment,
            policy,
            panel,
            spectator,
            summaries,
            episodes=arguments.episodes,
            policy_name=name,
            actor_id=actor_id,
            renderer=arguments.renderer,
            labels=labels,
            track=track,
        )
        panel.draw(
            panel_lines(
                spectator,
                policy=name,
                renderer=arguments.renderer,
                episodes_requested=arguments.episodes,
                elapsed_seconds=0.0,
                labels=labels,
                holding=True,
            )
        )
        panel.hold(arguments.hold_seconds)

    if arguments.no_panel:
        session(PlainPanel())
        return

    def inside_curses(screen: Any) -> None:
        curses.curs_set(0)
        screen.nodelay(True)
        session(CursesPanel(screen))

    curses.wrapper(inside_curses)


def main() -> int:
    return run(parse_arguments())


if __name__ == "__main__":
    raise SystemExit(main())
