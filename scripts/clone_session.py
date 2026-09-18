#!/usr/bin/env python3
"""Bring a disposable clone instance up offline, and snapshot it already running.

The game cannot cold-launch without a network: it stops at a Firebase
online-status check and an OFFLINE modal, and never reaches the battle home
screen (`M1B-E010`). It plays fine once the network is cut. So the only online
window is application startup, and this script exists to make that window short,
verified, and identical every time rather than a sequence typed by hand.

`snapshot` removes the window entirely: an emulator snapshot taken while the game
is up and idle with the radios already down restores into an already-started,
already-offline game. A restore also skips whatever intro a cold boot walks
through, and — the reason that matters most for the benchmark — it starts from
byte-identical account state, so the progression this account accumulates
between runs cannot drift between two arms of a comparison the way it did in the
frame-size sweep (`M1B-E018`).

A snapshot carries the bridge that was deployed when it was taken, and a stale
bridge does not answer the current client, so each snapshot is named for the
bridge inside it: `tower_clone_home_offline_<key>`, where the key is a hash of
`libtower_bridge.so` in the private build directory. `up` is therefore the normal
way an instance comes up:

    uv run python scripts/clone_session.py up

It restores that snapshot when the AVD holds one for the bridge we are about to
deploy, verifies it (offline by interface, game process alive, the bridge's own
readiness reading) and connects. Otherwise — no snapshot, a snapshot for another
bridge, or a restore that does not verify — it takes the cold path once, with its
one online window, and saves the snapshot the next `up` restores. `up --cold`
forces the cold path.

Between arms of a comparison, restore the pinned state deliberately so each arm
starts from the same account:

    uv run python scripts/clone_session.py restore     # the current bridge's snapshot

Several instances can run at once from the one clone AVD. `--read-only` gives
each instance its own writable overlay over the untouched base image, so N
actors need N overlays rather than N copies of a multi-gigabyte AVD. The
instance is addressed by `--index`, a top-level flag that precedes the
subcommand: index 0 is today's `emulator-5556`, and each further index takes
the next even console port. A read-only instance cannot save a snapshot; take
snapshots on index 0 without `--read-only`, run alone.

The emulator refuses to share one AVD unless *every* instance holding it is
`-read-only` — including index 0. Bringing up a second instance while index 0
is running writable is refused outright, so index 0 has to be started
`--read-only` too before any further index can attach. `launch_emulator` checks
this itself for any index above 0 and raises before touching the emulator at
all, because the emulator's own refusal of the second instance says nothing
about the first one being the cause.

UNVERIFIED ON THIS HOST: an attempt to add a second instance failed here
because index 0 was running writable, which is the constraint above; whether
several `-read-only` instances can then share the AVD is still unverified and
is checked by the next device stage, not by this file.

Nothing here ever taps, and nothing here reads a pixel. Readiness — "the game has
finished starting up, so it is safe to cut the network and to snapshot" — is the
bridge's own reading: it reports `main_unavailable` while the game is still
starting and `no_initialized_run` once it is up and idle at home. Screenshot
classification was the previous oracle, and it is renderer-dependent and
intermittently wrong; `visual_profile` stays for the review path, where a human
watches a checkpoint play and the picture is the point.

Readiness therefore needs the bridge deployed first, and `instrumented_bridge.sh
deploy` refuses to run against an online instance, so the cold path's order is
`start` (the
instance up and offline, the game not launched), then `deploy`, then `launch` —
the one short online window, which also covers deploy's own cold launch.

    uv run python scripts/clone_session.py start
    ./scripts/instrumented_bridge.sh deploy emulator-5556
    uv run python scripts/clone_session.py launch
    uv run python scripts/clone_session.py snapshot
    uv run python scripts/clone_session.py restore

Adding a second instance to share the AVD needs index 0 read-only too, and
`--index` precedes the subcommand it applies to:

    uv run python scripts/clone_session.py start --read-only
    uv run python scripts/clone_session.py --index 1 up --read-only
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_episodes import compatibility  # noqa: E402

from tower_rl.doctor import find_android_tool  # noqa: E402
from tower_rl.infrastructure.instrumented_bridge import (  # noqa: E402
    BridgeCompatibility,
    BridgeObservation,
    BridgeRunUnavailable,
    InstrumentedBridgeClient,
    InstrumentedBridgeError,
)

PACKAGE = "com.TechTreeGames.TheTower"
#: The disposable rooted clone. The canonical evaluation AVD is never touched.
CLONE_AVD = "tower_rl_instrumented_api36"
#: The canonical evaluation AVD: never launched by automation, at any index.
CANONICAL_AVD = "tower_rl_api36_play_x86_64"
#: Emulator console ports are even and two apart, and the adb serial is the
#: console port. Index 0 is the port every existing invocation already uses.
FIRST_CONSOLE_PORT = 5556
#: The bridge's fixed device port, mirrored from `instrumented_bridge.sh`. Every
#: instance listens on the same device port, so each forwards its own host port.
BRIDGE_DEVICE_PORT = 47651
FIRST_BRIDGE_HOST_PORT = 47652
#: The bridge reports this reason once the game is up and holding no run, which
#: is the home screen; while the game is still starting — the splash, or the
#: Firebase OFFLINE modal — it reports `main_unavailable` instead. See
#: `native/tower_bridge/README.md`.
GAME_IS_IDLE = "no_initialized_run"
#: Snapshots are named for the bridge build inside them: a snapshot carries the
#: bridge that was deployed when it was taken, and a bridge that is not the one
#: the client expects fails the handshake. The key lives in the name so the
#: emulator's own snapshot directory is the registry, with no sidecar file to
#: fall out of step with it, and stale keys are visible to the operator by name.
SNAPSHOT_PREFIX = "tower_clone_home_offline"
#: A restored instance has an already-started game, so readiness is a short
#: confirmation rather than the minutes a cold launch takes.
RESTORED_READY_TIMEOUT = 60.0
#: The only renderer this emulator will save or restore a snapshot under. The
#: game uses Vulkan, and `-gpu host` refuses to snapshot a Vulkan app
#: (`KO: ... UNSUPPORTED_VK_APP`); a snapshot also carries renderer state, so
#: restoring one under a different renderer would not be the state it claims
#: anyway. Bring-up under any other renderer takes the cold path outright.
SNAPSHOT_CAPABLE_RENDERER = "lavapipe"
#: The rate the guest paces the game at, in Hz. The game's frame pacing is a
#: guest-side vsync timer, so a decision's frames arrive at whatever rate the
#: guest display runs, and raising it is what makes an advance cheaper in wall
#: time (60 Hz -> 16.17 ms a frame, 120 Hz -> 8.3 ms). Two settings have to agree
#: or the guest keeps 60: `-vsync-rate` below sets the display's physical vsync
#: mode, and `raise_frame_rate` lifts SurfaceFlinger's per-uid game
#: frame-rate override (`ro.surface_flinger.game_default_frame_rate_override=60`)
#: that otherwise pins the game surface to 60 whatever mode the display is in.
#: They are one constant here precisely because they cannot be allowed to drift
#: apart: a 120 override against a 60 Hz mode still renders at 60.
#:
#: A third layer exists and is deliberately not a lever: the bridge sets Unity's
#: own `targetFrameRate` to `kUncappedFrameRate = 240`
#: (`native/tower_bridge/tower_bridge.cpp`), which `M1B-E041` found inert under
#: capture mode. The pair above is therefore the complete set only because that
#: third setting is non-binding.
GUEST_FRAME_RATE_HZ = 120
#: Measured ceiling, not a preference: `M1B-E042` found the solo knee at 300 Hz,
#: and 360 Hz a cliff where the guest reports the rate as adopted while
#: delivering 91 fps. The confirmation below reads the guest's own claim, which
#: at that point is false, so the bound is what keeps this constant from being
#: tuned past what measured fps supports.
assert GUEST_FRAME_RATE_HZ <= 300, "no measured fps supports a guest rate above 300 Hz"
#: How long SurfaceFlinger is given to apply a raised rate before the instance is
#: called unusable. Observed on device to take a beat, not to be slow.
FRAME_RATE_CONFIRM_TIMEOUT = 20.0
#: SurfaceFlinger reports the applied rate as a float it computed from a vsync
#: period, so an instance genuinely at the rate can read `120.000004`. The
#: readings are compared within this, because the failure worth refusing is a
#: surface still at 60, not a rounding difference.
FRAME_RATE_TOLERANCE_HZ = 0.5
#: The game's Unity activity, as `dumpsys activity activities` names it. Its
#: absence from the *resumed* activity is what separates a game that is merely
#: slow to start from one the guest's Google Play has killed: the process comes
#: back for a job service, so `pidof` answers, but nothing is on screen and the
#: in-process bridge is frozen.
GAME_ACTIVITY = "UnityPlayerActivity"
#: How many times one bring-up re-issues the launcher intent after that kill.
#: Two, because the hazard is a batch of Play installs passing through, not a
#: standing condition: if two relaunches do not outlast it, something else is
#: wrong and the readiness timeout should report it rather than loop forever.
MAX_RELAUNCHES = 2
#: How long a relaunch made outside the online window is given to reach home
#: again. Shorter than the cold launch's own 300s because the game has already
#: been past its Firebase check once this boot; long enough that a relaunch
#: competing with the rest of Play's install batch is not cut off mid-start.
RELAUNCH_READY_TIMEOUT = 180.0
#: How often readiness is re-read. The online window is held open until the
#: bridge calls the game ready, so the poll interval is time the instance spends
#: online for no reason; it is short there and stays cheap everywhere else.
POLL_SECONDS = 5.0
ONLINE_POLL_SECONDS = 1.0
#: Where the emulator's own output is captured. A module-level path rather than
#: a `tempfile.gettempdir()` call inside `launch_emulator`, so a test can point
#: it somewhere harmless: the log is opened `"wb"` before the launch, and a test
#: that fakes only `Popen` truncated the live instance's log every run.
EMULATOR_LOG_DIRECTORY = Path(tempfile.gettempdir())


class CloneError(RuntimeError):
    """The clone is not in the state this step requires."""


@dataclass(frozen=True)
class CloneInstance:
    """One emulator instance of the clone, identified by index.

    Instance identity is one concept — AVD, console port, adb serial — so it is
    derived in one place rather than restated by every caller.
    """

    index: int = 0
    avd: str = CLONE_AVD

    def __post_init__(self) -> None:
        if self.index < 0:
            raise CloneError(f"instance index must not be negative: {self.index}")
        if self.avd == CANONICAL_AVD:
            raise CloneError(f"refusing to touch the canonical evaluation AVD {CANONICAL_AVD}")

    @property
    def console_port(self) -> int:
        return FIRST_CONSOLE_PORT + 2 * self.index

    @property
    def serial(self) -> str:
        return f"emulator-{self.console_port}"

    @property
    def bridge_host_port(self) -> int:
        """The host side of this instance's forward to the bridge's device port."""
        return FIRST_BRIDGE_HOST_PORT + self.index


def adb(instance: CloneInstance, *args: str, timeout: float = 30.0) -> str:
    binary = find_android_tool("adb")
    if binary is None:
        raise CloneError("adb not found")
    result = subprocess.run(
        [str(binary), "-s", instance.serial, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def routable_interfaces(instance: CloneInstance) -> list[str]:
    """Interfaces other than loopback holding an IPv4 address.

    This is the offline check. `airplane_mode_on` is not: it reads 1 while the
    wifi radio is still up with a route, which is how every run before
    `M1B-E010` executed online while reporting itself offline.
    """
    lines = adb(instance, "shell", "ip", "-o", "-4", "addr", "show").splitlines()
    return [line.strip() for line in lines if line.strip() and " lo " not in line]


def require_offline(instance: CloneInstance) -> None:
    routable = routable_interfaces(instance)
    if routable:
        raise CloneError(f"{instance.serial} is online: {'; '.join(routable)}")


def set_radios(instance: CloneInstance, enabled: bool, *, settle: float = 25.0) -> None:
    """Turn the radios on or off and wait for the interface to follow."""
    state = "enable" if enabled else "disable"
    adb(instance, "shell", "svc", "wifi", state)
    adb(instance, "shell", "svc", "data", state)
    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        if bool(routable_interfaces(instance)) is enabled:
            return
        time.sleep(2.0)
    raise CloneError(f"radios did not turn {state} within {settle:.0f}s")


def game_pid(instance: CloneInstance) -> str:
    return adb(instance, "shell", "pidof", PACKAGE)


def game_activity_present(instance: CloneInstance) -> bool:
    """Whether the game holds the resumed activity, rather than a stale record.

    Not a substring of the whole dump: `dumpsys activity activities` keeps the
    killed `ActivityRecord` in its task history, so the game's component name is
    still in the dump long after the activity is gone. That is why the relaunch
    added for the Play-update kill never once fired across three 7-actor runs —
    on `emulator-5558` the name was in the dump at a moment `pidof` answered
    nothing at all. The resumed line is the reading that separates a game that is
    on screen from one whose process Play took away and gave back as a job
    service.
    """
    dump = adb(instance, "shell", "dumpsys", "activity", "activities")
    return any(
        "ResumedActivity" in line and PACKAGE in line and GAME_ACTIVITY in line
        for line in dump.splitlines()
    )


def launch_game(instance: CloneInstance) -> None:
    """Send the game's launcher intent. `monkey` sends the intent; it taps nothing."""
    adb(instance, "shell", "monkey", "-p", PACKAGE, "-c", "android.intent.category.LAUNCHER", "1")


def bridge_build_directory() -> Path:
    """Where the private bridge was built; never committed, never guessed."""
    configured = os.environ.get("TOWER_BRIDGE_BUILD_DIR")
    if configured:
        return Path(configured)
    live = Path("/tmp/tower-bridge-live.latest")
    if live.is_file():
        return Path(live.read_text().strip())
    raise CloneError("the private bridge build directory is unknown: set TOWER_BRIDGE_BUILD_DIR")


def expected_compatibility() -> BridgeCompatibility:
    """The private build's identity, which every bridge handshake is checked against."""
    try:
        return compatibility(bridge_build_directory())
    except (OSError, KeyError, ValueError) as error:
        raise CloneError(f"cannot read the private bridge build identity: {error}") from error


def bridge_key() -> str:
    """A stable identity for the bridge that is about to be deployed.

    The artifact itself is hashed rather than the version string in the build
    cache: the bridge changes on most commits without that string moving, and it
    is the binary that either answers the current client or does not.
    """
    binary = bridge_build_directory() / "libtower_bridge.so"
    try:
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    except OSError as error:
        raise CloneError(f"cannot read the bridge to key a snapshot: {error}") from error
    return digest[:12]


def keyed_snapshot_name(key: str) -> str:
    """The snapshot that holds this exact bridge, already started and offline."""
    return f"{SNAPSHOT_PREFIX}_{key}"


def snapshot_directory(instance: CloneInstance) -> Path:
    """Where this AVD keeps its snapshots: machine-local, always outside the repository."""
    home = os.environ.get("ANDROID_AVD_HOME") or str(Path.home() / ".android" / "avd")
    return Path(home) / f"{instance.avd}.avd" / "snapshots"


def snapshot_exists(instance: CloneInstance, name: str) -> bool:
    """Whether the AVD already holds that snapshot, with nothing running.

    The choice between restoring and cold-starting has to be made before any
    emulator is launched, so it is read from the AVD on disk rather than from
    `adb emu avd snapshot list`, which needs the very instance it would decide.
    """
    return (snapshot_directory(instance) / name).is_dir()


def read_bridge_state(instance: CloneInstance) -> BridgeObservation | BridgeRunUnavailable:
    """Ask the deployed bridge what the game is doing, over this instance's port.

    The forward is re-established here rather than assumed: it lives on the host
    side of adb and does not survive the emulator it pointed at, so a restored
    snapshot has none even though the bridge inside it is listening.
    """
    adb(instance, "forward", f"tcp:{instance.bridge_host_port}", f"tcp:{BRIDGE_DEVICE_PORT}")
    client = InstrumentedBridgeClient(
        "127.0.0.1",
        instance.bridge_host_port,
        expected_compatibility=expected_compatibility(),
        connect_timeout=5.0,
        read_timeout=30.0,
        heartbeat_timeout=30.0,
    )
    try:
        client.connect()
        return client.read_state()
    except InstrumentedBridgeError as error:
        raise CloneError(f"the bridge is not answering on {instance.serial}: {error}") from error
    finally:
        client.close()


def why_not_ready(instance: CloneInstance) -> str | None:
    """Why the game cannot be driven yet, or None once it can.

    Ready is not a picture: the process is alive, the bridge completes its
    handshake, and the state it reports is a coherent idle one. The bridge
    answers from the splash and from the OFFLINE modal too, so a handshake alone
    proves nothing about startup — what separates them is `main_unavailable`,
    which is the bridge saying `Main` does not exist yet.
    """
    if not game_pid(instance):
        return "the game is not running"
    try:
        state = read_bridge_state(instance)
    except CloneError as error:
        return str(error)
    if isinstance(state, BridgeRunUnavailable):
        if state.reason == GAME_IS_IDLE:
            return None
        return f"the game is still starting: {state.reason}"
    if state.round_active:
        return f"a round is already running at wave {state.wave}"
    return None


def wait_until_ready(
    instance: CloneInstance,
    *,
    timeout: float,
    poll: float = POLL_SECONDS,
    relaunch: Callable[[], None] | None = None,
    max_relaunches: int = MAX_RELAUNCHES,
    point: str = "while starting",
) -> None:
    """Wait for the bridge to call the game ready, relaunching a game Play killed.

    The guest's own Google Play installs app updates during the one online
    window, and installing WebView force-stops every process holding it, the game
    included (`Killing ... (adj 0): stop com.google.android.webview due to
    installPackageLI`, then `Force removing ActivityRecord{...
    UnityPlayerActivity}: app died`). Android restarts the game moments later for
    a job service only, so it has a pid and no activity, is frozen in the
    background, and the in-process bridge never answers again. Waiting that out
    costs the whole timeout, which is how a cold `-gpu host` bring-up lost an
    instance to a 300s `main_unavailable`.

    So the loss of the activity is detected and the launcher intent re-issued.
    Only a *lost* activity counts: the game is watched until it has an activity
    at least once, which is also what keeps the intent that started it from
    being re-sent before the activity appears. Re-issuing an intent is not a
    network operation, so a relaunch after the radios are down stays offline;
    nothing here touches a radio.

    `point` names where in the lifecycle this wait is happening, and it is
    printed with every relaunch: Play chooses when it installs, so which of the
    points that watch for the kill actually fires is the evidence a run leaves
    behind about where the kill landed this time.
    """
    deadline = time.monotonic() + timeout
    last = "nothing observed yet"
    had_activity = False
    relaunches = 0
    while time.monotonic() < deadline:
        reason = why_not_ready(instance)
        if reason is None:
            return
        if reason != last:
            print(f"  {instance.serial}: {reason}", flush=True)
            last = reason
        if relaunch is not None:
            if game_activity_present(instance):
                had_activity = True
            elif had_activity and relaunches < max_relaunches:
                relaunches += 1
                print(
                    f"  {instance.serial}: the game lost its activity {point}; "
                    f"re-issuing the launcher intent ({relaunches}/{max_relaunches})",
                    flush=True,
                )
                relaunch()
                # The next relaunch waits for the game to come back and be
                # killed again, rather than firing on the same absence.
                had_activity = False
                last = "relaunched, waiting for the game again"
        time.sleep(poll)
    raise CloneError(f"{instance.serial} never became ready: {last}")


def relaunch_if_activity_lost(
    instance: CloneInstance, *, point: str, timeout: float = RELAUNCH_READY_TIMEOUT
) -> bool:
    """Put the game back if Play has taken its activity away, and say whether it had.

    The kill is not confined to the readiness wait. Play downloads its WebView
    update during the one online window and installs it later, offline, at a
    time of its own choosing, so an instance that has already reached home and
    been cut off the network can still lose its activity — which is what two
    7-actor fleet runs recorded, at the check after the network is cut and again
    in the gap before the frame rate is raised. Both left a game with a pid, a
    job service and nothing on screen: no activity, no surface, and an
    in-process bridge that never answers.

    So the same recovery the readiness wait makes is available wherever bring-up
    asserts the game is there. Only the activity's absence triggers it: a game
    that is present but reports some other reason is a different fault and must
    be reported rather than restarted. Re-issuing the launcher intent is not a
    network operation, so this stays offline; nothing here touches a radio.
    """
    if game_activity_present(instance):
        return False
    print(
        f"  {instance.serial}: the game has no activity {point}; "
        f"re-issuing the launcher intent (offline)",
        flush=True,
    )
    launch_game(instance)
    wait_until_ready(
        instance,
        timeout=timeout,
        relaunch=lambda: launch_game(instance),
        point=point,
    )
    return True


def _captured_output(log_path: Path | None, *, limit: int = 4000) -> str:
    """The emulator's own recent output, for an error message to quote.

    Read defensively: the file may not exist yet, or may be gone, or (under
    test) may never have been written at all.
    """
    if log_path is None:
        return ""
    try:
        text = log_path.read_text(errors="replace").strip()
    except OSError:
        return ""
    return text[-limit:] if text else ""


def wait_for_boot(
    instance: CloneInstance,
    *,
    timeout: float,
    process: subprocess.Popen[bytes] | None = None,
    log_path: Path | None = None,
) -> None:
    """Poll for boot, but fail the moment the emulator process itself is gone.

    Waiting out the full timeout against a process that already exited is how
    a refusal like `Another emulator instance is running` looked like a hang:
    nothing distinguished "still booting" from "already dead" until now.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            output = _captured_output(log_path)
            raise CloneError(
                f"{instance.serial}: the emulator exited before boot completed "
                f"(code {process.returncode}): {output or 'no output captured'}"
            )
        if adb(instance, "shell", "getprop", "sys.boot_completed", timeout=10.0) == "1":
            return
        time.sleep(5.0)
    output = _captured_output(log_path)
    raise CloneError(
        f"{instance.serial} did not finish booting within {timeout:.0f}s: "
        f"{output or 'no output captured'}"
    )


def require_shareable(instance: CloneInstance, read_only: bool) -> None:
    """Refuse, before touching the emulator, what it would refuse anyway.

    The emulator requires every instance of an AVD to be `-read-only` once more
    than one is running against it. Index 0 may still launch writable on its
    own — to save a snapshot, most often — but any further index exists only to
    share that AVD with something else, so a writable one there is always
    rejected. Checking here means the failure names the instance and the
    reason instead of the emulator's opaque refusal reaching whichever instance
    happened to start second.
    """
    if instance.index > 0 and not read_only:
        raise CloneError(
            f"{instance.serial} shares {instance.avd} with instance 0 and must be "
            "launched --read-only; only index 0 may run writable, and only alone"
        )


def emulator_command(
    instance: CloneInstance,
    *,
    binary: str,
    renderer: str,
    snapshot: str | None,
    read_only: bool,
    cores: int,
) -> list[str]:
    """The exact invocation for one instance.

    `-read-only` is what lets several instances share the one AVD: the base
    image stays untouched and each instance writes to its own overlay. It is
    also incompatible with saving a snapshot, which is why it is a choice and
    not the default.

    `-vsync-rate` is half of `GUEST_FRAME_RATE_HZ`; `raise_frame_rate` is the
    other half. Every instance this builds is `-no-window`, which is why raising
    it here is safe: the emulator warns that exceeding the host display's refresh
    rate is undefined, and a headless instance is driving no host display at all.
    The windowed review path (`launch_avd.sh`) keeps default pacing for that
    reason, and because a human watching a checkpoint play wants the game's own
    speed, not the fleet's.
    """
    command = [
        binary, f"@{instance.avd}",
        "-gpu", renderer, "-no-audio", "-no-boot-anim", "-no-window",
        "-cores", str(cores), "-port", str(instance.console_port), "-no-snapshot-save",
        "-vsync-rate", str(GUEST_FRAME_RATE_HZ),
    ]
    if read_only:
        command.append("-read-only")
    command += ["-snapshot", snapshot] if snapshot else ["-no-snapshot-load"]
    return command


def launch_emulator(
    instance: CloneInstance,
    renderer: str,
    snapshot: str | None,
    *,
    read_only: bool = False,
    cores: int = 8,
) -> None:
    require_shareable(instance, read_only)
    binary = find_android_tool("emulator")
    if binary is None:
        raise CloneError("emulator not found")
    command = emulator_command(
        instance,
        binary=str(binary),
        renderer=renderer,
        snapshot=snapshot,
        read_only=read_only,
        cores=cores,
    )
    detail = [renderer]
    if snapshot:
        detail.append(f"snapshot {snapshot}")
    if read_only:
        detail.append("read-only")
    print(
        f"launching {instance.avd} on {instance.serial} ({', '.join(detail)})",
        flush=True,
    )
    # Captured rather than discarded: a launch that never boots is otherwise
    # indistinguishable from one that is merely slow, and the emulator's own
    # reply is the only account of why (`ERROR | Another emulator instance is
    # running ...`, for example). Kept on success too, overwritten by the next
    # launch, so a failure can always be explained without spamming this
    # process's own console when there is nothing to explain.
    log_path = EMULATOR_LOG_DIRECTORY / f"tower-rl-emulator-{instance.serial}.log"
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
    # Left at 300s rather than raised: the one timeout a fleet has hit on device
    # was caused by four simultaneous cold boots contending for the host (host
    # load 10.71, total CPU 1,835%), not by a boot that is slow on its own — the
    # other three reached home alone in 60-90s. `run_actors.stagger_bring_up`
    # removes that contention by sequencing bring-ups, so 300s stays a bound on
    # a single uncontended boot, where a slow boot is a genuine signal worth
    # surfacing rather than a limit to raise away.
    wait_for_boot(instance, timeout=300.0, process=process, log_path=log_path)


def kill_emulator(instance: CloneInstance) -> None:
    """Never leave an instance running when a run ends."""
    adb(instance, "emu", "kill", timeout=60.0)


def start(
    instance: CloneInstance, renderer: str, *, read_only: bool = False, cores: int = 8
) -> None:
    """Cold start: the instance up and offline, with the game not yet launched.

    The game is launched afterwards by `launch_game_at_home`, because readiness is
    now the bridge's own reading and the bridge has to be deployed first — and
    `instrumented_bridge.sh deploy` refuses to run against an online instance.
    """
    launch_emulator(instance, renderer, snapshot=None, read_only=read_only, cores=cores)
    set_radios(instance, False)
    require_offline(instance)
    print(f"{instance.serial} is up and offline; the game is not launched yet", flush=True)


def launch_game_at_home(instance: CloneInstance) -> None:
    """Cold-launch the game past its network check and leave it idle at home, offline.

    The game blocks on a Firebase check and an OFFLINE modal, so a cold launch
    needs a network however briefly. Anything that cold-launches the game on an
    offline clone — `instrumented_bridge.sh deploy` above all — must come back
    through here or it stays on that modal.

    Home is established without reading a pixel, by asking the bridge, so this
    requires the bridge already deployed and its overlay mounted: it runs after
    `deploy`, never before it. The radios come down only once the bridge says the
    game is up, and the result is verified by interface.

    That reading is the whole length of the window, so it is taken every
    `ONLINE_POLL_SECONDS` rather than on the ordinary poll: the game is ready
    seconds after it is launched, and every second between being ready and being
    read is a second the guest's Google Play spends installing updates over a
    running game. `wait_until_ready` relaunches the game if Play kills it anyway,
    and so does the check after the cut: Play installs what it downloaded at a
    time of its choosing, which on device has been after the network was already
    gone.
    """
    print("enabling radios for the startup check only", flush=True)
    set_radios(instance, True)
    adb(instance, "shell", "am", "force-stop", PACKAGE)
    launch_game(instance)
    wait_until_ready(
        instance,
        timeout=300.0,
        poll=ONLINE_POLL_SECONDS,
        relaunch=lambda: launch_game(instance),
    )
    print("the game is up and idle; cutting the network", flush=True)
    set_radios(instance, False)
    require_offline(instance)
    reason = why_not_ready(instance)
    if reason is not None and relaunch_if_activity_lost(instance, point="as the network was cut"):
        reason = why_not_ready(instance)
    if reason is not None:
        raise CloneError(f"the game did not survive the network being cut: {reason}")
    print(f"{instance.serial} is at home and offline", flush=True)


def game_uid(instance: CloneInstance) -> str:
    """The game's Android uid, which is how SurfaceFlinger names its override."""
    for line in adb(instance, "shell", "dumpsys", "package", PACKAGE).splitlines():
        match = re.search(r"\bappId=(\d+)", line)
        if match:
            return match.group(1)
    raise CloneError(f"{instance.serial}: {PACKAGE} has no uid; is it installed?")


def missing_surface_cause(applied_reading: str) -> str:
    """Name the likely cause when SurfaceFlinger publishes no applied rate at all.

    An absent per-uid applied rate is not the same failure as a rate that reads
    60: SurfaceFlinger publishes an entry per *surface*, so a game whose activity
    Play has killed leaves no entry to read rather than a wrong one. Two fleet
    runs lost an actor to a bare `applied frame rate absent` and the cause had to
    be reconstructed from logcat afterwards, so the message says it.
    """
    if applied_reading != "absent":
        return ""
    return (
        "; SurfaceFlinger publishes no applied rate for a uid with no surface, so "
        "the likely cause is that the game has lost its activity — see whether the "
        "guest's Play installed an update over it"
    )


def confirm_frame_rate(instance: CloneInstance) -> None:
    """Fail unless the display mode and the game's own override are both at the rate.

    The two levers fail silently and independently: a 60 Hz display mode renders
    at 60 whatever the override says, and a display at 240 Hz still renders the
    game at 60 while the per-uid game default override stands. Neither `emulator
    -vsync-rate` nor `cmd game set` reports back, so a run that quietly collected
    at 60 would look exactly like one that collected at 240 except in its
    throughput. SurfaceFlinger holds both readings, so they are read back here
    before anything is measured.
    """
    uid = game_uid(instance)
    applied_name = f"uid {uid} applied frame rate"
    rate = GUEST_FRAME_RATE_HZ
    # SurfaceFlinger applies a new override a beat after GameManagerService takes
    # it, so a reading taken the instant `cmd game set` returns still says 60.
    deadline = time.monotonic() + FRAME_RATE_CONFIRM_TIMEOUT
    while True:
        dump = adb(instance, "shell", "dumpsys", "SurfaceFlinger")
        mode = re.search(r"activeMode=\{[^}]*vsyncRate=([\d.]+) Hz", dump)
        override = re.search(rf"\{{{uid}, (\d+) \d+\}}", dump)
        applied = re.search(rf"\{{{uid}, ([\d.]+) Hz\}}", dump)
        readings = {
            "display vsync mode": mode.group(1) if mode else "absent",
            f"uid {uid} game mode override": override.group(1) if override else "absent",
            applied_name: applied.group(1) if applied else "absent",
        }
        wrong = [
            f"{name} {value}"
            for name, value in readings.items()
            if value == "absent" or abs(float(value) - float(rate)) > FRAME_RATE_TOLERANCE_HZ
        ]
        if not wrong:
            break
        if time.monotonic() >= deadline:
            raise CloneError(
                f"{instance.serial}: the guest is not at {rate} Hz: "
                + ", ".join(wrong)
                + missing_surface_cause(readings[applied_name])
            )
        time.sleep(1.0)
    print(f"{instance.serial}: confirmed at {rate} Hz: " + ", ".join(
        f"{name} {value}" for name, value in readings.items()
    ), flush=True)


def raise_frame_rate(instance: CloneInstance) -> None:
    """Lift the per-uid game frame-rate override, so the game may use the display.

    SurfaceFlinger ships a game default frame-rate override of 60 Hz that applies
    to the game's uid whatever mode the display is in, so `-vsync-rate` on its own
    leaves the game surface rendering at 60. `cmd game set --fps` replaces that
    override with ours. It is a device-wide setting on the running instance, so
    `instrumented_bridge.sh cleanup` resets it during teardown and no instance is
    left modified.

    This is not part of bring-up: every instance boots at the stock 60 Hz and is
    raised once its own bring-up has concluded, immediately before its first
    episode (`run_actors.collect_episodes`). It is not deferred any further than
    that. It once waited for the *whole fleet* to be up, on the reading that a
    raised peer killed a booting one; `M1B-E043` refuted that, and the wait was
    not harmless — it parked each ready instance idle for as long as the rest of
    the fleet took to boot, which is exactly the window the guest's Play uses to
    install what it downloaded, over a game nothing was watching.
    """
    adb(instance, "shell", "cmd", "game", "set", "--fps", str(GUEST_FRAME_RATE_HZ), PACKAGE)
    confirm_frame_rate(instance)


def restore(
    instance: CloneInstance,
    snapshot: str,
    *,
    renderer: str = "lavapipe",
    read_only: bool = False,
    cores: int = 8,
) -> None:
    """The point of the snapshot: never connect at all.

    The renderer is the one the snapshot was taken with: a snapshot holds
    renderer state, so restoring it under a different one is not the same state.
    """
    launch_emulator(instance, renderer, snapshot=snapshot, read_only=read_only, cores=cores)
    require_offline(instance)
    # The claim a snapshot makes is that the game is already started, so that is
    # what is checked. The bridge is deliberately not required here: a snapshot
    # may have been taken before any bridge was deployed into it, and the fleet
    # deploys and asserts bridge readiness immediately afterwards anyway.
    if not game_pid(instance):
        raise CloneError(f"restored snapshot has no running game: {snapshot}")
    print(
        f"restored {snapshot} on {instance.serial}: game running, offline, never connected",
        flush=True,
    )


def save_snapshot(
    instance: CloneInstance, name: str, *, renderer: str = SNAPSHOT_CAPABLE_RENDERER
) -> None:
    """Refuse to capture a state that is not the one worth restoring.

    The emulator's own reply is the only evidence of success: under `-gpu host`
    it refuses to snapshot a Vulkan app and answers `KO: ... UNSUPPORTED_VK_APP`
    while still exiting the command normally, so a reply has to be parsed rather
    than assumed. `OK` is required, and — cheap to check, since the directory is
    already how `snapshot_exists` decides whether a bring-up can restore — the
    snapshot directory must now exist too.
    """
    require_offline(instance)
    reason = why_not_ready(instance)
    if reason is not None:
        raise CloneError(f"refusing to snapshot: {reason}")
    reply = adb(instance, "emu", "avd", "snapshot", "save", name, timeout=300.0)
    if not reply.startswith("OK"):
        raise CloneError(
            f"{instance.serial} refused to save snapshot {name} under renderer "
            f"'{renderer}': {reply or 'no reply from the emulator'}"
        )
    if not snapshot_exists(instance, name):
        raise CloneError(
            f"{instance.serial} reported {reply!r} saving {name} but the snapshot "
            "directory does not exist"
        )
    print(reply, flush=True)
    print(f"snapshot {name}: game running, idle at home, offline", flush=True)


def discard_instance(instance: CloneInstance) -> None:
    """Stop an instance that cannot be used, so the next path starts from nothing.

    A restore that does not verify must leave no emulator behind: the cold path
    relaunches on the same console port, and a failure to stop is reported rather
    than hidden, because a stray instance is a device-safety problem by itself.
    """
    try:
        kill_emulator(instance)
    except Exception as error:
        print(f"warning: could not stop {instance.serial}: {error}", file=sys.stderr, flush=True)
        return
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not adb(instance, "get-state", timeout=10.0):
            return
        time.sleep(2.0)


def deploy_bridge(instance: CloneInstance) -> None:
    """Install the current bridge, through the one script that owns device safety."""
    script = Path(__file__).resolve().parent / "instrumented_bridge.sh"
    result = subprocess.run(
        [str(script), "deploy", instance.serial, str(instance.bridge_host_port)], text=True
    )
    if result.returncode != 0:
        raise CloneError(f"instrumented_bridge.sh deploy failed on {instance.serial}")


def cold_bring_up(
    instance: CloneInstance,
    renderer: str,
    *,
    deploy: Callable[[CloneInstance], None],
    read_only: bool = False,
    cores: int = 8,
) -> None:
    """The full path, and the only one that opens a network window.

    The window belongs to the game and to nothing else: the instance boots
    offline and `deploy` runs offline too (`instrumented_bridge.sh deploy`
    refuses an online instance, and whatever it leaves on the OFFLINE modal is
    force-stopped and launched again here). So the radios are up only from the
    game's launch to the bridge calling it ready, which is where `M1B-E010` says
    the network is genuinely needed — the Firebase check and the OFFLINE modal.
    """
    start(instance, renderer, read_only=read_only, cores=cores)
    require_offline(instance)
    deploy(instance)
    launch_game_at_home(instance)
    require_offline(instance)


def bring_up(
    instance: CloneInstance,
    renderer: str,
    *,
    deploy: Callable[[CloneInstance], None],
    read_only: bool = False,
    cores: int = 8,
    force_cold: bool = False,
) -> str:
    """Bring one instance up ready and offline, and say which path it took.

    Restoring is the normal path, for three reasons. It costs about ten seconds
    rather than minutes; it needs no network window at all, since the game in the
    snapshot has already been past its Firebase check; and every restore starts
    from byte-identical account state, so two arms of a comparison are not
    confounded by the progression the previous hours of play left behind
    (`M1B-E018`).

    It is only usable when the snapshot carries the bridge we are about to speak
    to, which is what the key in its name records. Anything else — no snapshot,
    a snapshot for another bridge, a restore that does not verify — falls through
    to the cold path, which pays for itself by saving the snapshot the next
    bring-up restores.
    """
    name = keyed_snapshot_name(bridge_key())
    if renderer != SNAPSHOT_CAPABLE_RENDERER:
        # `-gpu host` cannot save or usefully restore a snapshot (the emulator
        # refuses a Vulkan app's snapshot save outright), so there is nothing to
        # try there: attempting a restore or a save would only fail after paying
        # for the attempt. This is why host bring-up costs 45-60s instead of the
        # ~10s a restore takes.
        print(
            f"{instance.serial}: renderer '{renderer}' cannot snapshot a Vulkan app "
            f"(snapshots are {SNAPSHOT_CAPABLE_RENDERER}-only); taking the cold path",
            flush=True,
        )
        cold_bring_up(instance, renderer, deploy=deploy, read_only=read_only, cores=cores)
        return "cold"
    if not force_cold and snapshot_exists(instance, name):
        try:
            restore(instance, name, renderer=renderer, read_only=read_only, cores=cores)
            wait_until_ready(instance, timeout=RESTORED_READY_TIMEOUT)
            require_offline(instance)
            print(f"{instance.serial}: restored {name}, ready, never connected", flush=True)
            return "restored"
        except CloneError as error:
            print(f"{instance.serial}: {name} did not verify ({error}); cold path", flush=True)
            discard_instance(instance)
    cold_bring_up(instance, renderer, deploy=deploy, read_only=read_only, cores=cores)
    if read_only:
        # A `-read-only` instance writes to a throwaway overlay and cannot save a
        # snapshot; the pinned one is prepared on a writable instance instead.
        print(f"{instance.serial}: read-only, so {name} was not saved", flush=True)
    else:
        save_snapshot(instance, name, renderer=renderer)
    return "cold"


def report(instance: CloneInstance) -> None:
    routable = routable_interfaces(instance)
    reason = why_not_ready(instance)
    print(f"instance:  {instance.serial} ({instance.avd})")
    print(f"game pid:  {game_pid(instance) or 'not running'}")
    print(f"ready:     {reason or 'yes: up and idle, the bridge can drive it'}")
    print(f"network:   {'; '.join(routable) if routable else 'offline'}")
    try:
        name = keyed_snapshot_name(bridge_key())
    except CloneError as error:
        print(f"snapshot:  unknown: {error}")
        return
    held = "saved" if snapshot_exists(instance, name) else "not saved: the next bring-up is cold"
    print(f"snapshot:  {name} ({held})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="instance index; 0 is emulator-5556")
    parser.add_argument("--avd", default=CLONE_AVD, help="clone AVD; the canonical AVD is refused")
    sub = parser.add_subparsers(dest="command", required=True)
    started = sub.add_parser("start", help="cold start: the instance up and offline")
    started.add_argument("--renderer", default="lavapipe")
    restored = sub.add_parser("restore", help="launch from a snapshot without connecting")
    restored.add_argument("name", nargs="?", help="default: the snapshot for the current bridge")
    restored.add_argument("--renderer", default="lavapipe")
    up = sub.add_parser(
        "up", help="restore the snapshot for the current bridge, or cold-start and save one"
    )
    up.add_argument("--renderer", default="lavapipe")
    up.add_argument("--cold", action="store_true", help="skip any snapshot and take the cold path")
    for launching in (started, restored, up):
        launching.add_argument(
            "--read-only",
            action="store_true",
            help="share the AVD with other instances; cannot save a snapshot",
        )
        launching.add_argument("--cores", type=int, default=8)
    sub.add_parser(
        "launch", help="launch the game online, wait for the bridge, then cut the radios"
    )
    sub.add_parser("verify", help="report game process, readiness and network state")
    saved = sub.add_parser("snapshot", help="save a snapshot of the running, offline game")
    saved.add_argument("name", nargs="?", help="default: the snapshot for the current bridge")
    arguments = parser.parse_args()

    try:
        instance = CloneInstance(index=arguments.index, avd=arguments.avd)
        if arguments.command == "start":
            start(
                instance,
                arguments.renderer,
                read_only=arguments.read_only,
                cores=arguments.cores,
            )
        elif arguments.command == "launch":
            launch_game_at_home(instance)
        elif arguments.command == "up":
            bring_up(
                instance,
                arguments.renderer,
                deploy=deploy_bridge,
                read_only=arguments.read_only,
                cores=arguments.cores,
                force_cold=arguments.cold,
            )
            # A single instance is a fleet of one: it is up, so it may be
            # raised — but only a `--read-only` instance, which writes to a
            # throwaway overlay. `cmd game set` is GameManagerService state that
            # outlives the process, so raising a writable instance would leave
            # the clone AVD modified and a later `snapshot` would bake the
            # override into the state every future run restores.
            if arguments.read_only:
                raise_frame_rate(instance)
            else:
                print(
                    f"{instance.serial}: writable, so the guest stays at the stock rate; "
                    "raise it on a --read-only instance",
                    flush=True,
                )
        elif arguments.command == "snapshot":
            save_snapshot(instance, arguments.name or keyed_snapshot_name(bridge_key()))
        elif arguments.command == "restore":
            restore(
                instance,
                arguments.name or keyed_snapshot_name(bridge_key()),
                renderer=arguments.renderer,
                read_only=arguments.read_only,
                cores=arguments.cores,
            )
        else:
            report(instance)
    except CloneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
