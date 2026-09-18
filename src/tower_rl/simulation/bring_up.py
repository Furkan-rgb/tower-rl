"""Bringing one instance up ready and offline, by restore or by cold start.

The game cannot cold-launch without a network: it stops at a Firebase
online-status check and an OFFLINE modal, and never reaches the battle home
screen (`M1B-E010`). It plays fine once the network is cut. So the only online
window is application startup, and this module exists to make that window short,
verified, and identical every time.

A snapshot removes the window entirely: an emulator snapshot taken while the
game is up and idle with the radios already down restores into an
already-started, already-offline game. A restore also skips whatever intro a
cold boot walks through, and — the reason that matters most for the benchmark —
it starts from byte-identical account state, so the progression this account
accumulates between runs cannot drift between two arms of a comparison the way
it did in the frame-size sweep (`M1B-E018`).

A snapshot carries the bridge that was deployed when it was taken, and a stale
bridge does not answer the current client, so each snapshot is named for the
bridge inside it: `tower_clone_home_offline_<key>`, where the key is a hash of
`libtower_bridge.so` in the private build directory. `bring_up` therefore
restores that snapshot when the AVD holds one for the bridge about to be
deployed, verifies it (offline by interface, game process alive, the bridge's
own readiness reading) and returns. Otherwise — no snapshot, a snapshot for
another bridge, or a restore that does not verify — it takes the cold path once,
with its one online window, and saves the snapshot the next bring-up restores.

Nothing here ever taps, and nothing here reads a pixel. Readiness — "the game
has finished starting up, so it is safe to cut the network and to snapshot" — is
the bridge's own reading: it reports `main_unavailable` while the game is still
starting and `no_initialized_run` once it is up and idle at home. Screenshot
classification was the previous oracle, and it is renderer-dependent and
intermittently wrong.

Readiness therefore needs the bridge deployed first, and `instrumented_bridge.sh
deploy` refuses to run against an online instance, so the cold path's order is
`start` (the instance up and offline, the game not launched), then `deploy`,
then `launch_game_at_home` — the one short online window, which also covers
deploy's own cold launch.

Offline is read one way only, here: `ip -o -4 addr show` with `lo` discounted.
`airplane_mode_on` is not an oracle — it reads 1 while the wifi radio is still
up with a route, which is how every run before `M1B-E010` executed online while
reporting itself offline.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from tower_rl.simulation.bridge import bridge_build_directory, compatibility
from tower_rl.simulation.instance import (
    BRIDGE_DEVICE_PORT,
    GUEST_FRAME_RATE_HZ,
    PACKAGE,
    CloneError,
    CloneInstance,
    adb,
    kill_emulator,
    launch_emulator,
)
from tower_rl.simulation.instrumented_bridge import (
    BridgeCompatibility,
    BridgeObservation,
    BridgeRunUnavailable,
    InstrumentedBridgeClient,
    InstrumentedBridgeError,
)

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
#: The game's Unity activity, as `dumpsys activity activities` names it. Its
#: absence from the *resumed* activity is what separates a game that is merely
#: slow to start from one the guest's Google Play has killed: the process comes
#: back for a job service, so `pidof` answers, but nothing is on screen and the
#: in-process bridge is frozen.
GAME_ACTIVITY = "UnityPlayerActivity"
#: How many times the one readiness wait re-issues the launcher intent after
#: that kill. Two, because the hazard is a batch of Play installs passing
#: through, not a standing condition: if two relaunches do not outlast it,
#: something else is wrong and the readiness timeout should report it rather
#: than loop forever. This is the whole budget of a bring-up: the wait inside
#: `launch_game_at_home` is the only place a relaunch happens, because it is the
#: only place the radios are still up (`M1B-E049`).
MAX_RELAUNCHES = 2
#: How often readiness is re-read. The online window is held open until the
#: bridge calls the game ready, so the poll interval is time the instance spends
#: online for no reason; it is short there and stays cheap everywhere else.
POLL_SECONDS = 5.0
ONLINE_POLL_SECONDS = 1.0


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
    Re-issuing an intent is not a network operation, so nothing here touches a
    radio. This is nevertheless the ONLY place a relaunch can work: it runs
    while the radios are still up, and `M1B-E049` measured a relaunch after the
    cut sitting at `main_unavailable` until its timeout, because a launch with
    no network stops at the Firebase check and the OFFLINE modal (`M1B-E010`).
    Everywhere else the loss of the activity is reported, not repaired.
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
                    f"  {instance.serial}: the game lost its activity while starting; "
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


def start(
    instance: CloneInstance,
    renderer: str,
    *,
    read_only: bool = False,
    cores: int = 8,
    frame_rate_hz: int = GUEST_FRAME_RATE_HZ,
    windowed: bool = False,
) -> None:
    """Cold start: the instance up and offline, with the game not yet launched.

    The game is launched afterwards by `launch_game_at_home`, because readiness is
    now the bridge's own reading and the bridge has to be deployed first — and
    `instrumented_bridge.sh deploy` refuses to run against an online instance.
    """
    launch_emulator(
        instance,
        renderer,
        snapshot=None,
        read_only=read_only,
        cores=cores,
        frame_rate_hz=frame_rate_hz,
        windowed=windowed,
    )
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
    running game. `wait_until_ready` relaunches the game if Play kills it inside
    that window, which is the only place a relaunch can reach home. The check
    after the cut has no such remedy: Play installs what it downloaded at a time
    of its choosing, and when that lands after the network is gone the instance
    is lost and says so by name.
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
    if reason is not None:
        raise CloneError(
            f"the game did not survive the network being cut: {reason}"
            + lost_activity_cause(instance)
        )
    print(f"{instance.serial} is at home and offline", flush=True)


#: What a lost activity is called wherever bring-up finds one. Named because the
#: instance cannot be recovered from it — a relaunch without a network stops at
#: the OFFLINE modal (`M1B-E049`) — so the fleet log has to carry the cause
#: rather than a bare symptom.
GAME_ACTIVITY_LOST = (
    "the game has lost its activity (the guest's Play kills it to install an update)"
)


def lost_activity_cause(instance: CloneInstance) -> str:
    """The named reason appended when the game has no activity left."""
    return f"; {GAME_ACTIVITY_LOST}" if not game_activity_present(instance) else ""


def require_game_activity(instance: CloneInstance) -> None:
    """Fail by name unless the game still holds its activity.

    The last reading before an instance is measured. There is nothing to repair
    here: `M1B-E049` forced the kill on device and found that re-issuing the
    launcher intent after the network is cut leaves the game at
    `main_unavailable` until the timeout, because a launch with no network stops
    at the Firebase check and the OFFLINE modal (`M1B-E010`). So the fleet loses
    this instance either way, and what matters is that its log says why.
    """
    if not game_activity_present(instance):
        raise CloneError(f"{instance.serial}: {GAME_ACTIVITY_LOST}")


def restore(
    instance: CloneInstance,
    snapshot: str,
    *,
    renderer: str = "lavapipe",
    read_only: bool = False,
    cores: int = 8,
    frame_rate_hz: int = GUEST_FRAME_RATE_HZ,
    windowed: bool = False,
) -> None:
    """The point of the snapshot: never connect at all.

    The renderer is the one the snapshot was taken with: a snapshot holds
    renderer state, so restoring it under a different one is not the same state.
    """
    launch_emulator(
        instance,
        renderer,
        snapshot=snapshot,
        read_only=read_only,
        cores=cores,
        frame_rate_hz=frame_rate_hz,
        windowed=windowed,
    )
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


def cold_bring_up(
    instance: CloneInstance,
    renderer: str,
    *,
    deploy: Callable[[CloneInstance], None],
    read_only: bool = False,
    cores: int = 8,
    frame_rate_hz: int = GUEST_FRAME_RATE_HZ,
    windowed: bool = False,
) -> None:
    """The full path, and the only one that opens a network window.

    The window belongs to the game and to nothing else: the instance boots
    offline and `deploy` runs offline too (`instrumented_bridge.sh deploy`
    refuses an online instance, and whatever it leaves on the OFFLINE modal is
    force-stopped and launched again here). So the radios are up only from the
    game's launch to the bridge calling it ready, which is where `M1B-E010` says
    the network is genuinely needed — the Firebase check and the OFFLINE modal.
    """
    start(
        instance,
        renderer,
        read_only=read_only,
        cores=cores,
        frame_rate_hz=frame_rate_hz,
        windowed=windowed,
    )
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
    frame_rate_hz: int = GUEST_FRAME_RATE_HZ,
    windowed: bool = False,
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
        cold_bring_up(
            instance,
            renderer,
            deploy=deploy,
            read_only=read_only,
            cores=cores,
            frame_rate_hz=frame_rate_hz,
            windowed=windowed,
        )
        return "cold"
    if not force_cold and snapshot_exists(instance, name):
        try:
            restore(
                instance,
                name,
                renderer=renderer,
                read_only=read_only,
                cores=cores,
                frame_rate_hz=frame_rate_hz,
                windowed=windowed,
            )
            wait_until_ready(instance, timeout=RESTORED_READY_TIMEOUT)
            require_offline(instance)
            print(f"{instance.serial}: restored {name}, ready, never connected", flush=True)
            return "restored"
        except CloneError as error:
            print(f"{instance.serial}: {name} did not verify ({error}); cold path", flush=True)
            discard_instance(instance)
    cold_bring_up(
        instance,
        renderer,
        deploy=deploy,
        read_only=read_only,
        cores=cores,
        frame_rate_hz=frame_rate_hz,
        windowed=windowed,
    )
    if read_only:
        # A `-read-only` instance writes to a throwaway overlay and cannot save a
        # snapshot; the pinned one is prepared on a writable instance instead.
        print(f"{instance.serial}: read-only, so {name} was not saved", flush=True)
    else:
        save_snapshot(instance, name, renderer=renderer)
    return "cold"
