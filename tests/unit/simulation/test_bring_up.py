"""Bring-up readiness, without an emulator and without a pixel.

The clone is stood in for: adb answers from a fake instance, the bridge answers
from a fake client, and the clock advances only when the code sleeps. What is
under test is the decision this script exists to make — when the game has
finished starting up, so that it is safe to cut the network and to snapshot —
and the device-safety properties that hang off it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from tower_rl.simulation import bring_up, frame_rate, instance
from tower_rl.simulation.bridge import BRIDGE_SCRIPT
from tower_rl.simulation.bring_up import (
    launch_game_at_home,
    restore,
    save_snapshot,
    start,
)
from tower_rl.simulation.instance import CloneError, CloneInstance
from tower_rl.simulation.instrumented_bridge import (
    BridgeDisconnectedError,
    BridgeObservation,
    BridgeRunUnavailable,
)

STARTING = BridgeRunUnavailable(1, "main_unavailable")
IDLE = BridgeRunUnavailable(2, "no_initialized_run")


@pytest.fixture(autouse=True)
def emulator_log_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every test's emulator log out of the directory the live ones use.

    `launch_emulator` opens its log `"wb"` before the launch, so a test that
    fakes only `Popen` still truncates a file — and the name is derived from the
    index, so tests at index 0 and 1 were truncating the logs of the instances
    actually running on 5556 and 5558, destroying the only account a real
    failure leaves behind. Autouse rather than per-test: this must not depend on
    a new test remembering it.
    """
    directory = tmp_path / "emulator-logs"
    directory.mkdir()
    monkeypatch.setattr(instance, "EMULATOR_LOG_DIRECTORY", directory)
    return directory


def running_round(wave: int = 12) -> BridgeObservation:
    return BridgeObservation(
        sequence=3,
        lifecycle="active",
        wave=wave,
        cash=10.0,
        health=100.0,
        max_health=100.0,
        terminal=False,
        round_active=True,
        game_speed=1.0,
        play_time=5.0,
        upgrades=(),
    )


class FakeClone:
    """One emulator instance's adb surface, and the radio state it reports."""

    def __init__(
        self,
        *,
        online: bool = True,
        pid: str = "4242",
        snapshot_reply: str = "OK",
        create_snapshot_dir: bool = True,
    ) -> None:
        self.online = online
        #: What `pidof` answers, one reading per call, the last repeating — the
        #: same scripting as `activities`, because a Play update takes the
        #: process away and gives it back moments later as a job service.
        self.pids = [pid]
        self.commands: list[str] = []
        #: The emulator console's reply to `snapshot save`. Real replies are
        #: `OK` on success or `KO: <reason>` on refusal (for example
        #: `KO: Snapshot save is skipped. Reason: UNSUPPORTED_VK_APP`).
        self.snapshot_reply = snapshot_reply
        #: Whether a claimed `OK` actually leaves a snapshot directory behind,
        #: so the false-success path can be exercised on its own.
        self.create_snapshot_dir = create_snapshot_dir
        #: What `cmd game set --fps` has pinned, which is what the fake
        #: SurfaceFlinger dump then reports back for the game's uid.
        self.pinned_rate = 60
        #: The display's own vsync mode, which `-vsync-rate` sets and which the
        #: override cannot exceed. Separate from `pinned_rate` because the two
        #: levers fail silently and independently, and a run collecting at 60
        #: under a 120 override is exactly the failure being refused.
        self.display_rate = instance.GUEST_FRAME_RATE_HZ
        #: Whether `dumpsys activity activities` lists the game's activity, one
        #: reading per call, the last repeating. A Play update kills the
        #: activity while leaving the process, so this is scripted apart from
        #: the pid.
        self.activities = [True]
        #: How many launcher intents have been sent.
        self.launches = 0

    def adb(self, instance: CloneInstance, *args: str, timeout: float = 30.0) -> str:
        command = " ".join(args)
        self.commands.append(command)
        if command.startswith("shell ip -o -4 addr"):
            interfaces = "1: lo    inet 127.0.0.1/8 scope host lo"
            if self.online:
                interfaces += "\n2: wlan0    inet 10.0.2.16/24 scope global wlan0"
            return interfaces
        if command.startswith("shell svc"):
            self.online = args[-1] == "enable"
            return ""
        if command == "shell getprop sys.boot_completed":
            return "1"
        if command.startswith("shell pidof"):
            return self.pids[0] if len(self.pids) == 1 else self.pids.pop(0)
        if command == "shell dumpsys activity activities":
            present = self.activities[0] if len(self.activities) == 1 else self.activities.pop(0)
            if not present:
                return "  topResumedActivity=ActivityRecord{NexusLauncher}"
            return (
                "  topResumedActivity=ActivityRecord{com.TechTreeGames.TheTower/"
                "com.unity3d.player.UnityPlayerActivity}"
            )
        if command.startswith("shell monkey"):
            self.launches += 1
            return ""
        if command.startswith("shell cmd game set --fps"):
            self.pinned_rate = int(args[args.index("--fps") + 1])
            return ""
        if command.startswith("shell dumpsys package"):
            return "  Package [com.TechTreeGames.TheTower] (321faf):\n    appId=10218"
        if command == "shell dumpsys SurfaceFlinger":
            # The three readings `confirm_frame_rate` reads back, in the shapes
            # the real dump writes them.
            return (
                f"\t\tGameFrameRateOverrides=\n\t\t\t(uid, gameModeOverride, "
                f"gameDefaultOverride)={{10218, {self.pinned_rate} 60}}\n"
                f"FrameRateOverrides=\n    setFrameRate=\n"
                f"        (uid, frameRate)={{10218, {self.pinned_rate}.00 Hz}}\n"
                f"    activeMode={{id=0, hwcId=0, resolution=360x640, vsyncRate="
                f"{self.display_rate}.00 Hz, dpi=140.00x140.00}}\n"
            )
        if command.startswith("emu avd snapshot save"):
            if self.snapshot_reply.startswith("OK") and self.create_snapshot_dir:
                # A real save leaves the snapshot directory on disk; that is
                # what `snapshot_exists` checks after a reported success.
                name = args[-1]
                bring_up.snapshot_directory(instance).mkdir(parents=True, exist_ok=True)
                (bring_up.snapshot_directory(instance) / name).mkdir(exist_ok=True)
            return self.snapshot_reply
        return ""

    def index_of(self, prefix: str) -> int:
        for index, command in enumerate(self.commands):
            if command.startswith(prefix):
                return index
        raise AssertionError(f"no command starting with {prefix!r} in {self.commands}")


ADB_CALLERS = (instance, bring_up, frame_rate)


def patch_adb(monkeypatch: pytest.MonkeyPatch, answer: Any) -> None:
    """Replace `adb` everywhere it is bound.

    Every module that speaks to an instance imports `adb` by name, so a fake
    installed only on the module it is defined in would leave the copies the
    others hold pointing at the real one — and a test would reach a device.
    """
    for module in ADB_CALLERS:
        monkeypatch.setattr(module, "adb", answer)


def install(
    monkeypatch: pytest.MonkeyPatch,
    clone: FakeClone,
    readings: list[Any] | None = None,
) -> list[int]:
    """Wire the fake clone, the fake bridge and a clock that only sleeps forward.

    No screen-reading route is installed at all, because none exists: readiness
    is the bridge's own reading, and `test_no_screen_reading.py` holds the
    simulation to never reaching for one.
    """
    patch_adb(monkeypatch, clone.adb)
    monkeypatch.setattr(instance, "find_android_tool", lambda name: Path(name))
    monkeypatch.setattr(instance.subprocess, "Popen", lambda *_, **__: None)
    monkeypatch.setattr(bring_up, "expected_compatibility", lambda: None)

    now = [0.0]

    def advance(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr(instance.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(instance.time, "sleep", advance)

    ports: list[int] = []
    queue = list(readings or [])

    class FakeBridgeClient:
        def __init__(self, _host: str, port: int, **_: object) -> None:
            ports.append(port)

        def connect(self) -> None:
            return None

        def read_state(self) -> Any:
            state = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(state, Exception):
                raise state
            return state

        def close(self) -> None:
            return None

    monkeypatch.setattr(bring_up, "InstrumentedBridgeClient", FakeBridgeClient)
    return ports


def test_start_leaves_the_instance_offline_with_the_game_not_launched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`deploy` refuses an online instance, and it is what launches the game next."""
    clone = FakeClone()
    install(monkeypatch, clone)

    start(CloneInstance(), "host")

    assert not clone.online
    assert not [command for command in clone.commands if "monkey" in command]
    # The last thing start does is read the interfaces, which is the offline claim.
    assert clone.commands[-1].startswith("shell ip -o -4 addr")


def test_the_online_window_closes_only_once_the_bridge_reports_the_game_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bridge answers from the splash too, so `main_unavailable` is not ready."""
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [STARTING, STARTING, IDLE])

    launch_game_at_home(CloneInstance())

    assert not clone.online
    enabled = clone.index_of("shell svc wifi enable")
    launched = clone.index_of("shell monkey")
    disabled = clone.index_of("shell svc wifi disable")
    assert enabled < launched < disabled
    # Offline is verified by interface after the radios come down, not assumed.
    assert any(
        command.startswith("shell ip -o -4 addr") for command in clone.commands[disabled:]
    )


def test_a_game_that_never_finishes_starting_never_has_its_network_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OFFLINE modal answers the bridge forever; cutting there strands the game."""
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [STARTING])

    with pytest.raises(CloneError, match="never became ready"):
        launch_game_at_home(CloneInstance())

    assert clone.online
    assert not [command for command in clone.commands if "svc wifi disable" in command]


def test_a_bridge_that_does_not_answer_is_not_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [BridgeDisconnectedError("cannot connect to bridge")])

    with pytest.raises(CloneError, match="never became ready"):
        launch_game_at_home(CloneInstance())

    assert clone.online


def test_readiness_asks_the_bridge_over_this_instances_own_forwarded_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The forward does not survive the emulator it pointed at, so it is remade."""
    clone = FakeClone(online=False)
    ports = install(monkeypatch, clone, [IDLE])

    launch_game_at_home(CloneInstance(index=1))

    assert ports and set(ports) == {47653}
    assert "forward tcp:47653 tcp:47651" in clone.commands


@pytest.mark.parametrize(
    ("online", "pid", "readings", "refusal"),
    [
        (True, "4242", [IDLE], "is online"),
        (False, "", [IDLE], "the game is not running"),
        (False, "4242", [STARTING], "still starting"),
        (False, "4242", [running_round()], "a round is already running"),
    ],
)
def test_snapshot_refuses_a_state_that_is_not_worth_restoring(
    monkeypatch: pytest.MonkeyPatch,
    online: bool,
    pid: str,
    readings: list[Any],
    refusal: str,
) -> None:
    clone = FakeClone(online=online, pid=pid)
    install(monkeypatch, clone, readings)

    with pytest.raises(CloneError, match=refusal):
        save_snapshot(CloneInstance(), "tower_clone_home_offline")

    assert not [command for command in clone.commands if "snapshot save" in command]


def test_snapshot_saves_a_running_idle_offline_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANDROID_AVD_HOME", str(tmp_path / "avd"))
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])

    save_snapshot(CloneInstance(), "nonvisual_baseline_home_offline")

    assert clone.index_of("emu avd snapshot save nonvisual_baseline_home_offline") > 0


def test_snapshot_save_refused_under_the_wrong_renderer_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `KO` reply must fail loudly, naming the reason and the renderer."""
    monkeypatch.setenv("ANDROID_AVD_HOME", str(tmp_path / "avd"))
    clone = FakeClone(
        online=False,
        snapshot_reply="KO: Snapshot save is skipped. Reason: UNSUPPORTED_VK_APP",
    )
    install(monkeypatch, clone, [IDLE])

    with pytest.raises(CloneError, match="UNSUPPORTED_VK_APP"):
        save_snapshot(CloneInstance(), "nonvisual_baseline_home_offline", renderer="host")


def test_snapshot_save_that_claims_ok_but_leaves_no_directory_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Success is verified positively, not assumed from the reply alone."""
    monkeypatch.setenv("ANDROID_AVD_HOME", str(tmp_path / "avd"))
    clone = FakeClone(online=False, snapshot_reply="OK", create_snapshot_dir=False)
    install(monkeypatch, clone, [IDLE])

    with pytest.raises(CloneError, match="directory does not exist"):
        save_snapshot(CloneInstance(), "nonvisual_baseline_home_offline")


def test_restore_claims_only_what_a_snapshot_promises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A snapshot may predate any bridge deployment, so restore must not need one.

    It still has to prove the two things the snapshot is for: never connected,
    and the game already started.
    """
    clone = FakeClone(online=False)
    ports = install(monkeypatch, clone)

    restore(CloneInstance(), "nonvisual_baseline_home_offline")

    assert ports == []
    assert clone.index_of("shell pidof com.TechTreeGames.TheTower") > 0


def test_restore_refuses_a_snapshot_whose_game_is_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = FakeClone(online=False, pid="")
    install(monkeypatch, clone)

    with pytest.raises(CloneError, match="no running game"):
        restore(CloneInstance(), "nonvisual_baseline_home_offline")


def test_the_canonical_evaluation_avd_is_still_refused() -> None:
    with pytest.raises(CloneError, match="canonical evaluation AVD"):
        CloneInstance(avd=instance.CANONICAL_AVD)


def bridge_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: bytes = b"bridge-one"
) -> str:
    """Point the key at a private build directory holding this exact bridge."""
    build = tmp_path / "build"
    build.mkdir(exist_ok=True)
    (build / "libtower_bridge.so").write_bytes(contents)
    monkeypatch.setenv("TOWER_BRIDGE_BUILD_DIR", str(build))
    monkeypatch.setenv("ANDROID_AVD_HOME", str(tmp_path / "avd"))
    return bring_up.bridge_key()


def hold_snapshot(tmp_path: Path, name: str) -> None:
    """Put a snapshot of that name into the clone AVD, as the emulator would."""
    directory = tmp_path / "avd" / f"{instance.CLONE_AVD}.avd" / "snapshots" / name
    directory.mkdir(parents=True)


def record_launches(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    launches: list[list[str]] = []
    monkeypatch.setattr(
        instance.subprocess, "Popen", lambda command, **_: launches.append(command)
    )
    return launches


def test_the_snapshot_key_is_the_bridge_binary_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bridge changes on most commits without its version string moving."""
    first = bridge_build(tmp_path, monkeypatch)
    again = bridge_build(tmp_path, monkeypatch)
    other = bridge_build(tmp_path, monkeypatch, contents=b"bridge-two")

    assert first == again
    assert first != other
    assert bring_up.keyed_snapshot_name(first).endswith(first)
    assert bring_up.keyed_snapshot_name(first).startswith(bring_up.SNAPSHOT_PREFIX)


def test_a_snapshot_taken_for_another_bridge_does_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = bridge_build(tmp_path, monkeypatch, contents=b"bridge-two")
    hold_snapshot(tmp_path, bring_up.keyed_snapshot_name(stale))
    current = bridge_build(tmp_path, monkeypatch)

    assert not bring_up.snapshot_exists(
        CloneInstance(), bring_up.keyed_snapshot_name(current)
    )
    assert bring_up.snapshot_exists(
        CloneInstance(), bring_up.keyed_snapshot_name(stale)
    )


def test_a_matching_snapshot_is_restored_without_a_deploy_or_a_network_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This is the whole point: no radios, no cold launch, ten seconds."""
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, bring_up.keyed_snapshot_name(key))
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])
    launches = record_launches(monkeypatch)
    deployed: list[str] = []

    path = bring_up.bring_up(
        CloneInstance(), "lavapipe", deploy=lambda instance: deployed.append(instance.serial)
    )

    assert path == "restored"
    assert deployed == []
    assert not [command for command in clone.commands if "svc wifi enable" in command]
    assert not [command for command in clone.commands if "monkey" in command]
    assert not [command for command in clone.commands if "snapshot save" in command]
    assert launches[0][launches[0].index("-snapshot") + 1] == (
        bring_up.keyed_snapshot_name(key)
    )


@pytest.mark.parametrize("stale_key_present", [False, True])
def test_a_missing_or_mismatched_snapshot_takes_the_cold_path_and_saves_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale_key_present: bool
) -> None:
    if stale_key_present:
        stale = bridge_build(tmp_path, monkeypatch, contents=b"bridge-two")
        hold_snapshot(tmp_path, bring_up.keyed_snapshot_name(stale))
    key = bridge_build(tmp_path, monkeypatch)
    clone = FakeClone()
    install(monkeypatch, clone, [IDLE])
    launches = record_launches(monkeypatch)
    deployed: list[str] = []

    path = bring_up.bring_up(
        CloneInstance(), "lavapipe", deploy=lambda instance: deployed.append(instance.serial)
    )

    assert path == "cold"
    assert deployed == ["emulator-5556"]
    assert "-no-snapshot-load" in launches[0]
    # The online window is opened for the cold launch and closed again after it.
    enabled = clone.index_of("shell svc wifi enable")
    assert enabled < clone.index_of("shell monkey")
    assert any("svc wifi disable" in command for command in clone.commands[enabled:])
    assert not clone.online
    saved = f"emu avd snapshot save {bring_up.keyed_snapshot_name(key)}"
    assert clone.index_of(saved) > clone.index_of("shell monkey")


def test_the_cold_path_can_be_forced_over_a_matching_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, bring_up.keyed_snapshot_name(key))
    clone = FakeClone()
    install(monkeypatch, clone, [IDLE])
    launches = record_launches(monkeypatch)

    path = bring_up.bring_up(
        CloneInstance(), "lavapipe", deploy=lambda _: None, force_cold=True
    )

    assert path == "cold"
    assert "-no-snapshot-load" in launches[0]


class RestoredWithoutItsGame(FakeClone):
    """A snapshot that comes back with no game process; a cold launch starts one."""

    def adb(self, instance: CloneInstance, *args: str, timeout: float = 30.0) -> str:
        answer = super().adb(instance, *args, timeout=timeout)
        if " ".join(args).startswith("shell monkey"):
            self.pids = ["4242"]
        return answer


def test_a_restored_instance_whose_game_is_gone_falls_back_to_the_cold_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot that does not verify is not used, and is not left running."""
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, bring_up.keyed_snapshot_name(key))
    clone = RestoredWithoutItsGame(online=False, pid="")
    install(monkeypatch, clone, [IDLE])
    record_launches(monkeypatch)

    path = bring_up.bring_up(CloneInstance(), "lavapipe", deploy=lambda _: None)

    assert path == "cold"
    killed = clone.index_of("emu kill")
    assert killed < clone.index_of("shell monkey")
    saved = f"emu avd snapshot save {bring_up.keyed_snapshot_name(key)}"
    assert clone.index_of(saved) > killed


def test_a_restored_instance_that_never_reads_ready_falls_back_to_the_cold_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process can be alive while the game is still starting: `main_unavailable`.

    The restore window is `RESTORED_READY_TIMEOUT` at a five-second poll, so the
    readings below keep the restored instance starting for all of it and only
    then let the cold path reach home.
    """
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, bring_up.keyed_snapshot_name(key))
    clone = FakeClone(online=False)
    starting = int(bring_up.RESTORED_READY_TIMEOUT // 5.0)
    install(monkeypatch, clone, [STARTING] * starting + [IDLE])
    record_launches(monkeypatch)
    deployed: list[str] = []

    path = bring_up.bring_up(
        CloneInstance(), "lavapipe", deploy=lambda instance: deployed.append(instance.serial)
    )

    assert path == "cold"
    assert deployed == ["emulator-5556"]
    assert clone.index_of("emu kill") < clone.index_of("shell svc wifi enable")


def test_a_read_only_instance_takes_the_cold_path_without_saving_a_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-read-only` writes to a throwaway overlay, so it has no snapshot to save."""
    bridge_build(tmp_path, monkeypatch)
    clone = FakeClone()
    install(monkeypatch, clone, [IDLE])
    launches = record_launches(monkeypatch)

    path = bring_up.bring_up(
        CloneInstance(index=1), "lavapipe", deploy=lambda _: None, read_only=True
    )

    assert path == "cold"
    assert "-read-only" in launches[0]
    assert not [command for command in clone.commands if "snapshot save" in command]


def test_a_second_instance_cannot_be_launched_writable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refused before any adb or emulator call, since the emulator would refuse it too."""
    monkeypatch.setattr(instance, "find_android_tool", lambda name: Path(name))
    popen_calls: list[object] = []
    monkeypatch.setattr(
        instance.subprocess, "Popen", lambda *args, **kwargs: popen_calls.append(args)
    )

    with pytest.raises(CloneError, match="must be launched --read-only"):
        instance.launch_emulator(CloneInstance(index=1), "lavapipe", None, read_only=False)

    assert popen_calls == []


def test_a_second_instance_may_launch_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    instance.require_shareable(CloneInstance(index=1), True)
    instance.require_shareable(CloneInstance(index=0), False)


class ExitedProcess:
    """A fake `Popen` handle for a process that has already exited."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode


def test_a_launch_failure_surfaces_the_emulators_own_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The emulator refusing to start must not look like a hang."""
    monkeypatch.setattr(instance, "find_android_tool", lambda name: Path(name))
    monkeypatch.setattr(instance.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(instance.time, "sleep", lambda _: None)

    message = "ERROR | Another emulator instance is running. Please close it.\n"

    def fake_popen(command: list[str], *, stdout: Any, stderr: Any, **_: object) -> ExitedProcess:
        stdout.write(message.encode())
        stdout.flush()
        return ExitedProcess(1)

    monkeypatch.setattr(instance.subprocess, "Popen", fake_popen)

    with pytest.raises(CloneError, match="Another emulator instance is running"):
        instance.launch_emulator(CloneInstance(), "host", None)


def test_a_boot_timeout_includes_the_captured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow boot that never completes must still explain itself, not just time out."""
    monkeypatch.setattr(instance, "find_android_tool", lambda name: Path(name))
    patch_adb(monkeypatch, lambda *_, **__: "")

    now = [0.0]

    def advance(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr(instance.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(instance.time, "sleep", advance)

    class StillRunning:
        returncode = None

        def poll(self) -> None:
            return None

    def fake_popen(command: list[str], *, stdout: Any, stderr: Any, **_: object) -> StillRunning:
        stdout.write(b"emulator: INFO: boot still in progress\n")
        stdout.flush()
        return StillRunning()

    monkeypatch.setattr(instance.subprocess, "Popen", fake_popen)

    with pytest.raises(CloneError, match="boot still in progress"):
        instance.launch_emulator(CloneInstance(), "host", None)


def test_bring_up_under_host_takes_the_cold_path_without_attempting_a_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Snapshots are lavapipe-only; a host bring-up must not even try one.

    A matching snapshot is deliberately held here too, to prove the renderer
    check is what routes this, not a missing snapshot.
    """
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, bring_up.keyed_snapshot_name(key))
    clone = FakeClone()
    install(monkeypatch, clone, [IDLE])
    launches = record_launches(monkeypatch)
    deployed: list[str] = []

    path = bring_up.bring_up(
        CloneInstance(), "host", deploy=lambda instance: deployed.append(instance.serial)
    )

    assert path == "cold"
    assert deployed == ["emulator-5556"]
    assert "-no-snapshot-load" in launches[0]
    assert not [command for command in clone.commands if "snapshot save" in command]


def test_the_display_mode_is_launched_at_the_rate_but_the_game_is_not_raised_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bring-up sets the mode and leaves the game at 60; the raise is deferred.

    An instance already running at the raised rate while a peer booted killed
    that peer's Vulkan surface on device, and `-vsync-rate` alone changes
    nothing the game sees: the per-uid game frame-rate override still pins its
    surface to 60. So every instance boots at the stock rate, on both paths.
    """
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, bring_up.keyed_snapshot_name(key))
    rate = str(instance.GUEST_FRAME_RATE_HZ)

    for renderer, path in (("lavapipe", "restored"), ("host", "cold")):
        clone = FakeClone(online=renderer == "host")
        install(monkeypatch, clone, [IDLE])
        launches = record_launches(monkeypatch)

        assert bring_up.bring_up(CloneInstance(), renderer, deploy=lambda _: None) == path

        assert launches[0][launches[0].index("-vsync-rate") + 1] == rate
        assert not [command for command in clone.commands if "cmd game set" in command]
        assert clone.pinned_rate == 60


def test_raising_the_rate_pins_the_game_to_the_rate_the_display_was_launched_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finding is that the mode and the override have to agree.

    A raised `-vsync-rate` with the per-uid game override still in place renders
    at 60, and an override above a 60 Hz mode does too, so the two are one
    constant and the raise reads both back out of SurfaceFlinger before any
    episode is collected.
    """
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])
    rate = instance.GUEST_FRAME_RATE_HZ

    frame_rate.raise_frame_rate(CloneInstance())

    assert f"shell cmd game set --fps {rate} {instance.PACKAGE}" in clone.commands
    assert clone.pinned_rate == rate


def test_a_run_may_choose_its_rate_and_both_levers_follow_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The equivalence comparison's 60 Hz arm: one value, both levers.

    The two levers only ever fail apart, so a per-run rate has to reach the
    launch flag and the per-uid override from the same number, and the
    confirmation has to read that number back rather than the constant.
    """
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, bring_up.keyed_snapshot_name(key))
    clone = FakeClone(online=False)
    clone.display_rate = 60
    install(monkeypatch, clone, [IDLE])
    launches = record_launches(monkeypatch)

    bring_up.bring_up(CloneInstance(), "lavapipe", deploy=lambda _: None, frame_rate_hz=60)
    frame_rate.raise_frame_rate(CloneInstance(), 60)

    assert launches[0][launches[0].index("-vsync-rate") + 1] == "60"
    assert f"shell cmd game set --fps 60 {instance.PACKAGE}" in clone.commands
    assert clone.pinned_rate == 60


def test_a_rate_the_display_was_not_launched_at_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 60 Hz arm raised against a 120 Hz display mode is not a 60 Hz arm."""
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])

    with pytest.raises(CloneError, match="the guest is not at 60 Hz"):
        frame_rate.raise_frame_rate(CloneInstance(), 60)


def test_a_guest_still_at_sixty_is_refused_rather_than_collected_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither lever reports back, so a silent 60 Hz run must not look healthy."""
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])

    with pytest.raises(CloneError, match="the guest is not at"):
        frame_rate.confirm_frame_rate(CloneInstance())


def test_teardown_resets_the_frame_rate_override_it_pinned() -> None:
    """Bring-up modifies device state, so cleanup has to hand it back.

    Read from the script rather than run: cleanup is root-only device work
    (`su`, `mount`, `restorecon`), and standing that up in a fake would be a
    reimplementation rather than a test. What is checked is what reading cannot
    get wrong — the reset is issued, a nonzero return does not abort the rest of
    cleanup under `set -e`, and the restore is read back rather than announced.
    """
    cleanup = BRIDGE_SCRIPT.read_text()
    assert 'device shell cmd game reset "$package" > /dev/null 2>&1 ||' in cleanup
    assert "game_frame_rate_override: reset-issued (unverified)" in cleanup
    assert "game_frame_rate_override: NOT-reset" in cleanup


def test_a_game_killed_by_a_play_update_is_relaunched_and_reaches_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Play installs WebView over the running game; the activity dies, the pid stays."""
    clone = FakeClone(online=False)
    clone.activities = [True, False, True]
    install(monkeypatch, clone, [STARTING, STARTING, STARTING, IDLE])

    launch_game_at_home(CloneInstance())

    # The launcher intent that started it, plus exactly one relaunch.
    assert clone.launches == 2
    assert not clone.online


def test_a_game_that_keeps_being_killed_is_relaunched_at_most_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cap is what keeps a standing fault from looping instead of reporting."""
    clone = FakeClone(online=False)
    clone.activities = [True, False, True, False]
    install(monkeypatch, clone, [STARTING])

    with pytest.raises(CloneError, match="never became ready"):
        launch_game_at_home(CloneInstance())

    assert clone.launches == 1 + bring_up.MAX_RELAUNCHES


def test_a_relaunch_never_turns_a_radio_back_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kill can land after the network is cut, and a relaunch may not reopen it."""
    clone = FakeClone(online=False)
    clone.activities = [True, False, True]
    instance = CloneInstance()
    install(monkeypatch, clone, [STARTING, STARTING, STARTING, IDLE])

    bring_up.wait_until_ready(
        instance,
        timeout=300.0,
        poll=bring_up.ONLINE_POLL_SECONDS,
        relaunch=lambda: bring_up.launch_game(instance),
    )

    assert clone.launches == 1
    assert not clone.online
    assert not [command for command in clone.commands if command.startswith("shell svc")]


def test_an_emulator_log_under_test_never_touches_the_directory_the_live_ones_use(
    emulator_log_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test that fakes only `Popen` still opens this file `"wb"` for writing."""
    monkeypatch.setattr(instance, "find_android_tool", lambda name: Path(name))
    patch_adb(monkeypatch, lambda *_, **__: "1")
    record_launches(monkeypatch)

    instance.launch_emulator(CloneInstance(index=1), "host", None, read_only=True)

    assert (emulator_log_directory / "tower-rl-emulator-emulator-5558.log").is_file()
    assert emulator_log_directory == instance.EMULATOR_LOG_DIRECTORY


def test_an_override_the_display_mode_cannot_honour_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two levers fail independently: a 120 override on a 60 Hz mode renders 60."""
    clone = FakeClone(online=False)
    clone.display_rate = 60
    install(monkeypatch, clone, [IDLE])

    with pytest.raises(CloneError, match="display vsync mode 60"):
        frame_rate.raise_frame_rate(CloneInstance())

    # The override itself was taken; it is the display that cannot honour it.
    assert clone.pinned_rate == instance.GUEST_FRAME_RATE_HZ


def test_a_reading_a_hair_off_the_rate_is_still_the_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SurfaceFlinger computes the applied rate from a vsync period; 120.000004 is 120."""
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])
    patch_adb(
        monkeypatch,
        lambda instance, *args, **kwargs: (
            clone.adb(instance, *args, **kwargs).replace(".00 Hz}", ".000004 Hz}")
        ),
    )

    frame_rate.raise_frame_rate(CloneInstance())


def test_a_game_killed_as_the_network_is_cut_fails_by_name_and_stays_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kill lands after the cut too, and there it is fatal, not recoverable.

    `M1B-E049` forced this on device: re-issuing the launcher intent with the
    network gone leaves the game at `main_unavailable` until the timeout,
    because a launch without a network stops at the Firebase check and the
    OFFLINE modal (`M1B-E010`). So the instance is lost; what the run must carry
    is the cause, by name, and no attempt to reopen the network.
    """
    clone = FakeClone(online=False)
    # Ready, then no process at the post-cut check, and no activity behind it.
    clone.pids = ["4242", ""]
    clone.activities = [False]
    install(monkeypatch, clone, [IDLE])

    with pytest.raises(CloneError, match="did not survive the network being cut") as error:
        launch_game_at_home(CloneInstance())

    assert bring_up.GAME_ACTIVITY_LOST in str(error.value)
    # The launcher intent that started it, and nothing after it.
    assert clone.launches == 1
    assert not clone.online
    after_the_cut = clone.index_of("shell svc wifi disable")
    assert not [
        command
        for command in clone.commands[after_the_cut:]
        if command.startswith("shell svc") and command.endswith("enable")
    ]


def test_a_game_that_survived_the_cut_is_not_blamed_for_a_lost_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A game that is present but unready is a different fault and keeps its own message."""
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])

    assert bring_up.lost_activity_cause(CloneInstance()) == ""
    bring_up.require_game_activity(CloneInstance())


def test_a_lost_activity_before_the_rate_is_raised_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gap between bring-up and the first episode: reported, never restarted.

    A game with no activity has no surface, so the raise would otherwise fail
    with a bare `applied frame rate absent`. Nothing is relaunched here: after
    the cut there is no launch that reaches home.
    """
    clone = FakeClone(online=False)
    clone.activities = [False]
    install(monkeypatch, clone, [IDLE])

    with pytest.raises(CloneError, match="lost its activity"):
        bring_up.require_game_activity(CloneInstance())

    assert clone.launches == 0


def test_an_absent_applied_rate_names_the_missing_game_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare `applied frame rate absent` cost a fleet run its diagnosis."""
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])
    patch_adb(
        monkeypatch,
        lambda instance, *args, **kwargs: re.sub(
            r"\(uid, frameRate\)=\{10218, [\d.]+ Hz\}",
            "",
            clone.adb(instance, *args, **kwargs),
        ),
    )

    with pytest.raises(CloneError, match="lost its activity"):
        frame_rate.raise_frame_rate(CloneInstance())


def test_the_activity_a_kill_left_in_the_task_history_is_not_a_present_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the relaunch never fired on device: the dump keeps the dead record.

    `emulator-5558` (run F) was killed by the guest's Play at 16:12:59 —
    `Killing ...TheTower ... due to installPackageLI`, `Force removing
    ActivityRecord{... UnityPlayerActivity}: app died` — and moments later
    `pidof` answered nothing while the component name was still in the dump.
    Whatever else is in the history, only the resumed activity is the game
    being on screen.
    """
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])
    patch_adb(
        monkeypatch,
        lambda instance, *args, **kwargs: (
            "  topResumedActivity=ActivityRecord{NexusLauncher}\n"
            "  * Hist #0: ActivityRecord{u0 com.TechTreeGames.TheTower/"
            "com.unity3d.player.UnityPlayerActivity t70 f}}\n"
            if " ".join(args) == "shell dumpsys activity activities"
            else clone.adb(instance, *args, **kwargs)
        ),
    )

    assert not bring_up.game_activity_present(CloneInstance())
