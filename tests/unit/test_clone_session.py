"""Bring-up readiness, without an emulator and without a pixel.

The clone is stood in for: adb answers from a fake instance, the bridge answers
from a fake client, and the clock advances only when the code sleeps. What is
under test is the decision this script exists to make — when the game has
finished starting up, so that it is safe to cut the network and to snapshot —
and the device-safety properties that hang off it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import clone_session  # noqa: E402
from clone_session import (  # noqa: E402
    CloneError,
    CloneInstance,
    launch_game_at_home,
    restore,
    save_snapshot,
    start,
)

from tower_rl.infrastructure import adb_device, visual_profile  # noqa: E402
from tower_rl.infrastructure.instrumented_bridge import (  # noqa: E402
    BridgeDisconnectedError,
    BridgeObservation,
    BridgeRunUnavailable,
)

STARTING = BridgeRunUnavailable(1, "main_unavailable")
IDLE = BridgeRunUnavailable(2, "no_initialized_run")


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
        self.pid = pid
        self.commands: list[str] = []
        #: The emulator console's reply to `snapshot save`. Real replies are
        #: `OK` on success or `KO: <reason>` on refusal (for example
        #: `KO: Snapshot save is skipped. Reason: UNSUPPORTED_VK_APP`).
        self.snapshot_reply = snapshot_reply
        #: Whether a claimed `OK` actually leaves a snapshot directory behind,
        #: so the false-success path can be exercised on its own.
        self.create_snapshot_dir = create_snapshot_dir

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
            return self.pid
        if command.startswith("emu avd snapshot save"):
            if self.snapshot_reply.startswith("OK") and self.create_snapshot_dir:
                # A real save leaves the snapshot directory on disk; that is
                # what `snapshot_exists` checks after a reported success.
                name = args[-1]
                clone_session.snapshot_directory(instance).mkdir(parents=True, exist_ok=True)
                (clone_session.snapshot_directory(instance) / name).mkdir(exist_ok=True)
            return self.snapshot_reply
        return ""

    def index_of(self, prefix: str) -> int:
        for index, command in enumerate(self.commands):
            if command.startswith(prefix):
                return index
        raise AssertionError(f"no command starting with {prefix!r} in {self.commands}")


def install(
    monkeypatch: pytest.MonkeyPatch,
    clone: FakeClone,
    readings: list[Any] | None = None,
) -> list[int]:
    """Wire the fake clone, the fake bridge and a clock that only sleeps forward.

    Every screen-reading route is replaced by a double that fails if it is used
    at all: readiness must be decided without a screenshot and without the visual
    profile, whatever else changes.
    """
    monkeypatch.setattr(clone_session, "adb", clone.adb)
    monkeypatch.setattr(clone_session, "find_android_tool", lambda name: Path(name))
    monkeypatch.setattr(clone_session.subprocess, "Popen", lambda *_, **__: None)
    monkeypatch.setattr(clone_session, "expected_compatibility", lambda: None)

    def refuse_screenshot(*_: object, **__: object) -> None:
        raise AssertionError("readiness must not read the screen")

    monkeypatch.setattr(adb_device.AdbDevice, "screenshot", refuse_screenshot)
    monkeypatch.setattr(visual_profile, "classify", refuse_screenshot)

    now = [0.0]

    def advance(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr(clone_session.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(clone_session.time, "sleep", advance)

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

    monkeypatch.setattr(clone_session, "InstrumentedBridgeClient", FakeBridgeClient)
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
        CloneInstance(avd=clone_session.CANONICAL_AVD)


def bridge_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: bytes = b"bridge-one"
) -> str:
    """Point the key at a private build directory holding this exact bridge."""
    build = tmp_path / "build"
    build.mkdir(exist_ok=True)
    (build / "libtower_bridge.so").write_bytes(contents)
    monkeypatch.setenv("TOWER_BRIDGE_BUILD_DIR", str(build))
    monkeypatch.setenv("ANDROID_AVD_HOME", str(tmp_path / "avd"))
    return clone_session.bridge_key()


def hold_snapshot(tmp_path: Path, name: str) -> None:
    """Put a snapshot of that name into the clone AVD, as the emulator would."""
    directory = tmp_path / "avd" / f"{clone_session.CLONE_AVD}.avd" / "snapshots" / name
    directory.mkdir(parents=True)


def record_launches(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    launches: list[list[str]] = []
    monkeypatch.setattr(
        clone_session.subprocess, "Popen", lambda command, **_: launches.append(command)
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
    assert clone_session.keyed_snapshot_name(first).endswith(first)
    assert clone_session.keyed_snapshot_name(first).startswith(clone_session.SNAPSHOT_PREFIX)


def test_a_snapshot_taken_for_another_bridge_does_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = bridge_build(tmp_path, monkeypatch, contents=b"bridge-two")
    hold_snapshot(tmp_path, clone_session.keyed_snapshot_name(stale))
    current = bridge_build(tmp_path, monkeypatch)

    assert not clone_session.snapshot_exists(
        CloneInstance(), clone_session.keyed_snapshot_name(current)
    )
    assert clone_session.snapshot_exists(
        CloneInstance(), clone_session.keyed_snapshot_name(stale)
    )


def test_a_matching_snapshot_is_restored_without_a_deploy_or_a_network_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This is the whole point: no radios, no cold launch, ten seconds."""
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, clone_session.keyed_snapshot_name(key))
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])
    launches = record_launches(monkeypatch)
    deployed: list[str] = []

    path = clone_session.bring_up(
        CloneInstance(), "lavapipe", deploy=lambda instance: deployed.append(instance.serial)
    )

    assert path == "restored"
    assert deployed == []
    assert not [command for command in clone.commands if "svc wifi enable" in command]
    assert not [command for command in clone.commands if "monkey" in command]
    assert not [command for command in clone.commands if "snapshot save" in command]
    assert launches[0][launches[0].index("-snapshot") + 1] == (
        clone_session.keyed_snapshot_name(key)
    )


@pytest.mark.parametrize("stale_key_present", [False, True])
def test_a_missing_or_mismatched_snapshot_takes_the_cold_path_and_saves_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale_key_present: bool
) -> None:
    if stale_key_present:
        stale = bridge_build(tmp_path, monkeypatch, contents=b"bridge-two")
        hold_snapshot(tmp_path, clone_session.keyed_snapshot_name(stale))
    key = bridge_build(tmp_path, monkeypatch)
    clone = FakeClone()
    install(monkeypatch, clone, [IDLE])
    launches = record_launches(monkeypatch)
    deployed: list[str] = []

    path = clone_session.bring_up(
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
    saved = f"emu avd snapshot save {clone_session.keyed_snapshot_name(key)}"
    assert clone.index_of(saved) > clone.index_of("shell monkey")


def test_the_cold_path_can_be_forced_over_a_matching_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, clone_session.keyed_snapshot_name(key))
    clone = FakeClone()
    install(monkeypatch, clone, [IDLE])
    launches = record_launches(monkeypatch)

    path = clone_session.bring_up(
        CloneInstance(), "lavapipe", deploy=lambda _: None, force_cold=True
    )

    assert path == "cold"
    assert "-no-snapshot-load" in launches[0]


class RestoredWithoutItsGame(FakeClone):
    """A snapshot that comes back with no game process; a cold launch starts one."""

    def adb(self, instance: CloneInstance, *args: str, timeout: float = 30.0) -> str:
        answer = super().adb(instance, *args, timeout=timeout)
        if " ".join(args).startswith("shell monkey"):
            self.pid = "4242"
        return answer


def test_a_restored_instance_whose_game_is_gone_falls_back_to_the_cold_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot that does not verify is not used, and is not left running."""
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, clone_session.keyed_snapshot_name(key))
    clone = RestoredWithoutItsGame(online=False, pid="")
    install(monkeypatch, clone, [IDLE])
    record_launches(monkeypatch)

    path = clone_session.bring_up(CloneInstance(), "lavapipe", deploy=lambda _: None)

    assert path == "cold"
    killed = clone.index_of("emu kill")
    assert killed < clone.index_of("shell monkey")
    saved = f"emu avd snapshot save {clone_session.keyed_snapshot_name(key)}"
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
    hold_snapshot(tmp_path, clone_session.keyed_snapshot_name(key))
    clone = FakeClone(online=False)
    starting = int(clone_session.RESTORED_READY_TIMEOUT // 5.0)
    install(monkeypatch, clone, [STARTING] * starting + [IDLE])
    record_launches(monkeypatch)
    deployed: list[str] = []

    path = clone_session.bring_up(
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

    path = clone_session.bring_up(
        CloneInstance(index=1), "lavapipe", deploy=lambda _: None, read_only=True
    )

    assert path == "cold"
    assert "-read-only" in launches[0]
    assert not [command for command in clone.commands if "snapshot save" in command]


def test_bring_up_under_host_takes_the_cold_path_without_attempting_a_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Snapshots are lavapipe-only; a host bring-up must not even try one.

    A matching snapshot is deliberately held here too, to prove the renderer
    check is what routes this, not a missing snapshot.
    """
    key = bridge_build(tmp_path, monkeypatch)
    hold_snapshot(tmp_path, clone_session.keyed_snapshot_name(key))
    clone = FakeClone()
    install(monkeypatch, clone, [IDLE])
    launches = record_launches(monkeypatch)
    deployed: list[str] = []

    path = clone_session.bring_up(
        CloneInstance(), "host", deploy=lambda instance: deployed.append(instance.serial)
    )

    assert path == "cold"
    assert deployed == ["emulator-5556"]
    assert "-no-snapshot-load" in launches[0]
    assert not [command for command in clone.commands if "snapshot save" in command]
