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

    def __init__(self, *, online: bool = True, pid: str = "4242") -> None:
        self.online = online
        self.pid = pid
        self.commands: list[str] = []

    def adb(self, _instance: CloneInstance, *args: str, timeout: float = 30.0) -> str:
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


def test_snapshot_saves_a_running_idle_offline_game(monkeypatch: pytest.MonkeyPatch) -> None:
    clone = FakeClone(online=False)
    install(monkeypatch, clone, [IDLE])

    save_snapshot(CloneInstance(), "nonvisual_baseline_home_offline")

    assert clone.index_of("emu avd snapshot save nonvisual_baseline_home_offline") > 0


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
