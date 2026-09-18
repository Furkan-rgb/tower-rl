"""Which instance an actor addresses, and how its emulator is invoked.

No emulator and no adb: instance identity is derived arithmetic and the
invocation is a list of arguments, so both are read back directly.
"""

from __future__ import annotations

import pytest

from tower_rl.simulation.instance import (
    CANONICAL_AVD,
    CLONE_AVD,
    CloneError,
    CloneInstance,
    emulator_command,
)


def test_instance_index_derives_an_even_console_port_and_its_serial() -> None:
    assert (CloneInstance().console_port, CloneInstance().serial) == (5556, "emulator-5556")
    ports = [CloneInstance(index=index).console_port for index in range(4)]
    assert ports == [5556, 5558, 5560, 5562]
    assert all(port % 2 == 0 for port in ports)
    assert CloneInstance(index=3).serial == "emulator-5562"


def test_each_instance_gets_its_own_bridge_host_port() -> None:
    ports = [CloneInstance(index=index).bridge_host_port for index in range(3)]
    assert ports == [47652, 47653, 47654]
    assert len(set(ports)) == 3


def test_the_canonical_evaluation_avd_is_refused() -> None:
    with pytest.raises(CloneError, match="canonical"):
        CloneInstance(index=0, avd=CANONICAL_AVD)
    with pytest.raises(CloneError):
        CloneInstance(index=-1)


def test_instances_share_the_clone_avd_read_only_on_their_own_port() -> None:
    command = emulator_command(
        CloneInstance(index=2),
        binary="emulator",
        renderer="lavapipe",
        snapshot=None,
        read_only=True,
        cores=4,
    )
    assert command[1] == f"@{CLONE_AVD}"
    assert "-read-only" in command
    assert command[command.index("-port") + 1] == "5560"
    assert "-no-snapshot-load" in command


def test_a_writable_instance_is_not_launched_read_only() -> None:
    command = emulator_command(
        CloneInstance(),
        binary="emulator",
        renderer="lavapipe",
        snapshot="home_offline",
        read_only=False,
        cores=8,
    )
    assert "-read-only" not in command
    assert command[command.index("-snapshot") + 1] == "home_offline"


def test_an_instance_is_headless_unless_a_window_is_asked_for() -> None:
    """Every collecting and measuring path is `-no-window`; spectating is not.

    The flag is the whole difference: a spectated instance is the same
    read-only clone on the same port, with a window for a human to watch.
    """
    headless = emulator_command(
        CloneInstance(),
        binary="emulator",
        renderer="host",
        snapshot=None,
        read_only=True,
        cores=4,
    )
    watched = emulator_command(
        CloneInstance(),
        binary="emulator",
        renderer="host",
        snapshot=None,
        read_only=True,
        cores=4,
        windowed=True,
    )

    assert "-no-window" in headless, "headless is the default every fleet path takes"
    assert "-no-window" not in watched
    assert [part for part in headless if part != "-no-window"] == watched
