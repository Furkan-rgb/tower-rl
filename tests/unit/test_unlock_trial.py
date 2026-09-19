"""Which instance the unlock trial is allowed to write to.

The trial instrument writes to the live game, so the instance it reaches has to
be the instance the operator named. The port is what actually selects it — the
serial is a label the bridge socket never sees — so the port is derived from the
serial rather than defaulted beside it, and a serial the scheme does not cover
is refused rather than guessed at.

Nothing here opens a socket or starts an emulator: the derivation is a pure
function and is exercised as one.
"""

from __future__ import annotations

import pytest
import unlock_trial

from tower_rl.simulation.instance import CloneInstance


def test_the_bridge_port_is_the_one_the_fleet_forwards_for_that_serial() -> None:
    # Derived against the instance model itself rather than against numbers
    # copied out of it, so the two cannot drift apart silently.
    for index in range(4):
        instance = CloneInstance(index=index)

        assert unlock_trial.bridge_port_for(instance.serial) == instance.bridge_host_port


def test_the_canonical_evaluation_avd_is_refused_by_name() -> None:
    with pytest.raises(SystemExit, match="canonical evaluation AVD"):
        unlock_trial.bridge_port_for("emulator-5554")


@pytest.mark.parametrize(
    "serial",
    ["emulator-5557", "emulator-5554", "emulator-0", "emulator-", "localhost:5556", "5556"],
    ids=["odd", "canonical", "below-index-0", "empty", "not-an-emulator", "bare-port"],
)
def test_a_serial_off_the_console_port_scheme_is_refused_not_guessed_at(serial: str) -> None:
    # An odd port is the emulator's adb port, not its console port, and anything
    # below index 0 is not a fleet instance at all. Either way the bridge port
    # cannot be derived, and deriving one anyway would aim a write at whatever
    # instance that port reaches.
    with pytest.raises(SystemExit):
        unlock_trial.bridge_port_for(serial)


def test_a_port_that_disagrees_with_the_serial_refuses_rather_than_choosing_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = CloneInstance(index=1)
    monkeypatch.setattr(
        "sys.argv",
        ["unlock_trial.py", "--serial", instance.serial, "--port", str(instance.bridge_host_port)],
    )
    # The agreeing pair gets past the guard and stops at the next thing the
    # script does, which is reading the private build identity.
    monkeypatch.setattr(
        unlock_trial, "compatibility", lambda _: (_ for _ in ()).throw(RuntimeError("stop here"))
    )
    with pytest.raises(RuntimeError, match="stop here"):
        unlock_trial.main()

    monkeypatch.setattr(
        "sys.argv",
        ["unlock_trial.py", "--serial", instance.serial, "--port", "47652"],
    )
    with pytest.raises(SystemExit, match="does not belong to"):
        unlock_trial.main()
