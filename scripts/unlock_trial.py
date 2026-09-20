#!/usr/bin/env python3
"""Write every in-run upgrade availability flag true, and report what stands.

The trial instrument for board #54 and nothing else. It answers one question —
does a bridge write to `Main.upgradeUnlocked` / `upgradeDefenseUnlocked` /
`upgradeUtilityUnlocked` land on the live instance at all — by reading the three
arrays, writing them all true, and reading them again. Whether the write
*survives* a round, a save, or a relaunch is settled by a human running the
round and re-reading; this script does not start a round, tap anything, or take
a screenshot.

    uv run python scripts/unlock_trial.py --serial emulator-5556

Both commands exist only in a bridge built with `-DTOWER_BRIDGE_DIAGNOSTICS=ON`.
A production bridge does not reject them, it cannot parse them: it answers
`protocol_error` and drops the connection, and this script says what that means.
Select the diagnostics bridge with `TOWER_BRIDGE_BUILD_DIR` pointed at its build
tree; never repoint `state/bridge/current` at it, which is the pointer every
measured run takes and must keep naming the production artifact.

The forward to the bridge's device port is assumed to be up, exactly as
`compare_arms.py` assumes it: bring-up establishes it.

This changes what the game offers inside a live process. Run it on a disposable
clone only, with the interface offline, and discard the clone afterwards — never
against the canonical evaluation AVD, and never against an instance whose state
is going to be promoted to a named snapshot without a decision to do so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tower_rl.simulation.bridge import (  # noqa: E402
    bridge_build_directory,
    compatibility,
)
from tower_rl.simulation.instance import (  # noqa: E402
    CANONICAL_AVD,
    FIRST_BRIDGE_HOST_PORT,
    FIRST_CONSOLE_PORT,
)
from tower_rl.simulation.instrumented_bridge import (  # noqa: E402
    BridgeCompatibilityError,
    BridgeObservation,
    InstrumentedBridgeClient,
    InstrumentedBridgeError,
    UnlockFamilyState,
)

#: The canonical evaluation AVD's serial. Named here because it is the one
#: instance this script must never reach, and the port derivation below cannot
#: express that: its console port is below index 0's.
CANONICAL_SERIAL = "emulator-5554"


def bridge_port_for(serial: str) -> int:
    """The host port forwarded to the bridge on that instance, from its serial.

    The serial is the console port (`instance.py`: even, two apart, index 0 at
    5556), and each instance forwards its own host port from the same base. For
    a read-only runner an unmatched `--serial`/`--port` pair is a confusing
    session; for an instrument that *writes* to the game it is a write aimed at
    whichever instance the port happens to reach, which may be a measured one.
    So the port is derived here rather than defaulted, and a serial that is not
    on the scheme is refused instead of being guessed at.
    """
    if serial == CANONICAL_SERIAL:
        raise SystemExit(f"refusing to run against the canonical evaluation AVD ({CANONICAL_AVD})")
    prefix = "emulator-"
    if not serial.startswith(prefix) or not serial[len(prefix) :].isdigit():
        raise SystemExit(f"not an emulator serial: {serial!r}")
    port = int(serial[len(prefix) :])
    if port < FIRST_CONSOLE_PORT or port % 2 != 0:
        raise SystemExit(
            f"{serial} is not on the emulator console-port scheme "
            f"(even, from {FIRST_CONSOLE_PORT}); refusing to guess its bridge port"
        )
    return FIRST_BRIDGE_HOST_PORT + (port - FIRST_CONSOLE_PORT) // 2


def _report(title: str, families: tuple[UnlockFamilyState, ...]) -> None:
    print(title)
    for family in families:
        print(f"  {family.family:<8} length={family.length:<3} true={family.true_count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=f"emulator-{FIRST_CONSOLE_PORT}")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="the forwarded bridge port; derived from --serial, and must agree with it",
    )
    arguments = parser.parse_args()
    port = bridge_port_for(arguments.serial)
    if arguments.port is not None and arguments.port != port:
        # The port is what actually selects the instance. A pair that disagrees
        # names one instance and writes to another.
        raise SystemExit(
            f"--port {arguments.port} does not belong to {arguments.serial}, whose bridge "
            f"port is {port}"
        )

    expected = compatibility(bridge_build_directory())
    client = InstrumentedBridgeClient(
        "127.0.0.1",
        port,
        expected_compatibility=expected,
        connect_timeout=5.0,
        read_timeout=30.0,
        heartbeat_timeout=30.0,
    )
    try:
        client.connect()
        state = client.read_state()
        if not isinstance(state, BridgeObservation):
            raise SystemExit(f"the game holds no initialized run: {state.reason}")
        _report("before:", client.read_unlock_state(expected_sequence=state.sequence))

        state = client.read_state()
        if not isinstance(state, BridgeObservation):
            raise SystemExit(f"the run went away before the write: {state.reason}")
        _report("after write:", client.unlock_all_upgrades(expected_sequence=state.sequence))

        state = client.read_state()
        if not isinstance(state, BridgeObservation):
            raise SystemExit(f"the run went away after the write: {state.reason}")
        _report("read back:", client.read_unlock_state(expected_sequence=state.sequence))
    except BridgeCompatibilityError as error:
        # The production parser has no such command kind, so the frame fails to
        # parse and the bridge drops the connection. That is the artifact doing
        # its job, not a fault to debug.
        raise SystemExit(
            "this bridge has no unlock commands (production build?) — point "
            f"TOWER_BRIDGE_BUILD_DIR at the diagnostics build: {error}"
        ) from error
    except InstrumentedBridgeError as error:
        raise SystemExit(f"the bridge on {arguments.serial} did not answer: {error}") from error
    finally:
        client.close()

    print()
    print("The write is in memory only. To learn whether it survives a round:")
    print("  1. start one round the ordinary way, through the existing adapter")
    print("     (`InstrumentedRunAdapter.begin_episode`), and let it play out;")
    print("  2. run this script again with no further write and compare the")
    print("     'before:' counts against the 'read back:' counts above;")
    print("  3. for persistence across a relaunch, restart the game process and")
    print("     read once more before writing anything.")
    print("Counts that fall back are the game recomputing availability, which is")
    print("the outcome board #54 is looking for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
