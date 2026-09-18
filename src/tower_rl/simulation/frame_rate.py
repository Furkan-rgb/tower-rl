"""The rate the guest paces the game at, and the two levers that set it.

`-vsync-rate` is the other lever and belongs to the emulator invocation, so it
lives in `instance` with the constant both take their value from. What is here
is the second one: SurfaceFlinger's per-uid game frame-rate override, which
pins the game surface to 60 whatever mode the display is in until it is lifted,
and the read-back that refuses an instance where the two do not agree.
"""

from __future__ import annotations

import re
import time

from tower_rl.simulation.instance import (
    GUEST_FRAME_RATE_HZ,
    PACKAGE,
    CloneError,
    CloneInstance,
    adb,
)

#: How long SurfaceFlinger is given to apply a raised rate before the instance is
#: called unusable. Observed on device to take a beat, not to be slow.
FRAME_RATE_CONFIRM_TIMEOUT = 20.0
#: SurfaceFlinger reports the applied rate as a float it computed from a vsync
#: period, so an instance genuinely at the rate can read `120.000004`. The
#: readings are compared within this, because the failure worth refusing is a
#: surface still at 60, not a rounding difference.
FRAME_RATE_TOLERANCE_HZ = 0.5


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


def confirm_frame_rate(instance: CloneInstance, frame_rate_hz: int = GUEST_FRAME_RATE_HZ) -> None:
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
    rate = frame_rate_hz
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


def raise_frame_rate(instance: CloneInstance, frame_rate_hz: int = GUEST_FRAME_RATE_HZ) -> None:
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
    adb(instance, "shell", "cmd", "game", "set", "--fps", str(frame_rate_hz), PACKAGE)
    confirm_frame_rate(instance, frame_rate_hz)
