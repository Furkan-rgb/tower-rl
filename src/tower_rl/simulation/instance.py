"""One emulator instance of the clone: its identity, and its process.

The lowest layer of the simulation. Everything else here addresses an instance
through `CloneInstance` and speaks to it through `adb`, so instance identity —
index, AVD, console port, adb serial, forwarded bridge port — is derived in one
place rather than restated by every caller, and the canonical evaluation AVD is
refused before anything can reach it.

`launch_emulator` is the only place an emulator process is started, and
`wait_for_boot` lives with it because a launch that never boots is only
distinguishable from a slow one by watching the process it started.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tower_rl.simulation.android_sdk import find_android_tool

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
#: The rate the guest paces the game at, in Hz. The game's frame pacing is a
#: guest-side vsync timer, so a decision's frames arrive at whatever rate the
#: guest display runs, and raising it is what makes an advance cheaper in wall
#: time (60 Hz -> 16.17 ms a frame, 120 Hz -> 8.3 ms). Two settings have to agree
#: or the guest keeps 60: `-vsync-rate` below sets the display's physical vsync
#: mode, and `raise_frame_rate` lifts SurfaceFlinger's per-uid game
#: frame-rate override (`ro.surface_flinger.game_default_frame_rate_override=60`)
#: that otherwise pins the game surface to 60 whatever mode the display is in.
#: They are one constant here precisely because they cannot be allowed to drift
#: apart: a 120 override against a 60 Hz mode still renders at 60. A run that
#: needs another rate — the 60 Hz arm of the behavioural equivalence
#: comparison — passes `frame_rate_hz` down the bring-up chain instead, where it
#: still feeds both levers from the one value.
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
#: The measured ceiling, as a name because a per-run `frame_rate_hz` is held to
#: it too and not only this constant.
MAX_GUEST_FRAME_RATE_HZ = 300
assert GUEST_FRAME_RATE_HZ <= MAX_GUEST_FRAME_RATE_HZ, (
    "no measured fps supports a guest rate above 300 Hz"
)

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
    frame_rate_hz: int = GUEST_FRAME_RATE_HZ,
) -> list[str]:
    """The exact invocation for one instance.

    `-read-only` is what lets several instances share the one AVD: the base
    image stays untouched and each instance writes to its own overlay. It is
    also incompatible with saving a snapshot, which is why it is a choice and
    not the default.

    `-vsync-rate` is half of the guest rate; `raise_frame_rate` is the other
    half, and both take it from the same `frame_rate_hz` — which defaults to
    `GUEST_FRAME_RATE_HZ` and is a per-run choice only so one fleet run can be
    collected at another rate without editing the constant: the 60 Hz arm of the
    behavioural equivalence comparison is exactly that run.

    Every instance this builds is `-no-window`, which is why raising the rate
    here is safe: the emulator warns that exceeding the host display's refresh
    rate is undefined, and a headless instance is driving no host display at all.
    The windowed review path (`launch_avd.sh`) keeps default pacing for that
    reason, and because a human watching a checkpoint play wants the game's own
    speed, not the fleet's.
    """
    command = [
        binary, f"@{instance.avd}",
        "-gpu", renderer, "-no-audio", "-no-boot-anim", "-no-window",
        "-cores", str(cores), "-port", str(instance.console_port), "-no-snapshot-save",
        "-vsync-rate", str(frame_rate_hz),
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
    frame_rate_hz: int = GUEST_FRAME_RATE_HZ,
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
        frame_rate_hz=frame_rate_hz,
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
