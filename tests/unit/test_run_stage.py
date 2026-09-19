"""What `scripts/run_stage.sh` guarantees on every path out of a stage.

A stage is hours of device time, so the thing worth holding is not that the
happy path works: it is that the teardown happens anyway when the stage fails,
when cleanup itself fails, and when the supervisor is killed mid-stage. None of
that can wait for a device, so the whole environment the script reaches for is
stubbed here — `adb` and `instrumented_bridge.sh` in front of it on `PATH`, and
a `/proc` of the suite's own that the stub emulators appear in and disappear
from as they are killed.

The stubs are deliberately stateful: killing an instance removes both its adb
device line and its process, so the script's final verification reads a host
that really did empty out rather than one that was asserted to have.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
RUN_STAGE = REPOSITORY / "scripts" / "run_stage.sh"

ADB_STUB = """#!/usr/bin/env bash
set -uo pipefail
state="$TOWER_STUB_STATE"
if [ "$1" = "devices" ]; then
  echo "List of devices attached"
  cat "$state/devices" 2>/dev/null || true
  exit 0
fi
serial=""
if [ "$1" = "-s" ]; then serial="$2"; shift 2; fi
case "$*" in
  "emu kill")
    grep -v "^$serial	" "$state/devices" > "$state/devices.next" || true
    mv "$state/devices.next" "$state/devices"
    rm -rf "$TOWER_STAGE_PROC_ROOT/${serial#emulator-}"
    ;;
  "shell ip -o -4 addr show")
    if [ -e "$state/online-$serial" ]; then
      echo "2: eth0    inet 10.0.2.15/24 scope global eth0"
    fi
    echo "1: lo    inet 127.0.0.1/8 scope host lo"
    ;;
  *) echo "stub adb: unexpected $*" >&2; exit 1 ;;
esac
"""

BRIDGE_STUB = """#!/usr/bin/env bash
set -uo pipefail
echo "cleanup: $2 bridge_artifacts: removed"
echo "$2" >> "$TOWER_STUB_STATE/cleaned"
exit "${TOWER_STUB_CLEANUP_EXIT:-0}"
"""


@dataclass(frozen=True)
class Shims:
    """The world one run of the script sees: its stubs, its /proc, its log."""

    environment: dict[str, str]
    state: Path
    logs: Path

    @property
    def cleaned(self) -> list[str]:
        """Which serials the stub bridge was asked to clean, in order."""
        record = self.state / "cleaned"
        return record.read_text().split() if record.exists() else []

    def stage(self, script: str, name: str = "stage.sh") -> Path:
        """A stage command of the suite's own, to be supervised."""
        path = self.state / name
        path.write_text(f"#!/usr/bin/env bash\n{script}\n")
        path.chmod(0o755)
        return path


def write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


@pytest.fixture
def shims(tmp_path: Path) -> Shims:
    """A stubbed host with no instance running on it."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    write_executable(binaries / "adb", ADB_STUB)
    write_executable(binaries / "instrumented_bridge.sh", BRIDGE_STUB)

    state = tmp_path / "state"
    state.mkdir()
    (state / "devices").write_text("")
    proc = tmp_path / "proc"
    proc.mkdir()
    logs = tmp_path / "logs"

    environment = dict(os.environ)
    environment.update(
        PATH=f"{binaries}{os.pathsep}{environment['PATH']}",
        TOWER_STUB_STATE=str(state),
        TOWER_STAGE_PROC_ROOT=str(proc),
        TOWER_STAGE_LOG_DIRECTORY=str(logs),
    )
    return Shims(environment=environment, state=state, logs=logs)


def bring_up(shims: Shims, *serials: str, read_only: bool = True, avd: str | None = None) -> None:
    """Put stub instances on the stub host: an adb device and a process each."""
    name = avd or "tower_rl_instrumented_api36"
    devices = shims.state / "devices"
    devices.write_text(devices.read_text() + "".join(f"{serial}\tdevice\n" for serial in serials))
    proc = Path(shims.environment["TOWER_STAGE_PROC_ROOT"])
    for serial in serials:
        port = serial.removeprefix("emulator-")
        # The pid is the console port: the stub only needs them to agree with
        # each other, and `adb emu kill` removes the process this way.
        entry = proc / port
        entry.mkdir()
        arguments = ["/opt/android/emulator/emulator", f"@{name}", "-gpu", "host", "-no-window",
                     "-port", port, *(["-read-only"] if read_only else []), "-no-snapshot-load"]
        (entry / "cmdline").write_bytes(b"\0".join(a.encode() for a in arguments) + b"\0")
        (entry / "exe").symlink_to("/opt/android/emulator/qemu/linux-x86_64/qemu-system-x86_64")


def run_stage(
    shims: Shims, *arguments: str, instances: int = 2
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RUN_STAGE), "--name", "test-stage", "--instances", str(instances),
         "--shutdown-grace", "10", "--", *arguments],
        env=shims.environment, capture_output=True, text=True, timeout=90, check=False,
    )


def log_text(shims: Shims) -> str:
    logs = sorted(shims.logs.glob("test-stage-*.log"))
    assert logs, "the stage wrote no log"
    return logs[-1].read_text()


def test_a_stage_that_succeeds_is_cleaned_up_per_instance(shims: Shims) -> None:
    bring_up(shims, "emulator-5556", "emulator-5558")
    stage = shims.stage("echo collecting")

    result = run_stage(shims, str(stage))

    assert result.returncode == 0, result.stdout + result.stderr
    assert shims.cleaned == ["emulator-5556", "emulator-5558"]
    assert "verified: no qemu process, no adb device" in result.stdout
    summary = "stage test-stage: exit 0, cleanup ok, instances 2/2 cleaned"
    assert summary in result.stdout
    assert summary in log_text(shims)


def test_a_stage_that_fails_is_still_cleaned_up(shims: Shims) -> None:
    bring_up(shims, "emulator-5556", "emulator-5558")
    stage = shims.stage("echo collapsing >&2; exit 3")

    result = run_stage(shims, str(stage))

    assert result.returncode == 3
    assert shims.cleaned == ["emulator-5556", "emulator-5558"]
    assert "stage test-stage: exit 3, cleanup ok, instances 2/2 cleaned" in result.stdout


def test_a_cleanup_that_fails_fails_the_stage(shims: Shims) -> None:
    bring_up(shims, "emulator-5556", "emulator-5558")
    shims.environment["TOWER_STUB_CLEANUP_EXIT"] = "1"
    stage = shims.stage("echo collecting")

    result = run_stage(shims, str(stage))

    assert result.returncode != 0
    # The instance is killed even though its cleanup refused, and the next
    # instance is cleaned rather than skipped.
    assert shims.cleaned == ["emulator-5556", "emulator-5558"]
    assert "stage test-stage: exit 0, cleanup failed, instances 0/2 cleaned" in result.stdout


def test_an_instance_that_is_not_live_is_reported_rather_than_cleaned(shims: Shims) -> None:
    bring_up(shims, "emulator-5556")
    stage = shims.stage("echo collecting")

    result = run_stage(shims, str(stage))

    assert result.returncode == 0, result.stdout + result.stderr
    assert shims.cleaned == ["emulator-5556"]
    assert "cleanup: emulator-5558 is not live; not cleaned" in result.stdout
    assert "stage test-stage: exit 0, cleanup ok, instances 1/2 cleaned" in result.stdout


def test_a_stage_killed_mid_run_is_still_cleaned_up(shims: Shims, tmp_path: Path) -> None:
    bring_up(shims, "emulator-5556", "emulator-5558")
    started = tmp_path / "started"
    stage = shims.stage(
        f'trap "exit 0" INT TERM\ntouch {started}\nsleep 60 &\nwait $!'
    )
    output = tmp_path / "supervisor.out"

    with output.open("w") as sink:
        supervisor = subprocess.Popen(
            [str(RUN_STAGE), "--name", "test-stage", "--instances", "2",
             "--shutdown-grace", "10", "--", str(stage)],
            env=shims.environment, stdout=sink, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 30
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert started.exists(), "the stage command never started"
        supervisor.send_signal(signal.SIGTERM)
        returncode = supervisor.wait(timeout=60)

    written = output.read_text()
    assert returncode != 0, written
    assert shims.cleaned == ["emulator-5556", "emulator-5558"], written
    assert "stage test-stage: exit 143, cleanup ok, instances 2/2 cleaned" in written


def test_a_stage_whose_work_is_a_child_process_is_still_interrupted(
    shims: Shims, tmp_path: Path
) -> None:
    """The real stage is `uv run … python train.py`: the work is a grandchild.

    Measured on this host, `uv` neither acts on SIGINT nor passes it on, so a
    signal sent to the process this script started reaches nothing that can tear
    a fleet down. The shape is modelled here — a leader that ignores the signal
    and a child that acts on it — and what proves the signal arrived is that the
    stage ends well inside its grace period rather than being killed at the end
    of one.
    """
    bring_up(shims, "emulator-5556", "emulator-5558")
    started = tmp_path / "started"
    interrupted = tmp_path / "interrupted"
    child = shims.stage(
        f'trap "touch {interrupted}; exit 0" INT\ntouch {started}\nsleep 60 &\nwait $!',
        name="child.sh",
    )
    # `trap ":"` rather than `trap ""`: a shell that *ignores* a signal hands
    # SIG_IGN to its children, which a child cannot then trap - and `uv` does
    # not ignore SIGINT, it simply does not act on it or pass it on.
    stage = shims.stage(f'trap ":" INT\n{child}')
    output = tmp_path / "supervisor.out"

    with output.open("w") as sink:
        supervisor = subprocess.Popen(
            [str(RUN_STAGE), "--name", "test-stage", "--instances", "2",
             "--shutdown-grace", "30", "--", str(stage)],
            env=shims.environment, stdout=sink, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 30
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert started.exists(), "the stage command never started"
        supervisor.send_signal(signal.SIGTERM)
        returncode = supervisor.wait(timeout=60)

    written = output.read_text()
    assert interrupted.exists(), written
    assert returncode != 0, written
    assert "did not exit within" not in written
    assert "stage test-stage: exit 143, cleanup ok, instances 2/2 cleaned" in written


def test_an_instance_that_is_not_read_only_is_refused_before_launch(shims: Shims) -> None:
    bring_up(shims, "emulator-5556", read_only=False)
    stage = shims.stage("touch " + str(shims.state / "ran"))

    result = run_stage(shims, str(stage), instances=1)

    assert result.returncode == 2
    assert "was not launched -read-only" in result.stdout + result.stderr
    assert not (shims.state / "ran").exists()
    assert shims.cleaned == []


def test_an_instance_that_is_still_online_is_refused_before_launch(shims: Shims) -> None:
    bring_up(shims, "emulator-5556")
    (shims.state / "online-emulator-5556").touch()
    stage = shims.stage("touch " + str(shims.state / "ran"))

    result = run_stage(shims, str(stage), instances=1)

    assert result.returncode == 2
    assert "still has a routable interface" in result.stdout + result.stderr
    assert not (shims.state / "ran").exists()


def test_the_canonical_evaluation_avd_is_refused_before_launch(shims: Shims) -> None:
    bring_up(shims, "emulator-5556", avd="tower_rl_api36_play_x86_64")
    stage = shims.stage("touch " + str(shims.state / "ran"))

    result = run_stage(shims, str(stage), instances=1)

    assert result.returncode == 2
    assert "canonical evaluation AVD" in result.stdout + result.stderr
    assert not (shims.state / "ran").exists()
