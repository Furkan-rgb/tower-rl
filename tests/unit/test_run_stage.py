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
import re
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
echo "$*" >> "$state/adb-calls"
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

#: `timeout` is resolved through `PATH` like every other tool the script runs,
#: so the suite can put one in front of it: this records the real invocation —
#: the production bound is 120s and no test can wait that out — and then applies
#: a bound of its own through the real binary.
TIMEOUT_STUB = """#!/usr/bin/env bash
set -uo pipefail
echo "timeout $*" >> "$TOWER_STUB_STATE/adb-calls"
shift 3
exec /usr/bin/timeout -k 1 3 "$@"
"""

BRIDGE_STUB = """#!/usr/bin/env bash
set -uo pipefail
# A device that has stopped answering, and one that is merely slow.
[ -e "$TOWER_STUB_STATE/hang-$2" ] && sleep 60
[ -e "$TOWER_STUB_STATE/slow-$2" ] && sleep 2
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
    def adb_calls(self) -> str:
        """Every argument list the stub adb and the stub timeout were given."""
        return (self.state / "adb-calls").read_text()

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
    write_executable(binaries / "timeout", TIMEOUT_STUB)

    state = tmp_path / "state"
    state.mkdir()
    (state / "devices").write_text("")
    (state / "adb-calls").write_text("")
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


def bring_up(
    shims: Shims,
    *serials: str,
    read_only: bool = True,
    avd: str | None = None,
    state: str = "device",
) -> None:
    """Put stub instances on the stub host: an adb device and a process each."""
    name = avd or "tower_rl_instrumented_api36"
    devices = shims.state / "devices"
    devices.write_text(devices.read_text() + "".join(f"{serial}\t{state}\n" for serial in serials))
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


def wall_seconds(written: str) -> int:
    """The wall time the summary line reports, in seconds.

    Worth asserting on, because the difference between "the stage was
    interrupted and the teardown followed it" and "the teardown waited out the
    whole grace period against a stage that had already exited" is visible in
    nothing else: every other assertion in these tests passes either way.
    """
    summary = re.search(r"wall (\d+):(\d+):(\d+)", written)
    assert summary is not None, written
    hours, minutes, seconds = (int(part) for part in summary.groups())
    return hours * 3600 + minutes * 60 + seconds


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
    # The stage answered the interrupt; nothing waited out the 10s grace.
    assert wall_seconds(written) < 5, written


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
    assert wall_seconds(written) < 10, written


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


def launch(
    shims: Shims, stage: Path, *, instances: int = 2, grace: int = 10
) -> subprocess.Popen[bytes]:
    """Start a supervisor the test will signal, with its output on a file."""
    sink = (shims.state / "supervisor.out").open("w")
    return subprocess.Popen(
        [str(RUN_STAGE), "--name", "test-stage", "--instances", str(instances),
         "--shutdown-grace", str(grace), "--", str(stage)],
        env=shims.environment, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True,
    )


def test_a_second_signal_during_teardown_does_not_abandon_the_cleanup(
    shims: Shims, tmp_path: Path
) -> None:
    """Teardown ignores what would otherwise leave the device half cleaned."""
    bring_up(shims, "emulator-5556", "emulator-5558")
    (shims.state / "slow-emulator-5556").touch()
    started = tmp_path / "started"
    stage = shims.stage(f'trap "exit 0" INT\ntouch {started}\nsleep 60 &\nwait $!')

    supervisor = launch(shims, stage)
    deadline = time.monotonic() + 30
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert started.exists(), "the stage command never started"
    supervisor.send_signal(signal.SIGTERM)
    # Inside the first instance's cleanup, which the stub holds open for 2s.
    time.sleep(1.0)
    supervisor.send_signal(signal.SIGTERM)
    returncode = supervisor.wait(timeout=60)

    written = (shims.state / "supervisor.out").read_text()
    assert returncode != 0, written
    assert shims.cleaned == ["emulator-5556", "emulator-5558"], written
    assert "verified: no qemu process, no adb device" in written
    assert "stage test-stage: exit 143, cleanup ok, instances 2/2 cleaned" in written


def test_a_cleanup_that_never_returns_is_given_up_on_and_the_rest_still_runs(
    shims: Shims,
) -> None:
    bring_up(shims, "emulator-5556", "emulator-5558")
    (shims.state / "hang-emulator-5556").touch()
    stage = shims.stage("echo collecting")

    result = run_stage(shims, str(stage))

    assert result.returncode != 0
    assert "cleanup: emulator-5556 did not finish cleaning up within 120s" in result.stdout
    # The bound the supervisor actually asked for, whatever the suite shortened
    # it to: a TERM at 120s and a KILL ten seconds after that.
    assert "timeout -k 10 120" in shims.adb_calls
    # The instance that hung is still killed, the next one is still cleaned,
    # and the host is still verified.
    assert "emu kill" in shims.adb_calls
    assert shims.cleaned == ["emulator-5558"]
    assert "verified: no qemu process, no adb device" in result.stdout
    assert "stage test-stage: exit 0, cleanup failed, instances 1/2 cleaned" in result.stdout


def test_an_attached_instance_that_is_not_answering_is_refused_before_launch(
    shims: Shims,
) -> None:
    """Offline cannot be verified on an instance that will not answer."""
    bring_up(shims, "emulator-5556", state="offline")
    stage = shims.stage("touch " + str(shims.state / "ran"))

    result = run_stage(shims, str(stage), instances=1)

    assert result.returncode == 2
    assert "attached in state 'offline'" in result.stdout + result.stderr
    assert not (shims.state / "ran").exists()


def test_an_emulator_running_the_canonical_avd_is_never_killed(shims: Shims) -> None:
    """It appears only once the stage is running, so preflight cannot refuse it."""
    bring_up(shims, "emulator-5556")
    proc = Path(shims.environment["TOWER_STAGE_PROC_ROOT"])
    stage = shims.stage(
        f'printf "emulator-5558\\tdevice\\n" >> {shims.state / "devices"}\n'
        f'mkdir -p {proc / "5558"}\n'
        # `printf '%s\0'` rather than escapes inside one string: `\05558` reads
        # as an octal escape and silently becomes something else.
        f"printf '%s\\0' /opt/emulator @tower_rl_api36_play_x86_64 -port 5558"
        f' > {proc / "5558" / "cmdline"}\n'
    )

    result = run_stage(shims, str(stage), instances=1)

    assert result.returncode != 0
    assert "emulator-5558 is running the canonical evaluation AVD" in result.stdout
    assert "was left untouched" in result.stdout
    assert "-s emulator-5558 emu kill" not in shims.adb_calls
    assert "stage test-stage: exit 0, cleanup failed, instances 1/1 cleaned" in result.stdout


@pytest.mark.parametrize("grace", ["0", "-5", "forever"], ids=["zero", "negative", "words"])
def test_a_grace_period_that_is_not_a_positive_number_of_seconds_is_refused(
    shims: Shims, grace: str
) -> None:
    stage = shims.stage("touch " + str(shims.state / "ran"))

    result = subprocess.run(
        [str(RUN_STAGE), "--name", "test-stage", "--instances", "1",
         "--shutdown-grace", grace, "--", str(stage)],
        env=shims.environment, capture_output=True, text=True, timeout=60, check=False,
    )

    assert result.returncode == 2
    assert "--shutdown-grace must be" in result.stderr
    assert not (shims.state / "ran").exists()
