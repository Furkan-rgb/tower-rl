"""What the one call into `instrumented_bridge.sh` relays and what it refuses.

The script itself is not run: what is under test is that its output reaches the
operator's log tagged with the instance it came from — cleanup that reports
nothing is indistinguishable from cleanup that verified nothing — and that a
non-zero status becomes one actor's failure.
"""

from __future__ import annotations

import subprocess

import pytest

from tower_rl.simulation.bridge import ActorFailure, run_bridge
from tower_rl.simulation.instance import CloneInstance


def bridge_output(
    monkeypatch: pytest.MonkeyPatch, *, stdout: str, stderr: str = "", status: int = 0
) -> None:
    def script(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, status, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", script)


def test_the_cleanup_identity_report_reaches_the_operators_log(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cleanup verification is a device-safety property, so it must be visible."""
    bridge_output(
        monkeypatch,
        stdout=(
            "libunity_sha256: ffc1f3ef\nversionCode=1199\n"
            "libunity_mounts: 0\nbridge_artifacts: removed\n"
        ),
    )

    run_bridge("cleanup", CloneInstance(index=1))

    printed = capsys.readouterr().out
    assert "emulator-5558 cleanup: libunity_sha256: ffc1f3ef" in printed
    assert "emulator-5558 cleanup: libunity_mounts: 0" in printed
    assert "emulator-5558 cleanup: bridge_artifacts: removed" in printed


def test_a_failing_bridge_step_reports_its_output_as_well_as_failing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bridge_output(
        monkeypatch, stdout="libunity_mounts: 1\n", stderr="warning: still mounted\n", status=1
    )

    with pytest.raises(ActorFailure, match="still mounted"):
        run_bridge("cleanup", CloneInstance())

    printed = capsys.readouterr().out
    assert "emulator-5556 cleanup: libunity_mounts: 1" in printed
    assert "emulator-5556 cleanup: error: warning: still mounted" in printed
