"""Which bridge is deployed, and what the one call into the script relays.

The script itself is not run: what is under test is that its output reaches the
operator's log tagged with the instance it came from — cleanup that reports
nothing is indistinguishable from cleanup that verified nothing — that a
non-zero status becomes one actor's failure, and that the artifact it deploys is
resolved from a durable home which refuses by name rather than deploying
whatever a stale pointer happens to reach.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from tower_rl.simulation import bridge
from tower_rl.simulation.bridge import ActorFailure, run_bridge
from tower_rl.simulation.instance import CloneError, CloneInstance


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


def install_bridge(root: Path, contents: bytes = b"bridge") -> str:
    """One installed bridge under its own digest, with `current` pointing at it."""
    digest = hashlib.sha256(contents).hexdigest()
    (root / digest).mkdir(parents=True)
    (root / digest / "libtower_bridge.so").write_bytes(contents)
    (root / "current").symlink_to(digest)
    return digest


def test_the_installed_bridge_is_the_one_current_points_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)
    digest = install_bridge(tmp_path)

    assert bridge.installed_bridge_directory() == tmp_path / digest


def test_no_installed_bridge_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)

    with pytest.raises(CloneError, match="no bridge is installed"):
        bridge.installed_bridge_directory()


def test_a_dangling_pointer_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The failure a /tmp build directory caused after a reboot, now named."""
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)
    (tmp_path / "current").symlink_to(tmp_path / "gone-with-the-scratchpad")

    with pytest.raises(CloneError, match="dangles"):
        bridge.installed_bridge_directory()


def test_an_artifact_that_does_not_match_its_recorded_digest_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The directory name is the claim; the bytes are the evidence."""
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)
    digest = install_bridge(tmp_path)
    (tmp_path / digest / "libtower_bridge.so").write_bytes(b"something else")

    with pytest.raises(CloneError, match="not to the"):
        bridge.installed_bridge_directory()


def test_an_explicit_build_directory_still_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bridge under development is deployed straight out of its build tree."""
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)
    install_bridge(tmp_path)
    monkeypatch.setenv("TOWER_BRIDGE_BUILD_DIR", "/elsewhere/build")

    assert bridge.bridge_build_directory() == Path("/elsewhere/build")


def test_a_missing_deployment_script_fails_by_name_not_as_a_subprocess_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`BRIDGE_SCRIPT` is derived from this package's location and can miss."""
    monkeypatch.setattr(bridge, "BRIDGE_SCRIPT", tmp_path / "instrumented_bridge.sh")

    with pytest.raises(ActorFailure, match="bridge deployment script is missing"):
        run_bridge("deploy", CloneInstance())


def device_digest(monkeypatch: pytest.MonkeyPatch, output: str) -> list[str]:
    """Whatever the device says to the one read-back, and what was asked of it."""
    asked: list[str] = []

    def shell(instance: CloneInstance, *args: str, **_: object) -> str:
        asked.extend(args)
        return output

    monkeypatch.setattr(bridge, "adb", shell)
    return asked


def test_the_read_back_asks_for_root_the_way_that_answers_on_this_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`su 0 sha256sum` returned nothing on 14 instances; `su -c` is the form."""
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)
    digest = install_bridge(tmp_path)
    asked = device_digest(monkeypatch, f"{digest}  {bridge.DEPLOYED_BRIDGE_PATH}\n")

    assert bridge.confirm_deployed_bridge(CloneInstance()) == digest
    assert asked == ["shell", f"su -c 'sha256sum {bridge.DEPLOYED_BRIDGE_PATH}'"]


def test_a_confirmed_bridge_is_named_in_the_operators_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)
    digest = install_bridge(tmp_path)
    device_digest(monkeypatch, f"{digest}  {bridge.DEPLOYED_BRIDGE_PATH}\r\n")

    bridge.confirm_deployed_bridge(CloneInstance(index=1))

    assert f"emulator-5558 deploy: deployed bridge confirmed {digest}" in capsys.readouterr().out


def test_a_read_back_that_produced_nothing_fails_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two seven-actor fleets recorded "unconfirmed"; it is now a failure."""
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)
    install_bridge(tmp_path)
    device_digest(monkeypatch, "")

    with pytest.raises(ActorFailure, match="no digest came back"):
        bridge.confirm_deployed_bridge(CloneInstance())


def test_sha256sums_own_error_text_is_not_read_as_a_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The `M1B-E047` misfire: the first word of a failure is not a reading."""
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)
    install_bridge(tmp_path)
    device_digest(
        monkeypatch, f"sha256sum: {bridge.DEPLOYED_BRIDGE_PATH}: No such file or directory\n"
    )

    with pytest.raises(ActorFailure, match="no digest came back"):
        bridge.confirm_deployed_bridge(CloneInstance())


def test_a_bridge_that_is_not_the_one_this_host_deployed_fails_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)
    install_bridge(tmp_path)
    device_digest(monkeypatch, f"{'a' * 64}  {bridge.DEPLOYED_BRIDGE_PATH}\n")

    with pytest.raises(ActorFailure, match="the deployed bridge is aaa"):
        bridge.confirm_deployed_bridge(CloneInstance())


def test_deploying_reads_the_bridge_back_before_it_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One read-back, on the one path the CLI and the fleet both deploy through."""
    monkeypatch.setattr(bridge, "BRIDGE_STATE_DIRECTORY", tmp_path)
    install_bridge(tmp_path)
    bridge_output(monkeypatch, stdout="deployed: overlay mounted\n")
    device_digest(monkeypatch, "")

    with pytest.raises(ActorFailure, match="no digest came back"):
        bridge.deploy_bridge(CloneInstance())

    assert "emulator-5556 deploy: deployed: overlay mounted" in capsys.readouterr().out
