"""What the single-instance CLI owes its caller when a step fails.

A traceback is not a report. Every command here can fail for a reason the
operator has to act on — no bridge build, an AVD with no snapshot, a deploy the
script refused — and the contract is the same for all of them: the reason on
stderr by name, and a non-zero status.

This exists because the fleet and the CLI now share one `deploy_bridge`. It
raises `ActorFailure`, which is not a `CloneError` and deliberately is not made
one: the same name also covers a `run_episodes.py` process that failed, which
says nothing about the clone's state. So `main` names both, and this holds it to
that rather than to a hierarchy it does not have.
"""

from __future__ import annotations

import clone_session
import pytest

from tower_rl.simulation.bridge import ActorFailure
from tower_rl.simulation.instance import CloneError


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ActorFailure("instrumented_bridge.sh deploy failed on emulator-5556"),
            "instrumented_bridge.sh deploy failed",
        ),
        (CloneError("emulator-5556 is online: wlan0"), "is online"),
    ],
    ids=["a refused bridge deploy", "an instance that is still online"],
)
def test_a_failed_bring_up_exits_by_name_rather_than_by_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    expected: str,
) -> None:
    """Both names reach the operator the same way: stderr, and exit 1."""

    def fail(*_: object, **__: object) -> str:
        raise error

    monkeypatch.setattr(clone_session, "bring_up", fail)
    monkeypatch.setattr("sys.argv", ["clone_session.py", "up"])

    status = clone_session.main()

    assert status == 1
    assert expected in capsys.readouterr().err
