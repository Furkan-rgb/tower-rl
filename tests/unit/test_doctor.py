from __future__ import annotations

import json

from tower_rl.doctor import CheckResult, CheckStatus, render_json


def test_render_json_reports_failure() -> None:
    report = render_json(
        [CheckResult("example", CheckStatus.FAIL, "failed", {"reason": "test"})]
    )

    payload = json.loads(report)
    assert payload["ok"] is False
    assert payload["checks"][0]["status"] == "fail"


def test_sdk_discovery_covers_the_linux_workstation_location(monkeypatch, tmp_path) -> None:
    """An unattended run does not inherit an interactive PATH."""
    from pathlib import Path

    from tower_rl.doctor import find_android_tool

    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    adb = tmp_path / ".local/share/android-sdk/platform-tools/adb"
    adb.parent.mkdir(parents=True)
    adb.write_text("#!/bin/sh\n")

    assert find_android_tool("adb") == adb.resolve()
