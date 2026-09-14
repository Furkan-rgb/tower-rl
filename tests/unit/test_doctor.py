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
