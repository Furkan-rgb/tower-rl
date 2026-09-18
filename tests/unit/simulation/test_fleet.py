"""Preparing the one snapshot a read-only fleet restores from.

The fleet's own sequencing is exercised end to end where it is composed, in
`tests/unit/test_multi_actor.py`; what is here is the decision made before any
actor starts, with the snapshot registry and every lifecycle step injected.
"""

from __future__ import annotations

import pytest

from tower_rl.simulation import fleet
from tower_rl.simulation.instance import CloneInstance


def prepare(
    monkeypatch: pytest.MonkeyPatch, *, held: bool
) -> tuple[list[str], list[dict[str, object]]]:
    """Run the pre-fleet preparation against an injected snapshot registry."""
    steps: list[str] = []
    asked: list[dict[str, object]] = []

    def fake_bring_up(target: CloneInstance, renderer: str, **keywords: object) -> str:
        steps.append("bring_up")
        asked.append({"serial": target.serial, **keywords})
        return "cold"

    monkeypatch.setattr(fleet, "bridge_key", lambda: "abc123")
    monkeypatch.setattr(fleet, "snapshot_exists", lambda *_: held)
    monkeypatch.setattr(fleet, "bring_up", fake_bring_up)
    monkeypatch.setattr(
        fleet, "tear_down_instance", lambda target: steps.append("tear_down")
    )
    name = fleet.prepare_pinned_snapshot("lavapipe", 4)
    assert name.endswith("abc123")
    return steps, asked


def test_the_pinned_snapshot_is_prepared_once_on_a_writable_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-only actor cannot save one, so the fleet would otherwise stay cold."""
    steps, asked = prepare(monkeypatch, held=False)

    assert steps == ["bring_up", "tear_down"]
    assert asked == [{"serial": "emulator-5556", "deploy": fleet.deploy_bridge, "cores": 4}]


def test_nothing_is_prepared_when_the_snapshot_for_this_bridge_is_already_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert prepare(monkeypatch, held=True) == ([], [])
