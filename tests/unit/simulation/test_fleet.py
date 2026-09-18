"""Bringing a fleet up in turn, putting it down, and pinning its snapshot.

No emulator and no adb: opening an instance and tearing one down are both
injected, so what is read back is only what the primitive did with them — that
no bring-up began before the previous one concluded, that one failure costs one
actor rather than the fleet, and that neither a bridge that will not release nor
an instance that will not stop prevents the rest from being put down.

`stagger_bring_up`, the other half of `fleet`, is exercised where it is
composed: it gates actors that collect concurrently, and driving it means
driving `collect_episodes` and the `run_episodes.py` process behind it, which is
the runner's own composition. Those tests stay in `tests/unit/test_multi_actor.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from tower_rl.environment.run_port import RunPortError
from tower_rl.simulation import fleet
from tower_rl.simulation.fleet import bring_up_fleet, tear_down_fleet
from tower_rl.simulation.instance import CloneInstance


@dataclass(frozen=True)
class Opened:
    """Whatever opening an instance produced.

    `bring_up_fleet` is generic over this: it collects what `open_instance`
    returns and never looks inside it. Training hands it an `ActorInstance`;
    what the primitive owes is the order, the failures and the count.
    """

    serial: str


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


def test_a_fleet_brings_its_instances_up_one_at_a_time() -> None:
    """Four cold boots at once is the one thing the fleet measurement broke on."""
    spans: list[tuple[str, float, float]] = []

    def open_instance(instance: CloneInstance) -> Opened:
        started = time.monotonic()
        time.sleep(0.01)
        spans.append((instance.serial, started, time.monotonic()))
        return Opened(instance.serial)

    instances = [CloneInstance(index=index) for index in range(4)]

    ready, failures = bring_up_fleet(instances, open_instance)

    assert failures == []
    assert [item.serial for item in ready] == [item.serial for item in instances]
    # No bring-up began before the previous one had concluded.
    for (_, _, ended), (_, next_started, _) in zip(spans, spans[1:], strict=False):
        assert next_started >= ended


def test_an_instance_that_will_not_come_up_costs_one_actor() -> None:
    """A failed bring-up must not stall the instances behind it."""

    def open_instance(instance: CloneInstance) -> Opened:
        if instance.index == 1:
            raise RuntimeError("never left main_unavailable")
        return Opened(instance.serial)

    ready, failures = bring_up_fleet(
        [CloneInstance(index=index) for index in range(3)], open_instance
    )

    assert [item.serial for item in ready] == ["emulator-5556", "emulator-5560"]
    assert len(failures) == 1 and "emulator-5558" in failures[0]


class _FailingAdapter:
    """An adapter whose release reads a bridge that has stopped answering."""

    def release(self) -> None:
        raise RunPortError("the bridge could not report the run state")


class _RecordingAdapter:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class _RecordingClient:
    def __init__(self, port: int) -> None:
        self.port = port
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_teardown_continues_past_a_failing_release_and_puts_every_instance_down(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Leaving an emulator running is a safety failure, not an inconvenience.

    The release reads the bridge, so on a client that had stopped answering it
    raised inside the teardown's own `finally` and skipped every later release
    and every instance teardown: four emulators were left running, twice.
    """
    clients = [_RecordingClient(5555 + index) for index in range(2)]
    surviving = _RecordingAdapter()
    opened = [(_FailingAdapter(), clients[0]), (surviving, clients[1])]
    instances = [CloneInstance(index=index) for index in range(3)]
    torn: list[str] = []

    def tear_down(instance: CloneInstance) -> None:
        torn.append(instance.serial)
        if instance.index == 0:
            raise RuntimeError("adb would not stop this one")

    tear_down_fleet(opened, instances, tear_down)  # type: ignore[arg-type]

    # Neither the failing release nor the failing teardown stopped the rest.
    assert surviving.released and all(client.closed for client in clients)
    assert torn == [instance.serial for instance in instances]
    printed = capsys.readouterr().out
    assert "release failed" in printed and "teardown failed" in printed


def test_a_fleet_that_will_not_come_up_at_all_is_refused() -> None:
    def refuse(instance: CloneInstance) -> Opened:
        raise RuntimeError("no snapshot")

    with pytest.raises(SystemExit, match="no instance of the fleet came up"):
        bring_up_fleet([CloneInstance(index=0)], refuse)
