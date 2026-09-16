from __future__ import annotations

from collections.abc import Iterable

import pytest

from tower_rl.application.controller import (
    ControllerConfig,
    ControllerError,
    DeviceFailureError,
    TowerController,
)
from tower_rl.domain import (
    ActionOutcome,
    FieldReading,
    Observation,
    RunAction,
    ScreenState,
    TerminationReason,
)
from tower_rl.ports.android import CapturedFrame, InputReceipt, ScreenPoint


def _reading[T](value: T | None, frame_id: str) -> FieldReading[T]:
    return FieldReading(value, 1.0, frame_id, "test")


def _observation(
    frame_id: str,
    timestamp: float,
    screen: ScreenState,
    *,
    damage_level: int = 0,
    cash: float = 1.0,
) -> Observation:
    purchase_actions = tuple(action for action in RunAction if action is not RunAction.WAIT)
    levels = {action: _reading(0, frame_id) for action in purchase_actions}
    costs = {action: _reading(2.0, frame_id) for action in purchase_actions}
    levels[RunAction.BUY_DAMAGE] = _reading(damage_level, frame_id)
    costs[RunAction.BUY_DAMAGE] = _reading(0.5, frame_id)
    return Observation(
        frame_id=frame_id,
        captured_at_monotonic=timestamp,
        screen=screen,
        wave=_reading(1 if screen is ScreenState.ACTIVE_RUN else None, frame_id),
        cash_normalized=_reading(cash if screen is ScreenState.ACTIVE_RUN else None, frame_id),
        health_fraction=_reading(1.0 if screen is ScreenState.ACTIVE_RUN else None, frame_id),
        max_health_normalized=_reading(1.0, frame_id),
        upgrade_levels=levels,
        upgrade_costs_normalized=costs,
        action_mask=(RunAction.WAIT, RunAction.BUY_DAMAGE)
        if screen is ScreenState.ACTIVE_RUN
        else (RunAction.WAIT,),
        valid=True,
    )


class FakeDevice:
    def __init__(self, frame_count: int = 20, *, foreground: bool = True) -> None:
        self.frames = [
            CapturedFrame(f"frame-{index}", float(index), 1080, 1920, b"fake")
            for index in range(frame_count)
        ]
        self.taps: list[ScreenPoint] = []
        self.foreground = foreground

    def screenshot(self) -> CapturedFrame:
        return self.frames.pop(0)

    def tap(self, point: ScreenPoint) -> InputReceipt:
        self.taps.append(point)
        return InputReceipt(f"tap-{len(self.taps)}", 0.0)

    def app_foreground(self) -> bool:
        return self.foreground


class FakeExtractor:
    def __init__(
        self,
        extracted: Iterable[Observation] = (),
        classified: Iterable[ScreenState] = (),
        account_link_reminders: Iterable[bool] = (),
    ) -> None:
        self.extracted = iter(extracted)
        self.classified = iter(classified)
        self.account_link_reminders = iter(account_link_reminders)

    def extract(self, *_args: object, **_kwargs: object) -> Observation:
        return next(self.extracted)

    def classify(self, _png_bytes: bytes) -> ScreenState:
        return next(self.classified)

    def is_account_link_reminder(self, _png_bytes: bytes) -> bool:
        return next(self.account_link_reminders)


def test_await_death_retries_one_transient_unknown_then_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    device = FakeDevice()
    extractor = FakeExtractor(
        extracted=[_observation("result", 3.0, ScreenState.RESULT)],
        classified=[ScreenState.UNKNOWN, ScreenState.ACTIVE_RUN, ScreenState.RESULT],
    )
    controller = TowerController(device, extractor=extractor)
    monkeypatch.setattr("tower_rl.application.controller.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "tower_rl.application.controller.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    result = controller.await_death(timeout=3.0)

    assert result.screen is ScreenState.RESULT
    assert len(device.taps) == 0
    assert len(device.frames) == 16


def test_await_death_never_taps_a_generic_modal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    device = FakeDevice()
    controller = TowerController(
        device,
        extractor=FakeExtractor(
            extracted=[_observation("result", 2.0, ScreenState.RESULT)],
            classified=[ScreenState.MODAL, ScreenState.RESULT],
        ),
    )
    monkeypatch.setattr("tower_rl.application.controller.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "tower_rl.application.controller.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    result = controller.await_death(timeout=3.0)

    assert result.screen is ScreenState.RESULT
    assert device.taps == []


def test_result_home_action_revalidates_result_before_tapping() -> None:
    device = FakeDevice()
    controller = TowerController(
        device,
        extractor=FakeExtractor(classified=[ScreenState.MODAL]),
    )

    with pytest.raises(ControllerError, match="revalidation failed"):
        controller._tap_result_home()

    assert device.taps == []


def test_result_home_action_uses_profile_target_after_revalidation() -> None:
    device = FakeDevice()
    controller = TowerController(
        device,
        extractor=FakeExtractor(classified=[ScreenState.RESULT]),
    )

    tapped = controller._tap_result_home()

    assert tapped
    assert device.taps == [ScreenPoint(780 / 1079, 1420 / 1919)]


def test_result_home_action_accepts_reached_home_without_input() -> None:
    device = FakeDevice()
    controller = TowerController(
        device,
        extractor=FakeExtractor(classified=[ScreenState.HOME]),
    )

    tapped = controller._tap_result_home()

    assert not tapped
    assert device.taps == []


def test_generic_modal_is_not_dismissed_while_waiting_for_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice()
    controller = TowerController(
        device,
        extractor=FakeExtractor(
            extracted=[_observation("modal", 1.0, ScreenState.MODAL)],
            classified=[ScreenState.MODAL],
            account_link_reminders=[False],
        ),
    )
    monkeypatch.setattr("tower_rl.application.controller.time.sleep", lambda _seconds: None)

    with pytest.raises(ControllerError, match="unsupported modal"):
        controller._wait_for_home()

    assert device.taps == []


def test_account_link_reminder_is_revalidated_then_closed_at_safe_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice()
    controller = TowerController(
        device,
        extractor=FakeExtractor(
            extracted=[
                _observation("reminder", 1.0, ScreenState.MODAL),
                _observation("home", 2.0, ScreenState.HOME),
            ],
            classified=[ScreenState.MODAL],
            account_link_reminders=[True],
        ),
    )
    monkeypatch.setattr("tower_rl.application.controller.time.sleep", lambda _seconds: None)

    result = controller._wait_for_home()

    assert result.screen is ScreenState.HOME
    assert device.taps == [ScreenPoint(884 / 1079, 531 / 1919)]


def test_modal_revalidation_accepts_home_transition_without_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice()
    controller = TowerController(
        device,
        extractor=FakeExtractor(
            extracted=[
                _observation("stale-modal", 1.0, ScreenState.MODAL),
                _observation("home", 2.0, ScreenState.HOME),
            ],
            classified=[ScreenState.HOME],
        ),
    )
    monkeypatch.setattr("tower_rl.application.controller.time.sleep", lambda _seconds: None)

    result = controller._wait_for_home()

    assert result.screen is ScreenState.HOME
    assert device.taps == []


def test_await_death_fails_distinctly_after_persistent_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    device = FakeDevice()
    controller = TowerController(
        device,
        extractor=FakeExtractor(classified=[ScreenState.UNKNOWN, ScreenState.UNKNOWN]),
        config=ControllerConfig(max_consecutive_unknown_classifications=1),
    )
    monkeypatch.setattr("tower_rl.application.controller.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "tower_rl.application.controller.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    with pytest.raises(ControllerError, match="UNKNOWN|unknown"):
        controller.await_death(timeout=3.0)

    assert len(device.taps) == 0
    assert len(device.frames) == 18


def test_await_death_uses_configured_interval_beyond_old_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    sleeps: list[float] = []
    device = FakeDevice()
    controller = TowerController(
        device,
        extractor=FakeExtractor(
            extracted=[_observation("result", 151.0, ScreenState.RESULT)],
            classified=[ScreenState.ACTIVE_RUN] * 5 + [ScreenState.RESULT],
        ),
        config=ControllerConfig(decision_interval_seconds=30.0),
    )
    monkeypatch.setattr("tower_rl.application.controller.time.monotonic", lambda: clock[0])

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("tower_rl.application.controller.time.sleep", advance)

    result = controller.await_death()

    assert result.screen is ScreenState.RESULT
    assert clock[0] == 150.0
    assert sleeps == [30.0] * 5


def test_unknown_with_app_not_foreground_is_a_device_failure() -> None:
    device = FakeDevice(foreground=False)
    controller = TowerController(
        device,
        extractor=FakeExtractor(extracted=[_observation("unknown", 1.0, ScreenState.UNKNOWN)]),
    )

    with pytest.raises(DeviceFailureError) as raised:
        controller.observe()

    assert raised.value.termination_reason is TerminationReason.DEVICE_FAILURE


def test_reset_episode_propagates_device_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice(foreground=False)
    controller = TowerController(
        device,
        extractor=FakeExtractor(
            extracted=[_observation("result", 1.0, ScreenState.RESULT)],
            classified=[ScreenState.UNKNOWN],
        ),
    )
    monkeypatch.setattr("tower_rl.application.controller.time.sleep", lambda _seconds: None)

    with pytest.raises(DeviceFailureError) as raised:
        controller.reset_episode()

    assert raised.value.termination_reason is TerminationReason.DEVICE_FAILURE


def test_end_episode_propagates_device_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice(foreground=False)
    controller = TowerController(
        device,
        extractor=FakeExtractor(
            extracted=[
                _observation("active", 1.0, ScreenState.ACTIVE_RUN),
                _observation("unknown", 2.0, ScreenState.UNKNOWN),
            ]
        ),
    )
    monkeypatch.setattr("tower_rl.application.controller.time.sleep", lambda _seconds: None)

    with pytest.raises(DeviceFailureError) as raised:
        controller.end_episode()

    assert raised.value.termination_reason is TerminationReason.DEVICE_FAILURE


def test_step_returns_result_without_bootstrap_or_active_run_taps() -> None:
    device = FakeDevice()
    result_observations = [
        _observation(f"result-{index}", float(index + 1), ScreenState.RESULT) for index in range(3)
    ]
    controller = TowerController(
        device,
        extractor=FakeExtractor(extracted=result_observations),
    )

    result = controller.step(RunAction.WAIT)

    assert result.observation.screen is ScreenState.RESULT
    assert result.next_observation is None
    assert result.terminated
    assert result.termination_reason is TerminationReason.TOWER_DIED
    assert result.outcome is ActionOutcome.UNAVAILABLE
    assert device.taps == []


def test_purchase_step_keeps_before_and_after_observations_for_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice()
    before = _observation("before", 1.0, ScreenState.ACTIVE_RUN, damage_level=0)
    scan_one = _observation("scan-one", 2.0, ScreenState.ACTIVE_RUN, damage_level=0)
    scan_two = _observation("scan-two", 3.0, ScreenState.ACTIVE_RUN, damage_level=0)
    scan_three = _observation("scan-three", 4.0, ScreenState.ACTIVE_RUN, damage_level=0)
    pre_purchase = _observation("pre-purchase", 5.0, ScreenState.ACTIVE_RUN, damage_level=0)
    after = _observation("after", 6.0, ScreenState.ACTIVE_RUN, damage_level=1)
    extractor = FakeExtractor(
        extracted=[before, scan_one, scan_two, scan_three, pre_purchase, after]
    )
    controller = TowerController(device, extractor=extractor)
    monkeypatch.setattr("tower_rl.application.controller.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("tower_rl.application.controller.time.monotonic", lambda: 10.0)

    result = controller.step(RunAction.BUY_DAMAGE)

    assert result.observation is before
    assert result.next_observation is after
    assert result.outcome is ActionOutcome.EXECUTED
    assert before.upgrade_levels[RunAction.BUY_DAMAGE].value == 0
    assert after.upgrade_levels[RunAction.BUY_DAMAGE].value == 1
