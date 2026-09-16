"""Single-device Tier-1 controller and environment-facing action boundary."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from tower_rl.domain import (
    ActionOutcome,
    Observation,
    ObservationValidator,
    RunAction,
    ScreenState,
    StepResult,
    TerminationReason,
)
from tower_rl.domain.contracts import FieldReading
from tower_rl.ports.android import AndroidDevice, ScreenPoint
from tower_rl.vision import PROFILE, ObservationExtractor


@dataclass(frozen=True)
class ControllerConfig:
    decision_interval_seconds: float = 1.0
    action_timeout_seconds: float = 8.0
    max_recovery_attempts: int = 2
    max_consecutive_unknown_classifications: int = 1


class ControllerError(RuntimeError):
    """Raised when the controller cannot establish a valid game state."""


class DeviceFailureError(ControllerError):
    """Raised when the Android device can no longer present the game."""

    termination_reason = TerminationReason.DEVICE_FAILURE


class TowerController:
    """Maps semantic run actions to verified profile-owned UI interactions."""

    def __init__(
        self,
        device: AndroidDevice,
        *,
        extractor: ObservationExtractor | None = None,
        config: ControllerConfig | None = None,
    ) -> None:
        self.device = device
        self.extractor = extractor or ObservationExtractor()
        self.config = config or ControllerConfig()
        self.validator = ObservationValidator()
        self.active_tab = "attack"
        self.previous: Observation | None = None
        self._levels: dict[RunAction, FieldReading[int]] = {}
        self._costs: dict[RunAction, FieldReading[float]] = {}

    def observe(self) -> Observation:
        frame = self.device.screenshot()
        observation = self.extractor.extract(
            frame.png_bytes,
            frame_id=frame.frame_id,
            captured_at_monotonic=frame.captured_at_monotonic,
            active_tab=self.active_tab,
        )
        if observation.screen is ScreenState.UNKNOWN:
            self._raise_if_app_not_foreground()
        result = self.validator.validate(observation, self.previous)
        if not result.valid:
            observation = replace(observation, valid=False, invalid_reasons=result.reasons)
        self.previous = observation
        if observation.valid and observation.screen is ScreenState.ACTIVE_RUN:
            self._levels.update(
                {k: v for k, v in observation.upgrade_levels.items() if v.value is not None}
            )
            self._costs.update(
                {
                    k: v
                    for k, v in observation.upgrade_costs_normalized.items()
                    if v.value is not None
                }
            )
        return observation

    def _raise_if_app_not_foreground(self) -> None:
        app_foreground = getattr(self.device, "app_foreground", None)
        if app_foreground is not None and not app_foreground():
            raise DeviceFailureError(
                "device_failure: game app left the foreground or its process stopped"
            )

    def _tap(self, x: int, y: int) -> None:
        self.device.tap(ScreenPoint(x / 1079, y / 1919))

    def _tap_result_home(self) -> bool:
        """Take the stable Game Stats HOME action after immediate revalidation."""
        frame = self.device.screenshot()
        screen = self.extractor.classify(frame.png_bytes)
        if screen is ScreenState.UNKNOWN:
            self._raise_if_app_not_foreground()
        if screen is ScreenState.HOME:
            return False
        if screen is not ScreenState.RESULT:
            raise ControllerError(
                "result Home action revalidation failed: "
                f"expected={ScreenState.RESULT.value}; actual={screen.value}"
            )
        self._tap(*PROFILE.result_home_point)
        return True

    def _dismiss_account_link_reminder(self) -> bool:
        """Dismiss the known reminder only after immediate subtype revalidation."""
        frame = self.device.screenshot()
        screen = self.extractor.classify(frame.png_bytes)
        if screen is ScreenState.UNKNOWN:
            self._raise_if_app_not_foreground()
        if screen is ScreenState.HOME:
            return False
        if screen is not ScreenState.MODAL or not self.extractor.is_account_link_reminder(
            frame.png_bytes
        ):
            raise ControllerError(
                "unsupported modal while waiting for Battle home; no input sent"
            )
        self._tap(*PROFILE.account_link_reminder_close_point)
        return True

    def _wait_for_home(self, timeout: float = 12.0) -> Observation:
        """Wait for Battle home, dismissing only its known account-link reminder."""
        deadline = time.monotonic() + timeout
        last: Observation | None = None
        while time.monotonic() < deadline:
            last = self.observe()
            if last.valid and last.screen is ScreenState.HOME:
                return last
            if last.valid and last.screen is ScreenState.MODAL:
                self._dismiss_account_link_reminder()
            time.sleep(0.25)
        raise ControllerError(
            "timed out waiting for Battle home; "
            f"last={last.screen.value if last else 'none'}; "
            f"invalid_reasons={last.invalid_reasons if last else ()}"
        )

    def _wait_for(self, screen: ScreenState, timeout: float | None = None) -> Observation:
        deadline = time.monotonic() + (timeout or self.config.action_timeout_seconds)
        last: Observation | None = None
        while time.monotonic() < deadline:
            last = self.observe()
            if last.valid and last.screen is screen:
                return last
            time.sleep(0.25)
        raise ControllerError(
            f"timed out waiting for {screen.value}; "
            f"last={last.screen.value if last else 'none'}; "
            f"invalid_reasons={last.invalid_reasons if last else ()}"
        )

    def _scan_tab(self, tab: str) -> Observation:
        if self.active_tab != tab:
            x, y = PROFILE.tab_points[tab]
            self._tap(x, y)
            self.active_tab = tab
            time.sleep(0.25)
        return self.observe()

    def _bootstrap_scan(self) -> Observation:
        self._scan_tab("attack")
        self._scan_tab("defense")
        observation = self._scan_tab("attack")
        if any(
            self._levels.get(action) is None
            for action in (
                RunAction.BUY_DAMAGE,
                RunAction.BUY_ATTACK_SPEED,
                RunAction.BUY_CRITICAL_CHANCE,
                RunAction.BUY_CRITICAL_FACTOR,
            )
        ):
            observation = self._scan_tab("attack")
        return self._merge_cached(observation)

    def _merge_cached(self, observation: Observation) -> Observation:
        mask = [RunAction.WAIT]
        if observation.valid and observation.screen is ScreenState.ACTIVE_RUN:
            for action in RunAction:
                if action is RunAction.WAIT:
                    continue
                level = self._levels.get(action)
                cost = self._costs.get(action)
                cash = observation.cash_normalized.value
                cost_value = cost.value if cost is not None else None
                if level is not None and cost_value is not None and (
                    cash is None or cost_value <= cash
                ):
                    mask.append(action)
        return replace(
            observation,
            upgrade_levels=dict(self._levels),
            upgrade_costs_normalized=dict(self._costs),
            action_mask=tuple(mask),
        )

    def start_episode(self, *, scan_upgrades: bool = True) -> Observation:
        observation = self.observe()
        if not observation.valid:
            raise ControllerError(f"invalid start observation: {observation.invalid_reasons}")
        if observation.screen is ScreenState.HOME:
            self._tap(540, 1550)
            observation = self._wait_for(ScreenState.ACTIVE_RUN)
        elif observation.screen is not ScreenState.ACTIVE_RUN:
            raise ControllerError(f"cannot start from {observation.screen.value}")
        self.previous = None
        if scan_upgrades:
            self._levels.clear()
            self._costs.clear()
        return self._bootstrap_scan() if scan_upgrades else self._merge_cached(observation)

    def step(self, action: RunAction) -> StepResult:
        started = time.monotonic()
        observation = self.observe()
        if not observation.valid:
            return StepResult(
                observation,
                None,
                action,
                observation.action_mask,
                ActionOutcome.INVALID_OBSERVATION,
                0.0,
                False,
                True,
                TerminationReason.INVALID_OBSERVATION,
                time.monotonic() - started,
            )
        if observation.screen is ScreenState.RESULT:
            return StepResult(
                observation,
                None,
                action,
                observation.action_mask,
                ActionOutcome.UNAVAILABLE,
                0.0,
                True,
                False,
                TerminationReason.TOWER_DIED,
                time.monotonic() - started,
            )
        if observation.screen is not ScreenState.ACTIVE_RUN:
            return StepResult(
                observation,
                None,
                action,
                observation.action_mask,
                ActionOutcome.NAVIGATION_FAILED,
                0.0,
                False,
                True,
                TerminationReason.NAVIGATION_FAILURE,
                time.monotonic() - started,
            )

        pre_action_observation = observation
        scanned_observation = self._bootstrap_scan()
        if not scanned_observation.valid:
            return StepResult(
                pre_action_observation,
                None,
                action,
                pre_action_observation.action_mask,
                ActionOutcome.INVALID_OBSERVATION,
                0.0,
                False,
                True,
                TerminationReason.INVALID_OBSERVATION,
                time.monotonic() - started,
            )
        if scanned_observation.screen is ScreenState.RESULT:
            return StepResult(
                pre_action_observation,
                None,
                action,
                pre_action_observation.action_mask,
                ActionOutcome.UNAVAILABLE,
                0.0,
                True,
                False,
                TerminationReason.TOWER_DIED,
                time.monotonic() - started,
            )
        if scanned_observation.screen is not ScreenState.ACTIVE_RUN:
            return StepResult(
                pre_action_observation,
                None,
                action,
                pre_action_observation.action_mask,
                ActionOutcome.NAVIGATION_FAILED,
                0.0,
                False,
                True,
                TerminationReason.NAVIGATION_FAILURE,
                time.monotonic() - started,
            )
        if action is not RunAction.WAIT and action not in scanned_observation.action_mask:
            return StepResult(
                pre_action_observation,
                pre_action_observation,
                action,
                pre_action_observation.action_mask,
                ActionOutcome.UNAVAILABLE,
                0.0,
                False,
                False,
                None,
                time.monotonic() - started,
            )
        if action is RunAction.WAIT:
            time.sleep(self.config.decision_interval_seconds)
            outcome = ActionOutcome.EXECUTED
            next_observation = self._merge_cached(self.observe())
        else:
            target = PROFILE.action_targets[action]
            before_purchase = self._scan_tab(target.tab)
            if not before_purchase.valid:
                return StepResult(
                    pre_action_observation,
                    before_purchase,
                    action,
                    pre_action_observation.action_mask,
                    ActionOutcome.INVALID_OBSERVATION,
                    0.0,
                    False,
                    True,
                    TerminationReason.INVALID_OBSERVATION,
                    time.monotonic() - started,
                )
            if before_purchase.screen is ScreenState.RESULT:
                return StepResult(
                    pre_action_observation,
                    before_purchase,
                    action,
                    pre_action_observation.action_mask,
                    ActionOutcome.UNAVAILABLE,
                    0.0,
                    True,
                    False,
                    TerminationReason.TOWER_DIED,
                    time.monotonic() - started,
                )
            if before_purchase.screen is not ScreenState.ACTIVE_RUN:
                return StepResult(
                    pre_action_observation,
                    before_purchase,
                    action,
                    pre_action_observation.action_mask,
                    ActionOutcome.NAVIGATION_FAILED,
                    0.0,
                    False,
                    True,
                    TerminationReason.NAVIGATION_FAILURE,
                    time.monotonic() - started,
                )
            before_level = before_purchase.upgrade_levels.get(action)
            before_cash = before_purchase.cash_normalized.value
            self._tap(
                (target.tap_region.left + target.tap_region.right) // 2,
                (target.tap_region.top + target.tap_region.bottom) // 2,
            )
            time.sleep(0.25)
            next_observation = self._scan_tab(target.tab)
            after_level = next_observation.upgrade_levels.get(action)
            after_cash = next_observation.cash_normalized.value
            outcome = (
                ActionOutcome.EXECUTED
                if (
                    before_level is not None
                    and after_level is not None
                    and after_level.value is not None
                    and before_level.value is not None
                    and after_level.value > before_level.value
                )
                or (before_cash is not None and after_cash is not None and after_cash < before_cash)
                else ActionOutcome.AMBIGUOUS
            )
        terminated = next_observation.screen is ScreenState.RESULT
        reason = TerminationReason.TOWER_DIED if terminated else None
        reward = (
            float((next_observation.wave.value or 0) - (pre_action_observation.wave.value or 0))
            if (
                next_observation.wave.value is not None
                and pre_action_observation.wave.value is not None
            )
            else 0.0
        )
        return StepResult(
            pre_action_observation,
            next_observation,
            action,
            pre_action_observation.action_mask,
            outcome,
            reward,
            terminated,
            False,
            reason,
            time.monotonic() - started,
        )

    def recover(self, snapshot_name: str) -> None:
        self.device.restore_baseline(snapshot_name)
        self.previous = None
        self.active_tab = "attack"
        observation = self.observe()
        if not observation.valid or observation.screen is not ScreenState.HOME:
            raise ControllerError(f"baseline recovery failed: {observation.to_dict()}")

    def reset_episode(self, *, scan_upgrades: bool = True) -> Observation:
        """Return through the game's ordinary result/home flow and start again."""
        observation = self.observe()
        if observation.screen is ScreenState.MODAL:
            self._dismiss_account_link_reminder()
            observation = self._wait_for_home()
        if observation.screen is ScreenState.RESULT:
            time.sleep(2.0)
            for _ in range(3):
                if not self._tap_result_home():
                    break
                try:
                    self._wait_for_home(timeout=12.0)
                    break
                except DeviceFailureError:
                    raise
                except ControllerError:
                    continue
            else:
                raise ControllerError("result-to-home transition did not reach Battle home")
        elif observation.screen is not ScreenState.HOME:
            raise ControllerError(f"cannot reset from {observation.screen.value}")
        return self.start_episode(scan_upgrades=scan_upgrades)

    def end_episode(self) -> Observation:
        """Use the game's ordinary in-run menu to reach its result screen."""
        observation = self.observe()
        if not observation.valid or observation.screen is not ScreenState.ACTIVE_RUN:
            raise ControllerError(f"cannot end from {observation.screen.value}")
        for _ in range(2):
            self._tap(1015, 68)
            # The game's end-run menu is a transient Unity overlay, not one of
            # the stable classifier states; the confirmation is profile-owned.
            time.sleep(0.5)
            self._tap(960, 270)
            time.sleep(0.25)
            self._tap(720, 1095)
            try:
                return self._wait_for(ScreenState.RESULT, timeout=12.0)
            except DeviceFailureError:
                raise
            except ControllerError:
                continue
        raise ControllerError("end-run confirmation did not reach the result screen")

    def await_death(self, *, timeout: float = 600.0) -> Observation:
        """Wait under the deterministic WAIT policy until a genuine result frame appears."""
        deadline = time.monotonic() + timeout
        last_screen: ScreenState | None = None
        consecutive_unknowns = 0
        while time.monotonic() < deadline:
            frame = self.device.screenshot()
            last_screen = self.extractor.classify(frame.png_bytes)
            if last_screen is ScreenState.RESULT:
                return self.observe()
            if last_screen is ScreenState.UNKNOWN:
                self._raise_if_app_not_foreground()
                consecutive_unknowns += 1
                if consecutive_unknowns > self.config.max_consecutive_unknown_classifications:
                    raise ControllerError(
                        "persistent unknown screen classification while waiting for death; "
                        f"count={consecutive_unknowns}"
                    )
                time.sleep(self.config.decision_interval_seconds)
                continue
            consecutive_unknowns = 0
            if last_screen not in (ScreenState.ACTIVE_RUN, ScreenState.MODAL):
                raise ControllerError(
                    f"unexpected state while waiting for death: {last_screen.value}"
                )
            time.sleep(self.config.decision_interval_seconds)
        raise ControllerError(
            "timed out waiting for natural death; "
            f"last={last_screen.value if last_screen else 'none'}"
        )
