"""Pure domain contracts for Tier-1 observations and run decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum

OBSERVATION_SCHEMA_VERSION = "observation-v1"
ACTION_SCHEMA_VERSION = "run-action-v1"


class ScreenState(StrEnum):
    HOME = "battle_home_tier_1"
    TIER_SELECT = "tier_select"
    ACTIVE_RUN = "tier_1_active_run"
    RESULT = "tier_1_result"
    MODAL = "supported_modal"
    UNKNOWN = "unknown"


class RunAction(StrEnum):
    """The only actions a V1 policy may request."""

    WAIT = "WAIT"
    BUY_HEALTH = "BUY_HEALTH"
    BUY_DAMAGE = "BUY_DAMAGE"
    BUY_ATTACK_SPEED = "BUY_ATTACK_SPEED"
    BUY_CRITICAL_CHANCE = "BUY_CRITICAL_CHANCE"
    BUY_CRITICAL_FACTOR = "BUY_CRITICAL_FACTOR"


RUN_ACTIONS: tuple[RunAction, ...] = tuple(RunAction)


class ActionOutcome(StrEnum):
    EXECUTED = "executed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    NAVIGATION_FAILED = "navigation_failed"
    INVALID_OBSERVATION = "invalid_observation"


class TerminationReason(StrEnum):
    TOWER_DIED = "tower_died"
    USER_STOP = "user_stop"
    SAFETY_TIMEOUT = "safety_timeout"
    INVALID_OBSERVATION = "invalid_observation"
    NAVIGATION_FAILURE = "navigation_failure"
    DEVICE_FAILURE = "device_failure"
    BASELINE_DRIFT = "baseline_drift"


@dataclass(frozen=True)
class FieldReading[T]:
    """A visible field with provenance and confidence."""

    value: T | None
    confidence: float
    source_frame_id: str
    region_id: str
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    """Normalized, evidence-bearing observation delivered to a policy."""

    frame_id: str
    captured_at_monotonic: float
    screen: ScreenState
    wave: FieldReading[int]
    cash_normalized: FieldReading[float]
    health_fraction: FieldReading[float]
    max_health_normalized: FieldReading[float]
    upgrade_levels: Mapping[RunAction, FieldReading[int]]
    upgrade_costs_normalized: Mapping[RunAction, FieldReading[float]]
    action_mask: tuple[RunAction, ...]
    valid: bool
    invalid_reasons: tuple[str, ...] = ()
    schema_version: str = OBSERVATION_SCHEMA_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "captured_at_monotonic": self.captured_at_monotonic,
            "screen": self.screen.value,
            "wave": self.wave.to_dict(),
            "cash_normalized": self.cash_normalized.to_dict(),
            "health_fraction": self.health_fraction.to_dict(),
            "max_health_normalized": self.max_health_normalized.to_dict(),
            "upgrade_levels": {
                action.value: reading.to_dict() for action, reading in self.upgrade_levels.items()
            },
            "upgrade_costs_normalized": {
                action.value: reading.to_dict()
                for action, reading in self.upgrade_costs_normalized.items()
            },
            "action_mask": [action.value for action in self.action_mask],
            "valid": self.valid,
            "invalid_reasons": list(self.invalid_reasons),
            "schema_version": self.schema_version,
            "action_schema_version": self.action_schema_version,
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...]


class ObservationValidator:
    """Fail-closed temporal and logical validation for observations."""

    def validate(
        self, observation: Observation, previous: Observation | None = None
    ) -> ValidationResult:
        reasons: list[str] = list(observation.invalid_reasons)
        if observation.schema_version != OBSERVATION_SCHEMA_VERSION:
            reasons.append("observation schema version mismatch")
        if observation.action_schema_version != ACTION_SCHEMA_VERSION:
            reasons.append("action schema version mismatch")
        if not 0.0 <= observation.wave.confidence <= 1.0:
            reasons.append("wave confidence outside [0, 1]")
        if not 0.0 <= observation.cash_normalized.confidence <= 1.0:
            reasons.append("cash confidence outside [0, 1]")
        if not 0.0 <= observation.health_fraction.confidence <= 1.0:
            reasons.append("health confidence outside [0, 1]")
        if observation.cash_normalized.value is not None and observation.cash_normalized.value < 0:
            reasons.append("cash is negative")
        if (
            observation.health_fraction.value is not None
            and not 0 <= observation.health_fraction.value <= 1
        ):
            reasons.append("health fraction outside [0, 1]")
        if (
            observation.max_health_normalized.value is not None
            and observation.max_health_normalized.value < 0
        ):
            reasons.append("maximum health is negative")
        if any(action not in RUN_ACTIONS for action in observation.action_mask):
            reasons.append("action mask contains an unsupported action")
        if observation.screen is ScreenState.ACTIVE_RUN:
            for name, reading in (
                ("wave", observation.wave),
                ("cash", observation.cash_normalized),
                ("health", observation.health_fraction),
            ):
                if reading.value is None:
                    reasons.append(f"active-run {name} reading is missing")
        if previous is not None:
            if observation.captured_at_monotonic <= previous.captured_at_monotonic:
                reasons.append("frame timestamp is not newer than the previous observation")
            if (
                observation.screen is ScreenState.ACTIVE_RUN
                and previous.screen is ScreenState.ACTIVE_RUN
                and observation.wave.value is not None
                and previous.wave.value is not None
                and observation.wave.value < previous.wave.value
            ):
                reasons.append("wave moved backwards inside an active episode")
        return ValidationResult(valid=observation.valid and not reasons, reasons=tuple(reasons))


@dataclass(frozen=True)
class StepResult:
    """One validated environment transition."""

    observation: Observation
    next_observation: Observation | None
    action: RunAction
    action_mask: tuple[RunAction, ...]
    outcome: ActionOutcome
    reward: float
    terminated: bool
    truncated: bool
    termination_reason: TerminationReason | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "observation": self.observation.to_dict(),
            "next_observation": (
                self.next_observation.to_dict() if self.next_observation is not None else None
            ),
            "action": self.action.value,
            "action_mask": [action.value for action in self.action_mask],
            "outcome": self.outcome.value,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": (
                self.termination_reason.value if self.termination_reason is not None else None
            ),
            "elapsed_seconds": self.elapsed_seconds,
        }
