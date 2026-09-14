import pytest

from tower_rl.domain import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    FieldReading,
    Observation,
    ObservationValidator,
    RunAction,
    ScreenState,
)
from tower_rl.ports.android import ScreenPoint


def _reading[T](value: T | None, frame: str = "f1") -> FieldReading[T]:
    return FieldReading(value, 1.0, frame, "test")


def _observation(*, frame: str = "f1", timestamp: float = 1.0, wave: int = 1) -> Observation:
    levels = {action: _reading(0, frame) for action in RunAction if action is not RunAction.WAIT}
    costs = {action: _reading(1.0, frame) for action in RunAction if action is not RunAction.WAIT}
    return Observation(
        frame_id=frame,
        captured_at_monotonic=timestamp,
        screen=ScreenState.ACTIVE_RUN,
        wave=_reading(wave, frame),
        cash_normalized=_reading(0.5, frame),
        health_fraction=_reading(1.0, frame),
        max_health_normalized=_reading(1.0, frame),
        upgrade_levels=levels,
        upgrade_costs_normalized=costs,
        action_mask=(RunAction.WAIT,),
        valid=True,
    )


def test_valid_active_observation_round_trips() -> None:
    observation = _observation()

    result = ObservationValidator().validate(observation)

    assert result.valid
    assert observation.to_dict()["schema_version"] == OBSERVATION_SCHEMA_VERSION
    assert observation.to_dict()["action_schema_version"] == ACTION_SCHEMA_VERSION


def test_validator_rejects_stale_and_backward_observation() -> None:
    previous = _observation(frame="f1", timestamp=2.0, wave=2)
    current = _observation(frame="f2", timestamp=2.0, wave=1)

    result = ObservationValidator().validate(current, previous)

    assert not result.valid
    assert "frame timestamp is not newer than the previous observation" in result.reasons
    assert "wave moved backwards inside an active episode" in result.reasons


def test_screen_point_is_normalized_and_converts_to_pixels() -> None:
    assert ScreenPoint(0.5, 0.5).pixels(1080, 1920) == (540, 960)
    with pytest.raises(ValueError):
        ScreenPoint(1.1, 0.5)
