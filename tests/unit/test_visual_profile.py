from __future__ import annotations

from PIL import Image

from tower_rl.infrastructure.visual_profile import (
    ACTIVE_ANCHORS,
    EXPECTED_SIZE,
    HOME_ANCHORS,
    RESULT_ANCHORS,
    TOLERANCE,
    VISUAL_PROFILE_ID,
    Anchor,
    Screen,
    classify,
    describe_drift,
)


def _frame(anchors: tuple[Anchor, ...], *, fill: tuple[int, int, int] = (3, 3, 3)) -> Image.Image:
    """A frame carrying exactly one screen's recorded anchor values."""
    image = Image.new("RGB", EXPECTED_SIZE, fill)
    for point, value in anchors:
        image.putpixel(point, value)
    return image


def test_each_recorded_screen_classifies_as_itself() -> None:
    assert classify(_frame(HOME_ANCHORS)) is Screen.HOME
    assert classify(_frame(ACTIVE_ANCHORS)) is Screen.ACTIVE_RUN
    assert classify(_frame(RESULT_ANCHORS)) is Screen.RESULT


def test_one_repainted_anchor_fails_closed_rather_than_guessing() -> None:
    """The regression that blocked stage B: progression repainted one anchor."""
    image = _frame(HOME_ANCHORS)
    point, value = HOME_ANCHORS[0]
    image.putpixel(point, (255, 255, 255))

    assert classify(image) is Screen.UNKNOWN
    drift = describe_drift(image)[Screen.HOME.value]
    assert len(drift) == 1 and str(point) in drift[0]


def test_home_does_not_rely_on_a_single_anchor() -> None:
    """Seven independent anchors, so no one region can silently own the gate."""
    assert len(HOME_ANCHORS) >= 6
    points = {point for point, _ in HOME_ANCHORS}
    assert len(points) == len(HOME_ANCHORS)
    # Spread across the screen rather than clustered in one column.
    assert max(x for x, _ in points) - min(x for x, _ in points) > 300
    assert max(y for _, y in points) - min(y for _, y in points) > 1000


def test_the_result_gate_anchors_on_the_button_it_will_press() -> None:
    """A gate that cannot see RETRY must not permit tapping RETRY."""
    assert any(point == (300, 1417) for point, _ in RESULT_ANCHORS)

    image = _frame(RESULT_ANCHORS)
    image.putpixel((300, 1417), (20, 19, 53))  # button absent, panel still drawn

    assert classify(image) is Screen.UNKNOWN


def test_tolerance_admits_rendering_noise_but_not_a_different_screen() -> None:
    image = _frame(HOME_ANCHORS)
    for point, value in HOME_ANCHORS:
        image.putpixel(point, tuple(min(255, channel + TOLERANCE) for channel in value))
    assert classify(image) is Screen.HOME

    image = _frame(HOME_ANCHORS)
    for point, value in HOME_ANCHORS:
        image.putpixel(point, tuple(min(255, channel + TOLERANCE + 1) for channel in value))
    assert classify(image) is Screen.UNKNOWN


def test_an_unexpected_resolution_is_never_classified() -> None:
    assert classify(Image.new("RGB", (720, 1280), (54, 49, 118))) is Screen.UNKNOWN


def test_a_blank_frame_is_unknown_not_home() -> None:
    assert classify(Image.new("RGB", EXPECTED_SIZE, (0, 0, 0))) is Screen.UNKNOWN
    assert classify(Image.new("RGB", EXPECTED_SIZE, (28, 24, 53))) is Screen.UNKNOWN


def test_the_profile_is_versioned_with_its_progression_state() -> None:
    """M1B-E004: the visual profile is only valid for one progression profile."""
    assert "29.0.3" in VISUAL_PROFILE_ID
    assert "wave11" in VISUAL_PROFILE_ID
