from PIL import Image, ImageDraw

from tower_rl.android_probe import EXPECTED_SIZE, ScreenKind, classify_frame


def _frame(
    base: tuple[int, int, int], center: tuple[int, int, int], *, home_header: bool = False
) -> Image.Image:
    image = Image.new("RGB", EXPECTED_SIZE, base)
    draw = ImageDraw.Draw(image)
    if home_header:
        draw.rectangle((0, 0, 1079, 99), fill=(54, 49, 118))
    draw.rectangle((0, 900, 1079, 1100), fill=center)
    return image


def test_classifies_pinned_home_profile() -> None:
    image = _frame((28, 24, 53), (54, 49, 118), home_header=True)

    assert classify_frame(image) is ScreenKind.HOME


def test_classifies_active_run_and_result() -> None:
    active = _frame((8, 7, 17), (8, 11, 20))
    result = _frame((2, 2, 5), (20, 19, 53))

    assert classify_frame(active) is ScreenKind.ACTIVE_RUN
    assert classify_frame(result) is ScreenKind.RESULT


def test_classifies_wave_info_modal() -> None:
    image = _frame((8, 7, 17), (20, 19, 53))
    ImageDraw.Draw(image).point((540, 180), fill=(217, 221, 226))

    assert classify_frame(image) is ScreenKind.WAVE_INFO


def test_fails_closed_on_unrecognized_layout() -> None:
    assert classify_frame(Image.new("RGB", (720, 1280), (0, 0, 0))) is ScreenKind.UNKNOWN
