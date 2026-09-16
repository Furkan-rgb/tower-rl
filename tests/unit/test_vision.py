import io

from PIL import Image, ImageDraw

from tower_rl.android_probe import ScreenKind, classify_frame
from tower_rl.domain import RunAction
from tower_rl.vision import PROFILE, ObservationExtractor, _first_int


def test_decimal_upgrade_levels_are_read_as_integer_levels() -> None:
    assert _first_int("1.00") == 1
    assert _first_int("x1.20") == 1


def test_profile_covers_every_supported_purchase_action() -> None:
    purchase_actions = {action for action in RunAction if action is not RunAction.WAIT}

    assert set(PROFILE.action_targets) == purchase_actions
    assert all(target.tap_region.right <= 1080 for target in PROFILE.action_targets.values())


def test_non_active_frames_do_not_invoke_ocr() -> None:
    class UnexpectedOcr:
        def read(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("OCR should not run outside an active episode")

    image = Image.new("RGB", (1080, 1920), (28, 24, 53))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1079, 99), fill=(54, 49, 118))
    draw.rectangle((0, 900, 1079, 1100), fill=(54, 49, 118))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")

    observation = ObservationExtractor(ocr=UnexpectedOcr()).extract(  # type: ignore[arg-type]
        encoded.getvalue(), frame_id="home-1", captured_at_monotonic=1.0
    )

    assert observation.valid
    assert observation.wave.value is None


def test_account_link_reminder_uses_its_captured_geometry() -> None:
    image = Image.new("RGB", (1080, 1920), (20, 19, 53))
    for point in (
        (130, 500),
        (130, 600),
        (950, 500),
        (950, 600),
        (300, 1405),
        (780, 1405),
        (300, 1535),
        (780, 1535),
        (884, 531),
        (870, 520),
        (900, 550),
    ):
        image.putpixel(point, (255, 255, 255))

    assert classify_frame(image) is ScreenKind.ACCOUNT_LINK_REMINDER


def test_generic_modal_does_not_match_account_link_reminder() -> None:
    image = Image.new("RGB", (1080, 1920), (10, 10, 10))
    image.putpixel((540, 1000), (100, 100, 100))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")

    extractor = ObservationExtractor()

    assert classify_frame(image) is ScreenKind.MODAL
    assert not extractor.is_account_link_reminder(encoded.getvalue())
