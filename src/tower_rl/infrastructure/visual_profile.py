"""The calibrated visual profile used to gate the two boundary taps.

`M1B-E004` showed why this must be versioned alongside the progression profile:
playing the game advanced the account's wave record and coin balance, which
unlocked new home-screen UI, which repainted the region one calibrated anchor
sampled, which made the screen unclassifiable and correctly refused every tap.

Anchors here were chosen in structurally meaningful places - a panel interior, a
button we are about to press, the header bar - rather than wherever a pixel
happened to be constant, and every one was verified against live frames whose
labels came from the game's own lifecycle rather than from assumption.  Only the
sampled values are recorded; screenshots carry account state and are never
committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from PIL import Image

VISUAL_PROFILE_ID = "tower-play-29.0.3-clone-wave11-v2"
"""Bound to a progression profile. Advancing progression invalidates it."""

EXPECTED_SIZE = (1080, 1920)

#: Per-channel tolerance. Rendering is deterministic under lavapipe, so the
#: observed spread was zero; this allows for minor compression differences
#: without admitting a genuinely different screen.
TOLERANCE = 12

#: Seconds to let the result panel finish animating before classifying it. The
#: panel slides in, and frames captured earlier than this varied by up to 240 per
#: channel at the same pixel.
RESULT_SETTLE_SECONDS = 6.0

Anchor = tuple[tuple[int, int], tuple[int, int, int]]


class Screen(StrEnum):
    """Only the states the boundary logic needs to distinguish."""

    HOME = "battle_home_tier_1"
    ACTIVE_RUN = "tier_1_active_run"
    RESULT = "tier_1_result"
    UNKNOWN = "unknown"


#: Home: header bar, title, both panels, the BATTLE button's border and interior,
#: and the navigation bar. Seven independent anchors, so one repainted region
#: cannot silently break the gate the way a single anchor did.
HOME_ANCHORS: tuple[Anchor, ...] = (
    ((560, 80), (54, 49, 118)),
    ((400, 258), (63, 59, 102)),
    ((540, 840), (54, 49, 118)),
    ((540, 1000), (54, 49, 118)),
    ((300, 1487), (102, 35, 137)),
    ((540, 1553), (22, 6, 35)),
    ((90, 1860), (144, 136, 255)),
)

#: Active run: the health bar, the upper HUD and the playfield.
ACTIVE_ANCHORS: tuple[Anchor, ...] = (
    ((40, 1080), (14, 194, 153)),
    ((500, 480), (7, 97, 172)),
    ((780, 940), (180, 180, 181)),
)

#: Result: the Game Stats panel interior and both of its buttons. Anchoring on
#: the RETRY button means the gate confirms the control it is about to press is
#: actually rendered there.
RESULT_ANCHORS: tuple[Anchor, ...] = (
    ((540, 1100), (20, 19, 53)),
    ((100, 950), (20, 19, 53)),
    ((300, 1417), (255, 255, 255)),
    ((780, 1417), (149, 149, 154)),
)

PROFILE: dict[Screen, tuple[Anchor, ...]] = {
    Screen.HOME: HOME_ANCHORS,
    Screen.ACTIVE_RUN: ACTIVE_ANCHORS,
    Screen.RESULT: RESULT_ANCHORS,
}


@dataclass(frozen=True)
class Calibration:
    """One screen's evidence, kept together so drift is attributable."""

    screen: Screen
    anchors: tuple[Anchor, ...]
    tolerance: int = TOLERANCE

    def matches(self, image: Image.Image) -> bool:
        return all(
            _close(_pixel(image, point), expected, self.tolerance)
            for point, expected in self.anchors
        )

    def mismatches(self, image: Image.Image) -> tuple[str, ...]:
        """Which anchors disagree, so a drift report names the region."""
        return tuple(
            f"{point} expected {expected} saw {_pixel(image, point)}"
            for point, expected in self.anchors
            if not _close(_pixel(image, point), expected, self.tolerance)
        )


CALIBRATIONS: tuple[Calibration, ...] = tuple(
    Calibration(screen, anchors) for screen, anchors in PROFILE.items()
)


def classify(image: Image.Image) -> Screen:
    """Return the screen, or `UNKNOWN` on anything this profile does not cover.

    Every calibration must match in full. Requiring the conjunction is what makes
    a single repainted region fail closed into `UNKNOWN`, which refuses a tap,
    rather than silently matching the wrong screen and permitting one.
    """
    if image.size != EXPECTED_SIZE:
        return Screen.UNKNOWN
    frame = image.convert("RGB")
    matched = [calibration.screen for calibration in CALIBRATIONS if calibration.matches(frame)]
    # Two screens matching at once means the profile no longer discriminates, and
    # that is a drift signal rather than a reason to pick one.
    return matched[0] if len(matched) == 1 else Screen.UNKNOWN


def describe_drift(image: Image.Image) -> dict[str, tuple[str, ...]]:
    """Per-screen anchor mismatches, for diagnosing a recalibration."""
    frame = image.convert("RGB")
    return {
        calibration.screen.value: calibration.mismatches(frame)
        for calibration in CALIBRATIONS
    }


def _pixel(image: Image.Image, point: tuple[int, int]) -> tuple[int, int, int]:
    value = image.getpixel(point)
    assert isinstance(value, tuple)
    return (int(value[0]), int(value[1]), int(value[2]))


def _close(left: tuple[int, int, int], right: tuple[int, int, int], tolerance: int) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right, strict=True))
