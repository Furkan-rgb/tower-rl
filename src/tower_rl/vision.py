"""Live-screen extraction for the validated Tier-1 UI profile.

This module deliberately extracts only visible state.  It does not contain game
rules or inferred mechanics; unknown or low-confidence text remains unavailable.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from tower_rl.android_probe import ScreenKind, classify_frame
from tower_rl.domain import FieldReading, Observation, RunAction, ScreenState

PROFILE_ID = "ui-v1-play-29.0.3-1080x1920-lavapipe-swangle"
_FRAME_SIZE = (1080, 1920)
_LOCAL_TESSERACT_ROOT = Path.home() / ".local/share/tower-rl/tesseract-5.5.0/usr"


@dataclass(frozen=True)
class PixelRegion:
    """Pixel crop in the validated logical screen profile."""

    left: int
    top: int
    right: int
    bottom: int

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop((self.left, self.top, self.right, self.bottom))


@dataclass(frozen=True)
class ActionTarget:
    tab: str
    tap_region: PixelRegion
    value_region: PixelRegion
    cost_region: PixelRegion


@dataclass(frozen=True)
class UiProfile:
    """Coordinates and reading regions for one exact app/device profile."""

    profile_id: str
    cash: PixelRegion
    health: PixelRegion
    wave: PixelRegion
    attack_cards: PixelRegion
    defense_cards: PixelRegion
    action_targets: dict[RunAction, ActionTarget]
    tab_points: dict[str, tuple[int, int]]
    result_home_point: tuple[int, int]
    account_link_reminder_close_point: tuple[int, int]


PROFILE = UiProfile(
    profile_id=PROFILE_ID,
    cash=PixelRegion(0, 0, 320, 110),
    health=PixelRegion(0, 980, 540, 1140),
    wave=PixelRegion(540, 980, 1080, 1140),
    attack_cards=PixelRegion(0, 1240, 1080, 1665),
    defense_cards=PixelRegion(0, 1240, 1080, 1500),
    action_targets={
        RunAction.BUY_DAMAGE: ActionTarget(
            "attack", PixelRegion(30, 1255, 530, 1450),
            PixelRegion(280, 1270, 515, 1385), PixelRegion(280, 1385, 515, 1450),
        ),
        RunAction.BUY_ATTACK_SPEED: ActionTarget(
            "attack", PixelRegion(550, 1255, 1050, 1450),
            PixelRegion(800, 1270, 1040, 1385), PixelRegion(800, 1385, 1040, 1450),
        ),
        RunAction.BUY_CRITICAL_CHANCE: ActionTarget(
            "attack", PixelRegion(30, 1460, 530, 1660),
            PixelRegion(280, 1470, 515, 1585), PixelRegion(280, 1585, 515, 1650),
        ),
        RunAction.BUY_CRITICAL_FACTOR: ActionTarget(
            "attack", PixelRegion(550, 1460, 1050, 1660),
            PixelRegion(800, 1470, 1040, 1585), PixelRegion(800, 1585, 1040, 1650),
        ),
        RunAction.BUY_HEALTH: ActionTarget(
            "defense", PixelRegion(30, 1255, 530, 1450),
            PixelRegion(280, 1270, 515, 1385), PixelRegion(280, 1385, 515, 1450),
        ),
    },
    tab_points={"attack": (180, 1850), "defense": (540, 1850), "utility": (900, 1850)},
    result_home_point=(780, 1420),
    account_link_reminder_close_point=(884, 531),
)


@dataclass(frozen=True)
class OcrReading:
    text: str
    confidence: float
    reason: str | None = None


class OcrEngine:
    """Small subprocess boundary around a local Tesseract installation."""

    def __init__(self, executable: str | Path | None = None) -> None:
        self._environment = os.environ.copy()
        configured_executable = executable or self._environment.get("TOWER_RL_TESSERACT")
        local_executable = _LOCAL_TESSERACT_ROOT / "bin/tesseract"
        if configured_executable is None and local_executable.is_file():
            configured_executable = local_executable
            self._environment.setdefault(
                "TESSDATA_PREFIX",
                str(_LOCAL_TESSERACT_ROOT / "share/tesseract-ocr/5/tessdata"),
            )
            local_library_path = str(_LOCAL_TESSERACT_ROOT / "lib/x86_64-linux-gnu")
            inherited_library_path = self._environment.get("LD_LIBRARY_PATH")
            self._environment["LD_LIBRARY_PATH"] = (
                f"{local_library_path}{os.pathsep}{inherited_library_path}"
                if inherited_library_path
                else local_library_path
            )
        self.executable = str(configured_executable or "tesseract")
        tessdata = self._environment.get("TOWER_RL_TESSDATA")
        if tessdata:
            self._environment["TESSDATA_PREFIX"] = tessdata
        library_path = self._environment.get("TOWER_RL_TESSERACT_LIB")
        if library_path:
            self._environment["LD_LIBRARY_PATH"] = library_path

    def read(
        self,
        image: Image.Image,
        *,
        whitelist: str = "0123456789.$/%xXabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ",
    ) -> OcrReading:
        gray = image.convert("L").resize((image.width * 3, image.height * 3))
        stream = io.BytesIO()
        gray.save(stream, format="PNG")
        try:
            result = subprocess.run(
                [
                    self.executable,
                    "stdin",
                    "stdout",
                    "--psm",
                    "6",
                    "-c",
                    f"tessedit_char_whitelist={whitelist}",
                ],
                input=stream.getvalue(),
                capture_output=True,
                timeout=5,
                env=self._environment,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return OcrReading("", 0.0, f"ocr unavailable: {error}")
        if result.returncode != 0:
            return OcrReading(
                "", 0.0, result.stderr.decode(errors="replace").strip() or "ocr failed"
            )
        text = result.stdout.decode(errors="replace").strip()
        return OcrReading(text, 1.0 if text else 0.0, None if text else "ocr returned no text")


def _reading[T](
    value: T | None, frame_id: str, region_id: str, *, confidence: float, reason: str | None = None
) -> FieldReading[T]:
    return FieldReading(value, confidence, frame_id, region_id, reason)


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", text)


def _first_int(text: str) -> int | None:
    values = _numbers(text)
    if not values:
        return None
    try:
        return int(float(values[0].replace(",", ".")))
    except ValueError:
        return None


def _first_float(text: str) -> float | None:
    values = _numbers(text)
    if not values:
        return None
    try:
        return float(values[0].replace(",", "."))
    except ValueError:
        return None


def _screen_state(kind: ScreenKind) -> ScreenState:
    return {
        ScreenKind.HOME: ScreenState.HOME,
        ScreenKind.ACTIVE_RUN: ScreenState.ACTIVE_RUN,
        ScreenKind.WAVE_INFO: ScreenState.MODAL,
        ScreenKind.ACCOUNT_LINK_REMINDER: ScreenState.MODAL,
        ScreenKind.MODAL: ScreenState.MODAL,
        ScreenKind.RESULT: ScreenState.RESULT,
    }.get(kind, ScreenState.UNKNOWN)


def _bar_fraction(image: Image.Image) -> float | None:
    """Estimate current/max health from the visible cyan health bar."""

    rgb = image.convert("RGB")
    samples: list[int] = []
    for x in range(35, 508):
        pixel = rgb.getpixel((x, 1095))
        if not isinstance(pixel, tuple) or len(pixel) < 3:
            continue
        r, g, b = pixel[:3]
        if g > 140 and b > 100 and r < 120:
            samples.append(x)
    if not samples:
        return None
    return max(0.0, min(1.0, (max(samples) - min(samples) + 1) / 473))


class ObservationExtractor:
    """Extract visible scalar state from one screenshot and active tab."""

    def __init__(self, ocr: OcrEngine | None = None, profile: UiProfile = PROFILE) -> None:
        self.ocr = ocr or OcrEngine()
        self.profile = profile

    def _read_region(self, region: PixelRegion, image: Image.Image, whitelist: str) -> OcrReading:
        reading = self.ocr.read(region.crop(image), whitelist=whitelist)
        if not reading.text:
            reading = self.ocr.read(region.crop(image), whitelist=whitelist)
        return reading

    def classify(self, png_bytes: bytes) -> ScreenState:
        """Classify a frame without running active-game OCR."""
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return _screen_state(classify_frame(image))

    def is_account_link_reminder(self, png_bytes: bytes) -> bool:
        """Recognize only the captured, lifecycle-blocking account-link reminder."""
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return classify_frame(image) is ScreenKind.ACCOUNT_LINK_REMINDER

    def extract(
        self,
        png_bytes: bytes,
        *,
        frame_id: str,
        captured_at_monotonic: float,
        active_tab: str = "attack",
    ) -> Observation:
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        kind = classify_frame(image)
        screen = _screen_state(kind)
        inactive = OcrReading("", 0.0, "field is unavailable outside an active run")
        cash_ocr = inactive
        health_ocr = inactive
        wave_ocr = inactive
        cash_value: float | None = None
        health_fraction: float | None = None
        wave_value: int | None = None
        if screen is ScreenState.ACTIVE_RUN:
            cash_ocr = self._read_region(self.profile.cash, image, "0123456789.$")
            health_ocr = self._read_region(self.profile.health, image, "0123456789./")
            wave_ocr = self._read_region(self.profile.wave, image, "0123456789Wave/")
            cash_value = _first_float(cash_ocr.text)
            health_fraction = _bar_fraction(image)
            wave_match = re.search(r"wave\s*(\d+)", wave_ocr.text, flags=re.IGNORECASE)
            wave_value = int(wave_match.group(1)) if wave_match else _first_int(wave_ocr.text)
        levels: dict[RunAction, FieldReading[int]] = {
            action: _reading(
                None, frame_id, "upgrade-level", confidence=0.0, reason="tab not scanned"
            )
            for action in RunAction
            if action is not RunAction.WAIT
        }
        costs: dict[RunAction, FieldReading[float]] = {
            action: _reading(
                None, frame_id, "upgrade-cost", confidence=0.0, reason="tab not scanned"
            )
            for action in RunAction
            if action is not RunAction.WAIT
        }
        if screen is ScreenState.ACTIVE_RUN and active_tab == "attack":
            labels = [
                (RunAction.BUY_DAMAGE, "damage"),
                (RunAction.BUY_ATTACK_SPEED, "attack speed"),
                (RunAction.BUY_CRITICAL_CHANCE, "critical chance"),
                (RunAction.BUY_CRITICAL_FACTOR, "critical factor"),
            ]
        elif screen is ScreenState.ACTIVE_RUN and active_tab == "defense":
            labels = [(RunAction.BUY_HEALTH, "health")]
        else:
            labels = []
        for action, label in labels:
            target = self.profile.action_targets[action]
            value_ocr = self._read_region(target.value_region, image, "0123456789.$%/xX")
            cost_ocr = self._read_region(target.cost_region, image, "0123456789.$")
            level = _first_int(value_ocr.text)
            cost = _first_float(cost_ocr.text)
            for _ in range(2):
                if level is not None and cost is not None:
                    break
                if level is None:
                    value_ocr = self.ocr.read(
                        target.value_region.crop(image), whitelist="0123456789.$%/xX"
                    )
                    level = _first_int(value_ocr.text)
                if cost is None:
                    cost_ocr = self.ocr.read(
                        target.cost_region.crop(image), whitelist="0123456789.$"
                    )
                    cost = _first_float(cost_ocr.text)
            levels[action] = _reading(
                level, frame_id, f"{label}.level", confidence=value_ocr.confidence,
                reason=value_ocr.reason,
            )
            costs[action] = _reading(
                cost, frame_id, f"{label}.cost", confidence=cost_ocr.confidence,
                reason=cost_ocr.reason,
            )
        reasons: list[str] = []
        if image.size != _FRAME_SIZE:
            reasons.append("unexpected frame size")
        if screen is ScreenState.ACTIVE_RUN and (
            wave_value is None or cash_value is None or health_fraction is None
        ):
            reasons.append("active-run mandatory reading missing")
        valid = not reasons and screen is not ScreenState.UNKNOWN
        mask = [RunAction.WAIT]
        if valid and screen is ScreenState.ACTIVE_RUN:
            for action in RunAction:
                if action is RunAction.WAIT:
                    continue
                level = levels[action].value
                cost = costs[action].value
                if level is not None and cost is not None and (
                    cash_value is None or cost <= cash_value
                ):
                    mask.append(action)
        return Observation(
            frame_id=frame_id,
            captured_at_monotonic=captured_at_monotonic,
            screen=screen,
            wave=_reading(
                wave_value, frame_id, "wave", confidence=wave_ocr.confidence, reason=wave_ocr.reason
            ),
            cash_normalized=_reading(
                cash_value, frame_id, "cash", confidence=cash_ocr.confidence, reason=cash_ocr.reason
            ),
            health_fraction=_reading(
                health_fraction,
                frame_id,
                "health",
                confidence=health_ocr.confidence,
                reason=health_ocr.reason,
            ),
            max_health_normalized=_reading(
                1.0 if health_fraction is not None else None,
                frame_id,
                "max-health",
                confidence=health_ocr.confidence,
                reason=health_ocr.reason,
            ),
            upgrade_levels=levels,
            upgrade_costs_normalized=costs,
            action_mask=tuple(mask),
            valid=valid,
            invalid_reasons=tuple(reasons),
        )
