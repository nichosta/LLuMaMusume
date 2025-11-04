"""Utilities for detecting cinematic states from on-screen captures.

The detector operates on successive captures, combining a lightweight frame
difference metric with region-of-interest heuristics so we can distinguish
between free-form gameplay and the two cinematic flavours seen in Uma Musume:

* Fullscreen story cutscenes (menus/tabs suppressed, Skip appears after pausing)
* Primary-region cutscenes (gacha pulls, Skip chip lives inside the primary pane)

The logic is intentionally model-agnostic; callers feed in raw PIL images plus
high-level hints (e.g. whether the menus pane is currently usable, the button
labels returned by Vision).  The heuristics lean on three signals:

1. Whether the bright “pin” control in the top-left corner is still visible.
2. Variance/brightness in the tabs strip where the fullscreen Skip overlay lives.
3. Variance/brightness in the primary bottom-right corner where gacha Skip chips appear.

The detector tracks frame-to-frame deltas to tell “playing” from “paused” states
while smoothing single-frame noise. The individual scores are exposed so a CLI
or unit test harness can report them for tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

from lluma_os.config import CaptureConfig


class CinematicKind(str, Enum):
    """High-level categorisation of cinematic modes."""

    NONE = "none"
    FULLSCREEN = "fullscreen"
    PRIMARY = "primary"


class PlaybackState(str, Enum):
    """Whether a cinematic appears to be actively playing or paused."""

    UNKNOWN = "unknown"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass(slots=True)
class CinematicObservation:
    """Single-frame observation fed into the detector."""

    image: Image.Image
    menu_is_usable: Optional[bool] = None
    button_labels: Sequence[str] = ()


@dataclass(slots=True)
class CinematicDetectionResult:
    """Detector output for a single observation."""

    kind: CinematicKind
    playback: PlaybackState
    diff_score: float
    skip_hint_primary: bool
    skip_hint_tabs: bool
    skip_label_hint: bool
    menu_unusable_streak: int
    pin_present: bool
    pin_bright_ratio: float
    primary_diff_score: float


def _downscale_luma(image: Image.Image, *, width: int) -> np.ndarray:
    """Convert the frame to a small luma buffer for inexpensive diffing."""

    if image.width > width:
        ratio = width / image.width
        height = max(1, int(round(image.height * ratio)))
        resized = image.resize((width, height), Image.BILINEAR)
    else:
        resized = image.convert("L")
    luma = np.asarray(resized.convert("L"), dtype=np.float32) / 255.0
    return luma


def _roi_luma_stats(image: Image.Image, bounds: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the normalised luma mean and standard deviation for a crop."""

    x0, y0, x1, y1 = bounds
    width, height = image.size
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return (0.0, 0.0)
    crop = image.crop((x0, y0, x1, y1)).convert("L")
    arr = np.asarray(crop, dtype=np.float32)
    if arr.size == 0:
        return (0.0, 0.0)
    mean = float(arr.mean() / 255.0)
    std = float(arr.std() / 255.0)
    return (mean, std)


def _normalise_labels(labels: Iterable[str]) -> list[str]:
    """Lowercase and strip button labels for quick keyword checks."""

    normalised = []
    for label in labels:
        clean = label.strip().lower()
        if clean:
            normalised.append(clean)
    return normalised


def _clamp_bounds(bounds: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp rectangle bounds to the image dimensions."""

    x0, y0, x1, y1 = bounds
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    return (x0, y0, x1, y1)


class CinematicDetector:
    """Stateful helper that tracks whether we appear to be inside a cinematic."""

    def __init__(
        self,
        capture_config: CaptureConfig,
        *,
        downscale_width: int = 256,
        menu_unusable_streak_required: int = 2,
        playing_diff_threshold: float = 0.05,
        paused_diff_threshold: float = 0.005,
        tabs_skip_threshold: float = 0.08,
        primary_skip_threshold: float = 0.22,
        pin_bright_threshold: float = 0.5,
        tabs_bright_threshold: float = 0.12,
        primary_bright_threshold: float = 0.25,
    ) -> None:
        """Initialise the detector.

        Args:
            capture_config: Capture configuration (used for split ratios).
            downscale_width: Width to downscale frames to for diffing.
            menu_unusable_streak_required: Number of consecutive frames with an
                unusable menu pane before we assume fullscreen cinematics.
            playing_diff_threshold: Mean absolute luma diff above which we treat
                the frame as actively playing.
            paused_diff_threshold: Mean absolute luma diff below which we treat
                the frame as paused. Values between paused and playing are left
                as UNKNOWN so callers can decide how cautiously to probe.
            tabs_skip_threshold: Luma std-dev threshold in the tabs ROI that
                indicates the Skip chip is visible after pausing a fullscreen
                cinematic.
            primary_skip_threshold: Equivalent threshold for the primary-region
                Skip chip (gacha/key cinematics).
        """
        self._capture_config = capture_config
        self._downscale_width = downscale_width
        self._menu_unusable_streak_required = menu_unusable_streak_required
        self._playing_diff_threshold = playing_diff_threshold
        self._paused_diff_threshold = paused_diff_threshold
        self._tabs_skip_threshold = tabs_skip_threshold
        self._primary_skip_threshold = primary_skip_threshold
        self._pin_bright_threshold = pin_bright_threshold
        self._tabs_bright_threshold = tabs_bright_threshold
        self._primary_bright_threshold = primary_bright_threshold

        self._prev_luma: Optional[np.ndarray] = None
        self._prev_primary_luma: Optional[np.ndarray] = None
        self._menu_unusable_streak = 0

    def observe(self, observation: CinematicObservation) -> CinematicDetectionResult:
        """Process a single frame and return the updated cinematic diagnosis."""

        if not isinstance(observation.image, Image.Image):
            raise TypeError("observation.image must be a PIL.Image.Image instance")

        luma = _downscale_luma(observation.image, width=self._downscale_width)
        if self._prev_luma is None or self._prev_luma.shape != luma.shape:
            diff_score = 0.0
        else:
            diff_score = float(np.mean(np.abs(luma - self._prev_luma)))
        self._prev_luma = luma

        menu_hint = observation.menu_is_usable
        if menu_hint is False:
            self._menu_unusable_streak += 1
        elif menu_hint is True:
            self._menu_unusable_streak = 0

        width, height = observation.image.size
        split = self._capture_config.split
        left_pin_px = int(round(width * split.left_pin_ratio))
        primary_px = int(round(width * split.primary_ratio))
        menus_px = int(round(width * split.menus_ratio))
        primary_x0 = left_pin_px
        primary_x1 = primary_x0 + primary_px
        tabs_x0 = left_pin_px + primary_px + menus_px
        tabs_x1 = width

        tabs_skip_bounds = (
            tabs_x0,
            int(round(height * 0.7)),
            tabs_x1,
            height,
        )
        primary_skip_bounds = (
            primary_x0 + int(round(primary_px * 0.6)),
            int(round(height * 0.6)),
            primary_x1,
            height,
        )
        tabs_crop = observation.image.crop(
            _clamp_bounds(tabs_skip_bounds, width, height)
        ).convert("L")
        tabs_arr = np.asarray(tabs_crop, dtype=np.float32) / 255.0
        if tabs_arr.size:
            tabs_mean = float(tabs_arr.mean())
            tabs_std = float(tabs_arr.std())
            tabs_bright_ratio = float((tabs_arr >= 0.75).mean())
        else:
            tabs_mean = tabs_std = tabs_bright_ratio = 0.0

        primary_crop = observation.image.crop(
            _clamp_bounds(primary_skip_bounds, width, height)
        ).convert("L")
        primary_arr = np.asarray(primary_crop, dtype=np.float32) / 255.0
        if primary_arr.size:
            primary_mean = float(primary_arr.mean())
            primary_std = float(primary_arr.std())
            primary_bright_ratio = float((primary_arr >= 0.75).mean())
            if self._prev_primary_luma is not None and self._prev_primary_luma.shape == primary_arr.shape:
                primary_diff_score = float(np.mean(np.abs(primary_arr - self._prev_primary_luma)))
            else:
                primary_diff_score = 0.0
            self._prev_primary_luma = primary_arr
        else:
            primary_mean = primary_std = primary_bright_ratio = 0.0
            primary_diff_score = 0.0
            self._prev_primary_luma = None

        pin_width = max(1, int(round(width * self._capture_config.split.left_pin_ratio * 0.9)))
        pin_height = max(1, int(round(height * 0.15)))
        pin_bounds = _clamp_bounds((0, 0, pin_width, pin_height), width, height)
        pin_mean, pin_std = _roi_luma_stats(observation.image, pin_bounds)
        pin_crop = observation.image.crop(pin_bounds).convert("L")
        pin_arr = np.asarray(pin_crop, dtype=np.float32) / 255.0
        pin_bright_ratio = float((pin_arr >= 0.8).mean()) if pin_arr.size else 0.0
        pin_present = pin_bright_ratio >= self._pin_bright_threshold and pin_mean >= 0.6 and pin_std >= 0.05

        labels_norm = _normalise_labels(observation.button_labels)
        skip_label_hint = any("skip" in label for label in labels_norm)

        skip_hint_tabs = (
            tabs_std >= self._tabs_skip_threshold and tabs_bright_ratio >= self._tabs_bright_threshold
        )
        skip_hint_primary = (
            primary_std >= self._primary_skip_threshold
            and primary_bright_ratio >= self._primary_bright_threshold
        )

        if skip_label_hint:
            if not skip_hint_tabs and not skip_hint_primary:
                # Fall back to the most likely region if the heuristics are inconclusive.
                skip_hint_primary = True

        if skip_hint_primary:
            kind = CinematicKind.PRIMARY
        elif (
            not pin_present
            and (skip_hint_tabs or self._menu_unusable_streak >= self._menu_unusable_streak_required)
        ):
            kind = CinematicKind.FULLSCREEN
        elif skip_hint_tabs:
            kind = CinematicKind.FULLSCREEN
        elif pin_present:
            kind = CinematicKind.NONE
        else:
            kind = CinematicKind.NONE

        if kind is CinematicKind.PRIMARY:
            motion_score = max(primary_diff_score, diff_score)
        else:
            motion_score = diff_score

        if kind is CinematicKind.NONE:
            playback = PlaybackState.UNKNOWN
        else:
            if motion_score >= self._playing_diff_threshold:
                playback = PlaybackState.PLAYING
            elif motion_score <= self._paused_diff_threshold:
                playback = PlaybackState.PAUSED
            else:
                playback = PlaybackState.UNKNOWN

        return CinematicDetectionResult(
            kind=kind,
            playback=playback,
            diff_score=diff_score,
            skip_hint_primary=skip_hint_primary,
            skip_hint_tabs=skip_hint_tabs,
            skip_label_hint=skip_label_hint,
            menu_unusable_streak=self._menu_unusable_streak,
            pin_present=pin_present,
            pin_bright_ratio=pin_bright_ratio,
            primary_diff_score=primary_diff_score,
        )
