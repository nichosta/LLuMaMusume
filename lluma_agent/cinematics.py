"""Utilities for detecting cinematic or unstable screen states via frame diffs.

The updated detector focuses on lightweight image differencing so we can pause
turn processing whenever the capture stream is still in motion.  Instead of
relying on UI affordances such as pin or skip buttons, the detector analyses
frame-to-frame deltas alongside a handful of image heuristics:

* Global and primary-region luma differences determine whether the scene is
  still animating.
* Very bright or very dark loading screens are explicitly flagged so we do not
  release control while the game is transitioning.
* Stable UI anchors (bottom button clusters) suppress false positives from
  "aggressive" menus that retain static overlays while animating backgrounds.

The caller feeds raw PIL images plus optional hints (menu usability, button
labels).  The detector smooths transient noise by keeping the last active
cinematic kind and exposing motion scores so higher level coordination can
impose minimum low-motion streaks before releasing control.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple

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
    loading_screen: bool = False
    anchor_stable_regions: int = 0
    anchor_stable_ratio: float = 0.0
    anchor_max_diff: float = 0.0


@dataclass(slots=True)
class _AnchorMetrics:
    """Summary of stability measurements for bottom UI anchor regions."""

    stable_regions: int = 0
    stable_ratio: float = 0.0
    max_diff: float = 0.0
    mean_brightness: float = 0.0


def _downscale_luma(image: Image.Image, *, width: int) -> np.ndarray:
    """Convert the frame to a small luma buffer for inexpensive diffing."""

    if image.width > width:
        ratio = width / image.width
        height = max(1, int(round(image.height * ratio)))
        resized = image.resize((width, height), Image.BILINEAR)
    else:
        resized = image
    luma = np.asarray(resized.convert("L"), dtype=np.float32) / 255.0
    return luma


def _rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    """Compute luma from an RGB float array in the range [0, 1]."""

    if rgb.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ).astype(np.float32)


def _clamp_bounds(
    bounds: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
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
        playing_diff_threshold: float = 0.06,
        paused_diff_threshold: float = 0.01,
        primary_motion_threshold: float = 0.035,
        primary_bias_ratio: float = 0.65,
        anchor_diff_threshold: float = 0.08,
        anchor_required: int = 1,
        anchor_ratio_threshold: float = 0.25,
        anchor_brightness_min: float = 0.2,
        loading_sample_width: int = 96,
        loading_white_luma_min: float = 0.72,
        loading_white_std_max: float = 0.08,
        loading_bright_ratio_min: float = 0.55,
        loading_saturation_max: float = 0.09,
        loading_color_range_max: float = 0.32,
        loading_black_luma_max: float = 0.05,
        loading_black_std_max: float = 0.015,
    ) -> None:
        """Initialise the detector with diff thresholds and heuristics."""

        self._capture_config = capture_config
        self._downscale_width = downscale_width
        self._playing_diff_threshold = playing_diff_threshold
        self._paused_diff_threshold = paused_diff_threshold
        self._primary_motion_threshold = primary_motion_threshold
        self._primary_bias_ratio = primary_bias_ratio
        self._anchor_diff_threshold = anchor_diff_threshold
        self._anchor_required = anchor_required
        self._anchor_ratio_required = anchor_ratio_threshold
        self._anchor_brightness_min = anchor_brightness_min
        self._loading_sample_width = loading_sample_width
        self._loading_white_luma_min = loading_white_luma_min
        self._loading_white_std_max = loading_white_std_max
        self._loading_bright_ratio_min = loading_bright_ratio_min
        self._loading_saturation_max = loading_saturation_max
        self._loading_color_range_max = loading_color_range_max
        self._loading_black_luma_max = loading_black_luma_max
        self._loading_black_std_max = loading_black_std_max

        self._prev_luma: Optional[np.ndarray] = None
        self._prev_primary_luma: Optional[np.ndarray] = None
        self._menu_unusable_streak = 0
        self._last_active_kind: CinematicKind = CinematicKind.NONE

        # Regions (normalised) that typically host persistent buttons.
        self._anchor_regions: tuple[Tuple[float, float, float, float], ...] = (
            (0.05, 0.30, 0.78, 0.94),
            (0.33, 0.62, 0.78, 0.94),
            (0.66, 0.95, 0.78, 0.94),
        )

    def observe(self, observation: CinematicObservation) -> CinematicDetectionResult:
        """Process a single frame and return the updated cinematic diagnosis."""

        if not isinstance(observation.image, Image.Image):
            raise TypeError("observation.image must be a PIL.Image.Image instance")

        luma = _downscale_luma(observation.image, width=self._downscale_width)
        prev_luma = self._prev_luma
        anchor_metrics = _AnchorMetrics()

        if prev_luma is None or prev_luma.shape != luma.shape:
            diff_score = 0.0
        else:
            diff_map = np.abs(luma - prev_luma)
            diff_score = float(diff_map.mean())
            anchor_metrics = self._compute_anchor_metrics(diff_map, luma)
        self._prev_luma = luma

        menu_hint = observation.menu_is_usable
        if menu_hint is False:
            self._menu_unusable_streak += 1
        elif menu_hint is True:
            self._menu_unusable_streak = 0

        primary_diff_score = self._compute_primary_diff(observation.image)
        loading_screen = self._detect_loading_screen(observation.image)

        anchor_override = (
            anchor_metrics.stable_regions >= self._anchor_required
            and anchor_metrics.stable_ratio >= self._anchor_ratio_required
            and anchor_metrics.mean_brightness >= self._anchor_brightness_min
        )

        motion_score = max(diff_score, primary_diff_score)
        primary_bias = (
            primary_diff_score >= self._primary_motion_threshold
            and primary_diff_score >= diff_score * self._primary_bias_ratio
        )

        if loading_screen:
            kind = CinematicKind.FULLSCREEN
            self._last_active_kind = CinematicKind.FULLSCREEN
        elif motion_score >= self._playing_diff_threshold and not anchor_override:
            if menu_hint is True and primary_bias:
                kind = CinematicKind.PRIMARY
            else:
                kind = CinematicKind.FULLSCREEN
            self._last_active_kind = kind
        elif motion_score <= self._paused_diff_threshold and self._last_active_kind is not CinematicKind.NONE:
            kind = self._last_active_kind
        elif anchor_override or motion_score <= self._paused_diff_threshold:
            kind = CinematicKind.NONE
            self._last_active_kind = CinematicKind.NONE
        else:
            if self._last_active_kind is not CinematicKind.NONE:
                kind = self._last_active_kind
            else:
                kind = CinematicKind.FULLSCREEN
                self._last_active_kind = kind

        if kind is CinematicKind.NONE:
            playback = PlaybackState.UNKNOWN
        else:
            effective_motion = motion_score if not loading_screen else self._playing_diff_threshold
            if not loading_screen and effective_motion <= self._paused_diff_threshold:
                playback = PlaybackState.PAUSED
            elif effective_motion >= self._playing_diff_threshold:
                playback = PlaybackState.PLAYING
            else:
                playback = PlaybackState.UNKNOWN

        pin_present = anchor_override and not loading_screen
        pin_bright_ratio = 1.0 if pin_present else 0.0

        return CinematicDetectionResult(
            kind=kind,
            playback=playback,
            diff_score=diff_score,
            skip_hint_primary=False,
            skip_hint_tabs=False,
            skip_label_hint=False,
            menu_unusable_streak=self._menu_unusable_streak,
            pin_present=pin_present,
            pin_bright_ratio=pin_bright_ratio,
            primary_diff_score=primary_diff_score,
            loading_screen=loading_screen,
            anchor_stable_regions=anchor_metrics.stable_regions,
            anchor_stable_ratio=anchor_metrics.stable_ratio,
            anchor_max_diff=anchor_metrics.max_diff,
        )

    def _compute_primary_diff(self, image: Image.Image) -> float:
        """Return mean luma diff for the primary gameplay pane."""

        width, height = image.size
        bounds = self._primary_bounds(width, height)
        x0, y0, x1, y1 = bounds
        if x1 <= x0 or y1 <= y0:
            self._prev_primary_luma = None
            return 0.0

        primary_crop = image.crop(bounds).convert("L")
        primary_arr = np.asarray(primary_crop, dtype=np.float32) / 255.0
        if primary_arr.size == 0:
            self._prev_primary_luma = None
            return 0.0

        prev_primary = self._prev_primary_luma
        if prev_primary is not None and prev_primary.shape == primary_arr.shape:
            diff = float(np.mean(np.abs(primary_arr - prev_primary)))
        else:
            diff = 0.0
        self._prev_primary_luma = primary_arr
        return diff

    def _primary_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        if not self._capture_config.split.enabled:
            return (0, 0, width, height)
        split = self._capture_config.split
        left = int(round(width * split.left_pin_ratio))
        primary_width = int(round(width * split.primary_ratio))
        right = left + primary_width
        return _clamp_bounds((left, 0, right, height), width, height)

    def _detect_loading_screen(self, image: Image.Image) -> bool:
        """Detect white/pale or black loading screens via colour statistics."""

        sample = self._sample_rgb(image, self._loading_sample_width)
        if sample.size == 0:
            return False
        luma = _rgb_to_luma(sample)
        mean = float(luma.mean())
        std = float(luma.std())
        bright_ratio = float((luma >= 0.85).mean())
        dark_ratio = float((luma <= 0.1).mean())
        saturation = float(np.mean(sample.max(axis=-1) - sample.min(axis=-1)))
        color_range = float(sample.max() - sample.min())

        if mean <= self._loading_black_luma_max and std <= self._loading_black_std_max:
            return True
        if dark_ratio >= 0.9 and std <= self._loading_black_std_max * 1.5:
            return True
        if (
            bright_ratio >= self._loading_bright_ratio_min
            and std <= self._loading_white_std_max
            and saturation <= self._loading_saturation_max
        ):
            return True
        if (
            mean >= self._loading_white_luma_min
            and std <= self._loading_white_std_max
            and color_range <= self._loading_color_range_max
        ):
            return True

        # Inspect the central region as well; primary-region loading screens keep
        # the corners bright while the centre remains pale and almost static.
        h, w = luma.shape
        if h >= 4 and w >= 4:
            y0 = h // 4
            y1 = h - y0
            x0 = w // 4
            x1 = w - x0
            centre = luma[y0:y1, x0:x1]
            if centre.size:
                centre_mean = float(centre.mean())
                centre_std = float(centre.std())
                if (
                    centre_mean >= self._loading_white_luma_min * 0.95
                    and centre_std <= self._loading_white_std_max
                    and saturation <= self._loading_saturation_max * 1.2
                ):
                    return True
        return False

    def _sample_rgb(self, image: Image.Image, width: int) -> np.ndarray:
        """Downscale the frame for inexpensive RGB statistics."""

        if image.width <= 0 or image.height <= 0:
            return np.empty((0, 0, 3), dtype=np.float32)
        if image.width > width:
            ratio = width / image.width
            height = max(1, int(round(image.height * ratio)))
            resized = image.resize((width, height), Image.BILINEAR)
        else:
            resized = image
        return np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0

    def _compute_anchor_metrics(
        self, diff_map: np.ndarray, luma: np.ndarray
    ) -> _AnchorMetrics:
        """Measure stability across button anchor regions."""

        height, width = diff_map.shape
        stable = 0
        diffs: list[float] = []
        brightness_values: list[float] = []

        for x0_r, x1_r, y0_r, y1_r in self._anchor_regions:
            x0 = int(round(x0_r * width))
            x1 = int(round(x1_r * width))
            y0 = int(round(y0_r * height))
            y1 = int(round(y1_r * height))
            if x1 <= x0 or y1 <= y0:
                continue
            region_diff = diff_map[y0:y1, x0:x1]
            region_luma = luma[y0:y1, x0:x1]
            if region_diff.size == 0:
                continue
            diff_mean = float(region_diff.mean())
            brightness = float(region_luma.mean())
            diffs.append(diff_mean)
            brightness_values.append(brightness)
            if (
                diff_mean <= self._anchor_diff_threshold
                and brightness >= self._anchor_brightness_min
            ):
                stable += 1

        if not diffs:
            return _AnchorMetrics()

        regions = len(diffs)
        stable_ratio = stable / regions
        max_diff = max(diffs)
        mean_brightness = float(sum(brightness_values) / regions)
        return _AnchorMetrics(
            stable_regions=stable,
            stable_ratio=stable_ratio,
            max_diff=max_diff,
            mean_brightness=mean_brightness,
        )
