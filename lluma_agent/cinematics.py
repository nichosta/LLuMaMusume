"""Diff-focused cinematic detection heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from lluma_os.config import CaptureConfig


class CinematicKind(str, Enum):
    """High-level classification for gating decisions."""

    NONE = "none"
    CUTSCENE = "cutscene"
    LOADING = "loading"


class PlaybackState(str, Enum):
    """Lightweight state machine for motion based on frame diffs."""

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
    """Detector output derived from successive frame comparisons."""

    kind: CinematicKind
    playback: PlaybackState
    diff_score: float
    primary_diff_score: float
    changed_ratio: float
    is_loading_screen: bool
    aggressive_static: bool


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


class CinematicDetector:
    """Stateful helper that evaluates frame diffs to gate agent turns."""

    def __init__(
        self,
        capture_config: CaptureConfig,
        *,
        poll_interval_s: float = 1.0,
        downscale_width: int = 256,
        playing_diff_threshold: float = 0.02,
        paused_diff_threshold: float = 0.005,
        change_threshold: float = 0.04,
        bottom_region_ratio: float = 0.18,
        bottom_bright_value: float = 0.6,
        bottom_bright_ratio_threshold: float = 0.25,
        bottom_change_ratio_max: float = 0.7,
        loading_white_mean: float = 0.88,
        loading_white_std: float = 0.12,
        loading_white_bright_ratio: float = 0.6,
        loading_black_mean: float = 0.08,
        loading_black_std: float = 0.04,
        loading_black_dark_ratio: float = 0.6,
    ) -> None:
        self.poll_interval_s = poll_interval_s
        self._capture_config = capture_config
        self._downscale_width = downscale_width
        self._playing_diff_threshold = playing_diff_threshold
        self._paused_diff_threshold = paused_diff_threshold
        self._change_threshold = change_threshold
        self._bottom_region_ratio = bottom_region_ratio
        self._bottom_bright_value = bottom_bright_value
        self._bottom_bright_ratio_threshold = bottom_bright_ratio_threshold
        self._bottom_change_ratio_max = bottom_change_ratio_max
        self._loading_white_mean = loading_white_mean
        self._loading_white_std = loading_white_std
        self._loading_white_bright_ratio = loading_white_bright_ratio
        self._loading_black_mean = loading_black_mean
        self._loading_black_std = loading_black_std
        self._loading_black_dark_ratio = loading_black_dark_ratio

        self._prev_luma: Optional[np.ndarray] = None

    def observe(self, observation: CinematicObservation) -> CinematicDetectionResult:
        """Process a single frame and return the updated cinematic diagnosis."""

        if not isinstance(observation.image, Image.Image):
            raise TypeError("observation.image must be a PIL.Image.Image instance")

        luma = _downscale_luma(observation.image, width=self._downscale_width)
        diff_map: Optional[np.ndarray]
        if self._prev_luma is None or self._prev_luma.shape != luma.shape:
            diff_score = 0.0
            changed_ratio = 0.0
            diff_map = None
        else:
            diff_map = np.abs(luma - self._prev_luma)
            diff_score = float(diff_map.mean())
            changed_ratio = float((diff_map >= self._change_threshold).mean())
        self._prev_luma = luma

        primary_diff_score = self._compute_primary_diff(diff_map)
        is_loading_screen = self._detect_loading_screen(luma)
        aggressive_static = self._detect_aggressive_static(
            observation.menu_is_usable, luma, diff_map
        )

        if diff_score >= self._playing_diff_threshold:
            playback = PlaybackState.PLAYING
        elif diff_score <= self._paused_diff_threshold:
            playback = PlaybackState.PAUSED
        else:
            playback = PlaybackState.UNKNOWN

        if is_loading_screen:
            kind = CinematicKind.LOADING
        elif diff_map is not None and diff_score >= self._playing_diff_threshold and not aggressive_static:
            kind = CinematicKind.CUTSCENE
        else:
            kind = CinematicKind.NONE

        return CinematicDetectionResult(
            kind=kind,
            playback=playback,
            diff_score=diff_score,
            primary_diff_score=primary_diff_score,
            changed_ratio=changed_ratio,
            is_loading_screen=is_loading_screen,
            aggressive_static=aggressive_static,
        )

    def _compute_primary_diff(self, diff_map: Optional[np.ndarray]) -> float:
        if diff_map is None or diff_map.size == 0:
            return 0.0
        split = self._capture_config.split
        width = diff_map.shape[1]
        if not split.enabled:
            return float(diff_map.mean())
        left_pin_end = int(round(width * split.left_pin_ratio))
        primary_end = int(round(width * (split.left_pin_ratio + split.primary_ratio)))
        left_pin_end = max(0, min(left_pin_end, width))
        primary_end = max(left_pin_end, min(primary_end, width))
        primary_slice = diff_map[:, left_pin_end:primary_end]
        if primary_slice.size == 0:
            return float(diff_map.mean())
        return float(primary_slice.mean())

    def _detect_loading_screen(self, luma: np.ndarray) -> bool:
        if luma.size == 0:
            return False
        mean = float(luma.mean())
        std = float(luma.std())
        bright_ratio = float((luma >= self._bottom_bright_value).mean())
        dark_ratio = float((luma <= (1.0 - self._bottom_bright_value)).mean())
        is_white_loading = (
            mean >= self._loading_white_mean
            and std <= self._loading_white_std
            and bright_ratio >= self._loading_white_bright_ratio
        )
        is_black_loading = (
            mean <= self._loading_black_mean
            and std <= self._loading_black_std
            and dark_ratio >= self._loading_black_dark_ratio
        )
        return is_white_loading or is_black_loading

    def _detect_aggressive_static(
        self,
        menu_is_usable: Optional[bool],
        luma: np.ndarray,
        diff_map: Optional[np.ndarray],
    ) -> bool:
        if menu_is_usable:
            return True
        if diff_map is None or diff_map.size == 0:
            return False
        height = diff_map.shape[0]
        bottom_rows = max(1, int(round(height * self._bottom_region_ratio)))
        bottom_slice = diff_map[-bottom_rows:, :]
        bottom_changed_ratio = float((bottom_slice >= self._change_threshold).mean())
        bottom_luma = luma[-bottom_rows:, :]
        bottom_bright_ratio = float((bottom_luma >= self._bottom_bright_value).mean())
        return (
            bottom_bright_ratio >= self._bottom_bright_ratio_threshold
            and bottom_changed_ratio <= self._bottom_change_ratio_max
        )
