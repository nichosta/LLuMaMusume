"""Client-area capture utilities for Uma Musume."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .config import CaptureConfig
from .window import (
    ClientArea,
    UmaWindow,
    WindowFocusError,
    WindowGeometry,
    WindowNotFoundError,
)

try:  # pragma: no cover - import validated at runtime
    import mss
except Exception:  # pragma: no cover
    mss = None

Logger = logging.Logger


@dataclass(slots=True)
class CaptureValidation:
    """Basic image statistics used to validate captured content."""

    mean_luminance: float
    stddev_luminance: float
    size_px: Tuple[int, int]


@dataclass(slots=True)
class CaptureResult:
    """Return value from a capture call."""

    turn_id: str
    geometry: WindowGeometry
    raw_image: Image.Image
    raw_path: Path
    validation: CaptureValidation
    primary_image: Optional[Image.Image] = None
    primary_path: Optional[Path] = None
    menus_image: Optional[Image.Image] = None
    menus_path: Optional[Path] = None
    tabs_image: Optional[Image.Image] = None
    tabs_path: Optional[Path] = None


class CaptureError(RuntimeError):
    """Raised when capture fails validation or system calls."""


class CaptureManager:
    """Coordinates screen capture of the Uma Musume client window."""

    def __init__(self, window: UmaWindow, config: CaptureConfig, logger: Optional[Logger] = None) -> None:
        self._window = window
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._output_dir = config.output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, turn_id: str, *, reposition: bool = False) -> CaptureResult:
        """Capture the current client area for a turn."""

        try:
            geometry = self._window.ensure_ready(reposition=reposition)
        except (WindowNotFoundError, WindowFocusError) as exc:
            raise CaptureError(str(exc)) from exc
        bbox = self._build_bbox(geometry.client_area)
        screenshot = self._grab(bbox)
        raw_image = self._to_image(screenshot)
        validation = self._validate_capture(raw_image, geometry)

        raw_path = self._save(raw_image, turn_id)
        primary_image: Optional[Image.Image]
        primary_path: Optional[Path]
        menus_image: Optional[Image.Image]
        menus_path: Optional[Path]
        tabs_image: Optional[Image.Image]
        tabs_path: Optional[Path]
        (
            primary_image,
            menus_image,
            tabs_image,
        ) = (None, None, None)
        primary_path = menus_path = tabs_path = None

        if self._config.split.enabled:
            primary_image, menus_image, tabs_image = self._split_capture(raw_image, geometry.client_area)
            if primary_image is not None:
                primary_path = self._save(primary_image, f"{turn_id}-primary")
            if menus_image is not None:
                menus_path = self._save(menus_image, f"{turn_id}-menus")
            if tabs_image is not None:
                tabs_path = self._save(tabs_image, f"{turn_id}-tabs")

        self._enforce_retention()

        return CaptureResult(
            turn_id=turn_id,
            geometry=geometry,
            raw_image=raw_image,
            raw_path=raw_path,
            validation=validation,
            primary_image=primary_image,
            primary_path=primary_path,
            menus_image=menus_image,
            menus_path=menus_path,
            tabs_image=tabs_image,
            tabs_path=tabs_path,
        )

    def _build_bbox(self, client_area: ClientArea) -> dict:
        origin_x, origin_y = client_area.screen_origin
        width, height = client_area.size_physical
        return {"left": origin_x, "top": origin_y, "width": width, "height": height}

    def _grab(self, bbox: dict) -> "mss.base.ScreenShot":
        self._logger.debug("Capturing screen region %s", bbox)
        if mss is None:
            raise CaptureError("mss library is not available; install dependency before capturing")
        with mss.mss() as sct:
            return sct.grab(bbox)

    def _to_image(self, screenshot: "mss.base.ScreenShot") -> Image.Image:
        return Image.frombytes("RGB", screenshot.size, screenshot.rgb)

    def _validate_capture(self, image: Image.Image, geometry: WindowGeometry) -> CaptureValidation:
        width_expected, height_expected = geometry.client_area.size_physical
        if image.width != width_expected or image.height != height_expected:
            raise CaptureError(
                f"Capture size {image.width}x{image.height} did not match client area {width_expected}x{height_expected}"
            )

        array = np.asarray(image, dtype=np.float32)
        luminance = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
        mean_luma = float(luminance.mean())
        std_luma = float(luminance.std())

        validation_cfg = self._config.validation
        if mean_luma < validation_cfg.min_mean_luminance:
            raise CaptureError(f"Capture luminance too low ({mean_luma:.2f} < {validation_cfg.min_mean_luminance})")
        if std_luma < validation_cfg.min_luminance_stddev:
            raise CaptureError(
                f"Capture appears uniform (std {std_luma:.2f} < {validation_cfg.min_luminance_stddev})"
            )

        return CaptureValidation(mean_luminance=mean_luma, stddev_luminance=std_luma, size_px=(image.width, image.height))

    def _split_capture(
        self, image: Image.Image, client_area: ClientArea
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
        split_cfg = self._config.split
        if not split_cfg.enabled:
            return None, None, None

        width, height = image.size
        
        # Calculate section boundaries using simple ratios
        left_pin_end = int(round(width * split_cfg.left_pin_ratio))
        primary_end = int(round(width * (split_cfg.left_pin_ratio + split_cfg.primary_ratio)))
        menus_end = int(
            round(
                width
                * (
                    split_cfg.left_pin_ratio
                    + split_cfg.primary_ratio
                    + split_cfg.menus_ratio
                )
            )
        )
        
        # Ensure boundaries are valid
        left_pin_end = max(0, min(left_pin_end, width))
        primary_end = max(left_pin_end, min(primary_end, width))
        menus_end = max(primary_end, min(menus_end, width))
        
        # Crop the sections (discard left pin, keep primary and menus)
        primary_box = (left_pin_end, 0, primary_end, height)
        menus_box = (primary_end, 0, menus_end, height)
        tabs_box = (menus_end, 0, width, height)

        primary_image = image.crop(primary_box) if primary_box[2] > primary_box[0] else None
        menus_image = image.crop(menus_box) if menus_box[2] > menus_box[0] else None
        tabs_image = image.crop(tabs_box) if tabs_box[2] > tabs_box[0] else None
        return primary_image, menus_image, tabs_image

    def _save(self, image: Image.Image, stem: str) -> Path:
        safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
        path = self._output_dir / f"{safe_stem}.png"
        image.save(path, format="PNG")
        self._logger.debug("Saved capture to %s", path)
        return path

    def _enforce_retention(self) -> None:
        max_captures = self._config.retention.max_captures
        if max_captures <= 0:
            return

        excluded_suffixes = ("-primary.png", "-menus.png", "-tabs.png")
        raw_files = sorted(
            [p for p in self._output_dir.glob("*.png") if not p.name.endswith(excluded_suffixes)],
            key=lambda p: p.stat().st_mtime,
        )
        excess = len(raw_files) - max_captures
        for victim in raw_files[:excess]:
            self._delete_turn_files(victim)

    def _delete_turn_files(self, raw_file: Path) -> None:
        self._logger.debug("Deleting capture set rooted at %s", raw_file)
        raw_file.unlink(missing_ok=True)
        stem = raw_file.stem
        for suffix in ("-primary.png", "-menus.png", "-tabs.png"):
            candidate = raw_file.with_name(f"{stem}{suffix}")
            candidate.unlink(missing_ok=True)
