"""Configuration structures shared by the OS module.

Values originate from `config.yaml` (preferred) or environment overrides and are
expressed in logical pixels unless stated otherwise. The logical pixel space is
what Vision and agent tooling consume; conversion to physical pixels happens in
specialised helpers that know about the active DPI scaling factor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


@dataclass(slots=True)
class WindowPlacement:
    """Desired outer window bounds in logical pixels."""

    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class WindowConfig:
    """Window discovery and placement parameters."""

    title: str = "Umamusume"
    placement: WindowPlacement = field(
        default_factory=lambda: WindowPlacement(x=0, y=0, width=1920, height=1080)
    )
    scaling_factor: float = 1.5
    client_offset: Optional[Tuple[int, int]] = None  # dx, dy from outer top-left to client


@dataclass(slots=True)
class SplitConfig:
    """Controls optional primary/menus image splitting."""

    enabled: bool = True
    left_pin_px: int = 0
    primary_ratio: float = 0.5  # proportion of width assigned to primary view
    menus_ratio: float = 0.5  # fallback; recomputed from width at runtime


@dataclass(slots=True)
class CaptureValidationConfig:
    """Validation thresholds to guard against blank or occluded captures."""

    min_mean_luminance: float = 5.0
    min_luminance_stddev: float = 1.5


@dataclass(slots=True)
class RetentionConfig:
    """Capture file retention policy."""

    max_captures: int = 200


@dataclass(slots=True)
class CaptureConfig:
    """Top-level capture configuration blob."""

    output_dir: Path = Path("captures")
    post_action_capture: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    validation: CaptureValidationConfig = field(default_factory=CaptureValidationConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    scaling_factor: float = 1.5


def load_configs(path: Optional[Path] = None) -> Tuple[WindowConfig, CaptureConfig]:
    """Load window and capture configuration from YAML, falling back to defaults."""

    cfg_path = path or Path("config.yaml")
    window_cfg = WindowConfig()
    capture_cfg = CaptureConfig()

    if not cfg_path.exists():
        return window_cfg, capture_cfg

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    window_data = raw.get("window", {})
    _apply_window_config(window_cfg, window_data)

    capture_data = raw.get("capture", {})
    capture_scaling_override = capture_data.get("scaling_factor")
    _apply_capture_config(capture_cfg, capture_data)
    if capture_scaling_override is None:
        capture_cfg.scaling_factor = window_cfg.scaling_factor

    return window_cfg, capture_cfg


def _apply_window_config(config: WindowConfig, data: Dict) -> None:
    if not data:
        return

    title = data.get("window_title") or data.get("title")
    if title:
        config.title = title

    placement_data = data.get("placement") or {}
    if placement_data:
        config.placement = WindowPlacement(
            x=int(placement_data.get("x", config.placement.x)),
            y=int(placement_data.get("y", config.placement.y)),
            width=int(placement_data.get("width", config.placement.width)),
            height=int(placement_data.get("height", config.placement.height)),
        )

    dpi_data = data.get("dpi") or {}
    scaling_factor = data.get("scaling_factor") or dpi_data.get("scaling_factor")
    if scaling_factor is not None:
        config.scaling_factor = float(scaling_factor)

    client_offset = data.get("client_offset")
    if client_offset is not None and isinstance(client_offset, (list, tuple)) and len(client_offset) >= 2:
        config.client_offset = (int(client_offset[0]), int(client_offset[1]))


def _apply_capture_config(config: CaptureConfig, data: Dict) -> None:
    if not data:
        return

    output_dir = data.get("output_dir")
    if output_dir:
        config.output_dir = Path(str(output_dir))

    post_action = data.get("post_action")
    if post_action is not None:
        config.post_action_capture = bool(post_action)

    scaling_factor = data.get("scaling_factor")
    if scaling_factor is not None:
        config.scaling_factor = float(scaling_factor)

    split_data = data.get("split") or {}
    if split_data:
        config.split.enabled = bool(split_data.get("enabled", config.split.enabled))
        if "left_pin_px" in split_data:
            config.split.left_pin_px = int(split_data["left_pin_px"])
        ratios = split_data.get("ratios")
        if isinstance(ratios, dict):
            primary = ratios.get("primary")
            menus = ratios.get("menus")
            if primary is not None:
                config.split.primary_ratio = float(primary)
            if menus is not None:
                config.split.menus_ratio = float(menus)

    validation_data = data.get("validation") or {}
    if validation_data:
        if "min_mean_luminance" in validation_data:
            config.validation.min_mean_luminance = float(validation_data["min_mean_luminance"])
        if "min_luminance_stddev" in validation_data:
            config.validation.min_luminance_stddev = float(validation_data["min_luminance_stddev"])

    retention_data = data.get("retention") or {}
    if retention_data and "max_captures" in retention_data:
        config.retention.max_captures = int(retention_data["max_captures"])


__all__ = [
    "WindowPlacement",
    "WindowConfig",
    "SplitConfig",
    "CaptureValidationConfig",
    "RetentionConfig",
    "CaptureConfig",
    "load_configs",
]
