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
    placement: Optional[WindowPlacement] = None  # If None, window is not repositioned
    scaling_factor: float = 1.5
    client_offset: Optional[Tuple[int, int]] = None  # (dx, dy) from outer top-left to client, in logical pixels


@dataclass(slots=True)
class SplitConfig:
    """Controls optional primary/menus image splitting."""

    enabled: bool = True
    left_pin_ratio: float = 0.1   # proportion of width for left pin (discarded)
    primary_ratio: float = 0.4    # proportion of width assigned to primary view
    menus_ratio: float = 0.4      # proportion of width assigned to menus view
    tabs_ratio: float = 0.1       # proportion of width assigned to right-side tabs
    
    def __post_init__(self) -> None:
        """Validate that ratios sum to 1.0."""
        total = (
            self.left_pin_ratio
            + self.primary_ratio
            + self.menus_ratio
            + self.tabs_ratio
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")


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


@dataclass(slots=True)
class AgentConfig:
    """Agent and LLM configuration."""

    model: str = "claude-haiku-4-5"
    memory_dir: Path = Path("memory")
    logs_dir: Path = Path("logs")
    max_memory_tokens: int = 32000
    max_context_tokens: int = 32000
    request_timeout_s: int = 30
    turn_post_padding_s: float = 5.0
    allow_skip_cinematics: bool = False
    thinking_enabled: bool = True
    thinking_budget_tokens: int = 16000
    max_tokens: int = 4096
    max_history_messages: int = 20  # Max messages in conversation history (0 = unlimited)


def load_configs(path: Optional[Path] = None) -> Tuple[WindowConfig, CaptureConfig, AgentConfig]:
    """Load window, capture, and agent configuration from YAML, falling back to defaults."""

    cfg_path = path or Path("config.yaml")
    window_cfg = WindowConfig()
    capture_cfg = CaptureConfig()
    agent_cfg = AgentConfig()

    if not cfg_path.exists():
        return window_cfg, capture_cfg, agent_cfg

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    window_data = raw.get("window", {})
    _apply_window_config(window_cfg, window_data)

    capture_data = raw.get("capture", {})
    capture_scaling_override = capture_data.get("scaling_factor")
    _apply_capture_config(capture_cfg, capture_data)
    if capture_scaling_override is None:
        capture_cfg.scaling_factor = window_cfg.scaling_factor

    agent_data = raw.get("agent", {})
    _apply_agent_config(agent_cfg, agent_data)

    return window_cfg, capture_cfg, agent_cfg


def _apply_window_config(config: WindowConfig, data: Dict) -> None:
    if not data:
        return

    title = data.get("window_title") or data.get("title")
    if title:
        config.title = title

    placement_data = data.get("placement")
    if placement_data and isinstance(placement_data, dict):
        # Only create placement if all required fields are present
        if all(key in placement_data for key in ("x", "y", "width", "height")):
            config.placement = WindowPlacement(
                x=int(placement_data["x"]),
                y=int(placement_data["y"]),
                width=int(placement_data["width"]),
                height=int(placement_data["height"]),
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
        
        # Handle new three-ratio structure
        if "left_pin_ratio" in split_data:
            config.split.left_pin_ratio = float(split_data["left_pin_ratio"])
        if "primary_ratio" in split_data:
            config.split.primary_ratio = float(split_data["primary_ratio"])
        if "menus_ratio" in split_data:
            config.split.menus_ratio = float(split_data["menus_ratio"])
        if "tabs_ratio" in split_data:
            config.split.tabs_ratio = float(split_data["tabs_ratio"])
            
        # Legacy support for old ratios structure
        ratios = split_data.get("ratios")
        if isinstance(ratios, dict):
            left_pin = ratios.get("left_pin")
            primary = ratios.get("primary")
            menus = ratios.get("menus")
            tabs = ratios.get("tabs")
            if left_pin is not None:
                config.split.left_pin_ratio = float(left_pin)
            if primary is not None:
                config.split.primary_ratio = float(primary)
            if menus is not None:
                config.split.menus_ratio = float(menus)
            if tabs is not None:
                config.split.tabs_ratio = float(tabs)

    validation_data = data.get("validation") or {}
    if validation_data:
        if "min_mean_luminance" in validation_data:
            config.validation.min_mean_luminance = float(validation_data["min_mean_luminance"])
        if "min_luminance_stddev" in validation_data:
            config.validation.min_luminance_stddev = float(validation_data["min_luminance_stddev"])

    retention_data = data.get("retention") or {}
    if retention_data and "max_captures" in retention_data:
        config.retention.max_captures = int(retention_data["max_captures"])


def _apply_agent_config(config: AgentConfig, data: Dict) -> None:
    if not data:
        return

    if "model" in data:
        config.model = str(data["model"])

    if "memory_dir" in data:
        config.memory_dir = Path(str(data["memory_dir"]))

    if "logs_dir" in data:
        config.logs_dir = Path(str(data["logs_dir"]))

    if "max_memory_tokens" in data:
        config.max_memory_tokens = int(data["max_memory_tokens"])

    if "max_context_tokens" in data:
        config.max_context_tokens = int(data["max_context_tokens"])

    if "request_timeout_s" in data:
        config.request_timeout_s = int(data["request_timeout_s"])

    if "turn_post_padding_s" in data:
        config.turn_post_padding_s = float(data["turn_post_padding_s"])

    if "allow_skip_cinematics" in data:
        config.allow_skip_cinematics = bool(data["allow_skip_cinematics"])

    if "thinking_enabled" in data:
        config.thinking_enabled = bool(data["thinking_enabled"])

    if "thinking_budget_tokens" in data:
        config.thinking_budget_tokens = int(data["thinking_budget_tokens"])

    if "max_tokens" in data:
        config.max_tokens = int(data["max_tokens"])

    if "max_history_messages" in data:
        config.max_history_messages = int(data["max_history_messages"])


__all__ = [
    "WindowPlacement",
    "WindowConfig",
    "SplitConfig",
    "CaptureValidationConfig",
    "RetentionConfig",
    "CaptureConfig",
    "AgentConfig",
    "load_configs",
]
