"""OS layer for LLuMa Musume.

This package coordinates window management (finding, focusing, and sizing the
Uma Musume client) and client-area captures that are passed to the Vision
pipeline. Implementations target Windows 11 per AGENTS.md while remaining
importable from other platforms for tooling and tests.
"""

from .config import CaptureConfig, WindowConfig, load_configs
from .window import UmaWindow
from .capture import CaptureManager

__all__ = [
    "CaptureConfig",
    "WindowConfig",
    "UmaWindow",
    "CaptureManager",
    "load_configs",
]
