"""OS layer for LLuMa Musume.

This package coordinates window management (finding, focusing, and sizing the
Uma Musume client), client-area captures that are passed to the Vision pipeline,
and input handling for agent tool calls.

Implementations target Windows 11 per AGENTS.md while remaining importable from
other platforms for tooling and tests.

IMPORTANT: Call set_dpi_aware() once at program startup before any window,
capture, or input operations.
"""

from .capture import CaptureManager
from .config import CaptureConfig, WindowConfig, load_configs
from .input_handler import InputHandler
from .window import UmaWindow, set_dpi_aware

__all__ = [
    "CaptureConfig",
    "CaptureManager",
    "InputHandler",
    "UmaWindow",
    "WindowConfig",
    "load_configs",
    "set_dpi_aware",
]
