"""Input handling for Uma Musume agent tools.

Provides high-level input actions (pressButton, advanceDialogue, etc.) that
abstract away coordinates and timing. All actions validate window state before
execution and convert logical coordinates to physical screen pixels.
"""
from __future__ import annotations

import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .window import UmaWindow, WindowGeometry, WindowFocusError, WindowNotFoundError

Logger = logging.Logger

_IS_WINDOWS = sys.platform.startswith("win32")

if _IS_WINDOWS:
    try:
        import pyautogui  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyautogui is required on Windows hosts") from exc
else:  # pragma: no cover - tooling on non-Windows
    pyautogui = None  # type: ignore


class ButtonNotFoundError(RuntimeError):
    """Raised when a requested button is not present in the vision output."""


class ScrollbarNotFoundError(RuntimeError):
    """Raised when scrollbar is required but not detected."""


class WindowStateError(RuntimeError):
    """Raised when window is in an invalid state (closed, minimized, etc.)."""


@dataclass(slots=True)
class ButtonInfo:
    """Information about a detected button."""

    name: str  # Stripped name for matching
    full_label: str  # Original label from VLM
    bounds: Tuple[int, int, int, int]  # (x, y, width, height) in client logical pixels
    metadata: Dict[str, str]  # Parsed metadata tags


@dataclass(slots=True)
class ScrollbarInfo:
    """Scrollbar detection result."""

    track_bounds: Tuple[int, int, int, int]  # (x, y, width, height) in client logical pixels
    thumb_bounds: Tuple[int, int, int, int]
    can_scroll_up: bool
    can_scroll_down: bool
    thumb_ratio: float


@dataclass(slots=True)
class VisionOutput:
    """Complete vision processing result for one turn."""

    buttons: List[ButtonInfo]
    scrollbar: Optional[ScrollbarInfo]
    primary_center: Tuple[int, int]  # (x, y) in client logical pixels


@dataclass(slots=True)
class InputConfig:
    """Timing and behavior configuration for inputs."""

    jitter_min_ms: int = 20
    jitter_max_ms: int = 50
    click_duration_ms: int = 100
    key_duration_ms: int = 100


class InputHandler:
    """High-level input operations for the Uma Musume agent.

    Manages button clicks, dialogue advancement, keyboard inputs, and scrolling.
    All coordinates are converted from logical pixels to physical screen pixels
    before execution.
    """

    def __init__(
        self,
        window: UmaWindow,
        config: Optional[InputConfig] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self._window = window
        self._config = config or InputConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._vision_state: Optional[VisionOutput] = None
        self._geometry: Optional[WindowGeometry] = None

        if not _IS_WINDOWS:
            self._logger.warning("InputHandler instantiated on non-Windows; operations will fail")

    def update_vision_state(self, vision_output: VisionOutput, geometry: WindowGeometry) -> None:
        """Update the stored vision state and window geometry for the current turn."""
        self._vision_state = vision_output
        self._geometry = geometry
        self._logger.debug(
            "Vision state updated: %d buttons, scrollbar=%s",
            len(vision_output.buttons),
            "present" if vision_output.scrollbar else "absent",
        )

    def press_button(self, name: str) -> None:
        """Click the center of the button with the given name.

        Args:
            name: Button name to search for (case-insensitive, metadata stripped).

        Raises:
            ButtonNotFoundError: If no button with this name is found.
            WindowStateError: If the window is closed or cannot be focused.
        """
        self._ensure_ready()
        assert self._vision_state is not None  # noqa: S101 - ensured by _ensure_ready
        assert self._geometry is not None  # noqa: S101

        # Find button (case-insensitive match)
        button = self._find_button(name)
        if button is None:
            available = ", ".join(b.name for b in self._vision_state.buttons)
            raise ButtonNotFoundError(
                f"Button '{name}' not found. Available buttons: {available or '(none)'}"
            )

        # Calculate center in logical pixels
        x, y, w, h = button.bounds
        center_x = x + w // 2
        center_y = y + h // 2

        # Convert to physical screen pixels
        screen_x, screen_y = self._geometry.client_area.logical_to_screen(center_x, center_y)

        self._logger.info(
            "Clicking button '%s' at logical (%d, %d) -> screen (%d, %d)",
            button.name,
            center_x,
            center_y,
            screen_x,
            screen_y,
        )

        self._click(screen_x, screen_y)

    def advance_dialogue(self) -> None:
        """Click the center of the primary area to advance dialogue.

        Raises:
            WindowStateError: If the window is closed or cannot be focused.
        """
        self._ensure_ready()
        assert self._vision_state is not None  # noqa: S101
        assert self._geometry is not None  # noqa: S101

        logical_x, logical_y = self._vision_state.primary_center
        screen_x, screen_y = self._geometry.client_area.logical_to_screen(logical_x, logical_y)

        self._logger.info(
            "Advancing dialogue at logical (%d, %d) -> screen (%d, %d)",
            logical_x,
            logical_y,
            screen_x,
            screen_y,
        )

        self._click(screen_x, screen_y)

    def back(self) -> None:
        """Press ESC to go back.

        Raises:
            WindowStateError: If the window is closed or cannot be focused.
        """
        self._ensure_ready()
        self._logger.info("Pressing ESC (back)")
        self._press_key("esc")

    def confirm(self) -> None:
        """Press SPACE to confirm.

        Raises:
            WindowStateError: If the window is closed or cannot be focused.
        """
        self._ensure_ready()
        self._logger.info("Pressing SPACE (confirm)")
        self._press_key("space")

    def scroll_up(self) -> None:
        """Position cursor over scrollbar and press Z to scroll up.

        Raises:
            ScrollbarNotFoundError: If no scrollbar is detected.
            WindowStateError: If the window is closed or cannot be focused.
        """
        self._ensure_ready()
        assert self._vision_state is not None  # noqa: S101
        assert self._geometry is not None  # noqa: S101

        if self._vision_state.scrollbar is None:
            raise ScrollbarNotFoundError("Cannot scroll: no scrollbar detected in vision output")

        # Position cursor at center of scrollbar track
        track_x, track_y, track_w, track_h = self._vision_state.scrollbar.track_bounds
        center_x = track_x + track_w // 2
        center_y = track_y + track_h // 2
        screen_x, screen_y = self._geometry.client_area.logical_to_screen(center_x, center_y)

        self._logger.info("Scrolling up: cursor at screen (%d, %d), pressing Z", screen_x, screen_y)

        # Move cursor and press key
        self._move_to(screen_x, screen_y)
        self._press_key("z")

    def scroll_down(self) -> None:
        """Position cursor over scrollbar and press C to scroll down.

        Raises:
            ScrollbarNotFoundError: If no scrollbar is detected.
            WindowStateError: If the window is closed or cannot be focused.
        """
        self._ensure_ready()
        assert self._vision_state is not None  # noqa: S101
        assert self._geometry is not None  # noqa: S101

        if self._vision_state.scrollbar is None:
            raise ScrollbarNotFoundError("Cannot scroll: no scrollbar detected in vision output")

        # Position cursor at center of scrollbar track
        track_x, track_y, track_w, track_h = self._vision_state.scrollbar.track_bounds
        center_x = track_x + track_w // 2
        center_y = track_y + track_h // 2
        screen_x, screen_y = self._geometry.client_area.logical_to_screen(center_x, center_y)

        self._logger.info("Scrolling down: cursor at screen (%d, %d), pressing C", screen_x, screen_y)

        # Move cursor and press key
        self._move_to(screen_x, screen_y)
        self._press_key("c")

    def _ensure_ready(self) -> None:
        """Validate that window is ready and vision state is available."""
        if self._vision_state is None or self._geometry is None:
            raise WindowStateError("No vision state available; call update_vision_state first")

        # Ensure window is focused and accessible
        try:
            self._window.focus()
        except (WindowNotFoundError, WindowFocusError) as exc:
            raise WindowStateError("Window is not accessible") from exc

    def _find_button(self, name: str) -> Optional[ButtonInfo]:
        """Find a button by name (case-insensitive)."""
        if self._vision_state is None:
            return None

        search_name = name.strip().lower()
        for button in self._vision_state.buttons:
            if button.name.lower() == search_name:
                return button
        return None

    def _click(self, screen_x: int, screen_y: int) -> None:
        """Perform a mouse click at the given screen coordinates."""
        if pyautogui is None:
            raise RuntimeError("pyautogui not available")

        self._apply_jitter()
        pyautogui.moveTo(screen_x, screen_y)
        duration_sec = self._config.click_duration_ms / 1000.0
        pyautogui.click(duration=duration_sec)

    def _move_to(self, screen_x: int, screen_y: int) -> None:
        """Move cursor to the given screen coordinates."""
        if pyautogui is None:
            raise RuntimeError("pyautogui not available")

        self._apply_jitter()
        pyautogui.moveTo(screen_x, screen_y)

    def _press_key(self, key: str) -> None:
        """Press and release a keyboard key."""
        if pyautogui is None:
            raise RuntimeError("pyautogui not available")

        self._apply_jitter()
        duration_sec = self._config.key_duration_ms / 1000.0
        pyautogui.press(key, presses=1, interval=duration_sec)

    def _apply_jitter(self) -> None:
        """Apply random timing jitter to avoid mechanical patterns."""
        jitter_ms = random.randint(self._config.jitter_min_ms, self._config.jitter_max_ms)
        time.sleep(jitter_ms / 1000.0)


def parse_button_label(label: str) -> Tuple[str, Dict[str, str]]:
    """Parse a VLM button label into name and metadata.

    Format: "ButtonName|key=value|key2=value2"
    Escaping: \| and \= for literal pipe/equals in values.

    Returns:
        (name, metadata_dict)
    """
    # Split on unescaped pipes
    parts = re.split(r"(?<!\\)\|", label)
    name = parts[0].strip() if parts else ""
    metadata: Dict[str, str] = {}

    for part in parts[1:]:
        if "=" in part:
            key, _, value = part.partition("=")
            # Unescape
            key = key.replace("\\|", "|").replace("\\=", "=").strip()
            value = value.replace("\\|", "|").replace("\\=", "=").strip()
            metadata[key] = value

    return name, metadata


__all__ = [
    "ButtonInfo",
    "ButtonNotFoundError",
    "InputConfig",
    "InputHandler",
    "ScrollbarInfo",
    "ScrollbarNotFoundError",
    "VisionOutput",
    "WindowStateError",
    "parse_button_label",
]
