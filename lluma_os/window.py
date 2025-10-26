"""Window management utilities tailored for Uma Musume on Windows.

IMPORTANT: Call set_dpi_aware() once at program startup before creating
any UmaWindow instances or performing window/screen operations. Without
DPI awareness, window coordinates and mouse operations will be incorrect
on displays with scaling != 100%.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import WindowConfig, WindowPlacement

Logger = logging.Logger

_IS_WINDOWS = sys.platform.startswith("win32")

if _IS_WINDOWS:
    try:
        import pygetwindow as gw  # type: ignore
    except Exception as exc:  # pragma: no cover - import error surfaced once
        raise ImportError("pygetwindow is required on Windows hosts") from exc

    import ctypes
    from ctypes import wintypes
else:  # pragma: no cover - used only when running tooling on non-Windows hosts
    gw = None  # type: ignore
    ctypes = None  # type: ignore
    wintypes = None  # type: ignore


def set_dpi_aware() -> None:
    """Enable DPI awareness for accurate window coordinates and mouse input.

    MUST be called once at program startup before any window or screen operations.
    Without this, GetWindowRect() returns incorrect virtualized coordinates and
    PyAutoGUI clicks will miss their targets on displays with scaling != 100%.

    This function is safe to call multiple times and is a no-op on non-Windows platforms.
    """
    if not _IS_WINDOWS:
        return

    try:
        assert ctypes is not None  # noqa: S101 - guarded by _IS_WINDOWS
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.getLogger(__name__).warning("Failed to set DPI awareness: %s", exc)


class WindowNotFoundError(RuntimeError):
    """Raised when the Uma Musume window cannot be located."""


class WindowFocusError(RuntimeError):
    """Raised when the window cannot be brought to the foreground."""


@dataclass(slots=True)
class Rect:
    """Simple rectangle helper expressed in pixels."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.width, self.height


@dataclass(slots=True)
class ClientArea:
    """Client area metadata in both screen and logical spaces."""

    screen_origin: Tuple[int, int]  # physical origin (screen pixels)
    logical_size: Tuple[int, int]
    physical_size: Tuple[int, int]
    scaling_factor: float

    @property
    def width_logical(self) -> int:
        return self.logical_size[0]

    @property
    def height_logical(self) -> int:
        return self.logical_size[1]

    @property
    def size_logical(self) -> Tuple[int, int]:
        return self.logical_size

    @property
    def width_physical(self) -> int:
        return self.physical_size[0]

    @property
    def height_physical(self) -> int:
        return self.physical_size[1]

    @property
    def size_physical(self) -> Tuple[int, int]:
        return self.physical_size

    def logical_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert logical client coordinates into physical screen pixel coordinates.

        Assumes set_dpi_aware() was called at program startup; otherwise these
        physical coordinates will be incorrect on displays with scaling != 100%.
        """
        origin_x, origin_y = self.screen_origin
        scale = self.scaling_factor
        return int(round(origin_x + x * scale)), int(round(origin_y + y * scale))

    def screen_to_logical(self, x: float, y: float) -> Tuple[int, int]:
        """Convert absolute physical screen pixel coordinates into logical client coordinates.

        Assumes set_dpi_aware() was called at program startup; otherwise the
        input coordinates must already be physical pixels for correct conversion.
        """
        origin_x, origin_y = self.screen_origin
        scale = self.scaling_factor
        return int(round((x - origin_x) / scale)), int(round((y - origin_y) / scale))


@dataclass(slots=True)
class WindowGeometry:
    """Aggregated geometry data for the Uma Musume window.

    Note: outer_rect contains physical screen pixels (from pygetwindow after DPI awareness).
    Client area provides both logical and physical sizes/conversions.
    """

    outer_rect: Rect
    client_area: ClientArea


class UmaWindow:
    """High-level helper that encapsulates Uma Musume window operations."""

    def __init__(self, config: WindowConfig, logger: Optional[Logger] = None) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._window: Optional["gw.Win32Window"] = None

        if not _IS_WINDOWS:
            self._logger.warning("UmaWindow instantiated on non-Windows platform; operations will fail")

    @property
    def config(self) -> WindowConfig:
        return self._config

    def _require_windows(self) -> None:
        if not _IS_WINDOWS:
            raise RuntimeError("UmaWindow operations require Windows")

    def _locate_window(self, force_refresh: bool = False) -> "gw.Win32Window":
        self._require_windows()
        if self._window is not None and not force_refresh:
            return self._window

        assert gw is not None  # noqa: S101 - guarded by _require_windows
        matching = [w for w in gw.getWindowsWithTitle(self._config.title) if w.title == self._config.title]
        if not matching:
            raise WindowNotFoundError(f"Window titled '{self._config.title}' not found")

        self._window = matching[0]
        self._logger.debug("Located window %s", self._window)
        return self._window

    def ensure_placement(self) -> None:
        """Move and resize the window to the configured placement."""
        window = self._locate_window()
        placement = self._config.placement

        adjusted_width, adjusted_height, screen_bounds = self._compute_adjusted_logical_size(placement)
        if (adjusted_width, adjusted_height) != (placement.width, placement.height):
            self._logger.info(
                "Adjusting placement for screen bounds: requested %sx%s, using %sx%s (logical)",
                placement.width,
                placement.height,
                adjusted_width,
                adjusted_height,
            )
        else:
            self._logger.info(
                "Setting window placement to x=%s, y=%s, width=%s, height=%s",
                placement.x,
                placement.y,
                placement.width,
                placement.height,
            )

        physical_x, physical_y, physical_width, physical_height = self._logical_rect_to_physical(
            placement.x, placement.y, adjusted_width, adjusted_height, screen_bounds
        )

        hwnd = window._hWnd  # pylint: disable=protected-access
        if not ctypes.windll.user32.MoveWindow(hwnd, physical_x, physical_y, physical_width, physical_height, True):
            raise ctypes.WinError()  # pragma: no cover - direct system failure
        time.sleep(0.05)
        # Refresh cached geometry the next time it is requested.

    def move_resize(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        repaint: bool = True,
    ) -> None:
        """Move and/or resize the window using logical outer-bounds coordinates."""

        self._require_windows()
        window = self._locate_window(force_refresh=True)

        scaling_factor = self._config.scaling_factor
        current_logical_x = int(round(window.left / scaling_factor))
        current_logical_y = int(round(window.top / scaling_factor))
        current_logical_width = int(round(window.width / scaling_factor))
        current_logical_height = int(round(window.height / scaling_factor))

        logical_x = x if x is not None else current_logical_x
        logical_y = y if y is not None else current_logical_y
        logical_width = width if width is not None else current_logical_width
        logical_height = height if height is not None else current_logical_height

        screen_bounds = (
            ctypes.windll.user32.GetSystemMetrics(0),
            ctypes.windll.user32.GetSystemMetrics(1),
        )
        physical_x, physical_y, physical_width, physical_height = self._logical_rect_to_physical(
            logical_x, logical_y, logical_width, logical_height, screen_bounds
        )

        hwnd = window._hWnd  # pylint: disable=protected-access
        if not ctypes.windll.user32.MoveWindow(hwnd, physical_x, physical_y, physical_width, physical_height, repaint):
            raise ctypes.WinError()  # pragma: no cover - direct system failure
        time.sleep(0.05)

    def focus(self) -> None:
        """Bring the window to the foreground; raises if this fails."""
        window = self._locate_window()
        try:
            if window.isMinimized:
                self._logger.debug("Window is minimized; restoring")
                window.restore()
                time.sleep(0.05)
            window.activate()
            time.sleep(0.05)
        except Exception as exc:  # pragma: no cover - dependent on GUI state
            raise WindowFocusError("Failed to focus Uma Musume window") from exc

    def refresh_geometry(self) -> WindowGeometry:
        """Fetch up-to-date geometry information for the window."""
        window = self._locate_window()
        outer = Rect(window.left, window.top, window.width, window.height)
        client_area = self._query_client_area(window)
        geometry = WindowGeometry(outer_rect=outer, client_area=client_area)
        self._logger.debug(
            "Window geometry: outer=%s, client_origin=%s, client_logical=%s, client_physical=%s",
            outer.to_tuple(),
            client_area.screen_origin,
            client_area.size_logical,
            client_area.size_physical,
        )
        return geometry

    def ensure_ready(self, reposition: bool = False) -> WindowGeometry:
        """Ensure the window is found, optionally repositioned, focused, and measured."""

        if reposition:
            self.ensure_placement()
        else:
            self._locate_window(force_refresh=True)

        self.focus()
        return self.refresh_geometry()

    def _query_client_area(self, window: "gw.Win32Window") -> ClientArea:
        self._require_windows()
        assert ctypes is not None and wintypes is not None  # noqa: S101 - ensured by _require_windows

        hwnd = window._hWnd  # pylint: disable=protected-access
        rect = wintypes.RECT()
        if not ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect)):
            raise ctypes.WinError()  # pragma: no cover - direct system failure

        # IMPORTANT: After SetProcessDPIAware(), GetClientRect returns PHYSICAL pixel dimensions
        physical_width = max(int(rect.right - rect.left), 1)
        physical_height = max(int(rect.bottom - rect.top), 1)

        # Divide by scaling factor to get logical dimensions
        scaling_factor = self._config.scaling_factor
        logical_width = max(int(round(physical_width / scaling_factor)), 1)
        logical_height = max(int(round(physical_height / scaling_factor)), 1)

        config_offset = self._config.client_offset
        if config_offset is not None:
            # Config offset is in logical pixels, scale to physical
            offset_x = int(round(config_offset[0] * scaling_factor))
            offset_y = int(round(config_offset[1] * scaling_factor))
            screen_origin = (int(window.left + offset_x), int(window.top + offset_y))
        else:
            point = wintypes.POINT(0, 0)
            if not ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point)):
                raise ctypes.WinError()  # pragma: no cover

            screen_origin = (int(point.x), int(point.y))

        return ClientArea(
            screen_origin=screen_origin,
            logical_size=(logical_width, logical_height),
            physical_size=(physical_width, physical_height),
            scaling_factor=scaling_factor,
        )

    def _compute_adjusted_logical_size(
        self, placement: WindowPlacement
    ) -> Tuple[int, int, Tuple[int, int]]:
        """Clamp placement dimensions to stay on-screen while preserving aspect ratio."""

        self._require_windows()
        assert ctypes is not None  # noqa: S101

        # GetSystemMetrics returns physical pixels after SetProcessDPIAware()
        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        screen_bounds = (screen_width, screen_height)

        desired_width = max(placement.width, 1)
        desired_height = max(placement.height, 1)
        aspect_ratio = desired_width / desired_height

        scaling_factor = max(self._config.scaling_factor, 1e-3)
        desired_physical_width = desired_width * scaling_factor
        desired_physical_height = desired_height * scaling_factor

        scale = 1.0
        if desired_physical_width > screen_width:
            scale = min(scale, screen_width / desired_physical_width)
        if desired_physical_height > screen_height:
            scale = min(scale, screen_height / desired_physical_height)

        scale = min(scale, 1.0)
        adjusted_width = max(int(round(desired_width * scale)), 1)
        adjusted_height = max(int(round(desired_height * scale)), 1)

        # Ensure the recomputed height still respects bounds when rounding effects push it over.
        adjusted_physical_height = adjusted_height * scaling_factor
        if adjusted_physical_height > screen_height:
            adjusted_height = max(int(screen_height / scaling_factor), 1)
            adjusted_width = max(int(round(adjusted_height * aspect_ratio)), 1)

        adjusted_physical_width = adjusted_width * scaling_factor
        if adjusted_physical_width > screen_width:
            adjusted_width = max(int(screen_width / scaling_factor), 1)
            adjusted_height = max(int(round(adjusted_width / aspect_ratio)), 1)

        return adjusted_width, adjusted_height, screen_bounds

    def _logical_rect_to_physical(
        self,
        logical_x: int,
        logical_y: int,
        logical_width: int,
        logical_height: int,
        screen_bounds: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """Convert logical placement rectangle into physical pixels and clamp to the screen."""

        scaling_factor = self._config.scaling_factor
        screen_width, screen_height = screen_bounds

        physical_width = max(int(round(logical_width * scaling_factor)), 1)
        physical_height = max(int(round(logical_height * scaling_factor)), 1)
        physical_x = max(int(round(logical_x * scaling_factor)), 0)
        physical_y = max(int(round(logical_y * scaling_factor)), 0)

        max_x = max(screen_width - physical_width, 0)
        max_y = max(screen_height - physical_height, 0)
        physical_x = min(physical_x, max_x)
        physical_y = min(physical_y, max_y)

        return physical_x, physical_y, physical_width, physical_height


__all__ = [
    "ClientArea",
    "Rect",
    "UmaWindow",
    "WindowGeometry",
    "WindowFocusError",
    "WindowNotFoundError",
    "set_dpi_aware",
]
