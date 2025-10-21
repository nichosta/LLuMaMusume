"""Unit tests for helper utilities in the game loop coordinator."""
from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from lluma_agent.coordinator import (
    calculate_primary_center,
    calculate_region_offset,
    convert_vlm_box_to_pixels,
)
from lluma_os.config import CaptureConfig
from lluma_os.window import ClientArea, Rect, WindowGeometry


def _make_geometry(*, width_logical: int, height_logical: int, scale: float) -> WindowGeometry:
    physical_width = int(round(width_logical * scale))
    physical_height = int(round(height_logical * scale))
    return WindowGeometry(
        outer_rect=Rect(left=0, top=0, width=physical_width, height=physical_height),
        client_area=ClientArea(
            screen_origin=(0, 0),
            logical_size=(width_logical, height_logical),
            physical_size=(physical_width, physical_height),
            scaling_factor=scale,
        ),
    )


def test_convert_vlm_box_to_pixels_maps_normalised_box() -> None:
    result = convert_vlm_box_to_pixels([250, 100, 750, 600], 800, 600)
    assert result == (80, 150, 400, 300)


def test_calculate_primary_center_uses_logical_coordinates() -> None:
    capture_config = CaptureConfig()
    geometry = _make_geometry(width_logical=1600, height_logical=900, scale=1.5)

    # Primary crop corresponds to the configured primary region (40% of width).
    primary_width_physical = int(round((geometry.client_area.width_logical * capture_config.split.primary_ratio) * geometry.client_area.scaling_factor))
    primary_height_physical = geometry.client_area.height_physical
    primary_image = Image.new("RGB", (primary_width_physical, primary_height_physical))

    capture_result = SimpleNamespace(geometry=geometry, primary_image=primary_image)

    center_x, center_y = calculate_primary_center(capture_result, capture_config)

    # Left pin offset should shift the center right by 10% of the full width.
    expected_offset = calculate_region_offset(
        "primary",
        capture_config,
        geometry.client_area.width_logical,
    )[0]
    expected_width_logical = int(round(primary_width_physical / geometry.client_area.scaling_factor))

    assert center_x == expected_offset + expected_width_logical // 2
    assert center_y == geometry.client_area.height_logical // 2


def test_calculate_primary_center_without_primary_image() -> None:
    capture_config = CaptureConfig()
    geometry = _make_geometry(width_logical=1400, height_logical=880, scale=1.5)
    capture_result = SimpleNamespace(geometry=geometry, primary_image=None)

    assert calculate_primary_center(capture_result, capture_config) == (
        geometry.client_area.width_logical // 2,
        geometry.client_area.height_logical // 2,
    )

