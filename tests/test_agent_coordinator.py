"""Unit tests for helper utilities in the game loop coordinator."""
from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from lluma_agent.coordinator import (
    GameLoopCoordinator,
    calculate_primary_center,
    calculate_region_offset,
    convert_vlm_box_to_pixels,
)
from lluma_os.config import CaptureConfig, VisionCacheConfig
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


class _StubVision:
    def __init__(self, hash_value: str = "hash", distance: int = 0) -> None:
        self.hash_value = hash_value
        self.distance = distance

    def compute_perceptual_hash(self, image: Image.Image) -> str:
        return self.hash_value

    def hash_distance(self, hash1: str, hash2: str) -> int:
        return self.distance


class _StubLogger:
    def debug(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - trivial
        return None

    def info(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - trivial
        return None

    def warning(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - trivial
        return None

    def error(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - trivial
        return None


def _make_coordinator_for_cache_tests(
    *,
    menu_hits: int = 0,
    primary_hits: int = 0,
    distance: int = 0,
) -> GameLoopCoordinator:
    coordinator = object.__new__(GameLoopCoordinator)
    cache_config = VisionCacheConfig()
    cache_config.hash_distance_threshold = 8
    cache_config.max_age_turns = 10
    cache_config.menu_force_refresh_interval = 0
    cache_config.primary_force_refresh_interval = 0

    coordinator._vision = _StubVision(distance=distance)  # type: ignore[attr-defined]
    coordinator._agent_config = SimpleNamespace(vision_cache=cache_config)  # type: ignore[attr-defined]

    coordinator._menu_cache_hash = "hash"  # type: ignore[attr-defined]
    coordinator._menu_cache_turn = 5  # type: ignore[attr-defined]
    coordinator._menu_cache_consecutive_hits = menu_hits  # type: ignore[attr-defined]
    coordinator._menu_cache_forced_refresh = 0  # type: ignore[attr-defined]
    coordinator._last_menu_tab = "Tab"  # type: ignore[attr-defined]

    coordinator._primary_cache_hash = "hash"  # type: ignore[attr-defined]
    coordinator._primary_cache_turn = 5  # type: ignore[attr-defined]
    coordinator._primary_cache_consecutive_hits = primary_hits  # type: ignore[attr-defined]
    coordinator._primary_cache_forced_refresh = 0  # type: ignore[attr-defined]
    coordinator._primary_cache_buttons_raw = []  # type: ignore[attr-defined]
    coordinator._logger = _StubLogger()  # type: ignore[attr-defined]

    return coordinator


def test_menu_force_refresh_zero_disables_periodic_refresh() -> None:
    coordinator = _make_coordinator_for_cache_tests()

    should_call_vlm, current_hash, distance = coordinator._should_refresh_menu_cache(  # type: ignore[attr-defined]
        Image.new("RGB", (4, 4)),
        "Tab",
        current_turn=6,
    )

    assert should_call_vlm is False
    assert current_hash == "hash"
    assert distance == 0
    assert coordinator._menu_cache_forced_refresh == 0  # type: ignore[attr-defined]


def test_menu_force_refresh_triggers_when_interval_reached() -> None:
    coordinator = _make_coordinator_for_cache_tests(menu_hits=2)
    coordinator._agent_config.vision_cache.menu_force_refresh_interval = 2  # type: ignore[attr-defined]

    should_call_vlm, current_hash, distance = coordinator._should_refresh_menu_cache(  # type: ignore[attr-defined]
        Image.new("RGB", (4, 4)),
        "Tab",
        current_turn=6,
    )

    assert should_call_vlm is True
    assert current_hash == "hash"
    assert distance == 0
    assert coordinator._menu_cache_forced_refresh == 1  # type: ignore[attr-defined]


def test_primary_force_refresh_zero_disables_periodic_refresh() -> None:
    coordinator = _make_coordinator_for_cache_tests()

    should_call_vlm, current_hash, distance = coordinator._should_refresh_primary_cache(  # type: ignore[attr-defined]
        Image.new("RGB", (4, 4)),
        current_turn=6,
    )

    assert should_call_vlm is False
    assert current_hash == "hash"
    assert distance == 0
    assert coordinator._primary_cache_forced_refresh == 0  # type: ignore[attr-defined]


def test_primary_force_refresh_triggers_when_interval_reached() -> None:
    coordinator = _make_coordinator_for_cache_tests(primary_hits=3)
    coordinator._agent_config.vision_cache.primary_force_refresh_interval = 3  # type: ignore[attr-defined]

    should_call_vlm, current_hash, distance = coordinator._should_refresh_primary_cache(  # type: ignore[attr-defined]
        Image.new("RGB", (4, 4)),
        current_turn=6,
    )

    assert should_call_vlm is True
    assert current_hash == "hash"
    assert distance == 0
    assert coordinator._primary_cache_forced_refresh == 1  # type: ignore[attr-defined]

