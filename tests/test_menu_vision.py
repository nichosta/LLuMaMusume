"""Regression tests for the menu vision analyzer."""
from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from lluma_vision import MenuAnalyzer, MenuState, TabAvailability
from lluma_vision.menu_analyzer import ScrollbarInfo, TabInfo

DATA_DIR = Path(__file__).parent / "data" / "menu_captures"
PRIMARY_SCROLLBAR_DIR = Path(__file__).parent / "data" / "primary_scrollbars"


class MenuVisionRegressionTest(unittest.TestCase):
    """Validates menu analysis against curated reference screenshots."""

    def setUp(self) -> None:
        self.analyzer = MenuAnalyzer()

    def _load_menu_image(self, filename: str) -> Image.Image:
        image_path = DATA_DIR / filename
        self.assertTrue(image_path.exists(), f"Missing test asset: {image_path}")
        with Image.open(image_path) as base_image:
            return base_image.copy()

    def test_reference_captures(self) -> None:
        test_cases = [
            {
                "file": "top and bottom active, menu selected.png",
                "expected_usable": True,
                "expected_selected": "Menu",
                "expected_available": ["Jukebox", "Menu"],
            },
            {
                "file": "all blurred.png",
                "expected_usable": False,
                "expected_selected": None,
                "expected_available": [],
            },
        ]

        for case in test_cases:
            with self.subTest(filename=case["file"]):
                menu_image = self._load_menu_image(case["file"])
                result = self.analyzer.analyze_menu(menu_image)

                self.assertEqual(result.is_usable, case["expected_usable"])

                if not case["expected_usable"]:
                    continue

                self.assertEqual(result.selected_tab, case["expected_selected"])

                available_tabs = [
                    tab.name
                    for tab in result.tabs
                    if tab.availability == TabAvailability.AVAILABLE
                ]
                self.assertCountEqual(available_tabs, case["expected_available"])

    def test_parse_buttons_payload_normalises_values(self) -> None:
        analyzer = MenuAnalyzer()

        payload = """```json
        {
            "buttons": [
                {
                    "label": "Start|section=menus|state=active|type=button|conf=0.9",
                    "box_2d": [-10, 50, "100", 1205]
                }
            ]
        }
        ```"""

        parsed = analyzer._parse_buttons_payload(payload)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["label"], "Start|section=menus|state=active|type=button|conf=0.9")
        self.assertEqual(parsed[0]["box_2d"], [0.0, 50.0, 100.0, 1000.0])

    def test_parse_buttons_payload_accepts_dict_response(self) -> None:
        analyzer = MenuAnalyzer()

        payload = """
        {
            "buttons": [
                {"label": "Enter", "box_2d": [10, 20, 200, 220]}
            ]
        }
        """

        parsed = analyzer._parse_buttons_payload(payload)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["label"], "Enter")
        self.assertEqual(parsed[0]["box_2d"], [10.0, 20.0, 200.0, 220.0])

    def test_menu_state_available_tabs_helper(self) -> None:
        state = MenuState(
            is_usable=True,
            selected_tab="Menu",
            tabs=[
                TabInfo("Jukebox", False, TabAvailability.AVAILABLE),
                TabInfo("Sparks", False, TabAvailability.UNAVAILABLE),
                TabInfo("Menu", True, TabAvailability.AVAILABLE),
            ],
        )

        self.assertListEqual(state.available_tabs(), ["Jukebox", "Menu"])

    def test_prepare_api_image_trims_left_region(self) -> None:
        analyzer = MenuAnalyzer()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "source.png"
            source = Image.new("RGB", (100, 40), color=(255, 0, 0))
            source.save(path)

            encoded = analyzer._prepare_api_image(path, trim_left_ratio=MenuAnalyzer.LEFT_TRIM_RATIO)
            decoded = base64.b64decode(encoded)
            with Image.open(io.BytesIO(decoded)) as trimmed:
                self.assertEqual(trimmed.size, (85, 40))

    def test_detect_primary_scrollbar_samples(self) -> None:
        """Validate scrollbar heuristics against reference captures."""

        # No scrollbar present
        with Image.open(PRIMARY_SCROLLBAR_DIR / "20251011-161528-primary.png") as image:
            result = self.analyzer.detect_primary_scrollbar(image)
        self.assertIsNone(result)

        # Scrollbar near the bottom with travel remaining in both directions
        with Image.open(PRIMARY_SCROLLBAR_DIR / "20251011-184452-primary.png") as image:
            result = self.analyzer.detect_primary_scrollbar(image)

        self.assertIsInstance(result, ScrollbarInfo)
        assert result is not None  # help type checkers
        self.assertGreater(result.track_bounds[0], 880)
        self.assertLess(result.track_bounds[0], 940)
        self.assertTrue(result.can_scroll_up)
        self.assertTrue(result.can_scroll_down)
        self.assertAlmostEqual(result.thumb_ratio, 0.66, delta=0.05)
        self.assertGreater(result.thumb_bounds[3], 200)

        # Scrollbar positioned near the very top but still scrollable downward
        for filename in [
            "20251011-185948-primary.png",
            "20251011-185959-primary.png",
        ]:
            with Image.open(PRIMARY_SCROLLBAR_DIR / filename) as image:
                scroll = self.analyzer.detect_primary_scrollbar(image)

            self.assertIsInstance(scroll, ScrollbarInfo)
            assert scroll is not None
            self.assertGreater(scroll.track_bounds[0], 900)
            self.assertLess(scroll.track_bounds[0], 940)
            self.assertLess(scroll.track_bounds[3], 1500)
            self.assertLess(scroll.thumb_ratio, 0.15)
            self.assertGreater(scroll.thumb_bounds[3], 40)
            self.assertTrue(scroll.can_scroll_down)


if __name__ == "__main__":  # pragma: no cover - allows direct invocation
    unittest.main()
