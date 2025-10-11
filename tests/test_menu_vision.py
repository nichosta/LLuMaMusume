"""Regression tests for the menu vision analyzer."""
from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

from lluma_vision import MenuAnalyzer
from lluma_vision.menu_analyzer import TabAvailability

DATA_DIR = Path(__file__).parent / "data" / "menu_captures"
MENU_SECTION_START_RATIO = 0.5  # menus occupy the right half of the capture


class MenuVisionRegressionTest(unittest.TestCase):
    """Validates menu analysis against curated reference screenshots."""

    def setUp(self) -> None:
        self.analyzer = MenuAnalyzer()

    def _load_menu_section(self, filename: str) -> Image.Image:
        image_path = DATA_DIR / filename
        self.assertTrue(image_path.exists(), f"Missing test asset: {image_path}")

        with Image.open(image_path) as base_image:
            width, height = base_image.size
            menu_start_x = int(width * MENU_SECTION_START_RATIO)
            menu_image = base_image.crop((menu_start_x, 0, width, height)).copy()

        return menu_image

    def test_reference_captures(self) -> None:
        test_cases = [
            {
                "file": "career profile, bottom 3 disabled.png",
                "expected_usable": True,
                "expected_selected": "Career Profile",
                "expected_available": ["Jukebox", "Sparks", "Log", "Career Profile"],
            },
            {
                "file": "menu, all active.png",
                "expected_usable": True,
                "expected_selected": "Menu",
                "expected_available": ["Jukebox", "Menu"],
            },
            {
                "file": "menu, all blurred.png",
                "expected_usable": False,
                "expected_selected": None,
                "expected_available": [],
            },
        ]

        for case in test_cases:
            with self.subTest(filename=case["file"]):
                menu_image = self._load_menu_section(case["file"])
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


if __name__ == "__main__":  # pragma: no cover - allows direct invocation
    unittest.main()
