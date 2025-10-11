"""Regression tests for the menu vision analyzer."""
from __future__ import annotations

import base64
import io
import tempfile
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
            # Tab is career profile, top 4 tabs are enabled, bottom 4 tabs are disabled
            {
                "file": "career profile, bottom 3 disabled.png",
                "expected_usable": True,
                "expected_selected": "Career Profile",
                "expected_available": ["Jukebox", "Sparks", "Log", "Career Profile"],
            },
            # Tab is menu, all buttons in tab window are active, only topmost and bottommost tabs are selectable
            {
                "file": "menu, all active.png",
                "expected_usable": True,
                "expected_selected": "Menu",
                "expected_available": ["Jukebox", "Menu"],
            },
            # Tab is menu, but the entire image is blurred, indicating unusability
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


if __name__ == "__main__":  # pragma: no cover - allows direct invocation
    unittest.main()
