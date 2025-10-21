import tempfile
import unittest
from pathlib import Path

from PIL import Image

from lluma_os.capture import CaptureManager
from lluma_os.config import CaptureConfig
from lluma_os.window import ClientArea


class DummyWindow:
    def ensure_ready(self, reposition: bool = False):
        raise AssertionError("ensure_ready should not be called in this test")


class CaptureSplitTest(unittest.TestCase):
    def test_split_respects_ratios(self) -> None:
        config = CaptureConfig()
        config.split.enabled = True
        config.split.left_pin_ratio = 10 / 110    # 10px of 110px
        config.split.primary_ratio = 50 / 110     # 50px of 110px
        config.split.menus_ratio = 40 / 110       # 40px of 110px
        config.split.tabs_ratio = 10 / 110        # 10px of 110px (sum = 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = Path(tmpdir)
            manager = CaptureManager(DummyWindow(), config)
            image = Image.new("RGB", (110, 20), color=(255, 0, 0))
            client_area = ClientArea(
                screen_origin=(0, 0),
                logical_size=(110, 20),
                physical_size=(110, 20),
                scaling_factor=1.0,
            )

            primary, menus, tabs = manager._split_capture(image, client_area)

        self.assertIsNotNone(primary)
        self.assertIsNotNone(menus)
        self.assertIsNotNone(tabs)
        # left_pin: 0-10px (discarded)
        # primary: 10-60px (width=50)  
        # menus: 60-100px (width=40)
        # tabs: 100-110px (width=10)
        self.assertEqual(primary.size, (50, 20))
        self.assertEqual(menus.size, (40, 20))
        self.assertEqual(tabs.size, (10, 20))


if __name__ == "__main__":
    unittest.main()
