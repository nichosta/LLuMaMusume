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
        config.split.left_pin_px = 10
        config.split.primary_ratio = 0.5
        config.split.menus_ratio = 0.5

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

            primary, menus = manager._split_capture(image, client_area)

        self.assertIsNotNone(primary)
        self.assertIsNotNone(menus)
        self.assertEqual(primary.size, (60, 20))
        self.assertEqual(menus.size, (50, 20))


if __name__ == "__main__":
    unittest.main()
