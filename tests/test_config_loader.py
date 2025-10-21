import tempfile
import unittest
from pathlib import Path

from lluma_os.config import load_configs


class LoadConfigsTest(unittest.TestCase):
    def test_defaults_when_missing_file(self) -> None:
        window_cfg, capture_cfg, agent_cfg = load_configs(Path("nonexistent.yaml"))
        self.assertEqual(window_cfg.title, "Umamusume")
        self.assertEqual(window_cfg.placement.width, 1920)
        self.assertEqual(capture_cfg.output_dir, Path("captures"))
        self.assertEqual(capture_cfg.scaling_factor, window_cfg.scaling_factor)
        self.assertEqual(agent_cfg.model, "anthropic/claude-haiku-4.5")

    def test_overrides_apply(self) -> None:
        yaml_content = """
window:
  window_title: TestTitle
  placement:
    x: 10
    y: 20
    width: 1280
    height: 720
  dpi:
    scaling_factor: 2.0
capture:
  output_dir: temp_captures
  split:
    enabled: false
    left_pin_ratio: 0.1
    primary_ratio: 0.6
    menus_ratio: 0.2
    tabs_ratio: 0.1
  validation:
    min_mean_luminance: 15
    min_luminance_stddev: 2
  retention:
    max_captures: 20
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text(yaml_content, encoding="utf-8")

            window_cfg, capture_cfg, agent_cfg = load_configs(cfg_path)

        self.assertEqual(window_cfg.title, "TestTitle")
        self.assertEqual(window_cfg.placement.x, 10)
        self.assertEqual(window_cfg.scaling_factor, 2.0)
        self.assertEqual(capture_cfg.output_dir, Path("temp_captures"))
        self.assertFalse(capture_cfg.split.enabled)
        self.assertAlmostEqual(capture_cfg.split.left_pin_ratio, 0.1)
        self.assertAlmostEqual(capture_cfg.split.primary_ratio, 0.6)
        self.assertAlmostEqual(capture_cfg.split.menus_ratio, 0.2)
        self.assertAlmostEqual(capture_cfg.split.tabs_ratio, 0.1)
        self.assertEqual(capture_cfg.retention.max_captures, 20)
        self.assertEqual(capture_cfg.scaling_factor, window_cfg.scaling_factor)
        self.assertEqual(agent_cfg.logs_dir, Path("logs"))


if __name__ == "__main__":
    unittest.main()
