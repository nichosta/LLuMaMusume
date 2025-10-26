"""Command-line helper for manually triggering a client capture."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from .capture import CaptureError, CaptureManager
    from .config import load_configs
    from .window import UmaWindow
except ImportError:
    # Allow direct execution (python lluma_os/cli.py) by resolving the package root
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from lluma_os.capture import CaptureError, CaptureManager
    from lluma_os.config import load_configs
    from lluma_os.window import UmaWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture the Uma Musume client window once.")
    parser.add_argument("--turn-id", help="Identifier used for capture filenames")
    parser.add_argument("--config", type=Path, help="Path to config.yaml", default=Path("config.yaml"))
    parser.add_argument(
        "--reposition",
        action="store_true",
        help="Force the window to the configured placement before capturing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if not sys.platform.startswith("win32"):
        print("Capture CLI must be run on Windows", file=sys.stderr)
        return 1

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")

    window_cfg, capture_cfg, agent_config = load_configs(args.config)
    capture_cfg.scaling_factor = window_cfg.scaling_factor

    uma_window = UmaWindow(window_cfg)
    manager = CaptureManager(uma_window, capture_cfg)

    turn_id = args.turn_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        result = manager.capture(turn_id, reposition=args.reposition)
    except (CaptureError, RuntimeError) as exc:
        logging.error("Capture failed: %s", exc)
        return 1

    logging.info(
        "Capture stored at %s (primary=%s, menus=%s, tabs=%s)",
        result.raw_path,
        result.primary_path,
        result.menus_path,
        result.tabs_path,
    )
    logging.info(
        "Mean luminance %.2f / std %.2f across %dx%d",
        result.validation.mean_luminance,
        result.validation.stddev_luminance,
        result.validation.size_px[0],
        result.validation.size_px[1],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
