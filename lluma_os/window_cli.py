"""Command-line helper for debugging Uma Musume window placement."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    from .config import load_configs
    from .window import UmaWindow, WindowGeometry
except ImportError:  # pragma: no cover - direct execution fallback
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from lluma_os.config import load_configs  # type: ignore
    from lluma_os.window import UmaWindow, WindowGeometry  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect, move, or resize the Uma Musume window.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--reposition-first",
        action="store_true",
        help="Call ensure_placement() before executing the command.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Print the current outer/client geometry.")
    subparsers.add_parser("focus", help="Bring the window to the foreground.")
    subparsers.add_parser("place", help="Apply the placement from config.yaml.")

    move_parser = subparsers.add_parser("move", help="Move the window (logical pixels).")
    move_parser.add_argument("--x", type=int, help="Target X (logical outer pixels).")
    move_parser.add_argument("--y", type=int, help="Target Y (logical outer pixels).")

    resize_parser = subparsers.add_parser("resize", help="Resize the window (logical pixels).")
    resize_parser.add_argument("--width", type=int, help="Target width in logical outer pixels.")
    resize_parser.add_argument("--height", type=int, help="Target height in logical outer pixels.")

    set_parser = subparsers.add_parser("set", help="Move and resize in one step (logical pixels).")
    set_parser.add_argument("--x", type=int, help="Target X (logical outer pixels).")
    set_parser.add_argument("--y", type=int, help="Target Y (logical outer pixels).")
    set_parser.add_argument("--width", type=int, help="Target width in logical outer pixels.")
    set_parser.add_argument("--height", type=int, help="Target height in logical outer pixels.")

    return parser


def _print_geometry(geometry: WindowGeometry) -> None:
    outer = geometry.outer_rect
    client = geometry.client_area
    scale = client.scaling_factor

    logical_left = outer.left / scale
    logical_top = outer.top / scale
    logical_width = outer.width / scale
    logical_height = outer.height / scale

    print(
        f"Outer (physical): x={outer.left:.0f}, y={outer.top:.0f}, "
        f"width={outer.width:.0f}, height={outer.height:.0f}"
    )
    print(
        f"Outer (logical):  x={logical_left:.1f}, y={logical_top:.1f}, "
        f"width={logical_width:.1f}, height={logical_height:.1f}"
    )
    print(
        "Client: origin_physical=({0}, {1}), size_logical={2}x{3}, size_physical={4}x{5}".format(
            client.screen_origin[0],
            client.screen_origin[1],
            client.width_logical,
            client.height_logical,
            client.width_physical,
            client.height_physical,
        )
    )
    print(f"Scaling factor: {client.scaling_factor:.3f}")


def main(argv: list[str] | None = None) -> int:
    if not sys.platform.startswith("win32"):
        print("Window CLI must be run on Windows.", file=sys.stderr)
        return 1

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "move" and args.x is None and args.y is None:
        parser.error("move requires --x or --y.")
    if args.command == "resize" and args.width is None and args.height is None:
        parser.error("resize requires --width or --height.")
    if args.command == "set" and all(
        value is None for value in (args.x, args.y, args.width, args.height)
    ):
        parser.error("set requires at least one of --x/--y/--width/--height.")

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")

    window_cfg, _, _ = load_configs(args.config)
    uma_window = UmaWindow(window_cfg)

    if args.reposition_first:
        uma_window.ensure_placement()

    if args.command == "status":
        geometry = uma_window.refresh_geometry()
        _print_geometry(geometry)
        return 0

    if args.command == "focus":
        uma_window.focus()
        geometry = uma_window.refresh_geometry()
        _print_geometry(geometry)
        return 0

    if args.command == "place":
        uma_window.ensure_placement()
        geometry = uma_window.refresh_geometry()
        _print_geometry(geometry)
        return 0

    if args.command == "move":
        uma_window.move_resize(x=args.x, y=args.y)
        geometry = uma_window.refresh_geometry()
        _print_geometry(geometry)
        return 0

    if args.command == "resize":
        uma_window.move_resize(width=args.width, height=args.height)
        geometry = uma_window.refresh_geometry()
        _print_geometry(geometry)
        return 0

    if args.command == "set":
        uma_window.move_resize(x=args.x, y=args.y, width=args.width, height=args.height)
        geometry = uma_window.refresh_geometry()
        _print_geometry(geometry)
        return 0

    parser.error(f"Unknown command {args.command}")
    return 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())

