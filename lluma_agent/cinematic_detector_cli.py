"""CLI helper to exercise the cinematic detector against saved captures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from PIL import Image

try:
    from lluma_os.config import load_configs
    from .cinematics import CinematicDetector, CinematicObservation
except ImportError:
    # Allow direct execution by resolving the package root
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from lluma_os.config import load_configs
from lluma_agent.cinematics import CinematicDetector, CinematicObservation

def _load_metadata(path: Optional[Path]) -> Dict[str, dict]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    raise ValueError("metadata JSON must be a mapping keyed by filename")


def _resolve_menu_flag(filename: str, metadata: Dict[str, dict]) -> Optional[bool]:
    entry = metadata.get(filename)
    if isinstance(entry, dict):
        value = entry.get("menu_is_usable")
        if isinstance(value, bool):
            return value
    stem = Path(filename).stem.lower()
    if "cutscene" in stem:
        return False
    return None


def _resolve_button_labels(filename: str, metadata: Dict[str, dict]) -> Iterable[str]:
    entry = metadata.get(filename)
    if not isinstance(entry, dict):
        return ()
    labels = entry.get("button_labels") or entry.get("buttons")
    if isinstance(labels, list):
        return [str(item) for item in labels]
    return ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect cinematic detection heuristics against saved captures.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Capture files or directories to analyse (PNG/JPG are supported).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml (defaults to ./config.yaml).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional JSON file mapping filenames to detector hints (menu_is_usable, button_labels).",
    )
    parser.add_argument(
        "--assume-menu-enabled",
        action="store_true",
        help="Force menu_is_usable=True for all frames unless metadata overrides it.",
    )
    parser.add_argument(
        "--assume-menu-disabled",
        action="store_true",
        help="Force menu_is_usable=False for all frames unless metadata overrides it.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON lines instead of formatted text.",
    )
    return parser


def _collect_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.png")))
            files.extend(sorted(path.glob("*.jpg")))
            files.extend(sorted(path.glob("*.jpeg")))
        elif path.is_file():
            files.append(path)
    return sorted({f.resolve() for f in files})


def _format_result(path: Path, result) -> str:
    return (
        f"{path.name:45s}"
        f" kind={result.kind.value:11s}"
        f" playback={result.playback.value:7s}"
        f" diff={result.diff_score:0.4f}"
        f" primary_diff={result.primary_diff_score:0.4f}"
        f" skip_primary={int(result.skip_hint_primary)}"
        f" skip_tabs={int(result.skip_hint_tabs)}"
        f" skip_label={int(result.skip_label_hint)}"
        f" pin_present={int(result.pin_present)}"
        f" menu_streak={result.menu_unusable_streak}"
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.assume_menu_enabled and args.assume_menu_disabled:
        parser.error("Only one of --assume-menu-enabled/disabled may be set")

    _, capture_cfg, _ = load_configs(args.config)
    metadata = _load_metadata(args.metadata)
    files = _collect_files(args.paths)
    if not files:
        parser.error("No capture files found.")

    detector = CinematicDetector(capture_cfg)

    for path in files:
        image = Image.open(path)
        if args.assume_menu_enabled:
            menu_is_usable = True
        elif args.assume_menu_disabled:
            menu_is_usable = False
        else:
            menu_is_usable = _resolve_menu_flag(path.name, metadata)

        labels = list(_resolve_button_labels(path.name, metadata))
        result = detector.observe(
            CinematicObservation(
                image=image,
                menu_is_usable=menu_is_usable,
                button_labels=labels,
            )
        )
        if args.json:
            payload = {
                "file": path.name,
                "kind": result.kind.value,
                "playback": result.playback.value,
                "diff_score": result.diff_score,
                "primary_diff_score": result.primary_diff_score,
                "skip_hint_primary": result.skip_hint_primary,
                "skip_hint_tabs": result.skip_hint_tabs,
                "skip_label_hint": result.skip_label_hint,
                "pin_present": result.pin_present,
                "pin_bright_ratio": result.pin_bright_ratio,
                "menu_unusable_streak": result.menu_unusable_streak,
            }
            print(json.dumps(payload))
        else:
            print(_format_result(path, result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
