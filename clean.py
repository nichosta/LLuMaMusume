#!/usr/bin/env python3
"""Clean up logs, memory, and captures directories."""

import argparse
import shutil
import sys
from pathlib import Path


def clean_directory(dir_path: Path) -> None:
    """Remove all contents of a directory but keep the directory itself."""
    if not dir_path.exists():
        print(f"  {dir_path} does not exist, skipping")
        return

    if not dir_path.is_dir():
        print(f"  {dir_path} is not a directory, skipping")
        return

    count = 0
    for item in dir_path.iterdir():
        if item.is_file():
            item.unlink()
            count += 1
        elif item.is_dir():
            shutil.rmtree(item)
            count += 1

    print(f"  Removed {count} items from {dir_path}")


def main() -> None:
    """Clean logs, memory, and captures directories."""
    parser = argparse.ArgumentParser(
        description="Clean up project directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean.py       # Clean logs and captures (default)
  python clean.py all   # Clean logs, memory, and captures
  python clean.py log   # Clean only the logs/ directory
  python clean.py mem   # Clean only the memory/ directory
  python clean.py cap   # Clean only the captures/ directory
        """,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,  # Set default to None when no argument is provided
        choices=["all", "log", "mem", "cap"],
        help="What to clean: 'all', 'log', 'mem', or 'cap'. If omitted, cleans logs and captures.",
    )

    args = parser.parse_args()

    if args.target is None:
        # Default behavior: clean logs and captures
        dirs_to_clean = [Path("logs"), Path("captures")]
        print("Cleaning logs and captures directories (default)...")
    else:
        # User-specified behavior
        target_map = {
            "all": [Path("logs"), Path("memory"), Path("captures")],
            "log": [Path("logs")],
            "mem": [Path("memory")],
            "cap": [Path("captures")],
        }
        dirs_to_clean = target_map[args.target]

        if args.target == "all":
            print("Cleaning all project directories...")
        else:
            print(f"Cleaning {args.target} directory...")

    for dir_path in dirs_to_clean:
        clean_directory(dir_path)

    print("Done!")


if __name__ == "__main__":
    main()
