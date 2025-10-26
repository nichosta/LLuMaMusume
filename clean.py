#!/usr/bin/env python3
"""Clean up logs, memory, and captures directories."""

import shutil
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
    print("Cleaning project directories...")

    dirs_to_clean = [
        Path("logs"),
        Path("memory"),
        Path("captures"),
    ]

    for dir_path in dirs_to_clean:
        clean_directory(dir_path)

    print("Done!")


if __name__ == "__main__":
    main()
