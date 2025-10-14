"""Memory file management for agent scratchpad."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

Logger = logging.Logger


class MemoryError(RuntimeError):
    """Raised when memory operations fail or exceed budget."""


@dataclass(slots=True)
class MemoryFile:
    """Represents a single memory file."""

    name: str
    content: str
    token_count: int


class MemoryManager:
    """Manages agent memory files with token budget enforcement."""

    def __init__(
        self,
        memory_dir: Path,
        max_tokens: int = 32000,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize memory manager.

        Args:
            memory_dir: Directory to store memory files
            max_tokens: Maximum cumulative token budget for all files
            logger: Optional logger instance
        """
        self._memory_dir = memory_dir
        self._max_tokens = max_tokens
        self._logger = logger or logging.getLogger(__name__)
        self._files: Dict[str, MemoryFile] = {}

        # Create directory if needed
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        # Load existing files
        self._load_existing_files()

    def create_file(self, name: str) -> None:
        """Create a new empty memory file.

        Args:
            name: File name (must not exist)

        Raises:
            MemoryError: If file already exists or name is invalid
        """
        if not self._is_valid_name(name):
            raise MemoryError(f"Invalid file name: {name}")

        if name in self._files:
            raise MemoryError(f"Memory file '{name}' already exists")

        self._files[name] = MemoryFile(name=name, content="", token_count=0)
        self._save_file(name)
        self._logger.info("Created memory file: %s", name)

    def write_file(self, name: str, content: str) -> None:
        """Write content to a memory file (overwrites existing content).

        Args:
            name: File name
            content: New content

        Raises:
            MemoryError: If file doesn't exist or would exceed token budget
        """
        if name not in self._files:
            raise MemoryError(f"Memory file '{name}' does not exist; use createMemoryFile first")

        new_token_count = self._estimate_tokens(content)
        old_token_count = self._files[name].token_count
        total_tokens = self._total_tokens() - old_token_count + new_token_count

        if total_tokens > self._max_tokens:
            raise MemoryError(
                f"Writing to '{name}' would exceed memory budget: "
                f"{total_tokens} > {self._max_tokens} tokens"
            )

        self._files[name] = MemoryFile(name=name, content=content, token_count=new_token_count)
        self._save_file(name)
        self._logger.info("Wrote %d tokens to memory file: %s", new_token_count, name)

    def delete_file(self, name: str) -> None:
        """Delete a memory file.

        Args:
            name: File name

        Raises:
            MemoryError: If file doesn't exist
        """
        if name not in self._files:
            raise MemoryError(f"Memory file '{name}' does not exist")

        del self._files[name]
        file_path = self._memory_dir / name
        file_path.unlink(missing_ok=True)
        self._logger.info("Deleted memory file: %s", name)

    def read_file(self, name: str) -> str:
        """Read content from a memory file.

        Args:
            name: File name

        Returns:
            File content

        Raises:
            MemoryError: If file doesn't exist
        """
        if name not in self._files:
            raise MemoryError(f"Memory file '{name}' does not exist")
        return self._files[name].content

    def list_files(self) -> list[str]:
        """List all memory file names."""
        return sorted(self._files.keys())

    def get_all_content(self) -> str:
        """Get concatenated content of all memory files for injection into context.

        Returns:
            Markdown-formatted string with all memory files
        """
        if not self._files:
            return "*(No memory files)*\n"

        sections = ["# Memory Files\n"]
        for name in sorted(self._files.keys()):
            content = self._files[name].content
            sections.append(f"## {name}\n\n{content}\n")

        return "\n".join(sections)

    def total_tokens(self) -> int:
        """Get current total token count across all files."""
        return self._total_tokens()

    def _total_tokens(self) -> int:
        """Internal helper to compute total tokens."""
        return sum(f.token_count for f in self._files.values())

    def _load_existing_files(self) -> None:
        """Load any existing memory files from disk."""
        for file_path in self._memory_dir.glob("*"):
            if file_path.is_file() and self._is_valid_name(file_path.name):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    token_count = self._estimate_tokens(content)
                    self._files[file_path.name] = MemoryFile(
                        name=file_path.name,
                        content=content,
                        token_count=token_count,
                    )
                    self._logger.debug("Loaded memory file: %s (%d tokens)", file_path.name, token_count)
                except Exception as exc:
                    self._logger.warning("Failed to load memory file %s: %s", file_path.name, exc)

        total = self._total_tokens()
        self._logger.info("Loaded %d memory files (%d tokens)", len(self._files), total)

    def _save_file(self, name: str) -> None:
        """Save a memory file to disk."""
        file_path = self._memory_dir / name
        content = self._files[name].content
        file_path.write_text(content, encoding="utf-8")

    def _is_valid_name(self, name: str) -> bool:
        """Check if a file name is valid (no path separators, not empty)."""
        if not name or "/" in name or "\\" in name:
            return False
        return True

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content.

        Uses a simple heuristic: ~4 characters per token.
        This is conservative and should work for most content types.
        """
        return max(1, len(content) // 4)


__all__ = ["MemoryManager", "MemoryFile", "MemoryError"]
