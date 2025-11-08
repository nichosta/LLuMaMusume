"""Agent orchestration for Uma Musume gameplay."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import typing aid only
    from .agent import UmaAgent
    from .coordinator import GameLoopCoordinator
    from .memory import MemoryManager

__all__ = ["UmaAgent", "MemoryManager", "GameLoopCoordinator"]


def __getattr__(name: str):
    if name == "UmaAgent":
        from .agent import UmaAgent

        return UmaAgent
    if name == "MemoryManager":
        from .memory import MemoryManager

        return MemoryManager
    if name == "GameLoopCoordinator":
        from .coordinator import GameLoopCoordinator

        return GameLoopCoordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
