#!/usr/bin/env python3
"""Main entry point for LLuMa Musume agent."""
import logging
import sys
from pathlib import Path

from lluma_agent import GameLoopCoordinator
from lluma_os.config import load_configs


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Root logger instance
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger()


def main() -> int:
    """Main entry point."""
    logger = setup_logging(level=logging.INFO)

    logger.info("=" * 70)
    logger.info("LLuMa Musume - Autonomous Uma Musume Agent")
    logger.info("=" * 70)

    # Load configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.warning("config.yaml not found; using defaults")

    try:
        window_cfg, capture_cfg, agent_cfg = load_configs(config_path)
    except Exception as exc:
        logger.error("Failed to load configuration: %s", exc)
        return 1

    logger.info("Configuration loaded successfully")
    logger.info("  Window: %s @ %dx%d", window_cfg.title, window_cfg.placement.width, window_cfg.placement.height)
    logger.info("  DPI scaling: %.1fx", window_cfg.scaling_factor)
    logger.info("  Agent model: %s", agent_cfg.model)
    logger.info("  Memory dir: %s", agent_cfg.memory_dir)
    logger.info("  Logs dir: %s", agent_cfg.logs_dir)

    # Check for required environment variables
    import os
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable not set")
        logger.error("Please set it before running: export OPENROUTER_API_KEY='your-key-here'")
        return 1

    # Initialize and run coordinator
    try:
        coordinator = GameLoopCoordinator(
            window_config=window_cfg,
            capture_config=capture_cfg,
            agent_config=agent_cfg,
            logger=logger,
        )

        logger.info("Starting game loop...")
        logger.info("Press Ctrl+C to stop gracefully")
        logger.info("")

        coordinator.run(reposition=True)

        logger.info("Game loop terminated normally")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
