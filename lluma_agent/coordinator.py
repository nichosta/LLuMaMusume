"""Main game loop coordinator that orchestrates all subsystems."""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lluma_os.capture import CaptureManager, CaptureError
from lluma_os.config import AgentConfig, CaptureConfig, WindowConfig
from lluma_os.input_handler import (
    ButtonNotFoundError,
    InputConfig,
    InputHandler,
    ScrollbarNotFoundError,
    VisionOutput,
    WindowStateError,
    parse_button_label,
)
from lluma_os.window import UmaWindow, WindowNotFoundError, set_dpi_aware
from lluma_vision.menu_analyzer import MenuAnalyzer

from .agent import AgentError, UmaAgent, VisionData
from .memory import MemoryManager

Logger = logging.Logger


def convert_vlm_box_to_pixels(vlm_box: list, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    """Convert VLM normalized box to pixel coordinates.

    VLM outputs boxes as [ymin, xmin, ymax, xmax] in 0-1000 range.
    This converts to [x, y, width, height] in pixels.

    Args:
        vlm_box: VLM box in [ymin, xmin, ymax, xmax] format (0-1000 range)
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels

    Returns:
        (x, y, width, height) in pixels
    """
    ymin, xmin, ymax, xmax = vlm_box

    # Scale from 0-1000 range to actual pixels
    x_px = int(round((xmin / 1000.0) * image_width))
    y_px = int(round((ymin / 1000.0) * image_height))
    x_max_px = int(round((xmax / 1000.0) * image_width))
    y_max_px = int(round((ymax / 1000.0) * image_height))

    # Convert from min/max to x/y/width/height
    width = max(x_max_px - x_px, 1)
    height = max(y_max_px - y_px, 1)

    return (x_px, y_px, width, height)


def calculate_region_offset(region: str, capture_config: CaptureConfig, full_width: int) -> Tuple[int, int]:
    """Calculate the offset of a cropped region within the full client area.

    Args:
        region: Either "primary" or "menus"
        capture_config: Capture configuration with split ratios
        full_width: Full client width in logical pixels

    Returns:
        (x_offset, y_offset) in logical pixels
    """
    if not capture_config.split.enabled:
        return (0, 0)

    split = capture_config.split
    left_pin_end = int(round(full_width * split.left_pin_ratio))
    primary_end = int(round(full_width * (split.left_pin_ratio + split.primary_ratio)))

    if region == "primary":
        return (left_pin_end, 0)
    elif region == "menus":
        return (primary_end, 0)
    else:
        return (0, 0)


def calculate_primary_center(capture_result: Any, capture_config: CaptureConfig) -> Tuple[int, int]:
    """Determine the logical center of the primary region for dialogue clicks."""

    client_area = capture_result.geometry.client_area
    if capture_result.primary_image is None:
        return (
            client_area.width_logical // 2,
            client_area.height_logical // 2,
        )

    scale = client_area.scaling_factor
    primary_width_physical, primary_height_physical = capture_result.primary_image.size
    primary_width_logical = max(int(round(primary_width_physical / scale)), 1)
    primary_height_logical = max(int(round(primary_height_physical / scale)), 1)

    offset_x, offset_y = calculate_region_offset(
        "primary",
        capture_config,
        client_area.width_logical,
    )

    center_x = offset_x + primary_width_logical // 2
    center_y = offset_y + primary_height_logical // 2
    return (center_x, center_y)


class CoordinatorError(RuntimeError):
    """Raised when coordinator encounters unrecoverable errors."""


class GameLoopCoordinator:
    """Orchestrates the main game loop: capture → vision → agent → input."""

    def __init__(
        self,
        window_config: WindowConfig,
        capture_config: CaptureConfig,
        agent_config: AgentConfig,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the coordinator and all subsystems.

        Args:
            window_config: Window management configuration
            capture_config: Screenshot capture configuration
            agent_config: Agent and LLM configuration
            logger: Optional logger instance
        """
        self._logger = logger or logging.getLogger(__name__)
        self._agent_config = agent_config
        self._capture_config = capture_config
        self._should_stop = False
        self._turn_counter = 0

        # Cache for menu button results (reuse if tab hasn't changed and VLM fails)
        self._last_menu_tab: Optional[str] = None
        self._last_menu_buttons_raw: List[Dict[str, Any]] = []

        # Cache for full vision objects (needed by input handler, not serialized in logs)
        self._current_scrollbar_full: Optional[Any] = None  # ScrollbarInfo
        self._current_menu_state_full: Optional[Any] = None  # MenuState

        # Ensure DPI awareness is set before any window operations
        set_dpi_aware()

        # Create output directories
        agent_config.memory_dir.mkdir(parents=True, exist_ok=True)
        agent_config.logs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize subsystems
        self._logger.info("Initializing subsystems...")

        self._window = UmaWindow(window_config, logger=self._logger)
        self._capture = CaptureManager(self._window, capture_config, logger=self._logger)
        self._vision = MenuAnalyzer(logger=self._logger)
        self._memory = MemoryManager(
            agent_config.memory_dir,
            max_tokens=agent_config.max_memory_tokens,
            logger=self._logger,
        )
        self._agent = UmaAgent(
            self._memory,
            model=agent_config.model,
            max_context_tokens=agent_config.max_context_tokens,
            thinking_enabled=agent_config.thinking_enabled,
            thinking_budget_tokens=agent_config.thinking_budget_tokens,
            max_tokens=agent_config.max_tokens,
            summarization_threshold_tokens=agent_config.summarization_threshold_tokens,
            logger=self._logger,
        )
        self._input_handler = InputHandler(
            self._window,
            config=InputConfig(),
            logger=self._logger,
        )

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._logger.info("Coordinator initialized successfully")

    def run(self, reposition: bool = True) -> None:
        """Run the main game loop until stopped.

        Args:
            reposition: Whether to reposition window on first turn
        """
        self._logger.info("Starting game loop (reposition=%s)", reposition)
        first_turn = True

        try:
            while not self._should_stop:
                self._execute_turn(reposition=first_turn and reposition)
                first_turn = False

                # Post-turn padding
                if not self._should_stop:
                    time.sleep(self._agent_config.turn_post_padding_s)

        except KeyboardInterrupt:
            self._logger.info("Received keyboard interrupt, stopping gracefully")
        except WindowNotFoundError:
            self._logger.error("Game window closed; terminating immediately")
            sys.exit(1)
        except Exception as exc:
            self._logger.exception("Unhandled exception in game loop: %s", exc)
            raise CoordinatorError("Game loop failed") from exc
        finally:
            self._logger.info("Game loop terminated")

    def _execute_turn(self, reposition: bool = False) -> None:
        """Execute a single turn of the game loop.

        Args:
            reposition: Whether to reposition the window before capture
        """
        self._turn_counter += 1
        turn_id = f"turn_{self._turn_counter:06d}"
        self._logger.info("=" * 60)
        self._logger.info("Starting %s", turn_id)

        turn_log: Dict[str, Any] = {
            "turn_id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "reposition": reposition,
        }

        try:
            # Step 1: Capture
            self._logger.info("Capturing screenshots...")
            capture_result = self._capture.capture(turn_id, reposition=reposition)
            turn_log["capture"] = {
                "raw_path": str(capture_result.raw_path),
                "primary_path": str(capture_result.primary_path) if capture_result.primary_path else None,
                "menus_path": str(capture_result.menus_path) if capture_result.menus_path else None,
                "tabs_path": str(capture_result.tabs_path) if capture_result.tabs_path else None,
                "validation": asdict(capture_result.validation),
            }

            # Step 2: Vision processing
            self._logger.info("Processing vision data...")
            vision_data = self._process_vision(capture_result)
            turn_log["vision"] = {
                "buttons": vision_data.buttons,
                "scrollbar": vision_data.scrollbar,
                "menu_state": vision_data.menu_state,
            }

            # Check for no buttons (failsafe condition)
            if not vision_data.buttons:
                self._logger.warning("No buttons detected; attempting failsafe advanceDialogue")
                turn_log["failsafe"] = "no_buttons_detected"
                self._execute_failsafe_action(capture_result)
                self._save_turn_log(turn_id, turn_log)
                return

            # Step 3: Agent reasoning
            self._logger.info("Calling agent for decision...")
            turn_result = self._agent.execute_turn(
                turn_id=turn_id,
                vision_data=vision_data,
                primary_screenshot=capture_result.primary_path or capture_result.raw_path,
                menus_screenshot=capture_result.menus_path,
            )
            turn_log["agent"] = {
                "reasoning": turn_result.reasoning,
                "thinking": turn_result.thinking,
                "memory_actions": [
                    {"name": a["name"], "args_keys": list(a["arguments"].keys())}
                    for a in turn_result.memory_actions
                ],
                "input_action": turn_result.input_action,
                "execution_results": turn_result.execution_results,
                "usage": turn_result.usage,
            }

            # Step 4: Execute input action
            if turn_result.input_action:
                self._logger.info("Executing input action: %s", turn_result.input_action["name"])
                result = self._execute_input_action(turn_result.input_action, vision_data, capture_result)
                turn_log["input_execution"] = result
            else:
                self._logger.info("No input action from agent")
                turn_log["input_execution"] = {"status": "no_action"}

        except CaptureError as exc:
            self._logger.error("Capture failed: %s", exc)
            turn_log["error"] = {"type": "capture", "message": str(exc)}
        except AgentError as exc:
            self._logger.error("Agent failed: %s", exc)
            turn_log["error"] = {"type": "agent", "message": str(exc)}
        except Exception as exc:
            self._logger.exception("Turn execution failed: %s", exc)
            turn_log["error"] = {"type": "unknown", "message": str(exc)}
        finally:
            self._save_turn_log(turn_id, turn_log)
            self._logger.info("Completed %s", turn_id)

    def _process_vision(self, capture_result: Any) -> VisionData:
        """Process vision data from capture results.

        Args:
            capture_result: CaptureResult object from capture manager

        Returns:
            VisionData for agent consumption
        """
        # Clear cached full objects from previous turn
        self._current_scrollbar_full = None
        self._current_menu_state_full = None

        buttons = []
        scrollbar_info = None
        menu_state_dict: Dict[str, Any] = {
            "tab": None,
            "available": [],
        }

        # Process menus region
        if (
            capture_result.menus_path
            and capture_result.menus_path.exists()
            and capture_result.menus_image is not None
        ):
            # Menu state analysis (compact format for agent)
            menu_state = self._vision.analyze_menu(
                capture_result.menus_image,
                tabs_image=capture_result.tabs_image,
            )
            # Store full menu state for input handler (not in VisionData to keep it JSON-serializable)
            self._current_menu_state_full = menu_state

            # Compact format: just current tab and available tabs
            menu_state_dict = {
                "tab": menu_state.selected_tab,
                "available": menu_state.available_tabs(),
            }

            # Button detection in menus
            if menu_state.is_usable:
                menu_buttons_raw = self._vision.get_clickable_buttons(str(capture_result.menus_path))

                # If VLM returned no buttons but tab hasn't changed, reuse previous results
                if (
                    not menu_buttons_raw
                    and menu_state.selected_tab is not None
                    and menu_state.selected_tab == self._last_menu_tab
                    and self._last_menu_buttons_raw
                ):
                    self._logger.info(
                        "Menu VLM returned no buttons for tab '%s'; reusing %d buttons from previous turn",
                        menu_state.selected_tab,
                        len(self._last_menu_buttons_raw),
                    )
                    menu_buttons_raw = self._last_menu_buttons_raw

                # Update cache for next turn
                self._last_menu_tab = menu_state.selected_tab
                self._last_menu_buttons_raw = menu_buttons_raw

                # Images are in physical pixels, convert to logical for coordinate system
                menus_width_physical, menus_height_physical = capture_result.menus_image.size
                scaling_factor = capture_result.geometry.client_area.scaling_factor
                menus_width_logical = int(round(menus_width_physical / scaling_factor))
                menus_height_logical = int(round(menus_height_physical / scaling_factor))

                full_width = capture_result.geometry.client_area.width_logical
                offset_x, offset_y = calculate_region_offset("menus", self._capture_config, full_width)

                for btn_raw in menu_buttons_raw:
                    name, metadata = parse_button_label(btn_raw["label"])
                    # Convert VLM box [ymin, xmin, ymax, xmax] (0-1000) to logical pixel bounds
                    x, y, w, h = convert_vlm_box_to_pixels(
                        btn_raw["box_2d"],
                        menus_width_logical,
                        menus_height_logical
                    )
                    # Apply region offset to convert to full client coordinates
                    bounds = (x + offset_x, y + offset_y, w, h)

                    # Build compact button entry for agent (no bounds, no full_label)
                    # Bounds are preserved in VisionOutput for input handler
                    button_entry = {
                        "name": name,
                        "region": "menus",
                    }
                    # Add metadata only if non-empty (compact string format)
                    if metadata:
                        meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
                        button_entry["meta"] = meta_str

                    # Store bounds internally for input handler (will be added to VisionOutput)
                    button_entry["_bounds"] = bounds  # Temporary, used by input handler setup
                    buttons.append(button_entry)
            else:
                # Menu not usable; clear caches
                self._last_menu_tab = None
                self._last_menu_buttons_raw = []
                self._current_menu_state_full = None

        # Process primary region
        if capture_result.primary_path and capture_result.primary_path.exists():
            # Button detection
            primary_buttons_raw = self._vision.get_primary_elements(str(capture_result.primary_path))
            # Images are in physical pixels, convert to logical for coordinate system
            primary_width_physical, primary_height_physical = capture_result.primary_image.size
            scaling_factor = capture_result.geometry.client_area.scaling_factor
            primary_width_logical = int(round(primary_width_physical / scaling_factor))
            primary_height_logical = int(round(primary_height_physical / scaling_factor))

            full_width = capture_result.geometry.client_area.width_logical
            offset_x, offset_y = calculate_region_offset("primary", self._capture_config, full_width)

            for btn_raw in primary_buttons_raw:
                name, metadata = parse_button_label(btn_raw["label"])
                # Convert VLM box [ymin, xmin, ymax, xmax] (0-1000) to logical pixel bounds
                x, y, w, h = convert_vlm_box_to_pixels(
                    btn_raw["box_2d"],
                    primary_width_logical,
                    primary_height_logical
                )
                # Apply region offset to convert to full client coordinates
                bounds = (x + offset_x, y + offset_y, w, h)

                # Build compact button entry for agent (no bounds, no full_label)
                # Bounds are preserved in VisionOutput for input handler
                button_entry = {
                    "name": name,
                    "region": "primary",
                }
                # Add metadata only if non-empty (compact string format)
                if metadata:
                    meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
                    button_entry["meta"] = meta_str

                # Store bounds internally for input handler (will be added to VisionOutput)
                button_entry["_bounds"] = bounds  # Temporary, used by input handler setup
                buttons.append(button_entry)

            # Scrollbar detection (compact format for agent)
            scrollbar = self._vision.detect_primary_scrollbar(capture_result.primary_image)
            if scrollbar:
                # Store full scrollbar for input handler (not in VisionData to keep it JSON-serializable)
                self._current_scrollbar_full = scrollbar

                # Compact format: only up/down booleans (agent doesn't need bounds/ratios)
                scrollbar_info = {
                    "up": scrollbar.can_scroll_up,
                    "down": scrollbar.can_scroll_down,
                }

        return VisionData(
            buttons=buttons,
            scrollbar=scrollbar_info,
            menu_state=menu_state_dict,
        )

    def _execute_input_action(
        self, action: Dict[str, Any], vision_data: VisionData, capture_result: Any
    ) -> Dict[str, str]:
        """Execute an input action via the input handler.

        Args:
            action: Action dictionary with 'name' and 'arguments'
            vision_data: Vision data for the turn
            capture_result: Capture result for geometry

        Returns:
            Dictionary with status and optional error message
        """
        action_name = action["name"]
        args = action.get("arguments", {})

        # Build VisionOutput for input handler (extract bounds from _bounds field)
        button_infos = []
        for btn in vision_data.buttons:
            from lluma_os.input_handler import ButtonInfo

            # Reconstruct full_label from name + metadata for input handler
            full_label = btn["name"]
            metadata_dict = {}
            if "meta" in btn:
                # Parse compact metadata string back to dict
                for pair in btn["meta"].split(", "):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        metadata_dict[k] = v

            button_infos.append(
                ButtonInfo(
                    name=btn["name"],
                    full_label=full_label,
                    bounds=tuple(btn["_bounds"]),  # type: ignore - Extract stored bounds
                    metadata=metadata_dict,
                )
            )

        scrollbar_for_handler = None
        if self._current_scrollbar_full is not None:
            from lluma_os.input_handler import ScrollbarInfo

            # Use cached full scrollbar info
            full_scrollbar = self._current_scrollbar_full
            scrollbar_for_handler = ScrollbarInfo(
                track_bounds=tuple(full_scrollbar.track_bounds),  # type: ignore
                thumb_bounds=tuple(full_scrollbar.thumb_bounds),  # type: ignore
                can_scroll_up=full_scrollbar.can_scroll_up,
                can_scroll_down=full_scrollbar.can_scroll_down,
                thumb_ratio=full_scrollbar.thumb_ratio,
            )

        # Primary center (use center of primary image)
        primary_center = calculate_primary_center(capture_result, self._capture_config)

        vision_output = VisionOutput(
            buttons=button_infos,
            scrollbar=scrollbar_for_handler,
            primary_center=primary_center,
        )

        # Update input handler state
        self._input_handler.update_vision_state(vision_output, capture_result.geometry)

        # Execute action
        try:
            if action_name == "pressButton":
                button_name = args["name"]
                self._input_handler.press_button(button_name)
                return {"status": "success", "action": f"pressButton({button_name})"}

            elif action_name == "advanceDialogue":
                self._input_handler.advance_dialogue()
                return {"status": "success", "action": "advanceDialogue()"}

            elif action_name == "back":
                self._input_handler.back()
                return {"status": "success", "action": "back()"}

            elif action_name == "confirm":
                self._input_handler.confirm()
                return {"status": "success", "action": "confirm()"}

            elif action_name == "scrollUp":
                self._input_handler.scroll_up()
                return {"status": "success", "action": "scrollUp()"}

            elif action_name == "scrollDown":
                self._input_handler.scroll_down()
                return {"status": "success", "action": "scrollDown()"}

            else:
                return {"status": "error", "message": f"Unknown action: {action_name}"}

        except ButtonNotFoundError as exc:
            self._logger.error("Button not found: %s", exc)
            return {"status": "error", "message": str(exc)}
        except ScrollbarNotFoundError as exc:
            self._logger.error("Scrollbar not found: %s", exc)
            return {"status": "error", "message": str(exc)}
        except WindowStateError as exc:
            self._logger.error("Window state error: %s", exc)
            return {"status": "error", "message": str(exc)}
        except Exception as exc:
            self._logger.exception("Input action failed: %s", exc)
            return {"status": "error", "message": str(exc)}

    def _execute_failsafe_action(self, capture_result: Any) -> None:
        """Execute failsafe advanceDialogue when no buttons detected.

        Args:
            capture_result: Capture result for geometry
        """
        try:
            # Build minimal VisionOutput with no buttons
            if capture_result.primary_image:
                w, h = capture_result.primary_image.size
                primary_center = (w // 2, h // 2)
            else:
                primary_center = (500, 500)

            vision_output = VisionOutput(
                buttons=[],
                scrollbar=None,
                primary_center=primary_center,
            )

            self._input_handler.update_vision_state(vision_output, capture_result.geometry)
            self._input_handler.advance_dialogue()
            self._logger.info("Failsafe advanceDialogue executed successfully")

        except Exception as exc:
            self._logger.error("Failsafe action failed: %s", exc)

    def _save_turn_log(self, turn_id: str, log_data: Dict[str, Any]) -> None:
        """Save turn log to JSON file.

        Args:
            turn_id: Turn identifier
            log_data: Log data dictionary
        """
        log_path = self._agent_config.logs_dir / f"{turn_id}.json"
        try:
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)
        except Exception as exc:
            self._logger.error("Failed to save turn log: %s", exc)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle termination signals gracefully.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self._logger.info("Received signal %d, stopping after current turn", signum)
        self._should_stop = True


__all__ = ["GameLoopCoordinator", "CoordinatorError"]
