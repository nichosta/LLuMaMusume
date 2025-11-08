"""Menu state analysis for Uma Musume screenshots."""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

Logger = logging.Logger


class TabAvailability(Enum):
    """Tab availability states."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class TabInfo:
    """Information about a single menu tab."""
    name: str
    is_selected: bool
    availability: TabAvailability


@dataclass(slots=True)
class MenuState:
    """Complete menu state analysis result."""
    is_usable: bool  # False if blurred/inactive
    selected_tab: Optional[str]
    tabs: List[TabInfo]

    def available_tabs(self) -> List[str]:
        """Return the names of tabs detected as available for interaction."""

        return [tab.name for tab in self.tabs if tab.availability == TabAvailability.AVAILABLE]


@dataclass(slots=True)
class ScrollbarInfo:
    """Describes a detected vertical scrollbar in the primary view."""

    track_bounds: tuple[int, int, int, int]  # x, y, width, height
    thumb_bounds: tuple[int, int, int, int]  # x, y, width, height
    can_scroll_up: bool
    can_scroll_down: bool
    thumb_ratio: float  # 0.0 (top) → 1.0 (bottom)


class MenuAnalyzer:
    """Analyzes Uma Musume menu screenshots to extract tab states."""
    
    # Tab names from top to bottom
    TAB_NAMES = [
        "Jukebox",
        "Sparks",
        "Log",
        "Career Profile",
        "Agenda",
        "Item Request",
        "Menu"
    ]
    PRIMARY_SCROLLBAR_BAND_WIDTH = 220  # width of primary-image band used for scrollbar detection
    SCROLLBAR_MIN_WIDTH = 8
    SCROLLBAR_MAX_WIDTH = 16
    SCROLLBAR_SEARCH_WIDTH = 200  # restrict detection to this many pixels from the right edge
    SCROLLBAR_MIN_EDGE_STRENGTH = 18.0
    SCROLLBAR_MAX_SCORE = 4.0
    SCROLLBAR_SMOOTH_KERNEL = 9
    SCROLLBAR_THUMB_DELTA = 28.0
    SCROLLBAR_MIN_THUMB_RATIO = 0.02
    SCROLLBAR_MIN_THUMB_PIXELS = 20
    SCROLLBAR_END_MARGIN_RATIO = 0.015
    SCROLLBAR_END_MARGIN_MIN = 10

    def __init__(self, logger: Optional[Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._api_key: Optional[str] = None
        self._headers: Optional[Dict[str, str]] = None

    def analyze_menu(self, menu_image: Image.Image, tabs_image: Optional[Image.Image] = None) -> MenuState:
        """Analyze a menu section image and return the menu state.

        Args:
            menu_image: Cropped menu content pane (without left pin or tabs).
            tabs_image: Optional crop that isolates the vertical tab list.
        """

        # Step 1: Check if menu is usable (not blurred)
        is_usable = self._detect_usability(menu_image)
        tab_reference = tabs_image or menu_image
        if tab_reference is None:
            tab_reference = menu_image
        
        if not is_usable:
            # If blurred, we can't reliably detect tab states
            return MenuState(
                is_usable=False,
                selected_tab=None,
                tabs=[TabInfo(name, False, TabAvailability.UNKNOWN) for name in self.TAB_NAMES]
            )
        
        # Step 2: Detect selected tab (green highlight)
        selected_tab = self._detect_selected_tab(tab_reference)
        
        # Step 3: Detect available tabs (white vs gray boxes)
        tab_availabilities = self._detect_tab_availability(tab_reference)
        
        # Build tab info list
        tabs = []
        for i, name in enumerate(self.TAB_NAMES):
            is_selected = (name == selected_tab)
            availability = tab_availabilities.get(i, TabAvailability.UNKNOWN)
            tabs.append(TabInfo(name, is_selected, availability))
        
        return MenuState(
            is_usable=True,
            selected_tab=selected_tab,
            tabs=tabs
        )
    
    def _detect_usability(self, image: Image.Image) -> bool:
        """Detect if the menu is usable (not blurred/inactive)."""
        
        # Convert to grayscale numpy array for analysis
        gray = np.asarray(image.convert("L"), dtype=np.float32)

        # Small images cannot provide a meaningful sharpness signal
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            self._logger.debug("Menu image too small for sharpness detection; marking unusable")
            return False

        # Calculate image sharpness using Laplacian variance
        # Blurred images have low variance in the Laplacian
        laplacian_kernel = np.array(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )

        # Vectorized convolution using sliding window view for better performance
        windows = sliding_window_view(gray, (3, 3))
        laplacian_response = np.einsum("ijkl,kl->ij", windows, laplacian_kernel)
        sharpness = float(np.mean(laplacian_response * laplacian_response))
        
        # Threshold for determining if image is blurred
        # This threshold may need tuning based on actual data
        blur_threshold = 100.0
        
        is_sharp = sharpness > blur_threshold
        self._logger.debug(f"Menu sharpness: {sharpness:.2f}, usable: {is_sharp}")
        
        return is_sharp
    
    def _detect_selected_tab(self, image: Image.Image) -> Optional[str]:
        """Detect which tab is selected (has green highlight)."""
        
        img_array = np.asarray(image)
        height, width = img_array.shape[:2]
        
        # For green detection, use full width to capture all highlighting
        # The green highlight can extend across the full menu content area
        tab_height = height // len(self.TAB_NAMES)
        
        max_green_score = 0
        selected_tab_idx = None
        
        for i, tab_name in enumerate(self.TAB_NAMES):
            # Define region for this tab (full width for reliable green detection)
            y_start = i * tab_height
            y_end = min((i + 1) * tab_height, height)
            
            # Extract full tab region 
            tab_region = img_array[y_start:y_end, :, :]
            
            # Calculate green score for this region
            green_score = self._calculate_green_score(tab_region)
            
            self._logger.debug(f"Tab {tab_name}: green_score={green_score:.2f}")
            
            if green_score > max_green_score:
                max_green_score = green_score
                selected_tab_idx = i
        
        # Threshold for green detection (based on analysis of reference images)
        green_threshold = 10.0  # Green dominance threshold
        
        if max_green_score > green_threshold and selected_tab_idx is not None:
            selected_tab = self.TAB_NAMES[selected_tab_idx]
            self._logger.debug(f"Selected tab detected: {selected_tab}")
            return selected_tab
        
        self._logger.debug("No clearly selected tab detected")
        return None
    
    def _calculate_green_score(self, region: np.ndarray) -> float:
        """Calculate how 'green' a region is (green dominance for selected tabs)."""
        
        if len(region.shape) != 3 or region.shape[2] < 3:
            return 0.0
        
        # Extract RGB channels and calculate means
        r_mean = float(np.mean(region[:, :, 0]))
        g_mean = float(np.mean(region[:, :, 1]))
        b_mean = float(np.mean(region[:, :, 2]))
        
        # Green dominance: green channel minus the higher of red/blue
        green_dominance = g_mean - max(r_mean, b_mean)
        
        return green_dominance
    
    def _detect_tab_availability(self, image: Image.Image) -> dict[int, TabAvailability]:
        """Detect which tabs are available (white boxes) vs unavailable (gray boxes)."""
        
        img_array = np.asarray(image)
        height, width = img_array.shape[:2]
        
        # Focus on the right edge backing where the active/inactive fill resides.
        tab_width = max(int(width * 0.15), 1)
        tab_region_start = max(width - tab_width, 0)
        tab_region_end = width
        tab_height = height // len(self.TAB_NAMES)
        availabilities = {}

        for i, tab_name in enumerate(self.TAB_NAMES):
            # Define region for this tab (only the tab area, not full width)
            y_start = i * tab_height
            y_end = min((i + 1) * tab_height, height)

            # Extract only the tab region (right edge backing without the icons)
            tab_region = img_array[y_start:y_end, tab_region_start:tab_region_end, :]
            
            # Calculate brightness/saturation to distinguish white vs gray
            availability = self._classify_tab_availability(tab_region)
            availabilities[i] = availability
            
            self._logger.debug(f"Tab {tab_name}: {availability.value}")
        
        return availabilities
    
    def _classify_tab_availability(self, region: np.ndarray) -> TabAvailability:
        """Classify a tab region as available or unavailable."""
        
        if len(region.shape) != 3 or region.shape[2] < 3:
            return TabAvailability.UNKNOWN
        
        # Calculate mean brightness and color metrics
        r_mean = float(np.mean(region[:, :, 0]))
        g_mean = float(np.mean(region[:, :, 1]))
        b_mean = float(np.mean(region[:, :, 2]))
        
        brightness = (r_mean + g_mean + b_mean) / 3
        total = r_mean + g_mean + b_mean + 1e-6
        
        # Calculate RGB ratios
        r_ratio = r_mean / total
        g_ratio = g_mean / total
        b_ratio = b_mean / total
        
        # RGB balance: how much color variation vs neutral gray
        # Higher values = more colorful/available, lower = more gray/unavailable
        rgb_balance = abs(r_ratio - g_ratio) + abs(g_ratio - b_ratio) + abs(r_ratio - b_ratio)
        
        # Based on analysis of reference images:
        # Available tabs: rgb_balance > 0.040 OR brightness > 230
        # Unavailable tabs: rgb_balance <= 0.040 AND brightness <= 230
        balance_threshold = 0.075
        brightness_threshold = 225.0
        
        if rgb_balance > balance_threshold or brightness > brightness_threshold:
            if brightness > brightness_threshold:
                # Flat/no-edge regions with very high brightness correspond to greyed-out tabs
                region_gray = np.dot(region[..., :3], [0.299, 0.587, 0.114])
                gy, gx = np.gradient(region_gray)
                edge_strength = float(np.mean(np.sqrt(gx * gx + gy * gy)))
                edge_threshold = 0.04

                if edge_strength < edge_threshold:
                    return TabAvailability.UNAVAILABLE

            return TabAvailability.AVAILABLE
        else:
            return TabAvailability.UNAVAILABLE

    def get_clickable_buttons(self, image_path: str) -> list[dict]:
        """
        Uses a VLM to get clickable buttons from a menu image.

        Args:
            image_path: The path to the image file.

        Returns:
            List of button dictionaries with 'label' and 'box_2d' keys.
        """
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_file}")

        self._ensure_api_configured()

        system_prompt = (
            "You analyse Uma Musume menu content screenshots to find interactive buttons. "
            "The vertical tab column has already been separated into a different image, so this crop contains only the menu's content area. "
            "Respond with JSON listing each distinct clickable button once. "
            "Each record must contain a `label` string and `box_2d` array representing [ymin, xmin, ymax, xmax] "
            "with values in the range 0-1000 (normalised coordinates). The label should begin with the button name; "
            "optionally append metadata like `|section=menus` or `|hint=...`, but avoid confidence, state, or type tags. "
            "IMPORTANT: If two buttons would have the same name, add a distinguishing identifier to the name itself "
            "(not just in metadata tags) so each button name is unique."
        )

        user_instructions = (
            "Return a JSON object with a single property `buttons` that is an array of button records. "
            "Every record must have `label` (string) and `box_2d` (array of four floats). "
            "Exclude decorative or inactive elements. If no buttons are present, respond with `{\"buttons\": []}`."
        )

        base64_image = self._prepare_api_image(image_file)

        payload = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_instructions},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        }

        # Allow up to 3 total attempts (1 initial + 2 retries)
        max_attempts = 3
        content = None

        for attempt in range(max_attempts):
            buttons: list[dict] = []  # type: ignore
            parse_ok = False
            try:
                model_name = "google/gemini-2.5-flash-lite-preview-09-2025"
                request_body = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        payload,
                    ],
                    "response_format": {"type": "json_object"},
                    "reasoning": {"enabled": False},  # No reasoning is recommended by Google
                }

                self._logger.info(
                    "Calling OpenRouter menus vision model %s for %s",
                    model_name,
                    image_file.name,
                )
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=self._headers,
                    json=request_body,
                    timeout=60,
                )
                response.raise_for_status()
                response_data = response.json()
                content = self._extract_response_content(response_data)
                if content is not None:
                    buttons, parse_ok = self._parse_buttons_payload(
                        content, return_status=True
                    )
            except Exception as exc:  # pragma: no cover - network failure path
                self._logger.error(
                    "Error querying OpenRouter menus vision for %s: %s\n"
                    "System prompt: %s\n"
                    "User instructions: %s",
                    image_file.name,
                    exc,
                    system_prompt,
                    user_instructions,
                )
                # Try to log response if we got one
                try:
                    if 'response' in locals():
                        self._logger.error("Response status: %d, content: %s", response.status_code, response.text[:500])
                except Exception:
                    pass
                return []

            if parse_ok:
                # If we have buttons, we're done
                if buttons:
                    return buttons

                # If we have no buttons but more attempts left, log and retry
                if attempt < max_attempts - 1:
                    self._logger.warning(
                        "Menus vision returned empty list for %s; retrying (attempt %d/%d)",
                        image_file.name,
                        attempt + 2,
                        max_attempts,
                    )
                    continue
                else:
                    self._logger.info("Menus vision returned no buttons for %s after %d attempts", image_file.name, max_attempts)
                    return []

            # If parsing failed but we have more attempts left, log and retry
            if attempt < max_attempts - 1:
                self._logger.warning(
                    "Menus vision returned invalid JSON for %s; retrying (attempt %d/%d)\n"
                    "Response content: %s",
                    image_file.name,
                    attempt + 2,
                    max_attempts,
                    content[:500] if content else "(no content)",
                )
                continue

        # This part is reached after all attempts fail to parse
        self._logger.error(
            "Menus vision failed to parse JSON after %d attempts for %s",
            max_attempts,
            image_file.name,
        )
        return []

    def get_primary_elements(self, image_path: str) -> list[dict]:
        """Detect primary-region buttons via the VLM with retry logic."""

        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_file}")

        self._ensure_api_configured()
        base64_image = self._prepare_api_image(image_file)

        system_prompt = (
            "You analyse Uma Musume primary gameplay captures to find clickable UI elements. "
            "Return structured JSON so an agent can decide interactions. "
            "List each distinct clickable button once with its label and bounds. "
            "IMPORTANT: If two buttons would have the same name, add a distinguishing identifier to the name itself "
            "(not just in metadata tags) so each button name is unique."
        )

        user_instructions = (
            "Respond with a JSON object that includes a `buttons` array. "
            "Each button entry must have `label` and `box_2d` fields. Avoid confidence/state/type suffixes unless they add useful hints like `|hint=New`. "
            "If no buttons exist, use an empty array."
        )

        payload = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_instructions},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        }

        # Try up to 2 times if parse fails
        parse_retry_allowed = True
        content = None

        while True:
            buttons: list[dict] = []
            parse_ok = False
            try:
                model_name = "google/gemini-2.5-flash-lite-preview-09-2025"
                request_body = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        payload,
                    ],
                    "response_format": {"type": "json_object"},
                    "reasoning": {"enabled": False},  # No reasoning is recommended by Google
                }

                self._logger.info(
                    "Calling OpenRouter primary vision model %s for %s", model_name, image_file.name
                )
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=self._headers,
                    json=request_body,
                    timeout=60,
                )
                response.raise_for_status()
                response_data = response.json()
                content = self._extract_response_content(response_data)
                if content is not None:
                    buttons, parse_ok = self._parse_buttons_payload(
                        content, return_status=True
                    )
            except Exception as exc:  # pragma: no cover - network failure path
                self._logger.error(
                    "Error querying OpenRouter primary vision for %s: %s\n"
                    "System prompt: %s\n"
                    "User instructions: %s",
                    image_file.name,
                    exc,
                    system_prompt,
                    user_instructions,
                )
                # Try to log response if we got one
                try:
                    if 'response' in locals():
                        self._logger.error("Response status: %d, content: %s", response.status_code, response.text[:500])
                except Exception:
                    pass
                return []

            if parse_ok:
                if not buttons:
                    self._logger.info("Primary vision parsed successfully but returned no buttons for %s", image_file.name)
                return buttons

            if parse_retry_allowed:
                parse_retry_allowed = False
                self._logger.warning(
                    "Primary vision returned invalid JSON for %s; retrying once\n"
                    "Response content: %s",
                    image_file.name,
                    content[:500] if content else "(no content)",
                )
                continue

            # Parse failed twice; give up
            self._logger.error(
                "Primary vision failed to parse JSON after 2 attempts for %s\n"
                "Last response: %s",
                image_file.name,
                content[:500] if content else "(no content)",
            )
            return []

    def detect_primary_scrollbar(self, primary_image: Image.Image) -> Optional[ScrollbarInfo]:
        """Heuristically detect a vertical scrollbar within a primary capture."""

        arr = np.asarray(primary_image, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            self._logger.debug("Primary image lacks expected RGB channels; skipping scrollbar detection")
            return None

        height, width, _ = arr.shape
        if width < 2:
            return None

        band_width = min(self.PRIMARY_SCROLLBAR_BAND_WIDTH, width)
        band = arr[:, width - band_width :, :]

        # Compute luma channel for gradient/variance analysis
        luma = 0.2126 * band[:, :, 0] + 0.7152 * band[:, :, 1] + 0.0722 * band[:, :, 2]
        if luma.shape[1] < 2:
            return None

        gradients = np.abs(np.diff(luma, axis=1))
        mean_grad = gradients.mean(axis=0)
        grad_threshold = float(mean_grad.mean() + mean_grad.std())
        edge_candidates = [idx for idx, value in enumerate(mean_grad) if value >= grad_threshold]

        if not edge_candidates:
            return None

        best_candidate: Optional[tuple[float, int, int, float]] = None
        for left_edge in edge_candidates:
            for right_edge in edge_candidates:
                if right_edge <= left_edge:
                    continue

                window_width = right_edge - left_edge
                if not (self.SCROLLBAR_MIN_WIDTH <= window_width <= self.SCROLLBAR_MAX_WIDTH):
                    continue

                global_left = width - band_width + left_edge
                search_width = min(self.SCROLLBAR_SEARCH_WIDTH, band_width)
                if global_left < width - search_width:
                    continue

                edge_strength = float(mean_grad[left_edge] + mean_grad[right_edge - 1])
                if edge_strength < self.SCROLLBAR_MIN_EDGE_STRENGTH:
                    continue

                window = luma[:, left_edge:right_edge]
                inside_std = float(np.std(window))
                score = inside_std / (edge_strength + 1e-6)

                if score > self.SCROLLBAR_MAX_SCORE:
                    continue

                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = (score, left_edge, right_edge, edge_strength)

        if best_candidate is None:
            return None

        _, left_idx, right_idx, edge_strength = best_candidate
        track_x0 = width - band_width + left_idx
        track_x1 = width - band_width + right_idx

        window = luma[:, left_idx:right_idx]
        row_mean = window.mean(axis=1)

        # Smooth to reduce row-level noise
        kernel = np.ones(self.SCROLLBAR_SMOOTH_KERNEL, dtype=np.float32) / self.SCROLLBAR_SMOOTH_KERNEL
        smooth = np.convolve(row_mean, kernel, mode="same")

        bright_reference = float(np.percentile(smooth, 80))
        thumb_threshold = bright_reference - self.SCROLLBAR_THUMB_DELTA

        mask = smooth <= thumb_threshold
        min_thumb_span = max(
            self.SCROLLBAR_MIN_THUMB_PIXELS,
            int(height * self.SCROLLBAR_MIN_THUMB_RATIO),
        )

        thumb_segment = self._largest_segment(mask, min_thumb_span)
        if thumb_segment is None:
            self._logger.debug("Detected scrollbar track but no thumb segment matched thresholds")
            return None

        thumb_top, thumb_bottom = thumb_segment
        track_width = track_x1 - track_x0
        thumb_height = thumb_bottom - thumb_top + 1

        margin = max(self.SCROLLBAR_END_MARGIN_MIN, int(height * self.SCROLLBAR_END_MARGIN_RATIO))
        can_scroll_up = thumb_top > margin
        can_scroll_down = thumb_bottom < (height - margin)

        thumb_center = (thumb_top + thumb_bottom) / 2.0
        thumb_ratio = float(np.clip(thumb_center / max(height - 1, 1), 0.0, 1.0))

        track_bounds = (int(track_x0), 0, int(track_width), int(height))
        thumb_bounds = (int(track_x0), int(thumb_top), int(track_width), int(thumb_height))

        return ScrollbarInfo(
            track_bounds=track_bounds,
            thumb_bounds=thumb_bounds,
            can_scroll_up=can_scroll_up,
            can_scroll_down=can_scroll_down,
            thumb_ratio=thumb_ratio,
        )

    def _prepare_api_image(self, image_file: Path) -> str:
        """Load and encode the image for VLM API calls."""
        with Image.open(image_file) as image:
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded

    def _largest_segment(self, mask: np.ndarray, min_length: int) -> Optional[tuple[int, int]]:
        """Return the longest contiguous True segment meeting the minimum length."""

        best: Optional[tuple[int, int]] = None
        start: Optional[int] = None

        for idx, value in enumerate(mask):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                end = idx - 1
                if end - start + 1 >= min_length:
                    if best is None or (end - start) > (best[1] - best[0]):
                        best = (start, end)
                start = None

        if start is not None:
            end = len(mask) - 1
            if end - start + 1 >= min_length:
                if best is None or (end - start) > (best[1] - best[0]):
                    best = (start, end)

        return best

    def _ensure_api_configured(self) -> None:
        """Ensure OpenRouter API configuration is initialized."""

        if self._headers is not None:
            return

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/nichosta/LLuMaMusume",
            "X-Title": "LLuMa Musume Agent",
            "Content-Type": "application/json",
        }

    def _extract_response_content(self, response: Dict[str, Any]) -> Optional[str]:
        """Safely pull the content string from an OpenRouter response dict."""

        try:
            choices = response.get("choices")
            if not choices:
                return None
            message = choices[0]["message"]
            content = message.get("content")
            if not content:
                return None
            return str(content).strip()
        except Exception as exc:  # pragma: no cover - defensive guard
            self._logger.error("Unexpected OpenRouter response schema: %s", exc)
            return None

    def _parse_buttons_payload(
        self, raw_json: str, *, return_status: bool = False
    ) -> Any:
        """Parse a JSON payload containing a buttons array.

        When ``return_status`` is ``False`` this method mirrors the historical
        behaviour used by tests by returning only the parsed button list. Internal
        callers that need to know whether decoding succeeded can request the status
        flag via ``return_status=True``.
        """

        payload, parse_ok = self._load_json_document(raw_json)
        if not parse_ok:
            return ([], False) if return_status else []

        buttons_field: Any
        if isinstance(payload, dict):
            buttons_field = payload.get("buttons")
        else:
            buttons_field = payload

        parsed = self._parse_buttons_array(buttons_field)
        return (parsed, True) if return_status else parsed

    def _parse_buttons_array(self, buttons_field: Any) -> list[dict]:
        """Normalize a list of button entries."""

        if not isinstance(buttons_field, list):
            self._logger.debug("OpenRouter payload missing buttons list")
            return []

        buttons: list[dict] = []
        for idx, entry in enumerate(buttons_field):
            if not isinstance(entry, dict):
                self._logger.debug("Skipping non-dict button entry at index %d", idx)
                continue

            label = entry.get("label")
            box = entry.get("box_2d")
            if not label or not isinstance(label, str):
                self._logger.debug("Skipping button without valid label at index %d", idx)
                continue

            if not isinstance(box, list) or len(box) != 4:
                self._logger.debug("Skipping button with invalid box_2d at index %d", idx)
                continue

            valid_box = self._normalise_box_values(box)
            buttons.append({"label": label, "box_2d": valid_box})

        return buttons

    def _load_json_document(self, raw_json: str) -> Tuple[Optional[Any], bool]:
        """Strip common wrappers and parse JSON content."""

        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].strip()

        try:
            return json.loads(cleaned), True
        except json.JSONDecodeError as exc:
            self._logger.error("Failed to decode OpenRouter JSON: %s", exc)
            return None, False

    def _normalise_box_values(self, box: list[Any]) -> list[float]:
        """Convert box values to floats clamped to the expected 0-1000 range."""

        cleaned: list[float] = []
        for value in box:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = 0.0
            cleaned.append(max(0.0, min(1000.0, numeric)))
        return cleaned

    # Scrollbar detection is handled outside of the VLM for now.
