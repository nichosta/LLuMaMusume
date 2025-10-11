"""Menu state analysis for Uma Musume screenshots."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
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
    
    def __init__(self, logger: Optional[Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
    
    def analyze_menu(self, menu_image: Image.Image) -> MenuState:
        """Analyze a menu section image and return the menu state."""
        
        # Step 1: Check if menu is usable (not blurred)
        is_usable = self._detect_usability(menu_image)
        
        if not is_usable:
            # If blurred, we can't reliably detect tab states
            return MenuState(
                is_usable=False,
                selected_tab=None,
                tabs=[TabInfo(name, False, TabAvailability.UNKNOWN) for name in self.TAB_NAMES]
            )
        
        # Step 2: Detect selected tab (green highlight)
        selected_tab = self._detect_selected_tab(menu_image)
        
        # Step 3: Detect available tabs (white vs gray boxes)
        tab_availabilities = self._detect_tab_availability(menu_image)
        
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
        
        # For availability detection, use narrower area focused on tab buttons
        # The actual tab buttons are in the leftmost portion 
        tab_width = int(width * 0.25)  # Use 25% of menu width for availability detection
        tab_height = height // len(self.TAB_NAMES)
        availabilities = {}
        
        for i, tab_name in enumerate(self.TAB_NAMES):
            # Define region for this tab (only the tab area, not full width)
            y_start = i * tab_height
            y_end = min((i + 1) * tab_height, height)
            
            # Extract only the tab region (leftmost portion)
            tab_region = img_array[y_start:y_end, 0:tab_width, :]
            
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
        balance_threshold = 0.040
        brightness_threshold = 230.0
        
        if rgb_balance > balance_threshold or brightness > brightness_threshold:
            return TabAvailability.AVAILABLE
        else:
            return TabAvailability.UNAVAILABLE