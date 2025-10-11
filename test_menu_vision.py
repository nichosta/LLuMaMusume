"""Production test script for menu vision processing."""

import logging
from pathlib import Path

from PIL import Image

from lluma_vision import MenuAnalyzer

def test_menu_analysis():
    """Test menu analysis on reference screenshots."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    analyzer = MenuAnalyzer()
    captures_dir = Path("captures")
    
    # Test cases with expected results
    test_cases = [
        {
            "file": "career profile, bottom 3 disabled.png",
            "expected_usable": True,
            "expected_selected": "Career Profile",
            "expected_available": ["Jukebox", "Sparks", "Log", "Career Profile"],
        },
        {
            "file": "menu, all active.png", 
            "expected_usable": True,
            "expected_selected": "Menu",
            "expected_available": ["Jukebox", "Menu"],
        },
        {
            "file": "menu, all blurred.png",
            "expected_usable": False,
            "expected_selected": None,
            "expected_available": [],
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n=== Testing: {test_case['file']} ===")
        
        # Load image
        image_path = captures_dir / test_case['file']
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            all_passed = False
            continue
            
        image = Image.open(image_path)
        
        # Crop to menu section (based on config ratios)
        width, height = image.size
        menu_start_x = int(width * 0.5)  # menus section starts at 50%
        menu_image = image.crop((menu_start_x, 0, width, height))
        
        # Analyze menu
        result = analyzer.analyze_menu(menu_image)
        
        # Validation
        usable_ok = result.is_usable == test_case["expected_usable"]
        print(f"  Usability: {'✓' if usable_ok else '❌'} (expected {test_case['expected_usable']}, got {result.is_usable})")
        
        if result.is_usable:
            selected_ok = result.selected_tab == test_case["expected_selected"]
            print(f"  Selected: {'✓' if selected_ok else '❌'} (expected {test_case['expected_selected']}, got {result.selected_tab})")
            
            available_tabs = [tab.name for tab in result.tabs if tab.availability.value == "available"]
            available_ok = set(available_tabs) == set(test_case["expected_available"])
            print(f"  Available: {'✓' if available_ok else '❌'} (expected {test_case['expected_available']}, got {available_tabs})")
            
            if not (selected_ok and available_ok):
                all_passed = False
        else:
            print("  Skipping detailed validation for inactive menu")
        
        if not usable_ok:
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All tests PASSED!")
    else:
        print("❌ Some tests FAILED!")
    
    return all_passed


if __name__ == "__main__":
    test_menu_analysis()