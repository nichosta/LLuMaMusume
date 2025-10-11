# LLuMa Musume

Experimental harness for Uma Musume gameplay on Windows by an LLM agent.

## Project Status

**Current**: Vision processing pipeline for menu analysis ✅  
**Next**: Agent integration and input handling

### Implemented Components
- ✅ **OS Layer**: Window management and screen capture with configurable splitting
- ✅ **Vision Processing**: Menu state analysis (usability, selected tab, tab availability)
- 🚧 **Agent Layer**: Not yet implemented
- 🚧 **Input System**: Not yet implemented

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test the vision processing
python test_menu_vision.py

# Test OS layer capture
python -m lluma_os.cli --reposition --log-level DEBUG

# Run all tests
PYTHONPATH=/mnt/c/repositories/LLuMaMusume python3 -m unittest discover tests/ -v
```

## Vision Processing

The `lluma_vision` module can analyze Uma Musume menu screenshots to extract:
- Menu usability (active vs blurred)
- Selected tab (green highlighting)  
- Available vs unavailable tabs

```python
from lluma_vision import MenuAnalyzer
from PIL import Image

analyzer = MenuAnalyzer()
menu_image = Image.open("menu_screenshot.png")
result = analyzer.analyze_menu(menu_image)

print(f"Selected: {result.selected_tab}")
print(f"Available tabs: {[tab.name for tab in result.tabs if tab.availability.value == 'available']}")
```

## Configuration

Populate `config.yaml` (see `AGENTS.md` for details). Defaults to 1920x1080 window at top-left with 150% scaling.

Captures are stored as PNG files in `captures/` (`<turn_id>.png`, plus `-primary` and `-menus` crops when enabled).

## Architecture

See [AGENTS.md](AGENTS.md) for detailed design documentation.

## Development

- `debug_temp/`: Development scripts and reference files (not tracked in git)
- `captures/`: Reference screenshots for testing vision processing
