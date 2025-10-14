# LLuMa Musume - Usage Guide

## Prerequisites

1. **Windows 11 (64-bit)** with 150% DPI scaling
2. **Python 3.13+** installed
3. **Uma Musume Pretty Derby** (Steam version) installed
4. **OpenRouter API key** for Gemini 2.5 Flash access

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY='your-key-here'
```

On Windows (PowerShell):
```powershell
$env:OPENROUTER_API_KEY='your-key-here'
```

### 3. Launch the Game

1. Launch Uma Musume Pretty Derby from Steam
2. Set the game to **Windowed mode**
3. The agent will automatically position the window at (0,0) and resize to 1920×1080

## Running the Agent

### Basic Run

```bash
python main.py
```

The agent will:
- Position and focus the game window
- Take screenshots each turn
- Process vision data (button detection, menu analysis)
- Call the LLM for reasoning and decisions
- Execute input actions
- Log all activity to `logs/`
- Save memory files to `memory/`

### Stopping the Agent

Press **Ctrl+C** to stop gracefully after the current turn completes.

**WARNING:** If the game window is closed, the program terminates immediately (unrecoverable).

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
window:
  title: "Umamusume"
  scaling_factor: 1.5
  client_offset: [6, 30]
  placement:
    x: 0
    y: 0
    width: 1920
    height: 1080

capture:
  split:
    left_pin_ratio: 0.07874
    primary_ratio: 0.42126
    menus_ratio: 0.50

agent:
  model: "google/gemini-flash-2.5-latest"
  memory_dir: "memory"
  logs_dir: "logs"
  max_memory_tokens: 32000
  max_context_tokens: 32000
  request_timeout_s: 30
  turn_post_padding_s: 5.0
```

## Output Files

### Screenshots
- `captures/turn_XXXXXX.png` - Full client-area capture
- `captures/turn_XXXXXX-primary.png` - Primary gameplay region
- `captures/turn_XXXXXX-menus.png` - Menu/UI region

### Turn Logs
- `logs/turn_XXXXXX.json` - Detailed turn data (vision, reasoning, actions, results)

### Memory Files
- `memory/*.{yaml,md,txt}` - Agent's persistent memory files (managed by the agent itself)

## Agent Behavior

The agent:
- Learns the game organically (no pre-programmed strategies)
- Maintains its own memory files to track progress
- Takes exactly **1 input action per turn**
- Can use unlimited memory operations per turn
- Waits ~10-30s per turn (LLM + vision processing + padding)

### Available Input Actions

1. **pressButton(name)** - Click a detected button
2. **advanceDialogue()** - Click center of screen to advance text
3. **back()** - Press ESC
4. **confirm()** - Press SPACE
5. **scrollUp()** - Press Z (requires scrollbar detection)
6. **scrollDown()** - Press C (requires scrollbar detection)

### Memory Tools

1. **createMemoryFile(name)** - Create new file
2. **writeMemoryFile(name, content)** - Overwrite file content
3. **deleteMemoryFile(name)** - Delete file

## Troubleshooting

### "Window not found"
- Ensure the game is running and the window title is exactly "Umamusume"
- Check `config.yaml` window.title setting

### "OPENROUTER_API_KEY not set"
- Set the environment variable before running
- Verify the key is valid

### Vision errors
- Ensure screenshots are being saved to `captures/`
- Check that split ratios sum to 1.0 in config
- Verify DPI scaling is set correctly (150% for the reference setup)

### Agent making poor decisions
- The agent learns over time; early mistakes are expected
- Check `memory/` files to see what it has learned
- Review `logs/` to understand its reasoning

## Development

### Adding New Tools

1. Define tool schema in `lluma_agent/tools.py`
2. Add execution logic in `lluma_agent/coordinator.py::_execute_input_action()`
3. Update agent prompt in `lluma_agent/prompts.py` if needed

### Adjusting Vision

- Menu analysis: `lluma_vision/menu_analyzer.py`
- Button detection prompts: `lluma_vision/menu_analyzer.py` (VLM prompts section)

### Modifying Agent Prompt

Edit `lluma_agent/prompts.py::AGENT_SYSTEM_PROMPT`

## Testing Phase Notes

This is an experimental harness. The agent will:
- Explore the game autonomously
- Make mistakes and learn from them
- Build its own knowledge base in memory files
- Gradually improve at navigation and decision-making

**No time pressure** - let it run for extended periods to observe learning behavior.

**Manual supervision recommended** during initial testing phases.
