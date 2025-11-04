# AGENTS.md

## NOTE TO ALL CONTRIBUTORS (AI OR HUMAN): EDIT AGENTS.md AFTER ANY STRUCTURAL CHANGE

# LLuMa Musume

Experimental harness for Uma Musume gameplay on Windows by an LLM agent.

# Environment

- OS: Windows 11 24H2 (64-bit)
- Python: 3.13.8 (x64), uv for package management.
- Display: 2560×1600 (16:10 aspect ratio, single monitor recommended; display scaling 150%)
- Game: Uma Musume Pretty Derby (Steam)
- Game mode: Windowed (position and size are dynamically queried; manual adjustment is supported)
- Client area: ~2539×1384 physical pixels (~1693×923 logical pixels, measured in practice)
- Locale: English (NA)
- Admin: Not required

# Window & Coordinates

- Window title: Exact match "Umamusume" (Steam EN/NA). Fail fast if not found.
- Focus/visibility: Ensure window is restored, unminimized, and foregrounded before inputs. Abort inputs if focus fails.
- Placement: **Dynamic positioning** - Window geometry (position, size, client area) is queried fresh every turn via `refresh_geometry()`. The user may manually move or resize the window at any time without breaking the agent. Optional: Configure `window.placement` in `config.yaml` to auto-position the window at startup only.
- Client area origin: Use the client-area top-left as `(0,0)` for all in-game coordinates. Normalize all tool inputs and Vision outputs to this origin.
- DPI scaling (150%): After calling `SetProcessDPIAware()` (see OS → DPI Awareness section), all window APIs (`GetWindowRect`, `GetClientRect`, `ClientToScreen`) return physical screen pixels. Vision outputs client-relative logical pixels, which must be multiplied by 1.5 and added to the physical client origin. See OS section for detailed conversion formulas.
- Bounds source: Client-area bounds and window position are queried every turn using `ClientToScreen()` and `GetClientRect()` to ensure accuracy regardless of window position or DWM adjustments.
- Stability: Window dimensions/position are **NOT** assumed stable; they are re-measured every turn. This allows the window to be manually moved or resized without restarting the agent.
- Multi-monitor: Single monitor recommended. The agent works regardless of window position.
- Errors: If window is occluded/not found, log and skip the turn (no inputs) rather than sending blind clicks.

# OS

Handles the interactions between the rest of the code and the Uma Musume window. This comprises keyboard/mouse inputs in the actual window and window captures to be processed by Vision.

## DPI Awareness

**CRITICAL**: The program must call `user32.SetProcessDPIAware()` from ctypes at startup before any window or screen operations:

```python
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()
```

Without this, at 150% DPI scaling:
- `GetWindowRect()` returns incorrect virtualized/logical coordinates
- PyAutoGUI mouse clicks miss their targets by significant margins
- MSS screenshots capture wrong regions or incorrect sizes

After setting DPI awareness:
- pygetwindow returns window positions in **physical screen pixels**
- `GetClientRect()` returns client dimensions in **physical screen pixels** (must divide by scaling_factor to get logical size)
- PyAutoGUI expects coordinates in **physical screen pixels**
- MSS operates in **physical screen pixels**
- `config.yaml` offsets are in **logical pixels** and must be scaled

**Coordinate conversion** (Vision logical → PyAutoGUI physical):
```python
# Vision output: (x_logical, y_logical) relative to client area
# Client origin is obtained via ClientToScreen() each turn (physical pixels)
# Window geometry is refreshed every turn to handle manual moves/resizes

# From WindowGeometry obtained via refresh_geometry():
client_origin_x_phys = geometry.client_area.screen_origin[0]
client_origin_y_phys = geometry.client_area.screen_origin[1]
scaling_factor = geometry.client_area.scaling_factor  # 1.5 for 150% DPI

screen_x_phys = client_origin_x_phys + (x_logical * scaling_factor)
screen_y_phys = client_origin_y_phys + (y_logical * scaling_factor)
```

Note: `client_offset` in config.yaml is optional and rarely needed. By default, `ClientToScreen(hwnd, {0,0})` is used to determine the client area origin dynamically.

## Input

Uses PyAutoGUI for inputs and pygetwindow to ensure focus on the Uma Musume window.

## Output

Uses python MSS to capture the client-area of the Uma Musume window (via screen-region crop based on window bounds and DPI transform). One screenshot is taken per turn and provided to Vision.

Capture specifics
- Cadence: Single capture per turn. Turns are expected to take ~10–30s; there is currently no post-action/verification capture (the config flag exists but is unused).
- Region: Compute client-area rect (logical px), transform to screen px using the `scaling_factor` (150%), and pass to MSS as the crop box.
- Validation: Ensure the capture dimensions match the client area, then compute mean/standard deviation of luminance. If either falls below `CaptureValidationConfig` thresholds we abort the turn; there is no automatic retry.
- Format: Save raw PNG (lossless) for Vision. No JPEG preview path exists yet.
- Splitting: Produce `Primary`, `Menus`, and `Tabs` crops from the raw capture. Split uses four ratios that sum to 1.0: `left_pin_ratio` (discarded left strip), `primary_ratio` (gameplay area), `menus_ratio` (menu content), and `tabs_ratio` (right-side tab column). All ratios are percentages of total client width.
- Filenames: `captures/<turn_id>.png`, plus `captures/<turn_id>-primary.png`, `captures/<turn_id>-menus.png`, and `captures/<turn_id>-tabs.png` when split is enabled.
- Retention: After each capture we enforce the retention limit (default 200 raw captures) by deleting the oldest turn set.
- Occlusion: We rely on focus + top-level z-order; per-window capture is not used. If occluded content is suspected (e.g., unexpected uniform regions), abort the turn.
- Primary crop: feed to the VLM to enumerate main-area buttons. Scrollbar state is derived from local heuristics that examine the rightmost ~220 px band for the mauve/grey track and thumb (see `lluma_vision/menu_analyzer.py::detect_primary_scrollbar()` for implementation details).

Menu content nuances
- `lluma_vision.MenuAnalyzer` inspects the menus/tabs crops first. If the pane is blurred or otherwise unusable it skips the VLM step and reports `menu_state.is_usable=False` to the agent.
- When the pane is usable we always pass both the screenshot path and the structured button list emitted by the VLM; there are no tab-specific overrides.

# Vision

Handles processing the images into broadly usable structured data. Image processing proceeds in multiple steps; splicing the image (the gameplay screen is easily split into two parts, Primary and Menus), processing out inactive buttons and content, and finally VLM Image Understanding via OpenRouter to return named bounding boxes for interactable buttons (to be used to inform the model of available tool calls, and to process those tool calls into mouse and keyboard actions).

The raw screenshot (most likely only the Primary side) is also handed to the agent alongsize the processed data, under the assumption that its vision is enough for OCR (in visual novel dialogue and pop-ups).

## Vision I/O

Inputs
- Full client capture plus derived crops (`Primary`, `Menus`, `Tabs`) generated by `CaptureManager._split_capture()`.
- The menus/tabs crops are fed into `MenuAnalyzer` for heuristics (blur detection, tab availability) before any VLM call is issued.
- When the menu pane is usable we ship the trimmed menus crop to the VLM; the tabs crop is only used locally for heuristics.
- The primary crop is always provided to the VLM and also scanned locally for scrollbars.

Outputs (what the agent receives)
- `buttons`: Combined list of menu and primary detections. Each record contains the stripped `name`, original `full_label`, logical-pixel `bounds` `(x, y, w, h)`, parsed `metadata`, and the source `region` (`menus` or `primary`).
- `scrollbar`: Optional dict with `track_bounds`, `thumb_bounds`, `can_scroll_up`, `can_scroll_down`, and `thumb_ratio`. This comes from a heuristic pass over the rightmost ~220 logical pixels of the primary crop (`detect_primary_scrollbar`), not from the VLM.
- `menu_state`: Result from `MenuAnalyzer.analyze_menu()`. It reports whether the pane is usable, which tab is selected, and a list of tab objects (`name`, `is_selected`, `availability`). `available_tabs` is emitted as a convenience list for consumers. The analyzer uses Laplacian sharpness to spot blur overlays and simple brightness/color metrics to tell available vs. greyed-out tabs.

There are currently **no** `hotspots`, `overlays`, or global `meta` records in the coordinator output—those features are still on the roadmap.

Label encoding (single string parsed by the harness)
- Format: `name` followed by optional metadata tags like `|section=menus` or `|hint=New`.
- Required: button name; metadata tags are optional and only used when they add meaningful context.
- Escaping: replace `|` with `\|` and `=` with `\=` inside values.

VLM detection mode
- Use object detection that returns `box_2d: [ymin, xmin, ymax, xmax]` and `label: string` with coordinates normalized to 0–1000 relative to the input image.
- THE OUTPUTS OF THE VLM ARE IN Y,X ORDER. Convert them immediately before treating them as `(x, y, w, h)`.
- No native confidence provided; we rely solely on the returned boxes and labels.

Coordinate conversion
- Convert normalized coords into client-relative logical pixels before input handling.
  1. `crop_px = (norm / 1000) * crop_size_px` (convert normalized 0-1000 to crop-relative pixels)
  2. `client_px = crop_px + crop_origin_within_client` (add the crop’s logical offset)
  3. `client_logical_px = client_px` (captures are already in logical resolution)
- Account for the discarded left pin strip by adding `split.left_pin_ratio * client_area.width_logical` to the menus/primary X origin as needed.
- The OS layer later converts logical pixels to physical ones when sending PyAutoGUI clicks (see DPI Awareness section).

Empty/low-signal frames
- When neither crop yields any buttons the coordinator records the condition and immediately fires the failsafe `advanceDialogue()` once. There is no additional classification yet.

### VLM Prompts

The Vision pipeline uses `google/gemini-2.5-flash-lite-preview-09-2025` on OpenRouter for both menu and primary detections. Two prompts are issued:

**Menu Region Detection** (used by `get_clickable_buttons()`):
- System: "You analyse Uma Musume UI screenshots (already cropped to exclude fixed-position menu tabs) to find interactive buttons. Respond with JSON listing each distinct clickable button once. Each record must contain a `label` string and `box_2d` array representing [ymin, xmin, ymax, xmax] with values in the range 0-1000 (normalised coordinates). The label should begin with the button name; optionally append metadata like `|section=menus` or `|hint=...`, but avoid confidence, state, or type tags. IMPORTANT: If two buttons would have the same name, add a distinguishing identifier to the name itself (not just in metadata tags) so each button name is unique."
- User: "Return a JSON object with a single property `buttons` that is an array of button records. Every record must have `label` (string) and `box_2d` (array of four floats). Exclude decorative or inactive elements. If no buttons are present, respond with `{\"buttons\": []}`."

**Primary Region Detection** (used by `get_primary_elements()`):
- System: "You analyse Uma Musume primary gameplay captures to find clickable UI elements. Return structured JSON so an agent can decide interactions. List each distinct clickable button once with its label and bounds. IMPORTANT: If two buttons would have the same name, add a distinguishing identifier to the name itself (not just in metadata tags) so each button name is unique."
- User: "Respond with a JSON object that includes a `buttons` array. Each button entry must have `label` and `box_2d` fields. Avoid confidence/state/type suffixes unless they add useful hints like `|hint=New`. If no buttons exist, use an empty array."

Notes:
- Both prompts request JSON mode (`response_format: {type: "json_object"}`)
- If menu response is empty or malformed, retry up to two times, logging both the prompt and response of the failed calls. If three failures are recorded, the menu should be assumed to be empty.
- Primary images are sent untrimmed.
- Implementation: `lluma_vision/menu_analyzer.py`. Both calls reuse the same `OPENROUTER_API_KEY` that the agent uses.

### Image Processing Stack

Current runtime deps
- `pillow`: save PNG captures, crop primary/menus/tabs, re-open files for base64 encoding.
- `numpy`: Laplacian sharpness scores, tab-availability heuristics, and scrollbar gradient math.
- `requests`: direct OpenRouter calls (both for VLM and for the agent).

Processing today is intentionally minimal:
- No denoise, CLAHE, or gamma tweaks are applied; we rely on the VLM’s robustness.
- Menu crops are trimmed on the left via `LEFT_TRIM_RATIO` before encoding; the actual trim ratio is propagated so the coordinator can adjust button bounds. If the trimmed query returns no detections but was parsed successfully, we retry without trimming. If the response JSON is invalid we retry once with the same trim before giving up.
- Scrollbar detection inspects the rightmost band of the primary crop in raw RGB space.

Optional packages (`opencv-python-headless`, `scikit-image`, `ImageHash`, `pytesseract`, etc.) are kept in reserve for future experiments but are not imported by default.

# Agent

The reasoning agent lives in `lluma_agent`. It uses the Anthropic API with extended thinking enabled by default. The default model is `claude-haiku-4-5` (configurable via `agent.model`). Vision continues to use Gemini 2.5 Flash-Lite via OpenRouter; the agent model can be changed independently in `config.yaml`.

## Extended Thinking

The agent uses Claude's extended thinking capability to improve reasoning quality for complex gameplay decisions. Extended thinking allows the model to show its step-by-step reasoning process before producing final responses. Key features:

- **Thinking budget**: Configurable via `agent.thinking_budget_tokens` (default: 16000, minimum: 1024)
- **Transparency**: Thinking blocks are logged separately in turn logs for debugging
- **Toggle**: Can be disabled via `agent.thinking_enabled: false` in config.yaml
- **Supported models**: Haiku 4.5, Sonnet 4.5, Sonnet 4, Sonnet 3.7, Opus 4.1, Opus 4

The thinking process is visible in debug logs and saved in per-turn JSON logs under the `thinking` field.

## Tools

Defines the abstract tools used by the model to interact with the game. Coordinate-level actions are not exposed to the model; all inputs are expressed in semantic terms tied to Vision output or high-level intents.

### Input API

- `pressButton(name)`: Clicks the center of the interactable whose bounding box is labeled `name` by Vision. Requires that the button is visible, enabled, and within the client area.
- `advanceDialogue()`: Single click at the dialogue-advance hotspot (center region). Use for VN-style text advance and simple confirms.
- `back()`: Sends in-game Back via `ESC`.
- `confirm()`: Sends in-game Confirm via `SPACE`.
- `scrollUp()`: Sends `Z` to scroll up lists/menus.
- `scrollDown()`: Sends `C` to scroll down lists/menus.

Notes
- No raw coordinate tool (e.g., move/click) is exposed to the model. The harness may use internal coordinate helpers for window placement and diagnostics only.
- Scrollbars and list widgets may be inconsistent; prefer button navigation when available. Treat `scrollUp/scrollDown` as best-effort and avoid repeated spamming.
- `pressButton` clicks the geometric center of the detected bounding box for maximum reliability.

### Input Behavior & Safety

- Focus guard: Before any input, ensure the "Umamusume" window is visible and focused. If the window is closed or minimized, immediately terminate the program with an error.
- Window closure: If the game window is closed for any reason, the OS handler immediately stops the program.
- Timing: Apply small randomized jitter (currently 20–50 ms) to press/release timing to avoid mechanical patterns.
- Click duration: Mouse clicks and key presses are held for 100ms to ensure registration.
- Rate limits: One action per turn (strictly enforced). This may be increased after testing.
- Emergency stop: Use Ctrl-C in the terminal running the main script.

### Failsafe Policy

- If Vision reports no interactable buttons for the current turn, the coordinator attempts a single `advanceDialogue()` as a gentle nudge and records the condition.
- If a requested `pressButton(name)` is not visible/enabled, do not retry blindly. Report the missing/disabled state to the agent.
- Loop detection: If the agent detects it is stuck in a loop (same screen state for multiple turns), it should use `advanceDialogue()` or `back()` as failsafes to break the loop.

## Memory

The model has access to its own scratchpad; it is required to maintain most information about the game itself, including stats, inventory, and actual gameplay rules. The scratchpad is injected into the end of context right before the information for the current turn. The scratchpad is segmented into individual files to simplify the process of modification; the LLM is allowed to individually modify, create, and delete files itself, up to a cumulative 32k token allocation for the entire scratchpad. This is accomplished via the createMemoryFile(name), deleteMemoryFile(name), and writeMemoryFile(name, content) tool calls.

Memory file naming:
- All files reside in a single flat directory (no subdirectories or path separators)
- No naming restrictions beyond filesystem limitations
- Suggested convention: `player.yaml`, `run.yaml`, `ui_knowledge.md`, `todo.md`, etc.

## Turns

Turn structure and timing
- Each turn performs: ensure window (reposition only on the first turn) → capture → vision processing → agent reasoning → execute at most one input → persist the turn log → sleep for `agent.turn_post_padding_s` seconds (default 5s).
- Turn latency: Highly variable by model/load; budget ~10–30s. Add a fixed 5s padding between turns to smooth variance.
- Timeouts: The Anthropic API call uses the SDK's default timeout. Failures raise `AgentError` for the turn and no input is issued.
- Vision prompts (menus + primary) run sequentially inside `MenuAnalyzer`; there is no parallelism at present.

Context and summarization
- Historical turns are stored as compact summaries (~150 tokens vs ~8,500 for full context), dramatically reducing token accumulation.
- Automatic summarization is queued for the next turn whenever the previous request's input token usage exceeds the smaller of `agent.summarization_threshold_tokens` and 90% of the configured context window.
- When summarization runs, the agent summarizes its entire history (game progress, UI knowledge, current state, discoveries).
- The message history is then replaced with a single synthetic summary message.
- This prevents context overflow and allows indefinite session length.
- Turn transcripts include the agent's reasoning, thinking blocks, and tool calls.

Memory scratchpad
- The agent manages its scratchpad via `createMemoryFile(name)`, `deleteMemoryFile(name)`, `writeMemoryFile(name, content)`.
- Cumulative scratchpad budget: also 32k tokens across all files. Prefer compact structured text (e.g., YAML/JSON-lite) over verbose prose.
- Recommended files: `player.yaml`, `run.yaml`, `ui_knowledge.md`, `todo.md`. Delete or truncate obsolete files proactively.
- Note: The turn context budget and the memory scratchpad budget are both 32k tokens, but they are separate allocations (turn history vs. persistent memory).

Safety and limits
- One action per turn (strictly enforced). This may be increased after testing.
- If Vision returns no actionable buttons for the turn, attempt one `advanceDialogue()` then end the turn with a diagnostic (no blind inputs).

# Performance & Context Optimization

See **CONTEXT.md** for detailed analysis of prompt structure, token usage patterns, and optimization strategies. This includes:
- Token cost breakdown per turn
- Button metadata optimization strategies
- Historical turn trimming for reduced context accumulation
- Prompt caching implementation
- Cost projections and savings estimates

# Configuration

`config.yaml` (repo root) contains three sections: `window`, `capture`, and `agent`. All coordinates in this file are logical pixels.

Window
- `title`: Window title to match (default `"Umamusume"`).
- `placement`: Optional. If specified with `{ x, y, width, height }`, the window will be positioned at startup only. If omitted (default), the window position is never changed and is queried dynamically each turn.
- `scaling_factor`: 1.5 for the current 150 % DPI setup.
- `client_offset`: Optional. Logical pixel offset from outer top-left to client origin. Rarely needed; by default `ClientToScreen(hwnd, {0,0})` is used to determine the client area origin dynamically.

Capture
- `output_dir`: defaults to `captures/`.
- `post_action`: maps to `CaptureConfig.post_action_capture` but is not used in the current coordinator.
- `scaling_factor`: inherits from `window.scaling_factor` unless overridden.
- `split`: ratios must sum to 1.0; the checked-in config uses `left_pin=0.07874`, `primary=0.42126`, `menus=0.41497`, `tabs=0.08503`.
- `validation`: `min_mean_luminance=5.0`, `min_luminance_stddev=1.5`.
- `retention.max_captures`: 200 raw captures (associated primary/menus/tabs files follow the raw capture’s lifecycle).

Agent
- `model`: defaults to `claude-haiku-4-5`. Use Anthropic model identifiers (e.g., `claude-sonnet-4-5`, `claude-opus-4-1`).
- `memory_dir`, `logs_dir`: directories for scratchpad files and per-turn logs (`memory/`, `logs/`).
- `max_memory_tokens`, `max_context_tokens`: both default to 32 000.
- `thinking_enabled`: enable extended thinking (default `true`).
- `thinking_budget_tokens`: max tokens for internal reasoning (default 16 000, minimum 1 024).
- `max_tokens`: max output tokens (default 4 096, must exceed thinking_budget_tokens).
- `summarization_threshold_tokens`: queue summarization for the next turn when the last prompt's input tokens exceed the smaller of this value and 90% of `max_context_tokens` (default 64 000, set to 0 to disable).
- `request_timeout_s`: reserved for future use; the Anthropic SDK manages timeouts internally.
- `turn_post_padding_s`: sleep between turns (default 5 s).
- `allow_skip_cinematics`: reserved for future cinematic handling; no runtime logic reads it yet.

Input timing (20–50 ms jitter, 100 ms click/key duration) is hardcoded in `InputConfig` today; there is no dedicated config block.

Environment overrides are supported via `LLUMA_*` variables, but `config.yaml` is the expected source of truth.

# Initial State & Stopping

## Initial Game State
The agent should be state-agnostic and capable of orienting itself regardless of the initial game state. It should assess the current screen and available actions on its first turn and proceed accordingly.

## Stop Conditions
The agent runs indefinitely until manual intervention (Ctrl-C in the terminal). No automatic stop conditions are implemented. For production runs, the game account should be reset to a fresh state; for testing, a mostly fresh account is acceptable.

## Testing & Debugging
- No mock/replay testing infrastructure is currently available; all testing is performed against the live game.
- Debug logging can be configured via verbosity levels (error, warn, info, debug, trace).
- Captures are retained (last 200 turns by default) for post-mortem analysis.
- For quick window experiments (e.g., reproducing DPI or cursor drift issues) use `python -m lluma_os.window_cli <command>`; it can `status`, `focus`, `place`, `move`, `resize`, or `set` the Uma Musume window using logical-pixel coordinates. The `place` command requires `window.placement` to be configured in `config.yaml`.
- Cinematic heuristics live in `lluma_agent/cinematics.py`. The detector watches the frame-diff score alongside three quick visual checks: (1) whether the bright “pin” control in the top-left corner is still present, (2) how lively the bottom-right primary region is (persistent Skip chip), and (3) variance/brightness in the right-hand tabs strip when a fullscreen Skip overlay appears. The coordinator defers handing turns to the agent while these signals indicate active motion and only resumes after a couple of consecutive low-motion frames (or when the pin/skip overlays clear) so we never waste actions mid-cutscene.
- A CLI harness validates those heuristics offline: `python -m lluma_agent.cinematic_detector_cli <captures_or_dirs> [--metadata hints.json] [--assume-menu-disabled] [--json]`. Feed it saved PNGs (e.g. the samples in `captures/`) to see the derived diff, pin detection, skip hints, and the `FULLSCREEN`/`PRIMARY` classifications without hitting the live client.

## Error Recovery
- **Window closure**: If the game window is closed, the program immediately stops.
- **Checkpoint/resume**: Not currently implemented (TODO).
- **Loop detection**: The agent should detect repeated screen states and use `advanceDialogue()` or `back()` to break loops.

# Logging

Structured logs aid debugging and reproducibility without exposing sensitive data.

- Per-turn JSON: `logs/<turn_id>.json` (e.g., `logs/turn_000123.json`)
  - Includes timestamps, capture filenames + validation stats, menu state, detected buttons, agent reasoning, memory tool metadata, chosen input, and execution status/errors.
  - Excludes inline image data; files are referenced by relative path.

- Session-wide `.jsonl` logging is not implemented yet (future work).

- Screenshots
  - `captures/<turn_id>.png`, `captures/<turn_id>-primary.png`, `captures/<turn_id>-menus.png` (when enabled)
  - Retention respects `capture.retention`.

- Verbosity
  - Levels: error, warn, info (default), debug, trace.
  - Debug/trace may include intermediate Vision prompts (text only) and crop rectangles.

# Setup & Dependencies

Install Python 3.13.8 (x64), then install dependencies via `uv pip install -r requirements.txt`.

- Confirmed runtime deps
  - OS I/O: `pyautogui`, `pygetwindow`, `mss`
  - Imaging & array math: `pillow`, `numpy`
  - Networking/API clients: `requests` (OpenRouter uploads for both agent and vision)
  - Config/tests: `pyyaml`, `pytest`

- Provisional deps (enable if/when needed)
  - JSON performance: `orjson`
  - Retry utilities: `tenacity`
  - Image processing extras: `opencv-python-headless`, `scikit-image`, `ImageHash`, `pytesseract`

Environment
- Set `ANTHROPIC_API_KEY` in the environment for Agent calls and `OPENROUTER_API_KEY` for Vision calls.
- Ensure Steam window title is exactly `Umamusume`.
- First-run checklist: Ensure the game window is open and visible (position/size don't matter), confirm capture works, adjust split ratios as needed. Optionally configure `window.placement` in config.yaml for auto-positioning at startup.

# TODO / Future Work

- **Runtime DPI measurement**: Currently hardcoded at 1.5 (150% scaling); consider measuring at runtime for robustness across different display configurations.
- **Checkpoint/resume**: Implement save/restore capability to resume agent sessions after interruption.
- **Mock/replay testing**: Build infrastructure to test against recorded captures without running the live game.
- **VLM prompt refinement**: The current prompts may need iteration based on actual detection performance; monitor false positives/negatives and adjust accordingly.
- **Post-action capture & session logs**: Wire up `CaptureConfig.post_action_capture` and add a streamed session log (`logs/session.jsonl`).
- **Config plumbing**: Respect `agent.request_timeout_s` and `agent.allow_skip_cinematics`, and surface explicit knobs for input timing.
- **Cinematic handling**: Detect race/gacha cinematics and honour the skip policy safely.
- **Vision enrichments**: Extend the pipeline with hotspots/overlays/meta objects when reliable detectors are ready.
