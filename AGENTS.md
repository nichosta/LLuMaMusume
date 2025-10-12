# AGENTS.md

# LLuMa Musume

Experimental harness for Uma Musume gameplay on Windows by an LLM agent.

# Environment

- OS: Windows 11 24H2 (64-bit)
- Python: 3.13.8 (x64), uv for package management.
- Display: 2560×1600 (16:10 aspect ratio, single monitor recommended; display scaling 150%)
- Game: Uma Musume Pretty Derby (Steam)
- Game mode: Windowed, positioned at (0,0) with outer size 1920×1080 (approximately 16:9)
- Client area: 2538×1383 pixels (measured in practice)
- Locale: English (NA)
- GPU: No dedicated GPU required
- Admin: Not required

# Window & Coordinates

- Window title: Exact match "Umamusume" (Steam EN/NA). Fail fast if not found.
- Focus/visibility: Ensure window is restored, unminimized, and foregrounded before inputs. Abort inputs if focus fails.
- Placement: On startup, position window at `x=0, y=0` and size to `1920×1080` (outer size). Re-fetch bounds after resizing.
- Client area origin: Use the client-area top-left as `(0,0)` for all in-game coordinates. Normalize all tool inputs and Vision outputs to this origin.
- DPI scaling (150%): Treat window/game coordinates as logical pixels; compute transforms:
  - `screen_px = window_logical_px * 1.5 + client_origin_on_screen`
  - For captures, convert crop rectangles to screen pixels using the same factor.
  - Keep a single source of truth for `scaling_factor` (initially 1.5); measure at runtime if possible.
- Bounds source: Prefer client-area bounds. If only outer bounds are available, account for title bar and borders or measure client offsets at runtime after placement.
- Stability: Window dimensions are assumed stable during a run; refresh bounds after any explicit resize/reposition only.
- Multi-monitor: Single monitor recommended. If multiple, require window on primary display at `(0,0)`.
- Errors: If window is occluded/not found, log and skip the turn (no inputs) rather than sending blind clicks.

# OS

Handles the interactions between the rest of the code and the Uma Musume window. This comprises keyboard/mouse inputs in the actual window and window captures to be processed by Vision.

## Input

Uses PyAutoGUI for inputs and pygetwindow to ensure focus on the Uma Musume window.

## Output

Uses python MSS to capture the client-area of the Uma Musume window (via screen-region crop based on window bounds and DPI transform). One screenshot is taken per turn and provided to Vision.

Capture specifics
- Cadence: Single capture per turn. Turns are expected to take ~10–30s; no intra-turn recapture by default. Optional post-action capture can be enabled for debugging.
- Region: Compute client-area rect (logical px), transform to screen px using the `scaling_factor` (150%), and pass to MSS as the crop box.
- Validation: Verify capture size matches expected client-area size; sample a small pixel grid to ensure the image is non-black/non-uniform. If invalid, retry once after refocusing; otherwise skip the turn with a diagnostic.
- Format: Save raw PNG (lossless) for Vision. Optionally save a JPEG preview for logs.
- Splitting: Produce `Primary` and `Menus` crops from the raw capture. Split uses three ratios that sum to 1.0: `left_pin_ratio` (discarded left strip), `primary_ratio` (gameplay area), and `menus_ratio` (UI area). All ratios are percentages of total client width.
- Filenames: `captures/<turn_id>.png`, plus `captures/<turn_id>-primary.png` and `captures/<turn_id>-menus.png` when split is enabled.
- Retention: Keep the last N turns (configurable, e.g., 200) and clean older files on startup.
- Occlusion: We rely on focus + top-level z-order; per-window capture is not used. If occluded content is suspected (e.g., unexpected uniform regions), abort the turn.
- Primary crop: feed to the VLM to enumerate main-area buttons. Scrollbar state is derived from local heuristics that examine the rightmost ~220 px band for the mauve/grey track and thumb (see `lluma_vision/menu_analyzer.py::detect_primary_scrollbar()` for implementation details).

Menu content nuances
- Sparks, Log, and Career Profile tabs rarely expose dedicated buttons; when these are the selected tab, pass the raw menu crop to the agent instead of the (likely empty) button list so it can rely on direct OCR.
- For all other tabs, provide the structured button list emitted by the VLM in addition to the image.

# Vision

Handles processing the images into broadly usable structured data. Image processing proceeds in multiple steps; splicing the image (the gameplay screen is easily split into two parts, Primary and Menus), processing out inactive buttons and content, and finally VLM Image Understanding via OpenRouter to return named bounding boxes for interactable buttons (to be used to inform the model of available tool calls, and to process those tool calls into mouse and keyboard actions).

The raw screenshot (most likely only the Primary side) is also handed to the agent alongsize the processed data, under the assumption that its vision is enough for OCR (in visual novel dialogue and pop-ups).

## Vision I/O

- Inputs
  - Images: full client-area PNG plus derived crops `Primary` and `Menus`.
  - Optional masking: exclude the left-side pin strip region from detection (configured via `split.left_pin_ratio`).
  - Normalization: apply light denoise and brightness/contrast normalization to stabilize detection; do not alter geometry.

- VLM detection mode
  - Use object detection that returns `box_2d: [ymin, xmin, ymax, xmax]` and `label: string` with coordinates normalized to 0–1000 relative to the input image.
  - No native confidence provided; when feasible, request that the model appends an approximate confidence to the `label` (see label encoding). Treat it as advisory only.

- Label encoding (single string parsed by the harness)
  - Format: `name` followed by optional metadata tags like `|section=menus` or `|hint=New`.
  - Required: button name; metadata tags are optional and only used when they add meaningful context.
  - Escaping: replace `|` with `\|` and `=` with `\=` inside values.

- Output records (constructed by the harness from VLM outputs)
  - `buttons`: list of { `name`, `bounds` (client logical px: x,y,w,h), optional `section`/`hint` }.
  - `hotspots`: named regions like `dialogue_center`, `confirm_ok`, etc., same bounds convention.
  - `overlays`: detected UI states such as `tutorial_overlay`, `modal_dialog`, `loading_mask`.
  - `meta`: { `turn_id`, `client_size`, `scaling_factor`, `splits`: { `primary`, `menus` }, `left_pin_ratio` }.

- Coordinate conversion
  - The VLM provides normalized coords per input image (full or crop). Convert to crop pixels, then to client logical px.
  - Steps:
    1. `crop_px = (norm / 1000) * crop_size_px` (convert normalized 0-1000 to crop pixels)
    2. `screen_px = crop_px + crop_origin_screen` (add crop origin in screen coordinates)
    3. `client_logical_px = (screen_px - client_origin_on_screen) / scaling_factor` (convert to client-relative logical pixels)
  - If a left pin strip is cropped (based on `left_pin_ratio`), add its width to the X-origin before back-projection.
  - Note: `crop_origin_screen` is the absolute screen position of the crop's top-left; `client_origin_on_screen` is the absolute screen position of the client area's top-left.

- Inactive/disabled UI handling
  - Heuristics: look for blur, darkening, greyed text, lock icons, and unlock-condition text; encode as `state=inactive|locked` with a brief `hint` where possible.
  - Avoid hardcoding thresholds; rely on the VLM and qualitative cues in the prompt. Expect manual image testing to refine instructions.

- Empty/low-signal frames
  - If zero `buttons` are detected on both `Primary` and `Menus`, classify likely states:
    - dialogue-only screen (use `hotspots` to include `dialogue_center`)
    - modal/tutorial overlay (`overlays` should include `modal_dialog`)
    - transitional/loading state
  - The harness may attempt a single `advanceDialogue()` per the failsafe policy; otherwise it ends the turn and reports the condition.

### Cinematics (races and gacha)

- Goal: Do not interact during long-running cinematics; wait and report results when they conclude.
- Detection cues (non-exhaustive; refined via prompt iteration):
  - Races: lap/progress UI, position/order ticker, minimap/pace bars, absence of standard menu buttons; presence of a `Skip` button.
  - Gacha: card grid/reveal UI, star/rarity animations, `Skip`/`Results` buttons.
- Behavior:
  - Default: No inputs during cinematic. Poll each turn until a `Results`/`Next`/`Close` button appears, then proceed.
  - Optional: If config `allow_skip_cinematics=true`, click `Skip` once when detected; then proceed to results.
  - Reporting: Extract outcome from the first post-cinematic screen (e.g., race result placement/time, gacha pull list) via Vision labels and/or OCR from the raw capture.

### VLM Prompts

The Vision pipeline uses Google Gemini 2.5 Flash (as of October 2025, the latest in the Gemini series) via OpenRouter. Two prompts are used for button detection:

**Menu Region Detection** (used by `get_clickable_buttons()`):
- System: "You analyse Uma Musume UI screenshots (already cropped to exclude fixed-position menu tabs) to find interactive buttons. Respond with JSON listing each distinct clickable button once. Each record must contain a `label` string and `box_2d` array representing [ymin, xmin, ymax, xmax] with values in the range 0-1000 (normalised coordinates). The label should begin with the button name; optionally append metadata like `|section=menus` or `|hint=...`, but avoid confidence, state, or type tags."
- User: "Return a JSON object with a single property `buttons` that is an array of button records. Every record must have `label` (string) and `box_2d` (array of four floats). Exclude decorative or inactive elements. If no buttons are present, respond with `{\"buttons\": []}`."

**Primary Region Detection** (used by `get_primary_elements()`):
- System: "You analyse Uma Musume primary gameplay captures to find clickable UI elements. Return structured JSON so an agent can decide interactions. List each distinct clickable button once with its label and bounds."
- User: "Respond with a JSON object that includes a `buttons` array. Each button entry must have `label` and `box_2d` fields. Avoid confidence/state/type suffixes unless they add useful hints like `|hint=New`. If no buttons exist, use an empty array."

Notes:
- Both prompts request JSON mode (`response_format: {type: "json_object"}`)
- The left 15% of menu images is trimmed before VLM processing to remove fixed tab icons
- Primary images are sent untrimmed
- Implementation: `lluma_vision/menu_analyzer.py`

### Image Processing Stack

Principle: Rely primarily on the VLM; keep preprocessing minimal, photometric-only, and geometry-preserving. Use simple masks and metrics; move heavier processing out unless proven necessary.

- Confirmed deps
  - `pillow`: image I/O, cropping, PNG encode, simple masks.
  - `numpy`: array conversion (BGRA→RGB), basic stats/metrics (mean/variance, histograms).

- Provisional deps (enable only if needed after testing)
  - `opencv-python-headless`: optional denoise/CLAHE/morphology utilities.
  - `scikit-image`: optional unsharp mask, SSIM for change detection.
  - `ImageHash`: optional perceptual hash for quick duplicate/unchanged frame checks.
  - `pytesseract` (system Tesseract required): optional OCR for numeric/stat extraction when VLM OCR is unreliable.

- Allowed transforms (default)
  - Color conversion: MSS BGRA → RGB.
  - Light normalization: per-channel mean/variance normalize or mild gamma; no resizing.
  - Simple masks: exclude static UI strips (e.g., left pin) from detection.
  - Basic metrics: luminance mean/variance, histogram deltas to sanity-check captures and detect no-change frames.

# Agent

The LLM itself and its tools and abilities. The agent uses Google Gemini 2.5 Flash (as of October 2025, the latest in the Gemini series) via OpenRouter.

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

- Focus guard: Before any input, ensure the "Umamusume" window is visible and focused. If not, skip the action and surface an error to the agent.
- Window closure: If the game window is closed for any reason, the OS handler immediately stops the program.
- Debounce: Coalesce duplicate tool calls within 300 ms (configurable). Do not double-click unless explicitly required by the button definition.
- Timing: Apply small randomized jitter to press/release and inter-action delays to avoid mechanical timing (configurable).
- Rate limits: Cap actions per turn and per minute to avoid runaway loops. Exceeding limits aborts remaining inputs for the turn.
- Emergency stop: Use Ctrl-C in the terminal running the main script.

### Failsafe Policy

- If Vision reports no interactable buttons after N retries (configurable, e.g., 2–3) in a turn:
  - Attempt a single `advanceDialogue()` as a gentle nudge.
  - If still no change, end the turn with a clear diagnostic (no blind clicks).
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
- Each turn performs: window check → capture → Vision → Agent decision → inputs → optional verification capture → end-of-turn padding.
- Turn latency: Highly variable by model/load; budget ~10–30s. Add a fixed 5s padding between turns to smooth variance.
- Timeouts: Treat any single LLM request as failed after 30s; skip inputs for that turn and report the timeout.
- Parallel Vision: Multiple Vision calls per turn are allowed (e.g., separate `Primary` and `Menus` prompts) and may run in parallel.

Context and summarization
- The model receives prior-turn reasoning and tool metadata (excluding `writeMemoryFile` content, which it already knows) until the turn context exceeds ~32k tokens.
- Summarization trigger: when turn context > 32k, request a summary; carry the summary forward and discard detailed prior turns.
- Summary guidance: capture game state deltas (stats, inventory, resources), current objectives, recent UI location, and any unresolved blockers.

Memory scratchpad
- The agent manages its scratchpad via `createMemoryFile(name)`, `deleteMemoryFile(name)`, `writeMemoryFile(name, content)`.
- Cumulative scratchpad budget: also 32k tokens across all files. Prefer compact structured text (e.g., YAML/JSON-lite) over verbose prose.
- Recommended files: `player.yaml`, `run.yaml`, `ui_knowledge.md`, `todo.md`. Delete or truncate obsolete files proactively.
- Note: The turn context budget and the memory scratchpad budget are both 32k tokens, but they are separate allocations (turn history vs. persistent memory).

Safety and limits
- Max actions per turn: 10 (default). Max actions per minute: 60 (default). Exceeding limits aborts remaining inputs.
- If Vision returns no actionable buttons after retries, attempt one `advanceDialogue()` then end the turn with a diagnostic (no blind inputs).

# Configuration

Centralized configuration defines environment- and game-specific parameters. Use logical pixels for client-area measurements.

- Window
  - `window_title`: "Umamusume"
  - `placement`: `{ x: 0, y: 0, width: 1920, height: 1080 }` (outer size)
  - `dpi.scaling_factor`: 1.5 (for 150% system scaling)

- Capture & Vision
  - `capture.post_action`: false (optional extra capture after inputs)
  - `capture.retention`: 200 (number of turns to keep)
  - `split.left_pin_ratio`: 0.07874 (proportion of width for left pin strip, discarded)
  - `split.primary_ratio`: 0.42126 (proportion of width for primary gameplay area)
  - `split.menus_ratio`: 0.50 (proportion of width for menus/UI area)
  - `vision.parallel_calls`: true (allow per-crop prompts concurrently)

- Input behavior
  - `input.debounce_ms`: 300
  - `input.jitter_ms`: { min: 20, max: 50 }
  - `input.max_actions_per_turn`: 10
  - `input.max_actions_per_minute`: 60
  - `input.allow_skip_cinematics`: false

- Turns & timeouts
  - `turn.post_padding_s`: 5
  - `llm.request_timeout_s`: 30
  - `context.summarize_over_tokens`: 32000

Recommended location: `config.yaml` at repo root. Environment overrides allowed via `LLUMA_*` env vars (optional).

# Initial State & Stopping

## Initial Game State
The agent should be state-agnostic and capable of orienting itself regardless of the initial game state. It should assess the current screen and available actions on its first turn and proceed accordingly.

## Stop Conditions
The agent runs indefinitely until manual intervention (Ctrl-C in the terminal). No automatic stop conditions are implemented. For production runs, the game account should be reset to a fresh state; for testing, a mostly fresh account is acceptable.

## Testing & Debugging
- No mock/replay testing infrastructure is currently available; all testing is performed against the live game.
- Debug logging can be configured via verbosity levels (error, warn, info, debug, trace).
- Captures are retained (last 200 turns by default) for post-mortem analysis.

## Error Recovery
- **Window closure**: If the game window is closed, the program immediately stops.
- **Checkpoint/resume**: Not currently implemented (TODO).
- **Loop detection**: The agent should detect repeated screen states and use `advanceDialogue()` or `back()` to break loops.

# Logging

Structured logs aid debugging and reproducibility without exposing sensitive data.

- Per-turn JSON: `logs/turn-<turn_id>.json`
  - Includes: timestamps, window bounds, capture filenames, Vision detections (normalized and client px), tool calls, inputs issued, outcomes, and errors.
  - Excludes: inline image data; reference files by path only.

- Session log: `logs/session.jsonl`
  - One line per significant event (turn start/end, timeouts, emergency stop toggles, window placement, resize).

- Screenshots
  - `captures/<turn_id>.png`, `captures/<turn_id>-primary.png`, `captures/<turn_id>-menus.png` (when enabled)
  - Retention respects `capture.retention`.

- Verbosity
  - Levels: error, warn, info (default), debug, trace.
  - Debug/trace may include intermediate Vision prompts (text only) and crop rectangles.

# Setup & Dependencies

Install Python 3.13.8 (x64), then install dependencies via `pip install -r requirements.txt`.

- Confirmed runtime deps
  - OS I/O: `pyautogui`, `pygetwindow`, `mss`
  - Imaging: `pillow`, `numpy`
  - Agent/VLM: `openai` (for OpenRouter client)
  - Config: `pyyaml`

- Provisional deps (enable if/when needed)
  - JSON performance: `orjson`
  - Retry utilities: `tenacity`
  - Image processing extras: `opencv-python-headless`, `scikit-image`, `ImageHash`, `pytesseract`

Environment
- Set `OPENROUTER_API_KEY` in the environment for Vision and Agent calls.
- Ensure Steam window title is exactly `Umamusume`.
- First-run checklist: verify window placement (0,0 @ 1920×1080), confirm capture works, adjust split ratios as needed.

# TODO / Future Work

- **Runtime DPI measurement**: Currently hardcoded at 1.5 (150% scaling); consider measuring at runtime for robustness across different display configurations.
- **Checkpoint/resume**: Implement save/restore capability to resume agent sessions after interruption.
- **Mock/replay testing**: Build infrastructure to test against recorded captures without running the live game.
- **VLM prompt refinement**: The current prompts may need iteration based on actual detection performance; monitor false positives/negatives and adjust accordingly.
