# CONTEXT.md

## Purpose

This document tracks the prompt context structure, token usage patterns, and optimization strategies for the LLuMa Musume agent. Understanding context composition is critical for managing costs, speed, and rate limits.

---

## Current Context Structure

Each turn sends the following to the Anthropic API:

### 1. System Prompt (Static, ~2,300 tokens)
- Role description and UI layout explanation
- Turn structure overview
- Tool descriptions (6 input tools + 3 memory tools)
- Safety guidelines and edge case handling
- Decision-making guidance

**Caching potential:** HIGH (never changes)

### 2. Tool Definitions (Static, ~1,500 tokens)
- 9 tool schemas in Anthropic format
- Input tools: `pressButton`, `advanceDialogue`, `back`, `confirm`, `scrollUp`, `scrollDown`
- Memory tools: `createMemoryFile`, `writeMemoryFile`, `deleteMemoryFile`

**Caching potential:** HIGH (never changes)

### 3. Message History (Variable, up to ~20 messages)
- User messages: Turn context text only (images removed after current turn)
- Assistant messages: Thinking blocks + text + tool_use blocks (preserved for continuity)
- Tool result messages: Acknowledgment of tool execution

**Current pruning:** Keep last 20 messages (`max_history_messages`)

**Caching potential:** MEDIUM (stable prefix can be cached, but tail changes every turn)

### 4. Current Turn User Message (Variable, ~5K-10K tokens typical)

#### a. Turn Metadata (~50 tokens)
- Turn ID (e.g., `turn_000123`)
- Timestamp

#### b. Vision Data JSON (~3K-6K tokens, **HIGH COST AREA**)
**Buttons array** (biggest contributor):
- Each button: ~150-200 tokens
- Fields: `name`, `full_label`, `bounds`, `metadata`, `region`
- Example: 21 buttons = ~3,500-4,000 tokens

**Scrollbar** (when present): ~100 tokens
- Fields: `track_bounds`, `thumb_bounds`, `can_scroll_up`, `can_scroll_down`, `thumb_ratio`

**Menu state**: ~300-500 tokens
- Fields: `is_usable`, `selected_tab`, `tabs[]`, `available_tabs[]`
- Tab entries: `name`, `is_selected`, `availability`

#### c. Screenshots (1.5K-3K tokens)
- Primary region: ~1.5K tokens (always present)
- Menu region: ~1.5K tokens (when relevant)

#### d. Memory Files (up to 32K token budget)
- Agent-managed scratchpad files
- Injected at end of context
- Examples: `player.yaml`, `run.yaml`, `ui_knowledge.md`, `todo.md`

**Caching potential:** MEDIUM (memory files change infrequently, could cache stable ones)

---

## Token Usage Measurements

### Example Turn Analysis (turn_000006)
```
Input tokens:  24,479
Output tokens:     595 (includes ~300 thinking tokens)

Breakdown (estimated):
- System prompt:        ~2,300
- Tool definitions:     ~1,500
- Message history:      ~8,000-10,000 (varies by turn)
- Turn metadata:           ~50
- Vision JSON:          ~4,500 (21 buttons + scrollbar + menu_state)
- Screenshots:          ~3,000 (2 images)
- Memory files:         ~4,000-6,000 (varies)
```

### Growth Patterns
- **Images:** Fixed (1-2 per turn, older images not accumulated) → ~1.5K-3K/turn
- **Thinking blocks:** ~300-500 tokens/turn (kept in message history until pruned)
- **Button JSON:** Highly variable (10-30 buttons typical) → ~2K-6K/turn
- **Memory files:** Slow growth (agent manages proactively) → target <32K cumulative

---

## Optimization Targets

### Priority 1: Button Metadata Format (High Impact)

**Current format** (per button):
```json
{
  "name": "Archive",
  "full_label": "Archive | hint=NEW | section=Umamusume",
  "bounds": [907, 418, 193, 49],
  "metadata": {
    "hint": "NEW",
    "section": "Umamusume"
  },
  "region": "menus"
}
```
**Token cost:** ~180 tokens/button

**Redundancy identified:**
1. `full_label` duplicates `name` + `metadata` (agent never uses full_label directly)
2. `metadata` dict uses verbose JSON keys/braces even when empty or simple
3. `region` is always short but adds overhead for every button

**Optimization options:**

**Option A (RECOMMENDED): Remove bounds + compact metadata**
```json
{
  "name": "Archive",
  "meta": "hint=NEW",
  "region": "menus"
}
```
**Rationale:**
- **Bounds removed:** Agent has screenshot for spatial reasoning, bounds only used by input handler internally
- **`full_label` removed:** Redundant with name + meta
- **`metadata` → `meta`:** Compact string instead of dict (can omit if empty)

**Savings:** ~60-65% per button (~180 → ~65 tokens)

**Scrollbar trimmed:**
```json
{"up": true, "down": false}  // was: track_bounds, thumb_bounds, thumb_ratio, can_scroll_*
```

**Menu state trimmed:**
```json
{"tab": "Menu", "available": ["Jukebox", "Menu"]}  // was: is_usable, selected_tab, tabs[], available_tabs[]
```

**Total current-turn savings:** ~4,500 → ~1,600 tokens (**65% reduction**)

### Priority 2: Historical Turn Trimming (HIGHEST Impact)

**Current behavior:** Message history preserves full user messages (minus images):
- Turn metadata (~50 tokens)
- Full vision JSON (~4,500 tokens) ← **WASTEFUL**
- Memory files (~4,000+ tokens) ← **WASTEFUL (memory is re-injected every turn anyway)**

**Per historical turn:** ~8,500+ tokens
**After 10 turns:** ~85,000 tokens of redundant data

**New behavior:** Strip vision JSON and memory from historical user messages, replace with compact summary:

```
Turn turn_000005 (2025-11-01T18:15:22)
Buttons: Archive, Profile, Titles, Trophy Room, Friends, Options, Comics, Secrets, Info button, Back button, Story menu button, Home menu button, Race menu button, Scout menu button
Menu: tab=Menu, available=[Jukebox, Menu]
Scrollbar: none
```

**Per historical turn:** ~150-200 tokens (**98% reduction**)

**Rationale:**
1. **Button details:** Agent doesn't need bounds, metadata, or regions for past turns - just awareness of what was available
2. **Memory files:** Already re-injected in current turn, no need to preserve in history
3. **Vision JSON structure:** Not needed for continuity, agent just needs to know "what options were visible"
4. **Action taken:** Preserved in assistant's tool_use block (already in history)

**Implementation approach:**
- After agent responds, before storing user message in `_message_history`:
  - Extract button names only
  - Extract menu tab + available tabs
  - Extract scrollbar yes/no
  - Format as compact text (not JSON)
  - Replace the content

**Combined savings (historical turns):**
- Current: ~8,500 tokens/turn in history
- Optimized: ~150 tokens/turn in history
- **Net savings: ~8,350 tokens per historical turn**
- **After 10 turns: ~83,500 tokens saved**

### Priority 3: Prompt Caching (Medium-High Impact)

Anthropic's prompt caching allows reuse of static or semi-static prompt prefixes across turns.

**Cache breakpoints strategy:**

1. **Static system context** (system prompt + tools) → Cache indefinitely
   - System prompt (~2,300 tokens)
   - Tool definitions (~1,500 tokens)
   - **Savings:** ~3,800 tokens/turn after first turn

2. **Memory files** (when stable) → Cache until modified
   - Mark memory files with cache breakpoint after assistant response
   - Refresh when `writeMemoryFile` or `deleteMemoryFile` used
   - **Savings:** Variable (0-32K tokens/turn depending on memory size and stability)

3. **Message history prefix** (older messages) → Cache stable prefix
   - Mark a breakpoint after message N-5 (keep recent 5 messages uncached)
   - Refresh when prefix changes
   - **Savings:** Variable (~4K-8K tokens/turn typical)

**Implementation requirements:**
- Anthropic SDK `cache_control` parameter on message content blocks
- Track cache invalidation (memory file writes, history pruning)
- Monitor cache hit rates in usage stats

**Expected total savings:** 40-60% of input tokens after warmup

### Priority 4: Context Summarization (Low-Medium Impact, safety net)

When approaching context limits (e.g., >180K tokens cumulative), trigger automatic summarization.

**Trigger:** Total input tokens exceeds `max_context_tokens` threshold (current: 32K, but this is currently only for memory; we need a separate message history limit)

**Summarization strategy:**
1. Agent receives a special `summarizeContext` tool call request
2. Agent reviews full message history and produces:
   - High-level summary of game progress and discoveries
   - Current objectives and state
   - Key UI patterns learned
3. Replace all message history with a single summarized "system message"
4. Keep memory files intact (they have their own 32K budget)
5. Resume normal operation

**Implementation approach:**
- Add token counting to `_message_history` (track cumulative input tokens)
- When threshold exceeded, inject summarization request
- Replace `_message_history` with synthetic summary message
- Log summarization events for debugging

**Expected benefit:** Prevents context overflow, extends session longevity

---

## Implementation Roadmap

### Phase 1: Historical Turn Trimming (Highest Impact, Easy Win)
- [ ] Add `_format_compact_turn_summary()` method to `UmaAgent`
- [ ] Modify `execute_turn()` to store compact summary instead of full vision JSON
- [ ] Keep full vision JSON only in current turn user message
- [ ] Test that agent continuity is preserved
- [ ] Measure token savings (expect ~8K/turn after first few turns)

### Phase 2: Current Turn Vision JSON Optimization (High Impact, Medium Effort)
- [ ] Update coordinator's `_process_vision()` to emit compact button format for agent
  - Remove `bounds` from `VisionData` passed to agent (keep in `VisionOutput` for input handler)
  - Remove `full_label`
  - Change `metadata` → `meta` (compact string, omit if empty)
  - Keep `region` (useful for agent to understand menus vs primary)
- [ ] Update scrollbar format: `{"up": bool, "down": bool}`
- [ ] Update menu format: `{"tab": str, "available": [str]}`
- [ ] Update system prompt to document new format
- [ ] Ensure input handler still receives full bounds via `VisionOutput` (separate data path)
- [ ] Test button name matching still works
- [ ] Measure token savings (expect ~3K/turn reduction)

**Note:** The coordinator maintains two separate representations:
- `VisionData` (for agent): Compact, no bounds
- `VisionOutput` (for input handler): Full bounds included

### Phase 3: Prompt Caching (Medium-High Impact, Medium Complexity)
- [ ] Add cache breakpoints to system prompt + tools
- [ ] Implement memory file cache invalidation logic
- [ ] Add message history prefix caching (stable older messages)
- [ ] Monitor cache hit rates in turn logs
- [ ] Measure cost and latency improvements

### Phase 4: Context Summarization (Safety Net)
- [ ] Add cumulative token tracking to message history
- [ ] Define summarization trigger threshold
- [ ] Implement summarization prompt and flow
- [ ] Test summarization quality
- [ ] Deploy as safety net for long sessions

---

## Monitoring and Metrics

Track these in turn logs (`logs/<turn_id>.json`):

1. **Token usage per turn:**
   - `input_tokens` (total)
   - `output_tokens` (including thinking)
   - `cache_creation_input_tokens` (new cache writes)
   - `cache_read_input_tokens` (cache hits)

2. **Context composition:**
   - Button count (track variability)
   - Memory file count and total tokens
   - Message history length (number of messages)

3. **Cache effectiveness:**
   - Cache hit rate: `cache_read / (cache_read + uncached_input)`
   - Cost savings: cached tokens are ~90% cheaper

4. **Summarization events:**
   - Timestamp of summarization
   - Token count before/after
   - Summary quality (manual review)

---

## Cost Analysis

### Current Costs (Claude Haiku 4.5)
- Input: $0.80 / 1M tokens
- Output: $4.00 / 1M tokens
- Cached input: $0.08 / 1M tokens (10x cheaper)

**Per turn (example, no caching):**
- 24,479 input tokens = $0.0196
- 595 output tokens = $0.0024
- **Total: ~$0.022/turn**

### With Optimizations

**Phase 1: Historical turn trimming:**
- Per historical turn: ~8,500 → ~150 tokens
- After 10 turns: ~85K → ~1.5K tokens in history
- **Savings at turn 10: ~$0.067** (cumulative from all trimmed turns)
- **Savings at turn 20: ~$0.134** (grows linearly)

**Phase 2: Current turn vision optimization:**
- Vision JSON: ~4,500 → ~1,600 tokens/turn
- **Savings: ~$0.0023/turn** (10% cost reduction per turn)

**Phase 3: Prompt caching (assuming 60% cache hit after warmup):**
- ~14,700 cached tokens, ~6,000 uncached (after vision optimization)
- Input cost: (14,700 × $0.08 + 6,000 × $0.80) / 1M = $0.0060
- **Savings: ~$0.0136/turn** (69% input cost reduction)

**Combined optimizations (at turn 20):**
- Historical savings: ~$0.134 (cumulative)
- Per-turn savings: ~$0.016/turn
- **Cost per turn: ~$0.006** (73% reduction from baseline)

**At scale (100 turns):**
- Current: ~$2.20
- Optimized: ~$0.60
- **Net savings: $1.60** (73% reduction)

**At scale (1,000 turns):**
- Current: ~$22.00
- Optimized: ~$6.00
- **Net savings: $16.00** (73% reduction + faster inference + lower rate limits)

---

## Future Considerations

1. **Vision JSON compression:**
   - Investigate JSON alternatives (msgpack, compressed strings)
   - Trade-off: complexity vs. token savings

2. **Selective memory injection:**
   - Only inject memory files relevant to current screen state
   - Requires context-aware memory file tagging

3. **Message history pruning strategies:**
   - Prune less informative turns (e.g., simple dialogue advances)
   - Keep high-value turns (discoveries, decision points)

4. **VLM prompt optimization:**
   - Reduce button label verbosity at VLM output stage
   - Requires VLM prompt updates in `lluma_vision/menu_analyzer.py`

---

## References

- Agent implementation: `lluma_agent/agent.py`
- Coordinator (vision processing): `lluma_agent/coordinator.py`
- System prompt: `lluma_agent/prompts.py`
- Tool definitions: `lluma_agent/tools.py`
- Anthropic Caching docs: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
