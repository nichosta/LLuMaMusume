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
**Status:** Ready for caching implementation

### 2. Tool Definitions (Static, ~1,500 tokens)
- 9 tool schemas in Anthropic format
- Input tools: `pressButton`, `advanceDialogue`, `back`, `confirm`, `scrollUp`, `scrollDown`
- Memory tools: `createMemoryFile`, `writeMemoryFile`, `deleteMemoryFile`

**Caching potential:** HIGH (never changes)
**Status:** Ready for caching implementation

### 3. Message History (Variable, ~1-5K tokens typical after optimizations)
- User messages: Compact turn summaries for historical turns (~150-200 tokens each)
- Assistant messages: Thinking blocks + text + tool_use blocks (preserved for continuity)
- Tool result messages: Acknowledgment of tool execution

**Current optimization:** Historical turns use compact format (buttons names only, minimal menu/scrollbar state)
**Caching potential:** MEDIUM (stable prefix can be cached, but tail changes every turn)

### 4. Current Turn User Message (Variable, ~3K-6K tokens typical after optimizations)

#### a. Turn Metadata (~50 tokens)
- Turn ID (e.g., `turn_000123`)
- Timestamp

#### b. Vision Data JSON (~1K-3K tokens after optimization)
**Buttons array** (optimized format):
- Each button: ~65 tokens (down from ~180)
- Fields: `name`, `meta` (optional compact string), `region`
- Removed: `full_label`, `bounds`, verbose `metadata` dict
- Example: 21 buttons = ~1,365 tokens (down from ~3,780)

**Scrollbar** (when present): ~30 tokens (down from ~100)
- Compact format: `{"up": bool, "down": bool}`
- Removed: `track_bounds`, `thumb_bounds`, `thumb_ratio`

**Menu state**: ~50 tokens (down from ~300)
- Compact format: `{"tab": str, "available": [str]}`
- Removed: `is_usable`, `tabs[]` array with full objects

#### c. Screenshots (1.5K-3K tokens)
- Primary region: ~1.5K tokens (always present)
- Menu region: ~1.5K tokens (when relevant)

#### d. Memory Files (up to 32K token budget)
- Agent-managed scratchpad files
- Injected at end of context
- Examples: `player.yaml`, `run.yaml`, `ui_knowledge.md`, `todo.md`

**Caching potential:** MEDIUM-HIGH (memory files change infrequently, could cache stable ones)

---

## Implemented Optimizations (PR #12)

### ✅ Phase 1: Historical Turn Trimming (Commit be95a8b)

**Implementation:**
- `_format_compact_turn_context()` method in agent.py (lines 463-532)
- Stores only button names, minimal menu state, and scrollbar presence for historical turns
- Full vision JSON and memory files are stripped (memory is re-injected every turn anyway)

**Savings:**
- Per historical turn: ~8,500 → ~150-200 tokens (**98% reduction**)
- After 10 turns: ~85K → ~1.5K tokens in history
- After 20 turns: ~170K → ~3K tokens saved

**Example compact format:**
```
# Turn turn_000005
Timestamp: 2025-11-01T18:15:22.123456

Buttons: Archive, Profile, Titles, Friends, Options, Info button, Back button
Menu: tab=Menu, available=['Jukebox', 'Menu']
Scrollbar: none
```

### ✅ Phase 2: Current Turn Vision JSON Optimization (Commit eddae21)

**Implementation:**
- Coordinator emits compact button format for agent (coordinator.py, lines 490-503)
- Maintains separate `VisionData` (for agent, compact) and `VisionOutput` (for input handler, full bounds)
- Internal `_bounds` field stripped before sending to agent (agent.py, lines 274-278)
- System prompt updated to document compact format (prompts.py)

**Button format changes:**
- Before: `{"name": "Archive", "full_label": "Archive | hint=NEW", "bounds": [...], "metadata": {"hint": "NEW"}, "region": "menus"}` (~180 tokens)
- After: `{"name": "Archive", "meta": "hint=NEW", "region": "menus"}` (~65 tokens)
- **Savings: ~65% per button**

**Scrollbar format:**
- Before: Full bounds, thumb_bounds, thumb_ratio, can_scroll_* (~100 tokens)
- After: `{"up": true, "down": false}` (~30 tokens)

**Menu state format:**
- Before: Full tab objects with is_usable, selected_tab, tabs[] (~300 tokens)
- After: `{"tab": "Menu", "available": ["Jukebox", "Menu"]}` (~50 tokens)

**Total current-turn savings:** ~4,500 → ~1,600 tokens (**65% reduction**)

### ✅ Phase 4: Context Summarization (Commit 0b7021c)

**Implementation:**
- `_maybe_queue_summarization()` checks if input tokens exceed threshold (agent.py, lines 576-600)
- `_perform_summarization()` replaces entire history with single summary message (agent.py, lines 602-669)
- Trigger: min(summarization_threshold_tokens, 90% of max_context_tokens)
- Default threshold: 64,000 tokens (configurable via config.yaml)

**How it works:**
1. After each turn, check if last input token count exceeded trigger
2. If yes, queue summarization for **next turn** (before executing it)
3. Call model with existing history + summarization request
4. Extract summary and replace `_message_history` with single synthetic message
5. Resume normal operation with compact context

**SUMMARIZATION_PROMPT:** Asks agent to summarize game progress, UI knowledge, current state, and key discoveries

---

## Remaining Work

### 🚧 Phase 3: Prompt Caching (IN PROGRESS)

**Goal:** Reduce token costs by 40-60% after warmup via Anthropic's prompt caching

**Implementation plan:**

1. **Static system context caching** (highest value)
   - Add `cache_control: {type: "ephemeral"}` to system prompt
   - Add cache breakpoint after tools in messages array
   - **Savings:** ~3,800 tokens/turn after first turn (90% cheaper when cached)

2. **Memory file caching** (when stable)
   - Track which memory files have changed since last turn
   - Add cache breakpoint after memory content when stable
   - Invalidate when `writeMemoryFile` or `deleteMemoryFile` used
   - **Savings:** 0-32K tokens/turn depending on memory size and stability

3. **Message history prefix caching** (medium value)
   - Cache stable older messages in history
   - Keep recent 3-5 messages uncached (they change frequently)
   - Refresh when history is pruned or summarized
   - **Savings:** ~1K-3K tokens/turn typical

**Expected total savings:** 40-60% of input tokens after warmup

**Current status:** agent.py already captures cache stats (lines 196-199), ready for implementation

---

## Token Usage Measurements

### Before Optimizations (Early Testing)
```
Input tokens:  24,479
Output tokens:     595 (includes ~300 thinking tokens)

Breakdown (estimated):
- System prompt:        ~2,300
- Tool definitions:     ~1,500
- Message history:      ~8,000-10,000 (10+ full historical turns)
- Turn metadata:           ~50
- Vision JSON:          ~4,500 (21 buttons, full format)
- Screenshots:          ~3,000 (2 images)
- Memory files:         ~4,000-6,000
```

### After Optimizations (Current)
```
Input tokens:  ~12,000-15,000 (40-50% reduction)
Output tokens:     ~595

Breakdown (estimated):
- System prompt:        ~2,300
- Tool definitions:     ~1,500
- Message history:      ~1,500-3,000 (compact format)
- Turn metadata:           ~50
- Vision JSON:          ~1,600 (compact format)
- Screenshots:          ~3,000
- Memory files:         ~4,000-6,000
```

### With Caching (Projected)
```
Input tokens:  ~5,000-8,000 uncached + ~8,000-10,000 cached
Effective cost: ~$0.005/turn (vs ~$0.010 optimized, ~$0.020 baseline)
**70-75% cost reduction from baseline**
```

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

### Baseline (No Optimizations)
**Per turn:**
- 24,479 input tokens = $0.0196
- 595 output tokens = $0.0024
- **Total: ~$0.022/turn**

**At scale (100 turns):** ~$2.20

### With Phase 1 + 2 Optimizations (Current)
**Per turn after optimizations stabilize:**
- ~13,000 input tokens = $0.0104
- ~595 output tokens = $0.0024
- **Total: ~$0.013/turn**

**At scale (100 turns):** ~$1.30 (**41% savings**)

### With Phase 3: Prompt Caching (Projected)
**Assumptions:**
- System prompt + tools: ~3,800 tokens (100% cache hit after turn 1)
- Memory files: ~5,000 tokens (80% cache hit after stabilization)
- Message history prefix: ~2,000 tokens (70% cache hit)
- Current turn + recent history: ~5,000 tokens (0% cached)

**Token breakdown:**
- Cached: ~9,000 tokens @ $0.08/1M = $0.00072
- Uncached: ~5,000 tokens @ $0.80/1M = $0.00400
- Output: ~595 tokens @ $4.00/1M = $0.00238
- **Total: ~$0.007/turn**

**At scale (100 turns):** ~$0.70 (**68% savings from baseline, 46% from current**)

### With All Optimizations + Summarization
**At 1,000 turns (with 2-3 summarization events):**
- Baseline: ~$22.00
- Optimized + cached: ~$7.00
- **Net savings: $15.00 (68% reduction)**

**Additional benefits:**
- Faster inference (less tokens to process)
- Lower rate limit pressure
- Indefinite session length via summarization

---

## Future Considerations

1. **Selective memory injection:**
   - Only inject memory files relevant to current screen state
   - Requires context-aware memory file tagging
   - Could reduce memory overhead by 50-70% on some turns

2. **Message history pruning strategies:**
   - Prune less informative turns (e.g., simple dialogue advances)
   - Keep high-value turns (discoveries, decision points)
   - Could reduce history size by additional 30-40%

3. **Vision token optimization:**
   - Screenshot compression or downsampling (trade-off: VLM accuracy)
   - Skip menu screenshot when tab hasn't changed
   - Potential 20-30% reduction in image tokens

4. **VLM prompt optimization:**
   - Reduce button label verbosity at VLM output stage
   - Requires updates in `lluma_vision/menu_analyzer.py`
   - Could reduce VLM processing cost and initial button token count

---

## References

- Agent implementation: `lluma_agent/agent.py`
- Coordinator (vision processing): `lluma_agent/coordinator.py`
- System prompt: `lluma_agent/prompts.py`
- Tool definitions: `lluma_agent/tools.py`
- Anthropic Caching docs: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- PR #12 (context management): Commits be95a8b, eddae21, 0b7021c
