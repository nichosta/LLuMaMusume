"""Main agent orchestration for Uma Musume gameplay."""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from PIL import Image

from .memory import MemoryManager, MemoryError
from .prompts import AGENT_SYSTEM_PROMPT
from .tools import ALL_TOOLS, INPUT_TOOL_NAMES, MEMORY_TOOL_NAMES

Logger = logging.Logger


class AgentError(RuntimeError):
    """Raised when agent execution fails."""


@dataclass(slots=True)
class VisionData:
    """Structured vision data for a turn."""

    buttons: List[Dict[str, Any]]
    scrollbar: Optional[Dict[str, Any]]
    menu_state: Dict[str, Any]


@dataclass(slots=True)
class TurnResult:
    """Result of executing a turn."""

    turn_id: str
    reasoning: str
    thinking: Optional[str]  # Extended thinking content (if enabled)
    memory_actions: List[Dict[str, Any]]
    input_action: Optional[Dict[str, Any]]
    execution_results: List[str]
    timestamp: str
    usage: Optional[Dict[str, int]] = None  # Token usage stats from API


class UmaAgent:
    """Main agent that coordinates LLM reasoning and tool execution."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        model: str = "claude-haiku-4-5",
        max_context_tokens: int = 32000,
        thinking_enabled: bool = True,
        thinking_budget_tokens: int = 16000,
        max_tokens: int = 4096,
        max_history_messages: int = 20,
        summarization_threshold_tokens: int = 150000,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            memory_manager: Memory file manager instance
            model: Anthropic model identifier (e.g., "claude-haiku-4-5")
            max_context_tokens: Maximum context window size for turn history
            thinking_enabled: Whether to enable extended thinking
            thinking_budget_tokens: Max tokens for internal reasoning (min 1024, recommended 16000+)
            max_tokens: Maximum tokens for response (must exceed thinking_budget_tokens)
            max_history_messages: Maximum messages to keep in history (0 = unlimited)
            summarization_threshold_tokens: Trigger summarization when cumulative input exceeds this
            logger: Optional logger instance
        """
        self._memory = memory_manager
        self._logger = logger or logging.getLogger(__name__)
        self._client: Optional[Anthropic] = None
        self._message_history: List[Dict[str, Any]] = []  # Full message history for API
        self._turn_summaries: List[str] = []  # Human-readable summaries for logging
        self._model = model
        self._max_context_tokens = max_context_tokens
        self._thinking_enabled = thinking_enabled
        self._thinking_budget_tokens = thinking_budget_tokens
        self._max_tokens = max_tokens
        self._max_history_messages = max_history_messages
        self._summarization_threshold_tokens = summarization_threshold_tokens

        # Initialize Anthropic API client
        self._init_api()

    def execute_turn(
        self,
        turn_id: str,
        vision_data: VisionData,
        primary_screenshot: Path,
        menus_screenshot: Optional[Path] = None,
    ) -> TurnResult:
        """Execute a single turn: get LLM decision and execute tools.

        Context optimization: The current turn receives full vision data + screenshots,
        but when stored in message history, only a compact summary is kept (~150 tokens
        vs ~8,500). This prevents exponential token growth while preserving continuity.

        Args:
            turn_id: Unique turn identifier
            vision_data: Structured vision detection results
            primary_screenshot: Path to primary region screenshot
            menus_screenshot: Optional path to menus region screenshot

        Returns:
            Turn result with reasoning and executed actions

        Raises:
            AgentError: If turn execution fails
        """
        timestamp = datetime.now().isoformat()
        self._logger.info("Starting turn %s", turn_id)

        # Check if we need to summarize before executing this turn
        estimated_prompt_tokens = self._estimate_prompt_size()
        if self._should_summarize(estimated_prompt_tokens):
            self._logger.warning(
                "Estimated prompt size (%d tokens) exceeded threshold (%d), triggering summarization",
                estimated_prompt_tokens,
                self._summarization_threshold_tokens
            )
            self._perform_summarization()

        # Build context message (without turn history - that's in message_history now)
        user_message = self._format_context(turn_id, timestamp, vision_data)

        # Add screenshots (Anthropic format)
        message_content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_message},
            self._encode_image(primary_screenshot),
        ]

        if menus_screenshot is not None and menus_screenshot.exists():
            message_content.append(self._encode_image(menus_screenshot))

        # Build messages array for API call:
        # - All historical messages (without images)
        # - Current turn message (WITH images)
        # This prevents image token accumulation while keeping current turn visual
        messages_for_api = self._message_history + [{"role": "user", "content": message_content}]

        # Call LLM
        try:
            # Build thinking config
            thinking_config = None
            if self._thinking_enabled:
                thinking_config = {
                    "type": "enabled",
                    "budget_tokens": self._thinking_budget_tokens,
                }

            self._logger.info("Calling Anthropic model %s for turn %s (history: %d messages)",
                            self._model, turn_id, len(messages_for_api))
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=AGENT_SYSTEM_PROMPT,
                messages=messages_for_api,
                tools=ALL_TOOLS,
                thinking=thinking_config,
            )
        except Exception as exc:
            raise AgentError(f"LLM request failed: {exc}") from exc

        # Store compact turn summary in history (no images, no full vision JSON, no memory files)
        # This dramatically reduces token accumulation in message history
        compact_summary = self._format_compact_turn_context(turn_id, timestamp, vision_data)
        self._logger.debug("Storing compact turn summary in history (~%d chars vs ~%d for full context)",
                          len(compact_summary), len(user_message))
        self._message_history.append({
            "role": "user",
            "content": [{"type": "text", "text": compact_summary}]
        })

        # Extract reasoning and tool calls
        reasoning, thinking = self._extract_reasoning(response)
        tool_calls = self._extract_tool_calls(response)

        # Extract token usage
        usage = None
        if hasattr(response, 'usage'):
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            # Cache stats are optional
            if hasattr(response.usage, 'cache_creation_input_tokens'):
                usage["cache_creation_input_tokens"] = response.usage.cache_creation_input_tokens
            if hasattr(response.usage, 'cache_read_input_tokens'):
                usage["cache_read_input_tokens"] = response.usage.cache_read_input_tokens

            self._logger.info("Turn %s usage - Input: %d, Output: %d tokens",
                            turn_id, usage["input_tokens"], usage["output_tokens"])

        # Execute tools
        memory_actions, input_action, execution_results = self._execute_tools(tool_calls)

        # Store assistant response in message history (preserves thinking blocks)
        # IMPORTANT: Thinking blocks must be passed back UNMODIFIED per Anthropic docs
        # Use model_dump() to preserve all fields including signature
        assistant_content = []
        for block in response.content:
            # Convert Pydantic model to dict, preserving all fields
            block_dict = block.model_dump()
            assistant_content.append(block_dict)
        self._message_history.append({"role": "assistant", "content": assistant_content})

        # If the assistant used tools, we must add a tool_result message immediately after
        # This follows Anthropic's required conversation pattern for tool use
        tool_result_content = []
        for block in response.content:
            if block.type == "tool_use":
                # Create a success result for each tool
                # The actual execution happens via input_handler, so we just acknowledge success
                result_text = f"Tool {block.name} executed successfully."
                if block.name in MEMORY_TOOL_NAMES:
                    # Memory tools have actual results we can report
                    matching_result = next(
                        (r for r in execution_results if block.name in r),
                        result_text
                    )
                    result_text = matching_result

                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        # Add tool results message if any tools were used
        if tool_result_content:
            self._message_history.append({"role": "user", "content": tool_result_content})

        # Create human-readable summary for logging
        turn_summary = self._format_turn_summary(turn_id, reasoning, memory_actions, input_action)
        self._turn_summaries.append(turn_summary)

        # Prune old messages if history exceeds limit
        if self._max_history_messages > 0 and len(self._message_history) > self._max_history_messages:
            excess = len(self._message_history) - self._max_history_messages
            self._logger.info("Pruning %d old messages from history (keeping most recent %d)",
                            excess, self._max_history_messages)
            self._message_history = self._message_history[-self._max_history_messages:]

        return TurnResult(
            turn_id=turn_id,
            reasoning=reasoning,
            thinking=thinking,
            memory_actions=memory_actions,
            input_action=input_action,
            execution_results=execution_results,
            timestamp=timestamp,
            usage=usage,
        )

    def _format_context(self, turn_id: str, timestamp: str, vision_data: VisionData) -> str:
        """Format the context message for the LLM.

        Combines turn metadata, vision data, and memory files.
        Note: Turn history is now passed via the messages array, not embedded in text.
        """
        sections = []

        # Turn metadata
        sections.append(f"# Turn {turn_id}\n\nTimestamp: {timestamp}\n")

        # Vision data (JSON) - strip internal fields (_bounds)
        # These are only for the input handler, not the agent
        clean_buttons = [
            {k: v for k, v in btn.items() if not k.startswith("_")}
            for btn in vision_data.buttons
        ]

        vision_json = {
            "buttons": clean_buttons,
            "scrollbar": vision_data.scrollbar,
            "menu_state": vision_data.menu_state,
        }
        sections.append("# Vision Data\n")
        sections.append("```json")
        sections.append(json.dumps(vision_json, indent=2))
        sections.append("```\n")

        # Memory files
        sections.append(self._memory.get_all_content())

        return "\n".join(sections)

    def _execute_tools(
        self, tool_calls: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """Execute tool calls from the LLM response.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            Tuple of (memory_actions, input_action, execution_results)

        Raises:
            AgentError: If tool execution violates constraints
        """
        memory_actions: List[Dict[str, Any]] = []
        input_action: Optional[Dict[str, Any]] = None
        execution_results: List[str] = []

        # Separate memory and input tools
        memory_calls = [tc for tc in tool_calls if tc["name"] in MEMORY_TOOL_NAMES]
        input_calls = [tc for tc in tool_calls if tc["name"] in INPUT_TOOL_NAMES]

        # Validate: at most one input action
        if len(input_calls) > 1:
            raise AgentError(f"Agent attempted {len(input_calls)} input actions; only 1 allowed per turn")

        # Execute memory tools
        for tool_call in memory_calls:
            try:
                result = self._execute_memory_tool(tool_call)
                memory_actions.append(tool_call)
                execution_results.append(result)
            except Exception as exc:
                error_msg = f"Memory tool {tool_call['name']} failed: {exc}"
                self._logger.error(error_msg)
                execution_results.append(error_msg)

        # Store input action (will be executed by caller with input_handler)
        if input_calls:
            input_action = input_calls[0]
            execution_results.append(f"Input action queued: {input_action['name']}")

        return memory_actions, input_action, execution_results

    def _execute_memory_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute a memory management tool.

        Args:
            tool_call: Tool call dictionary with name and arguments

        Returns:
            Success message

        Raises:
            MemoryError: If the memory operation fails
        """
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        if name == "createMemoryFile":
            file_name = args["name"]
            self._memory.create_file(file_name)
            return f"Created memory file: {file_name}"

        elif name == "writeMemoryFile":
            file_name = args["name"]
            content = args["content"]
            self._memory.write_file(file_name, content)
            tokens = self._memory.total_tokens()
            return f"Wrote to {file_name} (total memory: {tokens} tokens)"

        elif name == "deleteMemoryFile":
            file_name = args["name"]
            self._memory.delete_file(file_name)
            return f"Deleted memory file: {file_name}"

        else:
            raise AgentError(f"Unknown memory tool: {name}")

    def _init_api(self) -> None:
        """Initialize Anthropic API client."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self._client = Anthropic(api_key=api_key)
        thinking_status = "enabled" if self._thinking_enabled else "disabled"
        self._logger.info(
            "Initialized Anthropic API with model: %s (thinking: %s, budget: %d tokens)",
            self._model,
            thinking_status,
            self._thinking_budget_tokens,
        )

    def _encode_image(self, image_path: Path) -> Dict[str, Any]:
        """Encode image to Anthropic's image format.

        Args:
            image_path: Path to image file

        Returns:
            Image content dict with type, source, and media_type
        """
        with Image.open(image_path) as img:
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                }

    def _extract_reasoning(self, response: Any) -> tuple[str, Optional[str]]:
        """Extract reasoning text and thinking blocks from Anthropic response.

        Args:
            response: Anthropic Message object

        Returns:
            Tuple of (reasoning_text, thinking_text)
        """
        try:
            reasoning_parts = []
            thinking_parts = []

            for block in response.content:
                if block.type == "text":
                    reasoning_parts.append(block.text)
                elif block.type == "thinking":
                    thinking_parts.append(block.thinking)
                    self._logger.debug("Extended thinking: %s", block.thinking[:200] + "...")

            reasoning = "\n".join(reasoning_parts).strip() if reasoning_parts else "(No reasoning provided)"
            thinking = "\n".join(thinking_parts).strip() if thinking_parts else None

            return reasoning, thinking
        except Exception as exc:
            self._logger.warning("Failed to extract reasoning: %s", exc)
            return "(Failed to extract reasoning)", None

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Anthropic response.

        Args:
            response: Anthropic Message object

        Returns:
            List of tool call dictionaries with 'name' and 'arguments' keys
        """
        try:
            calls = []
            for block in response.content:
                if block.type == "tool_use":
                    calls.append({
                        "name": block.name,
                        "arguments": block.input,
                    })

            return calls

        except Exception as exc:
            self._logger.warning("Failed to extract tool calls: %s", exc)
            return []

    def _format_compact_turn_context(
        self,
        turn_id: str,
        timestamp: str,
        vision_data: VisionData,
    ) -> str:
        """Format a compact turn summary for message history (not current turn).

        This creates a minimal text representation that preserves context continuity
        without the full vision JSON overhead. Used when storing turns in history.

        Example output:
            # Turn turn_000005
            Timestamp: 2025-11-01T18:15:22.123456

            Buttons: Archive, Profile, Titles, Friends, Options, Info button, Back button
            Menu: tab=Menu, available=['Jukebox', 'Menu']
            Scrollbar: none

        Args:
            turn_id: Turn identifier
            timestamp: Turn timestamp
            vision_data: Vision data for the turn

        Returns:
            Compact text summary (~150-200 tokens vs ~8,500 for full format)
        """
        lines = [f"# Turn {turn_id}", f"Timestamp: {timestamp}", ""]

        # Button names only (no bounds, metadata, or full labels)
        button_names = [btn["name"] for btn in vision_data.buttons]
        if button_names:
            # Format as comma-separated list, breaking into multiple lines if very long
            buttons_text = ", ".join(button_names)
            if len(buttons_text) > 200:
                # Break into chunks for readability
                chunks = []
                current_chunk = []
                current_length = 0
                for name in button_names:
                    if current_length + len(name) + 2 > 200 and current_chunk:
                        chunks.append(", ".join(current_chunk))
                        current_chunk = [name]
                        current_length = len(name)
                    else:
                        current_chunk.append(name)
                        current_length += len(name) + 2
                if current_chunk:
                    chunks.append(", ".join(current_chunk))
                buttons_text = "\n  ".join(chunks)
                lines.append(f"Buttons:\n  {buttons_text}")
            else:
                lines.append(f"Buttons: {buttons_text}")
        else:
            lines.append("Buttons: (none detected)")

        # Menu state (minimal)
        menu = vision_data.menu_state
        if menu.get("tab") or menu.get("available"):
            tab = menu.get("tab", "none")
            available = menu.get("available", [])
            lines.append(f"Menu: tab={tab}, available={available}")

        # Scrollbar (yes/no only)
        if vision_data.scrollbar and (vision_data.scrollbar.get("up") or vision_data.scrollbar.get("down")):
            lines.append("Scrollbar: present")
        else:
            lines.append("Scrollbar: none")

        return "\n".join(lines)

    def _format_turn_summary(
        self,
        turn_id: str,
        reasoning: str,
        memory_actions: List[Dict[str, Any]],
        input_action: Optional[Dict[str, Any]],
    ) -> str:
        """Format a compact summary of a turn for logging.

        Args:
            turn_id: Turn identifier
            reasoning: Agent's reasoning text
            memory_actions: List of memory tool calls
            input_action: Input action (if any)

        Returns:
            Markdown summary string
        """
        lines = [f"## Turn {turn_id}"]

        # Truncate reasoning if very long
        if len(reasoning) > 500:
            reasoning = reasoning[:500] + "..."
        lines.append(f"**Reasoning:** {reasoning}")

        # Memory actions (don't include full content)
        if memory_actions:
            action_names = [f"{a['name']}({a['arguments'].get('name', '')})" for a in memory_actions]
            lines.append(f"**Memory:** {', '.join(action_names)}")

        # Input action
        if input_action:
            args = input_action.get("arguments", {})
            if args:
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                lines.append(f"**Action:** {input_action['name']}({args_str})")
            else:
                lines.append(f"**Action:** {input_action['name']}()")

        lines.append("")
        return "\n".join(lines)

    def _estimate_prompt_size(self) -> int:
        """Estimate the size of the current message history in tokens.

        Uses a simple heuristic: count characters and divide by 4 (rough approximation).

        Returns:
            Estimated token count for the message history
        """
        if not self._message_history:
            return 0

        # Count total characters in message history
        total_chars = 0
        for message in self._message_history:
            content = message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total_chars += len(block["text"])
                    elif isinstance(block, dict) and "type" in block and block["type"] == "text":
                        total_chars += len(block.get("text", ""))

        # Rough approximation: 1 token ≈ 4 characters
        estimated_tokens = total_chars // 4
        return estimated_tokens

    def _should_summarize(self, estimated_prompt_tokens: int) -> bool:
        """Check if we should trigger summarization based on prompt size.

        Args:
            estimated_prompt_tokens: Estimated size of the current prompt in tokens

        Returns:
            True if estimated prompt size exceeds threshold
        """
        return (
            self._summarization_threshold_tokens > 0
            and estimated_prompt_tokens >= self._summarization_threshold_tokens
        )

    def _perform_summarization(self) -> None:
        """Summarize the entire message history and replace it with a compact summary.

        This is called when cumulative input tokens exceed the threshold to prevent
        context overflow. The agent is asked to summarize all its experiences,
        then the message history is replaced with a single synthetic message.
        """
        self._logger.info("Performing context summarization (current history: %d messages)",
                          len(self._message_history))

        if not self._message_history:
            self._logger.warning("Message history is empty, skipping summarization")
            return

        # Build summarization prompt
        summarization_prompt = """You have been playing Uma Musume for a while, and your message history is getting long.

Please provide a comprehensive summary of everything you've learned and accomplished so far. Include:

1. **Game Progress**: What have you done? What screens/features have you explored?
2. **UI Knowledge**: What UI patterns have you learned? What buttons appear where?
3. **Current State**: Where are you now in the game? What were you working on?
4. **Key Discoveries**: Any important observations about game mechanics or navigation?

Be concise but thorough. This summary will replace your entire message history, so include everything important.
Focus on factual observations and progress, not speculation."""

        try:
            # Call the model with the existing history + summarization request
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=AGENT_SYSTEM_PROMPT,
                messages=self._message_history + [
                    {"role": "user", "content": [{"type": "text", "text": summarization_prompt}]}
                ],
            )

            # Extract the summary text
            summary_parts = []
            for block in response.content:
                if block.type == "text":
                    summary_parts.append(block.text)

            if not summary_parts:
                self._logger.error("Summarization produced no text, keeping existing history")
                return

            summary = "\n".join(summary_parts)
            self._logger.info("Generated summary (%d characters)", len(summary))

            # Replace message history with synthetic summary message
            old_message_count = len(self._message_history)
            self._message_history = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""# Historical Context Summary

The following is a summary of your experience playing Uma Musume so far.
Your full message history has been condensed to save context.

{summary}

---

New turns will continue below this summary."""
                        }
                    ]
                }
            ]

            self._logger.info(
                "Summarization complete: %d messages → 1 summary message",
                old_message_count
            )

        except Exception as exc:
            self._logger.error("Summarization failed: %s. Keeping existing history.", exc)


__all__ = ["UmaAgent", "AgentError", "VisionData", "TurnResult"]
