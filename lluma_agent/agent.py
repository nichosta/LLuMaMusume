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

import requests
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
    memory_actions: List[Dict[str, Any]]
    input_action: Optional[Dict[str, Any]]
    execution_results: List[str]
    timestamp: str


class UmaAgent:
    """Main agent that coordinates LLM reasoning and tool execution."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        # model: str = "google/gemini-flash-2.5-latest",
        model: str = "anthropic/claude-haiku-4.5",
        max_context_tokens: int = 32000,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            memory_manager: Memory file manager instance
            model: LLM model identifier for OpenRouter
            max_context_tokens: Maximum context window size for turn history
            logger: Optional logger instance
        """
        self._memory = memory_manager
        self._logger = logger or logging.getLogger(__name__)
        self._api_key: str = ""
        self._headers: Dict[str, str] = {}
        self._turn_history: List[str] = []
        self._model = model
        self._max_context_tokens = max_context_tokens

        # Initialize OpenRouter API configuration
        self._init_api()

    def execute_turn(
        self,
        turn_id: str,
        vision_data: VisionData,
        primary_screenshot: Path,
        menus_screenshot: Optional[Path] = None,
    ) -> TurnResult:
        """Execute a single turn: get LLM decision and execute tools.

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

        # Build context message
        user_message = self._format_context(turn_id, timestamp, vision_data)

        # Add screenshots
        message_content = [
            {"type": "text", "text": user_message},
            {
                "type": "image_url",
                "image_url": {"url": self._encode_image(primary_screenshot)},
            },
        ]

        if menus_screenshot is not None and menus_screenshot.exists():
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(menus_screenshot)},
                }
            )

        # Call LLM
        try:
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": message_content},
                ],
                "tools": ALL_TOOLS,
                "tool_choice": "auto",
            }

            self._logger.info("Calling OpenRouter agent model %s for turn %s", self._model, turn_id)
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            response_data = response.json()
        except Exception as exc:
            raise AgentError(f"LLM request failed: {exc}") from exc

        # Extract reasoning and tool calls
        reasoning = self._extract_reasoning(response_data)
        tool_calls = self._extract_tool_calls(response_data)

        # Execute tools
        memory_actions, input_action, execution_results = self._execute_tools(tool_calls)

        # Record turn in history (for context window management)
        turn_summary = self._format_turn_summary(turn_id, reasoning, memory_actions, input_action)
        self._turn_history.append(turn_summary)

        # TODO: Implement context summarization when history exceeds max_context_tokens

        return TurnResult(
            turn_id=turn_id,
            reasoning=reasoning,
            memory_actions=memory_actions,
            input_action=input_action,
            execution_results=execution_results,
            timestamp=timestamp,
        )

    def _format_context(self, turn_id: str, timestamp: str, vision_data: VisionData) -> str:
        """Format the context message for the LLM.

        Combines turn metadata, turn history, vision data, and memory files.
        """
        sections = []

        # Turn metadata
        sections.append(f"# Turn {turn_id}\n\nTimestamp: {timestamp}\n")

        # Turn history (recent reasoning and actions)
        if self._turn_history:
            sections.append("# Previous Turns\n")
            # Show last 5 turns (TODO: implement proper summarization)
            recent = self._turn_history[-5:]
            sections.append("\n".join(recent))
            sections.append("")

        # Vision data (JSON)
        vision_json = {
            "buttons": vision_data.buttons,
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
        """Initialize OpenRouter API configuration."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/nichosta/LLuMaMusume",
            "X-Title": "LLuMa Musume Agent",
            "Content-Type": "application/json",
        }
        self._logger.info("Initialized OpenRouter API with model: %s", self._model)

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 data URL.

        Args:
            image_path: Path to image file

        Returns:
            Data URL string
        """
        with Image.open(image_path) as img:
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{b64}"

    def _extract_reasoning(self, response: Dict[str, Any]) -> str:
        """Extract reasoning text from LLM response.

        Args:
            response: OpenRouter API response dict

        Returns:
            Reasoning text
        """
        try:
            message = response["choices"][0]["message"]
            content = message.get("content")
            return content.strip() if content else "(No reasoning provided)"
        except Exception as exc:
            self._logger.warning("Failed to extract reasoning: %s", exc)
            return "(Failed to extract reasoning)"

    def _extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response.

        Args:
            response: OpenRouter API response dict

        Returns:
            List of tool call dictionaries with 'name' and 'arguments' keys
        """
        try:
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return []

            calls = []
            for tc in tool_calls:
                function = tc["function"]
                name = function["name"]
                raw_arguments = function.get("arguments", "")

                if raw_arguments is None:
                    arguments = {}
                elif isinstance(raw_arguments, str):
                    args_str = raw_arguments.strip()
                    if args_str:
                        try:
                            arguments = json.loads(args_str)
                        except json.JSONDecodeError:
                            self._logger.warning("Failed to parse tool arguments for %s: %s", name, args_str)
                            arguments = {}
                    else:
                        arguments = {}
                elif isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    self._logger.warning("Unexpected arguments payload for %s: %r", name, raw_arguments)
                    arguments = {}

                calls.append({"name": name, "arguments": arguments})

            return calls

        except Exception as exc:
            self._logger.warning("Failed to extract tool calls: %s", exc)
            return []

    def _format_turn_summary(
        self,
        turn_id: str,
        reasoning: str,
        memory_actions: List[Dict[str, Any]],
        input_action: Optional[Dict[str, Any]],
    ) -> str:
        """Format a compact summary of a turn for history.

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


__all__ = ["UmaAgent", "AgentError", "VisionData", "TurnResult"]
