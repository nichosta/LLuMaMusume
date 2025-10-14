"""Tool definitions and schemas for OpenAI function calling."""
from typing import Any, Dict, List

# Memory management tools
TOOL_CREATE_MEMORY_FILE = {
    "type": "function",
    "function": {
        "name": "createMemoryFile",
        "description": "Create a new empty memory file. The file must not already exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the memory file to create (e.g., 'player.yaml', 'todo.md'). Must not contain path separators.",
                },
            },
            "required": ["name"],
        },
    },
}

TOOL_WRITE_MEMORY_FILE = {
    "type": "function",
    "function": {
        "name": "writeMemoryFile",
        "description": "Write content to an existing memory file, overwriting its current contents. The file must already exist (use createMemoryFile first if needed).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the memory file to write to.",
                },
                "content": {
                    "type": "string",
                    "description": "New content for the file. This will completely replace existing content.",
                },
            },
            "required": ["name", "content"],
        },
    },
}

TOOL_DELETE_MEMORY_FILE = {
    "type": "function",
    "function": {
        "name": "deleteMemoryFile",
        "description": "Delete a memory file permanently. The file must exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the memory file to delete.",
                },
            },
            "required": ["name"],
        },
    },
}

# Input action tools
TOOL_PRESS_BUTTON = {
    "type": "function",
    "function": {
        "name": "pressButton",
        "description": "Click the center of a button detected by Vision. The button name must exactly match a name from the current turn's vision data.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the button to click. Must match a button name from vision data.",
                },
            },
            "required": ["name"],
        },
    },
}

TOOL_ADVANCE_DIALOGUE = {
    "type": "function",
    "function": {
        "name": "advanceDialogue",
        "description": "Click the center of the primary region to advance dialogue or confirm simple prompts. Use this for visual novel-style text advancement.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

TOOL_BACK = {
    "type": "function",
    "function": {
        "name": "back",
        "description": "Press ESC to navigate back or dismiss dialogs. Use this to exit menus or cancel actions.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

TOOL_CONFIRM = {
    "type": "function",
    "function": {
        "name": "confirm",
        "description": "Press SPACE to confirm actions. Use this when the game expects a confirmation.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

TOOL_SCROLL_UP = {
    "type": "function",
    "function": {
        "name": "scrollUp",
        "description": "Press Z to scroll up in lists or menus. Requires that a scrollbar was detected in the current turn's vision data.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

TOOL_SCROLL_DOWN = {
    "type": "function",
    "function": {
        "name": "scrollDown",
        "description": "Press C to scroll down in lists or menus. Requires that a scrollbar was detected in the current turn's vision data.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

# Tool collections
MEMORY_TOOLS = [
    TOOL_CREATE_MEMORY_FILE,
    TOOL_WRITE_MEMORY_FILE,
    TOOL_DELETE_MEMORY_FILE,
]

INPUT_TOOLS = [
    TOOL_PRESS_BUTTON,
    TOOL_ADVANCE_DIALOGUE,
    TOOL_BACK,
    TOOL_CONFIRM,
    TOOL_SCROLL_UP,
    TOOL_SCROLL_DOWN,
]

ALL_TOOLS = MEMORY_TOOLS + INPUT_TOOLS

# Tool name sets for validation
MEMORY_TOOL_NAMES = {
    "createMemoryFile",
    "writeMemoryFile",
    "deleteMemoryFile",
}

INPUT_TOOL_NAMES = {
    "pressButton",
    "advanceDialogue",
    "back",
    "confirm",
    "scrollUp",
    "scrollDown",
}


def get_tool_by_name(name: str) -> Dict[str, Any]:
    """Get tool schema by name.

    Args:
        name: Tool function name

    Returns:
        Tool schema dictionary

    Raises:
        ValueError: If tool name is not recognized
    """
    for tool in ALL_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    raise ValueError(f"Unknown tool name: {name}")


__all__ = [
    "ALL_TOOLS",
    "INPUT_TOOLS",
    "INPUT_TOOL_NAMES",
    "MEMORY_TOOLS",
    "MEMORY_TOOL_NAMES",
    "get_tool_by_name",
]
