"""Pydantic-AI message fixtures for testing.

Requires: pip install pydantic-ai
"""

from __future__ import annotations

import pytest

try:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

# Skip all fixtures if pydantic-ai is not installed
pytestmark = pytest.mark.skipif(not PYDANTIC_AI_AVAILABLE, reason="pydantic-ai not installed")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pydantic_ai_counter():
    """Provide a PydanticAITokenCounter."""
    return PydanticAITokenCounter()


@pytest.fixture
def pydantic_ai_sample_messages() -> list:
    """Basic pydantic-ai conversation (~10 messages)."""
    return [
        ModelRequest(parts=[SystemPromptPart(content="You are a helpful assistant.")]),
        ModelRequest(parts=[UserPromptPart(content="Hello, how are you?")]),
        ModelResponse(parts=[TextPart(content="I'm doing well, thank you!")]),
        ModelRequest(parts=[UserPromptPart(content="Can you help me with Python?")]),
        ModelResponse(parts=[TextPart(content="Of course! I'd be happy to help.")]),
        ModelRequest(parts=[UserPromptPart(content="How do I read a file?")]),
        ModelResponse(parts=[TextPart(content="To read a file, use open()...")]),
        ModelRequest(parts=[UserPromptPart(content="What about writing?")]),
        ModelResponse(parts=[TextPart(content="For writing, use open('w')...")]),
        ModelRequest(parts=[UserPromptPart(content="Thanks!")]),
        ModelResponse(parts=[TextPart(content="You're welcome!")]),
    ]


@pytest.fixture
def pydantic_ai_long_messages(pydantic_ai_sample_messages) -> list:
    """Extended pydantic-ai message history (~50+ messages)."""
    messages = pydantic_ai_sample_messages.copy()
    for i in range(20):
        messages.append(
            ModelRequest(parts=[UserPromptPart(content=f"Question {i}: How does feature X work?")])
        )
        messages.append(
            ModelResponse(
                parts=[TextPart(content=f"Answer {i}: Feature X works by doing Y and Z.")]
            )
        )
    return messages


@pytest.fixture
def pydantic_ai_mixed_messages() -> list:
    """
    Mixed pydantic-ai messages with various part types.

    Includes: text, tool calls, tool returns, thinking blocks.
    """
    return [
        # System prompt
        ModelRequest(parts=[SystemPromptPart(content="You are a coding assistant.")]),
        # User with text
        ModelRequest(parts=[UserPromptPart(content="What's in main.py?")]),
        # Assistant with tool call
        ModelResponse(
            parts=[
                TextPart(content="Let me check that file for you."),
                ToolCallPart(
                    tool_name="read_file",
                    args={"path": "main.py"},
                    tool_call_id="call_1",
                ),
            ]
        ),
        # Tool result
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="read_file",
                    content="def main():\n    print('hello')",
                    tool_call_id="call_1",
                )
            ]
        ),
        # Assistant response with thinking
        ModelResponse(
            parts=[
                ThinkingPart(content="Analyzing the code structure..."),
                TextPart(content="The main.py file contains a simple main function."),
            ]
        ),
        # More user interaction
        ModelRequest(parts=[UserPromptPart(content="Can you add error handling?")]),
        # Another tool call
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="edit_file",
                    args={"path": "main.py", "content": "..."},
                    tool_call_id="call_2",
                ),
            ]
        ),
        # Tool result
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="edit_file",
                    content="File updated successfully",
                    tool_call_id="call_2",
                )
            ]
        ),
        # Final response
        ModelResponse(parts=[TextPart(content="Done! I've added try/except blocks.")]),
    ]
