"""OpenAI Agents SDK message fixtures for testing.

Requires: pip install openai-agents
"""

from __future__ import annotations

import pytest

try:
    from agents.items import MessageOutputItem
    from openai.types.responses import (
        ResponseOutputMessage,
        ResponseOutputText,
    )

    from context_compactor.tokenizers.openai_agents import OpenAIAgentsTokenCounter

    OPENAI_AGENTS_AVAILABLE = True
except ImportError:
    OPENAI_AGENTS_AVAILABLE = False

# Skip all fixtures if openai-agents is not installed
pytestmark = pytest.mark.skipif(not OPENAI_AGENTS_AVAILABLE, reason="openai-agents not installed")


# =============================================================================
# Helper functions to create proper message objects
# =============================================================================


def _make_message_output(text: str, item_id: str = "msg_1") -> MessageOutputItem:
    """Create a MessageOutputItem with text content."""
    raw = ResponseOutputMessage(
        id=item_id,
        type="message",
        role="assistant",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        status="completed",
    )
    return MessageOutputItem(raw_item=raw, agent=None)


def _make_user_input(text: str, item_id: str = "input_1") -> dict:
    """Create a user input item (dict format for TResponseInputItem)."""
    return {
        "id": item_id,
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def _make_system_input(text: str, item_id: str = "sys_1") -> dict:
    """Create a system input item."""
    return {
        "id": item_id,
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": text}],
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def openai_agents_counter():
    """Provide an OpenAIAgentsTokenCounter."""
    return OpenAIAgentsTokenCounter()


@pytest.fixture
def openai_agents_sample_messages() -> list:
    """Basic openai-agents conversation (~10 messages)."""
    return [
        _make_system_input("You are a helpful assistant.", "sys_1"),
        _make_user_input("Hello, how are you?", "user_1"),
        _make_message_output("I'm doing well, thank you!", "msg_1"),
        _make_user_input("Can you help me with Python?", "user_2"),
        _make_message_output("Of course! I'd be happy to help.", "msg_2"),
        _make_user_input("How do I read a file?", "user_3"),
        _make_message_output("To read a file, use open()...", "msg_3"),
        _make_user_input("What about writing?", "user_4"),
        _make_message_output("For writing, use open('w')...", "msg_4"),
        _make_user_input("Thanks!", "user_5"),
        _make_message_output("You're welcome!", "msg_5"),
    ]


@pytest.fixture
def openai_agents_long_messages(openai_agents_sample_messages) -> list:
    """Extended openai-agents message history (~50+ messages)."""
    messages = openai_agents_sample_messages.copy()
    for i in range(20):
        messages.append(_make_user_input(f"Question {i}: How does feature X work?", f"user_q{i}"))
        messages.append(
            _make_message_output(f"Answer {i}: Feature X works by doing Y and Z.", f"msg_a{i}")
        )
    return messages


@pytest.fixture
def openai_agents_mixed_messages() -> list:
    """
    Mixed openai-agents messages with various item types.

    Includes: text messages, tool calls, tool outputs, reasoning.
    Note: Some types require more complex construction.
    """
    messages = []

    # System message
    messages.append(_make_system_input("You are a coding assistant.", "sys_1"))

    # User asks a question
    messages.append(_make_user_input("What's in main.py?", "user_1"))

    # Assistant response with explanation
    messages.append(_make_message_output("Let me check that file for you.", "msg_1"))

    # Note: Creating full ToolCallItem requires more OpenAI SDK setup
    # For testing purposes, we use a mix of dicts and MessageOutputItems

    # User follow-up
    messages.append(_make_user_input("Can you add error handling?", "user_2"))

    # Assistant response
    messages.append(_make_message_output("I'll add try/except blocks.", "msg_2"))

    # Another user question
    messages.append(_make_user_input("Does this look right?", "user_3"))

    # Final assistant response
    messages.append(_make_message_output("Yes, the output looks correct now!", "msg_3"))

    return messages
