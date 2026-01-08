"""Claude Agent SDK message fixtures for testing.

Requires: pip install claude-agent-sdk
"""

from __future__ import annotations

import pytest

try:
    from claude_agent_sdk.types import (
        AssistantMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from context_compactor.tokenizers.claude_agent import ClaudeAgentTokenCounter

    CLAUDE_AGENT_AVAILABLE = True
except ImportError:
    CLAUDE_AGENT_AVAILABLE = False

# Skip all fixtures if claude-agent-sdk is not installed
pytestmark = pytest.mark.skipif(not CLAUDE_AGENT_AVAILABLE, reason="claude-agent-sdk not installed")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def claude_agent_counter():
    """Provide a ClaudeAgentTokenCounter."""
    return ClaudeAgentTokenCounter()


@pytest.fixture
def claude_agent_sample_messages() -> list:
    """Basic claude-agent-sdk conversation (~10 messages)."""
    return [
        UserMessage(content="Hello, how are you?"),
        AssistantMessage(content=[TextBlock(type="text", text="I'm doing well, thank you!")]),
        UserMessage(content="Can you help me with Python?"),
        AssistantMessage(content=[TextBlock(type="text", text="Of course! I'd be happy to help.")]),
        UserMessage(content="How do I read a file?"),
        AssistantMessage(content=[TextBlock(type="text", text="To read a file, use open()...")]),
        UserMessage(content="What about writing?"),
        AssistantMessage(content=[TextBlock(type="text", text="For writing, use open('w')...")]),
        UserMessage(content="Thanks!"),
        AssistantMessage(content=[TextBlock(type="text", text="You're welcome!")]),
    ]


@pytest.fixture
def claude_agent_long_messages(claude_agent_sample_messages) -> list:
    """Extended claude-agent-sdk message history (~50+ messages)."""
    messages = claude_agent_sample_messages.copy()
    for i in range(20):
        messages.append(UserMessage(content=f"Question {i}: How does feature X work?"))
        messages.append(
            AssistantMessage(
                content=[
                    TextBlock(
                        type="text",
                        text=f"Answer {i}: Feature X works by doing Y and Z.",
                    )
                ]
            )
        )
    return messages


@pytest.fixture
def claude_agent_mixed_messages() -> list:
    """
    Mixed claude-agent-sdk messages with various block types.

    Includes: text, tool use, tool results, thinking blocks.
    """
    return [
        # User asks a question
        UserMessage(content="What's in main.py?"),
        # Assistant with thinking and tool use
        AssistantMessage(
            content=[
                ThinkingBlock(type="thinking", thinking="I need to read the file first..."),
                TextBlock(type="text", text="Let me check that file for you."),
                ToolUseBlock(
                    type="tool_use",
                    id="tool_1",
                    name="read_file",
                    input={"path": "main.py"},
                ),
            ]
        ),
        # User provides tool result
        UserMessage(
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="tool_1",
                    content="def main():\n    print('hello')",
                )
            ]
        ),
        # Assistant analyzes
        AssistantMessage(
            content=[
                TextBlock(
                    type="text",
                    text="The main.py file contains a simple main function.",
                )
            ]
        ),
        # User follow-up
        UserMessage(content="Can you add error handling?"),
        # Assistant with tool use
        AssistantMessage(
            content=[
                TextBlock(type="text", text="I'll add try/except blocks."),
                ToolUseBlock(
                    type="tool_use",
                    id="tool_2",
                    name="edit_file",
                    input={"path": "main.py", "content": "..."},
                ),
            ]
        ),
        # Tool result
        UserMessage(
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id="tool_2",
                    content="File updated successfully",
                )
            ]
        ),
        # Final response
        AssistantMessage(
            content=[TextBlock(type="text", text="Done! I've added error handling to main.py.")]
        ),
    ]
