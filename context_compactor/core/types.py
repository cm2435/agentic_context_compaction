"""SDK-specific type aliases for message types.

These type aliases define the union of message types for each supported SDK.
They are used for type hints in strategies and token counters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pydantic-ai message types
    from pydantic_ai.messages import ModelRequest, ModelResponse

    PydanticAIMessage = ModelRequest | ModelResponse
    """Union of pydantic-ai message types: ModelRequest | ModelResponse"""

    # openai-agents message types
    from agents.items import (
        HandoffCallItem,
        HandoffOutputItem,
        MCPApprovalRequestItem,
        MCPApprovalResponseItem,
        MCPListToolsItem,
        MessageOutputItem,
        ReasoningItem,
        ToolCallItem,
        ToolCallOutputItem,
        TResponseInputItem,
    )

    OpenAIAgentsItem = (
        MessageOutputItem
        | ToolCallItem
        | ToolCallOutputItem
        | ReasoningItem
        | HandoffCallItem
        | HandoffOutputItem
        | MCPListToolsItem
        | MCPApprovalRequestItem
        | MCPApprovalResponseItem
    )
    """Union of openai-agents RunItem types."""

    OpenAIAgentsMessage = OpenAIAgentsItem | TResponseInputItem
    """Union of openai-agents message types: RunItem variants | input items"""

    # claude-agent-sdk message types
    from claude_agent_sdk.types import AssistantMessage, UserMessage

    ClaudeAgentMessage = UserMessage | AssistantMessage
    """Union of claude-agent-sdk message types: UserMessage | AssistantMessage"""

else:
    # Runtime placeholders - actual types are only used for static analysis
    PydanticAIMessage = object
    OpenAIAgentsItem = object
    OpenAIAgentsMessage = object
    ClaudeAgentMessage = object


__all__ = [
    "PydanticAIMessage",
    "OpenAIAgentsItem",
    "OpenAIAgentsMessage",
    "ClaudeAgentMessage",
]
