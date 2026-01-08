"""Pydantic-ai specific compaction strategies.

These strategies understand pydantic-ai message types and can
construct proper SDK message types for summaries.

Requires: pip install context-compactor[pydantic-ai]
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

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

from ..core.protocols import TokenCounter

# Type alias
PydanticAIMessage = ModelRequest | ModelResponse

# Summarizer function type
Summarizer = Callable[[list[PydanticAIMessage]], Awaitable[str]]


@dataclass
class SummarizeMiddle:
    """
    Keep first/last messages, summarize middle with an LLM.

    Preserves the initial context and recent messages, while
    summarizing the middle portion into a compact form.

    Args:
        keep_first: Messages to keep from start (default 2)
        keep_last: Messages to keep from end (default 5)
        summarizer: Async function to summarize messages into text

    Example:
        ```python
        async def my_summarizer(messages: list[PydanticAIMessage]) -> str:
            # Use an LLM to summarize
            return await summarize_with_llm(messages)

        strategy = SummarizeMiddle(
            keep_first=2,
            keep_last=5,
            summarizer=my_summarizer,
        )
        ```
    """

    summarizer: Summarizer
    keep_first: int = 2
    keep_last: int = 5

    async def compact(
        self,
        messages: list[PydanticAIMessage],
        target_tokens: int,
        token_counter: TokenCounter[PydanticAIMessage],
    ) -> list[PydanticAIMessage]:
        """Keep first/last messages, summarize the middle."""
        total_keep = self.keep_first + self.keep_last

        if len(messages) <= total_keep:
            return messages

        first = messages[: self.keep_first]
        middle = messages[self.keep_first : -self.keep_last]
        last = messages[-self.keep_last :]

        if not middle:
            return first + last

        # Summarize middle messages
        summary_text = await self.summarizer(middle)

        # Create a proper SDK message for the summary
        summary_msg = ModelRequest(
            parts=[
                UserPromptPart(
                    content=f"[Summarized {len(middle)} previous messages]: {summary_text}"
                )
            ]
        )

        return first + [summary_msg] + last


@dataclass
class KeepToolCalls:
    """
    Keep all tool calls and their results, drop other content.

    Useful for preserving the action history while reducing
    conversational content.

    Args:
        keep_system_prompts: Also keep system prompts (default True)
        keep_recent_text: Number of recent text messages to keep (default 2)

    Example:
        ```python
        strategy = KeepToolCalls(keep_system_prompts=True, keep_recent_text=3)
        ```
    """

    keep_system_prompts: bool = True
    keep_recent_text: int = 2

    async def compact(
        self,
        messages: list[PydanticAIMessage],
        target_tokens: int,
        token_counter: TokenCounter[PydanticAIMessage],
    ) -> list[PydanticAIMessage]:
        """Keep tool calls and optionally system prompts."""
        result: list[PydanticAIMessage] = []
        recent_text: list[PydanticAIMessage] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                # Check for tool returns or system prompts
                has_tool_return = any(isinstance(p, ToolReturnPart) for p in msg.parts)
                has_system = any(isinstance(p, SystemPromptPart) for p in msg.parts)

                if has_tool_return:
                    result.append(msg)
                elif has_system and self.keep_system_prompts:
                    # Keep only system prompt parts
                    system_parts = [p for p in msg.parts if isinstance(p, SystemPromptPart)]
                    if system_parts:
                        result.append(ModelRequest(parts=system_parts))
                else:
                    # Track as potential recent text
                    recent_text.append(msg)

            elif isinstance(msg, ModelResponse):
                # Check for tool calls
                has_tool_call = any(isinstance(p, ToolCallPart) for p in msg.parts)

                if has_tool_call:
                    # Keep tool calls and any text that came with them
                    keep_parts = [
                        p
                        for p in msg.parts
                        if isinstance(p, TextPart | ToolCallPart)
                        and not isinstance(p, ThinkingPart)
                    ]
                    if keep_parts:
                        result.append(ModelResponse(parts=keep_parts))
                else:
                    # Track as potential recent text
                    recent_text.append(msg)

        # Add recent text messages
        if self.keep_recent_text > 0:
            result.extend(recent_text[-self.keep_recent_text :])

        return result


@dataclass
class DropThinking:
    """
    Remove ThinkingPart from all messages.

    Thinking blocks can be large and are often not needed for
    context continuity.

    Example:
        ```python
        strategy = DropThinking()
        compacted = await strategy.compact(messages, target_tokens, counter)
        ```
    """

    async def compact(
        self,
        messages: list[PydanticAIMessage],
        target_tokens: int,
        token_counter: TokenCounter[PydanticAIMessage],
    ) -> list[PydanticAIMessage]:
        """Remove thinking blocks from all messages."""
        result: list[PydanticAIMessage] = []

        for msg in messages:
            if isinstance(msg, ModelResponse):
                # Filter out thinking parts
                filtered_parts = [p for p in msg.parts if not isinstance(p, ThinkingPart)]
                if filtered_parts:
                    result.append(ModelResponse(parts=filtered_parts))
            else:
                result.append(msg)

        return result


@dataclass
class ChainedStrategy:
    """
    Apply multiple strategies in sequence.

    Useful for combining strategies like DropThinking + KeepFirstLast.

    Args:
        strategies: List of strategies to apply in order

    Example:
        ```python
        strategy = ChainedStrategy(strategies=[
            DropThinking(),
            KeepFirstLast(keep_first=2, keep_last=10),
        ])
        ```
    """

    strategies: list

    async def compact(
        self,
        messages: list[PydanticAIMessage],
        target_tokens: int,
        token_counter: TokenCounter[PydanticAIMessage],
    ) -> list[PydanticAIMessage]:
        """Apply each strategy in sequence."""
        result = messages
        for strategy in self.strategies:
            result = await strategy.compact(result, target_tokens, token_counter)
        return result
