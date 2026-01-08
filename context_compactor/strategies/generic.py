"""Generic compaction strategies that work with any SDK message type.

These strategies operate via the TokenCounter protocol and don't need
to understand the internal structure of messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from ..core.protocols import TokenCounter

MessageT = TypeVar("MessageT")


@dataclass
class KeepRecentMessages(Generic[MessageT]):
    """
    Keep only the N most recent messages.

    Simple strategy that discards older messages to fit within token budget.
    Works with any SDK message type.

    Args:
        keep_count: Number of recent messages to keep (default 10)

    Example:
        ```python
        strategy = KeepRecentMessages[PydanticAIMessage](keep_count=20)
        compacted = await strategy.compact(messages, target_tokens, counter)
        ```
    """

    keep_count: int = 10

    async def compact(
        self,
        messages: list[MessageT],
        target_tokens: int,
        token_counter: TokenCounter[MessageT],
    ) -> list[MessageT]:
        """Keep only the most recent messages."""
        if self.keep_count <= 0:
            return []
        if len(messages) <= self.keep_count:
            return messages
        return messages[-self.keep_count :]


@dataclass
class DropOldestUntilFits(Generic[MessageT]):
    """
    Drop messages from the start until under token budget.

    Iteratively removes the oldest messages until the total token
    count is under the target.

    Args:
        min_messages: Minimum messages to keep (default 2)

    Example:
        ```python
        strategy = DropOldestUntilFits[PydanticAIMessage](min_messages=3)
        compacted = await strategy.compact(messages, target_tokens, counter)
        ```
    """

    min_messages: int = 2

    async def compact(
        self,
        messages: list[MessageT],
        target_tokens: int,
        token_counter: TokenCounter[MessageT],
    ) -> list[MessageT]:
        """Drop oldest messages until under token budget."""
        result = list(messages)

        while len(result) > self.min_messages:
            current_tokens = token_counter.count_messages(result)
            if current_tokens <= target_tokens:
                break
            # Remove the oldest message (after any we want to preserve)
            result.pop(0)

        return result


@dataclass
class KeepFirstLast(Generic[MessageT]):
    """
    Keep first N and last M messages, drop the middle.

    Preserves the initial context (system prompts, first user message)
    and recent context, while dropping older middle messages.

    Args:
        keep_first: Number of messages to keep from start (default 2)
        keep_last: Number of messages to keep from end (default 5)

    Example:
        ```python
        strategy = KeepFirstLast[PydanticAIMessage](keep_first=2, keep_last=10)
        compacted = await strategy.compact(messages, target_tokens, counter)
        ```
    """

    keep_first: int = 2
    keep_last: int = 5

    async def compact(
        self,
        messages: list[MessageT],
        target_tokens: int,
        token_counter: TokenCounter[MessageT],
    ) -> list[MessageT]:
        """Keep first and last messages, drop the middle."""
        total_keep = self.keep_first + self.keep_last

        if len(messages) <= total_keep:
            return messages

        first = messages[: self.keep_first]
        last = messages[-self.keep_last :]

        return first + last


@dataclass
class SlidingWindow(Generic[MessageT]):
    """
    Sliding window that keeps recent messages within token budget.

    Keeps messages from the end, sliding the window forward as needed
    to stay within the token budget.

    Args:
        buffer_percent: Percentage of target to use as buffer (default 0.1)

    Example:
        ```python
        strategy = SlidingWindow[PydanticAIMessage]()
        compacted = await strategy.compact(messages, target_tokens, counter)
        ```
    """

    buffer_percent: float = 0.1

    async def compact(
        self,
        messages: list[MessageT],
        target_tokens: int,
        token_counter: TokenCounter[MessageT],
    ) -> list[MessageT]:
        """Keep a sliding window of recent messages within budget."""
        effective_target = int(target_tokens * (1 - self.buffer_percent))

        # Start from the end, add messages until we exceed budget
        result: list[MessageT] = []
        current_tokens = 0

        for msg in reversed(messages):
            msg_tokens = token_counter.count_single(msg)
            if current_tokens + msg_tokens > effective_target:
                break
            result.insert(0, msg)
            current_tokens += msg_tokens

        # Always keep at least 1 message
        if not result and messages:
            result = [messages[-1]]

        return result
