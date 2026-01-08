"""Generic protocols for context compaction.

These protocols are generic over the SDK message type, allowing type-safe
compaction strategies that work with native SDK types.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

# Generic type variable for SDK message types
MessageT = TypeVar("MessageT")


@runtime_checkable
class TokenCounter(Protocol[MessageT]):
    """
    Protocol for counting tokens in native SDK messages.

    Implementations are SDK-specific and know how to extract text
    from their respective message types.

    Example:
        ```python
        from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter

        counter = PydanticAITokenCounter(model="gpt-4o")
        tokens = counter.count_messages(messages)
        ```
    """

    def count_messages(self, messages: list[MessageT]) -> int:
        """
        Count total tokens across all messages.

        Args:
            messages: List of native SDK messages

        Returns:
            Total token count
        """
        ...

    def count_single(self, message: MessageT) -> int:
        """
        Count tokens in a single message.

        Args:
            message: A single native SDK message

        Returns:
            Token count for this message
        """
        ...


@runtime_checkable
class CompactionStrategy(Protocol[MessageT]):
    """
    Protocol for message compaction strategies.

    Strategies receive and return native SDK message types,
    allowing full control over what to keep, drop, or summarize.

    Example:
        ```python
        class MyStrategy:
            async def compact(
                self,
                messages: list[ModelRequest | ModelResponse],
                target_tokens: int,
                token_counter: TokenCounter,
            ) -> list[ModelRequest | ModelResponse]:
                # Keep only recent messages
                return messages[-10:]
        ```
    """

    async def compact(
        self,
        messages: list[MessageT],
        target_tokens: int,
        token_counter: TokenCounter[MessageT],
    ) -> list[MessageT]:
        """
        Compact messages to fit within target token count.

        Args:
            messages: List of native SDK messages to compact
            target_tokens: Target token count to compress to
            token_counter: Token counter for estimating sizes

        Returns:
            Compacted list of messages (same SDK types as input)
        """
        ...
