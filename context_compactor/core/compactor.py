"""Generic ContextCompactor for automatic context window management."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .hooks import CompactionHook, CompactionResult
from .protocols import CompactionStrategy, TokenCounter

MessageT = TypeVar("MessageT")


@dataclass
class ContextCompactor(Generic[MessageT]):
    """
    Generic context compactor for AI agent conversations.

    Monitors message history size and automatically compacts when approaching
    the context window limit, using a configurable strategy.

    The compactor is generic over the SDK message type (MessageT), ensuring
    type safety throughout the compaction pipeline.

    Args:
        max_context_tokens: Maximum tokens for the model's context window
        trigger_at_percent: Trigger compaction at this percentage of max (0.0-1.0)
        strategy: CompactionStrategy to use for compacting messages
        token_counter: TokenCounter for estimating token counts
        verbose: Print debug information during compaction
        hooks: List of CompactionHook instances to notify on compaction events

    Example:
        ```python
        from context_compactor import ContextCompactor
        from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter
        from context_compactor.strategies.pydantic_ai import SummarizeMiddle
        from context_compactor.hooks import LoggingHook

        compactor: ContextCompactor[PydanticAIMessage] = ContextCompactor(
            max_context_tokens=128_000,
            trigger_at_percent=0.8,
            strategy=SummarizeMiddle(keep_first=2, keep_last=5),
            token_counter=PydanticAITokenCounter(model="gpt-4o"),
            hooks=[LoggingHook()],
        )

        # Use with pydantic-ai
        agent = Agent(
            'openai:gpt-4o',
            history_processors=[pydantic_ai_adapter(compactor)],
        )
        ```
    """

    max_context_tokens: int
    strategy: CompactionStrategy[MessageT]
    token_counter: TokenCounter[MessageT]
    trigger_at_percent: float = 0.8
    verbose: bool = False
    hooks: list[CompactionHook] = field(default_factory=list)

    # Statistics (not init params)
    compactions_performed: int = field(default=0, init=False)
    tokens_saved: int = field(default=0, init=False)

    @property
    def trigger_threshold(self) -> int:
        """Token count that triggers compaction."""
        return int(self.max_context_tokens * self.trigger_at_percent)

    async def maybe_compact(self, messages: list[MessageT]) -> list[MessageT]:
        """
        Compact messages if over threshold, otherwise return unchanged.

        If compaction is triggered, all registered hooks are notified:
        1. `hook.on_start()` is called for each hook before compaction
        2. Compaction is performed using the configured strategy
        3. `hook.on_end(result)` is called for each hook after compaction

        Args:
            messages: List of native SDK messages

        Returns:
            Original messages if under threshold, compacted messages otherwise
        """
        current_tokens = self.token_counter.count_messages(messages)

        if self.verbose:
            print(
                f"[Compactor] Current: {current_tokens:,} / "
                f"{self.max_context_tokens:,} tokens "
                f"(threshold: {self.trigger_threshold:,})"
            )

        if current_tokens < self.trigger_threshold:
            return messages

        if self.verbose:
            pct = self.trigger_at_percent * 100
            print(f"[Compactor] Threshold exceeded ({pct:.0f}%), compacting...")

        # Fire all start hooks
        for hook in self.hooks:
            await hook.on_start()

        # Target 50% of max to leave room for new content
        target_tokens = int(self.max_context_tokens * 0.5)

        compacted = await self.strategy.compact(
            messages,
            target_tokens,
            self.token_counter,
        )

        new_tokens = self.token_counter.count_messages(compacted)

        # Update stats
        self.compactions_performed += 1
        self.tokens_saved += current_tokens - new_tokens

        # Build result for hooks
        result = CompactionResult(
            original_tokens=current_tokens,
            compacted_tokens=new_tokens,
            tokens_saved=current_tokens - new_tokens,
            original_message_count=len(messages),
            compacted_message_count=len(compacted),
            strategy_name=type(self.strategy).__name__,
        )

        # Fire all end hooks
        for hook in self.hooks:
            await hook.on_end(result)

        if self.verbose:
            reduction = (1 - new_tokens / current_tokens) * 100
            print(
                f"[Compactor] Reduced: {current_tokens:,} -> {new_tokens:,} "
                f"({reduction:.1f}% reduction)"
            )

        return compacted

    def reset_stats(self) -> None:
        """Reset compaction statistics."""
        self.compactions_performed = 0
        self.tokens_saved = 0

    def get_stats(self) -> dict:
        """Get compaction statistics."""
        return {
            "compactions_performed": self.compactions_performed,
            "tokens_saved": self.tokens_saved,
            "max_context_tokens": self.max_context_tokens,
            "trigger_threshold": self.trigger_threshold,
        }

    def as_processor(self) -> Callable[[list[MessageT]], list[MessageT]]:
        """
        Return a sync processor function for SDK integration.

        Note: This runs the async compact in a blocking manner.
        For async contexts, use maybe_compact directly.
        """
        import asyncio

        def processor(messages: list[MessageT]) -> list[MessageT]:
            return asyncio.get_event_loop().run_until_complete(self.maybe_compact(messages))

        return processor
