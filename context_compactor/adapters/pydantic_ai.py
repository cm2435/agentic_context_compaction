"""Pydantic-ai integration adapter.

Provides integration with pydantic-ai's history_processors mechanism.

Requires: pip install context-compactor[pydantic-ai]
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

if TYPE_CHECKING:
    from ..core.compactor import ContextCompactor

# Type alias
PydanticAIMessage = ModelRequest | ModelResponse


def pydantic_ai_adapter(
    compactor: ContextCompactor[PydanticAIMessage],
) -> Callable[[list[ModelMessage]], list[ModelMessage]]:
    """
    Create a history_processor function for pydantic-ai Agent.

    The returned function can be passed to Agent's history_processors
    parameter to automatically compact message history when it exceeds
    the configured threshold.

    Args:
        compactor: A ContextCompactor configured for pydantic-ai messages

    Returns:
        A sync function suitable for use as a history_processor

    Example:
        ```python
        from context_compactor import ContextCompactor
        from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter
        from context_compactor.strategies.generic import KeepRecentMessages
        from context_compactor.adapters.pydantic_ai import pydantic_ai_adapter
        from pydantic_ai import Agent

        compactor = ContextCompactor(
            max_context_tokens=128_000,
            strategy=KeepRecentMessages(keep_count=20),
            token_counter=PydanticAITokenCounter(),
        )

        agent = Agent(
            'openai:gpt-4o',
            history_processors=[pydantic_ai_adapter(compactor)],
        )
        ```
    """

    def process(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Filter to just Request/Response (ModelMessage union)
        typed_messages: list[PydanticAIMessage] = [
            msg for msg in messages if isinstance(msg, ModelRequest | ModelResponse)
        ]

        # Run async compaction synchronously
        coro = compactor.maybe_compact(typed_messages)
        try:
            asyncio.get_running_loop()
            # Already in async context - schedule in thread pool
            import concurrent.futures

            def run_coro() -> list[PydanticAIMessage]:
                return asyncio.run(coro)

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(run_coro)
                compacted = future.result()
        except RuntimeError:
            # No running loop - can use asyncio.run directly
            compacted = asyncio.run(coro)

        return compacted  # type: ignore

    return process


async def pydantic_ai_adapter_async(
    compactor: ContextCompactor[PydanticAIMessage],
) -> Callable[[list[ModelMessage]], list[ModelMessage]]:
    """
    Create an async-aware history_processor for pydantic-ai Agent.

    Note: pydantic-ai's history_processors are sync, but this wrapper
    handles the async compaction internally.
    """
    return pydantic_ai_adapter(compactor)
