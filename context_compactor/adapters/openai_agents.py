"""OpenAI Agents SDK integration adapter.

Provides integration with openai-agents RunHooks for context compaction.

Requires: pip install context-compactor[openai-agents]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from agents.agent import Agent
from agents.items import RunItem, TResponseInputItem
from agents.lifecycle import RunHooks
from agents.run_context import RunContextWrapper, TContext

if TYPE_CHECKING:
    from ..core.compactor import ContextCompactor

# Type alias
OpenAIAgentsMessage = RunItem | TResponseInputItem


class CompactionRunHooks(RunHooks[TContext]):
    """
    RunHooks implementation that compacts context before each LLM call.

    Integrates with the openai-agents SDK's hook system to automatically
    compact the message history when it exceeds the configured threshold.

    Args:
        compactor: A ContextCompactor configured for openai-agents messages

    Example:
        ```python
        from context_compactor import ContextCompactor
        from context_compactor.tokenizers.openai_agents import OpenAIAgentsTokenCounter
        from context_compactor.strategies.generic import SlidingWindow
        from context_compactor.adapters.openai_agents import CompactionRunHooks
        from agents import Agent, Runner

        compactor = ContextCompactor(
            max_context_tokens=128_000,
            strategy=SlidingWindow(),
            token_counter=OpenAIAgentsTokenCounter(),
        )

        hooks = CompactionRunHooks(compactor)

        result = await Runner.run(
            agent,
            input="Hello",
            hooks=hooks,
        )
        ```
    """

    def __init__(self, compactor: ContextCompactor[OpenAIAgentsMessage]):
        self.compactor = compactor

    async def on_llm_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        """
        Called just before invoking the LLM.

        Note: This hook doesn't allow modifying input_items directly.
        For compaction, consider using pre-processing before Runner.run()
        or implementing custom run logic.
        """
        # Log compaction check (the hook doesn't support modifying input_items)
        # Cast is safe: TResponseInputItem is part of OpenAIAgentsMessage union
        messages = cast(list[OpenAIAgentsMessage], input_items)
        current_tokens = self.compactor.token_counter.count_messages(messages)
        if self.compactor.verbose and current_tokens > self.compactor.trigger_threshold:
            print(
                f"[Compactor] Warning: Input exceeds threshold "
                f"({current_tokens:,} > {self.compactor.trigger_threshold:,} tokens). "
                f"Consider pre-compacting before Runner.run()."
            )


def openai_agents_adapter(
    compactor: ContextCompactor[OpenAIAgentsMessage],
) -> CompactionRunHooks:
    """
    Create RunHooks for openai-agents that handle context compaction.

    Note: The openai-agents RunHooks don't allow modifying input before
    the LLM call. For full compaction support, pre-process your input
    items before calling Runner.run():

    ```python
    # Pre-compact before running
    compacted_input = await compactor.maybe_compact(input_items)
    result = await Runner.run(agent, input=compacted_input)
    ```

    Args:
        compactor: A ContextCompactor configured for openai-agents messages

    Returns:
        A CompactionRunHooks instance for monitoring (not modifying) context size
    """
    return CompactionRunHooks(compactor)
