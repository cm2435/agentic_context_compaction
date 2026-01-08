"""Example: Using SlidingWindow with pydantic-ai.

This example uses token-aware sliding window compaction that automatically
fits as many recent messages as possible within the budget.

Requirements:
    pip install context-compactor[pydantic-ai]
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent

from context_compactor import ContextCompactor
from context_compactor.adapters.pydantic_ai import pydantic_ai_adapter
from context_compactor.strategies import SlidingWindow
from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter

# SlidingWindow automatically fits messages within token budget
compactor = ContextCompactor(
    max_context_tokens=8_000,  # Small context for demo
    trigger_at_percent=0.8,
    strategy=SlidingWindow(buffer_percent=0.1),  # Leave 10% buffer
    token_counter=PydanticAITokenCounter(),
    verbose=True,
)

agent = Agent(
    "openai:gpt-4o-mini",  # Using mini for cost efficiency
    system_prompt="You are a helpful assistant.",
    history_processors=[pydantic_ai_adapter(compactor)],
)


async def main():
    """Demonstrate sliding window compaction."""
    history = []

    for i in range(30):
        # Ask questions that generate varying response lengths
        prompt = f"Explain concept {i} in Python in detail"
        result = await agent.run(prompt, message_history=history)
        history = result.all_messages()

        tokens = compactor.token_counter.count_messages(history)
        print(f"Turn {i + 1}: {len(history)} messages, ~{tokens} tokens")

    print(f"\nStats: {compactor.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
