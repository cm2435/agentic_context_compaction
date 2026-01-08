"""Example: Using KeepRecentMessages with pydantic-ai.

This example shows how to integrate context compaction with pydantic-ai
using the simple KeepRecentMessages strategy.

Requirements:
    pip install context-compactor[pydantic-ai]
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent

from context_compactor import ContextCompactor
from context_compactor.adapters.pydantic_ai import pydantic_ai_adapter
from context_compactor.strategies import KeepRecentMessages
from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter

# Create the compactor with KeepRecentMessages strategy
compactor = ContextCompactor(
    max_context_tokens=128_000,
    trigger_at_percent=0.8,  # Compact at 80% full
    strategy=KeepRecentMessages(keep_count=20),  # Keep last 20 messages
    token_counter=PydanticAITokenCounter(),
    verbose=True,  # Print debug info
)

# Create the agent with compaction
agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a helpful coding assistant.",
    history_processors=[pydantic_ai_adapter(compactor)],
)


async def main():
    """Run a conversation that demonstrates compaction."""
    history = []

    # Simulate many turns to trigger compaction
    for i in range(25):
        prompt = f"Tell me about feature {i} of Python"
        result = await agent.run(prompt, message_history=history)
        history = result.all_messages()
        print(f"Turn {i + 1}: {len(history)} messages in history")

    print(f"\nFinal stats: {compactor.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
