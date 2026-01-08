"""Example: Using KeepRecentMessages with openai-agents SDK.

This example shows how to integrate context compaction with the
OpenAI Agents SDK using RunHooks.

Requirements:
    pip install context-compactor[openai-agents]
"""

from __future__ import annotations

import asyncio

from agents import Agent, Runner

from context_compactor import ContextCompactor
from context_compactor.adapters.openai_agents import openai_agents_adapter
from context_compactor.strategies import KeepRecentMessages
from context_compactor.tokenizers.openai_agents import OpenAIAgentsTokenCounter

# Create the compactor
compactor = ContextCompactor(
    max_context_tokens=128_000,
    trigger_at_percent=0.8,
    strategy=KeepRecentMessages(keep_count=20),
    token_counter=OpenAIAgentsTokenCounter(),
    verbose=True,
)

# Create hooks adapter
hooks = openai_agents_adapter(compactor)

# Define the agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful coding assistant.",
)


async def main():
    """Run a conversation with automatic compaction."""
    for i in range(25):
        # Run the agent
        await Runner.run(
            agent,
            input=f"Tell me about Python feature {i}",
            hooks=hooks,
        )
        print(f"Turn {i + 1}: Response received")

    print(f"\nStats: {compactor.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
