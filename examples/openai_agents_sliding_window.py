"""Example: Using SlidingWindow with openai-agents SDK.

This example uses token-aware sliding window for automatic
compaction based on the actual token budget.

Requirements:
    pip install context-compactor[openai-agents]
"""

from __future__ import annotations

import asyncio

from agents import Agent, Runner

from context_compactor import ContextCompactor
from context_compactor.adapters.openai_agents import openai_agents_adapter
from context_compactor.strategies import SlidingWindow
from context_compactor.tokenizers.openai_agents import OpenAIAgentsTokenCounter

# SlidingWindow fits as many messages as possible
compactor = ContextCompactor(
    max_context_tokens=16_000,
    trigger_at_percent=0.8,
    strategy=SlidingWindow(buffer_percent=0.1),
    token_counter=OpenAIAgentsTokenCounter(),
    verbose=True,
)

hooks = openai_agents_adapter(compactor)

agent = Agent(
    name="CodeHelper",
    instructions="You are a helpful coding assistant that explains concepts in detail.",
)


async def main():
    """Demonstrate sliding window with openai-agents."""
    for i in range(20):
        await Runner.run(
            agent,
            input=f"Explain Python feature {i} with examples",
            hooks=hooks,
        )
        print(f"Turn {i + 1}: Completed")

    print(f"\nStats: {compactor.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
