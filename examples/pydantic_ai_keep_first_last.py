"""Example: Using KeepFirstLast with pydantic-ai.

This example preserves the initial context (system prompt, first user message)
and recent messages while dropping the middle.

Requirements:
    pip install context-compactor[pydantic-ai]
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent

from context_compactor import ContextCompactor
from context_compactor.adapters.pydantic_ai import pydantic_ai_adapter
from context_compactor.strategies import KeepFirstLast
from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter

# Keep first 3 and last 10 messages
compactor = ContextCompactor(
    max_context_tokens=128_000,
    trigger_at_percent=0.8,
    strategy=KeepFirstLast(keep_first=3, keep_last=10),
    token_counter=PydanticAITokenCounter(),
    verbose=True,
)

agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a helpful assistant with context about the user's project.",
    history_processors=[pydantic_ai_adapter(compactor)],
)


async def main():
    """Run a long conversation where initial context is preserved."""
    history = []

    # First messages establish context
    result = await agent.run(
        "My project is called 'Deepflow' - it's an AI automation framework.",
        message_history=history,
    )
    history = result.all_messages()

    result = await agent.run(
        "It uses pydantic-ai for agent orchestration.",
        message_history=history,
    )
    history = result.all_messages()

    # Simulate many follow-up questions
    for i in range(20):
        result = await agent.run(
            f"How should I implement feature {i}?",
            message_history=history,
        )
        history = result.all_messages()
        print(f"Turn {i + 3}: {len(history)} messages")

    # The agent should still remember "Deepflow" from the first messages
    result = await agent.run(
        "What's my project called again?",
        message_history=history,
    )
    print(f"\nFinal response: {result.data}")
    print(f"Stats: {compactor.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
