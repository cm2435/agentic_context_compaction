"""Example: Using SummarizeMiddle with pydantic-ai.

This strategy keeps first/last messages and summarizes the middle using
an LLM call, providing context continuity while saving tokens.

Requirements:
    pip install context-compactor[pydantic-ai]
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent

from context_compactor import ContextCompactor
from context_compactor.adapters.pydantic_ai import pydantic_ai_adapter
from context_compactor.strategies.pydantic_ai import SummarizeMiddle
from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter

# Separate agent for summarization
summarizer_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Summarize the following conversation concisely, preserving key information.",
)


async def summarize_messages(messages: list) -> str:
    """Summarize a list of messages using an LLM."""
    # Convert messages to readable format
    conversation = []
    for msg in messages:
        for part in msg.parts:
            if hasattr(part, "content"):
                role = "User" if hasattr(part, "timestamp") else "Assistant"
                conversation.append(f"{role}: {part.content}")

    text = "\n".join(conversation)
    result = await summarizer_agent.run(f"Summarize this conversation:\n\n{text}")
    return result.data


# Create compactor with SummarizeMiddle
compactor = ContextCompactor(
    max_context_tokens=16_000,
    trigger_at_percent=0.8,
    strategy=SummarizeMiddle(
        keep_first=2,  # Keep initial context
        keep_last=5,  # Keep recent messages
        summarizer=summarize_messages,
    ),
    token_counter=PydanticAITokenCounter(),
    verbose=True,
)

# Main agent
agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a helpful project assistant.",
    history_processors=[pydantic_ai_adapter(compactor)],
)


async def main():
    """Demonstrate summarization-based compaction."""
    history = []

    # Build up conversation
    prompts = [
        "Let's plan a new feature for user authentication.",
        "What database should we use?",
        "How about adding OAuth support?",
        "Can you outline the API endpoints?",
        "What about rate limiting?",
        "How should we handle errors?",
        "What tests should we write?",
        "Can you summarize our plan so far?",
    ]

    for i, prompt in enumerate(prompts):
        result = await agent.run(prompt, message_history=history)
        history = result.all_messages()
        print(f"Turn {i + 1}: {len(history)} messages")

    print(f"\nFinal response about the plan:\n{result.data[:500]}...")
    print(f"\nStats: {compactor.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
