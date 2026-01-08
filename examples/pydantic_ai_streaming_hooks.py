"""Example: Streaming with compaction lifecycle hooks.

This example demonstrates how to use compaction hooks to provide
UI feedback during long-running compaction operations, similar to
Cursor's "summarizing conversation" spinner.

Requirements:
    pip install context-compactor[pydantic-ai] httpx

Use Case:
    When your agent conversation gets too long and needs compaction,
    your UI should:
    1. Show a "Summarizing context..." spinner when compaction starts
    2. Hide the spinner when compaction completes
    3. Resume showing streamed tokens

The hooks ensure these events fire in the correct order, before
streaming begins.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from context_compactor import CompactionResult, ContextCompactor, LoggingHook
from context_compactor.adapters.pydantic_ai import pydantic_ai_adapter
from context_compactor.strategies import KeepRecentMessages
from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter

# =============================================================================
# Custom Webhook Hook
# =============================================================================


@dataclass
class WebhookHook:
    """
    Hook that sends compaction events to a webhook.

    In a real application, this would send to your backend which
    then pushes to the frontend via WebSocket or SSE.
    """

    webhook_url: str

    async def on_start(self) -> None:
        """Send compaction started event."""
        # In production, use httpx:
        # async with httpx.AsyncClient() as client:
        #     await client.post(self.webhook_url, json={
        #         "type": "compaction_started",
        #         "message": "Summarizing context..."
        #     })
        print(f"[Webhook] POST {self.webhook_url}")
        print('  {"type": "compaction_started", "message": "Summarizing context..."}')

    async def on_end(self, result: CompactionResult) -> None:
        """Send compaction completed event."""
        # In production, use httpx:
        # async with httpx.AsyncClient() as client:
        #     await client.post(self.webhook_url, json={
        #         "type": "compaction_completed",
        #         "tokens_saved": result.tokens_saved,
        #         "original_tokens": result.original_tokens,
        #         "compacted_tokens": result.compacted_tokens,
        #     })
        print(f"[Webhook] POST {self.webhook_url}")
        print(
            f'  {{"type": "compaction_completed", '
            f'"tokens_saved": {result.tokens_saved}, '
            f'"original_tokens": {result.original_tokens}, '
            f'"compacted_tokens": {result.compacted_tokens}}}'
        )


# =============================================================================
# Setup
# =============================================================================

# Create the compactor with multiple hooks
compactor = ContextCompactor(
    max_context_tokens=128_000,
    trigger_at_percent=0.8,
    strategy=KeepRecentMessages(keep_count=15),
    token_counter=PydanticAITokenCounter(),
    hooks=[
        WebhookHook(webhook_url="https://your-app.com/events"),
        LoggingHook(prefix="[MyApp]"),
    ],
)

# Create the agent with compaction
agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a helpful assistant. Be concise.",
    history_processors=[pydantic_ai_adapter(compactor)],
)


# =============================================================================
# Simulate Long Conversation (for demo without real API calls)
# =============================================================================


def create_fake_history(num_turns: int) -> list[ModelMessage]:
    """Create a fake conversation history for demonstration."""
    messages: list[ModelMessage] = []
    for i in range(num_turns):
        # User message
        messages.append(
            ModelRequest(parts=[UserPromptPart(content=f"Tell me about topic {i}. " * 20)])
        )
    return messages


# =============================================================================
# Main
# =============================================================================


async def main():
    """Demonstrate the streaming + hooks flow."""
    print("=" * 60)
    print("Compaction Hooks Demo")
    print("=" * 60)
    print()

    # Create a long history that will trigger compaction
    print("Creating fake long conversation history...")
    fake_history = create_fake_history(50)  # 50 user messages
    print(f"History has {len(fake_history)} messages")
    print()

    # Check initial token count
    initial_tokens = compactor.token_counter.count_messages(fake_history)  # type: ignore
    print(f"Initial tokens: {initial_tokens:,}")
    print(f"Trigger threshold: {compactor.trigger_threshold:,}")
    print()

    # Manually trigger compaction to show hooks
    print("Triggering compaction...")
    print("-" * 40)
    compacted = await compactor.maybe_compact(fake_history)  # type: ignore
    print("-" * 40)
    print()

    # Show results
    final_tokens = compactor.token_counter.count_messages(compacted)  # type: ignore
    print(f"After compaction: {len(compacted)} messages, {final_tokens:,} tokens")
    print()

    # Show the event order that would happen in streaming
    print("=" * 60)
    print("Event Order in Streaming Scenario:")
    print("=" * 60)
    print("""
1. User sends message
2. history_processor runs
   → on_start() fires → UI shows spinner
   → compaction happens
   → on_end() fires → UI hides spinner
3. Stream begins
   → tokens flow to UI
4. Stream completes
""")

    print(f"Compactor stats: {compactor.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
