"""Example: Using KeepRecentMessages with claude-agent-sdk.

This example shows how to integrate context compaction with the
Claude Agent SDK using the PreCompact hook.

Requirements:
    pip install context-compactor[claude-agent]
"""

from __future__ import annotations

import asyncio

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from context_compactor import ContextCompactor
from context_compactor.adapters.claude_agent import claude_agent_adapter
from context_compactor.strategies import KeepRecentMessages
from context_compactor.tokenizers.claude_agent import ClaudeAgentTokenCounter

# Create the compactor
compactor = ContextCompactor(
    max_context_tokens=200_000,  # Claude's context window
    trigger_at_percent=0.8,
    strategy=KeepRecentMessages(keep_count=20),
    token_counter=ClaudeAgentTokenCounter(),
    verbose=True,
)

# Create hook configuration
hook_event, hook_matchers = claude_agent_adapter(compactor)


async def main():
    """Run a conversation with Claude Agent SDK."""
    options = ClaudeAgentOptions(
        hooks={hook_event: hook_matchers},
    )

    async with ClaudeSDKClient(options=options) as client:
        for i in range(25):
            await client.query(f"Tell me about feature {i}")
            print(f"Turn {i + 1}: Response received")

    print(f"\nStats: {compactor.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
