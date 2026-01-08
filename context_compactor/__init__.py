"""Context Compactor - Automatic context window management for AI agent SDKs.

This library provides generic, type-safe context compaction for multiple
AI agent SDKs including pydantic-ai, openai-agents, and claude-agent-sdk.

Example:
    ```python
    from context_compactor import ContextCompactor
    from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter
    from context_compactor.strategies.generic import KeepRecentMessages
    from context_compactor.adapters.pydantic_ai import pydantic_ai_adapter
    from pydantic_ai import Agent

    # Create a typed compactor
    compactor = ContextCompactor(
        max_context_tokens=128_000,
        strategy=KeepRecentMessages(keep_count=20),
        token_counter=PydanticAITokenCounter(),
    )

    # Integrate with pydantic-ai
    agent = Agent(
        'openai:gpt-4o',
        history_processors=[pydantic_ai_adapter(compactor)],
    )
    ```
"""

from .core.compactor import ContextCompactor
from .core.protocols import CompactionStrategy, TokenCounter
from .core.types import (
    ClaudeAgentMessage,
    OpenAIAgentsMessage,
    PydanticAIMessage,
)

__all__ = [
    # Core
    "ContextCompactor",
    "TokenCounter",
    "CompactionStrategy",
    # Type aliases
    "PydanticAIMessage",
    "OpenAIAgentsMessage",
    "ClaudeAgentMessage",
]

__version__ = "0.2.0"
