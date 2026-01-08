"""SDK integration adapters.

Each adapter provides integration with its respective SDK's
hook or processor mechanism for automatic context compaction.
"""

__all__: list[str] = []

# Optional: pydantic-ai adapter
try:
    from .pydantic_ai import pydantic_ai_adapter  # noqa: F401

    __all__.append("pydantic_ai_adapter")
except ImportError:
    pass

# Optional: openai-agents adapter
try:
    from .openai_agents import (
        CompactionRunHooks,  # noqa: F401
        openai_agents_adapter,  # noqa: F401
    )

    __all__.extend(["openai_agents_adapter", "CompactionRunHooks"])
except ImportError:
    pass

# Optional: claude-agent adapter
try:
    from .claude_agent import claude_agent_adapter  # noqa: F401

    __all__.append("claude_agent_adapter")
except ImportError:
    pass
