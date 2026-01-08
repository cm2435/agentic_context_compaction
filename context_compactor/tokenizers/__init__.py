"""Token counters for SDK-specific message types.

Each SDK has its own token counter that knows how to extract
text from its native message types.
"""

__all__: list[str] = []

# Optional: pydantic-ai counter
try:
    from .pydantic_ai import PydanticAITokenCounter  # noqa: F401

    __all__.append("PydanticAITokenCounter")
except ImportError:
    pass

# Optional: openai-agents counter
try:
    from .openai_agents import OpenAIAgentsTokenCounter  # noqa: F401

    __all__.append("OpenAIAgentsTokenCounter")
except ImportError:
    pass

# Optional: claude-agent counter
try:
    from .claude_agent import ClaudeAgentTokenCounter  # noqa: F401

    __all__.append("ClaudeAgentTokenCounter")
except ImportError:
    pass
