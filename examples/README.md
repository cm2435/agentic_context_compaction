# Context Compactor Examples

This directory contains examples showing how to use context-compactor with different AI agent SDKs and compaction strategies.

## Examples by SDK

### pydantic-ai

| Example | Strategy | Description |
|---------|----------|-------------|
| [`pydantic_ai_keep_recent.py`](pydantic_ai_keep_recent.py) | `KeepRecentMessages` | Simple: keep last N messages |
| [`pydantic_ai_keep_first_last.py`](pydantic_ai_keep_first_last.py) | `KeepFirstLast` | Preserve initial context + recent messages |
| [`pydantic_ai_sliding_window.py`](pydantic_ai_sliding_window.py) | `SlidingWindow` | Token-aware: fit as many messages as possible |
| [`pydantic_ai_summarize_middle.py`](pydantic_ai_summarize_middle.py) | `SummarizeMiddle` | LLM-based: summarize middle messages |

### openai-agents

| Example | Strategy | Description |
|---------|----------|-------------|
| [`openai_agents_keep_recent.py`](openai_agents_keep_recent.py) | `KeepRecentMessages` | Simple: keep last N messages |
| [`openai_agents_sliding_window.py`](openai_agents_sliding_window.py) | `SlidingWindow` | Token-aware: fit as many messages as possible |

### claude-agent-sdk

| Example | Strategy | Description |
|---------|----------|-------------|
| [`claude_agent_keep_recent.py`](claude_agent_keep_recent.py) | `KeepRecentMessages` | Simple: keep last N messages |

## Strategies Summary

| Strategy | Preserves | Best For |
|----------|-----------|----------|
| `KeepRecentMessages` | Last N messages | Simple truncation, short conversations |
| `KeepFirstLast` | First N + Last M | Long conversations where initial context matters |
| `SlidingWindow` | As many as fit in budget | Token-efficient, variable message lengths |
| `DropOldestUntilFits` | Messages that fit | Token-aware, minimal dropping |
| `SummarizeMiddle` | First/Last + Summary | Best context preservation, uses LLM |

## Running Examples

```bash
# Install with SDK support
pip install context-compactor[pydantic-ai]  # or openai-agents, claude-agent

# Run an example
python examples/pydantic_ai_keep_recent.py
```

## Writing Custom Strategies

You can write your own strategy by implementing the `CompactionStrategy` protocol:

```python
from context_compactor import CompactionStrategy, TokenCounter
from pydantic_ai.messages import ModelRequest, ModelResponse

class MyCustomStrategy:
    async def compact(
        self,
        messages: list[ModelRequest | ModelResponse],
        target_tokens: int,
        token_counter: TokenCounter,
    ) -> list[ModelRequest | ModelResponse]:
        # Your custom logic here
        # Full access to typed message parts!
        result = []
        for msg in messages:
            for part in msg.parts:
                # Inspect, filter, transform...
                pass
        return result
```

