# context-compactor

Automatic context window management for AI agent SDKs.

## Why?

Long-running agent conversations exceed model context windows, causing:
- API errors when context is too large
- Lost context when manually truncating  
- Expensive token usage on repeated information

This library provides automatic, intelligent context compaction with:
- **Generic & Type-Safe** — Works with native SDK message types
- **Multiple strategies** — From simple truncation to LLM summarization
- **SDK adapters** — pydantic-ai, openai-agents, claude-agent-sdk
- **Lifecycle hooks** — UI feedback during compaction (spinners, progress)

## Installation

```bash
# Core (no dependencies)
pip install context-compactor

# With pydantic-ai support
pip install context-compactor[pydantic-ai]

# With all SDKs
pip install context-compactor[all-sdks]
```

## Quick Start

### pydantic-ai

```python
from pydantic_ai import Agent
from context_compactor import ContextCompactor, LoggingHook
from context_compactor.adapters.pydantic_ai import pydantic_ai_adapter
from context_compactor.strategies import KeepRecentMessages
from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter

compactor = ContextCompactor(
    max_context_tokens=128_000,
    strategy=KeepRecentMessages(keep_count=20),
    token_counter=PydanticAITokenCounter(),
    hooks=[LoggingHook()],  # Optional: log compaction events
)

agent = Agent(
    'openai:gpt-4o',
    history_processors=[pydantic_ai_adapter(compactor)],
)

# Compaction happens automatically when context approaches limit
result = await agent.run("Continue...", message_history=long_history)
```

### openai-agents

```python
from agents import Agent, Runner
from context_compactor import ContextCompactor
from context_compactor.adapters.openai_agents import openai_agents_adapter
from context_compactor.strategies import SlidingWindow
from context_compactor.tokenizers.openai_agents import OpenAIAgentsTokenCounter

compactor = ContextCompactor(
    max_context_tokens=128_000,
    strategy=SlidingWindow(),
    token_counter=OpenAIAgentsTokenCounter(),
)

result = await Runner.run(
    agent,
    input="Hello",
    hooks=openai_agents_adapter(compactor),
)
```

### claude-agent-sdk

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from context_compactor import ContextCompactor
from context_compactor.adapters.claude_agent import claude_agent_adapter
from context_compactor.strategies import KeepFirstLast
from context_compactor.tokenizers.claude_agent import ClaudeAgentTokenCounter

compactor = ContextCompactor(
    max_context_tokens=200_000,
    strategy=KeepFirstLast(keep_first=2, keep_last=10),
    token_counter=ClaudeAgentTokenCounter(),
)

hook_event, hook_matchers = claude_agent_adapter(compactor)
options = ClaudeAgentOptions(hooks={hook_event: hook_matchers})

async with ClaudeSDKClient(options=options) as client:
    await client.query("Help me with this large codebase...")
```

## Lifecycle Hooks

Hooks let you react to compaction events—perfect for showing UI feedback like Cursor's "Summarizing conversation..." spinner.

### Built-in Hooks

```python
from context_compactor import LoggingHook, CallbackHook

# LoggingHook - prints to stdout
compactor = ContextCompactor(
    ...,
    hooks=[LoggingHook(prefix="[MyApp]")],
)

# CallbackHook - call your own async functions
async def on_start():
    await show_spinner("Summarizing context...")

async def on_end(result):
    await hide_spinner()
    print(f"Saved {result.tokens_saved} tokens")

compactor = ContextCompactor(
    ...,
    hooks=[CallbackHook(on_start_callback=on_start, on_end_callback=on_end)],
)
```

### Custom Hooks

Implement the `CompactionHook` protocol:

```python
from dataclasses import dataclass
from context_compactor import CompactionResult

@dataclass
class WebhookHook:
    """Send compaction events to your backend."""
    webhook_url: str
    
    async def on_start(self) -> None:
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json={
                "type": "compaction_started",
                "message": "Summarizing context..."
            })
    
    async def on_end(self, result: CompactionResult) -> None:
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json={
                "type": "compaction_completed",
                "tokens_saved": result.tokens_saved,
            })

# Use multiple hooks
compactor = ContextCompactor(
    max_context_tokens=128_000,
    strategy=KeepRecentMessages(keep_count=20),
    token_counter=PydanticAITokenCounter(),
    hooks=[
        WebhookHook(webhook_url="https://your-app.com/events"),
        LoggingHook(),
    ],
)
```

### Event Order

Hooks fire synchronously before/after compaction, ensuring correct ordering with streaming:

```
1. User sends message
2. history_processor runs
   → hook.on_start() fires → UI shows spinner
   → compaction happens
   → hook.on_end() fires → UI hides spinner
3. Stream begins → tokens flow to UI
```

## Compaction Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `KeepRecentMessages` | Keep last N messages | Simple truncation |
| `KeepFirstLast` | Keep first N + last M, drop middle | Preserve initial context |
| `SlidingWindow` | Fit as many recent as token budget allows | Token-efficient |
| `DropOldestUntilFits` | Remove oldest until under budget | Minimal dropping |
| `SummarizeMiddle` | Keep first/last, LLM-summarize middle | Best preservation |

## Custom Strategies

Write type-safe strategies that work with native SDK message types:

```python
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart

class KeepToolCalls:
    """Keep all tool interactions, drop regular text."""
    
    async def compact(
        self,
        messages: list[ModelRequest | ModelResponse],
        target_tokens: int,
        token_counter,
    ) -> list[ModelRequest | ModelResponse]:
        result = []
        for msg in messages:
            if isinstance(msg, ModelResponse):
                tool_parts = [p for p in msg.parts if isinstance(p, ToolCallPart)]
                if tool_parts:
                    result.append(ModelResponse(parts=tool_parts))
        return result
```

## Examples

See the [`examples/`](examples/) directory for complete working examples:

| SDK | Examples |
|-----|----------|
| pydantic-ai | [keep_recent](examples/pydantic_ai_keep_recent.py), [keep_first_last](examples/pydantic_ai_keep_first_last.py), [sliding_window](examples/pydantic_ai_sliding_window.py), [summarize_middle](examples/pydantic_ai_summarize_middle.py), [streaming_hooks](examples/pydantic_ai_streaming_hooks.py) |
| openai-agents | [keep_recent](examples/openai_agents_keep_recent.py), [sliding_window](examples/openai_agents_sliding_window.py) |
| claude-agent-sdk | [keep_recent](examples/claude_agent_keep_recent.py) |

## API Reference

### `ContextCompactor`

```python
ContextCompactor(
    max_context_tokens: int,           # Model's context window
    strategy: CompactionStrategy,      # How to compact
    token_counter: TokenCounter,       # How to count tokens
    trigger_at_percent: float = 0.8,   # Compact at 80% full
    verbose: bool = False,             # Print debug info
    hooks: list[CompactionHook] = [],  # Lifecycle hooks
)
```

### Methods

- `await compactor.maybe_compact(messages)` — Compact if over threshold
- `compactor.get_stats()` — Get compaction statistics
- `compactor.reset_stats()` — Reset statistics

### `CompactionResult`

Passed to `hook.on_end()`:

```python
@dataclass
class CompactionResult:
    original_tokens: int        # Tokens before compaction
    compacted_tokens: int       # Tokens after compaction
    tokens_saved: int           # original - compacted
    original_message_count: int
    compacted_message_count: int
    strategy_name: str          # e.g., "KeepRecentMessages"
```

## Development

```bash
# Clone and setup
git clone ...
cd context-compactor
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check .

# Type check
ty check context_compactor/
```

## License

MIT
