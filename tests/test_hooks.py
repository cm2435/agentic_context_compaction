"""Tests for compaction lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from context_compactor import (
    CallbackHook,
    CompactionResult,
    ContextCompactor,
    LoggingHook,
)
from context_compactor.strategies import KeepRecentMessages

from .conftest import MockMessage, MockTokenCounter

# =============================================================================
# Test Hook Implementation
# =============================================================================


@dataclass
class RecordingHook:
    """Hook that records all events for testing."""

    name: str = "test"
    events: list = None  # type: ignore

    def __post_init__(self):
        if self.events is None:
            self.events = []

    async def on_start(self) -> None:
        self.events.append(f"{self.name}:start")

    async def on_end(self, result: CompactionResult) -> None:
        self.events.append(f"{self.name}:end:{result.tokens_saved}")


# =============================================================================
# Hook Invocation Tests
# =============================================================================


class TestHookInvocation:
    """Tests for hook invocation during compaction."""

    @pytest.mark.asyncio
    async def test_single_hook_called(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Single hook should receive start and end calls."""
        events: list[str] = []

        @dataclass
        class SimpleHook:
            async def on_start(self) -> None:
                events.append("start")

            async def on_end(self, result: CompactionResult) -> None:
                events.append("end")

        compactor = ContextCompactor(
            max_context_tokens=50,  # Force compaction
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[SimpleHook()],
        )

        await compactor.maybe_compact(long_messages)

        assert events == ["start", "end"]

    @pytest.mark.asyncio
    async def test_multiple_hooks_all_called(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """All hooks in the list should be called."""
        events: list[str] = []
        hook_a = RecordingHook(name="A", events=events)
        hook_b = RecordingHook(name="B", events=events)

        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[hook_a, hook_b],
        )

        await compactor.maybe_compact(long_messages)

        # All starts fire before all ends
        assert len(events) == 4
        assert events[0] == "A:start"
        assert events[1] == "B:start"
        assert events[2].startswith("A:end:")
        assert events[3].startswith("B:end:")

    @pytest.mark.asyncio
    async def test_hooks_called_in_order(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Start hooks fire before end hooks."""
        events: list[str] = []
        hook = RecordingHook(events=events)

        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[hook],
        )

        await compactor.maybe_compact(long_messages)

        assert len(events) == 2
        assert events[0] == "test:start"
        assert events[1].startswith("test:end:")

    @pytest.mark.asyncio
    async def test_no_hooks_when_no_compaction(
        self,
        sample_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Hooks should NOT be called when under threshold."""
        events: list[str] = []
        hook = RecordingHook(events=events)

        compactor = ContextCompactor(
            max_context_tokens=1_000_000,  # Very high, no compaction
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[hook],
        )

        await compactor.maybe_compact(sample_messages)

        assert events == []  # No hooks called

    @pytest.mark.asyncio
    async def test_empty_hooks_list_works(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Compaction should work with no hooks."""
        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[],  # Empty list
        )

        result = await compactor.maybe_compact(long_messages)

        assert len(result) == 2  # Compaction still works
        assert compactor.compactions_performed == 1


# =============================================================================
# CompactionResult Tests
# =============================================================================


class TestCompactionResult:
    """Tests for CompactionResult data."""

    @pytest.mark.asyncio
    async def test_hook_receives_correct_result(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Hook should receive accurate CompactionResult."""
        received_result: CompactionResult | None = None

        @dataclass
        class CaptureHook:
            async def on_start(self) -> None:
                pass

            async def on_end(self, result: CompactionResult) -> None:
                nonlocal received_result
                received_result = result

        original_tokens = mock_counter.count_messages(long_messages)

        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=3),
            token_counter=mock_counter,
            hooks=[CaptureHook()],
        )

        result = await compactor.maybe_compact(long_messages)

        assert received_result is not None
        assert received_result.original_tokens == original_tokens
        assert received_result.compacted_tokens == mock_counter.count_messages(result)
        assert received_result.tokens_saved == original_tokens - received_result.compacted_tokens
        assert received_result.original_message_count == len(long_messages)
        assert received_result.compacted_message_count == len(result)
        assert received_result.strategy_name == "KeepRecentMessages"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestHookErrors:
    """Tests for hook error handling."""

    @pytest.mark.asyncio
    async def test_hook_error_propagates(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """If a hook raises, the error should propagate."""

        @dataclass
        class FailingHook:
            async def on_start(self) -> None:
                raise ValueError("Hook failed!")

            async def on_end(self, result: CompactionResult) -> None:
                pass

        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[FailingHook()],
        )

        with pytest.raises(ValueError, match="Hook failed!"):
            await compactor.maybe_compact(long_messages)

    @pytest.mark.asyncio
    async def test_end_hook_error_propagates(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """If on_end hook raises, the error should propagate."""

        @dataclass
        class FailingEndHook:
            async def on_start(self) -> None:
                pass

            async def on_end(self, result: CompactionResult) -> None:
                raise RuntimeError("End hook failed!")

        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[FailingEndHook()],
        )

        with pytest.raises(RuntimeError, match="End hook failed!"):
            await compactor.maybe_compact(long_messages)


# =============================================================================
# Built-in Hook Tests
# =============================================================================


class TestLoggingHook:
    """Tests for the built-in LoggingHook."""

    @pytest.mark.asyncio
    async def test_logging_hook_prints(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
        capsys,
    ):
        """LoggingHook should print to stdout."""
        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[LoggingHook(prefix="[Test]")],
        )

        await compactor.maybe_compact(long_messages)

        captured = capsys.readouterr()
        assert "[Test] Compaction started..." in captured.out
        assert "[Test] Compaction complete:" in captured.out
        assert "saved" in captured.out


class TestCallbackHook:
    """Tests for the built-in CallbackHook."""

    @pytest.mark.asyncio
    async def test_callback_hook_calls_functions(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """CallbackHook should call the provided functions."""
        events: list[str] = []

        async def on_start():
            events.append("callback_start")

        async def on_end(result: CompactionResult):
            events.append(f"callback_end:{result.tokens_saved}")

        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[CallbackHook(on_start_callback=on_start, on_end_callback=on_end)],
        )

        await compactor.maybe_compact(long_messages)

        assert events[0] == "callback_start"
        assert events[1].startswith("callback_end:")

    @pytest.mark.asyncio
    async def test_callback_hook_optional_callbacks(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """CallbackHook should work with only one callback provided."""
        events: list[str] = []

        async def on_end(result: CompactionResult):
            events.append("end_only")

        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[CallbackHook(on_end_callback=on_end)],  # No start callback
        )

        await compactor.maybe_compact(long_messages)

        assert events == ["end_only"]


# =============================================================================
# Streaming Order Simulation Tests
# =============================================================================


class TestStreamingOrderSimulation:
    """Tests simulating the streaming use case."""

    @pytest.mark.asyncio
    async def test_hooks_complete_before_stream_starts(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """
        Verify hooks fire in correct order relative to 'tokens'.

        This simulates the real use case where:
        1. Compaction starts (hooks fire)
        2. Compaction completes (hooks fire)
        3. THEN streaming tokens begin
        """
        events: list[tuple[str, str | int]] = []

        @dataclass
        class StreamingHook:
            async def on_start(self) -> None:
                events.append(("hook", "start"))

            async def on_end(self, result: CompactionResult) -> None:
                events.append(("hook", "end"))

        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[StreamingHook()],
        )

        # Simulate: compaction happens, THEN stream tokens arrive
        await compactor.maybe_compact(long_messages)

        # Simulate tokens arriving AFTER compaction completes
        for i in range(5):
            events.append(("token", i))

        # Verify order: all hook events before any tokens
        assert events[0] == ("hook", "start")
        assert events[1] == ("hook", "end")
        assert all(e[0] == "token" for e in events[2:])

    @pytest.mark.asyncio
    async def test_multiple_compactions_interleaved(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Test multiple compaction cycles with hooks."""
        events: list[str] = []
        hook = RecordingHook(events=events)

        compactor = ContextCompactor(
            max_context_tokens=50,
            strategy=KeepRecentMessages(keep_count=2),
            token_counter=mock_counter,
            hooks=[hook],
        )

        # First compaction
        await compactor.maybe_compact(long_messages)
        events.append("stream_1")

        # Second compaction (with new long messages)
        await compactor.maybe_compact(long_messages)
        events.append("stream_2")

        # Verify pattern: start, end, stream, start, end, stream
        assert events[0] == "test:start"
        assert events[1].startswith("test:end:")
        assert events[2] == "stream_1"
        assert events[3] == "test:start"
        assert events[4].startswith("test:end:")
        assert events[5] == "stream_2"
