"""Counter-example tests (invariant tests).

These tests verify that certain things NEVER happen, regardless of input.
They are critical for ensuring the compactor is bulletproof.
"""

from __future__ import annotations

import copy

import pytest

from context_compactor import ContextCompactor
from context_compactor.strategies import (
    DropOldestUntilFits,
    KeepFirstLast,
    KeepRecentMessages,
    SlidingWindow,
)

from .conftest import MockMessage, MockTokenCounter


class TestNoCompactionWhenUnderThreshold:
    """Verify messages are NEVER compacted when under threshold."""

    @pytest.mark.asyncio
    async def test_no_compaction_under_threshold_keep_recent(
        self,
        sample_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """KeepRecentMessages should NOT compact under threshold."""
        compactor = ContextCompactor(
            max_context_tokens=1_000_000,
            strategy=KeepRecentMessages(keep_count=5),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact(sample_messages)

        assert result == sample_messages
        assert compactor.compactions_performed == 0

    @pytest.mark.asyncio
    async def test_no_compaction_under_threshold_keep_first_last(
        self,
        sample_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """KeepFirstLast should NOT compact under threshold."""
        compactor = ContextCompactor(
            max_context_tokens=1_000_000,
            strategy=KeepFirstLast(keep_first=2, keep_last=3),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact(sample_messages)

        assert result == sample_messages
        assert compactor.compactions_performed == 0


class TestMessageOrderPreserved:
    """Verify remaining messages ALWAYS maintain original relative order."""

    @pytest.mark.asyncio
    async def test_order_preserved_keep_recent(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """KeepRecentMessages MUST preserve order."""
        compactor = ContextCompactor(
            max_context_tokens=100,
            strategy=KeepRecentMessages(keep_count=5),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact(long_messages)

        # Check that result is a subsequence of original (maintains order)
        original_indices = [long_messages.index(m) for m in result]
        assert original_indices == sorted(original_indices)

    @pytest.mark.asyncio
    async def test_order_preserved_keep_first_last(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """KeepFirstLast MUST preserve order."""
        compactor = ContextCompactor(
            max_context_tokens=100,
            strategy=KeepFirstLast(keep_first=2, keep_last=3),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact(long_messages)

        original_indices = [long_messages.index(m) for m in result]
        assert original_indices == sorted(original_indices)


class TestNoMessageCorruption:
    """Verify messages are NEVER mutated during compaction."""

    @pytest.mark.asyncio
    async def test_no_mutation_keep_recent(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Original messages MUST NOT be mutated."""
        original = copy.deepcopy(long_messages)
        compactor = ContextCompactor(
            max_context_tokens=100,
            strategy=KeepRecentMessages(keep_count=5),
            token_counter=mock_counter,
        )

        await compactor.maybe_compact(long_messages)

        assert long_messages == original

    @pytest.mark.asyncio
    async def test_no_mutation_sliding_window(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Original messages MUST NOT be mutated with SlidingWindow."""
        original = copy.deepcopy(long_messages)
        compactor = ContextCompactor(
            max_context_tokens=100,
            strategy=SlidingWindow(),
            token_counter=mock_counter,
        )

        await compactor.maybe_compact(long_messages)

        assert long_messages == original


class TestTokenCountAccuracy:
    """Verify token counting behaves correctly."""

    def test_token_count_monotonic(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Adding messages MUST increase token count."""
        count_5 = mock_counter.count_messages(long_messages[:5])
        count_10 = mock_counter.count_messages(long_messages[:10])
        count_20 = mock_counter.count_messages(long_messages[:20])

        assert count_5 < count_10 < count_20

    def test_token_count_non_negative(
        self,
        sample_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Token count MUST never be negative."""
        for msg in sample_messages:
            assert mock_counter.count_single(msg) >= 0

    def test_empty_messages_zero_tokens(self, mock_counter: MockTokenCounter):
        """Empty message list MUST have zero tokens."""
        assert mock_counter.count_messages([]) == 0


class TestCompactionReducesTokens:
    """Verify compaction ALWAYS reduces token count when triggered."""

    @pytest.mark.asyncio
    async def test_compaction_reduces_tokens(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Compaction MUST reduce token count."""
        compactor = ContextCompactor(
            max_context_tokens=100,
            strategy=KeepRecentMessages(keep_count=5),
            token_counter=mock_counter,
        )

        original_tokens = mock_counter.count_messages(long_messages)
        result = await compactor.maybe_compact(long_messages)
        result_tokens = mock_counter.count_messages(result)

        assert result_tokens < original_tokens


class TestEmptyInputHandling:
    """Verify empty input is handled safely."""

    @pytest.mark.asyncio
    async def test_empty_messages_safe_keep_recent(self, mock_counter: MockTokenCounter):
        """Empty message list MUST NOT cause errors with KeepRecentMessages."""
        compactor = ContextCompactor(
            max_context_tokens=100,
            strategy=KeepRecentMessages(keep_count=5),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact([])

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_messages_safe_keep_first_last(self, mock_counter: MockTokenCounter):
        """Empty message list MUST NOT cause errors with KeepFirstLast."""
        compactor = ContextCompactor(
            max_context_tokens=100,
            strategy=KeepFirstLast(keep_first=2, keep_last=3),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact([])

        assert result == []


class TestSingleMessagePreserved:
    """Verify single message is ALWAYS preserved."""

    @pytest.mark.asyncio
    async def test_single_message_preserved_keep_recent(
        self,
        single_message: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Single message MUST be preserved with KeepRecentMessages."""
        compactor = ContextCompactor(
            max_context_tokens=1,  # Extremely small
            strategy=KeepRecentMessages(keep_count=1),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact(single_message)

        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_single_message_preserved_sliding_window(
        self,
        single_message: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Single message MUST be preserved with SlidingWindow."""
        compactor = ContextCompactor(
            max_context_tokens=1,
            strategy=SlidingWindow(),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact(single_message)

        assert len(result) >= 1


class TestStrategyContractHonored:
    """Verify strategies honor their contracts."""

    @pytest.mark.asyncio
    async def test_keep_recent_honors_count(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """KeepRecentMessages(N) MUST keep exactly N messages when possible."""
        strategy = KeepRecentMessages(keep_count=5)

        result = await strategy.compact(long_messages, 1000, mock_counter)

        assert len(result) == 5
        assert result == long_messages[-5:]

    @pytest.mark.asyncio
    async def test_keep_first_last_honors_bounds(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """KeepFirstLast MUST keep first N and last M."""
        strategy = KeepFirstLast(keep_first=2, keep_last=3)

        result = await strategy.compact(long_messages, 1000, mock_counter)

        assert result[:2] == long_messages[:2]
        assert result[-3:] == long_messages[-3:]

    @pytest.mark.asyncio
    async def test_drop_oldest_respects_minimum(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """DropOldestUntilFits MUST respect min_messages."""
        strategy = DropOldestUntilFits(min_messages=10)

        result = await strategy.compact(long_messages, 1, mock_counter)

        assert len(result) >= 10


class TestNoSilentFailures:
    """Verify errors are NEVER swallowed."""

    @pytest.mark.asyncio
    async def test_strategy_failure_raises(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Strategy errors MUST propagate, not be swallowed."""

        class FailingStrategy:
            async def compact(self, messages, target, counter):
                raise ValueError("Intentional failure")

        compactor = ContextCompactor(
            max_context_tokens=1,  # Very low to force compaction
            strategy=FailingStrategy(),
            token_counter=mock_counter,
        )

        with pytest.raises(ValueError, match="Intentional failure"):
            await compactor.maybe_compact(long_messages)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_exact_threshold(
        self,
        mock_counter: MockTokenCounter,
    ):
        """Messages exactly at threshold should NOT be compacted."""
        messages = [MockMessage(role="user", content="x" * 100)]
        tokens = mock_counter.count_messages(messages)

        compactor = ContextCompactor(
            max_context_tokens=tokens + 1,  # Just above
            trigger_at_percent=1.0,  # Trigger at 100%
            strategy=KeepRecentMessages(keep_count=1),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact(messages)

        assert result == messages

    @pytest.mark.asyncio
    async def test_zero_keep_count_handled(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """KeepRecentMessages with keep_count=0 should return empty."""
        strategy = KeepRecentMessages(keep_count=0)

        result = await strategy.compact(long_messages, 1000, mock_counter)

        assert result == []

    @pytest.mark.asyncio
    async def test_very_small_keep_count_handled(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """KeepRecentMessages with keep_count=1 should keep only 1 message."""
        strategy = KeepRecentMessages(keep_count=1)

        result = await strategy.compact(long_messages, 1000, mock_counter)

        assert len(result) == 1
        assert result[0] == long_messages[-1]

    @pytest.mark.asyncio
    async def test_large_message_content(
        self,
        large_message: MockMessage,
        mock_counter: MockTokenCounter,
    ):
        """Very large message content should be handled correctly."""
        messages = [large_message]
        tokens = mock_counter.count_messages(messages)

        assert tokens > 100  # Should have many tokens
        assert tokens == mock_counter.count_single(large_message)
