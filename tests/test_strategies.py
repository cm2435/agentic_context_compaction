"""Tests for compaction strategies."""

from __future__ import annotations

import pytest

from context_compactor.strategies import (
    DropOldestUntilFits,
    KeepFirstLast,
    KeepRecentMessages,
    SlidingWindow,
)

from .conftest import MockMessage, MockTokenCounter


class TestKeepRecentMessages:
    """Tests for KeepRecentMessages strategy."""

    @pytest.mark.asyncio
    async def test_keeps_exact_count(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should keep exactly N recent messages."""
        strategy = KeepRecentMessages(keep_count=5)

        result = await strategy.compact(long_messages, 1000, mock_counter)

        assert len(result) == 5
        assert result == long_messages[-5:]

    @pytest.mark.asyncio
    async def test_no_change_under_count(
        self,
        sample_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should return unchanged if message count <= keep_count."""
        strategy = KeepRecentMessages(keep_count=20)

        result = await strategy.compact(sample_messages, 1000, mock_counter)

        assert result == sample_messages

    @pytest.mark.asyncio
    async def test_default_keep_count(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should use default keep_count of 10."""
        strategy = KeepRecentMessages()

        result = await strategy.compact(long_messages, 1000, mock_counter)

        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_preserves_order(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should preserve the order of kept messages."""
        strategy = KeepRecentMessages(keep_count=5)

        result = await strategy.compact(long_messages, 1000, mock_counter)

        # All messages should be from the original and in order
        for i, msg in enumerate(result):
            original_idx = long_messages.index(msg)
            if i > 0:
                prev_original_idx = long_messages.index(result[i - 1])
                assert original_idx > prev_original_idx


class TestDropOldestUntilFits:
    """Tests for DropOldestUntilFits strategy."""

    @pytest.mark.asyncio
    async def test_drops_until_fits(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should drop oldest messages until under target tokens."""
        strategy = DropOldestUntilFits(min_messages=2)
        target_tokens = 100  # Low to force significant dropping

        result = await strategy.compact(long_messages, target_tokens, mock_counter)

        result_tokens = mock_counter.count_messages(result)
        # Either under target or at minimum
        assert result_tokens <= target_tokens or len(result) == 2

    @pytest.mark.asyncio
    async def test_respects_min_messages(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should never drop below min_messages."""
        strategy = DropOldestUntilFits(min_messages=5)

        result = await strategy.compact(long_messages, 1, mock_counter)  # Impossibly low

        assert len(result) >= 5

    @pytest.mark.asyncio
    async def test_no_change_if_already_fits(
        self,
        sample_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should return unchanged if already under target."""
        strategy = DropOldestUntilFits()
        target_tokens = 1_000_000

        result = await strategy.compact(sample_messages, target_tokens, mock_counter)

        assert result == sample_messages


class TestKeepFirstLast:
    """Tests for KeepFirstLast strategy."""

    @pytest.mark.asyncio
    async def test_keeps_first_and_last(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should keep first N and last M messages."""
        strategy = KeepFirstLast(keep_first=2, keep_last=3)

        result = await strategy.compact(long_messages, 1000, mock_counter)

        assert result[:2] == long_messages[:2]
        assert result[-3:] == long_messages[-3:]
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_no_change_if_short(
        self,
        sample_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should return unchanged if total messages <= keep_first + keep_last."""
        strategy = KeepFirstLast(keep_first=5, keep_last=10)

        result = await strategy.compact(sample_messages, 1000, mock_counter)

        assert result == sample_messages

    @pytest.mark.asyncio
    async def test_drops_middle(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should drop middle messages."""
        strategy = KeepFirstLast(keep_first=2, keep_last=2)

        result = await strategy.compact(long_messages, 1000, mock_counter)

        # Middle messages should be gone
        middle_messages = long_messages[2:-2]
        for msg in middle_messages:
            assert msg not in result


class TestSlidingWindow:
    """Tests for SlidingWindow strategy."""

    @pytest.mark.asyncio
    async def test_fits_within_budget(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should keep messages that fit within token budget."""
        strategy = SlidingWindow(buffer_percent=0.1)
        target_tokens = 200

        result = await strategy.compact(long_messages, target_tokens, mock_counter)

        result_tokens = mock_counter.count_messages(result)
        effective_target = int(target_tokens * 0.9)
        assert result_tokens <= effective_target or len(result) == 1

    @pytest.mark.asyncio
    async def test_keeps_recent_first(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should prefer keeping most recent messages."""
        strategy = SlidingWindow()
        target_tokens = 200

        result = await strategy.compact(long_messages, target_tokens, mock_counter)

        # Last message should be in result
        assert result[-1] == long_messages[-1]

    @pytest.mark.asyncio
    async def test_at_least_one_message(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Should keep at least one message even if over budget."""
        strategy = SlidingWindow()

        result = await strategy.compact(long_messages, 1, mock_counter)  # Impossibly low

        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_empty_input(self, mock_counter: MockTokenCounter):
        """Should handle empty input gracefully."""
        strategy = SlidingWindow()

        result = await strategy.compact([], 1000, mock_counter)

        assert result == []
