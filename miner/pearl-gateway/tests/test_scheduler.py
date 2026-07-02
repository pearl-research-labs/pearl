"""
Test template scheduler functionality.
"""

from unittest.mock import AsyncMock, patch

import pytest
from pearl_gateway.config import PearlConfig
from pearl_gateway.scheduler import TemplateScheduler


@pytest.fixture
def mock_pearl_client():
    """Create a mock PearlNodeClient."""
    client = AsyncMock()
    client.get_block_template = AsyncMock()
    return client


@pytest.fixture
def mock_work_cache():
    """Create a mock WorkCache."""
    cache = AsyncMock()
    cache.update_template = AsyncMock(return_value=True)
    return cache


@pytest.fixture
async def scheduler(mock_pearl_client, mock_work_cache, sample_pearl_config):
    """Create a TemplateScheduler instance for testing."""
    scheduler_instance = TemplateScheduler(mock_pearl_client, mock_work_cache, sample_pearl_config)
    yield scheduler_instance
    # Ensure scheduler is properly stopped after each test
    if scheduler_instance.running:
        await scheduler_instance.stop()


class TestSchedulerStartStop:
    """Test scheduler start and stop functionality."""

    @pytest.mark.asyncio
    async def test_start_scheduler(self, scheduler, mock_pearl_client, sample_block_template):
        """Test starting the scheduler."""
        mock_pearl_client.get_block_template.return_value = sample_block_template

        await scheduler.start()

        assert scheduler.running is True
        assert scheduler.refresh_task is not None
        mock_pearl_client.get_block_template.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_scheduler_already_running(
        self, scheduler, mock_pearl_client, sample_block_template
    ):
        """Test starting scheduler when already running."""
        mock_pearl_client.get_block_template.return_value = sample_block_template

        # Start once
        await scheduler.start()
        call_count_after_first_start = mock_pearl_client.get_block_template.call_count

        # Start again
        await scheduler.start()

        # Should not call get_block_template again
        assert mock_pearl_client.get_block_template.call_count == call_count_after_first_start

    @pytest.mark.asyncio
    async def test_stop_scheduler(self, scheduler, mock_pearl_client, sample_block_template):
        """Test stopping the scheduler."""
        mock_pearl_client.get_block_template.return_value = sample_block_template

        # Start first
        await scheduler.start()
        assert scheduler.running is True

        # Stop
        await scheduler.stop()

        assert scheduler.running is False
        assert scheduler.refresh_task is None

    @pytest.mark.asyncio
    async def test_stop_scheduler_not_running(self, scheduler):
        """Test stopping scheduler when not running."""
        await scheduler.stop()

        assert scheduler.running is False
        assert scheduler.refresh_task is None


class TestTemplateRefresh:
    """Test template refresh functionality."""

    @pytest.mark.asyncio
    async def test_refresh_template_success(
        self, scheduler, mock_pearl_client, mock_work_cache, sample_block_template
    ):
        """Test successful template refresh."""
        mock_pearl_client.get_block_template.return_value = sample_block_template
        mock_work_cache.update_template.return_value = True

        with patch("time.time", return_value=1000.0):
            result = await scheduler.refresh_template()

        assert result is True
        assert scheduler.last_refresh_time == 1000.0
        mock_pearl_client.get_block_template.assert_called_once()
        mock_work_cache.update_template.assert_called_once_with(sample_block_template)

    @pytest.mark.asyncio
    async def test_refresh_template_no_update(
        self, scheduler, mock_pearl_client, mock_work_cache, sample_block_template
    ):
        """Test template refresh when cache doesn't update."""
        mock_pearl_client.get_block_template.return_value = sample_block_template
        mock_work_cache.update_template.return_value = False

        result = await scheduler.refresh_template()

        assert result is False
        mock_work_cache.update_template.assert_called_once_with(sample_block_template)

    @pytest.mark.asyncio
    async def test_refresh_template_pearl_client_error(
        self, scheduler, mock_pearl_client, mock_work_cache
    ):
        """Test template refresh when pearl client fails."""
        mock_pearl_client.get_block_template.side_effect = ConnectionError("Connection failed")

        result = await scheduler.refresh_template()

        assert result is False
        mock_work_cache.update_template.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_template_work_cache_error(
        self, scheduler, mock_pearl_client, mock_work_cache, sample_block_template
    ):
        """Test template refresh when work cache fails."""
        mock_pearl_client.get_block_template.return_value = sample_block_template
        mock_work_cache.update_template.side_effect = Exception("Cache error")

        result = await scheduler.refresh_template()

        assert result is False


class TestPeriodicRefresh:
    """Test periodic refresh functionality."""

    @pytest.mark.asyncio
    async def test_periodic_refresh_cycle(
        self, scheduler, mock_pearl_client, mock_work_cache, sample_block_template
    ):
        """Test periodic refresh starts the periodic task."""
        mock_pearl_client.get_block_template.return_value = sample_block_template

        await scheduler.start()

        # Verify the periodic task was created
        assert scheduler.refresh_task is not None
        assert not scheduler.refresh_task.cancelled()

        # Stop the scheduler to clean up
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_periodic_refresh_handles_exceptions(
        self, scheduler, mock_pearl_client, mock_work_cache
    ):
        """Test that periodic refresh handles exceptions gracefully."""
        # Make get_block_template fail initially, then succeed
        mock_pearl_client.get_block_template.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
        ]

        # Start scheduler (this will call refresh_template once during start)
        await scheduler.start()

        # Manually call refresh_template to test exception handling
        result1 = await scheduler.refresh_template()
        result2 = await scheduler.refresh_template()

        # Both should return False due to exceptions
        assert result1 is False
        assert result2 is False

        # Should have attempted to refresh multiple times
        assert mock_pearl_client.get_block_template.call_count >= 3  # 1 from start + 2 manual calls

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_periodic_refresh_cancellation(
        self, scheduler, mock_pearl_client, sample_block_template
    ):
        """Test that periodic refresh can be cancelled."""
        mock_pearl_client.get_block_template.return_value = sample_block_template

        # Start the scheduler
        await scheduler.start()
        assert scheduler.refresh_task is not None
        assert not scheduler.refresh_task.cancelled()

        # Stop the scheduler
        await scheduler.stop()

        # Task should be cancelled
        assert scheduler.refresh_task is None


class TestSchedulerIntegration:
    """Test TemplateScheduler integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_scheduler_lifecycle(
        self, scheduler, mock_pearl_client, mock_work_cache, sample_block_template
    ):
        """Test complete scheduler lifecycle."""
        mock_pearl_client.get_block_template.return_value = sample_block_template

        # 1. Initial state
        assert scheduler.running is False
        assert scheduler.refresh_task is None

        # 2. Start scheduler
        await scheduler.start()
        assert scheduler.running is True
        assert scheduler.refresh_task is not None

        # 3. Verify initial refresh happened
        mock_pearl_client.get_block_template.assert_called()
        mock_work_cache.update_template.assert_called()

        # 4. Stop scheduler
        await scheduler.stop()
        assert scheduler.running is False
        assert scheduler.refresh_task is None

    @pytest.mark.asyncio
    async def test_scheduler_restart(
        self, scheduler, mock_pearl_client, mock_work_cache, sample_block_template
    ):
        """Test restarting the scheduler."""
        mock_pearl_client.get_block_template.return_value = sample_block_template

        # Start
        await scheduler.start()
        first_task = scheduler.refresh_task

        # Stop
        await scheduler.stop()

        # Start again
        await scheduler.start()
        second_task = scheduler.refresh_task

        # Should be a different task
        assert first_task != second_task
        assert scheduler.running is True

    @pytest.mark.asyncio
    async def test_scheduler_with_different_intervals(
        self, mock_pearl_client, mock_work_cache, mining_address
    ):
        """Test scheduler behavior with different refresh intervals."""
        # Test with very short interval
        short_config = PearlConfig(
            mining_address=mining_address,
            rpc_url="https://127.0.0.1:18334",
            rpc_user="test_user",
            rpc_password="test_password",
            refresh_interval_seconds=1,
        )

        short_scheduler = TemplateScheduler(mock_pearl_client, mock_work_cache, short_config)
        assert short_scheduler.refresh_interval == 1

        # Test with longer interval
        long_config = PearlConfig(
            mining_address=mining_address,
            rpc_url="https://127.0.0.1:18334",
            rpc_user="test_user",
            rpc_password="test_password",
            refresh_interval_seconds=60,
        )

        long_scheduler = TemplateScheduler(mock_pearl_client, mock_work_cache, long_config)
        assert long_scheduler.refresh_interval == 60

        # Clean up any started schedulers
        if short_scheduler.running:
            await short_scheduler.stop()
        if long_scheduler.running:
            await long_scheduler.stop()

    @pytest.mark.asyncio
    async def test_scheduler_error_recovery(
        self, scheduler, mock_pearl_client, mock_work_cache, sample_block_template
    ):
        """Test scheduler recovery from errors."""
        # Set up alternating success/failure
        mock_pearl_client.get_block_template.side_effect = [
            sample_block_template,  # Initial success
            Exception("Temporary failure"),  # Failure
            sample_block_template,  # Recovery
        ]

        # Start scheduler
        await scheduler.start()

        # Manually trigger additional refreshes
        await scheduler.refresh_template()  # This should fail
        result = await scheduler.refresh_template()  # This should succeed

        assert result is True
        assert mock_pearl_client.get_block_template.call_count == 3

    @pytest.mark.asyncio
    async def test_scheduler_metrics_tracking(
        self, scheduler, mock_pearl_client, mock_work_cache, sample_block_template
    ):
        """Test that scheduler tracks refresh metrics."""
        mock_pearl_client.get_block_template.return_value = sample_block_template

        with patch("time.time", return_value=1234.5):
            await scheduler.refresh_template()

        assert scheduler.last_refresh_time == 1234.5
