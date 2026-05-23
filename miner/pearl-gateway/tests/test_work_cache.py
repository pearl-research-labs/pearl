"""
Test work cache functionality.
"""

import asyncio
from unittest.mock import patch

import pytest
from bitcoinutils.transactions import Transaction
from pearl_gateway.blockchain_utils.pearl_header import PearlHeader
from pearl_gateway.comm.dataclasses import BlockTemplate, MiningJob, MiningPausedError
from pearl_gateway.work_cache import WorkCache
from pearl_mining import IncompleteBlockHeader


# Work cache specific fixtures
@pytest.fixture
def work_cache():
    """Create a WorkCache instance for testing."""
    return WorkCache()


@pytest.fixture
def different_block_template():
    """Create a different BlockTemplate for testing updates."""
    return BlockTemplate(
        header=PearlHeader(
            incomplete_header=IncompleteBlockHeader(
                version=0x20000000,
                prev_block=bytes.fromhex("11" * 32),  # Different previous block hash
                merkle_root=bytes.fromhex("1234abcd" * 8),  # Different merkle root
                timestamp=1715748999,  # Different time
                nbits=0x207FFFFF,
            ),
        ),
        height=12346,  # Different height
        coinbase_tx=Transaction.from_raw(
            "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff0502a1050101ffffffff0100f2052a01000000434104678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5fac00000000"
        ),
        raw_transactions=[],
    )


class TestTemplateUpdate:
    """Test template update functionality."""

    @pytest.mark.asyncio
    async def test_update_template_first_time(self, work_cache, sample_block_template):
        """Test updating template for the first time."""
        with patch("time.time", return_value=1000.0):
            result = await work_cache.update_template(sample_block_template)

        assert result is True
        assert work_cache.current_template == sample_block_template
        assert work_cache.last_update_time == 1000.0

    @pytest.mark.asyncio
    async def test_update_template_same_template(self, work_cache, sample_block_template):
        """Test updating with the same template doesn't change anything."""
        # First update
        with patch("time.time", return_value=1000.0):
            result1 = await work_cache.update_template(sample_block_template)

        # Second update with same template
        with patch("time.time", return_value=1005.0):
            result2 = await work_cache.update_template(sample_block_template)

        assert result1 is True
        assert result2 is False

    @pytest.mark.asyncio
    async def test_update_template_different_template(
        self, work_cache, sample_block_template, different_block_template
    ):
        """Test updating with a different template."""
        # First update
        with patch("time.time", return_value=1000.0):
            await work_cache.update_template(sample_block_template)

        # Update with different template
        with patch("time.time", return_value=1010.0):
            result = await work_cache.update_template(different_block_template)

        assert result is True
        assert work_cache.current_template == different_block_template
        assert work_cache.last_update_time == 1010.0

    @pytest.mark.asyncio
    async def test_update_template_thread_safety(
        self, work_cache, sample_block_template, different_block_template
    ):
        """Test that template updates are thread-safe."""

        async def update_template_a():
            await work_cache.update_template(sample_block_template)

        async def update_template_b():
            await work_cache.update_template(different_block_template)

        # Run concurrent updates
        await asyncio.gather(update_template_a(), update_template_b())

        # One of the templates should be set
        assert work_cache.current_template is not None
        assert work_cache.current_template in [
            sample_block_template,
            different_block_template,
        ]


class TestMiningJob:
    """Test mining job functionality."""

    @pytest.mark.asyncio
    async def test_get_mining_job_no_template(self, work_cache):
        """Test getting mining job when no template is available."""
        with pytest.raises(MiningPausedError, match="no block template available"):
            await work_cache.get_mining_job()

    @pytest.mark.asyncio
    async def test_get_mining_job_with_template(self, work_cache, sample_block_template):
        """Test getting mining job with available template."""
        await work_cache.update_template(sample_block_template)

        job = await work_cache.get_mining_job()

        assert isinstance(job, MiningJob)
        assert (
            job.incomplete_header_bytes
            == sample_block_template.header.serialize_without_proof_commitment()
        )
        assert job.target == sample_block_template.target

    @pytest.mark.asyncio
    async def test_get_mining_job_multiple_calls(self, work_cache, sample_block_template):
        """Test getting multiple mining jobs from same template."""
        await work_cache.update_template(sample_block_template)

        job1 = await work_cache.get_mining_job()
        job2 = await work_cache.get_mining_job()
        job3 = await work_cache.get_mining_job()

        # All jobs should be identical since they're from the same template
        assert (
            job1.incomplete_header_bytes
            == job2.incomplete_header_bytes
            == job3.incomplete_header_bytes
        )
        assert job1.target == job2.target == job3.target

    @pytest.mark.asyncio
    async def test_get_mining_job_concurrent_access(self, work_cache, sample_block_template):
        """Test that mining job generation is thread-safe under concurrent access."""
        await work_cache.update_template(sample_block_template)

        async def get_job():
            return await work_cache.get_mining_job()

        # Get 10 jobs concurrently
        jobs = await asyncio.gather(*[get_job() for _ in range(10)])

        # All jobs should be identical
        for job in jobs:
            assert (
                job.incomplete_header_bytes
                == sample_block_template.header.serialize_without_proof_commitment()
            )
            assert job.target == sample_block_template.target


class TestCacheInvalidation:
    """Test cache invalidation functionality."""

    @pytest.mark.asyncio
    async def test_invalidate_empty_cache(self, work_cache):
        """Test invalidating an empty cache."""
        await work_cache.invalidate()

        assert work_cache.current_template is None
        assert work_cache.last_update_time == 0

    @pytest.mark.asyncio
    async def test_invalidate_cache_with_template(self, work_cache, sample_block_template):
        """Test invalidating cache with existing template."""
        # Set up cache with template
        await work_cache.update_template(sample_block_template)
        await work_cache.get_mining_job()  # Get a job

        # Invalidate
        await work_cache.invalidate()

        assert work_cache.current_template is None
        assert work_cache.last_update_time == 0

    @pytest.mark.asyncio
    async def test_get_mining_job_after_invalidation(self, work_cache, sample_block_template):
        """Test getting mining job after invalidation raises error."""
        await work_cache.update_template(sample_block_template)
        await work_cache.invalidate()

        with pytest.raises(MiningPausedError, match="no block template available"):
            await work_cache.get_mining_job()


class TestCacheIntegration:
    """Test WorkCache integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_cache_workflow(
        self, work_cache, sample_block_template, different_block_template
    ):
        """Test a complete cache workflow."""
        # 1. Start with empty cache
        assert work_cache.current_template is None

        # 2. Update with first template
        result1 = await work_cache.update_template(sample_block_template)
        assert result1 is True

        # 3. Get mining jobs
        job1 = await work_cache.get_mining_job()
        job2 = await work_cache.get_mining_job()

        # Jobs should be identical since they're from the same template
        assert job1.incomplete_header_bytes == job2.incomplete_header_bytes
        assert job1.target == job2.target

        # 4. Update with same template (should not update)
        result2 = await work_cache.update_template(sample_block_template)
        assert result2 is False

        # 5. Update with different template
        result3 = await work_cache.update_template(different_block_template)
        assert result3 is True
        assert work_cache.current_template == different_block_template

        # 6. Get job from new template
        job3 = await work_cache.get_mining_job()
        assert (
            job3.incomplete_header_bytes
            == different_block_template.header.serialize_without_proof_commitment()
        )
        assert job3.target == different_block_template.target

        # 7. Invalidate
        await work_cache.invalidate()
        assert work_cache.current_template is None

        # 8. Try to get job after invalidation
        with pytest.raises(MiningPausedError):
            await work_cache.get_mining_job()
