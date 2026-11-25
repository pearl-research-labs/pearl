"""
Test work cache functionality.
"""

import asyncio

import pytest
from bitcoinutils.transactions import Transaction
from pearl_gateway.blockchain_utils.pearl_header import PearlHeader
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
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
        required_cert_version=CertificateVersion.ZK_MOE,
    )


class TestTemplateUpdate:
    """Test template update functionality."""

    @pytest.mark.asyncio
    async def test_update_template_first_time(self, work_cache, sample_block_template):
        result = await work_cache.update_template(sample_block_template)

        assert result is True
        assert work_cache.current_template == sample_block_template

    @pytest.mark.asyncio
    async def test_update_template_same_template(self, work_cache, sample_block_template):
        result1 = await work_cache.update_template(sample_block_template)
        result2 = await work_cache.update_template(sample_block_template)

        assert result1 is True
        assert result2 is False

    @pytest.mark.asyncio
    async def test_update_template_different_template(
        self, work_cache, sample_block_template, different_block_template
    ):
        await work_cache.update_template(sample_block_template)
        result = await work_cache.update_template(different_block_template)

        assert result is True
        assert work_cache.current_template == different_block_template

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


class TestForceUpdate:
    """Test forcing a template swap past the prev-block-hash dedup (clean_jobs)."""

    @pytest.mark.asyncio
    async def test_force_updates_even_when_unchanged(self, work_cache, sample_block_template):
        assert await work_cache.update_template(sample_block_template) is True
        # Same template again -> normally deduped by prev-block-hash.
        assert await work_cache.update_template(sample_block_template) is False
        # force overrides the dedup so a clean_jobs notify always takes effect.
        assert await work_cache.update_template(sample_block_template, force=True) is True
        assert work_cache.current_template is sample_block_template


class TestShareTargetOverride:
    """Test pool share difficulty stored on BlockTemplate."""

    @pytest.mark.asyncio
    async def test_share_target_overrides_job_target(self, work_cache, sample_block_template):
        await work_cache.update_template(sample_block_template)
        network_target = sample_block_template.target
        sample_block_template.share_target = network_target * 100  # easier than network

        job = await work_cache.get_mining_job()
        assert job.target == network_target * 100

    @pytest.mark.asyncio
    async def test_no_share_target_uses_network_target(self, work_cache, sample_block_template):
        await work_cache.update_template(sample_block_template)

        job = await work_cache.get_mining_job()
        assert job.target == sample_block_template.target

    @pytest.mark.asyncio
    async def test_update_share_target_sets_current_template(
        self, work_cache, sample_block_template
    ):
        await work_cache.update_template(sample_block_template)

        await work_cache.update_share_target(12345)

        assert sample_block_template.share_target == 12345
        assert (await work_cache.get_mining_job()).target == 12345

    @pytest.mark.asyncio
    async def test_update_share_target_without_template_is_noop(self, work_cache):
        await work_cache.update_share_target(12345)

        assert work_cache.current_template is None


class TestCacheIntegration:
    """Test WorkCache integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_cache_workflow(
        self, work_cache, sample_block_template, different_block_template
    ):
        assert work_cache.current_template is None

        assert await work_cache.update_template(sample_block_template) is True
        job1 = await work_cache.get_mining_job()
        job2 = await work_cache.get_mining_job()
        assert job1.incomplete_header_bytes == job2.incomplete_header_bytes
        assert job1.target == job2.target

        assert await work_cache.update_template(sample_block_template) is False
        assert await work_cache.update_template(different_block_template) is True
        assert work_cache.current_template == different_block_template

        job3 = await work_cache.get_mining_job()
        assert (
            job3.incomplete_header_bytes
            == different_block_template.header.serialize_without_proof_commitment()
        )
        assert job3.target == different_block_template.target
