from miner_utils import get_logger

from pearl_gateway.comm.dataclasses import BlockTemplate, MiningJob, MiningPausedError

logger = get_logger(__name__)


class WorkCache:
    """In-memory cache of the latest block template for the single-threaded asyncio loop."""

    def __init__(self) -> None:
        self.current_template: BlockTemplate | None = None

    async def update_template(self, template: BlockTemplate, force: bool = False) -> bool:
        """Update the cached template when the tip changes or ``force`` is set.

        ``force=True`` swaps unconditionally past the prev-block-hash dedup — used
        for a Stratum ``clean_jobs`` or new-job notify on the same block.
        """
        is_new = (
            force
            or self.current_template is None
            or template.header.previous_block_hash
            != self.current_template.header.previous_block_hash
        )
        if not is_new:
            return False
        label = f"job {template.job_id}" if template.job_id else f"height {template.height}"
        logger.info(f"Updating block template to {label}")
        self.current_template = template
        return True

    async def get_mining_job(self) -> MiningJob:
        if self.current_template is None:
            logger.warning("No block template available")
            raise MiningPausedError("no block template available")
        return MiningJob.from_template(template=self.current_template)

    async def update_share_target(self, share_target: int, share_bits: int | None = None) -> None:
        """Update the pool share target on the current template, if one is cached."""
        if self.current_template is not None:
            self.current_template.share_target = share_target
            self.current_template.share_bits = share_bits

    async def clear(self) -> None:
        """Drop the cached template (pool session ended); miners see mining paused."""
        self.current_template = None
