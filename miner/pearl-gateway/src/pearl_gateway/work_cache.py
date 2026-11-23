import asyncio
import time

from miner_utils import get_logger

from pearl_gateway.comm.dataclasses import BlockTemplate, MiningJob, MiningPausedError

logger = get_logger(__name__)


class WorkCache:
    """
    Caches the latest block template and provides mining jobs to miners.
    Acts as an in-memory store between the Pearl node (or pool) and the miner.
    """

    def __init__(self):
        self.current_template: BlockTemplate | None = None
        self.last_update_time: float = 0
        self.lock = asyncio.Lock()  # For thread-safe access to template

    async def update_template(self, template: BlockTemplate, force: bool = False) -> bool:
        """
        Update the cached block template if it's different from current.
        Returns True if template was updated, False if unchanged.

        force=True swaps unconditionally, past the prev-block-hash dedup -- used for a
        Stratum clean_jobs notify, where the pool abandons prior work even on the same block.
        """
        async with self.lock:
            is_new = (
                force
                or self.current_template is None
                or template.header.previous_block_hash
                != self.current_template.header.previous_block_hash
            )

            if is_new:
                label = f"job {template.job_id}" if template.job_id else f"height {template.height}"
                logger.info(f"Updating block template to {label}")
                self.current_template = template
                self.last_update_time = time.time()
                return True
            else:
                age = time.time() - self.last_update_time
                logger.debug(f"Template unchanged (age: {age:.2f}s)")
                return False

    async def get_mining_job(self) -> MiningJob:
        """
        Get current mining job for a miner.
        Raises MiningPausedError if no valid template is available.
        """
        async with self.lock:
            if self.current_template is None:
                logger.warning("No block template available")
                raise MiningPausedError("no block template available")

            # Return a job with the current template
            return MiningJob.from_template(template=self.current_template)

    async def update_share_target(self, share_target: int, share_bits: int | None = None) -> None:
        """Update the pool share target on the current template, if one is cached."""
        async with self.lock:
            if self.current_template is not None:
                self.current_template.share_target = share_target
                self.current_template.share_bits = share_bits

    async def clear(self) -> None:
        """Drop the cached template (pool session ended); miners see mining paused."""
        async with self.lock:
            self.current_template = None

    async def get_template_age(self) -> float | None:
        """Get the age of the current template in seconds."""
        async with self.lock:
            if self.current_template is None:
                return None
            return time.time() - self.last_update_time
