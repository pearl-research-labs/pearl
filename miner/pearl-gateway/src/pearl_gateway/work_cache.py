import asyncio
import time

from miner_utils import get_logger

from pearl_gateway.comm.dataclasses import BlockTemplate, MiningJob, MiningPausedError

logger = get_logger(__name__)


class WorkCache:
    """
    Caches the latest block template and provides mining jobs to miners.
    Acts as an in-memory store between the Pearl node and the miner.
    """

    def __init__(
        self,
        mining_config_bytes: bytes | None = None,
        matrix_m: int | None = None,
        matrix_n: int | None = None,
    ):
        self.current_template: BlockTemplate | None = None
        self.last_update_time: float = 0
        self.lock = asyncio.Lock()  # For thread-safe access to template
        self.mining_config_bytes = mining_config_bytes
        self.matrix_m = matrix_m
        self.matrix_n = matrix_n

    async def update_template(self, template: BlockTemplate) -> bool:
        """
        Update the cached block template if it's different from current.
        Returns True if template was updated, False if unchanged.
        """
        async with self.lock:
            is_new = (
                self.current_template is None
                or template.header.previous_block_hash
                != self.current_template.header.previous_block_hash
            )

            if is_new:
                logger.info(f"Updating block template to height {template.height}")
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
            return MiningJob.from_template(
                template=self.current_template,
                mining_config_bytes=self.mining_config_bytes,
                matrix_m=self.matrix_m,
                matrix_n=self.matrix_n,
            )

    async def get_template_age(self) -> float | None:
        """Get the age of the current template in seconds."""
        async with self.lock:
            if self.current_template is None:
                return None
            return time.time() - self.last_update_time

    async def invalidate(self) -> None:
        """Invalidate the current template, forcing a refresh on next request."""
        async with self.lock:
            logger.info("Invalidating current template")
            self.current_template = None
            self.last_update_time = 0
