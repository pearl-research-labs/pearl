import asyncio
import time
from typing import NamedTuple

from miner_utils import get_logger

from pearl_gateway.comm.dataclasses import BlockTemplate, MiningJob, MiningPausedError

logger = get_logger(__name__)


class _TemplateIdentity(NamedTuple):
    height: int
    version: int
    previous_block_hash: bytes
    merkle_root: bytes
    target_bits: int
    required_cert_version: int


def _template_identity(template: BlockTemplate) -> _TemplateIdentity:
    """Fields that make cached miner work materially different.

    Timestamp is omitted because the node refreshes it without regenerating a
    block template, and updating on every tick would reject in-flight proofs.
    """
    header = template.header
    return _TemplateIdentity(
        height=template.height,
        version=header.version,
        previous_block_hash=header.previous_block_hash,
        merkle_root=header.merkle_root,
        target_bits=header.target_bits,
        required_cert_version=int(template.required_cert_version),
    )


class WorkCache:
    """
    Caches the latest block template and provides mining jobs to miners.
    Acts as an in-memory store between the Pearl node and the miner.
    """

    def __init__(self):
        self.current_template: BlockTemplate | None = None
        self.last_update_time: float = 0
        self.lock = asyncio.Lock()  # For thread-safe access to template

    async def update_template(self, template: BlockTemplate) -> bool:
        """
        Update the cached block template if it's different from current.
        Returns True if template was updated, False if unchanged.
        """
        async with self.lock:
            current_template = self.current_template
            if current_template is None:
                is_new = True
            else:
                is_new = _template_identity(template) != _template_identity(current_template)

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
            return MiningJob.from_template(template=self.current_template)

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
