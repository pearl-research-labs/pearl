from miner_utils import get_logger

from pearl_gateway.comm.dataclasses import BlockTemplate, MiningJob, MiningPausedError

logger = get_logger(__name__)


class WorkCache:
    """In-memory cache of the latest block template for the single-threaded asyncio loop."""

    def __init__(self) -> None:
        self.current_template: BlockTemplate | None = None

    async def update_template(self, template: BlockTemplate) -> bool:
        """Update the cached template when the tip changes. Returns True if updated."""
        is_new = (
            self.current_template is None
            or template.header.previous_block_hash
            != self.current_template.header.previous_block_hash
        )
        if not is_new:
            return False
        logger.info(f"Updating block template to height {template.height}")
        self.current_template = template
        return True

    async def get_mining_job(self) -> MiningJob:
        if self.current_template is None:
            logger.warning("No block template available")
            raise MiningPausedError("no block template available")
        return MiningJob.from_template(template=self.current_template)
