import asyncio
import contextlib

from miner_utils import get_logger

from pearl_gateway.config import PearlConfig
from pearl_gateway.pearl_client import PearlNodeClient
from pearl_gateway.work_cache import WorkCache

logger = get_logger(__name__)


class TemplateScheduler:
    """Periodically polls pearld for the latest block template and updates WorkCache."""

    def __init__(
        self,
        pearl_client: PearlNodeClient,
        work_cache: WorkCache,
        config: PearlConfig,
    ) -> None:
        self.pearl_client = pearl_client
        self.work_cache = work_cache
        self.refresh_interval = config.refresh_interval_seconds
        self.running = False
        self.refresh_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        await self.refresh_template()
        self.refresh_task = asyncio.create_task(self._periodic_refresh())
        logger.info(f"Template scheduler started with refresh interval {self.refresh_interval}s")

    async def stop(self) -> None:
        self.running = False
        if self.refresh_task is not None:
            self.refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.refresh_task
            self.refresh_task = None
        logger.info("Template scheduler stopped")

    async def _periodic_refresh(self) -> None:
        try:
            while self.running:
                await asyncio.sleep(self.refresh_interval)
                if not self.running:
                    break
                await self.refresh_template()
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False

    async def refresh_template(self) -> bool:
        try:
            template = await self.pearl_client.get_block_template()
            updated = await self.work_cache.update_template(template)
            if updated:
                logger.info(f"Template refreshed successfully (height: {template.height})")
            return updated
        except Exception as e:
            logger.error(f"Failed to refresh template: {e}")
            return False
