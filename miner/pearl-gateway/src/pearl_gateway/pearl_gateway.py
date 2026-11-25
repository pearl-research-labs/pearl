"""Main PearlGateway application wiring."""

from miner_utils import get_logger

from pearl_gateway.config import load_config
from pearl_gateway.miner_rpc.server import MinerRpcServer
from pearl_gateway.pearl_client import PearlNodeClient
from pearl_gateway.pool_stratum_client import PoolStratumClient
from pearl_gateway.proof_pool import ProofPool
from pearl_gateway.scheduler import TemplateScheduler
from pearl_gateway.submission_service import PoolSubmissionService, SubmissionService
from pearl_gateway.work_cache import WorkCache


class PearlGateway:
    """Orchestrates gateway components in solo (pearld) or pool (Stratum) mode."""

    def __init__(self, debug_mode: bool = False) -> None:
        self.config = load_config()
        self.logger = get_logger(__name__)
        self.work_cache = WorkCache()
        self.proof_pool = ProofPool()
        # Mode-specific components; the other mode's stay None.
        self.pearl_client: PearlNodeClient | None = None
        self.scheduler: TemplateScheduler | None = None
        self.pool_client: PoolStratumClient | None = None

        pool = self.config.pool
        if pool is not None:
            self.pool_client = PoolStratumClient(
                pool, pool.address or self.config.pearl.mining_address, self.work_cache
            )
            self.submission_service: SubmissionService = PoolSubmissionService(
                self.pool_client, self.proof_pool, self.work_cache
            )
        else:
            self.pearl_client = PearlNodeClient(self.config.pearl)
            self.scheduler = TemplateScheduler(
                self.pearl_client, self.work_cache, self.config.pearl
            )
            self.submission_service = SubmissionService(
                self.pearl_client, self.proof_pool, debug_mode=debug_mode
            )

        self.miner_rpc = MinerRpcServer(
            self.work_cache, self.submission_service, self.config.miner_rpc
        )
        self.running = False

    async def start(self) -> None:
        if self.running:
            self.logger.warning("PearlGateway is already running")
            return
        # Order is load-bearing: entrypoint.sh polls the UDS socket for readiness,
        # so connect upstream, warm ProofPool, then bind RPC.
        if self.pool_client is not None:
            await self.pool_client.start()
        else:
            assert self.pearl_client is not None and self.scheduler is not None
            await self.pearl_client.start()
            await self.scheduler.start()

        await self.proof_pool.start()
        await self.miner_rpc.start()
        self.running = True
        self.logger.info("PearlGateway started successfully")

    async def stop(self) -> None:
        if not self.running:
            self.logger.warning("PearlGateway is not running")
            return
        self.running = False
        # Stop intake first so no proof is orphaned or submitted after client teardown.
        await self.miner_rpc.stop()
        await self.proof_pool.stop()
        if self.pool_client is not None:
            await self.pool_client.stop()
        else:
            assert self.scheduler is not None and self.pearl_client is not None
            await self.scheduler.stop()
            await self.pearl_client.stop()
        self.logger.info("PearlGateway stopped")
