"""Main PearlGateway application wiring."""

from miner_utils import get_logger

from pearl_gateway.config import load_config
from pearl_gateway.miner_rpc.server import MinerRpcServer
from pearl_gateway.pearl_client import PearlNodeClient
from pearl_gateway.proof_pool import ProofPool
from pearl_gateway.scheduler import TemplateScheduler
from pearl_gateway.submission_service import SubmissionService
from pearl_gateway.work_cache import WorkCache


class PearlGateway:
    def __init__(self, debug_mode: bool = False) -> None:
        self.config = load_config()
        self.logger = get_logger(__name__)
        self.work_cache = WorkCache()
        self.pearl_client = PearlNodeClient(self.config.pearl)
        self.proof_pool = ProofPool()
        self.submission_service = SubmissionService(
            self.pearl_client, debug_mode=debug_mode, proof_pool=self.proof_pool
        )
        self.miner_rpc = MinerRpcServer(
            self.work_cache, self.submission_service, self.config.miner_rpc
        )
        self.scheduler = TemplateScheduler(self.pearl_client, self.work_cache, self.config.pearl)
        self.running = False

    async def start(self) -> None:
        if self.running:
            self.logger.warning("PearlGateway is already running")
            return
        # Order is load-bearing: entrypoint.sh polls the UDS socket for readiness,
        # so connect pearld, fetch initial template, and warm ProofPool before binding RPC.
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
        await self.scheduler.stop()
        await self.pearl_client.stop()
        self.logger.info("PearlGateway stopped")
