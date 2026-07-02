"""
Main application class that orchestrates all components.
"""

import os
import tempfile

from miner_utils import get_logger

from pearl_gateway.config import MinerRpcConfig, load_config
from pearl_gateway.miner_rpc.server import MinerRpcServer
from pearl_gateway.pearl_client import PearlNodeClient
from pearl_gateway.proof_pool import ProofPool
from pearl_gateway.scheduler import TemplateScheduler
from pearl_gateway.submission_service import SubmissionService
from pearl_gateway.work_cache import WorkCache


class PearlGateway:
    """
    Main application class that orchestrates all components.
    """

    def __init__(
        self,
        use_temp_socket: bool = False,
        debug_mode: bool = False,
    ):
        """
        Initialize the PearlGateway.

        Args:
            use_temp_socket: If True, use a temporary file for the UDS socket path.
                           Useful for testing to avoid socket conflicts.
            debug_mode: Enable debug mode logging.
        """
        # Load configuration from environment variables and defaults
        self.config = load_config()

        # Set up logging
        self.logger = get_logger(__name__)

        # Track temp socket for cleanup
        self._temp_socket_path: str | None = None

        # Override socket path for testing isolation
        if use_temp_socket:
            self._temp_socket_path = tempfile.mktemp(suffix=".sock")
            self.config.miner_rpc = MinerRpcConfig(
                transport="uds",
                socket_path=self._temp_socket_path,
            )

        # Initialize components (but don't start them yet)
        self.work_cache = WorkCache()
        self.pearl_client = PearlNodeClient(self.config.pearl)
        self.proof_pool = ProofPool()
        self.submission_service = SubmissionService(
            self.pearl_client, self.proof_pool, debug_mode=debug_mode
        )
        self.miner_rpc = MinerRpcServer(
            self.work_cache,
            self.submission_service,
            self.config.miner_rpc,
            max_pending_submissions=self.config.proof.max_pending,
        )
        self.scheduler = TemplateScheduler(self.pearl_client, self.work_cache, self.config.pearl)

        self.running = False
        self.logger.info("PearlGateway initialized")

    async def start(self):
        """Start all components of the gateway."""
        if self.running:
            self.logger.warning("PearlGateway is already running")
            return

        self.logger.info("Starting PearlGateway")

        # Start Pearl client
        await self.pearl_client.__aenter__()

        # Start the scheduler (which will immediately fetch a block template)
        await self.scheduler.start()

        # Spawn and warm the ZK proof worker before accepting miner submissions
        await self.proof_pool.start()

        # Start the Miner RPC server
        await self.miner_rpc.start()

        self.running = True

        self.logger.info("PearlGateway started successfully")

    async def stop(self):
        """Gracefully stop all components."""
        if not self.running:
            self.logger.warning("PearlGateway is not running")
            return

        self.logger.info("Stopping PearlGateway")
        self.running = False

        # Stop components in reverse order
        await self.miner_rpc.stop()
        # Stop intake (miner_rpc) before the proving worker so no proof is left in flight.
        await self.proof_pool.stop()
        await self.scheduler.stop()
        await self.pearl_client.__aexit__(None, None, None)

        # Clean up temp socket if we created one
        if self._temp_socket_path and os.path.exists(self._temp_socket_path):
            os.unlink(self._temp_socket_path)

        self.logger.info("PearlGateway stopped")
