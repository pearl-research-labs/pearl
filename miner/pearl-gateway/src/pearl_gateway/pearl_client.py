import asyncio
from typing import Any

import aiohttp
from miner_utils import get_logger

from pearl_gateway.comm.dataclasses import BlockTemplate
from pearl_gateway.config import PearlConfig
from pearl_gateway.rpc_types import GetBlockTemplateResponse

logger = get_logger(__name__)


class PearlNodeClient:
    """Client for interacting with Pearl Core's JSON-RPC API over HTTPS."""

    def __init__(self, config: PearlConfig):
        url = config.rpc_url

        self.rpc_url = url
        self.auth = aiohttp.BasicAuth(config.rpc_user, config.rpc_password)
        self.mining_address = config.mining_address
        self.session = None
        self.request_id = 0
        self.backoff_time = 1  # Initial backoff time in seconds
        self.max_backoff = 60  # Maximum backoff time in seconds

        logger.info(f"Using mining address: {self.mining_address}")

        logger.info(
            f"PearlNodeClient initialized with rpc_url: {self.rpc_url}, rpc_user: {config.rpc_user}"
        )

    def _create_session(self) -> aiohttp.ClientSession:
        """Create an HTTP session."""
        return aiohttp.ClientSession(auth=self.auth)

    async def __aenter__(self):
        self.session = self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    async def _make_rpc_call(self, method: str, params: list = None) -> dict[str, Any]:
        """Make a JSON-RPC call to the Pearl node."""
        if not self.session:
            self.session = self._create_session()

        logger.debug(f"Making RPC call to {self.rpc_url} with method {method}")
        logger.trace(f"{params=}")

        if params is None:
            params = []

        self.request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.request_id,
        }

        try:
            async with self.session.post(self.rpc_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Error communicating with Pearl node: HTTP {response.status}")
                    await asyncio.sleep(self.backoff_time)
                    self.backoff_time = min(self.backoff_time * 2, self.max_backoff)
                    raise ConnectionError(f"Pearl node returned HTTP {response.status}")

                data = await response.json()
                self.backoff_time = 1  # Reset backoff on successful call

                if "error" in data and data["error"] is not None:
                    logger.error(f"Pearl RPC error: {data['error']}")
                    raise ValueError(f"Pearl RPC error: {data['error']}")

                return data["result"]
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to Pearl node: {type(e)=} {e=}")
            await asyncio.sleep(self.backoff_time)
            self.backoff_time = min(self.backoff_time * 2, self.max_backoff)
            raise ConnectionError(f"Failed to connect to Pearl node: {e}") from e

    async def get_block_template(self) -> BlockTemplate:
        """Fetch the latest block template from the Pearl node."""
        logger.debug("Fetching block template from Pearl node")

        template_request = {
            "capabilities": ["coinbasevalue", "workid", "coinbase/append"],
            "rules": ["segwit"],
        }

        result = await self._make_rpc_call("getblocktemplate", [template_request])

        return BlockTemplate.from_get_block_template(
            GetBlockTemplateResponse.model_validate(result), mining_address=self.mining_address
        )

    async def submit_block(self, block_hex: str) -> str:
        """Submit a candidate block to the Pearl node."""
        logger.info("Submitting block to Pearl node")

        # The submitblock call returns null on success, or an error string on failure
        result = await self._make_rpc_call("submitblock", [block_hex])

        if result is None:
            return "accepted"
        else:
            logger.warning(f"Block rejected: {result}")
            return f"rejected: {result}"
