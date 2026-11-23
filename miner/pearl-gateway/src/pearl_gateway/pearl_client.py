from typing import Any

import aiohttp
from miner_utils import get_logger

from pearl_gateway.comm.dataclasses import BlockTemplate
from pearl_gateway.config import PearlConfig
from pearl_gateway.rpc_types import GetBlockTemplateResponse

logger = get_logger(__name__)


class PearlNodeClient:
    """JSON-RPC client for Pearl Core (pearld)."""

    def __init__(self, config: PearlConfig) -> None:
        self.rpc_url = config.rpc_url
        self.auth = aiohttp.BasicAuth(config.rpc_user, config.rpc_password)
        self.mining_address = config.mining_address
        self.session: aiohttp.ClientSession | None = None
        self.request_id = 0
        logger.info(
            f"PearlNodeClient initialized with rpc_url={self.rpc_url}, "
            f"mining_address={self.mining_address}"
        )

    async def start(self) -> None:
        if self.session is None:
            self.session = aiohttp.ClientSession(auth=self.auth)

    async def stop(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def _make_rpc_call(self, method: str, params: list[Any] | None = None) -> Any:
        if self.session is None:
            await self.start()
        assert self.session is not None

        self.request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params if params is not None else [],
            "id": self.request_id,
        }
        try:
            async with self.session.post(self.rpc_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Error communicating with Pearl node: HTTP {response.status}")
                    raise ConnectionError(f"Pearl node returned HTTP {response.status}")
                data = await response.json()
                if data.get("error") is not None:
                    logger.error(f"Pearl RPC error: {data['error']}")
                    raise ValueError(f"Pearl RPC error: {data['error']}")
                return data["result"]
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to Pearl node: {type(e)=} {e=}")
            raise ConnectionError(f"Failed to connect to Pearl node: {e}") from e

    async def get_block_template(self) -> BlockTemplate:
        template_request = {
            "capabilities": ["coinbasevalue", "workid", "coinbase/append"],
            "rules": ["segwit"],
        }
        result = await self._make_rpc_call("getblocktemplate", [template_request])
        return BlockTemplate.from_get_block_template(
            GetBlockTemplateResponse.model_validate(result), mining_address=self.mining_address
        )

    async def submit_block(self, block_hex: str) -> str:
        logger.info("Submitting block to Pearl node")
        result = await self._make_rpc_call("submitblock", [block_hex])
        if result is None:
            return "accepted"
        logger.warning(f"Block rejected: {result}")
        return f"rejected: {result}"
