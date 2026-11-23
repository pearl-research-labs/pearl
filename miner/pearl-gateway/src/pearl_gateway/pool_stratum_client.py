"""Stratum v1 client for connecting the gateway to a Pearl mining pool."""

import secrets
from typing import Any

from miner_utils import get_logger

from pearl_gateway.blockchain_utils.blockchain_utils import bits_to_target
from pearl_gateway.comm.dataclasses import BlockTemplate, StratumNotify, StratumSetDifficulty
from pearl_gateway.comm.stratum import (
    AsyncStratumConnection,
    JSONRPCError,
    parse_extranonce2_size,
)
from pearl_gateway.config import PoolConfig
from pearl_gateway.work_cache import WorkCache

logger = get_logger(__name__)


class PoolStratumClient:
    """Persistent Stratum v1 TCP client that receives work from a pool
    and forwards miner submissions back.

    Lifecycle: subscribe -> authorize -> listen for
    notifications. The transport (connect/TLS, reconnect backoff, response
    routing) lives in AsyncStratumConnection; this class owns the gateway
    specifics: parsing templates into the WorkCache and submitting certificates.
    """

    def __init__(self, config: PoolConfig, address: str, work_cache: WorkCache):
        self.config = config
        self.address = address
        self.work_cache = work_cache

        # Assigned by the pool during mining.subscribe
        self._extranonce: bytes = b""
        # Extranonce2 byte size from mining.subscribe; 0 means the pool advertises none.
        self._extranonce2_size: int = 0

        # Current share target pushed via mining.set_difficulty
        self.share_target: int | None = None
        # Set at every (re)subscribe: the next mining.notify must replace the cached
        # template, whose coinbase still holds the previous session's extranonce1.
        self._new_session = False

        self._conn = AsyncStratumConnection(
            config.stratum_host,
            config.stratum_port,
            tls=config.stratum_tls,
            reconnect_delay_seconds=config.reconnect_delay_seconds,
            max_reconnect_delay_seconds=config.max_reconnect_delay_seconds,
            handshake=self._subscribe_and_authorize,
            on_notification=self._handle_notification,
        )

    @property
    def is_connected(self) -> bool:
        return self._conn.is_connected

    @property
    def stratum_user(self) -> str:
        return f"{self.address}.{self.config.worker}"

    @property
    def authorize_password(self) -> str:
        """POOL_PASSWORD, with POOL_DIFFICULTY appended as ``d=N``: the start
        difficulty convention Pearl pools read from the password."""
        if self.config.difficulty is None:
            return self.config.password
        return ";".join(p for p in (self.config.password, f"d={self.config.difficulty}") if p)

    async def start(self) -> None:
        await self._conn.start()

    async def stop(self) -> None:
        await self._conn.stop()
        logger.info("PoolStratumClient stopped")

    async def submit_zk_cert(
        self,
        job_id: str,
        ntime_hex: str,
        zkcert_hex: str,
        extranonce2_hex: str | None = None,
    ) -> bool:
        """Send mining.submit with a ZKCertificate to the pool. Returns True if accepted.

        ``extranonce2_hex``, when the pool advertised one, is the third param
        (worker, job_id, extranonce2, ntime, zkcert_hex); omitted otherwise.
        """
        params: list[Any] = [self.stratum_user, job_id]
        if extranonce2_hex is not None:
            params.append(extranonce2_hex)
        params.append(ntime_hex)
        params.append(zkcert_hex)
        try:
            result = await self._conn.call("mining.submit", params)
            return result is True
        except JSONRPCError as e:
            logger.warning(f"Pool rejected share: {e}")
            return False
        except Exception:
            logger.exception("Error submitting proof to pool")
            return False

    async def _subscribe_and_authorize(self) -> None:
        """Run the Stratum subscribe + authorize handshake."""
        # 1. mining.subscribe -> [[subscriptions...], extranonce1, extranonce2_size?]
        result = await self._conn.call("mining.subscribe", ["pearl-gateway/1.0"])
        self._extranonce = bytes.fromhex(result[1])
        self._new_session = True
        self._extranonce2_size = (
            parse_extranonce2_size(result[2]) if len(result) > 2 and result[2] is not None else 0
        )
        logger.info(f"Subscribed: extranonce={result[1]} extranonce2_size={self._extranonce2_size}")

        # 2. mining.authorize
        auth_result = await self._conn.call(
            "mining.authorize", [self.stratum_user, self.authorize_password]
        )
        if auth_result is not True:
            raise ConnectionError(f"Pool authorization failed: {auth_result}")
        logger.info(f"Authorized as {self.stratum_user}")

    async def _handle_notification(self, method: str, params: list[Any]) -> None:
        if method == "mining.notify":
            await self._on_mining_notify(params)
        elif method == "mining.set_difficulty":
            await self._on_set_difficulty(params)
        else:
            logger.debug(f"Ignoring unknown pool notification: {method}")

    async def _on_mining_notify(self, params: list[Any]) -> None:
        """Handle mining.notify -- build a BlockTemplate and push it to the WorkCache.

        params: [job_id, prev_hash, coinb1, coinb2, merkle_branch[],
                 version, bits, ntime, clean_jobs, cert_version]
        """
        # One extranonce2 per job, shared by all this gateway's miners; the pool
        # only fixes its length and folds it into the coinbase.
        extranonce2 = secrets.token_bytes(self._extranonce2_size)
        notify = StratumNotify.from_params(
            params, extranonce=self._extranonce, extranonce2=extranonce2
        )

        logger.info(
            f"Received mining.notify: job={notify.job_id} clean={notify.clean_jobs} "
            f"cert_version={int(notify.cert_version)}"
        )

        if self.share_target is None:
            logger.warning(
                "mining.notify before mining.set_difficulty; miners use network difficulty until set"
            )

        template = BlockTemplate.from_stratum_notify(notify)
        template.share_target = self.share_target

        # The pool re-sends its current job on a new connection, usually without
        # clean_jobs; force the swap so miners stop working on the old extranonce1.
        force = notify.clean_jobs or self._new_session
        self._new_session = False
        await self.work_cache.update_template(template, force=force)

    async def _on_set_difficulty(self, params: list[Any]) -> None:
        """Handle mining.set_difficulty -- miners now find shares at the pool's difficulty."""
        set_difficulty = StratumSetDifficulty.from_params(params)
        self.share_target = bits_to_target(set_difficulty.bits)
        await self.work_cache.update_share_target(self.share_target)
        logger.info(f"Pool share difficulty updated: bits=0x{set_difficulty.bits:08x}")
