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

# Notifications worth holding until the handshake finishes, in replay order.
_DEFERRABLE = ("mining.set_difficulty", "mining.notify")


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

        # Current share target pushed via mining.set_difficulty (and its compact form)
        self.share_target: int | None = None
        self.share_bits: int | None = None

        # Notifications are held until the handshake has stored the session's
        # extranonce1: a pool may write work right behind the subscribe reply. Only
        # the latest of each kind matters, so at most two are held, whatever the
        # peer sends.
        self._session_ready = False
        self._deferred: dict[str, list[Any]] = {}

        self._conn = AsyncStratumConnection(
            config.stratum_host,
            config.stratum_port,
            tls=config.stratum_tls,
            reconnect_delay_seconds=config.reconnect_delay_seconds,
            max_reconnect_delay_seconds=config.max_reconnect_delay_seconds,
            handshake=self._subscribe_and_authorize,
            on_notification=self._handle_notification,
            on_disconnect=self._on_disconnect,
        )

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
        except (TimeoutError, ConnectionError, OSError):
            # A pool that stops answering without closing the socket would otherwise
            # keep us on a dead session; drop it so the connection loop reconnects.
            logger.warning("Pool did not answer mining.submit; reconnecting")
            self._conn.disconnect()
            return False
        except Exception:
            logger.exception("Error submitting proof to pool")
            return False

    async def _subscribe_and_authorize(self) -> None:
        """Run the Stratum subscribe + authorize handshake."""
        self._session_ready = False
        self._deferred = {}

        # 1. mining.subscribe -> [[subscriptions...], extranonce1, extranonce2_size?]
        result = await self._conn.call("mining.subscribe", ["pearl-gateway/1.0"])
        self._extranonce = bytes.fromhex(result[1])
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

        # Replay work the pool sent during the handshake, now that it can be built.
        # Difficulty first, so the replayed job carries the session's share target.
        # Loop: a notification arriving during a replay lands in the fresh dict;
        # without the loop it would stay stranded behind _session_ready.
        while self._deferred:
            deferred, self._deferred = self._deferred, {}
            for method in _DEFERRABLE:
                if (params := deferred.get(method)) is None:
                    continue
                try:
                    await self._dispatch(method, params)
                except Exception:
                    logger.exception(f"Ignoring pool notification that failed to handle: {method}")
        self._session_ready = True

    async def _on_disconnect(self) -> None:
        """Session ended: its job and share target are dead, so miners must not use them."""
        self._session_ready = False
        self._deferred = {}
        self.share_target = None
        self.share_bits = None
        await self.work_cache.clear()

    async def _handle_notification(self, method: str, params: list[Any]) -> None:
        if not self._session_ready:
            if method in _DEFERRABLE:
                self._deferred[method] = params
            return
        await self._dispatch(method, params)

    async def _dispatch(self, method: str, params: list[Any]) -> None:
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
        template.share_bits = self.share_bits

        # In pool mode every notify with a new job id is a new job, even on the same
        # block. A repeat of the current job keeps the cached template (and its
        # extranonce2), so proofs in flight for it stay valid.
        current = self.work_cache.current_template
        is_new = notify.clean_jobs or current is None or current.job_id != notify.job_id
        await self.work_cache.update_template(template, force=is_new)

    async def _on_set_difficulty(self, params: list[Any]) -> None:
        """Handle mining.set_difficulty -- miners now find shares at the pool's difficulty."""
        set_difficulty = StratumSetDifficulty.from_params(params)
        self.share_bits = set_difficulty.bits
        self.share_target = bits_to_target(set_difficulty.bits)
        await self.work_cache.update_share_target(self.share_target, self.share_bits)
        logger.info(f"Pool share difficulty updated: bits=0x{set_difficulty.bits:08x}")
