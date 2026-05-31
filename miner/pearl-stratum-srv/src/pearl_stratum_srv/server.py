"""Top-level pool server.

Owns:
  - node           (PearlNodeClient or any object with `.get_block_template()`
                    and `.submit_block(hex)` + async context-manager protocol)
  - work_cache     (WorkCache or any object with `.update_template(t)`)
  - submission     (SubmissionService or any object with
                    `.submit_plain_proof(proof, template) -> dict`)
  - JobRegistry    — maps stratum job_id → cached templates
  - asyncio TCP listener (asyncio.start_server)
  - set of live ClientConnection instances

Background tasks:
  - template_poller(): poll the node every poll_interval, mint a new
    JobEntry on each new template, broadcast mining.notify to all clients
    (clean_jobs=True so miners drop in-flight work on chain advance).

Dependencies are injected through __init__ so tests can swap in fakes
without monkeypatching pearl_gateway / pearl_mining. The classmethod
`from_settings(settings)` wires the real pearl-gateway services for prod.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pearl_stratum_srv.alerts import Alerter
from pearl_stratum_srv.auth import IpLimiter, IpQuotas
from pearl_stratum_srv.config import Settings
from pearl_stratum_srv.connection import ClientConnection
from pearl_stratum_srv.dashboard import History
from pearl_stratum_srv.job_registry import JobEntry, JobRegistry
from pearl_stratum_srv.metrics import Metrics, serve_http
from pearl_stratum_srv.node_rpc import LongPollingTemplateFetcher, PearldRpc, RpcError
from pearl_stratum_srv.payouts import PayoutPolicy, compute_pplns_payouts
from pearl_stratum_srv.share_db import ShareDb
from pearl_stratum_srv.vardiff import VardiffPolicy

if TYPE_CHECKING:
    from pearl_mining import PlainProof

_LOGGER = logging.getLogger(__name__)


class PoolServer:
    def __init__(
        self,
        settings: Settings,
        node: Any,
        work_cache: Any,
        submission: Any,
        template_fetcher: Any = None,
        verify_plain_proof: Any = None,
    ):
        self.settings = settings
        self.node = node
        self.work_cache = work_cache
        self.submission = submission
        # Optional. If supplied, the poller uses long-poll via this fetcher
        # instead of calling `node.get_block_template()` directly. Tests pass
        # a fake fetcher; from_settings() builds the real long-poll variant.
        self.template_fetcher = template_fetcher
        # Cheap pre-check: `verify_plain_proof(incomplete_header, plain_proof) -> (bool, str)`.
        # Returns True iff the plain proof is structurally valid AND its jackpot
        # hash meets the difficulty in the header's nbits — i.e. it IS a block.
        # Skipping prove+submit when this returns False is what keeps RAM and
        # CPU under control: Plonky2 prover would otherwise fire on EVERY share.
        # Tests inject a stub; from_settings() wires pearl_mining.verify_plain_proof.
        self.verify_plain_proof = verify_plain_proof

        self.jobs = JobRegistry(max_size=settings.job_history_size)
        self.metrics = Metrics()
        self.history = History()
        self._connections: set[ClientConnection] = set()
        self._next_conn_id = 0
        self._tcp_server: asyncio.base_events.Server | None = None
        self._metrics_server: asyncio.base_events.Server | None = None
        self._poller_task: asyncio.Task | None = None
        self._history_task: asyncio.Task | None = None
        self._alerter_task: asyncio.Task | None = None

        # Alerter — instantiated up front so the metrics endpoint can serve
        # /api/alerts even before serve_forever() starts the tick loop.
        self.alerter = Alerter(settings, self.metrics, self.history)

        # Public-pool services (None in solo mode).
        self.vardiff_policy = VardiffPolicy(
            initial_diff=1 << 20,
            target_shares_per_min=settings.vardiff_target_shares_per_min,
        )
        self.ip_limiter = IpLimiter(IpQuotas(
            max_concurrent=settings.max_connections_per_ip,
            max_new_per_minute=settings.max_new_connections_per_minute_per_ip,
        ))
        self.payout_policy = PayoutPolicy(
            fee_percent=settings.pool_fee_percent,
            pplns_n=settings.pplns_window_difficulty,
            operator_address=settings.mining_address,
            min_payout_sats=settings.min_payout_sats,
        )
        self.share_db: ShareDb | None = None  # set when public_pool=True in serve_forever()

    @classmethod
    def from_settings(cls, settings: Settings) -> "PoolServer":
        """Production wiring: build PearlNodeClient + WorkCache + SubmissionService
        + (optionally) a long-poll template fetcher."""
        from pearl_gateway.config import PearlConfig
        from pearl_gateway.pearl_client import PearlNodeClient
        from pearl_gateway.submission_service import SubmissionService
        from pearl_gateway.work_cache import WorkCache

        cfg = PearlConfig(
            rpc_url=settings.rpc_url,
            rpc_user=settings.rpc_user,
            rpc_password=settings.rpc_password,
            mining_address=settings.mining_address,
        )
        node = PearlNodeClient(cfg)

        # Long-poll fetcher (own aiohttp session) — bypasses pearl-gateway's
        # PearlNodeClient for fetches because that client doesn't surface
        # longpollid. We still use PearlNodeClient for submit_block via
        # SubmissionService.
        fetcher = None
        if settings.long_poll:
            rpc = PearldRpc(
                rpc_url=settings.rpc_url,
                rpc_user=settings.rpc_user,
                rpc_password=settings.rpc_password,
                request_timeout_secs=settings.long_poll_timeout_secs + 10.0,
            )
            fetcher = LongPollingTemplateFetcher(
                rpc=rpc,
                long_poll_timeout_secs=settings.long_poll_timeout_secs,
            )

        # Wire the cheap pre-check from py-pearl-mining.
        from pearl_mining import verify_plain_proof

        return cls(
            settings=settings,
            node=node,
            work_cache=WorkCache(),
            submission=SubmissionService(node, debug_mode=settings.debug_verify),
            template_fetcher=fetcher,
            verify_plain_proof=verify_plain_proof,
        )

    # ------------------------------------------------------------------ lifecycle

    async def serve_forever(self) -> None:
        """Boot listener + poller + metrics endpoint, run until cancelled."""
        await self.node.__aenter__()
        if self.template_fetcher is not None and hasattr(self.template_fetcher, "rpc"):
            await self.template_fetcher.rpc.__aenter__()
        if self.settings.public_pool:
            self.share_db = ShareDb(self.settings.share_db_path)
            await self.share_db.open()
            _LOGGER.info("public-pool mode: share_db open at %s", self.settings.share_db_path)
        try:
            self._poller_task = asyncio.create_task(self._poll_templates(), name="template-poller")
            self._tcp_server = await asyncio.start_server(
                self._on_client,
                host=self.settings.listen_host,
                port=self.settings.listen_port,
                # Default asyncio readline limit is 64 KiB. Real `mining.submit`
                # frames carry a ~368 KB base64 plain_proof on a single line
                # (STRATUM_CAPTURE.md §3g). 64 KiB would raise LimitOverrunError
                # on the first real submit. 1 MiB has headroom for future param
                # changes.
                limit=2**20,
            )
            _LOGGER.info(
                "stratum listener up on %s:%d (mining to %s)",
                self.settings.listen_host,
                self.settings.listen_port,
                self.settings.mining_address,
            )

            if self.settings.metrics_port > 0:
                self._metrics_server = await serve_http(
                    self.metrics,
                    host=self.settings.metrics_host,
                    port=self.settings.metrics_port,
                    max_template_age_seconds=self.settings.metrics_max_template_age_seconds,
                    history=self.history,
                    server=self,  # exposes share_db + settings to per-miner/operator routes
                )
                self._history_task = asyncio.create_task(self._sample_history(), name="history-tick")
                # Alerter delivery channels — open files / configure webhook.
                self.alerter.configure_deliveries()
                self._alerter_task = asyncio.create_task(self.alerter.run_forever(), name="alerter")
                _LOGGER.info(
                    "metrics + dashboard + alerter up on %s:%d (/, /metrics, /health, /api/stats, /api/history, /api/alerts)",
                    self.settings.metrics_host,
                    self.settings.metrics_port,
                )

            async with self._tcp_server:
                await self._tcp_server.serve_forever()
        finally:
            if self._poller_task is not None:
                self._poller_task.cancel()
                try:
                    await self._poller_task
                except asyncio.CancelledError:
                    pass
            if self._history_task is not None:
                self._history_task.cancel()
                try:
                    await self._history_task
                except asyncio.CancelledError:
                    pass
            if self._alerter_task is not None:
                self._alerter_task.cancel()
                try:
                    await self._alerter_task
                except asyncio.CancelledError:
                    pass
            if self._metrics_server is not None:
                self._metrics_server.close()
                await self._metrics_server.wait_closed()
            if self.template_fetcher is not None and hasattr(self.template_fetcher, "rpc"):
                await self.template_fetcher.rpc.__aexit__(None, None, None)
            if self.share_db is not None:
                await self.share_db.close()
                self.share_db = None
            await self.node.__aexit__(None, None, None)

    async def start_listener(self, port: int = 0) -> int:
        """Test helper: start TCP listener on a chosen port (0 = ephemeral) and
        return the bound port. Does NOT start the poller — tests drive templates
        manually via `ingest_template()`."""
        self._tcp_server = await asyncio.start_server(
            self._on_client,
            host=self.settings.listen_host,
            port=port,
            limit=2**20,  # match production: accept 368 KB submit lines
        )
        sock = self._tcp_server.sockets[0]
        return sock.getsockname()[1]

    async def stop_listener(self) -> None:
        """Test helper: tear down listener and close all client connections."""
        if self._tcp_server is not None:
            self._tcp_server.close()
            await self._tcp_server.wait_closed()
            self._tcp_server = None
        for conn in list(self._connections):
            try:
                conn.writer.close()
                await conn.writer.wait_closed()
            except Exception:
                pass

    # -------------------------------------------------------------- connection

    async def _on_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Source-IP gates (public-mode only). Cheap reject before allocating connection state.
        peer = writer.get_extra_info("peername")
        ip = peer[0] if isinstance(peer, tuple) else None
        if self.settings.public_pool and ip is not None:
            # Banned?
            if self.share_db is not None and await self.share_db.is_banned(ip):
                _LOGGER.info("rejected connection from banned ip %s", ip)
                writer.close()
                await writer.wait_closed()
                return
            # Rate-limited?
            ok, reason = self.ip_limiter.try_accept(ip)
            if not ok:
                _LOGGER.info("rejected connection: %s", reason)
                writer.close()
                await writer.wait_closed()
                return
            self.ip_limiter.note_open(ip)

        self._next_conn_id += 1
        conn = ClientConnection(reader, writer, self, self._next_conn_id)
        self._connections.add(conn)
        self.metrics.connected_miners = len(self._connections)
        try:
            await conn.run()
        finally:
            self._connections.discard(conn)
            self.metrics.connected_miners = len(self._connections)
            if self.settings.public_pool and ip is not None:
                self.ip_limiter.note_close(ip)

    def unregister(self, conn: ClientConnection) -> None:
        self._connections.discard(conn)
        self.metrics.connected_miners = len(self._connections)

    # ------------------------------------------------------------------ poller

    async def _poll_templates(self) -> None:
        """Background loop: fetch templates, broadcast on change.

        Two modes:
          - Long-poll (settings.long_poll=True, template_fetcher present):
            pearld blocks until the tip changes, so we react in <100ms.
          - Fixed-interval poll (legacy fallback): calls node.get_block_template()
            every poll_interval. Used when long_poll is disabled or fetcher
            is missing.

        On RPC errors we always fall back to the poll_interval sleep before
        retrying, and reset the longpollid so we don't hang on a stale handle.
        """
        interval = self.settings.poll_interval
        last_prev_hash: bytes | None = None
        use_long_poll = self.template_fetcher is not None

        while True:
            try:
                if use_long_poll:
                    template = await self._fetch_via_long_poll()
                else:
                    template = await self.node.get_block_template()
            except RpcError as e:
                _LOGGER.warning("getblocktemplate RPC error: %s", e)
                if use_long_poll:
                    self.template_fetcher.reset()
                await asyncio.sleep(interval)
                continue
            except Exception as e:
                _LOGGER.warning("getblocktemplate failed: %s", e)
                if use_long_poll:
                    self.template_fetcher.reset()
                await asyncio.sleep(interval)
                continue

            await self.work_cache.update_template(template)
            new_tip = template.header.previous_block_hash != last_prev_hash

            if new_tip:
                last_prev_hash = template.header.previous_block_hash
                await self.ingest_template(template)

            # In long-poll mode pearld already blocked on the call; no sleep
            # needed. In poll mode we wait the interval before re-asking.
            if not use_long_poll:
                await asyncio.sleep(interval)

    async def _fetch_via_long_poll(self):
        """Long-poll path: get raw dict from PearldRpc, convert via the
        pluggable converter hook so tests can bypass pearl-gateway."""
        raw = await self.template_fetcher.fetch()
        return self._template_from_raw_dict(raw)

    def _template_from_raw_dict(self, raw: dict):
        """Convert pearld's getblocktemplate response dict → BlockTemplate.

        Default impl uses pearl-gateway's helper (no duplication of coinbase
        assembly). Tests override this to return a fake BlockTemplate-like
        object without needing pearl-gateway installed.
        """
        from pearl_gateway.comm.dataclasses import BlockTemplate
        from pearl_gateway.rpc_types import GetBlockTemplateResponse

        return BlockTemplate.from_get_block_template(
            GetBlockTemplateResponse.model_validate(raw),
            mining_address=self.settings.mining_address,
        )

    async def ingest_template(self, template: Any) -> JobEntry:
        """Mint a new job for a template and broadcast notify. Used by both
        the poller and the integration tests."""
        entry = self.jobs.mint(template)
        self.metrics.template_height = entry.height
        self.metrics.template_minted_at = entry.minted_at
        self.metrics.jobs_in_registry = len(self.jobs)
        _LOGGER.info(
            "new template height=%d prev=%s job_id=%s clients=%d",
            entry.height,
            entry.prev_hash_hex[:16],
            entry.job_id,
            len(self._connections),
        )
        await self._broadcast_notify(clean=True)
        return entry

    async def _sample_history(self) -> None:
        """Background tick: snapshot Metrics into the history ring every 60s.

        The dashboard chart uses these samples — diffs between consecutive
        snapshots become per-minute share counts.
        """
        # Take an initial sample immediately so the chart has a baseline.
        self.history.record(self.metrics)
        while True:
            await asyncio.sleep(60.0)
            try:
                self.history.record(self.metrics)
            except Exception:
                _LOGGER.exception("history sampler failed")

    async def _broadcast_notify(self, clean: bool) -> None:
        """Push current job to every subscribed connection. Errors don't propagate."""
        for conn in list(self._connections):
            if not conn.subscribed:
                continue
            try:
                await conn.push_current_job(clean=clean)
            except Exception:
                _LOGGER.exception("broadcast to conn %d failed", conn.conn_id)

    # ------------------------------------------------------------- submit path

    async def handle_block_found(
        self, height: int, finder_addr: str, reward_sats: int
    ) -> None:
        """Public-pool only: persist the block, compute PPLNS payouts, queue them.

        We do NOT execute on-chain sends here — that stays a manual operator
        action via a separate CLI tool. This just records:
          1. A row in `blocks` table (block we found, who found the share, reward).
          2. Rows in `payouts` table (recipient, amount, share_count) with status='pending'.
        The operator reviews `pending_payouts()` and sends a `oyster sendmany`
        batch when satisfied.
        """
        if self.share_db is None:
            _LOGGER.warning("handle_block_found called in solo mode; ignoring")
            return

        block_id = await self.share_db.insert_block(
            height=height,
            finder_addr=finder_addr,
            reward_total=reward_sats,
        )

        # PPLNS window: walk back through accepted shares, summing difficulty
        # until we've covered `pplns_window_difficulty` total. Simpler v1:
        # last 24 hours of accepted shares (the share_db helper does this).
        # For volume-based windowing we'd add `shares_in_volume_window()`.
        import time as _time

        since = _time.time() - 86400.0  # last 24h; ample for typical PPLNS at our hashrate
        shares = await self.share_db.shares_in_window(since)

        result = compute_pplns_payouts(
            block_reward_sats=reward_sats,
            shares_by_addr=shares,
            policy=self.payout_policy,
        )

        await self.share_db.insert_payouts(
            block_id=block_id,
            entries=[(e.recipient, e.amount_sats, e.share_difficulty) for e in result.entries],
        )

        _LOGGER.warning(
            "block %d: reward=%d sats, fee=%d sats, %d recipients queued for payout",
            height,
            result.block_reward_sats,
            result.fee_sats,
            len(result.entries),
        )

    async def submit_share(
        self, plain_proof: "PlainProof", entry: JobEntry
    ) -> dict:
        """Verify the share, then submit-as-block ONLY if it meets network diff.

        Two-phase:

          1. CHEAP: pearl_mining.verify_plain_proof(header, plain_proof) →
             (True, _) iff the proof is structurally valid AND its jackpot hash
             meets the difficulty in header.nbits — i.e. it is a real block.
             Estimated ~30 μs per share.

          2. EXPENSIVE (only when phase 1 is True): SubmissionService.submit_plain_proof
             → generates the Plonky2 ZK proof (1-8 GB peak RAM, multi-second
             compute) and calls submitblock on pearld.

        Phase-2 fires once every few days at our fleet's hashrate. Phase-1
        fires on every share but is cheap. This is what keeps the pool from
        OOM-killing the host the moment a real miner connects.

        If verify_plain_proof isn't wired (e.g. unit tests), we fall through
        to the old behavior — every share goes to submission.
        """
        if self.verify_plain_proof is not None:
            try:
                is_block, msg = self.verify_plain_proof(
                    entry.template.header.incomplete_header, plain_proof
                )
            except Exception as e:
                _LOGGER.warning("verify_plain_proof raised: %s", e)
                return {"status": f"error: verify raised {e}"}
            if not is_block:
                # Valid share-level submission — counts as liveness, no block work.
                return {"status": "share_accepted", "reason": msg}

        # Block candidate confirmed (or we have no pre-check). Run the prove+submit.
        result = await self.submission.submit_plain_proof(plain_proof, entry.template)
        status = result.get("status", "error")
        if status == "accepted":
            self.metrics.record_block("accepted")
        elif status == "already_submitted":
            pass
        elif status.startswith("rejected"):
            pass
        elif status.startswith("error"):
            self.metrics.record_block("error")
        return result
