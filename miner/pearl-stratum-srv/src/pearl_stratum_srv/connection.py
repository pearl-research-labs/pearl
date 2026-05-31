"""Per-client stratum connection handler.

One asyncio task per TCP socket. Lives for the connection lifetime; reads
line-delimited JSON frames and dispatches to the matching `_handle_*`
method. Writes go through `send()` which serializes with an asyncio.Lock so
concurrent pushes (broadcast notify + reply to inbound submit) don't
interleave on the wire.

Implements only the methods alpha-miner actually sends:
  mining.configure   → ack with {"pearl/v1": true, "pearl/v1.share_format": "base64"}
  mining.subscribe   → return empty extranonce + push pearl.set_mining_params
                       + push current set_difficulty + current notify
  mining.authorize   → always true (LAN, trusted)
  mining.submit      → validate via SubmissionService; reply true/false/error[21]

We DO NOT implement pearl.challenge (LAN deployment; no DDoS risk).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING

from pearl_mining import PlainProof

from pearl_stratum_srv.auth import parse_worker_name
from pearl_stratum_srv.protocol import (
    INVALID_PARAMS_CODE,
    STALE_SHARE_CODE,
    UNKNOWN_METHOD_CODE,
    Request,
    encode_error,
    encode_notification,
    encode_response,
    parse_request,
)
from pearl_stratum_srv.vardiff import VardiffState, maybe_retarget

if TYPE_CHECKING:
    from pearl_stratum_srv.challenge import Challenge
    from pearl_stratum_srv.server import PoolServer

_LOGGER = logging.getLogger(__name__)


class ClientConnection:
    """Handles one stratum miner connection end-to-end."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        server: "PoolServer",
        conn_id: int,
    ):
        self.reader = reader
        self.writer = writer
        self.server = server
        self.conn_id = conn_id
        self.peer = writer.get_extra_info("peername")
        self.subscribed = False
        self.authorized = False
        self.worker_name: str | None = None        # raw `prl1...workerN` from mining.authorize
        self.worker_addr: str | None = None        # parsed Pearl address (payout target in public mode)
        self.worker_label: str = "default"         # the "workerN" suffix; defaults to "default"
        self._send_lock = asyncio.Lock()
        # public-pool-only state. Unused in solo mode.
        self.vardiff: VardiffState | None = None
        self.pending_challenge: "Challenge | None" = None
        self.challenge_passed: bool = False
        self.client_ip = self.peer[0] if isinstance(self.peer, tuple) and self.peer else None

    async def run(self) -> None:
        _LOGGER.info("conn %d open from %s", self.conn_id, self.peer)
        try:
            # In public mode, push pearl.challenge BEFORE accepting any frame.
            if self.server.settings.public_pool and self.server.settings.challenge_difficulty > 0:
                await self._issue_challenge()

            while True:
                line = await self.reader.readline()
                if not line:
                    break  # EOF
                if not line.strip():
                    continue
                await self._dispatch(line)
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("conn %d crashed", self.conn_id)
        finally:
            await self._close()

    async def _issue_challenge(self) -> None:
        from pearl_stratum_srv.challenge import Challenge

        ch = Challenge.issue(difficulty=self.server.settings.challenge_difficulty)
        self.pending_challenge = ch
        await self.send(encode_notification("pearl.challenge", ch.to_notification_params()))

    async def send(self, frame: bytes) -> None:
        async with self._send_lock:
            try:
                self.writer.write(frame)
                await self.writer.drain()
            except (ConnectionError, OSError) as e:
                _LOGGER.warning("conn %d send failed: %s", self.conn_id, e)
                raise

    async def _close(self) -> None:
        self.server.unregister(self)
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass
        _LOGGER.info("conn %d closed", self.conn_id)

    # ------------------------------------------------------------------ dispatch

    async def _dispatch(self, line: bytes) -> None:
        try:
            req = parse_request(line)
        except ValueError as e:
            _LOGGER.warning("conn %d bad frame: %s", self.conn_id, e)
            await self.send(encode_error(None, INVALID_PARAMS_CODE, str(e)))
            return

        # In public mode with an open challenge, accept ONLY the response.
        if (
            self.server.settings.public_pool
            and self.server.settings.challenge_difficulty > 0
            and not self.challenge_passed
            and req.method != "pearl.challenge_response"
        ):
            await self.send(encode_error(req.id, INVALID_PARAMS_CODE, "pearl.challenge required first"))
            return

        handler = self._HANDLERS.get(req.method)
        if handler is None:
            _LOGGER.warning("conn %d unknown method: %s", self.conn_id, req.method)
            await self.send(encode_error(req.id, UNKNOWN_METHOD_CODE, "unknown method"))
            return

        try:
            await handler(self, req)
        except Exception as e:
            _LOGGER.exception("conn %d handler %s raised", self.conn_id, req.method)
            await self.send(encode_error(req.id, INVALID_PARAMS_CODE, str(e)))

    # --------------------------------------------------------------- handlers

    async def _handle_configure(self, req: Request) -> None:
        # mining.configure params: [[extensions...], {ext_settings}]
        # Echo back support for pearl/v1 with base64 share format.
        result = {"pearl/v1": True, "pearl/v1.share_format": "base64"}
        await self.send(encode_response(req.id, result))

    async def _handle_subscribe(self, req: Request) -> None:
        # Response shape from STRATUM_CAPTURE.md §3b:
        #   [[[set_difficulty, conn-id], [notify, conn-id]], extranonce1, extranonce2_size]
        # Pearl/v1 uses no client-side extranonce: empty string + size 0.
        conn_tag = f"conn-{self.conn_id}"
        result = [
            [["mining.set_difficulty", conn_tag], ["mining.notify", conn_tag]],
            "",
            0,
        ]
        await self.send(encode_response(req.id, result))
        self.subscribed = True

        # Push pearl.set_mining_params (unsolicited, single-element list per
        # STRATUM_CAPTURE.md §3c — params is [<dict>], not just <dict>).
        await self.send(
            encode_notification(
                "pearl.set_mining_params",
                [self.server.settings.mining_params_payload()],
            )
        )

        # Initial difficulty: vardiff's `initial_diff` in public mode (per-worker),
        # 1 in solo (we accept everything for liveness).
        if self.server.settings.public_pool:
            self.vardiff = VardiffState()
            initial_diff = self.vardiff.init(self.server.vardiff_policy)
        else:
            initial_diff = 1
        await self.send(encode_notification("mining.set_difficulty", [initial_diff]))
        await self.push_current_job(clean=True)

    async def _handle_authorize(self, req: Request) -> None:
        params = req.params or []
        worker = params[0] if isinstance(params, list) and params else "anonymous"
        self.worker_name = str(worker)
        addr, label = parse_worker_name(self.worker_name)
        self.worker_addr = addr
        self.worker_label = label
        self.authorized = True

        # In public mode, reject if no valid Pearl address — payouts would
        # have no recipient.
        if self.server.settings.public_pool and addr is None:
            _LOGGER.warning("conn %d rejected: invalid wallet in %s", self.conn_id, self.worker_name)
            await self.send(encode_error(req.id, INVALID_PARAMS_CODE,
                                          "worker name must be prl1...address[.label]"))
            return

        _LOGGER.info("conn %d authorized %s (addr=%s label=%s)",
                     self.conn_id, self.worker_name, addr or "<solo>", label)
        await self.send(encode_response(req.id, True))

    async def _handle_challenge_response(self, req: Request) -> None:
        params = req.params or {}
        if not isinstance(params, dict):
            await self.send(encode_error(req.id, INVALID_PARAMS_CODE, "params must be object"))
            return
        ch = self.pending_challenge
        if ch is None:
            await self.send(encode_error(req.id, INVALID_PARAMS_CODE, "no challenge in flight"))
            return
        seed = params.get("seed", "")
        nonce = params.get("nonce", "")
        if ch.verify(seed, nonce):
            self.challenge_passed = True
            self.pending_challenge = None
            _LOGGER.info("conn %d passed pearl.challenge", self.conn_id)
            await self.send(encode_response(req.id, True))
        else:
            _LOGGER.warning("conn %d failed pearl.challenge", self.conn_id)
            await self.send(encode_error(req.id, INVALID_PARAMS_CODE, "bad challenge response"))

    async def _handle_submit(self, req: Request) -> None:
        # params: [worker, job_id, plain_proof_b64]
        params = req.params
        if not isinstance(params, list) or len(params) < 3:
            await self.send(encode_error(req.id, INVALID_PARAMS_CODE, "submit needs 3 params"))
            return
        _worker, job_id, plain_proof_b64 = params[0], params[1], params[2]

        entry = self.server.jobs.get(str(job_id))
        if entry is None:
            # Stale: chain advanced past this job_id. Keep socket open.
            _LOGGER.info("conn %d stale job_id=%s", self.conn_id, job_id)
            self.server.metrics.record_share(self.worker_name or "anonymous", "stale")
            await self._record_share_db(entry, "stale")
            await self.send(encode_error(req.id, STALE_SHARE_CODE, "Job not found"))
            return

        # Decode + hand to the submission pipeline.
        try:
            plain_proof_bytes = base64.b64decode(plain_proof_b64)
            plain_proof = PlainProof.from_bytes(plain_proof_bytes)
        except Exception as e:
            self.server.metrics.record_share(self.worker_name or "anonymous", "malformed")
            await self._record_share_db(entry, "malformed")
            await self._maybe_autoban_malformed()
            await self.send(encode_error(req.id, INVALID_PARAMS_CODE, f"bad plain_proof: {e}"))
            return

        result = await self.server.submit_share(plain_proof, entry)
        self.server.metrics.record_share(self.worker_name or "anonymous", "accepted")
        await self._record_share_db(entry, "accepted")

        # Vardiff tick — may push a new mining.set_difficulty.
        if self.vardiff is not None:
            self.vardiff.note_share()
            new_diff = maybe_retarget(self.vardiff, self.server.vardiff_policy)
            if new_diff is not None:
                await self.send(encode_notification("mining.set_difficulty", [new_diff]))

        # Ack the share. Block detection happens inside server.submit_share;
        # pool credit in public mode is per-share PPLNS (computed at block-find).
        await self.send(encode_response(req.id, True))

        if result.get("status") == "accepted":
            _LOGGER.warning(
                "conn %d BLOCK ACCEPTED at height %d by worker %s",
                self.conn_id,
                entry.height,
                self.worker_name,
            )
            # In public mode, queue payouts. server handles the PPLNS calc + persist.
            if self.server.settings.public_pool:
                await self.server.handle_block_found(
                    height=entry.height,
                    finder_addr=self.worker_addr or self.server.settings.mining_address,
                    reward_sats=getattr(entry.template, "coinbase_value_sats", 0),
                )

    async def _record_share_db(self, entry, outcome: str) -> None:
        """Persist this share if a ShareDb is wired (public-pool mode only)."""
        db = getattr(self.server, "share_db", None)
        if db is None:
            return
        # Difficulty: in public mode use the worker's current vardiff;
        # in solo there's no per-share accounting so this is best-effort.
        diff = self.vardiff.current_diff if self.vardiff else 1
        await db.insert_share(
            worker_addr=self.worker_addr or "anonymous",
            worker_label=self.worker_label,
            job_id=getattr(entry, "job_id", "<no-job>"),
            difficulty=diff,
            outcome=outcome,
            ip=self.client_ip,
        )

    async def _maybe_autoban_malformed(self) -> None:
        """Auto-ban this IP if it's flooded us with malformed shares."""
        db = getattr(self.server, "share_db", None)
        if db is None or self.client_ip is None:
            return
        threshold = self.server.settings.malformed_share_ban_threshold
        if threshold <= 0:
            return
        count = await db.malformed_rate_for_ip(self.client_ip, window_secs=300.0)
        if count >= threshold:
            duration = self.server.settings.malformed_share_ban_duration_secs
            await db.ban_ip(
                self.client_ip,
                f"{count} malformed shares in 5min",
                duration_secs=duration,
            )
            _LOGGER.warning("conn %d auto-banned IP %s for %ds", self.conn_id, self.client_ip, duration)

    _HANDLERS = {
        "mining.configure": _handle_configure,
        "mining.subscribe": _handle_subscribe,
        "mining.authorize": _handle_authorize,
        "mining.submit": _handle_submit,
        "pearl.challenge_response": _handle_challenge_response,
    }

    # ----------------------------------------------------------------- pushes

    async def push_current_job(self, clean: bool) -> None:
        """Send mining.notify for the current job. No-op if no template yet."""
        entry = self.server.jobs.latest()
        if entry is None:
            return
        params = [
            entry.job_id,
            entry.prev_hash_hex,
            entry.incomplete_header_hex,
            0,  # opaque seq int — alphapool sometimes uses a counter; harmless 0
            entry.ntime_hex,
            entry.nbits_hex,
            clean,
        ]
        try:
            await self.send(encode_notification("mining.notify", params))
        except (ConnectionError, OSError):
            pass  # close path will tear us down
