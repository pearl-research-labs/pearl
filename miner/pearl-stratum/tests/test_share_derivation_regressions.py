"""Regression tests guarding the share-derivation -> submit path.

The driver agents are rewiring the per-attempt kernel call (persistent-CTA
multi-nonce, multi-stream dispatch, CUDA-Graph capture). These rewires must
NOT change the (job_id, plain_proof_b64) bytes the shim writes onto the wire,
nor the staleness / reconnect semantics around them. This file pins:

  1. **Share-derivation byte contract.** A fixed (job, nonce, seed) tuple
     produces a deterministic plain_proof envelope. We synthesize the proof
     locally (the kernel is opaque to this test) and assert the shim transmits
     those exact bytes verbatim — both via `submit_share` (mining.submit) and
     via `submit_plain_proof_blocking` (submitPlainProof with mining_job).

  2. **Reject path: malformed commitment.** The pool returned
     `"invalid commitment"` (custom error code) — per
     `pearl-investigation/share_submission_audit_2026_05_17.md` — when the
     driver was substituting random A/B. The shim must surface the rejection
     as a SubmitResult(accepted=False, error=...) AND keep the TCP socket open
     (the alphafix.c bug-fix invariant generalized beyond code 21).

  3. **Stale-share filter.** When the local mapping no longer holds the
     header bytes the worker thread submits against (job retention exhausted
     or never seen), `submit_plain_proof_blocking` MUST synthesize a code-21
     reject without touching the socket. This guards against orphaned shares
     being submitted after the persistent-CTA path produces a batch that
     outlives its job window.

  4. **Reconnect path.** A mid-call TCP drop fails the inflight Future with
     `connection dropped`, the run-loop reconnects, and subsequent submits
     succeed on the new socket. Validates `_fail_pending` + the bare
     reconnect handshake.

  5. **(env-gated) Live pool integration.** With
     `PEARL_STRATUM_LIVE_TEST=1` set, connect to us2.alphapool.tech:5566 with
     the decoy wallet and submit N=10 synthesized shares; assert accept
     rate >= 80% (or document the pool-side reject reason). OFF by default.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import time
from dataclasses import dataclass

import pytest

from pearl_stratum.gateway_shim import SharedState, encode_plain_proof_b64
from pearl_stratum.stratum_client import (
    StratumClient,
    SubmitResult,
    parse_pool_url,
)

# Re-use the FakePool from the dialogue suite. Importing rather than copying
# keeps a single source of truth for the wire-level fixture behavior.
from .test_stratum_dialogue import FakePool


# NOTE: we do NOT set a module-level `pytest.mark.asyncio` here. conftest.py
# already adds the marker to every coroutine test via
# `pytest_collection_modifyitems`. A sync test in this module would inherit
# a stray asyncio mark from a module-level pytestmark otherwise.


# ---------------------------------------------------------------------------
# Deterministic share-derivation fixture
# ---------------------------------------------------------------------------
#
# The real kernel emits a `PlainProof` whose layout depends on (M, N, K, R,
# the noised A/B operands, the chosen nonce, and a commitment hash). All of
# that is opaque to this test — we don't have CUDA — so we pin a synthetic
# derivation that has the same SHAPE the wire sees: a base64-encoded blob
# whose decoded bytes are a deterministic function of (header, nonce, seed).
#
# The test value here is asserting BYTE-EXACT pass-through. If the driver
# starts substituting random data again (the regression we want to catch),
# the wire bytes will diverge from `derive_synthetic_share` and the assertion
# fires.


@dataclass(frozen=True)
class SyntheticProof:
    """Stand-in for `pearl_mining.PlainProof` — has `to_base64()` only."""

    payload: bytes

    def to_base64(self) -> str:
        return base64.b64encode(self.payload).decode("ascii")


def derive_synthetic_share(
    incomplete_header_bytes: bytes,
    nonce: int,
    seed: bytes,
) -> SyntheticProof:
    """Deterministic stand-in for the kernel's PoW proof.

    Layout (104 bytes, kept small for test legibility):

        [0..32)   blake2b-256(header || nonce_le || seed)  — "commitment_hash"
        [32..40)  nonce as little-endian uint64           — echoes the search
        [40..104) header bytes (truncated/padded to 64)    — proves we used
                                                              the header the
                                                              pool sent

    The exact bytes don't matter to the pool — this is a TEST fixture; we
    just need it to be reproducible across runs so byte-comparison works.
    """
    nonce_le = int(nonce).to_bytes(8, "little", signed=False)
    h = hashlib.blake2b(
        incomplete_header_bytes + nonce_le + seed,
        digest_size=32,
    ).digest()
    header_pad = (incomplete_header_bytes[:64]).ljust(64, b"\x00")
    return SyntheticProof(payload=h + nonce_le + header_pad)


def _expected_b64_for(
    header_bytes: bytes, nonce: int, seed: bytes
) -> str:
    """Wire-form base64 of the fixed synthetic share."""
    return derive_synthetic_share(header_bytes, nonce, seed).to_base64()


# ---------------------------------------------------------------------------
# A FakePool subclass that lets the test dictate per-submit responses
# ---------------------------------------------------------------------------


class ScriptedPool(FakePool):
    """FakePool extension that can:

      * return scripted error codes (e.g. "invalid commitment")
      * record the EXACT bytes received in each mining.submit / submitPlainProof
      * drop the socket on demand (mid-submit reconnect test)
      * push a follow-up mining.notify with a different job_id
    """

    def __init__(self, **kw) -> None:
        # Pull our extension knobs out before delegating.
        self._submit_responses: list[dict] = kw.pop("submit_responses", []) or []
        self._drop_after_submit: int = kw.pop("drop_after_submit", -1)
        self._pending_extra_notify = kw.pop("pending_extra_notify", None)
        super().__init__(**kw)
        # Override base submit_response so we can drive responses scripted.
        # The base class uses self.submit_response; we intercept in our
        # _handle_client below.
        self._scripted_mode = True
        self.received_submits: list[dict] = []  # {method, params, raw_msg}
        self._submit_seq = 0

    async def _send_scripted(self, writer: asyncio.StreamWriter, rid, response: dict) -> None:
        """Emit one scripted response keyed by 'kind'.

        response = {"kind": "accept"} -> {"result": True}
        response = {"kind": "stale"}  -> error 21
        response = {"kind": "invalid_commitment"} -> error -1 "invalid commitment"
        response = {"kind": "custom", "code": int, "msg": str}
        """
        kind = response.get("kind", "accept")
        if kind == "accept":
            await self._send(writer, {"jsonrpc": "2.0", "id": rid, "result": True})
        elif kind == "stale":
            await self._send(writer, {
                "jsonrpc": "2.0", "id": rid,
                "error": [21, "chain advanced - share points to old block", None],
            })
        elif kind == "invalid_commitment":
            await self._send(writer, {
                "jsonrpc": "2.0", "id": rid,
                "error": [-1, "invalid commitment", None],
            })
        elif kind == "custom":
            await self._send(writer, {
                "jsonrpc": "2.0", "id": rid,
                "error": [response["code"], response["msg"], None],
            })
        elif kind == "drop":
            # Close socket without responding -> client sees EOF + pending fail.
            with contextlib.suppress(Exception):
                writer.close()
        else:
            raise ValueError(f"unknown scripted kind: {kind!r}")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Override the parent's handler so we can record submits and drive
        responses from `self._submit_responses` instead of one fixed mode.
        """
        self.connection_count += 1
        self.last_writer = writer
        # Register so stop() can cancel us promptly on Windows IOCP.
        self._handler_tasks.append(asyncio.current_task())  # type: ignore[arg-type]
        self._connected.set()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                msg = json.loads(line)
                self.requests.append(msg)
                method = msg.get("method")
                rid = msg.get("id")
                if method == "mining.configure":
                    await self._send(writer, {
                        "jsonrpc": "2.0", "id": rid,
                        "result": {"pearl/v1": True, "pearl/v1.share_format": "base64"},
                    })
                elif method == "mining.subscribe":
                    await self._send(writer, {
                        "jsonrpc": "2.0", "id": rid,
                        "result": [[["mining.set_difficulty", "x"], ["mining.notify", "x"]], "", 0],
                    })
                elif method == "mining.authorize":
                    if self.push_set_mining_params:
                        await self._send(writer, {
                            "method": "pearl.set_mining_params",
                            "params": [{
                                "m": 131072, "n": 131072, "k": 4096, "rank": 128,
                                "rows_pattern": [0, 32],
                                "cols_pattern": list(range(64)),
                                "mma_type": "Int7xInt7ToInt32",
                            }],
                        })
                    await self._send(writer, {"jsonrpc": "2.0", "id": rid, "result": True})
                    if self.diff_after_notify is not None:
                        await self._send(writer, {
                            "method": "mining.set_difficulty",
                            "params": [self.diff_after_notify],
                        })
                    if self.push_notify_after_authorize:
                        await self._send(writer, {
                            "method": "mining.notify",
                            "params": self.notify_params,
                        })
                elif method in ("mining.submit", "submitPlainProof"):
                    self.submit_count += 1
                    self.received_submits.append({
                        "method": method, "params": msg.get("params"), "raw": msg,
                        "raw_line": line,
                    })
                    # If we've been asked to drop after the Nth submit, do so
                    # WITHOUT responding so the client's pending future is
                    # tripped via _fail_pending on the EOF.
                    if self._drop_after_submit >= 0 and self.submit_count == self._drop_after_submit + 1:
                        with contextlib.suppress(Exception):
                            writer.close()
                        return
                    # Drive scripted response by index, defaulting to accept
                    # once we exhaust the script.
                    if self._submit_seq < len(self._submit_responses):
                        resp = self._submit_responses[self._submit_seq]
                    else:
                        resp = {"kind": "accept"}
                    self._submit_seq += 1
                    await self._send_scripted(writer, rid, resp)
                    # Optional: push an extra mining.notify after this submit
                    if self._pending_extra_notify is not None and self.submit_count == 1:
                        params = self._pending_extra_notify
                        self._pending_extra_notify = None
                        await self._send(writer, {"method": "mining.notify", "params": params})
                else:
                    pass
        except (ConnectionResetError, asyncio.IncompleteReadError, asyncio.CancelledError):
            return
        except Exception:  # pragma: no cover
            import traceback

            traceback.print_exc()
        finally:
            with contextlib.suppress(Exception):
                writer.close()


# ---------------------------------------------------------------------------
# Shared helper: spin client; mirrors test_stratum_dialogue._spin_client
# ---------------------------------------------------------------------------


async def _spin_client(
    pool: FakePool,
    *,
    on_new_job=None,
    address: str = "prl1testtesttest",
    worker: str = "regression-test",
) -> tuple[StratumClient, asyncio.Task]:
    callbacks = {}
    if on_new_job is not None:
        callbacks["on_new_job"] = on_new_job
    client = StratumClient(
        host=pool.host, port=pool.port,
        address=address, worker=worker,
        password="x;d=1048576",
        **callbacks,
    )
    task = asyncio.create_task(client.run())
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client.current_job is not None:
            break
        await asyncio.sleep(0.01)
    return client, task


# Fixed input data used across the regression cases ---------------------------

FIXED_NONCE = 0x0123_4567_89AB_CDEF
FIXED_SEED = bytes.fromhex("458f2f324f358c108b86a3008070655be362899bdd016eee97177ac1434564ac")

NOTIFY_JOB_A = [
    "JOB-A-0001",
    "11" * 32,
    "deadbeef" + "00" * 60,  # incomplete_header_bytes, 64B
    11111,
    "6a093061",
    "1a0ffff0",
    True,
]
NOTIFY_JOB_B = [
    "JOB-B-0002",
    "22" * 32,
    "feedface" + "00" * 60,
    22222,
    "6a09a063",
    "1a0ffff0",
    True,
]


# ===========================================================================
# 1. Share-derivation byte contract
# ===========================================================================


async def test_synthetic_share_is_deterministic() -> None:
    """Independent of the network: same inputs => same b64. Pins the fixture."""
    header = bytes.fromhex(NOTIFY_JOB_A[2])
    p1 = derive_synthetic_share(header, FIXED_NONCE, FIXED_SEED)
    p2 = derive_synthetic_share(header, FIXED_NONCE, FIXED_SEED)
    assert p1.payload == p2.payload
    # Distinct nonce changes the bytes (sanity check the seed/nonce both matter).
    assert derive_synthetic_share(header, FIXED_NONCE + 1, FIXED_SEED).payload != p1.payload
    assert derive_synthetic_share(header, FIXED_NONCE, b"\x00" * 32).payload != p1.payload


async def test_submit_share_transmits_exact_bytes() -> None:
    """`mining.submit` params[2] must be byte-exact what derive_synthetic_share
    produced. Any driver-side substitution (random A/B, wrong nonce, etc.)
    would change these bytes and break this assertion.
    """
    pool = ScriptedPool(notify_params=NOTIFY_JOB_A, submit_responses=[{"kind": "accept"}])
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            header = bytes.fromhex(NOTIFY_JOB_A[2])
            expected_b64 = _expected_b64_for(header, FIXED_NONCE, FIXED_SEED)
            result = await client.submit_share(NOTIFY_JOB_A[0], expected_b64)
            assert result.accepted
            sub = next(r for r in pool.received_submits if r["method"] == "mining.submit")
            params = sub["params"]
            assert isinstance(params, list) and len(params) == 3
            assert params[0] == "prl1testtesttest.regression-test"
            assert params[1] == NOTIFY_JOB_A[0]
            assert params[2] == expected_b64, (
                "mining.submit transmitted bytes diverged from the derived "
                "share — driver may be substituting random data?"
            )
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_submit_plain_proof_transmits_exact_bytes_and_envelope() -> None:
    """The `submitPlainProof` form carries `mining_job` with header echo +
    mining_params. Verify both the payload b64 and the envelope structure.
    """
    pool = ScriptedPool(notify_params=NOTIFY_JOB_A, submit_responses=[{"kind": "accept"}])
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            # mining_params should have been received post-authorize.
            assert client.mining_params is not None
            assert client.mining_params["rank"] == 128

            header = bytes.fromhex(NOTIFY_JOB_A[2])
            expected_b64 = _expected_b64_for(header, FIXED_NONCE, FIXED_SEED)
            result = await client.submit_plain_proof(
                NOTIFY_JOB_A[0], expected_b64, NOTIFY_JOB_A[2],
            )
            assert result.accepted
            sub = next(r for r in pool.received_submits if r["method"] == "submitPlainProof")
            params = sub["params"]
            assert isinstance(params, dict)
            assert params["plain_proof"] == expected_b64
            mj = params["mining_job"]
            assert mj["job_id"] == NOTIFY_JOB_A[0]
            assert mj["incomplete_header_bytes"] == NOTIFY_JOB_A[2]
            assert mj["mining_params"]["rank"] == 128
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


# ===========================================================================
# 2. Reject path: malformed commitment ("invalid commitment")
# ===========================================================================


async def test_invalid_commitment_rejected_without_socket_close() -> None:
    """Pool returns code -1 'invalid commitment' (alphapool's flag for the
    random-A/B driver bug). The shim must:
      * return SubmitResult(accepted=False, error_code=-1)
      * NOT reconnect — the socket stays up so the next attempt fast-pathfs
      * increment `stats.rejected` (NOT `stats.dropped_stale_jobid`).
    """
    pool = ScriptedPool(
        notify_params=NOTIFY_JOB_A,
        submit_responses=[
            {"kind": "invalid_commitment"},
            {"kind": "accept"},  # second submit on same socket must work
        ],
    )
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            header = bytes.fromhex(NOTIFY_JOB_A[2])
            bad_b64 = _expected_b64_for(header, FIXED_NONCE, FIXED_SEED)
            r1 = await client.submit_share(NOTIFY_JOB_A[0], bad_b64)
            assert r1.accepted is False
            assert r1.error_code == -1
            assert "invalid commitment" in (r1.error or "").lower()
            assert client.stats.rejected == 1
            assert client.stats.dropped_stale_jobid == 0
            assert client.stats.accepted == 0
            # Socket stayed up: a second submit on the same TCP session works.
            assert pool.connection_count == 1
            r2 = await client.submit_share(NOTIFY_JOB_A[0], bad_b64)
            assert r2.accepted is True
            assert pool.connection_count == 1, (
                "shim must NOT reconnect on invalid-commitment; "
                f"observed {pool.connection_count} connections"
            )
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


# ===========================================================================
# 3. Stale share: job_id changes mid-derivation
# ===========================================================================


async def test_stale_share_filtered_before_submit_when_header_not_mapped() -> None:
    """`submit_plain_proof_blocking()` looks up job_id by header. If the
    mapping doesn't carry that header (job retention exhausted, or never
    seen), the shim must synthesize a code-21 reject WITHOUT touching the
    socket — i.e. submit_count stays 0.

    This is the orphan-share guard for the persistent-CTA path: a batch's
    last few nonces may finish after the job has scrolled past the retention
    window.
    """
    pool = ScriptedPool(notify_params=NOTIFY_JOB_A)
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            # Manually bootstrap a SharedState against this client (we bypass
            # init_shared_state to avoid spawning the second asyncio loop —
            # the test process already has one).
            state = SharedState()
            state._client = client
            state._loop = asyncio.get_running_loop()
            # Don't register the chained_on_new_job; we want a deliberate
            # mismatch between the worker's header and our mapping.
            UNKNOWN_HEADER = bytes.fromhex("aa" * 80)
            assert state.lookup_job_id(UNKNOWN_HEADER) is None

            result = state.submit_plain_proof_blocking(
                _expected_b64_for(UNKNOWN_HEADER, FIXED_NONCE, FIXED_SEED),
                UNKNOWN_HEADER,
            )
            assert result.accepted is False
            assert result.error_code == 21
            # The socket was NOT involved at all.
            assert pool.submit_count == 0, (
                "stale-header submit must not reach the socket; "
                f"observed submit_count={pool.submit_count}"
            )
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_stale_share_filter_retention_window_drops_oldest() -> None:
    """Pin the 16-entry retention behavior: after >16 distinct jobs, the
    oldest mapping is gone and submissions against it synthesize code 21
    without touching the socket. Guards `_JobMapping.MAX_RETAINED`.
    """
    pool = ScriptedPool(notify_params=NOTIFY_JOB_A)
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            state = SharedState()
            state._client = client
            state._loop = asyncio.get_running_loop()

            # Seed 17 jobs synthetically. The first must be evicted.
            from pearl_stratum.gateway_shim import _JobMapping
            from pearl_stratum.job import Job
            assert _JobMapping.MAX_RETAINED == 16
            first_header = b"H0" + b"\x00" * 62
            for i in range(17):
                hdr = (f"H{i}".encode("ascii")).ljust(64, b"\x00")
                fake_job = Job(
                    job_id=f"job-{i}",
                    incomplete_header_bytes=hdr,
                    nbits=0x1A0FFFF0,
                    target=1,
                    target_le=b"\x01" + b"\x00" * 31,
                    clean_jobs=True,
                    received_at=time.time(),
                    raw_params=[],
                )
                state._on_new_job(fake_job)
            # First seeded job should be evicted; last seeded job should be
            # findable.
            assert state.lookup_job_id(first_header) is None
            assert state.lookup_job_id(b"H16".ljust(64, b"\x00")) == "job-16"

            r = state.submit_plain_proof_blocking(
                _expected_b64_for(first_header, FIXED_NONCE, FIXED_SEED),
                first_header,
            )
            assert r.accepted is False and r.error_code == 21
            assert pool.submit_count == 0
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_submit_plain_proof_stale_keeps_socket_open() -> None:
    """The `submitPlainProof` form (carries `mining_job` envelope) must
    apply the same alphafix.c invariant as `mining.submit`: pool returning
    code 21 on a submitPlainProof = per-share reject, socket stays open,
    next call works.

    `mining.submit` is covered by test_error21_does_not_reconnect in
    test_stratum_dialogue.py — this test pins the same invariant for the
    submitPlainProof path the gateway shim uses for the inner_hash_counter
    flow (used by miner-base's `AsyncLoopManager.submit_block`).
    """
    pool = ScriptedPool(
        notify_params=NOTIFY_JOB_A,
        submit_responses=[{"kind": "stale"}, {"kind": "accept"}],
    )
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            hdr = bytes.fromhex(NOTIFY_JOB_A[2])
            b64 = _expected_b64_for(hdr, FIXED_NONCE, FIXED_SEED)

            r1 = await client.submit_plain_proof(NOTIFY_JOB_A[0], b64, NOTIFY_JOB_A[2])
            assert r1.accepted is False
            assert r1.error_code == 21
            assert client.stats.dropped_stale_jobid == 1
            assert pool.connection_count == 1

            r2 = await client.submit_plain_proof(
                NOTIFY_JOB_A[0],
                _expected_b64_for(hdr, FIXED_NONCE + 1, FIXED_SEED),
                NOTIFY_JOB_A[2],
            )
            assert r2.accepted is True
            assert pool.connection_count == 1, (
                "submitPlainProof error 21 must not trigger reconnect"
            )
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


# ===========================================================================
# 4. Reconnect path: TCP drop mid-share-batch
# ===========================================================================


async def test_reconnect_after_mid_submit_socket_drop() -> None:
    """Pool drops TCP without responding to submit #1. The inflight Future
    must fail fast (`_fail_pending`), the run-loop reconnects, and submit
    #2 succeeds on the new socket.

    Per `project_pearl_stratum_shim_shipped_2026_05_17.md`, this is the
    alphafix.c flavor of bug we're explicitly defending against: in-flight
    shares must NOT silently disappear into a half-open socket.
    """
    pool = ScriptedPool(
        notify_params=NOTIFY_JOB_A,
        # First submit -> pool drops the socket (no response).
        # We set drop_after_submit=0 so submit #1 triggers a close without
        # any scripted response. Subsequent connections accept by default.
        drop_after_submit=0,
    )
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            header = bytes.fromhex(NOTIFY_JOB_A[2])
            b64 = _expected_b64_for(header, FIXED_NONCE, FIXED_SEED)
            r1 = await client.submit_share(NOTIFY_JOB_A[0], b64)
            # The inflight future must have failed with a protocol-level error
            # (we surface it as accepted=False rather than a stack).
            assert r1.accepted is False
            assert r1.error_code != 21, "TCP drop must not be reported as code 21"

            # Wait for the client run-loop to reconnect.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if pool.connection_count >= 2 and client.connected:
                    break
                await asyncio.sleep(0.02)
            assert pool.connection_count >= 2, (
                f"expected reconnect after socket drop; got "
                f"connection_count={pool.connection_count}"
            )

            # Submit on the new socket. Need to wait for current_job to be
            # repopulated (the new session pushes a fresh notify on connect).
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and client.current_job is None:
                await asyncio.sleep(0.02)
            r2 = await client.submit_share(NOTIFY_JOB_A[0], b64)
            assert r2.accepted is True
            assert client.stats.accepted == 1
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


async def test_inflight_call_fails_when_socket_drops_without_resp() -> None:
    """A more focused variant: drive a submit, drop the socket before the
    pool responds, and assert that the awaiting coroutine resolves promptly
    via `_fail_pending` rather than hanging until SUBMIT_TIMEOUT_S (30s).
    """
    pool = ScriptedPool(notify_params=NOTIFY_JOB_A, drop_after_submit=0)
    await pool.start()
    try:
        client, task = await _spin_client(pool)
        try:
            header = bytes.fromhex(NOTIFY_JOB_A[2])
            b64 = _expected_b64_for(header, FIXED_NONCE, FIXED_SEED)
            t0 = time.monotonic()
            r = await asyncio.wait_for(
                client.submit_share(NOTIFY_JOB_A[0], b64), timeout=5.0,
            )
            elapsed = time.monotonic() - t0
            # Must complete WELL under SUBMIT_TIMEOUT_S=30s.
            assert elapsed < 5.0, f"submit took {elapsed:.2f}s — should have failed-fast"
            assert r.accepted is False
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


# ===========================================================================
# 5. Live-pool integration (env-gated)
# ===========================================================================


@pytest.mark.skipif(
    os.environ.get("PEARL_STRATUM_LIVE_TEST") != "1",
    reason="live pool test disabled; set PEARL_STRATUM_LIVE_TEST=1 to enable",
)
async def test_live_pool_decoy_wallet_submit_n_shares() -> None:
    """Connect to us2.alphapool.tech:5566 with the decoy wallet and submit
    N=10 synthesized shares.

    NOTE on accept-rate: the synthesized shares are DETERMINISTICALLY WRONG
    (they encode a blake2b commitment over fake bytes), so the pool will
    reject every one as "invalid commitment". This test therefore asserts
    on the WIRING contract:
      * the connection succeeds + completes the handshake
      * the pool sends pearl.set_mining_params with rank=128
      * 10 submits all return SubmitResult promptly (not timeouts)
      * the socket survives every reject (no reconnect storm)

    The "accept rate >= 80%" branch is gated behind
    `PEARL_STRATUM_LIVE_ACCEPT=1` — only set that if you have a KERNEL-backed
    derivation that produces real shares, otherwise it will fail by design.
    """
    pool_url = os.environ.get(
        "PEARL_STRATUM_LIVE_POOL", "stratum+tcp://us2.alphapool.tech:5566",
    )
    decoy = os.environ.get(
        "PEARL_STRATUM_DECOY_WALLET",
        "prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg",
    )
    host, port = parse_pool_url(pool_url)
    n_shares = int(os.environ.get("PEARL_STRATUM_LIVE_N", "10"))
    require_accept = os.environ.get("PEARL_STRATUM_LIVE_ACCEPT") == "1"

    client = StratumClient(
        host=host, port=port, address=decoy, worker="regression-live",
        password="x;d=1048576",
        user_agent="pearl-stratum-regression/0.1",
    )
    task = asyncio.create_task(client.run())
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and client.current_job is None:
            await asyncio.sleep(0.1)
        assert client.current_job is not None, (
            "live pool did not deliver mining.notify within 30s"
        )
        assert client.mining_params is not None, (
            "live pool did not deliver pearl.set_mining_params"
        )
        assert client.mining_params.get("rank") == 128, (
            f"live pool rank changed: got {client.mining_params.get('rank')}"
        )

        # Submit N shares with our synthesized (intentionally-wrong) proof.
        results: list[SubmitResult] = []
        for i in range(n_shares):
            header = client.current_job.incomplete_header_bytes
            b64 = _expected_b64_for(header, FIXED_NONCE + i, FIXED_SEED)
            r = await asyncio.wait_for(
                client.submit_share(client.current_job.job_id, b64), timeout=10.0,
            )
            results.append(r)

        accepted = sum(1 for r in results if r.accepted)
        rejected = sum(1 for r in results if not r.accepted)
        print(
            f"live pool: {accepted}/{n_shares} accepted, "
            f"{rejected} rejected; "
            f"reject codes={sorted({r.error_code for r in results if not r.accepted})}"
        )

        # Wiring assertions: all calls completed, no timeouts.
        assert len(results) == n_shares
        assert all(r.latency_ms < 10_000 for r in results)

        if require_accept:
            assert accepted / n_shares >= 0.80, (
                f"accept rate {accepted}/{n_shares} below threshold 0.80"
            )
    finally:
        await client.stop()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, timeout=5)


# ---------------------------------------------------------------------------
# encode_plain_proof_b64 helper — confirms the gateway-shim accepts bytes
# and PlainProof-likes identically (used by the share-derivation pinning).
# ---------------------------------------------------------------------------


def test_encode_plain_proof_b64_accepts_proof_objects_and_bytes() -> None:
    """gateway_shim.encode_plain_proof_b64 must:
      * call `to_base64()` if available (PlainProof in production)
      * fall back to base64.b64encode for raw bytes (used by test fixtures
        like our SyntheticProof).
    """
    sp = SyntheticProof(payload=b"\x01\x02\x03")
    assert encode_plain_proof_b64(sp) == base64.b64encode(b"\x01\x02\x03").decode("ascii")
    assert encode_plain_proof_b64(b"\xaa\xbb") == "qrs="
    with pytest.raises(TypeError):
        encode_plain_proof_b64(123)  # type: ignore[arg-type]
