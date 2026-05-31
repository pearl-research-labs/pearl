"""End-to-end integration tests.

Drives the real PoolServer over a real TCP socket with fake versions of
pearl-gateway services and pearl_mining. Validates:

  - subscribe handshake emits the expected push frames in the right order
  - authorize is unconditional
  - submit against a stale job_id returns error[21] with socket open
  - submit against a valid job_id is acked + reaches SubmissionService
  - new template broadcasts notify to every subscribed client (clean=True)

Runs without pearl_gateway or pearl_mining installed: conftest installs a
fake `pearl_mining` module into sys.modules BEFORE connection.py imports it.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any

import pytest

# pearl_mining shim is installed in conftest.py — safe to import server now.
from pearl_stratum_srv.config import Settings
from pearl_stratum_srv.server import PoolServer


# ------------------------------------------------------------ fake template


@dataclass
class _FakeHeader:
    timestamp: int = 0x6A093061
    target_bits: int = 0x1A0FFFF0
    previous_block_hash: bytes = b"\xab" * 32

    def serialize_without_proof_commitment(self) -> bytes:
        return b"\xfe" * 76

    @property
    def incomplete_header(self):
        """Matches the real PearlHeader.incomplete_header attribute that
        server.submit_share passes to verify_plain_proof."""
        return self


@dataclass
class _FakeTemplate:
    height: int
    header: _FakeHeader


# ----------------------------------------------------------- fake services


class _FakeNode:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class _FakeWorkCache:
    def __init__(self):
        self.updates: list = []

    async def update_template(self, template):
        self.updates.append(template)


class _FakeSubmissionService:
    def __init__(self, response: dict | None = None):
        self.response = response or {"status": "rejected: difficulty"}
        self.calls: list[tuple[Any, Any]] = []

    async def submit_plain_proof(self, plain_proof, template) -> dict:
        self.calls.append((plain_proof, template))
        return self.response


# -------------------------------------------------------------- fixtures


@pytest.fixture
def submission_service():
    return _FakeSubmissionService()


@pytest.fixture
async def running_server(submission_service):
    """A PoolServer with fake deps, listening on an ephemeral port.

    By default `verify_plain_proof` is wired to a stub that returns
    `(True, "ok")` so the legacy "every share reaches submission" tests
    keep passing without modification. Tests that want the production
    behavior (skip submission unless it's a block) can swap it via
    `server.verify_plain_proof = ...` before sending submits.
    """
    settings = Settings(
        rpc_url="http://stub",
        rpc_user="x",
        rpc_password="y",
        mining_address="prl1stub",
        listen_host="127.0.0.1",
        listen_port=0,
    )
    server = PoolServer(
        settings=settings,
        node=_FakeNode(),
        work_cache=_FakeWorkCache(),
        submission=submission_service,
        verify_plain_proof=lambda h, p: (True, "ok"),
    )
    port = await server.start_listener(port=0)
    try:
        yield server, port
    finally:
        await server.stop_listener()


async def _connect(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_connection("127.0.0.1", port)


async def _send(writer: asyncio.StreamWriter, obj: dict) -> None:
    writer.write((json.dumps(obj) + "\n").encode())
    await writer.drain()


async def _read_frame(reader: asyncio.StreamReader) -> dict:
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    assert line, "EOF while waiting for frame"
    return json.loads(line)


async def _read_frames(reader: asyncio.StreamReader, n: int) -> list[dict]:
    return [await _read_frame(reader) for _ in range(n)]


# ---------------------------------------------------------------- tests


async def test_subscribe_pushes_params_then_diff_then_notify(running_server):
    server, port = running_server
    # Seed a template so subscribe's notify push has something to send.
    await server.ingest_template(_FakeTemplate(height=0xD446, header=_FakeHeader()))

    reader, writer = await _connect(port)
    try:
        await _send(writer, {"id": 1, "method": "mining.subscribe", "params": ["alpha-miner/0.1"]})

        # 4 frames expected: subscribe reply, set_mining_params, set_difficulty, notify.
        frames = await _read_frames(reader, 4)

        # Frame 1 — subscribe reply.
        assert frames[0]["id"] == 1
        assert frames[0]["error"] is None
        result = frames[0]["result"]
        assert result[1] == ""  # empty extranonce1
        assert result[2] == 0   # zero extranonce2_size
        subs = {row[0] for row in result[0]}
        assert subs == {"mining.set_difficulty", "mining.notify"}

        # Frame 2 — pearl.set_mining_params, single-element list per spec.
        assert frames[1]["method"] == "pearl.set_mining_params"
        params_payload = frames[1]["params"][0]
        assert params_payload["rank"] == 128
        assert params_payload["m"] == 131072
        assert params_payload["mma_type"] == "Int7xInt7ToInt32"

        # Frame 3 — initial difficulty.
        assert frames[2]["method"] == "mining.set_difficulty"
        assert frames[2]["params"] == [1]

        # Frame 4 — notify carrying the current job.
        assert frames[3]["method"] == "mining.notify"
        notify_params = frames[3]["params"]
        assert notify_params[0].startswith("0000d446-")  # height-prefixed job_id
        assert notify_params[5] == "1a0ffff0"            # nbits hex
        assert notify_params[6] is True                  # clean_jobs
    finally:
        writer.close()
        await writer.wait_closed()


async def test_authorize_returns_true(running_server):
    server, port = running_server
    reader, writer = await _connect(port)
    try:
        await _send(
            writer,
            {"id": 7, "method": "mining.authorize", "params": ["prl1xyz.workerA", "x;d=1048576"]},
        )
        reply = await _read_frame(reader)
        assert reply == {"jsonrpc": "2.0", "id": 7, "result": True, "error": None}
    finally:
        writer.close()
        await writer.wait_closed()


async def test_submit_stale_job_id_returns_error_21_socket_stays_open(running_server):
    server, port = running_server
    # Don't seed any template — every job_id is stale.
    reader, writer = await _connect(port)
    try:
        # Need to subscribe first so connection is in a sensible state, but
        # the registry is empty so submits should still be stale.
        await _send(writer, {"id": 1, "method": "mining.subscribe", "params": []})
        # Subscribe pushes set_mining_params + set_difficulty + (no notify since no job).
        await _read_frames(reader, 3)

        # Stale submit.
        await _send(
            writer,
            {
                "id": 42,
                "method": "mining.submit",
                "params": ["worker", "deadbeef-0001", base64.b64encode(b"proof").decode()],
            },
        )
        err = await _read_frame(reader)
        assert err["id"] == 42
        assert err["error"] == [21, "Job not found", None]
        assert err["result"] is None

        # Socket must stay open — second submit should still work.
        await _send(
            writer,
            {
                "id": 43,
                "method": "mining.submit",
                "params": ["worker", "another-stale", base64.b64encode(b"proof").decode()],
            },
        )
        err2 = await _read_frame(reader)
        assert err2["id"] == 43
        assert err2["error"][0] == 21
    finally:
        writer.close()
        await writer.wait_closed()


async def test_valid_submit_acks_true_and_reaches_submission(running_server, submission_service):
    server, port = running_server
    template = _FakeTemplate(height=0xD500, header=_FakeHeader())
    entry = await server.ingest_template(template)

    reader, writer = await _connect(port)
    try:
        await _send(writer, {"id": 1, "method": "mining.subscribe", "params": []})
        await _read_frames(reader, 4)  # subscribe + 3 pushes (params, diff, notify)

        proof_bytes = b"AAAA-fake-plain-proof"
        await _send(
            writer,
            {
                "id": 99,
                "method": "mining.submit",
                "params": ["worker", entry.job_id, base64.b64encode(proof_bytes).decode()],
            },
        )
        reply = await _read_frame(reader)
        assert reply == {"jsonrpc": "2.0", "id": 99, "result": True, "error": None}

        # SubmissionService got the proof + the right template.
        assert len(submission_service.calls) == 1
        proof, t = submission_service.calls[0]
        assert proof.payload == proof_bytes
        assert t is template
    finally:
        writer.close()
        await writer.wait_closed()


async def test_block_acceptance_logged_but_share_still_acked_true(running_server, submission_service):
    submission_service.response = {"status": "accepted"}
    server, port = running_server
    entry = await server.ingest_template(_FakeTemplate(height=0xD600, header=_FakeHeader()))

    reader, writer = await _connect(port)
    try:
        await _send(writer, {"id": 1, "method": "mining.subscribe", "params": []})
        await _read_frames(reader, 4)
        await _send(
            writer,
            {
                "id": 50,
                "method": "mining.submit",
                "params": ["worker", entry.job_id, base64.b64encode(b"x").decode()],
            },
        )
        reply = await _read_frame(reader)
        # Per design, share-ack is always true; block-find is implicit via coinbase.
        assert reply["result"] is True
        assert reply["error"] is None
    finally:
        writer.close()
        await writer.wait_closed()


async def test_new_template_broadcasts_notify_with_clean_true_to_all_clients(running_server):
    server, port = running_server
    # Three clients subscribe.
    pairs = [await _connect(port) for _ in range(3)]
    try:
        for _i, (r, w) in enumerate(pairs):
            await _send(w, {"id": 1, "method": "mining.subscribe", "params": []})
            # Subscribe reply + 2 pushes (params, diff). No notify yet (no job).
            await _read_frames(r, 3)

        # New template arrives via the test helper (mimics what _poll_templates does).
        await server.ingest_template(_FakeTemplate(height=0xD700, header=_FakeHeader()))

        # Each client should receive a notify with clean_jobs=True.
        for r, _ in pairs:
            notify = await _read_frame(r)
            assert notify["method"] == "mining.notify"
            assert notify["params"][0].startswith("0000d700-")
            assert notify["params"][6] is True
    finally:
        for _r, w in pairs:
            w.close()
            await w.wait_closed()


async def test_unknown_method_returns_error_25(running_server):
    server, port = running_server
    reader, writer = await _connect(port)
    try:
        await _send(writer, {"id": 5, "method": "mining.something_unsupported", "params": []})
        reply = await _read_frame(reader)
        assert reply["id"] == 5
        assert reply["error"][0] == 25
    finally:
        writer.close()
        await writer.wait_closed()


async def test_metrics_updated_through_real_submit_flow(running_server, submission_service):
    """End-to-end: shares submitted over TCP must show up in Metrics."""
    server, port = running_server
    entry = await server.ingest_template(_FakeTemplate(height=0xD800, header=_FakeHeader()))
    assert server.metrics.template_height == 0xD800

    reader, writer = await _connect(port)
    try:
        await _send(writer, {"id": 1, "method": "mining.subscribe", "params": []})
        await _read_frames(reader, 4)
        await _send(
            writer,
            {"id": 2, "method": "mining.authorize", "params": ["prl1.rig04gpu0", "x"]},
        )
        await _read_frame(reader)

        # Two accepted shares.
        for sid in (10, 11):
            await _send(
                writer,
                {
                    "id": sid,
                    "method": "mining.submit",
                    "params": ["prl1.rig04gpu0", entry.job_id, base64.b64encode(b"x").decode()],
                },
            )
            await _read_frame(reader)

        # One stale share.
        await _send(
            writer,
            {
                "id": 12,
                "method": "mining.submit",
                "params": ["prl1.rig04gpu0", "deadbeef-9999", base64.b64encode(b"x").decode()],
            },
        )
        await _read_frame(reader)

        assert server.metrics.shares_total[("prl1.rig04gpu0", "accepted")] == 2
        assert server.metrics.shares_total[("prl1.rig04gpu0", "stale")] == 1
        assert server.metrics.connected_miners == 1
    finally:
        writer.close()
        await writer.wait_closed()


async def test_submission_skipped_when_verify_plain_proof_returns_false(
    running_server, submission_service
):
    """Phase-1 verify says 'not a block' → SubmissionService is NOT called.
    This is the load-bearing bug fix: it stops Plonky2 prover from firing on
    every share, which would OOM the host."""
    server, port = running_server
    server.verify_plain_proof = lambda h, p: (False, "difficulty not met")
    entry = await server.ingest_template(_FakeTemplate(height=0xE100, header=_FakeHeader()))

    reader, writer = await _connect(port)
    try:
        await _send(writer, {"id": 1, "method": "mining.subscribe", "params": []})
        await _read_frames(reader, 4)
        await _send(
            writer,
            {
                "id": 9,
                "method": "mining.submit",
                "params": ["worker", entry.job_id, base64.b64encode(b"x").decode()],
            },
        )
        reply = await _read_frame(reader)
        assert reply["result"] is True  # share still acked for liveness

        # The critical assertion: SubmissionService NEVER got called.
        assert submission_service.calls == []
        # Share still counted in metrics for liveness. Worker key is
        # "anonymous" because we didn't send mining.authorize first.
        assert server.metrics.shares_total[("anonymous", "accepted")] == 1
    finally:
        writer.close()
        await writer.wait_closed()


async def test_submission_invoked_when_verify_plain_proof_returns_true(
    running_server, submission_service
):
    """Phase-1 verify says 'this IS a block' → prove + submit fires."""
    server, port = running_server
    # Default fixture wires verify→(True, "ok") so this exercises the block path.
    submission_service.response = {"status": "accepted"}
    entry = await server.ingest_template(_FakeTemplate(height=0xE200, header=_FakeHeader()))

    reader, writer = await _connect(port)
    try:
        await _send(writer, {"id": 1, "method": "mining.subscribe", "params": []})
        await _read_frames(reader, 4)
        await _send(
            writer,
            {
                "id": 10,
                "method": "mining.submit",
                "params": ["worker", entry.job_id, base64.b64encode(b"x").decode()],
            },
        )
        await _read_frame(reader)

        assert len(submission_service.calls) == 1
        assert server.metrics.blocks_total["accepted"] == 1
    finally:
        writer.close()
        await writer.wait_closed()


async def test_verify_raising_exception_yields_clean_error_response(
    running_server, submission_service
):
    """If verify_plain_proof raises (e.g. Rust binding bug or bad input),
    we record nothing as a block and don't crash the connection."""
    server, port = running_server
    server.verify_plain_proof = lambda h, p: (_ for _ in ()).throw(RuntimeError("boom"))
    entry = await server.ingest_template(_FakeTemplate(height=0xE300, header=_FakeHeader()))

    reader, writer = await _connect(port)
    try:
        await _send(writer, {"id": 1, "method": "mining.subscribe", "params": []})
        await _read_frames(reader, 4)
        await _send(
            writer,
            {
                "id": 11,
                "method": "mining.submit",
                "params": ["worker", entry.job_id, base64.b64encode(b"x").decode()],
            },
        )
        ack = await _read_frame(reader)
        # Connection-level result is still ack=true for the wire; the verify
        # failure is logged + counted as a share (we don't disconnect).
        assert ack["result"] is True
        assert submission_service.calls == []
        # Subsequent submits still work on the same connection.
        await _send(writer, {"id": 12, "method": "mining.authorize", "params": ["w", "p"]})
        ok = await _read_frame(reader)
        assert ok["result"] is True
    finally:
        writer.close()
        await writer.wait_closed()


async def test_listener_accepts_400kb_submit_line(running_server, submission_service):
    """Default asyncio.StreamReader.readline limit is 64 KiB. Real alpha-miner
    submits carry a ~368 KB base64 plain_proof on one line. Pool must read
    these without LimitOverrunError."""
    server, port = running_server
    server.verify_plain_proof = lambda h, p: (False, "diff")  # skip prove path
    entry = await server.ingest_template(_FakeTemplate(height=0xE400, header=_FakeHeader()))

    reader, writer = await _connect(port)
    try:
        await _send(writer, {"id": 1, "method": "mining.subscribe", "params": []})
        await _read_frames(reader, 4)

        # 400 KB of base64 data — well over the default 64 KiB readline limit.
        big_proof = base64.b64encode(b"x" * 300_000).decode()
        assert len(big_proof) >= 400_000

        await _send(
            writer,
            {
                "id": 50,
                "method": "mining.submit",
                "params": ["worker", entry.job_id, big_proof],
            },
        )
        reply = await _read_frame(reader)
        # Got a real reply — listener parsed the giant line successfully.
        assert reply["id"] == 50
        assert reply["result"] is True
    finally:
        writer.close()
        await writer.wait_closed()


async def test_malformed_json_returns_error_without_closing(running_server):
    server, port = running_server
    reader, writer = await _connect(port)
    try:
        writer.write(b"{not valid json}\n")
        await writer.drain()
        reply = await _read_frame(reader)
        assert reply["error"][0] == -32602  # INVALID_PARAMS_CODE
        assert reply["id"] is None

        # Connection still alive — a normal frame works after.
        await _send(writer, {"id": 1, "method": "mining.authorize", "params": ["w", "p"]})
        ok = await _read_frame(reader)
        assert ok["result"] is True
    finally:
        writer.close()
        await writer.wait_closed()
