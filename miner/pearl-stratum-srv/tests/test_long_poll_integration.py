"""End-to-end long-poll integration: drives the FULL PoolServer stack
(PearldRpc → LongPollingTemplateFetcher → PoolServer._poll_templates →
ingest_template → mining.notify broadcast) against a fake pearld HTTP server.

Proves the long-poll path actually fires and produces notifies through the
real asyncio glue, not just the fetcher in unit-test isolation. The key
assertion: when the fake pearld releases its long-poll (simulating chain tip
advance), a connected stratum client receives a fresh `mining.notify` within
~100ms — vs. the up-to-2s window the fixed-interval poller would have.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import pytest

from pearl_stratum_srv.config import Settings
from pearl_stratum_srv.node_rpc import LongPollingTemplateFetcher, PearldRpc
from pearl_stratum_srv.server import PoolServer

from _fake_pearld import FakePearld  # reuse the fake HTTP server


# ----------------------------------------------------------- fake template


@dataclass
class _Header:
    timestamp: int
    target_bits: int
    previous_block_hash: bytes

    def serialize_without_proof_commitment(self) -> bytes:
        return b"\xfe" * 76

    @property
    def incomplete_header(self):
        return self


@dataclass
class _Template:
    height: int
    header: _Header


def _template_from_raw(raw: dict, _mining_addr: str) -> _Template:
    """Test substitute for pearl-gateway's BlockTemplate.from_get_block_template.
    Extracts only the stratum-visible fields we care about."""
    return _Template(
        height=raw["height"],
        header=_Header(
            timestamp=raw["curtime"],
            target_bits=int(raw["bits"], 16),
            previous_block_hash=bytes.fromhex(raw["previousblockhash"]),
        ),
    )


# ----------------------------------------------------------- fake services


class _Node:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None


class _WorkCache:
    async def update_template(self, t):
        pass


class _Submission:
    async def submit_plain_proof(self, p, t):
        return {"status": "rejected: dont-care"}


# ------------------------------------------------------- LP-enabled server


class _LongPollPoolServer(PoolServer):
    """Override the BlockTemplate conversion so we don't need pearl-gateway."""

    def _template_from_raw_dict(self, raw):
        return _template_from_raw(raw, self.settings.mining_address)


@pytest.fixture
async def fake_pearld():
    p = FakePearld()
    await p.start()
    try:
        yield p
    finally:
        await p.stop()


@pytest.fixture
async def long_poll_server(fake_pearld):
    """A real PoolServer wired to real PearldRpc + LongPollingTemplateFetcher
    pointed at the fake pearld."""
    rpc = PearldRpc(fake_pearld.url, "u", "p", request_timeout_secs=10.0)
    fetcher = LongPollingTemplateFetcher(rpc, long_poll_timeout_secs=5.0)

    settings = Settings(
        rpc_url=fake_pearld.url,
        rpc_user="u",
        rpc_password="p",
        mining_address="prl1stub",
        listen_host="127.0.0.1",
        listen_port=0,
        long_poll=True,
        poll_interval=0.5,  # fast retry on error in tests
    )
    server = _LongPollPoolServer(
        settings=settings,
        node=_Node(),
        work_cache=_WorkCache(),
        submission=_Submission(),
        template_fetcher=fetcher,
    )
    # Enter the RPC session manually (serve_forever would, but we drive
    # the poller manually here for tight control).
    await rpc.__aenter__()
    port = await server.start_listener(port=0)
    try:
        yield server, port, fake_pearld
    finally:
        await server.stop_listener()
        await rpc.__aexit__(None, None, None)


# ------------------------------------------------------------- helpers


async def _connect(port: int):
    return await asyncio.open_connection("127.0.0.1", port)


async def _send(writer, obj):
    writer.write((json.dumps(obj) + "\n").encode())
    await writer.drain()


async def _read_frame(reader):
    line = await asyncio.wait_for(reader.readline(), timeout=3.0)
    assert line, "EOF"
    return json.loads(line)


# --------------------------------------------------------------- tests


async def test_first_template_fetch_hits_pearld_with_no_longpollid(long_poll_server):
    server, _, fake = long_poll_server
    # Manually drive one poll cycle.
    poller = asyncio.create_task(server._poll_templates(), name="poll")
    try:
        # Wait for at least one request to land at fake pearld.
        for _ in range(50):
            if fake.received:
                break
            await asyncio.sleep(0.02)
        assert fake.received, "poller never hit pearld"
        first = fake.received[0]["params"][0]
        assert "longpollid" not in first
        assert "longpoll" in first["capabilities"]
    finally:
        poller.cancel()
        try:
            await poller
        except asyncio.CancelledError:
            pass


async def test_subscribed_client_gets_notify_after_chain_advance(long_poll_server):
    """The key win: client should see a fresh notify within ~100ms of pearld
    releasing the long-poll, not the 2s fixed-poll window."""
    server, port, fake = long_poll_server

    # First template is already prepared; poller fetches it immediately.
    poller = asyncio.create_task(server._poll_templates(), name="poll")
    try:
        # Wait for the first template to be ingested.
        for _ in range(100):
            if server.jobs.latest() is not None:
                break
            await asyncio.sleep(0.02)
        assert server.jobs.latest() is not None, "first template never ingested"
        first_job_id = server.jobs.latest().job_id

        # Subscribe a client; reads the initial 4 frames including the notify
        # for the current template.
        reader, writer = await _connect(port)
        try:
            await _send(writer, {"id": 1, "method": "mining.subscribe", "params": []})
            for _ in range(4):
                await _read_frame(reader)

            # Now simulate a chain advance: change pearld's template prev_hash
            # and rotate the longpollid, then release the long-poll.
            fake.template = dict(fake.template)
            fake.template["previousblockhash"] = "cd" * 32
            fake.template["height"] = fake.template["height"] + 1
            fake.template["longpollid"] = "lpid-v2"
            fake.block_long_poll()
            # The fetcher should currently be awaiting a long-poll response;
            # release it now.
            t0 = time.monotonic()
            fake.release_long_poll()

            # Client should get a fresh notify with a different job_id.
            notify = await _read_frame(reader)
            elapsed = time.monotonic() - t0
            assert notify["method"] == "mining.notify"
            assert notify["params"][0] != first_job_id
            assert notify["params"][6] is True  # clean_jobs=True on chain advance
            # Should arrive fast — well under fixed-poll's 2s. Generous bound for CI flake.
            assert elapsed < 1.0, f"notify took {elapsed:.2f}s — long-poll path likely broken"
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        poller.cancel()
        try:
            await poller
        except asyncio.CancelledError:
            pass


async def test_rpc_error_resets_longpollid_and_falls_back_to_poll(long_poll_server):
    """If pearld returns an RPC error, the fetcher must reset its longpollid
    so subsequent calls don't hang on a handle pearld no longer recognizes."""
    server, _, fake = long_poll_server
    poller = asyncio.create_task(server._poll_templates(), name="poll")
    try:
        # Wait for first successful fetch to seed longpollid.
        for _ in range(100):
            if server.template_fetcher._longpollid is not None:
                break
            await asyncio.sleep(0.02)
        assert server.template_fetcher._longpollid == "lpid-v1"

        # Now force pearld to error.
        fake.next_response = {
            "jsonrpc": "2.0",
            "id": 99,
            "result": None,
            "error": {"code": -8, "message": "lpid expired"},
        }
        # Wait long enough for the poller to hit the error path (poll_interval=0.5).
        await asyncio.sleep(1.0)
        # Longpollid should have been cleared.
        assert server.template_fetcher._longpollid is None

        # Clear the override so subsequent requests succeed.
        fake.next_response = None
        # Wait for recovery.
        for _ in range(100):
            if server.template_fetcher._longpollid is not None:
                break
            await asyncio.sleep(0.02)
        assert server.template_fetcher._longpollid == "lpid-v1"
    finally:
        poller.cancel()
        try:
            await poller
        except asyncio.CancelledError:
            pass


async def test_metrics_template_age_stays_fresh_under_long_poll(long_poll_server):
    """Sanity: with the long-poll loop running and tip updating, the metrics
    template_minted_at gauge tracks real time."""
    server, _, fake = long_poll_server
    poller = asyncio.create_task(server._poll_templates(), name="poll")
    try:
        # Wait for first ingest.
        for _ in range(100):
            if server.metrics.template_minted_at > 0:
                break
            await asyncio.sleep(0.02)
        first_minted = server.metrics.template_minted_at
        assert first_minted > 0

        # Trigger a chain advance + release.
        fake.template = dict(fake.template)
        fake.template["previousblockhash"] = "ef" * 32
        fake.template["height"] = fake.template["height"] + 1
        fake.template["longpollid"] = "lpid-v3"
        fake.block_long_poll()
        await asyncio.sleep(0.05)
        fake.release_long_poll()

        # Wait for the next ingest.
        for _ in range(100):
            if server.metrics.template_minted_at > first_minted:
                break
            await asyncio.sleep(0.02)
        assert server.metrics.template_minted_at > first_minted
        assert server.metrics.template_height == fake.template["height"]
    finally:
        poller.cancel()
        try:
            await poller
        except asyncio.CancelledError:
            pass
