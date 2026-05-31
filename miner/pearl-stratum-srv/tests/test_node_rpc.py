"""Long-poll template fetcher: drives a fake pearld HTTP server and checks
both the wire-level RPC behavior and the longpollid handshake."""

from __future__ import annotations

import asyncio
import json

import pytest

from pearl_stratum_srv.node_rpc import LongPollingTemplateFetcher, PearldRpc, RpcError

from _fake_pearld import FakePearld  # noqa: E402


@pytest.fixture
async def fake_pearld():
    p = FakePearld()
    await p.start()
    try:
        yield p
    finally:
        await p.stop()


# ----------------------------------------------------------------- tests


async def test_pearld_rpc_sends_jsonrpc_2_request(fake_pearld):
    async with PearldRpc(fake_pearld.url, "u", "p") as rpc:
        result = await rpc.call("getblocktemplate", [{"capabilities": ["coinbasevalue"]}])
    assert result["height"] == 54_374
    sent = fake_pearld.received[0]
    assert sent["jsonrpc"] == "2.0"
    assert sent["method"] == "getblocktemplate"
    assert sent["id"] == 1


async def test_pearld_rpc_raises_on_rpc_error(fake_pearld):
    fake_pearld.next_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": None,
        "error": {"code": -8, "message": "bad params"},
    }
    async with PearldRpc(fake_pearld.url, "u", "p") as rpc:
        with pytest.raises(RpcError, match="bad params"):
            await rpc.call("getblocktemplate", [])


async def test_fetcher_includes_longpollid_on_second_call(fake_pearld):
    async with PearldRpc(fake_pearld.url, "u", "p") as rpc:
        fetcher = LongPollingTemplateFetcher(rpc)
        await fetcher.fetch()
        await fetcher.fetch()

    # First request: no longpollid.
    req1 = fake_pearld.received[0]["params"][0]
    assert "longpollid" not in req1
    # Second request: should carry the one pearld returned.
    req2 = fake_pearld.received[1]["params"][0]
    assert req2["longpollid"] == "lpid-v1"


async def test_fetcher_includes_longpoll_capability(fake_pearld):
    async with PearldRpc(fake_pearld.url, "u", "p") as rpc:
        fetcher = LongPollingTemplateFetcher(rpc)
        await fetcher.fetch()
    req = fake_pearld.received[0]["params"][0]
    assert "longpoll" in req["capabilities"]


async def test_fetcher_blocks_until_pearld_responds(fake_pearld):
    """Simulates the real long-poll: pearld holds the call open until tip changes."""
    fake_pearld.block_long_poll()

    async with PearldRpc(fake_pearld.url, "u", "p") as rpc:
        fetcher = LongPollingTemplateFetcher(rpc)
        task = asyncio.create_task(fetcher.fetch())

        # Give the request time to land at the fake, then verify it's pending.
        await asyncio.sleep(0.1)
        assert not task.done(), "fetcher should still be blocked on pearld"

        # Release the simulated long-poll.
        fake_pearld.release_long_poll()
        result = await asyncio.wait_for(task, timeout=2.0)
        assert result["height"] == 54_374


async def test_fetcher_reset_clears_longpollid(fake_pearld):
    async with PearldRpc(fake_pearld.url, "u", "p") as rpc:
        fetcher = LongPollingTemplateFetcher(rpc)
        await fetcher.fetch()
        assert fetcher._longpollid == "lpid-v1"
        fetcher.reset()
        assert fetcher._longpollid is None
        await fetcher.fetch()
        # Reset → next request has no longpollid.
        req = fake_pearld.received[-1]["params"][0]
        assert "longpollid" not in req


async def test_fetcher_handles_missing_longpollid_in_response(fake_pearld):
    """If pearld doesn't return longpollid (e.g., long-poll disabled in node
    config), the fetcher just polls every interval — must not crash."""
    fake_pearld.template = {**fake_pearld.template}
    fake_pearld.template.pop("longpollid", None)

    async with PearldRpc(fake_pearld.url, "u", "p") as rpc:
        fetcher = LongPollingTemplateFetcher(rpc)
        await fetcher.fetch()
        assert fetcher._longpollid is None
        # Second call also works
        await fetcher.fetch()
    req = fake_pearld.received[1]["params"][0]
    assert "longpollid" not in req


async def test_pearld_rpc_outside_context_manager_raises():
    rpc = PearldRpc("http://x", "u", "p")
    with pytest.raises(RuntimeError, match="async with"):
        await rpc.call("getblocktemplate", [])
