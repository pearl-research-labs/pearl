"""Tests for ``PersistentUpstream`` and ``UpstreamPool``.

These connect to the in-process ``MockPool`` fixture (see ``conftest.py``).
"""

from __future__ import annotations

import asyncio

import pytest

from pearl_stratum_proxy.persistent_upstream import (
    CachedHandshake,
    PersistentUpstream,
    UpstreamPool,
)

WORKER = "prl1pja266dfa7kcg0xdagaacy0y7x60h7qrw3tcau4enx4gwnmmyxxvs7ep7ad.rig03v2.gpu2"


# ----------------------------------------------------------------------
# CachedHandshake.


def test_cache_complete_only_after_all_three_fields() -> None:
    c = CachedHandshake()
    assert not c.is_complete()
    c.configure_response = b"x"
    c.subscribe_response = b"y"
    assert not c.is_complete()
    c.authorize_response = b"z"
    assert c.is_complete()


# ----------------------------------------------------------------------
# PersistentUpstream lifecycle.


async def test_connect_opens_socket_and_is_alive(mock_pool) -> None:
    up = await PersistentUpstream.connect(WORKER, mock_pool.host, mock_pool.port)
    try:
        assert up.is_alive
        # Mock pool registers the inbound connection
        await asyncio.sleep(0.05)
        assert len(mock_pool.connections) == 1
    finally:
        await up.close()
    assert not up.is_alive


async def test_send_to_upstream_forwards_to_pool(mock_pool) -> None:
    up = await PersistentUpstream.connect(WORKER, mock_pool.host, mock_pool.port)
    try:
        await up.send_to_upstream(b'{"id":1,"method":"mining.configure","params":[["pearl/v1"],{}]}\n')
        # Give the pool a chance to receive and respond.
        await asyncio.sleep(0.1)
        assert any(o.get("method") == "mining.configure" for o in mock_pool.connections[0].received)
    finally:
        await up.close()


# ----------------------------------------------------------------------
# Attach / detach refcount enforcement (per-upstream per-client design).


async def test_attach_increments_refcount_to_one(mock_pool) -> None:
    up = await PersistentUpstream.connect(WORKER, mock_pool.host, mock_pool.port)
    try:
        # Build a fake client writer pair.
        client_reader, client_writer = await _make_loopback_pair()
        try:
            buffered = await up.attach_client(client_writer)
            assert buffered == []
            assert up.has_attached_client
        finally:
            client_writer.close()
            try:
                await client_writer.wait_closed()
            except Exception:
                pass
    finally:
        await up.close()


async def test_second_attach_raises(mock_pool) -> None:
    """Multiplexing two clients onto one upstream is a contract violation."""
    up = await PersistentUpstream.connect(WORKER, mock_pool.host, mock_pool.port)
    try:
        _r1, w1 = await _make_loopback_pair()
        _r2, w2 = await _make_loopback_pair()
        try:
            await up.attach_client(w1)
            with pytest.raises(RuntimeError, match="refcount"):
                await up.attach_client(w2)
        finally:
            for w in (w1, w2):
                w.close()
                try:
                    await w.wait_closed()
                except Exception:
                    pass
    finally:
        await up.close()


async def test_detach_then_reattach_works(mock_pool) -> None:
    up = await PersistentUpstream.connect(WORKER, mock_pool.host, mock_pool.port)
    try:
        _r1, w1 = await _make_loopback_pair()
        _r2, w2 = await _make_loopback_pair()
        try:
            await up.attach_client(w1)
            await up.detach_client()
            # Now a different client can attach.
            buf = await up.attach_client(w2)
            assert buf == []
        finally:
            for w in (w1, w2):
                w.close()
                try:
                    await w.wait_closed()
                except Exception:
                    pass
    finally:
        await up.close()


# ----------------------------------------------------------------------
# The headline feature: pool messages arriving while detached are
# buffered and replayed on the next attach.


async def test_pool_messages_buffered_while_no_client(mock_pool) -> None:
    """Reproduces the reconnect_drop_share scenario:
    1. Client is attached, gets initial state.
    2. Client detaches (the simulated FIN).
    3. Pool pushes a fresh mining.notify.
    4. New client attaches -> sees the notify immediately in the
       returned buffer.
    """
    up = await PersistentUpstream.connect(WORKER, mock_pool.host, mock_pool.port)
    try:
        _r1, w1 = await _make_loopback_pair()
        await up.attach_client(w1)
        # Detach (the swallow-close)
        await up.detach_client()
        w1.close()
        try:
            await w1.wait_closed()
        except Exception:
            pass

        # Pool pushes a notify to the upstream while no client attached.
        notify = (
            b'{"method":"mining.notify","params":'
            b'["NEWJOB","ph","cb",1,"nb","nt",true]}\n'
        )
        # Use the mock pool's connection writer to push.
        await asyncio.sleep(0.05)  # let pool register
        assert mock_pool.connections, "mock pool didn't see the upstream connect"
        await mock_pool.push(mock_pool.connections[0], notify)

        # Let the upstream reader loop see it.
        await asyncio.sleep(0.1)

        # New client attaches.
        _r2, w2 = await _make_loopback_pair()
        try:
            buffered = await up.attach_client(w2)
            assert notify in buffered, f"expected notify in buffered, got {buffered!r}"
            # And the upstream's cache learned the latest notify too.
            assert up.cache.last_notify == notify
        finally:
            w2.close()
            try:
                await w2.wait_closed()
            except Exception:
                pass
    finally:
        await up.close()


async def test_set_difficulty_and_set_mining_params_cached(mock_pool) -> None:
    """The cache should learn set_difficulty and pearl.set_mining_params
    from upstream notifications."""
    up = await PersistentUpstream.connect(WORKER, mock_pool.host, mock_pool.port)
    try:
        _r, w = await _make_loopback_pair()
        await up.attach_client(w)
        try:
            await asyncio.sleep(0.05)
            assert mock_pool.connections
            pool_conn = mock_pool.connections[0]

            sd_line = b'{"method":"mining.set_difficulty","params":[262144]}\n'
            mp_line = (
                b'{"method":"pearl.set_mining_params",'
                b'"params":[{"m":1,"n":2,"k":3,"rank":4}]}\n'
            )
            await mock_pool.push(pool_conn, sd_line)
            await mock_pool.push(pool_conn, mp_line)
            await asyncio.sleep(0.05)
            assert up.cache.last_set_difficulty == sd_line
            assert up.cache.set_mining_params == mp_line
        finally:
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass
    finally:
        await up.close()


async def test_buffer_bounded_oldest_dropped(mock_pool) -> None:
    """When the alpha-miner never reconnects, the proxy shouldn't OOM.
    The buffer is bounded and oldest entries are dropped."""
    up = await PersistentUpstream.connect(WORKER, mock_pool.host, mock_pool.port)
    try:
        # Don't attach a client. All inbound notifies should be buffered.
        await asyncio.sleep(0.05)
        assert mock_pool.connections
        pool_conn = mock_pool.connections[0]
        # Push more than the limit.
        limit = PersistentUpstream._PENDING_LIMIT
        for i in range(limit + 10):
            await mock_pool.push(
                pool_conn,
                (f'{{"method":"mining.notify","params":["J{i}"]}}\n').encode("utf-8"),
            )
        await asyncio.sleep(0.2)

        _r, w = await _make_loopback_pair()
        try:
            buffered = await up.attach_client(w)
            # We should never exceed the limit.
            assert len(buffered) <= limit
            # Oldest should have been dropped — first item shouldn't be J0.
            first_seen = buffered[0]
            assert b'"J0"' not in first_seen, "oldest entry not dropped"
        finally:
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass
    finally:
        await up.close()


# ----------------------------------------------------------------------
# UpstreamPool: per-worker isolation, get_or_create semantics.


async def test_pool_creates_distinct_upstreams_per_worker(mock_pool) -> None:
    pool = UpstreamPool(mock_pool.host, mock_pool.port)
    try:
        up_a = await pool.get_or_create(WORKER + ".A")
        up_b = await pool.get_or_create(WORKER + ".B")
        assert up_a is not up_b
        # Two distinct TCP connections to the pool.
        await asyncio.sleep(0.05)
        assert len(mock_pool.connections) == 2
        assert len(pool) == 2
    finally:
        await pool.close_all()


async def test_pool_returns_same_upstream_on_same_worker(mock_pool) -> None:
    pool = UpstreamPool(mock_pool.host, mock_pool.port)
    try:
        up_a = await pool.get_or_create(WORKER)
        up_b = await pool.get_or_create(WORKER)
        assert up_a is up_b
        assert len(pool) == 1
    finally:
        await pool.close_all()


async def test_pool_rebuilds_dead_upstream(mock_pool) -> None:
    """If the cached upstream's TCP died, get_or_create makes a new one."""
    pool = UpstreamPool(mock_pool.host, mock_pool.port)
    try:
        up_a = await pool.get_or_create(WORKER)
        await up_a.close()
        assert not up_a.is_alive
        up_b = await pool.get_or_create(WORKER)
        assert up_b is not up_a
        assert up_b.is_alive
    finally:
        await pool.close_all()


# ----------------------------------------------------------------------
# Helper: build a connected (reader, writer) pair over loopback.


async def _make_loopback_pair() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Returns a writer whose bytes get sent into the void via a tiny
    in-process echo server. Used to satisfy the asyncio.StreamWriter
    interface in attach tests without requiring a real client."""
    server: list[asyncio.base_events.Server] = []

    async def _drain(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            while True:
                d = await r.read(4096)
                if not d:
                    break
        finally:
            try:
                w.close()
                await w.wait_closed()
            except Exception:
                pass

    srv = await asyncio.start_server(_drain, host="127.0.0.1", port=0)
    server.append(srv)
    port = srv.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    return reader, writer
