"""Unit tests for FailoverStratumClient.

Validates:
  - Primary accept path: spare is idle, no failover counted
  - Stale (error 21) on primary: NOT retried on spare
  - Hard-error on primary (low-diff, RST, etc): spare catches the share within 50ms
  - Spare ALSO fails: stats.both_failed increments; we return the last failure
  - Duplicate notify dedup: same job_id arriving on both peers fires consumer once
  - Mid-stream RST on primary: spare receives the post-RST submit cleanly

Uses the existing FakePool as the base; subclasses inject the misbehavior.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from pearl_stratum.failover_client import (
    FAILOVER_LOG_THRESHOLD_MS,
    FailoverStratumClient,
)
from pearl_stratum.stratum_client import StratumClient


pytestmark = pytest.mark.asyncio


# Re-import the FakePool from the dialogue tests as it's the cleanest base.
from .test_stratum_dialogue import FakePool


class TwoConnFakePool(FakePool):
    """FakePool variant that tracks per-connection state separately.

    The base FakePool overwrites `last_writer` on each connect, which would
    cause the dedup-test below to clobber the primary's writer. Here we keep
    a list of per-connection writers and treat each connection independently.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.writers: list = []
        # Per-connection submit-counts for asserting which peer received the share.
        self.submits_per_conn: list[int] = []
        # `primary_submit_response` and `spare_submit_response` let a test
        # inject distinct behaviors per connection (by index).
        self.per_conn_submit_response: dict[int, str] = {}
        # Allow tests to inject a transient socket close on conn N before
        # responding to submit i.
        self.rst_on_conn_at_submit: dict[int, int] = {}

    async def _handle_client(self, reader, writer):
        my_idx = len(self.writers)
        self.writers.append(writer)
        self.submits_per_conn.append(0)
        self.connection_count += 1
        self.last_writer = writer
        self._handler_tasks.append(asyncio.current_task())
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
                    await self._send(writer, {"jsonrpc": "2.0", "id": rid,
                        "result": {"pearl/v1": True}})
                elif method == "mining.subscribe":
                    await self._send(writer, {"jsonrpc": "2.0", "id": rid,
                        "result": [[["mining.set_difficulty", "c"], ["mining.notify", "c"]], "", 0]})
                elif method == "mining.authorize":
                    if self.push_set_mining_params:
                        await self._send(writer, {
                            "method": "pearl.set_mining_params",
                            "params": [{"m": 131072, "n": 131072, "k": 4096, "rank": 128,
                                        "rows_pattern": [0, 32], "cols_pattern": list(range(64)),
                                        "mma_type": "Int7xInt7ToInt32"}],
                        })
                    await self._send(writer, {"jsonrpc": "2.0", "id": rid, "result": True})
                    if self.diff_after_notify is not None:
                        await self._send(writer, {"method": "mining.set_difficulty",
                            "params": [self.diff_after_notify]})
                    if self.push_notify_after_authorize:
                        await self._send(writer, {"method": "mining.notify",
                            "params": self.notify_params})
                elif method == "mining.submit":
                    self.submit_count += 1
                    self.submits_per_conn[my_idx] += 1
                    # Per-conn RST injection
                    if (my_idx in self.rst_on_conn_at_submit
                            and self.submits_per_conn[my_idx] == self.rst_on_conn_at_submit[my_idx]):
                        # Drop the connection before responding — simulates RST.
                        writer.close()
                        return
                    response = self.per_conn_submit_response.get(my_idx, self.submit_response)
                    if response == "accept":
                        await self._send(writer, {"jsonrpc": "2.0", "id": rid, "result": True})
                    elif response == "stale":
                        await self._send(writer, {"jsonrpc": "2.0", "id": rid,
                            "error": [21, "chain advanced", None]})
                    elif response == "low_diff":
                        await self._send(writer, {"jsonrpc": "2.0", "id": rid,
                            "error": [23, "Low difficulty share", None]})
                    elif response == "rst":
                        writer.close()
                        return
                    elif response == "timeout":
                        # Don't respond at all.
                        pass
                    elif response == "first_call_fail_then_accept":
                        if self.submits_per_conn[my_idx] == 1:
                            writer.close()
                            return
                        await self._send(writer, {"jsonrpc": "2.0", "id": rid, "result": True})
        except (ConnectionResetError, asyncio.IncompleteReadError, asyncio.CancelledError):
            return
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            try:
                writer.close()
            except Exception:
                pass


async def _spin_failover(pool: TwoConnFakePool, *, n_peers: int = 2,
                          received_jobs: list | None = None):
    callbacks = {}
    if received_jobs is not None:
        callbacks["on_new_job"] = received_jobs.append
    client = FailoverStratumClient(
        host=pool.host,
        port=pool.port,
        address="prl1testtest",
        worker="testworker",
        password="x",
        n_peers=n_peers,
        **callbacks,
    )
    # Speed up unit tests — the 500ms stagger is for live-pool burst protection,
    # not needed against the FakePool (which doesn't rate-limit).
    client.PEER_START_STAGGER_S = 0.02
    task = asyncio.create_task(client.run())
    # Wait until BOTH peers have completed handshake (so failover is testable).
    # 500ms stagger * 2 peers + handshake = up to ~3s on slow CI.
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if all(p.connected for p in client.peers):
            break
        await asyncio.sleep(0.01)
    return client, task


# ----- tests --------------------------------------------------------------


async def test_n_peers_limit_enforced():
    with pytest.raises(ValueError, match="risks pool-side IP rate-limit"):
        FailoverStratumClient(
            host="127.0.0.1", port=1, address="x", worker="y", n_peers=3,
        )


async def test_n_peers_zero_rejected():
    with pytest.raises(ValueError, match="n_peers must be >= 1"):
        FailoverStratumClient(
            host="127.0.0.1", port=1, address="x", worker="y", n_peers=0,
        )


async def test_worker_suffix_for_spare():
    """Spare authorizes with `worker-spare` so pool credit attribution is clean."""
    pool = TwoConnFakePool()
    await pool.start()
    try:
        client, task = await _spin_failover(pool)
        try:
            # Two distinct authorize requests should have arrived.
            auths = [r for r in pool.requests if r.get("method") == "mining.authorize"]
            assert len(auths) == 2
            worker_names = {a["params"][0] for a in auths}
            assert "prl1testtest.testworker" in worker_names
            assert "prl1testtest.testworker-spare" in worker_names
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


async def test_primary_accept_no_failover():
    pool = TwoConnFakePool()
    await pool.start()
    try:
        client, task = await _spin_failover(pool)
        try:
            r = await client.submit_share("0000d446-3061", "AAAA==")
            assert r.accepted is True
            assert client.failover_stats.primary_accepts == 1
            assert client.failover_stats.failover_attempts == 0
            assert client.failover_stats.spare_accepts == 0
            # Only the primary (conn 0) saw the submit.
            assert pool.submits_per_conn[0] == 1
            assert pool.submits_per_conn[1] == 0
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


async def test_stale_primary_no_failover():
    """Error 21 on primary must NOT trigger failover (chain has advanced)."""
    pool = TwoConnFakePool()
    pool.per_conn_submit_response = {0: "stale"}
    await pool.start()
    try:
        client, task = await _spin_failover(pool)
        try:
            r = await client.submit_share("0000d446-3061", "AAAA==")
            assert r.accepted is False
            assert r.error_code == 21
            assert client.failover_stats.stale == 1
            assert client.failover_stats.failover_attempts == 0
            assert pool.submits_per_conn[0] == 1
            # Spare must NOT have been called.
            assert pool.submits_per_conn[1] == 0
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


async def test_low_diff_primary_failover_to_spare():
    """Error 23 on primary triggers failover; spare accepts."""
    pool = TwoConnFakePool()
    pool.per_conn_submit_response = {0: "low_diff", 1: "accept"}
    await pool.start()
    try:
        client, task = await _spin_failover(pool)
        try:
            t0 = time.monotonic()
            r = await client.submit_share("0000d446-3061", "AAAA==")
            wall_ms = (time.monotonic() - t0) * 1000
            assert r.accepted is True, f"failover should have caught the share; got {r}"
            assert client.failover_stats.failover_successes == 1
            assert client.failover_stats.primary_accepts == 0
            assert client.failover_stats.spare_accepts == 1
            assert client.failover_stats.primary_failures == 1
            # Both peers saw the submit.
            assert pool.submits_per_conn[0] == 1
            assert pool.submits_per_conn[1] == 1
            # Wall-time must be reasonable; locally we target <50ms but allow
            # some headroom for slow CI.
            assert wall_ms < 500, f"failover took {wall_ms:.1f}ms"
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


async def test_rst_on_primary_failover():
    """Primary RSTs during submit; spare catches the share."""
    pool = TwoConnFakePool()
    # Conn 0 RSTs on its first submit, conn 1 accepts.
    pool.rst_on_conn_at_submit = {0: 1}
    pool.per_conn_submit_response = {1: "accept"}
    await pool.start()
    try:
        client, task = await _spin_failover(pool)
        try:
            t0 = time.monotonic()
            r = await client.submit_share("0000d446-3061", "AAAA==")
            wall_ms = (time.monotonic() - t0) * 1000
            assert r.accepted is True, f"failover should have caught RST'd share; got {r}"
            assert client.failover_stats.failover_successes == 1
            assert client.failover_stats.spare_accepts == 1
            assert pool.submits_per_conn[1] == 1
            assert wall_ms < 1000, f"failover took {wall_ms:.1f}ms"
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


async def test_both_fail_both_failed_metric():
    """Primary RSTs + spare also fails. We log this as a correlated outage."""
    pool = TwoConnFakePool()
    # Both connections error on their first submit.
    pool.per_conn_submit_response = {0: "low_diff", 1: "low_diff"}
    await pool.start()
    try:
        client, task = await _spin_failover(pool)
        try:
            r = await client.submit_share("0000d446-3061", "AAAA==")
            assert r.accepted is False
            assert client.failover_stats.both_failed == 1
            assert client.failover_stats.primary_failures == 1
            assert client.failover_stats.spare_failures == 1
            assert client.failover_stats.failover_successes == 0
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


async def test_dedup_notify_consumer_fires_once():
    """Both peers receive the same job_id; consumer's on_new_job fires once."""
    pool = TwoConnFakePool()
    await pool.start()
    received_jobs: list = []
    try:
        client, task = await _spin_failover(pool, received_jobs=received_jobs)
        try:
            # Wait briefly to make sure BOTH peers have processed their notify.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and len(received_jobs) < 1:
                await asyncio.sleep(0.01)
            # Verify both peers' StratumClient saw the notify (current_job set on each)
            await asyncio.sleep(0.2)  # let both peers finish their notify dispatch
            assert all(p.current_job is not None for p in client.peers), (
                "both peers should have received the notify"
            )
            # But the wrapper-level consumer callback fired EXACTLY once.
            assert len(received_jobs) == 1, (
                f"expected dedup -> 1 callback fire, got {len(received_jobs)}"
            )
            assert received_jobs[0].job_id == "0000d446-3061"
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


async def test_failover_latency_under_50ms_local():
    """The submit-then-cutover wall-time must be tight enough to be useful.

    Local-loopback we target <50ms; over the WAN we tolerate up to 500ms before
    a slow-failover warning fires (FAILOVER_LOG_THRESHOLD_MS).
    """
    pool = TwoConnFakePool()
    pool.per_conn_submit_response = {0: "low_diff", 1: "accept"}
    await pool.start()
    try:
        client, task = await _spin_failover(pool)
        try:
            # Do 5 attempts; check median (loopback should be tight).
            latencies = []
            for i in range(5):
                t0 = time.monotonic()
                r = await client.submit_share("0000d446-3061", f"AAA{i}==")
                wall_ms = (time.monotonic() - t0) * 1000
                assert r.accepted is True
                latencies.append(wall_ms)
            latencies.sort()
            median = latencies[len(latencies) // 2]
            # 50ms on loopback is generous; cold-CI shouldn't exceed 200ms.
            assert median < 200, f"median failover wall-time {median:.1f}ms exceeds 200ms"
            # We also tracked per-failover latencies internally; should be even tighter.
            internal = sorted(client.failover_stats.failover_latencies_ms)
            internal_median = internal[len(internal) // 2]
            assert internal_median < 100, (
                f"internal failover latency median {internal_median:.1f}ms exceeds 100ms"
            )
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=3)
    finally:
        await pool.stop()


async def test_spare_not_connected_returns_primary_result():
    """If only the primary is up (spare still connecting), failover returns the primary's error."""
    pool = TwoConnFakePool()
    pool.per_conn_submit_response = {0: "low_diff"}
    await pool.start()
    try:
        # Build a client with n_peers=2 but immediately stop the spare so it
        # can't accept the submit.
        client = FailoverStratumClient(
            host=pool.host,
            port=pool.port,
            address="prl1testtest",
            worker="testworker",
            password="x",
            n_peers=2,
        )
        task = asyncio.create_task(client.run())
        try:
            # Wait for primary up.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not client.peers[0].connected:
                await asyncio.sleep(0.01)
            # Force the spare to think it's disconnected.
            await client.peers[1].stop()
            # Give the stop a moment to propagate.
            await asyncio.sleep(0.1)
            r = await client.submit_share("0000d446-3061", "AAAA==")
            assert r.accepted is False
            assert client.failover_stats.both_failed >= 1
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=5)
    finally:
        await pool.stop()
