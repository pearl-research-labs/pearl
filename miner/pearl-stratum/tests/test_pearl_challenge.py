"""Tests for the `pearl.challenge` DDoS-pacer handler.

Covers:
  - `_solve_pearl_challenge` against the 9 captured fixtures from the RE memo.
    These are NOT brute-force searches — we just verify that the captured nonce
    satisfies the predicate (any qualifying nonce is accepted by the pool).
  - Forward search from 0 finds A qualifying nonce within a small space, with
    a low-difficulty fixture we craft.
  - Full handshake against a FakePool that pushes `pearl.challenge` BEFORE the
    client may send mining.configure: client solves it, pool acks, then normal
    handshake completes.
  - Mid-session `pearl.challenge` arriving between mining.notify and
    mining.submit: client solves it concurrently, submit still works.
  - Malformed challenge (missing seed / bad difficulty) raises
    StratumProtocolError.
"""

from __future__ import annotations

import asyncio
import json
import time

import blake3
import pytest

from pearl_stratum.stratum_client import (
    PEARL_CHALLENGE_METHOD,
    PEARL_CHALLENGE_RESPONSE_METHOD,
    StratumClient,
    StratumProtocolError,
    _solve_pearl_challenge,
)


# conftest auto-marks coroutine functions as asyncio; no module-level mark
# needed (and it would warn for the pure-Python parametrized tests below).


# ---- captured fixtures from 58_pearl_challenge_protocol.md ----------------

CAPTURED_FIXTURES = [
    ("6f0a8bbe8def92c6ae50de218c58c421ec6e4ec779583906793caadc74defa38",
     "000000013169186e"),
    ("a8f498ae55b8a74945f016afe809b7a94c9e6e8577c1f2e062b29a195f321e62",
     "000000005d311fe4"),
    ("78e99c43e022881c737ed3c119f348cca246c1dca706c5bcc33e6d4c5f47e060",
     "00000000e5c738ae"),
    ("6abbbba11fc3efd57d15dc066f5b856679e35b70af758a05a315e9acc0485f87",
     "000000001763549b"),
    ("b167e85a2baa4685811fdb5d843a94d24e7743e372c80ceff1ea10469febd0b4",
     "0000000227a1e061"),
    ("3d5f19a4e3563c55d8e6ccb3be3cd712aa3dfda56ece0993e624b206d89b8c90",
     "000000004095f380"),
    ("4dd49a80cb00de8b9135e6127ff19c7cd74f8d65d915d5dd64a18d6945620ab3",
     "000000004b622774"),
    ("fe8b56d4519453af28fe6406b2ea75580fa70999388ff83e69699c6474729fdc",
     "000000006f073b73"),
    ("34a52c2c61b5f2c030c19ce01a116400faa156b222b5e84ff26c6ea904e4a8b3",
     "0000000142d113a2"),
]


def _meets_difficulty(seed_hex: str, nonce_hex: str, difficulty: int) -> bool:
    seed = bytes.fromhex(seed_hex)
    nonce = int(nonce_hex, 16)
    h = blake3.blake3(seed + nonce.to_bytes(8, "little")).digest()
    full, rem = divmod(difficulty, 8)
    if h[:full] != b"\x00" * full:
        return False
    if rem == 0:
        return True
    return (h[full] >> (8 - rem)) == 0


# ---- pure PoW tests --------------------------------------------------------


@pytest.mark.parametrize("seed_hex,nonce_hex", CAPTURED_FIXTURES)
def test_captured_nonce_satisfies_predicate(seed_hex: str, nonce_hex: str) -> None:
    """Each captured (seed, nonce) tuple satisfies the difficulty-32 predicate."""
    assert _meets_difficulty(seed_hex, nonce_hex, difficulty=32)


def test_solver_finds_low_difficulty_nonce_quickly() -> None:
    """At difficulty=12 (~4096 hashes expected) the forward search must complete
    in under 1 sec and return a valid nonce.
    """
    seed_hex = "00" * 32
    t0 = time.monotonic()
    found = _solve_pearl_challenge(seed_hex, difficulty=12)
    dt = time.monotonic() - t0
    assert dt < 1.0, f"solver took {dt:.2f}s for diff=12 — too slow"
    # Verify it satisfies the predicate.
    assert _meets_difficulty(seed_hex, found, difficulty=12)
    # Sanity-check it's a 16-hex-char string (preserves leading zeros for u64).
    assert len(found) == 16
    int(found, 16)  # raises if not hex


def test_solver_difficulty_16_works() -> None:
    """Difficulty=16 (~65k hashes) still fits well under 5s on any host."""
    seed_hex = "ab" * 32
    t0 = time.monotonic()
    found = _solve_pearl_challenge(seed_hex, difficulty=16)
    dt = time.monotonic() - t0
    assert dt < 5.0
    assert _meets_difficulty(seed_hex, found, difficulty=16)


def test_solver_rejects_bad_seed_length() -> None:
    with pytest.raises(ValueError, match="32-byte seed"):
        _solve_pearl_challenge("00" * 16, difficulty=8)


# ---- FakePool with challenge support ---------------------------------------


class ChallengeFakePool:
    """FakePool variant that pushes a `pearl.challenge` BEFORE accepting any
    client traffic, mirroring alphapool v1.5 behavior.

    Also supports injecting a mid-session challenge after authorize.
    """

    def __init__(
        self,
        *,
        push_initial_challenge: bool = True,
        initial_challenge_difficulty: int = 12,
        push_midsession_challenge: bool = False,
        midsession_challenge_difficulty: int = 12,
    ) -> None:
        self.push_initial_challenge = push_initial_challenge
        self.initial_challenge_difficulty = initial_challenge_difficulty
        self.push_midsession_challenge = push_midsession_challenge
        self.midsession_challenge_difficulty = midsession_challenge_difficulty

        self.requests: list[dict] = []
        self.challenge_responses_received: list[dict] = []
        self.submit_count = 0
        self.connection_count = 0
        self.host = "127.0.0.1"
        self.port = 0

        # Seeds we used per connection (server picks them).
        self.last_seed_hex: str | None = None
        self.last_midsession_seed_hex: str | None = None

        self._server: asyncio.base_events.Server | None = None
        self.last_writer: asyncio.StreamWriter | None = None
        self._handler_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, host=self.host, port=0
        )
        sockets = self._server.sockets or ()
        if not sockets:
            raise RuntimeError("fake pool failed to bind")
        self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        for task in self._handler_tasks:
            if not task.done():
                task.cancel()
        for task in self._handler_tasks:
            try:
                await asyncio.wait_for(task, timeout=2)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        self._handler_tasks.clear()
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2)
            except asyncio.TimeoutError:
                pass
            self._server = None

    def _make_seed_hex(self, salt: int = 0) -> str:
        # Deterministic seed (not random); we just need a 32-byte hex value
        # that allows the client to find a low-difficulty nonce quickly.
        return f"{salt:064x}"

    async def _send(self, writer: asyncio.StreamWriter, obj: dict) -> None:
        writer.write((json.dumps(obj) + "\n").encode("utf-8"))
        await writer.drain()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connection_count += 1
        self.last_writer = writer
        self._handler_tasks.append(asyncio.current_task())  # type: ignore[arg-type]

        # PUSH INITIAL CHALLENGE before accepting any client traffic.
        if self.push_initial_challenge:
            self.last_seed_hex = self._make_seed_hex(salt=self.connection_count)
            await self._send(writer, {
                "id": None,
                "method": PEARL_CHALLENGE_METHOD,
                "params": {
                    "seed": self.last_seed_hex,
                    "difficulty": self.initial_challenge_difficulty,
                },
            })

        midsession_sent = False

        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                msg = json.loads(line)
                self.requests.append(msg)
                method = msg.get("method")
                rid = msg.get("id")
                if method == PEARL_CHALLENGE_RESPONSE_METHOD:
                    self.challenge_responses_received.append(msg)
                    # Verify the nonce actually satisfies the predicate.
                    params = msg.get("params", {})
                    seed_hex = params.get("seed", "")
                    nonce_hex = params.get("nonce", "")
                    diff = self.initial_challenge_difficulty
                    # If this looks like a mid-session response, use the
                    # mid-session difficulty.
                    if self.last_midsession_seed_hex and seed_hex == self.last_midsession_seed_hex:
                        diff = self.midsession_challenge_difficulty
                    ok = _meets_difficulty(seed_hex, nonce_hex, diff)
                    await self._send(writer, {
                        "jsonrpc": "2.0", "id": rid,
                        "result": True if ok else False,
                        "error": None if ok else [42, "bad nonce", None],
                    })
                elif method == "mining.configure":
                    await self._send(writer, {
                        "jsonrpc": "2.0", "id": rid,
                        "result": {"pearl/v1": True, "pearl/v1.share_format": "base64"},
                    })
                elif method == "mining.subscribe":
                    await self._send(writer, {
                        "jsonrpc": "2.0", "id": rid,
                        "result": [
                            [["mining.set_difficulty", "conn-test"],
                             ["mining.notify", "conn-test"]],
                            "", 0,
                        ],
                    })
                elif method == "mining.authorize":
                    await self._send(writer, {
                        "method": "pearl.set_mining_params",
                        "params": [{
                            "m": 131072, "n": 131072, "k": 4096, "rank": 128,
                            "rows_pattern": [0, 32],
                            "cols_pattern": list(range(64)),
                            "mma_type": "Int7xInt7ToInt32",
                        }],
                    })
                    await self._send(writer, {
                        "jsonrpc": "2.0", "id": rid, "result": True,
                    })
                    await self._send(writer, {
                        "method": "mining.set_difficulty",
                        "params": [1048576.0],
                    })
                    await self._send(writer, {
                        "method": "mining.notify",
                        "params": [
                            "0000d446-3061",
                            "46b849bae7551681283f02a20080cd3f0fd0dfad5e320b09b6af901291bfc554",
                            "0000402054c5bf911290afb6090b325eaddfd00f3fcd8000a2023f28811655e7ba49b846d262d62ab2f3dbbf2ddd73a2c00a9ccd9838264c4298998096ef5602b0bfec3b6130096a99a00618",
                            54342,
                            "6a093061",
                            "1a0ffff0",
                            True,
                        ],
                    })
                    if self.push_midsession_challenge and not midsession_sent:
                        midsession_sent = True
                        self.last_midsession_seed_hex = self._make_seed_hex(
                            salt=0xDEAD_BEEF + self.connection_count
                        )
                        await self._send(writer, {
                            "id": None,
                            "method": PEARL_CHALLENGE_METHOD,
                            "params": {
                                "seed": self.last_midsession_seed_hex,
                                "difficulty": self.midsession_challenge_difficulty,
                            },
                        })
                elif method == "mining.submit":
                    self.submit_count += 1
                    await self._send(writer, {"jsonrpc": "2.0", "id": rid, "result": True})
                else:
                    pass
        except (ConnectionResetError, asyncio.IncompleteReadError, asyncio.CancelledError):
            return
        finally:
            try:
                writer.close()
            except Exception:
                pass


# ---- integration tests -----------------------------------------------------


async def test_initial_pearl_challenge_solved_before_handshake() -> None:
    pool = ChallengeFakePool(
        push_initial_challenge=True,
        initial_challenge_difficulty=10,  # ~1024 hashes -> trivially fast
    )
    await pool.start()
    try:
        client = StratumClient(
            host=pool.host, port=pool.port,
            address="prl1testtesttest", worker="testworker", password="x",
        )
        task = asyncio.create_task(client.run())
        try:
            # Wait for handshake completion (mining_params dict populated).
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if client.connected and client.current_job is not None:
                    break
                await asyncio.sleep(0.02)
            assert client.connected, "client never finished handshake"

            # The pool received exactly one challenge_response, BEFORE any
            # mining.configure / subscribe / authorize.
            req_methods = [r.get("method") for r in pool.requests]
            assert req_methods[0] == PEARL_CHALLENGE_RESPONSE_METHOD, (
                f"first client request must be challenge_response, got {req_methods}"
            )
            assert req_methods[1:4] == [
                "mining.configure", "mining.subscribe", "mining.authorize"
            ]

            # The nonce in the response actually satisfies the predicate.
            resp = pool.challenge_responses_received[0]
            params = resp["params"]
            assert params["seed"] == pool.last_seed_hex
            assert _meets_difficulty(params["seed"], params["nonce"], difficulty=10)

            # Stats reflect the solved challenge.
            assert client.stats.challenges_solved == 1
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_no_pearl_challenge_legacy_pool_still_works() -> None:
    """A v1.4-style pool (doesn't push challenge) must still complete handshake,
    via the 2s initial-challenge timeout fallback.
    """
    pool = ChallengeFakePool(push_initial_challenge=False)
    await pool.start()
    try:
        client = StratumClient(
            host=pool.host, port=pool.port,
            address="prl1testtesttest", worker="testworker", password="x",
        )
        task = asyncio.create_task(client.run())
        try:
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                if client.connected and client.current_job is not None:
                    break
                await asyncio.sleep(0.02)
            assert client.connected
            assert client.stats.challenges_solved == 0
            # First client request was mining.configure, no challenge_response.
            req_methods = [r.get("method") for r in pool.requests]
            assert req_methods[0] == "mining.configure"
            assert PEARL_CHALLENGE_RESPONSE_METHOD not in req_methods
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_midsession_pearl_challenge_handled_during_mining() -> None:
    """Server pushes a `pearl.challenge` AFTER notify; client must solve and
    respond on the live socket without dropping submission throughput.
    """
    pool = ChallengeFakePool(
        push_initial_challenge=True,
        initial_challenge_difficulty=10,
        push_midsession_challenge=True,
        midsession_challenge_difficulty=10,
    )
    await pool.start()
    try:
        client = StratumClient(
            host=pool.host, port=pool.port,
            address="prl1testtesttest", worker="testworker", password="x",
        )
        task = asyncio.create_task(client.run())
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if client.connected and client.current_job is not None:
                    break
                await asyncio.sleep(0.02)
            assert client.connected

            # Wait briefly for the mid-session challenge to be solved.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if client.stats.challenges_solved >= 2:
                    break
                await asyncio.sleep(0.02)
            assert client.stats.challenges_solved >= 2, (
                f"expected 2 solved challenges (initial + mid), "
                f"got {client.stats.challenges_solved}"
            )

            # Submission still works after the mid-session challenge.
            result = await client.submit_share("0000d446-3061", "AAAA==")
            assert result.accepted is True
            assert client.stats.accepted == 1
            assert pool.connection_count == 1, (
                "mid-session challenge should NOT cause reconnect; "
                f"saw {pool.connection_count} connections"
            )

            # The mid-session challenge_response in the pool's log matches
            # the seed the server sent.
            mid = [
                r for r in pool.challenge_responses_received
                if r["params"]["seed"] == pool.last_midsession_seed_hex
            ]
            assert len(mid) == 1
            mid_nonce = mid[0]["params"]["nonce"]
            assert _meets_difficulty(
                pool.last_midsession_seed_hex, mid_nonce, difficulty=10
            )
        finally:
            await client.stop()
            await asyncio.wait_for(task, timeout=2)
    finally:
        await pool.stop()


async def test_malformed_initial_challenge_fails_handshake() -> None:
    """A challenge with missing/wrong-typed params is fatal."""
    pool = ChallengeFakePool(push_initial_challenge=False)
    await pool.start()
    try:
        client = StratumClient(
            host=pool.host, port=pool.port,
            address="prl1testtesttest", worker="testworker", password="x",
        )
        reader, writer = await asyncio.open_connection(pool.host, pool.port)
        client._reader = reader
        client._writer = writer
        # Push a malformed challenge directly.
        assert pool.last_writer is not None
        # Wait until pool sees the connection.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and pool.last_writer is None:
            await asyncio.sleep(0.01)

        # The client is wired to a real socket; we'll feed a malformed frame
        # via the pool's writer.
        # Note: pool.last_writer is the server-side stream for the just-opened
        # connection.
        pool.last_writer.write(
            (json.dumps({
                "id": None, "method": "pearl.challenge",
                "params": {"seed": "not_enough_hex", "difficulty": "wrong_type"},
            }) + "\n").encode()
        )
        await pool.last_writer.drain()

        try:
            with pytest.raises(StratumProtocolError):
                await client._handle_initial_challenge_if_any()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
    finally:
        await pool.stop()
