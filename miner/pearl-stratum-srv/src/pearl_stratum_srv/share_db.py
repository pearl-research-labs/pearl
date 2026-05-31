"""SQLite persistence for share accounting + payouts.

Schema (auto-created on first write):

  CREATE TABLE shares (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,           -- unix seconds
    worker_addr   TEXT    NOT NULL,           -- prl1... wallet (extracted from mining.authorize)
    worker_label  TEXT    NOT NULL,           -- the user-chosen subname (e.g. "rig04.gpu0")
    job_id        TEXT    NOT NULL,
    difficulty    INTEGER NOT NULL,           -- pool diff this share was credited at
    outcome       TEXT    NOT NULL,           -- 'accepted' | 'stale' | 'malformed'
    ip            TEXT
  );
  CREATE INDEX idx_shares_addr_ts ON shares(worker_addr, ts);
  CREATE INDEX idx_shares_ts      ON shares(ts);

  CREATE TABLE blocks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    height        INTEGER NOT NULL,
    block_hash    TEXT,
    finder_addr   TEXT    NOT NULL,           -- the worker whose share solved the block
    reward_total  INTEGER NOT NULL,           -- satoshis (subsidy + fees)
    outcome       TEXT    NOT NULL            -- 'pending' | 'confirmed' | 'orphaned'
  );

  CREATE TABLE payouts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    REAL    NOT NULL,
    block_id      INTEGER NOT NULL REFERENCES blocks(id),
    recipient     TEXT    NOT NULL,           -- prl1... address
    amount_sats   INTEGER NOT NULL,
    share_count   INTEGER NOT NULL,           -- shares this recipient contributed in the PPLNS window
    status        TEXT    NOT NULL,           -- 'pending' | 'sent' | 'failed' | 'cancelled'
    sent_txid     TEXT,
    notes         TEXT
  );
  CREATE INDEX idx_payouts_status ON payouts(status);

  CREATE TABLE banned_ips (
    ip            TEXT PRIMARY KEY,
    banned_at     REAL    NOT NULL,
    until_ts      REAL    NOT NULL,           -- null/0 = permanent
    reason        TEXT    NOT NULL
  );

Single-writer asyncio access. All writes go through a single asyncio.Lock to
serialize SQLite transactions; SQLite's own thread safety + WAL mode handles
the OS-level concurrency.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import time
from pathlib import Path
from typing import Iterable

_LOGGER = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS shares (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            REAL    NOT NULL,
  worker_addr   TEXT    NOT NULL,
  worker_label  TEXT    NOT NULL,
  job_id        TEXT    NOT NULL,
  difficulty    INTEGER NOT NULL,
  outcome       TEXT    NOT NULL,
  ip            TEXT
);
CREATE INDEX IF NOT EXISTS idx_shares_addr_ts ON shares(worker_addr, ts);
CREATE INDEX IF NOT EXISTS idx_shares_ts      ON shares(ts);

CREATE TABLE IF NOT EXISTS blocks (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            REAL    NOT NULL,
  height        INTEGER NOT NULL,
  block_hash    TEXT,
  finder_addr   TEXT    NOT NULL,
  reward_total  INTEGER NOT NULL,
  outcome       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS payouts (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at    REAL    NOT NULL,
  block_id      INTEGER NOT NULL REFERENCES blocks(id),
  recipient     TEXT    NOT NULL,
  amount_sats   INTEGER NOT NULL,
  share_count   INTEGER NOT NULL,
  status        TEXT    NOT NULL,
  sent_txid     TEXT,
  notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_payouts_status ON payouts(status);

CREATE TABLE IF NOT EXISTS banned_ips (
  ip            TEXT PRIMARY KEY,
  banned_at     REAL    NOT NULL,
  until_ts      REAL    NOT NULL,
  reason        TEXT    NOT NULL
);
"""


class ShareDb:
    """Async-friendly SQLite wrapper. Use as `async with`.

    Asyncio-friendly: all calls go through asyncio.to_thread so SQLite's
    synchronous API doesn't block the event loop. A single shared connection +
    asyncio.Lock serializes writes.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        def _setup():
            conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            return conn

        self._conn = await asyncio.to_thread(_setup)
        _LOGGER.info("share db open at %s (WAL)", self.path)

    async def close(self) -> None:
        if self._conn is not None:
            conn = self._conn
            self._conn = None
            await asyncio.to_thread(conn.close)

    async def __aenter__(self) -> "ShareDb":
        await self.open()
        return self

    async def __aexit__(self, *a) -> None:
        await self.close()

    # ---------------------------------------------------------- writes

    async def insert_share(
        self,
        worker_addr: str,
        worker_label: str,
        job_id: str,
        difficulty: int,
        outcome: str,
        ip: str | None = None,
        ts: float | None = None,
    ) -> int:
        ts = ts if ts is not None else time.time()
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                "INSERT INTO shares(ts, worker_addr, worker_label, job_id, difficulty, outcome, ip) "
                "VALUES (?,?,?,?,?,?,?)",
                (ts, worker_addr, worker_label, job_id, difficulty, outcome, ip),
            )
            return cur.lastrowid

    async def insert_block(
        self,
        height: int,
        finder_addr: str,
        reward_total: int,
        block_hash: str | None = None,
        outcome: str = "pending",
        ts: float | None = None,
    ) -> int:
        ts = ts if ts is not None else time.time()
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                "INSERT INTO blocks(ts, height, block_hash, finder_addr, reward_total, outcome) "
                "VALUES (?,?,?,?,?,?)",
                (ts, height, block_hash, finder_addr, reward_total, outcome),
            )
            return cur.lastrowid

    async def insert_payouts(self, block_id: int, entries: Iterable[tuple[str, int, int]]) -> None:
        """entries = iterable of (recipient_addr, amount_sats, share_count)."""
        now = time.time()
        rows = [(now, block_id, r, a, s, "pending") for (r, a, s) in entries]
        async with self._lock:
            await asyncio.to_thread(
                self._conn.executemany,
                "INSERT INTO payouts(created_at, block_id, recipient, amount_sats, share_count, status) "
                "VALUES (?,?,?,?,?,?)",
                rows,
            )

    async def mark_payout_sent(self, payout_id: int, txid: str) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._conn.execute,
                "UPDATE payouts SET status='sent', sent_txid=? WHERE id=?",
                (txid, payout_id),
            )

    async def ban_ip(self, ip: str, reason: str, duration_secs: float | None = None) -> None:
        now = time.time()
        until = (now + duration_secs) if duration_secs else 0.0
        async with self._lock:
            await asyncio.to_thread(
                self._conn.execute,
                "INSERT OR REPLACE INTO banned_ips(ip, banned_at, until_ts, reason) "
                "VALUES (?,?,?,?)",
                (ip, now, until, reason),
            )

    async def unban_ip(self, ip: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.execute, "DELETE FROM banned_ips WHERE ip=?", (ip,))

    # ----------------------------------------------------------- reads

    async def is_banned(self, ip: str) -> bool:
        now = time.time()

        def _q():
            cur = self._conn.execute(
                "SELECT until_ts FROM banned_ips WHERE ip=?", (ip,)
            )
            row = cur.fetchone()
            return row is not None and (row[0] == 0 or row[0] > now)

        return await asyncio.to_thread(_q)

    async def shares_in_window(self, since_ts: float, until_ts: float | None = None) -> list[tuple[str, int]]:
        """Returns [(worker_addr, total_difficulty)] over the window. Used by PPLNS."""
        until_ts = until_ts if until_ts is not None else time.time()

        def _q():
            cur = self._conn.execute(
                "SELECT worker_addr, SUM(difficulty) FROM shares "
                "WHERE outcome='accepted' AND ts >= ? AND ts < ? "
                "GROUP BY worker_addr",
                (since_ts, until_ts),
            )
            return cur.fetchall()

        return await asyncio.to_thread(_q)

    async def shares_for_worker(
        self, worker_addr: str, since_ts: float
    ) -> list[tuple[float, str, int, str]]:
        """Returns recent shares for one worker: (ts, outcome, difficulty, worker_label)."""

        def _q():
            cur = self._conn.execute(
                "SELECT ts, outcome, difficulty, worker_label FROM shares "
                "WHERE worker_addr=? AND ts >= ? ORDER BY ts DESC LIMIT 500",
                (worker_addr, since_ts),
            )
            return cur.fetchall()

        return await asyncio.to_thread(_q)

    async def pending_payouts(self) -> list[tuple[int, str, int, int, float]]:
        """Returns [(payout_id, recipient, amount_sats, share_count, created_at)]."""

        def _q():
            cur = self._conn.execute(
                "SELECT id, recipient, amount_sats, share_count, created_at "
                "FROM payouts WHERE status='pending' ORDER BY created_at"
            )
            return cur.fetchall()

        return await asyncio.to_thread(_q)

    async def block_count(self) -> int:
        def _q():
            cur = self._conn.execute("SELECT COUNT(*) FROM blocks WHERE outcome != 'orphaned'")
            return cur.fetchone()[0]

        return await asyncio.to_thread(_q)

    async def malformed_rate_for_ip(self, ip: str, window_secs: float) -> int:
        """Count malformed shares from this IP in the last window. For abuse detection."""
        cutoff = time.time() - window_secs

        def _q():
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM shares WHERE ip=? AND outcome='malformed' AND ts >= ?",
                (ip, cutoff),
            )
            return cur.fetchone()[0]

        return await asyncio.to_thread(_q)
