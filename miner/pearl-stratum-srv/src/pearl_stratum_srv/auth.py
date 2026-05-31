"""Wallet validation, IP rate limits, connection limits.

Public pool concerns the LAN solo pool didn't need:
  - parse `prl1...workerN` from mining.authorize → extract Pearl wallet address
  - reject malformed addresses early
  - cap concurrent connections per IP (slow loris / connection-flood mitigation)
  - cap new-connection rate per IP (login-flood mitigation)
  - integrate with ShareDb's banned_ips table

Address validation is intentionally minimal — we check the bech32 prefix `prl1`
plus reasonable length bounds. Cryptographic checksum validation would be
nice-to-have but isn't a security boundary: a wrong address means the miner
contributes shares against an address they don't control, hurting only them.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass


# Pearl addresses are bech32m taproot, format `<hrp>1p...` ~62 chars. We accept
# 50-100 to leave wiggle room for future encodings without being lax.
# Bech32 HRPs per pearl/node/chaincfg/params.go:
#   prl  — mainnet
#   tprl — testnet, testnet2
#   sprl — simnet (Pearl doesn't define one; future-proofing)
#   rprl — regtest
_PEARL_ADDR_MIN = 50
_PEARL_ADDR_MAX = 100
_PEARL_HRPS = ("prl1", "tprl1", "sprl1", "rprl1")
_BECH32_CHARSET = set("qpzry9x8gf2tvdw0s3jn54khce6mua7l")


def parse_worker_name(raw: str) -> tuple[str | None, str]:
    """Parse `prl1...workerN` → (wallet_addr, worker_label).

    Returns (None, raw) if the address part is invalid. Worker label can be
    anything; we default to "default" if absent. Examples:
        "prl1pgk8j7vj...28wg.rig04.gpu0" → ("prl1pgk8j7vj...28wg", "rig04.gpu0")
        "prl1pgk8j7vj...28wg"            → ("prl1pgk8j7vj...28wg", "default")
        "badaddress.worker"              → (None, "badaddress.worker")
    """
    if not isinstance(raw, str) or not raw:
        return None, raw
    addr, sep, label = raw.partition(".")
    label = label or "default"
    if validate_pearl_address(addr):
        return addr, label
    return None, raw


def validate_pearl_address(addr: str) -> bool:
    """Minimal bech32m sanity check for Pearl addresses (`prl1p...`)."""
    if not isinstance(addr, str):
        return False
    if not (_PEARL_ADDR_MIN <= len(addr) <= _PEARL_ADDR_MAX):
        return False
    # Pick the longest matching HRP so "tprl1..." doesn't get treated as
    # "prl1..." with a leading garbage `t`.
    matched_hrp = next((h for h in sorted(_PEARL_HRPS, key=len, reverse=True)
                       if addr.startswith(h)), None)
    if matched_hrp is None:
        return False
    # Bech32 charset check (everything after the HRP). NOT lowercased: bech32
    # explicitly forbids mixed-case strings, and our charset is the canonical
    # lowercase alphabet. Mixed-case addresses are invalid by spec (BIP-173).
    body = addr[len(matched_hrp):]
    return all(c in _BECH32_CHARSET for c in body)


# ---------------------------------------------------------- rate limits


@dataclass
class IpQuotas:
    """Per-IP concurrency + new-connection-rate limits.

    - `max_concurrent`: how many open TCP connections we allow from one IP
    - `max_new_per_minute`: token-bucket-ish rolling-window cap on connect rate
    """

    max_concurrent: int = 200
    max_new_per_minute: int = 60


class IpLimiter:
    """Track per-IP open connection counts + a 60s sliding connect-rate window.

    Thread-unsafe; designed for single-thread asyncio access from the listener.
    """

    def __init__(self, quotas: IpQuotas | None = None):
        self.quotas = quotas or IpQuotas()
        self._open: defaultdict[str, int] = defaultdict(int)
        self._connect_times: defaultdict[str, deque[float]] = defaultdict(deque)

    def try_accept(self, ip: str, now: float | None = None) -> tuple[bool, str]:
        """Check whether a new connection from `ip` may be accepted.

        Returns (allowed, reason). Caller MUST call note_open() and note_close()
        for accepted connections. Reason is non-empty on rejection.
        """
        now = now if now is not None else time.time()

        # Trim the per-IP connect-time window to the last 60s.
        times = self._connect_times[ip]
        cutoff = now - 60.0
        while times and times[0] < cutoff:
            times.popleft()

        if self._open[ip] >= self.quotas.max_concurrent:
            return False, f"concurrent connections from {ip} >= {self.quotas.max_concurrent}"
        if len(times) >= self.quotas.max_new_per_minute:
            return False, f"new connection rate from {ip} >= {self.quotas.max_new_per_minute}/min"
        return True, ""

    def note_open(self, ip: str, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self._open[ip] += 1
        self._connect_times[ip].append(now)

    def note_close(self, ip: str) -> None:
        if self._open[ip] > 0:
            self._open[ip] -= 1
        if self._open[ip] == 0:
            # Avoid unbounded dict growth for one-shot scanners.
            self._open.pop(ip, None)
            # Keep _connect_times — the trim-on-check above bounds it naturally.

    def open_count(self, ip: str) -> int:
        return self._open[ip]
