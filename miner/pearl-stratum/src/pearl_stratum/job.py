"""Pearl stratum `mining.notify` parsing into a Job dataclass.

Wire format observed on alphapool.tech (pearl/v1) — see
C:/Source/pearl-investigation/STRATUM_CAPTURE.md §3f:

    {"method": "mining.notify", "params": [
        "0000d446-3061",                # 0  job_id
        "46b849ba...c554",              # 1  prevhash (32B hex)
        "00004020...00618",             # 2  incomplete_header_bytes (hex)
        54342,                          # 3  job-seq (int, opaque)
        "6a093061",                     # 4  ntime (Unix epoch hex)
        "1a0ffff0",                     # 5  nbits (4B compact target, hex)
        true                            # 6  clean_jobs
    ]}

NOTE on field assignment: STRATUM_CAPTURE.md's index labels are tentative.
Param 4 `0x6a093061` = 1778987105 = the Unix epoch of the capture (a strong
signal it's ntime), and param 5 `0x1a0ffff0` decodes as a sane Bitcoin
testnet-difficulty-1 target — so we treat **param 5 as nbits**. If the pool
ever rearranges these, the parser will need to be updated; the test fixture
in tests/test_job_parser.py pins the current interpretation.

The pool also pushes an unsolicited `pearl.set_mining_params` once per
connection (m/n/k/rank/rows_pattern/cols_pattern/mma_type). That payload is
**not** part of `mining.notify` — the StratumClient holds it separately and
slots it into the `mining_job` envelope at submitPlainProof time.

`target_le` is derived from nbits. The kernel's `pow_target` GPU input is
32 little-endian bytes (Bitcoin convention is target = mantissa * 2^(8*(exp-3))
serialized big-endian; little-endian flip is the standard miner-side form).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def _bits_to_target_int(nbits: int) -> int:
    """Bitcoin-style compact target. Mirrors pearl_gateway.blockchain_utils.bits_to_target."""
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0xFFFFFF
    if exponent < 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa * (1 << (8 * (exponent - 3)))


def _target_int_to_le_bytes(target: int) -> bytes:
    """32-byte little-endian target. Saturates if target > 2^256-1 (shouldn't happen for valid nbits)."""
    target &= (1 << 256) - 1
    return target.to_bytes(32, "little")


@dataclass
class Job:
    job_id: str
    incomplete_header_bytes: bytes
    """Hex-decoded `incomplete_header_bytes` (param[2] of mining.notify)."""

    nbits: int
    """Compact nbits as int (e.g. 0x1a0ffff0)."""

    target: int
    """Full 256-bit PoW target derived from nbits."""

    target_le: bytes
    """32-byte little-endian form of `target`. Suitable for kernel `pow_target` input."""

    clean_jobs: bool
    """If True, miner must drop in-flight work and switch immediately."""

    received_at: float
    """Wall-clock time we parsed this notify, for staleness telemetry."""

    raw_params: list[Any] = field(default_factory=list)
    """Original positional params from `mining.notify`. Kept for replay / debugging."""


def parse_notify(params: list[Any]) -> Job:
    """Parse a `mining.notify` params array into a Job.

    Validates field types and lengths; raises ValueError on malformed input.
    The pool may send 6 or 7 positional params (some dialects omit clean_jobs);
    we default clean_jobs to True on absence (safer to drop in-flight work).
    """
    if not isinstance(params, list) or len(params) < 6:
        raise ValueError(f"mining.notify params must be a list of >=6 items, got {params!r}")

    job_id = params[0]
    if not isinstance(job_id, str):
        raise ValueError(f"job_id must be a string, got {type(job_id).__name__}")

    # params[1] is prevhash hex; we don't decode it (the incomplete_header_bytes
    # at params[2] is what the gateway/kernel consume). We keep it in raw_params.

    header_hex = params[2]
    if not isinstance(header_hex, str):
        raise ValueError(f"incomplete_header_bytes must be hex string, got {type(header_hex).__name__}")
    try:
        header_bytes = bytes.fromhex(header_hex)
    except ValueError as e:
        raise ValueError(f"incomplete_header_bytes is not valid hex: {e}") from e

    nbits_field = params[5]
    if isinstance(nbits_field, str):
        try:
            nbits = int(nbits_field, 16)
        except ValueError as e:
            raise ValueError(f"nbits is not valid hex: {e}") from e
    elif isinstance(nbits_field, int):
        nbits = nbits_field
    else:
        raise ValueError(f"nbits must be hex string or int, got {type(nbits_field).__name__}")
    if not (0 <= nbits <= 0xFFFFFFFF):
        raise ValueError(f"nbits out of 32-bit range: {nbits:#x}")

    target = _bits_to_target_int(nbits)
    if target == 0:
        raise ValueError(f"nbits {nbits:#x} produces zero target (unsolvable)")
    target_le = _target_int_to_le_bytes(target)

    clean_jobs = bool(params[6]) if len(params) >= 7 else True

    return Job(
        job_id=job_id,
        incomplete_header_bytes=header_bytes,
        nbits=nbits,
        target=target,
        target_le=target_le,
        clean_jobs=clean_jobs,
        received_at=time.time(),
        raw_params=list(params),
    )
