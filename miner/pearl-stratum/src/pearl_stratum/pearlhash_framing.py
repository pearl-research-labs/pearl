"""Pearlhash wire-format framing: ASCII-hex + '\\n'.

Each Pearlhash TCP frame is `hex(inner_bytes) + '\\n'` where `inner_bytes` is
the 4-byte counter prefix concatenated with the XOR-encrypted body (see
`pearlhash_cipher`). This module is the byte-level transport layer only —
no cipher, no protocol logic.

Reference: `36_pearlhash_shim.md` §1.2 (definitive wire format, verified across
17 frames of cap_long.pcap).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


def encode_frame(inner: bytes) -> bytes:
    """Wrap binary inner bytes for the wire: lowercase hex + LF."""
    return inner.hex().encode("ascii") + b"\n"


def decode_frame(wire_line: bytes) -> bytes:
    """Decode one wire line back to inner bytes. Accepts trailing LF or none.

    Raises `ValueError` on non-hex or odd-length input.
    """
    line = wire_line.rstrip(b"\r\n")
    if not line:
        raise ValueError("empty frame line")
    try:
        text = line.decode("ascii")
    except UnicodeDecodeError as e:
        raise ValueError(f"non-ASCII byte in wire line: {e}") from None
    try:
        return bytes.fromhex(text)
    except ValueError as e:
        raise ValueError(f"non-hex content in wire line: {e}") from None


async def iter_frames(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    """Async iterate inner-bytes from a Pearlhash StreamReader. EOF ends the iter."""
    while True:
        line = await reader.readline()
        if not line:
            return
        yield decode_frame(line)
