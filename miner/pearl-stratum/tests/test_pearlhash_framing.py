"""Tests for the Pearlhash hex+newline wire framing."""

from __future__ import annotations

import asyncio
import io

import pytest

from pearl_stratum.pearlhash_framing import decode_frame, encode_frame, iter_frames


def test_encode_roundtrip() -> None:
    inner = bytes(range(32))
    wire = encode_frame(inner)
    # Lowercase hex, then exactly one trailing LF.
    assert wire.endswith(b"\n")
    assert wire[:-1] == inner.hex().encode("ascii")
    assert decode_frame(wire) == inner


def test_encode_empty_inner() -> None:
    """An empty body still produces a single newline (0 hex chars + LF)."""
    wire = encode_frame(b"")
    assert wire == b"\n"


def test_decode_strips_crlf() -> None:
    """Both LF and CRLF terminators should decode."""
    inner = b"\xde\xad\xbe\xef"
    assert decode_frame(b"deadbeef\n") == inner
    assert decode_frame(b"deadbeef\r\n") == inner
    assert decode_frame(b"deadbeef") == inner


def test_decode_rejects_non_hex() -> None:
    with pytest.raises(ValueError, match="non-hex"):
        decode_frame(b"deadbeeg\n")


def test_decode_rejects_odd_length() -> None:
    with pytest.raises(ValueError, match="non-hex"):
        # `bytes.fromhex` raises on odd-length input.
        decode_frame(b"deadbee\n")


def test_decode_rejects_empty_line() -> None:
    with pytest.raises(ValueError, match="empty frame"):
        decode_frame(b"\n")


@pytest.mark.asyncio
async def test_iter_frames_stream_reader() -> None:
    """`iter_frames` over an asyncio.StreamReader yields inner-bytes per line."""
    payload = b"01020304\n0a0b0c\n"
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()

    frames = []
    async for inner in iter_frames(reader):
        frames.append(inner)

    assert frames == [b"\x01\x02\x03\x04", b"\x0a\x0b\x0c"]


@pytest.mark.asyncio
async def test_iter_frames_empty_stream() -> None:
    reader = asyncio.StreamReader()
    reader.feed_eof()
    frames = [f async for f in iter_frames(reader)]
    assert frames == []
