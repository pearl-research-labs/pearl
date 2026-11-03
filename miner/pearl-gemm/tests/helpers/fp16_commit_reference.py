"""Pure-Python oracle for the FP16 operand keyed-BLAKE3 Merkle commitment.

Faithful port of ``zk-pow/src/api/fp16/commitment.rs::commit_operand`` and the
``pearl_blake3::MerkleTree::with_chunk_len`` tree discipline it builds on:

* The committed byte image is the FP16 rows as little-endian ``u16`` row-major
  (``rows_to_bytes``), zero-padded up to a multiple of ``HashId::chunk_len``
  (``HashId::pad``).
* Leaves are fixed ``chunk_len``-byte chunks. Each leaf's hash is the keyed
  BLAKE3 *non-root* chunk chaining value with the BLAKE3 chunk counter equal to
  the leaf index (``Blake3Hasher::chunk_cv(chunk, i)``). Because every allowed
  ``chunk_len`` (128/256/512/1024) is <= one native BLAKE3 chunk (1024), a leaf
  is a single chunk of ``chunk_len/64`` full 64-byte blocks.
* Internal layers pair adjacent CVs with the keyed non-root parent compression
  (``merge_subtrees_non_root``); a lone odd node is promoted unchanged. When a
  layer has reduced to exactly two nodes the root is their keyed *root*-finalized
  parent compression (``merge_subtrees_root``).
* A single-leaf tree (padded image <= chunk_len) is special: its root is the
  ROOT-finalized keyed hash of the chunk itself, not a non-root chunk CV.

This is an independent BLAKE3 implementation (compression written from the spec)
so it cross-checks tree structure against the Rust oracle without sharing code
with the ``blake3`` crate. It is the trusted model for the ``sm_80`` kernel.
"""

from __future__ import annotations

# ---- BLAKE3 domain-separation flags (match pearl_blake3::hasher) ----
CHUNK_START = 1
CHUNK_END = 2
PARENT = 4
ROOT = 8
KEYED_HASH = 16

CHUNK_LEN = 1024
OUT_LEN = 32

IV = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

_MASK = 0xFFFFFFFF
_PERM = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

ALLOWED_CHUNK_LENS = (128, 256, 512, 1024)


def _rotr32(x: int, n: int) -> int:
    x &= _MASK
    return ((x >> n) | (x << (32 - n))) & _MASK


def _compress(cv, block, counter, block_len, flags):
    """One BLAKE3 compression. Returns the 16-word output state.

    ``cv`` is 8 words, ``block`` is 16 message words, ``counter`` a 64-bit int.
    The first 8 output words are the chaining value / 32-byte hash output.
    """
    s = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3],
        counter & _MASK, (counter >> 32) & _MASK, block_len & _MASK, flags & _MASK,
    ]
    v = list(block)

    def g(a, b, c, d, x, y):
        s[a] = (s[a] + s[b] + x) & _MASK
        s[d] = _rotr32(s[d] ^ s[a], 16)
        s[c] = (s[c] + s[d]) & _MASK
        s[b] = _rotr32(s[b] ^ s[c], 12)
        s[a] = (s[a] + s[b] + y) & _MASK
        s[d] = _rotr32(s[d] ^ s[a], 8)
        s[c] = (s[c] + s[d]) & _MASK
        s[b] = _rotr32(s[b] ^ s[c], 7)

    for rnd in range(7):
        g(0, 4, 8, 12, v[0], v[1])
        g(1, 5, 9, 13, v[2], v[3])
        g(2, 6, 10, 14, v[4], v[5])
        g(3, 7, 11, 15, v[6], v[7])
        g(0, 5, 10, 15, v[8], v[9])
        g(1, 6, 11, 12, v[10], v[11])
        g(2, 7, 8, 13, v[12], v[13])
        g(3, 4, 9, 14, v[14], v[15])
        if rnd < 6:
            v = [v[_PERM[i]] for i in range(16)]

    out = [0] * 16
    for i in range(8):
        out[i] = (s[i] ^ s[i + 8]) & _MASK
        out[i + 8] = (s[i + 8] ^ cv[i]) & _MASK
    return out


def _key_words(key: bytes) -> list[int]:
    assert len(key) == 32
    return [int.from_bytes(key[4 * i:4 * i + 4], "little") for i in range(8)]


def _blocks_of(data: bytes) -> list[list[int]]:
    """Split ``data`` (length a multiple of 64) into 64-byte blocks of 16 LE words."""
    assert len(data) % 64 == 0
    blocks = []
    for off in range(0, len(data), 64):
        blk = data[off:off + 64]
        blocks.append([int.from_bytes(blk[4 * w:4 * w + 4], "little") for w in range(16)])
    return blocks


def _chunk_cv(data: bytes, chunk_index: int, key_words: list[int], root: bool) -> list[int]:
    """Keyed BLAKE3 chunk chaining value (``root`` toggles ROOT finalization).

    ``data`` is one leaf's bytes, length a multiple of 64 and <= 1024.
    """
    blocks = _blocks_of(data)
    cv = list(key_words)
    n = len(blocks)
    for i, blk in enumerate(blocks):
        flags = KEYED_HASH
        if i == 0:
            flags |= CHUNK_START
        if i == n - 1:
            flags |= CHUNK_END
            if root:
                flags |= ROOT
        out = _compress(cv, blk, chunk_index, 64, flags)
        cv = out[:8]
    return cv


def _parent(left: list[int], right: list[int], key_words: list[int], root: bool) -> list[int]:
    msg = list(left) + list(right)
    flags = KEYED_HASH | PARENT | (ROOT if root else 0)
    out = _compress(key_words, msg, 0, 64, flags)
    return out[:8]


def _words_to_bytes(words: list[int]) -> bytes:
    return b"".join(int(w & _MASK).to_bytes(4, "little") for w in words[:8])


def rows_to_bytes(rows) -> bytes:
    """Little-endian ``u16`` row-major byte image of the FP16 operand."""
    import numpy as np

    arr = np.asarray(rows, dtype=np.uint16).reshape(-1)
    return arr.astype("<u2").tobytes()


def pad(data: bytes, chunk_len: int) -> bytes:
    rem = len(data) % chunk_len
    if rem:
        data = data + b"\x00" * (chunk_len - rem)
    return data


def commit_operand_root(rows, num_rows: int, k: int, key: bytes, chunk_len: int = 1024) -> bytes:
    """Return the 32-byte keyed-BLAKE3 Merkle root of the FP16 operand.

    Mirrors ``commit_operand(rows, num_rows, k, HashId::from(chunk_len), key)``.
    """
    assert chunk_len in ALLOWED_CHUNK_LENS, chunk_len
    kw = _key_words(key)
    data = pad(rows_to_bytes(rows), chunk_len)

    if len(data) == 0:
        return b"\x00" * 32
    if len(data) <= chunk_len:
        # Single-leaf tree: ROOT-finalized keyed hash of the (one) chunk.
        return _words_to_bytes(_chunk_cv(data, 0, kw, root=True))

    # Leaf layer: one non-root chunk CV per chunk_len-byte leaf.
    num_leaves = len(data) // chunk_len
    layer = [
        _chunk_cv(data[i * chunk_len:(i + 1) * chunk_len], i, kw, root=False)
        for i in range(num_leaves)
    ]
    # Reduce pairwise (promote lone odd node) until exactly two remain.
    while len(layer) > 2:
        nxt = []
        for i in range(0, len(layer), 2):
            if i + 1 < len(layer):
                nxt.append(_parent(layer[i], layer[i + 1], kw, root=False))
            else:
                nxt.append(layer[i])
        layer = nxt
    # Final combine is ROOT-finalized.
    return _words_to_bytes(_parent(layer[0], layer[1], kw, root=True))
