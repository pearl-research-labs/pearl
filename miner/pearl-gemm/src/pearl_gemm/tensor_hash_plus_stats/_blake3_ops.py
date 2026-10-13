"""Traced single-compression BLAKE3 helper shared by the small hash kernels.

Built on the compression primitives in ``._blake3`` (same
round/permutation trace-time expansion), with a parametric ``block_len`` for
short single-block messages (seed finalize, E1 noise lines, lottery hashes).
"""

import cutlass
import cutlass.cute as cute

from ._blake3 import (
    CHUNK_END,
    CHUNK_START,
    IV0,
    IV1,
    IV2,
    IV3,
    IV4,
    IV5,
    IV6,
    IV7,
    KEYED_HASH,
    ROOT,
    _blake3_permute,
    _blake3_round,
    _rotr32,
)

# Flags of a one-shot (single-block, <= 64-byte message) blake3 hash.
SINGLE_BLOCK_FLAGS = CHUNK_START | CHUNK_END | ROOT
SINGLE_BLOCK_KEYED_FLAGS = SINGLE_BLOCK_FLAGS | KEYED_HASH

_IV = (IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7)


def compress(cv, m, block_len, flags):
    """One BLAKE3 compression; ``cv`` (8) / ``m`` (16) are lists of Uint32 DSL values.

    ``block_len``/``flags`` are trace-time ints; the 64-bit counter is always 0
    (all callers hash single-block messages). Returns the 8-word output
    chaining value.
    """
    st = list(cv) + [
        cutlass.Uint32(IV0),
        cutlass.Uint32(IV1),
        cutlass.Uint32(IV2),
        cutlass.Uint32(IV3),
        cutlass.Uint32(0),
        cutlass.Uint32(0),
        cutlass.Uint32(block_len),
        cutlass.Uint32(flags),
    ]
    msg = list(m)
    for _ in range(6):
        _blake3_round(st, msg)
        msg = _blake3_permute(msg)
    _blake3_round(st, msg)
    return [st[i] ^ st[i + 8] for i in range(8)]


def iv_cv():
    """Fresh IV chaining value (unkeyed hash start)."""
    return [cutlass.Uint32(w) for w in _IV]


@cute.jit
def compress_rolled(cv, m, block_len, flags):
    """Rolled-loop twin of :func:`compress` (identical arithmetic).

    One static copy of the round body executed 7 times (BLAKE3 applies the
    SAME message permutation between rounds, so the rolled form only
    rebinds SSA values -- bit-identical results). The unrolled form is ~11
    KB of straight-line SASS whose cold fetch dominates the E1 prologue.
    """
    s0 = cv[0]
    s1 = cv[1]
    s2 = cv[2]
    s3 = cv[3]
    s4 = cv[4]
    s5 = cv[5]
    s6 = cv[6]
    s7 = cv[7]
    s8 = cutlass.Uint32(IV0)
    s9 = cutlass.Uint32(IV1)
    s10 = cutlass.Uint32(IV2)
    s11 = cutlass.Uint32(IV3)
    s12 = cutlass.Uint32(0)
    s13 = cutlass.Uint32(0)
    s14 = cutlass.Uint32(block_len)
    s15 = cutlass.Uint32(flags)
    m0 = m[0]
    m1 = m[1]
    m2 = m[2]
    m3 = m[3]
    m4 = m[4]
    m5 = m[5]
    m6 = m[6]
    m7 = m[7]
    m8 = m[8]
    m9 = m[9]
    m10 = m[10]
    m11 = m[11]
    m12 = m[12]
    m13 = m[13]
    m14 = m[14]
    m15 = m[15]
    for _round in cutlass.range(7, unroll=1):
        s0 = s0 + (s4 + m0)
        s12 = _rotr32(s12 ^ s0, 16)
        s8 = s8 + s12
        s4 = _rotr32(s4 ^ s8, 12)
        s0 = s0 + (s4 + m1)
        s12 = _rotr32(s12 ^ s0, 8)
        s8 = s8 + s12
        s4 = _rotr32(s4 ^ s8, 7)
        s1 = s1 + (s5 + m2)
        s13 = _rotr32(s13 ^ s1, 16)
        s9 = s9 + s13
        s5 = _rotr32(s5 ^ s9, 12)
        s1 = s1 + (s5 + m3)
        s13 = _rotr32(s13 ^ s1, 8)
        s9 = s9 + s13
        s5 = _rotr32(s5 ^ s9, 7)
        s2 = s2 + (s6 + m4)
        s14 = _rotr32(s14 ^ s2, 16)
        s10 = s10 + s14
        s6 = _rotr32(s6 ^ s10, 12)
        s2 = s2 + (s6 + m5)
        s14 = _rotr32(s14 ^ s2, 8)
        s10 = s10 + s14
        s6 = _rotr32(s6 ^ s10, 7)
        s3 = s3 + (s7 + m6)
        s15 = _rotr32(s15 ^ s3, 16)
        s11 = s11 + s15
        s7 = _rotr32(s7 ^ s11, 12)
        s3 = s3 + (s7 + m7)
        s15 = _rotr32(s15 ^ s3, 8)
        s11 = s11 + s15
        s7 = _rotr32(s7 ^ s11, 7)
        s0 = s0 + (s5 + m8)
        s15 = _rotr32(s15 ^ s0, 16)
        s10 = s10 + s15
        s5 = _rotr32(s5 ^ s10, 12)
        s0 = s0 + (s5 + m9)
        s15 = _rotr32(s15 ^ s0, 8)
        s10 = s10 + s15
        s5 = _rotr32(s5 ^ s10, 7)
        s1 = s1 + (s6 + m10)
        s12 = _rotr32(s12 ^ s1, 16)
        s11 = s11 + s12
        s6 = _rotr32(s6 ^ s11, 12)
        s1 = s1 + (s6 + m11)
        s12 = _rotr32(s12 ^ s1, 8)
        s11 = s11 + s12
        s6 = _rotr32(s6 ^ s11, 7)
        s2 = s2 + (s7 + m12)
        s13 = _rotr32(s13 ^ s2, 16)
        s8 = s8 + s13
        s7 = _rotr32(s7 ^ s8, 12)
        s2 = s2 + (s7 + m13)
        s13 = _rotr32(s13 ^ s2, 8)
        s8 = s8 + s13
        s7 = _rotr32(s7 ^ s8, 7)
        s3 = s3 + (s4 + m14)
        s14 = _rotr32(s14 ^ s3, 16)
        s9 = s9 + s14
        s4 = _rotr32(s4 ^ s9, 12)
        s3 = s3 + (s4 + m15)
        s14 = _rotr32(s14 ^ s3, 8)
        s9 = s9 + s14
        s4 = _rotr32(s4 ^ s9, 7)
        # fixed inter-round message permutation (rebinding only)
        n0 = m2
        n1 = m6
        n2 = m3
        n3 = m10
        n4 = m7
        n5 = m0
        n6 = m4
        n7 = m13
        n8 = m1
        n9 = m11
        n10 = m12
        n11 = m5
        n12 = m9
        n13 = m14
        n14 = m15
        n15 = m8
        m0 = n0
        m1 = n1
        m2 = n2
        m3 = n3
        m4 = n4
        m5 = n5
        m6 = n6
        m7 = n7
        m8 = n8
        m9 = n9
        m10 = n10
        m11 = n11
        m12 = n12
        m13 = n13
        m14 = n14
        m15 = n15
    return [
        s0 ^ s8,
        s1 ^ s9,
        s2 ^ s10,
        s3 ^ s11,
        s4 ^ s12,
        s5 ^ s13,
        s6 ^ s14,
        s7 ^ s15,
    ]
