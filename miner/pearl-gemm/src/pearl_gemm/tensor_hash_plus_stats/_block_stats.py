"""Commit-stats reduction over the block-scaled activation blobs -- no decoding.

Each 512-element block (512 int8 codes plus 64 BF16 scales, in the two
separate blobs ``pre_quant`` writes) yields one fp32 ``(sumsq, absmax)``
partial -- the partials
``noisy_quant`` combines into ``alpha``/``beta``. Because the scale is
constant within an 8-element group, both stats factor and never require
materializing a dequantized value:

- ``sumsq``: ``sum((q*s)^2) == s^2 * sum(q^2)`` per group. ``sum(q^2)`` is
  an exact small integer (``dp4a`` on the raw code bytes) and ``s^2`` is
  exact in fp32 (8+8 significand bits), so each group term rounds exactly
  once -- bit-identical to the reference
  ``PrequantMatrix.exact_norms``, which defines the protocol row sum
  of squares over these exact code-scale products.
- ``absmax``: ``max|bf16(q*s)| == bf16(max_groups s * max|q|)`` -- scaling by
  a non-negative constant commutes with max, and BF16 RTNE is monotone, so
  the rounding also moves out past the max over groups. An integer byte-max
  per group (prmt sign-extension + native ``max/min.s16x2``; the SIMD-video
  byte ops are emulated expensively on SM100), one exact multiply, and a
  single BF16 rounding per block equal the opened-row absmax bit-exactly.

The reduction rides the merkle roots kernel that hashes the codes blob (one
fused ``tensor_hash_plus_stats`` kernel). The codes come out of the
message-block registers the hasher already staged -- the codes blob is read
once for both -- and only the scales blob, a fifth of the bytes, is read
from global memory (one 16-byte load per 64 code bytes, issued ahead of the
compression that hides it). The stats work is independent of the BLAKE3
chain, whose serial 64-byte compressions leave the consumer warps
latency-bound with issue slots to spare.

A stats block is 512 code bytes and the reduction takes one of two shapes,
by Merkle leaf size (``chunk_size``):

- Leaves covering whole stats blocks: the consumer thread that hashes a
  block finishes it. On the TMA path a chunk is streamed through shared
  memory in windows of ``thread_load_size`` bytes, so only part of a block
  is resident at a time: ``_stats_carry`` folds each window's sub-sum into
  the block's fp32 pairwise tree, over the window count (a power of two)
  that spans a block. On the sync path the carry window is one 64-byte
  message block.
- Sub-block leaves (64..448): a block spans several consumers' chunks, so
  each stages its message blocks' ``(sumsq, absmax)`` pairs in smem and one
  owner thread per block folds the ``_STATS_MSG_BLOCKS`` pairs after the
  CTA barrier (``_fold_small_leaf_stats``).

Both shapes produce the balanced in-order tree over the block's 64 group
terms, so the partials are bit-identical across leaves and load paths; the
fp32 cross-group sum order is the only ulp-level freedom, absorbed by the
protocol's ``_round_l2_to_grid`` exactly as for the previous accumulation
orders.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32

from .._utils._stats_ops import _asm_f32, _asm_u32, bf16x2_hi_f32, bf16x2_lo_f32
from ..pre_quant._kernel import CHUNK_ELEMS as _CHUNK_ELEMS
from ._blake3 import MSG_BLOCK_SIZE as _MSG_BLOCK_SIZE

# One (sumsq, absmax) pair per 512-element block, which in the codes blob is
# 512 bytes and in the scales blob 128 bytes (one BF16 per 8 codes).
_STATS_CODE_BYTES = _CHUNK_ELEMS
# Hashed 64-byte message blocks per stats block: the granule the small-leaf
# cross-thread fold stages in smem (a message block never straddles a stats
# block, whatever the Merkle leaf size).
_STATS_MSG_BLOCKS = _STATS_CODE_BYTES // _MSG_BLOCK_SIZE
# A unit is the 16 code bytes covered by one packed pair of BF16 scales: the
# smallest piece of a block that needs no scale word twice.
_UNIT_CODE_BYTES = 16
_UNIT_CODE_WORDS = _UNIT_CODE_BYTES // 4
_GROUPS_PER_UNIT = 2


def _dp4a(a, b, c):
    """c + dot(a, b) over packed signed bytes (exact integer)."""
    return _asm_u32("dp4a.s32.s32 $0, $1, $2, $3;", "=r,r,r,r", Uint32(a), Uint32(b), Uint32(c))


def _prmt(a, ctl):
    """Byte-permute of ``a`` (second source 0); nibble 8|i sign-replicates."""
    return _asm_u32("prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", Uint32(a), Uint32(0), Uint32(ctl))


def _max_s16x2(a, b):
    return _asm_u32("max.s16x2 $0, $1, $2;", "=r,r,r", Uint32(a), Uint32(b))


def _min_s16x2(a, b):
    return _asm_u32("min.s16x2 $0, $1, $2;", "=r,r,r", Uint32(a), Uint32(b))


def _neg_s16x2(a):
    """Per-lane negate: ~a + 1 (no cross-lane carry: |lane| <= 127)."""
    return _asm_u32(
        "{\n\t"
        ".reg .b32 t, o;\n\t"
        "xor.b32 t, $1, 0xFFFFFFFF;\n\t"
        "mov.b32 o, 0x00010001;\n\t"
        "add.s16x2 $0, t, o;\n\t"
        "}",
        "=r,r",
        Uint32(a),
    )


def _select_f32(pred, a, b):
    """``a`` when ``pred`` (a 0/1 u32) is set, else ``b``; never rounds."""
    return _asm_f32(
        "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, $3, 0;\n\tselp.f32 $0, $1, $2, p;\n\t}",
        "=f,f,f,r",
        Float32(a),
        Float32(b),
        Uint32(pred),
    )


def _group_absmax_q(w0, w1):
    """max|q| over a group's 8 packed int8 codes, via native byte ops.

    Sign-extend the bytes into s16x2 lanes (prmt), then use
    ``max|x| == max(max(x), -min(x))`` so only one negate is needed.
    """
    p0 = _prmt(w0, Uint32(0x9180))
    p1 = _prmt(w0, Uint32(0xB3A2))
    p2 = _prmt(w1, Uint32(0x9180))
    p3 = _prmt(w1, Uint32(0xB3A2))
    mx = _max_s16x2(_max_s16x2(p0, p1), _max_s16x2(p2, p3))
    mn = _min_s16x2(_min_s16x2(p0, p1), _min_s16x2(p2, p3))
    ab = _max_s16x2(mx, _neg_s16x2(mn))
    h = _max_s16x2(ab, ab >> 16)
    return h & 0xFFFF


def _u32_f32(v):
    """Exact u32 -> fp32 conversion (values stay far below 2^24)."""
    return _asm_f32("cvt.rn.f32.u32 $0, $1;", "=f,r", Uint32(v))


def _unit_stats(rCodes, word, scale_word):
    """``(sumsq, absmax)`` over the 16 code bytes of one packed scale pair.

    ``word`` is the first of the unit's four code words in ``rCodes``; the
    scale pair's low lane scales the first group, the high lane the second.
    """
    scales = [bf16x2_lo_f32(scale_word), bf16x2_hi_f32(scale_word)]
    ssq = Float32(0.0)
    mx = Float32(0.0)
    for g in range(_GROUPS_PER_UNIT):
        s = scales[g]
        w0 = rCodes[word + 2 * g]
        w1 = rCodes[word + 2 * g + 1]
        # Exact integer reductions over the raw code bytes.
        q2 = _dp4a(w1, w1, _dp4a(w0, w0, Uint32(0)))
        maxq = _group_absmax_q(w0, w1)
        # Group sumsq term: s*s is exact, the product rounds once -- the same
        # single rounding as the reference row_norms.
        ssq = ssq + (s * s) * _u32_f32(q2)
        # Group absmax: s * max|q| is exact in fp32 (8+7 significand bits).
        # BF16 RTNE is monotone, so the block's single rounding of the
        # largest product equals the reference's max over rounded groups.
        mx = cute.arch.fmax(mx, s * _u32_f32(maxq))
    return ssq, mx


def _pairwise_sum(terms):
    """Balanced in-order fp32 tree over a power-of-two number of terms."""
    while len(terms) > 1:
        terms = [terms[i] + terms[i + 1] for i in range(0, len(terms), 2)]
    return terms[0]


def _msg_block_stats(rBlock, rScales, num_words: cutlass.Constexpr[int]):
    """``(sumsq, absmax)`` over the code words of one hashed message block.

    Both are reduced in the same balanced order the whole block uses, so
    where the message-block boundaries fall inside a stats block does not
    change the result.
    """
    sums = []
    mx = Float32(0.0)
    for u in range(num_words // _UNIT_CODE_WORDS):
        ssq_u, mx_u = _unit_stats(rBlock, u * _UNIT_CODE_WORDS, rScales[u])
        sums.append(ssq_u)
        mx = cute.arch.fmax(mx, mx_u)
    return _pairwise_sum(sums), mx


def _window_ones(index, levels: cutlass.Constexpr[int]):
    """``ones[l]``: the window index's low ``l`` bits are all set (0/1 u32).

    Level ``l``'s pairwise accumulator is complete exactly when it is set,
    so the whole carry is driven by these ``levels + 1`` uniform predicates
    (``ones[levels]`` marks the last window of a stats block).
    """
    ones = [Uint32(1)]
    for level in range(1, levels + 1):
        mask = (1 << level) - 1
        ones.append(((index & mask) == mask).to(Uint32))
    return ones


def _stats_carry(rAcc, sub, ones, levels: cutlass.Constexpr[int]):
    """Fold one window's sub-sum into the stats block's pairwise tree.

    ``rAcc[l]`` holds the sum of the last completed run of ``2^l`` windows.
    A window either completes level ``l`` (carrying up, ``ones[l + 1]``) or
    parks its running sum there, which is the binary-counter form of the
    balanced tree: the value returned on the block's last window is the
    balanced sum over all of its windows, in order.
    """
    x = sub
    for level in range(levels):
        carried = rAcc[level] + x
        rAcc[level] = _select_f32(ones[level] ^ ones[level + 1], x, rAcc[level])
        x = _select_f32(ones[level + 1], carried, x)
    return x
