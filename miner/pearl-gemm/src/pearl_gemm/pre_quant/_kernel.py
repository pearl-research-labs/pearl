"""Block-scale INT8 quantization of the BF16 activation (BF16 -> codes, scales).

One warp quantizes one 512-element pack block: each lane owns 16
consecutive BF16 elements (two 8-element scale groups), so the packed BF16
words a lane loads are exactly the groups it scales. The only job here is
producing the two committed blobs; the ``(sumsq, absmax)`` commit-stats
partials over the dequantized values are fused into the byte hasher
(``tensor_hash_plus_stats``), which re-derives them from the codes and
scales.

Output layout (``int8 blk8 bf16s``): two separate tensors, ``(m, k)`` int8
codes and ``(m, k/8)`` BF16 scales, each committed with its own keyed
Merkle tree. Both blob strides are powers of two, so a 512-element block
is 512 code bytes (exactly half a 1024-byte BLAKE3 chunk) plus 128 scale
bytes. Rows are whole blocks (``k % 512 == 0``); the global block index is
the row-major 512-element chunk index.

Quantization contract (bit-exact against ``PrequantMatrix.encode``
for finite, non-denormal inputs):

    amax  = max(amax(|group|), 2^-100)       # all-zero blocks get a tiny floor
    scale = bf16_rtne(amax / 127)            # fp32 IEEE division, never zero
    code  = clamp(rne(x * rcp(scale)), -127, 127)  # fp32 rcp then mul

Matches the reference's ``blocks * reciprocal(scales)`` (not ``blocks /
scales``): the two differ by 1 ULP for some inputs. The amax floor keeps the
reciprocal well-defined for all-zero groups, whose codes still quantize to 0.
"""

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from .._utils._stats_ops import (
    abs_max_bf16x2,
    bf16x2_hi_f32,
    bf16x2_lo_f32,
    hmax_bf16x2_f32,
)
from ..noisy_quant._quantization_ops import _f32x2_to_bf16x2
from ..protocol_constants import BLOCK_SCALE_GROUP

CHUNK_ELEMS = 512  # elements per packed block (and per (sumsq, absmax) pair)
_GROUPS_PER_THREAD = 2  # 16 elements per lane -> one pack block per warp
_CODE_MAX = 127.0  # symmetric int8 ceiling; -128 is never produced
_AMAX_FLOOR = 2.0**-100  # reference clamp_min: all-zero blocks get a tiny scale
_WARP = 32


@cute.kernel
def _pre_quant_kernel(
    mA: cute.Tensor,  # (m, k) BF16 activation
    mCodeWords: cute.Tensor,  # codes blob as u32 words (m * k / 4)
    mScaleWords: cute.Tensor,  # scales blob as u32 words (m * k / 16)
    threads_per_block: cutlass.Constexpr[int],
):
    m, k = mA.shape
    num_chunks: cutlass.Constexpr = m * k // CHUNK_ELEMS
    warps_per_block: cutlass.Constexpr = threads_per_block // _WARP

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % _WARP
    chunk = bidx * warps_per_block + tidx // _WARP

    # Warp-uniform guard (one chunk per warp).
    if chunk < num_chunks:
        # 16 consecutive BF16 elements = 8 packed words per lane.
        a_words = cute.make_tensor(
            cute.recast_ptr(mA.iterator, dtype=Uint32),
            cute.make_layout(m * k // 2),
        )
        rIn = cute.make_rmem_tensor(2 * 4 * _GROUPS_PER_THREAD, Uint32)
        cute.autovec_copy(cute.local_tile(a_words, (8,), (chunk * _WARP + lane,)), rIn)

        rCodes = cute.make_rmem_tensor(2 * _GROUPS_PER_THREAD, Uint32)
        scales = [Float32(0.0)] * _GROUPS_PER_THREAD
        for g in cutlass.range_constexpr(_GROUPS_PER_THREAD):
            # amax over the group's 8 |bf16| lanes, widened exactly to fp32.
            absw = abs_max_bf16x2(Uint32(0), rIn[4 * g])
            for w in cutlass.range_constexpr(1, 4):
                absw = abs_max_bf16x2(absw, rIn[4 * g + w])
            # Reference chain (PrequantMatrix.encode): amax floored at 2^-100
            # (all-zero blocks store a tiny positive scale and their codes
            # quantize to 0), fp32 divide, BF16 RTNE, widen back exactly.
            amax = cute.arch.fmax(hmax_bf16x2_f32(absw), Float32(_AMAX_FLOOR))
            s = (amax / Float32(_CODE_MAX)).to(cutlass.BFloat16).to(Float32)
            scales[g] = s
            # Match the reference's reciprocal-multiply (blocks * rcp(scales));
            # true division differs by 1 ULP for some inputs.
            inv = Float32(1.0) / s
            for cw in cutlass.range_constexpr(2):  # one packed code word per 2 input words
                packed = Uint32(0)
                for j in cutlass.range_constexpr(2):
                    word = rIn[4 * g + 2 * cw + j]
                    lo = cute.math.roundeven(bf16x2_lo_f32(word) * inv)
                    hi = cute.math.roundeven(bf16x2_hi_f32(word) * inv)
                    lo = cute.math.clamp(lo, Float32(-_CODE_MAX), Float32(_CODE_MAX))
                    hi = cute.math.clamp(hi, Float32(-_CODE_MAX), Float32(_CODE_MAX))
                    pair = (Uint32(lo.to(Int32)) & 0xFF) | ((Uint32(hi.to(Int32)) & 0xFF) << 8)
                    packed = packed | (pair << (16 * j))
                rCodes[2 * g + cw] = packed

        # A block is 512 code bytes (128 u32 words, 4 per lane) and 128 scale
        # bytes (32 u32 words, 1 per lane), so both blobs index off the same
        # slot. Offsets can exceed 2^31 for large activations; keep them
        # 64-bit.
        slot = Int64(chunk) * _WARP + lane
        cute.autovec_copy(rCodes, cute.local_tile(mCodeWords, (4,), (slot,)))
        # Two BF16 scales -> one u32 word (exact: values are bf16-valued).
        mScaleWords[slot] = _f32x2_to_bf16x2(scales[1], scales[0])


@cute.jit
def _pre_quant_launch(
    mA: cute.Tensor,
    mCodeWords: cute.Tensor,
    mScaleWords: cute.Tensor,
    stream: cuda_drv.CUstream,
    threads_per_block: cutlass.Constexpr[int],
):
    m, k = mA.shape
    assert k % CHUNK_ELEMS == 0 and k % BLOCK_SCALE_GROUP == 0
    num_chunks = m * k // CHUNK_ELEMS
    warps_per_block = threads_per_block // _WARP
    _pre_quant_kernel(mA, mCodeWords, mScaleWords, threads_per_block).launch(
        grid=[cute.ceil_div(num_chunks, warps_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
        stream=stream,
    )
