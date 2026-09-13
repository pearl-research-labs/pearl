"""B-side peel epilogue: rewrite the shared kernel's A-layout peel in place.

The fused prep kernel emits the peel in its A-side layout,
``[ beta (.) E2 || B' @ F1^T ]``. The B-side layout is
``[ (beta (.) E2@F2 - B') @ F1^T || -(beta (.) E2) ]`` (`build_b_rows`), whose
mid half splits algebraically as

    (beta (.) E2@F2 - B') @ F1^T = (beta (.) E2) @ (F2 @ F1^T) - B' @ F1^T

so the whole rewrite is per-row work against the tiny ``(R, R)`` Gram factor
``G = F2 @ F1^T``. Both mid variants are tolerance-path (the reference itself
recomputes ``E2@F2`` through its tolerance matmul here), while the
``-(beta (.) E2)`` half is an exact bf16 negation of the bit-exact half the
fused kernel wrote.

Two launches on the caller's stream: ``_GramFactor`` reduces G once into the
caller-owned ``gram`` workspace (one CTA per F1 row, k strided across its
threads -- G is O(R^2 k) work, far too much to recompute per row-block CTA),
then ``_PeelFixup`` rewrites the peel rows. F2's e4m3 codes are read back out
of the packed noise operand ``f2_hl``, so the epilogue needs no inputs the
launch does not already hold.
"""

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
from cutlass import Float32

from ..protocol_constants import PACKED_NOISE_K, R

_CTA_THREADS = R * R
_ROWS_PER_CTA = _CTA_THREADS


class _GramFactor:
    """G = F2 @ F1^T into the ``(R, R)`` gram workspace (fp32, tolerance path)."""

    def __init__(self, k: int):
        assert k > 0
        self.k = k

    @cute.jit
    def __call__(
        self,
        mF2hl: cute.Tensor,
        mF1: cute.Tensor,
        mGram: cute.Tensor,
        stream: cuda_drv.CUstream,
    ):
        self.kernel(mF2hl, mF1, mGram).launch(
            grid=(R, 1, 1),
            block=(_CTA_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mF2hl: cute.Tensor, mF1: cute.Tensor, mGram: cute.Tensor):
        f8 = cutlass.Float8E4M3FN
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sPart = smem.allocate_tensor(
            Float32, cute.make_layout((_CTA_THREADS, R), stride=(R, 1)), byte_alignment=16
        )

        # One CTA per F1 row r1 = bidx. Threads stride j, so a warp's F1
        # bytes coalesce and each thread's F2 row ``f2_hl[j, 0:R]`` (the
        # packing stores the R codes, then pad to PACKED_NOISE_K) arrives as
        # one vectorized load feeding all R of its accumulators.
        r1 = bidx
        f2_codes = cute.make_tensor(
            cute.recast_ptr(mF2hl.iterator, dtype=f8),
            cute.make_layout(self.k * PACKED_NOISE_K),
        )
        accs = cute.make_rmem_tensor(R, Float32)
        accs.fill(Float32(0.0))
        f2_row = cute.make_rmem_tensor(R, f8)
        n_full = self.k // _CTA_THREADS
        for step in cutlass.range(n_full, unroll=1):
            j = step * _CTA_THREADS + tidx
            f1_val = mF1[r1, j].to(Float32)
            cute.autovec_copy(
                cute.make_tensor(f2_codes.iterator + j * PACKED_NOISE_K, cute.make_layout(R)),
                f2_row,
            )
            for r2 in cutlass.range_constexpr(R):
                accs[r2] += f2_row[r2].to(Float32) * f1_val
        j = n_full * _CTA_THREADS + tidx
        if j < self.k:
            f1_val = mF1[r1, j].to(Float32)
            cute.autovec_copy(
                cute.make_tensor(f2_codes.iterator + j * PACKED_NOISE_K, cute.make_layout(R)),
                f2_row,
            )
            for r2 in cutlass.range_constexpr(R):
                accs[r2] += f2_row[r2].to(Float32) * f1_val
        for r2 in cutlass.range_constexpr(R):
            sPart[tidx, r2] = accs[r2]
        cute.arch.barrier()

        # Fixed binary-tree order keeps repeat launches bit-identical.
        offset = _CTA_THREADS // 2
        while offset >= 1:
            if tidx < offset:
                for r2 in cutlass.range_constexpr(R):
                    sPart[tidx, r2] = sPart[tidx, r2] + sPart[tidx + offset, r2]
            cute.arch.barrier()
            offset //= 2
        if tidx < R:
            mGram[tidx, r1] = sPart[0, tidx]


class _PeelFixup:
    """Per-thread row rewrite of the peel, in place, against the gram factor."""

    def __init__(self, n: int):
        assert n > 0
        self.n = n

    @cute.jit
    def __call__(
        self,
        mPeel: cute.Tensor,
        mGram: cute.Tensor,
        stream: cuda_drv.CUstream,
    ):
        self.kernel(mPeel, mGram).launch(
            grid=(cute.ceil_div(self.n, _ROWS_PER_CTA), 1, 1),
            block=(_CTA_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mPeel: cute.Tensor, mGram: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sG = smem.allocate_tensor(
            Float32, cute.make_layout((R, R), stride=(R, 1)), byte_alignment=16
        )
        sG[tidx // R, tidx % R] = mGram[tidx // R, tidx % R]
        cute.arch.barrier()

        # One row per thread, in place (each thread rewrites only the halves
        # it read, so there is no cross-thread hazard).
        row = bidx * _ROWS_PER_CTA + tidx
        if row < self.n:
            e_half = cute.make_rmem_tensor(R, Float32)
            umma_half = cute.make_rmem_tensor(R, Float32)
            for i in cutlass.range_constexpr(R):
                e_half[i] = mPeel[row, i].to(Float32)
                umma_half[i] = mPeel[row, R + i].to(Float32)
            for i in cutlass.range_constexpr(R):
                mid = Float32(0.0)
                for r in cutlass.range_constexpr(R):
                    mid += e_half[r] * sG[r, i]
                mPeel[row, i] = (mid - umma_half[i]).to(cutlass.BFloat16)
                # Exact bf16 negation of the bit-exact beta (.) E2 half.
                mPeel[row, R + i] = (-e_half[i]).to(cutlass.BFloat16)
