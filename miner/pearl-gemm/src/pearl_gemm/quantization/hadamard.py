"""Fused Hadamard + smooth-scale + 7-bit symmetric quantization kernel (CuTe DSL)."""

import math
from typing import Final

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Float32, Int8, Int32, const_expr
from cutlass.cute.nvgpu import cpasync

from .math_utils import TORCH_TO_CUTLASS, fmin_f32, rint_f32
from .stream_utils import get_stream

# ── Private helpers ───────────────────────────────────────────────────────


def _largest_divisor_leq(n, cap):
    """Largest divisor of *n* that is ≤ *cap* (always ≥ 1).

    Used to split the ``num_blocks_N`` X vec-blocks into equal smooth chunks
    with no partial remainder, so every chunk holds exactly ``blocks_per_chunk``
    whole blocks and ``num_chunks * blocks_per_chunk == num_blocks_N``. Runs once
    at trace time with ``n`` (= num_blocks_N) and ``cap`` both ≤ ~16.
    """
    return max((d for d in range(1, min(cap, n) + 1) if n % d == 0), default=1)


def _make_fake(dtype, shape, div=1):
    """2D fake tensor with a symbolic leading-dim stride (inner stride = 1)."""
    if dtype is None:
        return None
    stride = tuple(cute.sym_int64(divisibility=div) if i == 0 else 1 for i in range(len(shape)))
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=max(div * dtype.width // 8, 1)
    )


def _tiled_copy_2d(dtype, threads_per_row, num_threads, num_copy_elems=1, is_async=False):
    """Host-side: build a 2D tiled copy atom with row-major thread layout."""
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row), order=(1, 0)
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@cute.jit
def _get_copy_atom(dtype, num_copy_elems, is_async: cutlass.Constexpr[bool] = False):
    """Device-side: pick cp.async vs. sync copy based on *is_async* constexpr."""
    num_copy_bits = num_copy_elems * dtype.width
    # SIM108 off: const_expr branches emit different code paths at compile time;
    # the ternary form is not always equivalent under CuTe DSL tracing.
    if const_expr(is_async):  # noqa: SIM108
        copy_op = cpasync.CopyG2SOp()
    else:
        copy_op = cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@cute.jit
def _copy(src, dst, pred=None, is_async: cutlass.Constexpr[bool] = False):
    """Issue a tiled copy from *src* to *dst*, optionally predicated."""
    num_copy_elems = src.shape[0][0]
    copy_atom = _get_copy_atom(src.element_type, num_copy_elems, is_async)
    if const_expr(pred is not None):
        cute.copy(copy_atom, src, dst, pred=pred)
    else:
        cute.copy(copy_atom, src, dst)


@cute.jit
def _predicate_k(tAcA, limit):
    """Column-direction predicate tensor: ``True`` where col < *limit*."""
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@cute.jit
def _apply_smooth_chunk(
    x_work,
    tXcX_raw,
    sSmooth,
    max_abs_local,
    row_in_tile,
    N,
    si_start: cutlass.Constexpr[int],
    chunk_col_start: cutlass.Constexpr[int],
    elems_per_chunk: cutlass.Constexpr[int],
    smooth_broadcast: cutlass.Constexpr[bool],
    is_even_N: cutlass.Constexpr[bool],
):
    """Multiply one smooth-scale chunk into ``x_work`` (in place) and fold the
    result into the running ``|max|``, which is returned.

    The ``const_expr`` flags pick the broadcast (row 0) vs per-token smem row
    and whether column predication is needed. Each combination inlines to a
    single straight-line loop, so this replaces the four hand-unrolled branches
    that used to live in the kernel with no change to the generated code.
    """
    for si in cutlass.range_constexpr(elems_per_chunk):
        idx = const_expr(si_start + si)
        c = tXcX_raw[idx][1]
        local_c = c - chunk_col_start
        if const_expr(is_even_N):
            # Whole tile in bounds: no column predicate needed.
            if const_expr(smooth_broadcast):
                v = x_work[idx] * sSmooth[0, local_c]
            else:
                v = x_work[idx] * sSmooth[row_in_tile, local_c]
            x_work[idx] = v
            max_abs_local = cute.arch.fmax(max_abs_local, cute.arch.fmax(v, -v))
        else:
            # Predicate the smem read/write; OOB lanes keep their (zero) value.
            if c < N:
                if const_expr(smooth_broadcast):
                    x_work[idx] = x_work[idx] * sSmooth[0, local_c]
                else:
                    x_work[idx] = x_work[idx] * sSmooth[row_in_tile, local_c]
            max_abs_local = cute.arch.fmax(max_abs_local, cute.arch.fmax(x_work[idx], -x_work[idx]))
    return max_abs_local


# ── Host-side JIT ─────────────────────────────────────────────────────────


@cute.jit
def hadamard_quant_jit(
    mInput,
    mSmooth,
    mOutput,
    mScales,
    stream: cuda_drv.CUstream,
    N: cutlass.Constexpr[int],
    has_smooth: cutlass.Constexpr[bool],
    smooth_broadcast: cutlass.Constexpr[bool],
    has_hadamard: cutlass.Constexpr[bool],
    hadamard_k: cutlass.Constexpr[int],
    max_val: cutlass.Constexpr[int],
):
    num_threads = 128 if N <= 4096 else 256
    threads_per_row = min(N, num_threads)

    x_width = mInput.element_type.width
    vecsize = math.gcd(N, 128 // x_width)
    num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)
    tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)

    tiled_copy = _tiled_copy_2d(mInput.element_type, threads_per_row, num_threads, vecsize)

    # Float32 tiled copy for smooth scale: 4 float32 = 128 bits per load.
    # Chunk the smooth scale by whole X vec-blocks so each chunk's SMEM column stride
    # (smooth_smem_cols) equals X's per-chunk column advance, keeping
    # `local_c = col - ci*smooth_smem_cols` in [0, smooth_smem_cols) for every N.
    # blocks_per_chunk divides num_blocks_N so no columns are dropped, and the SMEM
    # budget stays ≤2048 cols.
    vecsize_f32 = math.gcd(N, 4)
    cols_per_block = vecsize * threads_per_row  # column span of one X vec-block
    blocks_per_chunk = _largest_divisor_leq(num_blocks_N, max(1, 2048 // cols_per_block))
    num_chunks = num_blocks_N // blocks_per_chunk
    smooth_smem_cols = blocks_per_chunk * cols_per_block
    tiled_copy_smooth = (
        _tiled_copy_2d(Float32, threads_per_row, num_threads, vecsize_f32, is_async=True)
        if has_smooth
        else tiled_copy
    )

    hadamard_quant_kernel(
        mInput,
        mSmooth,
        mOutput,
        mScales,
        tiler_mn,
        tiled_copy,
        tiled_copy_smooth,
        threads_per_row,
        has_hadamard,
        hadamard_k,
        smooth_broadcast,
        vecsize,
        vecsize_f32,
        smooth_smem_cols,
        num_chunks,
        max_val,
    ).launch(
        grid=[cute.ceil_div(mInput.shape[0], tiler_mn[0]), 1, 1],
        block=[num_threads, 1, 1],
        stream=stream,
    )


# ── Device-side kernel ────────────────────────────────────────────────────


@cute.kernel
def hadamard_quant_kernel(  # noqa: C901  (single fused pipeline; stages share register/smem state)
    mInput,
    mSmooth: cute.Tensor | None,
    mOutput,
    mScales,
    tiler_mn: cute.Shape,
    tiled_copy: cute.TiledCopy,
    tiled_copy_smooth: cute.TiledCopy,
    threads_per_row: cutlass.Constexpr[int],
    has_hadamard: cutlass.Constexpr[bool],
    hadamard_k: cutlass.Constexpr[int],
    smooth_broadcast: cutlass.Constexpr[bool] = False,
    vecsize: cutlass.Constexpr[int] = 8,
    vecsize_f32: cutlass.Constexpr[int] = 4,
    smooth_smem_cols: cutlass.Constexpr[int] = 2048,
    num_chunks: cutlass.Constexpr[int] = 4,
    max_val: cutlass.Constexpr[int] = 64,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    shape = mInput.shape

    smem = cutlass.utils.SmemAllocator()
    # No SMEM for X: load directly GMEM→registers to save 16KB SMEM and increase occupancy.

    # Reduction buffer for cross-warp max
    tv_layout = tiled_copy.layout_tv_tiled
    num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
    warps_per_row = (
        num_warps
        if cute.rank(tv_layout.shape[0]) == 1
        else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
    )
    reduction_buffer = smem.allocate_tensor(
        Float32,
        cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, 1)),
            order=(1, 0),
        ),
        byte_alignment=8,
    )

    # Shared memory for smooth scale (float32): double-buffered for pipelining.
    # Two chunk-sized SMEM buffers allow overlapping GMEM loads with smooth application.
    # smooth_smem_cols spans blocks_per_chunk whole X vec-blocks (≤2048 cols).
    if const_expr(mSmooth is not None):
        sSmooth0 = smem.allocate_tensor(
            Float32,
            cute.make_ordered_layout((tiler_mn[0], smooth_smem_cols), order=(1, 0)),
            byte_alignment=16,
        )
        sSmooth1 = smem.allocate_tensor(
            Float32,
            cute.make_ordered_layout((tiler_mn[0], smooth_smem_cols), order=(1, 0)),
            byte_alignment=16,
        )

    idX = cute.make_identity_tensor(shape)
    cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

    gX = cute.local_tile(mInput, tiler_mn, (bidx, 0))
    gO = cute.local_tile(mOutput, tiler_mn, (bidx, 0))
    gScales = cute.local_tile(mScales, (tiler_mn[0], 1), (bidx, 0))

    thr_copy = tiled_copy.get_slice(tidx)
    tXgX = thr_copy.partition_S(gX)
    tXcX_raw = thr_copy.partition_S(cX)

    is_even_N = const_expr(shape[1] == tiler_mn[1])
    tXpX = _predicate_k(tXcX_raw, limit=shape[1]) if not is_even_N else None

    tXcX = tXcX_raw[(0, None), None, None]
    row = tXcX[0][0]

    tXrX = cute.make_rmem_tensor_like(tXgX)
    num_elems_x = cute.size(tXrX)

    # ── Issue smooth chunk 0 async FIRST, then load X GMEM→registers ────────
    # Smooth chunk 0 loads in background while X loads occupy L/S units.
    # When the row tile overhangs N (is_even_N False — e.g. N not a multiple of
    # vecsize*threads_per_row), the trailing chunk's GMEM tile extends past
    # mSmooth's column extent. Predicate every chunk load by global column < N so
    # cp.async zero-fills (and skips the GMEM read) for OOB lanes. Interior chunks
    # get an all-true predicate (full-speed load); only the boundary chunk masks.
    smooth_needs_pred = const_expr(not is_even_N)
    if const_expr(mSmooth is not None):
        thr_copy_smooth = tiled_copy_smooth.get_slice(tidx)
        row_in_tile = row % tiler_mn[0]
        elems_per_chunk = const_expr(num_elems_x // num_chunks)
        if const_expr(smooth_needs_pred):
            idSmooth = cute.make_identity_tensor(mSmooth.shape)
        if const_expr(smooth_broadcast):
            gSmooth_tile0 = cute.local_tile(mSmooth, (1, smooth_smem_cols), (0, 0))
        else:
            gSmooth_tile0 = cute.local_tile(mSmooth, (tiler_mn[0], smooth_smem_cols), (bidx, 0))
        tSgS0 = thr_copy_smooth.partition_S(gSmooth_tile0)
        tSsS0 = thr_copy_smooth.partition_D(sSmooth0)
        if const_expr(smooth_needs_pred):
            if const_expr(smooth_broadcast):
                cSmooth_tile0 = cute.local_tile(idSmooth, (1, smooth_smem_cols), (0, 0))
            else:
                cSmooth_tile0 = cute.local_tile(
                    idSmooth, (tiler_mn[0], smooth_smem_cols), (bidx, 0)
                )
            tScS0 = thr_copy_smooth.partition_S(cSmooth_tile0)
            tSpS0 = _predicate_k(tScS0, limit=shape[1])
        else:
            tSpS0 = None
        if row < shape[0]:
            _copy(tSgS0, tSsS0, pred=tSpS0, is_async=True)
        cute.arch.cp_async_commit_group()  # group for smooth chunk 0

    # Load X directly from GMEM to registers (vectorized, hardware-pipelined).
    if is_even_N:
        if row < shape[0]:
            cute.autovec_copy(tXgX, tXrX)
    else:
        # Zero-init first: predicated load leaves masked elements uninitialized.
        for _zi in cutlass.range_constexpr(num_elems_x):
            tXrX[_zi] = mInput.element_type(0)
        if row < shape[0]:
            _copy(tXgX, tXrX, pred=tXpX)
    x = tXrX.load().to(Float32)

    # ── Hadamard + Working buffer ─────────────────────────────────────
    # _fwht_reg_mut returns the mutable x_buf directly, avoiding a separate x_work copy.
    if const_expr(has_hadamard):
        x_buf = cute.make_rmem_tensor(num_elems_x, Float32)
        x_buf.store(x)
        x_work = _fwht_reg_mut(x_buf, num_elems_x, vecsize, hadamard_k)
    else:
        x_work = cute.make_rmem_tensor(num_elems_x, Float32)
        x_work.store(x)

    # ── Smooth scale + inline max reduction ──────────────────────────────────
    # Fuse smooth application with max reduction to save a separate register pass.
    # Double-buffered: load chunk ci+1 while applying chunk ci.
    # sSmooth0 used for even chunks (0, 2, ...), sSmooth1 for odd chunks (1, 3, ...).
    max_abs_local = Float32(0.0)
    if const_expr(mSmooth is not None):
        for ci in cutlass.range_constexpr(num_chunks):
            # Sync before reusing a buffer that ci-1 has just read.
            # Without this, thread B's cp.async write to sSmooth0 (chunk ci+1) can
            # race with thread A's still-in-progress reads of sSmooth0 from iter ci-1.
            if const_expr(ci > 0):
                cute.arch.barrier()
            # Start loading next chunk ci+1 into the alternate buffer (pipeline stage)
            if const_expr(ci < num_chunks - 1):
                next_ci = const_expr(ci + 1)
                if const_expr(smooth_broadcast):
                    gSmooth_next = cute.local_tile(mSmooth, (1, smooth_smem_cols), (0, next_ci))
                else:
                    gSmooth_next = cute.local_tile(
                        mSmooth, (tiler_mn[0], smooth_smem_cols), (bidx, next_ci)
                    )
                tSgS_next = thr_copy_smooth.partition_S(gSmooth_next)
                # Alternate: even ci loads into sSmooth0 (already done for ci=0), odd into sSmooth1
                # For ci+1 even → sSmooth0, ci+1 odd → sSmooth1
                if const_expr(next_ci % 2 == 0):
                    tSsS_next = thr_copy_smooth.partition_D(sSmooth0)
                else:
                    tSsS_next = thr_copy_smooth.partition_D(sSmooth1)
                if const_expr(smooth_needs_pred):
                    if const_expr(smooth_broadcast):
                        cSmooth_next = cute.local_tile(
                            idSmooth, (1, smooth_smem_cols), (0, next_ci)
                        )
                    else:
                        cSmooth_next = cute.local_tile(
                            idSmooth, (tiler_mn[0], smooth_smem_cols), (bidx, next_ci)
                        )
                    tScS_next = thr_copy_smooth.partition_S(cSmooth_next)
                    tSpS_next = _predicate_k(tScS_next, limit=shape[1])
                else:
                    tSpS_next = None
                if row < shape[0]:
                    _copy(tSgS_next, tSsS_next, pred=tSpS_next, is_async=True)
                cute.arch.cp_async_commit_group()

            # Wait for current chunk (ci) to be ready: wait_group(1) if ci+1 in flight, else (0)
            if const_expr(ci < num_chunks - 1):
                cute.arch.cp_async_wait_group(1)  # allow 1 outstanding (next chunk)
            else:
                cute.arch.cp_async_wait_group(0)  # wait for all (last chunk)
            cute.arch.barrier()

            # Apply smooth chunk ci and inline max reduction (fused to save register pass)
            chunk_col_start = const_expr(ci * smooth_smem_cols)
            si_start = const_expr(ci * elems_per_chunk)

            if const_expr(ci % 2 == 0):  # noqa: SIM108  (const_expr branch, not a ternary)
                sSmooth_cur = sSmooth0
            else:
                sSmooth_cur = sSmooth1

            max_abs_local = _apply_smooth_chunk(
                x_work,
                tXcX_raw,
                sSmooth_cur,
                max_abs_local,
                row_in_tile,
                shape[1],
                si_start,
                chunk_col_start,
                elems_per_chunk,
                smooth_broadcast=smooth_broadcast,
                is_even_N=is_even_N,
            )

    if const_expr(mSmooth is None):
        # No smooth: compute max over x_work directly
        for i in cutlass.range(num_elems_x, unroll_full=True):
            abs_v = cute.arch.fmax(x_work[i], -x_work[i])
            max_abs_local = cute.arch.fmax(max_abs_local, abs_v)

    max_abs = cute.arch.warp_reduction(
        max_abs_local,
        cute.arch.fmax,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )

    if const_expr(warps_per_row > 1):
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        row_idx = warp_idx // warps_per_row
        col_idx = warp_idx % warps_per_row
        if lane_idx == 0:
            reduction_buffer[row_idx, (col_idx, 0)] = max_abs
        cute.arch.barrier()
        block_max = Float32(0.0)
        if lane_idx < warps_per_row:
            block_max = reduction_buffer[row_idx, (lane_idx, 0)]
        max_abs = cute.arch.warp_reduction(block_max, cute.arch.fmax)

    # scale = max_abs / max_val; avoid rcp for constant divisor (compile-time multiply)
    scale = max_abs * Float32(1.0 / max_val)
    # Add tiny epsilon so all-zero rows produce zero output (rather than NaN)
    inv_scale = Float32(max_val) * cute.arch.rcp_approx(max_abs + Float32(1e-30))

    # ── Quantize (multiply by inv_scale + branchless rounding + clamp) ───────
    num_elems = cute.size(tXrX)
    xq_buf = cute.make_rmem_tensor(num_elems, Int8)
    hi = Float32(max_val)
    lo = Float32(-max_val)
    for i in cutlass.range(num_elems, unroll_full=True):
        q = rint_f32(x_work[i] * inv_scale)
        q = fmin_f32(cute.arch.fmax(q, lo), hi)
        xq_buf[i] = q.to(Int32).to(Int8)
    xq_i8 = xq_buf.load()

    # ── Store quantized output ────────────────────────────────────────
    tXgO = thr_copy.partition_D(gO)
    tXrO = cute.make_rmem_tensor_like(tXgO)
    tXrO.store(xq_i8)

    if row < shape[0]:
        _copy(tXrO, tXgO, pred=tXpX)

    # ── Store scale (one per row) ─────────────────────────────────────
    if tXcX[0][1] == 0 and row < shape[0]:
        gScales[row % cute.size(gScales.shape[0]), 0] = scale


# ── FWHT ──────────────────────────────────────────────────────────────────


@cute.jit
def _fwht_reg_mut(x_buf, num_elems, vecsize: cutlass.Constexpr[int], k: cutlass.Constexpr[int]):  # noqa: C901
    """Register-based FWHT, returns mutable register tensor (avoids extra copy to x_work)."""
    log2_k = const_expr(int(math.log2(k)))
    log2_v = const_expr(int(math.log2(vecsize)))
    inv_sqrt_k = const_expr(1.0 / math.sqrt(k))
    num_groups = num_elems // vecsize

    for stage in cutlass.range_constexpr(min(log2_k, log2_v)):
        stride = const_expr(1 << stage)
        buf2 = cute.make_rmem_tensor(num_elems, Float32)
        for gi in cutlass.range(num_groups, unroll_full=True):
            base = gi * vecsize
            for ei in cutlass.range_constexpr(vecsize):
                partner_ei = const_expr(ei ^ stride)
                a = x_buf[base + ei]
                b_val = x_buf[base + partner_ei]
                if const_expr((ei & stride) == 0):
                    buf2[base + ei] = a + b_val
                else:
                    buf2[base + ei] = b_val - a
        x_buf = buf2

    if const_expr(log2_k > log2_v):
        lane_idx = cute.arch.lane_idx()
        for stage in cutlass.range_constexpr(log2_k - log2_v):
            xor_mask = const_expr(1 << stage)
            is_upper = Boolean(lane_idx & xor_mask)
            buf2 = cute.make_rmem_tensor(num_elems, Float32)
            for i in cutlass.range(num_elems, unroll_full=True):
                a = x_buf[i]
                b_val = cute.arch.shuffle_sync_bfly(a, offset=xor_mask)
                buf2[i] = (b_val - a) if is_upper else (a + b_val)
            x_buf = buf2

    kblock_threads = const_expr(k // vecsize)
    lane_idx_n = cute.arch.lane_idx()
    is_first_in_kblock = (lane_idx_n % kblock_threads) == 0
    for i in cutlass.range(num_elems, unroll_full=True):
        x_buf[i] = x_buf[i] * Float32(inv_sqrt_k)
    if is_first_in_kblock:
        for gi in cutlass.range(num_groups, unroll_full=True):
            x_buf[gi * vecsize] = -x_buf[gi * vecsize]

    # Return the mutable register tensor directly (saves creating a separate x_work buffer)
    return x_buf


# ── Compilation cache & public API ────────────────────────────────────────

DEFAULT_HADAMARD_BLOCK_SIZE: Final[int] = 16

_cache = {}


def quantize(
    x: torch.Tensor,
    xq: torch.Tensor,
    xq_scales: torch.Tensor,
    smooth_scale: torch.Tensor | None = None,
    max_val: int = 63,
    block_size: int = DEFAULT_HADAMARD_BLOCK_SIZE,
) -> None:
    """Fused Hadamard + smooth-scale + symmetric int8 quantization.

    Args:
        x: Input tensor (M, N), fp16 or bf16. Must be contiguous.
        xq: Pre-allocated output (M, N), int8.
        xq_scales: Pre-allocated scales (M, 1), fp32.
        smooth_scale: Optional smooth scale: (1, N) for broadcast across rows,
            or (M, N) for per-token.
        max_val: Maximum absolute quantized value (default 63).
        block_size: Hadamard block size (power of 2); 0 disables the Hadamard
            transform.
    """
    M, N = x.shape
    device = x.device

    has_smooth = smooth_scale is not None
    has_hadamard = block_size > 0
    is_smooth_broadcast = has_smooth and smooth_scale.shape[0] == 1

    x_dtype = TORCH_TO_CUTLASS[x.dtype]

    if has_hadamard:
        # _fwht_reg_mut() derives log2_k via int(math.log2(block_size)); a
        # non-power-of-2 would silently truncate and run the wrong transform.
        if block_size < 2 or (block_size & (block_size - 1)) != 0:
            raise ValueError(f"block_size must be a power of 2 ≥ 2, got {block_size}")
        if N % block_size != 0:
            raise ValueError(f"N (={N}) must be divisible by block_size (={block_size})")
        # A Hadamard block spans block_size // vecsize lanes via the cross-warp
        # butterfly; that span must fit within a single warp (32 lanes).
        vecsize = math.gcd(N, 128 // x_dtype.width)
        max_block_size = cute.arch.WARP_SIZE * vecsize
        if block_size > max_block_size:
            raise ValueError(
                f"block_size must be ≤ {max_block_size} for N={N} and {x.dtype}, got {block_size}"
            )
    hadamard_k = block_size if has_hadamard else 0
    cache_key = (N, x.dtype, has_smooth, is_smooth_broadcast, has_hadamard, hadamard_k, max_val)

    if cache_key not in _cache:
        batch_sym = cute.sym_int()
        x_div = math.gcd(N, 128 // x_dtype.width)
        s_div = math.gcd(N, 128 // 32)
        o_div = math.gcd(N, 128 // 8)

        x_fake = _make_fake(x_dtype, (batch_sym, N), x_div)
        if has_smooth:
            if is_smooth_broadcast:
                smooth_fake = _make_fake(Float32, (1, N), s_div)
            else:
                smooth_fake = _make_fake(Float32, (batch_sym, N), s_div)
        else:
            smooth_fake = None
        o_fake = _make_fake(Int8, (batch_sym, N), o_div)
        scales_fake = _make_fake(Float32, (batch_sym, 1))
        fake_stream = cute.runtime.make_fake_stream()

        _cache[cache_key] = cute.compile(
            hadamard_quant_jit,
            x_fake,
            smooth_fake,
            o_fake,
            scales_fake,
            fake_stream,
            N=N,
            has_smooth=has_smooth,
            smooth_broadcast=is_smooth_broadcast,
            has_hadamard=has_hadamard,
            hadamard_k=hadamard_k,
            max_val=max_val,
            options="--enable-tvm-ffi",
        )

    _cache[cache_key](x, smooth_scale, xq, xq_scales, get_stream(device.index))
