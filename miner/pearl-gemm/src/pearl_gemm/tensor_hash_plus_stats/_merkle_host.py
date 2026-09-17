"""Host-side tensor-hash pipeline (CuTe DSL).

One compiled launcher runs the whole pipeline on the caller's stream:

1. ``merkle_tree_roots_kernel``  - per-CTA roots into the ``roots`` scratchpad
   (on the activation path each consumer also reduces the commit-stats
   partials of the chunk it hashes, out of the same staged codes)
2. ``compute_blake_mt_kernel``   - Merkle tree over those roots
3. ``reduce_roots_kernel``       - final reduction (when more than one MT block)

The 32-byte digest is then copied from ``roots`` into ``out`` with an async
device-to-device copy on the current stream.

``tensor_hash_pair_jit`` runs the same three stages over *both* activation
blobs at once: each stage is one launch whose grid is partitioned by
``blockIdx`` between the blobs, so the chain's payload-independent latency
floor is paid once for the pair instead of once per blob. Each partition
carries only its own blob's byte length, tree shape and ``roots`` slice, so
both digests are identical to hashing the blobs separately.

Compiled variants are cached per compile-time constant (``_VariantKey`` /
``_PairVariantKey``); the data lengths stay dynamic.

The digest is a pure function of the padded byte stream and the key. When a
``stats`` tensor is supplied (the activation-commitment path, where ``data``
is the codes blob and ``scales`` its companion blob), the roots kernel
additionally reduces one fp32 ``(sumsq, absmax)`` partial per 512-element
block from the codes and scales (factored, no dequantization; see
``_block_stats.py``) -- the partials ``noisy_quant`` combines into
``alpha``/``beta``.
"""

from typing import NamedTuple

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cute.nvgpu import cpasync

from .._utils._compile import make_fake_stream, make_fake_tensor, single_flight_compile
from .._utils._stream import get_stream
from . import _blake3
from ._block_stats import _STATS_CODE_BYTES
from ._compute_blake_mt_kernel import compute_blake_mt_kernel, compute_blake_mt_pair_kernel
from ._merkle_tree_roots_kernel import (
    _NUM_PRODUCER_THREADS,
    SUPPORTED_CHUNKS_PER_THREAD,
    _derived_config,
    _make_smem_layouts,
    _num_roots_blocks,
    _stats_smem_floats,
    merkle_tree_roots_kernel,
    merkle_tree_roots_pair_kernel,
    merkle_tree_roots_sync_kernel,
    merkle_tree_roots_sync_pair_kernel,
)
from ._reduce_roots_kernel import reduce_roots_kernel

DEFAULT_THREAD_LOAD_SIZE = 128
SUPPORTED_THREADS_PER_BLOCK = (128, 256, 512)
SUPPORTED_NUM_STAGES = (2, 3, 4)
SUPPORTED_LEAVES_PER_MT_BLOCK = (256, 512, 1024)
SUPPORTED_THREAD_LOAD_SIZES = (64, 128, 256, 512)
# Supported Merkle leaf sizes: any whole multiple of the 64-byte message
# block within these bounds.
MIN_CHUNK_SIZE = 64
MAX_CHUNK_SIZE = 4096
_UINT32_MAX = 2**32 - 1


def supported_chunk_sizes() -> tuple[int, ...]:
    """Every ``chunk_size`` ``_validate_hash_tunables`` accepts."""
    return tuple(range(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE + 1, _blake3.MSG_BLOCK_SIZE))


def get_required_scratchpad_bytes(
    matrix_bytes: int, threads_per_block: int = 128, chunk_size: int = _blake3.CHUNK_SIZE
) -> int:
    """Scratchpad ("roots") bytes needed: one 32-byte root slot per data block
    (the sizing formula ``tensor_hash_launch`` validates its ``roots`` against)."""
    bytes_per_block = threads_per_block * chunk_size
    required_blocks = (matrix_bytes + bytes_per_block - 1) // bytes_per_block
    return required_blocks * _blake3.CHAINING_VALUE_SIZE


# CTA shared-memory budget the tune space and record legality test against:
# headroom under the SM100 opt-in maximum for what the estimate below does
# not model (barriers, TMA descriptors, alignment padding).
_MAX_TENSOR_HASH_SMEM = 220 * 1024


def tensor_hash_smem_fits(
    threads_per_block: int,
    num_stages: int,
    thread_load_size: int,
    stats_chunk: int | None = None,
    *,
    sync_loads: bool = False,
) -> bool:
    """Estimated CTA smem fit, shared by tune-space generation and persisted
    record legality so the two can never disagree on "launchable".

    The TMA staging ring (dual-pipeline at 512 threads; absent on the sync
    path) plus the leaves buffer, plus -- when ``stats_chunk`` is a sub-block
    leaf of a plus-stats candidate -- the fp32 stats staging region
    (whole-block leaves reduce stats in registers and stage nothing).
    ``stats_chunk=None`` is the raw hash.
    """
    ring_bytes = (
        0
        if sync_loads
        else (2 if threads_per_block == 512 else 1)
        * min(threads_per_block, 256)
        * thread_load_size
        * num_stages
    )
    stats_bytes = (
        4 * _stats_smem_floats(threads_per_block, stats_chunk, 1, True) if stats_chunk else 0
    )
    leaves_bytes = threads_per_block * _blake3.CHAINING_VALUE_SIZE
    return ring_bytes + leaves_bytes + stats_bytes <= _MAX_TENSOR_HASH_SMEM


def _stage1_geometry(threads_per_block, num_stages, thread_load_size, chunk_size):
    """Load geometry and smem layouts shared by every TMA stage-1 launch
    (single and pair; trace-time only)."""
    _, tma_threads, num_words_per_load, _, _ = _derived_config(
        threads_per_block, thread_load_size, chunk_size
    )
    sa_layout_staged, leaves_layout = _make_smem_layouts(
        threads_per_block, num_stages, thread_load_size, chunk_size
    )
    return tma_threads, num_words_per_load, sa_layout_staged, leaves_layout


@cute.jit
def _chunk_tma_atom(
    mData: cute.Tensor,
    data_len: Int64,
    sa_layout_staged: cute.ComposedLayout,
    tma_threads: cutlass.Constexpr[int],
    num_words_per_load: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
):
    """TMA atom and coordinate tensor over a blob's fully-backed leaf groups.

    A row is one leaf group (``chunks_per_thread`` consecutive chunks, so the
    coarsened kernel streams a thread's k chunks through the same tiling).
    The descriptor covers exactly the fully-backed groups (floor); the
    trailing partial chunk is loaded from gmem by the consumers instead.
    cuTensorMapEncodeTiled requires extents >= 1: when there is no full group
    the producer issues no TMA loads, so the descriptor is never dereferenced.
    """
    group_bytes = chunks_per_thread * chunk_size
    num_full_groups = data_len // group_bytes
    tma_group_rows = cutlass.max(num_full_groups, 1)
    mA = cute.make_tensor(
        cute.recast_ptr(mData.iterator, dtype=cutlass.Uint32),
        cute.make_layout((tma_group_rows, group_bytes // 4), stride=(group_bytes // 4, 1)),
    )
    return cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mA,
        cute.slice_(sa_layout_staged, (None, None, 0)),
        (tma_threads, num_words_per_load),
    )


@cute.jit
def tensor_hash_jit(
    mData: cute.Tensor,
    mKey: cute.Tensor,
    mRoots: cute.Tensor,
    mStats: cute.Tensor,
    mScales: cute.Tensor,
    mCounter: cute.Tensor,
    data_len: Int64,
    stream: cuda_drv.CUstream,
    threads_per_block: cutlass.Constexpr[int],
    num_stages: cutlass.Constexpr[int],
    leaves_per_mt_block: cutlass.Constexpr[int],
    thread_load_size: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
    mad_rot: cutlass.Constexpr[bool],
    sync_loads: cutlass.Constexpr[bool],
    apply_root: cutlass.Constexpr[bool],
    has_full_chunks: cutlass.Constexpr[bool],
    single_mt_block: cutlass.Constexpr[bool],
    fused_tail: cutlass.Constexpr[bool],
    with_stats: cutlass.Constexpr[bool],
):
    # We split data into chunks of chunk_size bytes, chunks into leaf groups
    # of chunks_per_thread (thread coarsening; 1 group == 1 chunk at k=1),
    # and groups into blocks (CTAs) of threads_per_block groups. Chunk/block
    # counts fit in 32 bits (data_len is at most 2^32 bytes); byte offsets
    # stay 64-bit.
    num_blocks = _num_roots_blocks(data_len, threads_per_block, chunk_size, chunks_per_thread)

    # 1. Compute the Merkle roots of each block's chunks.
    if cutlass.const_expr(sync_loads):
        # No-TMA/no-pipeline path: consumer threads only, direct
        # vectorized gmem loads, same digest (and same fused stats).
        merkle_tree_roots_sync_kernel(
            mData,
            mRoots,
            mKey,
            mStats,
            mScales,
            data_len,
            threads_per_block,
            chunk_size,
            chunks_per_thread,
            apply_root,
            with_stats,
        ).launch(
            grid=[num_blocks, 1, 1],
            block=[threads_per_block, 1, 1],
            stream=stream,
        )
    else:
        tma_threads, num_words_per_load, sa_layout_staged, leaves_layout = _stage1_geometry(
            threads_per_block, num_stages, thread_load_size, chunk_size
        )
        tma_atom, mA_coord = _chunk_tma_atom(
            mData,
            data_len,
            sa_layout_staged,
            tma_threads,
            num_words_per_load,
            chunk_size,
            chunks_per_thread,
        )
        merkle_tree_roots_kernel(
            tma_atom,
            mA_coord,
            mData,
            mRoots,
            mKey,
            mStats,
            mScales,
            mCounter,
            data_len,
            sa_layout_staged,
            leaves_layout,
            threads_per_block,
            num_stages,
            thread_load_size,
            chunk_size,
            chunks_per_thread,
            mad_rot,
            apply_root,
            has_full_chunks,
            fused_tail,
            with_stats,
        ).launch(
            grid=[num_blocks, 1, 1],
            block=[_NUM_PRODUCER_THREADS + threads_per_block, 1, 1],
            stream=stream,
        )

    # 2. Compute MT as per BLAKE structure on global: each CTA reduces a
    # Merkle tree of leaves_per_mt_block leaves. With the fused tail the last
    # roots-kernel CTA already did this inline (and stage 3 never applies:
    # fused_tail implies single_mt_block).
    if cutlass.const_expr(not fused_tail):
        num_blocks_for_mt = cute.ceil_div(num_blocks, leaves_per_mt_block)
        compute_blake_mt_kernel(
            mRoots, mKey, num_blocks, leaves_per_mt_block, single_mt_block
        ).launch(
            grid=[num_blocks_for_mt, 1, 1],
            block=[leaves_per_mt_block, 1, 1],
            stream=stream,
        )

        # 3. Further aggregation of roots if we have multiple MT blocks. The
        # grid is one CTA, so the kernel's second partition is never selected.
        if cutlass.const_expr(not single_mt_block):
            reduce_roots_kernel(
                mRoots,
                mKey,
                Int32(0),
                num_blocks_for_mt,
                Int32(0),
                num_blocks_for_mt,
                threads_per_block,
            ).launch(
                grid=[1, 1, 1],
                block=[threads_per_block, 1, 1],
                stream=stream,
            )


@cute.jit
def tensor_hash_pair_jit(
    mCodes: cute.Tensor,
    mScales: cute.Tensor,
    mKey: cute.Tensor,
    mRoots: cute.Tensor,
    mStats: cute.Tensor,
    codes_len: Int64,
    scales_len: Int64,
    stream: cuda_drv.CUstream,
    threads_per_block: cutlass.Constexpr[int],
    num_stages: cutlass.Constexpr[int],
    leaves_per_mt_block: cutlass.Constexpr[int],
    thread_load_size: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
    codes_mad_rot: cutlass.Constexpr[bool],
    scales_mad_rot: cutlass.Constexpr[bool],
    sync_loads: cutlass.Constexpr[bool],
    codes_tree: cutlass.Constexpr,
    scales_tree: cutlass.Constexpr,
    with_stats: cutlass.Constexpr[bool],
):
    """Launch the three stages once each over a grid covering both blobs.

    ``mRoots`` is the whole workspace: the codes tree owns its leading
    ``codes_blocks`` 32-byte slots and the scales tree the following
    ``scales_blocks``, so each blob's digest ends up at the start of its own
    slice. ``codes_tree`` / ``scales_tree`` are that blob's ``(apply_root,
    has_full_chunks, single_mt_block)`` flags, which its byte length fixes.
    """
    codes_blocks = _num_roots_blocks(codes_len, threads_per_block, chunk_size, chunks_per_thread)
    scales_blocks = _num_roots_blocks(scales_len, threads_per_block, chunk_size, chunks_per_thread)

    # 1. Both blobs' per-CTA roots, the codes blob first in the grid.
    if cutlass.const_expr(sync_loads):
        merkle_tree_roots_sync_pair_kernel(
            mCodes,
            mScales,
            mRoots,
            mKey,
            mStats,
            codes_len,
            scales_len,
            threads_per_block,
            chunk_size,
            chunks_per_thread,
            codes_tree,
            scales_tree,
            with_stats,
        ).launch(
            grid=[codes_blocks + scales_blocks, 1, 1],
            block=[threads_per_block, 1, 1],
            stream=stream,
        )
    else:
        tma_threads, num_words_per_load, sa_layout_staged, leaves_layout = _stage1_geometry(
            threads_per_block, num_stages, thread_load_size, chunk_size
        )
        tma_atom_codes, coord_codes = _chunk_tma_atom(
            mCodes,
            codes_len,
            sa_layout_staged,
            tma_threads,
            num_words_per_load,
            chunk_size,
            chunks_per_thread,
        )
        tma_atom_scales, coord_scales = _chunk_tma_atom(
            mScales,
            scales_len,
            sa_layout_staged,
            tma_threads,
            num_words_per_load,
            chunk_size,
            chunks_per_thread,
        )
        merkle_tree_roots_pair_kernel(
            tma_atom_codes,
            coord_codes,
            mCodes,
            tma_atom_scales,
            coord_scales,
            mScales,
            mRoots,
            mKey,
            mStats,
            codes_len,
            scales_len,
            sa_layout_staged,
            leaves_layout,
            threads_per_block,
            num_stages,
            thread_load_size,
            chunk_size,
            chunks_per_thread,
            codes_mad_rot,
            scales_mad_rot,
            codes_tree,
            scales_tree,
            with_stats,
        ).launch(
            grid=[codes_blocks + scales_blocks, 1, 1],
            block=[_NUM_PRODUCER_THREADS + threads_per_block, 1, 1],
            stream=stream,
        )

    # 2. Both blobs' groups of leaves_per_mt_block roots, in the same order.
    codes_groups = cute.ceil_div(codes_blocks, leaves_per_mt_block)
    scales_groups = cute.ceil_div(scales_blocks, leaves_per_mt_block)
    compute_blake_mt_pair_kernel(
        mRoots,
        mKey,
        codes_blocks,
        scales_blocks,
        leaves_per_mt_block,
        codes_tree[2],
        scales_tree[2],
    ).launch(
        grid=[codes_groups + scales_groups, 1, 1],
        block=[leaves_per_mt_block, 1, 1],
        stream=stream,
    )

    # 3. One CTA per blob whenever either blob's stage 2 left more than one
    # group; the other blob then reduces a single leaf, which is a no-op copy
    # of its digest onto itself, so no second launch is needed for it.
    if cutlass.const_expr(not codes_tree[2] or not scales_tree[2]):
        reduce_roots_kernel(
            mRoots, mKey, Int32(0), codes_groups, codes_blocks, scales_groups, threads_per_block
        ).launch(
            grid=[2, 1, 1],
            block=[threads_per_block, 1, 1],
            stream=stream,
        )


class _VariantKey(NamedTuple):
    """Compile-cache key: the compile-time constants of ``tensor_hash_jit``."""

    threads_per_block: int
    num_stages: int
    leaves_per_mt_block: int
    thread_load_size: int
    chunk_size: int
    chunks_per_thread: int
    mad_rot: bool
    sync_loads: bool
    apply_root: bool
    has_full_chunks: bool
    single_mt_block: bool
    fused_tail: bool
    with_stats: bool


class _PairVariantKey(NamedTuple):
    """Compile-cache key: the compile-time constants of ``tensor_hash_pair_jit``."""

    threads_per_block: int
    num_stages: int
    leaves_per_mt_block: int
    thread_load_size: int
    chunk_size: int
    chunks_per_thread: int
    codes_mad_rot: bool
    scales_mad_rot: bool
    sync_loads: bool
    codes_tree: tuple
    scales_tree: tuple
    with_stats: bool


_counter_cache = {}
# Slots per fused-tail counter allocation. One slot is handed to each stream
# and never reclaimed, so the pool grows a chunk at a time; sized to torch's
# per-device stream pool, one chunk covers every handle a process sees in
# practice. Chunks exist because a stream first seen under CUDA-graph capture
# has to find a free slot without allocating.
_COUNTER_SLOTS_PER_CHUNK = 32


def _fused_tail_counter(device):
    """Return the fused-tail arrival counter for the current stream.

    One slot per stream, so concurrent launches never share a ticket; the
    closing CTA resets its slot, so the next launch on that stream observes 0
    (rationale for owning this state at all is in the docs README).
    """
    entry = _counter_cache.get(device)
    if entry is None:
        entry = _counter_cache[device] = ({}, [])
    slots, free = entry
    # The handle the launch itself binds. Reused handles are safe: a destroyed
    # stream cannot retain tickets, and its slot is back to 0.
    raw_stream = torch.cuda.current_stream(device).cuda_stream
    counter = slots.get(raw_stream)
    if counter is None:
        if not free:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "a fused-tail tensor_hash on a stream not seen before must happen "
                    "outside CUDA-graph capture (its arrival counter would be "
                    "allocated from the graph pool)"
                )
            chunk = torch.zeros(_COUNTER_SLOTS_PER_CHUNK, dtype=torch.int32, device=device)
            free.extend(chunk[i : i + 1] for i in range(_COUNTER_SLOTS_PER_CHUNK))
        counter = slots[raw_stream] = free.pop()
    return counter


def _fake_blob():
    """A fake flat, 16-byte-aligned byte payload of dynamic length."""
    return make_fake_tensor(cutlass.Uint8, (cute.sym_int64(),), leading_dim=0, divisibility=16)


def _fake_key_roots_stats():
    """Fake key, roots workspace and stats operands, shared by both launchers."""
    return (
        make_fake_tensor(cutlass.Int32, (8,), leading_dim=0, divisibility=1),
        make_fake_tensor(cutlass.Int32, (cute.sym_int64(),), leading_dim=0, divisibility=1),
        make_fake_tensor(cutlass.Float32, (cute.sym_int64(),), leading_dim=0, divisibility=2),
    )


@single_flight_compile
def _compile_variant(
    threads_per_block,
    num_stages,
    leaves_per_mt_block,
    thread_load_size,
    chunk_size,
    chunks_per_thread,
    mad_rot,
    sync_loads,
    apply_root,
    has_full_chunks,
    single_mt_block,
    fused_tail,
    with_stats,
):
    key_fake, roots_fake, stats_fake = _fake_key_roots_stats()
    counter_fake = make_fake_tensor(cutlass.Int32, (1,), leading_dim=0, divisibility=1)
    return cute.compile(
        tensor_hash_jit,
        _fake_blob(),
        key_fake,
        roots_fake,
        stats_fake,
        _fake_blob(),
        counter_fake,
        Int64(0),
        make_fake_stream(),
        threads_per_block=threads_per_block,
        num_stages=num_stages,
        leaves_per_mt_block=leaves_per_mt_block,
        thread_load_size=thread_load_size,
        chunk_size=chunk_size,
        chunks_per_thread=chunks_per_thread,
        mad_rot=mad_rot,
        sync_loads=sync_loads,
        apply_root=apply_root,
        has_full_chunks=has_full_chunks,
        single_mt_block=single_mt_block,
        fused_tail=fused_tail,
        with_stats=with_stats,
        options="--enable-tvm-ffi",
    )


@single_flight_compile
def _compile_pair_variant(
    threads_per_block,
    num_stages,
    leaves_per_mt_block,
    thread_load_size,
    chunk_size,
    chunks_per_thread,
    codes_mad_rot,
    scales_mad_rot,
    sync_loads,
    codes_tree,
    scales_tree,
    with_stats,
):
    key_fake, roots_fake, stats_fake = _fake_key_roots_stats()
    return cute.compile(
        tensor_hash_pair_jit,
        _fake_blob(),
        _fake_blob(),
        key_fake,
        roots_fake,
        stats_fake,
        Int64(0),
        Int64(0),
        make_fake_stream(),
        threads_per_block=threads_per_block,
        num_stages=num_stages,
        leaves_per_mt_block=leaves_per_mt_block,
        thread_load_size=thread_load_size,
        chunk_size=chunk_size,
        chunks_per_thread=chunks_per_thread,
        codes_mad_rot=codes_mad_rot,
        scales_mad_rot=scales_mad_rot,
        sync_loads=sync_loads,
        codes_tree=codes_tree,
        scales_tree=scales_tree,
        with_stats=with_stats,
        options="--enable-tvm-ffi",
    )


def _validate_hash_tunables(
    threads_per_block,
    num_stages,
    leaves_per_mt_block,
    thread_load_size,
    chunk_size,
    chunks_per_thread=1,
) -> None:
    """Reject compile-time tunables the Merkle kernels cannot express."""
    for name, value in (
        ("threads_per_block", threads_per_block),
        ("num_stages", num_stages),
        ("leaves_per_mt_block", leaves_per_mt_block),
        ("thread_load_size", thread_load_size),
        ("chunk_size", chunk_size),
        ("chunks_per_thread", chunks_per_thread),
    ):
        # Exact type: integral floats can pass value checks; bool subclasses int.
        if type(value) is not int:
            raise ValueError(f"{name} must be an int")
    if threads_per_block not in SUPPORTED_THREADS_PER_BLOCK:
        raise ValueError(f"threads_per_block must be one of {SUPPORTED_THREADS_PER_BLOCK}")
    if num_stages not in SUPPORTED_NUM_STAGES:
        raise ValueError(f"num_stages must be one of {SUPPORTED_NUM_STAGES}")
    if leaves_per_mt_block not in SUPPORTED_LEAVES_PER_MT_BLOCK:
        raise ValueError(f"leaves_per_mt_block must be one of {SUPPORTED_LEAVES_PER_MT_BLOCK}")
    if thread_load_size not in SUPPORTED_THREAD_LOAD_SIZES:
        raise ValueError(f"thread_load_size must be one of {SUPPORTED_THREAD_LOAD_SIZES}")
    if not (
        MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE and chunk_size % _blake3.MSG_BLOCK_SIZE == 0
    ):
        raise ValueError(
            f"chunk_size must be a multiple of {_blake3.MSG_BLOCK_SIZE} in "
            f"[{MIN_CHUNK_SIZE}, {MAX_CHUNK_SIZE}]"
        )
    if chunk_size % thread_load_size:
        raise ValueError("thread_load_size must divide chunk_size")
    if chunks_per_thread not in SUPPORTED_CHUNKS_PER_THREAD:
        raise ValueError(f"chunks_per_thread must be one of {SUPPORTED_CHUNKS_PER_THREAD}")


def _effective_chunks_per_thread(chunks_per_thread, blobs_bytes, chunk_size, with_stats) -> int:
    """Thread-coarsening activation gate (silent fallback to k=1).

    Coarsening requires every hashed blob to be an exact multiple of
    ``k * chunk_size`` bytes (no partial chunk, no ragged group) with at
    least 2 leaf groups (the private fold must never be the message root).
    Stats may ride a coarsened hash at whole-stats-block leaves (each
    sub-chunk's carry drains at its own block boundaries); sub-block leaves
    stay at k=1, where the smem staging region is one pair per chunk message
    block. The digest is bitwise identical either way, so a tuned config's
    k can safely apply to just the payloads that support it.
    """
    if chunks_per_thread == 1 or (with_stats and chunk_size % _STATS_CODE_BYTES):
        return 1
    group_bytes = chunks_per_thread * chunk_size
    for num_bytes in blobs_bytes:
        if num_bytes % group_bytes or num_bytes < 2 * group_bytes:
            return 1
    return chunks_per_thread


def _validate_buffer_dtypes(data_bytes, key, out, roots) -> None:
    """Reject payloads and digest buffers the kernels cannot express."""
    if not 0 < data_bytes <= _UINT32_MAX:
        raise ValueError(f"data must contain between 1 and {_UINT32_MAX} bytes, got {data_bytes}")
    if key.dtype != torch.uint8 or key.numel() != _blake3.KEY_SIZE:
        raise ValueError(f"key must contain exactly {_blake3.KEY_SIZE} uint8 elements")
    if out.dtype != torch.uint8 or out.numel() != _blake3.CHAINING_VALUE_SIZE:
        raise ValueError(f"out must contain exactly {_blake3.CHAINING_VALUE_SIZE} uint8 elements")
    if roots.dtype != torch.uint8:
        raise ValueError("roots must be uint8")
    if roots.numel() % _blake3.CHAINING_VALUE_SIZE:
        raise ValueError(f"roots must have a multiple of {_blake3.CHAINING_VALUE_SIZE} bytes")


def _validate_buffer_alignment(data, key, roots) -> None:
    """Base-address alignment required by TMA and the vectorized accesses."""
    if data.data_ptr() % 16:
        raise ValueError("data must be 16-byte aligned (TMA requirement)")
    if key.data_ptr() % 4:
        raise ValueError("key must be 4-byte aligned")
    if roots.data_ptr() % 4:
        raise ValueError("roots must be 4-byte aligned")


def _require_tensors(*named) -> None:
    """Reject non-tensor operands, ahead of any attribute access on them."""
    for name, tensor in named:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")


def _validate_buffer_shapes(data, key, out, roots) -> int:
    """Check device, contiguity, dtype, and alignment; return the data bytes."""
    tensors = [("data", data), ("key", key), ("out", out), ("roots", roots)]
    _require_tensors(*tensors)
    for name, tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != data.device:
            raise ValueError(f"{name} must be on the same device as data")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    data_bytes = data.numel() * data.element_size()
    _validate_buffer_dtypes(data_bytes, key, out, roots)
    _validate_buffer_alignment(data, key, roots)
    return data_bytes


def _validate_tensor_hash_buffers(
    data,
    key,
    out,
    roots,
    threads_per_block,
    num_stages,
    leaves_per_mt_block,
    thread_load_size,
    chunk_size,
    chunks_per_thread=1,
):
    data_bytes = _validate_buffer_shapes(data, key, out, roots)
    _validate_hash_tunables(
        threads_per_block,
        num_stages,
        leaves_per_mt_block,
        thread_load_size,
        chunk_size,
        chunks_per_thread,
    )

    num_chunks = (data_bytes + chunk_size - 1) // chunk_size
    # Exact when chunks_per_thread > 1 (the activation gate requires it).
    num_groups = num_chunks // chunks_per_thread
    num_blocks = (num_groups + threads_per_block - 1) // threads_per_block
    # One root slot per k-chunk leaf group block (== the caller-facing k=1
    # sizing when coarsening is off; strictly less otherwise, so a workspace
    # sized by the public formula always fits).
    required = get_required_scratchpad_bytes(
        data_bytes, threads_per_block, chunk_size * chunks_per_thread
    )
    if roots.numel() < required:
        raise ValueError(
            f"roots must have at least {num_blocks} * {_blake3.CHAINING_VALUE_SIZE} bytes"
        )

    num_blocks_for_mt = (num_blocks + leaves_per_mt_block - 1) // leaves_per_mt_block
    if num_blocks_for_mt > threads_per_block:
        # reduce_roots_kernel is a single CTA: one thread (and one smem slot)
        # per remaining root. At chunk_size >= 1024 the 2^32-byte input cap
        # means this cannot be violated; smaller leaves can exceed it.
        raise ValueError(
            f"unsupported config: final reduction has {num_blocks_for_mt} roots for "
            f"{threads_per_block} threads (chunk_size={chunk_size}, {data_bytes} bytes); "
            "raise threads_per_block/leaves_per_mt_block or chunk_size"
        )
    return data_bytes, num_blocks, num_blocks_for_mt


def _validate_stats_operands(payload_name, payload_bytes, scales, stats, device) -> None:
    """The fused-stats operand contract shared by both launchers.

    The hashed payloads stay dtype-free byte blobs; this pins what the stats
    reduction assumes on top of them: whole 512-byte code blocks, a
    packed-BF16 scales companion on the hashed data's device (BF16, or a
    uint8 view of already-packed bytes -- any other dtype with the right
    byte count would be reinterpreted, corrupting every partial), and one
    fp32 ``(sumsq, absmax)`` pair per block in ``stats``, 16-byte aligned
    for the prep consumers' LDG.128 fetches.
    """
    if scales is None:
        raise ValueError("scales is required when stats is requested")
    _require_tensors(("scales", scales), ("stats", stats))
    for name, tensor in (("scales", scales), ("stats", stats)):
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"{name} must be on the same CUDA device as {payload_name}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    checks = (
        (
            payload_bytes % _STATS_CODE_BYTES == 0,
            f"{payload_name} must be a whole number of {_STATS_CODE_BYTES}-byte blocks",
        ),
        (
            scales.dtype in (torch.bfloat16, torch.uint8),
            "scales must be packed BF16 (bfloat16, or a uint8 byte view)",
        ),
        (
            scales.numel() * scales.element_size() * 4 == payload_bytes,
            "scales must hold one BF16 scale per 8 codes",
        ),
        (scales.data_ptr() % 16 == 0, "scales must be 16-byte aligned"),
        (stats.dtype == torch.float32, "stats must be float32"),
        (
            stats.numel() == 2 * (payload_bytes // _STATS_CODE_BYTES),
            "stats must hold one (sumsq, absmax) pair per stats block",
        ),
        (stats.data_ptr() % 16 == 0, "stats must be 16-byte aligned"),
    )
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def _validate_stats_compatible(chunk_size) -> None:
    """Merkle leaves the stats fusion supports (both load paths).

    A stats block is ``_STATS_CODE_BYTES`` of codes. Leaves covering a whole
    number of blocks reduce each block inside the thread that hashes it (the
    per-thread carry); sub-block leaves stage per-message-block partials and
    fold them across threads after the CTA barrier. A leaf above one block
    that is not a whole multiple would split blocks across threads at
    chunk-dependent ragged offsets, which neither reduction expresses.
    """
    if chunk_size > _STATS_CODE_BYTES and chunk_size % _STATS_CODE_BYTES:
        raise ValueError(
            f"stats requires chunk_size to be a whole multiple of {_STATS_CODE_BYTES} "
            f"or smaller than it"
        )


def tensor_hash_launch(
    data,
    key,
    out,
    roots,
    *,
    threads_per_block,
    num_stages,
    leaves_per_mt_block,
    thread_load_size=DEFAULT_THREAD_LOAD_SIZE,
    chunk_size=_blake3.CHUNK_SIZE,
    chunks_per_thread=1,
    mad_rot=False,
    sync_loads=False,
    stats=None,
    scales=None,
):
    """Keyed-BLAKE3 tensor hash via the CuTe DSL pipeline.

    Validation matches the production commitment pipeline (digests are
    bit-identical). When ``stats`` is given, ``data`` must be the codes blob
    (a whole number of 512-byte blocks), ``scales`` its companion blob, and
    ``stats`` receives one fp32 ``(sumsq, absmax)`` pair per block.
    """
    # Ahead of the coarsening gate below, which reads the payload length and
    # divides by the tunables before the buffer validation would reach them.
    _require_tensors(("data", data))
    _validate_hash_tunables(
        threads_per_block,
        num_stages,
        leaves_per_mt_block,
        thread_load_size,
        chunk_size,
        chunks_per_thread,
    )
    with_stats = stats is not None
    chunks_per_thread = _effective_chunks_per_thread(
        chunks_per_thread, (data.numel() * data.element_size(),), chunk_size, with_stats
    )
    data_bytes, num_blocks, num_blocks_for_mt = _validate_tensor_hash_buffers(
        data,
        key,
        out,
        roots,
        threads_per_block,
        num_stages,
        leaves_per_mt_block,
        thread_load_size,
        chunk_size,
        chunks_per_thread,
    )
    apply_root = num_blocks == 1
    has_full_chunks = data_bytes >= chunk_size
    single_mt_block = num_blocks_for_mt == 1
    # Fused stage-2 tail: the last-finishing CTA reduces the roots inline and
    # the stage-2 launch is skipped. At num_blocks == 1 the in-kernel tail is
    # a no-op (apply_root already finalized the digest at roots[0]; stage 2
    # was a pass-through) so fusing just drops a dead kernel launch.
    fused_tail = single_mt_block and num_blocks <= 2 and not sync_loads
    mad_rot = _mad_rot_dispatch(mad_rot, sync_loads, data_bytes)
    if with_stats:
        _validate_stats_compatible(chunk_size)
        _validate_stats_operands("data", data_bytes, scales, stats, data.device)

    cache_key = _VariantKey(
        threads_per_block,
        num_stages,
        leaves_per_mt_block,
        thread_load_size,
        chunk_size,
        chunks_per_thread,
        mad_rot,
        sync_loads,
        apply_root,
        has_full_chunks,
        single_mt_block,
        fused_tail,
        with_stats,
    )

    compiled = _compile_variant(*cache_key)
    compiled(
        data.view(torch.uint8).reshape(-1),
        key.view(torch.int32),
        roots.view(torch.int32),
        # The stats=None variants never touch mStats / mScales (with_stats is
        # a compile constant); the roots scratchpad stands in as a harmless
        # dummy for both.
        stats if with_stats else roots[:32].view(torch.float32),
        scales.view(torch.uint8).reshape(-1) if with_stats else roots[:32],
        _fused_tail_counter(data.device) if fused_tail else roots[:4].view(torch.int32),
        Int64(data_bytes),
        get_stream(data.device.index),
    )
    # Final 32-byte digest lives at the start of the scratchpad; async D2D copy
    # on the current stream.
    out.copy_(roots[: _blake3.CHAINING_VALUE_SIZE])


def _tree_flags(data_bytes, num_blocks, num_blocks_for_mt, chunk_size):
    """A blob's ``(apply_root, has_full_chunks, single_mt_block)`` flags.

    These are the only things about a blob's tree that the pipeline needs at
    compile time, so they are what a partitioned launch has to carry per blob:
    the two blobs differ in length by 4x and so reach different tree depths.
    """
    return (num_blocks == 1, data_bytes >= chunk_size, num_blocks_for_mt == 1)


def _mad_rot_dispatch(mad_rot, sync_loads, num_bytes):
    """Size-gated rotate-offload dispatch (rationale at
    ``_blake3.MAD_ROT_MIN_BYTES``): separate compiled variants, no branching
    in the kernel. The sync path always stays on SHF."""
    return mad_rot and not sync_loads and num_bytes >= _blake3.MAD_ROT_MIN_BYTES


def tensor_hash_pair_launch(
    codes,
    scales,
    key,
    root_codes,
    root_scales,
    roots,
    *,
    threads_per_block,
    num_stages,
    leaves_per_mt_block,
    thread_load_size=DEFAULT_THREAD_LOAD_SIZE,
    chunk_size=_blake3.CHUNK_SIZE,
    chunks_per_thread=1,
    mad_rot=False,
    sync_loads=False,
    stats=None,
):
    """Keyed-BLAKE3 Merkle chains over both activation blobs in one grid.

    Each of the three stages is a single launch whose grid is partitioned
    between the blobs: a CTA compares its ``blockIdx`` against the codes blob's
    CTA count and then works only from its own blob's byte length, tree shape
    and ``roots`` slice. The digests are therefore identical to hashing the
    blobs separately, while the chain's payload-independent latency floor
    (three dependent launches whose smallest unit of work is a whole 1024-byte
    chunk hashed serially by one thread) is paid once for the pair.

    ``roots`` is the caller's whole workspace (sized by
    ``tensor_hash_workspace_bytes``), split into the two blobs' back-to-back
    slices at the effective-coarsening block count, so each digest lands at
    the start of its own slice. ``stats`` rides the codes blob's partition of
    stage 1 exactly as it rides ``tensor_hash_launch``.
    """
    # Ahead of the workspace split below, which reads both blob lengths and
    # divides by the tunables before the per-blob validation would reach them.
    _require_tensors(("codes", codes), ("scales", scales))
    _validate_hash_tunables(
        threads_per_block,
        num_stages,
        leaves_per_mt_block,
        thread_load_size,
        chunk_size,
        chunks_per_thread,
    )
    codes_bytes = codes.numel() * codes.element_size()
    scales_bytes = scales.numel() * scales.element_size()
    with_stats = stats is not None
    # One compile-time k covers the whole pair grid, so both blobs must
    # support it (the scales blob is a quarter of the codes blob).
    chunks_per_thread = _effective_chunks_per_thread(
        chunks_per_thread, (codes_bytes, scales_bytes), chunk_size, with_stats
    )
    # The split point between the blobs' root slices follows the effective
    # coarsening (the scales tree starts right after the codes tree's k-based
    # block count); the caller's workspace is sized by the k=1 formula, which
    # is always at least as large.
    codes_workspace = get_required_scratchpad_bytes(
        codes_bytes, threads_per_block, chunk_size * chunks_per_thread
    )
    scales_workspace = get_required_scratchpad_bytes(
        scales_bytes, threads_per_block, chunk_size * chunks_per_thread
    )
    hash_kwargs = {
        "threads_per_block": threads_per_block,
        "num_stages": num_stages,
        "leaves_per_mt_block": leaves_per_mt_block,
        "thread_load_size": thread_load_size,
        "chunk_size": chunk_size,
        "chunks_per_thread": chunks_per_thread,
    }
    # Each blob is validated against its own slice, the alignment included: the
    # split point is a multiple of 32 bytes, so the second slice keeps the
    # 4-byte alignment the uint32 root view needs.
    codes_geometry = _validate_tensor_hash_buffers(
        codes, key, root_codes, roots[:codes_workspace], **hash_kwargs
    )
    scales_geometry = _validate_tensor_hash_buffers(
        scales,
        key,
        root_scales,
        roots[codes_workspace : codes_workspace + scales_workspace],
        **hash_kwargs,
    )
    if with_stats:
        _validate_stats_compatible(chunk_size)
        _validate_stats_operands("codes", codes_bytes, scales, stats, codes.device)

    # Per-blob gate: the scales blob is a quarter of the codes blob, so it
    # usually stays below it.
    codes_mad_rot = _mad_rot_dispatch(mad_rot, sync_loads, codes_bytes)
    scales_mad_rot = _mad_rot_dispatch(mad_rot, sync_loads, scales_bytes)

    cache_key = _PairVariantKey(
        threads_per_block,
        num_stages,
        leaves_per_mt_block,
        thread_load_size,
        chunk_size,
        chunks_per_thread,
        codes_mad_rot,
        scales_mad_rot,
        sync_loads,
        _tree_flags(*codes_geometry, chunk_size),
        _tree_flags(*scales_geometry, chunk_size),
        with_stats,
    )
    compiled = _compile_pair_variant(*cache_key)
    compiled(
        codes.view(torch.uint8).reshape(-1),
        scales.view(torch.uint8).reshape(-1),
        key.view(torch.int32),
        roots.view(torch.int32),
        # The stats=None variant never touches mStats (with_stats is a compile
        # constant); the roots scratchpad stands in as a harmless dummy.
        stats if with_stats else roots[:32].view(torch.float32),
        Int64(codes_bytes),
        Int64(scales_bytes),
        get_stream(codes.device.index),
    )
    # Each blob's digest lives at the start of its own slice; async D2D copies
    # on the current stream.
    root_codes.copy_(roots[: _blake3.CHAINING_VALUE_SIZE])
    root_scales.copy_(roots[codes_workspace : codes_workspace + _blake3.CHAINING_VALUE_SIZE])
