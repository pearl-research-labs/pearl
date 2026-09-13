"""Chunk hashing and per-CTA Merkle reduction."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync

from ._blake3 import (
    CHAINING_VALUE_SIZE_U32,
    CHUNK_END,
    CHUNK_START,
    FLAGS_INNER_NODE,
    KEYED_HASH,
    MSG_BLOCK_SIZE,
    MSG_BLOCK_SIZE_U32,
    ROOT,
    compress_msg_block,
)
from ._block_stats import (
    _STATS_CODE_BYTES,
    _STATS_MSG_BLOCKS,
    _UNIT_CODE_WORDS,
    _msg_block_stats,
    _pairwise_sum,
    _select_f32,
    _stats_carry,
    _window_ones,
)
from ._merkle_tree_utils import compute_blake_mt, compute_perfect_mt

# Named barriers for consumer-group synchronization. The DSL reserves
# ``bar.sync 0`` for cute.arch.barrier(), so use IDs 1/2.
_PRIMARY_CONSUMERS_BARRIER = 1
_SECONDARY_CONSUMERS_BARRIER = 2

# Producer threads reserved at the front of the CTA. Only warp 0 issues TMA
# loads; a full producer warpgroup (128) left warps 1-3 idling at the final
# barrier while inflating the CTA footprint (fewer CTAs per SM whenever the
# SM is thread-limited). One warp is sufficient.
_NUM_PRODUCER_THREADS = 32
_MAX_TMA_THREADS = 256  # each TMA descriptor dimension is limited to 256
# Chunks one consumer thread may hash (thread coarsening). Powers of two, so a
# thread's private fold is always an aligned BLAKE3 subtree.
SUPPORTED_CHUNKS_PER_THREAD = (1, 2, 4, 8)
# Scales are read in the unit one message block needs: 4 u32 words (eight
# packed BF16 scales) per 64 hashed code bytes, so 16 bytes and one load.
_SCALE_WORDS_PER_MSG_BLOCK = MSG_BLOCK_SIZE_U32 // _UNIT_CODE_WORDS
# Carry levels of the sync path's per-message-block window (a stats block is
# eight message blocks, so its pairwise tree carries over three doublings).
_MSG_CARRY_LEVELS = _STATS_MSG_BLOCKS.bit_length() - 1


def _small_leaf_stats(chunk_size, with_stats) -> bool:
    """Whether the stats fusion takes the sub-block path at this leaf size.

    Below one stats block a leaf's thread owns only part of a block, so the
    per-thread carry cannot finish it: each staged message block's partial is
    parked in smem and one owner thread per stats block folds the
    ``_STATS_MSG_BLOCKS`` pairs after the CTA barrier. At or above one block
    the host requires a whole multiple (``_validate_stats_compatible``) and
    the carry runs in registers.
    """
    return with_stats and chunk_size < _STATS_CODE_BYTES


def _stats_smem_floats(num_consumer_threads, chunk_size, chunks_per_thread, with_stats) -> int:
    """fp32 slots of the small-leaf staging region (0 when the path is off):
    one (sumsq, absmax) pair per hashed message block of the CTA."""
    if not _small_leaf_stats(chunk_size, with_stats):
        return 0
    return 2 * num_consumer_threads * chunks_per_thread * chunk_size // MSG_BLOCK_SIZE


def _derived_config(num_consumer_threads, thread_load_size, chunk_size, chunks_per_thread=1):
    """Compile-time load geometry shared by the host launcher and the kernel.

    ``chunks_per_thread`` only scales ``num_loads``; the TMA box, smem ring
    and leaves buffer are unchanged (see ``_consumer_loop`` for the
    coarsening mechanism).
    """
    assert num_consumer_threads % 128 == 0 and num_consumer_threads >= 128
    assert thread_load_size in (64, 128, 256, 512)
    assert chunk_size % thread_load_size == 0
    assert chunks_per_thread in SUPPORTED_CHUNKS_PER_THREAD
    use_dual = num_consumer_threads > _MAX_TMA_THREADS
    assert not use_dual or num_consumer_threads == 512, (
        "Dual pipeline mode currently only supports exactly 512 consumer threads"
    )
    tma_threads = _MAX_TMA_THREADS if use_dual else num_consumer_threads
    num_words_per_load = thread_load_size // 4
    num_blocks_per_load = thread_load_size // MSG_BLOCK_SIZE
    num_loads = chunks_per_thread * chunk_size // thread_load_size
    return use_dual, tma_threads, num_words_per_load, num_blocks_per_load, num_loads


def _make_smem_layouts(num_consumer_threads, num_pipe_stages, thread_load_size, chunk_size):
    """SMEM layouts for the staged TMA buffer and the leaves (host + kernel)."""
    _, tma_threads, num_words_per_load, _, _ = _derived_config(
        num_consumer_threads, thread_load_size, chunk_size
    )
    atom_kind = (
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64
        if thread_load_size == 64
        else cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128
    )
    sa_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(atom_kind, Uint32)
    sa_layout_staged = cute.tile_to_shape(
        sa_atom, (tma_threads, num_words_per_load, num_pipe_stages), order=(0, 1, 2)
    )
    leaves_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128, Uint32
    )
    leaves_layout = cute.tile_to_shape(
        leaves_atom, (CHAINING_VALUE_SIZE_U32, num_consumer_threads), order=(0, 1)
    )
    return sa_layout_staged, leaves_layout


def _make_shared_storage(leaves_cosize, sa_cosize, num_pipe_stages, use_dual):
    """SharedStorage struct (single or dual TMA pipelines).

    ``sClose`` broadcasts the fused-tail "this CTA closes the reduction"
    decision (made by thread 0 via a global atomic) to the whole CTA.
    """
    if use_dual:

        @cute.struct
        class SharedStorageDual:
            mbar0: cute.struct.MemRange[cutlass.Int64, 2 * num_pipe_stages]
            mbar1: cute.struct.MemRange[cutlass.Int64, 2 * num_pipe_stages]
            sClose: cute.struct.MemRange[cutlass.Int32, 1]
            sLeaves: cute.struct.Align[cute.struct.MemRange[Uint32, leaves_cosize], 1024]
            sA0: cute.struct.Align[cute.struct.MemRange[Uint32, sa_cosize], 1024]
            sA1: cute.struct.Align[cute.struct.MemRange[Uint32, sa_cosize], 1024]

        return SharedStorageDual

    @cute.struct
    class SharedStorageSingle:
        mbar0: cute.struct.MemRange[cutlass.Int64, 2 * num_pipe_stages]
        sClose: cute.struct.MemRange[cutlass.Int32, 1]
        sLeaves: cute.struct.Align[cute.struct.MemRange[Uint32, leaves_cosize], 1024]
        sA0: cute.struct.Align[cute.struct.MemRange[Uint32, sa_cosize], 1024]

    return SharedStorageSingle


@cute.jit
def _producer_loop(
    pipe,
    tma_atom,
    mA_coord,
    sA,
    group_bid,
    num_loads: cutlass.Constexpr[int],
    tma_threads: cutlass.Constexpr[int],
    num_words_per_load: cutlass.Constexpr[int],
    has_full_chunks: cutlass.Constexpr[bool],
):
    """TMA producer loop (producer_loop)."""
    state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, pipe.num_stages)

    # View of our CTA's tile of A, partitioned by load (thread_load_size bytes
    # per consumer chunk each).
    gA = cute.local_tile(mA_coord, (tma_threads, num_words_per_load), (group_bid, None))
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA, 0, 2),
    )

    for load_idx in cutlass.range(num_loads, unroll=1):
        pipe.producer_acquire(state)
        # With no full chunk the descriptor covers no backed memory; skip the
        # copy entirely (the barrier expects 0 transaction bytes in that case
        # and completes on the producer's arrival alone).
        if cutlass.const_expr(has_full_chunks):
            cute.copy(
                tma_atom,
                tAgA[(None, load_idx)],
                tAsA[(None, state.index)],
                tma_bar_ptr=pipe.producer_get_barrier(state),
            )
        pipe.producer_commit(state)
        state.advance()

    # Waits for all stages to be released (all consumer UNLOCKs), or if a stage
    # was never used it is just acquired since the phase is still inverted.
    pipe.producer_tail(state)


@cute.jit
def _producer_loop_dual(
    pipe0,
    pipe1,
    tma_atom,
    mA_coord,
    sA0,
    sA1,
    bidx,
    num_loads: cutlass.Constexpr[int],
    tma_threads: cutlass.Constexpr[int],
    num_words_per_load: cutlass.Constexpr[int],
    has_full_chunks: cutlass.Constexpr[bool],
):
    """TMA producer loop feeding two 256-consumer pipelines."""
    state0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, pipe0.num_stages)
    state1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, pipe1.num_stages)

    gA0 = cute.local_tile(mA_coord, (tma_threads, num_words_per_load), (bidx * 2, None))
    gA1 = cute.local_tile(mA_coord, (tma_threads, num_words_per_load), (bidx * 2 + 1, None))
    tAsA0, tAgA0 = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1), cute.group_modes(sA0, 0, 2), cute.group_modes(gA0, 0, 2)
    )
    tAsA1, tAgA1 = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1), cute.group_modes(sA1, 0, 2), cute.group_modes(gA1, 0, 2)
    )

    for load_idx in cutlass.range(num_loads, unroll=1):
        pipe0.producer_acquire(state0)
        pipe1.producer_acquire(state1)
        if cutlass.const_expr(has_full_chunks):
            cute.copy(
                tma_atom,
                tAgA0[(None, load_idx)],
                tAsA0[(None, state0.index)],
                tma_bar_ptr=pipe0.producer_get_barrier(state0),
            )
            cute.copy(
                tma_atom,
                tAgA1[(None, load_idx)],
                tAsA1[(None, state1.index)],
                tma_bar_ptr=pipe1.producer_get_barrier(state1),
            )
        pipe0.producer_commit(state0)
        pipe1.producer_commit(state1)
        state0.advance()
        state1.advance()

    pipe0.producer_tail(state0)
    pipe1.producer_tail(state1)


@cute.jit
def _load_partial_chunk_warp(
    mDataU8,
    mDataU32,
    sA,
    owner_row,
    stage,
    load_idx,
    last_chunk_len,
    data_len,
    lane,
    thread_load_size: cutlass.Constexpr[int],
    num_words_per_load: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
):
    """Load one slice of the trailing partial chunk straight from global memory
    into shared memory, zero-filling past the end of the data. At most one
    chunk per hash is read this way.

    Warp-cooperative: the owner thread's whole warp loads the tail slice in
    parallel (one word per lane) instead of the owner serially loading
    ``num_words_per_load`` words while its group stalls at the per-stage
    barrier. Caller must ``sync_warp`` before the owner compresses from smem.
    """
    # The partial chunk starts right after the last full chunk.
    chunk_start_byte = (data_len // chunk_size) * chunk_size
    # This load covers bytes [load_start_byte, load_start_byte + thread_load_size)
    # of the chunk, i.e. sA(owner_row, 0..num_words_per_load-1, stage).
    load_start_byte = load_idx * thread_load_size

    for w0 in cutlass.range_constexpr(0, num_words_per_load, 32):
        w = w0 + lane
        if w < num_words_per_load:
            word_start_byte = load_start_byte + w * 4
            word = Uint32(0)
            if word_start_byte < last_chunk_len:
                remaining = last_chunk_len - word_start_byte
                if remaining >= 4:
                    word = Uint32(mDataU32[(chunk_start_byte + word_start_byte) >> 2])
                else:
                    for b in cutlass.range_constexpr(3):
                        if b < remaining:
                            word = Uint32(
                                word
                                | (
                                    mDataU8[chunk_start_byte + word_start_byte + b].to(Uint32)
                                    << (8 * b)
                                )
                            )
            sA[owner_row, w, stage] = word


def _fold_private_subtree(rCVs, rChainingValue, rKey, chunks_per_thread):
    """Fold the k parked sub-chunk CVs into one leaf CV (in rChainingValue).

    Bottom-up over the thread's own aligned subtree, each level halving the
    live CVs in place -- at k=4, ``(0,1) -> 0`` and ``(2,3) -> 2``, then
    ``(0,2) -> 0``.

    Plain-Python helper (NOT @cute.jit): the loop bounds are trace-time
    constants derived from the constexpr ``chunks_per_thread``, so the whole
    k-1 compression fold unrolls into straight-line register code. (The DSL
    preprocessor would turn a ``while`` inside a jit function into a dynamic
    loop, breaking the constexpr indexing into ``rCVs``.)
    """
    cv = CHAINING_VALUE_SIZE_U32
    rParent = cute.make_rmem_tensor(2 * cv, Uint32)
    span = 1
    while span < chunks_per_thread:
        for base in range(0, chunks_per_thread, 2 * span):
            for i in range(cv):
                rParent[i] = rCVs[base * cv + i]
                rParent[cv + i] = rCVs[(base + span) * cv + i]
                rChainingValue[i] = rKey[i]
            compress_msg_block(rParent, rChainingValue, Uint32(0), Uint32(FLAGS_INNER_NODE))
            for i in range(cv):
                rCVs[base * cv + i] = rChainingValue[i]
        span *= 2


@cute.jit
def _load_msg_block(
    sA,
    smem_row,
    stage,
    block_in_load: cutlass.Constexpr[int],
):
    """Stage one 64-byte block of this thread's chunk into registers.

    ``smem_row`` indexes the per-group smem region. The copy uses 128-bit
    loads (4 words at a time); the SW64/SW128 swizzles keep 16-byte units
    contiguous. The commit stats read the same registers, so the codes blob
    reaches both consumers of it from this one staging.
    """
    rBlock = cute.make_rmem_tensor(MSG_BLOCK_SIZE_U32, Uint32)
    word_offset = block_in_load * MSG_BLOCK_SIZE_U32
    row = sA[(smem_row, None, stage)]
    for i in cutlass.range_constexpr(MSG_BLOCK_SIZE_U32 // 4):
        src4 = cute.local_tile(row, (4,), (word_offset // 4 + i,))
        dst4 = cute.local_tile(rBlock, (4,), (i,))
        cute.autovec_copy(src4, dst4)
    return rBlock


@cute.jit
def _compress_block(
    rBlock,
    rChainingValue,
    chunk_idx,
    block_idx,
    root_flag,
    chunk_size: cutlass.Constexpr[int],
    mad_rot: cutlass.Constexpr[bool],
):
    """Compress one 64-byte block of this thread's chunk (compress_block).

    All chunks are ``chunk_size`` bytes and all blocks 64 bytes, so BLAKE3's
    chunk counter is ``chunk_idx``, the chunk's index *within its own blob* --
    never one derived from ``blockIdx``, which a partitioned grid offsets.
    ``root_flag`` is ROOT when the whole message is a single chunk and this
    kernel must finalize it, else 0. ``mad_rot`` is the host's size-gated
    CLMAD rotate-offload dispatch (leaf compressions only; see ``_blake3``).
    """
    counter = Uint32(chunk_idx)

    # CHUNK_START on the first block, CHUNK_END (+ ROOT when the whole message
    # is this single chunk) on the last block of the chunk.
    is_first = (block_idx == 0).to(Uint32)
    is_last = (block_idx == chunk_size // MSG_BLOCK_SIZE - 1).to(Uint32)
    flags = (
        Uint32(KEYED_HASH)
        | (is_first * CHUNK_START)
        | (is_last * CHUNK_END)
        | (is_last * root_flag)
    )

    compress_msg_block(rBlock, rChainingValue, counter, flags, mad_rot=mad_rot)


# noqa C901: the compression mainloop. Its branches are compile-time stage
# and stats dispatch that must stay in one traced body.
@cute.jit
def _consumer_loop(  # noqa: C901
    pipe,
    sA,
    sLeaves,
    sStats,
    mDataU8,
    mDataU32,
    mScalesU32,
    mStats,
    rKey,
    data_len,
    num_stats_blocks,
    local_bid,
    tid_in_group,
    consumer_tid,
    root_flag,
    barrier_id: cutlass.Constexpr[int],
    group_size: cutlass.Constexpr[int],
    num_consumer_threads: cutlass.Constexpr[int],
    thread_load_size: cutlass.Constexpr[int],
    num_loads: cutlass.Constexpr[int],
    num_blocks_per_load: cutlass.Constexpr[int],
    num_words_per_load: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
    mad_rot: cutlass.Constexpr[bool],
    with_stats: cutlass.Constexpr[bool],
):
    """Consumer loop, hashing one leaf group per thread and reducing its stats.

    ``tid_in_group`` indexes the group's smem region; ``consumer_tid`` is the
    CTA-wide consumer index used for the leaves, and with ``local_bid`` -- the
    CTA's index within its blob's partition of the grid -- forms the leaf
    group (== chunk, at ``chunks_per_thread == 1``) this thread hashes.

    ``chunks_per_thread`` > 1 (thread coarsening): the thread's smem row
    holds k consecutive chunks streamed as k sub-chunk passes of
    ``num_loads / k`` pipeline stages each; every sub-chunk is a full BLAKE3
    chunk (own counter, CHUNK_START/END) whose CV is parked in registers, and
    the k CVs are folded into the thread's leaf with k-1 register-resident
    parent compressions at full occupancy. The host only enables this for
    exact multiples of k*chunk_size bytes (no partial chunk, no ragged
    group), with at least 2 groups (so the private fold is never the message
    root).

    On the activation path the thread also reduces the commit-stats partials
    of the chunks it hashes, out of the message-block registers, adding one
    16-byte scales load per compression (see ``_block_stats``). Leaves of one
    or more whole stats blocks carry the partials in registers and publish
    them here; sub-block leaves park each message block's pair in ``sStats``
    for the post-barrier cross-thread fold (``_fold_small_leaf_stats``).
    Only the publication is guarded: a chunk past the end of the data, or
    the trailing padded part of a partial chunk, reduces zero-filled smem
    and publishes nothing.
    """
    read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, pipe.num_stages)

    # Register tensor (chaining value) of our hash's state, initialized with the key
    rChainingValue = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32, Uint32)
    for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
        rChainingValue[i] = rKey[i]

    # Parked per-sub-chunk CVs for the coarsened register fold (k > 1 only;
    # the 1-word dummy keeps the allocation out of divergent control flow).
    rCVs = cute.make_rmem_tensor(
        chunks_per_thread * CHAINING_VALUE_SIZE_U32 if chunks_per_thread > 1 else 1,
        Uint32,
    )

    # Calculate if this warp holds the last (potentially partial) chunk. Use
    # ceiling division to include partial chunks in the count. Partial chunks
    # only exist at chunks_per_thread == 1 (host gate), where a leaf group is
    # exactly one chunk.
    num_chunks = Int32(cute.ceil_div(data_len, chunk_size))
    remainder = Int32(data_len % chunk_size)
    last_chunk_size = Int32(chunk_size)
    if remainder != 0:
        last_chunk_size = remainder
    group_idx = local_bid * num_consumer_threads + consumer_tid

    # Partial-tail handling: the whole warp containing the last (partial)
    # chunk's owner loads the tail cooperatively (one word per lane) instead
    # of the owner serially loading the slice each stage while the rest of
    # its group waits at the per-stage barrier.
    lane = cute.arch.lane_idx()
    owner_ctid = (num_chunks - 1) - local_bid * num_consumer_threads
    warp_has_partial = (
        last_chunk_size < chunk_size
        and owner_ctid >= 0
        and owner_ctid < num_consumer_threads
        and owner_ctid // 32 == consumer_tid // 32
    )
    owner_row = owner_ctid % group_size

    # Chunk-to-scales and chunk-to-stats maps, derived from the leaf size
    # (one scale unit per hashed 64-byte message block).
    small_stats: cutlass.Constexpr = _small_leaf_stats(chunk_size, with_stats)
    scale_units_per_chunk: cutlass.Constexpr = chunk_size // MSG_BLOCK_SIZE
    stats_blocks_per_chunk: cutlass.Constexpr = chunk_size // _STATS_CODE_BYTES
    # A stats block spans this many smem windows, so its pairwise sum carries
    # over that many loop iterations (one level per doubling). Meaningful on
    # the carry path only (a sub-block leaf never completes a block).
    windows_per_stats_block: cutlass.Constexpr = max(_STATS_CODE_BYTES // thread_load_size, 1)
    carry_levels: cutlass.Constexpr = windows_per_stats_block.bit_length() - 1
    if cutlass.const_expr(with_stats):
        # Clamp target for the scales of code bytes that do not exist.
        last_scale_unit = Int64(data_len // MSG_BLOCK_SIZE - 1)
        if cutlass.const_expr(not small_stats):
            rAcc = cute.make_rmem_tensor(max(carry_levels, 1), Float32)
            for level in cutlass.range_constexpr(carry_levels):
                rAcc[level] = Float32(0.0)
            rMax = cute.make_rmem_tensor(1, Float32)
            rMax[0] = Float32(0.0)

    loads_per_chunk: cutlass.Constexpr = num_loads // chunks_per_thread

    for j in cutlass.range_constexpr(chunks_per_thread):
        if cutlass.const_expr(j > 0):
            # Fresh chaining value for the next coarsened sub-chunk.
            for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
                rChainingValue[i] = rKey[i]
        chunk_idx = group_idx * chunks_per_thread + j

        for load_idx in cutlass.range(loads_per_chunk, unroll=1):
            # Wait for TMA load to complete
            pipe.consumer_wait(read_state)
            stage = read_state.index

            # Partial chunks only exist at chunks_per_thread == 1 (host gate).
            if cutlass.const_expr(chunks_per_thread == 1):  # noqa: SIM102
                if warp_has_partial:
                    _load_partial_chunk_warp(
                        mDataU8,
                        mDataU32,
                        sA,
                        owner_row,
                        stage,
                        load_idx,
                        last_chunk_size,
                        data_len,
                        lane,
                        thread_load_size,
                        num_words_per_load,
                        chunk_size,
                    )
                    cute.arch.sync_warp()

            if cutlass.const_expr(with_stats):
                scale_unit = (
                    Int64(chunk_idx) * scale_units_per_chunk + Int64(load_idx) * num_blocks_per_load
                )
                if cutlass.const_expr(not small_stats):
                    window = load_idx & (windows_per_stats_block - 1)
                    ones = _window_ones(window, carry_levels)
                    window_sums = []
                    window_max = Float32(0.0)

            # Process num_blocks_per_load blocks from this load
            for block_in_load in cutlass.range_constexpr(num_blocks_per_load):
                block_idx = load_idx * num_blocks_per_load + block_in_load
                rBlock = _load_msg_block(sA, tid_in_group, stage, block_in_load)
                if cutlass.const_expr(with_stats):
                    # Issued before the compression that hides its latency.
                    rScales = cute.make_rmem_tensor(_SCALE_WORDS_PER_MSG_BLOCK, Uint32)
                    cute.autovec_copy(
                        cute.local_tile(
                            mScalesU32,
                            (_SCALE_WORDS_PER_MSG_BLOCK,),
                            (cutlass.min(scale_unit + block_in_load, last_scale_unit),),
                        ),
                        rScales,
                    )
                _compress_block(
                    rBlock, rChainingValue, chunk_idx, block_idx, root_flag, chunk_size, mad_rot
                )
                if cutlass.const_expr(with_stats):
                    msg_ssq, msg_max = _msg_block_stats(rBlock, rScales, MSG_BLOCK_SIZE_U32)
                    if cutlass.const_expr(small_stats):
                        # Sub-block leaf: park this message block's pair for
                        # the post-barrier cross-thread fold.
                        rPark = cute.make_rmem_tensor(2, Float32)
                        rPark[0] = msg_ssq
                        rPark[1] = msg_max
                        cta_msg = (
                            consumer_tid * (chunks_per_thread * scale_units_per_chunk)
                            + j * scale_units_per_chunk
                            + block_idx
                        )
                        cute.autovec_copy(rPark, cute.local_tile(sStats, (2,), (cta_msg,)))
                    else:
                        window_sums.append(msg_ssq)
                        window_max = cute.arch.fmax(window_max, msg_max)

            if cutlass.const_expr(with_stats and not small_stats):
                sumsq = _stats_carry(rAcc, _pairwise_sum(window_sums), ones, carry_levels)
                # absmax needs no tree (max is exact and associative), only the
                # reset on a stats block's first window; the one BF16 rounding
                # rides the store.
                first = (window == 0).to(Uint32)
                absmax = cute.arch.fmax(_select_f32(first, Float32(0.0), rMax[0]), window_max)
                rMax[0] = absmax
                blk = chunk_idx * stats_blocks_per_chunk + (load_idx >> carry_levels)
                if ones[carry_levels] != 0 and blk < num_stats_blocks:
                    rSt = cute.make_rmem_tensor(2, Float32)
                    rSt[0] = sumsq
                    rSt[1] = absmax.to(cutlass.BFloat16).to(Float32)
                    cute.autovec_copy(rSt, cute.local_tile(mStats, (2,), (blk,)))

            # Sync all consumers after processing this stage, ensuring all of
            # them finish before any releases the stage.
            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=group_size)

            # Release the pipeline stage
            pipe.consumer_release(read_state)
            read_state.advance()

        if cutlass.const_expr(chunks_per_thread > 1):
            # Park this sub-chunk's CV for the register fold below.
            for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
                rCVs[j * CHAINING_VALUE_SIZE_U32 + i] = rChainingValue[i]

    if cutlass.const_expr(chunks_per_thread > 1):
        # Fold the k sub-chunk CVs into the thread's leaf: the k-1 parent
        # compressions of an aligned k-chunk subtree, entirely in registers
        # at full occupancy (no barriers, no shared memory). Never the
        # message root: the host requires >= 2 groups.
        _fold_private_subtree(rCVs, rChainingValue, rKey, chunks_per_thread)

    # Store final hash result to sLeaves
    for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
        sLeaves[i, consumer_tid] = rChainingValue[i]


@cute.jit
def _fold_small_leaf_stats(
    sStats,
    mStats,
    tidx,
    local_bid,
    num_stats_blocks,
    stats_blocks_per_cta: cutlass.Constexpr[int],
):
    """Fold the CTA's staged sub-block partials into its stats and publish.

    One owner thread per stats block reads the block's ``_STATS_MSG_BLOCKS``
    staged ``(sumsq, absmax)`` pairs -- written by however many consumers the
    sub-block leaf spread the block over -- and folds them pairwise: the same
    balanced in-order tree over the block's 64 group terms as the carry path,
    so the leaf size never moves the summation order. Caller must place a CTA
    barrier between the consumer loop's stores and this fold. Only in-range
    blocks publish, so staged garbage from chunks past the data end is dead.
    """
    if tidx < stats_blocks_per_cta:
        blk = local_bid * stats_blocks_per_cta + tidx
        if blk < num_stats_blocks:
            sums = []
            absmax = Float32(0.0)
            for i in cutlass.range_constexpr(_STATS_MSG_BLOCKS):
                rPair = cute.make_rmem_tensor(2, Float32)
                cute.autovec_copy(
                    cute.local_tile(sStats, (2,), (tidx * _STATS_MSG_BLOCKS + i,)), rPair
                )
                sums.append(rPair[0])
                absmax = cute.arch.fmax(absmax, rPair[1])
            rSt = cute.make_rmem_tensor(2, Float32)
            rSt[0] = _pairwise_sum(sums)
            rSt[1] = absmax.to(cutlass.BFloat16).to(Float32)
            cute.autovec_copy(rSt, cute.local_tile(mStats, (2,), (blk,)))


@cute.jit
def _stage1_cta_setup(
    mKey,
    sa_layout_staged,
    leaves_layout,
    num_pipe_stages: cutlass.Constexpr[int],
    use_dual: cutlass.Constexpr[bool],
    stats_smem_floats: cutlass.Constexpr[int],
):
    """Allocate the CTA's staging buffers and stage the key into registers.

    Done once per CTA, outside any blob partitioning: the load geometry is a
    compile-time constant shared by every blob in the grid, so a CTA reuses
    these buffers for whichever blob it selects. ``stats_smem_floats`` sizes
    the sub-block stats staging region (0 keeps it out of the CTA footprint).
    """
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(
        _make_shared_storage(
            cute.cosize(leaves_layout), cute.cosize(sa_layout_staged), num_pipe_stages, use_dual
        )
    )
    sStats = None
    if cutlass.const_expr(stats_smem_floats > 0):
        sStats = smem.allocate_tensor(
            Float32, cute.make_layout(stats_smem_floats), byte_alignment=16
        )
    # Key words, staged from gmem (the DSL has no user constant memory).
    key_ptr = cute.recast_ptr(mKey.iterator, dtype=Uint32)
    mKeyWords = cute.make_tensor(key_ptr, cute.make_layout(CHAINING_VALUE_SIZE_U32))
    rKey = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32, Uint32)
    for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
        rKey[i] = mKeyWords[i]
    return storage, sStats, rKey


@cute.jit
def _blob_roots(  # noqa: C901
    storage,
    sStats,
    rKey,
    tma_atom: cute.CopyAtom,
    mA_coord: cute.Tensor,
    mData: cute.Tensor,
    mRoots: cute.Tensor,
    mStats: cute.Tensor,
    mScales: cute.Tensor,
    data_len: Int64,
    local_bid,
    roots_slot_base,
    mCounter: cute.Tensor,
    sa_layout_staged: cute.ComposedLayout,
    leaves_layout: cute.ComposedLayout,
    num_consumer_threads: cutlass.Constexpr[int],
    num_pipe_stages: cutlass.Constexpr[int],
    thread_load_size: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
    mad_rot: cutlass.Constexpr[bool],
    apply_root: cutlass.Constexpr[bool],
    has_full_chunks: cutlass.Constexpr[bool],
    fused_tail: cutlass.Constexpr[bool],
    with_stats: cutlass.Constexpr[bool],
):
    """Hash one blob's share of this CTA: its chunks, then its leaves' root.

    ``local_bid`` is the CTA's index within the blob's own partition of the
    grid and ``roots_slot_base`` the blob's first 32-byte root slot in the
    shared workspace. Everything the digest depends on -- the chunk counters,
    which chunk is the trailing partial one, how many leaves the last CTA
    holds, whether ROOT is applied -- follows from ``data_len`` and
    ``local_bid`` alone, so a blob's tree is unaffected by what shares the
    grid with it.

    ``fused_tail`` (single-blob path only): the last CTA to publish its root
    (global atomic ticket in ``mCounter``) additionally reduces every CTA
    root to the final digest inline, and the host skips the stage-2 launch.
    """
    use_dual, tma_threads, num_words_per_load, num_blocks_per_load, num_loads = _derived_config(
        num_consumer_threads, thread_load_size, chunk_size, chunks_per_thread
    )
    consumers_per_group = num_consumer_threads // (2 if use_dual else 1)

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    # Warp 0 is producer, rest are consumers
    is_producer_warpgroup = warp_idx < _NUM_PRODUCER_THREADS // 32

    # Prefetch TMA descriptor from warp 0
    if warp_idx == 0:
        with cute.arch.elect_one():
            cpasync.prefetch_descriptor(tma_atom)

    num_groups, num_grid_blocks, root_flag, mDataU32, mRootsView = _blob_geometry(
        mData,
        mRoots,
        data_len,
        roots_slot_base,
        num_consumer_threads,
        chunk_size,
        chunks_per_thread,
        apply_root,
    )

    # Word view of the companion scales blob, which holds one BF16 per 8 codes
    # and so is a quarter of data_len bytes (only the codes blob's partition is
    # ever compiled with stats; with_stats is a compile constant).
    scales_u32_ptr = cute.recast_ptr(mScales.iterator, dtype=Uint32)
    mScalesU32 = cute.make_tensor(scales_u32_ptr, cute.make_layout(cute.ceil_div(data_len, 16)))
    num_stats_blocks = Int32(data_len // _STATS_CODE_BYTES)

    sLeaves = storage.sLeaves.get_tensor(leaves_layout.outer, swizzle=leaves_layout.inner)

    # Pipelines: the producer warp arrives once (elected); each consumer warp
    # in the pipeline's group signals once on release (lane 0).
    tx_count = tma_threads * num_words_per_load * 4 if has_full_chunks else 0
    producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumers_per_group // 32)

    consumer_tid = tidx - _NUM_PRODUCER_THREADS

    if cutlass.const_expr(use_dual):
        sA0 = storage.sA0.get_tensor(sa_layout_staged.outer, swizzle=sa_layout_staged.inner)
        sA1 = storage.sA1.get_tensor(sa_layout_staged.outer, swizzle=sa_layout_staged.inner)
        pipe0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar0.data_ptr(),
            num_stages=num_pipe_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
        )
        pipe1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar1.data_ptr(),
            num_stages=num_pipe_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
        )
        consumer_group_idx = consumer_tid // consumers_per_group
        tid_in_group = consumer_tid % consumers_per_group

        if warp_idx == 0:
            _producer_loop_dual(
                pipe0,
                pipe1,
                tma_atom,
                mA_coord,
                sA0,
                sA1,
                local_bid,
                num_loads,
                tma_threads,
                num_words_per_load,
                has_full_chunks,
            )
        elif not is_producer_warpgroup:
            if consumer_group_idx == 0:
                _consumer_loop(
                    pipe0,
                    sA0,
                    sLeaves,
                    sStats,
                    mData,
                    mDataU32,
                    mScalesU32,
                    mStats,
                    rKey,
                    data_len,
                    num_stats_blocks,
                    local_bid,
                    tid_in_group,
                    consumer_tid,
                    root_flag,
                    _PRIMARY_CONSUMERS_BARRIER,
                    consumers_per_group,
                    num_consumer_threads,
                    thread_load_size,
                    num_loads,
                    num_blocks_per_load,
                    num_words_per_load,
                    chunk_size,
                    chunks_per_thread,
                    mad_rot,
                    with_stats,
                )
            else:
                _consumer_loop(
                    pipe1,
                    sA1,
                    sLeaves,
                    sStats,
                    mData,
                    mDataU32,
                    mScalesU32,
                    mStats,
                    rKey,
                    data_len,
                    num_stats_blocks,
                    local_bid,
                    tid_in_group,
                    consumer_tid,
                    root_flag,
                    _SECONDARY_CONSUMERS_BARRIER,
                    consumers_per_group,
                    num_consumer_threads,
                    thread_load_size,
                    num_loads,
                    num_blocks_per_load,
                    num_words_per_load,
                    chunk_size,
                    chunks_per_thread,
                    mad_rot,
                    with_stats,
                )
    else:
        sA0 = storage.sA0.get_tensor(sa_layout_staged.outer, swizzle=sa_layout_staged.inner)
        pipe0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar0.data_ptr(),
            num_stages=num_pipe_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
        )
        if warp_idx == 0:
            _producer_loop(
                pipe0,
                tma_atom,
                mA_coord,
                sA0,
                local_bid,
                num_loads,
                tma_threads,
                num_words_per_load,
                has_full_chunks,
            )
        elif not is_producer_warpgroup:
            _consumer_loop(
                pipe0,
                sA0,
                sLeaves,
                sStats,
                mData,
                mDataU32,
                mScalesU32,
                mStats,
                rKey,
                data_len,
                num_stats_blocks,
                local_bid,
                consumer_tid,
                consumer_tid,
                root_flag,
                _PRIMARY_CONSUMERS_BARRIER,
                consumers_per_group,
                num_consumer_threads,
                thread_load_size,
                num_loads,
                num_blocks_per_load,
                num_words_per_load,
                chunk_size,
                chunks_per_thread,
                mad_rot,
                with_stats,
            )

    # Sync all threads (producer + consumer) before merkle tree reduction
    cute.arch.barrier()

    if cutlass.const_expr(_small_leaf_stats(chunk_size, with_stats)):
        _fold_small_leaf_stats(
            sStats,
            mStats,
            tidx,
            local_bid,
            num_stats_blocks,
            num_consumer_threads * chunks_per_thread * chunk_size // _STATS_CODE_BYTES,
        )

    _reduce_cta_leaves(
        sLeaves,
        mRootsView,
        rKey,
        tidx,
        local_bid,
        num_groups,
        num_grid_blocks,
        roots_slot_base,
        num_consumer_threads,
        apply_root,
    )

    # SIM102 off: the outer if is a compile-time const_expr branch, the inner
    # one is dynamic; combining them would change staging semantics.
    if cutlass.const_expr(fused_tail):  # noqa: SIM102
        # At num_grid_blocks == 1 the digest is already final (apply_root), so
        # there is nothing left to reduce.
        if num_grid_blocks > 1:
            _fused_tail_close(
                storage.sClose.get_tensor(cute.make_layout(1)),
                sLeaves,
                mRootsView,
                mCounter,
                rKey,
                tidx,
                roots_slot_base,
                num_grid_blocks,
            )


@cute.jit
def _fused_tail_close(
    sClose,
    sLeaves,
    mRootsView,
    mCounter: cute.Tensor,
    rKey,
    tidx,
    roots_slot_base,
    num_grid_blocks,
):
    """Let the final CTA reduce published roots as fused stage 2."""
    # Order this CTA's root stores (threads 0..7) before thread 0's
    # release-atomic below.
    cute.arch.barrier()
    if tidx == 0:
        old = cute.arch.atomic_add(mCounter.iterator, Int32(1), sem="acq_rel", scope="gpu")
        is_closer = old == num_grid_blocks - 1
        sClose[0] = is_closer.to(cutlass.Int32)
        if is_closer:
            # Safe to reset for the next launch: every peer has already passed
            # its own atomic, so nobody else touches this slot.
            cute.arch.store(mCounter.iterator, Int32(0), sem="relaxed", scope="gpu")
    cute.arch.barrier()
    if sClose[0] != 0:
        # We saw every root published (acquire pairs with each CTA's release):
        # reload them as leaves and reduce. The host only fuses at two CTAs,
        # so the leaf count is always a power of two.
        if tidx < num_grid_blocks:
            for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
                sLeaves[i, tidx] = mRootsView[i, roots_slot_base + tidx]
        cute.arch.barrier()
        compute_perfect_mt(sLeaves, num_grid_blocks, rKey, tidx, True)
        if tidx < CHAINING_VALUE_SIZE_U32:
            mRootsView[tidx, roots_slot_base] = sLeaves[tidx, 0]


@cute.jit
def _blob_geometry(
    mData: cute.Tensor,
    mRoots: cute.Tensor,
    data_len: Int64,
    roots_slot_base,
    num_consumer_threads: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
    apply_root: cutlass.Constexpr[bool],
):
    """Leaf-group counts, ROOT flag and u32 views, as both load paths need them.

    A leaf group is ``chunks_per_thread`` consecutive chunks (== one chunk at
    k=1); the host guarantees exact divisibility when k > 1, so the ceiling
    only matters at k=1 (partial trailing chunk).
    """
    num_chunks = Int32(cute.ceil_div(data_len, chunk_size))
    num_groups = num_chunks
    if cutlass.const_expr(chunks_per_thread > 1):
        num_groups = Int32(num_chunks // chunks_per_thread)
    num_grid_blocks = cute.ceil_div(num_groups, num_consumer_threads)

    # When the entire message is a single chunk, BLAKE3 sets ROOT on that
    # chunk's last block (no Merkle parent compression happens). apply_root is
    # set by the host only when this blob is a single CTA.
    root_flag = Uint32(0)
    if cutlass.const_expr(apply_root):
        root_flag = (num_chunks == 1).to(Uint32) * ROOT

    data_u32_ptr = cute.recast_ptr(mData.iterator, dtype=Uint32)
    mDataU32 = cute.make_tensor(data_u32_ptr, cute.make_layout(cute.ceil_div(data_len, 4)))
    roots_ptr = cute.recast_ptr(mRoots.iterator, dtype=Uint32)
    mRootsView = cute.make_tensor(
        roots_ptr,
        cute.make_layout(
            (CHAINING_VALUE_SIZE_U32, roots_slot_base + num_grid_blocks),
            stride=(1, CHAINING_VALUE_SIZE_U32),
        ),
    )
    return num_groups, num_grid_blocks, root_flag, mDataU32, mRootsView


@cute.jit
def _reduce_cta_leaves(
    sLeaves,
    mRootsView,
    rKey,
    tid,
    local_bid,
    num_groups,
    num_grid_blocks,
    roots_slot_base,
    num_consumer_threads: cutlass.Constexpr[int],
    apply_root: cutlass.Constexpr[bool],
):
    """Reduce this CTA's leaves to one root and publish it (shared tail).

    ``num_groups`` is the blob's leaf count (one leaf per thread-coarsening
    group; == the chunk count at k=1). ``tid`` may exceed the leaf count (the
    TMA kernel's producer threads also arrive here); every thread
    participates for the barriers.
    """
    # Determine actual number of leaves in this block. The Python reference
    # zero-pads any non-empty trailing data up to a full chunk, so every chunk
    # counted by the ceiling division (including a partial last chunk of any
    # size, even < 64 bytes) contributes one leaf.
    is_last_block = local_bid == num_grid_blocks - 1
    num_leaves = Int32(num_consumer_threads)
    if is_last_block:
        groups_in_this_block = num_groups % num_consumer_threads
        if groups_in_this_block != 0:
            num_leaves = groups_in_this_block

    # Reduce into a Merkle Tree (all threads participate for the barriers).
    # Choose algorithm based on whether num_leaves is a power of 2.
    if not is_last_block:
        # Non-last blocks always have power-of-2 leaves
        compute_perfect_mt(sLeaves, Int32(num_consumer_threads), rKey, tid, False)
    elif (num_leaves & (num_leaves - 1)) == 0:
        # Last block, power of 2: use perfect merkle tree. If this is the only
        # block, the final parent compression must apply ROOT.
        compute_perfect_mt(sLeaves, num_leaves, rKey, tid, apply_root)
    else:
        # Last block, not a power of 2: use BLAKE3's merkle tree structure.
        compute_blake_mt(sLeaves, num_leaves, rKey, tid, apply_root)

    # Copy the root to the output (use first 8 threads)
    if tid < CHAINING_VALUE_SIZE_U32:
        mRootsView[tid, roots_slot_base + local_bid] = sLeaves[tid, 0]


@cute.jit
def _num_roots_blocks(
    data_len,
    num_consumer_threads: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
):
    """CTAs a blob of ``data_len`` bytes needs: one leaf group per consumer."""
    num_chunks = Int32(cute.ceil_div(data_len, chunk_size))
    num_groups = num_chunks
    if cutlass.const_expr(chunks_per_thread > 1):
        num_groups = Int32(num_chunks // chunks_per_thread)
    return cute.ceil_div(num_groups, num_consumer_threads)


@cute.kernel
def merkle_tree_roots_kernel(
    tma_atom: cute.CopyAtom,
    mA_coord: cute.Tensor,
    mData: cute.Tensor,
    mRoots: cute.Tensor,
    mKey: cute.Tensor,
    mStats: cute.Tensor,
    mScales: cute.Tensor,
    mCounter: cute.Tensor,
    data_len: Int64,
    sa_layout_staged: cute.ComposedLayout,
    leaves_layout: cute.ComposedLayout,
    num_consumer_threads: cutlass.Constexpr[int],
    num_pipe_stages: cutlass.Constexpr[int],
    thread_load_size: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
    mad_rot: cutlass.Constexpr[bool],
    apply_root: cutlass.Constexpr[bool],
    has_full_chunks: cutlass.Constexpr[bool],
    fused_tail: cutlass.Constexpr[bool],
    with_stats: cutlass.Constexpr[bool],
):
    """One blob per grid: every CTA hashes its slice of ``mData``."""
    use_dual, _, _, _, _ = _derived_config(num_consumer_threads, thread_load_size, chunk_size)
    storage, sStats, rKey = _stage1_cta_setup(
        mKey,
        sa_layout_staged,
        leaves_layout,
        num_pipe_stages,
        use_dual,
        _stats_smem_floats(num_consumer_threads, chunk_size, chunks_per_thread, with_stats),
    )
    bidx, _, _ = cute.arch.block_idx()
    _blob_roots(
        storage,
        sStats,
        rKey,
        tma_atom,
        mA_coord,
        mData,
        mRoots,
        mStats,
        mScales,
        data_len,
        bidx,
        Int32(0),
        mCounter,
        sa_layout_staged,
        leaves_layout,
        num_consumer_threads,
        num_pipe_stages,
        thread_load_size,
        chunk_size,
        chunks_per_thread,
        mad_rot,
        apply_root,
        has_full_chunks,
        fused_tail,
        with_stats,
    )


@cute.kernel
def merkle_tree_roots_pair_kernel(
    tma_atom_codes: cute.CopyAtom,
    mA_coord_codes: cute.Tensor,
    mCodes: cute.Tensor,
    tma_atom_scales: cute.CopyAtom,
    mA_coord_scales: cute.Tensor,
    mScales: cute.Tensor,
    mRoots: cute.Tensor,
    mKey: cute.Tensor,
    mStats: cute.Tensor,
    codes_len: Int64,
    scales_len: Int64,
    sa_layout_staged: cute.ComposedLayout,
    leaves_layout: cute.ComposedLayout,
    num_consumer_threads: cutlass.Constexpr[int],
    num_pipe_stages: cutlass.Constexpr[int],
    thread_load_size: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
    codes_mad_rot: cutlass.Constexpr[bool],
    scales_mad_rot: cutlass.Constexpr[bool],
    codes_tree: cutlass.Constexpr,
    scales_tree: cutlass.Constexpr,
    with_stats: cutlass.Constexpr[bool],
):
    """Both activation blobs in one grid, partitioned by ``blockIdx``.

    The first ``codes_blocks`` CTAs hash the codes blob -- and, on the
    activation path, reduce its commit stats -- and the rest hash the scales
    blob with a blob-local block index. The two blobs differ in length, tree
    depth and TMA descriptor, so the partitions are separate instantiations of
    ``_blob_roots`` under a block-uniform branch, and each blob's roots land in
    its own slice of the workspace: the codes tree owns the first
    ``codes_blocks`` 32-byte slots and the scales tree the following ones.

    ``codes_tree`` / ``scales_tree`` are that blob's ``(apply_root,
    has_full_chunks)`` compile-time flags.
    """
    use_dual, _, _, _, _ = _derived_config(num_consumer_threads, thread_load_size, chunk_size)
    storage, sStats, rKey = _stage1_cta_setup(
        mKey,
        sa_layout_staged,
        leaves_layout,
        num_pipe_stages,
        use_dual,
        _stats_smem_floats(num_consumer_threads, chunk_size, chunks_per_thread, with_stats),
    )
    bidx, _, _ = cute.arch.block_idx()
    codes_blocks = _num_roots_blocks(codes_len, num_consumer_threads, chunk_size, chunks_per_thread)
    # The pair path never fuses stage 2 (the launch only disappears when BOTH
    # blobs' trees fit, which production shapes never reach); mRoots stands in
    # for the never-touched arrival counter.
    if bidx < codes_blocks:
        _blob_roots(
            storage,
            sStats,
            rKey,
            tma_atom_codes,
            mA_coord_codes,
            mCodes,
            mRoots,
            mStats,
            mScales,
            codes_len,
            bidx,
            Int32(0),
            mRoots,
            sa_layout_staged,
            leaves_layout,
            num_consumer_threads,
            num_pipe_stages,
            thread_load_size,
            chunk_size,
            chunks_per_thread,
            codes_mad_rot,
            codes_tree[0],
            codes_tree[1],
            False,
            with_stats,
        )
    else:
        # The scales blob is its own message: it carries no stats, and the
        # companion-blob argument is dead once with_stats is False.
        _blob_roots(
            storage,
            sStats,
            rKey,
            tma_atom_scales,
            mA_coord_scales,
            mScales,
            mRoots,
            mStats,
            mScales,
            scales_len,
            bidx - codes_blocks,
            codes_blocks,
            mRoots,
            sa_layout_staged,
            leaves_layout,
            num_consumer_threads,
            num_pipe_stages,
            thread_load_size,
            chunk_size,
            chunks_per_thread,
            scales_mad_rot,
            scales_tree[0],
            scales_tree[1],
            False,
            False,
        )


@cute.jit
def _sync_cta_setup(
    mKey,
    num_consumer_threads: cutlass.Constexpr[int],
    stats_smem_floats: cutlass.Constexpr[int],
):
    """Leaves buffer and key registers for the sync kernels (no TMA ring).

    The only smem is the leaves buffer, in a plain layout (the sync path has
    no swizzled staging to match), plus the sub-block stats staging region
    when ``stats_smem_floats`` asks for one.
    """
    smem = cutlass.utils.SmemAllocator()
    sBuf = smem.allocate_tensor(
        Uint32,
        cute.make_layout(num_consumer_threads * CHAINING_VALUE_SIZE_U32),
        byte_alignment=16,
    )
    sLeaves = cute.make_tensor(
        sBuf.iterator,
        cute.make_layout(
            (CHAINING_VALUE_SIZE_U32, num_consumer_threads),
            stride=(1, CHAINING_VALUE_SIZE_U32),
        ),
    )
    sStats = None
    if cutlass.const_expr(stats_smem_floats > 0):
        sStats = smem.allocate_tensor(
            Float32, cute.make_layout(stats_smem_floats), byte_alignment=16
        )
    key_ptr = cute.recast_ptr(mKey.iterator, dtype=Uint32)
    mKeyWords = cute.make_tensor(key_ptr, cute.make_layout(CHAINING_VALUE_SIZE_U32))
    rKey = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32, Uint32)
    for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
        rKey[i] = mKeyWords[i]
    return sLeaves, sStats, rKey


# noqa C901: the whole synchronous hash loop. Its branches are the bounds
# checks of the trailing partial chunk and the compile-time stats dispatch,
# which must stay in one traced body.
@cute.jit
def _blob_roots_sync(  # noqa: C901
    sLeaves,
    sStats,
    rKey,
    mData: cute.Tensor,
    mRoots: cute.Tensor,
    mStats: cute.Tensor,
    mScales: cute.Tensor,
    data_len: Int64,
    local_bid,
    roots_slot_base,
    num_consumer_threads: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
    apply_root: cutlass.Constexpr[bool],
    with_stats: cutlass.Constexpr[bool],
):
    """No-TMA, no-pipeline variant of ``_blob_roots`` (same digest and stats).

    One thread per leaf group, fully synchronous: no producer warpgroup, no
    smem staging ring, no mbarriers. Each thread reads its own chunks straight
    from global memory (128-bit vectorized loads for in-bounds blocks, bounded
    word/byte loads with zero padding for the trailing partial chunk) and
    hashes them in registers -- under coarsening the k sub-chunk CVs fold via
    the same register-resident subtree as the TMA path; the CTA then reduces
    its leaves with the same Merkle tail as the TMA kernel.

    On the activation path the thread also reduces the commit stats of the
    chunks it hashes, straight from the message-block registers: the carry
    window is one 64-byte block (the TMA path's smem window does not exist
    here), which folds a stats block's eight message sums into the same
    balanced in-order tree. Sub-block leaves stage per-message-block pairs in
    ``sStats`` for the shared post-barrier fold, exactly as on the TMA path.

    Note the loads are inherently uncoalesced across a warp (lanes stride by
    ``chunks_per_thread * chunk_size`` bytes) and nothing overlaps load
    latency with compute -- the trade that loses to the TMA pipeline on large
    payloads but wins on small (decode-size) ones, where the pipeline is pure
    overhead.
    """
    tidx, _, _ = cute.arch.thread_idx()

    num_groups, num_grid_blocks, root_flag, mDataU32, mRootsView = _blob_geometry(
        mData,
        mRoots,
        data_len,
        roots_slot_base,
        num_consumer_threads,
        chunk_size,
        chunks_per_thread,
        apply_root,
    )

    small_stats: cutlass.Constexpr = _small_leaf_stats(chunk_size, with_stats)
    scale_units_per_chunk: cutlass.Constexpr = chunk_size // MSG_BLOCK_SIZE
    stats_blocks_per_chunk: cutlass.Constexpr = chunk_size // _STATS_CODE_BYTES
    # Word view of the companion scales blob (see _blob_roots); dead unless
    # with_stats, when only the codes partition is compiled with it.
    scales_u32_ptr = cute.recast_ptr(mScales.iterator, dtype=Uint32)
    mScalesU32 = cute.make_tensor(scales_u32_ptr, cute.make_layout(cute.ceil_div(data_len, 16)))
    num_stats_blocks = Int32(data_len // _STATS_CODE_BYTES)

    group_idx = local_bid * num_consumer_threads + tidx
    if group_idx < num_groups:
        rChainingValue = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32, Uint32)
        rCVs = cute.make_rmem_tensor(
            chunks_per_thread * CHAINING_VALUE_SIZE_U32 if chunks_per_thread > 1 else 1,
            Uint32,
        )
        rBlock = cute.make_rmem_tensor(MSG_BLOCK_SIZE_U32, Uint32)
        num_blocks_per_chunk = chunk_size // MSG_BLOCK_SIZE

        if cutlass.const_expr(with_stats):
            # Clamp target for the scales of code bytes that do not exist.
            last_scale_unit = Int64(data_len // MSG_BLOCK_SIZE - 1)
            if cutlass.const_expr(not small_stats):
                rAcc = cute.make_rmem_tensor(_MSG_CARRY_LEVELS, Float32)
                for level in cutlass.range_constexpr(_MSG_CARRY_LEVELS):
                    rAcc[level] = Float32(0.0)
                rMax = cute.make_rmem_tensor(1, Float32)
                rMax[0] = Float32(0.0)

        for j in cutlass.range_constexpr(chunks_per_thread):
            for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
                rChainingValue[i] = rKey[i]
            chunk = group_idx * chunks_per_thread + j
            chunk_byte0 = Int64(chunk) * chunk_size
            # Dynamic loop (unroll=1): keeps the code size chunk-size-
            # independent (a constexpr loop would inline one ~450-instruction
            # compress body per block -- 64 copies at chunk_size=4096).
            for block_idx in cutlass.range(num_blocks_per_chunk, unroll=1):
                block_byte0 = chunk_byte0 + block_idx * MSG_BLOCK_SIZE
                if block_byte0 + MSG_BLOCK_SIZE <= data_len:
                    # Whole 64-byte block in bounds: four 128-bit vector loads.
                    for i in cutlass.range_constexpr(MSG_BLOCK_SIZE_U32 // 4):
                        src4 = cute.local_tile(mDataU32, (4,), (block_byte0 // 16 + i,))
                        dst4 = cute.local_tile(rBlock, (4,), (i,))
                        cute.autovec_copy(src4, dst4)
                else:
                    # Trailing partial block: bounded word/byte loads, zero-
                    # padded (the reference pads trailing data to a whole
                    # chunk).
                    for w in cutlass.range_constexpr(MSG_BLOCK_SIZE_U32):
                        byte0 = block_byte0 + w * 4
                        word = Uint32(0)
                        if byte0 + 4 <= data_len:
                            word = Uint32(mDataU32[byte0 >> 2])
                        elif byte0 < data_len:
                            remaining = data_len - byte0
                            for b in cutlass.range_constexpr(3):
                                if b < remaining:
                                    word = Uint32(word | (mData[byte0 + b].to(Uint32) << (8 * b)))
                        rBlock[w] = word

                if cutlass.const_expr(with_stats):
                    # Issued before the compression that hides its latency.
                    rScales = cute.make_rmem_tensor(_SCALE_WORDS_PER_MSG_BLOCK, Uint32)
                    scale_unit = Int64(chunk) * scale_units_per_chunk + block_idx
                    cute.autovec_copy(
                        cute.local_tile(
                            mScalesU32,
                            (_SCALE_WORDS_PER_MSG_BLOCK,),
                            (cutlass.min(scale_unit, last_scale_unit),),
                        ),
                        rScales,
                    )

                # The host never dispatches the rotate offload with sync_loads.
                _compress_block(
                    rBlock, rChainingValue, chunk, block_idx, root_flag, chunk_size, mad_rot=False
                )

                if cutlass.const_expr(with_stats):
                    msg_ssq, msg_max = _msg_block_stats(rBlock, rScales, MSG_BLOCK_SIZE_U32)
                    if cutlass.const_expr(small_stats):
                        rPark = cute.make_rmem_tensor(2, Float32)
                        rPark[0] = msg_ssq
                        rPark[1] = msg_max
                        cta_msg = (
                            tidx * (chunks_per_thread * scale_units_per_chunk)
                            + j * scale_units_per_chunk
                            + block_idx
                        )
                        cute.autovec_copy(rPark, cute.local_tile(sStats, (2,), (cta_msg,)))
                    else:
                        # One message block per carry window (a stats block
                        # spans eight of them, whatever the leaf size).
                        window = block_idx & (_STATS_MSG_BLOCKS - 1)
                        ones = _window_ones(window, _MSG_CARRY_LEVELS)
                        sumsq = _stats_carry(rAcc, msg_ssq, ones, _MSG_CARRY_LEVELS)
                        first = (window == 0).to(Uint32)
                        absmax = cute.arch.fmax(_select_f32(first, Float32(0.0), rMax[0]), msg_max)
                        rMax[0] = absmax
                        blk = chunk * stats_blocks_per_chunk + (block_idx >> _MSG_CARRY_LEVELS)
                        if ones[_MSG_CARRY_LEVELS] != 0 and blk < num_stats_blocks:
                            rSt = cute.make_rmem_tensor(2, Float32)
                            rSt[0] = sumsq
                            rSt[1] = absmax.to(cutlass.BFloat16).to(Float32)
                            cute.autovec_copy(rSt, cute.local_tile(mStats, (2,), (blk,)))

            if cutlass.const_expr(chunks_per_thread > 1):
                for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
                    rCVs[j * CHAINING_VALUE_SIZE_U32 + i] = rChainingValue[i]

        if cutlass.const_expr(chunks_per_thread > 1):
            _fold_private_subtree(rCVs, rChainingValue, rKey, chunks_per_thread)

        for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
            sLeaves[i, tidx] = rChainingValue[i]

    cute.arch.barrier()

    if cutlass.const_expr(small_stats):
        _fold_small_leaf_stats(
            sStats,
            mStats,
            tidx,
            local_bid,
            num_stats_blocks,
            num_consumer_threads * chunks_per_thread * chunk_size // _STATS_CODE_BYTES,
        )

    _reduce_cta_leaves(
        sLeaves,
        mRootsView,
        rKey,
        tidx,
        local_bid,
        num_groups,
        num_grid_blocks,
        roots_slot_base,
        num_consumer_threads,
        apply_root,
    )


@cute.kernel
def merkle_tree_roots_sync_kernel(
    mData: cute.Tensor,
    mRoots: cute.Tensor,
    mKey: cute.Tensor,
    mStats: cute.Tensor,
    mScales: cute.Tensor,
    data_len: Int64,
    num_consumer_threads: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
    apply_root: cutlass.Constexpr[bool],
    with_stats: cutlass.Constexpr[bool],
):
    """One blob per grid, hashed by the sync (no-TMA) path."""
    sLeaves, sStats, rKey = _sync_cta_setup(
        mKey,
        num_consumer_threads,
        _stats_smem_floats(num_consumer_threads, chunk_size, chunks_per_thread, with_stats),
    )
    bidx, _, _ = cute.arch.block_idx()
    _blob_roots_sync(
        sLeaves,
        sStats,
        rKey,
        mData,
        mRoots,
        mStats,
        mScales,
        data_len,
        bidx,
        Int32(0),
        num_consumer_threads,
        chunk_size,
        chunks_per_thread,
        apply_root,
        with_stats,
    )


@cute.kernel
def merkle_tree_roots_sync_pair_kernel(
    mCodes: cute.Tensor,
    mScales: cute.Tensor,
    mRoots: cute.Tensor,
    mKey: cute.Tensor,
    mStats: cute.Tensor,
    codes_len: Int64,
    scales_len: Int64,
    num_consumer_threads: cutlass.Constexpr[int],
    chunk_size: cutlass.Constexpr[int],
    chunks_per_thread: cutlass.Constexpr[int],
    codes_tree: cutlass.Constexpr,
    scales_tree: cutlass.Constexpr,
    with_stats: cutlass.Constexpr[bool],
):
    """Both blobs in one grid via the sync path, stats riding the codes
    partition exactly as on the TMA pair kernel."""
    sLeaves, sStats, rKey = _sync_cta_setup(
        mKey,
        num_consumer_threads,
        _stats_smem_floats(num_consumer_threads, chunk_size, chunks_per_thread, with_stats),
    )
    bidx, _, _ = cute.arch.block_idx()
    codes_blocks = _num_roots_blocks(codes_len, num_consumer_threads, chunk_size, chunks_per_thread)
    if bidx < codes_blocks:
        _blob_roots_sync(
            sLeaves,
            sStats,
            rKey,
            mCodes,
            mRoots,
            mStats,
            mScales,
            codes_len,
            bidx,
            Int32(0),
            num_consumer_threads,
            chunk_size,
            chunks_per_thread,
            codes_tree[0],
            with_stats,
        )
    else:
        # The scales blob is its own message: it carries no stats, and the
        # companion-blob argument is dead once with_stats is False.
        _blob_roots_sync(
            sLeaves,
            sStats,
            rKey,
            mScales,
            mRoots,
            mStats,
            mScales,
            scales_len,
            bidx - codes_blocks,
            codes_blocks,
            num_consumer_threads,
            chunk_size,
            chunks_per_thread,
            scales_tree[0],
            False,
        )
