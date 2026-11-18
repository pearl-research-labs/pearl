"""CuTe DSL Merkle reductions over shared-memory leaves.

Both helpers reduce 8-word leaf columns to one root in column 0. ``rKey`` is
the BLAKE3 key held in registers.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Uint32

from ._blake3 import (
    CHAINING_VALUE_SIZE_U32,
    FLAGS_INNER_NODE,
    FLAGS_ROOT,
    compress_msg_block,
)


@cute.jit
def compute_perfect_mt(sLeaves, num_leaves, rKey, tid, consider_root: cutlass.Constexpr[bool]):
    """Compute a perfect Merkle Tree with ``num_leaves`` (power-of-2) leaves.

    Parent compressions stay on plain-SHF rotates (see the ``mad_rot``
    policy block in ``_blake3``).
    """
    level_size = Int32(num_leaves)
    while level_size > 1:
        rChainingValue = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32, Uint32)
        rChunk = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32 * 2, Uint32)

        for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
            rChainingValue[i] = rKey[i]

        num_pairs = level_size >> 1
        if tid < num_pairs:
            for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
                rChunk[i] = sLeaves[i, 2 * tid]
                rChunk[i + CHAINING_VALUE_SIZE_U32] = sLeaves[i, 2 * tid + 1]
        # Child reads at 2*tid and 2*tid+1 overlap parent writes by neighboring
        # threads, so every level must finish its reads before stores begin.
        cute.arch.barrier()

        if tid < num_pairs:
            flags = Uint32(FLAGS_INNER_NODE)
            # SIM102 off: the outer if is a compile-time const_expr branch, the
            # inner one is dynamic; combining them would change staging semantics.
            if cutlass.const_expr(consider_root):  # noqa: SIM102
                if tid == 0 and num_pairs == 1:
                    flags = Uint32(FLAGS_ROOT)
            compress_msg_block(rChunk, rChainingValue, Uint32(0), flags)

            # And store the parent in the leaves tensor
            for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
                sLeaves[i, tid] = rChainingValue[i]
        cute.arch.barrier()

        level_size = num_pairs


@cute.jit
def compute_blake_mt(sLeaves, num_leaves, rKey, tid, consider_root: cutlass.Constexpr[bool]):  # noqa: C901
    """Merkle Tree in BLAKE3's structure for a non-power-of-2 ``num_leaves``.

    The spec decomposes ``num_leaves`` into perfect subtrees by its binary
    expansion (largest leftmost) and folds the subtree roots right-to-left
    through a serial hash chain: ``running = H(subtree_root || running)``,
    ROOT flag on the last (leftmost) fold.

    Schedule: the original port staggered subtree starts so every subtree
    root landed on the *final* level, then thread 0 walked the whole fold
    chain serially -- ``popcount(num_leaves) - 1`` latency-bound compressions
    while the rest of the CTA idled. Here every subtree instead reduces
    bottom-up from level 1, so the subtree for bit ``b`` finishes after level
    ``b``, and a dedicated fold thread consumes each root one level after it
    lands -- its fold compress for bit ``b`` runs in the compute phase of
    level ``b + 1``, concurrently with the surviving subtrees' compressions.
    Only the final fold (the largest subtree, ready after the last level)
    remains on the critical path: total depth is ``log2(largest_subtree) + 1``
    compressions, independent of popcount.

    Preconditions: ``num_leaves >= 3`` and not a power of two (callers use
    ``compute_perfect_mt`` otherwise); the CTA has more than
    ``num_leaves // 2`` threads (the fold thread is ``num_leaves >> 1``, the
    first thread free of pair work at every level).
    """
    # Each group of threads handles a perfect tree. Figure out which group we
    # are part of: the smem read offset for this group, the number of leaves in
    # this thread's group, and the virtual thread ID within the group. The scan
    # walks set bits of num_leaves from the highest possible position downward
    # (num_leaves <= 1024, so bit 10 covers every case); unset high bits are
    # skipped, so this equals a loop that starts at ceil(log2(num_leaves)).
    offset = Int32(0)
    our_num_leaves = Int32(0)
    virtual_tid = Int32(0)
    largest_subtree = Int32(0)
    found = Boolean(False)
    for i in cutlass.range_constexpr(10, -1, -1):
        bit_value = 1 << i
        if (num_leaves & bit_value) != 0:
            if largest_subtree == 0:
                largest_subtree = Int32(bit_value)
            if not found:
                if offset + bit_value > 2 * tid:
                    our_num_leaves = Int32(bit_value)
                    virtual_tid = tid - (offset >> 1)
                    found = Boolean(True)
                else:
                    offset = offset + bit_value  # we are part of this tree

    # Pair workers occupy tids [0, num_leaves >> 1); this thread rides along
    # with the reduction, folding subtree roots as they complete. The running
    # fold root lives in smem, at the column of the last-consumed subtree
    # (dead storage once read), so no registers stay live across the loop.
    is_fold_thread = tid == (num_leaves >> 1)
    running_col = Int32(-1)

    lvl = Int32(1)
    while (largest_subtree >> lvl) > 0:
        rChunk = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32 * 2, Uint32)
        rChainingValue = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32, Uint32)

        num_pairs = our_num_leaves >> lvl  # this subtree's pairs at this level (0 when done)

        # Every active thread compresses sLeaves[src0] || sLeaves[src1] into
        # sLeaves[dst]. Pair workers and the fold thread share ONE branch so
        # the fold compress executes in lockstep with the workers even when
        # they land in the same warp (a divergent fold branch would serialize
        # after the workers' compress and re-add ~1 compress of latency per
        # fold level).
        active = Boolean(False)
        src0 = Int32(0)
        src1 = Int32(0)
        dst = Int32(0)
        if virtual_tid < num_pairs:
            active = Boolean(True)
            src0 = offset + 2 * virtual_tid
            src1 = src0 + 1
            dst = offset + virtual_tid

        # The subtree for bit (lvl - 1) completed its root (at column
        # num_leaves - (num_leaves & (2^lvl - 1)), where its leaves start) on
        # the previous level; bit 0 is a bare leaf, ready from the start. The
        # fold result overwrites the consumed subtree's root column (dead to
        # every other thread), which becomes the new running-root column.
        if is_fold_thread:  # noqa: SIM102 (fold-bit test kept off the workers' path)
            if ((num_leaves >> (lvl - 1)) & 1) != 0:
                fold_col = num_leaves - (num_leaves & ((Int32(2) << (lvl - 1)) - 1))
                if running_col >= 0:
                    # Never the last fold (the largest subtree's bit is
                    # handled after the loop), so never ROOT here.
                    active = Boolean(True)
                    src0 = fold_col
                    src1 = running_col
                    dst = fold_col
                # else: first (rightmost) subtree root seeds the chain in
                # place -- no compress needed.
                running_col = fold_col

        # Read phase (race-protected by the previous level's post-compute
        # barrier).
        if active:
            for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
                rChunk[i] = sLeaves[i, src0]
                rChunk[i + CHAINING_VALUE_SIZE_U32] = sLeaves[i, src1]
        # Same race condition as in compute_perfect_mt.
        cute.arch.barrier()

        # Compute phase.
        if active:
            for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
                rChainingValue[i] = rKey[i]
            compress_msg_block(rChunk, rChainingValue, Uint32(0), Uint32(FLAGS_INNER_NODE))
            for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
                sLeaves[i, dst] = rChainingValue[i]
        cute.arch.barrier()

        lvl = lvl + 1

    # Final fold: the largest subtree's root (column 0) landed on the last
    # level; fold it with the running root of everything to its right.
    if is_fold_thread:
        rChunk = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32 * 2, Uint32)
        rChainingValue = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32, Uint32)
        for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
            rChunk[i] = sLeaves[i, 0]
            rChunk[i + CHAINING_VALUE_SIZE_U32] = sLeaves[i, running_col]
            rChainingValue[i] = rKey[i]
        flags = Uint32(FLAGS_INNER_NODE)
        if cutlass.const_expr(consider_root):
            flags = Uint32(FLAGS_ROOT)
        compress_msg_block(rChunk, rChainingValue, Uint32(0), flags)
        for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
            sLeaves[i, 0] = rChainingValue[i]
    cute.arch.barrier()
