"""CuTe DSL Merkle reduction over the per-CTA roots.

Each CTA reduces ``leaves_per_mt_block`` stage-1 roots to one output root.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32

from ._blake3 import CHAINING_VALUE_SIZE_U32
from ._merkle_tree_utils import compute_blake_mt, compute_perfect_mt


@cute.jit
def _mt_group_leaves(leaves_per_mt_block: cutlass.Constexpr[int]):
    """Allocate the CTA's leaf buffer: 8 rows of ``leaves_per_mt_block`` words.

    Allocated once per CTA, outside any blob partitioning, since every blob in
    the grid reduces groups of the same compile-time size.
    """
    smem = cutlass.utils.SmemAllocator()
    return smem.allocate_tensor(
        Uint32,
        cute.make_layout(
            (CHAINING_VALUE_SIZE_U32, leaves_per_mt_block),
            stride=(leaves_per_mt_block, 1),
        ),
        byte_alignment=16,
    )


@cute.jit
def _mt_group_root(
    mRoots,
    mKey,
    sLeaves,
    num_blocks,
    slot_base,
    group_idx,
    tid,
    leaves_per_mt_block: cutlass.Constexpr[int],
    is_single_group: cutlass.Constexpr[bool],
):
    """Reduce one group of a blob's stage-1 roots to a single root.

    ``slot_base`` is the blob's first 32-byte root slot in the shared
    workspace and ``group_idx`` the group's index within that blob, so both
    the leaves read and the root written stay inside the blob's slice. The
    group's own leaf count -- and hence whether BLAKE3's ragged tree is needed
    -- follows from the blob's ``num_blocks``.
    """
    n_groups = cute.ceil_div(num_blocks, leaves_per_mt_block)
    remainder_group_size = num_blocks % leaves_per_mt_block
    is_remainder_group = group_idx == n_groups - 1 and remainder_group_size > 0
    num_leaves = Int32(leaves_per_mt_block)
    if is_remainder_group:
        num_leaves = remainder_group_size
    offset = slot_base + group_idx * leaves_per_mt_block

    roots_ptr = cute.recast_ptr(mRoots.iterator, dtype=Uint32)
    key_ptr = cute.recast_ptr(mKey.iterator, dtype=Uint32)
    # Global is laid out as follows (each leaf's 8 words are stored contiguously):
    # [leaf0_w0][leaf0_w1]...[leaf0_w7][leaf1_w0][leaf1_w1]...[leaf1_w7]...
    mLeaves = cute.make_tensor(
        roots_ptr,
        cute.make_layout(
            (CHAINING_VALUE_SIZE_U32, offset + num_leaves),
            stride=(1, CHAINING_VALUE_SIZE_U32),
        ),
    )
    mKeyWords = cute.make_tensor(key_ptr, cute.make_layout(CHAINING_VALUE_SIZE_U32))

    rKey = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32, Uint32)
    for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
        rKey[i] = mKeyWords[i]

    if tid < num_leaves:
        for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
            sLeaves[i, tid] = mLeaves[i, offset + tid]
    cute.arch.barrier()

    # Compute our Merkle Tree's root.
    # If we're a normal (=not remainder) group, we can use compute_perfect_mt.
    use_blake_mt = is_remainder_group and (num_leaves & (num_leaves - 1)) != 0

    if use_blake_mt:
        compute_blake_mt(sLeaves, num_leaves, rKey, tid, is_single_group)
    else:
        compute_perfect_mt(sLeaves, num_leaves, rKey, tid, is_single_group)
    cute.arch.barrier()

    # Copy the root back from smem -> gmem, into the blob's own slot
    # ``group_idx`` (slot 0 of its slice when the blob has a single group, so
    # that root is already the blob's digest).
    if tid < CHAINING_VALUE_SIZE_U32:
        mRootsFlat = cute.make_tensor(
            roots_ptr,
            cute.make_layout((slot_base + n_groups) * CHAINING_VALUE_SIZE_U32),
        )
        base_offset = (slot_base + group_idx) * CHAINING_VALUE_SIZE_U32
        mRootsFlat[base_offset + tid] = sLeaves[tid, 0]


@cute.kernel
def compute_blake_mt_kernel(
    mRoots,  # flat Int32 tensor viewed as u32 words
    mKey,  # flat Int32 tensor (8 words)
    num_blocks,  # how many blocks do we need to reduce in total?
    leaves_per_mt_block: cutlass.Constexpr[int],
    is_single_block: cutlass.Constexpr[bool],
):
    """One blob per grid: every CTA reduces one group of its roots."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    _mt_group_root(
        mRoots,
        mKey,
        _mt_group_leaves(leaves_per_mt_block),
        num_blocks,
        Int32(0),
        bid,
        tid,
        leaves_per_mt_block,
        is_single_block,
    )


@cute.kernel
def compute_blake_mt_pair_kernel(
    mRoots,  # flat Int32 tensor viewed as u32 words
    mKey,  # flat Int32 tensor (8 words)
    codes_blocks,  # stage-1 roots of the codes blob
    scales_blocks,  # stage-1 roots of the scales blob
    leaves_per_mt_block: cutlass.Constexpr[int],
    codes_single_group: cutlass.Constexpr[bool],
    scales_single_group: cutlass.Constexpr[bool],
):
    """Both blobs' groups in one grid, partitioned by ``blockIdx``.

    The blobs reduce different numbers of leaves and only the one whose stage 2
    leaves a single group applies ROOT, so the partitions are separate
    instantiations under a block-uniform branch.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    sLeaves = _mt_group_leaves(leaves_per_mt_block)
    codes_groups = cute.ceil_div(codes_blocks, leaves_per_mt_block)
    if bid < codes_groups:
        _mt_group_root(
            mRoots,
            mKey,
            sLeaves,
            codes_blocks,
            Int32(0),
            bid,
            tid,
            leaves_per_mt_block,
            codes_single_group,
        )
    else:
        _mt_group_root(
            mRoots,
            mKey,
            sLeaves,
            scales_blocks,
            codes_blocks,
            bid - codes_groups,
            tid,
            leaves_per_mt_block,
            scales_single_group,
        )
