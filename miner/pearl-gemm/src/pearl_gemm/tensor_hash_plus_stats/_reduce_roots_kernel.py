"""CuTe DSL final reduction of per-MT-block roots.

One CTA per blob reduces its ``num_leaves`` roots to the digest at the start
of that blob's ``roots`` slice.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Uint32

from ._blake3 import CHAINING_VALUE_SIZE_U32
from ._merkle_tree_utils import compute_blake_mt, compute_perfect_mt


@cute.kernel
def reduce_roots_kernel(
    mRoots,  # flat Int32 tensor viewed as u32 words
    mKey,  # flat Int32 tensor (8 words)
    first_slot,  # first 32-byte root slot of the blob CTA 0 reduces
    first_leaves,  # how many leaves that blob reduces
    second_slot,  # the same pair for CTA 1, when the grid has two
    second_leaves,
    num_threads: cutlass.Constexpr[int],
):
    """Reduce one blob's stage-2 group roots per CTA, in place at its slice.

    A blob whose stage 2 left a single group passes ``num_leaves == 1``: the
    reduction is then empty and the CTA copies the blob's digest onto itself,
    so a two-CTA grid needs only one of the blobs to have several groups.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    slot_base = first_slot
    num_leaves = first_leaves
    if bid == 1:
        slot_base = second_slot
        num_leaves = second_leaves

    roots_ptr = cute.recast_ptr(mRoots.iterator, dtype=Uint32)
    key_ptr = cute.recast_ptr(mKey.iterator, dtype=Uint32)
    # GMEM viewed as an (8, num_leaves) matrix starting at the blob's slice:
    # each leaf's 8 words are stored contiguously, leaves are spaced 8 apart.
    leaves_layout = cute.make_layout(
        (CHAINING_VALUE_SIZE_U32, num_leaves), stride=(1, CHAINING_VALUE_SIZE_U32)
    )
    mLeaves = cute.make_tensor(roots_ptr, leaves_layout)
    mKeyWords = cute.make_tensor(key_ptr, cute.make_layout(CHAINING_VALUE_SIZE_U32))

    # Statically-sized SMEM buffer viewed through the same (dynamic) layout as
    # the GMEM leaves.
    smem = cutlass.utils.SmemAllocator()
    sBuf = smem.allocate_tensor(
        Uint32, cute.make_layout(num_threads * CHAINING_VALUE_SIZE_U32), byte_alignment=16
    )
    sLeaves = cute.make_tensor(sBuf.iterator, leaves_layout)

    rKey = cute.make_rmem_tensor(CHAINING_VALUE_SIZE_U32, Uint32)
    for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
        rKey[i] = mKeyWords[i]

    # Each thread with a valid index copies its leaf to SMEM
    if tid < num_leaves:
        for i in cutlass.range_constexpr(CHAINING_VALUE_SIZE_U32):
            sLeaves[i, tid] = mLeaves[i, slot_base + tid]
    # Synchronize before starting the reduction
    cute.arch.barrier()

    # And run the Merkle Tree reduction
    if cute.arch.popc(num_leaves) == 1:
        compute_perfect_mt(sLeaves, num_leaves, rKey, tid, True)
    else:
        compute_blake_mt(sLeaves, num_leaves, rKey, tid, True)

    # Copy the result back from smem -> gmem
    if tid < CHAINING_VALUE_SIZE_U32:
        mRootsFlat = cute.make_tensor(
            roots_ptr,
            cute.make_layout((slot_base + num_leaves) * CHAINING_VALUE_SIZE_U32),
        )
        mRootsFlat[slot_base * CHAINING_VALUE_SIZE_U32 + tid] = sLeaves[tid, 0]
