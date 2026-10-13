"""Lottery constants, the fold step, and hit publishing shared by every
fused mixed GEMM family (SM90, SM100, SM120) and the grouped variants."""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Int64, Uint32, const_expr

from ..pow._hit_signal import HIT_RECORD_MAGIC_WORDS, HitRecordLayout
from ..protocol_constants import PEEL_COLS
from ..tensor_hash_plus_stats._blake3 import _rotr32

R2 = PEEL_COLS  # 2r peel columns
DEFAULT_LTILE_ROWS = 4  # default merged lottery tile rows; 16 is also supported
DEFAULT_LTILE_COLS = 128  # default merged tile cols
# Committed merged lottery tile geometries the kernel implements: supported
# ``ltile_rows``, and per row count the supported ``ltile_cols``. The 4-row
# family keeps its historical column set; the 16-row family exists only to
# shrink the peel proof (16x32), so it supports a single column count.
SUPPORTED_LTILE_ROWS: tuple[int, ...] = (4, 16)
SUPPORTED_LTILE_COLS: dict[int, tuple[int, ...]] = {4: (64, 128, 192, 256), 16: (32,)}
LANES = 16  # subtiles (= message words) per lottery tile
_SEXT_PAD = 17  # padded word stride: compress reads sExt[.., lane, c] bank-free
_FOLD_MUL = 0x9E3779B1

# Independent 16-byte loads in flight per lane during the hit payload copy.
_HIT_COPY_UNROLL = 8


def _fold_word(folded, word):
    """One step of the lottery word fold: ``rotr32(folded * _FOLD_MUL + word, 19)``."""
    return _rotr32(folded * Uint32(_FOLD_MUL) + word, 19)


class _HitGroup(NamedTuple):
    """The expert of a grouped (MoE) hit: the record's ``M`` is its block's
    ``rows``, ``GROUP_ID`` its ``index``, and its payload the plane rows
    ``[row0, row0 + rows)``."""

    index: Int32
    row0: Int32
    rows: Int32


class _HitPublishMixin:
    """Arch-independent lottery digest compare and hit-record publishing.

    Shared by the fused GEMMs of every architecture: only the message ->
    (tile_row, tile_column) decode differs per kernel, so callers decode
    lane-locally and hand the coordinates in. Requires the host attributes
    ``problem_m/n/k`` (``problem_n/k`` for grouped hits), ``ltile_rows/ltile_cols``,
    ``snapshot_payload``, and (dense hits) the payload byte counts.
    """

    @cute.jit
    def _digest_below_threshold(self, digest, threshold):
        """Compare two little-endian 256-bit values without divergence."""
        decided = Boolean(False)
        wins = Boolean(True)
        for word_idx in cutlass.range_constexpr(7, -1, -1):
            digest_word = digest[word_idx]
            threshold_word = Uint32(threshold[word_idx])
            if not decided:  # noqa: SIM102 (DSL trace: dynamic Boolean)
                if digest_word != threshold_word:
                    wins = digest_word < threshold_word
                    decided = Boolean(True)
        return wins

    @cute.jit
    def _hit_warp_copy(self, dst: cute.Tensor, src: cute.Tensor, lane: Int32):
        """Warp-cooperative 16-byte-vector copy of a payload plane.

        The vector loop keeps ``_HIT_COPY_UNROLL`` independent loads in
        flight per lane -- a single dependent load/store chain leaves a lone
        warp at ~1 GB/s, which for a multi-hundred-MB plane would stall the
        hit kernel for hundreds of ms. Plane bytes are multiples of 16
        (k is a multiple of 64), so there is no byte tail.
        """
        src_vec = cute.recast_tensor(src, cutlass.Uint128)
        dst_vec = cute.recast_tensor(dst, cutlass.Uint128)
        num_vecs = cute.size(src_vec)
        stride = cute.arch.WARP_SIZE * _HIT_COPY_UNROLL
        staged = cute.make_rmem_tensor(_HIT_COPY_UNROLL, cutlass.Uint128)
        for it in cutlass.range(num_vecs // stride):
            base = it * stride + lane
            for u in cutlass.range_constexpr(_HIT_COPY_UNROLL):
                staged[u] = src_vec[base + u * cute.arch.WARP_SIZE]
            for u in cutlass.range_constexpr(_HIT_COPY_UNROLL):
                dst_vec[base + u * cute.arch.WARP_SIZE] = staged[u]
        for tail_base in cutlass.range_constexpr(
            (num_vecs // stride) * stride, num_vecs, cute.arch.WARP_SIZE
        ):
            tail = tail_base + lane
            if tail < num_vecs:
                dst_vec[tail] = src_vec[tail]

    @cute.jit
    def _hit_warp_copy_rows(
        self,
        dst: cute.Tensor,
        src: cute.Tensor,
        src_off_u32: Int64,
        num_u32: Int32,
        lane: Int32,
    ):
        """``_hit_warp_copy`` of one group's rows of a plane.

        The rows start at u32 offset ``src_off_u32`` (64-bit: a group deep
        into a large plane lies past 2^31 bytes) and span ``num_u32`` words
        (32-bit: bounded by the destination's capacity), both whole 16-byte
        vectors (k is a multiple of 64).
        """
        src_vec = cute.recast_tensor(src, cutlass.Uint128)
        dst_vec = cute.recast_tensor(dst, cutlass.Uint128)
        vec_off = src_off_u32 // 4
        num_vecs = num_u32 // 4
        stride = cute.arch.WARP_SIZE * _HIT_COPY_UNROLL
        full_iters = num_vecs // stride
        staged = cute.make_rmem_tensor(_HIT_COPY_UNROLL, cutlass.Uint128)
        for it in cutlass.range(full_iters):
            base = it * stride + lane
            for u in cutlass.range_constexpr(_HIT_COPY_UNROLL):
                staged[u] = src_vec[vec_off + Int64(base + u * cute.arch.WARP_SIZE)]
            for u in cutlass.range_constexpr(_HIT_COPY_UNROLL):
                dst_vec[base + u * cute.arch.WARP_SIZE] = staged[u]
        for tail in cutlass.range(full_iters * stride + lane, num_vecs, cute.arch.WARP_SIZE):
            dst_vec[tail] = src_vec[vec_off + Int64(tail)]

    @cute.jit
    def _snapshot_group_payload(
        self,
        group: _HitGroup,
        k: Int32,
        mCodesSrc: cute.Tensor,
        mScalesSrc: cute.Tensor,
        mCodesDst: cute.Tensor,
        mScalesDst: cute.Tensor,
        lane: Int32,
    ):
        """Copy the group's rows of both planes when they fit the signal.

        Returns the ``(codes, scales)`` byte counts, zero when skipped (the
        winning group is not known at launch, so capacity is decided here).
        64-bit throughout: a group's byte offset into a large plane and a hot
        group's byte count can exceed 2^31, and the signal admits byte counts
        up to 2^32 - 1, which a signed 32-bit narrowing would turn negative.
        Only the word counts are narrowed, after the capacity check bounds
        them below 2^30.
        """
        k64 = Int64(k)
        row0 = Int64(group.row0)
        codes_bytes64 = Int64(group.rows) * k64
        scales_bytes64 = Int64(group.rows) * (k64 // 4)  # (k / 8) bf16 per row
        codes_bytes = Int64(0)
        scales_bytes = Int64(0)
        fits = Boolean(codes_bytes64 <= Int64(cute.size(mCodesDst)) * 4) & Boolean(
            scales_bytes64 <= Int64(cute.size(mScalesDst)) * 4
        )
        if fits:
            codes_bytes = codes_bytes64
            scales_bytes = scales_bytes64
            self._hit_warp_copy_rows(
                mCodesDst, mCodesSrc, (row0 * k64) // 4, Int32(codes_bytes64 // 4), lane
            )
            self._hit_warp_copy_rows(
                mScalesDst, mScalesSrc, (row0 * (k64 // 4)) // 4, Int32(scales_bytes64 // 4), lane
            )
        return codes_bytes, scales_bytes

    @cute.jit
    def _publish_hit(
        self,
        tile_row: Int32,
        tile_column: Int32,
        in_bounds: Boolean,
        pow_key_words,
        threshold_words,
        mRecord: cute.Tensor,
        mLock: cute.Tensor,
        mHashB: cute.Tensor,
        mCodesSrc: cute.Tensor | None,
        mScalesSrc: cute.Tensor | None,
        mCodesDst: cute.Tensor | None,
        mScalesDst: cute.Tensor | None,
        layer_id: Int32,
        record_hits: Int32,
        local_hit: Boolean,
        group: _HitGroup | None = None,
    ):
        """Publish one hit into the persistent signal (whole warp calls this).

        Protocol (``pow/_hit_signal.py``): ballot the PUBLISHABLE (in-bounds)
        hits -> the leader (first such lane) claims the persistent first-wins
        latch it never releases -> warp-cooperative payload snapshot ->
        record fields + magic -> system fence -> doorbell LAST. Out-of-bounds
        hits are excluded BEFORE leader election: an edge-tile lane must not
        win the ballot and mask a publishable hit in the same warp.

        ``tile_row``/``tile_column``/``in_bounds`` are the caller's lane-local
        decode; the leader's values are the published ones. ``group`` (warp
        uniform) publishes with group semantics: ``M`` is the group's rows,
        ``GROUP_ID`` its index and the payload its rows of the planes.
        """
        lane = cute.arch.lane_idx()
        # The ballot is deliberately unconditional: even when publication is
        # runtime-disabled, the digest/threshold result reaches this convergent
        # observable operation and cannot be sunk with the compression chain.
        hit_mask = cute.arch.vote_ballot_sync(local_hit & in_bounds)
        if (record_hits != 0) & (hit_mask != 0):
            leader = cute.arch.popc((hit_mask & (0 - hit_mask)) - 1)
            # Losing the first-wins claim (unconsumed pending hit / concurrent
            # launch) just drops this rare hit.
            claimed = Int32(0)
            if lane == leader:
                prev = cute.arch.atomic_cas(
                    ptr=mLock.iterator.llvm_ptr,
                    cmp=Int32(0),
                    val=Int32(1),
                    sem="acq_rel",
                    scope="gpu",
                )
                if prev == 0:
                    claimed = Int32(1)
            claimed = cute.arch.shuffle_sync(claimed, leader)
            if claimed != 0:
                if const_expr(group is None):
                    record_m = Uint32(self.problem_m)
                    group_id = Uint32(0)  # dense hits have no group; a MoE hit may have left one
                    codes_bytes = Uint32(self.codes_payload_bytes if self.snapshot_payload else 0)
                    scales_bytes = Uint32(self.scales_payload_bytes if self.snapshot_payload else 0)
                    if const_expr(self.snapshot_payload):
                        self._hit_warp_copy(mCodesDst, mCodesSrc, lane)
                        self._hit_warp_copy(mScalesDst, mScalesSrc, lane)
                else:
                    record_m = group.rows.to(Uint32)
                    group_id = group.index.to(Uint32)
                    codes_bytes = Uint32(0)
                    scales_bytes = Uint32(0)
                    if const_expr(self.snapshot_payload):
                        codes64, scales64 = self._snapshot_group_payload(
                            group,
                            self.problem_k,
                            mCodesSrc,
                            mScalesSrc,
                            mCodesDst,
                            mScalesDst,
                            lane,
                        )
                        codes_bytes = codes64.to(Uint32)
                        scales_bytes = scales64.to(Uint32)
                # Every lane fences its own payload stores to system scope,
                # then the warp syncs, so the leader's publication below
                # cannot pass them.
                cute.arch.fence_acq_rel_sys()
                cute.arch.sync_warp()
                if lane == leader:
                    mRecord[HitRecordLayout.M] = record_m
                    mRecord[HitRecordLayout.N] = Uint32(self.problem_n)
                    mRecord[HitRecordLayout.K] = Uint32(self.problem_k)
                    mRecord[HitRecordLayout.TILE_ROW] = Uint32(tile_row)
                    mRecord[HitRecordLayout.TILE_COLUMN] = Uint32(tile_column)
                    mRecord[HitRecordLayout.LTILE_ROWS] = Uint32(self.ltile_rows)
                    mRecord[HitRecordLayout.LTILE_COLS] = Uint32(self.ltile_cols)
                    mRecord[HitRecordLayout.CODES_PAYLOAD_BYTES] = codes_bytes
                    mRecord[HitRecordLayout.SCALES_PAYLOAD_BYTES] = scales_bytes
                    mRecord[HitRecordLayout.LAYER_ID] = layer_id.to(Uint32)
                    for i in cutlass.range_constexpr(8):
                        mRecord[HitRecordLayout.TARGET + i] = threshold_words[i]
                        mRecord[HitRecordLayout.HASH_A + i] = pow_key_words[i]
                        mRecord[HitRecordLayout.HASH_B + i] = mHashB[i]
                    mRecord[HitRecordLayout.GROUP_ID] = group_id
                    mRecord[HitRecordLayout.MAGIC] = Uint32(HIT_RECORD_MAGIC_WORDS[0])
                    mRecord[HitRecordLayout.MAGIC + 1] = Uint32(HIT_RECORD_MAGIC_WORDS[1])
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.store(
                        mRecord.iterator,
                        Uint32(1),
                        sem="release",
                        scope="sys",
                    )
