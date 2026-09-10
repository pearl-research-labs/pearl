"""Persistent SM100 FP8 GEMM with lottery, peel, unscale, and hit publishing.

The tcgen05 UMMA accumulates in tensor memory (TMEM), so the fused phases are
homed around Quack's ``GemmSm100`` warp specialization:

- the AB-load warp appends one extra ring slot per output tile carrying the
  BF16 peel operands (aliased onto the FP8 A/B slot bytes, reduced expect-tx);
- the MMA warp runs the FP8 mainloop, publishes the pre-peel accumulator
  through a one-stage ``PipelineUmmaAsync`` handshake (UMMA commit), waits for
  the epilogue warps' fold release, then issues the BF16 peel UMMA into the
  same TMEM accumulator before the normal accumulator-pipeline commit;
- the epilogue warps load the pre-peel accumulator from TMEM (tcgen05.ld),
  fold it into lottery words in shared memory, release the peel handshake,
  hash the words with keyed BLAKE3, publish the first hit into the process's
  persistent hit signal (``pow/_hit_signal.py``), then run the TMA-store
  epilogue with the row/column unscale fused into the accumulator subtile
  loads.
"""

from functools import partial

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import quack.copy_utils as copy_utils
from cutlass import Boolean, Float32, Int32, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum, SmemPartition
from quack.epilogue.ops import EpiSmemBytes
from quack.gemm_base import NamedBarrierGemm
from quack.gemm_sm100 import GemmSm100
from quack.pipeline import PipelineUmmaAsync, make_pipeline_state
from quack.rounding import RoundingMode
from quack.tile_scheduler import TileSchedulerOptions
from quack.varlen_utils import VarlenArguments

from ..pow._hit_signal import HIT_RECORD_MAGIC_WORDS, HitRecordLayout
from ..protocol_constants import PEEL_COLS
from ..tensor_hash_plus_stats._blake3 import _rotr32
from ..tensor_hash_plus_stats._blake3_ops import SINGLE_BLOCK_KEYED_FLAGS, compress

R2 = PEEL_COLS  # 2r peel columns
DEFAULT_LTILE_ROWS = 4  # default merged lottery tile rows; 16 is also supported (see __init__)
DEFAULT_LTILE_COLS = 128  # default merged tile cols; see __init__ for the supported set
# Committed merged lottery tile geometries the kernel implements: supported
# ``ltile_rows``, and per row count the supported ``ltile_cols``. The 4-row
# family keeps its historical column set; the 16-row family exists only to
# shrink the peel proof (16x32), so it supports a single column count.
SUPPORTED_LTILE_ROWS: tuple[int, ...] = (4, 16)
SUPPORTED_LTILE_COLS: dict[int, tuple[int, ...]] = {4: (64, 128, 192, 256), 16: (32,)}
LANES = 16  # subtiles (= message words) per lottery tile
_SEXT_PAD = 17  # padded word stride: compress reads sExt[.., lane, c] bank-free
_FOLD_MUL = 0x9E3779B1

_EPI_THREADS = 128  # epilogue threads per CTA (4 warps): one hashed message each

# Independent 16-byte loads in flight per lane during the hit payload copy.
_HIT_COPY_UNROLL = 8


def cta_tile_m(tile_m: int) -> int:
    """Output rows each CTA owns: the 2-CTA pair splits mma-M 256 into 128-row CTAs."""
    return min(tile_m, 128)


def _word_of(column: int, words_per_row: int) -> int:
    """The word (within its column tile) a column feeds: its mod-8 pair class."""
    return (column >> 1) % words_per_row


def _thread_fold_cells(tiled_copy_t2r, thread: int) -> dict[int, list[tuple[int, int]]]:
    """One thread's t2r cells: per accumulator row, (register, column) pairs
    in ascending column order, from the copy's static TV layout."""
    layout_tv = tiled_copy_t2r.layout_dst_tv_tiled
    tiler_m, tiler_n = tiled_copy_t2r.tiler_mn[0], tiled_copy_t2r.tiler_mn[1]
    m_size = cute.size(tiler_m)
    by_row: dict[int, list[tuple[int, int]]] = {}
    for value in range(cute.size(layout_tv, mode=[1])):
        flat = layout_tv((thread, value))
        by_row.setdefault(tiler_m(flat % m_size), []).append((value, tiler_n(flat // m_size)))
    return {row: sorted(pairs, key=lambda p: p[1]) for row, pairs in sorted(by_row.items())}


def _check_complete_words(groups, base: int, words_per_row: int, ltile_cols: int, n_size: int):
    for group in groups:
        by_word = {}
        for _, column, slot in group:
            by_word.setdefault((column // ltile_cols, slot), []).append(column)
        for (column_tile, slot), columns in by_word.items():
            complete = [
                c
                for c in range(n_size)
                if c // ltile_cols == column_tile and _word_of(c, words_per_row) == base + slot
            ]
            assert columns == complete, (
                f"a row class must fold complete words in ascending order: {columns}"
            )


def _check_fold_thread_invariance(tiled_copy_t2r, groups, words_per_row: int, ltile_cols: int):
    covered = set()
    for thread in range(cute.size(tiled_copy_t2r.layout_dst_tv_tiled, mode=[0])):
        cells = _thread_fold_cells(tiled_copy_t2r, thread)
        assert len(cells) == len(groups), "row-class count must be thread-invariant"
        thread_base = _word_of(next(iter(cells.values()))[0][1], words_per_row)
        for pairs, ref_group in zip(cells.values(), groups, strict=True):
            for (value, column), (ref_value, ref_column, ref_slot) in zip(
                pairs, ref_group, strict=True
            ):
                assert value == ref_value, "t2r register order must be thread-invariant"
                # Relative word slot + lottery-tile, not exact column: the
                # 64-row 16dp mapping gives each thread a different mod-8
                # column-pair of the same word class, so ``column ==
                # ref_column`` would reject the layout the fold is written
                # for. Complete-word coverage is asserted separately.
                assert _word_of(column, words_per_row) - thread_base == ref_slot, (
                    "word slots must be thread-invariant"
                )
                assert column // ltile_cols == ref_column // ltile_cols, (
                    "column tiles must be thread-invariant"
                )
        for row, pairs in cells.items():
            for _, column in pairs:
                assert (row, column) not in covered, "t2r cells must not be shared"
                covered.add((row, column))
    m_size = cute.size(tiled_copy_t2r.tiler_mn[0])
    n_size = cute.size(tiled_copy_t2r.tiler_mn[1])
    assert len(covered) == m_size * n_size, "t2r threads must cover the subtile exactly once"


def _fold_register_groups(
    tiled_copy_t2r, words_per_row: int, ltile_cols: int
) -> tuple[list[list[tuple[int, int, int]]], int]:
    """Static per-row-class fold mapping of the TMEM load, in fold order.

    Every thread must fold complete lottery words. The committed t2r families
    give a thread either one accumulator row with every subtile column
    (128-row tiles, 32dp loads) or two rows whose columns are one word's
    congruence classes per row (64-row tiles, 16dp loads: the mma-M 64
    accumulator holds 16 lanes per warp block, so two threads split each row
    at mod-8 column-pair granularity). Both reduce to per-row-class
    ``(register, subtile column, word slot)`` triples in ascending column
    order, where a thread's absolute word index is its runtime word base
    (from its first column) plus the static slot. Derived (and asserted) from
    the tiled copy's static TV layout so any future atom change fails at
    compile time, not silently.
    """
    reference = _thread_fold_cells(tiled_copy_t2r, 0)
    base = _word_of(next(iter(reference.values()))[0][1], words_per_row)
    groups = [
        [(value, column, _word_of(column, words_per_row) - base) for value, column in pairs]
        for pairs in reference.values()
    ]
    slots = sorted({slot for group in groups for _, _, slot in group})
    words_per_class = len(slots)
    assert slots == list(range(words_per_class)), f"word slots must be contiguous: {slots}"
    n_size = cute.size(tiled_copy_t2r.tiler_mn[1])
    _check_complete_words(groups, base, words_per_row, ltile_cols, n_size)
    _check_fold_thread_invariance(tiled_copy_t2r, groups, words_per_row, ltile_cols)
    return groups, words_per_class


class _FusedGemmSm100(GemmSm100):
    """Quack's SM100 GEMM with lottery, peel, and unscale fused in."""

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int | None = None,
        cluster_m: int = 1,
        cluster_n: int = 1,
        *,
        snapshot_payload: bool = False,
        ltile_rows: int = DEFAULT_LTILE_ROWS,
        ltile_cols: int = DEFAULT_LTILE_COLS,
    ):
        # mma-M 256 is the 2-CTA pair (even cluster_m only); with an odd
        # cluster_m it degrades to the identical 128-row 1-CTA kernel. mma-M
        # 64 runs a 64-row 1-CTA kernel (its TMEM accumulator packs 16 lanes
        # per warp block; see _fold_register_groups).
        assert tile_m in (64, 128, 256), (
            f"SM100 fused GEMM needs tile_m 64, 128, or 256, got {tile_m}"
        )
        use_2cta_instrs = tile_m == 256 and cluster_m % 2 == 0
        mma_m = 256 if use_2cta_instrs else min(tile_m, 128)
        super().__init__(
            acc_dtype=Float32,
            a_dtype=cutlass.Float8E4M3FN,
            mma_tiler_mnk=(mma_m, tile_n) if tile_k is None else (mma_m, tile_n, tile_k),
            cluster_shape_mnk=(cluster_m, cluster_n, 1),
            use_clc_persistence=True,
            # The lottery consumes commitment-stage outputs at kernel start;
            # PDL overlap with the producer kernel would break that ordering.
            use_pdl=False,
        )
        # Override the substrate's 2-CTA auto-selection (it would pick 2-CTA
        # for mma-M 128 with an even cluster_m; the cluster then only
        # multicasts).
        self.use_2cta_instrs = use_2cta_instrs
        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.cta_m = mma_m // (2 if use_2cta_instrs else 1)
        assert self.cta_m == cta_tile_m(tile_m)
        self.tile_n = tile_n
        # Two committed tile families, both 512 elements (same effective work):
        # - 4 rows x {64,128,192,256} cols: rows (subtile {0}, grid {0..3});
        #   cols (subtile pairs {0,1} mod 8, grid {0,2,4,6}); each row
        #   contributes 4 message words (one per column-grid offset).
        # - 16 rows x 32 cols: rows (subtile {0}, grid {0..15}); cols
        #   (subtile {0..31}, grid {0}); each row folds its whole 32-column
        #   slice into 1 message word (lane = row within the tile).
        assert ltile_rows in SUPPORTED_LTILE_ROWS, f"unsupported lottery tile rows: {ltile_rows}"
        supported_cols = SUPPORTED_LTILE_COLS[ltile_rows]
        assert ltile_cols in supported_cols, (
            f"unsupported lottery tile cols for {ltile_rows} rows: {ltile_cols}"
        )
        # The 16-row family folds one contiguous 32-column row slice per word,
        # but the 64-row tile's TMEM loads split every accumulator row between
        # two threads, so no single thread can fold such a word (see
        # _fold_register_groups). Only the 4-row family fits cta_m 64.
        assert not (self.cta_m == 64 and ltile_rows != 4), (
            f"tile_m 64 supports only the 4-row lottery family, got {ltile_rows} rows"
        )
        self.ltile_rows = ltile_rows
        self.ltile_cols = ltile_cols
        # Message words each accumulator row contributes per lottery tile.
        self.words_per_row = LANES // ltile_rows
        assert tile_n % ltile_cols == 0, f"lottery needs tile_n % {ltile_cols} == 0"
        self.column_tiles = tile_n // ltile_cols
        # One message per epilogue thread: each of the cta_m accumulator rows
        # contributes its words to its lottery row's message, and epilogue
        # thread i hashes message i.
        self.msgs_cta = (self.cta_m // ltile_rows) * self.column_tiles
        assert self.msgs_cta <= _EPI_THREADS, (
            f"msgs_cta={self.msgs_cta} > {_EPI_THREADS} epilogue threads"
        )
        # The hit publish ballots whole warps, so the hashing gate is rounded
        # up to warp granularity; sub-warp tails are excluded per lane inside
        # _compress_and_publish.
        self.hash_threads = -(-self.msgs_cta // 32) * 32
        self.snapshot_payload = snapshot_payload
        # GemmTmaBase.epilogue reads this required specialization attribute.
        self.rounding_mode = RoundingMode.RN

    # Quack's stage-count hook requires the full signature; this epilogue
    # only needs its fused shared-memory byte count.
    @staticmethod
    def epi_smem_bytes(
        extra_bytes, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None
    ) -> EpiSmemBytes:
        return EpiSmemBytes(unstaged=extra_bytes)

    # No epilogue ops and RN rounding, so the stochastic-rounding seed mapping
    # is empty.
    def epi_begin_loop(self, params, epi_tensors, epi_coord) -> dict[str, cute.Tensor]:
        return {}

    def _fused_extra_smem_bytes(self) -> int:
        # Alignment slack for the fused fields ahead of the 1024-aligned
        # sD/sA/sB buffers, plus the fused buffers themselves.
        extra = 3072
        extra += self.msgs_cta * _SEXT_PAD * 4 + 128  # sExt
        extra += self.cta_m * 2 + self.tile_n * 4 + 32  # sAlA (bf16) + sAlB (f32)
        return extra

    # -- pre-peel handshake: UMMA commit by the MMA warp (full), fold release
    # by the epilogue warps (empty; both CTAs of a pair release the leader) --
    def make_prepeel_pipeline(self, cluster_layout_vmnk: cute.Layout) -> pipeline.PipelineAsync:
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_consumer_threads = self.num_epi_warps * (2 if self.use_2cta_instrs else 1)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_consumer_threads)
        return PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
            elect_one_release=True,
            # TMEM load consumers are already ordered by fence_view_async_tmem_load
            syncwarp_before_release=False,
        )

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
    def _publish_hit(
        self,
        message: Int32,
        tile_coord_mnkl,
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
    ):
        """Publish one hit into the persistent signal (whole warp calls this).

        Protocol (``pow/_hit_signal.py``): ballot the PUBLISHABLE (in-bounds)
        hits -> the leader (first such lane) claims the persistent first-wins
        latch it never releases -> warp-cooperative payload snapshot ->
        record fields + magic -> system fence -> doorbell LAST. Out-of-bounds
        hits are excluded BEFORE leader election: an edge-tile lane must not
        win the ballot and mask a publishable hit in the same warp.
        """
        lane = cute.arch.lane_idx()
        # Lane-local decode; the leader's values are the published ones.
        column_tile = message % self.column_tiles
        local_row = message // self.column_tiles
        tile_row = tile_coord_mnkl[0] * (self.cta_m // self.ltile_rows) + local_row
        tile_column = tile_coord_mnkl[1] * self.column_tiles + column_tile
        in_bounds = Boolean(tile_row * self.ltile_rows < self.problem_m) & Boolean(
            tile_column * self.ltile_cols < self.problem_n
        )
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
                if const_expr(self.snapshot_payload):
                    self._hit_warp_copy(mCodesDst, mCodesSrc, lane)
                    self._hit_warp_copy(mScalesDst, mScalesSrc, lane)
                # Every lane fences its own payload stores to system scope,
                # then the warp syncs, so the leader's publication below
                # cannot pass them.
                cute.arch.fence_acq_rel_sys()
                cute.arch.sync_warp()
                if lane == leader:
                    mRecord[HitRecordLayout.M] = Uint32(self.problem_m)
                    mRecord[HitRecordLayout.N] = Uint32(self.problem_n)
                    mRecord[HitRecordLayout.K] = Uint32(self.problem_k)
                    mRecord[HitRecordLayout.TILE_ROW] = Uint32(tile_row)
                    mRecord[HitRecordLayout.TILE_COLUMN] = Uint32(tile_column)
                    mRecord[HitRecordLayout.LTILE_ROWS] = Uint32(self.ltile_rows)
                    mRecord[HitRecordLayout.LTILE_COLS] = Uint32(self.ltile_cols)
                    mRecord[HitRecordLayout.CODES_PAYLOAD_BYTES] = Uint32(
                        self.codes_payload_bytes if self.snapshot_payload else 0
                    )
                    mRecord[HitRecordLayout.SCALES_PAYLOAD_BYTES] = Uint32(
                        self.scales_payload_bytes if self.snapshot_payload else 0
                    )
                    mRecord[HitRecordLayout.LAYER_ID] = layer_id.to(Uint32)
                    for i in cutlass.range_constexpr(8):
                        mRecord[HitRecordLayout.TARGET + i] = threshold_words[i]
                        mRecord[HitRecordLayout.HASH_A + i] = pow_key_words[i]
                        mRecord[HitRecordLayout.HASH_B + i] = mHashB[i]
                    mRecord[HitRecordLayout.MAGIC] = Uint32(HIT_RECORD_MAGIC_WORDS[0])
                    mRecord[HitRecordLayout.MAGIC + 1] = Uint32(HIT_RECORD_MAGIC_WORDS[1])
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.store(
                        mRecord.iterator,
                        Uint32(1),
                        sem="release",
                        scope="sys",
                    )

    @cute.jit
    def _compress_and_publish(
        self,
        sExt: cute.Tensor,
        chaining_value,
        threshold,
        mRecord: cute.Tensor,
        mLock: cute.Tensor,
        mHashB: cute.Tensor,
        mCodesSrc: cute.Tensor | None,
        mScalesSrc: cute.Tensor | None,
        mCodesDst: cute.Tensor | None,
        mScalesDst: cute.Tensor | None,
        layer_id: Int32,
        record_hits: Int32,
        tile_coord_mnkl,
        tidx: Int32,
    ):
        """Hash one staged message per epilogue thread and publish the winner."""
        if tidx < self.hash_threads:
            # The publish ballot needs a convergent warp, so a sub-warp
            # message tail keeps its whole warp hashing: excess lanes rehash
            # the last message and are excluded from the ballot below.
            # TODO: skip keyed BLAKE3 on dummy lanes (``if is_message:
            # compress``) while keeping them in the ballot. ``msgs_cta=16``
            # rounds ``hash_threads`` to 32, so those lanes currently rehash
            # the last real message. Known 1-warp warmup cost; digests stay
            # correct.
            if const_expr(self.msgs_cta % 32 == 0):
                is_message = Boolean(True)
                message = tidx
            else:
                is_message = Boolean(tidx < self.msgs_cta)
                message = cutlass.min(tidx, self.msgs_cta - 1)
            words = [sExt[message, column].to(Uint32) for column in range(LANES)]
            digest = compress(
                list(chaining_value),
                words,
                64,
                SINGLE_BLOCK_KEYED_FLAGS,
            )
            local_hit = self._digest_below_threshold(digest, threshold) & is_message
            self._publish_hit(
                message,
                tile_coord_mnkl,
                chaining_value,
                threshold,
                mRecord,
                mLock,
                mHashB,
                mCodesSrc,
                mScalesSrc,
                mCodesDst,
                mScalesDst,
                layer_id,
                record_hits,
                local_hit,
            )

    @cute.jit
    def _stage_alpha_slices(
        self,
        mAlA: cute.Tensor,
        mAlB: cute.Tensor,
        sAlA: cute.Tensor,
        sAlB: cute.Tensor,
        tile_coord_mnkl,
        tidx: Int32,
    ):
        """Stage this tile's row and column scales with asynchronous copies."""
        alpha_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        for global_scale, shared_scale, coordinate_mode, tile_extent in (
            (mAlA, sAlA, 0, self.cta_m),
            (mAlB, sAlB, 1, self.tile_n),
        ):
            thread_copy = copy_utils.tiled_copy_1d(
                global_scale.element_type,
                alpha_threads,
                32 // global_scale.element_type.width,
                is_async=True,
            ).get_slice(tidx)
            global_tile = cute.local_tile(
                global_scale,
                (tile_extent,),
                (tile_coord_mnkl[coordinate_mode],),
            )
            copy_source = thread_copy.partition_S(global_tile)
            copy_destination = thread_copy.partition_D(shared_scale)
            copy_coordinates = thread_copy.partition_S(cute.make_identity_tensor(tile_extent))
            # Skipped shared-memory entries feed only TMA-clipped output
            # elements, so stale values at a partial problem edge are safe.
            valid_elements = cutlass.min(
                cute.size(global_scale) - tile_coord_mnkl[coordinate_mode] * tile_extent,
                tile_extent,
            )
            for element in cutlass.range(
                cute.size(copy_destination.shape[1]),
                unroll_full=True,
            ):
                if copy_coordinates[0, element] < tile_extent:
                    predicate = cute.make_rmem_tensor(1, Boolean)
                    predicate[0] = copy_coordinates[0, element] < valid_elements
                    cute.copy(
                        thread_copy,
                        copy_source[None, element],
                        copy_destination[None, element],
                        pred=predicate,
                    )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _fold_prepeel_accumulator(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_tAcc: cute.Tensor,  # grouped (T2R, T2R_M, T2R_N, (EPI_M, EPI_N))
        tTR_rAcc: cute.Tensor,
        fold_groups: cutlass.Constexpr,
        words_per_class: cutlass.Constexpr,
        epi_tile_n: cutlass.Constexpr,
    ) -> cute.Tensor:
        """Fold this thread's pre-peel accumulator cells into lottery words.

        ``fold_groups`` (from ``_fold_register_groups``) lists, per row class,
        the thread's registers in ascending column order with their static
        word slot. Each partial accumulates one complete message word (the
        committed patterns guarantee whole words per thread; the thread's
        runtime word base offsets the slot at staging time):

        - 4-row tiles: word j of message (lottery row r, column tile ct)
          folds row ``4r + (j >> 2)`` at columns congruent to
          ``{2*(j&3), 2*(j&3)+1}`` mod 8 within the column tile.
        - 16-row tiles: word j folds row ``16r + j`` over the whole
          32-column tile.
        """
        num_partials = len(fold_groups) * self.column_tiles * words_per_class
        partials = cute.make_rmem_tensor(num_partials, Uint32)
        for word in cutlass.range_constexpr(num_partials):
            partials[word] = Uint32(0)
        for epi_n in cutlass.range_constexpr(self.tile_n // epi_tile_n):
            cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, (0, epi_n)], tTR_rAcc)
            for row_class, group in enumerate(fold_groups):
                for value, column_offset, word_slot in group:
                    # Trace-time (fold_groups and epi_n are static): the
                    # partial this column feeds. epi_tile_n % 8 == 0 keeps
                    # word slots subtile-invariant (asserted in kernel()).
                    column_tile = (epi_n * epi_tile_n + column_offset) // self.ltile_cols
                    word = (row_class * self.column_tiles + column_tile) * words_per_class
                    word += word_slot
                    partials[word] = _rotr32(
                        partials[word] * Uint32(_FOLD_MUL) + tTR_rAcc[value].bitcast(Uint32),
                        19,
                    )
        return partials

    @cute.jit
    def _stage_lottery_words(
        self,
        partials: cute.Tensor,
        class_rows,  # per row class, this thread's accumulator row (Int32)
        word_base: Int32,
        words_per_class: cutlass.Constexpr,
        sExt: cute.Tensor,
    ):
        """Write this thread's folded words into their messages' 64-byte blocks."""
        row_shift = self.ltile_rows.bit_length() - 1  # ltile_rows is 4 or 16
        for row_class, row in enumerate(class_rows):
            for column_tile in cutlass.range_constexpr(self.column_tiles):
                message = (row >> row_shift) * self.column_tiles + column_tile
                for slot in cutlass.range_constexpr(words_per_class):
                    lane = self.words_per_row * (row & (self.ltile_rows - 1)) + word_base + slot
                    sExt[message, lane] = partials[
                        (row_class * self.column_tiles + column_tile) * words_per_class + slot
                    ]

    @cute.jit
    def epi_load_acc_subtile_unscaled(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tiled_copy_r2s: cute.TiledCopy,
        tTR_tAcc: cute.Tensor,
        tTR_rAcc: cute.Tensor,
        tRS_sAlA: cute.Tensor,  # grouped broadcast smem view, congruent to tRS_rD
        tRS_sAlB: cute.Tensor,
        tRS_rD: cute.Tensor,
        epi_coord,
        acc_pipeline: pipeline.PipelineAsync,
        acc_consumer_state: pipeline.PipelineState,
        acc_release_idx: int,
        no_release: cutlass.Constexpr = False,
    ):
        """Load one accumulator subtile from TMEM and apply the unscale."""
        cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, epi_coord], tTR_rAcc)
        tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
        tRS_rD.store(tRS_rAcc.load())
        tRS_sAlA_cur = tRS_sAlA[None, None, None, epi_coord]
        tRS_sAlB_cur = tRS_sAlB[None, None, None, epi_coord]
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            tRS_rD[i] = (
                tRS_rD[i] * cute.arch.rcp_approx(tRS_sAlA_cur[i].to(Float32)) * tRS_sAlB_cur[i]
            )
        # Trace-time guard over a runtime predicate; see the note above.
        if const_expr(not no_release):  # noqa: SIM102
            if epi_coord[1] == acc_release_idx:
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)

    # noqa C901: a warp-specialized mainloop. Its branches are compile-time
    # dispatch that must stay in one traced body; splitting them changes the
    # pipeline state each fragment sees.
    @cute.jit
    def mma_fused(  # noqa: C901
        self,
        ab_pipeline: pipeline.PipelineAsync,
        acc_pipeline: pipeline.PipelineAsync,
        prepeel_pipeline: pipeline.PipelineAsync | None,
        ab_consumer_state: pipeline.PipelineState,
        acc_producer_state: pipeline.PipelineState,
        prepeel_producer_state: pipeline.PipelineState | None,
        tiled_mma: cute.TiledMma,
        tiled_mma_peel: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        tCrPa: cute.Tensor,
        tCrPb: cute.Tensor,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        is_leader_cta: Boolean,
    ):
        """FP8 mainloop, the pre-peel handshake, and the BF16 peel UMMA."""
        # Peek (try_wait) AB buffer full for k_tile = 0
        peek_ab_full_status = Boolean(True)
        if k_tile_cnt > 0 and is_leader_cta:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Wait for accumulator buffer empty (epilogue release)
        if is_leader_cta:
            acc_pipeline.producer_acquire(acc_producer_state)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            if is_leader_cta:
                ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    k_blk_coord = (None, None, k_blk_idx, ab_consumer_state.index)
                    cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt and is_leader_cta:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Publish the pre-peel accumulator to the fold (UMMA-async
        # commit), then wait for the epilogue warps' fold release before
        # the peel UMMA may overwrite it. commit -> advance -> acquire
        # pairs one fold release per tile with one acquire per tile on
        # the 1-stage handshake.
        if is_leader_cta:
            prepeel_pipeline.producer_commit(prepeel_producer_state)
        prepeel_producer_state.advance()
        # The peel rides the (k_tile_cnt + 1)-th ring slot of this tile.
        if is_leader_cta:
            prepeel_pipeline.producer_acquire(prepeel_producer_state)
            ab_pipeline.consumer_wait(ab_consumer_state)
            tiled_mma_peel.set(tcgen05.Field.ACCUMULATE, True)
            num_peel_k_blocks = cute.size(tCrPa, mode=[2])
            for k_blk_idx in cutlass.range(num_peel_k_blocks, unroll_full=True):
                k_blk_coord = (None, None, k_blk_idx, ab_consumer_state.index)
                cute.gemm(tiled_mma_peel, acc, tCrPa[k_blk_coord], tCrPb[k_blk_coord], acc)
            ab_pipeline.consumer_release(ab_consumer_state)
        ab_consumer_state.advance()
        # Async arrive accumulator buffer full (post-peel)
        if is_leader_cta:
            acc_pipeline.producer_commit(acc_producer_state)
        acc_producer_state.advance()
        return (
            ab_consumer_state,
            acc_producer_state,
            prepeel_producer_state,
            tiled_mma,
            tiled_mma_peel,
        )

    @cute.jit
    def __call__(
        self,
        mAq: cute.Tensor,  # (m, k) e4m3
        mBq: cute.Tensor,  # (n, k) e4m3
        mPa: cute.Tensor,  # (m, R2) bf16
        mPb: cute.Tensor,  # (n, R2) bf16
        mAlA: cute.Tensor,  # (m,) bf16
        mAlB: cute.Tensor,  # (n,) f32: reciprocal of alpha_b (precomputed)
        mKey: cute.Tensor,  # (8,) u32: pow_key (= cA)
        mThr: cute.Tensor,  # (8,) u32: 256-bit LE threshold
        mD: cute.Tensor,  # (m, n) bf16 out
        mRecord: cute.Tensor,  # (RECORD_WORDS,) u32 mapped pinned hit record
        mLock: cute.Tensor,  # (1,) i32 persistent device first-wins latch
        mHashB: cute.Tensor,  # (8,) u32: commitment hash B (stamped into hits)
        mCodesSrc: cute.Tensor | None,  # flat u32 view of a_codes (m, k) i8
        mScalesSrc: cute.Tensor | None,  # flat u32 view of a_scales (m, k/8) bf16
        mCodesDst: cute.Tensor | None,  # flat u32 codes payload region view
        mScalesDst: cute.Tensor | None,  # flat u32 scales payload region view
        layer_id: Int32,  # per-launch layer tag, stamped into published hits
        record_hits: Int32,  # runtime gate: hash always, publish only when nonzero
        max_active_clusters: Int32,
        stream: cuda_driver.CUstream,
    ):
        payload_tensors = (mCodesSrc, mScalesSrc, mCodesDst, mScalesDst)
        assert all((t is not None) == self.snapshot_payload for t in payload_tensors), (
            "payload tensor presence must match the compiled snapshot variant"
        )
        # Static problem shape and payload byte counts, baked into the hit
        # record writes (shapes are compile-time for a given launch).
        self.problem_m = cute.size(mAq, mode=[0])
        self.problem_n = cute.size(mBq, mode=[0])
        self.problem_k = cute.size(mAq, mode=[1])
        self.codes_payload_bytes = self.problem_m * self.problem_k
        self.scales_payload_bytes = self.problem_m * (self.problem_k // 8) * 2
        if const_expr(self.snapshot_payload):
            assert cute.size(mCodesSrc) * 4 == self.codes_payload_bytes
            assert cute.size(mScalesSrc) * 4 == self.scales_payload_bytes
            assert cute.size(mCodesDst) == cute.size(mCodesSrc)
            assert cute.size(mScalesDst) == cute.size(mScalesSrc)
        self.a_dtype = mAq.element_type
        self.b_dtype = mBq.element_type
        self.a_mma_dtype, self.b_mma_dtype = self.a_dtype, self.b_dtype
        self.a_unpack, self.b_unpack = False, False
        self.a_smem_dtype, self.b_smem_dtype = self.a_dtype, self.b_dtype
        self.d_dtype = mD.element_type
        self.c_dtype = None
        self.sf_dtype = None
        self.a_layout = LayoutEnum.from_tensor(mAq)
        self.b_layout = LayoutEnum.from_tensor(mBq)
        self.d_layout = LayoutEnum.from_tensor(mD)
        self.c_layout = None
        self.a_major_mode = self.a_layout.mma_major_mode()
        self.b_major_mode = self.b_layout.mma_major_mode()

        varlen_args = VarlenArguments()
        self._setup_attributes(self._fused_extra_smem_bytes(), varlen_args)

        atom_thr_size = cute.size(self.tiled_mma.thr_id.shape)

        # The scheduler and TMA-A/B helpers expect batched (x, y, l) tensors.
        mA_mkl = self.permute_batch_last(mAq, append_batch_if_2d=True)
        mB_nkl = self.permute_batch_last(mBq, append_batch_if_2d=True)
        mD_mnl = self.permute_batch_last(mD, append_batch_if_2d=True)
        mPa_mkl = self.permute_batch_last(mPa, append_batch_if_2d=True)
        mPb_nkl = self.permute_batch_last(mPb, append_batch_if_2d=True)

        # TMA load atoms for the FP8 mainloop operands
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            mA_mkl,
            a_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            mB_nkl,
            b_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        ) * atom_thr_size

        # TMA store for D
        tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
            mD, self.epi_smem_layout_staged, self.epi_tile, op_type="store"
        )

        # BF16 peel UMMA with the same instruction M/N (and CTA group) as
        # the FP8 mainloop, so it accumulates into the identical TMEM
        # layout.
        bf16 = cutlass.BFloat16
        tiled_mma_peel = sm100_utils.make_trivial_tiled_mma(
            bf16,
            bf16,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )
        assert tiled_mma_peel.partition_shape_C(
            self.mma_tiler[:2]
        ) == self.tiled_mma.partition_shape_C(self.mma_tiler[:2]), (
            "peel UMMA must share the mainloop accumulator TMEM layout"
        )

        # Peel operands ride the main AB TMA ring: one extra ring slot per
        # output tile, BF16 views aliased onto that slot's A/B regions.
        # Zero dedicated smem/mbars, so the stock ab_stage count survives.
        peel_tiler = (self.mma_tiler[0], self.mma_tiler[1], R2)
        pa_one = sm100_utils.make_smem_layout_a(tiled_mma_peel, peel_tiler, bf16, 1)
        pb_one = sm100_utils.make_smem_layout_b(tiled_mma_peel, peel_tiler, bf16, 1)
        pa_one_slice = cute.slice_(pa_one, (None, None, None, 0))
        pb_one_slice = cute.slice_(pb_one, (None, None, None, 0))
        a_slot_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_slot_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        assert cute.size_in_bytes(bf16, pa_one_slice) <= a_slot_bytes, (
            "peel A must fit inside one AB ring slot (tile_k >= 2*R2 bytes)"
        )
        assert cute.size_in_bytes(bf16, pb_one_slice) <= b_slot_bytes, (
            "peel B must fit inside one AB ring slot (tile_k >= 2*R2 bytes)"
        )

        def _peel_staged(one, slot_bytes):
            # Stage stride = one full A (resp. B) FP8 slot, in bf16 elems.
            outer = one.outer
            staged_outer = cute.make_layout(
                (outer.shape[0], outer.shape[1], outer.shape[2], self.ab_stage),
                stride=(outer.stride[0], outer.stride[1], outer.stride[2], slot_bytes // 2),
            )
            return cute.make_composed_layout(one.inner, 0, staged_outer)

        pa_smem_layout_staged = _peel_staged(pa_one, a_slot_bytes)
        pb_smem_layout_staged = _peel_staged(pb_one, b_slot_bytes)
        tma_atom_pa, tma_tensor_pa = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            mPa_mkl,
            pa_one_slice,
            peel_tiler,
            tiled_mma_peel,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_pb, tma_tensor_pb = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            mPb_nkl,
            pb_one_slice,
            peel_tiler,
            tiled_mma_peel,
            self.cluster_layout_vmnk.shape,
        )
        self.num_peel_tx_bytes = (
            cute.size_in_bytes(bf16, pa_one_slice) + cute.size_in_bytes(bf16, pb_one_slice)
        ) * atom_thr_size

        # Persistent CLC tile scheduler
        scheduler_args = TileSchedulerOptions(Int32(max_active_clusters))
        TileSchedulerCls = self.get_scheduler_class(varlen_m=False)
        tile_sched_args = self.get_scheduler_arguments(
            mA_mkl, mB_nkl, mD_mnl, scheduler_args, varlen_args, None
        )
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(tile_sched_params, Int32(max_active_clusters))

        self.buffer_align_bytes = 1024

        sext_size = self.msgs_cta * _SEXT_PAD
        sala_size = self.cta_m
        salb_size = self.tile_n
        self.sched_smem_size = 4 * self.sched_stage + 6

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged)

        @cute.struct
        class SharedStorage:
            sExt: cute.struct.Align[cute.struct.MemRange[Uint32, sext_size], 16]
            sAlA: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, sala_size], 16]
            sAlB: cute.struct.Align[cute.struct.MemRange[Float32, salb_size], 16]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size], self.buffer_align_bytes
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma,
            tiled_mma_peel,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_pa,
            tma_tensor_pa,
            tma_atom_pb,
            tma_tensor_pb,
            tma_atom_d,
            tma_tensor_d,
            mAlA,
            mAlB,
            mKey,
            mThr,
            mRecord,
            mLock,
            mHashB,
            mCodesSrc,
            mScalesSrc,
            mCodesDst,
            mScalesDst,
            layer_id,
            record_hits,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            pa_smem_layout_staged,
            pb_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
            TileSchedulerCls,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_peel: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_pa: cute.CopyAtom,
        mPa_mkl: cute.Tensor,
        tma_atom_pb: cute.CopyAtom,
        mPb_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mn: cute.Tensor,
        mAlA: cute.Tensor,
        mAlB: cute.Tensor,
        mKey: cute.Tensor,
        mThr: cute.Tensor,
        mRecord: cute.Tensor,
        mLock: cute.Tensor,
        mHashB: cute.Tensor,
        mCodesSrc: cute.Tensor | None,
        mScalesSrc: cute.Tensor | None,
        mCodesDst: cute.Tensor | None,
        mScalesDst: cute.Tensor | None,
        layer_id: Int32,
        record_hits: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        pa_smem_layout: cute.ComposedLayout,
        pb_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_pa, tma_atom_pb, tma_atom_d):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        sched_data_flat = smem.allocate_tensor(
            Int32,
            cute.make_layout(self.sched_smem_size),
            byte_alignment=16,
            partition=SmemPartition.RESERVED,
        )
        sched_data = cute.make_tensor(
            sched_data_flat.iterator, cute.make_layout((4, self.sched_stage))
        )
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cluster_layout_vmnk,
            is_leader_cta=is_leader_cta,
        )
        acc_pipeline = self.make_acc_pipeline(cluster_layout_vmnk=cluster_layout_vmnk)
        prepeel_pipeline = self.make_prepeel_pipeline(cluster_layout_vmnk=cluster_layout_vmnk)
        sched_pipeline = self.make_sched_pipeline(self.cluster_shape_mnk, has_C=False)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * (1 + self.num_epi_warps),
        )
        tmem = cutlass.utils.TmemAllocator(
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # Smem tensors: mainloop FP8 operands, their BF16 peel aliases, the
        # epilogue store buffer, lottery words, and staged alpha slices.
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        # Padded stride: compress reads sExt[msg, c] bank-conflict-free.
        sExt = storage.sExt.get_tensor(
            cute.make_layout((self.msgs_cta, LANES), stride=(_SEXT_PAD, 1))
        )
        sPa = storage.sA.get_tensor(
            pa_smem_layout.outer, swizzle=pa_smem_layout.inner, dtype=cutlass.BFloat16
        )
        sPb = storage.sB.get_tensor(
            pb_smem_layout.outer, swizzle=pb_smem_layout.inner, dtype=cutlass.BFloat16
        )
        sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sAlA = storage.sAlA.get_tensor(cute.make_layout(self.cta_m))
        sAlB = storage.sAlB.get_tensor(cute.make_layout(self.tile_n))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_peel = tiled_mma_peel.get_slice(mma_tile_coord_v)

        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        TileSchedulerCreate = partial(
            TileSchedulerCls.create,
            tile_sched_params,
            sched_data,
            sched_pipeline,
            throttle_barrier=self.clc_throttle_barrier,
        )

        k_tile_cnt = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.mma_tiler[2])

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ========================= AB load warp =========================
        if warp_idx == self.ab_load_warp_id:
            a_tma_multicast = {
                "cluster_shape": self.cluster_shape_mnk[:2],
                "multicast_dim": "M",
            }
            b_tma_multicast = {
                "cluster_shape": self.cluster_shape_mnk[:2],
                "multicast_dim": "N",
            }
            tile_scheduler = TileSchedulerCreate()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            is_throttle_producer = Boolean(True)
            if const_expr(cute.size(cluster_layout_vmnk) > 1):
                is_throttle_producer = Boolean(cute.arch.block_idx_in_cluster() == 0)
            while work_tile.is_valid_tile:
                tile_scheduler.throttle_producer_commit(is_throttle_producer)
                tile_coord_mnkl = work_tile.tile_idx
                mma_tile_coord_m = tile_coord_mnkl[0] // cute.size(tiled_mma.thr_id.shape)
                # (bM, bK, RestK)
                gA_mk = cute.local_tile(
                    mA_mkl[None, None, 0],
                    cute.select(self.mma_tiler, [0, 2]),
                    (mma_tile_coord_m, None),
                )
                # (bN, bK, RestK)
                gB_nk = cute.local_tile(
                    mB_nkl[None, None, 0],
                    cute.select(self.mma_tiler, [1, 2]),
                    (tile_coord_mnkl[1], None),
                )
                copy_A = copy_utils.tma_get_block_copy_fn(
                    tma_atom_a,
                    src_tensor=thr_mma.partition_A(gA_mk),
                    dst_tensor=sA,
                    tma_multicast=a_tma_multicast,
                )
                copy_B = copy_utils.tma_get_block_copy_fn(
                    tma_atom_b,
                    src_tensor=thr_mma.partition_B(gB_nk),
                    dst_tensor=sB,
                    tma_multicast=b_tma_multicast,
                )
                ab_producer_state = self.load_tma(
                    ab_pipeline, ab_producer_state, [copy_A, copy_B], k_tile_cnt
                )
                # A_peel + B_peel take the (k_tile_cnt + 1)-th ring slot
                # of this tile. extra_tx_count rebases the expect-tx from
                # the full A+B slot bytes down to the peel bytes.
                gPa_mk = cute.local_tile(
                    mPa_mkl[None, None, 0],
                    (self.mma_tiler[0], R2),
                    (mma_tile_coord_m, None),
                )
                gPb_nk = cute.local_tile(
                    mPb_nkl[None, None, 0],
                    (self.mma_tiler[1], R2),
                    (tile_coord_mnkl[1], None),
                )
                copy_Pa = copy_utils.tma_get_block_copy_fn(
                    tma_atom_pa,
                    src_tensor=thr_mma_peel.partition_A(gPa_mk),
                    dst_tensor=sPa,
                    tma_multicast=a_tma_multicast,
                )
                copy_Pb = copy_utils.tma_get_block_copy_fn(
                    tma_atom_pb,
                    src_tensor=thr_mma_peel.partition_B(gPb_nk),
                    dst_tensor=sPb,
                    tma_multicast=b_tma_multicast,
                )
                ab_pipeline.producer_acquire(
                    ab_producer_state,
                    extra_tx_count=self.num_peel_tx_bytes - self.num_tma_load_bytes,
                )
                peel_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                copy_Pa(0, ab_producer_state.index, tma_bar_ptr=peel_bar)
                copy_Pb(0, ab_producer_state.index, tma_bar_ptr=peel_bar)
                ab_pipeline.producer_commit(ab_producer_state)
                ab_producer_state.advance()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop
            ab_pipeline.producer_tail(ab_producer_state)

        # ========================= scheduler warp =========================
        if warp_idx == self.scheduler_warp_id:
            is_scheduler_warp = True
            if const_expr(cute.size(cluster_layout_vmnk) > 1):
                is_scheduler_warp = cute.arch.block_idx_in_cluster() == 0
            tile_scheduler = TileSchedulerCreate(is_scheduler_warp=is_scheduler_warp)
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop
            if is_scheduler_warp:
                tile_scheduler.producer_tail()
                tile_scheduler.cancel_pending_tail()

        # The epi-load warp has no C operand to load: it idles.

        # ========================= MMA warp =========================
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_K, STAGE)
            tCrA = tiled_mma.make_fragment_A(sA)
            # (MMA, MMA_N, MMA_K, STAGE)
            tCrB = tiled_mma.make_fragment_B(sB)
            # Peel operands are smem-descriptor fragments over the BF16
            # ring-slot aliases (stage mode selected at issue time).
            tCrPa = tiled_mma_peel.make_fragment_A(sPa)
            tCrPb = tiled_mma_peel.make_fragment_B(sPb)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            tile_scheduler = TileSchedulerCreate()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_consumer_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            acc_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            prepeel_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            while work_tile.is_valid_tile:
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[None, None, None, acc_producer_state.index]
                (
                    ab_consumer_state,
                    acc_producer_state,
                    prepeel_producer_state,
                    tiled_mma,
                    tiled_mma_peel,
                ) = self.mma_fused(
                    ab_pipeline,
                    acc_pipeline,
                    prepeel_pipeline,
                    ab_consumer_state,
                    acc_producer_state,
                    prepeel_producer_state,
                    tiled_mma,
                    tiled_mma_peel,
                    tCrA,
                    tCrB,
                    tCrPa,
                    tCrPb,
                    tCtAcc,
                    k_tile_cnt,
                    is_leader_cta,
                )
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop
            tmem_alloc_barrier.arrive()
            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

        # ========================= epilogue warps =========================
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            is_tma_warp = Boolean(warp_idx == self.epilog_warp_id[0])
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, epi_tile, use_2cta_instrs
            )
            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                tiled_copy_t2r, self.d_layout, self.d_dtype, tTR_rD, sD, epi_tidx
            )

            # Static fold mapping (asserts whole-word ownership per thread)
            # and the dynamic accumulator rows and word base this thread owns.
            epi_tile_n = const_expr(cute.size(epi_tile[1]))
            # Word slots and column tiles must be subtile-invariant: subtile
            # offsets may not rotate a column's mod-8 pair class.
            assert epi_tile_n % 8 == 0, f"epi_tile_n must be a multiple of 8, got {epi_tile_n}"
            fold_groups, words_per_class = _fold_register_groups(
                tiled_copy_t2r, self.words_per_row, self.ltile_cols
            )
            cAcc = cute.make_identity_tensor((self.cta_m, self.tile_n))
            cAcc_epi = cute.flat_divide(cAcc, epi_tile)
            tTR_cAcc = tiled_copy_t2r.get_slice(epi_tidx).partition_D(cAcc_epi)
            tTR_cAcc_first = tTR_cAcc[None, None, None, 0, 0]
            class_rows = [tTR_cAcc_first[group[0][0]][0] for group in fold_groups]
            first_column = tTR_cAcc_first[fold_groups[0][0][0]][1]
            word_base = (first_column >> 1) & (self.words_per_row - 1)

            # Broadcast smem views of the staged scales, partitioned
            # exactly like the accumulator so registers line up
            # element-wise.
            sAlA_bcast = cute.make_tensor(
                sAlA.iterator,
                cute.make_layout((self.cta_m, self.tile_n), stride=(1, 0)),
            )
            sAlB_bcast = cute.make_tensor(
                sAlB.iterator,
                cute.make_layout((self.cta_m, self.tile_n), stride=(0, 1)),
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N): the same subtile partition
            # the accumulator load uses.
            tRS_sAlA = tiled_copy_r2s.retile(
                thr_copy_t2r.partition_D(cute.flat_divide(sAlA_bcast, epi_tile))
            )
            tRS_sAlB = tiled_copy_r2s.retile(
                thr_copy_t2r.partition_D(cute.flat_divide(sAlB_bcast, epi_tile))
            )
            tRS_sAlA_g = cute.group_modes(tRS_sAlA, 3, cute.rank(tRS_sAlA))
            tRS_sAlB_g = cute.group_modes(tRS_sAlB, 3, cute.rank(tRS_sAlB))

            chaining_value = [mKey[i] for i in range(8)]
            threshold = [mThr[i] for i in range(8)]

            epi_tile_num = const_expr(
                cute.size(
                    cute.zipped_divide(cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile),
                    mode=[1],
                )
            )

            tile_scheduler = TileSchedulerCreate()
            work_tile = tile_scheduler.initial_work_tile_info()
            acc_consumer_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            prepeel_consumer_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            epi_store_pipeline = self.make_epi_store_pipeline()
            while work_tile.is_valid_tile:
                # Consume the next CLC grant early so its response wait
                # overlaps the fold + epilogue. get_current_work() pops one
                # scheduler-pipeline slot per call (advance_to_next_work is
                # per-warp bookkeeping for non-scheduler warps), so this loop
                # walks the same grant sequence as the other warps' loops.
                next_work_tile = tile_scheduler.get_current_work()
                tile_coord_mnkl = work_tile.tile_idx

                # Stage this tile's alpha slices while the mainloop runs.
                self._stage_alpha_slices(mAlA, mAlB, sAlA, sAlB, tile_coord_mnkl, epi_tidx)

                # (T2R, T2R_M, T2R_N, (EPI_M, EPI_N))
                tTR_tAcc = tTR_tAcc_base[None, None, None, None, None, acc_consumer_state.index]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                # Fold the pre-peel accumulator into lottery words, then
                # release the 1-stage handshake that was holding it against
                # the peel UMMA.
                prepeel_pipeline.consumer_wait(prepeel_consumer_state)
                partials = self._fold_prepeel_accumulator(
                    tiled_copy_t2r,
                    tTR_tAcc,
                    tTR_rAcc,
                    fold_groups,
                    words_per_class,
                    epi_tile_n,
                )
                cute.arch.fence_view_async_tmem_load()
                prepeel_pipeline.consumer_release(prepeel_consumer_state)
                prepeel_consumer_state.advance()
                self._stage_lottery_words(partials, class_rows, word_base, words_per_class, sExt)

                # Alpha staging must be visible before the unscale reads
                # it; the same barrier publishes sExt for the hash.
                cute.arch.cp_async_wait_group(0)
                self.epilogue_barrier.arrive_and_wait()

                # Keyed BLAKE3 over the staged messages. It overlaps the peel
                # UMMA.
                self._compress_and_publish(
                    sExt,
                    chaining_value,
                    threshold,
                    mRecord,
                    mLock,
                    mHashB,
                    mCodesSrc,
                    mScalesSrc,
                    mCodesDst,
                    mScalesDst,
                    layer_id,
                    record_hits,
                    tile_coord_mnkl,
                    epi_tidx,
                )

                # Wait for the post-peel accumulator, then run the
                # TMA-store epilogue with the unscale fused into the
                # subtile loads.
                acc_pipeline.consumer_wait(acc_consumer_state)
                copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                    tma_atom_d,
                    mD_mn,
                    self.cta_tile_shape_mnk[:2],
                    epi_tile,
                    sD,
                    tile_coord_mnkl,
                )
                load_acc_subtile = partial(
                    self.epi_load_acc_subtile_unscaled,
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tTR_tAcc,
                    tTR_rAcc,
                    tRS_sAlA_g,
                    tRS_sAlB_g,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    acc_release_idx=epi_tile_num - 1,
                )
                self.epilogue(
                    None,  # params
                    {},  # epi_smem_tensors
                    None,  # epi_pipeline
                    epi_store_pipeline,
                    None,  # epi_read_state
                    None,  # epi_producer_state
                    epi_tile,
                    load_acc_subtile,
                    tRS_rD,
                    None,  # tRS_rC
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tRS_sD,
                    None,  # tiled_copy_s2r
                    None,  # tSR_rC
                    None,  # tSR_sC
                    copy_D,
                    None,  # copy_C
                    tile_coord_mnkl,
                    None,  # varlen_manager
                    self.epilogue_barrier,
                    tile_scheduler,
                    epi_tidx,
                    is_tma_warp,
                )
                acc_consumer_state.advance()
                tile_scheduler.advance_to_next_work()
                work_tile = next_work_tile
                # End of persistent scheduler loop

            # Wait for D store complete. Trace-time guard over a runtime
            # predicate; see the note on epi_load_acc_subtile_unscaled.
            if is_tma_warp:
                epi_store_pipeline.producer_tail()
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
