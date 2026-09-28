"""Persistent SM120 FP8 GEMM with lottery, peel, unscale, and hit publishing.

Built on Quack's ``GemmSm120``: a TMA-fed ``mma.sync`` GEMM whose accumulator
lives in the mma warps' registers. The fused phases are homed on that
topology:

- one TMA load warp fills the FP8 A/B ring and, per output tile, one extra
  ring slot carrying the BF16 peel operands (aliased onto the FP8 slot
  bytes with a reduced expect-tx);
- the mma warps run the FP8 mainloop, fold their pre-peel accumulator
  registers into lottery words in shared memory, issue the BF16 peel
  ``mma.sync`` into the same registers, and run the TMA-store epilogue with
  the row/column unscale fused into the subtile loads;
- the producer warpgroup's otherwise idle warps (one per mma warpgroup)
  hash the staged words with keyed BLAKE3 and publish the first winner
  into the process's persistent hit signal, overlapping the peel and the
  next tile's mainloop.

Every mma warp owns one 16-row slab spanning the whole ``tile_n``
(``atom_layout_mnk = (tile_m / 16, 1, 1)``), so a lane holds complete
lottery words: rows ``T >> 2`` and ``(T >> 2) + 8`` of its slab at the
mod-8 column pair ``2 * (T & 3)``. That is the 4-row lottery family only;
the 16-row family's whole-row words are split across four lanes.
"""

import enum
from functools import partial

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as blackwell_helpers
import cutlass.utils.hopper_helpers as sm90_utils
import quack.copy_utils as copy_utils
from cutlass import Boolean, Float32, Int32, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum, SmemPartition, blockscaled_layout
from quack import sm80_utils
from quack.epilogue.ops import EpiSmemBytes
from quack.gemm_base import NamedBarrierGemm
from quack.gemm_sm120 import GemmSm120, _sf_group_vmk
from quack.pipeline import PipelineAsync as BasePipelineAsync
from quack.pipeline import PipelineTmaAsync as BasePipelineTmaAsync
from quack.pipeline import make_pipeline_state
from quack.rounding import RoundingMode
from quack.tile_scheduler import (
    PersistenceMode,
    RasterOrderOption,
    TileScheduler,
    TileSchedulerArguments,
)

from ..tensor_hash_plus_stats._blake3 import _rotr32
from ..tensor_hash_plus_stats._blake3_ops import SINGLE_BLOCK_KEYED_FLAGS, compress
from ._kernel import (
    _FOLD_MUL,
    _SEXT_PAD,
    DEFAULT_LTILE_COLS,
    LANES,
    R2,
    SUPPORTED_LTILE_COLS,
    _HitPublishMixin,
)

LTILE_ROWS = 4  # the only SM120 lottery tile row count (see module docstring)
WG_THREADS = 128
WARP_ROWS = 16  # accumulator rows per mma warp (one m16n8 atom row block)
WG_ROWS = 4 * WARP_ROWS  # rows per mma warpgroup, the unit a compression warp hashes
_MMA_INST_N = 8  # mma.sync m16n8 atom columns
_SF_BLOCK_N = 128  # N granularity of the block-scaled MMA's scale-factor smem atoms

_RASTER_GROUP_SIZE = 8  # tile-scheduler rasterization/serpentine group size

# Each compression-warp lane hashes ceil(msgs_wg / 32) messages from a
# bounded sExt allocation; 96 keeps that at three per lane.
_MAX_MSGS_WG = 96


class _NamedBarrier(enum.IntEnum):
    """Barrier IDs after Quack's reserved ``NamedBarrierGemm`` range (1..8)."""

    FOLD_DONE_WG0 = 9  # + mma warpgroup
    LOTTERY_BUFFER_FREE_WG0 = 11  # + mma warpgroup
    ALPHA_READY = 13


class _FusedGemmSm120(_HitPublishMixin, GemmSm120):
    """Quack's SM120 GEMM with lottery, peel, and unscale fused in."""

    # GemmSm120._setup_attributes inspects the epilogue-op tuple; this kernel has none.
    _epi_ops = ()

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int | None = None,
        *,
        snapshot_payload: bool = False,
        ltile_rows: int = LTILE_ROWS,
        ltile_cols: int = DEFAULT_LTILE_COLS,
    ):
        assert tile_m in (64, 128), f"SM120 fused GEMM needs tile_m 64 or 128, got {tile_m}"
        # mma.sync fragments split every accumulator row across four lanes at
        # mod-8 column-pair granularity: only the 4-row family's words are
        # lane-local.
        assert ltile_rows == LTILE_ROWS, (
            f"the SM120 lottery implements only the {LTILE_ROWS}-row tile family, got {ltile_rows}"
        )
        assert ltile_cols in SUPPORTED_LTILE_COLS[LTILE_ROWS], (
            f"unsupported lottery tile cols: {ltile_cols}"
        )
        assert tile_n % ltile_cols == 0, f"lottery needs tile_n % {ltile_cols} == 0"
        super().__init__(
            acc_dtype=Float32,
            a_dtype=cutlass.Float8E4M3FN,
            tile_shape_mnk=(tile_m, tile_n) if tile_k is None else (tile_m, tile_n, tile_k),
            cluster_shape_mnk=(1, 1, 1),  # SM120 has no thread-block clusters
            pingpong=False,
            is_persistent=True,
            # The lottery consumes commitment-stage outputs at kernel start;
            # PDL overlap with the producer kernel would break that ordering.
            use_pdl=False,
        )
        # Re-derive the warp topology for one mma warp per 16-row slab spanning
        # the whole tile_n (the substrate defaults to a 2-wide N split, which
        # would leave every lane with half of each lottery word).
        self.atom_layout_mnk = (tile_m // WARP_ROWS, 1, 1)
        self.num_mma_warps = tile_m // WARP_ROWS
        self.mma_warp_groups = self.num_mma_warps // 4
        self.threads_per_cta = (self.mma_warp_groups + 1) * WG_THREADS
        self.num_epi_warps = self.num_mma_warps
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        self.ab_load_warp_id = self.num_mma_warps
        self.tile_m, self.tile_n = tile_m, tile_n
        self.ltile_rows = LTILE_ROWS
        self.ltile_cols = ltile_cols
        self.column_tiles = tile_n // ltile_cols
        # Messages per mma warpgroup: its 64 rows are 16 lottery rows.
        self.msgs_wg = (WG_ROWS // LTILE_ROWS) * self.column_tiles
        assert self.msgs_wg <= _MAX_MSGS_WG, f"msgs_wg={self.msgs_wg} > {_MAX_MSGS_WG}"
        self.snapshot_payload = snapshot_payload
        # The producer warpgroup holds BLAKE3 state in its compression warps
        # (the stock 40-register budget spills it); the mma warps hold the
        # accumulator plus two in-flight B k-blocks. INVARIANT: the split
        # times the warpgroups must stay <= 64512 registers -- 65536 fills
        # the register file and setmaxnreg.inc then spin-waits forever.
        self.num_regs_load, self.num_regs_mma = (
            (56, 224) if self.mma_warp_groups == 2 else (80, 232)
        )
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
        # sD/sA/sB buffers, plus the fused buffers themselves. The peel costs
        # nothing: it aliases the A/B ring slots.
        extra = 3072
        extra += self.mma_warp_groups * self.msgs_wg * _SEXT_PAD * 4 + 128  # sExt
        extra += self.tile_m * 2 + self.tile_n * 4 + 32  # sAlA (bf16) + sAlB (f32)
        return extra

    # -- ab ring pipeline: Quack's TmaAsync wrapper (its producer_acquire takes
    # extra_tx_count, which the peel ring slot needs to rebase its expect-tx) --
    def make_ab_pipeline(self, tiled_mma: cute.TiledMma, cluster_layout_vmnk: cute.Layout):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_arrive_cnt = tiled_mma.size // cute.arch.WARP_SIZE  # one arrive per mma warp
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt)
        return BasePipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    # -- widen the scheduler pipeline: compression warps also consume tile slots --
    def make_sched_pipeline(self, cluster_layout_mnk: cute.Layout, varlen_k: bool = False):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_arrive_cnt = (
            self.num_mma_warps + self.num_ab_load_warps + self.mma_warp_groups
        ) * cute.size(cluster_layout_mnk)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt)
        return BasePipelineAsync.create(
            num_stages=self.sched_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            consumer_mask=None,
            defer_sync=True,
            elect_one_release=True,
        )

    def _setup_tiled_mma(self):
        super()._setup_tiled_mma()
        tile_n = self.cta_tile_shape_mnk[1]
        if self.use_mxf8_mma and tile_n > _SF_BLOCK_N and tile_n % _SF_BLOCK_N:
            # The unit-scale SFB fragment covers whole 128-column SF blocks and
            # slices cleanly only within one block or to whole blocks, so
            # 192-column tiles take the plain fp8 instruction instead: the
            # same Blackwell atom at half the rate.
            self.use_mxf8_mma = False
            self.use_mxf8f6f4_op = False
            self.tiled_mma = cute.make_tiled_mma(
                warp.MmaFP8Op(self.mma_a_dtype, self.acc_dtype, self.mma_inst_mnk),
                cute.make_layout(self.atom_layout_mnk),
                permutation_mnk=self.tiled_mma.permutation_mnk,
            )

    def _make_tiled_mma_peel(self) -> cute.TiledMma:
        """The BF16 peel MMA over the same warp layout and N permutation as the
        FP8 mainloop, so it accumulates into the identical register fragment."""
        op = warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (WARP_ROWS, _MMA_INST_N, 16))
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        permutation_n = cute.make_ordered_layout(
            (_MMA_INST_N, atom_n, self.mma_n_warp_run // _MMA_INST_N), order=(0, 2, 1)
        )
        return cute.make_tiled_mma(
            op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=(atom_m * WARP_ROWS, permutation_n, atom_k * 16),
        )

    def _check_fragment_ownership(self, tiled_mma: cute.TiledMma) -> None:
        """Assert the lane -> accumulator cell map the fold and unscale index by.

        Lane ``T`` of warp ``w`` holds rows ``16w + (T >> 2) + 8h`` and columns
        ``8 nb + 2 (T & 3) + c`` at fragment index ``((c, h), 0, nb)``. Derived
        from the tiled MMA's static C thread-value layout so an atom or
        permutation change fails at compile time, not silently.
        """
        layout_tv = tiled_mma.tv_layout_C_tiled
        m_size = cute.size(tiled_mma.permutation_mnk[0])
        n_size = cute.size(tiled_mma.permutation_mnk[1])
        values = cute.size(layout_tv, mode=[1])
        assert m_size == self.tile_m and values == 2 * 2 * (n_size // _MMA_INST_N)
        assert tiled_mma.partition_shape_C((self.tile_m, self.tile_n)) == (
            (2, 2),
            1,
            self.tile_n // _MMA_INST_N,
        )
        for thread in range(cute.size(layout_tv, mode=[0])):
            warp_idx, lane = divmod(thread, cute.arch.WARP_SIZE)
            for value in range(values):
                flat = layout_tv((thread, value))
                row, column = flat % m_size, flat // m_size
                c, h, nb = value % 2, (value // 2) % 2, value // 4
                assert row == WARP_ROWS * warp_idx + (lane >> 2) + 8 * h, "fragment row map"
                assert column == _MMA_INST_N * nb + 2 * (lane & 3) + c, "fragment column map"

    @cute.jit
    def _fold_lottery_word(
        self,
        acc: cute.Tensor,
        row_half: cutlass.Constexpr,
        column_tile: cutlass.Constexpr,
        blocks_per_column_tile: cutlass.Constexpr,
    ):
        """Fold one lane-local accumulator strip into a lottery word.

        Word ``j`` of a message folds row ``4r + (j >> 2)`` at the columns
        congruent to ``{2 (j & 3), 2 (j & 3) + 1}`` mod 8 within the column
        tile, in ascending column order.
        """
        folded = Uint32(0)
        for block in cutlass.range_constexpr(blocks_per_column_tile):
            for column in cutlass.range_constexpr(2):
                word = acc[
                    ((column, row_half), 0, column_tile * blocks_per_column_tile + block)
                ].bitcast(Uint32)
                folded = _rotr32(folded * Uint32(_FOLD_MUL) + word, 19)
        return folded

    @cute.jit
    def _stage_lottery_messages(
        self,
        acc: cute.Tensor,
        sExt: cute.Tensor,
        warp_group_idx: Int32,
        warp_in_wg: Int32,
        lane: Int32,
    ):
        """Fold and stage this warpgroup's pre-peel accumulator, then hand it to
        the compression warp (``FOLD_DONE``) once the previous tile's words
        have been consumed (``LOTTERY_BUFFER_FREE``)."""
        blocks_per_column_tile = self.ltile_cols // _MMA_INST_N
        cute.arch.barrier(
            barrier_id=int(_NamedBarrier.LOTTERY_BUFFER_FREE_WG0) + warp_group_idx,
            number_of_threads=WG_THREADS + cute.arch.WARP_SIZE,
        )
        for row_half in cutlass.range_constexpr(2):
            # Lottery row within the warpgroup's 64-row slab.
            thread_row = 4 * warp_in_wg + 2 * row_half + (lane >> 4)
            for column_tile in cutlass.range_constexpr(self.column_tiles):
                folded = self._fold_lottery_word(acc, row_half, column_tile, blocks_per_column_tile)
                message = thread_row * self.column_tiles + column_tile
                sExt[warp_group_idx, message, lane & (LANES - 1)] = folded
        cute.arch.barrier_arrive(
            barrier_id=int(_NamedBarrier.FOLD_DONE_WG0) + warp_group_idx,
            number_of_threads=WG_THREADS + cute.arch.WARP_SIZE,
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
        warp_group_idx: cutlass.Constexpr,
        lane: Int32,
    ):
        """Hash one mma warpgroup's staged messages and publish the winner.

        Each compression-warp lane hashes ceil(msgs_wg / 32) messages. The
        publish ballot needs a convergent warp, so a sub-warp message tail
        keeps its whole warp hashing: excess lanes rehash the last message
        and are excluded from the ballot.
        """
        cute.arch.barrier(
            barrier_id=int(_NamedBarrier.FOLD_DONE_WG0) + warp_group_idx,
            number_of_threads=WG_THREADS + cute.arch.WARP_SIZE,
        )
        for message_batch in cutlass.range_constexpr(cute.ceil_div(self.msgs_wg, 32)):
            lane_message = lane + 32 * message_batch
            if const_expr(32 * (message_batch + 1) <= self.msgs_wg):
                is_message = Boolean(True)
                message = lane_message
            else:
                is_message = Boolean(lane_message < self.msgs_wg)
                message = cutlass.min(lane_message, self.msgs_wg - 1)
            # Lane-local decode; the publish leader's values are published.
            column_tile = message % self.column_tiles
            thread_row = message // self.column_tiles
            tile_row = (
                tile_coord_mnkl[0] * (self.tile_m // LTILE_ROWS)
                + warp_group_idx * (WG_ROWS // LTILE_ROWS)
                + thread_row
            )
            tile_column = tile_coord_mnkl[1] * self.column_tiles + column_tile
            in_bounds = Boolean(tile_row * LTILE_ROWS < self.problem_m) & Boolean(
                tile_column * self.ltile_cols < self.problem_n
            )
            words = [sExt[warp_group_idx, message, column].to(Uint32) for column in range(LANES)]
            digest = compress(
                list(chaining_value),
                words,
                64,
                SINGLE_BLOCK_KEYED_FLAGS,
            )
            local_hit = self._digest_below_threshold(digest, threshold) & is_message
            self._publish_hit(
                tile_row,
                tile_column,
                in_bounds,
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
        cute.arch.barrier_arrive(
            barrier_id=int(_NamedBarrier.LOTTERY_BUFFER_FREE_WG0) + warp_group_idx,
            number_of_threads=WG_THREADS + cute.arch.WARP_SIZE,
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
        alpha_threads = self.num_mma_warps * cute.arch.WARP_SIZE
        for global_scale, shared_scale, coordinate_mode, tile_extent in (
            (mAlA, sAlA, 0, self.tile_m),
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
    def _row_inverse_scales(self, sAlA: cute.Tensor, warp_idx: Int32, lane: Int32):
        """Wait for the staged scales and load this lane's two row reciprocals."""
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(
            barrier_id=int(_NamedBarrier.ALPHA_READY),
            number_of_threads=self.num_mma_warps * cute.arch.WARP_SIZE,
        )
        row_inverse = cute.make_rmem_tensor(2, Float32)
        for row_half in cutlass.range_constexpr(2):
            row = WARP_ROWS * warp_idx + (lane >> 2) + 8 * row_half
            row_inverse[row_half] = cute.arch.rcp_approx(sAlA[row].to(Float32))
        return row_inverse

    @cute.jit
    def epi_load_acc_subtile_unscaled(
        self,
        tRS_rAcc: cute.Tensor,
        acc: cute.Tensor,
        row_inverse: cute.Tensor,
        sAlB: cute.Tensor,
        lane: Int32,
        tRS_rD: cute.Tensor,
        epi_coord,  # (epi_m, epi_n), constexpr
    ):
        """Multiply this subtile's acc slice by the alpha reciprocals, then load.

        ``tRS_rAcc`` is a register view over ``acc``, so scaling right before
        the subtile's copy keeps the multiplies inside the pipelined epilogue
        loop. The column reciprocals are read from shared memory here rather
        than held in registers (``tile_n / 4`` of them per lane).
        """
        assert self.epi_tile[0] == self.tile_m, "the epilogue subtile spans every warp slab"
        # The scale is applied to acc in place, so each subtile must be loaded
        # exactly once: the substrate's prepass variant would load it twice.
        assert not self.epi_needs_acc_prepass, (
            "in-place unscale needs a single acc load per subtile"
        )
        blocks = self.epi_tile[1] // _MMA_INST_N
        for block in cutlass.range_constexpr(blocks):
            nb = epi_coord[1] * blocks + block
            for row_half in cutlass.range_constexpr(2):
                for column in cutlass.range_constexpr(2):
                    column_inverse = sAlB[_MMA_INST_N * nb + 2 * (lane & 3) + column]
                    acc[((column, row_half), 0, nb)] = (
                        acc[((column, row_half), 0, nb)] * row_inverse[row_half] * column_inverse
                    )
        cute.autovec_copy(tRS_rAcc[(None, None, None, epi_coord)], tRS_rD)

    @cute.jit
    def _unit_scale_fragments(self, tiled_mma: cute.TiledMma, thr_mma, sB: cute.Tensor, tCrB, tidx):
        """Constant ue8m0 1.0 scale fragments for the block-scaled fp8 mma.

        The kind::mxf8f6f4 instruction runs at twice the plain fp8 rate and,
        with unit scales, computes the identical Blackwell atom (gated
        bit-identical per config in ``tests/test_mixed_gemm_configs.py``). The
        partition helpers only consume
        layout shapes, so a dummy tensor over any smem pointer serves.
        """
        sfa_layout = blockscaled_layout.sm120_make_smem_layout_sfa(
            tiled_mma, self.cta_tile_shape_mnk, 32, 1
        )
        # The SF blob is 128-N granular: pad tile_n up to whole blocks (64 and
        # 192 included) and slice the fragment back to the tile's N atoms.
        sfb_tile = (
            self.cta_tile_shape_mnk[0],
            -(-self.cta_tile_shape_mnk[1] // _SF_BLOCK_N) * _SF_BLOCK_N,
            self.cta_tile_shape_mnk[2],
        )
        sfb_layout = blockscaled_layout.sm120_make_smem_layout_sfb(tiled_mma, sfb_tile, 32, 1)
        sf_ptr = cute.recast_ptr(sB.iterator, dtype=cutlass.Float8E8M0FNU)
        sSFA_like = cute.make_tensor(sf_ptr, sfa_layout)
        sSFB_like = cute.make_tensor(sf_ptr, sfb_layout)
        tCrSFA = blackwell_helpers.partition_fragment_SFA(sSFA_like[None, None, 0], thr_mma, tidx)
        tCrSFB = blackwell_helpers.partition_fragment_SFB(sSFB_like[None, None, 0], thr_mma, tidx)
        k_atoms = self.cta_tile_shape_mnk[2] // 32
        tCrSFA = _sf_group_vmk(tCrSFA, k_atoms)
        tCrSFB = _sf_group_vmk(tCrSFB, k_atoms)
        if const_expr(cute.size(tCrSFB, mode=[1]) != cute.size(tCrB, mode=[1])):
            tCrSFB = cute.composition(
                tCrSFB, (None, cute.make_layout(cute.size(tCrB, mode=[1])), None)
            )
        cute.recast_tensor(tCrSFA, cutlass.Int8).fill(127)
        cute.recast_tensor(tCrSFB, cutlass.Int8).fill(127)
        return tCrSFA, tCrSFB

    @cute.jit
    def _issue_peel(
        self,
        ab_pipeline,
        ab_read_state,
        tiled_mma_peel: cute.TiledMma,
        acc: cute.Tensor,
        tiled_copy_pa,
        tCsPa_view,
        tCrPa,
        tCrPa_view,
        tiled_copy_pb,
        tCsPb_view,
        tCrPb,
        tCrPb_view,
    ):
        """Issue the BF16 peel from this tile's final A/B ring slot and release it."""
        ab_pipeline.consumer_wait(ab_read_state, ab_pipeline.consumer_try_wait(ab_read_state))
        slot = ab_read_state.index
        for k_block in cutlass.range_constexpr(cute.size(tCrPa, mode=[2])):
            cute.copy(
                tiled_copy_pa,
                tCsPa_view[None, None, k_block, slot],
                tCrPa_view[None, None, k_block],
            )
            cute.copy(
                tiled_copy_pb,
                tCsPb_view[None, None, k_block, slot],
                tCrPb_view[None, None, k_block],
            )
            cute.gemm(
                tiled_mma_peel, acc, tCrPa[None, None, k_block], tCrPb[None, None, k_block], acc
            )
        # TMA writes the slot through the async proxy while ldmatrix read it
        # through the generic proxy: fence before the release, one lane signals.
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        ab_pipeline.consumer_release(ab_read_state)
        ab_read_state.advance()
        return ab_read_state

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
        self.d_dtype = mD.element_type
        self.c_dtype = None
        self.sf_dtype = None
        self.a_smem_dtype = self.a_dtype
        self.b_smem_dtype = self.b_dtype
        self.a_tma_internal_dtype = None
        self.b_tma_internal_dtype = None
        self.varlen_m = False
        self.varlen_k = False
        self.a_layout = LayoutEnum.from_tensor(mAq)
        self.b_layout = LayoutEnum.from_tensor(mBq)
        self.d_layout = LayoutEnum.from_tensor(mD)
        self.c_layout = None
        assert self.a_layout.is_k_major_a() and self.b_layout.is_k_major_b(), (
            "the fused SM120 GEMM takes K-major (row-major (m, k) / (n, k)) FP8 operands"
        )

        self._setup_attributes(self._fused_extra_smem_bytes())
        self._check_fragment_ownership(self.tiled_mma)
        assert self.epi_tile[0] == self.tile_m and self.epi_tile[1] % _MMA_INST_N == 0

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = self.make_tma_load_atoms_and_tensors(
            mAq, mBq, a_smem_layout, b_smem_layout, varlen_k=False
        )
        self.num_tma_load_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
            mD, self.epi_smem_layout_staged, self.epi_tile, op_type="store"
        )
        bf16 = cutlass.BFloat16
        # Peel operands ride the main AB TMA ring: one extra ring slot per
        # output tile, BF16-swizzled views aliased onto that slot's A/B
        # regions (tile x 2*R2 bytes <= tile x tile_k FP8 bytes). Zero
        # dedicated smem or mbarriers, so the stock ab_stage count survives.
        assert self.cta_tile_shape_mnk[2] >= 2 * R2, (
            "peel must fit inside one AB slot (tile_k >= 2*R2 bytes)"
        )

        def _peel_staged(one, region_bytes_fp8):
            # Stage stride = one full A (resp. B) slot, in bf16 elems.
            outer = cute.make_layout(
                (one.outer.shape[0], one.outer.shape[1], self.ab_stage),
                stride=(one.outer.stride[0], one.outer.stride[1], region_bytes_fp8 // 2),
            )
            return cute.make_composed_layout(one.inner, 0, outer)

        peel_tiler = (self.tile_m, self.tile_n, R2)
        pa_one = sm90_utils.make_smem_layout_a(LayoutEnum.ROW_MAJOR, peel_tiler, bf16, 1)
        pb_one = sm90_utils.make_smem_layout_b(LayoutEnum.ROW_MAJOR, peel_tiler, bf16, 1)
        pa_smem_layout_staged = _peel_staged(
            pa_one, cute.size_in_bytes(self.a_dtype, a_smem_layout)
        )
        pb_smem_layout_staged = _peel_staged(
            pb_one, cute.size_in_bytes(self.b_dtype, b_smem_layout)
        )
        tma_atom_pa, tma_tensor_pa = self._make_tma_atoms_and_tensors(
            mPa, cute.slice_(pa_one, (None, None, 0)), (self.tile_m, R2), 1
        )
        tma_atom_pb, tma_tensor_pb = self._make_tma_atoms_and_tensors(
            mPb, cute.slice_(pb_one, (None, None, 0)), (self.tile_n, R2), 1
        )
        self.num_peel_tx_bytes = cute.size_in_bytes(
            bf16, cute.slice_(pa_one, (None, None, 0))
        ) + cute.size_in_bytes(bf16, cute.slice_(pb_one, (None, None, 0)))
        tiled_mma_peel = self._make_tiled_mma_peel()
        assert tiled_mma_peel.partition_shape_C(
            (self.tile_m, self.tile_n)
        ) == self.tiled_mma.partition_shape_C((self.tile_m, self.tile_n)), (
            "peel MMA must share the mainloop accumulator fragment"
        )

        tile_sched_args = TileSchedulerArguments(
            problem_shape_ntile_mnl=(
                cute.ceil_div(self.problem_m, self.tile_m),
                cute.ceil_div(self.problem_n, self.tile_n),
                1,
            ),
            raster_order=RasterOrderOption.Heuristic,
            group_size=Int32(_RASTER_GROUP_SIZE),
            cluster_shape_mnk=self.cluster_shape_mnk,
            persistence_mode=PersistenceMode.STATIC,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid = TileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        sext_size = self.mma_warp_groups * self.msgs_wg * _SEXT_PAD

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged)

        @cute.struct
        class SharedStorage:
            sExt: cute.struct.MemRange[Uint32, sext_size]
            sAlA: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, self.tile_m], 16]
            sAlB: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_n], 16]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size], self.buffer_align_bytes
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
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
            tma_atom_d,
            tma_tensor_d,
            tma_atom_pa,
            tma_tensor_pa,
            tma_atom_pb,
            tma_tensor_pb,
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
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            pa_smem_layout_staged,
            pb_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    # noqa C901: a warp-specialized persistent kernel. Its branches are
    # compile-time role dispatch that must stay in one traced body.
    @cute.kernel
    def kernel(  # noqa: C901
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_peel: cute.TiledMma | None,
        tma_atom_a: cute.CopyAtom,
        mAq: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mBq: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD: cute.Tensor,
        tma_atom_pa: cute.CopyAtom,
        mPa: cute.Tensor,
        tma_atom_pb: cute.CopyAtom,
        mPb: cute.Tensor,
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
        cluster_layout_mnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        pa_smem_layout: cute.ComposedLayout,
        pb_smem_layout: cute.ComposedLayout,
        tile_sched_params,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_pa, tma_atom_pb):
                cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
        )
        sched_pipeline = self.make_sched_pipeline(cluster_layout_mnk)
        sched_data = smem.allocate_tensor(
            Int32,
            cute.make_layout((4, self.sched_stage)),
            byte_alignment=16,
            partition=SmemPartition.RESERVED,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        # BF16-swizzled aliases over the A/B regions of every ring slot.
        sPa = storage.sA.get_tensor(
            pa_smem_layout.outer, swizzle=pa_smem_layout.inner, dtype=cutlass.BFloat16
        )
        sPb = storage.sB.get_tensor(
            pb_smem_layout.outer, swizzle=pb_smem_layout.inner, dtype=cutlass.BFloat16
        )
        sAlA = storage.sAlA.get_tensor(cute.make_layout(self.tile_m))
        sAlB = storage.sAlB.get_tensor(cute.make_layout(self.tile_n))
        # Padded stride: compress reads sExt[.., message, c] bank-conflict-free.
        sExt = storage.sExt.get_tensor(
            cute.make_layout(
                (self.mma_warp_groups, self.msgs_wg, LANES),
                stride=(self.msgs_wg * _SEXT_PAD, _SEXT_PAD, 1),
            )
        )

        TileSchedulerCreate = partial(
            TileScheduler.create, tile_sched_params, sched_data, sched_pipeline
        )
        k_tile_cnt = cute.ceil_div(cute.size(mAq, mode=[1]), self.cta_tile_shape_mnk[2])

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            # ========================= producer warpgroup =========================
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if warp_idx == self.ab_load_warp_id:
                # -------- TMA load warp (also the scheduler warp) --------
                cta_layout = cute.make_layout(1)
                gPa_all = cute.local_tile(mPa, (self.tile_m, R2), (None, 0))
                tPsPa, tPgPa = cpasync.tma_partition(
                    tma_atom_pa,
                    0,
                    cta_layout,
                    cute.group_modes(sPa, 0, 2),
                    cute.group_modes(gPa_all, 0, 2),
                )
                gPb_all = cute.local_tile(mPb, (self.tile_n, R2), (None, 0))
                tPsPb, tPgPb = cpasync.tma_partition(
                    tma_atom_pb,
                    0,
                    cta_layout,
                    cute.group_modes(sPb, 0, 2),
                    cute.group_modes(gPb_all, 0, 2),
                )

                tile_scheduler = TileSchedulerCreate(is_scheduler_warp=True)
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    gA_mk = cute.local_tile(
                        mAq,
                        cute.select(self.cta_tile_shape_mnk, [0, 2]),
                        (tile_coord_mnkl[0], None),
                    )
                    copy_A = copy_utils.tma_get_block_copy_fn(
                        tma_atom_a, src_tensor=gA_mk, dst_tensor=sA
                    )
                    gB_nk = cute.local_tile(
                        mBq,
                        cute.select(self.cta_tile_shape_mnk, [1, 2]),
                        (tile_coord_mnkl[1], None),
                    )
                    copy_B = copy_utils.tma_get_block_copy_fn(
                        tma_atom_b, src_tensor=gB_nk, dst_tensor=sB
                    )
                    ab_producer_state = self.load_tma(
                        ab_pipeline, ab_producer_state, [copy_A, copy_B], k_tile_cnt
                    )
                    # A_peel + B_peel take the (k_tile_cnt+1)-th ring slot
                    # of this tile. extra_tx_count rebases the expect-tx
                    # from the full A+B slot bytes down to the peel bytes.
                    ab_pipeline.producer_acquire(
                        ab_producer_state,
                        extra_tx_count=self.num_peel_tx_bytes - self.num_tma_load_bytes,
                    )
                    peel_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_pa,
                        tPgPa[(None, tile_coord_mnkl[0])],
                        tPsPa[(None, ab_producer_state.index)],
                        tma_bar_ptr=peel_bar,
                    )
                    cute.copy(
                        tma_atom_pb,
                        tPgPb[(None, tile_coord_mnkl[1])],
                        tPsPb[(None, ab_producer_state.index)],
                        tma_bar_ptr=peel_bar,
                    )
                    ab_pipeline.producer_commit(ab_producer_state)
                    ab_producer_state.advance()
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=True)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                ab_pipeline.producer_tail(ab_producer_state)
                tile_scheduler.producer_tail()

            # The producer warpgroup's otherwise idle warps hash: one
            # compression warp per mma warpgroup, replaying its scheduler walk.
            for warp_group_idx in cutlass.range_constexpr(self.mma_warp_groups):
                if warp_idx == self.ab_load_warp_id + 1 + warp_group_idx:
                    lane = cute.arch.lane_idx()
                    chaining_value = [mKey[i] for i in range(8)]
                    threshold = [mThr[i] for i in range(8)]
                    tile_scheduler = TileSchedulerCreate()
                    work_tile = tile_scheduler.initial_work_tile_info()
                    # The first fold has no preceding compression to wait on.
                    cute.arch.barrier_arrive(
                        barrier_id=int(_NamedBarrier.LOTTERY_BUFFER_FREE_WG0) + warp_group_idx,
                        number_of_threads=WG_THREADS + cute.arch.WARP_SIZE,
                    )
                    while work_tile.is_valid_tile:
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
                            work_tile.tile_idx,
                            warp_group_idx,
                            lane,
                        )
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()

        if warp_idx < self.ab_load_warp_id:
            # ========================= mma warps =========================
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(warp_idx == 0)
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // WG_THREADS)
            warp_in_wg = cute.arch.make_warp_uniform((tidx % WG_THREADS) // cute.arch.WARP_SIZE)
            lane = tidx % cute.arch.WARP_SIZE

            thr_mma = tiled_mma.get_slice(tidx)
            acc, tCsA, tCsB, tCrA, tCrB = sm80_utils.partition_fragment_ABC(
                thr_mma, self.cta_tile_shape_mnk, sA, sB
            )
            # ldmatrix s2r for B (K-major fp8: a k-major byte pair is one
            # 16-bit unit); A goes through the substrate's produce seam.
            atom_ldmatrix_b = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), self.b_dtype)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_ldmatrix_b, tiled_mma)
            tCsB_copy_view = smem_tiled_copy_B.get_slice(tidx).partition_S(sB)
            copy_block = self.canonical_a_load(tiled_mma, sA, tidx, tCrA)
            tCrSFA, tCrSFB = None, None
            if const_expr(self.use_mxf8_mma):
                tCrSFA, tCrSFB = self._unit_scale_fragments(tiled_mma, thr_mma, sB, tCrB, tidx)

            thr_mma_peel = tiled_mma_peel.get_slice(tidx)
            tCrPa = thr_mma_peel.make_fragment_A(thr_mma_peel.partition_A(sPa)[None, None, None, 0])
            tCrPb = thr_mma_peel.make_fragment_B(thr_mma_peel.partition_B(sPb)[None, None, None, 0])
            atom_ldmatrix_bf16 = copy_utils.get_smem_load_atom(cutlass.BFloat16)
            tiled_copy_pa = cute.make_tiled_copy_A(atom_ldmatrix_bf16, tiled_mma_peel)
            tiled_copy_pb = cute.make_tiled_copy_B(atom_ldmatrix_bf16, tiled_mma_peel)
            thr_copy_pa = tiled_copy_pa.get_slice(tidx)
            thr_copy_pb = tiled_copy_pb.get_slice(tidx)
            peel = partial(
                self._issue_peel,
                tiled_mma_peel=tiled_mma_peel,
                acc=acc,
                tiled_copy_pa=tiled_copy_pa,
                tCsPa_view=thr_copy_pa.partition_S(sPa),
                tCrPa=tCrPa,
                tCrPa_view=thr_copy_pa.retile(tCrPa),
                tiled_copy_pb=tiled_copy_pb,
                tCsPb_view=thr_copy_pb.partition_S(sPb),
                tCrPb=tCrPb,
                tCrPb_view=thr_copy_pb.retile(tCrPb),
            )

            ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            epi_store_pipeline = self.make_epi_store_pipeline()

            tile_scheduler = TileSchedulerCreate()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                # Stage this tile's alpha slices while the mainloop runs.
                self._stage_alpha_slices(mAlA, mAlB, sAlA, sAlB, tile_coord_mnkl, tidx)
                acc.fill(0.0)
                ab_read_state = self.mma(
                    ab_pipeline,
                    ab_read_state,
                    tiled_mma,
                    acc,
                    k_tile_cnt,
                    copy_block,
                    smem_tiled_copy_B,
                    tCsB_copy_view,
                    tCrA,
                    tCrB,
                    tCrSFA=tCrSFA,
                    tCrSFB=tCrSFB,
                )

                # The fold observes the accumulator before peel.
                self._stage_lottery_messages(acc, sExt, warp_group_idx, warp_in_wg, lane)

                ab_read_state = peel(ab_pipeline, ab_read_state)
                row_inverse = self._row_inverse_scales(sAlA, warp_idx, lane)
                copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                    tma_atom_d,
                    mD,
                    self.cta_tile_shape_mnk[:2],
                    self.epi_tile,
                    sD,
                    tile_coord_mnkl,
                )
                tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                    tiled_mma, self.d_layout, self.d_dtype, sD, tidx
                )
                tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
                load_acc_subtile = partial(
                    self.epi_load_acc_subtile_unscaled, tRS_rAcc, acc, row_inverse, sAlB, lane
                )
                self.epilogue(
                    None,  # params
                    {},  # epi_smem_tensors
                    None,  # epi_pipeline
                    epi_store_pipeline,
                    None,  # epi_read_state
                    None,  # epi_producer_state
                    self.epi_tile,
                    load_acc_subtile,
                    tRS_rD,
                    None,  # tRS_rC
                    None,  # tiled_copy_t2r
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
                    tidx,
                    is_tma_warp,
                )
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Wait for D store complete.
            if is_tma_warp:
                epi_store_pipeline.producer_tail()
