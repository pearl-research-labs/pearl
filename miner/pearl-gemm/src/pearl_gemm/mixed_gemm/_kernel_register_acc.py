"""Fused phases shared by the register-accumulator kernels (SM90 and SM120).

Quack's ``GemmSm90`` and its ``GemmSm120`` subclass both keep the accumulator
in the consumer warps' registers under one warp topology, which fixes most of
the fused kernel:

- the producer warpgroup's TMA load warp (also the scheduler warp) fills the
  FP8 A/B ring and, per output tile, one extra ring slot carrying the BF16
  peel operands, aliased onto that slot's FP8 bytes with a reduced
  expect-tx (``_load_warp``);
- the producer warpgroup's spare warps hash -- one compression warp
  per consumer warpgroup, replaying its scheduler walk -- and publish the
  first winner, overlapping the peel and the next tile's mainloop
  (``_compression_warps``);
- the consumer warpgroups run the mainloop, fold the pre-peel accumulator
  into ``sExt``, issue the peel into the same registers, and run the
  TMA-store epilogue with the row/column unscale fused into the subtile
  loads.

``_RegisterAccFusedGemm`` holds the launch (``__call__``), the shared memory
layout, the pipelines, the producer warpgroup, the lottery fold
(``_stage_lottery_messages``), the hash-and-publish loop, and the alpha
staging. Each family supplies its consumer warps (``_consumer_warps``), the fold's accumulator
fragment index (``_lottery_fragment_index``), the unscale, the peel MMA
(``_make_tiled_mma_peel``), the fragment-layout check, the TMA multicast, and
the message-to-row decode (``_message_lottery_row``) matching its fold.

``grouped_mixed_gemm``'s SM90 and SM120 subclasses compile the MoE variant of
the same kernel: each work tile belongs to one expert, whose operands are per-expert
views (ragged A / D rows, offset A_peel rows and B / B_peel columns) and
whose lottery restarts at the expert's first row. It overrides the per-tile
hooks (``_tile_group`` and the helpers that take its ``group``), each the
identity on a dense tile, so the dense trace is unchanged.
"""

import enum
from functools import partial
from typing import NamedTuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import quack.copy_utils as copy_utils
from cutlass import Boolean, Float32, Int32, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum, SmemPartition
from quack.cute_dsl_utils import mlir_namedtuple
from quack.epilogue.ops import EpiSmemBytes
from quack.pipeline import PipelineAsync as BasePipelineAsync
from quack.pipeline import PipelineTmaAsync as BasePipelineTmaAsync
from quack.pipeline import make_pipeline_state
from quack.tile_scheduler import (
    PersistenceMode,
    RasterOrderOption,
    TileScheduler,
    TileSchedulerArguments,
)

from ..tensor_hash_plus_stats._blake3_ops import SINGLE_BLOCK_KEYED_FLAGS, compress
from ._lottery import _SEXT_PAD, LANES, R2, _fold_word, _HitPublishMixin

LTILE_ROWS = 4  # the only register-accumulator lottery tile row count
WG_THREADS = 128
# sExt capacity, in lottery messages per consumer warpgroup: each
# compression-warp lane hashes at most three.
_MAX_MSGS_WG = 96
# setmaxnreg budget of the whole CTA: 65536 exactly fills the register file
# and setmaxnreg.inc then spin-waits forever.
_MAX_CTA_REGS = 64512
# Shared-memory slack Quack's stage solver must leave for the fused fields:
# the dummy fields and extra 1024-byte alignment pads of our SharedStorage
# over the stock one, plus the 16-byte alignment of sExt and the two alpha
# slices.
_SMEM_ALIGN_SLACK = 3072
_SEXT_ALIGN_SLACK = 128
_ALPHA_ALIGN_SLACK = 32

_RASTER_GROUP_SIZE = 8  # tile-scheduler rasterization/serpentine group size


class _NamedBarrier(enum.IntEnum):
    """Barrier IDs after Quack's reserved ``NamedBarrierGemm`` range (1..8);
    ``*_WG0 + consumer_warpgroup`` selects a warpgroup's copy."""

    FOLD_DONE_WG0 = 9
    FOLD_DONE_WG1 = 10
    LOTTERY_BUFFER_FREE_WG0 = 11
    LOTTERY_BUFFER_FREE_WG1 = 12
    ALPHA_READY = 13


@mlir_namedtuple
class _Grouping(NamedTuple):
    """Grouped-launch operands: ``m_indptr`` / ``m_valid`` (see
    ``grouped_mixed_gemm``) and the problem extents -- permuted rows
    ``cum_m``, per-expert columns ``n`` and ``k`` -- which a grouped launch
    compiles symbolically, so the kernel takes them as arguments."""

    m_indptr: cute.Tensor
    m_valid: cute.Tensor
    cum_m: Int32
    n: Int32
    k: Int32


class _TileGroup(NamedTuple):
    """One work tile's expert: its index, its block of the permuted rows
    ``[row0, row0 + rows)`` and its real (publishable) row count."""

    index: Int32
    row0: Int32
    rows: Int32
    valid_rows: Int32


class _RegisterAccFusedGemm(_HitPublishMixin):
    """Launch, producer warpgroup, and hashing shared by the SM90 and SM120
    fused GEMMs; mixed in ahead of the Quack substrate class."""

    # The grouped (MoE) subclass flips both; see the module docstring.
    grouped = False
    scheduler_cls = TileScheduler

    # Quack's stage-count hook requires the full signature; this epilogue
    # only needs its fused shared-memory byte count.
    @staticmethod
    def epi_smem_bytes(
        extra_bytes, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None
    ) -> EpiSmemBytes:
        return EpiSmemBytes(unstaged=extra_bytes)

    # No epilogue ops and RN rounding, so the stochastic-rounding seed mapping
    # is empty (same as ``_kernel.py``).
    def epi_begin_loop(self, params, epi_tensors, epi_coord) -> dict[str, cute.Tensor]:
        return {}

    def _fused_extra_smem_bytes(self) -> int:
        # The peel costs nothing: it aliases the A/B ring slots.
        extra = _SMEM_ALIGN_SLACK
        extra += self.mma_warp_groups * self.msgs_wg * _SEXT_PAD * 4 + _SEXT_ALIGN_SLACK
        # sAlA (bf16) + sAlB (f32)
        extra += self.tile_m * 2 + self.tile_n * 4 + _ALPHA_ALIGN_SLACK
        return extra

    # -- ab ring pipeline: Quack's TmaAsync wrapper (its producer_acquire takes
    # extra_tx_count, which the peel ring slot needs to rebase its expect-tx) --
    def make_ab_pipeline(self, tiled_mma: cute.TiledMma, cluster_layout_vmnk: cute.Layout):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # One arrive per consumer warp, delivered to every CTA of its
        # multicast groups (self counted once).
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * tiled_mma.size // cute.arch.WARP_SIZE
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
    def make_sched_pipeline(self, cluster_layout_mnk: cute.Layout):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        consumer_arrive_cnt = (
            self.mma_warp_groups * 4  # four warps per consumer warpgroup
            + self.num_ab_load_warps  # producer warpgroup's load warp
            + self.mma_warp_groups  # one compression warp per consumer warpgroup
        ) * cluster_size
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt)
        return BasePipelineAsync.create(
            num_stages=self.sched_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
            defer_sync=True,
            elect_one_release=True,
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
        group: _TileGroup | None,
        warp_group_idx: cutlass.Constexpr,
        lane: Int32,
    ):
        """Hash one consumer warpgroup's staged messages and publish the winner.

        Each compression-warp lane hashes ceil(msgs_wg / 32) messages. The
        publish ballot needs a convergent warp, so a sub-warp message tail
        (msgs_wg is a multiple of 16, not necessarily 32) keeps its whole
        warp hashing: excess lanes rehash the last message and are excluded
        from the ballot.
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
            tile_row = self._message_lottery_row(
                tile_coord_mnkl[0], message // self.column_tiles, warp_group_idx
            )
            tile_column = tile_coord_mnkl[1] * self.column_tiles + column_tile
            in_bounds, hit_group = self._lottery_tile_bounds(tile_row, tile_column, group)
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
                hit_group,
            )
        cute.arch.barrier_arrive(
            barrier_id=int(_NamedBarrier.LOTTERY_BUFFER_FREE_WG0) + warp_group_idx,
            number_of_threads=WG_THREADS + cute.arch.WARP_SIZE,
        )

    @cute.jit
    def _stage_lottery_messages(
        self,
        accumulator: cute.Tensor,
        sExt: cute.Tensor,
        warp_group_idx: Int32,
        warp_in_wg: Int32,
        lane: Int32,
    ):
        """Fold and stage one consumer warpgroup's pre-peel accumulator, then
        hand it to the compression warp (``FOLD_DONE``) once the previous
        tile's words have been consumed (``LOTTERY_BUFFER_FREE``).

        Both families' C fragments give lane ``T`` of warp ``w`` rows
        ``16 w + (T >> 2) + 8 h`` of each 64-row M-atom and columns
        ``8 nb + 2 (T & 3) + c``, so word ``T & 15`` of lottery row
        ``4 w + 2 h + (T >> 4)`` folds, in ascending column order, the
        column-tile cells at ``_lottery_fragment_index(mma_row, h, c, nb)``.
        """
        blocks_per_column_tile = self.ltile_cols // self.fragment_n_block
        cute.arch.barrier(
            barrier_id=int(_NamedBarrier.LOTTERY_BUFFER_FREE_WG0) + warp_group_idx,
            number_of_threads=WG_THREADS + cute.arch.WARP_SIZE,
        )
        for mma_row in cutlass.range_constexpr(self.mma_m_per_wg):
            for row_half in cutlass.range_constexpr(2):
                for column_tile in cutlass.range_constexpr(self.column_tiles):
                    folded = Uint32(0)
                    for block in cutlass.range_constexpr(blocks_per_column_tile):
                        for column in cutlass.range_constexpr(2):
                            word = accumulator[
                                self._lottery_fragment_index(
                                    mma_row,
                                    row_half,
                                    column,
                                    column_tile * blocks_per_column_tile + block,
                                )
                            ].bitcast(Uint32)
                            folded = _fold_word(folded, word)
                    thread_row = 4 * warp_in_wg + 2 * row_half + (lane >> 4)
                    message = (
                        thread_row * self.mma_m_per_wg + mma_row
                    ) * self.column_tiles + column_tile
                    sExt[warp_group_idx, message, lane & (LANES - 1)] = folded
        cute.arch.barrier_arrive(
            barrier_id=int(_NamedBarrier.FOLD_DONE_WG0) + warp_group_idx,
            number_of_threads=WG_THREADS + cute.arch.WARP_SIZE,
        )

    @cute.jit
    def _stage_alpha_slices(self, slices, tile_coord_mnkl, tidx: Int32):
        """Stage this tile's scale slices, each ``(global_scale, shared_scale,
        coordinate_mode, tile_extent)``, with asynchronous copies."""
        alpha_threads = self.mma_warp_groups * WG_THREADS
        for global_scale, shared_scale, coordinate_mode, tile_extent in slices:
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
    def _stage_tile_alphas(
        self,
        mAlA: cute.Tensor,
        mAlB: cute.Tensor,
        sAlA: cute.Tensor,
        sAlB: cute.Tensor,
        group: _TileGroup | None,
        tile_coord_mnkl,
        warp_idx: Int32,
        lane: Int32,
        tidx: Int32,
    ):
        """Stage this tile's row and column scales while the mainloop runs.
        Returns the register copy of this lane's row scales, which a dense
        tile does not need (the unscale reads ``sAlA``)."""
        self._stage_alpha_slices(
            ((mAlA, sAlA, 0, self.tile_m), (mAlB, sAlB, 1, self.tile_n)), tile_coord_mnkl, tidx
        )
        return None

    @cute.jit
    def _wait_alpha_slices(self):
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(
            barrier_id=int(_NamedBarrier.ALPHA_READY),
            number_of_threads=self.mma_warp_groups * WG_THREADS,
        )

    @cute.jit
    def _store_epilogue(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_d: cute.CopyAtom,
        mD: cute.Tensor,
        sD: cute.Tensor,
        acc: cute.Tensor,
        load_acc_subtile,
        epi_store_pipeline,
        tile_scheduler,
        tile_coord_mnkl,
        tidx: Int32,
        is_tma_warp: Boolean,
    ):
        """Quack's TMA-store epilogue for this tile; ``load_acc_subtile`` binds
        the family's fused unscale to the retiled accumulator view."""
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
        self.epilogue(
            None,  # params
            {},  # epi_smem_tensors
            None,  # epi_pipeline
            epi_store_pipeline,
            None,  # epi_read_state
            None,  # epi_producer_state
            self.epi_tile,
            partial(load_acc_subtile, tRS_rAcc),
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
        grouping: _Grouping | None = None,
    ):
        """Trace-time setup and launch. ``grouping`` (the grouped variant
        only) makes ``mAq`` the permuted ``(cum_m, k)`` rows, ``mBq`` the
        stacked ``(E * n, k)`` experts and every other operand likewise."""
        assert (grouping is not None) == self.grouped
        payload_tensors = (mCodesSrc, mScalesSrc, mCodesDst, mScalesDst)
        assert all((t is not None) == self.snapshot_payload for t in payload_tensors), (
            "payload tensor presence must match the compiled snapshot variant"
        )
        # Static problem shape and payload byte counts, baked into the hit
        # record writes (shapes are compile-time for a given launch). A
        # grouped launch's m is the permuted row total and n is per expert.
        self.problem_m = cute.size(mAq, mode=[0])
        self.problem_n = cute.size(mBq, mode=[0])
        self.problem_k = cute.size(mAq, mode=[1])
        if const_expr(grouping is not None):
            self.problem_m, self.problem_n, self.problem_k = grouping.cum_m, grouping.n, grouping.k
        # Grouped launches are compiled over symbolic shapes (the host checks
        # the planes) and size the snapshot per hit, the winning expert's rows
        # against the signal's capacity.
        if const_expr(grouping is None):
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
        # The substrate reads separate smem/TMA dtypes and the varlen modes;
        # the fused kernel is dense FP8 throughout.
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
            "the fused GEMM takes K-major (row-major (m, k) / (n, k)) FP8 operands"
        )

        self._setup_attributes(self._fused_extra_smem_bytes())
        self._check_accumulator_layout(self.tiled_mma)
        assert (
            WG_THREADS * (self.num_regs_load + self.mma_warp_groups * self.num_regs_mma)
            <= _MAX_CTA_REGS
        ), f"register split must fit the {_MAX_CTA_REGS}-register setmaxnreg budget"
        # The unscale scales the accumulator in place, so each subtile must be
        # loaded exactly once: the substrate's prepass variant would load it twice.
        assert not self.epi_needs_acc_prepass, (
            "in-place unscale needs a single acc load per subtile"
        )

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = self.make_tma_load_atoms_and_tensors(
            self._launch_row_operand(mAq), mBq, a_smem_layout, b_smem_layout, varlen_k=False
        )
        self.num_tma_load_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
            self._launch_row_operand(mD, store=True),
            self.epi_smem_layout_staged,
            self.epi_tile,
            op_type="store",
        )
        bf16 = cutlass.BFloat16
        # Peel operands ride the main AB TMA ring: one extra ring slot per
        # output tile, BF16-swizzled views aliased onto that slot's A/B
        # regions (tile x 2*R2 bytes <= tile x tile_k FP8 bytes). Zero
        # dedicated smem, mbarriers, or registers, so the stock ab_stage
        # count survives and heavy tiles don't spill.
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

        tile_sched_params = self._tile_scheduler_params(grouping)
        grid = self.scheduler_cls.get_grid_shape(tile_sched_params, max_active_clusters)

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
            grouping,
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
        tiled_mma_peel: cute.TiledMma,
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
        grouping: _Grouping | None,
    ):
        if const_expr(grouping is not None):
            # The device-side extents: the launch's values are host SSA.
            self.problem_m, self.problem_n, self.problem_k = grouping.cum_m, grouping.n, grouping.k
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

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
        # The peel's BF16-swizzled aliases of every ring slot (see __call__).
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
            self.scheduler_cls.create, tile_sched_params, sched_data, sched_pipeline
        )
        k_tile_cnt = cute.ceil_div(cute.size(mAq, mode=[1]), self.cta_tile_shape_mnk[2])

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            # ========================= producer warpgroup =========================
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if warp_idx == self.ab_load_warp_id:
                self._load_warp(
                    ab_pipeline,
                    TileSchedulerCreate,
                    tma_atom_a,
                    mAq,
                    tma_atom_b,
                    mBq,
                    tma_atom_pa,
                    mPa,
                    tma_atom_pb,
                    mPb,
                    sA,
                    sB,
                    sPa,
                    sPb,
                    cluster_layout_mnk,
                    k_tile_cnt,
                    grouping,
                )
            self._compression_warps(
                warp_idx,
                TileSchedulerCreate,
                sExt,
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
                grouping,
            )

        if warp_idx < self.ab_load_warp_id:
            # ========================= consumer warpgroups =========================
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            self._consumer_warps(
                warp_idx,
                tiled_mma,
                tiled_mma_peel,
                ab_pipeline,
                TileSchedulerCreate,
                tma_atom_d,
                mD,
                mAlA,
                mAlB,
                sA,
                sB,
                sD,
                sPa,
                sPb,
                sAlA,
                sAlB,
                sExt,
                k_tile_cnt,
                grouping,
            )

    @cute.jit
    def _load_warp(
        self,
        ab_pipeline,
        TileSchedulerCreate,
        tma_atom_a: cute.CopyAtom,
        mAq: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mBq: cute.Tensor,
        tma_atom_pa: cute.CopyAtom,
        mPa: cute.Tensor,
        tma_atom_pb: cute.CopyAtom,
        mPb: cute.Tensor,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sPa: cute.Tensor,
        sPb: cute.Tensor,
        cluster_layout_mnk: cute.Layout,
        k_tile_cnt: Int32,
        grouping: _Grouping | None,
    ):
        """TMA load warp (also the scheduler warp): the FP8 A/B ring, then the
        tile's peel slot."""
        a_tma_multicast, b_tma_multicast = self._ab_tma_multicast()
        is_scheduler_warp = True
        if const_expr(cute.size(cluster_layout_mnk) > 1):
            is_scheduler_warp = cute.arch.block_idx_in_cluster() == 0

        tPsPa, tPgPa, tPsPb, tPgPb = (None,) * 4
        if const_expr(not self.grouped):
            tPsPa, tPgPa, tPsPb, tPgPb = self._peel_tma_partitions(
                tma_atom_pa, mPa, sPa, tma_atom_pb, mPb, sPb
            )

        tile_scheduler = TileSchedulerCreate()
        work_tile = tile_scheduler.initial_work_tile_info()
        ab_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
        while work_tile.is_valid_tile:
            tile_coord_mnkl = work_tile.tile_idx
            group = self._tile_group(grouping, tile_coord_mnkl)
            gA_mk = cute.local_tile(
                self._row_operand(mAq, group),
                cute.select(self.cta_tile_shape_mnk, [0, 2]),
                (tile_coord_mnkl[0], None),
            )
            copy_A = copy_utils.tma_get_block_copy_fn(
                tma_atom_a,
                src_tensor=gA_mk,
                dst_tensor=sA,
                tma_multicast=a_tma_multicast,
            )
            gB_nk = cute.local_tile(
                self._column_operand(mBq, group),
                cute.select(self.cta_tile_shape_mnk, [1, 2]),
                (tile_coord_mnkl[1], None),
            )
            copy_B = copy_utils.tma_get_block_copy_fn(
                tma_atom_b,
                src_tensor=gB_nk,
                dst_tensor=sB,
                tma_multicast=b_tma_multicast,
            )
            ab_producer_state = self.load_tma(
                ab_pipeline, ab_producer_state, [copy_A, copy_B], k_tile_cnt
            )
            tile_sPa, tile_gPa, tile_sPb, tile_gPb = self._tile_peel_partitions(
                (tPsPa, tPgPa, tPsPb, tPgPb),
                tma_atom_pa,
                mPa,
                sPa,
                tma_atom_pb,
                mPb,
                sPb,
                group,
            )
            # A_peel + B_peel take the (k_tile_cnt+1)-th ring slot of this
            # tile. extra_tx_count rebases the expect-tx from the full A+B
            # slot bytes down to the peel bytes.
            ab_pipeline.producer_acquire(
                ab_producer_state,
                extra_tx_count=self.num_peel_tx_bytes - self.num_tma_load_bytes,
            )
            peel_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
            cute.copy(
                tma_atom_pa,
                tile_gPa[(None, tile_coord_mnkl[0])],
                tile_sPa[(None, ab_producer_state.index)],
                tma_bar_ptr=peel_bar,
            )
            cute.copy(
                tma_atom_pb,
                tile_gPb[(None, tile_coord_mnkl[1])],
                tile_sPb[(None, ab_producer_state.index)],
                tma_bar_ptr=peel_bar,
            )
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
            tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
            work_tile = tile_scheduler.get_current_work()
        ab_pipeline.producer_tail(ab_producer_state)
        if is_scheduler_warp:
            tile_scheduler.producer_tail()

    @cute.jit
    def _compression_warps(
        self,
        warp_idx: Int32,
        TileSchedulerCreate,
        sExt: cute.Tensor,
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
        grouping: _Grouping | None,
    ):
        """The producer warpgroup's spare warps: one compression warp
        per consumer warpgroup, replaying its scheduler walk."""
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
                        self._tile_group(grouping, work_tile.tile_idx),
                        warp_group_idx,
                        lane,
                    )
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()

    def _tile_scheduler_params(self, grouping: _Grouping | None):
        """The dense launch's persistent tile schedule; the grouped subclass
        schedules over experts instead."""
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
        return TileScheduler.to_underlying_arguments(tile_sched_args)

    # -- per-tile hooks: the identity on a dense tile; the grouped subclass
    # resolves each work tile to its expert's views --

    def _launch_row_operand(self, tensor: cute.Tensor, *, store: bool = False):
        """The launch-wide TMA view of a row operand (A', D)."""
        return tensor

    @cute.jit
    def _tile_group(self, grouping: _Grouping | None, tile_coord_mnkl) -> _TileGroup | None:
        """The work tile's group (``None``: dense)."""
        return None

    def _row_operand(self, tensor: cute.Tensor, group: _TileGroup | None, *, store: bool = False):
        """The tile's view of a row operand (A', D)."""
        return tensor

    def _column_operand(self, tensor: cute.Tensor, group: _TileGroup | None):
        """The tile's view of a stacked column operand (B', B_peel)."""
        return tensor

    @cute.jit
    def _tile_peel_partitions(
        self,
        hoisted,
        tma_atom_pa: cute.CopyAtom,
        mPa: cute.Tensor,
        sPa: cute.Tensor,
        tma_atom_pb: cute.CopyAtom,
        mPb: cute.Tensor,
        sPb: cute.Tensor,
        group: _TileGroup | None,
    ):
        """The tile's peel TMA partitions: a dense tile reuses the ``hoisted``
        launch-wide ones."""
        return hoisted

    @cute.jit
    def _lottery_tile_bounds(self, tile_row: Int32, tile_column: Int32, group: _TileGroup | None):
        """Whether lottery tile ``(tile_row, tile_column)`` may publish, and
        its ``_HitGroup`` (``None``: dense)."""
        in_bounds = Boolean(tile_row * LTILE_ROWS < self.problem_m) & Boolean(
            tile_column * self.ltile_cols < self.problem_n
        )
        return in_bounds, None

    @cute.jit
    def _peel_tma_partitions(
        self,
        tma_atom_pa: cute.CopyAtom,
        mPa: cute.Tensor,
        sPa: cute.Tensor,
        tma_atom_pb: cute.CopyAtom,
        mPb: cute.Tensor,
        sPb: cute.Tensor,
    ):
        """TMA partitions of the peel operands, indexed by M / N tile."""
        cta_layout = cute.make_layout(1)
        gPa_all = cute.local_tile(mPa, (self.tile_m, R2), (None, 0))  # (tile_m, R2, mt)
        tPsPa, tPgPa = cpasync.tma_partition(
            tma_atom_pa,
            0,
            cta_layout,
            cute.group_modes(sPa, 0, 2),
            cute.group_modes(gPa_all, 0, 2),
        )
        gPb_all = cute.local_tile(mPb, (self.tile_n, R2), (None, 0))  # (tile_n, R2, nt)
        tPsPb, tPgPb = cpasync.tma_partition(
            tma_atom_pb,
            0,
            cta_layout,
            cute.group_modes(sPb, 0, 2),
            cute.group_modes(gPb_all, 0, 2),
        )
        return tPsPa, tPgPa, tPsPb, tPgPb
