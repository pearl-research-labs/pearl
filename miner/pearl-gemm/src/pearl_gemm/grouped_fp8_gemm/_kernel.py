# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Derived from the CUTLASS CuTe DSL examples
# ``blackwell/blockwise_gemm/blockwise_gemm.py`` (mainloop, scale path,
# accumulator pipeline, epilogue) and ``blackwell/grouped_gemm.py`` (static
# group schedule), modified for M-grouped (MoE) operands described by an
# ``m_indptr`` prefix-sum array and addressed through launch-wide TMA
# descriptors with ragged row bounds instead of per-group descriptor updates.
"""SM100 grouped blockwise-scaled FP8 GEMM in CuTe DSL.

Computes, for every group ``g``::

    out[m_indptr[g]:m_indptr[g+1]] = (a[rows] * sfa) @ (b[g] * sfb[g]).T

with e4m3 inputs, 1x128 (A) / 128x128 (B) FP32 scale factors and a BF16 output.
Terminology: this module says *group*; a group is one expert's rows (the
runtime says *expert*). The design follows CUTLASS' ``KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100``
/``2SmSm100`` kernels: a 128 x ``tile_n`` x 128 CTA tile (``tile_n`` 128 or
256; 2-CTA MMA for ``mma_sm=2``), TMA-fed A/B stages, cp.async-fed scale
stages, a TMEM accumulator pipeline, a register scale-accumulate pass
(blockwise) or a direct TMEM-to-epilogue scale (full-K), and a TMA-store
epilogue. Grouping is
descriptor-free: A and C are described by one launch-wide TMA descriptor each
with a ragged row mode (see ``_RAGGED_ROWS``) so a group is a coordinate
offset whose rows past the group's end fall outside the descriptor (loads
zero-fill, stores clip), and B is addressed by its plain row offset. The
persistent tile schedule (scheduler warp) is either the SM90-style static
group schedule or a dynamic one fed by a global atomic tile counter; both
resolve linear tile indices against ``m_indptr`` with a warp-cooperative scan
(tiles rasterized along M within a group).

Warp roles (12 warps, 384 threads):

* warps 0-3: accumulator-update warps. Blockwise: TMEM -> registers,
  ``acc*sfa*sfb`` accumulated across K tiles, final accumulator back to a
  dedicated TMEM region. Full-K (``full_k_acc``): no accumulator work (they
  walk the tile pipeline); mining: fold + hash + publish of the pre-peel
  accumulator.
* warps 4-7: epilogue warps (TMEM -> registers -> SMEM -> TMA store).
  Blockwise reads the final region; full-K reads the accumulator stage
  directly and applies the scale (``sfa*sfb`` or the mining unscale) itself.
* warp 8: MMA warp
* warp 9: TMA load warp (A/B tiles)
* warp 10: scale load warp (cp.async SFA/SFB)
* warp 11: scheduler warp
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Boolean, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from quack.utils import store_shared_remote

from ..mixed_gemm._kernel import (
    _FOLD_MUL,
    _SEXT_PAD,
    DEFAULT_LTILE_COLS,
    DEFAULT_LTILE_ROWS,
    LANES,
    R2,
    SUPPORTED_LTILE_COLS,
    SUPPORTED_LTILE_ROWS,
    _fold_register_groups,
    _HitGroup,
    _HitPublishMixin,
    _thread_fold_cells,
)
from ..tensor_hash_plus_stats._blake3 import _rotr32
from ..tensor_hash_plus_stats._blake3_ops import SINGLE_BLOCK_KEYED_FLAGS, compress

# Ragged-row TMA views (the Triton / Quack ``ragged_tma`` trick). A row-grouped
# operand is described to TMA by ONE descriptor whose row mode has the fixed
# extent ``_RAGGED_ROWS`` and two extra unit-box modes. For a group of
# ``length`` rows starting at ``offset``, the row coordinate is
# ``_RAGGED_ROWS - length + r`` and the extra coordinates are
# ``(_RAGGED_ROWS, offset + length)``; the strides are chosen so the address
# is ``base + (offset + r) * row_stride`` modulo 2**64 (the
# ``_RAGGED_ROWS * 2**34`` term wraps to zero) while every ``r >= length``
# is out of the descriptor's bounds: loads zero-fill, stores are dropped.
# Neither the base pointer nor a per-group descriptor rewrite is needed.
_RAGGED_ROWS = 2**30
_RAGGED_EXTRA = 2**31 - 1
_RAGGED_WRAP = 2**64 // _RAGGED_ROWS  # 2**34

SCHEDULERS = ("static", "dynamic")
# Output rows per CTA for either UMMA shape (the 2-CTA instruction spans the
# pair, so the cluster tile is ``CTA_TILE_M * mma_sm`` rows).
CTA_TILE_M = 128
# Dynamic scheduler state, one Int32 pair per device: [0] next linear tile to
# hand out (past the initial static assignment), [1] clusters that have
# finished fetching. Both are zero between launches (the kernel resets them).
SCHED_COUNTER_WORDS = 2
# Groups resolved per warp-cooperative ``m_indptr`` window scan (one lane
# per group, the last lane only supplies the window's end row).
_SCAN_GROUPS_PER_ITER = 31

# Mining mode: the accumulator-update warps (4 warps, 128 threads) hash one
# lottery message per thread, as the dense ``mixed_gemm`` epilogue warps do.
_ACC_THREADS = 128


class GroupedGemmSm100(_HitPublishMixin):
    """Grouped e4m3 x e4m3 -> bf16 GEMM with 1x128 / 128x128 FP32 scales.

    ``mining=True`` compiles the MoE mining variant on top of the full-K
    accumulation order (``full_k_acc``): the blockwise scales are gone, warps
    0-3 fold the pre-peel TMEM accumulator of every lottery tile into a 64-byte
    message, hash it with keyed BLAKE3 and publish the first winner into the
    process-wide hit signal (``pow/_hit_signal.py``), and the MMA warp adds
    the BF16 rank-2r peel before the epilogue warps read the accumulator,
    apply the protocol's per-row ``1/alpha_a`` and per-column ``inv_alpha_b``
    unscale and TMA-store the BF16 tile.
    """

    def __init__(
        self,
        *,
        scale_major_k: bool,
        mma_sm: int,
        full_k_acc: bool = False,
        mining: bool = False,
        snapshot_payload: bool = False,
        ltile_rows: int = DEFAULT_LTILE_ROWS,
        ltile_cols: int = DEFAULT_LTILE_COLS,
        scheduler: str = "static",
        has_group_order: bool = False,
        tile_n: int = 128,
        cluster_n: int = 1,
    ):
        self._validate_variant(
            mma_sm, full_k_acc, mining, ltile_rows, ltile_cols, tile_n, cluster_n
        )
        if scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {scheduler!r}")
        if scheduler == "dynamic" and cluster_n != 1:
            # The claim is shared across the M pair only (the leader -> peer
            # tile mailbox); N peers would claim independently and diverge.
            raise ValueError("dynamic scheduling requires cluster_n=1")
        # Persistent tile schedule. ``static``: cluster ``c`` owns linear tiles
        # ``c, c + P, c + 2P, ...`` (P = clusters in the grid). ``dynamic``:
        # the same first tile, then every next tile is claimed from a global
        # counter (``sched_counter[0]``); the last cluster to finish resets
        # the counters so a launch is graph-replayable without a host memset.
        self.dynamic_sched = scheduler == "dynamic"
        # Optional group visiting order: the linear tile index runs over
        # ``group_order[0], group_order[1], ...`` instead of ``0, 1, ...``.
        self.has_group_order = has_group_order
        self.mining = mining
        self.snapshot_payload = snapshot_payload and mining
        self.ltile_rows = ltile_rows
        self.ltile_cols = ltile_cols
        self.scale_major_k = scale_major_k
        # Accumulation-order variant. ``False`` is the blockwise kernel: the
        # TMEM accumulator restarts every 128-wide K tile and the
        # accumulator-update warps fold ``acc_tile * sfa * sfb`` into a running
        # FP32 sum. ``True`` chains every K tile into one TMEM accumulator
        # (never restarted within an output tile), commits it once per tile,
        # and applies the K-block-0 scales once -- the accumulation order the
        # pearl mining lottery requires. Only valid for scales constant along K.
        self.full_k_acc = full_k_acc
        # Full-K variants hand each committed accumulator stage straight to
        # the epilogue warps (scale applied there, stage released after the
        # last TMEM load); only the blockwise kernel needs the intermediate
        # ``tmem_final_offset`` region and the accumulator-update round trip.
        self.direct_epilogue = full_k_acc
        self.acc_dtype: type[cutlass.Numeric] = cutlass.Float32
        self.use_2cta_instrs = mma_sm == 2
        self.tile_n = tile_n
        self.cluster_n = cluster_n
        # The cluster's N dimension multicasts A across CTAs that share an M
        # tile (B is multicast across the 2-CTA pair's M dimension).
        self.cluster_shape_mn = (mma_sm, cluster_n)
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (CTA_TILE_M * mma_sm, tile_n, 1)
        # TMEM: 128 x tile_n FP32 per accumulator stage per CTA out of 512
        # columns. 128-wide tiles keep three stages and leave columns 384-511
        # for the blockwise final region or the mining peel stage; 256-wide
        # tiles fill TMEM with two stages (full-K only: the direct epilogue
        # reads them in place, so no final region is needed).
        self.num_acc_stage = 512 // tile_n if tile_n > 128 else 3
        # Mining adds the BF16 peel to the accumulator on the MMA side.
        # 128-wide tiles land it in a separate TMEM peel stage the epilogue
        # adds in registers; 256-wide tiles have no spare TMEM, so the peel
        # UMMA accumulates in place into the tile's own stage once the fold
        # warps have snapshotted the pre-peel accumulator.
        self.peel_in_place = mining and tile_n > 128

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.acc_update_warp_id = (0, 1, 2, 3)
        self.epilog_warp_id = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.scale_warp_id = 10
        self.sched_warp_id = 11
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.acc_update_warp_id,
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.scale_warp_id,
                self.sched_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.acc_update_warp_id,
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.scale_warp_id,
            )
        )
        self.num_regs_uniform_warps = 64
        self.num_regs_sched_warps = 64
        self.num_regs_epilogue_warps = 168
        self.num_regs_acc_update_warps = 256

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32
            * len((self.mma_warp_id, *self.epilog_warp_id, *self.acc_update_warp_id)),
        )
        # Every TMEM user (MMA producer, acc-update readers/writers, epilogue
        # owners) joins this barrier before the allocator frees the
        # accumulator. The MMA warp arrives only after its ``producer_tail``,
        # i.e. after every acc-update TMEM read has been released; the tile
        # pipelines alone do not order the last tile's MMA/fold against the
        # epilogue reaching ``free``.
        self.tmem_free_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=32
            * len((self.mma_warp_id, *self.epilog_warp_id, *self.acc_update_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp,
        )
        # Mining: the accumulator-update warps publish staged lottery words
        # (and the column unscale slice) to each other through this barrier.
        self.acc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp * len(self.acc_update_warp_id),
        )
        # Lottery geometry (mining only). One message per acc-update thread:
        # each of the 128 accumulator rows contributes its words to its
        # lottery row's message, and thread i hashes message i. The lottery
        # tile lattice is in ``ltile_rows x ltile_cols`` units regardless of
        # the CTA tile width, so a 256-wide CTA tile simply holds twice the
        # column tiles of a 128-wide one.
        cta_m, cta_n = 128, tile_n
        self.words_per_row = LANES // ltile_rows
        self.column_tiles = cta_n // ltile_cols
        self.msgs_cta = (cta_m // ltile_rows) * self.column_tiles
        assert self.msgs_cta <= _ACC_THREADS
        # The hit publish ballots whole warps, so the hashing gate is rounded
        # up to warp granularity; sub-warp tails are excluded per lane.
        self.hash_threads = -(-self.msgs_cta // 32) * 32
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        # TMEM offset of the blockwise kernel's final accumulator region
        self.tmem_final_offset = 384

    @staticmethod
    def _validate_variant(mma_sm, full_k_acc, mining, ltile_rows, ltile_cols, tile_n, cluster_n):
        if mma_sm not in (1, 2):
            raise ValueError("mma_sm must be 1 or 2")
        if tile_n not in (128, 256):
            raise ValueError("tile_n must be 128 or 256")
        if cluster_n not in (1, 2):
            raise ValueError("cluster_n must be 1 or 2")
        # Wide tiles hold 128 x tile_n FP32 per CTA: TMEM has room for two
        # such accumulators and no separate final region, so only the
        # direct-epilogue (full-K) order fits; the blockwise order needs the
        # final region for its per-K-tile restart.
        if tile_n != 128 and not full_k_acc:
            raise ValueError("tile_n=256 requires the full-K accumulation order")
        if mining:
            GroupedGemmSm100._validate_mining_variant(
                mma_sm, full_k_acc, ltile_rows, ltile_cols, tile_n, cluster_n
            )

    @staticmethod
    def _validate_mining_variant(mma_sm, full_k_acc, ltile_rows, ltile_cols, tile_n, cluster_n):
        # The lottery's per-CTA geometry (one message per acc-update thread,
        # one accumulator row per thread) holds for the 128-row CTA tile of
        # either UMMA shape and either tile width; A multicast across an N
        # cluster is not wired into the mining pipelines.
        if cluster_n != 1:
            raise NotImplementedError("the mining variant supports cluster_n=1 only")
        if not full_k_acc:
            raise ValueError("mining requires the full-K accumulation order")
        if ltile_rows not in SUPPORTED_LTILE_ROWS:
            raise ValueError(f"unsupported lottery tile rows: {ltile_rows}")
        if ltile_cols not in SUPPORTED_LTILE_COLS[ltile_rows]:
            raise ValueError(f"unsupported lottery tile cols: {ltile_rows}x{ltile_cols}")

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs."""
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mn = (
            self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0],
            self.cta_tile_shape_mnk[1] * self.cluster_shape_mn[1],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.scale_granularity_m = 1
        self.scale_granularity_n = 128
        self.scale_granularity_k = 128
        self.scale_m_per_tile = self.cta_tile_shape_mnk[0] // self.scale_granularity_m
        self.scale_n_per_tile = self.cta_tile_shape_mnk[1] // self.scale_granularity_n
        self.scale_k_per_tile = self.cta_tile_shape_mnk[2] // self.scale_granularity_k

        if self.scale_k_per_tile != 1:
            raise ValueError("scale_k_per_tile must be 1")
        if self.scale_m_per_tile != self.cta_tile_shape_mnk[0]:
            raise ValueError("scale_m_per_tile must be cta_tile_m")
        if self.scale_n_per_tile * self.scale_granularity_n != self.cta_tile_shape_mnk[1]:
            raise ValueError("cta_tile_n must be a multiple of the N scale granularity")

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Mining replaces the scale stages with the lottery-word block, the
        # column-unscale slice and the dedicated peel operand stage.
        self.tiled_mma_peel = None
        self.pa_smem_layout_staged = None
        self.pb_smem_layout_staged = None
        self.num_peel_tx_bytes = 0
        mining_bytes = 0
        if self.mining:
            mining_bytes = self.msgs_cta * _SEXT_PAD * 4 + 128  # sExt
            mining_bytes += self.cta_tile_shape_mnk[1] * 4 + 32  # sAlB
            bf16 = cutlass.BFloat16
            # BF16 peel UMMA with the same instruction M/N (and CTA
            # group) as the FP8 mainloop, so it accumulates into the
            # identical TMEM layout.
            self.tiled_mma_peel = sm100_utils.make_trivial_tiled_mma(
                bf16,
                self.a_major_mode,
                self.b_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler[:2],
            )
            assert self.tiled_mma_peel.partition_shape_C(
                self.mma_tiler[:2]
            ) == tiled_mma.partition_shape_C(self.mma_tiler[:2]), (
                "peel UMMA must share the mainloop accumulator TMEM layout"
            )
            peel_tiler = (self.mma_tiler[0], self.mma_tiler[1], R2)
            self.pa_smem_layout_staged = sm100_utils.make_smem_layout_a(
                self.tiled_mma_peel, peel_tiler, bf16, 1
            )
            self.pb_smem_layout_staged = sm100_utils.make_smem_layout_b(
                self.tiled_mma_peel, peel_tiler, bf16, 1
            )
            peel_smem_bytes = cute.size_in_bytes(
                bf16, cute.slice_(self.pa_smem_layout_staged, (None, None, None, 0))
            ) + cute.size_in_bytes(
                bf16, cute.slice_(self.pb_smem_layout_staged, (None, None, None, 0))
            )
            # The peel loads of both CTAs of a pair complete on the
            # leader's transaction barrier, like the mainloop's.
            self.num_peel_tx_bytes = peel_smem_bytes * cute.size(tiled_mma.thr_id.shape)
            mining_bytes += 2 * 1024 + peel_smem_bytes  # sPa/sPb (+ align)

        (
            self.num_ab_stage,
            self.num_c_stage,
            self.num_scale_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sfa_dtype,
            self.sfb_dtype,
            self.scale_m_per_tile * self.scale_k_per_tile,
            self.scale_n_per_tile * self.scale_k_per_tile,
            self.num_smem_capacity,
            self.occupancy,
            mining_bytes=mining_bytes if self.mining else None,
        )

        # TMEM: ``num_acc_stage`` tile_n-column accumulator stages plus, in
        # 128-wide mining, one 128-column peel stage; 512 columns are
        # allocated either way (the allocation granularity is a power of
        # two). The wide tile's peel accumulates in place instead.
        self.num_peel_stage = 1 if (self.mining and not self.peel_in_place) else 0
        assert (self.num_acc_stage + self.num_peel_stage) * self.mma_tiler[1] <= 512

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )
        self.sfa_smem_layout_staged = None
        self.sfb_smem_layout_staged = None
        if self.mining:
            self.num_tmem_alloc_cols = 512
            return
        self.sfa_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_m_per_tile,
            ),
        )
        self.sfb_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_n, self.scale_n_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_n_per_tile,
            ),
        )

        self.num_tmem_alloc_cols = 512

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        m_indptr: cute.Tensor,
        sched_counter: cute.Tensor | None,
        group_order: cute.Tensor | None,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Launch the grouped GEMM.

        :param a: (cum_m, k) e4m3, K-major
        :param b: (num_groups * n, k) e4m3, K-major (groups stacked along rows)
        :param c: (cum_m, n) bf16, N-major
        :param sfa: (k/128, cum_m) fp32 for MN-major scales, (cum_m, k/128) for K-major
        :param sfb: (num_groups, k/128, n/128) fp32 for MN-major, (num_groups, n/128, k/128) for K-major
        :param m_indptr: (num_groups + 1,) int32 exclusive prefix sums (no alignment)
        :param sched_counter: (2,) int32 zeroed dynamic-scheduler counters
            (``scheduler="dynamic"`` only, else ``None``)
        :param group_order: (num_groups,) int32 permutation giving the group
            visiting order (``has_group_order`` only, else ``None``)

        Contract (checked by the host wrapper ``grouped_fp8_gemm``, which
        raises ``TypeError`` for non-tensors and ``ValueError`` otherwise
        before anything is launched; this method itself performs no
        validation): every operand is contiguous, on one CUDA device and
        16-byte aligned (``m_indptr`` / ``group_order`` 4-byte); ``k`` is a
        multiple of 16 and ``n`` of 8, with 128-multiples required for the
        scale layouts; ``sfa`` / ``sfb`` are the 1x128 / 128x128 unit scale
        planes in the compiled major mode. ``m_indptr`` should be
        non-decreasing exclusive prefix sums ending at ``cum_m`` and
        ``group_order`` a permutation, but their *values* are never read on
        the host (no device sync): the kernel clamps every group's rows into
        ``[0, cum_m)`` and every ``group_order`` entry into
        ``[0, num_groups)`` (``_clamp_rows`` / ``_clamp_group``), so a
        malformed table cannot address memory outside the operands; what it
        computes is unspecified. All operands, including ``m_indptr``, are
        read asynchronously on ``stream`` -- the caller must not modify them
        until the launch completes (stream ordering).
        """
        assert not self.mining, "mining variants launch through ``mine``"
        self._launch(
            a,
            b,
            c,
            sfa,
            sfb,
            m_indptr,
            c.shape[1],
            max_active_clusters,
            stream,
            sched_counter=sched_counter,
            group_order=group_order,
        )

    @cute.jit
    def mine(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        m_indptr: cute.Tensor,
        m_valid: cute.Tensor,
        sched_counter: cute.Tensor | None,
        group_order: cute.Tensor | None,
        a_peel: cute.Tensor,
        b_peel: cute.Tensor,
        alpha_a: cute.Tensor,
        inv_alpha_b: cute.Tensor,
        pow_key: cute.Tensor,
        threshold: cute.Tensor,
        record: cute.Tensor,
        lock: cute.Tensor,
        hash_b: cute.Tensor,
        codes_src: cute.Tensor | None,
        scales_src: cute.Tensor | None,
        codes_dst: cute.Tensor | None,
        scales_dst: cute.Tensor | None,
        n: Int32,
        layer_id: Int32,
        record_hits: Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Launch the MoE mining variant.

        :param a: (cum_m, k) e4m3 noisy A' rows, permuted so every group's rows
            are contiguous and in ascending token order
        :param b: (num_groups * n, k) e4m3 noisy B' (experts stacked along rows)
        :param c: (cum_m, n) bf16 unscaled peeled output
        :param m_indptr: (num_groups + 1,) int32 exclusive prefix sums (no alignment)
        :param m_valid: (num_groups,) int32 real token count per group; lottery
            tiles reaching past it (including the trailing partial tile of an
            unaligned group) are hashed but never published
        :param a_peel: (cum_m, 2R) bf16, :param b_peel: (num_groups * n, 2R) bf16
        :param alpha_a: (cum_m,) bf16 row scale, :param inv_alpha_b: (num_groups * n,) f32
        :param pow_key: (8,) u32 keyed-BLAKE3 key (= cA), :param threshold: (8,) u32 LE
        :param record: (64,) u32 mapped pinned hit record, :param lock: (1,) i32 latch
        :param hash_b: (8,) u32 commitment hash B stamped into hits
        :param codes_src: flat u32 view of the permuted (cum_m, k) int8 codes plane,
            :param scales_src: of the permuted (cum_m, k/8) bf16 scales plane; the
            winning group's rows are snapshotted into ``codes_dst`` / ``scales_dst``
            when they fit (decided at run time), otherwise the record is payload-less
        :param sched_counter: / :param group_order: as in ``__call__``
        """
        assert self.mining, "``mine`` is the mining variant's entry point"
        payload_tensors = (codes_src, scales_src, codes_dst, scales_dst)
        assert all((t is not None) == self.snapshot_payload for t in payload_tensors), (
            "payload tensor presence must match the compiled snapshot variant"
        )
        self._launch(
            a,
            b,
            c,
            None,
            None,
            m_indptr,
            n,
            max_active_clusters,
            stream,
            sched_counter=sched_counter,
            group_order=group_order,
            m_valid=m_valid,
            a_peel=a_peel,
            b_peel=b_peel,
            alpha_a=alpha_a,
            inv_alpha_b=inv_alpha_b,
            pow_key=pow_key,
            threshold=threshold,
            record=record,
            lock=lock,
            hash_b=hash_b,
            codes_src=codes_src,
            scales_src=scales_src,
            codes_dst=codes_dst,
            scales_dst=scales_dst,
            layer_id=layer_id,
            record_hits=record_hits,
        )

    def _launch(  # noqa: C901 (trace-time setup shared by both entry points)
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor | None,
        sfb: cute.Tensor | None,
        m_indptr: cute.Tensor,
        n,
        max_active_clusters,
        stream: cuda.CUstream,
        *,
        sched_counter=None,
        group_order=None,
        m_valid=None,
        a_peel=None,
        b_peel=None,
        alpha_a=None,
        inv_alpha_b=None,
        pow_key=None,
        threshold=None,
        record=None,
        lock=None,
        hash_b=None,
        codes_src=None,
        scales_src=None,
        codes_dst=None,
        scales_dst=None,
        layer_id=None,
        record_hits=None,
    ):
        k = a.shape[1]
        # Launch-wide TMA views. A and C get a ragged row mode: each group is a
        # coordinate offset and rows past its end fall outside the descriptor
        # (A zero-fills, C stores clip). B is addressed by its group row offset.
        a3 = self._ragged_rows_view(a)
        b3 = cute.make_tensor(
            b.iterator,
            cute.make_layout((b.shape[0], b.shape[1], 1), stride=(b.stride[0], 1, 0)),
        )
        c3 = self._ragged_rows_view(c)
        self.c_dtype: type[cutlass.Numeric] = c.element_type
        # Coordinate template of one MMA tile: the accumulator / epilogue
        # register partitions are shaped from it (never dereferenced).
        c_tile3 = cute.make_identity_tensor((self.mma_tiler[0], self.mma_tiler[1], 1))

        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        self.sfa_dtype: type[cutlass.Numeric] = sfa.element_type if sfa is not None else Float32
        self.sfb_dtype: type[cutlass.Numeric] = sfb.element_type if sfb is not None else Float32
        self.a_major_mode = cute.nvgpu.OperandMajorMode.K
        self.b_major_mode = cute.nvgpu.OperandMajorMode.K
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        assert (sched_counter is not None) == self.dynamic_sched, (
            "sched_counter presence must match the compiled scheduler"
        )
        assert (group_order is not None) == self.has_group_order, (
            "group_order presence must match the compiled variant"
        )

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = self._get_tma_atom_kind(atom_thr_size, self.is_a_mcast)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a3,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = self._get_tma_atom_kind(atom_thr_size, self.is_b_mcast)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b3,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        c_cta_v_layout = cute.composition(cute.make_identity_layout(c3.shape), self.epi_tile)
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c3,
            epi_smem_layout,
            c_cta_v_layout,
        )

        # Mining: BF16 peel operands through global descriptors.
        # Groups use row-offset coordinates (no descriptor rewrite).
        # The bounded C descriptor suppresses stores past the group's rows.
        # Wide B/B-peel loads may overhang its columns; stores and lottery
        # publication exclude those columns.
        tma_atom_pa, tma_tensor_pa, tma_atom_pb, tma_tensor_pb = (None,) * 4
        if self.mining:
            pa3 = cute.make_tensor(
                a_peel.iterator,
                cute.make_layout(
                    (a_peel.shape[0], a_peel.shape[1], 1), stride=(a_peel.stride[0], 1, 0)
                ),
            )
            pb3 = cute.make_tensor(
                b_peel.iterator,
                cute.make_layout(
                    (b_peel.shape[0], b_peel.shape[1], 1), stride=(b_peel.stride[0], 1, 0)
                ),
            )
            peel_tiler = (self.mma_tiler[0], self.mma_tiler[1], R2)
            tma_atom_pa, tma_tensor_pa = cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
                pa3,
                cute.slice_(self.pa_smem_layout_staged, (None, None, None, 0)),
                peel_tiler,
                self.tiled_mma_peel,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_pb, tma_tensor_pb = cute.nvgpu.make_tiled_tma_atom_B(
                cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
                pb3,
                cute.slice_(self.pb_smem_layout_staged, (None, None, None, 0)),
                peel_tiler,
                self.tiled_mma_peel,
                self.cluster_layout_vmnk.shape,
            )

        # Grid: persistent, one CTA per SM (as the CUTLASS group scheduler
        # does when problem shapes live on device), capped by an upper bound
        # on the tile count that needs no device sync.
        cum_m = a.shape[0]
        num_groups = m_indptr.shape[0] - 1
        cluster_tile_m, cluster_tile_n = self.cluster_tile_shape_mn
        n_tiles = (n + cluster_tile_n - 1) // cluster_tile_n
        m_tiles_bound = (cum_m + cluster_tile_m - 1) // cluster_tile_m + num_groups
        tiles_bound = m_tiles_bound * n_tiles
        num_clusters = cutlass.min(Int32(max_active_clusters), tiles_bound)
        grid = (num_clusters * self.cluster_shape_mn[0], self.cluster_shape_mn[1], 1)

        self.buffer_align_bytes = 1024

        self.shared_storage = self._make_shared_storage()

        self.kernel(
            tiled_mma,
            self.tiled_mma_peel,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_pa,
            tma_tensor_pa,
            tma_atom_pb,
            tma_tensor_pb,
            c_tile3,
            k,
            n,
            Int32(cum_m),
            sfa,
            sfb,
            m_indptr,
            m_valid,
            sched_counter,
            group_order,
            alpha_a,
            inv_alpha_b,
            pow_key,
            threshold,
            record,
            lock,
            hash_b,
            codes_src,
            scales_src,
            codes_dst,
            scales_dst,
            layer_id,
            record_hits,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.pa_smem_layout_staged,
            self.pb_smem_layout_staged,
            self.epi_tile,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    def _make_shared_storage(self):
        """The CTA's shared-memory struct for the compiled variant."""
        align = self.buffer_align_bytes
        a_size = cute.cosize(self.a_smem_layout_staged.outer)
        b_size = cute.cosize(self.b_smem_layout_staged.outer)
        num_ab, num_acc, num_tile = self.num_ab_stage, self.num_acc_stage, self.num_tile_stage
        # 2-CTA dynamic schedule: the leader relays each fetched linear tile
        # index to its peer through this mailbox (data + full/empty barriers).
        has_mailbox = self.dynamic_sched and self.use_2cta_instrs

        if not self.mining:
            c_smem_size = cute.cosize(self.c_smem_layout_staged.outer)
            num_scale = self.num_scale_stage

            @cute.struct
            class SharedStorage:
                # (cta_tile_m, cta_tile_n, group, valid)
                sInfo: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4 * num_tile], 1]
                if has_mailbox:
                    sFetch: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, num_tile], 16]
                    fetch_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_tile * 2]
                ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab * 2]
                scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_scale * 2]
                acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc * 2]
                tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_tile * 2]
                epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1 * 2]
                tmem_dealloc_mbar_ptr: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                # (EPI_TILE_M, EPI_TILE_N, STAGE)
                sC: cute.struct.Align[cute.struct.MemRange[self.c_dtype, c_smem_size], align]
                # (MMA, MMA_M, MMA_K, STAGE)
                sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, a_size], align]
                # (MMA, MMA_N, MMA_K, STAGE)
                sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, b_size], align]
                sSFA: cute.struct.Align[
                    cute.struct.MemRange[self.sfa_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                    align,
                ]
                sSFB: cute.struct.Align[
                    cute.struct.MemRange[self.sfb_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                    align,
                ]

            return SharedStorage

        sext_size = self.msgs_cta * _SEXT_PAD
        c_smem_size = cute.cosize(self.c_smem_layout_staged.outer)
        pa_size = cute.cosize(self.pa_smem_layout_staged.outer)
        pb_size = cute.cosize(self.pb_smem_layout_staged.outer)
        salb_size = self.cta_tile_shape_mnk[1]
        # Peeled-accumulator handshake slots: one per accumulator stage
        # when the peel accumulates in place, one for the separate stage.
        num_peel_acc = num_acc if self.peel_in_place else 1

        @cute.struct
        class SharedStorage:
            sInfo: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4 * num_tile], 1]
            if has_mailbox:
                sFetch: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, num_tile], 16]
                fetch_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_tile * 2]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_tile * 2]
            # the peeled accumulator (UMMA commit -> epilogue release) and
            # the dedicated peel operand stage (TMA -> UMMA)
            peel_acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_peel_acc * 2]
            peel_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # staged lottery words, one padded 16-word row per message
            sExt: cute.struct.Align[cute.struct.MemRange[Uint32, sext_size], 16]
            # this tile's column unscale slice
            sAlB: cute.struct.Align[cute.struct.MemRange[Float32, salb_size], 16]
            sC: cute.struct.Align[cute.struct.MemRange[self.c_dtype, c_smem_size], align]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, a_size], align]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, b_size], align]
            # (MMA, MMA_M, PEEL_K, 1) / (MMA, MMA_N, PEEL_K, 1) BF16 peel operands
            sPa: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, pa_size], align]
            sPb: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, pb_size], align]

        return SharedStorage

    @cute.jit
    def _clamp_rows(self, row0: Int32, row1: Int32, cum_m: Int32):
        """``[row0, row1)`` clamped into ``[0, cum_m]`` with ``row1 >= row0``.

        The only in-kernel validation of ``m_indptr``: the host never reads
        it (no device sync), so a malformed table -- an offset outside A, a
        decreasing pair -- must still keep every group inside the operands.
        Clamping gives that (rows outside ``[0, cum_m)`` are never addressed
        and a decreasing pair is an empty group); what such a table computes
        is unspecified, only where it writes is bounded.
        """
        row0 = cutlass.min(cutlass.max(row0, Int32(0)), cum_m)
        row1 = cutlass.min(cutlass.max(row1, row0), cum_m)
        return row0, row1

    @cute.jit
    def _clamp_group(self, group_idx: Int32, num_groups: Int32) -> Int32:
        """A ``group_order`` entry clamped into ``[0, num_groups)`` (the host
        never reads the table; an entry outside it must not index past
        ``m_indptr``)."""
        return cutlass.min(cutlass.max(group_idx, Int32(0)), num_groups - 1)

    @cute.jit
    def _group_rows(self, m_indptr: cute.Tensor, group_idx: Int32, cum_m: Int32):
        """(row offset, row count) of ``group_idx``, clamped into ``[0, cum_m)``."""
        m_off, m_end = self._clamp_rows(m_indptr[group_idx], m_indptr[group_idx + 1], cum_m)
        return m_off, m_end - m_off

    @staticmethod
    def _ragged_rows_view(t: cute.Tensor) -> cute.Tensor:
        """Rank-4 ragged-row TMA view of a row-major ``(rows, cols)`` tensor.

        Shape ``(_RAGGED_ROWS, cols, _RAGGED_EXTRA, _RAGGED_EXTRA)`` with
        strides ``(row, 1, 2**34 - row, row)``; see ``_RAGGED_ROWS``. The
        descriptor covers ``cols`` exactly, so column overhang zero-fills /
        clips as with a plain descriptor.
        """
        row = t.stride[0]
        return cute.make_tensor(
            t.iterator,
            cute.make_layout(
                (_RAGGED_ROWS, t.shape[1], _RAGGED_EXTRA, _RAGGED_EXTRA),
                stride=(row, 1, _RAGGED_WRAP - row, row),
            ),
        )

    @staticmethod
    def _ragged_group(t4: cute.Tensor, m_off: Int32, m_cnt: Int32) -> cute.Tensor:
        """Rank-2 ``(rows, cols)`` coordinate view of one group of a ragged view.

        Row ``r`` of the result addresses global row ``m_off + r`` for
        ``r < m_cnt`` and is out of the TMA descriptor's bounds otherwise.
        """
        return cute.domain_offset(
            (_RAGGED_ROWS - m_cnt, 0), t4[(None, None, _RAGGED_ROWS, m_off + m_cnt)]
        )

    # ------------------------------------------------------------------
    # Tile scheduler helpers (scheduler warp only)
    # ------------------------------------------------------------------

    @cute.jit
    def _warp_prefix_sum(self, val: Int32, lane_idx: Int32) -> Int32:
        """Inclusive prefix sum of ``val`` across the warp."""
        for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
            offset = 1 << i
            partial = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
            if lane_idx >= offset:
                val = val + partial
        return val

    @cute.jit
    def _locate_group(  # noqa: C901 (one warp-cooperative scan; splitting it hides the data flow)
        self,
        linear_idx: Int32,
        pos: Int32,
        tiles_before: Int32,
        group_end: Int32,
        m_tiles_g: Int32,
        m_indptr: cute.Tensor,
        mOrder: cute.Tensor | None,
        num_groups: Int32,
        cum_m: Int32,
        n_tiles: Int32,
        lane_idx: Int32,
    ):
        """Resolve the group (position in visiting order) owning ``linear_idx``.

        ``(pos, tiles_before, group_end, m_tiles_g)`` is the cached window of
        the previously resolved group: it owns linear tiles
        ``[tiles_before, group_end)`` and has ``m_tiles_g`` M tiles. A cluster's
        linear indices never decrease, so an index past the window scans
        forward from ``pos``, ``_SCAN_GROUPS_PER_ITER`` groups per iteration
        (lane ``i`` owns group ``pos + i``; the last lane supplies the window's
        end row), with a warp prefix sum of the per-group tile counts and a
        ballot to pick the group. Returns the updated window; ``pos ==
        num_groups`` means the index is past the last tile.
        """
        cluster_tile_m = self.cluster_tile_shape_mn[0]
        scan_groups = _SCAN_GROUPS_PER_ITER
        if linear_idx >= group_end:
            win_start = Int32(pos)
            win_end = Int32(tiles_before)
            win_tiles = Int32(0)
            cum_tiles = Int32(0)
            lane_m_tiles = Int32(0)
            while win_end <= linear_idx:
                g = win_start + lane_idx
                row0 = Int32(0)
                row1 = Int32(0)
                if cutlass.const_expr(mOrder is None):
                    if g <= num_groups:
                        row0 = cutlass.min(cutlass.max(m_indptr[g], Int32(0)), cum_m)
                    row1 = cute.arch.shuffle_sync_down(row0, offset=1)
                    row1 = cutlass.max(row1, row0)
                else:
                    if g < num_groups:
                        src = self._clamp_group(mOrder[g], num_groups)
                        row0, row1 = self._clamp_rows(m_indptr[src], m_indptr[src + 1], cum_m)
                lane_m_tiles = Int32(0)
                if Boolean(g < num_groups) & Boolean(lane_idx < scan_groups):
                    lane_m_tiles = (row1 - row0 + cluster_tile_m - 1) // cluster_tile_m
                cum_tiles = self._warp_prefix_sum(lane_m_tiles * n_tiles, lane_idx)
                win_tiles = cute.arch.shuffle_sync(cum_tiles, cute.arch.WARP_SIZE - 1)
                win_end = win_end + win_tiles
                if win_end <= linear_idx:
                    win_start = win_start + scan_groups
                    if win_start >= num_groups:
                        win_start = Int32(num_groups)
                        win_end = linear_idx + 1
            pos = Int32(win_start)
            if win_start < num_groups:
                win_begin = win_end - win_tiles
                # Groups of the window whose tiles all precede linear_idx.
                in_win = cute.arch.popc(
                    cute.arch.vote_ballot_sync(win_begin + cum_tiles <= linear_idx)
                )
                pos = win_start + in_win
                prev_tiles = Int32(0)
                if in_win > 0:
                    prev_tiles = cute.arch.shuffle_sync(cum_tiles, in_win - 1)
                m_tiles_g = cute.arch.shuffle_sync(lane_m_tiles, in_win)
                tiles_before = win_begin + prev_tiles
                group_end = tiles_before + m_tiles_g * n_tiles
        return pos, tiles_before, group_end, m_tiles_g

    @cute.jit
    def _fetch_next_tile(self, mSched: cute.Tensor, num_clusters: Int32, lane_idx: Int32) -> Int32:
        """Claim the next linear tile index from the global counter (whole warp)."""
        next_idx = Int32(0)
        if lane_idx == 0:
            next_idx = num_clusters + cute.arch.atomic_add(
                mSched.iterator, Int32(1), sem="relaxed", scope="gpu"
            )
        return cute.arch.shuffle_sync(next_idx, 0)

    @cute.jit
    def _retire_cluster(self, mSched: cute.Tensor, num_clusters: Int32, lane_idx: Int32):
        """Count this cluster as done; the last one re-zeroes both counters.

        Every cluster's final fetch precedes its retire (release), so the last
        retiree (acquire) resets after all fetches of this launch, leaving the
        counters zero for the next launch without a host memset.
        """
        if lane_idx == 0:
            done = cute.arch.atomic_add(mSched.iterator + 1, Int32(1), sem="acq_rel", scope="gpu")
            if done == num_clusters - 1:
                mSched[0] = Int32(0)
                mSched[1] = Int32(0)

    @cute.jit
    def _make_group_sfa(
        self, sfa: cute.Tensor, m_off: Int32, m_cnt: Int32, k_blocks: Int32
    ) -> cute.Tensor:
        """SFA of one group viewed as ((1, m), (128, k/128), 1)."""
        if cutlass.const_expr(self.scale_major_k):
            # (cum_m, k/128) row-major
            ptr = sfa.iterator + Int64(m_off) * Int64(sfa.stride[0])
            stride_m = sfa.stride[0]
            stride_k = 1
        else:
            # (k/128, cum_m) row-major
            ptr = sfa.iterator + Int64(m_off)
            stride_m = 1
            stride_k = sfa.stride[0]
        return cute.make_tensor(
            ptr,
            cute.make_layout(
                (
                    (self.scale_granularity_m, m_cnt),
                    (self.scale_granularity_k, k_blocks),
                    1,
                ),
                stride=((0, stride_m), (0, stride_k), 0),
            ),
        )

    @cute.jit
    def _make_group_sfb(
        self, sfb: cute.Tensor, group_idx: Int32, n_blocks: Int32, k_blocks: Int32
    ) -> cute.Tensor:
        """SFB of one group viewed as ((128, n/128), (128, k/128), 1)."""
        ptr = sfb.iterator + Int64(group_idx) * Int64(sfb.stride[0])
        if cutlass.const_expr(self.scale_major_k):
            # (num_groups, n/128, k/128)
            stride_n = sfb.stride[1]
            stride_k = 1
        else:
            # (num_groups, k/128, n/128)
            stride_n = 1
            stride_k = sfb.stride[1]
        return cute.make_tensor(
            ptr,
            cute.make_layout(
                (
                    (self.scale_granularity_n, n_blocks),
                    (self.scale_granularity_k, k_blocks),
                    1,
                ),
                stride=((0, stride_n), (0, stride_k), 0),
            ),
        )

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_peel: cute.TiledMma | None,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tma_atom_pa: cute.CopyAtom | None,
        mPa_mkl: cute.Tensor | None,
        tma_atom_pb: cute.CopyAtom | None,
        mPb_nkl: cute.Tensor | None,
        cTile_mnl: cute.Tensor,
        k: Int32,
        n: Int32,
        cum_m: Int32,
        mSFA: cute.Tensor | None,
        mSFB: cute.Tensor | None,
        m_indptr: cute.Tensor,
        m_valid: cute.Tensor | None,
        mSched: cute.Tensor | None,
        mOrder: cute.Tensor | None,
        mAlA: cute.Tensor | None,
        mAlB: cute.Tensor | None,
        mKey: cute.Tensor | None,
        mThr: cute.Tensor | None,
        mRecord: cute.Tensor | None,
        mLock: cute.Tensor | None,
        mHashB: cute.Tensor | None,
        mCodesSrc: cute.Tensor | None,
        mScalesSrc: cute.Tensor | None,
        mCodesDst: cute.Tensor | None,
        mScalesDst: cute.Tensor | None,
        layer_id: Int32 | None,
        record_hits: Int32 | None,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.Layout | cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout | None,
        sfb_smem_layout_staged: cute.Layout | None,
        pa_smem_layout_staged: cute.ComposedLayout | None,
        pb_smem_layout_staged: cute.ComposedLayout | None,
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()
        mining = cutlass.const_expr(self.mining)
        peel_in_place = cutlass.const_expr(self.peel_in_place)
        if cutlass.const_expr(mining):
            # The extents ``_HitPublishMixin`` stamps into hit records.
            self.problem_n, self.problem_k = n, k

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(mining):
                cpasync.prefetch_descriptor(tma_atom_pa)
                cpasync.prefetch_descriptor(tma_atom_pb)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Problem dimensions shared by all groups.
        num_groups = m_indptr.shape[0] - 1
        k_blocks = k // self.scale_granularity_k
        n_blocks = n // self.scale_granularity_n
        k_tile_cnt = cute.ceil_div(k, self.cta_tile_shape_mnk[2])

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        scale_pipeline = None
        if cutlass.const_expr(not mining):
            scale_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * 1,
            )
            scale_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            )
            scale_pipeline = pipeline.PipelineCpAsync.create(
                barrier_storage=storage.scale_mbar_ptr.data_ptr(),
                num_stages=self.num_scale_stage,
                producer_group=scale_pipeline_producer_group,
                consumer_group=scale_pipeline_consumer_group,
                defer_sync=True,
            )

        # Accumulator stage consumers: one elected release per consuming warp
        # (per CTA of the pair for 2-CTA UMMA). Blockwise: the accumulator-
        # update warps; full-K: the epilogue warps; mining: the fold warps,
        # plus the epilogue warps when they read the same stage (separate
        # peel stage). With the peel accumulated in place the epilogue reads
        # the stage through ``peel_acc_pipeline`` instead, and the fold's
        # release is what admits the peel UMMA.
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_warps = 0
        if cutlass.const_expr(mining):
            num_acc_consumer_warps += len(self.acc_update_warp_id)
        if cutlass.const_expr(not peel_in_place):
            num_acc_consumer_warps += len(self.epilog_warp_id)
        num_acc_consumer_threads = num_acc_consumer_warps * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Mining: the peeled accumulator the epilogue reads is
        # committed by the MMA warp's peel UMMA and released by one elected
        # thread per epilogue warp (per CTA of the pair). It lives in its own
        # single TMEM stage (128-wide tiles) or in the accumulator stage
        # itself (in place, one slot per accumulator stage). The peel
        # operands ride a dedicated one-stage TMA -> UMMA pipeline.
        peel_acc_pipeline, peel_pipeline = None, None
        if cutlass.const_expr(mining):
            peel_acc_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.peel_acc_mbar_ptr.data_ptr(),
                num_stages=self.num_acc_stage if peel_in_place else self.num_peel_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1),
                ),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            peel_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.peel_mbar_ptr.data_ptr(),
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=self.num_peel_tx_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

        # Blockwise only: the accumulator-update warps hand the scaled tile
        # to the epilogue through the final TMEM region.
        epi_pipeline = None
        if cutlass.const_expr(not self.direct_epilogue):
            epi_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.acc_update_warp_id),
            )
            epi_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            )
            epi_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.epi_mbar_ptr.data_ptr(),
                num_stages=1,
                producer_group=epi_pipeline_producer_group,
                consumer_group=epi_pipeline_consumer_group,
                defer_sync=True,
            )

        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        if cutlass.const_expr(self.dynamic_sched and use_2cta_instrs):  # noqa: SIM102 (trace-time guard over a runtime predicate)
            # Leader -> peer tile-index mailbox: full barriers (one remote
            # arrive + 4 tx bytes) live in the peer, empty barriers (one
            # remote arrive per read) in the leader; both CTAs init both.
            if warp_idx == 0:
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.num_tile_stage):
                        cute.arch.mbarrier_init(storage.fetch_mbar_ptr.data_ptr() + stage, 1)
                        cute.arch.mbarrier_init(
                            storage.fetch_mbar_ptr.data_ptr() + self.num_tile_stage + stage, 1
                        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C/Scale
        #
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA, sSFB = None, None
        if cutlass.const_expr(not mining):
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sExt, sAlB, sPa, sPb = None, None, None, None
        if cutlass.const_expr(mining):
            # Padded stride: compress reads sExt[msg, c] bank-conflict-free.
            sExt = storage.sExt.get_tensor(
                cute.make_layout((self.msgs_cta, LANES), stride=(_SEXT_PAD, 1))
            )
        if cutlass.const_expr(mining):
            sAlB = storage.sAlB.get_tensor(cute.make_layout(self.cta_tile_shape_mnk[1]))
            sPa = storage.sPa.get_tensor(
                pa_smem_layout_staged.outer, swizzle=pa_smem_layout_staged.inner
            )
            sPb = storage.sPb.get_tensor(
                pb_smem_layout_staged.outer, swizzle=pb_smem_layout_staged.inner
            )
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Partition the one-tile coordinate template for TiledMMA_C: the
        # accumulator-side register / TMEM partitions take their shapes from it.
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (bM, bN, 1, 1, 1)
        gC_tile = cute.local_tile(
            cTile_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        tCgC = thr_mma.partition_C(gC_tile)

        # CTA layouts of the A / B TMA multicast groups (per-tile partitions
        # are built in the TMA warp from the group's coordinate view).
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)

        # scale viewed as C tensor
        sSFA_view_as_C, sSFB_view_as_C = None, None
        if cutlass.const_expr(not mining):
            sSFA_view_as_C_layout = cute.make_layout(
                (
                    (self.scale_granularity_m, self.scale_m_per_tile),
                    self.cta_tile_shape_mnk[1],
                    self.num_scale_stage,
                ),
                stride=((0, 1), 0, self.scale_m_per_tile),
            )
            sSFB_view_as_C_layout = cute.make_layout(
                (
                    self.cta_tile_shape_mnk[0],
                    (self.scale_granularity_n, self.scale_n_per_tile),
                    self.num_scale_stage,
                ),
                stride=(0, (0, 1), self.scale_n_per_tile),
            )
            sSFA_view_as_C = cute.make_tensor(sSFA.iterator, sSFA_view_as_C_layout)
            sSFB_view_as_C = cute.make_tensor(sSFB.iterator, sSFB_view_as_C_layout)

        #
        # Tiled copies for scaleA/scaleB (one 32-bit cp.async per lane per element)
        #
        tiled_copy_sfa, tiled_copy_sfb, tAsSFA, tBsSFB = None, None, None, None
        if cutlass.const_expr(not mining):
            atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mSFA.element_type,
                num_bits_per_copy=mSFA.element_type.width,
            )
            tiled_copy_sfa = cute.make_tiled_copy_tv(
                atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
            )
            tiled_copy_sfb = cute.make_tiled_copy_tv(
                atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
            )
            thr_copy_sfa = tiled_copy_sfa.get_slice(lane_idx)
            thr_copy_sfb = tiled_copy_sfb.get_slice(lane_idx)
            tAsSFA = thr_copy_sfa.partition_D(sSFA)
            tBsSFB = thr_copy_sfb.partition_D(sSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # Accumulator stages, followed (mining) by the peel stage
        # at index ``num_acc_stage``: the BF16 peel UMMA lands in its own
        # TMEM region so it never waits for the fold's pre-peel read.
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage + self.num_peel_stage)
        )
        # Peel operands: smem-descriptor fragments and the TMA partitions of
        # the (row-offset) global peel tensors, tile coordinates per launch.
        tCrPa, tCrPb, thr_mma_peel = None, None, None
        if cutlass.const_expr(mining):
            thr_mma_peel = tiled_mma_peel.get_slice(mma_tile_coord_v)
            tCrPa = tiled_mma_peel.make_fragment_A(sPa)
            tCrPb = tiled_mma_peel.make_fragment_B(sPb)

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Specialized Schedule warp
        #
        if warp_idx == self.sched_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_sched_warps)

            # Persistent group schedule over the linear cluster-tile index
            # (groups in visiting order, N tiles outer, M tiles fastest).
            # The index -> (group, m tile, n tile) lookup is a warp-cooperative
            # scan of ``m_indptr`` with a cached current-group window, so a
            # tile in the same group as the previous one skips the scan.
            cluster_shape_m = self.cluster_shape_mn[0]
            cluster_shape_n = self.cluster_shape_mn[1]
            cluster_tile_n = self.cluster_tile_shape_mn[1]
            num_clusters = grid_dim[0] // cluster_shape_m
            cluster_id = bidx // cluster_shape_m
            cta_m_in_cluster = bidx % cluster_shape_m
            cta_n_in_cluster = bidy
            n_tiles = (n + cluster_tile_n - 1) // cluster_tile_n
            has_mailbox = cutlass.const_expr(self.dynamic_sched and use_2cta_instrs)

            linear_idx = Int32(cluster_id)
            # Cached window: group position ``pos`` owns linear tiles
            # [tiles_before, group_end); zero-width so the first tile scans.
            pos = Int32(0)
            tiles_before = Int32(0)
            group_end = Int32(0)
            m_tiles_g = Int32(1)

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )
            fetch_full_mbar, fetch_empty_mbar, sFetch = None, None, None
            fetch_producer_state, fetch_consumer_state = None, None
            if cutlass.const_expr(has_mailbox):
                fetch_full_mbar = storage.fetch_mbar_ptr.data_ptr()
                fetch_empty_mbar = fetch_full_mbar + self.num_tile_stage
                sFetch = storage.sFetch.get_tensor(cute.make_layout(self.num_tile_stage))
                fetch_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_tile_stage
                )
                fetch_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

            keep_going = Boolean(True)
            while keep_going:
                pos, tiles_before, group_end, m_tiles_g = self._locate_group(
                    linear_idx,
                    pos,
                    tiles_before,
                    group_end,
                    m_tiles_g,
                    m_indptr,
                    mOrder,
                    num_groups,
                    cum_m,
                    n_tiles,
                    lane_idx,
                )

                is_valid = pos < num_groups
                cta_tile_m = Int32(0)
                cta_tile_n = Int32(0)
                group_idx = Int32(pos)
                if is_valid:
                    local_idx = linear_idx - tiles_before
                    m_tile = local_idx % m_tiles_g
                    n_tile = local_idx // m_tiles_g
                    cta_tile_m = m_tile * cluster_shape_m + cta_m_in_cluster
                    cta_tile_n = n_tile * cluster_shape_n + cta_n_in_cluster
                    if cutlass.const_expr(mOrder is not None):
                        group_idx = self._clamp_group(mOrder[pos], num_groups)

                tile_info_pipeline.producer_acquire(tile_info_producer_state)

                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = cta_tile_m
                    sInfo[(1, tile_info_producer_state.index)] = cta_tile_n
                    sInfo[(2, tile_info_producer_state.index)] = group_idx
                    sInfo[(3, tile_info_producer_state.index)] = Int32(is_valid)

                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()

                if cutlass.const_expr(not self.dynamic_sched):
                    linear_idx = linear_idx + num_clusters
                else:
                    if is_valid:
                        if cutlass.const_expr(not has_mailbox):
                            linear_idx = self._fetch_next_tile(mSched, num_clusters, lane_idx)
                        else:
                            if is_leader_cta:
                                linear_idx = self._fetch_next_tile(mSched, num_clusters, lane_idx)
                                # Relay to the peer: wait for its previous read
                                # of this slot, then st.async the index into its
                                # mailbox against its full barrier.
                                cute.arch.mbarrier_wait(
                                    fetch_empty_mbar + fetch_producer_state.index,
                                    fetch_producer_state.phase,
                                )
                                if lane_idx == 0:
                                    cute.arch.mbarrier_arrive_and_expect_tx(
                                        fetch_full_mbar + fetch_producer_state.index,
                                        4,
                                        Int32(1),
                                    )
                                    store_shared_remote(
                                        linear_idx,
                                        sFetch.iterator + fetch_producer_state.index,
                                        fetch_full_mbar + fetch_producer_state.index,
                                        Int32(1),
                                    )
                                fetch_producer_state.advance()
                            else:
                                cute.arch.mbarrier_wait(
                                    fetch_full_mbar + fetch_consumer_state.index,
                                    fetch_consumer_state.phase,
                                )
                                linear_idx = sFetch[fetch_consumer_state.index]
                                cute.arch.sync_warp()
                                if lane_idx == 0:
                                    cute.arch.mbarrier_arrive(
                                        fetch_empty_mbar + fetch_consumer_state.index, Int32(0)
                                    )
                                fetch_consumer_state.advance()
                keep_going = is_valid

            tile_info_pipeline.producer_tail(tile_info_producer_state)
            if cutlass.const_expr(self.dynamic_sched):
                if cutlass.const_expr(has_mailbox):  # noqa: SIM102 (trace-time guard over a runtime predicate)
                    if is_leader_cta:
                        # Drain: every relayed index has been read by the peer
                        # before this CTA (and its mailbox barriers) retire.
                        for _ in cutlass.range_constexpr(self.num_tile_stage):
                            cute.arch.mbarrier_wait(
                                fetch_empty_mbar + fetch_producer_state.index,
                                fetch_producer_state.phase,
                            )
                            fetch_producer_state.advance()
                if is_leader_cta:
                    self._retire_cluster(mSched, num_clusters, lane_idx)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            is_valid_tile = tile_info[3] == 1

            peel_producer_state = None
            if cutlass.const_expr(mining):
                peel_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )

            while is_valid_tile:
                cur_group_idx = tile_info[2]
                m_off, m_cnt = self._group_rows(m_indptr, cur_group_idx, cum_m)
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    0,
                )

                # This group's A rows as a ragged coordinate view (rows past
                # ``m_cnt`` zero-fill) and its B rows as a plain row offset,
                # both through the launch-wide descriptors. n % 128 == 0 keeps
                # 128-wide B tiles inside the group; a 256-wide tile may
                # overhang into the next group's rows (or zero-fill), and the
                # epilogue never stores those columns.
                mA_g = self._ragged_group(mA_mkl, m_off, m_cnt)
                # (bM, bK, loopK)
                gA_mk = cute.local_tile(
                    mA_g,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (mma_tile_coord_mnl[0], None),
                )
                mB_g = cute.domain_offset((cur_group_idx * n, 0, 0), mB_nkl)
                # (bN, bK, loopK)
                gB_nk = cute.local_tile(
                    mB_g,
                    cute.slice_(self.mma_tiler, (0, None, None)),
                    (mma_tile_coord_mnl[1], None, 0),
                )
                # ((atom_v, rest_v), loopK)
                tAsA, tAgA_slice = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(thr_mma.partition_A(gA_mk), 0, 3),
                )
                tBsB, tBgB_slice = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(thr_mma.partition_B(gB_nk), 0, 3),
                )

                ab_producer_state.reset_count()
                peek_ab_empty_status = Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                tPsPa, tPgPa, tPsPb, tPgPb, peel_k_tile = None, None, None, None, None
                if cutlass.const_expr(mining):
                    # This tile's BF16 peel operands: the MMA tile's rows of
                    # A_peel (this CTA's half of them for the 2-CTA pair) and
                    # rows [group * n + tile_n * cta_n, +cta_n) of B_peel,
                    # addressed through the global descriptors by coordinate
                    # offset. They are issued once the A/B ring has wrapped:
                    # by then the MMA warp has consumed this tile's first
                    # k-tile, hence finished the previous tile's peel and
                    # freed the single peel stage, so the acquire never
                    # stalls the A/B loads, and the load has the rest of the
                    # mainloop to land. With the peel accumulated in place the
                    # MMA warp consumes the previous tile's peel operands
                    # ``num_ab_stage`` k-tiles into this tile, so the load
                    # moves one ring depth later to keep the acquire free.
                    row_a = m_off + mma_tile_coord_mnl[0] * self.mma_tiler[0]
                    row_b = cur_group_idx * n + tile_info[1] * self.cta_tile_shape_mnk[1]
                    gPa = cute.local_tile(
                        cute.domain_offset((row_a, 0, 0), mPa_mkl),
                        (self.mma_tiler[0], R2),
                        (0, None, None),
                    )
                    gPb = cute.local_tile(
                        cute.domain_offset((row_b, 0, 0), mPb_nkl),
                        (self.mma_tiler[1], R2),
                        (0, None, None),
                    )
                    tPsPa, tPgPa = cpasync.tma_partition(
                        tma_atom_pa,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sPa, 0, 3),
                        cute.group_modes(thr_mma_peel.partition_A(gPa), 0, 3),
                    )
                    tPsPb, tPgPb = cpasync.tma_partition(
                        tma_atom_pb,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sPb, 0, 3),
                        cute.group_modes(thr_mma_peel.partition_B(gPb), 0, 3),
                    )
                    peel_load_k_tile = self.num_ab_stage
                    if cutlass.const_expr(peel_in_place):
                        peel_load_k_tile = 2 * self.num_ab_stage + 1
                    peel_k_tile = cutlass.min(Int32(peel_load_k_tile), k_tile_cnt - 1)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    ab_producer_state.advance()
                    peek_ab_empty_status = Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    if cutlass.const_expr(mining):  # noqa: SIM102 (trace-time guard)
                        if k_tile == peel_k_tile:
                            peel_pipeline.producer_acquire(peel_producer_state)
                            peel_bar = peel_pipeline.producer_get_barrier(peel_producer_state)
                            cute.copy(
                                tma_atom_pa,
                                tPgPa[(None, 0, 0)],
                                tPsPa[(None, 0)],
                                tma_bar_ptr=peel_bar,
                            )
                            cute.copy(
                                tma_atom_pb,
                                tPgPb[(None, 0, 0)],
                                tPsPb[(None, 0)],
                                tma_bar_ptr=peel_bar,
                            )
                            peel_pipeline.producer_commit(peel_producer_state)

                if cutlass.const_expr(mining):
                    peel_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            ab_pipeline.producer_tail(ab_producer_state)
            if cutlass.const_expr(mining):
                peel_pipeline.producer_tail(peel_producer_state)

        #
        # Specialized Scale load warp
        #
        if cutlass.const_expr(mining):  # noqa: SIM102 (trace-time guard over a runtime predicate)
            if warp_idx == self.scale_warp_id:
                # Mining has no blockwise scales: the warp only keeps its place
                # in the tile-info pipeline (it is one of its consumers).
                cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
                self._walk_tile_info(tile_info_pipeline, sInfo)

        if cutlass.const_expr(not mining):  # noqa: SIM102 (trace-time guard)
            if warp_idx == self.scale_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)

                scale_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_scale_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
                is_valid_tile = tile_info[3] == 1

                while is_valid_tile:
                    cur_group_idx = tile_info[2]
                    m_off, m_cnt = self._group_rows(m_indptr, cur_group_idx, cum_m)
                    mSFA_g = self._make_group_sfa(mSFA, m_off, m_cnt, k_blocks)
                    mSFB_g = self._make_group_sfb(mSFB, cur_group_idx, n_blocks, k_blocks)

                    # (bM, bK, loopM, loopK, loopL)
                    gSFA_mkl = cute.local_tile(
                        mSFA_g,
                        cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
                        (None, None, None),
                    )
                    # (bN, bK, loopN, loopK, loopL)
                    gSFB_nkl = cute.local_tile(
                        mSFB_g,
                        cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
                        (None, None, None),
                    )
                    cSFA_mkl = cute.make_identity_tensor(cute.shape(mSFA_g))
                    cSFB_nkl = cute.make_identity_tensor(cute.shape(mSFB_g))
                    cSFA = cute.local_tile(
                        cSFA_mkl,
                        cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
                        (None, None, None),
                    )
                    cSFB = cute.local_tile(
                        cSFB_nkl,
                        cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
                        (None, None, None),
                    )
                    tAgSFA_mkl = thr_copy_sfa.partition_S(gSFA_mkl)
                    tAcSFA = thr_copy_sfa.partition_S(cSFA)
                    tBgSFB_nkl = thr_copy_sfb.partition_S(gSFB_nkl)
                    tBcSFB = thr_copy_sfb.partition_S(cSFB)

                    tApSFA = cute.make_rmem_tensor(
                        cute.make_layout(
                            cute.filter_zeros(cute.slice_(tAsSFA, (None, None, None, 0))).shape
                        ),
                        cutlass.Boolean,
                    )
                    tBpSFB = cute.make_rmem_tensor(
                        cute.make_layout(
                            cute.filter_zeros(cute.slice_(tBsSFB, (None, None, None, 0))).shape
                        ),
                        cutlass.Boolean,
                    )

                    # Scale stages per output tile: one per K tile (blockwise), or
                    # one for the whole tile (full-K chain; K-block 0's scales).
                    scale_iters = Int32(1) if cutlass.const_expr(self.full_k_acc) else k_tile_cnt

                    scale_producer_state.reset_count()
                    peek_scale_empty_status = Boolean(1)
                    if scale_producer_state.count < scale_iters:
                        peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                            scale_producer_state
                        )

                    for k_tile in cutlass.range(0, scale_iters, 1, unroll=1):  # noqa: B007 (DSL loop; count carried by pipeline state)
                        tAsSFA_pipe = cute.filter_zeros(
                            tAsSFA[(None, None, None, scale_producer_state.index)]
                        )
                        tBsSFB_pipe = cute.filter_zeros(
                            tBsSFB[(None, None, None, scale_producer_state.index)]
                        )
                        tAgSFA_k = cute.filter_zeros(
                            tAgSFA_mkl[
                                (None, None, None, tile_info[0], scale_producer_state.count, 0)
                            ]
                        )
                        tBgSFB_k = cute.filter_zeros(
                            tBgSFB_nkl[
                                (None, None, None, tile_info[1], scale_producer_state.count, 0)
                            ]
                        )
                        tAcSFA_compact = cute.filter_zeros(
                            cute.slice_(
                                tAcSFA,
                                (None, None, None, tile_info[0], scale_producer_state.count, 0),
                            )
                        )
                        tBcSFB_compact = cute.filter_zeros(
                            cute.slice_(
                                tBcSFB,
                                (None, None, None, tile_info[1], scale_producer_state.count, 0),
                            )
                        )
                        for i in cutlass.range_constexpr(cute.size(tApSFA, mode=[1])):
                            tApSFA[((0, 0), i, (0, 0))] = cute.elem_less(
                                tAcSFA_compact[(i)][0], mSFA_g.shape[0]
                            )
                        for i in cutlass.range_constexpr(cute.size(tBpSFB, mode=[1])):
                            tBpSFB[((0, 0), i, (0, 0))] = cute.elem_less(
                                tBcSFB_compact[(i)][0], mSFB_g.shape[0]
                            )

                        scale_pipeline.producer_acquire(
                            scale_producer_state, peek_scale_empty_status
                        )

                        cute.copy(tiled_copy_sfa, tAgSFA_k, tAsSFA_pipe, pred=tApSFA)
                        cute.copy(tiled_copy_sfb, tBgSFB_k, tBsSFB_pipe, pred=tBpSFB)

                        scale_pipeline.producer_commit(scale_producer_state)

                        scale_producer_state.advance()
                        peek_scale_empty_status = Boolean(1)
                        if scale_producer_state.count < scale_iters:
                            peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                                scale_producer_state
                            )

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                scale_pipeline.producer_tail(scale_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            tmem.wait_for_alloc()

            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            peel_acc_producer_state, peel_consumer_state = None, None
            peel_acc_commit_state, fold_done_state, pending_peel = None, None, None
            if cutlass.const_expr(mining):
                peel_acc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_acc_stage if peel_in_place else self.num_peel_stage,
                )
                peel_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, 1
                )
            if cutlass.const_expr(peel_in_place):
                # In-place peel: the previous tile's accumulator stage is
                # peeled one tile late, so its commit slot and the fold
                # warps' release of that stage (which admits the peel UMMA:
                # a consumer-phase wait on the accumulator pipeline's empty
                # barrier) trail the acquire state by one tile.
                peel_acc_commit_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_acc_stage
                )
                fold_done_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )
                pending_peel = Boolean(False)
                tiled_mma_peel.set(tcgen05.Field.ACCUMULATE, True)

            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                ab_consumer_state.reset_count()
                peek_ab_full_status = Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                acc_producer_state.reset_count()
                peek_acc_empty_status = Boolean(1)
                if cutlass.const_expr(not peel_in_place):  # noqa: SIM102 (trace-time guard)
                    if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                            acc_producer_state
                        )

                # Full-K chain: one accumulator stage per output tile, acquired
                # here and restarted (ACCUMULATE=False) only for its first
                # k-block; every K tile then chains into it. In-place peel:
                # the stage is free once the epilogue has stored its peeled
                # contents (the fold's release admits only the peel UMMA).
                if cutlass.const_expr(self.full_k_acc):
                    if is_leader_cta:
                        if cutlass.const_expr(peel_in_place):
                            peel_acc_pipeline.producer_acquire(peel_acc_producer_state)
                        else:
                            acc_pipeline.producer_acquire(acc_producer_state, peek_acc_empty_status)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                peel_mma_k_tile = None
                if cutlass.const_expr(mining):
                    # The peel UMMA is issued mid-mainloop, once its operands
                    # (loaded when the A/B ring wrapped) have landed and the
                    # previous tile's epilogue has long released the peel
                    # stage, so its barrier waits and issue slots hide under
                    # queued mainloop UMMAs instead of opening a gap between
                    # tiles. In place, the PREVIOUS tile's peel is issued one
                    # ring depth into this tile: its fold has finished by then
                    # and the TMA warp's next peel load waits on this release.
                    if cutlass.const_expr(peel_in_place):
                        peel_mma_k_tile = cutlass.min(Int32(self.num_ab_stage), k_tile_cnt - 1)
                    else:
                        peel_mma_k_tile = cutlass.max(
                            cutlass.min(Int32(self.num_ab_stage), k_tile_cnt - 1),
                            k_tile_cnt // 2,
                        )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                    # Blockwise: one accumulator stage per K tile, restarted
                    # every K tile.
                    if cutlass.const_expr(not self.full_k_acc):
                        if is_leader_cta:
                            acc_pipeline.producer_acquire(acc_producer_state, peek_acc_empty_status)

                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)

                        if cutlass.const_expr(mining and not peel_in_place):  # noqa: SIM102 (trace-time guard)
                            if k_tile == peel_mma_k_tile:
                                # The BF16 peel of this tile into the peel
                                # TMEM stage (restarted: ACCUMULATE=False on
                                # its first k-block), committed to the
                                # epilogue warps, which add it to the pre-peel
                                # accumulator in registers.
                                tCtPeel = tCtAcc_base[(None, None, None, self.num_acc_stage)]
                                peel_acc_pipeline.producer_acquire(peel_acc_producer_state)
                                peel_pipeline.consumer_wait(peel_consumer_state)
                                tiled_mma_peel.set(tcgen05.Field.ACCUMULATE, False)
                                num_peel_kblocks = cute.size(tCrPa, mode=[2])
                                for kblock_idx in cutlass.range(num_peel_kblocks, unroll_full=True):
                                    kblock_coord = (
                                        None,
                                        None,
                                        kblock_idx,
                                        peel_consumer_state.index,
                                    )
                                    cute.gemm(
                                        tiled_mma_peel,
                                        tCtPeel,
                                        tCrPa[kblock_coord],
                                        tCrPb[kblock_coord],
                                        tCtPeel,
                                    )
                                    tiled_mma_peel.set(tcgen05.Field.ACCUMULATE, True)
                                peel_pipeline.consumer_release(peel_consumer_state)
                                peel_acc_pipeline.producer_commit(peel_acc_producer_state)

                    if cutlass.const_expr(peel_in_place):  # noqa: SIM102 (trace-time guard)
                        if k_tile == peel_mma_k_tile:  # noqa: SIM102 (DSL dynamic Boolean)
                            if pending_peel:
                                if is_leader_cta:
                                    self._peel_in_place(
                                        tiled_mma_peel,
                                        tCtAcc_base,
                                        tCrPa,
                                        tCrPb,
                                        acc_pipeline,
                                        peel_pipeline,
                                        peel_acc_pipeline,
                                        fold_done_state,
                                        peel_consumer_state,
                                        peel_acc_commit_state,
                                    )
                                fold_done_state.advance()
                                peel_consumer_state.advance()
                                peel_acc_commit_state.advance()
                                pending_peel = Boolean(False)

                    ab_consumer_state.advance()
                    peek_ab_full_status = Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:  # noqa: SIM102 (DSL dynamic Boolean)
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                    if cutlass.const_expr(not self.full_k_acc):
                        if is_leader_cta:
                            acc_pipeline.producer_commit(acc_producer_state)

                        acc_producer_state.advance()
                        if acc_producer_state.count < k_tile_cnt:  # noqa: SIM102 (DSL dynamic Boolean)
                            if is_leader_cta:
                                peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                                    acc_producer_state
                                )

                if cutlass.const_expr(self.full_k_acc):
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                if cutlass.const_expr(mining and not peel_in_place):
                    peel_consumer_state.advance()
                    peel_acc_producer_state.advance()
                if cutlass.const_expr(peel_in_place):
                    peel_acc_producer_state.advance()
                    pending_peel = Boolean(True)

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            if cutlass.const_expr(peel_in_place):
                # The last tile's peel has no following mainloop to hide under.
                if pending_peel:
                    if is_leader_cta:
                        self._peel_in_place(
                            tiled_mma_peel,
                            tCtAcc_base,
                            tCrPa,
                            tCrPb,
                            acc_pipeline,
                            peel_pipeline,
                            peel_acc_pipeline,
                            fold_done_state,
                            peel_consumer_state,
                            peel_acc_commit_state,
                        )
                    fold_done_state.advance()
                    peel_consumer_state.advance()
                    peel_acc_commit_state.advance()
                # Every fold release was consumed by a peel; only the
                # epilogue's releases of the peeled stages remain outstanding.
                peel_acc_pipeline.producer_tail(peel_acc_producer_state)
            else:
                acc_pipeline.producer_tail(acc_producer_state)
                if cutlass.const_expr(mining):
                    peel_acc_pipeline.producer_tail(peel_acc_producer_state)
            # Every accumulator stage has been released back to this warp:
            # no MMA is in flight and no fold read of TMEM is outstanding.
            self.tmem_free_barrier.arrive()

        #
        # Specialized acc update warps: mining (fold, hash, publish)
        #
        if cutlass.const_expr(mining):  # noqa: SIM102 (trace-time guard)
            if warp_idx <= self.acc_update_warp_id[-1]:
                cute.arch.setmaxregister_increase(self.num_regs_acc_update_warps)
                tmem.wait_for_alloc()

                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

                epi_tidx = tidx % 128
                (
                    tiled_copy_t2r,
                    _,
                    tTR_tAcc_base,
                    tTR_rAcc,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self.acc_update_tmem_copy_and_partition(
                    epi_tidx,
                    tCtAcc_base,
                    None,
                    tCgC,
                    None,
                    None,
                    epi_tile,
                )

                # Static fold mapping (asserts whole-word ownership per thread)
                # and the dynamic accumulator row and word base this thread owns.
                cta_m, cta_n = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
                epi_tile_n = cutlass.const_expr(cute.size(epi_tile[1]))
                assert epi_tile_n % 8 == 0, f"epi_tile_n must be a multiple of 8, got {epi_tile_n}"
                fold_groups, words_per_class = _fold_register_groups(
                    tiled_copy_t2r, self.words_per_row, self.ltile_cols
                )
                assert len(fold_groups) == 1, "the mining variant expects one row class per thread"
                fold_group = fold_groups[0]
                cAcc = cute.make_identity_tensor((cta_m, cta_n))
                cAcc_epi = cute.flat_divide(cAcc, epi_tile)
                tTR_cAcc = tiled_copy_t2r.get_slice(epi_tidx).partition_D(cAcc_epi)
                tTR_cAcc = cute.group_modes(tTR_cAcc, 3, cute.rank(tTR_cAcc))
                subtile_cnt = cutlass.const_expr(cute.size(tTR_cAcc.shape, mode=[3]))
                tTR_cAcc_first = tTR_cAcc[(None, None, None, 0)]
                thread_row = tTR_cAcc_first[fold_group[0][0]][0]
                first_column = tTR_cAcc_first[fold_group[0][0]][1]
                word_base = (first_column >> 1) & (self.words_per_row - 1)
                row_shift = self.ltile_rows.bit_length() - 1  # ltile_rows is 4 or 16
                num_partials = self.column_tiles * words_per_class

                chaining_value = [mKey[i] for i in range(8)]
                threshold_words = [mThr[i] for i in range(8)]

                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
                is_valid_tile = tile_info[3] == 1

                while is_valid_tile:
                    tile_m = tile_info[0]
                    tile_n = tile_info[1]
                    cur_group_idx = tile_info[2]
                    m_off, m_cnt = self._group_rows(m_indptr, cur_group_idx, cum_m)
                    # Real rows of the group, clamped into its block (host
                    # never reads m_valid; a value past the block is the block).
                    m_valid_g = cutlass.min(cutlass.max(m_valid[cur_group_idx], Int32(0)), m_cnt)

                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    # Fold the pre-peel accumulator into lottery words, then
                    # release the stage: the MMA warp reuses it after the
                    # epilogue warps' release too (separate peel stage), or
                    # runs the in-place peel UMMA into it (wide tile).
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    partials = cute.make_rmem_tensor(num_partials, Uint32)
                    for word in cutlass.range_constexpr(num_partials):
                        partials[word] = Uint32(0)
                    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                        cute.copy(
                            tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc
                        )
                        for value, column_offset, word_slot in fold_group:
                            # Trace-time: the partial this column feeds.
                            column_tile = (
                                subtile_idx * epi_tile_n + column_offset
                            ) // self.ltile_cols
                            word = column_tile * words_per_class + word_slot
                            partials[word] = _rotr32(
                                partials[word] * Uint32(_FOLD_MUL)
                                + tTR_rAcc[value].bitcast(Uint32),
                                19,
                            )
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    # Stage this thread's folded words into their messages'
                    # 64-byte blocks: word j of message (lottery row r, column
                    # tile ct) folds row ``ltile_rows*r + (j >> 2)`` at columns
                    # congruent to ``{2*(j&3), 2*(j&3)+1}`` mod 8 (4-row family).
                    for column_tile in cutlass.range_constexpr(self.column_tiles):
                        message = (thread_row >> row_shift) * self.column_tiles + column_tile
                        for slot in cutlass.range_constexpr(words_per_class):
                            lane = (
                                self.words_per_row * (thread_row & (self.ltile_rows - 1))
                                + word_base
                                + slot
                            )
                            sExt[message, lane] = partials[column_tile * words_per_class + slot]
                    # Publishes sExt across the four warps.
                    self.acc_sync_barrier.arrive_and_wait()

                    # Keyed BLAKE3 over the staged messages; it overlaps the
                    # next tile's mainloop (and the epilogue).
                    self._compress_and_publish(
                        sExt,
                        chaining_value,
                        threshold_words,
                        mRecord,
                        mLock,
                        mHashB,
                        mCodesSrc,
                        mScalesSrc,
                        mCodesDst,
                        mScalesDst,
                        layer_id,
                        record_hits,
                        tile_m,
                        tile_n,
                        cur_group_idx,
                        m_off,
                        m_cnt,
                        m_valid_g,
                        epi_tidx,
                    )

                    # Separates this tile's sExt reads from the next tile's staging.
                    self.acc_sync_barrier.arrive_and_wait()

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()
                self.tmem_free_barrier.arrive()

        #
        # Specialized acc update warps: full-K (direct epilogue) -- no work
        # besides owning their share of the TMEM barrier and tile pipeline.
        #
        if cutlass.const_expr(not mining and self.direct_epilogue):  # noqa: SIM102 (trace-time guard)
            if warp_idx <= self.acc_update_warp_id[-1]:
                cute.arch.setmaxregister_increase(self.num_regs_acc_update_warps)
                tmem.wait_for_alloc()
                self._walk_tile_info(tile_info_pipeline, sInfo)
                self.tmem_free_barrier.arrive()

        #
        # Specialized acc update warps: blockwise scaling
        #
        if cutlass.const_expr(not mining and not self.direct_epilogue):  # noqa: SIM102 (trace-time guard)
            if warp_idx <= self.acc_update_warp_id[-1]:
                cute.arch.setmaxregister_increase(self.num_regs_acc_update_warps)
                tmem.wait_for_alloc()

                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
                tCtAcc_final = cute.make_tensor(
                    tCtAcc_base.iterator + self.tmem_final_offset, tCtAcc_base.layout
                )

                epi_tidx = tidx % 128
                (
                    tiled_copy_t2r,
                    tiled_copy_r2t,
                    tTR_tAcc_base,
                    tTR_rAcc,
                    tTR_rAcc_final,
                    tTR_sSFA,
                    tTR_sSFB,
                    tRT_rAcc,
                    tRT_tAcc_base,
                ) = self.acc_update_tmem_copy_and_partition(
                    epi_tidx,
                    tCtAcc_base,
                    tCtAcc_final,
                    tCgC,
                    sSFA_view_as_C,
                    sSFB_view_as_C,
                    epi_tile,
                )

                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )
                scale_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_scale_stage
                )
                epi_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
                is_valid_tile = tile_info[3] == 1

                while is_valid_tile:
                    tTR_rAcc_final.fill(0.0)

                    tTR_rSFA = cute.make_rmem_tensor(
                        cute.slice_(tTR_sSFA, (None, None, None, 0, None, 0)).shape,
                        self.acc_dtype,
                    )
                    tTR_rSFB = cute.make_rmem_tensor(
                        cute.slice_(tTR_sSFB, (None, None, None, 0, None, 0)).shape,
                        self.acc_dtype,
                    )

                    # Accumulator/scale stages consumed per output tile: one per K
                    # tile (blockwise) or a single full-K accumulator.
                    acc_iters = Int32(1) if cutlass.const_expr(self.full_k_acc) else k_tile_cnt

                    scale_consumer_state.reset_count()
                    peek_scale_full_status = Boolean(1)
                    if scale_consumer_state.count < acc_iters:
                        peek_scale_full_status = scale_pipeline.consumer_try_wait(
                            scale_consumer_state
                        )

                    acc_consumer_state.reset_count()
                    peek_acc_full_status = Boolean(1)
                    if acc_consumer_state.count < acc_iters:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                    for k_tile in cutlass.range(0, acc_iters, 1, unroll=1):  # noqa: B007 (DSL loop; count carried by pipeline state)
                        tTR_tAcc = tTR_tAcc_base[
                            (None, None, None, None, None, acc_consumer_state.index)
                        ]

                        scale_pipeline.consumer_wait(scale_consumer_state, peek_scale_full_status)

                        tTR_sSFA_slice = cute.slice_(
                            tTR_sSFA,
                            (None, None, None, 0, None, scale_consumer_state.index),
                        )
                        tTR_sSFB_slice = cute.slice_(
                            tTR_sSFB,
                            (None, None, None, 0, None, scale_consumer_state.index),
                        )

                        scale_atom_copy = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            self.acc_dtype,
                            num_bits_per_copy=self.acc_dtype.width,
                        )

                        cute.copy(scale_atom_copy, tTR_sSFA_slice, tTR_rSFA)
                        cute.copy(scale_atom_copy, tTR_sSFB_slice, tTR_rSFB)

                        acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)

                        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                        for subtile_idx in cutlass.range(subtile_cnt):
                            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                            tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
                            tTR_rSFA_subtile = tTR_rSFA[(None, None, None, subtile_idx)]
                            tTR_rSFB_subtile = tTR_rSFB[(None, None, None, subtile_idx)]

                            acc_vec = tTR_rAcc.load()
                            final_vec = tTR_rAcc_subtile.load()
                            scale_a = tTR_rSFA_subtile.load()
                            scale_b = tTR_rSFB_subtile.load()
                            scale = scale_a * scale_b
                            final_vec = acc_vec * scale + final_vec
                            tTR_rAcc_subtile.store(final_vec.to(self.acc_dtype))

                        scale_pipeline.consumer_release(scale_consumer_state)
                        scale_consumer_state.advance()

                        peek_scale_full_status = Boolean(1)
                        if scale_consumer_state.count < acc_iters:
                            peek_scale_full_status = scale_pipeline.consumer_try_wait(
                                scale_consumer_state
                            )

                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                        peek_acc_full_status = Boolean(1)
                        if acc_consumer_state.count < acc_iters:
                            peek_acc_full_status = acc_pipeline.consumer_try_wait(
                                acc_consumer_state
                            )

                    tRT_tAcc = tRT_tAcc_base[(None, None, None, None, None, 0)]
                    tRT_tAcc = cute.group_modes(tRT_tAcc, 3, cute.rank(tRT_tAcc))

                    epi_pipeline.producer_acquire(epi_producer_state)

                    cute.copy(tiled_copy_r2t, tTR_rAcc_final, tRT_tAcc)
                    cute.arch.fence_view_async_tmem_store()

                    epi_pipeline.producer_commit(epi_producer_state)
                    epi_producer_state.advance()

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()
                self.tmem_free_barrier.arrive()

        #
        # Specialized epilogue warps: full-K direct epilogue. Each committed
        # accumulator stage is read straight from TMEM, scaled (``sfa*sfb``
        # or the mining row/column unscale), converted and TMA-stored; the
        # stage is released right after its last TMEM load.
        #
        if cutlass.const_expr(self.direct_epilogue):  # noqa: SIM102 (trace-time guard)
            if warp_idx <= self.epilog_warp_id[-1] and warp_idx >= self.epilog_warp_id[0]:
                cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)

                tmem.allocate(self.num_tmem_alloc_cols)
                tmem.wait_for_alloc()

                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base_ = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

                epi_tidx = tidx % 128
                (
                    tiled_copy_t2r,
                    tTR_tAcc_base,
                    tTR_rAcc,
                ) = self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base_, tCgC, epi_tile, use_2cta_instrs
                )
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                # The SMEM side of the store partition is tile-invariant; the
                # GMEM side is rebuilt per tile from the group's ragged view.
                gC_any = cute.local_tile(
                    self._ragged_group(mC_mnl, Int32(0), Int32(0)),
                    cute.slice_(self.mma_tiler, (None, None, 0)),
                    (0, 0),
                )
                bSG_sC, _ = self.epilog_gmem_copy_and_partition(
                    tma_atom_c, thr_mma.partition_C(gC_any), epi_tile, sC
                )

                cta_m, cta_n = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
                epi_tile_n = cutlass.const_expr(cute.size(epi_tile[1]))
                tTR_tAcc_stage0 = tTR_tAcc_base[(None, None, None, None, None, 0)]
                tTR_tAcc_stage0 = cute.group_modes(tTR_tAcc_stage0, 3, cute.rank(tTR_tAcc_stage0))
                subtile_cnt = cutlass.const_expr(cute.size(tTR_tAcc_stage0.shape, mode=[3]))

                # Scale operands. Non-mining: the K-block-0 ``sfa``/``sfb``
                # stage viewed as C and partitioned like the accumulator.
                # Mining: this thread's accumulator row (``alpha_a``) and the
                # static subtile column of each t2r register (``sAlB``).
                tTR_sSFA, tTR_sSFB, scale_atom_copy = None, None, None
                reg_columns, thread_row = None, None
                if cutlass.const_expr(not mining):
                    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
                    tTR_sSFA = thr_copy_t2r.partition_D(cute.flat_divide(sSFA_view_as_C, epi_tile))
                    tTR_sSFB = thr_copy_t2r.partition_D(cute.flat_divide(sSFB_view_as_C, epi_tile))
                    scale_atom_copy = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.acc_dtype,
                        num_bits_per_copy=self.acc_dtype.width,
                    )
                if cutlass.const_expr(mining):
                    reg_columns = self._register_columns(tiled_copy_t2r)
                    cAcc = cute.make_identity_tensor((cta_m, cta_n))
                    cAcc_epi = cute.flat_divide(cAcc, epi_tile)
                    tTR_cAcc = tiled_copy_t2r.get_slice(epi_tidx).partition_D(cAcc_epi)
                    tTR_cAcc = cute.group_modes(tTR_cAcc, 3, cute.rank(tTR_cAcc))
                    thread_row = tTR_cAcc[(None, None, None, 0)][0][0]

                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )
                scale_consumer_state = None
                if cutlass.const_expr(not mining):
                    scale_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_scale_stage
                    )
                # Mining: the peeled accumulator handshake. With a
                # separate peel stage it is partitioned like an accumulator
                # stage and added in registers; in place, the accumulator
                # stage itself arrives already peeled through this pipeline.
                peel_acc_consumer_state, tTR_tPeel, tTR_rPeel = None, None, None
                if cutlass.const_expr(mining):
                    peel_acc_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer,
                        self.num_acc_stage if peel_in_place else self.num_peel_stage,
                    )
                if cutlass.const_expr(mining and not peel_in_place):
                    tTR_tPeel = tTR_tAcc_base[(None, None, None, None, None, self.num_acc_stage)]
                    tTR_tPeel = cute.group_modes(tTR_tPeel, 3, cute.rank(tTR_tPeel))
                    tTR_rPeel = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)

                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilog_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=c_producer_group,
                )

                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
                is_valid_tile = tile_info[3] == 1

                num_prev_subtiles = Int32(0)

                while is_valid_tile:
                    cur_group_idx = tile_info[2]
                    m_off, m_cnt = self._group_rows(m_indptr, cur_group_idx, cum_m)
                    mma_tile_coord_mnl = (
                        tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                        tile_info[1],
                        0,
                    )
                    # This tile of the group's ragged C view: rows past
                    # ``m_cnt`` are outside the descriptor and never stored
                    # (they belong to the next group).
                    mC_g = self._ragged_group(mC_mnl, m_off, m_cnt)
                    gC_mn = cute.local_tile(
                        mC_g,
                        cute.slice_(self.mma_tiler, (None, None, 0)),
                        (mma_tile_coord_mnl[0], mma_tile_coord_mnl[1]),
                    )
                    # ((ATOM_V, REST_V), EPI_M, EPI_N)
                    _, bSG_gC = self.epilog_gmem_copy_and_partition(
                        tma_atom_c, thr_mma.partition_C(gC_mn), epi_tile, sC
                    )
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    # Per-tile scale operands, fetched while the mainloop runs.
                    tTR_rSFA, tTR_rSFB = None, None
                    if cutlass.const_expr(not mining):
                        tTR_rSFA = cute.make_rmem_tensor(
                            cute.slice_(tTR_sSFA, (None, None, None, 0, None, 0)).shape,
                            self.acc_dtype,
                        )
                        tTR_rSFB = cute.make_rmem_tensor(
                            cute.slice_(tTR_sSFB, (None, None, None, 0, None, 0)).shape,
                            self.acc_dtype,
                        )
                        scale_pipeline.consumer_wait(scale_consumer_state)
                        cute.copy(
                            scale_atom_copy,
                            cute.slice_(
                                tTR_sSFA, (None, None, None, 0, None, scale_consumer_state.index)
                            ),
                            tTR_rSFA,
                        )
                        cute.copy(
                            scale_atom_copy,
                            cute.slice_(
                                tTR_sSFB, (None, None, None, 0, None, scale_consumer_state.index)
                            ),
                            tTR_rSFB,
                        )
                        scale_pipeline.consumer_release(scale_consumer_state)
                        scale_consumer_state.advance()
                    inv_alpha_row = Float32(1.0)
                    if cutlass.const_expr(mining):
                        # Stage this tile's column unscale slice (one element
                        # per thread) and fetch this thread's row scale; the
                        # barrier publishes the slice to the four warps. The
                        # previous tile's reads all precede its last subtile
                        # barrier, so the slot is free.
                        row_local = tile_info[0] * cta_m + thread_row
                        col_base = tile_info[1] * cta_n
                        for chunk in cutlass.range_constexpr(cta_n // 128):
                            column = chunk * 128 + epi_tidx
                            # The tile may overhang n; those columns are never
                            # stored, any finite value will do.
                            unscale = Float32(0.0)
                            if col_base + column < n:
                                unscale = mAlB[cur_group_idx * n + col_base + column]
                            sAlB[column] = unscale
                        if row_local < m_cnt:
                            inv_alpha_row = cute.arch.rcp_approx(
                                mAlA[m_off + row_local].to(Float32)
                            )
                        self.epilog_sync_barrier.arrive_and_wait()

                    # The stage to read: the accumulator stage (plus, with a
                    # separate peel stage, the peel), or -- in place -- the
                    # already peeled accumulator stage.
                    if cutlass.const_expr(not peel_in_place):
                        acc_pipeline.consumer_wait(acc_consumer_state)
                    if cutlass.const_expr(mining):
                        peel_acc_pipeline.consumer_wait(peel_acc_consumer_state)

                    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                        cute.copy(
                            tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc
                        )
                        if cutlass.const_expr(mining and not peel_in_place):
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tPeel[(None, None, None, subtile_idx)],
                                tTR_rPeel,
                            )
                        if cutlass.const_expr(subtile_idx == subtile_cnt - 1):
                            # Last TMEM load of the tile: hand the stage(s)
                            # back to the MMA warp before converting and storing.
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                if cutlass.const_expr(not peel_in_place):
                                    acc_pipeline.consumer_release(acc_consumer_state)
                                if cutlass.const_expr(mining):
                                    peel_acc_pipeline.consumer_release(peel_acc_consumer_state)

                        if cutlass.const_expr(not mining):
                            scale = (
                                tTR_rSFA[(None, None, None, subtile_idx)].load()
                                * tTR_rSFB[(None, None, None, subtile_idx)].load()
                            )
                            tTR_rAcc.store(tTR_rAcc.load() * scale)
                        if cutlass.const_expr(mining and not peel_in_place):
                            tTR_rAcc.store(tTR_rAcc.load() + tTR_rPeel.load())
                        if cutlass.const_expr(mining):
                            for value in cutlass.range_constexpr(len(reg_columns)):
                                column = subtile_idx * epi_tile_n + reg_columns[value]
                                tTR_rAcc[value] = tTR_rAcc[value] * inv_alpha_row * sAlB[column]

                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))

                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                    acc_consumer_state.advance()
                    if cutlass.const_expr(mining):
                        peel_acc_consumer_state.advance()

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                tmem.relinquish_alloc_permit()
                # Joins the other epilogue warps' last TMEM loads and the
                # MMA/fold warps' last stage before deallocating.
                self.tmem_free_barrier.arrive_and_wait()
                tmem.free(tmem_ptr)
                c_pipeline.producer_tail()

        #
        # Specialized epilogue warps: blockwise (final TMEM region)
        #
        if cutlass.const_expr(not self.direct_epilogue):  # noqa: SIM102 (trace-time guard)
            if warp_idx <= self.epilog_warp_id[-1] and warp_idx >= self.epilog_warp_id[0]:
                cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)

                tmem.allocate(self.num_tmem_alloc_cols)
                tmem.wait_for_alloc()

                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base_ = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
                tCtAcc_final = cute.make_tensor(
                    tCtAcc_base_.iterator + self.tmem_final_offset, tCtAcc_base_.layout
                )

                epi_tidx = tidx % 128
                (
                    tiled_copy_t2r,
                    tTR_tAcc_base,
                    tTR_rAcc,
                ) = self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_final, tCgC, epi_tile, use_2cta_instrs
                )

                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                # Store partition and ragged C view: as in the direct epilogue.
                gC_any = cute.local_tile(
                    self._ragged_group(mC_mnl, Int32(0), Int32(0)),
                    cute.slice_(self.mma_tiler, (None, None, 0)),
                    (0, 0),
                )
                bSG_sC, _ = self.epilog_gmem_copy_and_partition(
                    tma_atom_c, thr_mma.partition_C(gC_any), epi_tile, sC
                )

                epi_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, 1
                )

                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilog_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=c_producer_group,
                )

                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
                is_valid_tile = tile_info[3] == 1

                num_prev_subtiles = Int32(0)

                while is_valid_tile:
                    cur_group_idx = tile_info[2]
                    m_off, m_cnt = self._group_rows(m_indptr, cur_group_idx, cum_m)
                    mma_tile_coord_mnl = (
                        tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                        tile_info[1],
                        0,
                    )
                    mC_g = self._ragged_group(mC_mnl, m_off, m_cnt)
                    gC_mn = cute.local_tile(
                        mC_g,
                        cute.slice_(self.mma_tiler, (None, None, 0)),
                        (mma_tile_coord_mnl[0], mma_tile_coord_mnl[1]),
                    )
                    # ((ATOM_V, REST_V), EPI_M, EPI_N)
                    _, bSG_gC = self.epilog_gmem_copy_and_partition(
                        tma_atom_c, thr_mma.partition_C(gC_mn), epi_tile, sC
                    )

                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, epi_consumer_state.index)
                    ]

                    epi_pipeline.consumer_wait(epi_consumer_state)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    for subtile_idx in cutlass.range(subtile_cnt):
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))

                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                    epi_pipeline.consumer_release(epi_consumer_state)
                    epi_consumer_state.advance()

                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                tmem.relinquish_alloc_permit()
                # Joins the other epilogue warps' last TMEM loads and the
                # MMA/fold warps' last stage before deallocating.
                self.tmem_free_barrier.arrive_and_wait()
                tmem.free(tmem_ptr)
                c_pipeline.producer_tail()

    # ------------------------------------------------------------------
    # Mining helpers
    # ------------------------------------------------------------------

    @cute.jit
    def _peel_in_place(
        self,
        tiled_mma_peel: cute.TiledMma,
        tCtAcc_base: cute.Tensor,
        tCrPa: cute.Tensor,
        tCrPb: cute.Tensor,
        acc_pipeline: pipeline.PipelineUmmaAsync,
        peel_pipeline: pipeline.PipelineTmaUmma,
        peel_acc_pipeline: pipeline.PipelineUmmaAsync,
        fold_done_state: pipeline.PipelineState,
        peel_consumer_state: pipeline.PipelineState,
        peel_acc_commit_state: pipeline.PipelineState,
    ):
        """Accumulate the BF16 peel into a finished accumulator stage (MMA leader).

        Waits for the fold warps' release of the stage (their pre-peel TMEM
        read) and for the peel operands, chains the peel UMMA into the same
        TMEM columns and commits the peeled stage to the epilogue warps.
        ``tiled_mma_peel`` must already have ACCUMULATE set (it is never
        restarted in place).
        """
        tCtPeel = tCtAcc_base[(None, None, None, fold_done_state.index)]
        acc_pipeline.producer_acquire(fold_done_state)
        peel_pipeline.consumer_wait(peel_consumer_state)
        num_peel_kblocks = cute.size(tCrPa, mode=[2])
        for kblock_idx in cutlass.range(num_peel_kblocks, unroll_full=True):
            kblock_coord = (None, None, kblock_idx, peel_consumer_state.index)
            cute.gemm(
                tiled_mma_peel,
                tCtPeel,
                tCrPa[kblock_coord],
                tCrPb[kblock_coord],
                tCtPeel,
            )
        peel_pipeline.consumer_release(peel_consumer_state)
        peel_acc_pipeline.producer_commit(peel_acc_commit_state)

    @cute.jit
    def _walk_tile_info(self, tile_info_pipeline: pipeline.PipelineAsync, sInfo: cute.Tensor):
        """Consume the tile-info pipeline to its end without doing any work."""
        tile_info_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_tile_stage
        )
        is_valid_tile = Boolean(True)
        while is_valid_tile:
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            is_valid_tile = sInfo[(3, tile_info_consumer_state.index)] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

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
        tile_m: Int32,
        tile_n: Int32,
        group_idx: Int32,
        m_off: Int32,
        m_cnt: Int32,
        m_valid_g: Int32,
        tidx: Int32,
    ):
        """Hash one staged message per acc-update thread and publish the winner.

        Hits publish with group semantics through ``_HitPublishMixin``. A
        publishable lottery tile lies wholly inside the group's real rows
        (tiles reaching into rows ``>= m_valid`` do not exist in the
        protocol) and inside ``n`` (a wide CTA tile may overhang it).
        """
        if tidx < self.hash_threads:
            # The publish ballot needs a convergent warp, so a sub-warp
            # message tail keeps its whole warp hashing: excess lanes rehash
            # the last message and are excluded from the ballot.
            if cutlass.const_expr(self.msgs_cta % 32 == 0):
                is_message = Boolean(True)
                message = tidx
            else:
                is_message = Boolean(tidx < self.msgs_cta)
                message = cutlass.min(tidx, self.msgs_cta - 1)
            tile_row = (
                tile_m * (self.cta_tile_shape_mnk[0] // self.ltile_rows)
                + message // self.column_tiles
            )
            tile_column = tile_n * self.column_tiles + message % self.column_tiles
            words = [sExt[message, column].to(Uint32) for column in range(LANES)]
            digest = compress(
                list(chaining_value),
                words,
                64,
                SINGLE_BLOCK_KEYED_FLAGS,
            )
            local_hit = self._digest_below_threshold(digest, threshold) & is_message
            in_bounds = Boolean((tile_row + 1) * self.ltile_rows <= m_valid_g) & Boolean(
                (tile_column + 1) * self.ltile_cols <= self.problem_n
            )
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
                _HitGroup(group_idx, m_off, m_cnt),
            )

    def _register_columns(self, tiled_copy_t2r: cute.TiledCopy) -> list[int]:
        """Subtile column of each t2r register, asserted thread-invariant.

        The 128-row CTA tile's TMEM loads give every thread one accumulator
        row with the same column set, so the column unscale can index the
        staged slice with static offsets.
        """
        cells = _thread_fold_cells(tiled_copy_t2r, 0)
        assert len(cells) == 1, "the mining variant expects one accumulator row per thread"
        pairs = next(iter(cells.values()))
        for thread in range(1, cute.size(tiled_copy_t2r.layout_dst_tv_tiled, mode=[0])):
            other = next(iter(_thread_fold_cells(tiled_copy_t2r, thread).values()))
            assert other == pairs, "t2r register columns must be thread-invariant"
        columns = [0] * len(pairs)
        for value, column in pairs:
            columns[value] = column
        return columns

    def acc_update_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tAcc_final: cute.Tensor | None,
        gC_mnl: cute.Tensor,
        sSFA: cute.Tensor | None,
        sSFB: cute.Tensor | None,
        epi_tile: cute.Tile,
    ) -> tuple[
        cute.TiledCopy,
        cute.TiledCopy | None,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor | None,
        cute.Tensor | None,
        cute.Tensor | None,
        cute.Tensor | None,
        cute.Tensor | None,
    ]:
        """TMEM load/store tiled copies and partitions for the acc-update warps.

        ``tAcc_final=None`` (mining fold) skips the register -> TMEM store side.
        """
        tmem_load_atom = None
        tmem_store_atom = None
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
        else:
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )

        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_sSFA, tTR_sSFB = None, None
        if cutlass.const_expr(sSFA is not None):
            sSFA_epi = cute.flat_divide(sSFA, epi_tile)
            sSFB_epi = cute.flat_divide(sSFB, epi_tile)
            tTR_sSFA = thr_copy_t2r.partition_D(sSFA_epi)
            tTR_sSFB = thr_copy_t2r.partition_D(sSFB_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        if cutlass.const_expr(tAcc_final is None):
            return tiled_copy_t2r, None, tTR_tAcc, tTR_rAcc, None, tTR_sSFA, tTR_sSFB, None, None

        tAcc_final_epi = cute.flat_divide(tAcc_final[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_r2t = tcgen05.make_tmem_copy(
            tmem_store_atom, tAcc_final_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_r2t = tiled_copy_r2t.get_slice(tidx)

        tTR_rAcc_final_ = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rAcc_final = cute.group_modes(tTR_rAcc_final_, 3, cute.rank(tTR_rAcc_final_))

        tRT_gC = thr_copy_r2t.partition_S(gC_mnl_epi)
        tRT_tAcc_final = thr_copy_r2t.partition_D(tAcc_final_epi)
        tRT_rAcc_final_ = cute.make_rmem_tensor(
            tRT_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype
        )
        tRT_rAcc_final = cute.group_modes(tRT_rAcc_final_, 3, cute.rank(tRT_rAcc_final_))

        return (
            tiled_copy_t2r,
            tiled_copy_r2t,
            tTR_tAcc,
            tTR_rAcc,
            tTR_rAcc_final,
            tTR_sSFA,
            tTR_sSFB,
            tRT_rAcc_final,
            tRT_tAcc_final,
        )

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: cutlass.Boolean | bool,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """TMEM load tiled copy and partitions for the epilogue warps."""
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Register -> SMEM tiled copy and partitions for the epilogue warps."""
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tma_atom_c: cute.CopyAtom,
        tCgC_tile: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """SMEM -> GMEM TMA store partitions of one (MMA, MMA_M, MMA_N) C tile."""
        # (EPI_M, EPI_N, restM, restN)
        gC_epi = cute.flat_divide(tCgC_tile[((None, None), 0, 0)], epi_tile)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(gC_epi, 0, 2),
        )
        return bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sfa_dtype: type[cutlass.Numeric],
        sfb_dtype: type[cutlass.Numeric],
        sfa_count: int,
        sfb_count: int,
        num_smem_capacity: int,
        occupancy: int,
        mining_bytes: int | None = None,
    ) -> tuple[int, int, int, int]:
        """Stage counts for the A/B, C, scale and tile-info pipelines.

        ``mining_bytes`` (mining variants) replaces the scale stages' bytes.
        """
        num_c_stage = 2
        num_scale_stage = 10 if mining_bytes is None else 0
        num_tile_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        # mbarriers, tile info, tmem holding buffer
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        if mining_bytes is None:
            sfa_bytes = sfa_count * (sfa_dtype.width // 8) * num_scale_stage
            sfb_bytes = sfb_count * (sfb_dtype.width // 8) * num_scale_stage
            scale_bytes = math.ceil((sfa_bytes + sfb_bytes) / 1024) * 1024
        else:
            scale_bytes = math.ceil(mining_bytes / 1024) * 1024

        num_ab_stage = (
            num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes + scale_bytes)
        ) // ab_bytes_per_stage
        if mining_bytes is not None:
            # The mining variant's peel operand stage would otherwise
            # cost the mainloop an A/B stage; a single C staging buffer keeps
            # the plain full-K kernel's pipeline depth (the store epilogue has
            # the whole mainloop to drain it).
            num_ab_stage_one_c = (
                num_smem_capacity // occupancy
                - (mbar_helpers_bytes + c_bytes_per_stage + scale_bytes)
            ) // ab_bytes_per_stage
            if num_ab_stage_one_c > num_ab_stage:
                num_c_stage = 1
                c_bytes = c_bytes_per_stage
                num_ab_stage = num_ab_stage_one_c

        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes + scale_bytes)
        ) // (occupancy * c_bytes_per_stage)
        return num_ab_stage, num_c_stage, num_scale_stage, num_tile_stage

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> cpasync.CopyBulkTensorTileG2SMulticastOp | cpasync.CopyBulkTensorTileG2SOp:
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")
