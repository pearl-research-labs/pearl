"""Persistent SM120 FP8 GEMM with lottery, peel, unscale, and hit publishing.

Built on Quack's ``GemmSm120``: a TMA-fed ``mma.sync`` GEMM whose accumulator
lives in the mma warps' registers. The warp topology, launch, producer
warpgroup, and hashing are shared with SM90 (``_kernel_register_acc``); this
module supplies the mma warps: they run the FP8 mainloop, fold their
pre-peel accumulator registers into lottery words in shared memory, issue
the BF16 peel ``mma.sync`` into the same registers, and run the TMA-store
epilogue with the row/column unscale fused into the subtile loads.

Every mma warp owns one 16-row slab spanning the whole ``tile_n``
(``atom_layout_mnk = (tile_m / 16, 1, 1)``), so a lane holds complete
lottery words: rows ``T >> 2`` and ``(T >> 2) + 8`` of its slab at the
mod-8 column pair ``2 * (T & 3)``. That is the 4-row lottery family only;
the 16-row family's whole-row words are split across four lanes.

``grouped_mixed_gemm``'s ``_GroupedFusedGemmSm120`` is the MoE variant of
this kernel; it overrides the per-tile hooks of ``_kernel_register_acc``.
"""

from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as blackwell_helpers
import quack.copy_utils as copy_utils
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import warp
from cutlass.utils import blockscaled_layout
from quack import sm80_utils
from quack.gemm_base import NamedBarrierGemm
from quack.gemm_sm120 import GemmSm120, _sf_group_vmk
from quack.pipeline import make_pipeline_state
from quack.rounding import RoundingMode

from ._kernel_register_acc import (
    _MAX_MSGS_WG,
    LTILE_ROWS,
    WG_THREADS,
    _Grouping,
    _RegisterAccFusedGemm,
)
from ._lottery import DEFAULT_LTILE_COLS, SUPPORTED_LTILE_COLS

WARP_ROWS = 16  # accumulator rows per mma warp (one m16n8 atom row block)
WG_ROWS = 4 * WARP_ROWS  # rows per mma warpgroup, the unit a compression warp hashes
_MMA_INST_N = 8  # mma.sync m16n8 atom columns
_SF_BLOCK_N = 128  # N granularity of the block-scaled MMA's scale-factor smem atoms


class _FusedGemmSm120(_RegisterAccFusedGemm, GemmSm120):
    """Quack's SM120 GEMM with lottery, peel, and unscale fused in."""

    # GemmSm120._setup_attributes inspects the epilogue-op tuple; this kernel has none.
    _epi_ops = ()
    fragment_n_block = _MMA_INST_N

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
        # One 64-row M-atom per mma warpgroup.
        self.mma_m_per_wg = 1
        assert self.msgs_wg <= _MAX_MSGS_WG, f"msgs_wg={self.msgs_wg} > {_MAX_MSGS_WG}"
        self.snapshot_payload = snapshot_payload
        # The producer warpgroup holds BLAKE3 state in its compression warps
        # (the stock 40-register budget spills it); the mma warps hold the
        # accumulator plus two in-flight B k-blocks.
        self.num_regs_load, self.num_regs_mma = (
            (56, 224) if self.mma_warp_groups == 2 else (80, 232)
        )
        # GemmTmaBase.epilogue reads this required specialization attribute.
        self.rounding_mode = RoundingMode.RN

    def _ab_tma_multicast(self) -> tuple[None, None]:
        return None, None

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

    def _check_accumulator_layout(self, tiled_mma: cute.TiledMma) -> None:
        self._check_fragment_ownership(tiled_mma)
        assert self.epi_tile[0] == self.tile_m and self.epi_tile[1] % _MMA_INST_N == 0

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

    def _message_lottery_row(self, tile_m_idx, row_message, warp_group_idx):
        """Problem lottery row of ``_stage_lottery_messages``' row message
        index: mma warpgroups own consecutive 64-row slabs."""
        return (
            tile_m_idx * (self.tile_m // LTILE_ROWS)
            + warp_group_idx * (WG_ROWS // LTILE_ROWS)
            + row_message
        )

    @cute.jit
    def _lottery_fragment_index(self, mma_row, row_half, column, n_block):
        return ((column, row_half), 0, n_block)

    @cute.jit
    def _row_inverse_scales(
        self, sAlA: cute.Tensor, row_alpha: cute.Tensor | None, warp_idx: Int32, lane: Int32
    ):
        """Wait for the staged scales and load this lane's two row reciprocals."""
        self._wait_alpha_slices()
        row_inverse = cute.make_rmem_tensor(2, Float32)
        for row_half in cutlass.range_constexpr(2):
            row = WARP_ROWS * warp_idx + (lane >> 2) + 8 * row_half
            row_inverse[row_half] = cute.arch.rcp_approx(sAlA[row].to(Float32))
        return row_inverse

    @cute.jit
    def epi_load_acc_subtile_unscaled(
        self,
        acc: cute.Tensor,
        row_inverse: cute.Tensor,
        sAlB: cute.Tensor,
        lane: Int32,
        tRS_rAcc: cute.Tensor,
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
    def _consumer_warps(
        self,
        warp_idx: Int32,
        tiled_mma: cute.TiledMma,
        tiled_mma_peel: cute.TiledMma,
        ab_pipeline,
        TileSchedulerCreate,
        tma_atom_d: cute.CopyAtom,
        mD: cute.Tensor,
        mAlA: cute.Tensor,
        mAlB: cute.Tensor,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sD: cute.Tensor,
        sPa: cute.Tensor,
        sPb: cute.Tensor,
        sAlA: cute.Tensor,
        sAlB: cute.Tensor,
        sExt: cute.Tensor,
        k_tile_cnt: Int32,
        grouping: _Grouping | None,
    ):
        """The mma warps: FP8 mainloop, fold, peel, and epilogue."""
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
            group = self._tile_group(grouping, tile_coord_mnkl)
            # Stage this tile's alpha slices while the mainloop runs.
            row_alpha = self._stage_tile_alphas(
                mAlA, mAlB, sAlA, sAlB, group, tile_coord_mnkl, warp_idx, lane, tidx
            )
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
            row_inverse = self._row_inverse_scales(sAlA, row_alpha, warp_idx, lane)
            self._store_epilogue(
                tiled_mma,
                tma_atom_d,
                self._row_operand(mD, group, store=True),
                sD,
                acc,
                partial(self.epi_load_acc_subtile_unscaled, acc, row_inverse, sAlB, lane),
                epi_store_pipeline,
                tile_scheduler,
                tile_coord_mnkl,
                tidx,
                is_tma_warp,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        # Wait for D store complete.
        if is_tma_warp:
            epi_store_pipeline.producer_tail()
