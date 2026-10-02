"""Persistent SM90 FP8 GEMM with lottery, peel, unscale, and hit publishing.

Built on Quack's ``GemmSm90`` and the register-accumulator phases shared
with SM120 (``_kernel_register_acc``). Consumer warpgroups run the promoted
FP8 WGMMA mainloop (below), fold the pre-peel register accumulators into
shared lottery messages, issue the BF16 peel WGMMA, and apply row/column
unscale in the TMA-store epilogue. Producer-side compression warps hash and
publish while those phases run.

The lottery-critical accumulation is the Hopper device model's (Pearl
whitepaper, Appendix "E4M3FN matrix multiplication"; the verifier's ``H100``
replay in ``zk-pow/src/api/fp8/utils.rs``): each 128-term k-tile is one
promotion window, accumulated from +0 and added to the FP32 total with one
RNE rounding. ``_promoted_mainloop`` reuses one 64-row window across the
warpgroup's M-atoms, so a thread holds the FP32 total plus a single window
of accumulator registers (``_accumulator_registers``, at most
``_MAX_ACC_REGS``).

Only 4-row lottery tiles are supported because WGMMA splits each accumulator
row across four lanes.
"""

from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import quack.sm90_utils as gemm_sm90_utils
from cutlass import Boolean, Float32, Int32
from cutlass.cute.nvgpu import warpgroup
from quack.gemm_sm90 import GemmSm90
from quack.pipeline import make_pipeline_state
from quack.rounding import RoundingMode

from ..protocol_constants import SM90_PROMOTE_K
from ._kernel_register_acc import (
    _MAX_MSGS_WG,
    LTILE_ROWS,
    WG_THREADS,
    _Grouping,
    _RegisterAccFusedGemm,
)
from ._lottery import DEFAULT_LTILE_COLS, SUPPORTED_LTILE_COLS

# Accumulator registers per thread: the FP32 total plus one 64-row promotion
# window (``_promoted_mainloop``). The rest of the consumer's setmaxnreg
# budget holds the inverse scales, the epilogue fragments, and addressing.
_MAX_ACC_REGS = 192
WGMMA_M = 64  # the only Hopper WGMMA M; one window is one M-atom
_WGMMA_N_BLOCK = 8  # accumulator columns per (c, h) fragment block
# WGMMA N widths the fused epilogue and the accumulator budget admit.
SM90_TILE_NS = (128, 192)


def _accumulator_registers(tile_m: int, tile_n: int, atom_m: int) -> int:
    return (WGMMA_M + tile_m // atom_m) * tile_n // WG_THREADS


def _insert_unit_mode(tensor: cute.Tensor) -> cute.Tensor:
    """View a rank-2 ``(V, X)`` tensor as ``(V, 1, X)`` over the same storage."""
    v_shape, x_shape = tensor.layout.shape
    v_stride, x_stride = tensor.layout.stride
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout((v_shape, 1, x_shape), stride=(v_stride, 0, x_stride)),
    )


class _FusedGemmSm90(_RegisterAccFusedGemm, GemmSm90):
    """Quack's SM90 GEMM with lottery, peel, and unscale fused in."""

    fragment_n_block = _WGMMA_N_BLOCK

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int | None = None,
        cluster_m: int = 1,
        cluster_n: int = 1,
        *,
        snapshot_payload: bool = False,
        ltile_rows: int = LTILE_ROWS,
        ltile_cols: int = DEFAULT_LTILE_COLS,
    ):
        # GemmSm90 requires these fixed substrate choices as constructor
        # arguments; they are implementation constraints, not caller tuning.
        # The k-tile is the reference's promotion window (SM90_PROMOTE_K
        # terms); ``_promoted_mainloop`` replaces Quack's mainloop, and the
        # register split below replaces Quack's heuristic.
        assert tile_k in (None, SM90_PROMOTE_K), (
            f"SM90 tile_k must be the {SM90_PROMOTE_K}-term promotion window, got {tile_k}"
        )
        tile_k = SM90_PROMOTE_K
        assert tile_n in SM90_TILE_NS, f"SM90 tile_n must be one of {SM90_TILE_NS}, got {tile_n}"
        super().__init__(
            acc_dtype=Float32,
            a_dtype=cutlass.Float8E4M3FN,
            tile_shape_mnk=(tile_m, tile_n, tile_k),
            cluster_shape_mnk=(cluster_m, cluster_n, 1),
            pingpong=False,
            is_persistent=True,
            fp8_fast_accum=True,
            use_pdl=False,  # compression warps read commit-stage outputs at kernel start
        )
        assert self.atom_layout_mnk[1] == 1, (
            "lottery fold needs an unpermuted N mode (atom_layout_n == 1); "
            f"tile_m={tile_m}/tile_n={tile_n} implies atom_layout_n={self.atom_layout_mnk[1]}"
        )
        # At most two consumer warpgroups: the per-warpgroup barriers stop at
        # ``_NamedBarrier.*_WG1``, and a third warpgroup (atom_m == 3) has no
        # register split within ``_MAX_CTA_REGS``.
        assert self.mma_warp_groups <= 2, (
            f"SM90 fused GEMM supports at most 2 consumer warpgroups, "
            f"got {self.mma_warp_groups} (tile_m={tile_m}, tile_n={tile_n})"
        )
        self.tile_m, self.tile_n = tile_m, tile_n
        self.atom_m = self.atom_layout_mnk[0]
        self.mma_m_per_wg = tile_m // (WGMMA_M * self.atom_m)
        # WGMMA fragments split accumulator rows across lanes: only the
        # 4-row family's mod-8 column pairs are foldable thread-locally.
        assert ltile_rows == LTILE_ROWS, (
            f"the SM90 lottery implements only the {LTILE_ROWS}-row tile family, got {ltile_rows}"
        )
        self.ltile_rows = LTILE_ROWS
        # Merged lottery tile: 4 x ltile_cols, 16 thread-local subtiles of
        # 1 x (ltile_cols/4). The job commits the width (``select_tile``
        # prefers the narrowest); the kernel accepts every supported width.
        assert ltile_cols in SUPPORTED_LTILE_COLS[LTILE_ROWS], (
            f"unsupported lottery tile cols: {ltile_cols}"
        )
        self.ltile_cols = ltile_cols
        assert tile_n % ltile_cols == 0, f"lottery needs tile_n % {ltile_cols} == 0"
        self.column_tiles = tile_n // ltile_cols
        self.msgs_wg = (WGMMA_M // LTILE_ROWS) * self.column_tiles * self.mma_m_per_wg
        assert self.msgs_wg <= _MAX_MSGS_WG, f"msgs_wg={self.msgs_wg} > {_MAX_MSGS_WG}"
        self.snapshot_payload = snapshot_payload
        acc_regs = _accumulator_registers(tile_m, tile_n, self.atom_m)
        assert acc_regs <= _MAX_ACC_REGS, (
            f"tile {tile_m}x{tile_n} needs {acc_regs} accumulator registers per thread "
            f"(> {_MAX_ACC_REGS}) for the FP32 total plus the 64-row promotion window"
        )
        # Legal tiles need 128 or 192 accumulator registers. The producer
        # warpgroup's BLAKE3 state takes what the consumers leave (hash
        # spills are off the critical path).
        self.num_regs_load, self.num_regs_mma = (80, 208) if acc_regs <= 128 else (24, 240)
        # GemmTmaBase.epilogue reads this required specialization attribute.
        self.rounding_mode = RoundingMode.RN

    def _ab_tma_multicast(self) -> tuple[dict, dict]:
        cluster_shape = self.cluster_shape_mnk[:2]
        return (
            {"cluster_shape": cluster_shape, "multicast_dim": "M"},
            {"cluster_shape": cluster_shape, "multicast_dim": "N"},
        )

    def _make_tiled_mma_peel(self) -> cute.TiledMma:
        """The BF16 peel MMA with the same atom layout and tiler as the FP8
        mainloop, so it accumulates into the identical register fragment."""
        return sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            self.atom_layout_mnk,
            tiler_mn=(WGMMA_M, self.tile_n),
        )

    def _check_accumulator_layout(self, tiled_mma: cute.TiledMma) -> None:
        """Assert the lane -> accumulator cell map the fold, the inverse-scale
        loads, and the unscale index by.

        Thread ``T`` of warp ``w`` in consumer warpgroup ``g`` holds rows
        ``64 (mi atom_m + g) + 16 w + (T >> 2) + 8 h`` and columns
        ``8 nb + 2 (T & 3) + c`` at fragment index ``((c, h, nb), mi, 0)``.
        Derived from the tiled MMA's static C thread-value layout, so a Quack
        or CUTLASS layout change fails at compile time, not silently.
        """
        n_blocks = self.tile_n // _WGMMA_N_BLOCK
        assert tiled_mma.partition_shape_C((self.tile_m, self.tile_n)) == (
            (2, 2, n_blocks),
            self.mma_m_per_wg,
            1,
        ), "fragment shape"
        layout_tv = tiled_mma.tv_layout_C_tiled
        m_size = WGMMA_M * self.atom_m
        for thread in range(cute.size(layout_tv, mode=[0])):
            warp_group, warp_thread = divmod(thread, WG_THREADS)
            warp, lane = divmod(warp_thread, cute.arch.WARP_SIZE)
            for value in range(cute.size(layout_tv, mode=[1])):
                flat = layout_tv((thread, value))
                row, column = flat % m_size, flat // m_size
                c, h, nb = value % 2, (value // 2) % 2, value // 4
                assert row == WGMMA_M * warp_group + 16 * warp + (lane >> 2) + 8 * h, (
                    "fragment row map"
                )
                assert column == _WGMMA_N_BLOCK * nb + 2 * (lane & 3) + c, "fragment column map"
        assert self.epi_tile[0] % m_size == 0 and self.epi_tile[1] % _WGMMA_N_BLOCK == 0, (
            f"epi tile {self.epi_tile} must cover whole {m_size}x{_WGMMA_N_BLOCK} fragment blocks"
        )

    def _message_lottery_row(self, tile_m_idx, row_message, warp_group_idx):
        """Problem lottery row of ``_stage_lottery_messages``' row message
        index: consumer warpgroups own interleaved 64-row slabs."""
        mma_row = row_message % self.mma_m_per_wg
        thread_row = row_message // self.mma_m_per_wg
        return (
            tile_m_idx * (self.tile_m // LTILE_ROWS)
            + (mma_row * self.atom_m + warp_group_idx) * (WGMMA_M // LTILE_ROWS)
            + thread_row
        )

    @cute.jit
    def _promoted_mainloop(
        self,
        ab_pipeline,
        ab_read_state,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        k_tile_cnt: Int32,
    ):
        """The reference's promoted FP8 accumulation, waved per WGMMA M-atom.

        One k-tile is one 128-term promotion window. For each M-atom ``mb``,
        four K = 32 WGMMAs accumulate from +0 into the shared ``window``, the
        group drains with ``wait_group(0)``, and ``acc[mb] += window`` is the
        reference's single RNE add per window.
        """
        # ``window2`` is one M-atom of registers viewed (V, MMA_N) for the
        # promotion add; ``window`` is the same registers as the (V, 1, MMA_N)
        # accumulator the (V,M) x (V,N) => (V,M,N) gemm dispatch takes.
        window2 = cute.make_fragment_like(acc[None, 0, None])
        window = _insert_unit_mode(window2)
        # The FP32 total starts at +0 (also covers k_tile_cnt == 0).
        acc.fill(0.0)

        peek_ab_full_status = Boolean(True)
        if k_tile_cnt > 0:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            stage = ab_read_state.index
            for mb in cutlass.range_constexpr(cute.size(acc, mode=[1])):
                gemm_sm90_utils.gemm(
                    tiled_mma,
                    window,
                    _insert_unit_mode(tCrA[None, mb, None, stage]),  # (V, 1, MMA_K)
                    tCrB[None, None, None, stage],  # (V, MMA_N, MMA_K)
                    zero_init=True,
                    wg_wait=0,
                )
                # Promotion: C <- RNE_FP32(C + c), one rounding per window.
                total = acc[None, mb, None]
                total.store(total.load() + window2.load())
            ab_pipeline.consumer_release(ab_read_state)
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        return ab_read_state

    @cute.jit
    def _lottery_fragment_index(self, mma_row, row_half, column, n_block):
        return ((column, row_half, n_block), mma_row, 0)

    def _fragment_row(self, mma_row, row_half, warp_group_idx, warp_in_wg, lane):
        """Tile row of accumulator fragment ``(mma_row, row_half)``
        (``_check_accumulator_layout``)."""
        return (
            (mma_row * self.atom_m + warp_group_idx) * WGMMA_M
            + warp_in_wg * 16
            + (lane >> 2)
            + 8 * row_half
        )

    @cute.jit
    def _row_inverse_scales(
        self,
        sAlA: cute.Tensor,
        row_alpha: cute.Tensor | None,
        warp_group_idx: Int32,
        warp_in_wg: Int32,
        lane: Int32,
    ):
        """This thread's row reciprocals from the staged row scales."""
        row_inverse = cute.make_rmem_tensor((self.mma_m_per_wg, 2), Float32)
        for mma_row in cutlass.range_constexpr(self.mma_m_per_wg):
            for row_half in cutlass.range_constexpr(2):
                row = self._fragment_row(mma_row, row_half, warp_group_idx, warp_in_wg, lane)
                row_inverse[mma_row, row_half] = cute.arch.rcp_approx(sAlA[row].to(Float32))
        return row_inverse

    @cute.jit
    def _prepare_unscale_factors(
        self,
        sAlA: cute.Tensor,
        sAlB: cute.Tensor,
        row_alpha: cute.Tensor | None,
        warp_group_idx: Int32,
        warp_in_wg: Int32,
        lane: Int32,
    ):
        """Wait for staged scales and load per-thread inverse factors."""
        self._wait_alpha_slices()
        row_inverse = self._row_inverse_scales(sAlA, row_alpha, warp_group_idx, warp_in_wg, lane)
        n_blocks = self.tile_n // _WGMMA_N_BLOCK
        column_inverse = cute.make_rmem_tensor((n_blocks, 2), Float32)
        for column_block in cutlass.range_constexpr(n_blocks):
            for column in cutlass.range_constexpr(2):
                global_column = _WGMMA_N_BLOCK * column_block + 2 * (lane & 3) + column
                column_inverse[column_block, column] = sAlB[global_column]
        return row_inverse, column_inverse

    # -- unscale fused into the epilogue subtile loop --
    @cute.jit
    def epi_load_acc_subtile_unscaled(
        self,
        acc: cute.Tensor,
        rinv: cute.Tensor,
        cinv: cute.Tensor,
        tRS_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        epi_coord,  # (epi_m, epi_n), constexpr
    ):
        """Multiply this subtile's acc slice by the alpha reciprocals, then load.

        tRS_rAcc is a register view over ``acc``, so scaling acc right before
        the subtile's copy keeps the multiplies inside the pipelined epilogue
        loop (overlapping earlier subtiles' STSM/TMA stores). Each (mi, nb)
        belongs to exactly one epi tile (``_check_accumulator_layout``), so
        every element is scaled once.
        """
        mis = self.epi_tile[0] // (WGMMA_M * self.atom_m)
        nbs = self.epi_tile[1] // _WGMMA_N_BLOCK
        for mo in cutlass.range_constexpr(mis):
            mi = epi_coord[0] * mis + mo
            for no in cutlass.range_constexpr(nbs):
                nb = epi_coord[1] * nbs + no
                for h in cutlass.range_constexpr(2):
                    for c in cutlass.range_constexpr(2):
                        acc[((c, h, nb), mi, 0)] = (
                            acc[((c, h, nb), mi, 0)] * rinv[mi, h] * cinv[nb, c]
                        )
        cute.autovec_copy(tRS_rAcc[(None, None, None, epi_coord)], tRS_rD)

    @cute.jit
    def _issue_peel(
        self,
        ab_pipeline,
        ab_read_state,
        tiled_mma_peel: cute.TiledMma,
        accumulator: cute.Tensor,
        tCrPa: cute.Tensor,
        tCrPb: cute.Tensor,
    ):
        """Issue the peel WGMMA from this tile's final A/B ring slot."""
        ab_pipeline.consumer_wait(
            ab_read_state,
            ab_pipeline.consumer_try_wait(ab_read_state),
        )
        gemm_sm90_utils.gemm_w_idx(
            tiled_mma_peel,
            accumulator,
            tCrPa,
            tCrPb,
            zero_init=Boolean(False),
            A_idx=ab_read_state.index,
            B_idx=ab_read_state.index,
            wg_wait=-1,
        )

    @cute.jit
    def _drain_peel(self, ab_pipeline, ab_read_state):
        """Drain the peel WGMMA and release its A/B ring slot."""
        warpgroup.wait_group(0)
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
        """Consumer warpgroups: promoted mainloop, fold, peel, and epilogue."""
        is_tma_warp = Boolean(warp_idx == 0)
        tidx, _, _ = cute.arch.thread_idx()
        warp_group_idx = cute.arch.make_warp_uniform(tidx // WG_THREADS)
        warp_in_wg = cute.arch.make_warp_uniform((tidx % WG_THREADS) // cute.arch.WARP_SIZE)
        lane = tidx % cute.arch.WARP_SIZE
        warp_group_thread_layout = cute.make_layout(self.mma_warp_groups, stride=WG_THREADS)
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        # ``acc`` is the FP32 total the fold/peel/epilogue phases read.
        acc, tCrA, tCrB = gemm_sm90_utils.partition_fragment_ABC(
            thr_mma, self.cta_tile_shape_mnk, sA, sB
        )

        # Both peel operands are smem-descriptor fragments over the BF16
        # ring-slot aliases (stage mode selected by B_idx/A_idx at issue
        # time), exactly like the mainloop's tCrA/tCrB.
        thr_mma_peel = tiled_mma_peel.get_slice(warp_group_thread_layout(warp_group_idx))
        tCrPa = thr_mma_peel.make_fragment_A(thr_mma_peel.partition_A(sPa))
        tCrPb = thr_mma_peel.make_fragment_B(thr_mma_peel.partition_B(sPb))

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
            ab_read_state = self._promoted_mainloop(
                ab_pipeline, ab_read_state, tiled_mma, acc, tCrA, tCrB, k_tile_cnt
            )

            # The fold observes the accumulator before peel.
            self._stage_lottery_messages(acc, sExt, warp_group_idx, warp_in_wg, lane)

            self._issue_peel(ab_pipeline, ab_read_state, tiled_mma_peel, acc, tCrPa, tCrPb)
            # The scale loads hide the peel WGMMA latency; unscale stays
            # fused into the epilogue subtile loop.
            rinv, cinv = self._prepare_unscale_factors(
                sAlA, sAlB, row_alpha, warp_group_idx, warp_in_wg, lane
            )
            ab_read_state = self._drain_peel(ab_pipeline, ab_read_state)
            self._store_epilogue(
                tiled_mma,
                tma_atom_d,
                self._row_operand(mD, group, store=True),
                sD,
                acc,
                partial(self.epi_load_acc_subtile_unscaled, acc, rinv, cinv),
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
