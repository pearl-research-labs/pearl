"""SM120 MoE mining grouped GEMM: the fused SM120 kernel scheduled over experts.

``_GroupedFusedGemmSm120`` is ``mixed_gemm``'s ``_FusedGemmSm120`` compiled in
its grouped mode (see that module's docstring): the same warp roles, fold,
hash, peel and unscale, with every work tile resolved to one expert by
``_GroupedTileScheduler`` and the expert-local lottery of
``grouped_mixed_gemm``. It overrides the dense kernel's per-tile hooks with
the expert's operand views, scales and lottery bounds. The launch signature
is ``GroupedGemmSm100.mine``'s, so both families share the host.
"""

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import quack.copy_utils as copy_utils
from cutlass import Boolean, Float32, Int32
from quack.tile_scheduler import (
    PersistenceMode,
    RasterOrderOption,
    VarlenMTileScheduler,
    VarlenMTileSchedulerArguments,
)

from ..mixed_gemm._kernel import _HitGroup
from ..mixed_gemm._kernel_sm120 import (
    _RASTER_GROUP_SIZE,
    LTILE_ROWS,
    WARP_ROWS,
    _FusedGemmSm120,
    _Grouping,
    _TileGroup,
)


@cute.jit
def _group_rows(m_indptr: cute.Tensor, group: Int32, cum_m: Int32):
    """Expert ``group``'s ``(row0, row1)`` clamped into ``[0, cum_m]`` with
    ``row1 >= row0``: the host never reads ``m_indptr``, so a malformed table
    must still keep every expert inside the operands. ``cum_m`` fits Int32
    (the host bounds it by ``MAX_GROUPED_ROWS``)."""
    cum_m = Int32(cum_m)
    row0 = cutlass.min(cutlass.max(m_indptr[group], Int32(0)), cum_m)
    row1 = cutlass.min(cutlass.max(m_indptr[group + 1], row0), cum_m)
    return row0, row1


class _GroupedTileScheduler(VarlenMTileScheduler):
    """Quack's varlen-M static schedule over experts, with expert rows clamped
    (``_group_rows``) because the host never reads ``m_indptr``. The work
    tile's L coordinate is the expert."""

    @cute.jit
    def _get_num_m_blocks(
        self, lane: Int32, bidb_start: Int32, block_size: cutlass.Constexpr[int]
    ) -> Int32:
        """M tiles of expert ``lane + bidb_start`` (the last lane only closes
        the window, as in the base scan)."""
        num_groups = self.params.problem_shape_ncluster_mnl[2]
        group = lane + bidb_start
        m_blocks = Int32(0)
        if Boolean(group < num_groups) & Boolean(lane < cute.arch.WARP_SIZE - 1):
            row0, row1 = _group_rows(self.params.cu_seqlens_m, group, self.params.total_m)
            m_blocks = cute.ceil_div(row1 - row0, block_size)
        return m_blocks


class _GroupedFusedGemmSm120(_FusedGemmSm120):
    """The SM120 fused mining GEMM over MoE experts (``grouped_mixed_gemm``)."""

    grouped = True
    scheduler_cls = _GroupedTileScheduler

    def _tile_scheduler_params(self, grouping: _Grouping):
        tile_sched_args = VarlenMTileSchedulerArguments(
            problem_shape_ntile_mnl=(
                None,
                cute.ceil_div(self.problem_n, self.tile_n),
                cute.size(grouping.m_valid),
            ),
            total_m=self.problem_m,
            cu_seqlens_m=grouping.m_indptr,
            raster_order=RasterOrderOption.Heuristic,
            group_size=Int32(_RASTER_GROUP_SIZE),
            tile_shape_mn=(self.tile_m, self.tile_n),
            cluster_shape_mnk=self.cluster_shape_mnk,
            persistence_mode=PersistenceMode.STATIC,
        )
        return _GroupedTileScheduler.to_underlying_arguments(tile_sched_args)

    def _launch_row_operand(self, tensor: cute.Tensor, *, store: bool = False):
        """One launch-wide ragged TMA view: an expert is a coordinate offset
        whose rows past its block fall out of the descriptor (loads zero-fill,
        stores clip). Loads keep the pointer (wraparound form); the store may
        shift the base."""
        return copy_utils.create_ragged_tensor_for_tma(tensor, ragged_dim=0, ptr_shift=store)

    @cute.jit
    def _tile_group(self, grouping: _Grouping, tile_coord_mnkl) -> _TileGroup:
        """The expert a work tile belongs to (its L coordinate), with
        ``m_valid`` clamped into ``[0, rows]``."""
        index = tile_coord_mnkl[3]
        row0, row1 = _group_rows(grouping.m_indptr, index, self.problem_m)
        rows = row1 - row0
        valid_rows = cutlass.min(cutlass.max(grouping.m_valid[index], Int32(0)), rows)
        return _TileGroup(index, row0, rows, valid_rows)

    def _row_operand(self, tensor: cute.Tensor, group: _TileGroup, *, store: bool = False):
        """Row ``r`` is permuted row ``row0 + r``, out of the TMA descriptor
        from ``rows``."""
        return copy_utils.offset_ragged_tensor(
            tensor, group.row0, group.rows, ragged_dim=0, ptr_shift=store
        )

    def _column_operand(self, tensor: cute.Tensor, group: _TileGroup):
        """The expert's rows of a stacked ``(E * n, ...)`` operand. Columns a
        CTA tile overhangs past ``n`` are computed on the next expert's (or
        zero-filled) rows and never stored or published."""
        return cute.domain_offset((group.index * self.problem_n, 0), tensor)

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
        group: _TileGroup,
    ):
        """``A_peel`` from the expert's first row uses the same ragged view as
        ``A'``: the last expert's partial CTA tile must stay inside the
        gathered peel allocation (``cum_m`` rows)."""
        return self._peel_tma_partitions(
            tma_atom_pa,
            copy_utils.offset_ragged_tensor(mPa, group.row0, group.rows, ragged_dim=0),
            sPa,
            tma_atom_pb,
            self._column_operand(mPb, group),
            sPb,
        )

    @cute.jit
    def _stage_tile_alphas(
        self,
        mAlA: cute.Tensor,
        mAlB: cute.Tensor,
        sAlA: cute.Tensor,
        sAlB: cute.Tensor,
        group: _TileGroup,
        tile_coord_mnkl,
        warp_idx: Int32,
        lane: Int32,
        tidx: Int32,
    ):
        """Stage the expert's column scales; its row scales start at an
        arbitrary, possibly odd, row that ``cp.async`` cannot address, so each
        lane loads its two into registers. Rows past the expert's block
        (never stored) read 1."""
        row_alpha = cute.make_rmem_tensor(2, Float32)
        for row_half in cutlass.range_constexpr(2):
            row = (
                tile_coord_mnkl[0] * self.tile_m + WARP_ROWS * warp_idx + (lane >> 2) + 8 * row_half
            )
            row_alpha[row_half] = Float32(1.0)
            if row < group.rows:
                row_alpha[row_half] = mAlA[group.row0 + row].to(Float32)
        expert_alpha_b = cute.make_tensor(
            mAlB.iterator + group.index * self.problem_n, cute.make_layout(self.problem_n)
        )
        self._stage_alpha_slices(((expert_alpha_b, sAlB, 1, self.tile_n),), tile_coord_mnkl, tidx)
        return row_alpha

    @cute.jit
    def _row_inverse_scales(
        self, sAlA: cute.Tensor, row_alpha: cute.Tensor, warp_idx: Int32, lane: Int32
    ):
        self._wait_alpha_slices()
        row_inverse = cute.make_rmem_tensor(2, Float32)
        for row_half in cutlass.range_constexpr(2):
            row_inverse[row_half] = cute.arch.rcp_approx(row_alpha[row_half])
        return row_inverse

    @cute.jit
    def _lottery_tile_bounds(self, tile_row: Int32, tile_column: Int32, group: _TileGroup):
        """The lattice is expert-local: only lottery tiles wholly inside the
        expert's real rows (and ``n``) publish, with group semantics."""
        in_bounds = Boolean((tile_row + 1) * LTILE_ROWS <= group.valid_rows) & Boolean(
            (tile_column + 1) * self.ltile_cols <= self.problem_n
        )
        return in_bounds, _HitGroup(group.index, group.row0, group.rows)

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
        stream: cuda_driver.CUstream,
    ):
        """Launch over ``grouped_mixed_gemm``'s operands (see
        ``GroupedGemmSm100.mine`` for each argument)."""
        assert sched_counter is None and group_order is None, (
            "SM120 runs the static schedule in expert order"
        )
        self.__call__(
            a,
            b,
            a_peel,
            b_peel,
            alpha_a,
            inv_alpha_b,
            pow_key,
            threshold,
            c,
            record,
            lock,
            hash_b,
            codes_src,
            scales_src,
            codes_dst,
            scales_dst,
            layer_id,
            record_hits,
            max_active_clusters,
            stream,
            _Grouping(
                m_indptr,
                m_valid,
                Int32(cute.size(a, mode=[0])),
                n,
                Int32(cute.size(a, mode=[1])),
            ),
        )
