"""Fused activation stats, noising, quantization, and peel kernel.

Blackwell (SM100) implementation: the ``E1 @ F1`` noise dot and the
``A' @ F2^T`` peel both run on the real ``tcgen05.mma`` kind::f8f6f4 atom
(``MmaF8F6F4Op``).

- a dedicated MMA warp issues the noise UMMA
  ``noise^T (bk x rows) = F1_tile (bk x 32, e4m3) @ E1^T (rows x 32, e4m3)``
  per k-tile into a two-stage TMEM accumulator. K = 32 is the atom's fixed
  contraction depth; F1 arrives from ``pack_noise_factor`` already
  zero-padded to 32 columns and E1's operand tile is written zero-padded, so
  the atom computes the reference's zero-padded K = 32 product bit-exactly
  (it *is* the pinned arithmetic, accumulating from +0);
- consumers drain each accumulator stage through shared memory
  (``tcgen05.ld`` -> STS f32) and the quantize chain reads bf16x2 noise
  pairs with a single ``cvt.rn.bf16x2.f32`` rounding, so A' codes stay
  bit-identical to the reference;
- the peel UMMA consumes the staged A' tile (64-row operand buffer,
  pad rows zeroed once) against the staged F2 tile, accumulating the CTA's
  whole k-slice in TMEM.

The consumer C-fragment ownership (``tiled_mma_n``/``tiled_mma_u16``) is
layout scaffolding only -- those tiled MMAs issue no MMA instructions; the
only MMA atom the kernel executes is ``tcgen05.MmaF8F6F4Op``.
"""

import enum
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import SmemLayoutAtomKind, make_smem_layout_atom
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.blackwell_helpers import tile_to_mma_shape
from cutlass.utils.tmem_allocator import compute_tmem_cols_from_layout
from quack.cute_dsl_utils import mlir_namedtuple

from ..protocol_constants import BLOCK_SCALE_GROUP, PACKED_NOISE_K, PEEL_COLS, R
from ._quantization_ops import (
    _bf16x2_to_e4m3x2,
    _f32x2_to_bf16x2,
    _fma_bf16x2,
    _generate_noise_line,
    _mul_bf16x2,
    _open_int8x2_bf16x2,
    _open_int8x2_bf16x2_hi,
    _rndb,
    _scale_chain,
    _splat_bf16x2,
)

WG_THREADS = 128
HALF_TILE_ROWS = 16  # A is staged and consumed in 16-row half-tiles
_OUT_STAGES = 2  # staged A' output tiles feeding the TMA store warp
_SMEM_CAPACITY_BYTES = utils.get_smem_capacity_in_bytes("sm_100")


class NoiseLoadMode(enum.StrEnum):
    """How a CTA stages its A k-slice in shared memory.

    ``RESIDENT`` retains the full slice, ``RING`` loads each half-tile once
    through bounded stages alongside the factors, and ``MERGED`` shares each
    factor pipeline stage with one A half-tile per row half.
    """

    RESIDENT = "resident"
    RING = "ring"
    MERGED = "merged"


class _NamedBarrier(enum.IntEnum):
    CONSUMER = 3
    E1_READY = 4
    TMEM_PTR = 5


@mlir_namedtuple
class _ConsumerGlobals(NamedTuple):
    a_prime: cute.Tensor
    alpha: cute.Tensor
    beta: cute.Tensor
    peel: cute.Tensor
    stats: cute.Tensor


@mlir_namedtuple
class _ConsumerShared(NamedTuple):
    a_codes: cute.Tensor
    a_scales: cute.Tensor
    f1: cute.Tensor
    f2: cute.Tensor
    a_prime: cute.Tensor
    e1: cute.Tensor
    e1_pairs: cute.Tensor
    alpha: cute.Tensor
    beta: cute.Tensor


@mlir_namedtuple
class _ConsumerCoords(NamedTuple):
    thread: cutlass.Int32
    lane: cutlass.Int32
    row_block: cutlass.Int32
    tile_count: cutlass.Int32


class _NoisyQuant:
    """Warp-specialized noising at 16-row MMA granularity on the f8f6f4 UMMA atom."""

    def __init__(
        self,
        bk: int,
        stages: int,
        out_stages: int = _OUT_STAGES,
        msg_base: tuple = (),
        consts: tuple = (0.0, 0.0),
        load_mode: NoiseLoadMode = NoiseLoadMode.MERGED,
        rows: int = 16,
    ):
        # bk == 64 would need an M=64 noise UMMA whose interleaved TMEM
        # fragment none of the t2r drain paths cover.
        assert bk in (128, 256)
        assert stages >= 2
        assert out_stages >= 2
        assert msg_base
        assert rows in (16, 32, 64)
        assert isinstance(load_mode, NoiseLoadMode)
        self.load_mode = load_mode
        self.rows = rows
        self.bk = bk
        self.stages = stages
        self.out_stages = out_stages
        self.msg_base = msg_base
        self.consts = consts
        # MERGED tied A half-tiles to the factor pipeline's barriers; the
        # factors are consumed by the MMA warp alone, so A rides its own
        # ring with the merged stage budget.
        was_merged = self.load_mode == NoiseLoadMode.MERGED
        row_halves = self.rows // HALF_TILE_ROWS
        if was_merged:
            self.load_mode = NoiseLoadMode.RING
            self.ring_a_stages = self.stages * row_halves
        else:
            self.ring_a_stages = self.stages
        # Merged-mode configs use a tile-granular A handshake: one mbarrier
        # transaction covers a whole tile's half-tile TMA copies, so each
        # side pays one sleep/wake round trip per tile instead of one per
        # half-tile.
        # Native-RING pins keep the classic per-half-tile pipe.
        self.a_tile_pipe = was_merged
        self.a_pipe_stages = self.stages if self.a_tile_pipe else 0
        if self.a_tile_pipe:
            self.ring_a_stages = self.a_pipe_stages * row_halves
        # The peel UMMA consumes whole staged A' tiles, so there is no
        # 16-row early-drain staging.
        self.out_rows = self.rows
        self.bk_mma = min(self.bk, 128)
        self.nblk = self.bk // self.bk_mma
        self.direct = False
        self.acc_stages = 3 if self.direct else 2
        self.consumer_warps = 4
        self.consumer_threads = WG_THREADS
        # Columns per C-fragment N-permutation group (16 per consumer warp).
        self.n_group = 16 * self.consumer_warps
        self.producer_warp = self.consumer_warps
        self.output_warp = self.consumer_warps + 1
        self.mma_warp = self.consumer_warps + 2
        # Consumer-side E1 overlaps the BLAKE3 prologue with the stats combine.
        self.consumer_e1 = self.rows == 64

    # -- device helpers -----------------------------------------------------

    @cute.jit
    def _prefetch_tma_descriptors(self, tma_aq, tma_as, tma_f1, tma_f2, tma_q):
        """Prefetch all descriptors used by the producer and output warps."""
        cute.nvgpu.cpasync.prefetch_descriptor(tma_aq)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_as)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_f1)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_f2)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_q)

    @cute.jit
    def _copy_a_half_tile(
        self,
        tma_aq,
        tma_as,
        tAsAq,
        tAgAq,
        tAsAs,
        tAgAs,
        row_half,
        k_tile,
        shared_stage,
        barrier,
    ):
        """Stage one A half-tile: its code bytes and its group scales.

        Both copies land on the same barrier, so the consumer's single wait
        covers the whole 1.25 byte/element half-tile.
        """
        cute.copy(
            tma_aq,
            tAgAq[(None, row_half, k_tile)],
            tAsAq[(None, shared_stage)],
            tma_bar_ptr=barrier,
        )
        cute.copy(
            tma_as,
            tAgAs[(None, row_half, k_tile)],
            tAsAs[(None, shared_stage)],
            tma_bar_ptr=barrier,
        )

    @cute.jit
    def _wait_a_half_tile(
        self,
        a_pipe,
        tile_index,
        row_half,
        main_stage,
    ):
        """Wait for one A half-tile's stage and return its shared stage index."""
        row_halves = self.rows // HALF_TILE_ROWS
        half_tile_index = row_halves * tile_index + row_half
        i32 = cutlass.Int32

        if cutlass.const_expr(self.load_mode == NoiseLoadMode.MERGED):
            shared_stage = main_stage * row_halves + row_half
        elif cutlass.const_expr(self.load_mode == NoiseLoadMode.RESIDENT):
            shared_stage = i32(half_tile_index)
            a_pipe.consumer_wait(
                pipeline.PipelineState(
                    self.a_stage_count,
                    i32(half_tile_index),
                    i32(half_tile_index),
                    i32(0),
                )
            )
        else:
            shared_stage = i32(half_tile_index % self.a_stage_count)
            a_pipe.consumer_wait(
                pipeline.PipelineState(
                    self.a_stage_count,
                    i32(half_tile_index),
                    shared_stage,
                    i32((half_tile_index // self.a_stage_count) % 2),
                )
            )
        return shared_stage

    @cute.jit
    def _release_a_half_tile(self, a_pipe, tile_index, row_half):
        """Release one consumed A half-tile back to the ring producer."""
        if cutlass.const_expr(self.load_mode == NoiseLoadMode.RING):
            row_halves = self.rows // HALF_TILE_ROWS
            half_tile_index = row_halves * tile_index + row_half
            i32 = cutlass.Int32
            a_pipe.consumer_release(
                pipeline.PipelineState(
                    self.a_stage_count,
                    i32(half_tile_index),
                    i32(half_tile_index % self.a_stage_count),
                    i32((half_tile_index // self.a_stage_count) % 2),
                )
            )

    @cute.jit
    def _wait_a_tile(self, a_pipe, tile_index):
        """Wait for one whole A tile's stage (tile-granular pipe only)."""
        a_pipe.consumer_wait(
            pipeline.PipelineState(
                self.a_pipe_stages,
                cutlass.Int32(tile_index),
                cutlass.Int32(tile_index % self.a_pipe_stages),
                cutlass.Int32((tile_index // self.a_pipe_stages) % 2),
            )
        )

    @cute.jit
    def _release_a_tile(self, a_pipe, tile_index):
        """Release one consumed A tile (tile-granular pipe only)."""
        a_pipe.consumer_release(
            pipeline.PipelineState(
                self.a_pipe_stages,
                cutlass.Int32(tile_index),
                cutlass.Int32(tile_index % self.a_pipe_stages),
                cutlass.Int32((tile_index // self.a_pipe_stages) % 2),
            )
        )

    @cute.jit
    def _load_a_codes(
        self,
        a_pipe,
        tile_index,
        row_half,
        tiled_copy_aq,
        shared_code_partition,
        code_words,
    ):
        """Wait for one A half-tile and ldmatrix its codes; returns the stage."""
        if cutlass.const_expr(self.a_tile_pipe):
            row_halves = self.rows // HALF_TILE_ROWS
            stage = cutlass.Int32((tile_index % self.a_pipe_stages) * row_halves + row_half)
        else:
            stage = self._wait_a_half_tile(a_pipe, tile_index, row_half, 0)
        cute.copy(
            tiled_copy_aq,
            shared_code_partition[(None, None, None, stage)],
            tiled_copy_aq.retile(code_words),
        )
        return stage

    @cute.jit
    def _write_e1_peel_row(
        self,
        shared_e1,
        shared_beta,
        peel_output,
        thread_idx,
    ):
        """Write beta * E1 for one row using the reference rounding step."""
        beta_value = shared_beta[thread_idx]
        for column in cutlass.range_constexpr(R):
            e1_value = shared_e1[(thread_idx % 16, column, thread_idx // 16)].to(Float32)
            peel_output[thread_idx % 16, column] = _rndb(beta_value * e1_value).to(cutlass.BFloat16)

    @cute.jit
    def _combine_row_stats(
        self,
        stats: cute.Tensor,
        row: cutlass.Int32,
        row_width: cutlass.Int32,
    ):
        """Combine one row's commit partials and derive its scales.

        All (ssq, absmax) pairs are fetched into registers with independent
        wide loads first, then reduced in the pinned chunk order -- load
        order does not touch FP order, so alpha/beta stay bit-exact.
        """
        chunks = self.stats_chunks
        # A row's stats span 8 * chunks bytes, so 16B alignment (LDG.128)
        # holds whenever the chunk count is even.
        width = 4 if chunks % 2 == 0 else 2
        pairs = cute.make_rmem_tensor((width, chunks * 2 // width), Float32)
        src = cute.make_tensor(
            stats.iterator + row * (2 * chunks),
            cute.make_layout((width, chunks * 2 // width)),
        )
        for group in cutlass.range_constexpr(chunks * 2 // width):
            cute.autovec_copy(src[(None, group)], pairs[(None, group)])
        pairs = cute.make_tensor(pairs.iterator, cute.make_layout((2, chunks)))
        sum_squares = Float32(0.0)
        abs_max = Float32(0.0)
        for chunk in cutlass.range_constexpr(chunks):
            sum_squares = sum_squares + pairs[(0, chunk)]
            abs_max = cute.arch.fmax(abs_max, pairs[(1, chunk)])
        return _scale_chain(
            sum_squares,
            abs_max,
            Float32(row_width),
            self.consts[0],
            self.consts[1],
        )

    @cute.jit
    def _open_a_fragment(
        self,
        shared_scales,
        shared_stage,
        group_base,
        first_fragment_row,
        code_words,
        a_words,
    ):
        """Open one tile's A codes: both u16 pairs of each raw ldmatrix word
        share a scale splat and are opened in place by byte-pair prmt
        selectors, skipping the u16 extraction."""
        code_u32 = cute.recast_tensor(code_words, cutlass.Uint32)
        for repeat in cutlass.range_constexpr(self.bk // self.n_group):
            group = group_base + (self.n_group // 8) * repeat
            for row_half in cutlass.range_constexpr(2):
                splat = _splat_bf16x2(
                    cutlass.Uint32(
                        shared_scales[(first_fragment_row + 8 * row_half, group, shared_stage)]
                    )
                )
                word32 = code_u32[row_half + 2 * repeat]
                a_words[row_half + 2 * (0 + 2 * repeat)] = _open_int8x2_bf16x2(word32, splat)
                a_words[row_half + 2 * (1 + 2 * repeat)] = _open_int8x2_bf16x2_hi(word32, splat)

    @cute.jit
    def _quantize_fragment_tmem(
        self,
        noise_words: cute.Tensor,
        a_words: cute.Tensor,
        a_prime_words: cute.Tensor,
        alpha_pairs,
        beta_pairs,
    ):
        """The packed reference rounding chain, noise pre-packed as bf16x2.

        ``column_block`` runs over the permuted MMA's N instances
        (``2 * bk / n_group``); the word map ``(blk % 2) + 2h + 4*(blk // 2)``
        is the (d, g) -> u16-fragment decomposition (verified by trace-time
        coordinate prints).
        """
        for column_block in cutlass.range_constexpr(2 * self.bk // self.n_group):
            for row_half in cutlass.range_constexpr(2):
                pair = row_half + 2 * column_block
                word = (column_block % 2) + 2 * row_half + 4 * (column_block // 2)
                scaled = _fma_bf16x2(
                    alpha_pairs[row_half],
                    a_words[pair],
                    _mul_bf16x2(beta_pairs[row_half], noise_words[pair]),
                )
                a_prime_words[word] = _bf16x2_to_e4m3x2(scaled)

    @cute.jit
    def _generate_e1_rows(self, mE1, mKey, sE1, sE1B, first_row, lane, row_block):
        """Generate up to 32 E1 rows (one BLAKE3 row per lane) and stage them.

        The E1 prologue is cold-instruction-fetch bound, so the staging
        stores go through an rmem ``codes || zero-pad`` image and a few
        vectorized copies instead of R-unrolled scalar store loops.
        """
        f16 = cutlass.Float16
        f8 = cutlass.Float8E4M3FN
        e1_padded = cute.make_rmem_tensor(PACKED_NOISE_K, f8)
        e1_values = cute.make_rmem_tensor(R, f16)
        local_row = first_row + lane
        if local_row < self.rows:
            global_row = row_block * self.rows + local_row
            e1_codes = cute.make_tensor(e1_padded.iterator, cute.make_layout(R))
            e1_line = cute.make_rmem_tensor(R, Float32)
            _generate_noise_line(
                mKey,
                cutlass.Uint32(global_row),
                self.msg_base,
                e1_line,
            )
            e1_codes.store(e1_line.load().to(f8))
            # Pad the UMMA operand to the atom's K. Zero pad products are
            # exact zeros; skipped when R already fills the atom.
            if cutlass.const_expr(R < PACKED_NOISE_K):
                e1_zeros = cute.make_tensor(
                    e1_padded.iterator + R, cute.make_layout(PACKED_NOISE_K - R)
                )
                e1_zeros.fill(f8(0.0))
            e1_values.store(e1_codes.load().to(Float32).to(f16))  # exact hops
            cute.autovec_copy(
                e1_values,
                cute.make_tensor(
                    sE1.iterator + cute.crd2idx((local_row % 16, 0, local_row // 16), sE1.layout),
                    cute.make_layout(R),
                ),
            )
            cute.autovec_copy(
                e1_padded,
                sE1B[((local_row, None), 0, 0, 0)],
            )
            cute.autovec_copy(
                e1_codes,
                cute.local_tile(mE1, (R,), (global_row,)),
            )

    @cute.jit
    def _produce_factor_tile(
        self,
        tma_f1,
        tma_f2,
        tFsF1,
        tFgF1,
        tFsF2,
        tFgF2,
        factor_pipe,
        f2_pipe,
        factor_write_state,
        f2_write_state,
        k_tile,
    ):
        """Stage one tile's F1 and F2 factors on their UMMA pipes.

        Returns the advanced write states; cute.jit helpers do not propagate
        argument-object mutation, so callers must reassign them.
        """
        factor_pipe.producer_acquire(factor_write_state)
        factor_barrier = factor_pipe.producer_get_barrier(factor_write_state)
        stage = factor_write_state.index
        for blk in cutlass.range_constexpr(self.nblk):
            cute.copy(
                tma_f1,
                tFgF1[(None, k_tile * self.nblk + blk)],
                tFsF1[(None, stage * self.nblk + blk)],
                tma_bar_ptr=factor_barrier,
            )
        factor_pipe.producer_commit(factor_write_state)
        factor_write_state.advance()
        f2_pipe.producer_acquire(f2_write_state)
        cute.copy(
            tma_f2,
            tFgF2[(None, k_tile)],
            tFsF2[(None, f2_write_state.index)],
            tma_bar_ptr=f2_pipe.producer_get_barrier(f2_write_state),
        )
        f2_pipe.producer_commit(f2_write_state)
        f2_write_state.advance()
        return factor_write_state, f2_write_state

    @cute.jit
    def _produce_factors_and_a(
        self,
        tma_aq,
        tma_as,
        tma_f1,
        tma_f2,
        tAsAq,
        tAgAq,
        tAsAs,
        tAgAs,
        tFsF1,
        tFgF1,
        tFsF2,
        tFgF2,
        factor_pipe,
        f2_pipe,
        a_pipe,
        first_row_half,
        tile_count,
    ):
        """Load A half-tiles (own ring) and F1/F2 factor stages (UMMA pipe)."""
        row_halves = self.rows // HALF_TILE_ROWS
        a_write_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.a_pipe_stages if self.a_tile_pipe else self.a_stage_count,
        )
        if cutlass.const_expr(self.load_mode == NoiseLoadMode.RESIDENT):
            for half_tile_index in cutlass.range(row_halves * tile_count, unroll=1):
                a_pipe.producer_acquire(a_write_state)
                self._copy_a_half_tile(
                    tma_aq,
                    tma_as,
                    tAsAq,
                    tAgAq,
                    tAsAs,
                    tAgAs,
                    first_row_half + half_tile_index % row_halves,
                    half_tile_index // row_halves,
                    a_write_state.index,
                    a_pipe.producer_get_barrier(a_write_state),
                )
                a_pipe.producer_commit(a_write_state)
                a_write_state.advance()

        factor_write_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.stages
        )
        f2_write_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.stages
        )
        for tile_index in cutlass.range(tile_count, unroll=1):
            k_tile = tile_index
            factor_write_state, f2_write_state = self._produce_factor_tile(
                tma_f1,
                tma_f2,
                tFsF1,
                tFgF1,
                tFsF2,
                tFgF2,
                factor_pipe,
                f2_pipe,
                factor_write_state,
                f2_write_state,
                k_tile,
            )

            if cutlass.const_expr(self.a_tile_pipe):
                a_pipe.producer_acquire(a_write_state)
                a_barrier = a_pipe.producer_get_barrier(a_write_state)
                for row_half in cutlass.range_constexpr(row_halves):
                    self._copy_a_half_tile(
                        tma_aq,
                        tma_as,
                        tAsAq,
                        tAgAq,
                        tAsAs,
                        tAgAs,
                        first_row_half + row_half,
                        k_tile,
                        a_write_state.index * row_halves + row_half,
                        a_barrier,
                    )
                a_pipe.producer_commit(a_write_state)
                a_write_state.advance()
            elif cutlass.const_expr(self.load_mode == NoiseLoadMode.RING):
                for row_half in cutlass.range_constexpr(row_halves):
                    a_pipe.producer_acquire(a_write_state)
                    self._copy_a_half_tile(
                        tma_aq,
                        tma_as,
                        tAsAq,
                        tAgAq,
                        tAsAs,
                        tAgAs,
                        first_row_half + row_half,
                        k_tile,
                        a_write_state.index,
                        a_pipe.producer_get_barrier(a_write_state),
                    )
                    a_pipe.producer_commit(a_write_state)
                    a_write_state.advance()

    @cute.jit
    def _run_producer_warp_sm100(
        self,
        tma_aq,
        tma_as,
        tma_f1,
        tma_f2,
        tAsAq,
        tAgAq,
        tAsAs,
        tAgAs,
        tFsF1,
        tFgF1,
        tFsF2,
        tFgF2,
        factor_pipe,
        f2_pipe,
        a_pipe,
        first_row_half,
        tile_count,
    ):
        """Produce factor stages and A half-tiles."""
        self._produce_factors_and_a(
            tma_aq,
            tma_as,
            tma_f1,
            tma_f2,
            tAsAq,
            tAgAq,
            tAsAs,
            tAgAs,
            tFsF1,
            tFgF1,
            tFsF2,
            tFgF2,
            factor_pipe,
            f2_pipe,
            a_pipe,
            first_row_half,
            tile_count,
        )

    @cute.jit
    def _run_output_warp_sm100(
        self,
        tma_q,
        tQsQ,
        tQgQ,
        mE1,
        mKey,
        sE1,
        sE1B,
        output_pipe,
        lane,
        row_block,
        tile_count,
    ):
        """Generate E1 (unless the consumers own it), then drain A'."""
        if cutlass.const_expr(not self.consumer_e1):
            for row_sweep in cutlass.range_constexpr((self.rows + 31) // 32):
                self._generate_e1_rows(mE1, mKey, sE1, sE1B, 32 * row_sweep, lane, row_block)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier_arrive(
                barrier_id=_NamedBarrier.E1_READY,
                number_of_threads=self.consumer_threads + 64,
            )
        read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.out_stages
        )
        release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.out_stages
        )
        for store_index in cutlass.range(tile_count, unroll=1):
            output_pipe.consumer_wait(read_state)
            cute.copy(
                tma_q,
                tQsQ[(None, read_state.index)],
                tQgQ[(None, row_block, store_index)],
            )
            cute.arch.cp_async_bulk_commit_group()
            read_state.advance()
            if store_index >= self.out_stages - 1:
                cute.arch.cp_async_bulk_wait_group(self.out_stages - 1, read=True)
                output_pipe.consumer_release(release_state)
                release_state.advance()
        cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def _issue_noise_umma(
        self,
        tiled_mma_noise,
        tCtNoiseBase,
        tCrF1,
        tCrE1,
        acc_pipe,
        acc_write,
        factor_index,
    ):
        """Issue one k-tile's noise UMMA(s), fresh accumulation each.

        Transposed orientation (rows < 64): one acc stage holds the whole
        tile (``nblk`` M-blocks share the stage index). Direct orientation
        (rows == 64): each 128-column panel is its own acc stage, E1 is the
        A operand, and the F1 panel is B.
        """
        if cutlass.const_expr(self.direct):
            for pan in cutlass.range_constexpr(self.nblk):
                acc_pipe.producer_acquire(acc_write)
                tiled_mma_noise.set(tcgen05.Field.ACCUMULATE, False)
                cute.gemm(
                    tiled_mma_noise,
                    tCtNoiseBase[(None, None, None, acc_write.index)],
                    tCrE1[(None, None, 0, 0)],
                    tCrF1[(None, None, 0, factor_index * self.nblk + pan)],
                    tCtNoiseBase[(None, None, None, acc_write.index)],
                )
                acc_pipe.producer_commit(acc_write)
                acc_write.advance()
        else:
            acc_pipe.producer_acquire(acc_write)
            tiled_mma_noise.set(tcgen05.Field.ACCUMULATE, False)
            for blk in cutlass.range_constexpr(self.nblk):
                slot = acc_write.index * self.nblk + blk
                cute.gemm(
                    tiled_mma_noise,
                    tCtNoiseBase[(None, None, None, slot)],
                    tCrF1[(None, None, 0, factor_index * self.nblk + blk)],
                    tCrE1[(None, None, 0, 0)],
                    tCtNoiseBase[(None, None, None, slot)],
                )
            acc_pipe.producer_commit(acc_write)
            acc_write.advance()
        return tiled_mma_noise, acc_write

    @cute.jit
    def _run_mma_warp(
        self,
        tiled_mma_noise,
        tiled_mma_peel,
        tCrF1,
        tCrE1,
        tCrPa,
        tCrPb,
        tmem,
        tCtNoise_fake,
        tCtPeel_fake,
        noise_cols: cutlass.Constexpr,
        factor_pipe,
        f2_pipe,
        acc_pipe,
        peel_pipe,
        peel_done,
        tile_count,
    ):
        """Issue the noise and peel UMMAs; both are f8f6f4 atoms.

        The warp allocated TMEM before entering; the E1_READY barrier it
        arrives at below publishes the smem-held TMEM pointer, so every
        retriever reads the pointer only after it.
        """
        factor_read = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
        acc_write = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.acc_stages
        )
        peel_read = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.out_stages
        )
        peel_done_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

        f2_read = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)

        cute.arch.barrier(
            barrier_id=_NamedBarrier.E1_READY,
            number_of_threads=self.consumer_threads + (32 if self.consumer_e1 else 64),
        )
        acc_ptr = tmem.retrieve_ptr(Float32)
        tCtNoiseBase = cute.make_tensor(acc_ptr, tCtNoise_fake.layout)
        tCtPeel = cute.make_tensor(acc_ptr + noise_cols, tCtPeel_fake.layout)[(None, None, None, 0)]

        # Noise UMMA for tile 0. F1 stages are released at noise-UMMA
        # completion (their only reader), so the tile i+1 factor TMA is
        # decoupled from the tile i-1 peel handshake.
        factor_pipe.consumer_wait(factor_read)
        tiled_mma_noise, acc_write = self._issue_noise_umma(
            tiled_mma_noise, tCtNoiseBase, tCrF1, tCrE1, acc_pipe, acc_write, factor_read.index
        )
        factor_pipe.consumer_release(factor_read)
        factor_read.advance()

        tiled_mma_peel.set(tcgen05.Field.ACCUMULATE, False)
        peel_k_blocks = cute.size(tCrPa, mode=[2])
        for tile_index in cutlass.range(tile_count, unroll=1):
            if tile_index + 1 < tile_count:
                factor_pipe.consumer_wait(factor_read)
                tiled_mma_noise, acc_write = self._issue_noise_umma(
                    tiled_mma_noise,
                    tCtNoiseBase,
                    tCrF1,
                    tCrE1,
                    acc_pipe,
                    acc_write,
                    factor_read.index,
                )
                factor_pipe.consumer_release(factor_read)
                factor_read.advance()

            # Peel UMMA over this tile's staged A' against its F2 stage.
            peel_pipe.consumer_wait(peel_read)
            f2_pipe.consumer_wait(f2_read)
            for k_blk in cutlass.range_constexpr(peel_k_blocks):
                cute.gemm(
                    tiled_mma_peel,
                    tCtPeel,
                    tCrPa[(None, None, k_blk, peel_read.index)],
                    tCrPb[(None, None, k_blk, f2_read.index)],
                    tCtPeel,
                )
                tiled_mma_peel.set(tcgen05.Field.ACCUMULATE, True)
            peel_pipe.consumer_release(peel_read)
            peel_read.advance()

            # The release is UMMA-commit based (PipelineTmaUmma), so it
            # fires only after the peel UMMAs that read the stage complete.
            f2_pipe.consumer_release(f2_read)
            f2_read.advance()

        peel_done.producer_commit(peel_done_state)
        acc_pipe.producer_tail(acc_write)

    # -- consumers ----------------------------------------------------------

    # noqa C901: compile-time load-mode dispatch in the hot tile loop.
    @cute.jit
    def _run_consumer_warpgroup_sm100(  # noqa: C901
        self,
        globals_: _ConsumerGlobals,
        tiled_mma_n,
        tiled_mma_u16,
        shared: _ConsumerShared,
        mE1,
        mKey,
        sE1B,
        sNoise,
        tiled_t2r_noise,
        tmem,
        tCtNoise_fake,
        tCtPeel_fake,
        noise_cols: cutlass.Constexpr,
        tiled_t2r_peel,
        a_pipe,
        acc_pipe,
        output_pipe,
        peel_pipe,
        peel_done,
        coords: _ConsumerCoords,
    ):
        """Stats, TMEM noise drain, decode, quantize, and peel readback.

        TMEM is allocated by the MMA warp while this warpgroup runs its
        stats/E1 prologue; the pointer is retrieved after the E1_READY
        barrier, whose passage orders the allocator's smem write before
        every read here.
        """
        mAlpha, mBeta = globals_.alpha, globals_.beta
        mPeel, mStat = globals_.peel, globals_.stats
        sAq, sAs = shared.a_codes, shared.a_scales
        sAp = shared.a_prime
        sE1, sAlpha, sBeta = shared.e1, shared.alpha, shared.beta
        thread_idx, lane = coords.thread, coords.lane
        row_block = coords.row_block
        tile_count = coords.tile_count
        row_halves = self.rows // HALF_TILE_ROWS
        bf16 = cutlass.BFloat16

        noise_thread = tiled_mma_n.get_slice(thread_idx)
        noise_coords = noise_thread.partition_C(cute.make_identity_tensor((16, self.bk)))

        pair_template = cute.make_identity_tensor((16, self.bk // 2))
        pair_fragment_shape = tiled_mma_u16.get_slice(thread_idx).partition_C(pair_template).shape
        code_words = cute.make_rmem_tensor(pair_fragment_shape, cutlass.Uint16)
        quantized_word_fragments = [
            cute.make_rmem_tensor(pair_fragment_shape, cutlass.Uint16) for _ in range(row_halves)
        ]
        a_fragment_shape = noise_thread.partition_C(cute.make_identity_tensor((16, self.bk))).shape
        a_fragment = cute.make_rmem_tensor(a_fragment_shape, bf16)
        a_words = cute.recast_tensor(a_fragment, cutlass.Uint32)
        noise_words = cute.make_rmem_tensor(4 * self.bk // self.n_group, cutlass.Uint32)

        consumer_warp = thread_idx // 32
        scale_group_base = 2 * consumer_warp + ((lane >> 1) & 1)
        first_fragment_row = lane >> 2

        # TMEM noise drain partitioning: warp w owns TMEM lanes (= transposed
        # tile columns) 32*w..32*w+31 of each 128-column block.
        thr_t2r_noise = tiled_t2r_noise.get_slice(thread_idx)
        noise_rmem = [
            cute.make_rmem_tensor(
                thr_t2r_noise.partition_D(
                    cute.make_identity_tensor((self.bk_mma, self.rows))
                ).shape,
                Float32,
            )
            for _ in range(self.nblk)
        ]
        # Transposed f32 staging view (+16-float row skew for bank spread).
        sNoiseT = cute.make_tensor(
            sNoise.iterator,
            cute.make_layout(
                (self.bk_mma, self.rows, self.nblk),
                stride=(1, self.bk + 16, self.bk_mma),
            ),
        )
        sNoiseF = cute.make_tensor(
            cute.recast_ptr(sNoise.iterator, dtype=Float32),
            cute.make_layout((self.rows, self.bk), stride=(self.bk + 16, 1)),
        )

        a_copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), cutlass.Uint16
        )
        tiled_copy_aq = cute.make_tiled_copy_C(a_copy_atom, tiled_mma_u16)
        shared_code_partition = tiled_copy_aq.get_slice(thread_idx).partition_S(sAq)
        output_copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(False, 2), cutlass.Uint16
        )
        tiled_copy_output = cute.make_tiled_copy_C(output_copy_atom, tiled_mma_u16)
        shared_output_partition = tiled_copy_output.get_slice(thread_idx).partition_D(sAp)

        if thread_idx < self.rows:
            global_row = row_block * self.rows + thread_idx
            alpha, beta = self._combine_row_stats(
                mStat,
                global_row,
                tile_count * self.bk,
            )
            sAlpha[thread_idx] = alpha
            sBeta[thread_idx] = beta
            mAlpha[global_row] = alpha.to(bf16)
            mBeta[global_row] = beta.to(bf16)
        elif cutlass.const_expr(self.consumer_e1):
            # E1 stays on threads 64..127 (warps 2-3).
            if thread_idx >= 64 and thread_idx < 128:
                self._generate_e1_rows(
                    mE1,
                    mKey,
                    sE1,
                    sE1B,
                    32 * (consumer_warp - 2),
                    lane,
                    row_block,
                )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(
            barrier_id=_NamedBarrier.CONSUMER,
            number_of_threads=self.consumer_threads,
        )

        alpha_word_pairs = []
        beta_word_pairs = []
        for row_half in cutlass.range_constexpr(row_halves):
            alpha_row_0 = sAlpha[row_half * 16 + first_fragment_row]
            alpha_row_1 = sAlpha[row_half * 16 + first_fragment_row + 8]
            beta_row_0 = sBeta[row_half * 16 + first_fragment_row]
            beta_row_1 = sBeta[row_half * 16 + first_fragment_row + 8]
            alpha_word_pairs.append(
                (
                    _f32x2_to_bf16x2(alpha_row_0, alpha_row_0),
                    _f32x2_to_bf16x2(alpha_row_1, alpha_row_1),
                )
            )
            beta_word_pairs.append(
                (
                    _f32x2_to_bf16x2(beta_row_0, beta_row_0),
                    _f32x2_to_bf16x2(beta_row_1, beta_row_1),
                )
            )

        cute.arch.barrier(
            barrier_id=_NamedBarrier.E1_READY,
            number_of_threads=self.consumer_threads + (32 if self.consumer_e1 else 64),
        )
        acc_ptr = tmem.retrieve_ptr(Float32)
        tCtNoiseBase = cute.make_tensor(acc_ptr, tCtNoise_fake.layout)
        tCtPeel = cute.make_tensor(acc_ptr + noise_cols, tCtPeel_fake.layout)[
            ((None, None), 0, 0, 0)
        ]

        # The bit-exact beta * E1 peel half needs only sE1 and sBeta, both
        # final at this barrier -- write it here, hidden under the first
        # noise UMMA's latency, instead of serialized after the peel-readback
        # tail.
        if thread_idx < self.rows:
            self._write_e1_peel_row(
                sE1,
                sBeta,
                cute.local_tile(
                    mPeel, (16, PEEL_COLS), (row_block * row_halves + thread_idx // 16, 0)
                ),
                thread_idx,
            )

        acc_read = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
        output_write_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.out_stages
        )
        peel_write_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.out_stages
        )
        for tile_index in cutlass.range(tile_count, unroll=1):
            # Drain this tile's noise accumulator: TMEM -> registers -> the
            # (rows, bk) f32 staging tile, transposed on the way through.
            # Both column blocks' TMEM loads issue before any store so
            # their latencies overlap instead of serializing ld->sts pairs.
            acc_pipe.consumer_wait(acc_read)
            for blk in cutlass.range_constexpr(self.nblk):
                slot = acc_read.index * self.nblk + blk
                cute.copy(
                    tiled_t2r_noise,
                    thr_t2r_noise.partition_S(tCtNoiseBase[((None, None), 0, 0, slot)]),
                    noise_rmem[blk],
                )
            # WAR guard: no warp may overwrite the staging tile until
            # every warp has finished the previous tile's noise reads
            # (quantize reads rows other warps drained). Sits after the
            # tcgen05.ld issues so the wait hides under load latency.
            cute.arch.barrier(
                barrier_id=_NamedBarrier.CONSUMER,
                number_of_threads=self.consumer_threads,
            )
            for blk in cutlass.range_constexpr(self.nblk):
                cute.autovec_copy(
                    noise_rmem[blk],
                    thr_t2r_noise.partition_D(sNoiseT[(None, None, blk)]),
                )
            cute.arch.fence_view_async_tmem_load()
            acc_pipe.consumer_release(acc_read)
            acc_read.advance()
            cute.arch.barrier(
                barrier_id=_NamedBarrier.CONSUMER,
                number_of_threads=self.consumer_threads,
            )

            if cutlass.const_expr(self.a_tile_pipe):
                self._wait_a_tile(a_pipe, tile_index)
            for row_half in cutlass.range_constexpr(row_halves):
                stage = self._load_a_codes(
                    a_pipe,
                    tile_index,
                    row_half,
                    tiled_copy_aq,
                    shared_code_partition,
                    code_words,
                )
                # Noise pairs: two f32 staging reads and a single
                # cvt.rn.bf16x2.f32 rounding.
                for block in cutlass.range_constexpr(2 * self.bk // self.n_group):
                    col_lo = noise_coords[4 * block][1]
                    col_hi = noise_coords[4 * block + 1][1]
                    for fragment_half in cutlass.range_constexpr(2):
                        noise_row = 16 * row_half + first_fragment_row + 8 * fragment_half
                        noise_words[fragment_half + 2 * block] = _f32x2_to_bf16x2(
                            sNoiseF[(noise_row, col_hi)],
                            sNoiseF[(noise_row, col_lo)],
                        )
                self._open_a_fragment(
                    sAs,
                    stage,
                    scale_group_base,
                    first_fragment_row,
                    code_words,
                    a_words,
                )
                self._quantize_fragment_tmem(
                    noise_words,
                    a_words,
                    quantized_word_fragments[row_half],
                    alpha_word_pairs[row_half],
                    beta_word_pairs[row_half],
                )
                if cutlass.const_expr(self.a_tile_pipe):
                    if cutlass.const_expr(row_half == row_halves - 1):
                        self._release_a_tile(a_pipe, tile_index)
                else:
                    self._release_a_half_tile(a_pipe, tile_index, row_half)

                if cutlass.const_expr(row_half == 0):
                    output_pipe.producer_acquire(output_write_state)
                    peel_pipe.producer_acquire(peel_write_state)
                cute.copy(
                    tiled_copy_output,
                    tiled_copy_output.retile(quantized_word_fragments[row_half])[(None, 0, None)],
                    shared_output_partition[(None, row_half, None, output_write_state.index)],
                )
                if cutlass.const_expr(row_half == row_halves - 1):
                    cute.arch.fence_proxy("async.shared", space="cta")
                    output_pipe.producer_commit(output_write_state)
                    peel_pipe.producer_commit(peel_write_state)
                    output_write_state.advance()
                    peel_write_state.advance()

        self._finalize_peel_sm100(
            tiled_t2r_peel,
            tCtPeel,
            peel_done,
            mPeel,
            thread_idx,
            row_block,
        )

    @cute.jit
    def _finalize_peel_sm100(
        self,
        tiled_t2r_peel,
        tCtPeel,
        peel_done,
        mPeel,
        thread_idx,
        row_block,
    ):
        """Peel readback: wait for the last peel UMMA, then write the AF2
        columns (the beta * E1 half was already written before the tile
        loop -- it needs only sE1 and sBeta)."""
        bf16 = cutlass.BFloat16
        peel_done.consumer_wait(pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1))
        thr_t2r_peel = tiled_t2r_peel.get_slice(thread_idx)
        peel_coords = thr_t2r_peel.partition_D(cute.make_identity_tensor((64, R)))
        peel_rmem = cute.make_rmem_tensor(peel_coords.shape, Float32)
        cute.copy(tiled_t2r_peel, thr_t2r_peel.partition_S(tCtPeel), peel_rmem)
        cute.arch.fence_view_async_tmem_load()

        peel_output = cute.local_tile(mPeel, (self.rows, PEEL_COLS), (row_block, 0))
        for item in cutlass.range_constexpr(cute.size(peel_rmem)):
            local_row = peel_coords[item][0]
            peel_col = peel_coords[item][1]
            if local_row < self.rows:
                peel_output[(local_row, R + peel_col)] = peel_rmem[item].to(bf16)

    # noqa C901: the direct-topology hot loop; compile-time chunk dispatch.
    @cute.jit
    def _run_consumer_warpgroup_direct(  # noqa: C901
        self,
        globals_: _ConsumerGlobals,
        sAq,
        sAs,
        sApU16,
        sE1,
        sAlpha,
        sBeta,
        tiled_t2r_noise,
        tCtNoiseBase,
        tiled_t2r_peel,
        tCtPeel,
        a_pipe,
        acc_pipe,
        output_pipe,
        peel_pipe,
        peel_done,
        coords: _ConsumerCoords,
    ):
        """rows=64 consumers: noise arrives per 128-column panel straight
        from TMEM in each thread's own registers; codes, scales, and the A'
        store are all plain aligned shared-memory accesses. No
        ldmatrix/stmatrix, no staging round-trip, no per-tile barrier.

        Exactness of the orientation flip: the atom's K = 32 result here is
        the order-independent exact integer dot rounded once to f32 (every
        product lies on the 2^-8 grid, above the truncation window), so swapping which operand feeds which port cannot
        change a bit. "Same bytes, roles swapped" alone would not be a
        valid argument."""
        mAlpha, mBeta = globals_.alpha, globals_.beta
        mPeel, mStat = globals_.peel, globals_.stats
        thread_idx = coords.thread
        row_block = coords.row_block
        tile_count = coords.tile_count
        warp = thread_idx // 32

        thr_t2r = tiled_t2r_noise.get_slice(thread_idx)
        panel_coords = thr_t2r.partition_D(cute.make_identity_tensor((64, 128)))
        noise_rmem = cute.make_rmem_tensor(panel_coords.shape, Float32)
        n_vals = cute.size(panel_coords)
        # Ld16x256b hands each thread ((2, 2, 16), ...) values: an adjacent
        # column pair, times two rows 8 apart, times 16 8-column steps —
        # value i = pair + 2*row_half + 4*step. Both rows sit inside the
        # warp's own A half-tile. The bit-exact A' gate pins this shape.
        assert n_vals % 4 == 0
        row0 = panel_coords[0][0]
        row1 = panel_coords[2][0]

        if thread_idx < self.rows:
            global_row = row_block * self.rows + thread_idx
            alpha, beta = self._combine_row_stats(
                mStat,
                global_row,
                tile_count * self.bk,
            )
            sAlpha[thread_idx] = alpha
            sBeta[thread_idx] = beta
            mAlpha[global_row] = alpha.to(cutlass.BFloat16)
            mBeta[global_row] = beta.to(cutlass.BFloat16)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(
            barrier_id=_NamedBarrier.CONSUMER,
            number_of_threads=WG_THREADS,
        )
        alpha_pairs = []
        beta_pairs = []
        for half in cutlass.range_constexpr(2):
            row_h = row0 if half == 0 else row1
            alpha_h = sAlpha[row_h]
            beta_h = sBeta[row_h]
            alpha_pairs.append(_f32x2_to_bf16x2(alpha_h, alpha_h))
            beta_pairs.append(_f32x2_to_bf16x2(beta_h, beta_h))

        cute.arch.barrier(
            barrier_id=_NamedBarrier.E1_READY,
            number_of_threads=WG_THREADS + (32 if self.consumer_e1 else 64),
        )

        # Mirror of the transposed path: beta * E1 is written up front
        # (see _run_consumer_warpgroup_sm100).
        if thread_idx < self.rows:
            self._write_e1_peel_row(
                sE1,
                sBeta,
                cute.local_tile(
                    mPeel,
                    (16, PEEL_COLS),
                    (row_block * (self.rows // HALF_TILE_ROWS) + thread_idx // 16, 0),
                ),
                thread_idx,
            )

        acc_read = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.acc_stages)
        output_write_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.out_stages
        )
        peel_write_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.out_stages
        )
        for tile_index in cutlass.range(tile_count, unroll=1):
            stage = self._wait_a_half_tile(a_pipe, tile_index, warp, 0)
            for pan in cutlass.range_constexpr(self.nblk):
                acc_pipe.consumer_wait(acc_read)
                cute.copy(
                    tiled_t2r_noise,
                    thr_t2r.partition_S(tCtNoiseBase[((None, None), 0, 0, acc_read.index)]),
                    noise_rmem,
                )
                cute.arch.fence_view_async_tmem_load()
                acc_pipe.consumer_release(acc_read)
                acc_read.advance()
                # Acquire the A' stage only after the first panel's acc
                # release: blocking on the output ring before releasing TMEM
                # would couple the MMA warp's panel prefetch to the store
                # drain for nothing.
                if cutlass.const_expr(pan == 0):
                    output_pipe.producer_acquire(output_write_state)
                    peel_pipe.producer_acquire(peel_write_state)
                for step in cutlass.range_constexpr(n_vals // 4):
                    col0 = 128 * pan + panel_coords[4 * step][1]
                    for half in cutlass.range_constexpr(2):
                        i = 4 * step + 2 * half
                        row_h = row0 if half == 0 else row1
                        splat = _splat_bf16x2(
                            cutlass.Uint32(sAs[(row_h % 16, col0 // BLOCK_SCALE_GROUP, stage)])
                        )
                        code_pair = sAq[(row_h % 16, col0 // 2, stage)]
                        a_word = _open_int8x2_bf16x2(code_pair, splat)
                        noise_word = _f32x2_to_bf16x2(noise_rmem[i + 1], noise_rmem[i])
                        scaled = _fma_bf16x2(
                            alpha_pairs[half], a_word, _mul_bf16x2(beta_pairs[half], noise_word)
                        )
                        sApU16[
                            (row_h, ((col0 % 16) // 2, col0 // 16), output_write_state.index)
                        ] = _bf16x2_to_e4m3x2(scaled)
            self._release_a_half_tile(a_pipe, tile_index, warp)
            cute.arch.fence_proxy("async.shared", space="cta")
            output_pipe.producer_commit(output_write_state)
            peel_pipe.producer_commit(peel_write_state)
            output_write_state.advance()
            peel_write_state.advance()

        self._finalize_peel_sm100(
            tiled_t2r_peel,
            tCtPeel,
            peel_done,
            mPeel,
            thread_idx,
            row_block,
        )

    # -- launch -------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mAqCodes: cute.Tensor,  # (m, k) i8: committed activation codes blob
        mAs: cute.Tensor,  # (m, k/8) bf16: committed group-scales blob
        mAlpha: cute.Tensor,  # (m,) bf16 out
        mBeta: cute.Tensor,  # (m,) bf16 out
        mE1: cute.Tensor,  # (m*r,) e4m3 flat out
        mF1e4: cute.Tensor,  # (k, 2r) s8 view: zero-padded e4m3 F1 (pack_noise_factor)
        mF2: cute.Tensor,  # (r, k) e4m3: original codes
        mAp: cute.Tensor,  # (m, k) e4m3 out
        mPeel: cute.Tensor,  # (m, 2r) bf16 out
        mKey: cute.Tensor,  # (8,) u32: cA
        mStat: cute.Tensor,  # (2*m*k/512,) f32: commit per-chunk (ssq, absmax)
        stream,
    ):
        m, k = mAqCodes.shape
        # Static chunk count for the batched stats combine (k is a
        # trace-time int: the compile cache is keyed on (m, k)).
        self.stats_chunks = k // 512
        bk, stages = self.bk, self.stages
        out_stages = self.out_stages
        row_halves = self.rows // HALF_TILE_ROWS
        f8 = cutlass.Float8E4M3FN
        u16 = cutlass.Uint16

        # -- tcgen05 tiled MMAs (the only MMA atoms the kernel executes) --
        tiled_mma_noise = sm100_utils.make_trivial_tiled_mma(
            f8,
            f8,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (64, 128) if self.direct else (self.bk_mma, self.rows),
        )
        tiled_mma_peel = sm100_utils.make_trivial_tiled_mma(
            f8,
            f8,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (64, R),
        )
        tiler_noise = (64, 128, 32) if self.direct else (self.bk_mma, self.rows, 32)
        tiler_peel = (64, R, bk)

        # -- layout scaffolding (issues no MMA; see module docstring) --
        # The N permutation spreads the tile over ``consumer_warps`` warps:
        # each warp owns 16 contiguous columns per
        # ``n_group``-column group (n_group = 16 * consumer_warps), with the
        # instance pairs interleaved at 2-column granularity so a thread's
        # four bytes per 16-column strip stay contiguous and its noise pairs
        # stay column-adjacent (trace-time verified over every thread).
        op16 = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, Float32, (16, 8, 16))
        tiled_mma_n = cute.make_tiled_mma(
            op16,
            (1, self.consumer_warps, 1),
            permutation_mnk=(
                None,
                cute.make_layout((2, 4, self.consumer_warps, 2), stride=(1, 4, 16, 2)),
                None,
            ),
        )
        tiled_mma_u16 = cute.make_tiled_mma(op16, (1, self.consumer_warps, 1))

        a_stage_count = (
            row_halves * k // bk if self.load_mode == NoiseLoadMode.RESIDENT else self.ring_a_stages
        )

        # A codes staging (opened in registers, never an MMA operand): a
        # K-major 128B-swizzled atom tiled over the (16, bk/2) u16 half-tile
        # and staged. bk in (128, 256) makes the bk/2-element row a whole
        # number of 128-byte swizzle spans.
        sAq_layout = cute.tile_to_shape(
            make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, u16),
            (16, bk // 2, a_stage_count),
            order=(0, 1, 2),
        )
        sAs_layout = cute.make_layout(
            (16, bk // BLOCK_SCALE_GROUP, a_stage_count),
            stride=(bk // BLOCK_SCALE_GROUP, 1, 16 * bk // BLOCK_SCALE_GROUP),
        )
        # UMMA operand layouts. In the direct topology E1 is the noise A
        # operand and the F1 panels are B; both byte images match the
        # transposed orientation's (K-major (rows|panel, 32) tiles).
        if cutlass.const_expr(self.direct):
            sF1_layout = sm100_utils.make_smem_layout_b(
                tiled_mma_noise, tiler_noise, f8, self.nblk * stages
            )
            sE1B_layout = sm100_utils.make_smem_layout_a(tiled_mma_noise, tiler_noise, f8, 1)
        else:
            sF1_layout = sm100_utils.make_smem_layout_a(
                tiled_mma_noise, tiler_noise, f8, self.nblk * stages
            )
            sE1B_layout = sm100_utils.make_smem_layout_b(tiled_mma_noise, tiler_noise, f8, 1)
        sF2_layout = sm100_utils.make_smem_layout_b(tiled_mma_peel, tiler_peel, f8, stages)
        # The A' operand deliberately uses the unswizzled K_INTER canonical
        # atom (16-byte column panels every 64 rows): its byte image has an
        # exact plain-strided u16 twin, which is what lets the stmatrix
        # writes and the TMA store share the peel UMMA's bytes.
        peel_a_shape = tiled_mma_peel.partition_shape_A(cute.dice(tiler_peel, (1, None, 1)))
        sAp_one = tile_to_mma_shape(
            make_smem_layout_atom(SmemLayoutAtomKind.K_INTER, f8),
            cute.append(peel_a_shape, 1),
            order=(1, 2, 3),
        )
        # Full 64-row stages: K_INTER's 16-byte panels interleave the whole
        # 64-row M extent through the stage, so the stage stride cannot
        # shrink to rows*bk without stages aliasing real bytes. The pad rows
        # (rows..63) are never written and never zeroed -- their junk only
        # feeds accumulator rows the readback ignores.
        sAp_layout = cute.make_composed_layout(
            sAp_one.inner,
            0,
            cute.make_layout(
                (*sAp_one.outer.shape[:3], out_stages),
                stride=(*sAp_one.outer.stride[:3], 64 * bk),
            ),
        )
        panels = bk // 16
        sApU16_layout = cute.make_layout(
            (self.rows, (8, panels), out_stages),
            stride=(8, (1, 512), 32 * bk),
        )
        sApU16_store_layout = sApU16_layout

        def _tma_load(tensor, smem_layout_staged, tile_shape):
            return cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                tensor,
                cute.slice_(smem_layout_staged, (None, None, 0)),
                tile_shape,
            )

        mAqScales = cute.make_tensor(cute.recast_ptr(mAs.iterator, dtype=u16), mAs.layout)
        mAqCodesU16 = cute.make_tensor(
            cute.recast_ptr(mAqCodes.iterator, dtype=u16),
            cute.make_layout((m, k // 2), stride=(k // 2, 1)),
        )
        mApU16 = cute.make_tensor(
            cute.recast_ptr(mAp.iterator, dtype=u16),
            cute.make_layout((m, k // 2), stride=(k // 2, 1)),
        )
        # Recast the int8 view to its real e4m3 element type.
        mF1e4 = cute.make_tensor(
            cute.recast_ptr(mF1e4.iterator, dtype=f8),
            cute.make_layout((k, PACKED_NOISE_K), stride=(PACKED_NOISE_K, 1)),
        )

        tma_aq, tma_taq = _tma_load(mAqCodesU16, sAq_layout, (16, bk // 2))
        tma_as, tma_tas = _tma_load(mAqScales, sAs_layout, (16, bk // BLOCK_SCALE_GROUP))

        if cutlass.const_expr(self.direct):
            tma_f1, tma_tf1 = cute.nvgpu.make_tiled_tma_atom_B(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                mF1e4,
                cute.slice_(sF1_layout, (None, None, None, 0)),
                tiler_noise,
                tiled_mma_noise,
            )
        else:
            tma_f1, tma_tf1 = cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                mF1e4,
                cute.slice_(sF1_layout, (None, None, None, 0)),
                tiler_noise,
                tiled_mma_noise,
            )
        tma_f2, tma_tf2 = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mF2,
            cute.slice_(sF2_layout, (None, None, None, 0)),
            tiler_peel,
            tiled_mma_peel,
        )
        tma_q, tma_tq = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mApU16,
            cute.slice_(sApU16_store_layout, (None, None, 0)),
            (self.rows, bk // 2),
        )

        f1_slot_bytes = cute.size_in_bytes(f8, cute.slice_(sF1_layout, (None, None, None, 0)))
        f2_slot_bytes = cute.size_in_bytes(f8, cute.slice_(sF2_layout, (None, None, None, 0)))
        factor_tx_bytes = self.nblk * f1_slot_bytes
        f2_tx_bytes = f2_slot_bytes
        a_tile_bytes = cute.size_in_bytes(
            u16, cute.slice_(sAq_layout, (None, None, 0))
        ) + cute.size_in_bytes(u16, cute.slice_(sAs_layout, (None, None, 0)))

        @cute.struct
        class SharedStorage:
            factor_mbar: cute.struct.MemRange[cutlass.Int64, stages * 2]
            f2_mbar: cute.struct.MemRange[cutlass.Int64, stages * 2]
            a_mbar: cute.struct.MemRange[cutlass.Int64, a_stage_count * 2]
            output_mbar: cute.struct.MemRange[cutlass.Int64, out_stages * 2]
            peel_mbar: cute.struct.MemRange[cutlass.Int64, out_stages * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.acc_stages]
            peel_done_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            sAlpha: cute.struct.MemRange[Float32, 16 * row_halves]
            sBeta: cute.struct.MemRange[Float32, 16 * row_halves]
            sE1: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, 16 * row_halves * R], 128]
            sNoise: cute.struct.Align[
                cute.struct.MemRange[Float32, 4 if self.direct else self.rows * (bk + 16)],
                1024,
            ]
            sE1B: cute.struct.Align[cute.struct.MemRange[f8, cute.cosize(sE1B_layout)], 1024]
            sAq: cute.struct.Align[cute.struct.MemRange[u16, cute.cosize(sAq_layout)], 1024]
            sAs: cute.struct.Align[cute.struct.MemRange[u16, cute.cosize(sAs_layout)], 1024]
            sF1: cute.struct.Align[cute.struct.MemRange[f8, cute.cosize(sF1_layout)], 1024]
            sF2: cute.struct.Align[cute.struct.MemRange[f8, cute.cosize(sF2_layout)], 1024]
            sAp: cute.struct.Align[cute.struct.MemRange[f8, 64 * bk * out_stages], 1024]

        self.a_stage_count = a_stage_count
        self.shared_storage = SharedStorage
        assert SharedStorage.size_in_bytes() <= _SMEM_CAPACITY_BYTES, (
            f"smem overflow: {SharedStorage.size_in_bytes()} bytes"
        )

        grid = (m // self.rows, 1, 1)
        self.kernel(
            tma_aq,
            tma_taq,
            tma_as,
            tma_tas,
            tma_f1,
            tma_tf1,
            tma_f2,
            tma_tf2,
            tma_q,
            tma_tq,
            mAlpha,
            mBeta,
            mE1,
            mPeel,
            mKey,
            mStat,
            tiled_mma_noise,
            tiled_mma_peel,
            tiled_mma_n,
            tiled_mma_u16,
            sAq_layout,
            sAs_layout,
            sF1_layout,
            sE1B_layout,
            sF2_layout,
            sAp_layout,
            sApU16_layout,
            sApU16_store_layout,
            factor_tx_bytes,
            f2_tx_bytes,
            a_tile_bytes,
        ).launch(
            grid=grid,
            block=[self.consumer_threads + 96, 1, 1],
            stream=stream,
        )

    # noqa C901: warp-role dispatch; the branches are the topology.
    @cute.kernel
    def kernel(  # noqa: C901
        self,
        tma_aq: cute.CopyAtom,
        mAqCodes: cute.Tensor,
        tma_as: cute.CopyAtom,
        mAqScales: cute.Tensor,
        tma_f1: cute.CopyAtom,
        mF1e4: cute.Tensor,
        tma_f2: cute.CopyAtom,
        mF2: cute.Tensor,
        tma_q: cute.CopyAtom,
        mAp: cute.Tensor,
        mAlpha: cute.Tensor,
        mBeta: cute.Tensor,
        mE1: cute.Tensor,
        mPeel: cute.Tensor,
        mKey: cute.Tensor,
        mStat: cute.Tensor,
        tiled_mma_noise: cute.TiledMma,
        tiled_mma_peel: cute.TiledMma,
        tiled_mma_n: cute.TiledMma,
        tiled_mma_u16: cute.TiledMma,
        sAq_layout: cute.ComposedLayout,
        sAs_layout: cute.Layout,
        sF1_layout: cute.ComposedLayout,
        sE1B_layout: cute.ComposedLayout,
        sF2_layout: cute.ComposedLayout,
        sAp_layout: cute.ComposedLayout,
        sApU16_layout: cute.Layout,
        sApU16_store_layout: cute.Layout,
        factor_tx_bytes: cutlass.Constexpr,
        f2_tx_bytes: cutlass.Constexpr,
        a_tile_bytes: cutlass.Constexpr,
    ):
        bk, stages = self.bk, self.stages
        row_halves = self.rows // HALF_TILE_ROWS

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.producer_warp:
            self._prefetch_tma_descriptors(tma_aq, tma_as, tma_f1, tma_f2, tma_q)

        row_block, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        lane = thread_idx % 32
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=_NamedBarrier.TMEM_PTR,
            num_threads=32 * (self.consumer_warps + 1),  # consumer warps + MMA warp
        )
        tmem = cutlass.utils.TmemAllocator(
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp,
        )

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_warpgroup = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.consumer_threads)
        # Direct topology: each consumer warp owns exactly one A half-tile
        # stage (its own rows), so a stage has a single releasing warp.
        # Otherwise every consumer warp waits on and releases every half-tile
        # stage (TMA-async releases arrive once per calling warp).
        a_consumer = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 1 if self.direct else self.consumer_warps
        )
        output_consumer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        mma_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        factor_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.factor_mbar.data_ptr(),
            num_stages=stages,
            producer_group=producer_group,
            consumer_group=mma_group,
            tx_count=factor_tx_bytes,
            defer_sync=True,
        )
        f2_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.f2_mbar.data_ptr(),
            num_stages=stages,
            producer_group=producer_group,
            consumer_group=mma_group,
            tx_count=f2_tx_bytes,
            defer_sync=True,
        )
        a_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_mbar.data_ptr(),
            num_stages=self.a_pipe_stages if self.a_tile_pipe else self.a_stage_count,
            producer_group=producer_group,
            consumer_group=a_consumer,
            tx_count=(row_halves * a_tile_bytes) if self.a_tile_pipe else a_tile_bytes,
            defer_sync=True,
        )
        output_pipe = pipeline.PipelineAsync.create(
            barrier_storage=storage.output_mbar.data_ptr(),
            num_stages=self.out_stages,
            producer_group=consumer_warpgroup,
            consumer_group=output_consumer,
            defer_sync=True,
        )
        peel_pipe = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.peel_mbar.data_ptr(),
            num_stages=self.out_stages,
            producer_group=consumer_warpgroup,
            consumer_group=mma_group,
            defer_sync=True,
        )
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.acc_stages,
            producer_group=mma_group,
            consumer_group=consumer_warpgroup,
            defer_sync=True,
        )
        peel_done = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.peel_done_mbar.data_ptr(),
            num_stages=1,
            producer_group=mma_group,
            consumer_group=consumer_warpgroup,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sAq = storage.sAq.get_tensor(sAq_layout.outer, swizzle=sAq_layout.inner)
        sAs = storage.sAs.get_tensor(sAs_layout)
        sF1 = storage.sF1.get_tensor(sF1_layout.outer, swizzle=sF1_layout.inner)
        sE1B = storage.sE1B.get_tensor(sE1B_layout.outer, swizzle=sE1B_layout.inner)
        sF2 = storage.sF2.get_tensor(sF2_layout.outer, swizzle=sF2_layout.inner)
        sApE4 = storage.sAp.get_tensor(sAp_layout.outer, swizzle=sAp_layout.inner)
        sApU16 = cute.make_tensor(
            cute.recast_ptr(storage.sAp.data_ptr(), dtype=cutlass.Uint16),
            sApU16_layout,
        )
        sApStore = cute.make_tensor(
            cute.recast_ptr(storage.sAp.data_ptr(), dtype=cutlass.Uint16),
            sApU16_store_layout,
        )

        sNoise = storage.sNoise.get_tensor(cute.make_layout((self.rows, bk), stride=(bk + 16, 1)))
        sE1 = storage.sE1.get_tensor(cute.make_layout((16, R, row_halves), stride=(R, 1, 16 * R)))
        sAlpha = storage.sAlpha.get_tensor(cute.make_layout(self.rows))
        sBeta = storage.sBeta.get_tensor(cute.make_layout(self.rows))

        gAq = cute.local_tile(mAqCodes, (16, bk // 2), (None, None))
        gAs = cute.local_tile(mAqScales, (16, bk // BLOCK_SCALE_GROUP), (None, None))
        gAp = cute.local_tile(mAp, (self.rows, bk // 2), (None, None))
        gF1 = cute.local_tile(mF1e4, (self.bk_mma, PACKED_NOISE_K), (None, 0))
        gF2 = cute.local_tile(mF2, (R, bk), (0, None))  # (r, bk, kt)

        tile_count = cute.size(gAp, mode=[3])
        first_row_half = row_block * row_halves

        cta_layout = cute.make_layout(1)
        tAsAq, tAgAq = cpasync.tma_partition(
            tma_aq, 0, cta_layout, cute.group_modes(sAq, 0, 2), cute.group_modes(gAq, 0, 2)
        )
        tAsAs, tAgAs = cpasync.tma_partition(
            tma_as, 0, cta_layout, cute.group_modes(sAs, 0, 2), cute.group_modes(gAs, 0, 2)
        )
        tFsF1, tFgF1 = cpasync.tma_partition(
            tma_f1,
            0,
            cta_layout,
            cute.group_modes(sF1, 0, 3),
            cute.group_modes(gF1, 0, 2),
        )
        tFsF2, tFgF2 = cpasync.tma_partition(
            tma_f2,
            0,
            cta_layout,
            cute.group_modes(sF2, 0, 3),
            cute.group_modes(gF2, 0, 2),
        )
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_q,
            0,
            cta_layout,
            cute.group_modes(sApStore, 0, 2),
            cute.group_modes(gAp, 0, 2),
        )

        # TMEM accumulators: noise (2 stages x nblk column blocks) + peel.
        if cutlass.const_expr(self.direct):
            # The interleaved M=64 fragment has no derivable stage stride
            # (make_fragment_C leaves it dynamic), so stage the one-slot
            # layout by hand: one 128-column panel per stage.
            acc_shape_noise = tiled_mma_noise.partition_shape_C((64, 128))
            fake_one = tiled_mma_noise.make_fragment_C(cute.append(acc_shape_noise, 1))
            tCtNoise_fake = cute.make_tensor(
                fake_one.iterator,
                cute.make_layout(
                    (*fake_one.layout.shape[:3], self.acc_stages),
                    stride=(*fake_one.layout.stride[:3], 128),
                ),
            )
            noise_cols = 128 * self.acc_stages
        else:
            acc_shape_noise = tiled_mma_noise.partition_shape_C((self.bk_mma, self.rows))
            noise_slots = self.acc_stages * self.nblk
            tCtNoise_fake = tiled_mma_noise.make_fragment_C(
                cute.append(acc_shape_noise, noise_slots)
            )
            noise_cols = compute_tmem_cols_from_layout(tCtNoise_fake.layout, Float32)
        acc_shape_peel = tiled_mma_peel.partition_shape_C((64, R))
        tCtPeel_fake = tiled_mma_peel.make_fragment_C(cute.append(acc_shape_peel, 1))
        peel_cols = compute_tmem_cols_from_layout(tCtPeel_fake.layout, Float32)
        total_cols = max(32, 1 << (int(noise_cols) + int(peel_cols) - 1).bit_length())

        if cutlass.const_expr(self.direct):
            # 16-dp loads: the M=64 accumulator interleaves rows into lane
            # quads (row r -> lane r%16 + 32*(r//16)), so warp w's quad holds
            # rows 16w..16w+15 -- exactly its A half-tile's rows.
            tiled_t2r_noise = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition(16)), Float32),
                tCtNoise_fake[((None, None), 0, 0, 0)],
            )
        else:
            tiled_t2r_noise = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(self.rows)), Float32),
                tCtNoise_fake[((None, None), 0, 0, 0)],
            )
        tiled_t2r_peel = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld16x128bOp(tcgen05.Repetition(4)), Float32),
            tCtPeel_fake[((None, None), 0, 0, 0)],
        )

        pipeline_init_wait(cluster_shape_mn=(1, 1))

        if warp_idx == self.producer_warp:
            self._run_producer_warp_sm100(
                tma_aq,
                tma_as,
                tma_f1,
                tma_f2,
                tAsAq,
                tAgAq,
                tAsAs,
                tAgAs,
                tFsF1,
                tFgF1,
                tFsF2,
                tFgF2,
                factor_pipe,
                f2_pipe,
                a_pipe,
                first_row_half,
                tile_count,
            )
        elif warp_idx == self.output_warp:
            self._run_output_warp_sm100(
                tma_q,
                tQsQ,
                tQgQ,
                mE1,
                mKey,
                sE1,
                sE1B,
                output_pipe,
                lane,
                row_block,
                tile_count,
            )
        elif warp_idx == self.mma_warp:
            # The (prologue-idle) MMA warp is the TMEM allocator, so the
            # alloc overlaps the consumer stats/E1 prologue. E1_READY is the
            # pointer's publication point; the end-of-kernel barrier below
            # fences every consumer's last TMEM read before the free.
            tmem.allocate(total_cols)
            tmem.relinquish_alloc_permit()
            if cutlass.const_expr(self.direct):
                tCrE1 = tiled_mma_noise.make_fragment_A(sE1B)
                tCrF1 = tiled_mma_noise.make_fragment_B(sF1)
            else:
                tCrF1 = tiled_mma_noise.make_fragment_A(sF1)
                tCrE1 = tiled_mma_noise.make_fragment_B(sE1B)
            tCrPa = tiled_mma_peel.make_fragment_A(sApE4)
            tCrPb = tiled_mma_peel.make_fragment_B(sF2)
            self._run_mma_warp(
                tiled_mma_noise,
                tiled_mma_peel,
                tCrF1,
                tCrE1,
                tCrPa,
                tCrPb,
                tmem,
                tCtNoise_fake,
                tCtPeel_fake,
                noise_cols,
                factor_pipe,
                f2_pipe,
                acc_pipe,
                peel_pipe,
                peel_done,
                tile_count,
            )
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem.retrieve_ptr(Float32))
        elif warp_idx < self.producer_warp:
            if cutlass.const_expr(self.direct):
                self._run_consumer_warpgroup_direct(
                    _ConsumerGlobals(mAp, mAlpha, mBeta, mPeel, mStat),
                    sAq,
                    sAs,
                    sApU16,
                    sE1,
                    sAlpha,
                    sBeta,
                    tiled_t2r_noise,
                    # Retired path (statically off): predates the MMA-warp
                    # TMEM alloc scheme and would need its own retrieve.
                    tCtNoise_fake,
                    tiled_t2r_peel,
                    tCtPeel_fake[((None, None), 0, 0, 0)],
                    a_pipe,
                    acc_pipe,
                    output_pipe,
                    peel_pipe,
                    peel_done,
                    _ConsumerCoords(
                        thread_idx,
                        lane,
                        row_block,
                        tile_count,
                    ),
                )
            else:
                self._run_consumer_warpgroup_sm100(
                    _ConsumerGlobals(mAp, mAlpha, mBeta, mPeel, mStat),
                    tiled_mma_n,
                    tiled_mma_u16,
                    _ConsumerShared(
                        sAq,
                        sAs,
                        sF1,
                        sF2,
                        sApU16,
                        sE1,
                        sE1,  # sE1i slot unused on SM100
                        sAlpha,
                        sBeta,
                    ),
                    mE1,
                    mKey,
                    sE1B,
                    sNoise,
                    tiled_t2r_noise,
                    tmem,
                    tCtNoise_fake,
                    tCtPeel_fake,
                    noise_cols,
                    tiled_t2r_peel,
                    a_pipe,
                    acc_pipe,
                    output_pipe,
                    peel_pipe,
                    peel_done,
                    _ConsumerCoords(
                        thread_idx,
                        lane,
                        row_block,
                        tile_count,
                    ),
                )
            tmem_alloc_barrier.arrive()
