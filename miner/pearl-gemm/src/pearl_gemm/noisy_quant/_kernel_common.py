"""Architecture-neutral phases of the fused stats, noising, quantization, and peel kernel.

``_NoisyQuant`` is the base every family subclasses: the A staging and
decode, the TMA producer warp, the E1 / A'-store output warp, and the packed
quantize chain. The TMEM-free topology SM90 and SM120 share is
``_kernel_register_peel._NoisyQuantRegisterPeel``.
"""

import enum
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32
from quack.cute_dsl_utils import mlir_namedtuple

from ..protocol_constants import PACKED_NOISE_K, R
from ._quantization_ops import (
    _bf16x2_to_e4m3x2,
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
_NOISE_SKEW_ELEMS = 16  # the f32 noise staging tile's per-row skew (bank spread)
# The register-direct peel (SM90, SM120): four consumer warps, three of which
# publish their partials to warp 0.
_REGISTER_PEEL_WARPS = 4
_PEEL_PARTIAL_WARPS = _REGISTER_PEEL_WARPS - 1
# The register-direct peel's warp-level f16 atom (m16n8k16).
_PEEL_ATOM_MNK = (16, 8, 16)
_ATOM_N = _PEEL_ATOM_MNK[1]
_PEEL_STRIP_COLS = _PEEL_ATOM_MNK[2]  # one peel k-step = one 16-column A' strip
_C_FRAGMENT_VALUES = 4  # f32 values per thread per m16n8 C atom
# Per-thread peel accumulator of one 16-row half: 16 x R f32 over 32 lanes,
# published to the combining warp in float4 groups.
_PEEL_VALUES_PER_THREAD = 16 * R // 32
_PEEL_FRAGMENT_GROUPS = _PEEL_VALUES_PER_THREAD // _C_FRAGMENT_VALUES


def _noise_n_permutation(consumer_warps: int) -> cute.Layout:
    """The consumer fragment scaffolding's N permutation.

    Each warp owns 16 contiguous columns per ``16 * consumer_warps``-column
    group, with the instance pairs interleaved at 2-column granularity so a
    thread's four bytes per 16-column strip stay contiguous and its noise
    pairs stay column-adjacent. The quantize chain's word map is keyed to it.
    """
    return cute.make_layout((2, 4, consumer_warps, 2), stride=(1, 4, 16, 2))


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
    alpha: cute.Tensor
    beta: cute.Tensor


@mlir_namedtuple
class _ConsumerCoords(NamedTuple):
    thread: cutlass.Int32
    lane: cutlass.Int32
    row_block: cutlass.Int32
    tile_count: cutlass.Int32


class _NoisyQuant:
    """Shared phases of the warp-specialized noising kernel at 16-row granularity.

    Subclasses set the warp topology (``consumer_warps``, ``consumer_threads``,
    ``n_group``, ``producer_warp``, ``output_warp``, ``e1_ready_threads``,
    ``nblk``) and implement ``_e1_operand_row``, the consumers and the launch.
    """

    # Consumer-side E1 overlaps the BLAKE3 prologue with the stats combine.
    _consumer_e1_enabled = True

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
        # A MERGED request runs as a RING with the merged stage budget: A
        # rides its own pipe rather than the factor pipeline's barriers (on
        # SM100 the MMA warp alone consumes the factors).
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
        # half-tile. Native-RING pins keep the classic per-half-tile pipe.
        self.a_tile_pipe = was_merged
        self.a_pipe_stages = self.stages if self.a_tile_pipe else 0
        if self.a_tile_pipe:
            self.ring_a_stages = self.a_pipe_stages * row_halves
        self.consumer_e1 = self._consumer_e1_enabled and self.rows == 64

    # -- device helpers -----------------------------------------------------

    def _e1_operand_row(self, sE1B, local_row):
        """One row's K-padded codes in the family's noise-MMA E1 operand image."""
        raise NotImplementedError

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
    ):
        """Wait for one A half-tile's stage and return its shared stage index."""
        row_halves = self.rows // HALF_TILE_ROWS
        half_tile_index = row_halves * tile_index + row_half
        i32 = cutlass.Int32

        if cutlass.const_expr(self.load_mode == NoiseLoadMode.RESIDENT):
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
            stage = self._wait_a_half_tile(a_pipe, tile_index, row_half)
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
    def _quantize_fragment(
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
            # Pad the noise-MMA operand to the atom's K. Zero pad products are
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
            cute.autovec_copy(e1_padded, self._e1_operand_row(sE1B, local_row))
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
        """Stage one tile's F1 and F2 factors on their factor pipes.

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
        """Load A half-tiles (own ring) and F1/F2 factor stages (factor pipes)."""
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
    def _run_producer_warp(
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
        """Produce factor stages and A half-tiles.

        Architecture-neutral: the TMA producer is the same on every family.
        """
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
    def _run_output_warp(
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
        """Generate E1 (unless the consumers own it), then drain A'.

        Architecture-neutral: the E1 operand image differs per family only
        through ``_e1_operand_row``.
        """
        if cutlass.const_expr(not self.consumer_e1):
            for row_sweep in cutlass.range_constexpr((self.rows + 31) // 32):
                self._generate_e1_rows(mE1, mKey, sE1, sE1B, 32 * row_sweep, lane, row_block)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier_arrive(
                barrier_id=_NamedBarrier.E1_READY,
                number_of_threads=self.e1_ready_threads,
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
