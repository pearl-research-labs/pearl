"""SM120 (RTX PRO 6000 / GeForce Blackwell) variant of the fused stats, noising,
quantization, and peel kernel.

The lottery-critical noise dot runs on the warp-level ``mma.sync`` e4m3 atom
(``warp.MmaFP8Op`` m16n8k32), whose K = 32 datapath is bit-identical to the
``tcgen05`` kind::f8f6f4 atom the protocol reference models
(``tests/test_noisy_quant.py`` gates A' against the pinned digests), so the
noise term entering the quantize chain is bit-exact against SM100 by
construction:

- the four consumer warps compute
  ``noise (16 rows x bk) = E1_half (16 x K, e4m3) @ F1_tile^T (K x bk, e4m3)``
  per row half per k-tile straight into register accumulators, K =
  ``PACKED_NOISE_K`` (32) being the atom's contraction depth: one accumulator
  chain per output, from +0, over exactly the reference's zero-padded K = 32
  product;
- the noise tiled MMA carries the same ``(1, consumer_warps, 1)`` atom layout
  and pair-interleaving N permutation as the consumer fragment scaffolding
  (``tiled_mma_n``), so each thread's accumulator IS its quantize-chain noise
  fragment. There is no TMEM drain, no f32 staging tile and no cross-warp
  barrier per tile; the bf16x2 noise pairs are packed from the accumulator
  registers with the same single ``cvt.rn.bf16x2.f32`` rounding, so A' codes
  stay bit-identical to the reference;
- the ``A' @ F2^T`` peel is register-direct: exact packed e4m3 -> f16
  widenings of the still-live A' fragments feed warp-level f16 MMAs with f32
  accumulators (every e4m3 value is exact in f16, so the products are exact),
  each warp accumulating its own 16-column strips of every k-tile. The four
  warp partials are combined in a fixed order (the reference peel is a
  near-exact f32 path gated at tolerance, not bit-exact).

There is no MMA warp, no TMEM and no wide 8-warp consumer group: the block is
the four consumer warps plus the TMA producer warp and the E1/A'-store warp.
The operand tiles are staged through the Hopper K-major swizzle atoms, which
both the TMA and ``ldmatrix`` address natively.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .._utils._arch import Arch
from ..protocol_constants import BLOCK_SCALE_GROUP, PACKED_NOISE_K, PEEL_COLS, R
from ._kernel import (
    _OUT_STAGES,
    HALF_TILE_ROWS,
    WG_THREADS,
    NoiseLoadMode,
    _ConsumerCoords,
    _ConsumerGlobals,
    _ConsumerShared,
    _NamedBarrier,
    _NoisyQuant,
)
from ._quantization_ops import _e4m3x2_to_f16x2, _f32x2_to_bf16x2

_SMEM_CAPACITY_BYTES = Arch.SM120.smem_capacity_bytes
# The warp-level atoms: the e4m3 noise dot and the f16 register-direct peel.
_NOISE_ATOM_MNK = (16, 8, PACKED_NOISE_K)
_PEEL_ATOM_MNK = (16, 8, 16)
_ATOM_N = 8  # both atoms are m16n8
_PEEL_STRIP_COLS = _PEEL_ATOM_MNK[2]  # one peel k-step = one 16-column A' strip
_C_FRAGMENT_VALUES = 4  # f32 values per thread per m16n8 C atom
# Per-thread peel accumulator of one 16-row half: 16 x R f32 over 32 lanes,
# published to the combining warp in float4 groups.
_PEEL_VALUES_PER_THREAD = 16 * R // 32
_PEEL_FRAGMENT_GROUPS = _PEEL_VALUES_PER_THREAD // _C_FRAGMENT_VALUES


class _NoisyQuantSm120(_NoisyQuant):
    """Warp-specialized noising at 16-row granularity on the ``mma.sync`` e4m3 atom."""

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
        super().__init__(bk, stages, out_stages, msg_base, consts, load_mode, rows)
        # mma.sync is warp-level: no MMA warp, no TMEM drain, and no wide
        # 8-warp consumer group (its footprint win was TMEM-drain specific).
        self.wide = False
        self.drain_split = "none"
        self.consumer_warps = 4
        self.consumer_threads = WG_THREADS
        self.n_group = 16 * self.consumer_warps
        self.producer_warp = self.consumer_warps
        self.output_warp = self.consumer_warps + 1
        self.threads = self.consumer_threads + 64
        # The noise MMA consumes F1 as one whole (bk, PACKED_NOISE_K) operand
        # tile per stage (no 128-row UMMA panels).
        self.bk_mma = self.bk
        self.nblk = 1
        # E1_READY participants: the consumers plus the E1 generator (the
        # output warp), unless the consumers own E1 themselves.
        self.e1_ready_threads = self.consumer_threads + (0 if self.consumer_e1 else 32)
        self.peel_partial_warps = self.consumer_warps - 1

    # -- device helpers -------------------------------------------------------

    @cute.jit
    def _fence_stage_reads(self):
        """Order this warp's ldmatrix / LDS reads of a TMA stage before the
        elected lane's release arrive (a generic-proxy read followed by an
        async-proxy write has no ordering otherwise); Quack's SM120 mainloop
        convention, shared with ``_FusedGemmSm120._issue_peel``."""
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

    @cute.jit
    def _e1_operand_row(self, sE1B, local_row):
        """One row of the plain K-major (rows, PACKED_NOISE_K) E1 operand tile."""
        return sE1B[(local_row, None)]

    @cute.jit
    def _accumulate_register_direct_peel(
        self,
        shared_f2_partition,
        f2_codes,
        f2_values,
        quantized_word_fragments,
        peel_a_words,
        peel_a_fragment,
        peel_accumulators,
        tiled_mma_p,
        consumer_warp,
        f2_stage,
    ):
        """Accumulate this warp's peel columns directly from the A' registers.

        Warp ``w`` owns physical columns ``16w..16w+15`` of every
        ``n_group``-column group (the noise N permutation), so its k-steps
        are the strips ``w + consumer_warps * group``.
        """
        row_halves = self.rows // HALF_TILE_ROWS
        f2_code_words = cute.recast_tensor(f2_codes, cutlass.Uint16)
        f2_value_words = cute.recast_tensor(f2_values, cutlass.Uint32)
        for column_group in cutlass.range_constexpr(self.bk // self.n_group):
            factor_k_block = consumer_warp + self.consumer_warps * column_group
            cute.autovec_copy(
                shared_f2_partition[(None, None, factor_k_block, f2_stage)],
                f2_codes,
            )
            # Packed exact widening: every e4m3 value is representable in f16,
            # so one cvt per pair replaces the scalar f32 round trip.
            for word in cutlass.range_constexpr(cute.size(f2_value_words)):
                f2_value_words[word] = _e4m3x2_to_f16x2(f2_code_words[word])
            for row_half in cutlass.range_constexpr(row_halves):
                for packed_pair in cutlass.range_constexpr(4):
                    # The quantized words are stored in the u16 copy order;
                    # the MMA A operand wants (row-half, k-half) swapped.
                    word = 4 * column_group + ((packed_pair & 1) * 2 + (packed_pair >> 1))
                    peel_a_words[packed_pair] = _e4m3x2_to_f16x2(
                        quantized_word_fragments[row_half][word]
                    )
                cute.gemm(
                    tiled_mma_p,
                    peel_accumulators[row_half],
                    peel_a_fragment[(None, None, 0)],
                    f2_values,
                    peel_accumulators[row_half],
                )

    # noqa C901: the fixed-order cross-warp reduction. The nesting is the
    # reduction order itself, which is load-bearing for determinism.
    @cute.jit
    def _reduce_peel_warps(  # noqa: C901
        self,
        peel_accumulators,
        shared_partials: cute.Tensor,
        warp_partial: cute.Tensor,
        consumer_warp: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        """Combine the four warp-local peel fragments into warp 0, in
        deterministic warp order (warp 0, then 1, 2, 3), one row half at a
        time through the shared partials buffer."""
        row_halves = self.rows // HALF_TILE_ROWS
        for half in cutlass.range_constexpr(row_halves):
            if half > 0:
                # WAR guard: the buffer is reused per row half.
                cute.arch.barrier(
                    barrier_id=_NamedBarrier.CONSUMER,
                    number_of_threads=self.consumer_threads,
                )
            if consumer_warp > 0:
                for group in cutlass.range_constexpr(_PEEL_FRAGMENT_GROUPS):
                    for item in cutlass.range_constexpr(_C_FRAGMENT_VALUES):
                        warp_partial[item] = peel_accumulators[half][
                            _C_FRAGMENT_VALUES * group + item
                        ]
                    cute.autovec_copy(
                        warp_partial,
                        shared_partials[(None, lane, group, consumer_warp - 1)],
                    )
            cute.arch.barrier(
                barrier_id=_NamedBarrier.CONSUMER,
                number_of_threads=self.consumer_threads,
            )
            if consumer_warp == 0:
                for other_warp in cutlass.range_constexpr(self.peel_partial_warps):
                    for group in cutlass.range_constexpr(_PEEL_FRAGMENT_GROUPS):
                        cute.autovec_copy(
                            shared_partials[(None, lane, group, other_warp)],
                            warp_partial,
                        )
                        for item in cutlass.range_constexpr(_C_FRAGMENT_VALUES):
                            index = _C_FRAGMENT_VALUES * group + item
                            peel_accumulators[half][index] = (
                                peel_accumulators[half][index] + warp_partial[item]
                            )

    @cute.jit
    def _finalize_register_direct_peel(
        self,
        peel_accumulators,
        shared_partials,
        mPeel,
        lane,
        consumer_warp,
        first_fragment_row,
        row_block,
    ):
        """Combine the warp partials in their fixed order, then write the row
        block's AF2 columns (the beta * E1 half is written before the tile
        loop)."""
        bf16 = cutlass.BFloat16
        row_halves = self.rows // HALF_TILE_ROWS
        warp_partial = cute.make_rmem_tensor(_C_FRAGMENT_VALUES, Float32)
        self._reduce_peel_warps(
            peel_accumulators, shared_partials, warp_partial, consumer_warp, lane
        )

        peel_output = cute.local_tile(mPeel, (self.rows, PEEL_COLS), (row_block, 0))
        if consumer_warp == 0:
            # m16n8 C fragment: 4 values per 8-column N block, laid out
            # (column, row half) with lane & 3 selecting the column pair.
            for half in cutlass.range_constexpr(row_halves):
                for column_block in cutlass.range_constexpr(R // _ATOM_N):
                    for row_half in cutlass.range_constexpr(2):
                        for column in cutlass.range_constexpr(2):
                            fragment_index = (
                                column + 2 * row_half + _C_FRAGMENT_VALUES * column_block
                            )
                            local_row = 16 * half + first_fragment_row + 8 * row_half
                            peel_col = _ATOM_N * column_block + 2 * (lane & 3) + column
                            peel_output[(local_row, R + peel_col)] = peel_accumulators[half][
                                fragment_index
                            ].to(bf16)

    # -- consumers ------------------------------------------------------------

    # noqa C901: compile-time load-mode dispatch in the hot tile loop.
    @cute.jit
    def _run_consumer_warpgroup_sm120(  # noqa: C901
        self,
        globals_: _ConsumerGlobals,
        tiled_mma_noise,
        tiled_mma_p,
        tiled_mma_n,
        tiled_mma_u16,
        shared: _ConsumerShared,
        shared_peel_partials: cute.Tensor,
        mE1,
        mKey,
        sE1B,
        factor_pipe,
        f2_pipe,
        a_pipe,
        output_pipe,
        coords: _ConsumerCoords,
    ):
        """Stats, register noise dot, decode, quantize, and register-direct peel."""
        mAlpha, mBeta = globals_.alpha, globals_.beta
        mPeel, mStat = globals_.peel, globals_.stats
        sAq, sAs = shared.a_codes, shared.a_scales
        sF1, sF2, sAp = shared.f1, shared.f2, shared.a_prime
        sE1, sAlpha, sBeta = shared.e1, shared.alpha, shared.beta
        thread_idx, lane = coords.thread, coords.lane
        row_block, tile_count = coords.row_block, coords.tile_count
        row_halves = self.rows // HALF_TILE_ROWS
        bf16 = cutlass.BFloat16
        f16 = cutlass.Float16
        f8 = cutlass.Float8E4M3FN

        # -- consumer fragment scaffolding (identical to SM100) --
        pair_template = cute.make_identity_tensor((16, self.bk // 2))
        pair_fragment_shape = tiled_mma_u16.get_slice(thread_idx).partition_C(pair_template).shape
        code_words = cute.make_rmem_tensor(pair_fragment_shape, cutlass.Uint16)
        quantized_word_fragments = [
            cute.make_rmem_tensor(pair_fragment_shape, cutlass.Uint16) for _ in range(row_halves)
        ]
        a_fragment_shape = (
            tiled_mma_n.get_slice(thread_idx)
            .partition_C(cute.make_identity_tensor((16, self.bk)))
            .shape
        )
        a_fragment = cute.make_rmem_tensor(a_fragment_shape, bf16)
        a_words = cute.recast_tensor(a_fragment, cutlass.Uint32)
        noise_words = cute.make_rmem_tensor(4 * self.bk // self.n_group, cutlass.Uint32)

        consumer_warp = cute.arch.make_warp_uniform(thread_idx // 32)
        scale_group_base = 2 * consumer_warp + ((lane >> 1) & 1)
        first_fragment_row = lane >> 2

        # -- noise MMA operands: E1 (A, resident per row half) and F1 (B, per
        # stage), both ldmatrix-fed; the accumulator is one (16, bk) row half.
        # Its fragment order is (column pair, row half, N block), the very
        # (c, h, nb) decomposition the quantize chain reads (probe-verified
        # against tiled_mma_n's coordinates for every family).
        thr_noise = tiled_mma_noise.get_slice(thread_idx)
        ldmatrix_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), f8)
        tiled_copy_e1 = cute.make_tiled_copy_A(ldmatrix_atom, tiled_mma_noise)
        thr_copy_e1 = tiled_copy_e1.get_slice(thread_idx)
        sE1B_halves = cute.make_tensor(
            sE1B.iterator,
            cute.make_layout(
                (16, PACKED_NOISE_K, row_halves),
                stride=(PACKED_NOISE_K, 1, 16 * PACKED_NOISE_K),
            ),
        )
        shared_e1_copy = thr_copy_e1.partition_S(sE1B_halves)
        shared_e1_mma = thr_noise.partition_A(sE1B_halves)
        e1_fragments = [
            thr_noise.make_fragment_A(shared_e1_mma[(None, None, None, row_half)])
            for row_half in range(row_halves)
        ]
        e1_copy_views = [thr_copy_e1.retile(fragment) for fragment in e1_fragments]
        tiled_copy_f1 = cute.make_tiled_copy_B(ldmatrix_atom, tiled_mma_noise)
        thr_copy_f1 = tiled_copy_f1.get_slice(thread_idx)
        shared_f1_copy = thr_copy_f1.partition_S(sF1)
        f1_fragment = thr_noise.make_fragment_B(thr_noise.partition_B(sF1)[(None, None, None, 0)])
        f1_copy_view = thr_copy_f1.retile(f1_fragment)
        noise_acc = cute.make_rmem_tensor(thr_noise.partition_shape_C((16, self.bk)), Float32)

        # -- register-direct peel scaffolding (see the module docstring for
        # why the peel is f16 MMA). The A operand is one m16k16 A' fragment
        # per k-step, shaped from a (16, 16) f16 template.
        peel_thread = tiled_mma_p.get_slice(lane)
        peel_a_fragment = cute.make_fragment_like(
            peel_thread.partition_A(
                cute.make_tensor(
                    sE1.iterator,
                    cute.make_layout((16, _PEEL_STRIP_COLS), stride=(R, 1)),
                )
            ),
            f16,
        )
        peel_a_words = cute.recast_tensor(peel_a_fragment, cutlass.Uint32)
        shared_f2_partition = peel_thread.partition_B(sF2)
        f2_codes = cute.make_fragment_like(shared_f2_partition[(None, None, 0, 0)], f8)
        f2_values = cute.make_fragment_like(shared_f2_partition[(None, None, 0, 0)], f16)
        peel_accumulator_shape = peel_thread.partition_C(cute.make_identity_tensor((16, R))).shape
        peel_accumulators = [
            cute.make_rmem_tensor(peel_accumulator_shape, Float32) for _ in range(row_halves)
        ]

        a_copy_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), cutlass.Uint16)
        tiled_copy_aq = cute.make_tiled_copy_C(a_copy_atom, tiled_mma_u16)
        shared_code_partition = tiled_copy_aq.get_slice(thread_idx).partition_S(sAq)
        output_copy_atom = cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), cutlass.Uint16)
        tiled_copy_output = cute.make_tiled_copy_C(output_copy_atom, tiled_mma_u16)
        shared_output_partition = tiled_copy_output.get_slice(thread_idx).partition_D(sAp)

        if thread_idx < self.rows:
            global_row = row_block * self.rows + thread_idx
            alpha, beta = self._combine_row_stats(mStat, global_row, tile_count * self.bk)
            sAlpha[thread_idx] = alpha
            sBeta[thread_idx] = beta
            mAlpha[global_row] = alpha.to(bf16)
            mBeta[global_row] = beta.to(bf16)
        elif cutlass.const_expr(self.consumer_e1):
            # E1 on threads 64..127 (warps 2-3), overlapping the stats combine.
            if thread_idx >= 64:
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
            peel_accumulators[row_half].fill(0.0)

        # E1 is final past this barrier: load the resident A fragments once.
        cute.arch.barrier(
            barrier_id=_NamedBarrier.E1_READY,
            number_of_threads=self.e1_ready_threads,
        )
        for row_half in cutlass.range_constexpr(row_halves):
            cute.copy(
                tiled_copy_e1,
                shared_e1_copy[(None, None, None, row_half)],
                e1_copy_views[row_half],
            )

        # The bit-exact beta * E1 peel half needs only sE1 and sBeta, both
        # final here -- write it under the first tile's TMA latency.
        if thread_idx < self.rows:
            self._write_e1_peel_row(
                sE1,
                sBeta,
                cute.local_tile(
                    mPeel, (16, PEEL_COLS), (row_block * row_halves + thread_idx // 16, 0)
                ),
                thread_idx,
            )

        factor_read = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
        f2_read = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
        output_write_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.out_stages
        )
        for tile_index in cutlass.range(tile_count, unroll=1):
            # This tile's F1 operand: ldmatrix the whole (bk, K) B fragment,
            # then release the stage (its only reader) so the producer's
            # tile i+1 factor TMA is decoupled from this tile's quantize.
            factor_pipe.consumer_wait(factor_read)
            cute.copy(
                tiled_copy_f1,
                shared_f1_copy[(None, None, None, factor_read.index)],
                f1_copy_view,
            )
            self._fence_stage_reads()
            factor_pipe.consumer_release(factor_read)
            factor_read.advance()

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
                # Noise dot: noise (16, bk) = E1_half @ F1_tile^T, K = 32,
                # one atom per output from +0 -- the pinned arithmetic.
                noise_acc.fill(0.0)
                cute.gemm(
                    tiled_mma_noise,
                    noise_acc,
                    e1_fragments[row_half],
                    f1_fragment,
                    noise_acc,
                )
                # Noise pairs straight from the accumulator: fragment
                # (c, h, nb) at 4 * nb + 2 * h + c, and the same single
                # cvt.rn.bf16x2.f32 rounding as the SM100 staging path.
                for block in cutlass.range_constexpr(2 * self.bk // self.n_group):
                    for fragment_half in cutlass.range_constexpr(2):
                        pair = _C_FRAGMENT_VALUES * block + 2 * fragment_half
                        noise_words[fragment_half + 2 * block] = _f32x2_to_bf16x2(
                            noise_acc[pair + 1],
                            noise_acc[pair],
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
                        self._fence_stage_reads()
                        self._release_a_tile(a_pipe, tile_index)
                else:
                    self._fence_stage_reads()
                    self._release_a_half_tile(a_pipe, tile_index, row_half)

                if cutlass.const_expr(row_half == 0):
                    output_pipe.producer_acquire(output_write_state)
                cute.copy(
                    tiled_copy_output,
                    tiled_copy_output.retile(quantized_word_fragments[row_half])[(None, 0, None)],
                    shared_output_partition[(None, row_half, None, output_write_state.index)],
                )
                if cutlass.const_expr(row_half == row_halves - 1):
                    cute.arch.fence_proxy("async.shared", space="cta")
                    output_pipe.producer_commit(output_write_state)
                    output_write_state.advance()

            # Register-direct peel from the still-live quantized fragments
            # against this tile's staged F2 (exact f16 products, f32
            # accumulation -- see the module docstring).
            f2_pipe.consumer_wait(f2_read)
            self._accumulate_register_direct_peel(
                shared_f2_partition,
                f2_codes,
                f2_values,
                quantized_word_fragments,
                peel_a_words,
                peel_a_fragment,
                peel_accumulators,
                tiled_mma_p,
                consumer_warp,
                f2_read.index,
            )
            self._fence_stage_reads()
            f2_pipe.consumer_release(f2_read)
            f2_read.advance()

        self._finalize_register_direct_peel(
            peel_accumulators,
            shared_peel_partials,
            mPeel,
            lane,
            consumer_warp,
            first_fragment_row,
            row_block,
        )

    # -- launch ---------------------------------------------------------------

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

        # -- the warp-level MMA atoms the kernel executes --
        # The N permutation: each warp owns 16 contiguous columns per
        # n_group-column group, instance pairs interleaved at 2-column
        # granularity (see the SM100 scaffolding). Shared by the noise MMA
        # and the fragment scaffolding, which is what makes the accumulator
        # the quantize chain's noise fragment.
        noise_permutation = cute.make_layout((2, 4, self.consumer_warps, 2), stride=(1, 4, 16, 2))
        tiled_mma_noise = cute.make_tiled_mma(
            warp.MmaFP8Op(f8, Float32, _NOISE_ATOM_MNK),
            (1, self.consumer_warps, 1),
            permutation_mnk=(None, noise_permutation, None),
        )
        op16 = warp.MmaF16BF16Op(cutlass.Float16, Float32, _PEEL_ATOM_MNK)
        # Each warp computes all R peel columns for its own k-strips; the
        # warp partials are combined in fixed order. The K permutation
        # mirrors the noise N permutation within a 16-column strip (logical
        # b + 2m + 8h -> physical b + 4m + 2h), so the A' registers and the
        # staged F2 agree on which physical column each k step is.
        tiled_mma_p = cute.make_tiled_mma(
            op16,
            (1, 1, 1),
            permutation_mnk=(None, None, cute.make_layout((2, 4, 2), stride=(1, 4, 2))),
        )
        # -- consumer fragment scaffolding (identical to SM100, 4 warps) --
        tiled_mma_n = cute.make_tiled_mma(
            op16,
            (1, self.consumer_warps, 1),
            permutation_mnk=(None, noise_permutation, None),
        )
        tiled_mma_u16 = cute.make_tiled_mma(op16, (1, self.consumer_warps, 1))

        a_stage_count = (
            row_halves * k // bk if self.load_mode == NoiseLoadMode.RESIDENT else self.ring_a_stages
        )

        # A codes/scales staging is unchanged from SM100 (opened in registers).
        sAq_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.ROW_MAJOR, (16, 16, bk // 2), u16, a_stage_count
        )
        sAs_layout = cute.make_layout(
            (16, bk // BLOCK_SCALE_GROUP, a_stage_count),
            stride=(bk // BLOCK_SCALE_GROUP, 1, 16 * bk // BLOCK_SCALE_GROUP),
        )
        # mma.sync operand tiles, K-major through the Hopper swizzle atoms
        # (TMA-written, ldmatrix-read): F1 as (bk, K) noise B tiles, F2 as
        # (R, bk) peel B tiles. E1 is a plain (rows, K) tile the E1 generator
        # writes row by row.
        sF1_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.ROW_MAJOR, (bk, self.rows, PACKED_NOISE_K), f8, stages
        )
        sF2_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR, (64, R, bk), f8, stages
        )
        # A' staging is u16 pairs, stmatrix-written and TMA-stored from the
        # same swizzled image; the peel reads A' from registers, so no MMA
        # operand layout (and no 64-row pad) is needed. A stage holds a
        # whole tile: one handshake per tile.
        sAp_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.ROW_MAJOR, (self.rows, 16, bk // 2), u16, out_stages
        )

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
        tma_f1, tma_tf1 = _tma_load(mF1e4, sF1_layout, (bk, PACKED_NOISE_K))
        tma_f2, tma_tf2 = _tma_load(mF2, sF2_layout, (R, bk))
        tma_q, tma_tq = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mApU16,
            cute.slice_(sAp_layout, (None, None, 0)),
            (self.rows, bk // 2),
        )

        factor_tx_bytes = cute.size_in_bytes(f8, cute.slice_(sF1_layout, (None, None, 0)))
        f2_tx_bytes = cute.size_in_bytes(f8, cute.slice_(sF2_layout, (None, None, 0)))
        a_tile_bytes = cute.size_in_bytes(
            u16, cute.slice_(sAq_layout, (None, None, 0))
        ) + cute.size_in_bytes(u16, cute.slice_(sAs_layout, (None, None, 0)))

        @cute.struct
        class SharedStorage:
            factor_mbar: cute.struct.MemRange[cutlass.Int64, stages * 2]
            f2_mbar: cute.struct.MemRange[cutlass.Int64, stages * 2]
            a_mbar: cute.struct.MemRange[cutlass.Int64, a_stage_count * 2]
            output_mbar: cute.struct.MemRange[cutlass.Int64, out_stages * 2]
            sAlpha: cute.struct.MemRange[Float32, 16 * row_halves]
            sBeta: cute.struct.MemRange[Float32, 16 * row_halves]
            # The publishing warps' peel partials (float4 groups); warp 0 combines.
            sPeelPartials: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, self.peel_partial_warps * 32 * _PEEL_VALUES_PER_THREAD
                ],
                16,
            ]
            sE1: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, 16 * row_halves * R], 128]
            sE1B: cute.struct.Align[cute.struct.MemRange[f8, self.rows * PACKED_NOISE_K], 128]
            sAq: cute.struct.Align[cute.struct.MemRange[u16, cute.cosize(sAq_layout)], 1024]
            sAs: cute.struct.Align[cute.struct.MemRange[u16, cute.cosize(sAs_layout)], 1024]
            sF1: cute.struct.Align[cute.struct.MemRange[f8, cute.cosize(sF1_layout)], 1024]
            sF2: cute.struct.Align[cute.struct.MemRange[f8, cute.cosize(sF2_layout)], 1024]
            sAp: cute.struct.Align[cute.struct.MemRange[u16, cute.cosize(sAp_layout)], 1024]

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
            tiled_mma_p,
            tiled_mma_n,
            tiled_mma_u16,
            sAq_layout,
            sAs_layout,
            sF1_layout,
            sF2_layout,
            sAp_layout,
            factor_tx_bytes,
            f2_tx_bytes,
            a_tile_bytes,
        ).launch(
            grid=grid,
            block=[self.threads, 1, 1],
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
        tiled_mma_p: cute.TiledMma,
        tiled_mma_n: cute.TiledMma,
        tiled_mma_u16: cute.TiledMma,
        sAq_layout: cute.ComposedLayout,
        sAs_layout: cute.Layout,
        sF1_layout: cute.ComposedLayout,
        sF2_layout: cute.ComposedLayout,
        sAp_layout: cute.ComposedLayout,
        factor_tx_bytes: cutlass.Constexpr,
        f2_tx_bytes: cutlass.Constexpr,
        a_tile_bytes: cutlass.Constexpr,
    ):
        bk = self.bk
        stages = self.stages
        row_halves = self.rows // HALF_TILE_ROWS

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.producer_warp:
            self._prefetch_tma_descriptors(tma_aq, tma_as, tma_f1, tma_f2, tma_q)

        row_block, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        lane = thread_idx % 32

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_warpgroup = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.consumer_threads)
        # Every consumer warp waits on and releases every factor and A
        # stage (TMA-async releases arrive once per calling warp).
        warp_consumer = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.consumer_warps)
        output_consumer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)

        factor_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.factor_mbar.data_ptr(),
            num_stages=stages,
            producer_group=producer_group,
            consumer_group=warp_consumer,
            tx_count=factor_tx_bytes,
            defer_sync=True,
        )
        f2_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.f2_mbar.data_ptr(),
            num_stages=stages,
            producer_group=producer_group,
            consumer_group=warp_consumer,
            tx_count=f2_tx_bytes,
            defer_sync=True,
        )
        a_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_mbar.data_ptr(),
            num_stages=self.a_pipe_stages if self.a_tile_pipe else self.a_stage_count,
            producer_group=producer_group,
            consumer_group=warp_consumer,
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
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sAq = storage.sAq.get_tensor(sAq_layout.outer, swizzle=sAq_layout.inner)
        sAs = storage.sAs.get_tensor(sAs_layout)
        sF1 = storage.sF1.get_tensor(sF1_layout.outer, swizzle=sF1_layout.inner)
        sF2 = storage.sF2.get_tensor(sF2_layout.outer, swizzle=sF2_layout.inner)
        sAp = storage.sAp.get_tensor(sAp_layout.outer, swizzle=sAp_layout.inner)
        sE1B = storage.sE1B.get_tensor(
            cute.make_layout((self.rows, PACKED_NOISE_K), stride=(PACKED_NOISE_K, 1))
        )
        sE1 = storage.sE1.get_tensor(cute.make_layout((16, R, row_halves), stride=(R, 1, 16 * R)))
        sAlpha = storage.sAlpha.get_tensor(cute.make_layout(self.rows))
        sBeta = storage.sBeta.get_tensor(cute.make_layout(self.rows))
        shared_peel_partials = storage.sPeelPartials.get_tensor(
            cute.make_layout(
                (_C_FRAGMENT_VALUES, 32, _PEEL_FRAGMENT_GROUPS, self.peel_partial_warps),
                stride=(
                    1,
                    _C_FRAGMENT_VALUES,
                    _C_FRAGMENT_VALUES * 32,
                    _C_FRAGMENT_VALUES * 32 * _PEEL_FRAGMENT_GROUPS,
                ),
            )
        )

        gAq = cute.local_tile(mAqCodes, (16, bk // 2), (None, None))
        gAs = cute.local_tile(mAqScales, (16, bk // BLOCK_SCALE_GROUP), (None, None))
        gAp = cute.local_tile(mAp, (self.rows, bk // 2), (None, None))
        gF1 = cute.local_tile(mF1e4, (bk, PACKED_NOISE_K), (None, 0))  # (bk, K, kt)
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
            tma_f1, 0, cta_layout, cute.group_modes(sF1, 0, 2), cute.group_modes(gF1, 0, 2)
        )
        tFsF2, tFgF2 = cpasync.tma_partition(
            tma_f2, 0, cta_layout, cute.group_modes(sF2, 0, 2), cute.group_modes(gF2, 0, 2)
        )
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_q, 0, cta_layout, cute.group_modes(sAp, 0, 2), cute.group_modes(gAp, 0, 2)
        )

        pipeline_init_wait(cluster_shape_mn=(1, 1))

        if warp_idx == self.producer_warp:
            self._run_producer_warp(
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
            self._run_output_warp(
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
        elif warp_idx < self.producer_warp:
            self._run_consumer_warpgroup_sm120(
                _ConsumerGlobals(mAp, mAlpha, mBeta, mPeel, mStat),
                tiled_mma_noise,
                tiled_mma_p,
                tiled_mma_n,
                tiled_mma_u16,
                _ConsumerShared(
                    sAq,
                    sAs,
                    sF1,
                    sF2,
                    sAp,
                    sE1,
                    sE1,  # e1_pairs slot unused on SM120
                    sAlpha,
                    sBeta,
                ),
                shared_peel_partials,
                mE1,
                mKey,
                sE1B,
                factor_pipe,
                f2_pipe,
                a_pipe,
                output_pipe,
                _ConsumerCoords(thread_idx, lane, row_block, tile_count),
            )
