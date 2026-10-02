"""SM120 (RTX PRO 6000 / GeForce Blackwell) variant of the fused stats, noising,
quantization, and peel kernel.

The lottery-critical noise dot runs on the warp-level ``mma.sync`` e4m3 atom
(``warp.MmaFP8Op`` m16n8k32), whose K = 32 datapath is bit-identical to the
``tcgen05`` kind::f8f6f4 atom the verifier's ``B200`` replay models
(``zk-pow/src/api/fp8/utils.rs``), so the noise term entering the quantize
chain is bit-exact against SM100 by construction: the four consumer warps
compute ``noise (16 rows x bk) = E1_half (16 x K, e4m3) @ F1_tile^T (K x bk,
e4m3)`` per row half per k-tile straight into register accumulators, one
accumulator chain per output from +0. The noise tiled MMA carries the same
``(1, consumer_warps, 1)`` atom layout and pair-interleaving N permutation as
the consumer fragment scaffolding (``tiled_mma_n``), so each thread's
accumulator IS its quantize-chain noise fragment: there is no TMEM drain, no
f32 staging tile, and no cross-warp barrier per tile.

The topology, the quantize chain and the register-direct peel are
``_kernel_register_peel._NoisyQuantRegisterPeel``'s; this module supplies the noise
dot and the plain ``(rows, K)`` E1 operand tile. The operand tiles are staged
through the Hopper K-major swizzle atoms, which both the TMA and ``ldmatrix``
address natively.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.nvgpu import warp

from .._utils._arch import Arch
from ..protocol_constants import PACKED_NOISE_K
from ._kernel_common import _C_FRAGMENT_VALUES, HALF_TILE_ROWS, _noise_n_permutation
from ._kernel_register_peel import _NoisyQuantRegisterPeel
from ._quantization_ops import _f32x2_to_bf16x2

_NOISE_ATOM_MNK = (16, 8, PACKED_NOISE_K)  # the warp-level e4m3 noise atom


class _NoiseOperands(NamedTuple):
    tiled_mma: cute.TiledMma
    e1_copy: cute.TiledCopy
    e1_sources: cute.Tensor
    e1_views: list
    e1_fragments: list
    f1_copy: cute.TiledCopy
    f1_sources: cute.Tensor
    f1_view: cute.Tensor
    f1_fragment: cute.Tensor
    accumulator: cute.Tensor


class _NoisyQuantSm120(_NoisyQuantRegisterPeel):
    """Warp-specialized noising at 16-row granularity on the ``mma.sync`` e4m3 atom."""

    _smem_capacity_bytes = Arch.SM120.smem_capacity_bytes

    def _make_noise_mma(self):
        """The e4m3 ``mma.sync`` noise MMA, sharing ``tiled_mma_n``'s layout."""
        return cute.make_tiled_mma(
            warp.MmaFP8Op(cutlass.Float8E4M3FN, Float32, _NOISE_ATOM_MNK),
            (1, self.consumer_warps, 1),
            permutation_mnk=(None, _noise_n_permutation(self.consumer_warps), None),
        )

    def _e1_operand_layout(self):
        """A plain K-major ``(rows, PACKED_NOISE_K)`` tile the generator writes row by row."""
        return cute.make_layout((self.rows, PACKED_NOISE_K), stride=(PACKED_NOISE_K, 1))

    def _noise_staging_storage(self):
        """None: the quantize chain reads the accumulator registers."""
        return None

    def _noise_smem_views(self, storage):
        sE1B = storage.sE1B.get_tensor(self._e1_operand_layout())
        return sE1B, (sE1B,)

    def _make_noise_operands(self, tiled_mma_noise, fragment_coords, thread_idx, sF1, noise_smem):
        """E1 (A, resident per row half) and F1 (B, per stage), both
        ldmatrix-fed; the accumulator is one (16, bk) row half. Its fragment
        order is (column pair, row half, N block), the very (c, h, nb)
        decomposition the quantize chain reads (probe-verified against
        tiled_mma_n's coordinates for every family)."""
        (sE1B,) = noise_smem
        row_halves = self.rows // HALF_TILE_ROWS
        f8 = cutlass.Float8E4M3FN
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
        return _NoiseOperands(
            tiled_mma_noise,
            tiled_copy_e1,
            shared_e1_copy,
            e1_copy_views,
            e1_fragments,
            tiled_copy_f1,
            shared_f1_copy,
            f1_copy_view,
            f1_fragment,
            noise_acc,
        )

    @cute.jit
    def _load_e1_operand(self, noise):
        """ldmatrix the resident E1 A fragments once."""
        for row_half in cutlass.range_constexpr(self.rows // HALF_TILE_ROWS):
            cute.copy(
                noise.e1_copy,
                noise.e1_sources[(None, None, None, row_half)],
                noise.e1_views[row_half],
            )

    @cute.jit
    def _stage_noise_tile(self, noise, factor_pipe, factor_read):
        """ldmatrix the whole (bk, K) F1 B fragment, then release its stage
        (the fragment is its only reader)."""
        factor_pipe.consumer_wait(factor_read)
        cute.copy(
            noise.f1_copy,
            noise.f1_sources[(None, None, None, factor_read.index)],
            noise.f1_view,
        )
        self._fence_stage_reads()
        factor_pipe.consumer_release(factor_read)

    @cute.jit
    def _noise_words(self, noise, row_half, first_fragment_row, noise_words):
        """noise (16, bk) = E1_half @ F1_tile^T, K = 32, one atom per output
        from +0 -- the pinned arithmetic -- then the bf16x2 pairs straight
        from the accumulator: fragment (c, h, nb) at 4 * nb + 2 * h + c, one
        cvt.rn.bf16x2.f32 per pair (the reference's single RNE rounding)."""
        noise_acc = noise.accumulator
        noise_acc.fill(0.0)
        cute.gemm(
            noise.tiled_mma,
            noise_acc,
            noise.e1_fragments[row_half],
            noise.f1_fragment,
            noise_acc,
        )
        for block in cutlass.range_constexpr(2 * self.bk // self.n_group):
            for fragment_half in cutlass.range_constexpr(2):
                pair = _C_FRAGMENT_VALUES * block + 2 * fragment_half
                noise_words[fragment_half + 2 * block] = _f32x2_to_bf16x2(
                    noise_acc[pair + 1],
                    noise_acc[pair],
                )
