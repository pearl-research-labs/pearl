"""Hopper (SM90) variant of the fused stats, noising, quantization, and peel kernel.

The noise dot runs on the Hopper QGMMA atom (``wgmma.mma_async`` e4m3), the
opcode the verifier's H100 replay (``zk-pow/src/api/fp8/utils.rs``) models,
so the noise term is bit-exact against the Hopper reference. The
``A' @ F2^T`` peel uses warp-level f16 MMAs with f32 accumulators instead:
the reference peel is a near-exact f32 path that QGMMA's 14-bit window
cannot match. The topology, the quantize chain and the register-direct peel
are ``_kernel_register_peel._NoisyQuantRegisterPeel``'s; this module supplies
the noise dot, the E1 operand tile, and the f32 staging tile the transposed
noise accumulator is read back through.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Boolean, Float32, Int32
from cutlass.cute.nvgpu import warpgroup
from quack.sm90_utils import gemm_w_idx

from .._utils._arch import Arch
from ..protocol_constants import PACKED_NOISE_K
from ._kernel_common import _C_FRAGMENT_VALUES, _NOISE_SKEW_ELEMS, _NamedBarrier
from ._kernel_register_peel import _NoisyQuantRegisterPeel
from ._quantization_ops import _f32x2_to_bf16x2

_E1_PANEL_BYTES = 16  # one K_INTER K panel: 16 e4m3 codes per row


class _NoiseOperands(NamedTuple):
    tiled_mma: cute.TiledMma
    f1_fragment: cute.Tensor
    e1_fragment: cute.Tensor
    accumulator: cute.Tensor
    # (bk, rows) coordinates of the accumulator fragment.
    accumulator_coords: cute.Tensor
    # (16, bk) coordinates of the quantize chain's noise fragment.
    fragment_coords: cute.Tensor
    staging: cute.Tensor


class _NoisyQuantSm90(_NoisyQuantRegisterPeel):
    """Warp-specialized noising at 16-row granularity on Hopper QGMMA."""

    _smem_capacity_bytes = Arch.SM90.smem_capacity_bytes

    # The output warp generates E1; consumer-side E1 is not tuned on SM90.
    _consumer_e1_enabled = False

    def _make_noise_mma(self):
        """One ``(bk, rows)`` QGMMA tile over the consumer warpgroup."""
        f8 = cutlass.Float8E4M3FN
        return sm90_utils.make_trivial_tiled_mma(
            f8,
            f8,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            (1, 1, 1),
            tiler_mn=(64, self.rows),
        )

    def _e1_operand_layout(self):
        """The WGMMA B tile in the unswizzled K_INTER atom, whose byte image
        has a plain-strided twin (16-byte rows, K panels every ``rows`` rows)."""
        inter_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_INTER, cutlass.Float8E4M3FN
        )
        return cute.tile_to_shape(inter_atom, (self.rows, PACKED_NOISE_K, 1), order=(0, 1, 2))

    def _noise_staging_storage(self):
        """The skewed f32 ``(rows, bk)`` tile the accumulators are transposed through."""
        return cute.struct.Align[
            cute.struct.MemRange[Float32, self.rows * (self.bk + _NOISE_SKEW_ELEMS)], 1024
        ]

    def _noise_smem_views(self, storage):
        sE1B_layout = self._e1_operand_layout()
        sE1B = storage.sE1B.get_tensor(sE1B_layout.outer, swizzle=sE1B_layout.inner)
        sE1B_rows = cute.make_tensor(
            storage.sE1B.data_ptr(),
            cute.make_layout(
                (self.rows, (_E1_PANEL_BYTES, PACKED_NOISE_K // _E1_PANEL_BYTES)),
                stride=(_E1_PANEL_BYTES, (1, _E1_PANEL_BYTES * self.rows)),
            ),
        )
        sNoise = storage.sNoise.get_tensor(
            cute.make_layout((self.rows, self.bk), stride=(self.bk + _NOISE_SKEW_ELEMS, 1))
        )
        return sE1B_rows, (sE1B, sNoise)

    def _make_noise_operands(self, tiled_mma_noise, fragment_coords, thread_idx, sF1, noise_smem):
        """Smem-descriptor F1/E1 fragments and one whole transposed (bk, rows)
        register accumulator."""
        sE1B, sNoise = noise_smem
        thr_noise = tiled_mma_noise.get_slice(thread_idx)
        return _NoiseOperands(
            tiled_mma_noise,
            thr_noise.make_fragment_A(thr_noise.partition_A(sF1)),
            thr_noise.make_fragment_B(thr_noise.partition_B(sE1B)),
            cute.make_rmem_tensor(thr_noise.partition_shape_C((self.bk, self.rows)), Float32),
            thr_noise.partition_C(cute.make_identity_tensor((self.bk, self.rows))),
            fragment_coords,
            sNoise,
        )

    def _load_e1_operand(self, noise):
        """Nothing to load: the WGMMA reads E1 from shared memory."""

    @cute.jit
    def _stage_noise_tile(self, noise, factor_pipe, factor_read):
        """noise^T (bk, rows) = F1[stage] (bk, K) @ E1^T, K = PACKED_NOISE_K:
        one QGMMA group accumulating from +0 -- the pinned Hopper
        arithmetic -- drained (``wg_wait=0``) and staged transposed."""
        accumulator = noise.accumulator
        factor_pipe.consumer_wait(factor_read)
        gemm_w_idx(
            noise.tiled_mma,
            accumulator,
            noise.f1_fragment,
            noise.e1_fragment,
            zero_init=Boolean(True),
            A_idx=factor_read.index,
            B_idx=Int32(0),
            wg_wait=0,
        )
        factor_pipe.consumer_release(factor_read)
        # WAR guard: no warp may overwrite the staging tile until every warp
        # has finished the previous tile's noise reads (quantize reads rows
        # other warps staged).
        cute.arch.barrier(
            barrier_id=_NamedBarrier.CONSUMER,
            number_of_threads=self.consumer_threads,
        )
        coords = noise.accumulator_coords
        for item in cutlass.range(cute.size(accumulator), unroll_full=True):
            noise.staging[(coords[item][1], coords[item][0])] = accumulator[item]
        cute.arch.barrier(
            barrier_id=_NamedBarrier.CONSUMER,
            number_of_threads=self.consumer_threads,
        )

    @cute.jit
    def _noise_words(self, noise, row_half, first_fragment_row, noise_words):
        """Two f32 staging reads, then one cvt.rn.bf16x2.f32 per pair (the
        reference's single RNE rounding)."""
        coords = noise.fragment_coords
        for block in cutlass.range_constexpr(2 * self.bk // self.n_group):
            col_lo = coords[_C_FRAGMENT_VALUES * block][1]
            col_hi = coords[_C_FRAGMENT_VALUES * block + 1][1]
            for fragment_half in cutlass.range_constexpr(2):
                noise_row = 16 * row_half + first_fragment_row + 8 * fragment_half
                noise_words[fragment_half + 2 * block] = _f32x2_to_bf16x2(
                    noise.staging[(noise_row, col_hi)],
                    noise.staging[(noise_row, col_lo)],
                )
