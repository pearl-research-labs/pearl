"""Device kernel: one keyed-BLAKE3 noise line per thread."""

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
from cutlass import Float32

from ..noisy_quant._quantization_ops import _generate_noise_line
from ..protocol_constants import R

_THREADS_PER_BLOCK = 128


class _NoiseLines:
    """Embarrassingly parallel line draw: BLAKE3, decode, normalize, e4m3.

    ``msg_base`` fixes the label and instance index at compile time (exactly
    as ``_NoisyQuant`` fixes its E-line message); the line index is each
    thread's global index.
    """

    def __init__(self, count: int, msg_base: tuple):
        assert count > 0
        assert msg_base
        self.count = count
        self.msg_base = msg_base

    @cute.jit
    def __call__(self, mKey: cute.Tensor, mOut: cute.Tensor, stream: cuda_drv.CUstream):
        self.kernel(mKey, mOut).launch(
            grid=(cute.ceil_div(self.count, _THREADS_PER_BLOCK), 1, 1),
            block=(_THREADS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mKey: cute.Tensor, mOut: cute.Tensor):
        f8 = cutlass.Float8E4M3FN
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        index = bidx * _THREADS_PER_BLOCK + tidx
        if index < self.count:
            line = cute.make_rmem_tensor(R, Float32)
            _generate_noise_line(mKey, cutlass.Uint32(index), self.msg_base, line)
            line_codes = cute.make_rmem_tensor(R, f8)
            line_codes.store(line.load().to(f8))
            cute.autovec_copy(
                line_codes,
                cute.make_tensor(mOut.iterator + index * R, cute.make_layout(R)),
            )
