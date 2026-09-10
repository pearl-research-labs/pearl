"""Atomically re-arm a persistent hit signal on its CUDA device."""

from collections.abc import Callable

import cuda.bindings.driver as cuda_driver
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from .._utils._compile import get_or_compile
from .._utils._stream import get_stream
from ._hit_signal import HitRecordLayout

_reset_cache: dict[tuple, object] = {}


@cute.kernel
def _reset_kernel(record: cute.Tensor, lock: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        # Defense in depth for direct/internal callers: status publication is
        # the consumer's ownership token. lock=1/status=0 is still a producer
        # copying payload and must never be re-armed.
        status = cute.arch.load(
            record.iterator.llvm_ptr,
            Uint32,
            sem="acquire",
            scope="sys",
        )
        if status != 0:
            # A new producer may claim immediately after the CAS, so clear
            # every host-visible marker and make those stores system-visible
            # before atomically opening the device latch.
            record[HitRecordLayout.MAGIC] = Uint32(0)
            record[HitRecordLayout.MAGIC + 1] = Uint32(0)
            record[HitRecordLayout.STATUS] = Uint32(0)
            cute.arch.fence_acq_rel_sys()
            cute.arch.atomic_cas(
                ptr=lock.iterator.llvm_ptr,
                cmp=Int32(1),
                val=Int32(0),
                sem="acq_rel",
                scope="gpu",
            )


@cute.jit
def _reset_launch(
    record: cute.Tensor,
    lock: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    _reset_kernel(record, lock).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
        stream=stream,
    )


def prepare_hit_reset(
    record: torch.Tensor,
    lock: torch.Tensor,
    device: torch.device,
) -> Callable[[], None]:
    """Compile once and bind one signal's stable record/latch pointers."""
    index = device.index or 0
    args = (
        from_dlpack(record, assumed_align=16),
        from_dlpack(lock, assumed_align=16),
    )
    stream = get_stream(index)
    capability = torch.cuda.get_device_capability(device)
    compiled = get_or_compile(
        _reset_cache,
        (capability,),
        lambda: cute.compile(_reset_launch, *args, stream),
    )

    def reset() -> None:
        compiled(*args, get_stream(index))

    return reset
