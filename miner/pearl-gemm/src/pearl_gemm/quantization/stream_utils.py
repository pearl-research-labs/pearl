import cuda.bindings.driver as cuda_drv
import torch

_stream_cache: dict[int, tuple[int, cuda_drv.CUstream]] = {}


def get_stream(device_index: int = 0) -> cuda_drv.CUstream:
    """Return a cached ``CUstream`` for the current CUDA stream on *device_index*."""
    raw = torch._C._cuda_getCurrentRawStream(device_index)
    entry = _stream_cache.get(device_index)
    if entry is None or entry[0] != raw:
        cu = cuda_drv.CUstream(raw)
        _stream_cache[device_index] = (raw, cu)
        return cu
    return entry[1]
