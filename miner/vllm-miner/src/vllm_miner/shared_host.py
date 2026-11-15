"""Host tensors another process can map without ``/dev/shm``.

CPU B preparation runs in a helper process over every mined layer's host
planes (tens of GiB per GPU). ``torch.Tensor.share_memory_`` backs a tensor
with a file in ``/dev/shm``, whose tmpfs is a small fixed size in a serving
pod; an anonymous ``memfd`` is the same shared page cache without that limit
(it is charged to the process's memory cgroup like any allocation). The
creator keeps the descriptor open for the tensor's lifetime and a child
process re-opens it through ``/proc/<pid>/fd``.
"""

import contextlib
import ctypes
import mmap
import os
from dataclasses import dataclass

import torch

_ATTRIBUTE = "_pearl_shared_region"
_MFD_CLOEXEC = 1


def _memfd_create(name: str) -> int:
    """``os.memfd_create``, or glibc's own where Python was built without it
    (the serving image's python-build-standalone CPython)."""
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, os.MFD_CLOEXEC)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    libc.memfd_create.restype = ctypes.c_int
    fd = libc.memfd_create(name.encode(), _MFD_CLOEXEC)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, f"memfd_create: {os.strerror(error)}")
    return fd


class _Region:
    """Owns one memfd; closed with the tensor that carries it."""

    def __init__(self, fd: int, nbytes: int) -> None:
        self.fd = fd
        self.nbytes = nbytes

    def __del__(self) -> None:
        with contextlib.suppress(OSError):
            os.close(self.fd)


@dataclass(frozen=True)
class SharedHandle:
    """What a child of the creating process needs to map a shared tensor."""

    pid: int
    fd: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


def _map(fd: int, nbytes: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    mapping = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    # ``frombuffer`` keeps ``mapping`` alive for as long as any view of it.
    flat = torch.frombuffer(mapping, dtype=torch.uint8, count=nbytes)
    return flat.view(dtype).view(shape)


def shared_empty(shape: tuple[int, ...] | torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """An uninitialized (zero-filled) host tensor backed by a fresh memfd."""
    shape = tuple(int(dim) for dim in shape)
    nbytes = torch.empty(0, dtype=dtype).element_size()
    for dim in shape:
        nbytes *= dim
    if nbytes <= 0:
        raise ValueError(f"a shared host tensor needs a nonempty shape, got {shape}")
    fd = _memfd_create("pearl-host-tensor")
    region = _Region(fd, nbytes)
    os.ftruncate(fd, nbytes)
    tensor = _map(fd, nbytes, dtype, shape)
    setattr(tensor, _ATTRIBUTE, region)
    return tensor


def shared_copy(source: torch.Tensor) -> torch.Tensor:
    """``source`` (any device) copied into a new shared host tensor."""
    tensor = shared_empty(source.shape, source.dtype)
    tensor.copy_(source)
    return tensor


def shared_handle(tensor: torch.Tensor) -> SharedHandle | None:
    """The handle of a tensor :func:`shared_empty` returned (not of a view of
    it), or None for any other tensor."""
    region = getattr(tensor, _ATTRIBUTE, None)
    if region is None:
        return None
    return SharedHandle(os.getpid(), region.fd, region.nbytes, tensor.dtype, tuple(tensor.shape))


def attach(handle: SharedHandle) -> torch.Tensor:
    """Map ``handle``'s tensor in a child of the process that created it."""
    fd = os.open(f"/proc/{handle.pid}/fd/{handle.fd}", os.O_RDWR | os.O_CLOEXEC)
    try:
        return _map(fd, handle.nbytes, handle.dtype, handle.shape)
    finally:
        # The mapping holds its own reference to the file.
        os.close(fd)
