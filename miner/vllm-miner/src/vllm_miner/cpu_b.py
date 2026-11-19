"""Background CPU B-side preparation service and memfd-backed shared host tensors."""

import contextlib
import ctypes
import mmap
import multiprocessing
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing as mp
from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.protocol import Device
from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import STATE_WINDOW_DEPTH, MiningJob

from .settings import parse_cpu_list, runtime_settings

if TYPE_CHECKING:
    from .b_prepare_cpu import CpuBContext
    from .state import LayerState

_LOGGER = get_logger(__name__)
_POLL_S = 0.01
_CLOSE_TIMEOUT_S = 10.0
_HELPER_ENV = {"THP_MEM_ALLOC_ENABLE": "1"}
_NVIDIA_GPUS = Path("/proc/driver/nvidia/gpus")
_PCI_DEVICES = Path("/sys/bus/pci/devices")
_SLOTS = 3

# ---------------------------------------------------------------------------
# memfd-backed shared host tensors (avoids /dev/shm tmpfs limits)
# ---------------------------------------------------------------------------

_SHARED_ATTR = "_pearl_shared_region"
_MFD_CLOEXEC = 1


def _memfd_create(name: str) -> int:
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
    def __init__(self, fd: int, nbytes: int) -> None:
        self.fd = fd
        self.nbytes = nbytes

    def __del__(self) -> None:
        with contextlib.suppress(OSError):
            os.close(self.fd)


@dataclass(frozen=True)
class SharedHandle:
    pid: int
    fd: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


def _map_memfd(fd: int, nbytes: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    mapping = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    return torch.frombuffer(mapping, dtype=torch.uint8, count=nbytes).view(dtype).view(shape)


def shared_empty(shape: tuple[int, ...] | torch.Size, dtype: torch.dtype) -> torch.Tensor:
    shape = tuple(int(dim) for dim in shape)
    nbytes = torch.empty(0, dtype=dtype).element_size()
    for dim in shape:
        nbytes *= dim
    if nbytes <= 0:
        raise ValueError(f"a shared host tensor needs a nonempty shape, got {shape}")
    fd = _memfd_create("pearl-host-tensor")
    region = _Region(fd, nbytes)
    os.ftruncate(fd, nbytes)
    tensor = _map_memfd(fd, nbytes, dtype, shape)
    setattr(tensor, _SHARED_ATTR, region)
    return tensor


def shared_copy(source: torch.Tensor) -> torch.Tensor:
    tensor = shared_empty(source.shape, source.dtype)
    tensor.copy_(source)
    return tensor


def shared_handle(tensor: torch.Tensor) -> SharedHandle | None:
    region = getattr(tensor, _SHARED_ATTR, None)
    if region is None:
        return None
    return SharedHandle(os.getpid(), region.fd, region.nbytes, tensor.dtype, tuple(tensor.shape))


def attach(handle: SharedHandle) -> torch.Tensor:
    fd = os.open(f"/proc/{handle.pid}/fd/{handle.fd}", os.O_RDWR | os.O_CLOEXEC)
    try:
        return _map_memfd(fd, handle.nbytes, handle.dtype, handle.shape)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# CPU topology & helper process
# ---------------------------------------------------------------------------


def gpu_local_cpus(device: torch.device) -> tuple[int, ...]:
    """Return ``device``'s share of its NUMA-local CPUs for PEARL_CPU_B_PREPARE_CPUS=auto."""
    props = torch.cuda.get_device_properties(device)
    address = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    try:
        peers = sorted(entry.name for entry in _NVIDIA_GPUS.iterdir())
        local = {gpu: (_PCI_DEVICES / gpu / "local_cpulist").read_text().strip() for gpu in peers}
    except OSError as exc:
        raise RuntimeError(
            f"PEARL_CPU_B_PREPARE_CPUS=auto: cannot read GPU topology: {exc}"
        ) from exc
    if address not in local:
        raise RuntimeError(
            f"PEARL_CPU_B_PREPARE_CPUS=auto: {device} ({address}) is not under {_NVIDIA_GPUS}"
        )
    node = [gpu for gpu in peers if local[gpu] == local[address]]
    cpus = parse_cpu_list(local[address])
    index = node.index(address)
    share = cpus[index * len(cpus) // len(node) : (index + 1) * len(cpus) // len(node)]
    permitted = set(os.sched_getaffinity(0))
    share = tuple(cpu for cpu in share if cpu in permitted)
    if not share:
        raise RuntimeError(
            f"PEARL_CPU_B_PREPARE_CPUS=auto: none of {address}'s local CPUs "
            f"({local[address]}, share {index + 1} of {len(node)}) is in this process's affinity"
        )
    return share


@contextmanager
def _environ(updates: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name)
            else:
                os.environ[name] = value


@contextmanager
def _children_allowed() -> Iterator[None]:
    """Allow vLLM's daemonic multiproc workers to spawn the daemonic CPU B helper."""
    config = multiprocessing.current_process()._config
    daemon = config.pop("daemon", None)
    try:
        yield
    finally:
        if daemon is not None:
            config["daemon"] = daemon


class PreparationProcessError(RuntimeError):
    pass


def _check_cuda(status: object) -> None:
    if status != torch.cuda.cudart().cudaError.success:
        raise PreparationProcessError(f"CUDA host registration of a result slot failed: {status}")


def _pinned_shared(nbytes: int) -> torch.Tensor:
    buffer = shared_empty((nbytes,), torch.uint8)
    _check_cuda(torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), nbytes, 0))
    return buffer


def _shared(tensor: torch.Tensor) -> SharedHandle | torch.Tensor:
    handle = shared_handle(tensor)
    return handle if handle is not None else tensor.share_memory_()


def _attached(shared: SharedHandle | torch.Tensor) -> torch.Tensor:
    return attach(shared) if isinstance(shared, SharedHandle) else shared


def _slot_view(slot: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return slot[: shape.numel()].view(torch.float8_e4m3fn).view(shape)


def _serve(
    layers: list[tuple[SharedHandle | torch.Tensor, SharedHandle | torch.Tensor, int]],
    slots: list[SharedHandle | torch.Tensor],
    threads: int,
    cpus: tuple[int, ...],
    tasks: "mp.SimpleQueue",
    results: "mp.Queue",
    cancel: "mp.Event",
    ready: "mp.Event",
) -> None:
    from .b_prepare_cpu import RowScales, native, prepare_b_cpu, without_b_proof

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if cpus:
        os.sched_setaffinity(0, cpus)
    torch.set_num_threads(threads)
    native()
    layers = [(_attached(codes), _attached(scales), experts) for codes, scales, experts in layers]
    slots = [_attached(slot) for slot in slots]
    row_scales: dict[int, RowScales] = {}
    ready.set()
    while (task := tasks.get()) is not None:
        layer, job, device, slot = task
        codes, scales, experts = layers[layer]
        try:
            result = prepare_b_cpu(
                codes,
                scales,
                job,
                device=device,
                experts=experts,
                cancelled=cancel.is_set,
                row_scales=row_scales.get(layer),
            )
        except Exception as exc:
            results.put((repr(exc), None))
            continue
        if result is None:
            results.put((None, None))
            continue
        prepared, row_scales[layer] = result
        _slot_view(slots[slot], prepared.b_prime.shape).copy_(prepared.b_prime)
        results.put((None, without_b_proof(prepared) | {"b_prime": None}))


class ProcessPreparer:
    """Helper subprocess preparing layers one at a time into pinned shared DMA slots."""

    def __init__(
        self,
        states: "Sequence[LayerState]",
        *,
        threads: int,
        cpus: Sequence[int] = (),
        slots: int = _SLOTS,
    ) -> None:
        if threads < 1 or slots < 1:
            raise ValueError("the preparation process needs a thread and a result slot")
        self._states = tuple(states)
        self._planes = tuple((state.weight_cpu, state.weight_scale_cpu) for state in self._states)
        slot_bytes = max(codes.numel() for codes, _ in self._planes)
        self._slots = [_pinned_shared(slot_bytes) for _ in range(slots)]
        self._pinned = list(self._slots)
        context = mp.get_context("spawn")
        self._tasks = context.SimpleQueue()
        self._results = context.Queue()
        self._cancel = context.Event()
        ready = context.Event()
        self._process = context.Process(
            target=_serve,
            args=(
                [
                    (_shared(codes), _shared(scales), state.experts)
                    for (codes, scales), state in zip(self._planes, self._states, strict=True)
                ],
                [_shared(slot) for slot in self._slots],
                threads,
                tuple(cpus),
                self._tasks,
                self._results,
                self._cancel,
                ready,
            ),
            name="cpu-b-prepare",
            daemon=True,
        )
        with _environ(_HELPER_ENV), _children_allowed():
            self._process.start()
        while not ready.wait(_POLL_S):
            if not self._process.is_alive():
                raise PreparationProcessError(
                    f"the preparation process exited with code {self._process.exitcode}"
                )

    @property
    def slots(self) -> int:
        return len(self._slots)

    def prepare(
        self, layer: int, job: MiningJob, device: Device, slot: int, cancelled: Callable[[], bool]
    ) -> "CpuBContext | None":
        from .b_prepare_cpu import with_b_proof

        self._cancel.clear()
        self._tasks.put((layer, job, device, slot))
        while True:
            try:
                error, detached = self._results.get(timeout=_POLL_S)
            except queue.Empty:
                if not self._process.is_alive():
                    raise PreparationProcessError(
                        f"the preparation process exited with code {self._process.exitcode}"
                    ) from None
                if cancelled():
                    self._cancel.set()
                continue
            if error is not None:
                raise PreparationProcessError(error)
            if detached is None:
                return None
            codes, scales = self._planes[layer]
            detached["b_prime"] = _slot_view(self._slots[slot], codes.shape)
            return with_b_proof(detached, codes, scales)

    def close(self, unregister: bool = True) -> None:
        if self._process.is_alive():
            self._cancel.set()
            self._tasks.put(None)
            self._process.join(_CLOSE_TIMEOUT_S)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
        while unregister and self._pinned:
            _check_cuda(torch.cuda.cudart().cudaHostUnregister(self._pinned.pop().data_ptr()))


@dataclass
class CpuPrepared:
    prepared: "CpuBContext"
    _service: "CpuBService"
    _slot: int

    def uploaded(self, event: torch.cuda.Event) -> None:
        self._service._release(self._slot, event)


def b_key_depth(ancestor: bytes, job: MiningJob) -> int | None:
    window = [bytes(header) for header in job.ancestor_headers[:STATE_WINDOW_DEPTH]]
    return window.index(bytes(ancestor)) if bytes(ancestor) in window else None


class CpuBService:
    """Prepares each layer's next B side off-GPU ahead of ancestor expiration."""

    def __init__(
        self,
        states: "Sequence[LayerState]",
        manager: AsyncLoopManager,
        *,
        threads: int,
        cpus: Sequence[int] = (),
        replace_lead: int = 1,
    ) -> None:
        if not 1 <= replace_lead < STATE_WINDOW_DEPTH:
            raise ValueError(
                f"replace_lead must be 1..{STATE_WINDOW_DEPTH - 1} tip changes, got {replace_lead}"
            )
        self._states = tuple(states)
        self._index = {id(state): layer for layer, state in enumerate(self._states)}
        self._manager = manager
        self._replace_lead = replace_lead
        self._preparer = ProcessPreparer(self._states, threads=threads, cpus=cpus)
        self._cond = threading.Condition()
        self._job: MiningJob | None = None
        self._next_layer = 0
        self._ready: dict[int, tuple[CpuBContext, int]] = {}
        self._free = list(range(self._preparer.slots))
        self._pending: dict[int, torch.cuda.Event] = {}
        self._taken: set[int] = set()
        self._pass_started = 0.0
        self._pass_prepared = 0
        self._pass_taken = 0
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="cpu-b-prepare", daemon=True)
        self._worker.start()
        manager.register_mining_job_changed_callback(self._job_changed)
        self._job_changed()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._worker.join(_CLOSE_TIMEOUT_S)
        with self._cond:
            drained = self._cond.wait_for(lambda: not self._taken, _CLOSE_TIMEOUT_S)
            pending = list(self._pending.values())
        for event in pending:
            event.synchronize()
        if not drained:
            _LOGGER.warning("CPU B: an upload is still outstanding; its result slots stay pinned")
        self._preparer.close(unregister=drained)

    def waiting(self, state: "LayerState") -> bool:
        layer = self._index.get(id(state))
        with self._cond:
            return layer in self._ready

    def take(
        self, state: "LayerState", job: MiningJob, current: bytes | None
    ) -> CpuPrepared | None:
        layer = self._index.get(id(state))
        if layer is None:
            return None
        with self._cond:
            entry = self._ready.get(layer)
            if entry is None:
                return None
            prepared, slot = entry
            depth = b_key_depth(prepared.b_proof.ancestor_header, job)
            current_depth = None if current is None else b_key_depth(current, job)
            del self._ready[layer]
            if depth is None or (current_depth is not None and current_depth <= depth):
                self._free.append(slot)
                self._cond.notify_all()
                return None
            self._taken.add(slot)
            self._pass_taken += 1
            if self._pass_taken == self._pass_prepared and self._next_layer >= len(self._states):
                _LOGGER.info(
                    f"CPU B: {self._pass_taken} layer(s) mine on B prepared for the job's parent, "
                    f"{time.monotonic() - self._pass_started:.2f}s after the job change"
                )
        return CpuPrepared(prepared, self, slot)

    def discard(self) -> None:
        with self._cond:
            self._free.extend(slot for _, slot in self._ready.values())
            self._ready.clear()
            self._cond.notify_all()

    def _release(self, slot: int, event: torch.cuda.Event) -> None:
        with self._cond:
            self._taken.discard(slot)
            self._pending[slot] = event
            self._free.append(slot)
            self._cond.notify_all()

    def _job_changed(self) -> None:
        job = self._manager.get_mining_job()
        if job is None:
            return
        with self._cond:
            if self._pass_prepared:
                _LOGGER.info(
                    f"CPU B: {self._pass_prepared} layer(s) prepared and {self._pass_taken} "
                    "taken for the previous parent before this job change"
                )
            self._job = job
            self._next_layer = 0
            self._pass_started = time.monotonic()
            self._pass_prepared = 0
            self._pass_taken = 0
            for layer in [
                layer
                for layer, (prepared, _) in self._ready.items()
                if b_key_depth(prepared.b_proof.ancestor_header, job) is None
            ]:
                self._free.append(self._ready.pop(layer)[1])
            self._cond.notify_all()

    def _needs(self, layer: int, job: MiningJob) -> bool:
        entry = self._ready.get(layer)
        if entry is not None and b_key_depth(entry[0].b_proof.ancestor_header, job) == 0:
            return False
        state = self._states[layer]
        with state.lock:
            ctx = state.job_ctx
        if ctx is None:
            return True
        depth = b_key_depth(ctx.b_proof.ancestor_header, job)
        return depth is None or (depth > 0 and STATE_WINDOW_DEPTH - depth <= self._replace_lead)

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: (
                        self._closed
                        or (
                            self._job is not None
                            and self._next_layer < len(self._states)
                            and bool(self._free)
                        )
                    )
                )
                if self._closed:
                    return
                job = self._job
                assert job is not None
                layer = self._next_layer
                self._next_layer += 1
                if not self._needs(layer, job):
                    self._cond.notify_all()
                    continue
                slot = self._free.pop()
                pending = self._pending.pop(slot, None)
            if pending is not None:
                pending.synchronize()
            state = self._states[layer]
            try:
                prepared = self._preparer.prepare(
                    layer, job, state.committed_device, slot, partial(self._superseded, job)
                )
            except Exception as exc:
                _LOGGER.error(f"{state.layer_name}: CPU B preparation failed: {exc!r}")
                prepared = None
            with self._cond:
                if prepared is None or self._superseded(job):
                    self._free.append(slot)
                else:
                    previous = self._ready.pop(layer, None)
                    if previous is not None:
                        self._free.append(previous[1])
                    self._ready[layer] = (prepared, slot)
                    self._pass_prepared += 1
                    if self._next_layer >= len(self._states):
                        _LOGGER.info(
                            f"CPU B: prepared {self._pass_prepared} layer(s) for the job's "
                            f"parent in {time.monotonic() - self._pass_started:.2f}s"
                        )
                self._cond.notify_all()
            del prepared

    def _superseded(self, job: MiningJob) -> bool:
        latest = self._job
        return self._closed or (
            latest is not None and b_key_depth(job.parent_header, latest) is None
        )


_service: CpuBService | None = None


def _rank_cpus(cpus: Sequence[int]) -> tuple[int, ...]:
    cpus = tuple(cpus)
    if not cpus or not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return cpus
    world, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
    share = len(cpus) // world
    if world <= 1 or share < 1:
        return cpus
    return cpus[rank * share : (rank + 1) * share]


def start_cpu_b_service(manager: AsyncLoopManager) -> None:
    global _service
    from .b_prepare_cpu import ARITHMETIC
    from .state import all_states

    if _service is not None:
        return
    states = [state for state in all_states() if state.mineable]
    if not states:
        return
    unsupported = {state.committed_device for state in states} - set(ARITHMETIC)
    if unsupported:
        _LOGGER.info(
            f"CPU B preparation has no arithmetic for {sorted(d.name for d in unsupported)}; "
            "B stays on inline GPU preparation."
        )
        return
    settings = runtime_settings()
    threads = settings.cpu_b_prepare_threads
    cpus = (
        gpu_local_cpus(states[0].weight.device)
        if settings.cpu_b_prepare_cpus == "auto"
        else _rank_cpus(settings.cpu_b_prepare_cpus)
    )
    if cpus and threads > len(cpus):
        _LOGGER.warning(
            f"PEARL_CPU_B_PREPARE_THREADS={threads} exceeds the helper's {len(cpus)} CPUs"
        )
    _service = CpuBService(
        states,
        manager,
        threads=threads,
        cpus=cpus,
        replace_lead=settings.cpu_b_replace_lead,
    )
    cpulist = ",".join(str(c) for c in cpus) or "any"
    _LOGGER.info(
        f"CPU B: {len(states)} layers prepare their next B on the CPU "
        f"({threads} threads, CPUs {cpulist}), "
        f"{settings.cpu_b_replace_lead} tip change(s) before ancestor expiration"
    )


def take_cpu_prepared(
    state: "LayerState", job: MiningJob, current: bytes | None
) -> CpuPrepared | None:
    service = _service
    return None if service is None else service.take(state, job, current)


def cpu_prepared_waiting(state: "LayerState") -> bool:
    service = _service
    return service is not None and service.waiting(state)


def discard_cpu_prepared() -> None:
    if _service is not None:
        _service.discard()


def stop_cpu_b_service() -> None:
    global _service
    if _service is not None:
        _service.close()
        _service = None
