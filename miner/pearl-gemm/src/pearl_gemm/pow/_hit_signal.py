"""The mining hit signal: one persistent per-process record the ``mixed_gemm``
kernel publishes PoW hits into, consumed by a host thread in the same process
(device producer: ``mixed_gemm/_kernel.py``; host consumer: this module).

The public contract: FIRST hit wins -- the producer claims the persistent
latch it never releases, so a pending record is immutable until the consumer
re-arms it and later hits are dropped. ``doorbell()`` is one native acquire
CPU load (no CUDA API while idle); ``read_hit()`` snapshots the record plus the
publish-time payload planes without consuming; ``reset_hit()`` consumes and
re-arms. Payloads are best-effort (records publish payload-less when the
planes exceed capacity), and there is no job gating -- consumers reject hits
whose commitment does not match the current job. Memory layout and ordering
protocol are documented inline below.
"""

import ctypes
import threading
from dataclasses import dataclass, replace

import torch

from ..protocol_constants import BLOCK_SCALE_GROUP

_UINT32_MAX = 2**32 - 1

# `_hit_warp_copy` recasts each plane to 16-byte vectors with no tail.
# Codes rows are k bytes; scales rows are 2 * (k / BLOCK_SCALE_GROUP) bytes.
# The scales row is the tighter constraint: it is a whole number of 16-byte
# vectors iff k is a multiple of 8 * BLOCK_SCALE_GROUP (64 when the group is 8).
HIT_PAYLOAD_K_ALIGN = 8 * BLOCK_SCALE_GROUP

# The doorbell poll must be an ACQUIRE load: the producer publishes with a
# system-scope release store of the status word, and only an acquire-paired
# read of it orders the host's subsequent record reads on weakly-ordered
# CPUs (Grace) -- a plain load can observe status == 1 yet still read stale
# record words, dropping the found block as malformed. Python and Torch
# expose no host atomics, so bind gcc's libatomic (Debian/Ubuntu package
# ``libatomic1``) for a native 32-bit acquire load. Bound lazily on first
# ``HitSignal`` construction so a missing ``libatomic.so.1`` does not break
# every ``import pearl_gemm``; failing to load it is a hard error, never a
# silent plain-load fallback.
_ATOMIC_ACQUIRE = 2  # __ATOMIC_ACQUIRE, gcc/clang atomic builtin ABI
_atomic_load_u32 = None


def _bind_atomic_load_u32():
    """Resolve ``libatomic.so.1`` once; raise an actionable error if absent."""
    global _atomic_load_u32
    if _atomic_load_u32 is not None:
        return _atomic_load_u32
    try:
        lib = ctypes.CDLL("libatomic.so.1")
    except OSError as exc:
        raise RuntimeError(
            "HitSignal requires libatomic.so.1 for the doorbell acquire load; "
            "install the libatomic1 system package (Debian/Ubuntu)"
        ) from exc
    fn = lib.__atomic_load_4
    fn.restype = ctypes.c_uint32
    fn.argtypes = (ctypes.c_void_p, ctypes.c_int)
    _atomic_load_u32 = fn
    return fn


class HitSignalPoisonedError(RuntimeError):
    """The persistent producer latch can no longer be proven reusable."""


class HitRecordLayout:
    """Word offsets of the pinned record's uint32 fields: the single source of
    truth shared by the device producer (``mixed_gemm/_kernel.py``), the host
    consumer below, and the tests that forge records. 64 words = 256 bytes."""

    STATUS = 0  # the doorbell: 0 idle, 1 published
    M = 1
    N = 2
    K = 3
    TILE_ROW = 4  # winning lottery tile coordinates
    TILE_COLUMN = 5
    LTILE_ROWS = 6  # lottery tile geometry, so records are self-describing
    LTILE_COLS = 7
    CODES_PAYLOAD_BYTES = 8  # 0 = payload-less record
    SCALES_PAYLOAD_BYTES = 9
    # The caller's per-launch layer tag (mixed_gemm's ``layer_id``): the
    # useful miner runs many layers against ONE process-wide signal, and the
    # consumer callback uses this to look up the layer the hit belongs to.
    LAYER_ID = 10
    TARGET = 12  # 8 words: the launch threshold (LE word order); word 11 is unused
    HASH_A = 20  # 8 words: the launch's pow_key (v4: the jackpot key from seedA)
    HASH_B = 28  # 8 words: the launch's B-side stamp (v4: noise seedB)
    MAGIC = 36  # 2 words, written before the status flip
    RECORD_WORDS = 64


RECORD_WORDS = HitRecordLayout.RECORD_WORDS

# 64-bit numeric magic 0x4849545349473031, split low word first to preserve
# the little-endian record ABI (the bytes do NOT spell "HITSIG01"
# in a hexdump). Sanity magic: written before the status flip and zeroed by
# the consumer's reset, so a record that does not carry it never parses as
# valid.
HIT_RECORD_MAGIC_WORDS = (0x49473031, 0x48495453)


@dataclass(frozen=True)
class Hit:
    """One parsed hit-record snapshot (``read_hit`` does NOT consume).

    ``valid=False`` means a torn/forged record: every other field is zeroed
    and the caller must drop it (and still ``reset_hit()``). The payload
    tensors are views of the signal's staging buffers, valid until the next
    ``read_hit()``; ``None`` when the record published payload-less.
    """

    valid: bool
    m: int = 0
    n: int = 0
    k: int = 0
    tile_row: int = 0
    tile_column: int = 0
    ltile_rows: int = 0
    ltile_cols: int = 0
    layer_id: int = 0  # the launch's layer tag
    target: bytes = b""
    commitment_hash_A: bytes = b""
    commitment_hash_B: bytes = b""
    codes: torch.Tensor | None = None  # (m, k) int8
    scales: torch.Tensor | None = None  # (m, k // BLOCK_SCALE_GROUP) bf16

    def owned_copy(self) -> "Hit":
        """Copy staging-backed payload planes before the signal is re-armed."""
        return replace(
            self,
            codes=None if self.codes is None else self.codes.clone(),
            scales=None if self.scales is None else self.scales.clone(),
        )


@dataclass(frozen=True)
class HitSignalConfig:
    """Sizing for the per-process hit signal: worker-wide maxima over all
    mining shapes. The payload regions hold one A codes plane ((max_m, max_k)
    int8) and one A scales plane ((max_m, max_k/8) bf16). ``max_k`` must be a
    multiple of ``HIT_PAYLOAD_K_ALIGN`` (64): the producer copies both planes
    as 16-byte vectors with no tail. Payloads are best-effort: hits whose
    planes would not fit publish without the snapshot."""

    max_m: int = 256
    max_k: int = 4096

    def __post_init__(self) -> None:
        # Validate BEFORE allocating: a bad config must raise cleanly, not
        # attempt an absurd (possibly OOM-ing) allocation first.
        if type(self.max_m) is not int or type(self.max_k) is not int:
            raise TypeError(f"max_m and max_k must be ints, got {self}")
        if self.max_m <= 0 or self.max_k <= 0:
            raise ValueError(f"max_m and max_k must be positive, got {self}")
        if self.max_k % HIT_PAYLOAD_K_ALIGN:
            raise ValueError(f"max_k must be divisible by {HIT_PAYLOAD_K_ALIGN}, got {self}")
        if self.max_m * self.max_k > _UINT32_MAX:
            raise ValueError(f"max_m * max_k overflows uint32, got {self}")


class HitSignal:
    """Persistent per-process PoW hit signal (capture-safe, in-process consumer).

    Allocates the signal's buffers once and hands the same tensors to every
    ``mixed_gemm`` launch; poll ``doorbell()`` and consume hits with
    ``read_hit()`` / ``reset_hit()``.

    End-to-end flow (``m``/``k`` are the A operand's shape, maxima over every
    shape the signal will serve)::

        m, k = a_codes.shape  # (m, k) int8; a_scales is (m, k // 8) bf16
        hit_signal = HitSignal(HitSignalConfig(max_m=m, max_k=k), device="cuda")

        mixed_gemm(..., hit_signal, a_codes, a_scales, commitment_hash_b, ...)

        if hit_signal.doorbell():          # idle poll: no CUDA API
            hit = hit_signal.read_hit()    # Hit snapshot; does NOT consume
            if hit is not None and hit.valid:
                ...  # verify against the current job, open the proof
            hit_signal.reset_hit()         # ALWAYS re-arm, even for drops

    Omitting ``reset_hit()`` leaves the persistent first-wins latch closed,
    so every later hit is silently dropped.
    """

    def __init__(
        self,
        cfg: HitSignalConfig | None = None,
        device: torch.device | str | int | None = None,
    ) -> None:
        self.cfg = cfg or HitSignalConfig()
        device = torch.device(device) if device is not None else torch.device("cuda")
        if device.type != "cuda":
            raise ValueError(f"HitSignal requires a CUDA device, got {device}")
        if device.index is None:
            # Normalize so launch-time equality against tensor devices holds.
            device = torch.device(device.type, torch.cuda.current_device())
        self.device = device
        _bind_atomic_load_u32()

        codes_elems = self.cfg.max_m * self.cfg.max_k
        scales_elems = self.cfg.max_m * (self.cfg.max_k // BLOCK_SCALE_GROUP)
        # On supported CUDA/UVA platforms, this pinned record is
        # device-addressable, allowing the kernel to publish directly into
        # host-visible memory (exercised by tests/test_hit_signal.py).
        self.record = torch.zeros(RECORD_WORDS, dtype=torch.uint32, pin_memory=True)
        self.lock = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.codes_payload = torch.empty(codes_elems, dtype=torch.int8, device=self.device)
        self.scales_payload = torch.empty(scales_elems, dtype=torch.bfloat16, device=self.device)

        # Pinned staging for read_hit()'s D2H fetches, preallocated so the
        # consume path never issues an allocating CUDA call (which could
        # coincide with a graph capture elsewhere in the process).
        self._codes_staging = torch.empty(codes_elems, dtype=torch.int8, pin_memory=True)
        self._scales_staging = torch.empty(scales_elems, dtype=torch.bfloat16, pin_memory=True)
        # Dedicated stream for the consumer's rare-hit copies and atomic
        # reset: they must
        # never touch the compute stream (which would serialize against it
        # and could invalidate an in-progress capture elsewhere).
        self._consumer_stream = torch.cuda.Stream(device=self.device)
        self._consumer_mu = threading.Lock()
        self._poisoned_error: BaseException | None = None
        from ._reset import prepare_hit_reset

        self._reset = prepare_hit_reset(self.record, self.lock, self.device)
        # The pinned allocation lives as long as self.record, so the raw
        # status address the doorbell polls stays valid.
        self._status_addr = self.record.data_ptr() + 4 * HitRecordLayout.STATUS

    @property
    def codes_payload_capacity_bytes(self) -> int:
        return self.codes_payload.numel()

    @property
    def scales_payload_capacity_bytes(self) -> int:
        return self.scales_payload.numel() * self.scales_payload.element_size()

    def _raise_if_poisoned(self) -> None:
        if self._poisoned_error is not None:
            raise HitSignalPoisonedError(
                "persistent hit signal is poisoned after a failed reset"
            ) from self._poisoned_error

    def require_usable(self) -> None:
        """Fail before launch if a prior consumer could not re-arm the latch."""
        self._raise_if_poisoned()

    def doorbell(self) -> bool:
        """Poll for a published hit (one native acquire load, no CUDA API).

        The acquire pairs with the producer's system-scope release store of
        the status word, ordering every record read behind it.
        """
        self._raise_if_poisoned()
        return _bind_atomic_load_u32()(self._status_addr, _ATOMIC_ACQUIRE) != 0

    def _read_published_hit_locked(self) -> Hit:
        """Parse/fetch a record after this consumer observed publication."""
        layout = HitRecordLayout
        words = self.record.tolist()
        # Bounds-check everything the consumer will index with: a torn/forged
        # record must never parse as valid.
        m, n, k = words[layout.M], words[layout.N], words[layout.K]
        ltile_rows, ltile_cols = words[layout.LTILE_ROWS], words[layout.LTILE_COLS]
        codes_bytes = words[layout.CODES_PAYLOAD_BYTES]
        scales_bytes = words[layout.SCALES_PAYLOAD_BYTES]
        has_payload = codes_bytes != 0 or scales_bytes != 0
        magic_ok = (
            words[layout.MAGIC] == HIT_RECORD_MAGIC_WORDS[0]
            and words[layout.MAGIC + 1] == HIT_RECORD_MAGIC_WORDS[1]
        )
        valid = (
            magic_ok
            and m > 0
            and n > 0
            and k > 0
            and k % HIT_PAYLOAD_K_ALIGN == 0
            and ltile_rows > 0
            and ltile_cols > 0
            and 0 <= words[layout.TILE_ROW] * ltile_rows < m
            and 0 <= words[layout.TILE_COLUMN] * ltile_cols < n
            and (
                not has_payload
                or (
                    codes_bytes == m * k
                    and codes_bytes <= self.codes_payload_capacity_bytes
                    and scales_bytes == m * (k // BLOCK_SCALE_GROUP) * 2
                    and scales_bytes <= self.scales_payload_capacity_bytes
                )
            )
        )
        if not valid:
            return Hit(valid=False)

        codes = scales = None
        if has_payload:
            scales_elems = m * (k // BLOCK_SCALE_GROUP)
            codes = self._codes_staging.narrow(0, 0, m * k).view(m, k)
            scales = self._scales_staging.narrow(0, 0, scales_elems).view(m, k // BLOCK_SCALE_GROUP)
            # Both D2H fetches are enqueued on the dedicated stream and share
            # ONE synchronize.
            with torch.cuda.stream(self._consumer_stream):
                codes.copy_(self.codes_payload.narrow(0, 0, m * k).view(m, k), non_blocking=True)
                scales.copy_(
                    self.scales_payload.narrow(0, 0, scales_elems).view(m, k // BLOCK_SCALE_GROUP),
                    non_blocking=True,
                )
            self._consumer_stream.synchronize()

        record_bytes = self.record.view(torch.uint8)

        def _field(word_offset: int) -> bytes:
            return bytes(record_bytes[4 * word_offset : 4 * (word_offset + 8)].tolist())

        return Hit(
            valid=True,
            m=m,
            n=n,
            k=k,
            tile_row=words[layout.TILE_ROW],
            tile_column=words[layout.TILE_COLUMN],
            ltile_rows=ltile_rows,
            ltile_cols=ltile_cols,
            layer_id=words[layout.LAYER_ID],
            target=_field(layout.TARGET),
            commitment_hash_A=_field(layout.HASH_A),
            commitment_hash_B=_field(layout.HASH_B),
            codes=codes,
            scales=scales,
        )

    def _reset_hit_locked(self) -> None:
        with torch.cuda.stream(self._consumer_stream):
            self._reset()
        self._consumer_stream.synchronize()

    def read_hit(self) -> Hit | None:
        """Read without consuming; payload views live until the next read.

        Multi-threaded consumers should use :meth:`take_owned_hit` instead of
        splitting ownership across this method and :meth:`reset_hit`.
        """
        with self._consumer_mu:
            self._raise_if_poisoned()
            if not self.doorbell():
                return None
            return self._read_published_hit_locked()

    def _reset_or_poison_locked(self, prior_error: BaseException | None = None) -> None:
        try:
            self._reset_hit_locked()
        except BaseException as reset_error:
            cause: BaseException = reset_error
            if prior_error is not None:
                cause = BaseExceptionGroup(
                    "hit snapshot and reset both failed",
                    [prior_error, reset_error],
                )
            self._poisoned_error = cause
            raise HitSignalPoisonedError(
                "persistent hit signal reset failed; mining must stop on this device"
            ) from cause

    def take_owned_hit(self) -> Hit | None:
        """Atomically read, own, and re-arm one published record.

        The consumer mutex spans doorbell acquisition, payload D2H, cloning,
        and the synchronized reset kernel. A second host consumer therefore
        cannot reset a newer producer claim using a stale doorbell observation.
        Malformed records are returned as ``Hit(valid=False)`` and consumed.
        """
        with self._consumer_mu:
            self._raise_if_poisoned()
            if not self.doorbell():
                return None
            try:
                owned = self._read_published_hit_locked().owned_copy()
            except BaseException as read_error:
                self._reset_or_poison_locked(read_error)
                raise
            self._reset_or_poison_locked()
            return owned

    def reset_hit(self) -> None:
        """Consume the pending hit and re-arm the signal.

        Ordering is load-bearing: a tiny device kernel clears the mapped
        record's magic + doorbell, performs a system fence, then atomically
        changes the producer latch from 1 to 0. The atomic reset participates
        in the same GPU-scope modification order as producer ``atomic_cas``
        claims, so a copy-engine store can never overwrite a concurrent claim.
        The final synchronize makes re-arming a postcondition of returning.
        """
        with self._consumer_mu:
            self._raise_if_poisoned()
            # lock=1/status=0 is an in-flight producer claim, not consumer
            # ownership. Blind reset would reopen the latch under its payload.
            if not self.doorbell():
                return
            self._reset_or_poison_locked()
