"""Per-layer state for the FP16 (A100 / sm_80) plaintext scheme.

The parallel of :mod:`vllm_miner.state` for cert-v5 FP16 mining, deliberately
kept as its own module and registry so the FP8 ``LayerState`` is untouched.

How FP16 differs from FP8:

* **No prequantization.** FP8 encodes every weight into committed int8 planes +
  BF16 block scales (``PrequantMatrix``) and fuses the lottery into a mixed GEMM.
  FP16 commits the **raw FP16 weight rows** directly (``miner_base.fp16_commitment``
  commits the ``u16`` bit patterns), so this state just holds the BF16/FP16
  weight -- on GPU for the search, and a CPU copy for the certificate opening.
* **Serving is a plain linear.** The FP16 search
  (``pearl_gemm.fp16_miner.search_block``) only *searches* a noised matmul for a
  jackpot tile; it does not produce the layer's output. The layer therefore
  serves an ordinary BF16 matmul and runs the search opportunistically as a side
  effect (``vllm_miner.fp16_mining``).
* **No steady per-job B buffers / HitSignal.** FP16 derives its whole seed chain
  and noise inside ``search_block`` per forward from the header + operands, and
  returns the decoded winner synchronously, so none of the FP8 ``LayerBuffers`` /
  persistent hit-signal machinery applies.

The committed A100 tile is fixed (``zk-pow/src/api/fp16/params.rs``):
``P_A = [(4, Blake)]`` (h=4), ``P_B = [(4, Blake), (16, Fold)]`` (w=64), r=32,
contiguous tile offsets.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

import torch
from miner_base.devices import Device, is_fp16_capable, local_device
from miner_base.layout import AxisPattern, DimType
from miner_utils import get_logger

_LOGGER = get_logger(__name__)
_NEXT_FP16_LAYER_ID = itertools.count(1)

# The committed A100 lottery patterns and noise rank (bit-identical to the
# fixture in miner_base.fp16_block_submission / zk-pow fp16/params.rs).
FP16_ROWS_PATTERN = AxisPattern(((4, DimType.BLAKE),))  # h = 4
FP16_COLS_PATTERN = AxisPattern(((4, DimType.BLAKE), (16, DimType.FOLD)))  # w = 64
FP16_NOISE_RANK = 32
FP16_TILE_H = FP16_ROWS_PATTERN.tile_size  # 4
FP16_TILE_W = FP16_COLS_PATTERN.tile_size  # 64


def fp16_mines_on(device: torch.device) -> bool:
    """Whether ``device`` is an sm_80 (A100/GA100) the FP16 kernels run on.

    The FP16 analogue of ``state._mines_on`` -- and deliberately disjoint from
    it: ``state._mines_on`` admits SM90/100/120 for FP8 and omits SM80, so
    SM80 stays rejected for FP8 while this admits it for FP16 only."""
    return is_fp16_capable(device)


def _fp16_kernels_available() -> bool:
    try:
        import pearl_gemm.fp16_miner  # noqa: F401
    except Exception:
        _LOGGER.opt(exception=True).warning("pearl_gemm.fp16_miner import failed; no FP16 mining")
        return False
    return True


def can_mine_fp16_layer(n: int, k: int, device: torch.device) -> bool:
    """Whether this device/shape runs the FP16 (A100) search.

    Admits sm_80 **only** (FP8 families are served by ``state.can_mine_layer``),
    requires the FP16 kernels, and needs the committed tile to divide the
    weight: ``n`` a multiple of w=64 (the search scans ``n // w`` column tiles;
    a trailing partial tile is never committed). ``k`` is unconstrained -- the
    FP16 Merkle commitment zero-pads each row to the hash-id leaf."""
    if not fp16_mines_on(device) or not _fp16_kernels_available():
        return False
    return n > 0 and k > 0 and n % FP16_TILE_W == 0


@dataclass
class Fp16LayerState:
    """Everything one FP16 mining-enabled linear layer carries across forwards.

    ``weight`` is the raw ``(n, k)`` FP16/BF16 weight on CUDA (the committed B
    operand and the serving operand both); ``weight_cpu`` is the stable host
    copy the certificate opening commits from (bit-identical ``u16`` patterns).
    Unlike FP8 there are no int8 planes, block scales, steady B buffers, or FP8
    fallback operands.
    """

    layer_name: str
    layer_id: int
    weight: torch.Tensor  # (n, k) float16, contiguous, CUDA -- committed B
    weight_cpu: torch.Tensor  # (n, k) float16 host copy for the opening
    n: int
    k: int
    mineable: bool
    rows_pattern: AxisPattern = FP16_ROWS_PATTERN
    cols_pattern: AxisPattern = FP16_COLS_PATTERN
    r: int = FP16_NOISE_RANK
    lock: threading.Lock = field(default_factory=threading.Lock)
    disabled_reason: str | None = None

    @property
    def committed_device(self) -> Device:
        """The committed mining ``Device`` (always ``Device.A100`` for FP16)."""
        return local_device(self.weight.device, allow_fp16=True)

    def disable_mining(self, reason: str) -> None:
        """Permanently stop mining this layer (serving continues as a plain linear)."""
        with self.lock:
            first = self.disabled_reason is None
            if first:
                self.disabled_reason = reason
            self.mineable = False
        if first:
            _LOGGER.error(f"FP16 mining disabled for {self.layer_name}: {reason}")


def _to_committed_fp16(weight: torch.Tensor) -> torch.Tensor:
    """The committed FP16 view of a loaded 2D CUDA weight.

    BF16 weights are cast to FP16 (the committed operand dtype); an already-FP16
    weight is used as-is. The returned tensor is contiguous."""
    if weight.dim() != 2 or not weight.is_cuda:
        raise ValueError(f"expected a 2D CUDA weight, got {tuple(weight.shape)} on {weight.device}")
    if weight.dtype == torch.float16:
        committed = weight
    elif weight.dtype == torch.bfloat16:
        committed = weight.to(torch.float16)
    else:
        raise ValueError(f"FP16 mining expects a float16/bfloat16 weight, got {weight.dtype}")
    return committed.contiguous()


def create_fp16_layer_state(layer_name: str, weight: torch.Tensor) -> Fp16LayerState:
    """Build one FP16 layer state from a loaded weight (callers gate on
    :func:`can_mine_fp16_layer`)."""
    committed = _to_committed_fp16(weight)
    n, k = committed.shape
    if not can_mine_fp16_layer(n, k, committed.device):
        raise RuntimeError(f"FP16 layer {layer_name} (n={n}, k={k}) is not mineable on this device")
    return Fp16LayerState(
        layer_name=layer_name,
        layer_id=next(_NEXT_FP16_LAYER_ID),
        weight=committed,
        weight_cpu=committed.detach().to("cpu").clone(),
        n=n,
        k=k,
        mineable=True,
    )


_FP16_REGISTRY: dict[int, Fp16LayerState] = {}
_FP16_REGISTRY_LOCK = threading.RLock()


def register_fp16_state(state: Fp16LayerState) -> None:
    if not state.weight.is_contiguous():
        raise RuntimeError("FP16 mining weight must be contiguous for data_ptr lookup")
    pointer = state.weight.data_ptr()
    with _FP16_REGISTRY_LOCK:
        owner = _FP16_REGISTRY.get(pointer)
        if owner is not None and owner is not state:
            raise RuntimeError(f"FP16 mining weight pointer {pointer} is already registered")
        _FP16_REGISTRY[pointer] = state


def unregister_fp16_state(state: Fp16LayerState) -> None:
    with _FP16_REGISTRY_LOCK:
        if _FP16_REGISTRY.get(state.weight.data_ptr()) is state:
            del _FP16_REGISTRY[state.weight.data_ptr()]


def lookup_fp16_state(weight: torch.Tensor) -> Fp16LayerState | None:
    """The FP16 layer state registered for ``weight``, or ``None``.

    Returns ``None`` for FP8-registered or unregistered weights, so the linear
    op can probe FP16 first and fall through to the FP8 path unchanged."""
    with _FP16_REGISTRY_LOCK:
        return _FP16_REGISTRY.get(weight.data_ptr())


def all_fp16_states() -> list[Fp16LayerState]:
    with _FP16_REGISTRY_LOCK:
        return list(_FP16_REGISTRY.values())


def clear_fp16_registry() -> None:
    with _FP16_REGISTRY_LOCK:
        _FP16_REGISTRY.clear()
