"""Functional host launch for fused GEMM, lottery, peel, and unscale."""

from dataclasses import dataclass

import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import get_max_active_clusters

from .._utils._compile import get_or_compile
from .._utils._stream import get_stream
from .._utils._validation import require_tensor
from ..pow import HIT_PAYLOAD_K_ALIGN, HitSignal
from ..protocol_constants import BLOCK_SCALE_GROUP, SM100_CC_MAJOR
from ._kernel import (
    _EPI_THREADS,
    DEFAULT_LTILE_COLS,
    DEFAULT_LTILE_ROWS,
    R2,
    SUPPORTED_LTILE_COLS,
    SUPPORTED_LTILE_ROWS,
    _FusedGemmSm100,
    cta_tile_m,
)


@dataclass(frozen=True)
class MixedGemmConfig:
    """Compile-time kernel tuning parameters."""

    tile_m: int = 256
    tile_n: int = 128
    tile_k: int | None = None
    cluster_m: int = 2
    cluster_n: int = 1
    ltile_cols: int = DEFAULT_LTILE_COLS
    # Appended last on purpose: ``ltile_cols`` keeps its historical position, so
    # existing positional constructions stay valid.
    ltile_rows: int = DEFAULT_LTILE_ROWS


# Shared default for the public entry point; the config is frozen, so one
# instance is safe to reuse as an argument default.
_DEFAULT_CONFIG = MixedGemmConfig()

_compile_cache: dict[tuple, object] = {}


def _validate_tile_shape(config: MixedGemmConfig) -> None:
    """SM100 tile and cluster shape limits."""
    # Exact ints only: a float or bool can compare equal to an allowed value
    # (``64.0 == 64``, ``True == 1``) and still fail later in CuTe construction
    # or a bit-shift, matching ``_validate_lottery_tile``.
    for name, value in (
        ("tile_m", config.tile_m),
        ("tile_n", config.tile_n),
        ("cluster_m", config.cluster_m),
        ("cluster_n", config.cluster_n),
    ):
        if type(value) is not int:
            raise ValueError(f"{name} must be an int, got {value!r} ({type(value).__name__})")
    if config.tile_k is not None and type(config.tile_k) is not int:
        raise ValueError(
            f"tile_k must be an int, got {config.tile_k!r} ({type(config.tile_k).__name__})"
        )
    if config.tile_m not in (64, 128, 256):
        raise ValueError("SM100 tile_m must be 64, 128, or 256 (the MMA tiler M)")
    if config.tile_m == 64 and config.ltile_rows != 4:
        # The 64-row tile's TMEM loads split every accumulator row between
        # two threads, so no thread can fold a whole 16-row-family word.
        raise ValueError("tile_m 64 supports only the 4-row lottery family")
    if config.tile_n % 32 or not 32 <= config.tile_n <= 256:
        raise ValueError("SM100 tile_n must be a multiple of 32 in [32, 256]")
    if config.tile_k is not None and (config.tile_k < 2 * R2 or config.tile_k % 32):
        raise ValueError(
            f"SM100 tile_k must be None or a multiple of 32 at least {2 * R2} "
            "(BF16 peel columns must fit in one FP8 AB ring slot)"
        )
    if config.cluster_m & (config.cluster_m - 1) or config.cluster_n & (config.cluster_n - 1):
        raise ValueError("SM100 cluster dimensions must be powers of 2")
    # One epilogue thread hashes one message: the per-CTA message count must
    # fit the 128 epilogue threads (sub-warp counts are handled by the
    # kernel's warp-rounded hashing gate).
    msgs_cta = (cta_tile_m(config.tile_m) // config.ltile_rows) * (
        config.tile_n // config.ltile_cols
    )
    if msgs_cta > _EPI_THREADS:
        raise ValueError(
            "SM100 lottery needs (cta_tile_m / ltile_rows) * (tile_n / ltile_cols) "
            "to be no larger than 128"
        )


def _validate_lottery_tile(config: MixedGemmConfig) -> None:
    """The committed merged lottery tile: one of the two supported geometries,
    as exact ints.

    The kernel indexes and bit-shifts these (``ltile_rows.bit_length()``), and
    ``bool`` is an int subclass, so an integral float or a bool would pass the
    value checks and only fail later, during variant compilation.
    """
    for name, value in (("ltile_rows", config.ltile_rows), ("ltile_cols", config.ltile_cols)):
        if type(value) is not int:
            raise ValueError(f"{name} must be an int, got {value!r} ({type(value).__name__})")
    if config.ltile_rows not in SUPPORTED_LTILE_ROWS:
        raise ValueError(f"ltile_rows must be one of {SUPPORTED_LTILE_ROWS}")
    supported_ltile_cols = SUPPORTED_LTILE_COLS[config.ltile_rows]
    if config.ltile_cols not in supported_ltile_cols:
        raise ValueError(
            f"ltile_cols must be one of {supported_ltile_cols} for ltile_rows={config.ltile_rows}"
        )
    if config.tile_n % config.ltile_cols:
        raise ValueError("ltile_cols must divide tile_n")


def _validate_u32_tag(name: str, value: int) -> None:
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(
            f"{name} must be an int fitting an unsigned 32-bit record word, got "
            f"{value!r} ({type(value).__name__})"
        )


def validate_mixed_gemm_config(
    m: int,
    n: int,
    k: int,
    config: MixedGemmConfig,
) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    if k % HIT_PAYLOAD_K_ALIGN:
        raise ValueError(f"k must be divisible by {HIT_PAYLOAD_K_ALIGN}")
    _validate_lottery_tile(config)
    _validate_tile_shape(config)
    if config.cluster_m <= 0 or config.cluster_n <= 0:
        raise ValueError("cluster dimensions must be positive")
    if config.cluster_m * config.cluster_n > 8:
        raise ValueError("cluster size must not exceed 8 CTAs")
    if m % config.ltile_rows or n % config.ltile_cols:
        raise ValueError(
            f"lottery tiles must not straddle the problem edge: {m}x{n} "
            f"vs {config.ltile_rows}x{config.ltile_cols}"
        )
    # tile_m=256 selects the 2-CTA UMMA pair when cluster_m is even (128-row
    # CTAs either way); tile_m=64 runs the 64-row 1-CTA kernel.
    m_tiles = -(-m // cta_tile_m(config.tile_m))
    n_tiles = -(-n // config.tile_n)
    if m_tiles % config.cluster_m or n_tiles % config.cluster_n:
        raise ValueError(
            f"cluster {config.cluster_m}x{config.cluster_n} must divide the "
            f"{m_tiles}x{n_tiles} CTA grid"
        )


def _validate_record_hits(record_hits: bool) -> None:
    if type(record_hits) is not bool:
        raise TypeError(f"record_hits must be bool, got {type(record_hits).__name__}")


def _validate_output_operands(
    operands: dict[str, torch.Tensor],
    m: int,
    n: int,
    device: torch.device,
) -> None:
    """Validate the output and its peel/unscale operands."""
    for name, dtype, shape in (
        ("a_peel", torch.bfloat16, (m, R2)),
        ("b_peel", torch.bfloat16, (n, R2)),
        ("alpha_a", torch.bfloat16, (m,)),
        ("inv_alpha_b", torch.float32, (n,)),
        ("out", torch.bfloat16, (m, n)),
    ):
        require_tensor(
            name,
            operands[name],
            dtype=dtype,
            shape=shape,
            device=device,
            alignment=16,
        )


def _payload_views(
    hit_signal: HitSignal, a_codes: torch.Tensor, a_scales: torch.Tensor, m: int, k: int
) -> tuple[torch.Tensor | None, ...]:
    """Flat u32 (src codes, src scales, dst codes, dst scales) snapshot views.

    All None when the planes exceed the signal's capacity: the snapshot fits
    or the record publishes payload-less, decided host-side and compile-time,
    so payload-less kernels carry no copy code.

    Layout invariants (revalidate if the plane layout or the k alignment
    constraint ever changes): capacities, ``narrow`` lengths, and the record's
    payload fields all count BYTES (codes are int8 -> m*k bytes; scales are
    bf16 -> 2 bytes/elem); every plane is a whole number of 16-byte vectors
    (k % HIT_PAYLOAD_K_ALIGN == 0) on 16-byte-aligned storage (enforced by
    ``require_tensor`` in ``mixed_gemm`` and the allocations in ``HitSignal``),
    which is what makes the ``view(torch.uint32)`` reinterpretations and the
    kernel's vectorized copy legal.
    """
    scales_elems = m * (k // BLOCK_SCALE_GROUP)
    if (
        m * k > hit_signal.codes_payload_capacity_bytes
        or scales_elems * 2 > hit_signal.scales_payload_capacity_bytes
    ):
        return (None, None, None, None)
    return (
        a_codes.view(torch.uint32).reshape(-1),
        a_scales.view(torch.uint32).reshape(-1),
        hit_signal.codes_payload.narrow(0, 0, m * k).view(torch.uint32),
        hit_signal.scales_payload.narrow(0, 0, scales_elems).view(torch.uint32),
    )


def mixed_gemm(
    a_prime: torch.Tensor,
    b_prime: torch.Tensor,
    a_peel: torch.Tensor,
    b_peel: torch.Tensor,
    alpha_a: torch.Tensor,
    inv_alpha_b: torch.Tensor,
    pow_key: torch.Tensor,
    threshold: torch.Tensor,
    out: torch.Tensor,
    hit_signal: HitSignal,
    a_codes: torch.Tensor,
    a_scales: torch.Tensor,
    commitment_hash_b: torch.Tensor,
    *,
    config: MixedGemmConfig = _DEFAULT_CONFIG,
    layer_id: int = 0,
    record_hits: bool = True,
) -> None:
    """Launch fused GEMM into caller-owned outputs.

    ``hit_signal`` is the process-wide persistent hit signal (``pearl_gemm.
    pow.HitSignal``), the kernel's only channel for reporting a lottery hit:
    the first winner claims its latch, snapshots the proof-facing planes
    ``a_codes`` ((m, k) int8) and ``a_scales`` ((m, k/8) bf16) into the
    signal's payload regions (payload-less when they exceed capacity), and
    publishes a record stamped with ``pow_key``, ``threshold``,
    ``commitment_hash_b``, and ``layer_id``. ``pow_key`` is the v4 jackpot
    key ``Subkey("jackpot", noise seedA)`` (``a_keys`` bytes ``[64, 96)`` of
    the tensor-hash finalize): the key of the lottery compression ``J =
    blake3(extracted, key=pow_key)``. ``commitment_hash_b`` is any 32-byte
    B-side stamp the consumer uses to match the record to its job (the
    runtime stamps noise seedB); the kernel only copies it. ``layer_id`` is a runtime scalar
    (no recompile per value) identifying the launch's layer, so a consumer
    serving many layers through one signal (the useful miner) can look up the
    layer the hit belongs to; single-layer callers leave the default 0. It is
    serialized as an unsigned 32-bit record word and must be an ``int`` in
    ``[0, 2**32)``. ``record_hits=False`` still runs
    every lottery compression but prevents any latch claim, payload snapshot,
    or record publication; legacy/offline jobs use it to credit only executed
    work without retaining a winner.
    Like every other operand, ``a_codes`` and
    ``a_scales`` must be contiguous, 16-byte-aligned tensors of exactly
    those shapes on the launch device (validated below): the snapshot copy
    reinterprets their storage as whole 16-byte vectors, which the shapes
    guarantee since k % HIT_PAYLOAD_K_ALIGN == 0. Consume with
    ``hit_signal.read_hit()`` / ``reset_hit()``; see ``pow/_hit_signal.py``
    for the protocol.
    """
    if a_prime.ndim != 2 or b_prime.ndim != 2:
        raise ValueError("a_prime and b_prime must be 2D")
    m, k = a_prime.shape
    n, b_k = b_prime.shape
    if b_k != k:
        raise ValueError(f"b_prime must have k={k}, got {b_k}")

    device = a_prime.device
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] != SM100_CC_MAJOR:
        raise ValueError(
            f"mixed_gemm requires SM100, got sm{device_capability[0]}{device_capability[1]}"
        )
    validate_mixed_gemm_config(m, n, k, config)
    _validate_output_operands(
        {
            "out": out,
            "a_peel": a_peel,
            "b_peel": b_peel,
            "alpha_a": alpha_a,
            "inv_alpha_b": inv_alpha_b,
        },
        m,
        n,
        device,
    )
    _validate_u32_tag("layer_id", layer_id)
    _validate_record_hits(record_hits)
    if not isinstance(hit_signal, HitSignal):
        raise TypeError("hit_signal must be a pearl_gemm.pow.HitSignal")
    if hit_signal.device != device:
        raise ValueError(f"hit_signal must live on {device}, got {hit_signal.device}")
    hit_signal.require_usable()
    require_tensor(
        "a_prime",
        a_prime,
        dtype=torch.float8_e4m3fn,
        shape=(m, k),
        alignment=16,
    )
    require_tensor(
        "b_prime",
        b_prime,
        dtype=torch.float8_e4m3fn,
        shape=(n, k),
        device=device,
        alignment=16,
    )
    require_tensor(
        "pow_key",
        pow_key,
        dtype=torch.uint8,
        shape=(32,),
        device=device,
        alignment=16,
    )
    require_tensor(
        "threshold",
        threshold,
        dtype=torch.uint8,
        shape=(32,),
        device=device,
        alignment=16,
    )
    require_tensor(
        "a_codes",
        a_codes,
        dtype=torch.int8,
        shape=(m, k),
        device=device,
        alignment=16,
    )
    require_tensor(
        "a_scales",
        a_scales,
        dtype=torch.bfloat16,
        shape=(m, k // BLOCK_SCALE_GROUP),
        device=device,
        alignment=16,
    )
    require_tensor(
        "commitment_hash_b",
        commitment_hash_b,
        dtype=torch.uint8,
        shape=(32,),
        device=device,
        alignment=16,
    )
    payload_views = _payload_views(hit_signal, a_codes, a_scales, m, k)
    snapshot_payload = payload_views[0] is not None

    # Absent payload views stay ``None`` in their argument slot: the DSL then
    # has no operand there at all.
    tensors = (
        a_prime,
        b_prime,
        a_peel,
        b_peel,
        alpha_a,
        inv_alpha_b,
        pow_key.view(torch.uint32),
        threshold.view(torch.uint32),
        out,
        hit_signal.record.view(torch.uint32),
        hit_signal.lock,
        commitment_hash_b.view(torch.uint32),
        *payload_views,
    )
    args = tuple(from_dlpack(t, assumed_align=16) if t is not None else None for t in tensors)
    stream = get_stream(device.index or 0)
    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    cache_key = (
        device_capability,
        m,
        n,
        k,
        config,
        snapshot_payload,
    )

    def compile_variant():
        return cute.compile(
            _FusedGemmSm100(
                tile_m=config.tile_m,
                tile_n=config.tile_n,
                tile_k=config.tile_k,
                cluster_m=config.cluster_m,
                cluster_n=config.cluster_n,
                ltile_rows=config.ltile_rows,
                ltile_cols=config.ltile_cols,
                snapshot_payload=snapshot_payload,
            ),
            *args,
            Int32(layer_id),
            Int32(record_hits),
            Int32(max_active_clusters),
            stream,
        )

    compiled = get_or_compile(_compile_cache, cache_key, compile_variant)

    compiled(
        *args,
        Int32(layer_id),
        Int32(record_hits),
        Int32(max_active_clusters),
        stream,
    )
