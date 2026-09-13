"""Functional host launch for fused stats, noising, quantization, and peel."""

import math
from dataclasses import dataclass

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .._utils._compile import get_or_compile
from .._utils._stream import get_stream
from .._utils._validation import require_tensor
from ..protocol_constants import (
    BLOCK_SCALE_GROUP,
    DELTA,
    NOISE_TARGET_NORM,
    PACKED_NOISE_K,
    PEEL_COLS,
    SM100_CC_MAJOR,
    R,
)
from ._kernel import (
    _OUT_STAGES,
    _SMEM_CAPACITY_BYTES,
    NoiseLoadMode,
    _NoisyQuant,
)
from ._quantization_ops import _L_E1, _noise_base_words


@dataclass(frozen=True)
class NoisyQuantConfig:
    """Requested compile-time noising configuration.

    The grid is one CTA per ``noise_rows`` row block, so every reduction is
    one fixed-order chain.
    """

    noise_bk: int = 128
    noise_stages: int = 4
    noise_out_stages: int = _OUT_STAGES
    noise_load_mode: NoiseLoadMode | str = NoiseLoadMode.MERGED
    noise_rows: int = 32
    # TODO(perf): revisit factor-tile cluster multicast if traffic patterns change.

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "noise_load_mode",
            NoiseLoadMode(self.noise_load_mode),
        )


# Shared default for the public entry point; the config is frozen, so one
# instance is safe to reuse as an argument default.
_DEFAULT_CONFIG = NoisyQuantConfig()

_compile_cache: dict[tuple, object] = {}


# Shared-memory geometry mirrored from _kernel.py's SharedStorage:
_A_HALF_TILE_ROWS = 16  # A is staged in 16-row half-tiles
_F1_BYTES_PER_COLUMN = PACKED_NOISE_K  # e4m3 codes zero-padded to the atom's K
_F2_BYTES_PER_COLUMN = R
_APRIME_STAGE_ROWS = 64  # A' stages hold whole 64-row peel-operand tiles
_F32_BYTES = 4
_NOISE_SKEW_ELEMS = 16  # sNoise's per-row f32 skew (bank spread)
_E1_OPERAND_BYTES = 64 * PACKED_NOISE_K  # sE1B: one (64, PACKED_NOISE_K) e4m3 tile
_MBARRIER_BYTES = 8
_ALIGNED_REGIONS = 7  # 1 KiB-aligned SharedStorage regions, worst-case pad each


def _smem_bytes(k: int, config: NoisyQuantConfig) -> int:
    """Upper-bound the kernel's SharedStorage for one configuration.

    Mirrors ``_kernel.py``'s ``SharedStorage`` term by term, with alignment
    counted at its worst case so the estimate only over-reserves. The
    kernel's compile-time size assert remains the hard gate.
    """
    rows, bk = config.noise_rows, config.noise_bk
    stages, out_stages = config.noise_stages, config.noise_out_stages
    row_halves = rows // _A_HALF_TILE_ROWS
    if config.noise_load_mode == NoiseLoadMode.RESIDENT:
        a_stage_count = row_halves * k // bk
    elif config.noise_load_mode == NoiseLoadMode.RING:
        a_stage_count = stages
    else:
        a_stage_count = row_halves * stages
    # A is staged as its two blobs: one code byte per element plus one BF16
    # scale per BLOCK_SCALE_GROUP elements, i.e. 1.25 bytes/element instead of 2.
    a_bytes_per_row = bk + 2 * (bk // BLOCK_SCALE_GROUP)
    # factor/f2 pipes (2 each per stage), A ring, output + peel pipes, the
    # two-stage acc pipe (4), and peel_done (2).
    mbarrier_count = 4 * stages + 2 * a_stage_count + 4 * out_stages + 6
    # sAlpha/sBeta (f32 each), sE1 (f16 x rows x R).
    scalar_bytes = 2 * _F32_BYTES * rows + 2 * rows * R
    return (
        _A_HALF_TILE_ROWS * a_bytes_per_row * a_stage_count
        + (_F1_BYTES_PER_COLUMN + _F2_BYTES_PER_COLUMN) * bk * stages
        + _APRIME_STAGE_ROWS * bk * out_stages
        + _F32_BYTES * rows * (bk + _NOISE_SKEW_ELEMS)
        + _E1_OPERAND_BYTES
        + _MBARRIER_BYTES * mbarrier_count
        + scalar_bytes
        + _ALIGNED_REGIONS * 1024
    )


def _require_int_tunable(name: str, value: object, allowed: tuple[int, ...]) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an int, got {value!r} ({type(value).__name__})")
    if value not in allowed:
        raise ValueError(f"{name} must be {', '.join(map(str, allowed[:-1]))}, or {allowed[-1]}")


def _validate_noisy_quant_tunables(config: NoisyQuantConfig) -> None:
    """Reject tunable values the kernel has no code path for."""
    _require_int_tunable("noise_rows", config.noise_rows, (16, 32, 64))
    # bk == 64 would make the noise UMMA an M=64 tile, whose interleaved
    # TMEM fragment none of the t2r drain paths cover.
    _require_int_tunable("noise_bk", config.noise_bk, (128, 256))
    _require_int_tunable("noise_stages", config.noise_stages, (2, 3, 4))
    _require_int_tunable("noise_out_stages", config.noise_out_stages, (2, 3, 4))
    if not isinstance(config.noise_load_mode, NoiseLoadMode):
        raise ValueError("noise_load_mode must be a NoiseLoadMode")


def validate_noisy_quant_config(m: int, k: int, config: NoisyQuantConfig) -> None:
    """Reject tuning points the kernel cannot express or fit in shared memory."""
    _validate_noisy_quant_tunables(config)
    if m <= 0:
        raise ValueError("m must be positive")
    if k % 512:
        raise ValueError("k must be divisible by 512")
    if m % config.noise_rows:
        raise ValueError("m must be divisible by noise_rows")
    if m >= 1 << 32:
        raise ValueError("row index does not fit the E1 message's u32 line index")
    if _smem_bytes(k, config) > _SMEM_CAPACITY_BYTES:
        raise ValueError("noisy_quant configuration exceeds shared memory capacity")


def pack_noise_factor(f1: torch.Tensor) -> torch.Tensor:
    """Repack ``F1`` into the ``(k, PACKED_NOISE_K)`` int8 blob ``noisy_quant`` consumes.

    ``f1``: the (R, k) e4m3 factor from the commitment chain (same
    orientation as ``f2``). On SM100 the blob is the transposed e4m3 codes
    themselves, zero-padded to the f8f6f4 atom's K (``PACKED_NOISE_K``),
    TMA-staged directly as the noise UMMA's A operand. The zero padding is
    the reference's own atom-K zero-pad, so pad products are exact zeros.
    When ``R`` already equals the atom K there is no pad, and the blob is
    ``noise_lines``' ``(k, R)`` output viewed as int8 -- callers on the timed
    path may draw the lines straight into the blob instead of calling this.

    Any finite e4m3 code is a legal operand: at R = 32 the smallest
    normalized line entries are below 0.5.
    """
    if f1.dtype != torch.float8_e4m3fn or f1.dim() != 2 or f1.shape[0] != R:
        raise ValueError(f"f1 must be ({R}, k) float8_e4m3fn, got {f1.dtype} {tuple(f1.shape)}")
    if not torch.isfinite(f1.float()).all():
        raise ValueError("f1 contains non-finite e4m3 codes")
    codes = f1.view(torch.int8).t().contiguous()
    if PACKED_NOISE_K == R:
        return codes
    pad = torch.zeros(codes.shape[0], PACKED_NOISE_K - R, dtype=codes.dtype, device=codes.device)
    return torch.cat([codes, pad], dim=1).contiguous()


def _scale_constants() -> tuple[float, float]:
    """BF16-valued ``DELTA * sqrt(R)`` pair matching the reference ``const`` path.

    R=16 made both values powers of two; that is not a protocol requirement.
    """
    delta_r = float(torch.tensor(DELTA * math.sqrt(R), dtype=torch.bfloat16))
    delta_over_std = float(
        torch.tensor(
            DELTA * math.sqrt(R) / (NOISE_TARGET_NORM * NOISE_TARGET_NORM),
            dtype=torch.bfloat16,
        )
    )
    return delta_r, delta_over_std


# Caller-facing buffer names of the A-side entry point, in the shared role
# order of ``_launch_prep`` (the B-side names live in ``noisy_quant_b``).
_A_SIDE_NAMES = (
    "codes",
    "scales",
    "c_a",
    "commit_stats",
    "f1_hl",
    "f2",
    "alpha",
    "beta",
    "e1",
    "a_prime",
    "a_peel",
)


def _launch_prep(
    codes: torch.Tensor,
    scales: torch.Tensor,
    key: torch.Tensor,
    commit_stats: torch.Tensor,
    factor_hl: torch.Tensor,
    factor: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    e_lines: torch.Tensor,
    primed: torch.Tensor,
    peel: torch.Tensor,
    *,
    entry: str,
    names: tuple[str, ...],
    msg_base: tuple[int, ...],
    config: NoisyQuantConfig,
) -> None:
    """Validated launch shared by the A- and B-side prep entry points.

    The two sides run the identical kernel on mirrored operands; only the
    caller-facing buffer ``names``, the noise key, and the E-line ``msg_base``
    (label and instance index) differ. ``msg_base`` is part of the compile
    cache key, so both sides share ``_compile_cache``.
    """
    if codes.ndim != 2:
        raise ValueError(f"{names[0]} must be 2D")
    m, k = codes.shape
    validate_noisy_quant_config(m, k, config)
    device = codes.device
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] != SM100_CC_MAJOR:
        raise ValueError(
            f"{entry} requires SM100, got sm{device_capability[0]}{device_capability[1]}"
        )

    specs = (
        (codes, torch.int8, (m, k), 16),
        (scales, torch.bfloat16, (m, k // BLOCK_SCALE_GROUP), 16),
        (key, torch.uint8, (32,), 4),
        (commit_stats, torch.float32, (2 * (m * k // 512),), 16),
        (factor_hl, torch.int8, (k, PACKED_NOISE_K), 16),
        (factor, torch.float8_e4m3fn, (R, k), 16),
        (alpha, torch.bfloat16, (m,), 16),
        (beta, torch.bfloat16, (m,), 16),
        (e_lines, torch.float8_e4m3fn, (m, R), 16),
        (primed, torch.float8_e4m3fn, (m, k), 16),
        (peel, torch.bfloat16, (m, PEEL_COLS), 16),
    )
    for name, (tensor, dtype, shape, alignment) in zip(names, specs, strict=True):
        require_tensor(name, tensor, dtype=dtype, shape=shape, device=device, alignment=alignment)

    tensors = (
        codes,
        scales,
        alpha,
        beta,
        e_lines.view(-1),
        factor_hl,
        factor,
        primed,
        peel,
    )
    args = tuple(from_dlpack(t, assumed_align=16) for t in tensors)
    args += (
        from_dlpack(key.view(torch.uint32)),
        from_dlpack(commit_stats, assumed_align=16),
    )
    stream = get_stream(device.index or 0)
    cache_key = (device_capability, m, k, config, msg_base)

    def compile_variant():
        return cute.compile(
            _NoisyQuant(
                config.noise_bk,
                config.noise_stages,
                out_stages=config.noise_out_stages,
                msg_base=msg_base,
                consts=_scale_constants(),
                load_mode=config.noise_load_mode,
                rows=config.noise_rows,
            ),
            *args,
            stream,
        )

    compiled = get_or_compile(_compile_cache, cache_key, compile_variant)
    compiled(*args, stream)


def noisy_quant(
    codes: torch.Tensor,
    scales: torch.Tensor,
    c_a: torch.Tensor,
    commit_stats: torch.Tensor,
    f1_hl: torch.Tensor,
    f2: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    e1: torch.Tensor,
    a_prime: torch.Tensor,
    a_peel: torch.Tensor,
    *,
    config: NoisyQuantConfig = _DEFAULT_CONFIG,
) -> None:
    """Launch noising into caller-owned outputs.

    ``codes`` and ``scales`` are the two committed block-scaled activation
    blobs -- the very bytes ``pre_quant`` wrote and ``tensor_hash_plus_stats``
    hashed -- so prep reads the protocol streams directly instead of a
    dequantized BF16 copy. ``c_a`` is A's 32-byte noise-line key
    (``Subkey("noise-line", noise seedA)``; the finalize kernel's ``a_keys``
    words ``[8, 16)``), under which the E1 rows are drawn in-kernel.
    ``f1_hl`` is A's own basis ``F_A`` (drawn under the same key, so it is
    per-nonce like ``c_a``) and ``f2`` is B's ``F_B``.
    """
    _launch_prep(
        codes,
        scales,
        c_a,
        commit_stats,
        f1_hl,
        f2,
        alpha,
        beta,
        e1,
        a_prime,
        a_peel,
        entry="noisy_quant",
        names=_A_SIDE_NAMES,
        msg_base=_noise_base_words(_L_E1),
        config=config,
    )
