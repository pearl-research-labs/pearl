"""Functional host launch for fused stats, noising, quantization, and peel."""

import math
from dataclasses import dataclass

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .._utils._arch import Arch, arch_of, require_arch
from .._utils._compile import get_or_compile
from .._utils._stream import get_stream
from .._utils._validation import require_tensor
from ..protocol_constants import (
    BLOCK_SCALE_GROUP,
    DELTA,
    NOISE_TARGET_NORM,
    PACKED_NOISE_K,
    PEEL_COLS,
    R,
)
from ._kernel import (
    _OUT_STAGES,
    NoiseLoadMode,
    _NoisyQuant,
)
from ._kernel_sm120 import _NoisyQuantSm120
from ._quantization_ops import _L_E1, _noise_base_words

# One fused kernel per architecture family; the host path, the tuning knobs
# and every output are shared.
_KERNELS: dict[Arch, type[_NoisyQuant]] = {
    Arch.SM100: _NoisyQuant,
    Arch.SM120: _NoisyQuantSm120,
}
_SUPPORTED_ARCHS = tuple(_KERNELS)


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


# Library defaults per family, for launches without a tuned record. SM120's
# ~99 KB CTA takes a shallower factor pipeline than SM100's 227 KB; the
# merged A pipe keeps its footprint k-independent.
_DEFAULT_CONFIGS: dict[Arch, NoisyQuantConfig] = {
    Arch.SM100: NoisyQuantConfig(),
    Arch.SM120: NoisyQuantConfig(noise_stages=3),
}


def default_noisy_quant_config(arch: Arch | None = None) -> NoisyQuantConfig:
    """The library default configuration for ``arch`` (the current device's when omitted)."""
    return _DEFAULT_CONFIGS[arch_of() if arch is None else arch]


_compile_cache: dict[tuple, object] = {}


# Shared-memory geometry mirrored from the kernels' SharedStorage (shared terms):
_A_HALF_TILE_ROWS = 16  # A is staged in 16-row half-tiles
_F1_BYTES_PER_COLUMN = PACKED_NOISE_K  # e4m3 codes zero-padded to the atom's K
_F2_BYTES_PER_COLUMN = R
_F32_BYTES = 4
_NOISE_SKEW_ELEMS = 16  # sNoise's per-row f32 skew (bank spread)
_MBARRIER_BYTES = 8
_WARP_LANES = 32


@dataclass(frozen=True)
class _SmemGeometry:
    """The SharedStorage terms that differ between the two kernel families."""

    # A' stage rows (None: the tile's own rows). The SM100 peel UMMA consumes
    # whole 64-row operand tiles; SM120 stages only what the TMA store needs.
    aprime_stage_rows: int | None
    # The SM100 TMEM drain's skewed f32 (rows, bk) staging tile; SM120
    # quantizes straight from the accumulator registers.
    noise_staging: bool
    # E1 operand image rows (None: the tile's own rows): a 64-row UMMA tile
    # on SM100, the (rows, K) ldmatrix tile on SM120.
    e1_operand_rows: int | None
    # SM120's cross-warp peel partials: the publishing warps' 16 x R f32.
    peel_partials_bytes: int
    # Extra mbarriers: SM100's peel pipe (per output stage) plus its
    # two-stage acc pipe (4) and peel_done (2).
    extra_mbarriers_per_out_stage: int
    extra_mbarriers: int
    # 1 KiB-aligned SharedStorage regions, worst-case pad each.
    aligned_regions: int


_SMEM_GEOMETRY: dict[Arch, _SmemGeometry] = {
    # sNoise, sE1B, sAq, sAs, sF1, sF2, sAp are 1 KiB-aligned.
    Arch.SM100: _SmemGeometry(
        aprime_stage_rows=64,
        noise_staging=True,
        e1_operand_rows=64,
        peel_partials_bytes=0,
        extra_mbarriers_per_out_stage=2,
        extra_mbarriers=6,
        aligned_regions=7,
    ),
    # sAq, sAs, sF1, sF2, sAp are 1 KiB-aligned; three warps publish partials.
    Arch.SM120: _SmemGeometry(
        aprime_stage_rows=None,
        noise_staging=False,
        e1_operand_rows=None,
        peel_partials_bytes=3 * _WARP_LANES * (_A_HALF_TILE_ROWS * R // _WARP_LANES) * _F32_BYTES,
        extra_mbarriers_per_out_stage=0,
        extra_mbarriers=0,
        aligned_regions=5,
    ),
}


def _smem_bytes(k: int, config: NoisyQuantConfig, arch: Arch) -> int:
    """Upper-bound the family's kernel SharedStorage for one configuration.

    Mirrors the ``SharedStorage`` of ``_kernel.py`` (SM100) or
    ``_kernel_sm120.py`` term by term, with alignment counted at its worst
    case so the estimate only over-reserves. The kernel's compile-time size
    assert remains the hard gate.
    """
    geometry = _SMEM_GEOMETRY[arch]
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
    # factor/f2 pipes (2 each per stage), A ring, output pipe (2 per stage)
    # and the family's extra pipes.
    mbarrier_count = (
        4 * stages
        + 2 * a_stage_count
        + (2 + geometry.extra_mbarriers_per_out_stage) * out_stages
        + geometry.extra_mbarriers
    )
    # sAlpha/sBeta (f32 each), sE1 (f16 x rows x R).
    scalar_bytes = 2 * _F32_BYTES * rows + 2 * rows * R
    aprime_rows = geometry.aprime_stage_rows or rows
    e1_rows = geometry.e1_operand_rows or rows
    noise_staging_bytes = (
        _F32_BYTES * rows * (bk + _NOISE_SKEW_ELEMS) if geometry.noise_staging else 0
    )
    return (
        _A_HALF_TILE_ROWS * a_bytes_per_row * a_stage_count
        + (_F1_BYTES_PER_COLUMN + _F2_BYTES_PER_COLUMN) * bk * stages
        + aprime_rows * bk * out_stages
        + noise_staging_bytes
        + e1_rows * PACKED_NOISE_K
        + geometry.peel_partials_bytes
        + _MBARRIER_BYTES * mbarrier_count
        + scalar_bytes
        + geometry.aligned_regions * 1024
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


def validate_noisy_quant_config(
    m: int, k: int, config: NoisyQuantConfig, arch: Arch | None = None
) -> None:
    """Reject tuning points the kernel cannot express or fit in shared memory.

    The shared-memory fit is per architecture family (SM120 has ~99 KB per
    CTA against SM100's 227 KB); ``arch`` defaults to the current device's.
    """
    _validate_noisy_quant_tunables(config)
    if m <= 0:
        raise ValueError("m must be positive")
    if k % 512:
        raise ValueError("k must be divisible by 512")
    if m % config.noise_rows:
        raise ValueError("m must be divisible by noise_rows")
    if m >= 1 << 32:
        raise ValueError("row index does not fit the E1 message's u32 line index")
    arch = arch_of() if arch is None else arch
    if _smem_bytes(k, config, arch) > arch.smem_capacity_bytes:
        raise ValueError("noisy_quant configuration exceeds shared memory capacity")


def pack_noise_factor(f1: torch.Tensor) -> torch.Tensor:
    """Repack ``F1`` into the ``(k, PACKED_NOISE_K)`` int8 blob ``noisy_quant`` consumes.

    ``f1``: the (R, k) e4m3 factor from the commitment chain (same
    orientation as ``f2``). The blob is the transposed e4m3 codes
    themselves, zero-padded to the FP8 atom's K (``PACKED_NOISE_K``), and
    is TMA-staged directly as the noise MMA's F1 operand on both families
    (the tcgen05 UMMA's A operand on SM100, the ``mma.sync`` B operand on
    SM120 -- the same K-major byte image). The zero padding is
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
    config: NoisyQuantConfig | None,
) -> None:
    """Validated launch shared by the A- and B-side prep entry points.

    The two sides run the identical kernel on mirrored operands; only the
    caller-facing buffer ``names``, the noise key, and the E-line ``msg_base``
    (label and instance index) differ. ``msg_base`` is part of the compile
    cache key, so both sides share ``_compile_cache``. ``config=None`` is the
    device family's library default.
    """
    if codes.ndim != 2:
        raise ValueError(f"{names[0]} must be 2D")
    m, k = codes.shape
    device = codes.device
    arch = require_arch(entry, device, *_SUPPORTED_ARCHS)
    if config is None:
        config = default_noisy_quant_config(arch)
    validate_noisy_quant_config(m, k, config, arch)
    device_capability = torch.cuda.get_device_capability(device)

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
            _KERNELS[arch](
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
    config: NoisyQuantConfig | None = None,
) -> None:
    """Launch noising into caller-owned outputs.

    ``codes`` and ``scales`` are the two committed block-scaled activation
    blobs -- the very bytes ``pre_quant`` wrote and ``tensor_hash_plus_stats``
    hashed -- so prep reads the protocol streams directly instead of a
    dequantized BF16 copy. ``c_a`` is A's 32-byte noise-line key
    (``Subkey("noise-line", noise seedA)``; the finalize kernel's ``a_keys``
    words ``[8, 16)``), under which the E1 rows are drawn in-kernel.
    ``f1_hl`` is the packed basis ``F_A`` and ``f2`` is B's ``F_B``; both are
    drawn under B's noise-line key (``LABEL_F1`` / ``LABEL_F2``), so they are
    job constants, unlike the per-nonce ``c_a``. ``config=None`` selects
    the device family's library default (``default_noisy_quant_config``).
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
