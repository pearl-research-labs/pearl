"""CPU B preparation: compute one job's complete B side on the host via pearl_mining kernels."""

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from fractions import Fraction

import numpy as np
import pearl_mining
import torch
from blake3 import blake3
from miner_base.commitment import commit_planes
from miner_base.prequant import DEFAULT_BLOCK_SIZE, PrequantMatrix, RowNorms, round_l2_to_grid
from miner_base.protocol import Device, MiningConfiguration, noise_line_key, noise_seed_b
from pearl_gateway.comm.dataclasses import MiningJob

from .mining_config import (
    PACKED_NOISE_K,
    RANK,
    commitment_keys_for,
    expert_n,
    mining_configuration,
    threshold_bytes_for,
)
from .state import BProofContext

# Protocol constants duplicated here so CPU B helper imports without CuTe/CUDA.
NOISE_TARGET_NORM = 256
QUANT_MAX = 448.0
LABEL_F1 = bytes((0, 1))
LABEL_E2 = bytes((1, 0))
LABEL_F2 = bytes((1, 1))
STRIP_G = 0.08
STRIP_G_SMALL_RMS = 0.1
STRIP_G_HOPPER = 0.105

_NORM_FLOOR = 2.0**-32
_BLAKE3_CHUNK_LEN = 1024
_DIGEST_MAX_THREADS = 16
PREPARE_CHUNK_CELLS = 1 << 17
_BF16 = torch.bfloat16
_E4M3 = torch.float8_e4m3fn
_L2_FRAME_BITS = 27
_BF16_FRACTION_BITS = 7
_BF16_UNIT_EXP = 127 + _BF16_FRACTION_BITS
_BF16_INF_BITS = 0x7F80


@dataclass(frozen=True)
class DeviceArithmetic:
    delta: float
    window_bits: int
    strip_g: tuple[float, float]


ARITHMETIC = {
    Device.BLACKWELL: DeviceArithmetic(
        delta=0.5, window_bits=26, strip_g=(STRIP_G, STRIP_G_SMALL_RMS)
    ),
    Device.HOPPER: DeviceArithmetic(
        delta=1.0, window_bits=14, strip_g=(STRIP_G_HOPPER, STRIP_G_HOPPER)
    ),
}


@dataclass(frozen=True)
class CpuBContext:
    job: MiningJob
    config: MiningConfiguration
    key_a: bytes
    key_b: bytes
    seed_b: bytes
    p_b: bytes
    threshold: bytes
    b_proof: BProofContext
    f1: torch.Tensor
    f1_hl: torch.Tensor
    f2: torch.Tensor
    e2: torch.Tensor
    alpha_b: torch.Tensor
    beta_b: torch.Tensor
    inv_alpha_b: torch.Tensor
    b_prime: torch.Tensor
    b_peel: torch.Tensor


@dataclass(frozen=True)
class RowScales:
    alpha_b: torch.Tensor
    beta_b: torch.Tensor
    l2_b: torch.Tensor


def prepare_b_cpu(
    codes: torch.Tensor,
    scales: torch.Tensor,
    job: MiningJob,
    *,
    device: Device,
    experts: int = 0,
    cancelled: Callable[[], bool] | None = None,
    row_scales: RowScales | None = None,
) -> tuple[CpuBContext, RowScales] | None:
    """Prepare ``job``'s B side (keyed by its parent) on the host CPU."""
    arithmetic = ARITHMETIC.get(device)
    if arithmetic is None:
        raise ValueError(f"CPU B preparation has no arithmetic for {device!r}")
    if codes.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("CPU B preparation reads host planes")

    n, k = codes.shape
    config = mining_configuration(k, n, experts, device=device)
    key_a, key_b = commitment_keys_for(job)
    p_b = config.p_b(n)
    digest = _planes_digest([codes, scales], key_b, config.chunk_size)
    seed_b = noise_seed_b(digest, key_b, p_b)
    noise_key = noise_line_key(seed_b)

    f1 = _lines(noise_key, LABEL_F1, k).t().contiguous()
    f2 = _lines(noise_key, LABEL_F2, k).t().contiguous()
    f1_hl = torch.zeros(k, PACKED_NOISE_K, dtype=torch.int8)
    f1_hl[:, :RANK] = f1.view(torch.int8).t()
    e2 = _lines(noise_key, LABEL_E2, n)

    if row_scales is None:
        row_scales = _layer_row_scales(codes, scales, arithmetic.delta, cancelled)
    if row_scales is None or (cancelled is not None and cancelled()):
        return None

    b_prime, b_peel = _native_rows(codes, scales, row_scales, e2, f1, f2, arithmetic)
    alpha_b, beta_b = row_scales.alpha_b, row_scales.beta_b

    prepared = CpuBContext(
        job=job,
        config=config,
        key_a=key_a,
        key_b=key_b,
        seed_b=seed_b,
        p_b=p_b,
        threshold=threshold_bytes_for(job, k, expert_n(n, experts), device=device),
        b_proof=b_proof_for(config, job, key_b, seed_b, p_b, codes, scales),
        f1=f1,
        f1_hl=f1_hl,
        f2=f2,
        e2=e2,
        alpha_b=alpha_b,
        beta_b=beta_b,
        inv_alpha_b=torch.reciprocal(alpha_b.float()),
        b_prime=b_prime,
        b_peel=b_peel,
    )
    return prepared, row_scales


def b_proof_for(
    config: MiningConfiguration,
    job: MiningJob,
    key_b: bytes,
    seed_b: bytes,
    p_b: bytes,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> BProofContext:
    return BProofContext(
        key_b=key_b,
        ancestor_header=bytes(job.parent_header),
        seed_b=seed_b,
        p_b=p_b,
        codes=codes,
        scales=scales,
        commit_leaf=config.chunk_size,
    )


def without_b_proof(prepared: CpuBContext) -> dict[str, object]:
    return {f.name: getattr(prepared, f.name) for f in fields(CpuBContext) if f.name != "b_proof"}


def with_b_proof(
    detached: dict[str, object], codes: torch.Tensor, scales: torch.Tensor
) -> CpuBContext:
    proof = b_proof_for(
        detached["config"],
        detached["job"],
        detached["key_b"],
        detached["seed_b"],
        detached["p_b"],
        codes,
        scales,
    )
    return CpuBContext(**detached, b_proof=proof)


def _planes_digest(planes: list[torch.Tensor], key: bytes, leaf: int) -> bytes:
    if leaf != _BLAKE3_CHUNK_LEN:
        return commit_planes(planes, key, leaf).digest
    roots = []
    for plane in planes:
        data = plane.contiguous().view(torch.uint8).flatten().numpy()
        hasher = blake3(key=key, max_threads=min(torch.get_num_threads(), _DIGEST_MAX_THREADS))
        hasher.update(data)
        hasher.update(bytes(-data.size % leaf))
        roots.append(hasher.digest())
    return blake3(b"".join(roots), key=key).digest()


def _bytes(t: torch.Tensor) -> np.ndarray:
    """A contiguous host tensor's buffer, by element width."""
    view = {1: torch.uint8, 2: torch.uint16}[t.element_size()]
    return t.view(view).numpy()


def _lines(key: bytes, label: bytes, count: int) -> torch.Tensor:
    lines = torch.empty(count, RANK, dtype=_E4M3)
    side, factor = label
    pearl_mining.cpu_b_noise_lines(key, side, factor, _bytes(lines), torch.get_num_threads())
    return lines


def _native_rows(
    codes: torch.Tensor,
    scales: torch.Tensor,
    row_scales: RowScales,
    e2: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    arithmetic: DeviceArithmetic,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B', peel)`` from ``pearl_mining.cpu_b_prepare_rows``."""
    n, k = codes.shape
    b_prime = torch.empty(n, k, dtype=_E4M3)
    mid = torch.empty(n, RANK, dtype=_BF16)
    pearl_mining.cpu_b_prepare_rows(
        codes.contiguous().numpy(),
        *(
            _bytes(t.contiguous())
            for t in (scales, row_scales.alpha_b, row_scales.beta_b, row_scales.l2_b)
        ),
        *(_bytes(t.contiguous()) for t in (e2, f1, f2)),
        arithmetic.window_bits,
        *arithmetic.strip_g,
        _bytes(b_prime),
        _bytes(mid),
        torch.get_num_threads(),
    )
    peel = torch.cat([mid, -(row_scales.beta_b.unsqueeze(1) * e2.to(_BF16))], dim=1)
    return b_prime, peel


def _layer_row_scales(
    codes: torch.Tensor,
    scales: torch.Tensor,
    delta: float,
    cancelled: Callable[[], bool] | None,
) -> RowScales | None:
    n, k = codes.shape
    chunk_rows = max(1, PREPARE_CHUNK_CELLS // k)
    alpha_b, beta_b, l2_b = (torch.empty(n, dtype=_BF16) for _ in range(3))
    for start in range(0, n, chunk_rows):
        if cancelled is not None and cancelled():
            return None
        rows = slice(start, min(start + chunk_rows, n))
        alpha_b[rows], beta_b[rows], l2_b[rows] = _row_scales(
            PrequantMatrix(codes[rows], scales[rows]), delta
        )
    return RowScales(alpha_b=alpha_b, beta_b=beta_b, l2_b=l2_b)


def _row_scales(
    planes: PrequantMatrix, delta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    norms = exact_row_norms(planes)
    floor = torch.tensor(_NORM_FLOOR, dtype=_BF16)
    l2, linf = torch.maximum(norms.l2, floor), torch.maximum(norms.linf, floor)
    delta_r = torch.tensor(delta * math.sqrt(RANK), dtype=_BF16)
    delta_over_std = torch.tensor(delta * math.sqrt(RANK) / NOISE_TARGET_NORM**2, dtype=_BF16)
    alpha = torch.tensor(QUANT_MAX, dtype=_BF16) / _fma_bf16(delta_r, l2, linf)
    beta = (alpha * l2) * delta_over_std
    return alpha.squeeze(1), beta.squeeze(1), l2.squeeze(1)


def exact_row_norms(planes: PrequantMatrix) -> RowNorms:
    """Exact integer-frame row norms matching ``zk-pow`` ``prequant::exact_norms``."""
    n, k = planes.int_values.shape
    ceil_log2_k = (k - 1).bit_length()
    if ceil_log2_k > _L2_FRAME_BITS:
        raise ValueError(f"k={k} exceeds the verifier's L2 frame width")
    frame_bits = _L2_FRAME_BITS - ceil_log2_k
    ints = planes.int_values.to(torch.int64).reshape(n, -1, DEFAULT_BLOCK_SIZE)
    codes = planes.scales.view(torch.int16).to(torch.int64) & 0x7FFF
    exponent = codes >> _BF16_FRACTION_BITS
    significand = (codes & ((1 << _BF16_FRACTION_BITS) - 1)) | torch.where(
        exponent > 0, 1 << _BF16_FRACTION_BITS, 0
    )
    doubled_exp = 2 * exponent.clamp(min=1)
    block_terms = significand * significand * (ints * ints).sum(dim=-1)
    e_max = torch.where(block_terms != 0, doubled_exp, 0).amax(dim=1, keepdim=True)
    shift = (e_max - doubled_exp).clamp(min=0, max=63)
    frame_sums = ((block_terms << frame_bits) >> shift).sum(dim=1)
    l2_bits = [
        _rne_sqrt_bf16(Fraction(s) * Fraction(2) ** (e - 2 * _BF16_UNIT_EXP - frame_bits) / k)
        if s
        else 0
        for s, e in zip(frame_sums.tolist(), e_max.squeeze(1).tolist(), strict=True)
    ]
    l2 = round_l2_to_grid(torch.tensor(l2_bits, dtype=torch.int16).view(_BF16)).unsqueeze(1)
    if (l2.view(torch.int16) >= _BF16_INF_BITS).any():
        raise ValueError("a row's l2 rounds into the BF16 infinity code")
    linf = (
        (ints.abs().amax(dim=-1).float() * planes.scales.float().abs())
        .amax(dim=-1, keepdim=True)
        .to(_BF16)
    )
    return RowNorms(l2=l2, linf=linf)


def _bf16_value(bits: int) -> Fraction:
    exponent = bits >> _BF16_FRACTION_BITS
    significand = bits & ((1 << _BF16_FRACTION_BITS) - 1)
    if exponent:
        significand |= 1 << _BF16_FRACTION_BITS
    return significand * Fraction(2) ** (max(exponent, 1) - _BF16_UNIT_EXP)


def _rne_sqrt_bf16(square: Fraction) -> int:
    def past_midpoint(low: int) -> bool:
        midpoint = (_bf16_value(low) + _bf16_value(low + 1)) / 2
        boundary = midpoint * midpoint
        return square > boundary or (square == boundary and low % 2 == 1)

    seed = torch.tensor(math.sqrt(float(square)), dtype=torch.float32).to(_BF16)
    bits = int(seed.view(torch.int16))
    while past_midpoint(bits):
        bits += 1
    while bits and not past_midpoint(bits - 1):
        bits -= 1
    return bits


def _fma_bf16(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``a * b + c`` on BF16 operands with a single round-to-nearest-even to BF16."""
    a64, b64, c64 = a.to(torch.float64), b.to(torch.float64), c.to(torch.float64)
    product = a64 * b64
    total = product + c64
    rounded_part = total - c64
    residual = (product - rounded_part) + (c64 - (total - rounded_part))
    total32 = total.to(torch.float32)
    error = (total - total32.to(torch.float64)) + residual
    to_odd = (error != 0) & ((total32.view(torch.int32) & 1) == 0) & total32.isfinite()
    toward = torch.where(error > 0, torch.inf, -torch.inf).to(torch.float32)
    return torch.where(to_odd, torch.nextafter(total32, toward), total32).to(_BF16)
