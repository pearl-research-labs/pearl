"""CPU B preparation: one job's complete B side from a layer's host planes.

:func:`prepare_b_cpu` computes, in host memory and without CUDA, every operand
:func:`~vllm_miner.job_prep.prepare_layer` writes on the GPU (B', its peel,
the per-row scales, both F bases and E_B) together with the job metadata a
launch and a proof need. The layer is dense, or the stacked
``(experts * n_e, k)`` B of an MoE layer: that B is committed and noised like
a dense one, and only its committed configuration (hence ``pB`` and
``seedB``) and its lottery threshold are per expert.

It composes only the protocol's arithmetic: the verifier's exact row norms,
keyed-BLAKE3 noise lines, the fp32 fold ``fma(beta, E@F, (alpha * scale) *
q)`` with the e4m3 cast (and the strip rewrite on Blackwell), and the
committed device's FP8 noise-dot atom: one K = 32 group from +0 whose terms
are truncated to the device's window (Blackwell's ``tcgen05.mma`` and SM120's
``mma.sync``: 26 bits; Hopper's QGMMA: 14). Everything is therefore
bit-identical to the verifier's B side except the peel's mid half ``(beta
(.) E_B@F_B - B') @ F_A^T``, the scheme's deviable surface, held to
tolerance against fp64.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from fractions import Fraction

import torch
from blake3 import blake3
from miner_base.block_submission import commit_planes_for_leaf
from miner_base.commitment import MiningConfiguration
from miner_base.commitment_hash import noise_line_key, noise_seed_b
from miner_base.params import Device
from miner_base.prequant import PrequantMatrix, RowNorms, round_l2_to_grid
from miner_base.transcript import encode_u32_le
from pearl_gateway.comm.dataclasses import MiningJob

from .b_prepare_native import native
from .mining_config import (
    PACKED_NOISE_K,
    RANK,
    commitment_keys_for,
    expert_n,
    mining_configuration,
    threshold_bytes_for,
)
from .state import BProofContext

# pearl_gemm.protocol_constants and noise_lines' address labels, duplicated so
# this module imports without the CuTe DSL (which creates a CUDA context).
NOISE_TARGET_NORM = 256
INT_SQRT_PREC = 32
QUANT_MAX = 448.0
LABEL_F1 = bytes((0, 1))
LABEL_E2 = bytes((1, 0))
LABEL_F2 = bytes((1, 1))
STRIP_G = 0.08
STRIP_G_SMALL_RMS = 0.1
STRIP_SMALL_RMS = 8.0
STRIP_EXP_OFFSET = 9
STRIP_COIN_BLOCK = 4
STRIP_COIN_SHIFT = 8
STRIP_COIN_THRESHOLD = 112
# Row-norm floor of the per-row scale chain (a zero row would divide by zero).
_NORM_FLOOR = 2.0**-32
_NOISE_MESSAGE_BYTES = 64  # one BLAKE3 block per line address
_BLAKE3_CHUNK_LEN = 1024
# A plane's hash is memory-bound and stops scaling near this many threads.
_DIGEST_MAX_THREADS = 16

# Output cells per row chunk. Every B-side row depends only on its own B row
# and the shared F bases, so chunking bounds the transient (the noise-dot
# emulation holds ``RANK`` int64 products per cell) without changing a bit.
PREPARE_CHUNK_CELLS = 1 << 17

_BF16 = torch.bfloat16
_E4M3 = torch.float8_e4m3fn
_E4M3_MAGNITUDE_MASK = 0x7F


@dataclass(frozen=True)
class DeviceArithmetic:
    """The committed device's B-side arithmetic: its noise fraction (the
    per-row scales), its FP8 atom's window, and whether it rewrites the strip."""

    delta: float
    window_bits: int
    rewrites_strip: bool


ARITHMETIC = {
    Device.BLACKWELL: DeviceArithmetic(delta=0.5, window_bits=26, rewrites_strip=True),
    Device.HOPPER: DeviceArithmetic(delta=1.0, window_bits=14, rewrites_strip=False),
}

# The FP8 atom: e4m3 values are integers in units of 2^-9, so products are in
# 2^-18; a product term is sig * 2^(exp - 23) with sig = (sa * sb) << 17.
_ATOM_K = 32
_GFLOAT_BITS = 24
_PRODUCT_SHIFT = 17
_E4M3_FRACTION_BITS = 3
_BLACKWELL_WINDOW_BITS = 26
# A product's lowest bit sits 2 * _E4M3_FRACTION_BITS below its exponent and
# the Blackwell atom keeps 25 fractional bits below the largest, so a cell
# whose product exponents span at most this truncates nothing there: the atom
# is the exact sum, which then also fits fp64.
_UNTRUNCATED_EXP_SPAN = _BLACKWELL_WINDOW_BITS - 1 - 2 * _E4M3_FRACTION_BITS
_ZERO_EXP = -(1 << 24)  # stored exponent of a zero summand: never raises the anchor
_FP32_MIN_NORMAL_EXP = -126

# The verifier's l2 frame (zk-pow prequant::l2_frame_width): Wl2 = 27 - ceil(log2 k).
_L2_FRAME_BITS = 27
_BF16_FRACTION_BITS = 7
# |bf16| = significand * 2^(max(E, 1) - _BF16_UNIT_EXP) for exponent field E.
_BF16_UNIT_EXP = 127 + _BF16_FRACTION_BITS
_BF16_INF_BITS = 0x7F80


@dataclass(frozen=True)
class CpuBContext:
    """One job's prepared B side in host memory.

    Tensor fields carry the ``LayerBuffers`` names, dtypes and shapes; the
    32-byte words (``key_a``, ``seed_b``, ``threshold``) are what
    ``key_a_dev``, ``seed_b_dev`` and ``threshold_dev`` must hold. B is keyed
    by ``job``'s parent.
    """

    job: MiningJob
    config: MiningConfiguration
    key_a: bytes
    key_b: bytes
    seed_b: bytes
    p_b: bytes
    threshold: bytes
    b_proof: BProofContext
    f1: torch.Tensor  # (R, k) e4m3 F_A
    f1_hl: torch.Tensor  # (k, PACKED_NOISE_K) int8
    f2: torch.Tensor  # (R, k) e4m3 F_B
    e2: torch.Tensor  # (n, R) e4m3 E_B
    alpha_b: torch.Tensor  # (n,) bf16
    beta_b: torch.Tensor  # (n,) bf16
    inv_alpha_b: torch.Tensor  # (n,) fp32
    b_prime: torch.Tensor  # (n, k) e4m3
    b_peel: torch.Tensor  # (n, 2R) bf16


@dataclass(frozen=True)
class RowScales:
    """A layer's per-row ``alpha_b``/``beta_b`` and floored ``l2``. They depend
    on its planes only, so one preparation's serve every later job of that layer."""

    alpha_b: torch.Tensor  # (n,) bf16
    beta_b: torch.Tensor  # (n,) bf16
    l2_b: torch.Tensor  # (n,) bf16


def prepare_b_cpu(
    codes: torch.Tensor,
    scales: torch.Tensor,
    job: MiningJob,
    *,
    device: Device,
    experts: int = 0,
    cancelled: Callable[[], bool] | None = None,
    row_scales: RowScales | None = None,
    reference: bool = False,
) -> tuple[CpuBContext, RowScales] | None:
    """Prepare ``job``'s B side (keyed by its parent) for the layer committed
    as ``codes``/``scales``, with the row scales it used.

    ``experts`` is the layer's ``LayerState.experts``: 0 for a dense layer,
    else the experts whose rows the planes stack. ``device`` is the committed
    mining device of the GPU that will run the launches, not where this
    function runs. ``cancelled`` is polled between stages and the torch
    chain's row chunks; returns None once it reports True. ``row_scales``,
    from an earlier preparation of these planes, skips the per-row norms.
    ``reference`` runs the torch chain instead of the native kernels
    (``b_prepare_native``).
    """
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

    f1 = _lines(noise_key, LABEL_F1, k, reference).t().contiguous()
    f2 = _lines(noise_key, LABEL_F2, k, reference).t().contiguous()
    f1_hl = torch.zeros(k, PACKED_NOISE_K, dtype=torch.int8)
    f1_hl[:, :RANK] = f1.view(torch.int8).t()
    e2 = _lines(noise_key, LABEL_E2, n, reference)

    if row_scales is None:
        row_scales = _layer_row_scales(codes, scales, arithmetic.delta, cancelled)
    if row_scales is None or (cancelled is not None and cancelled()):
        return None
    if reference:
        rows = _reference_rows(codes, scales, row_scales, e2, f1, f2, arithmetic, cancelled)
    else:
        rows = _native_rows(codes, scales, row_scales, e2, f1, f2, arithmetic)
    if rows is None:
        return None
    b_prime, b_peel = rows
    alpha_b, beta_b = row_scales.alpha_b, row_scales.beta_b

    prepared = CpuBContext(
        job=job,
        config=config,
        key_a=key_a,
        key_b=key_b,
        seed_b=seed_b,
        p_b=p_b,
        # The lottery lattice is expert-local (``LayerState.lottery_n``).
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
    """``prepared``'s fields but ``b_proof``, which holds the host planes and a
    lock, so they can cross a process boundary."""
    return {
        field.name: getattr(prepared, field.name)
        for field in fields(CpuBContext)
        if field.name != "b_proof"
    }


def with_b_proof(
    detached: dict[str, object], codes: torch.Tensor, scales: torch.Tensor
) -> CpuBContext:
    """:func:`without_b_proof` undone over the planes it was prepared from."""
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
    """``commit_planes_for_leaf(planes, key, leaf).digest``.

    At BLAKE3's own chunk length the keyed Merkle root of the zero-padded
    plane is BLAKE3's keyed hash of it, which hashes on many threads.
    """
    if leaf != _BLAKE3_CHUNK_LEN:
        return commit_planes_for_leaf(planes, key, leaf).digest
    roots = []
    for plane in planes:
        data = plane.contiguous().view(torch.uint8).flatten().numpy()
        hasher = blake3(key=key, max_threads=min(torch.get_num_threads(), _DIGEST_MAX_THREADS))
        hasher.update(data)
        hasher.update(bytes(-data.size % leaf))
        roots.append(hasher.digest())
    return blake3(b"".join(roots), key=key).digest()


def _lines(key: bytes, label: bytes, count: int, reference: bool) -> torch.Tensor:
    if reference:
        return _noise_lines(key, label, range(count))
    return native().noise_lines(key, label, count, NOISE_TARGET_NORM, INT_SQRT_PREC)


def _layer_row_scales(
    codes: torch.Tensor,
    scales: torch.Tensor,
    delta: float,
    cancelled: Callable[[], bool] | None,
) -> RowScales | None:
    n, k = codes.shape
    alpha_b, beta_b, l2_b = (torch.empty(n, dtype=_BF16) for _ in range(3))
    for rows in _row_chunks(n, k):
        if cancelled is not None and cancelled():
            return None
        alpha_b[rows], beta_b[rows], l2_b[rows] = _row_scales(
            PrequantMatrix(codes[rows], scales[rows]), delta
        )
    return RowScales(alpha_b=alpha_b, beta_b=beta_b, l2_b=l2_b)


def _row_chunks(n: int, k: int) -> list[slice]:
    chunk_rows = max(1, PREPARE_CHUNK_CELLS // k)
    return [slice(start, min(start + chunk_rows, n)) for start in range(0, n, chunk_rows)]


def _native_rows(
    codes: torch.Tensor,
    scales: torch.Tensor,
    row_scales: RowScales,
    e2: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    arithmetic: DeviceArithmetic,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B', peel)`` from the fused native rows."""
    b_prime, mid = native().prepare_rows(
        codes,
        scales,
        row_scales.alpha_b,
        row_scales.beta_b,
        row_scales.l2_b,
        e2,
        f1,
        f2,
        arithmetic.window_bits,
        arithmetic.rewrites_strip,
    )
    peel = torch.cat([mid, -(row_scales.beta_b.unsqueeze(1) * e2.to(_BF16))], dim=1)
    return b_prime, peel


def _reference_rows(
    codes: torch.Tensor,
    scales: torch.Tensor,
    row_scales: RowScales,
    e2: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    arithmetic: DeviceArithmetic,
    cancelled: Callable[[], bool] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    n, k = codes.shape
    f2_dot = NoiseDotFactor.of(f2.t())
    b_prime = torch.empty(n, k, dtype=_E4M3)
    b_peel = torch.empty(n, 2 * RANK, dtype=_BF16)
    for rows in _row_chunks(n, k):
        if cancelled is not None and cancelled():
            return None
        b_prime[rows], b_peel[rows] = _prepare_rows(
            codes[rows],
            scales[rows],
            row_scales.alpha_b[rows].unsqueeze(1),
            row_scales.beta_b[rows].unsqueeze(1),
            row_scales.l2_b[rows].unsqueeze(1),
            e2[rows],
            f1,
            f2,
            f2_dot,
            arithmetic,
        )
    return b_prime, b_peel


def _row_scales(
    planes: PrequantMatrix, delta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(rows,)`` bf16 ``(alpha, beta, floored l2)`` of one chunk of B rows
    under the committed device's noise fraction ``delta``."""
    norms = exact_row_norms(planes)
    floor = torch.tensor(_NORM_FLOOR, dtype=_BF16)
    l2, linf = torch.maximum(norms.l2, floor), torch.maximum(norms.linf, floor)
    delta_r = torch.tensor(delta * math.sqrt(RANK), dtype=_BF16)
    delta_over_std = torch.tensor(delta * math.sqrt(RANK) / NOISE_TARGET_NORM**2, dtype=_BF16)
    # alpha bounds |alpha*B + beta*E@F| by QUANT_MAX at the row's peaks; beta
    # sets the noise-to-signal L2 ratio to delta.
    alpha = torch.tensor(QUANT_MAX, dtype=_BF16) / _fma_bf16(delta_r, l2, linf)
    beta = (alpha * l2) * delta_over_std
    return alpha.squeeze(1), beta.squeeze(1), l2.squeeze(1)


def _prepare_rows(
    codes: torch.Tensor,
    scales: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    l2: torch.Tensor,
    e: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    f2_dot: "NoiseDotFactor",
    arithmetic: DeviceArithmetic,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B', peel)`` for one chunk of B rows, under their ``(rows, 1)`` bf16
    ``alpha``/``beta``/floored ``l2`` and ``(rows, R)`` E_B lines ``e``."""
    noise = noise_dot(e, f2_dot, arithmetic.window_bits)
    alpha32 = alpha.float()
    coefficient = (alpha32 * scales.float()).repeat_interleave(codes.shape[1] // scales.shape[1], 1)
    mixed = _fma_f32(beta.float(), noise, coefficient * codes.float())
    if arithmetic.rewrites_strip:
        b_prime = rewrite_strip(mixed, alpha32 * l2.float())
    else:
        b_prime = _e4m3(mixed)

    ebfb = (e.float() @ f2.float()).to(_BF16)
    diff = _fma_bf16(beta, ebfb, -b_prime.to(_BF16))
    mid = (diff.float() @ f1.float().t()).to(_BF16)
    peel = torch.cat([mid, -(beta * e.to(_BF16))], dim=1)
    return b_prime, peel


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    """The verifier's e4m3 cast: round to nearest even, saturating to +-448."""
    return x.clamp(-QUANT_MAX, QUANT_MAX).to(_E4M3)


def rewrite_strip(mixed: torch.Tensor, scaled_rms: torch.Tensor) -> torch.Tensor:
    """e4m3 codes of the ``(rows, k)`` fp32 ``mixed`` under the Blackwell strip
    rewrite, ``scaled_rms`` the ``(rows, 1)`` fp32 ``alpha * l2``.

    A code whose magnitude byte is below that of ``e4m3(g * scaled_rms)``
    (``g`` per row) takes the row's safe exponent ``floor(log2(scaled_rms)) -
    STRIP_EXP_OFFSET`` with its mantissa LSB set, when the coin of its aligned
    ``STRIP_COIN_BLOCK`` columns passes: bits 14..8 of the group's first fp32
    mix below ``STRIP_COIN_THRESHOLD``.
    """
    code = _e4m3(mixed).view(torch.uint8).int()
    g = torch.where(scaled_rms < STRIP_SMALL_RMS, STRIP_G_SMALL_RMS, STRIP_G).float()
    bound = _e4m3(g * scaled_rms).view(torch.uint8).int() & _E4M3_MAGNITUDE_MASK
    field = ((scaled_rms.view(torch.int32) >> 23) & 0xFF) - (127 + STRIP_EXP_OFFSET - 7)
    safe = ((field & 0xF) << 3) | 1
    coin_field = (mixed.view(torch.int32) >> STRIP_COIN_SHIFT) & 0x7F
    coin = coin_field[:, ::STRIP_COIN_BLOCK] < STRIP_COIN_THRESHOLD
    coin = coin.repeat_interleave(STRIP_COIN_BLOCK, dim=1)[:, : mixed.shape[1]]
    inside = coin & ((code & _E4M3_MAGNITUDE_MASK) < bound)
    return torch.where(inside, (code & 0x87) | safe, code).to(torch.uint8).view(_E4M3)


def exact_row_norms(planes: PrequantMatrix) -> RowNorms:
    """The verifier's row norms (zk-pow ``prequant::exact_norms``).

    ``l2`` is the grid-rounded ``RNE_bf16(sqrt(S * 2^(E_MAX - 268 - Wl2) / k))``
    of the integer frame sum ``S = sum_b floor(M_b^2 * sum(int^2) * 2^Wl2 /
    2^(E_MAX - 2 E_b))`` over the row's blocks (scale ``M_b * 2^(E_b - 134)``,
    ``E_MAX`` the largest ``2 E_b`` of a nonzero block): one rounding in the
    square root. ``linf`` is exact.
    """
    n, k = planes.int_values.shape
    ceil_log2_k = (k - 1).bit_length()
    if ceil_log2_k > _L2_FRAME_BITS:
        raise ValueError(f"k={k} exceeds the verifier's L2 frame width")
    frame_bits = _L2_FRAME_BITS - ceil_log2_k
    ints = planes.int_values.to(torch.int64).reshape(n, -1, planes.block_size)
    codes = planes.scales.view(torch.int16).to(torch.int64) & 0x7FFF
    exponent = codes >> _BF16_FRACTION_BITS
    significand = (codes & ((1 << _BF16_FRACTION_BITS) - 1)) | torch.where(
        exponent > 0, 1 << _BF16_FRACTION_BITS, 0
    )
    doubled_exp = 2 * exponent.clamp(min=1)
    block_terms = significand * significand * (ints * ints).sum(dim=-1)
    e_max = torch.where(block_terms != 0, doubled_exp, 0).amax(dim=1, keepdim=True)
    # Shifted terms stay below 2^50, so a 63-bit shift already floors them to 0.
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
    return RowNorms(l2=l2, linf=planes.exact_norms().linf)


def _bf16_value(bits: int) -> Fraction:
    exponent = bits >> _BF16_FRACTION_BITS
    significand = bits & ((1 << _BF16_FRACTION_BITS) - 1)
    if exponent:
        significand |= 1 << _BF16_FRACTION_BITS
    return significand * Fraction(2) ** (max(exponent, 1) - _BF16_UNIT_EXP)


def _rne_sqrt_bf16(square: Fraction) -> int:
    """Bits of ``RNE_bf16(sqrt(square))`` for a positive finite ``square``,
    from a float seed refined by exact midpoint comparisons (ties to even)."""

    def past_midpoint(low: int) -> bool:
        midpoint = (_bf16_value(low) + _bf16_value(low + 1)) / 2
        boundary = midpoint * midpoint
        # A tie belongs to the even neighbour.
        return square > boundary or (square == boundary and low % 2 == 1)

    seed = torch.tensor(math.sqrt(float(square)), dtype=torch.float32).to(_BF16)
    bits = int(seed.view(torch.int16))
    while past_midpoint(bits):
        bits += 1
    while bits and not past_midpoint(bits - 1):
        bits -= 1
    return bits


def _noise_lines(key: bytes, label: bytes, indices: range) -> torch.Tensor:
    """``(len(indices), RANK)`` e4m3 noise lines under ``key`` at ``label``.

    Each line is ``RANK`` keyed-BLAKE3 bytes of the block ``label || u32
    LE(index)``, decoded to sign x magnitude in ``[1, 128]`` and scaled to L2
    norm ``NOISE_TARGET_NORM`` by an exact integer isqrt and one BF16
    division, then rounded to e4m3 -- the verifier's ``sample_line``.
    """
    raw = b"".join(
        blake3((label + encode_u32_le(i)).ljust(_NOISE_MESSAGE_BYTES, b"\x00"), key=key).digest(
            length=RANK
        )
        for i in indices
    )
    draws = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, RANK).to(torch.int64)
    lines = (1 - 2 * (draws >> 7)) * ((draws & 0x7F) + 1)
    norm_scaled = [
        math.isqrt(sumsq * INT_SQRT_PREC * INT_SQRT_PREC)
        for sumsq in (lines * lines).sum(dim=1).tolist()
    ]
    scale = torch.tensor(NOISE_TARGET_NORM * INT_SQRT_PREC, dtype=_BF16) / torch.tensor(
        norm_scaled, dtype=_BF16
    )
    return (lines.to(_BF16) * scale.unsqueeze(1)).to(_E4M3)


def _round_to_odd(total: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """fp64 ``total`` with exact ``residual`` (``x - total``) rounded to odd at
    fp32; a following rounding then equals the single rounding of ``x``."""
    total32 = total.to(torch.float32)
    error = (total - total32.to(torch.float64)) + residual
    to_odd = (error != 0) & ((total32.view(torch.int32) & 1) == 0) & total32.isfinite()
    toward = torch.where(error > 0, torch.inf, -torch.inf).to(torch.float32)
    return torch.where(to_odd, torch.nextafter(total32, toward), total32)


def _two_sum_residual(product: torch.Tensor, c: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    rounded_part = total - c
    return (product - rounded_part) + (c - (total - rounded_part))


def _fma_f32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``a * b + c`` on fp32 operands with ONE round-to-nearest-even (the
    verifier's ``fp32_fma``): the product is exact in fp64, and the fp64 sum
    is rounded to odd before its fp32 rounding."""
    a64, b64, c64 = a.to(torch.float64), b.to(torch.float64), c.to(torch.float64)
    product = a64 * b64
    total = product + c64
    residual = _two_sum_residual(product, c64, total)
    inexact_even = (residual != 0) & ((total.view(torch.int64) & 1) == 0) & total.isfinite()
    toward = torch.where(residual > 0, torch.inf, -torch.inf).to(torch.float64)
    return torch.where(inexact_even, torch.nextafter(total, toward), total).to(torch.float32)


def _fma_bf16(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``a * b + c`` on BF16 operands with ONE round-to-nearest-even to BF16."""
    a64, b64, c64 = a.to(torch.float64), b.to(torch.float64), c.to(torch.float64)
    product = a64 * b64
    total = product + c64
    return _round_to_odd(total, _two_sum_residual(product, c64, total)).to(_BF16)


def _decode_e4m3(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 -> (signed int value in units of 2^-9, stored exponent).

    Subnormals keep their 3-bit significand with stored exponent -6; zeros
    carry ``_ZERO_EXP``. Callers pass finite codes only.
    """
    codes = values.view(torch.uint8).to(torch.int32)
    exponent_bits = (codes >> 3) & 0xF
    mantissa = codes & 0x7
    significand = torch.where(exponent_bits != 0, mantissa | 0x8, mantissa)
    exponent_bits = exponent_bits.clamp(min=1)
    magnitude = significand << (exponent_bits - 1)
    signed = torch.where((codes >> 7) != 0, -magnitude, magnitude)
    stored = torch.where(magnitude == 0, _ZERO_EXP, exponent_bits - 7)
    return signed, stored


def _e4m3_terms(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 -> (signed significand, stored exponent ``max(E, 1) - 7``):
    ``value = significand * 2^(stored - 3)``."""
    codes = values.view(torch.uint8).to(torch.int32)
    exponent_bits = (codes >> 3) & 0xF
    mantissa = codes & 0x7
    significand = torch.where(exponent_bits != 0, mantissa | 0x8, mantissa)
    signed = torch.where((codes >> 7) != 0, -significand, significand)
    return signed, exponent_bits.clamp(min=1) - 7


def _exponent_spans(lines: torch.Tensor) -> torch.Tensor:
    """Per line of e4m3 values, the stored-exponent span (huge when it holds a zero)."""
    _, exponents = _decode_e4m3(lines)
    return exponents.amax(dim=1) - exponents.amin(dim=1)


@dataclass(frozen=True)
class NoiseDotFactor:
    """The ``(cols, R)`` right-hand factor of the noise dot, decoded once per layer."""

    e4m3: torch.Tensor  # (cols, R)
    wide_t: torch.Tensor  # (R, cols) float64
    exp_span: int  # the widest span over its lines

    @classmethod
    def of(cls, f_t: torch.Tensor) -> "NoiseDotFactor":
        if f_t.shape[1] > _ATOM_K:
            raise ValueError(f"the noise dot is a single K={_ATOM_K} atom, got R={f_t.shape[1]}")
        return cls(
            e4m3=f_t,
            wide_t=f_t.t().to(torch.float64),
            exp_span=int(_exponent_spans(f_t).max()),
        )


def noise_dot(e: torch.Tensor, f: NoiseDotFactor, window_bits: int) -> torch.Tensor:
    """``e @ f.T`` in fp32 as the committed device's FP8 atom computes it.

    On Blackwell an untruncated cell is its exact sum, which a float64 GEMM
    computes exactly in any order and rounding it towards zero to fp32 is the
    atom's result; every other cell runs :func:`_windowed_noise_dot`.
    """
    if e.shape[1] != f.e4m3.shape[1]:
        raise ValueError(f"noise dot ranks differ: {e.shape[1]} and {f.e4m3.shape[1]}")
    if window_bits != _BLACKWELL_WINDOW_BITS:
        return _windowed_noise_dot(e, f.e4m3, window_bits)
    out = _toward_zero_fp32(e.to(torch.float64) @ f.wide_t)
    windowed = _exponent_spans(e) + f.exp_span > _UNTRUNCATED_EXP_SPAN
    if windowed.any():
        out[windowed] = _windowed_noise_dot(e[windowed], f.e4m3, window_bits)
    return out


def _toward_zero_fp32(value: torch.Tensor) -> torch.Tensor:
    """float64 -> fp32 rounded towards zero."""
    nearest = value.to(torch.float32)
    away = nearest.to(torch.float64).abs() > value.abs()
    return torch.where(away, torch.nextafter(nearest, torch.zeros_like(nearest)), nearest)


def _windowed_noise_dot(e: torch.Tensor, f_t: torch.Tensor, window_bits: int) -> torch.Tensor:
    """``e @ f_t.T`` for ``(rows, R)`` and ``(cols, R)`` e4m3 factors, in fp32:
    the verifier's ``windowed_group_sum`` of one K = 32 atom from +0.

    Each nonzero product ``sig * 2^(exp - 23)`` (``sig = (sa * sb) << 17``) is
    truncated to the ``window_bits``-bit window anchored at the largest
    exponent, the terms are summed exactly, the sum is truncated to
    ``window_bits`` significand bits (and to fp32's subnormal grid), and a
    zero result is +0. ``R`` up to the atom's K zero-pads.
    """
    if e.shape[1] > _ATOM_K:
        raise ValueError(f"the noise dot is a single K={_ATOM_K} atom, got R={e.shape[1]}")
    e_sig, e_exp = _e4m3_terms(e)
    f_sig, f_exp = _e4m3_terms(f_t)
    products = e_sig[:, None, :].to(torch.int64) * f_sig[None, :, :]
    # Stored exponents are q - 7, so the product's exponent is their sum.
    exponent = torch.where(products == 0, _ZERO_EXP, e_exp[:, None, :] + f_exp[None, :, :])
    anchor = exponent.amax(dim=-1)
    diff = abs(window_bits - _GFLOAT_BITS)
    magnitude = products.abs() << _PRODUCT_SHIFT
    magnitude = magnitude >> diff if window_bits < _GFLOAT_BITS else magnitude << diff
    shift = (anchor.unsqueeze(-1) - exponent).clamp(min=0)
    aligned = torch.where(shift < 32, magnitude >> shift.clamp(max=63), 0)
    total = torch.where(products < 0, -aligned, aligned).sum(dim=-1)

    total_magnitude = total.abs()
    # bit_length via an exact float64 frexp (magnitudes < 2^37).
    bits = torch.frexp(total_magnitude.to(torch.float64)).exponent.to(torch.int64)
    result_exp = anchor + bits - window_bits
    significand = torch.where(
        bits > window_bits,
        total_magnitude >> (bits - window_bits).clamp(min=0),
        total_magnitude << (window_bits - bits).clamp(min=0),
    )
    sub = (_FP32_MIN_NORMAL_EXP - result_exp).clamp(min=0)
    significand = torch.where(sub >= 32, 0, significand >> sub.clamp(max=63))
    result_exp = torch.maximum(result_exp, torch.tensor(_FP32_MIN_NORMAL_EXP))
    significand = significand << diff if window_bits < _GFLOAT_BITS else significand >> diff
    value = torch.ldexp(significand.to(torch.float64), (result_exp - 23).to(torch.float64))
    value = torch.where(total < 0, -value, value).to(torch.float32)
    # Every zero (all products zero, or a sum truncated away) is +0.
    return torch.where(significand == 0, torch.zeros_like(value), value)
