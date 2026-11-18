"""Name every jackpot check on an opened tile.

The cert-v4 binding reports every policy rejection as one sentence. This module
replays the checks of ``zk-pow`` ``JackpotPolicy``
(``zk-pow/src/v4/api/jackpot_policy.rs``) that depend on the opened prequant
tile and records pass or fail for each, with the quantities behind the verdict.
The unpredictable-summand census requires the noised ``A' @ B'.T`` replay, which
an opened prequant tile does not carry, so that check logs its budget as
unevaluated.
"""

from __future__ import annotations

import bisect
import math
import struct
from dataclasses import dataclass

import torch

from .params import Device
from .prequant import PrequantMatrix

_TAU_IDLE = 8.0
_EPS_IDLE = 1.0 / 64.0
# Per-side signal-RMS floors from ``JackpotPolicy::for_device``: ``sigma >= delta * rms_min``.
_MIN_RMS = {"A": 4.0, "B": 8.0}
_MAX_E4M3 = 448.0
_NORM_FLOOR = 2.0**-32
_DELTA = {Device.HOPPER: 1.0, Device.BLACKWELL: 0.5}
_SIGMA_ENC_OFFSET = {Device.HOPPER: 14, Device.BLACKWELL: 13}
# ``(g at signal RMS >= 8, g below it)`` per committed device.
_STRIP_G = {
    Device.HOPPER: (0.105, 0.105),
    Device.BLACKWELL: (0.08, 0.1),
}
_RMS_THRESHOLD = 4.0
_SMALL_RMS = 8.0
_COVERAGE_SCALE = 1 << 16
_NOISE_COVERAGE_MIN_INVERSE = 25
_GAUSSIAN_SCALE = 0.65625
_CAP_OFFSET = 0.0431
_CAP_SLOPE = 0.2738
_CAP_PIVOT = 0.4
_SMALL_RMS_DIVISOR = 8
_FAR_GAP = 16
_PREDICT_BUDGET_DIVISOR = 20

# Exact integer thresholds ``ceil(2^(15 + i/64))`` for ``log2_fixed``.
_LOG_BOUNDS = (
    32768,
    33125,
    33486,
    33851,
    34219,
    34592,
    34969,
    35349,
    35734,
    36123,
    36517,
    36914,
    37316,
    37723,
    38133,
    38549,
    38968,
    39393,
    39822,
    40255,
    40694,
    41137,
    41585,
    42038,
    42495,
    42958,
    43426,
    43899,
    44377,
    44860,
    45348,
    45842,
    46341,
    46846,
    47356,
    47872,
    48393,
    48920,
    49453,
    49991,
    50536,
    51086,
    51642,
    52205,
    52773,
    53348,
    53929,
    54516,
    55109,
    55710,
    56316,
    56929,
    57549,
    58176,
    58810,
    59450,
    60097,
    60752,
    61413,
    62082,
    62758,
    63441,
    64132,
    64831,
)


@dataclass(frozen=True)
class _Dequantized:
    values: torch.Tensor  # opened bf16 rows
    l2: torch.Tensor  # floored bf16, one per row
    linf: torch.Tensor


@dataclass(frozen=True)
class Check:
    """One policy check, with the numbers a later histogram can parse."""

    name: str
    status: str  # "pass", "fail", or "unevaluated"
    quantities: tuple[tuple[str, str], ...] = ()

    def line(self) -> str:
        extra = " ".join(f"{key}={value}" for key, value in self.quantities)
        return f"{self.name}: {self.status}" + (f" {extra}" if extra else "")


def format_report(
    codes_a: torch.Tensor,
    scales_a: torch.Tensor,
    codes_b: torch.Tensor,
    scales_b: torch.Tensor,
    *,
    device: Device,
    rank: int,
) -> str:
    """Every check, verifier order, plus the first one the verifier would stop on."""
    checks = evaluate_checks(codes_a, scales_a, codes_b, scales_b, device, rank)
    lines = [f"jackpot policy rejected ({_summary(checks)})"]
    lines.extend(f"  {check.line()}" for check in checks)
    return "\n".join(lines)


def rejection_reason(
    codes_a: torch.Tensor,
    scales_a: torch.Tensor,
    codes_b: torch.Tensor,
    scales_b: torch.Tensor,
    device: Device,
    rank: int,
) -> str:
    """The earliest check the verifier would stop on.

    An unevaluated census counts: the verifier runs it before noise coverage,
    and this tile does not carry the replay that census needs.
    """
    return _summary(evaluate_checks(codes_a, scales_a, codes_b, scales_b, device, rank))


def evaluate_checks(
    codes_a: torch.Tensor,
    scales_a: torch.Tensor,
    codes_b: torch.Tensor,
    scales_b: torch.Tensor,
    device: Device,
    rank: int,
) -> tuple[Check, ...]:
    """Pass/fail for every check, including ones after the first failure."""
    delta = _DELTA[device]
    sides = (
        ("A", _dequant(codes_a, scales_a)),
        ("B", _dequant(codes_b, scales_b)),
    )
    prepared = tuple((name, tile, *_row_stats(tile, delta, rank)) for name, tile in sides)
    checks = [_liveness(name, tile, delta) for name, tile, _, _ in prepared]
    checks.extend(
        _noise_floor(name, sigmas, delta, _MIN_RMS[name]) for name, _, _, sigmas in prepared
    )
    height = codes_a.shape[0]
    width = codes_b.shape[0]
    k = codes_a.shape[1]
    checks.append(_census(k, height, width))
    checks.extend(_noise_coverage(name, tile, alphas, device) for name, tile, alphas, _ in prepared)
    return tuple(checks)


def _summary(checks: tuple[Check, ...] | list[Check]) -> str:
    for check in checks:
        if check.status in ("fail", "unevaluated"):
            return check.name
    return "passed"


def _dequant(codes: torch.Tensor, scales: torch.Tensor) -> _Dequantized:
    """Open a prequant operand: int8 codes and bf16 scales to bf16 rows."""
    operand = PrequantMatrix(codes, scales)
    norms = operand.exact_norms()
    floor = torch.tensor(_NORM_FLOOR, dtype=torch.bfloat16)
    return _Dequantized(
        operand.open(),
        torch.maximum(norms.l2, floor).reshape(-1),
        torch.maximum(norms.linf, floor).reshape(-1),
    )


def _row_stats(tile: _Dequantized, delta: float, rank: int) -> tuple[list[float], list[float]]:
    l2 = tile.l2.double().tolist()
    linf = tile.linf.double().tolist()
    alphas = [
        _row_alpha(row_l2, row_linf, delta, rank) for row_l2, row_linf in zip(l2, linf, strict=True)
    ]
    sigmas = [delta * alpha * row_l2 for alpha, row_l2 in zip(alphas, l2, strict=True)]
    return alphas, sigmas


def _liveness(name: str, tile: _Dequantized, delta: float) -> Check:
    """Dead entries are ``|X| >= tau * delta * l2`` over the whole side."""
    values = tile.values.double()
    bound = (_TAU_IDLE * delta) * tile.l2.double()
    dead = int((values.abs() >= bound[:, None]).sum().item())
    limit = _EPS_IDLE * values.numel()
    status = "pass" if dead <= limit else "fail"
    return Check(
        f"liveness ({name})",
        status,
        (("dead", str(dead)), ("limit", _num(limit))),
    )


def _noise_floor(name: str, sigmas: list[float], delta: float, min_rms: float) -> Check:
    """``sigma = delta * alpha * l2`` must reach ``delta * min_rms`` on every row."""
    floor = delta * min_rms
    failing = [sigma for sigma in sigmas if sigma < floor]
    status = "fail" if failing else "pass"
    minimum = min(sigmas) if sigmas else 0.0
    return Check(
        f"noise floor ({name})",
        status,
        (
            ("min_sigma", _num(minimum)),
            ("floor", _num(floor)),
            ("failing_rows", str(len(failing))),
        ),
    )


def _census(k: int, height: int, width: int) -> Check:
    """The two census quantities. ``skippable`` needs the A'/B' replay."""
    budget = (k * height * width) // _PREDICT_BUDGET_DIVISOR
    return Check(
        "unpredictable summands",
        "unevaluated",
        (("skippable", "unknown"), ("budget", str(budget))),
    )


def _row_kernel_sum(
    row_codes: list[int], alpha_code: int, l2_code: int, cap: int, device: Device
) -> int:
    lambda_zero = _lambda(device, alpha_code, 0, l2_code)
    counts: dict[int, int] = {}
    for code in row_codes:
        magnitude_code = int(code) & 0x7FFF
        counts[magnitude_code] = counts.get(magnitude_code, 0) + 1
    total = 0
    for code, count in counts.items():
        diff = _lambda(device, alpha_code, code, l2_code) - lambda_zero
        total += count * min(_gauss_kernel(diff), cap)
    return total


def _noise_coverage(name: str, tile: _Dequantized, alphas: list[float], device: Device) -> Check:
    """Check 4: each row's Q16 noise-coverage product is at least ``ceil(k*2^32/25)``."""
    clean_bits = tile.values.view(torch.uint16)
    l2_codes = [int(v) for v in tile.l2.view(torch.uint16).tolist()]
    threshold = _minimum_coverage_product(clean_bits.shape[1])
    failing = 0
    min_product: int | None = None
    bad_rms: float | None = None
    for row_idx, (alpha_val, l2_code) in enumerate(zip(alphas, l2_codes, strict=True)):
        rms = alpha_val * float(tile.l2[row_idx].float().item())
        try:
            coefficient, cap = _gauss_row_constants(rms, device)
        except ValueError:
            failing += 1
            bad_rms = rms
            continue
        kernel_sum = _row_kernel_sum(
            clean_bits[row_idx].tolist(), _bf16_code(alpha_val), l2_code, cap, device
        )
        product = coefficient * kernel_sum
        if min_product is None or product < min_product:
            min_product = product
        if product < threshold:
            failing += 1
    quantities: list[tuple[str, str]] = [
        ("threshold", str(threshold)),
        ("failing_rows", str(failing)),
    ]
    if min_product is not None:
        quantities.insert(0, ("min_product", str(min_product)))
    if bad_rms is not None:
        quantities.append(("signal_rms", _num(bad_rms)))
    return Check(
        f"noise coverage ({name})",
        "fail" if failing else "pass",
        tuple(quantities),
    )


def _minimum_coverage_product(k: int) -> int:
    """``ceil(k * 2^32 / 25)``."""
    return (k * _COVERAGE_SCALE**2 + _NOISE_COVERAGE_MIN_INVERSE - 1) // _NOISE_COVERAGE_MIN_INVERSE


def _gauss_kernel(difference: int) -> int:
    """``floor(2^16 * exp(-(2^(d/64) - 1) / 2))``, matching ``zk-pow`` ``gauss_kernel``."""
    if difference >= 291:
        return 0
    w = math.exp2(difference / 64.0) - 1.0
    return math.floor(_COVERAGE_SCALE * math.exp(-0.5 * w))


def _gauss_row_constants(signal_rms: float, device: Device) -> tuple[int, int]:
    """Q16 ``(coefficient, kernel cap)`` matching ``zk-pow`` ``gauss_row_constants``."""
    if not 4.0 <= signal_rms < 256.0:
        raise ValueError(f"signal RMS {signal_rms} is outside [4, 256)")
    rms = torch.tensor(signal_rms, dtype=torch.float32)
    g, g_small = _STRIP_G[device]
    factor = g if float(rms) >= _SMALL_RMS else g_small
    scaled = torch.tensor(factor, dtype=torch.float32) * rms
    code = float(scaled.to(torch.float8_e4m3fn).float().item())
    scale = float(_COVERAGE_SCALE)
    noise_rms = _DELTA[device] * float(rms.item())
    if noise_rms >= _RMS_THRESHOLD:
        return math.floor(scale * _GAUSSIAN_SCALE * code / noise_rms), _COVERAGE_SCALE
    cap = math.floor(scale * (_CAP_OFFSET + _CAP_SLOPE * (code - _CAP_PIVOT)))
    return _COVERAGE_SCALE // _SMALL_RMS_DIVISOR, _SMALL_RMS_DIVISOR * cap


def _bf16_e_star_m(code: int) -> tuple[int, int]:
    exp = (code >> 7) & 0xFF
    mantissa = code & 0x7F
    if exp == 0:
        return 1, mantissa
    return exp, 128 + mantissa


def _log2_fixed(n: int) -> int:
    if n == 0:
        return 0
    w = n.bit_length() - 1
    normalized = n << (15 - w)
    fraction = bisect.bisect_right(_LOG_BOUNDS, normalized) - 1
    return 64 * w + fraction


def _lambda(device: Device, alpha: int, x: int, l2f: int) -> int:
    """Exact integer summand score matching ``zk-pow``'s ``lambda``."""
    alpha_e, alpha_m = _bf16_e_star_m(alpha)
    x_e, x_m = _bf16_e_star_m(x)
    if x_m == 0:
        x_norm, x_enc = 0, 0
    else:
        product = alpha_m * x_m
        width = product.bit_length()
        x_norm = product << (16 - width)
        x_enc = alpha_e + x_e - 1 + width
    l2f_e, l2f_m = _bf16_e_star_m(l2f)
    sigma_product = alpha_m * l2f_m
    sigma_wide = int(sigma_product >= 1 << 15)
    sigma_norm = sigma_product << (1 - sigma_wide)
    sigma_enc = alpha_e + l2f_e + sigma_wide + _SIGMA_ENC_OFFSET[device]

    x_dominates = x_enc >= sigma_enc and x_norm != 0
    exponent_gap = abs(x_enc - sigma_enc)
    gap_is_far = exponent_gap >= _FAR_GAP
    near_gap = 0 if gap_is_far else exponent_gap
    norm_big, norm_small = (x_norm, sigma_norm) if x_dominates else (sigma_norm, x_norm)
    quotient = ((norm_small * norm_small) >> near_gap) >> near_gap
    sum_of_squares = norm_big * norm_big + (0 if gap_is_far else quotient)
    top = sum_of_squares >> 17
    return 128 * max(x_enc, sigma_enc) - 2624 + _log2_fixed(top)


def _bf16_code(value: float) -> int:
    return (_f32_bits(value) >> 16) & 0xFFFF


def _row_alpha(l2: float, linf: float, delta: float, rank: int) -> float:
    """``alpha = 448 / (delta*sqrt(rank)*l2 + linf)``, each step rounded to bf16."""
    bound = _bf16_fma(_delta_r(delta, rank), l2, linf)
    return _bf16_div(_MAX_E4M3, bound)


def _delta_r(delta: float, rank: int) -> float:
    """``bf16(delta * sqrt(rank))``, through f32, as the verifier derives it."""
    return _round_f32_to_bf16(_f32(delta * math.sqrt(rank)))


def _bf16_div(num: float, den: float) -> float:
    quotient = torch.tensor(num, dtype=torch.float32) / torch.tensor(den, dtype=torch.float32)
    return _round_f32_to_bf16(float(quotient))


def _bf16_fma(a: float, b: float, c: float) -> float:
    """``a*b + c`` rounded to bf16 once, matching the verifier's ``bf16_fma``.

    The product of two bf16 values is exact in f64. TwoSum recovers the exact
    sum; it rounds to f32 with ties toward odd so the final bf16 cast does not
    double-round.
    """
    product = a * b
    total = product + c
    residual = (product - (total - c)) + (c - (total - (total - c)))
    bits = _f32_bits(total)
    rounded = _f32_from_bits(bits)
    error = (total - rounded) + residual
    if error != 0.0 and (bits & 1) == 0 and math.isfinite(rounded):
        rounded = _f32_from_bits(_next_f32_bits(bits, up=error > 0.0))
    return _round_f32_to_bf16(rounded)


def _round_f32_to_bf16(value: float) -> float:
    bits = _f32_bits(value)
    # Round to nearest, ties to even: the kept significand's LSB is the tie bit.
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return _f32_from_bits((rounded & 0xFFFF) << 16)


def _next_f32_bits(bits: int, *, up: bool) -> int:
    """The next f32 bit pattern above or below ``bits`` (IEEE ``nextUp``)."""
    if bits & 0x7FFFFFFF == 0:
        return 0x00000001 if up else 0x80000001
    if up == (bits < 0x80000000):
        return (bits + 1) & 0xFFFFFFFF
    return (bits - 1) & 0xFFFFFFFF


def _f32(value: float) -> float:
    return _f32_from_bits(_f32_bits(value))


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _num(value: float) -> str:
    if math.isfinite(value) and value == math.trunc(value) and abs(value) < 1e15:
        return str(int(value))
    return format(value, ".8g")
