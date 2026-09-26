"""fp8_sim_hopper is exact: the vectorized simulator matches a scalar
transliteration of the C++ simulator (https://github.com/badasherez/gpu-simulator)
with the whitepaper's promotion rule (a window of 4 groups is added to an
FP32 total with one RNE rounding), bit for bit, across input shapes, scales
and every e4m3 code. CPU-only."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch
from miner_base.fp8_sim_common import ATOM_K
from miner_base.fp8_sim_hopper import _PROMOTE_GROUPS, _ZERO_EXP, matmul_fp8_sim_hopper

_INT_W = 14
_INT_SHIFT = 10
_FP32_MIN_EXP = -126


# --- scalar reference -------------------------------------------------------


def _mul_e4m3(a: int, b: int) -> tuple[bool, int, int]:
    """multiply_fp8e4m3_to_gfloat, on raw bytes."""
    sign_a, exp_a, sig_a = (a >> 7) & 1, (a >> 3) & 0xF, a & 0x7
    sign_b, exp_b, sig_b = (b >> 7) & 1, (b >> 3) & 0xF, b & 0x7
    s_a = (sig_a | 0x8) if exp_a != 0 else sig_a
    s_b = (sig_b | 0x8) if exp_b != 0 else sig_b
    exp = max(exp_a, 1) + max(exp_b, 1) - 14
    sig = (s_a * s_b) << 17
    sign = bool(sign_a ^ sign_b)
    return (sign, _ZERO_EXP, 0) if sig == 0 else (sign, exp, sig)


def _group_sum(terms: list[tuple[bool, int, int]]) -> tuple[bool, int, int]:
    """Hopper_fp8_simulator::group_sum: align to the max exponent, sum in a
    14-bit integer, renormalize with truncation."""
    max_exp = max([_ZERO_EXP] + [e for _, e, _ in terms])
    total = 0
    for sgn, e, s in terms:
        shift = max_exp - e
        if shift >= 32:
            continue
        aligned = (s >> _INT_SHIFT) >> shift
        total += -aligned if sgn else aligned
    sign, mag = total < 0, abs(total)
    width = mag.bit_length()
    if width == 0:
        return sign, _ZERO_EXP, 0
    exp = max_exp + width - _INT_W
    mag = mag >> (width - _INT_W) if width > _INT_W else mag << (_INT_W - width)
    if exp < _FP32_MIN_EXP:
        mag >>= _FP32_MIN_EXP - exp
        exp = _FP32_MIN_EXP
    mag <<= _INT_SHIFT
    return (sign, _ZERO_EXP, 0) if mag == 0 else (sign, exp, mag)


def _to_f32(sign: bool, exp: int, sig: int) -> float:
    """Gfloat::operator float (bit assembly)."""
    if sig == 0:
        return 0.0
    bits = 0x80000000 if sign else 0
    exp_bits = exp + 127 - (0 if sig & 0x800000 else 1)
    bits |= ((exp_bits & 0xFF) << 23) | (sig & 0x7FFFFF)
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _dot(a8: torch.Tensor, b8: torch.Tensor, i: int, j: int) -> float:
    """One cell: ascending groups of 32 into a window accumulator that
    restarts at +0 every 4 groups, each window promoted into an FP32 total
    with one RNE rounding (numpy's float32 add)."""
    k = a8.shape[1]
    total = np.float32(0.0)
    acc = (False, _ZERO_EXP, 0)
    groups = range(0, k, ATOM_K)
    for g, k0 in enumerate(groups):
        if g % _PROMOTE_GROUPS == 0:
            acc = (False, _ZERO_EXP, 0)
        terms = [acc] + [
            _mul_e4m3(int(a8[i, u]), int(b8[j, u])) for u in range(k0, min(k0 + ATOM_K, k))
        ]
        acc = _group_sum(terms)
        if (g + 1) % _PROMOTE_GROUPS == 0 or g == len(groups) - 1:
            total = np.float32(total + np.float32(_to_f32(*acc)))
    return float(total)


def matmul_scalar(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a8, b8 = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    out = torch.empty(a8.shape[0], b8.shape[0], dtype=torch.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = _dot(a8, b8, i, j)
    return out


# --- the test ---------------------------------------------------------------


def _all_codes_operands() -> tuple[torch.Tensor, torch.Tensor]:
    """Every e4m3 code (zeros, subnormals, both signs; NaN excluded) in both operands."""
    codes = torch.tensor([c for c in range(256) if (c & 0x7F) != 0x7F], dtype=torch.uint8)
    a = codes.repeat(2)[: 2 * 127].reshape(2, 127)
    b = codes.flip(0).repeat(3)[: 3 * 127].reshape(3, 127)
    return a.view(torch.float8_e4m3fn), b.view(torch.float8_e4m3fn)


def _random_operands(m: int, n: int, k: int, scale: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=gen) * scale).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=gen) * scale).to(torch.float8_e4m3fn)
    return a, b


# K below, at, above, and not a multiple of the 128-term promotion window;
# scales spanning subnormal-heavy to near-saturated products.
_SHAPES = [(4, 5, 16), (3, 3, 32), (2, 4, 96), (5, 2, 100), (2, 3, 128), (3, 2, 300), (2, 2, 512)]
_SCALES = [0.05, 1.0, 40.0]
_CASES = [
    pytest.param(_random_operands(m, n, k, s, seed=i), id=f"{m}x{n}x{k}-scale{s}")
    for i, ((m, n, k), s) in enumerate((shape, s) for shape in _SHAPES for s in _SCALES)
] + [pytest.param(_all_codes_operands(), id="all-e4m3-codes")]


@pytest.mark.parametrize("operands", _CASES)
def test_fp8_matmul_on_hopper_is_exact(operands):
    a, b = operands
    got = matmul_fp8_sim_hopper(a, b)
    want = matmul_scalar(a, b)
    assert torch.equal(got, want), (got - want).abs().max()
