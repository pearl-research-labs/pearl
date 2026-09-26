"""Vectorized ``torch`` port of the Hopper FP8 (QGMMA) MMA arithmetic.

Implements the ``QGMMA.64xNx32.F32.E4M3.E4M3`` tensor-core opcode arithmetic
(see https://github.com/badasherez/gpu-simulator for a reference
implementation):

* e4m3 products are EXACT: 4-bit x 4-bit significands -> <= 8 bits, kept in
  a 25-bit fixed-point significand (``(sa * sb) << 17``); subnormal inputs
  are NOT normalized before the multiply.
* accumulation is one group per instruction: the running accumulator plus
  the instruction's 32 products. Every term is aligned to the group's max
  exponent AT THE 14-BIT INTERNAL WIDTH (``(sig >> 10) >> shift`` --
  right-shift truncation, towards zero on magnitudes), the aligned terms
  are summed as integers, and the normalized result is truncated (towards
  zero) back to 14 bits.
* the group result expands exactly to fp32 (``sig14 << 10``); values that
  underflow the fp32 normal range are denormalized first.
* PROMOTION every ``_PROMOTE_GROUPS = 4`` instructions (128 terms along K):
  the chained WGMMAs accumulate into a fresh low-precision *window*
  accumulator (starting at +0), and after the window's last group its value
  is added to an FP32 total with one round-to-nearest-even,
  ``C <- RNE_FP32(C + c)`` (``C = +0`` before the first window). This is the
  standard mitigation for the 14-bit internal significand, adopted by
  production FP8 kernels (DeepSeek-V3 / DeepGEMM / QuACK) and pinned by the
  whitepaper (Section "Matrix multiplication on the whitelisted devices",
  Appendix "E4M3FN matrix multiplication"). Windows start at ``u = 0``; when
  ``128 !| K`` the last window is shorter (equivalent to zero-padding it).
"""

from __future__ import annotations

import torch

from .fp8_sim_common import (
    ATOM_K,
    E4M3_SIG_EXP_OFFSET,
    FP32_MANTISSA_BITS,
    FP32_MIN_EXP,
    fp8_to_raw_bytes,
    nan_mask,
    split_e4m3,
)

_ZERO_EXP = -139  # exponent sentinel for a zero-valued term
_INT_W = 14  # internal significand width (13 fractional + 1 implicit)
_INT_SHIFT = 10  # fp32 significand width (24) - _INT_W
_PROD_SHIFT = 17  # (sa * sb) << 17: 8-bit product -> 25-bit significand
# Exponent of a product in the group-sum convention (value = sig * 2^(exp - 23)):
# sig_a * sig_b * 2^(ea + eb - 2*10) == ((sig_a * sig_b) << 17) * 2^(ea + eb - 14 - 23).
_PROD_EXP_OFFSET = 2 * E4M3_SIG_EXP_OFFSET - (FP32_MANTISSA_BITS - _PROD_SHIFT)  # 14
_PROMOTE_GROUPS = 4  # instructions per promotion window (4 * 32 = 128 terms along K)

PROMOTE_GROUPS = _PROMOTE_GROUPS  # public alias: the jackpot policy's window size (in groups)


def _group_sum(
    sign: torch.Tensor, exp: torch.Tensor, sig: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One QGMMA accumulation group over the last dim; returns the result
    as a (sign, exponent, 24-bit significand) triple.

    Inputs are terms in the same convention (value =
    ``(-1)^sign * sig * 2^(exp - 23)``); zero terms must carry
    ``exp == _ZERO_EXP`` and ``sig == 0`` so they never set the max.
    """
    max_exp = exp.max(dim=-1).values
    # Align at the 14-bit internal width, truncating towards zero on the
    # positive significands (``(sig >> 10) >> shift``). Terms shifted out
    # entirely contribute 0 (``sig >> 10`` has <= 15 bits, so any shift
    # >= 15 already yields 0).
    shift = (max_exp.unsqueeze(-1) - exp).clamp_(max=31)
    aligned = (sig >> _INT_SHIFT) >> shift
    total = torch.where(sign != 0, -aligned, aligned).sum(dim=-1, dtype=torch.int32)

    r_sign = (total < 0).to(torch.int32)
    mag = total.abs()
    # bit_length(mag): frexp on float64 is exact (magnitudes < 2^21)
    width = torch.frexp(mag.to(torch.float64)).exponent.to(torch.int32)
    r_exp = max_exp + width - _INT_W
    # truncate/pad the normalized sum to the 14-bit internal width
    d = width - _INT_W
    r_sig = torch.where(d > 0, mag >> d.clamp(min=0), mag << (-d).clamp(min=0))
    # fp32 denormal clamp (before the exact <<10 expansion)
    sub = r_exp < FP32_MIN_EXP
    r_sig = torch.where(sub, r_sig >> (FP32_MIN_EXP - r_exp).clamp(min=0, max=31), r_sig)
    r_exp = torch.where(sub, torch.full_like(r_exp, FP32_MIN_EXP), r_exp)
    # a zero significand is +0.0 regardless of the sum's sign
    zero = r_sig == 0
    r_sign = torch.where(zero, torch.zeros_like(r_sign), r_sign)
    r_exp = torch.where(zero, torch.full_like(r_exp, _ZERO_EXP), r_exp)
    return r_sign, r_exp, r_sig << _INT_SHIFT


def _to_f32(sign: torch.Tensor, exp: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    """(sign, exponent, significand) triple -> exact fp32."""
    mag = torch.ldexp(sig.to(torch.float64), exp - FP32_MANTISSA_BITS)
    return torch.where(sign != 0, -mag, mag).to(torch.float32)


def matmul_fp8_sim_hopper(
    a: torch.Tensor, b: torch.Tensor, chunk_elems: int = 1 << 20
) -> torch.Tensor:
    """``a @ b.T`` with the QGMMA opcode arithmetic; e4m3 in, fp32 out.

    Inputs:
    - ``a``: (M, K) ``float8_e4m3fn``
    - ``b``: (N, K) ``float8_e4m3fn``

    Output:
    - (M, N) fp32: the matmul result

    Proceess:
    - Along the K dimension, consume in ascending groups of 32 (one instruction each) chained
    - Promote the window accumulator into the FP32 total ``_PROMOTE_GROUPS`` groups
    - A final partial group (K % 32) matches a zero-padded instruction (zero products change nothing)

    Correct-or-NaN: a NaN byte in a row of ``a`` or ``b`` makes every output
    element that row contracts into NaN.
    """
    total, _, _ = _matmul_fp8_sim_hopper(a, b, chunk_elems, replay=False)
    return total


def matmul_fp8_sim_hopper_replay(
    a: torch.Tensor, b: torch.Tensor, chunk_elems: int = 1 << 20
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like `matmul_fp8_sim_hopper` but also replays every cell's accumulation.

    Returns:
    - (M, N) fp32: the matmul result
    - (M, N, ceil(K/32)) fp32: the running total ``RNE_FP32(C + c)`` after each group
    - (M, N, ceil(K/32)) fp32: the window accumulator ``c`` after each group
    """
    return _matmul_fp8_sim_hopper(a, b, chunk_elems, replay=True)


def _matmul_fp8_sim_hopper(
    a: torch.Tensor, b: torch.Tensor, chunk_elems: int, replay: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Util implementation for `matmul_fp8_sim_hopper` and `matmul_fp8_sim_hopper_replay`.

    Inputs:
    - `a`: (M, K) `float8_e4m3fn`
    - `b`: (N, K) `float8_e4m3fn`
    - `chunk_elems`: the number of elements to process in each chunk
    - `replay`: whether to replay the accumulation

    See their docstrings for more details.
    """
    assert a.shape[1] == b.shape[1]
    a8, b8 = fp8_to_raw_bytes(a), fp8_to_raw_bytes(b)
    nan_a, nan_b = nan_mask(a8), nan_mask(b8)  # (m,), (n,)
    m, k = a.shape
    n = b.shape[0]
    num_groups = -(-k // ATOM_K)  # ceil(k / ATOM_K)
    dev = a.device
    out = torch.empty(m, n, dtype=torch.float32, device=dev)
    group_shape = (m, n, num_groups if replay else 0)
    totals = torch.empty(group_shape, dtype=torch.float32, device=dev)
    windows = torch.empty(group_shape, dtype=torch.float32, device=dev)

    # the number of rows to process in each chunk
    chunk_rows = max(1, chunk_elems // max(1, n))

    # Iterate over the rows of the input tensors in chunks.
    for m0 in range(0, m, chunk_rows):
        am = a8[m0 : m0 + chunk_rows]
        num_curr_rows = am.shape[0]
        # The FP32 total C (+0 before the first window).
        total = torch.zeros(num_curr_rows, n, dtype=torch.float32, device=dev)
        is_nan_product = nan_a[m0 : m0 + chunk_rows].unsqueeze(-1) | nan_b.unsqueeze(0)
        for group_idx, k0 in enumerate(range(0, k, ATOM_K)):
            if group_idx % _PROMOTE_GROUPS == 0:
                # A fresh window accumulator c = +0.
                acc_sign = torch.zeros(num_curr_rows, n, dtype=torch.int32, device=dev)
                acc_exp = torch.full((num_curr_rows, n), _ZERO_EXP, dtype=torch.int32, device=dev)
                acc_sig = torch.zeros(num_curr_rows, n, dtype=torch.int32, device=dev)
            # (s, e, m) = (sign, exponent, significand) for each element in the group.
            sa, ea, ma = split_e4m3(am[:, k0 : k0 + ATOM_K])  # (num_curr_rows, group_size)
            sb, eb, mb = split_e4m3(b8[:, k0 : k0 + ATOM_K])  # (n, group_size)

            # (p_sig, p_exp, p_sign) = (significand, exponent, sign) for each element in the group.
            p_sig = (
                ma.unsqueeze(1) * mb.unsqueeze(0)
            ) << _PROD_SHIFT  # (num_curr_rows, n, group_size)
            p_exp = ea.unsqueeze(1) + eb.unsqueeze(0) - _PROD_EXP_OFFSET
            p_exp = torch.where(p_sig == 0, _ZERO_EXP, p_exp)
            p_sign = sa.unsqueeze(1) ^ sb.unsqueeze(0)

            acc_sign, acc_exp, acc_sig = _group_sum(
                torch.cat([acc_sign.unsqueeze(-1), p_sign], dim=-1),
                torch.cat([acc_exp.unsqueeze(-1), p_exp], dim=-1),
                torch.cat([acc_sig.unsqueeze(-1), p_sig], dim=-1),
            )
            window = _to_f32(acc_sign, acc_exp, acc_sig)  # the window accumulator c
            if replay:
                # If `replay` is True, store the partial accumulator values for each group.
                windows[m0 : m0 + chunk_rows, :, group_idx] = torch.where(
                    is_nan_product, torch.nan, window
                )
                totals[m0 : m0 + chunk_rows, :, group_idx] = torch.where(
                    is_nan_product, torch.nan, total + window
                )
            # Last group: accumulate the window into the total.
            if (group_idx + 1) % _PROMOTE_GROUPS == 0 or group_idx == num_groups - 1:
                # Promotion: C <- RNE_FP32(C + c). torch's fp32 add is IEEE
                # round-to-nearest-even, and an exact zero sum yields +0.
                total = total + window
        out[m0 : m0 + chunk_rows] = torch.where(is_nan_product, torch.nan, total)
    return out, totals, windows
