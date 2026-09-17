"""Vectorized ``torch`` port of the Blackwell FP8 (tcgen05.mma kind::f8f6f4) atom.

Implements the ``tcgen05.mma`` kind::f8f6f4 e4m3 atom arithmetic
(cross-checked against B200 silicon):

* e4m3 products are EXACT and UN-NORMALIZED: 4-bit x 4-bit significands
  multiplied as integers in units of ``2^-18``; subnormal inputs keep the
  minimum normal exponent (stored exponent -6) and are NOT renormalized.
* one atom (contraction depth K = 32) is a SINGLE accumulation group: the
  FP32 accumulator C plus all 32 products (33 summands), no sub-grouping.
  The window anchor is the max STORED exponent over the nonzero summands
  INCLUDING C (zero summands never raise it); every summand is truncated
  towards zero at the window bottom ``2^(anchor - 25)`` (25 fractional
  bits below the anchor), and the aligned terms are summed exactly.
* the group sum is rounded to FP32 towards zero. With all-zero products C
  passes through exactly (including subnormal FP32 values).

``matmul_fp8_sim_blackwell(a, b)`` computes ``a @ b.T`` for e4m3 operands
consuming K in ascending chunks of 32 -- one atom per chunk, chained
through the FP32 accumulator, the zero-padded remainder chunk LAST (zero
products are no-ops in the group sum). This is the placement measured for
cuBLAS ``_scaled_mm`` / CUTLASS SM100 on Blackwell (pure chain, no promotion
stage, no split-K).

Security contract: never return a silently wrong finite result -- only the
bit-exact atom arithmetic, or NaN. e4m3 NaN bytes (``0x7F``/``0xFF``) have
no encoding in ``_decode_e4m3``, so a NaN in any row of ``a``/column of
``b`` poisons its contracted outputs with quiet NaN instead of a bogus
decode, mirroring IEEE NaN propagation on real silicon. Committed operands
are NaN-free by verifier checks, but derived operands (e.g. adversarial
noised quantization) can still carry NaN bytes.

Only what the reference miner needs is otherwise implemented: there is no
external C input (the chain starts at +0), and FP32 overflow to infinity is
not modeled (unreachable: |sum| <= K * 448^2 stays far below 2^128 for any
practical K).

Everything is integer ``torch`` ops (CPU or CUDA), bit-exact by
construction; see ``_atom`` for the group-accumulation and rounding
arithmetic.
"""

from __future__ import annotations

import torch

_T_FRAC = 25  # fractional bits kept below the accumulation-window anchor
_ATOM_K = 32  # contraction depth of one tcgen05.mma f8f6f4 atom
_PROD_UNIT = -18  # e4m3 int values are in units of 2^-9; products in 2^-18
_FP32_MIN_EXP = -126
_SENT = -(1 << 24)  # stored exponent of a zero summand: never raises the anchor


def _decode_e4m3(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw e4m3 bytes -> (int value in units of 2^-9, stored exponent).

    Subnormals keep their 3-bit significand un-normalized with stored exponent
    -6; zeros carry the ``_SENT`` sentinel. NaN bytes (``0x7F``/``0xFF``)
    decode to garbage -- callers must mask them via ``_nan_mask`` (see the
    module docstring's security contract).
    """
    c = codes.to(torch.int32)
    e = (c >> 3) & 0xF
    m = c & 0x7
    # normal: (8+m) * 2^(e-10) = ((8+m) << (e-1)) * 2^-9;  subnormal: m * 2^-9
    mag = torch.where(e != 0, (8 + m) << (e.clamp(min=1) - 1), m)
    iv = torch.where((c >> 7) != 0, -mag, mag)
    es = torch.where(e != 0, e - 7, torch.full_like(e, -6))
    return iv, torch.where(mag == 0, torch.full_like(es, _SENT), es)


def _shift_trunc(v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``v * 2^s`` onto the integer grid, truncating towards zero (int64 ``v``).

    Shift amounts are clamped to [-63, 63]: right shifts of magnitudes
    < 2^63 saturate to 0 correctly, and at every call site a logical left
    shift > 63 only ever applies to ``v == 0`` (a nonzero summand's stored
    exponent bounds the anchor, hence the shift).
    """
    left = s.clamp(min=0, max=63).to(torch.int64)
    right = (-s).clamp(min=0, max=63).to(torch.int64)
    mag = (v.abs() << left) >> right
    return torch.where(v < 0, -mag, mag)


def _atom(
    p_iv: torch.Tensor,
    p_es: torch.Tensor,
    acc_val: torch.Tensor,
    acc_grid: torch.Tensor,
    acc_es: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One accumulation group: C + the atom's products, over the last dim.

    Products are ``(p_iv, p_es)``: exact int64 values in units of
    ``2^_PROD_UNIT`` with their stored exponents (``_SENT`` for zero). The
    FP32 accumulator is the exact triple (signed int64 value on the grid
    ``2^acc_grid``, grid exponent, stored exponent -- ``_SENT`` for zero);
    the result is returned in the same triple form (its value is always an
    exactly representable FP32).
    """
    anchor = torch.maximum(p_es.max(dim=-1).values, acc_es)
    bottom = anchor - _T_FRAC
    # Align every summand at the window bottom, truncating towards zero;
    # aligned magnitudes are < 2^27 each (value < 2^(stored_exp + 2) and
    # stored_exp <= anchor), so the 33-term int64 sum is exact (< 2^33).
    total = _shift_trunc(p_iv, (_PROD_UNIT - bottom).unsqueeze(-1)).sum(dim=-1)
    total = total + _shift_trunc(acc_val, acc_grid - bottom)

    # Round total * 2^bottom to FP32 towards zero.
    mag = total.abs()
    # bit_length(mag): frexp on float64 is exact (magnitudes < 2^33)
    width = torch.frexp(mag.to(torch.float64)).exponent.to(torch.int32)
    e = width - 1 + bottom  # floor(log2 |value|)
    sub = e < _FP32_MIN_EXP
    grid = torch.where(sub, torch.full_like(e, -149), e - 23)
    q = _shift_trunc(mag, bottom - grid)  # 24-bit FP32 significand on 2^grid
    r_es = torch.where(sub, torch.full_like(e, _FP32_MIN_EXP), e)
    r_es = torch.where(q == 0, torch.full_like(r_es, _SENT), r_es)
    return torch.where(total < 0, -q, q), grid, r_es


def _nan_mask(codes: torch.Tensor) -> torch.Tensor:
    """Per-row/column ``bool``: does this raw e4m3 byte vector contain a NaN
    (``0x7F`` / ``0xFF``)? ``codes``: (rows, K) ``uint8``."""
    return ((codes & 0x7F) == 0x7F).any(dim=-1)


def matmul_fp8_sim_blackwell(
    a: torch.Tensor, b: torch.Tensor, chunk_elems: int = 1 << 18
) -> torch.Tensor:
    """``a @ b.T`` with the Blackwell f8f6f4 atom arithmetic; e4m3 in, fp32 out.

    ``a``: (M, K) ``float8_e4m3fn``; ``b``: (N, K) ``float8_e4m3fn``.
    K is consumed in ascending chunks of 32 (one atom each) chained through
    the FP32 accumulator; a final partial chunk (K % 32) matches the
    zero-padded LAST atom. Output rows are processed in chunks bounding
    peak memory at roughly ``chunk_elems * 32 * 8 * 4`` bytes.

    Correct-or-NaN: a NaN byte (``0x7F``/``0xFF``) in any row of ``a``/column
    of ``b`` poisons every output element it contracts into with NaN -- see
    the module docstring's security contract.
    """
    return _matmul_fp8_sim_blackwell(a, b, chunk_elems, partials=False)


def matmul_fp8_sim_blackwell_partials(
    a: torch.Tensor, b: torch.Tensor, chunk_elems: int = 1 << 18
) -> torch.Tensor:
    """Like :func:`matmul_fp8_sim_blackwell` but returns every cell's running
    FP32 accumulator after each of the ``ceil(K / 32)`` atoms (shape
    ``(M, N, ceil(K/32))``), the last slice equal to the final matmul. These
    are the partial sums ``c_v`` the jackpot policy's prefix-inclusive
    anchor (check 3, ``jackpot_policy.rs``) consumes.
    """
    return _matmul_fp8_sim_blackwell(a, b, chunk_elems, partials=True)


def _matmul_fp8_sim_blackwell(
    a: torch.Tensor, b: torch.Tensor, chunk_elems: int, partials: bool
) -> torch.Tensor:
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[1]
    m, k = a.shape
    n = b.shape[0]
    a8 = a.contiguous().view(torch.uint8)
    b8 = b.contiguous().view(torch.uint8)
    nan_a = _nan_mask(a8)  # (m,)
    nan_b = _nan_mask(b8)  # (n,)
    a_iv, a_es = _decode_e4m3(a8)
    b_iv, b_es = _decode_e4m3(b8)
    num_atoms = -(-k // _ATOM_K)  # ceil(k / _ATOM_K)
    out = torch.empty(m, n, num_atoms if partials else 1, dtype=torch.float32, device=a.device)
    rows = max(1, chunk_elems // max(1, n))
    for m0 in range(0, m, rows):
        aiv, aes = a_iv[m0 : m0 + rows], a_es[m0 : m0 + rows]
        mc = aiv.shape[0]
        acc_val = torch.zeros(mc, n, dtype=torch.int64, device=a.device)
        acc_grid = torch.zeros(mc, n, dtype=torch.int32, device=a.device)
        acc_es = torch.full((mc, n), _SENT, dtype=torch.int32, device=a.device)
        poisoned = nan_a[m0 : m0 + rows].unsqueeze(-1) | nan_b.unsqueeze(0)
        for g, k0 in enumerate(range(0, k, _ATOM_K)):
            # (mc, n, g) exact products; int32 * int32 can overflow, go int64
            p_iv = (
                aiv[:, None, k0 : k0 + _ATOM_K].to(torch.int64) * b_iv[None, :, k0 : k0 + _ATOM_K]
            )
            p_es = aes[:, None, k0 : k0 + _ATOM_K] + b_es[None, :, k0 : k0 + _ATOM_K]
            p_es = torch.where(p_iv == 0, _SENT, p_es)
            acc_val, acc_grid, acc_es = _atom(p_iv, p_es, acc_val, acc_grid, acc_es)
            if partials:
                # exact: |acc_val| < 2^24, and acc_val * 2^acc_grid is an FP32 value
                chunk = torch.ldexp(acc_val.to(torch.float64), acc_grid).to(torch.float32)
                out[m0 : m0 + rows, :, g] = torch.where(poisoned, torch.nan, chunk)
        if not partials:
            chunk = torch.ldexp(acc_val.to(torch.float64), acc_grid).to(torch.float32)
            out[m0 : m0 + rows, :, 0] = torch.where(poisoned, torch.nan, chunk)
    return out if partials else out[:, :, 0]
