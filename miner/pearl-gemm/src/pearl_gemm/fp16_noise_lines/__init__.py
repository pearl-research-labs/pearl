"""Bit-exact A100 (``sm_80``) FP16 deterministic noise-line generation for the
FP16 proof-of-useful-work scheme.

Reproduces ``zk-pow/src/api/fp16/noise.rs``'s ``sample_line`` / ``sample_noise``
bit-for-bit on real GA100 silicon: per-line keyed-BLAKE3 XOF, exact integer
``isqrt`` L2 normalization to the shared constant norm, and RNE rounding to FP16.
Produces the ``E`` ``(rows x r)`` and ``F`` ``(k x r)`` factors as ``u16``.
"""

from ._host import (
    FACTOR_E,
    FACTOR_F,
    LABEL_NOISE_LINE,
    SIDE_A,
    SIDE_B,
    Noise16,
    line_key,
    noise_lines,
    sample_noise,
)

__all__ = [
    "FACTOR_E",
    "FACTOR_F",
    "LABEL_NOISE_LINE",
    "SIDE_A",
    "SIDE_B",
    "Noise16",
    "line_key",
    "noise_lines",
    "sample_noise",
]
