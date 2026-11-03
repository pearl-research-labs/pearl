"""Host launch for the bit-exact A100 (``sm_80``) FP16 noise-line generation.

Reproduces ``zk-pow/src/api/fp16/noise.rs``'s ``sample_line`` / ``sample_noise``
on real GA100 silicon, bit-for-bit against the verifier. Each line is a keyed
BLAKE3 XOF draw normalized to the shared constant L2 norm and cast to FP16:

1. ``line_key = subkey(LABEL_NOISE_LINE, seed)`` -- keyed BLAKE3 of the 24-byte
   label ``pearl/v4/FP16/noise-line`` under the 32-byte ``seed``, computed on the
   host with the ``blake3`` crate-equivalent package (same keyed semantics as
   ``blake3_digest(label, Some(seed))``).
2. For each line index: keyed-BLAKE3-XOF the 64-byte address material
   ``side | factor | line(u32 LE)`` under ``line_key`` to ``rank`` bytes, decode
   to signed integers, L2-normalize with the exact integer ``isqrt`` + one-BF16
   division recipe, and round each entry to FP16 (``u16``).

The BLAKE3 compression + normalization run in a small hand-written CUDA
extension compiled with ``nvcc -arch=sm_80`` (loaded through
``torch.utils.cpp_extension``), exactly like ``fp16_gemm`` / ``fp16_noisy_quant``.
The ``line_key`` derivation (one keyed BLAKE3 per seed, not per line) is done on
the host and passed to the kernel as 8 little-endian ``u32`` words, mirroring how
the FP8 ``noise_lines`` kernel takes its side noise-line key.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import blake3
import torch

from .._utils._arch import Arch, require_arch

_SUPPORTED_ARCHS = (Arch.SM80,)

# The FP16 noise-line transcript label (``LABEL_NOISE_LINE`` in the reference).
LABEL_NOISE_LINE = b"pearl/v4/FP16/noise-line"

# Address discriminants (committed wire bytes), matching ``Side`` / ``NoiseFactor``.
SIDE_A = 0
SIDE_B = 1
FACTOR_E = 0
FACTOR_F = 1


@functools.lru_cache(maxsize=1)
def _extension():
    """Compile (once) and return the loaded ``sm_80`` CUDA extension."""
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "_kernel_sm80.cu")
    return load(
        name="pearl_fp16_noise_lines_sm80",
        sources=[src],
        extra_cuda_cflags=["-arch=sm_80", "-O3"],
        verbose=False,
    )


def line_key(seed: bytes) -> bytes:
    """``subkey(LABEL_NOISE_LINE, seed)`` -- the per-seed noise-line key.

    Keyed BLAKE3 of the label under the 32-byte ``seed``, identical to the
    reference's ``blake3_digest(LABEL_NOISE_LINE, Some(seed))``.
    """
    if len(seed) != 32:
        raise ValueError(f"seed must be 32 bytes, got {len(seed)}")
    return blake3.blake3(LABEL_NOISE_LINE, key=seed).digest(length=32)


def _line_key_words(seed: bytes, device: torch.device) -> torch.Tensor:
    """The 8 little-endian ``u32`` words of :func:`line_key`, on ``device``."""
    key = line_key(seed)
    words = [int.from_bytes(key[4 * i : 4 * i + 4], "little") for i in range(8)]
    # int32 view of the u32 words (two's-complement reinterpretation is fine; the
    # kernel reads them as raw 32-bit words).
    signed = [w - (1 << 32) if w >= (1 << 31) else w for w in words]
    return torch.tensor(signed, dtype=torch.int32, device=device)


def _indices_tensor(indices, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(indices, dtype=torch.int32, device=device)
    if t.ndim != 1:
        raise ValueError("indices must be 1D")
    return t.contiguous()


def noise_lines(
    seed: bytes,
    side: int,
    factor: int,
    indices,
    r: int,
    device: torch.device | int | None = None,
) -> torch.Tensor:
    """Draw normalized FP16 noise lines for ``indices`` under ``seed``.

    Returns a ``(len(indices), r)`` ``int16`` tensor of FP16 bit patterns (view
    as ``uint16``), each row the keyed-BLAKE3 line for ``(side, factor, index)``,
    bit-exact to ``sample_line(subkey(LABEL_NOISE_LINE, seed), side, factor,
    index, r)``.
    """
    dev = torch.device("cuda", device) if isinstance(device, int) else (device or torch.device("cuda"))
    require_arch("fp16_noise_lines", dev, *_SUPPORTED_ARCHS)
    key = _line_key_words(seed, dev)
    idx = _indices_tensor(indices, dev)
    return _extension().fp16_noise_lines(key, int(side), int(factor), idx, int(r))


@dataclass
class Noise16:
    """Both operands' FP16 noise factors (mirrors the Rust ``Noise16``).

    ``e_a``/``e_b`` are ``(|a_rows|, r)`` / ``(|b_cols|, r)``; ``f_a``/``f_b`` are
    ``(k, r)``. All are ``int16`` FP16 bit patterns (view as ``uint16``).
    """

    e_a: torch.Tensor
    f_a: torch.Tensor
    e_b: torch.Tensor
    f_b: torch.Tensor


def sample_noise(
    seed_a: bytes,
    seed_b: bytes,
    k: int,
    r: int,
    a_rows,
    b_cols,
    device: torch.device | int | None = None,
) -> Noise16:
    """Draw the four FP16 noise factors for one tile, matching ``sample_noise``.

    ``E_A`` keys off ``seed_a`` (``Side::A``, ``NoiseFactor::E``, per ``a_rows``);
    ``E_B``, ``F_A``, ``F_B`` all key off ``seed_b`` with distinct ``Side`` /
    ``NoiseFactor`` addresses (``F`` bases are the shared ``0..k`` basis). The
    layout is the reference's: ``E`` row-major ``(rows x r)``, ``F`` row-major
    ``(k x r)``.
    """
    dev = torch.device("cuda", device) if isinstance(device, int) else (device or torch.device("cuda"))
    f_lines = list(range(k))
    e_a = noise_lines(seed_a, SIDE_A, FACTOR_E, a_rows, r, dev)
    e_b = noise_lines(seed_b, SIDE_B, FACTOR_E, b_cols, r, dev)
    f_a = noise_lines(seed_b, SIDE_A, FACTOR_F, f_lines, r, dev)
    f_b = noise_lines(seed_b, SIDE_B, FACTOR_F, f_lines, r, dev)
    return Noise16(e_a=e_a, f_a=f_a, e_b=e_b, f_b=f_b)
