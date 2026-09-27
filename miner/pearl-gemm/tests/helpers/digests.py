"""Pinned digests of the protocol's bit-exact operands.

``tests/fixtures/reference_digests.json`` holds SHA-256 digests of the noise
lines and the noisy-quantized A/B operands for the fixed inputs these tests
build, recorded from the former bit-exact CPU reference implementation. The
inputs are drawn on the
host so they reproduce exactly; any drift in a kernel's bits changes its
digest. Each case also stores ``peel_tol``, the relative-error bound for the
peel's matmul half (twice the reference's own deviation from the fp64-exact
product, with an absolute floor).
"""

import hashlib
import json
from functools import cache
from pathlib import Path

import torch

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reference_digests.json"


@cache
def reference_digests() -> dict:
    return json.loads(_FIXTURE.read_text())


def digest(t: torch.Tensor) -> str:
    """SHA-256 of ``t``'s values in row-major order.

    FP8 hashes its raw bytes; other floats hash as float32 with ``-0.0``
    folded into ``+0.0``, so only the value (not the storage dtype or shape)
    is pinned.
    """
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.float8_e4m3fn:
        data = t.view(torch.uint8)
    elif t.is_floating_point():
        data = t.float() + 0.0
    else:
        data = t
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def fixture_input(rows: int, k: int, seed: int) -> torch.Tensor:
    """The pinned cases' BF16 operand, drawn on the host (CUDA RNG streams differ)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, k, generator=g, dtype=torch.float32).to(torch.bfloat16) * 1.5
