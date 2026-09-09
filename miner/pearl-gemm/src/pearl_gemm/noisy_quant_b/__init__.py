"""``noisy_quant_b``: the B-side sibling of ``noisy_quant``.

One fused kernel builds the B-side mixed-GEMM representation from the two
committed weight planes: open in registers, E2 from ``cB``, pinned noise dot,
quantize, peel. ``_host.py`` holds the functional caller-owned launch; the
device kernel is ``noisy_quant``'s, launched on mirrored operands.
"""

from ._host import (
    NoisyQuantBConfig,
    b_peel_for_a,
    noisy_quant_b,
)

__all__ = [
    "NoisyQuantBConfig",
    "b_peel_for_a",
    "noisy_quant_b",
]
