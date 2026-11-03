"""Bit-exact A100 (``sm_80``) FP16 end-to-end tile pipeline for the FP16
proof-of-useful-work scheme.

Chains the GA100-validated FP16 kernels (``fp16_noisy_quant`` -> ``fp16_policy``)
and the scheme-neutral lottery extractor to reproduce
``zk-pow/src/api/fp16/verify.rs::verify_tile`` / ``verify_tile_proof``
bit-for-bit on real GA100 silicon: rebuild the noised operands, replay and score
the A100 tile, fold it into the keyed-BLAKE3 jackpot ticket, and (for the proof
path) check the difficulty target. See :mod:`pearl_gemm.fp16_pipeline._host`.
"""

from ._host import (
    TileVerify,
    pipeline,
    verify_tile,
    verify_tile_proof,
)
from ._layout import (
    AxisPattern,
    check_jackpot_difficulty,
    compute_jackpot_ticket,
    lane_assignment,
    xor_fold_extract,
)

__all__ = [
    "AxisPattern",
    "TileVerify",
    "check_jackpot_difficulty",
    "compute_jackpot_ticket",
    "lane_assignment",
    "pipeline",
    "verify_tile",
    "verify_tile_proof",
    "xor_fold_extract",
]
