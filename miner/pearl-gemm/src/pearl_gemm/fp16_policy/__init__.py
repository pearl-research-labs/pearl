"""Bit-exact A100 (``sm_80``) FP16 "unpredictable accumulation steps" policy.

Replays the FP16 proof-of-useful-work tile on real GA100 tensor cores with the
same bit-exact accumulation as :mod:`pearl_gemm.fp16_gemm`, additionally emitting
the per-group policy census and folding it into the tile-global ``f_bp`` / ``rho``
/ ``accept`` report -- bit-for-bit against the verifier's
``zk-pow/src/api/fp16/policy.rs``. ``fp16_gemm`` is left untouched; this is a
standalone, census-capable module.
"""

from ._host import (
    PolicyReport,
    evaluate,
    policy_census,
    replay_and_evaluate,
)

__all__ = ["PolicyReport", "evaluate", "policy_census", "replay_and_evaluate"]
