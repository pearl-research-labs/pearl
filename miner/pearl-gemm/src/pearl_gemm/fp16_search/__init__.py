"""Bit-exact A100 (``sm_80``) full-matrix FP16 lottery search for the FP16
proof-of-useful-work scheme.

The FP16 analogue of the lottery fused inside the FP8 ``mixed_gemm``: scan every
committed tile of a noised matmul ``A' @ B'^T`` on real GA100 silicon and latch
the FIRST tile (lowest flat tile index) whose keyed-BLAKE3 jackpot ticket clears
the difficulty threshold. The per-tile ``a100_dot`` accumulation, the 16-lane
XOR-fold, the keyed-BLAKE3 ticket, and the 256-bit difficulty compare are all
bit-for-bit with the verifier (reusing the GA100-validated ``fp16_policy`` /
``fp16_commit`` math). Policy is not evaluated here -- the host driver runs
``fp16_policy`` on the latched tile before submission. See
:mod:`pearl_gemm.fp16_search._host`.
"""

from ._host import SearchHit, difficulty_bound, search

__all__ = [
    "SearchHit",
    "difficulty_bound",
    "search",
]
