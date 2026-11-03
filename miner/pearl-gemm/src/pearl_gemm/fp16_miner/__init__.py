"""Host-side FP16 (A100 / ``sm_80``) lottery-search driver for the FP16
proof-of-useful-work scheme.

The standalone analogue of the FP8 ``_launch_stages``: from a job header and the
full plaintext FP16 operands, it derives the bit-exact seed chain
(``zk-pow/src/api/fp16/noise.rs`` + ``plain_proof.rs``), commits both operands,
chains the GA100-validated FP16 sm_80 kernels (``fp16_commit`` ->
``fp16_noise_lines`` -> ``fp16_noisy_quant`` -> ``fp16_search``), runs the
full-matrix lottery search, and decodes the latched hit into a verifiable winning
tile (opened rows + sliced noise + ``seed_a`` + ticket + host-side policy report).

The seed-chain host reproduction (:mod:`pearl_gemm.fp16_miner._seed_chain`) is
torch-free and bit-for-bit against the Rust reference.
"""

from ._host import (
    Fp16JobParams,
    Fp16OperandParams,
    SeedChain,
    WinningTile,
    derive_seed_chain,
    difficulty_bound,
    search_block,
)
from ._seed_chain import (
    commitment_keys,
    encode_p_a,
    encode_p_b,
    encode_pattern,
    jackpot_pow_key,
    key_a,
    key_b,
    noise_seeds,
    subkey,
)

__all__ = [
    "Fp16JobParams",
    "Fp16OperandParams",
    "SeedChain",
    "WinningTile",
    "commitment_keys",
    "derive_seed_chain",
    "difficulty_bound",
    "encode_p_a",
    "encode_p_b",
    "encode_pattern",
    "jackpot_pow_key",
    "key_a",
    "key_b",
    "noise_seeds",
    "search_block",
    "subkey",
]
