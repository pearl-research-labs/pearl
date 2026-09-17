"""Finalize the A-side v4 chain on the caller's CUDA stream.

From the two device-produced plane roots down to everything the mining
launches consume, in five compressions of one thread
(``miner_base.commitment_hash``):

    HA            = blake3(root_codes || root_scales, key=keyA)          (64 B, keyed)
    noise seedA   = blake3(HA || seedB || keyA || pA, key=Subkey("seed-A"))  (107 B: 2 blocks)
    noise-line kA = blake3("pearl/v4/FP8/noise-line", key=seedA)       (23 B, keyed)
    jackpot key   = blake3("pearl/v4/FP8/jackpot",    key=seedA)       (20 B, keyed)

``pA = u32(m) || u8(hash_id) || pattern(6)`` is 11 bytes and a per-shape
compile-time constant; ``seedB`` and ``keyA`` are device words (per B and per
header). The output is the 96-byte ``a_keys = seedA || noise-line keyA ||
jackpot key`` (``miner_base.commitment_hash.AKeys``): word ``[8, 16)`` keys
``noisy_quant``'s E1 draw and the ``F_A`` ``noise_lines`` draw, word ``[16,
24)`` is ``mixed_gemm``'s ``pow_key``.
"""

import struct

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
from blake3 import blake3

from ._blake3 import CHUNK_END, CHUNK_START, KEYED_HASH, ROOT
from ._blake3_ops import SINGLE_BLOCK_KEYED_FLAGS, compress

# v4 transcript labels (``miner_base.commitment``). ``Subkey(label)``
# of an unkeyed label is its plain BLAKE3 digest.
LABEL_SEED_A = b"pearl/v4/FP8/seed-A"
LABEL_NOISE_LINE = b"pearl/v4/FP8/noise-line"
LABEL_JACKPOT = b"pearl/v4/FP8/jackpot"

P_A_BYTES = 11  # u32(m) | u8(hash_id) | pattern(6): the dense pA
_P_A_WORDS = (P_A_BYTES + 3) // 4
_SEED_A_MSG_BYTES = 32 * 3 + P_A_BYTES  # HA || seedB || keyA || pA
_SEED_A_BLOCK1_BYTES = _SEED_A_MSG_BYTES - 64
A_KEYS_BYTES = 96


def _words(data: bytes) -> tuple[int, ...]:
    """A <= 64-byte message as its 16 zero-padded little-endian words."""
    assert len(data) <= 64
    return struct.unpack("<16I", data.ljust(64, b"\x00"))


_SEED_A_KEY = _words(blake3(LABEL_SEED_A).digest())[:8]
_NOISE_LINE_LABEL = _words(LABEL_NOISE_LINE)
_JACKPOT_LABEL = _words(LABEL_JACKPOT)


def p_a_words(p_a: bytes) -> tuple[int, ...]:
    """``pA`` as the constexpr words the finalize splices into seedA's second block."""
    if not isinstance(p_a, bytes) or len(p_a) != P_A_BYTES:
        raise ValueError(f"p_a must be the {P_A_BYTES}-byte dense pA encoding")
    return struct.unpack(f"<{_P_A_WORDS}I", p_a.ljust(4 * _P_A_WORDS, b"\x00"))


def _const(words) -> list:
    return [cutlass.Uint32(w) for w in words]


@cute.kernel
def _finalize_kernel(
    gRootCodes: cute.Tensor,  # (8,) u32: merkle root of the codes plane
    gRootScales: cute.Tensor,  # (8,) u32: merkle root of the scales plane
    gKeyA: cute.Tensor,  # (8,) u32: keyA (the header's A-side opening key)
    gSeedB: cute.Tensor,  # (8,) u32: noise seedB
    gOut: cute.Tensor,  # (24,) u32: seedA || noise-line keyA || jackpot key
    p_a: tuple,
):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        key_a = [gKeyA[i] for i in range(8)]
        roots = [gRootCodes[i] for i in range(8)] + [gRootScales[i] for i in range(8)]
        hash_a = compress(key_a, roots, 64, SINGLE_BLOCK_KEYED_FLAGS)

        # seedA: a two-block single chunk under the "seed-A" subkey. Block 0
        # (HA || seedB) opens the chunk, block 1 (keyA || pA, 43 bytes) closes
        # it as the root.
        block0 = list(hash_a) + [gSeedB[i] for i in range(8)]
        cv = compress(_const(_SEED_A_KEY), block0, 64, CHUNK_START | KEYED_HASH)
        block1 = key_a + _const(p_a) + _const([0] * (8 - _P_A_WORDS))
        seed_a = compress(cv, block1, _SEED_A_BLOCK1_BYTES, CHUNK_END | ROOT | KEYED_HASH)

        noise_key = compress(
            seed_a, _const(_NOISE_LINE_LABEL), len(LABEL_NOISE_LINE), SINGLE_BLOCK_KEYED_FLAGS
        )
        jackpot_key = compress(
            seed_a, _const(_JACKPOT_LABEL), len(LABEL_JACKPOT), SINGLE_BLOCK_KEYED_FLAGS
        )
        for i in cutlass.range_constexpr(8):
            gOut[i] = seed_a[i]
            gOut[8 + i] = noise_key[i]
            gOut[16 + i] = jackpot_key[i]


@cute.jit
def _finalize_launch(
    gRootCodes: cute.Tensor,
    gRootScales: cute.Tensor,
    gKeyA: cute.Tensor,
    gSeedB: cute.Tensor,
    gOut: cute.Tensor,
    stream: cuda_drv.CUstream,
    p_a: cutlass.Constexpr,
):
    _finalize_kernel(gRootCodes, gRootScales, gKeyA, gSeedB, gOut, p_a).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
        stream=stream,
    )
