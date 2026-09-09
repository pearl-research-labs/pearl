"""The v4 commitment chain over device- or host-produced Merkle roots.

``pearl_gemm.tensor_hash_plus_stats`` hashes a committed operand's planes on
the GPU and returns one keyed Merkle root per plane. These are the host-side
combines that turn those roots into the per-side noise seeds and the lottery's
jackpot key, so a miner never builds the CPU Merkle tree just to learn a seed
(the opening tree is only needed for a proof and stays off the mining path).

    HX               = blake3(root(values_X) || root(scales_X), key=keyX)
    noise seedB      = H_"seed-B"(HB || keyB || pB)
    noise seedA      = H_"seed-A"(HA || seedB || keyA || pA)
    noise-line keyX  = Subkey("noise-line", seedX)      (the line generator's key)
    jackpot key      = Subkey("jackpot",    seedA)      (the winning test's key)

``HX`` is exactly ``commit_planes(planes, keyX, hash_id).digest``; the seeds
are ``miner_base.transcript.noise_seeds``. The tensor-hash finalize
kernel computes the A-side chain on the device from the same inputs
(:func:`a_keys`); this module is its host twin and the B-side's only
implementation.
"""

from collections.abc import Sequence
from typing import NamedTuple

from blake3 import blake3

from .commitment import (
    LABEL_JACKPOT,
    LABEL_NOISE_LINE,
    LABEL_SEED_A,
    LABEL_SEED_B,
    hash_labelled,
    subkey,
)


def operand_digest(roots: Sequence[bytes], key: bytes) -> bytes:
    """``HX``: the keyed digest of one operand's plane roots (values, scales)."""
    return blake3(b"".join(roots), key=key).digest()


def noise_seed_b(hash_b: bytes, key_b: bytes, p_b: bytes) -> bytes:
    """``noise seedB = H_"seed-B"(HB || keyB || pB)``."""
    return hash_labelled(hash_b + key_b + p_b, LABEL_SEED_B)


def noise_seed_a(hash_a: bytes, seed_b: bytes, key_a: bytes, p_a: bytes) -> bytes:
    """``noise seedA = H_"seed-A"(HA || seedB || keyA || pA)`` (dense: no routing)."""
    return hash_labelled(hash_a + seed_b + key_a + p_a, LABEL_SEED_A)


def noise_line_key(seed: bytes) -> bytes:
    """The keyed-BLAKE3 key one side's noise lines are drawn under."""
    return subkey(LABEL_NOISE_LINE, seed)


def jackpot_key(seed_a: bytes) -> bytes:
    """The key of ``J = H_"jackpot"(extracted; noise seedA)``: the lottery's PoW key."""
    return subkey(LABEL_JACKPOT, seed_a)


class AKeys(NamedTuple):
    """Everything downstream of ``HA``: the finalize kernel's 96-byte output."""

    seed_a: bytes
    noise_line_key: bytes
    jackpot_key: bytes

    @classmethod
    def from_bytes(cls, raw: bytes) -> "AKeys":
        if len(raw) != 96:
            raise ValueError(f"a_keys is 96 bytes, got {len(raw)}")
        return cls(bytes(raw[0:32]), bytes(raw[32:64]), bytes(raw[64:96]))

    def to_bytes(self) -> bytes:
        return self.seed_a + self.noise_line_key + self.jackpot_key


def a_keys(hash_a: bytes, seed_b: bytes, key_a: bytes, p_a: bytes) -> AKeys:
    """The A-side chain from ``HA`` down: ``(seedA, noise-line keyA, jackpot key)``."""
    seed_a = noise_seed_a(hash_a, seed_b, key_a, p_a)
    return AKeys(seed_a, noise_line_key(seed_a), jackpot_key(seed_a))
