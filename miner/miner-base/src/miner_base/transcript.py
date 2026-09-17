"""The FP8/v4 transcript: domain-separated hashing, keys, and noise seeds.

Twin of ``zk-pow/src/api/fp8/transcript.rs``

    Subkey(q, l)  = BLAKE3(l; key=q)          # keyed when q given, else unkeyed
    H_l(x; q)     = BLAKE3(x; key=Subkey(q, l))

    keyA          = H_"key-A"(sigma_hat)      # A-side tree key (and routing/offset keys)
    keyB          = H_"key-B"(sigma_d)        # B-side tree key
    noise seedB   = H_"seed-B"(HB || keyB || pB)
    noise seedA   = H_"seed-A"(HA || noise seedB || keyA || pA)

For MoE, only the seed-A message is extended:

    noise seedA   = H_"seed-A"(HA || HR || HO || noise seedB || keyA || pA)

The routing root ``HR`` and cumulative-count hash ``HO`` therefore bind routing
before A-side noise is known. Dense and MoE tickets use the same rule:

    J             = H_"jackpot"(z; noise seedA)

The ZK Fiat-Shamir salt (bound as the known-column digest) is the labelled
hash of the proposed header and the public-params encoding:

    statement_digest = H_"zk-public"(sigma_hat || public_data)

All messages are bare concatenations of already-encoded fixed-width fields --
no tags or length prefixes.
"""

from __future__ import annotations

import struct

from blake3 import blake3

# Complete v4 role labels (compile-time strings including the prefix), byte-for
# byte the constants in ``transcript.rs``.
LABEL_PREFIX = b"pearl/v4/FP8/"
LABEL_KEY_A = b"pearl/v4/FP8/key-A"
LABEL_KEY_B = b"pearl/v4/FP8/key-B"
LABEL_SEED_A = b"pearl/v4/FP8/seed-A"
LABEL_SEED_B = b"pearl/v4/FP8/seed-B"
LABEL_NOISE_LINE = b"pearl/v4/FP8/noise-line"
LABEL_JACKPOT = b"pearl/v4/FP8/jackpot"
LABEL_ZK_PUBLIC = b"pearl/v4/FP8/zk-public"


def bits_to_target(nbits: int) -> int:
    """Decode compact ``nBits`` to the verifier's unsigned 256-bit target."""
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0xFFFFFF
    if exponent == 0 or mantissa == 0 or mantissa & 0x800000:
        return 0
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return (mantissa << (8 * (exponent - 3))) & ((1 << 256) - 1)


def encode_u32_le(x: int) -> bytes:
    return struct.pack("<I", x)


def subkey(label: bytes, key: bytes | None = None) -> bytes:
    """Derive a 32-byte role key from a complete v4 label and an optional parent.

    ``Some(parent)`` derives in keyed mode; ``None`` uses unkeyed BLAKE3 (a
    different mode from keyed BLAKE3 under any IV bytes -- there is no fixed
    32-byte protocol key)."""
    assert label.startswith(LABEL_PREFIX), "v4 labels must start with pearl/v4/FP8/"
    return blake3(label, key=key).digest() if key is not None else blake3(label).digest()


def hash_labelled(message: bytes, label: bytes, key: bytes | None = None) -> bytes:
    """``H_l(message; key)``: hash ``message`` under :func:`subkey` of ``label``/``key``."""
    return blake3(message, key=subkey(label, key)).digest()


def commitment_keys(sigma_hat: bytes, sigma_d: bytes | None = None) -> tuple[bytes, bytes]:
    """``(keyA, keyB)`` -- the per-side opening keys from the header window.

    ``keyA = H_"key-A"(sigma_hat)`` keys A's trees (and, in MoE, the routing and
    offset hashes); ``keyB = H_"key-B"(sigma_d)`` keys B's trees, where
    ``sigma_d`` is the depth-``d`` ancestor header. The miner proposes at depth
    0, where ``sigma_d == sigma_hat`` (the default here).
    """
    if sigma_d is None:
        sigma_d = sigma_hat
    return (
        hash_labelled(sigma_hat, LABEL_KEY_A),
        hash_labelled(sigma_d, LABEL_KEY_B),
    )


def noise_seeds(
    hash_a: bytes,
    hash_b: bytes,
    key_a: bytes,
    key_b: bytes,
    p_a: bytes,
    p_b: bytes,
    *,
    hash_routing: bytes | None = None,
    hash_offsets: bytes | None = None,
) -> tuple[bytes, bytes]:
    """``(noise seedA, noise seedB)`` -- the two-stage Fiat-Shamir chain.

    The miner fixes ``B``/``pB`` before it obtains ``noise seedB``, then fixes
    ``A``/``pA`` before it obtains ``noise seedA``. MoE additionally passes
    both routing commitments; supplying only one is invalid.
    """
    if (hash_routing is None) != (hash_offsets is None):
        raise ValueError("MoE seed derivation requires both HR and HO")
    seed_b = hash_labelled(hash_b + key_b + p_b, LABEL_SEED_B)
    routing_commitments = b"" if hash_routing is None else hash_routing + hash_offsets
    seed_a = hash_labelled(
        hash_a + routing_commitments + seed_b + key_a + p_a,
        LABEL_SEED_A,
    )
    return seed_a, seed_b


def jackpot_digest(message: bytes, noise_seed_a: bytes) -> bytes:
    """``J = H_"jackpot"(message; noise seedA)`` for both dense and MoE."""
    return hash_labelled(message, LABEL_JACKPOT, noise_seed_a)


def digest(sigma_hat: bytes, public_data: bytes) -> bytes:
    """``H_"zk-public"(sigma_hat || public_data)`` — the ZK Fiat-Shamir salt.

    Twin of ``PublicParams::digest``. ``sigma_hat`` is the proposed block
    header encoding; ``public_data`` is the public-params encoding. The hash
    must be collision-resistant over admissible statements: the known columns
    are uniquely determined by those public inputs, so hashing the statement
    is a valid binding (it need not be a hash of the columns themselves).
    """
    return hash_labelled(sigma_hat + public_data, LABEL_ZK_PUBLIC)
