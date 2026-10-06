"""Torch-free, bit-exact host reproduction of the FP16 scheme's seed chain.

Reproduces ``zk-pow/src/api/fp16/noise.rs`` + ``plain_proof.rs`` key/seed
derivation on the host, so the miner driver can derive the same jackpot ``pow_key``
the verifier will re-derive from the certificate -- without parsing a proof (the
miner already holds the plaintext operands and commits them itself):

    keyA  = H_"key-A"(proposed_header)                           (A-side tree key)
    keyB  = H_"key-B"(ancestor_header)                           (B-side tree key)
    seedB = H_"seed-B"(root_B || keyB || pB)                     (B then ...)
    seedA = H_"seed-A"(root_A || seedB || keyA || pA)            (... A)
    pow_key = subkey("pearl/v4/FP8/jackpot", seedA)              (jackpot ticket key)

where every ``H_label(msg)`` is a keyed BLAKE3 ``keyed_hash(key=subkey(label), msg)``
and ``subkey(label, parent) = BLAKE3(label, key=parent)`` (unkeyed BLAKE3 of the
label when ``parent`` is ``None``) -- exactly ``pearl_blake3::blake3_digest`` and
``noise.rs::subkey`` / ``hash_labelled``.

``pA`` / ``pB`` are the per-side public-parameter encodings
(``Fp16JobParams::encode_p_a`` / ``encode_p_b``), reproduced byte-for-byte here,
including the committed ``AxisPattern`` wire form (``crate::api::layout``).

Nothing here touches CUDA, so it is importable under any Python and shared by the
standalone sm_80 harness and CI.
"""

from __future__ import annotations

import blake3

# FP16 transcript labels (``noise.rs``): a distinct domain from the FP8 labels so
# the two schemes never derive the same seed from the same inputs. The jackpot
# label is deliberately the shared FP8/v4 one (``noise.rs``: the jackpot key =
# ``subkey("pearl/v4/FP8/jackpot", seed_a)``).
LABEL_KEY_A = b"pearl/v4/FP16/key-A"
LABEL_KEY_B = b"pearl/v4/FP16/key-B"
LABEL_SEED_A = b"pearl/v4/FP16/seed-A"
LABEL_SEED_B = b"pearl/v4/FP16/seed-B"
LABEL_JACKPOT = b"pearl/v4/FP8/jackpot"

# ``crate::api::layout``: 6 fixed dim-byte slots, length split at <= 64, trailing
# slots padded with a length-1 ``DimType::None`` (=3) byte.
_PATTERN_NUM_DIMS = 6
_PATTERN_MAX_DIM_LEN = 64
_PATTERN_PAD_BYTE = 3  # DimType::None

# ``HashId`` discriminants (``crate::api::fp8::public_params::HashId``), keyed by
# the Merkle chunk length.
HASH_ID_FROM_CHUNK_LEN = {128: 0, 256: 1, 512: 2, 1024: 3}


def subkey(label: bytes, parent: bytes | None = None) -> bytes:
    """``blake3_digest(label, parent)`` -- the 32-byte role key.

    Unkeyed BLAKE3 of ``label`` when ``parent`` is ``None``, else keyed BLAKE3 of
    ``label`` under the 32-byte ``parent`` key.
    """
    if parent is None:
        return blake3.blake3(label).digest(length=32)
    if len(parent) != 32:
        raise ValueError(f"parent key must be 32 bytes, got {len(parent)}")
    return blake3.blake3(label, key=parent).digest(length=32)


def _hash_labelled(message: bytes, label: bytes) -> bytes:
    """``H_label(message)`` = ``keyed_hash(key=subkey(label), message)``
    (``noise.rs::hash_labelled``)."""
    return blake3.blake3(message, key=subkey(label)).digest(length=32)


def key_a(proposed_header: bytes) -> bytes:
    """``keyA = H_"key-A"(proposed_header)`` -- the A-side tree/opening key."""
    return _hash_labelled(proposed_header, LABEL_KEY_A)


def key_b(ancestor_header: bytes) -> bytes:
    """``keyB = H_"key-B"(ancestor_header)`` -- the B-side tree/opening key."""
    return _hash_labelled(ancestor_header, LABEL_KEY_B)


def commitment_keys(proposed_header: bytes, ancestor_header: bytes) -> tuple[bytes, bytes]:
    """``(keyA, keyB)`` -- the per-side opening keys (``noise.rs::commitment_keys``)."""
    return key_a(proposed_header), key_b(ancestor_header)


def noise_seeds(
    key_a_bytes: bytes,
    key_b_bytes: bytes,
    root_a: bytes,
    root_b: bytes,
    p_a: bytes,
    p_b: bytes,
) -> tuple[bytes, bytes]:
    """``(seedA, seedB)`` derived B-then-A (``noise.rs::noise_seeds``):

        seedB = H_"seed-B"(root_B || keyB || pB)
        seedA = H_"seed-A"(root_A || seedB || keyA || pA)
    """
    seed_b = _hash_labelled(root_b + key_b_bytes + p_b, LABEL_SEED_B)
    seed_a = _hash_labelled(root_a + seed_b + key_a_bytes + p_a, LABEL_SEED_A)
    return seed_a, seed_b


def jackpot_pow_key(seed_a: bytes) -> bytes:
    """``pow_key = subkey("pearl/v4/FP8/jackpot", seedA)`` -- the key the search
    kernel (and ``compute_jackpot_ticket``) folds the tile under."""
    return subkey(LABEL_JACKPOT, seed_a)


def encode_pattern(pattern) -> bytes:
    """The committed ``AxisPattern`` wire form (``crate::api::layout::encode``):
    exactly 6 bytes, one per split dim ``(length - 1) << 2 | type`` (greedy split
    at the largest divisor <= 64 of the remaining run), trailing slots padded with
    a length-1 ``DimType::None`` byte. ``pattern.dims`` is the canonical
    ``(length, dim_type)`` list (lengths >= 2, no adjacent same-type)."""
    dim_bytes: list[int] = []
    for length, dim_type in pattern.dims:
        rest = int(length)
        while rest > 1:
            part = None
            for d in range(min(_PATTERN_MAX_DIM_LEN, rest), 1, -1):
                if rest % d == 0:
                    part = d
                    break
            if part is None:
                raise ValueError(
                    f"dim length {length} has a prime factor > {_PATTERN_MAX_DIM_LEN} "
                    "and cannot be serialized"
                )
            dim_bytes.append(((part - 1) << 2) | int(dim_type))
            rest //= part
    if len(dim_bytes) > _PATTERN_NUM_DIMS:
        raise ValueError(
            f"pattern needs more than {_PATTERN_NUM_DIMS} dim bytes to serialize"
        )
    out = bytearray([_PATTERN_PAD_BYTE] * _PATTERN_NUM_DIMS)
    out[: len(dim_bytes)] = bytes(dim_bytes)
    return bytes(out)


def encode_p_a(k: int, r: int, device_tag: int, m: int, hash_id_a: int, pattern_a) -> bytes:
    """``pA = k ‖ r ‖ device ‖ m ‖ hash_id_A ‖ P_A``
    (``Fp16JobParams::encode_p_a``). All integers little-endian u32, tags one byte."""
    return b"".join(
        [
            int(k).to_bytes(4, "little"),
            int(r).to_bytes(4, "little"),
            bytes([int(device_tag) & 0xFF]),
            int(m).to_bytes(4, "little"),
            bytes([int(hash_id_a) & 0xFF]),
            encode_pattern(pattern_a),
        ]
    )


def encode_p_b(
    ancestor_header: bytes, k: int, r: int, device_tag: int, n: int, hash_id_b: int, pattern_b
) -> bytes:
    """``pB = σ_Δ ‖ k ‖ r ‖ device ‖ n ‖ hash_id_B ‖ P_B``
    (``Fp16JobParams::encode_p_b``): the ancestor header leads, then the same
    layout as :func:`encode_p_a` for the B side."""
    return b"".join(
        [
            bytes(ancestor_header),
            int(k).to_bytes(4, "little"),
            int(r).to_bytes(4, "little"),
            bytes([int(device_tag) & 0xFF]),
            int(n).to_bytes(4, "little"),
            bytes([int(hash_id_b) & 0xFF]),
            encode_pattern(pattern_b),
        ]
    )
