"""Keyed Merkle commitments and openings for FP8/v4 operands.

v4 commitments: every committed tree is keyed with its side's
opening key (``keyA`` / ``keyB``, :func:`~.transcript.commitment_keys`), and a
prequantized operand's two plane roots combine into the side's aggregate
digest ``HA`` / ``HB`` (``operand_digest_fp10`` in
``zk-pow/src/api/proof_utils.rs``):

    HA / HB = blake3(keyed_merkle_root(int8 values) || keyed_merkle_root(BF16 scales), key=keyA/keyB)

Shapes are bound by the public statement, not the digests. The selected
:class:`~.params.HashId` controls each tree's leaf size. For MoE, the flat
routing table commits under ``keyA`` (root ``HR``), while cumulative counts
are padded according to their own hash ID before hashing to ``HO``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from blake3 import blake3
from pearl_blake3 import MerkleProof, MerkleTree

from .layout import AxisPattern
from .params import (
    BlockHeader,
    CommonParams,
    Device,
    HashId,
    OperandParams,
    Quant,
    encode_p_a,
    encode_p_b,
    validate_lottery_layout,
)
from .transcript import (
    LABEL_JACKPOT,
    LABEL_KEY_A,
    LABEL_KEY_B,
    LABEL_NOISE_LINE,
    LABEL_SEED_A,
    LABEL_SEED_B,
    bits_to_target,
    commitment_keys,
    encode_u32_le,
    hash_labelled,
    jackpot_digest,
    noise_seeds,
    subkey,
)

__all__ = [
    "BlockHeader",
    "CommonParams",
    "Device",
    "HashId",
    "LABEL_JACKPOT",
    "LABEL_KEY_A",
    "LABEL_KEY_B",
    "LABEL_NOISE_LINE",
    "LABEL_SEED_A",
    "LABEL_SEED_B",
    "MatrixCommitment",
    "MatrixMerkleProof",
    "MiningConfiguration",
    "OperandParams",
    "PROTOCOL_RANK",
    "PlanarCommitment",
    "Quant",
    "bits_to_target",
    "commit_matrix",
    "commit_planes",
    "commitment_keys",
    "encode_p_a",
    "encode_p_b",
    "encode_u32_le",
    "hash_id_for_leaf",
    "hash_labelled",
    "jackpot_digest",
    "noise_seeds",
    "subkey",
]


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().flatten().view(torch.uint8).numpy().tobytes()


@dataclass(frozen=True)
class MatrixMerkleProof:
    """Merkle opening plus the matrix rows it authenticates."""

    proof: MerkleProof
    row_indices: list[int]

    @property
    def root(self) -> bytes:
        return bytes(self.proof.root)


@dataclass
class MatrixCommitment:
    """A committed 2D tensor: its keyed Merkle root plus the tree for openings."""

    tree: MerkleTree
    digest: bytes  # the keyed merkle_root
    rows: int
    row_nbytes: int
    hash_id: HashId

    def open(self, row_indices: list[int]) -> MatrixMerkleProof:
        leaf_idx = MerkleTree.compute_leaf_indices_from_rows(
            list(row_indices),
            (self.rows, self.row_nbytes),
            self.hash_id.chunk_len,
        )
        proof = self.tree.get_multileaf_proof(leaf_idx)
        return MatrixMerkleProof(proof, list(row_indices))


def commit_matrix(t: torch.Tensor, key: bytes, hash_id: HashId) -> MatrixCommitment:
    """Keyed chunk-Merkle commitment of one plane; its digest is the tree root.

    ``key`` is the side's opening key (``keyA`` for A/routing, ``keyB`` for B).
    """
    assert t.dim() == 2
    row_nbytes = t.shape[1] * t.element_size()
    tree = MerkleTree(
        data=hash_id.pad(_tensor_bytes(t)),
        key=key,
        chunk_len=hash_id.chunk_len,
    )
    return MatrixCommitment(tree, bytes(tree.root), t.shape[0], row_nbytes, hash_id)


@dataclass
class PlanarCommitment:
    """A committed operand made of several tensors (e.g. int8 values + BF16 scales):
    ``digest = blake3(root(int_values) || root(scales), key=key)`` -- the side's
    aggregate digest ``HA`` / ``HB`` (``operand_digest_fp10``), keyed by the
    side's opening key (``keyA`` / ``keyB``).
    Openings reveal the same row indices in every part.
    """

    parts: list[MatrixCommitment]
    digest: bytes

    def open(self, row_indices: list[int]) -> list[MatrixMerkleProof]:
        return [part.open(row_indices) for part in self.parts]


def commit_planes(
    planes: list[torch.Tensor],
    key: bytes,
    hash_id: HashId,
) -> PlanarCommitment:
    """Commit each tensor separately under ``key``, then combine the digests."""
    assert len(planes) >= 1
    rows = planes[0].shape[0]
    assert all(p.shape[0] == rows for p in planes), "planes must share the row count"
    parts = [commit_matrix(p, key, hash_id) for p in planes]
    digest = blake3(b"".join(part.digest for part in parts), key=key).digest()
    return PlanarCommitment(parts, digest)


def commit_routing(
    routing_tokens: list[int],
    key_a: bytes,
    hash_id: HashId,
) -> MerkleTree:
    """Commit the MoE flat routing table ``Rflat``: a keyed chunk tree over
    the little-endian u32 token indices, under ``keyA``. Its root is ``HR``."""
    raw = b"".join(encode_u32_le(token) for token in routing_tokens)
    return MerkleTree(
        data=hash_id.pad(raw),
        key=key_a,
        chunk_len=hash_id.chunk_len,
    )


def hash_offsets(
    end_offsets: list[int],
    key_a: bytes,
    hash_id: HashId,
) -> bytes:
    """``HO``: the keyed chunk-tree root of the MoE cumulative routing counts ``O``.

    The u32-LE encoding of ``O`` is zero-padded to the ``hash_idO`` chunk
    granularity first. A list that fits one chunk roots to the flat keyed
    BLAKE3 digest of the padded bytes; larger lists build the chunk tree
    (twin: ``offsets_root`` in ``zk-pow/src/api/fp8/plain_proof.rs``).
    """
    raw = b"".join(encode_u32_le(offset) for offset in end_offsets)
    tree = MerkleTree(
        data=hash_id.pad(raw),
        key=key_a,
        chunk_len=hash_id.chunk_len,
    )
    return bytes(tree.root)


# The verifier's whitelisted keyed-BLAKE3 Merkle leaf sizes.
_HASH_ID_FOR_LEAF = {hash_id.chunk_len: hash_id for hash_id in HashId}

# The only with-peel rank the cert-v4 verifier admits (``zk-pow``
# ``public_params``: "Rank must be exactly 32"); ``pearl_gemm.protocol_constants.R``
# is specialized to the same value.
PROTOCOL_RANK = 32


def hash_id_for_leaf(chunk_size: int) -> HashId:
    """The committed :class:`HashId` whose leaf is ``chunk_size`` bytes."""
    try:
        return _HASH_ID_FOR_LEAF[chunk_size]
    except KeyError:
        raise ValueError(
            f"no committed Merkle leaf of {chunk_size} bytes; "
            f"cert-v4 opens {sorted(_HASH_ID_FOR_LEAF)}"
        ) from None


@dataclass
class MiningConfiguration:
    """The parameters one layer's launches and proofs commit to.

    ``device`` and the with-peel ``rank`` (r) sit alongside one committed
    :class:`AxisPattern` per axis (the lottery tile), and one Merkle leaf per
    committed tree: ``chunk_size`` for the weight (B) planes and
    ``a_chunk_size`` (defaulting to it) for the activation (A) planes. Both
    leaves are protocol-visible through the operands' ``HashId``.
    """

    device: Device
    common_dim: int  # k
    rank: int  # r (noise rank / peel width)
    rows_pattern: AxisPattern
    cols_pattern: AxisPattern
    chunk_size: int = HashId.BLAKE3_CHUNK_1024.chunk_len
    a_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.a_chunk_size is None:
            self.a_chunk_size = self.chunk_size
        if self.rank != PROTOCOL_RANK:
            raise ValueError(f"cert-v4 mines at rank {PROTOCOL_RANK} only, got rank={self.rank}")
        validate_lottery_layout(self.a_params(1), self.b_params(1))

    @property
    def a_hash_id(self) -> HashId:
        assert self.a_chunk_size is not None
        return hash_id_for_leaf(self.a_chunk_size)

    @property
    def b_hash_id(self) -> HashId:
        return hash_id_for_leaf(self.chunk_size)

    def common_params(self) -> CommonParams:
        """``pB``'s shared fields: ``(k, r, quant, device)``."""
        return CommonParams(self.common_dim, self.rank, Quant.FP8_E4M3_PREQUANT, self.device)

    def a_params(self, m: int) -> OperandParams:
        """The A operand's committed ``(m, hash_idA, Prow)``."""
        return OperandParams(m, self.a_hash_id, self.rows_pattern)

    def b_params(self, n: int) -> OperandParams:
        """The B operand's committed ``(n, hash_idB, Pcol)``."""
        return OperandParams(n, self.b_hash_id, self.cols_pattern)

    def p_a(self, m: int) -> bytes:
        """The encoded ``pA`` bound into ``noise seedA``."""
        return encode_p_a(self.a_params(m))

    def p_b(self, n: int) -> bytes:
        """The encoded ``pB`` bound into ``noise seedB``."""
        return encode_p_b(self.common_params(), self.b_params(n))
