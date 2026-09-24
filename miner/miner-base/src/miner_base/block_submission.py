"""CPU construction and gateway handoff for certificate-v4 FP8 plain proofs.

GPU and framework adapters extract the committed planes for a winning tile;
this module validates that opening, builds and verifies the consensus
``PlainProofV4``, and submits admissible candidates through the miner RPC
client. The proof opens A's trees under ``keyA`` and B's under ``keyB``, both
derived from the proposed header (the miner proposes at depth 0, so the
ancestor header that keyed B is the proposed header itself).
"""

from dataclasses import dataclass, replace
from typing import Protocol

import pearl_mining
import torch
from blake3 import blake3
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_mining import (
    CERT_VERSION_PLAIN_FP8,
    IncompleteBlockHeader,
    PlainProofV4,
    verify_plain_proof_for_cert_version,
)

from .commitment import (
    BlockHeader,
    Device,
    HashId,
    MatrixMerkleProof,
    MiningConfiguration,
    OperandParams,
    PlanarCommitment,
    commit_planes,
    commitment_keys,
    hash_id_for_leaf,
)
from .layout import AxisPattern
from .mining_config import COMMITMENT_CHUNK_SIZE
from .prequant import DEFAULT_BLOCK_SIZE

# The pearl_mining binding currently exposes only ``(bool, str)``. Keep this
# exact consensus-policy message narrow until the binding exposes an error code.
_JACKPOT_INADMISSIBLE = "The jackpot is not admissible"

# Merkle leaf the GPU commit and the CPU proof chain open by default. Aliased
# here so winner handoff, B-proof rebuild, and ``MiningConfiguration.chunk_size``
# share one definition.
PROTOCOL_COMMITMENT_LEAF = COMMITMENT_CHUNK_SIZE


def commit_planes_for_leaf(
    planes: list[torch.Tensor], key: bytes, leaf: int = PROTOCOL_COMMITMENT_LEAF
) -> PlanarCommitment:
    """The CPU commitment for planes the GPU committed at ``leaf`` under ``key``."""
    return commit_planes(planes, key, hash_id_for_leaf(leaf))


class PlainProofClient(Protocol):
    def submit_plain_proof(self, plain_proof: PlainProofV4, mining_job: MiningJob) -> None: ...


@dataclass(frozen=True)
class PrebuiltCommitment:
    """A planar commitment paired with the side's opening key it was built under."""

    commitment: PlanarCommitment
    key: bytes


@dataclass(frozen=True)
class OpenedBlockInfo:
    """CPU-resident committed planes and indices for one winning tile."""

    a_row_indices: tuple[int, ...] | list[int]
    b_column_indices: tuple[int, ...] | list[int]
    a_codes: torch.Tensor  # (m, k) int8
    a_scales: torch.Tensor  # (m, k/8) bfloat16
    b_codes: torch.Tensor  # (n, k) int8
    b_scales: torch.Tensor  # (n, k/8) bfloat16
    mining_config: MiningConfiguration
    b_commitment: PrebuiltCommitment | None = None
    a_commitment: PrebuiltCommitment | None = None

    def owned_copy(self) -> "OpenedBlockInfo":
        """Validate and snapshot mutable inputs for an asynchronous handoff."""
        _validate_opening(self)
        prebuilt_a = self.a_commitment
        prebuilt_b = self.b_commitment
        return OpenedBlockInfo(
            a_row_indices=tuple(self.a_row_indices),
            b_column_indices=tuple(self.b_column_indices),
            # A prebuilt tree was built from these exact owned hit planes. Keep
            # that immutable ownership bundle together instead of cloning the
            # tensors away from the tree that will open them.
            a_codes=self.a_codes if prebuilt_a is not None else self.a_codes.clone(),
            a_scales=self.a_scales if prebuilt_a is not None else self.a_scales.clone(),
            # A prebuilt tree owns the B opening. Keep its stable plane metadata
            # by reference instead of cloning a weights-sized operand solely
            # for asynchronous proof construction.
            b_codes=self.b_codes if prebuilt_b is not None else self.b_codes.clone(),
            b_scales=self.b_scales if prebuilt_b is not None else self.b_scales.clone(),
            mining_config=replace(self.mining_config),
            b_commitment=prebuilt_b,
            a_commitment=prebuilt_a,
        )


def is_plain_fp8_job(job: MiningJob) -> bool:
    """Whether the issuing endpoint advertised certificate-v4 FP8 through ``job``."""
    return int(job.cert_version) == CERT_VERSION_PLAIN_FP8


def _validate_plane(plane: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if not isinstance(plane, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(plane).__name__}")
    if plane.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(plane.shape)}")
    if plane.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got device {plane.device}")
    if plane.dtype is not dtype:
        raise ValueError(f"{name} must be {dtype}, got {plane.dtype}")


def _validate_operand(
    codes: torch.Tensor,
    scales: torch.Tensor,
    codes_name: str,
    scales_name: str,
) -> None:
    _validate_plane(codes, codes_name, torch.int8)
    _validate_plane(scales, scales_name, torch.bfloat16)
    rows, cols = codes.shape
    if cols % DEFAULT_BLOCK_SIZE:
        raise ValueError(
            f"{codes_name} columns ({cols}) must be divisible by the block size "
            f"{DEFAULT_BLOCK_SIZE}"
        )
    expected = (rows, cols // DEFAULT_BLOCK_SIZE)
    if tuple(scales.shape) != expected:
        raise ValueError(
            f"{scales_name} must be {expected} for {codes_name} {(rows, cols)}, "
            f"got {tuple(scales.shape)}"
        )


def _validate_planes(name: str, codes: torch.Tensor, scales: torch.Tensor) -> tuple[int, int]:
    _validate_operand(codes, scales, f"{name.lower()}_codes", f"{name.lower()}_scales")
    rows, k = codes.shape
    if rows <= 0 or k <= 0:
        raise ValueError(f"{name} codes shape must be positive, got {codes.shape}")
    if not codes.is_contiguous() or not scales.is_contiguous():
        raise ValueError(f"{name} committed planes must be contiguous")
    return rows, k


def _validate_indices(
    name: str,
    indices: tuple[int, ...] | list[int],
    row_count: int,
    pattern: AxisPattern,
) -> None:
    normalized = tuple(indices)
    if any(type(index) is not int for index in normalized):
        raise TypeError(f"{name} must contain integers")
    offsets = tuple(pattern.tile_offsets)
    if len(normalized) != len(offsets):
        raise ValueError(
            f"{name} must select exactly one {len(offsets)}-row lottery tile, "
            f"got {len(normalized)} rows"
        )
    if any(index < 0 or index >= row_count for index in normalized):
        raise ValueError(f"{name} contains an index outside [0, {row_count})")
    base = normalized[0] - offsets[0]
    expected = tuple(base + offset for offset in offsets)
    if base < 0 or not pattern.offset_is_valid(base) or normalized != expected:
        raise ValueError(f"{name} does not match one committed lottery tile")


def _validate_opening(opened_block_info: OpenedBlockInfo) -> tuple[int, int, int]:
    m, k = _validate_planes("A", opened_block_info.a_codes, opened_block_info.a_scales)
    n, b_k = _validate_planes("B", opened_block_info.b_codes, opened_block_info.b_scales)
    if k != b_k:
        raise ValueError(f"common dimension mismatch: A has k={k}, B has k={b_k}")

    config = opened_block_info.mining_config
    if k != config.common_dim:
        raise ValueError(f"opening k={k} does not match mining configuration k={config.common_dim}")
    _validate_indices("a_row_indices", opened_block_info.a_row_indices, m, config.rows_pattern)
    _validate_indices(
        "b_column_indices", opened_block_info.b_column_indices, n, config.cols_pattern
    )
    return m, n, k


def _checked_commitment(
    prebuilt: PrebuiltCommitment,
    expected_key: bytes,
    planes: tuple[torch.Tensor, ...],
    hash_id: HashId,
) -> PlanarCommitment:
    """Validate a prebuilt commitment without rehashing its operand planes."""
    if prebuilt.key != expected_key:
        raise ValueError("prebuilt commitment belongs to another job")

    commitment = prebuilt.commitment
    if len(commitment.parts) != len(planes):
        raise ValueError(
            f"prebuilt commitment has {len(commitment.parts)} planes, expected {len(planes)}"
        )
    for index, (part, plane) in enumerate(zip(commitment.parts, planes, strict=True)):
        row_nbytes = plane.shape[1] * plane.element_size()
        if (part.rows, part.row_nbytes) != (plane.shape[0], row_nbytes):
            raise ValueError(
                f"prebuilt commitment plane {index} covers "
                f"{part.rows}x{part.row_nbytes}B rows, but the supplied plane is "
                f"{plane.shape[0]}x{row_nbytes}B"
            )
        if part.hash_id is not hash_id:
            raise ValueError(
                f"prebuilt commitment plane {index} uses leaf {part.hash_id.chunk_len}, "
                f"the configuration commits {hash_id.chunk_len}"
            )
        if part.digest != bytes(part.tree.root):
            raise ValueError(f"prebuilt commitment plane {index} digest does not match its tree")
    expected_digest = blake3(
        b"".join(part.digest for part in commitment.parts), key=expected_key
    ).digest()
    if commitment.digest != expected_digest:
        raise ValueError("prebuilt commitment digest does not match its own planes")
    return commitment


_NATIVE_DEVICE = {
    Device.BLACKWELL: pearl_mining.Device.B200,
}
_NATIVE_HASH_ID = {
    HashId.BLAKE3_CHUNK_128: pearl_mining.HashId.Blake3Chunk128,
    HashId.BLAKE3_CHUNK_256: pearl_mining.HashId.Blake3Chunk256,
    HashId.BLAKE3_CHUNK_512: pearl_mining.HashId.Blake3Chunk512,
    HashId.BLAKE3_CHUNK_1024: pearl_mining.HashId.Blake3Chunk1024,
}


def _native_matrix_proof(opening: MatrixMerkleProof) -> pearl_mining.MatrixMerkleProof:
    """The reference opening (a ``pearl_blake3`` proof) as the verifier's own type."""
    proof = opening.proof
    native = pearl_mining.MerkleProof(
        [bytes(leaf) for leaf in proof.leaf_data],
        list(proof.leaf_indices),
        bytes(proof.root),
        [bytes(sibling) for sibling in proof.siblings],
        proof.total_leaves,
    )
    return pearl_mining.MatrixMerkleProof(native, list(opening.row_indices))


def _native_operand(params: OperandParams) -> pearl_mining.OperandParams:
    return pearl_mining.OperandParams(
        params.num_rows,
        _NATIVE_HASH_ID[params.hash_id],
        pearl_mining.AxisPattern.from_bytes(bytes(params.pattern.to_bytes())),
    )


def create_proof(
    opened_block_info: OpenedBlockInfo, header: BlockHeader | IncompleteBlockHeader
) -> PlainProofV4:
    """Build a certificate-v4 FP8 ``PlainProofV4`` from a validated winning opening.

    ``header`` is the proposed (incomplete) block header; it keys both sides'
    trees since the miner proposes at ancestor depth 0.
    """
    m, n, k = _validate_opening(opened_block_info)
    config = opened_block_info.mining_config
    header_bytes = bytes(header.to_bytes())
    key_a, key_b = commitment_keys(header_bytes)
    a_planes = (opened_block_info.a_codes, opened_block_info.a_scales)
    if opened_block_info.a_commitment is None:
        comm_a = commit_planes(list(a_planes), key_a, config.a_hash_id)
    else:
        comm_a = _checked_commitment(
            opened_block_info.a_commitment, key_a, a_planes, config.a_hash_id
        )
    b_planes = (opened_block_info.b_codes, opened_block_info.b_scales)
    if opened_block_info.b_commitment is None:
        comm_b = commit_planes(list(b_planes), key_b, config.b_hash_id)
    else:
        comm_b = _checked_commitment(
            opened_block_info.b_commitment, key_b, b_planes, config.b_hash_id
        )
    a_values, a_scales = comm_a.open(list(opened_block_info.a_row_indices))
    bt_values, bt_scales = comm_b.open(list(opened_block_info.b_column_indices))

    common = config.common_params()
    return PlainProofV4(
        ancestor_header=IncompleteBlockHeader.from_bytes(header_bytes),
        common=pearl_mining.CommonParams(
            common.k,
            common.r,
            pearl_mining.Quant.Fp8E4M3Prequant,
            _NATIVE_DEVICE[common.device],
        ),
        a=_native_operand(config.a_params(m)),
        b=_native_operand(config.b_params(n)),
        values_a=_native_matrix_proof(a_values),
        values_b=_native_matrix_proof(bt_values),
        scales_a=_native_matrix_proof(a_scales),
        scales_b=_native_matrix_proof(bt_scales),
    )


def submit_opened_block(
    opened_block_info: OpenedBlockInfo,
    mining_job: MiningJob,
    client: PlainProofClient,
) -> PlainProofV4 | None:
    """Verify and hand one certificate-v4 FP8 proof to the gateway.

    ``None`` means the lottery winner failed the independent jackpot policy and
    was filtered normally. Otherwise, completion means the gateway accepted the
    JSON-RPC request into its own submission queue; downstream acceptance is a
    separate gateway-owned lifecycle.
    """
    if not is_plain_fp8_job(mining_job):
        raise ValueError(
            "FP8 proof submission requires certificate version "
            f"{CERT_VERSION_PLAIN_FP8}, got {int(mining_job.cert_version)}"
        )

    header = IncompleteBlockHeader.from_bytes(mining_job.incomplete_header_bytes)
    plain_proof = create_proof(opened_block_info, header)
    is_valid, message = verify_plain_proof_for_cert_version(
        mining_job.cert_version, header, plain_proof
    )
    if not is_valid:
        if message == _JACKPOT_INADMISSIBLE:
            return None
        raise RuntimeError(f"Plain proof verification failed: {message}")

    client.submit_plain_proof(plain_proof, mining_job)
    return plain_proof


__all__ = [
    "OpenedBlockInfo",
    "PrebuiltCommitment",
    "PlainProofClient",
    "create_proof",
    "is_plain_fp8_job",
    "submit_opened_block",
]
