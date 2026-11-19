"""CPU construction, local verification, and gateway handoff for cert-v4 FP8 plain proofs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import pearl_mining
import torch
from blake3 import blake3
from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_mining import (
    CERT_VERSION_PLAIN_FP8,
    IncompleteBlockHeader,
    PlainProofV4,
    diagnose_jackpot_policy_v4,
    verify_plain_proof_for_cert_version,
)

from .commitment import (
    MatrixMerkleProof,
    PlanarCommitment,
    commit_planes,
    commit_routing,
    hash_offsets,
    routing_leaf_indices,
    tensor_bytes,
)
from .prequant import DEFAULT_BLOCK_SIZE
from .protocol import (
    Device,
    HashId,
    MiningConfiguration,
    OperandParams,
    commitment_keys,
)

_LOGGER = get_logger(__name__)

_JACKPOT_INADMISSIBLE = "The jackpot is not admissible"

# Detailed rejection logs are capped for the life of the process so repeated
# rejections cannot flood logs. One warning marks the cap.
_JACKPOT_LOG_LIMIT = 32
_jackpot_logs_emitted = 0
_jackpot_log_cap_warned = False

_NATIVE_DEVICE = {
    Device.HOPPER: pearl_mining.Device.H100,
    Device.BLACKWELL: pearl_mining.Device.B200,
}
_NATIVE_HASH_ID = {
    HashId.BLAKE3_CHUNK_128: pearl_mining.HashId.Blake3Chunk128,
    HashId.BLAKE3_CHUNK_256: pearl_mining.HashId.Blake3Chunk256,
    HashId.BLAKE3_CHUNK_512: pearl_mining.HashId.Blake3Chunk512,
    HashId.BLAKE3_CHUNK_1024: pearl_mining.HashId.Blake3Chunk1024,
}


@dataclass(frozen=True)
class PrebuiltCommitment:
    commitment: PlanarCommitment
    key: bytes


@dataclass(frozen=True)
class MoEBlockInfo:
    expert_index: int
    inner_a_rows: tuple[int, ...]
    routing: tuple[int, ...]
    offsets: tuple[int, ...]

    def expert_rows(self) -> tuple[int, ...]:
        start = self.offsets[self.expert_index - 1] if self.expert_index else 0
        return tuple(self.routing[start : self.offsets[self.expert_index]])

    def owned_copy(self) -> MoEBlockInfo:
        return MoEBlockInfo(
            expert_index=self.expert_index,
            inner_a_rows=tuple(self.inner_a_rows),
            routing=tuple(self.routing),
            offsets=tuple(self.offsets),
        )


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
    moe: MoEBlockInfo | None = None
    b_ancestor_header: bytes | None = None  # Full 108-byte header; None selects parent.

    def owned_copy(self) -> OpenedBlockInfo:
        """Validate and snapshot mutable inputs for async handoff (keeping prebuilt bundles by ref)."""
        _validate_opening(self)
        return OpenedBlockInfo(
            moe=None if self.moe is None else self.moe.owned_copy(),
            a_row_indices=tuple(self.a_row_indices),
            b_column_indices=tuple(self.b_column_indices),
            a_codes=self.a_codes if self.a_commitment is not None else self.a_codes.clone(),
            a_scales=self.a_scales if self.a_commitment is not None else self.a_scales.clone(),
            b_codes=self.b_codes if self.b_commitment is not None else self.b_codes.clone(),
            b_scales=self.b_scales if self.b_commitment is not None else self.b_scales.clone(),
            mining_config=replace(self.mining_config),
            b_commitment=self.b_commitment,
            a_commitment=self.a_commitment,
            b_ancestor_header=None
            if self.b_ancestor_header is None
            else bytes(self.b_ancestor_header),
        )


def is_plain_fp8_job(job: MiningJob) -> bool:
    return int(job.cert_version) == CERT_VERSION_PLAIN_FP8


def _validate_planes(name: str, codes: torch.Tensor, scales: torch.Tensor) -> tuple[int, int]:
    for plane, label, dtype in (
        (codes, f"{name.lower()}_codes", torch.int8),
        (scales, f"{name.lower()}_scales", torch.bfloat16),
    ):
        if not isinstance(plane, torch.Tensor):
            raise ValueError(f"{label} must be a torch.Tensor, got {type(plane).__name__}")
        if plane.dim() != 2:
            raise ValueError(f"{label} must be 2D, got shape {tuple(plane.shape)}")
        if plane.device.type != "cpu":
            raise ValueError(f"{label} must be a CPU tensor, got device {plane.device}")
        if plane.dtype is not dtype:
            raise ValueError(f"{label} must be {dtype}, got {plane.dtype}")
    rows, k = codes.shape
    if rows <= 0 or k <= 0:
        raise ValueError(f"{name} codes shape must be positive, got {codes.shape}")
    if k % DEFAULT_BLOCK_SIZE:
        raise ValueError(
            f"{name.lower()}_codes columns ({k}) must be divisible by the block size "
            f"{DEFAULT_BLOCK_SIZE}"
        )
    expected = (rows, k // DEFAULT_BLOCK_SIZE)
    if tuple(scales.shape) != expected:
        raise ValueError(
            f"{name.lower()}_scales must be {expected} for {name.lower()}_codes {(rows, k)}, "
            f"got {tuple(scales.shape)}"
        )
    if not codes.is_contiguous() or not scales.is_contiguous():
        raise ValueError(f"{name} committed planes must be contiguous")
    return rows, k


def _validate_indices(
    name: str,
    indices: Sequence[int],
    row_count: int,
    size: int,
) -> None:
    normalized = tuple(indices)
    if any(type(index) is not int for index in normalized):
        raise TypeError(f"{name} must contain integers")
    if size <= 0 or len(normalized) != size:
        raise ValueError(
            f"{name} must select exactly one {size}-row lottery tile, got {len(normalized)} rows"
        )
    if any(index < 0 or index >= row_count for index in normalized):
        raise ValueError(f"{name} contains an index outside [0, {row_count})")
    base = normalized[0]
    if base % size or normalized != tuple(range(base, base + size)):
        raise ValueError(f"{name} does not match one committed lottery tile")


def _validate_opening(opened: OpenedBlockInfo) -> tuple[int, int, int]:
    m, k = _validate_planes("A", opened.a_codes, opened.a_scales)
    n, b_k = _validate_planes("B", opened.b_codes, opened.b_scales)
    if k != b_k:
        raise ValueError(f"common dimension mismatch: A has k={k}, B has k={b_k}")

    config = opened.mining_config
    if k != config.common_dim:
        raise ValueError(f"opening k={k} does not match mining configuration k={config.common_dim}")
    moe = opened.moe
    if (moe is None) != (config.experts == 0):
        raise ValueError("an MoE opening needs both mining_config.experts and moe")
    if moe is None:
        _validate_indices("a_row_indices", opened.a_row_indices, m, config.tile_rows)
        _validate_indices("b_column_indices", opened.b_column_indices, n, config.tile_cols)
        return m, n, k

    witness = (
        moe.expert_index,
        *moe.offsets,
        *moe.routing,
        *moe.inner_a_rows,
        *opened.a_row_indices,
        *opened.b_column_indices,
    )
    if any(type(v) is not int for v in witness):
        raise TypeError("MoE witness fields must be integers")
    if n % config.experts or not 0 <= moe.expert_index < config.experts:
        raise ValueError(f"expert {moe.expert_index} / stacked n={n} do not fit {config.experts}")
    n_e = n // config.experts
    _validate_routing_witness(moe, config.experts, m)
    expert_rows = moe.expert_rows()
    _validate_indices("inner_a_rows", moe.inner_a_rows, len(expert_rows), config.tile_rows)
    if tuple(opened.a_row_indices) != tuple(expert_rows[i] for i in moe.inner_a_rows):
        raise ValueError("a_row_indices are not the routed tokens of inner_a_rows")
    base = moe.expert_index * n_e
    _validate_indices(
        "b_column_indices",
        tuple(index - base for index in opened.b_column_indices),
        n_e,
        config.tile_cols,
    )
    return m, n, k


def _validate_routing_witness(moe: MoEBlockInfo, experts: int, m: int) -> None:
    if (
        len(moe.offsets) != experts
        or any(moe.offsets[i] > moe.offsets[i + 1] for i in range(experts - 1))
        or (moe.offsets and (moe.offsets[0] < 0 or moe.offsets[-1] != len(moe.routing)))
        or any(index < 0 or index >= m for index in moe.routing)
    ):
        raise ValueError("MoE routing witness does not describe this opening")
    bounds = (0, *moe.offsets)
    for expert in range(experts):
        segment = moe.routing[bounds[expert] : bounds[expert + 1]]
        if any(a >= b for a, b in zip(segment, segment[1:], strict=False)):
            raise ValueError(f"MoE routing witness R[{expert}] is not strictly increasing")


def _checked_commitment(
    prebuilt: PrebuiltCommitment,
    expected_key: bytes,
    planes: tuple[torch.Tensor, ...],
    hash_id: HashId,
) -> PlanarCommitment:
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


def _resolve_commitment(
    prebuilt: PrebuiltCommitment | None,
    planes: tuple[torch.Tensor, ...],
    key: bytes,
    hash_id: HashId,
) -> PlanarCommitment:
    if prebuilt is None:
        return commit_planes(planes, key, hash_id)
    return _checked_commitment(prebuilt, key, planes, hash_id)


def _native_merkle_proof(proof) -> pearl_mining.MerkleProof:
    return pearl_mining.MerkleProof(
        [bytes(leaf) for leaf in proof.leaf_data],
        list(proof.leaf_indices),
        bytes(proof.root),
        [bytes(sibling) for sibling in proof.siblings],
        proof.total_leaves,
    )


def _native_matrix_proof(opening: MatrixMerkleProof) -> pearl_mining.MatrixMerkleProof:
    return pearl_mining.MatrixMerkleProof(
        _native_merkle_proof(opening.proof), list(opening.row_indices)
    )


def _native_moe(
    moe: MoEBlockInfo, config: MiningConfiguration, key_a: bytes
) -> tuple[pearl_mining.MoeParams, pearl_mining.MoeWitness]:
    params = config.moe_params()
    assert params is not None
    routing, offsets, w = list(moe.routing), list(moe.offsets), moe.expert_index
    routing_tree = commit_routing(routing, key_a, params.hash_id_r)
    start = offsets[w - 1] if w else 0
    leaf_indices = routing_leaf_indices(start, offsets[w], params.hash_id_r)
    witness = pearl_mining.MoeWitness(
        w,
        offsets,
        hash_offsets(offsets, key_a, params.hash_id_o),
        _native_merkle_proof(routing_tree.get_multileaf_proof(leaf_indices)),
    )
    native_params = pearl_mining.MoeParams(
        params.experts, _NATIVE_HASH_ID[params.hash_id_r], _NATIVE_HASH_ID[params.hash_id_o]
    )
    return native_params, witness


def _native_operand(params: OperandParams) -> pearl_mining.OperandParams:
    return pearl_mining.OperandParams(
        params.num_rows,
        _NATIVE_HASH_ID[params.hash_id],
        params.tile_size,
    )


def create_proof(
    opened_block_info: OpenedBlockInfo,
    header: IncompleteBlockHeader,
    ancestor_header: bytes,
    ancestor_chain: Sequence[bytes] = (),
) -> PlainProofV4:
    """Build a cert-v4 FP8 ``PlainProofV4`` from a validated winning opening."""
    m, n, _ = _validate_opening(opened_block_info)
    config = opened_block_info.mining_config
    key_a, key_b = commitment_keys(bytes(header.to_bytes()), ancestor_header)
    comm_a = _resolve_commitment(
        opened_block_info.a_commitment,
        (opened_block_info.a_codes, opened_block_info.a_scales),
        key_a,
        config.a_hash_id,
    )
    comm_b = _resolve_commitment(
        opened_block_info.b_commitment,
        (opened_block_info.b_codes, opened_block_info.b_scales),
        key_b,
        config.b_hash_id,
    )
    a_values, a_scales = comm_a.open(list(opened_block_info.a_row_indices))
    bt_values, bt_scales = comm_b.open(list(opened_block_info.b_column_indices))
    moe = moe_witness = None
    if opened_block_info.moe is not None:
        moe, moe_witness = _native_moe(opened_block_info.moe, config, key_a)

    common = config.common_params()
    return PlainProofV4(
        ancestor_header=pearl_mining.BlockHeader.from_bytes(ancestor_header),
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
        moe=moe,
        moe_witness=moe_witness,
        ancestor_chain=[pearl_mining.BlockHeader.from_bytes(h) for h in ancestor_chain],
    )


def _format_tile_digests(opened: OpenedBlockInfo) -> str:
    a_idx = torch.as_tensor(list(opened.a_row_indices), dtype=torch.int64)
    b_idx = torch.as_tensor(list(opened.b_column_indices), dtype=torch.int64)
    parts = []
    for label, plane, idx in (
        ("A codes", opened.a_codes, a_idx),
        ("A scales", opened.a_scales, a_idx),
        ("B codes", opened.b_codes, b_idx),
        ("B scales", opened.b_scales, b_idx),
    ):
        tile = plane.index_select(0, idx)
        dtype = str(tile.dtype).removeprefix("torch.")
        parts.append(
            f"{label} shape={tuple(tile.shape)} dtype={dtype} blake3={blake3(tensor_bytes(tile)).hexdigest()}"
        )
    return " ".join(parts)


def _log_jackpot_rejection(
    opened: OpenedBlockInfo,
    header: IncompleteBlockHeader,
    plain_proof: PlainProofV4,
    message: str,
) -> None:
    """Debug-log Rust jackpot policy diagnostics and tile plane digests. Never raises."""
    global _jackpot_logs_emitted, _jackpot_log_cap_warned
    if _jackpot_logs_emitted >= _JACKPOT_LOG_LIMIT:
        if not _jackpot_log_cap_warned:
            _jackpot_log_cap_warned = True
            _LOGGER.warning(
                f"jackpot rejection log capped at {_JACKPOT_LOG_LIMIT} for this process; "
                "the counter is not reset per job"
            )
        return
    _jackpot_logs_emitted += 1
    detail = ""
    try:
        detail = _format_tile_digests(opened)
        report = diagnose_jackpot_policy_v4(header, plain_proof)
    except Exception as exc:
        _LOGGER.opt(exception=exc).debug(f"jackpot policy rejected ({message}): {detail}")
        return
    _LOGGER.debug(f"{report}\n  {detail}")


def submit_opened_block(
    opened_block_info: OpenedBlockInfo,
    mining_job: MiningJob,
    client: Any,
) -> PlainProofV4 | None:
    """Verify and hand one cert-v4 FP8 proof to the gateway. Returns ``None`` if policy-filtered."""
    if not is_plain_fp8_job(mining_job):
        raise ValueError(
            "FP8 proof submission requires certificate version "
            f"{CERT_VERSION_PLAIN_FP8}, got {int(mining_job.cert_version)}"
        )

    header = IncompleteBlockHeader.from_bytes(mining_job.incomplete_header_bytes)
    ancestor = (
        opened_block_info.b_ancestor_header
        if opened_block_info.b_ancestor_header is not None
        else mining_job.parent_header
    )
    plain_proof = create_proof(
        opened_block_info, header, ancestor, mining_job.ancestor_chain_to(ancestor)
    )
    is_valid, message = verify_plain_proof_for_cert_version(
        mining_job.cert_version, header, plain_proof
    )
    if not is_valid:
        if message.startswith(_JACKPOT_INADMISSIBLE):
            _log_jackpot_rejection(opened_block_info, header, plain_proof, message)
            return None
        raise RuntimeError(f"Plain proof verification failed: {message}")

    client.submit_plain_proof(plain_proof, mining_job)
    return plain_proof
