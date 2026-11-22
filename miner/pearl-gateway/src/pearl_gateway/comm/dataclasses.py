import base64
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar

import torch
from bitcoinutils.transactions import Transaction
from pearl_gateway.blockchain_utils.blockchain_utils import (
    bits_to_target,
    calculate_merkle_root,
    create_coinbase_transaction,
    double_sha256,
    reconstruct_stratum_merkle_root,
)
from pearl_gateway.blockchain_utils.pearl_header import PearlHeader
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.mining_configuration import (
    MiningConfiguration,
    MoEConfig,
    PearlMiningConfigurationFactory,
)
from pearl_gateway.rpc_types import (
    GetBlockTemplateResponse,
)
from pearl_mining import PENALTY_BASE_RANK, IncompleteBlockHeader, penalized_target_bound
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def get_bytes(data: str | bytes) -> bytes:
    if isinstance(data, str):
        return bytes.fromhex(data)
    return data


def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_decode(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def decode_dtype(encoded_dtype: str) -> torch.dtype:
    # dtype is serialized as "torch.dtype"
    return getattr(torch, encoded_dtype.replace("torch.", ""))


def _hex_to_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise ValueError("value must be hex encoded")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("value must be hex encoded") from exc


def _hex_to_32_bytes(value: str | bytes) -> bytes:
    decoded = _hex_to_bytes(value)
    if len(decoded) != 32:
        raise ValueError("value must be 32 bytes")
    return decoded


def _hex_to_internal_hash32(value: str | bytes) -> bytes:
    # Pool branch hashes are display-order hex; merkle hashing uses internal order.
    return _hex_to_32_bytes(value)[::-1]


def _hex_to_uint32(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError("value must be hex encoded")
    decoded = _hex_to_bytes(value)
    if len(decoded) != 4:
        raise ValueError("value must be 4 bytes")
    return int.from_bytes(decoded, byteorder="big")


def _to_cert_version(value: int) -> CertificateVersion:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("cert_version must be an integer")
    return CertificateVersion(value)


HexBytes = Annotated[bytes, BeforeValidator(_hex_to_bytes)]
Hash32 = Annotated[bytes, BeforeValidator(_hex_to_32_bytes)]
InternalHash32 = Annotated[bytes, BeforeValidator(_hex_to_internal_hash32)]
HexUint32 = Annotated[int, BeforeValidator(_hex_to_uint32)]
CertVersion = Annotated[CertificateVersion, BeforeValidator(_to_cert_version)]


class StratumNotify(BaseModel):
    """Validated mining.notify payload from a Stratum pool."""

    model_config = ConfigDict(strict=True)
    PARAM_FIELDS: ClassVar[tuple[str, ...]] = (
        "job_id",
        "prev_hash",
        "coinb1",
        "coinb2",
        "merkle_branch",
        "version",
        "bits",
        "ntime",
        "clean_jobs",
        "cert_version",
    )

    job_id: str = Field(min_length=1)
    prev_hash: Hash32
    coinb1: HexBytes
    coinb2: HexBytes
    merkle_branch: list[InternalHash32]
    version: HexUint32
    bits: HexUint32
    ntime: HexUint32
    clean_jobs: bool
    cert_version: CertVersion
    # extranonce1 exactly as the pool assigned it at subscribe.
    extranonce: bytes
    # Miner/proxy bytes appended after extranonce1; empty when the pool advertises no extranonce2.
    extranonce2: bytes = b""

    @classmethod
    def from_params(
        cls,
        params: list[Any],
        extranonce: bytes,
        extranonce2: bytes = b"",
    ) -> "StratumNotify":
        """Validate positional Stratum mining.notify params."""

        data = dict(zip(cls.PARAM_FIELDS, params, strict=True))
        data["extranonce"] = extranonce
        data["extranonce2"] = extranonce2
        return cls.model_validate(data)


class StratumSetDifficulty(BaseModel):
    """Validated mining.set_difficulty payload from a Stratum pool."""

    model_config = ConfigDict(strict=True)

    bits: int = Field(ge=1, le=0xFFFFFFFF)

    @classmethod
    def from_params(cls, params: list[Any]) -> "StratumSetDifficulty":
        """Validate positional Stratum mining.set_difficulty params."""
        if len(params) != 1:
            raise ValueError(f"mining.set_difficulty expects 1 param, got {len(params)}")
        return cls.model_validate({"bits": params[0]})


@dataclass
class BlockTemplate:
    """Represents a block template fetched from the Pearl node or a pool."""

    header: PearlHeader
    # None for pool templates: mining.notify carries no height.
    height: int | None
    raw_transactions: list[bytes]
    coinbase_tx: Transaction
    # Certificate version this block must carry under the crossover cutover.
    required_cert_version: CertificateVersion
    # Pool mode only: the Stratum job id, the pool's share target (easier than
    # the network target, so miners find shares), and the extranonce2 echoed
    # back in mining.submit (None when the pool advertises none).
    job_id: str | None = field(default=None)
    share_target: int | None = field(default=None)
    extranonce2: bytes | None = field(default=None)

    @classmethod
    def from_get_block_template(
        cls, data: GetBlockTemplateResponse, mining_address: str
    ) -> "BlockTemplate":
        previousblockhash = data.previousblockhash
        version = data.version
        bits = data.bits
        curtime = data.curtime

        coinbase_tx = create_coinbase_transaction(
            height=data.height,
            coinbase_value=data.coinbasevalue,
            mining_address=mining_address,
            coinbase_aux=data.coinbaseaux.model_dump(),
            default_witness_commitment=data.default_witness_commitment,
        )
        raw_transactions = [bytes.fromhex(tx.data) for tx in data.transactions]
        txids = [tx.txid for tx in data.transactions]

        coinbase_txid = coinbase_tx.get_txid()
        merkle_root = calculate_merkle_root([coinbase_txid] + txids)
        height = data.height

        bits_translation = bits_to_target(int(bits, 16))
        if int(data.target, 16) != bits_translation:
            raise ValueError(f"target and bits must match: {data.target} != {bits_translation}")

        return cls(
            header=PearlHeader(
                incomplete_header=IncompleteBlockHeader(
                    version=version,
                    prev_block=bytes.fromhex(previousblockhash),
                    merkle_root=merkle_root,
                    timestamp=curtime,
                    nbits=int(bits, 16),
                ),
            ),
            height=height,
            raw_transactions=raw_transactions,
            coinbase_tx=coinbase_tx,
            required_cert_version=CertificateVersion(data.requiredcertversion),
        )

    @classmethod
    def from_stratum_notify(cls, notify: StratumNotify) -> "BlockTemplate":
        """Build a BlockTemplate from a Stratum mining.notify message.

        Reconstructs coinbase = coinb1 + extranonce1 + extranonce2 + coinb2, then
        hashes it and walks the merkle branch to get the merkle root.
        """
        extranonce2 = notify.extranonce2
        coinbase_bytes = notify.coinb1 + notify.extranonce + extranonce2 + notify.coinb2
        merkle_root = reconstruct_stratum_merkle_root(
            double_sha256(coinbase_bytes), notify.merkle_branch
        )

        return cls(
            header=PearlHeader(
                incomplete_header=IncompleteBlockHeader(
                    version=notify.version,
                    prev_block=notify.prev_hash,
                    merkle_root=merkle_root,
                    timestamp=notify.ntime,
                    nbits=notify.bits,
                ),
            ),
            height=None,
            raw_transactions=[],
            # from_raw() is safe here because coinbase scripts are all <=75 bytes
            # (scriptSig ~20 B, P2TR output 34 B, witness commitment 38 B)
            # so Script.from_raw() never hits the OP_PUSHDATA1 parse bug that
            # silently drops data in segwit mode.
            coinbase_tx=Transaction.from_raw(coinbase_bytes.hex()),
            required_cert_version=notify.cert_version,
            job_id=notify.job_id,
            extranonce2=extranonce2 if extranonce2 else None,
        )

    def get_raw_transactions(self) -> list[bytes]:
        """Return all transactions as raw bytes, coinbase first."""
        # Safe to use to_bytes() here: the coinbase is constructed by us.
        coinbase_bytes = self.coinbase_tx.to_bytes(self.coinbase_tx.has_segwit)
        return [coinbase_bytes] + self.raw_transactions

    @property
    def bits(self) -> int:
        return self.header.target_bits

    @property
    def target(self) -> int:
        return bits_to_target(self.bits)


@dataclass
class CommitmentHash:
    noise_seed_A: bytes
    noise_seed_B: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_seed_A": b64_encode(self.noise_seed_A),
            "noise_seed_B": b64_encode(self.noise_seed_B),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommitmentHash":
        return cls(
            noise_seed_A=b64_decode(data["noise_seed_A"]),
            noise_seed_B=b64_decode(data["noise_seed_B"]),
        )


@dataclass
class MoEBlockInfo:
    """MoE-specific data for block submission"""

    expert_index: int
    num_experts: int
    n_per_expert: int
    top_k: int
    inner_a_rows: list[int]
    inner_b_cols: list[int]
    routing_data: torch.Tensor  # (m*top_k,) int32, expert-sorted token indices
    expert_routing_offsets: list[int]  # routing exclusive end offsets as per ZK verifier


@dataclass
class OpenedBlockInfo:
    A_row_indices: list[int]
    B_column_indices: list[int]
    A: torch.Tensor | None  # Non-noised matrix A, for PlainProof creation
    B_t: torch.Tensor | None  # Non-noised matrix B transposed, for PlainProof creation
    commitment_hash: CommitmentHash | None
    noise_rank: int
    moe: MoEBlockInfo | None = None
    noise_range: ClassVar[int] = 128

    def get_mining_config(self) -> MiningConfiguration:
        if self.A is None or self.B_t is None:
            raise ValueError("A and B must be provided")
        if self.moe is not None:
            return PearlMiningConfigurationFactory.create(
                common_dim=self.A.shape[1],
                rank=self.noise_rank,
                row_indices=self.moe.inner_a_rows,
                col_indices=self.moe.inner_b_cols,
                moe=MoEConfig(e=self.moe.num_experts, top_k=self.moe.top_k),
            )
        return PearlMiningConfigurationFactory.create(
            common_dim=self.A.shape[1],
            rank=self.noise_rank,
            row_indices=self.A_row_indices,
            col_indices=self.B_column_indices,
        )


@dataclass
class MiningJob:
    """Work unit provided to miners."""

    incomplete_header_bytes: bytes
    target: int
    # Certificate version required for this block.
    cert_version: CertificateVersion

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON-RPC response."""
        return {
            "incomplete_header_bytes": b64_encode(self.incomplete_header_bytes),
            "target": self.target,
            "cert_version": int(self.cert_version),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiningJob":
        """Create MiningJob from dictionary (JSON-RPC deserialization)."""

        return cls(
            incomplete_header_bytes=b64_decode(data["incomplete_header_bytes"]),
            target=data["target"],
            cert_version=CertificateVersion(data["cert_version"]),
        )

    @classmethod
    def from_template(cls, template: BlockTemplate) -> "MiningJob":
        """Create MiningJob from BlockTemplate."""
        return cls(
            incomplete_header_bytes=template.header.serialize_without_proof_commitment(),
            # Pool mode: mine to the pool's share target so miners submit shares.
            target=template.share_target if template.share_target is not None else template.target,
            cert_version=template.required_cert_version,
        )

    def adjust_target(self, mining_config: MiningConfiguration) -> int:
        """Calculate the rank-penalized PoW target for the mining job."""
        if mining_config.rank < PENALTY_BASE_RANK:
            raise ValueError(
                f"noise rank {mining_config.rank} is below the minimum "
                f"{PENALTY_BASE_RANK}; blocks would be rejected by consensus"
            )
        bound = penalized_target_bound(self.target, mining_config)
        if bound is None:
            raise ValueError(
                f"no penalized target for {self.target=} at rank "
                f"{mining_config.rank}: degenerate config or target too easy"
            )
        return bound


class MiningPausedError(Exception):
    """Exception raised when mining should be paused."""

    code = -32001
    message = "mining_paused"

    def __init__(self, details: str = ""):
        self.details = details
        super().__init__(f"{self.message}: {details}" if details else self.message)
