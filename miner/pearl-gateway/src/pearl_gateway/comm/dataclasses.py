import base64
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from bitcoinutils.transactions import Transaction
from pearl_gateway.blockchain_utils.blockchain_utils import (
    bits_to_target,
    calculate_merkle_root,
    create_coinbase_transaction,
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
from pearl_mining import IncompleteBlockHeader


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


@dataclass
class BlockTemplate:
    """Represents a block template fetched from the Pearl node."""

    header: PearlHeader
    height: int
    raw_transactions: list[bytes]
    coinbase_tx: Transaction
    # Certificate version this block must carry under the crossover cutover.
    required_cert_version: CertificateVersion = CertificateVersion.ZK_MOE

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
    # Certificate version required for this block;
    cert_version: CertificateVersion = CertificateVersion.ZK_MOE

    INNER_HASH_LIMIT: ClassVar[int] = 42
    MAX_TARGET: ClassVar[int] = 2**256 - 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON-RPC response."""
        return {
            "incomplete_header_bytes": b64_encode(self.incomplete_header_bytes),
            "target": self.target,
            "cert_version": int(self.cert_version),
        }

    @staticmethod
    def _get_difficulty_adjustment_factor(mining_config: MiningConfiguration) -> int:
        return (
            mining_config.hash_tile_h * mining_config.hash_tile_w * mining_config.rounded_common_dim
        )

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
            target=template.target,
            cert_version=template.required_cert_version,
        )

    def adjust_target(self, mining_config: MiningConfiguration) -> int:
        """Calculate the adjusted PoW target for the mining job.

        The target is scaled based on the work represented by the hash tile dimensions
        and noise rank.
        """
        # We reduce difficulty for larger hash tiles (as they represent more work)
        # and for larger rank (as it's the k dimension of the hash tile)
        difficulty_adjustment_factor = self._get_difficulty_adjustment_factor(mining_config)
        adjusted_target = self.target * difficulty_adjustment_factor
        if adjusted_target > self.MAX_TARGET:
            raise ValueError(f"Target is too easy: {self.target=}, {adjusted_target=}")
        return adjusted_target


class MiningPausedError(Exception):
    """Exception raised when mining should be paused."""

    code = -32001
    message = "mining_paused"

    def __init__(self, details: str = ""):
        self.details = details
        super().__init__(f"{self.message}: {details}" if details else self.message)
