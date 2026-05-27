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
from pearl_gateway.comm.mining_configuration import (
    MiningConfiguration,
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
class OpenedBlockInfo:
    A_row_indices: list[int]
    B_column_indices: list[int]
    A: torch.Tensor | None  # Non-noised matrix A, for PlainProof creation
    B_t: torch.Tensor | None  # Non-noised matrix B transposed, for PlainProof creation
    commitment_hash: CommitmentHash | None
    noise_rank: int
    noise_range: ClassVar[int] = 128

    def get_mining_config(self) -> MiningConfiguration:
        if self.A is None or self.B_t is None:
            raise ValueError("A and B must be provided")
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
    mining_config_bytes: bytes | None = None
    matrix_m: int | None = None
    matrix_n: int | None = None
    height: int | None = None
    alpha_notify_nbits: int | None = None

    INNER_HASH_LIMIT: ClassVar[int] = 42
    MAX_TARGET: ClassVar[int] = 2**256 - 1

    def __post_init__(self) -> None:
        match (self.matrix_m, self.matrix_n):
            case (None, None):
                pass
            case (int(m), int(n)) if m > 0 and n > 0:
                pass
            case _:
                raise ValueError("matrix_m and matrix_n must be provided together and nonzero")

        if self.height is not None and self.height < 0:
            raise ValueError("height must be non-negative")
        if self.alpha_notify_nbits is not None and not _valid_compact_target(
            self.alpha_notify_nbits
        ):
            raise ValueError("alpha_notify_nbits must be a positive compact target")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON-RPC response."""
        result: dict[str, Any] = {
            "incomplete_header_bytes": b64_encode(self.incomplete_header_bytes),
            "target": self.target,
        }
        if self.mining_config_bytes is not None:
            result["mining_config_bytes"] = b64_encode(self.mining_config_bytes)
        if self.matrix_m is not None and self.matrix_n is not None:
            result["matrix_m"] = self.matrix_m
            result["matrix_n"] = self.matrix_n
        if self.height is not None:
            result["height"] = self.height
        if self.alpha_notify_nbits is not None:
            result["alpha_notify_nbits"] = self.alpha_notify_nbits
        return result

    @staticmethod
    def _get_difficulty_adjustment_factor(mining_config: MiningConfiguration) -> int:
        return (
            mining_config.hash_tile_h * mining_config.hash_tile_w * mining_config.rounded_common_dim
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiningJob":
        """Create MiningJob from dictionary (JSON-RPC deserialization)."""

        matrix_m = _merge_optional_int(data.get("matrix_m"), data.get("m"), "matrix_m", "m")
        matrix_n = _merge_optional_int(data.get("matrix_n"), data.get("n"), "matrix_n", "n")
        alpha_notify_nbits = _merge_optional_int(
            data.get("alpha_notify_nbits"),
            data.get("notify_nbits"),
            "alpha_notify_nbits",
            "notify_nbits",
        )

        return cls(
            incomplete_header_bytes=b64_decode(data["incomplete_header_bytes"]),
            target=data["target"],
            mining_config_bytes=(
                b64_decode(data["mining_config_bytes"])
                if data.get("mining_config_bytes") is not None
                else None
            ),
            matrix_m=matrix_m,
            matrix_n=matrix_n,
            height=data.get("height"),
            alpha_notify_nbits=alpha_notify_nbits,
        )

    @classmethod
    def from_template(
        cls,
        template: BlockTemplate,
        mining_config_bytes: bytes | None = None,
        matrix_m: int | None = None,
        matrix_n: int | None = None,
    ) -> "MiningJob":
        """Create MiningJob from BlockTemplate."""
        return cls(
            incomplete_header_bytes=template.header.serialize_without_proof_commitment(),
            target=template.target,
            mining_config_bytes=mining_config_bytes,
            matrix_m=matrix_m,
            matrix_n=matrix_n,
            height=template.height,
            alpha_notify_nbits=template.bits,
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


def _merge_optional_int(
    canonical: Any | None,
    alias: Any | None,
    canonical_name: str,
    alias_name: str,
) -> int | None:
    if canonical is not None and alias is not None and canonical != alias:
        raise ValueError(f"{canonical_name} conflicts with {alias_name}")
    value = canonical if canonical is not None else alias
    return int(value) if value is not None else None


def _valid_compact_target(nbits: int) -> bool:
    exponent = nbits >> 24
    mantissa = nbits & 0x00FF_FFFF
    return exponent != 0 and mantissa != 0 and mantissa & 0x0080_0000 == 0
