import base64
from dataclasses import dataclass, field
from typing import Any

from bitcoinutils.transactions import Transaction
from pearl_gateway.blockchain_utils.blockchain_utils import (
    bits_to_target,
    calculate_merkle_root,
    create_coinbase_transaction,
)
from pearl_gateway.blockchain_utils.pearl_header import PearlHeader
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.rpc_types import GetBlockTemplateResponse
from pearl_mining import IncompleteBlockHeader

# Matches zk-pow's STATE_WINDOW_DEPTH: the proposed header is excluded.
STATE_WINDOW_DEPTH = 4


def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_decode(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


@dataclass
class BlockTemplate:
    """Block template fetched from the Pearl node."""

    header: PearlHeader
    height: int
    raw_transactions: list[bytes]
    coinbase_tx: Transaction
    required_cert_version: CertificateVersion
    ancestor_headers: list[bytes] = field(default_factory=list)

    @classmethod
    def from_get_block_template(
        cls, data: GetBlockTemplateResponse, mining_address: str
    ) -> "BlockTemplate":
        coinbase_tx = create_coinbase_transaction(
            height=data.height,
            coinbase_value=data.coinbasevalue,
            mining_address=mining_address,
            coinbase_aux=data.coinbaseaux.model_dump() if data.coinbaseaux else None,
            default_witness_commitment=data.default_witness_commitment,
        )
        raw_transactions = [bytes.fromhex(tx.data) for tx in data.transactions]
        txids = [tx.txid for tx in data.transactions]
        merkle_root = calculate_merkle_root([coinbase_tx.get_txid(), *txids])

        bits_int = int(data.bits, 16)
        bits_translation = bits_to_target(bits_int)
        if int(data.target, 16) != bits_translation:
            raise ValueError(f"target and bits must match: {data.target} != {bits_translation}")

        ancestor_headers = [bytes.fromhex(header) for header in data.ancestorheaders]
        if any(len(h) != PearlHeader.get_serialized_header_size() for h in ancestor_headers):
            raise ValueError("ancestor headers must be full 108-byte wire headers")

        return cls(
            header=PearlHeader(
                incomplete_header=IncompleteBlockHeader(
                    version=data.version,
                    prev_block=bytes.fromhex(data.previousblockhash),
                    merkle_root=merkle_root,
                    timestamp=data.curtime,
                    nbits=bits_int,
                ),
            ),
            height=data.height,
            raw_transactions=raw_transactions,
            coinbase_tx=coinbase_tx,
            required_cert_version=CertificateVersion(data.requiredcertversion),
            ancestor_headers=ancestor_headers,
        )

    def get_raw_transactions(self) -> list[bytes]:
        coinbase_bytes = self.coinbase_tx.to_bytes(self.coinbase_tx.has_segwit)
        return [coinbase_bytes, *self.raw_transactions]

    @property
    def bits(self) -> int:
        return self.header.target_bits

    @property
    def target(self) -> int:
        return bits_to_target(self.bits)


@dataclass
class MiningJob:
    """Work unit provided to miners."""

    incomplete_header_bytes: bytes
    target: int
    cert_version: CertificateVersion
    ancestor_headers: list[bytes] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incomplete_header_bytes": b64_encode(self.incomplete_header_bytes),
            "target": self.target,
            "cert_version": int(self.cert_version),
            "ancestor_headers": [b64_encode(header) for header in self.ancestor_headers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiningJob":
        return cls(
            incomplete_header_bytes=b64_decode(data["incomplete_header_bytes"]),
            target=data["target"],
            cert_version=CertificateVersion(data["cert_version"]),
            ancestor_headers=[b64_decode(header) for header in data.get("ancestor_headers", [])],
        )

    @classmethod
    def from_template(cls, template: BlockTemplate) -> "MiningJob":
        return cls(
            incomplete_header_bytes=template.header.serialize_without_proof_commitment(),
            target=template.target,
            cert_version=template.required_cert_version,
            ancestor_headers=list(template.ancestor_headers),
        )

    @property
    def parent_header(self) -> bytes:
        if not self.ancestor_headers:
            raise ValueError("mining job carries no ancestor headers to key B by")
        return self.ancestor_headers[0]

    def keeps_ancestor(self, ancestor: bytes) -> bool:
        return ancestor in self.ancestor_headers[:STATE_WINDOW_DEPTH]

    def ancestor_chain_to(self, ancestor: bytes) -> list[bytes]:
        window = self.ancestor_headers[:STATE_WINDOW_DEPTH]
        if ancestor not in window:
            raise ValueError("ancestor header is outside this job's state window")
        return window[: window.index(ancestor)]


class MiningPausedError(Exception):
    code = -32001
    message = "mining_paused"

    def __init__(self, details: str = ""):
        self.details = details
        super().__init__(f"{self.message}: {details}" if details else self.message)
