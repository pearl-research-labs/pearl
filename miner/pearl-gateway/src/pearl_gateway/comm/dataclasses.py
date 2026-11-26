import base64
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar

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
from pearl_gateway.rpc_types import GetBlockTemplateResponse
from pearl_mining import IncompleteBlockHeader
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

# Matches zk-pow's STATE_WINDOW_DEPTH: the proposed header is excluded.
STATE_WINDOW_DEPTH = 4


def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_decode(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


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


def _hex_to_ancestor_header(value: str | bytes) -> bytes:
    header = _hex_to_bytes(value)
    if len(header) != PearlHeader.get_serialized_header_size():
        raise ValueError("ancestor headers must be full 108-byte wire headers")
    return header


AncestorHeader = Annotated[bytes, BeforeValidator(_hex_to_ancestor_header)]


def _ancestor_window(value: Any) -> Any:
    # Only the state window is ever used (parent_header, ancestor_chain_to), and the
    # list is copied into every mining job, so drop anything past it.
    return value[:STATE_WINDOW_DEPTH] if isinstance(value, list) else value


AncestorWindow = Annotated[list[AncestorHeader], BeforeValidator(_ancestor_window)]


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
    # Optional 11th param: full 108-byte ancestor headers, parent first. The pool
    # sends it for cert-v4 (FP8) jobs, whose proofs key B by an ancestor header.
    ancestor_headers: AncestorWindow = Field(default_factory=list)
    # extranonce1 exactly as the pool assigned it at subscribe.
    extranonce: bytes
    # Miner/proxy bytes appended after extranonce1; empty when the pool advertises no extranonce2.
    extranonce2: bytes = b""

    @model_validator(mode="after")
    def _ancestors_match_cert_version(self) -> "StratumNotify":
        is_v4 = self.cert_version == CertificateVersion.PLAIN_FP8
        if is_v4 and not self.ancestor_headers:
            raise ValueError("cert-v4 mining.notify carries no ancestor headers to key B by")
        if not is_v4 and self.ancestor_headers:
            # parent_header would key B by them for any version, unlike solo mode.
            raise ValueError("ancestor headers are only valid on a cert-v4 mining.notify")
        return self

    @classmethod
    def from_params(
        cls,
        params: list[Any],
        extranonce: bytes,
        extranonce2: bytes = b"",
    ) -> "StratumNotify":
        """Validate positional Stratum mining.notify params.

        Ten params, or eleven when the pool appends ancestor headers (cert-v4
        jobs), so pools that never send the 11th keep working.
        """
        data: dict[str, Any] = {}
        if len(params) == len(cls.PARAM_FIELDS) + 1:
            *params, data["ancestor_headers"] = params
        data |= dict(zip(cls.PARAM_FIELDS, params, strict=True))
        data["extranonce"] = extranonce
        data["extranonce2"] = extranonce2
        return cls.model_validate(data)


class StratumSetDifficulty(BaseModel):
    """Validated mining.set_difficulty payload from a Stratum pool."""

    model_config = ConfigDict(strict=True)

    bits: int = Field(ge=1, le=0xFFFFFFFF)

    @field_validator("bits")
    @classmethod
    def _valid_compact(cls, bits: int) -> int:
        # Same rules as the consensus decoders: no sign bit, and a target in (0, 2**256).
        if bits & 0x00800000:
            raise ValueError("compact target has the sign bit set")
        if not 0 < bits_to_target(bits) < 1 << 256:
            raise ValueError("compact target out of range")
        return bits

    @classmethod
    def from_params(cls, params: list[Any]) -> "StratumSetDifficulty":
        """Validate positional Stratum mining.set_difficulty params."""
        if len(params) != 1:
            raise ValueError(f"mining.set_difficulty expects 1 param, got {len(params)}")
        return cls.model_validate({"bits": params[0]})


@dataclass
class BlockTemplate:
    """Block template fetched from the Pearl node or a Stratum pool."""

    header: PearlHeader
    # None for pool templates: mining.notify carries no height.
    height: int | None
    raw_transactions: list[bytes]
    coinbase_tx: Transaction
    required_cert_version: CertificateVersion
    ancestor_headers: list[bytes] = field(default_factory=list)
    # Pool mode only: the Stratum job id, the pool's share target (easier than
    # the network target, so miners find shares), and the extranonce2 echoed
    # back in mining.submit (None when the pool advertises none).
    job_id: str | None = field(default=None)
    share_target: int | None = field(default=None)
    # The same pool share target in compact form, as sent by mining.set_difficulty.
    share_bits: int | None = field(default=None)
    extranonce2: bytes | None = field(default=None)

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

        ancestor_headers = [_hex_to_ancestor_header(header) for header in data.ancestorheaders]

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

    @classmethod
    def from_stratum_notify(cls, notify: StratumNotify) -> "BlockTemplate":
        """Build a BlockTemplate from a Stratum mining.notify message.

        Reconstructs coinbase = coinb1 + extranonce1 + extranonce2 + coinb2, then
        hashes it and walks the merkle branch to get the merkle root. The raw
        coinbase hash is its txid only because the pool sends a witness-free
        coinb1/coinb2, as Stratum v1 pools do.
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
            ancestor_headers=list(notify.ancestor_headers),
            job_id=notify.job_id,
            extranonce2=extranonce2 if extranonce2 else None,
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
            # Pool mode: mine to the pool's share target so miners submit shares.
            target=template.share_target if template.share_target is not None else template.target,
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
