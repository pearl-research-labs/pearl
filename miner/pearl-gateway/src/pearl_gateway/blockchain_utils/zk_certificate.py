import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

from pearl_mining import PUBLICDATA_SIZE

from .blockchain_utils import double_sha256
from .pearl_header import PearlHeader

__all__ = [
    "PUBLICDATA_SIZE",
    "CertificateProof",
    "CertificateVersion",
    "ZKCertificate",
]


class CertificateVersion(IntEnum):
    """On-wire block certificate version (in sync with Go wire.CertificateVersion and Rust)."""

    ZK_DENSE = 1  # V1: dense (non-MoE) proofs only.
    ZK_MOE = 2  # V2: MoE and dense proofs.
    ZK_V3 = 3  # V3: V2 layout with the salted noise-seed derivation.
    PLAIN_FP8 = 4  # V4: FP8 proofs (pearl_mining.PlainProofV4).


_DENSE_STRUCT = struct.Struct(f"<I32s{PUBLICDATA_SIZE}sI")
_VAR_PREAMBLE = struct.Struct("<I32sI")
_U32_LE = struct.Struct("<I")

# Go wire.MaxZKProofSize.
_ZK_MAX_PROOF_DATA_SIZE = 62000

_VARIABLE_LENGTH_VERSIONS = {
    CertificateVersion.ZK_MOE,
    CertificateVersion.ZK_V3,
    CertificateVersion.PLAIN_FP8,
}


@dataclass(frozen=True)
class CertificateProof:
    public_data: bytes
    proof_data: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_data", bytes(self.public_data))
        object.__setattr__(self, "proof_data", bytes(self.proof_data))


@dataclass
class ZKCertificate:
    header_hash: bytes
    proof: CertificateProof
    cert_version: CertificateVersion = field(default=CertificateVersion.ZK_DENSE)
    ancestor_headers: list[PearlHeader] = field(default_factory=list)

    ZK_MAX_PROOF_DATA_SIZE: ClassVar[int] = _ZK_MAX_PROOF_DATA_SIZE
    MAX_ANCESTOR_HEADERS: ClassVar[int] = 3

    def __post_init__(self) -> None:
        if not isinstance(self.proof, CertificateProof):
            self.proof = CertificateProof(self.proof.public_data, self.proof.proof_data)
        self._validate()

    def _validate(self) -> None:
        if len(self.proof.proof_data) > _ZK_MAX_PROOF_DATA_SIZE:
            raise ValueError(
                f"Proof data is too large: {len(self.proof.proof_data)} bytes "
                f"(max {_ZK_MAX_PROOF_DATA_SIZE} bytes)"
            )
        if self.cert_version != CertificateVersion.PLAIN_FP8:
            if self.ancestor_headers:
                raise ValueError("Ancestor headers require a V4 certificate")
            return
        if len(self.header_hash) != 32:
            raise ValueError("V4 header hash must be 32 bytes")
        if len(self.proof.public_data) > _ZK_MAX_PROOF_DATA_SIZE:
            raise ValueError("V4 public data exceeds max size")
        if len(self.ancestor_headers) > self.MAX_ANCESTOR_HEADERS:
            raise ValueError("V4 certificate permits at most three ancestor headers")
        for header in self.ancestor_headers:
            if len(header.serialize()) != PearlHeader.get_serialized_header_size():
                raise ValueError("V4 ancestor header must include a full proof commitment")

    def serialize(self) -> bytes:
        """Serialize to the wire format expected by the Go node."""
        self._validate()
        public_data = self.proof.public_data
        proof = self.proof.proof_data

        if self.cert_version == CertificateVersion.ZK_DENSE:
            return (
                _DENSE_STRUCT.pack(
                    int(self.cert_version), self.header_hash, public_data, len(proof)
                )
                + proof
            )
        if self.cert_version in _VARIABLE_LENGTH_VERSIONS:
            encoded = (
                _VAR_PREAMBLE.pack(int(self.cert_version), self.header_hash, len(public_data))
                + public_data
                + _U32_LE.pack(len(proof))
                + proof
            )
            if self.cert_version == CertificateVersion.PLAIN_FP8:
                # Counts 0-3 have a single-byte canonical varint encoding.
                encoded += bytes([len(self.ancestor_headers)])
                encoded += b"".join(header.serialize() for header in self.ancestor_headers)
            return encoded
        raise ValueError(f"ZKCertificate.serialize does not support {self.cert_version!r}")

    @classmethod
    def deserialize(cls, data: bytes) -> "ZKCertificate":
        """Deserialize from raw wire bytes (version-first dispatch)."""
        (raw_version,) = _U32_LE.unpack_from(data, 0)
        cert_version = CertificateVersion(raw_version)
        ancestor_headers: list[PearlHeader] = []

        if cert_version == CertificateVersion.ZK_DENSE:
            _, header_hash, public_data, proof_data_len = _DENSE_STRUCT.unpack_from(data, 0)
            proof_data = data[_DENSE_STRUCT.size : _DENSE_STRUCT.size + proof_data_len]
        elif cert_version in _VARIABLE_LENGTH_VERSIONS:
            _, header_hash, pd_len = _VAR_PREAMBLE.unpack_from(data, 0)
            pd_start = _VAR_PREAMBLE.size
            pd_end = pd_start + pd_len
            if cert_version == CertificateVersion.PLAIN_FP8 and pd_len > _ZK_MAX_PROOF_DATA_SIZE:
                raise ValueError("V4 public data exceeds max size")
            (proof_data_len,) = _U32_LE.unpack_from(data, pd_end)
            public_data = data[pd_start:pd_end]
            proof_start = pd_end + _U32_LE.size
            proof_end = proof_start + proof_data_len
            if cert_version == CertificateVersion.PLAIN_FP8:
                if proof_data_len > _ZK_MAX_PROOF_DATA_SIZE:
                    raise ValueError("V4 proof data exceeds max size")
                if len(data) <= proof_end:
                    raise ValueError("Truncated V4 proof data or missing ancestor count")
                count = data[proof_end]
                if count > cls.MAX_ANCESTOR_HEADERS:
                    raise ValueError("V4 ancestor count must be between zero and three")
                header_size = PearlHeader.get_serialized_header_size()
                ancestors_start = proof_end + 1
                if len(data) != ancestors_start + count * header_size:
                    raise ValueError("Truncated V4 ancestor headers or trailing data")
                ancestor_headers = [
                    PearlHeader.deserialize(data[offset : offset + header_size])
                    for offset in range(ancestors_start, len(data), header_size)
                ]
            proof_data = data[proof_start:proof_end]
        else:
            raise ValueError(f"Unsupported certificate version: {raw_version}")

        return cls(
            header_hash=header_hash,
            proof=CertificateProof(public_data, proof_data),
            cert_version=cert_version,
            ancestor_headers=ancestor_headers,
        )

    @classmethod
    def from_pearl_header(
        cls,
        header: PearlHeader,
        proof: CertificateProof,
        cert_version: CertificateVersion = CertificateVersion.ZK_DENSE,
        *,
        ancestor_headers: list[PearlHeader] | None = None,
    ) -> "ZKCertificate":
        commitment = cls._get_proof_commitment(proof.public_data, cert_version=cert_version)
        if header.proof_commitment is None:
            header.proof_commitment = commitment
        elif header.proof_commitment != commitment:
            raise ValueError("Proof commitment mismatch")
        return cls(
            header_hash=double_sha256(header.serialize()),
            proof=proof,
            cert_version=cert_version,
            ancestor_headers=[] if ancestor_headers is None else ancestor_headers,
        )

    @staticmethod
    def _get_proof_commitment(
        public_data: bytes | bytearray,
        cert_version: CertificateVersion = CertificateVersion.ZK_DENSE,
    ) -> bytes:
        return double_sha256(_U32_LE.pack(int(cert_version)) + bytes(public_data))

    def get_proof_commitment(self) -> bytes:
        return self._get_proof_commitment(self.proof.public_data, cert_version=self.cert_version)
