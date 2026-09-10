import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

import numpy as np
from pearl_mining import PUBLICDATA_SIZE

from .blockchain_utils import double_sha256
from .pearl_header import PearlHeader

# Re-exported for callers that imported PUBLICDATA_SIZE from this module.
__all__ = [
    "PUBLICDATA_SIZE",
    "CertificateProof",
    "CertificateVersion",
    "ZKCertificate",
]


class CertificateVersion(IntEnum):
    """Block certificate version (the wire format a block's certificate uses).

    The values are on-wire version numbers. Keep the discriminants in sync
    with Go ``wire.CertificateVersion`` and Rust
    ``zk_pow::ffi::plain_proof::CertificateVersion``; add new versions as
    new members instead of renumbering existing ones.
    """

    ZK_DENSE = 1  # V1: dense (non-MoE) proofs only.
    ZK_MOE = 2  # V2: MoE and dense proofs.
    ZK_V3 = 3  # V3: V2 layout with the salted noise-seed derivation.
    PLAIN_FP8 = 4  # V4: FP8 proofs (``pearl_mining.PlainProofV4``).

    @property
    def uses_salted_seeds(self) -> bool:
        return self >= CertificateVersion.ZK_V3


_DENSE_DTYPE = np.dtype(
    [
        ("version", "<u4"),
        ("header_hash", "V32"),
        ("public_data", f"V{PUBLICDATA_SIZE}"),
        ("proof_data_len", "<u4"),
    ]
)

# V2/V3/V4: preamble is fixed, then variable-length public_data and proof follow.
_VARIABLE_PREAMBLE_DTYPE = np.dtype(
    [
        ("version", "<u4"),
        ("header_hash", "V32"),
        ("public_data_len", "<u4"),
    ]
)

_CERT_VERSION_SIZE = 4  # u32 LE
_PROOF_DATA_LEN_SIZE = 4  # u32 LE

# Go ``wire.MaxZKProofSize`` / ``MaxFp8ProofSize``.
_ZK_MAX_PROOF_DATA_SIZE = 60000
_FP8_MAX_PROOF_DATA_SIZE = 131072

_VARIABLE_LENGTH_VERSIONS = {
    CertificateVersion.ZK_MOE,
    CertificateVersion.ZK_V3,
    CertificateVersion.PLAIN_FP8,
}


@dataclass(frozen=True)
class CertificateProof:
    """Published ``(public_data, proof_data)`` pair for any certificate version.

    ``pearl_mining.ZKProof`` only accepts Int7 public-data sizes, so V4 FP8
    statements live here instead.
    """

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

    ZK_MAX_PROOF_DATA_SIZE: ClassVar[int] = _ZK_MAX_PROOF_DATA_SIZE
    FP8_MAX_PROOF_DATA_SIZE: ClassVar[int] = _FP8_MAX_PROOF_DATA_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.proof, CertificateProof):
            self.proof = CertificateProof(self.proof.public_data, self.proof.proof_data)
        max_proof = self._max_proof_data_size(self.cert_version)
        if len(self.proof.proof_data) > max_proof:
            raise ValueError(
                f"Proof data is too large: {len(self.proof.proof_data)} bytes "
                f"(max {max_proof} bytes)"
            )

    @staticmethod
    def _max_proof_data_size(cert_version: CertificateVersion) -> int:
        if cert_version == CertificateVersion.PLAIN_FP8:
            return _FP8_MAX_PROOF_DATA_SIZE
        return _ZK_MAX_PROOF_DATA_SIZE

    def serialize(self) -> bytes:
        """Serialize to the wire format expected by the Go node.

        ZK_DENSE (v1): Version(4) | HeaderHash(32) | PublicData(164) | ProofDataLen(4) | ProofData
        ZK_MOE / ZK_V3 / PLAIN_FP8: Version(4) | HeaderHash(32) | PublicDataLen(4) |
            PublicData(N) | ProofDataLen(4) | ProofData
        """
        public_data = bytes(self.proof.public_data)
        proof = bytes(self.proof.proof_data)

        if self.cert_version == CertificateVersion.ZK_DENSE:
            header = np.array(
                [(int(self.cert_version), self.header_hash, public_data, len(proof))],
                dtype=_DENSE_DTYPE,
            )
            return header.tobytes() + proof
        if self.cert_version in _VARIABLE_LENGTH_VERSIONS:
            preamble = np.array(
                [(int(self.cert_version), self.header_hash, len(public_data))],
                dtype=_VARIABLE_PREAMBLE_DTYPE,
            )
            proof_data_len = struct.pack("<I", len(proof))
            return preamble.tobytes() + public_data + proof_data_len + proof
        raise ValueError(f"ZKCertificate.serialize does not support {self.cert_version!r}")

    def get_serialized_size(self) -> int:
        pd_len = len(self.proof.public_data)
        proof_len = len(self.proof.proof_data)
        if self.cert_version == CertificateVersion.ZK_DENSE:
            return _DENSE_DTYPE.itemsize + proof_len
        if self.cert_version in _VARIABLE_LENGTH_VERSIONS:
            return _VARIABLE_PREAMBLE_DTYPE.itemsize + pd_len + _PROOF_DATA_LEN_SIZE + proof_len
        raise ValueError(
            f"ZKCertificate.get_serialized_size does not support {self.cert_version!r}"
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "ZKCertificate":
        """Deserialize from raw wire bytes (version-first dispatch)."""
        (raw_version,) = struct.unpack_from("<I", data, 0)
        cert_version = CertificateVersion(raw_version)

        if cert_version == CertificateVersion.ZK_DENSE:
            arr = np.frombuffer(data, dtype=_DENSE_DTYPE, count=1)[0]
            header_hash = bytes(arr["header_hash"])
            public_data = bytes(arr["public_data"])
            proof_data_len = int(arr["proof_data_len"])
            proof_data = data[_DENSE_DTYPE.itemsize : _DENSE_DTYPE.itemsize + proof_data_len]
        elif cert_version in _VARIABLE_LENGTH_VERSIONS:
            arr = np.frombuffer(data, dtype=_VARIABLE_PREAMBLE_DTYPE, count=1)[0]
            header_hash = bytes(arr["header_hash"])
            pd_len = int(arr["public_data_len"])
            pd_start = _VARIABLE_PREAMBLE_DTYPE.itemsize
            pd_end = pd_start + pd_len
            public_data = data[pd_start:pd_end]
            (proof_data_len,) = struct.unpack_from("<I", data, pd_end)
            proof_data = data[
                pd_end + _PROOF_DATA_LEN_SIZE : pd_end + _PROOF_DATA_LEN_SIZE + proof_data_len
            ]
        else:
            raise ValueError(f"Unsupported certificate version: {raw_version}")

        return cls(
            header_hash=header_hash,
            proof=CertificateProof(public_data, proof_data),
            cert_version=cert_version,
        )

    @classmethod
    def from_pearl_header(
        cls,
        header: PearlHeader,
        proof: CertificateProof,
        cert_version: CertificateVersion = CertificateVersion.ZK_DENSE,
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
        )

    @staticmethod
    def _get_proof_commitment(
        public_data: bytes | bytearray,
        cert_version: CertificateVersion = CertificateVersion.ZK_DENSE,
    ) -> bytes:
        return double_sha256(
            int(cert_version).to_bytes(_CERT_VERSION_SIZE, "little") + bytes(public_data)
        )

    def get_proof_commitment(self) -> bytes:
        return self._get_proof_commitment(self.proof.public_data, cert_version=self.cert_version)
