import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

import numpy as np
from pearl_mining import PUBLICDATA_SIZE, ZKProof

from .blockchain_utils import double_sha256
from .pearl_header import PearlHeader


class CertificateVersion(IntEnum):
    """Block certificate version (the wire format a block's certificate uses).

    The values are on-wire version numbers. Keep the discriminants in sync
    with Go ``wire.CertificateVersion`` and Rust
    ``zk_pow::ffi::plain_proof::CertificateVersion``; add new versions as
    new members instead of renumbering existing ones.
    """

    ZK_DENSE = 1  # V1: dense (non-MoE) proofs only.
    ZK_MOE = 2  # V2: MoE and dense proofs.


_DENSE_DTYPE = np.dtype(
    [
        ("version", "<u4"),
        ("header_hash", "V32"),
        ("public_data", f"V{PUBLICDATA_SIZE}"),
        ("proof_data_len", "<u4"),
    ]
)

# MoE (version 2): preamble is fixed, then variable-length public_data and proof follow.
_MOE_PREAMBLE_DTYPE = np.dtype(
    [
        ("version", "<u4"),
        ("header_hash", "V32"),
        ("public_data_len", "<u4"),
    ]
)

_CERT_VERSION_SIZE = 4  # u32 LE
_PROOF_DATA_LEN_SIZE = 4  # u32 LE


@dataclass
class ZKCertificate:
    header_hash: bytes
    proof: ZKProof
    cert_version: CertificateVersion = field(default=CertificateVersion.ZK_DENSE)

    ZK_MAX_PROOF_DATA_SIZE: ClassVar[int] = 60000

    def __post_init__(self) -> None:
        if len(self.proof.proof_data) > self.ZK_MAX_PROOF_DATA_SIZE:
            raise ValueError(
                f"Proof data is too large: {len(self.proof.proof_data)} bytes "
                f"(max {self.ZK_MAX_PROOF_DATA_SIZE} bytes)"
            )

    def serialize(self) -> bytes:
        """Serialize to the wire format expected by the Go node.

        ZK_DENSE (v1): Version(4) | HeaderHash(32) | PublicData(164) | ProofDataLen(4) | ProofData
        ZK_MOE   (v2): Version(4) | HeaderHash(32) | PublicDataLen(4) | PublicData(N) | ProofDataLen(4) | ProofData
        """
        public_data = bytes(self.proof.public_data)
        proof = bytes(self.proof.proof_data)

        if self.cert_version == CertificateVersion.ZK_DENSE:
            header = np.array(
                [(int(self.cert_version), self.header_hash, public_data, len(proof))],
                dtype=_DENSE_DTYPE,
            )
            return header.tobytes() + proof
        else:
            preamble = np.array(
                [(int(self.cert_version), self.header_hash, len(public_data))],
                dtype=_MOE_PREAMBLE_DTYPE,
            )
            proof_data_len = struct.pack("<I", len(proof))
            return preamble.tobytes() + public_data + proof_data_len + proof

    def get_serialized_size(self) -> int:
        pd_len = len(self.proof.public_data)
        proof_len = len(self.proof.proof_data)
        if self.cert_version == CertificateVersion.ZK_DENSE:
            return _DENSE_DTYPE.itemsize + proof_len
        else:
            return _MOE_PREAMBLE_DTYPE.itemsize + pd_len + _PROOF_DATA_LEN_SIZE + proof_len

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
        elif cert_version == CertificateVersion.ZK_MOE:
            arr = np.frombuffer(data, dtype=_MOE_PREAMBLE_DTYPE, count=1)[0]
            header_hash = bytes(arr["header_hash"])
            pd_len = int(arr["public_data_len"])
            pd_start = _MOE_PREAMBLE_DTYPE.itemsize
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
            proof=ZKProof(public_data, proof_data),
            cert_version=cert_version,
        )

    @classmethod
    def from_pearl_header(
        cls,
        header: PearlHeader,
        proof: ZKProof,
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
