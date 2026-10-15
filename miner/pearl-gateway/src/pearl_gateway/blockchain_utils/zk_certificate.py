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
    ZK_V3 = 3  # V3: V2 layout with the salted noise-seed derivation.

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
    # Keep in sync with Go wire.PublicDataMaxSizeV2 (V2/V3 public data cap).
    ZK_MAX_PUBLIC_DATA_SIZE_V2: ClassVar[int] = 4807

    def __post_init__(self) -> None:
        if len(self.header_hash) != 32:
            # Both wire layouts carry the header hash in a fixed
            # 32-byte field; numpy would otherwise silently zero-pad
            # or truncate it, and the serialized certificate would
            # commit to a different header than the caller set.
            raise ValueError(
                f"Header hash must be exactly 32 bytes, "
                f"got {len(self.header_hash)} bytes"
            )
        if len(self.proof.proof_data) > self.ZK_MAX_PROOF_DATA_SIZE:
            raise ValueError(
                f"Proof data is too large: {len(self.proof.proof_data)} bytes "
                f"(max {self.ZK_MAX_PROOF_DATA_SIZE} bytes)"
            )
        public_data_len = len(self.proof.public_data)
        if self.cert_version == CertificateVersion.ZK_DENSE:
            # The dense wire format has a fixed-size public data field;
            # numpy would otherwise silently zero-pad or truncate it,
            # and the proof commitment (hashed over the original bytes)
            # would no longer match the serialized certificate.
            if public_data_len != PUBLICDATA_SIZE:
                raise ValueError(
                    f"Dense public data must be exactly {PUBLICDATA_SIZE} bytes, "
                    f"got {public_data_len} bytes"
                )
        elif public_data_len > self.ZK_MAX_PUBLIC_DATA_SIZE_V2:
            raise ValueError(
                f"Public data is too large: {public_data_len} bytes "
                f"(max {self.ZK_MAX_PUBLIC_DATA_SIZE_V2} bytes)"
            )

    def serialize(self) -> bytes:
        """Serialize to the wire format expected by the Go node.

        ZK_DENSE (v1): Version(4) | HeaderHash(32) | PublicData(164) | ProofDataLen(4) | ProofData
        ZK_MOE   (v2): Version(4) | HeaderHash(32) | PublicDataLen(4) | PublicData(N) | ProofDataLen(4) | ProofData
        ZK_V3 (v3): same layout as v2 (only the noise-seed derivation differs).
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
        """Deserialize from raw wire bytes (version-first dispatch).

        Every declared length is checked against the bytes actually
        present, mirroring the Go node's wire readers (io.ReadFull plus
        the MaxZKProofSize / PublicDataMaxSizeV2 caps in
        node/wire/certificate_v1.go and certificate_v2.go). Python
        slicing silently shortens, so without these checks a truncated
        or length-lying blob would deserialize into a different, shorter
        certificate instead of failing. Trailing bytes beyond the
        certificate are left alone, as with a stream reader.
        """
        if len(data) < _CERT_VERSION_SIZE:
            raise ValueError(
                f"Certificate data too short: {len(data)} bytes "
                f"(need at least {_CERT_VERSION_SIZE} for the version)"
            )
        (raw_version,) = struct.unpack_from("<I", data, 0)
        cert_version = CertificateVersion(raw_version)

        if cert_version == CertificateVersion.ZK_DENSE:
            if len(data) < _DENSE_DTYPE.itemsize:
                raise ValueError(
                    f"Dense certificate data too short: {len(data)} bytes "
                    f"(need at least {_DENSE_DTYPE.itemsize} for the header)"
                )
            arr = np.frombuffer(data, dtype=_DENSE_DTYPE, count=1)[0]
            header_hash = bytes(arr["header_hash"])
            public_data = bytes(arr["public_data"])
            proof_data_len = int(arr["proof_data_len"])
            if proof_data_len > cls.ZK_MAX_PROOF_DATA_SIZE:
                raise ValueError(
                    f"Proof data is too large: {proof_data_len} bytes "
                    f"(max {cls.ZK_MAX_PROOF_DATA_SIZE} bytes)"
                )
            if _DENSE_DTYPE.itemsize + proof_data_len > len(data):
                raise ValueError(
                    f"Dense certificate truncated: declares {proof_data_len} "
                    f"proof bytes but only "
                    f"{len(data) - _DENSE_DTYPE.itemsize} are present"
                )
            proof_data = data[_DENSE_DTYPE.itemsize : _DENSE_DTYPE.itemsize + proof_data_len]
        elif cert_version in (CertificateVersion.ZK_MOE, CertificateVersion.ZK_V3):
            if len(data) < _MOE_PREAMBLE_DTYPE.itemsize:
                raise ValueError(
                    f"MoE certificate data too short: {len(data)} bytes "
                    f"(need at least {_MOE_PREAMBLE_DTYPE.itemsize} "
                    f"for the preamble)"
                )
            arr = np.frombuffer(data, dtype=_MOE_PREAMBLE_DTYPE, count=1)[0]
            header_hash = bytes(arr["header_hash"])
            pd_len = int(arr["public_data_len"])
            if pd_len > cls.ZK_MAX_PUBLIC_DATA_SIZE_V2:
                raise ValueError(
                    f"Public data is too large: {pd_len} bytes "
                    f"(max {cls.ZK_MAX_PUBLIC_DATA_SIZE_V2} bytes)"
                )
            pd_start = _MOE_PREAMBLE_DTYPE.itemsize
            pd_end = pd_start + pd_len
            if pd_end + _PROOF_DATA_LEN_SIZE > len(data):
                raise ValueError(
                    f"MoE certificate truncated: declares {pd_len} public "
                    f"bytes but only {len(data) - pd_start} bytes remain"
                )
            public_data = data[pd_start:pd_end]
            (proof_data_len,) = struct.unpack_from("<I", data, pd_end)
            if proof_data_len > cls.ZK_MAX_PROOF_DATA_SIZE:
                raise ValueError(
                    f"Proof data is too large: {proof_data_len} bytes "
                    f"(max {cls.ZK_MAX_PROOF_DATA_SIZE} bytes)"
                )
            if pd_end + _PROOF_DATA_LEN_SIZE + proof_data_len > len(data):
                raise ValueError(
                    f"MoE certificate truncated: declares {proof_data_len} "
                    f"proof bytes but only "
                    f"{len(data) - pd_end - _PROOF_DATA_LEN_SIZE} are present"
                )
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
