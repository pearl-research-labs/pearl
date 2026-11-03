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
    ``zk_pow::ffi::CertificateVersion``; add new versions as
    new members instead of renumbering existing ones.
    """

    ZK_DENSE = 1  # V1: dense (non-MoE) proofs only.
    ZK_MOE = 2  # V2: MoE and dense proofs.
    ZK_V3 = 3  # V3: V2 layout with the salted noise-seed derivation.
    PLAIN_FP8 = 4  # V4: FP8 proofs (``pearl_mining.PlainProofV4``).
    PLAIN_FP16 = 5  # V5: FP16 (A100) proofs (``pearl_mining.Fp16PlainProof``).

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

# V5 (FP16): no public-data blob at all. Preamble is version + header hash, then
# the length-prefixed proof and the ancestor count/headers follow.
_V5_PREAMBLE_DTYPE = np.dtype(
    [
        ("version", "<u4"),
        ("header_hash", "V32"),
    ]
)

_CERT_VERSION_SIZE = 4  # u32 LE
_PROOF_DATA_LEN_SIZE = 4  # u32 LE

# Go ``wire.MaxZKProofSize`` (V1-V4 proof-data cap).
_ZK_MAX_PROOF_DATA_SIZE = 60000

# Go ``wire.MaxFp16ZkCertProofSize``: the V5 (FP16) certificate carries a
# constant-size (~71 KiB) wrapped recursive proof, well above the V1-V4 cap, so
# it gets its own larger ceiling. Must match the Go node's V5 blob cap and the
# Rust FFI ``MAX_FP16_ZK_CERT_SIZE``.
_FP16_ZK_MAX_PROOF_DATA_SIZE = 79_000


def _max_proof_data_size(cert_version: "CertificateVersion") -> int:
    """The proof-data byte cap for ``cert_version`` (version-aware; V5 is larger)."""
    if cert_version == CertificateVersion.PLAIN_FP16:
        return _FP16_ZK_MAX_PROOF_DATA_SIZE
    return _ZK_MAX_PROOF_DATA_SIZE

_VARIABLE_LENGTH_VERSIONS = {
    CertificateVersion.ZK_MOE,
    CertificateVersion.ZK_V3,
    CertificateVersion.PLAIN_FP8,
}

# Versions that carry ancestor headers (parent, then grandparent) after the proof.
_ANCESTOR_VERSIONS = {
    CertificateVersion.PLAIN_FP8,
    CertificateVersion.PLAIN_FP16,
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
    ancestor_headers: list[PearlHeader] = field(default_factory=list)

    ZK_MAX_PROOF_DATA_SIZE: ClassVar[int] = _ZK_MAX_PROOF_DATA_SIZE
    MAX_ANCESTOR_HEADERS: ClassVar[int] = 3

    def __post_init__(self) -> None:
        if not isinstance(self.proof, CertificateProof):
            self.proof = CertificateProof(self.proof.public_data, self.proof.proof_data)
        self._validate()

    def _validate(self) -> None:
        max_proof_data = _max_proof_data_size(self.cert_version)
        if len(self.proof.proof_data) > max_proof_data:
            raise ValueError(
                f"Proof data is too large: {len(self.proof.proof_data)} bytes "
                f"(max {max_proof_data} bytes)"
            )
        if self.cert_version not in _ANCESTOR_VERSIONS:
            if self.ancestor_headers:
                raise ValueError("Ancestor headers require a V4/V5 certificate")
            return
        if len(self.header_hash) != 32:
            raise ValueError("header hash must be 32 bytes")
        if self.cert_version == CertificateVersion.PLAIN_FP8:
            if len(self.proof.public_data) > _ZK_MAX_PROOF_DATA_SIZE:
                raise ValueError("V4 public data exceeds max size")
        elif self.cert_version == CertificateVersion.PLAIN_FP16:
            # V5 carries no public-data blob: the whole statement rides in proof_data.
            if self.proof.public_data:
                raise ValueError("V5 (FP16) certificate carries no public data")
        max_ancestors = 2 if self.cert_version == CertificateVersion.PLAIN_FP16 else self.MAX_ANCESTOR_HEADERS
        if len(self.ancestor_headers) > max_ancestors:
            raise ValueError(f"certificate permits at most {max_ancestors} ancestor headers")
        for header in self.ancestor_headers:
            if len(header.serialize()) != PearlHeader.get_serialized_header_size():
                raise ValueError("ancestor header must include a full proof commitment")

    def serialize(self) -> bytes:
        """Serialize to the wire format expected by the Go node.

        ZK_DENSE (v1): Version(4) | HeaderHash(32) | PublicData(164) | ProofDataLen(4) | ProofData
        ZK_MOE / ZK_V3: Version(4) | HeaderHash(32) | PublicDataLen(4) |
            PublicData(N) | ProofDataLen(4) | ProofData
        PLAIN_FP8 (v4): the same prefix, then AncestorCount(1) | AncestorHeaders(108 each).
            The count is mandatory, including zero; parent precedes grandparent.
        PLAIN_FP16 (v5): Version(4) | HeaderHash(32) | ProofDataLen(4) | ProofData |
            AncestorCount(1) | AncestorHeaders(108 each). No public-data blob.
        """
        self._validate()
        public_data = bytes(self.proof.public_data)
        proof = bytes(self.proof.proof_data)

        if self.cert_version == CertificateVersion.PLAIN_FP16:
            preamble = np.array(
                [(int(self.cert_version), self.header_hash)],
                dtype=_V5_PREAMBLE_DTYPE,
            )
            encoded = preamble.tobytes() + struct.pack("<I", len(proof)) + proof
            # Counts 0-2 have a single-byte canonical varint encoding.
            encoded += bytes([len(self.ancestor_headers)])
            encoded += b"".join(header.serialize() for header in self.ancestor_headers)
            return encoded
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
            encoded = preamble.tobytes() + public_data + proof_data_len + proof
            if self.cert_version == CertificateVersion.PLAIN_FP8:
                # Counts 0–3 have a single-byte canonical varint encoding.
                encoded += bytes([len(self.ancestor_headers)])
                encoded += b"".join(header.serialize() for header in self.ancestor_headers)
            return encoded
        raise ValueError(f"ZKCertificate.serialize does not support {self.cert_version!r}")

    def get_serialized_size(self) -> int:
        self._validate()
        pd_len = len(self.proof.public_data)
        proof_len = len(self.proof.proof_data)
        if self.cert_version == CertificateVersion.PLAIN_FP16:
            return (
                _V5_PREAMBLE_DTYPE.itemsize
                + _PROOF_DATA_LEN_SIZE
                + proof_len
                + 1
                + len(self.ancestor_headers) * PearlHeader.get_serialized_header_size()
            )
        if self.cert_version == CertificateVersion.ZK_DENSE:
            return _DENSE_DTYPE.itemsize + proof_len
        if self.cert_version in _VARIABLE_LENGTH_VERSIONS:
            size = _VARIABLE_PREAMBLE_DTYPE.itemsize + pd_len + _PROOF_DATA_LEN_SIZE + proof_len
            if self.cert_version == CertificateVersion.PLAIN_FP8:
                size += 1 + len(self.ancestor_headers) * PearlHeader.get_serialized_header_size()
            return size
        raise ValueError(
            f"ZKCertificate.get_serialized_size does not support {self.cert_version!r}"
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "ZKCertificate":
        """Deserialize from raw wire bytes (version-first dispatch)."""
        (raw_version,) = struct.unpack_from("<I", data, 0)
        cert_version = CertificateVersion(raw_version)
        ancestor_headers = []

        if cert_version == CertificateVersion.PLAIN_FP16:
            arr = np.frombuffer(data, dtype=_V5_PREAMBLE_DTYPE, count=1)[0]
            header_hash = bytes(arr["header_hash"])
            public_data = b""
            proof_start = _V5_PREAMBLE_DTYPE.itemsize
            (proof_data_len,) = struct.unpack_from("<I", data, proof_start)
            # V5 carries the wrapped ZK proof (~71 KiB today); use the V5 ceiling,
            # matching serialize/_validate (_FP16_ZK_MAX_PROOF_DATA_SIZE). The
            # smaller _ZK_MAX_PROOF_DATA_SIZE (V1-V4) would reject real V5 proofs.
            if proof_data_len > _max_proof_data_size(cert_version):
                raise ValueError("V5 proof data exceeds max size")
            proof_body_start = proof_start + _PROOF_DATA_LEN_SIZE
            proof_end = proof_body_start + proof_data_len
            if len(data) <= proof_end:
                raise ValueError("Truncated V5 proof data or missing ancestor count")
            count = data[proof_end]
            if count > cls.MAX_ANCESTOR_HEADERS:
                raise ValueError("V5 ancestor count must be between zero and two")
            header_size = PearlHeader.get_serialized_header_size()
            ancestors_start = proof_end + 1
            if len(data) != ancestors_start + count * header_size:
                raise ValueError("Truncated V5 ancestor headers or trailing data")
            ancestor_headers = [
                PearlHeader.deserialize(data[offset : offset + header_size])
                for offset in range(ancestors_start, len(data), header_size)
            ]
            proof_data = data[proof_body_start:proof_end]
            return cls(
                header_hash=header_hash,
                proof=CertificateProof(public_data, proof_data),
                cert_version=cert_version,
                ancestor_headers=ancestor_headers,
            )

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
            if cert_version == CertificateVersion.PLAIN_FP8 and pd_len > _ZK_MAX_PROOF_DATA_SIZE:
                raise ValueError("V4 public data exceeds max size")
            (proof_data_len,) = struct.unpack_from("<I", data, pd_end)
            public_data = data[pd_start:pd_end]
            proof_start = pd_end + _PROOF_DATA_LEN_SIZE
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
        commitment = cls._get_proof_commitment(
            cls._committed_blob(proof, cert_version), cert_version=cert_version
        )
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
    def _committed_blob(
        proof: CertificateProof, cert_version: CertificateVersion
    ) -> bytes:
        """The blob the proof commitment binds over.

        V5 (FP16) has no public-data blob, so it commits over ``proof_data``
        (matching Go ``CertificateV5.ProofCommitment``); every other version
        commits over ``public_data`` (``CertificateV*.ProofCommitment``).
        """
        if cert_version == CertificateVersion.PLAIN_FP16:
            return bytes(proof.proof_data)
        return bytes(proof.public_data)

    @staticmethod
    def _get_proof_commitment(
        committed_blob: bytes | bytearray,
        cert_version: CertificateVersion = CertificateVersion.ZK_DENSE,
    ) -> bytes:
        return double_sha256(
            int(cert_version).to_bytes(_CERT_VERSION_SIZE, "little") + bytes(committed_blob)
        )

    def get_proof_commitment(self) -> bytes:
        return self._get_proof_commitment(
            self._committed_blob(self.proof, self.cert_version),
            cert_version=self.cert_version,
        )
