"""
Tests for ZKCertificate wire-format length validation.

The Go node's wire readers (node/wire/certificate_v1.go and
certificate_v2.go) read exactly the declared number of bytes and error
on short input, and reject proof_data_len > MaxZKProofSize (60000) and
public_data_len > PublicDataMaxSizeV2 (4807). The gateway's deserialize
must do the same: Python slicing silently shortens, so without explicit
checks a truncated or length-lying blob deserializes into a different,
shorter certificate instead of failing.
"""

import struct

import pytest
from pearl_mining import PUBLICDATA_SIZE, ZKProof

from pearl_gateway.blockchain_utils.zk_certificate import (
    CertificateVersion,
    ZKCertificate,
)

_DENSE_HEADER_SIZE = 4 + 32 + PUBLICDATA_SIZE  # version | hash | public data


def _cert(version, public_data=None, proof_data=b"\x09" * 100):
    if public_data is None:
        public_data = bytes([7] * PUBLICDATA_SIZE)
    return ZKCertificate(
        header_hash=bytes(range(32)),
        proof=ZKProof(public_data, proof_data),
        cert_version=version,
    )


class TestDeserializeLengthValidation:
    def test_dense_truncated_proof_rejected(self):
        wire = _cert(CertificateVersion.ZK_DENSE).serialize()
        with pytest.raises(ValueError):
            ZKCertificate.deserialize(wire[:-10])

    def test_dense_oversized_declared_proof_len_rejected(self):
        wire = _cert(CertificateVersion.ZK_DENSE).serialize()
        evil = (
            wire[: _DENSE_HEADER_SIZE]
            + struct.pack("<I", 60000)
            + wire[_DENSE_HEADER_SIZE + 4 :]
        )
        with pytest.raises(ValueError):
            ZKCertificate.deserialize(evil)

    def test_moe_truncated_proof_rejected(self):
        wire = _cert(CertificateVersion.ZK_MOE).serialize()
        with pytest.raises(ValueError):
            ZKCertificate.deserialize(wire[:-5])

    def test_moe_truncated_public_data_rejected(self):
        wire = _cert(CertificateVersion.ZK_MOE).serialize()
        with pytest.raises(ValueError):
            ZKCertificate.deserialize(wire[: 40 + PUBLICDATA_SIZE // 2])

    def test_moe_oversized_public_data_len_rejected(self):
        # Crafted by hand: a peer can send a public_data_len above the
        # Go cap even though this gateway refuses to construct one.
        wire = (
            struct.pack("<I", int(CertificateVersion.ZK_MOE))
            + bytes(range(32))
            + struct.pack("<I", 5000)
            + bytes(5000)
            + struct.pack("<I", 4)
            + b"\x09" * 4
        )
        with pytest.raises(ValueError):
            ZKCertificate.deserialize(wire)

    def test_v3_truncated_proof_rejected(self):
        wire = _cert(CertificateVersion.ZK_V3).serialize()
        with pytest.raises(ValueError):
            ZKCertificate.deserialize(wire[:-1])

    def test_too_short_for_version_rejected(self):
        with pytest.raises(ValueError):
            ZKCertificate.deserialize(b"\x01\x00")

    @pytest.mark.parametrize(
        "version",
        [
            CertificateVersion.ZK_DENSE,
            CertificateVersion.ZK_MOE,
            CertificateVersion.ZK_V3,
        ],
    )
    def test_valid_certificates_round_trip(self, version):
        cert = _cert(version)
        wire = cert.serialize()
        back = ZKCertificate.deserialize(wire)
        assert back.header_hash == cert.header_hash
        assert back.proof.public_data == cert.proof.public_data
        assert back.proof.proof_data == cert.proof.proof_data
        assert back.serialize() == wire


class TestHeaderHashSizeValidation:
    """The header hash sits in a fixed 32-byte field in both wire
    layouts: construction must not silently zero-pad or truncate it,
    or the serialized certificate commits to a different header than
    the caller set."""

    @pytest.mark.parametrize(
        "version",
        [
            CertificateVersion.ZK_DENSE,
            CertificateVersion.ZK_MOE,
            CertificateVersion.ZK_V3,
        ],
    )
    def test_short_header_hash_rejected(self, version):
        with pytest.raises(ValueError):
            ZKCertificate(
                header_hash=b"short",
                proof=ZKProof(bytes([7] * PUBLICDATA_SIZE), b"\x09" * 100),
                cert_version=version,
            )

    @pytest.mark.parametrize(
        "version",
        [
            CertificateVersion.ZK_DENSE,
            CertificateVersion.ZK_MOE,
            CertificateVersion.ZK_V3,
        ],
    )
    def test_long_header_hash_rejected(self, version):
        with pytest.raises(ValueError):
            ZKCertificate(
                header_hash=b"x" * 40,
                proof=ZKProof(bytes([7] * PUBLICDATA_SIZE), b"\x09" * 100),
                cert_version=version,
            )


class TestPublicDataSizeValidation:
    """The dense wire field is fixed-size: serialize must not silently
    zero-pad or truncate public data (the proof commitment is hashed
    over the original bytes, so padding would make the wire certificate
    disagree with its own commitment). MoE/V3 public data is capped at
    the Go node's PublicDataMaxSizeV2."""

    def test_short_public_data_rejected(self):
        with pytest.raises(ValueError):
            _cert(CertificateVersion.ZK_DENSE, public_data=bytes(100))

    def test_long_public_data_rejected(self):
        with pytest.raises(ValueError):
            _cert(CertificateVersion.ZK_DENSE, public_data=bytes(200))

    def test_moe_oversized_public_data_rejected_on_construct(self):
        with pytest.raises(ValueError):
            _cert(CertificateVersion.ZK_MOE, public_data=bytes(5000))
