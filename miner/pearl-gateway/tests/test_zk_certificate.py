import struct
from hashlib import sha256

import pytest
from pearl_gateway.blockchain_utils.pearl_block import PearlBlock
from pearl_gateway.blockchain_utils.pearl_header import PearlHeader
from pearl_gateway.blockchain_utils.zk_certificate import (
    CertificateProof,
    CertificateVersion,
    ZKCertificate,
)
from pearl_mining import MIN_MOE_PUBLICDATA_SIZE, PUBLICDATA_SIZE

HEADER_HASH = bytes(range(32))
PROOF_DATA = bytes([0x5A] * 128)
ANCESTOR_BYTES = (bytes(range(108)), bytes(range(108, 216)))
FP8_MAX_BLOB_SIZE = 131072


def _public_data(cert_version: CertificateVersion) -> bytes:
    # V1 pins public data to PUBLICDATA_SIZE; V2/V3/V4 length-prefix a larger blob.
    size = (
        PUBLICDATA_SIZE if cert_version == CertificateVersion.ZK_DENSE else MIN_MOE_PUBLICDATA_SIZE
    )
    return bytes(i % 256 for i in range(size))


def _v4_wire(public_data, proof_data, suffix=b"\x00", header_hash=HEADER_HASH):
    return (
        struct.pack("<I32sI", 4, header_hash, len(public_data))
        + public_data
        + struct.pack("<I", len(proof_data))
        + proof_data
        + suffix
    )


@pytest.mark.parametrize("cert_version", list(CertificateVersion))
def test_serialize_round_trip(cert_version):
    """Pin each version's wire layout and commitment independently of the codec."""
    public_data = _public_data(cert_version)
    certificate = ZKCertificate(
        header_hash=HEADER_HASH,
        proof=CertificateProof(public_data, PROOF_DATA),
        cert_version=cert_version,
    )

    wire = certificate.serialize()
    restored = ZKCertificate.deserialize(wire)

    expected = struct.pack("<I", int(cert_version)) + HEADER_HASH
    if cert_version != CertificateVersion.ZK_DENSE:
        expected += struct.pack("<I", len(public_data))
    expected += public_data + struct.pack("<I", len(PROOF_DATA)) + PROOF_DATA
    if cert_version == CertificateVersion.PLAIN_FP8:
        expected += b"\x00"
    assert wire == expected
    assert len(wire) == certificate.get_serialized_size()
    assert restored.cert_version == cert_version
    assert restored.header_hash == HEADER_HASH
    assert bytes(restored.proof.public_data) == public_data
    assert bytes(restored.proof.proof_data) == PROOF_DATA
    assert restored.ancestor_headers == []
    committed = struct.pack("<I", int(cert_version)) + public_data
    assert certificate.get_proof_commitment() == sha256(sha256(committed).digest()).digest()


@pytest.mark.parametrize("count", [1, 2])
def test_v4_serializes_complete_ancestors_in_order(count):
    public_data = b"public data"
    raw_headers = ANCESTOR_BYTES[:count]
    certificate = ZKCertificate(
        HEADER_HASH,
        CertificateProof(public_data, PROOF_DATA),
        CertificateVersion.PLAIN_FP8,
        ancestor_headers=[PearlHeader.deserialize(raw) for raw in raw_headers],
    )
    expected = _v4_wire(public_data, PROOF_DATA, bytes([count]) + b"".join(raw_headers))

    assert certificate.serialize() == expected
    assert certificate.get_serialized_size() == len(expected)
    restored = ZKCertificate.deserialize(expected)
    assert [header.serialize() for header in restored.ancestor_headers] == list(raw_headers)
    committed = struct.pack("<I", 4) + public_data
    assert certificate.get_proof_commitment() == sha256(sha256(committed).digest()).digest()


def test_v4_rejects_truncated_certificate():
    wire = _v4_wire(b"public", PROOF_DATA, b"\x02" + b"".join(ANCESTOR_BYTES))
    for end in range(len(wire)):
        with pytest.raises((ValueError, struct.error)):
            ZKCertificate.deserialize(wire[:end])


@pytest.mark.parametrize(
    "suffix",
    [
        b"\x03" + ANCESTOR_BYTES[0] * 3,
        b"\xfd\x00\x00",
        b"\xfd\x01\x00" + ANCESTOR_BYTES[0],
        b"\xfe\x02\x00\x00\x00" + b"".join(ANCESTOR_BYTES),
        b"\xff\x02\x00\x00\x00\x00\x00\x00\x00" + b"".join(ANCESTOR_BYTES),
    ],
    ids=["over limit", "noncanonical zero", "noncanonical one", "uint32 count", "uint64 count"],
)
def test_v4_rejects_invalid_ancestor_count(suffix):
    with pytest.raises(ValueError, match="ancestor count"):
        ZKCertificate.deserialize(_v4_wire(b"public", PROOF_DATA, suffix))


def test_v4_rejects_trailing_bytes():
    with pytest.raises(ValueError, match="trailing"):
        ZKCertificate.deserialize(_v4_wire(b"public", PROOF_DATA) + b"\x00")


@pytest.mark.parametrize("field", ["public_data", "proof_data"])
def test_v4_blob_size_limits(field):
    parts = {"public_data": b"public", "proof_data": PROOF_DATA}
    parts[field] = bytes(FP8_MAX_BLOB_SIZE)
    proof = CertificateProof(**parts)
    certificate = ZKCertificate(HEADER_HASH, proof, CertificateVersion.PLAIN_FP8)
    assert ZKCertificate.deserialize(certificate.serialize()).proof == proof

    parts[field] += b"\x00"
    with pytest.raises(ValueError):
        ZKCertificate(HEADER_HASH, CertificateProof(**parts), CertificateVersion.PLAIN_FP8)
    with pytest.raises(ValueError):
        ZKCertificate.deserialize(_v4_wire(**parts))
    certificate.proof = CertificateProof(**parts)
    with pytest.raises(ValueError):
        certificate.serialize()
    with pytest.raises(ValueError):
        certificate.get_serialized_size()


@pytest.mark.parametrize("cert_version", list(CertificateVersion))
def test_ancestor_count_limit_for_each_version(cert_version):
    certificate = ZKCertificate(
        HEADER_HASH, CertificateProof(_public_data(cert_version), PROOF_DATA), cert_version
    )
    header = PearlHeader.deserialize(ANCESTOR_BYTES[0])
    certificate.ancestor_headers = [header] * (
        3 if cert_version == CertificateVersion.PLAIN_FP8 else 1
    )
    with pytest.raises(ValueError):
        certificate.serialize()
    with pytest.raises(ValueError):
        certificate.get_serialized_size()


@pytest.mark.parametrize("commitment", [None, b"\x00"])
def test_v4_requires_complete_ancestor_headers(commitment):
    header = PearlHeader.deserialize(ANCESTOR_BYTES[0])
    header.proof_commitment = commitment
    with pytest.raises(ValueError):
        ZKCertificate(
            HEADER_HASH,
            CertificateProof(b"public", PROOF_DATA),
            CertificateVersion.PLAIN_FP8,
            ancestor_headers=[header],
        )


@pytest.mark.parametrize("count", [0, 2])
def test_v4_block_framing(count):
    raw_headers = ANCESTOR_BYTES[:count]
    header = PearlHeader(PearlHeader.deserialize(ANCESTOR_BYTES[0]).incomplete_header)
    public_data = b"public"
    certificate = ZKCertificate.from_pearl_header(
        header,
        CertificateProof(public_data, PROOF_DATA),
        CertificateVersion.PLAIN_FP8,
        ancestor_headers=[PearlHeader.deserialize(raw) for raw in raw_headers],
    )
    block = PearlBlock(header, [b"coinbase", b"transaction"], certificate)

    commitment = sha256(sha256(struct.pack("<I", 4) + public_data).digest()).digest()
    expected_header = ANCESTOR_BYTES[0][:76] + commitment
    expected_hash = sha256(sha256(expected_header).digest()).digest()
    expected_certificate = _v4_wire(
        public_data, PROOF_DATA, bytes([count]) + b"".join(raw_headers), expected_hash
    )
    assert block.serialize() == expected_certificate + expected_header + b"\x02coinbasetransaction"


@pytest.mark.parametrize(
    ("cert_version", "expected"),
    [
        (CertificateVersion.ZK_DENSE, False),
        (CertificateVersion.ZK_MOE, False),
        (CertificateVersion.ZK_V3, True),
        (CertificateVersion.PLAIN_FP8, True),
    ],
)
def test_uses_salted_seeds(cert_version, expected):
    """Test that only V3 and later select the salted noise-seed derivation."""
    assert cert_version.uses_salted_seeds is expected
