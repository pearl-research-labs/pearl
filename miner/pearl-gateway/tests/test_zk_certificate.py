import struct
from dataclasses import replace
from hashlib import sha256

import pytest
from pearl_gateway.blockchain_utils.pearl_block import PearlBlock
from pearl_gateway.blockchain_utils.pearl_header import PearlHeader
from pearl_gateway.blockchain_utils.zk_certificate import (
    CertificateProof,
    CertificateVersion,
    ZKCertificate,
    _max_proof_data_size,
)
from pearl_gateway.proof_generator import ProofGenerator
from pearl_mining import MIN_MOE_PUBLICDATA_SIZE, PUBLICDATA_SIZE

HEADER_HASH = bytes(range(32))
PROOF_DATA = bytes([0x5A] * 128)
ANCESTOR_BYTES = (
    bytes(range(108)),
    bytes(range(108, 216)),
    bytes(range(216, 256)) + bytes(range(68)),
)
ZK_MAX_BLOB_SIZE = 60000


def _public_data(cert_version: CertificateVersion) -> bytes:
    # V1 pins public data to PUBLICDATA_SIZE; V2/V3/V4 length-prefix a larger
    # blob; V5 (FP16) carries no public data at all (the statement is in proof_data).
    if cert_version == CertificateVersion.PLAIN_FP16:
        return b""
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
    # ZK_DENSE pins public data; V5 omits it; the rest length-prefix it.
    if cert_version not in (CertificateVersion.ZK_DENSE, CertificateVersion.PLAIN_FP16):
        expected += struct.pack("<I", len(public_data))
    expected += public_data + struct.pack("<I", len(PROOF_DATA)) + PROOF_DATA
    if cert_version in (CertificateVersion.PLAIN_FP8, CertificateVersion.PLAIN_FP16):
        expected += b"\x00"
    assert wire == expected
    assert len(wire) == certificate.get_serialized_size()
    assert restored.cert_version == cert_version
    assert restored.header_hash == HEADER_HASH
    assert bytes(restored.proof.public_data) == public_data
    assert bytes(restored.proof.proof_data) == PROOF_DATA
    assert restored.ancestor_headers == []
    # V5 binds proof_data (it has no public data); every other version binds public_data.
    committed_blob = PROOF_DATA if cert_version == CertificateVersion.PLAIN_FP16 else public_data
    committed = struct.pack("<I", int(cert_version)) + committed_blob
    assert certificate.get_proof_commitment() == sha256(sha256(committed).digest()).digest()


@pytest.mark.parametrize("count", [1, 2, 3])
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


def _v5_wire(proof_data, suffix=b"\x00", header_hash=HEADER_HASH):
    # V5: Version(4) | HeaderHash(32) | ProofLen(4) | ProofData | AncestorCount + headers.
    # No public-data blob.
    return (
        struct.pack("<I32sI", 5, header_hash, len(proof_data)) + proof_data + suffix
    )


@pytest.mark.parametrize("count", [0, 1, 2])
def test_v5_serializes_complete_ancestors_in_order(count):
    raw_headers = ANCESTOR_BYTES[:count]
    certificate = ZKCertificate(
        HEADER_HASH,
        CertificateProof(b"", PROOF_DATA),
        CertificateVersion.PLAIN_FP16,
        ancestor_headers=[PearlHeader.deserialize(raw) for raw in raw_headers],
    )
    expected = _v5_wire(PROOF_DATA, bytes([count]) + b"".join(raw_headers))

    assert certificate.serialize() == expected
    assert certificate.get_serialized_size() == len(expected)
    restored = ZKCertificate.deserialize(expected)
    assert restored.cert_version == CertificateVersion.PLAIN_FP16
    assert bytes(restored.proof.public_data) == b""
    assert bytes(restored.proof.proof_data) == PROOF_DATA
    assert [header.serialize() for header in restored.ancestor_headers] == list(raw_headers)
    # V5 binds proof_data, not public_data.
    committed = struct.pack("<I", 5) + PROOF_DATA
    assert certificate.get_proof_commitment() == sha256(sha256(committed).digest()).digest()


def test_v5_rejects_public_data():
    with pytest.raises(ValueError, match="no public data"):
        ZKCertificate(
            HEADER_HASH,
            CertificateProof(b"nonempty", PROOF_DATA),
            CertificateVersion.PLAIN_FP16,
        )


def test_v5_rejects_trailing_bytes():
    with pytest.raises(ValueError, match="trailing"):
        ZKCertificate.deserialize(_v5_wire(PROOF_DATA) + b"\x00")


def test_v5_round_trips_real_sized_proof():
    # The V5 wrapped ZK proof is ~71 KiB, well above the V1-V4 blob cap
    # (_ZK_MAX_PROOF_DATA_SIZE = 60000). Regression: deserialize must use the V5
    # ceiling (_FP16_ZK_MAX_PROOF_DATA_SIZE = 256 KiB), matching serialize, so a
    # real-sized certificate round-trips instead of raising "exceeds max size".
    big_proof = bytes((i * 31) % 256 for i in range(71 * 1024))
    assert len(big_proof) > ZK_MAX_BLOB_SIZE
    assert len(big_proof) <= _max_proof_data_size(CertificateVersion.PLAIN_FP16)
    certificate = ZKCertificate(
        HEADER_HASH,
        CertificateProof(b"", big_proof),
        CertificateVersion.PLAIN_FP16,
    )
    restored = ZKCertificate.deserialize(certificate.serialize())
    assert restored.cert_version == CertificateVersion.PLAIN_FP16
    assert bytes(restored.proof.proof_data) == big_proof


def test_v5_deserialize_rejects_oversize_proof():
    # Just past the V5 ceiling must still be rejected on deserialize.
    over = _max_proof_data_size(CertificateVersion.PLAIN_FP16) + 1
    with pytest.raises(ValueError, match="exceeds max size"):
        ZKCertificate.deserialize(_v5_wire(bytes(over)))


def test_v4_rejects_truncated_certificate():
    wire = _v4_wire(b"public", PROOF_DATA, b"\x03" + b"".join(ANCESTOR_BYTES))
    for end in range(len(wire)):
        with pytest.raises((ValueError, struct.error)):
            ZKCertificate.deserialize(wire[:end])


@pytest.mark.parametrize(
    "suffix",
    [
        b"\x04" + ANCESTOR_BYTES[0] * 4,
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
    parts[field] = bytes(ZK_MAX_BLOB_SIZE)
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
    # Versions that accept ancestors reject one over their per-version limit
    # (V4=3, V5=2); versions that do not accept ancestors reject any.
    certificate.ancestor_headers = [header] * (
        4
        if cert_version == CertificateVersion.PLAIN_FP8
        else 3
        if cert_version == CertificateVersion.PLAIN_FP16
        else 1
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


@pytest.mark.parametrize("count", [0, 3])
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


def test_build_block_certifies_the_supplied_ancestor_chain(sample_block_template):
    template = replace(sample_block_template, required_cert_version=CertificateVersion.PLAIN_FP8)
    block = ProofGenerator.build_block(b"public", PROOF_DATA, template, ANCESTOR_BYTES[1:])
    restored = ZKCertificate.deserialize(block.zk_certificate.serialize())
    assert [header.serialize() for header in restored.ancestor_headers] == list(ANCESTOR_BYTES[1:])
    block = ProofGenerator.build_block(b"public", PROOF_DATA, template)
    assert block.zk_certificate.ancestor_headers == []


@pytest.mark.parametrize("cert_version", list(CertificateVersion))
def test_all_versions_reject_oversized_proofs(cert_version):
    # The proof-data cap is version-aware (V5/FP16 carries a ~256 KiB wrapped ZK
    # proof; V1-V4 keep 60 KB), so size each over-limit blob to its own cap.
    proof = CertificateProof(
        _public_data(cert_version),
        bytes(_max_proof_data_size(cert_version) + 1),
    )

    with pytest.raises(ValueError, match="Proof data is too large"):
        ZKCertificate(HEADER_HASH, proof, cert_version)


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
