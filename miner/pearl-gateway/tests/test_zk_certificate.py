import pytest
from pearl_gateway.blockchain_utils.zk_certificate import (
    CertificateProof,
    CertificateVersion,
    ZKCertificate,
)
from pearl_mining import MIN_MOE_PUBLICDATA_SIZE, PUBLICDATA_SIZE

HEADER_HASH = bytes(range(32))
PROOF_DATA = bytes([0x5A] * 128)
CERTIFICATE_VERSION_SIZE = 4


def _public_data(cert_version: CertificateVersion) -> bytes:
    # V1 pins public data to PUBLICDATA_SIZE; V2/V3/V4 length-prefix a larger blob.
    size = (
        PUBLICDATA_SIZE if cert_version == CertificateVersion.ZK_DENSE else MIN_MOE_PUBLICDATA_SIZE
    )
    return bytes(i % 256 for i in range(size))


@pytest.mark.parametrize("cert_version", list(CertificateVersion))
def test_serialize_round_trip(cert_version):
    """Test that every supported certificate version survives a wire round-trip."""
    public_data = _public_data(cert_version)
    certificate = ZKCertificate(
        header_hash=HEADER_HASH,
        proof=CertificateProof(public_data, PROOF_DATA),
        cert_version=cert_version,
    )

    wire = certificate.serialize()
    restored = ZKCertificate.deserialize(wire)

    assert len(wire) == certificate.get_serialized_size()
    assert restored.cert_version == cert_version
    assert restored.header_hash == HEADER_HASH
    assert bytes(restored.proof.public_data) == public_data
    assert bytes(restored.proof.proof_data) == PROOF_DATA


def test_v3_uses_the_moe_wire_layout():
    """Test that V3 only differs from V2 in its version number."""
    public_data = _public_data(CertificateVersion.ZK_V3)
    proof = CertificateProof(public_data, PROOF_DATA)
    moe = ZKCertificate(HEADER_HASH, proof, CertificateVersion.ZK_MOE).serialize()
    v3 = ZKCertificate(HEADER_HASH, proof, CertificateVersion.ZK_V3).serialize()

    assert moe[CERTIFICATE_VERSION_SIZE:] == v3[CERTIFICATE_VERSION_SIZE:]


def test_v4_uses_the_variable_length_wire_layout():
    """Test that V4 uses the same length-prefixed layout as V2/V3."""
    public_data = _public_data(CertificateVersion.PLAIN_FP8)
    proof = CertificateProof(public_data, PROOF_DATA)
    v3 = ZKCertificate(HEADER_HASH, proof, CertificateVersion.ZK_V3).serialize()
    v4 = ZKCertificate(HEADER_HASH, proof, CertificateVersion.PLAIN_FP8).serialize()

    assert v3[CERTIFICATE_VERSION_SIZE:] == v4[CERTIFICATE_VERSION_SIZE:]


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
