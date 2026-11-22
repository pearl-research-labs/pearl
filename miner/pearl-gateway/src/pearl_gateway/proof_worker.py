"""ZK proving that runs in a separate worker process (see ``ProofPool``).

Objects cross the process boundary as bytes/str -- the native ``pearl_mining``
types are not picklable -- so callers pass serialized forms and we rebuild here.
"""

from pearl_mining import (
    IncompleteBlockHeader,
    PlainProof,
    generate_proof_for_cert_version,
    verify_proof_for_cert_version,
)


def prove(
    cert_version: int,
    incomplete_header_bytes: bytes,
    plain_proof_b64: str,
    debug: bool = False,
) -> tuple[bytes, bytes]:
    """Generate a ZK proof; return its ``(public_data, proof_data)`` bytes."""
    header = IncompleteBlockHeader.from_bytes(incomplete_header_bytes)
    plain_proof = PlainProof.from_base64(plain_proof_b64)

    zk_proof = generate_proof_for_cert_version(cert_version, header, plain_proof)

    if debug:
        result, msg = verify_proof_for_cert_version(cert_version, header, zk_proof)
        if not result:
            raise AssertionError(f"Failed to verify proof: {msg}")

    return bytes(zk_proof.public_data), bytes(zk_proof.proof_data)
