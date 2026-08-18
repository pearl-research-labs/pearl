"""
Example: using py-pearl-mining to build a V1/V2 gateway pipeline.

The recommended path is the certificate-version dispatchers: pass the
``requiredcertversion`` value from the node's getblocktemplate response and
the library picks the correct prover (V1 before the MoE fork, V2 at and
after it). MoE proofs submitted before the fork are rejected with ValueError.

``PlainProof.from_base64`` accepts both the current proof format and the
legacy V1 format from old miners, so no manual byte patching is needed.

Prerequisites:
  - pip install pearl-mining
  - A running miner sending base64-encoded PlainProofs over some transport
"""

import pearl_mining as pm


def make_header():
    """Construct a dummy block header for illustration."""
    return pm.IncompleteBlockHeader(
        version=1,
        prev_block=bytes(32),
        merkle_root=bytes(32),
        timestamp=0,
        nbits=0x207FFFFF,  # easiest difficulty
    )


# Recommended: dispatch on the node's required certificate version
def handle_miner_share(
    header: pm.IncompleteBlockHeader, raw_b64: str, required_cert_version: int
) -> pm.ZKProof:
    """
    Handle a plain proof share from any miner (old or new).

    ``required_cert_version`` is the ``requiredcertversion`` field of the
    node's getblocktemplate response: 1 (V1/dense) before the MoE fork,
    2 (V2/MoE) at and after it.
    """
    # Accepts both current and legacy V1 serializations.
    plain_proof = pm.PlainProof.from_base64(raw_b64)

    # Raises ValueError for an MoE share before the fork.
    pm.check_cert_version_eligible(required_cert_version, plain_proof)

    ok, msg = pm.verify_plain_proof_for_cert_version(required_cert_version, header, plain_proof)
    if not ok:
        raise ValueError(f"plain proof rejected: {msg}")

    return pm.generate_proof_for_cert_version(required_cert_version, header, plain_proof)


# Explicit version pinning (when you control the whole pipeline)
def generate_v1_certificate(header: pm.IncompleteBlockHeader, raw_b64: str) -> pm.ZKProof:
    """Produce a proof for a V1 certificate (valid only before the MoE fork)."""
    plain_proof = pm.PlainProof.from_base64(raw_b64)

    ok, msg = pm.verify_plain_proof_v1(header, plain_proof)
    if not ok:
        raise ValueError(f"V1 plain proof rejected: {msg}")

    zk = pm.generate_proof_v1(header, plain_proof)
    # zk.public_data is exactly V1_PUBLICDATA_SIZE (164) bytes
    assert len(zk.public_data) == pm.V1_PUBLICDATA_SIZE
    return zk


def generate_v2_certificate(header: pm.IncompleteBlockHeader, raw_b64: str) -> pm.ZKProof:
    """Produce a proof for a V2 certificate.

    Valid at and after the MoE fork. Accepts dense and MoE proofs.
    """
    plain_proof = pm.PlainProof.from_base64(raw_b64)

    ok, msg = pm.verify_plain_proof_v2(header, plain_proof)
    if not ok:
        raise ValueError(f"V2 plain proof rejected: {msg}")

    return pm.generate_proof_v2(header, plain_proof)


# Pool share validation at share difficulty
def validate_share(
    header: pm.IncompleteBlockHeader,
    raw_b64: str,
    required_cert_version: int,
    share_nbits: int,
) -> bool:
    """Verify a share against the pool's (easier) share target instead of the block target."""
    plain_proof = pm.PlainProof.from_base64(raw_b64)
    ok, _ = pm.verify_plain_proof_for_cert_version(
        required_cert_version, header, plain_proof, nbits_override=share_nbits
    )
    return ok
