"""ZK proving executed in a dedicated spawn worker process (see ProofPool).

Objects cross the process boundary as bytes/str because native pearl_mining
types are not picklable.
"""

from miner_utils import get_logger
from pearl_mining import (
    CERT_VERSION_PLAIN_FP8,
    Fp8Prover,
    Fp8Verifier,
    IncompleteBlockHeader,
    PlainProof,
    PlainProofV4,
    generate_proof_for_cert_version,
    verify_proof_for_cert_version,
)

_LOGGER = get_logger(__name__)
_fp8_prover: Fp8Prover | None = None


def ready_probe() -> bool:
    """Trivial task used to spawn the worker process before serving RPC."""
    return True


def _prove_fp8(
    header: IncompleteBlockHeader,
    plain_proof: PlainProofV4,
    debug: bool,
) -> tuple[bytes, bytes]:
    global _fp8_prover
    if _fp8_prover is None:
        _LOGGER.info(f"Building Fp8Prover for initial device {plain_proof.common.device}")
        _fp8_prover = Fp8Prover.setup(plain_proof.common.device)
    public_data, proof_data = _fp8_prover.prove(header, plain_proof)
    public_bytes, proof_bytes = bytes(public_data), bytes(proof_data)
    if debug:
        verifier = Fp8Verifier.generate(public_bytes)
        verifier.verify_block(header, plain_proof.ancestor_chain, public_bytes, proof_bytes)
    return public_bytes, proof_bytes


def prove(
    cert_version: int,
    incomplete_header_bytes: bytes,
    plain_proof_b64: str,
    debug: bool = False,
) -> tuple[bytes, bytes]:
    """Generate a ZK proof and return (public_data, proof_data) bytes."""
    header = IncompleteBlockHeader.from_bytes(incomplete_header_bytes)
    if cert_version == CERT_VERSION_PLAIN_FP8:
        return _prove_fp8(header, PlainProofV4.from_base64(plain_proof_b64), debug)

    plain_proof = PlainProof.from_base64(plain_proof_b64)
    zk_proof = generate_proof_for_cert_version(cert_version, header, plain_proof)
    if debug:
        result, msg = verify_proof_for_cert_version(cert_version, header, zk_proof)
        if not result:
            raise AssertionError(f"Failed to verify proof: {msg}")
    return bytes(zk_proof.public_data), bytes(zk_proof.proof_data)
