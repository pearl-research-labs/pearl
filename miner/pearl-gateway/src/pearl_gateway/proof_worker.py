"""ZK proving that runs in a separate worker process (see ``ProofPool``).

Objects cross the process boundary as bytes/str -- the native ``pearl_mining``
types are not picklable -- so callers pass serialized forms and we rebuild here.
"""

import os

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

    try:
        zk_proof = generate_proof_for_cert_version(cert_version, header, plain_proof)
        if debug:
            result, msg = verify_proof_for_cert_version(cert_version, header, zk_proof)
            if not result:
                raise AssertionError(f"Failed to verify proof: {msg}")
    except Exception:
        raise
    except BaseException:
        # A Rust panic (PyO3 PanicException is a BaseException) poisons the prover's
        # process-global circuit cache, so every later proof here would fail. Exit and
        # let ProofPool's BrokenExecutor path start a fresh worker.
        os._exit(1)

    return bytes(zk_proof.public_data), bytes(zk_proof.proof_data)
