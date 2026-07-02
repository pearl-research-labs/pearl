"""ZK proving that runs in a separate worker process (see ``ProofPool``).

Objects cross the process boundary as bytes/str -- the native ``pearl_mining``
types are not picklable -- so callers pass serialized forms and we rebuild here.
"""

import os

import pearl_mining
from miner_utils import get_logger
from pearl_mining import (
    IncompleteBlockHeader,
    PlainProof,
    generate_proof_for_cert_version,
    verify_proof_for_cert_version,
)

_LOGGER = get_logger(__name__)


def worker_init() -> None:
    """ProcessPoolExecutor initializer: warm this worker's circuits once at spawn."""
    # Must never raise: a failing initializer marks the whole pool broken, so on
    # error we log and let the worker warm lazily on its first proof instead.
    try:
        _run_warmup()
    except Exception:
        _LOGGER.exception("ZK warmup failed in worker; warming lazily on first proof")


def _run_warmup() -> None:
    warmup_hex = os.environ.get("PEARL_GATEWAY_WARMUP_SHAPE")
    if not warmup_hex:
        return

    mining_config = pearl_mining.MiningConfiguration.from_bytes(bytes.fromhex(warmup_hex))
    _LOGGER.info(
        f"Starting ZK warmup: common_dim={mining_config.common_dim}, rank={mining_config.rank}"
    )
    pearl_mining.warmup_prove_v2(mining_config)
    _LOGGER.info("ZK warmup completed")


def ready_probe() -> bool:
    """Trivial task used to force a worker to spawn (and thus run ``worker_init``)."""
    return True


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
