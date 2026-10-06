"""ZK proving that runs in a separate worker process (see ``ProofPool``).

Objects cross the process boundary as bytes/str -- the native ``pearl_mining``
types are not picklable -- so callers pass serialized forms and we rebuild here.
"""

import os

import pearl_mining
from miner_utils import get_logger
from pearl_mining import (
    CERT_VERSION_PLAIN_FP8,
    CERT_VERSION_PLAIN_FP16,
    Fp8Prover,
    Fp8Verifier,
    Fp16PlainProof,
    Fp16Prover,
    IncompleteBlockHeader,
    PlainProof,
    PlainProofV4,
    generate_proof_for_cert_version,
    verify_proof_for_cert_version,
)

_LOGGER = get_logger(__name__)

# Rust ``MMAType::Bf16ToFp8Fp32``; the Python binding only names the Int7 variant.
_MMA_BF16_TO_FP8_FP32 = 1

_fp8_prover: Fp8Prover | None = None
_fp16_prover: Fp16Prover | None = None


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

    raw = bytes.fromhex(warmup_hex)
    # Legacy Int7 wire: common_dim(4) | rank(2) | mma_type(2) | ... ; cert-v4
    # miners publish their ``pB`` instead, which no Int7 circuit warms.
    try:
        mining_config = pearl_mining.MiningConfiguration.from_bytes(raw)
    except Exception:
        _LOGGER.info("FP8 (cert-v4 pB) warmup deferred to the first FP8 proof")
        return
    if int.from_bytes(raw[6:8], "little") == _MMA_BF16_TO_FP8_FP32:
        _LOGGER.info("FP8 warmup deferred to the first FP8 proof")
        return

    _LOGGER.info(
        f"Starting Int7 ZK warmup: common_dim={mining_config.common_dim}, rank={mining_config.rank}"
    )
    pearl_mining.warmup_prove_v2(mining_config)
    _LOGGER.info("ZK warmup completed")


def ready_probe() -> bool:
    """Trivial task used to force a worker to spawn (and thus run ``worker_init``)."""
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


def _prove_fp16(
    header: IncompleteBlockHeader,
    plain_proof: Fp16PlainProof,
    debug: bool,
) -> tuple[bytes, bytes]:
    """Prove a winning FP16 (A100) tile into the header-bound ``Fp16ZkCertificate``.

    ``plain_proof`` is the opener bundle the miner assembles (job + the committed
    Merkle openings at the winning tile); the prover re-opens + authenticates the
    tile codes under ``header`` and produces the constant-size wrapped proof. The
    V5 certificate carries no public-data blob, so public_data is empty and the
    cert bytes ride entirely in proof_data (``CertificateV5.ProofData``).
    """
    global _fp16_prover
    if _fp16_prover is None:
        _LOGGER.info("Building Fp16Prover (per-geometry wrapper compiles on first proof)")
        _fp16_prover = Fp16Prover()
    cert_bytes = bytes(_fp16_prover.prove_from_plain_proof(header, plain_proof))
    if debug:
        # FP16 consensus verification is the node's job (verify_fp16_zk_cert_ffi);
        # there is no in-process Python verifier, so there is nothing to self-check
        # here without recompiling the (minutes-long) wrapper verifier.
        _LOGGER.info("FP16 debug self-verify is node-side only; skipping in-worker check")
    return b"", cert_bytes


def prove(
    cert_version: int,
    incomplete_header_bytes: bytes,
    plain_proof_b64: str,
    debug: bool = False,
) -> tuple[bytes, bytes]:
    """Generate a ZK proof; return its ``(public_data, proof_data)`` bytes."""
    header = IncompleteBlockHeader.from_bytes(incomplete_header_bytes)
    if cert_version == CERT_VERSION_PLAIN_FP8:
        return _prove_fp8(header, PlainProofV4.from_base64(plain_proof_b64), debug)
    if cert_version == CERT_VERSION_PLAIN_FP16:
        return _prove_fp16(header, Fp16PlainProof.from_base64(plain_proof_b64), debug)
    plain_proof = PlainProof.from_base64(plain_proof_b64)

    zk_proof = generate_proof_for_cert_version(cert_version, header, plain_proof)

    if debug:
        result, msg = verify_proof_for_cert_version(cert_version, header, zk_proof)
        if not result:
            raise AssertionError(f"Failed to verify proof: {msg}")

    return bytes(zk_proof.public_data), bytes(zk_proof.proof_data)
