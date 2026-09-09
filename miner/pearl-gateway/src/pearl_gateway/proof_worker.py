"""ZK proving that runs in a separate worker process (see ``ProofPool``).

Objects cross the process boundary as bytes/str -- the native ``pearl_mining``
types are not picklable -- so callers pass serialized forms and we rebuild here.
"""

import os

import pearl_mining
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

# Rust ``MMAType::Bf16ToFp8Fp32``; the Python binding only names the Int7 variant.
_MMA_BF16_TO_FP8_FP32 = 1

# The committed job shape one ``Fp8Prover`` setup is specific to (see
# ``_fp8_shape_key``): ``(k, r, quant, device)`` then each operand's
# ``(rows, hash_id, pattern bytes)``, A before B.
Fp8ShapeKey = tuple[int, int, object, object, int, object, bytes, int, object, bytes]

_fp8_prover: Fp8Prover | None = None
_fp8_prover_key: Fp8ShapeKey | None = None


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
        _LOGGER.info("FP8 (cert-v4 pB) warmup deferred to first Fp8Prover.setup for this shape")
        return
    if int.from_bytes(raw[6:8], "little") == _MMA_BF16_TO_FP8_FP32:
        _LOGGER.info("FP8 warmup deferred to first Fp8Prover.setup for this shape")
        return

    _LOGGER.info(
        f"Starting Int7 ZK warmup: common_dim={mining_config.common_dim}, rank={mining_config.rank}"
    )
    pearl_mining.warmup_prove_v2(mining_config)
    _LOGGER.info("ZK warmup completed")


def ready_probe() -> bool:
    """Trivial task used to force a worker to spawn (and thus run ``worker_init``)."""
    return True


def _fp8_shape_key(plain_proof: PlainProofV4) -> Fp8ShapeKey:
    """The committed job shape a prover setup is specific to: ``CommonParams``
    plus both operands' ``(rows, hash_id, pattern)``."""
    common, a, b = plain_proof.common, plain_proof.a, plain_proof.b
    return (
        int(common.k),
        int(common.r),
        common.quant,
        common.device,
        int(a.num_rows),
        a.hash_id,
        bytes(a.pattern.to_bytes()),
        int(b.num_rows),
        b.hash_id,
        bytes(b.pattern.to_bytes()),
    )


def _prove_fp8(
    header: IncompleteBlockHeader,
    plain_proof: PlainProofV4,
    debug: bool,
) -> tuple[bytes, bytes]:
    global _fp8_prover, _fp8_prover_key
    key = _fp8_shape_key(plain_proof)
    if _fp8_prover is None or _fp8_prover_key != key:
        _LOGGER.info(f"Building Fp8Prover for shape {key}")
        _fp8_prover = Fp8Prover.setup(header, plain_proof)
        _fp8_prover_key = key
    public_data, proof_data = _fp8_prover.prove(header, plain_proof)
    public_bytes, proof_bytes = bytes(public_data), bytes(proof_data)
    if debug:
        verifier = Fp8Verifier.generate(public_bytes)
        verifier.verify_block(header, public_bytes, proof_bytes)
    return public_bytes, proof_bytes


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
    plain_proof = PlainProof.from_base64(plain_proof_b64)

    zk_proof = generate_proof_for_cert_version(cert_version, header, plain_proof)

    if debug:
        result, msg = verify_proof_for_cert_version(cert_version, header, zk_proof)
        if not result:
            raise AssertionError(f"Failed to verify proof: {msg}")

    return bytes(zk_proof.public_data), bytes(zk_proof.proof_data)
