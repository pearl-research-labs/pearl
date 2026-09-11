# py-pearl-mining

Python library for Pearl proof-of-work ZK proof generation/verification.
Imports as `pearl_mining`.

## Building

Requires Python >= 3.12 and a Rust toolchain.

```bash
pip install maturin
maturin develop          # debug build, installs into current venv
maturin develop --release  # optimized build
```

## Mining

Mining searches for a matrix solution that satisfies the proof-of-work target.
Production miners run this search on GPUs; the resulting solution is then
packaged into a `PlainProof` (V1–V3) or `PlainProofV4` (FP8) using this library's
types (Merkle trees, matrix proofs, block header, mining configuration, etc.)
and submitted to the gateway.

The module also exposes a legacy Int7 `mine()` function that performs the full search loop
on the CPU. This is a naive implementation included for completeness and testing — it is not suitable for production use.

### Sanity-checking a PlainProof

Before submitting, you can verify the plain proof locally:

```python
from pearl_mining import IncompleteBlockHeader, verify_plain_proof_for_cert_version

header = IncompleteBlockHeader.from_bytes(header_bytes)
is_valid, message = verify_plain_proof_for_cert_version(cert_version, header, plain_proof)
```

`cert_version` is the `requiredcertversion` field from the node's
`getblocktemplate` response: `1` (dense), `2` (dense/MoE), `3` (salted seeds),
or `4` (FP8), according to the active forks.

## ZK Proof Generation and Verification

For V1–V3, the gateway converts a `PlainProof` into a ZK proof before submitting
a block to the node. The following `*_for_cert_version` dispatchers select the
corresponding prover and verifier. The explicitly versioned functions
(`generate_proof_v1` / `generate_proof_v2`, etc.) are also available.

### Generating a ZK proof

```python
from pearl_mining import generate_proof_for_cert_version

zk_proof = generate_proof_for_cert_version(cert_version, header, plain_proof)
# zk_proof.public_data — committed public data (config + proof hashes)
# zk_proof.proof_data  — raw plonky2 proof bytes
```

Raises `ValueError` when the proof cannot be certified at `cert_version`
(an MoE proof before the fork).

### Verifying a ZK proof

```python
from pearl_mining import verify_proof_for_cert_version

is_valid, message = verify_proof_for_cert_version(cert_version, header, zk_proof)
```

Returns `(True, "Verified")` on success, or `(False, reason)` on failure.

For V4, use `Fp8Prover.setup(header, plain_proof_v4)` followed by
`prover.prove(header, plain_proof_v4)` to obtain `(public_data, proof_data)`.
`Fp8Verifier.generate(public_data)` creates the corresponding verifier;
`verifier.verify_block(header, public_data, proof_data)` raises on rejection.
The node separately authenticates the carried ancestor headers during block validation.

## Wire Format

A `ZKCertificate` is assembled from the published `public_data` and `proof_data`.
The full block is then serialized as:

```
ZKCertificate.serialize() | PearlHeader.serialize() | TX_COUNT (varint) | TRANSACTIONS
```

V4 certificates append a CompactSize ancestor count (0–2) after the proof bytes,
then that many full 108-byte headers in parent, grandparent order. These headers
are excluded from the proof commitment. The current miner uses the proposed
header as its proof's ancestor (depth zero), so it writes a zero count.
