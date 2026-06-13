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
packaged into a `PlainProof` using the types from this library (Merkle trees,
matrix proofs, block header, mining configuration, etc.) and submitted to the
gateway.

The module also exposes a `mine()` function that performs the full search loop
on the CPU. This is a naive implementation included for completeness and testing — it is not suitable for production use.

### Sanity-checking a PlainProof

Before submitting, you can verify the plain proof locally:

```python
from pearl_mining import IncompleteBlockHeader, verify_plain_proof_for_cert_version

header = IncompleteBlockHeader.from_bytes(header_bytes)
is_valid, message = verify_plain_proof_for_cert_version(cert_version, header, plain_proof)
```

`cert_version` is the `requiredcertversion` field from the node's
`getblocktemplate` response: `1` (V1/dense certificate) before the MoE fork,
`2` (V2/MoE certificate) at and after it.

## ZK Proof Generation and Verification

The gateway converts a `PlainProof` into a ZK proof before submitting a block
to the node. Use the `*_for_cert_version` dispatchers. They select the correct
prover automatically around the MoE fork. The explicitly versioned functions
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

## Wire Format

After converting a `PlainProof` into a `ZKProof`, a `ZKCertificate` is assembled
from the proof's `public_data` and `proof_data` fields. The full block is then serialized as:

```
ZKCertificate.serialize() | PearlHeader.serialize() | TX_COUNT (varint) | TRANSACTIONS
```
