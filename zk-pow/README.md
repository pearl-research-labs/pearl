# zk-pow

[![ISC License](https://img.shields.io/badge/license-ISC-blue.svg)](http://copyfree.org)

Rust crate implementing Pearl's zero-knowledge proof-of-work (ZK-PoW) circuit. It converts a GPU-generated `PlainProof` (matrix solution) into a compact Plonky2 ZK proof that the Pearl node verifies on-chain.

## Architecture

```
PlainProof (GPU output)
    └─► zk_prove_plain_proof()  →  ZKProof { public_data, proof_data }
                                        └─► verify_block()  →  Ok / Err
```

- **`api/prove.rs`** — `zk_prove_plain_proof()`: converts a `PlainProof` into a `ZKProof`
- **`api/verify.rs`** — `verify_block()` / `verify_block_cached_circuits_only()`: verifies a `ZKProof`
- **`api/proof.rs`** — core types: `IncompleteBlockHeader`, `PublicProofParams`, `ZKProof`, `PlainProof`
- **`circuit/`** — Plonky2 circuit definition (Pearl STARK + recursion)
- **`ffi/`** — Python bindings (via `pyo3` feature) used by `py-pearl-mining`

## Usage

### Proving

```rust
use zk_pow::api::prove::zk_prove_plain_proof;
use zk_pow::circuit::circuit_utils::CircuitCache;

let mut cache = CircuitCache::default();
let result = zk_prove_plain_proof(block_header, &plain_proof, &mut cache, true)?;
// result.public_data — 164-byte committed public data
// result.proof_data  — raw Plonky2 proof bytes
```

### Verifying

```rust
use zk_pow::api::verify::verify_block;

verify_block(&public_params, &zk_proof, &mut cache)?;
```

### Cached verification (production)

```rust
use zk_pow::api::verify::verify_block_cached_circuits_only;

// Pre-compile circuits once, then reuse cache for all subsequent verifications
verify_block_cached_circuits_only(&public_params, &zk_proof, &cache, None)?;
```

## Building

```bash
# Build (requires Rust toolchain, see rust-toolchain.toml)
cargo build --release

# Run tests
cargo test

# Build with Python bindings
cargo build --release --features pyo3

# Build with embedded verifier cache
cargo build --release --features embedded_cache
```

## Features

| Feature | Description |
|---|---|
| `pyo3` | Enable Python bindings (used by `py-pearl-mining`) |
| `embedded_cache` | Embed pre-compiled verifier circuit into the binary |

## Wire Format

After proving, a `ZKCertificate` is assembled from `public_data` and `proof_data`. The full block is serialized as:

```
ZKCertificate.serialize() | PearlHeader.serialize() | TX_COUNT (varint) | TRANSACTIONS
```

## License

Licensed under the [copyfree](http://copyfree.org) ISC License.
