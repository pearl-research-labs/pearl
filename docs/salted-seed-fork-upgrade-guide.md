# Salted-Seed Hard Fork Upgrade Guide for Miners and Mining Pools

Pearl is doing a hard fork. At a fixed block height (the **fork height**), blocks switch
from the V2 (MoE) ZK certificate to the new V3 (salted noise-seed) certificate.

| Network  | Fork height (`SaltedSeedForkHeight`) |
| -------- | ------------------------------------ |
| Mainnet  | `99000`                              |
| Testnet  | `38648`                              |
| Testnet2 | `83109`                              |

**The short version:**

- V3 changes how the noise seeds are derived from the matrix commitments: each Merkle
  root is first salted with a keyed BLAKE3 hash that also commits the matrix dimensions
  (`m` for A, `n` for B). The ZK circuits, wire formats, and share formats are unchanged.
- Because the derivation feeds mining itself, **everything must be upgraded before the
  fork height: node, ZK proving code, and miners.** A miner running old software
  produces invalid shares from the fork height on. This is different from the MoE fork,
  where old miners kept working.
- Upgraded software switches automatically at the fork height. There is nothing to
  schedule on your side; the certificate version comes from `getblocktemplate` per block.

---

## Step 1: Upgrade your node to v1.4.1

Safe to do at any time before the fork. The node stays fully compatible with V2 blocks
and shares until the fork height.

`getblocktemplate` reports `requiredcertversion: 3` at and after the fork height
(`2` before it). As with the MoE fork: read the version from the template, do not
hardcode the fork height.

## Step 2: Upgrade your ZK proving code (pools)

Deploy before the fork height, or every block you build after the fork is rejected.

If you followed the MoE fork guide and use the certificate-version dispatchers, you
only need the new `pearl-mining` package (v0.3.0) — the dispatchers accept version 3:

- `check_cert_version_eligible`, `generate_proof_for_cert_version`,
  `verify_proof_for_cert_version`, and `verify_plain_proof_for_cert_version` handle
  `requiredcertversion` 1, 2, and 3. Old package versions raise
  `ValueError: unknown certificate version: 3`.
- Versioned entry points exist too: `generate_proof_v3`, `verify_proof_v3`,
  `verify_plain_proof_v3` (same circuits and circuit cache as V2; only the seed
  derivation differs).
- A share is bound to one derivation. A share mined under V2 rules fails V3
  verification and vice versa, so verify shares with the version of the template
  they were mined against.

If you call the C FFI directly: `mine`, `mine_moe`, `verify_plain_proof_ffi`, and
`prove_plain_proof_ffi` now take a `cert_version` argument. Rebuild against the new
header; do not run old binaries against the new library.

If you use the Rust crate directly: pass `SeedDerivation` (from
`zk_pow::api::proof`) to `zk_prove_plain_proof` / `verify_plain_proof`, or map it
from the certificate version with `CertificateVersion::seed_derivation()`.
Reference: `zk-pow/src/api/seed.rs`.

Only if you serialize certificates yourself: the V3 wire layout is identical to V2
(`Version(4) | HeaderHash(32) | PublicDataLen(4) | PublicData(N) | ProofDataLen(4) |
ProofData`) with `3` in the version field, and the header's proof commitment is
`double_sha256(cert_version_le32 + public_data)` with the prefix now `3`.

## Step 3: Upgrade your miners

Unlike the MoE fork, this is required: V3 share noise uses salted seeds, so
mining software that derives seeds the old way produces invalid shares from
the fork height on. Pools should expect
`verify_plain_proof_for_cert_version(3, ...)` to reject them.

If you run the reference miner stack (vllm-miner + gateway), upgrading is all you
need. The mining job now carries `cert_version` (and it is a required field of
`submitPlainProof`); the miner reads it per job and salts when the job requires V3.
Deploy at any time before the fork — it switches automatically at the fork height.

If you built custom mining software, implement the new derivation:

1. Compute the keyed Merkle roots of A and B exactly as today (`hash_a`, `hash_b`).
   The wire formats do not change; shares still carry the raw roots.
2. When the job's `cert_version` is 3, bind each root to its matrix dimension
   before the (unchanged) seed chain:

   ```text
   bound_a = blake3(hash_a || m_le32 || 0^28, key = blake3("pearl/cert-v3/noise-seed/A"))
   bound_b = blake3(hash_b || n_le32 || 0^28, key = blake3("pearl/cert-v3/noise-seed/B"))
   ```

   Each message is exactly one 64-byte BLAKE3 block: the 32-byte root, the
   dimension as a little-endian u32, and 28 zero bytes.
3. Use the bound roots wherever the raw roots fed the seed chain:
   `b_noise_seed = blake3(job_key || bound_b)`, then
   `a_noise_seed = blake3(b_noise_seed || bound_a)`.
4. MoE only: salt **before** the routing fold — `bound_a` (not `hash_a`) goes into
   `hash_activations = blake3(bound_a || hash_routing)`. The dimensions are
   `m` = token count and `n` = the **per-expert** intermediate dimension (B holds
   all experts stacked, but `n` does not include the expert count).

Reference implementations: `zk-pow/src/api/seed.rs` (Rust, with pinned test
vectors), `miner/miner-base/src/miner_base/commitment_hash.py` (Python,
`bind_root_a`/`bind_root_b`), and the CUDA kernel behind
`commitment_hash_from_merkle_roots(..., salted_dims=(m, n))` in `pearl-gemm`.

## Questions

Contact the Pearl team on the usual channels if anything is unclear.
