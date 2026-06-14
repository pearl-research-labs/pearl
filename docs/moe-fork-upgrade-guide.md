# MoE Hard Fork Upgrade Guide for Miners and Mining Pools

Pearl is doing a hard fork. At a fixed block height (the **fork height**), blocks switch
from the V1 (dense) ZK certificate to the new V2 (MoE) ZK certificate.

| Network | Fork height (`MoEForkHeight`) |
| ------- | ----------------------------- |
| Testnet | TBD                         |
| Mainnet | TBD              |

**The short version:**

- The fork adds support for **mining MoE models**. After the fork, you can
  upgrade your miners to mine MoE. This is optional; see Step 3.
- Your **hashing miners do not need to change.** Old dense proofs work before and
  after the fork.
- Your **node** and your **ZK proving code** (the code that turns a plain proof share
  into a ZK proof and submits the block) **must be upgraded before the fork height.**

---

## Step 1: Upgrade your node to v1.1.0

Do this first. It is safe to do at any time before the fork.

- The node is fully compatible with old miners and old V1 ZK proofs until the fork height.
- The only API change: `getblocktemplate` now returns a new field, `requiredcertversion`.

```json
{
  "height": 12345,
  "requiredcertversion": 1,
  "...": "..."
}
```

- `1` = the block must carry a V1 (dense) certificate before the fork.
- `2` = the block must carry a V2 (MoE) certificate at and after the fork.

Old miner software that ignores unknown JSON fields keeps working unchanged.
If your `getblocktemplate` parser is strict (rejects unknown fields), fix that first.

Use this field to choose the certificate version. Do not hardcode the fork height in
your pool. Read the version from the template.

Reference: `node/btcjson/chainsvrresults.go` (`RequiredCertVersion`),
`node/chaincfg/params.go` (`MoEForkHeight`, `RequiredCertVersion`).

## Step 2: Upgrade your ZK proving code (pools)

This applies to you if your miners submit **plain proof shares** and your pool does
the ZK proving before submitting the block. You must deploy this **before the fork
height**, or every block you build after the fork will be rejected.

### 2.1: Function names changed

In the new `pearl-mining` Python package (v0.2.0), every proving function has an
explicit version suffix:

- New V2 prover: `generate_proof_v2`, `verify_proof_v2`, `verify_plain_proof_v2`.
- Old V1 prover: `generate_proof_v1`, `verify_proof_v1`, `verify_plain_proof_v1`.

The old unsuffixed names (`generate_proof`, `verify_proof`, `verify_plain_proof`)
are **removed**. Code that still uses them fails at startup with `AttributeError`.

To check the package version at runtime:

- New package: `pearl_mining.__version__ == "0.2.0"`.
- Old package: no `pearl_mining.__version__` attribute.

You normally do not call the versioned functions directly. Use the dispatchers
below instead.

### 2.2: Use the certificate-version dispatchers

Pass `requiredcertversion` from the template, and the library picks the correct
prover for you:

```python
import pearl_mining as pm

def build_zk_proof(
    template: dict,
    header: pm.IncompleteBlockHeader,
    plain_proof: pm.PlainProof,
) -> tuple[pm.ZKProof, int]:
    cert_version = template["requiredcertversion"]

    ok, msg = pm.verify_plain_proof_for_cert_version(cert_version, header, plain_proof)
    if not ok:
        raise ValueError(f"share rejected: {msg}")

    # Raises ValueError for an MoE share before the fork.
    zk_proof = pm.generate_proof_for_cert_version(cert_version, header, plain_proof)
    return zk_proof, cert_version
```

The V2 prover accepts both dense (old) and MoE (new) plain proofs. After the fork,
shares from old miners still work. They are proven by the V2 prover.

To check a share cheaply without proving (for example at share intake), use
`pm.check_cert_version_eligible(cert_version, plain_proof)`. It raises `ValueError`
for an MoE share before the fork.

Both `verify_plain_proof_for_cert_version` and the versioned `verify_plain_proof_v1`
/ `verify_plain_proof_v2` accept `nbits_override` to verify shares against your
pool's share target instead of the block target.

### 2.3: Parse shares from old miners

`PlainProof.from_base64` accepts both the new format and the old (pre-fork) format
from old miners. Nothing to do:

```python
plain_proof = pm.PlainProof.from_base64(raw_b64)  # old or new format
```

**See `py-pearl-mining/examples/v1_v2_gateway_example.py` for a full example.**
It shows how to turn a miner's share into the ZK proof needed for the block.

### 2.4: Only if you serialize certificates yourself

If your pool builds the block certificate bytes itself (instead of using our gateway
code), two things changed for V2:

- **Wire format.** V2 has variable-length public data:

  ```text
  V1: Version(4) | HeaderHash(32) | PublicData(164) | ProofDataLen(4) | ProofData
  V2: Version(4) | HeaderHash(32) | PublicDataLen(4) | PublicData(N) | ProofDataLen(4) | ProofData
  ```

- **Proof commitment.** The header's proof commitment is
  `double_sha256(cert_version_le32 + public_data)`. The version prefix is now `2`
  for V2 certificates. It was always `1` before.

Reference: `miner/pearl-gateway/src/pearl_gateway/blockchain_utils/zk_certificate.py`.

### 2.5: If you use the Rust crate directly

If your pool links against the `zk-pow` Rust crate instead of the Python package,
the same logic is available in Rust:

- New V2 prover: `zk_pow::api::{prove, verify}`.
- Old V1 prover: `zk_pow::v1::api::{prove, verify}`. Note that V1 uses its own
  `IncompleteBlockHeader` and `MiningConfiguration` types, and a **separate circuit
  cache type**. Keep one cache of each kind.
- `verify_plain_proof` in both modules accepts `nbits_override` for share-difficulty
  checks.

The crossover rule and legacy parsing live in `zk_pow::ffi::plain_proof`:

```rust
use zk_pow::ffi::plain_proof::{check_cert_version_eligible, CertificateVersion, PlainProof};

// Accepts both the current format and the legacy (pre-fork) format.
let proof = PlainProof::deserialize_compat(&share_bytes)?;

// Errors for an MoE share before the fork, or an unknown version.
let zk_proof = match check_cert_version_eligible(required_cert_version, &proof)? {
    CertificateVersion::ZkDense => {
        zk_pow::v1::api::prove::zk_prove_plain_proof(v1_header, &proof, &mut v1_cache, true)?
    }
    CertificateVersion::ZkMoe => {
        zk_pow::api::prove::zk_prove_plain_proof(header, &proof, &mut v2_cache, true)?
    }
};
```

`proof.min_cert_version()` gives the lowest certificate version that can certify a
share (V1 for dense, V2 for MoE), if you want the cheap check by itself.

Reference: `py-pearl-mining/src/lib.rs` (the Python wrapper is a thin layer over
exactly these calls).

## Step 3 (optional, after the fork): Upgrade your miners

This step is only needed if you wish to mine MoE models. If you keep mining dense
models, your existing miners keep working after the fork with no changes.

New miners can produce MoE proofs. This is **optional** and must wait:

- **Do not deploy MoE-capable miners before the fork height.** An MoE share cannot be
  certified before the fork. It is wasted work, and your pool must reject it
  (`plain_proof.min_cert_version == 2` but the block requires `1`).
- After the fork, both dense and MoE shares are valid.

## Questions

Contact the Pearl team on the usual channels if anything is unclear.
