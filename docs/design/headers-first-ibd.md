# Headers-first parallel IBD (upstream btcd #2428)

**Status:** design evaluation only. Do not cherry-pick `btcsuite/btcd` PR #2428.

Upstream #2428 replaces the checkpoint-era headers-first sync with a parallel
header-then-body IBD: `ProcessBlockHeader`, a `bestHeader` tip, a new
`statusHeaderStored` index bit, and a `HaveBlock` that means “full block bytes
are available,” not “the hash is in the index.” That rewrite is the right
direction for Bitcoin Core-style IBD. It is the wrong drop-in for Pearl.

## Why a cherry-pick fails

1. **Certificate PoW is not in the 80-byte header.** Pearl serializes the zk
   certificate before a 108-byte header that carries `ProofCommitment`. Header
   messages already carry certificates (`MsgHeaders` up to ~6.5MB). Upstream
   header-only validation (PoW, difficulty, parent linkage) does not apply:
   accepting a header without the matching certificate would store an
   unverifiable commitment, and requiring the certificate makes the “header”
   path a near-full-block download. Any Pearl design must decide whether IBD
   downloads cert+header first, full blocks, or some compact cert proof.

2. **`netsync/manager.go` is Pearl-specific.** Sync-peer selection and
   announcement policy live in Pearl PRs #9 (forged version-height
   hardening), #29 (inbound fallback during IBD), #91 (per-peer quality gate
   for inv-driven download), #96 (skip peers without a higher block), and
   #101 (pinned first checkpoint, headers-first stall fix). #2428 rewrites
   that manager. A port would have to re-implement those policies on the new
   header-first state machine, not replay the upstream diff.

3. **On-disk index status is a migration.** #2428 changes what `HaveBlock` and
   the stored `blockStatus` bits mean. Pearl already has `statusDataStored` /
   `HaveData` from older btcd, but no `statusHeaderStored`, and the
   `upgrade.go` migration path was deleted. Reusing upstream’s bit layout
   without a Pearl-written upgrade would misread existing ffldb indexes.

4. **Height-constant forks, no BIP9.** Upstream IBD assumptions around
   version-bit activation and Bitcoin policy forks do not match Pearl’s
   always-on BIP30/BIP34/CSV and tapscript-only scripts.

## What a Pearl project would need

- A definition of the header-phase object: cert+header vs header-only, and
  which checks run before body download (commitment, WTEMA difficulty, parent).
- A `bestHeader` vs `bestBlock` split that cannot accept a header whose
  certificate later fails, and that cannot brick restart on a dirty header.
- An explicit ffldb status migration (Pearl owns `upgrade.go` again, or a
  one-shot reindex).
- Re-homing Pearl’s sync-peer and announcement policy onto the new manager.
- Integration tests for certificate-bearing `MsgHeaders` during IBD, not only
  Bitcoin-style 80-byte headers.

Until that design exists, keep the current checkpoint headers-first path and
the post-2026 btcd p2p/RPC/crypto ports absorbed elsewhere.
