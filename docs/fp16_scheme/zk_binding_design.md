# FP16 ZK proof — consensus-binding design (closing the two unbound gaps)

Status: **IMPLEMENTED (operand + output + noise binding in-circuit; header
binding at the consensus verifier gateway); the header-bound ZK certificate is
the wired consensus path.** This records how the FP16 batched ZK proof
(`zk-pow/src/circuit/fp16/`) is bound to the consensus statement. The two
originally-open gaps below — operand provenance / cross-cell sharing and
output/ticket binding — are now closed and tested in-circuit (§8/§8b). Header
binding is enforced at the consensus verifier: the opening keys (`KEY_A`/`KEY_B`),
the public-parameter encoding `p`, and the jackpot key enter the STARK from the
caller, but `verify_wrapped_proof_with_headers` (`circuit::fp16::wrapper`)
re-derives them from the proposed/ancestor headers + committed operand roots,
derives the `statement_digest` from the proof's own `HASH_JACKPOT`, and pins the
whole public-input vector by equality before verifying — so the statement is
header-bound at consensus. **The wired consensus path is the header-bound ZK
certificate** (`CertificateV5` → `verify_fp16_zk_cert_ffi`); the earlier plaintext
certificate (`api::fp16::verify`) is retired. The remaining residual is
provisioning, not soundness: the per-shape wrapper needs an embedded verifier
cache covering every consensus-legal degree profile (or the universal wrapper,
§8) before V5 is activated.

## 1. What the circuit proves today, and the gap

The batch proves the *arithmetic*: the matmul AIR (`matmul_a100_stark`) proves
each output cell is the A100 `a100_dot` of its row's `operand_codes`, and the
policy AIR proves the jackpot gate over the matmul's tightly-pinned census. It
does **not** bind that computation to the consensus statement:

1. **Operand provenance / cross-cell sharing (Finding 3).** `operand_codes_a/b`
   are free witness. `FP16DECODE` only checks each code *decodes*; nothing ties a
   code to the committed operand Merkle root, and nothing forces cells `(r,c1)`,
   `(r,c2)` of one output row to reuse the same `A[r,:]` (or a column to reuse
   `B[:,c]`). The AIR proves per-cell dot products over *arbitrary* operands.
2. **Output / ticket binding (Finding 4).** The output tile (`cell_result_f32_*`)
   and census totals are not public inputs. `Fp16System::bind_statement_digest`
   stores a **caller-supplied opaque** digest; the two main AIRs have **zero**
   public inputs. The circuit never constrains the digest to commit the operands
   or the tile, so a verified proof attests "some policy-passing tile of this
   geometry exists," not "this block's committed matmul."

### 1a. The noise subtlety the FP8 mechanism does not cover

In the real scheme the matmul is **not** over the committed rows. The plaintext
verifier (`api::fp16::verify::verify_tile`) does:

```
raw committed rows  --(noise derived from the committed root)-->  N = E @ F^T
noised = Q(alpha*raw + beta*N)   (per-row scales; f32 FMA; FP16 cast)
tile   = a100_matmul(noised_A, noised_B)
```

So `operand_codes` fed to the matmul must equal the **noised** operands, which are
a deterministic function of (committed raw codes, seed-derived noise). A correct
binding therefore cannot connect the matmul operands directly to the commitment
(as the FP8 single-hop analogy suggests). It must prove the whole chain
`committed raw -> noise -> noisy_quantize -> noised`, and the noise must itself be
bound to the committed root — otherwise a prover grinds the noise draw. FP8 hides
its noise inside `input_quant_stark`'s FMA; FP16 dropped that table and has **no**
noise stage in-circuit yet. These are exactly the `FP16 noisy-quant / noise-line /
tensor-hash` pieces the PR lists as deferred follow-ons.

## 2. FP8 reference architecture (what to mirror)

FP8 batch = 5 main AIRs + 15 LUTs (`circuit/fp8/ctl.rs:58-67,95`). Binding-relevant:

| FP8 table | role | FP16 relevance |
|---|---|---|
| **Blake3** (`blake3_stark`) | keyed-BLAKE3 Merkle roots over committed bytes → `HASH_A/HASH_B` PIs; folds XorFold's 16 lottery words → `HASH_JACKPOT` PI | **reuse (simplified)** |
| **InputQuant** (`input_quant_stark`) | int8+bf16-scale decode, QCAST, **noise FMA** | **replace** with an FP16 noisy-quant table (different math; no int8/scale) |
| **Scale** (`scale_stark`) | prequant L2 norm / σ chain | **drop** (FP16 commits raw u16, no prequant) — but the FP16 noisy-quant needs its own `row_norms`/`derive_row_scales` sub-logic |
| **Matmul** (`matmul_h100`/`matmul_b200_stark`) | the dot products | **have it** (`matmul_a100_stark`) |
| **XorFold** (`xor_fold_stark`) | f32 cell results → 16 lottery lane words | **reuse (~verbatim)** |

Binding CTL channels (`circuit/fp8/ctl.rs:160-170`):
- **operand codes**: `InputQuant (looked) <-> Matmul (looking)`, keyed by element
  index; **cross-cell sharing is enforced by a multiplicity on the looked side** —
  A elements carry multiplicity `w`, B elements `h` (`input_quant_stark/ctl.rs:
  121-157`). Cells in a row emit the same `operand_index_base` key, so differing
  per-cell operands cannot balance the single multiplicity-`w` looked tuple.
- **results**: `Matmul (looked) <-> XorFold (looking)`, tuple `(cell_id,
  result_lo, result_hi, skips)`, filter `is_cell_final*(1-is_padding)`.
- **lottery words**: `XorFold (looked) <-> Blake3 (looking)`, 16 `(lane_id,
  fold_out)` tuples → the jackpot block Blake3 hashes to `HASH_JACKPOT`.

Statement binding (`fp8/driver.rs:805,852,873`): `HASH_A/HASH_B/HASH_JACKPOT` are
Blake3 public inputs; `verify` forces `proof.public_inputs ==
batch_public_inputs(expected)` where the caller supplies the header-derived roots
and jackpot hash; difficulty is a **native** epilogue `check_jackpot_difficulty(
HASH_JACKPOT, nbits, h, w, k)` (`api/proof_utils.rs:123`). `statement_digest` is a
separate FS salt derived from the proven jackpot hash.

## 3. The full FP16 binding chain to build

```
            keyA = H(proposed_header)        keyB = H(ancestor_header)
                       |                               |
  committed raw A rows (u16) --Blake3 Merkle--> HASH_A        HASH_B <-- raw B rows
                       |  (root -> seed chain, B then A)      |
                       +------------- seed_A, seed_B ---------+
                                       |
        noise-line (BLAKE3 XOF -> normalize: isqrt + bf16 div) -> E_A,F_A,E_B,F_B
                                       |
                 noise matmul  N = E @ F^T   (an a100_matmul instance)
                                       |
   noisy-quant:  noised = Q(alpha*raw + beta*N)   (row_norms, derive_row_scales
                                       |            in bf16; f32 FMA; FP16 cast)
                        main matmul  tile = a100_matmul(noised_A, noised_B)
                                       |                         |
                               policy gate (census)        XorFold -> 16 lane words
                                                                 |
                                            Blake3 jackpot hash -> HASH_JACKPOT
                                                                 |
                               native: check_jackpot_difficulty(HASH_JACKPOT, nbits, h,w,k)
```

Every arrow is a CTL (or a committed-root/public-input equality). The security
load-bearing links: raw↔root (provenance), seed↔root (anti-grind noise),
noise↔noised↔matmul-operands (the noised codes are what gets multiplied), tile↔
ticket↔difficulty (work actually counts), and root/jackpot↔header PIs.

## 4. New components

**New STARKs (FP16 batch grows 6 → ~5 main + LUTs; update `NUM_FP16_MAIN_TABLES`,
batch order, `batch_public_inputs`):**

1. `fp16/blake3_stark` — mirror `fp8/blake3_stark`, **simplified**: one keyed
   Merkle root per operand over u16 LE rows (no second scales tree, no two-plane
   `operand_digest_fp10` fold, drop routing/offsets PIs unless MoE is in scope);
   plus the lottery-words → `HASH_JACKPOT` compression. PIs: `KEY_A`, `KEY_B`,
   `POW_KEY`, `HASH_A`, `HASH_B`, `HASH_JACKPOT`. Must also expose the seed-chain
   outputs (or a dedicated small AIR) so the noise stage keys off the proven root.
2. `fp16/noise_stark` (**net-new, no FP8 analogue**) — proves the noise-line draw:
   per factor, keyed-BLAKE3 XOF of the `side|factor|line` address under the seed,
   then `normalize_line` (signed magnitudes → integer `isqrt` → one bf16 division →
   FP16 cast). Mirrors `api/fp16/noise.rs` bit-for-bit.
3. `fp16/noisy_quant_stark` (**net-new**, replaces FP8 `input_quant_stark`) —
   proves `noised = Q(alpha*raw + beta*(E@F^T))`: `row_norms` (L2/Linf in bf16,
   grid-rounded), `derive_row_scales` (`alpha,beta` in bf16), the f32 FMA, the FP16
   round, and the `[-MAX_FP16, MAX_FP16]` clamp. Mirrors `api/fp16/quantization.rs`.
   The `N = E@F^T` can be a second instance of `matmul_a100_stark` (it is literally
   `a100_matmul(E, F)`), avoiding a bespoke AIR.
4. `fp16/xor_fold_stark` — mirror `fp8/xor_fold_stark` ~verbatim (f32 cell results
   → 16 lottery words; RC16 limbs already shared).

**New CTL channels** (add to `fp16/ctl.rs::all_cross_table_lookups`, currently 5):
- `Blake3 (looked, raw bytes) <-> NoisyQuant (looking)` — binds raw committed u16
  to the noisy-quant input (value = `byte_lo + 2^8*byte_hi`; stride 1, no pairing).
- `NoiseStark (looked E/F) <-> {noise-matmul, NoisyQuant} (looking)` — binds the
  noise factors into N and into the `beta*N` term.
- `NoisyQuant (looked noised codes) <-> Matmul (looking)` — the **operand** channel;
  carries the **multiplicity `w`/`h`** (A reused in `w` cells, B in `h`) exactly as
  `input_quant_stark/ctl.rs:121-157`, closing provenance **and** cross-cell sharing.
- `Matmul (looked results) <-> XorFold (looking)` — the results channel.
- `XorFold (looked lane words) <-> Blake3 (looking)` — the lottery-words channel.
- seed-chain binding: root PIs → noise seeds (CTL or in-Blake3 constraint).

**Driver/verify wiring:** publish `HASH_A/HASH_B/HASH_JACKPOT` (+ keys) as batch
PIs (`fp16/driver.rs` `batch_public_inputs`), force them equal to header-derived
expected PIs in `verify` (mirror `fp8/driver.rs:852`), derive the FS salt from the
proven jackpot hash (mirror `fp8/driver.rs:805`), and keep difficulty native via
`check_jackpot_difficulty`. This is what upgrades `statement_digest` from opaque to
statement-committing.

## 5. Reuse vs net-new

- **Reuse ~verbatim:** `xor_fold_stark`; the Blake3 hashing engine, byte-pair
  message channel, and commit-fold wrapper; the LUT oracle machinery (already
  shared); the `statement_digest`/known-column/wrapper plumbing (already present in
  `fp16/driver.rs`); `check_jackpot_difficulty`, `xor_fold_extract`,
  `compute_jackpot_ticket` (native, scheme-neutral).
- **Net-new (no FP8 analogue):** `noise_stark` (BLAKE3-XOF + normalize) and
  `noisy_quant_stark` (bf16 scale derivation + f32 FMA + FP16 cast). These are the
  deferred `noise-line` / `noisy-quant` pieces and carry the most new constraint
  surface (bf16 arithmetic, `isqrt`).
- **Drop:** `scale_stark`, the int8 path of `input_quant_stark`, QCAST/Div448 and
  the two-plane commitment fold.

## 5a. Blake3 reuse — UNBLOCKED and delivered (investigated, then built)

The FP8 `Blake3Stark` AIR is a general, program-driven BLAKE3 engine. The initial worry was
that `Blake3Program::from_blake_program` (`circuit/fp8/blake3_stark/stark.rs:435`) hard-asserts
the FP8 prequant four-plane shape (`A values / A scales / B values / B scales`) and panics on
anything else, and that there is no FP16 commitment→program compiler. **That turned out to be
only a limitation of the `from_blake_program` bridge, not of the engine.** The engine is
genuinely scheme-neutral at the `Blake3Instruction` / `generate_trace` level:

* `Blake3Instruction::lottery()` already exists (the jackpot compression);
* `generate_trace` already supports per-plane `HashId` leaf padding;
* a single keyed Merkle-tree root can bind **directly** to `HASH_A`/`HASH_B` (no FP8 commit-fold
  wrapper needed), keyed by `KEY_A`/`KEY_B`.

So the remediation was just §5a-item-2: a **new FP16-specific `Blake3Program` constructor** that
assembles the `Blake3Instruction` list directly, bypassing `from_blake_program`. **No shared
code was changed, and no separate commitment-program compiler was needed.** This is delivered in
`circuit/fp16/blake3_commit.rs`:

* `jackpot_program()` / `operand_tree_program()` / `fp16_blake3_program()` build the programs;
* `append_operand_tree()` compiles one operand's keyed tree (BLAKE3 chunk per leaf, keyed parent
  merges with odd-tail promotion, root on the top merge) **bit-exact with
  `pearl_blake3::MerkleTree::with_chunk_len`** for every allowed leaf size (128/256/512/1024);
* `HASH_A`/`HASH_B` match `api::fp16::commitment::commit_operand(...).root()` and `HASH_JACKPOT`
  matches `api::fp8::transcript::compute_jackpot_ticket`, both verified across shapes/sizes;
* parameterized CTL hooks (`ctl_lottery_words_looking_blake3`, `ctl_operand_bytes_looking_blake3`)
  are exposed for the batch-wiring step (counterparties: `xor_fold` and `noisy_quant`).

What remains is **only** the batch wiring (§6 step-by-step): register the table in
`circuit::fp16::ctl`, publish `HASH_A/HASH_B/HASH_JACKPOT` as driver public inputs forced equal
to header-derived values in `verify`, and wire the operand-bytes / lottery-words CTLs. Until
that, the program proves "there exist operands/lottery words whose keyed-BLAKE3 roots are these
PIs," not yet that they are *the committed* operands.

## 6. Incremental, reviewable build order

Each step is independently testable (honest-serves + tamper-rejects, like the
existing `consistency.rs` / `ctl.rs` tests) and should NOT be merged as a partial
"binding" — a half-bound proof looks bound but is not, so the chain lands behind a
single activation only once complete.

1. `xor_fold_stark` + results channel + lottery-words channel + `blake3_stark`
   jackpot-hash + `HASH_JACKPOT` PI + native difficulty. (Output/ticket binding —
   Finding 4, for the *current* raw-operand matmul.)
2. `blake3_stark` Merkle roots + `HASH_A/HASH_B` PIs + header-equality in verify.
3. `noise_stark` + `noisy_quant_stark` + noise-matmul instance + the raw↔noised↔
   matmul-operand channels with `w`/`h` multiplicity. (Operand provenance + the
   noise pipeline — Finding 3 + §1a.)
4. Seed-chain binding (root → seeds) so noise is anti-grind.
5. Wrapper: carry the new PIs; re-run the constant-size + tamper tests.

## 7. Risks / open questions

- **bf16 / isqrt in-AIR.** `normalize_line` and `derive_row_scales` use bf16
  mul/div/fma and an integer `isqrt`; these need degree-≤3 constraint designs
  (FP8's bf16 LUTs — `Div448`, etc. — are a starting point but FP16 uses different
  constants). Highest new-constraint risk.
- **Trace height.** The noise matmul `E@F^T` is `num_rows x k` by `k x r` — another
  large table; folds into the existing height budget (`stark_feasibility.md §4`).
- **MoE / routing.** Out of scope; FP16 is dense (`CertificateV5::IsMoE()==false`),
  so the routing/offsets Blake3 PIs are dropped.
- Until §6 steps 1–4 all land and the wrapper carries the PIs, the ZK proof must
  remain flagged as not-a-consensus-path (see `circuit::fp16` module docs).

## 8. Build status (implemented, by increment)

All of the FP16 ZK binding below is implemented, batched, and tested
(`cargo test --lib circuit::fp16`, 152 green, 0 ignored at the final increment §8b, incl. the recursive wrapper):

- **XorFold AIR** + **noise-line normalization AIR** + **noisy-quant AIR** (G1 mul,
  G2 single-rounding f32 FMA, G3 cast incl. subnormal/zero) + **per-row scale AIR**
  (row_norms + derive_row_scales) — all bit-exact vs their `api::fp16` ground truth,
  all ties-to-even uniquely pinned (IS_BOTTOM, no quarter-ulp grinding slack).
- **Blake3 commitment program** (operand trees → `HASH_A/HASH_B` bit-exact vs
  `commit_operand`; jackpot hash → `HASH_JACKPOT`) reusing the shared engine unchanged.
- **Output chain wired**: matmul results → XorFold → Blake3 jackpot → `HASH_JACKPOT`,
  statement digest derived from the proven jackpot, native `check_jackpot_difficulty`
  matching the plaintext decision.
- **Operand chain wired (Finding 3 CLOSED)**: committed bytes (`HASH_A/HASH_B`) → `raw`
  → quant `Q(·)` → noised → matmul operand codes, with the `w`/`h` reuse multiplicity
  that also forces cross-cell row/column sharing. Fail-closed tamper tests on every link.
- **Noise chain**: `N = E@F^T` proven (two per-side noise matmuls, distinct `F_A/F_B`
  matching the plaintext); `E/F` bound to the normalized noise lines proven by the
  noise AIR.
- A **latent G2 FMA rounding bug** (opposite-sign cancellation ties) was found and
  fixed (sign-aware sticky) when real noise first exercised it.

### 8a. The one remaining gap: seed→XOF binding (6e-3) — blocked on shared-AIR egress

The raw keyed-BLAKE3-XOF bytes that the noise AIR normalizes are still **free witness**,
and the noise seeds are fixed constants rather than derived from the committed roots. So
a prover can still **grind the noise** by choosing those bytes. Closing this requires
proving, in-circuit, `seedB=H(root_B‖keyB‖pB)`, `seedA=H(root_A‖seedB‖keyA‖pA)` and each
line's keyed XOF, then binding the XOF **output** bytes to the noise AIR's input.

This is **blocked under "reuse the shared Blake3 AIR unchanged"**: the shared
`circuit/fp8/blake3_stark` exposes a compression's **output** `cv_out` only internally
(CV-routing, keyed by row index) and to fixed public-input hash slots — there is **no
byte-decomposed, program-keyed, cross-table egress of `cv_out`**, which is exactly what
binding the per-line XOF output needs. A "seed-chain only" subset is not useful: the
seeds are a pure function of public values, but the prover grinds the XOF *bytes*
directly and never needs the seed, so only the output→noise-AIR byte binding reduces the
grindable surface.

The sound closures both have cost:
- **(A) Extend the shared Blake3 AIR** with an additive output-egress CTL channel (a new
  finalization-row readout of `cv_out`, keyed by a program base) + recompute seeds
  natively as public inputs. Smallest closure, but it changes the Blake3 **column
  layout** that FP8's *active* V4 consensus ZK path commits to — a consensus-compatibility
  change (circuit digest / trusted-setup / existing proofs) that needs explicit sign-off.
- **(B) Fork a FP16-specific Blake3 AIR** with the egress feature, leaving the FP8 engine
  byte-for-byte untouched. Consensus-safe for FP8, but ~3.6k lines duplicated.
- **(C) Leave as the documented final gap.** Everything else is bound; noise remains
  grindable (weakens anti-precompute, does not forge the matmul/output).

(Decision: option (B), the Blake3 fork — see §8b; 6e-3 is now CLOSED.)

Until (A) or (B) lands, the FP16 ZK proof binds the committed operands and the
output/ticket but not the noise derivation; it is a succinct proof of "the committed
operands, noised by *some* low-rank E@F, yield this policy-passing tile and jackpot."

### 8b. Update — 6e-3 CLOSED via the Blake3 fork (anti-grind achieved)

The §8a gap is now closed (decision: fork, not modify the shared engine):
- **6e-3a** forked the BLAKE3 AIR into `circuit/fp16/blake3_fp16_stark` with an additive
  output-egress CTL channel (cv_out → 16-bit limbs, flag-gated). FP8's engine is untouched
  (changes under `circuit/fp8` are additive-only — two new `LutTable` variants
  `Fp16Decode`/`Fp16Pow2` in `circuit/fp8/luts/`, consumed solely by the FP16
  batch; no existing FP8 table's layout/`slot_height` changes and the shared
  BLAKE3 engine is byte-for-byte untouched), fp8 tests green.
- **6e-3c** built `circuit/fp16/noise_blake3.rs` on the fork: 2 subkey compressions + the
  `h+w+2k` per-line keyed-XOF compressions (public material pinned), egressing each line's
  bytes and binding them to NoiseStark via the egress CTL — closing the free-witness-bytes grind.
- **6e-3d** derives the seeds natively from the committed roots
  (`noise_seeds(HASH_A/HASH_B, keys, p)`) and pins the noise-BLAKE3 `KEY_A/KEY_B` to them in
  `verify`, so the noise is operand-dependent and matches the plaintext — anti-grind achieved.

**The in-circuit FP16 ZK binding is now in place and tested** (`cargo test --lib circuit::fp16`:
152 green, 0 ignored, incl. the recursive wrapper): committed operand roots → seed-derived noise →
noised operands → matmul (cross-cell sharing) → policy → lottery tile → `HASH_JACKPOT` → statement
digest → native difficulty, every link fail-closed. **Header binding is enforced at the consensus
verifier gateway** (`verify_wrapped_proof_with_headers`): the opening keys and `p` are supplied to
the STARK by the caller, but the verifier re-derives KEY_A/KEY_B/jackpot-key + noise seeds from the
proposed/ancestor headers and committed roots and pins the whole public-input vector by equality,
so the statement digest is header-bound at consensus. The header-bound ZK certificate (`CertificateV5`)
is the wired consensus path; the plaintext certificate is retired. **One residual remains, and it is
provisioning not soundness:** the per-shape wrapper needs an embedded verifier cache covering every
consensus-legal degree profile (or the universal wrapper, §8) before V5 is activated on a network.
