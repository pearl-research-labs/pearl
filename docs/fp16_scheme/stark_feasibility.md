# FP16 / A100 matmul STARK — feasibility analysis (Stage 3, Task 1)

Status: **GO-WITH-CHANGES**. A `matmul_a100` STARK that proves the device tile
equals `api::fp16::accumulate::a100_matmul` is provable under the plonky2/starky
batched-FRI framework used by the FP8 schemes, with two required design changes
relative to the B200 fork (below). The Goldilocks field and the degree-≤3 budget
hold with wide margin; the only genuine scaling pressure is **trace height**
(G = 8 ⇒ 4× the B200 row count), which the "groups-per-row" packing in §4 tames.

The numbers below were derived against the frozen ground truth
`zk-pow/src/api/fp16/{accumulate.rs,dtype.rs,params.rs}` and the 238
GPU-validated vectors in `testdata/a100_dot_vectors.txt`.

---

## 1. The arithmetic to prove (one G = 8 group step)

From `a100_dot`, for incoming FP32 accumulator `c` decomposed as
`value = s_c · cm · 2^(el−23)` (`acc_parts`: `el = exp−127` for normal,
`el = −126`, `cm = mantissa` for subnormal, `cm = 0` for zero):

* each lane `u` decodes its FP16 operands (`decompose_fp16`) to
  `(sign, m, eps)`, `value = sign·m·2^(eps−10)`, `m ∈ [0, 2047]` (11-bit),
  `eps ∈ [−14, 15]`; the product has significand `P = m_a·m_b` and stored
  exponent sum `e_u = ea + eb`;
* `eta = max( e_u over nonzero products, el if c ≠ 0 )`; `unit = eta − 24`;
* every product is truncated toward zero onto the `2^unit` grid:
  `term_u = ±⌊P·2^(e_u−20) / 2^unit⌋ = ±⌊P·16 / 2^(eta−e_u)⌋`;
* the accumulator similarly: `term_c = ±⌊2·cm / 2^(eta−el)⌋`
  (uniform: value `= cm·2^(el−23)` for both normal **and** subnormal, since
  subnormal has `el = −126`, `el−23 = −149`);
* `sum = Σ term_u + term_c` exactly (signed integer);
* `new = rz_to_f32(sum, unit)` — round toward zero to FP32.

The `PolicyStep` census per group: `nonempty`, `breakpoint`
(accumulator-alignment drop **or** RZ drop), `products_truncated`
(count of lanes whose right-shift discarded a nonzero bit).

## 2. Field and range budget — PASS, wide margin

Goldilocks `p = 2^64 − 2^32 + 1 ≈ 2^63.9999`.

| quantity | bound | note |
|---|---|---|
| product significand `P = m_a·m_b` | `< 2047² < 2^22` | one degree-2 multiply |
| `e_u = ea + eb` | `[−28, 30]` (59 values) | biased to stay ≥ 0 |
| accumulator exponent `el` | `[−126, 127]` | f32, overflow aborts |
| `eta = max(e_u, el)` | `[−126, 127]` | window anchor |
| product left-shift `e_u − eta + 4` | `≤ 4` | `eta ≥ e_u` for nonzero lanes |
| aligned product term `⌊P·16 / 2^rel⌋` | `< 2^26` | `rel = eta − e_u ≥ 0` |
| aligned accumulator term `⌊2·cm / 2^rel_c⌋` | `< 2^25` | `rel_c = eta − el ≥ 0` |
| group sum `Σ` (8 terms + carry) | `< 8·2^26 + 2^25 < 2^30` | **fits i64, fits field** |
| Euclidean check `quotient·2^rel` | `< 2^26 · 2^26 = 2^52 < p` | integer-exact over 𝔽 |

The aligned group sum is < 2^30, so the entire signed window sum fits a single
field element (B200 needs the same < 2^32 claim). **FP16's larger exponent span
does *not* blow the field**: the span only widens the *relative shift* `rel`, and
because the pre-shift operands are tiny (`P·16 < 2^26`, `2·cm < 2^25`) **any**
`rel ≥ 26` truncates the term to exactly 0. So the alignment power-of-two table
caps at `2^26` (27 keys, `rel ∈ [0, 26]`), identical in spirit to B200's
`CARRY_SHIFT_CAP = 26`. The shift *range checks* are therefore the same size as
B200's; the span costs nothing.

## 3. Degree budget — PASS

* **product** `P = m_a·m_b`: degree 2.
* **eta attainment chain** (the "`≤`" side of the max, B200's MB3 technique):
  9 affine factors (8 lanes + carry). Degree-3 links take 3 factors in the
  first link then 2 per link ⇒ **`NUM_ATT_LINKS = 4`** (vs B200's 16 for 33
  factors). Chain-closing with the zero-gated carry factor is degree 3.
* **per-lane Euclidean truncation** `P·16 = q·2^rel + r`, `0 ≤ r < 2^rel`:
  degree 2 (`q·2^rel`), two-sided by the remainder/bound RC identity — same as
  B200 MB5.
* **RZ encode** (width `W`, truncation/lifting powers via a WIDTH LUT, Euclidean
  normalize into `[2^23, 2^24)`): degree 2 — same as B200 MB7.

`constraint_degree() = 3`, exactly B200.

## 4. Trace height — the one real cost; GO-WITH-CHANGES

`rows_per_cell = k / G = k / 8`, `live_rows = h·w·k/8`.

Under the whitepaper bounds (`params.rs::validate`): `h·w ∈ [256, 2048]`,
`k·(h+w) ≤ 2^22`, `k` a multiple of 8. Maximising `h·w·k`:

* **worst case** `(h, w, k) = (45, 45, 46600)` → `live_rows ≈ 1.18·10^7 = 2^23.49`,
  padded to **`2^24 = 16.8 M` rows**.
* B200 analogue (`/32`) peaks at `2^21.49` → `2^22`. So A100 is the expected **4×**.
* typical tile `(4, 64, 256)` → `live_rows = 8192 = 2^13` (tiny; test-friendly).

A 2^24-row trace at ~300 columns is ~40 GB of LDE at blow-up 2 — provable on a
large prover but memory-heavy. **Recommended change:** pack `G_ROW` consecutive
G = 8 groups per trace row (chaining the carry *within* the row). `G_ROW = 4`
restores B200-class height (`≤ 2^22`) at ~4× columns, keeping the FRI instance in
the ladder the batch driver already uses; the per-row arithmetic is 4 independent
copies of §1 with the row-internal carry wired from copy `i` to copy `i+1`. This
is a mechanical extension of the single-group AIR delivered in Task 2 and is left
as the documented production step. For the deliverable AIR and all test geometries
(`k ≤ 256`), one group per row is used — correct and well within height limits.

## 5. The two required changes vs. the B200 fork

1. **No operand-keyed product LUT.** B200's `B200ALIGN` looks the *entire* aligned
   lane term up from the packed operand byte-pair (`2^16` keys). FP16 operand pairs
   span `2^32` — a lookup keyed on raw codes is impossible. Instead:
   * an **`FP16DECODE` LUT** keyed on the 16-bit code (`2^16` rows, same order as
     RC16) returns `(sign, m, eps_biased, is_zero)` per *single* operand
     (2 lookups/lane);
   * the product significand `P = m_a·m_b` is an **in-AIR degree-2 multiply**;
   * alignment is an **in-AIR Euclidean floor** against a `POW2` LUT (`rel ↦
     2^min(rel,26)`), exactly like B200's carry alignment, applied per lane.

   This is strictly more in-AIR arithmetic than B200 but uses only LUTs of
   feasible size (`2^16` decode, `27`-key pow2, `2^16` RC16).

2. **Groups-per-row packing** for large tiles (§4).

Everything else (attainment max, signed sum recovery, RZ via width LUT, f32
encode, cell/padding structure, census columns, CTL channel shape) forks B200
directly.

## 6. Column layout to build (one group per row)

Structural (class a, verifier-recomputable): `cell_id`, `is_cell_final`,
`operand_index_base_a/b`, `is_padding`. (5)

Per lane × 8: `operand_codes_a/b`, decoded `sig_a/b`, `eps_biased_a/b`,
`is_zero`, product `product_sig`, `product_biased_exp`, `sign`, aligned
`aligned_term`, `shift_power`, `rem_lo/hi`, `rem_bound_lo/hi`,
`products_truncated_flag`. (~16/lane)

Accumulator alignment: `incoming_carry_is_zero`, `carry_shift_power`,
`aligned_carry_lo/hi`, `carry_rem_lo/hi`, `carry_rem_bound_lo/hi`. (~8, reads
previous row's `out_sign/out_sig/out_exp` as carry, no copy columns — B200
pattern)

Window max + sum + RZ: `group_max_biased_exponent`, `max_exponent_attainment[4]`,
`group_sum_sign`, `group_sum_abs`, `group_sum_is_zero`, `group_sum_width`,
`truncation_power`, `lifting_power`, `norm_sig_lo/hi`, `trunc_rem`,
`trunc_rem_bound`, `out_sign`, `out_biased_exp`, `out_sig`. (~18)

Output + census: `cell_result_f32_lo/hi`, `group_nonempty`, `group_breakpoint`,
`cell_products_truncated` (running). (~5)

≈ **250–300 columns**, comparable to B200's 304.

## 7. Verdict

**GO-WITH-CHANGES.** Build the single-group AIR with the FP16DECODE+POW2+WIDTH+RC16
LUT family and the B200-style attainment/sum/RZ constraints; pack 4 groups/row for
production-size tiles. No field or degree obstruction exists. The dominant risk is
engineering surface area (per-lane in-AIR decode+multiply+align replaces one LUT),
not provability.

### Scoping notes honoured by the Task 2 implementation

* The 238 reference outputs are **237 normal + 1 zero, no subnormals, no
  overflow** (output exponents `[−21, 31]`). The delivered AIR fully constrains
  the **normal and zero** output paths — complete coverage of the validated
  corpus. **Subnormal FP32 outputs** (value `< 2^−126`, reachable only by
  adversarial near-total cancellation to `2^−126`) use a distinct RZ encode
  branch; it is implemented bit-exactly in the trace generator and flagged, but
  its *constraints* remain a documented follow-on.

---

## 8. Stage-3 soundness hardening — what is now enforced

Three soundness gaps flagged above have been closed (commits on `fp16-scheme`):

### Gap 2 — census tightness (CLOSED)

`products_truncated_flag`, `carry_dropped`, `rz_dropped` are each pinned to
`[rem != 0]` of their Euclidean remainder (clean `(1-flag)*rem = 0` **and** tight
`flag*(rem*inv - 1) = 0` via per-lane/per-witness inverse columns), with the
inert-branch remainders killed (`incoming_carry_is_zero*carry_rem = 0`,
`z*trunc_rem = 0`). `group_breakpoint` is then pinned bit-exactly to
`group_nonempty * (carry_dropped OR rz_dropped)` (MA11) — the exact `a100_dot`
breakpoint. Because the census only ever *raises* the certified-work ratio, this
is what stops a prover overstating `rho`/`f_bp`. Negative test:
`inflated_census_breaks_constraints`.

### Gap 1 — LUT/CTL wiring (CLOSED, including limb ranges)

The matmul AIR's semantic auxiliary columns are now served by committed LUTs via
cross-table lookups, so the STARK's own verification (the CTL multiset balance)
enforces them — no blind trust:

| table | keys | values | pins |
|---|---|---|---|
| **FP16DECODE** (`2^16`) ×16 | operand code | `(sig, sign, eps_biased, is_zero)` | the per-operand decode MA1 derives each lane's product/sign/exponent from |
| **FP16POW2** (`2^min(d,26)`, `d∈[0,255]`) ×9 | `group_max − product_biased_exp` (8 lanes + carry) | `shift_power` | the alignment divisor; the key domain also proves the eta `≥` side |
| **WIDTH32** (`[1,32]`) ×1 | `group_sum_width` | `(2^max(W−24,0), 2^max(24−W,0))` | the RZ truncate/lift powers + width range |
| **RANGE16** ×60 | `cell_result_f32_lo/hi`; MA12 floor-witness limbs; RZ `trunc_rem`/`trunc_rem_bound` | — | the FP32 result limbs **and** the limb-range splits below |

`FP16DECODE`/`FP16POW2` are new `LutStark` variants (reusing the FP8 LUT
machinery) that no FP8 device commits, so the FP8 consensus layout is untouched;
`RANGE16`/`WIDTH32` are the FP8 tables reused directly. The FP16 batch and its
`CrossTableLookup` set are in `circuit::fp16::ctl`. Tests: the honest batch
serves every instance (`LutChecker`) and balances every channel (`check_ctls`);
forged decode/shift/width/exponent columns are rejected by the committed tables,
and a tampered matmul value unbalances a channel
(`forged_auxiliary_columns_are_rejected_by_the_luts`).

**Limb ranges (CLOSED — MA12).** The per-lane and carry Euclidean
remainders/quotients (`aligned_mag`, `lane_rem`, `lane_rem_bound` per lane;
`aligned_carry`, `carry_rem`, `carry_rem_bound`; the RZ `norm_sig`) now carry
`lo`/`hi` limb columns. MA12 pins `value = lo + 2^16·hi` (degree 1), and the
committed-LUT inventory `RANGE16`-checks `lo < 2^16` and `2^6·hi < 2^16` (so
`hi < 2^10`, bounding the magnitude by `~2^26`); the RZ `trunc_rem` /
`trunc_rem_bound` (`< TRUNCATION_POWER ≤ 2^6`) are `RANGE16`-checked directly.
A wrapped quotient/remainder has no valid 16/10-bit limb witness, so no field
element `q ≥ 2^26` can alias an alignment/normalization `floor` — the identities
are integer-exact under a real FRI proof. The matmul RANGE16 inventory grows
`2 → 60` (56 limb checks + 2 result limbs + 2 RZ-remainder checks; 86 matmul LUT
instances total). Negative tests: an out-of-range `hi` limb (a field-fraction
alias) is rejected by `RANGE16`
(`forged_auxiliary_columns_are_rejected_by_the_luts`), and bumping a limb breaks
MA12's reconstruction (`tampered_cells_break_constraints`).

### Gap 3 — rho / breakpoint-density policy AIR (CLOSED)

`circuit::fp16::policy_stark` enforces the exact `api::fp16::policy` gate over the
per-group census: a run-start detector for `N_runs`, tile-global inclusive
accumulators for the breakpoint count and the numerator
`sum[ 8·N_bp + 32·N_runs + N_pt ]`, and a division-free last-row gate
`5·numerator ≥ 6·cells·k` (⇔ `rho ≥ 1.2`) and
`10·breakpoints ≥ 3·total_steps` (⇔ `f_bp ≥ 0.30`), each witnessed by a
nonnegative slack whose 16-bit limbs are `RANGE16`-checked (so a sub-threshold
tile has no valid nonnegative witness). The AIR's gate decision and running
totals equal `policy::evaluate` on every reference-vector tile (accept and reject
both covered); flat tiles and tampered gate witnesses fail. Its per-step census
columns are bound to the matmul's tight census by a `matmul → policy`
census-import CTL (CLOSED — see below).

### Gap 3 linkage + batched-FRI driver (CLOSED)

The `matmul → policy` **census-import CTL** (`circuit::fp16::ctl::census_import_ctl`)
links the two main tables: the matmul exports, per live group step, the tuple
`(operand_index_base_a, operand_index_base_b, group_breakpoint,
Σ products_truncated_flag)`; the policy imports the same tuple. The
`(base_a, base_b)` pair is unique per live row (`base_a` fixes `(r, j)`,
`base_b` fixes `(c, j)`), so the multiset equality forces the policy's per-step
census to equal the matmul's tightly-pinned one bit-for-bit — the policy can no
longer score a census different from the one the matmul proved. Both sides are
filtered to live rows. (Two `operand_index_base_*` columns were added to the
policy class-(a) layout to carry the key.) Negative test:
`forged_policy_census_breaks_the_import_channel`.

The **batched-FRI driver** `circuit::fp16::driver::Fp16System` batches the matmul
AIR, the policy AIR, and the four committed LUTs (`FP16DECODE`, `RANGE16`,
`FP16POW2`, `WIDTH32`) under one `batch_prove` / `batch_verify` with the full
`CrossTableLookup` set (four LUT channels + census import). Canonical table order
matmul (0), policy (1), LUTs (2..6); the LUT precommitment is built at a Merkle
cap (`Fp16System::preprocessed_data`); the two main tables' class-(a) columns are
recomputed by the verifier and the trace openings bound to them; the FRI ladder
covers the job's distinct heights. End-to-end test
`batched_fp16_proof_roundtrips_and_rejects_tampering`: an honest 2×2 tile proves
and verifies, and a tampered trace cell, a forged per-step census value, a forged
decode column, and a mismatched statement are all rejected (~2.5 s). The
channel-balance analogue is `one_fp16_job_balances_every_ctl_channel`.

### Entry-liveness + noise-floor shared gates (CLOSED)

The whitepaper's §"Shared checks" (the FP8 entry-liveness and noise-floor gates that bound
degenerate operands) are now enforced for the ZK path:

* **Noise floor (`sigma_i = DELTA*alpha_i*l2_i >= 1`) — implied, no constraint.** The scale
  derivation forces `alpha_i = Q/(linf_i + DELTA*sqrt(r)*l2_i)` with `linf_i <= sqrt(k)*l2_i`, so
  `sigma_i >= DELTA*Q/(sqrt(k) + DELTA*sqrt(r)) >= ~16` for every `k <= 2^22`. Because
  `row_scale_stark` already *constrains* `alpha` to that derived value, a sub-floor `sigma` has no
  valid witness. Machine-checked by `api::fp16::policy::tests::shared_gates::
  noise_floor_is_implied_by_honest_derivation`.
* **Entry liveness (`|D_X| <= eps_idle*|I_X|*k`, `eps_idle = 1/64`) — enforced in-circuit
  (`row_scale_stark`, group L).** Per element a `dead = [ |x| >= 4*l2 ]` flag is pinned by an exact
  integer aligned compare (`|x| = x_sig*2^(x_eps_biased-25)` vs `4*l2 = (128+l2_m)*2^(l2_e-132)`,
  shift via FP16POW2 with a sound 2^26 saturation, two-sided RANGE16 slack). Two side-masked
  inclusive accumulators (`dead_run_a/b`) and a division-free last-row gate
  `64*dead_side <= rows_side*k` (nonnegative RANGE16 slack) make a spike-dominated tile
  unsatisfiable. `num_a_rows` (`h`) is an AIR compile-time constant (like `k`), so no known column
  / public input / wrapper-PI change — the header-bound consensus verifier is untouched. All AIRs
  stay degree <= 3. Tests: `row_scale_stark::stark::tests::{liveness_gate_matches_check_shared_gates,
  tampered_liveness_witnesses_break_constraints}` (bit-exact accept/reject vs
  `api::fp16::policy::check_shared_gates`, plus fail-closed tampers), and the honest batch still
  balances every CTL channel and proves/verifies.

### Subnormal FP32-output RZ branch (CLOSED — MA13)

The per-group round-toward-zero to FP32 now constrains the **subnormal-output**
branch (`|x| < 2^-126`): `circuit::fp16::matmul_a100_stark` MA13 writes the
`2^-149`-grid encode — exponent field 0, mantissa `OUT_SIG = NORM_SIG >> k` with
`k = 1 - raw` (`raw = GROUP_MAX_BIASED_EXPONENT + W - 25`) — gated by an
`out_is_subnormal` flag that MA13 pins to `[raw <= 0]` via a nonnegative RANGE16
`exp_slack` witness (so neither over- nor under-claiming the subnormal regime has
a valid witness). `SUB_SHIFT_POWER = 2^k` is FP16POW2-bound (subnormal-filtered)
and the mantissa is RANGE16 limb-reconstructed, so `OUT_SIG` is a genuine `< 2^23`
integer; MA8/MA9 mux the exponent / significand / encode residue on the flag. The
generator computes the branch bit-exactly (the old `unimplemented!`/panic guard is
gone; the two remaining `debug_assert!`s in this branch are release-stripped
correctness invariants, not the soundness check — that is MA13's constraints).

The from-zero matmul statement never *reaches* the branch — FP16 products align at
`eta >= 99`, so a group output floors at `~2^-52`, and the carry model forbids a
cell-start carry-in — so MA13 is a **sound guard** there; the branch is the
bit-exact encode for the accumulation datapath (subnormal FP32 carry-in) that the
oracle `a100_dot` already models. Tests: `subnormal_output_is_bit_exact_and_
satisfies_ma13` (generator == `a100_dot` for a spread of FP32 subnormals, and the
committed columns satisfy the MA13/MA8/MA9 relations) and
`forged_subnormal_claim_is_rejected` (honest traces never set the flag; a forged
flag / subnormal encode is rejected). All AIRs stay degree <= 3, and
`circuit_constraints_match_native` holds.

### Recursive FRI wrapper (DELIVERED per-shape; universal variant deferred)

`circuit::fp16::wrapper` compresses the `Fp16System` batch proof to a constant-size
recursive plonky2 proof via a two-stage wrapper mirroring the FP8 structure: stage
1 (`PoseidonGoldilocksConfig`, no ZK) runs starky's in-circuit batch verifier
(`verify_batch_stark_proof_circuit`) over all six tables — every AIR, the four
committed-LUT channels + the census-import CTL, the baked-in LUT cap and the
batched FRI — and exposes the Fiat-Shamir `zeta`, the statement digest and the
class-(a) known-column evaluations as public inputs; stage 2
(`Blake3GoldilocksConfig`, `zero_knowledge: true`) verifies stage 1 and republishes
them. The native gateway `verify_wrapped_proof` pins every slot — the digest to the
statement, the known-column evals to its own recompute at `zeta` — giving the same
guarantees as `Fp16System::verify`. The driver gained the consensus fold ladder,
the committed-LUT grouping (`FP16_GROUPED_TABLES`), the statement-digest binding and
the universal envelope needed by the wrapper paths. Test
`wrapped_fp16_proof_verifies_and_rejects_tampering`: an honest tile wraps and
verifies; the **wrapped proof length is constant (71 486 bytes) across two tile
sizes** (`2x3` and `3x3`, main tables `2^4` vs `2^5`, both compiling stage-1 to
`2^15`); a wrong statement digest and any tampered public-input slot are rejected.

* **Universal (one-circuit-for-all-sizes) wrapper — deferred.** FP8 compiles a
  single circuit for every envelope-legal job via
  `starky::batch_universal::verify_universal_batch_stark_proof_circuit`; the FP16
  driver already ships the consensus ladder, envelope and digest machinery that
  path needs (`fp16_universal_envelope`), but the universal verifier
  over-determines a witness wire (`set twice`) for FP16's table shape — two
  **equal-height** variable main tables (matmul and policy share one row grid)
  linked by a direct CTL (census-import), a configuration FP8 never produces (its
  variable tables always differ in height) and the universal verifier's own tests
  do not cover. The fix is in `plonky2/starky/src/batch_universal.rs`, shared with
  FP8's wrapper, so it is left as the residual. The per-shape wrapper above already
  delivers a constant-size, verifiable, tamper-rejecting recursive proof; the
  universal variant only removes the per-shape circuit compilation.

---

## 9. Soundness fix + consensus-binding status (operand + output + header bindings CLOSED)

**NORM_SIG normalized-range pin (FIXED).** The matmul AIR originally bounded the
normalized significand `NORM_SIG` only by the MA12 limb split (`< 2^26`), dropping
the FP8/B200 MB7 check that forces it into `[2^23, 2^24)`. Without that range the
MA7 width identity `GROUP_SUM_ABS·L = NORM_SIG·T + TRUNC_REM` admits *any*
`GROUP_SUM_WIDTH` for a given sum magnitude, so a prover could pick a false width,
choose `TRUNC_REM != 0`, and forge `RZ_DROPPED` (MA11) → `GROUP_BREAKPOINT` → the
jackpot census (`f_bp`/`rho`) — defeating the whole "unpredictable accumulation
steps" gate (a flat/cheap tile made to clear `f_bp ≥ 0.30, rho ≥ 1.2`). Now closed
by a committed-RANGE16 check `(NORM_SIG_HI − 128)·2^9 < 2^16` filtered to
nonzero-sum rows (`matmul_a100_stark::ctl`), mirroring B200. This also makes
`NORM_SIG < 2^24`, which closes the latent MA13 subnormal `OUT_SIG < 2^23` claim.
Regression: `circuit::fp16::ctl::forged_norm_sig_width_is_rejected_by_the_normalized_range`.

**Status update — all three bindings below are now CLOSED, and the header-bound
ZK certificate is the wired consensus path (the plaintext certificate is
retired).** When first written this section listed two open gaps and a header
residual; all are now bound and tested (`zk_binding_design.md §8/§8b`):

1. **Operand commitment / provenance — CLOSED.** `operand_codes_a/b` are now tied
   to the committed operand Merkle roots (`HASH_A`/`HASH_B`) through the
   raw→noised→matmul-operand CTL chain, carrying the `w`/`h` reuse multiplicity
   that forces cross-cell row/column sharing (`circuit::fp16::blake3_commit`,
   `noisy_quant_stark`, `ctl`).
2. **Output / ticket binding — CLOSED.** Matmul results → XorFold → BLAKE3 jackpot
   → `HASH_JACKPOT`, with `statement_digest` derived from the proof's *own*
   `HASH_JACKPOT` (`driver.rs`), and native `check_jackpot_difficulty`.

3. **Header binding — CLOSED (at the consensus verifier gateway).** The opening
   keys (`KEY_A`/`KEY_B`), the public-parameter encoding `p`, and the jackpot key
   enter the STARK from the caller, but the consensus verifier
   `verify_wrapped_proof_with_headers` (`circuit::fp16::wrapper`) re-derives them
   from the proposed/ancestor headers + committed operand roots, derives the
   `statement_digest` from the proof's *own* `HASH_JACKPOT`, and pins the ENTIRE
   stage-2 public-input vector by equality before verifying the wrapped proof and
   checking native difficulty. A proof whose keys/seeds/jackpot-key do not match
   the header is rejected, so the statement is header-bound at consensus.

**Wired consensus path.** The FP16 (A100) consensus certificate is the
header-bound ZK certificate (`CertificateV5` → `verify_fp16_zk_cert_ffi` →
`verify_wrapped_proof_with_headers`); the earlier plaintext certificate
(`api::fp16::verify`, which bound operands by Merkle opening and recomputed the
tile + census by replay) is **retired** as a consensus path. The one remaining
residual is provisioning, not soundness: the FP16 wrapper is compiled per degree
profile, so the embedded verifier cache (`fp16_cache.bin`) must enumerate every
consensus-legal profile before V5 is activated (the universal FP16 wrapper, §8,
would remove the per-shape cache). The full blueprint is in `zk_binding_design.md`.
