//! The FP16 / A100 batch driver: one batched-FRI proof for the whole thirteen-table system.
//!
//! [`Fp16System`] mirrors [`crate::v4::circuit::driver::Fp8System`] at FP16's (much smaller)
//! scale. It batches the two main AIRs — the A100 `HMMA` matmul
//! ([`crate::v5::circuit::matmul_a100_stark`]) and the rho/breakpoint policy gate
//! ([`crate::v5::circuit::policy_stark`]) — together with the four committed LUTs their
//! auxiliary columns are served by (`FP16DECODE`, `RANGE16`, `FP16POW2`, `WIDTH32`), under one
//! [`starky::batch_prover::batch_prove`] / [`starky::batch_verifier::batch_verify`] argument with
//! the full [`CrossTableLookup`] set of [`super::ctl`]:
//!
//! * the four LUT channels, so the matmul's decode/shift/width columns and the policy's
//!   gate-slack / per-step-count range checks are enforced by the batch's own verification
//!   (the CTL multiset balance), not by blind trust;
//! * the `matmul -> policy` census-import channel, so the policy scores the matmul's
//!   tightly-pinned per-step census rather than regenerating it;
//! * the two internal **quant-chain** channels (increment 6b): `row-scale -> noisy-quant` (the derived
//!   per-row `(alpha, beta)` the noisy-quant consumes) and the G1/G3 `noisy-quant <-> G2 FMA` pairing
//!   (binding G1's proven `t` and G3's cast input `noised` to the single-rounding FMA). Together with
//!   the three quant AIRs and the shared LUTs, these prove the noised operand
//!   `noised = Q(alpha*raw + beta*N)` end-to-end inside the batch. `raw` is bound to the commitment
//!   (6c) and the noise matrix `N = E@F^T` to the noise matmul (6e-1, below); `E`/`F` stay free
//!   witness (seed-binding is 6e-2/6e-3).
//!
//! The batch order is canonical: matmul (0), policy (1), xor_fold (2), blake3 (3), the three
//! quantization-chain AIRs — row-scale (4), noisy-quant G1/G3 (5), G2 FMA (6) — the noise matmul (7),
//! then the five LUTs (8..12) in descending committed height. The main tables' heights come from the AIR geometry `(h, w, k)` — the
//! verifier never takes the batch layout from the proof. The LUTs' precommitted columns are
//! committed once at setup ([`Fp16System::preprocessed_data`]); proofs copy the setup
//! commitment and FRI opens it alongside the trace oracles.
//!
//! **Scope.** This is the native batched prove/verify path: an honest tile's proof generates and
//! verifies, and tampering (a trace cell, a census value, a decode column, a gate slack) is
//! rejected. The recursive FRI wrapper to a constant-size proof (the FP8 `wrapper.rs` analogue)
//! is the documented follow-on.
//!
//! **Operand provenance (increments 6c + 6d — DONE).** The matmul provably multiplies the NOISED
//! quantization of the COMMITTED operands, closing audit Finding 3. Two keyed CTLs bind one operand
//! set through Blake3 + quant + matmul:
//!
//! * **Operand-bytes (6c).** The Blake3 operand trees' committed byte pairs (-> `HASH_A`/`HASH_B`)
//!   equal the quant chain's `raw` inputs, keyed by byte offset `2*e` (A elements `0..h*k`, B
//!   elements past them, one shared disjoint key space). So `raw` is no longer free witness: it is
//!   the committed operand (`HASH_A`/`HASH_B` bit-exact with `commit_operand(..).root()`).
//! * **Operand-codes (6d).** The matmul's `operand_codes_a/b` equal the noisy-quant `out` column,
//!   keyed by the shared element index `operand_index_base_{a,b} + lane`, with the `w`/`h` reuse
//!   multiplicity on the looked side. This single channel binds BOTH provenance (matmul code ==
//!   noised `out`) AND cross-cell sharing (cells `(r, c1)`/`(r, c2)` reuse the same `A[r, :]` element
//!   because they look up the same key; a column reuses `B[:, c]`). `operand_index_base_a/b` are
//!   class (a) known columns, so the keys are fixed and only the codes are witness.
//!
//! So `generate_batch_traces` flows ONE operand set: committed bytes (Blake3) == `raw` (quant)
//! --`Q(.)`--> `out` (quant) == matmul operand codes. The three quant AIRs + the two internal
//! channels prove `noised = Q(alpha*raw + beta*N)` over EVERY operand row of BOTH operands.
//!
//! **Noise `N` provenance (increments 6e-1 … 6e-3d — DONE).** `N = E@F^T` is proven in-batch (two
//! [`MatmulStarkA100`] instances, tables 7/8, bound to the noise noisy-quant consumes by the
//! noise-word CTL); `E`/`F` are the proven normalized lines of NoiseStark (E/F-operand CTL); the raw
//! keyed-XOF bytes NoiseStark normalizes are recomputed in-circuit by the forked noise-BLAKE3 engine
//! (table 10) from the seeds and bound via the egress-pair CTL (6e-3c); and the SEEDS are derived from
//! the committed operand roots — `seedA`/`seedB = noise_seeds(keys, {HASH_A, HASH_B}, p)`
//! ([`fp16_root_derived_seeds`], exactly [`crate::v5::api::noise::noise_seeds`]) — fed as the
//! noise-BLAKE3 `KEY_A`/`KEY_B` public inputs, which [`Fp16System::verify`] pins to the seeds
//! recomputed from the pinned `HASH_A`/`HASH_B` + keys + `p` (6e-3d). So the whole noise chain
//! (`E`/`F` -> `N` -> `beta*N` -> `noised` -> tile -> jackpot) is a pure function of the COMMITTED
//! operands, bit-exact with the plaintext's root-derived noise, and fully anti-grind.
//!
//! **Header-pinned verify gateway ([`Fp16System::verify_with_headers`] — DONE).** The opening keys
//! (`keyA`/`keyB`) and the `p_a`/`p_b` encodings that feed the noise seeds + the operand-tree
//! commitment keys are no longer taken from the circuit boundary in the sound entry: the gateway
//! derives `keyA = key_a(proposed)`, `keyB = key_b(ancestor)` ([`commitment_keys`]) and
//! `p_a`/`p_b` ([`Fp16JobParams::encode_p_a`]/`encode_p_b`) from the block headers + public job
//! params, bit-exact with the plaintext certificate, then pins the proof's operand-tree
//! `KEY_A`/`KEY_B`, the lottery-compression `JACKPOT_KEY` (= `jackpot_key(seed_a)` over the
//! header/root-derived `seed_a`, so the jackpot key cannot be ground), and the noise-BLAKE3
//! `KEY_A`/`KEY_B` (= `noise_seeds` of the header-derived keys/`p` over the proof's `HASH_A`/
//! `HASH_B`) to those header-derived values, DERIVES `statement_digest` from the proven
//! `HASH_JACKPOT` (not caller-supplied), and applies the native difficulty check
//! ([`check_jackpot_difficulty`]). So the ZK verify is pinned to the actual header exactly like the
//! plaintext cert path. The committed roots `HASH_A`/`HASH_B` still come from the proof (as they do
//! in the plaintext cert, whose root is proof-carried); the ONLY remaining boundary is inherent — the
//! consensus layer supplies the header + `nbits`, and must window-authenticate `job.ancestor_header`
//! (the plaintext FFI's SHA256d hash-walk) before calling the gateway, which assumes that
//! authentication exactly as the plaintext `verify_fp16_plain_proof` assumes its caller's. The
//! caller-supplied-expectation [`Fp16System::verify`] is retained for the non-consensus test paths.
//!
//! **Header binding CLOSED (the ZK proof is now a header-bound consensus certificate).** The output
//! tile is bound end-to-end: the matmul -> XorFold cell-results CTL feeds each proven
//! `cell_result_f32_*` into the lottery mixer, the XorFold -> Blake3 lottery-words CTL carries the 16
//! lane outputs into the jackpot compression, the Blake3 AIR binds that compression to the
//! `HASH_JACKPOT` public input, and the header gateways
//! ([`Fp16System::verify_with_headers`] for the batch proof and
//! [`crate::v5::circuit::wrapper::verify_wrapped_proof_with_headers`] for the wrapped/published
//! proof) pin `HASH_JACKPOT` to the consensus header by pinning the jackpot key to
//! `jackpot_key(seed_a)` and deriving `statement_digest` from the proven `HASH_JACKPOT`. A verified
//! proof therefore attests "this policy-passing tile folds, under the header-derived jackpot key for
//! this block, to a `HASH_JACKPOT` meeting difficulty." (The census totals remain non-public, which
//! is sound: the policy AIR gates them in-circuit.)
//!
//! The ZK proof is thus a sound stand-alone consensus certificate; the only REMAINING work to make
//! it the wired consensus path is deployment integration (the FFI entry, the Go/wire routing, and
//! the miner switching from plaintext-cert assembly to proof generation), tracked as follow-on
//! steps. Until that lands the wired consensus path remains the plaintext certificate
//! ([`crate::v5::api::verify`]). The full binding design (the Blake3 commitment, the seed-derived
//! noise + noisy-quant stages, the XorFold ticket, and the header-bound public inputs) is in
//! `docs/fp16_scheme/zk_binding_design.md`.

use anyhow::{Result, anyhow, ensure};
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::fri::FriConfig;
use plonky2::fri::reduction_strategies::FriReductionStrategy;
use plonky2::hash::hash_types::{HashOut, RichField};
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::plonk::config::GenericConfig;
use plonky2::util::log2_strict;
use plonky2::util::timing::TimingTree;
use primitive_types::U256;
use starky::batch_proof::BatchStarkProofWithPublicInputs;
use starky::batch_prover::{BatchStarkPreprocessedData, BatchStarkPreprocessedVerifierData, batch_prove};
use starky::batch_stark::BatchStark;
use starky::batch_universal::UniversalVerifierEnvelope;
use starky::batch_verifier::{BatchKnownColumns, batch_verify};
use starky::config::StarkConfig;
use starky::cross_table_lookup::CrossTableLookup;

use crate::v4::api::layout::{AxisPattern, DimType, lane_assignment};
use crate::v4::api::primitives::Hash256;
use crate::v4::api::proof_utils::check_jackpot_difficulty;
use crate::v4::api::transcript::{hash_labelled, jackpot_key};
use crate::v2::api::proof_utils::u32_field_array_to_hash;

use core::borrow::Borrow;

use super::blake3_commit::{bytes32_to_words, fp16_blake3_program};
use crate::v5::api::commitment::{commit_operand, rows_to_bytes};
use super::ctl::{
    BLAKE3_A100_TABLE, FP16_LUT_TABLES, MATMUL_A100_TABLE, NOISE_BLAKE3_A100_TABLE,
    NOISE_MATMUL_A_A100_TABLE, NOISE_MATMUL_B_A100_TABLE, NOISE_STARK_A100_TABLE,
    NOISY_QUANT_A100_TABLE, NOISY_QUANT_FMA_A100_TABLE, NUM_FP16_LUT_TABLES, NUM_FP16_MAIN_TABLES,
    NUM_FP16_TABLES, ROW_SCALE_A100_TABLE, XOR_FOLD_A100_TABLE, all_cross_table_lookups,
    fp16_lut_inventories, fp16_lut_table_idx,
};
use super::blake3_fp16_stark::columns::NUM_FP16_RAW_BLAKE3_PUBLIC_INPUTS;
use super::blake3_fp16_stark::stark::{Fp16RawBlake3KnownInputs, Fp16RawBlake3Stark, Fp16RawBlake3TraceInputs};
use super::noise_blake3::{
    noise_blake3_aux_msgs, noise_blake3_height_bits, noise_blake3_program, noise_blake3_public_inputs,
    noise_seed_key_words, NOISE_BLAKE3_JACKPOT_KEY, NOISE_BLAKE3_LOTTERY_WORDS,
};
use super::matmul_a100_stark::columns::{MatmulA100ColumnsView, NUM_MATMUL_A100_PUBLIC_INPUTS};
use super::matmul_a100_stark::stark::MatmulStarkA100;
use super::noise_stark::stark::{NoiseProgram, NoiseStark};
use super::noisy_quant_fma_stark::columns::FMA_COL_MAP;
use super::noisy_quant_fma_stark::stark::{FmaProgram, NoisyQuantFmaStark};
use super::noisy_quant_stark::columns::NOISY_QUANT_COL_MAP;
use super::noisy_quant_stark::stark::{NoisyQuantProgram, NoisyQuantStark};
use super::policy_stark::columns::NUM_POLICY_A100_PUBLIC_INPUTS;
use super::policy_stark::stark::PolicyStarkA100;
use super::row_scale_stark::stark::{RowScaleProgram, RowScaleStark};
use crate::v5::api::noise::{NoiseFactor, Side, commitment_keys, noise_seeds, sample_line_xof_bytes};
use crate::v5::api::plain_proof::Fp16JobParams;
use crate::v4::api::primitives::{IncompleteBlockHeader, Sides};
use super::xor_fold_stark::columns::XorFoldColumnsView;
use super::xor_fold_stark::stark::{XorFoldProgram, XorFoldStark};
use crate::v5::api::accumulate::a100_matmul;
use crate::v5::api::quantization::{noisy_quantize, row_norms};
use crate::v4::api::dtype::bf16_to_f32;
use crate::v4::circuit::blake3_stark::columns::{
    NUM_BLAKE3_PUBLIC_INPUTS, PI_HASH_A, PI_HASH_B, PI_HASH_JACKPOT, PI_JACKPOT_KEY, PI_KEY_A, PI_KEY_B,
};
use crate::v4::circuit::blake3_stark::stark::{Blake3KnownInputs, Blake3Stark, Blake3TraceInputs};
use crate::v4::api::public_params::HashId;
use crate::v4::circuit::luts::stark::boxed_lut_stark;
use crate::v4::circuit::luts::{LutChecker, lut_precommitted_values, lut_trace, num_precommitted_columns};

// ==================================================================================================
// Consensus proof shape (mirrors the FP8 consensus parameters)
// ==================================================================================================

/// Targeted (conjectured) security level in bits.
pub const STARK_SECURITY_BITS: usize = 120;
/// Logup/CTL challenge repetitions.
pub const STARK_NUM_CHALLENGES: usize = 3;
/// FRI rate `2^-1`.
pub const STARK_RATE_BITS: usize = 1;
/// Merkle cap height of every oracle and FRI commit layer.
pub const STARK_CAP_HEIGHT: usize = 4;
/// FRI grinding bits.
pub const STARK_POW_BITS: u32 = 18;
/// FRI query rounds meeting the security target at this rate.
pub const STARK_QUERY_ROUNDS: usize = (STARK_SECURITY_BITS - STARK_POW_BITS as usize).div_ceil(STARK_RATE_BITS);

/// The consensus FRI fold ladder (degree bits, strictly descending) for the FP16 batch. Every
/// reachable table height is a member and every gap is at most 3 bits, so the ladder doubles as
/// the universal-verifier fold schedule: it includes the four committed-LUT heights (`FP16DECODE`
/// / `RANGE16` at `2^16`, `FP16POW2` at `2^8`, `WIDTH32` at `2^5`) and the main-table range
/// [`FP16_MAIN_TABLE_DEGREE_RANGE`]. Baking it (rather than the job's distinct heights) into
/// [`fp16_stark_config`] makes the config a consensus constant for every on-ladder job, so the
/// Fiat-Shamir transcript the recursive wrapper replays in-circuit is identical across tile
/// sizes — the prerequisite for one compiled universal wrapper to verify every envelope-legal
/// job (the FP8 `wrapper.rs` design).
pub const FP16_REACHABLE_DEGREE_BITS: [usize; 7] = [16, 13, 10, 8, 6, 5, 4];

/// Batch indices of the grouped tables — the four committed LUTs (`NUM_FP16_MAIN_TABLES..`). Their
/// heights are consensus constants, so each role (trace / auxiliary / quotient) commits all of them
/// in one shared multi-height Merkle tree. Grouping is required by the universal recursive verifier
/// (its shared-tree Merkle walks are profile-independent only for grouped fixed-height tables), and
/// it also shrinks the native proof's cap count.
pub const FP16_GROUPED_TABLES: [usize; NUM_FP16_LUT_TABLES] = {
    let mut tables = [0; NUM_FP16_LUT_TABLES];
    let mut i = 0;
    while i < NUM_FP16_LUT_TABLES {
        tables[i] = NUM_FP16_MAIN_TABLES + i;
        i += 1;
    }
    tables
};

/// Inclusive degree-bits range `[lo, hi]` of the two main tables (matmul, policy — they share the
/// row grid) over the wrapper's consensus envelope. Both bounds lie on
/// [`FP16_REACHABLE_DEGREE_BITS`]. The floor is `4` so every table's LDE (`height + RATE`) strictly
/// exceeds the Merkle `cap_height` (4): a table whose LDE equals the cap has a degenerate (cap ==
/// leaves, zero-sibling) Merkle tree that the universal verifier's envelope-max padding does not
/// model. Jobs outside this range still prove natively (the config adds their heights as extra
/// boundaries); only the recursive wrapper is gated to the envelope.
pub const FP16_MAIN_TABLE_DEGREE_RANGE: (usize, usize) = (4, 6);

/// Inclusive degree-bits range of each per-side noise matmul table (`N_A`/`N_B = E@F^T`, increment
/// 6e-2). A side's `m x k` output (inner dimension [`FP16_QUANT_R`]) has live height
/// `m*k*(FP16_QUANT_R/GROUP)`, snapped up to a [`FP16_REACHABLE_DEGREE_BITS`] member; over the
/// wrapper's legal tiles this lands on `2^8..2^10`. Both bounds lie on the ladder.
pub const FP16_NOISE_MATMUL_DEGREE_RANGE: (usize, usize) = (8, 10);

/// Inclusive degree-bits range of the NoiseStark table (increment 6e-2). It proves every noise line
/// (`E_A`,`F_A`,`E_B`,`F_B` = `h+w+2k` lines) at rank [`FP16_QUANT_R`], live height
/// `(h+w+2k)*FP16_QUANT_R` snapped to the ladder; over the wrapper's legal tiles (`k >= 16`) this
/// lands on `2^13`.
pub const FP16_NOISE_STARK_DEGREE_RANGE: (usize, usize) = (13, 13);

/// Inclusive degree-bits range of the noise-BLAKE3 derivation table (increment 6e-3c). It runs two
/// subkey compressions, `h + w + 2k` per-line compressions and the mandatory lottery (`8` rows each),
/// its live height `8*(h+w+2k+3)` snapped up to a [`FP16_REACHABLE_DEGREE_BITS`] member via
/// [`crate::v5::circuit::noise_blake3::noise_blake3_height_bits`]. Over the wrapper's legal tiles
/// (`k >= 16`) this lands on `2^10..2^13`; both bounds lie on the ladder.
pub const FP16_NOISE_BLAKE3_DEGREE_RANGE: (usize, usize) = (10, 13);

/// The consensus universal-verifier envelope: the seven tile/quant main tables each range over
/// [`FP16_MAIN_TABLE_DEGREE_RANGE`], the two per-side noise matmuls over
/// [`FP16_NOISE_MATMUL_DEGREE_RANGE`], and NoiseStark over [`FP16_NOISE_STARK_DEGREE_RANGE`]; the
/// five LUT tables are fixed at their consensus heights; the fold ladder is
/// [`FP16_REACHABLE_DEGREE_BITS`]. The FP16 LUTs are committed per table (not grouped), so
/// `grouped_tables` is empty. One stage-1 wrapper circuit built on this envelope verifies every
/// envelope-legal job (the job's degree profile becomes a public input).
pub fn fp16_universal_envelope() -> UniversalVerifierEnvelope {
    let (lo, hi) = FP16_MAIN_TABLE_DEGREE_RANGE;
    // Main tables 0..=6 (tile + quant chain) share the tile row grid; tables 7/8 are the per-side
    // noise matmuls, table 9 is NoiseStark, table 10 is the noise-BLAKE3 derivation.
    let mut degree_ranges: Vec<(usize, usize)> = vec![(lo, hi); NUM_FP16_MAIN_TABLES - 4];
    degree_ranges.push(FP16_NOISE_MATMUL_DEGREE_RANGE);
    degree_ranges.push(FP16_NOISE_MATMUL_DEGREE_RANGE);
    degree_ranges.push(FP16_NOISE_STARK_DEGREE_RANGE);
    degree_ranges.push(FP16_NOISE_BLAKE3_DEGREE_RANGE);
    for &table in &FP16_LUT_TABLES {
        let bits = log2_strict(crate::v4::circuit::luts::lut_height(table));
        degree_ranges.push((bits, bits));
    }
    UniversalVerifierEnvelope {
        degree_ranges,
        ladder: FP16_REACHABLE_DEGREE_BITS.to_vec(),
        grouped_tables: FP16_GROUPED_TABLES.to_vec(),
    }
}

/// Domain-separation label for the FP16 statement digest. The FP16 scheme reuses the FP8 transcript
/// domain (its jackpot/key subkeys are already the `pearl/v4/FP8/` labels), so this stays under that
/// prefix — distinct from the FP8 block digest's own `pearl/v4/FP8/zk-public`.
/// [`derive_statement_digest`] hashes the proven lottery digest under it.
const LABEL_FP16_ZK_PUBLIC: &[u8] = b"pearl/v4/FP8/fp16-zk-public";

/// The proven lottery digest `J` reassembled from the Blake3 table's `HASH_JACKPOT` public-input
/// limbs (8 little-endian `u32` words, the STARK witness key encoding). The FP16 analogue of the
/// FP8 driver's `hash_jackpot`.
pub(crate) fn hash_jackpot<F: RichField>(blake3_pis: &[F]) -> Hash256 {
    let limbs: &[F; 8] = blake3_pis[PI_HASH_JACKPOT..PI_HASH_JACKPOT + 8]
        .try_into()
        .expect("HASH_JACKPOT occupies 8 Blake3 public-input limbs");
    u32_field_array_to_hash(limbs)
}

/// Derives the Fiat-Shamir statement digest from the PROVEN lottery digest `J` — the FP16 analogue
/// of the FP8 driver's `derive_statement_digest` callback (`H_"zk-public"(…)`). This ties the salt
/// the known-column binding absorbs to the batch's own `HASH_JACKPOT`, so a verified proof attests
/// "this policy-passing tile folds to this jackpot under this derived statement," not an opaque
/// caller-chosen salt. Binding `J` to the consensus block header (the full
/// `H_"zk-public"(σ̂ || public_data)` FP8 does) remains the documented follow-on; this increment
/// binds the output chain `tile -> HASH_JACKPOT -> statement_digest`.
pub(crate) fn derive_statement_digest(hash_jackpot: Hash256) -> Hash256 {
    hash_labelled(&hash_jackpot, LABEL_FP16_ZK_PUBLIC, None)
}

/// Reduces a 32-byte statement digest into four Goldilocks elements (little-endian integer mod
/// `p^4`) — the Fiat-Shamir salt the known-column binding absorbs, and the wrapper's pinned
/// `known-column digest` public inputs.
pub(crate) fn statement_digest_to_hash_out<F: RichField>(digest: Hash256) -> HashOut<F> {
    let p = U256::from(F::ORDER);
    let mut value = U256::from_little_endian(&digest);
    let mut elements = [F::ZERO; 4];
    for element in &mut elements {
        *element = F::from_canonical_u64((value % p).as_u64());
        value /= p;
    }
    HashOut { elements }
}

/// The FP16 batch [`StarkConfig`] for a job whose tables have the given `degree_bits` (any order,
/// duplicates allowed). The FRI reduction schedule is a [`FriReductionStrategy::Ladder`] whose
/// boundaries are [`FP16_REACHABLE_DEGREE_BITS`] unioned with the job's own heights (descending,
/// deduped), so every table's height is a fold boundary (batch FRI injects each instance at its
/// LDE size) and gaps are auto-split into steps of at most 3 bits. For an on-ladder job the union
/// equals the consensus ladder, so the config — hence the absorbed Fiat-Shamir transcript — is a
/// consensus constant (the universal-wrapper prerequisite).
pub fn fp16_stark_config(degree_bits: &[usize]) -> StarkConfig {
    let min = degree_bits.iter().copied().min().expect("at least one table");
    assert!(
        min + STARK_RATE_BITS >= STARK_CAP_HEIGHT,
        "the smallest table's LDE must cover the Merkle cap"
    );
    let mut boundaries: Vec<usize> = FP16_REACHABLE_DEGREE_BITS.iter().chain(degree_bits).copied().collect();
    boundaries.sort_unstable_by_key(|&bits| core::cmp::Reverse(bits));
    boundaries.dedup();
    StarkConfig::new(
        STARK_SECURITY_BITS,
        STARK_NUM_CHALLENGES,
        FriConfig {
            rate_bits: STARK_RATE_BITS,
            cap_height: STARK_CAP_HEIGHT,
            proof_of_work_bits: STARK_POW_BITS,
            reduction_strategy: FriReductionStrategy::Ladder(boundaries),
            num_query_rounds: STARK_QUERY_ROUNDS,
        },
    )
}

// ==================================================================================================
// Quantization chain geometry (increment 6b)
// ==================================================================================================

/// The fixed noise rank the quant chain uses (`dr`/`dos` are then compile-time bf16 constants).
pub const FP16_QUANT_R: usize = 32;

/// The smallest on-ladder ([`FP16_REACHABLE_DEGREE_BITS`]) trace height `>= live`. The quant chain's
/// three tables are padded to this common height so every legal tile keeps them on the consensus
/// FRI fold ladder (the universal-wrapper prerequisite), covering the real `(h + w) * k` operand
/// elements with trailing padding.
pub fn fp16_quant_num_rows(live: usize) -> usize {
    FP16_REACHABLE_DEGREE_BITS
        .iter()
        .map(|&b| 1usize << b)
        .filter(|&h| h >= live)
        .min()
        .expect("the live element count exceeds the ladder's top height")
}

/// The per-side noise-seed public-parameter encodings (`p_A` / `p_B`) for an `h x w` tile with inner
/// `k`, built bit-exact with [`crate::v5::api::plain_proof::Fp16JobParams::encode_p_a`] /
/// `encode_p_b` so the driver derives the noise seeds through the exact same `api::fp16::noise`
/// recipe as the plaintext certificate.
///
/// The driver's simplified job model commits the whole operand as the tile (so `num_rows = h` / `w`
/// and the pattern is a single dense tile dim) under a fixed placeholder ancestor header. In
/// production the consensus layer supplies the real header-derived params (ancestor, pattern, …) —
/// the same test-vs-header boundary the opening keys ([`Fp16System::prove`]'s `key_a`/`key_b`) sit
/// on. The content only matters in that both the circuit and the plaintext comparison fold the same
/// `p` into [`noise_seeds`].
pub(crate) fn fp16_noise_seed_params(
    h: usize,
    w: usize,
    k: usize,
    a_hash_id: HashId,
    b_hash_id: HashId,
) -> Sides<Vec<u8>> {
    use crate::v5::api::params::Fp16Device;
    use crate::v5::api::plain_proof::{Fp16JobParams, Fp16OperandParams};
    use crate::v4::api::layout::{AxisPattern, DimType};
    let pattern = |n: usize| AxisPattern::new(&[(n as u32, DimType::Blake)]).expect("dense tile pattern");
    let job = Fp16JobParams {
        ancestor_header: IncompleteBlockHeader::zero(),
        device: Fp16Device::A100,
        k: k as u32,
        r: FP16_QUANT_R as u32,
        operands: Sides {
            a: Fp16OperandParams { num_rows: h as u32, hash_id: a_hash_id, pattern: pattern(h) },
            b: Fp16OperandParams { num_rows: w as u32, hash_id: b_hash_id, pattern: pattern(w) },
        },
    };
    Sides { a: job.encode_p_a(), b: job.encode_p_b() }
}

/// The root-derived per-side noise seeds `noise_seeds({key_a, key_b}, {root_a, root_b}, {p_a, p_b})`
/// for an `h x w` tile over the committed operands `a_codes`/`b_codes` — the exact
/// [`crate::v5::api::noise::noise_seeds`] the plaintext certificate derives. `root_X =
/// commit_operand(..).root()` (bit-exact with the batch's `HASH_A`/`HASH_B`), so the seeds — and
/// hence every derived XOF byte, every noise line, `N`, the noised operands, the tile and the
/// jackpot — are a pure function of the COMMITTED operands: the noise is operand-dependent and fully
/// anti-grind. Pipeline order: operands -> commit -> roots -> `noise_seeds` -> noise.
pub(crate) fn fp16_root_derived_seeds(
    h: usize,
    w: usize,
    k: usize,
    a_hash_id: HashId,
    b_hash_id: HashId,
    key_a: Hash256,
    key_b: Hash256,
    a_codes: &[u16],
    b_codes: &[u16],
) -> anyhow::Result<Sides<Hash256>> {
    let root_a = commit_operand(a_codes, h, k, a_hash_id, key_a)?.root();
    let root_b = commit_operand(b_codes, w, k, b_hash_id, key_b)?.root();
    let p = fp16_noise_seed_params(h, w, k, a_hash_id, b_hash_id);
    Ok(noise_seeds(&Sides { a: key_a, b: key_b }, &Sides { a: root_a, b: root_b }, &p))
}

/// The real, seed-derived FP16 noise factors for an `h x w` tile under the ROOT-DERIVED `seeds`
/// ([`fp16_root_derived_seeds`]) — DISTINCT `F_A`/`F_B` (the plaintext's distinct `F`). Returns
/// `(E_A, F_A, E_B, F_B)`, each a row-major stack of normalized FP16 lines, bit-exact with
/// [`crate::v5::api::noise::sample_noise`] over `seeds` and global row/col indices `0..h` / `0..w`:
/// `E_A` is `h` lines, `F_A`/`F_B` are `k` lines, `E_B` is `w` lines; each line `r = FP16_QUANT_R`
/// entries. `N_A = E_A @ F_A^T`, `N_B = E_B @ F_B^T`.
pub(crate) fn fp16_tile_noise(seeds: Sides<Hash256>, h: usize, w: usize, k: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    use crate::v5::api::noise::sample_noise;
    let a_rows: Vec<u32> = (0..h as u32).collect();
    let b_cols: Vec<u32> = (0..w as u32).collect();
    let noise = sample_noise(k, FP16_QUANT_R as u16, seeds, &a_rows, &b_cols);
    (noise.a.e, noise.a.f, noise.b.e, noise.b.f)
}

/// The per-line noise-matmul reuse multiplicities (the `OPERAND_MULT` known column) in NoiseStark line
/// order `[E_A (h), F_A (k), E_B (w), F_B (k)]`: `k` for an `E` line, `h`/`w` for an `F` line. These are
/// seed-independent (pure geometry), so [`Fp16System::new`] derives the NoiseStark program from them
/// without any operand/seed.
pub(crate) fn fp16_noise_mults(h: usize, w: usize, k: usize) -> Vec<u64> {
    let mut mults: Vec<u64> = Vec::with_capacity(h + w + 2 * k);
    mults.extend(std::iter::repeat_n(k as u64, h)); // E_A lines
    mults.extend(std::iter::repeat_n(h as u64, k)); // F_A lines
    mults.extend(std::iter::repeat_n(k as u64, w)); // E_B lines
    mults.extend(std::iter::repeat_n(w as u64, k)); // F_B lines
    mults
}

/// The raw keyed-BLAKE3-XOF bytes of every noise line for an `h x w` tile under the ROOT-DERIVED
/// `seeds`, concatenated in NoiseStark line order `[E_A (h), F_A (k), E_B (w), F_B (k)]`. Feeding them
/// to NoiseStark yields output bit-exact with the normalized lines [`fp16_tile_noise`] returns. (The
/// companion multiplicities are [`fp16_noise_mults`], which is seed-independent.)
pub(crate) fn fp16_tile_noise_xof(seeds: Sides<Hash256>, h: usize, w: usize, k: usize) -> Vec<u8> {
    let r = FP16_QUANT_R as u16;
    let (sa, sb) = (seeds.a, seeds.b);
    let mut bytes: Vec<u8> = Vec::new();
    for row in 0..h as u32 {
        bytes.extend_from_slice(&sample_line_xof_bytes(&sa, Side::A, NoiseFactor::E, row, r));
    }
    for i in 0..k as u32 {
        bytes.extend_from_slice(&sample_line_xof_bytes(&sb, Side::A, NoiseFactor::F, i, r));
    }
    for col in 0..w as u32 {
        bytes.extend_from_slice(&sample_line_xof_bytes(&sb, Side::B, NoiseFactor::E, col, r));
    }
    for i in 0..k as u32 {
        bytes.extend_from_slice(&sample_line_xof_bytes(&sb, Side::B, NoiseFactor::F, i, r));
    }
    bytes
}

/// One side's noise matmul geometry (increment 6e-2): a
/// [`crate::v5::circuit::matmul_a100_stark::MatmulStarkA100`] of output `m x k` (A: `m = h`; B:
/// `m = w`) with inner dimension [`FP16_QUANT_R`], computing `N = E @ F^T`. Returns
/// `(m, k_out, k_inner, height)`, the height being the live row count `m*k*(FP16_QUANT_R/GROUP)`
/// snapped up to a [`FP16_REACHABLE_DEGREE_BITS`] member (forced via [`MatmulStarkA100::new_with_height`]).
pub(crate) fn fp16_noise_matmul_geometry(m: usize, k: usize) -> (usize, usize, usize, usize) {
    use super::matmul_a100_stark::columns::GROUP;
    let k_inner = FP16_QUANT_R;
    let live = m * k * (k_inner / GROUP);
    (m, k, k_inner, fp16_quant_num_rows(live))
}

/// The proven noised FP16 operands for an `h x w` tile over the committed `a_codes`/`b_codes`:
/// `noisy_quantize` over each operand row with the real ROOT-DERIVED noise (distinct `F_A`/`F_B`,
/// seeds = [`fp16_root_derived_seeds`] over the committed roots/keys/params), bit-exact with the
/// batch's quant-chain `out` column (hence with what the matmul multiplies — see
/// [`Fp16System::noised_operands`]). A free helper so test fixtures can pick an operand whose NOISED
/// tile clears the policy gate without constructing a full system; the keys/hash ids must match the
/// [`Fp16System`] the operand will be proven under so the derived noise is the same.
/// Seed-exact noised operands for a tile, through the full real pipeline
/// (root-derived seeds -> noise lines -> `noisy_quantize`). Public so offline
/// validation tooling (`docs/fp16_scheme/validation`) can reproduce consensus
/// noise bit-for-bit rather than approximating it.
pub fn fp16_noised_operands(
    h: usize,
    w: usize,
    k: usize,
    a_codes: &[u16],
    b_codes: &[u16],
    key_a: Hash256,
    key_b: Hash256,
    a_hash_id: HashId,
    b_hash_id: HashId,
) -> anyhow::Result<(Vec<u16>, Vec<u16>)> {
    use crate::v5::api::quantization::{noisy_quantize, row_norms};
    let r = FP16_QUANT_R;
    let seeds = fp16_root_derived_seeds(h, w, k, a_hash_id, b_hash_id, key_a, key_b, a_codes, b_codes)?;
    let (e_a, f_a, e_b, f_b) = fp16_tile_noise(seeds, h, w, k);
    let mut sides: Vec<Vec<u16>> = Vec::with_capacity(2);
    for (codes, num_rows, e, f) in [(a_codes, h, &e_a, &f_a), (b_codes, w, &e_b, &f_b)] {
        let norms: Vec<(u16, u16)> = (0..num_rows)
            .map(|i| row_norms(&codes[i * k..(i + 1) * k]))
            .collect::<anyhow::Result<_>>()?;
        sides.push(noisy_quantize(codes, e, f, &norms, r)?.noised_part);
    }
    Ok((sides.remove(0), sides.remove(0)))
}

// ==================================================================================================
// The system
// ==================================================================================================

/// One FP16 tile's fully derived proving/verifying context. Built from the public geometry
/// `(h, w, k)` alone; the prover supplies the private operand codes to [`Self::prove`].
pub struct Fp16System<F: RichField + Extendable<D>, const D: usize> {
    matmul: MatmulStarkA100<F, D>,
    policy: PolicyStarkA100<F, D>,
    /// The XorFold lottery mixer: folds the matmul output tile into the 16 lottery lanes.
    xor_fold: XorFoldStark<F, D>,
    /// The full FP16 Blake3 commitment: operand-A tree (-> `HASH_A`, keyed `KEY_A`), operand-B tree
    /// (-> `HASH_B`, keyed `KEY_B`), and the keyed compression of the 16 lottery words to
    /// `HASH_JACKPOT`.
    blake3: Blake3Stark<F, D>,
    /// The per-operand Merkle leaf sizes ([`HashId::chunk_len`]) the Blake3 operand trees commit
    /// under. Geometry of the Blake3 program, so they are fixed at construction.
    a_hash_id: HashId,
    b_hash_id: HashId,
    /// The quantization chain (increment 6b): the per-row scale derivation, the fused noisy-quant
    /// (G1 pre-FMA multiply + G3 FP16 cast), and the single-rounding G2 FMA. They prove the noised
    /// operand `noised = Q(alpha*raw + beta*N)` over a fixed in-envelope witness operand row; `raw`
    /// and `N` are free witness (binding them is increments 6c/6e).
    row_scale: RowScaleStark<F, D>,
    noisy_quant: NoisyQuantStark<F, D>,
    noisy_quant_fma: NoisyQuantFmaStark<F, D>,
    /// The A-side noise matmul (increment 6e-2): `N_A = E_A @ F_A^T` (`h x k`, inner [`FP16_QUANT_R`]).
    /// Its cell results bind to noisy-quant's A-side `(NOISE_LO, NOISE_HI)` (noise-word CTL); its
    /// `E_A`/`F_A` operands bind to NoiseStark's proven normalized lines (E/F-operand CTL). Padded to
    /// an on-ladder height.
    noise_matmul_a: MatmulStarkA100<F, D>,
    /// The B-side noise matmul (increment 6e-2): `N_B = E_B @ F_B^T` (`w x k`, inner [`FP16_QUANT_R`]),
    /// with DISTINCT `F_B` (restores the plaintext's distinct `F`). Cell results bind to noisy-quant's
    /// B-side noise words (offset `h*k`); `E_B`/`F_B` bind to NoiseStark (offset `(h+k)*r`).
    noise_matmul_b: MatmulStarkA100<F, D>,
    /// The NoiseStark line AIR (increment 6e-2): proves `normalize_line` for every noise line of the
    /// tile (`E_A`, `F_A`, `E_B`, `F_B`), its normalized FP16 output bound to the two noise matmuls'
    /// `E`/`F` operands. The raw XOF bytes stay free witness (seed-keyed-XOF binding is 6e-3).
    noise_stark: NoiseStark<F, D>,
    /// The noise-BLAKE3 derivation AIR (increment 6e-3d): recomputes every noise line's
    /// keyed-BLAKE3-XOF bytes in-circuit from the seeds (its `KEY_A`/`KEY_B` public inputs, which are
    /// the ROOT-DERIVED noise seeds), egressing each line's `cv_out`. The egress-pair CTL binds those
    /// limbs to NoiseStark's XOF-byte inputs, so the noise is bound to the seeds, and the seeds are
    /// bound to the committed operand roots. Padded to an on-ladder height.
    noise_blake3: Fp16RawBlake3Stark<F, D>,
    /// The per-side noise-seed public-parameter encodings (`p_A`/`p_B`) folded into the seed chain
    /// ([`fp16_noise_seed_params`]). Together with the committed roots (`HASH_A`/`HASH_B`) and the
    /// opening keys — both carried in the Blake3 public inputs — they determine the root-derived noise
    /// seeds `verify` pins the noise-BLAKE3 `KEY_A`/`KEY_B` to.
    noise_seed_params: Sides<Vec<u8>>,
    /// The five committed-LUT AIRs at their exact widths, in committed order.
    luts: [Box<dyn BatchStark<F, D>>; NUM_FP16_LUT_TABLES],
    h: usize,
    w: usize,
    k: usize,
    /// Per-table trace degree bits in canonical batch order (matmul, policy, then the LUTs).
    degree_bits: [usize; NUM_FP16_TABLES],
    /// The config for this job's degree profile.
    config: StarkConfig,
    /// Class (a) known columns of the two main tables (the LUT positions are empty).
    known: BatchKnownColumns<F>,
    /// All CTL channels (four LUT channels + census import), table indices in canonical order.
    ctls: Vec<CrossTableLookup<F>>,
}

impl<F: RichField + Extendable<D>, const D: usize> Fp16System<F, D> {
    /// Derives the full batching context for an `h x w` tile with inner dimension `k`.
    ///
    /// The lottery layout (which of the `h*w` output cells each of the 16 lanes folds) is derived
    /// canonically from `(h, w)` via [`default_lane_layout`] — the committed extractor layout of
    /// [`crate::v4::api::layout`]. `h*w` must admit a 16-lane Blake split (see [`default_lane_layout`]).
    pub fn new(h: usize, w: usize, k: usize, a_hash_id: HashId, b_hash_id: HashId) -> Self {
        let matmul = MatmulStarkA100::<F, D>::new(h, w, k);
        let policy = PolicyStarkA100::<F, D>::new(h, w, k);
        let (row_axis, col_axis) = default_lane_layout(h, w);
        let xor_fold = XorFoldStark::<F, D>::new(XorFoldProgram {
            lanes: lane_assignment(&row_axis, &col_axis),
        });
        // The committed operand byte lengths: A is `h x k` FP16 (`h*k*2` LE bytes), B is `w x k`
        // over the transposed operand (`w*k*2` LE bytes). These are the `commit_operand` row-byte
        // lengths the two keyed Merkle trees are built over.
        let a_len = h * k * 2;
        let b_len = w * k * 2;
        let blake3 = Blake3Stark::<F, D>::new(fp16_blake3_program(
            a_len,
            a_hash_id.chunk_len(),
            b_len,
            b_hash_id.chunk_len(),
        ));

        // The quantization chain (increment 6b-extended). It proves `noised = Q(alpha*raw + beta*N)`
        // over EVERY operand row of BOTH operands: `h` A rows then `w` B rows, each `k` elements
        // (rank `FP16_QUANT_R`). `raw` is the caller's real tile operands; `N` is still free witness
        // (binding `raw`/`N` to the operand commitment / `E@F` is increments 6c/6e). The three tables
        // are padded to a common on-ladder height (`fp16_quant_num_rows`) covering `(h + w) * k`.
        let quant_operand_rows = h + w;
        let quant_live = quant_operand_rows * k;
        let quant_height = fp16_quant_num_rows(quant_live);
        let row_scale =
            RowScaleStark::<F, D>::new(RowScaleProgram::with_rows_ab(h, w, k, FP16_QUANT_R, quant_height));
        // The operand-codes (6d) looked-side multiplicity: an A-side element (operand rows `0..h`) is
        // reused in `w` output cells, a B-side element (rows `h..h+w`) in `h` cells.
        let noisy_quant = NoisyQuantStark::<F, D>::new(NoisyQuantProgram::with_rows(quant_live, k, quant_height, h, w, h));
        let noisy_quant_fma = NoisyQuantFmaStark::<F, D>::new(FmaProgram::with_rows(quant_live, quant_height));
        // The two per-side noise matmuls `N_A = E_A @ F_A^T` (`h x k`) and `N_B = E_B @ F_B^T`
        // (`w x k`), inner `FP16_QUANT_R`, each padded to an on-ladder height (6e-2, distinct `F`).
        let (man, _, kin_a, noise_height_a) = fp16_noise_matmul_geometry(h, k);
        let noise_matmul_a = MatmulStarkA100::<F, D>::new_with_height(man, k, kin_a, noise_height_a);
        let (mbn, _, kin_b, noise_height_b) = fp16_noise_matmul_geometry(w, k);
        let noise_matmul_b = MatmulStarkA100::<F, D>::new_with_height(mbn, k, kin_b, noise_height_b);
        // NoiseStark: every noise line (`E_A`,`F_A`,`E_B`,`F_B`) at rank `FP16_QUANT_R`, with the
        // per-line reuse multiplicities, padded to an on-ladder height covering `(h+w+2k)*r` entries.
        let noise_mults = fp16_noise_mults(h, w, k);
        let noise_live = noise_mults.len() * FP16_QUANT_R;
        let noise_stark_height = fp16_quant_num_rows(noise_live);
        let noise_stark =
            NoiseStark::<F, D>::new(NoiseProgram::with_lines(FP16_QUANT_R, noise_mults, noise_stark_height));

        // The noise-BLAKE3 derivation table (6e-3d): recompute every line's keyed-XOF in-circuit from
        // the seeds (the engine's KEY_A/KEY_B public inputs), egressing each line's cv_out for the
        // egress-pair CTL. Padded to an on-ladder height. The program structure is seed-independent;
        // the seeds enter as the KEY_A/KEY_B public inputs, which are now the ROOT-DERIVED noise seeds
        // ([`fp16_root_derived_seeds`]) — operand-dependent, so they are computed per proof/verify
        // from the committed roots rather than stored here. `noise_seed_params` carries the `p`
        // half of the seed chain.
        let noise_blake3_bits = noise_blake3_height_bits(h, w, k);
        let noise_blake3 =
            Fp16RawBlake3Stark::<F, D>::new(noise_blake3_program(h, w, k, Some(noise_blake3_bits)));
        let noise_seed_params = fp16_noise_seed_params(h, w, k, a_hash_id, b_hash_id);

        // Class (a) known columns: each main table's leading geometry/layout columns, bit-exact with
        // its trace. The verifier recomputes these and binds the trace openings to them.
        let matmul_known = matmul.known_values();
        let policy_known = policy.known_values();
        let xor_fold_known = xor_fold.program.known_values::<F>();
        let blake3_known = blake3.program.known_values::<F>(&Blake3KnownInputs {
            a_values_len: a_len,
            a_scales_len: 0,
            b_values_len: b_len,
            b_scales_len: 0,
        });
        let row_scale_known = row_scale.program.known_values::<F>();
        let noisy_quant_known = noisy_quant.program.known_values::<F>();
        let fma_known = noisy_quant_fma.program.known_values::<F>();
        let noise_matmul_a_known = noise_matmul_a.known_values();
        let noise_matmul_b_known = noise_matmul_b.known_values();
        let noise_stark_known = noise_stark.program.known_values::<F>();
        let noise_blake3_known = noise_blake3.program.known_values::<F>(&Fp16RawBlake3KnownInputs {
            a_values_len: 0,
            a_scales_len: 0,
            b_values_len: 0,
            b_scales_len: 0,
        });
        let mut columns_per_table = vec![Vec::new(); NUM_FP16_TABLES];
        let mut values_per_table = vec![Vec::new(); NUM_FP16_TABLES];
        for (table, known) in [
            (MATMUL_A100_TABLE, matmul_known),
            (super::ctl::POLICY_A100_TABLE, policy_known),
            (XOR_FOLD_A100_TABLE, xor_fold_known),
            (BLAKE3_A100_TABLE, blake3_known),
            (ROW_SCALE_A100_TABLE, row_scale_known),
            (NOISY_QUANT_A100_TABLE, noisy_quant_known),
            (NOISY_QUANT_FMA_A100_TABLE, fma_known),
            (NOISE_MATMUL_A_A100_TABLE, noise_matmul_a_known),
            (NOISE_MATMUL_B_A100_TABLE, noise_matmul_b_known),
            (NOISE_STARK_A100_TABLE, noise_stark_known),
            (NOISE_BLAKE3_A100_TABLE, noise_blake3_known),
        ] {
            columns_per_table[table] = (0..known.len()).collect();
            values_per_table[table] = known;
        }
        let known = BatchKnownColumns {
            digest: None,
            columns_per_table,
            values_per_table,
        };

        // Per-table heights -> degree bits (canonical order). Each main table's height comes from
        // its own geometry; the LUTs' heights are consensus constants.
        assert_eq!(matmul.num_rows(), policy.num_rows(), "matmul and policy share the row grid");
        let main_heights = [
            matmul.num_rows(),
            policy.num_rows(),
            xor_fold.program.num_rows(),
            blake3.program.num_rows(),
            row_scale.program.num_rows(),
            noisy_quant.program.num_rows(),
            noisy_quant_fma.program.num_rows(),
            noise_matmul_a.num_rows(),
            noise_matmul_b.num_rows(),
            noise_stark.program.num_rows(),
            noise_blake3.program.num_rows(),
        ];
        let degree_bits: [usize; NUM_FP16_TABLES] = core::array::from_fn(|t| {
            if t < NUM_FP16_MAIN_TABLES {
                log2_strict(main_heights[t])
            } else {
                log2_strict(crate::v4::circuit::luts::lut_height(FP16_LUT_TABLES[t - NUM_FP16_MAIN_TABLES]))
            }
        });
        let config = fp16_stark_config(&degree_bits);
        let ctls = all_cross_table_lookups::<F>(h, w, k);

        Self {
            matmul,
            policy,
            xor_fold,
            blake3,
            a_hash_id,
            b_hash_id,
            row_scale,
            noisy_quant,
            noisy_quant_fma,
            noise_matmul_a,
            noise_matmul_b,
            noise_stark,
            noise_blake3,
            noise_seed_params,
            luts: FP16_LUT_TABLES.map(boxed_lut_stark::<F, D>),
            h,
            w,
            k,
            degree_bits,
            config,
            known,
            ctls,
        }
    }

    /// The config of this job's degree profile.
    pub fn config(&self) -> &StarkConfig {
        &self.config
    }

    /// Per-table trace degree bits in canonical batch order.
    pub fn degree_bits(&self) -> &[usize; NUM_FP16_TABLES] {
        &self.degree_bits
    }

    /// The tile geometry (`|I_A|`, `|I_B|`, inner `k`) — used by the header-bound gateways for the
    /// native `check_jackpot_difficulty` call.
    pub(crate) fn tile_geometry(&self) -> (usize, usize, usize) {
        (self.h, self.w, self.k)
    }

    /// The (shared) row height of the two main tables.
    pub fn num_rows(&self) -> usize {
        self.matmul.num_rows()
    }

    /// The root-derived noise seeds for this tile's committed operands: `noise_seeds({key_a, key_b},
    /// {root_a, root_b}, {p_a, p_b})` with `root_X = commit_operand(..).root()` and `p` this system's
    /// [`Self::new`]-time [`fp16_noise_seed_params`]. The single source of truth for the witness noise
    /// (`generate_batch_traces`) — the verify path recomputes the same seeds straight from the Blake3
    /// public inputs ([`Self::noise_seeds_from_blake3_pis`]).
    pub(crate) fn root_derived_seeds(&self, a_codes: &[u16], b_codes: &[u16], key_a: Hash256, key_b: Hash256) -> Result<Sides<Hash256>> {
        // Fold THIS system's `p` encodings (the [`Self::new`] placeholder by default, or the real
        // header-derived job's encodings when [`Self::set_noise_seed_params`] has been called by the
        // header gateway's honest prover) into the seed chain -- so the prover and
        // [`Self::verify_with_headers`] agree on `p`. Equivalent to [`fp16_root_derived_seeds`] when
        // `noise_seed_params` is the default placeholder.
        let root_a = commit_operand(a_codes, self.h, self.k, self.a_hash_id, key_a)?.root();
        let root_b = commit_operand(b_codes, self.w, self.k, self.b_hash_id, key_b)?.root();
        Ok(noise_seeds(
            &Sides { a: key_a, b: key_b },
            &Sides { a: root_a, b: root_b },
            &self.noise_seed_params,
        ))
    }

    /// Overrides the per-side noise-seed public-parameter encodings `p_A`/`p_B` (default: the
    /// [`Self::new`] placeholder job's [`fp16_noise_seed_params`]). The header gateway's honest prover
    /// sets these to the REAL job's [`Fp16JobParams::encode_p_a`]/`encode_p_b`, so the prover's seed
    /// chain folds the header-derived public parameters that [`Self::verify_with_headers`] recomputes
    /// from the job and pins to. Only affects the seed derivation ([`Self::root_derived_seeds`]) and
    /// the noise-BLAKE3 `KEY_A`/`KEY_B` public-input pins; the table geometry is `p`-independent.
    pub(crate) fn set_noise_seed_params(&mut self, p: Sides<Vec<u8>>) {
        self.noise_seed_params = p;
    }

    /// The root-derived noise seeds read straight from a Blake3 public-input vector: the opening keys
    /// (`PI_KEY_A`/`PI_KEY_B`) and the committed operand roots (`PI_HASH_A`/`PI_HASH_B`) folded with
    /// this system's `p` encodings through [`noise_seeds`]. Because [`Self::verify`] pins
    /// `HASH_A`/`HASH_B`/`KEY_A`/`KEY_B` to the expected (header-derived) limbs and this is what fixes
    /// the noise-BLAKE3 `KEY_A`/`KEY_B` public inputs, a prover cannot substitute seeds not derived
    /// from the committed roots.
    fn noise_seeds_from_blake3_pis(&self, blake3_pis: &[F]) -> Sides<Hash256> {
        let read = |base: usize| -> Hash256 {
            let limbs: &[F; 8] = blake3_pis[base..base + 8].try_into().expect("8 Blake3 PI limbs");
            u32_field_array_to_hash(limbs)
        };
        let keys = Sides { a: read(PI_KEY_A), b: read(PI_KEY_B) };
        let roots = Sides { a: read(PI_HASH_A), b: read(PI_HASH_B) };
        noise_seeds(&keys, &roots, &self.noise_seed_params)
    }

    /// The all-channel CTL set (four LUT channels + census import) in canonical table order — for
    /// the recursive wrapper, which re-runs the batch verifier in-circuit.
    pub(crate) fn ctls(&self) -> &[CrossTableLookup<F>] {
        &self.ctls
    }

    /// The class (a) known columns in canonical table order. The column indices shape the wrapper
    /// circuit; the values feed the native evaluation-at-zeta recompute. The digest slot is filled
    /// by [`Self::bind_statement_digest`].
    pub(crate) fn known(&self) -> &BatchKnownColumns<F> {
        &self.known
    }

    /// Binds `statement_digest` into the known-column Fiat-Shamir slot. [`Self::prove`] must be
    /// preceded by this (with the digest of the statement) for the wrapper paths, whose in-circuit
    /// replay pins the digest; the native [`Self::verify`] is unaffected by the slot's contents.
    pub fn bind_statement_digest(&mut self, statement_digest: Hash256) {
        self.known.digest = Some(statement_digest_to_hash_out(statement_digest));
    }

    /// Cross-checks a caller-supplied `statement_digest` against the one bound into this system, if
    /// any: a mismatch means the system and digest came from different statements.
    pub fn ensure_statement_digest_binding(&self, statement_digest: Hash256) -> Result<()> {
        ensure!(
            self.known
                .digest
                .is_none_or(|bound| bound == statement_digest_to_hash_out(statement_digest)),
            "the statement_digest disagrees with the one bound into the system's known columns"
        );
        Ok(())
    }

    /// The batch tables in canonical order: matmul, policy, xor_fold, blake3, then the five
    /// committed LUTs.
    pub(crate) fn batch_starks(&self) -> [&dyn BatchStark<F, D>; NUM_FP16_TABLES] {
        core::array::from_fn(|t| {
            if t == MATMUL_A100_TABLE {
                &self.matmul as &dyn BatchStark<F, D>
            } else if t == super::ctl::POLICY_A100_TABLE {
                &self.policy as &dyn BatchStark<F, D>
            } else if t == XOR_FOLD_A100_TABLE {
                &self.xor_fold as &dyn BatchStark<F, D>
            } else if t == BLAKE3_A100_TABLE {
                &self.blake3 as &dyn BatchStark<F, D>
            } else if t == ROW_SCALE_A100_TABLE {
                &self.row_scale as &dyn BatchStark<F, D>
            } else if t == NOISY_QUANT_A100_TABLE {
                &self.noisy_quant as &dyn BatchStark<F, D>
            } else if t == NOISY_QUANT_FMA_A100_TABLE {
                &self.noisy_quant_fma as &dyn BatchStark<F, D>
            } else if t == NOISE_MATMUL_A_A100_TABLE {
                &self.noise_matmul_a as &dyn BatchStark<F, D>
            } else if t == NOISE_MATMUL_B_A100_TABLE {
                &self.noise_matmul_b as &dyn BatchStark<F, D>
            } else if t == NOISE_STARK_A100_TABLE {
                &self.noise_stark as &dyn BatchStark<F, D>
            } else if t == NOISE_BLAKE3_A100_TABLE {
                &self.noise_blake3 as &dyn BatchStark<F, D>
            } else {
                self.luts[t - NUM_FP16_MAIN_TABLES].as_ref()
            }
        })
    }

    /// The batch public inputs: only the Blake3 table carries any (its 64-limb hash/key slots, with
    /// `HASH_JACKPOT` the lottery output); every other table is public-input-free.
    pub(crate) fn batch_public_inputs(&self, blake3_pis: &[F]) -> [Vec<F>; NUM_FP16_TABLES] {
        core::array::from_fn(|t| {
            if t == BLAKE3_A100_TABLE {
                blake3_pis.to_vec()
            } else if t == NOISE_BLAKE3_A100_TABLE {
                // The noise-BLAKE3 table's KEY_A/KEY_B public inputs ARE the ROOT-DERIVED noise seeds
                // (6e-3d): derive them from the operand roots + keys carried in `blake3_pis` plus this
                // system's `p` encodings. In `verify` `blake3_pis` is the pinned expectation, so this
                // forces the proof's seeds to be `noise_seeds` of the committed roots.
                let seeds = self.noise_seeds_from_blake3_pis(blake3_pis);
                noise_blake3_public_inputs::<F>(&seeds.a, &seeds.b)
            } else {
                Vec::new()
            }
        })
    }

    /// The setup-time commitment to the four LUTs' precommitted columns, placed at their batch
    /// positions (`2..6`). The flat column list is job-independent, so the Merkle cap is a
    /// consensus constant.
    pub fn preprocessed_data<C: GenericConfig<D, F = F>>(&self, timing: &mut TimingTree) -> BatchStarkPreprocessedData<F, C, D> {
        let mut values = vec![Vec::new(); NUM_FP16_TABLES];
        let mut columns = vec![Vec::new(); NUM_FP16_TABLES];
        for (i, &table) in FP16_LUT_TABLES.iter().enumerate() {
            let pos = fp16_lut_table_idx(i);
            values[pos] = lut_precommitted_values::<F>(table)
                .into_iter()
                .map(PolynomialValues::new)
                .collect();
            columns[pos] = (0..num_precommitted_columns(table)).collect();
        }
        BatchStarkPreprocessedData::new(values, columns, &self.config, timing)
    }

    /// The verifier's view of the LUT precommitment: the consensus `cap` plus the per-table
    /// preprocessed column indices (pinned by the statement, not taken from the proof).
    pub fn preprocessed_verifier_data<C: GenericConfig<D, F = F>>(
        &self,
        cap: &MerkleCap<F, C::Hasher>,
    ) -> BatchStarkPreprocessedVerifierData<F, C, D> {
        let mut columns_per_table = vec![Vec::new(); NUM_FP16_TABLES];
        for (i, &table) in FP16_LUT_TABLES.iter().enumerate() {
            columns_per_table[fp16_lut_table_idx(i)] = (0..num_precommitted_columns(table)).collect();
        }
        BatchStarkPreprocessedVerifierData {
            cap: cap.clone(),
            columns_per_table,
        }
    }

    /// Proves one tile: regenerates the matmul and policy traces from the operand codes (the
    /// policy's per-step census is bound to the matmul's by the census-import CTL), accumulates
    /// the committed-LUT multiplicities over both inventories, assembles the LUT traces, and runs
    /// the batch prover. `a_codes` is `h*k` row-major FP16, `b_codes` is `w*k` row-major over the
    /// transposed B. `preprocessed` is the setup-time LUT commitment ([`Self::preprocessed_data`]).
    pub fn prove<C: GenericConfig<D, F = F>>(
        &mut self,
        a_codes: &[u16],
        b_codes: &[u16],
        key_a: Hash256,
        key_b: Hash256,
        jackpot_key: Hash256,
        preprocessed: &BatchStarkPreprocessedData<F, C, D>,
        timing: &mut TimingTree,
    ) -> Result<BatchStarkProofWithPublicInputs<F, C, D>> {
        let (traces, blake3_pis) = self.generate_batch_traces(a_codes, b_codes, key_a, key_b, jackpot_key)?;
        // Derive the Fiat-Shamir salt from the PROVEN jackpot (the FP8 pattern — the digest is no
        // longer a caller-opaque salt), and bind it into the known-column slot before the batch
        // prover absorbs the known columns. [`Self::statement_digest`] recovers it for the wrapper.
        let statement_digest = derive_statement_digest(hash_jackpot::<F>(&blake3_pis));
        self.bind_statement_digest(statement_digest);
        self.prove_batch_traces(traces, &blake3_pis, preprocessed, timing)
    }

    /// The Fiat-Shamir statement digest [`Self::prove`] derives from a proof's own `HASH_JACKPOT`
    /// limbs — [`derive_statement_digest`] of [`hash_jackpot`]. The wrapper paths
    /// ([`crate::v5::circuit::wrapper`]) pass this to [`Self::bind_statement_digest`]'s binding
    /// check and to the known-column digest public input, so a verified wrapped proof is pinned to
    /// the batch's own proven jackpot.
    pub fn statement_digest(blake3_pis: &[F]) -> Hash256 {
        derive_statement_digest(hash_jackpot::<F>(blake3_pis))
    }

    /// Generates the three quantization-chain traces (row-scale, noisy-quant G1/G3, G2 FMA) in
    /// canonical batch order, over EVERY operand row of both operands (A rows then B rows).
    /// The chain is: row-scale derives `(alpha, beta)` from `raw`; G1 computes `t = RNE(beta*N)`; G2
    /// computes `noised = fma(alpha, raw, t)`; G3 casts `noised` to FP16. `N = E@F^T` is computed
    /// natively (the A100 datapath) and fed as the witness noise. The internal CTLs then bind
    /// row-scale's scales to noisy-quant and G1/G3's `(t, noised)` to G2 inside the batch.
    fn generate_quant_traces(
        &self,
        seeds: Sides<Hash256>,
        a_codes: &[u16],
        b_codes: &[u16],
    ) -> ([Vec<PolynomialValues<F>>; 3], Vec<u16>, Vec<u16>) {
        let k = self.k;
        let r = FP16_QUANT_R;

        // Per-element witness arrays over EVERY operand row of both operands (A rows then B rows).
        let mut alpha_vec: Vec<u16> = Vec::new();
        let mut beta_vec: Vec<u16> = Vec::new();
        let mut raw_vec: Vec<u16> = Vec::new();
        let mut noise_vec: Vec<f32> = Vec::new();

        // Real, ROOT-DERIVED noise with DISTINCT F_A/F_B: N_A = E_A@F_A^T, N_B = E_B@F_B^T on the
        // committed A100 FP16 datapath, bit-exact with `sample_noise(noise_seeds(roots, keys, p))` +
        // `a100_matmul`.
        let (e_a, f_a, e_b, f_b) = fp16_tile_noise(seeds, self.h, self.w, k);
        for (codes, num_rows, e, f) in [(a_codes, self.h, &e_a, &f_a), (b_codes, self.w, &e_b, &f_b)] {
            let norms: Vec<(u16, u16)> =
                (0..num_rows).map(|i| row_norms(&codes[i * k..(i + 1) * k]).expect("in-envelope operand norms")).collect();
            // The scales come from the real plaintext kernel; the row-scale AIR re-proves their
            // derivation, and the scales-import CTL forces noisy-quant to consume them per row.
            let built = noisy_quantize(codes, e, f, &norms, r).expect("in-envelope quant operand");
            let noise = a100_matmul(e, f, None, num_rows, k, r);
            for i in 0..num_rows {
                for j in 0..k {
                    alpha_vec.push(built.alpha[i]);
                    beta_vec.push(built.beta[i]);
                    raw_vec.push(codes[i * k + j]);
                    noise_vec.push(noise[i * k + j]);
                }
            }
        }

        // G1 witness `t = beta*noise` (as f32), decomposed into the G2 input fields.
        let n = raw_vec.len();
        let t: Vec<f32> = (0..n).map(|e| bf16_to_f32(beta_vec[e]) * noise_vec[e]).collect();
        let t_sign: Vec<u64> = t.iter().map(|&x| u64::from(x.to_bits() >> 31)).collect();
        let t_mant: Vec<u64> = t
            .iter()
            .map(|&x| if (x.to_bits() >> 23) & 0xFF == 0 { 0 } else { (1u64 << 23) + u64::from(x.to_bits() & 0x7F_FFFF) })
            .collect();
        let t_exp: Vec<u64> = t.iter().map(|&x| u64::from((x.to_bits() >> 23) & 0xFF)).collect();

        // G2: derive `noised` from the single-rounding FMA (its proven output, read back to f32).
        let fma_rows = self.noisy_quant_fma.program.generate_trace::<F>(&alpha_vec, &raw_vec, &t_sign, &t_mant, &t_exp);
        let noised: Vec<f32> = (0..n)
            .map(|e| {
                let lo = fma_rows[e][FMA_COL_MAP.noised_lo].to_canonical_u64();
                let hi = fma_rows[e][FMA_COL_MAP.noised_hi].to_canonical_u64();
                f32::from_bits(((hi << 16) | lo) as u32)
            })
            .collect();

        // G1 + G3 (noisy-quant), fed the G2-derived `noised`.
        let nq_rows = self.noisy_quant.program.generate_trace::<F>(&alpha_vec, &raw_vec, &beta_vec, &noise_vec, &noised);
        // The proven noised FP16 codes (the `out` column) over every element, A rows then B rows.
        // These — not the clean inputs — are what the matmul multiplies, so the operand-codes (6d)
        // channel binds the matmul's codes to this exact `out` column.
        let noised_codes: Vec<u16> =
            (0..n).map(|e| nq_rows[e][NOISY_QUANT_COL_MAP.out].to_canonical_u64() as u16).collect();
        let noised_a = noised_codes[..self.h * k].to_vec();
        let noised_b = noised_codes[self.h * k..].to_vec();
        // Per-row scale derivation over A rows then B rows.
        let mut all_raw = a_codes.to_vec();
        all_raw.extend_from_slice(b_codes);
        let rs_rows = self.row_scale.program.generate_trace::<F>(&all_raw);

        ([column_major(&rs_rows), column_major(&nq_rows), column_major(&fma_rows)], noised_a, noised_b)
    }

    /// The proven noised FP16 operands the batch's matmul/policy multiply: the noisy-quant `out`
    /// column over the A rows (`h*k`) and the B rows (`w*k`), bit-exact with
    /// [`crate::v5::api::quantization::noisy_quantize`] over the committed `raw` and the ROOT-DERIVED
    /// noise ([`Self::root_derived_seeds`] over the committed roots / `key_a`/`key_b` / `p`). Exposed
    /// so tests can build the plaintext tile / jackpot and check the noised tile clears the policy gate
    /// over the SAME operands the matmul uses.
    pub(crate) fn noised_operands(&self, a_codes: &[u16], b_codes: &[u16], key_a: Hash256, key_b: Hash256) -> (Vec<u16>, Vec<u16>) {
        let seeds = self.root_derived_seeds(a_codes, b_codes, key_a, key_b).expect("in-envelope operands");
        let (_, noised_a, noised_b) = self.generate_quant_traces(seeds, a_codes, b_codes);
        (noised_a, noised_b)
    }

    /// Generates the batch traces (canonical order) from the operand codes and jackpot key,
    /// returning them together with the Blake3 table's witness-generated public inputs (the
    /// `HASH_JACKPOT` slot binds the lottery output). The matmul output tile is read off the matmul's
    /// finished cells, folded by XorFold into the 16 lottery lanes, and committed by the Blake3
    /// jackpot compression; the committed-LUT multiplicities are accumulated over all four main
    /// tables' inventories (which also re-validates every instance is served). Exposed so tests can
    /// tamper a trace cell before proving.
    pub fn generate_batch_traces(
        &self,
        a_codes: &[u16],
        b_codes: &[u16],
        key_a: Hash256,
        key_b: Hash256,
        jackpot_key: Hash256,
    ) -> Result<([Vec<PolynomialValues<F>>; NUM_FP16_TABLES], Vec<F>)> {
        ensure!(a_codes.len() == self.h * self.k, "a_codes must be h*k FP16 codes");
        ensure!(b_codes.len() == self.w * self.k, "b_codes must be w*k FP16 codes");

        // The committed operand byte streams: the exact `u16` LE row bytes the two Blake3 operand
        // trees (-> HASH_A / HASH_B) hash, bit-identical to `commit_operand`'s `rows_to_bytes` input.
        let a_bytes = rows_to_bytes(a_codes);
        let b_bytes = rows_to_bytes(b_codes);

        // ---- Noise seeds (6e-3d): operands -> commit -> roots -> `noise_seeds` -> noise. The seeds
        // are `noise_seeds({key_a, key_b}, {root_a, root_b}, {p_a, p_b})` over the natively-committed
        // roots (bit-exact with the Blake3 trace's HASH_A/HASH_B), so every noise value below is a pure
        // function of the COMMITTED operands — the noise is operand-dependent and fully anti-grind. ----
        let seeds = self.root_derived_seeds(a_codes, b_codes, key_a, key_b)?;

        // ---- Quantization chain (increments 6b/6c/6d): prove `noised = Q(alpha*raw + beta*N)` over
        // the COMMITTED `raw` (the clean `a_codes`/`b_codes`), and return the proven noised FP16 codes
        // (`out`) the matmul multiplies. The internal CTLs (row-scale -> noisy-quant scales, G1/G3 <->
        // G2 FMA pairing) bind the chain together; the operand-bytes (6c) and operand-codes (6d)
        // channels bind `raw` to the Blake3 commitment and `out` to the matmul operands below. ----
        let ([rs_cols, nq_cols, fma_cols_], noised_a, noised_b) = self.generate_quant_traces(seeds, a_codes, b_codes);

        // ---- Noise matmuls (6e-2): prove `N_A = E_A@F_A^T` and `N_B = E_B@F_B^T` in-batch over the
        // SAME real root-derived `E`/`F` (distinct `F_A`/`F_B`) the quant chain's noise came from, so
        // their cell results are bit-exact with the noise noisy-quant consumed (noise-word CTL, keyed by
        // the global element index). NoiseStark proves every line's `normalize_line`, its output bound
        // to these matmuls' `E`/`F` operands (E/F-operand CTL). ----
        let (e_a, f_a, e_b, f_b) = fp16_tile_noise(seeds, self.h, self.w, self.k);
        let noise_mat_a_cols = column_major(&self.noise_matmul_a.generate_trace(&e_a, &f_a, None));
        let noise_mat_b_cols = column_major(&self.noise_matmul_b.generate_trace(&e_b, &f_b, None));
        let noise_xof = fp16_tile_noise_xof(seeds, self.h, self.w, self.k);
        let noise_stark_cols = column_major(&self.noise_stark.program.generate_trace::<F>(&noise_xof));

        // ---- Noise-BLAKE3 (6e-3d): recompute every line's keyed-XOF in-circuit from the ROOT-DERIVED
        // seeds (fed as the engine's KEY_A/KEY_B) and egress each line's cv_out. The egress-pair CTL
        // binds those limbs to NoiseStark's XOF bytes, so the noise bytes are the keyed-BLAKE3 XOF of
        // the root-derived seeds — operand-dependent, closing the anti-grind gap. ----
        let (nb_seed_ka, nb_seed_kb) = noise_seed_key_words(&seeds.a, &seeds.b);
        let nb_aux = noise_blake3_aux_msgs(self.h, self.w, self.k);
        let (nb_rows, nb_pis) = self.noise_blake3.program.generate_trace::<F>(&Fp16RawBlake3TraceInputs {
            a_values: &[],
            a_scales: &[],
            b_values: &[],
            b_scales: &[],
            routing_words: &[],
            offsets_words: &[],
            aux_msgs: &nb_aux,
            aux_cvs: &[],
            lottery_words: NOISE_BLAKE3_LOTTERY_WORDS,
            key_a: nb_seed_ka,
            key_b: nb_seed_kb,
            jackpot_key: bytes32_to_words(&NOISE_BLAKE3_JACKPOT_KEY),
            a_hash_id: HashId::Blake3Chunk1024,
            b_hash_id: HashId::Blake3Chunk1024,
            routing_hash_id: HashId::Blake3Chunk1024,
            offsets_hash_id: HashId::Blake3Chunk1024,
        });
        let noise_blake3_cols = column_major(&nb_rows);
        let noise_blake3_pis = noise_blake3_public_inputs::<F>(&seeds.a, &seeds.b);
        debug_assert_eq!(
            nb_pis.to_vec(),
            noise_blake3_pis,
            "noise-BLAKE3 witness public inputs must equal the root-derived seed pins"
        );

        // ---- Matmul: the output tile and its per-cell f32 result words, over the NOISED operands. ----
        let mat_rows = self.matmul.generate_trace(&noised_a, &noised_b, None);
        let mut cell_words = vec![0u32; self.h * self.w];
        for row in &mat_rows {
            let v: &MatmulA100ColumnsView<F> = row.borrow();
            if v.is_cell_final == F::ONE && v.is_padding == F::ZERO {
                let cell = v.cell_id.to_canonical_u64() as usize;
                cell_words[cell] =
                    (v.cell_result_f32_lo.to_canonical_u64() | (v.cell_result_f32_hi.to_canonical_u64() << 16)) as u32;
            }
        }
        let mat_cols = column_major(&mat_rows);
        // The policy scores the NOISED tile (the census-import CTL binds its per-step census to the
        // matmul's, so both must be over the same noised operands).
        let pol_cols = column_major(&self.policy.generate_trace(&noised_a, &noised_b));

        // ---- XorFold: fold the matmul's finished cell words into the 16 lottery lanes. ----
        let xf_rows = self.xor_fold.program.generate_trace::<F>(&cell_words);
        let mut lottery_words = [0u32; 16];
        for row in &xf_rows {
            let v: &XorFoldColumnsView<F> = row.borrow();
            if v.is_lane_final == F::ONE {
                lottery_words[v.lane_id.to_canonical_u64() as usize] = (v.rotation_input_top13.to_canonical_u64()
                    + (v.rotation_input_bottom19_limb_0.to_canonical_u64() << 13)
                    + (v.rotation_input_bottom19_limb_1.to_canonical_u64() << 29))
                    as u32;
            }
        }
        let xf_cols = column_major(&xf_rows);

        // ---- Blake3: the A/B operand Merkle trees (-> HASH_A / HASH_B under KEY_A / KEY_B) and the
        // keyed compression of the 16 lottery words (-> HASH_JACKPOT). The operand bytes fed here are
        // the committed CLEAN `a_codes`/`b_codes` (the quant chain's `raw`), so HASH_A / HASH_B are
        // bit-exact with `commit_operand(..).root()`. The operand-bytes (6c) CTL binds these committed
        // bytes to the quant `raw`, and the operand-codes (6d) CTL binds the quant `out` to the
        // matmul operands, so the matmul provably multiplies the noised quantization of THESE committed
        // operands. ----
        let (b3_rows, b3_pis) = self.blake3.program.generate_trace::<F>(&Blake3TraceInputs {
            a_values: &a_bytes,
            a_scales: &[],
            b_values: &b_bytes,
            b_scales: &[],
            routing_words: &[],
            offsets_words: &[],
            aux_msgs: &[],
            aux_cvs: &[],
            lottery_words,
            key_a: bytes32_to_words(&key_a),
            key_b: bytes32_to_words(&key_b),
            jackpot_key: bytes32_to_words(&jackpot_key),
            a_hash_id: self.a_hash_id,
            b_hash_id: self.b_hash_id,
            routing_hash_id: HashId::Blake3Chunk1024,
            offsets_hash_id: HashId::Blake3Chunk1024,
        });
        let b3_cols = column_major(&b3_rows);
        let b3_pis = b3_pis.to_vec();
        // The witness noise's seeds (derived from the natively-committed roots) must equal the seeds
        // the verify path derives straight from the proof's Blake3 public inputs (HASH_A/HASH_B +
        // keys) — i.e. the native commit roots match the proven HASH_A/HASH_B. Guarantees the proof's
        // pinned noise-BLAKE3 KEY_A/KEY_B equal the trace's.
        debug_assert_eq!(
            self.noise_seeds_from_blake3_pis(&b3_pis),
            seeds,
            "root-derived seeds must match the Blake3-PI-derived seeds"
        );

        // Committed-LUT multiplicities over all main tables' inventories; the checker also
        // re-validates every instance is served (fails fast on an unbalanceable proof).
        let mut checker = LutChecker::<F>::new(&FP16_LUT_TABLES);
        for (table_idx, lookups) in &fp16_lut_inventories::<F>() {
            let (trace, pis): (&Vec<PolynomialValues<F>>, &[F]) = match *table_idx {
                super::ctl::MATMUL_A100_TABLE => (&mat_cols, &[]),
                super::ctl::POLICY_A100_TABLE => (&pol_cols, &[]),
                XOR_FOLD_A100_TABLE => (&xf_cols, &[]),
                BLAKE3_A100_TABLE => (&b3_cols, &b3_pis),
                ROW_SCALE_A100_TABLE => (&rs_cols, &[]),
                NOISY_QUANT_A100_TABLE => (&nq_cols, &[]),
                NOISY_QUANT_FMA_A100_TABLE => (&fma_cols_, &[]),
                NOISE_MATMUL_A_A100_TABLE => (&noise_mat_a_cols, &[]),
                NOISE_MATMUL_B_A100_TABLE => (&noise_mat_b_cols, &[]),
                NOISE_STARK_A100_TABLE => (&noise_stark_cols, &[]),
                NOISE_BLAKE3_A100_TABLE => (&noise_blake3_cols, noise_blake3_pis.as_slice()),
                other => unreachable!("fp16 inventory table {other}"),
            };
            checker
                .check_trace(lookups, trace, pis, &format!("fp16 table {table_idx}"))
                .map_err(|e| anyhow!(e))?;
        }

        let mut traces: Vec<Vec<PolynomialValues<F>>> = vec![
            mat_cols,
            pol_cols,
            xf_cols,
            b3_cols,
            rs_cols,
            nq_cols,
            fma_cols_,
            noise_mat_a_cols,
            noise_mat_b_cols,
            noise_stark_cols,
            noise_blake3_cols,
        ];
        for table in FP16_LUT_TABLES {
            traces.push(lut_trace::<F>(table, checker.multiplicities.table_columns(table)));
        }
        let traces: [Vec<PolynomialValues<F>>; NUM_FP16_TABLES] = traces.try_into().map_err(|_| anyhow!("table count"))?;
        Ok((traces, b3_pis))
    }

    /// Runs the batch prover over pre-assembled traces and the Blake3 public inputs (the second half
    /// of [`Self::prove`]).
    pub fn prove_batch_traces<C: GenericConfig<D, F = F>>(
        &self,
        batch_traces: [Vec<PolynomialValues<F>>; NUM_FP16_TABLES],
        blake3_pis: &[F],
        preprocessed: &BatchStarkPreprocessedData<F, C, D>,
        timing: &mut TimingTree,
    ) -> Result<BatchStarkProofWithPublicInputs<F, C, D>> {
        for (t, trace) in batch_traces.iter().enumerate() {
            ensure!(
                log2_strict(trace[0].len()) == self.degree_bits[t],
                "batch table {t}: trace height differs from the statement's"
            );
        }
        batch_prove::<F, C, D, NUM_FP16_TABLES>(
            &self.batch_starks(),
            &self.config,
            batch_traces,
            &self.batch_public_inputs(blake3_pis),
            &self.ctls,
            &FP16_GROUPED_TABLES,
            Some(preprocessed),
            Some(&self.known),
            timing,
        )
    }

    /// Verifies one tile's proof against this statement and the consensus LUT cap: the degree
    /// profile must equal the statement's, the public inputs must be empty, and the batch verifier
    /// then checks every constraint, the class (a) openings against the recomputed known columns,
    /// the CTL balances and the batched FRI argument.
    pub fn verify<C: GenericConfig<D, F = F>>(
        &self,
        proof: &BatchStarkProofWithPublicInputs<F, C, D>,
        expected_blake3_pis: &[F],
        lut_cap: &MerkleCap<F, C::Hasher>,
    ) -> Result<()> {
        ensure!(
            proof.proof.degree_bits == self.degree_bits,
            "the proof's degree profile differs from the statement's"
        );
        ensure!(
            expected_blake3_pis.len() == NUM_BLAKE3_PUBLIC_INPUTS,
            "the expected Blake3 public inputs must be {NUM_BLAKE3_PUBLIC_INPUTS} limbs"
        );
        // Pin every public-input slot to the expectation — the Blake3 table's HASH_A/HASH_B/
        // HASH_JACKPOT/key slots, and (6e-3d) the noise-BLAKE3 table's KEY_A/KEY_B slots, which
        // `batch_public_inputs` sets to `noise_seeds` recomputed from the expected HASH_A/HASH_B +
        // keys + this system's `p`. So the proof's noise seeds are forced to be the root-derived ones;
        // a prover cannot substitute seeds not derived from the committed operand roots.
        self.verify_pinned::<C>(proof, &self.batch_public_inputs(expected_blake3_pis), lut_cap)
    }

    /// The degree-profile + full public-input pin + batched-constraint/CTL/FRI verification, against a
    /// fully-assembled expected public-input set (one `Vec<F>` per table). Both [`Self::verify`] (which
    /// builds the expectation from caller-supplied Blake3 limbs) and [`Self::verify_with_headers`]
    /// (which builds it from the block headers) funnel through here, so the batch argument is checked
    /// identically; only the source of the pinned expectation differs.
    fn verify_pinned<C: GenericConfig<D, F = F>>(
        &self,
        proof: &BatchStarkProofWithPublicInputs<F, C, D>,
        expected: &[Vec<F>; NUM_FP16_TABLES],
        lut_cap: &MerkleCap<F, C::Hasher>,
    ) -> Result<()> {
        ensure!(
            proof.proof.degree_bits == self.degree_bits,
            "the proof's degree profile differs from the statement's"
        );
        ensure!(
            proof.public_inputs == *expected,
            "the proof's public inputs differ from the expected ones"
        );
        // Only the Blake3 table declares public inputs; the matmul, policy and XorFold AIRs do not.
        const _: () = assert!(
            NUM_MATMUL_A100_PUBLIC_INPUTS == 0
                && NUM_POLICY_A100_PUBLIC_INPUTS == 0
                && super::xor_fold_stark::columns::NUM_XOR_FOLD_PUBLIC_INPUTS == 0
        );
        batch_verify::<F, C, D, NUM_FP16_TABLES>(
            &self.batch_starks(),
            &self.config,
            proof,
            &self.ctls,
            &FP16_GROUPED_TABLES,
            Some(&self.preprocessed_verifier_data::<C>(lut_cap)),
            Some(&self.known),
            &Default::default(),
        )
    }

    /// The header-pinned consensus verify GATEWAY: the sound ZK entry that mirrors the plaintext
    /// certificate's [`crate::v5::api::verify::verify_fp16_plain_proof`] key/seed derivation, so
    /// NOTHING feeding the noise seeds or the operand-tree commitment keys is test-supplied — it all
    /// comes from the block headers + the public job params, exactly as the plaintext path does.
    ///
    /// Given the proposed header, the proof-carried `job` (ancestor header, `k`/`r`/device, per-side
    /// `num_rows`/`hash_id`/pattern) and `nbits`, it:
    /// 1. derives `keyA = key_a(proposed)`, `keyB = key_b(ancestor)` ([`commitment_keys`]) and
    ///    `p_a`/`p_b` ([`Fp16JobParams::encode_p_a`]/`encode_p_b`) — bit-exact with the plaintext cert;
    /// 2. builds the EXPECTED Blake3 public inputs from the PROOF's committed `HASH_A`/`HASH_B`/
    ///    `HASH_JACKPOT` (the operand roots + lottery output, which — as in the plaintext cert, whose
    ///    root likewise comes from the proof — are not header-derivable) with the operand-tree
    ///    `KEY_A`/`KEY_B` OVERWRITTEN by the header-derived keys, and sets the noise-BLAKE3
    ///    `KEY_A`/`KEY_B` to `noise_seeds({keyA, keyB}, {HASH_A, HASH_B}, {p_a, p_b})`;
    /// 3. pins `proof.public_inputs` to that expected set (so a proof whose operand-tree keys or whose
    ///    noise seeds are not header-derived is rejected), runs the full batch verification, and checks
    ///    `check_jackpot_difficulty` on the proven jackpot against `nbits`.
    ///
    /// The ONLY remaining consensus boundary is inherent: the caller supplies the header + `nbits`, and
    /// is responsible for authenticating the `job.ancestor_header` as a member of the state window the
    /// same way the plaintext FFI does ([`crate::bindings`]' hash-walk) before calling this — the
    /// gateway assumes that window authentication, exactly as the plaintext `verify_fp16_plain_proof`
    /// assumes its caller has.
    /// Header-bound EXPECTED operand-Blake3 public inputs: the proof's own committed `HASH_A`/
    /// `HASH_B`/`HASH_JACKPOT` (operand roots + lottery output — not header-derivable, exactly as
    /// the plaintext cert takes the root from the proof) with the three keys OVERWRITTEN by their
    /// header/root-derived values:
    /// * `PI_KEY_A = key_a(proposed)`, `PI_KEY_B = key_b(ancestor)` (the operand-tree opening keys);
    /// * `PI_JACKPOT_KEY = jackpot_key(seed_a)` — the lottery-compression key, where
    ///   `seed_a = noise_seeds(keys, {HASH_A, HASH_B}, p).a`, exactly as the plaintext
    ///   [`compute_jackpot_ticket`](crate::v4::api::transcript::compute_jackpot_ticket) keys the
    ///   ticket. Pinning it stops a prover grinding the jackpot key to hit difficulty without redoing
    ///   the tile work.
    ///
    /// Returns the expected operand-Blake3 PIs and the root-derived noise `seeds` (which
    /// [`noise_blake3_public_inputs`] turns into the noise-BLAKE3 table's `KEY_A`/`KEY_B` pins). This
    /// is the single source of truth for the key/seed/pow-key derivation shared by
    /// [`Self::verify_with_headers`] and the wrapper's header-bound gateway
    /// ([`crate::v5::circuit::wrapper::verify_wrapped_proof_with_headers`]).
    pub(crate) fn header_bound_blake3_pis(
        &self,
        proof_blake3_pis: &[F],
        proposed_header: &IncompleteBlockHeader,
        job: &Fp16JobParams,
    ) -> (Vec<F>, Sides<Hash256>) {
        let keys = commitment_keys(proposed_header, &job.ancestor_header);
        let p = Sides { a: job.encode_p_a(), b: job.encode_p_b() };
        let hash_to_words = |h: Hash256| -> [F; 8] {
            core::array::from_fn(|i| F::from_canonical_u32(u32::from_le_bytes(h[4 * i..4 * i + 4].try_into().unwrap())))
        };
        let mut e = proof_blake3_pis.to_vec();
        e[PI_KEY_A..PI_KEY_A + 8].copy_from_slice(&hash_to_words(keys.a));
        e[PI_KEY_B..PI_KEY_B + 8].copy_from_slice(&hash_to_words(keys.b));
        let root_a: [F; 8] = e[PI_HASH_A..PI_HASH_A + 8].try_into().unwrap();
        let root_b: [F; 8] = e[PI_HASH_B..PI_HASH_B + 8].try_into().unwrap();
        let roots = Sides { a: u32_field_array_to_hash(&root_a), b: u32_field_array_to_hash(&root_b) };
        let seeds = noise_seeds(&keys, &roots, &p);
        e[PI_JACKPOT_KEY..PI_JACKPOT_KEY + 8].copy_from_slice(&hash_to_words(jackpot_key(&seeds.a)));
        (e, seeds)
    }

    pub fn verify_with_headers<C: GenericConfig<D, F = F>>(
        &self,
        proof: &BatchStarkProofWithPublicInputs<F, C, D>,
        proposed_header: &IncompleteBlockHeader,
        job: &Fp16JobParams,
        nbits: u32,
        lut_cap: &MerkleCap<F, C::Hasher>,
    ) -> Result<()> {
        ensure!(
            proof.public_inputs[BLAKE3_A100_TABLE].len() == NUM_BLAKE3_PUBLIC_INPUTS,
            "the proof's Blake3 public inputs must be {NUM_BLAKE3_PUBLIC_INPUTS} limbs"
        );
        // 1-3. Header-derived expected operand-Blake3 PIs (keyA/keyB + the lottery-compression
        //    jackpot key pinned to their header/root-derived values) and the root-derived noise seeds.
        let (expected_blake3, seeds) =
            self.header_bound_blake3_pis(&proof.public_inputs[BLAKE3_A100_TABLE], proposed_header, job);
        let expected: [Vec<F>; NUM_FP16_TABLES] = core::array::from_fn(|t| {
            if t == BLAKE3_A100_TABLE {
                expected_blake3.clone()
            } else if t == NOISE_BLAKE3_A100_TABLE {
                noise_blake3_public_inputs::<F>(&seeds.a, &seeds.b)
            } else {
                Vec::new()
            }
        });

        // 4. Pin + full batch verification, then the native difficulty check on the proven jackpot —
        //    reusing `check_jackpot_difficulty` exactly as the plaintext `verify_tile_proof` does.
        self.verify_pinned::<C>(proof, &expected, lut_cap)?;
        let jackpot = hash_jackpot::<F>(&proof.public_inputs[BLAKE3_A100_TABLE]);
        check_jackpot_difficulty(&jackpot, nbits, self.h as u32, self.w as u32, self.k as u32)
    }

    /// The consensus-meaningful output gateway (the FP8 `zk.rs` `verify` + `native_epilogue`
    /// analogue): runs [`Self::verify`] — the full batch verification plus the public-input pinning
    /// that forces `HASH_A`/`HASH_B`/`HASH_JACKPOT` and the three keys to the caller-supplied
    /// `expected_blake3_pis` (the consensus layer supplies these from the block header's proof
    /// commitment + opening keys) — then applies the native difficulty check on the proven
    /// `HASH_JACKPOT`, reusing [`check_jackpot_difficulty`] exactly as the plaintext
    /// [`crate::v5::api::verify::verify_tile_proof`] and the FP8 `zk.rs` epilogue do. Because
    /// [`Self::verify`] has pinned `proof.public_inputs == expected`, the `HASH_JACKPOT` the
    /// difficulty check reads off the proof equals the expected limbs. Returns `Ok(())` only if the
    /// proof verifies, its public inputs equal the expectation, and the jackpot clears `nbits`.
    pub fn verify_with_difficulty<C: GenericConfig<D, F = F>>(
        &self,
        proof: &BatchStarkProofWithPublicInputs<F, C, D>,
        expected_blake3_pis: &[F],
        nbits: u32,
        lut_cap: &MerkleCap<F, C::Hasher>,
    ) -> Result<()> {
        self.verify::<C>(proof, expected_blake3_pis, lut_cap)?;
        // The plain winning condition on the proven lottery digest (pinned equal to the expectation
        // by the public-input check above), against the block's `nbits` — the succinct analogue of
        // the plaintext certificate's difficulty epilogue.
        let jackpot = hash_jackpot::<F>(&proof.public_inputs[BLAKE3_A100_TABLE]);
        check_jackpot_difficulty(&jackpot, nbits, self.h as u32, self.w as u32, self.k as u32)
    }
}

/// The canonical lottery layout for an `h x w` tile: a pair of [`AxisPattern`]s whose Blake digits
/// select exactly [`JACKPOT_ENTRIES`](crate::v4::api::layout::JACKPOT_ENTRIES) `= 16` lanes, covering all
/// `h*w` cells. We split the 16 lanes as `br * bc` with `br | h` and `bc | w` (preferring the largest
/// `br`), putting the remaining factor of each axis into a leading Fold dim. `h*w` must admit such a
/// split (every geometry the extractor envelope allows does); a mismatched geometry panics.
pub fn default_lane_layout(h: usize, w: usize) -> (AxisPattern, AxisPattern) {
    let axis = |fold: usize, blake: usize| -> AxisPattern {
        let mut dims = Vec::new();
        if fold > 1 {
            dims.push((fold as u32, DimType::Fold));
        }
        if blake > 1 {
            dims.push((blake as u32, DimType::Blake));
        }
        if dims.is_empty() {
            dims.push((1, DimType::Fold));
        }
        AxisPattern::new(&dims).expect("valid canonical axis pattern")
    };
    for br in (1..=16).rev() {
        if 16 % br != 0 {
            continue;
        }
        let bc = 16 / br;
        if h % br == 0 && w % bc == 0 {
            return (axis(h / br, br), axis(w / bc, bc));
        }
    }
    panic!("tile {h}x{w} admits no 16-lane Blake split (need br|h, bc|w, br*bc=16)");
}

/// Row-major trace rows -> the column-major `PolynomialValues` layout of the batch prover.
fn column_major<F: RichField, const N: usize>(rows: &[[F; N]]) -> Vec<PolynomialValues<F>> {
    (0..N)
        .map(|c| PolynomialValues::new(rows.iter().map(|r| r[c]).collect()))
        .collect()
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::plonk::config::PoseidonGoldilocksConfig;

    use super::*;
    use crate::v5::api::accumulate::{GROUP as ACC_GROUP, a100_dot};
    use crate::v5::api::policy::replay_and_evaluate;
    use crate::v4::api::transcript::{compute_jackpot_ticket, jackpot_key};
    use crate::v4::api::utils::xor_fold_extract;
    use crate::v4::api::layout::lane_assignment;
    use crate::v5::api::commitment::commit_operand;
    use crate::v4::api::proof_utils::check_jackpot_difficulty;
    use crate::v4::circuit::blake3_stark::columns::{PI_HASH_A, PI_HASH_B, PI_HASH_JACKPOT};
    use super::super::ctl::{
        BLAKE3_A100_TABLE, MATMUL_A100_TABLE, NOISE_MATMUL_A_A100_TABLE, NOISE_MATMUL_B_A100_TABLE,
        NOISY_QUANT_A100_TABLE, NOISY_QUANT_FMA_A100_TABLE, ROW_SCALE_A100_TABLE, XOR_FOLD_A100_TABLE,
    };
    use super::super::matmul_a100_stark::columns::MATMUL_A100_COL_MAP;
    use super::super::noisy_quant_fma_stark::columns::FMA_COL_MAP;
    use super::super::noisy_quant_stark::columns::NOISY_QUANT_COL_MAP;
    use super::super::row_scale_stark::columns::ROW_SCALE_COL_MAP;
    use super::super::xor_fold_stark::columns::XOR_FOLD_COL_MAP;
    use crate::v5::api::quantization::{noisy_quantize, row_norms};

    type F = GoldilocksField;
    type C = PoseidonGoldilocksConfig;
    const D: usize = 2;

    const VECTORS: &str = include_str!("../api/testdata/a100_dot_vectors.txt");
    const SEED_A: Hash256 = [0x3c; 32];
    /// The operand-tree opening keys and leaf size for these tests. (In production these come from
    /// the seed chain / header; passing them in is this increment's documented scope — see the
    /// driver module docs.) `Blake3Chunk512` keeps the `4x4`/`4x16` Blake3 tables on the consensus
    /// ladder (`2^8`).
    const KEY_A: Hash256 = [0x11; 32];
    const KEY_B: Hash256 = [0x22; 32];
    const OP_HASH_ID: HashId = HashId::Blake3Chunk512;

    /// The 8 LE `u32` field words of `commit_operand(rows, num_rows, k, OP_HASH_ID, key).root()`.
    fn operand_root_words(rows: &[u16], num_rows: usize, k: usize, key: Hash256) -> Vec<F> {
        let root = commit_operand(rows, num_rows, k, OP_HASH_ID, key).unwrap().root();
        (0..8)
            .map(|i| F::from_canonical_u32(u32::from_le_bytes(root[4 * i..4 * i + 4].try_into().unwrap())))
            .collect()
    }

    /// The smallest `1 x 1 x k` cell (`k >= 16`, `32 | GROUP | k`) from the GPU reference vectors
    /// whose NOISED `4 x 4` tile (`a0`/`b0` on every A/B row, each noised under the fixed witness
    /// noise) clears the policy gate — the operands the batch's matmul actually multiplies. (Noising
    /// adds breakpoints, so a noised tile typically clears the gate more readily than its clean twin.)
    fn accepting_cell() -> (usize, Vec<u16>, Vec<u16>) {
        let mut best: Option<(usize, Vec<u16>, Vec<u16>)> = None;
        for line in VECTORS.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let t: Vec<&str> = line.split_whitespace().collect();
            let k: usize = t[0].parse().unwrap();
            if k % ACC_GROUP != 0 || k < 16 {
                continue;
            }
            if best.as_ref().is_some_and(|(bk, _, _)| k >= *bk) {
                continue;
            }
            let a0: Vec<u16> = t[1..1 + k].iter().map(|x| x.parse().unwrap()).collect();
            let b0: Vec<u16> = t[1 + k..1 + 2 * k].iter().map(|x| x.parse().unwrap()).collect();
            let (a, b) = (a0.repeat(4), b0.repeat(4));
            let Ok((noised_a, noised_b)) = fp16_noised_operands(4, 4, k, &a, &b, KEY_A, KEY_B, OP_HASH_ID, OP_HASH_ID) else { continue };
            if replay_and_evaluate(&noised_a, &noised_b, 4, 4, k).1.accept {
                best = Some((k, a0, b0));
            }
        }
        best.expect("reference vectors must contain a cell whose noised 4x4 tile accepts")
    }

    /// The 8 LE `u32` field words of the plaintext jackpot digest for the `h x w` tile whose cell
    /// `(r, c)` is the A100 dot of the NOISED operand rows `(noised_a[r], noised_b[c])` — the same
    /// noised operands the batch's matmul multiplies (the matmul is over the quant chain's `out`).
    fn plaintext_jackpot_words(noised_a: &[u16], noised_b: &[u16], h: usize, w: usize, k: usize) -> Vec<F> {
        let mut tile = vec![0f32; h * w];
        for r in 0..h {
            for c in 0..w {
                tile[r * w + c] = a100_dot(&noised_a[r * k..r * k + k], &noised_b[c * k..c * k + k], 0.0, None);
            }
        }
        let (row_axis, col_axis) = default_lane_layout(h, w);
        let msg = xor_fold_extract(&tile, &lane_assignment(&row_axis, &col_axis));
        let jk = compute_jackpot_ticket(&SEED_A, &msg).jackpot;
        (0..8)
            .map(|i| F::from_canonical_u32(u32::from_le_bytes(jk[4 * i..4 * i + 4].try_into().unwrap())))
            .collect()
    }

    /// The driver's output -> jackpot chain end to end: an honest tile proves and verifies, its
    /// committed `HASH_JACKPOT` equals the plaintext jackpot, and a tampered cell result and a
    /// tampered lottery word are rejected by the batch verifier.
    #[test]
    fn batch_binds_output_tile_to_hash_jackpot_and_rejects_tampering() {
        let (k, a0, b0) = accepting_cell();
        // A 4x4 tile: 16 output cells, one per lottery lane.
        let (h, w) = (4usize, 4usize);
        let (a, b) = (a0.repeat(h), b0.repeat(w));
        let mut system = Fp16System::<F, D>::new(h, w, k, OP_HASH_ID, OP_HASH_ID);
        let jk = jackpot_key(&SEED_A);
        let mut timing = TimingTree::default();
        let preprocessed = system.preprocessed_data::<C>(&mut timing);
        let lut_cap = preprocessed.cap();

        // ---- Honest proof: proves and verifies. ----
        let proof = system
            .prove::<C>(&a, &b, KEY_A, KEY_B, jk, &preprocessed, &mut timing)
            .expect("honest FP16 tile must prove");
        let blake3_pis = proof.public_inputs[BLAKE3_A100_TABLE].clone();
        system
            .verify::<C>(&proof, &blake3_pis, &lut_cap)
            .expect("honest FP16 proof must verify");

        // ---- HASH_JACKPOT is bit-exact with the plaintext extractor -> jackpot transcript over the
        // NOISED operands the matmul multiplies (the quant chain's `out`). ----
        let (noised_a, noised_b) = system.noised_operands(&a, &b, KEY_A, KEY_B);
        assert_eq!(
            &blake3_pis[PI_HASH_JACKPOT..PI_HASH_JACKPOT + 8],
            plaintext_jackpot_words(&noised_a, &noised_b, h, w, k).as_slice(),
            "committed HASH_JACKPOT must equal the plaintext jackpot over the noised operands"
        );

        // ---- HASH_A / HASH_B are bit-exact with the committed operand Merkle roots the plaintext
        // certificate opens against (`commit_operand(..).root()`). A is `h x k`, B is `w x k`. ----
        assert_eq!(
            &blake3_pis[PI_HASH_A..PI_HASH_A + 8],
            operand_root_words(&a, h, k, KEY_A).as_slice(),
            "committed HASH_A must equal commit_operand(A).root()"
        );
        assert_eq!(
            &blake3_pis[PI_HASH_B..PI_HASH_B + 8],
            operand_root_words(&b, w, k, KEY_B).as_slice(),
            "committed HASH_B must equal commit_operand(B).root()"
        );

        // ---- Tampering: mutate one honest trace cell and assert the batch verifier rejects. ----
        let xf = &XOR_FOLD_COL_MAP;
        let (honest, _) = system.generate_batch_traces(&a, &b, KEY_A, KEY_B, jk).expect("honest traces");
        // Every live XorFold row here is lane-final (16 lanes of one cell); pick the first.
        let xf_live = (0..honest[XOR_FOLD_A100_TABLE][xf.is_pad].values.len())
            .find(|&r| honest[XOR_FOLD_A100_TABLE][xf.is_pad].values[r] == F::ZERO)
            .expect("a live XorFold row");

        let prove_tampered = |col: usize, row: usize| {
            let (mut traces, blake3_pis) = system.generate_batch_traces(&a, &b, KEY_A, KEY_B, jk).expect("honest traces");
            traces[XOR_FOLD_A100_TABLE][col].values[row] += F::ONE;
            let mut timing = TimingTree::default();
            match system.prove_batch_traces::<C>(traces, &blake3_pis, &preprocessed, &mut timing) {
                Ok(bad) => system.verify::<C>(&bad, &blake3_pis, &lut_cap).is_err(),
                Err(_) => true,
            }
        };

        // (a) a tampered cell result: the folded cell word no longer matches the matmul's proven
        // output, unbalancing the matmul -> XorFold cell-results channel.
        assert!(
            prove_tampered(xf.cell_result_f32_lo, xf_live),
            "a tampered cell result must be rejected by the cell-results CTL"
        );
        // (b) a tampered lottery word: the lane's folded output word changes, unbalancing the
        // XorFold -> Blake3 lottery-words channel (and breaking the mixer's rotation split).
        assert!(
            prove_tampered(xf.rotation_input_top13, xf_live),
            "a tampered lottery word must be rejected by the lottery-words CTL"
        );
    }

    /// The quantization chain (increment 6b) proves `noised = Q(alpha*raw + beta*N)` end-to-end
    /// in-batch: an honest tile's batch proof verifies, the proven noisy-quant `out` column is
    /// bit-exact with the plaintext `noisy_quantize` kernel over the same (fixed, in-envelope) witness
    /// operand, and tampering any of the three quant tables (row-scale scales, the G1/G3 noisy-quant,
    /// or the G2 FMA) is rejected by the batch verifier — via the quant AIR constraints, the shared
    /// LUTs, or the two internal quant-chain CTLs (scales import, FMA pairing).
    #[test]
    fn batch_proves_quant_chain_and_rejects_tampering() {
        let (k, a0, b0) = accepting_cell();
        let (h, w) = (4usize, 4usize);
        let (a, b) = (a0.repeat(h), b0.repeat(w));
        let mut system = Fp16System::<F, D>::new(h, w, k, OP_HASH_ID, OP_HASH_ID);
        let jk = jackpot_key(&SEED_A);
        let mut timing = TimingTree::default();
        let preprocessed = system.preprocessed_data::<C>(&mut timing);
        let lut_cap = preprocessed.cap();

        // ---- Honest proof: proves and verifies with the three quant tables wired in. ----
        let proof = system
            .prove::<C>(&a, &b, KEY_A, KEY_B, jk, &preprocessed, &mut timing)
            .expect("honest FP16 tile with quant chain must prove");
        let blake3_pis = proof.public_inputs[BLAKE3_A100_TABLE].clone();
        system.verify::<C>(&proof, &blake3_pis, &lut_cap).expect("honest quant-chain proof must verify");

        // ---- The proven noisy-quant `out` column is bit-exact with `noisy_quantize` over the FULL
        // A operand AND the FULL B operand (every operand row of both), over the REAL ROOT-DERIVED
        // noise with DISTINCT F_A/F_B (N_A = E_A@F_A^T, N_B = E_B@F_B^T), per side. ----
        let mut expected_out: Vec<u16> = Vec::new();
        let seeds = fp16_root_derived_seeds(h, w, k, OP_HASH_ID, OP_HASH_ID, KEY_A, KEY_B, &a, &b).unwrap();
        let (e_a, f_a, e_b, f_b) = fp16_tile_noise(seeds, h, w, k);
        for (codes, num_rows, e, f) in [(&a, h, &e_a, &f_a), (&b, w, &e_b, &f_b)] {
            let norms: Vec<(u16, u16)> =
                (0..num_rows).map(|i| row_norms(&codes[i * k..(i + 1) * k]).unwrap()).collect();
            let built = noisy_quantize(codes, e, f, &norms, FP16_QUANT_R).unwrap();
            expected_out.extend_from_slice(&built.noised_part);
        }
        let (honest, _) = system.generate_batch_traces(&a, &b, KEY_A, KEY_B, jk).expect("honest traces");
        for (j, &want) in expected_out.iter().enumerate() {
            let got = honest[NOISY_QUANT_A100_TABLE][NOISY_QUANT_COL_MAP.out].values[j].to_canonical_u64() as u16;
            assert_eq!(got, want, "proven noised out column mismatch at element {j}");
        }
        assert_eq!(expected_out.len(), (h + w) * k, "quant chain covers every operand row of both operands");

        // ---- Tampering any quant table is rejected by batch_verify. ----
        let prove_tampered = |table: usize, col: usize, row: usize| {
            let (mut traces, blake3_pis) = system.generate_batch_traces(&a, &b, KEY_A, KEY_B, jk).expect("honest traces");
            traces[table][col].values[row] += F::ONE;
            let mut timing = TimingTree::default();
            match system.prove_batch_traces::<C>(traces, &blake3_pis, &preprocessed, &mut timing) {
                Ok(bad) => system.verify::<C>(&bad, &blake3_pis, &lut_cap).is_err(),
                Err(_) => true,
            }
        };

        // (a) row-scale: a tampered derived scale breaks the row-scale AIR / the scales-import CTL.
        assert!(
            prove_tampered(ROW_SCALE_A100_TABLE, ROW_SCALE_COL_MAP.alpha_code, 0),
            "a tampered row-scale alpha must be rejected"
        );
        // (b) noisy-quant: a tampered FP16 output breaks the G3 cast encode.
        assert!(
            prove_tampered(NOISY_QUANT_A100_TABLE, NOISY_QUANT_COL_MAP.out, 0),
            "a tampered noisy-quant output must be rejected"
        );
        // (c) G2 FMA: a tampered `noised` breaks the FMA AIR / the FMA-pairing CTL.
        assert!(
            prove_tampered(NOISY_QUANT_FMA_A100_TABLE, FMA_COL_MAP.noised_lo, 0),
            "a tampered G2 FMA noised must be rejected"
        );
    }

    /// Increments 6c + 6d — the operand-provenance capstone (audit Finding 3). The matmul provably
    /// multiplies the NOISED quantization of the COMMITTED operands, end to end:
    /// * (a) the honest tile proves+verifies with the matmul over `noisy_quantize(committed a/b)` —
    ///   the `batch_proves_quant_chain` + `batch_binds_output_tile` tests already pin the `out` column
    ///   and the noised jackpot; here we pin HASH_A/HASH_B to the committed operands and fail-close the
    ///   two binding channels;
    /// * (d) HASH_A / HASH_B are bit-exact with `commit_operand` of the committed (clean) operands the
    ///   noised codes derive from;
    /// * (b) tampering a matmul operand code away from the noised committed value is rejected (the
    ///   noised->matmul operand-codes channel, 6d);
    /// * (c) tampering a committed operand byte (Blake3) is rejected (the operand-bytes channel, 6c).
    #[test]
    fn batch_binds_matmul_operands_to_committed_and_rejects_tampering() {
        use crate::v4::circuit::blake3_stark::columns::{BLAKE3_COL_MAP, PI_HASH_A, PI_HASH_B};
        let (k, a0, b0) = accepting_cell();
        let (h, w) = (4usize, 4usize);
        let (a, b) = (a0.repeat(h), b0.repeat(w));
        let mut system = Fp16System::<F, D>::new(h, w, k, OP_HASH_ID, OP_HASH_ID);
        let jk = jackpot_key(&SEED_A);
        let mut timing = TimingTree::default();
        let preprocessed = system.preprocessed_data::<C>(&mut timing);
        let lut_cap = preprocessed.cap();

        // ---- (a) honest proof over the noised committed operands verifies. ----
        let proof = system
            .prove::<C>(&a, &b, KEY_A, KEY_B, jk, &preprocessed, &mut timing)
            .expect("honest FP16 tile must prove");
        let blake3_pis = proof.public_inputs[BLAKE3_A100_TABLE].clone();
        system.verify::<C>(&proof, &blake3_pis, &lut_cap).expect("honest proof must verify");

        // ---- (d) HASH_A / HASH_B commit the clean operands the noised codes derive from. ----
        assert_eq!(
            &blake3_pis[PI_HASH_A..PI_HASH_A + 8],
            operand_root_words(&a, h, k, KEY_A).as_slice(),
            "HASH_A must equal commit_operand of the committed A operand"
        );
        assert_eq!(
            &blake3_pis[PI_HASH_B..PI_HASH_B + 8],
            operand_root_words(&b, w, k, KEY_B).as_slice(),
            "HASH_B must equal commit_operand of the committed B operand"
        );

        let prove_tampered = |table: usize, col: usize, row: usize| {
            let (mut traces, blake3_pis) = system.generate_batch_traces(&a, &b, KEY_A, KEY_B, jk).expect("honest traces");
            traces[table][col].values[row] += F::ONE;
            let mut timing = TimingTree::default();
            match system.prove_batch_traces::<C>(traces, &blake3_pis, &preprocessed, &mut timing) {
                Ok(bad) => system.verify::<C>(&bad, &blake3_pis, &lut_cap).is_err(),
                Err(_) => true,
            }
        };

        let m = &MATMUL_A100_COL_MAP;
        let bm = &BLAKE3_COL_MAP;
        let (honest, _) = system.generate_batch_traces(&a, &b, KEY_A, KEY_B, jk).expect("honest traces");
        let mat_live = (0..honest[MATMUL_A100_TABLE][m.is_padding].values.len())
            .find(|&r| honest[MATMUL_A100_TABLE][m.is_padding].values[r] == F::ZERO)
            .expect("a live matmul row");
        let b3_live = (0..honest[BLAKE3_A100_TABLE][bm.is_int8_message].values.len())
            .find(|&r| honest[BLAKE3_A100_TABLE][bm.is_int8_message].values[r] == F::ONE)
            .expect("a live operand-values row");

        // ---- (b) a matmul operand code tampered away from the noised committed value is rejected
        // (the operand-codes channel binds it to the noisy-quant `out`). ----
        assert!(
            prove_tampered(MATMUL_A100_TABLE, m.operand_codes_a[0], mat_live),
            "a tampered matmul operand code must be rejected by the operand-codes (6d) channel"
        );
        // ---- (c) a committed operand byte (Blake3) tampered is rejected (the operand-bytes channel
        // binds the committed byte to the quant `raw`). ----
        assert!(
            prove_tampered(BLAKE3_A100_TABLE, bm.uint8_data[0], b3_live),
            "a tampered committed operand byte must be rejected by the operand-bytes (6c) channel"
        );
    }

    /// The batch's committed operand roots (HASH_A / HASH_B) are bit-exact with
    /// `commit_operand(..).root()` across several tile shapes and every allowed leaf size — and the
    /// A and B sides are keyed independently (A under KEY_A / `a_hash_id`, B under KEY_B /
    /// `b_hash_id`). Exercises the whole generate_batch_traces -> Blake3 operand-tree path, including
    /// a mixed-`HashId` A/B job to prove the two trees do not cross-contaminate.
    #[test]
    fn batch_binds_operand_roots_to_hash_a_and_hash_b() {
        let (k, a0, b0) = accepting_cell();
        let jk = jackpot_key(&SEED_A);
        // (h, w, a_hash_id, b_hash_id): every leaf size, two shapes, plus a mixed-id job.
        let cases = [
            (4usize, 4usize, HashId::Blake3Chunk128, HashId::Blake3Chunk128),
            (2, 8, HashId::Blake3Chunk256, HashId::Blake3Chunk256),
            (4, 4, HashId::Blake3Chunk512, HashId::Blake3Chunk1024),
            (8, 2, HashId::Blake3Chunk1024, HashId::Blake3Chunk512),
        ];
        for (h, w, a_hash_id, b_hash_id) in cases {
            let (a, b) = (a0.repeat(h), b0.repeat(w));
            let system = Fp16System::<F, D>::new(h, w, k, a_hash_id, b_hash_id);
            let (_, blake3_pis) = system
                .generate_batch_traces(&a, &b, KEY_A, KEY_B, jk)
                .expect("honest traces");

            let root_a = commit_operand(&a, h, k, a_hash_id, KEY_A).unwrap().root();
            let root_b = commit_operand(&b, w, k, b_hash_id, KEY_B).unwrap().root();
            let words = |root: Hash256| -> Vec<F> {
                (0..8)
                    .map(|i| F::from_canonical_u32(u32::from_le_bytes(root[4 * i..4 * i + 4].try_into().unwrap())))
                    .collect()
            };
            assert_eq!(
                &blake3_pis[PI_HASH_A..PI_HASH_A + 8],
                words(root_a).as_slice(),
                "HASH_A mismatch for {h}x{w} a_id={a_hash_id:?}"
            );
            assert_eq!(
                &blake3_pis[PI_HASH_B..PI_HASH_B + 8],
                words(root_b).as_slice(),
                "HASH_B mismatch for {h}x{w} b_id={b_hash_id:?}"
            );
        }
    }

    /// Increment 6e-2 — the per-side noise matmuls with DISTINCT `F_A`/`F_B`. The `beta*N` noise the
    /// noisy-quant G1 rounding consumes is proven in-batch by two `MatmulStarkA100` instances —
    /// `N_A = E_A @ F_A^T` (`h x k`) and `N_B = E_B @ F_B^T` (`w x k`) — matching the plaintext
    /// `sample_noise`'s distinct-`F` layout (6e-1 used one shared `F`, which diverged). The noise-word
    /// CTL binds their cell results to noisy-quant's `(NOISE_LO, NOISE_HI)` (A cells -> `[0,h*k)`, B
    /// cells -> `[h*k,(h+w)*k)`), and the E/F-operand CTL binds their `E`/`F` operands to NoiseStark's
    /// proven normalized lines. This test shows:
    /// * (a) the honest tile proves+verifies with the two noise matmuls + NoiseStark wired in;
    /// * (b) the matmuls' cell results are bit-exact with `a100_matmul(e, f)` per side (distinct `F`) AND
    ///   equal the noise noisy-quant consumed (so the noised `out` is still bit-exact);
    /// * (c) tampering a noise matmul's `N` output at a finished cell is rejected (noise-word channel);
    /// * (d) tampering noisy-quant's `N` input (`NOISE_LO`) is rejected (same channel, consuming side).
    ///
    /// E/F are now bound to NoiseStark's proven normalized lines (the E/F-operand channel; a tampered
    /// line is rejected in `ctl::tests::forged_noise_line_breaks_the_ef_channel`).
    /// CAVEAT: the raw XOF bytes NoiseStark normalizes remain free witness (seed-keyed-BLAKE3-XOF
    /// binding is increment 6e-3), so the noise is still grindable through those bytes; but `E`/`F` are
    /// now provably the normalized seed-derived lines with distinct `F_A`/`F_B` matching the plaintext.
    #[test]
    fn batch_binds_noise_to_ef_matmul_and_rejects_tampering() {
        let (k, a0, b0) = accepting_cell();
        let (h, w) = (4usize, 4usize);
        let (a, b) = (a0.repeat(h), b0.repeat(w));
        let mut system = Fp16System::<F, D>::new(h, w, k, OP_HASH_ID, OP_HASH_ID);
        let jk = jackpot_key(&SEED_A);
        let mut timing = TimingTree::default();
        let preprocessed = system.preprocessed_data::<C>(&mut timing);
        let lut_cap = preprocessed.cap();

        // ---- (a) honest proof verifies. ----
        let proof = system
            .prove::<C>(&a, &b, KEY_A, KEY_B, jk, &preprocessed, &mut timing)
            .expect("honest FP16 tile with noise matmul must prove");
        let blake3_pis = proof.public_inputs[BLAKE3_A100_TABLE].clone();
        system.verify::<C>(&proof, &blake3_pis, &lut_cap).expect("honest noise-matmul proof must verify");

        // ---- (b) the proven noise is bit-exact with `a100_matmul` per side with DISTINCT F_A/F_B
        // (N_A = E_A@F_A^T, N_B = E_B@F_B^T) AND equals the noise noisy-quant consumed (A rows then B
        // rows, element index i*k + j). ----
        let r = FP16_QUANT_R;
        let seeds = fp16_root_derived_seeds(h, w, k, OP_HASH_ID, OP_HASH_ID, KEY_A, KEY_B, &a, &b).unwrap();
        let (e_a, f_a, e_b, f_b) = fp16_tile_noise(seeds, h, w, k);
        let mut expected_noise = a100_matmul(&e_a, &f_a, None, h, k, r);
        expected_noise.extend(a100_matmul(&e_b, &f_b, None, w, k, r));
        assert_eq!(expected_noise.len(), (h + w) * k);
        // Distinct F is the correctness fix: F_A != F_B per the plaintext `sample_noise`.
        assert_ne!(f_a, f_b, "F_A and F_B must be distinct (Side::A vs Side::B)");

        let m2 = &MATMUL_A100_COL_MAP;
        let nq = &NOISY_QUANT_COL_MAP;
        let (honest, _) = system.generate_batch_traces(&a, &b, KEY_A, KEY_B, jk).expect("honest traces");

        // The two noise matmuls' proven cell results (A cells -> [0,h*k), B cells -> [h*k,(h+w)*k)).
        let mut matmul_noise = vec![f32::NAN; (h + w) * k];
        for (table, base) in [(NOISE_MATMUL_A_A100_TABLE, 0usize), (NOISE_MATMUL_B_A100_TABLE, h * k)] {
            let noise_tbl = &honest[table];
            for row in 0..noise_tbl[m2.is_cell_final].values.len() {
                if noise_tbl[m2.is_cell_final].values[row] == F::ONE && noise_tbl[m2.is_padding].values[row] == F::ZERO {
                    let cell = noise_tbl[m2.cell_id].values[row].to_canonical_u64() as usize;
                    let lo = noise_tbl[m2.cell_result_f32_lo].values[row].to_canonical_u64();
                    let hi = noise_tbl[m2.cell_result_f32_hi].values[row].to_canonical_u64();
                    matmul_noise[base + cell] = f32::from_bits(((hi << 16) | lo) as u32);
                }
            }
        }
        for (e, &want) in expected_noise.iter().enumerate() {
            assert_eq!(
                matmul_noise[e].to_bits(),
                want.to_bits(),
                "noise matmul cell {e} must be bit-exact with a100_matmul(e, f)"
            );
        }

        // The noise noisy-quant consumed (NOISE_LO/HI per live element) must equal the SAME product.
        for row in 0..honest[NOISY_QUANT_A100_TABLE][nq.is_pad].values.len() {
            if honest[NOISY_QUANT_A100_TABLE][nq.is_pad].values[row] == F::ZERO {
                let e = honest[NOISY_QUANT_A100_TABLE][nq.element_index].values[row].to_canonical_u64() as usize;
                let lo = honest[NOISY_QUANT_A100_TABLE][nq.noise_lo].values[row].to_canonical_u64();
                let hi = honest[NOISY_QUANT_A100_TABLE][nq.noise_hi].values[row].to_canonical_u64();
                let consumed = f32::from_bits(((hi << 16) | lo) as u32);
                assert_eq!(
                    consumed.to_bits(),
                    expected_noise[e].to_bits(),
                    "noisy-quant consumed noise at element {e} must equal the E@F product"
                );
            }
        }

        let prove_tampered = |table: usize, col: usize, row: usize| {
            let (mut traces, blake3_pis) = system.generate_batch_traces(&a, &b, KEY_A, KEY_B, jk).expect("honest traces");
            traces[table][col].values[row] += F::ONE;
            let mut timing = TimingTree::default();
            match system.prove_batch_traces::<C>(traces, &blake3_pis, &preprocessed, &mut timing) {
                Ok(bad) => system.verify::<C>(&bad, &blake3_pis, &lut_cap).is_err(),
                Err(_) => true,
            }
        };

        // ---- (c) tamper the noise matmul's N output at a finished live cell -> rejected. The
        // noise-word CTL binds this `cell_result` to noisy-quant's consumed noise; `cell_result` also
        // feeds the matmul's own MA9 encode, so a lone tamper is caught end to end (the CTL-attributed
        // isolation is in `ctl::tests::forged_noise_word_breaks_the_noise_channel`). ----
        let noise_tbl_a = &honest[NOISE_MATMUL_A_A100_TABLE];
        let noise_final = (0..noise_tbl_a[m2.is_cell_final].values.len())
            .find(|&row| {
                noise_tbl_a[m2.is_cell_final].values[row] == F::ONE
                    && noise_tbl_a[m2.is_padding].values[row] == F::ZERO
            })
            .expect("a finished live noise-matmul cell");
        assert!(
            prove_tampered(NOISE_MATMUL_A_A100_TABLE, m2.cell_result_f32_lo, noise_final),
            "a tampered noise-matmul N output must be rejected (noise-word channel + MA9)"
        );

        // ---- (d) tamper noisy-quant's N input (NOISE_LO) at a live element -> rejected (the
        // noise-word CTL unbinds it from the proven product; `NOISE_LO` also feeds the G1 rounding). ----
        let nq_live = (0..honest[NOISY_QUANT_A100_TABLE][nq.is_pad].values.len())
            .find(|&row| honest[NOISY_QUANT_A100_TABLE][nq.is_pad].values[row] == F::ZERO)
            .expect("a live noisy-quant element");
        assert!(
            prove_tampered(NOISY_QUANT_A100_TABLE, nq.noise_lo, nq_live),
            "a tampered noisy-quant N input must be rejected (noise-word 6e-1 channel + G1)"
        );
    }

    /// Increment 6e-3d — the noise is now DERIVED FROM THE COMMITTED OPERAND ROOTS, closing the last
    /// anti-grind gap. This test shows:
    /// * (a) the root-derived seeds match `api::fp16::noise::noise_seeds` over `commit_operand(..).root()`
    ///   + the opening keys + the `p` encodings, and the honest tile's noise (E/F, N, noised, tile,
    ///   jackpot) is bit-exact with the plaintext `sample_noise` over those seeds (the honest proof
    ///   verifies and the other capstone tests pin N / noised-out / jackpot over the same seeds);
    /// * (b) operand-dependence: changing an operand (hence its root) changes the derived seeds and the
    ///   whole noise draw — the noise is no longer a fixed public value;
    /// * (c) `verify` pins the noise-BLAKE3 `KEY_A`/`KEY_B` public inputs to `noise_seeds` recomputed
    ///   from the proof's `HASH_A`/`HASH_B` + keys + `p`, so a seed not derived from the committed roots
    ///   is rejected.
    #[test]
    fn noise_seeds_are_root_derived_operand_dependent_and_pinned() {
        use crate::v5::api::noise::noise_seeds as api_noise_seeds;
        use crate::v5::circuit::blake3_fp16_stark::columns::PI_KEY_A as NB_PI_KEY_A;
        use super::super::ctl::NOISE_BLAKE3_A100_TABLE;

        let (k, a0, b0) = accepting_cell();
        let (h, w) = (4usize, 4usize);
        let (a, b) = (a0.repeat(h), b0.repeat(w));
        let mut system = Fp16System::<F, D>::new(h, w, k, OP_HASH_ID, OP_HASH_ID);
        let jk = jackpot_key(&SEED_A);
        let mut timing = TimingTree::default();
        let preprocessed = system.preprocessed_data::<C>(&mut timing);
        let lut_cap = preprocessed.cap();

        // ---- (a) the driver's root-derived seeds equal `api::fp16::noise::noise_seeds` over the real
        // committed roots + keys + p — i.e. the EXACT plaintext seed chain. ----
        let seeds = system.root_derived_seeds(&a, &b, KEY_A, KEY_B).expect("root-derived seeds");
        let root_a = commit_operand(&a, h, k, OP_HASH_ID, KEY_A).unwrap().root();
        let root_b = commit_operand(&b, w, k, OP_HASH_ID, KEY_B).unwrap().root();
        let p = fp16_noise_seed_params(h, w, k, OP_HASH_ID, OP_HASH_ID);
        let want = api_noise_seeds(&Sides { a: KEY_A, b: KEY_B }, &Sides { a: root_a, b: root_b }, &p);
        assert_eq!(seeds, want, "driver seeds must equal api::fp16::noise::noise_seeds over the committed roots");

        // The honest proof verifies with the root-derived noise wired through every noise table.
        let proof = system
            .prove::<C>(&a, &b, KEY_A, KEY_B, jk, &preprocessed, &mut timing)
            .expect("honest FP16 tile with root-derived noise must prove");
        let blake3_pis = proof.public_inputs[BLAKE3_A100_TABLE].clone();
        system.verify::<C>(&proof, &blake3_pis, &lut_cap).expect("honest root-derived-noise proof must verify");

        // ---- (b) operand-dependence: a different operand -> different root -> different seeds ->
        // different noise draw. Changing A moves seedA (which folds root_A); changing B moves seedB. ----
        let mut a2 = a.clone();
        a2[0] ^= 0x4000; // perturb one A code (a high mantissa bit; stays a valid FP16 pattern)
        let mut b2 = b.clone();
        b2[0] ^= 0x4000;
        let seeds_a2 = system.root_derived_seeds(&a2, &b, KEY_A, KEY_B).expect("seeds for perturbed A");
        let seeds_b2 = system.root_derived_seeds(&a, &b2, KEY_A, KEY_B).expect("seeds for perturbed B");
        assert_ne!(seeds.a, seeds_a2.a, "perturbing A must move seedA (operand-dependence)");
        assert_ne!(seeds.b, seeds_b2.b, "perturbing B must move seedB (operand-dependence)");
        // The full noise draw changes with the seeds.
        assert_ne!(
            fp16_tile_noise(seeds, h, w, k),
            fp16_tile_noise(seeds_a2, h, w, k),
            "a changed operand must change the noise draw"
        );

        // ---- (c) `verify` pins the noise-BLAKE3 KEY_A to `noise_seeds` of the committed roots: a
        // noise seed not derived from the roots (here a tampered KEY_A limb) is rejected. ----
        let mut bad = proof.clone();
        bad.public_inputs[NOISE_BLAKE3_A100_TABLE][NB_PI_KEY_A] += F::ONE;
        assert!(
            system.verify::<C>(&bad, &blake3_pis, &lut_cap).is_err(),
            "a noise seed (noise-BLAKE3 KEY_A) not derived from the committed roots must be rejected by verify's pinning"
        );
    }

    /// Increment 6f — the consensus-meaningful OUTPUT gateway. An honest tile proves; the
    /// Fiat-Shamir statement digest is DERIVED from the proof's own `HASH_JACKPOT` (not a
    /// caller-opaque salt); [`Fp16System::verify_with_difficulty`] ACCEPTS at an easy target and
    /// REJECTS at an impossible (all-zero) one — and its accept/reject decision equals the plaintext
    /// [`check_jackpot_difficulty`] on the SAME tile for both targets. A tampered `HASH_JACKPOT` /
    /// wrong expected public input is rejected by the gateway's public-input pinning.
    #[test]
    fn verify_with_difficulty_matches_plaintext_and_derives_statement_digest() {
        let (k, a0, b0) = accepting_cell();
        let (h, w) = (4usize, 4usize);
        let (a, b) = (a0.repeat(h), b0.repeat(w));
        let mut system = Fp16System::<F, D>::new(h, w, k, OP_HASH_ID, OP_HASH_ID);
        let jk = jackpot_key(&SEED_A);
        let mut timing = TimingTree::default();
        let preprocessed = system.preprocessed_data::<C>(&mut timing);
        let lut_cap = preprocessed.cap();

        let proof = system
            .prove::<C>(&a, &b, KEY_A, KEY_B, jk, &preprocessed, &mut timing)
            .expect("honest FP16 tile must prove");
        let blake3_pis = proof.public_inputs[BLAKE3_A100_TABLE].clone();

        // ---- The statement digest is DERIVED from the proof's own jackpot limbs. ----
        let expected_digest = derive_statement_digest(hash_jackpot::<F>(&blake3_pis));
        assert_eq!(expected_digest, Fp16System::<F, D>::statement_digest(&blake3_pis));
        system
            .ensure_statement_digest_binding(expected_digest)
            .expect("prove must bind the jackpot-derived statement digest");
        assert!(
            system.ensure_statement_digest_binding([0xAB; 32]).is_err(),
            "a statement digest other than the jackpot-derived one must be rejected"
        );

        // ---- Difficulty: the ZK gateway decision equals the plaintext one on the SAME (noised) tile. ----
        let (noised_a, noised_b) = system.noised_operands(&a, &b, KEY_A, KEY_B);
        let mut tile = vec![0f32; h * w];
        for r in 0..h {
            for c in 0..w {
                tile[r * w + c] = a100_dot(&noised_a[r * k..r * k + k], &noised_b[c * k..c * k + k], 0.0, None);
            }
        }
        let (row_axis, col_axis) = default_lane_layout(h, w);
        let msg = xor_fold_extract(&tile, &lane_assignment(&row_axis, &col_axis));
        let plain_jackpot = compute_jackpot_ticket(&SEED_A, &msg).jackpot;

        // Easy target saturates the bound (accepts); the all-zero target is impossible (rejects) —
        // the same two targets the plaintext `verify_tile_proof` test uses.
        const EASY_NBITS: u32 = 0x207fffff;
        const IMPOSSIBLE_NBITS: u32 = 0;
        for nbits in [EASY_NBITS, IMPOSSIBLE_NBITS] {
            let zk = system.verify_with_difficulty::<C>(&proof, &blake3_pis, nbits, &lut_cap).is_ok();
            let plain = check_jackpot_difficulty(&plain_jackpot, nbits, h as u32, w as u32, k as u32).is_ok();
            assert_eq!(zk, plain, "ZK difficulty decision must equal the plaintext one at nbits={nbits:#x}");
        }
        system
            .verify_with_difficulty::<C>(&proof, &blake3_pis, EASY_NBITS, &lut_cap)
            .expect("easy target must accept");
        assert!(
            system.verify_with_difficulty::<C>(&proof, &blake3_pis, IMPOSSIBLE_NBITS, &lut_cap).is_err(),
            "the impossible (all-zero) target must reject"
        );

        // ---- A tampered HASH_JACKPOT / wrong expected public input is rejected by the pinning. ----
        let mut bad_jackpot = blake3_pis.clone();
        bad_jackpot[PI_HASH_JACKPOT] += F::ONE;
        assert!(
            system.verify_with_difficulty::<C>(&proof, &bad_jackpot, EASY_NBITS, &lut_cap).is_err(),
            "a tampered expected HASH_JACKPOT must be rejected"
        );
        let mut bad_key = blake3_pis.clone();
        bad_key[0] += F::ONE; // a KEY_A limb
        assert!(
            system.verify_with_difficulty::<C>(&proof, &bad_key, EASY_NBITS, &lut_cap).is_err(),
            "a wrong expected key public input must be rejected"
        );
    }

    /// The header-pinned verify GATEWAY ([`Fp16System::verify_with_headers`]): nothing feeding the
    /// noise seeds or the operand-tree commitment keys is test-supplied — it is all derived from the
    /// block headers + the public job params, exactly as the plaintext certificate derives it. This
    /// shows:
    /// * (bit-exact) the gateway's `keyA`/`keyB` equal `api::fp16::noise::commitment_keys` and its noise
    ///   seeds equal `api::fp16::noise::noise_seeds` over the committed operand roots + header keys + `p`;
    /// * (accept) an honest tile proved under those header-derived keys/`p` verifies through the gateway;
    /// * (reject) a DIFFERENT proposed header (moves `keyA` -> `seedA`), a different ancestor header
    ///   (moves `keyB`/`p_b` -> `seedB`), and a different `p` alone (a changed pattern, `keys` fixed, moves
    ///   `seedA`) are each rejected by the header-derived public-input pinning;
    /// * (difficulty) the gateway's accept/reject at an easy vs an impossible `nbits` matches the plaintext
    ///   `check_jackpot_difficulty` on the SAME noised tile.
    #[test]
    fn verify_with_headers_binds_to_the_header() {
        use crate::v5::api::noise::{commitment_keys as api_commitment_keys, key_a, key_b, noise_seeds as api_noise_seeds};
        use crate::v5::api::params::Fp16Device;
        use crate::v5::api::plain_proof::{Fp16JobParams, Fp16OperandParams};
        use crate::v4::api::layout::{AxisPattern, DimType};
        use crate::v4::api::primitives::IncompleteBlockHeader;

        const EASY_NBITS: u32 = 0x207fffff;
        const IMPOSSIBLE_NBITS: u32 = 0;

        // The headers that key the commitment + seed chain: the proposed header keys the A side, the
        // ancestor header the B side (window-authenticated upstream, as the plaintext cert assumes).
        let proposed = IncompleteBlockHeader::new_for_test(EASY_NBITS);
        let ancestor = IncompleteBlockHeader::zero();
        let keys = api_commitment_keys(&proposed, &ancestor);
        // Bit-exact with the plaintext cert's per-side opening keys.
        assert_eq!(keys.a, key_a(&proposed), "gateway keyA must equal api key_a(proposed)");
        assert_eq!(keys.b, key_b(&ancestor), "gateway keyB must equal api key_b(ancestor)");

        // A cell whose NOISED 4x4 tile clears the policy gate under THESE header-derived keys (+ the
        // job's `p`, which for a zero-ancestor dense-tile job equals the system's default encoding, so
        // the free `fp16_noised_operands` draws the same noise the proven system will).
        let dense = |n: usize| AxisPattern::new(&[(n as u32, DimType::Blake)]).expect("dense tile pattern");
        let (h, w) = (4usize, 4usize);
        let (k, a0, b0) = {
            let mut best: Option<(usize, Vec<u16>, Vec<u16>)> = None;
            for line in VECTORS.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let t: Vec<&str> = line.split_whitespace().collect();
                let kk: usize = t[0].parse().unwrap();
                if kk % ACC_GROUP != 0 || kk < 16 {
                    continue;
                }
                if best.as_ref().is_some_and(|(bk, _, _)| kk >= *bk) {
                    continue;
                }
                let a0: Vec<u16> = t[1..1 + kk].iter().map(|x| x.parse().unwrap()).collect();
                let b0: Vec<u16> = t[1 + kk..1 + 2 * kk].iter().map(|x| x.parse().unwrap()).collect();
                let (a, b) = (a0.repeat(h), b0.repeat(w));
                let Ok((na, nb)) = fp16_noised_operands(h, w, kk, &a, &b, keys.a, keys.b, OP_HASH_ID, OP_HASH_ID) else {
                    continue;
                };
                if replay_and_evaluate(&na, &nb, h, w, kk).1.accept {
                    best = Some((kk, a0, b0));
                }
            }
            best.expect("a reference cell whose header-noised 4x4 tile accepts")
        };
        let (a, b) = (a0.repeat(h), b0.repeat(w));

        // The public job params (the plaintext cert's `Fp16JobParams`), from which the gateway derives
        // `p_a`/`p_b`. The honest prover folds the SAME `p` into its seed chain via `set_noise_seed_params`.
        let job = Fp16JobParams {
            ancestor_header: ancestor,
            device: Fp16Device::A100,
            k: k as u32,
            r: FP16_QUANT_R as u32,
            operands: Sides {
                a: Fp16OperandParams { num_rows: h as u32, hash_id: OP_HASH_ID, pattern: dense(h) },
                b: Fp16OperandParams { num_rows: w as u32, hash_id: OP_HASH_ID, pattern: dense(w) },
            },
        };

        let mut system = Fp16System::<F, D>::new(h, w, k, OP_HASH_ID, OP_HASH_ID);
        system.set_noise_seed_params(Sides { a: job.encode_p_a(), b: job.encode_p_b() });
        let mut timing = TimingTree::default();
        let preprocessed = system.preprocessed_data::<C>(&mut timing);
        let lut_cap = preprocessed.cap();

        // The gateway's noise seeds are bit-exact with api::fp16::noise::noise_seeds over the committed
        // operand roots (HASH_A/HASH_B) + header keys + the job's `p`.
        let root_a = commit_operand(&a, h, k, OP_HASH_ID, keys.a).unwrap().root();
        let root_b = commit_operand(&b, w, k, OP_HASH_ID, keys.b).unwrap().root();
        let p = Sides { a: job.encode_p_a(), b: job.encode_p_b() };
        let want_seeds = api_noise_seeds(&keys, &Sides { a: root_a, b: root_b }, &p);
        assert_eq!(
            system.root_derived_seeds(&a, &b, keys.a, keys.b).unwrap(),
            want_seeds,
            "the header-derived seeds must equal api::fp16::noise::noise_seeds"
        );

        // The honest jackpot (lottery-compression) key is jackpot_key(seed_a) over the HEADER-DERIVED
        // seed_a (want_seeds.a) -- exactly what the gateway now pins PI_JACKPOT_KEY to. (A test that
        // keyed it on an unrelated seed would be rejected by that pin, as it should be.)
        let jk = jackpot_key(&want_seeds.a);

        // ---- (accept) the honest tile, proved under the header-derived keys/p, verifies. ----
        let proof = system
            .prove::<C>(&a, &b, keys.a, keys.b, jk, &preprocessed, &mut timing)
            .expect("honest header-keyed FP16 tile must prove");
        system
            .verify_with_headers::<C>(&proof, &proposed, &job, EASY_NBITS, &lut_cap)
            .expect("the honest proof must verify through the header gateway");

        // ---- (bind + reject) the jackpot (lottery-compression) key is pinned to the header/root-
        // derived jackpot_key(seed_a); a proof carrying any other jackpot key is rejected (closes the
        // pow-key grind: a prover cannot pick the key that hashes the lottery words to HASH_JACKPOT). ----
        {
            use crate::v4::circuit::blake3_stark::columns::PI_JACKPOT_KEY;
            let hw = |h: Hash256| -> [F; 8] {
                core::array::from_fn(|i| F::from_canonical_u32(u32::from_le_bytes(h[4 * i..4 * i + 4].try_into().unwrap())))
            };
            assert_eq!(
                proof.public_inputs[BLAKE3_A100_TABLE][PI_JACKPOT_KEY..PI_JACKPOT_KEY + 8],
                hw(jackpot_key(&want_seeds.a)),
                "the honest proof's jackpot key must be jackpot_key(header-derived seed_a)"
            );
            let mut forged = proof.clone();
            forged.public_inputs[BLAKE3_A100_TABLE][PI_JACKPOT_KEY] += F::ONE;
            assert!(
                system.verify_with_headers::<C>(&forged, &proposed, &job, EASY_NBITS, &lut_cap).is_err(),
                "a proof whose jackpot key is not jackpot_key(seed_a) must be rejected by the gateway pin"
            );
        }

        // ---- (reject) a different proposed header moves keyA -> seedA; the proof's header-derived
        // KEY_A / noise-BLAKE3 seed pins no longer match. ----
        let wrong_proposed = IncompleteBlockHeader { timestamp: proposed.timestamp ^ 0x5A5A, ..proposed };
        assert!(
            system.verify_with_headers::<C>(&proof, &wrong_proposed, &job, EASY_NBITS, &lut_cap).is_err(),
            "a different proposed header must be rejected (keyA/seedA no longer header-derived)"
        );

        // ---- (reject) a different ancestor header moves keyB/p_b -> seedB. ----
        let mut wrong_ancestor_job = job.clone();
        wrong_ancestor_job.ancestor_header =
            IncompleteBlockHeader { prev_block: [0x77; 32], ..ancestor };
        assert!(
            system.verify_with_headers::<C>(&proof, &proposed, &wrong_ancestor_job, EASY_NBITS, &lut_cap).is_err(),
            "a different ancestor header must be rejected (keyB/p_b/seedB no longer header-derived)"
        );

        // ---- (reject) a different `p` alone (a changed A pattern, keys unchanged) moves seedA. ----
        let mut wrong_p_job = job.clone();
        wrong_p_job.operands.a.pattern = AxisPattern::new(&[(h as u32, DimType::Fold)]).unwrap();
        assert!(
            system.verify_with_headers::<C>(&proof, &proposed, &wrong_p_job, EASY_NBITS, &lut_cap).is_err(),
            "a different p (changed pattern) must be rejected (seedA no longer header-derived)"
        );

        // ---- (difficulty) the gateway decision equals the plaintext one on the SAME noised tile. ----
        let (noised_a, noised_b) = system.noised_operands(&a, &b, keys.a, keys.b);
        let mut tile = vec![0f32; h * w];
        for r in 0..h {
            for c in 0..w {
                tile[r * w + c] = a100_dot(&noised_a[r * k..r * k + k], &noised_b[c * k..c * k + k], 0.0, None);
            }
        }
        let (row_axis, col_axis) = default_lane_layout(h, w);
        let msg = xor_fold_extract(&tile, &lane_assignment(&row_axis, &col_axis));
        let plain_jackpot = compute_jackpot_ticket(&want_seeds.a, &msg).jackpot;
        for nbits in [EASY_NBITS, IMPOSSIBLE_NBITS] {
            let zk = system.verify_with_headers::<C>(&proof, &proposed, &job, nbits, &lut_cap).is_ok();
            let plain = check_jackpot_difficulty(&plain_jackpot, nbits, h as u32, w as u32, k as u32).is_ok();
            assert_eq!(zk, plain, "gateway difficulty decision must equal the plaintext one at nbits={nbits:#x}");
        }
    }
}
