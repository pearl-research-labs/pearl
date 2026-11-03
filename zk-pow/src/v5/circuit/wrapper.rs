//! The two-stage recursive wrapper for the FP16 / A100 batch proof — the FP16 analogue of
//! [`crate::v4::circuit::wrapper`].
//!
//! **Why wrap.** [`Fp16System`]'s native batched multi-STARK proof grows with the table set and is
//! not zero-knowledge as a published object. The wrapper encodes the whole batch verification
//! inside a plonky2 circuit and publishes a constant-size recursive proof instead:
//!
//! - **Stage 1** (Poseidon recursion, no ZK): the in-circuit batch verifier
//!   ([`verify_batch_stark_proof_circuit`]) re-runs the full batch verification — every AIR's
//!   constraints, the four committed-LUT channels and the census-import CTL, the setup-time LUT cap
//!   (baked in as a constant) and the batched FRI argument. Its public inputs are, in order:
//!
//!   ```text
//!   every table's STARK public inputs (batch order — all empty for FP16)
//!   | known-column digest (4)
//!   | zeta (2)
//!   | known-column evals at zeta      (2 per column, batch order)
//!   | known-column evals at g_t*zeta  (2 per column, batch order)
//!   ```
//!
//!   The class (a) ("known") columns cannot be evaluated in-circuit (they are full-height per-job
//!   columns), so the circuit *exposes* the challenge point `zeta` (connected to the in-circuit
//!   Fiat-Shamir challenge) and the claimed evaluations (connected to the proof's trace openings),
//!   and the native gateway [`verify_wrapped_proof`] recomputes those evaluations at `zeta` from its
//!   own statement-derived known values and pins every slot. A prover therefore cannot lie about
//!   any class (a) column without breaking the in-circuit FRI binding or failing the gateway's
//!   public-input equality.
//!
//! - **Stage 2** (ZK wrap): a plonky2 circuit with `zero_knowledge: true` that verifies the stage-1
//!   proof and forwards its public inputs unchanged. Stage 1's verifier data is baked into stage 2
//!   as circuit constants, so a stage-2 proof attests to exactly this statement's stage-1 circuit.
//!   The published artifact is the stage-2 proof only.
//!
//! Like the FP8 wrapper, stage 1 proves under [`PoseidonGoldilocksConfig`] and stage 2 under
//! [`Blake3GoldilocksConfig`]. FP16 has no device variants and its main tables carry no public
//! inputs, so the wrapper omits the FP8 wrapper's compiled-device and geometry public inputs.
//!
//! **Scope — universality (the FP8 D1 redesign).** This wrapper is compiled for one degree profile
//! (one tile shape); a different tile size needs a separately compiled circuit (and may emit a
//! different-length proof). FP8 compresses *every* envelope-legal job to one compiled, constant-size
//! circuit via starky's *universal* batch verifier. The FP16 driver already ships the consensus
//! ladder, envelope and statement-digest machinery that path needs
//! ([`crate::v5::circuit::driver::fp16_universal_envelope`]), but
//! `starky::batch_universal::verify_universal_batch_stark_proof_circuit` over-determines a witness
//! wire for FP16's table shape — two *equal-height* variable main tables (matmul and policy share
//! one row grid) linked by a direct CTL (census-import), a configuration FP8 never produces (its
//! variable tables always differ in height) and the universal verifier's own tests do not cover.
//! Closing that is the documented residual (`docs/fp16_scheme/stark_feasibility.md §8`).

use anyhow::{Result, ensure};
use hashbrown::HashMap;
use plonky2::field::extension::FieldExtension;
use plonky2::field::extension::quadratic::QuadraticExtension;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOutTarget, NUM_HASH_OUT_ELTS};
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitData, VerifierCircuitData};
use plonky2::plonk::config::{Blake3GoldilocksConfig, GenericConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::timed;
use plonky2::util::timing::TimingTree;
use plonky2_maybe_rayon::{MaybeIntoParIter, ParallelIterator};
use starky::batch_proof::{BatchStarkProofWithPublicInputs, BatchStarkProofWithPublicInputsTarget};
use starky::batch_recursive_verifier::{
    BatchKnownColumnsTarget, add_virtual_batch_stark_proof_with_pis, set_batch_stark_proof_with_pis_target,
    verify_batch_stark_proof_circuit,
};
use starky::verifier::eval_columns_at_zeta_and_next;

use super::ctl::NUM_FP16_TABLES;
use super::blake3_fp16_stark::columns::NUM_FP16_RAW_BLAKE3_PUBLIC_INPUTS;
use crate::v4::circuit::blake3_stark::columns::NUM_BLAKE3_PUBLIC_INPUTS;
use super::driver::{FP16_GROUPED_TABLES, FP16_REACHABLE_DEGREE_BITS, Fp16System, hash_jackpot, statement_digest_to_hash_out};
use crate::v5::api::plain_proof::Fp16JobParams;
use crate::v4::api::primitives::{Hash256, IncompleteBlockHeader};
use crate::v4::api::proof_utils::check_jackpot_difficulty;
use crate::v4::circuit::circuit_utils::build_recursion_config;

/// The wrapper's field, extension degree and per-stage hasher configurations.
pub type F = GoldilocksField;
pub const D: usize = 2;
/// Stage-1 configuration: the batch proof's own config (Poseidon caps for the in-circuit
/// Fiat-Shamir replay) and the cheap recursion hasher.
pub type InnerC = PoseidonGoldilocksConfig;
/// Stage-2 (published) configuration: Blake3 outer hashing.
pub type OuterC = Blake3GoldilocksConfig;

/// Sanctioned FRI parameters of the two wrapper stages (the same consensus combinations as the
/// FP8 wrapper). Rate `2^-3` is the floor for both stages: the recursion gates carry degree-7
/// constraints, so the quotient needs an LDE blowup of at least `2^3`.
pub const STAGE_1_RATE_BITS: usize = 3;
pub const STAGE_1_POW_BITS: usize = 18;
pub const STAGE_2_RATE_BITS: usize = 7;
pub const STAGE_2_POW_BITS: usize = 22;

/// Flat offset of the exposed challenge point `zeta`. The batch's STARK public inputs come first
/// (only the Blake3 table carries any — its 64 hash/key limbs), then the known-column digest, then
/// `zeta`.
pub fn zeta_offset(_system: &Fp16System<F, D>) -> usize {
    // Both PI-bearing tables are published before the known-column digest: the operand/jackpot Blake3
    // table (64 limbs) and, since 6e-3c, the noise-BLAKE3 derivation table (64 limbs, the pinned
    // seeds + inert lottery digest). Every other table registers an empty public-input vector.
    NUM_BLAKE3_PUBLIC_INPUTS + NUM_FP16_RAW_BLAKE3_PUBLIC_INPUTS + NUM_HASH_OUT_ELTS
}

/// Total public-input count of a wrapper proof for this batch shape.
fn num_wrapper_public_inputs(system: &Fp16System<F, D>) -> usize {
    let num_known: usize = system.known().columns_per_table.iter().map(Vec::len).sum();
    zeta_offset(system) + D + 2 * D * num_known
}

/// The two compiled wrapper circuits, plus the witness targets the prover fills. These are compiled
/// for one degree profile (tile shape); see the module docs on universality.
pub struct Fp16WrapperCircuits {
    /// Stage 1: the in-circuit batch verifier (Poseidon recursion, no ZK).
    stage1: CircuitData<F, InnerC, D>,
    proof_target: BatchStarkProofWithPublicInputsTarget<D>,
    known_digest_target: HashOutTarget,
    /// Stage 2: the ZK wrap publishing stage 1's public inputs.
    stage2: CircuitData<F, OuterC, D>,
    stage1_proof_target: ProofWithPublicInputsTarget<D>,
}

/// A virtual extension target registered (component-wise) as the next public inputs.
fn add_ext_public_input(builder: &mut CircuitBuilder<F, D>) -> ExtensionTarget<D> {
    let et = builder.add_virtual_extension_target();
    builder.register_public_inputs(&et.0);
    et
}

impl Fp16WrapperCircuits {
    /// Compiles both stages for `system`'s degree profile, baking `lut_cap` (the consensus LUT
    /// commitment) into stage 1 as constants.
    pub fn build(
        system: &Fp16System<F, D>,
        lut_cap: &MerkleCap<F, <InnerC as GenericConfig<D>>::Hasher>,
        timing: &mut TimingTree,
    ) -> Result<Self> {
        let known = system.known();
        ensure!(
            system
                .degree_bits()
                .iter()
                .all(|bits| FP16_REACHABLE_DEGREE_BITS.contains(bits)),
            "wrapper: the degree profile {:?} must lie on the consensus ladder",
            system.degree_bits()
        );

        // ---- Stage 1: the in-circuit batch verifier. ----
        let config_1 = build_recursion_config(STAGE_1_RATE_BITS, STAGE_1_POW_BITS, 1, false);
        let mut builder = CircuitBuilder::<F, D>::new(config_1);
        let starks = system.batch_starks();
        let preprocessed = system.preprocessed_verifier_data::<InnerC>(lut_cap);
        let prep_columns = preprocessed.columns_per_table.clone();
        let preprocessed_target = preprocessed.constant_target(&mut builder);

        let proof_target = add_virtual_batch_stark_proof_with_pis::<F, D, NUM_FP16_TABLES>(
            &mut builder,
            &starks,
            system.config(),
            system.degree_bits(),
            &prep_columns,
            system.ctls(),
            &FP16_GROUPED_TABLES,
        )?;

        // The known-column surface: a digest slot and one claimed evaluation per column per point.
        let known_digest_target = builder.add_virtual_hash();
        let evals_at_zeta: Vec<Vec<ExtensionTarget<D>>> = known
            .columns_per_table
            .iter()
            .map(|columns| (0..columns.len()).map(|_| builder.add_virtual_extension_target()).collect())
            .collect();
        let evals_at_g_zeta: Vec<Vec<ExtensionTarget<D>>> = known
            .columns_per_table
            .iter()
            .map(|columns| (0..columns.len()).map(|_| builder.add_virtual_extension_target()).collect())
            .collect();
        let known_target = BatchKnownColumnsTarget::<D> {
            digest: Some(known_digest_target),
            columns_per_table: known.columns_per_table.clone(),
            evals_at_zeta,
            evals_at_g_zeta,
        };

        let zeta = verify_batch_stark_proof_circuit::<F, InnerC, D, NUM_FP16_TABLES>(
            &mut builder,
            &starks,
            system.config(),
            &proof_target,
            system.ctls(),
            &FP16_GROUPED_TABLES,
            Some(&preprocessed_target),
            Some(&known_target),
            &HashMap::new(),
        )?;

        // Public-input layout (see module docs): table public inputs (all empty for FP16), the
        // known-column digest, zeta, then the claimed evaluations.
        for pis in &proof_target.public_inputs {
            builder.register_public_inputs(pis);
        }
        builder.register_public_inputs(&known_digest_target.elements);
        let zeta_pi = add_ext_public_input(&mut builder);
        builder.connect_extension(zeta_pi, zeta);
        for evals in known_target.evals_at_zeta.iter().chain(&known_target.evals_at_g_zeta) {
            for eval in evals {
                builder.register_public_inputs(&eval.0);
            }
        }

        let stage1_gates = builder.num_gates();
        let stage1 = timed!(timing, "build the stage-1 FP16 wrapper circuit", builder.build::<InnerC>());
        log::info!(
            "stage-1 FP16 wrapper circuit: {stage1_gates} gates -> 2^{} rows",
            stage1.common.degree_bits(),
        );

        // ---- Stage 2: the ZK wrap. ----
        let config_2 = build_recursion_config(STAGE_2_RATE_BITS, STAGE_2_POW_BITS, 2, true);
        debug_assert!(config_2.zero_knowledge);
        let mut builder = CircuitBuilder::<F, D>::new(config_2);
        let stage1_proof_target = builder.add_virtual_proof_with_pis(&stage1.common);
        builder.register_public_inputs(&stage1_proof_target.public_inputs);
        let stage1_verifier_target = builder.constant_verifier_data(&stage1.verifier_only);
        builder.verify_proof::<InnerC>(&stage1_proof_target, &stage1_verifier_target, &stage1.common);
        let stage2 = timed!(timing, "build the stage-2 (ZK) FP16 wrapper circuit", builder.build::<OuterC>());
        log::info!("stage-2 FP16 wrapper circuit: 2^{} gates", stage2.common.degree_bits());

        Ok(Self {
            stage1,
            proof_target,
            known_digest_target,
            stage2,
            stage1_proof_target,
        })
    }

    /// Wraps one batch proof: proves stage 1 (the in-circuit batch verification), then stage 2 (the
    /// ZK wrap). Returns the publishable stage-2 proof. `batch_proof` must be a proof of `system`'s
    /// statement; `statement_digest` is the Fiat-Shamir salt the inner batch proof absorbed.
    pub fn prove(
        &self,
        system: &Fp16System<F, D>,
        batch_proof: &BatchStarkProofWithPublicInputs<F, InnerC, D>,
        statement_digest: Hash256,
        timing: &mut TimingTree,
    ) -> Result<ProofWithPublicInputs<F, OuterC, D>> {
        system.ensure_statement_digest_binding(statement_digest)?;
        let mut pw = PartialWitness::new();
        set_batch_stark_proof_with_pis_target(&mut pw, &self.proof_target, batch_proof)?;
        pw.set_hash_target(self.known_digest_target, statement_digest_to_hash_out(statement_digest))?;
        let stage1_proof = timed!(timing, "prove the stage-1 FP16 wrapper", self.stage1.prove(pw))?;

        let mut pw = PartialWitness::new();
        pw.set_proof_with_pis_target(&self.stage1_proof_target, &stage1_proof)?;
        timed!(timing, "prove the stage-2 (ZK) FP16 wrapper", self.stage2.prove(pw))
    }

    /// The verifier's view: stage 2's verifier data (the only circuit a verifier needs).
    pub fn verifier_data(&self) -> VerifierCircuitData<F, OuterC, D> {
        self.stage2.verifier_data()
    }
}

/// Verifies a wrapped (stage-2) proof against the statement: it pins *every* public-input slot —
/// the known-column digest to `statement_digest`, and the known-column evaluations to the verifier's
/// native recomputation at the proof's `zeta` from the statement's known values (`zeta` is the one
/// prover-supplied slot; the stage-1 circuit constrains it to the inner batch proof's Fiat-Shamir
/// challenge) — then verifies the plonky2 proof against `verifier_data` (which must be the ZK stage).
/// Together with the in-circuit batch verification this gives exactly the guarantees of
/// [`Fp16System::verify`].
pub fn verify_wrapped_proof(
    system: &Fp16System<F, D>,
    verifier_data: &VerifierCircuitData<F, OuterC, D>,
    proof: &ProofWithPublicInputs<F, OuterC, D>,
    statement_digest: Hash256,
) -> Result<()> {
    ensure!(
        verifier_data.common.config.zero_knowledge,
        "the wrapped verifier circuit must be the ZK stage"
    );
    system.ensure_statement_digest_binding(statement_digest)?;
    let total = num_wrapper_public_inputs(system);
    ensure!(
        proof.public_inputs.len() == total,
        "wrapped proof: wrong public input count (got {}, statement expects {})",
        proof.public_inputs.len(),
        total
    );
    let z = zeta_offset(system);
    let zeta = QuadraticExtension([proof.public_inputs[z], proof.public_inputs[z + 1]]);
    // The batch public inputs are the wrapper vector's prefix (Blake3's 64 limbs; every other
    // table empty), read back from the proof to rebuild the expected layout.
    let blake3_pis = &proof.public_inputs[..NUM_BLAKE3_PUBLIC_INPUTS];
    let expected = expected_wrapper_public_inputs(system, statement_digest, zeta, blake3_pis)?;
    ensure!(
        proof.public_inputs == expected,
        "wrapped proof: public inputs do not match the statement's expectations"
    );
    verifier_data.verify(proof.clone())
}

/// The header-bound consensus gateway for a wrapped (stage-2) FP16 proof — the wrapper-level analogue
/// of [`Fp16System::verify_with_headers`], and the sound entry the consensus layer calls.
///
/// Unlike [`verify_wrapped_proof`] (which trusts a caller-supplied `statement_digest` and the
/// proof's own key public inputs), this derives EVERYTHING header-bound, exactly as the plaintext
/// certificate [`crate::v5::api::verify::verify_fp16_plain_proof`] does:
/// 1. reads the proof's committed operand-Blake3 PIs (`HASH_A`/`HASH_B`/`HASH_JACKPOT`) from the
///    wrapper vector's prefix, and overwrites the three keys with their header/root-derived values
///    via [`Fp16System::header_bound_blake3_pis`] (`KEY_A=key_a(proposed)`, `KEY_B=key_b(ancestor)`,
///    `JACKPOT_KEY=jackpot_key(seed_a)`), so a proof whose opening keys, noise seeds, or jackpot
///    key are not header-derived fails the pin;
/// 2. DERIVES `statement_digest` from the (header-bound) `HASH_JACKPOT` — it is not caller-supplied;
/// 3. pins the whole stage-2 public-input vector to that header-bound expectation and runs the ZK
///    verification, then checks `check_jackpot_difficulty` on the proven jackpot against `nbits`.
///
/// The one remaining consensus boundary is inherent (as for the plaintext FFI): the caller supplies
/// the proposed header + `nbits` and must authenticate `job.ancestor_header` as a state-window member
/// (the hash-walk) before calling.
pub fn verify_wrapped_proof_with_headers(
    system: &mut Fp16System<F, D>,
    verifier_data: &VerifierCircuitData<F, OuterC, D>,
    proof: &ProofWithPublicInputs<F, OuterC, D>,
    proposed_header: &IncompleteBlockHeader,
    job: &Fp16JobParams,
    nbits: u32,
) -> Result<()> {
    ensure!(
        verifier_data.common.config.zero_knowledge,
        "the wrapped verifier circuit must be the ZK stage"
    );
    // Consensus / wrapper-legal envelope gate (defense in depth; the FFI also gates
    // before building the system). Rejects any out-of-envelope geometry cleanly.
    {
        let (h, w, k) = system.tile_geometry();
        crate::v5::api::params::Fp16Params { device: job.device, h, w, k, r: job.r as usize }.validate()?;
    }
    // Fold THIS job's `p` into the system exactly as the prover did
    // (`Fp16Prover::prove`): `Fp16System::new` installs only a placeholder `p`
    // (zero ancestor), so without this the recomputed seed / known-column public
    // inputs would not match a proof whose job carries a real ancestor header.
    system.set_noise_seed_params(crate::v4::api::primitives::Sides {
        a: job.encode_p_a(),
        b: job.encode_p_b(),
    });
    let system = &*system;
    let total = num_wrapper_public_inputs(system);
    ensure!(
        proof.public_inputs.len() == total,
        "wrapped proof: wrong public input count (got {}, statement expects {})",
        proof.public_inputs.len(),
        total
    );
    let z = zeta_offset(system);
    let zeta = QuadraticExtension([proof.public_inputs[z], proof.public_inputs[z + 1]]);

    // Header-bind the operand-Blake3 PIs: the proof's committed roots/jackpot with the three keys
    // overwritten by their header/root-derived values (the noise seeds follow from the roots+keys).
    let proof_blake3_pis = &proof.public_inputs[..NUM_BLAKE3_PUBLIC_INPUTS];
    let (expected_blake3, _seeds) = system.header_bound_blake3_pis(proof_blake3_pis, proposed_header, job);

    // The statement digest is DERIVED from the header-bound jackpot (not caller-supplied), closing
    // the output-to-header binding. (The HASH_JACKPOT slot is unchanged by the key overwrite.)
    let statement_digest = Fp16System::<F, D>::statement_digest(&expected_blake3);
    system.ensure_statement_digest_binding(statement_digest)?;
    let expected = expected_wrapper_public_inputs(system, statement_digest, zeta, &expected_blake3)?;
    ensure!(
        proof.public_inputs == expected,
        "wrapped proof: public inputs do not match the header-bound statement"
    );

    // Native difficulty on the proven, now header-key-bound jackpot — exactly as the plaintext
    // `verify_tile_proof` and `Fp16System::verify_with_headers` do.
    let jackpot = hash_jackpot::<F>(&expected_blake3);
    let (h, w, k) = system.tile_geometry();
    check_jackpot_difficulty(&jackpot, nbits, h as u32, w as u32, k as u32)?;

    verifier_data.verify(proof.clone())
}

/// The full stage-2 public-input vector a statement expects, given the proof's `zeta`.
pub fn expected_wrapper_public_inputs(
    system: &Fp16System<F, D>,
    statement_digest: Hash256,
    zeta: QuadraticExtension<GoldilocksField>,
    blake3_pis: &[F],
) -> Result<Vec<F>> {
    ensure!(
        !FieldExtension::<D>::is_in_basefield(&zeta),
        "wrapped proof: zeta must lie strictly in the extension field"
    );
    let total = num_wrapper_public_inputs(system);
    let known = system.known();
    let mut expected: Vec<F> = Vec::with_capacity(total);
    // Only the Blake3 table carries STARK public inputs; the digest follows them.
    for pis in system.batch_public_inputs(blake3_pis) {
        expected.extend(pis);
    }
    expected.extend(statement_digest_to_hash_out::<F>(statement_digest).elements);
    expected.extend(zeta.0);
    // The class (a) recompute: evaluate the statement's own known columns at the proof's zeta and
    // g_t * zeta — the same binding `batch_verify` performs natively. Each table is independent.
    type TableEvals = (
        Vec<QuadraticExtension<GoldilocksField>>,
        Vec<QuadraticExtension<GoldilocksField>>,
    );
    let per_table: Vec<TableEvals> = (0..NUM_FP16_TABLES)
        .into_par_iter()
        .map(|t| {
            if known.columns_per_table[t].is_empty() {
                (vec![], vec![])
            } else {
                let columns: Vec<&PolynomialValues<F>> = known.values_per_table[t].iter().collect();
                eval_columns_at_zeta_and_next::<F, D>(&columns, zeta, system.degree_bits()[t])
            }
        })
        .collect();
    let mut evals_at_zeta = Vec::new();
    let mut evals_at_g_zeta = Vec::new();
    for (at_zeta, at_g_zeta) in per_table {
        evals_at_zeta.extend(at_zeta);
        evals_at_g_zeta.extend(at_g_zeta);
    }
    expected.extend(evals_at_zeta.iter().flat_map(|e| e.0));
    expected.extend(evals_at_g_zeta.iter().flat_map(|e| e.0));
    ensure!(
        expected.len() == total,
        "wrapped proof: internal public-input layout mismatch (built {}, layout says {})",
        expected.len(),
        total
    );
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::v5::api::accumulate::GROUP;
    use crate::v5::api::policy::replay_and_evaluate;

    const VECTORS: &str = include_str!("../api/testdata/a100_dot_vectors.txt");

    /// Builds an `h x w` tile (one reference cell replicated into every A row and B col) whose NOISED
    /// quantization — the operands the batch's matmul/policy actually multiply — clears the policy
    /// gate. Picks the smallest `k` (`k >= 16`, `k % GROUP == 0`) that qualifies for this shape.
    fn accepting_tile(h: usize, w: usize) -> (usize, Vec<u16>, Vec<u16>) {
        use crate::v5::circuit::driver::fp16_noised_operands;
        let mut best: Option<(usize, Vec<u16>, Vec<u16>)> = None;
        for line in VECTORS.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let t: Vec<&str> = line.split_whitespace().collect();
            let k: usize = t[0].parse().unwrap();
            if k % GROUP != 0 || k < 16 {
                continue;
            }
            if best.as_ref().is_some_and(|(bk, _, _)| k >= *bk) {
                continue;
            }
            let a0: Vec<u16> = t[1..1 + k].iter().map(|x| x.parse().unwrap()).collect();
            let b0: Vec<u16> = t[1 + k..1 + 2 * k].iter().map(|x| x.parse().unwrap()).collect();
            let (a, b) = (a0.repeat(h), b0.repeat(w));
            let Ok((noised_a, noised_b)) = fp16_noised_operands(h, w, k, &a, &b, KEY_A, KEY_B, OP_HASH_ID, OP_HASH_ID) else { continue };
            if replay_and_evaluate(&noised_a, &noised_b, h, w, k).1.accept {
                best = Some((k, a, b));
            }
        }
        best.expect("reference vectors must contain a cell whose noised tile accepts for this shape")
    }

    /// Task A capstone: an honest FP16 tile -> batch proof -> recursive wrapped proof VERIFIES, and
    /// a tampered statement or proof is REJECTED. The wrapped (stage-2) proof is a constant-size
    /// recursive artifact whose length is independent of the tile's inner dimension. (This wrapper
    /// is compiled per degree profile; the universal, one-circuit-for-all-sizes variant is the
    /// documented residual — see the module docs and `stark_feasibility.md §8`.)
    //
    // The batch's Blake3 table now runs the full FP16 program (operand-A tree -> HASH_A, operand-B
    // tree -> HASH_B, plus the jackpot compression). Under `Blake3Chunk512` the `4x4` and `4x16`
    // operand trees each stand at `2^8` live-row height — on `FP16_REACHABLE_DEGREE_BITS` — so the
    // table clears the ladder floor the jackpot-only program (`2^3`) sat below, and the wrapper
    // compiles and wraps. The operand-bytes (6c) and operand-codes (6d) CTLs now bind the committed
    // operand bytes to the quant `raw` and the matmul operands to the noised `out`, so the wrapped
    // proof attests the matmul multiplies the noised quantization of the committed operands.
    const KEY_A: Hash256 = [0x11; 32];
    const KEY_B: Hash256 = [0x22; 32];
    /// Keeps both tiles' Blake3 operand trees on the consensus ladder (`2^8`).
    const OP_HASH_ID: crate::v4::api::public_params::HashId =
        crate::v4::api::public_params::HashId::Blake3Chunk512;

    #[test]
    #[ignore = "builds + proves two recursive wrappers (~13 min each); run explicitly"]
    fn wrapped_fp16_proof_verifies_and_rejects_tampering() {
        let _ = env_logger::builder().format_timestamp(None).try_init();
        let mut timing = TimingTree::default();

        let (k, a, b) = accepting_tile(4, 4); // 16 output cells, one per lottery lane.
        let jk = crate::v4::api::transcript::jackpot_key(&[0x5a; 32]);
        let mut system = Fp16System::<F, D>::new(4, 4, k, OP_HASH_ID, OP_HASH_ID);
        let preprocessed = system.preprocessed_data::<InnerC>(&mut timing);
        let lut_cap = preprocessed.cap();

        let circuits = Fp16WrapperCircuits::build(&system, &lut_cap, &mut timing).expect("wrapper compiles");
        let vd = circuits.verifier_data();
        assert!(vd.common.config.zero_knowledge, "the published stage carries ZK blinding");

        let bp = system.prove::<InnerC>(&a, &b, KEY_A, KEY_B, jk, &preprocessed, &mut timing).expect("batch proof");
        let bp_blake3_pis = bp.public_inputs[crate::v5::circuit::ctl::BLAKE3_A100_TABLE].clone();
        // The statement digest is now DERIVED from the proof's own HASH_JACKPOT (not a caller-opaque
        // salt); `prove` bound it, and the wrapper pins it through to the known-column digest PI.
        let digest = Fp16System::<F, D>::statement_digest(&bp_blake3_pis);
        system.verify::<InnerC>(&bp, &bp_blake3_pis, &lut_cap).expect("native batch verify");
        let wrapped = circuits.prove(&system, &bp, digest, &mut timing).expect("wrap");

        // Honest wrapped proof verifies, and is a constant-size recursive artifact.
        verify_wrapped_proof(&system, &vd, &wrapped, digest).expect("honest wrapped proof verifies");
        let len = wrapped.to_bytes().len();
        log::info!("wrapped FP16 proof size: {len} bytes");

        // A second, larger tile (4x8 -> main tables at 2^6 vs the 4x4 tile's 2^5, both on the main
        // range [4,6]; its Blake3 operand trees are still 2^8) wraps to a proof of the SAME length:
        // the recursive artifact's size is independent of the tile's row count (it is a function only
        // of the wrapper circuits, which share the dominating 2^16 committed-LUT shape).
        let (k3, a3, b3) = accepting_tile(4, 8);
        let mut system3 = Fp16System::<F, D>::new(4, 8, k3, OP_HASH_ID, OP_HASH_ID);
        assert_ne!(system3.degree_bits()[0], system.degree_bits()[0], "tiles must differ in main height");
        let preprocessed3 = system3.preprocessed_data::<InnerC>(&mut timing);
        let circuits3 = Fp16WrapperCircuits::build(&system3, &preprocessed3.cap(), &mut timing).expect("wrapper 4x8");
        let bp3 = system3.prove::<InnerC>(&a3, &b3, KEY_A, KEY_B, jk, &preprocessed3, &mut timing).expect("batch proof 4x8");
        // A different tile folds to a different jackpot, hence a different derived statement digest.
        let digest3 = Fp16System::<F, D>::statement_digest(&bp3.public_inputs[crate::v5::circuit::ctl::BLAKE3_A100_TABLE]);
        let wrapped3 = circuits3.prove(&system3, &bp3, digest3, &mut timing).expect("wrap 4x8");
        verify_wrapped_proof(&system3, &circuits3.verifier_data(), &wrapped3, digest3).expect("4x8 wrapped verifies");
        assert_eq!(wrapped3.to_bytes().len(), len, "the wrapped proof size must be constant across tile sizes");

        // Tamper (statement): a different statement digest must be rejected.
        let other_digest: Hash256 = [9u8; 32];
        assert!(
            verify_wrapped_proof(&system, &vd, &wrapped, other_digest).is_err(),
            "a wrong statement digest must be rejected"
        );

        // Tamper (proof): flipping any public-input slot — the exposed zeta or a known-column eval —
        // breaks the pinned-public-input equality or the plonky2 verification.
        for &slot in &[
            zeta_offset(&system),           // the exposed zeta
            zeta_offset(&system) + D,       // the first known-column eval
            wrapped.public_inputs.len() - 1, // the last known-column eval
        ] {
            let mut bad = wrapped.clone();
            bad.public_inputs[slot] += F::ONE;
            assert!(
                verify_wrapped_proof(&system, &vd, &bad, digest).is_err(),
                "tampered wrapped public-input slot {slot} must be rejected"
            );
        }
    }

    /// The header-bound consensus gateway [`verify_wrapped_proof_with_headers`]: the committed
    /// honest ZK-cert fixture verifies under its header; a different proposed header, an impossible
    /// difficulty, or any tampered public-input slot is rejected. Nothing feeding the keys/seeds/
    /// jackpot key is test-supplied — it is all re-derived from the block header + job params.
    ///
    /// FAST (no proving): loads the committed fixture (`node/zkpow/testdata/fp16_zk_cert_a100.bin`,
    /// a wrapper-legal 4x64 tile) and the production verifier cache (`src/api/fp16/fp16_cache.bin`)
    /// and verifies through the cache exactly as the FFI does. Skips if either (gitignored cache /
    /// committed fixture) is absent. Regenerate the fixture + cache with
    /// `api::fp16::zk::fixture::regenerate_zk_cert_fixture` and `build_cache`.
    #[test]
    fn wrapped_fp16_proof_header_binding() {
        use crate::v5::api::zk_cert::Fp16ZkCertificate;
        use crate::v4::api::primitives::IncompleteBlockHeader;
        use crate::v5::circuit::verifier_cache::Fp16VerifierCache;
        use plonky2::plonk::proof::ProofWithPublicInputs;
        use std::path::Path;

        const EASY_NBITS: u32 = 0x207fffff;
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let (cache_bytes, fixture) = match (
            std::fs::read(manifest.join("src/api/fp16/fp16_cache.bin")),
            std::fs::read(manifest.join("../node/zkpow/testdata/fp16_zk_cert_a100.bin")),
        ) {
            (Ok(c), Ok(f)) => (c, f),
            _ => {
                eprintln!("skip: fp16_cache.bin / ZK-cert fixture absent (run build_cache + regenerate_zk_cert_fixture)");
                return;
            }
        };

        // The asymmetric honest header the fixture was keyed under (matches `honest_fixture` /
        // the `regenerate_zk_cert_fixture` generator); the fixture's own 76-byte header prefix.
        let proposed = IncompleteBlockHeader {
            version: 0,
            prev_block: std::array::from_fn(|i| i as u8),
            merkle_root: std::array::from_fn(|i| 0x40u8 + i as u8),
            timestamp: 0x6666_6666,
            nbits: EASY_NBITS,
        };
        let cert_len = u32::from_le_bytes(fixture[76..80].try_into().unwrap()) as usize;
        let cert = Fp16ZkCertificate::from_bytes(&fixture[80..80 + cert_len]).expect("parse ZK cert");
        let (h, w, k) = cert.tile_geometry();
        let (ah, bh) = (cert.job.operands.a.hash_id, cert.job.operands.b.hash_id);

        let cache = Fp16VerifierCache::from_bytes(&cache_bytes).expect("parse production cache");
        let mut system = Fp16System::<F, D>::new(h, w, k, ah, bh);
        let vd = cache.get(system.degree_bits()).expect("production cache covers the fixture profile").circuit();
        let proof = ProofWithPublicInputs::from_bytes(cert.proof_bytes.clone(), &vd.common).expect("deserialize proof");

        // (accept) the header-bound gateway verifies the honest wrapped proof.
        verify_wrapped_proof_with_headers(&mut system, vd, &proof, &proposed, &cert.job, EASY_NBITS)
            .expect("the honest header-bound wrapped proof verifies");

        // (reject) a different proposed header re-derives keyA/seedA/jackpot key -> pin fails.
        let wrong = IncompleteBlockHeader { timestamp: proposed.timestamp ^ 0x5A5A, ..proposed };
        assert!(
            verify_wrapped_proof_with_headers(&mut system, vd, &proof, &wrong, &cert.job, EASY_NBITS).is_err(),
            "a different proposed header must be rejected by the header-bound pin"
        );

        // (reject) an impossible difficulty target fails the native epilogue.
        assert!(
            verify_wrapped_proof_with_headers(&mut system, vd, &proof, &proposed, &cert.job, 0).is_err(),
            "an impossible nbits must be rejected"
        );

        // (reject) a tampered public-input slot breaks the pinned equality / plonky2 verification.
        let mut bad = proof.clone();
        bad.public_inputs[zeta_offset(&system)] += F::ONE;
        assert!(
            verify_wrapped_proof_with_headers(&mut system, vd, &bad, &proposed, &cert.job, EASY_NBITS).is_err(),
            "a tampered wrapped public-input slot must be rejected"
        );
    }
}
