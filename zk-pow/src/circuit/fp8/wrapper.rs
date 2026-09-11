//! Recursive verification and compact encoding for FP8 batch proofs.
//!
//! # Stage 1: batch verification
//!
//! [`verify_universal_batch_stark_proof_circuit`] verifies the arithmetic constraints,
//! cross-table lookups and batched FRI proof against a constant lookup-table cap.
//! The trace degrees are circuit inputs bounded by [`fp8_universal_envelope`].
//!
//! The public inputs appear in this order:
//!
//! ```text
//! table public inputs
//! table degree bits
//! statement digest (4 field elements)
//! zeta (2 field elements)
//! known-column evaluations at zeta
//! known-column evaluations at g_t*zeta
//! ```
//!
//! `zeta` is the inner proof's Fiat–Shamir evaluation challenge. For table `t`,
//! `g_t` is the generator of its trace domain, so `g_t*zeta` opens the next row.
//! The circuit binds the exposed evaluations to the trace openings.
//!
//! [`verify_wrapped_proof`] reconstructs and checks the entire public-input vector.
//! It evaluates the known columns from the statement at both points and supplies
//! geometry inputs such as `K`, `WL2`, key offsets and multiplicities. Those inputs
//! must come from the same statement; the circuit does not derive their mutual
//! consistency. The proof supplies `zeta`, whose value is constrained by the inner verifier.
//!
//! # Stage 2: zero-knowledge wrapper
//!
//! Stage 2 verifies stage 1 using constant verifier data and forwards its public
//! inputs. It enables zero knowledge; only the stage-2 proof is published.
//! Stage 1 uses [`PoseidonGoldilocksConfig`], and stage 2 uses [`Blake3GoldilocksConfig`].
//!
//! # Compact encoding and trusted setup
//!
//! The wire format is `zeta (16 bytes) || outer proof`. It omits the public inputs
//! and the constants/sigmas oracle data. Here sigmas are the proof system's
//! permutation polynomials, distinct from the FP8 noise scales.
//!
//! Verification reconstructs the public inputs from the statement and the omitted
//! oracle data from trusted polynomial coefficients. The reconstruction uses each
//! Fiat–Shamir challenge at its normal transcript position; the prover does not
//! choose those oracle values.
//!
//! [`Fp8CircuitCache`] reuses compiled circuits across job geometries, with one
//! entry per lookup-table cap. Geometry and degrees remain verified public inputs.

use anyhow::{Context, Result, anyhow, ensure};
use hashbrown::HashMap;
use plonky2::field::extension::FieldExtension;
use plonky2::field::extension::quadratic::QuadraticExtension;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::{HashOutTarget, NUM_HASH_OUT_ELTS};
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitData, VerifierCircuitData};
use plonky2::plonk::config::{Blake3GoldilocksConfig, GenericConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::{CompactProofWithPublicInputs, ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::timed;
use plonky2::util::serialization::{Buffer, Read, Write};
use plonky2::util::timing::TimingTree;
use starky::batch_proof::BatchStarkProofWithPublicInputs;
use starky::batch_recursive_verifier::BatchKnownColumnsTarget;
use starky::batch_universal::{
    UniversalBatchStarkVerifierTarget, UniversalVerifierEnvelope, set_universal_batch_stark_proof_with_pis_target,
    verify_universal_batch_stark_proof_circuit,
};
use starky::verifier::eval_columns_at_zeta_and_next;

use super::ctl::{NUM_ALL_TABLES, NUM_TABLES};
use super::driver::{FP8_REACHABLE_DEGREE_BITS, Fp8System, fp8_universal_envelope};
use super::known_values::hash256_to_hash_out;
use crate::api::primitives::Hash256;
use crate::circuit::fp8::circuit_utils::build_recursion_config;

/// The wrapper's field, extension degree and per-stage hasher configurations.
pub type F = GoldilocksField;
pub const D: usize = 2;
/// Stage 1 uses algebraic hashing for recursive verification.
pub type InnerC = PoseidonGoldilocksConfig;
/// Stage 2 uses Blake3 for the published proof.
pub type OuterC = Blake3GoldilocksConfig;

/// FRI parameters for each recursive stage. Degree-seven Poseidon constraints
/// require at least three rate bits for the quotient's LDE blowup.
pub const STAGE_1_RATE_BITS: usize = 3;
pub const STAGE_1_POW_BITS: usize = 18;
pub const STAGE_2_RATE_BITS: usize = 7;
pub const STAGE_2_POW_BITS: usize = 22;

/// Flat offset of batch position `position`'s STARK public inputs inside the wrapper's
/// public-input vector (the batch tables' public inputs are the vector's prefix, in batch
/// order — see the module docs for the full layout).
pub fn table_pis_offset(system: &Fp8System<F, D>, position: usize) -> usize {
    system.batch_starks()[..position].iter().map(|s| s.num_public_inputs()).sum()
}

/// Flat offset of the per-table degree-bits slots (`NUM_ALL_TABLES` of them, canonical
/// batch order), directly after the tables' STARK public inputs.
pub fn degree_bits_offset(system: &Fp8System<F, D>) -> usize {
    table_pis_offset(system, NUM_ALL_TABLES)
}

/// Flat offset of the exposed challenge point `zeta`: the degree slots and the known-column
/// digest occupy the slots before it, the known-column evaluations the slots after.
pub fn zeta_offset(system: &Fp8System<F, D>) -> usize {
    degree_bits_offset(system) + NUM_ALL_TABLES + NUM_HASH_OUT_ELTS
}

/// Total public-input count of a wrapper proof for this batch shape.
fn num_wrapper_public_inputs(system: &Fp8System<F, D>) -> usize {
    let num_known: usize = system.known().columns_per_table.iter().map(Vec::len).sum();
    zeta_offset(system) + D + 2 * D * num_known
}

/// Split the public-input prefix by table. [`verify_wrapped_proof`] checks its values.
pub fn split_batch_public_inputs(system: &Fp8System<F, D>, flat: &[F]) -> Result<Vec<Vec<F>>> {
    ensure!(
        flat.len() == num_wrapper_public_inputs(system),
        "wrapped proof: wrong public input count (got {}, statement expects {})",
        flat.len(),
        num_wrapper_public_inputs(system)
    );
    let mut offset = 0;
    Ok(system
        .batch_starks()
        .iter()
        .map(|s| {
            let n = s.num_public_inputs();
            let pis = flat[offset..offset + n].to_vec();
            offset += n;
            pis
        })
        .collect())
}

/// Compiled circuits indexed by LUT cap; reused by [`Fp8WrapperCircuits::with_cache`].
#[derive(Default)]
pub struct Fp8CircuitCache {
    circuits: HashMap<Fp8CircuitKey, Fp8WrapperCircuits>,
}

impl Fp8CircuitCache {
    /// Number of compiled circuit sets (one per LUT cap seen).
    pub fn len(&self) -> usize {
        self.circuits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.circuits.is_empty()
    }
}

/// The LUT cap selects the circuits; geometry and degrees are verified inputs.
#[derive(Clone, Hash, PartialEq, Eq)]
struct Fp8CircuitKey {
    lut_cap: Vec<u64>,
}

impl Fp8CircuitKey {
    fn new(lut_cap: &MerkleCap<F, <InnerC as GenericConfig<D>>::Hasher>) -> Self {
        Self {
            lut_cap: lut_cap
                .0
                .iter()
                .flat_map(|h| h.elements)
                .map(|e| e.to_canonical_u64())
                .collect(),
        }
    }
}

/// The two compiled wrapper circuits, plus the witness targets the
/// prover fills. Build once ([`Fp8WrapperCircuits::build`]), reuse across every
/// envelope-legal job.
pub struct Fp8WrapperCircuits {
    /// Stage 1: the universal in-circuit batch verifier (Poseidon recursion, no ZK).
    pub stage1: CircuitData<F, InnerC, D>,
    universal_target: UniversalBatchStarkVerifierTarget<D>,
    known_digest_target: HashOutTarget,
    /// The consensus envelope the stage-1 circuit was built on (needed to pad witnesses).
    envelope: UniversalVerifierEnvelope,
    /// Stage 2: the ZK wrap publishing stage 1's public inputs.
    pub stage2: CircuitData<F, OuterC, D>,
    stage1_proof_target: ProofWithPublicInputsTarget<D>,
}

/// A virtual extension target registered (component-wise) as the next public inputs.
fn add_ext_public_input(builder: &mut CircuitBuilder<F, D>) -> ExtensionTarget<D> {
    let target = builder.add_virtual_extension_target();
    builder.register_public_inputs(&target.0);
    target
}

impl Fp8WrapperCircuits {
    /// [`Self::build`] behind `cache`: compiles both stages the first time,
    /// reuses the cached circuits for every later job.
    pub fn with_cache<'c>(
        system: &Fp8System<F, D>,
        lut_cap: &MerkleCap<F, <InnerC as GenericConfig<D>>::Hasher>,
        cache: &'c mut Fp8CircuitCache,
        timing: &mut TimingTree,
    ) -> Result<&'c Self> {
        let key = Fp8CircuitKey::new(lut_cap);
        if !cache.circuits.contains_key(&key) {
            let circuits = Self::build(system, lut_cap, timing)?;
            cache.circuits.insert(key.clone(), circuits);
        }
        Ok(cache.circuits.get(&key).expect("just inserted"))
    }

    /// Compile both stages with `lut_cap` fixed in stage 1.
    /// `system` must use the supported degree schedule. Job geometry is supplied
    /// through public inputs and known-column evaluations.
    pub fn build(
        system: &Fp8System<F, D>,
        lut_cap: &MerkleCap<F, <InnerC as GenericConfig<D>>::Hasher>,
        timing: &mut TimingTree,
    ) -> Result<Self> {
        let known = system.known();
        ensure!(
            system
                .degree_bits()
                .iter()
                .all(|bits| FP8_REACHABLE_DEGREE_BITS.contains(bits)),
            "universal wrapper: the degree profile {:?} must lie on the consensus ladder",
            system.degree_bits()
        );
        let envelope = fp8_universal_envelope();

        // Stage 1: the universal batch verifier in-circuit.
        let config_1 = build_recursion_config(STAGE_1_RATE_BITS, STAGE_1_POW_BITS, 1, false);
        let mut builder = CircuitBuilder::<F, D>::new(config_1);
        let starks = system.batch_starks();
        let preprocessed = system.preprocessed_verifier_data::<InnerC>(lut_cap);
        let preprocessed_target = preprocessed.constant_target(&mut builder);

        // Allocate known-column claims now; register them after the table public inputs.
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

        let universal = verify_universal_batch_stark_proof_circuit::<F, InnerC, D, NUM_ALL_TABLES>(
            &mut builder,
            &starks,
            system.config(),
            &envelope,
            system.ctls(),
            Some(&preprocessed_target),
            Some(&known_target),
            &HashMap::new(),
        )?;

        // Register public inputs in the order consumed by `expected_wrapper_public_inputs`.
        for pis in &universal.proof_with_pis.public_inputs {
            builder.register_public_inputs(pis);
        }
        builder.register_public_inputs(&universal.degree_bits);
        builder.register_public_inputs(&known_digest_target.elements);
        let zeta_pi = add_ext_public_input(&mut builder);
        // Bind the exposed zeta to the Fiat-Shamir challenge derived inside the verifier.
        builder.connect_extension(zeta_pi, universal.zeta);
        for evals in known_target.evals_at_zeta.iter().chain(&known_target.evals_at_g_zeta) {
            for eval in evals {
                builder.register_public_inputs(&eval.0);
            }
        }

        let stage1_gates = builder.num_gates();
        let stage1 = timed!(timing, "build the stage-1 wrapper circuit", builder.build::<InnerC>());
        log::info!(
            "stage-1 universal wrapper circuit: {stage1_gates} gates -> 2^{} rows",
            stage1.common.degree_bits(),
        );

        // Stage 2: the ZK wrap.
        let config_2 = build_recursion_config(STAGE_2_RATE_BITS, STAGE_2_POW_BITS, 2, true);
        debug_assert!(config_2.zero_knowledge);
        let mut builder = CircuitBuilder::<F, D>::new(config_2);
        let stage1_proof_target = builder.add_virtual_proof_with_pis(&stage1.common);
        // Forward stage 1's public inputs unchanged.
        builder.register_public_inputs(&stage1_proof_target.public_inputs);
        // Stage 2 can verify only the stage-1 circuit compiled above.
        let stage1_verifier_target = builder.constant_verifier_data(&stage1.verifier_only);
        builder.verify_proof::<InnerC>(&stage1_proof_target, &stage1_verifier_target, &stage1.common);
        let stage2 = timed!(timing, "build the stage-2 (ZK) wrapper circuit", builder.build::<OuterC>());
        log::info!("stage-2 wrapper circuit: 2^{} gates", stage2.common.degree_bits());

        Ok(Self {
            stage1,
            universal_target: universal,
            known_digest_target,
            envelope,
            stage2,
            stage1_proof_target,
        })
    }

    /// Prove both recursive stages and return the outer zero-knowledge proof.
    /// `statement_digest` must match the batch proof's Fiat-Shamir binding,
    /// including the generated jackpot digest.
    pub fn prove(
        &self,
        system: &Fp8System<F, D>,
        batch_proof: &BatchStarkProofWithPublicInputs<F, InnerC, D>,
        statement_digest: Hash256,
        timing: &mut TimingTree,
    ) -> Result<ProofWithPublicInputs<F, OuterC, D>> {
        system.ensure_statement_digest_binding(statement_digest)?;
        let mut witness = PartialWitness::new();
        set_universal_batch_stark_proof_with_pis_target(
            &mut witness,
            &self.universal_target,
            batch_proof,
            &self.envelope,
            system.config(),
        )?;
        witness.set_hash_target(self.known_digest_target, hash256_to_hash_out(statement_digest))?;
        let stage1_proof = timed!(timing, "prove the stage-1 wrapper", self.stage1.prove(witness))?;

        let mut witness = PartialWitness::new();
        witness.set_proof_with_pis_target(&self.stage1_proof_target, &stage1_proof)?;
        timed!(timing, "prove the stage-2 (ZK) wrapper", self.stage2.prove(witness))
    }

    /// The verifier's view: stage 2's verifier data (the only circuit a verifier needs).
    pub fn verifier_data(&self) -> VerifierCircuitData<F, OuterC, D> {
        self.stage2.verifier_data()
    }

    /// Stage 2's constants/sigmas polynomial coefficients.
    pub fn constants_sigmas_polynomials(&self) -> Vec<PolynomialCoeffs<F>> {
        self.stage2.prover_only.constants_sigmas_commitment.polynomials.clone()
    }
}

/// Verify the outer proof and its statement-derived public inputs.
///
/// Reconstruct the vector from `expected_public_inputs`, the degree profile,
/// `statement_digest` and known-column evaluations. Zeta is supplied by the
/// proof and constrained by the inner verifier. `verifier_data` must be the
/// trusted zero-knowledge outer circuit.
pub fn verify_wrapped_proof(
    system: &Fp8System<F, D>,
    verifier_data: &VerifierCircuitData<F, OuterC, D>,
    proof: &ProofWithPublicInputs<F, OuterC, D>,
    expected_public_inputs: &[Vec<F>; NUM_TABLES],
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
    let zeta_start = zeta_offset(system);
    let zeta = QuadraticExtension([proof.public_inputs[zeta_start], proof.public_inputs[zeta_start + 1]]);
    let expected = expected_wrapper_public_inputs(system, expected_public_inputs, statement_digest, zeta)?;
    ensure!(
        proof.public_inputs == expected,
        "wrapped proof: public inputs do not match the statement's expectations"
    );
    verifier_data.verify(proof.clone())
}

/// The full stage-2 public-input vector a statement expects, given the proof's `zeta`.
pub fn expected_wrapper_public_inputs(
    system: &Fp8System<F, D>,
    expected_public_inputs: &[Vec<F>; NUM_TABLES],
    statement_digest: Hash256,
    zeta: QuadraticExtension<GoldilocksField>,
) -> Result<Vec<F>> {
    ensure!(
        !FieldExtension::<D>::is_in_basefield(&zeta),
        "wrapped proof: zeta must lie strictly in the extension field"
    );
    let total = num_wrapper_public_inputs(system);
    let known = system.known();
    let mut expected: Vec<F> = Vec::with_capacity(total);
    for pis in system.batch_public_inputs(expected_public_inputs) {
        expected.extend(pis);
    }
    expected.extend(system.degree_bits().iter().map(|&bits| F::from_canonical_usize(bits)));
    expected.extend(hash256_to_hash_out::<F>(statement_digest).elements);
    expected.extend(zeta.0);
    // Recompute known columns at both points checked by native batch verification.
    let mut evals_at_zeta = Vec::new();
    let mut evals_at_g_zeta = Vec::new();
    for t in 0..NUM_ALL_TABLES {
        if known.columns_per_table[t].is_empty() {
            continue;
        }
        let columns: Vec<&PolynomialValues<F>> = known.values_per_table[t].iter().collect();
        let (at_zeta, at_g_zeta) = eval_columns_at_zeta_and_next::<F, D>(&columns, zeta, system.degree_bits()[t]);
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

/// Zeta preamble: two canonical Goldilocks elements, eight little-endian bytes each.
pub const COMPACT_ZETA_PREAMBLE: usize = 16;

/// Encodes a wrapped (stage-2) proof in the published compact form:
/// `zeta (16 bytes) || compact plonky2 proof`.
pub fn compact_proof_data(system: &Fp8System<F, D>, proof: &ProofWithPublicInputs<F, OuterC, D>) -> Result<Vec<u8>> {
    let total = num_wrapper_public_inputs(system);
    ensure!(
        proof.public_inputs.len() == total,
        "compact encode: wrong public input count (got {}, statement expects {})",
        proof.public_inputs.len(),
        total
    );
    let zeta_start = zeta_offset(system);
    let mut data = Vec::new();
    data.write_field_vec(&[proof.public_inputs[zeta_start], proof.public_inputs[zeta_start + 1]])
        .expect("writing to a byte vector cannot fail");
    debug_assert_eq!(data.len(), COMPACT_ZETA_PREAMBLE);
    let compact: CompactProofWithPublicInputs<F, OuterC, D> = proof.clone().into();
    data.extend(compact.to_proof_bytes());
    Ok(data)
}

/// Verify [`compact_proof_data`] using trusted setup polynomials.
///
/// The polynomial coefficients must come from the trusted outer setup
/// ([`Fp8WrapperCircuits::constants_sigmas_polynomials`]). Public inputs are
/// reconstructed from the statement and encoded zeta; non-canonical encodings are rejected.
pub fn verify_compact_wrapped_proof(
    system: &Fp8System<F, D>,
    verifier_data: &VerifierCircuitData<F, OuterC, D>,
    constants_sigmas_polynomials: &[PolynomialCoeffs<F>],
    proof_data: &[u8],
    expected_public_inputs: &[Vec<F>; NUM_TABLES],
    statement_digest: Hash256,
) -> Result<()> {
    ensure!(
        verifier_data.common.config.zero_knowledge,
        "the wrapped verifier circuit must be the ZK stage"
    );
    system.ensure_statement_digest_binding(statement_digest)?;
    ensure!(
        proof_data.len() > COMPACT_ZETA_PREAMBLE,
        "compact wrapped proof: too short (needs a {COMPACT_ZETA_PREAMBLE}-byte zeta preamble and a proof body)"
    );
    let (zeta_bytes, proof_bytes) = proof_data.split_at(COMPACT_ZETA_PREAMBLE);
    let zeta_limbs = Buffer::new(zeta_bytes)
        .read_field_vec(2)
        .map_err(|e| anyhow!("compact wrapped proof: non-canonical zeta encoding: {e:?}"))?;
    let zeta = QuadraticExtension([zeta_limbs[0], zeta_limbs[1]]);

    let expected = expected_wrapper_public_inputs(system, expected_public_inputs, statement_digest, zeta)?;
    // Check the reconstructed vector length before hashing it for verification.
    ensure!(
        expected.len() == verifier_data.common.num_public_inputs,
        "compact wrapped proof: the statement's public-input layout ({}) does not match the circuit ({})",
        expected.len(),
        verifier_data.common.num_public_inputs
    );
    let compact = CompactProofWithPublicInputs::<F, OuterC, D>::from_bytes(proof_bytes, expected, &verifier_data.common)
        .map_err(|e| anyhow!("compact wrapped proof: malformed proof bytes: {e:?}"))?;
    ensure!(
        compact.to_proof_bytes() == proof_bytes,
        "compact wrapped proof: non-canonical proof encoding"
    );
    // Verification reconstructs omitted constants/sigmas data from the setup polynomials
    // after drawing the corresponding Fiat-Shamir challenges.
    compact
        .verify(
            &verifier_data.verifier_only,
            &verifier_data.common,
            constants_sigmas_polynomials,
        )
        // The FFI exposes this context for both statement mismatches and invalid proofs.
        .context(
            "compact fp8 proof does not verify against the expected statement (mismatched header/statement or an invalid proof)",
        )
}
