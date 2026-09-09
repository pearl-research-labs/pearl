//! The two-stage recursive wrapper for fp8 batch proofs — the fp8 analogue of the
//! deployed v2 `pearl_circuit` architecture.
//!
//! **Why wrap.** The batch multi-STARK proof of [`super::driver::Fp8System`] is large and
//! is *not* zero-knowledge as a published object (its FRI oracles are unblinded; only the
//! trace carries the AIR-level blinding cells). The wrapper encodes the whole batch
//! verification inside a plonky2 circuit and publishes a constant-size recursive proof
//! instead:
//!
//! - **Stage 1** (Poseidon recursion, no ZK): the *universal* batch verifier
//!   ([`verify_universal_batch_stark_proof_circuit`]) re-runs the full batch verification
//!   in-circuit — every AIR's constraints, the lookups and CTL balances, the setup-time LUT
//!   cap (baked into the circuit as a constant) and the batched FRI argument — for **any**
//!   degree profile inside the consensus envelope
//!   ([`fp8_universal_envelope`]): the per-table trace heights are circuit inputs, not
//!   compile-time constants, so one compiled circuit covers every
//!   envelope-legal job (the D1 redesign). Its public inputs are, in order:
//!
//!   ```text
//!   every table's STARK public inputs (batch order)
//!   | per-table trace degree bits (NUM_ALL_TABLES)
//!   | known-column digest (4)
//!   | zeta (2)
//!   | known-column evals at zeta      (2 per column, batch order)
//!   | known-column evals at g_t*zeta  (2 per column, batch order)
//!   ```
//!
//!   The class (a) ("known") columns cannot be evaluated in-circuit (they are full-height
//!   per-job columns), so — exactly like the deployed v2 layer 1 — the circuit *exposes*
//!   the challenge point `zeta` (connected to the Fiat-Shamir challenge derived in-circuit)
//!   and the claimed evaluations (connected to the proof's trace openings), and the native
//!   gateway [`verify_wrapped_proof`] recomputes the evaluations at `zeta` from its own
//!   statement-derived known values and pins every slot — including the degree slots, which
//!   it pins to the statement's own profile. A prover can therefore not lie about any
//!   class (a) column or any table height without either breaking the in-circuit FRI
//!   binding or failing the gateway's public-input equality.
//!
//! - **Stage 2** (ZK wrap): a plonky2 circuit with `zero_knowledge: true` (blinding per
//!   <https://eprint.iacr.org/2024/1037.pdf>, the same configuration as the deployed v2
//!   layer 2) that verifies the stage-1 proof and forwards its public inputs unchanged.
//!   Stage 1's verifier data is baked into stage 2 as circuit constants, so a stage-2 proof
//!   attests to exactly this statement's stage-1 circuit. The published artifact is the
//!   stage-2 proof only; the batch proof and the stage-1 proof never leave the prover.
//!
//! **Compact published encoding.** The published `proof_data` is the *compact* serialization
//! of the stage-2 proof. The wire drops everything the verifier can rebuild from the statement
//! and its trusted setup:
//!
//! - the **public-input vector** — every slot is a pure function of the statement except
//!   the inner challenge point `zeta`, which travels as a 16-byte preamble;
//! - the **constants/sigmas oracle data** — the openings at `zeta` and, per FRI query,
//!   the tree-0 leaf evaluations and Merkle proofs. The verifier recomputes these
//!   from the setup's trusted `constants_sigmas_polynomials`, so the prover supplies —
//!   and can influence — none of them.
//!
//! Like the deployed v2 wrapper, stage 1 proves under [`PoseidonGoldilocksConfig`] (cheap
//! recursion) and stage 2 under [`Blake3GoldilocksConfig`] (on-chain-friendly hashing).
//!
//! **Circuit identity.** The compiled circuits are consensus constants:
//! the AIR identities are program-independent (geometry enters as public inputs and known
//! columns — phases A/B3), and the known-column index layout, the CTL set, the LUT cap,
//! the envelope and the FRI ladder are consensus constants. A
//! [`Fp8WrapperCircuits`] is therefore reusable across *all* envelope-legal
//! jobs, and any mismatch fails closed: a proof outside the envelope cannot satisfy the
//! degree flags, and the gateway independently derives every pinned slot from its own
//! statement. [`Fp8CircuitCache`] memoizes the compiled circuits
//! ([`Fp8WrapperCircuits::with_cache`]).
//!
//! **Geometry public inputs.** Scale's `K`/`WL2`, InputQuant's `2^WL2` and InputQuant's
//! four CTL geometry slots (`h*k`, `h*k/8`, `w`, `h`) are inner-STARK public inputs
//! re-exported as stage-1 public inputs — plain wires, with no in-circuit relations
//! between them. Their well-formedness (`WL2 = 27 - ceil(log2 K)`, `2^WL2` its power,
//! the CTL slots matching `h`, `w`, `k`) is the gateway's job: it pins every slot to
//! its own recomputation from the statement, rejecting any other assignment.

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
// `build_recursion_config` lives in the frozen v2 clone since the mainline circuit
// module no longer carries the legacy recursion stack.
use crate::v2::circuit::circuit_utils::build_recursion_config;

/// The wrapper's field, extension degree and per-stage hasher configurations.
pub type F = GoldilocksField;
pub const D: usize = 2;
/// Stage-1 configuration: the batch proof's own config (its caps must be algebraic for the
/// in-circuit Fiat-Shamir replay) and the cheap recursion hasher.
pub type InnerC = PoseidonGoldilocksConfig;
/// Stage-2 (published) configuration: Blake3 outer hashing, like the deployed v2 layer 2.
pub type OuterC = Blake3GoldilocksConfig;

/// Sanctioned FRI parameters of the two wrapper stages — the same consensus combinations as
/// the deployed v2 wrapper (`pearl_circuit::STAGE_1_PARAMS` / `STAGE_2_PARAMS`). Rate
/// `2^-3` is the floor for both stages: the recursion gates carry degree-7 constraints
/// (Poseidon), so the quotient needs an LDE blowup of at least `2^3`.
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

/// Splits the wrapper's flat public-input prefix back into per-table STARK public inputs
/// (batch positions). Purely syntactic; [`verify_wrapped_proof`] is the judge of the values.
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

/// In-memory cache of compiled wrapper circuits, keyed by the baked
/// LUT cap. Own one for the prover's lifetime and pass it to
/// [`Fp8WrapperCircuits::with_cache`]: the first job compiles, every later job
/// reuses — whatever its geometry or degree profile.
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

/// Everything that determines the compiled circuits: the baked LUT cap — a pure function of
/// the committed tables and the consensus config.
///
/// Nothing else is key material (the D1 universal design): the degree profile and the
/// geometry are runtime inputs — the profile rides the degree public inputs, the geometry
/// rides the `K`/`WL2`/`2^WL2` public inputs and the known columns (whose evaluations the
/// gateway pins natively) — and the AIR identities, the CTL set, the known-column index
/// layout, the envelope and the FRI ladder are consensus constants.
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
    let et = builder.add_virtual_extension_target();
    builder.register_public_inputs(&et.0);
    et
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

    /// Compiles both stages, baking `lut_cap` (the
    /// consensus LUT commitment) into stage 1 as constants.
    ///
    /// The compiled circuits do not depend on `system`'s job: the AIRs are
    /// program-independent, and every degree- or geometry-dependent quantity is a circuit
    /// input (the second-geometry leg of `wrapped_api_roundtrip_and_tamper_rejection` pins
    /// this). `system`'s profile must lie on the consensus ladder — its config then *is*
    /// the consensus config the universal circuit's Fiat-Shamir replay absorbs.
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

        // ---- Stage 1: the universal batch verifier in-circuit. ----
        let config_1 = build_recursion_config(STAGE_1_RATE_BITS, STAGE_1_POW_BITS, 1, false);
        let mut builder = CircuitBuilder::<F, D>::new(config_1);
        let starks = system.batch_starks();
        let preprocessed = system.preprocessed_verifier_data::<InnerC>(lut_cap);
        let preprocessed_target = preprocessed.constant_target(&mut builder);

        // The known-column surface: a digest slot and one claimed evaluation per column
        // per point, created up front (registered as public inputs below, after the
        // tables' own public inputs).
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

        // The public-input layout (see the module docs): table public inputs, the degree
        // slots (the universal verifier's height inputs — the gateway pins them to the
        // statement), the known-column digest, zeta, then the claimed evaluations.
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

        // ---- Stage 2: the ZK wrap. ----
        let config_2 = build_recursion_config(STAGE_2_RATE_BITS, STAGE_2_POW_BITS, 2, true);
        debug_assert!(config_2.zero_knowledge);
        let mut builder = CircuitBuilder::<F, D>::new(config_2);
        let stage1_proof_target = builder.add_virtual_proof_with_pis(&stage1.common);
        // Forward stage 1's public inputs unchanged.
        builder.register_public_inputs(&stage1_proof_target.public_inputs);
        // Stage 1's verifier data is a circuit constant: a stage-2 proof commits to exactly
        // this statement's stage-1 circuit (no native digest pinning needed, unlike the
        // deployed v2 layer 2 which exposes the layer-1 digest as public inputs).
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

    /// Wraps one batch proof: proves stage 1 (the in-circuit batch verification), then
    /// stage 2 (the ZK wrap). Returns the publishable stage-2 proof.
    ///
    /// `batch_proof` must be a proof of `system`'s statement; any mismatch fails closed
    /// when the witness is set (an
    /// off-envelope profile is rejected by the degree flags). `statement_digest` is the
    /// Fiat-Shamir salt the inner batch proof absorbed (`PublicParams::digest` after `J`).
    /// Cross-checked against the system's own bound digest, if any.
    pub fn prove(
        &self,
        system: &Fp8System<F, D>,
        batch_proof: &BatchStarkProofWithPublicInputs<F, InnerC, D>,
        statement_digest: Hash256,
        timing: &mut TimingTree,
    ) -> Result<ProofWithPublicInputs<F, OuterC, D>> {
        system.ensure_statement_digest_binding(statement_digest)?;
        let mut pw = PartialWitness::new();
        set_universal_batch_stark_proof_with_pis_target(
            &mut pw,
            &self.universal_target,
            batch_proof,
            &self.envelope,
            system.config(),
        )?;
        pw.set_hash_target(self.known_digest_target, hash256_to_hash_out(statement_digest))?;
        let stage1_proof = timed!(timing, "prove the stage-1 wrapper", self.stage1.prove(pw))?;

        let mut pw = PartialWitness::new();
        pw.set_proof_with_pis_target(&self.stage1_proof_target, &stage1_proof)?;
        timed!(timing, "prove the stage-2 (ZK) wrapper", self.stage2.prove(pw))
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

/// Verifies a wrapped (stage-2) proof against the statement:
///
/// 1. pins *every* public-input slot — the per-table STARK public inputs to
///    `expected_public_inputs` (at their batch positions), the degree slots to the
///    statement's own profile (the heights the universal stage-1 circuit
///    verified are inputs), the known-column digest to `statement_digest`, and the
///    known-column evaluations to the verifier's *native recomputation* at the proof's
///    `zeta` from the statement's known values (`zeta` itself is the one prover-supplied
///    slot: the stage-1 circuit constrains it to equal the inner batch proof's Fiat-Shamir
///    challenge);
/// 2. verifies the plonky2 proof against `verifier_data` (which must be the ZK stage).
///
/// Together with the in-circuit batch verification this gives exactly the guarantees of
/// [`Fp8System::verify`] on the underlying batch proof.
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
    let z = zeta_offset(system);
    let zeta = QuadraticExtension([proof.public_inputs[z], proof.public_inputs[z + 1]]);
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
    // The class (a) recompute: evaluate the statement's own known columns at the proof's
    // zeta and g_t * zeta — the same binding `batch_verify` performs natively.
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
    // Layout invariant, fail closed: the assembled vector must fill the layout exactly.
    ensure!(
        expected.len() == total,
        "wrapped proof: internal public-input layout mismatch (built {}, layout says {})",
        expected.len(),
        total
    );
    Ok(expected)
}

/// Byte length of the compact wire preamble: `zeta` as 2 canonical Goldilocks limbs
/// (8 bytes each, little-endian) — the layout of the deployed v2 certificate's zeta field.
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
    let z = zeta_offset(system);
    let mut data = Vec::new();
    data.write_field_vec(&[proof.public_inputs[z], proof.public_inputs[z + 1]])
        .expect("writing to a byte vector cannot fail");
    debug_assert_eq!(data.len(), COMPACT_ZETA_PREAMBLE);
    let compact: CompactProofWithPublicInputs<F, OuterC, D> = proof.clone().into();
    data.extend(compact.to_proof_bytes());
    Ok(data)
}

/// Verifies a compact wrapped proof ([`compact_proof_data`]) against the statement.
/// The compact counterpart of [`verify_wrapped_proof`], giving the same guarantees:
///
/// 1. reads `zeta` from the 16-byte preamble, rejecting non-canonical limb encodings
///    (`read_field_vec`) — the only prover-supplied slot on the wire;
/// 2. *imposes* the statement's own expected public-input vector
///    ([`expected_wrapper_public_inputs`]): the proof is verified against the hash of the
///    verifier-built vector, so a proof of any other statement fails the Fiat-Shamir
///    public-input binding — the compact analogue of the full path's slot equality check
///    (this is the deployed v2 verification pattern);
/// 3. rejects non-canonical proof encodings by re-encoding: exactly one byte string is
///    accepted per proof, the same malleability rule `Fp8Verifier::decode_proof` enforces
///    for the full format;
/// 4. runs [`CompactProofWithPublicInputs::verify`], which rebuilds the constants/sigmas
///    openings and per-query tree-0 evaluations from `constants_sigmas_polynomials`
///    (trusted setup, [`Fp8WrapperCircuits::constants_sigmas_polynomials`]) *after* each
///    Fiat-Shamir challenge is drawn at its standard transcript position, then delegates
///    to the standard plonky2 verifier with tree-0 Merkle authentication skipped — sound
///    because those values are the verifier's own, never the prover's.
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
    // Fail closed against the compiled circuit before hashing the imposed vector: the
    // statement's layout and the circuit's registered public-input count must agree.
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
    // With imposed public inputs, a proof for a different statement and a cryptographically
    // invalid proof are the same failure (the Fiat-Shamir public-input binding breaks), so
    // the message names both. This is the outermost context: FFI errors surface only this.
    compact
        .verify(
            &verifier_data.verifier_only,
            &verifier_data.common,
            constants_sigmas_polynomials,
        )
        .context(
            "compact fp8 proof does not verify against the expected statement (mismatched header/statement or an invalid proof)",
        )
}
