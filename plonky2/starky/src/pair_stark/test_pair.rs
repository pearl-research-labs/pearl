#![allow(clippy::too_many_arguments)]
extern crate alloc;

mod pair_air;
mod pair_air_ext;
mod pair_trace;

use core::marker::PhantomData;
use std::time::Instant;

use anyhow::Result;
use log::{info, Level};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::fri::reduction_strategies::FriReductionStrategy;
use plonky2::fri::FriConfig;
use plonky2::hash::hash_types::{HashOut, HashOutTarget, RichField};
use plonky2::iop::challenger::Challenger;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::{
    AlgebraicHasher, GenericConfig, KeccakGoldilocksConfig, PoseidonGoldilocksConfig,
};
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::util::serialization::{Buffer, Read, Write};
use plonky2::util::timing::TimingTree;
use starky::config::StarkConfig;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::evaluation_frame::StarkFrame;
use starky::proof::{StarkProofWithPublicInputs, StarkProofWithPublicInputsTarget};
use starky::prover::prove;
use starky::recursive_verifier::{
    add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target, verify_stark_proof_circuit,
    PreprocessedDataTarget,
};
use starky::stark::Stark;
use starky::util::trace_rows_to_poly_values;
use starky::verifier::{eval_columns_at_point, verify_stark_proof, PreprocessedData};

use crate::pair_trace::generate_trace_rows;

pub const PAIR_COLUMNS: usize = 4;
pub const PAIR_PUBLIC_INPUTS: usize = 3;
const PREPROCESSED_COLUMNS: [usize; 2] = [0, 1];

/// STARK for the following programmable-variant of fibonacci, that introduces preprocessed columns a,b:
///
/// F[0]= public_input[0]
/// F[1] = public_input[1]
/// F[N] = public_input[2]
/// and for 1 < i < N:
/// if a[i] == 0:
///     F[i] = F[i-1] + F[i-2] + b[i]
/// if a[i] == 1:
///     F[i] = F[i-1] * F[i-2] + b[i]
///
/// Specifically in this example we set a[i] = i % 2 and b[i] = i,
/// but prover & verifier can set any agreed value of a, b with only O(N) cost for both.
///
#[derive(Copy, Clone, Debug)]
pub struct PAirStark<F: RichField + Extendable<D>, const D: usize> {
    num_rows: usize,
    mismatch_preprocessed: bool,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> PAirStark<F, D> {
    pub fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            mismatch_preprocessed: false,
            _phantom: PhantomData,
        }
    }

    pub fn with_mismatch(mut self) -> Self {
        self.mismatch_preprocessed = true;
        self
    }

    /// Generates the trace.
    pub fn generate_trace(&self) -> (Vec<PolynomialValues<F>>, [F; PAIR_PUBLIC_INPUTS]) {
        let (trace_rows, public_inputs) = generate_trace_rows(self.num_rows);
        let trace_poly_values = trace_rows_to_poly_values(trace_rows);
        (trace_poly_values, public_inputs)
    }

    fn preprocessed_values(&self) -> Vec<Vec<F>> {
        let (trace, _) = self.generate_trace();
        PREPROCESSED_COLUMNS
            .into_iter()
            .enumerate()
            .map(|(pos, i)| {
                let mut col_values = trace[i].values.clone();
                if self.mismatch_preprocessed && pos == 1 {
                    // In the original trace, trace[i=1][row=10] is 10 (always equal to row).
                    // When we want to simulate a mismatch between trace and a preprocessed column, we override
                    // the preprocessed column and check that verification fails.
                    col_values[10] = F::from_canonical_u32(9);
                }
                col_values
            })
            .collect()
    }

    /// Creates a `PreprocessedData` for this stark. Uses `HashOut::ZERO` as the digest,
    /// which is a valid opaque identifier for tests where a real hash is not needed.
    fn make_preprocessed_data(&self) -> PreprocessedData<F> {
        let columns = self
            .preprocessed_values()
            .into_iter()
            .map(PolynomialValues::new)
            .collect();
        PreprocessedData {
            digest: HashOut::ZERO,
            columns,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for PAirStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, PAIR_COLUMNS, PAIR_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, PAIR_COLUMNS, PAIR_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        pair_air::air_eval_packed::<F, FE, P, D, D2>(vars, yield_constr);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        pair_air_ext::eval_ext_circuit(builder, vars, yield_constr);
    }

    fn constraint_degree(&self) -> usize {
        3 // can be reduced to 2, used only in add3 gate
    }
}

/// Builds the verifier circuit data without generating a proof.
///
/// Returns the circuit data, the stark proof target, and the eval PI targets
/// for setting witnesses in `prove_first_recursion_layer`.
fn build_verifier_circuit_data<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F> + 'static,
    S: Stark<F, D> + Copy,
    const D: usize,
>(
    stark: S,
    stark_config: &StarkConfig,
    recursion_config: &CircuitConfig,
    degree_bits: usize,
    num_preprocessed_cols: usize,
) -> (
    CircuitData<F, C, D>,
    StarkProofWithPublicInputsTarget<D>,
    Vec<ExtensionTarget<D>>,
    Vec<ExtensionTarget<D>>,
)
where
    C::Hasher: AlgebraicHasher<F>,
{
    let mut builder = CircuitBuilder::<F, D>::new(recursion_config.clone());
    let pt =
        add_virtual_stark_proof_with_pis(&mut builder, &stark, stark_config, degree_bits, 0, 0);

    for inner_pi in pt.public_inputs.iter() {
        let outer_pi = builder.add_virtual_public_input();
        builder.connect(*inner_pi, outer_pi);
    }

    // Register the preprocessed column evaluations at zeta and g*zeta as public inputs.
    // The circuit constrains these to equal the proof's actual openings.
    let add_eval_pi_vec = |builder: &mut CircuitBuilder<F, D>| -> Vec<ExtensionTarget<D>> {
        (0..num_preprocessed_cols)
            .map(|_| {
                let t = builder.add_virtual_extension_target();
                builder.register_public_input(t.0[0]);
                builder.register_public_input(t.0[1]);
                t
            })
            .collect()
    };
    let evals_at_zeta = add_eval_pi_vec(&mut builder);
    let evals_at_g_zeta = add_eval_pi_vec(&mut builder);

    // Use HashOut::ZERO as a constant digest for the preprocessed columns.
    // Both prover and verifier must agree on this digest.
    let preprocessed = if !stark_config.preprocessed_columns.is_empty() {
        let digest_targets: Vec<_> = HashOut::<F>::ZERO
            .elements
            .iter()
            .map(|&e| builder.constant(e))
            .collect();
        Some(PreprocessedDataTarget {
            digest: Some(HashOutTarget::from_vec(digest_targets)),
            evals_at_zeta: evals_at_zeta.clone(),
            evals_at_g_zeta: evals_at_g_zeta.clone(),
        })
    } else {
        None
    };

    let _stark_zeta = verify_stark_proof_circuit::<F, C, S, D>(
        &mut builder,
        stark,
        &pt,
        stark_config,
        None,
        preprocessed.as_ref(),
    );

    let data = builder.build::<C>();
    (data, pt, evals_at_zeta, evals_at_g_zeta)
}

fn recursive_prove<OuterC: GenericConfig<2, F = GoldilocksField> + 'static>(
    cur_config: &CircuitConfig,
    prev_config: &CircuitConfig,
    prev_circuit: &CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>,
    prev_proof: &ProofWithPublicInputs<GoldilocksField, PoseidonGoldilocksConfig, 2>,
) -> anyhow::Result<(
    ProofWithPublicInputs<GoldilocksField, OuterC, 2>,
    CircuitData<GoldilocksField, OuterC, 2>,
)> {
    type F = GoldilocksField;
    type InnerC = PoseidonGoldilocksConfig;
    const D: usize = 2;

    let CircuitData {
        prover_only: _,
        verifier_only: verifier_data,
        common: common_data,
    } = prev_circuit;

    let mut builder = CircuitBuilder::<F, D>::new(cur_config.clone());
    let mut pt = builder.add_virtual_proof_with_pis(common_data);
    for inner_pi in pt.public_inputs.iter_mut() {
        let outer_pi = builder.add_virtual_public_input();
        builder.connect(*inner_pi, outer_pi);
    }
    let inner_data = builder.add_virtual_verifier_data(prev_config.fri_config.cap_height);

    // Add constraints to ensure the inner verifier data matches the expected circuit
    let expected_circuit_digest = builder.constant_hash(verifier_data.circuit_digest);
    builder.connect_hashes(inner_data.circuit_digest, expected_circuit_digest);

    for (i, expected_cap) in verifier_data.constants_sigmas_cap.0.iter().enumerate() {
        let expected_cap_target = builder.constant_hash(*expected_cap);
        builder.connect_hashes(inner_data.constants_sigmas_cap.0[i], expected_cap_target);
    }

    builder.verify_proof::<InnerC>(&pt, &inner_data, common_data);
    let new_circuit_data = builder.build::<OuterC>();

    let mut pw = PartialWitness::new();
    pw.set_proof_with_pis_target(&pt, prev_proof)?;
    pw.set_verifier_data_target(&inner_data, verifier_data)?;

    #[cfg(debug_assertions)]
    let timer_start = std::time::Instant::now();

    let proof = plonky2::plonk::prover::prove::<GoldilocksField, OuterC, 2>(
        &new_circuit_data.prover_only,
        &new_circuit_data.common,
        pw,
        &mut TimingTree::new("prove", Level::Debug),
    )?;

    #[cfg(debug_assertions)]
    println!(
        "Second recursion net prove time: {:?}",
        timer_start.elapsed()
    );

    Ok((proof, new_circuit_data))
}

#[cfg(test)]
mod tests {
    use sorted_vec::SortedSet;

    use super::*;

    const LOG2_ROWS: usize = 11;

    fn stark_config() -> StarkConfig {
        StarkConfig {
            security_bits: 100,
            num_challenges: 2,
            fri_config: FriConfig {
                rate_bits: 3,
                cap_height: 4,
                proof_of_work_bits: 10,
                batching_proof_of_work_bits: 0,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 8),
                num_query_rounds: 30,
            },
            preprocessed_columns: SortedSet::from_iter(PREPROCESSED_COLUMNS),
        }
    }

    /// Helper function to generate a STARK proof with the given preprocessed data.
    fn generate_stark_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F> + 'static,
        const D: usize,
    >(
        stark: PAirStark<F, D>,
        config: &StarkConfig,
        preprocessed_data: Option<&PreprocessedData<F>>,
    ) -> Result<StarkProofWithPublicInputs<F, C, D>> {
        let (trace_poly_values, public_inputs_arr) = stark.generate_trace();
        let digest: &[F] = preprocessed_data
            .map(|pd| pd.digest.elements.as_slice())
            .unwrap_or(&[]);
        let mut timing = TimingTree::new("prove", log::Level::Info);
        prove::<F, C, _, D>(
            stark,
            config,
            trace_poly_values,
            &public_inputs_arr,
            None,
            &mut timing,
            digest,
        )
    }

    /// Helper function to generate the first recursion layer proof.
    ///
    /// Re-derives `zeta` from the STARK proof (same Fiat-Shamir transcript),
    /// computes the preprocessed column evaluations at `zeta` and `g*zeta`,
    /// and sets them as witnesses for the eval public inputs.
    fn prove_first_recursion_layer<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F> + 'static,
        const D: usize,
    >(
        stark: PAirStark<F, D>,
        stark_proof: &StarkProofWithPublicInputs<F, C, D>,
        config: &StarkConfig,
        recursion_config: &CircuitConfig,
        preprocessed_data: &PreprocessedData<F>,
    ) -> Result<(CircuitData<F, C, D>, ProofWithPublicInputs<F, C, D>)>
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        let degree_bits = stark_proof.proof.recover_degree_bits(config);
        let num_prep_cols = preprocessed_data.columns.len();
        let (circuit_data, pt, evals_at_zeta_targets, evals_at_g_zeta_targets) =
            build_verifier_circuit_data(
                stark,
                config,
                recursion_config,
                degree_bits,
                num_prep_cols,
            );
        let mut pw = PartialWitness::new();

        set_stark_proof_with_pis_target(&mut pw, &pt, stark_proof, degree_bits)?;

        // Re-derive zeta and g*zeta from the STARK proof's Fiat-Shamir transcript.
        let (zeta, g_zeta) = get_zeta_and_g_zeta(stark, stark_proof, config, preprocessed_data);

        // Compute evaluations and set witnesses for the eval PI targets.
        let evals_at_zeta = eval_columns_at_point::<F, D>(
            &preprocessed_data.columns.iter().collect::<Vec<_>>(),
            zeta,
            None,
        );
        let evals_at_g_zeta = eval_columns_at_point::<F, D>(
            &preprocessed_data.columns.iter().collect::<Vec<_>>(),
            g_zeta,
            None,
        );

        for (target, value) in evals_at_zeta_targets.iter().zip(evals_at_zeta.iter()) {
            pw.set_extension_target(*target, *value)?;
        }
        for (target, value) in evals_at_g_zeta_targets.iter().zip(evals_at_g_zeta.iter()) {
            pw.set_extension_target(*target, *value)?;
        }

        let proof = circuit_data.prove(pw)?;
        circuit_data.verify(proof.clone())?;
        Ok((circuit_data, proof))
    }

    /// Re-derives the STARK challenge `zeta` and `g*zeta` from a proof's Fiat-Shamir transcript.
    fn get_zeta_and_g_zeta<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        stark: PAirStark<F, D>,
        proof: &StarkProofWithPublicInputs<F, C, D>,
        config: &StarkConfig,
        preprocessed_data: &PreprocessedData<F>,
    ) -> (F::Extension, F::Extension) {
        let degree_bits = proof.proof.recover_degree_bits(config);
        let mut challenger = Challenger::<F, C::Hasher>::new();
        let challenges = proof.get_challenges(
            &stark,
            &mut challenger,
            None,
            None,
            false,
            config,
            None,
            Some(&preprocessed_data.digest),
        );
        let zeta: F::Extension = challenges.stark_zeta;
        let g = F::primitive_root_of_unity(degree_bits);
        // In a generic context (F: RichField + Extendable<D>, g: F), scalar_mul is
        // unambiguous because FieldExtension<D> is the only impl with BaseField = F.
        let g_zeta: F::Extension = zeta.scalar_mul(g);
        (zeta, g_zeta)
    }

    /// Computes extension-field evaluations for a set of preprocessed columns at a given point,
    /// and flattens them into a Vec<F> for use as public inputs.
    fn compute_evals_flat<F: RichField + Extendable<D>, const D: usize>(
        preprocessed_data: &PreprocessedData<F>,
        point: F::Extension,
    ) -> Vec<F> {
        eval_columns_at_point::<F, D>(
            &preprocessed_data.columns.iter().collect::<Vec<_>>(),
            point,
            None,
        )
        .into_iter()
        .flat_map(|e| e.to_basefield_array().to_vec())
        .collect()
    }

    /// Builds the public inputs vector with mismatched preprocessed column evaluations.
    ///
    /// The returned `Vec<F>` contains the STARK public inputs followed by wrong eval claims
    /// (computed from a mismatch preprocessing), suitable for constructing a proof that
    /// should fail verification.
    fn build_mismatched_pis<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        stark_correct: PAirStark<F, D>,
        stark_proof: &StarkProofWithPublicInputs<F, C, D>,
        config: &StarkConfig,
        preprocessed_data_correct: &PreprocessedData<F>,
    ) -> Vec<F> {
        let (zeta, g_zeta) = get_zeta_and_g_zeta(
            stark_correct,
            stark_proof,
            config,
            preprocessed_data_correct,
        );
        let preprocessed_data_mismatch = stark_correct.with_mismatch().make_preprocessed_data();
        let (_, trace_pis) = stark_correct.generate_trace();
        let mut pis = trace_pis.to_vec();
        pis.extend(compute_evals_flat::<F, D>(
            &preprocessed_data_mismatch,
            zeta,
        ));
        pis.extend(compute_evals_flat::<F, D>(
            &preprocessed_data_mismatch,
            g_zeta,
        ));
        pis
    }

    #[test]
    fn test_pair_stark_preprocessed_correct() -> Result<()> {
        const D: usize = 2;
        type C = KeccakGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let stark = PAirStark::<F, D>::new(1 << LOG2_ROWS);
        let config = stark_config();
        let preprocessed_data = stark.make_preprocessed_data();
        let proof = generate_stark_proof::<F, C, D>(stark, &config, Some(&preprocessed_data))?;
        verify_stark_proof(stark, proof, &config, None, Some(&preprocessed_data))?;
        Ok(())
    }

    #[test]
    fn test_pair_stark_preprocessed_incorrect() {
        const D: usize = 2;
        type C = KeccakGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let stark_correct = PAirStark::<F, D>::new(1 << LOG2_ROWS);
        let config = stark_config();
        let preprocessed_data_correct = stark_correct.make_preprocessed_data();

        // Generate proof with correct preprocessed columns.
        let proof = generate_stark_proof::<F, C, D>(
            stark_correct,
            &config,
            Some(&preprocessed_data_correct),
        )
        .unwrap();

        // Verify with correct preprocessed data should succeed.
        verify_stark_proof(
            stark_correct,
            proof.clone(),
            &config,
            None,
            Some(&preprocessed_data_correct),
        )
        .expect("Verification with correct preprocessed columns should succeed");

        // Create stark with mismatched preprocessed columns and build PreprocessedData for it.
        // The digest is still ZERO (same FS transcript), but the column values differ.
        let stark_mismatch = PAirStark::<F, D>::new(1 << LOG2_ROWS).with_mismatch();
        let preprocessed_data_mismatch = stark_mismatch.make_preprocessed_data();
        let (_, pi) = stark_mismatch.generate_trace();
        let proof_with_mismatch_pis = StarkProofWithPublicInputs::<_, _, _> {
            proof: proof.proof,
            public_inputs: pi.to_vec(),
        };

        // Verification should fail: the eval check will detect the column mismatch.
        verify_stark_proof(
            stark_mismatch,
            proof_with_mismatch_pis,
            &config,
            None,
            Some(&preprocessed_data_mismatch),
        )
        .expect_err("Verification with mismatched preprocessed columns should fail!");
    }

    #[test]
    fn test_pair_stark_preprocessed_recursive_correct() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = stark_config();
        let stark = PAirStark::<F, D>::new(1 << LOG2_ROWS);
        let preprocessed_data = stark.make_preprocessed_data();
        let proof = generate_stark_proof::<F, C, D>(stark, &config, Some(&preprocessed_data))?;
        verify_stark_proof(
            stark,
            proof.clone(),
            &config,
            None,
            Some(&preprocessed_data),
        )?;

        let recursion_config = CircuitConfig::standard_recursion_config();

        let start_time = Instant::now();

        let (circuit_data, proof_1) = prove_first_recursion_layer(
            stark,
            &proof,
            &config,
            &recursion_config,
            &preprocessed_data,
        )?;

        let elapsed = start_time.elapsed();
        info!("Recursive proof build+prove time: {elapsed:.3?}");

        // Serialize and deserialize proof (public inputs known separately)
        let mut proof_bytes = Vec::new();
        proof_bytes
            .write_proof(&proof_1.proof, &[])
            .expect("write to vec");
        info!("Recursive proof size: {} KB", proof_bytes.len() / 1024);

        let proof = Buffer::new(&proof_bytes)
            .read_proof(&circuit_data.common, &[])
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        circuit_data.verifier_data().verify(ProofWithPublicInputs {
            proof,
            public_inputs: proof_1.public_inputs.clone(),
        })?;
        Ok(())
    }

    #[test]
    fn test_pair_stark_preprocessed_recursive_incorrect() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = stark_config();
        let stark_correct = PAirStark::<F, D>::new(1 << LOG2_ROWS);
        let preprocessed_data_correct = stark_correct.make_preprocessed_data();

        // Step 1: Generate proof with correct preprocessed columns.
        let proof_correct = generate_stark_proof::<F, C, D>(
            stark_correct,
            &config,
            Some(&preprocessed_data_correct),
        )
        .unwrap();

        // Step 2: Generate the first recursion layer with correct preprocessing.
        let recursion_config = CircuitConfig::standard_recursion_config();
        let (circuit_data, recursive_proof) = prove_first_recursion_layer(
            stark_correct,
            &proof_correct,
            &config,
            &recursion_config,
            &preprocessed_data_correct,
        )
        .unwrap();

        circuit_data
            .verify(recursive_proof.clone())
            .expect("Verification with correct preprocessed columns should succeed");

        // Step 3: Build proof with wrong eval PIs (correct STARK proof, but wrong eval claims).
        let wrong_pis = build_mismatched_pis(
            stark_correct,
            &proof_correct,
            &config,
            &preprocessed_data_correct,
        );
        let proof_with_wrong_pis = ProofWithPublicInputs::<F, C, D> {
            proof: recursive_proof.proof,
            public_inputs: wrong_pis,
        };

        // Step 4: Verification should fail.
        circuit_data
            .verify(proof_with_wrong_pis)
            .expect_err("Verification with mismatched preprocessed columns should fail!");
    }

    #[test]
    fn test_pair_stark_preprocessed_recursive_squared_correct() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        // Step 1: Generate STARK proof.
        let config = stark_config();
        let stark = PAirStark::<F, D>::new(1 << LOG2_ROWS);
        let preprocessed_data = stark.make_preprocessed_data();
        let stark_proof =
            generate_stark_proof::<F, C, D>(stark, &config, Some(&preprocessed_data))?;
        verify_stark_proof(
            stark,
            stark_proof.clone(),
            &config,
            None,
            Some(&preprocessed_data),
        )?;

        // Step 2: First recursion layer.
        let recursion_config_1 = CircuitConfig::standard_recursion_config();
        let (circuit_data_1, proof_1) = prove_first_recursion_layer(
            stark,
            &stark_proof,
            &config,
            &recursion_config_1,
            &preprocessed_data,
        )?;

        // Step 3: Second recursion layer.
        let recursion_config_2 = CircuitConfig::standard_recursion_config();
        let (proof_2, circuit_data_2) = recursive_prove::<C>(
            &recursion_config_2,
            &recursion_config_1,
            &circuit_data_1,
            &proof_1,
        )?;

        // Step 4: Verify the final proof.
        circuit_data_2.verify(proof_2.clone())?;

        Ok(())
    }

    #[test]
    fn test_pair_stark_preprocessed_recursive_squared_incorrect() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        // Step 1: Generate STARK proof with correct preprocessed columns.
        let config = stark_config();
        let stark_correct = PAirStark::<F, D>::new(1 << LOG2_ROWS);
        let preprocessed_data_correct = stark_correct.make_preprocessed_data();
        let stark_proof = generate_stark_proof::<F, C, D>(
            stark_correct,
            &config,
            Some(&preprocessed_data_correct),
        )
        .unwrap();

        // Step 2: First recursion layer with correct preprocessing.
        let recursion_config_1 = CircuitConfig::standard_recursion_config();
        let (circuit_data_1, proof_1) = prove_first_recursion_layer(
            stark_correct,
            &stark_proof,
            &config,
            &recursion_config_1,
            &preprocessed_data_correct,
        )
        .unwrap();

        // Step 3: Second recursion layer with correct witness.
        let recursion_config_2 = CircuitConfig::standard_recursion_config();
        let (proof_2, circuit_data_2) = recursive_prove::<C>(
            &recursion_config_2,
            &recursion_config_1,
            &circuit_data_1,
            &proof_1,
        )
        .unwrap();
        circuit_data_2.verify(proof_2.clone()).unwrap();

        // Step 4: Build a second-level proof with wrong eval public inputs; verification should fail.
        let wrong_pis = build_mismatched_pis(
            stark_correct,
            &stark_proof,
            &config,
            &preprocessed_data_correct,
        );
        let proof_2_mismatch = ProofWithPublicInputs::<F, C, D> {
            proof: proof_2.proof,
            public_inputs: wrong_pis,
        };

        // Step 5: Verification should fail.
        circuit_data_2
            .verify(proof_2_mismatch)
            .expect_err("Verification with mismatched preprocessed columns should fail!");
    }
}

// To run the tests correctly:
//    RUSTFLAGS="-C target-cpu=native" cargo test --release -p starky --test pair_stark -- --nocapture
// Specific run:
//    RUSTFLAGS="-C target-cpu=native" cargo test --release -p starky --test pair_stark test_pair_stark_preprocessed_correct -- --nocapture
