//! Implementation of the STARK recursive verifier, i.e. where proof
//! verification if encoded in a plonky2 circuit.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::iter::once;

use anyhow::{ensure, Result};
use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::fri::witness_util::set_fri_proof_target;
use plonky2::hash::hash_types::{HashOutTarget, RichField};
use plonky2::iop::challenger::RecursiveChallenger;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::iop::witness::WitnessWrite;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};
use plonky2::util::reducing::ReducingFactorTarget;
use plonky2::with_context;

use crate::config::StarkConfig;
use crate::cross_table_lookup::CtlCheckVarsTarget;
use crate::proof::{
    StarkOpeningSetTarget, StarkProof, StarkProofChallengesTarget, StarkProofTarget,
    StarkProofWithPublicInputs, StarkProofWithPublicInputsTarget,
};
use crate::stark::Stark;
use crate::vanishing_poly::compute_eval_vanishing_poly_circuit;

/// Circuit analogue of `PreprocessedData<F>` (see `verifier.rs`).
///
/// evals must be ordered according to `StarkConfig::preprocessed_columns`.
#[derive(Debug, Clone)]
pub struct PreprocessedDataTarget<const D: usize> {
    /// Opaque digest targets absorbed by the Fiat-Shamir challenger (None = no preprocessing).
    pub digest: Option<HashOutTarget>,
    /// Expected evaluations of each preprocessed column at zeta, in `preprocessed_columns` order.
    pub evals_at_zeta: Vec<ExtensionTarget<D>>,
    /// Expected evaluations of each preprocessed column at g*zeta, in `preprocessed_columns` order.
    pub evals_at_g_zeta: Vec<ExtensionTarget<D>>,
}

/// Connects expected preprocessed column evaluations to the trace openings at `zeta` and `g*zeta`.
fn connect_preprocessed_evals_to_stark_openings<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    inner_config: &StarkConfig,
    prep: &PreprocessedDataTarget<D>,
    openings: &StarkOpeningSetTarget<D>,
) {
    let num_prep_cols = inner_config.preprocessed_columns.len();
    assert_eq!(
        prep.evals_at_zeta.len(),
        num_prep_cols,
        "evals_at_zeta length must match preprocessed_columns count"
    );
    assert_eq!(
        prep.evals_at_g_zeta.len(),
        num_prep_cols,
        "evals_at_g_zeta length must match preprocessed_columns count"
    );
    for (j, &col_idx) in inner_config.preprocessed_columns.iter().enumerate() {
        builder.connect_extension(prep.evals_at_zeta[j], openings.local_values[col_idx]);
        builder.connect_extension(prep.evals_at_g_zeta[j], openings.next_values[col_idx]);
    }
}

/// Encodes the verification of a [`StarkProofWithPublicInputsTarget`]
/// for some statement in a circuit.
///
/// If `preprocessed` is `Some`, its `digest` is fed into the Fiat-Shamir challenger and its
/// `evals_at_zeta` / `evals_at_g_zeta` are connected to the corresponding proof openings.
/// The eval slices must be in `inner_config.preprocessed_columns` order and have the same
/// length as `inner_config.preprocessed_columns`.
///
/// Returns `stark_zeta` so the caller can connect it to a public input.
pub fn verify_stark_proof_circuit<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    stark: S,
    proof_with_pis: &StarkProofWithPublicInputsTarget<D>,
    inner_config: &StarkConfig,
    min_degree_bits_to_support: Option<usize>,
    preprocessed: Option<&PreprocessedDataTarget<D>>,
) -> ExtensionTarget<D>
where
    C::Hasher: AlgebraicHasher<F>,
{
    let max_degree_bits_to_support = proof_with_pis.proof.recover_degree_bits(inner_config);

    let preprocessed_digest = preprocessed.and_then(|prep| {
        connect_preprocessed_evals_to_stark_openings(
            builder,
            inner_config,
            prep,
            &proof_with_pis.proof.openings,
        );
        prep.digest.as_ref().map(|d| d.elements.as_slice())
    });
    let mut challenger = RecursiveChallenger::<F, C::Hasher, D>::new(builder);
    let challenges = with_context!(
        builder,
        "compute challenges",
        proof_with_pis.get_challenges::<F, C, S>(
            &stark,
            builder,
            &mut challenger,
            None,
            None,
            max_degree_bits_to_support,
            false,
            inner_config,
            preprocessed_digest,
        )
    );

    verify_stark_proof_with_challenges_circuit::<F, C, S, D>(
        builder,
        &stark,
        &proof_with_pis.proof,
        &proof_with_pis.public_inputs,
        &challenges,
        None,
        inner_config,
        max_degree_bits_to_support,
        min_degree_bits_to_support,
    );

    challenges.stark_zeta // the stark_zeta target
}

/// Recursively verifies an inner STARK proof.
pub fn verify_stark_proof_with_challenges_circuit<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &S,
    proof: &StarkProofTarget<D>,
    public_inputs: &[Target],
    challenges: &StarkProofChallengesTarget<D>,
    ctl_vars: Option<&[CtlCheckVarsTarget<F, D>]>,
    inner_config: &StarkConfig,
    degree_bits: usize,
    min_degree_bits_to_support: Option<usize>,
) where
    C::Hasher: AlgebraicHasher<F>,
{
    check_lookup_options(stark, proof, challenges).unwrap();

    let one = builder.one_extension();
    let two = builder.two();

    let num_ctl_polys = ctl_vars
        .map(|v| v.iter().map(|ctl| ctl.helper_columns.len()).sum::<usize>())
        .unwrap_or_default();

    // degree_bits should be nonzero.
    let _ = builder.inverse(proof.degree_bits);

    let quotient_polys = &proof.openings.quotient_polys;
    let ctl_zs_first = &proof.openings.ctl_zs_first;

    let max_num_of_bits_in_degree = degree_bits + 1;
    let degree = builder.exp(two, proof.degree_bits, max_num_of_bits_in_degree);
    let degree_bits_vec = builder.split_le(degree, max_num_of_bits_in_degree);

    let zeta_pow_deg = builder.exp_extension_from_bits(challenges.stark_zeta, &degree_bits_vec);
    let z_h_zeta = builder.sub_extension(zeta_pow_deg, one);

    // Calculate primitive_root_of_unity(degree_bits)
    let two_adicity = builder.constant(F::from_canonical_usize(F::TWO_ADICITY));
    let two_adicity_sub_degree_bits = builder.sub(two_adicity, proof.degree_bits);
    let two_exp_two_adicity_sub_degree_bits =
        builder.exp(two, two_adicity_sub_degree_bits, F::TWO_ADICITY);
    let base = builder.constant(F::POWER_OF_TWO_GENERATOR);
    let g = builder.exp(base, two_exp_two_adicity_sub_degree_bits, F::TWO_ADICITY);

    let num_lookup_columns = stark.num_lookup_helper_columns(inner_config);
    let lookup_challenges = stark.uses_lookups().then(|| {
        challenges
            .lookup_challenge_set
            .as_ref()
            .unwrap()
            .challenges
            .iter()
            .map(|ch| ch.beta)
            .collect::<Vec<_>>()
    });

    let vanishing_polys_zeta = compute_eval_vanishing_poly_circuit(
        builder,
        stark,
        &proof.openings,
        ctl_vars,
        lookup_challenges.as_ref(),
        public_inputs,
        challenges.stark_alphas.clone(),
        challenges.stark_zeta,
        degree_bits,
        proof.degree_bits,
        num_lookup_columns,
    );

    // Check each polynomial identity, of the form `vanishing(x) = Z_H(x) quotient(x)`, at zeta.
    let mut scale = ReducingFactorTarget::new(zeta_pow_deg);
    if let Some(quotient_polys) = quotient_polys {
        for (i, chunk) in quotient_polys
            .chunks(stark.quotient_degree_factor())
            .enumerate()
        {
            let recombined_quotient = scale.reduce(chunk, builder);
            let computed_vanishing_poly = builder.mul_extension(z_h_zeta, recombined_quotient);
            builder.connect_extension(vanishing_polys_zeta[i], computed_vanishing_poly);
        }
    }

    let merkle_caps = once(proof.trace_cap.clone())
        .chain(proof.auxiliary_polys_cap.clone())
        .chain(proof.quotient_polys_cap.clone())
        .collect_vec();

    let fri_instance = stark.fri_instance_target(
        builder,
        challenges.stark_zeta,
        g,
        num_ctl_polys,
        ctl_zs_first.as_ref().map_or(0, |c| c.len()),
        inner_config,
    );

    let fri_params = inner_config.fri_params(degree_bits);

    let one = builder.one();
    let degree_sub_one = builder.sub(degree, one);
    // Used to check if we want to skip a Fri query step.
    let degree_sub_one_bits_vec = builder.split_le(degree_sub_one, degree_bits);

    let zero = builder.zero();
    if let Some(min_degree_bits_to_support) = min_degree_bits_to_support {
        builder.verify_fri_proof_with_multiple_degree_bits::<C>(
            &fri_instance,
            &proof.openings.to_fri_openings(zero),
            &challenges.fri_challenges,
            &merkle_caps,
            &proof.opening_proof,
            &fri_params,
            proof.degree_bits,
            &degree_sub_one_bits_vec,
            min_degree_bits_to_support,
        );
    } else {
        builder.verify_fri_proof::<C>(
            &fri_instance,
            &proof.openings.to_fri_openings(zero),
            &challenges.fri_challenges,
            &merkle_caps,
            &proof.opening_proof,
            &fri_params,
        );
    }
}

/// Adds a new `StarkProofWithPublicInputsTarget` to this circuit.
pub fn add_virtual_stark_proof_with_pis<
    F: RichField + Extendable<D>,
    S: Stark<F, D>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &S,
    config: &StarkConfig,
    degree_bits: usize,
    num_ctl_helper_zs: usize,
    num_ctl_zs: usize,
) -> StarkProofWithPublicInputsTarget<D> {
    let proof = add_virtual_stark_proof::<F, S, D>(
        builder,
        stark,
        config,
        degree_bits,
        num_ctl_helper_zs,
        num_ctl_zs,
    );
    let public_inputs = builder.add_virtual_targets(S::PUBLIC_INPUTS);
    StarkProofWithPublicInputsTarget {
        proof,
        public_inputs,
    }
}

/// Adds a new `StarkProofTarget` to this circuit.
pub fn add_virtual_stark_proof<F: RichField + Extendable<D>, S: Stark<F, D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &S,
    config: &StarkConfig,
    degree_bits: usize,
    num_ctl_helper_zs: usize,
    num_ctl_zs: usize,
) -> StarkProofTarget<D> {
    let fri_params = config.fri_params(degree_bits);
    let cap_height = fri_params.config.cap_height;

    // Single oracle for all trace columns (online + preprocessed)
    let mut num_leaves_per_oracle = vec![S::COLUMNS];

    // Add auxiliary polynomials oracle if needed (for lookups or CTLs)
    if stark.uses_lookups() || stark.requires_ctls() {
        num_leaves_per_oracle.push(stark.num_lookup_helper_columns(config) + num_ctl_helper_zs);
    }

    // Add quotient polynomials oracle if needed
    if stark.quotient_degree_factor() > 0 {
        num_leaves_per_oracle.push(stark.quotient_degree_factor() * config.num_challenges);
    }

    let auxiliary_polys_cap = (stark.uses_lookups() || stark.requires_ctls())
        .then(|| builder.add_virtual_cap(cap_height));

    let quotient_polys_cap =
        (stark.constraint_degree() > 0).then(|| builder.add_virtual_cap(cap_height));

    StarkProofTarget {
        trace_cap: builder.add_virtual_cap(cap_height),
        auxiliary_polys_cap,
        quotient_polys_cap,
        openings: add_virtual_stark_opening_set::<F, S, D>(
            builder,
            stark,
            num_ctl_helper_zs,
            num_ctl_zs,
            config,
        ),
        opening_proof: builder.add_virtual_fri_proof(&num_leaves_per_oracle, &fri_params),
        degree_bits: builder.add_virtual_target(),
    }
}

fn add_virtual_stark_opening_set<F: RichField + Extendable<D>, S: Stark<F, D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &S,
    num_ctl_helper_zs: usize,
    num_ctl_zs: usize,
    config: &StarkConfig,
) -> StarkOpeningSetTarget<D> {
    StarkOpeningSetTarget {
        local_values: builder.add_virtual_extension_targets(S::COLUMNS),
        next_values: builder.add_virtual_extension_targets(S::COLUMNS),
        auxiliary_polys: (stark.uses_lookups() || stark.requires_ctls()).then(|| {
            builder.add_virtual_extension_targets(
                stark.num_lookup_helper_columns(config) + num_ctl_helper_zs,
            )
        }),
        auxiliary_polys_next: (stark.uses_lookups() || stark.requires_ctls()).then(|| {
            builder.add_virtual_extension_targets(
                stark.num_lookup_helper_columns(config) + num_ctl_helper_zs,
            )
        }),
        ctl_zs_first: stark
            .requires_ctls()
            .then(|| builder.add_virtual_targets(num_ctl_zs)),
        quotient_polys: (stark.constraint_degree() > 0).then(|| {
            builder.add_virtual_extension_targets(
                stark.quotient_degree_factor() * config.num_challenges,
            )
        }),
    }
}

/// Set the targets in a `StarkProofWithPublicInputsTarget` to
/// their corresponding values in a `StarkProofWithPublicInputs`.
pub fn set_stark_proof_with_pis_target<F, C: GenericConfig<D, F = F>, W, const D: usize>(
    witness: &mut W,
    stark_proof_with_pis_target: &StarkProofWithPublicInputsTarget<D>,
    stark_proof_with_pis: &StarkProofWithPublicInputs<F, C, D>,
    pis_degree_bits: usize,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    C::Hasher: AlgebraicHasher<F>,
    W: WitnessWrite<F>,
{
    let StarkProofWithPublicInputs {
        proof,
        public_inputs,
    } = stark_proof_with_pis;
    let StarkProofWithPublicInputsTarget {
        proof: pt,
        public_inputs: pi_targets,
    } = stark_proof_with_pis_target;

    // Set public inputs
    for (&pi_t, &pi) in pi_targets.iter().zip_eq(public_inputs) {
        witness.set_target(pi_t, pi)?;
    }

    set_stark_proof_target(witness, pt, proof, pis_degree_bits)
}

/// Set the targets in a [`StarkProofTarget`] to their corresponding values in a
/// [`StarkProof`].
pub fn set_stark_proof_target<F, C: GenericConfig<D, F = F>, W, const D: usize>(
    witness: &mut W,
    proof_target: &StarkProofTarget<D>,
    proof: &StarkProof<F, C, D>,
    pis_degree_bits: usize,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    C::Hasher: AlgebraicHasher<F>,
    W: WitnessWrite<F>,
{
    witness.set_target(
        proof_target.degree_bits,
        F::from_canonical_usize(pis_degree_bits),
    )?;
    witness.set_cap_target(&proof_target.trace_cap, &proof.trace_cap)?;

    if let (Some(quotient_polys_cap_target), Some(quotient_polys_cap)) =
        (&proof_target.quotient_polys_cap, &proof.quotient_polys_cap)
    {
        witness.set_cap_target(quotient_polys_cap_target, quotient_polys_cap)?;
    }

    // Set the opening targets directly (rather than via `to_fri_openings`, whose circuit
    // version requires a zero `Target` to lift `ctl_zs_first` to extension targets).
    let ot = &proof_target.openings;
    let os = &proof.openings;
    for (&t, &v) in ot.local_values.iter().zip_eq(&os.local_values) {
        witness.set_extension_target(t, v)?;
    }
    for (&t, &v) in ot.next_values.iter().zip_eq(&os.next_values) {
        witness.set_extension_target(t, v)?;
    }
    debug_assert_eq!(ot.auxiliary_polys.is_some(), os.auxiliary_polys.is_some());
    for (&t, &v) in ot
        .auxiliary_polys
        .iter()
        .flatten()
        .zip_eq(os.auxiliary_polys.iter().flatten())
    {
        witness.set_extension_target(t, v)?;
    }
    debug_assert_eq!(
        ot.auxiliary_polys_next.is_some(),
        os.auxiliary_polys_next.is_some()
    );
    for (&t, &v) in ot
        .auxiliary_polys_next
        .iter()
        .flatten()
        .zip_eq(os.auxiliary_polys_next.iter().flatten())
    {
        witness.set_extension_target(t, v)?;
    }
    debug_assert_eq!(ot.ctl_zs_first.is_some(), os.ctl_zs_first.is_some());
    for (&t, &v) in ot
        .ctl_zs_first
        .iter()
        .flatten()
        .zip_eq(os.ctl_zs_first.iter().flatten())
    {
        witness.set_target(t, v)?;
    }
    debug_assert_eq!(ot.quotient_polys.is_some(), os.quotient_polys.is_some());
    for (&t, &v) in ot
        .quotient_polys
        .iter()
        .flatten()
        .zip_eq(os.quotient_polys.iter().flatten())
    {
        witness.set_extension_target(t, v)?;
    }

    if let (Some(auxiliary_polys_cap_target), Some(auxiliary_polys_cap)) = (
        &proof_target.auxiliary_polys_cap,
        &proof.auxiliary_polys_cap,
    ) {
        witness.set_cap_target(auxiliary_polys_cap_target, auxiliary_polys_cap)?;
    }

    set_fri_proof_target(witness, &proof_target.opening_proof, &proof.opening_proof)
}

/// Utility function to check that all lookups data wrapped in `Option`s are `Some` iff
/// the STARK uses a permutation argument.
fn check_lookup_options<F: RichField + Extendable<D>, S: Stark<F, D>, const D: usize>(
    stark: &S,
    proof: &StarkProofTarget<D>,
    challenges: &StarkProofChallengesTarget<D>,
) -> Result<()> {
    let options_is_some = [
        proof.auxiliary_polys_cap.is_some(),
        proof.openings.auxiliary_polys.is_some(),
        proof.openings.auxiliary_polys_next.is_some(),
        challenges.lookup_challenge_set.is_some(),
    ];
    ensure!(
        options_is_some
            .iter()
            .all(|&b| b == stark.uses_lookups() || stark.requires_ctls()),
        "Lookups data doesn't match with STARK configuration."
    );
    Ok(())
}
