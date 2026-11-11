#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use anyhow::{ensure, Result};
use itertools::Itertools;

use crate::field::extension::{flatten, Extendable, FieldExtension};
use crate::field::interpolation::{barycentric_weights, interpolate};
use crate::field::types::Field;
use crate::fri::proof::{FriChallenges, FriInitialTreeProof, FriProof, FriQueryRound};
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo, FriOpenings, ZETA_BATCH_IDX};
use crate::fri::validate_shape::validate_fri_proof_shape;
use crate::fri::{FriConfig, FriParams};
use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
use crate::hash::merkle_tree::MerkleCap;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::util::reducing::ReducingFactor;
use crate::util::{lde_coset_point, log2_strict, reverse_bits, reverse_index_bits_in_place};

/// Computes P'(x^arity) from {P(x*g^i)}_(i=0..arity), where g is a `arity`-th root of unity
/// and P' is the FRI reduced polynomial.
pub(crate) fn compute_evaluation<F: Field + Extendable<D>, const D: usize>(
    x: F,
    x_index_within_coset: usize,
    arity_bits: usize,
    evals: &[F::Extension],
    beta: F::Extension,
) -> F::Extension {
    let arity = 1 << arity_bits;
    debug_assert_eq!(evals.len(), arity);

    let g = F::primitive_root_of_unity(arity_bits);

    // The evaluation vector needs to be reordered first.
    let mut evals = evals.to_vec();
    reverse_index_bits_in_place(&mut evals);
    let rev_x_index_within_coset = reverse_bits(x_index_within_coset, arity_bits);
    let coset_start = x * g.exp_u64((arity - rev_x_index_within_coset) as u64);
    // The answer is gotten by interpolating {(x*g^i, P(x*g^i))} and evaluating at beta.
    let points = g
        .powers()
        .map(|y| (coset_start * y).into())
        .zip(evals)
        .collect_vec();
    let barycentric_weights = barycentric_weights(&points);
    interpolate(&points, beta, &barycentric_weights)
}

pub(crate) fn fri_verify_proof_of_work<F: RichField + Extendable<D>, const D: usize>(
    fri_pow_response: F,
    config: &FriConfig,
) -> Result<()> {
    verify_proof_of_work::<F, D>(fri_pow_response, config.proof_of_work_bits)
}

pub(crate) fn verify_proof_of_work<F: RichField + Extendable<D>, const D: usize>(
    fri_pow_response: F,
    proof_of_work_bits: u32,
) -> Result<()> {
    ensure!(proof_of_work_bits < F::order().bits() as u32);
    ensure!(
        fri_pow_response.to_canonical_u64().leading_zeros()
            >= proof_of_work_bits + (64 - F::order().bits()) as u32,
        "Invalid proof of work witness."
    );

    Ok(())
}

/// Verifies a FRI proof, skipping Merkle proof verification for oracle indices in
/// `oracles_to_skip`. Use when the verifier has computed those oracle evaluations itself.
/// Pass `&[]` to skip nothing.
pub fn verify_fri_proof<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    instance: &FriInstanceInfo<F, D>,
    openings: &FriOpenings<F, D>,
    challenges: &FriChallenges<F, D>,
    initial_merkle_caps: &[MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    params: &FriParams,
    oracles_to_skip: &[usize],
) -> Result<()> {
    validate_fri_proof_shape::<F, C, D>(proof, instance, params, oracles_to_skip)?;

    // Size of the LDE domain.
    let n = params.lde_size();

    // Check PoW.
    crate::fri::batching_pow::verify_batching_pow::<F, D>(
        challenges.batching_pow_response,
        &params.config,
    )?;
    fri_verify_proof_of_work(challenges.fri_pow_response, &params.config)?;

    // Check that parameters are coherent.
    ensure!(
        params.config.num_query_rounds == proof.query_round_proofs.len(),
        "Number of query rounds does not match config."
    );

    let precomputed_reduced_evals = PrecomputedReducedOpenings::from_os_and_alpha(
        openings,
        challenges.fri_alpha,
        params.hiding,
    );
    for (&x_index, round_proof) in challenges
        .fri_query_indices
        .iter()
        .zip(&proof.query_round_proofs)
    {
        fri_verifier_query_round::<F, C, D>(
            instance,
            challenges,
            &precomputed_reduced_evals,
            initial_merkle_caps,
            proof,
            x_index,
            n,
            round_proof,
            params,
            oracles_to_skip,
        )?;
    }

    Ok(())
}

fn fri_verify_initial_proof<F: RichField, H: Hasher<F>>(
    x_index: usize,
    proof: &FriInitialTreeProof<F, H>,
    initial_merkle_caps: &[MerkleCap<F, H>],
    oracles_to_skip: &[usize],
) -> Result<()> {
    for (i, ((evals, merkle_proof), cap)) in proof
        .evals_proofs
        .iter()
        .zip_eq(initial_merkle_caps.iter())
        .enumerate()
    {
        if oracles_to_skip.contains(&i) {
            continue;
        }
        verify_merkle_proof_to_cap::<F, H>(evals.clone(), x_index, cap, merkle_proof)?;
    }

    Ok(())
}

pub(crate) fn fri_combine_initial<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    instance: &FriInstanceInfo<F, D>,
    proof: &FriInitialTreeProof<F, C::Hasher>,
    alpha: F::Extension,
    subgroup_x: F,
    precomputed_reduced_evals: &PrecomputedReducedOpenings<F, D>,
    params: &FriParams,
) -> F::Extension {
    assert!(D > 1, "Not implemented for D=1.");
    let subgroup_x = F::Extension::from_basefield(subgroup_x);
    let mut alpha = ReducingFactor::new(alpha);
    let mut sum = F::Extension::ZERO;

    // If we are in the zk case, the `R` polynomial (the last polynomials in batch `ZETA_BATCH_IDX`) is added to
    // the batch polynomial independently, without being quotiented. So the final polynomial becomes:
    // `final_poly = R(X) + sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`, where `n` is the degree
    // of the batch polynomial.
    for (idx, (batch, reduced_openings)) in instance
        .batches
        .iter()
        .zip(&precomputed_reduced_evals.reduced_openings_at_point)
        .enumerate()
    {
        let FriBatchInfo { point, polynomials } = batch;
        let has_r_poly = params.hiding && (idx == ZETA_BATCH_IDX);
        let last_poly = polynomials.len() - has_r_poly as usize;
        let evals = polynomials
            .iter()
            .map(|p| {
                let poly_blinding = instance.oracles[p.oracle_index].blinding;
                let salted = params.hiding && poly_blinding;
                proof.unsalted_eval(p.oracle_index, p.polynomial_index, salted)
            })
            .map(F::Extension::from_basefield)
            .collect_vec();
        let reduced_evals = alpha.reduce(evals[..last_poly].iter());
        let numerator = reduced_evals - *reduced_openings;
        let denominator = subgroup_x - *point;
        sum = alpha.shift(sum);
        sum += numerator / denominator;

        // If we are in the zk case, we still have to add `R(X)` to the batch.
        if has_r_poly {
            let reduced_r_eval = alpha.reduce(evals[last_poly..].iter());
            sum = alpha.shift(sum);
            sum += reduced_r_eval;
        }
    }

    sum
}

fn fri_verifier_query_round<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    instance: &FriInstanceInfo<F, D>,
    challenges: &FriChallenges<F, D>,
    precomputed_reduced_evals: &PrecomputedReducedOpenings<F, D>,
    initial_merkle_caps: &[MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    mut x_index: usize,
    n: usize,
    round_proof: &FriQueryRound<F, C::Hasher, D>,
    params: &FriParams,
    oracles_to_skip: &[usize],
) -> Result<()> {
    fri_verify_initial_proof::<F, C::Hasher>(
        x_index,
        &round_proof.initial_trees_proof,
        initial_merkle_caps,
        oracles_to_skip,
    )?;
    // `subgroup_x` is `subgroup[x_index]`, i.e., the actual field element in the domain.
    let log_n = log2_strict(n);
    let mut subgroup_x = lde_coset_point(x_index, log_n);

    // old_eval is the last derived evaluation; it will be checked for consistency with its
    // committed "parent" value in the next iteration.
    let mut old_eval = fri_combine_initial::<F, C, D>(
        instance,
        &round_proof.initial_trees_proof,
        challenges.fri_alpha,
        subgroup_x,
        precomputed_reduced_evals,
        params,
    );

    for (i, &arity_bits) in params.reduction_arity_bits.iter().enumerate() {
        let arity = 1 << arity_bits;
        let evals = &round_proof.steps[i].evals;

        // Split x_index into the index of the coset x is in, and the index of x within that coset.
        let coset_index = x_index >> arity_bits;
        let x_index_within_coset = x_index & (arity - 1);

        // Check consistency with our old evaluation from the previous round.
        ensure!(evals[x_index_within_coset] == old_eval);

        // Infer P(y) from {P(x)}_{x^arity=y}.
        old_eval = compute_evaluation(
            subgroup_x,
            x_index_within_coset,
            arity_bits,
            evals,
            challenges.fri_betas[i],
        );

        verify_merkle_proof_to_cap::<F, C::Hasher>(
            flatten(evals),
            coset_index,
            &proof.commit_phase_merkle_caps[i],
            &round_proof.steps[i].merkle_proof,
        )?;

        // Update the point x to x^arity.
        subgroup_x = subgroup_x.exp_power_of_2(arity_bits);

        x_index = coset_index;
    }

    // Final check of FRI. After all the reductions, we check that the final polynomial is equal
    // to the one sent by the prover.
    ensure!(
        proof.final_poly.eval(subgroup_x.into()) == old_eval,
        "Final polynomial evaluation is invalid."
    );

    Ok(())
}

/// For each opening point, holds the reduced (by `alpha`) evaluations of each polynomial that's
/// opened at that point.
#[derive(Clone, Debug)]
pub(crate) struct PrecomputedReducedOpenings<F: RichField + Extendable<D>, const D: usize> {
    pub reduced_openings_at_point: Vec<F::Extension>,
}

impl<F: RichField + Extendable<D>, const D: usize> PrecomputedReducedOpenings<F, D> {
    pub(crate) fn from_os_and_alpha(
        openings: &FriOpenings<F, D>,
        alpha: F::Extension,
        is_zk: bool,
    ) -> Self {
        // We commit to an extra polynomial in the case of zk: The random `R` polynomial.
        // It should not be taken into account when computing the reduced openings.
        let reduced_openings_at_point = openings
            .batches
            .iter()
            .enumerate()
            .map(|(idx, batch)| {
                let has_r_poly = is_zk && (idx == ZETA_BATCH_IDX);
                let last_values = batch.values.len() - has_r_poly as usize;
                ReducingFactor::new(alpha).reduce(batch.values[..last_values].iter())
            })
            .collect();
        Self {
            reduced_openings_at_point,
        }
    }
}
