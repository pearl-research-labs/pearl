#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use anyhow::ensure;
use itertools::Itertools;
use plonky2_field::extension::{flatten, Extendable, FieldExtension};
use plonky2_field::types::Field;

use crate::batch_fri::{batch_alpha_runs, AlphaRun};
use crate::fri::proof::{FriChallenges, FriInitialTreeProof, FriProof, FriQueryRound};
use crate::fri::structure::{FriInstanceInfo, FriOpenings};
use crate::fri::validate_shape::validate_batch_fri_proof_shape;
use crate::fri::verifier::{compute_evaluation, fri_verify_proof_of_work};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::{verify_batch_merkle_proof_to_cap, verify_merkle_proof_to_cap};
use crate::hash::merkle_tree::MerkleCap;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::util::reducing::ReducingFactor;
use crate::util::reverse_bits;

/// Verifies a batch FRI proof whose initial oracles each hold one raw leaf group per
/// instance, in instance order — the layout [`BatchFriOracle`] commits to.
///
/// `degree_bits` contains the degree (in bits, without the rate) of each FRI instance, sorted
/// in descending order with no duplicates. `instances[i]`, `openings[i]` describe the
/// polynomials of degree `degree_bits[i]` and their claimed openings.
///
/// [`BatchFriOracle`]: crate::batch_fri::oracle::BatchFriOracle
pub fn verify_batch_fri_proof<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    degree_bits: &[usize],
    instances: &[FriInstanceInfo<F, D>],
    openings: &[FriOpenings<F, D>],
    challenges: &FriChallenges<F, D>,
    initial_merkle_caps: &[MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    params: &FriParams,
) -> anyhow::Result<()> {
    verify_batch_fri_proof_with_leaves::<F, C, D>(
        degree_bits,
        instances,
        openings,
        challenges,
        initial_merkle_caps,
        &|oracle, evals| {
            let mut start = 0;
            let mut leaves = Vec::new();
            let mut heights = Vec::new();
            for (inst, &db) in instances.iter().zip(degree_bits) {
                let len = inst.oracles[oracle].num_polys;
                if len > 0 {
                    leaves.push(evals[start..start + len].to_vec());
                    heights.push(db + params.config.rate_bits);
                    start += len;
                }
            }
            (leaves, heights)
        },
        proof,
        params,
    )
}

/// Verifies a batch FRI proof using `initial_leaves(oracle, evals)` to reconstruct
/// the committed leaf groups and their LDE heights. The callback is verifier-owned:
/// it must bind every evaluation once in the oracle's prescribed tree order.
/// FRI combination continues to read the flat evaluations in polynomial order.
pub fn verify_batch_fri_proof_with_leaves<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    degree_bits: &[usize],
    instances: &[FriInstanceInfo<F, D>],
    openings: &[FriOpenings<F, D>],
    challenges: &FriChallenges<F, D>,
    initial_merkle_caps: &[MerkleCap<F, C::Hasher>],
    initial_leaves: &impl Fn(usize, &[F]) -> (Vec<Vec<F>>, Vec<usize>),
    proof: &FriProof<F, C::Hasher, D>,
    params: &FriParams,
) -> anyhow::Result<()> {
    ensure!(!params.hiding, "Batch FRI does not support hiding mode.");
    validate_batch_fri_proof_shape::<F, C, D>(proof, instances, degree_bits, params, &[])?;

    // Check PoW.
    fri_verify_proof_of_work(challenges.fri_pow_response, &params.config)?;

    // Check that parameters are coherent.
    ensure!(
        params.config.num_query_rounds == proof.query_round_proofs.len(),
        "Number of query rounds does not match config."
    );

    // Canonical alpha runs, with each run's claimed openings reduced by its
    // local alpha powers and its global alpha power, all runs-aligned.
    let runs = batch_alpha_runs(instances);
    let reduced_openings = runs
        .iter()
        .map(|run| {
            let values = &openings[run.instance].batches[run.batch].values
                [run.flat_start..run.flat_start + run.len];
            ReducingFactor::new(challenges.fri_alpha).reduce(values.iter())
        })
        .collect_vec();
    let alpha_pows = runs
        .iter()
        .map(|run| challenges.fri_alpha.exp_u64(run.alpha_offset as u64))
        .collect_vec();

    // From here on, work in the LDE domain: oracle heights are `degree + rate` bits.
    let lde_bits = degree_bits
        .iter()
        .map(|d| d + params.config.rate_bits)
        .collect_vec();
    for (&x_index, round_proof) in challenges
        .fri_query_indices
        .iter()
        .zip(&proof.query_round_proofs)
    {
        batch_fri_verifier_query_round::<F, C, D>(
            &lde_bits,
            instances,
            challenges,
            &runs,
            &reduced_openings,
            &alpha_pows,
            initial_merkle_caps,
            initial_leaves,
            proof,
            x_index,
            round_proof,
            params,
        )?;
    }

    Ok(())
}

fn batch_fri_verify_initial_proof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>(
    global_lde_bits: usize,
    initial_leaves: &impl Fn(usize, &[F]) -> (Vec<Vec<F>>, Vec<usize>),
    x_index: usize,
    proof: &FriInitialTreeProof<F, H>,
    initial_merkle_caps: &[MerkleCap<F, H>],
) -> anyhow::Result<()> {
    ensure!(
        proof.evals_proofs.len() == initial_merkle_caps.len(),
        "Initial Merkle cap count does not match the oracles."
    );
    for (oracle_index, ((evals, merkle_proof), cap)) in proof
        .evals_proofs
        .iter()
        .zip(initial_merkle_caps)
        .enumerate()
    {
        let (leaves, heights) = initial_leaves(oracle_index, evals);
        ensure!(!leaves.is_empty() && leaves.len() == heights.len());
        ensure!(heights.windows(2).all(|w| w[0] >= w[1]));
        ensure!(heights[0] <= global_lde_bits);
        ensure!(heights[0] == merkle_proof.siblings.len() + cap.height());
        // Oracles whose tallest group is smaller than the tallest instance are opened at the
        // correspondingly shifted index.
        let oracle_x_index = x_index >> (global_lde_bits - heights[0]);

        verify_batch_merkle_proof_to_cap::<F, H>(
            &leaves,
            &heights,
            oracle_x_index,
            cap,
            merkle_proof,
        )?;
    }

    Ok(())
}

fn batch_fri_combine_initial<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    instances: &[FriInstanceInfo<F, D>],
    index: usize,
    proof: &FriInitialTreeProof<F, C::Hasher>,
    alpha: F::Extension,
    subgroup_x: F,
    runs: &[AlphaRun],
    reduced_openings: &[F::Extension],
    alpha_pows: &[F::Extension],
    params: &FriParams,
) -> F::Extension {
    assert!(D > 1, "Not implemented for D=1.");
    let subgroup_x = F::Extension::from_basefield(subgroup_x);
    let mut sum = F::Extension::ZERO;

    // `Σ_batches Σ_runs alpha^offset(run) · (Σ_j alpha^j (ev_j − op_j)) / (x − z_batch)`,
    // per [`AlphaRun`]. The denominator is shared within a batch.
    for (k, batch) in instances[index].batches.iter().enumerate() {
        let mut batch_sum = F::Extension::ZERO;
        for (r, run) in runs
            .iter()
            .enumerate()
            .filter(|(_, run)| run.instance == index && run.batch == k)
        {
            let evals = batch.polynomials[run.flat_start..run.flat_start + run.len]
                .iter()
                .map(|p| {
                    let poly_blinding = instances[index].oracles[p.oracle_index].blinding;
                    let salted = params.hiding && poly_blinding;
                    proof.unsalted_eval(p.oracle_index, p.polynomial_index, salted)
                })
                .map(F::Extension::from_basefield);
            let reduced_evals = ReducingFactor::new(alpha).reduce(evals);
            batch_sum += alpha_pows[r] * (reduced_evals - reduced_openings[r]);
        }
        let denominator = subgroup_x - batch.point;
        sum += batch_sum / denominator;
    }

    sum
}

fn batch_fri_verifier_query_round<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    lde_bits: &[usize],
    instances: &[FriInstanceInfo<F, D>],
    challenges: &FriChallenges<F, D>,
    runs: &[AlphaRun],
    reduced_openings: &[F::Extension],
    alpha_pows: &[F::Extension],
    initial_merkle_caps: &[MerkleCap<F, C::Hasher>],
    initial_leaves: &impl Fn(usize, &[F]) -> (Vec<Vec<F>>, Vec<usize>),
    proof: &FriProof<F, C::Hasher, D>,
    mut x_index: usize,
    round_proof: &FriQueryRound<F, C::Hasher, D>,
    params: &FriParams,
) -> anyhow::Result<()> {
    batch_fri_verify_initial_proof::<F, C::Hasher, D>(
        lde_bits[0],
        initial_leaves,
        x_index,
        &round_proof.initial_trees_proof,
        initial_merkle_caps,
    )?;
    let mut n = lde_bits[0];
    // `subgroup_x` is `subgroup[x_index]`, i.e., the actual field element in the domain.
    let mut subgroup_x = F::MULTIPLICATIVE_GROUP_GENERATOR
        * F::primitive_root_of_unity(n).exp_u64(reverse_bits(x_index, n) as u64);

    let mut batch_index = 0;
    // old_eval is the last derived evaluation; it will be checked for consistency with its
    // committed "parent" value in the next iteration.
    let mut old_eval = batch_fri_combine_initial::<F, C, D>(
        instances,
        batch_index,
        &round_proof.initial_trees_proof,
        challenges.fri_alpha,
        subgroup_x,
        runs,
        reduced_openings,
        alpha_pows,
        params,
    );
    batch_index += 1;

    for (i, &arity_bits) in params.reduction_arity_bits.iter().enumerate() {
        let arity = 1 << arity_bits;
        let evals = &round_proof.steps[i].evals;

        // Split x_index into the index of the coset x is in, and the index of x within that coset.
        let coset_index = x_index >> arity_bits;
        let x_index_within_coset = x_index & (arity - 1);

        // Check consistency with our old evaluation from the previous round.
        ensure!(evals[x_index_within_coset] == old_eval);

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
        n -= arity_bits;

        if batch_index < lde_bits.len() && n == lde_bits[batch_index] {
            let subgroup_x_init = F::MULTIPLICATIVE_GROUP_GENERATOR
                * F::primitive_root_of_unity(n).exp_u64(reverse_bits(x_index, n) as u64);
            let eval = batch_fri_combine_initial::<F, C, D>(
                instances,
                batch_index,
                &round_proof.initial_trees_proof,
                challenges.fri_alpha,
                subgroup_x_init,
                runs,
                reduced_openings,
                alpha_pows,
                params,
            );
            old_eval = old_eval * challenges.fri_betas[i] + eval;
            batch_index += 1;
        }
    }
    ensure!(
        batch_index == instances.len(),
        "Wrong number of folded instances."
    );

    // Final check of FRI. After all the reductions, we check that the final polynomial is equal
    // to the one sent by the prover.
    ensure!(
        proof.final_poly.eval(subgroup_x.into()) == old_eval,
        "Final polynomial evaluation is invalid."
    );

    Ok(())
}
