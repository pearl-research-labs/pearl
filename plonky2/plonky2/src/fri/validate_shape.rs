#[cfg(not(feature = "std"))]
use alloc::vec;

use anyhow::ensure;

use crate::field::extension::Extendable;
use crate::fri::proof::{FriProof, FriQueryRound, FriQueryStep};
use crate::fri::structure::FriInstanceInfo;
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::plonk::config::GenericConfig;
use crate::plonk::plonk_common::salt_size;

pub(crate) fn validate_fri_proof_shape<F, C, const D: usize>(
    proof: &FriProof<F, C::Hasher, D>,
    instance: &FriInstanceInfo<F, D>,
    params: &FriParams,
    oracles_to_skip: &[usize],
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    validate_batch_fri_proof_shape::<F, C, D>(
        proof,
        core::slice::from_ref(instance),
        &[params.degree_bits],
        params,
        oracles_to_skip,
    )
}

/// Validates the shape of a (possibly batched) FRI proof.
///
/// `degree_bits` contains the degree (in bits, without the rate) of each FRI instance, sorted
/// in descending order with no duplicates. `degree_bits[0]` must match `params.degree_bits`.
/// The expected initial Merkle proof length of each oracle is derived from the tallest
/// instance in which that oracle holds polynomials.
pub(crate) fn validate_batch_fri_proof_shape<F, C, const D: usize>(
    proof: &FriProof<F, C::Hasher, D>,
    instances: &[FriInstanceInfo<F, D>],
    degree_bits: &[usize],
    params: &FriParams,
    oracles_to_skip: &[usize],
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let FriProof {
        commit_phase_merkle_caps,
        query_round_proofs,
        final_poly,
        pow_witness: _pow_witness,
    } = proof;

    ensure!(degree_bits.len() == instances.len());
    ensure!(degree_bits[0] == params.degree_bits);
    ensure!(degree_bits.windows(2).all(|pair| pair[0] > pair[1]));
    ensure!(
        commit_phase_merkle_caps.len() == params.reduction_arity_bits.len(),
        "FRI commit-phase cap count must match the reduction schedule"
    );

    let cap_height = params.config.cap_height;
    for cap in commit_phase_merkle_caps {
        ensure!(cap.height() == cap_height);
    }

    for query_round in query_round_proofs {
        let FriQueryRound {
            initial_trees_proof,
            steps,
        } = query_round;

        let oracle_count = initial_trees_proof.evals_proofs.len();
        let mut leaf_len = vec![0; oracle_count];
        // The tallest degree at which each oracle holds polynomials.
        let mut oracle_degree_bits = vec![None; oracle_count];
        for (inst, &db) in instances.iter().zip(degree_bits) {
            ensure!(oracle_count == inst.oracles.len());
            for (i, oracle) in inst.oracles.iter().enumerate() {
                leaf_len[i] += oracle.num_polys + salt_size(oracle.blinding && params.hiding);
                if oracle.num_polys > 0 && oracle_degree_bits[i].is_none() {
                    oracle_degree_bits[i] = Some(db);
                }
            }
        }
        for (i, (leaf, merkle_proof)) in initial_trees_proof.evals_proofs.iter().enumerate() {
            if oracles_to_skip.contains(&i) {
                continue;
            }
            ensure!(leaf.len() == leaf_len[i]);
            let oracle_degree_bits = oracle_degree_bits[i]
                .ok_or_else(|| anyhow::anyhow!("Oracle {i} holds no polynomials"))?;
            ensure!(
                merkle_proof.len() + cap_height == oracle_degree_bits + params.config.rate_bits
            );
        }

        ensure!(steps.len() == params.reduction_arity_bits.len());
        let mut codeword_len_bits = params.lde_bits();
        for (step, arity_bits) in steps.iter().zip(&params.reduction_arity_bits) {
            let FriQueryStep {
                evals,
                merkle_proof,
            } = step;

            let arity = 1 << arity_bits;
            codeword_len_bits -= arity_bits;

            ensure!(evals.len() == arity);
            ensure!(merkle_proof.len() + cap_height == codeword_len_bits);
        }
    }

    ensure!(final_poly.len() == params.final_poly_len());

    Ok(())
}
