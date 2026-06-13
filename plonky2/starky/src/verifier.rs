//! Implementation of the STARK verifier.

#[cfg(not(feature = "std"))]
use alloc::{borrow::Cow, vec::Vec};
use core::any::type_name;
use core::iter::once;
use plonky2_util::log2_strict;
#[cfg(feature = "std")]
use std::borrow::Cow;

use anyhow::{anyhow, ensure, Result};
use itertools::Itertools;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::fri::verifier::verify_fri_proof;
use plonky2::fri::FriParams;
use plonky2::hash::hash_types::{HashOut, RichField};
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::iop::challenger::Challenger;
use plonky2::plonk::config::GenericConfig;
use plonky2::plonk::plonk_common::reduce_with_powers;
use plonky2_maybe_rayon::*;

use crate::config::StarkConfig;
use crate::cross_table_lookup::CtlCheckVars;
use crate::proof::{StarkOpeningSet, StarkProof, StarkProofChallenges, StarkProofWithPublicInputs};
use crate::stark::Stark;
use crate::vanishing_poly::compute_eval_vanishing_poly;

/// Bundles the preprocessed column data used by both the prover and the native verifier.
///
/// `digest` is an opaque, collision-resistant identifier for the preprocessed columns.
/// It is absorbed into the Fiat-Shamir challenger before zeta is derived, ensuring
/// that zeta is random with respect to the preprocessed data (grinding resistance).
/// It does NOT have to be the literal hash of the column values — any unique identifier
/// that the prover and verifier agree on is acceptable.
///
/// `columns` holds the polynomial evaluations of the preprocessed columns on the trace subgroup,
/// in the same order as `StarkConfig::preprocessed_columns` (sorted by column index).
#[derive(Debug)]
pub struct PreprocessedData<F: Field> {
    /// Opaque, collision-resistant identifier for the preprocessed columns.
    pub digest: HashOut<F>,
    /// Evaluations of the preprocessed columns on the trace subgroup, in `preprocessed_columns` order.
    pub columns: Vec<PolynomialValues<F>>,
}

/// Evaluate multiple polynomials (given as evaluations on an n-th roots-of-unity subgroup)
/// at an extension-field point, using O(n) subgroup-specialized barycentric interpolation.
///
/// Formula per column: f(z) = (z^n - 1) / n * sum_i (y_i * omega^i) / (z - omega^i)
///
/// For D=2 extension field z = [a, b], the denominators (z - omega^i) have norm
/// (a - omega^i)^2 - W*b^2, which is a base-field value. The batch inversion and
/// accumulation of (s1, s2) therefore stay in the base field. The norm computation
/// is shared across all columns, and per-column accumulation is parallelized via
/// Rayon fold-reduce.
///
/// All columns must have the same length (a power of two), equal to the trace domain size.
pub fn eval_columns_at_point<F: RichField + Extendable<D>, const D: usize>(
    columns: &[&PolynomialValues<F>],
    point: F::Extension,
    omega_pows: Option<&[F]>,
) -> Vec<F::Extension> {
    // Impl specific to D=2.
    const { assert!(D == 2, "eval_columns_at_point only supports D=2") };

    if columns.is_empty() {
        return vec![];
    }
    let n = columns[0].len();
    debug_assert!(n.is_power_of_two());
    debug_assert!(columns.iter().all(|c| c.len() == n));

    let lg_n = log2_strict(n);
    let num_cols = columns.len();

    // Extract [a, b] components of the extension-field point.
    let components = point.to_basefield_array();
    let a = components[0];
    let b = components[1];
    let w = <F as Extendable<D>>::W;

    let zn_minus_one = point.exp_power_of_2(lg_n) - F::Extension::ONE;
    let n_inv = F::inverse_2exp(lg_n);

    // Precompute W*b^2 once (base field).
    let wb2 = w * b.square();

    // Pass 1: build norms denom_i = (a - omega^i)^2 - W*b^2 (base field), then batch-invert.
    // Split into chunks for parallel inversion.
    const PARALLEL_CHUNK_SIZE: usize = 1024;
    let chunk_size = PARALLEL_CHUNK_SIZE.min(n).max(1);

    let omega_pows = match omega_pows {
        Some(slice) => {
            debug_assert_eq!(slice.len(), n, "caller-provided omega_pows length mismatch");
            Cow::Borrowed(slice)
        }
        None => Cow::Owned(F::two_adic_subgroup(lg_n)),
    };

    if b == F::ZERO {
        if let Some(j) = omega_pows.iter().position(|&op| op == a) {
            return columns
                .iter()
                .map(|c| F::Extension::from(c.values[j]))
                .collect();
        }
    }

    let mut inv_norms: Vec<F> = vec![F::ZERO; n];
    inv_norms
        .par_chunks_mut(chunk_size)
        .zip(omega_pows.par_chunks(chunk_size))
        .for_each(|(inv_chunk, omega_chunk)| {
            for (slot, &op) in inv_chunk.iter_mut().zip(omega_chunk) {
                *slot = (a - op).square() - wb2;
            }
            let inverses = F::batch_multiplicative_inverse(inv_chunk);
            inv_chunk.copy_from_slice(&inverses);
        });

    // Pass 2: parallel fold-reduce over row indices; each thread accumulates (s1, s2)
    // for all columns locally, then reduce combines the per-thread totals.
    let col_slices: Vec<&[F]> = columns.iter().map(|c| c.values.as_slice()).collect();

    #[cfg(feature = "parallel")]
    let sums: Vec<(F, F)> = (0..n)
        .into_par_iter()
        .fold(
            || vec![(F::ZERO, F::ZERO); num_cols],
            |mut acc, i| {
                let op = omega_pows[i];
                let base = inv_norms[i] * op; // inv_norm_i * omega^i
                for (j, col) in col_slices.iter().enumerate() {
                    let q = base * col[i]; // inv_norm_i * y_ij * omega^i
                    acc[j].0 += q;
                    acc[j].1 += q * op; // * omega^i -> omega^{2i} term
                }
                acc
            },
        )
        .reduce(
            || vec![(F::ZERO, F::ZERO); num_cols],
            |mut va, vb| {
                for j in 0..num_cols {
                    va[j].0 += vb[j].0;
                    va[j].1 += vb[j].1;
                }
                va
            },
        );
    #[cfg(not(feature = "parallel"))]
    let sums: Vec<(F, F)> = (0..n).fold(vec![(F::ZERO, F::ZERO); num_cols], |mut acc, i| {
        let op = omega_pows[i];
        let base = inv_norms[i] * op; // inv_norm_i * omega^i
        for (j, col) in col_slices.iter().enumerate() {
            let q = base * col[i]; // inv_norm_i * y_ij * omega^i
            acc[j].0 += q;
            acc[j].1 += q * op; // * omega^i -> omega^{2i} term
        }
        acc
    });

    let scale = zn_minus_one * F::Extension::from(n_inv);

    sums.into_iter()
        .map(|(s1, s2)| {
            // Assemble extension-field result from base-field sums.
            // sum = [a*s1 - s2, -b*s1]
            let mut sum_arr = [F::ZERO; D];
            sum_arr[0] = a * s1 - s2;
            sum_arr[1] = F::NEG_ONE * b * s1;
            let sum = F::Extension::from_basefield_array(sum_arr);
            scale * sum
        })
        .collect()
}

/// Evaluate a set of columns at both `zeta` and `g * zeta`, where `g` is the primitive root of
/// unity of order `1 << degree_bits` (the trace subgroup generator).
///
/// This is the standard STARK-verifier pattern for checking preprocessed-column openings, and is
/// also useful outside the verifier when computing the expected openings of plaintext preprocessed
/// data (e.g. to build public-input encodings).  Returns `(evals_at_zeta, evals_at_g_zeta)`, each
/// in the same order as `columns`.
pub fn eval_columns_at_zeta_and_next<F: RichField + Extendable<D>, const D: usize>(
    columns: &[&PolynomialValues<F>],
    zeta: F::Extension,
    degree_bits: usize,
) -> (Vec<F::Extension>, Vec<F::Extension>) {
    let g = F::primitive_root_of_unity(degree_bits);
    let omega_pows = F::two_adic_subgroup(degree_bits);
    let evals_at_zeta = eval_columns_at_point::<F, D>(columns, zeta, Some(&omega_pows));
    let evals_at_g_zeta =
        eval_columns_at_point::<F, D>(columns, zeta.scalar_mul(g), Some(&omega_pows));
    (evals_at_zeta, evals_at_g_zeta)
}

/// Verifies a [`StarkProofWithPublicInputs`] against a STARK statement.
///
/// If `preprocessed_data` is provided:
/// - Its `digest` is absorbed by the Fiat-Shamir challenger.
/// - The native evaluations of each preprocessed column at `zeta` and `g*zeta` are computed
///   and checked against the proof openings, binding the proof to the correct preprocessed data.
pub fn verify_stark_proof<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D>,
    const D: usize,
>(
    stark: S,
    proof_with_pis: StarkProofWithPublicInputs<F, C, D>,
    config: &StarkConfig,
    verifier_circuit_fri_params: Option<FriParams>,
    preprocessed_data: Option<&PreprocessedData<F>>,
) -> Result<()> {
    let mut challenger = Challenger::<F, C::Hasher>::new();

    let challenges = proof_with_pis.get_challenges(
        &stark,
        &mut challenger,
        None,
        None,
        false,
        config,
        verifier_circuit_fri_params,
        preprocessed_data.map(|d| &d.digest),
    );

    verify_stark_proof_with_challenges(
        &stark,
        &proof_with_pis.proof,
        &challenges,
        None,
        &proof_with_pis.public_inputs,
        config,
        preprocessed_data,
    )
}

/// Evaluates each preprocessed column at `zeta` and `g*zeta` and checks that the results
/// match the corresponding openings in `proof`.  Column ordering follows
/// `config.preprocessed_columns`.
fn check_preprocessed_evals<F, C, const D: usize>(
    data: &PreprocessedData<F>,
    proof: &StarkProof<F, C, D>,
    challenges: &StarkProofChallenges<F, D>,
    config: &StarkConfig,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let column_refs: Vec<&PolynomialValues<F>> = data.columns.iter().collect();
    let (evals_at_zeta, evals_at_g_zeta) = eval_columns_at_zeta_and_next::<F, D>(
        &column_refs,
        challenges.stark_zeta,
        proof.recover_degree_bits(config),
    );

    let sorted_indices: Vec<usize> = config.preprocessed_columns.iter().cloned().collect();
    ensure!(
        sorted_indices.len() == data.columns.len(),
        "Preprocessed column count mismatch between config and PreprocessedData"
    );
    for (j, &col_idx) in sorted_indices.iter().enumerate() {
        ensure!(
            proof.openings.local_values[col_idx] == evals_at_zeta[j],
            "Preprocessed column {} evaluation mismatch at zeta",
            col_idx
        );
        ensure!(
            proof.openings.next_values[col_idx] == evals_at_g_zeta[j],
            "Preprocessed column {} evaluation mismatch at g*zeta",
            col_idx
        );
    }
    Ok(())
}

/// Verifies a [`StarkProofWithPublicInputs`] against a STARK statement,
/// with the provided [`StarkProofChallenges`].
/// It also supports optional cross-table lookups data and challenges,
/// in case this proof is part of a multi-STARK system.
pub fn verify_stark_proof_with_challenges<F, C, S, const D: usize>(
    stark: &S,
    proof: &StarkProof<F, C, D>,
    challenges: &StarkProofChallenges<F, D>,
    ctl_vars: Option<&[CtlCheckVars<F, F::Extension, F::Extension, D>]>,
    public_inputs: &[F],
    config: &StarkConfig,
    preprocessed_data: Option<&PreprocessedData<F>>,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D>,
{
    log::debug!("Checking proof: {}", type_name::<S>());

    let (num_ctl_z_polys, num_ctl_polys) = ctl_vars
        .map(|ctls| {
            (
                ctls.len(),
                ctls.iter().map(|ctl| ctl.helper_columns.len()).sum(),
            )
        })
        .unwrap_or_default();

    validate_proof_shape(
        stark,
        proof,
        public_inputs,
        config,
        num_ctl_polys,
        num_ctl_z_polys,
    )?;

    let degree_bits = proof.recover_degree_bits(config);
    let num_lookup_columns = stark.num_lookup_helper_columns(config);

    let lookup_challenges = if stark.uses_lookups() {
        Some(
            challenges
                .lookup_challenge_set
                .as_ref()
                .unwrap()
                .challenges
                .iter()
                .map(|ch| ch.beta)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let vanishing_polys_zeta = compute_eval_vanishing_poly::<F, S, D>(
        stark,
        &proof.openings,
        ctl_vars,
        lookup_challenges.as_ref(),
        &stark.lookups(),
        public_inputs,
        challenges.stark_alphas.clone(),
        challenges.stark_zeta,
        degree_bits,
        num_lookup_columns,
    );

    // Check each polynomial identity, of the form `vanishing(x) = Z_H(x) quotient(x)`, at zeta.
    let zeta_pow_deg = challenges.stark_zeta.exp_power_of_2(degree_bits);
    let z_h_zeta = zeta_pow_deg - F::Extension::ONE;
    // `quotient_polys_zeta` holds `num_challenges * quotient_degree_factor` evaluations.
    // Each chunk of `quotient_degree_factor` holds the evaluations of `t_0(zeta),...,t_{quotient_degree_factor-1}(zeta)`
    // where the "real" quotient polynomial is `t(X) = t_0(X) + t_1(X)*X^n + t_2(X)*X^{2n} + ...`.
    // So to reconstruct `t(zeta)` we can compute `reduce_with_powers(chunk, zeta^n)` for each
    // `quotient_degree_factor`-sized chunk of the original evaluations.

    if stark.quotient_degree_factor() > 0 {
        let quotient_polys = proof
            .openings
            .quotient_polys
            .as_ref()
            .expect("Quotient polys should be provided");
        ensure!(
            vanishing_polys_zeta.len() * stark.quotient_degree_factor() == quotient_polys.len(),
            "Vanishing/quotient polynomial count mismatch"
        );
        for (i, chunk) in quotient_polys
            .chunks(stark.quotient_degree_factor())
            .enumerate()
        {
            ensure!(
                vanishing_polys_zeta[i] == z_h_zeta * reduce_with_powers(chunk, zeta_pow_deg),
                "Mismatch between evaluation and opening of quotient polynomial"
            );
        }
    }

    if let Some(data) = preprocessed_data {
        check_preprocessed_evals::<F, C, D>(data, proof, challenges, config)?;
    }

    let merkle_caps = once(proof.trace_cap.clone())
        .chain(proof.auxiliary_polys_cap.clone())
        .chain(proof.quotient_polys_cap.clone())
        .collect_vec();

    let num_ctl_zs = ctl_vars
        .map(|vars| {
            vars.iter()
                .map(|ctl| ctl.helper_columns.len())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let fri_proof = &proof.opening_proof;

    // sanity check of having same number of oracles
    for query_round in fri_proof.query_round_proofs.iter() {
        ensure!(merkle_caps.len() == query_round.initial_trees_proof.evals_proofs.len());
    }

    verify_fri_proof::<F, C, D>(
        &stark.fri_instance(
            challenges.stark_zeta,
            F::primitive_root_of_unity(degree_bits),
            num_ctl_polys,
            num_ctl_zs,
            config,
        ),
        &proof.openings.to_fri_openings(),
        &challenges.fri_challenges,
        &merkle_caps,
        &proof.opening_proof,
        &config.fri_params(degree_bits),
        &[],
    )?;

    Ok(())
}

fn validate_proof_shape<F, C, S, const D: usize>(
    stark: &S,
    proof: &StarkProof<F, C, D>,
    public_inputs: &[F],
    config: &StarkConfig,
    num_ctl_helpers: usize,
    num_ctl_zs: usize,
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D>,
{
    let degree_bits = proof.recover_degree_bits(config);

    let StarkProof {
        trace_cap,
        auxiliary_polys_cap,
        quotient_polys_cap,
        openings,
        // The shape of the opening proof will be checked in the FRI verifier (see
        // validate_fri_proof_shape), so we ignore it here.
        opening_proof: _,
    } = proof;

    let StarkOpeningSet {
        local_values,
        next_values,
        auxiliary_polys,
        auxiliary_polys_next,
        ctl_zs_first,
        quotient_polys,
    } = openings;

    ensure!(public_inputs.len() == S::PUBLIC_INPUTS);

    let fri_params = config.fri_params(degree_bits);
    let cap_height = fri_params.config.cap_height;

    ensure!(trace_cap.height() == cap_height);
    if stark.constraint_degree() == 0 {
        ensure!(quotient_polys_cap.is_none());
    } else {
        ensure!(quotient_polys_cap.is_some());
        ensure!(quotient_polys_cap.as_ref().map(|q| q.height()) == Some(cap_height));
    }

    ensure!(local_values.len() == S::COLUMNS);
    ensure!(next_values.len() == S::COLUMNS);
    ensure!(if let Some(quotient_polys) = quotient_polys {
        quotient_polys.len() == stark.num_quotient_polys(config)
    } else {
        stark.num_quotient_polys(config) == 0
    });

    check_lookup_options::<F, C, S, D>(
        stark,
        auxiliary_polys_cap,
        auxiliary_polys,
        auxiliary_polys_next,
        num_ctl_helpers,
        num_ctl_zs,
        ctl_zs_first,
        config,
    )?;

    Ok(())
}

/// Utility function to check that all lookups data wrapped in `Option`s are `Some` iff
/// the STARK uses a permutation argument.
fn check_lookup_options<F, C, S, const D: usize>(
    stark: &S,
    auxiliary_polys_cap: &Option<MerkleCap<F, <C as GenericConfig<D>>::Hasher>>,
    auxiliary_polys: &Option<Vec<<F as Extendable<D>>::Extension>>,
    auxiliary_polys_next: &Option<Vec<<F as Extendable<D>>::Extension>>,
    num_ctl_helpers: usize,
    num_ctl_zs: usize,
    ctl_zs_first: &Option<Vec<F>>,
    config: &StarkConfig,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D>,
{
    if stark.uses_lookups() || stark.requires_ctls() {
        let num_auxiliary = stark.num_lookup_helper_columns(config) + num_ctl_helpers + num_ctl_zs;
        let cap_height = config.fri_config.cap_height;

        let auxiliary_polys_cap = auxiliary_polys_cap
            .as_ref()
            .ok_or_else(|| anyhow!("Missing auxiliary_polys_cap"))?;
        let auxiliary_polys = auxiliary_polys
            .as_ref()
            .ok_or_else(|| anyhow!("Missing auxiliary_polys"))?;
        let auxiliary_polys_next = auxiliary_polys_next
            .as_ref()
            .ok_or_else(|| anyhow!("Missing auxiliary_polys_next"))?;

        if let Some(ctl_zs_first) = ctl_zs_first {
            ensure!(ctl_zs_first.len() == num_ctl_zs);
        }

        ensure!(auxiliary_polys_cap.height() == cap_height);
        ensure!(auxiliary_polys.len() == num_auxiliary);
        ensure!(auxiliary_polys_next.len() == num_auxiliary);
    } else {
        ensure!(auxiliary_polys_cap.is_none());
        ensure!(auxiliary_polys.is_none());
        ensure!(auxiliary_polys_next.is_none());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use plonky2::field::extension::quadratic::QuadraticExtension;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::polynomial::PolynomialValues;
    use plonky2::field::types::{Field, Sample};

    use super::eval_columns_at_point;
    use crate::vanishing_poly::eval_l_0_and_l_last;

    #[test]
    fn test_eval_l_0_and_l_last() {
        type F = GoldilocksField;
        let log_n = 5;
        let n = 1 << log_n;

        let x = F::rand(); // challenge point
        let expected_l_first_x = PolynomialValues::selector(n, 0).ifft().eval(x);
        let expected_l_last_x = PolynomialValues::selector(n, n - 1).ifft().eval(x);

        let (l_first_x, l_last_x) = eval_l_0_and_l_last(log_n, x);
        assert_eq!(l_first_x, expected_l_first_x);
        assert_eq!(l_last_x, expected_l_last_x);
    }

    /// For random columns given as evaluations on the 2^log_n-th roots-of-unity subgroup,
    /// `eval_columns_at_point` must agree with the canonical reference: IFFT the values to
    /// coefficients and evaluate the resulting polynomial at the extension-field point.
    ///
    /// Covers a range of log_n (including the degenerate n=1 case), both `omega_pows = None`
    /// and a caller-supplied `omega_pows`, and the empty-columns edge case.
    #[test]
    fn test_eval_columns_at_point_matches_ifft_and_eval() {
        type F = GoldilocksField;
        const D: usize = 2;
        type FE = QuadraticExtension<F>;

        const NUM_COLS: usize = 3;

        for log_n in 0..20 {
            let n = 1 << log_n;

            let columns: Vec<PolynomialValues<F>> = (0..NUM_COLS)
                .map(|_| PolynomialValues::new(F::rand_vec(n)))
                .collect();
            let column_refs: Vec<&PolynomialValues<F>> = columns.iter().collect();

            let omega_pows = F::two_adic_subgroup(log_n);

            // `F::MULTIPLICATIVE_GROUP_GENERATOR` has full order `p - 1` in Goldilocks, so it is
            // never a 2^log_n-th root of unity for any log_n we exercise here.
            let base_outside_subgroup = F::MULTIPLICATIVE_GROUP_GENERATOR;
            debug_assert!(!omega_pows.contains(&base_outside_subgroup));

            let zetas: [FE; 3] = [
                FE::rand(),
                FE::from(omega_pows[n / 2]),
                FE::from(base_outside_subgroup),
            ];

            for (label, zeta) in ["random FE", "subgroup", "base-field outside subgroup"]
                .into_iter()
                .zip(zetas)
            {
                let expected: Vec<FE> = columns
                    .iter()
                    .map(|c| c.clone().ifft().to_extension::<D>().eval(zeta))
                    .collect();

                let got_unsupplied = eval_columns_at_point::<F, D>(&column_refs, zeta, None);
                assert_eq!(
                    got_unsupplied, expected,
                    "log_n={log_n}, zeta={label}, omega_pows = None"
                );

                let got_supplied =
                    eval_columns_at_point::<F, D>(&column_refs, zeta, Some(&omega_pows));
                assert_eq!(
                    got_supplied, expected,
                    "log_n={log_n}, zeta={label}, omega_pows supplied"
                );
            }
        }

        let empty_refs: Vec<&PolynomialValues<F>> = Vec::new();
        let zeta = FE::rand();
        assert!(eval_columns_at_point::<F, D>(&empty_refs, zeta, None).is_empty());
    }
}
