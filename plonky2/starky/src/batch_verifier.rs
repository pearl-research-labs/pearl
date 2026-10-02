//! Verifier for batched multi-STARK proofs.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{bail, ensure, Result};
use hashbrown::HashMap;
use plonky2::batch_fri::verifier::verify_batch_fri_proof;
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, RichField};
use plonky2::iop::challenger::Challenger;
use plonky2::plonk::config::{GenericConfig, Hasher};
use plonky2::plonk::plonk_common::reduce_with_powers;

use crate::batch_proof::BatchStarkProofWithPublicInputs;
use crate::batch_prover::BatchStarkPreprocessedVerifierData;
use crate::batch_stark::{BatchOracle, BatchStark, BatchStarkLayout};
use crate::config::StarkConfig;
use crate::cross_table_lookup::{verify_cross_table_lookups, CrossTableLookup, CtlCheckVars};
use crate::get_challenges::get_dummy_polys;
use crate::lookup::get_grand_product_challenge_set;
use crate::proof::StarkOpeningSet;
use crate::verifier::eval_columns_at_zeta_and_next;

/// Columns whose values are fully known to the verifier ("known columns").
///
/// Unlike the *preprocessed* columns of
/// [`BatchStarkPreprocessedData`][crate::batch_prover::BatchStarkPreprocessedData],
/// known columns are not committed at setup time: the prover commits to them
/// as ordinary online trace columns, and the verifier recomputes their
/// openings at the challenge points from the known values and checks them
/// against the claimed openings. This is the batch analogue of the
/// single-STARK [`PreprocessedData`][crate::verifier::PreprocessedData]
/// scheme.
#[derive(Debug, Clone)]
pub struct BatchKnownColumns<F: RichField> {
    /// Opaque, collision-resistant identifier for the known columns, absorbed
    /// by the Fiat-Shamir challenger (`None` = nothing absorbed). Must match
    /// between the prover and the verifier.
    pub digest: Option<HashOut<F>>,
    /// For each table, the sorted indices of its known columns within the
    /// table's trace.
    pub columns_per_table: Vec<Vec<usize>>,
    /// For each table, the values of its known columns (each of the table's
    /// trace length), in `columns_per_table` order.
    pub values_per_table: Vec<Vec<PolynomialValues<F>>>,
}

impl<F: RichField> BatchKnownColumns<F> {
    /// Binds the known-column indices and digest into the Fiat-Shamir
    /// transcript. Must mirror `BatchKnownColumnsTarget::observe` in
    /// `batch_recursive_verifier.rs`.
    pub(crate) fn observe<H: Hasher<F>>(&self, challenger: &mut Challenger<F, H>) {
        for columns in &self.columns_per_table {
            // Length-prefix each table's indices for domain separation.
            challenger.observe_element(F::from_canonical_usize(columns.len()));
            for &c in columns {
                challenger.observe_element(F::from_canonical_usize(c));
            }
        }
        if let Some(digest) = &self.digest {
            challenger.observe_elements(&digest.elements);
        }
    }

    /// Checks this data against the batch shape: per-table lengths, sorted
    /// in-range column indices, values of the tables' trace lengths, and
    /// disjointness from the setup-time preprocessed columns.
    pub(crate) fn validate(
        &self,
        num_columns: &[usize],
        degree_bits: &[usize],
        prep_columns: &[Vec<usize>],
    ) -> Result<()> {
        let n = num_columns.len();
        ensure!(
            self.columns_per_table.len() == n,
            "Known columns: table count mismatch"
        );
        ensure!(
            self.values_per_table.len() == n,
            "Known-column values: table count mismatch"
        );
        for t in 0..n {
            let columns = &self.columns_per_table[t];
            let values = &self.values_per_table[t];
            ensure!(
                columns.len() == values.len(),
                "Table {t}: known column/value count mismatch"
            );
            ensure!(
                columns.windows(2).all(|w| w[0] < w[1])
                    && columns.last().is_none_or(|&c| c < num_columns[t]),
                "Table {t}: known column indices must be sorted, distinct and in range"
            );
            ensure!(
                columns.iter().all(|c| !prep_columns[t].contains(c)),
                "Table {t}: a column cannot be both preprocessed and known"
            );
            ensure!(
                values.iter().all(|v| v.len() == 1 << degree_bits[t]),
                "Table {t}: known-column values must have the table's trace length"
            );
        }
        Ok(())
    }
}

/// Verifies a batched multi-STARK proof:
/// - the constraints (including lookups and CTLs) of every table are checked
///   at the challenge point against the quotient openings;
/// - the cross-table lookups are checked against each other;
/// - all openings are checked with a single batched FRI argument, against
///   the proof's trace/auxiliary/quotient caps (one per solo table plus one
///   per grouped role) and the single setup-time preprocessed cap.
///
/// If `known_columns` is provided, its digest is absorbed by the Fiat-Shamir
/// challenger and the openings of each known column at `zeta` and `g_t *
/// zeta` are recomputed from the known values and checked against the claimed
/// openings, binding the proof to the known data. It must be `Some` iff the
/// prover was given the same data.
///
/// `grouped_tables` must equal the list given to the prover (see
/// [`batch_prove`][crate::batch_prover::batch_prove]).
///
/// The keys of `ctl_extra_looking_sums` are positions within
/// `cross_table_lookups` of CTLs whose looking sums also include values not
/// associated with any table.
pub fn batch_verify<F, C, const D: usize, const N: usize>(
    starks: &[&dyn BatchStark<F, D>; N],
    config: &StarkConfig,
    proof_with_pis: &BatchStarkProofWithPublicInputs<F, C, D>,
    cross_table_lookups: &[CrossTableLookup<F>],
    grouped_tables: &[usize],
    preprocessed: Option<&BatchStarkPreprocessedVerifierData<F, C, D>>,
    known_columns: Option<&BatchKnownColumns<F>>,
    ctl_extra_looking_sums: &HashMap<usize, Vec<F>>,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let proof = &proof_with_pis.proof;
    let public_inputs = &proof_with_pis.public_inputs;
    ensure!(proof.openings.len() == N);
    ensure!(proof.degree_bits.len() == N);
    ensure!(public_inputs.len() == N);
    ensure!(
        config.preprocessed_columns.is_empty(),
        "In batch mode, preprocessed columns are described by \
         `BatchStarkPreprocessedVerifierData`, not by `StarkConfig::preprocessed_columns`."
    );

    let empty_prep = vec![vec![]; N];
    let prep_columns = preprocessed.map_or(&empty_prep, |p| &p.columns_per_table);
    let layout = BatchStarkLayout::new(
        starks.as_slice(),
        config,
        &proof.degree_bits,
        prep_columns,
        cross_table_lookups,
        grouped_tables,
    )?;

    validate_proof_shape(starks, &layout, proof_with_pis, preprocessed, config)?;
    if let Some(known) = known_columns {
        known.validate(&layout.num_columns, &proof.degree_bits, prep_columns)?;
    }

    // Rebuild the Fiat-Shamir transcript: public inputs, preprocessed cap,
    // known columns, config, trace caps.
    let mut challenger = Challenger::<F, C::Hasher>::new();
    for pis in public_inputs {
        challenger.observe_elements(pis);
    }
    if let Some(prep) = preprocessed {
        challenger.observe_cap(&prep.cap);
    }
    if let Some(known) = known_columns {
        known.observe(&mut challenger);
    }
    config.observe(&mut challenger);
    for cap in &proof.trace_caps {
        challenger.observe_cap(cap);
    }

    let has_aux = (0..N).any(|t| layout.num_aux_polys(t) > 0);
    let challenge_set =
        has_aux.then(|| get_grand_product_challenge_set(&mut challenger, config.num_challenges));
    if let Some(caps) = &proof.auxiliary_polys_caps {
        for cap in caps {
            challenger.observe_cap(cap);
        }
    }

    let lookup_challenges: Option<Vec<F>> = challenge_set
        .as_ref()
        .map(|set| set.challenges.iter().map(|ch| ch.beta).collect());
    let lookup_challenges_of = |t: usize| -> Option<&Vec<F>> {
        starks[t]
            .uses_lookups()
            .then(|| lookup_challenges.as_ref().unwrap())
    };

    // The real CTL check vars of every table, extracted from the openings.
    let ctl_vars_per_table: Vec<Option<Vec<CtlCheckVars<F, F::Extension, F::Extension, D>>>> = (0
        ..N)
        .map(|t| {
            starks[t].requires_ctls().then(|| {
                CtlCheckVars::from_openings(
                    t,
                    &proof.openings[t],
                    cross_table_lookups,
                    challenge_set.as_ref().unwrap(),
                    layout.num_lookup_columns[t],
                    layout.num_ctl_helpers[t],
                    &layout.num_ctl_helpers_by_ctl[t],
                )
            })
        })
        .collect();

    // Constraint-binding grind: mirror the prover's transcript.
    let alphas_prime = challenger.get_n_challenges(config.num_challenges);
    for t in 0..N {
        let pow_degree = core::cmp::max(2, starks[t].constraint_degree() + 1);
        let dummy_openings = get_dummy_polys::<F, C, D>(
            &mut challenger,
            starks[t].num_columns(),
            layout.num_aux_polys(t),
            pow_degree,
        );

        let num_lookup_columns = layout.num_lookup_columns[t];
        let total_num_ctl_helpers = layout.num_ctl_helpers[t];
        let dummy_ctl_vars = ctl_vars_per_table[t].as_ref().map(|ctl_vars| {
            let mut start_index = 0;
            ctl_vars
                .iter()
                .enumerate()
                .map(|(i, ctl_check_vars)| {
                    let num_ctl_helper_cols = ctl_check_vars.helper_columns.len();
                    let helper_columns = dummy_openings.auxiliary_polys.as_ref().unwrap()
                        [num_lookup_columns + start_index
                            ..num_lookup_columns + start_index + num_ctl_helper_cols]
                        .to_vec();
                    let ctl_vars = CtlCheckVars::<F, F::Extension, F::Extension, D> {
                        helper_columns,
                        local_z: dummy_openings.auxiliary_polys.as_ref().unwrap()
                            [num_lookup_columns + total_num_ctl_helpers + i],
                        next_z: dummy_openings.auxiliary_polys_next.as_ref().unwrap()
                            [num_lookup_columns + total_num_ctl_helpers + i],
                        challenges: ctl_check_vars.challenges,
                        columns: ctl_check_vars.columns.clone(),
                        filter: ctl_check_vars.filter.clone(),
                    };
                    start_index += num_ctl_helper_cols;
                    ctl_vars
                })
                .collect::<Vec<_>>()
        });

        let zeta_prime = challenger.get_extension_challenge::<D>();
        let constraint_evals = starks[t].eval_vanishing_poly_ext(
            &dummy_openings,
            dummy_ctl_vars.as_deref(),
            lookup_challenges_of(t),
            &public_inputs[t],
            alphas_prime.clone(),
            zeta_prime,
            proof.degree_bits[t],
            num_lookup_columns,
        );
        challenger.observe_extension_elements(&constraint_evals);
    }

    let alphas = challenger.get_n_challenges(config.num_challenges);
    if let Some(caps) = &proof.quotient_polys_caps {
        for cap in caps {
            challenger.observe_cap(cap);
        }
    }
    let zeta = challenger.get_extension_challenge::<D>();

    // Observe the openings table by table (canonical, profile-independent order).
    for os in &proof.openings {
        os.observe(&mut challenger);
    }
    let fri_openings = layout.fri_openings(&proof.openings);

    let max_degree_bits = layout.distinct_degree_bits[0];
    let fri_challenges = challenger.fri_challenges::<C, D>(
        &proof.opening_proof.commit_phase_merkle_caps,
        &proof.opening_proof.final_poly,
        proof.opening_proof.pow_witness,
        max_degree_bits,
        &config.fri_config,
        None,
        None,
    );

    // Check each table's polynomial identities `vanishing(x) = Z_H(x) quotient(x)` at zeta.
    for t in 0..N {
        if starks[t].quotient_degree_factor() == 0 {
            continue;
        }
        let vanishing_polys_zeta = starks[t].eval_vanishing_poly_ext(
            &proof.openings[t],
            ctl_vars_per_table[t].as_deref(),
            lookup_challenges_of(t),
            &public_inputs[t],
            alphas.clone(),
            zeta,
            proof.degree_bits[t],
            layout.num_lookup_columns[t],
        );

        let zeta_pow_deg = zeta.exp_power_of_2(proof.degree_bits[t]);
        let z_h_zeta = zeta_pow_deg - F::Extension::ONE;
        let quotient_polys = proof.openings[t]
            .quotient_polys
            .as_ref()
            .expect("Quotient polys should be provided");
        ensure!(
            vanishing_polys_zeta.len() * starks[t].quotient_degree_factor() == quotient_polys.len(),
            "Table {t}: vanishing/quotient polynomial count mismatch"
        );
        for (i, chunk) in quotient_polys
            .chunks(starks[t].quotient_degree_factor())
            .enumerate()
        {
            ensure!(
                vanishing_polys_zeta[i] == z_h_zeta * reduce_with_powers(chunk, zeta_pow_deg),
                "Table {t}: mismatch between evaluation and opening of quotient polynomial"
            );
        }
    }

    // Known columns: recompute the openings of each known column at zeta and
    // `g_t * zeta` from the known values and check them against the claimed
    // openings.
    if let Some(known) = known_columns {
        for t in 0..N {
            let column_refs: Vec<&PolynomialValues<F>> = known.values_per_table[t].iter().collect();
            if column_refs.is_empty() {
                continue;
            }
            let (evals_at_zeta, evals_at_g_zeta) =
                eval_columns_at_zeta_and_next::<F, D>(&column_refs, zeta, proof.degree_bits[t]);
            for (j, &c) in known.columns_per_table[t].iter().enumerate() {
                ensure!(
                    proof.openings[t].local_values[c] == evals_at_zeta[j],
                    "Table {t}: known column {c} evaluation mismatch at zeta"
                );
                ensure!(
                    proof.openings[t].next_values[c] == evals_at_g_zeta[j],
                    "Table {t}: known column {c} evaluation mismatch at g*zeta"
                );
            }
        }
    }

    // Check that the looking and looked CTL sums match across tables.
    let ctl_zs_first: [Vec<F>; N] =
        core::array::from_fn(|t| proof.openings[t].ctl_zs_first.clone().unwrap_or_default());
    verify_cross_table_lookups::<F, D, N>(
        cross_table_lookups,
        ctl_zs_first,
        ctl_extra_looking_sums,
        config,
    )?;

    // Check all the openings with a single batched FRI argument. Caps are
    // listed in `layout.oracles` order: that is the FRI oracle index space
    // (`FriPolynomialInfo.oracle_index`). Observation order (prep, then
    // trace, then aux, then quotient) is independent and already replayed
    // above.
    let instances = layout.fri_instances::<F, D>(zeta);
    let mut trace_i = 0;
    let mut aux_i = 0;
    let mut quot_i = 0;
    let mut caps = Vec::with_capacity(layout.oracles.len());
    for &oracle in &layout.oracles {
        match oracle {
            BatchOracle::Trace(_) | BatchOracle::GroupedTrace => {
                caps.push(proof.trace_caps[trace_i].clone());
                trace_i += 1;
            }
            BatchOracle::Preprocessed => {
                caps.push(preprocessed.expect("missing preprocessed data").cap.clone());
            }
            BatchOracle::Auxiliary(_) | BatchOracle::GroupedAuxiliary => {
                caps.push(
                    proof
                        .auxiliary_polys_caps
                        .as_ref()
                        .expect("missing auxiliary caps")[aux_i]
                        .clone(),
                );
                aux_i += 1;
            }
            BatchOracle::Quotient(_) | BatchOracle::GroupedQuotient => {
                caps.push(
                    proof
                        .quotient_polys_caps
                        .as_ref()
                        .expect("missing quotient caps")[quot_i]
                        .clone(),
                );
                quot_i += 1;
            }
        }
    }
    verify_batch_fri_proof::<F, C, D>(
        &layout.distinct_degree_bits,
        &instances,
        &fri_openings,
        &fri_challenges,
        &caps,
        &proof.opening_proof,
        &config.fri_params(max_degree_bits),
    )?;

    Ok(())
}

/// Checks the shape of the proof against the layout: one cap per table that
/// has polynomials of that role, and per-table opening lengths.
fn validate_proof_shape<F, C, const D: usize, const N: usize>(
    starks: &[&dyn BatchStark<F, D>; N],
    layout: &BatchStarkLayout,
    proof_with_pis: &BatchStarkProofWithPublicInputs<F, C, D>,
    preprocessed: Option<&BatchStarkPreprocessedVerifierData<F, C, D>>,
    config: &StarkConfig,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let proof = &proof_with_pis.proof;
    let cap_height = config.fri_config.cap_height;

    let mut n_trace = 0;
    let mut has_prep = false;
    let mut n_aux = 0;
    let mut n_quot = 0;
    for &oracle in &layout.oracles {
        match oracle {
            BatchOracle::Trace(_) | BatchOracle::GroupedTrace => n_trace += 1,
            BatchOracle::Preprocessed => has_prep = true,
            BatchOracle::Auxiliary(_) | BatchOracle::GroupedAuxiliary => n_aux += 1,
            BatchOracle::Quotient(_) | BatchOracle::GroupedQuotient => n_quot += 1,
        }
    }
    ensure!(
        proof.trace_caps.len() == n_trace,
        "trace_caps length {} != {}",
        proof.trace_caps.len(),
        n_trace
    );
    for cap in &proof.trace_caps {
        ensure!(cap.height() == cap_height);
    }

    match (&proof.auxiliary_polys_caps, n_aux > 0) {
        (Some(caps), true) => {
            ensure!(caps.len() == n_aux);
            for cap in caps {
                ensure!(cap.height() == cap_height);
            }
        }
        (None, false) => {}
        (Some(_), false) => bail!("unexpected auxiliary_polys_caps"),
        (None, true) => bail!("missing auxiliary_polys_caps"),
    }

    match (&proof.quotient_polys_caps, n_quot > 0) {
        (Some(caps), true) => {
            ensure!(caps.len() == n_quot);
            for cap in caps {
                ensure!(cap.height() == cap_height);
            }
        }
        (None, false) => {}
        (Some(_), false) => bail!("unexpected quotient_polys_caps"),
        (None, true) => bail!("missing quotient_polys_caps"),
    }

    match (preprocessed, has_prep) {
        (Some(prep), true) => ensure!(prep.cap.height() == cap_height),
        (None, false) => {}
        (Some(_), false) => bail!("unexpected preprocessed data"),
        (None, true) => bail!("missing preprocessed cap"),
    }

    for t in 0..N {
        ensure!(proof_with_pis.public_inputs[t].len() == starks[t].num_public_inputs());
        let StarkOpeningSet {
            local_values,
            next_values,
            auxiliary_polys,
            auxiliary_polys_next,
            ctl_zs_first,
            quotient_polys,
        } = &proof.openings[t];
        ensure!(local_values.len() == layout.num_columns[t]);
        ensure!(next_values.len() == layout.num_columns[t]);
        let num_aux = layout.num_aux_polys(t);
        ensure!(auxiliary_polys.as_ref().map_or(0, |v| v.len()) == num_aux);
        ensure!(auxiliary_polys_next.as_ref().map_or(0, |v| v.len()) == num_aux);
        ensure!(auxiliary_polys.is_some() == (num_aux > 0));
        ensure!(auxiliary_polys_next.is_some() == (num_aux > 0));
        ensure!(ctl_zs_first.as_ref().map_or(0, |v| v.len()) == layout.num_ctl_zs[t]);
        ensure!(ctl_zs_first.is_some() == (layout.num_ctl_zs[t] > 0));
        ensure!(quotient_polys.as_ref().map_or(0, |v| v.len()) == layout.num_quotient_polys[t]);
        ensure!(quotient_polys.is_some() == (layout.num_quotient_polys[t] > 0));
    }

    Ok(())
}
