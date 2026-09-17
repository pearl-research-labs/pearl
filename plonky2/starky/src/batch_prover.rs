//! Prover for batched multi-STARK proofs: each solo table's trace, auxiliary
//! and quotient polynomials are committed in their own Merkle trees, the
//! *grouped* tables share one multi-height tree per role, the preprocessed
//! columns of all tables share one setup-time tree, and all tables share one
//! batched FRI argument.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{ensure, Result};
use hashbrown::HashSet;
use plonky2::batch_fri::oracle::BatchFriOracle;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::fft::FftRootTable;
use plonky2::field::packable::Packable;
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::field::types::Field;
use plonky2::field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2::hash::hash_types::RichField;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::iop::challenger::Challenger;
use plonky2::plonk::config::GenericConfig;
use plonky2::timed;
use plonky2::util::timing::TimingTree;
use plonky2::util::{log2_ceil, log2_strict, transpose};
use plonky2_maybe_rayon::*;

use crate::batch_proof::{BatchStarkProof, BatchStarkProofWithPublicInputs};
use crate::batch_stark::{BatchOracle, BatchRole, BatchStark, BatchStarkLayout};
use crate::batch_verifier::BatchKnownColumns;
use crate::config::StarkConfig;
use crate::constraint_consumer::ConstraintConsumer;
use crate::cross_table_lookup::{
    cross_table_lookup_data, get_ctl_auxiliary_polys, CrossTableLookup, CtlCheckVars, CtlData,
};
use crate::get_challenges::get_dummy_polys;
use crate::lookup::{get_grand_product_challenge_set, lookup_helper_columns, LookupCheckVars};
use crate::proof::StarkOpeningSet;

/// Batched preprocessed data, committed once at setup time in a single
/// multi-height oracle shared by every table with preprocessed columns.
#[derive(Debug)]
pub struct BatchStarkPreprocessedData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    /// The batched commitment to the preprocessed columns of all tables.
    /// Polynomials are ordered by descending trace degree, ties by
    /// table index (`BatchStarkLayout::prep_poly_start`);
    /// same-degree tables share a Merkle leaf.
    pub commitment: BatchFriOracle<F, C, D>,
    /// For each table, the sorted indices of its preprocessed columns within
    /// the table's trace.
    pub columns_per_table: Vec<Vec<usize>>,
}

/// The verifier's view of [`BatchStarkPreprocessedData`]: the Merkle cap of
/// the setup-time commitment and the per-table preprocessed column indices.
#[derive(Debug, Clone)]
pub struct BatchStarkPreprocessedVerifierData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    /// Merkle cap of the batched preprocessed commitment.
    pub cap: MerkleCap<F, C::Hasher>,
    /// For each table, the sorted indices of its preprocessed columns.
    pub columns_per_table: Vec<Vec<usize>>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    BatchStarkPreprocessedData<F, C, D>
{
    /// Commits to all tables' preprocessed columns in one oracle.
    /// `values_per_table[t]` holds the values of table `t`'s preprocessed
    /// columns (each of the table's trace length), in the order of
    /// `columns_per_table[t]`.
    pub fn new(
        mut values_per_table: Vec<Vec<PolynomialValues<F>>>,
        columns_per_table: Vec<Vec<usize>>,
        config: &StarkConfig,
        timing: &mut TimingTree,
    ) -> Self {
        assert_eq!(values_per_table.len(), columns_per_table.len());
        for (values, columns) in values_per_table.iter().zip(&columns_per_table) {
            assert_eq!(values.len(), columns.len());
            assert!(
                values.iter().all(|v| v.len() == values[0].len()),
                "A table's preprocessed columns must all have the table's trace length."
            );
        }
        assert!(
            values_per_table.iter().any(|v| !v.is_empty()),
            "Cannot commit to empty preprocessed data; pass `None` instead."
        );
        // Flatten in the canonical oracle order — descending trace degree,
        // ties by table index — matching `BatchStarkLayout::prep_poly_start`.
        let mut order: Vec<usize> = (0..values_per_table.len())
            .filter(|&t| !values_per_table[t].is_empty())
            .collect();
        order.sort_by_key(|&t| core::cmp::Reverse(values_per_table[t][0].len()));
        let mut values = Vec::new();
        for &t in &order {
            values.append(&mut values_per_table[t]);
        }
        let fft_tables: Vec<Option<&FftRootTable<F>>> = vec![None; values.len()];
        let commitment = BatchFriOracle::from_values(
            values,
            config.fri_config.rate_bits,
            false,
            config.fri_config.cap_height,
            timing,
            &fft_tables,
        );
        Self {
            commitment,
            columns_per_table,
        }
    }

    /// The Merkle cap of the setup-time commitment.
    pub fn cap(&self) -> MerkleCap<F, C::Hasher> {
        self.commitment.batch_merkle_tree.cap.clone()
    }

    /// Extracts the verifier's view of this preprocessed data.
    pub fn verifier_data(&self) -> BatchStarkPreprocessedVerifierData<F, C, D> {
        BatchStarkPreprocessedVerifierData {
            cap: self.cap(),
            columns_per_table: self.columns_per_table.clone(),
        }
    }
}

/// Proves a batch of STARK tables with a single batched FRI argument.
///
/// - `trace_poly_values[t]` holds the *full* trace of table `t` (online and
///   preprocessed columns). The traces are consumed: the online columns are
///   moved into that table's trace oracle, and only a sparse copy of the
///   lookup/CTL columns is kept.
/// - `preprocessed`, if provided, is the setup-time batched commitment to the
///   preprocessed columns; the corresponding columns of `trace_poly_values`
///   must match the committed values.
/// - `known_columns`, if provided, describes the columns whose values the
///   verifier knows in full. They are committed like any other online trace
///   column, but their indices and digest are absorbed by the Fiat-Shamir
///   challenger, and the verifier recomputes and checks their openings. Must
///   be `Some` iff the verifier is given the same data.
/// - `cross_table_lookups` describe the CTLs connecting the tables.
/// - `grouped_tables` lists the tables (sorted) that share one multi-height
///   Merkle tree per role instead of per-table trees — worthwhile for tables
///   whose heights are protocol constants, as each role then costs one
///   Merkle path per FRI query for the whole group. Part of the transcript
///   shape: the verifier must be given the same list.
///
/// The FRI reduction strategy of `config` must fold through the LDE degree of
/// every distinct table height (e.g. with `FriReductionStrategy::Fixed`), as
/// batch FRI injects each shorter instance when the folded codeword reaches
/// its size.
pub fn batch_prove<F, C, const D: usize, const N: usize>(
    starks: &[&dyn BatchStark<F, D>; N],
    config: &StarkConfig,
    trace_poly_values: [Vec<PolynomialValues<F>>; N],
    public_inputs: &[Vec<F>; N],
    cross_table_lookups: &[CrossTableLookup<F>],
    grouped_tables: &[usize],
    preprocessed: Option<&BatchStarkPreprocessedData<F, C, D>>,
    known_columns: Option<&BatchKnownColumns<F>>,
    timing: &mut TimingTree,
) -> Result<BatchStarkProofWithPublicInputs<F, C, D>>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let rate_bits = config.fri_config.rate_bits;
    let cap_height = config.fri_config.cap_height;
    ensure!(
        config.preprocessed_columns.is_empty(),
        "In batch mode, preprocessed columns are described by `BatchStarkPreprocessedData`, \
         not by `StarkConfig::preprocessed_columns`."
    );

    let degree_bits: Vec<usize> = trace_poly_values
        .iter()
        .map(|t| log2_strict(t[0].len()))
        .collect();
    let empty_prep = vec![vec![]; N];
    let prep_columns = preprocessed.map_or(&empty_prep, |p| &p.columns_per_table);
    let layout = BatchStarkLayout::new(
        starks.as_slice(),
        config,
        &degree_bits,
        prep_columns,
        cross_table_lookups,
        grouped_tables,
    )?;

    for t in 0..N {
        ensure!(
            trace_poly_values[t].len() == starks[t].num_columns(),
            "Table {t}: trace column count mismatch."
        );
        ensure!(
            public_inputs[t].len() == starks[t].num_public_inputs(),
            "Table {t}: public input count mismatch."
        );
        ensure!(
            starks[t].constraint_degree() <= (1 << rate_bits) + 1,
            "Table {t}: the constraint degree must be <= blowup factor + 1."
        );
    }

    // Check that the known columns of the traces match the values the
    // verifier knows.
    if let Some(known) = known_columns {
        known.validate(&layout.num_columns, &degree_bits, prep_columns)?;
        for t in 0..N {
            for (k, &c) in known.columns_per_table[t].iter().enumerate() {
                ensure!(
                    trace_poly_values[t][c] == known.values_per_table[t][k],
                    "Table {t}, column {c}: trace disagrees with the known-column values."
                );
            }
        }
    }

    let max_degree_bits = degree_bits.iter().copied().max().unwrap();
    let fri_params = config.fri_params(max_degree_bits);

    // In debug mode, check that the preprocessed columns of the traces match
    // the setup-time commitment.
    #[cfg(debug_assertions)]
    if let Some(prep) = preprocessed {
        for t in 0..N {
            if layout.prep_columns[t].is_empty() {
                continue;
            }
            let start = layout.prep_poly_start(t);
            for (k, &c) in layout.prep_columns[t].iter().enumerate() {
                debug_assert_eq!(
                    trace_poly_values[t][c].clone().ifft(),
                    prep.commitment.polynomials[start + k],
                    "Table {t}, column {c}: trace disagrees with the preprocessed commitment."
                );
            }
        }
    }

    // After the trace columns are moved into the batched oracle below, the
    // lookup and CTL polynomials still need trace values: keep a sparse
    // per-table copy of just those columns, with cheap placeholders elsewhere
    // (mirrors `create_sparse_trace_for_lookups` in `prover.rs`).
    let lookups_per_table: Vec<_> = (0..N).map(|t| starks[t].lookups()).collect();
    let sparse_traces: [Vec<PolynomialValues<F>>; N] = core::array::from_fn(|t| {
        let mut keep: HashSet<usize> = lookups_per_table[t]
            .iter()
            .flat_map(|lookup| lookup.all_column_indices())
            .collect();
        for ctl in cross_table_lookups {
            keep.extend(ctl.column_indices_of_table(t));
        }
        keep.insert(0); // Always include column 0 for degree.
        trace_poly_values[t]
            .iter()
            .enumerate()
            .map(|(c, poly)| {
                if keep.contains(&c) {
                    poly.clone()
                } else {
                    PolynomialValues::zero(1)
                }
            })
            .collect()
    });

    // Flattens per-table polynomial values into the grouped oracle's
    // canonical order — descending trace degree, ties by table index — and
    // commits them in one multi-height tree
    // (cf. `BatchStarkLayout::grouped_poly_start`).
    let commit_grouped = |mut per_table: Vec<(usize, Vec<PolynomialValues<F>>)>,
                          timing: &mut TimingTree|
     -> Option<BatchFriOracle<F, C, D>> {
        per_table.retain(|(_, values)| !values.is_empty());
        if per_table.is_empty() {
            return None;
        }
        per_table.sort_by_key(|&(t, _)| (core::cmp::Reverse(degree_bits[t]), t));
        let values: Vec<PolynomialValues<F>> = per_table
            .into_iter()
            .flat_map(|(_, values)| values)
            .collect();
        let fft_tables: Vec<Option<&FftRootTable<F>>> = vec![None; values.len()];
        Some(BatchFriOracle::from_values(
            values,
            rate_bits,
            false,
            cap_height,
            timing,
            &fft_tables,
        ))
    };

    // Commit each solo table's online columns in its own Merkle tree and the
    // grouped tables' in one shared tree. Preprocessed columns are dropped
    // here: the setup-time commitment already holds them.
    let mut trace_oracles: Vec<Option<BatchFriOracle<F, C, D>>> = Vec::with_capacity(N);
    let mut trace_caps = Vec::new();
    let grouped_trace_oracle = timed!(timing, "compute trace commitments", {
        let mut grouped_values: Vec<(usize, Vec<PolynomialValues<F>>)> = Vec::new();
        for (t, columns) in trace_poly_values.into_iter().enumerate() {
            let online = &layout.online_columns[t];
            let values: Vec<PolynomialValues<F>> = columns
                .into_iter()
                .enumerate()
                .filter(|(c, _)| online.binary_search(c).is_ok())
                .map(|(_, poly)| poly)
                .collect();
            if layout.is_grouped(t) {
                grouped_values.push((t, values));
                trace_oracles.push(None);
                continue;
            }
            if values.is_empty() {
                trace_oracles.push(None);
                continue;
            }
            let fft_tables: Vec<Option<&FftRootTable<F>>> = vec![None; values.len()];
            let oracle = BatchFriOracle::from_values(
                values,
                rate_bits,
                false,
                cap_height,
                timing,
                &fft_tables,
            );
            trace_caps.push(oracle.batch_merkle_tree.cap.clone());
            trace_oracles.push(Some(oracle));
        }
        // The grouped cap comes after the solo caps, matching the oracle order.
        let grouped = commit_grouped(grouped_values, timing);
        if let Some(oracle) = &grouped {
            trace_caps.push(oracle.batch_merkle_tree.cap.clone());
        }
        grouped
    });

    // Fiat-Shamir: public inputs, preprocessed cap, known columns, config,
    // trace caps (table order).
    let mut challenger = Challenger::<F, C::Hasher>::new();
    for pis in public_inputs {
        challenger.observe_elements(pis);
    }
    if let Some(prep) = preprocessed {
        challenger.observe_cap(&prep.commitment.batch_merkle_tree.cap);
    }
    if let Some(known) = known_columns {
        known.observe(&mut challenger);
    }
    config.observe(&mut challenger);
    for cap in &trace_caps {
        challenger.observe_cap(cap);
    }

    // Draw a single challenge set, shared by the lookups and CTLs of all tables.
    let has_aux = (0..N).any(|t| layout.num_aux_polys(t) > 0);
    let challenge_set =
        has_aux.then(|| get_grand_product_challenge_set(&mut challenger, config.num_challenges));

    // Compute the CTL data of every table.
    let ctl_data_per_table: Option<[CtlData<F>; N]> =
        (!cross_table_lookups.is_empty()).then(|| {
            timed!(
                timing,
                "compute CTL data",
                cross_table_lookup_data::<F, D, N>(
                    &sparse_traces,
                    public_inputs,
                    cross_table_lookups,
                    challenge_set.as_ref().unwrap(),
                    layout.ctl_constraint_degree,
                )
            )
        });
    #[cfg(debug_assertions)]
    if let Some(ctl_data) = &ctl_data_per_table {
        for t in 0..N {
            debug_assert_eq!(
                ctl_data[t].num_ctl_helper_polys().iter().sum::<usize>(),
                layout.num_ctl_helpers[t]
            );
            debug_assert_eq!(ctl_data[t].zs_columns.len(), layout.num_ctl_zs[t]);
        }
    }
    let ctl_data_of = |t: usize| ctl_data_per_table.as_ref().map(|d| &d[t]);

    // Lookup challenges, shared by every table using lookups.
    let lookup_challenges: Option<Vec<F>> = challenge_set
        .as_ref()
        .map(|set| set.challenges.iter().map(|ch| ch.beta).collect());
    let lookup_challenges_of = |t: usize| -> Option<&Vec<F>> {
        starks[t]
            .uses_lookups()
            .then(|| lookup_challenges.as_ref().unwrap())
    };

    // Compute the auxiliary polynomials (lookup helpers, then CTL helpers and Zs) of each table.
    let aux_polys_per_table: Vec<Vec<PolynomialValues<F>>> = timed!(
        timing,
        "compute auxiliary polynomials",
        (0..N)
            .map(|t| {
                let mut aux = lookup_challenges_of(t)
                    .map(|challenges| {
                        lookups_per_table[t]
                            .iter()
                            .flat_map(|lookup| {
                                challenges.iter().map(move |&challenge| (lookup, challenge))
                            })
                            .collect::<Vec<_>>()
                            .into_par_iter()
                            .flat_map(|(lookup, challenge)| {
                                lookup_helper_columns(
                                    lookup,
                                    &sparse_traces[t],
                                    &public_inputs[t],
                                    challenge,
                                    starks[t].constraint_degree(),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if let Some(ctl_polys) = get_ctl_auxiliary_polys(ctl_data_of(t)) {
                    aux.extend(ctl_polys);
                }
                debug_assert_eq!(aux.len(), layout.num_aux_polys(t));
                aux
            })
            .collect()
    );

    // Commit each solo table's auxiliary polynomials in its own Merkle tree
    // and the grouped tables' in one shared tree.
    let mut aux_oracles: Vec<Option<BatchFriOracle<F, C, D>>> = Vec::with_capacity(N);
    let mut auxiliary_polys_caps: Option<Vec<MerkleCap<F, C::Hasher>>> = has_aux.then(Vec::new);
    let grouped_aux_oracle = timed!(timing, "compute auxiliary commitments", {
        let mut grouped_values: Vec<(usize, Vec<PolynomialValues<F>>)> = Vec::new();
        for (t, aux) in aux_polys_per_table.into_iter().enumerate() {
            if layout.is_grouped(t) {
                grouped_values.push((t, aux));
                aux_oracles.push(None);
                continue;
            }
            if aux.is_empty() {
                aux_oracles.push(None);
                continue;
            }
            let fft_tables: Vec<Option<&FftRootTable<F>>> = vec![None; aux.len()];
            let oracle =
                BatchFriOracle::from_values(aux, rate_bits, false, cap_height, timing, &fft_tables);
            if let Some(caps) = &mut auxiliary_polys_caps {
                caps.push(oracle.batch_merkle_tree.cap.clone());
            }
            aux_oracles.push(Some(oracle));
        }
        let grouped = commit_grouped(grouped_values, timing);
        if let Some(oracle) = &grouped {
            auxiliary_polys_caps
                .as_mut()
                .expect("grouped auxiliary polys imply has_aux")
                .push(oracle.batch_merkle_tree.cap.clone());
        }
        grouped
    });
    if let Some(caps) = &auxiliary_polys_caps {
        for cap in caps {
            challenger.observe_cap(cap);
        }
    }

    // Constraint-binding grind: evaluate all constraints on random dummy
    // openings and absorb the results, so that the quotient challenges
    // `alphas` are bound to the constraints.
    let alphas_prime = challenger.get_n_challenges(config.num_challenges);
    for t in 0..N {
        let pow_degree = core::cmp::max(2, starks[t].constraint_degree() + 1);
        let dummy_openings = get_dummy_polys::<F, C, D>(
            &mut challenger,
            starks[t].num_columns(),
            layout.num_aux_polys(t),
            pow_degree,
        );

        // Dummy CTL check vars: real challenges/columns/filters, dummy evaluations.
        let num_lookup_columns = layout.num_lookup_columns[t];
        let total_num_ctl_helpers = layout.num_ctl_helpers[t];
        let dummy_ctl_vars = ctl_data_of(t).map(|data| {
            let mut start_index = 0;
            data.zs_columns
                .iter()
                .enumerate()
                .map(|(i, zs_columns)| {
                    let num_ctl_helper_cols = zs_columns.helper_columns.len();
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
                        challenges: zs_columns.challenge,
                        columns: zs_columns.columns.clone(),
                        filter: zs_columns.filter.clone(),
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
            degree_bits[t],
            num_lookup_columns,
        );
        challenger.observe_extension_elements(&constraint_evals);
    }

    let alphas = challenger.get_n_challenges(config.num_challenges);

    // Resolves table `t`'s trace or auxiliary oracle and its position within
    // it: `(oracle, degree index, leaf offset)` — solo oracles hold one table
    // at `(0, 0)`, grouped tables address the shared multi-height oracle.
    let role_oracle = |role: BatchRole, t: usize| -> (&BatchFriOracle<F, C, D>, usize, usize) {
        let (oracle, grouped) = match role {
            BatchRole::Trace if layout.is_grouped(t) => (&grouped_trace_oracle, true),
            BatchRole::Trace => (&trace_oracles[t], false),
            BatchRole::Auxiliary if layout.is_grouped(t) => (&grouped_aux_oracle, true),
            BatchRole::Auxiliary => (&aux_oracles[t], false),
            BatchRole::Quotient => unreachable!("quotient oracles are not read back"),
        };
        let (degree_index, leaf_offset) = if grouped {
            layout.grouped_leaf_position(role, t)
        } else {
            (0, 0)
        };
        (
            oracle
                .as_ref()
                .expect("the table has polynomials of this role"),
            degree_index,
            leaf_offset,
        )
    };

    // Compute and commit to the quotient polynomials of all tables.
    let quotient_chunks_per_table: Vec<Option<Vec<PolynomialCoeffs<F>>>> = timed!(
        timing,
        "compute quotient polynomials",
        (0..N)
            .map(|t| {
                let quotient_polys =
                    compute_batch_quotient_polys::<F, <F as Packable>::Packing, C, D>(
                        starks[t],
                        &layout,
                        t,
                        role_oracle(BatchRole::Trace, t),
                        preprocessed.map(|p| &p.commitment),
                        (layout.num_aux_polys(t) > 0).then(|| role_oracle(BatchRole::Auxiliary, t)),
                        lookup_challenges_of(t),
                        &lookups_per_table[t],
                        ctl_data_of(t),
                        &public_inputs[t],
                        alphas.clone(),
                        config,
                    )?;
                Ok(quotient_polys.map(|polys| {
                    let degree = 1 << degree_bits[t];
                    polys
                        .into_par_iter()
                        .flat_map(|mut quotient_poly| {
                            quotient_poly
                                .trim_to_len(degree * starks[t].quotient_degree_factor())
                                .expect(
                                    "Quotient has failed, the vanishing polynomial is not \
                                     divisible by Z_H",
                                );
                            // Split quotient into degree-n chunks.
                            quotient_poly.chunks(degree)
                        })
                        .collect::<Vec<_>>()
                }))
            })
            .collect::<Result<_>>()?
    );
    let has_quotient = quotient_chunks_per_table.iter().any(|q| q.is_some());
    let mut quotient_oracles: Vec<Option<BatchFriOracle<F, C, D>>> = Vec::with_capacity(N);
    let mut quotient_polys_caps: Option<Vec<MerkleCap<F, C::Hasher>>> = has_quotient.then(Vec::new);
    let grouped_quotient_oracle = timed!(timing, "compute quotient commitments", {
        let mut grouped_coeffs: Vec<(usize, Vec<PolynomialCoeffs<F>>)> = Vec::new();
        for (t, chunks) in quotient_chunks_per_table.into_iter().enumerate() {
            let coeffs = chunks.unwrap_or_default();
            if layout.is_grouped(t) {
                grouped_coeffs.push((t, coeffs));
                quotient_oracles.push(None);
                continue;
            }
            if coeffs.is_empty() {
                quotient_oracles.push(None);
                continue;
            }
            let fft_tables: Vec<Option<&FftRootTable<F>>> = vec![None; coeffs.len()];
            let oracle = BatchFriOracle::from_coeffs(
                coeffs,
                rate_bits,
                false,
                cap_height,
                timing,
                &fft_tables,
            );
            if let Some(caps) = &mut quotient_polys_caps {
                caps.push(oracle.batch_merkle_tree.cap.clone());
            }
            quotient_oracles.push(Some(oracle));
        }
        // Same canonical order as `commit_grouped`; quotient chunks each have
        // their table's trace degree, so the sort key is unchanged.
        grouped_coeffs.retain(|(_, coeffs)| !coeffs.is_empty());
        grouped_coeffs.sort_by_key(|&(t, _)| (core::cmp::Reverse(degree_bits[t]), t));
        if grouped_coeffs.is_empty() {
            None
        } else {
            let coeffs: Vec<PolynomialCoeffs<F>> = grouped_coeffs
                .into_iter()
                .flat_map(|(_, coeffs)| coeffs)
                .collect();
            let fft_tables: Vec<Option<&FftRootTable<F>>> = vec![None; coeffs.len()];
            let oracle = BatchFriOracle::from_coeffs(
                coeffs,
                rate_bits,
                false,
                cap_height,
                timing,
                &fft_tables,
            );
            quotient_polys_caps
                .as_mut()
                .expect("grouped quotient polys imply has_quotient")
                .push(oracle.batch_merkle_tree.cap.clone());
            Some(oracle)
        }
    });
    if let Some(caps) = &quotient_polys_caps {
        for cap in caps {
            challenger.observe_cap(cap);
        }
    }

    let zeta = challenger.get_extension_challenge::<D>();
    // To avoid leaking witness data, we want to ensure that our opening locations, `zeta` and
    // `g * zeta`, are not in any table's subgroup. It suffices to check the largest subgroup,
    // as it contains all the smaller ones.
    ensure!(
        zeta.exp_power_of_2(max_degree_bits) != F::Extension::ONE,
        "Opening point is in the subgroup."
    );

    // Resolves table `t`'s `role` oracle and the flat index of the table's
    // first polynomial within it (0 for solo tables).
    let poly_range = |role: BatchRole, t: usize| -> (&BatchFriOracle<F, C, D>, usize) {
        let oracle = match (role, layout.is_grouped(t)) {
            (BatchRole::Trace, true) => &grouped_trace_oracle,
            (BatchRole::Trace, false) => &trace_oracles[t],
            (BatchRole::Auxiliary, true) => &grouped_aux_oracle,
            (BatchRole::Auxiliary, false) => &aux_oracles[t],
            (BatchRole::Quotient, true) => &grouped_quotient_oracle,
            (BatchRole::Quotient, false) => &quotient_oracles[t],
        };
        let start = if layout.is_grouped(t) {
            layout.grouped_poly_start(role, t)
        } else {
            0
        };
        (
            oracle
                .as_ref()
                .expect("the table has polynomials of this role"),
            start,
        )
    };

    // Compute all per-table openings.
    let openings: Vec<StarkOpeningSet<F, D>> = timed!(
        timing,
        "compute openings",
        (0..N)
            .map(|t| {
                compute_table_openings(
                    &layout,
                    t,
                    zeta,
                    poly_range(BatchRole::Trace, t),
                    preprocessed.map(|p| &p.commitment),
                    (layout.num_aux_polys(t) > 0).then(|| poly_range(BatchRole::Auxiliary, t)),
                    (layout.num_quotient_polys[t] > 0).then(|| poly_range(BatchRole::Quotient, t)),
                )
            })
            .collect()
    );

    // Observe the openings table by table (canonical, profile-independent order).
    for os in &openings {
        os.observe(&mut challenger);
    }

    // Batched FRI argument for all openings.
    let instances = layout.fri_instances::<F, D>(zeta);
    let oracle_refs: Vec<&BatchFriOracle<F, C, D>> = layout
        .oracles
        .iter()
        .map(|&oracle| match oracle {
            BatchOracle::Trace(t) => trace_oracles[t].as_ref().unwrap(),
            BatchOracle::Preprocessed => &preprocessed.unwrap().commitment,
            BatchOracle::Auxiliary(t) => aux_oracles[t].as_ref().unwrap(),
            BatchOracle::Quotient(t) => quotient_oracles[t].as_ref().unwrap(),
            BatchOracle::GroupedTrace => grouped_trace_oracle.as_ref().unwrap(),
            BatchOracle::GroupedAuxiliary => grouped_aux_oracle.as_ref().unwrap(),
            BatchOracle::GroupedQuotient => grouped_quotient_oracle.as_ref().unwrap(),
        })
        .collect();
    let opening_proof = timed!(
        timing,
        "compute batched openings proof",
        BatchFriOracle::prove_openings(
            &layout.distinct_degree_bits,
            &instances,
            &oracle_refs,
            &mut challenger,
            &fri_params,
            timing,
        )
    );

    let proof = BatchStarkProof {
        trace_caps,
        auxiliary_polys_caps,
        quotient_polys_caps,
        openings,
        opening_proof,
        degree_bits,
    };

    // Drop the large structures in parallel to avoid blocking on deallocation.
    #[cfg(feature = "parallel")]
    plonky2_maybe_rayon::rayon::spawn(move || {
        drop(trace_oracles);
        drop(aux_oracles);
        drop(quotient_oracles);
        drop(grouped_trace_oracle);
        drop(grouped_aux_oracle);
        drop(grouped_quotient_oracle);
        drop(sparse_traces);
    });

    Ok(BatchStarkProofWithPublicInputs {
        proof,
        public_inputs: public_inputs.to_vec(),
    })
}

/// Computes the quotient polynomials of table `t`: `(sum alpha^i C_i(x)) / Z_H(x)`
/// for each `alpha`, where the `C_i`s are the table's constraints, evaluated
/// over a coset of the table's quotient evaluation domain.
///
/// The trace and auxiliary oracles come with the table's position inside
/// them, `(oracle, degree index, leaf offset)` — `(0, 0)` for solo oracles.
fn compute_batch_quotient_polys<F, P, C, const D: usize>(
    stark: &dyn BatchStark<F, D>,
    layout: &BatchStarkLayout,
    t: usize,
    trace_oracle: (&BatchFriOracle<F, C, D>, usize, usize),
    prep_oracle: Option<&BatchFriOracle<F, C, D>>,
    aux_oracle: Option<(&BatchFriOracle<F, C, D>, usize, usize)>,
    lookup_challenges: Option<&Vec<F>>,
    lookups: &[crate::lookup::Lookup<F>],
    ctl_data: Option<&CtlData<F>>,
    public_inputs: &[F],
    alphas: Vec<F>,
    config: &StarkConfig,
) -> Result<Option<Vec<PolynomialCoeffs<F>>>>
where
    F: RichField + Extendable<D> + Packable<Packing = P>,
    P: PackedField<Scalar = F>,
    C: GenericConfig<D, F = F>,
{
    if stark.quotient_degree_factor() == 0 {
        return Ok(None);
    }

    let degree_bits = layout.degree_bits[t];
    let degree = 1 << degree_bits;
    let rate_bits = config.fri_config.rate_bits;

    let quotient_degree_bits = log2_ceil(stark.quotient_degree_factor());
    ensure!(
        quotient_degree_bits <= rate_bits,
        "Having constraints of degree higher than the rate is not supported yet."
    );
    let step = 1 << (rate_bits - quotient_degree_bits);
    // When opening the `Z`s polys at the "next" point, need to look at the point `next_step` steps away.
    let next_step = 1 << quotient_degree_bits;

    // Evaluation of the first Lagrange polynomial on the LDE domain.
    let lagrange_first = PolynomialValues::selector(degree, 0).lde_onto_coset(quotient_degree_bits);
    // Evaluation of the last Lagrange polynomial on the LDE domain.
    let lagrange_last =
        PolynomialValues::selector(degree, degree - 1).lde_onto_coset(quotient_degree_bits);

    let z_h_on_coset = ZeroPolyOnCoset::<F>::new(degree_bits, quotient_degree_bits);

    // Retrieves the LDE values of `n` of the table's polynomials from a role
    // oracle at index `i_start`, addressing the table's degree index and
    // leaf offset within the (possibly shared) oracle.
    let get_values_packed =
        |(oracle, degree_index, leaf_offset): (&BatchFriOracle<F, C, D>, usize, usize),
         n: usize,
         i_start: usize|
         -> Vec<P> {
            oracle.get_lde_values_packed(degree_index, i_start, step, leaf_offset, n)
        };

    let num_columns = layout.num_columns[t];
    let num_lookup_columns = layout.num_lookup_columns[t];
    let num_ctl_columns = ctl_data
        .map(|data| data.num_ctl_helper_polys())
        .unwrap_or_default();
    let total_num_helper_cols: usize = num_ctl_columns.iter().sum();
    let num_aux = layout.num_aux_polys(t);

    // Assembles the full trace row (online and preprocessed columns interleaved
    // back in their original positions) at index `i_start`.
    let get_trace_values_packed = |i_start: usize| -> Vec<P> {
        let online = get_values_packed(trace_oracle, layout.online_columns[t].len(), i_start);
        let mut row = vec![P::ZEROS; num_columns];
        for (k, &c) in layout.online_columns[t].iter().enumerate() {
            row[c] = online[k];
        }
        if !layout.prep_columns[t].is_empty() {
            // The shared preprocessed oracle is multi-height: address the
            // table's columns through its degree index and leaf offset.
            let (degree_index, leaf_offset) = layout.prep_leaf_position(t);
            let prep = prep_oracle
                .expect("Preprocessed columns without a preprocessed oracle")
                .get_lde_values_packed(
                    degree_index,
                    i_start,
                    step,
                    leaf_offset,
                    layout.prep_columns[t].len(),
                );
            for (k, &c) in layout.prep_columns[t].iter().enumerate() {
                row[c] = prep[k];
            }
        }
        row
    };

    // Last element of the subgroup.
    let last = F::primitive_root_of_unity(degree_bits).inverse();
    let size = degree << quotient_degree_bits;
    let coset = F::cyclic_subgroup_coset_known_order(
        F::primitive_root_of_unity(degree_bits + quotient_degree_bits),
        F::coset_shift(),
        size,
    );

    // We will step by `P::WIDTH`, and in each iteration, evaluate the quotient polynomial at
    // a batch of `P::WIDTH` points.
    let quotient_values = (0..size)
        .into_par_iter()
        .step_by(P::WIDTH)
        .flat_map_iter(|i_start| {
            let i_next_start = (i_start + next_step) % size;
            let i_range = i_start..i_start + P::WIDTH;

            let x = *P::from_slice(&coset[i_range.clone()]);
            let z_last = x - last;
            let lagrange_basis_first = *P::from_slice(&lagrange_first.values[i_range.clone()]);
            let lagrange_basis_last = *P::from_slice(&lagrange_last.values[i_range]);

            let mut consumer = ConstraintConsumer::new(
                alphas.clone(),
                z_last,
                lagrange_basis_first,
                lagrange_basis_last,
            );

            let local_values = get_trace_values_packed(i_start);
            let next_values = get_trace_values_packed(i_next_start);

            let (aux_local, aux_next) = if num_aux > 0 {
                let aux_oracle = aux_oracle.expect("Auxiliary polys without an auxiliary oracle");
                (
                    get_values_packed(aux_oracle, num_aux, i_start),
                    get_values_packed(aux_oracle, num_aux, i_next_start),
                )
            } else {
                (vec![], vec![])
            };

            let lookup_vars = lookup_challenges.map(|challenges| {
                LookupCheckVars::new(
                    aux_local[..num_lookup_columns].to_vec(),
                    aux_next[..num_lookup_columns].to_vec(),
                    challenges.to_vec(),
                )
            });

            let ctl_vars = ctl_data.map(|data| {
                let mut start_index = 0;
                data.zs_columns
                    .iter()
                    .enumerate()
                    .map(|(i, zs_columns)| {
                        let num_ctl_helper_cols = num_ctl_columns[i];
                        let helper_columns = aux_local[num_lookup_columns + start_index
                            ..num_lookup_columns + start_index + num_ctl_helper_cols]
                            .to_vec();
                        let ctl_vars = CtlCheckVars::<F, F, P, 1> {
                            helper_columns,
                            local_z: aux_local[num_lookup_columns + total_num_helper_cols + i],
                            next_z: aux_next[num_lookup_columns + total_num_helper_cols + i],
                            challenges: zs_columns.challenge,
                            columns: zs_columns.columns.clone(),
                            filter: zs_columns.filter.clone(),
                        };
                        start_index += num_ctl_helper_cols;
                        ctl_vars
                    })
                    .collect::<Vec<_>>()
            });

            // Evaluate the polynomial combining all constraints, including
            // those associated to the lookup and CTL arguments.
            stark.eval_vanishing_poly_packed(
                &local_values,
                &next_values,
                public_inputs,
                lookups,
                lookup_vars,
                ctl_vars.as_deref(),
                &mut consumer,
            );

            let mut constraints_evals = consumer.accumulators();
            // We divide the constraints evaluations by `Z_H(x)`.
            let denominator_inv: P = z_h_on_coset.eval_inverse_packed(i_start);
            for eval in &mut constraints_evals {
                *eval *= denominator_inv;
            }

            let num_challenges = alphas.len();
            (0..P::WIDTH).map(move |i| {
                (0..num_challenges)
                    .map(|j| constraints_evals[j].as_slice()[i])
                    .collect()
            })
        })
        .collect::<Vec<_>>();

    Ok(Some(
        transpose(&quotient_values)
            .into_par_iter()
            .map(PolynomialValues::new)
            .map(|values| values.coset_ifft(F::coset_shift()))
            .collect(),
    ))
}

/// Computes the opening set of table `t` at `zeta`, `g_t * zeta` (where `g_t`
/// generates the table's trace subgroup) and, for CTL Z polynomials, `1`.
/// `local_values`/`next_values` contain the full column set, with the online
/// and preprocessed columns interleaved back in their original positions.
///
/// The trace, auxiliary and quotient oracles come with the flat index of the
/// table's first polynomial within them (0 for solo oracles).
fn compute_table_openings<F, C, const D: usize>(
    layout: &BatchStarkLayout,
    t: usize,
    zeta: F::Extension,
    trace_oracle: (&BatchFriOracle<F, C, D>, usize),
    prep_oracle: Option<&BatchFriOracle<F, C, D>>,
    aux_oracle: Option<(&BatchFriOracle<F, C, D>, usize)>,
    quotient_oracle: Option<(&BatchFriOracle<F, C, D>, usize)>,
) -> StarkOpeningSet<F, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let g = F::primitive_root_of_unity(layout.degree_bits[t]);
    let zeta_next = zeta.scalar_mul(g);

    // Evaluates a range of an oracle's polynomials at `z`, shifted by the
    // table's flat start within the oracle.
    let eval_polys = |(oracle, start): (&BatchFriOracle<F, C, D>, usize),
                      range: core::ops::Range<usize>,
                      z: F::Extension|
     -> Vec<F::Extension> {
        oracle.polynomials[start + range.start..start + range.end]
            .par_iter()
            .map(|p| p.to_extension::<D>().eval(z))
            .collect()
    };

    // Assembles the full trace openings at `z`.
    let eval_full_trace = |z: F::Extension| -> Vec<F::Extension> {
        let online = eval_polys(trace_oracle, 0..layout.online_columns[t].len(), z);
        let mut row = vec![F::Extension::ZERO; layout.num_columns[t]];
        for (k, &c) in layout.online_columns[t].iter().enumerate() {
            row[c] = online[k];
        }
        if !layout.prep_columns[t].is_empty() {
            // The table's columns sit at a flat range of the shared
            // preprocessed oracle.
            let start = layout.prep_poly_start(t);
            let prep = eval_polys(
                (
                    prep_oracle.expect("Preprocessed columns without a preprocessed oracle"),
                    0,
                ),
                start..start + layout.prep_columns[t].len(),
                z,
            );
            for (k, &c) in layout.prep_columns[t].iter().enumerate() {
                row[c] = prep[k];
            }
        }
        row
    };

    let has_aux = layout.num_aux_polys(t) > 0;
    let auxiliary_polys = has_aux.then(|| {
        eval_polys(
            aux_oracle.expect("Auxiliary polys without an auxiliary oracle"),
            0..layout.num_aux_polys(t),
            zeta,
        )
    });
    let auxiliary_polys_next =
        has_aux.then(|| eval_polys(aux_oracle.unwrap(), 0..layout.num_aux_polys(t), zeta_next));

    // CTL Z polynomials are also opened at 1 (the first row of the trace).
    let ctl_zs_first = (layout.num_ctl_zs[t] > 0).then(|| {
        let (aux_oracle, aux_start) = aux_oracle.unwrap();
        let start = aux_start + layout.num_lookup_columns[t] + layout.num_ctl_helpers[t];
        aux_oracle.polynomials[start..start + layout.num_ctl_zs[t]]
            .par_iter()
            .map(|p| p.eval(F::ONE))
            .collect()
    });

    let quotient_polys = (layout.num_quotient_polys[t] > 0).then(|| {
        eval_polys(
            quotient_oracle.expect("Quotient polys without a quotient oracle"),
            0..layout.num_quotient_polys[t],
            zeta,
        )
    });

    StarkOpeningSet {
        local_values: eval_full_trace(zeta),
        next_values: eval_full_trace(zeta_next),
        auxiliary_polys,
        auxiliary_polys_next,
        ctl_zs_first,
        quotient_polys,
    }
}
