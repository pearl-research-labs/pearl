//! This crate provides support for cross-table lookups.
//!
//! If a STARK S_1 calls an operation that is carried out by another STARK S_2,
//! S_1 provides the inputs to S_2 and reads the output from S_1. To ensure that
//! the operation was correctly carried out, we must check that the provided inputs
//! and outputs are correctly read. Cross-table lookups carry out that check.
//!
//! To achieve this, smaller CTL tables are created on both sides: looking and looked tables.
//! In our example, we create a table S_1' comprised of columns -- or linear combinations
//! of columns -- of S_1, and rows that call operations carried out in S_2. We also create a
//! table S_2' comprised of columns -- or linear combinations od columns -- of S_2 and rows
//! that carry out the operations needed by other STARKs. Then, S_1' is a looking table for
//! the looked S_2', since we want to check that the operation outputs in S_1' are indeeed in S_2'.
//! Furthermore, the concatenation of all tables looking into the looked side must be equal to
//! the concatenation of the looked tables. The looked side may itself consist of several
//! *slots* (several column groups, possibly of the same trace) — e.g. one slot per fold of a
//! folded lookup table; both sides go through the same helper-column batching.
//!
//! To achieve this, we construct, for each table, a permutation polynomial Z(x).
//! Z(x) is computed as the product of all its column combinations.
//! To check it was correctly constructed, we check:
//! - Z(gw) = Z(w) * combine(w) where combine(w) is the column combination at point w.
//! - Z(g^(n-1)) = combine(1).
//! - The verifier also checks that the sum of looking table Z polynomial openings is equal
//!   to the sum of looked table Z polynomial openings.
//!
//! Note that the first two checks are written that way because Z polynomials are computed
//! upside down for convenience.
//!
//! Additionally, we support cross-table lookups over two rows. The permutation principle
//! is similar, but we provide not only `local_values` but also `next_values` -- corresponding to
//! the current and next row values -- when computing the linear combinations.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use anyhow::{ensure, Result};
use hashbrown::HashMap;
use itertools::Itertools;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::challenger::Challenger;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::GenericConfig;
use plonky2_maybe_rayon::*;

use crate::config::StarkConfig;
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::evaluation_frame::StarkEvaluationFrame;
use crate::lookup::{
    eval_helper_columns, eval_helper_columns_circuit, get_grand_product_challenge_set, Column,
    Filter, GrandProductChallenge, GrandProductChallengeSet,
};
use crate::proof::{StarkOpeningSet, StarkOpeningSetTarget, StarkProof, StarkProofTarget};
use crate::stark::Stark;

/// An alias for `usize`, to represent the index of a STARK table in a multi-STARK setting.
pub type TableIdx = usize;

/// Row-chunk granularity for the per-chunk helper accumulation.
const CTL_ACC_CHUNK: usize = 1 << 15;

/// Number of blocks in the parallel Z suffix scan.
const CTL_SCAN_BLOCKS: usize = 96;

/// A `table` index with a linear combination of columns and a filter.
/// `filter` is used to determine the rows to select in `table`.
/// `columns` represents linear combinations of the columns of `table`.
#[derive(Clone, Debug)]
pub struct TableWithColumns<F: Field> {
    table: TableIdx,
    columns: Vec<Column<F>>,
    filter: Filter<F>,
}

impl<F: Field> TableWithColumns<F> {
    /// Generates a new `TableWithColumns` given a `table` index, a linear combination of columns `columns` and a `filter`.
    pub fn new(table: TableIdx, columns: Vec<Column<F>>, filter: Filter<F>) -> Self {
        Self {
            table,
            columns,
            filter,
        }
    }
}

/// Cross-table lookup data consisting in the lookup tables (`looked_tables`) and all the tables that look into `looked_tables` (`looking_tables`).
/// Each table corresponds to a STARK's table whose rows have been filtered out and whose columns have been through a linear combination (see `eval_table`). The concatenation of the looking tables should result in the concatenation of the looked tables.
///
/// The looked side may consist of several *slots* — different column groups of the same trace
/// (or of different traces), e.g. one slot per fold of a folded lookup table. Both sides are
/// treated symmetrically: on each side, the instances of one trace are batched into one Z
/// polynomial, with helper columns whenever a trace carries more than one instance.
#[derive(Clone, Debug)]
pub struct CrossTableLookup<F: Field> {
    /// Column linear combinations for all tables that are looking into the looked tables.
    pub(crate) looking_tables: Vec<TableWithColumns<F>>,
    /// Column linear combinations for all slots of the looked side.
    pub(crate) looked_tables: Vec<TableWithColumns<F>>,
}

impl<F: Field> CrossTableLookup<F> {
    /// Creates a new `CrossTableLookup` given some looking tables and the looked tables (slots).
    /// All tables should have the same width, and on each side all instances of the same trace
    /// must be adjacent (the prover and the verifier batch them into one Z polynomial per
    /// trace, in order of first appearance).
    pub fn new(
        looking_tables: Vec<TableWithColumns<F>>,
        looked_tables: Vec<TableWithColumns<F>>,
    ) -> Self {
        assert!(!looked_tables.is_empty(), "a CTL needs a looked side");
        let width = looked_tables[0].columns.len();
        assert!(looking_tables
            .iter()
            .chain(&looked_tables)
            .all(|twc| twc.columns.len() == width));
        for side in [&looking_tables, &looked_tables] {
            let mut deduped: Vec<TableIdx> = Vec::new();
            for twc in side {
                if deduped.last() != Some(&twc.table) {
                    assert!(
                        !deduped.contains(&twc.table),
                        "instances of table {} are not adjacent",
                        twc.table
                    );
                    deduped.push(twc.table);
                }
            }
        }
        Self {
            looking_tables,
            looked_tables,
        }
    }

    /// Returns all trace column indices referenced by this `CrossTableLookup`
    /// (from both looking and looked tables, including columns and filters).
    pub fn all_column_indices(&self) -> Vec<usize> {
        self.looking_tables
            .iter()
            .chain(&self.looked_tables)
            .flat_map(|twc| {
                twc.columns
                    .iter()
                    .flat_map(|c| c.relevant_columns())
                    .chain(twc.filter.relevant_columns())
            })
            .unique()
            .collect()
    }

    /// Returns the trace column indices of table `table` referenced by this
    /// `CrossTableLookup` (columns and filters of the looking/looked entries
    /// of that table).
    pub fn column_indices_of_table(&self, table: TableIdx) -> Vec<usize> {
        self.looking_tables
            .iter()
            .chain(&self.looked_tables)
            .filter(|twc| twc.table == table)
            .flat_map(|twc| {
                twc.columns
                    .iter()
                    .flat_map(|c| c.relevant_columns())
                    .chain(twc.filter.relevant_columns())
            })
            .unique()
            .collect()
    }

    /// Given a table, returns:
    /// - the total number of helper columns for this table, over all Cross-table lookups,
    /// - the total number of z polynomials for this table, over all Cross-table lookups,
    /// - the number of helper columns for this table, for each Cross-table lookup, split as
    ///   `[looking, looked]` (a table can carry helper columns on both sides of one CTL).
    pub fn num_ctl_helpers_zs_all(
        ctls: &[Self],
        table: TableIdx,
        num_challenges: usize,
        constraint_degree: usize,
    ) -> (usize, usize, Vec<[usize; 2]>) {
        let mut num_helpers = 0;
        let mut num_ctls = 0;
        let mut num_helpers_by_ctl = vec![[0; 2]; ctls.len()];
        for (i, ctl) in ctls.iter().enumerate() {
            for (side, tables) in [&ctl.looking_tables, &ctl.looked_tables]
                .into_iter()
                .enumerate()
            {
                let num_appearances = tables.iter().filter(|twc| twc.table == table).count();
                if num_appearances > 1 {
                    num_helpers_by_ctl[i][side] = num_appearances.div_ceil(constraint_degree - 1);
                    num_helpers += num_helpers_by_ctl[i][side];
                }
                if num_appearances > 0 {
                    num_ctls += 1;
                }
            }
        }
        (
            num_helpers * num_challenges,
            num_ctls * num_challenges,
            num_helpers_by_ctl,
        )
    }
}

/// Cross-table lookup data for one table.
#[derive(Clone, Default, Debug)]
pub struct CtlData<'a, F: Field> {
    /// Data associated with all Z(x) polynomials for one table.
    pub zs_columns: Vec<CtlZData<'a, F>>,
}

/// Cross-table lookup data associated with one Z(x) polynomial.
/// One Z(x) polynomial can be associated to multiple tables,
/// built from the same STARK.
#[derive(Clone, Debug)]
pub struct CtlZData<'a, F: Field> {
    /// Helper columns to verify the Z polynomial values.
    pub(crate) helper_columns: Vec<PolynomialValues<F>>,
    /// Z polynomial values.
    pub(crate) z: PolynomialValues<F>,
    /// Cross-table lookup challenge.
    pub challenge: GrandProductChallenge<F>,
    /// Vector of column linear combinations for the current tables.
    pub(crate) columns: Vec<&'a [Column<F>]>,
    /// Vector of filter columns for the current table.
    /// Each filter evaluates to either 1 or 0.
    pub(crate) filter: Vec<Filter<F>>,
}

impl<'a, F: Field> CtlZData<'a, F> {
    /// Returns new CTL data from the provided arguments.
    pub fn new(
        helper_columns: Vec<PolynomialValues<F>>,
        z: PolynomialValues<F>,
        challenge: GrandProductChallenge<F>,
        columns: Vec<&'a [Column<F>]>,
        filter: Vec<Filter<F>>,
    ) -> Self {
        Self {
            helper_columns,
            z,
            challenge,
            columns,
            filter,
        }
    }
}

impl<F: Field> CtlData<'_, F> {
    /// Returns all the cross-table lookup helper polynomials.
    pub(crate) fn ctl_helper_polys(&self) -> Vec<PolynomialValues<F>> {
        let num_polys = self
            .zs_columns
            .iter()
            .fold(0, |acc, z| acc + z.helper_columns.len());
        let mut res = Vec::with_capacity(num_polys);
        for z in &self.zs_columns {
            res.extend(z.helper_columns.clone());
        }

        res
    }

    /// Moves the CTL helper and Z polynomials' values out, leaving the table's
    /// `CtlData` with its metadata (challenges, columns, filters) and helper
    /// counts. The committed-oracle consumers of `CtlData` never read the
    /// value polys — they read them back from the auxiliary commitment — so
    /// proving pipelines that own their `CtlData` can hand the values to the
    /// auxiliary-polynomial assembly without a multi-gigabyte deep copy.
    /// The helper *counts* (`zs_columns[i].helper_columns.len()`) are
    /// preserved for layout math downstream.
    pub(crate) fn take_ctl_polys(&mut self) -> Vec<PolynomialValues<F>> {
        let num_helpers: usize = self.zs_columns.iter().map(|z| z.helper_columns.len()).sum();
        let num_zs = self.zs_columns.len();
        let mut res = Vec::with_capacity(num_helpers + num_zs);
        for z in &mut self.zs_columns {
            for pv in &mut z.helper_columns {
                res.push(core::mem::take(pv));
            }
        }
        for z in &mut self.zs_columns {
            res.push(core::mem::take(&mut z.z));
        }
        res
    }

    /// Returns all the Z cross-table-lookup polynomials.
    pub(crate) fn ctl_z_polys(&self) -> Vec<PolynomialValues<F>> {
        let mut res = Vec::with_capacity(self.zs_columns.len());
        for z in &self.zs_columns {
            res.push(z.z.clone());
        }

        res
    }
    /// Returns the number of helper columns for each STARK in each
    /// `CtlZData`.
    pub(crate) fn num_ctl_helper_polys(&self) -> Vec<usize> {
        let mut res = Vec::with_capacity(self.zs_columns.len());
        for z in &self.zs_columns {
            res.push(z.helper_columns.len());
        }

        res
    }
}

/// Outputs a tuple of (challenges, data) of CTL challenges and all
/// the CTL data necessary to prove a multi-STARK system.
pub fn get_ctl_data<'a, F, C, const D: usize, const N: usize>(
    config: &StarkConfig,
    trace_poly_values: &[Vec<PolynomialValues<F>>; N],
    public_inputs: &[Vec<F>; N],
    all_cross_table_lookups: &'a [CrossTableLookup<F>],
    challenger: &mut Challenger<F, C::Hasher>,
    max_constraint_degree: usize,
) -> (GrandProductChallengeSet<F>, [CtlData<'a, F>; N])
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    // Get challenges for the cross-table lookups.
    let ctl_challenges = get_grand_product_challenge_set(challenger, config.num_challenges);

    // For each STARK, compute its cross-table lookup Z polynomials
    // and get the associated `CtlData`.
    let ctl_data = cross_table_lookup_data::<F, D, N>(
        trace_poly_values,
        public_inputs,
        all_cross_table_lookups,
        &ctl_challenges,
        max_constraint_degree,
    );

    (ctl_challenges, ctl_data)
}

/// Gets the auxiliary polynomials associated to these CTL data.
pub(crate) fn get_ctl_auxiliary_polys<F: Field>(
    ctl_data: Option<&CtlData<F>>,
) -> Option<Vec<PolynomialValues<F>>> {
    ctl_data.map(|data| {
        let mut ctl_polys = data.ctl_helper_polys();
        ctl_polys.extend(data.ctl_z_polys());
        ctl_polys
    })
}

/// Move-based variant of [`get_ctl_auxiliary_polys`] for proving pipelines
/// that own their [`CtlData`]: the helper and Z polynomials' buffers are
/// moved into the auxiliary-polynomial assembly instead of cloned, and the
/// `CtlData` retains its metadata and helper counts for downstream layout
/// math. Functionally identical to the clone-based variant.
pub(crate) fn take_ctl_auxiliary_polys<F: Field>(
    ctl_data: Option<&mut CtlData<F>>,
) -> Option<Vec<PolynomialValues<F>>> {
    ctl_data.map(CtlData::take_ctl_polys)
}

/// Generates all the cross-table lookup data, for all tables.
/// - `trace_poly_values` corresponds to the trace values for all tables.
/// - `public_inputs` corresponds to the public inputs of all tables (the column expressions
///   may carry public-input terms).
/// - `cross_table_lookups` corresponds to all the cross-table lookups, i.e. the looked and looking tables, as described in `CrossTableLookup`.
/// - `ctl_challenges` corresponds to the challenges used for CTLs.
/// - `constraint_degree` is the maximal constraint degree for the table.
///
/// For each `CrossTableLookup`, and each looking/looked table, the partial products for the CTL are computed, and added to the said table's `CtlZData`.
pub(crate) fn cross_table_lookup_data<'a, F: RichField, const D: usize, const N: usize>(
    trace_poly_values: &[Vec<PolynomialValues<F>>; N],
    public_inputs: &[Vec<F>; N],
    cross_table_lookups: &'a [CrossTableLookup<F>],
    ctl_challenges: &GrandProductChallengeSet<F>,
    constraint_degree: usize,
) -> [CtlData<'a, F>; N] {
    // The unit of parallelizable work: one (table, side-instances) group of one side of
    // one CTL under one challenge — exactly one `partial_sums_with_evals` call.
    //
    // The trace-side evaluations (column linear combinations and filters) are
    // challenge-independent: only the beta/gamma combining, the batch inversion and
    // the Z suffix sums vary per challenge. Pass 1 therefore evaluates each
    // (ctl, side, table-group)'s columns and filters once; pass 2 enumerates works in
    // the exact serial order the verifier mirrors (per CTL, per challenge, looking
    // side's groups first), sharing the group's precomputed evaluations.
    struct CtlWork<'a, F: Field> {
        table: TableIdx,
        group: usize,
        challenge: GrandProductChallenge<F>,
        columns: Vec<&'a [Column<F>]>,
        filter: Vec<Filter<F>>,
    }

    // Pass 1: evaluate each group's columns and filters once. Groups are
    // numbered in (ctl, side, table) order, matching pass 2's enumeration.
    // The (table, instance) descriptors are collected first so the expensive
    // evaluations can run in parallel across groups.
    struct GroupDesc<'a, F: Field> {
        table: TableIdx,
        columns: Vec<&'a [Column<F>]>,
        filters: Vec<&'a Filter<F>>,
    }
    struct GroupEvals<F: Field> {
        columns: Vec<Vec<Vec<F>>>, // per instance: per column element: rows
        filters: Vec<Vec<F>>,      // per instance: rows
    }
    let mut group_descs: Vec<GroupDesc<'a, F>> = Vec::new();
    // Per CTL: the id of its first group.
    let mut ctl_first_group: Vec<usize> = Vec::new();

    for ctl in cross_table_lookups {
        let CrossTableLookup {
            looking_tables,
            looked_tables,
        } = ctl;
        log::debug!("Processing CTL for {:?}", looked_tables[0].table);
        ctl_first_group.push(group_descs.len());
        for tables in [looking_tables, looked_tables] {
            // Group this side's instances by trace: each group shares
            // helper columns and one Z polynomial.
            for (table, group) in &tables.iter().group_by(|twc| twc.table) {
                let group: Vec<&TableWithColumns<F>> = group.collect();
                group_descs.push(GroupDesc {
                    table,
                    columns: group.iter().map(|twc| &twc.columns[..]).collect(),
                    filters: group.iter().map(|twc| &twc.filter).collect(),
                });
            }
        }
    }

    // Evaluate the groups in parallel; each group's columns and filters are
    // independent of every other group's. Within a group, the instances'
    // evaluations are independent too, so big groups (many instances over a
    // 2^21-row trace) also spread across the pool.
    let group_evals: Vec<GroupEvals<F>> = group_descs
        .into_par_iter()
        .map(|desc| {
            let trace = &trace_poly_values[desc.table];
            let pis = &public_inputs[desc.table];
            let columns = desc
                .columns
                .par_iter()
                .map(|col| {
                    col.iter()
                        .map(|c| c.eval_all_rows_with_inputs(trace, pis))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let filters = desc
                .filters
                .par_iter()
                .map(|f| f.eval_all_rows_with_inputs(trace, pis))
                .collect::<Vec<_>>();
            GroupEvals { columns, filters }
        })
        .collect();

    // Pass 2: enumerate works in the exact serial order, pairing each with
    // its group's shared evaluations.
    let mut works: Vec<CtlWork<'a, F>> = Vec::new();
    for (ctl, &first) in cross_table_lookups.iter().zip(&ctl_first_group) {
        let CrossTableLookup {
            looking_tables,
            looked_tables,
        } = ctl;
        for &challenge in &ctl_challenges.challenges {
            let mut group_id = first;
            for tables in [looking_tables, looked_tables] {
                for (table, group) in &tables.iter().group_by(|twc| twc.table) {
                    let group: Vec<&TableWithColumns<F>> = group.collect();
                    works.push(CtlWork {
                        table,
                        group: group_id,
                        challenge,
                        columns: group.iter().map(|twc| &twc.columns[..]).collect(),
                        filter: group.iter().map(|twc| twc.filter.clone()).collect(),
                    });
                    group_id += 1;
                }
            }
        }
    }

    let results: Vec<Vec<PolynomialValues<F>>> = works
        .par_iter()
        .map(|work| {
            let ge = &group_evals[work.group];
            partial_sums_with_evals(&ge.columns, &ge.filters, work.challenge, constraint_degree)
        })
        .collect::<Vec<_>>();
    let mut ctl_data_per_table = [0; N].map(|_| CtlData::default());
    for (work, mut helpers_zs) in works.into_iter().zip(results) {
        let z = helpers_zs
            .pop()
            .expect("partial sums always return a Z poly");
        ctl_data_per_table[work.table].zs_columns.push(CtlZData {
            helper_columns: helpers_zs,
            z,
            challenge: work.challenge,
            columns: work.columns,
            filter: work.filter,
        });
    }
    ctl_data_per_table
}

/// Computes the cross-table lookup partial sums for one table and given column linear combinations,
/// over pre-evaluated column and filter values.
///
/// `columns` holds, per instance, the row values of each element of the instance's column linear
/// combination; `filters` holds, per instance, the row values of the instance's filter.
/// The combination with the challenge, the batch inversion, the filter multiply, and the Z suffix
/// sum are the only challenge-dependent steps.
///
/// The values `v` are grouped together, in groups of size `constraint_degree - 1`. For each group,
/// we construct a helper column: h = \sum_i 1/(v_i).
///
/// The sum is updated: `s += \sum h_i`, and is pushed to the vector of partial sums `z`.
/// Returns the helper columns and `z`.
///
/// Rows are processed in parallel across the instance's row range: the per-instance
/// `combine` map and the batch inversion both parallelize over row chunks, so a
/// single work over a 2^22-row table can use the whole rayon pool even when few
/// CTL works exist.
fn partial_sums_with_evals<F: Field>(
    columns: &[Vec<Vec<F>>],
    filters: &[Vec<F>],
    challenge: GrandProductChallenge<F>,
    constraint_degree: usize,
) -> Vec<PolynomialValues<F>> {
    let degree = filters.first().map_or(0, |f| f.len());
    debug_assert!(columns.iter().flatten().all(|c| c.len() == degree));

    let group = constraint_degree.checked_sub(1).unwrap_or(1);

    // Row-parallel combine+inverse+filter for one instance: splits `degree`
    // rows into chunks, computes each chunk's `combine` (a pure map) into its
    // slice of the single preallocated output buffer, inverts it with an
    // independent Montgomery trick, and multiplies by the filter.
    // Functionally identical to the serial version: the Montgomery trick is
    // exact for any chunking, and the chunks are independent row ranges.
    let combine_inverse_filter_into = |instance: &[Vec<F>], filter: &[F], out: &mut [F]| {
        let num_chunks = if out.len() >= (1 << 15) { 16 } else { 1 };
        let chunk_len = out.len().div_ceil(num_chunks);
        out.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(ci, combined)| {
                let lo = ci * chunk_len;
                if instance.len() == 1 {
                    for (o, &x) in combined.iter_mut().zip(&instance[0][lo..]) {
                        *o = x + challenge.gamma;
                    }
                } else {
                    for (d, o) in combined.iter_mut().enumerate() {
                        *o = challenge.combine_base(instance.iter().map(|vals| vals[lo + d]));
                    }
                }
                let inverse = F::batch_multiplicative_inverse(combined);
                combined.copy_from_slice(&inverse);
                for (o, &f) in combined.iter_mut().zip(&filter[lo..]) {
                    *o *= f;
                }
            });
    };

    let chunks = (0..columns.len()).step_by(group);
    // Helper columns: one per chunk of `group` instances. Chunks are
    // independent — each accumulates its instances into its own pair of
    // row buffers — so they run in parallel; within a chunk the instances
    // stay serial (they share the combine buffer), but every buffer pass
    // (combine+inverse+filter, and the accumulate into the helper column)
    // is row-parallel.
    let chunk_starts: Vec<usize> = chunks.collect();
    let mut helper_columns: Vec<PolynomialValues<F>> = chunk_starts
        .into_par_iter()
        .map(|chunk_start| {
            let instances = &columns[chunk_start..(chunk_start + group).min(columns.len())];
            let chunk_filters = &filters[chunk_start..(chunk_start + group).min(filters.len())];
            let mut combined = vec![F::ZERO; degree];
            let mut acc = vec![F::ZERO; degree];
            for (k, (instance, filter)) in instances.iter().zip(chunk_filters).enumerate() {
                combine_inverse_filter_into(instance, filter, &mut combined);
                acc.par_chunks_mut(CTL_ACC_CHUNK)
                    .zip(combined.par_chunks(CTL_ACC_CHUNK))
                    .for_each(|(a, c)| {
                        if k == 0 {
                            a.copy_from_slice(c);
                        } else {
                            for (a, c) in a.iter_mut().zip(c) {
                                *a += *c;
                            }
                        }
                    });
            }
            acc.into()
        })
        .collect();

    // Z suffix sums: `z[degree-1] = sum of the last helper values`, then each
    // earlier `z[i] = z[i+1] + sum_h h[i]` — i.e. `z[i] = sum_{j>=i}` of the
    // per-row helper sums. The per-row sums parallelize over rows; the suffix
    // scan itself uses a blocked two-level scheme: parallel block sums, a
    // short serial scan over the block totals, then parallel in-block scans.
    let num_helpers = helper_columns.len();
    let per_row_sums: Vec<F> = if num_helpers == 1 {
        helper_columns[0].values.clone()
    } else {
        (0..degree)
            .into_par_iter()
            .with_min_len(1024)
            .map(|i| helper_columns.iter().map(|col| col.values[i]).sum::<F>())
            .collect()
    };
    let mut z = vec![F::ZERO; degree];
    let block_len = degree.div_ceil(CTL_SCAN_BLOCKS).max(1);
    let num_blocks = degree.div_ceil(block_len);
    // Parallel block totals…
    let mut block_totals = vec![F::ZERO; num_blocks];
    block_totals
        .par_iter_mut()
        .enumerate()
        .for_each(|(b, total)| {
            let lo = b * block_len;
            let hi = ((b + 1) * block_len).min(degree);
            *total = per_row_sums[lo..hi].iter().copied().sum();
        });
    // …a serial suffix scan over the (few) block totals: tails[b] holds the
    // sum of every row after block b…
    let mut tails = vec![F::ZERO; num_blocks + 1];
    for b in (0..num_blocks).rev() {
        tails[b] = tails[b + 1] + block_totals[b];
    }
    // …then the in-block suffix scans in parallel.
    z.par_chunks_mut(block_len).enumerate().for_each(|(b, zb)| {
        let lo = b * block_len;
        let mut acc = tails[b + 1];
        let sums = &per_row_sums[lo..lo + zb.len()];
        for i in (0..zb.len()).rev() {
            acc += sums[i];
            zb[i] = acc;
        }
    });

    if columns.len() > 1 {
        helper_columns.push(z.into());
    } else {
        helper_columns = vec![z.into()];
    }

    helper_columns
}

/// Data necessary to check the cross-table lookups of a given table.
#[derive(Clone, Debug)]
pub struct CtlCheckVars<'a, F, FE, P, const D2: usize>
where
    F: Field,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    /// Helper columns to check that the Z polyomial
    /// was constructed correctly.
    pub(crate) helper_columns: Vec<P>,
    /// Evaluation of the trace polynomials at point `zeta`.
    pub(crate) local_z: P,
    /// Evaluation of the trace polynomials at point `g * zeta`
    pub(crate) next_z: P,
    /// Cross-table lookup challenges.
    pub(crate) challenges: GrandProductChallenge<F>,
    /// Column linear combinations of the `CrossTableLookup`s.
    pub(crate) columns: Vec<&'a [Column<F>]>,
    /// Filter that evaluates to either 1 or 0.
    pub(crate) filter: Vec<Filter<F>>,
}

impl<'a, F: RichField + Extendable<D>, const D: usize>
    CtlCheckVars<'a, F, F::Extension, F::Extension, D>
{
    /// Extracts the `CtlCheckVars` from a single proof.
    pub fn from_proof<C: GenericConfig<D, F = F>>(
        table_idx: TableIdx,
        proof: &StarkProof<F, C, D>,
        cross_table_lookups: &'a [CrossTableLookup<F>],
        ctl_challenges: &'a GrandProductChallengeSet<F>,
        num_lookup_columns: usize,
        total_num_helper_columns: usize,
        num_helper_ctl_columns: &[[usize; 2]],
    ) -> Vec<Self> {
        Self::from_openings(
            table_idx,
            &proof.openings,
            cross_table_lookups,
            ctl_challenges,
            num_lookup_columns,
            total_num_helper_columns,
            num_helper_ctl_columns,
        )
    }

    /// Extracts the `CtlCheckVars` from a STARK opening set.
    pub fn from_openings(
        table_idx: TableIdx,
        openings: &StarkOpeningSet<F, D>,
        cross_table_lookups: &'a [CrossTableLookup<F>],
        ctl_challenges: &'a GrandProductChallengeSet<F>,
        num_lookup_columns: usize,
        total_num_helper_columns: usize,
        num_helper_ctl_columns: &[[usize; 2]],
    ) -> Vec<Self> {
        // Get all cross-table lookup polynomial openings for the provided STARK opening set.
        let ctl_zs = {
            let auxiliary_polys = openings
                .auxiliary_polys
                .as_ref()
                .expect("We cannot have CTLs without auxiliary polynomials.");
            let auxiliary_polys_next = openings
                .auxiliary_polys_next
                .as_ref()
                .expect("We cannot have CTLs without auxiliary polynomials.");

            auxiliary_polys
                .iter()
                .skip(num_lookup_columns)
                .zip(auxiliary_polys_next.iter().skip(num_lookup_columns))
                .collect::<Vec<_>>()
        };

        let mut z_index = 0;
        let mut start_index = 0;
        let mut ctl_vars = vec![];

        for (
            i,
            CrossTableLookup {
                looking_tables,
                looked_tables,
            },
        ) in cross_table_lookups.iter().enumerate()
        {
            for &challenges in &ctl_challenges.challenges {
                // Mirror the prover's `cross_table_lookup_data`: per challenge, the looking
                // side's per-trace batches come first, then the looked side's. On each side,
                // group this table's instances together (they share helper columns and a Z).
                for (side, tables) in [looking_tables, looked_tables].into_iter().enumerate() {
                    let count = tables.iter().filter(|twc| twc.table == table_idx).count();
                    if count == 0 {
                        continue;
                    }
                    let mut columns = Vec::with_capacity(count);
                    let mut filter = Vec::with_capacity(count);
                    for twc in tables.iter().filter(|twc| twc.table == table_idx) {
                        columns.push(&twc.columns[..]);
                        filter.push(twc.filter.clone());
                    }

                    let (local_z, next_z) = ctl_zs[total_num_helper_columns + z_index];
                    let helper_columns = ctl_zs
                        [start_index..start_index + num_helper_ctl_columns[i][side]]
                        .iter()
                        .map(|&(h, _)| *h)
                        .collect::<Vec<_>>();

                    start_index += num_helper_ctl_columns[i][side];
                    z_index += 1;

                    ctl_vars.push(Self {
                        helper_columns,
                        local_z: *local_z,
                        next_z: *next_z,
                        challenges,
                        columns,
                        filter,
                    });
                }
            }
        }

        ctl_vars
    }
}

/// Checks the cross-table lookup Z polynomials for each table:
/// - Checks that the CTL `Z` partial sums are correctly updated.
/// - Checks that the final value of the CTL sum is the combination of all STARKs' CTL polynomials.
///
/// CTL `Z` partial sums are upside down: the complete sum is on the first row, and
/// the first term is on the last row. This allows the transition constraint to be:
/// `combine(w) * (Z(w) - Z(gw)) = filter` where combine is called on the local row
/// and not the next. This enables CTLs across two rows.
pub(crate) fn eval_cross_table_lookup_checks<F, FE, P, S, const D: usize, const D2: usize>(
    vars: &S::EvaluationFrame<FE, P, D2>,
    ctl_vars: &[CtlCheckVars<F, FE, P, D2>],
    consumer: &mut ConstraintConsumer<P>,
    constraint_degree: usize,
) where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
    S: Stark<F, D>,
{
    let local_values = vars.get_local_values();
    let next_values = vars.get_next_values();
    let public_inputs = vars.get_public_inputs();

    for lookup_vars in ctl_vars {
        let CtlCheckVars {
            helper_columns,
            local_z,
            next_z,
            challenges,
            columns,
            filter,
        } = lookup_vars;

        // Challenge-combine a column tuple on demand; each tuple is consumed at most
        // once below, so no per-point tuple-evaluation vector is materialized.
        let combine_tuple = |i: usize| {
            challenges.combine_owned(
                columns[i]
                    .iter()
                    .map(|c| c.eval_with_next(local_values, next_values, public_inputs)),
            )
        };

        // Check helper columns.
        eval_helper_columns(
            filter,
            columns,
            local_values,
            next_values,
            public_inputs,
            helper_columns,
            constraint_degree,
            challenges,
            consumer,
        );

        if !helper_columns.is_empty() {
            let h_sum = helper_columns.iter().fold(P::ZEROS, |acc, x| acc + *x);
            // Check value of `Z(g^(n-1))`
            consumer.constraint_last_row(*local_z - h_sum);
            // Check `Z(w) = Z(gw) + \sum h_i`
            consumer.constraint_transition(*local_z - *next_z - h_sum);
        } else if columns.len() > 1 {
            let combin0 = combine_tuple(0);
            let combin1 = combine_tuple(1);

            let f0 = filter[0].eval_filter(local_values, next_values, public_inputs);
            let f1 = filter[1].eval_filter(local_values, next_values, public_inputs);

            consumer
                .constraint_last_row(combin0 * combin1 * *local_z - f0 * combin1 - f1 * combin0);
            consumer.constraint_transition(
                combin0 * combin1 * (*local_z - *next_z) - f0 * combin1 - f1 * combin0,
            );
        } else {
            let combin0 = combine_tuple(0);
            let f0 = filter[0].eval_filter(local_values, next_values, public_inputs);
            consumer.constraint_last_row(combin0 * *local_z - f0);
            consumer.constraint_transition(combin0 * (*local_z - *next_z) - f0);
        }
    }
}

/// Circuit version of `CtlCheckVars`. Data necessary to check the cross-table lookups of a given table.
#[derive(Clone, Debug)]
pub struct CtlCheckVarsTarget<F: Field, const D: usize> {
    ///Evaluation of the helper columns to check that the Z polyomial
    /// was constructed correctly.
    pub(crate) helper_columns: Vec<ExtensionTarget<D>>,
    /// Evaluation of the trace polynomials at point `zeta`.
    pub(crate) local_z: ExtensionTarget<D>,
    /// Evaluation of the trace polynomials at point `g * zeta`.
    pub(crate) next_z: ExtensionTarget<D>,
    /// Cross-table lookup challenges.
    pub(crate) challenges: GrandProductChallenge<Target>,
    /// Column linear combinations of the `CrossTableLookup`s.
    pub(crate) columns: Vec<Vec<Column<F>>>,
    /// Filter that evaluates to either 1 or 0.
    pub(crate) filter: Vec<Filter<F>>,
}

impl<'a, F: Field, const D: usize> CtlCheckVarsTarget<F, D> {
    /// Circuit version of `from_proof`, for a single STARK.
    pub fn from_proof(
        table: TableIdx,
        proof: &StarkProofTarget<D>,
        cross_table_lookups: &'a [CrossTableLookup<F>],
        ctl_challenges: &'a GrandProductChallengeSet<Target>,
        num_lookup_columns: usize,
        total_num_helper_columns: usize,
        num_helper_ctl_columns: &[[usize; 2]],
    ) -> Vec<Self> {
        Self::from_openings(
            table,
            &proof.openings,
            cross_table_lookups,
            ctl_challenges,
            num_lookup_columns,
            total_num_helper_columns,
            num_helper_ctl_columns,
        )
    }

    /// Circuit version of `from_openings`, for a single STARK.
    pub fn from_openings(
        table: TableIdx,
        openings: &StarkOpeningSetTarget<D>,
        cross_table_lookups: &'a [CrossTableLookup<F>],
        ctl_challenges: &'a GrandProductChallengeSet<Target>,
        num_lookup_columns: usize,
        total_num_helper_columns: usize,
        num_helper_ctl_columns: &[[usize; 2]],
    ) -> Vec<Self> {
        // Get all cross-table lookup polynomial openings.
        let ctl_zs = {
            let ctl_zs = openings
                .auxiliary_polys
                .as_ref()
                .expect("We cannot have CTLs without auxiliary polynomials.")
                .iter()
                .skip(num_lookup_columns);
            let ctl_zs_next = openings
                .auxiliary_polys_next
                .as_ref()
                .expect("We cannot have CTLs without auxiliary polynomials.")
                .iter()
                .skip(num_lookup_columns);
            ctl_zs.zip(ctl_zs_next).collect::<Vec<_>>()
        };

        // Put each cross-table lookup polynomial into the correct table's data.
        // If a CTL polynomial is extracted from the looking/looked table `t``,
        // then we add it to the `CtlCheckVars` of table `t``.
        let mut z_index = 0;
        let mut start_index = 0;
        let mut ctl_vars = vec![];
        for (
            i,
            CrossTableLookup {
                looking_tables,
                looked_tables,
            },
        ) in cross_table_lookups.iter().enumerate()
        {
            for &challenges in &ctl_challenges.challenges {
                // Mirror the prover's `cross_table_lookup_data`: per challenge, the looking
                // side's per-trace batches come first, then the looked side's. On each side,
                // group this table's instances together (they share helper columns and a Z).
                for (side, tables) in [looking_tables, looked_tables].into_iter().enumerate() {
                    let count = tables.iter().filter(|twc| twc.table == table).count();
                    if count == 0 {
                        continue;
                    }
                    let mut columns = Vec::with_capacity(count);
                    let mut filter = Vec::with_capacity(count);
                    for twc in tables.iter().filter(|twc| twc.table == table) {
                        columns.push(twc.columns.clone());
                        filter.push(twc.filter.clone());
                    }
                    let (local_z, next_z) = ctl_zs[total_num_helper_columns + z_index];
                    let helper_columns = ctl_zs
                        [start_index..start_index + num_helper_ctl_columns[i][side]]
                        .iter()
                        .map(|(&h, _)| h)
                        .collect::<Vec<_>>();

                    start_index += num_helper_ctl_columns[i][side];
                    z_index += 1;
                    ctl_vars.push(Self {
                        helper_columns,
                        local_z: *local_z,
                        next_z: *next_z,
                        challenges,
                        columns,
                        filter,
                    });
                }
            }
        }

        ctl_vars
    }
}

/// Circuit version of `eval_cross_table_lookup_checks`. Checks the cross-table lookup Z polynomials for each table:
/// - Checks that the CTL `Z` partial sums are correctly updated.
/// - Checks that the final value of the CTL sum is the combination of all STARKs' CTL polynomials.
///
/// CTL `Z` partial sums are upside down: the complete sum is on the first row, and
/// the first term is on the last row. This allows the transition constraint to be:
/// `combine(w) * (Z(w) - Z(gw)) = filter` where combine is called on the local row
/// and not the next. This enables CTLs across two rows.
pub(crate) fn eval_cross_table_lookup_checks_circuit<
    S: Stark<F, D>,
    F: RichField + Extendable<D>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    vars: &S::EvaluationFrameTarget,
    ctl_vars: &[CtlCheckVarsTarget<F, D>],
    consumer: &mut RecursiveConstraintConsumer<F, D>,
    constraint_degree: usize,
) {
    let local_values = vars.get_local_values();
    let next_values = vars.get_next_values();
    let public_inputs = vars.get_public_inputs();

    for lookup_vars in ctl_vars {
        let CtlCheckVarsTarget {
            helper_columns,
            local_z,
            next_z,
            challenges,
            columns,
            filter,
        } = lookup_vars;

        // Compute all linear combinations on the current table, and combine them using the challenge.
        let evals = columns
            .iter()
            .map(|col| {
                col.iter()
                    .map(|c| {
                        c.eval_with_next_circuit(builder, local_values, next_values, public_inputs)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Check helper columns.
        eval_helper_columns_circuit(
            builder,
            filter,
            &evals,
            local_values,
            next_values,
            public_inputs,
            helper_columns,
            constraint_degree,
            challenges,
            consumer,
        );

        let z_diff = builder.sub_extension(*local_z, *next_z);
        if !helper_columns.is_empty() {
            // Check value of `Z(g^(n-1))`
            let h_sum = builder.add_many_extension(helper_columns);

            let last_row = builder.sub_extension(*local_z, h_sum);
            consumer.constraint_last_row(builder, last_row);
            // Check `Z(w) = Z(gw) * (filter / combination)`

            let transition = builder.sub_extension(z_diff, h_sum);
            consumer.constraint_transition(builder, transition);
        } else if columns.len() > 1 {
            let combin0 = challenges.combine_circuit(builder, &evals[0]);
            let combin1 = challenges.combine_circuit(builder, &evals[1]);

            let f0 =
                filter[0].eval_filter_circuit(builder, local_values, next_values, public_inputs);
            let f1 =
                filter[1].eval_filter_circuit(builder, local_values, next_values, public_inputs);

            let combined = builder.mul_sub_extension(combin1, *local_z, f1);
            let combined = builder.mul_extension(combined, combin0);
            let constr = builder.arithmetic_extension(F::NEG_ONE, F::ONE, f0, combin1, combined);
            consumer.constraint_last_row(builder, constr);

            let combined = builder.mul_sub_extension(combin1, z_diff, f1);
            let combined = builder.mul_extension(combined, combin0);
            let constr = builder.arithmetic_extension(F::NEG_ONE, F::ONE, f0, combin1, combined);
            // Mirrors the native evaluator: `combin0 * combin1 * (Z(w) - Z(gw)) = f0 * combin1
            // + f1 * combin0` is the *transition* relation (the last-row check above pins
            // `Z(g^(n-1))`).
            consumer.constraint_transition(builder, constr);
        } else {
            let combin0 = challenges.combine_circuit(builder, &evals[0]);
            let f0 =
                filter[0].eval_filter_circuit(builder, local_values, next_values, public_inputs);

            let constr = builder.mul_sub_extension(combin0, *local_z, f0);
            consumer.constraint_last_row(builder, constr);
            let constr = builder.mul_sub_extension(combin0, z_diff, f0);
            consumer.constraint_transition(builder, constr);
        }
    }
}

/// Verifies all cross-table lookups.
/// The key of `ctl_extra_looking_sums` is the corresponding CTL's position within `cross_table_lookups`.
pub fn verify_cross_table_lookups<F: RichField + Extendable<D>, const D: usize, const N: usize>(
    cross_table_lookups: &[CrossTableLookup<F>],
    ctl_zs_first: [Vec<F>; N],
    ctl_extra_looking_sums: &HashMap<usize, Vec<F>>,
    config: &StarkConfig,
) -> Result<()> {
    let mut ctl_zs_openings = ctl_zs_first.iter().map(|v| v.iter()).collect::<Vec<_>>();
    for (
        index,
        CrossTableLookup {
            looking_tables,
            looked_tables,
        },
    ) in cross_table_lookups.iter().enumerate()
    {
        let ctl_extra_looking_sum = ctl_extra_looking_sums.get(&index);
        // We want to iterate on each trace only once per side (its instances share one Z).
        let mut filtered_looking_tables = vec![];
        for table in looking_tables {
            if !filtered_looking_tables.contains(&(table.table)) {
                filtered_looking_tables.push(table.table);
            }
        }
        let mut filtered_looked_tables = vec![];
        for table in looked_tables {
            if !filtered_looked_tables.contains(&(table.table)) {
                filtered_looked_tables.push(table.table);
            }
        }
        for c in 0..config.num_challenges {
            // Compute the combination of all looking table CTL polynomial openings.

            let looking_zs_sum = filtered_looking_tables
                .iter()
                .map(|&table| *ctl_zs_openings[table].next().unwrap())
                .sum::<F>()
                // Get elements looking into the looked tables that are not associated to any STARK.
                + ctl_extra_looking_sum.map(|v| v[c]).unwrap_or_default();

            // Compute the combination of all looked table CTL polynomial openings.
            let looked_zs_sum = filtered_looked_tables
                .iter()
                .map(|&table| *ctl_zs_openings[table].next().unwrap())
                .sum::<F>();
            // Ensure that the combination of looking table openings is equal to the
            // combination of looked table openings.
            ensure!(
                looking_zs_sum == looked_zs_sum,
                "Cross-table lookup {:?} verification failed.",
                index
            );
        }
    }
    debug_assert!(ctl_zs_openings.iter_mut().all(|iter| iter.next().is_none()));

    Ok(())
}

/// Circuit version of `verify_cross_table_lookups`. Verifies all cross-table lookups.
/// The key of `ctl_extra_looking_sums` is the corresponding CTL's position within `cross_table_lookups`.
pub fn verify_cross_table_lookups_circuit<
    F: RichField + Extendable<D>,
    const D: usize,
    const N: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    cross_table_lookups: Vec<CrossTableLookup<F>>,
    ctl_zs_first: [Vec<Target>; N],
    ctl_extra_looking_sums: &HashMap<usize, Vec<Target>>,
    inner_config: &StarkConfig,
) {
    let mut ctl_zs_openings = ctl_zs_first.iter().map(|v| v.iter()).collect::<Vec<_>>();
    for (
        index,
        CrossTableLookup {
            looking_tables,
            looked_tables,
        },
    ) in cross_table_lookups.into_iter().enumerate()
    {
        let ctl_extra_looking_sum = ctl_extra_looking_sums.get(&index);
        // We want to iterate on each trace only once per side (its instances share one Z).
        let mut filtered_looking_tables = vec![];
        for table in looking_tables {
            if !filtered_looking_tables.contains(&(table.table)) {
                filtered_looking_tables.push(table.table);
            }
        }
        let mut filtered_looked_tables = vec![];
        for table in looked_tables {
            if !filtered_looked_tables.contains(&(table.table)) {
                filtered_looked_tables.push(table.table);
            }
        }
        for c in 0..inner_config.num_challenges {
            // Compute the combination of all looking table CTL polynomial openings.
            let mut looking_zs_sum = builder.add_many(
                filtered_looking_tables
                    .iter()
                    .map(|&table| *ctl_zs_openings[table].next().unwrap()),
            );

            // Get elements looking into the looked tables that are not associated to any STARK.
            // Note: do not use `unwrap_or_default()` here, as `Target::default()` is an
            // arbitrary target, not the zero constant.
            if let Some(extra_sum) = ctl_extra_looking_sum.map(|v| v[c]) {
                looking_zs_sum = builder.add(looking_zs_sum, extra_sum);
            }

            // Compute the combination of all looked table CTL polynomial openings.
            let looked_zs_sum = builder.add_many(
                filtered_looked_tables
                    .iter()
                    .map(|&table| *ctl_zs_openings[table].next().unwrap()),
            );
            // Verify that the combination of looking table openings is equal to the
            // combination of looked table openings.
            builder.connect(looked_zs_sum, looking_zs_sum);
        }
    }
    debug_assert!(ctl_zs_openings.iter_mut().all(|iter| iter.next().is_none()));
}

/// Debugging module used to assert correctness of the different CTLs of a multi-STARK system,
/// that can be used during the proof generation process.
///
/// **Note**: This is an expensive check.
pub mod debug_utils {
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    use hashbrown::HashMap;
    use plonky2::field::polynomial::PolynomialValues;
    use plonky2::field::types::{Field, PrimeField64};

    use super::{CrossTableLookup, TableIdx, TableWithColumns};

    type MultiSet<F> = HashMap<Vec<F>, Vec<(TableIdx, usize)>>;

    /// Check that the provided traces and cross-table lookups are consistent.
    /// `public_inputs` are the per-table public inputs (the column expressions may carry
    /// public-input terms).
    /// The key of `extra_looking_values` is the corresponding CTL's position within `cross_table_lookups`.
    /// Filter values are interpreted as row multiplicities (matching the prover/constraint
    /// semantics, where the filter is the logup numerator); they must be small integers.
    pub fn check_ctls<F: PrimeField64>(
        trace_poly_values: &[Vec<PolynomialValues<F>>],
        public_inputs: &[Vec<F>],
        cross_table_lookups: &[CrossTableLookup<F>],
        extra_looking_values: &HashMap<usize, Vec<Vec<F>>>,
    ) {
        for (i, ctl) in cross_table_lookups.iter().enumerate() {
            check_ctl(
                trace_poly_values,
                public_inputs,
                ctl,
                i,
                extra_looking_values.get(&i),
            );
        }
    }

    fn check_ctl<F: PrimeField64>(
        trace_poly_values: &[Vec<PolynomialValues<F>>],
        public_inputs: &[Vec<F>],
        ctl: &CrossTableLookup<F>,
        ctl_index: usize,
        extra_looking_values: Option<&Vec<Vec<F>>>,
    ) {
        let CrossTableLookup {
            looking_tables,
            looked_tables,
        } = ctl;

        // Maps `m` with `(table, i) in m[row]` iff the `i`-th row of `table` is equal to `row` and
        // the filter is 1. Without default values, the CTL check holds iff `looking_multiset == looked_multiset`.
        let mut looking_multiset = MultiSet::<F>::new();
        let mut looked_multiset = MultiSet::<F>::new();

        for table in looking_tables {
            process_table(
                trace_poly_values,
                public_inputs,
                table,
                &mut looking_multiset,
            );
        }
        for table in looked_tables {
            process_table(
                trace_poly_values,
                public_inputs,
                table,
                &mut looked_multiset,
            );
        }

        // Include extra looking values if any for this `ctl_index`.
        if let Some(values) = extra_looking_values {
            for row in values.iter() {
                // The table and the row index don't matter here, as we just want to enforce
                // that the special extra values do appear when looking against the specified table.
                looking_multiset
                    .entry(row.to_vec())
                    .or_default()
                    .push((0, 0));
            }
        }

        let empty = &vec![];
        // Check that every row in the looking tables appears in the looked table the same number of times.
        for (row, looking_locations) in &looking_multiset {
            let looked_locations = looked_multiset.get(row).unwrap_or(empty);
            check_locations(looking_locations, looked_locations, ctl_index, row);
        }
        // Check that every row in the looked tables appears in the looked table the same number of times.
        for (row, looked_locations) in &looked_multiset {
            let looking_locations = looking_multiset.get(row).unwrap_or(empty);
            check_locations(looking_locations, looked_locations, ctl_index, row);
        }
    }

    fn process_table<F: PrimeField64>(
        trace_poly_values: &[Vec<PolynomialValues<F>>],
        public_inputs: &[Vec<F>],
        table: &TableWithColumns<F>,
        multiset: &mut MultiSet<F>,
    ) {
        let trace = &trace_poly_values[table.table];
        let pis = &public_inputs[table.table];
        for i in 0..trace[0].len() {
            let filter = table.filter.eval_table(trace, i, pis);
            // The filter value is the row's multiplicity — it enters the lookup argument as
            // the logup numerator (`get_helper_cols` / `eval_helper_columns` are algebraic in
            // it): 0/1 are the usual off/on cases, small integers > 1 make one row stand for
            // that many copies of its tuple (e.g. a looked-side multiplicity).
            let multiplicity = filter.to_canonical_u64();
            // The bound only guards this checker: the branch below materializes one entry per
            // copy, so a data column mistakenly wired in as a filter (a near-arbitrary field
            // element) would try to allocate ~2^63 tuples. Padded traces make a shared LUT's
            // zero key legitimately reach ~2^21 lookups, hence the headroom.
            assert!(
                multiplicity <= 1 << 24,
                "filter value {filter:?} is not a small multiplicity"
            );
            if multiplicity > 0 {
                let row = table
                    .columns
                    .iter()
                    .map(|c| c.eval_table(trace, i, pis))
                    .collect::<Vec<_>>();
                multiset
                    .entry(row)
                    .or_default()
                    .extend(core::iter::repeat_n(
                        (table.table, i),
                        multiplicity as usize,
                    ));
            }
        }
    }

    fn check_locations<F: Field>(
        looking_locations: &[(TableIdx, usize)],
        looked_locations: &[(TableIdx, usize)],
        ctl_index: usize,
        row: &[F],
    ) {
        if looking_locations.len() != looked_locations.len() {
            panic!(
                "CTL #{ctl_index}:\n\
                 Row {row:?} is present {l0} times in the looking tables, but {l1} times in the looked table.\n\
                 Looking locations (Table, Row index): {looking_locations:?}.\n\
                 Looked locations (Table, Row index): {looked_locations:?}.",
                l0 = looking_locations.len(),
                l1 = looked_locations.len(),
            );
        }
    }
}
