//! Support for proving multiple STARK tables with a single batched FRI
//! argument.
//!
//! This module provides:
//! - [`BatchStark`], an object-safe view of a [`Stark`] (which itself is not
//!   object-safe due to its associated constants and generic methods). It is
//!   blanket-implemented for every `S: Stark<F, D>`, so heterogeneous tables
//!   can be handled uniformly as `&dyn BatchStark<F, D>`.
//! - [`BatchStarkLayout`], the polynomial layout shared by the batch prover,
//!   verifier and recursive verifier: which polynomial of which table lives
//!   where in the batched FRI oracles, and how the FRI instances, openings
//!   and claimed evaluations are ordered.
//!
//! Batched proving commits one Merkle tree per table per role (trace,
//! preprocessed, auxiliary, quotient — skipping a table that has no
//! polynomials of that role), except that a designated set of *grouped*
//! tables shares one multi-height tree per role (like the setup-time
//! preprocessed oracle). Fiat-Shamir observes the solo caps in table-index
//! order within each role, then the role's grouped cap. FRI is unchanged:
//! one instance per distinct degree, covering every table of that degree
//! from its oracles.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{ensure, Result};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packable::Packable;
use plonky2::field::types::Field;
use plonky2::fri::structure::{
    FriBatchInfo, FriBatchInfoTarget, FriInstanceInfo, FriInstanceInfoTarget, FriOpeningBatch,
    FriOpeningBatchTarget, FriOpenings, FriOpeningsTarget, FriOracleInfo, FriPolynomialInfo,
};
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::config::StarkConfig;
use crate::constraint_consumer::ConstraintConsumer;
use crate::cross_table_lookup::{CrossTableLookup, CtlCheckVars, CtlCheckVarsTarget};
use crate::evaluation_frame::StarkEvaluationFrame;
use crate::lookup::{Lookup, LookupCheckVars};
use crate::proof::{StarkOpeningSet, StarkOpeningSetTarget};
use crate::stark::Stark;
use crate::vanishing_poly::{
    compute_eval_vanishing_poly, compute_eval_vanishing_poly_circuit, eval_vanishing_poly,
};

/// Object-safe view of a [`Stark`], usable as `&dyn BatchStark<F, D>` in the
/// batch prover and verifiers. It is blanket-implemented for every
/// `S: Stark<F, D>`.
pub trait BatchStark<F: RichField + Extendable<D>, const D: usize>: Sync {
    /// The total number of columns in the trace (`Stark::COLUMNS`).
    fn num_columns(&self) -> usize;

    /// The total number of public inputs (`Stark::PUBLIC_INPUTS`).
    fn num_public_inputs(&self) -> usize;

    /// The maximum constraint degree of this table.
    fn constraint_degree(&self) -> usize;

    /// The maximum quotient polynomial degree factor of this table.
    fn quotient_degree_factor(&self) -> usize;

    /// The number of quotient polynomials committed for this table.
    fn num_quotient_polys(&self, config: &StarkConfig) -> usize;

    /// All the [`Lookup`]s performed by this table.
    fn lookups(&self) -> Vec<Lookup<F>>;

    /// The total number of lookup helper columns of this table.
    fn num_lookup_helper_columns(&self, config: &StarkConfig) -> usize;

    /// Whether this table uses lookups.
    fn uses_lookups(&self) -> bool;

    /// Whether this table participates in cross-table lookups.
    fn requires_ctls(&self) -> bool;

    /// Evaluates all constraints (table, lookups and CTLs) at an extension
    /// point given the claimed openings, returning the `alpha`-combined
    /// accumulators. This is the object-safe counterpart of
    /// `compute_eval_vanishing_poly`.
    fn eval_vanishing_poly_ext(
        &self,
        opening_set: &StarkOpeningSet<F, D>,
        ctl_vars: Option<&[CtlCheckVars<F, F::Extension, F::Extension, D>]>,
        lookup_challenges: Option<&Vec<F>>,
        public_inputs: &[F],
        alphas: Vec<F>,
        zeta: F::Extension,
        degree_bits: usize,
        num_lookup_columns: usize,
    ) -> Vec<F::Extension>;

    /// Evaluates all constraints (table, lookups and CTLs) on a batch of
    /// packed base-field points, accumulating them into `consumer`. Used
    /// during quotient polynomial computation.
    fn eval_vanishing_poly_packed(
        &self,
        local_values: &[<F as Packable>::Packing],
        next_values: &[<F as Packable>::Packing],
        public_inputs: &[F],
        lookups: &[Lookup<F>],
        lookup_vars: Option<LookupCheckVars<F, F, <F as Packable>::Packing, 1>>,
        ctl_vars: Option<&[CtlCheckVars<F, F, <F as Packable>::Packing, 1>]>,
        consumer: &mut ConstraintConsumer<<F as Packable>::Packing>,
    );

    /// Circuit version of [`Self::eval_vanishing_poly_ext`]. This is the
    /// object-safe counterpart of `compute_eval_vanishing_poly_circuit`.
    fn eval_vanishing_poly_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        opening_set: &StarkOpeningSetTarget<D>,
        ctl_vars: Option<&[CtlCheckVarsTarget<F, D>]>,
        lookup_challenges: Option<&Vec<Target>>,
        public_inputs: &[Target],
        alphas: Vec<Target>,
        zeta: ExtensionTarget<D>,
        degree_bits: usize,
        degree_bits_target: Target,
        num_lookup_columns: usize,
    ) -> Vec<ExtensionTarget<D>>;
}

impl<F, S, const D: usize> BatchStark<F, D> for S
where
    F: RichField + Extendable<D>,
    S: Stark<F, D>,
{
    fn num_columns(&self) -> usize {
        S::COLUMNS
    }

    fn num_public_inputs(&self) -> usize {
        S::PUBLIC_INPUTS
    }

    fn constraint_degree(&self) -> usize {
        Stark::constraint_degree(self)
    }

    fn quotient_degree_factor(&self) -> usize {
        Stark::quotient_degree_factor(self)
    }

    fn num_quotient_polys(&self, config: &StarkConfig) -> usize {
        Stark::num_quotient_polys(self, config)
    }

    fn lookups(&self) -> Vec<Lookup<F>> {
        Stark::lookups(self)
    }

    fn num_lookup_helper_columns(&self, config: &StarkConfig) -> usize {
        Stark::num_lookup_helper_columns(self, config)
    }

    fn uses_lookups(&self) -> bool {
        Stark::uses_lookups(self)
    }

    fn requires_ctls(&self) -> bool {
        Stark::requires_ctls(self)
    }

    fn eval_vanishing_poly_ext(
        &self,
        opening_set: &StarkOpeningSet<F, D>,
        ctl_vars: Option<&[CtlCheckVars<F, F::Extension, F::Extension, D>]>,
        lookup_challenges: Option<&Vec<F>>,
        public_inputs: &[F],
        alphas: Vec<F>,
        zeta: F::Extension,
        degree_bits: usize,
        num_lookup_columns: usize,
    ) -> Vec<F::Extension> {
        compute_eval_vanishing_poly::<F, S, D>(
            self,
            opening_set,
            ctl_vars,
            lookup_challenges,
            &Stark::lookups(self),
            public_inputs,
            alphas,
            zeta,
            degree_bits,
            num_lookup_columns,
        )
    }

    fn eval_vanishing_poly_packed(
        &self,
        local_values: &[<F as Packable>::Packing],
        next_values: &[<F as Packable>::Packing],
        public_inputs: &[F],
        lookups: &[Lookup<F>],
        lookup_vars: Option<LookupCheckVars<F, F, <F as Packable>::Packing, 1>>,
        ctl_vars: Option<&[CtlCheckVars<F, F, <F as Packable>::Packing, 1>]>,
        consumer: &mut ConstraintConsumer<<F as Packable>::Packing>,
    ) {
        let vars = S::EvaluationFrame::from_values(local_values, next_values, public_inputs);
        eval_vanishing_poly::<F, F, <F as Packable>::Packing, S, D, 1>(
            self,
            &vars,
            lookups,
            lookup_vars,
            ctl_vars,
            consumer,
        );
    }

    fn eval_vanishing_poly_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        opening_set: &StarkOpeningSetTarget<D>,
        ctl_vars: Option<&[CtlCheckVarsTarget<F, D>]>,
        lookup_challenges: Option<&Vec<Target>>,
        public_inputs: &[Target],
        alphas: Vec<Target>,
        zeta: ExtensionTarget<D>,
        degree_bits: usize,
        degree_bits_target: Target,
        num_lookup_columns: usize,
    ) -> Vec<ExtensionTarget<D>> {
        compute_eval_vanishing_poly_circuit::<F, S, D>(
            builder,
            self,
            opening_set,
            ctl_vars,
            lookup_challenges,
            public_inputs,
            alphas,
            zeta,
            degree_bits,
            degree_bits_target,
            num_lookup_columns,
        )
    }
}

/// One FRI oracle of the batch. The order of [`BatchStarkLayout::oracles`] is
/// the oracle index space of the batched FRI argument
/// (`FriPolynomialInfo::oracle_index`).
///
/// Trace, auxiliary and quotient polynomials are committed per table by
/// default: their height grouping follows the job's degree profile, so no
/// grouping can be shared across jobs. Tables whose heights *are* shared
/// across jobs (e.g. fixed-height lookup tables) can instead be designated
/// as *grouped*: they share one multi-height tree per role, costing one
/// Merkle path per FRI query per role instead of one per table per role.
/// Preprocessed columns are committed once at setup time in a *single*
/// multi-height oracle for the same reason.
///
/// Every multi-height oracle (grouped or preprocessed) orders its
/// polynomials by descending trace degree, ties by table index; same-degree
/// tables share a Merkle leaf.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum BatchOracle {
    /// The online (non-preprocessed) trace columns of one solo table.
    Trace(usize),
    /// The online trace columns of all grouped tables, in one oracle.
    GroupedTrace,
    /// The preprocessed columns of every table, in one setup-time oracle.
    /// Polynomials are ordered by descending trace degree, ties by table
    /// index (see [`BatchStarkLayout::prep_poly_start`]).
    Preprocessed,
    /// The lookup helper and CTL helper/Z polynomials of one solo table.
    Auxiliary(usize),
    /// The auxiliary polynomials of all grouped tables, in one oracle.
    GroupedAuxiliary,
    /// The quotient polynomials of one solo table.
    Quotient(usize),
    /// The quotient polynomials of all grouped tables, in one oracle.
    GroupedQuotient,
}

/// A polynomial role within the batch: the per-table polynomial families
/// committed at proving time (preprocessed columns are a setup-time role of
/// their own).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum BatchRole {
    /// Online trace columns.
    Trace,
    /// Lookup helper and CTL helper/Z polynomials.
    Auxiliary,
    /// Quotient polynomials.
    Quotient,
}

/// Polynomial layout of a batch of STARK tables: how the polynomials of each
/// table map into the per-table FRI oracles, and how FRI instances and
/// openings are ordered. Built identically by the prover, the verifier and
/// the recursive verifier.
#[derive(Debug)]
pub(crate) struct BatchStarkLayout {
    /// Per-table trace degree bits (table-index order; not necessarily sorted).
    pub degree_bits: Vec<usize>,
    /// Distinct degree bits, descending. One FRI instance per entry.
    pub distinct_degree_bits: Vec<usize>,
    /// For each degree group, the tables of that degree (in table-index order).
    pub tables_by_group: Vec<Vec<usize>>,
    /// The FRI oracles, in oracle-index order: one trace oracle per solo
    /// table with online columns (table-index order) then the grouped trace
    /// oracle (if the group has online columns), one shared preprocessed
    /// oracle if any table has preprocessed columns, then auxiliary and
    /// quotient oracles likewise.
    pub oracles: Vec<BatchOracle>,
    /// The tables sharing one multi-height oracle per role, sorted by table
    /// index (possibly empty). All other tables are *solo*.
    pub grouped_tables: Vec<usize>,
    /// For each table, the sorted indices of its preprocessed columns.
    pub prep_columns: Vec<Vec<usize>>,
    /// For each table, the indices of its online trace columns (the
    /// complement of `prep_columns`), in increasing order.
    pub online_columns: Vec<Vec<usize>>,
    /// For each table, its total number of trace columns.
    pub num_columns: Vec<usize>,
    /// For each table, its number of lookup helper columns.
    pub num_lookup_columns: Vec<usize>,
    /// For each table, its total number of CTL helper columns.
    pub num_ctl_helpers: Vec<usize>,
    /// For each table, its number of CTL Z polynomials.
    pub num_ctl_zs: Vec<usize>,
    /// For each table, the number of CTL helper columns for each CTL, split
    /// as `[looking, looked]` (a table can carry helper columns on both sides
    /// of one CTL).
    pub num_ctl_helpers_by_ctl: Vec<Vec<[usize; 2]>>,
    /// For each table, its number of quotient polynomials.
    pub num_quotient_polys: Vec<usize>,
    /// The constraint degree used to chunk CTL helper columns (the maximum
    /// constraint degree over all tables).
    pub ctl_constraint_degree: usize,
}

impl BatchStarkLayout {
    /// Builds the layout for the given batch. `degree_bits[t]` is table `t`'s
    /// trace degree; tables need not be sorted. `prep_columns` holds, for each
    /// table, the sorted indices of its preprocessed columns (empty when the
    /// table has none). `grouped_tables` lists the tables sharing one
    /// multi-height oracle per role (sorted; pass `&[]` for all-solo). It is
    /// part of the transcript shape: prover and verifier must agree on it.
    pub(crate) fn new<F: RichField + Extendable<D>, const D: usize>(
        starks: &[&dyn BatchStark<F, D>],
        config: &StarkConfig,
        degree_bits: &[usize],
        prep_columns: &[Vec<usize>],
        cross_table_lookups: &[CrossTableLookup<F>],
        grouped_tables: &[usize],
    ) -> Result<Self> {
        let num_tables = starks.len();
        ensure!(degree_bits.len() == num_tables);
        ensure!(prep_columns.len() == num_tables);
        ensure!(
            grouped_tables.windows(2).all(|w| w[0] < w[1])
                && grouped_tables.last().is_none_or(|&t| t < num_tables),
            "Grouped table indices must be sorted, unique and in range."
        );

        let mut distinct_degree_bits = degree_bits.to_vec();
        distinct_degree_bits.sort_unstable_by(|a, b| b.cmp(a));
        distinct_degree_bits.dedup();
        let tables_by_group: Vec<Vec<usize>> = distinct_degree_bits
            .iter()
            .map(|&d| (0..num_tables).filter(|&t| degree_bits[t] == d).collect())
            .collect();

        // Split each table's columns into preprocessed and online.
        let mut online_columns: Vec<Vec<usize>> = Vec::with_capacity(num_tables);
        let mut num_columns = Vec::with_capacity(num_tables);
        for (t, prep) in prep_columns.iter().enumerate() {
            let columns = starks[t].num_columns();
            ensure!(
                prep.windows(2).all(|w| w[0] < w[1]),
                "Preprocessed column indices must be sorted and unique."
            );
            ensure!(
                prep.last().is_none_or(|&c| c < columns),
                "Preprocessed column index out of range."
            );
            online_columns.push((0..columns).filter(|c| !prep.contains(c)).collect());
            num_columns.push(columns);
        }

        // The constraint degree used to chunk CTL helper columns; all tables
        // with CTL helper columns must have this exact constraint degree.
        let ctl_constraint_degree = starks
            .iter()
            .map(|s| s.constraint_degree())
            .max()
            .unwrap_or(0);

        let mut num_lookup_columns = Vec::with_capacity(num_tables);
        let mut num_ctl_helpers = Vec::with_capacity(num_tables);
        let mut num_ctl_zs = Vec::with_capacity(num_tables);
        let mut num_ctl_helpers_by_ctl = Vec::with_capacity(num_tables);
        let mut num_quotient_polys = Vec::with_capacity(num_tables);
        for (t, stark) in starks.iter().enumerate() {
            let (helpers, zs, by_ctl) = CrossTableLookup::num_ctl_helpers_zs_all(
                cross_table_lookups,
                t,
                config.num_challenges,
                ctl_constraint_degree,
            );
            if helpers > 0 {
                ensure!(
                    stark.constraint_degree() == ctl_constraint_degree
                        && ctl_constraint_degree >= 2,
                    "Table {t} has CTL helper columns, so its constraint degree must equal \
                     the maximum constraint degree of the batch (>= 2)."
                );
            }
            ensure!(
                (zs > 0) == stark.requires_ctls(),
                "Table {t}: `requires_ctls()` must be true iff the table appears in some CTL."
            );
            // The CTL Z-consistency constraint on the last row (`combin * z - filter`,
            // multiplied by the last-row Lagrange basis) has degree ~3n, so CTL tables
            // need a quotient degree factor of at least 2.
            ensure!(
                zs == 0 || stark.constraint_degree() >= 3,
                "Table {t} participates in CTLs, so its constraint degree must be >= 3."
            );
            num_lookup_columns.push(stark.num_lookup_helper_columns(config));
            num_ctl_helpers.push(helpers);
            num_ctl_zs.push(zs);
            num_ctl_helpers_by_ctl.push(by_ctl);
            num_quotient_polys.push(stark.num_quotient_polys(config));
        }

        // Per role: solo tables with polynomials of that role (table-index
        // order), then the grouped oracle if any grouped table has some.
        let is_grouped = |t: usize| grouped_tables.binary_search(&t).is_ok();
        let mut oracles: Vec<BatchOracle> = Vec::new();
        oracles.extend(
            (0..num_tables)
                .filter(|&t| !is_grouped(t) && !online_columns[t].is_empty())
                .map(BatchOracle::Trace),
        );
        if grouped_tables
            .iter()
            .any(|&t| !online_columns[t].is_empty())
        {
            oracles.push(BatchOracle::GroupedTrace);
        }
        if prep_columns.iter().any(|p| !p.is_empty()) {
            oracles.push(BatchOracle::Preprocessed);
        }
        let num_aux = |t: usize| num_lookup_columns[t] + num_ctl_helpers[t] + num_ctl_zs[t];
        oracles.extend(
            (0..num_tables)
                .filter(|&t| !is_grouped(t) && num_aux(t) > 0)
                .map(BatchOracle::Auxiliary),
        );
        if grouped_tables.iter().any(|&t| num_aux(t) > 0) {
            oracles.push(BatchOracle::GroupedAuxiliary);
        }
        oracles.extend(
            (0..num_tables)
                .filter(|&t| !is_grouped(t) && num_quotient_polys[t] > 0)
                .map(BatchOracle::Quotient),
        );
        if grouped_tables.iter().any(|&t| num_quotient_polys[t] > 0) {
            oracles.push(BatchOracle::GroupedQuotient);
        }

        Ok(Self {
            degree_bits: degree_bits.to_vec(),
            distinct_degree_bits,
            tables_by_group,
            oracles,
            grouped_tables: grouped_tables.to_vec(),
            prep_columns: prep_columns.to_vec(),
            online_columns,
            num_columns,
            num_lookup_columns,
            num_ctl_helpers,
            num_ctl_zs,
            num_ctl_helpers_by_ctl,
            num_quotient_polys,
            ctl_constraint_degree,
        })
    }

    /// The number of tables in the batch.
    pub(crate) fn num_tables(&self) -> usize {
        self.degree_bits.len()
    }

    /// The number of auxiliary polynomials (lookup helpers + CTL helpers +
    /// CTL Zs) of table `t`.
    pub(crate) fn num_aux_polys(&self, t: usize) -> usize {
        self.num_lookup_columns[t] + self.num_ctl_helpers[t] + self.num_ctl_zs[t]
    }

    /// Whether table `t` shares the grouped role oracles.
    pub(crate) fn is_grouped(&self, t: usize) -> bool {
        self.grouped_tables.binary_search(&t).is_ok()
    }

    /// The number of polynomials table `t` commits for `role`.
    pub(crate) fn role_num_polys(&self, role: BatchRole, t: usize) -> usize {
        match role {
            BatchRole::Trace => self.online_columns[t].len(),
            BatchRole::Auxiliary => self.num_aux_polys(t),
            BatchRole::Quotient => self.num_quotient_polys[t],
        }
    }

    /// The grouped tables holding polynomials of `role`, in the grouped
    /// oracle's flat order: descending trace degree, ties by table index.
    pub(crate) fn grouped_role_tables(&self, role: BatchRole) -> Vec<usize> {
        self.tables_by_group
            .iter()
            .flatten()
            .copied()
            .filter(|&t| self.is_grouped(t) && self.role_num_polys(role, t) > 0)
            .collect()
    }

    /// Flat index of table `t`'s first polynomial within the grouped `role`
    /// oracle (cf. [`Self::prep_poly_start`]).
    pub(crate) fn grouped_poly_start(&self, role: BatchRole, t: usize) -> usize {
        debug_assert!(self.is_grouped(t) && self.role_num_polys(role, t) > 0);
        self.grouped_role_tables(role)
            .iter()
            .take_while(|&&u| u != t)
            .map(|&u| self.role_num_polys(role, u))
            .sum()
    }

    /// Position of table `t`'s `role` polynomials inside the grouped
    /// oracle's batch Merkle tree: `(index of the table's degree among the
    /// oracle's distinct degrees, offset within that degree's leaf)`
    /// (cf. [`Self::prep_leaf_position`]).
    pub(crate) fn grouped_leaf_position(&self, role: BatchRole, t: usize) -> (usize, usize) {
        debug_assert!(self.is_grouped(t) && self.role_num_polys(role, t) > 0);
        let members = self.grouped_role_tables(role);
        let mut degree_index = 0;
        let mut leaf_offset = 0;
        for (i, &u) in members.iter().enumerate() {
            if i > 0 && self.degree_bits[u] != self.degree_bits[members[i - 1]] {
                degree_index += 1;
                leaf_offset = 0;
            }
            if u == t {
                return (degree_index, leaf_offset);
            }
            leaf_offset += self.role_num_polys(role, u);
        }
        panic!("table {t} not in the grouped {role:?} oracle");
    }

    /// The grouped `role` oracle's leaf groups as `(lde height, group
    /// length)`, in tree order (descending height).
    pub(crate) fn grouped_leaf_groups(
        &self,
        role: BatchRole,
        rate_bits: usize,
    ) -> Vec<(usize, usize)> {
        let mut groups: Vec<(usize, usize)> = Vec::new();
        for &t in &self.grouped_role_tables(role) {
            let height = self.degree_bits[t] + rate_bits;
            let len = self.role_num_polys(role, t);
            match groups.last_mut() {
                Some((h, l)) if *h == height => *l += len,
                _ => groups.push((height, len)),
            }
        }
        groups
    }

    /// The table owning each flat polynomial of the grouped `role` oracle.
    pub(crate) fn grouped_flat_tables(&self, role: BatchRole) -> Vec<usize> {
        self.grouped_role_tables(role)
            .iter()
            .flat_map(|&t| core::iter::repeat_n(t, self.role_num_polys(role, t)))
            .collect()
    }

    /// The oracle index of table `t`'s `role` polynomials, and the flat
    /// index of the table's first polynomial within that oracle (0 for solo
    /// tables).
    pub(crate) fn role_position(&self, role: BatchRole, t: usize) -> (usize, usize) {
        debug_assert!(self.role_num_polys(role, t) > 0);
        let (target, start) = if self.is_grouped(t) {
            let target = match role {
                BatchRole::Trace => BatchOracle::GroupedTrace,
                BatchRole::Auxiliary => BatchOracle::GroupedAuxiliary,
                BatchRole::Quotient => BatchOracle::GroupedQuotient,
            };
            (target, self.grouped_poly_start(role, t))
        } else {
            let target = match role {
                BatchRole::Trace => BatchOracle::Trace(t),
                BatchRole::Auxiliary => BatchOracle::Auxiliary(t),
                BatchRole::Quotient => BatchOracle::Quotient(t),
            };
            (target, 0)
        };
        let oracle_index = self
            .oracles
            .iter()
            .position(|&o| o == target)
            .expect("the table has polynomials of this role, so its oracle exists");
        (oracle_index, start)
    }

    /// The number of polynomials of each FRI oracle.
    pub(crate) fn num_leaves_per_oracle(&self) -> Vec<usize> {
        self.oracles
            .iter()
            .map(|&oracle| match oracle {
                BatchOracle::Trace(t) => self.online_columns[t].len(),
                BatchOracle::Preprocessed => self.prep_columns.iter().map(Vec::len).sum(),
                BatchOracle::Auxiliary(t) => self.num_aux_polys(t),
                BatchOracle::Quotient(t) => self.num_quotient_polys[t],
                BatchOracle::GroupedTrace => self.grouped_num_polys(BatchRole::Trace),
                BatchOracle::GroupedAuxiliary => self.grouped_num_polys(BatchRole::Auxiliary),
                BatchOracle::GroupedQuotient => self.grouped_num_polys(BatchRole::Quotient),
            })
            .collect()
    }

    /// Total number of polynomials of the grouped `role` oracle.
    pub(crate) fn grouped_num_polys(&self, role: BatchRole) -> usize {
        self.grouped_tables
            .iter()
            .map(|&t| self.role_num_polys(role, t))
            .sum()
    }

    /// Trace degree bits of each FRI oracle's tallest polynomial.
    pub(crate) fn oracle_degree_bits(&self) -> Vec<usize> {
        self.oracles
            .iter()
            .map(|&oracle| match oracle {
                BatchOracle::Trace(t) | BatchOracle::Auxiliary(t) | BatchOracle::Quotient(t) => {
                    self.degree_bits[t]
                }
                BatchOracle::Preprocessed => (0..self.num_tables())
                    .filter(|&t| !self.prep_columns[t].is_empty())
                    .map(|t| self.degree_bits[t])
                    .max()
                    .expect(
                        "the preprocessed oracle exists, so some table has preprocessed columns",
                    ),
                BatchOracle::GroupedTrace => {
                    self.degree_bits[self.grouped_role_tables(BatchRole::Trace)[0]]
                }
                BatchOracle::GroupedAuxiliary => {
                    self.degree_bits[self.grouped_role_tables(BatchRole::Auxiliary)[0]]
                }
                BatchOracle::GroupedQuotient => {
                    self.degree_bits[self.grouped_role_tables(BatchRole::Quotient)[0]]
                }
            })
            .collect()
    }

    /// Flat index of table `t`'s first polynomial within the shared
    /// preprocessed oracle, whose polynomials are ordered by descending trace
    /// degree, ties by table index.
    pub(crate) fn prep_poly_start(&self, t: usize) -> usize {
        debug_assert!(!self.prep_columns[t].is_empty());
        let g = self.group_of_table(t);
        self.tables_by_group[..g]
            .iter()
            .flatten()
            .chain(self.tables_by_group[g].iter().take_while(|&&u| u != t))
            .map(|&u| self.prep_columns[u].len())
            .sum()
    }

    /// Position of table `t`'s preprocessed columns inside the shared
    /// oracle's batch Merkle tree: `(index of the table's degree among the
    /// oracle's distinct degrees, offset within that degree's leaf)`.
    pub(crate) fn prep_leaf_position(&self, t: usize) -> (usize, usize) {
        debug_assert!(!self.prep_columns[t].is_empty());
        let g = self.group_of_table(t);
        let degree_index = self.tables_by_group[..g]
            .iter()
            .filter(|group| group.iter().any(|&u| !self.prep_columns[u].is_empty()))
            .count();
        let leaf_offset = self.tables_by_group[g]
            .iter()
            .take_while(|&&u| u != t)
            .map(|&u| self.prep_columns[u].len())
            .sum();
        (degree_index, leaf_offset)
    }

    /// The degree group index of table `t`.
    pub(crate) fn group_of_table(&self, t: usize) -> usize {
        self.distinct_degree_bits
            .iter()
            .position(|&db| db == self.degree_bits[t])
            .unwrap()
    }

    /// Whether the FRI instance of group `g` has a "CTL Zs at 1" batch.
    fn group_has_ctl_batch(&self, g: usize) -> bool {
        self.tables_by_group[g]
            .iter()
            .any(|&t| self.num_ctl_zs[t] > 0)
    }

    /// Polynomials opened in one batch of FRI instance `g`. Grouped and
    /// preprocessed oracles contribute the ranges of the group's tables in
    /// table-index order; [`Self::batch_values`] mirrors this order exactly.
    fn batch_polys(&self, g: usize, with_quotient: bool) -> Vec<FriPolynomialInfo> {
        let mut polys = vec![];
        for (oracle_index, &oracle) in self.oracles.iter().enumerate() {
            match oracle {
                BatchOracle::Trace(t) if self.group_of_table(t) == g => {
                    polys.extend(FriPolynomialInfo::from_range(
                        oracle_index,
                        0..self.online_columns[t].len(),
                    ));
                }
                BatchOracle::Preprocessed => {
                    for &t in &self.tables_by_group[g] {
                        if self.prep_columns[t].is_empty() {
                            continue;
                        }
                        let start = self.prep_poly_start(t);
                        polys.extend(FriPolynomialInfo::from_range(
                            oracle_index,
                            start..start + self.prep_columns[t].len(),
                        ));
                    }
                }
                BatchOracle::Auxiliary(t) if self.group_of_table(t) == g => {
                    polys.extend(FriPolynomialInfo::from_range(
                        oracle_index,
                        0..self.num_aux_polys(t),
                    ));
                }
                BatchOracle::Quotient(t) if with_quotient && self.group_of_table(t) == g => {
                    polys.extend(FriPolynomialInfo::from_range(
                        oracle_index,
                        0..self.num_quotient_polys[t],
                    ));
                }
                BatchOracle::GroupedTrace => {
                    polys.extend(self.grouped_batch_polys(g, oracle_index, BatchRole::Trace));
                }
                BatchOracle::GroupedAuxiliary => {
                    polys.extend(self.grouped_batch_polys(g, oracle_index, BatchRole::Auxiliary));
                }
                BatchOracle::GroupedQuotient if with_quotient => {
                    polys.extend(self.grouped_batch_polys(g, oracle_index, BatchRole::Quotient));
                }
                _ => {}
            }
        }
        polys
    }

    /// The grouped `role` oracle's polynomials belonging to degree group `g`,
    /// in table-index order.
    fn grouped_batch_polys(
        &self,
        g: usize,
        oracle_index: usize,
        role: BatchRole,
    ) -> Vec<FriPolynomialInfo> {
        let mut polys = vec![];
        for &t in &self.tables_by_group[g] {
            if !self.is_grouped(t) || self.role_num_polys(role, t) == 0 {
                continue;
            }
            let start = self.grouped_poly_start(role, t);
            polys.extend(FriPolynomialInfo::from_range(
                oracle_index,
                start..start + self.role_num_polys(role, t),
            ));
        }
        polys
    }

    /// CTL Z polynomials of group `g`, opened at 1, in table-index order
    /// (matching the value order of [`Self::fri_openings`]).
    fn ctl_batch_polys(&self, g: usize) -> Vec<FriPolynomialInfo> {
        let mut polys = vec![];
        for &t in &self.tables_by_group[g] {
            if self.num_ctl_zs[t] == 0 {
                continue;
            }
            let (oracle_index, base) = self.role_position(BatchRole::Auxiliary, t);
            let start = base + self.num_lookup_columns[t] + self.num_ctl_helpers[t];
            polys.extend(FriPolynomialInfo::from_range(
                oracle_index,
                start..start + self.num_ctl_zs[t],
            ));
        }
        polys
    }

    /// One `FriOracleInfo` per global oracle: `num_polys` counts only the
    /// polynomials of group `g`.
    fn instance_oracles(&self, g: usize) -> Vec<FriOracleInfo> {
        let grouped_in_g = |role: BatchRole| -> usize {
            self.tables_by_group[g]
                .iter()
                .filter(|&&t| self.is_grouped(t))
                .map(|&t| self.role_num_polys(role, t))
                .sum()
        };
        self.oracles
            .iter()
            .map(|&oracle| {
                let num_polys = match oracle {
                    BatchOracle::Trace(t) if self.group_of_table(t) == g => {
                        self.online_columns[t].len()
                    }
                    BatchOracle::Preprocessed => self.tables_by_group[g]
                        .iter()
                        .map(|&t| self.prep_columns[t].len())
                        .sum(),
                    BatchOracle::Auxiliary(t) if self.group_of_table(t) == g => {
                        self.num_aux_polys(t)
                    }
                    BatchOracle::Quotient(t) if self.group_of_table(t) == g => {
                        self.num_quotient_polys[t]
                    }
                    BatchOracle::GroupedTrace => grouped_in_g(BatchRole::Trace),
                    BatchOracle::GroupedAuxiliary => grouped_in_g(BatchRole::Auxiliary),
                    BatchOracle::GroupedQuotient => grouped_in_g(BatchRole::Quotient),
                    _ => 0,
                };
                FriOracleInfo {
                    num_polys,
                    blinding: false,
                }
            })
            .collect()
    }

    /// The FRI instances of the batch, one per distinct degree.
    pub(crate) fn fri_instances<F: RichField + Extendable<D>, const D: usize>(
        &self,
        zeta: F::Extension,
    ) -> Vec<FriInstanceInfo<F, D>> {
        (0..self.distinct_degree_bits.len())
            .map(|g| {
                let g_subgroup = F::primitive_root_of_unity(self.distinct_degree_bits[g]);
                let zeta_next = zeta.scalar_mul(g_subgroup);
                let mut batches = vec![
                    FriBatchInfo {
                        point: zeta,
                        polynomials: self.batch_polys(g, true),
                    },
                    FriBatchInfo {
                        point: zeta_next,
                        polynomials: self.batch_polys(g, false),
                    },
                ];
                if self.group_has_ctl_batch(g) {
                    batches.push(FriBatchInfo {
                        point: F::Extension::ONE,
                        polynomials: self.ctl_batch_polys(g),
                    });
                }
                FriInstanceInfo {
                    oracles: self.instance_oracles(g),
                    batches,
                }
            })
            .collect()
    }

    /// Circuit version of [`Self::fri_instances`].
    pub(crate) fn fri_instances_target<F: RichField + Extendable<D>, const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        zeta: ExtensionTarget<D>,
    ) -> Vec<FriInstanceInfoTarget<D>> {
        (0..self.distinct_degree_bits.len())
            .map(|g| {
                let g_subgroup = builder.constant_extension(F::Extension::from_basefield(
                    F::primitive_root_of_unity(self.distinct_degree_bits[g]),
                ));
                let zeta_next = builder.mul_extension(g_subgroup, zeta);
                let mut batches = vec![
                    FriBatchInfoTarget {
                        point: zeta,
                        polynomials: self.batch_polys(g, true),
                    },
                    FriBatchInfoTarget {
                        point: zeta_next,
                        polynomials: self.batch_polys(g, false),
                    },
                ];
                if self.group_has_ctl_batch(g) {
                    batches.push(FriBatchInfoTarget {
                        point: builder.one_extension(),
                        polynomials: self.ctl_batch_polys(g),
                    });
                }
                FriInstanceInfoTarget {
                    oracles: self.instance_oracles(g),
                    batches,
                }
            })
            .collect()
    }

    /// Assembles the per-instance FRI openings from the per-table opening
    /// sets, in the exact order of [`Self::fri_instances`].
    pub(crate) fn fri_openings<F: RichField + Extendable<D>, const D: usize>(
        &self,
        openings: &[StarkOpeningSet<F, D>],
    ) -> Vec<FriOpenings<F, D>> {
        debug_assert_eq!(openings.len(), self.num_tables());
        (0..self.distinct_degree_bits.len())
            .map(|g| {
                let zeta_batch = FriOpeningBatch {
                    values: self.batch_values(g, true, |t| {
                        (
                            &openings[t].local_values,
                            openings[t].auxiliary_polys.as_deref(),
                            openings[t].quotient_polys.as_deref(),
                        )
                    }),
                };
                let zeta_next_batch = FriOpeningBatch {
                    values: self.batch_values(g, false, |t| {
                        (
                            &openings[t].next_values,
                            openings[t].auxiliary_polys_next.as_deref(),
                            None,
                        )
                    }),
                };
                let mut batches = vec![zeta_batch, zeta_next_batch];
                if self.group_has_ctl_batch(g) {
                    let values = self.tables_by_group[g]
                        .iter()
                        .flat_map(|&t| {
                            openings[t]
                                .ctl_zs_first
                                .iter()
                                .flatten()
                                .map(|&v| F::Extension::from_basefield(v))
                        })
                        .collect();
                    batches.push(FriOpeningBatch { values });
                }
                FriOpenings { batches }
            })
            .collect()
    }

    /// Circuit version of [`Self::fri_openings`]. `zero` is a `Target` with
    /// value 0, used to lift the base-field `ctl_zs_first` openings into
    /// extension targets.
    pub(crate) fn fri_openings_target<const D: usize>(
        &self,
        zero: Target,
        openings: &[StarkOpeningSetTarget<D>],
    ) -> Vec<FriOpeningsTarget<D>> {
        debug_assert_eq!(openings.len(), self.num_tables());
        (0..self.distinct_degree_bits.len())
            .map(|g| {
                let zeta_batch = FriOpeningBatchTarget {
                    values: self.batch_values(g, true, |t| {
                        (
                            &openings[t].local_values,
                            openings[t].auxiliary_polys.as_deref(),
                            openings[t].quotient_polys.as_deref(),
                        )
                    }),
                };
                let zeta_next_batch = FriOpeningBatchTarget {
                    values: self.batch_values(g, false, |t| {
                        (
                            &openings[t].next_values,
                            openings[t].auxiliary_polys_next.as_deref(),
                            None,
                        )
                    }),
                };
                let mut batches = vec![zeta_batch, zeta_next_batch];
                if self.group_has_ctl_batch(g) {
                    let values = self.tables_by_group[g]
                        .iter()
                        .flat_map(|&t| {
                            openings[t]
                                .ctl_zs_first
                                .iter()
                                .flatten()
                                .map(|&v| v.to_ext_target(zero))
                        })
                        .collect();
                    batches.push(FriOpeningBatchTarget { values });
                }
                FriOpeningsTarget { batches }
            })
            .collect()
    }

    /// Collects the claimed evaluations of one opening batch of group `g`, in
    /// the same order as [`Self::batch_polys`]. `values_of_table` returns,
    /// for a table, its (trace values, auxiliary values, quotient values) at
    /// the batch's point; trace values are the *full* column set, from which
    /// the online and preprocessed values are extracted.
    fn batch_values<'a, T: Copy + 'a>(
        &self,
        g: usize,
        with_quotient: bool,
        values_of_table: impl Fn(usize) -> (&'a Vec<T>, Option<&'a [T]>, Option<&'a [T]>),
    ) -> Vec<T> {
        // Grouped arms mirror `grouped_batch_polys`: the group's grouped
        // tables in table-index order.
        let grouped_in_g = |g: usize| {
            self.tables_by_group[g]
                .iter()
                .copied()
                .filter(|&t| self.is_grouped(t))
        };
        let mut values = vec![];
        for &oracle in &self.oracles {
            match oracle {
                BatchOracle::Trace(t) if self.group_of_table(t) == g => {
                    let (trace_values, _, _) = values_of_table(t);
                    values.extend(self.online_columns[t].iter().map(|&c| trace_values[c]));
                }
                // Same order as `batch_polys`: the group's tables in table-index order.
                BatchOracle::Preprocessed => {
                    for &t in &self.tables_by_group[g] {
                        let (trace_values, _, _) = values_of_table(t);
                        values.extend(self.prep_columns[t].iter().map(|&c| trace_values[c]));
                    }
                }
                BatchOracle::Auxiliary(t) if self.group_of_table(t) == g => {
                    let (_, aux_values, _) = values_of_table(t);
                    debug_assert_eq!(aux_values.map_or(0, |v| v.len()), self.num_aux_polys(t));
                    values.extend(aux_values.into_iter().flatten().copied());
                }
                BatchOracle::Quotient(t) if with_quotient && self.group_of_table(t) == g => {
                    let (_, _, quotient_values) = values_of_table(t);
                    debug_assert_eq!(
                        quotient_values.map_or(0, |v| v.len()),
                        self.num_quotient_polys[t]
                    );
                    values.extend(quotient_values.into_iter().flatten().copied());
                }
                BatchOracle::GroupedTrace => {
                    for t in grouped_in_g(g) {
                        let (trace_values, _, _) = values_of_table(t);
                        values.extend(self.online_columns[t].iter().map(|&c| trace_values[c]));
                    }
                }
                BatchOracle::GroupedAuxiliary => {
                    for t in grouped_in_g(g) {
                        let (_, aux_values, _) = values_of_table(t);
                        debug_assert_eq!(aux_values.map_or(0, |v| v.len()), self.num_aux_polys(t));
                        values.extend(aux_values.into_iter().flatten().copied());
                    }
                }
                BatchOracle::GroupedQuotient if with_quotient => {
                    for t in grouped_in_g(g) {
                        let (_, _, quotient_values) = values_of_table(t);
                        debug_assert_eq!(
                            quotient_values.map_or(0, |v| v.len()),
                            self.num_quotient_polys[t]
                        );
                        values.extend(quotient_values.into_iter().flatten().copied());
                    }
                }
                _ => {}
            }
        }
        values
    }
}
