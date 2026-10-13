//! End-to-end tests for batched multi-STARK proofs: two tables of different
//! heights connected by a cross-table lookup, both with preprocessed columns
//! committed in a single batched setup-time oracle, one column whose values
//! the verifier knows in full ("known column"), proven with one batched FRI
//! argument and verified both natively and recursively.

use core::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use crate::stark::Stark;

/// The value-space size of one looked slot: the looked table serves the values
/// `[base, base + SLOT_OFFSET)` in its first slot, `[base + SLOT_OFFSET, base + 2*SLOT_OFFSET)`
/// in its second, and so on (see [`BatchLookedStark`]).
pub(crate) const SLOT_OFFSET: u64 = 1 << 5;

/// Number of slots (column groups) of the looked table.
pub(crate) const NUM_LOOKED_SLOTS: usize = 3;

/// The "looking" table: `NUM_ROWS` rows, of which the first
/// `NUM_LOOKED_SLOTS * looked_rows` have their filter column set and hold `base + i` in the
/// value column — together they cover all slots of the looked table exactly once.
///
/// Columns: `[v, f, p]` where `p` is a *preprocessed* column with
/// `p[i] = base + (i % (NUM_LOOKED_SLOTS * looked_rows))`.
///
/// Constraints:
/// - first row: `v = base` (public input);
/// - `f` is boolean;
/// - `f * f * (v - p) = 0` (degree 3), i.e. filtered rows copy the
///   preprocessed value.
#[derive(Copy, Clone)]
pub(crate) struct BatchLookingStark<F: RichField + Extendable<D>, const D: usize> {
    _phantom: PhantomData<F>,
}

const LOOKING_COLUMNS: usize = 3;
const LOOKING_PUBLIC_INPUTS: usize = 1;

impl<F: RichField + Extendable<D>, const D: usize> BatchLookingStark<F, D> {
    pub(crate) const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// The preprocessed column `p[i] = base + (i % (NUM_LOOKED_SLOTS * looked_rows))`.
    pub(crate) fn prep_column(
        num_rows: usize,
        looked_rows: usize,
        base: u64,
    ) -> PolynomialValues<F> {
        let covered = NUM_LOOKED_SLOTS * looked_rows;
        PolynomialValues::new(
            (0..num_rows)
                .map(|i| F::from_canonical_u64(base + (i % covered) as u64))
                .collect(),
        )
    }

    /// The full trace, `[v, f, p]` columns.
    pub(crate) fn generate_trace(
        num_rows: usize,
        looked_rows: usize,
        base: u64,
    ) -> Vec<PolynomialValues<F>> {
        let covered = NUM_LOOKED_SLOTS * looked_rows;
        assert!(covered <= num_rows);
        let v = (0..num_rows)
            .map(|i| {
                if i < covered {
                    F::from_canonical_u64(base + i as u64)
                } else {
                    F::from_canonical_u64(0xdead)
                }
            })
            .collect();
        let f = (0..num_rows).map(|i| F::from_bool(i < covered)).collect();
        vec![
            PolynomialValues::new(v),
            PolynomialValues::new(f),
            Self::prep_column(num_rows, looked_rows, base),
        ]
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for BatchLookingStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, LOOKING_COLUMNS, LOOKING_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, LOOKING_COLUMNS, LOOKING_PUBLIC_INPUTS>;

    fn constraint_degree(&self) -> usize {
        3
    }

    fn requires_ctls(&self) -> bool {
        true
    }

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let local_values = vars.get_local_values();
        let public_inputs = vars.get_public_inputs();
        let v = local_values[0];
        let f = local_values[1];
        let p = local_values[2];

        // First row: `v = base`.
        yield_constr.constraint_first_row(v - public_inputs[0]);
        // `f` is boolean.
        yield_constr.constraint(f * (f - P::ONES));
        // Filtered rows copy the preprocessed value (degree 3).
        yield_constr.constraint(f * f * (v - p));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let local_values = vars.get_local_values();
        let public_inputs = vars.get_public_inputs();
        let v = local_values[0];
        let f = local_values[1];
        let p = local_values[2];
        let one = builder.one_extension();

        let first_row = builder.sub_extension(v, public_inputs[0]);
        yield_constr.constraint_first_row(builder, first_row);

        let f_minus_one = builder.sub_extension(f, one);
        let f_boolean = builder.mul_extension(f, f_minus_one);
        yield_constr.constraint(builder, f_boolean);

        let v_minus_p = builder.sub_extension(v, p);
        let f_squared = builder.mul_extension(f, f);
        let copy = builder.mul_extension(f_squared, v_minus_p);
        yield_constr.constraint(builder, copy);
    }
}

/// The "looked" table: `looked_rows` rows whose values span three lookup *slots* — the value
/// columns `w`, `w2 = w + SLOT_OFFSET` and `w3 = w + 2*SLOT_OFFSET`. A CTL looking for the
/// union `[base, base + 3*SLOT_OFFSET)` uses the three slots as a multi-slot looked side,
/// which requires helper columns on the looked side (3 slots at constraint degree 3 make
/// ceil(3/2) = 2 helpers, exercising both the paired and the lone helper chunk).
///
/// Columns: `[w, u, w2, w3]` where `w` is a *preprocessed* column with `w[j] = base + j`, and
/// `u = w + 1`, `w2`, `w3` are online columns.
///
/// Its declared constraint degree is 3, as required for CTL tables (the CTL
/// Z-consistency constraint on the last row has degree ~3n).
#[derive(Copy, Clone)]
pub(crate) struct BatchLookedStark<F: RichField + Extendable<D>, const D: usize> {
    /// Whether this table participates in a CTL (allows reusing this STARK in
    /// CTL-free tests).
    with_ctl: bool,
    _phantom: PhantomData<F>,
}

const LOOKED_COLUMNS: usize = 4;
const LOOKED_PUBLIC_INPUTS: usize = 0;

impl<F: RichField + Extendable<D>, const D: usize> BatchLookedStark<F, D> {
    pub(crate) const fn new(with_ctl: bool) -> Self {
        Self {
            with_ctl,
            _phantom: PhantomData,
        }
    }

    /// The preprocessed column `w[j] = base + j`.
    pub(crate) fn prep_column(looked_rows: usize, base: u64) -> PolynomialValues<F> {
        PolynomialValues::new(
            (0..looked_rows)
                .map(|j| F::from_canonical_u64(base + j as u64))
                .collect(),
        )
    }

    /// The full trace, `[w, u, w2, w3]` columns.
    pub(crate) fn generate_trace(looked_rows: usize, base: u64) -> Vec<PolynomialValues<F>> {
        let w = Self::prep_column(looked_rows, base);
        let u = PolynomialValues::new(w.values.iter().map(|&w| w + F::ONE).collect());
        let offset = F::from_canonical_u64(SLOT_OFFSET);
        let w2 = PolynomialValues::new(w.values.iter().map(|&w| w + offset).collect());
        let w3 = PolynomialValues::new(w.values.iter().map(|&w| w + offset.double()).collect());
        vec![w, u, w2, w3]
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for BatchLookedStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, LOOKED_COLUMNS, LOOKED_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, LOOKED_COLUMNS, LOOKED_PUBLIC_INPUTS>;

    fn constraint_degree(&self) -> usize {
        3
    }

    fn requires_ctls(&self) -> bool {
        self.with_ctl
    }

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let local_values = vars.get_local_values();
        let w = local_values[0];
        let u = local_values[1];
        let w2 = local_values[2];
        let w3 = local_values[3];
        let offset = FE::from_canonical_u64(SLOT_OFFSET);

        // `u = w + 1`.
        yield_constr.constraint(u - w - P::ONES);
        // The slot columns are the preprocessed ramp, shifted: `w2 = w + SLOT_OFFSET`,
        // `w3 = w2 + SLOT_OFFSET`.
        yield_constr.constraint(w2 - w - offset);
        yield_constr.constraint(w3 - w2 - offset);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let local_values = vars.get_local_values();
        let w = local_values[0];
        let u = local_values[1];
        let w2 = local_values[2];
        let w3 = local_values[3];
        let one = builder.one_extension();
        let offset = F::from_canonical_u64(SLOT_OFFSET);

        let u_minus_w = builder.sub_extension(u, w);
        let constraint = builder.sub_extension(u_minus_w, one);
        yield_constr.constraint(builder, constraint);

        let w2_minus_w = builder.sub_extension(w2, w);
        let constraint = builder.add_const_extension(w2_minus_w, -offset);
        yield_constr.constraint(builder, constraint);

        let w3_minus_w2 = builder.sub_extension(w3, w2);
        let constraint = builder.add_const_extension(w3_minus_w2, -offset);
        yield_constr.constraint(builder, constraint);
    }
}

/// A table whose main, preprocessed and known columns are *interleaved*:
/// `[a, p0, k, b, p1, c]`, where `a`, `b`, `c` are main (online) columns,
/// `p0`, `p1` are preprocessed columns and `k` is a known column, each with a
/// pairwise-different value pattern:
///
/// - `a[i]  = i + 1`
/// - `p0[i] = 2i + 3`   (preprocessed)
/// - `k[i]  = i^2 + 5`  (known)
/// - `b[i]  = a[i] * p0[i]`
/// - `p1[i] = 7i + 2`   (preprocessed)
/// - `c[i]  = b[i] + k[i] * p1[i]`
///
/// The constraints involve every column, including the next-row values of a
/// main and a preprocessed column, so misrouting any column between the
/// batched trace/preprocessed oracles (or their openings) breaks the proof:
///
/// - `b = a * p0`
/// - `c = b + k * p1`
/// - transition: `a' = a + 1`
/// - transition: `p1' = p1 + 7`
#[derive(Copy, Clone)]
struct InterleavedStark<F: RichField + Extendable<D>, const D: usize> {
    _phantom: PhantomData<F>,
}

const INTERLEAVED_COLUMNS: usize = 6;
const INTERLEAVED_PUBLIC_INPUTS: usize = 0;

impl<F: RichField + Extendable<D>, const D: usize> InterleavedStark<F, D> {
    const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// The preprocessed columns `[p0, p1]`, in the order of their indices
    /// `{1, 4}` within the trace.
    fn prep_values(num_rows: usize) -> Vec<PolynomialValues<F>> {
        let p0 = (0..num_rows)
            .map(|i| F::from_canonical_u64(2 * i as u64 + 3))
            .collect();
        let p1 = (0..num_rows)
            .map(|i| F::from_canonical_u64(7 * i as u64 + 2))
            .collect();
        vec![PolynomialValues::new(p0), PolynomialValues::new(p1)]
    }

    /// The known column `k` (index 2 within the trace).
    fn known_column(num_rows: usize) -> PolynomialValues<F> {
        PolynomialValues::new(
            (0..num_rows)
                .map(|i| F::from_canonical_u64((i * i + 5) as u64))
                .collect(),
        )
    }

    /// The full trace, `[a, p0, k, b, p1, c]` columns.
    fn generate_trace(num_rows: usize) -> Vec<PolynomialValues<F>> {
        let prep = Self::prep_values(num_rows);
        let (p0, p1) = (prep[0].clone(), prep[1].clone());
        let k = Self::known_column(num_rows);
        let a: Vec<F> = (0..num_rows)
            .map(|i| F::from_canonical_u64(i as u64 + 1))
            .collect();
        let b: Vec<F> = (0..num_rows).map(|i| a[i] * p0.values[i]).collect();
        let c = (0..num_rows)
            .map(|i| b[i] + k.values[i] * p1.values[i])
            .collect();
        vec![
            PolynomialValues::new(a),
            p0,
            k,
            PolynomialValues::new(b),
            p1,
            PolynomialValues::new(c),
        ]
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for InterleavedStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, INTERLEAVED_COLUMNS, INTERLEAVED_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget = StarkFrame<
        ExtensionTarget<D>,
        ExtensionTarget<D>,
        INTERLEAVED_COLUMNS,
        INTERLEAVED_PUBLIC_INPUTS,
    >;

    fn constraint_degree(&self) -> usize {
        2
    }

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let local_values = vars.get_local_values();
        let next_values = vars.get_next_values();
        let &[a, p0, k, b, p1, c] = local_values else {
            unreachable!()
        };
        let a_next = next_values[0];
        let p1_next = next_values[4];

        yield_constr.constraint(b - a * p0);
        yield_constr.constraint(c - b - k * p1);
        yield_constr.constraint_transition(a_next - a - P::ONES);
        yield_constr.constraint_transition(p1_next - p1 - FE::from_canonical_u64(7));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let local_values = vars.get_local_values();
        let next_values = vars.get_next_values();
        let &[a, p0, k, b, p1, c] = local_values else {
            unreachable!()
        };
        let a_next = next_values[0];
        let p1_next = next_values[4];

        let a_p0 = builder.mul_extension(a, p0);
        let mul_check = builder.sub_extension(b, a_p0);
        yield_constr.constraint(builder, mul_check);

        let k_p1 = builder.mul_extension(k, p1);
        let b_plus_k_p1 = builder.add_extension(b, k_p1);
        let sum_check = builder.sub_extension(c, b_plus_k_p1);
        yield_constr.constraint(builder, sum_check);

        let a_diff = builder.sub_extension(a_next, a);
        let a_check = builder.add_const_extension(a_diff, F::NEG_ONE);
        yield_constr.constraint_transition(builder, a_check);

        let p1_diff = builder.sub_extension(p1_next, p1);
        let p1_check = builder.add_const_extension(p1_diff, -F::from_canonical_u64(7));
        yield_constr.constraint_transition(builder, p1_check);
    }
}

/// A minimal standalone table with no lookups, CTLs or preprocessed columns:
/// columns `[x, y]` with `y = x * x`.
#[derive(Copy, Clone)]
pub(crate) struct SquareStark<F: RichField + Extendable<D>, const D: usize> {
    _phantom: PhantomData<F>,
}

const SQUARE_COLUMNS: usize = 2;
const SQUARE_PUBLIC_INPUTS: usize = 0;

impl<F: RichField + Extendable<D>, const D: usize> SquareStark<F, D> {
    pub(crate) const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    pub(crate) fn generate_trace(num_rows: usize) -> Vec<PolynomialValues<F>> {
        let x: Vec<F> = (0..num_rows).map(F::from_canonical_usize).collect();
        let y = x.iter().map(|&x| x * x).collect();
        vec![PolynomialValues::new(x), PolynomialValues::new(y)]
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for SquareStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, SQUARE_COLUMNS, SQUARE_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, SQUARE_COLUMNS, SQUARE_PUBLIC_INPUTS>;

    fn constraint_degree(&self) -> usize {
        2
    }

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let local_values = vars.get_local_values();
        yield_constr.constraint(local_values[1] - local_values[0] * local_values[0]);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let local_values = vars.get_local_values();
        let x_squared = builder.mul_extension(local_values[0], local_values[0]);
        let constraint = builder.sub_extension(local_values[1], x_squared);
        yield_constr.constraint(builder, constraint);
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use hashbrown::HashMap;
    use plonky2::field::polynomial::PolynomialValues;
    use plonky2::field::types::Field;
    use plonky2::fri::reduction_strategies::FriReductionStrategy;
    use plonky2::fri::FriConfig;
    use plonky2::hash::poseidon::PoseidonHash;
    use plonky2::iop::ext_target::ExtensionTarget;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, Hasher, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::{BatchLookedStark, BatchLookingStark, InterleavedStark, SquareStark};
    use crate::batch_proof::BatchStarkProofWithPublicInputs;
    use crate::batch_prover::{batch_prove, BatchStarkPreprocessedData};
    use crate::batch_recursive_verifier::{
        add_virtual_batch_stark_proof_with_pis, set_batch_stark_proof_with_pis_target,
        verify_batch_stark_proof_circuit, BatchKnownColumnsTarget,
    };
    use crate::batch_stark::BatchStark;
    use crate::batch_verifier::{batch_verify, BatchKnownColumns};
    use crate::config::StarkConfig;
    use crate::cross_table_lookup::{debug_utils::check_ctls, CrossTableLookup, TableWithColumns};
    use crate::lookup::{Column, Filter};

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    use super::{NUM_LOOKED_SLOTS, SLOT_OFFSET};

    const LOOKING_ROWS: usize = 1 << 7;
    const LOOKED_ROWS: usize = 1 << 5;
    const BASE: u64 = 100;
    // The looked table's slots tile the looking side's contiguous value range
    // `[BASE, BASE + 3 * LOOKED_ROWS)` exactly.
    const _: () = assert!(LOOKED_ROWS as u64 == SLOT_OFFSET);
    const _: () = assert!(NUM_LOOKED_SLOTS * LOOKED_ROWS <= LOOKING_ROWS);

    /// A small test configuration whose FRI reduction arities land exactly on
    /// both instance degrees (LDEs of 2^8 and 2^6): 8 -> 6 -> 4.
    fn test_config() -> StarkConfig {
        StarkConfig::new(
            40,
            2,
            FriConfig {
                rate_bits: 1,
                cap_height: 2,
                proof_of_work_bits: 8,
                reduction_strategy: FriReductionStrategy::Fixed(vec![2, 2]),
                num_query_rounds: 40,
            },
        )
    }

    /// The CTL connecting the two tables: filtered `v` values of the looking
    /// table must form the same multiset as the union of the looked table's
    /// three (always-on) slots `w`, `w2` and `w3`. The multi-slot looked side
    /// makes the looked table carry ceil(3/2) = 2 helper columns per
    /// challenge.
    fn ctls() -> Vec<CrossTableLookup<F>> {
        vec![CrossTableLookup::new(
            vec![TableWithColumns::new(
                0,
                vec![Column::single(0)],
                Filter::from_column(Column::single(1)),
            )],
            looked_slots(),
        )]
    }

    /// The looked table's slots: the value columns `w` (0), `w2` (2) and `w3` (3).
    fn looked_slots() -> Vec<TableWithColumns<F>> {
        [0, 2, 3]
            .map(|c| TableWithColumns::new(1, vec![Column::single(c)], Filter::default()))
            .to_vec()
    }

    struct TestSystem {
        stark_a: BatchLookingStark<F, D>,
        stark_b: BatchLookedStark<F, D>,
        config: StarkConfig,
        traces: [Vec<PolynomialValues<F>>; 2],
        public_inputs: [Vec<F>; 2],
        cross_table_lookups: Vec<CrossTableLookup<F>>,
        preprocessed: BatchStarkPreprocessedData<F, C, D>,
        prep_columns: Vec<Vec<usize>>,
        known_columns: BatchKnownColumns<F>,
    }

    fn build_test_system() -> TestSystem {
        let config = test_config();
        let stark_a = BatchLookingStark::<F, D>::new();
        let stark_b = BatchLookedStark::<F, D>::new(true);
        let trace_a = BatchLookingStark::<F, D>::generate_trace(LOOKING_ROWS, LOOKED_ROWS, BASE);
        let trace_b = BatchLookedStark::<F, D>::generate_trace(LOOKED_ROWS, BASE);
        let cross_table_lookups = ctls();

        // Sanity-check the traces against the CTL semantics (the test CTL
        // expressions carry no public-input terms).
        check_ctls(
            &[trace_a.clone(), trace_b.clone()],
            &[vec![], vec![]],
            &cross_table_lookups,
            &HashMap::new(),
        );

        // Batched setup-time commitment to the preprocessed columns of both
        // tables: column 2 of the looking table and column 0 of the looked one.
        let prep_columns = vec![vec![2], vec![0]];
        let preprocessed = BatchStarkPreprocessedData::<F, C, D>::new(
            vec![
                vec![BatchLookingStark::<F, D>::prep_column(
                    LOOKING_ROWS,
                    LOOKED_ROWS,
                    BASE,
                )],
                vec![BatchLookedStark::<F, D>::prep_column(LOOKED_ROWS, BASE)],
            ],
            prep_columns.clone(),
            &config,
            &mut TimingTree::default(),
        );

        // The filter column (index 1) of the looking table is fully known to
        // the verifier: declare it as a known column. It stays committed with
        // the trace, but the verifier recomputes and checks its openings.
        let known_values = vec![
            vec![PolynomialValues::new(
                (0..LOOKING_ROWS)
                    .map(|i| F::from_bool(i < NUM_LOOKED_SLOTS * LOOKED_ROWS))
                    .collect(),
            )],
            vec![],
        ];
        let flat_known: Vec<F> = known_values
            .iter()
            .flatten()
            .flat_map(|v| v.values.iter().copied())
            .collect();
        let known_columns = BatchKnownColumns {
            digest: Some(PoseidonHash::hash_no_pad(&flat_known)),
            columns_per_table: vec![vec![1], vec![]],
            values_per_table: known_values,
        };

        TestSystem {
            stark_a,
            stark_b,
            config,
            traces: [trace_a, trace_b],
            public_inputs: [vec![F::from_canonical_u64(BASE)], vec![]],
            cross_table_lookups,
            preprocessed,
            prep_columns,
            known_columns,
        }
    }

    fn prove_test_system(system: &TestSystem) -> Result<BatchStarkProofWithPublicInputs<F, C, D>> {
        let starks: [&dyn BatchStark<F, D>; 2] = [&system.stark_a, &system.stark_b];
        batch_prove::<F, C, D, 2>(
            &starks,
            &system.config,
            system.traces.clone(),
            &system.public_inputs,
            &system.cross_table_lookups,
            &[],
            Some(&system.preprocessed),
            Some(&system.known_columns),
            &mut TimingTree::default(),
        )
    }

    #[test]
    fn test_batch_stark_with_ctl_and_preprocessed() -> Result<()> {
        let system = build_test_system();
        let starks: [&dyn BatchStark<F, D>; 2] = [&system.stark_a, &system.stark_b];
        let proof = prove_test_system(&system)?;

        batch_verify::<F, C, D, 2>(
            &starks,
            &system.config,
            &proof,
            &system.cross_table_lookups,
            &[],
            Some(&system.preprocessed.verifier_data()),
            Some(&system.known_columns),
            &HashMap::new(),
        )?;

        // A proof with a tampered public input must be rejected.
        let mut bad_proof = proof.clone();
        bad_proof.public_inputs[0][0] += F::ONE;
        assert!(batch_verify::<F, C, D, 2>(
            &starks,
            &system.config,
            &bad_proof,
            &system.cross_table_lookups,
            &[],
            Some(&system.preprocessed.verifier_data()),
            Some(&system.known_columns),
            &HashMap::new(),
        )
        .is_err());

        // A proof checked against different preprocessed data must be rejected.
        let mut bad_prep = system.preprocessed.verifier_data();
        bad_prep.cap.0[0].elements[0] += F::ONE;
        assert!(batch_verify::<F, C, D, 2>(
            &starks,
            &system.config,
            &proof,
            &system.cross_table_lookups,
            &[],
            Some(&bad_prep),
            Some(&system.known_columns),
            &HashMap::new(),
        )
        .is_err());

        // A verifier knowing different known-column values must reject the
        // proof (the recomputed openings no longer match).
        let mut bad_known = system.known_columns.clone();
        bad_known.values_per_table[0][0].values[0] += F::ONE;
        assert!(batch_verify::<F, C, D, 2>(
            &starks,
            &system.config,
            &proof,
            &system.cross_table_lookups,
            &[],
            Some(&system.preprocessed.verifier_data()),
            Some(&bad_known),
            &HashMap::new(),
        )
        .is_err());

        // A verifier with a different known-column digest must reject the
        // proof (the Fiat-Shamir transcripts diverge).
        let mut bad_digest = system.known_columns.clone();
        bad_digest.digest.as_mut().unwrap().elements[0] += F::ONE;
        assert!(batch_verify::<F, C, D, 2>(
            &starks,
            &system.config,
            &proof,
            &system.cross_table_lookups,
            &[],
            Some(&system.preprocessed.verifier_data()),
            Some(&bad_digest),
            &HashMap::new(),
        )
        .is_err());

        // A verifier not given the known-column data must reject the proof.
        assert!(batch_verify::<F, C, D, 2>(
            &starks,
            &system.config,
            &proof,
            &system.cross_table_lookups,
            &[],
            Some(&system.preprocessed.verifier_data()),
            None,
            &HashMap::new(),
        )
        .is_err());

        Ok(())
    }

    #[test]
    fn test_batch_stark_missing_looked_slot_rejected() -> Result<()> {
        // A CTL whose looked side omits one of the three slots is imbalanced:
        // the looking side covers all three slots' values. The prover happily
        // produces the (internally consistent) Z polynomials, but the
        // cross-table sum check must reject the proof.
        let system = build_test_system();
        let starks: [&dyn BatchStark<F, D>; 2] = [&system.stark_a, &system.stark_b];
        let bad_ctls = vec![CrossTableLookup::new(
            vec![TableWithColumns::new(
                0,
                vec![Column::single(0)],
                Filter::from_column(Column::single(1)),
            )],
            looked_slots()[..2].to_vec(),
        )];

        let proof = batch_prove::<F, C, D, 2>(
            &starks,
            &system.config,
            system.traces.clone(),
            &system.public_inputs,
            &bad_ctls,
            &[],
            Some(&system.preprocessed),
            Some(&system.known_columns),
            &mut TimingTree::default(),
        )?;
        let err = batch_verify::<F, C, D, 2>(
            &starks,
            &system.config,
            &proof,
            &bad_ctls,
            &[],
            Some(&system.preprocessed.verifier_data()),
            Some(&system.known_columns),
            &HashMap::new(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("Cross-table lookup"),
            "expected a CTL sum mismatch, got: {err}"
        );
        Ok(())
    }

    /// Evaluates a polynomial given by constant (circuit-independent)
    /// coefficients at an extension target, via Horner's rule.
    fn eval_constant_poly_circuit(
        builder: &mut CircuitBuilder<F, D>,
        coeffs: &[F],
        point: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let mut acc = builder.zero_extension();
        for &c in coeffs.iter().rev() {
            let c = builder.constant_extension(c.into());
            acc = builder.mul_add_extension(acc, point, c);
        }
        acc
    }

    #[test]
    fn test_batch_stark_recursive_verifier() -> Result<()> {
        init_logger();
        let system = build_test_system();
        let starks: [&dyn BatchStark<F, D>; 2] = [&system.stark_a, &system.stark_b];
        let proof = prove_test_system(&system)?;
        batch_verify::<F, C, D, 2>(
            &starks,
            &system.config,
            &proof,
            &system.cross_table_lookups,
            &[],
            Some(&system.preprocessed.verifier_data()),
            Some(&system.known_columns),
            &HashMap::new(),
        )?;

        // Verify the batch proof inside a plonky2 circuit.
        let circuit_config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(circuit_config);
        let proof_target = add_virtual_batch_stark_proof_with_pis(
            &mut builder,
            &starks,
            &system.config,
            &proof.proof.degree_bits,
            &system.prep_columns,
            &system.cross_table_lookups,
            &[],
        )?;
        let prep_target = system
            .preprocessed
            .verifier_data()
            .constant_target(&mut builder);

        // Known columns: the expected evaluations are virtual targets here;
        // they are recomputed in-circuit below, once zeta is available.
        let known_target = BatchKnownColumnsTarget::<D> {
            digest: system
                .known_columns
                .digest
                .map(|digest| builder.constant_hash(digest)),
            columns_per_table: system.known_columns.columns_per_table.clone(),
            evals_at_zeta: vec![vec![builder.add_virtual_extension_target()], vec![]],
            evals_at_g_zeta: vec![vec![builder.add_virtual_extension_target()], vec![]],
        };

        let zeta = verify_batch_stark_proof_circuit::<F, C, D, 2>(
            &mut builder,
            &starks,
            &system.config,
            &proof_target,
            &system.cross_table_lookups,
            &[],
            Some(&prep_target),
            Some(&known_target),
            &HashMap::new(),
        )?;

        // The verifier recomputes the known-column openings in-circuit:
        // evaluate the column's (constant) coefficient form at the returned
        // zeta and at g * zeta, and bind the results to the expected
        // evaluations connected to the proof openings above.
        let coeffs = system.known_columns.values_per_table[0][0].clone().ifft();
        let eval_at_zeta = eval_constant_poly_circuit(&mut builder, &coeffs.coeffs, zeta);
        builder.connect_extension(eval_at_zeta, known_target.evals_at_zeta[0][0]);
        let g = builder
            .constant_extension(F::primitive_root_of_unity(proof.proof.degree_bits[0]).into());
        let g_zeta = builder.mul_extension(g, zeta);
        let eval_at_g_zeta = eval_constant_poly_circuit(&mut builder, &coeffs.coeffs, g_zeta);
        builder.connect_extension(eval_at_g_zeta, known_target.evals_at_g_zeta[0][0]);

        builder.print_gate_counts(0);

        let mut pw = PartialWitness::new();
        set_batch_stark_proof_with_pis_target(&mut pw, &proof_target, &proof)?;

        let data = builder.build::<C>();
        let recursive_proof = data.prove(pw)?;
        data.verify(recursive_proof)
    }

    #[test]
    fn test_batch_stark_two_tables_prep_no_ctl() -> Result<()> {
        // Two tables of different heights with preprocessed columns but no
        // CTLs: isolates the batched-preprocessed path.
        let config = test_config();
        let stark_a = BatchLookedStark::<F, D>::new(false);
        let stark_b = BatchLookedStark::<F, D>::new(false);
        let starks: [&dyn BatchStark<F, D>; 2] = [&stark_a, &stark_b];
        let trace_a = BatchLookedStark::<F, D>::generate_trace(LOOKING_ROWS, BASE);
        let trace_b = BatchLookedStark::<F, D>::generate_trace(LOOKED_ROWS, BASE);

        let prep_columns = vec![vec![0], vec![0]];
        let preprocessed = BatchStarkPreprocessedData::<F, C, D>::new(
            vec![
                vec![BatchLookedStark::<F, D>::prep_column(LOOKING_ROWS, BASE)],
                vec![BatchLookedStark::<F, D>::prep_column(LOOKED_ROWS, BASE)],
            ],
            prep_columns,
            &config,
            &mut TimingTree::default(),
        );

        let proof = batch_prove::<F, C, D, 2>(
            &starks,
            &config,
            [trace_a, trace_b],
            &[vec![], vec![]],
            &[],
            &[],
            Some(&preprocessed),
            None,
            &mut TimingTree::default(),
        )?;
        batch_verify::<F, C, D, 2>(
            &starks,
            &config,
            &proof,
            &[],
            &[],
            Some(&preprocessed.verifier_data()),
            None,
            &HashMap::new(),
        )
    }

    #[test]
    fn test_batch_stark_two_tables_no_ctl_no_prep() -> Result<()> {
        // Two tables of different heights, no CTLs, no lookups, no
        // preprocessed data: isolates the mixed-height trace/quotient path.
        let config = test_config();
        let stark_a = SquareStark::<F, D>::new();
        let stark_b = SquareStark::<F, D>::new();
        let starks: [&dyn BatchStark<F, D>; 2] = [&stark_a, &stark_b];
        let trace_a = SquareStark::<F, D>::generate_trace(LOOKING_ROWS);
        let trace_b = SquareStark::<F, D>::generate_trace(LOOKED_ROWS);

        let proof = batch_prove::<F, C, D, 2>(
            &starks,
            &config,
            [trace_a, trace_b],
            &[vec![], vec![]],
            &[],
            &[],
            None,
            None,
            &mut TimingTree::default(),
        )?;
        batch_verify::<F, C, D, 2>(
            &starks,
            &config,
            &proof,
            &[],
            &[],
            None,
            None,
            &HashMap::new(),
        )
    }

    #[test]
    fn test_batch_stark_single_table_no_ctl_no_prep() -> Result<()> {
        // Degenerate batch: a single table, no CTLs, no lookups, no
        // preprocessed data. Only the trace and quotient oracles exist.
        let config = StarkConfig::new(
            40,
            2,
            FriConfig {
                rate_bits: 1,
                cap_height: 2,
                proof_of_work_bits: 8,
                reduction_strategy: FriReductionStrategy::Fixed(vec![2]),
                num_query_rounds: 40,
            },
        );
        let stark = SquareStark::<F, D>::new();
        let starks: [&dyn BatchStark<F, D>; 1] = [&stark];
        let trace = SquareStark::<F, D>::generate_trace(1 << 6);

        let proof = batch_prove::<F, C, D, 1>(
            &starks,
            &config,
            [trace],
            &[vec![]],
            &[],
            &[],
            None,
            None,
            &mut TimingTree::default(),
        )?;
        batch_verify::<F, C, D, 1>(
            &starks,
            &config,
            &proof,
            &[],
            &[],
            None,
            None,
            &HashMap::new(),
        )
    }

    #[test]
    fn test_batch_stark_interleaved_prep_and_known() -> Result<()> {
        // A single table whose main, preprocessed and known columns are
        // interleaved ([main, prep, known, main, prep, main]) with
        // pairwise-different value patterns, all involved in the constraints
        // (including transition constraints on a main and a preprocessed
        // column): checks that every column kind is routed to its correct
        // slot within the batched oracles, the reassembled evaluation frames
        // and the openings.
        let config = StarkConfig::new(
            40,
            2,
            FriConfig {
                rate_bits: 1,
                cap_height: 2,
                proof_of_work_bits: 8,
                reduction_strategy: FriReductionStrategy::Fixed(vec![2]),
                num_query_rounds: 40,
            },
        );
        let stark = InterleavedStark::<F, D>::new();
        let starks: [&dyn BatchStark<F, D>; 1] = [&stark];
        let num_rows = 1 << 6;
        let trace = InterleavedStark::<F, D>::generate_trace(num_rows);

        let prep_columns = vec![vec![1, 4]];
        let preprocessed = BatchStarkPreprocessedData::<F, C, D>::new(
            vec![InterleavedStark::<F, D>::prep_values(num_rows)],
            prep_columns,
            &config,
            &mut TimingTree::default(),
        );

        let known_column = InterleavedStark::<F, D>::known_column(num_rows);
        let known_columns = BatchKnownColumns {
            digest: Some(PoseidonHash::hash_no_pad(&known_column.values)),
            columns_per_table: vec![vec![2]],
            values_per_table: vec![vec![known_column]],
        };

        let proof = batch_prove::<F, C, D, 1>(
            &starks,
            &config,
            [trace],
            &[vec![]],
            &[],
            &[],
            Some(&preprocessed),
            Some(&known_columns),
            &mut TimingTree::default(),
        )?;
        batch_verify::<F, C, D, 1>(
            &starks,
            &config,
            &proof,
            &[],
            &[],
            Some(&preprocessed.verifier_data()),
            Some(&known_columns),
            &HashMap::new(),
        )?;

        // A verifier believing the known column sits at a different index
        // (here a main column holding different values) must reject the
        // proof.
        let mut misplaced = known_columns.clone();
        misplaced.columns_per_table[0] = vec![3];
        assert!(batch_verify::<F, C, D, 1>(
            &starks,
            &config,
            &proof,
            &[],
            &[],
            Some(&preprocessed.verifier_data()),
            Some(&misplaced),
            &HashMap::new(),
        )
        .is_err());

        Ok(())
    }

    #[test]
    fn test_batch_stark_grouped_tables() -> Result<()> {
        // The full CTL + preprocessed + known system, with both tables
        // (of different heights) grouped: each role (trace, auxiliary,
        // quotient) is committed in one shared multi-height tree, so the
        // proof carries exactly one cap per role.
        let system = build_test_system();
        let starks: [&dyn BatchStark<F, D>; 2] = [&system.stark_a, &system.stark_b];
        let grouped: &[usize] = &[0, 1];

        let proof = batch_prove::<F, C, D, 2>(
            &starks,
            &system.config,
            system.traces.clone(),
            &system.public_inputs,
            &system.cross_table_lookups,
            grouped,
            Some(&system.preprocessed),
            Some(&system.known_columns),
            &mut TimingTree::default(),
        )?;
        assert_eq!(proof.proof.trace_caps.len(), 1);
        assert_eq!(proof.proof.auxiliary_polys_caps.as_ref().unwrap().len(), 1);
        assert_eq!(proof.proof.quotient_polys_caps.as_ref().unwrap().len(), 1);

        let verify = |proof: &BatchStarkProofWithPublicInputs<F, C, D>,
                      grouped_tables: &[usize]|
         -> Result<()> {
            batch_verify::<F, C, D, 2>(
                &starks,
                &system.config,
                proof,
                &system.cross_table_lookups,
                grouped_tables,
                Some(&system.preprocessed.verifier_data()),
                Some(&system.known_columns),
                &HashMap::new(),
            )
        };
        verify(&proof, grouped)?;

        // A grouping mismatch between prover and verifier must be rejected:
        // the cap counts (and the whole transcript) differ.
        assert!(verify(&proof, &[]).is_err());
        let solo_proof = prove_test_system(&system)?;
        assert!(verify(&solo_proof, grouped).is_err());

        // Tampering any grouped cap must be rejected.
        for role in 0..3 {
            let mut bad = proof.clone();
            let cap = match role {
                0 => &mut bad.proof.trace_caps[0],
                1 => &mut bad.proof.auxiliary_polys_caps.as_mut().unwrap()[0],
                _ => &mut bad.proof.quotient_polys_caps.as_mut().unwrap()[0],
            };
            cap.0[0].elements[0] += F::ONE;
            assert!(verify(&bad, grouped).is_err());
        }

        Ok(())
    }

    fn init_logger() {
        let _ = env_logger::builder().format_timestamp(None).try_init();
    }
}
