//! A Lookup protocol leveraging logarithmic derivatives,
//! introduced in <https://eprint.iacr.org/2022/1530.pdf>.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::borrow::Borrow;
use core::fmt::Debug;
use core::iter::repeat;

#[cfg(feature = "std")]
use itertools::Itertools;
use num_bigint::BigUint;
use plonky2::field::batch_util::{batch_add_inplace, batch_multiply_inplace};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::challenger::{Challenger, RecursiveChallenger};
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, Hasher};
use plonky2::plonk::plonk_common::{
    reduce_with_powers, reduce_with_powers_circuit, reduce_with_powers_ext_circuit,
};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::evaluation_frame::StarkEvaluationFrame;
use crate::stark::Stark;

/// Represents a filter, which evaluates to 1 if the row must be considered and 0 if it should be ignored.
/// It's an arbitrary degree 2 combination of columns: `products` are the degree 2 terms, and `constants` are
/// the degree 1 terms.
#[derive(Clone, Debug)]
pub struct Filter<F: Field> {
    products: Vec<(Column<F>, Column<F>)>,
    constants: Vec<Column<F>>,
}

/// The default filter is always on.
impl<F: Field> Default for Filter<F> {
    fn default() -> Self {
        Self {
            products: vec![],
            constants: vec![Column::constant(F::ONE)],
        }
    }
}

impl<F: Field> Filter<F> {
    /// Returns a filter from the provided `products` and `constants` vectors.
    pub fn new(products: Vec<(Column<F>, Column<F>)>, constants: Vec<Column<F>>) -> Self {
        Self {
            products,
            constants,
        }
    }

    /// Returns a filter made of a single column.
    pub fn from_column(col: Column<F>) -> Self {
        Self {
            products: vec![],
            constants: vec![col],
        }
    }

    /// Given the column values for the current and next rows and the table's public inputs,
    /// evaluates the filter.
    pub(crate) fn eval_filter<FE, P, const D: usize>(
        &self,
        v: &[P],
        next_v: &[P],
        public_inputs: &[FE],
    ) -> P
    where
        FE: FieldExtension<D, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        self.products
            .iter()
            .map(|(col1, col2)| {
                col1.eval_with_next(v, next_v, public_inputs)
                    * col2.eval_with_next(v, next_v, public_inputs)
            })
            .sum::<P>()
            + self
                .constants
                .iter()
                .map(|col| col.eval_with_next(v, next_v, public_inputs))
                .sum::<P>()
    }

    /// Circuit version of `eval_filter`:
    /// Given the column values for the current and next rows, evaluates the filter.
    pub(crate) fn eval_filter_circuit<const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        v: &[ExtensionTarget<D>],
        next_v: &[ExtensionTarget<D>],
        public_inputs: &[ExtensionTarget<D>],
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        let prods = self
            .products
            .iter()
            .map(|(col1, col2)| {
                let col1_eval = col1.eval_with_next_circuit(builder, v, next_v, public_inputs);
                let col2_eval = col2.eval_with_next_circuit(builder, v, next_v, public_inputs);
                builder.mul_extension(col1_eval, col2_eval)
            })
            .collect::<Vec<_>>();

        let consts = self
            .constants
            .iter()
            .map(|col| col.eval_with_next_circuit(builder, v, next_v, public_inputs))
            .collect::<Vec<_>>();

        let prods = builder.add_many_extension(prods);
        let consts = builder.add_many_extension(consts);
        builder.add_extension(prods, consts)
    }

    /// Evaluate on a row of a table given in column-major form.
    pub fn eval_table(&self, table: &[PolynomialValues<F>], row: usize, public_inputs: &[F]) -> F {
        self.products
            .iter()
            .map(|(col1, col2)| {
                col1.eval_table(table, row, public_inputs)
                    * col2.eval_table(table, row, public_inputs)
            })
            .sum::<F>()
            + self
                .constants
                .iter()
                .map(|col| col.eval_table(table, row, public_inputs))
                .sum()
    }

    /// Evaluates the filter on all rows, with the public inputs.
    /// Optimized for batch evaluation
    pub fn eval_all_rows_with_inputs(
        &self,
        table: &[PolynomialValues<F>],
        public_inputs: &[F],
    ) -> Vec<F> {
        let num_rows = table[0].len();
        let mut result = vec![F::ZERO; num_rows];

        // Accumulate product terms (degree-2): col1 * col2
        for (col1, col2) in &self.products {
            let col1_vals = col1.eval_all_rows_with_inputs(table, public_inputs);
            let col2_vals = col2.eval_all_rows_with_inputs(table, public_inputs);
            result
                .iter_mut()
                .zip(col1_vals.iter())
                .zip(col2_vals.iter())
                .for_each(|((r, v1), v2)| *r += *v1 * *v2);
        }

        // Accumulate linear terms
        for col in &self.constants {
            let col_vals = col.eval_all_rows_with_inputs(table, public_inputs);
            result
                .iter_mut()
                .zip(col_vals.iter())
                .for_each(|(r, v)| *r += *v);
        }

        result
    }

    /// Evaluates the filter on all rows.
    /// Optimized for batch evaluation
    pub fn eval_all_rows(&self, table: &[PolynomialValues<F>]) -> Vec<F> {
        self.eval_all_rows_with_inputs(table, &[])
    }

    /// All STARK columns contributing to this filter.
    pub fn relevant_columns(&self) -> Vec<usize> {
        self.products
            .iter()
            .flat_map(|(col1, col2)| {
                col1.relevant_columns()
                    .into_iter()
                    .chain(col2.relevant_columns())
            })
            .chain(self.constants.iter().flat_map(|col| col.relevant_columns()))
            .unique()
            .collect()
    }
}

/// A virtual column: a linear expression evaluated on a trace row. Writing `v`/`v'` for the
/// current/next row values and `PI` for the table's public inputs, a `Column` represents
///
/// `Σ f_i·v[c_i] + Σ g_j·v'[c_j] + Σ PI[s]·v[c] + Σ PI[s] + constant`
///
/// with the four sums given by `linear_combination`, `next_row_linear_combination`,
/// `public_input_scaled` and `public_input_offsets`. Public inputs are per-proof scalars, so
/// the public-input terms leave the expression's degree in the trace unchanged (the scaled
/// terms are degree 1, the offsets degree 0); they let one fixed expression — hence one
/// compiled verifier circuit — cover every instance geometry.
///
/// The `public_inputs` slice passed to any evaluation method must cover every slot the
/// expression references (out-of-range slots panic); expressions without public-input terms
/// never read the slice, so `&[]` is fine for them.
#[derive(Clone)]
pub struct Column<F: Field> {
    linear_combination: Vec<(usize, F)>,
    next_row_linear_combination: Vec<(usize, F)>,
    /// Terms `PI[s]·v[c]` (current row), one per `(c, s)` pair.
    public_input_scaled: Vec<(usize, usize)>,
    /// Terms `PI[s]`, one per slot `s`, added to the constant.
    public_input_offsets: Vec<usize>,
    constant: F,
}

impl<F: Field> Debug for Column<F> {
    /// Prints the public-input fields only when non-empty: expressions without such terms
    /// keep their pre-public-input `derive(Debug)` output, which downstream fingerprint
    /// tests hash to pin frozen protocol versions.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut s = f.debug_struct("Column");
        s.field("linear_combination", &self.linear_combination)
            .field(
                "next_row_linear_combination",
                &self.next_row_linear_combination,
            );
        if !self.public_input_scaled.is_empty() {
            s.field("public_input_scaled", &self.public_input_scaled);
        }
        if !self.public_input_offsets.is_empty() {
            s.field("public_input_offsets", &self.public_input_offsets);
        }
        s.field("constant", &self.constant).finish()
    }
}

impl<F: Field> Column<F> {
    /// Returns the representation of a single column in the current row.
    pub fn single(c: usize) -> Self {
        Self {
            linear_combination: vec![(c, F::ONE)],
            next_row_linear_combination: vec![],
            public_input_scaled: vec![],
            public_input_offsets: vec![],
            constant: F::ZERO,
        }
    }

    /// The single column `c` offset by a public input: `v[c] + PI[s]`.
    pub fn single_plus_public_input(c: usize, s: usize) -> Self {
        Self {
            public_input_offsets: vec![s],
            ..Self::single(c)
        }
    }

    /// A public input minus a single column: `PI[s] - v[c]`.
    pub fn public_input_minus_single(s: usize, c: usize) -> Self {
        Self {
            linear_combination: vec![(c, -F::ONE)],
            public_input_offsets: vec![s],
            ..Self::zero()
        }
    }

    /// `PI[s] - sum coeff·v[c]` over the given `(column, coeff)` terms.
    pub fn public_input_minus_linear_combination<I: IntoIterator<Item = (usize, F)>>(
        s: usize,
        iter: I,
    ) -> Self {
        Self {
            linear_combination: iter.into_iter().map(|(c, coeff)| (c, -coeff)).collect(),
            public_input_offsets: vec![s],
            ..Self::zero()
        }
    }

    /// The single column `c` scaled by a public input: `PI[s]·v[c]`.
    pub fn single_times_public_input(c: usize, s: usize) -> Self {
        Self {
            public_input_scaled: vec![(c, s)],
            ..Self::zero()
        }
    }

    /// A public input alone: `PI[s]`.
    pub fn public_input(s: usize) -> Self {
        Self {
            public_input_offsets: vec![s],
            ..Self::zero()
        }
    }

    /// A public-input-scaled column plus a plain column: `PI[s]·v[c_scaled] + v[c]`.
    pub fn single_times_public_input_plus_single(c_scaled: usize, s: usize, c: usize) -> Self {
        Self {
            public_input_scaled: vec![(c_scaled, s)],
            ..Self::single(c)
        }
    }

    /// A public-input-scaled column plus a public-input offset: `PI[s]·v[c] + PI[s_off]`.
    pub fn single_times_public_input_plus_public_input(c: usize, s: usize, s_off: usize) -> Self {
        Self {
            public_input_scaled: vec![(c, s)],
            public_input_offsets: vec![s_off],
            ..Self::zero()
        }
    }

    /// Returns multiple single columns in the current row.
    pub fn singles<I: IntoIterator<Item = impl Borrow<usize>>>(
        cs: I,
    ) -> impl Iterator<Item = Self> {
        cs.into_iter().map(|c| Self::single(*c.borrow()))
    }

    /// Returns the representation of a single column in the next row.
    pub fn single_next_row(c: usize) -> Self {
        Self {
            next_row_linear_combination: vec![(c, F::ONE)],
            ..Self::zero()
        }
    }

    /// Returns multiple single columns for the next row.
    pub fn singles_next_row<I: IntoIterator<Item = impl Borrow<usize>>>(
        cs: I,
    ) -> impl Iterator<Item = Self> {
        cs.into_iter().map(|c| Self::single_next_row(*c.borrow()))
    }

    /// Returns a linear combination corresponding to a constant.
    pub fn constant(constant: F) -> Self {
        Self {
            linear_combination: vec![],
            next_row_linear_combination: vec![],
            public_input_scaled: vec![],
            public_input_offsets: vec![],
            constant,
        }
    }

    /// Returns a linear combination corresponding to 0.
    pub fn zero() -> Self {
        Self::constant(F::ZERO)
    }

    /// Returns a linear combination corresponding to 1.
    pub fn one() -> Self {
        Self::constant(F::ONE)
    }

    /// Given an iterator of `(usize, F)` and a constant, returns the association linear combination of columns for the current row.
    pub fn linear_combination_with_constant<I: IntoIterator<Item = (usize, F)>>(
        iter: I,
        constant: F,
    ) -> Self {
        let v = iter.into_iter().collect::<Vec<_>>();
        assert!(!v.is_empty());

        // Because this is a debug assertion, we only check it when the `std`
        // feature is activated, as `Itertools::unique` relies on collections.
        #[cfg(feature = "std")]
        debug_assert_eq!(
            v.iter().map(|(c, _)| c).unique().count(),
            v.len(),
            "Duplicate columns."
        );

        Self {
            linear_combination: v,
            ..Self::constant(constant)
        }
    }

    /// Given an iterator of `(usize, F)` and a constant, returns the associated linear combination of columns for the current and the next rows.
    pub fn linear_combination_and_next_row_with_constant<I: IntoIterator<Item = (usize, F)>>(
        iter: I,
        next_row_iter: I,
        constant: F,
    ) -> Self {
        let v = iter.into_iter().collect::<Vec<_>>();
        let next_row_v = next_row_iter.into_iter().collect::<Vec<_>>();

        assert!(!v.is_empty() || !next_row_v.is_empty());

        // Because these are debug assertions, we only check them when the `std`
        // feature is activated, as `Itertools::unique` relies on collections.
        #[cfg(feature = "std")]
        {
            debug_assert_eq!(
                v.iter().map(|(c, _)| c).unique().count(),
                v.len(),
                "Duplicate columns."
            );
            debug_assert_eq!(
                next_row_v.iter().map(|(c, _)| c).unique().count(),
                next_row_v.len(),
                "Duplicate columns."
            );
        }

        Self {
            linear_combination: v,
            next_row_linear_combination: next_row_v,
            ..Self::constant(constant)
        }
    }

    /// Returns a linear combination of columns, with no additional constant.
    pub fn linear_combination<I: IntoIterator<Item = (usize, F)>>(iter: I) -> Self {
        Self::linear_combination_with_constant(iter, F::ZERO)
    }

    /// Given an iterator of columns (c_0, ..., c_n) containing bits in little endian order:
    /// returns the representation of c_0 + 2 * c_1 + ... + 2^n * c_n.
    pub fn le_bits<I: IntoIterator<Item = impl Borrow<usize>>>(cs: I) -> Self {
        Self::linear_combination(cs.into_iter().map(|c| *c.borrow()).zip(F::TWO.powers()))
    }

    /// Given an iterator of columns (c_0, ..., c_n) containing bits in little endian order:
    /// returns the representation of c_0 + 2 * c_1 + ... + 2^n * c_n + k where `k` is an
    /// additional constant.
    pub fn le_bits_with_constant<I: IntoIterator<Item = impl Borrow<usize>>>(
        cs: I,
        constant: F,
    ) -> Self {
        Self::linear_combination_with_constant(
            cs.into_iter().map(|c| *c.borrow()).zip(F::TWO.powers()),
            constant,
        )
    }

    /// Given an iterator of columns (c_0, ..., c_n) containing bytes in little endian order:
    /// returns the representation of c_0 + 256 * c_1 + ... + 256^n * c_n.
    pub fn le_bytes<I: IntoIterator<Item = impl Borrow<usize>>>(cs: I) -> Self {
        Self::linear_combination(
            cs.into_iter()
                .map(|c| *c.borrow())
                .zip(F::from_canonical_u16(256).powers()),
        )
    }

    /// Given an iterator of columns, returns the representation of their sum.
    pub fn sum<I: IntoIterator<Item = impl Borrow<usize>>>(cs: I) -> Self {
        Self::linear_combination(cs.into_iter().map(|c| *c.borrow()).zip(repeat(F::ONE)))
    }

    /// If the column is a single column, returns its index.
    pub fn get_single_column_idx(&self) -> Option<usize> {
        if self.linear_combination.len() == 1
            && self.next_row_linear_combination.is_empty()
            && self.public_input_scaled.is_empty()
            && self.public_input_offsets.is_empty()
            && self.constant.is_zero()
            && self.linear_combination[0].1.is_one()
        {
            Some(self.linear_combination[0].0)
        } else {
            None
        }
    }

    /// All STARK columns contributing to this virtual column.
    pub fn relevant_columns(&self) -> Vec<usize> {
        self.linear_combination
            .iter()
            .chain(&self.next_row_linear_combination)
            .map(|&(c, _)| c)
            .chain(self.public_input_scaled.iter().map(|&(c, _)| c))
            .collect()
    }

    /// Given the column values for the current row and the table's public inputs, returns the
    /// evaluation of the expression (next-row terms are ignored).
    pub(crate) fn eval<FE, P, const D: usize>(&self, v: &[P], public_inputs: &[FE]) -> P
    where
        FE: FieldExtension<D, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        self.linear_combination
            .iter()
            .map(|&(c, f)| v[c] * FE::from_basefield(f))
            .sum::<P>()
            + self
                .public_input_scaled
                .iter()
                .map(|&(c, s)| v[c] * public_inputs[s])
                .sum::<P>()
            + self
                .public_input_offsets
                .iter()
                .map(|&s| public_inputs[s])
                .sum::<FE>()
            + FE::from_basefield(self.constant)
    }

    /// Given the column values for the current and next rows and the table's public inputs,
    /// evaluates the full expression.
    pub(crate) fn eval_with_next<FE, P, const D: usize>(
        &self,
        v: &[P],
        next_v: &[P],
        public_inputs: &[FE],
    ) -> P
    where
        FE: FieldExtension<D, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        self.linear_combination
            .iter()
            .map(|&(c, f)| v[c] * FE::from_basefield(f))
            .sum::<P>()
            + self
                .next_row_linear_combination
                .iter()
                .map(|&(c, f)| next_v[c] * FE::from_basefield(f))
                .sum::<P>()
            + self
                .public_input_scaled
                .iter()
                .map(|&(c, s)| v[c] * public_inputs[s])
                .sum::<P>()
            + self
                .public_input_offsets
                .iter()
                .map(|&s| public_inputs[s])
                .sum::<FE>()
            + FE::from_basefield(self.constant)
    }

    /// Evaluate on a row of a table given in column-major form.
    pub fn eval_table(&self, table: &[PolynomialValues<F>], row: usize, public_inputs: &[F]) -> F {
        self.linear_combination
            .iter()
            .map(|&(c, f)| table[c].values[row] * f)
            .sum::<F>()
            + self
                .next_row_linear_combination
                .iter()
                .map(|&(c, f)| table[c].values[(row + 1) % table[c].values.len()] * f)
                .sum::<F>()
            + self
                .public_input_scaled
                .iter()
                .map(|&(c, s)| table[c].values[row] * public_inputs[s])
                .sum::<F>()
            + self
                .public_input_offsets
                .iter()
                .map(|&s| public_inputs[s])
                .sum::<F>()
            + self.constant
    }

    /// Evaluates the column on all rows, with the public inputs.
    /// Optimized for batch evaluation.
    pub fn eval_all_rows_with_inputs(
        &self,
        table: &[PolynomialValues<F>],
        public_inputs: &[F],
    ) -> Vec<F> {
        let num_rows = table[0].len();

        // Row-independent part: the constant plus the public-input offsets.
        let constant = self
            .public_input_offsets
            .iter()
            .map(|&s| public_inputs[s])
            .sum::<F>()
            + self.constant;
        let mut result = vec![constant; num_rows];

        // Accumulate current-row linear combination terms, with the public-input-scaled
        // terms folded in as additional `(column, factor)` pairs.
        let scaled = self
            .public_input_scaled
            .iter()
            .map(|&(c, s)| (c, public_inputs[s]));
        for (col_idx, factor) in self.linear_combination.iter().copied().chain(scaled) {
            let col_values = &table[col_idx].values;
            result
                .iter_mut()
                .zip(col_values.iter())
                .for_each(|(r, v)| *r += *v * factor);
        }

        // Accumulate next-row linear combination terms
        for &(col_idx, factor) in &self.next_row_linear_combination {
            let col_values = &table[col_idx].values;
            for (i, r) in result.iter_mut().enumerate() {
                *r += col_values[(i + 1) % num_rows] * factor;
            }
        }

        result
    }

    /// Evaluates the column on all rows.
    /// Optimized for batch evaluation.
    pub fn eval_all_rows(&self, table: &[PolynomialValues<F>]) -> Vec<F> {
        self.eval_all_rows_with_inputs(table, &[])
    }

    /// Adds this expression's public-input terms to `acc`:
    /// `Σ PI[s]·v[c]` (scaled terms) and `Σ PI[s]` (offsets).
    fn add_public_input_terms_circuit<const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        v: &[ExtensionTarget<D>],
        public_inputs: &[ExtensionTarget<D>],
        mut acc: ExtensionTarget<D>,
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        for &(c, s) in &self.public_input_scaled {
            let term = builder.mul_extension(public_inputs[s], v[c]);
            acc = builder.add_extension(acc, term);
        }
        if !self.public_input_offsets.is_empty() {
            let offset = builder
                .add_many_extension(self.public_input_offsets.iter().map(|&s| public_inputs[s]));
            acc = builder.add_extension(acc, offset);
        }
        acc
    }

    /// Circuit version of `eval`: Given a row's targets, returns their linear combination.
    pub(crate) fn eval_circuit<const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        v: &[ExtensionTarget<D>],
        public_inputs: &[ExtensionTarget<D>],
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        let pairs = self
            .linear_combination
            .iter()
            .map(|&(c, f)| {
                (
                    v[c],
                    builder.constant_extension(F::Extension::from_basefield(f)),
                )
            })
            .collect::<Vec<_>>();
        let constant = builder.constant_extension(F::Extension::from_basefield(self.constant));
        let acc = builder.inner_product_extension(F::ONE, constant, pairs);
        self.add_public_input_terms_circuit(builder, v, public_inputs, acc)
    }

    /// Circuit version of `eval_with_next`:
    /// Given the targets of the current and next row, returns the sum of their linear combinations.
    pub(crate) fn eval_with_next_circuit<const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        v: &[ExtensionTarget<D>],
        next_v: &[ExtensionTarget<D>],
        public_inputs: &[ExtensionTarget<D>],
    ) -> ExtensionTarget<D>
    where
        F: RichField + Extendable<D>,
    {
        let mut pairs = self
            .linear_combination
            .iter()
            .map(|&(c, f)| {
                (
                    v[c],
                    builder.constant_extension(F::Extension::from_basefield(f)),
                )
            })
            .collect::<Vec<_>>();
        let next_row_pairs = self.next_row_linear_combination.iter().map(|&(c, f)| {
            (
                next_v[c],
                builder.constant_extension(F::Extension::from_basefield(f)),
            )
        });
        pairs.extend(next_row_pairs);
        let constant = builder.constant_extension(F::Extension::from_basefield(self.constant));
        let acc = builder.inner_product_extension(F::ONE, constant, pairs);
        self.add_public_input_terms_circuit(builder, v, public_inputs, acc)
    }
}

pub(crate) type ColumnFilter<'a, F> = (&'a [Column<F>], &'a Filter<F>);

/// A [`Lookup`] defines a set of `columns`` whose values should appear in a
/// `table_column` (i.e. the lookup table associated to these looking columns),
/// along with a `frequencies_column` indicating the frequency of each looking
/// column in the looked table.
///
/// It also features a `filter_columns` vector, optionally adding at most one
/// filter per looking column.
///
/// The lookup argumented implemented here is based on logarithmic derivatives,
/// a technique described with the whole lookup protocol in
/// <https://eprint.iacr.org/2022/1530>.
#[derive(Debug, Clone)]
pub struct Lookup<F: Field> {
    /// Columns whose values should be contained in the lookup table.
    /// These are the f_i(x) polynomials in the logUp paper.
    pub columns: Vec<Column<F>>,
    /// Column containing the lookup table.
    /// This is the t(x) polynomial in the logUp paper.
    pub table_column: Column<F>,
    /// Column containing the frequencies of `columns` in `table_column`.
    /// This is the m(x) polynomial in the paper.
    pub frequencies_column: Column<F>,

    /// Columns to filter some elements. There is at most one filter
    /// column per column to lookup.
    pub filter_columns: Vec<Filter<F>>,
}

impl<F: Field> Lookup<F> {
    /// Outputs the number of helper columns needed by this [`Lookup`].
    pub fn num_helper_columns(&self, constraint_degree: usize) -> usize {
        // One helper column for each column batch of size `constraint_degree-1`,
        // then one column for the inverse of `table + challenge` and one for the `Z` polynomial.

        self.columns
            .len()
            .div_ceil(constraint_degree.checked_sub(1).unwrap_or(1))
            + 1
    }

    /// Returns all trace column indices referenced by this Lookup.
    pub fn all_column_indices(&self) -> Vec<usize> {
        self.columns
            .iter()
            .flat_map(|c| c.relevant_columns())
            .chain(self.table_column.relevant_columns())
            .chain(self.frequencies_column.relevant_columns())
            .chain(
                self.filter_columns
                    .iter()
                    .flat_map(|f| f.relevant_columns()),
            )
            .unique()
            .collect()
    }
}

/// Randomness for a single instance of a permutation check protocol.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct GrandProductChallenge<T: Copy + Eq + PartialEq + Debug> {
    /// Randomness used to combine multiple columns into one.
    pub beta: T,
    /// Random offset that's added to the beta-reduced column values.
    pub gamma: T,
}

impl<F: Field> GrandProductChallenge<F> {
    /// Combines a series of values `t_i` with these challenge random values.
    /// In particular, given `beta` and `gamma` challenges, this will compute
    /// `(Σ t_i * beta^i) + gamma`.
    pub fn combine<'a, FE, P, T: IntoIterator<Item = &'a P>, const D2: usize>(&self, terms: T) -> P
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
        T::IntoIter: DoubleEndedIterator,
    {
        reduce_with_powers(terms, FE::from_basefield(self.beta)) + FE::from_basefield(self.gamma)
    }

    /// Like `combine`, but takes an iterator of owned values instead of references.
    pub fn combine_base<T: IntoIterator<Item = F>>(&self, terms: T) -> F
    where
        T::IntoIter: DoubleEndedIterator,
    {
        terms
            .into_iter()
            .rev()
            .reduce(|acc, term| acc * self.beta + term)
            .map_or(self.gamma, |sum| sum + self.gamma)
    }
}

impl GrandProductChallenge<Target> {
    pub(crate) fn combine_circuit<F: RichField + Extendable<D>, const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        terms: &[ExtensionTarget<D>],
    ) -> ExtensionTarget<D> {
        let reduced = reduce_with_powers_ext_circuit(builder, terms, self.beta);
        let gamma = builder.convert_to_ext(self.gamma);
        builder.add_extension(reduced, gamma)
    }
}

impl GrandProductChallenge<Target> {
    /// Circuit version of `combine`.
    pub fn combine_base_circuit<F: RichField + Extendable<D>, const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        terms: &[Target],
    ) -> Target {
        let reduced = reduce_with_powers_circuit(builder, terms, self.beta);
        builder.add(reduced, self.gamma)
    }
}

/// Like `GrandProductChallenge`, but with `num_challenges` copies to boost soundness.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct GrandProductChallengeSet<T: Copy + Eq + PartialEq + Debug> {
    /// A sequence of `num_challenges` challenge pairs, where `num_challenges`
    /// is defined in [`StarkConfig`][crate::config::StarkConfig].
    pub challenges: Vec<GrandProductChallenge<T>>,
}

impl GrandProductChallengeSet<Target> {
    /// Serializes this `GrandProductChallengeSet` of `Target`s.
    pub fn to_buffer(&self, buffer: &mut Vec<u8>) -> IoResult<()> {
        buffer.write_usize(self.challenges.len())?;
        for challenge in &self.challenges {
            buffer.write_target(challenge.beta)?;
            buffer.write_target(challenge.gamma)?;
        }
        Ok(())
    }

    /// Serializes a `GrandProductChallengeSet` of `Target`s from the provided buffer.
    pub fn from_buffer(buffer: &mut Buffer) -> IoResult<Self> {
        let length = buffer.read_usize()?;
        let mut challenges = Vec::with_capacity(length);
        for _ in 0..length {
            challenges.push(GrandProductChallenge {
                beta: buffer.read_target()?,
                gamma: buffer.read_target()?,
            });
        }

        Ok(GrandProductChallengeSet { challenges })
    }
}

fn get_grand_product_challenge<F: RichField, H: Hasher<F>>(
    challenger: &mut Challenger<F, H>,
) -> GrandProductChallenge<F> {
    let beta = challenger.get_challenge();
    let gamma = challenger.get_challenge();
    GrandProductChallenge { beta, gamma }
}

/// Generates a new `GrandProductChallengeSet` containing `num_challenges`
/// pairs of challenges from the current `challenger` state.
pub fn get_grand_product_challenge_set<F: RichField, H: Hasher<F>>(
    challenger: &mut Challenger<F, H>,
    num_challenges: usize,
) -> GrandProductChallengeSet<F> {
    let challenges = (0..num_challenges)
        .map(|_| get_grand_product_challenge(challenger))
        .collect();
    GrandProductChallengeSet { challenges }
}

fn get_grand_product_challenge_target<
    F: RichField + Extendable<D>,
    H: AlgebraicHasher<F>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    challenger: &mut RecursiveChallenger<F, H, D>,
) -> GrandProductChallenge<Target> {
    let beta = challenger.get_challenge(builder);
    let gamma = challenger.get_challenge(builder);
    GrandProductChallenge { beta, gamma }
}

/// Circuit version of `get_grand_product_challenge_set`.
pub fn get_grand_product_challenge_set_target<
    F: RichField + Extendable<D>,
    H: AlgebraicHasher<F>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    challenger: &mut RecursiveChallenger<F, H, D>,
    num_challenges: usize,
) -> GrandProductChallengeSet<Target> {
    let challenges = (0..num_challenges)
        .map(|_| get_grand_product_challenge_target(builder, challenger))
        .collect();
    GrandProductChallengeSet { challenges }
}

/// logUp protocol from <https://ia.cr/2022/1530>
/// Compute the helper columns for the lookup argument.
/// Given columns `f0,...,fk` and a column `t`, such that `∪fi ⊆ t`, and challenges `x`,
/// this computes the helper columns `h_i = 1/(x+f_2i) + 1/(x+f_2i+1)`, `g = 1/(x+t)`,
/// and `Z(gx) = Z(x) + sum h_i(x) - m(x)g(x)` where `m` is the frequencies column.
pub(crate) fn lookup_helper_columns<F: Field>(
    lookup: &Lookup<F>,
    trace_poly_values: &[PolynomialValues<F>],
    public_inputs: &[F],
    challenge: F,
    constraint_degree: usize,
) -> Vec<PolynomialValues<F>> {
    assert_eq!(lookup.columns.len(), lookup.filter_columns.len());

    let num_total_logup_entries = trace_poly_values[0].values.len() * lookup.columns.len();
    assert!(BigUint::from(num_total_logup_entries) < F::characteristic());

    let num_helper_columns = lookup.num_helper_columns(constraint_degree);

    let looking_cols = lookup
        .columns
        .iter()
        .map(|col| vec![col.clone()])
        .collect::<Vec<Vec<Column<F>>>>();

    let grand_challenge = GrandProductChallenge {
        beta: F::ONE,
        gamma: challenge,
    };

    let columns_filters = looking_cols
        .iter()
        .zip(lookup.filter_columns.iter())
        .map(|(col, filter)| (&col[..], filter))
        .collect::<Vec<_>>();
    // For each batch of `constraint_degree-1` columns `fi`, compute `sum 1/(f_i+challenge)` and
    // add it to the helper columns.
    // Note: these are the h_k(x) polynomials in the paper, with a few differences:
    //       * Here, the first ratio m_0(x)/phi_0(x) is not included with the columns batched up to create the
    //         h_k polynomials; instead there's a separate helper column for it (see below).
    //       * Here, we use 1 instead of -1 as the numerator (and subtract later).
    //       * Here, for now, the batch size (l) is always constraint_degree - 1 = 2.
    //       * Here, there are filters for the columns, to only select some rows
    //         in a given column.
    let mut helper_columns = get_helper_cols(
        trace_poly_values,
        trace_poly_values[0].len(),
        &columns_filters,
        public_inputs,
        grand_challenge,
        constraint_degree,
    );

    // Add `1/(table+challenge)` to the helper columns.
    // This is 1/phi_0(x) = 1/(x + t(x)) from the paper.
    // Here, we don't include m(x) in the numerator, instead multiplying it with this column later.
    let mut table = lookup
        .table_column
        .eval_all_rows_with_inputs(trace_poly_values, public_inputs);
    for x in table.iter_mut() {
        *x = challenge + *x;
    }
    let table_inverse: Vec<F> = F::batch_multiplicative_inverse(&table);

    // Compute the `Z` polynomial with `Z(1)=0` and `Z(gx) = Z(x) + sum h_i(x) - frequencies(x)g(x)`.
    // This enforces the check from the paper, that the sum of the h_k(x) polynomials is 0 over H.
    // In the paper, that sum includes m(x)/(x + t(x)) = frequencies(x)/g(x), because that was bundled
    // into the h_k(x) polynomials.
    let frequencies = &lookup
        .frequencies_column
        .eval_all_rows_with_inputs(trace_poly_values, public_inputs);
    let mut z = Vec::with_capacity(frequencies.len());
    z.push(F::ZERO);
    for i in 0..frequencies.len() - 1 {
        let x = helper_columns[..num_helper_columns - 1]
            .iter()
            .map(|col| col.values[i])
            .sum::<F>()
            - frequencies[i] * table_inverse[i];
        z.push(z[i] + x);
    }
    helper_columns.push(z.into());

    helper_columns
}

/// Core constraint generation for helper columns.
/// Takes closures that compute combined and filter values on demand (avoids vector allocation).
fn constrain_helper_columns<P: PackedField>(
    num_columns: usize,
    shifted_looked_values: impl Fn(usize) -> P,
    filter_val: impl Fn(usize) -> P,
    helper_columns: &[P],
    constraint_degree: usize,
    consumer: &mut ConstraintConsumer<P>,
) {
    let chunk_size = constraint_degree.checked_sub(1).unwrap_or(1);
    let mut col_idx = 0;
    for &h in helper_columns {
        let remaining = num_columns - col_idx;
        match remaining.min(chunk_size) {
            2 => {
                let (c0, c1) = (
                    shifted_looked_values(col_idx),
                    shifted_looked_values(col_idx + 1),
                );
                let (f0, f1) = (filter_val(col_idx), filter_val(col_idx + 1));
                consumer.constraint(c1 * c0 * h - f0 * c1 - f1 * c0);
                col_idx += 2;
            }
            1 => {
                consumer.constraint(shifted_looked_values(col_idx) * h - filter_val(col_idx));
                col_idx += 1;
            }
            _ => todo!("Allow other constraint degrees"),
        }
    }
}

/// Given data associated to a lookup, check the associated helper polynomials.
pub(crate) fn eval_helper_columns<F, FE, P, const D: usize, const D2: usize>(
    filter: &[Filter<F>],
    columns: &[Vec<P>],
    local_values: &[P],
    next_values: &[P],
    public_inputs: &[FE],
    helper_columns: &[P],
    constraint_degree: usize,
    challenges: &GrandProductChallenge<F>,
    consumer: &mut ConstraintConsumer<P>,
) where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    if helper_columns.is_empty() {
        return;
    }
    constrain_helper_columns(
        columns.len(),
        |i| challenges.combine(&columns[i]),
        |i| filter[i].eval_filter(local_values, next_values, public_inputs),
        helper_columns,
        constraint_degree,
        consumer,
    );
}

/// Circuit version of `eval_helper_columns`.
/// Given data associated to a lookup (either a CTL or a range-check), check the associated helper polynomials.
pub(crate) fn eval_helper_columns_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    filter: &[Filter<F>],
    columns: &[Vec<ExtensionTarget<D>>],
    local_values: &[ExtensionTarget<D>],
    next_values: &[ExtensionTarget<D>],
    public_inputs: &[ExtensionTarget<D>],
    helper_columns: &[ExtensionTarget<D>],
    constraint_degree: usize,
    challenges: &GrandProductChallenge<Target>,
    consumer: &mut RecursiveConstraintConsumer<F, D>,
) {
    if !helper_columns.is_empty() {
        let chunk_size = constraint_degree.checked_sub(1).unwrap_or(1);
        for (chunk, (fs, &h)) in columns
            .chunks(chunk_size)
            .zip(filter.chunks(chunk_size).zip(helper_columns))
        {
            match chunk.len() {
                2 => {
                    let combin0 = challenges.combine_circuit(builder, &chunk[0]);
                    let combin1 = challenges.combine_circuit(builder, &chunk[1]);

                    let f0 = fs[0].eval_filter_circuit(
                        builder,
                        local_values,
                        next_values,
                        public_inputs,
                    );
                    let f1 = fs[1].eval_filter_circuit(
                        builder,
                        local_values,
                        next_values,
                        public_inputs,
                    );

                    let constr = builder.mul_sub_extension(combin0, h, f0);
                    let constr = builder.mul_extension(constr, combin1);
                    let f1_constr = builder.mul_extension(f1, combin0);
                    let constr = builder.sub_extension(constr, f1_constr);

                    consumer.constraint(builder, constr);
                }
                1 => {
                    let combin = challenges.combine_circuit(builder, &chunk[0]);
                    let f0 = fs[0].eval_filter_circuit(
                        builder,
                        local_values,
                        next_values,
                        public_inputs,
                    );
                    let constr = builder.mul_sub_extension(combin, h, f0);
                    consumer.constraint(builder, constr);
                }

                _ => todo!("Allow other constraint degrees"),
            }
        }
    }
}

/// Given a STARK's trace, and the data associated to one lookup (either CTL or range check),
/// returns the associated helper polynomials.
pub(crate) fn get_helper_cols<F: Field>(
    trace: &[PolynomialValues<F>],
    degree: usize,
    columns_filters: &[ColumnFilter<F>],
    public_inputs: &[F],
    challenge: GrandProductChallenge<F>,
    constraint_degree: usize,
) -> Vec<PolynomialValues<F>> {
    let num_helper_columns = columns_filters
        .len()
        .div_ceil(constraint_degree.checked_sub(1).unwrap_or(1));

    let chunks = columns_filters.chunks(constraint_degree.checked_sub(1).unwrap_or(1));
    let helper_columns: Vec<_> = chunks
        .filter_map(|cols_filts| {
            cols_filts
                .iter()
                .map(|(col, filter)| {
                    // Optimization: for single-element columns, combine([x]) = x + gamma (no beta multiplication)
                    let mut combined = if col.len() == 1 {
                        let mut col_values = col[0].eval_all_rows_with_inputs(trace, public_inputs);
                        col_values.iter_mut().for_each(|v| *v += challenge.gamma);
                        col_values
                    } else {
                        // Multi-element columns: precompute all rows, combine via iterator
                        let col_all_rows: Vec<Vec<F>> = col
                            .iter()
                            .map(|c| c.eval_all_rows_with_inputs(trace, public_inputs))
                            .collect();
                        (0..degree)
                            .map(|d| {
                                challenge
                                    .combine_base(col_all_rows.iter().map(|col_vals| col_vals[d]))
                            })
                            .collect()
                    };

                    combined = F::batch_multiplicative_inverse(&combined);
                    let filter_col = filter.eval_all_rows_with_inputs(trace, public_inputs);
                    batch_multiply_inplace(&mut combined, &filter_col);
                    combined
                })
                .reduce(|mut acc, combined| {
                    batch_add_inplace(&mut acc, &combined);
                    acc
                })
                .map(PolynomialValues::from)
        })
        .collect();
    assert_eq!(helper_columns.len(), num_helper_columns);

    helper_columns
}

/// Evaluations of the lookup helper columns at a point (or batch of points),
/// together with the lookup challenges, as needed to check the lookup
/// argument constraints.
#[derive(Debug)]
pub struct LookupCheckVars<F, FE, P, const D2: usize>
where
    F: Field,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    pub(crate) local_values: Vec<P>,
    pub(crate) next_values: Vec<P>,
    pub(crate) challenges: Vec<F>,
}

impl<F, FE, P, const D2: usize> LookupCheckVars<F, FE, P, D2>
where
    F: Field,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    /// Bundles the local and next helper column evaluations with the lookup
    /// challenges.
    pub fn new(local_values: Vec<P>, next_values: Vec<P>, challenges: Vec<F>) -> Self {
        Self {
            local_values,
            next_values,
            challenges,
        }
    }
}

/// Constraints for the logUp lookup argument.
pub(crate) fn eval_packed_lookups_generic<F, FE, P, S, const D: usize, const D2: usize>(
    stark: &S,
    lookups: &[Lookup<F>],
    vars: &S::EvaluationFrame<FE, P, D2>,
    lookup_vars: LookupCheckVars<F, FE, P, D2>,
    yield_constr: &mut ConstraintConsumer<P>,
) where
    F: RichField + Extendable<D>,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
    S: Stark<F, D>,
{
    let local_values = vars.get_local_values();
    let next_values = vars.get_next_values();
    let public_inputs = vars.get_public_inputs();
    let degree = stark.constraint_degree();
    let mut start = 0;
    for lookup in lookups {
        let num_helper_columns = lookup.num_helper_columns(degree);

        // Precompute challenge-independent evaluations (flat array, no nested vecs)
        let lookup_columns: Vec<P> = lookup
            .columns
            .iter()
            .map(|col| col.eval_with_next(local_values, next_values, public_inputs))
            .collect();
        let table_eval = lookup.table_column.eval(local_values, public_inputs);
        let freq_eval = lookup.frequencies_column.eval(local_values, public_inputs);

        for &challenge in &lookup_vars.challenges {
            let gamma = FE::from_basefield(challenge);
            let helper_columns = &lookup_vars.local_values[start..start + num_helper_columns - 1];

            // For each chunk, check that `h_i (x+f_2i) (x+f_{2i+1}) = (x+f_2i) * filter_{2i+1} + (x+f_{2i+1}) * filter_2i`
            // if the chunk has length 2 or if it has length 1, check that `h_i * (x+f_2i) = filter_2i`, where x is the challenge
            if !helper_columns.is_empty() {
                constrain_helper_columns(
                    lookup_columns.len(),
                    |i| lookup_columns[i] + gamma,
                    |i| {
                        lookup.filter_columns[i].eval_filter(
                            local_values,
                            next_values,
                            public_inputs,
                        )
                    },
                    helper_columns,
                    degree,
                    yield_constr,
                );
            }

            // Check the `Z` polynomial.
            let z = lookup_vars.local_values[start + num_helper_columns - 1];
            let next_z = lookup_vars.next_values[start + num_helper_columns - 1];
            let table_with_challenge = table_eval + gamma;
            let helper_sum: P = lookup_vars.local_values[start..start + num_helper_columns - 1]
                .iter()
                .copied()
                .sum();
            let y = helper_sum * table_with_challenge - freq_eval;
            // Check that in the first row, z = 0;
            yield_constr.constraint_first_row(z);
            yield_constr.constraint((next_z - z) * table_with_challenge - y);
            start += num_helper_columns;
        }
    }
}

#[derive(Debug)]
pub(crate) struct LookupCheckVarsTarget<const D: usize> {
    pub(crate) local_values: Vec<ExtensionTarget<D>>,
    pub(crate) next_values: Vec<ExtensionTarget<D>>,
    pub(crate) challenges: Vec<Target>,
}

pub(crate) fn eval_ext_lookups_circuit<
    F: RichField + Extendable<D>,
    S: Stark<F, D>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    stark: &S,
    vars: &S::EvaluationFrameTarget,
    lookup_vars: LookupCheckVarsTarget<D>,
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let degree = stark.constraint_degree();
    let lookups = stark.lookups();

    let local_values = vars.get_local_values();
    let next_values = vars.get_next_values();
    let public_inputs = vars.get_public_inputs();
    let mut start = 0;
    for lookup in lookups {
        let num_helper_columns = lookup.num_helper_columns(degree);
        let col_values = lookup
            .columns
            .iter()
            .map(|col| {
                vec![col.eval_with_next_circuit(builder, local_values, next_values, public_inputs)]
            })
            .collect::<Vec<_>>();

        for &challenge in &lookup_vars.challenges {
            let grand_challenge = GrandProductChallenge {
                beta: builder.one(),
                gamma: challenge,
            };

            eval_helper_columns_circuit(
                builder,
                &lookup.filter_columns,
                &col_values,
                local_values,
                next_values,
                public_inputs,
                &lookup_vars.local_values[start..start + num_helper_columns - 1],
                degree,
                &grand_challenge,
                yield_constr,
            );
            let challenge = builder.convert_to_ext(challenge);

            let z = lookup_vars.local_values[start + num_helper_columns - 1];
            let next_z = lookup_vars.next_values[start + num_helper_columns - 1];
            let table_column =
                lookup
                    .table_column
                    .eval_circuit(builder, vars.get_local_values(), public_inputs);
            let table_with_challenge = builder.add_extension(table_column, challenge);
            let mut y = builder.add_many_extension(
                &lookup_vars.local_values[start..start + num_helper_columns - 1],
            );

            let frequencies_column = lookup.frequencies_column.eval_circuit(
                builder,
                vars.get_local_values(),
                public_inputs,
            );
            y = builder.mul_extension(y, table_with_challenge);
            y = builder.sub_extension(y, frequencies_column);

            // Check that in the first row, z = 0;
            yield_constr.constraint_first_row(builder, z);
            let mut constraint = builder.sub_extension(next_z, z);
            constraint = builder.mul_extension(constraint, table_with_challenge);
            constraint = builder.sub_extension(constraint, y);
            yield_constr.constraint(builder, constraint);
            start += num_helper_columns;
        }
    }
}
