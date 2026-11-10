//! AIR for one binary32 `alpha * block_scale` coefficient per eight input elements.
//!
//! # Certified operation
//!
//! InputQuant's signal is `c*q`, with raw int8 `q` and
//! `c = RN32(alpha*block_scale)`. The row scale `alpha` is positive-normal bf16; the
//! signed block scale is finite bf16. With `M(alpha)=128+alpha_fraction` and
//! `M(scale)=128*[scale_exponent!=0]+scale_fraction`, this AIR proves
//!
//! ```text
//! P = M(alpha)*M(scale) < 2^16
//! t = E(alpha)+E*(scale)-268         if P != 0, else -149
//! |c| = RN32(P*2^t).
//! ```
//!
//! The shared `super::fp32` rounding certificate proves normalization, subnormal fade,
//! the division remainder, ties-to-even, and any significand carry. Product/inverse
//! equations bind the zero flag. The scale sign is not needed to round the magnitude;
//! InputQuant restores it when forming the signed signal. A bf16 product has at most
//! 16 significant bits, so the rounded coefficient times an int8 fits binary32's 24-bit
//! precision on the finite domain. This avoids a second signal-product rounding gadget.
//!
//! # CTL contract and schedule
//!
//! The trace has `(h+w)*k/8` live rows, first A blocks then B blocks, padded to a power
//! of two. Verifier-known `(block_key,is_live)` columns fix the enumeration. Padding
//! uses `alpha=1, scale=0`, hence a zero coefficient, and emits no CTL tuple.
//!
//! Each live row exports
//! `(block_key, alpha fields, scale magnitude fields, coefficient magnitude fields)`.
//! InputQuant looks up this tuple once at each live block start and copies the coefficient
//! across that block. The common key and input fields bind it to the same authenticated
//! scale and Scale-certified alpha used by that block; a valid but unrelated coefficient
//! cannot be substituted. Input domains are inherited from this live InputQuant binding;
//! the local rounding lookups certify output fields, powers, and integer ranges.
//!
//! There are no inter-row arithmetic constraints. Native and recursive verification use
//! the same degree-three evaluator; the CTL and committed-LUT arguments complete the AIR.

use anyhow::{Result, ensure};
use core::{borrow::Borrow, marker::PhantomData};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::cross_table_lookup::{TableIdx, TableWithColumns};
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use starky::lookup::{Column, Filter};
use starky::stark::Stark;

use super::columns_view::columns_view;
use super::fp32::{Round32Columns, evaluate_rounding, rounding_lookups};
use super::luts::ctl::LutLookup;
use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CoefficientColumns<T: Copy> {
    pub block_key: T,
    pub is_live: T,
    pub alpha_exponent: T,
    pub alpha_fraction: T,
    pub scale_exponent: T,
    pub scale_fraction: T,
    pub scale_exponent_is_zero: T,
    pub product: T,
    pub product_inverse: T,
    pub product_is_zero: T,
    pub binary_scale: T,
    pub rounding: Round32Columns<T>,
}

pub const NUM_COEFFICIENT_COLUMNS: usize = size_of::<CoefficientColumns<u8>>();
pub const NUM_COEFFICIENT_KNOWN_COLUMNS: usize = 2;
const _: () = assert!(NUM_COEFFICIENT_COLUMNS == 36);
columns_view!(CoefficientColumns, NUM_COEFFICIENT_COLUMNS, COEFFICIENT_COLUMNS);

pub(crate) fn coefficient_looked<F: Field>(table: TableIdx) -> TableWithColumns<F> {
    let columns = &COEFFICIENT_COLUMNS;
    let output = &columns.rounding.output;
    TableWithColumns::new(
        table,
        Column::singles([
            columns.block_key,
            columns.alpha_exponent,
            columns.alpha_fraction,
            columns.scale_exponent,
            columns.scale_fraction,
            columns.scale_exponent_is_zero,
            output.exponent,
            output.fraction_low,
            output.fraction_high,
            output.is_exponent_zero,
        ])
        .collect(),
        Filter::from_column(Column::single(columns.is_live)),
    )
}

pub(crate) fn coefficient_lookups<F: Field>() -> Vec<LutLookup<F>> {
    rounding_lookups(&COEFFICIENT_COLUMNS.rounding, COEFFICIENT_COLUMNS.binary_scale)
}

#[derive(Clone, Debug)]
pub struct CoefficientStark<F: RichField + Extendable<D>, const D: usize> {
    live_blocks: usize,
    marker: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> CoefficientStark<F, D> {
    pub fn new(h: usize, w: usize, k: usize) -> Self {
        assert!(k > 0 && k.is_multiple_of(8));
        Self {
            live_blocks: (h + w) * (k / 8),
            marker: PhantomData,
        }
    }

    /// Geometry-derived trace height, including power-of-two padding.
    pub fn num_rows(&self) -> usize {
        self.live_blocks.next_power_of_two()
    }

    pub fn known_values(&self) -> Vec<PolynomialValues<F>> {
        let height = self.num_rows();
        vec![
            PolynomialValues::new(
                (0..height)
                    .map(|i| {
                        if i < self.live_blocks {
                            F::from_canonical_usize(i)
                        } else {
                            F::ZERO
                        }
                    })
                    .collect(),
            ),
            PolynomialValues::new((0..height).map(|i| F::from_bool(i < self.live_blocks)).collect()),
        ]
    }

    pub fn generate_trace(&self, blocks: &[[u16; 2]]) -> Result<Vec<[F; NUM_COEFFICIENT_COLUMNS]>> {
        ensure!(
            blocks.len() == self.live_blocks,
            "coefficient block count differs from geometry"
        );
        (0..self.live_blocks.next_power_of_two())
            .map(|index| {
                let [alpha, scale] = blocks.get(index).copied().unwrap_or([0x3f80, 0]);
                let alpha_exponent = u64::from(alpha >> 7);
                let alpha_fraction = u64::from(alpha & 127);
                let scale_exponent = u64::from((scale >> 7) & 255);
                let scale_fraction = u64::from(scale & 127);
                ensure!(
                    (1..=254).contains(&alpha_exponent) && scale_exponent <= 254,
                    "nonfinite coefficient input"
                );
                let scale_exponent_is_zero = scale_exponent == 0;
                let product = (128 + alpha_fraction) * (128 * u64::from(!scale_exponent_is_zero) + scale_fraction);
                let binary_scale = if product == 0 {
                    -149
                } else {
                    alpha_exponent as i64 + scale_exponent as i64 + i64::from(scale_exponent_is_zero) - 268
                };
                let product_field = F::from_canonical_u64(product);
                Ok(CoefficientColumns {
                    block_key: if index < blocks.len() {
                        F::from_canonical_usize(index)
                    } else {
                        F::ZERO
                    },
                    is_live: F::from_bool(index < blocks.len()),
                    alpha_exponent: F::from_canonical_u64(alpha_exponent),
                    alpha_fraction: F::from_canonical_u64(alpha_fraction),
                    scale_exponent: F::from_canonical_u64(scale_exponent),
                    scale_fraction: F::from_canonical_u64(scale_fraction),
                    scale_exponent_is_zero: F::from_bool(scale_exponent_is_zero),
                    product: product_field,
                    product_inverse: if product == 0 { F::ZERO } else { product_field.inverse() },
                    product_is_zero: F::from_bool(product == 0),
                    binary_scale: if binary_scale < 0 {
                        -F::from_canonical_u64(binary_scale.unsigned_abs())
                    } else {
                        F::from_canonical_u64(binary_scale as u64)
                    },
                    rounding: Round32Columns::generate(product, binary_scale, false, false)?,
                }
                .into())
            })
            .collect()
    }
}

fn evaluate<V: Copy + Default, S: Copy, E: Evaluator<V, S>>(columns: &CoefficientColumns<V>, evaluator: &mut E) {
    let one = evaluator.i32(1);
    let zero = evaluator.i32(0);
    let significand_base = evaluator.i32(128);
    let normal_scale = evaluator.sub(one, columns.scale_exponent_is_zero);
    let scale_significand = evaluator.mad(normal_scale, significand_base, columns.scale_fraction);
    let alpha_significand = evaluator.add(significand_base, columns.alpha_fraction);
    let product = evaluator.mul(alpha_significand, scale_significand);
    evaluator.constraint_eq(columns.product, product);
    let exponent_sum = evaluator.add(columns.alpha_exponent, columns.scale_exponent);
    let exponent_sum = evaluator.add(exponent_sum, columns.scale_exponent_is_zero);
    let offset = evaluator.i32(268);
    let product_scale = evaluator.sub(exponent_sum, offset);
    let nonzero = evaluator.sub(one, columns.product_is_zero);
    let product_scale = evaluator.mul(nonzero, product_scale);
    let zero_scale = evaluator.i32(149);
    let zero_scale = evaluator.mul(columns.product_is_zero, zero_scale);
    let product_scale = evaluator.sub(product_scale, zero_scale);
    evaluator.constraint_eq(columns.binary_scale, product_scale);
    let inverse_product = evaluator.mul(columns.product, columns.product_inverse);
    evaluator.constraint_eq(inverse_product, nonzero);
    let zero_constraint = evaluator.mul(columns.product, columns.product_is_zero);
    evaluator.constraint(zero_constraint);
    let padding = evaluator.sub(one, columns.is_live);
    for (column, value) in [
        (columns.alpha_exponent, 127),
        (columns.alpha_fraction, 0),
        (columns.scale_exponent, 0),
        (columns.scale_fraction, 0),
        (columns.scale_exponent_is_zero, 1),
    ] {
        let expected = evaluator.i32(value);
        evaluator.constraint_eq_if(padding, column, expected);
    }
    evaluate_rounding(evaluator, &columns.rounding, columns.product, zero, zero);
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for CoefficientStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_COEFFICIENT_COLUMNS, 0>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;
    type EvaluationFrameTarget = StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_COEFFICIENT_COLUMNS, 0>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        frame: &Self::EvaluationFrame<FE, P, D2>,
        consumer: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let row: &[P; NUM_COEFFICIENT_COLUMNS] = frame.get_local_values().try_into().unwrap();
        evaluate(row.borrow(), &mut NativeEvaluator::new(consumer));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        frame: &Self::EvaluationFrameTarget,
        consumer: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let row: &[ExtensionTarget<D>; NUM_COEFFICIENT_COLUMNS] = frame.get_local_values().try_into().unwrap();
        evaluate(row.borrow(), &mut SymbolicEvaluator::new(builder, consumer));
    }

    fn constraint_degree(&self) -> usize {
        3
    }
    fn requires_ctls(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4::api::dtype::bf16_to_f32;
    use crate::v4::circuit::luts::{LutChecker, LutTable};
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};
    use starky::util::trace_rows_to_poly_values;

    type F = GoldilocksField;
    type Certificate = CoefficientStark<F, 2>;

    fn accepts(stark: &Certificate, rows: &[[F; NUM_COEFFICIENT_COLUMNS]]) -> bool {
        for row in rows {
            let frame = StarkFrame::from_values(row, row, &[]);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(65537)],
                F::ONE,
                F::ONE,
                F::ONE,
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            if consumer.accumulators().into_iter().any(|value| value != F::ZERO) {
                return false;
            }
        }
        let tables = [LutTable::Range16, LutTable::Pair128, LutTable::ExpInfo, LutTable::Fp32Tools];
        LutChecker::<F>::new(&tables)
            .check_trace(
                &coefficient_lookups(),
                &trace_rows_to_poly_values(rows.to_vec()),
                &[],
                "coefficient",
            )
            .is_ok()
    }

    #[test]
    fn coefficients_match_binary32_including_zero_subnormal_and_carry() {
        let mut inputs = vec![
            [0x3f80, 0],
            [0x3f80, 0x8000],
            [0x3780, 1],
            [0x3700, 1],
            [0x3700, 3],
            [0x3fff, 0x3fff],
        ];
        let mut random = 0x3db91f0325u64;
        for _ in 0..2048 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let alpha = ((random % 254 + 1) * 128 + (random >> 16 & 127)) as u16;
            let scale = (((random >> 24) % 255) * 128 + (random >> 40 & 127)) as u16;
            if (bf16_to_f32(alpha) * bf16_to_f32(scale)).is_finite() {
                inputs.push([alpha, scale]);
            }
        }
        let stark = Certificate::new(inputs.len(), 0, 8);
        let rows = stark.generate_trace(&inputs).unwrap();
        assert!(accepts(&stark, &rows));
        for (row, &[alpha, scale]) in rows.iter().zip(&inputs) {
            let columns: &CoefficientColumns<F> = row.borrow();
            assert_eq!(
                columns.rounding.output.word(),
                (bf16_to_f32(alpha) * bf16_to_f32(scale)).to_bits() & 0x7fffffff
            );
        }
        let known = stark.known_values();
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row[0], known[0].values[i]);
            assert_eq!(row[1], known[1].values[i]);
        }
    }

    #[test]
    fn malformed_rounding_fields_and_overflow_are_rejected() {
        let stark = Certificate::new(1, 0, 8);
        let rows = stark.generate_trace(&[[0x3fff, 0x3fff]]).unwrap();
        assert!(accepts(&stark, &rows));
        for column in COEFFICIENT_COLUMNS.product..NUM_COEFFICIENT_COLUMNS {
            let mut changed = rows.clone();
            changed[0][column] += F::ONE;
            assert!(!accepts(&stark, &changed), "accepted changed column {column}");
        }
        assert!(stark.generate_trace(&[[0x7f7f, 0x7f7f]]).is_err());
        for scale in [0x7f80, 0x7fc0, 0xff80] {
            assert!(stark.generate_trace(&[[0x3f80, scale]]).is_err());
        }
    }

    #[test]
    fn native_and_recursive_evaluators_agree_at_degree_three() {
        test_stark_low_degree::<F, Certificate, 2>(Certificate::new(1, 0, 8)).unwrap();
        test_stark_circuit_constraints::<F, PoseidonGoldilocksConfig, Certificate, 2>(Certificate::new(1, 0, 8)).unwrap();
    }
}
