//! Prover-only interpolation, opening evaluation and CTL preparation.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use hashbrown::HashMap;
use plonky2::batch_fri::oracle::BatchFriOracle;
use plonky2::field::extension::{Extendable, FieldExtension, OEF};
use plonky2::field::fft::{fft_root_table, ifft_with_options};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::config::GenericConfig;
use plonky2_maybe_rayon::*;

use crate::cross_table_lookup::CtlData;
use crate::lookup::Column;

/// Interpolates columns, sharing roots by height and skipping constant transforms.
pub(super) fn ifft_columns<F: RichField>(
    values: Vec<PolynomialValues<F>>,
) -> Vec<PolynomialCoeffs<F>> {
    let mut roots = HashMap::new();
    for column in &values {
        roots
            .entry(column.len())
            .or_insert_with(|| fft_root_table(column.len()));
    }
    values
        .into_par_iter()
        .map(|column| {
            if column.values[1..].iter().all(|&v| v == column.values[0]) {
                let mut coeffs = vec![F::ZERO; column.len()];
                coeffs[0] = column.values[0];
                return PolynomialCoeffs::new(coeffs);
            }
            let table = &roots[&column.len()];
            ifft_with_options(column, None, Some(table))
        })
        .collect()
}

/// Fetches `buffer.len()` columns at `index * step`, starting at `offset`.
/// Scalar rows are borrowed; packed rows use the caller's scratch buffer.
pub(super) fn lde_row<'a, F, P, C, const D: usize>(
    (oracle, degree_index, offset): (&'a BatchFriOracle<F, C, D>, usize, usize),
    index: usize,
    step: usize,
    buffer: &'a mut [P],
) -> &'a [P]
where
    F: RichField + Extendable<D>,
    P: PackedField<Scalar = F>,
    C: GenericConfig<D, F = F>,
{
    if P::WIDTH == 1 {
        P::pack_slice(oracle.get_lde_values(degree_index, index, step, offset, buffer.len()))
    } else {
        oracle.get_lde_values_packed_into(degree_index, index, step, offset, buffer);
        buffer
    }
}

/// Replaces each CTL tuple with its challenge-weighted affine column.
pub(super) fn combine_ctl_columns<F: Field>(
    data: &CtlData<F>,
    public_inputs: &[F],
    num_columns: usize,
) -> Vec<Vec<[Column<F>; 1]>> {
    data.zs_columns
        .iter()
        .map(|zs| {
            zs.columns
                .iter()
                .map(|columns| {
                    [precombine_ctl_columns(
                        columns,
                        zs.challenge.beta,
                        public_inputs,
                        num_columns,
                    )]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

/// Builds the single affine column `sum_i beta^i * columns[i]`, binding public
/// inputs to their values. The ordinary CTL combiner still adds gamma once.
fn precombine_ctl_columns<F: Field>(
    columns: &[Column<F>],
    beta: F,
    public_inputs: &[F],
    num_columns: usize,
) -> Column<F> {
    let mut local = vec![F::ZERO; num_columns];
    let mut next = vec![F::ZERO; num_columns];
    let mut constant = F::ZERO;
    for (column, weight) in columns.iter().zip(beta.powers()) {
        let (linear, next_row, scaled, offsets, c) = column.parts();
        for &(i, factor) in linear {
            local[i] += weight * factor;
        }
        for &(i, factor) in next_row {
            next[i] += weight * factor;
        }
        for &(i, slot) in scaled {
            local[i] += weight * public_inputs[slot];
        }
        constant += weight * (c + offsets.iter().map(|&slot| public_inputs[slot]).sum::<F>());
    }
    let [local, next] = [local, next].map(|row| {
        row.into_iter()
            .enumerate()
            .filter(|(_, value)| value.is_nonzero())
            .collect::<Vec<_>>()
    });
    if local.is_empty() && next.is_empty() {
        Column::constant(constant)
    } else {
        Column::linear_combination_and_next_row_with_constant(local, next, constant)
    }
}

/// Evaluates ascending base-field coefficients at `point`. Quadratic extensions
/// use eight independent chains and their trace/norm relation, requiring only
/// base-field multiplications within each chain. Other extensions use Horner.
pub(super) fn evaluate_opening<F: RichField + Extendable<D>, const D: usize>(
    coefficients: &[F],
    point: F::Extension,
) -> F::Extension {
    let step = point * point;
    let step = step * step;
    if D == 2 {
        let step = step * step;
        // Y² = trace(Y)Y - norm(Y) for Y in a quadratic extension.
        let coordinates = step.to_basefield_array();
        let trace = coordinates[0] + coordinates[0];
        let norm = coordinates[0] * coordinates[0]
            - coordinates[1] * coordinates[1] * <F::Extension as OEF<D>>::W;
        let mut linear = [F::ZERO; 8];
        let mut constant = [F::ZERO; 8];
        for chunk in coefficients.chunks(8).rev() {
            for (i, &coefficient) in chunk.iter().enumerate() {
                let previous = linear[i];
                linear[i] = previous * trace + constant[i];
                constant[i] = coefficient - previous * norm;
            }
        }
        return (0..8).rev().fold(F::Extension::ZERO, |acc, i| {
            acc * point + step.scalar_mul(linear[i]) + F::Extension::from_basefield(constant[i])
        });
    }
    coefficients
        .iter()
        .rev()
        .fold(F::Extension::ZERO, |acc, &c| {
            acc * point + F::Extension::from_basefield(c)
        })
}

#[cfg(test)]
mod opening_evaluation_tests {
    use super::*;
    use plonky2::field::goldilocks_field::GoldilocksField as F;
    use plonky2::field::types::Sample;

    #[test]
    fn openings_match_horner() {
        type E = <F as Extendable<2>>::Extension;
        let extension_point =
            E::from_basefield_array([F::from_canonical_u64(2), F::from_canonical_u64(3)]);
        for length in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 17, 31, 128, 1025] {
            let coefficients: Vec<_> = (0..length)
                .map(|i| F::from_canonical_usize(19 * i + 7))
                .collect();
            for point in [E::ZERO, E::ONE, extension_point] {
                let expected = coefficients
                    .iter()
                    .rev()
                    .fold(E::ZERO, |acc, &c| acc * point + E::from(c));
                assert_eq!(evaluate_opening::<F, 2>(&coefficients, point), expected);
            }
        }
    }

    #[test]
    fn precombined_ctl_matches_original_expression() {
        let columns = vec![
            Column::constant(F::ONE),
            Column::single(0),
            Column::linear_combination_and_next_row_with_constant(
                vec![(0, F::TWO), (1, -F::ONE)],
                vec![(2, F::ONE)],
                F::TWO,
            ),
            Column::single_times_public_input_plus_single(1, 0, 0),
            Column::single_times_public_input_plus_public_input(2, 1, 0),
            Column::public_input_minus_single(1, 2),
        ];
        for beta in [F::ZERO, F::ONE, -F::ONE, F::from_canonical_u64(19)] {
            for public in [vec![F::ZERO; 2], F::rand_vec(2)] {
                let local = F::rand_vec(3);
                let next = F::rand_vec(3);
                for tuple in [columns.as_slice(), &columns[..0], &columns[1..3]] {
                    let combined = precombine_ctl_columns(tuple, beta, &public, 3);
                    let challenge = crate::lookup::GrandProductChallenge {
                        beta,
                        gamma: F::TWO,
                    };
                    let expected = challenge.combine_base(
                        tuple
                            .iter()
                            .map(|c| c.eval_with_next::<F, F, 1>(&local, &next, &public)),
                    );
                    let actual = challenge
                        .combine_base([combined.eval_with_next::<F, F, 1>(&local, &next, &public)]);
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
