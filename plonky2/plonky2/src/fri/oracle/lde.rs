//! Prover-only LDE evaluation, blinding and Merkle-leaf preparation.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::field::fft::FftRootTable;
use crate::field::polynomial::PolynomialCoeffs;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use plonky2_maybe_rayon::*;

use super::SALT_SIZE;

/// Evaluates equal-degree columns on the LDE coset, then appends random salt columns.
pub(super) fn values<F: RichField>(
    polynomials: &[PolynomialCoeffs<F>],
    rate_bits: usize,
    blinding: bool,
    fft_root_table: Option<&FftRootTable<F>>,
) -> Vec<Vec<F>> {
    let degree = polynomials[0].len();

    let parallel = polynomials.len() * 2 < rayon::current_num_threads()
        || (rate_bits >= 8 && degree <= 1 << 13);

    // Append independent salt columns when hiding.
    let salt_size = if blinding { SALT_SIZE } else { 0 };

    polynomials
        .par_iter()
        .map(|p| {
            assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
            if p.coeffs[1..].iter().all(|x| x.is_zero()) {
                return vec![p.coeffs[0]; degree << rate_bits];
            }

            if parallel && rate_bits >= 2 && degree << rate_bits >= 1 << 18 {
                crate::util::fft_par::coset_lde_par(&p.coeffs, F::coset_shift(), rate_bits).values
            } else {
                // Shift only the original coefficients; padding coefficients are zero.
                let shifted = PolynomialCoeffs::new(
                    F::coset_shift()
                        .powers()
                        .zip(&p.coeffs)
                        .map(|(r, &c)| r * c)
                        .collect(),
                );
                shifted
                    .lde(rate_bits)
                    .fft_with_options(Some(rate_bits), fft_root_table)
                    .values
            }
        })
        .chain((0..salt_size).into_par_iter().map(|_| {
            // Split long salt columns so OS sampling can use the whole pool.
            let mut salt = vec![F::ZERO; degree << rate_bits];
            salt.par_chunks_mut(1 << 16)
                .for_each(|chunk| chunk.copy_from_slice(&F::rand_vec(chunk.len())));
            salt
        }))
        .collect()
}

/// Zero-pads coefficients in parallel when the padded buffer is large.
pub(super) fn pad<F: Field>(poly: PolynomialCoeffs<F>, rate_bits: usize) -> PolynomialCoeffs<F> {
    let padded_len = poly.len() << rate_bits;
    if rate_bits < 4 || padded_len < 1 << 18 {
        return poly.lde(rate_bits);
    }
    PolynomialCoeffs::new(
        (0..padded_len)
            .into_par_iter()
            .with_min_len(4096)
            .map(|i| poly.coeffs.get(i).copied().unwrap_or(F::ZERO))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;

    #[test]
    fn small_coset_ldes_match_original_fft() {
        type F = GoldilocksField;
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for degree_bits in [0, 3, 6, 10] {
                    let p = PolynomialCoeffs::new(
                        (0..1 << degree_bits)
                            .map(|i| {
                                F::from_noncanonical_u64(
                                    (i as u64).wrapping_mul(0x9e3779b97f4a7c15),
                                )
                            })
                            .collect(),
                    );
                    for rate_bits in [0, 1, 3, 4, 8, 9] {
                        let roots = crate::field::fft::fft_root_table(p.len() << rate_bits);
                        let expected = p
                            .lde(rate_bits)
                            .coset_fft_with_options(F::coset_shift(), Some(rate_bits), Some(&roots))
                            .values;
                        for num_polynomials in [1, threads + 1] {
                            let mut polynomials = vec![p.clone(); num_polynomials];
                            let mut expected_batch = vec![expected.clone(); num_polynomials];
                            if num_polynomials > 1 {
                                for constant in [F::ZERO, F::from_canonical_u64(7)] {
                                    let mut coeffs = vec![F::ZERO; p.len()];
                                    coeffs[0] = constant;
                                    polynomials.push(PolynomialCoeffs::new(coeffs));
                                    expected_batch.push(vec![constant; expected.len()]);
                                }
                            }
                            for cached_roots in [None, Some(&roots)] {
                                assert_eq!(
                                    values(&polynomials, rate_bits, false, cached_roots),
                                    expected_batch
                                );
                            }
                            if degree_bits == 10 && rate_bits == 8 && num_polynomials == 1 {
                                let blinded = values(&polynomials, rate_bits, true, None);
                                assert_eq!(&blinded[..num_polynomials], &expected_batch);
                                assert_eq!(blinded.len(), num_polynomials + SALT_SIZE);
                                assert!(blinded
                                    .iter()
                                    .all(|column| column.len() == expected.len()));
                            }
                        }
                    }
                }
            });
        }
    }
}
