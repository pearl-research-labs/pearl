//! Prover coset evaluation and interpolation using the parallel FFT.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use plonky2_field::fft::fft_root_table;
use plonky2_field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;
use plonky2_util::log2_strict;

use super::{fft_par_padded, MIN_LG};

/// Evaluates a power-of-two coefficient prefix on a domain `2^rate_bits` times
/// larger. The omitted coefficients are zero; output roots and order match FFT.
pub(crate) fn coset_lde_par<F: Field + Send + Sync>(
    coeffs: &[F],
    shift: F,
    rate_bits: usize,
) -> PolynomialValues<F> {
    let n = coeffs.len() << rate_bits;
    if rate_bits >= 4 && coeffs.len() <= 1 << 13 && n >= 1 << MIN_LG {
        // Use fewer, longer transforms for small polynomials at high rates.
        let coset_bits = if rate_bits >= 8 { 5 } else { rate_bits };
        let padding_bits = rate_bits - coset_bits;
        let count = 1 << coset_bits;
        let roots = fft_root_table::<F>(coeffs.len() << padding_bits);
        let omega = F::primitive_root_of_unity(log2_strict(n));
        let shifts: Vec<_> = omega.powers().take(count).map(|w| shift * w).collect();
        let fft_len = coeffs.len() << padding_bits;
        let cosets: Vec<_> = shifts
            .par_iter()
            .map(|&shift| {
                let mut coeffs: Vec<_> = shift.powers().zip(coeffs).map(|(w, &x)| w * x).collect();
                coeffs.resize(fft_len, F::ZERO);
                PolynomialCoeffs::new(coeffs)
                    .fft_with_options(Some(padding_bits), Some(&roots))
                    .values
            })
            .collect();
        let mut values = vec![F::ZERO; fft_len * shifts.len()];
        values
            .par_chunks_mut(shifts.len())
            .enumerate()
            .for_each(|(k, row)| {
                for (j, value) in row.iter_mut().enumerate() {
                    *value = cosets[j][k];
                }
            });
        return PolynomialValues::new(values);
    }
    // Scale coeffs[i] *= shift^i in parallel, then transform.
    let mut scaled = vec![F::ZERO; coeffs.len()];
    scaled
        .par_chunks_mut(1 << 12)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let mut w = shift.exp_u64((ci << 12) as u64);
            for (dst, &src) in chunk.iter_mut().zip(&coeffs[ci << 12..]) {
                *dst = src * w;
                w *= shift;
            }
        });
    fft_par_padded(scaled, n)
}

/// Interpolates coset evaluations in natural order, using the existing parallel FFT
/// for large inputs. Inverse normalization and coset scaling are fused in one pass.
pub(crate) fn coset_ifft_par<F: Field>(
    values: PolynomialValues<F>,
    shift: F,
) -> PolynomialCoeffs<F> {
    let n = values.len();
    if n < (1 << 18) {
        return values.coset_ifft(shift);
    }
    let norm = F::inverse_2exp(crate::util::log2_strict(n));
    let inv_shift = shift.inverse();
    let transformed = crate::util::fft_par::fft_par(PolynomialCoeffs::new(values.values));
    let mut coeffs = vec![F::ZERO; n];
    coeffs
        .par_chunks_mut(4096)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let start = ci * 4096;
            let mut w = norm * inv_shift.exp_u64(start as u64);
            for (offset, value) in chunk.iter_mut().enumerate() {
                let index = start + offset;
                *value = transformed.values[if index == 0 { 0 } else { n - index }] * w;
                w *= inv_shift;
            }
        });
    PolynomialCoeffs::new(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::extension::{Extendable, FieldExtension};
    use crate::field::goldilocks_field::GoldilocksField as F;

    fn check_padded_prefixes<FE: Field>(make: impl Fn(usize) -> FE, shift: FE) {
        for (bits, live) in [
            (17, 512),
            (18, 0),
            (18, 1),
            (18, 1023),
            (18, 8192),
            (18, 16385),
            (18, 65537),
            (20, 131072),
            (21, 8192),
        ] {
            let poly = PolynomialCoeffs::new(
                (0..1 << bits)
                    .map(|i| if i < live { make(i) } else { FE::ZERO })
                    .collect(),
            );
            let degree = live.max(1).next_power_of_two();
            assert_eq!(
                coset_lde_par(&poly.coeffs[..degree], shift, bits - log2_strict(degree)),
                poly.coset_fft(shift),
            );
        }
    }

    #[test]
    fn padded_prefixes_match_serial() {
        let word = |i: usize| {
            F::from_noncanonical_u64(
                [
                    0,
                    1,
                    0xffff_ffff_0000_0000,
                    0xffff_ffff_0000_0001,
                    0xffff_ffff_0000_0002,
                    u64::MAX,
                ][i % 6]
                    .wrapping_add((i as u64 / 6).wrapping_mul(977)),
            )
        };
        crate::util::in_thread_pools(|_| {
            check_padded_prefixes(word, F::coset_shift());
            type E = <F as Extendable<2>>::Extension;
            check_padded_prefixes(
                |i| E::from_basefield_array([word(i), word(i + 1)]),
                F::coset_shift().into(),
            );
            for shift in [F::ZERO, F::ONE] {
                let coeffs: Vec<_> = (0..1024).map(word).collect();
                assert_eq!(
                    coset_lde_par(&coeffs, shift, 8),
                    PolynomialCoeffs::new(coeffs).lde(8).coset_fft(shift)
                );
            }
        });
    }
    #[test]
    fn parallel_coset_ifft_matches_serial() {
        crate::util::in_thread_pools(|_| {
            for bits in [0, 1, 10, 18, 19] {
                let values = PolynomialValues::new(
                    (0..1 << bits)
                        .map(|i| {
                            F::from_noncanonical_u64((i as u64).wrapping_mul(0x9e3779b97f4a7c15))
                        })
                        .collect(),
                );
                for shift in [F::ONE, F::coset_shift()] {
                    assert_eq!(
                        coset_ifft_par(values.clone(), shift),
                        values.clone().coset_ifft(shift)
                    );
                }
            }
        });
    }
}
