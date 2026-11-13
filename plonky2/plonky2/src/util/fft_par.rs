//! Parallel FFT for large transforms, via the classic four-step
//! (Cooley-Tukey) decomposition.
//!
//! A serial radix-2 FFT over $N = 2^{23}$ extension-field elements takes
//! hundreds of milliseconds and no outer loop can subdivide it, so the
//! remaining serial FFTs inside otherwise-parallel phases dominate their
//! wall time. The four-step split factors the transform into
//! $N = N_1 \cdot N_2$ independent sub-FFTs of lengths $N_1$ and $N_2$
//! which run through the shared rayon pool.
//!
//! With input indexed $n = N_2 n_1 + n_2$ and output $k = k_1 + N_1 k_2$:
//!
//! $$X[k_1 + N_1 k_2] = \sum_{n_2} \omega_N^{n_2 k_1}\, \omega_{N_2}^{n_2 k_2}
//!   \left[\sum_{n_1} x[N_2 n_1 + n_2]\, \omega_{N_1}^{n_1 k_1}\right]$$
//!
//! i.e. (1) one length-$N_1$ FFT per stride-$N_2$ residue class $n_2$,
//! (2) a twiddle pass, (3) one length-$N_2$ FFT per output column $k_1$,
//! (4) a final transpose into natural output order.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_field::fft::{fft_root_table, fft_with_options};
use plonky2_field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;
use plonky2_util::log2_strict;

/// Transforms below this size run faster serially; the four-step's extra
/// passes and task overheads only pay off on the big buffers.
const MIN_LG: usize = 18;

/// Parallel FFT of `coeffs` (natural-order in and out), returned as values.
///
/// Below the size threshold this is exactly `fft`; above it the transform
/// decomposes into $\sqrt{N}$-sized sub-FFTs that share the rayon pool.
pub fn fft_par<F: Field + Send + Sync>(coeffs: PolynomialCoeffs<F>) -> PolynomialValues<F> {
    let n = coeffs.len();
    fft_par_padded(coeffs.coeffs, n)
}

/// The omitted suffix is zero; both the prefix and output lengths are powers of two.
fn fft_par_padded<F: Field + Send + Sync>(mut coeffs: Vec<F>, n: usize) -> PolynomialValues<F> {
    if n < (1 << MIN_LG) {
        let zero_factor = log2_strict(n) - log2_strict(coeffs.len());
        coeffs.resize(n, F::ZERO);
        return fft_with_options(PolynomialCoeffs::new(coeffs), Some(zero_factor), None);
    }

    let lg_n = log2_strict(n);
    // Balanced factorization: both sub-FFT lengths sit near sqrt(N).
    let lg_n1 = lg_n.div_ceil(2);
    let lg_n2 = lg_n - lg_n1;
    let n1 = 1usize << lg_n1;
    let n2 = 1usize << lg_n2;

    let omega = F::primitive_root_of_unity(lg_n);

    // Sub-FFT root tables, built once and shared across all row tasks.
    let table1 = fft_root_table::<F>(n1);
    let table2 = fft_root_table::<F>(n2);

    // Step 1: A[n2][k1] = DFT_{N1} over n1 of the stride-N2 residue class n2.
    // Gather each class first so the sub-FFTs run on contiguous rows.
    let a: Vec<Vec<F>> = (0..n2.min(coeffs.len()))
        .into_par_iter()
        .map(|n2_idx| {
            let mut class: Vec<F> = (n2_idx..coeffs.len())
                .step_by(n2)
                .map(|i| coeffs[i])
                .collect();
            let zero_factor = lg_n1 - log2_strict(class.len());
            class.resize(n1, F::ZERO);
            fft_with_options(
                PolynomialCoeffs::new(class),
                Some(zero_factor),
                Some(&table1),
            )
            .values
        })
        .collect();

    // Steps 2+3: B[k1][k2] = DFT_{N2} over n2 of
    //   A[n2][k1] * omega_N^{n2 k1},
    // the twiddle folded into the column gather.
    let zero_factor = lg_n2 - log2_strict(a.len());
    let b: Vec<Vec<F>> = (0..n1)
        .into_par_iter()
        .map(|k1| {
            let step = omega.exp_u64(k1 as u64); // omega_N^{k1}
            let mut w = F::ONE;
            let mut column: Vec<F> = a
                .iter()
                .map(|row| {
                    let v = row[k1] * w;
                    w *= step;
                    v
                })
                .collect();
            column.resize(n2, F::ZERO);
            fft_with_options(
                PolynomialCoeffs::new(column),
                Some(zero_factor),
                Some(&table2),
            )
            .values
        })
        .collect();

    // Step 4: transpose B into natural order, X[k1 + N1*k2] = B[k1][k2].
    // Consecutive outputs run over k1 first, so each chunk of N1 holds one k2.
    let mut out = vec![F::ZERO; n];
    out.par_chunks_mut(n1).enumerate().for_each(|(k2, chunk)| {
        for (k1, dst) in chunk.iter_mut().enumerate() {
            *dst = b[k1][k2];
        }
    });
    PolynomialValues::new(out)
}

/// Evaluates `coeffs` on `shift * H`, in natural order, with `|H| = coeffs.len()`.
pub fn coset_fft_par<F: Field + Send + Sync>(
    coeffs: &PolynomialCoeffs<F>,
    shift: F,
) -> PolynomialValues<F> {
    coset_lde_par(&coeffs.coeffs, shift, 0)
}

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
    use plonky2_field::extension::{Extendable, FieldExtension};
    use plonky2_field::fft::fft;
    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::polynomial::PolynomialCoeffs;
    use plonky2_field::types::Field;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn matches_serial_below_threshold() {
        let n = 1 << 10;
        let coeffs: Vec<F> = (0..n)
            .map(|i| F::from_canonical_usize(i * 977 % 101))
            .collect();
        let poly = PolynomialCoeffs::new(coeffs.clone());
        assert_eq!(fft_par(poly.clone()).values, fft(poly).values);
    }

    #[test]
    fn matches_serial_above_threshold() {
        // Cross the parallel threshold and exercise the unbalanced split
        // (lg 19 = 10 + 9).
        let n = 1 << 19;
        let coeffs: Vec<F> = (0..n)
            .map(|i| {
                F::from_canonical_u64((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % (1 << 32))
            })
            .collect();
        let poly = PolynomialCoeffs::new(coeffs);
        let par = fft_par(poly.clone()).values;
        let serial = fft(poly).values;
        assert_eq!(par, serial);
    }

    #[test]
    fn coset_matches_serial() {
        let n = 1 << 19;
        let coeffs: Vec<F> = (0..n)
            .map(|i| {
                F::from_canonical_u64((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) % (1 << 32))
            })
            .collect();
        let poly = PolynomialCoeffs::new(coeffs);
        let shift = F::coset_shift();
        let par = coset_fft_par(&poly, shift).values;
        let serial = poly.coset_fft(shift).values;
        assert_eq!(par, serial);
    }

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
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
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
    }
    #[test]
    fn parallel_coset_ifft_matches_serial() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    for bits in [0, 1, 10, 18, 19] {
                        let values = PolynomialValues::new(
                            (0..1 << bits)
                                .map(|i| {
                                    F::from_noncanonical_u64(
                                        (i as u64).wrapping_mul(0x9e3779b97f4a7c15),
                                    )
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
}
