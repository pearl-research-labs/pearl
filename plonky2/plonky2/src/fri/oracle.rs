#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo, FriPolynomialInfo, ZETA_BATCH_IDX};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::fft_par::coset_fft_par;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place, transpose};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();
        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let leaves = timed!(timing, "transpose LDEs", {
            if lde_values.len() <= 32 {
                crate::util::transpose_bitrev(&lde_values)
            } else {
                let mut leaves = transpose(&lde_values);
                reverse_leaf_indices(&mut leaves);
                leaves
            }
        });
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new(leaves, cap_height)
        );
        // Frees lde_values in parallel to the main job
        rayon::spawn(move || {
            drop(lde_values);
        });

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    pub(crate) fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();

        // Small source polynomials avoid a large padded FFT even in wide batches.
        // Narrow batches also use cosets to expose more parallel work.
        let parallel_cosets = rate_bits >= 4
            && (polynomials.len() * 2 < rayon::current_num_threads()
                || (rate_bits >= 8 && degree <= 1 << 13));
        // Fewer, longer cosets amortize task and interleaving costs for small inputs.
        let coset_bits = if rate_bits >= 8 && degree <= 1 << 13 {
            5
        } else {
            rate_bits
        };
        let padding_bits = rate_bits - coset_bits;
        let small_roots =
            parallel_cosets.then(|| crate::field::fft::fft_root_table(degree << padding_bits));
        let shifts: Vec<_> = if parallel_cosets {
            F::primitive_root_of_unity(log2_strict(degree) + rate_bits)
                .powers()
                .take(1 << coset_bits)
                .map(|w| F::coset_shift() * w)
                .collect()
        } else {
            Vec::new()
        };

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                if p.coeffs[1..].iter().all(|x| x.is_zero()) {
                    return vec![p.coeffs[0]; degree << rate_bits];
                }

                if let Some(roots) = &small_roots {
                    Self::lde_on_cosets(p, &shifts, roots, padding_bits)
                } else if rate_bits >= 2
                    && degree << rate_bits >= 1 << 18
                    && polynomials.len() * 2 < rayon::current_num_threads()
                {
                    crate::util::fft_par::coset_lde_par(&p.coeffs, F::coset_shift(), rate_bits)
                        .values
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

    /// Evaluates `p` on `shifts`, padding each FFT by `2^padding_bits`.
    /// `roots` must match that padded length.
    /// Returns natural LDE order: index `j + shifts.len() * k` selects coset `j`,
    /// point `k`, where `shifts[j] = coset_shift * full_domain_root^j`.
    fn lde_on_cosets(
        p: &PolynomialCoeffs<F>,
        shifts: &[F],
        roots: &FftRootTable<F>,
        padding_bits: usize,
    ) -> Vec<F> {
        let fft_len = p.len() << padding_bits;
        let cosets: Vec<_> = shifts
            .par_iter()
            .map(|&shift| {
                let mut coeffs: Vec<_> =
                    shift.powers().zip(&p.coeffs).map(|(w, &x)| w * x).collect();
                coeffs.resize(fft_len, F::ZERO);
                PolynomialCoeffs::new(coeffs)
                    .fft_with_options(Some(padding_bits), Some(roots))
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
        values
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.leaves[index];
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        assert_eq!(
            oracles.len(),
            instance.oracles.len(),
            "Number of oracles mismatch"
        );
        let batching_pow_witness =
            crate::fri::batching_pow::prove::<F, C, D>(challenger, &fri_params.config);
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        //
        // If we are in the zk case, the `R` polynomial (the last polynomials in batch `ZETA_BATCH_IDX`) is added to
        // the batch polynomial independently, without being quotiented. So the final polynomial becomes:
        // `final_poly = R(X) + sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`.
        // Then, since the degree of `R` is double that of the batch polynomial in our implementation, we need to
        // compute one extra step in FRI to reach the correct degree.

        let is_zk = fri_params.hiding;

        for (idx, FriBatchInfo { point, polynomials }) in instance.batches.iter().enumerate() {
            // Add one random polynomial to FRI's input polynomial. Assign it to batch whose index is 0.
            let has_r_poly = is_zk && (idx == ZETA_BATCH_IDX);
            let last_poly = polynomials.len() - has_r_poly as usize;
            // Collect the coefficients of all the polynomials in `polynomials` until `last_poly`.
            let polys_coef = |fri_poly: &FriPolynomialInfo| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            };
            let polys_refs: Vec<_> = polynomials[..last_poly].iter().map(polys_coef).collect();
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(&polys_refs)
            );
            let mut quotient = composition_poly.divide_by_linear(*point);
            quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;

            // If we are in the zk case, we add `R(X)` to the batch after multiplying by `alpha`.
            if has_r_poly {
                let blinding_polys_refs: Vec<_> =
                    polynomials[last_poly..].iter().map(polys_coef).collect();
                let blinding_poly = alpha.reduce_polys_base(&blinding_polys_refs);
                alpha.shift_poly(&mut final_poly);
                final_poly += blinding_poly;
            }
        }

        let (lde_final_poly, lde_final_values) = {
            let rate_bits = fri_params.config.rate_bits;
            let padded_len = final_poly.len() << rate_bits;
            let lde_final_poly = if rate_bits >= 4 && padded_len >= 1 << 18 {
                PolynomialCoeffs::new(
                    (0..padded_len)
                        .into_par_iter()
                        .with_min_len(4096)
                        .map(|i| {
                            final_poly
                                .coeffs
                                .get(i)
                                .copied()
                                .unwrap_or(F::Extension::ZERO)
                        })
                        .collect(),
                )
            } else {
                final_poly.lde(rate_bits)
            };
            let lde_final_values = timed!(
                timing,
                &format!("perform final FFT {}", lde_final_poly.len()),
                coset_fft_par(&lde_final_poly, F::coset_shift().into())
            );
            (lde_final_poly, lde_final_values)
        };

        {
            fri_proof::<F, C, D>(
                &oracles
                    .par_iter()
                    .map(|c| &c.merkle_tree)
                    .collect::<Vec<_>>(),
                lde_final_poly,
                lde_final_values,
                challenger,
                fri_params,
                batching_pow_witness,
                final_poly_coeff_len,
                max_num_query_steps,
                timing,
            )
        }
    }
}

/// Reorders leaf headers in place; leaf payloads are never copied.
fn reverse_leaf_indices<T: Send>(rows: &mut [T]) {
    if rows.len() < 1 << 18 {
        reverse_index_bits_in_place(rows);
        return;
    }
    let bits = log2_strict(rows.len());
    struct RowPtr<T>(*mut T);
    // SAFETY: the mutable slice is inaccessible during the join. Workers
    // transfer only disjoint pairs, and T: Send permits cross-thread moves.
    unsafe impl<T: Send> Sync for RowPtr<T> {}
    impl<T> RowPtr<T> {
        unsafe fn swap(&self, i: usize, j: usize) {
            unsafe {
                core::ptr::swap(self.0.add(i), self.0.add(j));
            }
        }
    }
    let pointer = RowPtr(rows.as_mut_ptr());
    (0..rows.len())
        .into_par_iter()
        .with_min_len(4096)
        .for_each(|i| {
            let j = reverse_bits(i, bits);
            if i < j {
                // Bit reversal is an involution: each non-fixed index belongs to
                // exactly one pair, and only its smaller member performs the swap.
                unsafe {
                    pointer.swap(i, j);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::plonk::config::PoseidonGoldilocksConfig;

    #[test]
    fn parallel_leaf_reversal_matches_serial() {
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for bits in [0, 1, 7, 18, 19] {
                let mut rows: Vec<_> = (0..1usize << bits).map(|i| vec![i, i ^ 0x5a5a]).collect();
                let mut expected = rows.clone();
                reverse_index_bits_in_place(&mut expected);
                pool.install(|| reverse_leaf_indices(&mut rows));
                assert_eq!(rows, expected);
            }
        }
    }

    #[test]
    fn small_coset_ldes_match_original_fft() {
        type F = GoldilocksField;
        type C = PoseidonGoldilocksConfig;
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
                                    PolynomialBatch::<F, C, 2>::lde_values(
                                        &polynomials,
                                        rate_bits,
                                        false,
                                        cached_roots
                                    ),
                                    expected_batch
                                );
                            }
                            if degree_bits == 10 && rate_bits == 8 && num_polynomials == 1 {
                                let blinded = PolynomialBatch::<F, C, 2>::lde_values(
                                    &polynomials,
                                    rate_bits,
                                    true,
                                    None,
                                );
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
