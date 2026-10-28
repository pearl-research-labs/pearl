#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use core::ops::Range;

use itertools::Itertools;
use plonky2_field::extension::Extendable;
use plonky2_field::fft::FftRootTable;
use plonky2_field::packed::PackedField;
use plonky2_field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;
use plonky2_util::log2_strict;

use crate::batch_fri::batch_alpha_runs;
use crate::batch_fri::prover::batch_fri_proof_with_openings;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::proof::FriProof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::batch_merkle_tree::BatchMerkleTree;
use crate::hash::hash_types::RichField;
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::timed;
use crate::util::fft_par::coset_fft_par;
use crate::util::timing::TimingTree;
use crate::util::{reverse_bits, transpose_bitrev};
use plonky2_field::extension::FieldExtension;
use plonky2_field::packable::Packable;

/// Hash commitment leaf rows into Merkle leaves (as field-element vectors).
///
/// Delegates to the hasher's `hash_no_pad_batch`, which by default hashes
/// rows in parallel with the scalar `hash_no_pad`; hashers with SIMD batch
/// implementations may override it to amortize permutation work across
/// lanes.
fn hash_leaf_rows<F, C, const D: usize>(rows: &[Vec<F>]) -> Vec<Vec<F>>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    C::Hasher::hash_no_pad_batch(rows)
}

/// `(self - self(z)) / (X - z)` with a blocked parallel scan.
///
/// The classic Horner recurrence `q[i] = q[i+1] * z + c[i+1]` (with
/// `q[n-1] = c[n-1]`) is a linear scan over the whole coefficient array;
/// the deg-22 openings instances run it twice over 2^22 extension
/// coefficients on the critical path. Blocked scan (same shape as the CTL
/// Z-suffix scan): the scan flows from high indices down, and each block's
/// out-state is linear in its in-state (`out = in * z^len + L` where `L` is
/// the block's own Horner sum), so block-local sums evaluate in parallel,
/// a serial pass combines the `#blocks` carries, and the blocks then run
/// their recurrences in parallel.
///
/// Value-identical to `PolynomialCoeffs::divide_by_linear`: the same
/// `acc = acc * z + c` recurrence over the same coefficients, evaluated as
/// independent segments.
pub(crate) fn divide_by_linear_par<FE: Field>(
    coeffs: &[FE],
    z: FE,
) -> PolynomialCoeffs<FE> {
    let n = coeffs.len();
    if n == 0 {
        return PolynomialCoeffs::empty();
    }
    // Below the block size the serial Horner scan is optimal (and the
    // small-input path is exactly the original implementation).
    const BLOCK: usize = 1 << 14;
    if n <= BLOCK {
        let mut bs = coeffs
            .iter()
            .rev()
            .scan(FE::ZERO, |acc, &c| {
                *acc = *acc * z + c;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        bs.pop();
        bs.reverse();
        return PolynomialCoeffs::new(bs);
    }
    let num_blocks = n.div_ceil(BLOCK);
    let block_len = |b: usize| ((b + 1) * BLOCK).min(n) - b * BLOCK;

    // Per-block local Horner walk `L_b`: the state the block produces from
    // its own coefficients alone (entry state zero), processing descending
    // from the block's top index — the same direction as the serial scan.
    let locals: Vec<FE> = (0..num_blocks)
        .into_par_iter()
        .map(|b| {
            let lo = b * BLOCK;
            let hi = ((b + 1) * BLOCK).min(n);
            let mut s = FE::ZERO;
            for i in (lo..hi).rev() {
                s = s * z + coeffs[i];
            }
            s
        })
        .collect();

    // Serial carry pass down the blocks (top to bottom): with `len_b` the
    // block's length, `out_b = in_b * z^{len_b} + L_b`, and `out_b` feeds the
    // block below as its entry state. The top block's entry state is zero.
    let mut in_state = vec![FE::ZERO; num_blocks];
    let mut carry = FE::ZERO;
    for b in (0..num_blocks).rev() {
        in_state[b] = carry;
        carry = carry * z.exp_u64(block_len(b) as u64) + locals[b];
    }

    // Per-block recurrences in parallel: `q[hi-1] = in_b`, then
    // `q[i] = q[i+1] * z + c[i]` descending through the block (the original
    // scan's recurrence, restricted to the block).
    let mut q = vec![FE::ZERO; n - 1];
    q.par_chunks_mut(BLOCK)
        .enumerate()
        .for_each(|(b, out)| {
            let lo = b * BLOCK;
            let hi = ((b + 1) * BLOCK).min(n);
            let mut s = in_state[b];
            for i in (lo..hi).rev() {
                if i < n - 1 {
                    out[i - lo] = s;
                }
                if i > lo {
                    s = s * z + coeffs[i];
                }
            }
        });
    PolynomialCoeffs::new(q)
}

/// Represents a batch FRI oracle, i.e. a batch of polynomials with different degrees which have
/// been Merkle-ized in a [`BatchMerkleTree`].
#[derive(Eq, PartialEq, Debug)]
pub struct BatchFriOracle<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub batch_merkle_tree: BatchMerkleTree<F, C::Hasher>,
    // The degree bits of each polynomial group.
    pub degree_bits: Vec<usize>,
    pub rate_bits: usize,
    pub blinding: bool,
    // Digest trees retain raw rows for quotient evaluation and FRI openings.
    row_values: Vec<Vec<Vec<F>>>,
    leaf_ranges: Vec<Range<usize>>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    BatchFriOracle<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: &[Option<&FftRootTable<F>>],
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
    /// The polynomials must be sorted by degree, from largest to smallest.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: &[Option<&FftRootTable<F>>],
    ) -> Self {
        let mut degree_bits = polynomials
            .iter()
            .map(|p| log2_strict(p.len()))
            .collect_vec();
        assert!(degree_bits.windows(2).all(|pair| { pair[0] >= pair[1] }));

        // Partition into maximal same-degree groups (in order).
        let num_polynomials = polynomials.len();
        let mut groups: Vec<Range<usize>> = Vec::new();
        {
            let mut group_start = 0;
            for (i, d) in degree_bits.iter().enumerate() {
                if i == num_polynomials - 1 || *d > degree_bits[i + 1] {
                    groups.push(group_start..i + 1);
                    group_start = i + 1;
                }
            }
        }

        // The degree groups are independent: run their LDE + transpose
        // concurrently (nested parallel iterators cooperate through work
        // stealing), instead of one group at a time with the pool idling
        // whenever a group holds few polynomials. `leaves` keeps the
        // original tall-first group order.
        let leaves: Vec<Vec<Vec<F>>> = groups
            .par_iter()
            .map(|range| {
                let lde_values = PolynomialBatch::<F, C, D>::lde_values(
                    &polynomials[range.clone()],
                    rate_bits,
                    blinding,
                    fft_root_table[range.start],
                );
                transpose_bitrev(&lde_values)
            })
            .collect();

        let batch_merkle_tree = timed!(
            timing,
            "build batch Merkle tree",
            BatchMerkleTree::new(leaves, cap_height)
        );

        degree_bits.sort_unstable();
        degree_bits.dedup();
        degree_bits.reverse();
        assert_eq!(batch_merkle_tree.leaves.len(), degree_bits.len());
        Self {
            polynomials,
            batch_merkle_tree,
            degree_bits,
            rate_bits,
            blinding,
            row_values: Vec::new(),
            leaf_ranges: Vec::new(),
        }
    }

    /// Commits one digest matrix per table. `ranges` partition `polynomials` in
    /// table order; each range has a fixed width and a single polynomial degree.
    /// The tree sorts matrices by height, preserving table order at ties.
    pub fn from_coeffs_with_digests(
        polynomials: Vec<PolynomialCoeffs<F>>,
        mut ranges: Vec<Range<usize>>,
        rate_bits: usize,
        cap_height: usize,
        timing: &mut TimingTree,
    ) -> Self {
        let mut end = 0;
        for range in &ranges {
            assert_eq!(range.start, end);
            assert!(!range.is_empty());
            assert!(range.end <= polynomials.len());
            let degree = polynomials[range.start].len();
            assert!(polynomials[range.clone()].iter().all(|p| p.len() == degree));
            end = range.end;
        }
        assert_eq!(end, polynomials.len());
        ranges.sort_by_key(|r| (core::cmp::Reverse(polynomials[r.start].len()), r.start));
        let degree_bits: Vec<_> = ranges
            .iter()
            .map(|r| log2_strict(polynomials[r.start].len()))
            .collect();
        // The height groups are mutually independent, and the tall ones carry
        // only a handful of polynomials — running the groups one at a time
        // leaves most cores idle while the tallest group's few serial FFTs
        // dominate the phase. Fusing each group's LDE + transpose + leaf-hash
        // into one job lets the pool interleave them: while one worker
        // hashes group A's rows (memory-bound, wide), another streams group
        // B's LDE (compute-bound); the hash — the phase's long pole — runs
        // concurrently with the other groups' setup instead of strictly
        // after it. `row_values` keeps the raw rows for openings and the
        // digest matrices go to the tree; both in `ranges` order.
        let (row_values, digest_leaves): (Vec<_>, Vec<_>) = ranges
            .par_iter()
            .map(|range| {
                let lde = PolynomialBatch::<F, C, D>::lde_values(
                    &polynomials[range.clone()],
                    rate_bits,
                    false,
                    None,
                );
                let rows = transpose_bitrev(&lde);
                let digests = hash_leaf_rows::<F, C, D>(&rows);
                (rows, digests)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();
        let batch_merkle_tree = timed!(
            timing,
            "build digest Merkle tree",
            BatchMerkleTree::new(digest_leaves, cap_height)
        );
        Self {
            polynomials,
            batch_merkle_tree,
            degree_bits,
            rate_bits,
            blinding: false,
            row_values,
            leaf_ranges: ranges,
        }
    }

    /// Flat opened evaluations in polynomial order, even when tree order differs.
    pub fn open_values(&self, index: usize) -> Vec<F> {
        if self.row_values.is_empty() {
            return self.batch_merkle_tree.values(index).concat();
        }
        let mut values = vec![F::ZERO; self.polynomials.len()];
        for (i, range) in self.leaf_ranges.iter().enumerate() {
            let row = index >> (self.degree_bits[0] - self.degree_bits[i]);
            values[range.clone()].copy_from_slice(&self.row_values[i][row]);
        }
        values
    }

    /// Tree group containing the polynomial at flat index `start`.
    pub fn digest_leaf_index(&self, start: usize) -> usize {
        self.leaf_ranges
            .iter()
            .position(|range| range.start == start)
            .expect("a table starts at a digest leaf range")
    }

    /// Produces a batch opening proof.
    ///
    /// `degree_bits` contains the degree (in bits) of each FRI instance, sorted in descending
    /// order with no duplicates. Instance `i` groups all polynomials of degree `degree_bits[i]`.
    pub fn prove_openings(
        degree_bits: &[usize],
        instances: &[FriInstanceInfo<F, D>],
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D>
    where
        F: Packable,
        <F as Packable>::Packing: PackedField<Scalar = F>,
    {
        assert_eq!(degree_bits.len(), instances.len());
        assert!(D > 1, "Not implemented for D=1.");
        assert!(
            !fri_params.hiding,
            "Batch FRI does not support hiding mode."
        );
        let alpha = challenger.get_extension_challenge::<D>();
        let runs = batch_alpha_runs(instances);
        timing.push("final-poly LDE", log::Level::Info);

        // The final low-degree polynomial of each FRI instance is built from
        // that instance's own batches (opening points and run reductions over
        // the shared oracles); nothing couples one instance to another, so
        // all of them — the run reductions, the linear divisions, and the
        // final LDE coset FFTs — compute in parallel. Inner scopes are
        // untimed here (the tree is not Sync); the enclosing
        // "compute batched openings proof" scope still reports the total.
        type FinalLde<FE> = (PolynomialCoeffs<FE>, PolynomialValues<FE>);
        let final_lde: Vec<FinalLde<F::Extension>> = instances
            .into_par_iter()
            .enumerate()
            .map(|(i, instance)| {
                // Final low-degree polynomial that goes into FRI: the sum of
                // the batches' quotient polys. All instances share the full
                // degree `1 << degree_bits[i]` (the quotient polys are padded
                // to it), so preallocate once and accumulate each batch's
                // quotient in parallel instead of growing a fresh poly by
                // serial `+=` per batch.
                let degree = 1 << degree_bits[i];
                let mut final_poly = PolynomialCoeffs::<F::Extension>::zero(degree);

                // Each batch consists of an opening point `z` and polynomials `{f_j}_j` to be
                // opened at that point. Per [`AlphaRun`], the instance's polynomial is
                //   `E_i = Σ_batches (B(X) - B(z)) / (X - z)`,
                //   `B(X) = Σ_runs alpha^offset(run) · Σ_j alpha^j · f_run,j(X)`,
                // giving every opened claim a distinct, canonical power of `alpha`.
                // There are usually two batches for the openings at `zeta` and `g * zeta`.
                for (k, FriBatchInfo { point, polynomials }) in instance.batches.iter().enumerate()
                {
                    let batch_runs: Vec<&crate::batch_fri::AlphaRun> = runs
                        .iter()
                        .filter(|r| r.instance == i && r.batch == k)
                        .collect();
                    let run_specs: Vec<(F::Extension, Vec<&PolynomialCoeffs<F>>)> = batch_runs
                        .iter()
                        .map(|run| {
                            let polys: Vec<&PolynomialCoeffs<F>> = polynomials
                                [run.flat_start..run.flat_start + run.len]
                                .iter()
                                .map(|fri_poly| {
                                    &oracles[fri_poly.oracle_index].polynomials
                                        [fri_poly.polynomial_index]
                                })
                                .collect();
                            (alpha.exp_u64(run.alpha_offset as u64), polys)
                        })
                        .collect();
                    debug_assert!(
                        run_specs
                            .iter()
                            .flat_map(|(_, polys)| polys.iter())
                            .all(|p| p.len() == degree),
                        "fused composition indexes polys up to the instance degree"
                    );
                    // Tiled packed composition. The batch's runs all sweep the
                    // same coefficient range, so tile over coefficients and
                    // keep one accumulator tile per task: every poly's tile
                    // slice is contiguous (prefetch-friendly) and the FE
                    // weight broadcasts into a packed multiply-add. The
                    // FE-weighted base axpy `acc += w * F::from(x)`
                    // decomposes over the extension coordinates into
                    // `acc_d += w_d * x` — packed base arithmetic, no
                    // extension-typed ops in the hot loop. Tiling also
                    // removes the small-instance pool starvation the
                    // per-run sweeps suffered behind the tall instance.
                    let mut composition_poly = PolynomialCoeffs::<F::Extension>::zero(degree);
                    {
                        const TILE: usize = 4096;
                        composition_poly
                            .coeffs
                            .par_chunks_mut(TILE)
                            .enumerate()
                            .for_each(|(t, dst_tile)| {
                                let lo = t * TILE;
                                let hi = lo + dst_tile.len();
                                // Per-coordinate base accumulator tiles.
                                let mut local: [Vec<F>; D] =
                                    core::array::from_fn(|_| vec![F::ZERO; hi - lo]);
                                for (run_alpha, polys) in run_specs.iter() {
                                    let mut w = *run_alpha;
                                    for p in polys.iter() {
                                        let w_arr = w.to_basefield_array();
                                        for (d, acc_d) in local.iter_mut().enumerate() {
                                            let width = <F as Packable>::Packing::WIDTH;
                                            let wd_vec: Vec<F> = (0..width).map(|_| w_arr[d]).collect();
                                            let wd = *<<F as Packable>::Packing as PackedField>::from_slice(
                                                &wd_vec,
                                            );
                                            let src = &p.coeffs[lo..hi];
                                            let src_packed =
                                                <<F as Packable>::Packing as PackedField>::pack_slice(src);
                                            let mut acc_packed =
                                                <<F as Packable>::Packing as PackedField>::pack_slice_mut(acc_d);
                                            for (dst_chunk, src_chunk) in acc_packed
                                                .iter_mut()
                                                .zip(src_packed.iter())
                                            {
                                                *dst_chunk = *dst_chunk + *src_chunk * wd;
                                            }
                                        }
                                        w *= alpha;
                                    }
                                }
                                // Store the FE tile from the coordinate
                                // accumulators.
                                for (off, dst) in dst_tile.iter_mut().enumerate() {
                                    let arr: [F; D] =
                                        core::array::from_fn(|d| local[d][off]);
                                    *dst = F::Extension::from_basefield_array(arr);
                                }
                            });
                    }
                    let quotient = divide_by_linear_par(&composition_poly.coeffs, *point);
                    debug_assert_eq!(quotient.len() + 1, degree);
                    final_poly
                        .coeffs
                        .par_iter_mut()
                        .zip(&quotient.coeffs)
                        .for_each(|(f, &q)| *f += q);
                }

                assert_eq!(final_poly.len(), degree);
                let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
                let lde_final_values = coset_fft_par(&lde_final_poly, F::coset_shift().into());
                (lde_final_poly, lde_final_values)
            })
            .collect();

        let mut final_lde_polynomial_coeff = Vec::with_capacity(final_lde.len());
        let mut final_lde_polynomial_values = Vec::with_capacity(final_lde.len());
        for (coeffs, values) in final_lde {
            final_lde_polynomial_coeff.push(coeffs);
            final_lde_polynomial_values.push(values);
        }
        timing.pop();

        batch_fri_proof_with_openings::<F, C, D>(
            &oracles
                .iter()
                .map(|o| &o.batch_merkle_tree)
                .collect::<Vec<_>>(),
            &|oracle, index| oracles[oracle].open_values(index),
            &final_lde_polynomial_coeff,
            &final_lde_polynomial_values,
            challenger,
            fri_params,
            timing,
        )
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(
        &self,
        degree_bits_index: usize,
        index: usize,
        step: usize,
        slice_start: usize,
        slice_len: usize,
    ) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_bits[degree_bits_index] + self.rate_bits);
        let rows = if self.row_values.is_empty() {
            &self.batch_merkle_tree.leaves
        } else {
            &self.row_values
        };
        let slice = &rows[degree_bits_index][index];
        &slice[slice_start..slice_start + slice_len]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(
        &self,
        degree_bits_index: usize,
        index_start: usize,
        step: usize,
        slice_start: usize,
        slice_len: usize,
    ) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| {
                self.get_lde_values(
                    degree_bits_index,
                    index_start + i,
                    step,
                    slice_start,
                    slice_len,
                )
            })
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

    /// [`Self::get_lde_values_packed`] into a caller-provided buffer, so
    /// quotient-loop row fetches can reuse allocations across evaluation
    /// points. `buf` must have exactly the desired slice length; the packed
    /// value of column `j` lands in `buf[j]`.
    pub fn get_lde_values_packed_into<P>(
        &self,
        degree_bits_index: usize,
        index_start: usize,
        step: usize,
        slice_start: usize,
        buf: &mut [P],
    ) where
        P: PackedField<Scalar = F>,
    {
        let slice_len = buf.len();
        // The `P::WIDTH` point rows, fetched once and transposed column-wise
        // into the buffer (mirrors `get_lde_values_packed` without the
        // per-call allocation).
        let rows: Vec<&[F]> = (0..P::WIDTH)
            .map(|w| {
                self.get_lde_values(
                    degree_bits_index,
                    index_start + w,
                    step,
                    slice_start,
                    slice_len,
                )
            })
            .collect();
        for (j, slot) in buf.iter_mut().enumerate() {
            let mut packed = P::ZEROS;
            packed
                .as_slice_mut()
                .iter_mut()
                .zip(rows.iter())
                .for_each(|(packed_w, row_w)| *packed_w = row_w[j]);
            *slot = packed;
        }
    }
}

#[cfg(test)]
mod test {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::Sample;

    use super::*;
    use crate::batch_fri::verifier::verify_batch_fri_proof;
    use crate::fri::reduction_strategies::FriReductionStrategy;
    use crate::fri::structure::{
        FriBatchInfo, FriBatchInfoTarget, FriInstanceInfo, FriInstanceInfoTarget, FriOpeningBatch,
        FriOpeningBatchTarget, FriOpenings, FriOpeningsTarget, FriOracleInfo, FriPolynomialInfo,
    };
    use crate::fri::witness_util::set_fri_proof_target;
    use crate::fri::FriConfig;
    use crate::iop::challenger::RecursiveChallenger;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::plonk::prover::prove;

    const D: usize = 2;

    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    type H = <C as GenericConfig<D>>::Hasher;

    #[test]
    fn batch_prove_openings() -> anyhow::Result<()> {
        let mut timing = TimingTree::default();

        let k0 = 9;
        let k1 = 8;
        let k2 = 6;
        let reduction_arity_bits = vec![1, 2, 1];
        let fri_params = FriParams {
            config: FriConfig {
                rate_bits: 1,
                cap_height: 0,
                proof_of_work_bits: 0,
                reduction_strategy: FriReductionStrategy::Fixed(reduction_arity_bits.clone()),
                num_query_rounds: 10,
            },
            hiding: false,
            degree_bits: k0,
            reduction_arity_bits,
        };

        let n0 = 1 << k0;
        let n1 = 1 << k1;
        let n2 = 1 << k2;
        let trace0 = PolynomialValues::new(F::rand_vec(n0));
        let trace1_0 = PolynomialValues::new(F::rand_vec(n1));
        let trace1_1 = PolynomialValues::new(F::rand_vec(n1));
        let trace2 = PolynomialValues::new(F::rand_vec(n2));

        let trace_oracle: BatchFriOracle<GoldilocksField, C, D> = BatchFriOracle::from_values(
            vec![
                trace0.clone(),
                trace1_0.clone(),
                trace1_1.clone(),
                trace2.clone(),
            ],
            fri_params.config.rate_bits,
            fri_params.hiding,
            fri_params.config.cap_height,
            &mut timing,
            &[None; 4],
        );

        let mut challenger = Challenger::<F, H>::new();
        challenger.observe_cap(&trace_oracle.batch_merkle_tree.cap);
        let zeta = challenger.get_extension_challenge::<D>();
        let eta = challenger.get_extension_challenge::<D>();
        let poly0 = &trace_oracle.polynomials[0];
        let poly1_0 = &trace_oracle.polynomials[1];
        let poly1_1 = &trace_oracle.polynomials[2];
        let poly2 = &trace_oracle.polynomials[3];

        let mut challenger = Challenger::<F, H>::new();
        let mut verifier_challenger = challenger.clone();

        let fri_instance_0 = FriInstanceInfo {
            oracles: vec![FriOracleInfo {
                num_polys: 1,
                blinding: false,
            }],
            batches: vec![
                FriBatchInfo {
                    point: zeta,
                    polynomials: vec![FriPolynomialInfo {
                        oracle_index: 0,
                        polynomial_index: 0,
                    }],
                },
                FriBatchInfo {
                    point: eta,
                    polynomials: vec![FriPolynomialInfo {
                        oracle_index: 0,
                        polynomial_index: 0,
                    }],
                },
            ],
        };
        let fri_instance_1 = FriInstanceInfo {
            oracles: vec![FriOracleInfo {
                num_polys: 2,
                blinding: false,
            }],
            batches: vec![
                FriBatchInfo {
                    point: zeta,
                    polynomials: vec![
                        FriPolynomialInfo {
                            oracle_index: 0,
                            polynomial_index: 1,
                        },
                        FriPolynomialInfo {
                            oracle_index: 0,
                            polynomial_index: 2,
                        },
                    ],
                },
                FriBatchInfo {
                    point: eta,
                    polynomials: vec![FriPolynomialInfo {
                        oracle_index: 0,
                        polynomial_index: 2,
                    }],
                },
            ],
        };
        let fri_instance_2 = FriInstanceInfo {
            oracles: vec![FriOracleInfo {
                num_polys: 1,
                blinding: false,
            }],
            batches: vec![FriBatchInfo {
                point: zeta,
                polynomials: vec![FriPolynomialInfo {
                    oracle_index: 0,
                    polynomial_index: 3,
                }],
            }],
        };
        let fri_instances = vec![fri_instance_0, fri_instance_1, fri_instance_2];
        let poly0_zeta = poly0.to_extension::<D>().eval(zeta);
        let poly0_eta = poly0.to_extension::<D>().eval(eta);
        let fri_opening_batch_0 = FriOpenings {
            batches: vec![
                FriOpeningBatch {
                    values: vec![poly0_zeta],
                },
                FriOpeningBatch {
                    values: vec![poly0_eta],
                },
            ],
        };
        let poly10_zeta = poly1_0.to_extension::<D>().eval(zeta);
        let poly11_zeta = poly1_1.to_extension::<D>().eval(zeta);
        let poly11_eta = poly1_1.to_extension::<D>().eval(eta);
        let fri_opening_batch_1 = FriOpenings {
            batches: vec![
                FriOpeningBatch {
                    values: vec![poly10_zeta, poly11_zeta],
                },
                FriOpeningBatch {
                    values: vec![poly11_eta],
                },
            ],
        };
        let poly2_zeta = poly2.to_extension::<D>().eval(zeta);
        let fri_opening_batch_2 = FriOpenings {
            batches: vec![FriOpeningBatch {
                values: vec![poly2_zeta],
            }],
        };
        let fri_openings = vec![
            fri_opening_batch_0,
            fri_opening_batch_1,
            fri_opening_batch_2,
        ];

        let proof = BatchFriOracle::prove_openings(
            &[k0, k1, k2],
            &fri_instances,
            &[&trace_oracle],
            &mut challenger,
            &fri_params,
            &mut timing,
        );

        let fri_challenges = verifier_challenger.fri_challenges::<C, D>(
            &proof.commit_phase_merkle_caps,
            &proof.final_poly,
            proof.pow_witness,
            k0,
            &fri_params.config,
            None,
            None,
        );
        let degree_bits = [k0, k1, k2];
        let merkle_cap = trace_oracle.batch_merkle_tree.cap;
        verify_batch_fri_proof::<GoldilocksField, C, D>(
            &degree_bits,
            &fri_instances,
            &fri_openings,
            &fri_challenges,
            core::slice::from_ref(&merkle_cap),
            &proof,
            &fri_params,
        )?;

        // Test recursive verifier
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let num_leaves_per_oracle = vec![4];
        let fri_proof_target = builder.add_virtual_fri_proof(&num_leaves_per_oracle, &fri_params);
        let zeta_target = builder.constant_extension(zeta);
        let eta_target = builder.constant_extension(eta);
        let fri_instance_info_target_0 = FriInstanceInfoTarget {
            oracles: vec![FriOracleInfo {
                num_polys: 1,
                blinding: false,
            }],
            batches: vec![
                FriBatchInfoTarget {
                    point: zeta_target,
                    polynomials: vec![FriPolynomialInfo {
                        oracle_index: 0,
                        polynomial_index: 0,
                    }],
                },
                FriBatchInfoTarget {
                    point: eta_target,
                    polynomials: vec![FriPolynomialInfo {
                        oracle_index: 0,
                        polynomial_index: 0,
                    }],
                },
            ],
        };
        let fri_instance_info_target_1 = FriInstanceInfoTarget {
            oracles: vec![FriOracleInfo {
                num_polys: 2,
                blinding: false,
            }],
            batches: vec![
                FriBatchInfoTarget {
                    point: zeta_target,
                    polynomials: vec![
                        FriPolynomialInfo {
                            oracle_index: 0,
                            polynomial_index: 1,
                        },
                        FriPolynomialInfo {
                            oracle_index: 0,
                            polynomial_index: 2,
                        },
                    ],
                },
                FriBatchInfoTarget {
                    point: eta_target,
                    polynomials: vec![FriPolynomialInfo {
                        oracle_index: 0,
                        polynomial_index: 2,
                    }],
                },
            ],
        };
        let fri_instance_info_target_2 = FriInstanceInfoTarget {
            oracles: vec![FriOracleInfo {
                num_polys: 1,
                blinding: false,
            }],
            batches: vec![FriBatchInfoTarget {
                point: zeta_target,
                polynomials: vec![FriPolynomialInfo {
                    oracle_index: 0,
                    polynomial_index: 3,
                }],
            }],
        };

        let poly0_zeta_target = builder.constant_extension(poly0_zeta);
        let poly0_eta_target = builder.constant_extension(poly0_eta);
        let fri_opening_batch_0 = FriOpeningsTarget {
            batches: vec![
                FriOpeningBatchTarget {
                    values: vec![poly0_zeta_target],
                },
                FriOpeningBatchTarget {
                    values: vec![poly0_eta_target],
                },
            ],
        };
        let poly10_zeta_target = builder.constant_extension(poly10_zeta);
        let poly11_zeta_target = builder.constant_extension(poly11_zeta);
        let poly11_eta_target = builder.constant_extension(poly11_eta);
        let fri_opening_batch_1 = FriOpeningsTarget {
            batches: vec![
                FriOpeningBatchTarget {
                    values: vec![poly10_zeta_target, poly11_zeta_target],
                },
                FriOpeningBatchTarget {
                    values: vec![poly11_eta_target],
                },
            ],
        };
        let poly2_zeta_target = builder.constant_extension(poly2_zeta);
        let fri_opening_batch_2 = FriOpeningsTarget {
            batches: vec![FriOpeningBatchTarget {
                values: vec![poly2_zeta_target],
            }],
        };
        let fri_openings_target = [
            fri_opening_batch_0,
            fri_opening_batch_1,
            fri_opening_batch_2,
        ];

        let mut challenger = RecursiveChallenger::<F, H, D>::new(&mut builder);
        let fri_challenges_target = challenger.fri_challenges(
            &mut builder,
            &fri_proof_target.commit_phase_merkle_caps,
            &fri_proof_target.final_poly,
            fri_proof_target.pow_witness,
            &fri_params.config,
        );

        let merkle_cap_target = builder.constant_merkle_cap(&merkle_cap);

        let fri_instance_info_target = vec![
            fri_instance_info_target_0,
            fri_instance_info_target_1,
            fri_instance_info_target_2,
        ];

        builder.verify_batch_fri_proof::<C>(
            &degree_bits,
            &fri_instance_info_target,
            &fri_openings_target,
            &fri_challenges_target,
            &[merkle_cap_target],
            &fri_proof_target,
            &fri_params,
        );

        let mut pw = PartialWitness::new();
        set_fri_proof_target(&mut pw, &fri_proof_target, &proof)?;

        let data = builder.build::<C>();
        let proof = prove::<F, C, D>(&data.prover_only, &data.common, pw, &mut timing)?;
        data.verify(proof.clone())?;

        Ok(())
    }

    /// The blocked parallel linear division must be value-identical to the
    /// serial `PolynomialCoeffs::divide_by_linear` across sizes (including
    /// non-power-of-two lengths and the block boundary).
    #[test]
    fn divide_by_linear_par_matches_serial() {
        use plonky2_field::polynomial::PolynomialCoeffs as PC;

        const BLOCK: usize = 1 << 14;
        for &n in &[0usize, 1, 2, 3, BLOCK - 1, BLOCK, BLOCK + 1, 3 * BLOCK + 7] {
            let coeffs = F::rand_vec(n);
            let z = F::rand();
            let serial = PC::new(coeffs.clone()).divide_by_linear(z);
            let par = divide_by_linear_par(&coeffs, z);
            assert_eq!(serial.len(), par.len(), "len mismatch at n={n}");
            assert_eq!(serial.coeffs, par.coeffs, "value mismatch at n={n}");
        }
    }
}
