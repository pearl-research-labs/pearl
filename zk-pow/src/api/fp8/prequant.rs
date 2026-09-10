//! The miner's prequantized input dtype — int8 values + per-block BF16 scales —
//! mirroring `miner_base.prequant`.
//!
//! Each committed operand is TWO strips: an int8 values tensor `(n x k)` and a
//! per-block BF16 scales tensor `(n x k/BLOCK_SIZE)`. A block of [`BLOCK_SIZE`]
//! contiguous int8 values shares one scale and opens to BF16 as
//! `int_value * scale`. The two strips are committed separately and combined
//! (`blake3(commit(int8) || commit(scales))`, the reference's `commit_planes`);
//! the plaintext FP8 verifier opens them here and feeds the result to the
//! quantization scheme, exactly as the miner opens its inputs before mining.
//!
//! A strip ([`PrequantSlice`]) is a checked concatenation of equal-width rows.
//! Hashing and wire code see the stored bytes; signed int8 interpretation
//! happens only at the arithmetic boundary (`byte as i8`). One operand's two
//! strips are bundled in [`PrequantOperand`].

use anyhow::{Result, ensure};

use crate::api::fp8::dtype::{bf16_to_f32, f32_to_bf16};

/// Block size of the whitelisted `int8 blk8 bf16s` format: one BF16 scale per
/// [`BLOCK_SIZE`] contiguous int8 values (`DEFAULT_BLOCK_SIZE` in the reference).
pub const BLOCK_SIZE: usize = 8;

/// Low explicit BF16 mantissa bits cleared from `l2` (see [`round_l2_to_grid`]).
pub const L2_ROUNDED_BITS: u32 = 2;

/// Round a non-negative BF16 to the nearest multiple of `2^L2_ROUNDED_BITS`
/// ulps (ties up), clearing its low [`L2_ROUNDED_BITS`] mantissa bits.
/// Non-negative IEEE-754 floats order the same as their bit patterns, so this is
/// "add half a grid step to the u16 view, clear the low bits", any carry into
/// the kept mantissa/exponent bits falling out of the plain integer addition.
/// Vs. ceiling to the same grid it is equally robust to a 1-ulp host
/// discrepancy but has half the mean bias (0.5 vs 1.5 ulp). Called wherever a
/// row's exact `sumsq` is turned into `l2`, so a <=1-ulp f32 discrepancy in
/// `sumsq` between miner and verifier can't change the result.
pub fn round_l2_to_grid(l2: u16) -> u16 {
    let step: u16 = 1 << L2_ROUNDED_BITS;
    l2.wrapping_add(step >> 1) & !(step - 1)
}

/// Per-row `(l2, linf)` (BF16 bits) computed straight from the prequant planes,
/// bypassing [`open_prequant`]'s per-element BF16 rounding — mirroring
/// `PrequantMatrix.exact_norms` in the reference miner.
///
/// Per block: `scale^2 * sum(int_i^2)` and `scale * max(|int_i|)`.
/// `sum(int_i^2)` over a `block_size`-wide int8 block is exact in f32
/// (`<= 127^2 * 8` for the default block size), so each block's contribution to
/// `sumsq` only picks up the single f32 rounding of the `scale^2` multiply.
/// `l2 = rms(X) = sqrt(sumsq / k)`, grid-rounded ([`round_l2_to_grid`]) so the
/// per-row f32 block-sum order can differ from the reference's by an ulp
/// without changing the result.
pub fn exact_norms(int_values: &[i8], scales: &[u16], num_rows: usize, k: usize, block_size: usize) -> Result<Vec<(u16, u16)>> {
    ensure!(
        block_size > 0 && k > 0 && k.is_multiple_of(block_size),
        "k={k} must be a positive multiple of block_size={block_size}"
    );
    let n_blocks = k / block_size;
    ensure!(int_values.len() == num_rows * k, "int_values must be num_rows x k");
    ensure!(
        scales.len() == num_rows * n_blocks,
        "scales must be num_rows x (k / block_size)"
    );
    let mut norms = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let mut sumsq = 0.0f32;
        let mut linf = f32::NEG_INFINITY;
        for b in 0..n_blocks {
            let ints = &int_values[i * k + b * block_size..i * k + (b + 1) * block_size];
            // Both block reductions are exact in f32: sum of int8 squares
            // (<= 127^2 * block_size) and the int8 abs-max.
            let block_sumsq: f32 = ints.iter().map(|&v| (v as f32) * (v as f32)).sum();
            let block_amax = ints.iter().map(|&v| (v as i32).unsigned_abs()).max().expect("block_size > 0") as f32;
            let scale = bf16_to_f32(scales[i * n_blocks + b]); // exact
            sumsq += block_sumsq * (scale * scale);
            linf = linf.max(block_amax * scale.abs()); // 15-bit product, exact in f32
        }
        ensure!(sumsq.is_finite(), "row {i} sum of squares overflows f32");
        let l2 = round_l2_to_grid(f32_to_bf16((sumsq / k as f32).sqrt())?);
        norms.push((l2, f32_to_bf16(linf)?));
    }
    Ok(norms)
}

/// Open a stack of prequantized rows to BF16:
/// `opened[i][j] = int8[i][j] * scale[i][j / block_size]`.
///
/// `int_values` is `(num_rows x k)` row-major int8; `scales` is
/// `(num_rows x k/block_size)` row-major BF16 (`u16`). The `int8 * scale` product
/// is exact in f32 (int8 has <= 7 magnitude bits, BF16 8), so the only rounding
/// is the final RNE cast back to BF16 — mirroring `PrequantMatrix.open`.
pub fn open_prequant(int_values: &[i8], scales: &[u16], num_rows: usize, k: usize, block_size: usize) -> Result<Vec<u16>> {
    ensure!(
        block_size > 0 && k.is_multiple_of(block_size),
        "k={k} must be a multiple of block_size={block_size}"
    );
    let n_blocks = k / block_size;
    ensure!(int_values.len() == num_rows * k, "int_values must be num_rows x k");
    ensure!(
        scales.len() == num_rows * n_blocks,
        "scales must be num_rows x (k / block_size)"
    );
    let mut out = Vec::with_capacity(num_rows * k);
    for i in 0..num_rows {
        for j in 0..k {
            let value = int_values[i * k + j] as f32; // exact
            let scale = bf16_to_f32(scales[i * n_blocks + j / block_size]); // exact
            out.push(f32_to_bf16(value * scale)?); // product exact in f32, single RNE -> BF16
        }
    }
    Ok(out)
}

/// A Merkle-committed slice of `row_bytes`-wide rows, stored as one contiguous
/// byte buffer in row order. The row count is `bytes.len() / row_bytes` (see
/// [`row_count`](Self::row_count)).
///
/// Represents one of [`PrequantOperand`]'s strips.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrequantSlice {
    row_bytes: usize, // bytes per row
    bytes: Vec<u8>,
}

impl PrequantSlice {
    /// Concatenate `rows` after checking that every row has `expected_row_bytes`.
    pub fn try_from_rows(rows: Vec<Vec<u8>>, expected_row_bytes: usize) -> Result<Self> {
        let total = Self::total_bytes(rows.len(), expected_row_bytes)?;
        let mut bytes = Vec::with_capacity(total);
        for (i, row) in rows.iter().enumerate() {
            ensure!(
                row.len() == expected_row_bytes,
                "committed row {i} has {} bytes, expected {expected_row_bytes}",
                row.len()
            );
            bytes.extend_from_slice(row);
        }
        debug_assert_eq!(bytes.len(), total);
        Ok(Self {
            row_bytes: expected_row_bytes,
            bytes,
        })
    }

    /// Concatenate signed rows, storing each `i8` as its canonical `u8` bit pattern.
    pub fn try_from_i8_rows(rows: Vec<Vec<i8>>, expected_row_bytes: usize) -> Result<Self> {
        let rows = rows
            .into_iter()
            .map(|row| row.into_iter().map(|b| b as u8).collect())
            .collect();
        Self::try_from_rows(rows, expected_row_bytes)
    }

    /// Build a slice from already-concatenated row-major bytes.
    pub fn try_from_concatenated(row_bytes: usize, bytes: Vec<u8>) -> Result<Self> {
        ensure!(row_bytes > 0, "committed row width must be positive");
        ensure!(
            bytes.len().is_multiple_of(row_bytes),
            "concatenated length {} is not a multiple of row width {row_bytes}",
            bytes.len()
        );
        Ok(Self { row_bytes, bytes })
    }

    fn total_bytes(row_count: usize, row_bytes: usize) -> Result<usize> {
        ensure!(row_bytes > 0, "committed row width must be positive");
        row_count
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("committed-row length overflow"))
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub fn row_count(&self) -> usize {
        self.bytes.len() / self.row_bytes
    }

    pub fn row(&self, index: usize) -> &[u8] {
        let n = self.row_count();
        assert!(index < n, "committed row {index} out of range ({n} rows)");
        let start = index * self.row_bytes;
        &self.bytes[start..start + self.row_bytes]
    }

    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.bytes.chunks_exact(self.row_bytes)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// One FP8 operand's two committed strips (int8 values and per-block BF16 scales).
///
/// FP8 format: every consecutive 8 entries are encoded as 8 int8 values and 1 BF16 scale.
/// Cross-strip row-count equality is checked by [`num_rows`](Self::num_rows).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrequantOperand {
    pub values: PrequantSlice,
    pub scales: PrequantSlice,
}

impl PrequantOperand {
    /// Number of rows in both strips, after checking they agree.
    pub fn num_rows(&self) -> Result<usize> {
        let n = self.values.row_count();
        ensure!(
            self.scales.row_count() == n,
            "values and scales strips must have equal row counts"
        );
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-checked against the reference `PrequantMatrix.open` (torch),
    /// generated by running `miner_base`.
    #[test]
    fn open_prequant_matches_reference_vectors() {
        // 2 rows x 16 cols, block_size 8 => 2 blocks/row.
        let int_values: [i8; 32] = [
            5, -3, 127, 0, 1, -1, 64, -64, 2, 100, -100, 7, -7, 33, -33, 12, // row 0
            -1, 1, -2, 2, 3, -3, 4, -4, 8, -8, 16, -16, 32, -32, 63, -63, // row 1
        ];
        // scales (BF16 bits): row 0 = [0x3ca0, 0x4020], row 1 = [0x3f00, 0x3d00].
        let scales: [u16; 4] = [0x3ca0, 0x4020, 0x3f00, 0x3d00];
        let expected: [u16; 32] = [
            0x3dc8, 0xbd70, 0x401f, 0x0000, 0x3ca0, 0xbca0, 0x3fa0, 0xbfa0, // row 0
            0x40a0, 0x437a, 0xc37a, 0x418c, 0xc18c, 0x42a5, 0xc2a5, 0x41f0, //
            0xbf00, 0x3f00, 0xbf80, 0x3f80, 0x3fc0, 0xbfc0, 0x4000, 0xc000, // row 1
            0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x3f80, 0xbf80, 0x3ffc, 0xbffc, //
        ];
        let opened = open_prequant(&int_values, &scales, 2, 16, BLOCK_SIZE).unwrap();
        assert_eq!(opened, expected);
    }

    #[test]
    fn open_prequant_validates_shapes() {
        assert!(open_prequant(&[1, 2, 3], &[0x3f80], 1, 3, 8).is_err()); // k not multiple of block
        assert!(open_prequant(&[0i8; 8], &[0x3f80, 0x3f80], 1, 8, 8).is_err()); // scales too long
        assert!(open_prequant(&[0i8; 7], &[0x3f80], 1, 8, 8).is_err()); // int_values too short
    }

    /// Cross-checked against the reference `PrequantMatrix.exact_norms` (torch),
    /// generated by running `miner_base`
    /// on the same fixture as `open_prequant_matches_reference_vectors`.
    #[test]
    fn exact_norms_matches_reference_vectors() {
        let int_values: [i8; 32] = [
            5, -3, 127, 0, 1, -1, 64, -64, 2, 100, -100, 7, -7, 33, -33, 12, // row 0
            -1, 1, -2, 2, 3, -3, 4, -4, 8, -8, 16, -16, 32, -32, 63, -63, // row 1
        ];
        let scales: [u16; 4] = [0x3ca0, 0x4020, 0x3f00, 0x3d00];
        let norms = exact_norms(&int_values, &scales, 2, 16, BLOCK_SIZE).unwrap();
        assert_eq!(norms, vec![(0x42bc, 0x437a), (0x3fa0, 0x4000)]);
    }

    #[test]
    fn exact_norms_validates_shapes() {
        assert!(exact_norms(&[1, 2, 3], &[0x3f80], 1, 3, 8).is_err()); // k not multiple of block
        assert!(exact_norms(&[0i8; 8], &[0x3f80, 0x3f80], 1, 8, 8).is_err()); // scales too long
        assert!(exact_norms(&[0i8; 7], &[0x3f80], 1, 8, 8).is_err()); // int_values too short
    }

    #[test]
    fn round_l2_clears_tail_bits_and_rounds_to_nearest() {
        // Tail already zero: unchanged.
        assert_eq!(round_l2_to_grid(0x3F80), 0x3F80); // 1.0
        // Round to the nearest multiple of 4, ties up.
        assert_eq!(round_l2_to_grid(0x3F81), 0x3F80);
        assert_eq!(round_l2_to_grid(0x3F82), 0x3F84); // tie rounds up
        assert_eq!(round_l2_to_grid(0x3F83), 0x3F84);
        assert_eq!(round_l2_to_grid(0x3F84), 0x3F84);
        // Carry propagates through the mantissa into the exponent.
        assert_eq!(round_l2_to_grid(0x3FFF), 0x4000);
    }

    #[test]
    fn zero_width_and_empty_inner_rows_reject() {
        assert!(PrequantSlice::try_from_rows(vec![], 0).is_err());
        assert!(PrequantSlice::try_from_rows(vec![vec![]], 1).is_err());
        assert!(PrequantSlice::try_from_i8_rows(vec![vec![]], 4).is_err());
        assert!(PrequantSlice::try_from_concatenated(0, vec![]).is_err());
    }

    #[test]
    fn ragged_rows_reject() {
        assert!(PrequantSlice::try_from_rows(vec![vec![1, 2], vec![3]], 2).is_err());
        assert!(PrequantSlice::try_from_i8_rows(vec![vec![1i8, 2], vec![3, 4, 5]], 2).is_err());
        assert!(PrequantSlice::try_from_concatenated(3, vec![1, 2, 3, 4]).is_err());
    }

    #[test]
    fn total_length_overflow_rejects() {
        let err = PrequantSlice::try_from_rows(vec![Vec::new(); 3], usize::MAX).expect_err("3 * usize::MAX must overflow");
        assert!(err.to_string().contains("overflow"), "{err}");
        let err = PrequantSlice::try_from_i8_rows(vec![Vec::new(); 2], usize::MAX).expect_err("2 * usize::MAX must overflow");
        assert!(err.to_string().contains("overflow"), "{err}");
    }

    #[test]
    fn empty_slice_with_positive_width_is_valid() {
        let slice = PrequantSlice::try_from_rows(vec![], 8).unwrap();
        assert_eq!(slice.row_bytes(), 8);
        assert_eq!(slice.row_count(), 0);
        assert!(slice.as_bytes().is_empty());
        assert_eq!(slice.rows().len(), 0);
    }

    #[test]
    fn row_count_index_and_iteration_are_exact() {
        let slice = PrequantSlice::try_from_rows(vec![vec![1, 2, 3], vec![4, 5, 6]], 3).unwrap();
        assert_eq!(slice.row_count(), 2);
        assert_eq!(slice.row_bytes(), 3);
        assert_eq!(slice.row(0), &[1, 2, 3]);
        assert_eq!(slice.row(1), &[4, 5, 6]);
        let collected: Vec<&[u8]> = slice.rows().collect();
        assert_eq!(collected, vec![&[1, 2, 3][..], &[4, 5, 6][..]]);
        assert_eq!(slice.rows().len(), 2);
    }

    #[test]
    fn byte_order_equals_nested_concatenation() {
        let nested = [vec![10u8, 11, 12, 13], vec![20, 21, 22, 23]];
        let slice = PrequantSlice::try_from_rows(nested.to_vec(), 4).unwrap();
        let concat: Vec<u8> = nested.iter().flatten().copied().collect();
        assert_eq!(slice.as_bytes(), concat.as_slice());
        assert_eq!(PrequantSlice::try_from_concatenated(4, concat.clone()).unwrap(), slice);
        // The stored buffer is the concatenation; `as_bytes` does not allocate a copy.
        assert_eq!(slice.as_bytes().as_ptr(), slice.row(0).as_ptr());
        let moved = vec![1u8, 2, 3, 4, 5, 6];
        let ptr = moved.as_ptr();
        let slice = PrequantSlice::try_from_concatenated(3, moved).unwrap();
        assert_eq!(slice.as_bytes().as_ptr(), ptr, "from_concatenated must keep the allocation");
    }

    #[test]
    fn signed_value_bytes_match_i8_bit_pattern() {
        let signed = vec![vec![-128i8, -1, 0, 127]];
        let slice = PrequantSlice::try_from_i8_rows(signed.clone(), 4).unwrap();
        assert_eq!(slice.as_bytes(), &[0x80, 0xff, 0x00, 0x7f]);
        let roundtrip: Vec<i8> = slice.as_bytes().iter().map(|&b| b as i8).collect();
        assert_eq!(roundtrip, signed[0]);
    }

    #[test]
    fn contiguous_open_matches_nested_i8_rows() {
        let ints = [vec![5i8, -3, 127, 0, 1, -1, 64, -64], vec![-1, 1, -2, 2, 3, -3, 4, -4]];
        let k = ints[0].len();
        let scale_codes = [0x3f80u16, 0x4000];
        let scale_rows: Vec<Vec<u8>> = scale_codes.iter().map(|c| c.to_le_bytes().to_vec()).collect();
        let values = PrequantSlice::try_from_i8_rows(ints.to_vec(), k).unwrap();
        let scales = PrequantSlice::try_from_rows(scale_rows, 2).unwrap();

        let nested_ints: Vec<i8> = ints.iter().flatten().copied().collect();
        let contig_ints: Vec<i8> = values.as_bytes().iter().map(|&b| b as i8).collect();
        let contig_scale_bits: Vec<u16> = scales
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        assert_eq!(contig_ints, nested_ints);
        assert_eq!(contig_scale_bits, scale_codes);

        let n = ints.len();
        assert_eq!(
            open_prequant(&contig_ints, &contig_scale_bits, n, k, BLOCK_SIZE).unwrap(),
            open_prequant(&nested_ints, &scale_codes, n, k, BLOCK_SIZE).unwrap()
        );
        assert_eq!(
            exact_norms(&contig_ints, &contig_scale_bits, n, k, BLOCK_SIZE).unwrap(),
            exact_norms(&nested_ints, &scale_codes, n, k, BLOCK_SIZE).unwrap()
        );
    }

    #[test]
    fn scale_bytes_are_little_endian_bf16_pairs() {
        let scale: u16 = 0x3f80;
        let rows = vec![scale.to_le_bytes().to_vec(), 0xbf00u16.to_le_bytes().to_vec()];
        let slice = PrequantSlice::try_from_rows(rows, 2).unwrap();
        let decoded: Vec<u16> = slice
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        assert_eq!(decoded, vec![0x3f80, 0xbf00]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn row_index_out_of_range_panics() {
        let slice = PrequantSlice::try_from_rows(vec![vec![1, 2]], 2).unwrap();
        let _ = slice.row(1);
    }
}
