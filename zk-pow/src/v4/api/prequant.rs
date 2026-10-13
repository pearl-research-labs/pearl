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

use anyhow::{Context, Result, ensure};

use crate::v4::api::dtype::{bf16_to_f32, check_not_nan_or_inf_bf16, f32_to_bf16};
use crate::v4::circuit::scale_stark::stark::rne_sqrt_hat;

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
/// Vs. ceiling to the same grid it has half the mean bias (0.5 vs 1.5 ulp).
/// Applied to the exactly rounded square root in [`exact_norms`] (ScaleStark group G).
pub fn round_l2_to_grid(l2: u16) -> u16 {
    let step: u16 = 1 << L2_ROUNDED_BITS;
    l2.wrapping_add(step >> 1) & !(step - 1)
}

/// `Wl2 = 27 - ceil(log2 k)`: the precision shift of the block-integer L2 frame sum
/// (ScaleStark's `WL2_PUBLIC_INPUT`, InputQuantStark's `2^Wl2`); `[11, 17]` over the
/// envelope `1024 <= k <= 2^16`. With `p_b < 2^37` it keeps every shifted block product
/// below `2^54` and the frame sum below `2^61`. Changing 27 changes the per-block floors
/// and therefore protocol output.
pub fn l2_frame_width(k: usize) -> Result<u32> {
    let ceil_log2_k = k.next_power_of_two().trailing_zeros();
    ensure!(ceil_log2_k <= 27, "k={k} exceeds the L2 frame width");
    Ok(27 - ceil_log2_k)
}

/// Per-row `(l2, linf)` (BF16 bits) computed straight from the prequant planes,
/// bypassing [`open_prequant`]'s per-element BF16 rounding. Bit-identical to the
/// ZK certificate (InputQuantStark group B + ScaleStark groups Q and G).
///
/// `l2` is `grid4(RNE_bf16(sqrt(v)))` ([`round_l2_to_grid`]) of the canonical
/// mean square `v = S * 2^(E_MAX - 268 - Wl2) / k`, where per block
/// `p_b = M(scale_b)^2 * sum(int_i^2)`, `E_MAX = max 2*E*(scale_b)` over blocks
/// with `p_b != 0`, and `S = sum floor(p_b * 2^Wl2 / 2^(E_MAX - 2*E*(scale_b)))`
/// over the same blocks — exact integers, one rounding in the square root.
///
/// `linf` is `RNE_bf16(max(|int_i| * |scale|))`, exact in f32 before the cast.
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
    let wl2 = l2_frame_width(k)?;
    int_values
        .chunks_exact(k)
        .zip(scales.chunks_exact(n_blocks))
        .enumerate()
        .map(|(i, (ints, scales))| row_norms(ints, scales, block_size, wl2).with_context(|| format!("row {i}")))
        .collect()
}

/// One row of [`exact_norms`]. Two passes over the row: the frame exponent must be known
/// before any block term can be floored.
fn row_norms(ints: &[i8], scales: &[u16], block_size: usize, wl2: u32) -> Result<(u16, u16)> {
    // `|scale| = M * 2^(E* - 134)`; returns `(M, 2*E*)`.
    let decode = |code: u16| {
        let exp_field = (code & 0x7FFF) >> 7;
        let m = u64::from(code & 0x7F) + if exp_field == 0 { 0 } else { 128 };
        (m, 2 * u32::from(exp_field.max(1)))
    };
    let blocks = || ints.chunks_exact(block_size).zip(scales);

    // A block is live (`p_b != 0`) iff it has a nonzero int and a nonzero scale.
    let mut e_max = 0;
    let mut linf = 0.0f32;
    for (b, (block, &code)) in blocks().enumerate() {
        check_not_nan_or_inf_bf16(code).with_context(|| format!("block {b} scale"))?;
        let amax = block.iter().map(|v| v.unsigned_abs()).max().expect("block_size > 0");
        let (m, doubled_exp) = decode(code);
        if amax != 0 && m != 0 {
            e_max = e_max.max(doubled_exp);
        }
        linf = linf.max(f32::from(amax) * bf16_to_f32(code & 0x7FFF)); // exact in f32
    }
    let linf = f32_to_bf16(linf)?;

    // Dead blocks contribute 0 whatever their (saturated) shift; p_b < 2^33 and Wl2 <= 27
    // keep every shifted product below 2^60 and S below 2^62.
    let s: u64 = blocks()
        .map(|(block, &code)| {
            let (m, doubled_exp) = decode(code);
            let p = m * m * block.iter().map(|&v| u64::from(v.unsigned_abs()).pow(2)).sum::<u64>();
            (p << wl2).checked_shr(e_max.saturating_sub(doubled_exp)).unwrap_or(0)
        })
        .sum();
    let l2 = if s == 0 {
        0
    } else {
        // The finite linf above bounds sqrt(v) below bf16 overflow, which `rne_sqrt_hat` asserts.
        let claim = rne_sqrt_hat(s, i64::from(e_max), i64::from(wl2), ints.len() as u64);
        round_l2_to_grid(((claim.exp << 7) + claim.mantissa) as u16)
    };
    ensure!(l2 < 0x7F80, "l2 snaps into the bf16 infinity code");
    Ok((l2, linf))
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
    for (i, &scale) in scales.iter().enumerate() {
        check_not_nan_or_inf_bf16(scale).with_context(|| format!("scale {i}"))?;
    }
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
        ensure!(expected_row_bytes > 0, "committed row width must be positive");
        let total = rows
            .len()
            .checked_mul(expected_row_bytes)
            .ok_or_else(|| anyhow::anyhow!("committed-row length overflow"))?;
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
        assert!(open_prequant(&[1i8; 8], &[0x7F80], 1, 8, 8).is_err()); // non-finite scale
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

    /// A pseudo-random prequant row: full-range int8 values, scales in `[2^-7, 2)`.
    fn seeded_row(seed: u64, k: usize) -> (Vec<i8>, Vec<u16>) {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let ints = (0..k).map(|_| next() as i8).collect();
        let scales = (0..k / BLOCK_SIZE).map(|_| 0x3C00 + (next() % 0x400) as u16).collect();
        (ints, scales)
    }

    /// L2 norm computation fixtures
    #[test]
    fn exact_norms_l2_matches_the_zk_certificate() {
        // (seed, k, l2); the f32 formula gave 0x4228 on both rows.
        for (seed, k, l2) in [(877_130, 1024, 0x4224), (110_113, 2048, 0x422c)] {
            let (ints, scales) = seeded_row(seed, k);
            assert_eq!(exact_norms(&ints, &scales, 1, k, BLOCK_SIZE).unwrap()[0].0, l2, "seed {seed}");
        }
        // Every element 2^60: the f32 sum of squares overflowed and rejected the row.
        let k = 1024;
        let norms = exact_norms(&vec![1; k], &vec![(127 + 60) << 7; k / BLOCK_SIZE], 1, k, BLOCK_SIZE).unwrap();
        assert_eq!(norms[0].0, 0x5d80);
    }

    #[test]
    fn exact_norms_rejects_non_finite_scales() {
        let err = exact_norms(&[1i8; 8], &[0x7F80], 1, 8, 8).unwrap_err();
        assert!(format!("{err:#}").contains("not finite"), "{err:#}");
    }

    #[test]
    fn l2_frame_width_pins_the_protocol_formula() {
        assert_eq!(l2_frame_width(1024).unwrap(), 17);
        assert_eq!(l2_frame_width(2080).unwrap(), 15); // ceil(log2 2080) = 12
        assert_eq!(l2_frame_width(1 << 16).unwrap(), 11);
        assert_eq!(l2_frame_width(1 << 27).unwrap(), 0);
        assert!(l2_frame_width((1 << 27) + 1).is_err());
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
