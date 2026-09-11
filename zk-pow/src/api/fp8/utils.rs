//! Software replay of B200 FP8 accumulation and the lottery message fold.
//!
//! Matrix multiplication stores A as `m x k` and B as `n x k`, both row-major:
//! output cell `(i, j)` is the dot product of row `i` of A and row `j` of B.
//! Each accumulation group combines 32 exact FP8 products with the previous
//! f32 sum. Terms align to the largest exponent in a window with 25 fractional
//! bits; alignment and the final f32 conversion truncate toward zero.
//!
//! The lottery fold distributes the output cells among 16 lanes using
//! [`lane_assignment`](crate::api::layout::lane_assignment). Each lane starts at
//! zero and consumes its cells' raw f32 words in order:
//!
//! ```text
//! lane = rotl32((lane * 0x9E3779B1 + word) mod 2^32, 13)
//! ```
//!
//! The multiplier and 13-bit rotation are fixed by the protocol. The final
//! lanes form the 64-byte little-endian message hashed by the jackpot check.

use anyhow::{Result, ensure};

use crate::api::fp8::dtype::{check_not_nan_or_inf_bf16, check_not_nan_or_inf_f32};
use crate::api::layout::JACKPOT_ENTRIES;

const FP32_MIN_NONZERO_EXPONENT: i32 = -126;

/// Fresh products per group; the running accumulator occupies one additional slot.
pub const MMA_GROUP_PRODUCTS: usize = 32;

/// Accumulation window: one bit at the largest stored exponent and 25 bits below it.
const B200_FP8_WIDTH: u16 = 26;

pub trait Dtype<F, T> {
    fn mul(&self, a: F, b: F) -> T;

    fn add(&self, a: F, b: F) -> T;
}

/// Matmul intermediate: `(-1)^sign * significand * 2^(exponent - 23)`.
/// The signed exponent permits a sentinel for zero terms, which are identified
/// by `significand == 0`.
pub struct GFloat {
    pub sign: bool,
    pub exponent: i32,
    pub significand: u32,
}

impl From<f32> for GFloat {
    fn from(value: f32) -> Self {
        check_not_nan_or_inf_f32(value).expect("GFloat input must be a finite f32");
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1 == 1;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let significand = bits & 0x7FFFFF;

        if exponent == 0 && significand == 0 {
            return GFloat {
                sign,
                exponent: -133,
                significand: 0,
            };
        }
        // Subnormals use exponent `1 - bias` and have no implicit leading bit.
        let (exponent, significand) = if exponent != 0 {
            (exponent, 0x800000 | significand)
        } else {
            (1, significand)
        };

        GFloat {
            sign,
            exponent: exponent - 127,
            significand,
        }
    }
}

impl From<GFloat> for f32 {
    fn from(components: GFloat) -> f32 {
        // A zero significand preserves its sign regardless of the sentinel exponent.
        if components.significand == 0 {
            return f32::from_bits((components.sign as u32) << 31);
        }

        let mut exponent = components.exponent + 127;
        if (components.significand & 0x800000) == 0 {
            exponent -= 1;
        }
        let biased_exponent = u32::try_from(exponent).expect("GFloat exponent below f32 range");
        let bits =
            (components.sign as u32) << 31 | (biased_exponent & 0xFF) << 23 | (components.significand & 0x7FFFFF);

        let value = f32::from_bits(bits);
        check_not_nan_or_inf_f32(value).expect("GFloat must decode to a finite f32");
        value
    }
}

fn multiply_fp8_to_gfloat(a: u8, b: u8, zero_exponent: i32) -> GFloat {
    let sign_a = (a >> 7) & 1 == 1;
    let mut exp_a = (a >> 3) & 0x0F;
    let fraction_a = a & 0x07;

    let sign_b = (b >> 7) & 1 == 1;
    let mut exp_b = (b >> 3) & 0x0F;
    let fraction_b = b & 0x07;

    let significand_a = if exp_a != 0 { fraction_a | 0x08 } else { fraction_a };
    let significand_b = if exp_b != 0 { fraction_b | 0x08 } else { fraction_b };
    if exp_a == 0 {
        exp_a = 1;
    }

    if exp_b == 0 {
        exp_b = 1;
    }

    let result_significand = (significand_a as u32 * significand_b as u32) << 17;
    if result_significand == 0 {
        GFloat {
            sign: sign_a ^ sign_b,
            exponent: zero_exponent,
            significand: 0,
        }
    } else {
        GFloat {
            sign: sign_a ^ sign_b,
            exponent: exp_a as i32 + exp_b as i32 - 14,
            significand: result_significand,
        }
    }
}

/// Sum one group in a fixed-width accumulation window, then truncate to f32 precision.
/// `significand_width` includes the bit at the alignment exponent; B200 uses 26.
fn windowed_group_sum(values: &[GFloat], zero_exponent: i32, significand_width: u16) -> GFloat {
    let fp32_significand_bits = 24;
    let narrow_window = significand_width < fp32_significand_bits;
    let precision_shift = significand_width.abs_diff(fp32_significand_bits) as u32;
    let significand_width = significand_width as i32;
    // Preserve the sign when a nonzero sum underflows: `xor_fold_extract`
    // uses raw f32 bits, which distinguish +0 and -0.
    let signed_zero = |sign: bool| GFloat {
        sign,
        exponent: zero_exponent,
        significand: 0,
    };

    // Zero terms do not participate in choosing the alignment exponent.
    let Some(max_exponent) = values.iter().filter(|g| g.significand != 0).map(|g| g.exponent).max() else {
        return signed_zero(false); // every term is zero => +0
    };
    // Align and truncate each magnitude before adding the signed terms exactly.
    let mut signed_sum: i64 = 0;
    for term in values {
        if term.significand == 0 {
            continue;
        }
        let shift = max_exponent - term.exponent;
        if shift >= 32 {
            continue; // fully shifted out
        }
        let shifted = if narrow_window {
            term.significand >> precision_shift
        } else {
            term.significand << precision_shift
        };
        let aligned = (shifted >> shift) as i64;
        signed_sum += if term.sign { -aligned } else { aligned };
    }

    let sign = signed_sum < 0;
    let mut significand = signed_sum.unsigned_abs() as u32;
    if significand == 0 {
        return signed_zero(sign);
    }

    // Renormalize the sum to the window width, discarding any low overflow bits.
    let sum_bit_length = 32 - significand.leading_zeros() as i32;
    let mut exponent = max_exponent + sum_bit_length - significand_width;
    if sum_bit_length > significand_width {
        significand >>= (sum_bit_length - significand_width) as u32;
    } else {
        significand <<= (significand_width - sum_bit_length) as u32;
    }
    // Shift subnormal results to f32's minimum exponent, -126.
    if exponent < FP32_MIN_NONZERO_EXPONENT {
        let shift = FP32_MIN_NONZERO_EXPONENT - exponent;
        if shift >= 32 {
            significand = 0;
        } else {
            significand >>= shift
        };
        exponent = FP32_MIN_NONZERO_EXPONENT;
    }
    // Convert the window to f32's 24-bit significand, truncating any extra precision.
    significand = if narrow_window {
        significand << precision_shift
    } else {
        significand >> precision_shift
    };
    if significand == 0 {
        return signed_zero(sign);
    }
    GFloat {
        sign,
        exponent,
        significand,
    }
}

/// Consume exact E4M3 products in groups of [`MMA_GROUP_PRODUCTS`] plus
/// the carry, using [`windowed_group_sum`]. With `record_partials`, also
/// return each cell's accumulator after every group, including a remainder.
#[allow(clippy::too_many_arguments)]
fn matmul_fp8_windowed(
    significand_width: u16,
    a: &[u8],
    b: &[u8],
    carry_in: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
    record_partials: bool,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    ensure!(a.len() == m * k);
    ensure!(b.len() == n * k);
    if let Some(carry_in) = carry_in {
        ensure!(carry_in.len() == m * n, "accumulator must have length m * n");
    }

    let zero_exponent: i32 = -139;
    let group_size = MMA_GROUP_PRODUCTS + 1;

    let mut output: Vec<f32> = vec![0.0; m * n];
    let mut partials: Vec<Vec<f32>> = Vec::with_capacity(if record_partials { m * n } else { 0 });
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_col = &b[j * k..(j + 1) * k];
            let initial = carry_in.map_or(0.0, |carry_in| carry_in[i * n + j]);
            check_not_nan_or_inf_f32(initial)?;
            let mut cell_partials: Vec<f32> =
                Vec::with_capacity(if record_partials { k.div_ceil(MMA_GROUP_PRODUCTS) } else { 0 });
            let mut group_terms = vec![GFloat::from(initial)];
            for (a_val, b_val) in a_row.iter().zip(b_col) {
                group_terms.push(multiply_fp8_to_gfloat(*a_val, *b_val, zero_exponent));
                if group_terms.len() == group_size {
                    let group_sum: f32 = windowed_group_sum(&group_terms, zero_exponent, significand_width).into();
                    if record_partials {
                        cell_partials.push(group_sum);
                    }
                    group_terms = vec![GFloat::from(group_sum)];
                }
            }
            // Finish a remainder group. When k is a multiple of MMA_GROUP_PRODUCTS,
            // only the previous group result remains.
            let final_value: f32 = windowed_group_sum(&group_terms, zero_exponent, significand_width).into();
            if record_partials && group_terms.len() > 1 {
                cell_partials.push(final_value);
            }
            output[i * n + j] = final_value;
            if record_partials {
                partials.push(cell_partials);
            }
        }
    }

    Ok((output, partials))
}

/// Round an f32 result to nearest BF16, ties to even. Panics if the result is non-finite.
pub fn fp32_to_bf16_rne(a: f32) -> u16 {
    let bits = a.to_bits();

    // Round to nearest, ties to even.
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    let bf16 = ((bits + rounding_bias) >> 16) as u16;
    check_not_nan_or_inf_bf16(bf16).expect("fp32_to_bf16 must produce a finite bf16");
    bf16
}

/// NVIDIA B200 FP8 matmul emulation: a 25-fractional-bit accumulation window
/// with truncation toward zero.
pub struct B200 {}

impl B200 {
    /// Multiply FP8 operands with B200 accumulation, optionally continuing an earlier sum.
    ///
    /// Inputs are row-major: `a` is `m x k`, `b` is `n x k`, and optional `acc`
    /// is `m x n`. Row `j` of `b` holds the mathematical right operand's column `j`.
    /// Each output cell starts from its corresponding `acc` value, or zero.
    pub fn matmul_fp8(&self, a: &[u8], b: &[u8], acc: Option<&[f32]>, m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        Ok(matmul_fp8_windowed(B200_FP8_WIDTH, a, b, acc, m, n, k, false)?.0)
    }

    /// Like [`B200::matmul_fp8`] without carry-in, returning each cell's
    /// running FP32 sum after every group (`ceil(k / MMA_GROUP_PRODUCTS)` values).
    /// The jackpot policy includes these partials in its magnitude bound.
    pub fn matmul_fp8_partials(&self, a: &[u8], b: &[u8], m: usize, n: usize, k: usize) -> Result<Vec<Vec<f32>>> {
        Ok(matmul_fp8_windowed(B200_FP8_WIDTH, a, b, None, m, n, k, true)?.1)
    }
}

/// Fold the tile's f32 words into a 64-byte jackpot message.
/// `lane_indices`, from [`lane_assignment`](crate::api::layout::lane_assignment),
/// specifies the cells and their order within each lane.
pub fn xor_fold_extract(c_tile: &[f32], lane_indices: &[Vec<usize>]) -> [u8; 4 * JACKPOT_ENTRIES] {
    assert_eq!(lane_indices.len(), JACKPOT_ENTRIES, "expected {JACKPOT_ENTRIES} lanes");
    let tile_elements: usize = lane_indices.iter().map(Vec::len).sum();
    assert_eq!(c_tile.len(),
        tile_elements, "tile shape does not match the committed layout");
    let mut lanes = [0u32; JACKPOT_ENTRIES];
    for (lane, cell_indices) in lanes.iter_mut().zip(lane_indices) {
        for &i in cell_indices {
            *lane = lane
                .wrapping_mul(0x9E3779B1)
                .wrapping_add(c_tile[i].to_bits())
                .rotate_left(13);
        }
    }
    let mut message = [0u8; 4 * JACKPOT_ENTRIES];
    for (i, lane) in lanes.iter().enumerate() {
        message[i * 4..(i + 1) * 4].copy_from_slice(&lane.to_le_bytes());
    }
    message
}

pub fn bf16_from_i8s(values: &[i8]) -> Result<Vec<u16>> {
    ensure!(values.len().is_multiple_of(2), "Input length must be even");
    values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| {
            let low = chunk[0] as u8 as u16;
            ensure!(chunk[1] >= 0, "Input prequant scales must be positive");
            let high = chunk[1] as u8 as u16;
            let scale = (high << 8) | low;
            ensure!(scale > 0, "Input prequant scales must be positive");
            Ok(scale)
        })
        .collect::<Result<Vec<u16>>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An 8x8 tile uses 16 lanes of four words, with both axes selecting
    /// subtile [0, 1] on grid [0, 2, 4, 6].
    #[test]
    #[allow(clippy::approx_constant)]
    fn xor_fold_extract_matches_python_reference_vector() {
        use crate::api::layout::{
            AxisPattern,
            DimType::{Blake, Fold},
            lane_assignment,
        };

        let axis = AxisPattern::new(&[(2, Fold), (4, Blake)]).unwrap();
        let lanes = lane_assignment(&axis, &axis);
        assert_eq!(lanes[0], vec![0, 1, 8, 9], "lane 0 folds the top-left 2x2 subtile");

        let base = [1.0f32, -2.5, 0.0, 3.14159, 1e-3, -0.0, 448.0, 2.0, 0.5, -1.0];
        let tile: Vec<f32> = (0..64).map(|i| base[i % base.len()]).collect();
        let expected_hex = "433c17856b70ae6a97e608b6a6578200a657820069d113a6433c17856b70ae6a6b70ae6a97e608b6a657820069d113a669d113a6433c17856b70ae6a97e608b6";
        let expected: Vec<u8> = (0..expected_hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&expected_hex[i..i + 2], 16).unwrap())
            .collect();
        assert_eq!(xor_fold_extract(&tile, &lanes).to_vec(), expected);
    }

    // Zero-product regression: sentinel exponents must not overflow alignment arithmetic.

    #[test]
    fn matmul_fp8_handles_zeros_without_panicking() {
        let hw = B200 {};
        let k = 16; // one TILE_D-deep tile
        // 0x38 = fp8 E4M3 encoding of 1.0; 0x00 = zero. Interleave so every output
        // cell mixes zero and non-zero products (the case that stressed group_sum).
        let a: Vec<u8> = (0..2 * k).map(|i| if i % 3 == 0 { 0x00 } else { 0x38 }).collect();
        let b: Vec<u8> = (0..2 * k).map(|i| if i % 2 == 0 { 0x38 } else { 0x00 }).collect();

        let out1 = hw.matmul_fp8(&a, &b, None, 2, 2, k).unwrap();
        let out2 = hw.matmul_fp8(&a, &b, None, 2, 2, k).unwrap();
        assert_eq!(out1, out2, "matmul must be deterministic");
        assert!(out1.iter().all(|x| x.is_finite()), "results must be finite");

        // Accumulate a second tile via the carry-in accumulator; must not panic.
        let out3 = hw.matmul_fp8(&a, &b, Some(&out1), 2, 2, k).unwrap();
        assert!(out3.iter().all(|x| x.is_finite()));
    }

    /// A full matmul must match carry-chained matmuls split at every 32-product boundary.
    #[test]
    fn matmul_group_boundaries_align_to_32_products() {
        let hw = B200 {};
        let (m, n, k) = (2usize, 2usize, 96usize);

        // Varied codes, including zeros.
        let a: Vec<u8> = (0..m * k).map(|i| [0x00, 0x38, 0xB0, 0x42, 0x2C][i % 5]).collect();
        let b: Vec<u8> = (0..n * k).map(|i| [0x38, 0x00, 0x3C, 0xB4, 0x29][i % 5]).collect();
        let full = hw.matmul_fp8(&a, &b, None, m, n, k).unwrap();
        let mut acc: Option<Vec<f32>> = None;
        for col in (0..k).step_by(MMA_GROUP_PRODUCTS) {
            let a_tile: Vec<u8> = (0..m)
                .flat_map(|i| a[i * k + col..i * k + col + MMA_GROUP_PRODUCTS].to_vec())
                .collect();
            let b_tile: Vec<u8> = (0..n)
                .flat_map(|j| b[j * k + col..j * k + col + MMA_GROUP_PRODUCTS].to_vec())
                .collect();
            acc = Some(
                hw.matmul_fp8(&a_tile, &b_tile, acc.as_deref(), m, n, MMA_GROUP_PRODUCTS)
                    .unwrap(),
            );
        }
        assert_eq!(full, acc.unwrap(), "fp8 grouping must collapse every 32 products");
    }

    /// Each partial must match a carry-chained matmul over that prefix.
    #[test]
    fn matmul_fp8_partials_chain_the_group_accumulator() {
        let (m, n, k) = (2usize, 2usize, 96usize);
        let a: Vec<u8> = (0..m * k).map(|i| [0x00, 0x38, 0xB0, 0x42, 0x2C][i % 5]).collect();
        let b: Vec<u8> = (0..n * k).map(|i| [0x38, 0x00, 0x3C, 0xB4, 0x29][i % 5]).collect();

        let hw = B200 {};
        let full = hw.matmul_fp8(&a, &b, None, m, n, k).unwrap();
        let partials = hw.matmul_fp8_partials(&a, &b, m, n, k).unwrap();
        assert_eq!(partials.len(), m * n);
        for (cell, cell_partials) in partials.iter().enumerate() {
            assert_eq!(cell_partials.len(), k / MMA_GROUP_PRODUCTS, "one partial per group");
            assert_eq!(*cell_partials.last().unwrap(), full[cell], "last partial is the final value");
        }
        // Each partial equals the carry-in chained matmul over its prefix.
        let mut acc: Option<Vec<f32>> = None;
        for (g, col) in (0..k).step_by(MMA_GROUP_PRODUCTS).enumerate() {
            let a_tile: Vec<u8> = (0..m)
                .flat_map(|i| a[i * k + col..i * k + col + MMA_GROUP_PRODUCTS].to_vec())
                .collect();
            let b_tile: Vec<u8> = (0..n)
                .flat_map(|j| b[j * k + col..j * k + col + MMA_GROUP_PRODUCTS].to_vec())
                .collect();
            let out = hw
                .matmul_fp8(&a_tile, &b_tile, acc.as_deref(), m, n, MMA_GROUP_PRODUCTS)
                .unwrap();
            for cell in 0..m * n {
                assert_eq!(partials[cell][g], out[cell], "partial {g} of cell {cell}");
            }
            acc = Some(out);
        }

        // Remainder group: k = 40 -> 2 partials (32 + trailing 8) per cell.
        let short = hw.matmul_fp8_partials(&a[..m * 40], &b[..n * 40], m, n, 40).unwrap();
        assert!(short.iter().all(|p| p.len() == 2), "ceil(40 / 32) = 2 partials per cell");
    }

    #[test]
    fn matmul_fp8_all_zeros_is_zero() {
        let hw = B200 {};
        let k = 16;
        let out = hw.matmul_fp8(&vec![0u8; 2 * k], &vec![0u8; 2 * k], None, 2, 2, k).unwrap();
        assert_eq!(out, vec![0.0f32; 4], "an all-zero tile must produce exact zeros");
    }

    /// An underflowing negative term must retain -0.0 for the raw-bit lottery fold.
    #[test]
    fn group_sum_zero_result_preserves_sign_on_underflow() {
        // exponent -252 (a product of two smallest normals) forces the
        // subnormal-underflow path, which shifts the significand to zero.
        let neg = GFloat {
            sign: true,
            exponent: -252,
            significand: 0x0080_0000,
        };
        let pos = GFloat {
            sign: false,
            exponent: -252,
            significand: 0x0080_0000,
        };
        let neg_out: f32 = windowed_group_sum(&[neg], -133, 26).into();
        let pos_out: f32 = windowed_group_sum(&[pos], -133, 26).into();
        assert_eq!(neg_out.to_bits(), 0x8000_0000, "negative underflow must yield -0.0");
        assert_eq!(pos_out.to_bits(), 0x0000_0000, "positive underflow must yield +0.0");
    }

    /// Subnormals have exponent `1 - bias` without an implicit bit; preserve their scale.
    #[test]
    fn gfloat_roundtrips_subnormal_f32_at_true_scale() {
        // 2^-127 (half the smallest normal): exponent -126, no implicit bit.
        let g = GFloat::from(f32::MIN_POSITIVE / 2.0);
        assert_eq!((g.sign, g.exponent, g.significand), (false, -126, 0x40_0000));
        for bits in [0x0000_0001u32, 0x0040_0000, 0x007F_FFFF, 0x8000_0001, 0x803F_FFFF] {
            let x = f32::from_bits(bits);
            assert_eq!(f32::from(GFloat::from(x)).to_bits(), bits, "round-trip of {bits:#010x}");
        }
    }

    /// A group of zero products must preserve representable subnormal carry-ins.
    #[test]
    fn subnormal_carry_renormalizes_through_the_fp8_window() {
        let (m, n, k) = (1usize, 1usize, 32usize);
        let zeros = vec![0u8; k];
        let carry_2p127 = [f32::from_bits(0x0040_0000)]; // 2^-127
        let carry_2p140 = [f32::from_bits(0x0000_0200)]; // 2^-140
        let out = B200 {}.matmul_fp8(&zeros, &zeros, Some(&carry_2p127), m, n, k).unwrap();
        assert_eq!(out[0].to_bits(), 0x0040_0000, "2^-127 carry survives the 26-bit window");
        let out = B200 {}.matmul_fp8(&zeros, &zeros, Some(&carry_2p140), m, n, k).unwrap();
        assert_eq!(out[0].to_bits(), 0x0000_0200, "2^-140 carry survives the 26-bit window");
    }

    /// B200 reference vector: row-major E4M3 inputs (`a: m x k`, `b: n x k`),
    /// optional FP32 carry-in and expected FP32 output, stored as bit patterns.
    struct B200Vector {
        m: usize,
        n: usize,
        k: usize,
        a: &'static [u8],
        b: &'static [u8],
        c: Option<&'static [u32]>,
        expected: &'static [u32],
    }

    // Expected bit patterns generated with a Python B200 simulator, seed 0xB200.
    const B200_FP8_VECTORS: &[B200Vector] = &[
        // single atom, no C
        B200Vector {
            m: 2,
            n: 2,
            k: 32,
            a: &[
                0xBA, 0xED, 0x6C, 0x19, 0xC7, 0x81, 0x8D, 0x95, 0xD8, 0x14, 0x89, 0x28, 0x00, 0xD0, 0x00, 0x6E, 0xB8, 0x47, 0x9C,
                0x29, 0x70, 0x02, 0x83, 0x82, 0x05, 0x84, 0x85, 0x03, 0xAE, 0x00, 0xA6, 0x12, 0x6D, 0x00, 0x15, 0x6D, 0x8D, 0x11,
                0x06, 0xB8, 0x2F, 0xC3, 0x80, 0x85, 0x50, 0x27, 0x9B, 0xB5, 0x47, 0xB6, 0x85, 0x9B, 0xCC, 0x6B, 0xFD, 0x9E, 0xA1,
                0x30, 0xC5, 0xCE, 0x83, 0x00, 0x65, 0xDB,
            ],
            b: &[
                0x07, 0x80, 0x88, 0x16, 0xD3, 0x07, 0xEA, 0x95, 0xED, 0x80, 0x80, 0x41, 0x00, 0x00, 0xD9, 0x80, 0xCD, 0x00, 0x80,
                0x03, 0xEC, 0x77, 0xF7, 0x32, 0x7E, 0x7B, 0x83, 0x8B, 0xD3, 0xEB, 0x87, 0xFD, 0x63, 0xAB, 0xFC, 0x81, 0x50, 0xC2,
                0x87, 0x88, 0xBD, 0x26, 0x20, 0x00, 0x2E, 0x86, 0x00, 0x41, 0xD4, 0xE8, 0x80, 0x68, 0xB6, 0xB5, 0x39, 0xDD, 0x87,
                0x86, 0x00, 0x2B, 0x29, 0xDA, 0x00, 0x00,
            ],
            c: None,
            expected: &[0xC6255A89, 0xC7105D2B, 0x47FF57AD, 0x457BC025],
        },
        // two atoms + C
        B200Vector {
            m: 2,
            n: 3,
            k: 64,
            a: &[
                0x1C, 0x63, 0x80, 0x36, 0x80, 0x80, 0x00, 0x53, 0x18, 0x64, 0x35, 0x82, 0xF0, 0x3C, 0x38, 0x6B, 0xF7, 0x80, 0xB3,
                0x24, 0xE3, 0xCE, 0x00, 0x03, 0x7B, 0x86, 0x7A, 0x85, 0x67, 0x24, 0xB7, 0x36, 0x7A, 0x80, 0xD4, 0x80, 0x7B, 0x80,
                0x1E, 0xEE, 0xA5, 0x44, 0x56, 0x80, 0x34, 0x00, 0x00, 0xAB, 0x12, 0x54, 0x47, 0x43, 0x90, 0x80, 0x51, 0xE9, 0x7B,
                0x85, 0xB1, 0x00, 0x60, 0x53, 0x81, 0xE9, 0xFA, 0x34, 0x00, 0x24, 0xE8, 0x7A, 0xAA, 0x82, 0xDF, 0x84, 0x8E, 0x60,
                0x00, 0x00, 0x50, 0x2A, 0x95, 0x80, 0x91, 0x9B, 0x3D, 0xCA, 0xA2, 0x5B, 0x85, 0x53, 0x82, 0x00, 0x86, 0xC0, 0xB0,
                0x2A, 0x80, 0x78, 0x49, 0x06, 0x00, 0x3A, 0x00, 0x84, 0x04, 0x00, 0x87, 0xE5, 0x77, 0x02, 0xC0, 0x84, 0x8A, 0x80,
                0xFD, 0x0C, 0x80, 0x9A, 0xB0, 0x00, 0x1C, 0x56, 0x59, 0xAE, 0x45, 0xC2, 0x38, 0xBD,
            ],
            b: &[
                0xAF, 0x5B, 0x18, 0x86, 0xFE, 0xBE, 0x41, 0x85, 0x00, 0x8C, 0x9B, 0x76, 0x45, 0x81, 0x07, 0xE2, 0xD5, 0xE0, 0x04,
                0x05, 0x5C, 0xCF, 0xFD, 0x70, 0x61, 0xA7, 0x02, 0x00, 0xBF, 0xD3, 0x01, 0x51, 0x21, 0x02, 0x7D, 0x33, 0x8D, 0x68,
                0xBC, 0x0C, 0x5E, 0x98, 0xA1, 0xF2, 0x87, 0x0C, 0x80, 0x80, 0x52, 0x71, 0x85, 0x00, 0x00, 0x80, 0xC6, 0xA6, 0x2D,
                0x84, 0x83, 0xE1, 0xA0, 0x99, 0x13, 0x8D, 0x80, 0x00, 0x71, 0x37, 0xD0, 0xA2, 0xD8, 0x80, 0x70, 0x00, 0x46, 0xDC,
                0x07, 0xEA, 0x36, 0xF1, 0x00, 0xF6, 0x38, 0x00, 0xA1, 0xFA, 0xE3, 0xC3, 0x7A, 0x85, 0x25, 0x2D, 0x8E, 0xE1, 0x44,
                0x00, 0x9F, 0x49, 0x0E, 0x00, 0x68, 0x04, 0x0F, 0x80, 0x1F, 0x7B, 0xCD, 0xB5, 0xFA, 0x80, 0xBC, 0x18, 0x4E, 0x93,
                0x80, 0x2A, 0x87, 0xAA, 0xA7, 0xB4, 0x80, 0xD1, 0xB3, 0xAB, 0xD3, 0x00, 0x8A, 0x01, 0x64, 0x2B, 0x87, 0xF0, 0x80,
                0x00, 0xE5, 0xD5, 0x38, 0x80, 0x92, 0x4B, 0xB7, 0x00, 0xBC, 0xC6, 0x86, 0x3F, 0x66, 0xAA, 0x16, 0xD3, 0x08, 0x4D,
                0x9D, 0x3F, 0x00, 0xCD, 0x00, 0xEA, 0x00, 0xA9, 0xE3, 0xC9, 0xCE, 0x22, 0x80, 0x20, 0x9D, 0x00, 0x8B, 0x67, 0xE1,
                0x00, 0x13, 0xC9, 0x02, 0x87, 0xD4, 0xDD, 0x72, 0x00, 0x31, 0x66, 0x95, 0x80, 0x62, 0x01, 0x37, 0x05, 0xF9, 0xF6,
                0x40, 0xFD,
            ],
            c: Some(&[0xBD9FA467, 0x477EB5B4, 0xBDF5F55A, 0xCA497EA4, 0xBFB81301, 0x45FF09EA]),
            expected: &[0x46065E5E, 0x4839D1F5, 0x468BB300, 0xCA4685A2, 0xC79903BF, 0xC790BF52],
        },
        // remainder atom + C
        B200Vector {
            m: 1,
            n: 2,
            k: 40,
            a: &[
                0x80, 0x02, 0xC7, 0x3D, 0xA3, 0xB7, 0x1F, 0x8F, 0xD7, 0xD7, 0x84, 0x3E, 0x0D, 0x81, 0x00, 0x13, 0x05, 0xDD, 0x82,
                0xCF, 0x01, 0x8C, 0x2A, 0xEB, 0x86, 0x00, 0x24, 0x01, 0x55, 0xEC, 0x9F, 0xDE, 0x83, 0x4F, 0x21, 0xDA, 0x27, 0x00,
                0xF6, 0xC0,
            ],
            b: &[
                0x13, 0x04, 0xF6, 0x5C, 0x87, 0x19, 0x01, 0x5F, 0x80, 0x03, 0xFE, 0xD2, 0x87, 0x00, 0x7D, 0x91, 0x80, 0xD2, 0xEB,
                0xFE, 0x5A, 0xE0, 0xF2, 0x8B, 0xE2, 0xE6, 0x32, 0x22, 0x4E, 0xBF, 0x66, 0xD0, 0x73, 0x63, 0x00, 0x5E, 0xF4, 0xA3,
                0xC0, 0xC5, 0x88, 0x84, 0xFA, 0x01, 0x35, 0x31, 0x85, 0xBA, 0x87, 0x80, 0xD8, 0x57, 0x5D, 0xE2, 0x8C, 0x5D, 0x03,
                0x00, 0x01, 0x81, 0x00, 0x85, 0x80, 0x00, 0xDF, 0x05, 0x28, 0x2D, 0xA5, 0xD2, 0xF8, 0x41, 0xB4, 0x3B, 0xD0, 0xF4,
                0xDE, 0xB4, 0x7B, 0xEA,
            ],
            c: Some(&[0x34CE0AA6, 0xC10607AE]),
            expected: &[0x459F83DF, 0xC78DFE95],
        },
        // three atoms, no C
        B200Vector {
            m: 3,
            n: 2,
            k: 96,
            a: &[
                0x35, 0x53, 0x7B, 0x97, 0x0B, 0x7B, 0x99, 0x38, 0x23, 0xD8, 0x5B, 0x86, 0x53, 0x8F, 0x8F, 0x8E, 0x76, 0x50, 0xF3,
                0xC6, 0xC7, 0x08, 0xEE, 0x90, 0x91, 0xF0, 0x8E, 0x5E, 0x5E, 0x51, 0x1D, 0x6F, 0xBF, 0xFB, 0x53, 0x98, 0x69, 0x2C,
                0xFA, 0x02, 0x32, 0x5B, 0x34, 0x13, 0xD0, 0x9C, 0x03, 0xC1, 0x10, 0xE9, 0x1D, 0x1C, 0x09, 0x02, 0x2A, 0xB9, 0xDC,
                0xA2, 0xFE, 0xA0, 0xD9, 0x80, 0x7B, 0xE3, 0x04, 0xFD, 0xF6, 0x5B, 0x23, 0x00, 0x18, 0x83, 0x68, 0x02, 0xC5, 0x72,
                0x43, 0x4A, 0x66, 0x13, 0x34, 0x78, 0xF2, 0xC4, 0xAF, 0x12, 0xEF, 0x0B, 0xBA, 0x4B, 0x1F, 0xC9, 0x83, 0x6F, 0x8F,
                0x33, 0xFA, 0x3F, 0x0E, 0x9B, 0x98, 0xE9, 0x4A, 0xCD, 0x8B, 0xB2, 0x96, 0x84, 0x67, 0xBC, 0x1D, 0xDF, 0xC2, 0xF1,
                0xE0, 0x5A, 0x7C, 0x48, 0xD7, 0x79, 0xD2, 0xD8, 0xC5, 0x99, 0x15, 0x52, 0x76, 0x5F, 0xD0, 0xD0, 0xC2, 0x17, 0x1B,
                0x0A, 0x9E, 0x0B, 0x4C, 0x4D, 0x66, 0x9E, 0xA4, 0x6A, 0x17, 0x67, 0x4B, 0xBE, 0xED, 0x79, 0x48, 0xBC, 0xB1, 0xF1,
                0xDE, 0x6C, 0xF8, 0x54, 0x4C, 0x2F, 0x3D, 0x1D, 0x7E, 0xE9, 0x20, 0x54, 0xDF, 0xAF, 0xF3, 0xD4, 0x38, 0x39, 0x91,
                0x80, 0x47, 0xC2, 0x76, 0x2A, 0xB0, 0x64, 0x77, 0x44, 0x18, 0xA5, 0xBD, 0x6B, 0xA0, 0xD0, 0x27, 0x86, 0x17, 0x75,
                0x10, 0x1F, 0x72, 0x5A, 0x21, 0x8A, 0xB4, 0x7A, 0xC5, 0x47, 0x42, 0x1E, 0x21, 0xBB, 0xD6, 0x00, 0x6E, 0xB8, 0xF1,
                0x20, 0x46, 0xBE, 0xFA, 0x19, 0x83, 0xF9, 0xA2, 0x95, 0x90, 0x94, 0xC0, 0xCD, 0xC7, 0xBD, 0x25, 0xC2, 0xC5, 0xC9,
                0xCE, 0x2F, 0x05, 0xF3, 0x4C, 0x7B, 0xE4, 0x73, 0xB8, 0xB5, 0x84, 0xB7, 0x5B, 0x65, 0xE5, 0xF3, 0x1A, 0x4E, 0xB1,
                0xFA, 0x92, 0xA9, 0x5E, 0x9C, 0x7D, 0x51, 0x14, 0xF1, 0x96, 0x7E, 0xA1, 0x82, 0xD4, 0x5E, 0x21, 0x75, 0x0F, 0xE9,
                0x11, 0x40, 0xC1, 0xF6, 0x8B, 0xBA, 0xEE, 0x73, 0x1C, 0x8B, 0xD2, 0x74, 0xD2, 0x3C, 0x19, 0xC7, 0x81, 0x21, 0xAC,
                0x37, 0x91, 0x22,
            ],
            b: &[
                0xAB, 0xAD, 0xF8, 0x57, 0x44, 0x54, 0x8D, 0x9C, 0xB4, 0xF6, 0x70, 0xC1, 0xBB, 0x19, 0x85, 0x4F, 0x70, 0xE1, 0xBB,
                0x9D, 0x6A, 0xD5, 0xA9, 0x97, 0x56, 0xB9, 0x15, 0x61, 0x50, 0xD2, 0x7B, 0x52, 0x51, 0x6A, 0x95, 0xFE, 0xAF, 0x79,
                0x65, 0xBF, 0x17, 0xA1, 0x70, 0xB4, 0xDA, 0xFC, 0x8F, 0x23, 0x22, 0x98, 0xD3, 0x4C, 0x50, 0xE8, 0xDD, 0x2D, 0x70,
                0x44, 0x1A, 0x7E, 0xD2, 0xA4, 0x09, 0x58, 0x47, 0xC0, 0x00, 0x92, 0xCB, 0xB4, 0x47, 0xC0, 0xB6, 0xF4, 0x05, 0xED,
                0xAF, 0x76, 0xC3, 0xB4, 0x78, 0x4F, 0xB4, 0xB3, 0x8E, 0x77, 0x79, 0x47, 0x2B, 0xC4, 0x56, 0x49, 0xFE, 0x80, 0xE2,
                0x08, 0x60, 0x24, 0xDE, 0x44, 0xA6, 0x16, 0x64, 0xB5, 0x6B, 0x02, 0x92, 0x3F, 0x94, 0xD4, 0xEB, 0xCC, 0xD8, 0x7E,
                0x5D, 0xF0, 0x48, 0x2C, 0x60, 0x92, 0x90, 0x9C, 0xF9, 0x92, 0x4F, 0xFC, 0x7D, 0xAF, 0x95, 0xD6, 0x7C, 0x17, 0x15,
                0xD8, 0x7A, 0x72, 0xA1, 0x87, 0x3B, 0x62, 0x5C, 0x39, 0xC9, 0x02, 0x79, 0x61, 0xAC, 0x95, 0x43, 0xD7, 0x9F, 0xE1,
                0x30, 0x5E, 0x06, 0xC0, 0x21, 0x0F, 0x99, 0xC1, 0x77, 0xFE, 0x7D, 0x1F, 0xEA, 0x6D, 0x85, 0xC2, 0x5A, 0xA4, 0x35,
                0x35, 0x23, 0x52, 0x0E, 0x9F, 0xE9, 0xF1, 0x39, 0x08, 0xB8, 0x58, 0xD6, 0x0B, 0x07, 0xD6, 0xB8, 0xB6, 0x48, 0x1E,
                0xFE, 0xCA,
            ],
            c: None,
            expected: &[0xC80C9E2C, 0xC760B96E, 0x47B7B146, 0x481A5E3F, 0xC791C160, 0xC8553F0B],
        },
        // C = 1024 anchors the window at 2^10. Each ±2^-18 product truncates
        // individually toward zero; C passes through unchanged.
        B200Vector {
            m: 1,
            n: 1,
            k: 32,
            a: &[
                0x01, 0x81, 0x01, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            b: &[
                0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            c: Some(&[0x44800000]),
            expected: &[0x44800000],
        },
        // keeps 25 fractional bits (see the companion window-precision test)
        B200Vector {
            m: 1,
            n: 1,
            k: 32,
            a: &[
                0x38, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            b: &[
                0x38, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            c: None,
            expected: &[0x3F800020],
        },
        // subnormal-only operands
        B200Vector {
            m: 2,
            n: 2,
            k: 32,
            a: &[
                0x82, 0x07, 0x01, 0x83, 0x81, 0x83, 0x82, 0x87, 0x81, 0x82, 0x06, 0x82, 0x82, 0x86, 0x83, 0x07, 0x81, 0x84, 0x06,
                0x04, 0x81, 0x06, 0x84, 0x81, 0x02, 0x05, 0x07, 0x07, 0x85, 0x82, 0x86, 0x07, 0x83, 0x87, 0x04, 0x85, 0x03, 0x03,
                0x02, 0x86, 0x06, 0x06, 0x05, 0x83, 0x02, 0x05, 0x06, 0x05, 0x86, 0x84, 0x83, 0x07, 0x04, 0x83, 0x03, 0x83, 0x01,
                0x85, 0x05, 0x82, 0x03, 0x84, 0x06, 0x82,
            ],
            b: &[
                0x85, 0x84, 0x82, 0x81, 0x07, 0x84, 0x06, 0x86, 0x02, 0x03, 0x81, 0x04, 0x06, 0x07, 0x04, 0x84, 0x83, 0x86, 0x84,
                0x02, 0x83, 0x07, 0x83, 0x07, 0x05, 0x07, 0x87, 0x02, 0x03, 0x85, 0x01, 0x82, 0x06, 0x01, 0x85, 0x02, 0x81, 0x03,
                0x81, 0x02, 0x07, 0x06, 0x81, 0x81, 0x83, 0x01, 0x03, 0x87, 0x04, 0x86, 0x01, 0x05, 0x81, 0x86, 0x01, 0x84, 0x04,
                0x04, 0x82, 0x03, 0x07, 0x87, 0x06, 0x02,
            ],
            c: None,
            expected: &[0xB9500000, 0xB9E40000, 0x3A080000, 0x39D20000],
        },
        // accumulator passthrough: all-zero products return C bit-exactly
        // (odd mantissas untouched; -0.0 comes back as +0.0, matching the
        // simulator, which does not model the sign of exact zeros).
        B200Vector {
            m: 2,
            n: 2,
            k: 32,
            a: &[
                0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00,
                0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
                0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00,
                0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
            ],
            b: &[
                0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
                0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00,
                0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
                0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00,
            ],
            c: Some(&[0x00000000, 0x80000000, 0xC0600001, 0x3FFFFFFF]),
            expected: &[0x00000000, 0x00000000, 0xC0600001, 0x3FFFFFFF],
        },
        // exact cancellation to +0: 12*0.5 + (-12)*0.5
        B200Vector {
            m: 1,
            n: 1,
            k: 32,
            a: &[
                0x44, 0xC4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            b: &[
                0x30, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            c: None,
            expected: &[0x00000000],
        },
    ];

    /// Compare raw output bits against the stored vectors.
    /// Covers full/remainder groups, carry-ins, subnormals, truncation and cancellation.
    #[test]
    fn b200_matmul_fp8_matches_python_reference_vectors() {
        let hw = B200 {};
        for (idx, v) in B200_FP8_VECTORS.iter().enumerate() {
            let c: Option<Vec<f32>> = v.c.map(|bits| bits.iter().map(|&b| f32::from_bits(b)).collect());
            let out = hw.matmul_fp8(v.a, v.b, c.as_deref(), v.m, v.n, v.k).unwrap();
            let got: Vec<u32> = out.iter().map(|x| x.to_bits()).collect();
            assert_eq!(got, v.expected, "vector {idx} differs from the expected output");
        }
    }

    /// The window's precision: anchored at 2^0 by a 1.0 * 1.0 product, a
    /// 2^-9 * 2^-9 = 2^-18 product lies inside the 25 fractional bits and is
    /// kept exactly.
    #[test]
    fn b200_fp8_window_keeps_small_products() {
        let mut a = vec![0u8; 32];
        let mut b = vec![0u8; 32];
        (a[0], a[1]) = (0x38, 0x01); // 1.0, 2^-9 (min subnormal)
        (b[0], b[1]) = (0x38, 0x01);
        let b200 = B200 {}.matmul_fp8(&a, &b, None, 1, 1, 32).unwrap();
        assert_eq!(b200[0].to_bits(), 0x3F800020, "B200 keeps 1 + 2^-18 exactly");
    }
}
