use anyhow::{Result, ensure};

use crate::api::fp8::dtype::{check_not_nan_or_inf_bf16, check_not_nan_or_inf_f32};
use crate::api::layout::JACKPOT_ENTRIES;

const FP32_MIN_NONZERO_EXPONENT: i32 = -126;

/// Products per hardware accumulation group: the MMA's group sum consumes the
/// running accumulator plus this many fresh products (32 + 1 slots) before
/// renormalizing.
pub const MMA_GROUP_PRODUCTS: usize = 32;

/// Internal significand width of the B200 FP8 (`kind::f8f6f4`) accumulation
/// window: 25 fractional bits below the anchor, result rounded to FP32
/// towards zero (B200.md).
const B200_FP8_WIDTH: u16 = 26;

pub trait Dtype<F, T> {
    fn mul(&self, a: F, b: F) -> T;

    fn add(&self, a: F, b: F) -> T;
}

/// Intermediate fixed-point view of an f32 used by the matmul emulation.
/// `exponent` is a signed exponent so a "zero" term can carry a sentinel exponent
/// (`zero_exp`, chosen well below any real exponent) without the alignment math
/// underflowing. A `GFloat` is canonically zero iff `significand == 0`.
pub struct GFloat {
    pub sign: bool,
    pub exponent: i32,
    pub significand: u32,
}

impl From<f32> for GFloat {
    fn from(value: f32) -> Self {
        // NaN and ±inf are banned throughout the FP8 module.
        check_not_nan_or_inf_f32(value).expect("GFloat input must be a finite f32");
        let a_bits = value.to_bits();
        let sign = (a_bits >> 31) & 1 == 1;
        let exponent = ((a_bits >> 23) & 0xFF) as i32;
        let significand = a_bits & 0x7FFFFF;

        if exponent == 0 && significand == 0 {
            // Exact zero.
            return GFloat {
                sign,
                exponent: -133,
                significand: 0,
            };
        }
        // Subnormals (exponent field 0) keep their denormalized significand —
        // no implicit bit — at the scale of exponent field 1, the same
        // `exponent = 1 - bias` rule the fp8/bf16 product decoders apply to
        // subnormal operands.
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
    fn from(val: GFloat) -> f32 {
        // A zero-significand `GFloat` decodes to a signed zero regardless of the
        // (sentinel) exponent it carries.
        if val.significand == 0 {
            return f32::from_bits((val.sign as u32) << 31);
        }

        let mut exponent = val.exponent + 127;
        if (val.significand & 0x800000) == 0 {
            exponent -= 1;
        }
        let exp_bits = u32::try_from(exponent).expect("GFloat exponent below f32 range");
        let value_u32 = (val.sign as u32) << 31 | (exp_bits & 0xFF) << 23 | (val.significand & 0x7FFFFF);

        let value = f32::from_bits(value_u32);
        // NaN and ±inf are banned throughout the FP8 module.
        check_not_nan_or_inf_f32(value).expect("GFloat must decode to a finite f32");
        value
    }
}

fn multiply_fp8_to_gfloat(a: u8, b: u8, zero_exp: i32) -> GFloat {
    let sign_a = (a >> 7) & 1 == 1;
    let mut exp_a = (a >> 3) & 0x0F;
    let sig_a = a & 0x07;

    let sign_b = (b >> 7) & 1 == 1;
    let mut exp_b = (b >> 3) & 0x0F;
    let sig_b = b & 0x07;

    let significand_a = if exp_a != 0 { sig_a | 0x08 } else { sig_a };
    let significand_b = if exp_b != 0 { sig_b | 0x08 } else { sig_b };
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
            exponent: zero_exp,
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

/// One hardware accumulation group at the given window width
/// (`significand_width`; B200 FP8: 26). Aligns every
/// non-zero term to the group's largest stored exponent, truncates each
/// towards zero to the window's fractional bits, sums exactly, and
/// renormalizes the result to `significand_width` bits.
fn windowed_group_sum(values: &[GFloat], zero_exp: i32, significand_width: u16) -> GFloat {
    let base_sig = 24;
    let is_neg_shift = significand_width < base_sig;
    let diff = significand_width.abs_diff(base_sig) as u32;
    let significand_width = significand_width as i32;
    // A zero result carries the SIGN computed from the accumulator, not a
    // hardcoded positive zero: the reference simulator returns
    // `Gfloat(sign, ZERO_EXP, 0)` with `sign = significand < 0`, so a
    // negative accumulation that underflows to a zero significand must stay
    // a `-0`. This matters bit-for-bit downstream — `xor_fold_extract`
    // hashes `f32::to_bits()`, and `-0.0` (`0x80000000`) and `+0.0` differ.
    let signed_zero = |sign: bool| GFloat {
        sign,
        exponent: zero_exp,
        significand: 0,
    };

    // Align every non-zero term to the largest exponent, then sum. Zero terms
    // (significand == 0) contribute nothing and carry the `zero_exp` sentinel,
    // so they neither set the alignment exponent nor add to the sum.
    let Some(max_exponent) = values.iter().filter(|g| g.significand != 0).map(|g| g.exponent).max() else {
        return signed_zero(false); // every term is zero => +0 (as in the reference)
    };
    let mut acc: i64 = 0;
    for v in values {
        if v.significand == 0 {
            continue;
        }
        let shift = max_exponent - v.exponent; // >= 0, since max_exponent is the max
        if shift >= 32 {
            continue; // fully shifted out
        }
        let shifted = if is_neg_shift {
            v.significand >> diff
        } else {
            v.significand << diff
        };
        let aligned = (shifted >> shift) as i64;
        acc += if v.sign { -aligned } else { aligned };
    }

    let sign = acc < 0;
    let mut significand = acc.unsigned_abs() as u32;
    if significand == 0 {
        return signed_zero(sign);
    }

    let width = 32 - significand.leading_zeros() as i32;
    let mut exponent = max_exponent + width - significand_width;
    if width > significand_width {
        significand >>= (width - significand_width) as u32;
    } else {
        significand <<= (significand_width - width) as u32;
    }
    if exponent < FP32_MIN_NONZERO_EXPONENT {
        let shift = FP32_MIN_NONZERO_EXPONENT - exponent;
        if shift >= 32 {
            significand = 0;
        } else {
            significand >>= shift
        };
        exponent = FP32_MIN_NONZERO_EXPONENT;
    }
    significand = if is_neg_shift {
        significand << diff
    } else {
        significand >> diff
    };
    if significand == 0 {
        // Underflowed to zero after the final (de)normalization shift: keep
        // the computed sign so a negative sum that vanishes stays `-0`.
        return signed_zero(sign);
    }
    GFloat {
        sign,
        exponent,
        significand,
    }
}

/// The FP8 matmul datapath: exact e4m3 products consumed in ascending
/// chained accumulation groups of [`MMA_GROUP_PRODUCTS`] products plus the
/// carry, each group collapsed by [`windowed_group_sum`] at the given
/// window width. When `record_partials` is set, each cell's running FP32
/// accumulator after every group collapse (`ceil(k / 32)` values, the last
/// equal to the cell's final value) is returned alongside the output.
#[allow(clippy::too_many_arguments)]
fn matmul_fp8_windowed(
    significand_width: u16,
    a: &[u8],
    b: &[u8],
    acc: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
    record_partials: bool,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    ensure!(a.len() == m * k);
    ensure!(b.len() == n * k);
    if let Some(acc) = acc {
        ensure!(acc.len() == m * n, "accumulator must have length m * n");
    }

    let zero_exp: i32 = -139;
    // The hardware collapses the accumulator every MMA_GROUP_PRODUCTS
    // products: each group_sum sees the carry plus 32 fresh products.
    let group_size = MMA_GROUP_PRODUCTS + 1;

    let mut out: Vec<f32> = vec![0.0; m * n];
    let mut partials: Vec<Vec<f32>> = Vec::with_capacity(if record_partials { m * n } else { 0 });
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_col = &b[j * k..(j + 1) * k];
            // Seed the cell's accumulator with the carry-in (0 when absent).
            let initial = acc.map_or(0.0, |acc| acc[i * n + j]);
            // NaN and ±inf are banned throughout the FP8 module.
            check_not_nan_or_inf_f32(initial)?;
            let mut cell_partials: Vec<f32> =
                Vec::with_capacity(if record_partials { k.div_ceil(MMA_GROUP_PRODUCTS) } else { 0 });
            let mut accumulator = vec![GFloat::from(initial)];
            for (a_val, b_val) in a_row.iter().zip(b_col) {
                accumulator.push(multiply_fp8_to_gfloat(*a_val, *b_val, zero_exp));
                if accumulator.len() == group_size {
                    let collapsed: f32 = windowed_group_sum(&accumulator, zero_exp, significand_width).into();
                    if record_partials {
                        cell_partials.push(collapsed);
                    }
                    accumulator = vec![GFloat::from(collapsed)];
                }
            }
            // The final (possibly partial) group; when k is a multiple of the
            // group size the accumulator holds a single already-collapsed
            // value and this renormalization is the identity.
            let final_value: f32 = windowed_group_sum(&accumulator, zero_exp, significand_width).into();
            if record_partials && accumulator.len() > 1 {
                cell_partials.push(final_value);
            }
            out[i * n + j] = final_value;
            if record_partials {
                partials.push(cell_partials);
            }
        }
    }

    Ok((out, partials))
}

/// f32 -> BF16 cast, round to nearest, ties to even: the COMPUTE-path operand
/// conversion.
pub fn fp32_to_bf16_rne(a: f32) -> u16 {
    let bits = a.to_bits();

    // Round to nearest, ties to even.
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    let bf16 = ((bits + rounding_bias) >> 16) as u16;
    // NaN and ±inf are banned throughout the FP8 module.
    check_not_nan_or_inf_bf16(bf16).expect("fp32_to_bf16 must produce a finite bf16");
    bf16
}

/// NVIDIA B200 (Blackwell) `tcgen05.mma` device, per the measured
/// characteristics in `B200.md` (`B200_fp8_simulator`, `kind::f8f6f4` atoms):
/// an FP32 carry plus 32 exact, un-normalized products per group, groups
/// chained ascending with the (zero-padded) remainder last, the window
/// anchored at the group's max stored exponent (carry included), per-term
/// truncation towards zero; 25 fractional bits are kept below the anchor and
/// the group result is rounded to full FP32 towards zero. Cross-checked
/// bit-for-bit against the B200.md Python simulator
/// (`b200_matmul_fp8_matches_python_reference_vectors`).
pub struct B200 {}

impl B200 {
    /// The bit-exact FP8 MMA. `a` is `m × k` and `b` is `n × k` (both
    /// row-major; `b` holds the operand transposed, so output cell `(i, j)` is
    /// the dot product of row `i` of `a` with row `j` of `b`). `acc`, when
    /// present, is the `m × n` carry-in accumulator: cell `(i, j)` is seeded
    /// with `acc[i * n + j]` before the tile's products are summed, so
    /// successive tiles accumulate on the hardware.
    pub fn matmul_fp8(&self, a: &[u8], b: &[u8], acc: Option<&[f32]>, m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        Ok(matmul_fp8_windowed(B200_FP8_WIDTH, a, b, acc, m, n, k, false)?.0)
    }

    /// Like [`B200::matmul_fp8`] (no carry-in) but returns, for each output
    /// cell, the running FP32 accumulator value after every accumulation group
    /// — `ceil(k / MMA_GROUP_PRODUCTS)` values per cell, the last being the
    /// cell's final value. These are the partial sums `c_v` the jackpot
    /// policy's prefix-inclusive anchor consumes.
    pub fn matmul_fp8_partials(&self, a: &[u8], b: &[u8], m: usize, n: usize, k: usize) -> Result<Vec<Vec<f32>>> {
        Ok(matmul_fp8_windowed(B200_FP8_WIDTH, a, b, None, m, n, k, true)?.1)
    }
}

/// Reference lottery epilogue (`XorFoldExtractor` in the reference miner): fold
/// each of the tile's 16 committed subtiles into one lane. The lane layout
/// (which tile element feeds which lane, and in what order) is committed in
/// advance via the config's grid patterns — `lane_indices[j]` lists lane `j`'s
/// flat indices into the row-major tile in fold order (the committed
/// [`lane_assignment`](crate::api::layout::lane_assignment), 8.8). Each lane
/// folds its subtile's `matmul_dtype` (f32) words — reinterpreted as their
/// same-width unsigned integers — as
/// `lane = rotl32(lane * 0x9E3779B1 + word, 13)`, and the lanes serialize
/// little-endian into 64 bytes: exactly one BLAKE3 block for the lottery hash
/// `blake3(extracted, key=pow_key)`.
pub fn xor_fold_extract(c_tile: &[f32], lane_indices: &[Vec<usize>]) -> [u8; 4 * JACKPOT_ENTRIES] {
    assert_eq!(lane_indices.len(), JACKPOT_ENTRIES, "expected {JACKPOT_ENTRIES} lanes");
    let tile_elems: usize = lane_indices.iter().map(Vec::len).sum();
    assert_eq!(c_tile.len(), tile_elems, "tile shape does not match the committed layout");
    let mut lanes = [0u32; JACKPOT_ENTRIES];
    for (lane, idxs) in lanes.iter_mut().zip(lane_indices) {
        for &i in idxs {
            *lane = lane
                .wrapping_mul(0x9E3779B1)
                .wrapping_add(c_tile[i].to_bits())
                .rotate_left(13);
        }
    }
    let mut out = [0u8; 4 * JACKPOT_ENTRIES];
    for (i, lane) in lanes.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&lane.to_le_bytes());
    }
    out
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

    /// Cross-implementation test vector: the reference `XorFoldExtractor`
    /// (each committed subtile's f32 words folded into its own rotl-mixed lane
    /// per `lane_assignment`, lanes serialized LE) produces exactly these 64
    /// bytes for this input. Layout: both axes use subtile [0, 1] on grid
    /// [0, 2, 4, 6] — an 8x8 merged tile, 16 lanes of 4 words each.
    #[test]
    #[allow(clippy::approx_constant)]
    fn xor_fold_extract_matches_python_reference_vector() {
        use crate::api::layout::{
            AxisPattern,
            DimType::{Blake, Fold},
            lane_assignment,
        };

        // Fold [0, 1] under Blake [0, 2, 4, 6]: the old subtile/grid pair.
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

    // Regression tests for the `zero_exp` handling in the matmul emulation. The
    // former sentinel scheme overflowed the group sum's exponent math (a `u16`
    // subtraction) once the matmul was actually driven; these exercise that exact
    // path and assert it stays finite, deterministic, and zero-preserving.

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

    /// The accumulator groups align to MMA_GROUP_PRODUCTS (32) products: one
    /// matmul over the full contraction dim must equal chaining the carry-in
    /// accumulator at every 32-column boundary — the hardware's group sum
    /// collapses exactly there (32 products + the accumulator per group).
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

    /// The partial sums are the running accumulator of the SAME bit-exact
    /// datapath: per cell there is one partial per 32-product group, each
    /// equal to the carry-in chained matmul over that prefix, and the last
    /// partial equals the cell's final value. A trailing partial group (k not
    /// a multiple of 32) contributes one more partial.
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

    /// A single non-zero term whose exponent is so low that the value underflows
    /// to a zero significand keeps the accumulator's SIGN: a negative term must
    /// decode to `-0.0`, not `+0.0`. This mirrors the reference simulator's
    /// `Gfloat(sign, ZERO_EXP, 0)` and matters because `xor_fold_extract` hashes
    /// the raw f32 bits, where `-0.0` (`0x80000000`) and `+0.0` differ.
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

    /// Subnormal f32s (legal since the module-wide subnormal ban was lifted)
    /// must convert at their true scale: IEEE-754 exponent field 0 means
    /// `2^(1 - bias)` with no implicit bit — the same rule the fp8/bf16
    /// product decoders apply to subnormal operands (the reference
    /// simulator's "subnormal exponent = 1 - bias"). An off-by-one here
    /// halves every subnormal carry inside the accumulation window.
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

    /// A subnormal FP32 carry-in renormalized through the FP8 accumulation
    /// window (an all-zero group leaves the carry as the only live term):
    /// the 26-bit window keeps subnormal carries like `2^-127` and `2^-140`
    /// exactly.
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

    /// One cross-implementation test case for the B200 FP8 matmul: FP8 E4M3
    /// operand bytes (`a` is m x k, `b` is n x k, row-major, b transposed),
    /// an optional FP32 accumulator and the expected output, both as u32 bit
    /// patterns (bit-exactness is the whole point).
    struct B200Vector {
        m: usize,
        n: usize,
        k: usize,
        a: &'static [u8],
        b: &'static [u8],
        c: Option<&'static [u32]>,
        expected: &'static [u32],
    }

    // Generated (one-off, seed 0xB200) by running the pure-Python
    // `B200_fp8_simulator` from B200.md on the inputs below. Do not edit by hand.
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
        // window truncates each tiny product towards zero: C = 1024.0 anchors
        // the window at 2^10 (bottom 2^-15), so every +-(2^-18) product
        // truncates towards zero INDIVIDUALLY (floor would pull the negative
        // ones to -2^-15) and C passes through exactly.
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

    /// Bit-exact cross-check of `B200::matmul_fp8` against the pure-Python
    /// `B200_fp8_simulator` of B200.md (itself verified against B200 silicon
    /// via the C++ extension). The vectors cover single/chained/remainder
    /// atoms, FP32 accumulators, subnormal operands, per-summand window
    /// truncation, accumulator passthrough, and exact cancellation.
    #[test]
    fn b200_matmul_fp8_matches_python_reference_vectors() {
        let hw = B200 {};
        for (idx, v) in B200_FP8_VECTORS.iter().enumerate() {
            let c: Option<Vec<f32>> = v.c.map(|bits| bits.iter().map(|&b| f32::from_bits(b)).collect());
            let out = hw.matmul_fp8(v.a, v.b, c.as_deref(), v.m, v.n, v.k).unwrap();
            let got: Vec<u32> = out.iter().map(|x| x.to_bits()).collect();
            assert_eq!(got, v.expected, "vector {idx} diverged from the B200.md simulator");
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
