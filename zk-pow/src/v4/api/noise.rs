//! Deterministic FP8 noise for the opened rows of A and columns of B.
//!
//! Each operand's noise is a matrix product `E @ F` with inner dimension `rank`.
//! A noise "line" is a vector of `rank` entries: one row of E or one column of F.
//! Lines are sampled independently from public seeds and indices, so any required
//! line can be regenerated without sampling the rest of the matrix.
//!
//! A CONSTANT norm makes an `E@F` entry (`<e_row, f_col>` of two lines) have a
//! KNOWN peak (`~c^2`, Cauchy-Schwarz) and rms (`~c^2/sqrt(r)`), so the quant
//! scheme derives its per-row scales from `X`'s norms alone, never measuring
//! `E@F`. The exact procedure is chosen purely for cross-host bit-exactness;
//! the draw need not be Gaussian, only deterministic and public.
//!
//! Each line is random-access: the noise-line key is `Subkey("noise-line",
//! seed)`, and each line is addressed by a `(side, factor, line)` tuple
//! zero-padded to one 64-byte BLAKE3 block. `E` lines use the selected global
//! row/column index; `F` lines use `0..k` and never vary by expert.
//!
//! `E_A` is keyed by `noise seedA`. Both F bases are keyed by `noise seedB`
//! (`F_A` uses `Side::A` addresses so it is a distinct draw from `F_B`); `E_B`
//! is keyed by `noise seedB`. The `F` basis is stored `k x r` row-major (the
//! transpose of the reference's `(r x k)` view), so line `i` is column `i` of
//! `F` — exactly the operand layout both device atoms expect for `E @ F`.

use crate::v4::api::compute::{bf16_div, bf16_mul};
use crate::v4::api::dtype::{bf16_to_f32, f32_to_bf16, f32_to_fp8_e4m3};
use crate::v4::api::primitives::{Hash256, IncompleteBlockHeader, Sides};
use crate::v4::api::public_params::PublicParams;
use crate::v4::api::quantization::NOISE_TARGET_NORM;
use crate::v4::api::transcript::{LABEL_NOISE_LINE, subkey};

/// Keep five fractional bits of the norm by computing `floor(32 * sqrt(sum(x_i^2)))`.
const INT_SQRT_PREC: u64 = 32;

/// Stores one [`OperandNoise`] per matrix operand.
/// `a` contains A's E and F factors; `b` contains B's E and F factors.
pub type Noise = Sides<OperandNoise>;

/// Factors for one operand's noise matrix `E @ F`.
/// For `n` selected A rows or B columns, the product has shape `n x k`,
/// where `k` is the matrix multiplication's reduction dimension.
pub struct OperandNoise {
    /// E as `n x rank` FP8 E4M3 bytes, stored row-major in the requested index order.
    pub e: Vec<u8>,
    /// F transposed: `k x rank` FP8 E4M3 bytes, stored row-major.
    /// Each stored row is one column of F, matching B200's matrix-multiplication layout.
    pub f: Vec<u8>,
}

/// Fixed seed-address line bytes, shared with the device-quantization vectors in
/// [`crate::v4::api::quantization::tests`].
#[cfg(test)]
pub(crate) fn decode_hex_for_test(raw: &str) -> Vec<u8> {
    assert!(raw.len().is_multiple_of(2), "hex string must have even length");
    (0..raw.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&raw[i..i + 2], 16).expect("hex must decode"))
        .collect()
}

/// Which operand a noise line belongs to. The discriminants are the committed
/// wire bytes.
#[repr(u8)]
pub(crate) enum Side {
    A = 0,
    B = 1,
}

/// Selects a factor in the operand's low-rank noise product `E @ F`.
/// Each operand has its own E and F matrices.
///
/// `sample_line` writes this choice as the second byte of its hash input:
/// 0 for E, 1 for F.
#[repr(u8)]
pub(crate) enum NoiseFactor {
    E = 0,
    F = 1,
}

/// Sample one row of E or one column of F as `rank` FP8 E4M3 entries.
///
/// Pass A's noise seed with `Side::A`, or B's noise seed with `Side::B`.
/// For E, `line` is the global A-row or B-column index. For F, it is a column index in `0..k`.
///
/// Derive a key by hashing [`LABEL_NOISE_LINE`] under `seed`. Keyed BLAKE3 then
/// expands this 64-byte input into `rank` bytes:
///
/// ```text
/// side (1 byte) | factor (1 byte) | line (4 bytes, little-endian) | 58 zero bytes
/// ```
///
/// Decode and scale those bytes with [`normalize_line`] to obtain the noise samples.
pub(crate) fn sample_line(seed: &Hash256, side: Side, factor: NoiseFactor, line: u32, rank: u16) -> Vec<u8> {
    let key = subkey(LABEL_NOISE_LINE, Some(seed));
    let mut line_address = Vec::with_capacity(64);
    line_address.push(side as u8);
    line_address.push(factor as u8);
    line_address.extend_from_slice(&line.to_le_bytes());
    assert!(line_address.len() <= 64, "noise line material must fit one BLAKE3 block");
    line_address.resize(64, 0);

    let mut bytes = vec![0u8; usize::from(rank)];
    let mut hasher = blake3::Hasher::new_keyed(&key);
    hasher.update(&line_address);
    hasher.finalize_xof().fill(&mut bytes);

    normalize_line(&bytes)
}

/// Convert random bytes to FP8 samples with Euclidean norm near [`NOISE_TARGET_NORM`].
fn normalize_line(bytes: &[u8]) -> Vec<u8> {
    // The high bit is the sign; adding one to the low seven bits gives magnitudes 1..=128, excluding zero.
    let signed_samples: Vec<i64> = bytes
        .iter()
        .map(|&b| {
            let sign = 1 - 2 * ((b >> 7) as i64);
            let magnitude = ((b & 0x7F) as i64) + 1;
            sign * magnitude
        })
        .collect();

    // With fewer than 2^16 samples of magnitude at most 128, the scaled sum of squares stays below 2^40.
    let sum_of_squares: u64 = signed_samples.iter().map(|&sample| (sample * sample) as u64).sum();
    // Scaling by 32 preserves five fractional bits when the integer square root rounds down.
    let scaled_norm = (sum_of_squares * (INT_SQRT_PREC * INT_SQRT_PREC)).isqrt();

    // Apply the same factor to the target norm, then round both operands to BF16 before dividing.
    let scale_numerator = f32_to_bf16((NOISE_TARGET_NORM * INT_SQRT_PREC as f64) as f32).expect("8192 is representable");
    let scale_denominator = f32_to_bf16(scaled_norm as f32).expect("norm_scaled < 2^24 is representable");
    let scale = bf16_div(scale_numerator, scale_denominator).expect("noise-line scale is finite");

    // Preserve the BF16 rounding before the final FP8 rounding; both use nearest, ties to even.
    signed_samples
        .iter()
        .map(|&sample| {
            let sample_bf16 = f32_to_bf16(sample as f32).expect("|x_i| <= 128 is representable in bf16");
            let entry = bf16_mul(sample_bf16, scale).expect("noise entry is finite");
            f32_to_fp8_e4m3(bf16_to_f32(entry)).expect("noise entry is finite")
        })
        .collect()
}

/// Generate E for the requested A rows and B columns, and a full F basis for each operand.
///
/// `a_rows`/`b_cols` are the selected global row/column indices; the `E` lines
/// key off those indices, and the `F` basis is `0..k` for both sides.
/// `rank` is the peel rank `r`. Both F bases are keyed by `seeds.b`.
///
/// Each line is a pure keyed-XOF draw (no shared state), so the four bases are
/// sampled in parallel over lines; the order-preserving parallel iterators keep
/// the byte layout identical to the sequential draw.
pub(crate) fn sample_noise(k: usize, rank: u16, seeds: Sides<Hash256>, a_rows: &[u32], b_cols: &[u32]) -> Noise {
    use plonky2_maybe_rayon::*;

    let e_a: Vec<u8> = a_rows
        .par_iter()
        .flat_map_iter(|&row| sample_line(&seeds.a, Side::A, NoiseFactor::E, row, rank))
        .collect();
    let e_b: Vec<u8> = b_cols
        .par_iter()
        .flat_map_iter(|&col| sample_line(&seeds.b, Side::B, NoiseFactor::E, col, rank))
        .collect();
    let f_a: Vec<u8> = (0..k as u32)
        .into_par_iter()
        .flat_map_iter(|i| sample_line(&seeds.b, Side::A, NoiseFactor::F, i, rank))
        .collect();
    let f_b: Vec<u8> = (0..k as u32)
        .into_par_iter()
        .flat_map_iter(|i| sample_line(&seeds.b, Side::B, NoiseFactor::F, i, rank))
        .collect();

    Noise {
        a: OperandNoise { e: e_a, f: f_a },
        b: OperandNoise { e: e_b, f: f_b },
    }
}

/// Rebuild the noise factors from the public statement and proposed header.
///
/// The `E` lines key off the selected global row/column indices (in MoE they are
/// the winner's `i_a` outer rows and the `expert_col + base + i` columns, so the
/// global address itself gives per-expert pair-uniqueness); both `F` bases are
/// `0..k` lines keyed by seedB.
pub fn compute_fp8_noise(params: &PublicParams, proposed_header: &IncompleteBlockHeader) -> Noise {
    let a_rows = params.a_rows_indices();
    let b_cols = params.b_rows_indices();
    let k = params.common_dim() as usize;
    let rank = params.rank() as u16;
    let seeds = params.noise_seeds(proposed_header);
    sample_noise(k, rank, seeds, &a_rows, &b_cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4::api::dtype::fp8_e4m3_to_f32;
    use crate::v4::api::primitives::Sides;

    use super::decode_hex_for_test as decode_hex;

    fn fixed_seeds() -> Sides<Hash256> {
        Sides {
            a: [0x22u8; 32],
            b: [0x11u8; 32],
        }
    }

    #[test]
    fn noise_factor_discriminants() {
        assert_eq!(Side::A as u8, 0);
        assert_eq!(Side::B as u8, 1);
        assert_eq!(NoiseFactor::E as u8, 0);
        assert_eq!(NoiseFactor::F as u8, 1);
    }

    #[test]
    fn noise_shapes_and_determinism() {
        let seeds = fixed_seeds();
        let (k, r) = (128, 16u16);
        let noise = sample_noise(k, r, seeds, &[0, 8, 64], &[1, 2]);
        assert_eq!(noise.a.e.len(), 3 * r as usize);
        assert_eq!(noise.b.e.len(), 2 * r as usize);
        assert_eq!(noise.a.f.len(), k * r as usize);
        assert_eq!(noise.b.f.len(), k * r as usize);

        let again = sample_noise(k, r, seeds, &[0, 8, 64], &[1, 2]);
        assert_eq!(noise.a.e, again.a.e);
        assert_eq!(noise.b.f, again.b.f);
    }

    #[test]
    fn noise_line_is_normalized_to_target_norm() {
        let seeds = Sides {
            a: [4u8; 32],
            b: [3u8; 32],
        };
        let r = 64u16;
        let noise = sample_noise(64, r, seeds, &[7], &[]);
        let sq_norm: f32 = noise.a.e.iter().map(|&c| fp8_e4m3_to_f32(c).powi(2)).sum();
        let norm = sq_norm.sqrt();
        assert!(
            (norm - NOISE_TARGET_NORM as f32).abs() < 0.1 * NOISE_TARGET_NORM as f32,
            "line L2 norm {norm} should be near {NOISE_TARGET_NORM}"
        );
        assert!(noise.a.e.iter().all(|&c| fp8_e4m3_to_f32(c) != 0.0), "noise never draws zero");
    }

    // the E lines track the selected global rows/columns,
    // the shared F bases never do, and the seeds feed E_X (seedX)
    // and both F bases (seedB).
    #[test]
    fn noise_lines_are_keyed_by_address_side_and_seed() {
        let base = Sides {
            a: [9u8; 32],
            b: [7u8; 32],
        };
        let a_changed = Sides {
            a: [8u8; 32],
            b: [7u8; 32],
        };
        let b_changed = Sides {
            a: [9u8; 32],
            b: [6u8; 32],
        };
        let n0 = sample_noise(64, 32, base, &[0], &[0]);
        let n_a = sample_noise(64, 32, a_changed, &[0], &[0]);
        let n_b = sample_noise(64, 32, b_changed, &[0], &[0]);

        // Global addressing: only the E lines key on the selected indices.
        let n_cols = sample_noise(64, 32, base, &[0], &[256]);
        assert_ne!(n0.b.e, n_cols.b.e, "distinct global B-columns must draw distinct E-lines");
        assert_eq!(n0.a.e, n_cols.a.e, "the A E-line depends only on its row");
        assert_eq!(n0.a.f, n_cols.a.f, "the F basis never varies across instances");

        // Side addressing: FA and FB share seedB but never a Side address.
        assert_ne!(n0.a.f, n0.b.f, "FA and FB are distinct Side addresses under seedB");

        // Seed addressing: EA from seedA; both F bases from seedB alone.
        assert_eq!(n0.a.f, n_a.a.f, "FA is independent of seedA");
        assert_ne!(n0.a.e, n_a.a.e, "EA still draws from seedA");
        assert_eq!(n0.b.e, n_a.b.e, "EB is independent of seedA");
        assert_ne!(n0.a.f, n_b.a.f, "FA draws from seedB");
        assert_eq!(n0.b.f, n_a.b.f, "FB is independent of seedA");
        assert_ne!(n0.b.f, n_b.b.f, "FB draws from seedB");
    }

    #[test]
    fn sample_line_pins_fixed_seed_address_bytes() {
        let seed_a: Hash256 = [0x22u8; 32];
        let seed_b: Hash256 = [0x11u8; 32];
        for (seed, side, factor, line, expected_hex) in [
            (
                &seed_a,
                Side::A,
                NoiseFactor::E,
                7u32,
                "be3adfead867e55be469dd50e9e85c50dfdc64e2dfe4eae2e8d6596a6746dd65",
            ),
            (
                &seed_b,
                Side::A,
                NoiseFactor::F,
                0u32,
                "c65268603369d96362e16a5f6ae6615fe6db43eae463e2d3eae55ae3bb5f54e3",
            ),
            (
                &seed_b,
                Side::B,
                NoiseFactor::E,
                7u32,
                "e6dce3e05ce4684ae1dee56555596ae3605de9e6e968e961dce1c2e8d75ee559",
            ),
            (
                &seed_b,
                Side::B,
                NoiseFactor::F,
                0u32,
                "59de69d9696352dee8cee7d4e6da66615be9df69dd66dfe0d7e44f685fe96553",
            ),
        ] {
            assert_eq!(sample_line(seed, side, factor, line, 32), decode_hex(expected_hex));
        }
    }
}
