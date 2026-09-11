//! Deterministic FP8 noise for the opened rows of A and columns of B.
//!
//! Each operand's noise is a matrix product `E @ F` with inner dimension `rank`.
//! A noise "line" is a vector of `rank` entries: one row of E or one column of F.
//! Lines are sampled independently from public seeds and indices, so any required
//! line can be regenerated without sampling the rest of the matrix.
//!
//! Every line is scaled toward the same Euclidean norm, [`NOISE_TARGET_NORM`],
//! then rounded to FP8 E4M3. The target is independent of `rank`; quantization
//! uses it when choosing how much noise to add.

use crate::api::fp8::compute::{bf16_div, bf16_mul};
use crate::api::fp8::dtype::{bf16_to_f32, f32_to_bf16, f32_to_fp8_e4m3};
use crate::api::fp8::public_params::PublicParams;
use crate::api::fp8::quantization::NOISE_TARGET_NORM;
use crate::api::fp8::transcript::{LABEL_NOISE_LINE, subkey};
use crate::api::primitives::{Hash256, IncompleteBlockHeader, Sides};

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

/// Identifies the operand whose noise is being sampled: matrix A or matrix B.
///
/// `sample_line` writes this choice as the first byte of its hash input:
/// 0 for A, 1 for B.
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
/// `a_rows` and `b_cols` are global matrix indices; E preserves their order.
/// `k` is the reduction dimension and `rank` is the inner dimension of `E @ F`.
/// For a fixed seed and side, F is independent of which rows or expert were selected.
pub(crate) fn sample_noise(k: usize, rank: u16, seeds: Sides<Hash256>, a_rows: &[u32], b_cols: &[u32]) -> Noise {
    let e_a = a_rows
        .iter()
        .flat_map(|&row| sample_line(&seeds.a, Side::A, NoiseFactor::E, row, rank))
        .collect();
    let e_b = b_cols
        .iter()
        .flat_map(|&col| sample_line(&seeds.b, Side::B, NoiseFactor::E, col, rank))
        .collect();
    let f_a = (0..k as u32)
        .flat_map(|i| sample_line(&seeds.a, Side::A, NoiseFactor::F, i, rank))
        .collect();
    let f_b = (0..k as u32)
        .flat_map(|i| sample_line(&seeds.b, Side::B, NoiseFactor::F, i, rank))
        .collect();

    Noise {
        a: OperandNoise { e: e_a, f: f_a },
        b: OperandNoise { e: e_b, f: f_b },
    }
}

/// Rebuild the noise factors from the public statement and proposed header.
///
/// The statement selects the A rows and B columns. For mixture-of-experts (MoE)
/// jobs, these are the winning expert's global indices. The header and statement
/// determine the seeds, so the prover and verifier can regenerate the same noise.
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
    use crate::api::fp8::dtype::fp8_e4m3_to_f32;
    use crate::api::primitives::Sides;

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

    #[test]
    fn global_addresses_disambiguate_e_lines() {
        let seeds = Sides {
            a: [6u8; 32],
            b: [5u8; 32],
        };
        let a = sample_noise(64, 32, seeds, &[0], &[0]);
        let b = sample_noise(64, 32, seeds, &[0], &[256]);
        assert_ne!(a.b.e, b.b.e, "distinct global B-columns must draw distinct E-lines");
        assert_eq!(a.a.e, b.a.e, "the A E-line depends only on its row");
        assert_eq!(a.a.f, b.a.f, "the F basis never varies across instances");
    }

    #[test]
    fn f_basis_is_side_deterministic_and_expert_independent() {
        let seeds = Sides {
            a: [9u8; 32],
            b: [7u8; 32],
        };
        let a = sample_noise(64, 32, seeds, &[], &[]);
        let b = sample_noise(64, 32, seeds, &[], &[]);
        assert_eq!(a.a.f, b.a.f, "FA draws solely from the A seed");
        assert_eq!(a.b.f, b.b.f, "FB draws solely from the B seed");
    }
}
