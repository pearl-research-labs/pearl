//! Per-row noise scaling and FP8 quantization.
//!
//! For each clean row `X_i`, derive a scale `alpha_i` and a noise scale `beta_i`.
//! The caller supplies the row's RMS and absolute maximum, using
//! [`prequant::exact_norms`](crate::api::fp8::prequant::exact_norms) for prequant
//! inputs or [`Quant::row_norms`] for raw BF16.
//!
//! Write `l2` and `linf` for these norms after applying [`NORM_FLOOR`]. Let `r`
//! be the noise rank and `N = NOISE_TARGET_NORM`. With `R16` denoting rounding
//! to nearest BF16, ties to even, the scale chain is:
//!
//! ```text
//! d            = R16(DELTA * sqrt(r))
//! c            = R16(DELTA * sqrt(r) / N^2)
//! noised_bound = R16(d*l2 + linf)
//! alpha        = R16(448 / noised_bound)
//! beta         = R16(R16(alpha*l2) * c)
//! ```
//!
//! The denominator comes from `|E_i dot F_j| <= N^2` for noise lines of norm N:
//! before rounding, the noised magnitude is bounded by `alpha*linf + beta*N^2`.
//! Actual noise lines have only approximately this norm, so the final cast clamps
//! to the finite E4M3 range `[-448, 448]`.
//!
//! Quantization preserves these rounding boundaries (`R8` rounds to E4M3,
//! also to nearest with ties to even):
//!
//! ```text
//! n_ij  = R16(B200(E @ F)_ij)
//! t_ij  = R16(beta_i * n_ij)
//! y_ij  = R16(alpha_i * X_ij + t_ij)
//! X'_ij = R8(clamp(y_ij, -448, 448))
//! ```
//!
//! Both multiply-adds above round once, using [`compute::bf16_fma`](crate::api::fp8::compute::bf16_fma).
//! [`B200::matmul_fp8`] supplies the noise product's accumulation semantics.

use anyhow::{Context, Result, ensure};
use itertools::Itertools;

use crate::api::fp8::{
    compute::{bf16_clamp_sym, bf16_div, bf16_fma, bf16_max, bf16_mul},
    dtype::{bf16_to_f32, f32_to_bf16, f32_to_fp8_e4m3, fp8_e4m3_to_f32},
    noise::OperandNoise,
    prequant::round_l2_to_grid,
    utils::{B200, fp32_to_bf16_rne},
};

/// Largest finite FP8 E4M3 magnitude.
pub const MAX_E4M3: f32 = 448.0;
/// Target ratio between the noise and clean-row L2 norms.
pub const DELTA: f64 = 0.5;
/// Target L2 norm before rounding each noise line.
pub const NOISE_TARGET_NORM: f64 = 256.0;
/// Minimum RMS and absolute maximum for scale derivation; avoids division by zero on zero rows.
pub const NORM_FLOOR: f32 = 1.0 / 4_294_967_296.0; // 2^-32, exact

/// One side's rebuilt operand: the FP8 values of `A'` (or `B'`) plus the
/// per-row scales used to build them (consumed again by the jackpot policy).
pub struct BuiltRows {
    /// (rows x k) FP8 E4M3 values, row-major.
    pub noised_part: Vec<u8>,
    /// Per-row `alpha` (BF16): scale on the clean operand.
    pub alpha: Vec<u16>,
    /// Per-row `beta` (BF16): scale on the `E @ F` noise.
    pub beta: Vec<u16>,
    /// Per-row RMS (BF16), grid-rounded and floored. The jackpot policy uses
    /// it in `sigma_i = DELTA * alpha_i * l2_i`.
    pub l2: Vec<u16>,
}

/// Per-row noise and quantization. Each [`Quant`](crate::api::fp8::public_params::Quant)
/// variant selects an implementation. `F` encodes inputs (BF16 as `u16`),
/// `T` encodes outputs (E4M3 as `u8`). The derived scales are not serialized.
pub trait Quant<F, T> {
    /// Return `(rms, abs_max)` for a raw BF16 row.
    /// Prequant inputs use `exact_norms` in [`prequant`](crate::api::fp8::prequant).
    fn row_norms(&self, row: &[F]) -> Result<(F, F)>;

    /// Return `(alpha, beta)` from the norms and noise rank.
    /// Expects both norms floored at [`NORM_FLOOR`].
    fn derive_row_scales(&self, l2: F, linf: F, r: usize) -> Result<(F, F)>;

    /// Scale each clean row, add its noise, and convert the result to FP8.
    ///
    /// `rows` is `num_rows x k`, and `norms` holds each row's `(rms, abs_max)`.
    /// The noise factors are stored row-major: `e` is `num_rows x r`, and
    /// `f` is `k x r`, with row `j` holding column `j` of the mathematical F.
    fn noisy_quantize(&self, rows: &[F], noise: &OperandNoise, norms: &[(F, F)]) -> Result<BuiltRows>;

    /// `Q(alpha * x)` without noise.
    fn quantize_clean(&self, alpha: F, x: F) -> Result<T>;

    /// The set of actual (unencoded) differences between grid values, excluding 0. Returned
    /// as exact f32 so the differences are the real values, not re-quantized.
    fn difference_grid(&self) -> Vec<f32>;
}

/// `DELTA * sqrt(r)` as BF16.
/// `scale_constants_double_rounding_is_exact` checks ranks 1..=4096 against
/// direct f64-to-BF16 rounding.
fn delta_r_bf16(r: usize) -> Result<u16> {
    f32_to_bf16((DELTA * (r as f64).sqrt()) as f32)
}

/// `DELTA * sqrt(r) / NOISE_TARGET_NORM^2` as BF16.
/// Uses the same rounding path as [`delta_r_bf16`].
fn delta_over_std_bf16(r: usize) -> Result<u16> {
    f32_to_bf16((DELTA * (r as f64).sqrt() / (NOISE_TARGET_NORM * NOISE_TARGET_NORM)) as f32)
}

/// [`Quant`] implementation for
/// [`Fp8E4M3Prequant`](crate::api::fp8::public_params::Quant::Fp8E4M3Prequant).
pub struct Fp8E4M3Quant;

impl Quant<u16, u8> for Fp8E4M3Quant {
    /// Compute the row's RMS and absolute maximum in f32, then round to BF16
    /// and coarsen the RMS with [`round_l2_to_grid`]. A zero row returns `(0, 0)`;
    /// `noisy_quantize` applies [`NORM_FLOOR`].
    fn row_norms(&self, row: &[u16]) -> Result<(u16, u16)> {
        let k = row.len();
        ensure!(k > 0, "row must be non-empty");
        let sum_of_squares: f32 = row
            .iter()
            .map(|&x| {
                let value = bf16_to_f32(x);
                value * value
            })
            .sum();
        ensure!(sum_of_squares.is_finite(), "row sum of squares overflows f32");
        let l2 = round_l2_to_grid(f32_to_bf16((sum_of_squares / k as f32).sqrt())?);

        // Absolute value and maximum are exact for decoded BF16 inputs.
        let abs_max = row.iter().map(|&x| bf16_to_f32(x).abs()).fold(0.0f32, f32::max);
        let linf = f32_to_bf16(abs_max)?;
        Ok((l2, linf))
    }

    /// Derive the clean and noise scales from norms already floored at [`NORM_FLOOR`].
    fn derive_row_scales(&self, l2: u16, linf: u16, r: usize) -> Result<(u16, u16)> {
        let max_e4m3 = f32_to_bf16(MAX_E4M3)?; // exact
        let delta_r = delta_r_bf16(r)?;
        let delta_over_std = delta_over_std_bf16(r)?;

        // Round the magnitude bound once, after the multiply-add.
        let noised_bound = bf16_fma(delta_r, l2, linf).context("noised bound")?;
        let alpha = bf16_div(max_e4m3, noised_bound).context("alpha scale")?;
        // Each multiplication in the noise scale has its own BF16 rounding.
        let beta = bf16_mul(bf16_mul(alpha, l2)?, delta_over_std).context("beta scale")?;
        Ok((alpha, beta))
    }

    /// Build the noised FP8 operand and retain its per-row scales for the jackpot checks.
    fn noisy_quantize(&self, rows: &[u16], noise: &OperandNoise, norms: &[(u16, u16)]) -> Result<BuiltRows> {
        let row_factors = &noise.e;
        let basis_factors = &noise.f;
        let num_rows = norms.len();
        // The caller supplies rows, norms and noise factors with matching dimensions.
        debug_assert!(rows.len().is_multiple_of(num_rows), "rows must be num_rows x k");
        debug_assert!(row_factors.len().is_multiple_of(num_rows), "e must be num_rows x r");
        let k = rows.len() / num_rows;
        let noise_rank = row_factors.len() / num_rows;
        debug_assert!(basis_factors.len() == k * noise_rank, "f must be k x r");

        let noise_fp32 = B200 {}.matmul_fp8(row_factors, basis_factors, None, num_rows, k, noise_rank)?;
        let noise_bf16: Vec<u16> = noise_fp32.into_iter().map(fp32_to_bf16_rne).collect();

        let max_e4m3 = f32_to_bf16(MAX_E4M3)?;
        let floor = f32_to_bf16(NORM_FLOOR)?; // exact (power of two)
        let mut noised_part = Vec::with_capacity(num_rows * k);
        let mut alphas = Vec::with_capacity(num_rows);
        let mut betas = Vec::with_capacity(num_rows);
        let mut row_rms = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let row = &rows[i * k..(i + 1) * k];
            let l2 = bf16_max(norms[i].0, floor);
            let linf = bf16_max(norms[i].1, floor);
            let (alpha, beta) = self
                .derive_row_scales(l2, linf, noise_rank)
                .with_context(|| format!("row {i}"))?;
            for j in 0..k {
                // Round the noise term before the fused clean multiply-add.
                let noised = bf16_fma(alpha, row[j], bf16_mul(beta, noise_bf16[i * k + j])?)?;
                let clamped = bf16_clamp_sym(noised, max_e4m3);
                noised_part.push(f32_to_fp8_e4m3(bf16_to_f32(clamped))?);
            }
            alphas.push(alpha);
            betas.push(beta);
            row_rms.push(l2);
        }
        Ok(BuiltRows {
            noised_part,
            alpha: alphas,
            beta: betas,
            l2: row_rms,
        })
    }

    fn quantize_clean(&self, alpha: u16, x: u16) -> Result<u8> {
        let max_e4m3 = f32_to_bf16(MAX_E4M3)?;
        let scaled = bf16_clamp_sym(bf16_mul(alpha, x)?, max_e4m3);
        f32_to_fp8_e4m3(bf16_to_f32(scaled))
    }

    /// Distinct nonzero differences between finite E4M3 values, returned as sorted exact f32s.
    fn difference_grid(&self) -> Vec<f32> {
        // E4M3 grid values: every code except the two NaN encodings (`S.1111.111`).
        let values: Vec<f32> = (0u16..=0xFF)
            .map(|c| c as u8)
            .filter(|&c| c & 0x7F != 0x7F)
            .map(fp8_e4m3_to_f32)
            .collect();
        let mut grid: Vec<f32> = values
            .iter()
            .cartesian_product(&values)
            .filter_map(|(&v1, &v2)| if v1 != v2 { Some(v1 - v2) } else { None })
            .collect();
        grid.sort_by(f32::total_cmp);
        grid.dedup();
        grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct f64-to-BF16 rounding reference for the tested scale-constant inputs.
    fn f64_to_bf16_single(x: f64) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 63) as u16) << 15;
        let exp = ((bits >> 52) & 0x7FF) as i64 - 1023 + 127;
        let man = bits & 0xF_FFFF_FFFF_FFFF;
        let (mut kept, rem, half) = ((man >> 45) as u16, man & ((1u64 << 45) - 1), 1u64 << 44);
        let mut e = exp as u16;
        if rem > half || (rem == half && kept & 1 == 1) {
            kept += 1;
            if kept == 0x80 {
                kept = 0;
                e += 1;
            }
        }
        sign | (e << 7) | kept
    }

    #[test]
    fn scale_constants_double_rounding_is_exact() {
        for r in 1..=4096usize {
            assert_eq!(
                delta_r_bf16(r).unwrap(),
                f64_to_bf16_single(DELTA * (r as f64).sqrt()),
                "delta_r double rounding diverges at r={r}"
            );
            assert_eq!(
                delta_over_std_bf16(r).unwrap(),
                f64_to_bf16_single(DELTA * (r as f64).sqrt() / (NOISE_TARGET_NORM * NOISE_TARGET_NORM)),
                "delta_over_std double rounding diverges at r={r}"
            );
        }
    }

    /// Expected values generated with PyTorch.
    #[test]
    fn scale_constants_match_reference_vectors() {
        let cases: &[(usize, u16, u16)] = &[
            (1, 0x3f00, 0x3700),
            (4, 0x3f80, 0x3780),
            (16, 0x4000, 0x3800),
            (64, 0x4080, 0x3880),
            (128, 0x40b5, 0x38b5),
            (256, 0x4100, 0x3900),
            (1024, 0x4180, 0x3980),
            (4096, 0x4200, 0x3a00),
        ];
        for &(r, dr, dos) in cases {
            assert_eq!(delta_r_bf16(r).unwrap(), dr, "delta_r r={r}");
            assert_eq!(delta_over_std_bf16(r).unwrap(), dos, "delta_over_std r={r}");
        }
    }

    /// Expected values generated with PyTorch.
    #[test]
    fn row_norms_matches_reference_vectors() {
        let bf = |v: f32| f32_to_bf16(v).unwrap();
        // rows -> (l2 bits, linf bits).
        let cases: &[(&[f32], u16, u16)] = &[
            (&[1.0; 8], 0x3f80, 0x3f80),
            (&[1.0, -2.0, 3.0, -4.0], 0x4030, 0x4080),
            (&[0.0; 4], 0x0000, 0x0000),
            (&[0.5, 0.25, -0.75, 1.5, -2.0, 0.125, 3.0, -1.0], 0x3fbc, 0x4040),
        ];
        for &(vals, l2, linf) in cases {
            let row: Vec<u16> = vals.iter().map(|&v| bf(v)).collect();
            assert_eq!(Fp8E4M3Quant.row_norms(&row).unwrap(), (l2, linf), "row {vals:?}");
        }
    }

    /// Expected values generated with PyTorch.
    #[test]
    fn derive_row_scales_matches_reference_vectors() {
        // (l2 bits, linf bits, r) -> (alpha bits, beta bits). Inputs are
        // pre-floored, as `noisy_quantize` passes them.
        let cases: &[(u16, u16, usize, u16, u16)] = &[
            (0x3f80, 0x3f80, 16, 0x4315, 0x3b95),
            (0x3f80, 0x4080, 16, 0x4295, 0x3b15),
            (0x2f80, 0x2f80, 16, 0x5315, 0x3b95), // zero row, floored to 2^-32 on both norms
            (0x3f00, 0x4040, 64, 0x42b3, 0x3b33),
        ];
        for &(l2, linf, r, alpha, beta) in cases {
            assert_eq!(
                Fp8E4M3Quant.derive_row_scales(l2, linf, r).unwrap(),
                (alpha, beta),
                "l2={l2:#06x} linf={linf:#06x} r={r}"
            );
        }
    }

    #[test]
    fn zero_row_gets_floored_norms_and_positive_noise_scale() {
        // A zero row's raw norms are (0, 0); noisy_quantize floors both at
        // NORM_FLOOR = 2^-32 before the derivation, so alpha stays finite and
        // beta strictly positive (the row still receives real noise).
        let zeros = vec![0u16; 64];
        assert_eq!(Fp8E4M3Quant.row_norms(&zeros).unwrap(), (0, 0));
        let floor = f32_to_bf16(NORM_FLOOR).unwrap();
        let (alpha, beta) = Fp8E4M3Quant.derive_row_scales(floor, floor, 16).unwrap();
        assert!(bf16_to_f32(alpha).is_finite() && bf16_to_f32(alpha) > 0.0);
        assert!(bf16_to_f32(beta) > 0.0, "floored l2 keeps the noise scale positive");
    }

    #[test]
    fn noisy_quantize_produces_valid_deterministic_codes() {
        let (num_rows, k, r) = (2usize, 64usize, 16usize);
        // Rows of alternating ±1.0.
        let rows: Vec<u16> = (0..num_rows * k).map(|i| if i % 2 == 0 { 0x3F80 } else { 0xBF80 }).collect();
        // Mesh-like noise factors: ±0.5 (E4M3 0x30 / 0xB0).
        let e: Vec<u8> = (0..num_rows * r).map(|i| if i % 3 == 0 { 0xB0 } else { 0x30 }).collect();
        let f: Vec<u8> = (0..k * r).map(|i| if i % 5 == 0 { 0xB0 } else { 0x30 }).collect();
        let noise = OperandNoise { e, f };
        let norms: Vec<(u16, u16)> = rows.chunks(k).map(|row| Fp8E4M3Quant.row_norms(row).unwrap()).collect();

        let built = Fp8E4M3Quant.noisy_quantize(&rows, &noise, &norms).unwrap();
        assert_eq!(built.noised_part.len(), num_rows * k);
        assert_eq!(built.alpha.len(), num_rows);
        assert_eq!(built.l2.len(), num_rows);
        // Codes decode to finite values within the E4M3 range.
        for &c in &built.noised_part {
            assert!(fp8_e4m3_to_f32(c).abs() <= MAX_E4M3);
        }
        // Deterministic.
        let again = Fp8E4M3Quant.noisy_quantize(&rows, &noise, &norms).unwrap();
        assert_eq!(built.noised_part, again.noised_part);
        assert_eq!(built.alpha, again.alpha);
        assert_eq!(built.beta, again.beta);
        assert_eq!(built.l2, again.l2);
    }
}
