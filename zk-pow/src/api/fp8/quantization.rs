//! The FP8 quant scheme (per-row scaled noising + FP8 quantization), mirroring
//! `miner_base.quantization`.
//!
//! The scheme is *fused*: from each row's norms it derives per-row scales
//! `alpha`, `beta` and forms `alpha (.) X + beta (.) (E @ F)` (`(.)` = per-row
//! broadcast), quantized to FP8 in one step. There are NO committed scales:
//! the verifier recomputes `alpha`/`beta` from the opened operands, so the FP8
//! values it rebuilds are bit-identical to the miner's. The `(l2, linf)` norms
//! feeding the derivation are computed by the CALLER straight from an
//! operand's exact pre-BF16-rounding values (`exact_norms` for prequant
//! operands, [`Quant::row_norms`] for raw-BF16 ones) and passed into
//! [`Quant::noisy_quantize`] — mirroring the reference's `RowNorms` flow.
//!
//! All scale arithmetic is element-wise `compute_dtype` (BF16) via
//! [`compute`](crate::api::fp8::compute): every input decodes exactly to f32,
//! the op runs in f32, and the result rounds back to BF16 with ties-to-even —
//! the same semantics as the reference's torch BF16 kernels, which have no
//! accumulation order to pin. The one exception is the `E @ F` noise product,
//! which runs on the bit-exact FP8 MMA (`B200::matmul_fp8`).

use anyhow::{Context, Result, ensure};
use itertools::Itertools;

use crate::api::fp8::{
    compute::{bf16_clamp_sym, bf16_div, bf16_fma, bf16_max, bf16_mul},
    dtype::{bf16_to_f32, f32_to_bf16, f32_to_fp8_e4m3, fp8_e4m3_to_f32},
    noise::OperandNoise,
    prequant::round_l2_to_grid,
    utils::{B200, fp32_to_bf16_rne},
};

/// Largest finite FP8 E4M3 magnitude (the quant grid ceiling).
pub const MAX_E4M3: f32 = 448.0;
/// Noise-to-signal ratio (in L2) of the injected `E @ F` noise.
pub const DELTA: f64 = 0.5;
/// Constant L2 norm every noise line is renormalized to (`NOISE_TARGET_NORM` in
/// the reference); an `E @ F` entry is Cauchy-Schwarz-bounded by its square.
pub const NOISE_TARGET_NORM: f64 = 256.0;
/// Floor applied to both `l2` and `linf` before the scale derivation (the
/// reference's `Fp8QuantScheme.row_norms`): a zero row would otherwise divide
/// by zero in `noisy_quantize`, and the floor also guarantees every row a
/// non-vanishing noise scale (`beta > 0`). Exact in BF16 (a power of two).
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
    /// Per-row `l2` (BF16): the floored, grid-rounded rms the scales derive
    /// from. The jackpot policy's noise scale is `sigma_i = DELTA * alpha_i *
    /// l2_i` (the std of the noise actually added to row `i`, in quantized
    /// units).
    pub l2: Vec<u16>,
}

/// A per-row fused noising + quantization scheme. Each
/// [`Quant`](crate::api::fp8::public_params::Quant) variant maps to one
/// implementation of this trait; only the 1-byte discriminant is serialized
/// into `pB` — never any scales, which the verifier recomputes from the
/// opened rows. `F` is the row storage encoding (BF16 as `u16`), `T` the
/// quantized code encoding (E4M3 as `u8`).
pub trait Quant<F, T> {
    /// Per-row `(l2, linf)` norms of a RAW-BF16 row (the fallback for operands
    /// with no prequant structure to compute exact norms from; mirrors the
    /// reference's `_row_norms` raw branch). Prequant operands use
    /// `exact_norms` (in [`prequant`](crate::api::fp8::prequant)) instead.
    fn row_norms(&self, row: &[F]) -> Result<(F, F)>;

    /// Per-row scales `(alpha, beta)` from the norms and the noise rank.
    /// Expects `l2`/`linf` already floored at [`NORM_FLOOR`] (the scheme's
    /// `row_norms` floor in the reference), as `noisy_quantize` does.
    fn derive_row_scales(&self, l2: F, linf: F, r: usize) -> Result<(F, F)>;

    /// Fused `Q(alpha (.) X + beta (.) (E @ F))` over a stack of rows.
    ///
    /// `rows` is `(num_rows x k)` row-major; `noise` carries the paired factors
    /// `(e, f)` — `e` is `(num_rows x r)` and `f` is `(k x r)` (as stored, `f`
    /// row `j` holding column `j` of the reference's `F`). `norms` holds each
    /// row's `(l2, linf)`, computed by the caller from the operand's exact
    /// values (see [`Quant::row_norms`]).
    fn noisy_quantize(&self, rows: &[F], noise: &OperandNoise, norms: &[(F, F)]) -> Result<BuiltRows>;

    /// `Q(alpha * x)` WITHOUT noise.
    fn quantize_clean(&self, alpha: F, x: F) -> Result<T>;

    /// The set of actual (unencoded) differences between grid values, excluding 0. Returned
    /// as exact f32 so the differences are the real values, not re-quantized.
    fn difference_grid(&self) -> Vec<f32>;
}

/// The scale constant `DELTA * sqrt(r)` as BF16 (`delta_r` in the reference):
/// the coefficient of `l2` in the noised-value bound.
///
/// The reference computes it in f64 (a Python float) and torch rounds
/// f64 -> BF16 in one step; here the f64 result goes through f32 first. The
/// double rounding is provably harmless for every rank up to 4096 (see
/// `scale_constants_double_rounding_is_exact`), so the two paths agree bit-for-bit.
fn delta_r_bf16(r: usize) -> Result<u16> {
    f32_to_bf16((DELTA * (r as f64).sqrt()) as f32)
}

/// The scale constant `DELTA * sqrt(r) / NOISE_TARGET_NORM^2` as BF16
/// (`delta_over_std` in the reference): the target noise-to-signal ratio times
/// the rms/peak ratio `sqrt(r)`, over the squared noise-line norm. Same
/// double-rounding guarantee as [`delta_r_bf16`].
fn delta_over_std_bf16(r: usize) -> Result<u16> {
    f32_to_bf16((DELTA * (r as f64).sqrt() / (NOISE_TARGET_NORM * NOISE_TARGET_NORM)) as f32)
}

/// FP8 E4M3 per-row fused quantization — the [`Quant`] implementation
/// corresponding to [`Quant::Fp8E4M3Prequant`].
pub struct Fp8E4M3Quant;

impl Quant<u16, u8> for Fp8E4M3Quant {
    /// Per-row `(l2, linf)`: `rms(X) = sqrt(mean_j X_j^2)` and `||X||_inf`.
    ///
    /// `l2` is summed in f32 and then cast down to a BF16 variant with its low
    /// `L2_ROUNDED_BITS` mantissa bits cleared, so miner and verifier agree
    /// on `l2` even if their f32 sums differ by an ulp (the reference reduces
    /// in torch's pairwise order; this sums sequentially — the cleared bits
    /// absorb the difference).
    ///
    /// A zero row yields `(0, 0)`; `noisy_quantize`'s [`NORM_FLOOR`] guards
    /// the division instead of an epsilon bump on `linf`.
    fn row_norms(&self, row: &[u16]) -> Result<(u16, u16)> {
        let k = row.len();
        ensure!(k > 0, "row must be non-empty");
        let sumsq: f32 = row
            .iter()
            .map(|&x| {
                let v = bf16_to_f32(x);
                v * v
            })
            .sum();
        ensure!(sumsq.is_finite(), "row sum of squares overflows f32");
        let l2 = round_l2_to_grid(f32_to_bf16((sumsq / k as f32).sqrt())?);

        // BF16 abs/amax are exact (sign strip + compare), so folding over the
        // decoded f32 values reproduces them bit-for-bit.
        let abs_max = row.iter().map(|&x| bf16_to_f32(x).abs()).fold(0.0f32, f32::max);
        let linf = f32_to_bf16(abs_max)?;
        Ok((l2, linf))
    }

    /// Derive the per-row scales `(alpha, beta)` from the (pre-floored) norms.
    ///
    /// Pick `alpha`, `beta` so that per row (1) `|alpha*X + beta*E@F| <= MAX_E4M3`
    /// (no saturation loss) and (2) `rms(beta*E@F) = DELTA * rms(alpha*X)`. With
    /// an `E @ F` entry Cauchy-Schwarz-bounded by `NOISE_TARGET_NORM^2` and with
    /// rms `~ NOISE_TARGET_NORM^2 / sqrt(r)`, that gives
    ///   `alpha = MAX_E4M3 / (linf + DELTA*sqrt(r) * l2)` — signal peak + noise peak
    ///   `beta  = alpha * l2 * delta_over_std`
    /// with `delta_over_std = DELTA*sqrt(r) / NOISE_TARGET_NORM^2`. The
    /// [`NORM_FLOOR`] the caller applied to `l2`/`linf` keeps `alpha` finite on
    /// a (near-)zero row (a huge scale on a ~zero row is harmless —
    /// reconstruction divides it back out) and `beta` strictly positive.
    fn derive_row_scales(&self, l2: u16, linf: u16, r: usize) -> Result<(u16, u16)> {
        let max_e4m3 = f32_to_bf16(MAX_E4M3)?; // exact
        let delta_r = delta_r_bf16(r)?;
        let delta_over_std = delta_over_std_bf16(r)?;

        let noised_bound = bf16_fma(delta_r, l2, linf).context("noised bound")?;
        let alpha = bf16_div(max_e4m3, noised_bound).context("alpha scale")?;
        let beta = bf16_mul(bf16_mul(alpha, l2)?, delta_over_std).context("beta scale")?;
        Ok((alpha, beta))
    }

    /// The `E @ F` product runs on the bit-exact hardware FP8 MMA and is cast
    /// MATMUL -> COMPUTE (f32 -> BF16, RNE); the noised value is a single-rounding
    /// FMA `alpha*X + beta*(E@F)`, clamped to `±MAX_E4M3` before the BF16 -> E4M3
    /// cast.
    fn noisy_quantize(&self, rows: &[u16], noise: &OperandNoise, norms: &[(u16, u16)]) -> Result<BuiltRows> {
        let e = &noise.e;
        let f = &noise.f;
        let num_rows = norms.len();
        // The inputs are the verifier's own construction (`open_and_noisy_quantize`
        // feeding `open_prequant`/`exact_norms`), so these are debug invariants,
        // not runtime validation of untrusted data.
        debug_assert!(rows.len().is_multiple_of(num_rows), "rows must be num_rows x k");
        debug_assert!(e.len().is_multiple_of(num_rows), "e must be num_rows x r");
        let k = rows.len() / num_rows;
        let r = e.len() / num_rows;
        debug_assert!(f.len() == k * r, "f must be k x r");

        let noise = B200 {}.matmul_fp8(e, f, None, num_rows, k, r)?;
        let noise: Vec<u16> = noise.into_iter().map(fp32_to_bf16_rne).collect();

        let max_e4m3 = f32_to_bf16(MAX_E4M3)?;
        let floor = f32_to_bf16(NORM_FLOOR)?; // exact (power of two)
        let mut noised_part = Vec::with_capacity(num_rows * k);
        let mut alphas = Vec::with_capacity(num_rows);
        let mut betas = Vec::with_capacity(num_rows);
        let mut l2s = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let row = &rows[i * k..(i + 1) * k];
            // The scheme's norm floor (the reference's `row_norms` method).
            let l2 = bf16_max(norms[i].0, floor);
            let linf = bf16_max(norms[i].1, floor);
            let (alpha, beta) = self.derive_row_scales(l2, linf, r).with_context(|| format!("row {i}"))?;
            for j in 0..k {
                // noised = fma(alpha, X, beta * noise): a single rounding, more
                // accurate than a separate mul + add.
                let noised = bf16_fma(alpha, row[j], bf16_mul(beta, noise[i * k + j])?)?;
                // Very rarely this clamp has any effect (and only if r > 12).
                let clamped = bf16_clamp_sym(noised, max_e4m3);
                noised_part.push(f32_to_fp8_e4m3(bf16_to_f32(clamped))?);
            }
            alphas.push(alpha);
            betas.push(beta);
            l2s.push(l2);
        }
        Ok(BuiltRows {
            noised_part,
            alpha: alphas,
            beta: betas,
            l2: l2s,
        })
    }

    fn quantize_clean(&self, alpha: u16, x: u16) -> Result<u8> {
        let max_e4m3 = f32_to_bf16(MAX_E4M3)?;
        let scaled = bf16_clamp_sym(bf16_mul(alpha, x)?, max_e4m3);
        f32_to_fp8_e4m3(bf16_to_f32(scaled))
    }

    /// The set of all actual differences between E4M3 grid values: for every
    /// ordered pair `(v1, v2)` of E4M3 grid values, the real value `v1 - v2`
    /// (iterating ordered pairs yields both `v1 - v2` and `v2 - v1`). Returned as
    /// exact, unencoded f32, deduplicated and sorted so the result is deterministic.
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
        // Differences are all finite, so `total_cmp` is a total order; dedup then
        // merges numerically-equal entries (including ±0.0).
        grid.sort_by(f32::total_cmp);
        grid.dedup();
        grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference single-rounding f64 -> bf16 (RNE), used to prove the production
    /// f64 -> f32 -> bf16 path never double-rounds for any sanctioned rank.
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

    /// Cross-checked against the reference `ComputeOps.const` (torch), generated by running `miner_base` under torch.
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

    /// Cross-checked against the reference `Fp8QuantScheme.row_norms`, generated by
    /// running `miner_base` under torch.
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

    /// Cross-checked against the reference scale derivation (torch), generated by running `miner_base` under torch.
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
