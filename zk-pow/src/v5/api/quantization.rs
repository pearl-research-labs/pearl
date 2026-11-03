//! FP16 per-row fused noisy quantization for the A100 scheme.
//!
//! Mirrors the FP8 scheme ([`crate::v4::api::quantization`]) with two
//! deliberate differences:
//!
//! 1. The quantization target is FP16 (ceiling `Q = 65504`) rather than FP8.
//! 2. The noised-value FMA `alpha*X + beta*(E@F)` is evaluated in **f32** and
//!    rounded once to FP16, not in BF16. BF16 carries only 8 significand bits —
//!    coarser than FP16 — so computing the FMA in BF16 (as the FP8 path does,
//!    where the FP8 target is coarser than BF16 anyway) would throw away the
//!    precision this scheme exists to keep. The per-row scales are still derived
//!    in BF16 (they are scalars; see the FP8 rationale) and applied in f32.
//!
//! The scale derivation is unchanged: pick `alpha`, `beta` so each row's noised
//! value fits `[-Q, Q]` without saturation and the noise carries relative
//! Euclidean weight `delta`. With the noise `E@F` built on the A100 FP16
//! datapath and its lines renormalized to `NOISE_TARGET_NORM`,
//!   `alpha = Q / (linf + delta*sqrt(r) * l2)`,
//!   `beta  = alpha * l2 * (delta*sqrt(r) / NOISE_TARGET_NORM^2)`.

use anyhow::{Context, Result};

use super::accumulate::a100_matmul;
use super::dtype::{f32_to_fp16, fp16_to_f32};
use crate::v4::api::compute::{bf16_div, bf16_fma, bf16_max, bf16_mul};
use crate::v4::api::dtype::{bf16_to_f32, f32_to_bf16};
use crate::v4::api::prequant::round_l2_to_grid;
use crate::v4::api::quantization::NOISE_TARGET_NORM;

/// Largest finite FP16 magnitude (the quant grid ceiling).
pub const MAX_FP16: f32 = 65504.0;
/// Per-row relative noise weight `delta` for the A100 scheme.
pub const DELTA: f64 = 0.5;
/// Norm floor applied to `l2`/`linf` before scale derivation (`2^-32`, exact in
/// BF16); guards the division and keeps `beta > 0` on a (near-)zero row.
pub const NORM_FLOOR: f32 = 1.0 / 4_294_967_296.0;

/// One side's rebuilt FP16 operand plus the per-row scales (reused by the
/// jackpot policy and by noise peeling).
pub struct BuiltRows16 {
    /// `rows x k` FP16 values (bit patterns), row-major.
    pub noised_part: Vec<u16>,
    /// Per-row `alpha` (BF16): scale on the clean operand.
    pub alpha: Vec<u16>,
    /// Per-row `beta` (BF16): scale on the `E@F` noise.
    pub beta: Vec<u16>,
    /// Per-row `l2` (BF16): the floored, grid-rounded rms the scales derive from.
    pub l2: Vec<u16>,
}

/// `delta * sqrt(r)` as BF16.
fn delta_r_bf16(r: usize) -> Result<u16> {
    f32_to_bf16((DELTA * (r as f64).sqrt()) as f32)
}

/// `delta * sqrt(r) / NOISE_TARGET_NORM^2` as BF16.
fn delta_over_std_bf16(r: usize) -> Result<u16> {
    f32_to_bf16((DELTA * (r as f64).sqrt() / (NOISE_TARGET_NORM * NOISE_TARGET_NORM)) as f32)
}

/// Per-row `(l2, linf)` from an FP16 operand row (decoded exactly to f32):
/// `l2 = grid4(rms(X))`, `linf = ||X||_inf`.
pub fn row_norms(row: &[u16]) -> Result<(u16, u16)> {
    let k = row.len();
    anyhow::ensure!(k > 0, "row must be non-empty");
    let sumsq: f32 = row
        .iter()
        .map(|&x| {
            let v = fp16_to_f32(x);
            v * v
        })
        .sum();
    anyhow::ensure!(sumsq.is_finite(), "row sum of squares overflows f32");
    let l2 = round_l2_to_grid(f32_to_bf16((sumsq / k as f32).sqrt())?);
    let abs_max = row.iter().map(|&x| fp16_to_f32(x).abs()).fold(0.0f32, f32::max);
    let linf = f32_to_bf16(abs_max)?;
    Ok((l2, linf))
}

/// Derive `(alpha, beta)` (BF16) from the floored norms, exactly as the FP8
/// scheme but with the FP16 ceiling.
pub fn derive_row_scales(l2: u16, linf: u16, r: usize) -> Result<(u16, u16)> {
    let max_fp16 = f32_to_bf16(MAX_FP16)?;
    let delta_r = delta_r_bf16(r)?;
    let delta_over_std = delta_over_std_bf16(r)?;
    let noised_bound = bf16_fma(delta_r, l2, linf).context("noised bound")?;
    let alpha = bf16_div(max_fp16, noised_bound).context("alpha scale")?;
    let beta = bf16_mul(bf16_mul(alpha, l2)?, delta_over_std).context("beta scale")?;
    Ok((alpha, beta))
}

/// Fused per-row noisy quantization of an FP16 operand.
///
/// `rows` is `num_rows x k` FP16 bit patterns; `e` is `num_rows x r` and `f` is
/// `k x r` FP16 noise lines (so `N = E @ F^T` is `num_rows x k`), both computed
/// on the A100 datapath; `norms` is the per-row `(l2, linf)` from [`row_norms`].
/// Returns the noised FP16 operand and the scales. The FMA runs in f32.
pub fn noisy_quantize(
    rows: &[u16],
    e: &[u16],
    f: &[u16],
    norms: &[(u16, u16)],
    r: usize,
) -> Result<BuiltRows16> {
    let num_rows = norms.len();
    anyhow::ensure!(num_rows > 0 && rows.len() % num_rows == 0, "rows must be num_rows x k");
    let k = rows.len() / num_rows;
    anyhow::ensure!(e.len() == num_rows * r && f.len() == k * r, "noise shape mismatch");

    // N = E @ F^T on the committed A100 FP16 datapath (bit-exact).
    let noise = a100_matmul(e, f, None, num_rows, k, r);

    let floor = f32_to_bf16(NORM_FLOOR)?;
    let mut noised_part = Vec::with_capacity(num_rows * k);
    let mut alphas = Vec::with_capacity(num_rows);
    let mut betas = Vec::with_capacity(num_rows);
    let mut l2s = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let l2 = bf16_max(norms[i].0, floor);
        let linf = bf16_max(norms[i].1, floor);
        let (alpha, beta) = derive_row_scales(l2, linf, r).with_context(|| format!("row {i}"))?;
        let af = bf16_to_f32(alpha);
        let bf = bf16_to_f32(beta);
        for j in 0..k {
            // Single-rounding FMA in f32, then round once to FP16.
            let noised = af.mul_add(fp16_to_f32(rows[i * k + j]), bf * noise[i * k + j]);
            let clamped = noised.clamp(-MAX_FP16, MAX_FP16);
            noised_part.push(f32_to_fp16(clamped).context("FP16 cast")?);
        }
        alphas.push(alpha);
        betas.push(beta);
        l2s.push(l2);
    }
    Ok(BuiltRows16 { noised_part, alpha: alphas, beta: betas, l2: l2s })
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: usize = 32;

    #[test]
    fn unit_row_alpha_matches_formula() {
        // For l2 = linf = 1, alpha = Q / (1 + delta*sqrt(r)).
        let one = f32_to_bf16(1.0).unwrap();
        let (alpha, _beta) = derive_row_scales(one, one, R).unwrap();
        let delta_r = bf16_to_f32(delta_r_bf16(R).unwrap());
        let expect = bf16_div(f32_to_bf16(MAX_FP16).unwrap(), f32_to_bf16(1.0 + delta_r).unwrap()).unwrap();
        assert_eq!(alpha, expect);
    }

    #[test]
    fn output_is_valid_fp16_in_range() {
        let k = 64;
        let rows: Vec<u16> = (0..k).map(|j| f32_to_fp16((j as f32 - 32.0) * 0.1).unwrap()).collect();
        let e: Vec<u16> = (0..R).map(|t| f32_to_fp16(((t % 5) as f32 - 2.0) * 16.0).unwrap()).collect();
        let f: Vec<u16> = (0..k * R).map(|t| f32_to_fp16(((t % 7) as f32 - 3.0) * 16.0).unwrap()).collect();
        let norms = [row_norms(&rows).unwrap()];
        let built = noisy_quantize(&rows, &e, &f, &norms, R).unwrap();
        assert_eq!(built.noised_part.len(), k);
        for &b in &built.noised_part {
            let v = fp16_to_f32(b);
            assert!(v.is_finite() && v.abs() <= MAX_FP16, "out-of-range {v}");
        }
    }

    #[test]
    fn zero_noise_dequantizes_to_fp16_accuracy() {
        // With zero noise, A' = RNE_fp16(alpha * A); dividing alpha back out
        // recovers A to FP16 rounding accuracy (not the coarser BF16 the FP8
        // path would give).
        let k = 48;
        let rows: Vec<u16> = (0..k).map(|j| f32_to_fp16(1.0 + (j as f32) * 0.03).unwrap()).collect();
        let e = vec![0u16; R];
        let f = vec![0u16; k * R];
        let norms = [row_norms(&rows).unwrap()];
        let built = noisy_quantize(&rows, &e, &f, &norms, R).unwrap();
        let af = bf16_to_f32(built.alpha[0]);
        let mut max_rel = 0f32;
        for j in 0..k {
            let orig = fp16_to_f32(rows[j]);
            let recovered = fp16_to_f32(built.noised_part[j]) / af;
            let rel = (recovered - orig).abs() / orig.abs();
            max_rel = max_rel.max(rel);
        }
        // FP16 has 10 mantissa bits; one rounding plus the scale ulp stays below 2^-9.
        assert!(max_rel < 2f32.powi(-9), "max relative error {max_rel} exceeds FP16 accuracy");
    }

    /// Dumps `noisy_quantize` oracle vectors (inputs, pre-floor norms, scales,
    /// final codes) for the sm_80 miner-kernel bit-exactness harness. Writes to
    /// `$FP16_NQ_DUMP` (default `/tmp/fp16_noisy_quant_vectors.txt`). Run with:
    /// `cargo test --lib api::fp16::quantization::tests::dump_noisy_quant_vectors -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn dump_noisy_quant_vectors() {
        use std::fmt::Write as _;

        // Deterministic xorshift64* -> fp16 at a magnitude scale, with ~10% zeros.
        fn rng(state: &mut u64) -> f32 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            // uniform in [-1, 1)
            ((*state >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        }
        fn gen_fp16(state: &mut u64, n: usize, scale: f32) -> Vec<u16> {
            (0..n)
                .map(|_| {
                    let z = rng(state);
                    let v = if rng(state) < -0.8 { 0.0 } else { z * scale };
                    f32_to_fp16(v).unwrap()
                })
                .collect()
        }

        // (num_rows, k, r, scale): gemm-friendly (num_rows%16, k%8, r%16) so the
        // whole pipeline (incl. the E@F^T noise) runs on the sm_80 kernels.
        let cases = [
            (16usize, 8usize, 16usize, 0.25f32),
            (16, 16, 16, 2.0),
            (32, 8, 32, 8.0),
            (16, 64, 16, 1.0),
            (32, 24, 48, 4.0),
            (48, 40, 16, 16.0),
        ];
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut out = String::new();
        for (ci, &(num_rows, k, r, scale)) in cases.iter().enumerate() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(ci as u64 + 1);
            let rows = gen_fp16(&mut state, num_rows * k, scale);
            let e = gen_fp16(&mut state, num_rows * r, scale);
            let f = gen_fp16(&mut state, k * r, scale);
            let norms: Vec<(u16, u16)> =
                rows.chunks_exact(k).map(|row| row_norms(row).unwrap()).collect();
            let built = noisy_quantize(&rows, &e, &f, &norms, r).unwrap();

            writeln!(out, "CASE {num_rows} {k} {r}").unwrap();
            let line = |v: &[u16]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
            writeln!(out, "rows {}", line(&rows)).unwrap();
            writeln!(out, "e {}", line(&e)).unwrap();
            writeln!(out, "f {}", line(&f)).unwrap();
            let l2raw: Vec<u16> = norms.iter().map(|x| x.0).collect();
            let linfraw: Vec<u16> = norms.iter().map(|x| x.1).collect();
            writeln!(out, "l2raw {}", line(&l2raw)).unwrap();
            writeln!(out, "linfraw {}", line(&linfraw)).unwrap();
            writeln!(out, "alpha {}", line(&built.alpha)).unwrap();
            writeln!(out, "beta {}", line(&built.beta)).unwrap();
            writeln!(out, "l2 {}", line(&built.l2)).unwrap();
            writeln!(out, "noised {}", line(&built.noised_part)).unwrap();
        }
        let path = std::env::var("FP16_NQ_DUMP")
            .unwrap_or_else(|_| "/tmp/fp16_noisy_quant_vectors.txt".to_string());
        std::fs::write(&path, out).unwrap();
        eprintln!("wrote noisy-quant vectors to {path}");
    }

    #[test]
    fn noise_carries_relative_weight_near_delta() {
        // Build normalized FP16 noise lines; the per-row rms of beta*N should be
        // ~delta times the rms of alpha*X. Checked within a factor of 2 (the
        // scale derivation targets delta = 0.5).
        let k = 256;
        let rows: Vec<u16> = (0..k).map(|j| f32_to_fp16(((j * 37 % 97) as f32 - 48.0) * 0.5).unwrap()).collect();
        // Noise lines: random +-scale signs so E@F entries cancel randomly, giving
        // rms ~ NOISE_TARGET_NORM^2 / sqrt(r) as the scale derivation assumes.
        // A small xorshift keeps the test deterministic.
        let scale = (NOISE_TARGET_NORM as f32) / (R as f32).sqrt();
        let mut state = 0x2545F491_4F6CDD1Du64;
        let mut sign = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 { scale } else { -scale }
        };
        let e: Vec<u16> = (0..R).map(|_| f32_to_fp16(sign()).unwrap()).collect();
        let f: Vec<u16> = (0..k * R).map(|_| f32_to_fp16(sign()).unwrap()).collect();
        let norms = [row_norms(&rows).unwrap()];
        let built = noisy_quantize(&rows, &e, &f, &norms, R).unwrap();
        let af = bf16_to_f32(built.alpha[0]);
        let noise = a100_matmul(&e, &f, None, 1, k, R);
        let bf = bf16_to_f32(built.beta[0]);
        let sig_rms = (rows.iter().map(|&x| (af * fp16_to_f32(x)).powi(2)).sum::<f32>() / k as f32).sqrt();
        let noise_rms = (noise.iter().map(|&n| (bf * n).powi(2)).sum::<f32>() / k as f32).sqrt();
        let ratio = noise_rms / sig_rms;
        assert!((0.2..1.2).contains(&ratio), "noise/signal rms ratio {ratio} not near delta=0.5");
    }
}
