//! Deterministic low-rank noise factors for one matmul, mirroring the reference
//! miner (`miner_base.noise`).
//!
//! All factors are stacks of "lines" from one keyed-BLAKE3 rule
//! ([`sample_line`]). A line is `r` XOF bytes -> sign x UNIFORM magnitude in
//! `[1, 128]` (never zero, no modulo bias), L2-NORMALIZED to the constant norm
//! `c := NOISE_TARGET_NORM` (independent of `r`), then rounded to FP8 E4M3.
//! Normalization is exact integer math plus one BF16 division, hence
//! byte-identical on any host: `norm_scaled = isqrt(sumsq * INT_SQRT_PREC^2)`
//! (exact FLOOR integer sqrt, so `floor(||x||_2 * INT_SQRT_PREC)`), then
//! `scale = bf16(NOISE_TARGET_NORM * INT_SQRT_PREC) / bf16(norm_scaled)` (the
//! one BF16 rounding), `entry_i = fp8(bf16(x_i) * scale)`.
//!
//! A CONSTANT norm makes an `E@F` entry (`<e_row, f_col>` of two lines) have a
//! KNOWN peak (`~c^2`, Cauchy-Schwarz) and rms (`~c^2/sqrt(r)`), so the quant
//! scheme derives its per-row scales from `X`'s norms alone, never measuring
//! `E@F`. The exact procedure is chosen purely for cross-host bit-exactness;
//! the draw need not be Gaussian, only deterministic and public.
//!
//! Each line is random-access: the per-side noise seed is
//! subkeyed under `pearl/v4/FP8/noise-line`, and each line is addressed by a
//! `(side, factor, line)` tuple zero-padded to one 64-byte BLAKE3 block. `E`
//! lines use the selected global row/column index; `F` lines use `0..k` and
//! never vary by expert. The `F` basis is stored `k x r` row-major (the
//! transpose of the reference's `(r x k)` view), so line `i` is column `i` of
//! `F` — exactly the operand layout `B200::matmul_fp8` expects for `E @ F`.

use crate::api::fp8::compute::{bf16_div, bf16_mul};
use crate::api::fp8::dtype::{bf16_to_f32, f32_to_bf16, f32_to_fp8_e4m3};
use crate::api::fp8::public_params::PublicParams;
use crate::api::fp8::quantization::NOISE_TARGET_NORM;
use crate::api::fp8::transcript::{LABEL_NOISE_LINE, subkey};
use crate::api::primitives::{Hash256, IncompleteBlockHeader, Sides};

/// Fixed-point factor carrying `log2(32) = 5` fractional norm bits through the
/// exact integer `isqrt` (`_INT_SQRT_PREC` in the reference). It cancels in the
/// scale division; it only preserves precision in the floored square root.
const INT_SQRT_PREC: u64 = 32;

/// Deterministic per-matmul noise factors, one side per operand: `a` holds
/// `(e, f)` (A-side, `EA @ FA`), `b` holds `(e, f)` (B-side, `EB @ FB`).
pub type Noise = Sides<OperandNoise>;

/// One operand's paired noise factors: `e` (the row/col-keyed E-lines) and `f`
/// (the shared F basis), whose product `E @ F` is the injected noise.
pub struct OperandNoise {
    /// `(h x r)` or `(w x r)` E4M3 values, row-major.
    pub e: Vec<u8>,
    /// `(k x r)` E4M3 values, row-major (the reference's `F` transposed).
    pub f: Vec<u8>,
}

/// Which operand a noise line belongs to. The discriminants are the committed
/// wire bytes.
#[repr(u8)]
pub(crate) enum Side {
    A = 0,
    B = 1,
}

/// Which factor a noise line contributes to: the row/col-keyed `E` lines or the
/// shared `F` basis. The discriminants are the committed wire bytes.
#[repr(u8)]
pub(crate) enum NoiseFactor {
    E = 0,
    F = 1,
}

/// Draws one keyed, L2-normalized line of `rank` FP8 E4M3 entries.
///
/// Derives the line key as [`subkey`] of [`LABEL_NOISE_LINE`] under `seed`, then
/// keyed-BLAKE3-XOF's the address `side | factor | line(u32 LE)` — zero-padded to
/// a constant 64 bytes (one BLAKE3 block) — to `rank` output bytes and normalizes
/// them (see [`normalize_line`]). The caller resolves its `side` and the matching
/// `seed`.
pub(crate) fn sample_line(seed: &Hash256, side: Side, factor: NoiseFactor, line: u32, rank: u16) -> Vec<u8> {
    let key = subkey(LABEL_NOISE_LINE, Some(seed));
    let mut material = Vec::with_capacity(64);
    material.push(side as u8);
    material.push(factor as u8);
    material.extend_from_slice(&line.to_le_bytes());
    assert!(material.len() <= 64, "noise line material must fit one BLAKE3 block");
    material.resize(64, 0);

    let mut bytes = vec![0u8; usize::from(rank)];
    let mut hasher = blake3::Hasher::new_keyed(&key);
    hasher.update(&material);
    hasher.finalize_xof().fill(&mut bytes);

    normalize_line(&bytes)
}

/// Decodes `bytes` into a signed integer line and renormalizes it to L2 norm
/// [`NOISE_TARGET_NORM`], cast to FP8 E4M3.
///
/// `norm_scaled = floor(||x||_2 * INT_SQRT_PREC)` (exact integer `isqrt`), then
/// `scale = bf16(NOISE_TARGET_NORM * INT_SQRT_PREC) / bf16(norm_scaled)` (one
/// BF16 rounding), then `entry_i = fp8(bf16(x_i) * scale)`.
fn normalize_line(bytes: &[u8]) -> Vec<u8> {
    // Each `x_i` in ±[1, 128]: bit 7 the sign, `(b & 0x7F) + 1` the magnitude.
    // `x_i` is stored as i64 (only to carry the sign; the magnitude never
    // exceeds 128) so `x_i^2 <= 2^14`, and with `r <= u16::MAX` the sum of
    // squares stays within `u64` for the exact `.isqrt()` below.
    let x: Vec<i64> = bytes
        .iter()
        .map(|&b| {
            let sign = 1 - 2 * ((b >> 7) as i64); // +1 (bit 7 = 0) or -1
            let magnitude = ((b & 0x7F) as i64) + 1; // uniform in [1, 128], never 0
            sign * magnitude
        })
        .collect();

    let sumsq: u64 = x.iter().map(|&xi| (xi * xi) as u64).sum();
    let norm_scaled = (sumsq * (INT_SQRT_PREC * INT_SQRT_PREC)).isqrt();

    let numer = f32_to_bf16((NOISE_TARGET_NORM * INT_SQRT_PREC as f64) as f32).expect("8192 is representable");
    let denom = f32_to_bf16(norm_scaled as f32).expect("norm_scaled < 2^24 is representable");
    let scale = bf16_div(numer, denom).expect("noise-line scale is finite");

    x.iter()
        .map(|&xi| {
            let xb = f32_to_bf16(xi as f32).expect("|x_i| <= 128 is representable in bf16");
            let entry = bf16_mul(xb, scale).expect("noise entry is finite");
            f32_to_fp8_e4m3(bf16_to_f32(entry)).expect("noise entry is finite")
        })
        .collect()
}

/// Draws the four noise factors for one matmul instance.
///
/// `a_rows`/`b_cols` are the selected global row/column indices; the `E` lines
/// key off those indices, and the `F` basis is `0..k` for both sides.
/// `rank` is the peel rank `r`.
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

/// Derives this job's deterministic FP8 noise factors directly from the public
/// statement (`k`, `rank`, the per-side noise seeds and the selected global
/// row/column indices), without touching the compiled Blake program.
///
/// The `E` lines key off the selected global row/column indices (in MoE they are
/// the winner's `i_a` outer rows and the `expert_col + base + i` columns, so the
/// global address itself gives per-expert pair-uniqueness); the `F` basis is
/// `0..k` for both sides.
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
