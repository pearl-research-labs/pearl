//! Jackpot admissibility checks over the opened tile, using B200 replay.
//!
//! The tile has h rows, w columns and reduction dimension k. For either operand X,
//! `X_bar_iu = alpha_i * X_iu` is the scaled clean entry, and
//! `sigma_i^X = DELTA * alpha_i * l2_i` uses the grid-rounded, floored RMS in
//! [`BuiltRows`]. These products are exact; they are not BF16-rounded again.
//! A' and B' below are the rebuilt, noised FP8 operands.
//!
//! A tile passes only if all four checks pass. Counts cover the whole tile or
//! operand strip, rather than requiring each row to meet the same fraction.
//!
//! 1. **Entry liveness.** For each side X with n_X rows (`n_A = h`, `n_B = w`):
//!    `D_X = {(i,u): |X_bar_iu| >= tau_idle * sigma_i^X}` and
//!    `|D_X| <= eps_idle * n_X * k`.
//! 2. **Noise floor.** `sigma_i^X >= sigma_min` for every row of both sides.
//! 3. **Tamed products.** With c_ij,t the B200 accumulator after group t, define
//!    `M_ij = max(max_u |A'_iu * B'_ju|, max_t |c_ij,t|)` and
//!    `e_ij = floor(log2 M_ij)` for nonzero cells. Then
//!    `U = {(i,j): 2^e_ij > tau_tame * sqrt(k) * sigma_i^A * sigma_j^B}` must
//!    satisfy `|U| <= eps_tame * h * w`. Zero cells are tamed.
//! 4. **Unpredictable summands.** Let `S_X = X_bar_iu^2 + (sigma_i^X)^2`.
//!    [`lambda`] gives an integer lower bound on `64*(log2(S_X) + 508)`.
//!    For nonzero cells the implemented rule is
//!
//!    ```text
//!    E_ij = e_ij + 139
//!    skip(i,j,u) iff lambda_A(i,u) + lambda_B(j,u) < 128*E_ij + 45376
//!    sum_{i,j,u} skip(i,j,u) <= eps_pred * k * h * w
//!    ```
//!
//!    Zero cells never skip. The rule approximates the real-valued test
//!    `v_iju = S_A*S_B / 2^21 < ulp(M_ij)^2`, where
//!    `ulp(M_ij)^2 = 2^(2*(e_ij - 25))`. Downward score rounding can add skips;
//!    every ideal skip is counted, and a counted skip satisfies
//!    `v_iju < ulp(M_ij)^2 * 2^(2.2/64)`. The integer rule defines acceptance;
//!    [`unpredictability`](crate::circuit::fp8::unpredictability) derives its constants and error bound.
//!
//! Call after rejecting non-finite inputs. [`PublicParams`](crate::api::fp8::public_params::PublicParams)
//! bounds `k <= 2^16`.

use crate::api::fp8::dtype::{bf16_to_f32, fp8_e4m3_to_f32};
use crate::api::fp8::quantization::{BuiltRows, DELTA};
use crate::api::fp8::utils::{B200, xor_fold_extract};
use crate::api::layout::{AxisPattern, JACKPOT_ENTRIES, lane_assignment};
use crate::circuit::fp8::unpredictability::{cell_exponent, lambda, skip};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Jackpot thresholds; defaults are defined in [`Self::default`].
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct JackpotPolicy {
    /// Liveness threshold (check 1): entry `u` of row `i` is dead iff
    /// `|X_bar_iu| >= tau_idle * sigma_i`.
    pub tau_idle: f64,
    /// Max fraction of dead entries per tile side (check 1).
    pub eps_idle: f64,
    /// Floor on every row's injected noise std `sigma_i` (check 2).
    pub sigma_min: f64,
    /// Tamed-products threshold (check 3): cell `(i,j)` is untamed iff
    /// `2^(floor(log2 M_ij)) > tau_tame * sqrt(k) * sigma_i^A * sigma_j^B`
    /// ([`untamed_exact`]). `tau_tame^2` must be an exact integer so the
    /// predicate can be decided in integers ([`Self::tau_sq_k`]).
    pub tau_tame: f64,
    /// Max fraction of untamed cells over the whole tile (check 3).
    pub eps_tame: f64,
    /// Max fraction of skippable summands over the whole tile (check 4).
    pub eps_pred: f64,
}

impl Default for JackpotPolicy {
    fn default() -> Self {
        Self {
            tau_idle: 8.0,
            eps_idle: 0.015625, // 1/64
            sigma_min: 1.0,
            tau_tame: 256.0,    // tau_tame^2 = 2^16, folded into the ZK XFPOW2 key
            eps_tame: 0.015625, // 1/64
            eps_pred: 0.0625,   // 1/16
        }
    }
}

/// One operand strip of the opened tile: the clean BF16 rows (`rows x k`,
/// row-major) and the rebuilt noised operand (`A'`/`B'` codes + per-row
/// `alpha`/`l2`).
pub struct OperandStrip {
    pub clean: Vec<u16>,
    pub built: BuiltRows,
}

/// 64-byte lottery message from `xor_fold_extract(A' @ B'.T, lanes)`,
/// hashed under the jackpot key to produce `hash_jackpot`.
pub type JackpotMessage = [u8; 4 * JACKPOT_ENTRIES];

/// Per-row noise stds `sigma_i = DELTA * alpha_i * l2_i` (in quantized units),
/// from the floored norms and scales recorded at quantization time.
fn sigmas(built: &BuiltRows) -> Vec<f64> {
    built
        .alpha
        .iter()
        .zip(&built.l2)
        .map(|(&alpha, &l2)| DELTA * bf16_to_f32(alpha) as f64 * bf16_to_f32(l2) as f64)
        .collect()
}

/// Exact noise-scale representation: `sigma = (-1)^negative * sig * 2^exp`.
/// `sig < 2^16`; zero has `sig = 0` and `negative = false`.
#[derive(Copy, Clone)]
struct SigmaPow {
    sig: u64,
    exp: i64,
    negative: bool,
}

/// Decode finite BF16 as `(negative, sig, exp)` with
/// `value = (-1)^negative * sig * 2^exp` and `sig <= 255`.
fn bf16_sig_exp(bits: u16) -> (bool, u64, i64) {
    debug_assert!(bits & 0x7FFF < 0x7F80, "finite bf16 expected");
    let exp_field = ((bits >> 7) & 0xFF) as i64;
    let frac = (bits & 0x7F) as u64;
    let (sig, exp) = if exp_field == 0 {
        (frac, -133) // subnormal: frac * 2^(1 - 127 - 7)
    } else {
        (128 + frac, exp_field - 134) // (2^7 + frac) * 2^(field - 127 - 7)
    };
    (bits & 0x8000 != 0, sig, exp)
}

/// Exact decode of a finite nonnegative f32: `x = sig * 2^exp`, with
/// `sig = 0` iff `x = 0`. `sig < 2^24`.
fn f32_sig_exp(x: f32) -> (u64, i64) {
    let bits = x.to_bits();
    let exp_field = ((bits >> 23) & 0xFF) as i64;
    let frac = (bits & 0x007F_FFFF) as u64;
    if exp_field == 0 {
        (frac, -149) // subnormal: frac * 2^(1 - 127 - 23)
    } else {
        ((1 << 23) | frac, exp_field - 150) // (2^23 + frac) * 2^(field - 127 - 23)
    }
}

/// Exact counterpart of [`sigmas`]: multiply the BF16 significands and
/// fold `DELTA = 0.5` into the exponent.
/// Returns per-row sigmas as integer significands and binary exponents,
/// for the exact integer comparison in `untamed_exact`.
fn sigma_pows(built: &BuiltRows) -> Vec<SigmaPow> {
    built
        .alpha
        .iter()
        .zip(&built.l2)
        .map(|(&alpha, &l2)| {
            let (a_neg, a_sig, a_exp) = bf16_sig_exp(alpha);
            let (l_neg, l_sig, l_exp) = bf16_sig_exp(l2);
            let sig = a_sig * l_sig;
            SigmaPow {
                sig,
                exp: a_exp + l_exp - 1,
                negative: sig != 0 && a_neg != l_neg,
            }
        })
        .collect()
}

/// Check `2^floor(log2 m) > tau_tame * sqrt(k) * sigma_i * sigma_j`
/// for finite nonnegative `m`, using integer arithmetic.
///
/// After handling zero and negative bounds, squaring and collecting powers of two gives
///
/// ```text
/// e_m = floor(log2 m)
/// g   = 2*e_m - 2*(si.exp + sj.exp)
/// rhs = tau_sq_k * (si.sig * sj.sig)^2
/// untamed iff 2^g > rhs iff g >= bitlen(rhs)    (rhs > 0)
/// ```
///
/// Since `tau_sq_k <= 2^20 * 2^16` and each sigma significand is below `2^16`,
/// `rhs < 2^36 * (2^32)^2 = 2^100`, so the comparison fits u128.
///
/// For a positive bound B, `2^e_m <= m < 2^(e_m+1)` gives
/// `m > 2*B => untamed => m > B`. Equality `2^e_m = B` is tamed.
fn untamed_exact(m: f32, tau_sq_k: u64, si: SigmaPow, sj: SigmaPow) -> bool {
    debug_assert!(m.is_finite() && m >= 0.0);
    let (m_significand, m_exponent) = f32_sig_exp(m);
    let sigma_product = si.sig * sj.sig; // < 2^32
    if tau_sq_k == 0 || sigma_product == 0 {
        return m_significand != 0; // bound = 0: untamed iff m > 0
    }
    if si.negative != sj.negative {
        return true; // bound < 0 <= m
    }
    if m_significand == 0 {
        return false; // m = 0, bound > 0
    }
    // Bit length accounts for the unnormalized significand of subnormals.
    let m_binade = m_exponent + (64 - i64::from(m_significand.leading_zeros())) - 1;
    let rhs = (tau_sq_k as u128) * (sigma_product as u128) * (sigma_product as u128);
    // 2^g > rhs <=> g >= bitlen(rhs) (rhs > 0 here).
    let threshold_exponent = 2 * m_binade - 2 * (si.exp + sj.exp);
    threshold_exponent >= i64::from(128 - rhs.leading_zeros())
}

impl JackpotPolicy {
    /// Return `Ok(None)` if a policy check rejects the tile; otherwise return
    /// its 64-byte lottery message. `a`/`b` hold clean and rebuilt operands;
    /// the patterns determine the fold lanes. [`B200`] supplies replay partials.
    pub fn evaluate(
        &self,
        a: &OperandStrip,
        b: &OperandStrip,
        k: usize,
        rows_pattern: &AxisPattern,
        cols_pattern: &AxisPattern,
    ) -> Result<Option<JackpotMessage>> {
        let Some(replayed_tile) = self.run_checks(a, b, k)? else {
            return Ok(None);
        };
        let lanes = lane_assignment(rows_pattern, cols_pattern);
        Ok(Some(xor_fold_extract(&replayed_tile, &lanes)))
    }

    /// Jackpot policy hecks 1-4, returning the bit-exact replayed first-stage accumulator
    /// `A' @ B'.T` (row-major, one value per tile cell) on success, `None` on
    /// rejection.
    fn run_checks(&self, a: &OperandStrip, b: &OperandStrip, k: usize) -> Result<Option<Vec<f32>>> {
        // The caller supplies clean and rebuilt operands with matching dimensions.
        debug_assert!(k > 0, "k must be positive");
        for (name, side) in [("A", a), ("B", b)] {
            debug_assert!(
                side.clean.len() == side.built.noised_part.len(),
                "mismatch in length of clean and noised {name}"
            );
            debug_assert!(side.clean.len().is_multiple_of(k), "length of {name} must be a multiple of k");
            let rows = side.clean.len() / k;
            debug_assert!(
                side.built.alpha.len() == rows && side.built.l2.len() == rows,
                "one (alpha, l2) per {name} row"
            );
        }

        // Check 1: entry liveness, per side.
        if !self.liveness_ok(a) || !self.liveness_ok(b) {
            return Ok(None);
        }

        let sigma_a = sigmas(&a.built);
        let sigma_b = sigmas(&b.built);

        // Check 2: noise floor, sigma_i >= sigma_min on both sides.
        if sigma_a.iter().chain(&sigma_b).any(|&s| s < self.sigma_min) {
            return Ok(None);
        }

        // Checks 3 and 4: tamed products and unpredictable summands, sharing one pass over the tile.
        self.tamed_and_unpredictable_ok(a, b, k, sigma_a.len(), sigma_b.len())
    }

    /// Check 1: the fraction of dead entries over all of the side's tile
    /// entries is at most `eps_idle`. Entry `u` of row `i` is dead iff
    /// `|X_bar_iu| >= tau_idle * sigma_i`. Since `X_bar_iu = alpha_i * X_iu`
    /// and `sigma_i = DELTA * alpha_i * l2_i` (with `alpha_i > 0`), this is
    /// `alpha`-free: `|X_iu| >= tau_idle * DELTA * l2_i`.
    fn liveness_ok(&self, side: &OperandStrip) -> bool {
        let clean = &side.clean;
        let l2 = &side.built.l2;
        let k = clean.len() / l2.len();
        let dead_entries: usize = clean
            .chunks(k)
            .zip(l2)
            .map(|(row, &l2)| {
                let dead_threshold = self.tau_idle * DELTA * bf16_to_f32(l2) as f64;
                row.iter().filter(|&&x| bf16_to_f32(x).abs() as f64 >= dead_threshold).count()
            })
            .sum();
        dead_entries as f64 <= self.eps_idle * clean.len() as f64
    }

    /// Integer threshold for [`untamed_exact`]. Require `tau_tame^2` in
    /// `[0, 2^20]` and `k <= 2^16` ([`PublicParams`](crate::api::fp8::public_params::PublicParams))
    /// so its u128 arithmetic fits. The AIR folds default `tau_tame^2 = 2^16`
    /// into the XFPOW2 key's shift.
    fn tau_sq_k(&self, k: usize) -> Result<u64> {
        let tau_sq = self.tau_tame * self.tau_tame;
        ensure!(
            tau_sq.fract() == 0.0 && (0.0..=(1u64 << 20) as f64).contains(&tau_sq),
            "tau_tame^2 must be an exact integer in [0, 2^20], got tau_tame = {}",
            self.tau_tame
        );
        ensure!(k <= 1 << 16, "k must be at most 2^16, got {k}");
        Ok(tau_sq as u64 * k as u64)
    }

    /// Checks 3 (tamed products) and 4 (unpredictable summands). Both consume
    /// the replay magnitude `M_ij`, so they share one pass over the tile.
    /// Returns the replay's final per-cell accumulator on success.
    fn tamed_and_unpredictable_ok(
        &self,
        a: &OperandStrip,
        b: &OperandStrip,
        k: usize,
        h: usize,
        w: usize,
    ) -> Result<Option<Vec<f32>>> {
        let partials = B200 {}.matmul_fp8_partials(&a.built.noised_part, &b.built.noised_part, h, w, k)?;

        let noised_a: Vec<f32> = a.built.noised_part.iter().map(|&c| fp8_e4m3_to_f32(c)).collect();
        let noised_b: Vec<f32> = b.built.noised_part.iter().map(|&c| fp8_e4m3_to_f32(c)).collect();

        // Scores match the `lambda` values used by the InputQuant and Matmul tables.
        let summand_scores = |side: &OperandStrip| -> Vec<u64> {
            side.clean
                .iter()
                .enumerate()
                .map(|(idx, &x)| lambda(side.built.alpha[idx / k], x, side.built.l2[idx / k]))
                .collect()
        };
        let scores_a = summand_scores(a);
        let scores_b = summand_scores(b);

        let tau_sq_k = self.tau_sq_k(k)?;
        let sigma_pow_a = sigma_pows(&a.built);
        let sigma_pow_b = sigma_pows(&b.built);
        let mut untamed_cells: u64 = 0;

        let mut skippable_summands: u64 = 0;

        for i in 0..h {
            for j in 0..w {
                // Maximum absolute product or partial sum for this output cell.
                let mut max_magnitude = 0.0f32;
                for u in 0..k {
                    max_magnitude = max_magnitude.max((noised_a[i * k + u] * noised_b[j * k + u]).abs());
                }
                for &c in &partials[i * w + j] {
                    max_magnitude = max_magnitude.max(c.abs());
                }

                // Check 3: tamed products, decided exactly in integers.
                if untamed_exact(max_magnitude, tau_sq_k, sigma_pow_a[i], sigma_pow_b[j]) {
                    untamed_cells += 1;
                }

                // Check 4: skippable summands.
                let magnitude_exponent = cell_exponent(f64::from(max_magnitude));
                for u in 0..k {
                    if skip(scores_a[i * k + u], scores_b[j * k + u], magnitude_exponent) {
                        skippable_summands += 1;
                    }
                }
            }
        }

        let tile_cells = (h * w) as f64;
        if untamed_cells as f64 > self.eps_tame * tile_cells {
            return Ok(None);
        }
        if skippable_summands as f64 > self.eps_pred * k as f64 * tile_cells {
            return Ok(None);
        }

        // Each cell's last partial is its final accumulator value (pinned by
        // `matmul_fp8_partials_chain_the_group_accumulator`).
        Ok(Some(
            partials
                .iter()
                .map(|cell| *cell.last().expect("k > 0 yields at least one partial per cell"))
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fp8::noise::OperandNoise;
    use crate::api::fp8::quantization::{Fp8E4M3Quant, Quant};

    fn side(clean: Vec<u16>, built: BuiltRows) -> OperandStrip {
        OperandStrip { clean, built }
    }

    /// Build one side from clean rows with real norms, scales, and noise.
    fn build(rows: &[u16], k: usize, r: usize, seed: u8) -> (Vec<u16>, BuiltRows) {
        let num_rows = rows.len() / k;
        // Deterministic mesh-like FP8 noise factors keyed off `seed`.
        let e: Vec<u8> = (0..num_rows * r)
            .map(|i| if (i + seed as usize).is_multiple_of(3) { 0xB0 } else { 0x30 })
            .collect();
        let f: Vec<u8> = (0..k * r)
            .map(|i| if (i + seed as usize).is_multiple_of(5) { 0xB0 } else { 0x30 })
            .collect();
        let noise = OperandNoise { e, f };
        let norms: Vec<(u16, u16)> = rows.chunks(k).map(|row| Fp8E4M3Quant.row_norms(row).unwrap()).collect();
        let built = Fp8E4M3Quant.noisy_quantize(rows, &noise, &norms).unwrap();
        (rows.to_vec(), built)
    }

    /// Varied, well-conditioned clean rows (values in [-2, 2], no outliers).
    fn honest_rows(num_rows: usize, k: usize) -> Vec<u16> {
        let vals = [0x3F80u16, 0xBF00, 0x3E80, 0xBFC0, 0x3F00, 0x4000, 0xBF80, 0x3D80];
        (0..num_rows * k).map(|i| vals[(i * 7 + i / k) % vals.len()]).collect()
    }

    #[test]
    fn honest_tile_is_admissible_with_default_thresholds() {
        let (k, r) = (64usize, 16usize);
        let (a, a_built) = build(&honest_rows(2, k), k, r, 0);
        let (b, b_built) = build(&honest_rows(3, k), k, r, 1);
        let verdict = JackpotPolicy::default()
            .run_checks(&side(a, a_built), &side(b, b_built), k)
            .unwrap();
        assert!(verdict.is_some());
    }

    #[test]
    fn sigma_floor_rejects_rows_with_vanishing_noise_scale() {
        let (k, r) = (64usize, 16usize);
        let (a, mut a_built) = build(&honest_rows(2, k), k, r, 0);
        let (b, b_built) = build(&honest_rows(2, k), k, r, 1);
        // Force row 0's recorded l2 to a tiny value: sigma_0 = DELTA * alpha_0
        // * l2_0 collapses below sigma_min = 1.
        a_built.l2[0] = 0x2F80; // 2^-32
        let verdict = JackpotPolicy::default()
            .run_checks(&side(a, a_built), &side(b, b_built), k)
            .unwrap();
        assert!(verdict.is_none(), "sigma below sigma_min must reject");
    }

    #[test]
    fn liveness_rejects_spike_dominated_rows() {
        let (k, r) = (16usize, 16usize);
        // One unit spike per row gives RMS sqrt(1/16) = 0.25. The spike is dead
        // at equality: 1 = tau_idle*DELTA*0.25 = 8*0.5*0.25.
        // Two dead entries out of 32 exceed eps_idle = 1/64.
        let mut a_rows = vec![0u16; 2 * k];
        a_rows[0] = 0x3F80;
        a_rows[k] = 0x3F80;
        let (a, a_built) = build(&a_rows, k, r, 0);
        let (b, b_built) = build(&honest_rows(2, k), k, r, 1);
        // Disable the sigma floor so the liveness check is what rejects (the
        // zero-heavy row also has a tiny sigma).
        let policy = JackpotPolicy {
            sigma_min: 0.0,
            ..JackpotPolicy::default()
        };
        let verdict = policy.run_checks(&side(a, a_built), &side(b, b_built), k).unwrap();
        assert!(verdict.is_none(), "dead fraction above eps_idle must reject");
    }

    #[test]
    fn unpredictability_rejects_when_summands_are_predictable() {
        // A spike creates a large replay magnitude. Lowering the recorded RMS
        // makes the zero entries' scores small enough to be skippable.
        let (k, r) = (1024usize, 16usize);
        let mut rows = vec![0u16; 2 * k];
        rows[0] = 0x3F80; // 1.0
        rows[k] = 0x3F80;
        let (a, mut a_built) = build(&rows, k, r, 0);
        let (b, mut b_built) = build(&rows, k, r, 1);
        for l2 in a_built.l2.iter_mut().chain(b_built.l2.iter_mut()) {
            *l2 = 0x2F80; // bf16 code for 2^-32
        }
        // Neutralize checks 1-3 to isolate check 4 (the tiny sigmas trip the
        // sigma floor and make the spiky cells untamed too).
        let policy = JackpotPolicy {
            eps_idle: 1.0,
            sigma_min: 0.0,
            eps_tame: 1.0,
            ..JackpotPolicy::default()
        };
        let verdict = policy.run_checks(&side(a, a_built), &side(b, b_built), k).unwrap();
        assert!(verdict.is_none(), "predictable summands must reject");
    }

    #[test]
    fn tamed_products_rejects_coherent_tiles() {
        // Lower the recorded RMS so the replay products exceed the tamed bound.
        let (k, r) = (32usize, 16usize);
        let (a, mut a_built) = build(&honest_rows(2, k), k, r, 0);
        let (b, mut b_built) = build(&honest_rows(2, k), k, r, 1);
        for l2 in a_built.l2.iter_mut().chain(b_built.l2.iter_mut()) {
            *l2 = 0x2F80; // bf16 code for 2^-32
        }
        // Neutralize the other checks to isolate check 3.
        let policy = JackpotPolicy {
            eps_idle: 1.0,
            sigma_min: 0.0,
            eps_pred: 1.0,
            ..JackpotPolicy::default()
        };
        let verdict = policy.run_checks(&side(a, a_built), &side(b, b_built), k).unwrap();
        assert!(verdict.is_none(), "tiny sigma makes every live cell untamed");
    }

    /// `SigmaPow` for a single row with the given bf16 `alpha`/`l2` codes.
    fn sigma_pow(alpha: u16, l2: u16) -> SigmaPow {
        sigma_pows(&BuiltRows {
            noised_part: vec![],
            alpha: vec![alpha],
            beta: vec![0],
            l2: vec![l2],
        })[0]
    }

    #[test]
    fn untamed_exact_flags_whole_binades() {
        // sigma = 1, k = 1024 and tau = 100 give bound 3200.
        // The first untamed binade starts at 4096.
        let s = sigma_pow(0x4000, 0x3F80);
        let policy = JackpotPolicy {
            tau_tame: 100.0,
            ..JackpotPolicy::default()
        };
        let tau_sq_k = policy.tau_sq_k(1024).unwrap();
        assert!(!untamed_exact(f32::from_bits(4096f32.to_bits() - 1), tau_sq_k, s, s));
        assert!(untamed_exact(4096.0, tau_sq_k, s, s), "the binade boundary is untamed");
        // Test equality and the factor-two bound.
        assert!(!untamed_exact(3200.0, tau_sq_k, s, s));
        assert!(untamed_exact(6400.0, tau_sq_k, s, s));
    }

    #[test]
    fn untamed_exact_is_strict_at_power_of_two_ties() {
        // sigma = 1, k = 1024 and tau = 256 give bound 8192 = 2^13.
        // The strict comparison accepts [2^13, 2^14) and rejects [2^14, 2^15).
        let s = sigma_pow(0x4000, 0x3F80);
        let tau_sq_k = JackpotPolicy::default().tau_sq_k(1024).unwrap();
        assert!(
            !untamed_exact(8192.0, tau_sq_k, s, s),
            "m == bound == 2^13 is tamed (2^13 > 2^13 is false)"
        );
        assert!(!untamed_exact(f32::from_bits(16384f32.to_bits() - 1), tau_sq_k, s, s));
        assert!(untamed_exact(16384.0, tau_sq_k, s, s));
    }

    #[test]
    fn untamed_exact_handles_zero_and_extreme_magnitudes() {
        let s = sigma_pow(0x4000, 0x3F80); // sigma = 1
        let zero = sigma_pow(0x4000, 0); // l2 = 0 => sigma = 0
        let tau_sq_k = JackpotPolicy::default().tau_sq_k(1024).unwrap();
        // Zero bound: untamed iff m > 0; the smallest subnormal qualifies.
        assert!(!untamed_exact(0.0, tau_sq_k, zero, s));
        assert!(untamed_exact(f32::from_bits(1), tau_sq_k, zero, s));
        // Zero m against a positive bound is tamed.
        assert!(!untamed_exact(0.0, tau_sq_k, s, s));
        // Far-apart magnitudes exercise the bit-length fast path on both sides.
        assert!(untamed_exact(f32::MAX, tau_sq_k, s, s));
        assert!(!untamed_exact(f32::from_bits(1), tau_sq_k, s, s));
        // Tiny sigmas (subnormal-adjacent bf16 codes) with a large m.
        let tiny = sigma_pow(0x0001, 0x0001); // both subnormal significands
        assert!(untamed_exact(1.0, tau_sq_k, tiny, tiny));
    }

    #[test]
    fn untamed_exact_matches_f64_definition_away_from_ties() {
        // Differential check against the definitional f64 evaluation
        // `ufp(m) = 2^floor(log2 m) > tau * sqrt(k) * sigma_i * sigma_j` on
        // pseudo-random operands, skipping near-ties where f64 rounding of
        // the bound may legitimately disagree.
        let policy = JackpotPolicy::default();
        let k = 4096usize;
        let tau_sq_k = policy.tau_sq_k(k).unwrap();
        let sqrt_k = (k as f64).sqrt();
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut checked = 0u32;
        for _ in 0..20_000 {
            let r = next();
            // Positive finite bf16 codes (exponent fields 1..=254).
            let bf16 = |r: u16| ((1 + (r >> 7) % 254) << 7) | (r & 0x7F);
            let alpha_i = bf16(r as u16);
            let l2_i = bf16((r >> 16) as u16);
            let alpha_j = bf16((r >> 32) as u16);
            let l2_j = bf16((r >> 48) as u16);
            // Nonnegative finite f32 m near the sigma product's scale, so
            // verdicts are non-degenerate both ways.
            let si = DELTA * bf16_to_f32(alpha_i) as f64 * bf16_to_f32(l2_i) as f64;
            let sj = DELTA * bf16_to_f32(alpha_j) as f64 * bf16_to_f32(l2_j) as f64;
            let bound = policy.tau_tame * sqrt_k * si * sj;
            let m = (bound * (0.5 + (next() % 1024) as f64 / 512.0)) as f32;
            if !m.is_finite() || m == 0.0 || bound < f64::MIN_POSITIVE || bound > f64::MAX / 4.0 {
                continue;
            }
            // Exact binade of m from the raw bits (subnormal-safe), then the
            // definitional comparison with ufp(m) = 2^e_m exact in f64.
            let e_m = {
                let bits = m.to_bits() & 0x7FFF_FFFF;
                let exp_field = (bits >> 23) as i32;
                if exp_field != 0 {
                    exp_field - 127
                } else {
                    (32 - (bits & 0x7F_FFFF).leading_zeros()) as i32 - 150
                }
            };
            let ufp = 2f64.powi(e_m);
            // Skip near-ties: f64 evaluates the bound to ~0.5 ulp, so only
            // trust it when ufp is clearly on one side.
            if (ufp - bound).abs() < bound * 1e-9 {
                continue;
            }
            checked += 1;
            assert_eq!(
                untamed_exact(m, tau_sq_k, sigma_pow(alpha_i, l2_i), sigma_pow(alpha_j, l2_j)),
                ufp > bound,
                "m = {m:e} (ufp = {ufp:e}), bound = {bound:e}"
            );
        }
        assert!(checked > 10_000, "differential test degenerated: {checked} cases");
    }

    /// Binade-based flags lie between magnitude-based flags at thresholds
    /// `tau` and `2*tau`. Both comparisons use exact integer arithmetic.
    #[test]
    fn ufp_flags_are_subset_of_magnitude_flags_within_factor_two_band() {
        /// Exact magnitude comparison `m > tau * sqrt(k) * si * sj`, using squares.
        fn magnitude_untamed(m: f32, tau_sq_k: u64, si: SigmaPow, sj: SigmaPow) -> bool {
            let (mm, em) = f32_sig_exp(m);
            let pp = si.sig * sj.sig;
            if tau_sq_k == 0 || pp == 0 {
                return mm != 0;
            }
            if si.negative != sj.negative {
                return true;
            }
            if mm == 0 {
                return false;
            }
            let lhs = (mm as u128) * (mm as u128);
            let rhs = (tau_sq_k as u128) * (pp as u128) * (pp as u128);
            let shift = 2 * (em - (si.exp + sj.exp));
            let bits = |x: u128| 128 - i64::from(x.leading_zeros());
            let (lhs_bits, rhs_bits) = (bits(lhs) + shift, bits(rhs));
            if lhs_bits != rhs_bits {
                return lhs_bits > rhs_bits;
            }
            if shift >= 0 {
                (lhs << shift) > rhs
            } else {
                lhs > (rhs << -shift)
            }
        }

        let policy = JackpotPolicy::default();
        let k = 1024usize;
        let tau_sq_k = policy.tau_sq_k(k).unwrap(); // 2^16 * 1024 = 2^26
        let double_tau_sq_k = tau_sq_k * 4; // (2*tau)^2 * k
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (mut flagged, mut mag_flagged) = (0u32, 0u32);
        for _ in 0..20_000 {
            let r = next();
            let bf16 = |r: u16| ((1 + (r >> 7) % 254) << 7) | (r & 0x7F);
            let si = sigma_pow(bf16(r as u16), bf16((r >> 16) as u16));
            let sj = sigma_pow(bf16((r >> 32) as u16), bf16((r >> 48) as u16));
            let scale = DELTA
                * bf16_to_f32(bf16(r as u16)) as f64
                * bf16_to_f32(bf16((r >> 16) as u16)) as f64
                * DELTA
                * bf16_to_f32(bf16((r >> 32) as u16)) as f64
                * bf16_to_f32(bf16((r >> 48) as u16)) as f64
                * policy.tau_tame
                * (k as f64).sqrt();
            let m = (scale * (0.25 + (next() % 2048) as f64 / 512.0)) as f32;
            if !m.is_finite() {
                continue;
            }
            let new = untamed_exact(m, tau_sq_k, si, sj);
            let mag = magnitude_untamed(m, tau_sq_k, si, sj);
            flagged += new as u32;
            mag_flagged += mag as u32;
            if new {
                assert!(mag, "a flagged cell must be magnitude-untamed: m = {m:e}");
            }
            if magnitude_untamed(m, double_tau_sq_k, si, sj) {
                assert!(new, "a cell untamed at 2*tau must stay flagged: m = {m:e}");
            }
        }
        // The m range straddles the threshold, so both predicates fire often
        // and differ on a nontrivial sliver (the coarsening band).
        assert!(
            flagged > 1_000 && mag_flagged > flagged,
            "degenerate: {mag_flagged} vs {flagged}"
        );
    }

    #[test]
    fn tau_sq_k_rejects_non_integer_squares() {
        let mut policy = JackpotPolicy::default();
        assert_eq!(policy.tau_sq_k(1024).unwrap(), 256 * 256 * 1024);
        policy.tau_tame = 1.5; // 2.25: not an integer square
        assert!(policy.tau_sq_k(1024).is_err());
        policy.tau_tame = 256.0;
        assert!(policy.tau_sq_k(1 << 17).is_err(), "k above 2^16 must error");
    }
}
