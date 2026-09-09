//! The tile-level jackpot policy.
//!
//! The verifier replays the winning tile bit-exactly on the committed device's
//! FP8 MMA semantics — that replay is the ground truth. On top of it, this
//! policy certifies that producing the tile cost `~tile_elems * k` fresh
//! multiply-adds. Design rules: (i) prefer checks that depend only on clean
//! data + public constants ("x-only"), so a committed matrix passes or fails
//! identically for every noise draw and grinding gains nothing; (ii) count
//! density-type violations over the WHOLE lottery tile, so honest counts
//! concentrate around their mean and never fail on tail fluctuations.
//!
//! Notation. Per-row noise scales, in quantized units:
//! `sigma_i = delta * l2(A_bar_i) = DELTA * alpha_i * l2_i`. Scaled clean
//! entries: `A_bar_iu = alpha_i * X_iu`, `B_bar_ju = alpha_j' * X_ju`.
//!
//! The policy approves a single opened tile iff every check below passes. It
//! runs after the verifier has rejected Infinity/NaN inputs on intermediate
//! results.
//!
//! 1. **Entry liveness.** `|D_X| <= eps_idle * |I_X| * k` per side, where
//!    `D_X = {(i,u): |X_bar_iu| >= tau_idle * sigma_i^X}` (dead entries).
//! 2. **Noise floor.** `sigma_i^X >= sigma_min` for every row of both sides.
//! 3. **Tamed products.** `|U| <= eps_tame * |I_A| * |I_B|`, where
//!    `U = {(i,j): 2^(e_ij) > tau_tame * sqrt(k) * sigma_i^A * sigma_j^B}`,
//!    `e_ij = floor(log2 M_ij)` is the binade of the replay magnitude and
//!    `M_ij = max{max_u |A_tilde_iu * B_tilde_ju|, max_t |c_ij,t|}`.
//! 4. **Unpredictable summands.** `|S| <= eps_pred * k * |I_A| * |I_B|`, where
//!    `S` collects the summands `(i,j,u)` with `v_iju < ulp(M_ij)^2`, for
//!    `v_iju = (A_bar_iu^2 + (sigma_i^A)^2)(B_bar_ju^2 + (sigma_j^B)^2) / 2^21`
//!    and `ulp(x)^2 = 2^(2*(floor(log2 x) - W))`, `W = 25`. Decided exactly in
//!    integers by taking log2 of the defining inequality, at fixed-point
//!    resolution `1/64`: each summand half is scored once
//!    (`unpredictability::lambda`) with a one-sided lower bound
//!    `lambda_X ~= 64 * log2(X_bar_iu^2 + (sigma_i^X)^2)` -- never above
//!    the true value, less than `1.1` steps below it -- and `(i,j,u)`
//!    is skippable iff
//!    `lambda_A + lambda_B < 64 * log2(2^21 * ulp(M_ij)^2)`.
//!
//! The verdict is a BINARY gate (rejected / accepted): a `D`-dependent credit
//! would incentivize shaping data toward incoherence.
//!
//! There is no policy-internal `k` cap: [`PublicParams`](crate::api::fp8::public_params::PublicParams)
//! construction bounds `k <= 2^16` upstream; check 4's window `W = 25` is
//! slack at any practical `k`.

use crate::api::fp8::dtype::{bf16_to_f32, fp8_e4m3_to_f32};
use crate::api::fp8::quantization::{BuiltRows, DELTA};
use crate::api::fp8::utils::{B200, xor_fold_extract};
use crate::api::layout::{AxisPattern, JACKPOT_ENTRIES, lane_assignment};
use crate::circuit::fp8::unpredictability::{cell_exponent, lambda, skip};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Consensus-tunable thresholds, with default values `tau_idle=8, eps_idle=1/64,
/// sigma_min=1, tau_tame=256, eps_tame=1/64, eps_pred=1/16`.
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

/// The lottery message: `xor_fold_extract(A' @ B'.T, lane_assignment(rows_pattern,
/// cols_pattern))` — 64 bytes, one BLAKE3 block, the preimage hashed under
/// `pow_key` to form `hash_jackpot`.
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

/// A row's noise std factored exactly: `sigma = (-1)^negative * sig * 2^exp`,
/// with `sig = 0` iff `sigma = 0` (`negative` is then false). `sig < 2^16`
/// (product of two bf16 significands).
#[derive(Copy, Clone)]
struct SigmaPow {
    sig: u64,
    exp: i64,
    negative: bool,
}

/// Exact bf16 decode: `value = (-1)^negative * sig * 2^exp` with `sig = 0`
/// iff the value is zero, as `(negative, sig, exp)`. `sig <= 255` (8-bit
/// significand). Requires a finite encoding.
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

/// Exact counterpart of [`sigmas`]: `sigma_i = DELTA * alpha_i * l2_i` with
/// the bf16 significands multiplied as integers and `DELTA = 0.5` folded
/// into the exponent.
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

/// Check 3's untamed predicate, decided exactly over
/// the reals: `ufp(m) > tau_tame * sqrt(k) * sigma_i * sigma_j`, where
/// `ufp(m) = 2^(e_m)`, `e_m = floor(log2 m)`, is the largest power of two not
/// exceeding the replay magnitude, via comparing squares
///
/// ```text
/// 2^(2*e_m)  >  tau_sq_k * (si.sig * sj.sig)^2 * 2^(2*(si.exp + sj.exp))
/// ```
///
/// with `tau_sq_k = tau_tame^2 * k`. The left side is a pure power of two, so
/// after moving every power of two left the comparison collapses to one
/// bit-length test: `2^g > rhs <=> g >= bitlen(rhs)` for `rhs > 0`, with
/// `rhs = tau_sq_k * pp^2 < 2^36 * 2^64 = 2^100` (fits u128). `m` must be
/// finite and nonnegative (it is a max of magnitudes).
///
/// Consuming `ufp(m)` instead of `m` keeps the verdict within a factor-2 band
/// of the magnitude form: untamed implies `m >= ufp(m) > bound`, and
/// `m > 2 * bound` implies untamed (`m < 2 * ufp(m)`).
fn untamed_exact(m: f32, tau_sq_k: u64, si: SigmaPow, sj: SigmaPow) -> bool {
    debug_assert!(m.is_finite() && m >= 0.0);
    let (mm, em) = f32_sig_exp(m);
    let pp = si.sig * sj.sig; // < 2^32
    if tau_sq_k == 0 || pp == 0 {
        return mm != 0; // bound = 0: untamed iff m > 0
    }
    if si.negative != sj.negative {
        return true; // bound < 0 <= m
    }
    if mm == 0 {
        return false; // m = 0, bound > 0
    }
    // Binade of m: e_m = floor(log2 m) = em + bitlen(mm) - 1 (subnormal-safe:
    // f32_sig_exp keeps subnormal significands unnormalized, the bit length
    // absorbs the difference).
    let e_m = em + (64 - i64::from(mm.leading_zeros())) - 1;
    let rhs = (tau_sq_k as u128) * (pp as u128) * (pp as u128);
    // 2^g > rhs <=> g >= bitlen(rhs) (rhs > 0 here).
    let g = 2 * e_m - 2 * (si.exp + sj.exp);
    g >= i64::from(128 - rhs.leading_zeros())
}

impl JackpotPolicy {
    /// Computes the tile's policy verdict: `Ok(None)` on rejection, or
    /// `Ok(Some(message))` with the lottery message — the replayed first-stage
    /// product folded over the committed 16-subtile lane layout, exactly one
    /// BLAKE3 block. Only the folded message leaves the policy (to be hashed
    /// under `pow_key`); the bit-exact `A' @ B'.T` never does.
    ///
    /// `a` / `b` are the tile's two sides; `rows_pattern` / `cols_pattern`
    /// commit the tile's grid layout. The committed device's bit-exact FP8
    /// datapath ([`B200`]) supplies the partial sums for check 4.
    pub fn evaluate(
        &self,
        a: &OperandStrip,
        b: &OperandStrip,
        k: usize,
        rows_pattern: &AxisPattern,
        cols_pattern: &AxisPattern,
    ) -> Result<Option<JackpotMessage>> {
        let Some(acc) = self.run_checks(a, b, k)? else {
            return Ok(None);
        };
        let lanes = lane_assignment(rows_pattern, cols_pattern);
        Ok(Some(xor_fold_extract(&acc, &lanes)))
    }

    /// Checks 1-4, returning the bit-exact replayed first-stage accumulator
    /// `A' @ B'.T` (row-major, one value per tile cell) on success, `None` on
    /// rejection.
    fn run_checks(&self, a: &OperandStrip, b: &OperandStrip, k: usize) -> Result<Option<Vec<f32>>> {
        // `k` and the operand dimensions are the verifier's own construction
        // (open_and_noisy_quantize), so these are
        // debug invariants, not runtime validation of untrusted input.
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

        // Checks 3 & 4 share the replay (products + partial sums); on success
        // the replay's final per-cell accumulator is returned for the lottery fold.
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
        let dead: usize = clean
            .chunks(k)
            .zip(l2)
            .map(|(row, &l2)| {
                let dead_bound = self.tau_idle * DELTA * bf16_to_f32(l2) as f64;
                row.iter().filter(|&&x| bf16_to_f32(x).abs() as f64 >= dead_bound).count()
            })
            .sum();
        dead as f64 <= self.eps_idle * clean.len() as f64
    }

    /// `tau_tame^2 * k`, the exact integer threshold constant consumed by
    /// [`untamed_exact`]. (The AIR carries only `k` as its public input: the
    /// default `tau_tame^2 = 2^16` is folded into the ZK XFPOW2 key's shift.)
    /// Errors on policies whose `tau_tame^2` is not an integer in `[0, 2^20]`
    /// or `k > 2^16` (the
    /// [`PublicParams`](crate::api::fp8::public_params::PublicParams) bound):
    /// the exact predicate's width guarantees hold only on that domain, and
    /// such a configuration is a consensus misconfiguration, not a tile fault.
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
        // The k/32 partial sums each cell's bit-exact replay encounters.
        let partials = B200 {}.matmul_fp8_partials(&a.built.noised_part, &b.built.noised_part, h, w, k)?;

        // Exact decodes of the noised FP8 codes (fp8 -> fp32 is exact).
        let a_prime: Vec<f32> = a.built.noised_part.iter().map(|&c| fp8_e4m3_to_f32(c)).collect();
        let b_prime: Vec<f32> = b.built.noised_part.iter().map(|&c| fp8_e4m3_to_f32(c)).collect();

        // Check 4's per-element integer summand scores — the same lambda values the ZK
        // InputQuant table commits and the Matmul lanes consume.
        let lambda_of = |side: &OperandStrip| -> Vec<u64> {
            side.clean
                .iter()
                .enumerate()
                .map(|(idx, &x)| lambda(side.built.alpha[idx / k], x, side.built.l2[idx / k]))
                .collect()
        };
        let a_lambda = lambda_of(a);
        let b_lambda = lambda_of(b);

        let tau_sq_k = self.tau_sq_k(k)?;
        let sigma_pow_a = sigma_pows(&a.built);
        let sigma_pow_b = sigma_pows(&b.built);
        let mut untamed: u64 = 0;

        let mut skippable: u64 = 0;

        for i in 0..h {
            for j in 0..w {
                // Replay magnitude M_ij = max{max_u |A'_iu * B'_ju|, max_t |c_ij,t|}
                // — raw magnitudes, no snapping. Check 3 consumes only its
                // binade floor(log2 M_ij), the exact quantity the ZK Matmul
                // AIR commits per cell, so the two verdicts cannot diverge.
                let mut m_ij = 0.0f32;
                for u in 0..k {
                    m_ij = m_ij.max((a_prime[i * k + u] * b_prime[j * k + u]).abs());
                }
                for &c in &partials[i * w + j] {
                    m_ij = m_ij.max(c.abs());
                }

                // Check 3: tamed products, decided exactly in integers.
                if untamed_exact(m_ij, tau_sq_k, sigma_pow_a[i], sigma_pow_b[j]) {
                    untamed += 1;
                }

                // Check 4: skippable summands.
                let e_cell = cell_exponent(f64::from(m_ij));
                for u in 0..k {
                    if skip(a_lambda[i * k + u], b_lambda[j * k + u], e_cell) {
                        skippable += 1;
                    }
                }
            }
        }

        let tile_cells = (h * w) as f64;
        if untamed as f64 > self.eps_tame * tile_cells {
            return Ok(None);
        }
        if skippable as f64 > self.eps_pred * k as f64 * tile_cells {
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
        // Both rows of A: one spike at 1.0, the rest exactly 0. Per row
        // l2 = sqrt(1/16) = 0.25, so each spike is dead
        // (1.0 >= tau_idle*DELTA*0.25 = 1.0) -- 2/32 of the A side, above
        // eps_idle = 1/64. (A row can hold at most one dead entry: dead means
        // x^2 >= k*l2^2 = sum x^2, so a second spike per row cannot work.)
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
        // Check 4 rejects when v_iju < ulp(M_ij)^2 for too many summands. One
        // spike per row concentrates the replay magnitude: M_ij ~ 421^2
        // (e_cell = 17), so ulp(M_ij)^2 = 2^(2*(17 - 25)) = 2^-16. Shrinking
        // the recorded l2 to 2^-32 collapses sigma = 0.5*alpha*2^-32 ~ 2^-24,
        // and each of the k - 1 zero entries then scores only
        // v_iju = (si*sj)^2 / 2^21 ~ 2^-118, far below 2^-16 -- nearly every
        // summand is skippable, far above eps_pred.
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
        // Two rows whose products M_ij are large relative to sigma_i*sigma_j
        // drive the untamed count above eps_tame. Build honest rows then shrink
        // the recorded l2 to the quantizer floor 2^-32: sigma = 0.5*alpha*2^-32
        // is tiny, so the tamed bound tau_tame*sqrt(k)*si*sj falls far below
        // the honest-scale products M_ij.
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
        // alpha = 2.0, l2 = 1.0 => sigma = DELTA * 2 * 1 = 1 exactly. With
        // k = 1024, tau = 100 (a non-power-of-two threshold): bound =
        // 100 * 32 * 1 * 1 = 3200. The predicate flags exactly the binades
        // that start above the bound: untamed iff 2^e > 3200 iff e >= 12 iff
        // m >= 2^12 = 4096.
        let s = sigma_pow(0x4000, 0x3F80);
        let policy = JackpotPolicy {
            tau_tame: 100.0,
            ..JackpotPolicy::default()
        };
        let tau_sq_k = policy.tau_sq_k(1024).unwrap();
        assert!(!untamed_exact(f32::from_bits(4096f32.to_bits() - 1), tau_sq_k, s, s));
        assert!(untamed_exact(4096.0, tau_sq_k, s, s), "the binade boundary is untamed");
        // m == bound is tamed (ufp(3200) = 2048 <= 3200) under the strict
        // `>`; and any m > 2 * bound is flagged (m < 2 * ufp(m)): the
        // factor-2 band.
        assert!(!untamed_exact(3200.0, tau_sq_k, s, s));
        assert!(untamed_exact(6400.0, tau_sq_k, s, s));
    }

    #[test]
    fn untamed_exact_is_strict_at_power_of_two_ties() {
        // The default tau = 256, k = 1024, sigma = 1: bound = 256 * 32 =
        // 8192 = 2^13 is an exact power of two, so a whole binade starts
        // exactly at the bound. The predicate stays strict there: [2^13, 2^14)
        // has 2^13 > 8192 false (tamed — m == bound stays tamed under the
        // strict `>`), while [2^14, 2^15) is untamed.
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

    /// The ufp predicate is a one-sided coarsening of the magnitude form
    /// `m > bound`: every flagged cell is magnitude-untamed (`m >= ufp(m)`),
    /// and every cell magnitude-untamed at threshold 2*tau is flagged
    /// (`m < 2*ufp(m)`) — i.e. the flags sit between the tau and 2*tau
    /// magnitude flags. Both references are decided exactly in integers, so
    /// there are no ties to skip.
    #[test]
    fn ufp_flags_are_subset_of_magnitude_flags_within_factor_two_band() {
        /// The magnitude predicate `m > tau * sqrt(k) * si * sj`, decided
        /// exactly by comparing squares (the pre-binade `untamed_exact`).
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
        let tau_sq_k = policy.tau_sq_k(k).unwrap(); // 2^14 * 1024 = 2^24
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
