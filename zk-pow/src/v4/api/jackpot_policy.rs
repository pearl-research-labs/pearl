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
//! The policy approves a single opened tile iff every check below passes. It runs after the
//! verifier has rejected Infinity/NaN inputs on intermediate results.
//!
//! 1. **Entry liveness.** `|D_X| <= eps_idle * |I_X| * k` per side, where
//!    `D_X = {(i,u): |X_bar_iu| >= tau_idle * sigma_i^X}` (dead entries).
//! 2. **Noise floor.** `sigma_i^X >= sigma_min` for every row of both sides.
//! 3. **Unpredictable summands.** At most `floor(k*|I_A|*|I_B|/20)` summands may be
//!    skippable. The real-valued comparison is `v_iju < T_iju`, where
//!    `v_iju = (A_bar_iu^2 + (sigma_i^A)^2)(B_bar_ju^2 + (sigma_j^B)^2) / 2^21`
//!    estimates the variance of the product of the two FP8 quantization errors, and
//!    `T_iju = max(ulp_Device(Z_ij,s), 32*ulp23(M_ij))^2/32` uses the window magnitude
//!    `Z` and cell magnitude `M`. For each factor `S = X_bar^2 + sigma^2`, [`lambda`]
//!    computes a fixed-point logarithmic lower bound, encoded as an integer
//!    lower bound on `64*(log2(S) + 508)`.
//!    With `e(x)=floor(log2|x|)`, `W=13` on H100 and `W=25` on B200, set
//!    `q=max(e(Z)-W,e(M)-18)` and `E_GRID=q+157`, omitting zero magnitudes.
//!    The exact consensus rule is `lambda_A + lambda_B < 128*E_GRID + 45952`.
//!    When both magnitudes are zero, `E_GRID=0` and no summands are skipped.
//! 4. **Row security** (downscaling hardware only, i.e. B200). Every row `i`
//!    of both sides keeps its mean Gaussian security proxy at or above `1/25`. With the
//!    signal RMS `s_i = alpha_i * l2_i`, the row's E4M3 strip code `c_i` and
//!    `g_iu = exp(-2 X_bar_iu^2 / s_i^2)`, the proxy is `21/16 * c_i / s_i * g_iu` for
//!    `s_i >= 8` and `min(g_iu / 8, GET_CAP(c_i))` below. The consensus rule bounds `g`
//!    through check 3's `lambda` certificate and is evaluated in exact integers; see
//!    `JackpotPolicy::row_security_ok`.
//!
//! The verdict is a BINARY gate (rejected / accepted): a `D`-dependent credit
//! would incentivize shaping data toward incoherence.
//!
//! [`PublicParams`](crate::v4::api::public_params::PublicParams) bounds `k <= 2^16`.

use crate::v4::api::dtype::{bf16_to_f32, f32_to_fp8_e4m3, fp8_e4m3_to_f32};
use crate::v4::api::layout::{JACKPOT_ENTRIES, lane_assignment};
use crate::v4::api::public_params::Device;
use crate::v4::api::quantization::BuiltRows;
use crate::v4::api::utils::{B200, H100, SMALL_RMS_THRESHOLD, downscale_g, xor_fold_extract};
use crate::v4::circuit::unpredictability::{budget, grid_exponent, grid_exponent_from_binades, lambda, skip};
use anyhow::{Result, ensure};

/// Whitepaper-fixed jackpot policy for one committed device.
#[derive(Copy, Clone, Debug)]
pub(crate) struct JackpotPolicy {
    device: Device,
    /// Liveness threshold (check 1): entry `u` of row `i` is dead iff
    /// `|X_bar_iu| >= tau_idle * sigma_i`.
    pub(crate) tau_idle: f64,
    /// Max fraction of dead entries per tile side (check 1).
    pub(crate) eps_idle: f64,
    /// Floor on every A row's injected noise std `sigma_i` (check 2).
    pub(crate) rms_min_a: f64,
    /// Floor on every B row's injected noise std `sigma_i` (check 2).
    pub(crate) rms_min_b: f64,
}

impl Default for JackpotPolicy {
    fn default() -> Self {
        Self::for_device(Device::B200)
    }
}

impl JackpotPolicy {
    /// Returns the fixed policy for `device`.
    pub(crate) const fn for_device(device: Device) -> Self {
        Self {
            device,
            tau_idle: 8.0,
            eps_idle: 0.015625, // 1/64
            rms_min_a: 4.0,
            rms_min_b: 16.0,
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

/// The lottery message: `xor_fold_extract(A' @ B'.T, lane_assignment(h, w))` — 64
/// bytes, one BLAKE3 block, hashed under `pow_key` to form `hash_jackpot`.
pub type JackpotMessage = [u8; 4 * JACKPOT_ENTRIES];

/// Check 4's fixed-point resolution: the row coefficient `A`, the kernel `T` and the
/// cap are integers in units of `1/PROXY_SCALE`, so a row's `A * sum_u T'_u` is in
/// units of `1/PROXY_SCALE^2`.
const PROXY_SCALE: u64 = 1 << 16;
/// Check 4 accepts a row iff its mean proxy is at least `SECURITY_MIN = 1/SECURITY_MIN_INVERSE`.
const SECURITY_MIN_INVERSE: u64 = 25;
/// One-sided slack of check 3's certificate: `0 <= 64 (log2 S + 508) - lambda < LAMBDA_ERROR`
/// (the exhaustive supremum over the LOG16 keys is 1.0105).
const LAMBDA_ERROR: f64 = 1.02;

/// Per-row noise stds `sigma_i = DELTA * alpha_i * l2_i` (in quantized units),
/// from the floored norms and scales recorded at quantization time.
fn sigmas(built: &BuiltRows, device: Device) -> Vec<f64> {
    built
        .alpha
        .iter()
        .zip(&built.l2)
        .map(|(&alpha, &l2)| device.delta() * bf16_to_f32(alpha) as f64 * bf16_to_f32(l2) as f64)
        .collect()
}

impl JackpotPolicy {
    /// Computes the tile's policy verdict: `Ok(None)` on rejection, or
    /// `Ok(Some(message))` with the lottery message — the replayed first-stage
    /// product folded over sixteen consecutive row-major chunks, exactly one
    /// BLAKE3 block. Only the folded message leaves the policy (to be hashed
    /// under `pow_key`); the bit-exact `A' @ B'.T` never does.
    ///
    /// `a` / `b` are the tile's two sides; `h` / `w` commit the tile's shape.
    /// The committed device's bit-exact FP8
    /// datapath ([`H100`] or [`B200`]) supplies the magnitudes for check 3.
    pub(crate) fn evaluate(
        &self,
        a: &OperandStrip,
        b: &OperandStrip,
        k: usize,
        h: usize,
        w: usize,
    ) -> Result<Option<JackpotMessage>> {
        let Some(acc) = self.run_checks(a, b, k)? else {
            return Ok(None);
        };
        let lanes = lane_assignment(h, w);
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

        let sigma_a = sigmas(&a.built, self.device);
        let sigma_b = sigmas(&b.built, self.device);

        // Check 2: noise floor, sigma_i >= sigma_min on both sides.
        if sigma_a.iter().any(|&s| s < self.device.delta() * self.rms_min_a)
            || sigma_b.iter().any(|&s| s < self.device.delta() * self.rms_min_b)
        {
            return Ok(None);
        }

        // Check 3: replay A' @ B'.T to measure M/Z and count skippable summands.
        let Some(acc) = self.census_ok(a, b, k, sigma_a.len(), sigma_b.len())? else {
            return Ok(None);
        };

        // Check 4: row security of every row, on downscaling hardware only.
        if self.device == Device::B200 && !(Self::row_security_ok(a, k)? && Self::row_security_ok(b, k)?) {
            return Ok(None);
        }
        // The final matrix feeds the lottery fold.
        Ok(Some(acc))
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
                let dead_bound = self.tau_idle * self.device.delta() * bf16_to_f32(l2) as f64;
                row.iter().filter(|&&x| bf16_to_f32(x).abs() as f64 >= dead_bound).count()
            })
            .sum();
        dead as f64 <= self.eps_idle * clean.len() as f64
    }

    /// Check 3: count skippable summands using the replay's M/Z magnitudes.
    /// Return the final matrix if the count is within budget, or `None` on rejection.
    fn census_ok(&self, a: &OperandStrip, b: &OperandStrip, k: usize, h: usize, w: usize) -> Result<Option<Vec<f32>>> {
        // Exact decodes of the noised FP8 codes (fp8 -> fp32 is exact).
        let a_prime: Vec<f32> = a.built.noised_part.iter().map(|&c| fp8_e4m3_to_f32(c)).collect();
        let b_prime: Vec<f32> = b.built.noised_part.iter().map(|&c| fp8_e4m3_to_f32(c)).collect();

        // Check 3's per-element integer summand scores — the same lambda values the ZK
        // InputQuant table commits and the Matmul lanes consume.
        let lambda_of = |side: &OperandStrip| -> Vec<u64> {
            side.clean
                .iter()
                .enumerate()
                .map(|(idx, &x)| lambda(self.device, side.built.alpha[idx / k], x, side.built.l2[idx / k]))
                .collect()
        };
        let a_lambda = lambda_of(a);
        let b_lambda = lambda_of(b);

        let (acc, grids): (Vec<f32>, Vec<Vec<u64>>) = match self.device {
            Device::H100 => {
                let (acc, magnitudes) = H100 {}.matmul_fp8_policy_replay(&a.built.noised_part, &b.built.noised_part, h, w, k)?;
                let grids = magnitudes
                    .into_iter()
                    .map(|cell| {
                        cell.z_windows
                            .into_iter()
                            .map(|z| grid_exponent_from_binades(Device::H100, z, cell.m_cell))
                            .collect()
                    })
                    .collect();
                (acc, grids)
            }
            Device::B200 => {
                let partials = B200 {}.matmul_fp8_partials(&a.built.noised_part, &b.built.noised_part, h, w, k)?;
                let mut acc = Vec::with_capacity(h * w);
                let mut grids = Vec::with_capacity(h * w);
                for i in 0..h {
                    for j in 0..w {
                        let mut magnitude = 0.0f32;
                        for u in 0..k {
                            magnitude = magnitude.max((a_prime[i * k + u] * b_prime[j * k + u]).abs());
                        }
                        for &partial in &partials[i * w + j] {
                            magnitude = magnitude.max(partial.abs());
                        }
                        grids.push(vec![grid_exponent(Device::B200, f64::from(magnitude), f64::from(magnitude))]);
                        acc.push(*partials[i * w + j].last().expect("k > 0 yields a partial"));
                    }
                }
                (acc, grids)
            }
        };

        let mut skippable: u64 = 0;
        for i in 0..h {
            for j in 0..w {
                let window_terms = if self.device == Device::H100 { 128 } else { k };
                for (window, &e_grid) in grids[i * w + j].iter().enumerate() {
                    let start = window * window_terms;
                    let end = (start + window_terms).min(k);
                    for u in start..end {
                        if skip(a_lambda[i * k + u], b_lambda[j * k + u], e_grid) {
                            skippable += 1;
                        }
                    }
                }
            }
        }

        if skippable > budget(k, h, w) {
            return Ok(None);
        }
        Ok(Some(acc))
    }

    /// Check 4: every row's mean row security proxy is at least
    /// `1/SECURITY_MIN_INVERSE`.
    /// Let `g_u = exp(-w_u/2)` such that `w_u = 4 * (X / sigma)^2`.
    /// The security proxy function is:
    /// if `sigma >= 8`:
    ///     `p_u = 21/16 * 2 * X / sigma * exp(-w_u/2)`
    /// else:
    ///     `p_u = min(exp(-w_u/2) / 8, GET_CAP(X))`.
    /// The security proxy check is then:
    ///     sum_u p_u >= k / SECURITY_MIN_INVERSE.
    /// This method approximates the security proxy check in integer arithmetic,
    /// making use of the already computed `lambda` values from check 3.`
    /// Returns `Ok(true)` if the check passes, `Ok(false)` if it fails, and Èrr` on error.`
    fn row_security_ok(side: &OperandStrip, k: usize) -> Result<bool> {
        let threshold = (k as u64 * PROXY_SCALE * PROXY_SCALE).div_ceil(SECURITY_MIN_INVERSE);
        for (row, (&alpha, &l2)) in side.clean.chunks(k).zip(side.built.alpha.iter().zip(&side.built.l2)) {
            let (coefficient, kernel_cap) = gauss_row_constants(alpha, l2)?;
            let lambda_noise = lambda(Device::B200, alpha, 0, l2);
            let kernel_sum: u64 = row
                .iter()
                .map(|&x| {
                    // `lambda` is monotone in `|x|`, so the difference never underflows.
                    let difference = lambda(Device::B200, alpha, x, l2) - lambda_noise;
                    gauss_kernel(difference).min(kernel_cap)
                })
                .sum();
            if coefficient * kernel_sum < threshold {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Check 4's per-row constants `(A, T_cap)`, which are used for the following check:
/// A * sum_u min(T_u, T_cap) >= k * PROXY_SCALE^2 / SECURITY_MIN_INVERSE, where T_u is computed by [`gauss_kernel`].
fn gauss_row_constants(alpha: u16, l2: u16) -> Result<(u64, u64)> {
    const GAUSSIAN_SCALE: f64 = 1.3125;
    // GET_CAP(code) = CAP_OFFSET + CAP_SLOPE * (code - CAP_PIVOT).
    const CAP_OFFSET: f64 = 0.0431;
    const CAP_SLOPE: f64 = 0.2738;
    const CAP_PIVOT: f64 = 0.4;
    /// The small-RMS branch scales the Gaussian by `1/8`.
    const SMALL_RMS_DIVISOR: u64 = 8;

    // The BF16 product has at most 16 significant bits: this f32 product is exact.
    let scaled_rms = bf16_to_f32(alpha) * bf16_to_f32(l2);
    ensure!(
        (4.0..256.0).contains(&scaled_rms),
        "B200 security proxy requires signal RMS in [4, 256)"
    );
    // Preserve the reference's RN32 multiplication followed by RN E4M3.
    let code = f64::from(fp8_e4m3_to_f32(f32_to_fp8_e4m3(downscale_g(&[scaled_rms])[0] * scaled_rms)?));
    let scale = PROXY_SCALE as f64;
    if scaled_rms >= SMALL_RMS_THRESHOLD {
        let coefficient = (scale * GAUSSIAN_SCALE * code / f64::from(scaled_rms)).floor() as u64;
        Ok((coefficient, PROXY_SCALE))
    } else {
        let cap = (scale * (CAP_OFFSET + CAP_SLOPE * (code - CAP_PIVOT))).floor() as u64;
        Ok((PROXY_SCALE / SMALL_RMS_DIVISOR, SMALL_RMS_DIVISOR * cap))
    }
}

/// Check 4's Gaussian kernel at check 3's certified bound: the lambda difference `d`
/// certifies `w <= w_hi(d) = 2^((d + LAMBDA_ERROR) / 64) - 1`, and
/// `T(d) = floor(2^16 * exp(-w_hi(d) / 2))` is the matching lower bound on
/// `2^16 * exp(-w / 2)`.
fn gauss_kernel(difference: u64) -> u64 {
    let w_hi = ((difference as f64 + LAMBDA_ERROR) / 64.0).exp2() - 1.0;
    (PROXY_SCALE as f64 * (-0.5 * w_hi).exp()).floor() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4::api::noise::OperandNoise;
    use crate::v4::api::prequant::{BLOCK_SIZE, exact_norms, open_prequant};
    use crate::v4::api::quantization::Fp8E4M3Quant;
    use crate::v4::circuit::unpredictability::LAMBDA_MAX;

    fn side(clean: Vec<u16>, built: BuiltRows) -> OperandStrip {
        OperandStrip { clean, built }
    }

    /// Build one side from clean rows with real norms, scales, and noise.
    fn build(rows: &[u16], k: usize, r: usize, seed: u8) -> (Vec<u16>, BuiltRows) {
        let num_rows = rows.len() / k;
        let codes: Vec<i8> = rows.iter().map(|&x| (bf16_to_f32(x) * 16.0) as i8).collect();
        let scales = vec![0x3D80; num_rows * (k / BLOCK_SIZE)];
        let clean = open_prequant(&codes, &scales, num_rows, k, BLOCK_SIZE).unwrap();
        assert_eq!(clean, rows, "fixture must round-trip through int8 codes and block scales");
        // Deterministic mesh-like FP8 noise factors keyed off `seed`.
        let e: Vec<u8> = (0..num_rows * r)
            .map(|i| if (i + seed as usize).is_multiple_of(3) { 0xB0 } else { 0x30 })
            .collect();
        let f: Vec<u8> = (0..k * r)
            .map(|i| if (i + seed as usize).is_multiple_of(5) { 0xB0 } else { 0x30 })
            .collect();
        let noise = OperandNoise { e, f };
        let quant = Fp8E4M3Quant::new(Device::B200);
        let norms = exact_norms(&codes, &scales, num_rows, k, BLOCK_SIZE).unwrap();
        let built = quant.noisy_quantize(&codes, &scales, &noise, &norms, Device::B200).unwrap();
        (clean, built)
    }

    /// Varied, well-conditioned clean rows (values in [-2, 2], no outliers).
    fn honest_rows(num_rows: usize, k: usize) -> Vec<u16> {
        let vals = [0x3F80u16, 0xBF00, 0x3E80, 0xBFC0, 0x3F00, 0x4000, 0xBF80, 0x3D80];
        (0..num_rows * k).map(|i| vals[(i * 7 + i / k) % vals.len()]).collect()
    }

    #[test]
    fn sigma_floor_is_correct_for_every_device() {
        for device in [Device::H100, Device::B200] {
            assert_eq!(JackpotPolicy::for_device(device).rms_min_a, 4.0);
            assert_eq!(JackpotPolicy::for_device(device).rms_min_b, 16.0);
        }
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
            rms_min_a: 0.0,
            rms_min_b: 0.0,
            ..JackpotPolicy::default()
        };
        let verdict = policy.run_checks(&side(a, a_built), &side(b, b_built), k).unwrap();
        assert!(verdict.is_none(), "dead fraction above eps_idle must reject");
    }

    #[test]
    fn unpredictability_rejects_when_summands_are_predictable() {
        // Check 3 rejects when too many summands fall below the M/Z threshold.
        // On B200, Z = M and the threshold is (32*ulp_23(M))^2/32.
        // One spike per row gives e(M) = 17, so the threshold is 2^-7.
        // Shrinking l2 to 2^-32 gives sigma = 0.5*alpha*2^-32 ~ 2^-24;
        // each zero entry then has v_iju = (si*sj)^2/2^21 ~ 2^-118.
        // Nearly every summand is skippable, exceeding the 1/20 budget.
        let (k, r) = (1024usize, 16usize);
        let mut rows = vec![0u16; 2 * k];
        rows[0] = 0x3F80; // 1.0
        rows[k] = 0x3F80;
        let (a, mut a_built) = build(&rows, k, r, 0);
        let (b, mut b_built) = build(&rows, k, r, 1);
        for l2 in a_built.l2.iter_mut().chain(b_built.l2.iter_mut()) {
            *l2 = 0x2F80; // bf16 code for 2^-32
        }
        // Neutralize checks 1-2 to isolate check 3.
        let policy = JackpotPolicy {
            eps_idle: 1.0,
            rms_min_a: 0.0,
            rms_min_b: 0.0,
            ..JackpotPolicy::default()
        };
        let verdict = policy.run_checks(&side(a, a_built), &side(b, b_built), k).unwrap();
        assert!(verdict.is_none(), "predictable summands must reject");
    }

    #[test]
    fn gauss_kernel_matches_its_table_and_vanishes_past_289() {
        // Spot values of T(d) = floor(2^16 exp(-(2^((d + 1.02)/64) - 1)/2)) from 60-digit arithmetic.
        assert_eq!(gauss_kernel(0), 65173);
        assert_eq!(gauss_kernel(1), 64815);
        assert_eq!(gauss_kernel(100), 24273);
        assert_eq!(gauss_kernel(150), 8300);
        assert_eq!(gauss_kernel(289), 1);
        assert_eq!((0..290).map(gauss_kernel).sum::<u64>(), 5_550_425);
        // The floors are robust: no un-floored value lies within 1e-3 of an integer, far
        // beyond a faithfully rounded libm's error, so every platform computes the same table.
        for d in 0..290u64 {
            let exact = PROXY_SCALE as f64 * (-0.5 * (((d as f64 + LAMBDA_ERROR) / 64.0).exp2() - 1.0)).exp();
            assert!((exact - exact.round()).abs() > 1e-3, "d={d}: {exact}");
        }
        assert!((290..=LAMBDA_MAX).all(|d| gauss_kernel(d) == 0));
    }

    #[test]
    fn gauss_row_constants_are_exact() {
        // Large row, s = 16: code = e4m3(RN32(0.08 * 16)) = 1.25 and
        // A = floor(2^16 * 21/16 * 1.25 / 16) = 6720; the cap is vacuous.
        assert_eq!(gauss_row_constants(0x4180, 0x3F80).unwrap(), (6720, 65536));
        // Small row, s = 4: code = e4m3(RN32(0.1 * 4)) = 0.40625, GET_CAP = 0.04481125,
        // floor(2^16 * GET_CAP) = 2936, so A = 2^13 and T_cap = 8 * 2936.
        assert_eq!(gauss_row_constants(0x4080, 0x3F80).unwrap(), (8192, 23488));
        // Below the downscaling domain no B200 row exists.
        assert!(gauss_row_constants(0x4040, 0x3F80).is_err());
    }

    #[test]
    fn row_security_rejects_flat_rows_and_accepts_varied_rows() {
        let (k, r) = (64usize, 16usize);
        // The varied fixture rows sit comfortably above the 1/25 floor.
        let (a, a_built) = build(&honest_rows(2, k), k, r, 0);
        assert!(JackpotPolicy::row_security_ok(&side(a, a_built), k).unwrap());
        // Rows whose entries all sit at the row's RMS (`X = +-l2`) have g = exp(-2)
        // everywhere: mean proxy ~ 21/16 * 0.08 * 0.135 ~ 0.014 < 0.04.
        let flat: Vec<u16> = (0..2 * k).map(|i| if i % 3 == 0 { 0xBF80 } else { 0x3F80 }).collect();
        let (a, a_built) = build(&flat, k, r, 0);
        assert!(!JackpotPolicy::row_security_ok(&side(a, a_built), k).unwrap());
    }
}
