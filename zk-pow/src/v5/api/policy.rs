//! The "unpredictable accumulation steps" jackpot policy for the FP16 scheme.
//!
//! During its bit-exact tile replay the verifier records, per cell and per
//! group of 8 products, a [`PolicyStep`]: whether the step was a *breakpoint*
//! (the accumulator alignment or the FP32 rounding discarded a nonzero bit) and
//! how many products were truncated. From these it derives two tile-global
//! quantities and accepts only when both clear their thresholds:
//!
//! * `f_bp` — the breakpoint density, the fraction of group steps that are
//!   breakpoints. It lower-bounds how much of the accumulation is irrecoverable
//!   from the exact sum alone.
//! * `rho` — the certified-work ratio, an attacker-favorable lower bound on the
//!   cost of reconstructing the tile relative to honest work (Appendix "Certified
//!   -work ratio" of the FP16 whitepaper).
//!
//! The thresholds `f_bp >= 0.30` and `rho >= 1.2` are the calibrated FP16 gate.
//! Honest and realistic FP16 workloads sit well above them; the construction is
//! what makes BF16 ineligible (its products rarely truncate), so these
//! thresholds apply only to the FP16 `Quant` value.

use super::accumulate::{a100_dot, PolicyStep, GROUP};
use super::dtype::fp16_to_f32;
use super::quantization::{BuiltRows16, DELTA};
use crate::v4::api::dtype::bf16_to_f32;

/// Noise rank `r`, which is also the attacker's assumed per-cell cost to obtain
/// exact prefix sums over any `k`-range via the low-rank structure. The single
/// source of truth is [`super::params::NOISE_RANK`]; the policy numerator uses
/// it as the `r * N_runs` coefficient.
pub const NOISE_RANK: u64 = super::params::NOISE_RANK as u64;

/// Minimum breakpoint density for an acceptable FP16 tile.
pub const MIN_FBP: f64 = 0.30;
/// Minimum certified-work ratio for an acceptable FP16 tile.
pub const MIN_RHO: f64 = 1.2;

/// Liveness threshold (whitepaper "Shared checks"): entry `u` of row `i` is dead
/// iff `|X_iu| >= TAU_IDLE * DELTA * l2_i`.
pub const TAU_IDLE: f64 = 8.0;
/// Maximum fraction of dead entries permitted per tile side.
pub const EPS_IDLE: f64 = 0.015625; // 1/64
/// Floor on every row's injected noise std `sigma_i = DELTA * alpha_i * l2_i`.
pub const SIGMA_MIN: f64 = 1.0;

/// The FP8 entry-liveness and noise-floor "shared checks" (whitepaper
/// §"Shared checks"), carried over to bound degenerate operands. These gate the
/// tile *in addition to* the unpredictable-accumulation-steps policy
/// ([`evaluate`]): the density gate is a tail-concentrated safety net, while
/// these two rule out the degenerate-operand classes (noise-dominated or
/// spike-dominated rows) directly, mirroring [`crate::v4::api::jackpot_policy`].
///
/// Both checks are "x-only": they depend only on the clean operand and the
/// floored per-row `l2`/`alpha` recorded at quantization, so they pass or fail
/// identically for every noise draw (grinding the noise gains nothing).
///
/// `a_rows`/`b_rows` are the clean FP16 operands (`rows x k`, row-major);
/// `a_built`/`b_built` carry the per-row floored `l2` and scale `alpha`. Returns
/// `Err` naming the first failing side/row; `Ok(())` admits both sides.
pub fn check_shared_gates(
    a_rows: &[u16],
    a_built: &BuiltRows16,
    b_rows: &[u16],
    b_built: &BuiltRows16,
    k: usize,
) -> anyhow::Result<()> {
    for (name, rows, built) in [("A", a_rows, a_built), ("B", b_rows, b_built)] {
        // Noise floor: sigma_i = DELTA * alpha_i * l2_i >= SIGMA_MIN for every row.
        for (i, (&alpha, &l2)) in built.alpha.iter().zip(&built.l2).enumerate() {
            let sigma = DELTA * bf16_to_f32(alpha) as f64 * bf16_to_f32(l2) as f64;
            anyhow::ensure!(
                sigma >= SIGMA_MIN,
                "{name} side row {i}: injected noise std {sigma:.4} below floor {SIGMA_MIN}"
            );
        }
        // Entry liveness: the dead fraction over the side's tile entries is at
        // most EPS_IDLE, with `|X_iu| >= TAU_IDLE * DELTA * l2_i` the dead test
        // (alpha-free, since sigma_i = DELTA * alpha_i * l2_i and alpha_i > 0).
        let dead: usize = rows
            .chunks(k)
            .zip(&built.l2)
            .map(|(row, &l2)| {
                let dead_bound = TAU_IDLE * DELTA * bf16_to_f32(l2) as f64;
                row.iter().filter(|&&x| fp16_to_f32(x).abs() as f64 >= dead_bound).count()
            })
            .sum();
        anyhow::ensure!(
            dead as f64 <= EPS_IDLE * rows.len() as f64,
            "{name} side: {dead} dead entries exceed the {EPS_IDLE} idle fraction of {}",
            rows.len()
        );
    }
    Ok(())
}

/// The outcome of the policy over one opened tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyReport {
    /// Breakpoint density over all group steps in the tile.
    pub f_bp: f64,
    /// Certified-work ratio (attacker-favorable cost lower bound / honest cost).
    pub rho: f64,
    /// `true` iff the tile clears both thresholds.
    pub accept: bool,
}

/// Evaluates the policy over a tile given each cell's per-group census and the
/// inner dimension `k`. `cells` is `|I_A| * |I_B|`.
///
/// `rho = sum_cells[ G*N_bp + r*N_runs + N_pt ] / (cells * k)`, where `N_bp` is
/// the number of breakpoint steps, `N_runs` the number of maximal runs of
/// non-breakpoint steps, and `N_pt` the products truncated outside breakpoints.
pub fn evaluate(census: &[Vec<PolicyStep>], k: usize) -> PolicyReport {
    let cells = census.len();
    assert!(cells > 0 && k > 0, "empty tile");
    let steps_per_cell = k.div_ceil(GROUP);
    let total_steps = (cells * steps_per_cell) as f64;

    let mut breakpoints = 0u64;
    let mut numerator = 0u64;
    for cell in census {
        let mut n_bp = 0u64;
        let mut n_pt = 0u64;
        let mut n_runs = 0u64;
        let mut in_run = false;
        for step in cell {
            if step.breakpoint {
                n_bp += 1;
                in_run = false;
            } else {
                // Non-breakpoint step (empty no-ops included: they cost nothing
                // but belong to the surrounding run).
                n_pt += step.products_truncated as u64;
                if !in_run {
                    n_runs += 1;
                    in_run = true;
                }
            }
        }
        breakpoints += n_bp;
        numerator += GROUP as u64 * n_bp + NOISE_RANK * n_runs + n_pt;
    }

    let f_bp = breakpoints as f64 / total_steps;
    let rho = numerator as f64 / (cells * k) as f64;
    PolicyReport {
        f_bp,
        rho,
        accept: f_bp >= MIN_FBP && rho >= MIN_RHO,
    }
}

/// Replays an `m x n` tile against the A100 datapath, collecting each cell's
/// per-group census, and evaluates the policy. `a` is `m x k`, `b` is `n x k`
/// (row-major FP16 bit patterns). Convenience wrapper over [`a100_dot`] +
/// [`evaluate`]; also returns the recomputed tile for the ticket.
pub fn replay_and_evaluate(
    a: &[u16],
    b: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<f32>, PolicyReport) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let mut tile = vec![0f32; m * n];
    let mut census = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut steps = Vec::with_capacity(k.div_ceil(GROUP));
            tile[i * n + j] = a100_dot(
                &a[i * k..i * k + k],
                &b[j * k..j * k + k],
                0.0,
                Some(&mut steps),
            );
            census.push(steps);
        }
    }
    let report = evaluate(&census, k);
    (tile, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(bp: bool, pt: u32) -> PolicyStep {
        PolicyStep { nonempty: true, breakpoint: bp, products_truncated: pt }
    }

    #[test]
    fn rho_formula() {
        // One cell, k = 40 (5 group steps): bp, nonbp(2), nonbp(1), bp, nonbp.
        let cell = vec![
            step(true, 0),
            step(false, 2),
            step(false, 1),
            step(true, 0),
            step(false, 0),
        ];
        let r = evaluate(&[cell], 40);
        // N_bp=2, N_runs=2 (steps 1-2, step 4), N_pt=3.
        // numerator = 8*2 + 32*2 + 3 = 83; rho = 83/40 = 2.075; f_bp = 2/5 = 0.4.
        assert!((r.f_bp - 0.4).abs() < 1e-12);
        assert!((r.rho - 83.0 / 40.0).abs() < 1e-12);
        assert!(r.accept);
    }

    // --- Shared entry-liveness + noise-floor gates ---

    mod shared_gates {
        use super::super::{check_shared_gates, EPS_IDLE};
        use crate::v5::api::dtype::f32_to_fp16;
        use crate::v5::api::quantization::{noisy_quantize, row_norms, BuiltRows16};

        const K: usize = 64;
        const R: usize = 32;

        /// Build one side (clean rows + the quantization record) with real
        /// noise, mirroring the honest regime.
        fn build(rows: &[u16]) -> (Vec<u16>, BuiltRows16) {
            let num_rows = rows.len() / K;
            // Deterministic +-scale noise lines (near NOISE_TARGET_NORM / sqrt(r)).
            let noise_val = f32_to_fp16(45.0).unwrap();
            let neg = f32_to_fp16(-45.0).unwrap();
            let e: Vec<u16> = (0..num_rows * R).map(|i| if i % 3 == 0 { neg } else { noise_val }).collect();
            let f: Vec<u16> = (0..K * R).map(|i| if i % 5 == 0 { neg } else { noise_val }).collect();
            let norms: Vec<(u16, u16)> = rows.chunks(K).map(|row| row_norms(row).unwrap()).collect();
            let built = noisy_quantize(rows, &e, &f, &norms, R).unwrap();
            (rows.to_vec(), built)
        }

        /// Well-conditioned clean rows (spread magnitudes, no spikes).
        fn honest_rows(num_rows: usize) -> Vec<u16> {
            (0..num_rows * K)
                .map(|i| f32_to_fp16(((i * 37 % 97) as f32 - 48.0) * 0.5).unwrap())
                .collect()
        }

        #[test]
        fn admits_honest_rows() {
            let (a, ab) = build(&honest_rows(4));
            let (b, bb) = build(&honest_rows(16));
            check_shared_gates(&a, &ab, &b, &bb, K).expect("honest rows must pass the shared gates");
        }

        /// The noise floor `sigma_i = DELTA*alpha_i*l2_i >= SIGMA_MIN` is
        /// UNCONDITIONALLY satisfied by honestly-DERIVED scales, for every finite
        /// FP16 row and every k in the whitepaper envelope. Reason: the scale
        /// derivation sets `alpha = Q / (linf + DELTA*sqrt(r)*l2)` with
        /// `linf = max|x| <= sqrt(k)*rms ~ sqrt(k)*l2`, so
        /// `sigma = DELTA*Q*l2 / (linf + DELTA*sqrt(r)*l2) >= DELTA*Q / (sqrt(k) +
        /// DELTA*sqrt(r))`, which for k <= 2^22 (`sqrt(k) <= 2048`) is `>= ~16`.
        /// Flooring l2 only raises sigma. So the noise-floor check can only fail
        /// on an INCONSISTENT (alpha, l2) pair -- which neither the plaintext
        /// verifier (it derives both) nor the FP16 ZK circuit (row_scale_stark
        /// *constrains* alpha = f(l2, linf)) can present. This test is the
        /// machine-checked witness that the floor is implied on the derived path;
        /// it is why the ZK circuit needs NO explicit noise-floor constraint.
        #[test]
        fn noise_floor_is_implied_by_honest_derivation() {
            use crate::v5::api::quantization::DELTA;
            use crate::v4::api::dtype::bf16_to_f32;

            // An adversarial corpus of single rows: spikes, near-zero, flat,
            // wide-magnitude, across several k (incl. the small/large extremes the
            // envelope allows for a tile; the bound is monotone decreasing in k).
            let mut worst = f64::INFINITY;
            for k in [8usize, 16, 64, 256, 1024, 2048] {
                let mut cases: Vec<Vec<u16>> = Vec::new();
                // flat ones; single spike + zeros; two spikes; geometric spread;
                // all (near) zero; alternating tiny/huge.
                let one = f32_to_fp16(1.0).unwrap();
                let max = f32_to_fp16(60000.0).unwrap();
                let tiny = f32_to_fp16(6e-5).unwrap();
                cases.push(vec![one; k]);
                cases.push({ let mut r = vec![f32_to_fp16(0.0).unwrap(); k]; r[0] = max; r });
                cases.push({ let mut r = vec![f32_to_fp16(0.0).unwrap(); k]; r[0] = max; r[1] = max; r });
                cases.push((0..k).map(|j| f32_to_fp16(2f32.powi((j % 20) as i32 - 10)).unwrap()).collect());
                cases.push(vec![tiny; k]);
                cases.push((0..k).map(|j| if j % 2 == 0 { max } else { tiny }).collect());
                for row in &cases {
                    let norms = [row_norms(row).unwrap()];
                    let built = noisy_quantize(row, &vec![0u16; R], &vec![0u16; k * R], &norms, R).unwrap();
                    let sigma = DELTA * bf16_to_f32(built.alpha[0]) as f64 * bf16_to_f32(built.l2[0]) as f64;
                    worst = worst.min(sigma);
                    assert!(sigma >= 1.0, "noise floor violated by a DERIVED scale (k={k}): sigma={sigma}");
                }
            }
            // The observed minimum is well above 1 (the derived floor has large margin).
            assert!(worst > 2.0, "derived sigma minimum {worst} unexpectedly close to the floor");
        }

        #[test]
        fn noise_floor_rejects_vanishing_sigma() {
            let (a, mut ab) = build(&honest_rows(4));
            let (b, bb) = build(&honest_rows(16));
            // Force row 0's floored l2 to 2^-32, so sigma = DELTA*alpha*l2 << 1.
            ab.l2[0] = 0x2F80; // bf16 code for 2^-32
            let err = check_shared_gates(&a, &ab, &b, &bb, K).expect_err("sub-floor sigma must reject");
            assert!(format!("{err:#}").contains("below floor"), "got: {err:#}");
        }

        #[test]
        fn liveness_rejects_spike_dominated_rows() {
            // Each A row: two unit spikes, the rest zero. Per row l2 = sqrt(2/K),
            // so each spike hits the dead bound TAU_IDLE*DELTA*l2 = 4*sqrt(2/K) < 1,
            // i.e. two dead entries per row -> dead fraction 2/K > EPS_IDLE = 1/64.
            let spikes = 2usize;
            let mut a_rows = vec![0u16; 4 * K];
            for r in 0..4 {
                for s in 0..spikes {
                    a_rows[r * K + s] = f32_to_fp16(1.0).unwrap();
                }
            }
            let (a, ab) = build(&a_rows);
            let (b, bb) = build(&honest_rows(16));
            assert!(spikes as f64 / K as f64 > EPS_IDLE, "test geometry must exceed the idle fraction");
            let err = check_shared_gates(&a, &ab, &b, &bb, K).expect_err("spike-dominated rows must reject");
            assert!(format!("{err:#}").contains("dead entries"), "got: {err:#}");
        }
    }

    #[test]
    fn flat_tile_is_rejected() {
        // No breakpoints, no truncation (the BF16-weak / flat-input regime):
        // one long run, rho = 32/80 = 0.4, f_bp = 0.
        let cell: Vec<PolicyStep> = (0..10).map(|_| step(false, 0)).collect();
        let r = evaluate(&[cell], 80);
        assert_eq!(r.f_bp, 0.0);
        assert!(r.rho < MIN_RHO);
        assert!(!r.accept);
    }

    // ----------------------------------------------------------------------
    // Oracle dump for the sm_80 `fp16_policy` miner kernel (additive, ignored).
    //
    // Emits, for many random shapes (varied k incl. non-multiples of 8, varied
    // magnitudes that exercise breakpoints and product truncations), the exact
    // per-cell integer census (`n_bp`, `n_runs`, `n_pt`), the `(breakpoints,
    // numerator)` tile totals, the `f_bp`/`rho` f64 bit patterns, `accept`, and
    // the recomputed tile (u32 bits). The GA100 kernel is asserted bit-exact
    // against this. Run with:
    //   PEARL_POLICY_DUMP=/path/vectors.txt cargo test -p zk-pow --lib \
    //       api::fp16::policy::tests::dump_policy_vectors -- --ignored --nocapture
    // ----------------------------------------------------------------------

    /// Per-cell run count mirroring `evaluate`'s inner loop (so the dump carries
    /// the exact integer census the fused in-kernel reduction must reproduce).
    fn cell_counts(steps: &[PolicyStep]) -> (u64, u64, u64) {
        let (mut n_bp, mut n_pt, mut n_runs) = (0u64, 0u64, 0u64);
        let mut in_run = false;
        for s in steps {
            if s.breakpoint {
                n_bp += 1;
                in_run = false;
            } else {
                n_pt += s.products_truncated as u64;
                if !in_run {
                    n_runs += 1;
                    in_run = true;
                }
            }
        }
        (n_bp, n_runs, n_pt)
    }

    #[test]
    #[ignore = "oracle dump for the sm_80 fp16_policy kernel; writes PEARL_POLICY_DUMP"]
    fn dump_policy_vectors() {
        use super::super::dtype::decompose_fp16;
        use std::io::Write;

        // Tiny deterministic SplitMix64 PRNG (no external dep).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        // A finite FP16 bit pattern with a bounded exponent (keeps dot products
        // well inside the finite range) and a ~12% chance of exact zero. The
        // magnitude spread across a row is what makes products truncate and
        // group steps hit breakpoints.
        let mut gen_fp16 = |exp_lo: i32, exp_hi: i32| -> u16 {
            if (next() % 100) < 12 {
                return if next() & 1 == 1 { 0x8000 } else { 0x0000 };
            }
            let sign = ((next() & 1) as u16) << 15;
            let span = (exp_hi - exp_lo + 1) as u64;
            let exp = (exp_lo + (next() % span) as i32) as u16; // 1..=30 field
            let man = (next() % 0x400) as u16;
            let bits = sign | (exp << 10) | man;
            // decompose_fp16 would reject 0x1F; our exp range stays < 0x1F.
            debug_assert!((bits >> 10) & 0x1F != 0x1F);
            let _ = decompose_fp16(bits);
            bits
        };

        // (m, n, k) shapes: varied k including non-multiples of 8, a 1xk, and
        // mixed exponent windows (narrow => flat/rejected, wide => breakpoints).
        let shapes: &[(usize, usize, usize, i32, i32)] = &[
            (1, 1, 7, 1, 20),
            (2, 3, 8, 1, 24),
            (3, 2, 9, 1, 24),
            (4, 4, 16, 1, 26),
            (2, 2, 31, 1, 28),
            (5, 3, 33, 1, 28),
            (4, 6, 64, 1, 28),
            (3, 3, 100, 1, 28),
            (6, 5, 127, 1, 28),
            (2, 4, 256, 1, 28),
            (8, 8, 40, 10, 12), // narrow window: near-flat
            (7, 7, 48, 1, 29),  // wide window: heavy breakpoints
            (1, 16, 72, 1, 27),
            (16, 1, 72, 1, 27),
            // Flat single-exponent windows: near-zero breakpoint density and low
            // rho -> the rejected regime (validates the accept=false path and
            // f_bp/rho near the 0.30/1.2 thresholds bit-exactly).
            (6, 6, 64, 15, 15),
            (4, 5, 80, 20, 20),
            (8, 8, 128, 1, 29), // large tile, widest window: heavy breakpoints
            (10, 7, 200, 1, 28),
        ];

        let path = std::env::var("PEARL_POLICY_DUMP")
            .unwrap_or_else(|_| "/tmp/pearl_policy_vectors.txt".to_string());
        let mut out = String::new();
        out.push_str(&format!("{}\n", shapes.len()));
        for &(m, n, k, elo, ehi) in shapes {
            let a: Vec<u16> = (0..m * k).map(|_| gen_fp16(elo, ehi)).collect();
            let b: Vec<u16> = (0..n * k).map(|_| gen_fp16(elo, ehi)).collect();
            let (tile, report) = replay_and_evaluate(&a, &b, m, n, k);

            // Recompute the per-cell census and integer totals for the dump.
            let mut percell: Vec<(u64, u64, u64)> = Vec::with_capacity(m * n);
            let mut breakpoints = 0u64;
            let mut numerator = 0u64;
            for i in 0..m {
                for j in 0..n {
                    let mut steps = Vec::new();
                    let _ = a100_dot(&a[i * k..i * k + k], &b[j * k..j * k + k], 0.0, Some(&mut steps));
                    let (nb, nr, npt) = cell_counts(&steps);
                    breakpoints += nb;
                    numerator += GROUP as u64 * nb + NOISE_RANK * nr + npt;
                    percell.push((nb, nr, npt));
                }
            }

            out.push_str(&format!("SHAPE {m} {n} {k}\n"));
            // a bits, b bits.
            out.push_str("A");
            for &x in &a {
                out.push_str(&format!(" {x}"));
            }
            out.push('\n');
            out.push_str("B");
            for &x in &b {
                out.push_str(&format!(" {x}"));
            }
            out.push('\n');
            // tile u32 bits.
            out.push_str("TILE");
            for &v in &tile {
                out.push_str(&format!(" {}", v.to_bits()));
            }
            out.push('\n');
            // per-cell (n_bp n_runs n_pt) triples, row-major.
            out.push_str("CENSUS");
            for (nb, nr, npt) in &percell {
                out.push_str(&format!(" {nb} {nr} {npt}"));
            }
            out.push('\n');
            // totals + f_bp/rho f64 bits + accept.
            out.push_str(&format!(
                "EVAL {} {} {} {} {}\n",
                breakpoints,
                numerator,
                report.f_bp.to_bits(),
                report.rho.to_bits(),
                report.accept as u8
            ));
        }
        let mut f = std::fs::File::create(&path).expect("create dump");
        f.write_all(out.as_bytes()).expect("write dump");
        eprintln!("wrote {} shapes to {path}", shapes.len());
    }

    #[test]
    fn honest_spread_operands_pass() {
        // Drive the policy from the embedded reference vectors: the ones with
        // large k and heavy truncation model honest/realistic FP16 workloads.
        // Their aggregate breakpoint density and work ratio clear the gate.
        const VECTORS: &str = include_str!("testdata/a100_dot_vectors.txt");
        let mut census: Vec<Vec<PolicyStep>> = Vec::new();
        let mut k_used = 0;
        for line in VECTORS.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let t: Vec<&str> = line.split_whitespace().collect();
            let k: usize = t[0].parse().unwrap();
            if k < 256 {
                continue; // use the large-k (realistic chain) vectors
            }
            let a: Vec<u16> = t[1..1 + k].iter().map(|x| x.parse().unwrap()).collect();
            let b: Vec<u16> = t[1 + k..1 + 2 * k].iter().map(|x| x.parse().unwrap()).collect();
            let mut steps = Vec::new();
            let _ = a100_dot(&a, &b, 0.0, Some(&mut steps));
            census.push(steps);
            k_used = k;
        }
        assert!(census.len() >= 10, "need several large-k vectors");
        let r = evaluate(&census, k_used);
        assert!(r.f_bp >= MIN_FBP, "f_bp {} below {MIN_FBP}", r.f_bp);
        assert!(r.rho >= MIN_RHO, "rho {} below {MIN_RHO}", r.rho);
        assert!(r.accept);
    }
}
