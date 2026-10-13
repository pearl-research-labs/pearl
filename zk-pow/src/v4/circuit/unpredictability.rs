//! Integer rule for the unpredictable-summands check and its tile-wide skip budget.
//!
//! For cell `(i, j)` and product index `u`, the real-valued test is `v_iju < T_iju`:
//!
//! ```text
//! au = alpha_A[i] * X_A[i,u]              bu = alpha_B[j] * X_B[j,u]
//! si = delta(device) * alpha_A[i] * l2f_A[i]
//! sj = delta(device) * alpha_B[j] * l2f_B[j]
//! v_iju = (au^2 + si^2) * (bu^2 + sj^2) / 2^21
//! T_iju = max(ulp_Device(Z_ij,s), 32 * ulp_23(M_ij))^2 / 32
//! ```
//!
//! `X_A` and `X_B` are clean BF16 entries, `alpha` is the row scale, and `l2f` is
//! the floored row norm. Thus `au`/`bu` are scaled clean entries and `si`/`sj` are
//! row noise standard deviations, all in quantized units. The variance estimate
//! for each FP8 quantization error is `2^-10.5 * (entry^2 + sigma^2)`; their product
//! gives the `2^-21` factor in `v`. `M` is the cell's replay magnitude and `Z` is
//! the magnitude of the window containing `u`; B200 sets `Z = M`.
//!
//! Write `q = max(e(Z_ij,s) - W, e(M_ij) - 18)`, where `e(x)` is
//! `floor(log2 |x|)` for nonzero `x`. A zero M or Z contribution has zero ULP; only a
//! jointly-zero threshold disables skipping. `W` is 13 on H100 and 25 on B200.
//! Skippability is decided using a fixed-point lower bound on `log2(v)`
//! and the integer comparison below.
//! The tile passes iff this count is at most `floor(k*h*w/20)`.
//!
//! # The exact sum-of-squares rule (the consensus definition)
//!
//! Each summand magnitude `S = au^2 + si^2` is scored once, in base-2 logs at
//! [`FRAC_BITS`]` = 6` fraction bits (unit `2^-6` octave). Every step is exact integer
//! arithmetic — no floating point anywhere, so the rule is bit-stable across platforms.
//!
//! Write both squares over 16-bit normalized significands: `au = n_x * 2^(e_x - 15)` and
//! `si = n_s * 2^(e_s - 15)` with `n_x, n_s in [2^15, 2^16)` (`n_x = 0` when `x = 0`).
//! With `e_big = max(e_x, e_s)`, `g = 2*|e_x - e_s|`, and `(n_big, n_small)` the
//! matching significand pair,
//!
//! ```text
//! S = 2^(2*e_big - 30) * (n_big^2 + n_small^2 / 2^g).
//! ```
//!
//! The sum of squares, its top slice, and the lambda score are
//!
//! ```text
//! V      = n_big^2 + floor(n_small^2 / 2^g)   if g <= 30, else n_big^2   (the far cut)
//! kappa  = floor(V / 2^17)                    in [2^13, 2^16)
//! lambda = 64*(2*e_big - 13 + SCORE_BIAS) + G(kappa),   G(kappa) = floor(64 * log2 kappa).
//! ```
//!
//! The `-13 = -(30 - 17)` is due to the initial left shift by 30 to compute `V`: we only
//! shift back right by 17 to get `kappa`.
//! The far cut is exact, not an approximation:
//! `g` is even and `n_small^2 < 2^32 <= 2^g` there, so `V` equals the uncut sum on every
//! gap; `x = 0` lands on it, scoring `si^2` alone. Every loss is a floor, so the error is
//! one-sided:
//!
//! ```text
//! 64*(log2 S + SCORE_BIAS) - 1.1  <  lambda  <=  64*(log2 S + SCORE_BIAS)
//! ```
//!
//! (the `kappa` truncation `< 2^-13` relative plus the final floor — under `1.1` units of
//! `2^-6` octave in total, never above the true value).
//!
//! # The skip predicate
//!
//! With `E_GRID = q + 157` (`0` only when both M and Z are zero), substituting the scores
//! into the threshold and folding constants gives the consensus rule
//!
//! ```text
//! skip  <=>  E_GRID != 0  and  lambda_A + lambda_B < 128 * E_GRID + C
//! C = 64*(16 + 2*SCORE_BIAS) - 128*157 = 45952.
//! ```
//!
//! Here `16 = 21 - 5`: `v` has a `2^21` denominator while the threshold divides by
//! `32 = 2^5`.
//!
//! The fixed-point rule relates to the ideal real-valued comparison as
//! follows: `v < T` implies a skip, and a skip implies `v < T * 2^(2.2/64)` for
//! `T > 0`. These bounds follow from each lambda's downward error of less than
//! `1.1` units. The fixed-point rule defines the consensus count.
//!
//! # AIR wiring and soundness direction
//!
//! InputQuant commits the full score witness per element (normalized significands, the
//! dominance bit, the gap, the far bit, both floor-division stages, `kappa`, and `lambda`);
//! POW2D serves the two shift powers, LOG16 serves `G(kappa)`, and ScaleStark exports the
//! sigma significand and encoding through the group tuple. Each Matmul lane commits a
//! boolean skip flag; claiming *non-skip* costs one RC16 on the affine key
//!
//! ```text
//! LAMBDA_A + LAMBDA_B - 128*E_GRID - C,
//! ```
//!
//! filtered by `GRID_NONZERO * (1 - SKIP_FLAG)`; jointly-zero grids pin `SKIP_FLAG = 0`.
//! A true skip's key is negative and wraps far outside `[0, 2^16)`, forcing the flag to 1.
//! Claiming *skip* is free: a prover can only overstate the census, never understate it.
//! Inflating M, Z, or E_GRID is likewise monotone toward rejection.
//!
//! # The budget
//!
//! The plaintext accepts iff `count <= floor(k*h*w/20)`, computed directly in integers.

use crate::v4::api::public_params::Device;

/// Fixed-point fraction bits of the log domain: all scores are integers in units of
/// `2^-FRAC_BITS = 1/64` octave.
pub const FRAC_BITS: u32 = 6;

/// Bias added inside the score `lambda ~= 64*(log2 S + SCORE_BIAS)` so every score is
/// positive. Sized against the smallest squared addend in the scheme:
/// `(alpha*x)^2 = (2^-120 * 2^-133)^2 = 2^-506` (alpha at its domain floor times the
/// smallest subnormal x). Live sums sit far higher: `S >= sigma^2 >= 2^-306`.
pub const SCORE_BIAS: u64 = 508;

/// Exponent gap at which the small addend is dropped from the sum of squares. Exact, not an
/// approximation: the dropped term `floor(n_small^2 / 2^(2*gap))` is identically zero
/// there, since `n_small^2 < 2^32 <= 2^(2*gap)` once the gap reaches 16. Keeps the
/// per-stage shift key `|e_x - e_s|` inside POW2D's `[0, 19]` domain.
pub const FAR_GAP: u64 = 16;

/// Smallest lambda on a live row: `x = 0` with `alpha` and `l2f` at their domain floors
/// gives `S = sigma^2 = (2^-120 * 2^-32 / 2)^2 = 2^-306` exactly, so
/// `lambda = 64*(-306 + 508) = 12928` (a power of two: every floor in the rule is exact).
pub const LAMBDA_MIN_LIVE: u64 = 12_928;

/// Largest lambda on a satisfiable trace: the AIR caps every addend encoding at 436 (the
/// encoding of the largest in-scheme `alpha*x`), so each addend's exponent is at most
/// `436 - 268 = 168` and each addend is below `2^169`. Then
/// `S = (alpha*x)^2 + sigma^2 < 2^338 + 2^338 = 2^339`, and
/// `lambda <= 64*(log2 S + 508) < 64*(339 + 508) = 54208`. Honest tiles sit far lower.
pub const LAMBDA_MAX: u64 = 54_207;

/// Precision of the shared M branch: `ulp_23(M) * 32 = 2^(e(M)-18)`.
pub const M_WINDOW_BITS: i64 = 18;

/// Bias of the nonnegative grid encoding `E_GRID = q + GRID_EXPONENT_BIAS`.
pub const GRID_EXPONENT_BIAS: i64 = 157;

/// The skip threshold's additive constant:
/// `lambda_A + lambda_B < 128 * E_GRID + SKIP_THRESHOLD_OFFSET`.
pub const SKIP_THRESHOLD_OFFSET: u64 = 64 * (16 + 2 * SCORE_BIAS) - 128 * (GRID_EXPONENT_BIAS as u64);

/// Bit length of a 16-bit significand product — the WIDTH16 LUT's first value column.
pub const fn sig_width(sig_product: u64) -> u64 {
    assert!(sig_product < 1 << 16, "WIDTH16 key domain");
    (u64::BITS - sig_product.leading_zeros()) as u64
}

/// Product-nonzero flag — the WIDTH16 LUT's second value column. Gates the exponent term
/// of the element encoding, pinning the zero sentinel exactly.
pub const fn sig_nonzero(sig_product: u64) -> u64 {
    (sig_product != 0) as u64
}

/// `ceil(2^(15 + i/64))` for `i in [0, 64)`: the exact integer decision bounds of the
/// fractional part of [`log2_fixed`]. An integer `n in [2^15, 2^16)` satisfies
/// `n >= LOG_BOUNDS[i]` iff `log2(n) >= 15 + i/64` (the bounds are never integers except
/// at `i = 0`, with the nearest miss at distance 1.25e-2 — checked at 80-digit precision).
pub const LOG_BOUNDS: [u64; 64] = [
    32768, 33125, 33486, 33851, 34219, 34592, 34969, 35349, 35734, 36123, 36517, 36914, 37316, 37723, 38133, 38549, 38968, 39393,
    39822, 40255, 40694, 41137, 41585, 42038, 42495, 42958, 43426, 43899, 44377, 44860, 45348, 45842, 46341, 46846, 47356, 47872,
    48393, 48920, 49453, 49991, 50536, 51086, 51642, 52205, 52773, 53348, 53929, 54516, 55109, 55710, 56316, 56929, 57549, 58176,
    58810, 59450, 60097, 60752, 61413, 62082, 62758, 63441, 64132, 64831,
];

/// `G(n) = floor(64 * log2 n)` for `n in [1, 2^16)`, and `0` at `n = 0` (the phantom-row
/// sentinel) — the LOG16 LUT's value column. Exact: splits off the bit length, normalizes
/// into `[2^15, 2^16)`, and classifies the fraction against the [`LOG_BOUNDS`] integers.
pub fn log2_fixed(n: u64) -> u64 {
    assert!(n < 1 << 16, "LOG16 key domain");
    if n == 0 {
        return 0;
    }
    let w = sig_width(n) - 1;
    let normalized = n << (15 - w);
    let fraction = (LOG_BOUNDS.partition_point(|&b| b <= normalized) - 1) as u64;
    64 * w + fraction
}

/// The full witness of one summand score — every value the InputQuant AIR commits
/// for its element, produced by [`lambda_witness`]. All fields are exact integers.
#[derive(Clone, Copy, Debug)]
pub struct LambdaWitness {
    /// `[enc(alpha*x) >= enc(sigma)]`; `0` when `x = 0`.
    pub x_dominates: bool,
    /// `|enc(alpha*x) - enc(sigma)|` — the addend exponent gap in binades.
    pub exponent_gap: u64,
    /// `[exponent_gap >= FAR_GAP]` — the small addend is dropped.
    pub gap_is_far: bool,
    /// `exponent_gap` on near rows, `0` on far rows — the POW2D shift key.
    pub near_gap: u64,
    /// `2^near_gap` — the shared divisor of both floor stages.
    pub near_gap_pow: u64,
    /// The dominant addend's normalized significand, in `[2^15, 2^16)`.
    pub norm_big: u64,
    /// The other addend's normalized significand: `[2^15, 2^16)` or `0`.
    pub norm_small: u64,
    /// `floor(norm_small^2 / 2^near_gap)` — the first floor stage.
    pub half_quotient: u64,
    /// `norm_small^2 - half_quotient * near_gap_pow`, in `[0, near_gap_pow)`.
    pub half_remainder: u64,
    /// `floor(norm_small^2 / 2^(2*near_gap))` — the second floor stage.
    pub quotient: u64,
    /// `half_quotient - quotient * near_gap_pow`, in `[0, near_gap_pow)`.
    pub remainder: u64,
    /// `kappa = floor(V / 2^17)` for the sum of squares
    /// `V = norm_big^2 + (1 - far) * quotient`; in `[2^13, 2^16)`.
    pub sum_of_squares_top: u64,
    /// `V - 2^17 * kappa`, in `[0, 2^17)`.
    pub sum_of_squares_rest: u64,
    /// `G(kappa) = floor(64 * log2 kappa)` — the LOG16 value at the top slice.
    pub log_fraction: u64,
    /// The summand score `lambda = 128*max(enc(alpha*x), enc(sigma)) - 2624 + G(kappa)`,
    /// which is `64*(log2 S + SCORE_BIAS)` up to the rule's floors.
    pub lambda: u64,
}

/// Evaluates the sum-of-squares rule on one summand. Inputs are the two addends' normalized
/// significands and encodings: `|alpha*x| = x_norm * 2^(x_enc - 268 - 15)` (`x_norm` and
/// `x_enc` both `0` when `x = 0`) and `sigma = sigma_norm * 2^(sigma_enc - 268 - 15)` with
/// `x_norm, sigma_norm in [2^15, 2^16) ∪ {0}` and `enc(y) = floor(log2 y) + 268`
/// (`268 = 2*134`, one bf16 exponent bias per factor).
pub fn lambda_witness(x_norm: u64, x_enc: u64, sigma_norm: u64, sigma_enc: u64) -> LambdaWitness {
    debug_assert!((1 << 15..1 << 16).contains(&sigma_norm), "sigma is never zero in-scheme");
    debug_assert!(x_norm == 0 || (1 << 15..1 << 16).contains(&x_norm));
    debug_assert_eq!(x_norm == 0, x_enc == 0);
    let x_dominates = x_enc >= sigma_enc && x_norm != 0;
    let exponent_gap = x_enc.abs_diff(sigma_enc);
    let gap_is_far = exponent_gap >= FAR_GAP;
    let near_gap = if gap_is_far { 0 } else { exponent_gap };
    let near_gap_pow = 1u64 << near_gap;
    let (norm_big, norm_small) = if x_dominates {
        (x_norm, sigma_norm)
    } else {
        (sigma_norm, x_norm)
    };
    let small_squared = norm_small * norm_small;
    let half_quotient = small_squared >> near_gap;
    let half_remainder = small_squared - (half_quotient << near_gap);
    let quotient = half_quotient >> near_gap;
    let remainder = half_quotient - (quotient << near_gap);
    let sum_of_squares = norm_big * norm_big + if gap_is_far { 0 } else { quotient };
    let sum_of_squares_top = sum_of_squares >> 17;
    let sum_of_squares_rest = sum_of_squares & ((1 << 17) - 1);
    debug_assert!((1 << 13..1 << 16).contains(&sum_of_squares_top));
    let log_fraction = log2_fixed(sum_of_squares_top);
    let lambda = 128 * x_enc.max(sigma_enc) - 2624 + log_fraction;
    LambdaWitness {
        x_dominates,
        exponent_gap,
        gap_is_far,
        near_gap,
        near_gap_pow,
        norm_big,
        norm_small,
        half_quotient,
        half_remainder,
        quotient,
        remainder,
        sum_of_squares_top,
        sum_of_squares_rest,
        log_fraction,
        lambda,
    }
}

/// Splits a finite bf16 code into `(E*, M)`: `M(V) = 128*(1 - eiz) + mantissa`,
/// `E*(V) = exp + eiz`, `V = ±M(V) * 2^(E*(V) - 134)`. Panics on inf/NaN codes (banned
/// everywhere).
fn bf16_e_star_m(code: u16) -> (u64, u64) {
    let exp = u64::from((code >> 7) & 0xFF);
    assert_ne!(exp, 255, "non-finite bf16 code {code:#06x} is banned");
    let mantissa = u64::from(code & 0x7F);
    if exp == 0 { (1, mantissa) } else { (exp, 128 + mantissa) }
}

/// The element's lambda: the score of `S = (alpha*x)^2 + sigma^2` with
/// `sigma = delta(device) * alpha * l2f` — the exact integer the InputQuant AIR commits and the
/// operand channel carries to the Matmul lanes. `alpha` and `l2f` must be positive-normal
/// (in-scheme `l2f >= 2^-32`); both products below are exact.
pub fn lambda(device: Device, alpha: u16, x: u16, l2f: u16) -> u64 {
    let (alpha_e, alpha_m) = bf16_e_star_m(alpha);
    assert!(alpha_m >= 128, "alpha must be normal");
    let (x_e, x_m) = bf16_e_star_m(x);
    let product = alpha_m * x_m;
    let width = sig_width(product);
    let (x_norm, x_enc) = if x_m == 0 {
        (0, 0)
    } else {
        // |alpha*x| = product * 2^(alpha_e + x_e - 268), so enc = e + 268 = alpha_e + x_e - 1 + width.
        (product << (16 - width), alpha_e + x_e - 1 + width)
    };
    let (l2f_e, l2f_m) = bf16_e_star_m(l2f);
    assert!(l2f_m >= 128, "l2f must be normal (the 2^-32 norm floor)");
    let sigma_product = alpha_m * l2f_m;
    let sigma_wide = u64::from(sigma_product >= 1 << 15);
    let sigma_norm = sigma_product << (1 - sigma_wide);
    // The device power-of-two delta changes only the exponent; the significand is shared.
    let sigma_enc = alpha_e + l2f_e + sigma_wide + device.sigma_encoding_offset();
    lambda_witness(x_norm, x_enc, sigma_norm, sigma_enc).lambda
}

/// `floor(log2(m))` for a nonnegative protocol magnitude, or `None` at zero.
pub fn magnitude_exponent(m: f64) -> Option<i64> {
    assert!(m.is_finite() && m >= 0.0, "policy magnitudes must be finite and nonnegative");
    if m == 0.0 {
        return None;
    }
    let biased = (m.to_bits() >> 52) & 0x7FF;
    assert!((1..2047).contains(&biased), "a nonzero replay magnitude is f64-normal");
    Some(biased as i64 - 1023)
}

/// The cell magnitude's AIR encoding `e(M) + 139`, or zero when `M = 0`.
pub fn cell_exponent(m: f64) -> u64 {
    magnitude_exponent(m).map_or(0, |e| (e + 139) as u64)
}

/// `E_GRID = max(e(Z)-W, e(M)-18) + 157`, with zero contributions omitted.
/// Returns zero only when both magnitudes are zero.
pub fn grid_exponent(device: Device, z: f64, m: f64) -> u64 {
    grid_exponent_from_binades(
        device,
        magnitude_exponent(z).map(|e| e as i32),
        magnitude_exponent(m).map(|e| e as i32),
    )
}

/// Integer-only counterpart of [`grid_exponent`] for exact replay binades.
pub fn grid_exponent_from_binades(device: Device, z: Option<i32>, m: Option<i32>) -> u64 {
    let encode = |q: i64| {
        let encoded = q + GRID_EXPONENT_BIAS;
        assert!(encoded > 0, "nonzero protocol grids must have a positive encoding");
        encoded as u64
    };
    let z_q = z.map(|e| i64::from(e) - i64::from(device.fp8_window_bits()));
    let m_q = m.map(|e| i64::from(e) - M_WINDOW_BITS);
    match (z_q, m_q) {
        (None, None) => 0,
        (Some(q), None) | (None, Some(q)) => encode(q),
        (Some(z_q), Some(m_q)) => encode(z_q.max(m_q)),
    }
}

/// The canonical strict skip predicate. `e_grid == 0` is the jointly-zero threshold.
pub fn skip(lambda_a: u64, lambda_b: u64, e_grid: u64) -> bool {
    e_grid != 0 && lambda_a + lambda_b < 128 * e_grid + SKIP_THRESHOLD_OFFSET
}

/// The exact consensus skip budget `floor(k*h*w/20)`.
pub fn budget(k: usize, h: usize, w: usize) -> u64 {
    (k as u64) * (h as u64) * (w as u64) / 20
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4::api::dtype::bf16_to_f32;

    /// A varied sweep of finite bf16 codes: zeros, subnormals, normals across the range.
    fn bf16_samples() -> Vec<u16> {
        let mut v = vec![0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0x0100];
        for i in 0..600u32 {
            let code = (i.wrapping_mul(2654435761) >> 16) as u16;
            if (code >> 7) & 0xFF != 255 {
                v.push(code);
            }
        }
        v
    }

    /// Alpha codes drawn from the sanctioned domain: positive-normal with
    /// `E* in [7, 167]` (`alpha = RNE_bf16(448 / nb)` over the floored-linf range).
    fn alpha_samples() -> Vec<u16> {
        (0..40u16).map(|i| ((7 + i * 4) << 7) | (i * 11 % 128)).collect()
    }

    /// Normal l2f codes (the scheme floors norms at `2^-32`, exp field 95).
    fn l2f_samples() -> Vec<u16> {
        (0..30u16).map(|i| ((95 + i * 5) << 7) | (i * 23 % 128)).collect()
    }

    #[test]
    fn width_and_nonzero_match_their_real_forms() {
        assert_eq!((sig_width(0), sig_nonzero(0)), (0, 0));
        for sp in (3..1u64 << 16).step_by(3).chain([1, 2, 127, 128, 65025, 65535]) {
            assert_eq!(sig_width(sp), u64::from(sp.ilog2()) + 1);
            assert_eq!(sig_nonzero(sp), 1);
        }
    }

    #[test]
    fn log2_fixed_matches_its_real_form() {
        assert_eq!(log2_fixed(0), 0, "phantom-row sentinel");
        for n in 1..1u64 << 16 {
            let exact = (64.0 * (n as f64).log2()).floor() as u64;
            assert_eq!(log2_fixed(n), exact, "log2_fixed({n})");
        }
        // Powers of two sit on the lattice exactly.
        for t in 0..16 {
            assert_eq!(log2_fixed(1 << t), 64 * t);
        }
        assert_eq!(log2_fixed(1 << 13), 832);
        assert_eq!(log2_fixed((1 << 16) - 1), 1023);
    }

    /// The sum of squares and both floor stages are exact, and lambda matches the direct
    /// evaluation `128*enc_big - 2624 + floor(64*log2(floor(V/2^17)))`.
    #[test]
    fn lambda_witness_is_the_exact_score_chain() {
        let cases = [
            (0u64, 0u64, 1 << 15, 115u64),
            (1 << 15, 300, 65535, 300),
            (65535, 310, 1 << 15, 290),
            (40000, 200, 50000, 209),
            (40000, 209, 50000, 200),
            (40000, 210, 50000, 198),
            (50000, 315, 65535, 300),
            (50000, 150, 40000, 400),
            (33000, 436, 65535, 435),
        ];
        for (x_norm, x_enc, sigma_norm, sigma_enc) in cases {
            let w = lambda_witness(x_norm, x_enc, sigma_norm, sigma_enc);
            let gap = x_enc.abs_diff(sigma_enc);
            let (big, small) = if x_enc >= sigma_enc && x_norm != 0 {
                (x_norm, sigma_norm)
            } else {
                (sigma_norm, x_norm)
            };
            let v = if gap >= FAR_GAP {
                big * big
            } else {
                big * big + ((small * small) >> (2 * gap))
            };
            assert_eq!(w.sum_of_squares_top, v >> 17);
            assert_eq!(w.sum_of_squares_rest, v & ((1 << 17) - 1));
            assert_eq!(w.half_quotient, (small * small) >> w.near_gap);
            assert_eq!(w.quotient, (small * small) >> (2 * w.near_gap));
            assert!(w.half_remainder < w.near_gap_pow && w.remainder < w.near_gap_pow);
            assert_eq!(w.lambda, 128 * x_enc.max(sigma_enc) - 2624 + log2_fixed(v >> 17));
        }
    }

    /// The load-bearing approximation band: `64*(log2 S + SCORE_BIAS) - 1.1 < lambda <=
    /// 64*(log2 S + SCORE_BIAS)`, one-sided (all losses are floors).
    #[test]
    fn lambda_stays_in_the_one_sided_band() {
        let mut max_lambda = 0u64;
        for (i, &alpha) in alpha_samples().iter().enumerate() {
            for (j, &x) in bf16_samples().iter().enumerate() {
                let l2f = l2f_samples()[(i * 7 + j) % 30];
                let lam = lambda(Device::B200, alpha, x, l2f);
                max_lambda = max_lambda.max(lam);
                let au = bf16_to_f32(alpha) as f64 * bf16_to_f32(x) as f64;
                let sigma = 0.5 * bf16_to_f32(alpha) as f64 * bf16_to_f32(l2f) as f64;
                let s = au * au + sigma * sigma;
                let ideal = 64.0 * (s.log2() + SCORE_BIAS as f64);
                assert!(
                    (lam as f64) > ideal - 1.1 && (lam as f64) <= ideal + 1e-6,
                    "lambda={lam} ideal={ideal} (alpha={alpha:#06x} x={x:#06x} l2f={l2f:#06x})"
                );
                assert!((LAMBDA_MIN_LIVE..=LAMBDA_MAX).contains(&lam));
            }
        }
        assert!(max_lambda > LAMBDA_MIN_LIVE, "sweep exercises non-degenerate scores");
    }

    /// Bit-exactness against the uncut sum `n_big^2 + floor(n_small^2/2^g)`: identical on
    /// every gap — the far branch (gap >= 16) only drops a term that is already zero, since
    /// `n_small^2 < 2^32 <= 2^(2*gap)` there.
    #[test]
    fn the_sum_of_squares_matches_the_uncut_form_on_every_gap() {
        let mut live_quotients = 0u64;
        for gap in 0..=20 {
            for (x_norm, sigma_norm) in [(1 << 15, 65535), (40000, 50000), (65535, 1 << 15), (33000, 60000)] {
                let w = lambda_witness(x_norm, 300, sigma_norm, 300 - gap);
                let uncut = x_norm * x_norm + ((sigma_norm * sigma_norm) >> (2 * gap));
                assert_eq!(w.lambda, 128 * 300 - 2624 + log2_fixed(uncut >> 17), "gap {gap}");
                live_quotients += u64::from(!w.gap_is_far && w.quotient != 0);
            }
        }
        assert!(live_quotients > 0, "the sweep exercises live quotients across the near range");
        // The widest in-scheme gaps (x = 0 scores sigma alone) sit on the same identity.
        let w = lambda_witness(0, 0, 40000, 300);
        assert_eq!(w.lambda, 128 * 300 - 2624 + log2_fixed((40000u64 * 40000) >> 17));
    }

    #[test]
    fn offset_matches_the_folded_constant_and_the_certificate_fits_rc16() {
        assert_eq!(SKIP_THRESHOLD_OFFSET, 45_952);
        assert_eq!(
            SKIP_THRESHOLD_OFFSET,
            64 * (16 + 2 * SCORE_BIAS) - 128 * GRID_EXPONENT_BIAS as u64
        );
        // The lowest possible encoded grid is still high enough that every
        // satisfiable non-skip key fits the RC16 range.
        const {
            assert!(2 * LAMBDA_MAX - 128 * 114 - SKIP_THRESHOLD_OFFSET < 1 << 16);
        }
    }

    /// Bounds the integer skip rule's error against the real-valued threshold.
    #[test]
    fn skip_is_sandwiched_around_the_ideal_rule() {
        let alphas = alpha_samples();
        let codes = bf16_samples();
        let l2fs = l2f_samples();
        let anchors: Vec<f64> = vec![0.0, 2f64.powi(-18), 2f64.powi(-9), 1.0, 1234.5, 2f64.powi(40)];
        let (mut skips, mut nonskips) = (0u64, 0u64);
        for (i, &aa) in alphas.iter().enumerate() {
            let ab = alphas[(i * 7 + 3) % alphas.len()];
            for (j, &xa) in codes.iter().enumerate() {
                let xb = codes[(j * 13 + 5) % codes.len()];
                let l2a = l2fs[(j * 17 + 1) % l2fs.len()];
                let l2b = l2fs[(j * 29 + 11) % l2fs.len()];
                let la = lambda(Device::B200, aa, xa, l2a);
                let lb = lambda(Device::B200, ab, xb, l2b);
                let au = bf16_to_f32(aa) as f64 * bf16_to_f32(xa) as f64;
                let bu = bf16_to_f32(ab) as f64 * bf16_to_f32(xb) as f64;
                let si = 0.5 * bf16_to_f32(aa) as f64 * bf16_to_f32(l2a) as f64;
                let sj = 0.5 * bf16_to_f32(ab) as f64 * bf16_to_f32(l2b) as f64;
                let v = (au * au + si * si) * (bu * bu + sj * sj) / 2097152.0; // 2^21
                for &m in &anchors {
                    let e_grid = grid_exponent(Device::B200, m, m);
                    let int_skip = skip(la, lb, e_grid);
                    let threshold = if m == 0.0 {
                        0.0
                    } else {
                        let e = m.log2().floor() as i32;
                        let q = (e - Device::B200.fp8_window_bits() as i32).max(e - M_WINDOW_BITS as i32);
                        2f64.powi(2 * q - 5)
                    };
                    if v < threshold {
                        assert!(int_skip, "an ideal skip must be an integer skip");
                    }
                    if int_skip {
                        assert!(
                            v < threshold * 2f64.powf(2.2 / 64.0),
                            "integer skip must hug the ideal boundary"
                        );
                        skips += 1;
                    } else {
                        nonskips += 1;
                    }
                }
            }
        }
        assert!(skips > 0 && nonskips > 0, "sweep must exercise both outcomes");
    }

    #[test]
    fn only_a_jointly_zero_grid_disables_skipping() {
        assert!(!skip(0, 0, 0));
        assert!(!skip(LAMBDA_MIN_LIVE, LAMBDA_MIN_LIVE, 0));
        assert!(skip(0, LAMBDA_MIN_LIVE, 300));
        assert!(skip(LAMBDA_MIN_LIVE, 0, 300));
        assert_eq!(grid_exponent(Device::H100, 0.0, 0.0), 0);
        assert_ne!(grid_exponent(Device::H100, 1.0, 0.0), 0);
        assert_ne!(grid_exponent(Device::H100, 0.0, 1.0), 0);
    }

    #[test]
    fn grid_exponent_applies_device_z_and_shared_m_branches() {
        assert_eq!(grid_exponent(Device::H100, 128.0, 256.0), 151);
        assert_eq!(grid_exponent(Device::B200, 256.0, 256.0), 147);
        // On H100, a large window-local c can dominate the cell-wide M branch.
        assert_eq!(grid_exponent(Device::H100, 1024.0, 256.0), 154);
    }

    #[test]
    fn removed_check_3_has_no_independent_rejection_gate() {
        // This represents a large replay magnitude against arbitrarily tiny sigma,
        // which the removed tamed-products predicate rejected. The consolidated
        // policy depends only on E_GRID and lambda: high-score summands do not skip.
        let e_grid = grid_exponent(Device::B200, 256.0, 256.0);
        assert!(!skip(LAMBDA_MAX, LAMBDA_MAX, e_grid));
        assert_eq!(budget(1024, 1, 1), 51);
    }

    #[test]
    fn lambda_min_is_attained_at_the_domain_floor() {
        // alpha = 2^-120 (exp field 7), x = 0, l2f = 2^-32 (exp field 95): sigma alone,
        // sigma_norm = 2^15, sigma_enc = 115, sum of squares 2^30, kappa 2^13.
        assert_eq!(lambda(Device::B200, 7 << 7, 0, 95 << 7), LAMBDA_MIN_LIVE);
        assert_eq!(lambda(Device::H100, 7 << 7, 0, 95 << 7), LAMBDA_MIN_LIVE + 128);
    }

    #[test]
    fn budget_matches_the_plaintext_comparison() {
        for (k, h, w) in [(1024, 4, 64), (2048, 16, 16), (32, 5, 7), (65536, 32, 64), (100, 2, 10)] {
            let t = budget(k, h, w);
            let summands = (k as u64) * (h as u64) * (w as u64);
            assert_eq!(t, summands / 20);
            assert!(20 * t <= summands, "budget itself must pass");
            assert!(20 * (t + 1) > summands, "budget + 1 must fail");
        }
        assert_eq!(budget(20, 1, 1), 1);
        assert_eq!(budget(2048, 4, 16), 2048 * 64 / 20);
    }
}
