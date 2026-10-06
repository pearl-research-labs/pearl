//! Per-row norms, scaling constants and policy checks used by InputQuant.
//!
//! # Inputs and number conventions
//!
//! Each live row receives one completed InputQuant group. Its tuple contains the
//! framed sum `S`, doubled frame exponent `F`, maximum decoded absolute value,
//! claims for `alpha` and `beta`, and the dead-entry threshold and count.
//! The verifier supplies `k` and `Wl2 = 27 - ceil(log2(k))` as public inputs.
//!
//! The represented row mean square is:
//!
//! ```text
//! v = S * 2^(F - 268 - Wl2) / k
//! ```
//!
//! [`super::super::input_quant_stark::stark`] defines the block floors that produce
//! `S` and `F`. InputQuant keeps `S < 2^61`; this AIR allows `S < 2^62`.
//! Both bounds lie below the Goldilocks modulus `p = 2^64 - 2^32 + 1`.
//!
//! For a finite nonnegative BF16 value, define:
//!
//! ```text
//! M(V)  = 128*(1 - exp_is_zero) + mantissa
//! E*(V) = exp + exp_is_zero
//! V     = M(V) * 2^(E*(V) - 134)
//! ```
//!
//! `ceil(log2(k))` is the smallest integer `c` with `k <= 2^c`.
//!
//! Thus the canonical real row mean-square is
//! `v = S * 2^(E_MAX - 268 - Wl2) / k`, where `268 = 2*134`. The envelope
//! `1024 <= k <= 2^16` gives `11 <= Wl2 <= 17`. InputQuant's format envelope proves
//! `S < 2^61`; this table enforces the slightly wider `S < 2^62`, still below the Goldilocks
//! modulus `p = 2^64 - 2^32 + 1`.
//!
//! # Scale chain
//!
//! `RNE_bf16` means one round to finite bf16 using round-to-nearest, ties-to-even.
//!
//! Let `delta` be the committed device ratio (`1` on Hopper, `1/2` on Blackwell), noise rank
//! `r`, and `N = 256`. The verifier derives the public bf16 constants
//! `dr = RNE_bf16(delta*sqrt(r))` and
//! `dos = RNE_bf16(delta*sqrt(r)/N^2)`. Each row proves:
//!
//! 1. `l2 = grid4(RNE_bf16(sqrt(v)))`, where `grid4` rounds the bf16 code to the nearest
//!    multiple of four, ties upward; this clears the two low code bits as required by the
//!    protocol;
//! 2. `linf`, the nonnegative bf16 value encoded by `MAX_ABS`;
//! 3. `l2f = max(l2, 2^-32)` and `linf_f = max(linf, 2^-32)` — the scheme's norm floor (the
//!    reference `Fp8QuantScheme.row_norms`), which keeps the division below finite and the
//!    noise scale positive on (near-)zero rows;
//! 4. `noised_bound = RNE_bf16(dr*l2f + linf_f)`; no separate denominator floor follows — the
//!    reference has none, and `noised_bound >= linf_f >= 2^-32` anyway (RNE is monotone and
//!    `linf_f` is representable);
//! 5. `alpha = RNE_bf16(448/noised_bound)`, where 448 is the largest finite E4M3 magnitude;
//! 6. `beta = RNE_bf16(RNE_bf16(alpha*l2f)*dos)`, preserving the reference operation order.
//!
//! Rows are ordered A then B and padded with zero tuples. Padding is excluded from the group
//! lookup.
//!
//! # Jackpot liveness gate (check 1)
//!
//! Each row also carries the group's `DEAD_COUNT` (CTL-bound to InputQuant, which proves the
//! per-element device-threshold certificates against the group's `DEAD_BOUND`). T1 is the CTL
//! binding itself: the looked tuple's bound component is the affine expression
//! device-specific BF16 code of `tau_idle * delta * l2f`, so the bound needs no Scale column.
//! T2 accumulates the
//! counts into per-side running totals (frozen through padding), and on the last row the T3
//! RC16 gates each total against its `DEAD_LIMIT` public input
//! `floor(eps_idle * side_entries)` — the integer form of the plaintext
//! `dead as f64 <= eps_idle * entries as f64` — through a boolean high-bit witness
//! (the gate covers slacks up to 2^17; at `eps_idle = 1/64` the honest slack
//! stays below 2^16).
//!
//! # Jackpot noise-floor gate (check 2)
//!
//! Every opened row must carry noise of scale at least `sigma_min` in quantized units:
//! `sigma = delta * alpha * l2f >= sigma_min`, with the whitepaper-fixed `sigma_min = 1`.
//! S2 normalizes H4's exact pre-rounding significand product, giving the affine encoding
//! `SIGMA_ENC = E(alpha) + E(l2f) + SIGMA_SIG_IS_WIDE + offset(device) = e(sigma) + 268`.
//! Therefore one unfiltered RC16 lookup of `SIGMA_ENC - 268` proves `sigma >= 1`; its maximum
//! is only 255 on the pinned exponent domain, while a negative value wraps far outside the
//! table range. This is bit-for-bit the plaintext policy comparison: the native check computes
//! `delta * alpha * l2f` in f64, where the product of two 8-bit significands is exact.
//!
//! # Exact square-root check
//!
//! The AIR avoids floating point by moving to the **hat frame**, which multiplies values by
//! `2^127`. Squaring therefore multiplies `v` by `2^254`, giving
//! `v_hat = S*2^(E_MAX - 14 - Wl2)/k`; the otherwise surprising 14 is
//! `268 - 254`. A bf16 claim with integer significand `t = M(l2_claim)` and effective exponent
//! `f = E*(l2_claim)` represents `t*2^(f-7)` in this frame.
//!
//! In quarter-ulp units, the exact round-to-nearest-even acceptance boundaries are:
//!
//! ```text
//! dr  = RNE_bf16(delta*sqrt(r))
//! dos = RNE_bf16(delta*sqrt(r)/N^2)
//! ```
//!
//! Multiplication by four expresses quarter ulps; `+/-2` is half an ulp. The honest witness
//! sets `IS_BOTTOM = 1` exactly for `t = 128, f >= 2`, where the next lower bf16 lies in the
//! finer preceding binade and the lower midpoint is one quarter ulp below the claim. Bounds are strict exactly when
//! `t` is odd, implementing ties-to-even. A zero `S` forces positive zero; a zero claim on
//! nonzero `S` keeps only the upper bound. The implementation clears powers of two and compares
//! bounded 16-bit limbs, so these are integer—not field-wrapped—inequalities.
//!
//! 1. `l2 = grid4(RNE_bf16(sqrt(v)))`. This is the row's root mean square (RMS).
//!    `grid4` rounds the BF16 code to the nearest multiple of four, with ties upward.
//! 2. `linf` equals the maximum decoded `|X|` supplied by InputQuant.
//! 3. `l2f = max(l2, 2^-32)` and `linf_f = max(linf, 2^-32)`.
//! 4. `noised_bound = RNE_bf16(dr*l2f + linf_f)`, with one fused multiply-add rounding.
//! 5. `alpha = RNE_bf16(448/noised_bound)`, where 448 is the largest finite E4M3 magnitude.
//! 6. `beta = RNE_bf16(RNE_bf16(alpha*l2f)*dos)`, preserving both rounding steps.
//!
//! The norm floors keep `alpha` finite and the noise scale positive on zero rows.
//! The denominator needs no further floor: rounding is monotone and
//! `noised_bound >= linf_f >= 2^-32`.
//!
//! # Entry liveness (jackpot check 1)
//!
//! InputQuant proves and counts the per-element condition `|X| >= 4*l2f`.
//! The factor four is the default `tau_idle * DELTA = 8 * 0.5`.
//! The group-tuple lookup binds the threshold to the BF16 code `code(l2f) + 256`;
//! adding 256 raises the exponent field by two, multiplying the value by four.
//!
//! Scale accumulates the counts separately for A and B. On the last trace row,
//! the range checks require:
//!
//! ```text
//! dead_total_A <= floor(eps_idle * h * k)
//! dead_total_B <= floor(eps_idle * w * k)
//! ```
//!
//! # Noise floor (jackpot check 2)
//!
//! Every live row must satisfy `sigma = alpha*l2f/2 >= 1`. Both factors are positive
//! normal BF16 values. Their exact product can be written as:
//!
//! ```text
//! SIG   = M(alpha)*M(l2f),          2^14 <= SIG < 2^16
//! E_SUM = E*(alpha) + E*(l2f)
//! alpha*l2f = SIG * 2^(E_SUM - 268)
//! ```
//!
//! The floor check therefore becomes:
//!
//! ```text
//! alpha*l2f >= 2  iff  SIG >= 2^(269 - E_SUM)
//!                iff  E_SUM >= 255, or (E_SUM = 254 and SIG >= 2^15)
//! ```
//!
//! The significand bounds rule out `E_SUM <= 253` and make `E_SUM >= 255`
//! sufficient. Group F enforces the remaining choice with a branch bit and range
//! checks. Scale exports the exact sigma significand and exponent to Tamed, and
//! a normalized version of the same value to InputQuant for summand scoring.
//!
//! # Square-root certificate
//!
//! The square-root check uses a rescaled value, called the hat frame: multiply
//! the square root by `2^127`, and hence its square by `2^254`. Then:
//!
//! ```text
//! v_hat = 2^254*v = S*2^(F - 14 - Wl2)/k,    14 = 268 - 254
//! y_hat = t*2^(f - 7),                      t = M(claim), f = E*(claim)
//! ```
//!
//! For the claimed BF16 square root, define the rounding boundaries in units of
//! one quarter of its binade's spacing. A binade is the interval between successive
//! powers of two. The boundaries and the squared acceptance interval are:
//!
//! ```text
//! B_LO = 4*t - 2 + is_bottom
//! B_HI = 4*t + 2
//! B_LO^2*k <= S*2^(F + 4 - Wl2 - 2*f) <= B_HI^2*k
//! ```
//!
//! `is_bottom = 1` when `t = 128` and `f >= 2`. At such a power of two, the
//! preceding binade has half the spacing, so the lower midpoint is one quarter
//! unit away rather than one half. Both inequalities are strict when `t` is odd;
//! this implements ties to even. `S = 0` forces positive zero. A zero claim on
//! nonzero `S` needs only the upper bound.
//!
//! Q5 converts the power-of-two alignment into whole limbs and a small bit shift:
//!
//! ```text
//! G      = 2*f + Wl2 - 4 - F
//! G + 32 = 16*q + 15 - r2,    q in [0, 4], r2 in [0, 15]
//! SS     = S*2^r2
//! ```
//!
//! The equivalent bracket is:
//!
//! ```text
//! (B_LO^2*k*2^15)*2^(16*q) <= SS*2^32 <= (B_HI^2*k*2^15)*2^(16*q)
//! ```
//!
//! Multiplication by `2^(16*q)` shifts the claim by `q` base-`2^16` limbs;
//! `2^32` shifts `SS` by two limbs. Range-checked digits and borrow bits enforce
//! these comparisons over the integers. An initial borrow of `t mod 2` makes
//! ties fail exactly when `t` is odd.
//!
//! # Trace binding
//!
//! Rows are ordered A then B, followed by padding excluded from imports and counts.
//! The verifier recomputes and binds the schedule columns. [`super::ctl`] supplies
//! the range and rounding lookups, including the `2^17` bound that prevents an
//! oversized FMA significand from selecting another RNERND rounding slot.

use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use std::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use starky::stark::Stark;

use super::columns::{
    ALIGNED_LOWER_LIMBS, ALIGNED_UPPER_LIMBS, CLAIM_LIMBS, DEAD_LIMIT_A_PUBLIC_INPUT, DEAD_LIMIT_B_PUBLIC_INPUT,
    DOS_EXP_PUBLIC_INPUT, DOS_MANTISSA_PUBLIC_INPUT, DR_EXP_PUBLIC_INPUT, DR_MANTISSA_PUBLIC_INPUT, FmaBlock, K_PUBLIC_INPUT,
    L2_SUM_LIMBS, MulBlock, NUM_SCALE_COLUMNS, NUM_SCALE_PUBLIC_INPUTS, ScaleColumnsView, UPPER_BORROW_BITS, WL2_PUBLIC_INPUT,
};
use crate::v4::api::compute::{bf16_div, bf16_fma, bf16_max, bf16_mul};
use crate::v4::api::dtype::f32_to_bf16;
use crate::v4::api::jackpot_policy::JackpotPolicy;
use crate::v4::api::prequant::{BLOCK_SIZE, l2_frame_width};
use crate::v4::api::public_params::{Device, PublicParams};
use crate::v4::api::quantization::{Fp8E4M3Quant, MAX_E4M3, NORM_FLOOR};

// Frame constants (bf16 field conventions)

/// `value = M * 2^(E* - 134)`: the bf16 unit offset (127 bias + 7 mantissa denominator).
const EXP_UNIT: i64 = 134;
/// A product of two bf16 significands lives at unit `2^(E*a + E*b - 268)`.
const PROD_UNIT: i64 = 2 * EXP_UNIT;
/// bf16 code of `MAX_E4M3 = 448.0` (DIV448's numerator; asserted in `ScaleProgram::new`).
pub(crate) const CODE_448: u16 = 0x43E0;
/// BF16 code of the floor applied to both norms before scale derivation.
/// `2^-32` has exponent field `127 - 32 = 95` and mantissa zero.
/// Checked against `quantization::NORM_FLOOR` in `ScaleProgram::new`.
pub(crate) const NORM_FLOOR_CODE: u16 = 0x2F80;
// The floor is a normal power of two, so its floored fields are M = 128, E* = exponent field.
const _: () = assert!(NORM_FLOOR_CODE & 0x7F == 0 && NORM_FLOOR_CODE >> 7 >= 1 && NORM_FLOOR_CODE >> 7 <= 254);
/// The floor's effective exponent `E*(2^-32) = 95` (its significand is 128).
const NORM_FLOOR_E_STAR: u64 = (NORM_FLOOR_CODE >> 7) as u64;
/// CLAMP22 key embedding: adding 400 maps its signed fade-depth domain `[-400, 200]` to keys
/// `[0, 600]`.
const CLAMP22_EMBED: i64 = 400;
/// The FMA clamp argument is `-133 - KEY_SCALE`, where -133 is bf16's smallest-subnormal
/// exponent; adding 400 gives key `267 - KEY_SCALE`.
const FMA_CLAMP_KEY_CONST: i64 = 267;
/// A product's LSB scale is `E*a + E*b - 268`, so `-133 - LSB_SCALE` is
/// `135 - E*a - E*b`; adding 400 gives key `535 - E*a - E*b`.
const MUL_CLAMP_KEY_CONST: i64 = 535;
/// RNERND slot bias: the table's slot index is the *signed* subnormal fade cut plus 7. Signed
/// cuts run `[-7, 18]`: -7 already covers every `KEY_SCALE >= -126` (all such small-significand
/// results are normal), and 18 provably rounds every `V < 2^17` to zero.
pub(crate) const CUT_BIAS: i64 = 7;
/// RNERND's largest slot index (signed cut 18); the slot count is `SLOT_MAX + 1 = 26`.
pub(crate) const SLOT_MAX: i64 = 25;
/// Largest exponent gap generated by ScaleStark's FMA fill; the shared POW2D table
/// also serves InputQuant gaps 18 and 19.
const POW2D_MAX: u64 = 17;
/// `B_HI^2 - B_LO^2 = 32t - 1021*IS_BOTTOM` (with `IS_BOTTOM => t = 128`, so `8*b*t = 1024*b`
/// and `(4t+2)^2 - (4t-2+b)^2 = 32t + 3b - 8bt = 32t - 1021b`).
const UPPER_SQ_BOTTOM_DELTA: i64 = 1021;

// Program, tuples and trace generation

/// The per-group aggregate tuple ScaleStark receives from InputQuantStark over the group-tuple
/// CTL: everything this AIR knows about the raw row.
#[derive(Clone, Copy, Debug)]
pub struct ScaleRowTuple {
    /// The exact block-integer L2 frame sum `S < 2^62` (InputQuantStark's group B).
    pub l2_frame_sum: u64,
    /// The row frame exponent `E_MAX = max over live blocks of 2*E*(scale_b)` (0 if none) —
    /// already doubled/even.
    pub frame_doubled_scale_exponent: u32,
    /// bf16 code of the running max `|X|` over the *decoded* elements, sign cleared.
    pub max_abs: u32,
    /// The row's jackpot dead-entry count `|{u : |X_u| >= 4*l2f}|` (check 1).
    pub dead_count: u32,
}

impl ScaleRowTuple {
    /// Reproduce InputQuant's aggregate for one row of int8 values and BF16 block scales.
    /// Uses the block-floor RMS calculation described in the module overview,
    /// the maximum decoded magnitude, and the resulting dead-entry count.
    ///
    /// Panics on non-finite scales and on decoded elements overflowing bf16.
    pub fn from_prequant_row_for_device(int8: &[i8], scales: &[u16], wl2: u32, device: Device) -> Self {
        assert!(int8.len().is_multiple_of(BLOCK_SIZE), "row length is a multiple of 8");
        assert_eq!(scales.len(), int8.len() / BLOCK_SIZE, "one scale per block");

        // Decode with open_prequant's BF16 rounding before taking the maximum magnitude.
        let mut abs_codes = Vec::with_capacity(int8.len());
        let mut max_abs = 0u32;
        for (j, &v) in int8.iter().enumerate() {
            let scale = scales[j / BLOCK_SIZE];
            assert!((scale & 0x7FFF) >> 7 != 255, "non-finite scale");
            let x = if v == 0 {
                0
            } else {
                let int_code = f32_to_bf16(f32::from(v)).expect("int8 is exactly representable in bf16");
                bf16_mul(int_code, scale).expect("decoded element overflows bf16")
            };
            let abs = u32::from(x & 0x7FFF);
            abs_codes.push(abs);
            max_abs = max_abs.max(abs);
        }

        // Zero blocks contribute nothing and do not affect the common frame exponent.
        let mut frame_doubled_scale_exponent = 0u64;
        let blocks: Vec<(u64, u64)> = int8
            .chunks(BLOCK_SIZE)
            .zip(scales)
            .map(|(chunk, &scale)| {
                let n: u64 = chunk.iter().map(|&v| (i64::from(v) * i64::from(v)) as u64).sum();
                let sf = Bf16Fields::from_code(scale & 0x7FFF);
                let p = sf.m() * sf.m() * n;
                let e2 = if p == 0 { 0 } else { 2 * sf.e_star() as u64 };
                frame_doubled_scale_exponent = frame_doubled_scale_exponent.max(e2);
                (p, e2)
            })
            .collect();
        let mut s: u64 = 0;
        for (p, e2) in blocks {
            if p == 0 {
                continue;
            }
            let sigma = frame_doubled_scale_exponent - e2;
            // p * 2^wl2 < 2^(33 + 16) = 2^49; shifts beyond the live window floor to 0.
            s += if sigma >= 64 { 0 } else { (p << wl2) >> sigma };
        }
        assert!(s < 1 << 62, "frame sum exceeds the Q1 cap");

        // Derive the row RMS with the same square-root rounding as the circuit.
        let claim = if s == 0 {
            SqrtClaim {
                exp: 0,
                mantissa: 0,
                exp_is_zero: true,
            }
        } else {
            rne_sqrt_hat(s, frame_doubled_scale_exponent as i64, i64::from(wl2), int8.len() as u64)
        };
        // Grid-round and floor the RMS; adding 256 to its code gives the threshold 4*l2f.
        let snapped = ((claim.exp << 7) + claim.mantissa + 2) & !3;
        let dead_bound = snapped.max(u64::from(NORM_FLOOR_CODE)) + device.liveness_code_shift();
        let dead_count = abs_codes.iter().filter(|&&abs| u64::from(abs) >= dead_bound).count() as u32;

        Self {
            l2_frame_sum: s,
            frame_doubled_scale_exponent: frame_doubled_scale_exponent as u32,
            max_abs,
            dead_count,
        }
    }
}

/// The public geometry of one FP8 job, as ScaleStark sees it: `h` A-rows and `w` B-rows of
/// `k` elements, noise rank `r`. The AIR identity is program-independent — the geometry
/// enters only through public inputs (`k`, `Wl2`, and the jackpot `DEAD_LIMIT`s).
#[derive(Clone, Debug)]
pub struct ScaleProgram {
    /// A-side rows.
    pub h: usize,
    /// B-side rows.
    pub w: usize,
    /// Row length. Protocol preconditions: `k % 32 == 0`,
    /// `PublicParams::MIN_K <= k <= PublicParams::MAX_K` — they keep `Wl2 in [11, 17]`,
    /// inside the sqrt shift window's honest-completeness proof.
    pub k: usize,
    /// Noise rank (the `r` of `dr = bf16(DELTA * sqrt(r))`).
    pub r: usize,
    /// Committed arithmetic profile.
    pub device: Device,
}

impl ScaleProgram {
    /// Builds the program, asserting the geometry preconditions the AIR's soundness bounds
    /// assume. Production reaches this constructor only after `PublicParams` validates the
    /// same envelope with errors; these assertions also protect direct internal callers.
    pub fn new_for_device(h: usize, w: usize, k: usize, r: usize, device: Device) -> Self {
        assert!(h >= 1 && w >= 1, "empty geometry");
        assert!(
            k.is_multiple_of(32) && (PublicParams::MIN_K..=PublicParams::MAX_K).contains(&k),
            "unsanctioned k"
        );
        assert!(r >= 1, "noise rank must be positive");
        assert_eq!(
            f32_to_bf16(MAX_E4M3).expect("448 is finite"),
            CODE_448,
            "CODE_448 is the bf16 code of MAX_E4M3 (DIV448's numerator)"
        );
        assert_eq!(
            f32_to_bf16(NORM_FLOOR).expect("2^-32 is finite"),
            NORM_FLOOR_CODE,
            "NORM_FLOOR_CODE is the bf16 code of the scheme's norm floor"
        );
        assert_eq!(
            JackpotPolicy::for_device(device).sigma_min,
            1.0,
            "the circuit freezes the whitepaper sigma_min = 1"
        );
        Self { h, w, k, r, device }
    }

    /// `Wl2` — the block-L2 frame width ([`l2_frame_width`]); `in [11, 17]` for sanctioned `k`.
    pub fn wl2(&self) -> u32 {
        l2_frame_width(self.k).expect("sanctioned k")
    }

    /// Live rows: one per matrix row, A rows then B rows.
    pub fn live_rows(&self) -> usize {
        self.h + self.w
    }

    /// Trace height: the live rows padded to the next power of two with all-zero-tuple
    /// `IS_PAD = 1` rows (excluded from the group-tuple CTL).
    pub fn num_rows(&self) -> usize {
        self.live_rows().next_power_of_two()
    }

    /// `dr = bf16(DELTA * sqrt(r))` — mirrors the private `quantization::delta_r_bf16`
    /// bit for bit (kept in sync by the derive_row_scales differential test).
    pub fn dr_code(&self) -> u16 {
        Fp8E4M3Quant::new(self.device).delta_r_bf16(self.r).expect("dr is finite")
    }

    /// `dos = bf16(DELTA * sqrt(r) / NOISE_TARGET_NORM^2)` — mirrors the private
    /// `quantization::delta_over_std_bf16`.
    pub fn dos_code(&self) -> u16 {
        Fp8E4M3Quant::new(self.device)
            .delta_over_std_bf16(self.r)
            .expect("dos is finite")
    }

    /// A side's dead-entry allowance `floor(eps_idle * entries)`, in f64 exactly like the
    /// plaintext gate `dead as f64 <= eps_idle * entries as f64` (an integer `dead` passes
    /// that gate iff `dead <= floor(...)`). `< 2^16` for every sanctioned geometry
    /// (`side*k < 2^22`, `eps_idle = 1/64`), comfortably inside the T3 gate's RC16 +
    /// boolean-high-bit capacity of `2^17`.
    fn dead_limit(&self, entries: usize) -> u64 {
        let policy = JackpotPolicy::default();
        // The device-specific code shift T1/InputQuant bake in is exactly
        // `tau_idle * delta`, an integral power of two.
        debug_assert_eq!(
            policy.tau_idle * self.device.delta(),
            2f64.powi(self.device.lg2_delta() + 3),
            "tau_idle * delta must remain an exact power of two"
        );
        let limit = (policy.eps_idle * entries as f64).floor() as u64;
        assert!(limit < 1 << 17, "DEAD_LIMIT exceeds the high-bit RC16 gate");
        limit
    }

    /// The expected public inputs of a proof under this program: the dr/dos decode fields,
    /// the geometry parameters `k` / `Wl2`, and the jackpot `DEAD_LIMIT`s — all pure
    /// functions of the program: the verifier assembles its expectation from the statement
    /// alone (`crate::v4::api::zk` pins every slot), and `generate_trace` produces the
    /// identical values.
    pub fn public_inputs<F: RichField>(&self) -> [F; NUM_SCALE_PUBLIC_INPUTS] {
        let dr = Bf16Fields::from_code(self.dr_code());
        let dos = Bf16Fields::from_code(self.dos_code());
        assert!(!dr.exp_is_zero && !dos.exp_is_zero, "dr/dos are structurally normal");
        let mut pis = [F::ZERO; NUM_SCALE_PUBLIC_INPUTS];
        pis[DR_EXP_PUBLIC_INPUT] = F::from_canonical_u64(dr.exp);
        pis[DR_MANTISSA_PUBLIC_INPUT] = F::from_canonical_u64(dr.mantissa);
        pis[DOS_EXP_PUBLIC_INPUT] = F::from_canonical_u64(dos.exp);
        pis[DOS_MANTISSA_PUBLIC_INPUT] = F::from_canonical_u64(dos.mantissa);
        pis[K_PUBLIC_INPUT] = F::from_canonical_u64(self.k as u64);
        pis[WL2_PUBLIC_INPUT] = F::from_canonical_u64(u64::from(self.wl2()));
        pis[DEAD_LIMIT_A_PUBLIC_INPUT] = F::from_canonical_u64(self.dead_limit(self.h * self.k));
        pis[DEAD_LIMIT_B_PUBLIC_INPUT] = F::from_canonical_u64(self.dead_limit(self.w * self.k));
        pis
    }

    /// Generates the trace and public inputs from h A-row tuples followed by w B-row tuples.
    /// Panics on invalid tuples or rejected policy/rounding results.
    pub fn generate_trace<F: RichField>(
        &self,
        a_rows: &[ScaleRowTuple],
        b_rows: &[ScaleRowTuple],
    ) -> (Vec<[F; NUM_SCALE_COLUMNS]>, [F; NUM_SCALE_PUBLIC_INPUTS]) {
        assert_eq!(a_rows.len(), self.h, "one tuple per A row");
        assert_eq!(b_rows.len(), self.w, "one tuple per B row");
        let num_rows = self.num_rows();
        let wl2 = self.wl2();
        let k = self.k as u64;

        // Reject policy failures before constructing the trace.
        let total_dead = |rows: &[ScaleRowTuple]| rows.iter().map(|t| u64::from(t.dead_count)).sum::<u64>();
        let (limit_a, limit_b) = (self.dead_limit(self.h * self.k), self.dead_limit(self.w * self.k));
        let (total_a, total_b) = (total_dead(a_rows), total_dead(b_rows));
        assert!(
            total_a <= limit_a && total_b <= limit_b,
            "policy-rejected witness: dead entries exceed the eps_idle allowance"
        );

        // The all-zero tuple filling the trailing pad rows (a valid zero-sum row fill).
        let pad_tuple = ScaleRowTuple {
            l2_frame_sum: 0,
            frame_doubled_scale_exponent: 0,
            max_abs: 0,
            dead_count: 0,
        };

        let mut rows: Vec<[F; NUM_SCALE_COLUMNS]> = Vec::with_capacity(num_rows);
        let (mut running_dead_a, mut running_dead_b) = (0u64, 0u64);
        for i in 0..num_rows {
            let is_pad = i >= self.live_rows();
            let is_a = i < self.h;
            let tuple = if is_pad {
                &pad_tuple
            } else if is_a {
                &a_rows[i]
            } else {
                &b_rows[i - self.h]
            };
            let mut row = [F::ZERO; NUM_SCALE_COLUMNS];
            {
                let v: &mut ScaleColumnsView<F> = row.borrow_mut();

                // Sqrt claim (group Q) + everything downstream of the tuple.
                let claim = if tuple.l2_frame_sum == 0 {
                    SqrtClaim {
                        exp: 0,
                        mantissa: 0,
                        exp_is_zero: true,
                    }
                } else {
                    rne_sqrt_hat(
                        tuple.l2_frame_sum,
                        i64::from(tuple.frame_doubled_scale_exponent),
                        i64::from(wl2),
                        k,
                    )
                };
                // Pad rows key like B row 0 inside fill_data_row and are re-keyed to the
                // all-zero verifier-known shape below (the CTL filter drops them anyway).
                self.fill_data_row(v, tuple, is_a && !is_pad, if is_pad { self.h } else { i }, claim, true);
                if is_pad {
                    v.group_key = F::ZERO;
                    v.is_a_row = F::ZERO;
                    v.is_pad = F::ONE;
                }
                v.is_last_row = F::from_bool(i == num_rows - 1);

                // The jackpot running dead totals (group T; frozen through pads).
                if !is_pad {
                    if is_a {
                        running_dead_a += u64::from(tuple.dead_count);
                    } else {
                        running_dead_b += u64::from(tuple.dead_count);
                    }
                }
                v.running_dead_a = F::from_canonical_u64(running_dead_a);
                v.running_dead_b = F::from_canonical_u64(running_dead_b);
                // The T3 slack high bits, meaningful on the gate row only (honest fill
                // elsewhere is the same value; the totals are already final past the live
                // rows).
                v.dead_slack_hi_a = F::from_canonical_u64((limit_a - total_a) >> 16);
                v.dead_slack_hi_b = F::from_canonical_u64((limit_b - total_b) >> 16);
            }
            rows.push(row);
        }
        (rows, self.public_inputs())
    }

    /// Recomputes the leading schedule columns in trace order from public geometry.
    pub fn known_values<F: RichField>(&self) -> Vec<PolynomialValues<F>> {
        let num_rows = self.num_rows();
        let mut group_key = Vec::with_capacity(num_rows);
        let mut is_last_row = Vec::with_capacity(num_rows);
        let mut is_a_row = Vec::with_capacity(num_rows);
        let mut is_pad = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let pad = i >= self.live_rows();
            let is_a = i < self.h;
            let key = if pad {
                0
            } else if is_a {
                (i + 1) * self.k - 1
            } else {
                self.h * self.k + (i - self.h + 1) * self.k - 1
            };
            group_key.push(F::from_canonical_usize(key));
            is_last_row.push(F::from_bool(i == num_rows - 1));
            is_a_row.push(F::from_bool(is_a && !pad));
            is_pad.push(F::from_bool(pad));
        }
        [group_key, is_last_row, is_a_row, is_pad]
            .into_iter()
            .map(PolynomialValues::new)
            .collect()
    }

    /// Fills a row from its aggregate tuple and a chosen sqrt claim.
    /// Tamper tests pass `expect_valid = false` to rebuild a consistent downstream
    /// witness for a wrong claim, isolating rejection by the sqrt certificate.
    pub(crate) fn fill_data_row<F: RichField>(
        &self,
        v: &mut ScaleColumnsView<F>,
        tuple: &ScaleRowTuple,
        is_a: bool,
        row_index: usize,
        claim: SqrtClaim,
        expect_valid: bool,
    ) {
        let k = self.k as u64;
        let wl2 = self.wl2();
        let dr = Bf16Fields::from_code(self.dr_code());
        let dos = Bf16Fields::from_code(self.dos_code());

        // Tuple sanity (CTL-guaranteed for honest tuples; hard-asserted here).
        assert!(tuple.l2_frame_sum < 1 << 62, "S exceeds the Q1 cap");
        assert!(
            tuple.max_abs < 1 << 15 && tuple.max_abs >> 7 != 255,
            "MAX_ABS is not a finite code"
        );
        assert!(
            tuple.frame_doubled_scale_exponent <= 508 && tuple.frame_doubled_scale_exponent.is_multiple_of(2),
            "FRAME_DOUBLED_SCALE_EXPONENT is a doubled scale exponent"
        );
        // A block is live iff some element decodes nonzero: a nonzero `int * scale` is at
        // least `2^-133` (one unit of bf16's subnormal grid) and so never rounds to zero, so
        // the block sum and the decoded max vanish together.
        assert_eq!(tuple.l2_frame_sum == 0, tuple.max_abs == 0, "S = 0 iff the row is all zero");
        assert!(u64::from(tuple.dead_count) <= k, "at most k dead entries per row");

        // Verifier-known.
        let group_index = if is_a { row_index } else { row_index - self.h };
        let key = if is_a {
            (group_index + 1) * self.k - 1
        } else {
            self.h * self.k + (group_index + 1) * self.k - 1
        };
        v.group_key = F::from_canonical_usize(key);
        v.is_a_row = F::from_bool(is_a);

        // Tuple columns.
        v.l2_frame_sum = F::from_canonical_u64(tuple.l2_frame_sum);
        v.frame_doubled_scale_exponent = F::from_canonical_u64(u64::from(tuple.frame_doubled_scale_exponent));
        v.dead_count = F::from_canonical_u64(u64::from(tuple.dead_count));

        // Sqrt bracket witnesses (group Q).
        fill_sqrt_block(
            v,
            tuple.l2_frame_sum,
            u64::from(tuple.frame_doubled_scale_exponent),
            wl2,
            k,
            claim,
            expect_valid,
        );

        // Grid snap (group G) and linf decode (group N).
        let sqrt_code = (claim.exp << 7) + claim.mantissa;
        let snapped = (sqrt_code + 2) & !3;
        v.grid_snap_quotient = F::from_canonical_u64(snapped / 4);
        v.grid_snap_remainder = F::from_canonical_u64(sqrt_code + 2 - snapped);
        let l2 = Bf16Fields::from_code_u64(snapped);
        assert!(l2.exp <= 254, "policy-rejected witness: l2 snapped into the infinity code");
        v.l2_exp = F::from_canonical_u64(l2.exp);
        v.l2_mantissa = F::from_canonical_u64(l2.mantissa);
        v.l2_exp_is_zero = F::from_bool(l2.exp_is_zero);
        let linf = Bf16Fields::from_code_u64(u64::from(tuple.max_abs));
        v.linf_exp = F::from_canonical_u64(linf.exp);
        v.linf_mantissa = F::from_canonical_u64(linf.mantissa);
        v.linf_exp_is_zero = F::from_bool(linf.exp_is_zero);

        // Norm floors (group H0), the reference scheme's `row_norms`:
        // l2f = max(l2, 2^-32) and linf_f = max(linf, 2^-32), by code order on nonnegatives.
        let (l2f, l2_ge, l2_slack) = floor_norm(l2);
        v.l2_ge_floor = F::from_bool(l2_ge);
        v.l2_floor_order_slack = F::from_canonical_u64(l2_slack);
        v.l2_floored_significand = F::from_canonical_u64(l2f.m());
        v.l2_floored_exponent = F::from_canonical_u64(l2f.e_star() as u64);
        let (linf_f, linf_ge, linf_slack) = floor_norm(linf);
        v.linf_ge_floor = F::from_bool(linf_ge);
        v.linf_floor_order_slack = F::from_canonical_u64(linf_slack);
        v.linf_floored_significand = F::from_canonical_u64(linf_f.m());
        v.linf_floored_exponent = F::from_canonical_u64(linf_f.e_star() as u64);

        // The scale chain (group H), each step asserted bit-exact vs. native.
        let noised_bound_fma = fill_fma(dr, l2f, linf_f);
        write_fma(v, &noised_bound_fma);
        let nb_native = bf16_fma(self.dr_code(), l2f.code(), linf_f.code()).expect("native fma");
        assert_eq!(noised_bound_fma.out.code(), nb_native, "H1: FMA fill diverged from bf16_fma");

        // H3 divides by the noised bound directly — the reference has no denominator floor;
        // nb >= linf_f >= 2^-32 anyway (RNE is monotone and the floor is representable).
        let nb_code = noised_bound_fma.out.code();
        assert!(nb_code >= NORM_FLOOR_CODE, "H3: the noised bound sits above the norm floor");
        let alpha_code = bf16_div(CODE_448, nb_code).expect("H3: alpha division");
        let alpha = Bf16Fields::from_code(alpha_code);
        assert!(!alpha.exp_is_zero && alpha.exp <= 254, "H3: alpha is structurally normal");
        v.alpha_exp = F::from_canonical_u64(alpha.exp);
        v.alpha_mantissa = F::from_canonical_u64(alpha.mantissa);
        // S1 (jackpot check 3): the exact sigma significand M(alpha) * M(l2f).
        let sigma_significand = (128 + alpha.mantissa) * l2f.m();
        v.sigma_significand = F::from_canonical_u64(sigma_significand);
        // S2 (jackpot check 4): its width bit — the significand spans [2^14, 2^16).
        let sigma_sig_is_wide = sigma_significand >= 1 << 15;
        v.sigma_sig_is_wide = F::from_bool(sigma_sig_is_wide);
        // S3 (jackpot check 4): the normalized form, in [2^15, 2^16).
        v.normalized_sigma_significand = F::from_canonical_u64(sigma_significand << u64::from(!sigma_sig_is_wide));

        let b1 = fill_mul(alpha, l2f);
        write_mul(&mut v.alpha_l2_multiply, &b1);
        assert_eq!(
            b1.out.code(),
            bf16_mul(alpha_code, l2f.code()).expect("native mul"),
            "H4: MUL fill diverged from bf16_mul"
        );
        let b2 = fill_mul(b1.out, dos);
        write_mul(&mut v.beta_scale_multiply, &b2);
        assert_eq!(
            b2.out.code(),
            bf16_mul(b1.out.code(), self.dos_code()).expect("native mul"),
            "H5: MUL fill diverged from bf16_mul"
        );

        // The whole chain in one shot: alpha/beta must equal the native `derive_row_scales`
        // on the floored norms — the exact `noisy_quantize` flow (floor, FMA, DIV, MUL, MUL) —
        // on EVERY row, all-zero rows included.
        let native = Fp8E4M3Quant::new(self.device)
            .derive_row_scales(l2f.code(), linf_f.code(), self.r)
            .expect("native derive_row_scales");
        assert_eq!(
            (alpha_code, b2.out.code()),
            native,
            "H: the scale chain diverged from the native derive_row_scales"
        );

        // ---- The jackpot noise floor (group F): sigma = DELTA*alpha*l2f >= sigma_min = 1.
        // S2 makes its exact exponent encoding affine; the matching RC16 gate checks that the
        // encoding is at least 268, i.e. e(sigma) >= 0. ----
        let e_sum = alpha.e_star() + l2f.e_star();
        let sigma_enc = e_sum + i64::from(sigma_sig_is_wide) + self.device.sigma_encoding_offset() as i64;
        if expect_valid {
            assert!(
                sigma_enc >= PROD_UNIT,
                "policy-rejected witness: row noise scale below the sigma floor (jackpot check 2)"
            );
        }
    }
}

/// `max(x, 2^-32)` by code order (H0's trace-side mirror; code order = value order on
/// nonnegative bf16): the floored fields, the order bit, and the two-sided slack — asserted
/// bit-exact vs. the native `bf16_max`, the op behind the reference scheme's `row_norms` floor.
fn floor_norm(x: Bf16Fields) -> (Bf16Fields, bool, u64) {
    let ge = x.code() >= NORM_FLOOR_CODE;
    let slack = if ge {
        u64::from(x.code() - NORM_FLOOR_CODE)
    } else {
        u64::from(NORM_FLOOR_CODE - x.code()) - 1
    };
    let floored = if ge { x } else { Bf16Fields::from_code(NORM_FLOOR_CODE) };
    assert_eq!(floored.code(), bf16_max(x.code(), NORM_FLOOR_CODE), "H0 vs bf16_max");
    (floored, ge, slack)
}

// bf16 field plumbing and the shared RNERND reference

/// The (sign-free) bf16 field triple: `M = 128*(1 - EXP_IS_ZERO) + MANTISSA`,
/// `E* = EXP + EXP_IS_ZERO`, `value = M * 2^(E* - 134)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bf16Fields {
    exp: u64,
    mantissa: u64,
    exp_is_zero: bool,
}

impl Bf16Fields {
    fn from_code_u64(code: u64) -> Self {
        assert!(code < 1 << 15, "nonnegative finite bf16 code expected");
        let exp = code >> 7;
        assert!(exp <= 254, "finite bf16 code expected");
        Self {
            exp,
            mantissa: code & 0x7F,
            exp_is_zero: exp == 0,
        }
    }

    fn from_code(code: u16) -> Self {
        Self::from_code_u64(u64::from(code))
    }

    fn code(&self) -> u16 {
        (self.exp as u16) << 7 | self.mantissa as u16
    }

    /// Significand `M in {0} u [1, 255]`.
    fn m(&self) -> u64 {
        if self.exp_is_zero {
            self.mantissa
        } else {
            128 + self.mantissa
        }
    }

    /// Effective exponent `E* in [1, 254]`.
    fn e_star(&self) -> i64 {
        self.exp as i64 + i64::from(self.exp_is_zero)
    }
}

fn bit_length(x: u64) -> u32 {
    64 - x.leading_zeros()
}

/// The committed RNERND table's row function, keyed by the 17-bit significand `v` and the
/// CLAMP22-produced slot `slot = clamp(cut, -7, 18) + 7`, where `cut = -133 - KEY_SCALE` is the
/// *signed* subnormal fade depth of the significand's LSB scale. It RNE-rounds `v` at
/// `pos = max(width(v) - 8, cut)` (ties-to-even, mantissa-overflow renormalization) and
/// classifies the output; a negative `pos` is the exact encode-time left shift of a small
/// significand.
///
/// Returns `(OUT_MANTISSA, WIDTH_ADJUST, OUT_IS_ZERO, OUT_EXP_IS_ZERO)`:
/// - normal (`Q >= 128`): `OUT_MANTISSA = Q - 128`, `WIDTH_ADJUST = final pos + 134`, so a
///   normal output satisfies `OUT_EXP = KEY_SCALE + WIDTH_ADJUST`;
/// - subnormal (`Q < 128`: the cut dominated, so `pos = cut` puts `Q` exactly on bf16's
///   `2^-133` grid): `OUT_MANTISSA = Q` plain, `WIDTH_ADJUST = 0` (unused: the exponent
///   bindings are gated);
/// - zero: all zero, both flags set.
///
/// The pair `(v, slot)` fully determines the classification — no extra scale input is needed.
/// On every unclamped slot the fade depth pins `KEY_SCALE = -133 - cut` exactly; at the clamp
/// boundaries the behavior is uniform (slot 0 serves every `KEY_SCALE >= -126`, where any
/// nonzero result is normal with a scale-independent mantissa; slot 25 serves every
/// `KEY_SCALE <= -151`, where every `v < 2^17` rounds to zero).
pub(crate) fn rnernd_reference(v: u64, slot: u64) -> (u64, u64, bool, bool) {
    assert!(v < 1 << 17 && slot <= SLOT_MAX as u64, "RNERND key domain");
    if v == 0 {
        return (0, 0, true, true);
    }
    let cut = slot as i64 - CUT_BIAS;
    let w = i64::from(bit_length(v));
    let pos = (w - 8).max(cut);
    let mut q = if pos <= 0 {
        v << (-pos) as u32
    } else {
        let half = 1u64 << (pos - 1);
        let rem = v & ((1 << pos) - 1);
        let q0 = v >> pos;
        q0 + u64::from(rem > half || (rem == half && q0 & 1 == 1))
    };
    let mut pos_final = pos;
    if q == 256 {
        q = 128;
        pos_final += 1;
    }
    if q == 0 {
        return (0, 0, true, true);
    }
    if q >= 128 {
        (q - 128, (pos_final + 134) as u64, false, false)
    } else {
        // The cut dominated (width-dominated rounding always lands in [128, 255]), so the
        // result is subnormal.
        (q, 0, false, true)
    }
}

// The MUL and FMA gadget fills (trace-side mirrors of groups H4/H5 and H1)

/// Witness values of one MUL gadget instance.
struct MulFill {
    sig_product: u64,
    cut_depth: u64,
    width_adjust: u64,
    out: Bf16Fields,
    out_is_zero: bool,
}

/// `out = RNE_bf16(a * b)` on nonnegative operands, exactly as constraints M1-M6 prove it.
fn fill_mul(a: Bf16Fields, b: Bf16Fields) -> MulFill {
    let sig = a.m() * b.m();
    let cut = (MUL_CLAMP_KEY_CONST - CLAMP22_EMBED - a.e_star() - b.e_star() + CUT_BIAS).clamp(0, SLOT_MAX) as u64;
    let (mant, wa, oiz, oez) = rnernd_reference(sig, cut);
    let out_exp = if oiz || oez {
        0
    } else {
        let e = a.e_star() + b.e_star() - PROD_UNIT + wa as i64;
        assert!((1..=254).contains(&e), "M6: output exponent in range");
        e as u64
    };
    MulFill {
        sig_product: sig,
        cut_depth: cut,
        width_adjust: wa,
        out: Bf16Fields {
            exp: out_exp,
            mantissa: mant,
            exp_is_zero: oiz || oez,
        },
        out_is_zero: oiz,
    }
}

fn write_mul<F: RichField>(blk: &mut MulBlock<F>, fill: &MulFill) {
    blk.sig_product = F::from_canonical_u64(fill.sig_product);
    blk.cut_depth = F::from_canonical_u64(fill.cut_depth);
    blk.out_exp = F::from_canonical_u64(fill.out.exp);
    blk.out_mantissa = F::from_canonical_u64(fill.out.mantissa);
    blk.width_adjust = F::from_canonical_u64(fill.width_adjust);
    blk.out_is_zero = F::from_bool(fill.out_is_zero);
    blk.out_exp_is_zero = F::from_bool(fill.out.exp_is_zero);
}

/// Witness values of the all-nonnegative FMA gadget (the W1-W12 constraint pattern).
struct FmaFill {
    sig_product: u64,
    product_scale_ge_addend: bool,
    scale_gap_slack: u64,
    is_far_gap: bool,
    far_gap_slack: u64,
    exp_gap_capped: u64,
    exp_gap_pow2: u64,
    aligned_product: u64,
    aligned_addend: u64,
    is_wide: bool,
    compression_shift: u64,
    shift_pow2: u64,
    compression_quotient: u64,
    compression_quotient_lsb: u64,
    compression_remainder: u64,
    sticky: bool,
    rounding_significand_key: u64,
    key_scale: i64,
    cut_used: u64,
    width_adjust: u64,
    out: Bf16Fields,
    out_is_zero: bool,
}

/// `out = RNE_bf16(a * b + c)` on nonnegative operands (`a` normal), exactly as W1-W12 prove it.
fn fill_fma(a: Bf16Fields, b: Bf16Fields, c: Bf16Fields) -> FmaFill {
    assert!(!a.exp_is_zero, "the FMA's product-left operand (dr) is normal");
    // W1 — the exact product significand and the two units.
    let m_p = a.m() * b.m();
    let ep = a.e_star() + b.e_star() - PROD_UNIT;
    let m_c = c.m();
    let ec = c.e_star() - EXP_UNIT;
    // W2 — which unit is coarser.
    let ge = ep >= ec;
    let scale_gap_slack = if ge { ep - ec } else { ec - ep - 1 } as u64;
    let d = if ge { ep - ec } else { ec - ep } as u64;
    // W3/W4: cap the gap when the smaller operand affects only rounding.
    // The thresholds are 10 for an 8-bit addend and 18 for a 16-bit product;
    // all terms are positive, so no subtraction-borrow adjustment is needed.
    let thr = if ge { 10 } else { 18 };
    let is_far = d >= thr;
    let far_gap_slack = if is_far { d - thr } else { 0 };
    let capped = if is_far { thr - 1 } else { d };
    assert!(capped <= POW2D_MAX);
    let pow2 = 1u64 << capped;
    // W5/W6 — align to the finer unit and fold (pure addition: all operands nonnegative).
    let (al_p, al_c) = if ge { (m_p * pow2, m_c) } else { (m_p, m_c * pow2) };
    let v = al_p + al_c;
    assert!(v < 1 << 34);
    // W8/W9 — round-to-odd compression to 17 bits when the fold is wide.
    let is_wide = v >= 1 << 17;
    let (shift, shift_pow2, k, r) = if is_wide {
        let s = u64::from(bit_length(v)) - 17;
        (s, 1u64 << s, v >> s, v & ((1u64 << s) - 1))
    } else {
        (0, 1, 0, 0)
    };
    let sticky = r != 0;
    let k0 = k & 1;
    // W10 — the RNERND key and its unit.
    let rounding_significand_key = if is_wide { k + u64::from(sticky) * (1 - k0) } else { v };
    assert!(rounding_significand_key < 1 << 17);
    let key_scale =
        if ge { ec } else { ep } + if is_far { (d - capped) as i64 } else { 0 } + if is_wide { shift as i64 } else { 0 };
    // W11/W12 — the shared rounding back-end and the exponent binding.
    // clamp(-133 - KEY_SCALE, -7, 18) + 7, i.e. CLAMP22 at key FMA_CLAMP_KEY_CONST - KEY_SCALE.
    let cut = (FMA_CLAMP_KEY_CONST - CLAMP22_EMBED - key_scale + CUT_BIAS).clamp(0, SLOT_MAX) as u64;
    let (mant, wa, oiz, oez) = rnernd_reference(rounding_significand_key, cut);
    let out_exp = if oiz || oez {
        0
    } else {
        let e = key_scale + wa as i64;
        assert!((1..=254).contains(&e), "FMA output exponent in range");
        e as u64
    };
    FmaFill {
        sig_product: m_p,
        product_scale_ge_addend: ge,
        scale_gap_slack,
        is_far_gap: is_far,
        far_gap_slack,
        exp_gap_capped: capped,
        exp_gap_pow2: pow2,
        aligned_product: al_p,
        aligned_addend: al_c,
        is_wide,
        compression_shift: shift,
        shift_pow2,
        compression_quotient: k,
        compression_quotient_lsb: k0,
        compression_remainder: r,
        sticky,
        rounding_significand_key,
        key_scale,
        cut_used: cut,
        width_adjust: wa,
        out: Bf16Fields {
            exp: out_exp,
            mantissa: mant,
            exp_is_zero: oiz || oez,
        },
        out_is_zero: oiz,
    }
}

fn write_fma<F: RichField>(v: &mut ScaleColumnsView<F>, fill: &FmaFill) {
    let f = &mut v.noised_bound_fma;
    f.sig_product = F::from_canonical_u64(fill.sig_product);
    f.product_scale_ge_addend = F::from_bool(fill.product_scale_ge_addend);
    f.scale_gap_slack = F::from_canonical_u64(fill.scale_gap_slack);
    f.is_far_gap = F::from_bool(fill.is_far_gap);
    f.far_gap_slack = F::from_canonical_u64(fill.far_gap_slack);
    f.exp_gap_capped = F::from_canonical_u64(fill.exp_gap_capped);
    f.exp_gap_pow2 = F::from_canonical_u64(fill.exp_gap_pow2);
    f.aligned_product = F::from_canonical_u64(fill.aligned_product);
    f.aligned_addend = F::from_canonical_u64(fill.aligned_addend);
    f.is_wide = F::from_bool(fill.is_wide);
    f.compression_shift = F::from_canonical_u64(fill.compression_shift);
    f.shift_pow2 = F::from_canonical_u64(fill.shift_pow2);
    f.compression_quotient = F::from_canonical_u64(fill.compression_quotient);
    f.compression_quotient_lsb = F::from_canonical_u64(fill.compression_quotient_lsb);
    f.compression_remainder = F::from_canonical_u64(fill.compression_remainder);
    f.compression_remainder_inverse = if fill.compression_remainder == 0 {
        F::ZERO
    } else {
        F::from_canonical_u64(fill.compression_remainder).inverse()
    };
    f.sticky = F::from_bool(fill.sticky);
    f.rounding_significand_key = F::from_canonical_u64(fill.rounding_significand_key);
    f.rounding_significand_key_high_bit = F::from_canonical_u64(fill.rounding_significand_key >> 16);
    f.key_scale = fe_i64(fill.key_scale);
    f.cut_used = F::from_canonical_u64(fill.cut_used);
    f.out_exp = F::from_canonical_u64(fill.out.exp);
    f.out_mantissa = F::from_canonical_u64(fill.out.mantissa);
    f.width_adjust = F::from_canonical_u64(fill.width_adjust);
    f.out_is_zero = F::from_bool(fill.out_is_zero);
    f.out_exp_is_zero = F::from_bool(fill.out.exp_is_zero);
}

fn fe_i64<F: RichField>(x: i64) -> F {
    if x >= 0 {
        F::from_canonical_u64(x as u64)
    } else {
        -F::from_canonical_u64(x.unsigned_abs())
    }
}

// The exact sqrt claim (group Q trace side)

/// A sqrt claim's bf16 fields (sign-free).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SqrtClaim {
    pub exp: u64,
    pub mantissa: u64,
    pub exp_is_zero: bool,
}

impl SqrtClaim {
    fn from_tf(t: i64, f: i64) -> Self {
        if t >= 128 {
            Self {
                exp: f as u64,
                mantissa: (t - 128) as u64,
                exp_is_zero: false,
            }
        } else {
            assert_eq!(f, 1, "sub-128 significands only exist in the f = 1 binade");
            Self {
                exp: 0,
                mantissa: t as u64,
                exp_is_zero: true,
            }
        }
    }

    fn t(&self) -> i64 {
        if self.exp_is_zero {
            self.mantissa as i64
        } else {
            128 + self.mantissa as i64
        }
    }

    fn f(&self) -> i64 {
        self.exp as i64 + i64::from(self.exp_is_zero)
    }
}

/// Compares `s * 2^d` against `c` exactly (`s < 2^62`, `c <= 2^57` in every call site).
fn cmp_shifted(s: u64, d: i64, c: u128) -> Ordering {
    debug_assert!(c > 0);
    if s == 0 {
        return Ordering::Less;
    }
    if d >= 0 {
        if d >= 66 {
            return Ordering::Greater; // s >= 1, so s*2^66 > 2^57 >= c
        }
        (u128::from(s) << d).cmp(&c)
    } else {
        let g = -d;
        if g >= 71 {
            return Ordering::Less; // c >= 1, so c*2^71 > 2^62 > s
        }
        u128::from(s).cmp(&(c << g))
    }
}

/// The exact acceptance test of the claim `(t, f)` against `v_hat = s * 2^(e_max-14-wl2)/k`
/// (module docs): returns `(lower_ok, upper_ok)`.
pub(crate) fn sqrt_bracket_ok(s: u64, e_max: i64, wl2: i64, k: u64, t: i64, f: i64) -> (bool, bool) {
    let bottom = i64::from(t == 128 && f >= 2);
    let b_lo = 4 * t - 2 + bottom;
    let b_hi = 4 * t + 2;
    let odd = t & 1 == 1;
    let d = e_max + 4 - wl2 - 2 * f;
    let hi = cmp_shifted(s, d, (b_hi * b_hi) as u128 * u128::from(k));
    let hi_ok = hi == Ordering::Less || (hi == Ordering::Equal && !odd);
    if t == 0 {
        // One-sided zero claim: the lower arm is gated off (`LOWER_BRACKET_IS_ACTIVE = 0`) — RNE's
        // zero region is `sqrt(v_hat) <= 2^-7`, the upper boundary alone (non-strict: 0 even).
        return (true, hi_ok);
    }
    let lo = cmp_shifted(s, d, (b_lo * b_lo) as u128 * u128::from(k));
    (lo == Ordering::Greater || (lo == Ordering::Equal && !odd), hi_ok)
}

/// `RNE_bf16(sqrt(v_hat))` in the hat frame, `v_hat = s * 2^(e_max - 14 - wl2) / k`, `s > 0` —
/// an f64 seed refined by the exact integer bracket (the bracket, not the seed, is the
/// definition; the seed only locates it). A sum inside RNE's zero region
/// (`sqrt(v_hat) <= 2^-7`) returns the +0 claim `(t, f) = (0, 1)` — legal under the one-sided
/// zero-claim arm.
pub(crate) fn rne_sqrt_hat(s: u64, e_max: i64, wl2: i64, k: u64) -> SqrtClaim {
    assert!(s > 0);
    let log2y = ((s as f64).log2() - (k as f64).log2() + (e_max - 14 - wl2) as f64) / 2.0;
    let mut f = log2y.floor() as i64;
    let mut t: i64;
    if f < 1 {
        f = 1;
        t = (log2y.exp2() * 64.0).round() as i64; // value * 2^6 (the f = 1 grid unit)
        t = t.clamp(0, 128);
    } else {
        t = ((log2y - f as f64).exp2() * 128.0).round() as i64;
        if t >= 256 {
            t = 128;
            f += 1;
        }
        t = t.max(128);
    }
    assert!(f <= 254, "sqrt exceeds the bf16 range (unreachable for finite rows)");

    for _ in 0..1024 {
        let (lo_ok, hi_ok) = sqrt_bracket_ok(s, e_max, wl2, k, t, f);
        if lo_ok && hi_ok {
            return SqrtClaim::from_tf(t, f);
        }
        if !lo_ok {
            // Claim too big: step to the predecessor. (t = 0 never fails its lower arm — the
            // one-sided zero claim — so t stays nonnegative.)
            debug_assert!(t > 0);
            t -= 1;
            if t == 127 && f >= 2 {
                t = 255;
                f -= 1;
            }
        } else {
            // Claim too small: step to the successor.
            t += 1;
            if t == 256 {
                assert!(f < 254, "sqrt exceeds the bf16 range");
                t = 128;
                f += 1;
            }
        }
    }
    panic!("sqrt bracket search did not converge (s = {s}, e_max = {e_max}, wl2 = {wl2}, k = {k})");
}

/// Fills every group-Q column for the given claim (honest callers set `expect_valid` and get
/// hard asserts; tamper tests pass `false` and let the constraints/LUT checks flag the row).
pub(crate) fn fill_sqrt_block<F: RichField>(
    v: &mut ScaleColumnsView<F>,
    s: u64,
    frame_doubled_scale_exponent: u64,
    wl2: u32,
    k: u64,
    claim: SqrtClaim,
    expect_valid: bool,
) {
    let is_zero = s == 0;
    v.sqrt_exp = F::from_canonical_u64(claim.exp);
    v.sqrt_mantissa = F::from_canonical_u64(claim.mantissa);
    v.sqrt_exp_is_zero = F::from_bool(claim.exp_is_zero);
    v.frame_sum_is_zero = F::from_bool(is_zero);
    v.frame_sum_inverse = if is_zero {
        F::ZERO
    } else {
        F::from_canonical_u64(s).inverse()
    };
    v.sqrt_mantissa_parity = F::from_canonical_u64(claim.mantissa & 1);
    v.sqrt_mantissa_half = F::from_canonical_u64(claim.mantissa >> 1);

    // Q2b — the zero-claim flag and the committed lower-arm gate.
    let t = claim.t();
    let t_zero = t == 0;
    let lower_bracket_active = !is_zero && !t_zero;
    v.sqrt_claim_is_zero = F::from_bool(t_zero);
    v.lower_bracket_is_active = F::from_bool(lower_bracket_active);

    // Correct the lower midpoint at normal powers of two above the smallest normal value.
    let mant_is_zero = claim.mantissa == 0;
    let exp_ge2 = claim.exp >= 2;
    let bottom = mant_is_zero && !claim.exp_is_zero && exp_ge2;
    v.sqrt_is_binade_bottom = F::from_bool(bottom);
    v.sqrt_mantissa_is_zero = F::from_bool(mant_is_zero);
    v.sqrt_mantissa_inverse = if mant_is_zero {
        F::ZERO
    } else {
        F::from_canonical_u64(claim.mantissa).inverse()
    };
    v.sqrt_exponent_at_least_two = F::from_bool(exp_ge2);

    // Q1 — the frame-sum limbs.
    for i in 0..L2_SUM_LIMBS {
        v.frame_sum_limbs[i] = F::from_canonical_u64((s >> (16 * i)) & 0xFFFF);
    }

    // Q7 — midpoint squares and the shifted claim-side products.
    let b_lo = 4 * t - 2 + i64::from(bottom);
    let msq = (b_lo * b_lo) as u64;
    v.lower_boundary_squared_limbs[0] = F::from_canonical_u64(msq & 0xFFFF);
    v.lower_boundary_squared_limbs[1] = F::from_canonical_u64(msq >> 16);
    let upper_sq = msq as i64 + 32 * t - UPPER_SQ_BOTTOM_DELTA * i64::from(bottom);
    assert!(upper_sq >= 0);
    let lower_boundary_product = (u128::from(msq) * u128::from(k)) << 15;
    let upper_boundary_product = (upper_sq as u128 * u128::from(k)) << 15;
    assert_eq!(lower_boundary_product & 0xFFFF, 0, "32 | k makes claim-side limb 0 vanish");
    let mut lower_boundary_product_limbs = [0u64; CLAIM_LIMBS];
    let mut upper_boundary_product_limbs = [0u64; CLAIM_LIMBS];
    for i in 0..CLAIM_LIMBS {
        lower_boundary_product_limbs[i] = ((lower_boundary_product >> (16 * (i + 1))) & 0xFFFF) as u64;
        upper_boundary_product_limbs[i] = ((upper_boundary_product >> (16 * (i + 1))) & 0xFFFF) as u64;
        v.lower_boundary_product_limbs[i] = F::from_canonical_u64(lower_boundary_product_limbs[i]);
        v.upper_boundary_product_limbs[i] = F::from_canonical_u64(upper_boundary_product_limbs[i]);
    }
    if expect_valid {
        assert!(lower_boundary_product < 1 << 53 && upper_boundary_product < 1 << 53);
    }

    // Choose the whole-limb and residual bit shifts derived in Q5 of eval_scale_constraints.
    let e_max = frame_doubled_scale_exponent as i64;
    let g = 2 * claim.f() + i64::from(wl2) - 4 - e_max;
    if expect_valid {
        assert!((-32..=47).contains(&g), "honest shift window (module docs)");
    }
    let gamma = (g + 32).clamp(0, 79);
    let q = gamma / 16; // in {0..4}
    let r2 = 16 * q + 15 - gamma; // in [0, 15]
    v.alignment_quotient_is_1 = F::from_bool(q == 1);
    v.alignment_quotient_is_2 = F::from_bool(q == 2);
    v.alignment_quotient_is_3 = F::from_bool(q == 3);
    v.alignment_quotient_is_4 = F::from_bool(q == 4);
    v.sum_shift_remainder = fe_i64(r2);
    let pow2 = 1u64 << (r2 as u32);
    v.sum_shift_power = F::from_canonical_u64(pow2);

    // Q6 — the shifted sum SS = S * 2^r2, limb-wise with carries.
    let mut shifted_sum_limbs = [0u64; L2_SUM_LIMBS];
    let mut shifted_sum_carries = [0u64; L2_SUM_LIMBS];
    let mut carry: u128 = 0;
    for i in 0..L2_SUM_LIMBS {
        let acc = u128::from((s >> (16 * i)) & 0xFFFF) * u128::from(pow2) + carry;
        shifted_sum_limbs[i] = (acc & 0xFFFF) as u64;
        carry = acc >> 16;
        shifted_sum_carries[i] = carry as u64;
        v.shifted_sum_limbs[i] = F::from_canonical_u64(shifted_sum_limbs[i]);
        v.shifted_sum_carries[i] = F::from_canonical_u64(shifted_sum_carries[i]);
    }

    // Q8 — build the shifted lower/upper bounds and subtraction witnesses, as derived
    // in eval_scale_constraints below. A zero-claim flag replaces the lower bound with 0;
    // a zero frame sum replaces both bounds with 0.
    let lower_bound_active = u64::from(lower_bracket_active);
    let upper_bound_active = u64::from(!is_zero);
    let mut aligned_lower_bound_limbs = [0u64; 8];
    let mut aligned_upper_bound_limbs = [0u64; 8];
    for destination_limb in 1..=ALIGNED_UPPER_LIMBS {
        let source_limb = destination_limb as i64 - q;
        if (1..=CLAIM_LIMBS as i64).contains(&source_limb) {
            aligned_lower_bound_limbs[destination_limb] =
                lower_bound_active * lower_boundary_product_limbs[(source_limb - 1) as usize];
            aligned_upper_bound_limbs[destination_limb] =
                upper_bound_active * upper_boundary_product_limbs[(source_limb - 1) as usize];
        }
        if destination_limb <= ALIGNED_LOWER_LIMBS {
            v.aligned_lower_boundary_limbs[destination_limb - 1] =
                F::from_canonical_u64(aligned_lower_bound_limbs[destination_limb]);
        }
        v.aligned_upper_boundary_limbs[destination_limb - 1] = F::from_canonical_u64(aligned_upper_bound_limbs[destination_limb]);
    }
    let scaled_sum_limbs = [
        0,
        0,
        shifted_sum_limbs[0],
        shifted_sum_limbs[1],
        shifted_sum_limbs[2],
        shifted_sum_limbs[3],
        shifted_sum_carries[3],
        0,
    ];
    let mantissa_parity = claim.mantissa & 1;
    let (lower_borrows, lower_bound_holds) = borrow_bits(&aligned_lower_bound_limbs, &scaled_sum_limbs, mantissa_parity);
    let (upper_borrows, upper_bound_holds) = borrow_bits(&scaled_sum_limbs, &aligned_upper_bound_limbs, mantissa_parity);
    if expect_valid && !is_zero {
        // A zero-claim flag replaces the lower bound with zero, so its comparison always holds.
        assert!(
            lower_bound_holds && upper_bound_holds,
            "honest claim fails its own bracket (s = {s})"
        );
    }
    for (column, borrow) in v.lower_comparison_borrows.iter_mut().zip(lower_borrows) {
        *column = F::from_bool(borrow);
    }
    for (column, borrow) in v.upper_comparison_borrows.iter_mut().zip(upper_borrows) {
        *column = F::from_bool(borrow);
    }
}

/// Computes borrows for `minuend - subtrahend - initial_borrow` using eight 16-bit limbs.
/// `initial_borrow` is the mantissa parity, subtracted from the lowest digit for ties-to-even.
/// Returns the seven borrow bits and whether the final subtraction is nonnegative.
fn borrow_bits(subtrahend: &[u64; 8], minuend: &[u64; 8], initial_borrow: u64) -> ([bool; UPPER_BORROW_BITS], bool) {
    let mut borrow_flags = [false; UPPER_BORROW_BITS];
    let mut borrow: i64 = 0;
    for limb_position in 0..8 {
        let digit_difference = minuend[limb_position] as i64
            - subtrahend[limb_position] as i64
            - borrow
            - if limb_position == 0 { initial_borrow as i64 } else { 0 };
        borrow = i64::from(digit_difference < 0);
        if limb_position < UPPER_BORROW_BITS {
            borrow_flags[limb_position] = digit_difference < 0;
        }
    }
    (borrow_flags, borrow == 0)
}

// Constraints, written once against the generic `Evaluator`

use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// `M = 128*(1 - EXP_IS_ZERO) + MANTISSA` as an affine expression.
fn m_field<V: Copy, S: Copy, E: Evaluator<V, S>>(eval: &mut E, exp_is_zero: V, mantissa: V) -> V {
    let c128 = eval.i32(128);
    let scaled = eval.mul(c128, exp_is_zero);
    let base = eval.sub(c128, scaled);
    eval.add(base, mantissa)
}

/// `E* = EXP + EXP_IS_ZERO` as an affine expression.
fn e_star_field<V: Copy, S: Copy, E: Evaluator<V, S>>(eval: &mut E, exp: V, exp_is_zero: V) -> V {
    eval.add(exp, exp_is_zero)
}

/// The shared MUL constraints M1/M4/M5 (M2/M3/M6 are LUT instances; see `ctl::scale_lut_lookups`).
/// `e_star_sum` must be the affine `E*(a) + E*(b)`.
fn mul_block_constraints<V: Copy + Default, S: Copy, E: Evaluator<V, S>>(
    eval: &mut E,
    blk: &MulBlock<V>,
    sig_expected: V,
    e_star_sum: V,
) {
    let one = eval.i32(1);
    // M1 — the exact significand product.
    eval.constraint_eq(blk.sig_product, sig_expected);
    // M4 — output exponent on the normal path: OUT_EXP = E*a + E*b - 268 + WIDTH_ADJUST.
    let unit = eval.i32(PROD_UNIT as i32);
    let rebased = eval.sub(e_star_sum, unit);
    let expected_exp = eval.add(rebased, blk.width_adjust);
    let inner = eval.sub(blk.out_exp, expected_exp);
    let not_oiz = eval.sub(one, blk.out_is_zero);
    let not_oez = eval.sub(one, blk.out_exp_is_zero);
    let gated = eval.mul(not_oiz, not_oez);
    let c = eval.mul(gated, inner);
    eval.constraint(c);
    // M5 — zero/subnormal bindings.
    let c = eval.mul(blk.out_exp_is_zero, blk.out_exp);
    eval.constraint(c);
    let c = eval.mul(blk.out_is_zero, blk.out_exp);
    eval.constraint(c);
    let c = eval.mul(blk.out_is_zero, blk.out_mantissa);
    eval.constraint(c);
    // No booleanness pins: the flags are RNERND value components, boolean by the table's
    // construction.
}

/// Evaluates every arithmetic constraint of ScaleStark. Lookup-borne facts (LUT oracle) are
/// *not* emitted here — see the module docs and `ctl::scale_lut_lookups`. The constraint set
/// is program-independent: the geometry parameters `k` / `Wl2` enter as public inputs.
pub(crate) fn eval_scale_constraints<V, S, E>(vars: &StarkFrame<V, S, NUM_SCALE_COLUMNS, NUM_SCALE_PUBLIC_INPUTS>, eval: &mut E)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_SCALE_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &ScaleColumnsView<V> = lv.borrow();
    // Only the group-T running totals read the next row; every other group is local to one
    // row (one matrix row's aggregates per trace row).
    let nv: &[V; NUM_SCALE_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let nv: &ScaleColumnsView<V> = nv.borrow();
    let pis = vars.get_public_inputs();
    let dr_exp = eval.scalar(pis[DR_EXP_PUBLIC_INPUT]);
    let dr_mantissa = eval.scalar(pis[DR_MANTISSA_PUBLIC_INPUT]);
    let dos_exp = eval.scalar(pis[DOS_EXP_PUBLIC_INPUT]);
    let dos_mantissa = eval.scalar(pis[DOS_MANTISSA_PUBLIC_INPUT]);
    // The geometry parameters, as public inputs (degree 0 in the constraint polynomials —
    // swapping the former baked constants for these scalars changes no constraint degree).
    // The verifier pins both to statement-derived values inside the envelope
    // (`k % 32 == 0`, `1024 <= k <= 2^16`, hence `Wl2 in [11, 17]`) — the range every
    // soundness bound below assumes.
    let k_pi = eval.scalar(pis[K_PUBLIC_INPUT]);
    let wl2_pi = eval.scalar(pis[WL2_PUBLIC_INPUT]);

    let one = eval.i32(1);
    let limb = eval.u64(1 << 16);

    // The verifier binds the group keys, row flags and padding schedule; see `known_values`.

    // Q — the sqrt bracket (frame derivation in the module docs).

    // Q1 — S recomposition from its range-checked limbs (top limb capped < 2^14 by a lookup, so both
    // sides are < 2^62 < p and the equality is over Z).
    let mut s_recomposed = lv.frame_sum_limbs[0];
    for i in 1..L2_SUM_LIMBS {
        let w = eval.u64(1 << (16 * i));
        s_recomposed = eval.mad(lv.frame_sum_limbs[i], w, s_recomposed);
    }
    eval.constraint_eq(lv.l2_frame_sum, s_recomposed);

    // Q2 — the two-sided zero-sum pair; a zero sum forces the +0 claim.
    let z = lv.frame_sum_is_zero;
    let frame_sum_is_nonzero = eval.sub(one, z);
    let prod = eval.mul(lv.l2_frame_sum, lv.frame_sum_inverse);
    eval.constraint_eq(prod, frame_sum_is_nonzero);
    let c = eval.mul(z, lv.l2_frame_sum);
    eval.constraint(c);
    let c = eval.mul(z, lv.sqrt_exp);
    eval.constraint(c);
    let c = eval.mul(z, lv.sqrt_mantissa);
    eval.constraint(c);
    let not_ez = eval.sub(one, lv.sqrt_exp_is_zero);
    let c = eval.mul(z, not_ez);
    eval.constraint(c);

    // Q2b — sqrt_claim_is_zero = 1 forces a +0 claim and sets the lower comparison
    // bound to 0.
    // The converse is unnecessary: with a positive frame sum and a zero claim,
    // leaving the flag at 0 makes both squared bounds equal. The unrounded square
    // root must then be halfway between 0 and the smallest positive bfloat16 value,
    // which correctly rounds to 0. For a zero frame sum, Q2 already forces +0.
    eval.constraint_bool(lv.sqrt_claim_is_zero);
    let c = eval.mul(lv.sqrt_claim_is_zero, lv.sqrt_mantissa);
    eval.constraint(c);
    let not_ez_claim = eval.sub(one, lv.sqrt_exp_is_zero);
    let c = eval.mul(lv.sqrt_claim_is_zero, not_ez_claim);
    eval.constraint(c);
    let not_t_zero = eval.sub(one, lv.sqrt_claim_is_zero);
    let lower_bracket_active = eval.mul(frame_sum_is_nonzero, not_t_zero);
    eval.constraint_eq(lv.lower_bracket_is_active, lower_bracket_active);

    // Q3 — claim decode hygiene: EXPINFO(SQRT_EXP; SQRT_EXP_IS_ZERO) and the PAIR128 mantissa
    // check are LUT instances (the flag's booleanness is table-supplied).

    // Q4 — mantissa parity split. Q3 range-checks both MANTISSA and HALF to seven bits,
    // so MANTISSA = 2*HALF + PARITY holds over integers and fixes PARITY to MANTISSA mod 2.
    let two = eval.i32(2);
    let half2 = eval.mul(two, lv.sqrt_mantissa_half);
    let split = eval.add(half2, lv.sqrt_mantissa_parity);
    eval.constraint_eq(lv.sqrt_mantissa, split);
    eval.constraint_bool(lv.sqrt_mantissa_parity);

    // Q-bottom — correct the lower midpoint at normal powers of two above 2^-126.
    // For a positive claim q = t*Delta, let Delta be the spacing to the next larger value.
    // At these powers of two the predecessor is only Delta/2 below q, so the lower midpoint
    // is q - Delta/4 instead of q - Delta/2. Q7 implements this as B_LO = 4*t - 2 + IS_BOTTOM.
    // The constraints force IS_BOTTOM = 1 when MANTISSA = 0 and EXP >= 2:
    // the inverse pair detects MANTISSA = 0 exactly, and EXP >= 2 forces EXP_GE2 = 1.
    // Setting IS_BOTTOM also forces MANTISSA = 0 and EXP != 0, hence t = 128 for Q7's upper-bound formula.
    // At the smallest normal (EXP = 1, MANTISSA = 0), either flag value is allowed;
    // setting it only tightens the lower bound. The honest witness uses 0 there.
    eval.constraint_bool(lv.sqrt_is_binade_bottom);
    let c = eval.mul(lv.sqrt_is_binade_bottom, lv.sqrt_mantissa);
    eval.constraint(c);
    let c = eval.mul(lv.sqrt_is_binade_bottom, lv.sqrt_exp_is_zero);
    eval.constraint(c);
    let prod = eval.mul(lv.sqrt_mantissa, lv.sqrt_mantissa_inverse);
    let not_mz = eval.sub(one, lv.sqrt_mantissa_is_zero);
    eval.constraint_eq(prod, not_mz);
    let c = eval.mul(lv.sqrt_mantissa_is_zero, lv.sqrt_mantissa);
    eval.constraint(c);
    eval.constraint_bool(lv.sqrt_exponent_at_least_two);
    let not_ge2 = eval.sub(one, lv.sqrt_exponent_at_least_two);
    let exp_m1 = eval.sub(lv.sqrt_exp, one);
    let quad = eval.mul(lv.sqrt_exp, exp_m1);
    let c = eval.mul(not_ge2, quad);
    eval.constraint(c);
    let not_bottom = eval.sub(one, lv.sqrt_is_binade_bottom);
    let mz_ge2 = eval.mul(lv.sqrt_mantissa_is_zero, lv.sqrt_exponent_at_least_two);
    let c = eval.mul(not_bottom, mz_ge2);
    eval.constraint(c);

    // Q5 — prepare the checks a^2 <= v <= c^2 that prove correct square-root rounding:
    // the mean square v must lie between the squares of the claim's lower and upper
    // rounding boundaries a and c. Both inequalities are strict when the claim's mantissa
    // is odd (ties-to-even); a zero claim uses lower bound 0.
    // Q5 puts these comparisons on a common integer scale; Q8 checks the inequalities.
    //
    // Notation:
    //   S = l2_frame_sum, the integer sum received from InputQuant;
    //   k = the number of elements in the matrix row (public input k_pi);
    //   F = frame_doubled_scale_exponent, the exponent defining the sum's scale;
    //   W = Wl2, the sum's precision shift (public input wl2_pi);
    //   f = sqrt_exp + sqrt_exp_is_zero, the claim's effective exponent (1 if subnormal/zero).
    // The represented mean square is v = (S/k)*2^(F - 268 - W).
    // Here 268 = 2*134, and 134 = bfloat16's exponent bias 127 + its 7 fraction bits.
    //
    // The claim's spacing unit is 2^(f-134). Rounding boundaries use quarter units,
    // so each active boundary is B*2^(f-136), with 136 = 134 + 2. Its integer coefficient B
    // is 4*t-2+b for the lower boundary or 4*t+2 for the upper boundary (Q7), where
    // t = 128*(1-sqrt_exp_is_zero) + sqrt_mantissa and b = sqrt_is_binade_bottom.
    // Comparing v with the boundary's square compares (S/k)*2^(F-268-W) with B^2*2^(2*f-272).
    // Multiply both sides by k*2^(-F+268+W) to obtain
    //   S  versus  k*B^2*2^G,  where G = 2*f + W - 4 - F and 4 = 272 - 268.
    // This applies to each active boundary; Q8 handles the zero cases and strict endpoints.
    //
    // Large integers are stored in 16-bit limbs, so a shift by 16*q bits moves
    // the limbs by q positions. Add 32 = 2*16 to G to allow negative shifts down to -32.
    // Write G+32 = 16*q+s with remainder s in 0..15. Set r2 = 15-s so that
    //   G+32+r2 = 16*q+15.
    // Multiplying the comparison by 2^(32+r2) now gives only nonnegative integer shifts:
    //   S*2^(32+r2)  versus  (k*B^2*2^15)*2^(16*q).
    // Thus Q6 shifts S by r2 bits, Q7 forms k*B^2*2^15, and Q8 places these results
    // two limbs and q limbs higher, respectively.
    //
    // The boundary product occupies limb positions 1..3 (position 0 is zero), so q in
    // 0..4 fits within Q8's eight positions 0..7. The four flags below select q=1,2,3,4;
    // all zero selects q=0. Their boolean sum prevents selecting more than one position.
    // The lookups constrain r2 = sum_shift_remainder to 0..15 and sum_shift_power to 2^r2.
    // Hence G+32 ranges from 0 to 4*16+15 = 79, supporting exactly -32 <= G <= 47.
    // Q5 enforces this alignment; Q8 enforces the actual rounding inequalities.
    eval.constraint_bool(lv.alignment_quotient_is_1);
    eval.constraint_bool(lv.alignment_quotient_is_2);
    eval.constraint_bool(lv.alignment_quotient_is_3);
    eval.constraint_bool(lv.alignment_quotient_is_4);
    let q_flag_sum = {
        let s12 = eval.add(lv.alignment_quotient_is_1, lv.alignment_quotient_is_2);
        let s34 = eval.add(lv.alignment_quotient_is_3, lv.alignment_quotient_is_4);
        eval.add(s12, s34)
    };
    eval.constraint_bool(q_flag_sum);
    let f_claim = e_star_field(eval, lv.sqrt_exp, lv.sqrt_exp_is_zero);
    let two_f = eval.mul(two, f_claim);
    let g32_val = eval.sub(two_f, lv.frame_doubled_scale_exponent);
    // G + 32 = 2*f - F + W + 28, where 28 = 32 - 4.
    let c28 = eval.i32(28);
    let wl2_c = eval.add(wl2_pi, c28);
    let g32_val = eval.add(g32_val, wl2_c);
    let sixteen = eval.i32(16);
    let q_val = {
        let three = eval.i32(3);
        let four_c = eval.i32(4);
        let q12 = eval.mad(two, lv.alignment_quotient_is_2, lv.alignment_quotient_is_1);
        let q3 = eval.mul(three, lv.alignment_quotient_is_3);
        let q4 = eval.mul(four_c, lv.alignment_quotient_is_4);
        let q123 = eval.add(q12, q3);
        eval.add(q123, q4)
    };
    let sixteen_q = eval.mul(sixteen, q_val);
    let fifteen = eval.i32(15);
    let rhs = eval.add(sixteen_q, fifteen);
    let rhs = eval.sub(rhs, lv.sum_shift_remainder);
    eval.constraint_eq(g32_val, rhs);

    // Q6 — SS = S * 2^r2, limb-wise schoolbook with range-checked carries (each equation < 2^32 on
    // both sides, so exact over Z; the top limb of SS is SHIFTED_SUM_CARRIES_3).
    let mut carry_prev: Option<V> = None;
    for i in 0..L2_SUM_LIMBS {
        let prod = eval.mul(lv.frame_sum_limbs[i], lv.sum_shift_power);
        let lhs = match carry_prev {
            None => prod,
            Some(cp) => eval.add(prod, cp),
        };
        let rhs = eval.mad(lv.shifted_sum_carries[i], limb, lv.shifted_sum_limbs[i]);
        eval.constraint_eq(lhs, rhs);
        carry_prev = Some(lv.shifted_sum_carries[i]);
    }

    // Q7 — midpoint squares. B_LO = 4t - 2 + IS_BOTTOM with t = 128*(1 - EZ) + MANTISSA;
    // the recomposition is < 2^32 on both sides, so the limbs are exact and limb 1 < 2^4 is
    // forced. The claim-side products commit limbs 1..3 of B^2 * k * 2^15 (limb 0 is provably
    // zero; both sides < 2^57 with the RC16 caps, so exact over Z).
    let t_sig = m_field(eval, lv.sqrt_exp_is_zero, lv.sqrt_mantissa);
    let four = eval.i32(4);
    let four_t = eval.mul(four, t_sig);
    let b_lo = eval.sub(four_t, two);
    let b_lo = eval.add(b_lo, lv.sqrt_is_binade_bottom);
    let b_lo_sq = eval.mul(b_lo, b_lo);
    let msq = eval.mad(lv.lower_boundary_squared_limbs[1], limb, lv.lower_boundary_squared_limbs[0]);
    eval.constraint_eq(msq, b_lo_sq);

    // The lowest 16-bit limb is omitted because k is a multiple of 32,
    // making L = k*B_LO^2*2^15 divisible by 2^20.
    // Q8 shifts L by 16*q bits to compare it with S*2^(32+r2), as derived in Q5.
    let c2_15 = eval.u64(1 << 15);
    let kp = eval.mul(k_pi, c2_15);
    let lk_expected = eval.mul(msq, kp);
    let mut lk_recomposed = eval.i32(0);
    for i in 0..CLAIM_LIMBS {
        let w = eval.u64(1 << (16 * (i + 1)));
        lk_recomposed = eval.mad(lv.lower_boundary_product_limbs[i], w, lk_recomposed);
    }
    eval.constraint_eq(lk_recomposed, lk_expected);
    // Reuse the lower square to obtain the upper square. With b = sqrt_is_binade_bottom,
    // B_HI = 4*t+2 and B_LO = 4*t-2+b, so
    //   B_HI^2 - B_LO^2 = 32*t - 8*t*b + 4*b - b^2.
    // Q-bottom makes b boolean (b^2 = b) and forces t = 128 when b = 1 (t*b = 128*b).
    // Thus the difference is 32*t - 1024*b + 3*b = 32*t - 1021*b.
    let thirty_two = eval.i32(32);
    let delta = eval.mul(thirty_two, t_sig);
    let bottom_delta = eval.i32(UPPER_SQ_BOTTOM_DELTA as i32);
    let bottom_term = eval.mul(bottom_delta, lv.sqrt_is_binade_bottom);
    let upper_sq = eval.add(msq, delta);
    let upper_sq = eval.sub(upper_sq, bottom_term);
    let uk_expected = eval.mul(upper_sq, kp);
    let mut uk_recomposed = eval.i32(0);
    for i in 0..CLAIM_LIMBS {
        let w = eval.u64(1 << (16 * (i + 1)));
        uk_recomposed = eval.mad(lv.upper_boundary_product_limbs[i], w, uk_recomposed);
    }
    eval.constraint_eq(uk_recomposed, uk_expected);

    // Q8 — construct the aligned bounds for the integer comparison derived in Q5:
    //   lower_bound + sqrt_mantissa_parity <= scaled_sum <= upper_bound - sqrt_mantissa_parity,
    // where scaled_sum = l2_frame_sum * 2^(32 + sum_shift_remainder).
    // The selected limb offset multiplies Q7's boundary products by 2^(16*offset).
    // The subtraction lookups in ctl.rs then prove both inequalities, including ties-to-even.
    let unshifted_boundary_selector = eval.sub(one, q_flag_sum);
    let boundary_shift_selectors = [
        (0usize, unshifted_boundary_selector),
        (1, lv.alignment_quotient_is_1),
        (2, lv.alignment_quotient_is_2),
        (3, lv.alignment_quotient_is_3),
        (4, lv.alignment_quotient_is_4),
    ];
    // source_limb = destination_limb - limb_offset. Source positions 1..3 are stored
    // at array indices 0..2; every other source digit is zero, including position 0.
    for destination_limb in 1..=ALIGNED_UPPER_LIMBS {
        let mut expected_lower_limb = eval.i32(0);
        for (limb_offset, shift_selected) in boundary_shift_selectors {
            let source_limb = destination_limb as i64 - limb_offset as i64;
            if (1..=CLAIM_LIMBS as i64).contains(&source_limb) {
                let selected_lower_limb = eval.mul(shift_selected, lv.lower_boundary_product_limbs[(source_limb - 1) as usize]);
                expected_lower_limb = eval.add(expected_lower_limb, selected_lower_limb);
            }
        }
        // Q2/Q2b replace the lower bound with zero when the sum is zero or the zero-claim
        // flag is set. Parity is then zero, so the lower comparison is 0 <= scaled_sum.
        let gated_lower_limb = eval.mul(lv.lower_bracket_is_active, expected_lower_limb);
        if destination_limb <= ALIGNED_LOWER_LIMBS {
            eval.constraint_eq(lv.aligned_lower_boundary_limbs[destination_limb - 1], gated_lower_limb);
        } else {
            // scaled_sum < 2^109 has no digit at position 7 (weight 2^112).
            // Keep the omitted lower digit's alignment equation: 0 = computed digit.
            eval.constraint(gated_lower_limb);
        }
        let mut expected_upper_limb = eval.i32(0);
        for (limb_offset, shift_selected) in boundary_shift_selectors {
            let source_limb = destination_limb as i64 - limb_offset as i64;
            if (1..=CLAIM_LIMBS as i64).contains(&source_limb) {
                let selected_upper_limb = eval.mul(shift_selected, lv.upper_boundary_product_limbs[(source_limb - 1) as usize]);
                expected_upper_limb = eval.add(expected_upper_limb, selected_upper_limb);
            }
        }
        // Replace the upper bound with zero only for a zero sum (Q2 also forces zero parity).
        // A positive sum keeps its upper threshold even when the claimed square root is zero.
        let gated_upper_limb = eval.mul(frame_sum_is_nonzero, expected_upper_limb);
        eval.constraint_eq(lv.aligned_upper_boundary_limbs[destination_limb - 1], gated_upper_limb);
    }
    // ctl.rs range-checks the subtraction digits; these Boolean constraints complete its
    // borrow equations for scaled_sum - lower_bound - parity and upper_bound - scaled_sum - parity.
    for borrow in lv.lower_comparison_borrows {
        eval.constraint_bool(borrow);
    }
    for borrow in lv.upper_comparison_borrows {
        eval.constraint_bool(borrow);
    }

    // G — grid snap (nearest multiple of four code ulps, ties up):
    // SQRT_CODE + 2 = 4*GRID_SNAP_QUOTIENT + GRID_SNAP_REMAINDER,
    // with GRID_SNAP_REMAINDER in [0, 4). The snapped code is 4*GRID_SNAP_QUOTIENT; EXPINFO
    // rejects a result in exponent field 255.
    let c128 = eval.i32(128);
    let sqrt_code = eval.mad(lv.sqrt_exp, c128, lv.sqrt_mantissa);
    let code_plus2 = eval.add(sqrt_code, two);
    let four_q = eval.mul(four, lv.grid_snap_quotient);
    let snapped = eval.add(four_q, lv.grid_snap_remainder);
    eval.constraint_eq(code_plus2, snapped);
    let l2_code = eval.mad(lv.l2_exp, c128, lv.l2_mantissa);
    eval.constraint_eq(four_q, l2_code);
    // (L2_EXP_IS_ZERO is an EXPINFO value component, boolean by the table's construction.)

    // ==== N — the group lookup binds this encoding directly to InputQuant's maximum. ====
    let linf_code = eval.mad(lv.linf_exp, c128, lv.linf_mantissa);
    // (LINF_EXP_IS_ZERO is an EXPINFO value component, boolean by the table's construction.)

    // ==== H0 — the scheme's norm floors: l2f = max(l2, 2^-32) and linf_f = max(linf, 2^-32).
    // For each norm, ge is Boolean: 1 when norm >= 2^-32, otherwise 0.
    // A range-checked difference proves that the selected branch is correct.
    // With m_raw/e_raw the norm's integer significand/effective exponent, the output columns satisfy:
    //   m_floored = ge*m_raw + (1-ge)*128
    //   e_floored = ge*e_raw + (1-ge)*95
    // The floor uses (128, 95) because 2^-32 = 128*2^(95-134).
    // Storing these results avoids expanding these products inside later multiplication constraints. ====
    let floor_code = eval.u64(u64::from(NORM_FLOOR_CODE));
    let floor_e_star = eval.u64(NORM_FLOOR_E_STAR);
    let m_l2 = m_field(eval, lv.l2_exp_is_zero, lv.l2_mantissa);
    let m_linf = m_field(eval, lv.linf_exp_is_zero, lv.linf_mantissa);
    let e_l2 = e_star_field(eval, lv.l2_exp, lv.l2_exp_is_zero);
    let e_linf = e_star_field(eval, lv.linf_exp, lv.linf_exp_is_zero);
    for (code, m_raw, e_raw, ge, slack, m_floored, e_floored) in [
        (
            l2_code,
            m_l2,
            e_l2,
            lv.l2_ge_floor,
            lv.l2_floor_order_slack,
            lv.l2_floored_significand,
            lv.l2_floored_exponent,
        ),
        (
            linf_code,
            m_linf,
            e_linf,
            lv.linf_ge_floor,
            lv.linf_floor_order_slack,
            lv.linf_floored_significand,
            lv.linf_floored_exponent,
        ),
    ] {
        eval.constraint_bool(ge);
        let not_ge = eval.sub(one, ge);
        // Two-sided order proof: SLACK = GE*(code - floor) + (1 - GE)*(floor - code - 1),
        // range-checked (both codes < 2^15, so the field equation is the integer one).
        let code_minus_floor = eval.sub(code, floor_code);
        let ge_side = eval.mul(ge, code_minus_floor);
        let floor_minus_code = eval.sub(floor_code, code);
        let floor_minus_code_m1 = eval.sub(floor_minus_code, one);
        let lt_side = eval.mul(not_ge, floor_minus_code_m1);
        let expected = eval.add(ge_side, lt_side);
        eval.constraint_eq(slack, expected);
        // The floored fields: the raw (M, E*) at or above the floor, the floor's (128, 95)
        // below — exactly the fields of max(code, 0x2F80) (the floor is normal, mantissa 0).
        let m_sel_ge = eval.mul(ge, m_raw);
        let m_sel_lt = eval.mul(not_ge, c128);
        let m_sel = eval.add(m_sel_ge, m_sel_lt);
        eval.constraint_eq(m_floored, m_sel);
        let e_sel_ge = eval.mul(ge, e_raw);
        let e_sel_lt = eval.mul(not_ge, floor_e_star);
        let e_sel = eval.add(e_sel_ge, e_sel_lt);
        eval.constraint_eq(e_floored, e_sel);
    }

    // H1 — the all-nonnegative FMA `noised_bound = RNE(dr * l2f + linf_f)` (W1-W12).
    let noised_bound_fma: &FmaBlock<V> = &lv.noised_bound_fma;
    // W1: SIG_PRODUCT = (128 + DR_MANTISSA) * M(l2f); EP = DR_EXP + E*(l2f) - 268;
    // EC = E*(linf_f) - 134 (dr is normal, public).
    let m_dr = eval.add(c128, dr_mantissa);
    let sig_expected = eval.mul(m_dr, lv.l2_floored_significand);
    eval.constraint_eq(noised_bound_fma.sig_product, sig_expected);
    let prod_unit = eval.i32(PROD_UNIT as i32);
    let ep = eval.add(dr_exp, lv.l2_floored_exponent);
    let ep = eval.sub(ep, prod_unit);
    let exp_unit = eval.i32(EXP_UNIT as i32);
    let ec = eval.sub(lv.linf_floored_exponent, exp_unit);
    // Now, prepare to align dr * l2f and linf_f to the same exponent (the finer of the two) for the FMA.
    // W2: GE boolean; SCALE_GAP_SLACK = GE*(EP - EC) + (1 - GE)*(EC - EP - 1) (RC16'd).
    eval.constraint_bool(noised_bound_fma.product_scale_ge_addend);
    let not_ge = eval.sub(one, noised_bound_fma.product_scale_ge_addend);
    let ep_minus_ec = eval.sub(ep, ec);
    let ge_side = eval.mul(noised_bound_fma.product_scale_ge_addend, ep_minus_ec);
    let ec_minus_ep = eval.sub(ec, ep);
    let ec_minus_ep_m1 = eval.sub(ec_minus_ep, one);
    let lt_side = eval.mul(not_ge, ec_minus_ep_m1);
    let expected = eval.add(ge_side, lt_side);
    eval.constraint_eq(noised_bound_fma.scale_gap_slack, expected);
    // The gap magnitude d = SLACK + 1 - GE (affine).
    let d_gap = eval.add(noised_bound_fma.scale_gap_slack, one);
    let d_gap = eval.sub(d_gap, noised_bound_fma.product_scale_ge_addend);
    // W3: far flag; on far rows d - THR = FAR_GAP_SLACK (range-checked), THR = 18 - 8*GE.
    eval.constraint_bool(noised_bound_fma.is_far_gap);
    let eight = eval.i32(8);
    let thr = {
        let e = eval.i32(18);
        let ge8 = eval.mul(eight, noised_bound_fma.product_scale_ge_addend);
        eval.sub(e, ge8)
    };
    let d_minus_thr = eval.sub(d_gap, thr);
    let far_diff = eval.sub(d_minus_thr, noised_bound_fma.far_gap_slack);
    let c = eval.mul(noised_bound_fma.is_far_gap, far_diff);
    eval.constraint(c);
    // W4: near rows use the true gap, far rows the cap THR - 1; POW2D pins the power
    // (and the cap's range).
    let not_far = eval.sub(one, noised_bound_fma.is_far_gap);
    let capped_diff = eval.sub(noised_bound_fma.exp_gap_capped, d_gap);
    let c = eval.mul(not_far, capped_diff);
    eval.constraint(c);
    let thr_m1 = eval.sub(thr, one);
    let cap_diff = eval.sub(noised_bound_fma.exp_gap_capped, thr_m1);
    let c = eval.mul(noised_bound_fma.is_far_gap, cap_diff);
    eval.constraint(c);
    // W5: align to the finer unit (committed for the degree budget).
    let shifted_p = eval.mul(noised_bound_fma.sig_product, noised_bound_fma.exp_gap_pow2);
    let al_p_ge = eval.mul(noised_bound_fma.product_scale_ge_addend, shifted_p);
    let al_p_lt = eval.mul(not_ge, noised_bound_fma.sig_product);
    let al_p = eval.add(al_p_ge, al_p_lt);
    eval.constraint_eq(noised_bound_fma.aligned_product, al_p);
    let shifted_c = eval.mul(lv.linf_floored_significand, noised_bound_fma.exp_gap_pow2);
    let al_c_lt = eval.mul(not_ge, shifted_c);
    let al_c_ge = eval.mul(noised_bound_fma.product_scale_ge_addend, lv.linf_floored_significand);
    let al_c = eval.add(al_c_lt, al_c_ge);
    eval.constraint_eq(noised_bound_fma.aligned_addend, al_c);
    // W6: the aligned sum is linear, so later constraints can use it without a separate column.
    let folded_magnitude = eval.add(noised_bound_fma.aligned_product, noised_bound_fma.aligned_addend);
    // W8: RNERND takes 17-bits inputs, so we compress the aligned sum to 17 bits, keeping
    // enough information for rounding. So is_wide = 1 when the aligned sum is >= 2^17.
    // The wide split aligned_magnitude = K*2^compression_shift + R (K's floor RC16 is filtered on IS_WIDE; R's two
    // RC16s make the split exact; narrow rows pin SHIFT = 0, SHIFT_POW2 = 1).
    eval.constraint_bool(noised_bound_fma.is_wide);
    let not_wide = eval.sub(one, noised_bound_fma.is_wide);
    let c = eval.mul(not_wide, noised_bound_fma.compression_shift);
    eval.constraint(c);
    let pow_m1 = eval.sub(noised_bound_fma.shift_pow2, one);
    let c = eval.mul(not_wide, pow_m1);
    eval.constraint(c);
    let k_shifted = eval.mul(noised_bound_fma.compression_quotient, noised_bound_fma.shift_pow2);
    let split = eval.add(k_shifted, noised_bound_fma.compression_remainder);
    let split_diff = eval.sub(folded_magnitude, split);
    let c = eval.mul(noised_bound_fma.is_wide, split_diff);
    eval.constraint(c);
    // W9: sticky = [R != 0] two-sidedly; K0 is K's parity (the (K - K0)/2 RC16).
    let sticky = eval.mul(
        noised_bound_fma.compression_remainder,
        noised_bound_fma.compression_remainder_inverse,
    );
    eval.constraint_eq(sticky, noised_bound_fma.sticky);
    let not_sticky = eval.sub(one, noised_bound_fma.sticky);
    let c = eval.mul(not_sticky, noised_bound_fma.compression_remainder);
    eval.constraint(c);
    eval.constraint_bool(noised_bound_fma.compression_quotient_lsb);
    // W10: the RNERND key (round-to-odd on wide rows) and its committed unit.
    let key_diff = eval.sub(noised_bound_fma.rounding_significand_key, folded_magnitude);
    let c = eval.mul(not_wide, key_diff);
    eval.constraint(c);
    let not_k0 = eval.sub(one, noised_bound_fma.compression_quotient_lsb);
    let odd_bump = eval.mul(noised_bound_fma.sticky, not_k0);
    let k_odd = eval.add(noised_bound_fma.compression_quotient, odd_bump);
    let key_diff = eval.sub(noised_bound_fma.rounding_significand_key, k_odd);
    let c = eval.mul(noised_bound_fma.is_wide, key_diff);
    eval.constraint(c);
    eval.constraint_bool(noised_bound_fma.rounding_significand_key_high_bit);
    let fine_ge = eval.mul(noised_bound_fma.product_scale_ge_addend, ec);
    let fine_lt = eval.mul(not_ge, ep);
    let fine = eval.add(fine_ge, fine_lt);
    let far_restore_inner = eval.sub(d_gap, noised_bound_fma.exp_gap_capped);
    let far_restore = eval.mul(noised_bound_fma.is_far_gap, far_restore_inner);
    let wide_shift = eval.mul(noised_bound_fma.is_wide, noised_bound_fma.compression_shift);
    let key_scale = eval.add(fine, far_restore);
    let key_scale = eval.add(key_scale, wide_shift);
    eval.constraint_eq(noised_bound_fma.key_scale, key_scale);
    // W11: CLAMP22 selects the cut and RNERND binds the rounded fields and flags.
    // W12 binds the exponent on normal outputs and zeroes it otherwise.
    let expected_exp = eval.add(noised_bound_fma.key_scale, noised_bound_fma.width_adjust);
    let inner = eval.sub(noised_bound_fma.out_exp, expected_exp);
    let not_oiz = eval.sub(one, noised_bound_fma.out_is_zero);
    let not_oez = eval.sub(one, noised_bound_fma.out_exp_is_zero);
    let gated = eval.mul(not_oiz, not_oez);
    let c = eval.mul(gated, inner);
    eval.constraint(c);
    let c = eval.mul(noised_bound_fma.out_exp_is_zero, noised_bound_fma.out_exp);
    eval.constraint(c);
    let c = eval.mul(noised_bound_fma.out_is_zero, noised_bound_fma.out_exp);
    eval.constraint(c);
    let c = eval.mul(noised_bound_fma.out_is_zero, noised_bound_fma.out_mantissa);
    eval.constraint(c);

    // H3 — alpha = RNE(448 / noised_bound) is the DIV448 lookup on the FMA output's
    // affine code (+ its RC16s + the alpha-mantissa PAIR128); nothing arithmetic to do here.
    // No denominator floor exists — the reference floors the norms (H0), never the noised
    // bound, and noised_bound >= linf_f >= 2^-32 by RNE monotonicity.

    // H4 — beta_1 = RNE(alpha * l2f) (the FLOORED l2, exactly like the reference).
    // Alpha is structurally normal: M = 128 + mantissa, E* = ALPHA_EXP.
    let m_alpha = eval.add(c128, lv.alpha_mantissa);
    let sig_expected = eval.mul(m_alpha, lv.l2_floored_significand);
    let e_sum = eval.add(lv.alpha_exp, lv.l2_floored_exponent);
    mul_block_constraints(eval, &lv.alpha_l2_multiply, sig_expected, e_sum);

    // S1 — jackpot check 3: the committed sigma significand M(alpha)*M(l2f) (the sigma
    // CTL tuple's degree-1 handle; its exponent rides the tuple as an affine expression).
    eval.constraint_eq(lv.sigma_significand, sig_expected);

    // S2 — jackpot check 4: the sigma significand's width bit. Boolean here; the two
    // filtered RC16s (`ctl::scale_lut_lookups`) force it to 1 exactly when
    // SIGMA_SIGNIFICAND >= 2^15, making the sigma encoding exported on the group-tuple
    // channel exact.
    eval.constraint_bool(lv.sigma_sig_is_wide);

    // ==== S3 — jackpot check 4: SIGMA_NORM = SIGMA_SIGNIFICAND * (2 - SIGMA_SIG_IS_WIDE),
    // the significand normalized into [2^15, 2^16). Pad rows carry the same valid scale
    // chain but are excluded from the group-tuple CTL. ====
    let two = eval.i32(2);
    let doubling = eval.sub(two, lv.sigma_sig_is_wide);
    let norm_expected = eval.mul(lv.sigma_significand, doubling);
    eval.constraint_eq(lv.normalized_sigma_significand, norm_expected);

    // ==== H5 — beta = RNE(beta_1 * dos); its output columns are used directly in the group tuple. ====
    let b1 = &lv.alpha_l2_multiply;
    let m_b1 = m_field(eval, b1.out_exp_is_zero, b1.out_mantissa);
    let e_b1 = e_star_field(eval, b1.out_exp, b1.out_exp_is_zero);
    let m_dos = eval.add(c128, dos_mantissa);
    let sig_expected = eval.mul(m_b1, m_dos);
    let e_sum = eval.add(e_b1, dos_exp);
    mul_block_constraints(eval, &lv.beta_scale_multiply, sig_expected, e_sum);

    // T — the jackpot liveness gate (check 1).

    // T1 — the group-tuple CTL binds InputQuant's committed DEAD_BOUND to the floored-L2
    // threshold with no column and no constraint here: the looked tuple's bound component is
    // the device-specific affine BF16 code of `tau_idle * delta * l2f`
    // (`ctl::ctl_looked_scale_group_tuple`). InputQuant's C
    // group proves the per-element liveness certificates against that bound and counts
    // them into DEAD_COUNT, CTL-bound to this row.

    // T2 — the per-side running totals. Row 0 is an A row (`h >= 1`, pinned by the
    // verifier-recomputed verifier-known columns), so the anchors are unconditional. The B-side
    // increment gate is 1 - IS_A - IS_PAD: equal to (1 - IS_A)*(1 - IS_PAD) because
    // IS_A*IS_PAD = 0 identically on the known columns, and one degree lower. Pad rows
    // increment neither side — their DEAD_COUNT is outside the group-tuple CTL's filter and
    // must not leak into the totals.
    let anchor_a = eval.sub(lv.running_dead_a, lv.dead_count);
    eval.constraint_first_row(anchor_a);
    eval.constraint_first_row(lv.running_dead_b);
    let inc_a = eval.mul(nv.is_a_row, nv.dead_count);
    let expected_a = eval.add(lv.running_dead_a, inc_a);
    let c = eval.sub(nv.running_dead_a, expected_a);
    eval.constraint_transition(c);
    let b_gate = {
        let a_or_pad = eval.add(nv.is_a_row, nv.is_pad);
        eval.sub(one, a_or_pad)
    };
    let inc_b = eval.mul(b_gate, nv.dead_count);
    let expected_b = eval.add(lv.running_dead_b, inc_b);
    let c = eval.sub(nv.running_dead_b, expected_b);
    eval.constraint_transition(c);

    // T3 — the DEAD_LIMIT gates are the two RC16 lookups
    // `DEAD_LIMIT - RUNNING_DEAD - 2^16·DEAD_SLACK_HI` filtered on IS_LAST_ROW
    // (`ctl::scale_lut_lookups`); the only arithmetic here is the booleanness of the two
    // high-bit witnesses. Soundness: every live DEAD_COUNT is CTL-bound to an InputQuant
    // group total of at most k boolean flags, so the running totals stay below 2^22 while
    // the limits sit below 2^16 — an honest slack fits the high bit plus one RC16 limb,
    // and an over-limit total wraps the slack to ~p, outside RC16's [0, 2^16) window for
    // either high-bit value.
    eval.constraint_bool(lv.dead_slack_hi_a);
    eval.constraint_bool(lv.dead_slack_hi_b);

    // ==== F — the jackpot noise floor (check 2) is the unfiltered RC16 lookup
    // `SIGMA_ENC - 268`, where S2 makes `SIGMA_ENC = e(sigma) + 268` affine. No additional
    // witness column or arithmetic constraint is needed. ====
}

// The Stark impl

/// ScaleStark: one row per matrix row — the l2/linf norms and the alpha/beta scale chain
/// (see the module docs). A CTL party of the fp8 batch; the batch driver is the only
/// supported proving path.
#[derive(Clone, Debug)]
pub struct ScaleStark<F, const D: usize> {
    pub program: ScaleProgram,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> ScaleStark<F, D> {
    pub fn new(program: ScaleProgram) -> Self {
        Self {
            program,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for ScaleStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_SCALE_COLUMNS, NUM_SCALE_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget = StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_SCALE_COLUMNS, NUM_SCALE_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_scale_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_scale_constraints(vars, &mut evaluator);
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    // Party to the group-tuples channel plus the committed LUT channels.
    fn requires_ctls(&self) -> bool {
        true
    }

    // No in-trace lookups: every table this AIR consumes is a committed LUT oracle instance
    // (declared in `super::ctl::scale_lut_lookups`), and the group-tuple channel is a CTL.
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};
    use starky::util::trace_rows_to_poly_values;

    use super::*;
    use crate::v4::circuit::luts::LutChecker;
    use crate::v4::circuit::scale_stark::columns::SCALE_COL_MAP;
    use crate::v4::circuit::scale_stark::ctl::scale_lut_lookups;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type S = ScaleStark<F, D>;

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    /// 4 + 4 = 8 rows (power of two, no padding), k = 4096 (power of two: single-element and
    /// constant rows have *exact* f32 norms); `Wl2 = 27 - 12 = 15`.
    fn test_program() -> ScaleProgram {
        ScaleProgram::new_for_device(4, 4, 4096, 4, Device::B200)
    }

    /// A deterministic nonnegative finite bf16 code with exponent field in `[lo_exp, hi_exp]`
    /// (normal for exponents >= 1; subnormal or zero at exponent 0).
    fn test_code(i: usize, salt: u64, lo_exp: u64, hi_exp: u64) -> u16 {
        let x = (i as u64).wrapping_mul(salt).rotate_left(17) ^ salt;
        let exp = lo_exp + x % (hi_exp - lo_exp + 1);
        let mant = (x >> 32) & 0x7F;
        ((exp << 7) | mant) as u16
    }

    /// A deterministic prequant row: pseudorandom full-range int8 values and normal
    /// (positive) block scales with exponent fields in `[lo_exp, hi_exp]` — every decoded
    /// element is normal or zero, since `|int * scale| in [2^(lo_exp-127), 2^(hi_exp-119))`.
    fn gen_prequant_row(k: usize, salt: u64, lo_exp: u64, hi_exp: u64) -> (Vec<i8>, Vec<u16>) {
        let ints = (0..k)
            .map(|i| {
                let x = (i as u64).wrapping_mul(salt).rotate_left(23) ^ salt;
                (x & 0xFF) as u8 as i8
            })
            .collect();
        let scales = (0..k / BLOCK_SIZE)
            .map(|b| test_code(b, salt ^ 0xABCD_EF01_2345_6789, lo_exp, hi_exp))
            .collect();
        (ints, scales)
    }

    /// The test tuple inventory — every structural path: generic, all-zero, single-element
    /// (exact f32 norm), constant power-of-two (binade-bottom claim, `IS_BOTTOM = 1`), tiny
    /// and subnormal scale exponents (subnormal decodes and max-abs; subnormal sqrt claim
    /// snapping to l2 = 0), huge exponents, mixed, and a constant min-normal row
    /// (`t = 128, f = 1`: the bottom-flag carve-out, `IS_BOTTOM = 0`).
    fn test_tuples(program: &ScaleProgram) -> (Vec<ScaleRowTuple>, Vec<ScaleRowTuple>) {
        let k = program.k;
        let n_blocks = k / BLOCK_SIZE;
        let wl2 = program.wl2();
        let mk = |ints: &[i8], scales: &[u16]| ScaleRowTuple::from_prequant_row_for_device(ints, scales, wl2, program.device);
        let (gi, gs) = gen_prequant_row(k, 0x9E3779B97F4A7C15, 110, 140);
        let mut single_ints = vec![0i8; k];
        single_ints[7] = 1;
        let mut single_scales = vec![0x3F80u16; n_blocks];
        single_scales[0] = (130 << 7) | 55; // X_7 = 1 * scale = the scale code exactly
        let (mi, ms) = gen_prequant_row(k, 0xA0761D6478BD642F, 40, 215);
        let a = vec![
            mk(&gi, &gs),
            mk(&vec![0i8; k], &vec![0x3F80u16; n_blocks]),
            mk(&single_ints, &single_scales),
            // 64 * 2^-13 = 2^-7 per element: the constant power-of-two row (t = 128, f = 120).
            mk(&vec![64i8; k], &vec![114 << 7; n_blocks]),
        ];
        let tiny: Vec<u16> = (0..n_blocks).map(|b| test_code(b, 0xC2B2AE3D27D4EB4F, 0, 6)).collect();
        let huge: Vec<u16> = (0..n_blocks).map(|b| test_code(b, 0xD1B54A32D192ED03, 200, 250)).collect();
        let b = vec![
            // int 1 everywhere: X = scale exactly (RNE-exact products).
            mk(&vec![1i8; k], &tiny),
            mk(&vec![1i8; k], &huge),
            mk(&mi, &ms),
            // X = 2^-126 (min normal) everywhere: t = 128, f = 1 — the bottom-flag carve-out.
            mk(&vec![1i8; k], &vec![1 << 7; n_blocks]),
        ];
        (a, b)
    }

    fn test_trace() -> (ScaleProgram, Vec<[F; NUM_SCALE_COLUMNS]>, [F; NUM_SCALE_PUBLIC_INPUTS]) {
        let program = test_program();
        let (a, b) = test_tuples(&program);
        let (rows, pis) = program.generate_trace::<F>(&a, &b);
        (program, rows, pis)
    }

    // Checking harnesses: arithmetic constraints (with the proper first/last/transition
    // selectors) and lookup membership in the actual committed table definitions.
    // ==============================================================================================

    fn check_constraints(
        program: &ScaleProgram,
        rows: &[[F; NUM_SCALE_COLUMNS]],
        pis: &[F; NUM_SCALE_PUBLIC_INPUTS],
    ) -> Result<(), String> {
        let stark = S::new(program.clone());
        let n = rows.len();
        for i in 0..n {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], pis);
            // Standard selectors: z_first on row 0, z_last on the last row, z_transition
            // excluding the last -> first wrap — exactly like the real prover.
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::from_bool(i != n - 1),
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            for acc in consumer.accumulators() {
                if acc != F::ZERO {
                    return Err(format!("constraints do not vanish on row {i}"));
                }
            }
        }
        Ok(())
    }

    fn check_lut_lookups(
        program: &ScaleProgram,
        rows: &[[F; NUM_SCALE_COLUMNS]],
        pis: &[F; NUM_SCALE_PUBLIC_INPUTS],
    ) -> Result<(), String> {
        let lookups = scale_lut_lookups::<F>(program);
        let mut tables: Vec<_> = lookups.iter().map(|lookup| lookup.table).collect();
        tables.sort_unstable();
        tables.dedup();
        let polys = trace_rows_to_poly_values(rows.to_vec());
        LutChecker::<F>::new(&tables).check_trace(&lookups, &polys, pis, "ScaleStark")
    }

    #[test]
    fn lookup_checker_uses_shared_pow2d_domain_and_values() {
        let (program, rows, pis) = test_trace();
        // Check lookup membership alone: these edits do not rebuild the FMA arithmetic.
        // The shared table includes InputQuant's shifts 18 and 19 even though Scale's
        // honest FMA fill caps its gap at 17.
        for shift in [18, 19] {
            let mut changed = rows.clone();
            let v: &mut ScaleColumnsView<F> = changed[0].borrow_mut();
            v.noised_bound_fma.exp_gap_capped = F::from_canonical_u64(shift);
            v.noised_bound_fma.exp_gap_pow2 = F::from_canonical_u64(1 << shift);
            check_lut_lookups(&program, &changed, &pis).expect("the shared table serves this power of two");
        }
        for (shift, power) in [(20, 1 << 20), (19, (1 << 19) + 1)] {
            let mut changed = rows.clone();
            let v: &mut ScaleColumnsView<F> = changed[0].borrow_mut();
            v.noised_bound_fma.exp_gap_capped = F::from_canonical_u64(shift);
            v.noised_bound_fma.exp_gap_pow2 = F::from_canonical_u64(power);
            let error =
                check_lut_lookups(&program, &changed, &pis).expect_err("an out-of-domain key or incorrect power must fail");
            assert!(error.contains("Pow2D"), "unexpected rejection: {error}");
        }
    }

    fn check_all(
        program: &ScaleProgram,
        rows: &[[F; NUM_SCALE_COLUMNS]],
        pis: &[F; NUM_SCALE_PUBLIC_INPUTS],
    ) -> Result<(), String> {
        check_constraints(program, rows, pis)?;
        check_lut_lookups(program, rows, pis)
    }

    // Honest-trace tests

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        let (program, rows, pis) = test_trace();
        check_constraints(&program, &rows, &pis).unwrap();

        // The lower protocol boundary reaches Wl2 = 17, one bit wider than the old
        // envelope. Exercise both constraints and lookups in this existing honest test.
        for (device, dr, dos) in [(Device::H100, 0x40B5, 0x38B5), (Device::B200, 0x4035, 0x3835)] {
            let program = ScaleProgram::new_for_device(4, 4, PublicParams::MIN_K, 32, device);
            assert_eq!(program.wl2(), 17);
            assert_eq!(program.dr_code(), dr);
            assert_eq!(program.dos_code(), dos);
            let (a, b) = test_tuples(&program);
            let (rows, pis) = program.generate_trace::<F>(&a, &b);
            check_all(&program, &rows, &pis).unwrap();
        }
    }

    #[test]
    fn honest_trace_satisfies_all_lut_lookups() {
        let (program, rows, pis) = test_trace();
        check_lut_lookups(&program, &rows, &pis).unwrap();
    }

    #[test]
    fn padded_trace_satisfies_constraints_and_lut_lookups() {
        // Non-power-of-two geometry: 3 + 4 = 7 live rows pad to 8 (the last row is the pad
        // row).
        let program = ScaleProgram::new_for_device(3, 4, 4096, 4, Device::B200);
        assert_eq!(program.num_rows(), 8);
        let (a8, b) = test_tuples(&test_program());
        let a: Vec<ScaleRowTuple> = a8[..3].to_vec();
        let (rows, pis) = program.generate_trace::<F>(&a, &b);
        assert_eq!(rows.len(), 8);
        // The known columns are bit-exact with the trace fill (the batch verifier's check).
        let known = program.known_values::<F>();
        for (c, col) in known.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(col.values[r], row[c], "known column {c} row {r}");
            }
        }
        check_constraints(&program, &rows, &pis).unwrap();
        check_lut_lookups(&program, &rows, &pis).unwrap();
    }

    #[test]
    fn trace_chain_is_bit_exact_vs_native_derive_row_scales() {
        // Compare floored norms and derived alpha/beta with the native functions, including zero rows.
        let (program, rows, _) = test_trace();
        for (i, row) in rows.iter().enumerate() {
            let v: &ScaleColumnsView<F> = row.borrow();
            let l2_code = (4 * to_u64(v.grid_snap_quotient)) as u16;
            let linf_code = ((to_u64(v.linf_exp) << 7) | to_u64(v.linf_mantissa)) as u16;
            let l2f = bf16_max(l2_code, NORM_FLOOR_CODE);
            let linf_f = bf16_max(linf_code, NORM_FLOOR_CODE);
            let (alpha, beta) = Fp8E4M3Quant::new(program.device)
                .derive_row_scales(l2f, linf_f, program.r)
                .expect("native chain on the floored norms");
            // The committed floored fields are the floored codes' (M, E*).
            let m = |code: u16| u64::from(if code >> 7 == 0 { code } else { 128 + (code & 0x7F) });
            let e_star = |code: u16| u64::from((code >> 7).max(1));
            assert_eq!(to_u64(v.l2_floored_significand), m(l2f), "row {i}: floored l2 M");
            assert_eq!(to_u64(v.l2_floored_exponent), e_star(l2f), "row {i}: floored l2 E*");
            assert_eq!(to_u64(v.linf_floored_significand), m(linf_f), "row {i}: floored linf M");
            assert_eq!(to_u64(v.linf_floored_exponent), e_star(linf_f), "row {i}: floored linf E*");
            let alpha_trace = (to_u64(v.alpha_exp) << 7 | to_u64(v.alpha_mantissa)) as u16;
            let beta_trace = (to_u64(v.beta_scale_multiply.out_exp) << 7 | to_u64(v.beta_scale_multiply.out_mantissa)) as u16;
            assert_eq!(alpha_trace, alpha, "row {i}: alpha differs from the native scheme");
            assert_eq!(beta_trace, beta, "row {i}: beta differs from the native scheme");
        }
    }

    #[test]
    fn all_zero_row_is_provable_with_reference_scales() {
        // A zero row must satisfy all constraints with finite alpha and positive beta after flooring.
        let (program, rows, pis) = test_trace();
        let v: &ScaleColumnsView<F> = rows[1].borrow();
        assert_eq!(v.frame_sum_is_zero, F::ONE, "test premise: row 1 is the all-zero row");
        assert_eq!((v.linf_exp, v.linf_mantissa), (F::ZERO, F::ZERO), "test premise: zero linf");
        assert_eq!((v.l2_ge_floor, v.linf_ge_floor), (F::ZERO, F::ZERO), "both norms floored");
        let (alpha, beta) = Fp8E4M3Quant::new(program.device)
            .derive_row_scales(NORM_FLOOR_CODE, NORM_FLOOR_CODE, program.r)
            .expect("native scales on the floor");
        assert_eq!((to_u64(v.alpha_exp) << 7 | to_u64(v.alpha_mantissa)) as u16, alpha);
        assert_eq!(
            (to_u64(v.beta_scale_multiply.out_exp) << 7 | to_u64(v.beta_scale_multiply.out_mantissa)) as u16,
            beta
        );
        assert_ne!(beta, 0, "the floored l2 keeps the noise scale strictly positive");
        check_all(&program, &rows, &pis).unwrap();
    }

    #[test]
    fn sqrt_matches_native_row_norms_on_exact_rows() {
        // For these rows, native f32 summation is exact and agrees with the framed integer
        // sum. Compare the resulting snapped L2 values.
        let program = test_program();
        let k = program.k;
        let n_blocks = k / BLOCK_SIZE;
        let mut single_ints = vec![0i8; k];
        single_ints[7] = 1;
        let mut single_scales = vec![0x3F80u16; n_blocks];
        single_scales[0] = (130 << 7) | 55;
        let mut single_decoded = vec![0u16; k];
        single_decoded[7] = (130 << 7) | 55;
        // Exclude rows whose f32 squares underflow. Their framed integer sums retain
        // values that the native f32 chain loses.
        let rows: Vec<(Vec<i8>, Vec<u16>, Vec<u16>)> = vec![
            (vec![0i8; k], vec![0x3F80u16; n_blocks], vec![0u16; k]),
            (single_ints, single_scales, single_decoded),
            // 64 * 2^-13 = 2^-7: decoded code 120 << 7.
            (vec![64i8; k], vec![114 << 7; n_blocks], vec![120 << 7; k]),
            // 1 * scale = scale exactly.
            (vec![1i8; k], vec![(90 << 7) | 127; n_blocks], vec![(90 << 7) | 127; k]),
            (vec![1i8; k], vec![60 << 7; n_blocks], vec![60 << 7; k]),
        ];
        for (ri, (ints, scales, decoded)) in rows.iter().enumerate() {
            let (l2_native, linf_native) = Fp8E4M3Quant::new(program.device).row_norms(decoded).expect("native norms");
            let tuple = ScaleRowTuple::from_prequant_row_for_device(ints, scales, program.wl2(), Device::B200);
            let claim = if tuple.l2_frame_sum == 0 {
                SqrtClaim {
                    exp: 0,
                    mantissa: 0,
                    exp_is_zero: true,
                }
            } else {
                rne_sqrt_hat(
                    tuple.l2_frame_sum,
                    i64::from(tuple.frame_doubled_scale_exponent),
                    i64::from(program.wl2()),
                    k as u64,
                )
            };
            let snapped = (((claim.exp << 7) + claim.mantissa + 2) & !3) as u16;
            assert_eq!(snapped, l2_native, "row {ri}: snapped l2 differs from row_norms");
            assert_eq!(tuple.max_abs as u16, linf_native, "row {ri}: linf differs from row_norms");
        }
    }

    #[test]
    fn tuple_frame_sum_matches_hand_computation() {
        // B-group semantics on a two-block row. Block 0: ints {1, 2}, scale 2^3 (code 130<<7):
        // n = 1 + 4 = 5, M = 128, p = 128^2 * 5, e2 = 260 (the frame). Block 1: int -3, scale
        // 3.0 (code (128<<7)|64): n = 9, M = 192, p = 192^2 * 9, e2 = 256, sigma = 4.
        let program = test_program();
        let wl2 = program.wl2();
        let n_blocks = program.k / BLOCK_SIZE;
        let mut ints = vec![0i8; program.k];
        let mut scales = vec![0x3F80u16; n_blocks];
        ints[0] = 1;
        ints[1] = 2;
        scales[0] = 130 << 7;
        ints[8] = -3;
        scales[1] = (128 << 7) | 64;
        let tuple = ScaleRowTuple::from_prequant_row_for_device(&ints, &scales, wl2, Device::B200);
        assert_eq!(tuple.frame_doubled_scale_exponent, 260);
        // Decoded elements: 8 (code 130<<7), 16 (code 131<<7), -9 (|code| (130<<7)|16).
        assert_eq!(tuple.max_abs, (131 << 7) as u32);
        let expected = ((128u64 * 128 * 5) << wl2) + (((192u64 * 192 * 9) << wl2) >> 4);
        assert_eq!(tuple.l2_frame_sum, expected);
    }

    // The sqrt bracket : uniqueness, ties, the quarter-ulp zone, f64 differential

    fn claim_pred(c: SqrtClaim) -> SqrtClaim {
        let (mut t, mut f) = (c.t(), c.f());
        assert!(t > 0, "no predecessor below +0");
        t -= 1;
        if t == 127 && f >= 2 {
            t = 255;
            f -= 1;
        }
        SqrtClaim::from_tf(t, f)
    }

    fn claim_succ(c: SqrtClaim) -> SqrtClaim {
        let (mut t, mut f) = (c.t(), c.f());
        t += 1;
        if t == 256 {
            t = 128;
            f += 1;
        }
        SqrtClaim::from_tf(t, f)
    }

    fn bracket_accepts(s: u64, e_max: i64, wl2: i64, k: u64, c: SqrtClaim) -> bool {
        let (lo, hi) = sqrt_bracket_ok(s, e_max, wl2, k, c.t(), c.f());
        lo && hi
    }

    #[test]
    fn sqrt_bracket_accepts_exactly_one_claim_fuzz() {
        // For pseudorandom (s, e_max, k): the bracket accepts the RNE claim and rejects both
        // neighbors — the uniqueness the AIR's soundness rests on.
        let mut x: u64 = 0x243F6A8885A308D3;
        let mut step = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for k in [
            PublicParams::MIN_K as u64,
            2048,
            2080,
            4096,
            32768,
            PublicParams::MAX_K as u64,
        ] {
            let clk = 64 - (k - 1).leading_zeros() as i64;
            let wl2 = 27 - clk;
            for _ in 0..2000 {
                let frame_exp = 1 + step() % 250;
                let e_max = 2 * frame_exp as i64;
                // Live sums start at 2^wl2 (the frame-attaining block's minimum term), so zero
                // and subnormal claims are in scope — the one-sided zero arm is part of the
                // uniqueness contract too. The cap keeps sqrt(v_hat) < 2^254.5 (claims with a
                // representable bf16 code; rows beyond are protocol-unprovable: l2 rounds to
                // inf, and honest sums are < 2^43 anyway), while still covering the entire
                // committed S < 2^62 range for small frames.
                let lo = wl2;
                let hi = (550 - e_max).min(61);
                let s = (1u64 << lo) + step() % ((1u64 << hi) - (1u64 << lo));
                let claim = rne_sqrt_hat(s, e_max, wl2, k);
                assert!(bracket_accepts(s, e_max, wl2, k, claim), "RNE claim rejected");
                if claim.t() > 0 {
                    assert!(
                        !bracket_accepts(s, e_max, wl2, k, claim_pred(claim)),
                        "predecessor also accepted (s={s}, e_max={e_max}, k={k}, claim=({}, {}))",
                        claim.t(),
                        claim.f(),
                    );
                }
                assert!(
                    !bracket_accepts(s, e_max, wl2, k, claim_succ(claim)),
                    "successor also accepted (s={s}, e_max={e_max}, k={k}, claim=({}, {}))",
                    claim.t(),
                    claim.f(),
                );
            }
        }
    }

    #[test]
    fn sqrt_bracket_ties_go_to_even() {
        // Construct exact upper-midpoint ties: S*2^D = (4t + 2)^2 * k, i.e.
        // sqrt(v_hat) = (t + 1/2)*2^(f-7) exactly. The even neighbor accepts (non-strict), the
        // odd rejects (strict) — on both sides of the tie.
        let k = 4096u64;
        let wl2 = 15i64;
        for (t, f) in [
            (128i64, 20i64),
            (129, 20),
            (200, 5),
            (254, 100),
            (255, 100),
            (130, 1),
            (131, 1),
        ] {
            // Choose e_max to make D = e_max + 4 - wl2 - 2f nonpositive and s integer:
            // s = (4t + 2)^2 * k * 2^(-D).
            let e_max = 2 * f + wl2 - 4 - 6; // D = -6
            if e_max < 2 {
                continue;
            }
            let s = (4 * t + 2).pow(2) as u64 * k * 64;
            if s >= 1 << 62 {
                continue;
            }
            let tie_even = t % 2 == 0;
            let this = SqrtClaim::from_tf(t, f);
            let succ = claim_succ(this);
            assert_eq!(
                bracket_accepts(s, e_max, wl2, k, this),
                tie_even,
                "tie: claim below must accept iff even (t={t}, f={f})"
            );
            assert_eq!(
                bracket_accepts(s, e_max, wl2, k, succ),
                !tie_even,
                "tie: claim above must accept iff even (t={t}, f={f})"
            );
            // And RNE lands on whichever neighbor is even.
            let rne = rne_sqrt_hat(s, e_max, wl2, k);
            let expect = if tie_even { this } else { succ };
            assert_eq!(rne, expect, "tie RNE (t={t}, f={f})");
        }
    }

    #[test]
    fn sqrt_bracket_quarter_ulp_zone_is_unambiguous() {
        // The binade-boundary interval: sqrt(v_hat) in [127.5, 128) * 2^(f-7) — between the top-of-binade
        // claim's value (255 * 2^(f-8) = 127.5 * 2^(f-7)) and the binade start (128 * 2^(f-7)).
        // With the naive half-ulp lower midpoint both (255, f-1) and (128, f) accept the whole
        // zone; the IS_BOTTOM quarter-ulp correction partitions it exactly: (255, f-1) owns
        // [127.5, 127.75), (128, f) owns [127.75, 128), and the boundary goes to the even t.
        //
        // Parametrization: sqrt(v_hat) = X * 2^(f-9) (X in eighth-of-binade quarter-ulp units)
        // gives s = X^2 * k * 2^g with e_max = 2f + wl2 - 4 - g.
        let k = 4096u64;
        let wl2 = 15i64;
        let f = 40i64;
        let bottom = SqrtClaim::from_tf(128, f);
        let top = SqrtClaim::from_tf(255, f - 1);
        // Strictly inside (127.5, 127.75): X = 510.5 quarter-ulps; scale doubled to stay
        // integer: sqrt = 1021 * 2^(f-10), s = 1021^2 * k * 2^g, e_max = 2f + wl2 - 6 - g.
        {
            let (g, s) = (4, 1021 * 1021 * k * 16);
            let e_max = 2 * f + wl2 - 6 - g;
            assert!(
                bracket_accepts(s, e_max, wl2, k, top),
                "zone below quarter: top claim owns it"
            );
            assert!(
                !bracket_accepts(s, e_max, wl2, k, bottom),
                "zone below quarter: bottom must reject"
            );
            assert_eq!(rne_sqrt_hat(s, e_max, wl2, k), top);
        }
        // Strictly inside (127.75, 128): X = 511.5, doubled to 1023.
        {
            let (g, s) = (4, 1023 * 1023 * k * 16);
            let e_max = 2 * f + wl2 - 6 - g;
            assert!(
                bracket_accepts(s, e_max, wl2, k, bottom),
                "zone above quarter: bottom claim owns it"
            );
            assert!(!bracket_accepts(s, e_max, wl2, k, top), "zone above quarter: top must reject");
            assert_eq!(rne_sqrt_hat(s, e_max, wl2, k), bottom);
        }
        // The exact quarter boundary X = 511 (sqrt = 127.75 * 2^(f-7)): bottom (t = 128, even)
        // accepts non-strictly, top (t = 255, odd) rejects strictly — ties-to-even across the
        // binade edge.
        {
            let (g, s) = (6, 511 * 511 * k * 64);
            let e_max = 2 * f + wl2 - 4 - g;
            assert!(bracket_accepts(s, e_max, wl2, k, bottom), "quarter tie: even bottom accepts");
            assert!(!bracket_accepts(s, e_max, wl2, k, top), "quarter tie: odd top rejects");
            assert_eq!(rne_sqrt_hat(s, e_max, wl2, k), bottom);
        }
        // The f = 1 carve-out: (128, 1)'s predecessor is the subnormal (127, 1) in the
        // same-ulp-width binade, so there is NO quarter correction: sqrt = 127.625 * 2^-6
        // (above the plain midpoint 127.5) belongs to (128, 1), and (127, 1) rejects.
        {
            let (g, s) = (4, 1021 * 1021 * k * 16);
            let e_max = 2 + wl2 - 6 - g; // 2*f with f = 1
            assert!(e_max >= 2, "carve-out sample must be constructible");
            let bottom1 = SqrtClaim::from_tf(128, 1);
            let sub127 = SqrtClaim::from_tf(127, 1);
            assert!(bracket_accepts(s, e_max, wl2, k, bottom1), "f = 1: plain half-ulp midpoint");
            assert!(
                !bracket_accepts(s, e_max, wl2, k, sub127),
                "f = 1: the subnormal neighbor rejects"
            );
            assert_eq!(rne_sqrt_hat(s, e_max, wl2, k), bottom1);
        }
    }

    #[test]
    fn sqrt_bracket_differential_vs_f64() {
        // Machine-check of the frame choice : for exactly-representable v_hat, the bracket
        // claim equals RNE_bf16(sqrt(v_hat)) computed through f64 (correctly rounded) — skipping
        // samples too close to a rounding boundary for the double rounding f64 -> f32 -> bf16
        // to be trusted.
        let k = 4096u64; // power of two: s * 2^e2 / k is exact in f64 for s < 2^52
        let wl2 = 15i64;
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let mut step = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut checked = 0u32;
        for _ in 0..4000 {
            // frame_exp <= 129 keeps the hat-frame sqrt below 2^126 (f32-representable, so the
            // f64 -> f32 -> bf16 reference chain stays finite); >= 30 keeps it far from the
            // subnormal fade.
            let frame_exp = 30 + step() % 100;
            let e_max = 2 * frame_exp as i64;
            let s = (1u64 << 48) + step() % ((1u64 << 52) - (1u64 << 48));
            let v_hat = s as f64 * ((e_max - 14 - wl2) as f64).exp2() / k as f64;
            let y = v_hat.sqrt();
            // Skip near-boundary samples (bf16 has 8 significand bits; require 2^-20 clearance).
            let scaled = y / y.log2().floor().exp2() * 128.0; // in [128, 256)
            let frac = (scaled * 2.0).fract(); // distance to the nearest half-integer grid
            if (frac - 0.5).abs() < 1e-3 || !(1e-3..=1.0 - 1e-3).contains(&frac) {
                continue;
            }
            // The hat frame maps the bf16 grid onto itself code-for-code (module docs), so the
            // bracket's claim code equals the bf16 code of the *real* value y * 2^-127.
            let expected = f32_to_bf16((y * (-127.0f64).exp2()) as f32).expect("in range");
            let claim = rne_sqrt_hat(s, e_max, wl2, k);
            let claim_code = ((claim.exp << 7) + claim.mantissa) as u16;
            assert_eq!(
                claim_code, expected,
                "bracket disagrees with f64 RNE (s={s}, e_max={e_max}, y={y:e})"
            );
            checked += 1;
        }
        assert!(checked > 3000, "too many samples skipped ({checked} checked)");
    }

    #[test]
    fn sqrt_zero_claim_region_is_one_sided() {
        // A zero claim on a nonzero sum is valid through the upper midpoint, inclusive
        // (the tie rounds to even zero). Check below, at and above that boundary.
        let k = 4096u64;
        let wl2 = 15i64;
        // t = 0, f = 1: D = e_max + 4 - wl2 - 2 = e_max - 13; choose e_max = 2: D = -11:
        // boundary S*2^D = 4k at S = 4k * 2^11.
        let s_tie = (4 * k) << 11;
        let zero = SqrtClaim {
            exp: 0,
            mantissa: 0,
            exp_is_zero: true,
        };
        assert!(bracket_accepts(s_tie, 2, wl2, k, zero), "the boundary ties to even 0");
        assert!(
            bracket_accepts(s_tie - 1, 2, wl2, k, zero),
            "strictly inside the zero region (one-sided arm)"
        );
        assert!(bracket_accepts(1, 2, wl2, k, zero), "deep inside the zero region");
        assert!(
            !bracket_accepts(s_tie + 1, 2, wl2, k, zero),
            "above the boundary the zero claim loses"
        );
        assert_eq!(rne_sqrt_hat(s_tie, 2, wl2, k), zero);
        assert_eq!(rne_sqrt_hat(s_tie - 1, 2, wl2, k), zero);
        assert_eq!(rne_sqrt_hat(1, 2, wl2, k), zero);
        // Just above the boundary RNE steps to the smallest subnormal (t = 1, f = 1) — and the
        // one-sided bracket rejects zero there while accepting (1, 1).
        let above = rne_sqrt_hat(s_tie + 1, 2, wl2, k);
        assert_eq!((above.t(), above.f()), (1, 1));
        assert!(bracket_accepts(s_tie + 1, 2, wl2, k, above));
    }

    // Gadget fills vs. the native bf16 chain

    /// Deterministic nonnegative bf16 code: zero, subnormal-free normal small/large.
    fn fuzz_code(x: u64) -> u16 {
        match x % 8 {
            0 => 0,
            1 => ((1 + (x >> 8) % 8) << 7 | (x >> 40) & 0x7F) as u16, // tiny exponents
            2 => ((240 + (x >> 8) % 15) << 7 | (x >> 40) & 0x7F) as u16, // huge exponents
            _ => ((1 + (x >> 8) % 254) << 7 | (x >> 40) & 0x7F) as u16,
        }
    }

    #[test]
    fn mul_fill_matches_native_bf16_mul_fuzz() {
        let mut x: u64 = 0xA0761D6478BD642F;
        let mut step = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut checked = 0u32;
        for _ in 0..20000 {
            let a = fuzz_code(step());
            let b = fuzz_code(step());
            let Ok(native) = bf16_mul(a, b) else {
                continue; // overflow to non-finite: protocol-rejected upstream
            };
            // fill_mul asserts M6 (output exponent in range) — native Ok means in range.
            let fill = fill_mul(Bf16Fields::from_code(a), Bf16Fields::from_code(b));
            assert_eq!(fill.out.code(), native, "MUL fill vs native ({a:#06x} * {b:#06x})");
            checked += 1;
        }
        assert!(checked > 15000);
    }

    #[test]
    fn fma_fill_matches_native_bf16_fma_fuzz() {
        let program = test_program();
        let dr = Bf16Fields::from_code(program.dr_code());
        let mut x: u64 = 0xE7037ED1A0B428DB;
        let mut step = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut checked = 0u32;
        for _ in 0..20000 {
            let b = fuzz_code(step());
            let c = fuzz_code(step());
            let Ok(native) = bf16_fma(program.dr_code(), b, c) else {
                continue;
            };
            let fill = fill_fma(dr, Bf16Fields::from_code(b), Bf16Fields::from_code(c));
            assert_eq!(fill.out.code(), native, "FMA fill vs native (dr * {b:#06x} + {c:#06x})");
            checked += 1;
        }
        assert!(checked > 15000);
    }

    // Stark harness: degree, circuit parity, prove/verify

    #[test]
    fn degree_is_at_most_three() {
        test_stark_low_degree::<F, S, D>(S::new(test_program())).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        test_stark_circuit_constraints::<F, C, S, D>(S::new(test_program())).unwrap();
    }

    // No standalone prove/verify smoke test: this table is a CTL party (`requires_ctls`), so a
    // proof without the cross-table argument is not a supported object. The end-to-end proving
    // path is covered by `circuit::driver::tests::batch_proof_roundtrips_and_rejects_tampering`.

    // ==============================================================================================
    // Tamper tests: every forgery class must be rejected by the constraints or the LUT facts

    /// Rebuilds row `idx` from its tuple with a chosen (tampered) sqrt claim, keeping the
    /// structural columns — the row stays internally consistent downstream of the claim,
    /// so ONLY the bracket can reject it.
    fn rebuild_with_claim(
        program: &ScaleProgram,
        rows: &mut [[F; NUM_SCALE_COLUMNS]],
        idx: usize,
        tuple: &ScaleRowTuple,
        claim: SqrtClaim,
    ) {
        let orig = rows[idx];
        let ov: &ScaleColumnsView<F> = orig.borrow();
        let mut row = [F::ZERO; NUM_SCALE_COLUMNS];
        {
            let v: &mut ScaleColumnsView<F> = row.borrow_mut();
            program.fill_data_row(v, tuple, idx < program.h, idx, claim, false);
            v.is_last_row = ov.is_last_row;
            // `fill_data_row` never writes the running dead totals (`generate_trace`
            // accumulates them across rows), so the rebuilt row would hold zeros. Restore
            // the originals: the tamper leaves DEAD_COUNT untouched, so they still satisfy
            // the T2 recurrence and only the sqrt bracket can reject the row.
            v.running_dead_a = ov.running_dead_a;
            v.running_dead_b = ov.running_dead_b;
        }
        rows[idx] = row;
    }

    #[test]
    fn tamper_sqrt_claim_one_ulp_is_rejected() {
        let (program, rows, pis) = test_trace();
        let (a, _) = test_tuples(&program);
        for (name, shift) in [("pred", false), ("succ", true)] {
            let mut tampered = rows.clone();
            let honest = rne_sqrt_hat(
                a[0].l2_frame_sum,
                i64::from(a[0].frame_doubled_scale_exponent),
                i64::from(program.wl2()),
                program.k as u64,
            );
            let forged = if shift { claim_succ(honest) } else { claim_pred(honest) };
            rebuild_with_claim(&program, &mut tampered, 0, &a[0], forged);
            assert!(
                check_all(&program, &tampered, &pis).is_err(),
                "one-ulp-{name} sqrt claim must be rejected"
            );
        }
    }

    #[test]
    fn lower_comparison_rejects_underflow_without_final_borrow_column() {
        let program = ScaleProgram::new_for_device(1, 1, 4096, 4, Device::B200);
        let ones = ScaleRowTuple::from_prequant_row_for_device(
            &vec![1i8; program.k],
            &vec![0x3F80u16; program.k / BLOCK_SIZE],
            program.wl2(),
            program.device,
        );
        let (mut rows, pis) = program.generate_trace::<F>(&[ones], &[ones]);
        check_all(&program, &rows, &pis).unwrap();
        let honest: &ScaleColumnsView<F> = rows[0].borrow();
        let honest_grid_quotient = honest.grid_snap_quotient;

        // sqrt(1) = 1, but claim its successor 129/128. Both codes snap to 1,
        // so the remaining scale chain cannot detect this incorrect square-root claim.
        rebuild_with_claim(&program, &mut rows, 0, &ones, SqrtClaim::from_tf(129, 127));
        let forged: &ScaleColumnsView<F> = rows[0].borrow();
        assert_eq!(forged.grid_snap_quotient, honest_grid_quotient);
        check_constraints(&program, &rows, &pis).expect("the lower subtraction must reject through its lookup");
        let final_digit = forged.shifted_sum_carries[3]
            - forged.aligned_lower_boundary_limbs[ALIGNED_LOWER_LIMBS - 1]
            - forged.lower_comparison_borrows[super::super::columns::LOWER_BORROW_BITS - 1];
        assert_eq!(final_digit, -F::ONE, "position 6 must not borrow out to hide the underflow");
        assert!(check_lut_lookups(&program, &rows, &pis).is_err());
    }

    #[test]
    fn lower_alignment_rejects_nonzero_omitted_digit() {
        let program = ScaleProgram::new_for_device(1, 1, 1 << 16, 4, Device::B200);
        // Isolate the alignment equation: choose a bounded aggregate and forged claim
        // with lower_bound = 7*2^112 + scaled_sum. Truncating position 7 would accept.
        let tuple = ScaleRowTuple {
            l2_frame_sum: 260 << 49,
            frame_doubled_scale_exponent: 254,
            max_abs: 0x3F80,
            dead_count: 0,
        };
        let (mut rows, pis) = program.generate_trace::<F>(&[tuple], &[tuple]);
        check_all(&program, &rows, &pis).unwrap();
        rebuild_with_claim(&program, &mut rows, 0, &tuple, SqrtClaim::from_tf(240, 140));
        let forged: &ScaleColumnsView<F> = rows[0].borrow();
        assert_eq!(forged.alignment_quotient_is_4, F::ONE);
        assert_eq!(forged.lower_bracket_is_active, F::ONE);
        assert_eq!(to_u64(forged.lower_boundary_product_limbs[CLAIM_LIMBS - 1]), 7);
        let stored_lower_bound: u128 = forged
            .aligned_lower_boundary_limbs
            .iter()
            .enumerate()
            .map(|(i, &limb)| u128::from(to_u64(limb)) << (16 * (i + 1)))
            .sum();
        let scaled_sum = u128::from(tuple.l2_frame_sum) << (32 + to_u64(forged.sum_shift_remainder));
        assert_eq!(stored_lower_bound, scaled_sum);
        check_lut_lookups(&program, &rows, &pis).expect("the stored lower digits alone cannot detect the omitted digit");
        assert!(
            check_constraints(&program, &rows, &pis).is_err(),
            "the omitted digit must be constrained to zero"
        );
    }

    #[test]
    fn tamper_zero_claim_flag_is_rejected() {
        // Q2b: SQRT_CLAIM_IS_ZERO on a live t != 0 claim is shape-unsatisfiable (it forces
        // mantissa = 0 and exp_is_zero = 1, i.e. t = 0).
        let (program, rows, pis) = test_trace();
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[0].borrow_mut();
            assert_eq!(v.sqrt_claim_is_zero, F::ZERO, "test premise: row 0 claims t != 0");
            v.sqrt_claim_is_zero = F::ONE;
        }
        assert!(
            check_all(&program, &tampered, &pis).is_err(),
            "SQRT_CLAIM_IS_ZERO on a nonzero claim must be rejected"
        );
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[1].borrow_mut();
            assert_eq!(v.frame_sum_is_zero, F::ONE, "test premise: row 1 is the all-zero row");
            v.sqrt_claim_is_zero = F::ZERO;
            // Keep LOWER_BRACKET_IS_ACTIVE consistent with the forged flag
            // (frame_sum_is_nonzero * not_t_zero = 0 * 1),
            // so only downstream effects can reject — on a dead row both arms stay gated and
            // nothing catches it, but flipping the *gate* itself must be caught:
            v.lower_bracket_is_active = F::ONE;
        }
        assert!(
            check_all(&program, &tampered, &pis).is_err(),
            "LOWER_BRACKET_IS_ACTIVE = 1 on a zero-sum row must be rejected"
        );
    }

    #[test]
    fn tamper_lower_bracket_gate_is_rejected() {
        // The dangerous direction: dropping the lower arm on a live nonzero claim would let
        // oversized claims through — Q2b pins LOWER_BRACKET_IS_ACTIVE =
        // (1 - Z)*(1 - T_ZERO) exactly.
        let (program, rows, pis) = test_trace();
        for idx in [0usize, 3] {
            let mut tampered = rows.clone();
            {
                let v: &mut ScaleColumnsView<F> = tampered[idx].borrow_mut();
                assert_eq!(v.lower_bracket_is_active, F::ONE, "test premise: live nonzero claim");
                v.lower_bracket_is_active = F::ZERO;
            }
            assert!(
                check_all(&program, &tampered, &pis).is_err(),
                "clearing LOWER_BRACKET_IS_ACTIVE on live row {idx} must be rejected"
            );
        }
    }

    #[test]
    fn tamper_alignment_quotient_one_hot_is_rejected() {
        // The widened window's one-hot: a second q flag breaks the boolean flag sum; moving q
        // off the honest value breaks Q5's window equation (r2 leaves [0, 15]) or the Q8 mux.
        let (program, rows, pis) = test_trace();
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[0].borrow_mut();
            v.alignment_quotient_is_3 = F::ONE;
            v.alignment_quotient_is_4 = F::ONE;
        }
        assert!(
            check_all(&program, &tampered, &pis).is_err(),
            "two q flags at once must be rejected"
        );
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[0].borrow_mut();
            // Shift the one-hot by one position without touching r2 or the aligned limbs.
            let (q1, q2, q3, q4) = (
                v.alignment_quotient_is_1,
                v.alignment_quotient_is_2,
                v.alignment_quotient_is_3,
                v.alignment_quotient_is_4,
            );
            v.alignment_quotient_is_2 = q1;
            v.alignment_quotient_is_3 = q2;
            v.alignment_quotient_is_4 = q3;
            v.alignment_quotient_is_1 = q4;
        }
        assert!(
            check_all(&program, &tampered, &pis).is_err(),
            "rotating the q one-hot must be rejected"
        );
    }

    #[test]
    fn tamper_bottom_flag_is_rejected() {
        // A3 (row 3) is a constant power-of-two row: bottom claim (t = 128, exp = 120), flag 1.
        let (program, rows, pis) = test_trace();
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[3].borrow_mut();
            assert_eq!(v.sqrt_is_binade_bottom, F::ONE, "test premise: row 3 is a bottom claim");
            v.sqrt_is_binade_bottom = F::ZERO;
        }
        assert!(
            check_all(&program, &tampered, &pis).is_err(),
            "clearing IS_BOTTOM on a binade-bottom claim must be rejected"
        );
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[0].borrow_mut();
            assert_eq!(v.sqrt_is_binade_bottom, F::ZERO);
            v.sqrt_is_binade_bottom = F::ONE;
        }
        assert!(
            check_all(&program, &tampered, &pis).is_err(),
            "setting IS_BOTTOM on a non-bottom claim must be rejected"
        );
    }

    #[test]
    fn tamper_grid_snap_is_rejected() {
        let (program, rows, pis) = test_trace();
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[0].borrow_mut();
            // Shift the quotient and fix up the remainder so G1 still balances: G2/G3 must
            // catch it (GRID_SNAP_REMAINDER leaves [0, 3] / the l2 decode mismatches).
            v.grid_snap_quotient += F::ONE;
            v.grid_snap_remainder -= F::from_canonical_u64(4);
        }
        assert!(check_all(&program, &tampered, &pis).is_err());
    }

    #[test]
    fn tamper_alpha_is_rejected() {
        let (program, rows, pis) = test_trace();
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[0].borrow_mut();
            v.alpha_exp += F::ONE; // no longer the DIV448 value for this noised bound
        }
        assert!(check_all(&program, &tampered, &pis).is_err());
    }

    #[test]
    fn tamper_norm_floor_is_rejected() {
        let (program, rows, pis) = test_trace();
        // Flip each H0 order bit on a floored row (1: the all-zero row, GE = 0) and an
        // unfloored row (0: generic, GE = 1): the two-sided slack leaves RC16's range and the
        // floored-field muxes break.
        for (idx, which) in [(1usize, "l2"), (1, "linf"), (0, "l2"), (0, "linf")] {
            let mut tampered = rows.clone();
            {
                let v: &mut ScaleColumnsView<F> = tampered[idx].borrow_mut();
                match which {
                    "l2" => v.l2_ge_floor = F::ONE - v.l2_ge_floor,
                    _ => v.linf_ge_floor = F::ONE - v.linf_ge_floor,
                }
            }
            assert!(
                check_all(&program, &tampered, &pis).is_err(),
                "flipping the {which} floor bit on row {idx} must be rejected"
            );
        }
        // An unfloored zero significand must fail the floor mux.
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[1].borrow_mut();
            v.l2_floored_significand = F::ZERO;
        }
        assert!(
            check_all(&program, &tampered, &pis).is_err(),
            "forging the floored significand must be rejected"
        );
    }

    #[test]
    fn tamper_beta_output_is_rejected() {
        let (program, rows, pis) = test_trace();
        let beta = &SCALE_COL_MAP.beta_scale_multiply;
        for column in [beta.out_exp, beta.out_mantissa, beta.out_exp_is_zero] {
            let mut tampered = rows.clone();
            let value = &mut tampered[0][column];
            *value = if *value == F::ZERO { F::ONE } else { *value - F::ONE };
            assert!(
                check_all(&program, &tampered, &pis).is_err(),
                "changing beta output column {column} must fail its multiplication constraints or lookups"
            );
        }
    }

    #[test]
    fn tamper_noised_bound_fma_rounding_key_high_bit_is_rejected() {
        let (program, rows, pis) = test_trace();
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[0].borrow_mut();
            v.noised_bound_fma.rounding_significand_key_high_bit = F::ONE - v.noised_bound_fma.rounding_significand_key_high_bit;
            // RC16(ROUNDING_SIGNIFICAND_KEY - 2^16*b) breaks
        }
        assert!(check_all(&program, &tampered, &pis).is_err());
    }

    #[test]
    fn tamper_running_dead_totals_is_rejected() {
        let (program, rows, pis) = test_trace();
        // Understating the final total breaks the T2 transition into the last row.
        let mut tampered = rows.clone();
        let n = tampered.len();
        {
            let v: &mut ScaleColumnsView<F> = tampered[n - 1].borrow_mut();
            v.running_dead_a -= F::ONE;
        }
        assert!(
            check_all(&program, &tampered, &pis).is_err(),
            "understating the A dead total must be rejected"
        );
        // Forging a row's DEAD_COUNT breaks the first-row anchor.
        let mut tampered = rows.clone();
        {
            let v: &mut ScaleColumnsView<F> = tampered[0].borrow_mut();
            v.dead_count += F::ONE;
        }
        assert!(
            check_all(&program, &tampered, &pis).is_err(),
            "a forged dead count must be rejected"
        );
    }

    #[test]
    fn over_limit_dead_totals_are_rejected_by_the_gate() {
        // Push the A side past DEAD_LIMIT_A while keeping T2 satisfied: bump row 0's
        // DEAD_COUNT by LIMIT + 1 and propagate the bump through every running total — only
        // the T3 RC16 (DEAD_LIMIT_A - RUNNING_DEAD_A on the last row) is left to reject,
        // its key wrapping far outside [0, 2^16).
        let (program, rows, pis) = test_trace();
        let bump = pis[DEAD_LIMIT_A_PUBLIC_INPUT] + F::ONE;
        let mut tampered = rows.clone();
        for row in tampered.iter_mut() {
            let v: &mut ScaleColumnsView<F> = row.borrow_mut();
            v.running_dead_a += bump;
        }
        {
            let v: &mut ScaleColumnsView<F> = tampered[0].borrow_mut();
            v.dead_count += bump;
        }
        check_constraints(&program, &tampered, &pis).expect("T2 stays satisfied — only the gate rejects");
        assert!(
            check_lut_lookups(&program, &tampered, &pis).is_err(),
            "the T3 RC16 must reject an over-limit dead total"
        );
        // The boolean high bit cannot absorb the wrap either.
        for row in tampered.iter_mut() {
            let v: &mut ScaleColumnsView<F> = row.borrow_mut();
            v.dead_slack_hi_a = F::ONE;
        }
        check_constraints(&program, &tampered, &pis).expect("a set high bit is still boolean");
        assert!(
            check_lut_lookups(&program, &tampered, &pis).is_err(),
            "the high bit must not absorb an over-limit total"
        );
    }

    #[test]
    fn dead_limit_high_bit_extends_the_gate_window() {
        // A 17-bit slack (DEAD_LIMIT - RUNNING_DEAD >= 2^16, the envelope-edge allowance)
        // passes exactly through the boolean high bit: with it the RC16 key drops into
        // the window, without it the key sits at >= 2^16 and the gate rejects.
        let (program, rows, pis) = test_trace();
        let mut big = pis;
        big[DEAD_LIMIT_A_PUBLIC_INPUT] += F::from_canonical_u64(1 << 16);
        assert!(
            check_lut_lookups(&program, &rows, &big).is_err(),
            "a 17-bit slack must overflow the single RC16 limb"
        );
        let mut lifted = rows.clone();
        for row in lifted.iter_mut() {
            let v: &mut ScaleColumnsView<F> = row.borrow_mut();
            v.dead_slack_hi_a = F::ONE;
        }
        check_constraints(&program, &lifted, &big).expect("the high bit is boolean");
        check_lut_lookups(&program, &lifted, &big).expect("the high bit brings the 17-bit slack into the RC16 window");
    }

    #[test]
    fn dead_limit_public_inputs_are_load_bearing() {
        // Lowering DEAD_LIMIT_A below the honest total makes the honest trace fail the T3
        // gate: the policy threshold really is the public input, not the trace.
        let (program, rows, pis) = test_trace();
        let total_a = {
            let v: &ScaleColumnsView<F> = rows.last().unwrap().borrow();
            v.running_dead_a
        };
        assert_ne!(total_a, F::ZERO, "test premise: the A spike row has a dead entry");
        let mut bad = pis;
        bad[DEAD_LIMIT_A_PUBLIC_INPUT] = total_a - F::ONE;
        assert!(
            check_lut_lookups(&program, &rows, &bad).is_err(),
            "an under-total DEAD_LIMIT_A must reject the honest trace"
        );
    }

    // The jackpot noise-floor gate (group F)

    /// Fills one scratch row from a single-spike prequant row (`ints[0] = 1`, block-0 scale
    /// = `scale_code`, everything else zero) and returns the tuple with the gate's decision
    /// data `(E_SUM, SIG)`.
    fn spike_row_gate_data(program: &ScaleProgram, scale_code: u16) -> (ScaleRowTuple, i64, u64) {
        let mut ints = vec![0i8; program.k];
        ints[0] = 1;
        let mut scales = vec![0x3F80u16; program.k / BLOCK_SIZE];
        scales[0] = scale_code;
        let tuple = ScaleRowTuple::from_prequant_row_for_device(&ints, &scales, program.wl2(), Device::B200);
        let claim = rne_sqrt_hat(
            tuple.l2_frame_sum,
            i64::from(tuple.frame_doubled_scale_exponent),
            i64::from(program.wl2()),
            program.k as u64,
        );
        let mut row = [F::ZERO; NUM_SCALE_COLUMNS];
        let v: &mut ScaleColumnsView<F> = row.borrow_mut();
        program.fill_data_row(v, &tuple, true, 0, claim, false);
        let e_sum = (to_u64(v.alpha_exp) + to_u64(v.l2_floored_exponent)) as i64;
        (tuple, e_sum, to_u64(v.alpha_l2_multiply.sig_product))
    }

    /// Searches single-spike rows for one whose honest gate data lands in the boundary
    /// binade `E_SUM = 254`, on the requested side of the significand pivot. A spike row
    /// keeps `linf/l2f ~ sqrt(k)`, so `sigma = DELTA*alpha*l2f ~ 224/sqrt(k)`: k = 2^14
    /// lands `alpha*l2f` in [2, 4) (`pass = true` reachable) and k = 2^16 in [1, 2)
    /// (`pass = false` reachable); which binade of SIG the product decomposes into varies
    /// with the scale mantissa.
    fn find_boundary_spike(program: &ScaleProgram, pass: bool) -> ScaleRowTuple {
        for exp in [130u64, 131, 129] {
            for mantissa in 0..128u64 {
                let (tuple, e_sum, sig) = spike_row_gate_data(program, ((exp << 7) | mantissa) as u16);
                if e_sum == 254 && ((sig >= 1 << 15) == pass) {
                    return tuple;
                }
            }
        }
        panic!(
            "no spike row lands in the boundary binade with SIG {} 2^15",
            if pass { ">=" } else { "<" }
        );
    }

    /// A full honest B200 trace whose row 2 sits exactly at the sigma floor:
    /// `E_SUM = 254`, `SIG >= 2^15`, hence `SIGMA_ENC = 254 + 1 + 13 = 268`.
    fn boundary_binade_trace() -> (ScaleProgram, Vec<[F; NUM_SCALE_COLUMNS]>, [F; NUM_SCALE_PUBLIC_INPUTS]) {
        let program = ScaleProgram::new_for_device(4, 4, 1 << 14, 4, Device::B200);
        let (mut a, b) = test_tuples(&program);
        a[2] = find_boundary_spike(&program, true);
        let (rows, pis) = program.generate_trace::<F>(&a, &b);
        (program, rows, pis)
    }

    #[test]
    fn sigma_floor_boundary_binade_row_is_accepted() {
        let (program, rows, pis) = boundary_binade_trace();
        {
            let v: &ScaleColumnsView<F> = rows[2].borrow();
            assert_eq!(
                to_u64(v.alpha_exp)
                    + to_u64(v.l2_floored_exponent)
                    + to_u64(v.sigma_sig_is_wide)
                    + program.device.sigma_encoding_offset(),
                PROD_UNIT as u64,
                "test premise: row 2 sits exactly at sigma = 1"
            );
        }
        check_all(&program, &rows, &pis).unwrap();
    }

    /// A 2-row k = 2^16 trace whose A row violates the noise floor with `E_SUM = 254` and
    /// `SIG < 2^15`, built with `expect_valid = false` (the honest builder panics on it) and
    /// honest running totals — every arithmetic constraint holds, so only the sigma-floor
    /// RC16 can reject.
    fn below_floor_trace() -> (ScaleProgram, Vec<[F; NUM_SCALE_COLUMNS]>, [F; NUM_SCALE_PUBLIC_INPUTS]) {
        let program = ScaleProgram::new_for_device(1, 1, PublicParams::MAX_K, 4, Device::B200);
        let spike = find_boundary_spike(&program, false);
        let ones = ScaleRowTuple::from_prequant_row_for_device(
            &vec![1i8; program.k],
            &vec![0x3F80u16; program.k / BLOCK_SIZE],
            program.wl2(),
            Device::B200,
        );
        let mut rows = vec![[F::ZERO; NUM_SCALE_COLUMNS]; 2];
        for (i, (tuple, is_a)) in [(&spike, true), (&ones, false)].into_iter().enumerate() {
            let claim = rne_sqrt_hat(
                tuple.l2_frame_sum,
                i64::from(tuple.frame_doubled_scale_exponent),
                i64::from(program.wl2()),
                program.k as u64,
            );
            let v: &mut ScaleColumnsView<F> = rows[i].borrow_mut();
            program.fill_data_row(v, tuple, is_a, i, claim, false);
            v.is_last_row = F::from_bool(i == 1);
            v.running_dead_a = F::from_canonical_u64(u64::from(spike.dead_count));
            v.running_dead_b = F::from_canonical_u64(if i == 1 { u64::from(ones.dead_count) } else { 0 });
        }
        let pis = program.public_inputs();
        (program, rows, pis)
    }

    #[test]
    fn sigma_floor_below_floor_row_is_rejected_by_the_gate() {
        let (program, rows, pis) = below_floor_trace();
        {
            let v: &ScaleColumnsView<F> = rows[0].borrow();
            assert_eq!(
                to_u64(v.alpha_exp)
                    + to_u64(v.l2_floored_exponent)
                    + to_u64(v.sigma_sig_is_wide)
                    + program.device.sigma_encoding_offset(),
                PROD_UNIT as u64 - 1,
                "test premise: the spike row sits immediately below sigma = 1"
            );
        }
        check_constraints(&program, &rows, &pis).expect("the sigma floor is lookup-only");
        assert!(
            check_lut_lookups(&program, &rows, &pis).is_err(),
            "the sigma-floor RC16 must reject a below-floor encoding"
        );
    }

    #[test]
    #[should_panic(expected = "below the sigma floor")]
    fn generate_trace_rejects_sigma_floor_violations() {
        let program = ScaleProgram::new_for_device(1, 1, PublicParams::MAX_K, 4, Device::B200);
        let spike = find_boundary_spike(&program, false);
        let ones = ScaleRowTuple::from_prequant_row_for_device(
            &vec![1i8; program.k],
            &vec![0x3F80u16; program.k / BLOCK_SIZE],
            program.wl2(),
            Device::B200,
        );
        let _ = program.generate_trace::<F>(&[spike], &[ones]);
    }

    #[test]
    fn sigma_floor_gate_matches_native_policy_comparison() {
        // The gate's affine encoding vs the plaintext jackpot comparison
        // `delta * alpha * l2f >= sigma_min` in f64 (exact: 16-bit significand products,
        // exponents far from f64's limits) — exhaustive around both devices' pivots and
        // fuzzed across the full pinned domain.
        let value = |m: u64, e_star: i64| m as f64 * ((e_star - 134) as f64).exp2();
        for device in [Device::H100, Device::B200] {
            let policy = JackpotPolicy::for_device(device);
            let check = |alpha_exp: i64, alpha_mantissa: u64, l2f_e_star: i64, l2f_m: u64| {
                let alpha_m = 128 + alpha_mantissa;
                let e_sum = alpha_exp + l2f_e_star;
                let sig = alpha_m * l2f_m;
                let wide = i64::from(sig >= 1 << 15);
                let gate = e_sum + wide + device.sigma_encoding_offset() as i64 >= PROD_UNIT;
                let sigma = device.delta() * value(alpha_m, alpha_exp) * value(l2f_m, l2f_e_star);
                assert_eq!(
                    gate,
                    sigma >= policy.sigma_min,
                    "{device:?} gate vs plaintext at alpha = ({alpha_exp}, {alpha_mantissa}), \
                     l2f = ({l2f_e_star}, {l2f_m})"
                );
            };
            for e_sum in 252..=256i64 {
                for l2f_e_star in [95i64, 200] {
                    for alpha_mantissa in 0..128u64 {
                        for l2f_mantissa in 0..128u64 {
                            check(e_sum - l2f_e_star, alpha_mantissa, l2f_e_star, 128 + l2f_mantissa);
                        }
                    }
                }
            }
            let mut x: u64 = 0x243F6A8885A308D3;
            let mut step = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            for _ in 0..100_000 {
                let r = step();
                // alpha structurally normal: exp in [1, 254]; l2f >= 2^-32: E* in [95, 254].
                let alpha_exp = 1 + (r & 0xFF) as i64 % 254;
                let alpha_mantissa = (r >> 8) & 0x7F;
                let l2f_e_star = 95 + ((r >> 15) & 0xFF) as i64 % 160;
                let l2f_mantissa = (r >> 23) & 0x7F;
                check(alpha_exp, alpha_mantissa, l2f_e_star, 128 + l2f_mantissa);
            }
        }
    }

    /// The geometry public inputs are load-bearing: perturbing the `k` or `Wl2` slot away
    /// from the trace's values breaks the Q5/Q7 bracket identities on live nonzero rows.
    #[test]
    fn tamper_geometry_public_inputs_is_rejected() {
        let (program, rows, pis) = test_trace();
        check_constraints(&program, &rows, &pis).unwrap();

        let mut bad_k = pis;
        bad_k[K_PUBLIC_INPUT] += F::from_canonical_u64(32);
        assert!(
            check_constraints(&program, &rows, &bad_k).is_err(),
            "a shifted k public input must break the Q7 claim products"
        );

        let mut bad_wl2 = pis;
        bad_wl2[WL2_PUBLIC_INPUT] += F::ONE;
        assert!(
            check_constraints(&program, &rows, &bad_wl2).is_err(),
            "a shifted Wl2 public input must break the Q5 alignment window"
        );
    }
}
