//! Proves the FP16 per-row norm + scale derivation (see [`super`] module docs).
//!
//! # Constraint groups (see the per-group comments in [`eval_row_scale_constraints`])
//!
//! * **D / M** — per-element FP16 decode (FP16DECODE) and the running maximum magnitude code.
//! * **Q** — the exact integer square `SQ_SIG = X_SIG^2`, normalized to a 24-bit significand.
//! * **A** — the sequential f32 RNE accumulation `acc_out = RNE_f32(acc_in + sq)` (a windowed,
//!   guard+sticky fold mirroring [`crate::v5::circuit::noisy_quant_fma_stark`], here both terms
//!   nonnegative), chained row-to-row; `sumsq` is pinned to the last row's `acc_out`.
//! * **V** — `q = RNE_f32(sumsq / k)` (a divide-by-constant bracket).
//! * **S** — `s = RNE_f32(sqrt(q))` (a squared quarter-ulp bracket).
//! * **C / G / F** — `l2raw = f32_to_bf16(s)`, `l2grid = round_l2_to_grid(l2raw)`,
//!   `l2 = bf16_max(l2grid, floor)`.
//! * **L** — `linf = bf16_max(f32_to_bf16(max|x|), floor)`.
//! * **H1/H3/H4/H5** — the scale chain `noised_bound = bf16_fma(dr, l2, linf)`,
//!   `alpha = bf16_div(2^16, noised_bound)`, `m1 = bf16_mul(alpha, l2)`, `beta = bf16_mul(m1, dos)`.
//!
//! # Soundness envelope (documented, not an unproved shortcut)
//!
//! `r` is the fixed noise rank 32, so `dr`/`dos` are compile-time constants. The row is in scope
//! when every intermediate stays f32-normal (no subnormal f32 sum/quotient/root — the squares are
//! `>= 2^-48` and `k <= 2^16`, so a nonzero row keeps `sumsq`, `q`, `s` normal) and the final
//! `l2`/`alpha` do not snap into the bf16 infinity code. An out-of-scope row is simply unprovable
//! (the generator asserts the envelope), exactly as [`crate::v5::circuit::noise_stark`] treats
//! its rejected boundaries. Every ties-to-even rounding carries the full machinery (parity split +
//! binade-bottom `IS_BOTTOM` where the binade is asymmetric) with NO quarter-ulp grinding slack.

use core::borrow::{Borrow, BorrowMut};
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

use super::columns::{NUM_ROW_SCALE_COLUMNS, NUM_ROW_SCALE_PUBLIC_INPUTS, ROW_SCALE_COL_MAP, RowScaleColumnsView};
use crate::v5::api::dtype::{fp16_decode_fields, fp16_to_f32};
use crate::v5::api::quantization::{DELTA, MAX_FP16, NORM_FLOOR};
use crate::v4::api::compute::{bf16_div, bf16_fma, bf16_mul};
use crate::v4::api::dtype::f32_to_bf16;
use crate::v4::api::prequant::round_l2_to_grid;
use crate::v4::api::quantization::NOISE_TARGET_NORM;
use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// bf16 unit offset: a finite bf16 is `M * 2^(E* - 134)` (`134 = bias 127 + 7 fraction bits`).
const BF16_UNIT: u64 = 134;
/// `acc_out_exp = eta + ww + carry + ACC_EXP_ADD` (`= 100 - 64`, the `64` being the fold's biased
/// value-MSB offset, which keeps every term's MSB nonnegative since the smallest nonzero square is
/// `2^-48`, so the unbiased MSB bottoms out at `-48`).
const ACC_EXP_ADD: u64 = 36;
/// A term is "far" (contributes only to the sticky bit) when its relative shift reaches this (the
/// 24-bit significand plus 3 guard bits).
const FAR_CAP: u64 = 27;
/// Division-bracket key base for `alpha = RNE(2^16 / nb)`: `2 + 16 + 2*134` (numerator `2^16`).
pub(crate) const DIV_KEY_BASE: u64 = 2 + 16 + 2 * BF16_UNIT;
/// bf16 code of the norm floor `NORM_FLOOR = 2^-32` (exp field 95, mantissa 0).
pub(crate) const FLOOR_CODE: u64 = 0x2F80;

fn inv_f<F: RichField>(x: u64) -> F {
    if x == 0 { F::ZERO } else { F::from_canonical_u64(x).inverse() }
}

fn inv_diff<F: RichField>(a: u64, b: u64) -> F {
    let d = F::from_canonical_u64(a) - F::from_canonical_u64(b);
    if d == F::ZERO { F::ZERO } else { d.inverse() }
}

fn bit_length(x: u64) -> u64 {
    64 - u64::from(x.leading_zeros())
}

/// `x >> shift` rounded to nearest, ties to even.
fn round_shift(x: u64, shift: u64) -> u64 {
    if shift == 0 {
        return x;
    }
    let dropped = x & ((1u64 << shift) - 1);
    let kept = x >> shift;
    let half = 1u64 << (shift - 1);
    if dropped > half || (dropped == half && (kept & 1) == 1) { kept + 1 } else { kept }
}

/// Nonnegative f32 fields `(sig24, biased_exp, is_zero)`: `value = sig24 * 2^(exp - 150)`,
/// `sig24 in [2^23, 2^24)` for normals, `(0, 0, true)` for zero. Panics on subnormal/non-finite
/// (out of the documented envelope).
fn f32_fields(x: f32) -> (u64, u64, bool) {
    assert!(x.is_finite() && x >= 0.0, "f32 field extraction expects a finite nonnegative value");
    if x == 0.0 {
        return (0, 0, true);
    }
    let bits = x.to_bits();
    let exp = u64::from((bits >> 23) & 0xFF);
    assert!(exp != 0, "envelope: f32 intermediate must be normal (not subnormal)");
    ((1u64 << 23) | u64::from(bits & 0x7F_FFFF), exp, false)
}

/// The committed geometry: the number of operand rows (A rows then B rows), the row length `k`, the
/// fixed noise rank `r`, and the trace height (a power of two covering `num_operand_rows * k`).
#[derive(Clone, Debug)]
pub struct RowScaleProgram {
    /// Number of operand rows proved end to end (`h + w` in the batch; `1` standalone). Each is a
    /// self-contained `k`-element block.
    pub num_operand_rows: usize,
    /// Number of leading A-side operand rows (`h`); the remaining `num_operand_rows - num_a_rows`
    /// are B-side (`w`). An AIR compile-time constant (like `k`) — it fixes the per-side liveness
    /// thresholds `num_a_rows*k` / `num_b_rows*k`, so it needs NO known column / no public input.
    pub num_a_rows: usize,
    /// Number of FP16 values in each operand row.
    pub k: usize,
    /// Noise rank (fixed at 32; `dr`/`dos` are then compile-time constants).
    pub r: usize,
    /// Trace height (a power of two `>= num_operand_rows * k`).
    pub num_rows: usize,
}

impl RowScaleProgram {
    /// Builds a single-operand-row program padded to the next power of two (standalone / unit-test
    /// path). The batch uses [`Self::with_rows`] to pin an on-ladder height.
    pub fn new(k: usize, r: usize) -> Self {
        Self::with_rows(1, k, r, (k).next_power_of_two().max(2))
    }

    /// Builds the program for `num_operand_rows` rows of `k` elements at an explicit `num_rows`
    /// height, asserting the compile-time facts the AIR bakes in: `dr`/`dos` are the exact bf16
    /// constants of `DELTA*sqrt(r)` and `DELTA*sqrt(r)/N^2`, `MAX_FP16` rounds to the power of two
    /// `2^16` in bf16, and `NORM_FLOOR` is `2^-32`.
    pub fn with_rows(num_operand_rows: usize, k: usize, r: usize, num_rows: usize) -> Self {
        assert!(k >= 1, "a row has at least one element");
        assert!(num_operand_rows >= 1, "at least one operand row");
        assert!(num_rows.is_power_of_two(), "trace height must be a power of two");
        assert!(num_rows >= num_operand_rows * k, "trace height must cover every live block");
        assert_eq!(f32_to_bf16(MAX_FP16).expect("65504 finite"), 0x4780, "MAX_FP16 rounds to 2^16 in bf16");
        assert_eq!(f32_to_bf16(NORM_FLOOR).expect("2^-32 finite"), FLOOR_CODE as u16, "NORM_FLOOR = 2^-32");
        // dr/dos are structurally normal bf16 constants (used by the scale chain).
        let dr = Self::dr_code_for(r);
        let dos = Self::dos_code_for(r);
        assert!((1..=254).contains(&(dr >> 7)) && (1..=254).contains(&(dos >> 7)), "dr/dos are normal bf16");
        // Standalone / single-side path: every row is A-side (the B gate is then vacuous, `w = 0`).
        Self { num_operand_rows, num_a_rows: num_operand_rows, k, r, num_rows }
    }

    /// The batch path: `num_a_rows` A rows (`h`) followed by `num_b_rows` B rows (`w`), laid out A
    /// then B exactly as the operand commitment / noisy-quant expect. Fixes both per-side liveness
    /// thresholds.
    pub fn with_rows_ab(num_a_rows: usize, num_b_rows: usize, k: usize, r: usize, num_rows: usize) -> Self {
        let mut p = Self::with_rows(num_a_rows + num_b_rows, k, r, num_rows);
        p.num_a_rows = num_a_rows;
        p
    }

    fn dr_code_for(r: usize) -> u16 {
        f32_to_bf16((DELTA * (r as f64).sqrt()) as f32).expect("dr finite")
    }

    fn dos_code_for(r: usize) -> u16 {
        f32_to_bf16((DELTA * (r as f64).sqrt() / (NOISE_TARGET_NORM * NOISE_TARGET_NORM)) as f32).expect("dos finite")
    }

    pub fn dr_code(&self) -> u16 {
        Self::dr_code_for(self.r)
    }

    pub fn dos_code(&self) -> u16 {
        Self::dos_code_for(self.r)
    }

    pub fn live_rows(&self) -> usize {
        self.num_operand_rows * self.k
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The live-row -> operand-row index map, `is_block_start`, and `is_block_final` as closures over
    /// a trace row (a pure function of the geometry; the trailing padding block starts at the first
    /// padding row).
    fn block_schedule(&self, i: usize) -> (bool, usize, bool, bool) {
        let live = self.live_rows();
        let is_pad = i >= live;
        let operand_row_index = if is_pad { self.num_operand_rows - 1 } else { i / self.k };
        // Block starts: the first row of each live k-block, plus the first padding row (one trailing
        // padding block).
        let is_block_start = (!is_pad && i % self.k == 0) || (is_pad && i == live);
        let is_block_final = !is_pad && i % self.k == self.k - 1;
        (is_pad, operand_row_index, is_block_start, is_block_final)
    }

    /// The class (a) ("known") columns (`IS_PAD`, `OPERAND_ROW_INDEX`, `IS_BLOCK_START`,
    /// `IS_BLOCK_FINAL`), pure functions of the geometry.
    pub fn known_values<F: RichField>(&self) -> Vec<PolynomialValues<F>> {
        let num_rows = self.num_rows();
        let mut is_pad = Vec::with_capacity(num_rows);
        let mut ori = Vec::with_capacity(num_rows);
        let mut bstart = Vec::with_capacity(num_rows);
        let mut bfinal = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let (p, o, bs, bf) = self.block_schedule(i);
            is_pad.push(F::from_bool(p));
            ori.push(F::from_canonical_usize(o));
            bstart.push(F::from_bool(bs));
            bfinal.push(F::from_bool(bf));
        }
        vec![
            PolynomialValues::new(is_pad),
            PolynomialValues::new(ori),
            PolynomialValues::new(bstart),
            PolynomialValues::new(bfinal),
        ]
    }

    /// Generates the trace from `num_operand_rows * k` raw FP16 operand codes (A rows then B rows),
    /// bit-exact with the reference per-row `row_norms` + the floor + `derive_row_scales`.
    pub fn generate_trace<F: RichField>(&self, rows_codes: &[u16]) -> Vec<[F; NUM_ROW_SCALE_COLUMNS]> {
        assert_eq!(rows_codes.len(), self.live_rows(), "one code per live block element");
        let num_rows = self.num_rows();
        let live = self.live_rows();

        // ---- Pass 1: the sequential f32 sum of squares and running max, PER BLOCK. The fold resets
        // at every block start (`is_block_start`); padding rows carry code 0 and form one trailing
        // block whose (unused) scalars are the all-zero-row floor case. `sumsq` pins to each live
        // block's final row via `is_block_final`. ----
        let mut acc: f32 = 0.0;
        let mut run_max: u16 = 0;
        let mut per_elem: Vec<RowScaleColumnsView<F>> = Vec::with_capacity(num_rows);
        // Each block's (start_row, end_row_inclusive, sumsq, max_abs_code); scalars are filled from
        // these and overlaid onto every row of the block.
        let mut blocks: Vec<(usize, usize, f32, u64)> = Vec::new();
        let mut block_start_idx = 0usize;
        for i in 0..num_rows {
            let (is_pad, operand_row_index, is_block_start, is_block_final) = self.block_schedule(i);
            if is_block_start {
                acc = 0.0;
                run_max = 0;
                block_start_idx = i;
            }
            let code = if is_pad { 0 } else { rows_codes[i] };
            let mut v = RowScaleColumnsView::<F>::default();
            v.is_pad = F::from_bool(is_pad);
            v.operand_row_index = F::from_canonical_usize(operand_row_index);
            v.is_block_start = F::from_bool(is_block_start);
            v.is_block_final = F::from_bool(is_block_final);
            v.code = F::from_canonical_u16(code);
            let _ = live;
            let (x_sig, x_sign, x_eps_biased, x_is_zero) = fp16_decode_fields(code);
            v.x_sig = F::from_canonical_u64(x_sig);
            v.x_sign = F::from_canonical_u64(x_sign);
            v.x_eps_biased = F::from_canonical_u64(x_eps_biased);
            v.x_is_zero = F::from_canonical_u64(x_is_zero);
            let abs_code = u64::from(code & 0x7FFF);
            v.abs_code = F::from_canonical_u64(abs_code);

            // Running max of the magnitude code.
            let prev = u64::from(run_max);
            let ge = abs_code >= prev;
            v.max_ge = F::from_bool(ge);
            v.max_slack = F::from_canonical_u64(if ge { abs_code - prev } else { prev - abs_code - 1 });
            if ge {
                run_max = (abs_code) as u16;
            }
            v.run_max = F::from_canonical_u64(u64::from(run_max));

            // Exact square significand and its 24-bit normalization.
            let sq_sig = x_sig * x_sig;
            v.sq_sig = F::from_canonical_u64(sq_sig);
            let (sq_w, sq_lift, sq_norm) = if sq_sig == 0 {
                (0u64, 0u64, 0u64)
            } else {
                let w = bit_length(sq_sig);
                let lift = 1u64 << (24 - w);
                (w, lift, sq_sig * lift)
            };
            v.sq_w = F::from_canonical_u64(sq_w);
            v.sq_lift = F::from_canonical_u64(sq_lift);
            v.sq_norm = F::from_canonical_u64(sq_norm);
            v.sq_norm_lo = F::from_canonical_u64(sq_norm & 0xFFFF);
            v.sq_norm_hi = F::from_canonical_u64(sq_norm >> 16);

            // acc_in fields.
            let (acc_in_sig, acc_in_exp, acc_in_zero) = f32_fields(acc);
            v.acc_in_sig = F::from_canonical_u64(acc_in_sig);
            v.acc_in_sig_lo = F::from_canonical_u64(acc_in_sig & 0xFFFF);
            v.acc_in_sig_hi = F::from_canonical_u64(acc_in_sig >> 16);
            v.acc_in_exp = F::from_canonical_u64(acc_in_exp);
            v.acc_in_zero = F::from_bool(acc_in_zero);

            // ---- fold: acc_out = RNE_f32(acc_in + sq). Both terms nonnegative. ----
            let sq_zero = x_is_zero == 1;
            // Biased value-MSBs: msb_b = exp_field - 127 + ACC_MSB_BIAS = exp_field - 63.
            // sq's exp field (if nonzero) = 2*x_eps_biased + sq_w + 76.
            let sq_exp_field = 2 * x_eps_biased + sq_w + 76;
            let sq_msb = if sq_zero { 0 } else { sq_exp_field - 63 };
            let acc_msb = if acc_in_zero { 0 } else { acc_in_exp - 63 };
            v.sq_msb = F::from_canonical_u64(sq_msb);
            v.acc_msb = F::from_canonical_u64(acc_msb);
            let eta = sq_msb.max(acc_msb);
            v.eta = F::from_canonical_u64(eta);
            // rel = eta - msb (msb is already 0 for a zero term, so its rel becomes eta; the term is
            // still excluded via `active` below — this matches the eval constraint `rel = eta - msb`).
            let rel_sq = eta - sq_msb;
            let rel_acc = eta - acc_msb;
            v.rel_sq = F::from_canonical_u64(rel_sq);
            v.rel_acc = F::from_canonical_u64(rel_acc);
            let far_sq = rel_sq >= FAR_CAP;
            let far_acc = rel_acc >= FAR_CAP;
            v.far_sq = F::from_bool(far_sq);
            v.far_sq_slack = F::from_canonical_u64(if far_sq { rel_sq - FAR_CAP } else { FAR_CAP - 1 - rel_sq });
            v.far_acc = F::from_bool(far_acc);
            v.far_acc_slack = F::from_canonical_u64(if far_acc { rel_acc - FAR_CAP } else { FAR_CAP - 1 - rel_acc });
            let active_sq = !sq_zero && !far_sq;
            let active_acc = !acc_in_zero && !far_acc;
            v.active_sq = F::from_bool(active_sq);
            v.active_acc = F::from_bool(active_acc);

            let fill_align = |sig: u64, rel: u64, active: bool| -> (u64, u64, u64, u64) {
                if active {
                    let pow = 1u64 << rel;
                    let a = (sig * 8) / pow;
                    let rem = sig * 8 - a * pow;
                    (pow, a, rem, pow - 1 - rem)
                } else {
                    (1, 0, 0, 0)
                }
            };
            let (pow_sq, aligned_sq, rem_sq, rem_sq_bound) = fill_align(sq_norm, rel_sq, active_sq);
            v.pow_sq = F::from_canonical_u64(pow_sq);
            v.aligned_sq = F::from_canonical_u64(aligned_sq);
            v.aligned_sq_lo = F::from_canonical_u64(aligned_sq & 0xFFFF);
            v.aligned_sq_hi = F::from_canonical_u64(aligned_sq >> 16);
            v.rem_sq = F::from_canonical_u64(rem_sq);
            v.rem_sq_lo = F::from_canonical_u64(rem_sq & 0xFFFF);
            v.rem_sq_hi = F::from_canonical_u64(rem_sq >> 16);
            v.rem_sq_bound = F::from_canonical_u64(rem_sq_bound);
            v.rem_sq_bound_lo = F::from_canonical_u64(rem_sq_bound & 0xFFFF);
            v.rem_sq_bound_hi = F::from_canonical_u64(rem_sq_bound >> 16);
            let (pow_acc, aligned_acc, rem_acc, rem_acc_bound) = fill_align(acc_in_sig, rel_acc, active_acc);
            v.pow_acc = F::from_canonical_u64(pow_acc);
            v.aligned_acc = F::from_canonical_u64(aligned_acc);
            v.aligned_acc_lo = F::from_canonical_u64(aligned_acc & 0xFFFF);
            v.aligned_acc_hi = F::from_canonical_u64(aligned_acc >> 16);
            v.rem_acc = F::from_canonical_u64(rem_acc);
            v.rem_acc_lo = F::from_canonical_u64(rem_acc & 0xFFFF);
            v.rem_acc_hi = F::from_canonical_u64(rem_acc >> 16);
            v.rem_acc_bound = F::from_canonical_u64(rem_acc_bound);
            v.rem_acc_bound_lo = F::from_canonical_u64(rem_acc_bound & 0xFFFF);
            v.rem_acc_bound_hi = F::from_canonical_u64(rem_acc_bound >> 16);

            let rem_sq_nz = rem_sq != 0;
            let rem_acc_nz = rem_acc != 0;
            v.rem_sq_nz = F::from_bool(rem_sq_nz);
            v.rem_sq_nz_inv = inv_f::<F>(rem_sq);
            v.rem_acc_nz = F::from_bool(rem_acc_nz);
            v.rem_acc_nz_inv = inv_f::<F>(rem_acc);
            let or_sq = far_sq || rem_sq_nz;
            let or_acc = far_acc || rem_acc_nz;
            v.or_sq = F::from_bool(or_sq);
            v.or_acc = F::from_bool(or_acc);
            let sticky_sq = !sq_zero && or_sq;
            let sticky_acc = !acc_in_zero && or_acc;
            v.sticky_sq = F::from_bool(sticky_sq);
            v.sticky_acc = F::from_bool(sticky_acc);
            let far_sticky = sticky_sq || sticky_acc;
            v.far_sticky = F::from_bool(far_sticky);

            let w = aligned_sq + aligned_acc;
            v.w_abs = F::from_canonical_u64(w);
            v.w_abs_lo = F::from_canonical_u64(w & 0xFFFF);
            v.w_abs_hi = F::from_canonical_u64(w >> 16);
            let w_is_zero = w == 0;
            v.w_is_zero = F::from_bool(w_is_zero);
            v.w_abs_inv = inv_f::<F>(w);

            let (ww, trunc_w, lift_w, m_rz, rz_rem) = if w_is_zero {
                (0, 1, 1, 0, 0)
            } else {
                let ww = bit_length(w);
                let tr = 1u64 << ww.saturating_sub(24);
                let lf = 1u64 << 24u64.saturating_sub(ww);
                let m = w * lf / tr;
                (ww, tr, lf, m, w * lf - m * tr)
            };
            v.ww = F::from_canonical_u64(ww);
            v.trunc_w = F::from_canonical_u64(trunc_w);
            v.lift_w = F::from_canonical_u64(lift_w);
            v.m_rz = F::from_canonical_u64(m_rz);
            v.m_rz_lo = F::from_canonical_u64(m_rz & 0xFFFF);
            v.m_rz_hi = F::from_canonical_u64(m_rz >> 16);
            v.rz_rem = F::from_canonical_u64(rz_rem);
            v.rz_rem_bound = F::from_canonical_u64(if w_is_zero { 0 } else { trunc_w - 1 - rz_rem });
            let rz_parity = m_rz & 1;
            v.rz_parity = F::from_canonical_u64(rz_parity);
            v.rz_half = F::from_canonical_u64(((m_rz & 0xFFFF) - rz_parity) / 2);
            let gt = 2 * rz_rem > trunc_w;
            v.gt = F::from_bool(gt);
            v.gt_slack = F::from_canonical_u64(if gt { 2 * rz_rem - trunc_w - 1 } else { trunc_w - 2 * rz_rem });
            let eq = 2 * rz_rem == trunc_w;
            v.eq = F::from_bool(eq);
            v.eq_inv = inv_diff::<F>(trunc_w, 2 * rz_rem);
            let or_rs = far_sticky || rz_parity == 1;
            v.or_rs = F::from_bool(or_rs);
            let round_up = gt || (eq && or_rs);
            v.round_up = F::from_bool(round_up);
            let m_out = m_rz + u64::from(round_up);
            let carry = m_out == (1 << 24);
            v.carry = F::from_bool(carry);
            v.carry_inv = inv_diff::<F>(1 << 24, m_out);
            let m_out_final = if carry { 1 << 23 } else { m_out };

            // acc_out.
            let (acc_out_sig, acc_out_exp, acc_out_zero) = if w_is_zero {
                (0, 0, true)
            } else {
                (m_out_final, eta + ww + u64::from(carry) + ACC_EXP_ADD, false)
            };
            v.acc_out_sig = F::from_canonical_u64(acc_out_sig);
            v.acc_out_sig_lo = F::from_canonical_u64(acc_out_sig & 0xFFFF);
            v.acc_out_sig_hi = F::from_canonical_u64(acc_out_sig >> 16);
            v.acc_out_exp = F::from_canonical_u64(acc_out_exp);
            v.acc_out_zero = F::from_bool(acc_out_zero);

            // Advance the real f32 accumulator and cross-check.
            let v_f32 = fp16_to_f32(code);
            let sq_f32 = v_f32 * v_f32;
            let acc_next = acc + sq_f32;
            let (want_sig, want_exp, want_zero) = f32_fields(acc_next);
            debug_assert_eq!(
                (acc_out_sig, acc_out_exp, acc_out_zero),
                (want_sig, want_exp, want_zero),
                "fold diverged from the reference f32 accumulation at code {code:#06x}"
            );
            acc = acc_next;

            per_elem.push(v);

            // A block ends at the last trace row or just before the next block start.
            let block_end = i == num_rows - 1 || self.block_schedule(i + 1).2;
            if block_end {
                blocks.push((block_start_idx, i, acc, u64::from(run_max)));
            }
        }

        // ---- Assemble rows: per-element columns overlaid with each block's (constant) scalars.
        // The per-element entry-liveness `dead` flag is filled here (it needs the block's floored
        // `l2`), plus the per-row `is_b_side` mask. ----
        let mut rows: Vec<[F; NUM_ROW_SCALE_COLUMNS]> = Vec::with_capacity(num_rows);
        rows.resize(num_rows, [F::ZERO; NUM_ROW_SCALE_COLUMNS]);
        for &(start, end, sumsq, max_abs_code) in &blocks {
            let scalars = self.fill_scalars::<F>(sumsq, max_abs_code);
            let l2_e = scalars.l2_e.to_canonical_u64();
            let l2_m = scalars.l2_m.to_canonical_u64();
            for (r, pe) in per_elem[start..=end].iter().enumerate() {
                let mut v = *pe;
                copy_scalars(&mut v, &scalars);
                let i = start + r;
                let (is_pad, operand_row_index, _, _) = self.block_schedule(i);
                self.fill_dead::<F>(&mut v, pe.x_sig.to_canonical_u64(), pe.x_eps_biased.to_canonical_u64(), l2_e, l2_m);
                // Per-row side mask + its two-sided pin slack (gated live in the AIR).
                let b_side = operand_row_index >= self.num_a_rows;
                v.is_b_side = F::from_bool(b_side);
                let h = self.num_a_rows as u64;
                let ori = operand_row_index as u64;
                v.b_side_slack = F::from_canonical_u64(if b_side { ori - h } else { h - 1 - ori });
                let _ = is_pad;
                rows[i] = v.into();
            }
        }

        // ---- Final pass: per-side inclusive dead accumulators + the last-row gate slacks. ----
        let m = &ROW_SCALE_COL_MAP;
        let (mut run_a, mut run_b) = (0u64, 0u64);
        for i in 0..num_rows {
            let (is_pad, operand_row_index, _, _) = self.block_schedule(i);
            let dead = rows[i][m.dead].to_canonical_u64();
            if !is_pad {
                if operand_row_index >= self.num_a_rows {
                    run_b += dead;
                } else {
                    run_a += dead;
                }
            }
            rows[i][m.dead_run_a] = F::from_canonical_u64(run_a);
            rows[i][m.dead_run_b] = F::from_canonical_u64(run_b);
        }
        let k = self.k as u64;
        let w = (self.num_operand_rows - self.num_a_rows) as u64;
        let a_slack = (self.num_a_rows as u64 * k).wrapping_sub(64 * run_a);
        let b_slack = (w * k).wrapping_sub(64 * run_b);
        let last = num_rows - 1;
        rows[last][m.a_gate_slack_lo] = F::from_canonical_u64(a_slack & 0xFFFF);
        rows[last][m.a_gate_slack_hi] = F::from_canonical_u64((a_slack >> 16) & 0xFFFF);
        rows[last][m.b_gate_slack_lo] = F::from_canonical_u64(b_slack & 0xFFFF);
        rows[last][m.b_gate_slack_hi] = F::from_canonical_u64((b_slack >> 16) & 0xFFFF);
        rows
    }

    /// Fills the per-element entry-liveness witnesses (`dead` and its aligned-compare gadget) for
    /// `dead = [ |x| >= 4*l2 ]`, with `|x| = x_sig * 2^(x_eps_biased - 25)` and
    /// `4*l2 = (128 + l2_m) * 2^(l2_e - 132)`. Exact-integer, bit-identical to the f64 compare in
    /// [`crate::v5::api::policy::check_shared_gates`].
    fn fill_dead<F: RichField>(&self, v: &mut RowScaleColumnsView<F>, x_sig: u64, x_eps_biased: u64, l2_e: u64, l2_m: u64) {
        // s = exponent of |x|'s significand minus exponent of 4*l2's significand.
        let s = x_eps_biased as i64 - l2_e as i64 + 107;
        let sign = s >= 0;
        let key = s.unsigned_abs();
        let pow = 1u64 << key.min(26);
        let (pow_a, pow_b) = if sign { (pow, 1) } else { (1, pow) };
        let mb = 128 + l2_m;
        let lhs = x_sig * pow_a;
        let rhs = mb * pow_b;
        let dead = lhs >= rhs;
        let slack = if dead { lhs - rhs } else { rhs - lhs - 1 };
        v.dead = F::from_bool(dead);
        v.dead_sign = F::from_bool(sign);
        v.dead_key = F::from_canonical_u64(key);
        v.dead_pow = F::from_canonical_u64(pow);
        v.dead_pow_a = F::from_canonical_u64(pow_a);
        v.dead_lhs = F::from_canonical_u64(lhs);
        v.dead_rhs = F::from_canonical_u64(rhs);
        v.dead_slack_lo = F::from_canonical_u64(slack & 0xFFFF);
        v.dead_slack_mid = F::from_canonical_u64((slack >> 16) & 0xFFFF);
        v.dead_slack_hi = F::from_canonical_u64(slack >> 32);
    }

    /// Fills the scalar (whole-row) columns from the finished `sumsq` (f32) and `max_abs_code`.
    fn fill_scalars<F: RichField>(&self, sumsq: f32, max_abs_code: u64) -> RowScaleColumnsView<F> {
        let mut v = RowScaleColumnsView::<F>::default();
        let k = self.k as u64;

        let (sumsq_sig, sumsq_exp, sumsq_zero) = f32_fields(sumsq);
        v.sumsq_sig = F::from_canonical_u64(sumsq_sig);
        v.sumsq_sig_lo = F::from_canonical_u64(sumsq_sig & 0xFFFF);
        v.sumsq_sig_hi = F::from_canonical_u64(sumsq_sig >> 16);
        v.sumsq_exp = F::from_canonical_u64(sumsq_exp);
        v.sumsq_zero = F::from_bool(sumsq_zero);
        v.max_abs_code = F::from_canonical_u64(max_abs_code);

        // ---- q = RNE_f32(sumsq / k). ----
        let q = sumsq / self.k as f32;
        let (q_sig, q_exp, q_zero) = f32_fields(q);
        v.q_sig = F::from_canonical_u64(q_sig);
        v.q_sig_lo = F::from_canonical_u64(q_sig & 0xFFFF);
        v.q_sig_hi = F::from_canonical_u64(q_sig >> 16);
        v.q_exp = F::from_canonical_u64(q_exp);
        v.q_zero = F::from_bool(q_zero);
        if !q_zero {
            let dexp = sumsq_exp - q_exp; // >= 0 (dividing by k >= 1 lowers the exponent)
            assert!((0..=26).contains(&dexp), "envelope: division shift out of FP16POW2 domain ({dexp})");
            v.q_dexp = F::from_canonical_u64(dexp);
            v.q_pow = F::from_canonical_u64(1u64 << dexp);
            let bottom = u64::from(q_sig == (1 << 23));
            let parity = q_sig & 1;
            v.q_bottom = F::from_canonical_u64(bottom);
            v.q_bottom_inv = inv_diff::<F>(q_sig, 1 << 23);
            v.q_parity = F::from_canonical_u64(parity);
            v.q_half = F::from_canonical_u64(((q_sig & 0xFFFF) - parity) / 2);
            let blo = (4 * q_sig - 2 + bottom) * k;
            let bhi = (4 * q_sig + 2) * k;
            let target = 4 * sumsq_sig * (1u64 << dexp);
            v.q_blo = F::from_canonical_u64(blo);
            v.q_bhi = F::from_canonical_u64(bhi);
            let sl = target - blo - parity;
            let su = bhi - target - parity;
            v.q_sl_lo = F::from_canonical_u64(sl & 0xFFFF);
            v.q_sl_mid = F::from_canonical_u64((sl >> 16) & 0xFFFF);
            v.q_sl_hi = F::from_canonical_u64(sl >> 32);
            v.q_su_lo = F::from_canonical_u64(su & 0xFFFF);
            v.q_su_mid = F::from_canonical_u64((su >> 16) & 0xFFFF);
            v.q_su_hi = F::from_canonical_u64(su >> 32);
        }

        // ---- s = RNE_f32(sqrt(q)). ----
        let s = q.sqrt();
        let (s_sig, s_exp, _s_zero) = f32_fields(s);
        v.s_sig = F::from_canonical_u64(s_sig);
        v.s_sig_lo = F::from_canonical_u64(s_sig & 0xFFFF);
        v.s_sig_hi = F::from_canonical_u64(s_sig >> 16);
        v.s_exp = F::from_canonical_u64(s_exp);
        if !q_zero {
            let sdexp = q_exp + 152 - 2 * s_exp; // Ds - 2, in [25, 26]
            assert!((0..=26).contains(&sdexp), "envelope: sqrt shift out of FP16POW2 domain ({sdexp})");
            v.s_dexp = F::from_canonical_u64(sdexp);
            v.s_pow = F::from_canonical_u64(1u64 << sdexp);
            let bottom = u64::from(s_sig == (1 << 23));
            let parity = s_sig & 1;
            v.s_bottom = F::from_canonical_u64(bottom);
            v.s_bottom_inv = inv_diff::<F>(s_sig, 1 << 23);
            v.s_parity = F::from_canonical_u64(parity);
            v.s_half = F::from_canonical_u64(((s_sig & 0xFFFF) - parity) / 2);
            let blo = (4 * s_sig - 2 + bottom) * (4 * s_sig - 2 + bottom);
            let bhi = (4 * s_sig + 2) * (4 * s_sig + 2);
            let target = 4 * q_sig * (1u64 << sdexp);
            v.s_blo = F::from_canonical_u64(blo);
            v.s_bhi = F::from_canonical_u64(bhi);
            let sl = target - blo - parity;
            let su = bhi - target - parity;
            v.s_sl_lo = F::from_canonical_u64(sl & 0xFFFF);
            v.s_sl_mid = F::from_canonical_u64((sl >> 16) & 0xFFFF);
            v.s_sl_hi = F::from_canonical_u64(sl >> 32);
            v.s_su_lo = F::from_canonical_u64(su & 0xFFFF);
            v.s_su_mid = F::from_canonical_u64((su >> 16) & 0xFFFF);
            v.s_su_hi = F::from_canonical_u64(su >> 32);
        }

        // ---- l2raw = f32_to_bf16(s): round the 24-bit significand to 8 bits (shift 16). ----
        let l2raw_code: u64 = if q_zero {
            0
        } else {
            fill_round24to8::<F>(
                s_sig, s_exp, &mut v.l2raw_mant, &mut v.l2raw_exp, &mut v.l2raw_carry, &mut v.l2raw_bottom,
                &mut v.l2raw_bottom_inv, &mut v.l2raw_parity, &mut v.l2raw_half, &mut v.l2raw_blo, &mut v.l2raw_bhi,
                &mut v.l2raw_sl_lo, &mut v.l2raw_sl_hi, &mut v.l2raw_su_lo, &mut v.l2raw_su_hi,
            )
        };
        v.l2raw_code = F::from_canonical_u64(l2raw_code);
        debug_assert_eq!(
            l2raw_code as u16,
            if sumsq == 0.0 { 0 } else { f32_to_bf16((sumsq / self.k as f32).sqrt()).expect("l2raw finite") },
            "l2raw diverged from f32_to_bf16(sqrt(sumsq/k))"
        );

        // ---- l2grid = round_l2_to_grid(l2raw). ----
        let grid_in = l2raw_code + 2;
        let grid_q = grid_in >> 2;
        let grid_r = grid_in - 4 * grid_q;
        let l2grid_code = 4 * grid_q;
        v.grid_q = F::from_canonical_u64(grid_q);
        v.grid_r = F::from_canonical_u64(grid_r);
        v.grid_r_b0 = F::from_canonical_u64(grid_r & 1);
        v.grid_r_b1 = F::from_canonical_u64((grid_r >> 1) & 1);
        v.l2grid_code = F::from_canonical_u64(l2grid_code);
        debug_assert_eq!(l2grid_code as u16, round_l2_to_grid(l2raw_code as u16), "grid diverged");

        // ---- l2 = bf16_max(l2grid, floor). ----
        let l2_ge = l2grid_code >= FLOOR_CODE;
        v.l2_ge = F::from_bool(l2_ge);
        v.l2_floor_slack =
            F::from_canonical_u64(if l2_ge { l2grid_code - FLOOR_CODE } else { FLOOR_CODE - l2grid_code - 1 });
        let l2_code = if l2_ge { l2grid_code } else { FLOOR_CODE };
        v.l2_code = F::from_canonical_u64(l2_code);
        v.l2_e = F::from_canonical_u64(l2_code >> 7);
        v.l2_m = F::from_canonical_u64(l2_code & 0x7F);

        // ---- linf = bf16_max(f32_to_bf16(max|x|), floor). ----
        let (max_sig, _max_sign, max_eps_biased, max_is_zero) = fp16_decode_fields(max_abs_code as u16);
        v.max_sig = F::from_canonical_u64(max_sig);
        v.max_eps_biased = F::from_canonical_u64(max_eps_biased);
        v.max_is_zero = F::from_canonical_u64(max_is_zero);
        let linf_raw_code: u64 = if max_is_zero == 1 {
            0
        } else {
            let max_w = bit_length(max_sig);
            let max_lift = 1u64 << (24 - max_w);
            let max_norm = max_sig * max_lift;
            v.max_w = F::from_canonical_u64(max_w);
            v.max_lift = F::from_canonical_u64(max_lift);
            v.max_norm = F::from_canonical_u64(max_norm);
            v.max_norm_lo = F::from_canonical_u64(max_norm & 0xFFFF);
            v.max_norm_hi = F::from_canonical_u64(max_norm >> 16);
            // base exp for round24to8: linf_raw_exp = max_eps_biased + max_w + 101 (+carry).
            let base_exp = max_eps_biased + max_w + 101;
            fill_round24to8::<F>(
                max_norm, base_exp, &mut v.linf_raw_mant, &mut v.linf_raw_exp, &mut v.linf_raw_carry,
                &mut v.linf_raw_bottom, &mut v.linf_raw_bottom_inv, &mut v.linf_raw_parity, &mut v.linf_raw_half,
                &mut v.linf_raw_blo, &mut v.linf_raw_bhi, &mut v.linf_raw_sl_lo, &mut v.linf_raw_sl_hi,
                &mut v.linf_raw_su_lo, &mut v.linf_raw_su_hi,
            )
        };
        v.linf_raw_code = F::from_canonical_u64(linf_raw_code);
        debug_assert_eq!(
            linf_raw_code as u16,
            f32_to_bf16(fp16_to_f32(max_abs_code as u16)).expect("linf_raw finite"),
            "linf_raw diverged from f32_to_bf16(max|x|)"
        );
        let linf_ge = linf_raw_code >= FLOOR_CODE;
        v.linf_ge = F::from_bool(linf_ge);
        v.linf_floor_slack =
            F::from_canonical_u64(if linf_ge { linf_raw_code - FLOOR_CODE } else { FLOOR_CODE - linf_raw_code - 1 });
        let linf_code = if linf_ge { linf_raw_code } else { FLOOR_CODE };
        v.linf_code = F::from_canonical_u64(linf_code);
        v.linf_e = F::from_canonical_u64(linf_code >> 7);
        v.linf_m = F::from_canonical_u64(linf_code & 0x7F);

        // ---- noised_bound = bf16_fma(dr, l2, linf). ----
        let dr = u64::from(self.dr_code());
        let (dr_m, dr_e) = (128 + (dr & 0x7F), dr >> 7);
        let (l2_m, l2_e) = (128 + (l2_code & 0x7F), l2_code >> 7);
        let (linf_m, linf_e) = (128 + (linf_code & 0x7F), linf_code >> 7);
        let nb_prod = dr_m * l2_m;
        v.nb_prod = F::from_canonical_u64(nb_prod);
        // linf is the coarser term: shift = E(linf) - E(l2) + 6 >= 0.
        assert!(linf_e + 6 >= l2_e, "envelope: noised-bound alignment underflow");
        let nb_shift = linf_e + 6 - l2_e;
        assert!(nb_shift <= 26, "envelope: noised-bound shift out of FP16POW2 domain ({nb_shift})");
        v.nb_shift = F::from_canonical_u64(nb_shift);
        v.nb_shiftpow = F::from_canonical_u64(1u64 << nb_shift);
        let nb_w = nb_prod + linf_m * (1u64 << nb_shift);
        v.nb_w = F::from_canonical_u64(nb_w);
        v.nb_w_lo = F::from_canonical_u64(nb_w & 0xFFFF);
        v.nb_w_hi = F::from_canonical_u64(nb_w >> 16);
        // Round W (exact sum) to the 8-bit significand. exp_base = E(dr) + E(l2).
        let nb_code = fill_round_sig_to_bf16::<F>(
            nb_w, dr_e + l2_e, &mut v.nb_gm, &mut v.nb_pow, &mut v.nb_mant, &mut v.nb_exp, &mut v.nb_bottom,
            &mut v.nb_bottom_inv, &mut v.nb_parity, &mut v.nb_half, &mut v.nb_blo, &mut v.nb_bhi, &mut v.nb_sl_lo,
            &mut v.nb_sl_hi, &mut v.nb_su_lo, &mut v.nb_su_hi,
        );
        v.nb_code = F::from_canonical_u64(nb_code);
        debug_assert_eq!(
            nb_code as u16,
            bf16_fma(self.dr_code(), l2_code as u16, linf_code as u16).expect("nb finite"),
            "noised_bound diverged from bf16_fma"
        );

        // ---- alpha = bf16_div(2^16, noised_bound) (power-of-two numerator). ----
        let alpha = u64::from(bf16_div(0x4780, nb_code as u16).expect("alpha finite"));
        let (alpha_m, alpha_e) = (128 + (alpha & 0x7F), alpha >> 7);
        assert!((1..=254).contains(&alpha_e), "envelope: alpha normal");
        v.alpha_exp = F::from_canonical_u64(alpha_e);
        v.alpha_mant = F::from_canonical_u64(alpha & 0x7F);
        let a_shift = DIV_KEY_BASE - nb_exp_of(nb_code) - alpha_e;
        assert!((0..=26).contains(&a_shift), "envelope: alpha division shift out of FP16POW2 domain ({a_shift})");
        v.alpha_pow = F::from_canonical_u64(1u64 << a_shift);
        let bottom = u64::from(alpha_m == 128);
        let parity = alpha_m & 1;
        v.alpha_bottom = F::from_canonical_u64(bottom);
        v.alpha_bottom_inv = inv_diff::<F>(alpha_m, 128);
        v.alpha_parity = F::from_canonical_u64(parity);
        v.alpha_half = F::from_canonical_u64(((alpha & 0x7F) - parity) / 2); // parity split on the 7-bit mantissa
        let nb_m = 128 + (nb_code & 0x7F);
        let blo = (4 * alpha_m - 2 + bottom) * nb_m;
        let bhi = (4 * alpha_m + 2) * nb_m;
        let dpow = 1u64 << a_shift;
        v.alpha_blo = F::from_canonical_u64(blo);
        v.alpha_bhi = F::from_canonical_u64(bhi);
        v.alpha_sl = F::from_canonical_u64(dpow - blo - parity);
        v.alpha_su = F::from_canonical_u64(bhi - dpow - parity);
        v.alpha_code = F::from_canonical_u64(alpha);

        // ---- m1 = bf16_mul(alpha, l2); beta = bf16_mul(m1, dos). ----
        let m1 = u64::from(bf16_mul(alpha as u16, l2_code as u16).expect("m1 finite"));
        fill_bf16_mul::<F>(
            alpha_m, alpha_e, l2_m, l2_e, m1, &mut v.m1_prod, &mut v.m1_mant, &mut v.m1_exp, &mut v.m1_bottom,
            &mut v.m1_bottom_inv, &mut v.m1_parity, &mut v.m1_half, &mut v.m1_pow, &mut v.m1_gm, &mut v.m1_blo,
            &mut v.m1_bhi, &mut v.m1_sl, &mut v.m1_su,
        );
        v.m1_code = F::from_canonical_u64(m1);
        let dos = u64::from(self.dos_code());
        let (dos_m, dos_e) = (128 + (dos & 0x7F), dos >> 7);
        let (m1_m, m1_e) = (128 + (m1 & 0x7F), m1 >> 7);
        let beta = u64::from(bf16_mul(m1 as u16, dos as u16).expect("beta finite"));
        fill_bf16_mul::<F>(
            m1_m, m1_e, dos_m, dos_e, beta, &mut v.beta_prod, &mut v.beta_mant, &mut v.beta_exp, &mut v.beta_bottom,
            &mut v.beta_bottom_inv, &mut v.beta_parity, &mut v.beta_half, &mut v.beta_pow, &mut v.beta_gm,
            &mut v.beta_blo, &mut v.beta_bhi, &mut v.beta_sl, &mut v.beta_su,
        );
        v.beta_code = F::from_canonical_u64(beta);

        v
    }
}

/// bf16 biased exponent field of a code.
fn nb_exp_of(code: u64) -> u64 {
    code >> 7
}

/// Fills a "round a 24-bit significand `sig24` to the 8-bit bf16 significand" gadget at the fixed
/// shift 16 (with a mantissa-overflow carry). `base_exp` is the output exponent before the carry.
/// Returns the bf16 code. All boundary products use the constant factor `2^16`.
#[allow(clippy::too_many_arguments)]
fn fill_round24to8<F: RichField>(
    sig24: u64, base_exp: u64, mant: &mut F, exp: &mut F, carry: &mut F, bottom: &mut F, bottom_inv: &mut F,
    parity: &mut F, half: &mut F, blo: &mut F, bhi: &mut F, sl_lo: &mut F, sl_hi: &mut F, su_lo: &mut F, su_hi: &mut F,
) -> u64 {
    let m_pre = round_shift(sig24, 16); // in [128, 256]
    let is_carry = m_pre == 256;
    let m_out = if is_carry { 128 } else { m_pre };
    let out_exp = base_exp + u64::from(is_carry);
    let out_mant = m_out - 128;
    *mant = F::from_canonical_u64(out_mant);
    *exp = F::from_canonical_u64(out_exp);
    *carry = F::from_bool(is_carry);
    let is_bottom = u64::from(m_pre == 128);
    *bottom = F::from_canonical_u64(is_bottom);
    *bottom_inv = inv_diff::<F>(m_pre, 128);
    let par = m_pre & 1;
    *parity = F::from_canonical_u64(par);
    *half = F::from_canonical_u64((m_pre - par) / 2);
    let pow = 1u64 << 16;
    let blo_v = (4 * m_pre - 2 + is_bottom) * pow;
    let bhi_v = (4 * m_pre + 2) * pow;
    *blo = F::from_canonical_u64(blo_v);
    *bhi = F::from_canonical_u64(bhi_v);
    let sl = 4 * sig24 - blo_v - par;
    let su = bhi_v - 4 * sig24 - par;
    *sl_lo = F::from_canonical_u64(sl & 0xFFFF);
    *sl_hi = F::from_canonical_u64(sl >> 16);
    *su_lo = F::from_canonical_u64(su & 0xFFFF);
    *su_hi = F::from_canonical_u64(su >> 16);
    (out_exp << 7) | out_mant
}

/// Fills a "round an exact integer significand `w` (with LSB scale `unit`) to a bf16 code" gadget:
/// `gm = bitlen(w) - 8`, `M_out = round_shift(w, gm) in [128, 255]`, `exp = unit + gm + 134`. The
/// shift `2^gm` is an FP16POW2 value and the bracket uses the binade-bottom correction.
#[allow(clippy::too_many_arguments)]
fn fill_round_sig_to_bf16<F: RichField>(
    w: u64, exp_base: u64, gm: &mut F, pow: &mut F, mant: &mut F, exp: &mut F, bottom: &mut F, bottom_inv: &mut F,
    parity: &mut F, half: &mut F, blo: &mut F, bhi: &mut F, sl_lo: &mut F, sl_hi: &mut F, su_lo: &mut F, su_hi: &mut F,
) -> u64 {
    let mut g = bit_length(w) - 8;
    let mut m_out = round_shift(w, g);
    if m_out == 256 {
        m_out = 128;
        g += 1;
    }
    assert!((128..256).contains(&m_out), "bf16 significand normalized");
    // out_exp = (E(dr) + E(l2) - 268) + g + 134 = exp_base + g - 134, with exp_base = E(dr)+E(l2).
    let out_exp = exp_base + g - BF16_UNIT;
    let out_mant = m_out - 128;
    *gm = F::from_canonical_u64(g);
    *pow = F::from_canonical_u64(1u64 << g);
    *mant = F::from_canonical_u64(out_mant);
    *exp = F::from_canonical_u64(out_exp);
    let is_bottom = u64::from(m_out == 128);
    *bottom = F::from_canonical_u64(is_bottom);
    *bottom_inv = inv_diff::<F>(m_out, 128);
    let par = m_out & 1;
    *parity = F::from_canonical_u64(par);
    *half = F::from_canonical_u64((out_mant - par) / 2); // parity split is on the 7-bit mantissa
    let p = 1u64 << g;
    let blo_v = (4 * m_out - 2 + is_bottom) * p;
    let bhi_v = (4 * m_out + 2) * p;
    *blo = F::from_canonical_u64(blo_v);
    *bhi = F::from_canonical_u64(bhi_v);
    let sl = 4 * w - blo_v - par;
    let su = bhi_v - 4 * w - par;
    *sl_lo = F::from_canonical_u64(sl & 0xFFFF);
    *sl_hi = F::from_canonical_u64(sl >> 16);
    *su_lo = F::from_canonical_u64(su & 0xFFFF);
    *su_hi = F::from_canonical_u64(su >> 16);
    (out_exp << 7) | out_mant
}

/// Fills a bf16 RNE multiply `out = RNE_bf16(a*b)` gadget (a/b normal positive). `out` is the
/// reference code (for the exponent); the exact product significand is rounded to 8 bits.
#[allow(clippy::too_many_arguments)]
fn fill_bf16_mul<F: RichField>(
    a_m: u64, a_e: u64, b_m: u64, b_e: u64, out: u64, prod: &mut F, mant: &mut F, exp: &mut F, bottom: &mut F,
    bottom_inv: &mut F, parity: &mut F, half: &mut F, pow: &mut F, gm: &mut F, blo: &mut F, bhi: &mut F, sl: &mut F,
    su: &mut F,
) {
    let p = a_m * b_m;
    *prod = F::from_canonical_u64(p);
    let mut g = bit_length(p) - 8;
    let mut m_out = round_shift(p, g);
    if m_out == 256 {
        m_out = 128;
        g += 1;
    }
    let out_exp = a_e + b_e + g - BF16_UNIT; // = a_e + b_e + g - 134
    debug_assert_eq!((out_exp << 7) | (m_out - 128), out, "bf16_mul fill mismatch");
    let out_mant = m_out - 128;
    *mant = F::from_canonical_u64(out_mant);
    *exp = F::from_canonical_u64(out_exp);
    *gm = F::from_canonical_u64(g);
    *pow = F::from_canonical_u64(1u64 << g);
    let is_bottom = u64::from(m_out == 128);
    *bottom = F::from_canonical_u64(is_bottom);
    *bottom_inv = inv_diff::<F>(m_out, 128);
    let par = m_out & 1;
    *parity = F::from_canonical_u64(par);
    *half = F::from_canonical_u64((out_mant - par) / 2); // parity split is on the 7-bit mantissa
    let pw = 1u64 << g;
    let blo_v = (4 * m_out - 2 + is_bottom) * pw;
    let bhi_v = (4 * m_out + 2) * pw;
    *blo = F::from_canonical_u64(blo_v);
    *bhi = F::from_canonical_u64(bhi_v);
    *sl = F::from_canonical_u64(4 * p - blo_v - par);
    *su = F::from_canonical_u64(bhi_v - 4 * p - par);
}

/// Overlays the scalar columns of `src` onto `dst` (the whole-row columns that are identical on
/// every trace row). Copies exactly the scalar fields (everything from `sumsq_sig` onward).
fn copy_scalars<F: RichField>(dst: &mut RowScaleColumnsView<F>, src: &RowScaleColumnsView<F>) {
    let d: &mut [F; NUM_ROW_SCALE_COLUMNS] = dst.borrow_mut();
    let s: &[F; NUM_ROW_SCALE_COLUMNS] = src.borrow();
    let start = super::columns::ROW_SCALE_COL_MAP.sumsq_sig;
    d[start..].copy_from_slice(&s[start..]);
}

// ==================================================================================================
// Constraints (added incrementally; see the module docs for the groups).
// ==================================================================================================

/// `flag = [x == 0]`, gated: `flag` boolean, `flag*x = 0`, `x*inv = 1 - flag`.
fn is_zero_flag<V, S, E>(eval: &mut E, x: V, inv: V, flag: V, gate: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let one = eval.u64(1);
    eval.constraint_bool(flag);
    let fx = eval.mul(flag, x);
    let c = eval.mul(gate, fx);
    eval.constraint(c);
    let xi = eval.mul(x, inv);
    let omf = eval.sub(one, flag);
    let d = eval.sub(xi, omf);
    let c = eval.mul(gate, d);
    eval.constraint(c);
}

/// `a OR b` for booleans, committed to `out` (`out = a + b - a*b`), gated.
fn or_flag<V, S, E>(eval: &mut E, a: V, b: V, out: V, gate: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let ab = eval.mul(a, b);
    let apb = eval.add(a, b);
    let expect = eval.sub(apb, ab);
    let d = eval.sub(out, expect);
    let c = eval.mul(gate, d);
    eval.constraint(c);
}

/// `value = 2^16 * hi + lo`.
fn recon2<V, S, E>(eval: &mut E, lo: V, hi: V) -> V
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let c = eval.u64(1 << 16);
    eval.mad(hi, c, lo)
}

/// `mant = 2*half + parity` with `parity` boolean (`half` is RANGE16'd in `ctl.rs`), gated.
fn parity_split<V, S, E>(eval: &mut E, mant: V, half: V, parity: V, gate: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    eval.constraint_bool(parity);
    let two = eval.u64(2);
    let two_half = eval.mul(two, half);
    let rec = eval.add(two_half, parity);
    let d = eval.sub(mant, rec);
    let c = eval.mul(gate, d);
    eval.constraint(c);
}

/// The committed scale constants passed to [`eval_row_scale_constraints`] (compile-time for the
/// fixed rank `r`).
#[derive(Clone, Copy)]
pub(crate) struct ScaleConsts {
    pub k: u64,
    pub dr_m: u64,
    pub dr_e: u64,
    pub dos_m: u64,
    pub dos_e: u64,
    /// Number of A-side operand rows `h` (the `is_b_side` threshold, as `operand_row_index >= h`).
    pub num_a_rows: u64,
    /// `h * k` — the A-side liveness gate RHS (`64 * dead_A <= h*k`).
    pub a_gate_rhs: u64,
    /// `w * k` — the B-side liveness gate RHS (`64 * dead_B <= w*k`).
    pub b_gate_rhs: u64,
}

pub(crate) fn eval_row_scale_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_ROW_SCALE_COLUMNS, NUM_ROW_SCALE_PUBLIC_INPUTS>,
    eval: &mut E,
    consts: ScaleConsts,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv_arr: &[V; NUM_ROW_SCALE_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let nv_arr: &[V; NUM_ROW_SCALE_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let lv: &RowScaleColumnsView<V> = lv_arr.borrow();
    let nv: &RowScaleColumnsView<V> = nv_arr.borrow();

    let one = eval.u64(1);
    let two = eval.u64(2);
    let c8 = eval.u64(8);
    let c2_15 = eval.u64(1 << 15);
    let c2_23 = eval.u64(1 << 23);
    let c2_24 = eval.u64(1 << 24);
    let c26 = eval.u64(26);
    let c27 = eval.u64(FAR_CAP);
    let c63 = eval.u64(63);

    eval.constraint_bool(lv.is_pad);

    // ============================= GROUP D: decode + pad forcing. =============================
    eval.constraint_bool(lv.x_sign);
    eval.constraint_bool(lv.x_is_zero);
    // abs_code = code - x_sign*2^15.
    let sign_hi = eval.mul(lv.x_sign, c2_15);
    let abs_expect = eval.sub(lv.code, sign_hi);
    eval.constraint_eq(lv.abs_code, abs_expect);
    // Padding rows carry a zero element: code = 0, x_sig = 0, x_sign = 0, x_is_zero = 1.
    let c = eval.mul(lv.is_pad, lv.code);
    eval.constraint(c);
    let c = eval.mul(lv.is_pad, lv.x_sig);
    eval.constraint(c);
    let c = eval.mul(lv.is_pad, lv.x_sign);
    eval.constraint(c);
    let one_m_xz = eval.sub(one, lv.x_is_zero);
    let c = eval.mul(lv.is_pad, one_m_xz);
    eval.constraint(c);

    // ============================= GROUP M: running max of ABS_CODE. =============================
    eval.constraint_bool(lv.max_ge);
    // First row (prev = 0): run_max = abs_code; slack = ge*abs - (1-ge)*(abs+1).
    {
        let not_ge = eval.sub(one, lv.max_ge);
        let rm = eval.mul(lv.max_ge, lv.abs_code); // prev = 0
        eval.constraint_first_row_eq(lv.run_max, rm);
        let abs_p1 = eval.add(lv.abs_code, one);
        let lo = eval.mul(not_ge, abs_p1);
        let slack = eval.sub(rm, lo); // ge*abs - (1-ge)*(abs+1)
        let d = eval.sub(lv.max_slack, slack);
        eval.constraint_first_row(d);
    }
    // Transition (prev = lv.run_max, but 0 at a block start so the running max resets per block):
    // applied to nv's compare.
    {
        let not_bs = eval.sub(one, nv.is_block_start);
        let prev = eval.mul(not_bs, lv.run_max);
        let not_ge = eval.sub(one, nv.max_ge);
        let hi = eval.mul(nv.max_ge, nv.abs_code);
        let rm = eval.mad(not_ge, prev, hi); // ge*abs + (1-ge)*prev
        eval.constraint_transition_eq(nv.run_max, rm);
        let diff_hi = eval.sub(nv.abs_code, prev);
        let a = eval.mul(nv.max_ge, diff_hi); // ge*(abs - prev)
        let prev_m_abs = eval.sub(prev, nv.abs_code);
        let prev_m_abs_m1 = eval.sub(prev_m_abs, one);
        let b = eval.mul(not_ge, prev_m_abs_m1); // (1-ge)*(prev - abs - 1)
        let slack = eval.add(a, b);
        let d = eval.sub(nv.max_slack, slack);
        eval.constraint_transition(d);
    }

    // ============================= GROUP Q: square, normalized to 24 bits. =============================
    let sq_expect = eval.mul(lv.x_sig, lv.x_sig);
    eval.constraint_eq(lv.sq_sig, sq_expect);
    let sq_norm_expect = eval.mul(lv.sq_sig, lv.sq_lift);
    eval.constraint_eq(lv.sq_norm, sq_norm_expect);
    let sq_norm_rec = recon2(eval, lv.sq_norm_lo, lv.sq_norm_hi);
    eval.constraint_eq(lv.sq_norm, sq_norm_rec);

    // ============================= GROUP A: acc chain + fold. =============================
    eval.constraint_bool(lv.acc_in_zero);
    eval.constraint_bool(lv.acc_out_zero);
    let acc_in_rec = recon2(eval, lv.acc_in_sig_lo, lv.acc_in_sig_hi);
    eval.constraint_eq(lv.acc_in_sig, acc_in_rec);
    // Per-block reset: on every block start (incl. trace row 0, always a block start), acc_in = 0.
    eval.constraint_bool(lv.is_block_start);
    eval.constraint_bool(lv.is_block_final);
    {
        let c = eval.mul(lv.is_block_start, lv.acc_in_sig);
        eval.constraint(c);
        let c = eval.mul(lv.is_block_start, lv.acc_in_exp);
        eval.constraint(c);
        let acc_in_z_m1 = eval.sub(lv.acc_in_zero, one);
        let c = eval.mul(lv.is_block_start, acc_in_z_m1);
        eval.constraint(c);
    }
    // Transition inside a block (nv NOT a block start): nv.acc_in = lv.acc_out.
    {
        let not_bs = eval.sub(one, nv.is_block_start);
        let d = eval.sub(nv.acc_in_sig, lv.acc_out_sig);
        let c = eval.mul(not_bs, d);
        eval.constraint_transition(c);
        let d = eval.sub(nv.acc_in_exp, lv.acc_out_exp);
        let c = eval.mul(not_bs, d);
        eval.constraint_transition(c);
        let d = eval.sub(nv.acc_in_zero, lv.acc_out_zero);
        let c = eval.mul(not_bs, d);
        eval.constraint_transition(c);
    }

    let nz_sq = eval.sub(one, lv.x_is_zero);
    let nz_acc = eval.sub(one, lv.acc_in_zero);
    // sq_msb = (1 - x_is_zero)*(2*x_eps_biased + sq_w + 13); acc_msb = (1 - acc_in_zero)*(acc_in_exp - 63).
    {
        let c13 = eval.u64(13);
        let two_eps = eval.mul(two, lv.x_eps_biased);
        let s = eval.add(two_eps, lv.sq_w);
        let s = eval.add(s, c13);
        let e = eval.mul(nz_sq, s);
        eval.constraint_eq(lv.sq_msb, e);
        let am = eval.sub(lv.acc_in_exp, c63);
        let e = eval.mul(nz_acc, am);
        eval.constraint_eq(lv.acc_msb, e);
    }
    // rel = eta - msb; attainment rel_sq*rel_acc = 0.
    let rel_sq_expect = eval.sub(lv.eta, lv.sq_msb);
    eval.constraint_eq(lv.rel_sq, rel_sq_expect);
    let rel_acc_expect = eval.sub(lv.eta, lv.acc_msb);
    eval.constraint_eq(lv.rel_acc, rel_acc_expect);
    let att = eval.mul(lv.rel_sq, lv.rel_acc);
    eval.constraint(att);
    // far flags + two-sided slacks: slack = far*(rel - 27) + (1 - far)*(26 - rel).
    for (far, rel, slack) in
        [(lv.far_sq, lv.rel_sq, lv.far_sq_slack), (lv.far_acc, lv.rel_acc, lv.far_acc_slack)]
    {
        eval.constraint_bool(far);
        let rel_m27 = eval.sub(rel, c27);
        let hi = eval.mul(far, rel_m27);
        let not_far = eval.sub(one, far);
        let c26_m_rel = eval.sub(c26, rel);
        let lo = eval.mul(not_far, c26_m_rel);
        let expect = eval.add(hi, lo);
        let d = eval.sub(slack, expect);
        eval.constraint(d);
    }
    // active = (1 - far)*(1 - is_zero).
    let not_far_sq = eval.sub(one, lv.far_sq);
    let as_expect = eval.mul(not_far_sq, nz_sq);
    eval.constraint_eq(lv.active_sq, as_expect);
    let not_far_acc = eval.sub(one, lv.far_acc);
    let aa_expect = eval.mul(not_far_acc, nz_acc);
    eval.constraint_eq(lv.active_acc, aa_expect);
    eval.constraint_bool(lv.active_sq);
    eval.constraint_bool(lv.active_acc);

    // Alignment (Euclidean floor) for each term: sig*8 = aligned*pow + rem, rem + rem_bound + 1 = pow.
    #[allow(clippy::type_complexity)]
    let align: [(V, V, V, V, V, V, V, V, V, V, V, V, V); 2] = [
        (
            lv.active_sq, lv.sq_norm, lv.pow_sq, lv.aligned_sq, lv.aligned_sq_lo, lv.aligned_sq_hi, lv.rem_sq,
            lv.rem_sq_lo, lv.rem_sq_hi, lv.rem_sq_bound, lv.rem_sq_bound_lo, lv.rem_sq_bound_hi, lv.pow_sq,
        ),
        (
            lv.active_acc, lv.acc_in_sig, lv.pow_acc, lv.aligned_acc, lv.aligned_acc_lo, lv.aligned_acc_hi,
            lv.rem_acc, lv.rem_acc_lo, lv.rem_acc_hi, lv.rem_acc_bound, lv.rem_acc_bound_lo, lv.rem_acc_bound_hi,
            lv.pow_acc,
        ),
    ];
    for (active, sig, pow, aligned, aligned_lo, aligned_hi, rem, rem_lo, rem_hi, rem_bound, rem_bound_lo, rem_bound_hi, _p) in
        align
    {
        let sig8 = eval.mul(c8, sig);
        let ap = eval.mul(aligned, pow);
        let floor_e = eval.sub(sig8, ap);
        let floor_e = eval.sub(floor_e, rem);
        let c = eval.mul(active, floor_e);
        eval.constraint(c);
        let rs = eval.add(rem, rem_bound);
        let rs = eval.add(rs, one);
        let rs = eval.sub(rs, pow);
        let c = eval.mul(active, rs);
        eval.constraint(c);
        let not_active = eval.sub(one, active);
        let c = eval.mul(not_active, aligned);
        eval.constraint(c);
        let c = eval.mul(not_active, rem);
        eval.constraint(c);
        let rec = recon2(eval, aligned_lo, aligned_hi);
        eval.constraint_eq(aligned, rec);
        let rec = recon2(eval, rem_lo, rem_hi);
        eval.constraint_eq(rem, rec);
        let rec = recon2(eval, rem_bound_lo, rem_bound_hi);
        eval.constraint_eq(rem_bound, rec);
    }

    // Sticky bits.
    let rem_sq_zero = eval.sub(one, lv.rem_sq_nz);
    eval.constraint_bool(lv.rem_sq_nz);
    is_zero_flag(eval, lv.rem_sq, lv.rem_sq_nz_inv, rem_sq_zero, one);
    let rem_acc_zero = eval.sub(one, lv.rem_acc_nz);
    eval.constraint_bool(lv.rem_acc_nz);
    is_zero_flag(eval, lv.rem_acc, lv.rem_acc_nz_inv, rem_acc_zero, one);
    or_flag(eval, lv.far_sq, lv.rem_sq_nz, lv.or_sq, one);
    or_flag(eval, lv.far_acc, lv.rem_acc_nz, lv.or_acc, one);
    let st = eval.mul(nz_sq, lv.or_sq);
    eval.constraint_eq(lv.sticky_sq, st);
    let st = eval.mul(nz_acc, lv.or_acc);
    eval.constraint_eq(lv.sticky_acc, st);
    or_flag(eval, lv.sticky_sq, lv.sticky_acc, lv.far_sticky, one);
    eval.constraint_bool(lv.sticky_sq);
    eval.constraint_bool(lv.sticky_acc);
    eval.constraint_bool(lv.far_sticky);

    // Window sum W = aligned_sq + aligned_acc.
    let w_expect = eval.add(lv.aligned_sq, lv.aligned_acc);
    eval.constraint_eq(lv.w_abs, w_expect);
    let w_rec = recon2(eval, lv.w_abs_lo, lv.w_abs_hi);
    eval.constraint_eq(lv.w_abs, w_rec);
    eval.constraint_bool(lv.w_is_zero);
    is_zero_flag(eval, lv.w_abs, lv.w_abs_inv, lv.w_is_zero, one);
    let nz_w = eval.sub(one, lv.w_is_zero);

    // Renormalize: W*lift = m_rz*trunc + rz_rem, rz_rem + rz_rem_bound + 1 = trunc.
    let lifted = eval.mul(lv.w_abs, lv.lift_w);
    let trunc = eval.mul(lv.m_rz, lv.trunc_w);
    let diff = eval.sub(lifted, trunc);
    let diff = eval.sub(diff, lv.rz_rem);
    let c = eval.mul(nz_w, diff);
    eval.constraint(c);
    let rs = eval.add(lv.rz_rem, lv.rz_rem_bound);
    let rs = eval.add(rs, one);
    let rs = eval.sub(rs, lv.trunc_w);
    let c = eval.mul(nz_w, rs);
    eval.constraint(c);
    let m_rz_rec = recon2(eval, lv.m_rz_lo, lv.m_rz_hi);
    eval.constraint_eq(lv.m_rz, m_rz_rec);
    parity_split(eval, lv.m_rz_lo, lv.rz_half, lv.rz_parity, one);
    // RNE.
    eval.constraint_bool(lv.gt);
    let two_rem = eval.mul(two, lv.rz_rem);
    {
        let hi_arg = eval.sub(two_rem, lv.trunc_w);
        let hi_arg = eval.sub(hi_arg, one);
        let hi = eval.mul(lv.gt, hi_arg);
        let not_gt = eval.sub(one, lv.gt);
        let lo_arg = eval.sub(lv.trunc_w, two_rem);
        let lo = eval.mul(not_gt, lo_arg);
        let expect = eval.add(hi, lo);
        let d = eval.sub(lv.gt_slack, expect);
        eval.constraint(d);
    }
    eval.constraint_bool(lv.eq);
    let eq_arg = eval.sub(lv.trunc_w, two_rem);
    is_zero_flag(eval, eq_arg, lv.eq_inv, lv.eq, one);
    or_flag(eval, lv.far_sticky, lv.rz_parity, lv.or_rs, one);
    eval.constraint_bool(lv.or_rs);
    eval.constraint_bool(lv.round_up);
    let eqor = eval.mul(lv.eq, lv.or_rs);
    let ru = eval.add(lv.gt, eqor);
    eval.constraint_eq(lv.round_up, ru);
    eval.constraint_bool(lv.carry);
    let m_out = eval.add(lv.m_rz, lv.round_up);
    let carry_arg = eval.sub(c2_24, m_out);
    is_zero_flag(eval, carry_arg, lv.carry_inv, lv.carry, one);
    // acc_out.
    let carry_term = eval.mul(lv.carry, c2_23);
    let m_out_final = eval.sub(m_out, carry_term);
    let sig_core = eval.mul(nz_w, m_out_final);
    eval.constraint_eq(lv.acc_out_sig, sig_core);
    let acc_out_rec = recon2(eval, lv.acc_out_sig_lo, lv.acc_out_sig_hi);
    eval.constraint_eq(lv.acc_out_sig, acc_out_rec);
    {
        let c_add = eval.u64(ACC_EXP_ADD);
        let s = eval.add(lv.eta, lv.ww);
        let s = eval.add(s, lv.carry);
        let s = eval.add(s, c_add);
        let e = eval.mul(nz_w, s);
        eval.constraint_eq(lv.acc_out_exp, e);
    }
    // acc_out_zero = w_is_zero.
    eval.constraint_eq(lv.acc_out_zero, lv.w_is_zero);

    // ============================= SCALARS: per-block constancy + block-final pins. =============
    // Scalars are constant WITHIN a block (they may jump across a block start). Gating the
    // transition equality by `1 - nv.is_block_start` lets each operand row carry its own derivation.
    let scalar_start = super::columns::ROW_SCALE_COL_MAP.sumsq_sig;
    {
        let not_bs = eval.sub(one, nv.is_block_start);
        for i in scalar_start..NUM_ROW_SCALE_COLUMNS {
            let d = eval.sub(nv_arr[i], lv_arr[i]);
            let c = eval.mul(not_bs, d);
            eval.constraint_transition(c);
        }
    }
    // On each LIVE block's final row, sumsq = that block's acc_out and max_abs_code = its run_max.
    {
        let bf = lv.is_block_final;
        for (scalar, acc_out) in [
            (lv.sumsq_sig, lv.acc_out_sig),
            (lv.sumsq_exp, lv.acc_out_exp),
            (lv.sumsq_zero, lv.acc_out_zero),
            (lv.max_abs_code, lv.run_max),
        ] {
            let d = eval.sub(scalar, acc_out);
            let c = eval.mul(bf, d);
            eval.constraint(c);
        }
    }

    // ============================= GROUP L: entry-liveness gate. =============================
    // Per element, `dead = [ |x| >= 4*l2 ]` with `|x| = x_sig*2^(x_eps_biased-25)` and
    // `4*l2 = (128+l2_m)*2^(l2_e-132)`; the aligned integer compare shifts the lower-exponent side
    // up by `|s|` (FP16POW2), with `s = x_eps_biased - l2_e + 107`. The two per-side dead counts are
    // then gated `64*dead_side <= rows_side*k` (eps_idle = 1/64) by a nonnegative RANGE16 slack, so
    // a tile with too many dead entries on either side has no satisfying witness.
    let live = eval.sub(one, lv.is_pad);
    eval.constraint_bool(lv.dead);
    eval.constraint_bool(lv.dead_sign);
    eval.constraint_bool(lv.is_b_side);

    // L2: dead_key = s * (2*dead_sign - 1). With `dead_key` in FP16POW2's nonneg key domain this
    // pins dead_key = |s| AND the sign.
    let c107 = eval.u64(107);
    let s = eval.sub(lv.x_eps_biased, lv.l2_e);
    let s = eval.add(s, c107);
    let two_sign = eval.mul(two, lv.dead_sign);
    let two_sign_m1 = eval.sub(two_sign, one);
    let key_expect = eval.mul(s, two_sign_m1);
    let d = eval.sub(lv.dead_key, key_expect);
    let c = eval.mul(live, d);
    eval.constraint(c);

    // L3: dead_pow_a = dead_sign ? dead_pow : 1 = dead_sign*(dead_pow-1) + 1.
    let pow_m1 = eval.sub(lv.dead_pow, one);
    let sel = eval.mul(lv.dead_sign, pow_m1);
    let powa_expect = eval.add(sel, one);
    let d = eval.sub(lv.dead_pow_a, powa_expect);
    let c = eval.mul(live, d);
    eval.constraint(c);

    // L4: dead_lhs = x_sig * dead_pow_a.
    let lhs_expect = eval.mul(lv.x_sig, lv.dead_pow_a);
    let d = eval.sub(lv.dead_lhs, lhs_expect);
    let c = eval.mul(live, d);
    eval.constraint(c);

    // L5: dead_rhs = (128 + l2_m) * pow_b, pow_b = dead_pow + 1 - dead_pow_a.
    let c128 = eval.u64(128);
    let mb = eval.add(c128, lv.l2_m);
    let pow_b = eval.add(lv.dead_pow, one);
    let pow_b = eval.sub(pow_b, lv.dead_pow_a);
    let rhs_expect = eval.mul(mb, pow_b);
    let d = eval.sub(lv.dead_rhs, rhs_expect);
    let c = eval.mul(live, d);
    eval.constraint(c);

    // L6: comparison. slack = dead*(lhs-rhs) + (1-dead)*(rhs-lhs-1), slack >= 0 (3 RANGE16 limbs).
    let c2_32 = eval.u64(1u64 << 32);
    let lo_mid = recon2(eval, lv.dead_slack_lo, lv.dead_slack_mid);
    let slack = eval.mad(lv.dead_slack_hi, c2_32, lo_mid);
    let lhs_m_rhs = eval.sub(lv.dead_lhs, lv.dead_rhs);
    let rhs_m_lhs = eval.sub(lv.dead_rhs, lv.dead_lhs);
    let rhs_m_lhs_m1 = eval.sub(rhs_m_lhs, one);
    let hi_t = eval.mul(lv.dead, lhs_m_rhs);
    let not_dead = eval.sub(one, lv.dead);
    let lo_t = eval.mul(not_dead, rhs_m_lhs_m1);
    let expect = eval.add(hi_t, lo_t);
    let d = eval.sub(slack, expect);
    let c = eval.mul(live, d);
    eval.constraint(c);

    // L7: is_b_side = [operand_row_index >= num_a_rows], two-sided slack pin.
    let h = eval.u64(consts.num_a_rows);
    let ori_m_h = eval.sub(lv.operand_row_index, h);
    let hi_b = eval.mul(lv.is_b_side, ori_m_h);
    let hm1 = eval.u64(consts.num_a_rows.saturating_sub(1));
    let hm1_m_ori = eval.sub(hm1, lv.operand_row_index);
    let not_b = eval.sub(one, lv.is_b_side);
    let lo_b = eval.mul(not_b, hm1_m_ori);
    let bexpect = eval.add(hi_b, lo_b);
    let d = eval.sub(lv.b_side_slack, bexpect);
    let c = eval.mul(live, d);
    eval.constraint(c);

    // L8: per-side inclusive dead accumulators (side-masked global sums; frozen on padding).
    // First row (always a live A-side cell start): anchor each to this row's own contribution.
    let not_b0 = eval.sub(one, lv.is_b_side);
    let contrib_a0 = eval.mul(not_b0, lv.dead);
    let d = eval.sub(lv.dead_run_a, contrib_a0);
    eval.constraint_first_row(d);
    let contrib_b0 = eval.mul(lv.is_b_side, lv.dead);
    let d = eval.sub(lv.dead_run_b, contrib_b0);
    eval.constraint_first_row(d);
    // Transition: add the NEXT row's masked contribution.
    let live_next = eval.sub(one, nv.is_pad);
    let not_b_next = eval.sub(one, nv.is_b_side);
    let mask_a = eval.mul(live_next, not_b_next);
    let add_a = eval.mul(mask_a, nv.dead);
    let step_a = eval.add(lv.dead_run_a, add_a);
    let d = eval.sub(nv.dead_run_a, step_a);
    eval.constraint_transition(d);
    let mask_b = eval.mul(live_next, nv.is_b_side);
    let add_b = eval.mul(mask_b, nv.dead);
    let step_b = eval.add(lv.dead_run_b, add_b);
    let d = eval.sub(nv.dead_run_b, step_b);
    eval.constraint_transition(d);

    // L9: last-row per-side gates `rows_side*k - 64*dead_side = slack >= 0` (RANGE16 limbs).
    let c64 = eval.u64(64);
    let a_rhs = eval.u64(consts.a_gate_rhs);
    let sf_a = eval.mul(c64, lv.dead_run_a);
    let gate_a = eval.sub(a_rhs, sf_a);
    let a_slack = recon2(eval, lv.a_gate_slack_lo, lv.a_gate_slack_hi);
    let d = eval.sub(gate_a, a_slack);
    eval.constraint_last_row(d);
    let b_rhs = eval.u64(consts.b_gate_rhs);
    let sf_b = eval.mul(c64, lv.dead_run_b);
    let gate_b = eval.sub(b_rhs, sf_b);
    let b_slack = recon2(eval, lv.b_gate_slack_lo, lv.b_gate_slack_hi);
    let d = eval.sub(gate_b, b_slack);
    eval.constraint_last_row(d);

    eval_scale_chain(lv, eval, consts);
}

/// The scalar norm + scale chain (groups V/S/C/G/F/L/H). All columns are constant across rows, so
/// these apply on every row (ungated) and reference only scalar columns.
fn eval_scale_chain<V, S, E>(lv: &RowScaleColumnsView<V>, eval: &mut E, consts: ScaleConsts)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let one = eval.u64(1);
    let two = eval.u64(2);
    let c4 = eval.u64(4);
    let c128 = eval.u64(128);
    let c2_7 = eval.u64(1 << 7);
    let c2_16 = eval.u64(1 << 16);
    let c2_23 = eval.u64(1 << 23);
    let c6 = eval.u64(6);
    let c134 = eval.u64(BF16_UNIT);
    let floor = eval.u64(FLOOR_CODE);
    let k = eval.u64(consts.k);
    let dr_m = eval.u64(consts.dr_m);
    let dr_e = eval.u64(consts.dr_e);
    let dos_m = eval.u64(consts.dos_m);
    let dos_e = eval.u64(consts.dos_e);

    let recon2l = |eval: &mut E, lo: V, hi: V| -> V {
        let c = eval.u64(1 << 16);
        eval.mad(hi, c, lo)
    };
    let recon3l = |eval: &mut E, lo: V, mid: V, hi: V| -> V {
        let c16 = eval.u64(1 << 16);
        let c32 = eval.u64(1 << 32);
        let a = eval.mul(hi, c32);
        let b = eval.mul(mid, c16);
        let s = eval.add(a, b);
        eval.add(s, lo)
    };

    // ---- sumsq / q limb reconstructions + zero flags. ----
    let sumsq_rec = recon2l(eval, lv.sumsq_sig_lo, lv.sumsq_sig_hi);
    eval.constraint_eq(lv.sumsq_sig, sumsq_rec);
    eval.constraint_bool(lv.sumsq_zero);
    eval.constraint_bool(lv.q_zero);
    // q_zero = sumsq_zero (q = 0 iff sumsq = 0).
    eval.constraint_eq(lv.q_zero, lv.sumsq_zero);
    let nz = eval.sub(one, lv.q_zero);
    let q_rec = recon2l(eval, lv.q_sig_lo, lv.q_sig_hi);
    eval.constraint_eq(lv.q_sig, q_rec);
    // Zero branch: q_sig = 0.
    let c = eval.mul(lv.q_zero, lv.q_sig);
    eval.constraint(c);

    // ============================= GROUP V: q = RNE_f32(sumsq / k). =============================
    // q_dexp = sumsq_exp - q_exp.
    {
        let d = eval.sub(lv.sumsq_exp, lv.q_exp);
        let diff = eval.sub(lv.q_dexp, d);
        let c = eval.mul(nz, diff);
        eval.constraint(c);
    }
    eval.constraint_bool(lv.q_bottom);
    is_zero_flag_scaled(eval, lv.q_sig, c2_23, lv.q_bottom_inv, lv.q_bottom, nz);
    parity_split_scaled(eval, lv.q_sig_lo, lv.q_half, lv.q_parity, nz);
    // q_blo = (4*q_sig - 2 + q_bottom)*k; q_bhi = (4*q_sig + 2)*k.
    let four_q = eval.mul(c4, lv.q_sig);
    {
        let lo_coef = eval.sub(four_q, two);
        let lo_coef = eval.add(lo_coef, lv.q_bottom);
        let blo = eval.mul(lo_coef, k);
        let d = eval.sub(lv.q_blo, blo);
        let c = eval.mul(nz, d);
        eval.constraint(c);
        let hi_coef = eval.add(four_q, two);
        let bhi = eval.mul(hi_coef, k);
        let d = eval.sub(lv.q_bhi, bhi);
        let c = eval.mul(nz, d);
        eval.constraint(c);
    }
    // target = 4*sumsq_sig*q_pow; slacks recon3.
    {
        let four_s = eval.mul(c4, lv.sumsq_sig);
        let target = eval.mul(four_s, lv.q_pow);
        let sl = recon3l(eval, lv.q_sl_lo, lv.q_sl_mid, lv.q_sl_hi);
        let e = eval.sub(target, lv.q_blo);
        let e = eval.sub(e, lv.q_parity);
        let d = eval.sub(sl, e);
        let c = eval.mul(nz, d);
        eval.constraint(c);
        let su = recon3l(eval, lv.q_su_lo, lv.q_su_mid, lv.q_su_hi);
        let e = eval.sub(lv.q_bhi, target);
        let e = eval.sub(e, lv.q_parity);
        let d = eval.sub(su, e);
        let c = eval.mul(nz, d);
        eval.constraint(c);
    }

    // ============================= GROUP S: s = RNE_f32(sqrt(q)). =============================
    let s_rec = recon2l(eval, lv.s_sig_lo, lv.s_sig_hi);
    eval.constraint_eq(lv.s_sig, s_rec);
    let c = eval.mul(lv.q_zero, lv.s_sig);
    eval.constraint(c);
    // s_dexp = q_exp + 152 - 2*s_exp.
    {
        let c152 = eval.u64(152);
        let two_se = eval.mul(two, lv.s_exp);
        let d = eval.add(lv.q_exp, c152);
        let d = eval.sub(d, two_se);
        let diff = eval.sub(lv.s_dexp, d);
        let c = eval.mul(nz, diff);
        eval.constraint(c);
    }
    eval.constraint_bool(lv.s_bottom);
    is_zero_flag_scaled(eval, lv.s_sig, c2_23, lv.s_bottom_inv, lv.s_bottom, nz);
    parity_split_scaled(eval, lv.s_sig_lo, lv.s_half, lv.s_parity, nz);
    // s_blo = (4*s_sig - 2 + s_bottom)^2; s_bhi = (4*s_sig + 2)^2.
    let four_s_sig = eval.mul(c4, lv.s_sig);
    {
        let lo_b = eval.sub(four_s_sig, two);
        let lo_b = eval.add(lo_b, lv.s_bottom);
        let sq = eval.mul(lo_b, lo_b);
        let d = eval.sub(lv.s_blo, sq);
        let c = eval.mul(nz, d);
        eval.constraint(c);
        let hi_b = eval.add(four_s_sig, two);
        let sq = eval.mul(hi_b, hi_b);
        let d = eval.sub(lv.s_bhi, sq);
        let c = eval.mul(nz, d);
        eval.constraint(c);
    }
    // target = 4*q_sig*s_pow; slacks recon3.
    {
        let four_q = eval.mul(c4, lv.q_sig);
        let target = eval.mul(four_q, lv.s_pow);
        let sl = recon3l(eval, lv.s_sl_lo, lv.s_sl_mid, lv.s_sl_hi);
        let e = eval.sub(target, lv.s_blo);
        let e = eval.sub(e, lv.s_parity);
        let d = eval.sub(sl, e);
        let c = eval.mul(nz, d);
        eval.constraint(c);
        let su = recon3l(eval, lv.s_su_lo, lv.s_su_mid, lv.s_su_hi);
        let e = eval.sub(lv.s_bhi, target);
        let e = eval.sub(e, lv.s_parity);
        let d = eval.sub(su, e);
        let c = eval.mul(nz, d);
        eval.constraint(c);
    }

    // ============================= GROUP C: l2raw = f32_to_bf16(s). =============================
    round24to8_constraints(
        eval, lv.s_sig, lv.s_exp, lv.l2raw_mant, lv.l2raw_exp, lv.l2raw_carry, lv.l2raw_bottom,
        lv.l2raw_bottom_inv, lv.l2raw_parity, lv.l2raw_half, lv.l2raw_blo, lv.l2raw_bhi, lv.l2raw_sl_lo,
        lv.l2raw_sl_hi, lv.l2raw_su_lo, lv.l2raw_su_hi, lv.l2raw_code, nz,
    );
    // Zero branch: l2raw_code = 0.
    let c = eval.mul(lv.q_zero, lv.l2raw_code);
    eval.constraint(c);

    // ============================= GROUP G: l2grid = round_l2_to_grid(l2raw). =============================
    eval.constraint_bool(lv.grid_r_b0);
    eval.constraint_bool(lv.grid_r_b1);
    let two_b1 = eval.mul(two, lv.grid_r_b1);
    let gr = eval.add(two_b1, lv.grid_r_b0);
    eval.constraint_eq(lv.grid_r, gr);
    // l2raw_code + 2 = 4*grid_q + grid_r.
    {
        let lhs = eval.add(lv.l2raw_code, two);
        let four_q = eval.mul(c4, lv.grid_q);
        let rhs = eval.add(four_q, lv.grid_r);
        eval.constraint_eq(lhs, rhs);
    }
    let l2grid = eval.mul(c4, lv.grid_q);
    eval.constraint_eq(lv.l2grid_code, l2grid);

    // ============================= GROUP F: l2 = bf16_max(l2grid, floor). =============================
    max_with_floor(eval, lv.l2grid_code, floor, lv.l2_ge, lv.l2_floor_slack, lv.l2_code);
    // l2_code = 128*l2_e + l2_m.
    let l2_rec = eval.mad(lv.l2_e, c2_7, lv.l2_m);
    eval.constraint_eq(lv.l2_code, l2_rec);

    // ============================= GROUP L: linf = bf16_max(f32_to_bf16(max|x|), floor). =============================
    eval.constraint_bool(lv.max_is_zero);
    let nz_max = eval.sub(one, lv.max_is_zero);
    let max_norm_rec = recon2l(eval, lv.max_norm_lo, lv.max_norm_hi);
    eval.constraint_eq(lv.max_norm, max_norm_rec);
    // max_norm = max_sig * max_lift (0 when max is zero).
    let mn = eval.mul(lv.max_sig, lv.max_lift);
    eval.constraint_eq(lv.max_norm, mn);
    // base_exp = max_eps_biased + max_w + 101.
    let base_exp = {
        let c101 = eval.u64(101);
        let s = eval.add(lv.max_eps_biased, lv.max_w);
        eval.add(s, c101)
    };
    round24to8_constraints(
        eval, lv.max_norm, base_exp, lv.linf_raw_mant, lv.linf_raw_exp, lv.linf_raw_carry, lv.linf_raw_bottom,
        lv.linf_raw_bottom_inv, lv.linf_raw_parity, lv.linf_raw_half, lv.linf_raw_blo, lv.linf_raw_bhi,
        lv.linf_raw_sl_lo, lv.linf_raw_sl_hi, lv.linf_raw_su_lo, lv.linf_raw_su_hi, lv.linf_raw_code, nz_max,
    );
    let c = eval.mul(lv.max_is_zero, lv.linf_raw_code);
    eval.constraint(c);
    max_with_floor(eval, lv.linf_raw_code, floor, lv.linf_ge, lv.linf_floor_slack, lv.linf_code);
    let linf_rec = eval.mad(lv.linf_e, c2_7, lv.linf_m);
    eval.constraint_eq(lv.linf_code, linf_rec);

    // ============================= GROUP H1: noised_bound = bf16_fma(dr, l2, linf). =============================
    let l2_sig = eval.add(c128, lv.l2_m);
    let linf_sig = eval.add(c128, lv.linf_m);
    // nb_prod = dr_m * l2_sig.
    let nb_prod = eval.mul(dr_m, l2_sig);
    eval.constraint_eq(lv.nb_prod, nb_prod);
    // nb_shift = linf_e - l2_e + 6.
    {
        let d = eval.sub(lv.linf_e, lv.l2_e);
        let d = eval.add(d, c6);
        eval.constraint_eq(lv.nb_shift, d);
    }
    // nb_w = nb_prod + linf_sig * nb_shiftpow.
    let add = eval.mul(linf_sig, lv.nb_shiftpow);
    let nb_w = eval.add(lv.nb_prod, add);
    eval.constraint_eq(lv.nb_w, nb_w);
    let nb_w_rec = recon2l(eval, lv.nb_w_lo, lv.nb_w_hi);
    eval.constraint_eq(lv.nb_w, nb_w_rec);
    // Round W to 8 bits: M_out = 128 + nb_mant, bracket on 4*W with 2^nb_gm.
    let nb_m = eval.add(c128, lv.nb_mant);
    eval.constraint_bool(lv.nb_bottom);
    is_zero_flag(eval, lv.nb_mant, lv.nb_bottom_inv, lv.nb_bottom, one);
    parity_split_scaled(eval, lv.nb_mant, lv.nb_half, lv.nb_parity, one);
    let four_m = eval.mul(c4, nb_m);
    {
        let lo_coef = eval.sub(four_m, two);
        let lo_coef = eval.add(lo_coef, lv.nb_bottom);
        let blo = eval.mul(lo_coef, lv.nb_pow);
        eval.constraint_eq(lv.nb_blo, blo);
        let hi_coef = eval.add(four_m, two);
        let bhi = eval.mul(hi_coef, lv.nb_pow);
        eval.constraint_eq(lv.nb_bhi, bhi);
    }
    {
        let four_w = eval.mul(c4, lv.nb_w);
        let sl = recon2l(eval, lv.nb_sl_lo, lv.nb_sl_hi);
        let e = eval.sub(four_w, lv.nb_blo);
        let e = eval.sub(e, lv.nb_parity);
        eval.constraint_eq(sl, e);
        let su = recon2l(eval, lv.nb_su_lo, lv.nb_su_hi);
        let e = eval.sub(lv.nb_bhi, four_w);
        let e = eval.sub(e, lv.nb_parity);
        eval.constraint_eq(su, e);
    }
    // nb_exp = (dr_e + l2_e) + nb_gm - 134; nb_code = 128*nb_exp + nb_mant.
    {
        let base = eval.add(dr_e, lv.l2_e);
        let s = eval.add(base, lv.nb_gm);
        let s = eval.sub(s, c134);
        eval.constraint_eq(lv.nb_exp, s);
    }
    let nb_code = eval.mad(lv.nb_exp, c2_7, lv.nb_mant);
    eval.constraint_eq(lv.nb_code, nb_code);

    // ============================= GROUP H3: alpha = bf16_div(2^16, noised_bound). =============================
    let alpha_m = eval.add(c128, lv.alpha_mant);
    eval.constraint_bool(lv.alpha_bottom);
    is_zero_flag(eval, lv.alpha_mant, lv.alpha_bottom_inv, lv.alpha_bottom, one);
    parity_split_scaled(eval, lv.alpha_mant, lv.alpha_half, lv.alpha_parity, one);
    let four_a = eval.mul(c4, alpha_m);
    {
        let lo_coef = eval.sub(four_a, two);
        let lo_coef = eval.add(lo_coef, lv.alpha_bottom);
        let blo = eval.mul(lo_coef, nb_m);
        eval.constraint_eq(lv.alpha_blo, blo);
        let hi_coef = eval.add(four_a, two);
        let bhi = eval.mul(hi_coef, nb_m);
        eval.constraint_eq(lv.alpha_bhi, bhi);
    }
    {
        // alpha_pow = 2^A; slack sl = alpha_pow - alpha_blo - alpha_parity, su = alpha_bhi - alpha_pow - alpha_parity.
        let e = eval.sub(lv.alpha_pow, lv.alpha_blo);
        let e = eval.sub(e, lv.alpha_parity);
        eval.constraint_eq(lv.alpha_sl, e);
        let e = eval.sub(lv.alpha_bhi, lv.alpha_pow);
        let e = eval.sub(e, lv.alpha_parity);
        eval.constraint_eq(lv.alpha_su, e);
    }
    let alpha_code = eval.mad(lv.alpha_exp, c2_7, lv.alpha_mant);
    eval.constraint_eq(lv.alpha_code, alpha_code);

    // ============================= GROUP H4/H5: m1 = alpha*l2; beta = m1*dos. =============================
    bf16_mul_constraints(
        eval, alpha_m, lv.alpha_exp, l2_sig, lv.l2_e, lv.m1_prod, lv.m1_mant, lv.m1_exp, lv.m1_bottom,
        lv.m1_bottom_inv, lv.m1_parity, lv.m1_half, lv.m1_pow, lv.m1_gm, lv.m1_blo, lv.m1_bhi, lv.m1_sl, lv.m1_su,
        lv.m1_code,
    );
    let m1_sig = eval.add(c128, lv.m1_mant);
    bf16_mul_constraints(
        eval, m1_sig, lv.m1_exp, dos_m, dos_e, lv.beta_prod, lv.beta_mant, lv.beta_exp, lv.beta_bottom,
        lv.beta_bottom_inv, lv.beta_parity, lv.beta_half, lv.beta_pow, lv.beta_gm, lv.beta_blo, lv.beta_bhi,
        lv.beta_sl, lv.beta_su, lv.beta_code,
    );

    let _ = (c2_16, c2_7);
}

/// `flag = [x == scaled]` is-zero gadget on the difference `x - scaled` (`scaled` a constant/value).
fn is_zero_flag_scaled<V, S, E>(eval: &mut E, x: V, scaled: V, inv: V, flag: V, gate: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let arg = eval.sub(x, scaled);
    is_zero_flag(eval, arg, inv, flag, gate);
}

/// `mant = 2*half + parity`, parity boolean, gated.
fn parity_split_scaled<V, S, E>(eval: &mut E, mant: V, half: V, parity: V, gate: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    parity_split(eval, mant, half, parity, gate);
}

/// `out = max(x, floor)` by code order: `ge` boolean, two-sided slack, `out = ge*x + (1-ge)*floor`.
fn max_with_floor<V, S, E>(eval: &mut E, x: V, floor: V, ge: V, slack: V, out: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let one = eval.u64(1);
    eval.constraint_bool(ge);
    let not_ge = eval.sub(one, ge);
    // slack = ge*(x - floor) + (1-ge)*(floor - x - 1).
    let x_m_f = eval.sub(x, floor);
    let hi = eval.mul(ge, x_m_f);
    let f_m_x = eval.sub(floor, x);
    let f_m_x_m1 = eval.sub(f_m_x, one);
    let lo = eval.mul(not_ge, f_m_x_m1);
    let expect = eval.add(hi, lo);
    let d = eval.sub(slack, expect);
    eval.constraint(d);
    // out = ge*x + (1-ge)*floor.
    let a = eval.mul(ge, x);
    let b = eval.mul(not_ge, floor);
    let o = eval.add(a, b);
    eval.constraint_eq(out, o);
}

/// Round a 24-bit significand `sig24` to the 8-bit bf16 significand at the fixed shift 16 (with a
/// mantissa-overflow carry): `M_pre = 128 + mant + 128*carry`, bracket `(4*M_pre -2 +bottom)*2^16 <=
/// 4*sig24 <= (4*M_pre+2)*2^16`, `out_exp = base_exp + carry`, `code = 128*out_exp + mant`. Gated.
#[allow(clippy::too_many_arguments)]
fn round24to8_constraints<V, S, E>(
    eval: &mut E, sig24: V, base_exp: V, mant: V, exp: V, carry: V, bottom: V, bottom_inv: V, parity: V, half: V,
    blo: V, bhi: V, sl_lo: V, sl_hi: V, su_lo: V, su_hi: V, code: V, gate: V,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let two = eval.u64(2);
    let c4 = eval.u64(4);
    let c128 = eval.u64(128);
    let c2_7 = eval.u64(1 << 7);
    let c2_16 = eval.u64(1 << 16);
    eval.constraint_bool(carry);
    // carry => mant = 0.
    let cm = eval.mul(carry, mant);
    let c = eval.mul(gate, cm);
    eval.constraint(c);
    // M_pre = 128 + mant + 128*carry.
    let c128_carry = eval.mul(c128, carry);
    let m_pre = eval.add(c128, mant);
    let m_pre = eval.add(m_pre, c128_carry);
    // bottom = [M_pre == 128] = [mant + 128*carry == 0].
    let mpre_rel = eval.sub(m_pre, c128);
    eval.constraint_bool(bottom);
    is_zero_flag(eval, mpre_rel, bottom_inv, bottom, gate);
    // parity split on M_pre.
    parity_split(eval, m_pre, half, parity, gate);
    // blo = (4*M_pre - 2 + bottom)*2^16; bhi = (4*M_pre + 2)*2^16.
    let four_m = eval.mul(c4, m_pre);
    let lo_coef = eval.sub(four_m, two);
    let lo_coef = eval.add(lo_coef, bottom);
    let blo_e = eval.mul(lo_coef, c2_16);
    let d = eval.sub(blo, blo_e);
    let c = eval.mul(gate, d);
    eval.constraint(c);
    let hi_coef = eval.add(four_m, two);
    let bhi_e = eval.mul(hi_coef, c2_16);
    let d = eval.sub(bhi, bhi_e);
    let c = eval.mul(gate, d);
    eval.constraint(c);
    // sl = 4*sig24 - blo - parity; su = bhi - 4*sig24 - parity.
    let four_s = eval.mul(c4, sig24);
    let sl = eval.mad(sl_hi, c2_16, sl_lo);
    let e = eval.sub(four_s, blo);
    let e = eval.sub(e, parity);
    let d = eval.sub(sl, e);
    let c = eval.mul(gate, d);
    eval.constraint(c);
    let su = eval.mad(su_hi, c2_16, su_lo);
    let e = eval.sub(bhi, four_s);
    let e = eval.sub(e, parity);
    let d = eval.sub(su, e);
    let c = eval.mul(gate, d);
    eval.constraint(c);
    // out_exp = base_exp + carry.
    let oe = eval.add(base_exp, carry);
    let d = eval.sub(exp, oe);
    let c = eval.mul(gate, d);
    eval.constraint(c);
    // code = 128*exp + mant.
    let code_e = eval.mad(exp, c2_7, mant);
    let d = eval.sub(code, code_e);
    let c = eval.mul(gate, d);
    eval.constraint(c);
}

/// `out = RNE_bf16(a*b)` for normal positive operands: `P = a_sig*b_sig`, rounded to the 8-bit
/// significand at `2^gm` (FP16POW2, pinned via `mant in [0,127]`), `out_exp = a_e + b_e + gm - 134`.
#[allow(clippy::too_many_arguments)]
fn bf16_mul_constraints<V, S, E>(
    eval: &mut E, a_sig: V, a_e: V, b_sig: V, b_e: V, prod: V, mant: V, exp: V, bottom: V, bottom_inv: V, parity: V,
    half: V, pow: V, gm: V, blo: V, bhi: V, sl: V, su: V, code: V,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let one = eval.u64(1);
    let two = eval.u64(2);
    let c4 = eval.u64(4);
    let c128 = eval.u64(128);
    let c2_7 = eval.u64(1 << 7);
    let c134 = eval.u64(BF16_UNIT);
    let p = eval.mul(a_sig, b_sig);
    eval.constraint_eq(prod, p);
    let m_out = eval.add(c128, mant);
    eval.constraint_bool(bottom);
    is_zero_flag(eval, mant, bottom_inv, bottom, one);
    parity_split(eval, mant, half, parity, one);
    let four_m = eval.mul(c4, m_out);
    let lo_coef = eval.sub(four_m, two);
    let lo_coef = eval.add(lo_coef, bottom);
    let blo_e = eval.mul(lo_coef, pow);
    eval.constraint_eq(blo, blo_e);
    let hi_coef = eval.add(four_m, two);
    let bhi_e = eval.mul(hi_coef, pow);
    eval.constraint_eq(bhi, bhi_e);
    let four_p = eval.mul(c4, prod);
    let e = eval.sub(four_p, blo);
    let e = eval.sub(e, parity);
    eval.constraint_eq(sl, e);
    let e = eval.sub(bhi, four_p);
    let e = eval.sub(e, parity);
    eval.constraint_eq(su, e);
    // out_exp = a_e + b_e + gm - 134 (gm is the FP16POW2 key tying `pow = 2^gm` in ctl).
    let base = eval.add(a_e, b_e);
    let base = eval.add(base, gm);
    let base = eval.sub(base, c134);
    eval.constraint_eq(exp, base);
    // code = 128*exp + mant.
    let code_e = eval.mad(exp, c2_7, mant);
    eval.constraint_eq(code, code_e);
}

/// FP16 RowScaleStark. A CTL party (`requires_ctls`): the FP16-batch LUT facts and the
/// operand/scales hooks are CTL-bound in later stages, so the batch driver is the only supported
/// proving path.
#[derive(Clone, Debug)]
pub struct RowScaleStark<F: RichField + Extendable<D>, const D: usize> {
    pub program: RowScaleProgram,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> RowScaleStark<F, D> {
    pub fn new(program: RowScaleProgram) -> Self {
        Self { program, _phantom: PhantomData }
    }

    fn scale_consts(&self) -> ScaleConsts {
        let dr = u64::from(self.program.dr_code());
        let dos = u64::from(self.program.dos_code());
        let h = self.program.num_a_rows as u64;
        let w = (self.program.num_operand_rows - self.program.num_a_rows) as u64;
        let k = self.program.k as u64;
        ScaleConsts {
            k,
            dr_m: 128 + (dr & 0x7F),
            dr_e: dr >> 7,
            dos_m: 128 + (dos & 0x7F),
            dos_e: dos >> 7,
            num_a_rows: h,
            a_gate_rhs: h * k,
            b_gate_rhs: w * k,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for RowScaleStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_ROW_SCALE_COLUMNS, NUM_ROW_SCALE_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_ROW_SCALE_COLUMNS, NUM_ROW_SCALE_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_row_scale_constraints(vars, &mut evaluator, self.scale_consts());
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_row_scale_constraints(vars, &mut evaluator, self.scale_consts());
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    fn requires_ctls(&self) -> bool {
        true
    }
}

// ==================================================================================================
// Tests
// ==================================================================================================

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};

    use super::super::columns::ROW_SCALE_COL_MAP;
    use super::*;
    use crate::v5::api::dtype::f32_to_fp16;
    use crate::v5::api::quantization::{derive_row_scales, row_norms};
    use crate::v4::api::compute::bf16_max;

    const D: usize = 2;
    type F = GoldilocksField;
    type Stk = RowScaleStark<F, D>;

    const R: usize = 32;

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    fn constraints_violated(stark: &Stk, rows: &[[F; NUM_ROW_SCALE_COLUMNS]]) -> bool {
        let n = rows.len();
        (0..n).any(|i| {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], &[]);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::from_bool(i != n - 1),
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            consumer.accumulators().iter().any(|&acc| acc != F::ZERO)
        })
    }

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        for (k, seed, scale) in [(64usize, 1u64, 4.0f32), (48, 7, 0.5), (96, 11, 20.0), (32, 3, 1.0), (16, 9, 0.01)] {
            let program = RowScaleProgram::new(k, R);
            let row = sample_row(k, seed, scale);
            let rows = program.generate_trace::<F>(&row);
            assert!(
                !constraints_violated(&Stk::new(program), &rows),
                "honest trace violated a constraint (k={k} seed={seed} scale={scale})"
            );
        }
    }

    #[test]
    fn degree_is_at_most_three() {
        use starky::stark_testing::test_stark_low_degree;
        let program = RowScaleProgram::new(32, R);
        test_stark_low_degree::<F, Stk, D>(Stk::new(program)).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        use plonky2::plonk::config::PoseidonGoldilocksConfig;
        use starky::stark_testing::test_stark_circuit_constraints;
        let program = RowScaleProgram::new(32, R);
        test_stark_circuit_constraints::<F, PoseidonGoldilocksConfig, Stk, D>(Stk::new(program)).unwrap();
    }

    #[test]
    fn honest_floor_trace_satisfies_constraints() {
        let k = 32;
        let program = RowScaleProgram::new(k, R);
        let rows = program.generate_trace::<F>(&vec![0u16; k]);
        assert!(!constraints_violated(&Stk::new(program), &rows), "all-zero row trace must pass");
    }

    #[test]
    fn tampered_traces_fail() {
        let k = 48;
        let program = RowScaleProgram::new(k, R);
        let row = sample_row(k, 7, 3.0);
        let rows = program.generate_trace::<F>(&row);
        let stark = Stk::new(program);
        assert!(!constraints_violated(&stark, &rows), "baseline honest trace must pass");
        let m = &ROW_SCALE_COL_MAP;
        for (name, col) in [
            ("code", m.code),
            ("abs_code", m.abs_code),
            ("run_max", m.run_max),
            ("sq_sig", m.sq_sig),
            ("sq_norm", m.sq_norm),
            ("acc_out_sig", m.acc_out_sig),
            ("acc_out_exp", m.acc_out_exp),
            ("w_abs", m.w_abs),
            ("m_rz", m.m_rz),
            ("round_up", m.round_up),
            ("eta", m.eta),
            ("sumsq_sig", m.sumsq_sig),
            ("q_sig", m.q_sig),
            ("q_blo", m.q_blo),
            ("s_sig", m.s_sig),
            ("s_blo", m.s_blo),
            ("l2raw_code", m.l2raw_code),
            ("l2grid_code", m.l2grid_code),
            ("l2_code", m.l2_code),
            ("linf_raw_code", m.linf_raw_code),
            ("linf_code", m.linf_code),
            ("nb_prod", m.nb_prod),
            ("nb_w", m.nb_w),
            ("nb_code", m.nb_code),
            ("alpha_code", m.alpha_code),
            ("alpha_blo", m.alpha_blo),
            ("m1_code", m.m1_code),
            ("beta_code", m.beta_code),
        ] {
            let mut forged = rows.clone();
            forged[0][col] += F::ONE;
            assert!(constraints_violated(&stark, &forged), "{name} tamper undetected");
        }
    }

    /// Binade-bottom uniqueness: find a trace with a power-of-two bf16 multiply (`m1_bottom` or
    /// `beta_bottom` set), then show (a) clearing the flag violates the is-zero gadget and (b)
    /// reverting the lower boundary to the uncorrected `(4*M - 2)*pow` (one `pow` below the honest
    /// `+ BOTTOM` value) is rejected — the ties-to-even rounding is pinned with no quarter-ulp slack.
    #[test]
    fn binade_bottom_correction_pins_the_rounding() {
        let m = &ROW_SCALE_COL_MAP;
        // An all-ones row: mean square = 1, sqrt = 1, so l2raw = f32_to_bf16(1.0) = 0x3F80, a power
        // of two (significand 2^23 -> M_pre = 128). The round24to8 binade bottom fires.
        let k = 32;
        let program = RowScaleProgram::new(k, R);
        let row = vec![f32_to_fp16(1.0).unwrap(); k];
        let rows = program.generate_trace::<F>(&row);
        let stark = Stk::new(RowScaleProgram::new(k, R));
        assert!(!constraints_violated(&stark, &rows), "honest power-of-two trace must pass");
        assert_eq!(to_u64(rows[0][m.l2raw_bottom]), 1, "l2raw is a power of two");
        assert_eq!(to_u64(rows[0][m.l2raw_code]) as u16, 0x3F80, "l2raw = 1.0");

        // (a) Clear the bottom flag on every row (scalar column) -> is-zero gadget rejects it.
        let mut f = rows.clone();
        for rr in f.iter_mut() {
            rr[m.l2raw_bottom] = F::ZERO;
        }
        assert!(constraints_violated(&stark, &f), "clearing l2raw_bottom must be rejected");

        // (b) Revert the lower boundary to the uncorrected `(4*M_pre - 2)*2^16` (one 2^16 below the
        // honest `+ BOTTOM` value): the AIR's `blo = (... + bottom)*2^16` now rejects it.
        let mut f = rows.clone();
        let factor = F::from_canonical_u64(1 << 16);
        for rr in f.iter_mut() {
            rr[m.l2raw_blo] -= factor;
        }
        assert!(constraints_violated(&stark, &f), "uncorrected half-ulp lower boundary must be rejected");
    }

    fn sample_row(k: usize, seed: u64, scale: f32) -> Vec<u16> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..k)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = ((s >> 40) as i32 % 2000 - 1000) as f32 / 1000.0 * scale;
                f32_to_fp16(v).unwrap()
            })
            .collect()
    }

    #[test]
    fn trace_is_bit_exact_vs_reference() {
        for (k, seed, scale) in [(64usize, 1u64, 4.0f32), (48, 7, 0.5), (96, 11, 20.0), (32, 3, 1.0)] {
            let row = sample_row(k, seed, scale);
            let program = RowScaleProgram::new(k, R);
            let rows = program.generate_trace::<F>(&row);
            assert_eq!(rows.len(), k.next_power_of_two().max(2));

            let (ref_l2grid, ref_linf_raw) = row_norms(&row).unwrap();
            let floor = f32_to_bf16(NORM_FLOOR).unwrap();
            let l2 = bf16_max(ref_l2grid, floor);
            let linf = bf16_max(ref_linf_raw, floor);
            let (alpha, beta) = derive_row_scales(l2, linf, R).unwrap();

            let v: &RowScaleColumnsView<F> = rows[0].borrow();
            assert_eq!(to_u64(v.l2grid_code) as u16, ref_l2grid, "l2grid k={k} seed={seed}");
            assert_eq!(to_u64(v.linf_raw_code) as u16, ref_linf_raw, "linf_raw k={k} seed={seed}");
            assert_eq!(to_u64(v.l2_code) as u16, l2, "l2 k={k} seed={seed}");
            assert_eq!(to_u64(v.linf_code) as u16, linf, "linf k={k} seed={seed}");
            assert_eq!(to_u64(v.alpha_code) as u16, alpha, "alpha k={k} seed={seed}");
            assert_eq!(to_u64(v.beta_code) as u16, beta, "beta k={k} seed={seed}");
        }
    }

    #[test]
    fn floor_case_all_zero_row() {
        let k = 32;
        let row = vec![0u16; k];
        let program = RowScaleProgram::new(k, R);
        let rows = program.generate_trace::<F>(&row);
        let (ref_l2grid, ref_linf_raw) = row_norms(&row).unwrap();
        let floor = f32_to_bf16(NORM_FLOOR).unwrap();
        let l2 = bf16_max(ref_l2grid, floor);
        let linf = bf16_max(ref_linf_raw, floor);
        let (alpha, beta) = derive_row_scales(l2, linf, R).unwrap();
        let v: &RowScaleColumnsView<F> = rows[0].borrow();
        assert_eq!(to_u64(v.l2_code) as u16, l2, "floored l2");
        assert_eq!(to_u64(v.linf_code) as u16, linf, "floored linf");
        assert_eq!(to_u64(v.alpha_code) as u16, alpha, "alpha");
        assert_eq!(to_u64(v.beta_code) as u16, beta, "beta");
        assert_eq!(l2, floor);
        assert_eq!(linf, floor);
    }

    #[test]
    fn known_column_matches_fill() {
        let k = 48;
        let program = RowScaleProgram::new(k, R);
        let row = sample_row(k, 5, 2.0);
        let rows = program.generate_trace::<F>(&row);
        let known = program.known_values::<F>();
        assert_eq!(known.len(), 4);
        for (r, row) in rows.iter().enumerate() {
            for (j, col) in [
                ROW_SCALE_COL_MAP.is_pad,
                ROW_SCALE_COL_MAP.operand_row_index,
                ROW_SCALE_COL_MAP.is_block_start,
                ROW_SCALE_COL_MAP.is_block_final,
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(known[j].values[r], row[col], "known col {j} row {r}");
            }
        }
    }

    /// The extended multi-operand-row path: several rows of `k` elements proved end to end, each a
    /// self-contained per-row derivation bit-exact with the reference, plus a trailing padding block.
    #[test]
    fn multi_row_trace_is_bit_exact_and_satisfies_constraints() {
        use crate::v5::api::quantization::{derive_row_scales, row_norms};
        use crate::v4::api::compute::bf16_max;
        let k = 32usize;
        let num_operand_rows = 6usize; // e.g. h=2, w=4
        let num_rows = (num_operand_rows * k).next_power_of_two(); // 192 -> 256
        let program = RowScaleProgram::with_rows(num_operand_rows, k, R, num_rows);
        let mut codes = Vec::new();
        for i in 0..num_operand_rows {
            codes.extend(sample_row(k, 1 + i as u64, 0.25 + i as f32));
        }
        let rows = program.generate_trace::<F>(&codes);
        assert_eq!(rows.len(), num_rows);
        assert!(!constraints_violated(&Stk::new(program), &rows), "multi-row honest trace must pass");

        let floor = f32_to_bf16(NORM_FLOOR).unwrap();
        for i in 0..num_operand_rows {
            let row = &codes[i * k..(i + 1) * k];
            let (ref_l2grid, ref_linf_raw) = row_norms(row).unwrap();
            let l2 = bf16_max(ref_l2grid, floor);
            let linf = bf16_max(ref_linf_raw, floor);
            let (alpha, beta) = derive_row_scales(l2, linf, R).unwrap();
            // Every live row of block i carries that row's scalars.
            let v: &RowScaleColumnsView<F> = rows[i * k].borrow();
            assert_eq!(to_u64(v.alpha_code) as u16, alpha, "alpha row {i}");
            assert_eq!(to_u64(v.beta_code) as u16, beta, "beta row {i}");
            assert_eq!(to_u64(v.operand_row_index) as usize, i, "operand_row_index row {i}");
        }
    }

    // ---- Entry-liveness gate (group L) ----

    /// Build the A-then-B row_scale program + trace for an `h x w x k` tile.
    fn liveness_trace(h: usize, w: usize, k: usize, a: &[u16], b: &[u16]) -> (Stk, Vec<[F; NUM_ROW_SCALE_COLUMNS]>) {
        let num_rows = ((h + w) * k).next_power_of_two();
        let program = RowScaleProgram::with_rows_ab(h, w, k, R, num_rows);
        let mut codes = a.to_vec();
        codes.extend_from_slice(b);
        let rows = program.generate_trace::<F>(&codes);
        (Stk::new(program), rows)
    }

    /// The oracle decision from the plaintext `check_shared_gates` (noise irrelevant to the scales).
    fn oracle_accepts(h: usize, w: usize, k: usize, a: &[u16], b: &[u16]) -> bool {
        use crate::v5::api::policy::check_shared_gates;
        use crate::v5::api::quantization::{noisy_quantize, row_norms};
        let norms_a: Vec<_> = (0..h).map(|i| row_norms(&a[i * k..i * k + k]).unwrap()).collect();
        let norms_b: Vec<_> = (0..w).map(|j| row_norms(&b[j * k..j * k + k]).unwrap()).collect();
        let ba = noisy_quantize(a, &vec![0u16; h * R], &vec![0u16; k * R], &norms_a, R).unwrap();
        let bb = noisy_quantize(b, &vec![0u16; w * R], &vec![0u16; k * R], &norms_b, R).unwrap();
        check_shared_gates(a, &ba, b, &bb, k).is_ok()
    }

    /// The circuit gate accepts EXACTLY the tiles `check_shared_gates` accepts: an honest
    /// spread-magnitude tile (no dead entries) passes, and a spike-dominated tile (dead fraction
    /// above eps_idle = 1/64 on the A side) makes the gate unsatisfiable.
    #[test]
    fn liveness_gate_matches_check_shared_gates() {
        let (h, w, k) = (4usize, 16usize, 64usize);

        // Honest: spread magnitudes, |x| < 4*rms everywhere -> 0 dead.
        let a: Vec<u16> = (0..h).flat_map(|i| sample_row(k, 100 + i as u64, 3.0)).collect();
        let b: Vec<u16> = (0..w).flat_map(|j| sample_row(k, 500 + j as u64, 3.0)).collect();
        assert!(oracle_accepts(h, w, k, &a, &b), "honest tile must be oracle-admissible");
        let (stk, rows) = liveness_trace(h, w, k, &a, &b);
        assert!(!constraints_violated(&stk, &rows), "the circuit gate must ACCEPT an honest tile");

        // Spike-dominated A: each A row has two large spikes among tiny values -> two dead entries
        // per row -> 8 dead > h*k/64 = 4, so side A fails liveness.
        let big = f32_to_fp16(60000.0).unwrap();
        let tiny = f32_to_fp16(0.001).unwrap();
        let mut a_spike = vec![tiny; h * k];
        for i in 0..h {
            a_spike[i * k] = big;
            a_spike[i * k + 1] = big;
        }
        assert!(!oracle_accepts(h, w, k, &a_spike, &b), "spike-dominated tile must be oracle-rejected");
        let (stk, rows) = liveness_trace(h, w, k, &a_spike, &b);
        assert!(constraints_violated(&stk, &rows), "the circuit gate must REJECT a spike-dominated tile");
    }

    /// A prover cannot clear the gate by understating the dead count (the inclusive accumulator and
    /// the RANGE16 gate slack have no satisfying witness for a failing tile), nor by tampering the
    /// per-element dead flag, the running accumulator, or the gate slack on an honest tile.
    #[test]
    fn tampered_liveness_witnesses_break_constraints() {
        let (h, w, k) = (4usize, 16usize, 64usize);
        let a: Vec<u16> = (0..h).flat_map(|i| sample_row(k, 11 + i as u64, 3.0)).collect();
        let b: Vec<u16> = (0..w).flat_map(|j| sample_row(k, 71 + j as u64, 3.0)).collect();
        let (stk, rows) = liveness_trace(h, w, k, &a, &b);
        assert!(!constraints_violated(&stk, &rows), "baseline honest tile must pass");
        let m = &ROW_SCALE_COL_MAP;
        let last = rows.len() - 1;
        // Flipping a per-element dead flag, the running accumulators, or the side mask on a live
        // row breaks the compare slack (L6) / the accumulator transition (L8) / the side pin (L7).
        for col in [m.dead, m.dead_run_a, m.dead_run_b, m.is_b_side] {
            let mut t = rows.clone();
            t[k][col] += F::ONE; // row k is the start of operand row 1 (a live row)
            assert!(constraints_violated(&stk, &t), "tampering column {col} must break a constraint");
        }
        // Inflating the final A gate slack (claiming more headroom than the count allows) breaks L9.
        let mut t = rows.clone();
        t[last][m.a_gate_slack_lo] += F::ONE;
        assert!(constraints_violated(&stk, &t), "a forged gate slack must break the last-row gate");
    }
}
