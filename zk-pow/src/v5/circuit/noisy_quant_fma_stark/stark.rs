//! Proves group G2 of the fused noisy quantization: the single-rounding f32 FMA
//! `noised = fma(af, X, t)`, bit-exact vs the Rust `af.mul_add(fp16_to_f32(raw), t)` of
//! [`crate::v5::api::quantization::noisy_quantize`], where `af = bf16_to_f32(alpha)`,
//! `X = fp16_to_f32(raw)`, and `t = RNE_f32(bf*N)` is group G1's output (carried in as
//! `(T_SIGN, T_MANT, T_EXP)`).
//!
//! # The datapath (mirrors the A100 accumulation AIR, adapted to RNE and 2 terms)
//!
//! `af*X` is the EXACT product with significand `Mp = M_a * M_x < 2^19` (both operands decode
//! losslessly to f32, and `19 < 24`, so the product is exact). `t` has a 24-bit significand `M_t`.
//! Both are normalized to 24-bit significands (`Mp_norm = Mp << (24-b_p)` via WIDTH32; `M_t` is
//! already 24-bit), given biased value-MSBs `P_MSB`, `T_MSB`. The window anchor
//! `eta = max(P_MSB, T_MSB)` over the nonzero terms; the window unit is `2^(eta-26)` (3 guard bits
//! below the 24-bit significand). Each term contributes `floor(sig * 8 / 2^rel)`,
//! `rel = eta - term_MSB` (FP16POW2; a term with `rel >= 27` falls entirely below the window and
//! contributes only to the sticky bit). The two signed aligned terms sum exactly to `|W| < 2^28`
//! (a far-gap sticky records any bits dropped below the window); the sum is renormalized into
//! `[2^23, 2^24)` (WIDTH32) and rounded once (RNE, ties to even) using the exact quarter-ulp
//! bracket / parity / sticky machinery. The result `noised` is emitted as an f32.
//!
//! # Soundness / completeness envelope
//!
//! `alpha` is a normal positive BF16; `raw` is any FP16 code (zero/subnormal/normal — FP16DECODE
//! serves all); `t` is G1's output (24-bit normal, or zero). The ONLY envelope restriction is that
//! the f32 **result** is normal (`noised_exp >= 1`) — a subnormal f32 result needs `|noised| <
//! 2^-126`, which the in-scheme datapath (alpha ~ the FP16 ceiling over a row norm) never produces;
//! the generator asserts it (so an out-of-envelope row is simply unprovable, exactly as
//! [`crate::v5::circuit::noise_stark`]). A tiny FP16 element is fine here: it yields a normal f32
//! `noised` (whose subsequent FP16 *cast* may be subnormal — that is group G3's concern, not G2's).
//! Every RNE rounding carries the full ties-to-even machinery (parity split + binade-bottom via the
//! `[2^23, 2^24)` normalized-range pin + the far-gap sticky), with NO quarter-ulp grinding slack.

use core::borrow::Borrow;
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

use super::columns::{FmaColumnsView, NUM_FMA_COLUMNS, NUM_FMA_PUBLIC_INPUTS};
use crate::v5::api::dtype::fp16_decode_fields;
use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// `noised_exp = eta_biased + w + carry - EXP_OFFSET` (= `(eta-512) + w + 100 + carry`); the
/// value-MSB bias is 512 and GUARD = 3 (both folded into the literal offsets 352/385/412 below).
const EXP_OFFSET: u64 = 412;

fn inv_f<F: RichField>(x: u64) -> F {
    if x == 0 { F::ZERO } else { F::from_canonical_u64(x).inverse() }
}

/// Inverse of the field difference `a - b` (zero when equal); handles `a < b` without u64 underflow.
fn inv_diff<F: RichField>(a: u64, b: u64) -> F {
    let d = F::from_canonical_u64(a) - F::from_canonical_u64(b);
    if d == F::ZERO { F::ZERO } else { d.inverse() }
}

fn bit_length(x: u64) -> u32 {
    64 - x.leading_zeros()
}

/// The committed geometry: just the element count.
#[derive(Clone, Debug)]
pub struct FmaProgram {
    pub num_elems: usize,
    /// Trace height (a power of two `>= num_elems`).
    pub num_rows: usize,
}

impl FmaProgram {
    pub fn new(num_elems: usize) -> Self {
        Self::with_rows(num_elems, num_elems.next_power_of_two().max(2))
    }

    /// Program at an explicit `num_rows` height (the batch pins an on-ladder height).
    pub fn with_rows(num_elems: usize, num_rows: usize) -> Self {
        assert!(num_elems >= 1, "at least one element");
        assert!(num_rows.is_power_of_two() && num_rows >= num_elems, "height covers the live elements");
        Self { num_elems, num_rows }
    }

    pub fn live_rows(&self) -> usize {
        self.num_elems
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn known_values<F: RichField>(&self) -> Vec<PolynomialValues<F>> {
        let num_rows = self.num_rows();
        let is_pad = (0..num_rows).map(|i| F::from_bool(i >= self.num_elems)).collect();
        vec![PolynomialValues::new(is_pad)]
    }

    /// Generates the trace. Each of `alpha` (BF16 code), `raw` (FP16 code), and `t` (as
    /// `(t_sign, t_mant=M_t, t_exp)` from G1) has one entry per live element. The emitted `noised`
    /// is bit-exact with `(bf16_to_f32(alpha)).mul_add(fp16_to_f32(raw), t_value)`.
    pub fn generate_trace<F: RichField>(
        &self,
        alpha: &[u16],
        raw: &[u16],
        t_sign: &[u64],
        t_mant: &[u64],
        t_exp: &[u64],
    ) -> Vec<[F; NUM_FMA_COLUMNS]> {
        let n = self.num_elems;
        assert!(alpha.len() == n && raw.len() == n && t_sign.len() == n && t_mant.len() == n && t_exp.len() == n);
        let num_rows = self.num_rows();
        let mut rows = Vec::with_capacity(num_rows);

        for e in 0..n {
            let mut v = FmaColumnsView::<F>::default();

            // ---- Decode alpha (normal positive BF16). ----
            let alpha_code = alpha[e];
            let alpha_exp = u64::from(alpha_code >> 7);
            let alpha_mant = u64::from(alpha_code & 0x7F);
            assert!((1..=254).contains(&alpha_exp), "envelope: alpha normal BF16");
            let m_a = 128 + alpha_mant;
            v.alpha = F::from_canonical_u16(alpha_code);
            v.alpha_exp = F::from_canonical_u64(alpha_exp);
            v.alpha_mant = F::from_canonical_u64(alpha_mant);

            // ---- Decode raw (FP16DECODE). ----
            let (x_sig, x_sign, x_eps_biased, x_is_zero) = fp16_decode_fields(raw[e]);
            v.raw = F::from_canonical_u16(raw[e]);
            v.x_sig = F::from_canonical_u64(x_sig);
            v.x_sign = F::from_canonical_u64(x_sign);
            v.x_eps_biased = F::from_canonical_u64(x_eps_biased);
            v.x_is_zero = F::from_canonical_u64(x_is_zero);

            // ---- t (from G1). ----
            let ts = t_sign[e];
            let mt = t_mant[e];
            let te = t_exp[e];
            let t_is_zero = mt == 0;
            v.t_sign = F::from_canonical_u64(ts);
            v.t_mant = F::from_canonical_u64(mt);
            v.t_exp = F::from_canonical_u64(te);
            v.t_is_zero = F::from_bool(t_is_zero);
            v.t_mant_inv = inv_f::<F>(mt);

            // ---- Exact product significand + 24-bit normalization. ----
            let p_is_zero = x_is_zero == 1;
            let mp = m_a * x_sig;
            v.p_is_zero = F::from_bool(p_is_zero);
            v.mp = F::from_canonical_u64(mp);
            let (bp, lift_p, mp_norm) = if p_is_zero {
                (0u64, 1u64, 0u64)
            } else {
                let b = u64::from(bit_length(mp));
                let lift = 1u64 << (24 - b);
                (b, lift, mp * lift)
            };
            v.bp = F::from_canonical_u64(bp);
            v.trunc_p = F::ONE; // 2^max(bp-24,0) = 1 for bp <= 24
            v.lift_p = F::from_canonical_u64(lift_p);
            v.mp_norm = F::from_canonical_u64(mp_norm);
            v.mp_norm_lo = F::from_canonical_u64(mp_norm & 0xFFFF);
            v.mp_norm_hi = F::from_canonical_u64(mp_norm >> 16);

            // Biased value-MSBs.
            let p_msb_b = alpha_exp + x_eps_biased + bp + 352; // (L_p + b_p - 1) + 512; only valid if P nonzero
            let t_msb_b = te + 385; // (t_exp - 127) + 512
            let eff_p = if p_is_zero { 0 } else { p_msb_b };
            let eff_t = if t_is_zero { 0 } else { t_msb_b };
            let eta = eff_p.max(eff_t);
            v.eta = F::from_canonical_u64(eta);
            let rel_p = eta - eff_p;
            let rel_t = eta - eff_t;
            v.rel_p = F::from_canonical_u64(rel_p);
            v.rel_t = F::from_canonical_u64(rel_t);
            let far_p = rel_p >= 27;
            let far_t = rel_t >= 27;
            v.far_p = F::from_bool(far_p);
            v.far_p_slack = F::from_canonical_u64(if far_p { rel_p - 27 } else { 26 - rel_p });
            v.far_t = F::from_bool(far_t);
            v.far_t_slack = F::from_canonical_u64(if far_t { rel_t - 27 } else { 26 - rel_t });
            let active_p = !far_p && !p_is_zero;
            let active_t = !far_t && !t_is_zero;
            v.active_p = F::from_bool(active_p);
            v.active_t = F::from_bool(active_t);

            // ---- Alignment. ----
            let (pow_p, aligned_p, rem_p) = if active_p {
                let pow = 1u64 << rel_p;
                let a = (mp_norm * 8) / pow;
                (pow, a, mp_norm * 8 - a * pow)
            } else {
                (1u64, 0u64, 0u64)
            };
            v.pow_p = F::from_canonical_u64(pow_p);
            v.aligned_p = F::from_canonical_u64(aligned_p);
            v.aligned_p_lo = F::from_canonical_u64(aligned_p & 0xFFFF);
            v.aligned_p_hi = F::from_canonical_u64(aligned_p >> 16);
            v.rem_p = F::from_canonical_u64(rem_p);
            v.rem_p_lo = F::from_canonical_u64(rem_p & 0xFFFF);
            v.rem_p_hi = F::from_canonical_u64(rem_p >> 16);
            let rem_p_bound = if active_p { pow_p - 1 - rem_p } else { 0 };
            v.rem_p_bound = F::from_canonical_u64(rem_p_bound);
            v.rem_p_bound_lo = F::from_canonical_u64(rem_p_bound & 0xFFFF);
            v.rem_p_bound_hi = F::from_canonical_u64(rem_p_bound >> 16);

            let (pow_t, aligned_t, rem_t) = if active_t {
                let pow = 1u64 << rel_t;
                let a = (mt * 8) / pow;
                (pow, a, mt * 8 - a * pow)
            } else {
                (1u64, 0u64, 0u64)
            };
            v.pow_t = F::from_canonical_u64(pow_t);
            v.aligned_t = F::from_canonical_u64(aligned_t);
            v.aligned_t_lo = F::from_canonical_u64(aligned_t & 0xFFFF);
            v.aligned_t_hi = F::from_canonical_u64(aligned_t >> 16);
            v.rem_t = F::from_canonical_u64(rem_t);
            v.rem_t_lo = F::from_canonical_u64(rem_t & 0xFFFF);
            v.rem_t_hi = F::from_canonical_u64(rem_t >> 16);
            let rem_t_bound = if active_t { pow_t - 1 - rem_t } else { 0 };
            v.rem_t_bound = F::from_canonical_u64(rem_t_bound);
            v.rem_t_bound_lo = F::from_canonical_u64(rem_t_bound & 0xFFFF);
            v.rem_t_bound_hi = F::from_canonical_u64(rem_t_bound >> 16);

            // ---- Sticky bits. ----
            let rem_p_nz = rem_p != 0;
            let rem_t_nz = rem_t != 0;
            v.rem_p_nz = F::from_bool(rem_p_nz);
            v.rem_p_nz_inv = inv_f::<F>(rem_p);
            v.rem_t_nz = F::from_bool(rem_t_nz);
            v.rem_t_nz_inv = inv_f::<F>(rem_t);
            let or_p = far_p || rem_p_nz;
            let or_t = far_t || rem_t_nz;
            v.or_p = F::from_bool(or_p);
            v.or_t = F::from_bool(or_t);
            let sticky_p = !p_is_zero && or_p;
            let sticky_t = !t_is_zero && or_t;
            v.sticky_p = F::from_bool(sticky_p);
            v.sticky_t = F::from_bool(sticky_t);
            let far_sticky = sticky_p || sticky_t;
            v.far_sticky = F::from_bool(far_sticky);

            // ---- Signed window sum. ----
            let term_p: i128 = if x_sign == 1 { -(aligned_p as i128) } else { aligned_p as i128 };
            let term_t: i128 = if ts == 1 { -(aligned_t as i128) } else { aligned_t as i128 };
            let w_signed = term_p + term_t;
            let w_sign = w_signed < 0;
            let w_abs = w_signed.unsigned_abs() as u64;
            let w_is_zero = w_signed == 0;
            v.w_sign = F::from_bool(w_sign);
            v.w_abs = F::from_canonical_u64(w_abs);
            v.w_abs_lo = F::from_canonical_u64(w_abs & 0xFFFF);
            v.w_abs_hi = F::from_canonical_u64(w_abs >> 16);
            v.w_is_zero = F::from_bool(w_is_zero);
            v.w_abs_inv = inv_f::<F>(w_abs);

            // ---- Renormalize + RNE. ----
            let (ww, trunc_w, lift_w, m_rz, rz_rem) = if w_is_zero {
                (0u64, 1u64, 1u64, 0u64, 0u64)
            } else {
                let w = u64::from(bit_length(w_abs));
                let tr = 1u64 << w.saturating_sub(24);
                let lf = 1u64 << 24u64.saturating_sub(w);
                let m = w_abs * lf / tr;
                (w, tr, lf, m, w_abs * lf - m * tr)
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
            let eq = 2 * rz_rem == trunc_w;
            v.gt = F::from_bool(gt);
            v.gt_slack = F::from_canonical_u64(if gt { 2 * rz_rem - trunc_w - 1 } else { trunc_w - 2 * rz_rem });
            v.eq = F::from_bool(eq);
            v.eq_inv = inv_diff::<F>(trunc_w, 2 * rz_rem);
            // Sign-aware sticky: the single sub-window sticky term (the non-dominant one) pushes |W|
            // up only if it shares W's sign; an opposite-sign remainder (subtraction) makes |W|
            // smaller, so a tie must round DOWN. `x_sign`/`ts` are the term signs; `w_sign` is W's.
            let ws = u64::from(w_sign);
            let sticky_up =
                (sticky_p && x_sign == ws) || (sticky_t && ts == ws);
            v.sticky_up = F::from_bool(sticky_up);
            let or_rs = sticky_up || (!far_sticky && rz_parity == 1);
            v.or_rs = F::from_bool(or_rs);
            let round_up = gt || (eq && or_rs);
            v.round_up = F::from_bool(round_up);
            let m_out = m_rz + u64::from(round_up);
            let carry = m_out == (1 << 24);
            v.carry = F::from_bool(carry);
            v.carry_inv = inv_f::<F>((1 << 24) - m_out);
            let m_out_final = if carry { 1 << 23 } else { m_out };

            // ---- Result f32. ----
            let (noised_exp, noised_mant, noised_sign) = if w_is_zero {
                (0u64, 0u64, 0u64)
            } else {
                let exp = eta + ww + u64::from(carry) - EXP_OFFSET;
                assert!((1..=254).contains(&exp), "envelope: f32-normal result (noised_exp {exp})");
                (exp, m_out_final - (1 << 23), u64::from(w_sign))
            };
            v.noised_exp = F::from_canonical_u64(noised_exp);
            v.noised_sign = F::from_canonical_u64(noised_sign);
            v.noised_mant = F::from_canonical_u64(noised_mant);
            let nbits = (noised_sign << 31) | (noised_exp << 23) | noised_mant;
            v.noised_lo = F::from_canonical_u64(nbits & 0xFFFF);
            v.noised_hi = F::from_canonical_u64(nbits >> 16);

            // ---- Bit-exact cross-check vs the reference f32 FMA. ----
            let af = crate::v4::api::dtype::bf16_to_f32(alpha_code);
            let xval = crate::v5::api::dtype::fp16_to_f32(raw[e]);
            let tval = if t_is_zero {
                if ts == 1 { -0.0f32 } else { 0.0f32 }
            } else {
                f32::from_bits(((ts as u32) << 31) | ((te as u32) << 23) | ((mt - (1 << 23)) as u32))
            };
            let reference = af.mul_add(xval, tval);
            let ref_bits = reference.to_bits();
            if w_is_zero {
                debug_assert_eq!(ref_bits & 0x7FFF_FFFF, 0, "reference should be zero when W == 0");
            } else {
                debug_assert_eq!(nbits as u32, ref_bits, "noised {nbits:#010x} vs reference {ref_bits:#010x}");
            }

            rows.push(v.into());
        }

        for _ in n..num_rows {
            rows.push(FmaColumnsView::<F> { is_pad: F::ONE, ..Default::default() }.into());
        }
        rows
    }
}

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

/// `a OR b` for booleans, committed to `out`: `out = a + b - a*b`, gated.
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

pub(crate) fn eval_fma_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_FMA_COLUMNS, NUM_FMA_PUBLIC_INPUTS>,
    eval: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_FMA_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &FmaColumnsView<V> = lv.borrow();

    let one = eval.u64(1);
    let live = eval.sub(one, lv.is_pad);
    eval.constraint_bool(lv.is_pad);

    let c128 = eval.u64(128);
    let c8 = eval.u64(8);
    let c2_16 = eval.u64(1 << 16);
    let c2_23 = eval.u64(1 << 23);
    let c2_24 = eval.u64(1 << 24);
    let c2_31 = eval.u64(1 << 31);
    let two = eval.u64(2);

    let gated = |eval: &mut E, expr: V| {
        let c = eval.mul(live, expr);
        eval.constraint(c);
    };

    // ---- alpha decode: code = alpha_exp*128 + alpha_mant. ----
    let beta_rec = eval.mad(lv.alpha_exp, c128, lv.alpha_mant);
    let d = eval.sub(lv.alpha, beta_rec);
    gated(eval, d);

    // ---- booleans. ----
    eval.constraint_bool(lv.x_sign);
    eval.constraint_bool(lv.x_is_zero);
    eval.constraint_bool(lv.t_sign);
    eval.constraint_bool(lv.far_p);
    eval.constraint_bool(lv.far_t);
    eval.constraint_bool(lv.active_p);
    eval.constraint_bool(lv.active_t);
    eval.constraint_bool(lv.rem_p_nz);
    eval.constraint_bool(lv.rem_t_nz);
    eval.constraint_bool(lv.or_p);
    eval.constraint_bool(lv.or_t);
    eval.constraint_bool(lv.sticky_p);
    eval.constraint_bool(lv.sticky_t);
    eval.constraint_bool(lv.far_sticky);
    eval.constraint_bool(lv.w_sign);
    eval.constraint_bool(lv.w_is_zero);
    eval.constraint_bool(lv.gt);
    eval.constraint_bool(lv.eq);
    eval.constraint_bool(lv.or_rs);
    eval.constraint_bool(lv.round_up);
    eval.constraint_bool(lv.carry);
    eval.constraint_bool(lv.p_is_zero);

    // ---- t_is_zero = [M_t == 0]; p_is_zero = x_is_zero. ----
    is_zero_flag(eval, lv.t_mant, lv.t_mant_inv, lv.t_is_zero, live);
    let d = eval.sub(lv.p_is_zero, lv.x_is_zero);
    gated(eval, d);

    // ---- product significand Mp = (128 + alpha_mant) * x_sig. ----
    let m_a = eval.add(c128, lv.alpha_mant);
    let mp_expect = eval.mul(m_a, lv.x_sig);
    let d = eval.sub(lv.mp, mp_expect);
    gated(eval, d);
    // Mp_norm = Mp * lift_p (WIDTH32 pins lift_p = 2^(24-b_p), and the [2^23,2^24) range of
    // Mp_norm pins b_p = bitlen(Mp)). Active only on nonzero-P rows.
    let nonzero_p = eval.sub(one, lv.p_is_zero);
    let mp_norm_expect = eval.mul(lv.mp, lv.lift_p);
    let d = eval.sub(lv.mp_norm, mp_norm_expect);
    // Degree-3 (nonzero_p * degree-2); 0 on padding rows (all columns 0), so no extra live gate.
    let c = eval.mul(nonzero_p, d);
    eval.constraint(c);
    // Zero-P rows: Mp_norm = 0.
    let c = eval.mul(lv.p_is_zero, lv.mp_norm);
    gated(eval, c);
    // Mp_norm limb reconstruction.
    let mp_norm_rec = eval.mad(lv.mp_norm_hi, c2_16, lv.mp_norm_lo);
    let d = eval.sub(lv.mp_norm, mp_norm_rec);
    gated(eval, d);

    // ---- eta / rel. EFF = (1 - is_zero)*MSB_biased; rel = eta - EFF. ----
    // P_MSB_biased = alpha_exp + x_eps_biased + b_p + 352.
    let c352 = eval.u64(352);
    let p_msb = {
        let s = eval.add(lv.alpha_exp, lv.x_eps_biased);
        let s = eval.add(s, lv.bp);
        eval.add(s, c352)
    };
    let eff_p = eval.mul(nonzero_p, p_msb);
    let rel_p_expect = eval.sub(lv.eta, eff_p);
    let d = eval.sub(lv.rel_p, rel_p_expect);
    gated(eval, d);
    // T_MSB_biased = t_exp + 385.
    let c385 = eval.u64(385);
    let t_msb = eval.add(lv.t_exp, c385);
    let nonzero_t = eval.sub(one, lv.t_is_zero);
    let eff_t = eval.mul(nonzero_t, t_msb);
    let rel_t_expect = eval.sub(lv.eta, eff_t);
    let d = eval.sub(lv.rel_t, rel_t_expect);
    gated(eval, d);
    // Attainment: rel_p * rel_t = 0 (one term defines eta; rel_p, rel_t >= 0 via RANGE16 pin eta
    // as the max). Both-zero -> eta = 0.
    let rr = eval.mul(lv.rel_p, lv.rel_t);
    gated(eval, rr);

    // ---- far flags + two-sided slacks: far*(rel-27) + (1-far)*(26-rel) >= 0. ----
    let c27 = eval.u64(27);
    let c26 = eval.u64(26);
    for (far, rel, slack) in [(lv.far_p, lv.rel_p, lv.far_p_slack), (lv.far_t, lv.rel_t, lv.far_t_slack)] {
        let rel_m27 = eval.sub(rel, c27);
        let hi = eval.mul(far, rel_m27);
        let not_far = eval.sub(one, far);
        let c26_m_rel = eval.sub(c26, rel);
        let lo = eval.mul(not_far, c26_m_rel);
        let expect = eval.add(hi, lo);
        let d = eval.sub(slack, expect);
        gated(eval, d);
    }
    // active = (1 - far)*(1 - is_zero).
    let d = {
        let a = eval.sub(one, lv.far_p);
        let ap = eval.mul(a, nonzero_p);
        eval.sub(lv.active_p, ap)
    };
    gated(eval, d);
    let d = {
        let a = eval.sub(one, lv.far_t);
        let ap = eval.mul(a, nonzero_t);
        eval.sub(lv.active_t, ap)
    };
    gated(eval, d);

    // ---- alignment Euclidean floors: active rows: sig*8 = aligned*pow + rem, 0 <= rem < pow. ----
    // (pow is FP16POW2(rel) on active rows, 1 otherwise.)
    #[allow(clippy::type_complexity)]
    let align_terms: [(V, V, V, V, V, V, V, V, V, V, V, V, V); 2] = [
        (
            lv.active_p, lv.mp_norm, lv.pow_p, lv.aligned_p, lv.aligned_p_lo, lv.aligned_p_hi, lv.rem_p, lv.rem_p_lo,
            lv.rem_p_hi, lv.rem_p_bound, lv.rem_p_bound_lo, lv.rem_p_bound_hi, lv.far_p,
        ),
        (
            lv.active_t, lv.t_mant, lv.pow_t, lv.aligned_t, lv.aligned_t_lo, lv.aligned_t_hi, lv.rem_t, lv.rem_t_lo,
            lv.rem_t_hi, lv.rem_t_bound, lv.rem_t_bound_lo, lv.rem_t_bound_hi, lv.far_t,
        ),
    ];
    for (active, sig, pow, aligned, aligned_lo, aligned_hi, rem, rem_lo, rem_hi, rem_bound, rem_bound_lo, rem_bound_hi, _far) in
        align_terms
    {
        let sig8 = eval.mul(c8, sig);
        let ap = eval.mul(aligned, pow);
        let floor = eval.sub(sig8, ap);
        let floor = eval.sub(floor, rem);
        // Degree-3 (active * degree-2); 0 on padding (active = 0 there), so no extra live gate.
        let c = eval.mul(active, floor);
        eval.constraint(c);
        // rem + rem_bound + 1 = pow (active rows).
        let rem_sum = eval.add(rem, rem_bound);
        let rem_sum = eval.add(rem_sum, one);
        let rem_id = eval.sub(rem_sum, pow);
        let c = eval.mul(active, rem_id);
        gated(eval, c);
        // Inactive rows: aligned = 0, rem = 0.
        let not_active = eval.sub(one, active);
        let c = eval.mul(not_active, aligned);
        gated(eval, c);
        let c = eval.mul(not_active, rem);
        gated(eval, c);
        // Limb reconstructions (aligned, rem, rem_bound).
        let rec = eval.mad(aligned_hi, c2_16, aligned_lo);
        let d = eval.sub(aligned, rec);
        gated(eval, d);
        let rec = eval.mad(rem_hi, c2_16, rem_lo);
        let d = eval.sub(rem, rec);
        gated(eval, d);
        let rec = eval.mad(rem_bound_hi, c2_16, rem_bound_lo);
        let d = eval.sub(rem_bound, rec);
        gated(eval, d);
    }

    // ---- sticky bits. rem_*_nz = [rem != 0], so [rem == 0] = 1 - rem_*_nz. ----
    let rem_p_zero = eval.sub(one, lv.rem_p_nz);
    is_zero_flag(eval, lv.rem_p, lv.rem_p_nz_inv, rem_p_zero, live);
    let rem_t_zero = eval.sub(one, lv.rem_t_nz);
    is_zero_flag(eval, lv.rem_t, lv.rem_t_nz_inv, rem_t_zero, live);
    // or_p = far_p OR rem_p_nz; or_t likewise.
    or_flag(eval, lv.far_p, lv.rem_p_nz, lv.or_p, live);
    or_flag(eval, lv.far_t, lv.rem_t_nz, lv.or_t, live);
    // sticky = (1 - is_zero)*or.
    let st = eval.mul(nonzero_p, lv.or_p);
    let d = eval.sub(lv.sticky_p, st);
    gated(eval, d);
    let st = eval.mul(nonzero_t, lv.or_t);
    let d = eval.sub(lv.sticky_t, st);
    gated(eval, d);
    or_flag(eval, lv.sticky_p, lv.sticky_t, lv.far_sticky, live);

    // ---- signed window sum: |W| = (1-2*w_sign) * ((1-2*x_sign)*aligned_p + (1-2*t_sign)*aligned_t). ----
    let sgn = |eval: &mut E, s: V, a: V| {
        let two_s = eval.mul(two, s);
        let factor = eval.sub(one, two_s);
        eval.mul(factor, a)
    };
    let tp = sgn(eval, lv.x_sign, lv.aligned_p);
    let tt = sgn(eval, lv.t_sign, lv.aligned_t);
    let w_signed = eval.add(tp, tt);
    let two_ws = eval.mul(two, lv.w_sign);
    let wfac = eval.sub(one, two_ws);
    let w_abs_expect = eval.mul(wfac, w_signed);
    // Degree-3 (sign factor * degree-2 signed sum); 0 on padding, so no extra live gate.
    let d = eval.sub(lv.w_abs, w_abs_expect);
    eval.constraint(d);
    is_zero_flag(eval, lv.w_abs, lv.w_abs_inv, lv.w_is_zero, live);
    // A zero sum is +0 (sign pinned off).
    let c = eval.mul(lv.w_is_zero, lv.w_sign);
    gated(eval, c);
    // |W| limb reconstruction.
    let w_rec = eval.mad(lv.w_abs_hi, c2_16, lv.w_abs_lo);
    let d = eval.sub(lv.w_abs, w_rec);
    gated(eval, d);

    // ---- renormalize |W| (WIDTH32) + RNE. nonzero-W rows only. ----
    let nz_w = eval.sub(one, lv.w_is_zero);
    // |W| * lift_w = M_rz * trunc_w + rz_rem, 0 <= rz_rem < trunc_w.
    let lifted = eval.mul(lv.w_abs, lv.lift_w);
    let truncated = eval.mul(lv.m_rz, lv.trunc_w);
    let diff = eval.sub(lifted, truncated);
    let diff = eval.sub(diff, lv.rz_rem);
    // Degree-3 (nz_w * degree-2); 0 on padding (all columns 0), so no extra live gate.
    let c = eval.mul(nz_w, diff);
    eval.constraint(c);
    let rem_sum = eval.add(lv.rz_rem, lv.rz_rem_bound);
    let rem_sum = eval.add(rem_sum, one);
    let rem_id = eval.sub(rem_sum, lv.trunc_w);
    let c = eval.mul(nz_w, rem_id);
    gated(eval, c);
    // M_rz limb reconstruction (hi in [128,256) pinned by ctl -> M_rz in [2^23, 2^24)).
    let m_rz_rec = eval.mad(lv.m_rz_hi, c2_16, lv.m_rz_lo);
    let d = eval.sub(lv.m_rz, m_rz_rec);
    gated(eval, d);
    // parity split on M_rz low limb.
    eval.constraint_bool(lv.rz_parity);
    let two_half = eval.mul(two, lv.rz_half);
    let recomposed = eval.add(two_half, lv.rz_parity);
    let d = eval.sub(lv.m_rz_lo, recomposed);
    gated(eval, d);
    // gt = [2*rz_rem > trunc_w], two-sided slack.
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
        gated(eval, d);
    }
    // eq = [trunc_w - 2*rz_rem == 0].
    let eq_arg = eval.sub(lv.trunc_w, two_rem);
    is_zero_flag(eval, eq_arg, lv.eq_inv, lv.eq, live);
    // Sign-aware sticky: STICKY_UP = STICKY_P*[X_SIGN==W_SIGN] + STICKY_T*[T_SIGN==W_SIGN]. At most
    // one of STICKY_P/STICKY_T is set (the dominant term has rel=0, hence no sub-window remainder),
    // so the sum is boolean. `[a==b]` for booleans is `1 - (a-b)^2`.
    eval.constraint_bool(lv.sticky_up);
    {
        let match_flag = |eval: &mut E, a: V, b: V| {
            let diff = eval.sub(a, b);
            let sq = eval.mul(diff, diff);
            eval.sub(one, sq)
        };
        let mp = match_flag(eval, lv.x_sign, lv.w_sign);
        let mt = match_flag(eval, lv.t_sign, lv.w_sign);
        let up_p = eval.mul(lv.sticky_p, mp);
        let up_t = eval.mul(lv.sticky_t, mt);
        let up = eval.add(up_p, up_t);
        let d = eval.sub(lv.sticky_up, up);
        // Already 0 on padding (all operand/sticky columns are 0 there), so no live gate is needed —
        // which also keeps this constraint at degree 3 (sticky * (1 - (sign-diff)^2)).
        eval.constraint(d);
    }
    // or_rs = STICKY_UP + (1 - FAR_STICKY)*RZ_PARITY (the two summands are disjoint, so it is boolean):
    // a same-sign remainder rounds the tie up; with no remainder, ties-to-even uses the parity.
    {
        let not_far = eval.sub(one, lv.far_sticky);
        let parity_term = eval.mul(not_far, lv.rz_parity);
        let expect = eval.add(lv.sticky_up, parity_term);
        let d = eval.sub(lv.or_rs, expect);
        gated(eval, d);
    }
    // round_up = gt + eq*or_rs.
    let eqor = eval.mul(lv.eq, lv.or_rs);
    let ru = eval.add(lv.gt, eqor);
    let d = eval.sub(lv.round_up, ru);
    gated(eval, d);
    // carry = [2^24 - (M_rz + round_up) == 0].
    let m_out = eval.add(lv.m_rz, lv.round_up);
    let carry_arg = eval.sub(c2_24, m_out);
    is_zero_flag(eval, carry_arg, lv.carry_inv, lv.carry, live);

    // ---- result f32. nonzero-W: noised_exp = eta + w + carry - 412; mant = M_out_final - 2^23. ----
    // M_out_final = M_rz + round_up - carry*2^23.
    let carry_term = eval.mul(lv.carry, c2_23);
    let m_out_final = eval.sub(m_out, carry_term);
    // noised_exp.
    let c412 = eval.u64(EXP_OFFSET);
    let exp_core = {
        let s = eval.add(lv.eta, lv.ww);
        let s = eval.add(s, lv.carry);
        eval.sub(s, c412)
    };
    let exp_expect = eval.mul(nz_w, exp_core);
    let d = eval.sub(lv.noised_exp, exp_expect);
    gated(eval, d);
    // noised_mant = (1 - w_is_zero)*(M_out_final - 2^23).
    let mant_core = eval.sub(m_out_final, c2_23);
    let mant_expect = eval.mul(nz_w, mant_core);
    let d = eval.sub(lv.noised_mant, mant_expect);
    gated(eval, d);
    // noised_sign = (1 - w_is_zero)*w_sign.
    let sign_expect = eval.mul(nz_w, lv.w_sign);
    let d = eval.sub(lv.noised_sign, sign_expect);
    gated(eval, d);
    // noised limb reconstruction: lo + 2^16*hi = sign*2^31 + exp*2^23 + mant.
    let fields = {
        let s = eval.mul(lv.noised_sign, c2_31);
        let e = eval.mul(lv.noised_exp, c2_23);
        let se = eval.add(s, e);
        eval.add(se, lv.noised_mant)
    };
    let limbs = eval.mad(lv.noised_hi, c2_16, lv.noised_lo);
    let d = eval.sub(fields, limbs);
    gated(eval, d);
}

/// FP16 single-rounding FMA AIR (group G2). A CTL party (`requires_ctls`).
#[derive(Clone, Debug)]
pub struct NoisyQuantFmaStark<F: RichField + Extendable<D>, const D: usize> {
    pub program: FmaProgram,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> NoisyQuantFmaStark<F, D> {
    pub fn new(program: FmaProgram) -> Self {
        Self { program, _phantom: PhantomData }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for NoisyQuantFmaStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_FMA_COLUMNS, NUM_FMA_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_FMA_COLUMNS, NUM_FMA_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_fma_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_fma_constraints(vars, &mut evaluator);
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
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};
    use starky::util::trace_rows_to_poly_values;

    use super::super::columns::FMA_COL_MAP;
    use super::super::ctl::fma_lut_lookups;
    use super::*;
    use crate::v5::api::dtype::{f32_to_fp16, fp16_to_f32};
    use crate::v4::api::dtype::{bf16_to_f32, f32_to_bf16};
    use crate::v4::circuit::luts::LutTable;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type Stk = NoisyQuantFmaStark<F, D>;

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    /// Decode a finite f32 `t` into `(sign, M_t, exp)` as group G1 emits it.
    fn t_fields(t: f32) -> (u64, u64, u64) {
        let b = t.to_bits();
        let sign = u64::from(b >> 31);
        let exp = u64::from((b >> 23) & 0xFF);
        if exp == 0 {
            (sign, 0, 0) // zero (envelope: t is G1's output = normal or zero)
        } else {
            (sign, (1u64 << 23) + u64::from(b & 0x7F_FFFF), exp)
        }
    }

    fn trace_from(items: &[(u16, u16, f32)]) -> (FmaProgram, Vec<[F; NUM_FMA_COLUMNS]>) {
        let n = items.len();
        let program = FmaProgram::new(n);
        let alpha: Vec<u16> = items.iter().map(|x| x.0).collect();
        let raw: Vec<u16> = items.iter().map(|x| x.1).collect();
        let tf: Vec<(u64, u64, u64)> = items.iter().map(|x| t_fields(x.2)).collect();
        let ts: Vec<u64> = tf.iter().map(|x| x.0).collect();
        let tm: Vec<u64> = tf.iter().map(|x| x.1).collect();
        let te: Vec<u64> = tf.iter().map(|x| x.2).collect();
        let rows = program.generate_trace::<F>(&alpha, &raw, &ts, &tm, &te);
        (program, rows)
    }

    fn constraints_violated(stark: &Stk, rows: &[[F; NUM_FMA_COLUMNS]]) -> bool {
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

    /// Deterministic in-envelope sample: normal `alpha`, finite `raw`, moderate `t` (occasionally 0).
    fn sample(seed: u64) -> (u16, u16, f32) {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s
        };
        let alpha = f32_to_bf16(0.5 + (next() % 64) as f32 * 0.5).unwrap();
        // A finite FP16 code (exponent field != 0x1F): signed, any mantissa.
        let raw = {
            let sign = ((next() & 1) as u16) << 15;
            let exp = ((next() % 31) as u16) << 10; // [0, 30]
            let man = (next() % 1024) as u16;
            sign | exp | man
        };
        let t = if next() % 7 == 0 {
            0.0
        } else {
            let mag = 2f32.powi(((next() % 20) as i32) - 8) * (1.0 + (next() % 1000) as f32 / 1000.0);
            if next() & 1 == 0 { mag } else { -mag }
        };
        (alpha, raw, t)
    }

    fn ref_noised(alpha: u16, raw: u16, t: f32) -> f32 {
        bf16_to_f32(alpha).mul_add(fp16_to_f32(raw), t)
    }

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        let items: Vec<_> = (0..60).map(sample).collect();
        let (program, rows) = trace_from(&items);
        assert_eq!(rows.len(), 64);
        let known = program.known_values::<F>();
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(known[0].values[r], row[FMA_COL_MAP.is_pad], "is_pad row {r}");
        }
        assert!(!constraints_violated(&Stk::new(program), &rows), "honest trace violated a constraint");
    }

    #[test]
    fn fma_bit_exact_vs_reference_fma() {
        // Cover: cancellation, far gaps, zero t, zero X, carry, power-of-two result, random.
        let mut items: Vec<(u16, u16, f32)> = Vec::new();
        // Cancellation: af*X = +k, t = -k (exact).
        items.push((f32_to_bf16(2.0).unwrap(), f32_to_fp16(3.0).unwrap(), -6.0)); // 6 - 6 = 0
        items.push((f32_to_bf16(2.0).unwrap(), f32_to_fp16(3.0).unwrap(), -5.75)); // near-cancel
        // Far gap: af*X ~ 1e3, t ~ 1e-6.
        items.push((f32_to_bf16(8.0).unwrap(), f32_to_fp16(100.0).unwrap(), 1e-6));
        items.push((f32_to_bf16(8.0).unwrap(), f32_to_fp16(100.0).unwrap(), -1e-6));
        // t dominates, P tiny.
        items.push((f32_to_bf16(1.0).unwrap(), f32_to_fp16(2f32.powi(-14)).unwrap(), 1234.5));
        // Zero t.
        items.push((f32_to_bf16(3.0).unwrap(), f32_to_fp16(5.0).unwrap(), 0.0));
        // Zero X (raw = 0): noised = t.
        items.push((f32_to_bf16(3.0).unwrap(), 0u16, 7.25));
        // Power-of-two result.
        items.push((f32_to_bf16(1.0).unwrap(), f32_to_fp16(1.0).unwrap(), 0.0));
        items.push((f32_to_bf16(1.0).unwrap(), f32_to_fp16(1.0).unwrap(), 3.0)); // 1+3 = 4
        // Random coverage.
        for s in 0..400 {
            items.push(sample(s));
        }
        // Keep only in-envelope rows (f32-normal result) to avoid the generator's envelope panic.
        items.retain(|&(a, r, t)| {
            let n = ref_noised(a, r, t);
            let e = (n.abs().to_bits() >> 23) & 0xFF;
            n != 0.0 && (1..=254).contains(&e) || n == 0.0
        });
        let (_program, rows) = trace_from(&items);
        for (e, &(a, r, t)) in items.iter().enumerate() {
            let reference = ref_noised(a, r, t);
            let got_lo = to_u64(rows[e][FMA_COL_MAP.noised_lo]);
            let got_hi = to_u64(rows[e][FMA_COL_MAP.noised_hi]);
            let got = (got_hi << 16) | got_lo;
            if reference == 0.0 {
                assert_eq!(got & 0x7FFF_FFFF, 0, "elem {e}: expected zero noised");
            } else {
                assert_eq!(got as u32, reference.to_bits(), "elem {e}: noised mismatch (alpha={a:#06x} raw={r:#06x} t={t})");
            }
        }
    }

    /// Bit-exact against the real `noisy_quantize`: derive `t = bf*N` and `noised` from the AIR, and
    /// check the AIR's `noised` reproduces the kernel's per-element FMA.
    #[test]
    fn fma_bit_exact_vs_noisy_quantize() {
        use crate::v5::api::quantization::{noisy_quantize, row_norms};
        const R: usize = 32;
        let k = 48usize;
        let rows_in: Vec<u16> = (0..k)
            .map(|j| f32_to_fp16(if j == 0 { 7.5 } else { 1.0 + (j % 5) as f32 * 0.25 }).unwrap())
            .collect();
        let e: Vec<u16> = (0..R).map(|t| f32_to_fp16(((t % 5) as f32 - 2.0) * 8.0).unwrap()).collect();
        let f: Vec<u16> = (0..k * R).map(|t| f32_to_fp16(((t % 7) as f32 - 3.0) * 8.0).unwrap()).collect();
        let norms = [row_norms(&rows_in).unwrap()];
        let built = noisy_quantize(&rows_in, &e, &f, &norms, R).unwrap();
        let noise = crate::v5::api::accumulate::a100_matmul(&e, &f, None, 1, k, R);
        let bf = bf16_to_f32(built.beta[0]);
        let af = bf16_to_f32(built.alpha[0]);

        let items: Vec<(u16, u16, f32)> =
            (0..k).map(|j| (built.alpha[0], rows_in[j], bf * noise[j])).collect();
        let (_program, trace) = trace_from(&items);
        for j in 0..k {
            let noised_ref = af.mul_add(fp16_to_f32(rows_in[j]), bf * noise[j]);
            let got = (to_u64(trace[j][FMA_COL_MAP.noised_hi]) << 16) | to_u64(trace[j][FMA_COL_MAP.noised_lo]);
            assert_eq!(got as u32, noised_ref.to_bits(), "noised mismatch elem {j}");
            // And the cast of the AIR-derived noised matches the kernel's published FP16 code.
            let out = f32_to_fp16(f32::from_bits(got as u32)).unwrap();
            assert_eq!(out, built.noised_part[j], "end-to-end FP16 code mismatch elem {j}");
        }
    }

    #[test]
    fn tampered_traces_fail() {
        let items: Vec<_> = (0..16).map(sample).collect();
        let (program, rows) = trace_from(&items);
        let stark = Stk::new(program);
        assert!(!constraints_violated(&stark, &rows), "baseline honest trace must pass");
        for (name, col) in [
            ("alpha", FMA_COL_MAP.alpha),
            ("x_sig", FMA_COL_MAP.x_sig),
            ("mp", FMA_COL_MAP.mp),
            ("mp_norm", FMA_COL_MAP.mp_norm),
            ("eta", FMA_COL_MAP.eta),
            ("rel_p", FMA_COL_MAP.rel_p),
            ("aligned_p", FMA_COL_MAP.aligned_p),
            ("aligned_t", FMA_COL_MAP.aligned_t),
            ("w_abs", FMA_COL_MAP.w_abs),
            ("w_sign", FMA_COL_MAP.w_sign),
            ("m_rz", FMA_COL_MAP.m_rz),
            ("round_up", FMA_COL_MAP.round_up),
            ("noised_exp", FMA_COL_MAP.noised_exp),
            ("noised_mant", FMA_COL_MAP.noised_mant),
            ("noised_lo", FMA_COL_MAP.noised_lo),
        ] {
            let mut forged = rows.clone();
            forged[0][col] += F::ONE;
            assert!(constraints_violated(&stark, &forged), "{name} tamper undetected");
        }
    }

    /// RNE uniqueness (ties-to-even): find a row whose final round is an exact tie (`eq = 1`) with
    /// `round_up = 0` (even significand, no sticky); forging `round_up = 1` is rejected by the
    /// `round_up = gt + eq*(far_sticky OR parity)` constraint — no quarter-ulp slack.
    #[test]
    fn fma_round_ties_to_even_is_pinned() {
        let mut found = false;
        for s in 0..20000u64 {
            let item = sample(s);
            let n = ref_noised(item.0, item.1, item.2);
            let ebit = (n.abs().to_bits() >> 23) & 0xFF;
            if n == 0.0 || !(1..=254).contains(&ebit) {
                continue;
            }
            let (program, rows) = trace_from(&[item]);
            if to_u64(rows[0][FMA_COL_MAP.eq]) == 1 && to_u64(rows[0][FMA_COL_MAP.round_up]) == 0 {
                let stark = Stk::new(program);
                assert!(!constraints_violated(&stark, &rows), "honest tie row passes");
                let mut forged = rows.clone();
                forged[0][FMA_COL_MAP.round_up] = F::ONE; // round up at an even tie
                assert!(constraints_violated(&stark, &forged), "ties-to-even: forced round-up rejected");
                found = true;
                break;
            }
        }
        assert!(found, "expected to find an exact-tie row in the sample space");
    }

    #[test]
    fn honest_lut_keys_are_in_domain() {
        let items: Vec<_> = (0..40).map(sample).collect();
        let (program, rows) = trace_from(&items);
        let live = program.live_rows();
        let polys = trace_rows_to_poly_values(rows);
        let lu = fma_lut_lookups::<F>();
        for (li, lookup) in lu.iter().enumerate() {
            for r in 0..live {
                match lookup.table {
                    LutTable::Range16 => {
                        // Filtered lookups (the [128,256) hi checks) are only meaningful where active.
                        let active = lookup.filter.eval_table(&polys, r, &[]).to_canonical_u64();
                        if active == 0 {
                            continue;
                        }
                        let k = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                        assert!(k < 1 << 16, "RANGE16 lookup {li} key {k} out of range at row {r}");
                    }
                    LutTable::Fp16Pow2 => {
                        let active = lookup.filter.eval_table(&polys, r, &[]).to_canonical_u64();
                        if active == 0 {
                            continue;
                        }
                        let k = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                        let v = lookup.values[0].eval_table(&polys, r, &[]).to_canonical_u64();
                        assert!(k <= 255, "FP16POW2 key {k} at row {r}");
                        assert_eq!(v, 1 << k.min(26), "FP16POW2 value {v} != 2^min({k},26) at row {r}");
                    }
                    LutTable::Width32 => {
                        let active = lookup.filter.eval_table(&polys, r, &[]).to_canonical_u64();
                        if active == 0 {
                            continue;
                        }
                        let k = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                        assert!((1..=32).contains(&k), "WIDTH32 key {k} at row {r}");
                        let tp = lookup.values[0].eval_table(&polys, r, &[]).to_canonical_u64();
                        let lp = lookup.values[1].eval_table(&polys, r, &[]).to_canonical_u64();
                        assert_eq!(tp, 1 << (k.saturating_sub(24)), "trunc at {r}");
                        assert_eq!(lp, 1 << (24u64.saturating_sub(k)), "lift at {r}");
                    }
                    LutTable::Fp16Decode => {}
                    other => panic!("unexpected table {other:?}"),
                }
            }
        }
    }

    #[test]
    fn degree_is_at_most_three() {
        let (program, _) = trace_from(&(0..8).map(sample).collect::<Vec<_>>());
        test_stark_low_degree::<F, Stk, D>(Stk::new(program)).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        let (program, _) = trace_from(&(0..8).map(sample).collect::<Vec<_>>());
        test_stark_circuit_constraints::<F, C, Stk, D>(Stk::new(program)).unwrap();
    }
}
