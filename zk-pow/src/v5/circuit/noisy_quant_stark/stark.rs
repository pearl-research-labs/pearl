//! Proves two of the three f32 roundings of the plaintext ground truth
//! [`crate::v5::api::quantization::noisy_quantize`]'s fused per-element noisy quantization. Per
//! element `(i, j)`, that kernel computes (with `af = bf16_to_f32(alpha)`, `bf =
//! bf16_to_f32(beta)`, `X = fp16_to_f32(raw)`, and the per-element noise `N = (E@F^T)_ij`):
//!
//! ```text
//! t      = RNE_f32(bf * N)                          // G1: pre-FMA f32 multiply
//! noised = af.mul_add(X, t)                         // G2: SINGLE-rounding f32 FMA — DEFERRED
//! out    = f32_to_fp16(clamp(noised, -MAX, MAX))    // G3: clamp + FP16 cast
//! ```
//!
//! # Scope (included vs deferred)
//!
//! * **G1 (included).** `t = RNE_f32(bf * N)`. `bf` is a normal BF16 (significand
//!   `M_beta in [128, 255]`); `N` is a normal f32 (significand `M_N in [2^23, 2^24)`) or exactly
//!   zero. The exact product significand `P = M_beta * M_N < 2^32` is rounded once to the 24-bit
//!   f32 significand `M_t` by the ties-to-even quarter-ulp *bracket*
//!   `(4*M_t - 2 + BOTTOM)*2^d1 <= 4*P <= (4*M_t + 2)*2^d1`, the shift `2^d1` an FP16POW2 value and
//!   `M_t in [2^23, 2^24)` pinning `d1 = bitlen(P) - 24` uniquely. A zero (or, out of envelope,
//!   subnormal) noise word gates the bracket off and forces `t = 0`.
//!
//! * **G3 (included).** `out = f32_to_fp16(noised)`. The clamp is subsumed: `f32_to_fp16` already
//!   saturates on overflow, so `f32_to_fp16(clamp(x, +/-MAX)) == f32_to_fp16(x)` for every finite
//!   `x` (verified exhaustively over a random f32 sweep). All four cast branches are proved —
//!   saturate (`E >= 143`), normal (`113 <= E <= 142`, shift 13), subnormal (`102 <= E <= 112`,
//!   shift `126 - E` via FP16POW2) and zero (`E <= 101`) — classified by three monotone flags with
//!   two-sided range slacks. The rounding uses the quarter-ulp bracket with the parity tie-break; it
//!   is **uniform** (no IS_BOTTOM, because the cast rounds within a fixed exponent), unlike G1's
//!   product round. The mantissa-overflow carry (`q = 2^11`) and the FP16-inf boundary (`e_adj > 15`)
//!   fold into the output encode.
//!
//! * **G2 (the single-rounding f32 FMA `noised = fma(af, X, t)`)** lives in the sibling AIR
//!   [`crate::v5::circuit::noisy_quant_fma_stark`] (a signed, arbitrarily-aligned add with
//!   cancellation and one RNE round, via A100-style windowed alignment). Its
//!   `ctl_fma_pairing_looking` exposes the same tuple this module's [`super::ctl::ctl_fma_hook_looking`]
//!   does, so the batch binds `t` (G1 here) and `noised` (G3 here) to G2's proven FMA. `noised` is
//!   therefore a proven value, not a free witness.
//!
//! # Soundness envelope (documented, not an unproved shortcut)
//!
//! `beta` normal positive; `N` normal f32 or exactly zero; `noised` a normal f32 (`E in [1, 254]`).
//! A subnormal-f32 *operand* (`|.| < 2^-126`) is out of envelope — the generator asserts against it
//! (it never arises in-scheme: alpha scales a row toward the FP16 ceiling) — but a tiny FP16 *output*
//! is fully handled (G3's subnormal/zero branches), so a tiny element does not crash the prover. G1's
//! product round carries the full ties-to-even machinery with the binade-bottom (`BOTTOM`) `+1`
//! correction; G3's cast is uniform with the parity tie-break. Both pin the rounded significand to
//! the **unique** RNE result with no residual grinding freedom.

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

use super::columns::{NUM_NOISY_QUANT_COLUMNS, NUM_NOISY_QUANT_PUBLIC_INPUTS, NoisyQuantColumnsView};
use crate::v5::api::dtype::f32_to_fp16;
use crate::v4::api::dtype::bf16_to_f32;
use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// Field inverse of a small integer, or zero when it is zero (the is-zero gadget's witness).
fn inv_f<F: RichField>(x: u64) -> F {
    if x == 0 { F::ZERO } else { F::from_canonical_u64(x).inverse() }
}

/// Inverse of the field difference `a - b` (zero when `a == b`); handles `a < b` without u64
/// underflow (the is-zero gadget's witness for a difference key).
fn inv_diff<F: RichField>(a: u64, b: u64) -> F {
    let d = F::from_canonical_u64(a) - F::from_canonical_u64(b);
    if d == F::ZERO { F::ZERO } else { d.inverse() }
}

/// Signed i128 to field (the lower bracket boundary is negative when `q` is small/zero).
fn fe_i128<F: RichField>(x: i128) -> F {
    if x >= 0 { F::from_canonical_u64(x as u64) } else { -F::from_canonical_u64((-x) as u64) }
}

/// `x >> shift` rounded to nearest, ties to even (the f32 significand rounding; mirrors the
/// `round_shift` of [`crate::v5::api::dtype`]).
fn round_shift(x: u64, shift: u32) -> u64 {
    if shift == 0 {
        return x;
    }
    let dropped = x & ((1u64 << shift) - 1);
    let kept = x >> shift;
    let half = 1u64 << (shift - 1);
    if dropped > half || (dropped == half && (kept & 1) == 1) { kept + 1 } else { kept }
}

fn bit_length(x: u64) -> u32 {
    64 - x.leading_zeros()
}

/// One decoded + proved element of the fused noisy quantization. The committed geometry is just the
/// element count; the identity is otherwise program-independent.
#[derive(Clone, Debug)]
pub struct NoisyQuantProgram {
    /// Number of live operand elements (`(h + w) * k` in the batch).
    pub num_elems: usize,
    /// Block size (elements per operand row); `OPERAND_ROW_INDEX = floor(element / k)`.
    pub k: usize,
    /// Trace height (a power of two `>= num_elems`).
    pub num_rows: usize,
    /// Number of A-side operand rows (`h` in the batch). Operand rows `0..num_a_rows` are the A
    /// operand (reused `mult_a` times each by the matmul), the rest are the B operand (reused
    /// `mult_b` times). Standalone programs use `num_elems/k` (all A) with unit multiplicity.
    pub num_a_rows: usize,
    /// Matmul reuse multiplicity of an A-side element (`w` in the batch): `OPERAND_MULT` on A rows.
    pub mult_a: usize,
    /// Matmul reuse multiplicity of a B-side element (`h` in the batch): `OPERAND_MULT` on B rows.
    pub mult_b: usize,
}

impl NoisyQuantProgram {
    /// Single-block standalone program (`k = num_elems`), padded to the next power of two. Unit
    /// reuse multiplicity (no matmul counterpart in the standalone tests).
    pub fn new(num_elems: usize) -> Self {
        Self::with_rows(num_elems, num_elems, num_elems.next_power_of_two().max(2), 1, 1, 1)
    }

    /// Multi-operand-row program: `num_elems` elements grouped into `k`-element blocks, at an
    /// explicit `num_rows` height. `num_a_rows` splits the operand rows into the A operand
    /// (`0..num_a_rows`, element reuse `mult_a`) and the B operand (reuse `mult_b`) — the
    /// operand-codes (6d) looked-side multiplicity.
    pub fn with_rows(num_elems: usize, k: usize, num_rows: usize, num_a_rows: usize, mult_a: usize, mult_b: usize) -> Self {
        assert!(num_elems >= 1, "at least one element");
        assert!(k >= 1 && num_elems % k == 0, "num_elems must be a whole number of k-element rows");
        assert!(num_rows.is_power_of_two() && num_rows >= num_elems, "height covers the live elements");
        assert!(num_a_rows <= num_elems / k, "A rows cannot exceed the operand-row count");
        Self { num_elems, k, num_rows, num_a_rows, mult_a, mult_b }
    }

    pub fn live_rows(&self) -> usize {
        self.num_elems
    }

    /// Trace height.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// `OPERAND_ROW_INDEX` for a trace row (`floor(element / k)`; the final live index on padding).
    fn operand_row_index(&self, i: usize) -> usize {
        if i >= self.num_elems { self.num_elems / self.k - 1 } else { i / self.k }
    }

    /// `ELEMENT_INDEX` for a trace row (the row's global element index `e`; the final live index on
    /// padding, filtered out of every CTL).
    fn element_index(&self, i: usize) -> usize {
        if i >= self.num_elems { self.num_elems - 1 } else { i }
    }

    /// `OPERAND_MULT` for a trace row: `mult_a` for an A-side element (operand row `< num_a_rows`),
    /// else `mult_b` — the matmul reuse multiplicity the operand-codes (6d) looked side carries.
    fn operand_mult(&self, i: usize) -> usize {
        if self.operand_row_index(i) < self.num_a_rows { self.mult_a } else { self.mult_b }
    }

    /// The class (a) ("known") columns `IS_PAD`, `OPERAND_ROW_INDEX`, `ELEMENT_INDEX` and
    /// `OPERAND_MULT`, pure functions of geometry.
    pub fn known_values<F: RichField>(&self) -> Vec<PolynomialValues<F>> {
        let num_rows = self.num_rows();
        let is_pad = (0..num_rows).map(|i| F::from_bool(i >= self.num_elems)).collect();
        let ori = (0..num_rows).map(|i| F::from_canonical_usize(self.operand_row_index(i))).collect();
        let eidx = (0..num_rows).map(|i| F::from_canonical_usize(self.element_index(i))).collect();
        let mult = (0..num_rows).map(|i| F::from_canonical_usize(self.operand_mult(i))).collect();
        vec![
            PolynomialValues::new(is_pad),
            PolynomialValues::new(ori),
            PolynomialValues::new(eidx),
            PolynomialValues::new(mult),
        ]
    }

    /// Generates the trace. Each of `alpha`/`raw`/`beta` (BF16/FP16 codes), `noise`/`noised`
    /// (f32 values) has one entry per live element. `out` (G3) and `t` (G1) are computed internally,
    /// bit-exact with the reference kernel. Panics on an out-of-envelope element.
    pub fn generate_trace<F: RichField>(
        &self,
        alpha: &[u16],
        raw: &[u16],
        beta: &[u16],
        noise: &[f32],
        noised: &[f32],
    ) -> Vec<[F; NUM_NOISY_QUANT_COLUMNS]> {
        let n = self.num_elems;
        assert!(alpha.len() == n && raw.len() == n && beta.len() == n && noise.len() == n && noised.len() == n);
        let num_rows = self.num_rows();
        let mut rows = Vec::with_capacity(num_rows);

        for e in 0..n {
            let mut v = NoisyQuantColumnsView::<F>::default();
            v.operand_row_index = F::from_canonical_usize(self.operand_row_index(e));
            v.element_index = F::from_canonical_usize(self.element_index(e));
            v.operand_mult = F::from_canonical_usize(self.operand_mult(e));
            v.alpha = F::from_canonical_u16(alpha[e]);
            v.raw = F::from_canonical_u16(raw[e]);

            // ---- G1: t = RNE_f32(bf * N). ----
            let beta_code = beta[e];
            let beta_exp = u64::from(beta_code >> 7);
            let beta_mant = u64::from(beta_code & 0x7F);
            assert!((1..=254).contains(&beta_exp), "envelope: beta must be a normal BF16");
            let m_beta = 128 + beta_mant;
            v.beta = F::from_canonical_u16(beta_code);
            v.beta_exp = F::from_canonical_u64(beta_exp);
            v.beta_mant = F::from_canonical_u64(beta_mant);

            let nbits = noise[e].to_bits();
            let noise_sign = u64::from(nbits >> 31);
            let noise_exp = u64::from((nbits >> 23) & 0xFF);
            let noise_mant = u64::from(nbits & 0x7F_FFFF);
            assert!(noise_exp != 255, "noise must be finite");
            let noise_is_zero = noise_exp == 0;
            if noise_is_zero {
                assert_eq!(noise_mant, 0, "envelope: a zero-exponent noise word must be exactly zero");
            }
            v.noise_lo = F::from_canonical_u64(u64::from(nbits & 0xFFFF));
            v.noise_hi = F::from_canonical_u64(u64::from(nbits >> 16));
            v.noise_sign = F::from_canonical_u64(noise_sign);
            v.noise_exp = F::from_canonical_u64(noise_exp);
            v.noise_mant = F::from_canonical_u64(noise_mant);
            v.noise_mant_lo = F::from_canonical_u64(noise_mant & 0xFFFF);
            v.noise_mant_hi = F::from_canonical_u64(noise_mant >> 16);
            v.noise_is_zero = F::from_bool(noise_is_zero);
            v.noise_exp_inv = inv_f::<F>(noise_exp);

            let mn = if noise_is_zero { 0 } else { (1u64 << 23) + noise_mant };
            let pm = m_beta * mn;
            v.mn = F::from_canonical_u64(mn);
            v.pm = F::from_canonical_u64(pm);
            v.pm_lo = F::from_canonical_u64(pm & 0xFFFF);
            v.pm_hi = F::from_canonical_u64(pm >> 16);

            if noise_is_zero {
                v.t_pow = F::ONE; // 2^0, an in-domain FP16POW2 key on the gated-off row.
            } else {
                // d1 = bitlen(P) - 24, with the mantissa-overflow renormalization folded in.
                let mut d1 = bit_length(pm) - 24;
                let mut m_t = round_shift(pm, d1);
                if m_t == (1u64 << 24) {
                    m_t = 1u64 << 23;
                    d1 += 1;
                }
                assert!((1 << 23..1 << 24).contains(&m_t), "M_t normalized");
                // Cross-check against the reference f32 multiply.
                let t_f32 = bf16_to_f32(beta_code) * noise[e];
                let tbits = t_f32.to_bits();
                debug_assert_eq!((1u64 << 23) + u64::from(tbits & 0x7F_FFFF), m_t, "M_t vs f32 mul");
                let t_exp = u64::from((tbits >> 23) & 0xFF);
                debug_assert_eq!(t_exp, beta_exp + noise_exp + u64::from(d1) - 134, "t exp relation");

                let pow = 1u64 << d1;
                let bottom = u64::from(m_t == (1 << 23));
                let parity = m_t & 1;
                // The parity split rides the low limb (M_t = t_mant_lo + 2^16*hi; 2^16*hi is even, so
                // M_t and t_mant_lo share parity) so T_HALF stays a single RANGE16 limb (< 2^15).
                let half = ((m_t & 0xFFFF) - parity) / 2;
                let blo = (4 * m_t - 2 + bottom) * pow;
                let bhi = (4 * m_t + 2) * pow;
                let sl = 4 * pm - blo - parity;
                let su = bhi - 4 * pm - parity;
                v.t_shift = F::from_canonical_u64(u64::from(d1));
                v.t_pow = F::from_canonical_u64(pow);
                v.t_mant = F::from_canonical_u64(m_t);
                v.t_mant_lo = F::from_canonical_u64(m_t & 0xFFFF);
                v.t_mant_hi = F::from_canonical_u64(m_t >> 16);
                v.t_bottom = F::from_canonical_u64(bottom);
                v.t_bottom_inv = inv_f::<F>(m_t - (1 << 23));
                v.t_parity = F::from_canonical_u64(parity);
                v.t_half = F::from_canonical_u64(half);
                v.t_blo = F::from_canonical_u64(blo);
                v.t_bhi = F::from_canonical_u64(bhi);
                v.t_sl_mid = F::from_canonical_u64((sl >> 16) & 0xFFFF);
                v.t_sl_hi = F::from_canonical_u64(sl >> 32);
                v.t_su_mid = F::from_canonical_u64((su >> 16) & 0xFFFF);
                v.t_su_hi = F::from_canonical_u64(su >> 32);
                v.t_exp = F::from_canonical_u64(t_exp);
            }
            v.nz_live = F::from_bool(!noise_is_zero);

            // ---- G3: out = f32_to_fp16(noised) (normal / subnormal / zero / saturating). ----
            let xbits = noised[e].to_bits();
            let noised_sign = u64::from(xbits >> 31);
            let noised_exp = u64::from((xbits >> 23) & 0xFF);
            let noised_mant = u64::from(xbits & 0x7F_FFFF);
            assert!((1..=254).contains(&noised_exp), "noised must be a normal f32 (exp {noised_exp})");
            v.noised_lo = F::from_canonical_u64(u64::from(xbits & 0xFFFF));
            v.noised_hi = F::from_canonical_u64(u64::from(xbits >> 16));
            v.noised_sign = F::from_canonical_u64(noised_sign);
            v.noised_exp = F::from_canonical_u64(noised_exp);
            v.noised_mant = F::from_canonical_u64(noised_mant);
            v.noised_mant_lo = F::from_canonical_u64(noised_mant & 0xFFFF);
            v.noised_mant_hi = F::from_canonical_u64(noised_mant >> 16);

            // Branch classification.
            let f_sat = noised_exp >= 143;
            let ge113 = noised_exp >= 113;
            let ge102 = noised_exp >= 102;
            let f_norm = ge113 && !f_sat;
            let f_sub = ge102 && !ge113;
            let f_zero = !ge102;
            v.f_sat = F::from_bool(f_sat);
            v.sat_slack = F::from_canonical_u64(if f_sat { noised_exp - 143 } else { 142 - noised_exp });
            v.ge113 = F::from_bool(ge113);
            v.ge113_slack = F::from_canonical_u64(if ge113 { noised_exp - 113 } else { 112 - noised_exp });
            v.ge102 = F::from_bool(ge102);
            v.ge102_slack = F::from_canonical_u64(if ge102 { noised_exp - 102 } else { 101 - noised_exp });
            v.f_norm = F::from_bool(f_norm);
            v.f_sub = F::from_bool(f_sub);
            v.f_zero = F::from_bool(f_zero);

            let full = (1u64 << 23) + noised_mant;
            let round_active = f_norm || f_sub;
            let cast_shift: u64 = if f_norm { 13 } else if f_sub { 126 - noised_exp } else { 0 };
            let cast_pow: u64 = 1 << cast_shift;
            let q = if round_active { round_shift(full, cast_shift as u32) } else { 0 };
            let q_bottom = u64::from(q == (1 << 10));
            let parity = q & 1;
            v.cast_shift = F::from_canonical_u64(cast_shift);
            v.cast_pow = F::from_canonical_u64(cast_pow);
            v.q = F::from_canonical_u64(q);
            v.q_lo_slack = F::from_canonical_u64(if f_norm { q - (1 << 10) } else { 0 });
            v.q_bottom = F::from_canonical_u64(q_bottom);
            v.q_bottom_inv = inv_diff::<F>(q, 1 << 10);
            v.q_parity = F::from_canonical_u64(parity);
            v.q_half = F::from_canonical_u64((q - parity) / 2);
            if round_active {
                // The cast rounds within a FIXED exponent (uniform grid) — this is `round_shift`,
                // so there is NO binade-bottom asymmetry (unlike G1's product round). The bracket is
                // the plain quarter-ulp bracket (4q-2 .. 4q+2) with the parity tie-break.
                let blo_i: i128 = (4 * q as i128 - 2) * cast_pow as i128;
                let bhi: u64 = (4 * q + 2) * cast_pow;
                let sl: i128 = 4 * full as i128 - blo_i - parity as i128;
                let su: i128 = bhi as i128 - 4 * full as i128 - parity as i128;
                debug_assert!(sl >= 0 && su >= 0 && (sl >> 16) < (1 << 16) && (su >> 16) < (1 << 16));
                v.cast_blo = fe_i128::<F>(blo_i);
                v.cast_bhi = F::from_canonical_u64(bhi);
                v.cast_sl_hi = F::from_canonical_u64((sl >> 16) as u64);
                v.cast_su_hi = F::from_canonical_u64((su >> 16) as u64);
            }
            let carry = u64::from(q == (1 << 11));
            v.carry = F::from_canonical_u64(carry);
            v.carry_inv = inv_diff::<F>(1 << 11, q);
            let e142 = u64::from(noised_exp == 142);
            v.e142 = F::from_canonical_u64(e142);
            v.e142_inv = inv_diff::<F>(142, noised_exp);
            let cce = carry * e142;
            v.cce = F::from_canonical_u64(cce);
            let is_sat_eff = u64::from(f_sat) + u64::from(f_norm) * cce;
            v.is_sat_eff = F::from_canonical_u64(is_sat_eff);
            let fnns = u64::from(f_norm) * (1 - cce);
            v.fnns = F::from_canonical_u64(fnns);
            let mf = if f_norm { (1 - carry) * (q - (1 << 10)) } else { 0 };
            v.mf = F::from_canonical_u64(mf);

            let out = f32_to_fp16(noised[e]).expect("finite noised casts to FP16");
            v.out = F::from_canonical_u16(out);
            // Cross-check the encode the constraints rebuild (normal_mag only meaningful when f_norm).
            let normal_mag = if f_norm { (noised_exp - 112 + carry) * (1 << 10) + mf } else { 0 };
            let expect = (noised_sign << 15) + is_sat_eff * 0x7BFF + fnns * normal_mag + u64::from(f_sub) * q;
            debug_assert_eq!(u64::from(out), expect, "G3 encode mismatch (exp {noised_exp})");

            rows.push(v.into());
        }

        for i in n..num_rows {
            rows.push(
                NoisyQuantColumnsView::<F> {
                    is_pad: F::ONE,
                    operand_row_index: F::from_canonical_usize(self.operand_row_index(i)),
                    element_index: F::from_canonical_usize(self.element_index(i)),
                    operand_mult: F::from_canonical_usize(self.operand_mult(i)),
                    ..Default::default()
                }
                .into(),
            );
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

/// `mant = 2*half + parity` with `parity` boolean (`half` is RANGE16'd in `ctl.rs`).
fn parity_split<V, S, E>(eval: &mut E, mant: V, half: V, parity: V, gate: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    eval.constraint_bool(parity);
    let two = eval.u64(2);
    let two_half = eval.mul(two, half);
    let recomposed = eval.add(two_half, parity);
    let d = eval.sub(mant, recomposed);
    let c = eval.mul(gate, d);
    eval.constraint(c);
}

/// Evaluates every arithmetic constraint (degree <= 3). All range facts (RANGE16 limbs/slacks, the
/// FP16POW2 shift) live in [`super::ctl`].
pub(crate) fn eval_noisy_quant_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_NOISY_QUANT_COLUMNS, NUM_NOISY_QUANT_PUBLIC_INPUTS>,
    eval: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_NOISY_QUANT_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &NoisyQuantColumnsView<V> = lv.borrow();

    let one = eval.u64(1);
    let live = eval.sub(one, lv.is_pad);
    eval.constraint_bool(lv.is_pad);

    let c128 = eval.u64(128);
    let c4 = eval.u64(4);
    let two = eval.u64(2);

    // ================= GROUP G1 — t = RNE_f32(bf * N). =================
    // Decode beta: code = beta_exp*128 + beta_mant (beta is a normal positive BF16).
    let beta_rec = eval.mad(lv.beta_exp, c128, lv.beta_mant);
    let d = eval.sub(lv.beta, beta_rec);
    let c = eval.mul(live, d);
    eval.constraint(c);

    // Decode N: bits = sign*2^31 + exp*2^23 + mant; mant = lo + 2^16*hi; bits = lo16 + 2^16*hi16.
    let c2_16 = eval.u64(1 << 16);
    let c2_23 = eval.u64(1 << 23);
    let c2_31 = eval.u64(1 << 31);
    let noise_mant_rec = eval.mad(lv.noise_mant_hi, c2_16, lv.noise_mant_lo);
    let d = eval.sub(lv.noise_mant, noise_mant_rec);
    let c = eval.mul(live, d);
    eval.constraint(c);
    let nbits_fields = {
        let s = eval.mul(lv.noise_sign, c2_31);
        let e = eval.mul(lv.noise_exp, c2_23);
        let se = eval.add(s, e);
        eval.add(se, lv.noise_mant)
    };
    let nbits_limbs = eval.mad(lv.noise_hi, c2_16, lv.noise_lo);
    let d = eval.sub(nbits_fields, nbits_limbs);
    let c = eval.mul(live, d);
    eval.constraint(c);
    eval.constraint_bool(lv.noise_sign);

    // noise_is_zero = [noise_exp == 0].
    is_zero_flag(eval, lv.noise_exp, lv.noise_exp_inv, lv.noise_is_zero, live);

    // mn = (1 - noise_is_zero)*2^23 + noise_mant.
    let nz = eval.sub(one, lv.noise_is_zero);
    let hi = eval.mul(nz, c2_23);
    let mn_expect = eval.add(hi, lv.noise_mant);
    let d = eval.sub(lv.mn, mn_expect);
    let c = eval.mul(live, d);
    eval.constraint(c);

    // P = M_beta * M_N = (128 + beta_mant) * mn; P = pm_lo + 2^16*pm_hi.
    let m_beta = eval.add(c128, lv.beta_mant);
    let pm_expect = eval.mul(m_beta, lv.mn);
    let d = eval.sub(lv.pm, pm_expect);
    let c = eval.mul(live, d);
    eval.constraint(c);
    let pm_rec = eval.mad(lv.pm_hi, c2_16, lv.pm_lo);
    let d = eval.sub(lv.pm, pm_rec);
    let c = eval.mul(live, d);
    eval.constraint(c);

    // t fields (reconstruct M_t from limbs; M_t_hi carries the [128,256) pin from ctl.rs).
    let t_mant_rec = eval.mad(lv.t_mant_hi, c2_16, lv.t_mant_lo);
    let d = eval.sub(lv.t_mant, t_mant_rec);
    let c = eval.mul(live, d);
    eval.constraint(c);

    // Nonzero-noise rows re-prove the full bracket; zero-noise rows force t = 0. NZ_LIVE is a
    // committed boolean = live*(1 - noise_is_zero), used as a degree-1 gate so the degree-2 boundary
    // constraints stay degree <= 3; it is 0 on padding rows (live = 0 there).
    eval.constraint_bool(lv.nz_live);
    let nz_live_expect = eval.mul(live, nz);
    let d = eval.sub(lv.nz_live, nz_live_expect);
    eval.constraint(d);
    let live_nz = lv.nz_live;
    // Bracket boundaries blo = (4*M_t - 2 + BOTTOM)*2^d1, bhi = (4*M_t + 2)*2^d1.
    let four_mt = eval.mul(c4, lv.t_mant);
    let lo_coef = {
        let m2 = eval.sub(four_mt, two);
        eval.add(m2, lv.t_bottom)
    };
    let hi_coef = eval.add(four_mt, two);
    let blo_expect = eval.mul(lo_coef, lv.t_pow);
    let d = eval.sub(lv.t_blo, blo_expect);
    let c = eval.mul(live_nz, d);
    eval.constraint(c);
    let bhi_expect = eval.mul(hi_coef, lv.t_pow);
    let d = eval.sub(lv.t_bhi, bhi_expect);
    let c = eval.mul(live_nz, d);
    eval.constraint(c);
    // Parity split on the low limb (shares M_t's parity; keeps T_HALF a single RANGE16 limb).
    parity_split(eval, lv.t_mant_lo, lv.t_half, lv.t_parity, live_nz);
    // t_bottom = [M_t - 2^23 == 0].
    let mt_rel = eval.sub(lv.t_mant, c2_23);
    is_zero_flag(eval, mt_rel, lv.t_bottom_inv, lv.t_bottom, live_nz);
    // t_exp = beta_exp + noise_exp + t_shift - 134 (nonzero rows).
    let c134 = eval.u64(134);
    let s = eval.add(lv.beta_exp, lv.noise_exp);
    let s = eval.add(s, lv.t_shift);
    let texp_expect = eval.sub(s, c134);
    let d = eval.sub(lv.t_exp, texp_expect);
    let c = eval.mul(live_nz, d);
    eval.constraint(c);
    // Zero-noise rows: t_mant = t_exp = t_shift = 0, t_pow = 1.
    let live_z = eval.mul(live, lv.noise_is_zero);
    let c = eval.mul(live_z, lv.t_mant);
    eval.constraint(c);
    let c = eval.mul(live_z, lv.t_exp);
    eval.constraint(c);
    let c = eval.mul(live_z, lv.t_shift);
    eval.constraint(c);
    let pow_m1 = eval.sub(lv.t_pow, one);
    let c = eval.mul(live_z, pow_m1);
    eval.constraint(c);

    // ================= GROUP G3 — out = f32_to_fp16(noised) (all four branches). =================
    eval.constraint_bool(lv.noised_sign);
    let noised_mant_rec = eval.mad(lv.noised_mant_hi, c2_16, lv.noised_mant_lo);
    let d = eval.sub(lv.noised_mant, noised_mant_rec);
    gated_g3(eval, live, d);
    let xbits_fields = {
        let s = eval.mul(lv.noised_sign, c2_31);
        let e = eval.mul(lv.noised_exp, c2_23);
        let se = eval.add(s, e);
        eval.add(se, lv.noised_mant)
    };
    let xbits_limbs = eval.mad(lv.noised_hi, c2_16, lv.noised_lo);
    let d = eval.sub(xbits_fields, xbits_limbs);
    gated_g3(eval, live, d);

    let c142 = eval.u64(142);
    let c142_m_e = eval.sub(c142, lv.noised_exp);
    // Monotone branch flags via two-sided slacks: flag*(E - lo) + (1-flag)*(hi - E) >= 0.
    eval.constraint_bool(lv.f_sat);
    eval.constraint_bool(lv.ge113);
    eval.constraint_bool(lv.ge102);
    for (flag, lo, hi, slack) in [
        (lv.f_sat, 143u64, 142u64, lv.sat_slack),
        (lv.ge113, 113, 112, lv.ge113_slack),
        (lv.ge102, 102, 101, lv.ge102_slack),
    ] {
        let clo = eval.u64(lo);
        let chi = eval.u64(hi);
        let e_m_lo = eval.sub(lv.noised_exp, clo);
        let hi_m_e = eval.sub(chi, lv.noised_exp);
        let a = eval.mul(flag, e_m_lo);
        let not_f = eval.sub(one, flag);
        let b = eval.mul(not_f, hi_m_e);
        let expect = eval.add(a, b);
        let d = eval.sub(slack, expect);
        gated_g3(eval, live, d);
    }
    // f_norm = ge113*(1 - f_sat); f_sub = ge102*(1 - ge113); f_zero = 1 - ge102.
    eval.constraint_bool(lv.f_norm);
    eval.constraint_bool(lv.f_sub);
    eval.constraint_bool(lv.f_zero);
    let not_sat = eval.sub(one, lv.f_sat);
    let fn_expect = eval.mul(lv.ge113, not_sat);
    let d = eval.sub(lv.f_norm, fn_expect);
    gated_g3(eval, live, d);
    let not_113 = eval.sub(one, lv.ge113);
    let fs_expect = eval.mul(lv.ge102, not_113);
    let d = eval.sub(lv.f_sub, fs_expect);
    gated_g3(eval, live, d);
    let fz_expect = eval.sub(one, lv.ge102);
    let d = eval.sub(lv.f_zero, fz_expect);
    gated_g3(eval, live, d);
    let round_active = eval.add(lv.f_norm, lv.f_sub);

    // cast_shift = 13*f_norm + (126 - E)*f_sub; cast_pow = 2^cast_shift (FP16POW2, ctl).
    let c13 = eval.u64(13);
    let c126 = eval.u64(126);
    let shift_norm = eval.mul(c13, lv.f_norm);
    let c126_m_e = eval.sub(c126, lv.noised_exp);
    let shift_sub = eval.mul(c126_m_e, lv.f_sub);
    let shift_expect = eval.add(shift_norm, shift_sub);
    let d = eval.sub(lv.cast_shift, shift_expect);
    gated_g3(eval, live, d);

    // Cast bracket (rounding branches): cast_blo = (4q - 2 + bottom)*cast_pow, cast_bhi = (4q+2)*pow.
    let c2_10 = eval.u64(1 << 10);
    let four_q = eval.mul(c4, lv.q);
    // Uniform quarter-ulp bracket (no IS_BOTTOM: the cast rounds within a fixed exponent).
    let lo_coef = eval.sub(four_q, two);
    let hi_coef = eval.add(four_q, two);
    let blo_expect = eval.mul(lo_coef, lv.cast_pow);
    let d = eval.sub(lv.cast_blo, blo_expect);
    let c = eval.mul(round_active, d); // degree 3, 0 off the rounding branches
    eval.constraint(c);
    let bhi_expect = eval.mul(hi_coef, lv.cast_pow);
    let d = eval.sub(lv.cast_bhi, bhi_expect);
    let c = eval.mul(round_active, d);
    eval.constraint(c);
    // Off the rounding branches: cast_blo = cast_bhi = q = 0 (so the ctl slacks vanish).
    let not_round = eval.sub(one, round_active);
    let c = eval.mul(not_round, lv.q);
    gated_g3(eval, live, c);
    let c = eval.mul(not_round, lv.cast_blo);
    gated_g3(eval, live, c);
    let c = eval.mul(not_round, lv.cast_bhi);
    gated_g3(eval, live, c);
    parity_split(eval, lv.q, lv.q_half, lv.q_parity, live);
    // q_lo_slack = q - 2^10 on the normal branch (pins q >= 2^10 there).
    let q_rel = eval.sub(lv.q, c2_10);
    let d = eval.sub(lv.q_lo_slack, q_rel);
    let c = eval.mul(lv.f_norm, d);
    gated_g3(eval, live, c);
    // q_bottom = [q - 2^10 == 0]; carry = [2^11 - q == 0]; e142 = [142 - E == 0].
    is_zero_flag(eval, q_rel, lv.q_bottom_inv, lv.q_bottom, live);
    let c2_11 = eval.u64(1 << 11);
    let q_hi_rel = eval.sub(c2_11, lv.q);
    is_zero_flag(eval, q_hi_rel, lv.carry_inv, lv.carry, live);
    is_zero_flag(eval, c142_m_e, lv.e142_inv, lv.e142, live);
    // cce = carry*e142; is_sat_eff = f_sat + f_norm*cce; fnns = f_norm*(1 - cce).
    eval.constraint_bool(lv.cce);
    eval.constraint_bool(lv.is_sat_eff);
    eval.constraint_bool(lv.fnns);
    let cce_expect = eval.mul(lv.carry, lv.e142);
    let d = eval.sub(lv.cce, cce_expect);
    gated_g3(eval, live, d);
    let fncce = eval.mul(lv.f_norm, lv.cce);
    let ise_expect = eval.add(lv.f_sat, fncce);
    let d = eval.sub(lv.is_sat_eff, ise_expect);
    gated_g3(eval, live, d);
    let fnns_expect = eval.sub(lv.f_norm, fncce);
    let d = eval.sub(lv.fnns, fnns_expect);
    gated_g3(eval, live, d);
    // MF = (1 - carry)*(q - 2^10) on the normal branch (degree-budget helper).
    let not_carry = eval.sub(one, lv.carry);
    let mf_expect = eval.mul(not_carry, q_rel);
    let d = eval.sub(lv.mf, mf_expect);
    let c = eval.mul(lv.f_norm, d);
    eval.constraint(c);

    // Output encode:
    //   out = sign*2^15 + IS_SAT_EFF*0x7BFF + FNNS*((E-112+carry)*2^10 + MF) + F_SUB*q.
    let c2_15 = eval.u64(1 << 15);
    let sign_hi = eval.mul(lv.noised_sign, c2_15);
    let c7bff = eval.u64(0x7BFF);
    let c112 = eval.u64(112);
    let expfield = {
        let em = eval.sub(lv.noised_exp, c112);
        eval.add(em, lv.carry)
    };
    let exp_term = eval.mul(expfield, c2_10);
    let normal_mag = eval.add(exp_term, lv.mf);
    let sat_term = eval.mul(lv.is_sat_eff, c7bff);
    let norm_term = eval.mul(lv.fnns, normal_mag);
    let sub_term = eval.mul(lv.f_sub, lv.q);
    let out_expect = eval.add(sign_hi, sat_term);
    let out_expect = eval.add(out_expect, norm_term);
    let out_expect = eval.add(out_expect, sub_term);
    let d = eval.sub(lv.out, out_expect);
    gated_g3(eval, live, d);
}

/// Emit `live * expr = 0` (degree = 1 + deg(expr)); use only when `expr` is degree <= 2.
fn gated_g3<V, S, E>(eval: &mut E, live: V, expr: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let c = eval.mul(live, expr);
    eval.constraint(c);
}

/// FP16 NoisyQuantStark. A CTL party (`requires_ctls`): the FP16POW2 shift and the RANGE16 range
/// facts are served by the committed LUTs, and the operand / noise / FMA / output hooks are
/// CTL-bound in later stages, so the batch driver is the only supported proving path.
#[derive(Clone, Debug)]
pub struct NoisyQuantStark<F: RichField + Extendable<D>, const D: usize> {
    pub program: NoisyQuantProgram,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> NoisyQuantStark<F, D> {
    pub fn new(program: NoisyQuantProgram) -> Self {
        Self { program, _phantom: PhantomData }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for NoisyQuantStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_NOISY_QUANT_COLUMNS, NUM_NOISY_QUANT_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_NOISY_QUANT_COLUMNS, NUM_NOISY_QUANT_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_noisy_quant_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_noisy_quant_constraints(vars, &mut evaluator);
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

    use super::super::columns::NOISY_QUANT_COL_MAP;
    use super::super::ctl::noisy_quant_lut_lookups;
    use super::*;
    use crate::v5::api::dtype::fp16_to_f32;
    use crate::v5::api::quantization::{MAX_FP16, noisy_quantize, row_norms};
    use crate::v4::api::dtype::f32_to_bf16;
    use crate::v4::circuit::luts::LutTable;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type Stk = NoisyQuantStark<F, D>;

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    fn constraints_violated(stark: &Stk, rows: &[[F; NUM_NOISY_QUANT_COLUMNS]]) -> bool {
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

    /// A deterministic in-envelope sample: a normal `noised` in the FP16-normal range, a normal
    /// `beta`, and a normal `noise` word. Returns `(alpha, raw, beta, noise, noised)`.
    fn sample(seed: u64) -> (u16, u16, u16, f32, f32) {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s
        };
        let raw = ((next() >> 40) as u16) & 0x3FF | 0x3C00; // a normal FP16 near 1..2
        let alpha = f32_to_bf16(1.0 + (next() % 7) as f32 * 0.5).unwrap();
        let beta = f32_to_bf16(0.25 + (next() % 5) as f32 * 0.1).unwrap();
        // Always a nonzero normal noise word (keeps the sample-based tests on the G1 bracket path).
        let noise = (((next() % 1000) + 1) as f32) * if next() & 1 == 0 { 0.5 } else { -0.5 };
        // A normal mid-range noised value (FP16-normal: |x| in [2^-6, 2^10] here).
        let mag = 2f32.powi(((next() % 16) as i32) - 6) * (1.0 + (next() % 1000) as f32 / 1000.0);
        let sign = if next() & 1 == 0 { 1.0 } else { -1.0 };
        let noised = sign * mag;
        (alpha, raw, beta, noise, noised)
    }

    fn trace_from(
        items: &[(u16, u16, u16, f32, f32)],
    ) -> (NoisyQuantProgram, Vec<[F; NUM_NOISY_QUANT_COLUMNS]>) {
        let n = items.len();
        let program = NoisyQuantProgram::new(n);
        let alpha: Vec<u16> = items.iter().map(|x| x.0).collect();
        let raw: Vec<u16> = items.iter().map(|x| x.1).collect();
        let beta: Vec<u16> = items.iter().map(|x| x.2).collect();
        let noise: Vec<f32> = items.iter().map(|x| x.3).collect();
        let noised: Vec<f32> = items.iter().map(|x| x.4).collect();
        let rows = program.generate_trace::<F>(&alpha, &raw, &beta, &noise, &noised);
        (program, rows)
    }

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        let items: Vec<_> = (0..40).map(sample).collect();
        let (program, rows) = trace_from(&items);
        assert_eq!(rows.len(), 64);
        let known = program.known_values::<F>();
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(known[0].values[r], row[NOISY_QUANT_COL_MAP.is_pad], "is_pad row {r}");
        }
        assert!(!constraints_violated(&Stk::new(program), &rows), "honest trace violated a constraint");
    }

    /// Several hand-picked `noised` values exercise the G3 branches: a power-of-two rounding, a
    /// ties-to-even case, a clamp/saturation case, and ordinary normal casts — each compared against
    /// `f32_to_fp16(clamp(.))` (the plaintext reference, whose clamp `f32_to_fp16` subsumes).
    #[test]
    fn g3_cast_is_bit_exact_incl_clamp_and_power_of_two() {
        let noised_vals: Vec<f32> = vec![
            1.0,                    // power of two -> q at binade bottom (0x3C00)
            2.0,
            -4.0,
            1024.0,
            65504.0,                // exactly MAX
            70000.0,                // clamp/saturate -> 0x7BFF
            -80000.0,               // clamp/saturate -> 0xFBFF
            12345.0,
            f32::from_bits(0x477FE001), // just above a tie near the top normal binade
            0.013_f32,
            333.0,
        ];
        let items: Vec<_> = noised_vals
            .iter()
            .map(|&x| (f32_to_bf16(1.0).unwrap(), 0x3C00u16, f32_to_bf16(0.5).unwrap(), 2.0f32, x))
            .collect();
        let (_program, rows) = trace_from(&items);
        for (e, &x) in noised_vals.iter().enumerate() {
            let got = to_u64(rows[e][NOISY_QUANT_COL_MAP.out]) as u16;
            let want = f32_to_fp16(x.clamp(-MAX_FP16, MAX_FP16)).unwrap();
            assert_eq!(got, want, "G3 cast mismatch for noised={x}");
        }
        // Spot-check the documented codes.
        assert_eq!(to_u64(rows[0][NOISY_QUANT_COL_MAP.out]) as u16, 0x3C00); // 1.0
        assert_eq!(to_u64(rows[5][NOISY_QUANT_COL_MAP.out]) as u16, 0x7BFF); // saturate +
        assert_eq!(to_u64(rows[6][NOISY_QUANT_COL_MAP.out]) as u16, 0xFBFF); // saturate -
    }

    /// Bit-exact against the real `noisy_quantize`, with `noised` DERIVED BY THE AIR (not supplied):
    /// G1 (this module) proves `t`, the G2 sibling AIR proves `noised = fma(af, X, t)`, and G3 (this
    /// module) casts that AIR-derived `noised`. The end-to-end FP16 codes must equal the kernel's.
    /// The row deliberately contains a tiny element (subnormal FP16 output) and a large one
    /// (near-saturation), exercising every G3 branch.
    #[test]
    fn g1_and_g3_bit_exact_vs_noisy_quantize() {
        use super::super::super::noisy_quant_fma_stark::{FMA_COL_MAP, FmaProgram};
        const R: usize = 32;
        let k = 48usize;
        let rows: Vec<u16> = (0..k)
            .map(|j| {
                let v = if j == 0 {
                    7.5 // near-saturation after scaling
                } else if j == 1 {
                    2f32.powi(-9) // tiny -> subnormal/zero FP16 output after scaling
                } else {
                    1.0 + (j % 5) as f32 * 0.25
                };
                f32_to_fp16(v).unwrap()
            })
            .collect();
        let e: Vec<u16> = (0..R).map(|t| f32_to_fp16(((t % 5) as f32 - 2.0) * 8.0).unwrap()).collect();
        let f: Vec<u16> = (0..k * R).map(|t| f32_to_fp16(((t % 7) as f32 - 3.0) * 8.0).unwrap()).collect();
        let norms = [row_norms(&rows).unwrap()];
        let built = noisy_quantize(&rows, &e, &f, &norms, R).unwrap();
        let noise = crate::v5::api::accumulate::a100_matmul(&e, &f, None, 1, k, R);
        let bf = crate::v4::api::dtype::bf16_to_f32(built.beta[0]);

        // Build t from G1 (= bf*noise), then DERIVE noised via the G2 AIR.
        let t: Vec<f32> = (0..k).map(|j| bf * noise[j]).collect();
        let t_sign: Vec<u64> = t.iter().map(|&x| u64::from(x.to_bits() >> 31)).collect();
        let t_mant: Vec<u64> =
            t.iter().map(|&x| if (x.to_bits() >> 23) & 0xFF == 0 { 0 } else { (1u64 << 23) + u64::from(x.to_bits() & 0x7F_FFFF) }).collect();
        let t_exp: Vec<u64> = t.iter().map(|&x| u64::from((x.to_bits() >> 23) & 0xFF)).collect();
        let fma_prog = FmaProgram::new(k);
        let alpha_vec = vec![built.alpha[0]; k];
        let fma_trace = fma_prog.generate_trace::<F>(&alpha_vec, &rows, &t_sign, &t_mant, &t_exp);
        let noised: Vec<f32> = (0..k)
            .map(|j| {
                let lo = to_u64(fma_trace[j][FMA_COL_MAP.noised_lo]);
                let hi = to_u64(fma_trace[j][FMA_COL_MAP.noised_hi]);
                f32::from_bits(((hi << 16) | lo) as u32)
            })
            .collect();

        // Feed the AIR-derived noised to this module (G1 proves t; G3 casts noised).
        let items: Vec<_> = (0..k).map(|j| (built.alpha[0], rows[j], built.beta[0], noise[j], noised[j])).collect();
        let (_program, trace) = trace_from(&items);
        for j in 0..k {
            // noised is DERIVED by the G2 AIR; confirm it reproduces the reference FMA exactly.
            let ref_noised = crate::v4::api::dtype::bf16_to_f32(built.alpha[0]).mul_add(fp16_to_f32(rows[j]), t[j]);
            assert_eq!(noised[j].to_bits(), ref_noised.to_bits(), "G2 noised mismatch elem {j}");
            let got = to_u64(trace[j][NOISY_QUANT_COL_MAP.out]) as u16;
            assert_eq!(got, built.noised_part[j], "end-to-end FP16 code mismatch at element {j}");
        }
        assert!(built.noised_part.iter().any(|&c| c & 0x7FFF >= 0x7000), "expected a near-ceiling element");
    }

    /// G3 subnormal + zero + smallest-normal boundary casts, bit-exact vs `f32_to_fp16(clamp(.))`.
    #[test]
    fn g3_subnormal_and_zero_outputs() {
        let vals: Vec<f32> = vec![
            2f32.powi(-15),               // subnormal FP16 (0x0200)
            2f32.powi(-20),               // deeper subnormal
            2f32.powi(-24),               // smallest subnormal (0x0001)
            2f32.powi(-25),               // ties to even -> 0
            2f32.powi(-26),               // underflow -> 0
            -2f32.powi(-16),              // signed subnormal
            2f32.powi(-14),               // smallest NORMAL (0x0400) via the subnormal round boundary
            2f32.powi(-14) - 2f32.powi(-25), // just below smallest normal
            f32::from_bits((102u32 << 23) | 0x400000), // E=102 subnormal with a nonzero rounding
        ];
        let items: Vec<_> = vals
            .iter()
            .map(|&x| (f32_to_bf16(1.0).unwrap(), 0x3C00u16, f32_to_bf16(0.5).unwrap(), 2.0f32, x))
            .collect();
        let (program, rows) = trace_from(&items);
        for (i, &x) in vals.iter().enumerate() {
            let got = to_u64(rows[i][NOISY_QUANT_COL_MAP.out]) as u16;
            assert_eq!(got, f32_to_fp16(x.clamp(-MAX_FP16, MAX_FP16)).unwrap(), "subnormal cast mismatch for {x}");
        }
        assert!(!constraints_violated(&Stk::new(program), &rows), "subnormal/zero trace must satisfy constraints");
    }

    #[test]
    fn zero_noise_gives_zero_t() {
        let items: Vec<_> = vec![(
            f32_to_bf16(1.0).unwrap(),
            0x3C00u16,
            f32_to_bf16(0.5).unwrap(),
            0.0f32, // zero noise
            3.5f32,
        )];
        let (program, rows) = trace_from(&items);
        assert_eq!(to_u64(rows[0][NOISY_QUANT_COL_MAP.t_mant]), 0);
        assert_eq!(to_u64(rows[0][NOISY_QUANT_COL_MAP.noise_is_zero]), 1);
        assert_eq!(to_u64(rows[0][NOISY_QUANT_COL_MAP.t_pow]), 1);
        assert!(!constraints_violated(&Stk::new(program), &rows));
    }

    #[test]
    fn tampered_traces_fail() {
        let items: Vec<_> = (0..16).map(sample).collect();
        let (program, rows) = trace_from(&items);
        let stark = Stk::new(program);
        assert!(!constraints_violated(&stark, &rows), "baseline honest trace must pass");
        let cases = [
            ("beta", NOISY_QUANT_COL_MAP.beta),
            ("noise_sign", NOISY_QUANT_COL_MAP.noise_sign),
            ("mn", NOISY_QUANT_COL_MAP.mn),
            ("pm", NOISY_QUANT_COL_MAP.pm),
            ("t_mant", NOISY_QUANT_COL_MAP.t_mant),
            ("t_blo", NOISY_QUANT_COL_MAP.t_blo),
            ("t_exp", NOISY_QUANT_COL_MAP.t_exp),
            ("noised_exp", NOISY_QUANT_COL_MAP.noised_exp),
            ("noised_mant", NOISY_QUANT_COL_MAP.noised_mant),
            ("f_sat", NOISY_QUANT_COL_MAP.f_sat),
            ("q", NOISY_QUANT_COL_MAP.q),
            ("cast_blo", NOISY_QUANT_COL_MAP.cast_blo),
            ("is_sat_eff", NOISY_QUANT_COL_MAP.is_sat_eff),
            ("out", NOISY_QUANT_COL_MAP.out),
        ];
        for (name, col) in cases {
            let mut forged = rows.clone();
            forged[0][col] += F::ONE;
            assert!(constraints_violated(&stark, &forged), "{name} tamper undetected");
        }
    }

    /// Binade-bottom uniqueness: at a power-of-two rounding the former (relaxed) witness — the lower
    /// boundary WITHOUT the `+BOTTOM` correction, differing by exactly the former quarter-ulp slack —
    /// must be REJECTED, while the honest (corrected) trace passes. `noised = 1.0` makes `q = 2^10`
    /// (FP16 1.0, binade bottom); a crafted `beta`/`noise` makes `M_t` a power of two for G1.
    #[test]
    fn binade_bottom_correction_pins_the_rounding() {
        // G3: noised = 1.0 -> q = 2^10, q_bottom = 1. G1: beta = 1.0, noise = 2.0 -> bf*noise = 2.0,
        // M_t = 2^23 (power of two), t_bottom = 1.
        let items = vec![(f32_to_bf16(1.0).unwrap(), 0x3C00u16, f32_to_bf16(1.0).unwrap(), 2.0f32, 1.0f32)];
        let (program, rows) = trace_from(&items);
        let stark = Stk::new(program);
        assert!(!constraints_violated(&stark, &rows), "honest power-of-two trace must pass");
        assert_eq!(to_u64(rows[0][NOISY_QUANT_COL_MAP.q_bottom]), 1, "q is a power of two");
        assert_eq!(to_u64(rows[0][NOISY_QUANT_COL_MAP.t_bottom]), 1, "M_t is a power of two");

        // (a) Revert each lower boundary to the uncorrected value (subtract one factor: the cast
        // factor 2^13, the G1 factor 2^d1 = t_pow). The AIR's `blo = (... + BOTTOM)*factor` rejects it.
        let mut forged = rows.clone();
        forged[0][NOISY_QUANT_COL_MAP.cast_blo] -= F::from_canonical_u64(1 << 13);
        assert!(constraints_violated(&stark, &forged), "cast: uncorrected half-ulp boundary must be rejected");
        let mut forged = rows.clone();
        let tpow = rows[0][NOISY_QUANT_COL_MAP.t_pow];
        forged[0][NOISY_QUANT_COL_MAP.t_blo] -= tpow;
        assert!(constraints_violated(&stark, &forged), "G1: uncorrected half-ulp boundary must be rejected");

        // (b) Clearing either BOTTOM flag at a power of two violates the is-zero gadget.
        let mut forged = rows.clone();
        forged[0][NOISY_QUANT_COL_MAP.q_bottom] = F::ZERO;
        assert!(constraints_violated(&stark, &forged), "q_bottom must stay pinned to 1");
        let mut forged = rows.clone();
        forged[0][NOISY_QUANT_COL_MAP.t_bottom] = F::ZERO;
        assert!(constraints_violated(&stark, &forged), "t_bottom must stay pinned to 1");
    }

    /// RNE uniqueness (anti-grind): at an exact ties-to-even boundary, the honest `q` is the EVEN
    /// neighbor; the odd neighbor is the "near-miss" that the former quarter-ulp grinding slack would
    /// have admitted. A fully-consistent forged witness for the odd `q` still satisfies every
    /// *arithmetic* constraint — it is the RANGE16 bound on the bracket slack (which goes negative
    /// => out of domain) that rejects it. This demonstrates the rounding is pinned with no residual
    /// slack.
    #[test]
    fn cast_rne_is_uniquely_pinned_by_the_slack_range() {
        // noised = f32(E=130, mant=4096): full = 2^23 + 4096, low 13 bits = 0x1000 (an exact tie),
        // full>>13 = 1024 (EVEN) -> honest q = 1024 (ties to even, rounds down).
        let noised = f32::from_bits((130u32 << 23) | 4096);
        let items = vec![(f32_to_bf16(1.0).unwrap(), 0x3C00u16, f32_to_bf16(0.5).unwrap(), 2.0f32, noised)];
        let (program, rows) = trace_from(&items);
        let stark = Stk::new(program.clone());
        assert_eq!(to_u64(rows[0][NOISY_QUANT_COL_MAP.q]), 1024, "honest q is the even neighbor");
        assert!(!constraints_violated(&stark, &rows), "honest tie trace passes");

        // Honest trace: every RANGE16 key is in domain.
        let lu = noisy_quant_lut_lookups::<F>();
        let in_domain = |rows: &[[F; NUM_NOISY_QUANT_COLUMNS]]| -> bool {
            let polys = trace_rows_to_poly_values(rows.to_vec());
            lu.iter().filter(|l| l.table == LutTable::Range16).all(|l| {
                (0..program.live_rows()).all(|r| l.keys[0].eval_table(&polys, r, &[]).to_canonical_u64() < 1 << 16)
            })
        };
        assert!(in_domain(&rows), "honest tie trace keys all in RANGE16 domain");

        // Forge the ODD neighbor q = 1025 with a fully consistent arithmetic witness.
        let mut f = rows.clone();
        let set = |f: &mut [[F; NUM_NOISY_QUANT_COLUMNS]], col: usize, val: u64| {
            f[0][col] = F::from_canonical_u64(val);
        };
        set(&mut f, NOISY_QUANT_COL_MAP.q, 1025);
        set(&mut f, NOISY_QUANT_COL_MAP.q_parity, 1);
        set(&mut f, NOISY_QUANT_COL_MAP.q_half, 512);
        set(&mut f, NOISY_QUANT_COL_MAP.q_bottom, 0);
        f[0][NOISY_QUANT_COL_MAP.q_bottom_inv] = F::ONE; // inv(1025 - 1024)
        set(&mut f, NOISY_QUANT_COL_MAP.q_lo_slack, 1);
        set(&mut f, NOISY_QUANT_COL_MAP.cast_blo, (4 * 1025 - 2) * (1 << 13));
        set(&mut f, NOISY_QUANT_COL_MAP.cast_bhi, (4 * 1025 + 2) * (1 << 13));
        set(&mut f, NOISY_QUANT_COL_MAP.carry, 0);
        f[0][NOISY_QUANT_COL_MAP.carry_inv] = F::from_canonical_u64(2048 - 1025).inverse();
        set(&mut f, NOISY_QUANT_COL_MAP.mf, 1);
        // Upper slack stays in range (0x7FFF); the lower slack is the one that goes negative.
        set(&mut f, NOISY_QUANT_COL_MAP.cast_su_hi, 0);
        set(&mut f, NOISY_QUANT_COL_MAP.cast_sl_hi, 0);
        // Output encode for the forged (odd) q: exp field 18, mantissa 1.
        set(&mut f, NOISY_QUANT_COL_MAP.out, (18 << 10) | 1);

        // The arithmetic constraints alone do NOT catch the near-miss...
        assert!(!constraints_violated(&stark, &f), "arithmetic alone cannot reject the odd neighbor");
        // ...but the RANGE16 bound on the (now negative) lower bracket slack does.
        assert!(!in_domain(&f), "the forged near-miss slack leaves RANGE16's domain (RNE is pinned)");
    }

    /// Every committed-LUT key the honest trace presents lands in its table's domain (RANGE16 keys
    /// `< 2^16`, FP16POW2 keys `<= 255` with `value = 2^min(key, 26)`).
    #[test]
    fn honest_lut_keys_are_in_domain() {
        let items: Vec<_> = (0..24).map(sample).collect();
        let (program, rows) = trace_from(&items);
        let live = program.live_rows();
        let polys = trace_rows_to_poly_values(rows);
        let lu = noisy_quant_lut_lookups::<F>();
        for (li, lookup) in lu.iter().enumerate() {
            for r in 0..live {
                match lookup.table {
                    LutTable::Range16 => {
                        let k = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                        assert!(k < 1 << 16, "RANGE16 lookup {li} key {k} out of range at row {r}");
                    }
                    LutTable::Fp16Pow2 => {
                        let k = lookup.keys[0].eval_table(&polys, r, &[]).to_canonical_u64();
                        let v = lookup.values[0].eval_table(&polys, r, &[]).to_canonical_u64();
                        assert!(k <= 255, "FP16POW2 key {k} out of range at row {r}");
                        assert_eq!(v, 1 << k.min(26), "FP16POW2 value {v} != 2^min({k},26) at row {r}");
                    }
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
