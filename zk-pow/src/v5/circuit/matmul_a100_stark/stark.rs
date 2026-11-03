//! Proves the NVIDIA A100 (`sm_80`) `HMMA.16816.F32` FP16 matmul accumulation,
//! one `G = 8` group per trace row.
//!
//! This AIR is the ZK analogue of `crate::v5::api::accumulate::a100_dot`, which
//! it must match bit-for-bit. See `docs/fp16_scheme/stark_feasibility.md` for the
//! derivation and the GO-WITH-CHANGES verdict; the structure forks
//! `circuit::fp8::matmul_b200_stark`, with the two documented changes:
//!
//! * FP16 operand pairs span `2^32`, so there is no operand-keyed product LUT.
//!   Each lane decodes its two operands (an `FP16DECODE` LUT, 2^16 keys), the
//!   product significand is an in-AIR multiply, and the alignment is an in-AIR
//!   Euclidean floor against a `POW2` LUT — exactly B200's carry-alignment gadget,
//!   applied per lane.
//! * `G = 8` gives `k/8` rows per cell (4x B200). For production-size tiles the
//!   feasibility doc recommends packing several groups per row; the delivered AIR
//!   keeps one group per row (correct; test geometries stay small).
//!
//! # Exponent axis
//!
//! Biased by +127 (the FP32 bias). A nonzero product's `e_u = eps(a)+eps(b)` maps
//! to `PRODUCT_BIASED_EXP = e_u + 127 in [99, 157]`; the carry's clamped exponent
//! `el` maps to `el + 127 in [1, 254]` (the FP32 exponent field). Sentinel 0 marks
//! a zero product / zero carry.
//!
//! # One group step (live row)
//!
//! `eta = max(PRODUCT_BIASED_EXP over nonzero lanes, carry exponent)`; the window
//! unit is `2^(eta - 24)`. Each lane contributes `±floor(P*16 / 2^rel)`,
//! `rel = eta - PRODUCT_BIASED_EXP` (the `*16` lifts the product's
//! `2^(e_u-20)` scale onto the window; `P*16 < 2^26` so `rel >= 26` drops it). The
//! carry contributes `±floor(2*cm / 2^rel_c)` (`2*cm < 2^25`). The 8 terms and the
//! carry sum exactly (`|sum| < 2^30`); the sum is truncated toward zero to a 24-bit
//! significand and re-biased to FP32. A cell-final row exports the FP32 word.
//!
//! # Scope (honest)
//!
//! The normal, zero, and subnormal output paths are all constrained. The normal and
//! zero paths cover the 238 GPU-validated reference vectors (237 normal + 1 zero).
//! Subnormal FP32 outputs (value `< 2^-126`) take a distinct RZ branch — exponent field
//! 0, mantissa `NORM_SIG >> k` on the `2^-149` grid — now written as MA13 and gated by
//! the `out_is_subnormal` flag, which MA13 pins to `[raw exp <= 0]` via a nonnegative
//! RANGE16 slack. The from-zero datapath never reaches the subnormal branch (FP16
//! products align at `eta >= 99`, so a group output floors at `~2^-52`), so MA13 is a
//! sound guard there; the branch is exercised by the subnormal carry-in the oracle
//! `a100_dot` models, and the generator computes it bit-exactly.
//!
//! The semantic auxiliary columns — the per-operand decode (`sig_*`, `sign_*`,
//! `eps_*`, `is_zero_*`), the per-lane/carry alignment powers (`*_shift_power`) and
//! the RZ truncate/lift powers (`truncation_power`/`lifting_power`), plus the FP32
//! result limbs — are enforced by committed-LUT cross-table lookups (see
//! [`super::ctl`] and [`crate::v5::circuit::ctl`]): FP16DECODE, FP16POW2, WIDTH32,
//! and RANGE16. The per-step census that feeds the policy AIR is pinned bit-exactly
//! by MA3/MA11. The 16/10-bit limb-range RANGE16 splits (MA12) make the per-lane and
//! carry Euclidean floors integer-exact against field-fraction aliasing (the
//! `aligned_*`/`*_rem`/`*_bound` 26-bit magnitudes and the RZ `norm_sig`/`trunc_rem`),
//! so the AIR is sound under a real FRI proof — see [`super::ctl`] and
//! [`crate::v5::circuit::driver`]. **Remaining follow-on:** the recursive FRI wrapper
//! to a constant-size proof.

use core::borrow::Borrow;
use std::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
#[allow(unused_imports)]
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use starky::stark::Stark;

#[allow(unused_imports)]
use super::columns::MATMUL_A100_COL_MAP;
use super::columns::{
    GROUP, MatmulA100ColumnsView, NUM_ATT_LINKS, NUM_MATMUL_A100_COLUMNS, NUM_MATMUL_A100_PUBLIC_INPUTS,
};
use crate::v5::api::dtype::fp16_decode_fields;
use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// Internal accumulator precision (FP32 significand bits).
const W: u64 = 24;
/// The window keeps 24 fractional bits: shifts of a lifted operand cap here, since
/// `P*16 < 2^26` and `2*cm < 2^25` are both floored to 0 by any larger shift.
const SHIFT_CAP: u64 = 26;
/// `GROUP_MAX_BIASED_EXPONENT + W - 25`: the `-25` is the 24 window bits plus the
/// leading-bit position, mirroring B200's `-26` at its 25-bit window.
const OUT_EXP_OFFSET: u64 = 25;
/// Normalized significands live in `[2^23, 2^24)`.
const SIG24_MIN: u64 = 1 << 23;

// ==================================================================================================
// Program geometry
// ==================================================================================================

/// The A100 `HMMA` AIR for one FP16 matmul: `out[r,c] = sum_t A[r,t] * B[t,c]`.
#[derive(Clone, Debug)]
pub struct MatmulStarkA100<F: RichField + Extendable<D>, const D: usize> {
    /// Output rows (rows of A).
    pub h: usize,
    /// Output columns (rows of B — B is the transposed logical operand).
    pub w: usize,
    /// Inner dimension; must be a multiple of [`GROUP`].
    pub k: usize,
    /// Committed trace height (a power of two `>= live_rows`). Defaults to the next power of
    /// two above the live rows ([`Self::new`]); a larger on-ladder height can be forced with
    /// [`Self::new_with_height`] (used by the FP16 batch's noise matmul to keep its height on the
    /// consensus FRI fold ladder). The extra rows are single-row phantom cells.
    height: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> MatmulStarkA100<F, D> {
    pub fn new(h: usize, w: usize, k: usize) -> Self {
        assert_eq!(k % GROUP, 0, "k must be a multiple of the group size {GROUP}");
        let height = (h * w * (k / GROUP)).next_power_of_two();
        Self { h, w, k, height, _phantom: PhantomData }
    }

    /// Like [`Self::new`] but pads the trace to `height` (a power of two `>= live_rows`) rather
    /// than only to the next power of two. The FP16 batch uses this for the `N = E@F^T` noise
    /// matmul so its committed height lands on a [`crate::v5::circuit::driver::FP16_REACHABLE_DEGREE_BITS`]
    /// member (the next power of two is not always on the ladder); the extra rows are phantom cells.
    pub fn new_with_height(h: usize, w: usize, k: usize, height: usize) -> Self {
        assert_eq!(k % GROUP, 0, "k must be a multiple of the group size {GROUP}");
        assert!(height.is_power_of_two(), "the forced height must be a power of two");
        assert!(height >= h * w * (k / GROUP), "the forced height must cover the live rows");
        Self { h, w, k, height, _phantom: PhantomData }
    }

    /// Trace rows per output cell: its `k/8` live group-steps.
    pub fn rows_per_cell(&self) -> usize {
        self.k / GROUP
    }

    /// Rows covering real cells (`h*w` cells, row-major).
    pub fn live_rows(&self) -> usize {
        self.h * self.w * self.rows_per_cell()
    }

    /// Trace height: live rows padded to [`Self::height`] with single-row phantom cells.
    pub fn num_rows(&self) -> usize {
        self.height
    }

    /// The class (a) ("known") columns — pure functions of the AIR geometry,
    /// bit-exact with [`Self::generate_trace`]'s fill.
    pub fn known_values(&self) -> Vec<PolynomialValues<F>> {
        let (h, w, k) = (self.h, self.w, self.k);
        let rpc = self.rows_per_cell();
        let num_rows = self.num_rows();
        let mut cell_id = Vec::with_capacity(num_rows);
        let mut is_cell_final = Vec::with_capacity(num_rows);
        let mut base_a = Vec::with_capacity(num_rows);
        let mut base_b = Vec::with_capacity(num_rows);
        let mut is_padding = Vec::with_capacity(num_rows);
        for r in 0..h {
            for c in 0..w {
                for j in 0..rpc {
                    cell_id.push(F::from_canonical_usize(r * w + c));
                    is_cell_final.push(F::from_bool(j == rpc - 1));
                    base_a.push(F::from_canonical_usize(r * k + j * GROUP));
                    base_b.push(F::from_canonical_usize(h * k + c * k + j * GROUP));
                    is_padding.push(F::ZERO);
                }
            }
        }
        for t in 0..num_rows - self.live_rows() {
            cell_id.push(F::from_canonical_usize(h * w + t));
            is_cell_final.push(F::ONE);
            base_a.push(F::ZERO);
            base_b.push(F::ZERO);
            is_padding.push(F::ONE);
        }
        [cell_id, is_cell_final, base_a, base_b, is_padding]
            .into_iter()
            .map(PolynomialValues::new)
            .collect()
    }
}

// ==================================================================================================
// Decoded operands and the carry triple
// ==================================================================================================

/// One decoded FP16 x FP16 product. The per-operand fields (`sig_*`, `sign_*`, `eps_*`,
/// `is_zero_*`) are exactly the FP16DECODE LUT outputs CTL-bound into the trace; everything
/// else is the in-AIR derivation MA1 re-checks against them.
#[derive(Clone, Copy, Debug)]
struct A100Product {
    sig_a: u64,
    sig_b: u64,
    /// Raw sign bits of the two operands (FP16DECODE value 1).
    sign_a: bool,
    sign_b: bool,
    /// Biased stored exponents `eps + 15 in [1, 30]` (FP16DECODE value 2).
    eps_a: u64,
    eps_b: u64,
    /// `[sig == 0]` per operand (FP16DECODE value 3).
    is_zero_a: bool,
    is_zero_b: bool,
    /// `sig_a * sig_b < 2^22`.
    product_sig: u64,
    /// `sign_a xor sign_b` (irrelevant on zero lanes, whose aligned term is 0).
    sign: bool,
    /// `ea + eb + 127 in [99, 157]`, or 0 for a zero product.
    biased_exp: u64,
    is_zero: bool,
}

impl A100Product {
    fn new(code_a: u16, code_b: u16) -> Self {
        let (sig_a, sign_a, eps_a, zero_a) = fp16_decode_fields(code_a);
        let (sig_b, sign_b, eps_b, zero_b) = fp16_decode_fields(code_b);
        let (is_zero_a, is_zero_b) = (zero_a == 1, zero_b == 1);
        let is_zero = is_zero_a || is_zero_b;
        A100Product {
            sig_a,
            sig_b,
            sign_a: sign_a == 1,
            sign_b: sign_b == 1,
            eps_a,
            eps_b,
            is_zero_a,
            is_zero_b,
            product_sig: sig_a * sig_b,
            sign: (sign_a == 1) != (sign_b == 1),
            // (eps_a+eps_b+97) = (ea+15)+(eb+15)+97 = ea+eb+127; gated to 0 on zero lanes.
            biased_exp: if is_zero { 0 } else { eps_a + eps_b + 97 },
            is_zero,
        }
    }
}

/// A group output / the carry entering the next row: `value = sign * cm * 2^(el-23)`
/// with `el = biased_exp - 127`. Mirrors `acc_parts` of the oracle.
#[derive(Clone, Copy, Debug)]
struct Carry {
    sign: bool,
    /// 24-bit significand (normal), subnormal mantissa, or 0.
    cm: u64,
    /// `el + 127` (= the FP32 exponent field); 0 for a zero carry.
    biased_exp: u64,
    is_zero: bool,
    is_subnormal: bool,
}

impl Carry {
    const ZERO: Self = Self { sign: false, cm: 0, biased_exp: 0, is_zero: true, is_subnormal: false };

    /// Decomposes an incoming FP32 accumulator (the oracle's `acc_parts`).
    fn from_f32(c: f32) -> Self {
        debug_assert!(c.is_finite());
        if c == 0.0 {
            return Self::ZERO;
        }
        let bits = c.to_bits();
        let sign = c.is_sign_negative();
        let exp_field = (bits >> 23) & 0xFF;
        let man = (bits & 0x7F_FFFF) as u64;
        if exp_field > 0 {
            Self { sign, cm: 0x80_0000 | man, biased_exp: exp_field as u64, is_zero: false, is_subnormal: false }
        } else {
            // Subnormal: el clamped to -126 -> biased_exp 1, cm = mantissa.
            Self { sign, cm: man, biased_exp: 1, is_zero: false, is_subnormal: true }
        }
    }

    /// The exact FP32 bit pattern of this carry.
    fn to_f32_bits(self) -> u64 {
        if self.cm == 0 {
            return 0;
        }
        if self.is_subnormal {
            return ((self.sign as u64) << 31) | self.cm;
        }
        ((self.sign as u64) << 31) | (self.biased_exp << 23) | (self.cm - SIG24_MIN)
    }
}

fn bit_length(x: u64) -> u64 {
    (64 - x.leading_zeros()) as u64
}

// ==================================================================================================
// Trace generation
// ==================================================================================================

impl<F: RichField + Extendable<D>, const D: usize> MatmulStarkA100<F, D> {
    /// Generates the trace: `a_codes` row-major (`h*k`), `b_codes` row-major over
    /// the transposed B (`w*k`), optional `m*n` FP32 carry-in. Cells row-major, one
    /// row per `G = 8` group step. The committed cell results match
    /// [`crate::v5::api::accumulate::a100_matmul`] bit-for-bit.
    pub fn generate_trace(
        &self,
        a_codes: &[u16],
        b_codes: &[u16],
        acc: Option<&[f32]>,
    ) -> Vec<[F; NUM_MATMUL_A100_COLUMNS]> {
        let (h, w, k) = (self.h, self.w, self.k);
        assert_eq!(a_codes.len(), h * k, "a_codes must be h*k FP16 codes");
        assert_eq!(b_codes.len(), w * k, "b_codes must be w*k FP16 codes");
        let rpc = self.rows_per_cell();
        let num_rows = self.num_rows();
        let mut rows: Vec<[F; NUM_MATMUL_A100_COLUMNS]> = Vec::with_capacity(num_rows);

        for r in 0..h {
            for c in 0..w {
                let cell_id = r * w + c;
                let mut carry = Carry::from_f32(acc.map_or(0.0, |a| a[r * w + c]));
                let mut incoming_carry_is_zero = carry.is_zero;
                let cell_start = rows.len();
                let mut cell_pt = 0u64;
                let mut cell_bp = 0u64;

                for j in 0..rpc {
                    let is_final = j == rpc - 1;
                    let mut row = MatmulA100ColumnsView::<F> {
                        cell_id: F::from_canonical_usize(cell_id),
                        is_cell_final: F::from_bool(is_final),
                        operand_index_base_a: F::from_canonical_usize(r * k + j * GROUP),
                        operand_index_base_b: F::from_canonical_usize(h * k + c * k + j * GROUP),
                        incoming_carry_is_zero: F::from_bool(incoming_carry_is_zero),
                        ..Default::default()
                    };

                    // Decode the 8 lanes.
                    let mut lanes = [A100Product::new(0, 0); GROUP];
                    for i in 0..GROUP {
                        let ca = a_codes[r * k + j * GROUP + i];
                        let cb = b_codes[c * k + j * GROUP + i];
                        lanes[i] = A100Product::new(ca, cb);
                        row.operand_codes_a[i] = F::from_canonical_u16(ca);
                        row.operand_codes_b[i] = F::from_canonical_u16(cb);
                        row.sig_a[i] = F::from_canonical_u64(lanes[i].sig_a);
                        row.sig_b[i] = F::from_canonical_u64(lanes[i].sig_b);
                        row.sign_a[i] = F::from_bool(lanes[i].sign_a);
                        row.sign_b[i] = F::from_bool(lanes[i].sign_b);
                        row.eps_a[i] = F::from_canonical_u64(lanes[i].eps_a);
                        row.eps_b[i] = F::from_canonical_u64(lanes[i].eps_b);
                        row.is_zero_a[i] = F::from_bool(lanes[i].is_zero_a);
                        row.is_zero_b[i] = F::from_bool(lanes[i].is_zero_b);
                        row.lane_sign[i] = F::from_bool(lanes[i].sign);
                        row.product_biased_exp[i] = F::from_canonical_u64(lanes[i].biased_exp);
                        row.product_sig[i] = F::from_canonical_u64(lanes[i].product_sig);
                    }

                    // Window anchor eta (biased): max over nonzero products and the carry.
                    let mut eta: u64 = if incoming_carry_is_zero { 0 } else { carry.biased_exp };
                    for lane in &lanes {
                        if !lane.is_zero {
                            eta = eta.max(lane.biased_exp);
                        }
                    }
                    row.group_max_biased_exponent = F::from_canonical_u64(eta);
                    let nonempty = eta != 0;
                    row.group_nonempty = F::from_bool(nonempty);
                    row.group_max_inv = if eta == 0 { F::ZERO } else { F::from_canonical_u64(eta).inverse() };

                    // Attainment chain over the 8 lane factors.
                    let factor = |i: usize| F::from_canonical_u64(eta) - row.product_biased_exp[i];
                    row.max_exponent_attainment[0] = factor(0) * factor(1) * factor(2);
                    row.max_exponent_attainment[1] = row.max_exponent_attainment[0] * factor(3) * factor(4);
                    row.max_exponent_attainment[2] = row.max_exponent_attainment[1] * factor(5) * factor(6);
                    row.max_exponent_attainment[3] = row.max_exponent_attainment[2] * factor(7);

                    // Align + sum the 8 products.
                    let mut sum: i64 = 0;
                    for i in 0..GROUP {
                        if lanes[i].is_zero {
                            // Zero lane: honest shift power keeps the Euclidean identity 0 = 0.
                            let rel = eta.saturating_sub(lanes[i].biased_exp).min(SHIFT_CAP);
                            row.lane_shift_power[i] = F::from_canonical_u64(1 << rel);
                            row.lane_rem_bound[i] = F::from_canonical_u64((1 << rel) - 1);
                            continue;
                        }
                        let rel = (eta - lanes[i].biased_exp).min(SHIFT_CAP);
                        let sp = 1u64 << rel;
                        let p16 = lanes[i].product_sig * 16;
                        let aligned = p16 / sp;
                        let rem = p16 - aligned * sp;
                        row.lane_shift_power[i] = F::from_canonical_u64(sp);
                        row.aligned_mag[i] = F::from_canonical_u64(aligned);
                        row.lane_rem[i] = F::from_canonical_u64(rem);
                        row.lane_rem_bound[i] = F::from_canonical_u64(sp - 1 - rem);
                        let truncated = rem != 0;
                        row.products_truncated_flag[i] = F::from_bool(truncated);
                        if truncated {
                            row.lane_rem_inv[i] = F::from_canonical_u64(rem).inverse();
                        }
                        cell_pt += u64::from(truncated);
                        sum += if lanes[i].sign { -(aligned as i64) } else { aligned as i64 };
                    }

                    // Align + add the carry.
                    let mut carry_dropped = false;
                    if incoming_carry_is_zero {
                        row.carry_shift_power = F::ONE; // inert; remainder identity picks (0,0,1).
                    } else {
                        let rel = (eta - carry.biased_exp).min(SHIFT_CAP);
                        let sp = 1u64 << rel;
                        let two_cm = 2 * carry.cm;
                        let aligned = two_cm / sp;
                        let rem = two_cm - aligned * sp;
                        row.carry_shift_power = F::from_canonical_u64(sp);
                        row.aligned_carry = F::from_canonical_u64(aligned);
                        row.carry_rem = F::from_canonical_u64(rem);
                        row.carry_rem_bound = F::from_canonical_u64(sp - 1 - rem);
                        carry_dropped = rem != 0;
                        if carry_dropped {
                            row.carry_rem_inv = F::from_canonical_u64(rem).inverse();
                        }
                        sum += if carry.sign { -(aligned as i64) } else { aligned as i64 };
                    }
                    row.carry_dropped = F::from_bool(carry_dropped);

                    // Signed sum + round toward zero to FP32.
                    row.group_sum_sign = F::from_bool(sum < 0);
                    row.group_sum_abs = F::from_canonical_u64(sum.unsigned_abs());
                    row.group_sum_is_zero = F::from_bool(sum == 0);
                    debug_assert!(sum.unsigned_abs() < 1 << 30, "window sums fit 30 bits");

                    let mut rz_dropped = false;
                    let out = if sum == 0 {
                        row.truncation_power = F::ONE;
                        row.lifting_power = F::ONE;
                        row.sub_shift_power = F::ONE;
                        Carry::ZERO
                    } else {
                        let mag = sum.unsigned_abs();
                        let width = bit_length(mag);
                        let tp = 1u64 << width.saturating_sub(W);
                        let lp = 1u64 << W.saturating_sub(width);
                        let norm_sig = mag * lp / tp;
                        let trunc_rem = mag * lp - norm_sig * tp;
                        row.group_sum_width = F::from_canonical_u64(width);
                        row.truncation_power = F::from_canonical_u64(tp);
                        row.lifting_power = F::from_canonical_u64(lp);
                        row.norm_sig = F::from_canonical_u64(norm_sig);
                        row.trunc_rem = F::from_canonical_u64(trunc_rem);
                        row.trunc_rem_bound = F::from_canonical_u64(tp - 1 - trunc_rem);
                        if trunc_rem != 0 {
                            row.trunc_rem_inv = F::from_canonical_u64(trunc_rem).inverse();
                            rz_dropped = true;
                        }
                        // `raw` is the normal-path FP32 biased exponent `eta + W - 25`. When it is
                        // `>= 1` the output is a normal FP32; when it is `<= 0` the output is an
                        // FP32 subnormal (`|x| < 2^-126`), encoded on the `2^-149` grid with
                        // exponent field 0 and a `< 2^23` mantissa = `NORM_SIG >> k`, `k = 1 - raw`
                        // (MA13). The from-zero datapath never takes the subnormal branch (FP16
                        // products align at `eta >= 99`, so a group output floors at `~2^-52`); it
                        // is reached only by a subnormal carry-in (the accumulation datapath the
                        // oracle `a100_dot` models). `exp_slack` is the nonnegative flag witness.
                        let raw = eta as i64 + width as i64 - OUT_EXP_OFFSET as i64;
                        if raw >= 1 {
                            let out_biased_exp = raw as u64;
                            row.exp_slack = F::from_canonical_u64(out_biased_exp - 1);
                            row.sub_shift_power = F::ONE;
                            Carry { sign: sum < 0, cm: norm_sig, biased_exp: out_biased_exp, is_zero: false, is_subnormal: false }
                        } else {
                            let k = (1 - raw) as u64; // = 26 - eta - width >= 1
                            let sub_shift_power = 1u64 << k;
                            let mantissa = norm_sig / sub_shift_power;
                            debug_assert_eq!(norm_sig % sub_shift_power, 0, "a reachable subnormal keeps whole NORM_SIG bits");
                            debug_assert!(mantissa < SIG24_MIN, "a subnormal mantissa fits 23 bits");
                            row.out_is_subnormal = F::ONE;
                            row.exp_slack = F::from_canonical_u64(k - 1); // = -raw
                            row.sub_shift_exp = F::from_canonical_u64(k);
                            row.sub_shift_power = F::from_canonical_u64(sub_shift_power);
                            (row.out_sig_lo, row.out_sig_hi) = split_limbs(F::from_canonical_u64(mantissa));
                            Carry { sign: sum < 0, cm: mantissa, biased_exp: 0, is_zero: mantissa == 0, is_subnormal: true }
                        }
                    };
                    row.rz_dropped = F::from_bool(rz_dropped);
                    let breakpoint = nonempty && (carry_dropped || rz_dropped);
                    row.out_sign = F::from_bool(out.sign);
                    row.out_biased_exp = F::from_canonical_u64(out.biased_exp);
                    row.out_sig = F::from_canonical_u64(out.cm);
                    row.group_breakpoint = F::from_bool(breakpoint);
                    cell_bp += u64::from(breakpoint);
                    row.cell_products_truncated = F::from_canonical_u64(cell_pt);
                    row.cell_breakpoints = F::from_canonical_u64(cell_bp);

                    if is_final {
                        let bits = out.to_f32_bits();
                        row.cell_result_f32_lo = F::from_canonical_u64(bits & 0xFFFF);
                        row.cell_result_f32_hi = F::from_canonical_u64(bits >> 16);
                    }

                    incoming_carry_is_zero = sum == 0;
                    carry = out;
                    fill_limbs(&mut row);
                    rows.push(row.into());
                }
                let _ = cell_start;
            }
        }

        for t in 0..num_rows - self.live_rows() {
            rows.push(phantom_row::<F>(h * w + t).into());
        }
        rows
    }
}

/// An all-zero single-row trailing phantom cell.
fn phantom_row<F: RichField>(cell_id: usize) -> MatmulA100ColumnsView<F> {
    let mut row = MatmulA100ColumnsView::<F> {
        cell_id: F::from_canonical_usize(cell_id),
        is_cell_final: F::ONE,
        is_padding: F::ONE,
        incoming_carry_is_zero: F::ONE,
        carry_shift_power: F::ONE,
        group_sum_is_zero: F::ONE,
        truncation_power: F::ONE,
        lifting_power: F::ONE,
        ..Default::default()
    };
    for i in 0..GROUP {
        row.lane_shift_power[i] = F::ONE;
        // Phantom lanes read operand code 0, which FP16DECODE serves as (0, 0, 15, 1): both
        // operands are zero, so the product sentinel stays 0 and MA1's derivation holds.
        row.is_zero_a[i] = F::ONE;
        row.is_zero_b[i] = F::ONE;
        row.eps_a[i] = F::from_canonical_u64(15);
        row.eps_b[i] = F::from_canonical_u64(15);
    }
    fill_limbs(&mut row);
    row
}

/// `(value & 0xFFFF, value >> 16)` of a canonical field integer — the `(lo, hi)` limbs MA12
/// reconstructs and RANGE16-checks.
fn split_limbs<F: RichField>(value: F) -> (F, F) {
    let v = value.to_canonical_u64();
    (F::from_canonical_u64(v & 0xFFFF), F::from_canonical_u64(v >> 16))
}

/// Fills every MA12 limb column from the corresponding Euclidean-floor witness column, so the
/// reconstruction constraints hold on every row (live, zero-lane, and phantom).
fn fill_limbs<F: RichField>(row: &mut MatmulA100ColumnsView<F>) {
    for i in 0..GROUP {
        (row.aligned_mag_lo[i], row.aligned_mag_hi[i]) = split_limbs(row.aligned_mag[i]);
        (row.lane_rem_lo[i], row.lane_rem_hi[i]) = split_limbs(row.lane_rem[i]);
        (row.lane_rem_bound_lo[i], row.lane_rem_bound_hi[i]) = split_limbs(row.lane_rem_bound[i]);
    }
    (row.aligned_carry_lo, row.aligned_carry_hi) = split_limbs(row.aligned_carry);
    (row.carry_rem_lo, row.carry_rem_hi) = split_limbs(row.carry_rem);
    (row.carry_rem_bound_lo, row.carry_rem_bound_hi) = split_limbs(row.carry_rem_bound);
    (row.norm_sig_lo, row.norm_sig_hi) = split_limbs(row.norm_sig);
}

// ==================================================================================================
// Constraints
// ==================================================================================================

/// `1 - 2*bit`.
fn double_complement<V: Copy, S: Copy, E: Evaluator<V, S>>(eval: &mut E, one: V, bit: V) -> V {
    let twice = eval.add(bit, bit);
    eval.sub(one, twice)
}

/// Evaluates every arithmetic constraint of `MatmulStarkA100` (degree <= 3). LUT
/// facts (decode, POW2, WIDTH, range) are not emitted here; see the module docs.
pub(crate) fn eval_matmul_a100_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_MATMUL_A100_COLUMNS, NUM_MATMUL_A100_PUBLIC_INPUTS>,
    eval: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_MATMUL_A100_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &MatmulA100ColumnsView<V> = lv.borrow();
    let nv: &[V; NUM_MATMUL_A100_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let nv: &MatmulA100ColumnsView<V> = nv.borrow();

    let one = eval.i32(1);
    let not_final = eval.sub(one, lv.is_cell_final);
    let z = lv.group_sum_is_zero;
    let one_minus_z = eval.sub(one, z);
    let limb_shift = eval.u64(1 << 16);

    eval.constraint_bool(lv.is_padding);

    // ---- MA1 — product significand + decode derivations + per-lane booleans. The decoded
    // per-operand fields (`sig_*`, `sign_*`, `eps_*`, `is_zero_*`) are FP16DECODE LUT outputs
    // CTL-bound to the operand codes (see `super::ctl`); here they are turned into the lane's
    // product significand, sign, and biased stored exponent, so a prover cannot forge the sign
    // or exponent a code decodes to. ----
    for i in 0..GROUP {
        // product_sig = sig_a * sig_b.
        let prod = eval.mul(lv.sig_a[i], lv.sig_b[i]);
        let c = eval.sub(lv.product_sig[i], prod);
        eval.constraint(c);
        eval.constraint_bool(lv.sign_a[i]);
        eval.constraint_bool(lv.sign_b[i]);
        eval.constraint_bool(lv.is_zero_a[i]);
        eval.constraint_bool(lv.is_zero_b[i]);
        eval.constraint_bool(lv.lane_sign[i]);
        eval.constraint_bool(lv.products_truncated_flag[i]);
        // lane_sign = sign_a XOR sign_b = sign_a + sign_b - 2*sign_a*sign_b.
        let sasb = eval.mul(lv.sign_a[i], lv.sign_b[i]);
        let two_sasb = eval.add(sasb, sasb);
        let xor = eval.add(lv.sign_a[i], lv.sign_b[i]);
        let xor = eval.sub(xor, two_sasb);
        let c = eval.sub(lv.lane_sign[i], xor);
        eval.constraint(c);
        // product_biased_exp = (1-is_zero_a)*(1-is_zero_b) * (eps_a + eps_b + 97), the
        // zero-sentinel being the all-or-either-zero case (padding reads code 0 => is_zero,
        // so this subsumes the old padding pin). 97 = 127 - 2*15 re-biases eps+15 twins to f32.
        let ninety_seven = eval.i32(97);
        let eps_sum = eval.add(lv.eps_a[i], lv.eps_b[i]);
        let eps_sum = eval.add(eps_sum, ninety_seven);
        let nz_a = eval.sub(one, lv.is_zero_a[i]);
        let nz_b = eval.sub(one, lv.is_zero_b[i]);
        let nz = eval.mul(nz_a, nz_b);
        let expect = eval.mul(nz, eps_sum);
        let c = eval.sub(lv.product_biased_exp[i], expect);
        eval.constraint(c);
    }

    // ---- MA2 — eta attainment (the "<=" side of the max; ">=" is the POW2 key
    // domain). Links are local; the closing (reading the next row as the carry's
    // consumer) multiplies in the zero-gated carry factor and demands the product
    // vanish. ----
    let factor = |eval: &mut E, view: &MatmulA100ColumnsView<V>, i: usize| {
        eval.sub(view.group_max_biased_exponent, view.product_biased_exp[i])
    };
    let f0 = factor(eval, lv, 0);
    let f1 = factor(eval, lv, 1);
    let f2 = factor(eval, lv, 2);
    let f01 = eval.mul(f0, f1);
    let link = eval.msub(f01, f2, lv.max_exponent_attainment[0]);
    eval.constraint(link);
    for l in 1..NUM_ATT_LINKS - 1 {
        let fa = factor(eval, lv, 2 * l + 1);
        let fb = factor(eval, lv, 2 * l + 2);
        let fab = eval.mul(fa, fb);
        let link = eval.msub(lv.max_exponent_attainment[l - 1], fab, lv.max_exponent_attainment[l]);
        eval.constraint(link);
    }
    let f_last = factor(eval, lv, GROUP - 1);
    let link = eval.msub(lv.max_exponent_attainment[NUM_ATT_LINKS - 2], f_last, lv.max_exponent_attainment[NUM_ATT_LINKS - 1]);
    eval.constraint(link);
    // Closing: the next row's chain tail times its zero-gated carry factor vanishes.
    let carry_is_live = eval.sub(one, nv.incoming_carry_is_zero);
    let carry_gap = eval.sub(nv.group_max_biased_exponent, lv.out_biased_exp);
    let carry_factor = eval.mul(carry_is_live, carry_gap);
    let carry_factor = eval.add(carry_factor, nv.incoming_carry_is_zero);
    let closing = eval.mul(nv.max_exponent_attainment[NUM_ATT_LINKS - 1], carry_factor);
    eval.constraint(closing);

    // group_nonempty = [GROUP_MAX_BIASED_EXPONENT != 0].
    eval.constraint_bool(lv.group_nonempty);
    let prod = eval.mul(lv.group_max_biased_exponent, lv.group_max_inv);
    let c = eval.sub(prod, lv.group_nonempty);
    eval.constraint(c);
    let not_nonempty = eval.sub(one, lv.group_nonempty);
    let c = eval.mul(not_nonempty, lv.group_max_biased_exponent);
    eval.constraint(c);

    // ---- MA3 — per-lane Euclidean truncation (local). ----
    for i in 0..GROUP {
        let sixteen = eval.i32(16);
        let p16 = eval.mul(sixteen, lv.product_sig[i]);
        let shifted = eval.mul(lv.aligned_mag[i], lv.lane_shift_power[i]);
        let floor_diff = eval.sub(p16, shifted);
        let floor_diff = eval.sub(floor_diff, lv.lane_rem[i]);
        eval.constraint(floor_diff);
        let rem_sum = eval.add(lv.lane_rem[i], lv.lane_rem_bound[i]);
        let rem_sum = eval.add(rem_sum, one);
        let rem_id = eval.sub(rem_sum, lv.lane_shift_power[i]);
        eval.constraint(rem_id);
        // Census (clean direction): a non-truncated lane has zero remainder.
        let not_trunc = eval.sub(one, lv.products_truncated_flag[i]);
        let c = eval.mul(not_trunc, lv.lane_rem[i]);
        eval.constraint(c);
        // Census (tight direction): a truncated lane has nonzero remainder, pinned by
        // the inverse witness (`flag * (rem*inv - 1) = 0`). Together with the clean
        // direction this makes `products_truncated_flag == [LANE_REM != 0]` exactly, so
        // the per-lane census cannot be inflated to overstate certified work.
        let rem_inv = eval.mul(lv.lane_rem[i], lv.lane_rem_inv[i]);
        let rem_inv_m1 = eval.sub(rem_inv, one);
        let c = eval.mul(lv.products_truncated_flag[i], rem_inv_m1);
        eval.constraint(c);
    }

    // ---- MA4 — carry alignment (anchored at the producing row; reads next row). ----
    let carry_is_nonzero_next = eval.sub(one, nv.incoming_carry_is_zero);
    let two = eval.i32(2);
    let two_sig = eval.mul(two, lv.out_sig);
    let shifted = eval.mul(nv.aligned_carry, nv.carry_shift_power);
    let floor_diff = eval.sub(two_sig, shifted);
    let floor_diff = eval.sub(floor_diff, nv.carry_rem);
    let c = eval.mul(carry_is_nonzero_next, floor_diff);
    eval.constraint(c);
    // Unfiltered remainder identity (local).
    let rem_sum = eval.add(lv.carry_rem, lv.carry_rem_bound);
    let rem_sum = eval.add(rem_sum, one);
    let rem_id = eval.sub(rem_sum, lv.carry_shift_power);
    eval.constraint(rem_id);
    // Zero carries contribute nothing (kills the stale value, incl. the cyclic wrap).
    let kill = eval.mul(lv.incoming_carry_is_zero, lv.aligned_carry);
    eval.constraint(kill);

    // ---- MA5 — signed window sum (anchored at the producing row). ----
    let mut terms: Vec<V> = Vec::with_capacity(GROUP);
    for i in 0..GROUP {
        let sign_factor = double_complement(eval, one, nv.lane_sign[i]);
        terms.push(eval.mul(sign_factor, nv.aligned_mag[i]));
    }
    let terms_sum = eval.sum(&terms);
    let carry_sign_factor = double_complement(eval, one, lv.out_sign);
    let signed_carry = eval.mul(carry_sign_factor, nv.aligned_carry);
    let bracket = eval.add(terms_sum, signed_carry);
    let group_sum_sign_factor = double_complement(eval, one, nv.group_sum_sign);
    let recovered = eval.mul(group_sum_sign_factor, bracket);
    let sum_eq = eval.sub(nv.group_sum_abs, recovered);
    eval.constraint(sum_eq);
    eval.constraint_bool(lv.group_sum_sign);
    eval.constraint_bool(lv.group_sum_is_zero);
    let ban = eval.mul(lv.group_sum_is_zero, lv.group_sum_abs);
    eval.constraint(ban);
    let plus_zero = eval.mul(lv.group_sum_is_zero, lv.group_sum_sign);
    eval.constraint(plus_zero);

    // ---- MA6 — carry-zero flag propagation. ----
    let diff = eval.sub(nv.incoming_carry_is_zero, z);
    let c = eval.mul(not_final, diff);
    eval.constraint(c);
    let reset = eval.sub(nv.incoming_carry_is_zero, one);
    let c = eval.mul(lv.is_cell_final, reset);
    eval.constraint(c);
    let first_anchor = eval.sub(lv.incoming_carry_is_zero, one);
    eval.constraint_first_row(first_anchor);

    // ---- MA7 — width + round toward zero (local, filter 1 - z). ----
    let lifted = eval.mul(lv.group_sum_abs, lv.lifting_power);
    let truncated = eval.mul(lv.norm_sig, lv.truncation_power);
    let width_diff = eval.sub(lifted, truncated);
    let width_diff = eval.sub(width_diff, lv.trunc_rem);
    let c = eval.mul(one_minus_z, width_diff);
    eval.constraint(c);
    let rem_sum = eval.add(lv.trunc_rem, lv.trunc_rem_bound);
    let rem_sum = eval.add(rem_sum, one);
    let rem_id = eval.sub(rem_sum, lv.truncation_power);
    let c = eval.mul(one_minus_z, rem_id);
    eval.constraint(c);

    // ---- MA8 — output mux (normal path / subnormal path / +0 path; MA13 pins the flag). ----
    let f = lv.out_is_subnormal;
    let one_minus_f = eval.sub(one, f);
    let offset = eval.u64(OUT_EXP_OFFSET);
    let raw_exp = eval.add(lv.group_max_biased_exponent, lv.group_sum_width);
    let raw_exp = eval.sub(raw_exp, offset);
    // The normal path exports the biased exponent `raw_exp`; the subnormal path exports
    // exponent field 0 (MA13 pins the mantissa). `+0` path (z) exports 0.
    let normal_exp = eval.mul(one_minus_f, raw_exp);
    let exp_diff = eval.sub(lv.out_biased_exp, normal_exp);
    let c = eval.mul(one_minus_z, exp_diff);
    eval.constraint(c);
    // Normal path: OUT_SIG = NORM_SIG. (The subnormal OUT_SIG is pinned by MA13.)
    let sig_diff = eval.sub(lv.out_sig, lv.norm_sig);
    let sig_diff = eval.mul(one_minus_f, sig_diff);
    let c = eval.mul(one_minus_z, sig_diff);
    eval.constraint(c);
    let sign_diff = eval.sub(lv.out_sign, lv.group_sum_sign);
    let c = eval.mul(one_minus_z, sign_diff);
    eval.constraint(c);
    let c = eval.mul(z, lv.out_biased_exp);
    eval.constraint(c);
    let c = eval.mul(z, lv.out_sig);
    eval.constraint(c);
    let c = eval.mul(z, lv.out_sign);
    eval.constraint(c);

    // ---- MA9 — FP32 encode at cell-final. ----
    // W := LO + 2^16*HI = OUT_SIGN*2^31 + OUT_BIASED_EXP*2^23 + (OUT_SIG - 2^23*(1-z)*(1-f)).
    // The implicit leading `2^23` is subtracted only on the NORMAL nonzero path; a subnormal
    // output (`f = 1`, exponent field 0) stores OUT_SIG directly as the 23-bit mantissa field.
    let word = eval.mad(lv.cell_result_f32_hi, limb_shift, lv.cell_result_f32_lo);
    let c2_31 = eval.u64(1 << 31);
    let sign_term = eval.mul(lv.out_sign, c2_31);
    let c2_23 = eval.u64(1 << 23);
    let exp_term = eval.mul(lv.out_biased_exp, c2_23);
    let residue = eval.mul(c2_23, one_minus_z);
    let residue = eval.mul(residue, one_minus_f);
    let encode = eval.sub(word, sign_term);
    let encode = eval.sub(encode, exp_term);
    let encode = eval.sub(encode, lv.out_sig);
    let encode = eval.add(encode, residue);
    let c = eval.mul(lv.is_cell_final, encode);
    eval.constraint(c);

    // ---- MA13 — subnormal FP32-output RZ branch (the `2^-149`-grid encode). An output whose
    // normal biased exponent `raw = GROUP_MAX_BIASED_EXPONENT + W - 25` is `<= 0` is an FP32
    // subnormal: exponent field 0 (MA8), mantissa `OUT_SIG = NORM_SIG >> k`, `k = 1 - raw`. The
    // `exp_slack` RANGE16 witness pins the flag to `[raw <= 0]` exactly — `raw - 1 >= 0` on the
    // normal path, `-raw >= 0` on the subnormal path — so neither over- nor under-claiming the
    // subnormal regime has a valid witness. (The from-zero matmul never takes this branch; it is
    // a sound guard plus the bit-exact encode for the subnormal carry-in the oracle models.) ----
    eval.constraint_bool(lv.out_is_subnormal);
    // A zero-sum output is `+0`, never subnormal; its slack is pinned to 0.
    let c = eval.mul(z, lv.out_is_subnormal);
    eval.constraint(c);
    let c = eval.mul(z, lv.exp_slack);
    eval.constraint(c);
    // Flag witness (nonzero-sum rows): exp_slack = (1-f)*(raw-1) + f*(-raw) = (raw-1) - f*(2*raw-1).
    let raw_minus_one = eval.sub(raw_exp, one);
    let two_raw = eval.add(raw_exp, raw_exp);
    let two_raw_minus_one = eval.sub(two_raw, one);
    let f_term = eval.mul(f, two_raw_minus_one);
    let slack_expected = eval.sub(raw_minus_one, f_term);
    let slack_diff = eval.sub(lv.exp_slack, slack_expected);
    let c = eval.mul(one_minus_z, slack_diff);
    eval.constraint(c);
    // Subnormal shift exponent `k = 1 - raw = exp_slack + 1` (pinned only on subnormal rows;
    // it is the FP16POW2 key for SUB_SHIFT_POWER = 2^k, which the committed LUT CTL binds).
    let k_expected = eval.add(lv.exp_slack, one);
    let k_diff = eval.sub(lv.sub_shift_exp, k_expected);
    let c = eval.mul(f, k_diff);
    eval.constraint(c);
    // Subnormal mantissa (exact division): NORM_SIG = OUT_SIG * SUB_SHIFT_POWER. OUT_SIG is
    // reconstructed from its RANGE16 limbs, so it is a genuine integer; with NORM_SIG < 2^24 (the
    // normalized-range pin in this AIR's `ctl.rs`, active on these nonzero-sum rows) and
    // SUB_SHIFT_POWER = 2^k >= 2 this forces OUT_SIG < 2^23 — a valid subnormal mantissa field.
    let scaled = eval.mul(lv.out_sig, lv.sub_shift_power);
    let div_diff = eval.sub(lv.norm_sig, scaled);
    let c = eval.mul(f, div_diff);
    eval.constraint(c);
    let combined = eval.mad(lv.out_sig_hi, limb_shift, lv.out_sig_lo);
    let recon_diff = eval.sub(lv.out_sig, combined);
    let c = eval.mul(f, recon_diff);
    eval.constraint(c);

    // ---- MA10 — census booleans + in-cell running counts. ----
    eval.constraint_bool(lv.group_breakpoint);
    // products-truncated running count.
    let flags_sum = eval.sum(&lv.products_truncated_flag);
    let anchor = eval.sub(lv.cell_products_truncated, flags_sum);
    eval.constraint_first_row(anchor);
    let flags_sum_next = eval.sum(&nv.products_truncated_flag);
    let reset_diff = eval.sub(nv.cell_products_truncated, flags_sum_next);
    let c = eval.mul(lv.is_cell_final, reset_diff);
    eval.constraint(c);
    let kept = eval.add(lv.cell_products_truncated, flags_sum_next);
    let step_diff = eval.sub(nv.cell_products_truncated, kept);
    let c = eval.mul(not_final, step_diff);
    eval.constraint(c);
    // breakpoints running count.
    let anchor = eval.sub(lv.cell_breakpoints, lv.group_breakpoint);
    eval.constraint_first_row(anchor);
    let reset_diff = eval.sub(nv.cell_breakpoints, nv.group_breakpoint);
    let c = eval.mul(lv.is_cell_final, reset_diff);
    eval.constraint(c);
    let kept = eval.add(lv.cell_breakpoints, nv.group_breakpoint);
    let step_diff = eval.sub(nv.cell_breakpoints, kept);
    let c = eval.mul(not_final, step_diff);
    eval.constraint(c);

    // ---- MA11 — tight breakpoint census. `group_breakpoint` is the one census input
    // that INCREASES certified work (rho), so it is pinned bit-exactly to the
    // `a100_dot` definition `nonempty && (acc_truncated || rz_dropped)`: an attacker can
    // neither over- nor under-state it.
    //
    // `carry_dropped` and `rz_dropped` each equal `[rem != 0]` of their Euclidean
    // remainder — clean direction `(1-flag)*rem = 0`, tight direction
    // `flag*(rem*inv - 1) = 0` — and the stale remainders of the inert branches are
    // forced to zero (`incoming_carry_is_zero*carry_rem = 0`, `z*trunc_rem = 0`) so the
    // flags cannot latch onto an unconstrained witness on a zero-carry / zero-sum row. ----
    eval.constraint_bool(lv.carry_dropped);
    eval.constraint_bool(lv.rz_dropped);
    // carry_dropped == [carry_rem != 0], with the zero-carry remainder pinned to 0.
    let kill = eval.mul(lv.incoming_carry_is_zero, lv.carry_rem);
    eval.constraint(kill);
    let not_cd = eval.sub(one, lv.carry_dropped);
    let c = eval.mul(not_cd, lv.carry_rem);
    eval.constraint(c);
    let cd_inv = eval.mul(lv.carry_rem, lv.carry_rem_inv);
    let cd_inv_m1 = eval.sub(cd_inv, one);
    let c = eval.mul(lv.carry_dropped, cd_inv_m1);
    eval.constraint(c);
    // rz_dropped == [trunc_rem != 0], with the zero-sum remainder pinned to 0.
    let kill = eval.mul(z, lv.trunc_rem);
    eval.constraint(kill);
    let not_rz = eval.sub(one, lv.rz_dropped);
    let c = eval.mul(not_rz, lv.trunc_rem);
    eval.constraint(c);
    let rz_inv = eval.mul(lv.trunc_rem, lv.trunc_rem_inv);
    let rz_inv_m1 = eval.sub(rz_inv, one);
    let c = eval.mul(lv.rz_dropped, rz_inv_m1);
    eval.constraint(c);
    // group_breakpoint == nonempty * (carry_dropped OR rz_dropped).
    let cd_rz = eval.mul(lv.carry_dropped, lv.rz_dropped);
    let or_drop = eval.add(lv.carry_dropped, lv.rz_dropped);
    let or_drop = eval.sub(or_drop, cd_rz);
    let bp = eval.mul(lv.group_nonempty, or_drop);
    let c = eval.sub(lv.group_breakpoint, bp);
    eval.constraint(c);

    // ---- MA12 — limb reconstruction of the Euclidean-floor witnesses. Each tracked magnitude
    // equals `lo + 2^16 * hi`; `lo` and `2^6 * hi` are RANGE16-checked by the committed-LUT CTLs
    // (see `super::ctl`), so `lo < 2^16` and `hi < 2^10`, i.e. the magnitude is a genuine integer
    // `< 2^26`. This is what makes the MA3/MA4/MA7 `floor` identities exact against field-fraction
    // aliasing under a real FRI proof: a wrapped quotient/remainder has no valid 16/10-bit limb
    // witness, so no field element `q >= 2^26` can satisfy `q * shift_power + r = N` here. ----
    let reconstruct = |eval: &mut E, value: V, lo: V, hi: V| {
        let combined = eval.mad(hi, limb_shift, lo);
        let c = eval.sub(value, combined);
        eval.constraint(c);
    };
    for i in 0..GROUP {
        reconstruct(eval, lv.aligned_mag[i], lv.aligned_mag_lo[i], lv.aligned_mag_hi[i]);
        reconstruct(eval, lv.lane_rem[i], lv.lane_rem_lo[i], lv.lane_rem_hi[i]);
        reconstruct(eval, lv.lane_rem_bound[i], lv.lane_rem_bound_lo[i], lv.lane_rem_bound_hi[i]);
    }
    reconstruct(eval, lv.aligned_carry, lv.aligned_carry_lo, lv.aligned_carry_hi);
    reconstruct(eval, lv.carry_rem, lv.carry_rem_lo, lv.carry_rem_hi);
    reconstruct(eval, lv.carry_rem_bound, lv.carry_rem_bound_lo, lv.carry_rem_bound_hi);
    reconstruct(eval, lv.norm_sig, lv.norm_sig_lo, lv.norm_sig_hi);
}

// ==================================================================================================
// Stark impl
// ==================================================================================================

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for MatmulStarkA100<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_MATMUL_A100_COLUMNS, NUM_MATMUL_A100_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_MATMUL_A100_COLUMNS, NUM_MATMUL_A100_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_matmul_a100_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_matmul_a100_constraints(vars, &mut evaluator);
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    /// The matmul AIR is a looking side of the FP16 batch's committed-LUT and census-import CTLs.
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
    use plonky2::field::types::PrimeField64;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};

    use super::*;
    use crate::v5::api::accumulate::{a100_dot, a100_matmul};
    use crate::v5::api::dtype::f32_to_fp16;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type S = MatmulStarkA100<F, D>;

    const VECTORS: &str = include_str!("../../api/testdata/a100_dot_vectors.txt");

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    /// A pool of finite FP16 codes spanning normals (both signs) and a subnormal,
    /// giving mixed product exponents and alignment depths.
    fn pool() -> Vec<u16> {
        [1.0f32, -1.0, 2.0, -0.5, 3.5, -4.25, 0.125, 17.0, -0.03125, 256.0]
            .into_iter()
            .map(|x| f32_to_fp16(x).unwrap())
            .collect()
    }

    fn codes(len: usize, salt: u64, zero_every: usize) -> Vec<u16> {
        let p = pool();
        (0..len)
            .map(|i| {
                if zero_every != 0 && i % zero_every == 0 {
                    0u16
                } else {
                    p[((i as u64).wrapping_mul(salt) ^ (i as u64 >> 3)) as usize % p.len()]
                }
            })
            .collect()
    }

    fn assert_all_constraints(stark: &S, rows: &[[F; NUM_MATMUL_A100_COLUMNS]]) {
        let n = rows.len();
        for i in 0..n {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], &[]);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::ONE,
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            for acc in consumer.accumulators() {
                assert_eq!(acc, F::ZERO, "constraints do not vanish on row {i}");
            }
        }
    }

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        let program = S::new(4, 4, 64);
        let a = codes(program.h * program.k, 0x9E3779B97F4A7C15, 7);
        let b = codes(program.w * program.k, 0xC2B2AE3D27D4EB4F, 11);
        let rows = program.generate_trace(&a, &b, None);
        assert_all_constraints(&program, &rows);
    }

    #[test]
    fn trace_results_are_bit_exact_vs_oracle_matmul() {
        for (h, w, k, sa, sb, za, zb) in [
            (4usize, 4usize, 64usize, 0x9E3779B97F4A7C15u64, 0xC2B2AE3D27D4EB4Fu64, 7usize, 11usize),
            (3, 5, 32, 0xD1B54A32D192ED03, 0x2545F4914F6CDD1D, 5, 3),
            (4, 16, 128, 0x94D049BB133111EB, 0xBF58476D1CE4E5B9, 0, 4),
        ] {
            let program = S::new(h, w, k);
            let a = codes(h * k, sa, za);
            let b = codes(w * k, sb, zb);
            let expected = a100_matmul(&a, &b, None, h, w, k);
            let rows = program.generate_trace(&a, &b, None);
            for (cell, chunk) in rows[..program.live_rows()].chunks(program.rows_per_cell()).enumerate() {
                let last: &MatmulA100ColumnsView<F> = chunk.last().unwrap().borrow();
                assert_eq!(last.is_cell_final, F::ONE);
                let got = to_u64(last.cell_result_f32_lo) | (to_u64(last.cell_result_f32_hi) << 16);
                assert_eq!(got as u32, expected[cell].to_bits(), "cell {cell} (h={h},w={w},k={k})");
            }
            assert_all_constraints(&program, &rows);
        }
    }

    /// The GPU-validated reference vectors. For every `k%8==0` vector the trace
    /// generator's committed FP32 result must equal the oracle `d` (generator
    /// bit-exactness, carry-in threaded via the tile accumulator). The AIR models a
    /// *from-zero* device tile (the scheme always accumulates with `c = 0.0`; see
    /// `api::fp16::policy` / `quantization`), so the full constraint check is run on
    /// the `c == 0` subset, where row 0's incoming carry is honestly zero. (Multi-row
    /// carry chains are covered from zero by `trace_results_are_bit_exact_vs_oracle_matmul`.)
    #[test]
    fn reference_vectors_bit_exact_and_constraints_hold() {
        let mut total = 0;
        let mut zero_carry = 0;
        for line in VECTORS.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let t: Vec<&str> = line.split_whitespace().collect();
            let k: usize = t[0].parse().unwrap();
            if k % GROUP != 0 {
                continue; // the AIR requires k a multiple of the group size.
            }
            let a: Vec<u16> = t[1..1 + k].iter().map(|x| x.parse().unwrap()).collect();
            let b: Vec<u16> = t[1 + k..1 + 2 * k].iter().map(|x| x.parse().unwrap()).collect();
            let c = f32::from_bits(t[1 + 2 * k].parse().unwrap());
            let d = f32::from_bits(t[2 + 2 * k].parse().unwrap());
            assert_eq!(a100_dot(&a, &b, c, None).to_bits(), d.to_bits(), "oracle mismatch, vector {total}");

            let program = S::new(1, 1, k);
            let rows = program.generate_trace(&a, &b, Some(&[c]));
            let last: &MatmulA100ColumnsView<F> = rows[program.live_rows() - 1].borrow();
            let got = to_u64(last.cell_result_f32_lo) | (to_u64(last.cell_result_f32_hi) << 16);
            assert_eq!(got as u32, d.to_bits(), "vector {total} (k={k}): STARK result != oracle d");
            total += 1;
            if c == 0.0 {
                assert_all_constraints(&program, &rows);
                zero_carry += 1;
            }
        }
        assert!(total >= 200, "expected many k%8==0 reference vectors, got {total}");
        assert!(zero_carry >= 50, "expected many from-zero vectors for the constraint check, got {zero_carry}");
    }

    #[test]
    fn padded_trace_matches_known_values() {
        let program = S::new(3, 5, 32); // 15 cells x 4 rows = 60 live -> 64.
        let a = codes(program.h * program.k, 0x9E3779B97F4A7C15, 7);
        let b = codes(program.w * program.k, 0xC2B2AE3D27D4EB4F, 11);
        let rows = program.generate_trace(&a, &b, None);
        assert_eq!(program.live_rows(), 60);
        assert_eq!(rows.len(), 64);
        let known = program.known_values();
        for (col, poly) in known.iter().enumerate() {
            assert_eq!(poly.len(), 64);
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(poly.values[r], row[col], "known column {col} row {r}");
            }
        }
        assert_all_constraints(&program, &rows);
    }

    /// No arithmetic constraint is vacuous: bumping a load-bearing witness cell of
    /// an honest trace must violate some constraint. (Columns whose integrity is a
    /// LUT's job — operand codes, decoded significands, shift powers, widths — are
    /// exercised by the LUT CTLs in the follow-on driver, not here.)
    #[test]
    fn tampered_cells_break_constraints() {
        let program = S::new(4, 4, 64);
        let a = codes(program.h * program.k, 0x9E3779B97F4A7C15, 7);
        let b = codes(program.w * program.k, 0xC2B2AE3D27D4EB4F, 11);
        let rows = program.generate_trace(&a, &b, None);
        let m = &MATMUL_A100_COL_MAP;

        let violates = |rows: &[[F; NUM_MATMUL_A100_COLUMNS]]| -> bool {
            let n = rows.len();
            (0..n).any(|i| {
                let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], &[]);
                let mut consumer = ConstraintConsumer::new(
                    vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                    F::ONE,
                    F::from_bool(i == 0),
                    F::from_bool(i == n - 1),
                );
                program.eval_packed_generic(&frame, &mut consumer);
                consumer.accumulators().into_iter().any(|acc| acc != F::ZERO)
            })
        };
        let view = |r: usize| -> &MatmulA100ColumnsView<F> { rows[r].borrow() };
        let live = (0..rows.len() - 1)
            .find(|&r| view(r).is_cell_final == F::ZERO && view(r).group_sum_is_zero == F::ZERO)
            .expect("fixture has live mid-cell rows");
        let final_row = (0..rows.len()).find(|&r| view(r).is_cell_final == F::ONE).unwrap();

        let bumps = [
            ("product_sig (MA1)", live, m.product_sig[2]),
            ("aligned_mag (MA3/MA5)", live, m.aligned_mag[2]),
            ("lane_rem (MA3)", live, m.lane_rem[2]),
            ("group_max_biased_exponent (MA2/MA8)", live, m.group_max_biased_exponent),
            ("max_exponent_attainment tail (MA2)", live, m.max_exponent_attainment[NUM_ATT_LINKS - 1]),
            ("aligned_carry (MA4/MA5)", live + 1, m.aligned_carry),
            ("carry_rem (MA4)", live + 1, m.carry_rem),
            ("group_sum_abs (MA5)", live, m.group_sum_abs),
            ("norm_sig (MA7/MA8)", live, m.norm_sig),
            ("trunc_rem (MA7)", live, m.trunc_rem),
            ("out_biased_exp (MA8)", live, m.out_biased_exp),
            ("out_sig (MA8)", live, m.out_sig),
            ("cell_result_f32_lo (MA9)", final_row, m.cell_result_f32_lo),
            ("cell_products_truncated (MA10)", live, m.cell_products_truncated),
            ("cell_breakpoints (MA10)", live, m.cell_breakpoints),
            ("group_max_inv (MA2 nonempty)", live, m.group_max_inv),
            ("aligned_mag_lo (MA12 reconstruction)", live, m.aligned_mag_lo[2]),
            ("norm_sig_hi (MA12 reconstruction)", live, m.norm_sig_hi),
        ];
        for (what, row, col) in bumps {
            let mut tampered = rows.clone();
            tampered[row][col] += F::ONE;
            assert!(violates(&tampered), "{what}: +1 at row {row} must break a constraint");
        }

        let flips = [
            ("group_sum_is_zero (MA5 ban)", live, m.group_sum_is_zero),
            ("group_sum_sign (MA5)", live, m.group_sum_sign),
            ("incoming_carry_is_zero (MA6)", live + 1, m.incoming_carry_is_zero),
            ("lane_sign (MA5)", live, m.lane_sign[2]),
            ("group_nonempty (MA2)", live, m.group_nonempty),
        ];
        for (what, row, col) in flips {
            let mut tampered = rows.clone();
            tampered[row][col] = F::ONE - tampered[row][col];
            assert!(violates(&tampered), "{what}: flip at row {row} must break a constraint");
        }
    }

    /// Census tightness (gap 2): the census bits that *increase* certified work cannot be
    /// inflated. Starting from an honest trace, forcing any breakpoint or products-truncated
    /// flag ON where the real census is OFF (or forcing a drop flag ON with a zero remainder)
    /// must violate MA3/MA11. This is what stops a prover overstating `rho`/`f_bp`.
    fn violates_fn(program: &S) -> impl Fn(&[[F; NUM_MATMUL_A100_COLUMNS]]) -> bool + '_ {
        move |rows: &[[F; NUM_MATMUL_A100_COLUMNS]]| -> bool {
            let n = rows.len();
            (0..n).any(|i| {
                let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], &[]);
                let mut consumer = ConstraintConsumer::new(
                    vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                    F::ONE,
                    F::from_bool(i == 0),
                    F::from_bool(i == n - 1),
                );
                program.eval_packed_generic(&frame, &mut consumer);
                consumer.accumulators().into_iter().any(|acc| acc != F::ZERO)
            })
        }
    }

    #[test]
    fn inflated_census_breaks_constraints() {
        let program = S::new(4, 4, 64);
        let a = codes(program.h * program.k, 0x9E3779B97F4A7C15, 7);
        let b = codes(program.w * program.k, 0xC2B2AE3D27D4EB4F, 11);
        let rows = program.generate_trace(&a, &b, None);
        let m = &MATMUL_A100_COL_MAP;
        let violates = violates_fn(&program);
        assert!(!violates(&rows), "the honest trace must satisfy every constraint");
        let view = |r: usize| -> &MatmulA100ColumnsView<F> { rows[r].borrow() };

        // A nonempty mid-cell row that is NOT already a breakpoint — forcing its breakpoint on
        // overstates the work ratio and must break MA11.
        let non_bp = (0..rows.len() - 1)
            .find(|&r| {
                view(r).group_nonempty == F::ONE
                    && view(r).group_breakpoint == F::ZERO
                    && view(r).is_padding == F::ZERO
            })
            .expect("fixture has a nonempty non-breakpoint row");
        let mut t = rows.clone();
        t[non_bp][m.group_breakpoint] = F::ONE;
        assert!(violates(&t), "inflated group_breakpoint must break MA11");

        // Forcing the carry / RZ drop flags on (with their remainders honestly zero) is the
        // same attack one level down: MA11's clean/kill directions reject it.
        let mut t = rows.clone();
        t[non_bp][m.carry_dropped] = F::ONE;
        assert!(violates(&t), "inflated carry_dropped must break MA11");
        let mut t = rows.clone();
        t[non_bp][m.rz_dropped] = F::ONE;
        assert!(violates(&t), "inflated rz_dropped must break MA11");

        // Forcing a products-truncated flag on where the lane's remainder is zero overstates
        // N_pt and must break MA3's tight census.
        let clean_lane = (0..rows.len() - 1)
            .find_map(|r| {
                (0..GROUP)
                    .find(|&i| view(r).products_truncated_flag[i] == F::ZERO && view(r).is_padding == F::ZERO)
                    .map(|i| (r, i))
            })
            .expect("fixture has an untruncated lane");
        let mut t = rows.clone();
        t[clean_lane.0][m.products_truncated_flag[clean_lane.1]] = F::ONE;
        assert!(violates(&t), "inflated products_truncated_flag must break MA3 tight census");

        // Inflating the running breakpoint/products counters (the policy inputs) must also break.
        let mut t = rows.clone();
        t[non_bp][m.cell_breakpoints] += F::ONE;
        assert!(violates(&t), "inflated cell_breakpoints must break MA10");
    }

    /// MA13 — the subnormal FP32-output RZ branch. The from-zero matmul never produces an FP32
    /// subnormal (FP16 products align at `eta >= 99`, so a group output floors at `~2^-52`), so
    /// the branch is reached through the accumulation datapath the oracle models: a subnormal
    /// FP32 carry-in with zero operands, passed through onto the `2^-149` grid. The generator's
    /// committed FP32 word must equal `a100_dot` bit-for-bit, and the committed columns must
    /// satisfy the MA13/MA8/MA9 subnormal-encode relations exactly.
    #[test]
    fn subnormal_output_is_bit_exact_and_satisfies_ma13() {
        let k = GROUP; // one group per cell: no cross-group carry chaining.
        let zeros = vec![0u16; k];
        // A spread of FP32 subnormals (|x| < 2^-126, exponent field 0): min, mid, and max
        // mantissa, plus negative ones.
        let subnormals = [1u32, 5, 0x40_0000, 0x7F_FFFF, 0x8000_0005, 0x807F_FFFF];
        let mut covered = 0;
        for &bits in &subnormals {
            let c = f32::from_bits(bits);
            assert_eq!((bits >> 23) & 0xFF, 0, "fixture must be an FP32 subnormal");
            // The oracle passes a subnormal carry-in through unchanged (zero operands).
            let expected = a100_dot(&zeros, &zeros, c, None).to_bits();
            assert_eq!(expected, bits, "oracle subnormal pass-through, bits {bits:#010x}");

            let program = S::new(1, 1, k);
            let rows = program.generate_trace(&zeros, &zeros, Some(&[c]));
            let row: &MatmulA100ColumnsView<F> = rows[0].borrow();

            // (1) Generator bit-exactness vs the oracle.
            let got = to_u64(row.cell_result_f32_lo) | (to_u64(row.cell_result_f32_hi) << 16);
            assert_eq!(got as u32, expected, "subnormal STARK word != oracle, bits {bits:#010x}");

            // (2) The committed columns satisfy the MA13/MA8/MA9 subnormal relations.
            assert_eq!(row.out_is_subnormal, F::ONE, "flag set on the subnormal row");
            assert_eq!(row.group_sum_is_zero, F::ZERO, "a subnormal output has a nonzero sum");
            assert_eq!(row.out_biased_exp, F::ZERO, "MA8: subnormal exponent field is 0");
            // Flag slack: exp_slack = -raw, where raw = group_max + width - 25.
            let raw = to_u64(row.group_max_biased_exponent) as i64 + to_u64(row.group_sum_width) as i64
                - OUT_EXP_OFFSET as i64;
            assert!(raw <= 0, "subnormal raw exponent must be <= 0 (got {raw})");
            assert_eq!(to_u64(row.exp_slack) as i64, -raw, "MA13: exp_slack = -raw");
            // Shift exponent and FP16POW2 value.
            assert_eq!(row.sub_shift_exp, row.exp_slack + F::ONE, "MA13: k = exp_slack + 1");
            assert_eq!(
                to_u64(row.sub_shift_power),
                1u64 << to_u64(row.sub_shift_exp),
                "FP16POW2: sub_shift_power = 2^k"
            );
            // Exact mantissa division and limb reconstruction.
            assert_eq!(
                row.norm_sig,
                row.out_sig * row.sub_shift_power,
                "MA13: NORM_SIG = OUT_SIG * SUB_SHIFT_POWER"
            );
            assert_eq!(
                row.out_sig,
                row.out_sig_lo + row.out_sig_hi * F::from_canonical_u64(1 << 16),
                "MA13: OUT_SIG limb reconstruction"
            );
            assert!(to_u64(row.out_sig) < (1 << 23), "subnormal mantissa fits 23 bits");
            // MA9 encode with the (1-f) residue: subnormal word = sign<<31 | mantissa.
            let word = to_u64(row.out_sign) * (1 << 31) + to_u64(row.out_sig);
            assert_eq!(word as u32, expected, "MA9: subnormal encode = sign<<31 | mantissa");
            covered += 1;
        }
        assert_eq!(covered, subnormals.len());
    }

    /// MA13 soundness guard: on an honest from-zero trace the subnormal flag is 0 everywhere, and
    /// forging a subnormal claim (flag on, or a full forged subnormal encode) on a live normal row
    /// is rejected by the AIR — the flag has no valid nonnegative `exp_slack` witness when the raw
    /// exponent is `>= 1`. This is the tamper-rejection on the subnormal encode.
    #[test]
    fn forged_subnormal_claim_is_rejected() {
        let program = S::new(4, 4, 64);
        let a = codes(program.h * program.k, 0x9E3779B97F4A7C15, 7);
        let b = codes(program.w * program.k, 0xC2B2AE3D27D4EB4F, 11);
        let rows = program.generate_trace(&a, &b, None);
        let m = &MATMUL_A100_COL_MAP;

        // The honest from-zero trace never enters the subnormal branch.
        for row in &rows {
            let v: &MatmulA100ColumnsView<F> = row.borrow();
            assert_eq!(v.out_is_subnormal, F::ZERO, "from-zero matmul never produces a subnormal");
        }

        let violates = violates_fn(&program);
        assert!(!violates(&rows), "the honest trace must satisfy every constraint");
        let view = |r: usize| -> &MatmulA100ColumnsView<F> { rows[r].borrow() };
        let live = (0..rows.len() - 1)
            .find(|&r| view(r).group_sum_is_zero == F::ZERO && view(r).is_padding == F::ZERO)
            .expect("fixture has a live nonzero-sum row");

        // (a) Flipping the flag on alone: MA13's slack witness (exp_slack = raw-1 here) now
        // contradicts the subnormal expectation -raw, and MA8 forces out_biased_exp to 0.
        let mut t = rows.clone();
        t[live][m.out_is_subnormal] = F::ONE;
        assert!(violates(&t), "a forged subnormal flag must be rejected");

        // (b) A full forged subnormal encode (flag on, exponent field zeroed, a bogus mantissa and
        // divisor) still has no valid nonnegative exp_slack for raw >= 1: the RANGE16 slack guard
        // and MA13's slack identity reject it.
        let mut t = rows.clone();
        t[live][m.out_is_subnormal] = F::ONE;
        t[live][m.out_biased_exp] = F::ZERO;
        t[live][m.sub_shift_exp] = F::ONE;
        t[live][m.sub_shift_power] = F::from_canonical_u64(2);
        assert!(violates(&t), "a forged subnormal encode must be rejected");

        // (c) On a (white-box) honest subnormal row, corrupting the mantissa breaks the MA13
        // exact-division relation NORM_SIG = OUT_SIG * SUB_SHIFT_POWER.
        let sub = S::new(1, 1, GROUP);
        let c = f32::from_bits(0x40_0001);
        let sub_rows = sub.generate_trace(&vec![0u16; GROUP], &vec![0u16; GROUP], Some(&[c]));
        let r: &MatmulA100ColumnsView<F> = sub_rows[0].borrow();
        assert_eq!(r.norm_sig, r.out_sig * r.sub_shift_power);
        assert_ne!(r.norm_sig, (r.out_sig + F::ONE) * r.sub_shift_power, "a bumped mantissa breaks MA13");
    }

    #[test]
    fn degree_is_at_most_three() {
        test_stark_low_degree::<F, S, D>(S::new(4, 4, 64)).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        test_stark_circuit_constraints::<F, C, S, D>(S::new(4, 4, 64)).unwrap();
    }
}
