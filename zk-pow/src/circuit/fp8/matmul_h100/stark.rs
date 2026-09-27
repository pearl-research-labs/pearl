//! Proves the H100 32-lane fp8 E4M3 matrix-multiplication accumulation, bit for bit with
//! `fp8_sim.py::_group_sum`.
//!
//! # One hardware step per row
//!
//! A live row consumes 32 operand pairs because one H100 WGMMA group step has 32 lanes. An
//! E4M3 operand has a four-bit integer significand, so a product significand is at most
//! `15*15 = 225`. The simulator converts each product to the 14-bit carry frame as
//!
//! ```text
//! lane_term = +/-floor(product_significand * 2^7 / 2^shift)
//! shift     = group_max_exponent - product_exponent
//! ```
//!
//! The factor `2^7` is the hardware conversion from its 24-bit product representation to the
//! 14-bit accumulation representation: the product starts shifted by 17 and the carry path
//! drops 10 bits. Therefore an unshifted lane term is `< 2^15`.
//!
//! “FP22” here names the H100 simulator's internal product/carry frame, not an IEEE format.
//! Its exponents are shifted by +139 so lookup keys are nonnegative. Nonzero product exponents
//! span `[127, 155]`; zero products use sentinel 114. The row chooses the maximum exponent
//! across its 32 products and the incoming carry, aligns all 33 signed terms to it, and sums
//! them exactly as an integer. The sum is
//! `< 2^20`, then truncation toward zero normalizes it to a 14-bit significand in
//! `[2^13, 2^14)`. Fourteen is the H100 carry precision. A shift of 14 or more discards a
//! carry completely because its significand is `< 2^14`.
//!
//! The next local-carry exponent is
//! `group_max_exponent + bit_length(abs(sum)) - 14`. Four atoms form one
//! 128-product window. The local carry then resets to +0 and the window
//! result is added to a separate FP32 accumulator with exact IEEE RNE
//! promotion. A partial final window is promoted the same way. Cell-final
//! rows export that global accumulator; all zero results are canonical +0.
//!
//! # Padding
//!
//! Each cell is exactly its `k/32` live group steps; single-row phantom cells pad the full
//! trace to a power of two and read no operands, emit no output.
//!
//! Constraint groups M3-M16 prove the arithmetic transitions, the cell binade `E_CELL`
//! (jackpot check 3), and the per-lane skip census (jackpot check 4). A complete batch
//! proof also includes the range, power-of-two, PRODALIGN15, POW2G, and
//! WIDTHNORM lookup relations wired by the FP8 driver; evaluating this
//! table's AIR alone does not supply those external table facts.

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

use super::super::unpredictability::skip;
use super::columns::{
    GROUP_WIDTH, MATMUL_COL_MAP, MatmulColumnsView, NUM_ATT_LINKS, NUM_MATMUL_COLUMNS, NUM_MATMUL_PUBLIC_INPUTS, PromotionView,
};
use crate::circuit::utils::evaluator::Evaluator;
use crate::circuit::utils::native_evaluator::NativeEvaluator;
use crate::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// FP22 exponent of a nonzero fp8 product: `ea + eb + 125`; the 125 is `139 - 14`, converting
/// simulator exponent `ea + eb - 14` to the nonnegative +139 convention.
const PROD_FP22_OFFSET: u64 = 125;
/// Product-zero sentinel and minimum reachable nonzero carry exponent. Nonzero products start
/// at 127, so a sentinel lane cannot overstate the group maximum.
const FP22_ZERO_SENTINEL: u64 = 114;
/// FP22 (bias 139) to f32 (bias 127) exponent-field rebias, used by the M12 encode.
const F32_REBIAS: u64 = 12;
/// The hardware aligns carries at the 14-bit internal width: shifts are capped at 14 (POW2G's
/// min-cap), which floors a far carry to zero exactly like `(sig >> 10) >> shift` does.
const CARRY_SHIFT_CAP: u64 = 14;
/// Normalized significands live in `[2^13, 2^14)`.
const SIG14_MIN: u64 = 1 << 13;
const SIG24_MIN: u64 = 1 << 23;
const PROMOTION_NEAR_GAP: u64 = 25;
const LIMB_BASE: u64 = 1 << 16;
/// The shift on every jackpot check 3 binade (`E_CELL`, `LANE_BINADES`, B200's counterparts):
/// `E = floor(log2 |value|) + 139` — the FP22 shift, shared by both hardware variants so all
/// binades are nonnegative and comparable. 0 marks a zero value.
pub(crate) const BINADE_BIAS: u64 = 139;

// ==================================================================================================
// Program and trace generation
// ==================================================================================================

/// The H100 WGMMA AIR for one FP8 matmul: `out[r, c] = sum_t A[r, t] * B[t, c]`.
///
/// The structural columns (`cell_id`, `is_cell_final`, and both operand index bases) are
/// recomputable from this AIR's geometry; operand codes are witness data.
#[derive(Clone, Debug)]
pub struct MatmulStarkH100<F: RichField + Extendable<D>, const D: usize> {
    /// Output rows (rows of A).
    pub h: usize,
    /// Output columns (columns of B).
    pub w: usize,
    /// Inner dimension; must be a multiple of [`GROUP_WIDTH`].
    pub k: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> MatmulStarkH100<F, D> {
    pub fn new(h: usize, w: usize, k: usize) -> Self {
        Self {
            h,
            w,
            k,
            _phantom: PhantomData,
        }
    }

    /// The number of trace rows per output cell: its `k/32` live group-steps.
    pub fn rows_per_cell(&self) -> usize {
        self.k / GROUP_WIDTH
    }

    /// The rows covering real cells: `h*w` cells of [`Self::rows_per_cell`] rows, row-major.
    pub fn live_rows(&self) -> usize {
        self.h * self.w * self.rows_per_cell()
    }

    /// Trace height: the cell rows padded to the next power of two with single-row phantom
    /// cells (all-zero, `is_padding = is_cell_final = 1`, `cell_id = h*w + t`).
    pub fn num_rows(&self) -> usize {
        self.live_rows().next_power_of_two()
    }

    /// Generates the H100 Matmul trace and public inputs.
    ///
    /// - `a_codes`: fp8 E4M3 codes of A, row-major (`a_codes[r * k + t] = A[r, t]`).
    /// - `b_codes`: fp8 E4M3 codes of B, **column-major** (`b_codes[c * k + t] = B[t, c]`), i.e.
    ///   B-transposed row-major — the same operand layout as `Device::matmul_fp8`'s `b`, and
    ///   the layout the `operand_index_base_b + lane` CTL keys assume.
    /// - `a_lambdas`/`b_lambdas`: the elements' summand scores (jackpot check 4), same
    ///   layouts as the codes; the operand-code channel binds them to InputQuant's values.
    ///
    /// Panics if the geometry is unsupported.
    pub fn generate_trace(
        &self,
        a_codes: &[u8],
        b_codes: &[u8],
        a_lambdas: &[u64],
        b_lambdas: &[u64],
    ) -> (Vec<[F; NUM_MATMUL_COLUMNS]>, [F; NUM_MATMUL_PUBLIC_INPUTS]) {
        let (h, w, k) = (self.h, self.w, self.k);
        assert_eq!(a_codes.len(), h * k, "a_codes must be h*k row-major fp8 codes");
        assert_eq!(b_codes.len(), w * k, "b_codes must be w*k column-major fp8 codes");
        assert_eq!(a_lambdas.len(), h * k, "a_lambdas must align with a_codes");
        assert_eq!(b_lambdas.len(), w * k, "b_lambdas must align with b_codes");
        assert_eq!(k % GROUP_WIDTH, 0, "k must be a multiple of the group width");
        let live_rows_per_cell = k / GROUP_WIDTH;
        let num_rows = self.num_rows();

        let mut rows: Vec<[F; NUM_MATMUL_COLUMNS]> = Vec::with_capacity(num_rows);
        for r in 0..h {
            for c in 0..w {
                let cell_id = r * w + c;

                // ---- The k/32 live group-steps (group-sum emulation columns). ----
                let mut carry = OutTriple::ZERO;
                let mut incoming_carry_is_zero = true;
                let mut global = GlobalTriple::ZERO;
                // Consolidated policy anchors, filled in after the cell replay.
                let cell_start = rows.len();
                let mut e_cell = 0u64;
                let mut e_z_window = 0u64;
                let mut e_z_windows = Vec::with_capacity(live_rows_per_cell.div_ceil(4));
                for j in 0..live_rows_per_cell {
                    let is_final = j == live_rows_per_cell - 1;
                    let is_window_final = (j + 1).is_multiple_of(4) || is_final;
                    let mut row = MatmulColumnsView {
                        cell_id: F::from_canonical_usize(cell_id),
                        is_cell_final: F::from_bool(is_final),
                        is_window_final: F::from_bool(is_window_final),
                        operand_index_base_a: F::from_canonical_usize(r * k + j * GROUP_WIDTH),
                        operand_index_base_b: F::from_canonical_usize(h * k + c * k + j * GROUP_WIDTH),
                        incoming_carry_is_zero: F::from_bool(incoming_carry_is_zero),
                        ..Default::default()
                    };

                    // Decode the 32 lanes and receive their summand scores.
                    let mut lanes = [Fp8Product::ZERO; GROUP_WIDTH];
                    for i in 0..GROUP_WIDTH {
                        let operand_codes_a = a_codes[r * k + j * GROUP_WIDTH + i];
                        let operand_codes_b = b_codes[c * k + j * GROUP_WIDTH + i];
                        lanes[i] = Fp8Product::new(operand_codes_a, operand_codes_b);
                        row.operand_codes_a[i] = F::from_canonical_u8(operand_codes_a);
                        row.operand_codes_b[i] = F::from_canonical_u8(operand_codes_b);
                        row.prod_fp22_exp[i] = F::from_canonical_u64(lanes[i].fp22_exp);
                        row.lambda_a[i] = F::from_canonical_u64(a_lambdas[r * k + j * GROUP_WIDTH + i]);
                        row.lambda_b[i] = F::from_canonical_u64(b_lambdas[c * k + j * GROUP_WIDTH + i]);
                        // M13: the lane's binade, as served by PRODALIGN15.
                        let lane_binade = lanes[i].biased_binade();
                        row.lane_binades[i] = F::from_canonical_u64(lane_binade);
                        e_cell = e_cell.max(lane_binade);
                        if lane_binade != 0 {
                            // Add 5 to convert e(product)+139 to Hopper's grid encoding e(product)-13+157.
                            e_z_window = e_z_window.max(lane_binade + 5);
                        }
                    }

                    // The group's max FP22 exponent over the nonzero terms (zero terms never set
                    // the max, mirroring the simulator's -139 sentinel); the free witness on
                    // all-zero rows is pinned to the products' sentinel — the honest max of a
                    // row of sentinels — so every PRODALIGN15 key stays in-domain (shift 0) and
                    // the attainment chain vanishes through the sentinel factors.
                    let mut group_max_product_exponent: Option<u64> = (!incoming_carry_is_zero).then_some(carry.exp);
                    for lane in &lanes {
                        if !lane.is_zero {
                            group_max_product_exponent =
                                Some(group_max_product_exponent.map_or(lane.fp22_exp, |m| m.max(lane.fp22_exp)));
                        }
                    }
                    let group_max_product_exponent = group_max_product_exponent.unwrap_or(FP22_ZERO_SENTINEL);
                    row.group_max_product_exponent = F::from_canonical_u64(group_max_product_exponent);
                    // The M3 attainment chain: running products of the affine lane factors
                    // F_i = GROUP_MAX_PRODUCT_EXPONENT - PROD_FP22_EXP_i (three into the first link, two per link
                    // after, the last one alone) — the definitional fill of the link equalities.
                    let factor = |i: usize| F::from_canonical_u64(group_max_product_exponent) - row.prod_fp22_exp[i];
                    row.max_exponent_attainment[0] = factor(0) * factor(1) * factor(2);
                    for j in 1..NUM_ATT_LINKS - 1 {
                        row.max_exponent_attainment[j] =
                            row.max_exponent_attainment[j - 1] * factor(2 * j + 1) * factor(2 * j + 2);
                    }
                    row.max_exponent_attainment[NUM_ATT_LINKS - 1] =
                        row.max_exponent_attainment[NUM_ATT_LINKS - 2] * factor(GROUP_WIDTH - 1);

                    // Align the 33 terms to the max exponent and sum exactly, as `_group_sum`:
                    // products at the 15-bit frame `sig_a*sig_b*2^7 = (sig << 17) >> 10`, the
                    // carry at its 14-bit significand, both floor-shifted towards zero.
                    let mut total: i64 = 0;
                    for i in 0..GROUP_WIDTH {
                        if lanes[i].is_zero {
                            continue; // ALIGNED_LANE_TERMS stays 0 (the table's zero-pair rows hold 0).
                        }
                        let shift = group_max_product_exponent - lanes[i].fp22_exp; // >= 0: max dominates.
                        let mag = ((lanes[i].sig << 7) >> shift.min(63)) as i64;
                        row.aligned_lane_terms[i] = if lanes[i].sign {
                            -F::from_canonical_u64(mag as u64)
                        } else {
                            F::from_canonical_u64(mag as u64)
                        };
                        total += if lanes[i].sign { -mag } else { mag };
                    }
                    if incoming_carry_is_zero {
                        // Canonical witness on the inactive carry path.
                        row.incoming_carry_shift_power = F::ONE;
                    } else {
                        let shift = (group_max_product_exponent - carry.exp).min(CARRY_SHIFT_CAP); // POW2G's min-cap.
                        let shift_pow2 = 1u64 << shift;
                        let aligned = carry.sig14 >> shift;
                        let remainder = carry.sig14 - aligned * shift_pow2;
                        row.incoming_carry_shift_power = F::from_canonical_u64(shift_pow2);
                        row.aligned_incoming_carry = F::from_canonical_u64(aligned);
                        row.incoming_carry_remainder = F::from_canonical_u64(remainder);
                        total += if carry.sign { -(aligned as i64) } else { aligned as i64 };
                    }

                    // Sum, renormalize (WIDTHNORM semantics), and emit through the output mux.
                    row.group_sum_sign = F::from_bool(total < 0);
                    row.group_sum_abs = F::from_canonical_u64(total.unsigned_abs());
                    row.group_sum_is_zero = F::from_bool(total == 0);
                    debug_assert!(total.unsigned_abs() < 1 << 20, "frame sums fit 20 bits");
                    let out = if total == 0 {
                        OutTriple::ZERO // The +0 path (Z = GROUP_SUM_IS_ZERO; all-zero rows sum to 0).
                    } else {
                        let mag = total.unsigned_abs();
                        let width = bit_length(mag);
                        let sig14 = if width >= 14 {
                            mag >> (width - 14)
                        } else {
                            mag << (14 - width)
                        };
                        row.group_sum_width = F::from_canonical_u64(width);
                        let exp = group_max_product_exponent + width - 14;
                        debug_assert!(exp >= FP22_ZERO_SENTINEL, "nonzero outputs never drop below the sentinel");
                        OutTriple {
                            sign: total < 0,
                            exp,
                            sig14,
                        }
                    };
                    row.group_output_exponent = F::from_canonical_u64(out.exp);
                    row.group_output_significand = F::from_canonical_u64(out.sig14);

                    let (promotion, projected) = promotion_witness::<F>(global, out);
                    let exact_binade = projected.exp - promotion.binade_correction.to_canonical_u64();
                    row.promotion = promotion;
                    let global_out = if is_window_final { projected } else { global };
                    global = global_out;

                    if is_final {
                        let bits = u64::from(global.to_f32().to_bits());
                        row.cell_result_f32_lo = F::from_canonical_u64(bits & 0xFFFF);
                        row.cell_result_f32_hi = F::from_canonical_u64(bits >> 16);
                    }
                    // M covers each exact unrounded C+c, while Z covers each chained local c.
                    e_cell = e_cell.max(exact_binade);
                    if out.sig14 != 0 {
                        // Add 5 to convert e(c)+139 to Hopper's grid encoding e(c)-13+157.
                        e_z_window = e_z_window.max(out.exp + 5);
                    }
                    if is_window_final {
                        e_z_windows.push(e_z_window);
                        e_z_window = 0;
                        incoming_carry_is_zero = true;
                        carry = OutTriple::ZERO;
                    } else {
                        incoming_carry_is_zero = total == 0;
                        carry = out;
                    }
                    rows.push(row.into());
                }

                // ---- M13-M16 post-pass: fill cell/window anchors and the skip census. ----
                let map = MATMUL_COL_MAP;
                let mut cell_skips = 0u64;
                for (j, row) in rows[cell_start..].iter_mut().enumerate() {
                    let e_grid = e_cell.max(e_z_windows[j / 4]);
                    row[map.e_cell] = F::from_canonical_u64(e_cell);
                    row[map.e_grid] = F::from_canonical_u64(e_grid);
                    row[map.cell_nonzero] = F::from_bool(e_grid != 0);
                    for i in 0..GROUP_WIDTH {
                        let lambda_a = a_lambdas[r * k + j * GROUP_WIDTH + i];
                        let lambda_b = b_lambdas[c * k + j * GROUP_WIDTH + i];
                        let skipped = skip(lambda_a, lambda_b, e_grid);
                        row[map.skip_flag[i]] = F::from_bool(skipped);
                        cell_skips += u64::from(skipped);
                    }
                    row[map.cell_skips] = F::from_canonical_u64(cell_skips);
                }
            }
        }

        // ---- Trailing padding: one all-zero single-row phantom cell per padding row. ----
        for t in 0..num_rows - self.live_rows() {
            rows.push(phantom_row::<F>(h * w + t).into());
        }

        (rows, [])
    }

    /// The class (a) ("known") column values — the leading
    /// [`NUM_MATMUL_H100_KNOWN_COLUMNS`](super::columns::NUM_MATMUL_H100_KNOWN_COLUMNS) trace columns in
    /// their `columns.rs` order (`cell_id`, the cell/window boundary flags,
    /// `operand_index_base_a`, `operand_index_base_b`, `is_padding`), pure functions of the
    /// program geometry.
    /// Bit-exact with [`Self::generate_trace`]'s fill; the batch verifier recomputes exactly
    /// this and checks the trace openings against it (`starky`'s `BatchKnownColumns`).
    pub fn known_values(&self) -> Vec<PolynomialValues<F>> {
        let (h, w, k) = (self.h, self.w, self.k);
        let live_rows_per_cell = k / GROUP_WIDTH;
        let num_rows = self.num_rows();
        let mut cell_id = Vec::with_capacity(num_rows);
        let mut is_cell_final = Vec::with_capacity(num_rows);
        let mut is_window_final = Vec::with_capacity(num_rows);
        let mut operand_index_base_a = Vec::with_capacity(num_rows);
        let mut operand_index_base_b = Vec::with_capacity(num_rows);
        let mut is_padding = Vec::with_capacity(num_rows);
        for r in 0..h {
            for c in 0..w {
                for j in 0..live_rows_per_cell {
                    cell_id.push(F::from_canonical_usize(r * w + c));
                    is_cell_final.push(F::from_bool(j == live_rows_per_cell - 1));
                    is_window_final.push(F::from_bool((j + 1).is_multiple_of(4) || j == live_rows_per_cell - 1));
                    operand_index_base_a.push(F::from_canonical_usize(r * k + j * GROUP_WIDTH));
                    operand_index_base_b.push(F::from_canonical_usize(h * k + c * k + j * GROUP_WIDTH));
                    is_padding.push(F::ZERO);
                }
            }
        }
        for t in 0..num_rows - self.live_rows() {
            cell_id.push(F::from_canonical_usize(h * w + t));
            is_cell_final.push(F::ONE);
            is_window_final.push(F::ONE);
            operand_index_base_a.push(F::ZERO);
            operand_index_base_b.push(F::ZERO);
            is_padding.push(F::ONE);
        }
        [
            cell_id,
            is_cell_final,
            is_window_final,
            operand_index_base_a,
            operand_index_base_b,
            is_padding,
        ]
        .into_iter()
        .map(PolynomialValues::new)
        .collect()
    }
}

/// The all-zero single-row trailing phantom cell. 32 zero products entering a zero carry —
/// every group-sum constraint holds on the zero fill: the sentinel exponents make the M3
/// attainment factors vanish (`GROUP_MAX_PRODUCT_EXPONENT = 114`, the honest max of a row of
/// sentinels), the unfiltered M5 remainder identity pins `INCOMING_CARRY_SHIFT_POWER = 1`, and
/// the zero sum takes the +0 path.
fn phantom_row<F: RichField>(cell_id: usize) -> MatmulColumnsView<F> {
    // Check 3 on the zero fill: all lane binades and GROUP_OUTPUT_EXPONENT are 0, so
    // E_CELL = 0 passes every range check.
    MatmulColumnsView {
        cell_id: F::from_canonical_usize(cell_id),
        is_cell_final: F::ONE,
        is_window_final: F::ONE,
        is_padding: F::ONE,
        incoming_carry_is_zero: F::ONE,
        incoming_carry_shift_power: F::ONE,
        group_max_product_exponent: F::from_canonical_u64(FP22_ZERO_SENTINEL),
        prod_fp22_exp: [F::from_canonical_u64(FP22_ZERO_SENTINEL); GROUP_WIDTH],
        group_sum_is_zero: F::ONE,
        promotion: promotion_witness(GlobalTriple::ZERO, OutTriple::ZERO).0,
        ..Default::default()
    }
}

/// One decoded fp8 x fp8 product (`fp8_sim._split_e4m3` + the product formation of
/// `matmul_fp8_sim`); also the PRODALIGN15 row function (`super::super::luts`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Fp8Product {
    /// Exact significand `sig_a * sig_b` in `[0, 225]` (4-bit x 4-bit, subnormals unnormalized).
    pub(crate) sig: u64,
    /// XOR of the operand signs.
    pub(crate) sign: bool,
    /// FP22 exponent (`ea + eb + 125`), or [`FP22_ZERO_SENTINEL`] for zero products.
    pub(crate) fp22_exp: u64,
    pub(crate) is_zero: bool,
}

impl Fp8Product {
    const ZERO: Self = Self {
        sig: 0,
        sign: false,
        fp22_exp: FP22_ZERO_SENTINEL,
        is_zero: true,
    };

    pub(crate) fn new(operand_codes_a: u8, operand_codes_b: u8) -> Self {
        let (sign_a, exp_a, sig_a) = split_e4m3(operand_codes_a);
        let (sign_b, exp_b, sig_b) = split_e4m3(operand_codes_b);
        let sig = sig_a * sig_b;
        if sig == 0 {
            return Self::ZERO;
        }
        Self {
            sig,
            sign: sign_a != sign_b,
            fp22_exp: exp_a + exp_b + PROD_FP22_OFFSET,
            is_zero: false,
        }
    }

    /// The product's check-3 binade `floor(log2 |product|) + 139`, or 0 for a zero product
    /// (the PRODALIGN15 `BINADE` column). `|product| = SIG * 2^(FP22_EXP - 145)` (`-125`
    /// unbiases FP22, `-20` is the two e4m3 significand scales), so the binade is
    /// `bit_length(SIG) - 1 + FP22_EXP - 145 + 139`. Nonzero values span `[121, 156]`.
    pub(crate) fn biased_binade(&self) -> u64 {
        if self.is_zero {
            0
        } else {
            bit_length(self.sig) + self.fp22_exp + BINADE_BIAS - 146
        }
    }
}

/// Raw e4m3 byte -> (sign, exponent bits with subnormals clamped to 1, significand with the
/// implicit bit for normals) — `fp8_sim._split_e4m3` exactly. NaN codes (0x7F/0xFF) are not
/// special-cased here: InputQuant's quantization never emits them (QCAST saturates), and the
/// PRODALIGN15 table rows for NaN operands will carry whatever the differential-tested generator
/// says; this mirror only feeds trace generation.
fn split_e4m3(code: u8) -> (bool, u64, u64) {
    let sign = code >> 7 == 1;
    let exp = ((code >> 3) & 0xF) as u64;
    let man = (code & 0x7) as u64;
    let sig = if exp != 0 { man | 0x8 } else { man };
    (sign, exp.max(1), sig)
}

/// A group-sum output (equivalently, the carry entering the next row): the Gfloat triple in the
/// FP22 frame. `OutTriple::ZERO` is the +0 constant of the zero path.
#[derive(Clone, Copy, Debug)]
struct OutTriple {
    sign: bool,
    exp: u64,
    sig14: u64,
}

impl OutTriple {
    const ZERO: Self = Self {
        sign: false,
        exp: 0,
        sig14: 0,
    };

    /// The exact f32 bit pattern of this output (`Gfloat::operator float`; the M12 encode):
    /// rebias FP22 -> f32 (`-12`), drop the implicit bit, widen the 13 explicit significand
    /// bits by `2^10`. Zero encodes to +0.
    fn to_f32_bits(self) -> u64 {
        if self.sig14 == 0 {
            return 0;
        }
        ((self.sign as u64) << 31) | ((self.exp - F32_REBIAS) << 23) | ((self.sig14 - SIG14_MIN) << 10)
    }
}

#[derive(Clone, Copy, Debug)]
struct GlobalTriple {
    sign: bool,
    exp: u64,
    sig24: u64,
}

impl GlobalTriple {
    const ZERO: Self = Self {
        sign: false,
        exp: 0,
        sig24: 0,
    };

    fn to_f32(self) -> f32 {
        if self.sig24 == 0 {
            return 0.0;
        }
        f32::from_bits(((self.sign as u32) << 31) | (((self.exp - F32_REBIAS) as u32) << 23) | ((self.sig24 - SIG24_MIN) as u32))
    }
}

fn limbs<const N: usize>(mut value: u64) -> [u64; N] {
    core::array::from_fn(|_| {
        let limb = value & (LIMB_BASE - 1);
        value >>= 16;
        limb
    })
}

fn promotion_witness<F: RichField>(global_in: GlobalTriple, local: OutTriple) -> (PromotionView<F>, GlobalTriple) {
    let mut view = PromotionView {
        global_in_is_zero: F::from_bool(global_in.sig24 == 0),
        global_in_sign: F::from_bool(global_in.sign),
        global_in_exponent: F::from_canonical_u64(global_in.exp),
        global_in_significand: F::from_canonical_u64(global_in.sig24),
        ..Default::default()
    };
    let local_global = GlobalTriple {
        sign: local.sign,
        exp: local.exp,
        sig24: local.sig14 << 10,
    };
    let out = if global_in.sig24 == 0 {
        local_global
    } else if local_global.sig24 == 0 {
        global_in
    } else {
        let in_ge = global_in.exp >= local_global.exp;
        let (large, small) = if in_ge {
            (global_in, local_global)
        } else {
            (local_global, global_in)
        };
        let gap = large.exp - small.exp;
        view.exponent_order = F::from_bool(in_ge);
        view.exponent_gap = F::from_canonical_u64(gap);
        if gap > PROMOTION_NEAR_GAP {
            view.far_active = F::ONE;
            view.gap_power = F::ONE;
            view.binade_correction = F::from_bool(large.sign != small.sign && large.sig24 == SIG24_MIN);
            large
        } else {
            let gap_power = 1u64 << gap;
            view.near_active = F::ONE;
            view.gap_power = F::from_canonical_u64(gap_power);
            let global_scale = if in_ge { gap_power } else { 1 };
            let local_scale = if in_ge { 1 } else { gap_power };
            let scaled_global = global_in.sig24 * global_scale;
            let scaled_local = local_global.sig24 * local_scale;
            view.global_scale = F::from_canonical_u64(global_scale);
            view.scaled_global_significand = F::from_canonical_u64(scaled_global);
            view.scaled_local_significand = F::from_canonical_u64(scaled_local);
            let signed = |sign, magnitude: u64| if sign { -i128::from(magnitude) } else { i128::from(magnitude) };
            let exact = signed(global_in.sign, scaled_global) + signed(local_global.sign, scaled_local);
            let magnitude = exact.unsigned_abs() as u64;
            view.exact_abs = F::from_canonical_u64(magnitude);
            if magnitude == 0 {
                GlobalTriple::ZERO
            } else {
                let width = bit_length(magnitude);
                view.exact_width = F::from_canonical_u64(width);
                let low = width <= 24;
                view.near_low = F::from_bool(low);
                view.near_high = F::from_bool(!low);
                let power = 1u64 << width.abs_diff(24);
                view.shift_power = F::from_canonical_u64(power);
                let (quotient, round_up) = if low {
                    view.magnitude_or_remainder_limbs = limbs(magnitude).map(F::from_canonical_u64);
                    (magnitude * power, false)
                } else {
                    let quotient = magnitude / power;
                    let remainder = magnitude % power;
                    let half = power / 2;
                    let ge = remainder >= half;
                    let eq = remainder == half;
                    view.magnitude_or_remainder_limbs = limbs(remainder).map(F::from_canonical_u64);
                    view.remainder_bound_limbs = limbs(power - 1 - remainder).map(F::from_canonical_u64);
                    view.remainder_cmp_limbs =
                        limbs(if ge { remainder - half } else { half - 1 - remainder }).map(F::from_canonical_u64);
                    view.remainder_ge_half = F::from_bool(ge);
                    view.remainder_eq_half = F::from_bool(eq);
                    if !eq {
                        view.remainder_eq_inverse =
                            (F::from_canonical_u64(2 * remainder) - F::from_canonical_u64(power)).inverse();
                    }
                    view.quotient_lsb = F::from_canonical_u64(quotient & 1);
                    (quotient, remainder > half || (eq && quotient & 1 == 1))
                };
                view.quotient_limbs = limbs(quotient).map(F::from_canonical_u64);
                view.round_up = F::from_bool(round_up);
                let rounded = quotient + u64::from(round_up);
                let carry = rounded == 1 << 24;
                view.binade_correction = F::from_bool(carry);
                if !carry {
                    view.carry_slack_inverse = F::from_canonical_u64((1 << 24) - rounded).inverse();
                }
                GlobalTriple {
                    sign: exact < 0,
                    exp: small.exp + width - 24 + u64::from(carry),
                    sig24: rounded - u64::from(carry) * SIG24_MIN,
                }
            }
        }
    };
    view.global_out_is_zero = F::from_bool(out.sig24 == 0);
    view.global_out_sign = F::from_bool(out.sign);
    view.global_out_exponent = F::from_canonical_u64(out.exp);
    view.global_out_significand = F::from_canonical_u64(out.sig24);

    let native = global_in.to_f32() + f32::from_bits(local.to_f32_bits() as u32);
    let native = if native == 0.0 { 0.0 } else { native };
    debug_assert_eq!(out.to_f32().to_bits(), native.to_bits());
    (view, out)
}

/// `bit_length(x)` for `x > 0` (0 for 0), the simulator's `width`.
fn bit_length(x: u64) -> u64 {
    (64 - x.leading_zeros()) as u64
}

// ==================================================================================================
// Constraints, written once against the generic `Evaluator`
// ==================================================================================================

/// Promotion arithmetic on normalized inputs. The caller anchors/propagates the global
/// tuple; normalization, power and range lookups must enforce the bounds used below.
/// Every emitted polynomial has degree at most three.
fn eval_promotion_constraints<V, S, E>(p: &PromotionView<V>, local: [V; 4], eval: &mut E)
where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let [local_zero, local_sign, local_exp, local_sig14] = local;
    let one = eval.i32(1);
    let two = eval.i32(2);
    let c2_16 = eval.u64(1 << 16);
    let c2_23 = eval.u64(1 << 23);
    let c2_24 = eval.u64(1 << 24);
    let c2_10 = eval.u64(1 << 10);
    let local_sig = eval.mul(local_sig14, c2_10);
    for bit in [
        p.global_in_is_zero,
        p.global_in_sign,
        p.exponent_order,
        p.near_active,
        p.far_active,
        p.near_low,
        p.near_high,
        p.remainder_ge_half,
        p.remainder_eq_half,
        p.quotient_lsb,
        p.round_up,
        p.binade_correction,
        p.global_out_is_zero,
        p.global_out_sign,
    ] {
        eval.constraint_bool(bit);
    }
    // The global input is 0 when `global_in_is_zero`.
    for value in [p.global_in_sign, p.global_in_exponent, p.global_in_significand] {
        let c = eval.mul(p.global_in_is_zero, value);
        eval.constraint(c);
    }
    // The global output is 0 when `global_out_is_zero`.
    for value in [p.global_out_sign, p.global_out_exponent, p.global_out_significand] {
        let c = eval.mul(p.global_out_is_zero, value);
        eval.constraint(c);
    }

    // When both local and global input are nonzero, then one of near or far is active.
    let global_nonzero = eval.sub(one, p.global_in_is_zero);
    let local_nonzero = eval.sub(one, local_zero);
    let both_nonzero = eval.mul(global_nonzero, local_nonzero);
    let near_or_far = eval.add(p.near_active, p.far_active);
    eval.constraint_eq(near_or_far, both_nonzero);
    // When both local and global input are zero, the output is zero.
    let both_zero = eval.mul(p.global_in_is_zero, local_zero);
    eval.constraint_eq_if(both_zero, p.global_out_is_zero, one);
    // When local is zero, then the global output is the global input.
    let global_only = eval.mul(global_nonzero, local_zero);
    for (output, input) in [
        (p.global_out_is_zero, p.global_in_is_zero),
        (p.global_out_sign, p.global_in_sign),
        (p.global_out_exponent, p.global_in_exponent),
        (p.global_out_significand, p.global_in_significand),
    ] {
        eval.constraint_eq_if(global_only, output, input);
    }
    // When the global input is zero, then the global output is the local value.
    let local_only = eval.mul(p.global_in_is_zero, local_nonzero);
    let zero = eval.i32(0);
    for (output, input) in [
        (p.global_out_is_zero, zero),
        (p.global_out_sign, local_sign),
        (p.global_out_exponent, local_exp),
        (p.global_out_significand, local_sig),
    ] {
        eval.constraint_eq_if(local_only, output, input);
    }

    // When both inputs are nonzero, Pow2G checks the gap, its alignment power, and the far flag.
    // Only gaps 0..=89 are allowed: gaps 0..=25 require (gap_power, far_active) = (2^exponent_gap, 0),
    // while gaps 26..=89 require (1, 1).
    // The lookup tags its flag to prevent matching carry or normalization entries.
    let exponent_diff = eval.sub(p.global_in_exponent, local_exp);
    let twice_order = eval.mul(two, p.exponent_order);
    let order_sign = eval.sub(twice_order, one);
    let signed_gap = eval.mul(order_sign, exponent_diff);
    eval.constraint_eq_if(near_or_far, p.exponent_gap, signed_gap);
    // On near rows, align the significand between `global` and `local`.
    // The larger exponent is shifted left by `exponent_gap`.
    let power_minus_one = eval.sub(p.gap_power, one);
    let scale = eval.mad(p.exponent_order, power_minus_one, one);
    eval.constraint_eq_if(p.near_active, p.global_scale, scale);
    let scales_sum = eval.add(p.gap_power, one);
    let local_scale = eval.sub(scales_sum, p.global_scale);
    let scaled_global = eval.mul(p.global_in_significand, p.global_scale);
    let scaled_local = eval.mul(local_sig, local_scale);
    eval.constraint_eq_if(p.near_active, p.scaled_global_significand, scaled_global);
    eval.constraint_eq_if(p.near_active, p.scaled_local_significand, scaled_local);

    // The projected (global + local) sign is also the exact sign: this domain never underflows.
    let global_factor = double_complement(eval, one, p.global_in_sign);
    let local_factor = double_complement(eval, one, local_sign);
    let exact_factor = double_complement(eval, one, p.global_out_sign);
    let signed_global = eval.mul(global_factor, p.scaled_global_significand);
    let signed_local = eval.mul(local_factor, p.scaled_local_significand);
    let signed_sum = eval.add(signed_global, signed_local);
    let signed_exact = eval.mul(exact_factor, p.exact_abs);
    eval.constraint_eq_if(p.near_active, signed_exact, signed_sum);
    let zero_magnitude = eval.mul(p.global_out_is_zero, p.exact_abs);
    eval.constraint(zero_magnitude);
    // `near_nonzero` is only activated when the `near` flag is active and the global output is nonzero
    // This also enforces that only one of `near_low` and `near_high` is active at a time.
    let near_nonzero = eval.add(p.near_low, p.near_high);
    let output_nonzero = eval.sub(one, p.global_out_is_zero);
    let expected_nonzero = eval.mul(p.near_active, output_nonzero);
    eval.constraint_eq(near_nonzero, expected_nonzero);

    let quotient = eval.mad(p.quotient_limbs[1], c2_16, p.quotient_limbs[0]);
    let magnitude_or_remainder = eval.mad(p.magnitude_or_remainder_limbs[1], c2_16, p.magnitude_or_remainder_limbs[0]);
    let remainder_bound = eval.mad(p.remainder_bound_limbs[1], c2_16, p.remainder_bound_limbs[0]);
    let remainder_cmp = eval.mad(p.remainder_cmp_limbs[1], c2_16, p.remainder_cmp_limbs[0]);

    // Normalization.
    // Required lookups on near nonzero rows: `exact_width` in 1..=49, `shift_power` = 2^abs(`exact_width` - 24),
    // `near_low` iff `exact_width` <= 24, `quotient` in [2^23, 2^24), and 16-bit limbs.
    // Equations below: `near_low` enforces `quotient` = `exact_abs` * `shift_power`; `near_high` enforces
    // `exact_abs` = `quotient` * `shift_power` + `remainder` with 0 <= `remainder` < shift_power. These establish the claimed bit length.
    // Low products are < 2^55 and high sums < 2^49, so these equations cannot wrap around in the field.
    eval.constraint_eq_if(p.near_low, p.exact_abs, magnitude_or_remainder);
    let low_normalized = eval.mul(p.exact_abs, p.shift_power);
    eval.constraint_eq_if(p.near_low, quotient, low_normalized);
    let high_exact = eval.mad(quotient, p.shift_power, magnitude_or_remainder);
    eval.constraint_eq_if(p.near_high, p.exact_abs, high_exact);
    let remainder_sum = eval.add(magnitude_or_remainder, remainder_bound);
    let remainder_sum = eval.add(remainder_sum, one);
    eval.constraint_eq_if(p.near_high, remainder_sum, p.shift_power);

    // Compare twice the remainder with the shift power, avoiding a stored half.
    // if `remainder_ge_half == 1`, ensure 2 * `remainder` = 2^shift + 2 * `remainder_cmp`,
    // otherwise ensure: 2 * `remainder` + 2 * `remainder_cmp` + 2 = 2^shift,
    // where required limb range checks make `remainder_cmp` nonnegative.
    let twice_remainder = eval.mul(two, magnitude_or_remainder);
    let rem_half = eval.sub(twice_remainder, p.shift_power);
    let twice_ge = eval.mul(two, p.remainder_ge_half);
    let cmp_sign = eval.sub(twice_ge, one);
    let not_ge = eval.sub(one, p.remainder_ge_half);
    let cmp_rhs = eval.msub(cmp_sign, remainder_cmp, not_ge);
    let cmp_rhs = eval.mul(two, cmp_rhs);
    eval.constraint_eq_if(p.near_high, rem_half, cmp_rhs);
    // Ensure that `remainder_eq_half` = 1 iff rem = 2^{shift - 1} on near-high rows.
    let eq_zero = eval.mul(p.remainder_eq_half, rem_half);
    let c = eval.mul(p.near_high, eq_zero);
    eval.constraint(c);
    let eq_product = eval.mul(rem_half, p.remainder_eq_inverse);
    let not_eq = eval.sub(one, p.remainder_eq_half);
    eval.constraint_eq_if(p.near_high, eq_product, not_eq);
    // Required on near-high rows: RC16((quotient_limbs[0] - quotient_lsb) / 2) binds quotient parity.
    let tie_increment = eval.mul(p.remainder_eq_half, p.quotient_lsb);
    let increment_if_ge = eval.add(not_eq, tie_increment);
    let round_up = eval.mul(p.remainder_ge_half, increment_if_ge);
    eval.constraint_eq(p.round_up, round_up);
    let not_high = eval.sub(one, p.near_high);
    let inactive_round = eval.mul(not_high, p.round_up);
    eval.constraint(inactive_round);

    // Only near-high rounding or a far subtraction can lower the exact binade.
    let correction_active = eval.add(p.near_high, p.far_active);
    let correction_inactive = eval.sub(one, correction_active);
    let c = eval.mul(correction_inactive, p.binade_correction);
    eval.constraint(c);
    // On near-high rows, the correction is 1 exactly when rounding reaches 2^24.
    let rounded = eval.add(quotient, p.round_up);
    let carry_slack = eval.sub(c2_24, rounded);
    let carry_zero = eval.mul(p.binade_correction, carry_slack);
    eval.constraint_eq_if(p.near_high, carry_zero, zero);
    let carry_or_nonzero = eval.mad(carry_slack, p.carry_slack_inverse, p.binade_correction);
    eval.constraint_eq_if(p.near_high, carry_or_nonzero, one);

    let min_exp = eval.mux(p.exponent_order, p.global_in_exponent, local_exp);
    let output_exp = eval.add(min_exp, p.exact_width);
    let twenty_four = eval.i32(24);
    let output_exp = eval.sub(output_exp, twenty_four);
    let output_exp = eval.add(output_exp, p.binade_correction);
    eval.constraint_eq_if(near_nonzero, p.global_out_exponent, output_exp);
    let carry_sig = eval.mul(p.binade_correction, c2_23);
    let output_sig = eval.sub(rounded, carry_sig);
    eval.constraint_eq_if(near_nonzero, p.global_out_significand, output_sig);

    // Far inputs round to the operand with larger absolute value; cancellation is impossible.
    eval.constraint_eq_if(p.far_active, p.global_out_is_zero, zero);
    for (output, local, global) in [
        (p.global_out_sign, local_sign, p.global_in_sign),
        (p.global_out_exponent, local_exp, p.global_in_exponent),
        (p.global_out_significand, local_sig, p.global_in_significand),
    ] {
        let selected = eval.mux(p.exponent_order, local, global);
        eval.constraint_eq_if(p.far_active, output, selected);
    }
    // A far-binade drop requires opposite signs and a power-of-two larger magnitude.
    // Omitting a real drop only overstates M.
    let far_drop = eval.mul(p.far_active, p.binade_correction);
    let sign_sum = eval.add(p.global_in_sign, local_sign);
    eval.constraint_eq_if(far_drop, sign_sum, one);
    eval.constraint_eq_if(far_drop, p.global_out_significand, c2_23);
}

/// Evaluates every arithmetic constraint of MatmulStark. Lookup-borne facts (LUT oracle) are
/// *not* emitted here — the required relations are documented alongside the constraints.
/// The constraint set is program-independent and reads no public inputs.
pub(crate) fn eval_matmul_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_MATMUL_COLUMNS, NUM_MATMUL_PUBLIC_INPUTS>,
    eval: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_MATMUL_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &MatmulColumnsView<V> = lv.borrow();
    let nv: &[V; NUM_MATMUL_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let nv: &MatmulColumnsView<V> = nv.borrow();

    let one = eval.i32(1);
    let not_final = eval.sub(one, lv.is_cell_final);
    let not_window_final = eval.sub(one, lv.is_window_final);
    // Z := GROUP_SUM_IS_ZERO — "this row's output is zero" (degree 1; no separate all-zero
    // flag is needed). On an all-zero row, every ALIGNED_LANE_TERMS entry is table-bound to 0
    // and M5 kills ALIGNED_INCOMING_CARRY, so M6 forces GROUP_SUM_ABS = 0 and M7 forces Z = 1.
    let z = lv.group_sum_is_zero;
    let one_minus_z = eval.sub(one, z);

    // The transition constraints below are *plain* (they also bind the last-row -> first-row
    // wrap); the AIR leans on the wrap for the row-0 instances of the M6 sum and the M4 carry
    // anchor. Soundness of the wrap needs the last row to be cell-final — a class (a) fact of
    // IS_CELL_FINAL, carried by the batch verifier's known-column binding (see M4).

    // ---- M1 — product lookups x32 (PRODALIGN15; wired in ctl.rs). ----
    // The tuple binds ALIGNED_LANE_TERMS / PROD_FP22_EXP / OPERAND_CODES_A/B per lane. Its
    // missing negative-shift slots enforce GROUP_MAX_PRODUCT_EXPONENT >= PROD_FP22_EXP_i.
    // Padding rows read no operands, so this local pin forces the sentinel exponent — nonzero
    // products span [127, 155], so 114 selects exactly the table's zero-product rows, whose
    // tuple fixes ALIGNED_LANE_TERMS = 0. Without it, padding lanes could poison the group sum.
    let sentinel = eval.u64(FP22_ZERO_SENTINEL);
    for i in 0..GROUP_WIDTH {
        let off_sentinel = eval.sub(lv.prod_fp22_exp[i], sentinel);
        let c = eval.mul(lv.is_padding, off_sentinel);
        eval.constraint(c);
    }

    // ---- M3 — max attainment by vanishing product. ----
    // LUT key domains prove GROUP_MAX_PRODUCT_EXPONENT >= every term. The chain pins the
    // product of all lane differences, adds the zero-gated carry factor, and requires the
    // result to vanish. Thus some term attains the claimed maximum. A zero lane can attain only
    // at the sentinel 114, which is self-defeating if any nonzero term is present.
    let factor =
        |eval: &mut E, view: &MatmulColumnsView<V>, i: usize| eval.sub(view.group_max_product_exponent, view.prod_fp22_exp[i]);
    let f0 = factor(eval, lv, 0);
    let f1 = factor(eval, lv, 1);
    let f2 = factor(eval, lv, 2);
    let f01 = eval.mul(f0, f1);
    let link = eval.msub(f01, f2, lv.max_exponent_attainment[0]);
    eval.constraint(link);
    for j in 1..NUM_ATT_LINKS - 1 {
        let fa = factor(eval, lv, 2 * j + 1);
        let fb = factor(eval, lv, 2 * j + 2);
        let fab = eval.mul(fa, fb);
        let link = eval.msub(lv.max_exponent_attainment[j - 1], fab, lv.max_exponent_attainment[j]);
        eval.constraint(link);
    }
    let f_last = factor(eval, lv, GROUP_WIDTH - 1);
    let link = eval.msub(
        lv.max_exponent_attainment[NUM_ATT_LINKS - 2],
        f_last,
        lv.max_exponent_attainment[NUM_ATT_LINKS - 1],
    );
    eval.constraint(link);
    // Closing factor:
    // (1 - INCOMING_CARRY_IS_ZERO') *
    //     (GROUP_MAX_PRODUCT_EXPONENT' - GROUP_OUTPUT_EXPONENT)
    // + INCOMING_CARRY_IS_ZERO'.
    // A zero or cell-start carry contributes 1 and cannot attain a stale exponent.
    let carry_live = eval.sub(one, nv.incoming_carry_is_zero);
    let carry_diff = eval.sub(nv.group_max_product_exponent, lv.group_output_exponent);
    let carry_factor = eval.mul(carry_live, carry_diff);
    let carry_factor = eval.add(carry_factor, nv.incoming_carry_is_zero);
    let closing = eval.mul(nv.max_exponent_attainment[NUM_ATT_LINKS - 1], carry_factor);
    eval.constraint(closing);

    // ---- M4 — window-local carry-zero flag by propagation. ----
    // Inside a 128-product window the next flag is exactly Z. Every window
    // boundary resets the tensor-core accumulator to +0.
    let carry_next_diff = eval.sub(nv.incoming_carry_is_zero, z);
    let c = eval.mul(not_window_final, carry_next_diff);
    eval.constraint(c);
    let carry_next_reset = eval.sub(nv.incoming_carry_is_zero, one);
    let c = eval.mul(lv.is_window_final, carry_next_reset);
    eval.constraint(c);
    let first_anchor = eval.sub(lv.incoming_carry_is_zero, one);
    eval.constraint_first_row(first_anchor);
    // IS_CELL_FINAL is class (a) (verifier-recomputed, checked against the trace openings —
    // `MatmulStarkH100::known_values`), so its schedule facts hold without in-AIR pins: it is
    // boolean, and the last row is cell-final or a padding cell-final. This makes every plain
    // transition's cyclic wrap sound.
    eval.constraint_bool(lv.is_padding);

    // ---- M5 — carry alignment (anchored at the producing row). ----
    // POW2G(GROUP_MAX_PRODUCT_EXPONENT' - GROUP_OUTPUT_EXPONENT;
    //       INCOMING_CARRY_SHIFT_POWER') [filter 1 - INCOMING_CARRY_IS_ZERO']
    // proves the shift and its ordering. The identity below proves the previous output's exact
    // floor split.
    let carry_nonzero_next = eval.sub(one, nv.incoming_carry_is_zero);
    let shifted = eval.mul(nv.aligned_incoming_carry, nv.incoming_carry_shift_power);
    let floor_diff = eval.sub(lv.group_output_significand, shifted);
    let floor_diff = eval.sub(floor_diff, nv.incoming_carry_remainder);
    let c = eval.mul(carry_nonzero_next, floor_diff);
    eval.constraint(c);
    // Local kill: zero carries contribute nothing (also covers cell starts and the cyclic wrap).
    let kill = eval.mul(lv.incoming_carry_is_zero, lv.aligned_incoming_carry);
    eval.constraint(kill);

    // ---- M6 — signed sum (anchored at the producing row; row 0's instance rides the wrap,
    // where ALIGNED_INCOMING_CARRY(0) = 0 makes the stale GROUP_OUTPUT_SIGN factor inert). ----
    let terms_sum = eval.sum(&nv.aligned_lane_terms);
    let carry_sign_factor = double_complement(eval, one, lv.group_sum_sign); // 1 - 2*GROUP_OUTPUT_SIGN
    let signed_carry = eval.mul(carry_sign_factor, nv.aligned_incoming_carry);
    let bracket = eval.add(terms_sum, signed_carry);
    let group_sum_sign_factor = double_complement(eval, one, nv.group_sum_sign); // 1 - 2*GROUP_SUM_SIGN'
    let recovered = eval.mul(group_sum_sign_factor, bracket);
    let sum_eq = eval.sub(nv.group_sum_abs, recovered);
    eval.constraint(sum_eq);
    // GROUP_SUM_SIGN must be a bit for the magnitude recovery to pin
    // GROUP_SUM_ABS = |sum|. WIDTHNORM bounds GROUP_SUM_ABS < 2^20, making the sign unique.
    eval.constraint_bool(lv.group_sum_sign);
    // GROUP_SUM_IS_ZERO is explicit: the product constraint proves one direction and M7's
    // filtered significand floor proves the other.
    eval.constraint_bool(lv.group_sum_is_zero);
    let ban = eval.mul(lv.group_sum_is_zero, lv.group_sum_abs);
    eval.constraint(ban);
    // Cancelled sums are +0.
    let plus_zero = eval.mul(lv.group_sum_is_zero, lv.group_sum_sign);
    eval.constraint(plus_zero);

    // ---- M7 — width + renormalization: WIDTHNORM(GROUP_SUM_ABS; GROUP_SUM_WIDTH, GROUP_OUTPUT_SIGNIFICAND)
    // [filter 1 - GROUP_SUM_IS_ZERO] and the significand floor check are wired in ctl.rs.

    // ---- M10 — output mux (normal path / +0 path). ----
    let exp_raw = eval.add(lv.group_max_product_exponent, lv.group_sum_width);
    let fourteen = eval.i32(14);
    let exp_raw = eval.sub(exp_raw, fourteen); // EXP_RAW := GROUP_MAX_PRODUCT_EXPONENT + GROUP_SUM_WIDTH - 14 (affine).
    let exp_diff = eval.sub(lv.group_output_exponent, exp_raw);
    let c = eval.mul(one_minus_z, exp_diff);
    eval.constraint(c);
    let c = eval.mul(z, lv.group_output_exponent);
    eval.constraint(c);
    let c = eval.mul(z, lv.group_output_significand);
    eval.constraint(c);

    // ---- M11 — reset-and-promote FP32 accumulator. ----
    let p = &lv.promotion;
    // e = window-final, f = cell-final, with f <= e from the known schedule.
    // Directly select the next input: retain C, commit the projection, or reset to +0.
    let commit = eval.sub(lv.is_window_final, lv.is_cell_final);
    for (next, input, projected, reset) in [
        (nv.promotion.global_in_is_zero, p.global_in_is_zero, p.global_out_is_zero, one),
        (nv.promotion.global_in_sign, p.global_in_sign, p.global_out_sign, eval.i32(0)),
        (
            nv.promotion.global_in_exponent,
            p.global_in_exponent,
            p.global_out_exponent,
            eval.i32(0),
        ),
        (
            nv.promotion.global_in_significand,
            p.global_in_significand,
            p.global_out_significand,
            eval.i32(0),
        ),
    ] {
        let retained = eval.mul(not_window_final, input);
        let committed = eval.mad(commit, projected, retained);
        let expected = eval.mad(lv.is_cell_final, reset, committed);
        eval.constraint_eq(next, expected);
    }
    eval.constraint_first_row_eq(p.global_in_is_zero, one);
    eval.constraint_first_row(p.global_in_sign);
    eval.constraint_first_row(p.global_in_exponent);
    eval.constraint_first_row(p.global_in_significand);
    // Every atom projects `global_in + local_out`; projected exponent minus correction bounds the exact-sum binade.
    // The committed global state adopts that projection only at window ends.
    eval_promotion_constraints(
        p,
        [z, lv.group_sum_sign, lv.group_output_exponent, lv.group_output_significand],
        eval,
    );

    // ---- M12 — promoted-f32 encode at cell-final. ----
    // W := LO + 2^16*HI is the global FP32 accumulator after the
    // final reset-and-promote window.
    // The result limbs are RC16-bound.
    let c2_16 = eval.u64(1 << 16);
    let c2_23 = eval.u64(1 << 23);
    let w = eval.mad(lv.cell_result_f32_hi, c2_16, lv.cell_result_f32_lo);
    let c2_31 = eval.u64(1 << 31);
    let sign_term = eval.mul(p.global_out_sign, c2_31);
    let exp_term = eval.mul(p.global_out_exponent, c2_23);
    let sig_term = p.global_out_significand;
    let c13_2_23 = eval.u64(13 << 23);
    let global_nonzero = eval.sub(one, p.global_out_is_zero);
    let residue = eval.mul(c13_2_23, global_nonzero);
    let encode = eval.sub(w, sign_term);
    let encode = eval.sub(encode, exp_term);
    let encode = eval.sub(encode, sig_term);
    let encode = eval.add(encode, residue);
    let c = eval.mul(lv.is_cell_final, encode);
    eval.constraint(c);

    // ---- M13 — consolidated-policy anchors. E_CELL is constant over a cell; E_GRID is
    // constant over one local window. Range checks in ctl.rs make E_CELL bound every
    // product and exact C+c binade, and E_GRID bound both E_CELL and the local-window grid.
    let e_cell_step = eval.sub(nv.e_cell, lv.e_cell);
    let c = eval.mul(not_final, e_cell_step);
    eval.constraint(c);
    let not_window_final = eval.sub(one, lv.is_window_final);
    let e_grid_step = eval.sub(nv.e_grid, lv.e_grid);
    let c = eval.mul(not_window_final, e_grid_step);
    eval.constraint(c);

    // ---- M14 — the grid-nonzero flag. NZ is boolean and its complement pins E_GRID = 0.
    // Required range checks make a false zero claim on a nonzero M/Z component impossible. ----
    let nz = lv.cell_nonzero;
    eval.constraint_bool(nz);
    let not_nz = eval.sub(one, nz);
    let c = eval.mul(not_nz, lv.e_grid);
    eval.constraint(c);

    // ---- M15 — per-lane skip flags (jackpot check 4). Boolean; pinned to 0 on padding rows
    // and on zero cells (a zero cell never counts as skipped). On a nonzero cell, lane u is
    // skipped exactly when
    //   LAMBDA_A + LAMBDA_B < 128*E_GRID + 45952,
    // using the consolidated M/Z threshold and InputQuant's summand scores.
    // Claiming non-skip costs the lane's RC16 (ctl.rs) on the key
    //   LAMBDA_A + LAMBDA_B - 128*E_GRID - 45952,
    // filter `CELL_NONZERO * (1 - SKIP_FLAG)`: a true skip makes the key negative, wrapping
    // it far outside [0, 2^16) and forcing the flag to 1. Satisfiable traces keep honest
    // non-skip keys below 2^16 (LAMBDA <= 54207, and E_CELL >= 121 on nonzero cells).
    // Overstating the census is allowed and self-defeating. ----
    for i in 0..GROUP_WIDTH {
        eval.constraint_bool(lv.skip_flag[i]);
        let c = eval.mul(lv.is_padding, lv.skip_flag[i]);
        eval.constraint(c);
        let c = eval.mul(not_nz, lv.skip_flag[i]);
        eval.constraint(c);
    }

    // ---- M16 — the in-cell skip census. CELL_SKIPS anchors to the row's flag sum at each
    // cell start and otherwise accumulates the next row's flags; the cell-final value rides
    // the result-and-census channel to XorFold's budget gate. The cyclic wrap is inert: the last row
    // is cell-final (class (a)), so the wrap instance is the next cell's anchor — row 0's
    // explicit first-row anchor is redundant with it but keeps the recurrence local. ----
    let flags_sum = eval.sum(&lv.skip_flag);
    let anchor = eval.sub(lv.cell_skips, flags_sum);
    eval.constraint_first_row(anchor);
    let flags_sum_next = eval.sum(&nv.skip_flag);
    let reset_diff = eval.sub(nv.cell_skips, flags_sum_next);
    let c = eval.mul(lv.is_cell_final, reset_diff);
    eval.constraint(c);
    let kept = eval.add(lv.cell_skips, flags_sum_next);
    let step_diff = eval.sub(nv.cell_skips, kept);
    let c = eval.mul(not_final, step_diff);
    eval.constraint(c);
}

/// `1 - 2*bit`: maps a sign bit to the +/-1 factor recovering a magnitude from a signed value.
fn double_complement<V: Copy, S: Copy, E: Evaluator<V, S>>(eval: &mut E, one: V, bit: V) -> V {
    let twice = eval.add(bit, bit);
    eval.sub(one, twice)
}

// ==================================================================================================
// Stark impl
// ==================================================================================================

/// A CTL party of the fp8 batch (`requires_ctls()`): its proofs carry the cross-table
/// openings of the channels declared in `super::super::ctl`, so the batch driver is the
/// only supported proving path — there is no standalone uni-STARK proof object.
impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for MatmulStarkH100<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_MATMUL_COLUMNS, NUM_MATMUL_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget = StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_MATMUL_COLUMNS, NUM_MATMUL_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_matmul_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_matmul_constraints(vars, &mut evaluator);
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    // Party to the operand-codes and cell-results channels plus the committed LUT channels
    // (declared in `super::super::ctl`).
    fn requires_ctls(&self) -> bool {
        true
    }
}

// ==================================================================================================
// Tests
// ==================================================================================================

#[cfg(test)]
mod tests {
    use core::borrow::BorrowMut;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};

    use super::*;
    use crate::api::fp8::public_params::Device;
    use crate::api::fp8::utils::H100;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type S = MatmulStarkH100<F, D>;

    /// fp8 codes with exponent field 7 or 8 and mixed signs (never NaN). Products of pool codes
    /// have significands in [64, 121] (7 bits), so their f32 exponents land in {127, 128, 129}:
    /// with the zero run that is at most 4 runs per cell — fits every k >= 128 geometry.
    const CODE_POOL: [u8; 8] = [0x38, 0x40, 0xB9, 0x3A, 0xC1, 0x3B, 0xBA, 0x42];

    /// Deterministic operand codes: a zero every `zero_every` positions (0 = never), pool codes
    /// elsewhere.
    fn test_codes(len: usize, salt: u64, zero_every: usize) -> Vec<u8> {
        (0..len)
            .map(|i| {
                if zero_every != 0 && i % zero_every == 0 {
                    0x00
                } else {
                    CODE_POOL[((i as u64).wrapping_mul(salt) ^ (i as u64 >> 3)) as usize % CODE_POOL.len()]
                }
            })
            .collect()
    }

    /// Deterministic summand scores in the honest range (`lambda <= 54207`), spread so both
    /// skip verdicts occur.
    fn test_lambdas(len: usize, salt: u64) -> Vec<u64> {
        (0..len).map(|i| 12_928 + (i as u64).wrapping_mul(salt) % 41_280).collect()
    }

    fn test_program() -> S {
        S::new(4, 4, 160)
    }

    fn test_trace() -> (S, Vec<[F; NUM_MATMUL_COLUMNS]>, [F; NUM_MATMUL_PUBLIC_INPUTS]) {
        let program = test_program();
        let a = test_codes(program.h * program.k, 0x9E3779B97F4A7C15, 8);
        let b = test_codes(program.w * program.k, 0xC2B2AE3D27D4EB4F, 11);
        let (rows, pis) = program.generate_trace(
            &a,
            &b,
            &test_lambdas(program.h * program.k, 0xA24BAED4963EE407),
            &test_lambdas(program.w * program.k, 0x9FB21C651E98DF25),
        );
        (program, rows, pis)
    }

    // ---- An independent, line-by-line port of fp8_sim.py (the H100 WGMMA truth), used as the
    // ---- bit-exactness oracle for trace generation. Terms are (sign, exp, sig) Gfloat triples
    // ---- in the simulator's convention: value = sig * 2^(exp - 23).
    const REF_ZERO_EXP: i64 = -139;

    fn ref_split_e4m3(code: u8) -> (i64, i64, i64) {
        let sign = (code >> 7) as i64;
        let exp = ((code >> 3) & 0xF) as i64;
        let man = (code & 0x7) as i64;
        let sig = if exp != 0 { man | 0x8 } else { man };
        (sign, exp.max(1), sig)
    }

    fn ref_group_sum(terms: &[(i64, i64, i64)]) -> (i64, i64, i64) {
        let group_max_product_exponent = terms.iter().map(|t| t.1).max().unwrap();
        let mut total: i64 = 0;
        for &(sign, exp, sig) in terms {
            let shift = (group_max_product_exponent - exp).min(31);
            let aligned = (sig >> 10) >> shift;
            total += if sign != 0 { -aligned } else { aligned };
        }
        let sign = (total < 0) as i64;
        let mag = total.abs();
        if mag == 0 {
            return (0, REF_ZERO_EXP, 0); // +0 regardless of the sum's sign.
        }
        let width = 64 - (mag as u64).leading_zeros() as i64;
        let mut exp = group_max_product_exponent + width - 14;
        let mut sig = if width > 14 {
            mag >> (width - 14)
        } else {
            mag << (14 - width)
        };
        if exp < -126 {
            // fp32 denormal clamp — dead on the fp8 path but kept for fidelity.
            sig >>= (-126 - exp).min(31);
            exp = -126;
        }
        if sig == 0 {
            return (0, REF_ZERO_EXP, 0);
        }
        (sign, exp, sig << 10)
    }

    /// Hopper reset-and-promote replay: rank-32 atoms, a local reset every
    /// four atoms, and ascending-window FP32 additions.
    fn ref_matmul_fp8(a: &[u8], b: &[u8], h: usize, w: usize, k: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(h * w);
        for r in 0..h {
            for c in 0..w {
                let mut acc = (0i64, REF_ZERO_EXP, 0i64);
                let mut global = 0.0f32;
                for k0 in (0..k).step_by(GROUP_WIDTH) {
                    let mut terms = vec![acc];
                    for t in k0..k0 + GROUP_WIDTH {
                        let (sa, ea, ma) = ref_split_e4m3(a[r * k + t]);
                        let (sb, eb, mb) = ref_split_e4m3(b[c * k + t]);
                        let sig = ma * mb;
                        if sig == 0 {
                            terms.push((0, REF_ZERO_EXP, 0));
                        } else {
                            terms.push((sa ^ sb, ea + eb - 14, sig << 17));
                        }
                    }
                    acc = ref_group_sum(&terms);
                    if k0 + GROUP_WIDTH == k || (k0 + GROUP_WIDTH).is_multiple_of(4 * GROUP_WIDTH) {
                        let (sign, exp, sig) = acc;
                        let local = if sig == 0 {
                            0.0
                        } else {
                            let mag = sig as f64 * ((exp - 23) as f64).exp2();
                            (if sign != 0 { -mag } else { mag }) as f32
                        };
                        global += local;
                        if global == 0.0 {
                            global = 0.0;
                        }
                        acc = (0, REF_ZERO_EXP, 0);
                    }
                }
                out.push(global.to_bits());
            }
        }
        out
    }

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    fn promotion_bits(p: &PromotionView<F>) -> u32 {
        if p.global_out_is_zero == F::ONE {
            return 0;
        }
        ((to_u64(p.global_out_sign) as u32) << 31)
            | (((to_u64(p.global_out_exponent) - F32_REBIAS) as u32) << 23)
            | ((to_u64(p.global_out_significand) - SIG24_MIN) as u32)
    }

    fn assert_all_constraints(stark: S, rows: &[[F; NUM_MATMUL_COLUMNS]], pis: &[F]) {
        let n = rows.len();
        for i in 0..n {
            // Plain constraints must hold on every row *including* the last -> first wrap, which
            // the AIR uses for the row-0 instances of the sum and the register resets.
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], pis);
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
        let (program, rows, pis) = test_trace();
        assert_all_constraints(program, &rows, &pis);
    }

    fn promotion_rounding_trace(
        negative: bool,
        small_products: usize,
    ) -> (S, Vec<[F; NUM_MATMUL_COLUMNS]>, [F; NUM_MATMUL_PUBLIC_INPUTS]) {
        let program = S::new(1, 1, 160);
        let sign = if negative { 0x80 } else { 0 };
        let mut a = vec![0; program.k];
        let mut b = vec![0; program.k];
        // The first window produces +/-128. Each 0x01 * 0x01 in the second
        // contributes +/-2^-18, one quarter of an FP32 ulp at magnitude 128.
        a[..128].fill(0x38 | sign);
        b[..128].fill(0x38);
        a[128..128 + small_products].fill(0x01 | sign);
        b[128..128 + small_products].fill(0x01);
        let lambdas = test_lambdas(program.k, 1);
        let (rows, pis) = program.generate_trace(&a, &b, &lambdas, &lambdas);
        (program, rows, pis)
    }

    #[test]
    fn promotion_rounding_accepts_halfway_boundaries() {
        for negative in [false, true] {
            for (small_products, rounded_ulps) in [(1, 0), (2, 0), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2)] {
                let (program, rows, pis) = promotion_rounding_trace(negative, small_products);
                let row: &MatmulColumnsView<F> = rows[4].borrow();
                let expected_bits = (0x4300_0000 + rounded_ulps) | if negative { 1 << 31 } else { 0 };
                assert_eq!(row.promotion.near_high, F::ONE);
                assert_eq!(
                    row.promotion.remainder_eq_half,
                    F::from_bool(small_products == 2 || small_products == 6)
                );
                assert_eq!(promotion_bits(&row.promotion), expected_bits);
                let result_bits = to_u64(row.cell_result_f32_lo) | (to_u64(row.cell_result_f32_hi) << 16);
                assert_eq!(result_bits as u32, expected_bits);
                assert_all_constraints(program, &rows, &pis);
            }
        }
    }

    #[test]
    fn promotion_rejects_false_halfway_witness() {
        let (program, mut rows, pis) = promotion_rounding_trace(false, 3);
        assert_all_constraints(program.clone(), &rows, &pis);
        let row: &mut MatmulColumnsView<F> = rows[4].borrow_mut();
        let p = &mut row.promotion;
        assert_eq!(promotion_bits(p), 0x4300_0001);
        assert_eq!(p.near_high, F::ONE);
        assert_eq!(p.remainder_ge_half, F::ONE);
        assert_eq!(p.remainder_eq_half, F::ZERO);
        assert_eq!(p.quotient_lsb, F::ZERO);
        assert_eq!(p.binade_correction, F::ZERO);

        // Falsely claim a tie and suppress the required increment of the even
        // quotient. Repair the dependent witnesses so only the false tie rejects.
        p.remainder_eq_half = F::ONE;
        p.remainder_eq_inverse = F::ZERO;
        p.round_up = F::ZERO;
        let quotient = p.quotient_limbs[0] + F::from_canonical_u64(1 << 16) * p.quotient_limbs[1];
        p.global_out_significand = quotient;
        p.carry_slack_inverse = (F::from_canonical_u64(1 << 24) - quotient).inverse();
        row.cell_result_f32_lo = F::ZERO;
        row.cell_result_f32_hi = F::from_canonical_u64(0x4300);
        assert_eq!(promotion_bits(p), 0x4300_0000);

        let frame = StarkFrame::from_values(&rows[4], &rows[5], &pis);
        let mut consumer = ConstraintConsumer::new(
            vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
            F::ONE,
            F::ZERO,
            F::ZERO,
        );
        program.eval_packed_generic(&frame, &mut consumer);
        assert!(
            consumer.accumulators().iter().any(|&acc| acc != F::ZERO),
            "a false halfway flag must not permit an incorrect FP32 result"
        );
    }

    #[test]
    fn promotion_marks_far_subtraction_below_a_power_of_two() {
        let global = GlobalTriple {
            sign: false,
            exp: 153,
            sig24: SIG24_MIN,
        };
        let local = OutTriple {
            sign: true,
            exp: 121,
            sig14: SIG14_MIN,
        };
        let (promotion, rounded) = promotion_witness::<F>(global, local);
        assert_eq!(rounded.exp, 153);
        assert_eq!(promotion.binade_correction, F::ONE);
        let exact_binade = rounded.exp - promotion.binade_correction.to_canonical_u64();
        assert_eq!(exact_binade, 152);
    }

    fn promotion_constraints_hold(p: &PromotionView<F>, local: OutTriple) -> bool {
        let mut consumer = ConstraintConsumer::new(
            vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
            F::ONE,
            F::ZERO,
            F::ZERO,
        );
        let local = [
            F::from_bool(local.sig14 == 0),
            F::from_bool(local.sign),
            F::from_canonical_u64(local.exp),
            F::from_canonical_u64(local.sig14),
        ];
        eval_promotion_constraints(p, local, &mut NativeEvaluator::new(&mut consumer));
        consumer.accumulators().iter().all(|&acc| acc == F::ZERO)
    }

    fn promotion_lookups_hold(
        p: PromotionView<F>,
        local: OutTriple,
        checker: &mut crate::circuit::fp8::luts::LutChecker<F>,
    ) -> bool {
        use crate::circuit::fp8::luts::LutTable;
        let row = MatmulColumnsView {
            promotion: p,
            incoming_carry_is_zero: F::ONE,
            incoming_carry_shift_power: F::ONE,
            group_sum_is_zero: F::from_bool(local.sig14 == 0),
            group_output_significand: F::from_canonical_u64(local.sig14),
            group_output_exponent: F::from_canonical_u64(local.exp),
            e_cell: F::from_canonical_u64(300),
            e_grid: F::from_canonical_u64(300),
            ..Default::default()
        };
        let row: [F; NUM_MATMUL_COLUMNS] = row.into();
        let trace: Vec<_> = row.into_iter().map(|v| PolynomialValues::new(vec![v])).collect();
        let lookups: Vec<_> = super::super::ctl::matmul_lut_lookups::<F>()
            .into_iter()
            .filter(|lookup| matches!(lookup.table, LutTable::Range16 | LutTable::Pow2G))
            .collect();
        checker.check_trace(&lookups, &trace, &[], "promotion").is_ok()
    }

    #[test]
    fn promotion_lookups_reject_repaired_integer_and_slot_aliases() {
        use crate::circuit::fp8::luts::{LutChecker, LutTable};
        let mut checker = LutChecker::<F>::new(&[LutTable::Range16, LutTable::Pow2G]);
        let local = OutTriple {
            sign: false,
            exp: 139,
            sig14: SIG14_MIN + 1,
        };
        let (p, _) = promotion_witness::<F>(
            GlobalTriple {
                sign: false,
                exp: 139,
                sig24: SIG24_MIN,
            },
            local,
        );
        assert!(promotion_constraints_hold(&p, local));
        assert!(promotion_lookups_hold(p, local, &mut checker));

        // Preserve Q while replacing its high limb with a field fraction; its scaled range key still fits.
        let mut forged = p;
        forged.quotient_limbs[1] += forged.quotient_limbs[0] / F::from_canonical_u64(LIMB_BASE);
        forged.quotient_limbs[0] = F::ZERO;
        assert!(promotion_constraints_hold(&forged, local));
        assert!(!promotion_lookups_hold(forged, local, &mut checker));

        // Width -127 addresses the gap-1 entry with the same power. The operation tag must reject it.
        let mut forged = p;
        forged.exact_width -= F::from_canonical_u64(152);
        forged.global_out_exponent -= F::from_canonical_u64(152);
        assert!(promotion_constraints_hold(&forged, local));
        assert!(!promotion_lookups_hold(forged, local, &mut checker));

        // Recomposition alone allows limb aliases, even when the represented integer is unchanged.
        let mut forged = p;
        forged.magnitude_or_remainder_limbs[0] += F::from_canonical_u64(LIMB_BASE);
        forged.magnitude_or_remainder_limbs[1] -= F::ONE;
        assert!(promotion_constraints_hold(&forged, local));
        assert!(!promotion_lookups_hold(forged, local, &mut checker));

        // Falsify quotient parity at an exact tie and repair the rounded result and inverse.
        let local = OutTriple {
            sign: false,
            exp: 140,
            sig14: SIG14_MIN,
        };
        let (mut forged, _) = promotion_witness::<F>(
            GlobalTriple {
                sign: false,
                exp: 139,
                sig24: SIG24_MIN + 1,
            },
            local,
        );
        assert_eq!(forged.remainder_eq_half, F::ONE);
        assert_eq!(forged.quotient_lsb, F::ZERO);
        assert!(promotion_lookups_hold(forged, local, &mut checker));
        forged.quotient_lsb = F::ONE;
        forged.round_up = F::ONE;
        forged.global_out_significand += F::ONE;
        forged.carry_slack_inverse = (F::from_canonical_u64(1 << 24) - forged.global_out_significand).inverse();
        assert!(promotion_constraints_hold(&forged, local));
        assert!(!promotion_lookups_hold(forged, local, &mut checker));
    }

    #[test]
    fn committed_lookups_reject_understated_policy_anchors_and_census() {
        use crate::circuit::fp8::luts::{LutChecker, LutTable};
        let (program, rows, pis) = test_trace();
        let lookups = super::super::ctl::matmul_lut_lookups::<F>();
        let mut checker = LutChecker::<F>::new(&[LutTable::Range16, LutTable::Pow2G, LutTable::WidthNorm, LutTable::ProdAlign15]);
        let check = |checker: &mut LutChecker<F>, rows: &[[F; NUM_MATMUL_COLUMNS]]| {
            let trace: Vec<_> = (0..NUM_MATMUL_COLUMNS)
                .map(|c| PolynomialValues::new(rows.iter().map(|r| r[c]).collect()))
                .collect();
            checker.check_trace(&lookups, &trace, &pis, "matmul")
        };
        check(&mut checker, &rows).unwrap();
        for attack in 0..3 {
            let mut forged = rows.clone();
            for row in &mut forged {
                let row: &mut MatmulColumnsView<F> = row.borrow_mut();
                match attack {
                    0 => row.e_cell = F::ZERO,
                    1 if row.cell_nonzero == F::ONE => row.e_grid = F::ONE,
                    1 => {}
                    _ => {
                        row.skip_flag.fill(F::ZERO);
                        row.cell_skips = F::ZERO;
                    }
                }
            }
            // The arithmetic permits conservative anchors/census; the LUT certificates bind their lower bounds.
            assert_all_constraints(program.clone(), &forged, &pis);
            assert!(check(&mut checker, &forged).is_err(), "policy attack {attack}");
        }
    }

    #[test]
    fn promotion_correction_matches_exact_integer_binades() {
        use crate::circuit::fp8::luts::{LutChecker, LutTable};
        let mut checker = LutChecker::<F>::new(&[LutTable::Range16, LutTable::Pow2G]);
        let mut seen = [false; 5]; // cancellation, near-low, near-high, rounding carry, far drop.
        for gap in [0, 1, 2, 12, 13, 23, 24, 25, 26, 27, 32, 63, 64, 89] {
            for global_sig in [
                SIG24_MIN,
                SIG24_MIN + 1,
                SIG24_MIN + 2,
                SIG24_MIN + 1023,
                2 * SIG24_MIN - 2,
                2 * SIG24_MIN - 1,
            ] {
                for local_sig in [SIG14_MIN, SIG14_MIN + 1, 2 * SIG14_MIN - 1] {
                    for global_larger in [false, true] {
                        for global_sign in [false, true] {
                            for local_sign in [false, true] {
                                let global = GlobalTriple {
                                    sign: global_sign,
                                    exp: 100 + if global_larger { gap } else { 0 },
                                    sig24: global_sig,
                                };
                                let local = OutTriple {
                                    sign: local_sign,
                                    exp: 100 + if global_larger { 0 } else { gap },
                                    sig14: local_sig,
                                };
                                let (p, rounded) = promotion_witness::<F>(global, local);
                                assert!(promotion_constraints_hold(&p, local), "{global:?} + {local:?}");
                                assert!(promotion_lookups_hold(p, local, &mut checker), "{global:?} + {local:?}");
                                let expected = global.to_f32() + f32::from_bits(local.to_f32_bits() as u32);
                                let expected_bits = if expected == 0.0 { 0 } else { expected.to_bits() };
                                assert_eq!(rounded.to_f32().to_bits(), expected_bits);

                                // Align in i128, independently of the promotion's width/rounding witness.
                                let signed = |negative, value: i128| if negative { -value } else { value };
                                let exact = signed(global_sign, i128::from(global_sig) << (global.exp - 100))
                                    + signed(local_sign, i128::from(local_sig << 10) << (local.exp - 100));
                                let exact_binade = if exact == 0 {
                                    0
                                } else {
                                    100 + u64::from(exact.unsigned_abs().ilog2()) - 23
                                };
                                assert_eq!(rounded.exp - to_u64(p.binade_correction), exact_binade);
                                seen[0] |= p.global_out_is_zero == F::ONE;
                                seen[1] |= p.near_low == F::ONE;
                                seen[2] |= p.near_high == F::ONE;
                                seen[3] |= p.near_high == F::ONE && p.binade_correction == F::ONE;
                                seen[4] |= p.far_active == F::ONE && p.binade_correction == F::ONE;

                                let mut forged = p;
                                forged.binade_correction = F::ONE - p.binade_correction;
                                if p.near_high == F::ONE {
                                    // Repair dependent outputs and the inverse before testing the carry certificate.
                                    let delta = forged.binade_correction - p.binade_correction;
                                    forged.global_out_exponent += delta;
                                    forged.global_out_significand -= delta * F::from_canonical_u64(SIG24_MIN);
                                    let quotient = p.quotient_limbs[0] + F::from_canonical_u64(LIMB_BASE) * p.quotient_limbs[1];
                                    let slack = F::from_canonical_u64(1 << 24) - quotient - p.round_up;
                                    forged.carry_slack_inverse = if slack == F::ZERO {
                                        F::ZERO
                                    } else {
                                        (F::ONE - forged.binade_correction) / slack
                                    };
                                }
                                let conservative = p.far_active == F::ONE && p.binade_correction == F::ONE;
                                assert_eq!(promotion_constraints_hold(&forged, local), conservative);
                            }
                        }
                    }
                }
            }
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn promotion_rejects_correction_on_zero_and_single_input_paths() {
        let global = GlobalTriple {
            sign: true,
            exp: 139,
            sig24: SIG24_MIN,
        };
        let local = OutTriple {
            sign: false,
            exp: 139,
            sig14: SIG14_MIN,
        };
        for (global, local) in [
            (GlobalTriple::ZERO, OutTriple::ZERO),
            (global, OutTriple::ZERO),
            (GlobalTriple::ZERO, local),
            (global, local),
        ] {
            let (mut p, _) = promotion_witness::<F>(global, local);
            assert!(promotion_constraints_hold(&p, local));
            assert_eq!(p.binade_correction, F::ZERO);
            p.binade_correction = F::ONE;
            assert!(!promotion_constraints_hold(&p, local));
        }
    }

    fn assert_policy_replay(program: S, a: &[u8], b: &[u8]) {
        use crate::circuit::fp8::unpredictability::grid_exponent_from_binades;

        let (h, w, k) = (program.h, program.w, program.k);
        let a_lambdas = test_lambdas(h * k, 0xA24BAED4963EE407);
        let b_lambdas = test_lambdas(w * k, 0x9FB21C651E98DF25);
        let (rows, pis) = program.generate_trace(a, b, &a_lambdas, &b_lambdas);

        let (_, magnitudes) = H100 {}.matmul_fp8_policy_replay(a, b, h, w, k).unwrap();
        let partials = H100 {}.matmul_fp8_partials(a, b, h, w, k).unwrap();
        for cell in 0..h * w {
            let (r, c) = (cell / w, cell % w);
            let cell_rows = &rows[cell * program.rows_per_cell()..(cell + 1) * program.rows_per_cell()];
            assert_eq!(cell_rows.len(), partials[cell].len());
            for (atom, (row, expected)) in cell_rows.iter().zip(&partials[cell]).enumerate() {
                let v: &MatmulColumnsView<F> = row.borrow();
                assert_eq!(
                    promotion_bits(&v.promotion),
                    expected.to_bits(),
                    "cell {cell} atom {atom}: projected globalized partial differs"
                );
            }
            let expected = magnitudes[cell].m_cell.map_or(0, |e| (e + 139) as u64);
            let mut expected_skips = 0;
            for (atom, row) in cell_rows.iter().enumerate() {
                let v: &MatmulColumnsView<F> = row.borrow();
                let window = atom / 4;
                let expected_grid =
                    grid_exponent_from_binades(Device::H100, magnitudes[cell].z_windows[window], magnitudes[cell].m_cell);
                assert_eq!(to_u64(v.e_cell), expected, "cell {cell}");
                assert_eq!(to_u64(v.e_grid), expected_grid, "cell {cell} window {window}");
                assert_eq!(v.cell_nonzero, F::from_bool(expected_grid != 0), "cell {cell}");
                for i in 0..GROUP_WIDTH {
                    let u = atom * GROUP_WIDTH + i;
                    let skipped = expected_grid != 0 && a_lambdas[r * k + u] + b_lambdas[c * k + u] < 128 * expected_grid + 45952;
                    assert_eq!(v.skip_flag[i], F::from_bool(skipped), "cell {cell} atom {atom} lane {i}");
                    expected_skips += u64::from(skipped);
                }
                assert_eq!(to_u64(v.cell_skips), expected_skips, "cell {cell} atom {atom}");
                if atom + 1 == program.rows_per_cell() {
                    let bits = to_u64(v.cell_result_f32_lo) | (to_u64(v.cell_result_f32_hi) << 16);
                    assert_eq!(bits as u32, partials[cell][atom].to_bits());
                }
            }
        }
        assert_all_constraints(program, &rows, &pis);
    }

    #[test]
    fn policy_anchors_match_the_plaintext_hopper_replay() {
        for k in [32, 64, 96, 128, 160, 192, 224, 256, 544, 1024, 65536] {
            let program = S::new(2, 3, k);
            let mut a = test_codes(program.h * k, 0x9E3779B97F4A7C15, 8);
            a[..k].fill(0);
            let b = test_codes(program.w * k, 0xC2B2AE3D27D4EB4F, 11);
            assert_policy_replay(program, &a, &b);
        }
    }

    #[test]
    fn policy_grid_tracks_each_window_and_the_whole_cell() {
        let program = S::new(1, 1, 288);
        let mut a = vec![0; program.k];
        let mut b = vec![0; program.k];
        a[..128].fill(0x01);
        b[..128].fill(0x01);
        a[128..256].fill(0x7e);
        b[128..256].fill(0x7e);
        // The empty final window still uses M from the earlier large window.
        assert_policy_replay(program.clone(), &a, &b);
        let lambdas = test_lambdas(program.k, 1);
        let (rows, _) = program.generate_trace(&a, &b, &lambdas, &lambdas);
        let first: &MatmulColumnsView<F> = rows[0].borrow();
        let large: &MatmulColumnsView<F> = rows[4].borrow();
        let last: &MatmulColumnsView<F> = rows[8].borrow();
        assert_eq!(first.e_grid, first.e_cell);
        assert_eq!(last.e_grid, first.e_cell);
        assert_eq!(to_u64(large.e_grid), to_u64(first.e_cell) + 5);
    }

    #[test]
    fn trace_results_are_bit_exact_vs_fp8_sim() {
        let (program, rows, _) = test_trace();
        let (h, w, k) = (program.h, program.w, program.k);
        let a = test_codes(h * k, 0x9E3779B97F4A7C15, 8);
        let b = test_codes(w * k, 0xC2B2AE3D27D4EB4F, 11);
        let expected = ref_matmul_fp8(&a, &b, h, w, k);
        let native: Vec<u32> = H100 {}
            .matmul_fp8(&a, &b, None, h, w, k)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(expected, native, "independent reset-and-promote replay differs");
        let rows_per_cell = program.rows_per_cell();
        for (cell, chunk) in rows[..program.live_rows()].chunks(rows_per_cell).enumerate() {
            let last: &MatmulColumnsView<F> = chunk.last().unwrap().borrow();
            assert_eq!(last.is_cell_final, F::ONE);
            assert_eq!(last.is_padding, F::ZERO);
            let got = to_u64(last.cell_result_f32_lo) | (to_u64(last.cell_result_f32_hi) << 16);
            assert_eq!(got as u32, expected[cell], "cell {cell}: f32 result differs from fp8_sim");
        }
    }

    #[test]
    fn padded_trace_shape_and_known_values() {
        // 15 cells of k/32 = 4 rows -> 60 live rows, padded to 64 by 4 single-row phantom
        // cells. The known columns must be bit-exact with the trace fill (the batch verifier's
        // check), and every phantom row must hold zero products on the +0 path (M1's padding
        // pin and the operand-channel filter).
        let program = S::new(3, 5, 128);
        let a = test_codes(program.h * program.k, 0x9E3779B97F4A7C15, 8);
        let b = test_codes(program.w * program.k, 0xC2B2AE3D27D4EB4F, 11);
        let (rows, _) = program.generate_trace(
            &a,
            &b,
            &test_lambdas(program.h * program.k, 0xA24BAED4963EE407),
            &test_lambdas(program.w * program.k, 0x9FB21C651E98DF25),
        );
        assert_eq!(program.rows_per_cell(), 4);
        assert_eq!(program.live_rows(), 60);
        assert_eq!(program.num_rows(), 64);
        assert_eq!(rows.len(), 64);
        let known = program.known_values();
        assert_eq!(known.len(), super::super::columns::NUM_MATMUL_H100_KNOWN_COLUMNS);
        for (c, col) in known.iter().enumerate() {
            assert_eq!(col.len(), 64);
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(col.values[r], row[c], "known column {c} row {r}");
            }
        }
        for (r, row) in rows.iter().enumerate() {
            let v: &MatmulColumnsView<F> = row.borrow();
            assert_eq!(v.is_padding == F::ONE, r >= program.live_rows(), "row {r}");
            if v.is_padding == F::ONE {
                assert_eq!(v.group_sum_is_zero, F::ONE, "phantom row {r} must sit on the +0 path");
                for i in 0..GROUP_WIDTH {
                    assert_eq!(v.operand_codes_a[i], F::ZERO, "phantom row {r} lane {i}");
                    assert_eq!(v.operand_codes_b[i], F::ZERO, "phantom row {r} lane {i}");
                    assert_eq!(
                        v.prod_fp22_exp[i],
                        F::from_canonical_u64(FP22_ZERO_SENTINEL),
                        "phantom row {r} lane {i}"
                    );
                    assert_eq!(v.aligned_lane_terms[i], F::ZERO, "phantom row {r} lane {i}");
                }
            }
        }
    }

    #[test]
    fn ref_port_matches_native_h100_on_single_groups() {
        // For k = 32 (one WGMMA instruction) the native `H100::matmul_fp8` and `fp8_sim` agree
        // (their TILE_D = 16 chunking discrepancy — a known native-semantics divergence — only
        // bites for k > 32).
        // Zero results are normalized: the AIR/simulator force +0 while the native GFloat keeps
        // the accumulator's sign.
        let hw = H100 {};
        let (h, w, k) = (8, 8, GROUP_WIDTH);
        for salt in [3u64, 0x9E3779B97F4A7C15, 0xD1B54A32D192ED03] {
            let a = test_codes(h * k, salt, 5);
            let b = test_codes(w * k, salt.wrapping_mul(0x2545F4914F6CDD1D), 7);
            let expected: Vec<u32> = hw
                .matmul_fp8(&a, &b, None, h, w, k)
                .unwrap()
                .into_iter()
                .map(|v| if v == 0.0 { 0 } else { v.to_bits() })
                .collect();
            let got: Vec<u32> = ref_matmul_fp8(&a, &b, h, w, k)
                .into_iter()
                .map(|bits| if bits == 0x8000_0000 { 0 } else { bits })
                .collect();
            assert_eq!(got, expected, "fp8_sim port disagrees with native H100 (salt {salt:#x})");
        }
    }

    #[test]
    fn degree_is_at_most_three() {
        test_stark_low_degree::<F, S, D>(test_program()).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        test_stark_circuit_constraints::<F, C, S, D>(test_program()).unwrap();
    }

    // No standalone prove/verify smoke test: this table is a CTL party (`requires_ctls`), so a
    // proof without the cross-table argument is not a supported object — the proof shape
    // promises CTL openings the single-table prover has no data for. The end-to-end proving
    // path is covered by `fp8::driver::tests::batch_proof_roundtrips_and_rejects_tampering`.
}
