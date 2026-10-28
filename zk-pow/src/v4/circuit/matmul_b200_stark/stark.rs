//! Proves B200's 32-lane fp8 E4M3 matrix-multiplication accumulation.
//!
//! # E4M3 decode and stored exponents
//!
//! Decode an operand as `+/-mag*2^(shift-9)`: for a normal code,
//! `mag = 8 + mantissa` and `shift = exponent_field - 1`; for a subnormal,
//! `mag = mantissa` and `shift = 0`. Thus an exact nonzero product is
//!
//! ```text
//! product = +/-P*2^(L-18)
//! P       = mag_a*mag_b in [1, 225]
//! L       = shift_a + shift_b in [0, 28].
//! ```
//!
//! B200's stored product exponent is `L - 12`. Adding bias 38 gives the committed
//! `PRODUCT_BIASED_EXPONENT = L + 26`, which spans `[26, 54]`; zero uses sentinel 0 (which is
//! what MB1 pins on padding rows).
//!
//! # One hardware step per row
//!
//! A live row consumes 32 products and the preceding row's carry. Let
//! `E = GROUP_MAX_BIASED_EXPONENT` be the maximum stored exponent among them and
//! `REL_i = E - PRODUCT_BIASED_EXPONENT_i`. B200's accumulation window keeps 25 fractional
//! bits below that anchor, so each lane contributes
//!
//! ```text
//! lane_term_i = +/-floor(P_i*2^19 / 2^REL_i).
//! ```
//!
//! The 19 follows directly from the decode above and the 25-bit window:
//! dividing `P*2^(L-18)` by the window unit leaves `P*2^(19-REL)`. An attaining lane is
//! therefore `< 225*2^19 < 2^27`; a relative shift of 27 or more discards it completely.
//!
//! The carry retains a full 24-bit f32 significand `C` and has biased exponent `CE`. In the
//! same window it contributes
//! `+/-floor(4*C / 2^(E-CE))`. The factor four converts the carry's 23 fractional bits to the
//! window's 25. Because `4*C < 2^26`, shifts are capped at 26: every larger shift also yields
//! zero.
//!
//! The 32 lane terms and carry sum exactly as a signed integer, with magnitude `< 2^32`.
//! B200 truncates this sum toward zero to a normalized 24-bit significand in
//! `[2^23, 2^24)`. If its bit length is `W`, the next biased exponent is `E + W - 26`;
//! 26 is the 25 window bits plus the leading-bit position. Adding `127 - 38 = 89` converts that
//! exponent to the f32 field. A cell-final row exports the resulting exact f32 word; a zero sum
//! exports positive zero.
//!
//! # Padding
//!
//! Each cell is exactly its `k/32` live atom-steps; single-row phantom cells pad the full
//! trace to a power of two and read no operands, emit no output.
//!
//! MB1-MB16 label the constraint groups. MB13 commits the cell's replay-magnitude binade
//! E_CELL (jackpot check 3); MB14-MB16 commit the per-lane skip census over the summand
//! scores received from InputQuant (jackpot check 4).
//! A complete batch proof also includes the B200ALIGN, POW2GB, WIDTH32, and RC16 relations
//! declared in `super::ctl`.

use core::borrow::Borrow;
use std::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use starky::stark::Stark;

use super::columns::{
    GROUP_WIDTH, HK_PUBLIC_INPUT, HW_PUBLIC_INPUT, MATMUL_B200_COL_MAP, MatmulB200ColumnsView, NUM_ATT_LINKS,
    NUM_MATMUL_B200_COLUMNS, NUM_MATMUL_PUBLIC_INPUTS, ROWS_PER_CELL_PUBLIC_INPUT, W_PUBLIC_INPUT,
};
use crate::v4::circuit::unpredictability::skip;
use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// Biased stored-exponent sum of a nonzero fp8 product:
/// `ea + eb + 24 = (ea + eb - 14) + 38`, where -14 is the two-operand E4M3 stored-scale
/// correction and +38 reserves 0 as the zero sentinel.
const PRODUCT_BIASED_EXPONENT_OFFSET: u64 = 24;
/// The B200 window keeps 25 fractional bits below the anchor: carry shifts are capped at 26
/// (POW2GB's min-cap), which floors a far carry to zero exactly like the window drop does
/// (`4 * GROUP_OUTPUT_SIGNIFICAND < 2^26`).
const CARRY_SHIFT_CAP: u64 = 26;
/// Stored exponent bias 38 to f32 bias 127: `127 - 38 = 89`.
const F32_REBIAS: u64 = 89;
/// Truncated significands live in `[2^23, 2^24)`.
const SIG24_MIN: u64 = 1 << 23;
/// B200's full f32 significand width, to which each group output is truncated toward zero.
const OUT_WIDTH: u64 = 24;
/// A nonzero partial sum is `SIG24 * 2^(EXPB - 61)` with `SIG24 in [2^23, 2^24)`
/// (`EXPB = GROUP_OUTPUT_BIASED_EXPONENT`), so `floor(log2 |partial|) = EXPB - 38` and its
/// check-3 binade is `EXPB - 38 + 139 = EXPB + 101`. MB13's range check subtracts this offset.
pub(crate) const PARTIAL_BINADE_OFFSET: u64 = BINADE_BIAS - 38;
/// The shift on every jackpot check 3 binade (`E_CELL`, `LANE_BINADES`):
/// `E = floor(log2 |value|) + 139`, so all binades are nonnegative and comparable.
/// 0 marks a zero value.
pub(crate) const BINADE_BIAS: u64 = 139;

// ==================================================================================================
// Program and trace generation
// ==================================================================================================

/// The B200 tcgen05 AIR for one FP8 matmul: `out[r, c] = sum_t A[r, t] * B[t, c]`.
///
/// Schedule constraints read `[ROWS_PER_CELL, W, H*W, H*K]`; the shared schedule docs
/// constrain canonical coordinates. The AIR has no known columns.
#[derive(Clone, Debug)]
pub struct MatmulStarkB200<F: RichField + Extendable<D>, const D: usize> {
    /// Output rows (rows of A).
    pub h: usize,
    /// Output columns (columns of B).
    pub w: usize,
    /// Inner dimension; must be a multiple of [`GROUP_WIDTH`].
    pub k: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> MatmulStarkB200<F, D> {
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
    /// cells (all-zero, `is_padding = is_cell_final = 1`; the coordinate walk continues at
    /// position `h*w + t`).
    pub fn num_rows(&self) -> usize {
        self.live_rows().next_power_of_two()
    }

    /// The geometry public inputs the MS schedule constraints read:
    /// `[ROWS_PER_CELL, W, H*W, H*K]`. Generated with the trace and pinned by the verifier's
    /// expectation ([`super::super::driver::Fp8System::verify`]).
    pub fn public_inputs(&self) -> [F; NUM_MATMUL_PUBLIC_INPUTS] {
        [
            F::from_canonical_usize(self.rows_per_cell()),
            F::from_canonical_usize(self.w),
            F::from_canonical_usize(self.h * self.w),
            F::from_canonical_usize(self.h * self.k),
        ]
    }
}

/// Schedule columns shared by both device AIRs and their witness generators.
pub(crate) struct MatmulScheduleView<F> {
    pub is_cell_final: F,
    pub is_padding: F,
    pub cell_step: F,
    pub cell_step_inv: F,
    pub out_row: F,
    pub out_col: F,
    pub padding_onset_inv: F,
    pub operand_index_base_a: F,
    pub operand_index_base_b: F,
    pub col_carry: F,
    pub col_carry_inv: F,
}

impl<F: RichField> MatmulScheduleView<F> {
    /// The honest schedule at global row `row` for `hw` live cells of `rows_per_cell` rows,
    /// `w` output columns. Phantom rows (cell index >= `hw`) are single-row cells whose
    /// `(OUT_ROW, OUT_COL)` walk — and with it the operand bases — simply continues.
    pub fn at(row: usize, rows_per_cell: usize, w: usize, hw: usize) -> Self {
        let live = hw * rows_per_cell;
        let (cell, step) = if row < live {
            (row / rows_per_cell, row % rows_per_cell)
        } else {
            (hw + (row - live), 0)
        };
        let is_padding = cell >= hw;
        let is_cell_final = is_padding || step == rows_per_cell - 1;
        let cell_f = F::from_canonical_usize(cell);
        let hw_f = F::from_canonical_usize(hw);
        let k = rows_per_cell * GROUP_WIDTH;
        let hk = (hw / w) * k;
        Self {
            is_cell_final: F::from_bool(is_cell_final),
            is_padding: F::from_bool(is_padding),
            cell_step: F::from_canonical_usize(step),
            cell_step_inv: (F::from_canonical_usize(step) - F::from_canonical_usize(rows_per_cell - 1))
                .try_inverse()
                .unwrap_or(F::ZERO),
            out_row: F::from_canonical_usize(cell / w),
            out_col: F::from_canonical_usize(cell % w),
            padding_onset_inv: (cell_f - hw_f).try_inverse().unwrap_or(F::ZERO),
            operand_index_base_a: F::from_canonical_usize((cell / w) * k),
            operand_index_base_b: F::from_canonical_usize(hk + (cell % w) * k),
            col_carry: F::from_bool(cell % w == w - 1),
            col_carry_inv: (F::from_canonical_usize(cell % w) - F::from_canonical_usize(w - 1))
                .try_inverse()
                .unwrap_or(F::ZERO),
        }
    }
}

/// One decoded fp8 x fp8 product in the B200 frame (`multiply_fp8_to_gfloat` re-derived on raw
/// codes); also the B200ALIGN row function (`super::super::luts`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct B200Product {
    /// Exact significand `sig_a * sig_b` in `[0, 225]` (4-bit x 4-bit, subnormals unnormalized).
    pub(crate) sig: u64,
    /// XOR of the operand signs.
    pub(crate) sign: bool,
    /// Biased stored-exponent sum (`ea + eb + 24 in [26, 54]`), or the sentinel 0 for zero
    /// products.
    pub(crate) biased_exponent: u64,
    pub(crate) is_zero: bool,
}

impl B200Product {
    const ZERO: Self = Self {
        sig: 0,
        sign: false,
        biased_exponent: 0,
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
            biased_exponent: exp_a + exp_b + PRODUCT_BIASED_EXPONENT_OFFSET,
            is_zero: false,
        }
    }

    /// The product's check-3 binade `floor(log2 |product|) + 139`, or 0 for a zero product
    /// (the B200ALIGN `BINADE` column). `|product| = SIG * 2^(BIASED_EXPONENT - 44)`,
    /// so the binade is `bit_length(SIG) - 1 + BIASED_EXPONENT - 44 + 139`. Nonzero
    /// values span `[121, 156]`.
    pub(crate) fn biased_binade(&self) -> u64 {
        if self.is_zero {
            0
        } else {
            bit_length(self.sig) + self.biased_exponent + BINADE_BIAS - 45
        }
    }
}

/// Raw e4m3 byte -> (sign, exponent bits with subnormals clamped to 1, significand with the
/// implicit bit for normals) — the shared E4M3 split (`multiply_fp8_to_gfloat` /
/// `fp8_sim._split_e4m3`); B200 does *not* renormalize subnormal operands either.
fn split_e4m3(code: u8) -> (bool, u64, u64) {
    let sign = code >> 7 == 1;
    let exp = ((code >> 3) & 0xF) as u64;
    let man = (code & 0x7) as u64;
    let sig = if exp != 0 { man | 0x8 } else { man };
    (sign, exp.max(1), sig)
}

/// A group-sum output (equivalently, the carry entering the next row): sign, biased-38
/// exponent, 24-bit significand. `ZERO` is the +0 constant of the zero path (B200's zero
/// condition is exactly `SUM = 0`; -0 never exits a group).
#[derive(Clone, Copy, Debug)]
struct OutTriple {
    sign: bool,
    expb: u64,
    sig24: u64,
}

impl OutTriple {
    const ZERO: Self = Self {
        sign: false,
        expb: 0,
        sig24: 0,
    };

    /// The exact f32 bit pattern of this output (`GFloat -> f32`; the MB12 encode): rebias
    /// +38 -> +127 (`+89`) and drop the implicit bit — B200 keeps the full 23-bit fraction
    /// (no widening step). Zero encodes to +0.
    fn to_f32_bits(self) -> u64 {
        if self.sig24 == 0 {
            return 0;
        }
        ((self.sign as u64) << 31) | ((self.expb + F32_REBIAS) << 23) | (self.sig24 - SIG24_MIN)
    }
}

/// `bit_length(x)` for `x > 0` (0 for 0).
fn bit_length(x: u64) -> u64 {
    (64 - x.leading_zeros()) as u64
}

impl<F: RichField + Extendable<D>, const D: usize> MatmulStarkB200<F, D> {
    /// Generates this AIR's trace and public inputs: `a_codes` row-major, `b_codes`
    /// column-major (B transposed), one row per 32-lane atom-step, cells row-major.
    ///
    /// `a_lambdas` and `b_lambdas` are InputQuant's deterministic per-element
    /// unpredictability scores. Their operand-code CTL binds each score to the corresponding
    /// post-noise FP8 E4M3 code, so neither vector is an independent prover choice.
    pub fn generate_trace(
        &self,
        a_codes: &[u8],
        b_codes: &[u8],
        a_lambdas: &[u64],
        b_lambdas: &[u64],
    ) -> (Vec<[F; NUM_MATMUL_B200_COLUMNS]>, [F; NUM_MATMUL_PUBLIC_INPUTS]) {
        let (h, w, k) = (self.h, self.w, self.k);
        assert_eq!(a_codes.len(), h * k, "a_codes must be h*k row-major fp8 codes");
        assert_eq!(b_codes.len(), w * k, "b_codes must be w*k column-major fp8 codes");
        assert_eq!(a_lambdas.len(), h * k, "a_lambdas must align with a_codes");
        assert_eq!(b_lambdas.len(), w * k, "b_lambdas must align with b_codes");
        assert_eq!(k % GROUP_WIDTH, 0, "k must be a multiple of the group width");
        let live_rows_per_cell = k / GROUP_WIDTH;
        let num_rows = self.num_rows();

        // Cells are independent, so the trace fills one chunk of `rows_per_cell` (one cell)
        // per parallel job; the schedule's inverse witnesses take only a handful of distinct
        // values each, precomputed once below instead of inverted per row.
        let step_inv: Vec<F> = (0..live_rows_per_cell)
            .map(|s| {
                (F::from_canonical_usize(s) - F::from_canonical_usize(live_rows_per_cell - 1))
                    .try_inverse()
                    .unwrap_or(F::ZERO)
            })
            .collect();
        let col_inv: Vec<F> = (0..w)
            .map(|c| {
                (F::from_canonical_usize(c) - F::from_canonical_usize(w - 1))
                    .try_inverse()
                    .unwrap_or(F::ZERO)
            })
            .collect();

        let mut rows: Vec<[F; NUM_MATMUL_B200_COLUMNS]> = vec![[F::ZERO; NUM_MATMUL_B200_COLUMNS]; num_rows];
        {
            let live = self.live_rows();
            let map = MATMUL_B200_COL_MAP;
            let rows = &mut rows;
            // ---- Live cells: one chunk of `rows_per_cell` rows per cell. ----
            use plonky2_maybe_rayon::{IndexedParallelIterator, MaybeParChunksMut, ParallelIterator};
            rows[..live]
                .par_chunks_mut(live_rows_per_cell)
                .enumerate()
                .for_each(|(cell, chunk)| {
                    let (r, c) = (cell / w, cell % w);
                    let cell_f = F::from_canonical_usize(cell);
                    let (out_row, out_col) = (cell / w, cell % w);
                    let (padding_onset_inv, col_carry, col_carry_inv, base_a, base_b) = (
                        (cell_f - F::from_canonical_usize(h * w)).try_inverse().unwrap_or(F::ZERO),
                        F::from_bool(c == w - 1),
                        col_inv[c],
                        F::from_canonical_usize(r * k),
                        F::from_canonical_usize(h * k + c * k),
                    );
                    // ---- The k/32 live atom-steps (window-sum emulation columns). ----
                    let mut carry = OutTriple::ZERO;
                    let mut incoming_carry_is_zero = true;
                    // Jackpot check 3 (MB13): E_CELL = max of floor(log2 |x|) + 139 over the cell's
                    // products and partial sums (0 if all products are zero); filled in after the loop.
                    let mut e_cell = 0u64;
                    for (j, row_slot) in chunk.iter_mut().enumerate() {
                        let is_final = j == live_rows_per_cell - 1;
                        let mut row = MatmulB200ColumnsView {
                            is_cell_final: F::from_bool(is_final),
                            incoming_carry_is_zero: F::from_bool(incoming_carry_is_zero),
                            ..Default::default()
                        };

                        // Decode the 32 lanes and receive their summand scores.
                        let mut lanes = [B200Product::ZERO; GROUP_WIDTH];
                        for i in 0..GROUP_WIDTH {
                            let operand_codes_a = a_codes[r * k + j * GROUP_WIDTH + i];
                            let operand_codes_b = b_codes[c * k + j * GROUP_WIDTH + i];
                            lanes[i] = B200Product::new(operand_codes_a, operand_codes_b);
                            row.operand_codes_a[i] = F::from_canonical_u8(operand_codes_a);
                            row.operand_codes_b[i] = F::from_canonical_u8(operand_codes_b);
                            row.product_biased_exponents[i] = F::from_canonical_u64(lanes[i].biased_exponent);
                            row.lambda_a[i] = F::from_canonical_u64(a_lambdas[r * k + j * GROUP_WIDTH + i]);
                            row.lambda_b[i] = F::from_canonical_u64(b_lambdas[c * k + j * GROUP_WIDTH + i]);
                            // MB13: the lane's binade, as served by B200ALIGN.
                            let lane_binade = lanes[i].biased_binade();
                            row.lane_binades[i] = F::from_canonical_u64(lane_binade);
                            e_cell = e_cell.max(lane_binade);
                        }

                        // The window anchor: the max biased stored exponent over the nonzero summands
                        // (zero terms never set it, mirroring `windowed_group_sum`'s nonzero filter);
                        // the all-zero row's honest anchor is the sentinel 0 — every B200ALIGN key
                        // then sits at rel 0 and the attainment chain vanishes through the sentinel
                        // factors.
                        let mut group_max_biased_exponent: Option<u64> = (!incoming_carry_is_zero).then_some(carry.expb);
                        for lane in &lanes {
                            if !lane.is_zero {
                                group_max_biased_exponent =
                                    Some(group_max_biased_exponent.map_or(lane.biased_exponent, |m| m.max(lane.biased_exponent)));
                            }
                        }
                        let group_max_biased_exponent = group_max_biased_exponent.unwrap_or(0);
                        row.group_max_biased_exponent = F::from_canonical_u64(group_max_biased_exponent);
                        // The MB3 attainment chain: running products of the affine lane factors
                        // F_i = GROUP_MAX_BIASED_EXPONENT - PRODUCT_BIASED_EXPONENT_i (three into the first link,
                        // two per link after, the last one alone).
                        let factor =
                            |i: usize| F::from_canonical_u64(group_max_biased_exponent) - row.product_biased_exponents[i];
                        row.max_exponent_attainment[0] = factor(0) * factor(1) * factor(2);
                        for l in 1..NUM_ATT_LINKS - 1 {
                            row.max_exponent_attainment[l] =
                                row.max_exponent_attainment[l - 1] * factor(2 * l + 1) * factor(2 * l + 2);
                        }
                        row.max_exponent_attainment[NUM_ATT_LINKS - 1] =
                            row.max_exponent_attainment[NUM_ATT_LINKS - 2] * factor(GROUP_WIDTH - 1);

                        // Truncate the 33 summands into the window and sum exactly, as
                        // `windowed_group_sum` at width 26: products at the 27-bit grid `P * 2^19`,
                        // the carry at `4 * sig24`, each floor-shifted towards zero by its rel-shift.
                        let mut total: i64 = 0;
                        for i in 0..GROUP_WIDTH {
                            if lanes[i].is_zero {
                                continue; // ALIGNED_LANE_TERMS stays 0 for a zero product.
                            }
                            // Nonnegative because the anchor dominates every lane.
                            let rel = group_max_biased_exponent - lanes[i].biased_exponent;
                            let mag = ((lanes[i].sig << 19) >> rel.min(63)) as i64;
                            row.aligned_lane_terms[i] = if lanes[i].sign {
                                -F::from_canonical_u64(mag as u64)
                            } else {
                                F::from_canonical_u64(mag as u64)
                            };
                            total += if lanes[i].sign { -mag } else { mag };
                        }
                        if incoming_carry_is_zero {
                            // Free columns on zero-carry rows; the trace picks the (0, 0, 1) solution
                            // of the unfiltered remainder identity.
                            row.incoming_carry_shift_power = F::ONE;
                        } else {
                            let d = (group_max_biased_exponent - carry.expb).min(CARRY_SHIFT_CAP);
                            let shift_pow2 = 1u64 << d;
                            let aligned = (4 * carry.sig24) >> d;
                            let remainder = 4 * carry.sig24 - aligned * shift_pow2;
                            let bound = shift_pow2 - 1 - remainder;
                            row.incoming_carry_shift_power = F::from_canonical_u64(shift_pow2);
                            row.aligned_incoming_carry_lo = F::from_canonical_u64(aligned & 0xFFFF);
                            row.aligned_incoming_carry_hi = F::from_canonical_u64(aligned >> 16);
                            row.incoming_carry_remainder_lo = F::from_canonical_u64(remainder & 0xFFFF);
                            row.incoming_carry_remainder_hi = F::from_canonical_u64(remainder >> 16);
                            row.incoming_carry_remainder_bound_lo = F::from_canonical_u64(bound & 0xFFFF);
                            row.incoming_carry_remainder_bound_hi = F::from_canonical_u64(bound >> 16);
                            total += if carry.sign { -(aligned as i64) } else { aligned as i64 };
                        }

                        // Sum, truncate toward zero (WIDTH32 semantics), and emit through the mux.
                        row.group_sum_sign = F::from_bool(total < 0);
                        row.group_sum_abs = F::from_canonical_u64(total.unsigned_abs());
                        row.group_sum_is_zero = F::from_bool(total == 0);
                        debug_assert!(total.unsigned_abs() < 1 << 32, "window sums fit 32 bits");
                        let out = if total == 0 {
                            // The +0 path (Z = GROUP_SUM_IS_ZERO); WIDTH32 is filtered off, so the width
                            // block holds the inert (0; 1, 1; 0, 0, 0) fill.
                            row.truncation_power = F::ONE;
                            row.lifting_power = F::ONE;
                            OutTriple::ZERO
                        } else {
                            let mag = total.unsigned_abs();
                            let width = bit_length(mag);
                            let (truncation_power, lifting_power) = (
                                1u64 << width.saturating_sub(OUT_WIDTH),
                                1u64 << OUT_WIDTH.saturating_sub(width),
                            );
                            let sig24 = mag * lifting_power / truncation_power;
                            let truncation_remainder = mag * lifting_power - sig24 * truncation_power;
                            row.group_sum_width = F::from_canonical_u64(width);
                            row.truncation_power = F::from_canonical_u64(truncation_power);
                            row.lifting_power = F::from_canonical_u64(lifting_power);
                            row.normalized_group_sum_significand_lo = F::from_canonical_u64(sig24 & 0xFFFF);
                            row.normalized_group_sum_significand_hi = F::from_canonical_u64(sig24 >> 16);
                            row.truncation_remainder = F::from_canonical_u64(truncation_remainder);
                            row.truncation_remainder_bound = F::from_canonical_u64(truncation_power - 1 - truncation_remainder);
                            let expb = group_max_biased_exponent + width - 26;
                            debug_assert!(expb >= 1, "nonzero outputs never reach the zero sentinel");
                            OutTriple {
                                sign: total < 0,
                                expb,
                                sig24,
                            }
                        };
                        row.group_output_sign = F::from_bool(out.sign);
                        row.group_output_biased_exponent = F::from_canonical_u64(out.expb);
                        row.group_output_significand = F::from_canonical_u64(out.sig24);

                        if is_final {
                            let bits = out.to_f32_bits();
                            row.cell_result_f32_lo = F::from_canonical_u64(bits & 0xFFFF);
                            row.cell_result_f32_hi = F::from_canonical_u64(bits >> 16);
                        }
                        // Check 3: partial sums enter the max too, at binade EXPB + 101
                        // (`PARTIAL_BINADE_OFFSET`); zero partials have none.
                        if out.sig24 != 0 {
                            e_cell = e_cell.max(out.expb + PARTIAL_BINADE_OFFSET);
                        }
                        incoming_carry_is_zero = total == 0;
                        carry = out;

                        // ---- Schedule witnesses (the shared MS block; values match
                        // `MatmulScheduleView::at` exactly). ----
                        row.is_padding = F::ZERO;
                        row.cell_step = F::from_canonical_usize(j);
                        row.cell_step_inv = step_inv[j];
                        row.out_row = F::from_canonical_usize(out_row);
                        row.out_col = F::from_canonical_usize(out_col);
                        row.padding_onset_inv = padding_onset_inv;
                        row.operand_index_base_a = base_a;
                        row.operand_index_base_b = base_b;
                        row.col_carry = col_carry;
                        row.col_carry_inv = col_carry_inv;

                        *row_slot = row.into();
                    }

                    // ---- MB13-MB16 post-pass: fill the cell-constant E_CELL and the per-lane
                    // skip census (skip verdicts need the finished E_CELL). ----
                    let mut cell_skips = 0u64;
                    for (j, row) in chunk.iter_mut().enumerate() {
                        row[map.e_cell] = F::from_canonical_u64(e_cell);
                        row[map.cell_nonzero] = F::from_bool(e_cell != 0);
                        for i in 0..GROUP_WIDTH {
                            let lambda_a = a_lambdas[r * k + j * GROUP_WIDTH + i];
                            let lambda_b = b_lambdas[c * k + j * GROUP_WIDTH + i];
                            let skipped = skip(lambda_a, lambda_b, e_cell);
                            row[map.skip_flag[i]] = F::from_bool(skipped);
                            cell_skips += u64::from(skipped);
                        }
                        row[map.cell_skips] = F::from_canonical_u64(cell_skips);
                    }
                });

            // ---- Trailing padding: one all-zero single-row phantom cell per padding row,
            // its schedule columns matching `MatmulScheduleView::at` exactly. ----
            let live = self.live_rows();
            rows[live..].par_chunks_mut(1).enumerate().for_each(|(t, chunk)| {
                let row = &mut chunk[0];
                *row = phantom_row::<F>().into();
                let s = MatmulScheduleView::<F>::at(live + t, live_rows_per_cell, w, h * w);
                row[map.is_cell_final] = s.is_cell_final;
                row[map.is_padding] = s.is_padding;
                row[map.cell_step] = s.cell_step;
                row[map.cell_step_inv] = s.cell_step_inv;
                row[map.out_row] = s.out_row;
                row[map.out_col] = s.out_col;
                row[map.padding_onset_inv] = s.padding_onset_inv;
                row[map.operand_index_base_a] = s.operand_index_base_a;
                row[map.operand_index_base_b] = s.operand_index_base_b;
                row[map.col_carry] = s.col_carry;
                row[map.col_carry_inv] = s.col_carry_inv;
            });
        }

        (rows, self.public_inputs())
    }
}

/// An all-zero single-row trailing phantom cell (the schedule columns are filled by the
/// post-pass). It enters a zero carry with `GROUP_MAX_BIASED_EXPONENT = 0`, uses the inert
/// `(0; 1, 1; 0, 0, 0)` width block, and sets `INCOMING_CARRY_SHIFT_POWER = 1` for the
/// remainder identity.
fn phantom_row<F: RichField>() -> MatmulB200ColumnsView<F> {
    MatmulB200ColumnsView {
        incoming_carry_is_zero: F::ONE,
        incoming_carry_shift_power: F::ONE,
        group_sum_is_zero: F::ONE,
        truncation_power: F::ONE,
        lifting_power: F::ONE,
        ..Default::default()
    }
}

// ==================================================================================================
// Constraints, written once against the generic `Evaluator`
// ==================================================================================================

/// Shared schedule constraints using `[ROWS_PER_CELL, W, H*W, H*K]`.
///
/// The counters fix each cell's length and the padding suffix. Anchored at `(0, 0)`,
/// MS13/MS14 walk the canonical row-major coordinates `(OUT_ROW, OUT_COL)`; MS8 freezes
/// them within a cell. The weighted sum `OUT_ROW*W + OUT_COL` is the flat cell position:
/// the MS9/MS10 padding onset reads it, and the XorFold channel keys on it directly as a
/// public-input-scaled CTL expression. MS11/MS12 bind the operand bases to `OUT_ROW*K`
/// and `H*K + OUT_COL*K`. No coordinate range assumption comes from the CTLs.
/// All constraints have degree <= 3, including transition selectors.
pub(crate) fn eval_matmul_schedule_constraints<V, S, E>(
    lv: &MatmulScheduleView<V>,
    nv: &MatmulScheduleView<V>,
    public_inputs: &[S; NUM_MATMUL_PUBLIC_INPUTS],
    eval: &mut E,
) where
    V: Copy,
    S: Copy,
    E: Evaluator<V, S>,
{
    let rows_per_cell = eval.scalar(public_inputs[ROWS_PER_CELL_PUBLIC_INPUT]);
    let w = eval.scalar(public_inputs[W_PUBLIC_INPUT]);
    let hw = eval.scalar(public_inputs[HW_PUBLIC_INPUT]);
    let one = eval.i32(1);
    let not_final = eval.sub(one, lv.is_cell_final);
    let not_padding = eval.sub(one, lv.is_padding);

    // MS1 — the first row anchors the walk: step 0 of cell (0, 0), live.
    eval.constraint_first_row(lv.cell_step);
    eval.constraint_first_row(lv.out_row);
    eval.constraint_first_row(lv.out_col);
    eval.constraint_first_row(lv.is_padding);
    eval.constraint_bool(lv.is_padding);

    // The flat cell position `OUT_ROW*W + OUT_COL`: the MS7 advance and MS9/MS10
    // padding-onset operand (the XorFold channel key reuses the same expression).
    let cell_pos = eval.mad(lv.out_row, w, lv.out_col);

    // MS3 — the in-cell step counter: +1 inside a cell, reset after a cell-final row
    // (plain cyclic; the wrap closes because the last row is cell-final).
    let step_next = eval.add(lv.cell_step, one);
    let kept = eval.mul(step_next, not_final);
    let c = eval.sub(nv.cell_step, kept);
    eval.constraint(c);

    // MS4/MS5 — IS_CELL_FINAL is exactly the cell-bound indicator on live rows: a raised
    // flag forces CELL_STEP = ROWS_PER_CELL-1 (MS4, padding-gated: phantom steps stay 0),
    // and a suppressed flag forces CELL_STEP != ROWS_PER_CELL-1 through the inverse witness
    // (MS5; no gate needed — phantom rows have the flag raised).
    let last_step = eval.sub(rows_per_cell, one);
    let step_gap = eval.sub(lv.cell_step, last_step);
    let raised = eval.mul(lv.is_cell_final, step_gap);
    let c = eval.mul(not_padding, raised);
    eval.constraint(c);
    let witnessed = eval.mul(step_gap, lv.cell_step_inv);
    let witnessed = eval.sub(witnessed, one);
    let c = eval.mul(not_final, witnessed);
    eval.constraint(c);

    // MS6 — every phantom row is its own single-row cell.
    let c = eval.mul(lv.is_padding, not_final);
    eval.constraint(c);

    // MS7 — advance the weighted coordinate sum by one at cell boundaries; exclude the wrap.
    let next_pos = eval.mad(nv.out_row, w, nv.out_col);
    let advance = eval.sub(next_pos, cell_pos);
    let advance = eval.sub(advance, one);
    let c = eval.mul(lv.is_cell_final, advance);
    eval.constraint_transition(c);

    // MS8 — the cell indices freeze inside a cell (plain cyclic; cell-final last row).
    let dc = eval.sub(nv.out_col, lv.out_col);
    let c = eval.mul(not_final, dc);
    eval.constraint(c);
    let dr = eval.sub(nv.out_row, lv.out_row);
    let c = eval.mul(not_final, dr);
    eval.constraint(c);

    // MS9/MS10 — IS_PADDING: changes only entering position H*W (MS9, transition), forced
    // to 1 there by the inverse witness (MS10): a live claim needs the position gap
    // `OUT_ROW*W + OUT_COL - H*W` to be invertible.
    let dpad = eval.sub(nv.is_padding, lv.is_padding);
    let onset = eval.sub(next_pos, hw);
    let c = eval.mul(dpad, onset);
    eval.constraint_transition(c);
    let gap = eval.sub(cell_pos, hw);
    let witnessed = eval.mul(gap, lv.padding_onset_inv);
    let witnessed = eval.sub(witnessed, one);
    let c = eval.mul(not_padding, witnessed);
    eval.constraint(c);

    // MS11/MS12 — the operand flat-index bases: anchored on the first row (A at 0, B at the
    // H*K public input) and stepped by K times the pinned cell-index deltas (transitions;
    // K = ROWS_PER_CELL * GROUP_WIDTH).
    let hk = eval.scalar(public_inputs[HK_PUBLIC_INPUT]);
    let group_width = eval.i32(GROUP_WIDTH as i32);
    let k = eval.mul(rows_per_cell, group_width);
    eval.constraint_first_row(lv.operand_index_base_a);
    let anchor_b = eval.sub(lv.operand_index_base_b, hk);
    eval.constraint_first_row(anchor_b);
    let dbase = eval.sub(nv.operand_index_base_a, lv.operand_index_base_a);
    let dr = eval.sub(nv.out_row, lv.out_row);
    let step = eval.mul(k, dr);
    let c = eval.sub(dbase, step);
    eval.constraint_transition(c);
    let dbase = eval.sub(nv.operand_index_base_b, lv.operand_index_base_b);
    let dc_col = eval.sub(nv.out_col, lv.out_col);
    let step = eval.mul(k, dc_col);
    let c = eval.sub(dbase, step);
    eval.constraint_transition(c);

    // MS13 — COL_CARRY = [OUT_COL = W-1]: a zero gap forces 1; a nonzero gap forces 0.
    let w_minus_one = eval.sub(w, one);
    let col_gap = eval.sub(lv.out_col, w_minus_one);
    let c = eval.mul(lv.col_carry, col_gap);
    eval.constraint(c);
    let not_carry = eval.sub(one, lv.col_carry);
    let witnessed = eval.mul(col_gap, lv.col_carry_inv);
    let witnessed = eval.sub(witnessed, one);
    let c = eval.mul(not_carry, witnessed);
    eval.constraint(c);

    // MS14 — at cell boundaries, COL' = COL + 1 - W*COL_CARRY and ROW' = ROW + COL_CARRY.
    // MS13 makes the column reset linear; transitions exclude the last -> first wrap.
    let w_carry = eval.mul(w, lv.col_carry);
    let dc_step = eval.sub(dc_col, one);
    let dc_step = eval.add(dc_step, w_carry);
    let c = eval.mul(lv.is_cell_final, dc_step);
    eval.constraint_transition(c);
    let dr_boundary = eval.sub(dr, lv.col_carry);
    let c = eval.mul(lv.is_cell_final, dr_boundary);
    eval.constraint_transition(c);
}

/// Evaluates every arithmetic constraint of MatmulStarkB200. Lookup-borne facts (LUT oracle)
/// are *not* emitted here — see `super::ctl::matmul_b200_lut_lookups`. The constraint set
/// reads only the geometry public inputs `[ROWS_PER_CELL, W, H*W, H*K]` (the group-MS schedule
/// gadgets); everything else is program-independent.
pub(crate) fn eval_matmul_b200_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_MATMUL_B200_COLUMNS, NUM_MATMUL_PUBLIC_INPUTS>,
    eval: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_MATMUL_B200_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &MatmulB200ColumnsView<V> = lv.borrow();
    let nv: &[V; NUM_MATMUL_B200_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let nv: &MatmulB200ColumnsView<V> = nv.borrow();

    let one = eval.i32(1);
    eval_matmul_schedule_constraints(
        &MatmulScheduleView {
            is_cell_final: lv.is_cell_final,
            is_padding: lv.is_padding,
            cell_step: lv.cell_step,
            cell_step_inv: lv.cell_step_inv,
            out_row: lv.out_row,
            out_col: lv.out_col,
            padding_onset_inv: lv.padding_onset_inv,
            operand_index_base_a: lv.operand_index_base_a,
            operand_index_base_b: lv.operand_index_base_b,
            col_carry: lv.col_carry,
            col_carry_inv: lv.col_carry_inv,
        },
        &MatmulScheduleView {
            is_cell_final: nv.is_cell_final,
            is_padding: nv.is_padding,
            cell_step: nv.cell_step,
            cell_step_inv: nv.cell_step_inv,
            out_row: nv.out_row,
            out_col: nv.out_col,
            padding_onset_inv: nv.padding_onset_inv,
            operand_index_base_a: nv.operand_index_base_a,
            operand_index_base_b: nv.operand_index_base_b,
            col_carry: nv.col_carry,
            col_carry_inv: nv.col_carry_inv,
        },
        vars.get_public_inputs().try_into().unwrap(),
        eval,
    );
    let not_final = eval.sub(one, lv.is_cell_final);
    // Z := GROUP_SUM_IS_ZERO — B200's output-zero condition is exactly SUM = 0.
    let z = lv.group_sum_is_zero;
    let one_minus_z = eval.sub(one, z);
    let limb_shift = eval.u64(1 << 16);
    // The affine limb recompositions (never columns).
    let aligned_incoming_carry_next = eval.mad(nv.aligned_incoming_carry_hi, limb_shift, nv.aligned_incoming_carry_lo);
    let incoming_carry_remainder_next = eval.mad(nv.incoming_carry_remainder_hi, limb_shift, nv.incoming_carry_remainder_lo);
    let incoming_carry_remainder = eval.mad(lv.incoming_carry_remainder_hi, limb_shift, lv.incoming_carry_remainder_lo);
    let incoming_carry_remainder_bound = eval.mad(
        lv.incoming_carry_remainder_bound_hi,
        limb_shift,
        lv.incoming_carry_remainder_bound_lo,
    );
    let aligned_incoming_carry = eval.mad(lv.aligned_incoming_carry_hi, limb_shift, lv.aligned_incoming_carry_lo);
    let normalized_group_sum_significand = eval.mad(
        lv.normalized_group_sum_significand_hi,
        limb_shift,
        lv.normalized_group_sum_significand_lo,
    );

    // The transition constraints below are *plain* (they also bind the last-row -> first-row
    // wrap); the wrap instances are made inert by the last row being cell-final (forced by
    // the MS schedule constraints above) and row 0's anchored zero carry.

    // ---- MB1 — product lookups x32 (B200ALIGN; served by the LUT oracle). ----
    // The tuple binds ALIGNED_LANE_TERMS / PRODUCT_BIASED_EXPONENT / OPERAND_CODES_A/B per
    // lane, and its key domain enforces GROUP_MAX_BIASED_EXPONENT >= PRODUCT_BIASED_EXPONENT_i.
    // Padding rows read no operands, so their lanes must be forced to zero products —
    // nonzero biased exponents span [26, 54], so PRODUCT_BIASED_EXPONENT = 0 selects the
    // table's zero-product rows, whose tuple gives ALIGNED_LANE_TERMS = 0. MB3 then forces
    // GROUP_MAX_BIASED_EXPONENT = 0.
    for i in 0..GROUP_WIDTH {
        let c = eval.mul(lv.is_padding, lv.product_biased_exponents[i]);
        eval.constraint(c);
    }

    // ---- MB3 — anchor attainment by vanishing product (the "<=" side of
    // GROUP_MAX_BIASED_EXPONENT; the ">=" side is the LUT key domains). Chain links pin
    // MAX_EXPONENT_ATTAINMENT_16 = prod_{i<32}(GROUP_MAX_BIASED_EXPONENT - PRODUCT_BIASED_EXPONENT_i)
    // exactly (<= 2 new factors per link, degree <= 3); the closing constraint, anchored at
    // the producing row, multiplies in the zero-gated carry factor and demands the product
    // vanish. Zero lanes need no gate: attaining through the sentinel forces
    // GROUP_MAX_BIASED_EXPONENT = 0,
    // keyless for every nonzero summand — self-defeating except on all-zero rows, where 0 IS
    // the honest anchor.
    let factor = |eval: &mut E, view: &MatmulB200ColumnsView<V>, i: usize| {
        eval.sub(view.group_max_biased_exponent, view.product_biased_exponents[i])
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
    let f_last = factor(eval, lv, GROUP_WIDTH - 1);
    let link = eval.msub(
        lv.max_exponent_attainment[NUM_ATT_LINKS - 2],
        f_last,
        lv.max_exponent_attainment[NUM_ATT_LINKS - 1],
    );
    eval.constraint(link);
    // Closing: the chain includes the zero-gated incoming-carry exponent factor.
    let incoming_carry_is_live = eval.sub(one, nv.incoming_carry_is_zero);
    let incoming_carry_exponent_gap = eval.sub(nv.group_max_biased_exponent, lv.group_output_biased_exponent);
    let incoming_carry_factor = eval.mul(incoming_carry_is_live, incoming_carry_exponent_gap);
    let incoming_carry_factor = eval.add(incoming_carry_factor, nv.incoming_carry_is_zero);
    let closing = eval.mul(nv.max_exponent_attainment[NUM_ATT_LINKS - 1], incoming_carry_factor);
    eval.constraint(closing);

    // ---- MB4 — carry-zero flag by propagation. ----
    let incoming_carry_next_diff = eval.sub(nv.incoming_carry_is_zero, z);
    let c = eval.mul(not_final, incoming_carry_next_diff);
    eval.constraint(c);
    let incoming_carry_next_reset = eval.sub(nv.incoming_carry_is_zero, one);
    let c = eval.mul(lv.is_cell_final, incoming_carry_next_reset);
    eval.constraint(c);
    let first_anchor = eval.sub(lv.incoming_carry_is_zero, one);
    eval.constraint_first_row(first_anchor);
    // IS_CELL_FINAL's schedule facts — booleanness and the cell-final last row, the wrap
    // soundness of every plain transition here — are forced by the MS constraints.

    // ---- MB5 — carry alignment (anchored at the producing row). ----
    // POW2GB(GROUP_MAX_BIASED_EXPONENT' - GROUP_OUTPUT_BIASED_EXPONENT;
    // INCOMING_CARRY_SHIFT_POWER') [filter 1 - INCOMING_CARRY_IS_ZERO'] is a LUT instance.
    // Its key domain enforces GROUP_MAX_BIASED_EXPONENT' >= GROUP_OUTPUT_BIASED_EXPONENT.
    // The AIR enforces the exact floor split of 4x the previous output's significand, made
    // two-sided by the remainder identity and exact over the integers by the limb range
    // checks (quotient, remainder and bound all < 2^26; without the quotient check the field
    // would admit a fractional-quotient alias — every value here fits far below p, so ranged
    // equations over the field are equations over Z).
    let incoming_carry_is_nonzero_next = eval.sub(one, nv.incoming_carry_is_zero);
    let four = eval.i32(4);
    let four_sig = eval.mul(four, lv.group_output_significand);
    let shifted = eval.mul(aligned_incoming_carry_next, nv.incoming_carry_shift_power);
    let floor_diff = eval.sub(four_sig, shifted);
    let floor_diff = eval.sub(floor_diff, incoming_carry_remainder_next);
    let c = eval.mul(incoming_carry_is_nonzero_next, floor_diff);
    eval.constraint(c);
    // Unfiltered remainder identity bounds the remainder below the shift power.
    let incoming_carry_remainder_sum = eval.add(incoming_carry_remainder, incoming_carry_remainder_bound);
    let incoming_carry_remainder_sum = eval.add(incoming_carry_remainder_sum, one);
    let incoming_carry_remainder_identity = eval.sub(incoming_carry_remainder_sum, lv.incoming_carry_shift_power);
    eval.constraint(incoming_carry_remainder_identity);
    // Local kill: zero carries contribute nothing (the limbs are RC16-nonnegative, so the
    // weighted sum vanishes only when both do; also covers cell starts and the cyclic wrap).
    let kill = eval.mul(lv.incoming_carry_is_zero, aligned_incoming_carry);
    eval.constraint(kill);

    // ---- MB6 — signed sum (anchored at the producing row; row 0's instance rides the wrap,
    // where ALIGNED_INCOMING_CARRY(0) = 0 makes the stale output sign inert). ----
    let terms_sum = eval.sum(&nv.aligned_lane_terms);
    let carry_sign_factor = double_complement(eval, one, lv.group_output_sign);
    let signed_carry = eval.mul(carry_sign_factor, aligned_incoming_carry_next);
    let bracket = eval.add(terms_sum, signed_carry);
    let group_sum_sign_factor = double_complement(eval, one, nv.group_sum_sign);
    let recovered = eval.mul(group_sum_sign_factor, bracket);
    let sum_eq = eval.sub(nv.group_sum_abs, recovered);
    eval.constraint(sum_eq);
    // GROUP_SUM_SIGN must be a bit for the magnitude recovery to pin GROUP_SUM_ABS.
    eval.constraint_bool(lv.group_sum_sign);
    // GROUP_SUM_IS_ZERO is explicit; MB7's normalization floor excludes false nonzero claims.
    eval.constraint_bool(lv.group_sum_is_zero);
    let ban = eval.mul(lv.group_sum_is_zero, lv.group_sum_abs);
    eval.constraint(ban);
    // Truncated-to-zero sums are +0 (-0 never exits a group).
    let plus_zero = eval.mul(lv.group_sum_is_zero, lv.group_sum_sign);
    eval.constraint(plus_zero);

    // ---- MB7 — width + truncation toward zero. WIDTH32 binds GROUP_SUM_WIDTH,
    // TRUNCATION_POWER, and LIFTING_POWER. The Euclidean identity and normalized range pin the
    // exact bit width and truncation.
    let lifted = eval.mul(lv.group_sum_abs, lv.lifting_power);
    let truncated = eval.mul(normalized_group_sum_significand, lv.truncation_power);
    let width_diff = eval.sub(lifted, truncated);
    let width_diff = eval.sub(width_diff, lv.truncation_remainder);
    let c = eval.mul(one_minus_z, width_diff);
    eval.constraint(c);
    let truncation_remainder_sum = eval.add(lv.truncation_remainder, lv.truncation_remainder_bound);
    let truncation_remainder_sum = eval.add(truncation_remainder_sum, one);
    let truncation_remainder_identity = eval.sub(truncation_remainder_sum, lv.truncation_power);
    let c = eval.mul(one_minus_z, truncation_remainder_identity);
    eval.constraint(c);

    // ---- MB10 — output mux (normal path / +0 path). ----
    let group_output_biased_exponent_raw = eval.add(lv.group_max_biased_exponent, lv.group_sum_width);
    let twenty_six = eval.i32(26);
    let group_output_biased_exponent_raw = eval.sub(group_output_biased_exponent_raw, twenty_six);
    let exp_diff = eval.sub(lv.group_output_biased_exponent, group_output_biased_exponent_raw);
    let c = eval.mul(one_minus_z, exp_diff);
    eval.constraint(c);
    let sig_diff = eval.sub(lv.group_output_significand, normalized_group_sum_significand);
    let c = eval.mul(one_minus_z, sig_diff);
    eval.constraint(c);
    let sign_diff = eval.sub(lv.group_output_sign, lv.group_sum_sign);
    let c = eval.mul(one_minus_z, sign_diff);
    eval.constraint(c);
    let c = eval.mul(z, lv.group_output_biased_exponent);
    eval.constraint(c);
    let c = eval.mul(z, lv.group_output_significand);
    eval.constraint(c);
    let c = eval.mul(z, lv.group_output_sign);
    eval.constraint(c);

    // ---- MB12 — f32 encode at cell-final. ----
    // W := LO + 2^16*HI must equal GROUP_OUTPUT_SIGN*2^31
    // + (GROUP_OUTPUT_BIASED_EXPONENT + 89)*2^23 + (GROUP_OUTPUT_SIGNIFICAND - 2^23)
    // on the normal path and 0 on the zero path; folding the mux gives
    // MB10 has already zeroed the group output on the zero path.
    let w = eval.mad(lv.cell_result_f32_hi, limb_shift, lv.cell_result_f32_lo);
    let c2_31 = eval.u64(1 << 31);
    let sign_term = eval.mul(lv.group_output_sign, c2_31);
    let c2_23 = eval.u64(1 << 23);
    let exp_term = eval.mul(lv.group_output_biased_exponent, c2_23);
    let c88_2_23 = eval.u64(88 << 23);
    let residue = eval.mul(c88_2_23, one_minus_z);
    let encode = eval.sub(w, sign_term);
    let encode = eval.sub(encode, exp_term);
    let encode = eval.sub(encode, lv.group_output_significand);
    let encode = eval.sub(encode, residue);
    let c = eval.mul(lv.is_cell_final, encode);
    eval.constraint(c);

    // ---- MB13 — consolidated-policy grid exponent (stored as e_cell through Task 6).
    // Here, we only need to check that e_cell is constant within one cell computation. To check that it is an upper bound of the
    // exponents of the products and the partial products, we use CTLs.
    let e_cell_step = eval.sub(nv.e_cell, lv.e_cell);
    let c = eval.mul(not_final, e_cell_step);
    eval.constraint(c);

    // ---- MB14 — grid-nonzero flag. NZ is boolean and its complement pins E_GRID = 0. ----
    let nz = lv.cell_nonzero;
    eval.constraint_bool(nz);
    let not_nz = eval.sub(one, nz);
    let c = eval.mul(not_nz, lv.e_cell);
    eval.constraint(c);

    // ---- MB15 — consolidated-policy skip flags. Boolean; pinned to 0 on padding rows
    // and on zero grids. On a nonzero grid, lane u is
    // skipped exactly when
    //   LAMBDA_A + LAMBDA_B < 128*E_GRID + 45952.
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

    // ---- MB16 — the in-cell skip census. CELL_SKIPS anchors to the row's flag sum at each
    // cell start and otherwise accumulates the next row's flags; the cell-final value rides
    // the result channel to XorFold's budget gate. The cyclic wrap is inert: the last row
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

/// A CTL party of the fp8 batch (`requires_ctls()`): the batch driver is the only
/// supported proving path.
impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for MatmulStarkB200<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_MATMUL_B200_COLUMNS, NUM_MATMUL_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_MATMUL_B200_COLUMNS, NUM_MATMUL_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_matmul_b200_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_matmul_b200_constraints(vars, &mut evaluator);
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    // Party to the operand-codes and cell-results channels plus the committed LUT channels
    // (halves in `super::ctl`, assembled by `super::super::ctl`).
    fn requires_ctls(&self) -> bool {
        true
    }

    // (No in-trace `lookups()`: the former MB14 per-cell histogram logup was removed with the
    // jackpot policy.)
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

    use super::*;
    use crate::v4::api::utils::B200;
    use crate::v4::circuit::matmul_b200_stark::columns::MATMUL_B200_COL_MAP;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type S = MatmulStarkB200<F, D>;

    /// fp8 codes with exponent field 7 or 8 and mixed signs (never NaN); product f32
    /// exponents land in {127, 128, 129}, so with the zero run at most 4 runs per cell.
    const CODE_POOL: [u8; 8] = [0x38, 0x40, 0xB9, 0x3A, 0xC1, 0x3B, 0xBA, 0x42];

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

    /// A wide-dynamic-range pool (exponent fields 1..14, subnormals included) exercising deep
    /// rel-shifts, window drops and truncation in both width directions.
    const WIDE_POOL: [u8; 10] = [0x08, 0x78, 0xF9, 0x0F, 0x30, 0xC5, 0x58, 0x01, 0xB3, 0x62];

    fn wide_codes(len: usize, salt: u64, zero_every: usize) -> Vec<u8> {
        (0..len)
            .map(|i| {
                if zero_every != 0 && i % zero_every == 0 {
                    0x00
                } else {
                    WIDE_POOL[((i as u64).wrapping_mul(salt) ^ (i as u64 >> 2)) as usize % WIDE_POOL.len()]
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
        S::new(4, 4, 128)
    }

    fn test_trace() -> (S, Vec<[F; NUM_MATMUL_B200_COLUMNS]>, [F; NUM_MATMUL_PUBLIC_INPUTS]) {
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

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    fn assert_all_constraints(stark: S, rows: &[[F; NUM_MATMUL_B200_COLUMNS]], pis: &[F]) {
        let n = rows.len();
        for i in 0..n {
            // Plain constraints must hold on every row *including* the last -> first wrap;
            // `z_last` is zero on the wrap pair, matching starky's semantics for the MS
            // transition constraints (MS7/MS9).
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], pis);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::from_bool(i != n - 1),
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

    #[test]
    fn e_cell_is_the_binade_of_the_plaintext_replay_magnitude() {
        // Check 3 (MB13): the committed E_CELL equals floor(log2 M_ij) + 139, with M_ij the
        // replay magnitude exactly as `jackpot_policy` computes it (max |product| and
        // |partial sum| of the cell), and 0 for an all-zero cell — row 0 of A is zeroed to
        // hit that case on live cells. The trace must still satisfy every constraint; the
        // wide pool drives partials through both truncation directions.
        use crate::v4::api::dtype::fp8_e4m3_to_f32;
        for (make, salt_a, salt_b, program) in [
            (
                test_codes as fn(usize, u64, usize) -> Vec<u8>,
                0x9E3779B97F4A7C15u64,
                0xC2B2AE3D27D4EB4Fu64,
                test_program(),
            ),
            (wide_codes, 0xD1B54A32D192ED03, 0x2545F4914F6CDD1D, wide_program()),
        ] {
            let (h, w, k) = (program.h, program.w, program.k);
            let mut a = make(h * k, salt_a, 8);
            a[..k].fill(0);
            let b = make(w * k, salt_b, 11);
            let (rows, pis) = program.generate_trace(
                &a,
                &b,
                &test_lambdas(h * k, 0xA24BAED4963EE407),
                &test_lambdas(w * k, 0x9FB21C651E98DF25),
            );

            let partials = B200 {}.matmul_fp8_partials(&a, &b, h, w, k).unwrap();
            for cell in 0..h * w {
                let (r, c) = (cell / w, cell % w);
                let mut m_ij = 0f32;
                for u in 0..k {
                    m_ij = m_ij.max((fp8_e4m3_to_f32(a[r * k + u]) * fp8_e4m3_to_f32(b[c * k + u])).abs());
                }
                for &p in &partials[cell] {
                    m_ij = m_ij.max(p.abs());
                }
                // A nonzero M_ij is f32-normal (some product is nonzero, so M_ij >= 2^-18):
                // its binade is the exponent field minus the bias.
                let expected = if m_ij == 0.0 {
                    0
                } else {
                    ((m_ij.to_bits() >> 23) as i64 - 127 + 139) as u64
                };
                assert_eq!(expected == 0, r == 0, "exactly row 0's cells are all-zero (cell {cell})");
                let v: &MatmulB200ColumnsView<F> = rows[cell * program.rows_per_cell()].borrow();
                assert_eq!(to_u64(v.e_cell), expected, "cell {cell}");
                // Check 4 on the same trace: zero cells pin the nonzero flag and every skip flag.
                assert_eq!(v.cell_nonzero, F::from_bool(expected != 0), "cell {cell}");
                if expected == 0 {
                    for i in 0..GROUP_WIDTH {
                        assert_eq!(v.skip_flag[i], F::ZERO, "zero cell {cell} lane {i}");
                    }
                }
            }
            assert_all_constraints(program, &rows, &pis);
        }
    }

    /// A long-k geometry (k = 2048 gives 64 rows per cell, no head padding) paired with the
    /// wide pool's ~36 distinct f32 exponents.
    fn wide_program() -> S {
        S::new(2, 2, 2048)
    }

    #[test]
    fn wide_range_trace_satisfies_all_constraints() {
        // Wide-exponent operands push the window through deep rel-shifts (terms dropped
        // entirely), carry alignments beyond the 26-cap, and both truncation directions.
        let program = wide_program();
        let a = wide_codes(program.h * program.k, 0x9E3779B97F4A7C15, 7);
        let b = wide_codes(program.w * program.k, 0x2545F4914F6CDD1D, 13);
        let (rows, pis) = program.generate_trace(
            &a,
            &b,
            &test_lambdas(program.h * program.k, 0xA24BAED4963EE407),
            &test_lambdas(program.w * program.k, 0x9FB21C651E98DF25),
        );
        assert_all_constraints(program, &rows, &pis);
    }

    /// The bit-exactness oracle: the native `B200::matmul_fp8` (itself cross-checked against
    /// B200.md's Python simulator) consumes K in ascending 32-groups with the chained carry —
    /// exactly the row chain — so the trace's cell results must match it bit for bit on any
    /// geometry, zeros included.
    #[test]
    fn trace_results_are_bit_exact_vs_native_b200() {
        let hw = B200 {};
        for (salt_a, salt_b, zero_a, zero_b, wide) in [
            (0x9E3779B97F4A7C15u64, 0xC2B2AE3D27D4EB4Fu64, 8usize, 11usize, false),
            (0xD1B54A32D192ED03, 0x2545F4914F6CDD1D, 5, 7, true),
            (0x94D049BB133111EB, 0xBF58476D1CE4E5B9, 0, 3, true),
        ] {
            let program = if wide { wide_program() } else { S::new(4, 4, 128) };
            let (h, w, k) = (program.h, program.w, program.k);
            let make = if wide { wide_codes } else { test_codes };
            let a = make(h * k, salt_a, zero_a);
            let b = make(w * k, salt_b, zero_b);
            let expected: Vec<u32> = hw
                .matmul_fp8(&a, &b, None, h, w, k)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let (rows, _) = program.generate_trace(
                &a,
                &b,
                &test_lambdas(h * k, 0xA24BAED4963EE407),
                &test_lambdas(w * k, 0x9FB21C651E98DF25),
            );
            for (cell, chunk) in rows[..program.live_rows()].chunks(program.rows_per_cell()).enumerate() {
                let last: &MatmulB200ColumnsView<F> = chunk.last().unwrap().borrow();
                assert_eq!(last.is_cell_final, F::ONE);
                assert_eq!(last.is_padding, F::ZERO);
                let got = to_u64(last.cell_result_f32_lo) | (to_u64(last.cell_result_f32_hi) << 16);
                assert_eq!(
                    got as u32, expected[cell],
                    "cell {cell}: f32 result differs from native B200 (salts {salt_a:#x}/{salt_b:#x})"
                );
            }
        }
    }

    /// No arithmetic constraint is vacuous: bumping any load-bearing witness cell of an
    /// honest trace must violate some constraint somewhere. Columns whose integrity is the
    /// committed oracle's job (`PRODUCT_BIASED_EXPONENT`, the operand codes,
    /// `TRUNCATION_POWER`/`LIFTING_POWER`, and remainder bounds) are exercised by the
    /// LutChecker in the consistency driver
    /// instead — a tamper there shifts a lookup key/value, not an in-AIR identity.
    #[test]
    fn tampered_cells_break_constraints() {
        let (program, rows, pis) = test_trace();
        let m = &MATMUL_B200_COL_MAP;
        let stark = program;

        let violates = |rows: &[[F; NUM_MATMUL_B200_COLUMNS]]| -> bool {
            let n = rows.len();
            (0..n).any(|i| {
                let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], &pis);
                let mut consumer = ConstraintConsumer::new(
                    vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                    F::from_bool(i != n - 1),
                    F::from_bool(i == 0),
                    F::from_bool(i == n - 1),
                );
                stark.eval_packed_generic(&frame, &mut consumer);
                consumer.accumulators().into_iter().any(|acc| acc != F::ZERO)
            })
        };
        let view = |r: usize| -> &MatmulB200ColumnsView<F> { rows[r].borrow() };

        // Anchor rows: a live (group sum != 0) mid-cell row — its produced carry is consumed live
        // by the next row — and a cell-final row.
        let live = (0..rows.len() - 1)
            .find(|&r| view(r).is_cell_final == F::ZERO && view(r).group_sum_is_zero == F::ZERO)
            .expect("the fixture has live mid-cell rows");
        let final_row = (0..rows.len()).find(|&r| view(r).is_cell_final == F::ONE).unwrap();

        let bumps = [
            (
                "GROUP_MAX_BIASED_EXPONENT (MB10 raw exponent)",
                live,
                m.group_max_biased_exponent,
            ),
            (
                "MAX_EXPONENT_ATTAINMENT tail (MB3 chain)",
                live,
                m.max_exponent_attainment[NUM_ATT_LINKS - 1],
            ),
            ("ALIGNED_LANE_TERMS (MB6 sum)", live, m.aligned_lane_terms[5]),
            ("ALIGNED_INCOMING_CARRY_LO (MB5/MB6)", live + 1, m.aligned_incoming_carry_lo),
            (
                "INCOMING_CARRY_SHIFT_POWER (MB5 remainder identity)",
                live,
                m.incoming_carry_shift_power,
            ),
            ("GROUP_SUM_ABS (MB6 recovery)", live, m.group_sum_abs),
            (
                "NORMALIZED_GROUP_SUM_SIGNIFICAND_LO (MB7 Euclid / MB10 mux)",
                live,
                m.normalized_group_sum_significand_lo,
            ),
            ("TRUNCATION_REMAINDER (MB7 Euclid)", live, m.truncation_remainder),
            (
                "GROUP_OUTPUT_BIASED_EXPONENT (MB10 mux)",
                live,
                m.group_output_biased_exponent,
            ),
            ("GROUP_OUTPUT_SIGNIFICAND (MB10 mux)", live, m.group_output_significand),
            ("CELL_RESULT_F32_LO (MB12 encode)", final_row, m.cell_result_f32_lo),
            ("E_CELL (MB13 cell constancy)", live, m.e_cell),
            ("CELL_SKIPS (MB16 census)", live, m.cell_skips),
        ];
        for (what, row, col) in bumps {
            let mut tampered = rows.clone();
            tampered[row][col] += F::ONE;
            assert!(violates(&tampered), "{what}: +1 at row {row} must break a constraint");
        }

        let flips = [
            ("GROUP_SUM_IS_ZERO (MB6 zero ban)", live, m.group_sum_is_zero),
            ("GROUP_SUM_SIGN (MB6 recovery)", live, m.group_sum_sign),
            ("INCOMING_CARRY_IS_ZERO (MB4 propagation)", live + 1, m.incoming_carry_is_zero),
            ("IS_CELL_FINAL (MB4 reset)", live, m.is_cell_final),
            ("IS_PADDING (MB1 padding pin)", live, m.is_padding),
            ("CELL_NONZERO (MB14 zero pin)", live, m.cell_nonzero),
            ("SKIP_FLAG (MB16 census anchor)", live, m.skip_flag[3]),
        ];
        for (what, row, col) in flips {
            let mut tampered = rows.clone();
            tampered[row][col] = F::ONE - tampered[row][col];
            assert!(violates(&tampered), "{what}: flip at row {row} must break a constraint");
        }

        // ---- Group MS (schedule gadget) tampers. ----

        // MS3/MS4: shifting the in-cell step counter breaks the walk at the splice (and the
        // cell bound at the final row).
        let mut t = rows.clone();
        for row in t.iter_mut().skip(5) {
            row[m.cell_step] += F::ONE;
        }
        assert!(violates(&t), "MS3 cell-step shift undetected");

        // MS5: suppressing a true cell boundary is unsatisfiable (the inverse witness cannot
        // exist at the bound).
        let final_live = (0..rows.len())
            .find(|&r| view(r).is_cell_final == F::ONE && view(r).is_padding == F::ZERO)
            .unwrap();
        let mut t = rows.clone();
        t[final_live][m.is_cell_final] = F::ZERO;
        assert!(violates(&t), "MS5 cell-final suppression undetected");

        // MS7/MS8: a relocated cell walk (OUT_COL bumped from some row on) breaks the
        // in-cell freeze or the single-step boundary advance of the weighted sum.
        let mut t = rows.clone();
        for row in t.iter_mut().skip(8) {
            row[m.out_col] += F::ONE;
        }
        assert!(violates(&t), "MS7/MS8 cell-walk relocation undetected");

        // MS9: raising IS_PADDING before the onset is rejected (the flag may change only
        // entering position H*W).
        let mut t = rows.clone();
        for row in t.iter_mut().skip(rows.len() - 2) {
            row[m.is_padding] = F::ONE;
        }
        assert!(violates(&t), "MS9 early padding onset undetected");

        // MS11/MS12: relocating an operand base — pointing a cell's reads at another operand
        // row/column — breaks the first-row anchor or the K-scaled walk recurrence, whether
        // shifted from row 0 or from mid-trace.
        for (what, col) in [
            ("MS11 BASE_A", m.operand_index_base_a),
            ("MS12 BASE_B", m.operand_index_base_b),
        ] {
            let mut t = rows.clone();
            for row in t.iter_mut() {
                row[col] += F::from_canonical_usize(stark.k);
            }
            assert!(violates(&t), "{what} whole-plane relocation undetected");
            let mut t = rows.clone();
            for row in t.iter_mut().skip(6) {
                row[col] += F::from_canonical_usize(stark.k);
            }
            assert!(violates(&t), "{what} mid-trace relocation undetected");
        }

        // MS13/MS14: a dishonest coordinate *split* — every cell from cell 2 on claims
        // (OUT_ROW + 1, OUT_COL - W), with the operand bases following. This preserves
        // the weighted sum OUT_ROW*W + OUT_COL — hence the boundary advance (MS7), the
        // padding onset (MS9/MS10) and the XorFold key — plus the freeze (MS8) and the
        // base recurrences (MS11/MS12): it satisfies MS1..MS12 identically and redirects
        // every A read of those cells at the next
        // operand row. Only the explicit coordinate walk rejects it: the splice boundary
        // has OUT_COL' - OUT_COL - 1 + W*COL_CARRY != 0 (MS14), whatever carry flag and
        // inverse witness the forger supplies (MS13 leaves it no third option).
        let w_f = F::from_canonical_usize(stark.w);
        let splice = 2 * stark.rows_per_cell();
        for carry_choice in [F::ZERO, F::ONE] {
            let mut t = rows.clone();
            for row in t.iter_mut().skip(splice) {
                row[m.out_row] += F::ONE;
                row[m.out_col] -= w_f;
                row[m.operand_index_base_a] += F::from_canonical_usize(stark.k);
                row[m.operand_index_base_b] -= F::from_canonical_usize(stark.w * stark.k);
                row[m.col_carry] = carry_choice;
                let gap = row[m.out_col] - (w_f - F::ONE);
                row[m.col_carry_inv] = gap.try_inverse().unwrap_or(F::ZERO);
            }
            assert!(
                violates(&t),
                "MS13/MS14 dishonest coordinate split (carry={carry_choice}) undetected"
            );
        }
    }

    #[test]
    fn padded_trace_matches_schedule_fill() {
        // 15 cells of k/32 = 4 rows -> 60 live rows padded to 64 by 4 phantom cells; the
        // structural schedule columns must match the honest `MatmulScheduleView` on every row —
        // padding rows included, where the `(OUT_ROW, OUT_COL)` walk simply continues —
        // and the trace must satisfy the AIR (so the MS constraints pin exactly this
        // fill). The AIR acceptance is covered by `tampered_traces_fail`'s honest check on
        // the primary fixture; here we pin the padded geometry's schedule semantics.
        let program = S::new(3, 5, 128);
        let a = test_codes(program.h * program.k, 0x9E3779B97F4A7C15, 8);
        let b = test_codes(program.w * program.k, 0xC2B2AE3D27D4EB4F, 11);
        let (rows, pis) = program.generate_trace(
            &a,
            &b,
            &test_lambdas(program.h * program.k, 0xA24BAED4963EE407),
            &test_lambdas(program.w * program.k, 0x9FB21C651E98DF25),
        );
        assert_eq!(program.live_rows(), 60);
        assert_eq!(rows.len(), 64);
        assert_eq!(pis, program.public_inputs());
        let m = &MATMUL_B200_COL_MAP;
        let hw = program.h * program.w;
        for (r, row) in rows.iter().enumerate() {
            let s = MatmulScheduleView::<F>::at(r, program.rows_per_cell(), program.w, hw);
            for (what, got, want) in [
                ("IS_CELL_FINAL", row[m.is_cell_final], s.is_cell_final),
                ("IS_PADDING", row[m.is_padding], s.is_padding),
                ("CELL_STEP", row[m.cell_step], s.cell_step),
                ("CELL_STEP_INV", row[m.cell_step_inv], s.cell_step_inv),
                ("OUT_ROW", row[m.out_row], s.out_row),
                ("OUT_COL", row[m.out_col], s.out_col),
                ("PADDING_ONSET_INV", row[m.padding_onset_inv], s.padding_onset_inv),
                ("OPERAND_INDEX_BASE_A", row[m.operand_index_base_a], s.operand_index_base_a),
                ("OPERAND_INDEX_BASE_B", row[m.operand_index_base_b], s.operand_index_base_b),
                ("COL_CARRY", row[m.col_carry], s.col_carry),
                ("COL_CARRY_INV", row[m.col_carry_inv], s.col_carry_inv),
            ] {
                assert_eq!(got, want, "{what} row {r}");
            }
            // The phantom suffix: single-row cells continuing the walk at position hw + t.
            if r >= program.live_rows() {
                assert_eq!(row[m.is_padding], F::ONE, "row {r}");
                assert_eq!(
                    row[m.out_row] * F::from_canonical_usize(program.w) + row[m.out_col],
                    F::from_canonical_usize(hw + r - program.live_rows()),
                    "row {r}"
                );
            }
        }
        // The padded trace satisfies every constraint (phantom rows and the wrap included).
        assert_all_constraints(program, &rows, &pis);
    }

    #[test]
    fn degree_is_at_most_three() {
        test_stark_low_degree::<F, S, D>(test_program()).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        test_stark_circuit_constraints::<F, C, S, D>(test_program()).unwrap();
    }

    // No standalone prove/verify smoke test: this table is a CTL party (`requires_ctls`); the
    // end-to-end proving path is covered by the B200 driver test
    // (`circuit::driver::tests::b200_batch_proof_roundtrips`).
}
