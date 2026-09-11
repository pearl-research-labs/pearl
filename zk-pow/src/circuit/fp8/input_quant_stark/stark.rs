//! Input quantization, row statistics and summand scores.
//!
//! # What is proved
//!
//! Each trace row handles one A element and one B element. A group of `k` trace rows
//! represents one matrix row on each side; each block of eight elements shares a scale.
//! For each live side, the table proves:
//!
//! 1. `X = RNE_bf16(int8 * block_scale)`;
//! 2. the row's framed sum of squares and maximum decoded `|X|`;
//! 3. `noise_term = RNE_bf16(beta * n)`, where `n` is noise derived from public seeds;
//! 4. `noised = RNE_bf16(alpha*X + noise_term)`, with one rounding after the addition;
//! 5. the FP8 E4M3 cast, the dead-entry flag and its running count;
//! 6. the integer summand score used by jackpot check 4.
//!
//! `RNE_bf16` means round to nearest BF16, with ties to even. At a midpoint, choose
//! the representable value whose significand has an even least-significant bit.
//!
//! Blake3 binds the committed int8 bytes and block-scale codes. Scale verifies the
//! completed row statistics and the claims for `alpha`, `beta` and the noise scale
//! `sigma`. Matmul consumes the FP8 codes and summand scores.
//!
//! # BF16 arithmetic
//!
//! BF16 has exponent bias 127 and seven fraction bits. For a finite value `V`, let
//! `exp_is_zero` be 1 when its exponent field is zero, and 0 otherwise. Define:
//!
//! ```text
//! M(V)  = 128*(1 - exp_is_zero) + mantissa
//! E*(V) = exp + exp_is_zero
//! V     = (-1)^sign * M(V) * 2^(E*(V) - 134)
//! ```
//!
//! Here `134 = 127 + 7`. Using `E* = 1` for exponent field zero makes the formula
//! valid for subnormals and zero as well as normal values. An exact product has
//! integer significand `M(a)*M(b)` and binary scale `2^(E*(a) + E*(b) - 268)`.
//!
//! The constraints are equations in the Goldilocks field, whose modulus is
//! `p = 2^64 - 2^32 + 1`. The bounds beside each constraint keep the relevant
//! integer differences within `(-p, p)`, so equality modulo `p` implies equality
//! over the integers.
//!
//! # Framed row statistics
//!
//! L2 denotes the row's root mean square (RMS). Its sum of squares uses the raw
//! int8 values and block scales, before per-element BF16 rounding. For block `b`:
//!
//! ```text
//! n_b = sum_{j in block b} int8_j^2
//! p_b = M(scale_b)^2 * n_b
//! ```
//!
//! The exact block contribution is `p_b * 2^(2*E*(scale_b) - 268)`. These exponents
//! can be too far apart to add the contributions in one field element. Instead,
//! all blocks use a common frame `F`: the maximum `2*E*(scale_b)` over blocks with
//! `p_b != 0`, or zero if every block is zero.
//!
//! `Wl2` controls how many low bits are retained when a block is shifted into that frame:
//!
//! ```text
//! Wl2     = 27 - ceil(log2(k))
//! shift_b = F - 2*E*(scale_b)                       for p_b != 0
//! term_b  = floor(p_b * 2^Wl2 / 2^shift_b)          for p_b != 0; otherwise 0
//! S       = sum_b term_b
//! sum_sq  = S * 2^(F - 268 - Wl2)
//! ```
//!
//! Each floor discards less than one frame unit, `2^(F - 268 - Wl2)`. This defines
//! the protocol's sum of squares; summing the rounded `X_j^2` would give a different
//! quantity. Scale divides `sum_sq` by `k` and checks its rounded square root.
//! The independent running maximum tracks decoded `|X|` for the infinity norm, `linf`.
//!
//! For supported `2048 <= k <= 2^16`, `11 <= Wl2 <= 16`. Honest blocks satisfy
//! `n_b <= 8*128^2 = 2^17` and `p_b < 2^33`; the gadget reserves the wider bound
//! `p_b < 2^37`. With `k/8` blocks, the choice of 27 keeps `S < 2^61` and
//! `p_b*2^Wl2 < 2^53`. These fit inside Scale's `2^62` sum bound and the division
//! gadget's `2^54` dividend bound.
//!
//! # Fused multiply-add (FMA)
//!
//! The product `alpha*X` stays exact until the final BF16 rounding. The addend
//! `noise_term` has already been rounded by the preceding multiplication. Write:
//!
//! ```text
//! alpha*X    = (-1)^S_P * M_P * 2^EP,    M_P = M(alpha)*M(X) < 2^16
//! noise_term = (-1)^S_C * M_C * 2^EC,    M_C = M(noise_term) < 2^8
//! EP         = E*(alpha) + E*(X) - 268
//! EC         = E*(noise_term) - 134
//! ```
//!
//! `S_P` and `S_C` are the sign bits. The gadget aligns the integer significands
//! to a common binary scale, then adds or subtracts them. Rounding `alpha*X`
//! before this addition would change the operation, especially near cancellation.
//!
//! Large exponent gaps are capped once the smaller operand can affect only rounding.
//! The threshold is 10 when the product has the coarser binary unit, and 20 when
//! the addend does. The latter covers subtraction across a binade boundary
//! (a power of two). W3 derives these bounds; W10 adjusts the common scale to
//! preserve the dominant operand.
//!
//! RNERND is the lookup table that performs the final BF16 rounding. It accepts an
//! integer significand below `2^17`. A wider aligned magnitude `T` is first
//! compressed to 17 bits using round-to-odd:
//!
//! ```text
//! T   = q*2^c + r,    0 <= r < 2^c
//! key = q + [r != 0]*(1 - (q mod 2))
//! ```
//!
//! Here `c` is the compression shift, and brackets denote 0/1 indicators. An exact
//! quotient is unchanged; an inexact quotient is made odd. The key's binary scale
//! increases by `c`. Keeping 17 bits before rounding to BF16's eight precision bits
//! preserves the final round-to-nearest, ties-to-even result.
//!
//! The rounding slot is `clamp(-133 - key_scale, -7, 18) + 7`. Its signed cuts
//! distinguish normal and subnormal results even after cancellation leaves a small
//! significand. For `X = 0`, an artificial exponent drop of 400 selects the path
//! that preserves the addend; 400 exceeds the maximum legal operand-scale gap of 386.
//!
//! # Summand scores
//!
//! For jackpot check 4, each element scores the quantity
//! `S_X = (alpha*X)^2 + sigma^2`, where `sigma = alpha*l2f/2` and `l2f` is the
//! grid-rounded, floored RMS. The products in `S_X` are exact.
//!
//! Group V normalizes both magnitudes to 16-bit significands. It aligns their
//! squares, takes an integer logarithm and exports `summand_score` to Matmul.
//! The score satisfies:
//!
//! ```text
//! 64*(log2(S_X) + 508) - 1.1 < summand_score <= 64*(log2(S_X) + 508)
//! ```
//!
//! [`super::super::unpredictability`] defines the integer rule and derives this
//! error bound. A zero `X` contributes only the sigma term. Padding scores are zero
//! and are excluded from the operand-code channel.
//!
//! # Trace binding
//!
//! The height is `next_power_of_two(max(h, w)*k)`. A is live on `[0, h*k)` and B
//! on `[0, w*k)`; each side has a valid padding witness beyond its own prefix.
//! The final trace row closes a group, including a truncated padding group.
//!
//! The verifier recomputes the noise and schedule columns and checks their openings.
//! Their boolean and boundary properties follow from that equality. Lookup tables
//! bind range, decode and rounding outputs; free witness flags are constrained
//! boolean here. [`super::ctl`] declares the lookups and cross-table channels.

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

use super::columns::{
    Bf16FieldsView, BlockL2Columns, FmaBlockView, InputQuantColumnsView, MulBlockView, MulDecodeBlockView,
    NUM_INPUT_QUANT_COLUMNS, NUM_INPUT_QUANT_KNOWN_COLUMNS, NUM_INPUT_QUANT_PUBLIC_INPUTS, WL2_POW_PUBLIC_INPUT,
};
use crate::api::fp8::compute::{bf16_clamp_sym, bf16_fma, bf16_max, bf16_mul};
use crate::api::fp8::dtype::{bf16_to_f32, f32_to_fp8_e4m3};
use crate::api::fp8::prequant::BLOCK_SIZE;
use crate::api::fp8::quantization::{DELTA, Fp8E4M3Quant, Quant};
use crate::circuit::fp8::scale_stark::stark::{CUT_BIAS, NORM_FLOOR_CODE, SLOT_MAX, rne_sqrt_hat, rnernd_reference};
use crate::circuit::fp8::unpredictability::{LambdaWitness, lambda_witness};
use crate::circuit::utils::evaluator::Evaluator;
use crate::circuit::utils::native_evaluator::NativeEvaluator;
use crate::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// The bf16 M-form bias: `value = M * 2^(E* - 134)`, so a product's LSB scale is
/// `E*(a) + E*(b) - 268`.
const MUL_LSB_OFFSET: i64 = 268;
/// bf16 subnormal grid LSB scale (`2^-133`); the signed fade cut is
/// `clamp(-133 - LSB_SCALE, -7, 18)` and the RNERND slot adds [`CUT_BIAS`].
const SUBNORMAL_LSB: i64 = -133;
/// Artificial product-scale drop used when `X = ±0`. Zero has no meaningful exponent; moving
/// it below every legal addend scale makes the FMA preserve the addend. The largest possible
/// gap is `120 - (-266) = 386`, and 400 also keeps the resulting lookup keys in range.
const ZERO_PRODUCT_SCALE_DROP: i64 = 400;
/// The wide-path width: folds of 17+ bits are round-to-odd compressed before RNERND.
const WIDE_WIDTH: u64 = 17;
/// bf16 code of 448 (the fp8 E4M3 clamp bound; `fp8_to_bf16(0x7E)`).
const BF16_CODE_448: u16 = 0x43E0;
/// Base of the protocol formula `Wl2 = 27 - ceil(log2 k)`. Under the reserved
/// `p_b < 2^37` bound, this keeps the framed sum below `2^61` and each shifted block product
/// below `2^53`. Changing 27 would change the per-block floors and therefore protocol output.
const WL2_BASE: u32 = 27;
/// Far-shift threshold: a block with `sigma >= 54` floors to `TERM = 0` exactly,
/// because `block_l2_product * 2^Wl2 < 2^54`.
const FAR_SIGMA: u64 = 54;

// Program and trace generation

/// The public geometry of one FP8 input-quantization run.
///
/// Noise decode fields, schedule flags, and indices are verifier-known from this program and
/// the public seeds. The strip bytes and scales remain private witness data.
#[derive(Clone, Debug)]
pub struct InputQuantProgram {
    /// A-strip opened rows.
    pub h: usize,
    /// B-strip opened columns. May differ from `h`: each side has its own liveness flag
    /// (`A_LIVE`/`B_LIVE`) and the shorter side is phantom-filled past its end.
    pub w: usize,
    /// Elements per matrix row; the protocol precondition is `k ≡ 0 (mod 32)`.
    pub k: usize,
    /// Scale-block size of the `int8 blk8 bf16s` format; must equal
    /// [`BLOCK_SIZE`] (= 8).
    pub block_size: usize,
    /// Noise rank `r`, needed to derive the honest alpha/beta witness
    /// (`derive_row_scales` on the `2^-32`-floored norms, as `noisy_quantize`).
    pub r: usize,
}

impl InputQuantProgram {
    /// Rows carrying at least one live element: A occupies `[0, h*k)`, B occupies `[0, w*k)`.
    pub fn live_rows(&self) -> usize {
        self.h.max(self.w) * self.k
    }

    /// Trace height: one row per element pair, grouped into matrix rows of `k` consecutive
    /// trace rows and padded to the next power of two with both-sides-dead phantom rows.
    pub fn num_rows(&self) -> usize {
        self.live_rows().next_power_of_two()
    }

    /// `Wl2 = 27 - ceil(log2 k)`: the job's block-L2 frame width (in `[11, 16]` over the
    /// envelope `2048 <= k <= 2^16`).
    pub fn wl2(&self) -> u32 {
        WL2_BASE - ceil_log2(self.k)
    }

    /// Generates the InputQuantStark trace and public inputs.
    ///
    /// - `a_int8` / `b_int8`: the int8 values planes, row-major (`h*k` resp. `w*k` values; the
    ///   B strip's w opened columns are its "rows" here, matching `B200::matmul_fp8`'s `b`).
    /// - `a_scales` / `b_scales`: the bf16 block-scale planes (`h*k/8` resp. `w*k/8` codes).
    /// - `a_noise` / `b_noise`: bf16 noise codes `(E @ F)[element]`, derived from the public
    ///   seeds and independently recomputed by the verifier.
    ///
    /// Panics if the geometry is unsupported or a bf16 operation overflows.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_trace<F: RichField>(
        &self,
        a_int8: &[i8],
        a_scales: &[u16],
        a_noise: &[u16],
        b_int8: &[i8],
        b_scales: &[u16],
        b_noise: &[u16],
    ) -> (Vec<[F; NUM_INPUT_QUANT_COLUMNS]>, [F; NUM_INPUT_QUANT_PUBLIC_INPUTS]) {
        let (h, w, k) = (self.h, self.w, self.k);
        assert_eq!(
            self.block_size, BLOCK_SIZE,
            "only the whitelisted int8 blk8 bf16s format is supported"
        );
        assert_eq!(k % 32, 0, "protocol precondition: k ≡ 0 (mod 32)");
        let num_rows = self.num_rows();
        assert_eq!(a_int8.len(), h * k, "a_int8 must be h*k row-major int8 values");
        assert_eq!(b_int8.len(), w * k, "b_int8 must be w*k row-major int8 values");
        assert_eq!(a_scales.len(), h * k / BLOCK_SIZE, "a_scales must be h*k/8 bf16 codes");
        assert_eq!(b_scales.len(), w * k / BLOCK_SIZE, "b_scales must be w*k/8 bf16 codes");
        assert_eq!(a_noise.len(), h * k, "a_noise must be h*k bf16 codes");
        assert_eq!(b_noise.len(), w * k, "b_noise must be w*k bf16 codes");

        // Dead-side phantom inputs use zero bytes, scales, and noise. The trailing dead group
        // may be truncated (`k` need not divide the power-of-two height); its length is still a
        // multiple of BLOCK_SIZE.
        let zero_int8 = vec![0i8; k];
        let zero_scales = vec![0u16; k / BLOCK_SIZE];
        let zero_noise = vec![0u16; k];

        let mut rows: Vec<[F; NUM_INPUT_QUANT_COLUMNS]> = Vec::with_capacity(num_rows);
        let mut start = 0;
        while start < num_rows {
            let g = start / k;
            let len = k.min(num_rows - start);
            let blocks = k / BLOCK_SIZE;
            let (a_live, b_live) = (g < h, g < w);
            let side_a = if a_live {
                SideWitness::build(
                    self,
                    &a_int8[g * k..(g + 1) * k],
                    &a_scales[g * blocks..(g + 1) * blocks],
                    &a_noise[g * k..(g + 1) * k],
                    true,
                )
            } else {
                SideWitness::build(
                    self,
                    &zero_int8[..len],
                    &zero_scales[..len / BLOCK_SIZE],
                    &zero_noise[..len],
                    false,
                )
            };
            let side_b = if b_live {
                SideWitness::build(
                    self,
                    &b_int8[g * k..(g + 1) * k],
                    &b_scales[g * blocks..(g + 1) * blocks],
                    &b_noise[g * k..(g + 1) * k],
                    true,
                )
            } else {
                SideWitness::build(
                    self,
                    &zero_int8[..len],
                    &zero_scales[..len / BLOCK_SIZE],
                    &zero_noise[..len],
                    false,
                )
            };
            for j in 0..len {
                let mut row = InputQuantColumnsView::<F>::default();
                let element_index = start + j;
                row.is_group_final = F::from_bool(j == len - 1);
                row.element_index = F::from_canonical_usize(element_index);
                row.is_even_row = F::from_bool(element_index % 2 == 0);
                row.is_block_start = F::from_bool(element_index % BLOCK_SIZE == 0);
                row.block_index = F::from_canonical_usize(element_index / BLOCK_SIZE);
                row.a_live = F::from_bool(a_live);
                row.b_live = F::from_bool(b_live);
                side_a.fill_a(&mut row, j);
                side_b.fill_b(&mut row, j);
                rows.push(row.into());
            }
            start += len;
        }
        blind_trace(&mut rows);
        (rows, self.public_inputs())
    }

    /// The table's public inputs (`columns.rs` slot order): `2^Wl2` and the geometry scalars
    /// `[h*k, h*k/8, w, h]` the CTL column expressions read.
    pub fn public_inputs<F: RichField>(&self) -> [F; NUM_INPUT_QUANT_PUBLIC_INPUTS] {
        [
            F::from_canonical_u64(1u64 << self.wl2()),
            F::from_canonical_usize(self.h * self.k),
            F::from_canonical_usize(self.h * self.k / BLOCK_SIZE),
            F::from_canonical_usize(self.w),
            F::from_canonical_usize(self.h),
        ]
    }

    /// Values of the leading [`NUM_INPUT_QUANT_KNOWN_COLUMNS`] verifier-known columns. They
    /// contain both sides' public-seed noise fields (or +0 on dead rows), plus geometry-derived
    /// indices, schedule flags, and liveness. The batch verifier recomputes these values and
    /// checks that the trace openings match them exactly.
    pub fn known_values<F: RichField>(&self, a_noise: &[u16], b_noise: &[u16]) -> Vec<PolynomialValues<F>> {
        let (h, w, k, num_rows) = (self.h, self.w, self.k, self.num_rows());
        assert_eq!(a_noise.len(), h * k, "a_noise must be h*k codes");
        assert_eq!(b_noise.len(), w * k, "b_noise must be w*k codes");

        let mut cols: Vec<Vec<F>> = (0..NUM_INPUT_QUANT_KNOWN_COLUMNS)
            .map(|_| Vec::with_capacity(num_rows))
            .collect();
        for r in 0..num_rows {
            for (base, codes) in [(0, a_noise), (4, b_noise)] {
                // Dead rows carry the phantom +0 noise fields.
                let f = if r < codes.len() {
                    Bf16Fields::from_code(codes[r])
                } else {
                    Bf16Fields::default()
                };
                cols[base].push(F::from_bool(f.sign));
                cols[base + 1].push(F::from_canonical_u64(f.exp));
                cols[base + 2].push(F::from_canonical_u64(f.mantissa));
                cols[base + 3].push(F::from_bool(f.exp_is_zero()));
            }
            // Period-k group finals, plus the trace's very last row (the trailing dead group
            // may be truncated when k is not a power of two).
            cols[8].push(F::from_bool(r % k == k - 1 || r == num_rows - 1));
            cols[9].push(F::from_canonical_usize(r));
            cols[10].push(F::from_bool(r % 2 == 0));
            cols[11].push(F::from_bool(r % BLOCK_SIZE == 0));
            cols[12].push(F::from_canonical_usize(r / BLOCK_SIZE));
            cols[13].push(F::from_bool(r < h * k));
            cols[14].push(F::from_bool(r < w * k));
        }
        cols.into_iter().map(PolynomialValues::new).collect()
    }
}

/// Randomizes four unused row-0 cells to blind the trace commitment.
///
/// The wrapper publishes the Fiat–Shamir challenge `zeta`, which is derived from
/// the commitment. Without committed randomness, it could fingerprint a low-entropy
/// witness. Four random field elements supply about 256 bits of entropy.
///
/// Row 0 is not block-final. Every constraint or lookup reading these cells is
/// disabled there, so the randomness leaves the computation unchanged.
fn blind_trace<F: RichField>(rows: &mut [[F; NUM_INPUT_QUANT_COLUMNS]]) {
    let row: &mut InputQuantColumnsView<F> = rows[0].borrow_mut();
    row.block_l2_a.shift_remainder_power = F::rand();
    row.block_l2_a.shift_complement_power = F::rand();
    row.block_l2_b.shift_remainder_power = F::rand();
    row.block_l2_b.shift_complement_power = F::rand();
}

/// `ceil(log2 n)` for `n >= 1`.
fn ceil_log2(n: usize) -> u32 {
    n.next_power_of_two().trailing_zeros()
}

/// `bit_length(x)` for `x > 0` (0 for 0): the position of the leading 1.
fn bit_len(x: u64) -> u64 {
    (64 - x.leading_zeros()) as u64
}

/// Embeds a (possibly negative) i64 into the field (`p - |v|` for negatives).
fn field_i64<F: RichField>(v: i64) -> F {
    if v >= 0 {
        F::from_canonical_u64(v as u64)
    } else {
        -F::from_canonical_u64(v.unsigned_abs())
    }
}

// bf16 field decode and the shared LUT mirrors (trace-generation side)

/// A bf16 value's decode fields (trace-gen twin of [`Bf16FieldsView`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Bf16Fields {
    sign: bool,
    exp: u64,
    mantissa: u64,
}

impl Bf16Fields {
    /// Splits a finite bf16 code into fields. Panics on inf/NaN codes (exponent field 255 has
    /// no EXPINFO row — banned everywhere).
    fn from_code(code: u16) -> Self {
        let exp = u64::from((code >> 7) & 0xFF);
        assert_ne!(exp, 255, "non-finite bf16 code {code:#06x} is banned");
        Self {
            sign: code >> 15 == 1,
            exp,
            mantissa: u64::from(code & 0x7F),
        }
    }

    fn code(&self) -> u16 {
        ((self.sign as u16) << 15) | ((self.exp as u16) << 7) | self.mantissa as u16
    }

    fn exp_is_zero(&self) -> bool {
        self.exp == 0
    }

    /// `M(V) = (1 - EXP_IS_ZERO)*128 + MANTISSA`: the significand with the implicit bit resolved.
    fn m(&self) -> u64 {
        (if self.exp == 0 { 0 } else { 128 }) + self.mantissa
    }

    /// `E*(V) = EXP + EXP_IS_ZERO`: the effective exponent.
    fn e_star(&self) -> i64 {
        self.exp as i64 + i64::from(self.exp == 0)
    }

    /// `ABS(V) = EXP*2^7 + MANTISSA`: the sign-stripped code (monotone in magnitude).
    fn abs_code(&self) -> u64 {
        self.exp * 128 + self.mantissa
    }
}

/// The INT8DEC mirror: exact bf16 decode fields of a raw two's-complement byte (byte 0 -> +0,
/// byte 0x80 = -128 -> `(1, 134, 0)`). Exact because every int8 fits 8 significand bits.
fn int8dec(byte: u8) -> Bf16Fields {
    let v = byte as i8;
    if v == 0 {
        return Bf16Fields::default();
    }
    let mag = (v as i64).unsigned_abs(); // in [1, 128]
    let w = bit_len(mag);
    Bf16Fields {
        sign: v < 0,
        exp: 126 + w,
        mantissa: (mag << (8 - w)) - 128,
    }
}

/// The CLAMP22 mirror: the RNERND slot `clamp(x, -7, 18) + 7` for the signed fade argument `x`
/// (the table is keyed `x + 400`, domain `x in [-400, 200]`).
fn clamp_slot(x: i64) -> u64 {
    (x + CUT_BIAS).clamp(0, SLOT_MAX) as u64
}

/// Computes `floor(p * 2^Wl2 / 2^sigma)`. Since the dividend is below `2^54`, the AIR's
/// `sigma >= 54` far branch is exactly zero; the `sigma >= 64` branch here also avoids an
/// invalid native shift.
pub(crate) fn block_l2_term(p: u64, sigma: u64, wl2: u32) -> u64 {
    let x = p << wl2; // p < 2^37 envelope, Wl2 <= 16: < 2^53
    if sigma >= 64 { 0 } else { x >> sigma }
}

/// A rounded bf16 output block (the RNERND value tuple plus the exponent the consuming
/// constraint reconstructs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RoundedBf16 {
    /// Output exponent field (0 on the zero and subnormal branches, per U5/M5/W12).
    pub(crate) exp: u64,
    /// Output mantissa field (RNERND value).
    pub(crate) mantissa: u64,
    /// Output is ±0 (RNERND value).
    pub(crate) is_zero: bool,
    /// Output exponent field is zero (RNERND value).
    pub(crate) exp_is_zero: bool,
    /// RNERND's exponent adjustment: `final rounding pos + 134` on normal outputs and zero
    /// otherwise. A normal result satisfies `out_exp = lsb_scale + width_adjust`.
    pub(crate) width_adjust: u64,
}

impl RoundedBf16 {
    fn fields(&self, sign: bool) -> Bf16Fields {
        Bf16Fields {
            sign,
            exp: self.exp,
            mantissa: self.mantissa,
        }
    }
}

/// Round `v * 2^lsb_scale` to BF16 using the RNERND table's reference calculation.
///
/// `v` must fit 17 bits. `slot` selects the rounding position for this scale,
/// clamped so subnormal results land on BF16's `2^-133` grid.
/// Panics on overflow, matching the native arithmetic's non-finite error.
pub(crate) fn rnernd_scaled(v: u64, slot: u64, lsb_scale: i64) -> RoundedBf16 {
    debug_assert!(v < 1 << WIDE_WIDTH, "RNERND significands are pre-compressed to 17 bits");
    debug_assert_eq!(
        slot,
        clamp_slot(SUBNORMAL_LSB - lsb_scale),
        "slot must be the CLAMP22 of the LSB scale"
    );
    let (mantissa, width_adjust, is_zero, exp_is_zero) = rnernd_reference(v, slot);
    // Reconstruct the exponent exactly as the consuming U4/M4/W12 constraints do.
    let exp = if exp_is_zero {
        0
    } else {
        let e = lsb_scale + width_adjust as i64;
        assert!(e >= 1, "normal RNERND output below the bf16 normal range");
        assert!(e <= 254, "bf16 overflow: the native path errors and the AIR is unsatisfiable");
        e as u64
    };
    RoundedBf16 {
        exp,
        mantissa,
        is_zero,
        exp_is_zero,
        width_adjust,
    }
}

/// One bf16 multiply gadget instance (the M1-M6 pattern): exact significand product, CLAMP22
/// cut, RNERND round. Mirrors `bf16_mul` bit-for-bit (asserted in tests over wide sweeps).
#[derive(Clone, Copy, Debug, Default)]
struct MulWitness {
    sig_product: u64,
    cut_depth: u64,
    out: RoundedBf16,
}

fn mul_gadget(a: Bf16Fields, b: Bf16Fields) -> MulWitness {
    let sig_product = a.m() * b.m();
    let cut_depth = clamp_slot(135 - a.e_star() - b.e_star());
    let out = rnernd_scaled(sig_product, cut_depth, a.e_star() + b.e_star() - MUL_LSB_OFFSET);
    MulWitness {
        sig_product,
        cut_depth,
        out,
    }
}

/// Witness values for one fused multiply-add. Magnitudes are stored as nonnegative integers;
/// sign bits are applied by the AIR when it reconstructs the signed sum.
#[derive(Clone, Copy, Debug, Default)]
struct FmaWitness {
    product_scale_ge_addend: bool,
    scale_gap_slack: u64,
    is_far_gap: bool,
    far_gap_slack: u64,
    exp_gap_capped: u64,
    exp_gap_pow2: u64,
    aligned_product: u64,
    aligned_addend: u64,
    folded_magnitude: u64,
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
    out_sign: bool,
    out: RoundedBf16,
}

/// Computes the same single-rounding result as native `bf16_fma`:
/// `RNE_bf16((-1)^s_p*m_p*2^ep_raw + (-1)^s_c*m_c*2^ec)`.
///
/// It aligns both integer significands, forms their exact signed sum, compresses a wide sum to
/// 17 bits with round-to-odd, and invokes RNERND once. When the product is zero, its exponent is
/// meaningless; `product_is_zero` selects the artificial low scale that preserves the addend.
fn fma_gadget(m_p: u64, ep_raw: i64, product_is_zero: bool, s_p: bool, m_c: u64, ec: i64, s_c: bool) -> FmaWitness {
    debug_assert_eq!(product_is_zero, m_p == 0);
    let ep = ep_raw - if product_is_zero { ZERO_PRODUCT_SCALE_DROP } else { 0 };
    let product_scale_ge_addend = ep >= ec;
    let d = if product_scale_ge_addend { ep - ec } else { ec - ep } as u64;
    let scale_gap_slack = if product_scale_ge_addend { d } else { d - 1 };
    // The far caps are one below their thresholds. Cap 9 gives
    // M_C/2^9 < 2^8/2^9 = 1/2 product unit. In the reverse direction, cap 19 gives
    // M_P/2^19 < 2^16/2^19 = 1/8 addend unit, below the worst 1/4-unit bf16 midpoint.
    let thr = if product_scale_ge_addend { 10 } else { 20 };
    let is_far_gap = d >= thr;
    let far_gap_slack = if is_far_gap { d - thr } else { 0 };
    let exp_gap_capped = if is_far_gap { thr - 1 } else { d };
    let exp_gap_pow2 = 1u64 << exp_gap_capped;
    let (aligned_product, aligned_addend) = if product_scale_ge_addend {
        (m_p * exp_gap_pow2, m_c)
    } else {
        (m_p, m_c * exp_gap_pow2)
    };
    let signed = (if s_p {
        -(aligned_product as i64)
    } else {
        aligned_product as i64
    }) + (if s_c {
        -(aligned_addend as i64)
    } else {
        aligned_addend as i64
    });
    let folded_magnitude = signed.unsigned_abs();
    // W7: exact zeros take the IEEE ±0 rule (cancellation -> +0; both-zero -> -0 only when both
    // operands are -0); nonzero folds keep the fold sign (including values that *round* to zero).
    let out_sign = if folded_magnitude == 0 { s_p && s_c } else { signed < 0 };
    let is_wide = folded_magnitude >= 1 << WIDE_WIDTH;
    let (compression_shift, compression_quotient, compression_remainder) = if is_wide {
        let compression_shift = bit_len(folded_magnitude) - WIDE_WIDTH;
        (
            compression_shift,
            folded_magnitude >> compression_shift,
            folded_magnitude & ((1u64 << compression_shift) - 1),
        )
    } else {
        (0, 0, 0)
    };
    let shift_pow2 = 1u64 << compression_shift;
    let compression_quotient_lsb = compression_quotient & 1;
    let sticky = compression_remainder != 0;
    // Round-to-odd: bump an even head iff the remainder is nonzero (Boldo-Melquiond, 17 >= 8+2).
    let rounding_significand_key = if is_wide {
        compression_quotient + u64::from(sticky && compression_quotient_lsb == 0)
    } else {
        folded_magnitude
    };
    let key_scale = if product_scale_ge_addend { ec } else { ep }
        + if is_far_gap { (d - exp_gap_capped) as i64 } else { 0 }
        + if is_wide { compression_shift as i64 } else { 0 };
    let cut_used = clamp_slot(SUBNORMAL_LSB - key_scale);
    let out = rnernd_scaled(rounding_significand_key, cut_used, key_scale);
    FmaWitness {
        product_scale_ge_addend,
        scale_gap_slack,
        is_far_gap,
        far_gap_slack,
        exp_gap_capped,
        exp_gap_pow2,
        aligned_product,
        aligned_addend,
        folded_magnitude,
        is_wide,
        compression_shift,
        shift_pow2,
        compression_quotient,
        compression_quotient_lsb,
        compression_remainder,
        sticky,
        rounding_significand_key,
        key_scale,
        cut_used,
        out_sign,
        out,
    }
}

/// The QCAST mirror: `f32_to_fp8_e4m3(bf16_to_f32(clamp_±448(code)))` — cast and clamp, one
/// table (equals the native clamp-then-cast composition exactly). Panics on non-finite codes
/// (the table generator sentinels those keys).
pub(crate) fn qcast(code: u16) -> u8 {
    f32_to_fp8_e4m3(bf16_to_f32(bf16_clamp_sym(code, BF16_CODE_448))).expect("clamped bf16 is finite")
}

// The protocol L2 (integer frame sum -> RNE sqrt -> grid snap) and the group scale witness

/// The native `round_l2_to_grid` (private in `api::fp8::quantization`, mirrored here): round a
/// nonnegative bf16 to the nearest multiple of 4 ulps, ties up, by code arithmetic.
fn round_l2_to_grid(l2: u16) -> u16 {
    l2.wrapping_add(2) & !3
}

/// One group's (matrix row's) scale witness: the block frame, the alpha/beta the noising
/// uses, and the jackpot liveness bound (all CTL-bound to ScaleStark's derivation).
#[derive(Clone, Debug)]
struct GroupScales {
    /// The row's group-B doubled scale exponent frame; define
    /// `F = frame_doubled_scale_exponent = max_live 2*E*(scale_b)`.
    frame_doubled_scale_exponent: u64,
    alpha: Bf16Fields,
    beta: Bf16Fields,
    alpha_code: u16,
    beta_code: u16,
    /// The liveness threshold `code(l2f) + 256`: the bf16 code of the plaintext dead bound
    /// `tau_idle * DELTA * l2f = 4 * l2f` (`l2f` is positive normal, so +256 raises its
    /// exponent field by exactly 2).
    dead_bound: u64,
    /// The group's noise-std encoding `enc(sigma) = e(sigma) + 268` for
    /// `sigma = DELTA * alpha * l2f` (positive on live groups; 0 on dead groups).
    sigma_biased_exponent: u64,
    /// The noise-std normalized significand: `sigma = normalized_sigma_significand * 2^(sigma_biased_exponent - 268 - 15)`
    /// with `normalized_sigma_significand in [2^15, 2^16)` on live groups; 0 on dead groups.
    normalized_sigma_significand: u64,
}

/// Witness for [`BlockL2Columns`]; fields use the same conventions as the trace.
#[derive(Clone, Copy, Debug, Default)]
struct BlockL2Witness {
    block_int_squared_sum: u64,
    block_l2_product: u64,
    block_doubled_scale_exponent: u64,
    running_max_doubled_scale_exponent: u64,
    scaled_block_product_limbs: [u64; 4],
    shift_remainder_power: u64,
    shift_complement_power: u64,
    shift_limb_selector: [bool; 4],
    selected_limb_quotient: u64,
    selected_limb_remainder: u64,
    is_far_shift: bool,
    running_l2_frame_sum: u64,
}

/// One decoded element with its full per-row witness.
#[derive(Clone, Copy, Debug)]
struct ElementWitness {
    byte: u8,
    int_f: Bf16Fields,
    scale_f: Bf16Fields,
    dec: MulWitness,
    x_sign: bool,
    blk: BlockL2Witness,
    max_abs: u64,
    noise_f: Bf16Fields,
    beta_mul: MulWitness,
    fma: FmaWitness,
    code_noised: u8,
    is_dead: bool,
    dead_count: u64,
    /// Bit length of the scaled significand product `M(alpha)*M(X)`.
    scaled_product_width: u64,
    /// The scaled element's exponent encoding `enc(alpha*x)`, or 0 when `X = 0`.
    scaled_biased_exponent: u64,
    /// The scaled element's normalized significand `M(alpha)*M(X) * 2^(16 - scaled_product_width)`,
    /// in `[2^15, 2^16)`, or 0 when `X = 0`.
    scaled_significand: u64,
    /// The normalizing power `2^(16 - scaled_product_width)`.
    scaled_normalization_power: u64,
    /// The summand's score witness (jackpot check 4), ending in the lambda value.
    score: LambdaWitness,
}

/// One side's full group witness (k elements + the group scales).
struct SideWitness {
    elements: Vec<ElementWitness>,
    scales: GroupScales,
}

impl SideWitness {
    /// Decodes one matrix row of one side and computes every witness column. `live = false`
    /// builds a dead group's phantom fill: the inputs must be all-zero and the group scales are
    /// the fixed phantom pair `alpha = 2^-126`, `beta = +0` (no native derivation — a dead
    /// group opens no prequant row). The group length is `int8.len()` — `k` for live groups;
    /// the trailing dead group may be shorter (still a multiple of `BLOCK_SIZE`).
    fn build(program: &InputQuantProgram, int8: &[i8], scales: &[u16], noise: &[u16], live: bool) -> Self {
        let k = int8.len();
        debug_assert!(k.is_multiple_of(BLOCK_SIZE), "group length must be whole blocks");
        debug_assert!(!live || k == program.k, "live groups are always full-length");
        debug_assert!(live || int8.iter().all(|&v| v == 0), "dead groups take zero inputs");
        let wl2 = program.wl2();

        // Pass 1: decode X = RNE_bf16(int8 * scale) per element (groups P/U).
        let mut xs: Vec<(u8, Bf16Fields, Bf16Fields, MulWitness, bool)> = Vec::with_capacity(k);
        for j in 0..k {
            let byte = int8[j] as u8;
            let int_f = int8dec(byte);
            let scale_f = Bf16Fields::from_code(scales[j / BLOCK_SIZE]);
            let dec = mul_gadget(int_f, scale_f);
            let x_sign = int_f.sign ^ scale_f.sign; // P4: the IEEE multiply sign rule.
            debug_assert_eq!(
                dec.out.fields(x_sign).code(),
                bf16_mul(int_f.code(), scale_f.code()).expect("decode product is finite"),
                "decode MUL mirror diverged from bf16_mul"
            );
            xs.push((byte, int_f, scale_f, dec, x_sign));
        }

        // Pass 2: compute the canonical prequant L2 frame, decoded max |X|, and row scales.
        // The L2 uses raw int8 squares times exact scale squares; it does not square the
        // already-rounded decoded X values.
        let n_blocks = k / BLOCK_SIZE;
        // First compute each block's integer square sum/product and its doubled scale exponent.
        let block_e2: Vec<u64> = (0..n_blocks)
            .map(|b| {
                let nsq: u64 = int8[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE]
                    .iter()
                    .map(|&v| (i64::from(v) * i64::from(v)) as u64)
                    .sum();
                let m = Bf16Fields::from_code(scales[b]).m();
                if m * m * nsq == 0 {
                    0
                } else {
                    2 * Bf16Fields::from_code(scales[b]).e_star() as u64
                }
            })
            .collect();
        // The group frame is the maximum live doubled scale exponent, or zero if none (B6/B7).
        let frame_doubled_scale_exponent = block_e2.iter().copied().max().unwrap();
        // Fill the row-by-row witnesses for each block floor and the running framed sum S.
        let mut blk_rows: Vec<BlockL2Witness> = Vec::with_capacity(k);
        let (mut nsq_run, mut max_e2_run, mut l2_acc_run) = (0u64, 0u64, 0u64);
        for (j, x) in xs.iter().enumerate() {
            let int_sq = {
                let v = i64::from(int8[j]);
                (v * v) as u64
            };
            nsq_run = if j % BLOCK_SIZE == 0 { int_sq } else { nsq_run + int_sq };
            let m = x.2.m();
            let block_l2_product = m * m * nsq_run;
            let is_block_final = j % BLOCK_SIZE == BLOCK_SIZE - 1;
            let live = is_block_final && block_l2_product != 0;
            let block_doubled_scale_exponent = if live { block_e2[j / BLOCK_SIZE] } else { 0 };
            debug_assert!(!live || block_doubled_scale_exponent > 0);
            max_e2_run = max_e2_run.max(block_doubled_scale_exponent);
            let x_div = block_l2_product << wl2;
            debug_assert!(x_div < 1 << 54, "B10: the dividend cap");
            let scaled_block_product_limbs = [x_div & 0xFFFF, (x_div >> 16) & 0xFFFF, (x_div >> 32) & 0xFFFF, x_div >> 48];
            let mut w = BlockL2Witness {
                block_int_squared_sum: nsq_run,
                block_l2_product,
                block_doubled_scale_exponent,
                running_max_doubled_scale_exponent: max_e2_run,
                scaled_block_product_limbs,
                shift_remainder_power: 1,
                shift_complement_power: 1 << 16,
                ..BlockL2Witness::default()
            };
            if live {
                let sigma = frame_doubled_scale_exponent - block_doubled_scale_exponent;
                if sigma >= FAR_SIGMA {
                    w.is_far_shift = true; // The term is exactly zero: X < 2^54 <= 2^sigma.
                } else {
                    let qh = (sigma / 16) as usize;
                    let rh = sigma % 16;
                    w.shift_limb_selector[qh] = true;
                    w.shift_remainder_power = 1 << rh;
                    w.shift_complement_power = 1 << (16 - rh);
                    w.selected_limb_quotient = scaled_block_product_limbs[qh] >> rh;
                    w.selected_limb_remainder = scaled_block_product_limbs[qh] & ((1 << rh) - 1);
                    let h: u64 = scaled_block_product_limbs
                        .iter()
                        .skip(qh + 1)
                        .rev()
                        .fold(0, |acc, &l| (acc << 16) + l);
                    let term = h * w.shift_complement_power + w.selected_limb_quotient;
                    debug_assert_eq!(
                        term,
                        block_l2_term(block_l2_product, sigma, wl2),
                        "B16: the split is the floor"
                    );
                    l2_acc_run += term;
                }
            }
            w.running_l2_frame_sum = l2_acc_run;
            blk_rows.push(w);
        }
        let s = l2_acc_run;
        // A live frame block has sigma = 0 and a positive contribution, so S and the frame are
        // zero together.
        assert_eq!(
            s == 0,
            frame_doubled_scale_exponent == 0,
            "S and the doubled scale exponent frame are zero together"
        );
        let max_abs_final = xs.iter().map(|x| x.3.out.fields(x.4).abs_code()).max().unwrap();
        // The canonical mean square is S * 2^(F - 268 - Wl2) / k. ScaleStark proves the bf16
        // square root with an exact integer bracket; `rne_sqrt_hat` is the native mirror of
        // that check. The result is snapped to the nearest four-code grid, then floored at
        // 2^-32 (with linf) for the scale derivation below, so this witness is bit-exact with
        // the ScaleStark tuple for any sanctioned k — all-zero rows included.
        let l2_code = if s == 0 {
            0
        } else {
            let claim = rne_sqrt_hat(s, frame_doubled_scale_exponent as i64, i64::from(wl2), program.k as u64);
            let sqrt_code = ((claim.exp << 7) + claim.mantissa) as u16;
            let snapped = round_l2_to_grid(sqrt_code);
            assert!(snapped & 0x7FFF < 0x7F80, "grid snap overflowed bf16 (native error path)");
            snapped
        };
        let linf_code = max_abs_final as u16;
        let (alpha_code, beta_code) = if live {
            // The scheme's norm floor (the reference `row_norms`): alpha/beta derive from the
            // FLOORED norms, exactly as `noisy_quantize` computes them — total even on all-zero
            // rows. The raw codes stay in the committed columns and the group-tuple CTL;
            // ScaleStark floors them in-circuit (group H0) before its chain.
            Fp8E4M3Quant
                .derive_row_scales(
                    bf16_max(l2_code, NORM_FLOOR_CODE),
                    bf16_max(linf_code, NORM_FLOOR_CODE),
                    program.r,
                )
                .expect("scale derivation is total on floored norms")
        } else {
            // Phantom scales for dead groups: alpha = 2^-126 (the smallest positive normal —
            // S1's RC16s demand a positive normal alpha on every row), beta = +0 (kills the
            // noise term). No ScaleStark counterpart exists: the group-tuple CTL is
            // liveness-filtered.
            (0x0080, 0x0000)
        };
        let alpha = Bf16Fields::from_code(alpha_code);
        let beta = Bf16Fields::from_code(beta_code);
        assert!(
            !alpha.sign && (1..=254).contains(&alpha.exp),
            "alpha must be positive normal (S1)"
        );
        assert!(!beta.sign, "beta is nonnegative");
        // The liveness bound in bf16-code space. A code past the largest finite abs code
        // (possible when `E*(l2f) >= 253`) marks every entry alive, matching the plaintext:
        // every finite `|X|` is below `4*l2f` there. Dead groups take the floored-zero bound;
        // their flags vanish anyway (X = 0) and C1 pins them to zero.
        let dead_bound = u64::from(bf16_max(l2_code, NORM_FLOOR_CODE)) + 256;
        // The group's noise-std encoding and normalized significand (jackpot check 4). With
        // `l2f` the floored L2 norm, `sigma = DELTA * alpha * l2f` is the exact product
        // `M(alpha)*M(l2f) * 2^(E(alpha) + E(l2f) - 269)`, and the significand product spans
        // [2^14, 2^16) (both factors are normal), so with `wide = [product >= 2^15]`:
        // `enc(sigma) = e(sigma) + 268 = E(alpha) + E(l2f) + wide + 13` and
        // `normalized_sigma_significand = product * 2^(1 - wide)` lands in [2^15, 2^16).
        let (sigma_biased_exponent, normalized_sigma_significand) = if live {
            let l2f = Bf16Fields::from_code(bf16_max(l2_code, NORM_FLOOR_CODE));
            let product = (128 + alpha.mantissa) * l2f.m();
            let wide = u64::from(product >= 1 << 15);
            let enc = alpha.exp + l2f.exp + wide + 13;
            debug_assert_eq!(
                enc as i64 - 268,
                {
                    let sigma = DELTA * f64::from(bf16_to_f32(alpha_code)) * f64::from(bf16_to_f32(l2f.code()));
                    ((sigma.to_bits() >> 52) & 0x7FF) as i64 - 1023
                },
                "sigma encoding diverged from the exact exponent"
            );
            (enc, product << (1 - wide))
        } else {
            (0, 0)
        };
        let scales_w = GroupScales {
            frame_doubled_scale_exponent,
            alpha,
            beta,
            alpha_code,
            beta_code,
            dead_bound,
            sigma_biased_exponent,
            normalized_sigma_significand,
        };

        // Pass 3: the noising path per element (groups M/W/Q/C).
        let mut elements = Vec::with_capacity(k);
        let (mut max_abs, mut dead_count) = (0u64, 0u64);
        for (j, &(byte, int_f, scale_f, dec, x_sign)) in xs.iter().enumerate() {
            let x = dec.out.fields(x_sign);
            max_abs = if j == 0 { x.abs_code() } else { max_abs.max(x.abs_code()) };

            let noise_f = Bf16Fields::from_code(noise[j]);
            // M: NOISE_TERM = beta * n; output sign = NOISE_SIGN (beta >= 0).
            let beta_mul = mul_gadget(scales_w.beta, noise_f);
            let nt_code = beta_mul.out.fields(noise_f.sign).code();
            debug_assert_eq!(
                nt_code,
                bf16_mul(scales_w.beta_code, noise_f.code()).expect("noise term is finite"),
                "beta MUL mirror diverged from bf16_mul"
            );
            // W: NOISED = fma(alpha, x, beta*n), one rounding.
            let m_p = (128 + scales_w.alpha.mantissa) * x.m();
            let ep_raw = scales_w.alpha.exp as i64 + x.e_star() - MUL_LSB_OFFSET;
            let nt = beta_mul.out.fields(noise_f.sign);
            let fma = fma_gadget(m_p, ep_raw, x.m() == 0, x_sign, nt.m(), nt.e_star() - 134, noise_f.sign);
            let noised_code = fma.out.fields(fma.out_sign).code();
            debug_assert_eq!(
                noised_code,
                bf16_fma(scales_w.alpha_code, x.code(), nt_code).expect("noised value is finite"),
                "FMA mirror diverged from bf16_fma"
            );
            // Q: quantize. C: the jackpot liveness flag — the bf16-code form of the plaintext
            // `|X| >= tau_idle * DELTA * l2f` (abs codes are value-ordered on finite bf16).
            let code_noised = qcast(noised_code);
            let is_dead = x.abs_code() >= scales_w.dead_bound;
            debug_assert_eq!(
                is_dead,
                f64::from(bf16_to_f32(x.code()).abs())
                    >= 8.0 * DELTA * f64::from(bf16_to_f32(bf16_max(l2_code, NORM_FLOOR_CODE))),
                "liveness flag diverged from the plaintext dead predicate"
            );
            dead_count += u64::from(is_dead);

            // The scaled element's exponent encoding and the summand encoding lambda
            // (jackpot check 4): `|alpha*x| = M(alpha)*M(X) * 2^(E(alpha) + E*(X) - 268)`, so a
            // nonzero X has `enc(alpha*x) = E(alpha) + E*(X) - 1 + bit_length(M(alpha)*M(X))`.
            let scaled_product_width = u64::from(64 - m_p.leading_zeros());
            let scaled_biased_exponent = if x.m() == 0 {
                0
            } else {
                let enc = (scales_w.alpha.exp as i64 + x.e_star() - 1) as u64 + scaled_product_width;
                debug_assert_eq!(
                    enc as i64 - 268,
                    {
                        let scaled = f64::from(bf16_to_f32(scales_w.alpha_code)) * f64::from(bf16_to_f32(x.code())).abs();
                        ((scaled.to_bits() >> 52) & 0x7FF) as i64 - 1023
                    },
                    "scaled-element encoding diverged from the exact exponent"
                );
                enc
            };
            let scaled_normalization_power = 1u64 << (16 - scaled_product_width);
            let scaled_significand = m_p * scaled_normalization_power;
            // The summand's score witness (jackpot check 4). A dead group's phantom rows carry
            // the all-zero witness with the shift power pinned to 2^0; its lambda is 0.
            let score = if live {
                lambda_witness(
                    scaled_significand,
                    scaled_biased_exponent,
                    scales_w.normalized_sigma_significand,
                    scales_w.sigma_biased_exponent,
                )
            } else {
                LambdaWitness {
                    x_dominates: false,
                    exponent_gap: 0,
                    gap_is_far: false,
                    near_gap: 0,
                    near_gap_pow: 1,
                    dominant_significand: 0,
                    smaller_significand: 0,
                    half_quotient: 0,
                    half_remainder: 0,
                    quotient: 0,
                    remainder: 0,
                    sum_of_squares_top: 0,
                    sum_of_squares_rest: 0,
                    log_fraction: 0,
                    lambda: 0,
                }
            };

            elements.push(ElementWitness {
                byte,
                int_f,
                scale_f,
                dec,
                x_sign,
                blk: blk_rows[j],
                max_abs,
                noise_f,
                beta_mul,
                fma,
                code_noised,
                is_dead,
                dead_count,
                scaled_product_width,
                scaled_biased_exponent,
                scaled_significand,
                scaled_normalization_power,
                score,
            });
        }

        SideWitness {
            elements,
            scales: scales_w,
        }
    }

    fn fill_a<F: RichField>(&self, row: &mut InputQuantColumnsView<F>, j: usize) {
        let e = &self.elements[j];
        fill_bf16(&mut row.noise_a, e.noise_f);
        row.int8_byte_a = F::from_canonical_u8(e.byte);
        fill_bf16(&mut row.int_bf16_a, e.int_f);
        fill_bf16(&mut row.scale_a, e.scale_f);
        fill_mul_decode(&mut row.decode_multiply_a, e.dec);
        row.x_a_sign = F::from_bool(e.x_sign);
        row.x_a_exp = F::from_canonical_u64(e.dec.out.exp);
        row.x_a_mantissa = F::from_canonical_u64(e.dec.out.mantissa);
        row.x_a_exp_is_zero = F::from_bool(e.dec.out.exp_is_zero);
        row.x_a_is_zero = F::from_bool(e.dec.out.is_zero);
        fill_blk_l2(&mut row.block_l2_a, e.blk, self.scales.frame_doubled_scale_exponent);
        row.max_abs_a = F::from_canonical_u64(e.max_abs);
        row.alpha_a_exp = F::from_canonical_u64(self.scales.alpha.exp);
        row.alpha_a_mantissa = F::from_canonical_u64(self.scales.alpha.mantissa);
        row.beta_a_exp = F::from_canonical_u64(self.scales.beta.exp);
        row.beta_a_mantissa = F::from_canonical_u64(self.scales.beta.mantissa);
        row.beta_a_exp_is_zero = F::from_bool(self.scales.beta.exp_is_zero());
        fill_mul(&mut row.noise_scale_multiply_a, e.beta_mul);
        row.noised_fma_product_a = F::from_canonical_u64((128 + self.scales.alpha.mantissa) * e.dec.out.fields(e.x_sign).m());
        fill_fma(&mut row.noised_value_fma_a, e.fma);
        row.code_noised_a = F::from_canonical_u8(e.code_noised);
        row.dead_bound_a = F::from_canonical_u64(self.scales.dead_bound);
        row.is_dead_a = F::from_bool(e.is_dead);
        row.dead_count_a = F::from_canonical_u64(e.dead_count);
        row.scaled_product_width_a = F::from_canonical_u64(e.scaled_product_width);
        row.scaled_biased_exponent_a = F::from_canonical_u64(e.scaled_biased_exponent);
        row.sigma_biased_exponent_a = F::from_canonical_u64(self.scales.sigma_biased_exponent);
        row.normalized_sigma_significand_a = F::from_canonical_u64(self.scales.normalized_sigma_significand);
        row.scaled_normalization_power_a = F::from_canonical_u64(e.scaled_normalization_power);
        row.scaled_significand_a = F::from_canonical_u64(e.scaled_significand);
        row.x_dominates_a = F::from_bool(e.score.x_dominates);
        row.exponent_gap_a = F::from_canonical_u64(e.score.exponent_gap);
        row.gap_is_far_a = F::from_bool(e.score.gap_is_far);
        row.near_gap_a = F::from_canonical_u64(e.score.near_gap);
        row.near_gap_pow_a = F::from_canonical_u64(e.score.near_gap_pow);
        row.dominant_significand_a = F::from_canonical_u64(e.score.dominant_significand);
        row.smaller_significand_a = F::from_canonical_u64(e.score.smaller_significand);
        row.half_quotient_a = F::from_canonical_u64(e.score.half_quotient);
        row.half_remainder_a = F::from_canonical_u64(e.score.half_remainder);
        row.quotient_a = F::from_canonical_u64(e.score.quotient);
        row.remainder_a = F::from_canonical_u64(e.score.remainder);
        row.sum_of_squares_top_a = F::from_canonical_u64(e.score.sum_of_squares_top);
        row.sum_of_squares_rest_lo_a = F::from_canonical_u64(e.score.sum_of_squares_rest & 0xFFFF);
        row.sum_of_squares_rest_hi_a = F::from_canonical_u64(e.score.sum_of_squares_rest >> 16);
        row.log_fraction_a = F::from_canonical_u64(e.score.log_fraction);
        row.summand_score_a = F::from_canonical_u64(e.score.lambda);
    }

    fn fill_b<F: RichField>(&self, row: &mut InputQuantColumnsView<F>, j: usize) {
        let e = &self.elements[j];
        fill_bf16(&mut row.noise_b, e.noise_f);
        row.int8_byte_b = F::from_canonical_u8(e.byte);
        fill_bf16(&mut row.int_bf16_b, e.int_f);
        fill_bf16(&mut row.scale_b, e.scale_f);
        fill_mul_decode(&mut row.decode_multiply_b, e.dec);
        row.x_b_sign = F::from_bool(e.x_sign);
        row.x_b_exp = F::from_canonical_u64(e.dec.out.exp);
        row.x_b_mantissa = F::from_canonical_u64(e.dec.out.mantissa);
        row.x_b_exp_is_zero = F::from_bool(e.dec.out.exp_is_zero);
        row.x_b_is_zero = F::from_bool(e.dec.out.is_zero);
        fill_blk_l2(&mut row.block_l2_b, e.blk, self.scales.frame_doubled_scale_exponent);
        row.max_abs_b = F::from_canonical_u64(e.max_abs);
        row.alpha_b_exp = F::from_canonical_u64(self.scales.alpha.exp);
        row.alpha_b_mantissa = F::from_canonical_u64(self.scales.alpha.mantissa);
        row.beta_b_exp = F::from_canonical_u64(self.scales.beta.exp);
        row.beta_b_mantissa = F::from_canonical_u64(self.scales.beta.mantissa);
        row.beta_b_exp_is_zero = F::from_bool(self.scales.beta.exp_is_zero());
        fill_mul(&mut row.noise_scale_multiply_b, e.beta_mul);
        row.noised_fma_product_b = F::from_canonical_u64((128 + self.scales.alpha.mantissa) * e.dec.out.fields(e.x_sign).m());
        fill_fma(&mut row.noised_value_fma_b, e.fma);
        row.code_noised_b = F::from_canonical_u8(e.code_noised);
        row.dead_bound_b = F::from_canonical_u64(self.scales.dead_bound);
        row.is_dead_b = F::from_bool(e.is_dead);
        row.dead_count_b = F::from_canonical_u64(e.dead_count);
        row.scaled_product_width_b = F::from_canonical_u64(e.scaled_product_width);
        row.scaled_biased_exponent_b = F::from_canonical_u64(e.scaled_biased_exponent);
        row.sigma_biased_exponent_b = F::from_canonical_u64(self.scales.sigma_biased_exponent);
        row.normalized_sigma_significand_b = F::from_canonical_u64(self.scales.normalized_sigma_significand);
        row.scaled_normalization_power_b = F::from_canonical_u64(e.scaled_normalization_power);
        row.scaled_significand_b = F::from_canonical_u64(e.scaled_significand);
        row.x_dominates_b = F::from_bool(e.score.x_dominates);
        row.exponent_gap_b = F::from_canonical_u64(e.score.exponent_gap);
        row.gap_is_far_b = F::from_bool(e.score.gap_is_far);
        row.near_gap_b = F::from_canonical_u64(e.score.near_gap);
        row.near_gap_pow_b = F::from_canonical_u64(e.score.near_gap_pow);
        row.dominant_significand_b = F::from_canonical_u64(e.score.dominant_significand);
        row.smaller_significand_b = F::from_canonical_u64(e.score.smaller_significand);
        row.half_quotient_b = F::from_canonical_u64(e.score.half_quotient);
        row.half_remainder_b = F::from_canonical_u64(e.score.half_remainder);
        row.quotient_b = F::from_canonical_u64(e.score.quotient);
        row.remainder_b = F::from_canonical_u64(e.score.remainder);
        row.sum_of_squares_top_b = F::from_canonical_u64(e.score.sum_of_squares_top);
        row.sum_of_squares_rest_lo_b = F::from_canonical_u64(e.score.sum_of_squares_rest & 0xFFFF);
        row.sum_of_squares_rest_hi_b = F::from_canonical_u64(e.score.sum_of_squares_rest >> 16);
        row.log_fraction_b = F::from_canonical_u64(e.score.log_fraction);
        row.summand_score_b = F::from_canonical_u64(e.score.lambda);
    }
}

fn fill_blk_l2<F: RichField>(dst: &mut BlockL2Columns<F>, w: BlockL2Witness, frame_doubled_scale_exponent: u64) {
    dst.block_int_squared_sum = F::from_canonical_u64(w.block_int_squared_sum);
    dst.block_l2_product = F::from_canonical_u64(w.block_l2_product);
    dst.block_l2_product_nonzero = F::from_bool(w.block_l2_product != 0);
    dst.block_l2_product_inverse = if w.block_l2_product == 0 {
        F::ZERO
    } else {
        F::from_canonical_u64(w.block_l2_product).inverse()
    };
    dst.block_doubled_scale_exponent = F::from_canonical_u64(w.block_doubled_scale_exponent);
    dst.running_max_doubled_scale_exponent = F::from_canonical_u64(w.running_max_doubled_scale_exponent);
    dst.frame_doubled_scale_exponent = F::from_canonical_u64(frame_doubled_scale_exponent);
    for (d, l) in dst.scaled_block_product_limbs.iter_mut().zip(w.scaled_block_product_limbs) {
        *d = F::from_canonical_u64(l);
    }
    dst.shift_remainder_power = F::from_canonical_u64(w.shift_remainder_power);
    dst.shift_complement_power = F::from_canonical_u64(w.shift_complement_power);
    for (d, q) in dst.shift_limb_selector.iter_mut().zip(w.shift_limb_selector) {
        *d = F::from_bool(q);
    }
    dst.selected_limb_quotient = F::from_canonical_u64(w.selected_limb_quotient);
    dst.selected_limb_remainder = F::from_canonical_u64(w.selected_limb_remainder);
    dst.is_far_shift = F::from_bool(w.is_far_shift);
    dst.running_l2_frame_sum = F::from_canonical_u64(w.running_l2_frame_sum);
}

fn fill_bf16<F: RichField>(dst: &mut Bf16FieldsView<F>, f: Bf16Fields) {
    dst.sign = F::from_bool(f.sign);
    dst.exp = F::from_canonical_u64(f.exp);
    dst.mantissa = F::from_canonical_u64(f.mantissa);
    dst.exp_is_zero = F::from_bool(f.exp_is_zero());
}

fn fill_mul_decode<F: RichField>(dst: &mut MulDecodeBlockView<F>, m: MulWitness) {
    dst.sig_product = F::from_canonical_u64(m.sig_product);
    dst.cut_depth = F::from_canonical_u64(m.cut_depth);
    dst.width_adjust = F::from_canonical_u64(m.out.width_adjust);
}

fn fill_mul<F: RichField>(dst: &mut MulBlockView<F>, m: MulWitness) {
    dst.sig_product = F::from_canonical_u64(m.sig_product);
    dst.cut_depth = F::from_canonical_u64(m.cut_depth);
    dst.out_exp = F::from_canonical_u64(m.out.exp);
    dst.out_mantissa = F::from_canonical_u64(m.out.mantissa);
    dst.width_adjust = F::from_canonical_u64(m.out.width_adjust);
    dst.out_is_zero = F::from_bool(m.out.is_zero);
    dst.out_exp_is_zero = F::from_bool(m.out.exp_is_zero);
}

fn fill_fma<F: RichField>(dst: &mut FmaBlockView<F>, w: FmaWitness) {
    dst.product_scale_ge_addend = F::from_bool(w.product_scale_ge_addend);
    dst.scale_gap_slack = F::from_canonical_u64(w.scale_gap_slack);
    dst.is_far_gap = F::from_bool(w.is_far_gap);
    dst.far_gap_slack = F::from_canonical_u64(w.far_gap_slack);
    dst.exp_gap_capped = F::from_canonical_u64(w.exp_gap_capped);
    dst.exp_gap_pow2 = F::from_canonical_u64(w.exp_gap_pow2);
    dst.aligned_product = F::from_canonical_u64(w.aligned_product);
    dst.aligned_addend = F::from_canonical_u64(w.aligned_addend);
    dst.folded_magnitude = F::from_canonical_u64(w.folded_magnitude);
    dst.folded_is_zero = F::from_bool(w.folded_magnitude == 0);
    dst.folded_inverse = if w.folded_magnitude == 0 {
        F::ZERO
    } else {
        F::from_canonical_u64(w.folded_magnitude).inverse()
    };
    dst.is_wide = F::from_bool(w.is_wide);
    dst.compression_shift = F::from_canonical_u64(w.compression_shift);
    dst.shift_pow2 = F::from_canonical_u64(w.shift_pow2);
    dst.compression_quotient = F::from_canonical_u64(w.compression_quotient);
    dst.compression_quotient_lsb = F::from_canonical_u64(w.compression_quotient_lsb);
    dst.compression_remainder = F::from_canonical_u64(w.compression_remainder);
    dst.compression_remainder_inverse = if w.compression_remainder == 0 {
        F::ZERO
    } else {
        F::from_canonical_u64(w.compression_remainder).inverse()
    };
    dst.sticky = F::from_bool(w.sticky);
    dst.rounding_significand_key = F::from_canonical_u64(w.rounding_significand_key);
    dst.rounding_significand_key_high_bit = F::from_canonical_u64(w.rounding_significand_key >> 16);
    dst.key_scale = field_i64(w.key_scale);
    dst.cut_used = F::from_canonical_u64(w.cut_used);
    dst.out_sign = F::from_bool(w.out_sign);
    dst.out_exp = F::from_canonical_u64(w.out.exp);
    dst.out_mantissa = F::from_canonical_u64(w.out.mantissa);
    dst.width_adjust = F::from_canonical_u64(w.out.width_adjust);
    dst.out_is_zero = F::from_bool(w.out.is_zero);
    dst.out_exp_is_zero = F::from_bool(w.out.exp_is_zero);
}

// Constraints, written once against the generic `Evaluator`

/// One side's column values, copied out of the row view so the A/B constraint logic is written
/// once.
struct SideVals<V: Copy> {
    noise: Bf16FieldsView<V>,
    int_f: Bf16FieldsView<V>,
    scale: Bf16FieldsView<V>,
    mul_decode: MulDecodeBlockView<V>,
    x_sign: V,
    x_exp: V,
    x_mantissa: V,
    x_exp_is_zero: V,
    x_is_zero: V,
    int8_byte: V,
    blk: BlockL2Columns<V>,
    max_abs: V,
    alpha_exp: V,
    alpha_mantissa: V,
    beta_exp: V,
    beta_mantissa: V,
    beta_exp_is_zero: V,
    mul_beta_n: MulBlockView<V>,
    fma_sig_product: V,
    fma: FmaBlockView<V>,
    live: V,
    dead_bound: V,
    is_dead: V,
    dead_count: V,
    scaled_product_width: V,
    scaled_biased_exponent: V,
    sigma_biased_exponent: V,
    normalized_sigma_significand: V,
    scaled_normalization_power: V,
    scaled_significand: V,
    x_dominates: V,
    exponent_gap: V,
    gap_is_far: V,
    near_gap: V,
    near_gap_pow: V,
    dominant_significand: V,
    smaller_significand: V,
    half_quotient: V,
    half_remainder: V,
    quotient: V,
    remainder: V,
    sum_of_squares_top: V,
    sum_of_squares_rest_lo: V,
    sum_of_squares_rest_hi: V,
    log_fraction: V,
    lambda: V,
}

fn side_a<V: Copy>(v: &InputQuantColumnsView<V>) -> SideVals<V> {
    SideVals {
        noise: v.noise_a,
        int_f: v.int_bf16_a,
        scale: v.scale_a,
        mul_decode: v.decode_multiply_a,
        x_sign: v.x_a_sign,
        x_exp: v.x_a_exp,
        x_mantissa: v.x_a_mantissa,
        x_exp_is_zero: v.x_a_exp_is_zero,
        x_is_zero: v.x_a_is_zero,
        int8_byte: v.int8_byte_a,
        blk: v.block_l2_a,
        max_abs: v.max_abs_a,
        alpha_exp: v.alpha_a_exp,
        alpha_mantissa: v.alpha_a_mantissa,
        beta_exp: v.beta_a_exp,
        beta_mantissa: v.beta_a_mantissa,
        beta_exp_is_zero: v.beta_a_exp_is_zero,
        mul_beta_n: v.noise_scale_multiply_a,
        fma_sig_product: v.noised_fma_product_a,
        fma: v.noised_value_fma_a,
        live: v.a_live,
        dead_bound: v.dead_bound_a,
        is_dead: v.is_dead_a,
        dead_count: v.dead_count_a,
        scaled_product_width: v.scaled_product_width_a,
        scaled_biased_exponent: v.scaled_biased_exponent_a,
        sigma_biased_exponent: v.sigma_biased_exponent_a,
        normalized_sigma_significand: v.normalized_sigma_significand_a,
        scaled_normalization_power: v.scaled_normalization_power_a,
        scaled_significand: v.scaled_significand_a,
        x_dominates: v.x_dominates_a,
        exponent_gap: v.exponent_gap_a,
        gap_is_far: v.gap_is_far_a,
        near_gap: v.near_gap_a,
        near_gap_pow: v.near_gap_pow_a,
        dominant_significand: v.dominant_significand_a,
        smaller_significand: v.smaller_significand_a,
        half_quotient: v.half_quotient_a,
        half_remainder: v.half_remainder_a,
        quotient: v.quotient_a,
        remainder: v.remainder_a,
        sum_of_squares_top: v.sum_of_squares_top_a,
        sum_of_squares_rest_lo: v.sum_of_squares_rest_lo_a,
        sum_of_squares_rest_hi: v.sum_of_squares_rest_hi_a,
        log_fraction: v.log_fraction_a,
        lambda: v.summand_score_a,
    }
}

fn side_b<V: Copy>(v: &InputQuantColumnsView<V>) -> SideVals<V> {
    SideVals {
        noise: v.noise_b,
        int_f: v.int_bf16_b,
        scale: v.scale_b,
        mul_decode: v.decode_multiply_b,
        x_sign: v.x_b_sign,
        x_exp: v.x_b_exp,
        x_mantissa: v.x_b_mantissa,
        x_exp_is_zero: v.x_b_exp_is_zero,
        x_is_zero: v.x_b_is_zero,
        int8_byte: v.int8_byte_b,
        blk: v.block_l2_b,
        max_abs: v.max_abs_b,
        alpha_exp: v.alpha_b_exp,
        alpha_mantissa: v.alpha_b_mantissa,
        beta_exp: v.beta_b_exp,
        beta_mantissa: v.beta_b_mantissa,
        beta_exp_is_zero: v.beta_b_exp_is_zero,
        mul_beta_n: v.noise_scale_multiply_b,
        fma_sig_product: v.noised_fma_product_b,
        fma: v.noised_value_fma_b,
        live: v.b_live,
        dead_bound: v.dead_bound_b,
        is_dead: v.is_dead_b,
        dead_count: v.dead_count_b,
        scaled_product_width: v.scaled_product_width_b,
        scaled_biased_exponent: v.scaled_biased_exponent_b,
        sigma_biased_exponent: v.sigma_biased_exponent_b,
        normalized_sigma_significand: v.normalized_sigma_significand_b,
        scaled_normalization_power: v.scaled_normalization_power_b,
        scaled_significand: v.scaled_significand_b,
        x_dominates: v.x_dominates_b,
        exponent_gap: v.exponent_gap_b,
        gap_is_far: v.gap_is_far_b,
        near_gap: v.near_gap_b,
        near_gap_pow: v.near_gap_pow_b,
        dominant_significand: v.dominant_significand_b,
        smaller_significand: v.smaller_significand_b,
        half_quotient: v.half_quotient_b,
        half_remainder: v.half_remainder_b,
        quotient: v.quotient_b,
        remainder: v.remainder_b,
        sum_of_squares_top: v.sum_of_squares_top_b,
        sum_of_squares_rest_lo: v.sum_of_squares_rest_lo_b,
        sum_of_squares_rest_hi: v.sum_of_squares_rest_hi_b,
        log_fraction: v.log_fraction_b,
        lambda: v.summand_score_b,
    }
}

/// `M(V) = (1 - EXP_IS_ZERO)*128 + MANTISSA` as an expression (degree 1).
fn m_expr<V: Copy, S: Copy, E: Evaluator<V, S>>(eval: &mut E, exp_is_zero: V, mantissa: V) -> V {
    let c128 = eval.i32(128);
    let one = eval.i32(1);
    let not_eiz = eval.sub(one, exp_is_zero);
    eval.mad(not_eiz, c128, mantissa)
}

/// `1 - 2*bit`: maps a sign bit to the ±1 factor recovering a magnitude from a signed value.
fn sign_factor<V: Copy, S: Copy, E: Evaluator<V, S>>(eval: &mut E, bit: V) -> V {
    let one = eval.i32(1);
    let twice = eval.add(bit, bit);
    eval.sub(one, twice)
}

/// Binds the output fields shared by each bf16 multiplication gadget. A normal nonzero result
/// must have exponent `E*(a) + E*(b) - 268 + width_adjust`; subnormal and zero paths force the
/// appropriate fields to zero. RNERND supplies the output flags from table rows containing
/// only bits, so no duplicate boolean constraints are needed here.
fn eval_mul_out<V: Copy, S: Copy, E: Evaluator<V, S>>(eval: &mut E, blk: &MulBlockView<V>, e_star_a: V, e_star_b: V) {
    let one = eval.i32(1);
    let c_off = eval.i32(MUL_LSB_OFFSET as i32);
    // M4: bind the exponent only on a normal, nonzero result.
    let e_sum = eval.add(e_star_a, e_star_b);
    let base = eval.sub(e_sum, c_off);
    let target = eval.add(base, blk.width_adjust);
    let exp_diff = eval.sub(blk.out_exp, target);
    let not_oiz = eval.sub(one, blk.out_is_zero);
    let not_oeiz = eval.sub(one, blk.out_exp_is_zero);
    let gate = eval.mul(not_oiz, not_oeiz);
    let c = eval.mul(gate, exp_diff);
    eval.constraint(c);
    // M5: a subnormal/zero has exponent field zero; an exact zero also has mantissa zero.
    let c = eval.mul(blk.out_exp_is_zero, blk.out_exp);
    eval.constraint(c);
    let c = eval.mul(blk.out_is_zero, blk.out_exp);
    eval.constraint(c);
    let c = eval.mul(blk.out_is_zero, blk.out_mantissa);
    eval.constraint(c);
}

/// Computes the square of a two's-complement int8 byte:
/// `byte^2 - 512*sign*byte + 2^16*sign`. INT8DEC binds `sign`; the formula is `byte^2` for a
/// positive value and `(byte - 256)^2` for a negative one.
fn int_sq_expr<V: Copy, S: Copy, E: Evaluator<V, S>>(eval: &mut E, byte: V, sign: V) -> V {
    let sq = eval.mul(byte, byte);
    let c512 = eval.i32(512);
    let signed = eval.mul(sign, byte);
    let sub = eval.mul(c512, signed);
    let c_hi = eval.u64(1 << 16);
    let hi = eval.mul(c_hi, sign);
    let t = eval.sub(sq, sub);
    eval.add(t, hi)
}

/// Recomposes all 16-bit dividend limbs strictly above the selected shift-boundary limb.
fn blk_high_limbs<V: Copy, S: Copy, E: Evaluator<V, S>>(eval: &mut E, blk: &BlockL2Columns<V>) -> V {
    let w16 = eval.u64(1 << 16);
    let w32 = eval.u64(1 << 32);
    let above0 = {
        let a = eval.mad(blk.scaled_block_product_limbs[2], w16, blk.scaled_block_product_limbs[1]);
        eval.mad(blk.scaled_block_product_limbs[3], w32, a)
    };
    let above1 = eval.mad(blk.scaled_block_product_limbs[3], w16, blk.scaled_block_product_limbs[2]);
    let h0 = eval.mul(blk.shift_limb_selector[0], above0);
    let h1 = eval.mul(blk.shift_limb_selector[1], above1);
    let h2 = eval.mul(blk.shift_limb_selector[2], blk.scaled_block_product_limbs[3]);
    let h01 = eval.add(h0, h1);
    eval.add(h01, h2)
}

/// Evaluates one side's constraints (the A and B sides are fully symmetric). `wl2_pow` is the
/// public-input value `2^Wl2` used by B10; as a public input it is a degree-0 scalar, so
/// constraint degrees are unchanged.
fn eval_side<V, S, E>(eval: &mut E, lv: &SideVals<V>, nv: &SideVals<V>, gf: V, next_block_start: V, wl2_pow: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let one = eval.i32(1);
    let c128 = eval.i32(128);
    let not_gf = eval.sub(one, gf);

    // Group P — bind the raw int8 value and its eight-element block scale.
    // P1 is an INT8DEC lookup from the raw byte to its exact bf16 fields. The table also proves
    // its sign and exponent-zero outputs are bits.
    // P2 keeps every scale field unchanged inside an eight-row block. A next-row block start
    // disables the equality, including at the cyclic last-row-to-first-row boundary.
    let not_bs = eval.sub(one, next_block_start);
    for (f_next, f_local) in [
        (nv.scale.sign, lv.scale.sign),
        (nv.scale.exp, lv.scale.exp),
        (nv.scale.mantissa, lv.scale.mantissa),
        (nv.scale.exp_is_zero, lv.scale.exp_is_zero),
    ] {
        let diff = eval.sub(f_next, f_local);
        let c = eval.mul(not_bs, diff);
        eval.constraint(c);
    }
    // P3 proves that the free scale sign is a bit. Lookup tables separately bind the exponent,
    // exponent-zero flag, and seven-bit mantissa.
    eval.constraint_bool(lv.scale.sign);
    // P4: X_SIGN = INT_SIGN + SCALE_SIGN - 2*INT_SIGN*SCALE_SIGN (multiply sign rule, deg 2).
    let sum = eval.add(lv.int_f.sign, lv.scale.sign);
    let prod = eval.mul(lv.int_f.sign, lv.scale.sign);
    let twice = eval.add(prod, prod);
    let xor = eval.sub(sum, twice);
    let c = eval.sub(lv.x_sign, xor);
    eval.constraint(c);

    // Group U — prove X = RNE_bf16(int8 * block_scale).
    // U1 binds the exact, unrounded integer significand product M(int8)*M(scale).
    let m_int = m_expr(eval, lv.int_f.exp_is_zero, lv.int_f.mantissa);
    let m_scale = m_expr(eval, lv.scale.exp_is_zero, lv.scale.mantissa);
    let sig = eval.mul(m_int, m_scale);
    let c = eval.sub(lv.mul_decode.sig_product, sig);
    eval.constraint(c);
    // U2/U3 are lookups: CLAMP22 chooses the subnormal cut and RNERND returns the rounded
    // mantissa and zero/subnormal flags. U4 binds the exponent on a normal nonzero result.
    let e_star_int = eval.add(lv.int_f.exp, lv.int_f.exp_is_zero);
    let e_star_scale = eval.add(lv.scale.exp, lv.scale.exp_is_zero);
    let c_off = eval.i32(MUL_LSB_OFFSET as i32);
    let e_sum = eval.add(e_star_int, e_star_scale);
    let base = eval.sub(e_sum, c_off);
    let target = eval.add(base, lv.mul_decode.width_adjust);
    let exp_diff = eval.sub(lv.x_exp, target);
    let not_xiz = eval.sub(one, lv.x_is_zero);
    let not_xeiz = eval.sub(one, lv.x_exp_is_zero);
    let gate = eval.mul(not_xiz, not_xeiz);
    let c = eval.mul(gate, exp_diff);
    eval.constraint(c);
    // U5 forces the stored exponent to zero on subnormal/zero results and the mantissa to zero
    // on an exact zero. The sign still follows the multiplication rule from P4.
    let c = eval.mul(lv.x_exp_is_zero, lv.x_exp);
    eval.constraint(c);
    let c = eval.mul(lv.x_is_zero, lv.x_exp);
    eval.constraint(c);
    let c = eval.mul(lv.x_is_zero, lv.x_mantissa);
    eval.constraint(c);
    // U6 (RC16(254 - X_EXP)) is a LUT instance.

    // Group S — validate alpha/beta and keep all row-level values group-constant.
    // S1-S3 are lookups that prove alpha is positive normal and beta is finite/nonnegative with
    // valid bf16 fields. S4 keeps alpha, beta, and the L2 frame exponent fixed across the group.
    for (f_next, f_local) in [
        (nv.alpha_exp, lv.alpha_exp),
        (nv.alpha_mantissa, lv.alpha_mantissa),
        (nv.beta_exp, lv.beta_exp),
        (nv.beta_mantissa, lv.beta_mantissa),
        (nv.beta_exp_is_zero, lv.beta_exp_is_zero),
        (nv.blk.frame_doubled_scale_exponent, lv.blk.frame_doubled_scale_exponent),
    ] {
        let diff = eval.sub(f_next, f_local);
        let c = eval.mul(not_gf, diff);
        eval.constraint(c);
    }

    // Group B — compute the canonical L2 sum from eight-element prequant blocks.
    // A row is block-final exactly when the next row starts a block. This also handles the
    // cyclic last-row-to-first-row boundary.
    let ibf = next_block_start;
    let not_ibf = not_bs;
    // B2 resets the running integer-square sum at a block start and otherwise adds the next
    // int8 square. The explicit first-row check anchors the cyclic recurrence.
    let int_sq_local = int_sq_expr(eval, lv.int8_byte, lv.int_f.sign);
    let int_sq_next = int_sq_expr(eval, nv.int8_byte, nv.int_f.sign);
    let anchor = eval.sub(lv.blk.block_int_squared_sum, int_sq_local);
    eval.constraint_first_row(anchor);
    let kept = eval.mul(not_ibf, lv.blk.block_int_squared_sum);
    let expected = eval.add(kept, int_sq_next);
    let c = eval.sub(nv.blk.block_int_squared_sum, expected);
    eval.constraint(c);
    // B3 multiplies the running integer-square sum by M(scale)^2. On the block-final row this
    // is p_b; P2 has already proved that the scale stayed constant through the block.
    let m_scale_sq = eval.mul(m_scale, m_scale);
    let p_expected = eval.mul(m_scale_sq, lv.blk.block_int_squared_sum);
    let c = eval.sub(lv.blk.block_l2_product, p_expected);
    eval.constraint(c);
    // B4 proves BLOCK_L2_PRODUCT_NONZERO iff the product is nonzero. The first equation gives
    // the forward direction through an inverse; the second forbids a false zero flag.
    let prod = eval.mul(lv.blk.block_l2_product, lv.blk.block_l2_product_inverse);
    let c = eval.sub(prod, lv.blk.block_l2_product_nonzero);
    eval.constraint(c);
    let not_pnz = eval.sub(one, lv.blk.block_l2_product_nonzero);
    let c = eval.mul(not_pnz, lv.blk.block_l2_product);
    eval.constraint(c);
    // B5 emits 2*E*(scale) only for a nonzero completed block; all other rows emit zero.
    let e2_scale = eval.add(e_star_scale, e_star_scale);
    let live_e2 = eval.mul(lv.blk.block_l2_product_nonzero, e2_scale);
    let cand_diff = eval.sub(lv.blk.block_doubled_scale_exponent, live_e2);
    let c = eval.mul(ibf, cand_diff);
    eval.constraint(c);
    let c = eval.mul(not_ibf, lv.blk.block_doubled_scale_exponent);
    eval.constraint(c);
    // B6 tracks the largest candidate in the group. At a group boundary it starts from the
    // next row's candidate; inside a group it must equal either the previous max or the new
    // candidate. Range lookups separately prove that it never decreases and dominates both.
    let e2_start_diff = eval.sub(nv.blk.running_max_doubled_scale_exponent, nv.blk.block_doubled_scale_exponent);
    let c = eval.mul(gf, e2_start_diff);
    eval.constraint(c);
    let e2_grow = eval.sub(
        nv.blk.running_max_doubled_scale_exponent,
        lv.blk.running_max_doubled_scale_exponent,
    );
    let attain = eval.mul(e2_start_diff, e2_grow);
    let c = eval.mul(not_gf, attain);
    eval.constraint(c);
    // B7 copies the completed running maximum into the group-constant frame exponent.
    let frame_diff = eval.sub(lv.blk.frame_doubled_scale_exponent, lv.blk.running_max_doubled_scale_exponent);
    let c = eval.mul(gf, frame_diff);
    eval.constraint(c);
    // B8: `is_far_shift` is boolean and block-final-only. Its ctl.rs range check proves the
    // frame shift is at least 54, so the below-2^54 dividend shifts to zero.
    eval.constraint_bool(lv.blk.is_far_shift);
    let c = eval.mul(not_ibf, lv.blk.is_far_shift);
    eval.constraint(c);
    // B10: decompose `block_l2_product * 2^Wl2` into four 16-bit
    // `scaled_block_product_limbs`; `2^Wl2` is the public input. The top-limb cap makes the below-2^54 field equality a unique
    // integer decomposition.
    let mut x_recomposed = lv.blk.scaled_block_product_limbs[0];
    for i in 1..4 {
        let w = eval.u64(1 << (16 * i));
        x_recomposed = eval.mad(lv.blk.scaled_block_product_limbs[i], w, x_recomposed);
    }
    let x_expected = eval.mul(wl2_pow, lv.blk.block_l2_product);
    let c = eval.sub(x_recomposed, x_expected);
    eval.constraint(c);
    // B11 supplies 2^r and 2^(16-r), where r is the shift within a 16-bit limb. B13 proves
    // exactly one limb selector is active on a nonzero, block-final, non-far row, and none is
    // active elsewhere. Their sum below is the near-live gate.
    for q in lv.blk.shift_limb_selector {
        eval.constraint_bool(q);
    }
    let nl = {
        let s01 = eval.add(lv.blk.shift_limb_selector[0], lv.blk.shift_limb_selector[1]);
        let s23 = eval.add(lv.blk.shift_limb_selector[2], lv.blk.shift_limb_selector[3]);
        eval.add(s01, s23)
    };
    let not_far = eval.sub(one, lv.blk.is_far_shift);
    let live_gate = eval.mul(lv.blk.block_l2_product_nonzero, not_far);
    let nl_expected = eval.mul(ibf, live_gate);
    let c = eval.sub(nl, nl_expected);
    eval.constraint(c);
    // B14 divides the selected limb by 2^r into quotient and remainder. Rows that emit no
    // block term force both outputs to zero.
    let l_sel = {
        let t0 = eval.mul(lv.blk.shift_limb_selector[0], lv.blk.scaled_block_product_limbs[0]);
        let t1 = eval.mul(lv.blk.shift_limb_selector[1], lv.blk.scaled_block_product_limbs[1]);
        let t2 = eval.mul(lv.blk.shift_limb_selector[2], lv.blk.scaled_block_product_limbs[2]);
        let t3 = eval.mul(lv.blk.shift_limb_selector[3], lv.blk.scaled_block_product_limbs[3]);
        let s01 = eval.add(t0, t1);
        let s23 = eval.add(t2, t3);
        eval.add(s01, s23)
    };
    let tl_scaled = eval.mul(lv.blk.selected_limb_quotient, lv.blk.shift_remainder_power);
    let split = eval.add(tl_scaled, lv.blk.selected_limb_remainder);
    let split_diff = eval.sub(l_sel, split);
    let c = eval.mul(nl, split_diff);
    eval.constraint(c);
    let not_nl = eval.sub(one, nl);
    let c = eval.mul(not_nl, lv.blk.selected_limb_quotient);
    eval.constraint(c);
    let c = eval.mul(not_nl, lv.blk.selected_limb_remainder);
    eval.constraint(c);
    // B15 lookup checks make that Euclidean division unique: the quotient is 16-bit and the
    // remainder is strictly below 2^r.
    // B16 recomposes floor(X/2^sigma): H contains all limbs above the selected one, while the
    // selected-limb quotient supplies its surviving high bits. The running sum adds one such
    // term per completed near block; far and non-final rows add zero. Row 0 is mid-block and
    // therefore has a zero term, so a simple zero anchor closes the cyclic accumulator without
    // increasing the constraint degree.
    eval.constraint_first_row(lv.blk.running_l2_frame_sum);
    let term_next = {
        let h = blk_high_limbs(eval, &nv.blk);
        let hc = eval.mul(h, nv.blk.shift_complement_power);
        eval.add(hc, nv.blk.selected_limb_quotient)
    };
    let kept = eval.mul(not_gf, lv.blk.running_l2_frame_sum);
    let expected = eval.add(kept, term_next);
    let c = eval.sub(nv.blk.running_l2_frame_sum, expected);
    eval.constraint(c);

    // Group F — compute linf's source, the largest decoded |X| in the group.
    // For nonnegative finite bf16 codes, EXP*128 + MANTISSA is ordered by magnitude.
    let abs_next = eval.mad(nv.x_exp, c128, nv.x_mantissa);
    // F1 starts each group from its first element, including across the cyclic trace boundary.
    let start_diff = eval.sub(nv.max_abs, abs_next);
    let c = eval.mul(gf, start_diff);
    eval.constraint(c);
    // F2 says the next max is either the old max or the new element.
    let grow = eval.sub(nv.max_abs, lv.max_abs);
    let attain = eval.mul(start_diff, grow);
    let c = eval.mul(not_gf, attain);
    eval.constraint(c);
    // F3 range lookups prove the chosen value dominates both inputs and never decreases.

    // Group M — compute noise_term = RNE_bf16(beta*n).
    // M1 binds the exact significand product. The verifier supplies n's decoded fields, which
    // may represent a subnormal.
    let m_beta = m_expr(eval, lv.beta_exp_is_zero, lv.beta_mantissa);
    let m_noise = m_expr(eval, lv.noise.exp_is_zero, lv.noise.mantissa);
    let sig = eval.mul(m_beta, m_noise);
    let c = eval.sub(lv.mul_beta_n.sig_product, sig);
    eval.constraint(c);
    // M2 (CLAMP22), M3 (RNERND), M6 (RC16) are LUT instances; M4/M5 below.
    let e_star_beta = eval.add(lv.beta_exp, lv.beta_exp_is_zero);
    let e_star_noise = eval.add(lv.noise.exp, lv.noise.exp_is_zero);
    eval_mul_out(eval, &lv.mul_beta_n, e_star_beta, e_star_noise);

    // Group W — prove `NOISED = RNE_bf16(alpha*X + noise_term)` with one final rounding.
    let f = &lv.fma;
    // W1: bind the exact product significand M(alpha)*M(X) (alpha is normal: M = 128 + mantissa).
    let m_x = m_expr(eval, lv.x_exp_is_zero, lv.x_mantissa);
    let m_alpha = eval.add(c128, lv.alpha_mantissa);
    let prod = eval.mul(m_alpha, m_x);
    let c = eval.sub(lv.fma_sig_product, prod);
    eval.constraint(c);
    // Binary scales of the two exact integer significands:
    //   alpha*X    = (+/-M(alpha)*M(X)) * 2^EP
    //   noise_term = (+/-M(noise_term)) * 2^EC.
    // Alpha is normal, so EP = ALPHA_EXP + E*(X) - 268 when X is nonzero. A zero
    // product has no meaningful scale; subtracting 400 (larger than the maximum legal gap,
    // 386) forces the addend-preserving far path. U3 already proves X_IS_ZERO iff M(X) = 0.
    let e_star_x = eval.add(lv.x_exp, lv.x_exp_is_zero);
    let ep_base = eval.add(lv.alpha_exp, e_star_x);
    let c_off = eval.i32(MUL_LSB_OFFSET as i32);
    let ep_raw = eval.sub(ep_base, c_off);
    let c_drop = eval.i32(ZERO_PRODUCT_SCALE_DROP as i32);
    let drop = eval.mul(c_drop, lv.x_is_zero);
    let ep = eval.sub(ep_raw, drop);
    let e_star_nt = eval.add(lv.mul_beta_n.out_exp, lv.mul_beta_n.out_exp_is_zero);
    let c134 = eval.i32(134);
    let ec = eval.sub(e_star_nt, c134);
    let m_c = m_expr(eval, lv.mul_beta_n.out_exp_is_zero, lv.mul_beta_n.out_mantissa);
    // W2: prove which operand has the coarser binary unit and record the gap exactly.
    // GE = 1 means EP >= EC and SCALE_GAP_SLACK = EP - EC = d.
    // GE = 0 means EC > EP and SCALE_GAP_SLACK = EC - EP - 1 = d - 1.
    // Subtracting one in the second branch makes the branches disjoint at EP = EC.
    eval.constraint_bool(f.product_scale_ge_addend);
    let not_ge = eval.sub(one, f.product_scale_ge_addend);
    let ep_m_ec = eval.sub(ep, ec);
    let ec_m_ep = eval.sub(ec, ep);
    let ec_m_ep_m1 = eval.sub(ec_m_ep, one);
    let ge_term = eval.mul(f.product_scale_ge_addend, ep_m_ec);
    let nge_term = eval.mul(not_ge, ec_m_ep_m1);
    let slack_target = eval.add(ge_term, nge_term);
    let c = eval.sub(f.scale_gap_slack, slack_target);
    eval.constraint(c);
    // W3: IS_FAR_GAP may be set only when the smaller operand is sticky-only. Since a far cap
    // is one below its threshold:
    //   GE = 1: M_C < 2^8, so cap 9 gives M_C/2^9 < 1/2 product unit; d >= 10 is smaller still.
    //   GE = 0: the worst subtraction is below M_C = 128, whose lower bf16 neighbor is 127.5;
    //           its midpoint is 127.75, only 1/4 addend unit away. Cap 19 gives
    //           M_P/2^19 < 2^16/2^19 = 1/8 < 1/4, retaining one safety bit.
    // Hence the thresholds are 10 and 20. FAR_GAP_SLACK records d - threshold.
    eval.constraint_bool(f.is_far_gap);
    let c9 = eval.i32(9);
    let c19 = eval.i32(19);
    let d_m_thr = eval.mad(c9, f.product_scale_ge_addend, f.scale_gap_slack);
    let d_m_thr = eval.sub(d_m_thr, c19);
    let far_diff = eval.sub(d_m_thr, f.far_gap_slack);
    let c = eval.mul(f.is_far_gap, far_diff);
    eval.constraint(c);
    // W4: near rows use the true gap d. Far rows use the last gap below the threshold:
    // 9 when GE = 1, or 19 when GE = 0. W10 restores the omitted scale distance, so the
    // dominant operand remains exact while the smaller operand becomes a sticky nudge.
    let not_far = eval.sub(one, f.is_far_gap);
    let d = eval.add(f.scale_gap_slack, one);
    let d = eval.sub(d, f.product_scale_ge_addend);
    let near_diff = eval.sub(f.exp_gap_capped, d);
    let c = eval.mul(not_far, near_diff);
    eval.constraint(c);
    let c10 = eval.i32(10);
    let ge10 = eval.mul(c10, f.product_scale_ge_addend);
    let thr_m1 = eval.sub(c19, ge10);
    let far_cap = eval.sub(f.exp_gap_capped, thr_m1);
    let c = eval.mul(f.is_far_gap, far_cap);
    eval.constraint(c);
    // W5: express both magnitudes in one common unit. The operand with the larger exponent
    // has the coarser unit, so multiply its integer significand by 2^EXP_GAP_CAPPED.
    let pow2_m1 = eval.sub(f.exp_gap_pow2, one);
    let sel_p = eval.mad(f.product_scale_ge_addend, pow2_m1, one); // GE*POW2 + (1 - GE)
    let al_p_target = eval.mul(lv.fma_sig_product, sel_p);
    let c = eval.sub(f.aligned_product, al_p_target);
    eval.constraint(c);
    let sel_c = eval.mad(not_ge, pow2_m1, one); // (1 - GE)*POW2 + GE
    let al_c_target = eval.mul(m_c, sel_c);
    let c = eval.sub(f.aligned_addend, al_c_target);
    eval.constraint(c);
    // W6: add/subtract the aligned integers exactly and commit the result as
    // (-1)^OUT_SIGN * FOLDED_MAGNITUDE. Alpha and beta are nonnegative, so the product sign is
    // X_SIGN and the addend sign is NOISE_SIGN. OUT_SIGN must be a genuine bit before it is
    // used in the `1 - 2*OUT_SIGN` sign factor.
    let s_out = sign_factor(eval, f.out_sign);
    let lhs = eval.mul(s_out, f.folded_magnitude);
    let s_p = sign_factor(eval, lv.x_sign);
    let s_c = sign_factor(eval, lv.noise.sign);
    let c_term = eval.mul(s_c, f.aligned_addend);
    let rhs = eval.mad(s_p, f.aligned_product, c_term);
    let c = eval.sub(lhs, rhs);
    eval.constraint(c);
    eval.constraint_bool(f.out_sign);
    // W7: prove FOLDED_IS_ZERO iff FOLDED_MAGNITUDE is zero. On exact cancellation, use the
    // IEEE zero-sign rule: only (-0) + (-0) produces -0; opposite signs produce +0.
    let v_prod = eval.mul(f.folded_magnitude, f.folded_inverse);
    let not_zr = eval.sub(one, f.folded_is_zero);
    let c = eval.sub(v_prod, not_zr);
    eval.constraint(c);
    let c = eval.mul(f.folded_is_zero, f.folded_magnitude);
    eval.constraint(c);
    let sign_prod = eval.mul(lv.x_sign, lv.noise.sign);
    let sign_diff = eval.sub(f.out_sign, sign_prod);
    let c = eval.mul(f.folded_is_zero, sign_diff);
    eval.constraint(c);
    // W8: RNERND accepts a significand below 2^17. Narrow rows therefore use shift 0 and
    // power 1. Wide rows prove the Euclidean split
    //   V = COMPRESSION_QUOTIENT * 2^COMPRESSION_SHIFT + COMPRESSION_REMAINDER.
    // The quotient is range-checked in [2^16, 2^17), which uniquely forces
    // COMPRESSION_SHIFT = bit_length(V) - 17.
    eval.constraint_bool(f.is_wide);
    let not_wide = eval.sub(one, f.is_wide);
    let c = eval.mul(not_wide, f.compression_shift);
    eval.constraint(c);
    let sp_m1 = eval.sub(f.shift_pow2, one);
    let c = eval.mul(not_wide, sp_m1);
    eval.constraint(c);
    let head = eval.mul(f.compression_quotient, f.shift_pow2);
    let split = eval.sub(f.folded_magnitude, head);
    let split = eval.sub(split, f.compression_remainder);
    let c = eval.mul(f.is_wide, split);
    eval.constraint(c);
    // W9: STICKY is 1 exactly when compression discarded a nonzero remainder. The quotient's
    // least-significant bit records whether its retained 17-bit head is already odd.
    let r_prod = eval.mul(f.compression_remainder, f.compression_remainder_inverse);
    let c = eval.sub(r_prod, f.sticky);
    eval.constraint(c);
    let not_sticky = eval.sub(one, f.sticky);
    let c = eval.mul(not_sticky, f.compression_remainder);
    eval.constraint(c);
    eval.constraint_bool(f.compression_quotient_lsb);
    // W10: choose the significand sent to RNERND. Narrow rows send V unchanged. Wide rows use
    // round-to-odd: keep the quotient if it is already odd or exact, otherwise add one.
    let key_diff = eval.sub(f.rounding_significand_key, f.folded_magnitude);
    let c = eval.mul(not_wide, key_diff);
    eval.constraint(c);
    let not_k0 = eval.sub(one, f.compression_quotient_lsb);
    let bump = eval.mul(f.sticky, not_k0);
    let rto = eval.sub(f.rounding_significand_key, f.compression_quotient);
    let rto = eval.sub(rto, bump);
    let c = eval.mul(f.is_wide, rto);
    eval.constraint(c);
    // The RNERND lookup packs `significand + 2^17*cut`, so that packed key alone does not prove
    // the significand is below 2^17: an oversized significand could alias the next cut slot.
    // This boolean high bit plus the RC16 split in `super::ctl` supplies the missing bound.
    eval.constraint_bool(f.rounding_significand_key_high_bit);
    // Reconstruct the binary scale of the RNERND key. Start at the finer input scale
    // min(EP, EC). A far row raises that scale by the part of the gap omitted by W4; a wide row
    // raises it again by the number of bits removed in W8.
    let base = eval.mux(f.product_scale_ge_addend, ep, ec);
    let d_m_capped = eval.sub(d, f.exp_gap_capped);
    let far_corr = eval.mul(f.is_far_gap, d_m_capped);
    let wide_corr = eval.mul(f.is_wide, f.compression_shift);
    let ks_target = eval.add(base, far_corr);
    let ks_target = eval.add(ks_target, wide_corr);
    let c = eval.sub(f.key_scale, ks_target);
    eval.constraint(c);
    // W11 is lookup-backed: CLAMP22 derives the subnormal cut from KEY_SCALE, and RNERND
    // returns the rounded mantissa, exponent adjustment, and zero/subnormal flags.
    // W12 binds a normal nonzero output exponent to KEY_SCALE + WIDTH_ADJUST. The remaining
    // equations force exponent and mantissa fields to zero on the corresponding zero paths.
    let not_oiz = eval.sub(one, f.out_is_zero);
    let not_oeiz = eval.sub(one, f.out_exp_is_zero);
    let gate = eval.mul(not_oiz, not_oeiz);
    let target = eval.add(f.key_scale, f.width_adjust);
    let exp_diff = eval.sub(f.out_exp, target);
    let c = eval.mul(gate, exp_diff);
    eval.constraint(c);
    let c = eval.mul(f.out_exp_is_zero, f.out_exp);
    eval.constraint(c);
    let c = eval.mul(f.out_is_zero, f.out_exp);
    eval.constraint(c);
    let c = eval.mul(f.out_is_zero, f.out_mantissa);
    eval.constraint(c);

    // Group Q — the lookup-backed fp8 cast: QCAST returns the E4M3 code (byte range
    // included).

    // Group C — the jackpot liveness flag `IS_DEAD = [ABS(X) >= DEAD_BOUND]` and its
    // in-group count (the group-final value rides ScaleStark's group-tuple CTL channel).
    // C1: IS_DEAD is boolean and zero on phantom rows. The RC16 certificate pair in ctl.rs is
    // filtered by IS_DEAD resp. (1 - IS_DEAD)*LIVE, so on live rows exactly one certificate
    // fires and pins the flag; booleanness is what makes those filters sound.
    eval.constraint_bool(lv.is_dead);
    let not_live = eval.sub(one, lv.live);
    let c = eval.mul(not_live, lv.is_dead);
    eval.constraint(c);
    // C2: DEAD_BOUND is group-constant. The group-final value rides the Scale tuple, where T1
    // binds it to the floored-L2 claim, so every row's certificate uses the true bound.
    let bound_diff = eval.sub(nv.dead_bound, lv.dead_bound);
    let c = eval.mul(not_gf, bound_diff);
    eval.constraint(c);
    // C3: the running dead count resets at a group boundary and otherwise adds the next row's
    // flag. The explicit first-row anchor closes the cyclic recurrence.
    let anchor = eval.sub(lv.dead_count, lv.is_dead);
    eval.constraint_first_row(anchor);
    let kept = eval.mul(not_gf, lv.dead_count);
    let expected = eval.add(kept, nv.is_dead);
    let c = eval.sub(nv.dead_count, expected);
    eval.constraint(c);

    // Group V: the summand score for S_X = (alpha*X)^2 + sigma^2.
    // Use the following names for the committed biased exponents and significands:
    //
    //   E_x = scaled_biased_exponent       n_x = scaled_significand
    //   E_s = sigma_biased_exponent        n_s = normalized_sigma_significand
    //   |alpha*X| = n_x * 2^(E_x - 268 - 15)
    //   sigma     = n_s * 2^(E_s - 268 - 15)
    //
    // Nonzero significands lie in [2^15, 2^16). Zero X uses E_x = n_x = 0.
    // `unpredictability` derives the score's error bound; V1–V11 enforce its integer rule.
    //
    // V1: let w = bit_length(M(alpha)*M(X)), supplied by WIDTH16. For nonzero X:
    //
    //   E_x = ALPHA_EXP + E*(X) - 1 + w.
    //
    // WIDTH16 also binds the product-nonzero flag. Zero X has w = 0 and uses sentinel E_x = 0.
    let not_xz = eval.sub(one, lv.x_is_zero);
    let enc_base = eval.add(lv.alpha_exp, e_star_x);
    let enc_base = eval.sub(enc_base, one);
    let enc_gated = eval.mul(not_xz, enc_base);
    let x_enc_target = eval.add(enc_gated, lv.scaled_product_width);
    let c = eval.sub(lv.scaled_biased_exponent, x_enc_target);
    eval.constraint(c);
    // V2/V3: keep sigma's exponent and normalized significand group-constant.
    // The group-final tuple binds them to Scale. Padding scores are filtered from exports.
    let sig_diff = eval.sub(nv.sigma_biased_exponent, lv.sigma_biased_exponent);
    let c = eval.mul(not_gf, sig_diff);
    eval.constraint(c);
    let sig_norm_diff = eval.sub(nv.normalized_sigma_significand, lv.normalized_sigma_significand);
    let c = eval.mul(not_gf, sig_norm_diff);
    eval.constraint(c);
    // V4: normalize M(alpha)*M(X) to 16 bits using the POW2D shift 16 - product width.
    // The normalized significand is in [2^15, 2^16) for nonzero X, otherwise zero.
    let norm_target = eval.mul(lv.fma_sig_product, lv.scaled_normalization_power);
    let c = eval.sub(lv.scaled_significand, norm_target);
    eval.constraint(c);
    // V5: X_DOMINATES selects the addend with the larger exponent:
    //
    //   exponent_gap = (2*X_DOMINATES - 1)*(E_x - E_s) = |E_x - E_s|.
    //
    // RC16(exponent_gap) forces the choice when E_x != E_s: the wrong bit gives a
    // negative range-check key. On live rows, zero X has E_x = 0 < E_s, so sigma
    // dominates and the gap is far. At equal exponents, either ordering gives the same score.
    eval.constraint_bool(lv.x_dominates);
    let two_b = eval.add(lv.x_dominates, lv.x_dominates);
    let sign = eval.sub(two_b, one);
    let enc_diff = eval.sub(lv.scaled_biased_exponent, lv.sigma_biased_exponent);
    let gap_target = eval.mul(sign, enc_diff);
    let c = eval.sub(lv.exponent_gap, gap_target);
    eval.constraint(c);
    // V6: GAP_IS_FAR = [EXPONENT_GAP >= 16], the far cut — exact, the dropped quotient is
    // zero at any such gap. Boolean here; each direction is a filtered RC16 (ctl.rs): key
    // `EXPONENT_GAP - 16` under the bit, key `15 - EXPONENT_GAP` under its complement.
    eval.constraint_bool(lv.gap_is_far);
    // V7: NEAR_GAP = (1 - GAP_IS_FAR) * EXPONENT_GAP — the POW2D key, in [0, 15], pinning
    // NEAR_GAP_POW = 2^NEAR_GAP (ctl.rs).
    let not_far = eval.sub(one, lv.gap_is_far);
    let near_target = eval.mul(not_far, lv.exponent_gap);
    let c = eval.sub(lv.near_gap, near_target);
    eval.constraint(c);
    // V8: order normalized significands by exponent.
    let big_target = eval.mux(lv.x_dominates, lv.normalized_sigma_significand, lv.scaled_significand);
    let c = eval.sub(lv.dominant_significand, big_target);
    eval.constraint(c);
    let small_target = eval.mux(lv.x_dominates, lv.scaled_significand, lv.normalized_sigma_significand);
    let c = eval.sub(lv.smaller_significand, small_target);
    eval.constraint(c);
    // V9: the two exact floor stages, dividing SMALLER_SIGNIFICAND^2 by 2^NEAR_GAP twice:
    //
    //   SMALLER_SIGNIFICAND^2  = HALF_QUOTIENT * NEAR_GAP_POW + HALF_REMAINDER
    //   HALF_QUOTIENT = QUOTIENT * NEAR_GAP_POW + REMAINDER
    //
    // Each remainder is pinned into [0, NEAR_GAP_POW) by its RC16 pair (ctl.rs), making both
    // quotients exact floors.
    //
    // No identity can wrap the field. V10 keeps QUOTIENT's signed value inside
    // (-2^32, 2^33 + 2^17). This bounds HALF_QUOTIENT to (-2^47, 2^49) and each
    // product to (-2^62, 2^63 + 2^48). The difference between the two sides of either
    // identity lies in (-p, p), so equality in the field implies integer equality.
    let small_sq = eval.mul(lv.smaller_significand, lv.smaller_significand);
    let stage1 = eval.mad(lv.half_quotient, lv.near_gap_pow, lv.half_remainder);
    let c = eval.sub(small_sq, stage1);
    eval.constraint(c);
    let stage2 = eval.mad(lv.quotient, lv.near_gap_pow, lv.remainder);
    let c = eval.sub(lv.half_quotient, stage2);
    eval.constraint(c);
    // V10: split the aligned sum of squares into its top slice and low 17 bits:
    //
    //   V = dominant_significand^2 + (1 - gap_is_far)*quotient
    //     = 2^17*sum_of_squares_top + rest
    //   rest = sum_of_squares_rest_lo + 2^16*sum_of_squares_rest_hi.
    //
    // RC16 and a boolean high bit force 0 <= rest < 2^17, so the top slice is
    // floor(V / 2^17). LOG16 bounds that slice and returns
    // log_fraction = floor(64*log2(sum_of_squares_top)).
    eval.constraint_bool(lv.sum_of_squares_rest_hi);
    let big_sq = eval.mul(lv.dominant_significand, lv.dominant_significand);
    let kept_quotient = eval.mul(not_far, lv.quotient);
    let sum_of_squares = eval.add(big_sq, kept_quotient);
    let c2_17 = eval.u64(1 << 17);
    let top_term = eval.mul(lv.sum_of_squares_top, c2_17);
    let c2_16 = eval.u64(1 << 16);
    let hi_term = eval.mul(lv.sum_of_squares_rest_hi, c2_16);
    let rest = eval.add(lv.sum_of_squares_rest_lo, hi_term);
    let split = eval.add(top_term, rest);
    let c = eval.sub(sum_of_squares, split);
    eval.constraint(c);
    // V11: convert the aligned logarithm back to the score of S_X:
    //
    //   score = LIVE * (128*max(E_x, E_s) - 2624 + log_fraction)
    //   -2624 = 64*(508 - 13) - 128*268.
    //
    // Squaring the normalized significands introduces 30 fractional bits; taking
    // V's top slice removes 17, leaving the correction -13. The other terms add
    // the score bias 508 and remove the exponent bias 268 twice. Padding scores are zero.
    let enc_big = eval.mux(lv.x_dominates, lv.sigma_biased_exponent, lv.scaled_biased_exponent);
    let enc_term = eval.mul(c128, enc_big);
    let c2624 = eval.i32(2624);
    let biased = eval.sub(enc_term, c2624);
    let score = eval.add(biased, lv.log_fraction);
    let lambda_target = eval.mul(lv.live, score);
    let c = eval.sub(lv.lambda, lambda_target);
    eval.constraint(c);
}

/// Evaluates every arithmetic constraint of InputQuantStark. Range checks, decode tables, and
/// rounding tables are declared separately in `super::ctl`. The job's L2 precision scale
/// `2^Wl2` enters as the [`WL2_POW_PUBLIC_INPUT`] public input.
pub(crate) fn eval_input_quant_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_INPUT_QUANT_COLUMNS, NUM_INPUT_QUANT_PUBLIC_INPUTS>,
    eval: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv: &[V; NUM_INPUT_QUANT_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &InputQuantColumnsView<V> = lv.borrow();
    let nv: &[V; NUM_INPUT_QUANT_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let nv: &InputQuantColumnsView<V> = nv.borrow();
    let wl2_pow = eval.scalar(vars.get_public_inputs()[WL2_POW_PUBLIC_INPUT]);

    // Verifier-known columns need no duplicate AIR equations. The verifier recomputes the
    // noise fields, indices, liveness, and row/block/group flags and checks their openings.
    // This proves the boundary facts used below: the final row closes a block and group,
    // even/odd rows alternate from row 0, and each block start has
    // ELEMENT_INDEX = 8*BLOCK_INDEX.

    // Both sides (groups P/U/V/S/B/F/M/W/Q/C).
    let lva = side_a(lv);
    let nva = side_a(nv);
    eval_side(eval, &lva, &nva, lv.is_group_final, nv.is_block_start, wl2_pow);
    let lvb = side_b(lv);
    let nvb = side_b(nv);
    eval_side(eval, &lvb, &nvb, lv.is_group_final, nv.is_block_start, wl2_pow);
}

// Stark impl

/// Input-quantization AIR. The batch driver is the supported proving path.
///
/// The arithmetic constraints require the cross-table channels and committed
/// lookup tables declared in [`super::ctl`]. A standalone proof of this table
/// would not establish those relations.
#[derive(Clone, Debug)]
pub struct InputQuantStark<F: RichField + Extendable<D>, const D: usize> {
    pub program: InputQuantProgram,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> InputQuantStark<F, D> {
    pub fn new(program: InputQuantProgram) -> Self {
        Self {
            program,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for InputQuantStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_INPUT_QUANT_COLUMNS, NUM_INPUT_QUANT_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_INPUT_QUANT_COLUMNS, NUM_INPUT_QUANT_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_input_quant_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_input_quant_constraints(vars, &mut evaluator);
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    // Party to the strip-bytes, block-scales, operand-codes and group-tuples channels, plus
    // the committed LUT channels.
    fn requires_ctls(&self) -> bool {
        true
    }
}

// Tests

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};
    use starky::util::trace_rows_to_poly_values;

    use super::super::columns::INPUT_QUANT_COL_MAP;
    use super::super::ctl::input_quant_lut_lookups;
    use super::*;
    use crate::api::fp8::prequant::open_prequant;
    use crate::circuit::fp8::ctl::{LutLookup, LutTable};
    use crate::circuit::fp8::unpredictability::log2_fixed;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type S = InputQuantStark<F, D>;

    fn test_program() -> InputQuantProgram {
        // 4 groups x 64 elements = 256 rows.
        InputQuantProgram {
            h: 4,
            w: 4,
            k: 64,
            block_size: BLOCK_SIZE,
            r: 16,
        }
    }

    /// Deterministic int8 plane: mixes zeros, ±127, -128 and mid-range values.
    fn test_int8(len: usize, salt: u64) -> Vec<i8> {
        const POOL: [i8; 12] = [5, -3, 127, 0, 1, -1, 64, -64, 100, -100, -128, 33];
        (0..len)
            .map(|i| {
                if i % 9 == 0 {
                    0
                } else {
                    POOL[((i as u64).wrapping_mul(salt) ^ (i as u64 >> 2)) as usize % POOL.len()]
                }
            })
            .collect()
    }

    /// Deterministic scale plane containing signed normal and subnormal scales and one
    /// zero-scale block per group. The zero block exercises the sum-of-squares, policy, and
    /// zero-product FMA paths; the subnormal scales decode elements into bf16's subnormal
    /// range (and, with large ints, back into the normal range through rounding).
    fn test_scales(len: usize, salt: u64) -> Vec<u16> {
        // 1.0, 2.0, 0.5, -1.5, 3.0, 0.25, -0.375, 1.25, 2^-133, -2^-127 as bf16 codes.
        const POOL: [u16; 10] = [0x3F80, 0x4000, 0x3F00, 0xBFC0, 0x4040, 0x3E80, 0xBEC0, 0x3FA0, 0x0001, 0x8040];
        (0..len)
            .map(|i| {
                if i % 8 == 1 {
                    0x0000 // one zero-scale block per 8 blocks
                } else {
                    POOL[((i as u64).wrapping_mul(salt) ^ (i as u64 >> 1)) as usize % POOL.len()]
                }
            })
            .collect()
    }

    /// Deterministic noise plane: zero, subnormal, tiny and mid-range bf16 codes of both signs —
    /// exercises far/near gaps, wide/narrow folds and subnormal noise terms.
    fn test_noise(len: usize, salt: u64) -> Vec<u16> {
        const POOL: [u16; 10] = [
            0x3E80, // 0.25
            0xBE80, // -0.25
            0x0000, // +0
            0x0040, // subnormal
            0x4100, // 8.0
            0xB880, // -2^-14
            0x2000, // 2^-63
            0x8010, // -subnormal
            0x3F80, // 1.0
            0xC280, // -64.0
        ];
        (0..len)
            .map(|i| POOL[((i as u64).wrapping_mul(salt) ^ (i as u64 >> 3)) as usize % POOL.len()])
            .collect()
    }

    struct TestInputs {
        a_int8: Vec<i8>,
        a_scales: Vec<u16>,
        a_noise: Vec<u16>,
        b_int8: Vec<i8>,
        b_scales: Vec<u16>,
        b_noise: Vec<u16>,
    }

    fn test_inputs(program: &InputQuantProgram) -> TestInputs {
        let (h, w, k) = (program.h, program.w, program.k);
        TestInputs {
            a_int8: test_int8(h * k, 0x9E3779B97F4A7C15),
            a_scales: test_scales(h * k / BLOCK_SIZE, 0xC2B2AE3D27D4EB4F),
            a_noise: test_noise(h * k, 0xD1B54A32D192ED03),
            b_int8: test_int8(w * k, 0xA0761D6478BD642F),
            b_scales: test_scales(w * k / BLOCK_SIZE, 0xE7037ED1A0B428DB),
            b_noise: test_noise(w * k, 0x2545F4914F6CDD1D),
        }
    }

    fn test_trace() -> (
        InputQuantProgram,
        Vec<[F; NUM_INPUT_QUANT_COLUMNS]>,
        [F; NUM_INPUT_QUANT_PUBLIC_INPUTS],
    ) {
        let program = test_program();
        let inp = test_inputs(&program);
        let (rows, pis) = program.generate_trace::<F>(
            &inp.a_int8,
            &inp.a_scales,
            &inp.a_noise,
            &inp.b_int8,
            &inp.b_scales,
            &inp.b_noise,
        );
        (program, rows, pis)
    }

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    /// Runs the constraint set over every row pair (including the last -> first wrap) and
    /// returns whether every accumulator vanished.
    ///
    /// `z_last` is zero on the last pair, matching starky's semantics. This AIR has no
    /// transition-only constraints, so every ordinary constraint—including each cyclic
    /// accumulator's last-row-to-first-row boundary—is evaluated on the wrap pair.
    fn constraints_vanish(stark: &S, rows: &[[F; NUM_INPUT_QUANT_COLUMNS]], pis: &[F; NUM_INPUT_QUANT_PUBLIC_INPUTS]) -> bool {
        let n = rows.len();
        for i in 0..n {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], pis);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::from_bool(i != n - 1),
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            if consumer.accumulators().into_iter().any(|acc| acc != F::ZERO) {
                return false;
            }
        }
        true
    }

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        let (program, rows, pis) = test_trace();
        let stark = S::new(program);
        let n = rows.len();
        for i in 0..n {
            // Ordinary constraints also hold on the last-row-to-first-row wrap. The verifier-
            // known schedule marks the last row group-final, so this pair correctly resets the
            // group-0 accumulators and running maximum.
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], &pis);
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
    fn padded_asymmetric_trace_satisfies_constraints_and_lut_domains() {
        // Unequal live prefixes and k = 96 exercise both-side padding and a truncated
        // tail group. The final row must close that group for cyclic constraints.
        let program = InputQuantProgram {
            h: 3,
            w: 5,
            k: 96,
            block_size: BLOCK_SIZE,
            r: 16,
        };
        let inp = test_inputs(&program);
        let (rows, pis) = program.generate_trace::<F>(
            &inp.a_int8,
            &inp.a_scales,
            &inp.a_noise,
            &inp.b_int8,
            &inp.b_scales,
            &inp.b_noise,
        );
        assert_eq!(rows.len(), 512, "next_power_of_two(5 * 96)");
        let stark = S::new(program.clone());
        assert!(constraints_vanish(&stark, &rows, &pis), "padded trace violates a constraint");

        // Every verifier-known column, including liveness, matches independent recomputation.
        let known = program.known_values::<F>(&inp.a_noise, &inp.b_noise);
        assert_eq!(known.len(), NUM_INPUT_QUANT_KNOWN_COLUMNS);
        for (c, poly) in known.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(row[c], poly.values[r], "known column {c} differs at row {r}");
            }
        }
        let m = &INPUT_QUANT_COL_MAP;
        assert_eq!(rows[287][m.a_live], F::ONE);
        assert_eq!(rows[288][m.a_live], F::ZERO);
        assert_eq!(rows[479][m.b_live], F::ONE);
        assert_eq!(rows[480][m.b_live], F::ZERO);
        assert_eq!(rows[511][m.is_group_final], F::ONE, "the truncated tail must close");

        // Every declared LUT instance stays in its table's domain on the padded trace (the
        // phantom fill's derived fields and keys must be as legal as live ones — unfiltered
        // instances run on dead rows too).
        let polys = trace_rows_to_poly_values(rows.clone());
        let lookups = input_quant_lut_lookups::<F>();
        for (li, lookup) in lookups.iter().enumerate() {
            for r in 0..rows.len() {
                let filter = lookup.filter.eval_table(&polys, r, &[]);
                assert!(
                    filter == F::ZERO || filter == F::ONE,
                    "lookup {li} row {r}: non-boolean filter"
                );
                if filter == F::ZERO {
                    continue;
                }
                let keys: Vec<u64> = lookup.keys.iter().map(|c| to_u64(c.eval_table(&polys, r, &[]))).collect();
                let values: Vec<u64> = lookup.values.iter().map(|c| to_u64(c.eval_table(&polys, r, &[]))).collect();
                check_lut_semantics(lookup, &keys, &values, li, r);
            }
        }
    }

    #[test]
    fn trace_is_bit_exact_vs_native_pipeline() {
        // Compare decoded X, noised codes and liveness flags against their native functions.
        let (program, rows, _) = test_trace();
        let inp = test_inputs(&program);
        let (h, k) = (program.h, program.k);
        let opened_a = open_prequant(&inp.a_int8, &inp.a_scales, h, k, BLOCK_SIZE).unwrap();
        let opened_b = open_prequant(&inp.b_int8, &inp.b_scales, program.w, k, BLOCK_SIZE).unwrap();
        // Per-group floored L2 norms (the trace's exact-integer L2 coincides with the native
        // f32-summation row_norms on this data; `group_scales_match_native_derivation` asserts
        // that).
        let l2f = |opened: &[u16], g: usize| {
            bf16_max(
                Fp8E4M3Quant.row_norms(&opened[g * k..(g + 1) * k]).unwrap().0,
                NORM_FLOOR_CODE,
            )
        };
        // The plaintext dead bound and predicate, exactly as `JackpotPolicy::liveness_ok`
        // (`tau_idle = 8`, `dead iff |X| >= tau_idle * DELTA * l2f`, all in f64).
        let native_dead = |opened: &[u16], g: usize, x_code: u16| {
            f64::from(bf16_to_f32(x_code).abs()) >= 8.0 * DELTA * f64::from(bf16_to_f32(l2f(opened, g)))
        };

        for (r, row) in rows.iter().enumerate() {
            let v: &InputQuantColumnsView<F> = row.borrow();
            let g = r / k;
            for (side, opened, noise) in [(&side_a(v), &opened_a, &inp.a_noise), (&side_b(v), &opened_b, &inp.b_noise)] {
                let x_code = (to_u64(side.x_sign) << 15 | to_u64(side.x_exp) << 7 | to_u64(side.x_mantissa)) as u16;
                assert_eq!(x_code, opened[r], "row {r}: X differs from open_prequant");
                let alpha_code = (to_u64(side.alpha_exp) << 7 | to_u64(side.alpha_mantissa)) as u16;
                let beta_code = (to_u64(side.beta_exp) << 7 | to_u64(side.beta_mantissa)) as u16;
                // Noised: fma(alpha, x, beta*n), clamped and cast (native noisy_quantize).
                let nt = bf16_mul(beta_code, noise[r]).unwrap();
                let noised = bf16_fma(alpha_code, x_code, nt).unwrap();
                let expected_noised = f32_to_fp8_e4m3(bf16_to_f32(bf16_clamp_sym(noised, BF16_CODE_448))).unwrap();
                let got_noised = if std::ptr::eq(opened, &opened_a) {
                    to_u64(v.code_noised_a)
                } else {
                    to_u64(v.code_noised_b)
                } as u8;
                assert_eq!(got_noised, expected_noised, "row {r}: noised fp8 code differs from native");
                // Liveness: the committed bound is code(l2f) + 256 and the flag matches the
                // plaintext f64 predicate.
                assert_eq!(
                    to_u64(side.dead_bound),
                    u64::from(l2f(opened, g)) + 256,
                    "row {r}: dead bound differs from code(l2f) + 256"
                );
                assert_eq!(
                    to_u64(side.is_dead),
                    u64::from(native_dead(opened, g, x_code)),
                    "row {r}: dead flag differs from the plaintext liveness predicate"
                );
            }
        }
        // Group-final dead counts equal the native per-row dead-entry counts.
        for (opened, count_col, groups) in [
            (&opened_a, INPUT_QUANT_COL_MAP.dead_count_a, h),
            (&opened_b, INPUT_QUANT_COL_MAP.dead_count_b, program.w),
        ] {
            for g in 0..groups {
                let native: u64 = opened[g * k..(g + 1) * k]
                    .iter()
                    .filter(|&&x| native_dead(opened, g, x))
                    .count() as u64;
                assert_eq!(
                    to_u64(rows[(g + 1) * k - 1][count_col]),
                    native,
                    "group {g}: dead count differs from the native liveness count"
                );
            }
        }
    }

    #[test]
    fn group_scales_match_native_derivation() {
        // Compare scales derived from norms floored at 2^-32. For this fixture, native
        // f32 summation and the integer L2 calculation give the same norms.
        let (program, rows, _) = test_trace();
        let inp = test_inputs(&program);
        let (h, k) = (program.h, program.k);
        let opened_a = open_prequant(&inp.a_int8, &inp.a_scales, h, k, BLOCK_SIZE).unwrap();
        let opened_b = open_prequant(&inp.b_int8, &inp.b_scales, program.w, k, BLOCK_SIZE).unwrap();
        for g in 0..h {
            for (opened, alpha_exp_col, alpha_man_col, beta_exp_col, beta_man_col, bound_col) in [
                (
                    &opened_a,
                    INPUT_QUANT_COL_MAP.alpha_a_exp,
                    INPUT_QUANT_COL_MAP.alpha_a_mantissa,
                    INPUT_QUANT_COL_MAP.beta_a_exp,
                    INPUT_QUANT_COL_MAP.beta_a_mantissa,
                    INPUT_QUANT_COL_MAP.dead_bound_a,
                ),
                (
                    &opened_b,
                    INPUT_QUANT_COL_MAP.alpha_b_exp,
                    INPUT_QUANT_COL_MAP.alpha_b_mantissa,
                    INPUT_QUANT_COL_MAP.beta_b_exp,
                    INPUT_QUANT_COL_MAP.beta_b_mantissa,
                    INPUT_QUANT_COL_MAP.dead_bound_b,
                ),
            ] {
                let (l2, linf) = Fp8E4M3Quant.row_norms(&opened[g * k..(g + 1) * k]).unwrap();
                let (alpha, beta) = Fp8E4M3Quant
                    .derive_row_scales(bf16_max(l2, NORM_FLOOR_CODE), bf16_max(linf, NORM_FLOOR_CODE), program.r)
                    .unwrap();
                let row = &rows[g * k];
                let got_alpha = (to_u64(row[alpha_exp_col]) << 7 | to_u64(row[alpha_man_col])) as u16;
                let got_beta = (to_u64(row[beta_exp_col]) << 7 | to_u64(row[beta_man_col])) as u16;
                assert_eq!(got_alpha, alpha, "group {g}: alpha differs from the native derivation");
                assert_eq!(got_beta, beta, "group {g}: beta differs from the native derivation");
                assert_eq!(
                    to_u64(row[bound_col]),
                    u64::from(bf16_max(l2, NORM_FLOOR_CODE)) + 256,
                    "group {g}: dead bound differs from the native floored L2"
                );
            }
        }
    }

    #[test]
    fn all_zero_group_is_provable_with_reference_scales() {
        // An all-zero row must still produce finite alpha and positive beta after the
        // 2^-32 norm floor. Zero the first A row's int8 values while retaining its noise;
        // check constraints, lookups and the native scale derivation.
        let program = test_program();
        let mut inp = test_inputs(&program);
        let k = program.k;
        inp.a_int8[..k].fill(0);
        let (rows, pis) = program.generate_trace::<F>(
            &inp.a_int8,
            &inp.a_scales,
            &inp.a_noise,
            &inp.b_int8,
            &inp.b_scales,
            &inp.b_noise,
        );

        let v: &InputQuantColumnsView<F> = rows[k - 1].borrow();
        assert_eq!(v.max_abs_a, F::ZERO, "test premise: the A group decodes to all zeros");
        let (alpha, beta) = Fp8E4M3Quant
            .derive_row_scales(NORM_FLOOR_CODE, NORM_FLOOR_CODE, program.r)
            .expect("native scales on the floor");
        let got = |exp: F, man: F| (to_u64(exp) << 7 | to_u64(man)) as u16;
        assert_eq!(got(v.alpha_a_exp, v.alpha_a_mantissa), alpha, "alpha on the floored norms");
        assert_eq!(got(v.beta_a_exp, v.beta_a_mantissa), beta, "beta on the floored norms");
        assert_ne!(beta, 0, "the floored l2 keeps the noise scale strictly positive");

        let stark = S::new(program.clone());
        assert!(
            constraints_vanish(&stark, &rows, &pis),
            "all-zero A group must satisfy the AIR"
        );

        // Every declared LUT instance must stay in-domain on this trace too.
        let polys = trace_rows_to_poly_values(rows.clone());
        let lookups = input_quant_lut_lookups::<F>();
        for (li, lookup) in lookups.iter().enumerate() {
            for r in 0..rows.len() {
                let filter = lookup.filter.eval_table(&polys, r, &[]);
                assert!(
                    filter == F::ZERO || filter == F::ONE,
                    "lookup {li} row {r}: non-boolean filter"
                );
                if filter == F::ZERO {
                    continue;
                }
                let keys: Vec<u64> = lookup.keys.iter().map(|c| to_u64(c.eval_table(&polys, r, &[]))).collect();
                let values: Vec<u64> = lookup.values.iter().map(|c| to_u64(c.eval_table(&polys, r, &[]))).collect();
                check_lut_semantics(lookup, &keys, &values, li, r);
            }
        }
    }

    /// Finite bf16 codes covering zeros, subnormals, boundary normals, both signs and mid/large
    /// magnitudes.
    const INTERESTING_CODES: [u16; 26] = [
        0x0000, 0x8000, // ±0
        0x0001, 0x8001, 0x0040, 0x007F, // subnormals
        0x0080, 0x8080, // smallest normals
        0x00FF, 0x0100, // exponent 1/2 boundary
        0x3F80, 0xBF80, // ±1
        0x3F81, 0x3F00, 0x4000, 0x4040, 0x42F8, // 1+ulp, 0.5, 2, 3, 124
        0x0F80, 0x8F93, // tiny normals
        0x2AAB, 0x3521, 0x4C88, // mid-range
        0x5A31, 0xDA31, // large
        0x7E00, 0x7F7F, // huge (overflow with large partners; native errors, pairs skipped)
    ];

    #[test]
    fn mul_gadget_matches_native_bf16_mul() {
        // Full pair sweep of the interesting codes; pairs whose product overflows bf16 are
        // skipped (bf16_mul errors there; the AIR bans them via RC16(254 - OUT_EXP) and the
        // gadget panics).
        let mut checked = 0usize;
        for &a in &INTERESTING_CODES {
            for &b in &INTERESTING_CODES {
                let Ok(expected) = bf16_mul(a, b) else { continue };
                let fa = Bf16Fields::from_code(a);
                let fb = Bf16Fields::from_code(b);
                let m = mul_gadget(fa, fb);
                assert_eq!(
                    m.out.fields(fa.sign ^ fb.sign).code(),
                    expected,
                    "mul mirror diverged on {a:#06x} * {b:#06x}"
                );
                checked += 1;
            }
        }
        assert!(checked > 500, "sweep unexpectedly small ({checked})");
        // Every int8 byte against a scale sample: the exact decode path (P1 + group U).
        for byte in 0u8..=255 {
            let int_f = int8dec(byte);
            for &scale in &[0x3F80u16, 0xBFC0, 0x0040, 0x0000, 0x4310, 0x1000] {
                let fs = Bf16Fields::from_code(scale);
                let expected = bf16_mul(int_f.code(), scale).unwrap();
                let m = mul_gadget(int_f, fs);
                assert_eq!(
                    m.out.fields(int_f.sign ^ fs.sign).code(),
                    expected,
                    "int8 decode diverged on {byte} * {scale:#06x}"
                );
            }
        }
    }

    #[test]
    fn fma_gadget_matches_native_bf16_fma() {
        // Alpha is positive normal; X and the addend may be any finite bf16 (subnormal X
        // included, as a subnormal decode produces). Crafted cases include:
        // - exact cancellation (V = 0) and near-cancellation (V = 1);
        // - cancellation to an exact subnormal at KEY_SCALE = -133 (cut-0 subnormal branch);
        // - X = ±0 with a far unit gap, which must preserve the addend exactly;
        // - subnormal/zero addends against wide products (far collapse, sticky-only addend).
        let alphas: [u16; 7] = [0x3F80, 0x3F81, 0x4310, 0x2E80, 0x5C00, 0x0080, 0x4C10];
        let xs: [u16; 13] = [
            0x0000, 0x8000, 0x3F80, 0xBF81, 0x3F81, 0x0080, 0x4040, 0xC2F8, 0x0100, 0x5A31, 0x0001, 0x807F, 0x0040,
        ];
        let addends: [u16; 12] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x3F80, 0xBF82, 0x0080, 0x4100, 0xB880, 0x2000, 0x5900,
        ];
        let mut checked = 0usize;
        let mut check = |alpha: u16, x: u16, c: u16| {
            let Ok(expected) = bf16_fma(alpha, x, c) else { return };
            let fa = Bf16Fields::from_code(alpha);
            let fx = Bf16Fields::from_code(x);
            let fc = Bf16Fields::from_code(c);
            assert!(!fa.sign && fa.exp >= 1, "alpha must be positive normal");
            let m_p = fa.m() * fx.m();
            let ep_raw = fa.exp as i64 + fx.e_star() - MUL_LSB_OFFSET;
            let w = fma_gadget(m_p, ep_raw, fx.m() == 0, fx.sign, fc.m(), fc.e_star() - 134, fc.sign);
            assert_eq!(
                w.out.fields(w.out_sign).code(),
                expected,
                "FMA mirror diverged on fma({alpha:#06x}, {x:#06x}, {c:#06x})"
            );
            checked += 1;
        };
        for &alpha in &alphas {
            for &x in &xs {
                for &c in &addends {
                    check(alpha, x, c);
                }
            }
        }
        // Crafted cancellation triples: alpha = 1 + ulp (M = 129), x = ±(1 + ulp) (M(X) = 129):
        // M_P = 16641; addend M_C = 130 at gap 7 gives AL_C = 16640 -> V = 1.
        check(0x3F81, 0x3F81, 0xBF82); // V = 1 at EP = -14: normal cancellation result 2^-14
        check(0x3F81, 0xBF81, 0x3F82); // mirrored signs
        check(0x0401, 0x3F81, 0x8402); // V = 1 at EP = -133: exact subnormal result 2^-133
        check(0x1481, 0x3F81, 0x9482); // V = 1 at EP = -100: a normal result despite V < 128
        check(0x3F81, 0x3F81, 0xBF81); // V = 129 (mantissa-borrow pattern)
        // Cancellation with V < 128 at scales [-132, -127] exercises signed cut slots 1–6
        // and their normal/subnormal boundary.
        check(0x0481, 0x3F81, 0x8482); // V = 1 at EP = -132: subnormal 0x0002
        check(0x0501, 0x3F81, 0x8502); // V = 1 at EP = -131: subnormal 0x0004
        check(0x0581, 0x3F81, 0x8582); // V = 1 at EP = -130: subnormal 0x0008
        check(0x0601, 0x3F81, 0x8602); // V = 1 at EP = -129: subnormal 0x0010
        check(0x0681, 0x3F81, 0x8682); // V = 1 at EP = -128: subnormal 0x0020
        check(0x0701, 0x3F81, 0x8702); // V = 1 at EP = -127: subnormal 0x0040
        check(0x0781, 0x3F81, 0x8782); // V = 1 at EP = -126: the smallest normal, 0x0080
        check(0x0481, 0x3F82, 0x8483); // V = 2 at EP = -132: subnormal 0x0004
        check(0x0701, 0x3FC1, 0x8742); // V = 65 at EP = -127: normal 0x0302 despite V < 128
        check(0x0581, 0x3FBF, 0x85C0); // V = 63 at EP = -130: normal 0x017C despite V < 128
        // A zero X with huge alpha and a tiny addend still returns the addend: the product's
        // artificial scale must not change its mathematical value of zero.
        for &x in &[0x0000u16, 0x8000] {
            for &c in &[0x0001u16, 0x8001, 0x0040, 0x8080, 0x00FF, 0x2000] {
                check(0x4C10, x, c);
                check(0x7E00, x, c);
            }
        }
        // Both-zero ±0 sign rules.
        check(0x3F80, 0x0000, 0x0000);
        check(0x3F80, 0x8000, 0x0000);
        check(0x3F80, 0x0000, 0x8000);
        check(0x3F80, 0x8000, 0x8000);
        assert!(checked > 500, "sweep unexpectedly small ({checked})");
    }

    #[test]
    fn lut_instance_inventory_holds_on_honest_trace() {
        // Check all evaluated lookup tuples against their table semantics.
        let (_, rows, _) = test_trace();
        let n = rows.len();
        let polys = trace_rows_to_poly_values(rows.clone());
        let lookups = input_quant_lut_lookups::<F>();
        for (li, lookup) in lookups.iter().enumerate() {
            for r in 0..n {
                let filter = lookup.filter.eval_table(&polys, r, &[]);
                assert!(
                    filter == F::ZERO || filter == F::ONE,
                    "lookup {li} row {r}: non-boolean filter"
                );
                if filter == F::ZERO {
                    continue;
                }
                let keys: Vec<u64> = lookup.keys.iter().map(|c| to_u64(c.eval_table(&polys, r, &[]))).collect();
                let values: Vec<u64> = lookup.values.iter().map(|c| to_u64(c.eval_table(&polys, r, &[]))).collect();
                check_lut_semantics(lookup, &keys, &values, li, r);
            }
        }
    }

    /// Checks one evaluated LUT instance against the table's semantics.
    fn check_lut_semantics(lookup: &LutLookup<F>, keys: &[u64], values: &[u64], li: usize, r: usize) {
        match lookup.table {
            LutTable::Range16 => {
                assert!(keys[0] < 1 << 16, "lookup {li} row {r}: RC16 key {} out of range", keys[0]);
            }
            LutTable::Pair128 => {
                assert!(
                    keys[0] < 128 && keys[1] < 128,
                    "lookup {li} row {r}: PAIR128 out of range {keys:?}"
                );
            }
            LutTable::ExpInfo => {
                assert!(keys[0] <= 254, "lookup {li} row {r}: EXPINFO key {} out of domain", keys[0]);
                assert_eq!(values[0], u64::from(keys[0] == 0), "lookup {li} row {r}: EXPINFO flag");
            }
            LutTable::Clamp22 => {
                assert!(keys[0] <= 600, "lookup {li} row {r}: CLAMP22 key {} out of domain", keys[0]);
                assert_eq!(
                    values[0],
                    clamp_slot(keys[0] as i64 - 400),
                    "lookup {li} row {r}: CLAMP22 value"
                );
            }
            LutTable::Int8Dec => {
                assert!(keys[0] <= 255, "lookup {li} row {r}: INT8DEC key out of domain");
                let f = int8dec(keys[0] as u8);
                assert_eq!(
                    values,
                    &[u64::from(f.sign), f.exp, f.mantissa, u64::from(f.exp_is_zero())],
                    "lookup {li} row {r}: INT8DEC decode"
                );
            }
            LutTable::Pow2D => {
                // InputQuant's addend-coarser FMA path needs capped shifts through 19.
                assert!(keys[0] <= 19, "lookup {li} row {r}: POW2D key {} out of domain", keys[0]);
                assert_eq!(values[0], 1 << keys[0], "lookup {li} row {r}: POW2D value");
            }
            LutTable::RneRnd => {
                let (v, slot) = (keys[0] & ((1 << 17) - 1), keys[0] >> 17);
                assert!(slot <= SLOT_MAX as u64, "lookup {li} row {r}: RNERND slot out of domain");
                // The `(v, slot)` pair fully determines the row (the slot encodes the signed
                // fade cut, hence the normal/subnormal classification).
                let (mant, wa, iz, eiz) = rnernd_reference(v, slot);
                assert_eq!(
                    values,
                    &[mant, wa, u64::from(iz), u64::from(eiz)],
                    "lookup {li} row {r}: RNERND row"
                );
            }
            LutTable::Qcast => {
                assert!(
                    keys[0] < 1 << 16 && (keys[0] >> 7) & 0xFF != 255,
                    "lookup {li} row {r}: QCAST key not finite"
                );
                assert_eq!(
                    values[0],
                    u64::from(qcast(keys[0] as u16)),
                    "lookup {li} row {r}: QCAST value"
                );
            }
            LutTable::Width16 => {
                assert!(keys[0] < 1 << 16, "lookup {li} row {r}: WIDTH16 key out of domain");
                assert_eq!(values[0], bit_len(keys[0]), "lookup {li} row {r}: WIDTH16 width");
                assert_eq!(values[1], u64::from(keys[0] != 0), "lookup {li} row {r}: WIDTH16 nonzero");
            }
            LutTable::Log16 => {
                assert!(keys[0] < 1 << 16, "lookup {li} row {r}: LOG16 key out of domain");
                assert_eq!(values[0], log2_fixed(keys[0]), "lookup {li} row {r}: LOG16 value");
            }
            _ => panic!("lookup {li}: table {:?} is not in InputQuant's inventory", lookup.table),
        }
    }

    #[test]
    fn tampered_traces_fail() {
        let (program, rows, pis) = test_trace();
        let stark = S::new(program);
        assert!(constraints_vanish(&stark, &rows, &pis), "honest trace must pass");
        let m = &INPUT_QUANT_COL_MAP;

        // U1 (group U): a corrupted decode significand product breaks the M(INT)*M(SCALE) bind.
        let mut t = rows.clone();
        t[5][m.decode_multiply_a.sig_product] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "U1 tamper undetected");

        // F2 (group F): a corrupted running max is neither the previous max nor the new element.
        let mut t = rows.clone();
        t[3][m.max_abs_a] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "F2 tamper undetected");

        // W6/W8 (group W): a corrupted fold magnitude breaks the signed-fold equation.
        let vi = (0..rows.len())
            .find(|&i| rows[i][m.noised_value_fma_a.folded_magnitude] != F::ZERO)
            .unwrap();
        let mut t = rows.clone();
        t[vi][m.noised_value_fma_a.folded_magnitude] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "W6 tamper undetected");

        // W6/W7 (group W): flipping the committed output sign breaks the fold or the ±0 rule.
        let mut t = rows.clone();
        t[vi][m.noised_value_fma_a.out_sign] = F::ONE - t[vi][m.noised_value_fma_a.out_sign];
        assert!(!constraints_vanish(&stark, &t, &pis), "OUT_SIGN tamper undetected");

        // C3 (group C): a corrupted dead counter breaks the running-count transition.
        let mut t = rows.clone();
        t[7][m.dead_count_b] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "C3 tamper undetected");

        // C1 (group C): a non-boolean dead flag breaks the booleanness bind (and would
        // corrupt the RC16 certificate filters).
        let mut t = rows.clone();
        t[7][m.is_dead_a] += F::TWO;
        assert!(!constraints_vanish(&stark, &t, &pis), "C1 tamper undetected");

        // C2 (group C): the dead bound must be group-constant (only its group-final value is
        // CTL-bound to ScaleStark).
        let mut t = rows.clone();
        t[2][m.dead_bound_a] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "C2 tamper undetected");

        // S4 (group S): alpha must be group-constant.
        let mut t = rows.clone();
        t[2][m.alpha_a_exp] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "S4 tamper undetected");

        // B10: tampering the `2^Wl2` public input breaks the scaled-block-product bind on
        // every live nonzero block (the AIR is program-independent; the geometry enters only
        // through this pinned public input).
        let mut bad_pis = pis;
        bad_pis[WL2_POW_PUBLIC_INPUT] = bad_pis[WL2_POW_PUBLIC_INPUT].double();
        assert!(
            !constraints_vanish(&stark, &rows, &bad_pis),
            "WL2_POW public-input tamper undetected"
        );

        // The rounding-key high bit must be boolean. Together with the separate RC16 lookup
        // tested below, this proves the full key is below 2^17.
        let mut t = rows.clone();
        t[0][m.noised_value_fma_a.rounding_significand_key_high_bit] += F::TWO;
        assert!(
            !constraints_vanish(&stark, &t, &pis),
            "D7 ROUNDING_SIGNIFICAND_KEY_HIGH_BIT booleanness tamper undetected"
        );

        // Move a block boundary into the middle of a real block and adjust later block indices.
        // This naive forgery already breaks the AIR because it resets the square accumulator on
        // real mid-block data. More generally, even a forgery that rewrote every dependent
        // witness would fail because the verifier independently recomputes the schedule.
        let mut t = rows.clone();
        t[9][m.is_block_start] = F::ONE;
        for row in t.iter_mut().skip(9) {
            row[m.block_index] += F::ONE;
        }
        assert!(
            !constraints_vanish(&stark, &t, &pis),
            "naive block-boundary relocation undetected"
        );
        let program2 = test_program();
        let inp = test_inputs(&program2);
        let known = program2.known_values::<F>(&inp.a_noise, &inp.b_noise);
        let bs_col = m.is_block_start;
        let blk_col = m.block_index;
        assert_ne!(known[bs_col].values[9], t[9][bs_col], "IS_BLOCK_START must diverge at row 9");
        assert_ne!(known[blk_col].values[9], t[9][blk_col], "block_index must diverge at row 9");

        // Group B (block-integer L2) tampers, one per constraint family.
        let n = rows.len();
        let is_block_final = |r: usize| rows[(r + 1) % n][m.is_block_start] == F::ONE;

        // B2: a corrupted in-block integer-square sum breaks the block-sum transition.
        let mut t = rows.clone();
        t[4][m.block_l2_a.block_int_squared_sum] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B2 tamper undetected");

        // B3: `block_l2_product` must equal M(scale)^2 times `block_int_squared_sum`.
        let mut t = rows.clone();
        t[6][m.block_l2_a.block_l2_product] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B3 tamper undetected");

        // B4: claiming a live block dead breaks the two-sided inverse pair (a dead block's
        // candidate would drop out of the frame max and its TERM out of S).
        let live_bf = (0..n)
            .find(|&r| is_block_final(r) && rows[r][m.block_l2_a.block_l2_product_nonzero] == F::ONE)
            .expect("test data has live blocks");
        let mut t = rows.clone();
        t[live_bf][m.block_l2_a.block_l2_product_nonzero] = F::ZERO;
        assert!(!constraints_vanish(&stark, &t, &pis), "B4 tamper undetected");

        // B5: a nonzero frame candidate on a mid-block row.
        let mut t = rows.clone();
        t[2][m.block_l2_a.block_doubled_scale_exponent] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B5 tamper undetected");

        // B6: a running max that is neither the previous max nor the new candidate.
        let mut t = rows.clone();
        t[10][m.block_l2_a.running_max_doubled_scale_exponent] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B6 tamper undetected");

        // B7: a forged (group-constant) frame no longer matches the attained max at the
        // group-final row — inflating the frame to shrink every TERM is rejected.
        let mut t = rows.clone();
        for r in t.iter_mut().take(program2.k) {
            r[m.block_l2_a.frame_doubled_scale_exponent] += F::TWO;
        }
        assert!(!constraints_vanish(&stark, &t, &pis), "B7 frame tamper undetected");

        // B8: `is_far_shift` is block-final-only.
        let mut t = rows.clone();
        t[3][m.block_l2_a.is_far_shift] = F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B8 tamper undetected");

        // B10: a corrupted scaled-block-product limb breaks dividend recomposition.
        let mut t = rows.clone();
        t[5][m.block_l2_a.scaled_block_product_limbs[1]] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B10 tamper undetected");

        // B13: a selector bit on a mid-block row breaks the near-live gate sum.
        let mut t = rows.clone();
        t[1][m.block_l2_a.shift_limb_selector[0]] = F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B13 tamper undetected");

        // B14: a corrupted selected-limb quotient on a near-live block-final row breaks the
        // Euclidean split (and would shift the block's TERM).
        let near = (0..n)
            .find(|&r| {
                m.block_l2_a
                    .shift_limb_selector
                    .iter()
                    .fold(F::ZERO, |acc, &q| acc + rows[r][q])
                    == F::ONE
            })
            .expect("test data has near-live blocks");
        let mut t = rows.clone();
        t[near][m.block_l2_a.selected_limb_quotient] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B14 tamper undetected");

        // B16: forging the group-final frame sum S — the value the Scale tuple carries.
        let gi = (0..n)
            .find(|&r| rows[r][m.is_group_final] == F::ONE)
            .expect("test data has a group-final row");
        let mut t = rows.clone();
        t[gi][m.block_l2_a.running_l2_frame_sum] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B16 S tamper undetected");
        // ...and the B side's accumulator chain is enforced independently.
        let mut t = rows.clone();
        t[gi][m.block_l2_b.running_l2_frame_sum] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B16 B-side tamper undetected");

        // Group V (check 4 score witness) tampers, one per constraint family.

        // V3: NORMALIZED_SIGMA_SIGNIFICAND must be group-constant.
        let mut t = rows.clone();
        t[2][m.normalized_sigma_significand_a] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V3 tamper undetected");

        // V5: flipping the dominance bit negates the gap identity on a split-encoding row.
        let vr = (0..n)
            .find(|&r| rows[r][m.exponent_gap_a] != F::ZERO)
            .expect("test data has rows with distinct addend encodings");
        let mut t = rows.clone();
        t[vr][m.x_dominates_a] = F::ONE - t[vr][m.x_dominates_a];
        assert!(!constraints_vanish(&stark, &t, &pis), "V5 tamper undetected");

        // V7: claiming a near row far zeroes the expected NEAR_GAP.
        let nr = (0..n)
            .find(|&r| rows[r][m.near_gap_a] != F::ZERO)
            .expect("test data has near rows");
        let mut t = rows.clone();
        t[nr][m.gap_is_far_a] = F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V7 tamper undetected");

        // V8: a corrupted dominant significand breaks the ordering mux.
        let mut t = rows.clone();
        t[vr][m.dominant_significand_a] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V8 tamper undetected");

        // V9: a corrupted second-stage quotient breaks the Euclidean split.
        let mut t = rows.clone();
        t[vr][m.quotient_a] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V9 tamper undetected");

        // V10: a corrupted top slice breaks the 2^17 decomposition.
        let mut t = rows.clone();
        t[vr][m.sum_of_squares_top_b] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V10 tamper undetected");

        // V11: a forged score no longer matches its committed sum of squares.
        let mut t = rows.clone();
        t[vr][m.summand_score_a] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V11 tamper undetected");
    }

    /// Proves the FMA rounding significand is below `2^17`. The RNERND lookup packs significand
    /// and cut depth into one integer, so its domain alone cannot prevent an oversized
    /// significand from aliasing another cut slot. A committed high bit plus a 16-bit
    /// low-limb range check closes that ambiguity.
    #[test]
    fn d7_rounding_key_high_bit_range_proof_pins_rounding_key() {
        let (_, rows, _pis) = test_trace();
        let m = &INPUT_QUANT_COL_MAP;

        // Each FMA side has an RC16 lookup of
        // `rounding_significand_key - 2^16*rounding_significand_key_high_bit`. Locate both
        // descriptors and confirm that they read the intended columns.
        let polys = trace_rows_to_poly_values(rows.clone());
        let lookups = input_quant_lut_lookups::<F>();
        let limb = F::from_canonical_u64(1 << 16);
        let d7_key = |f: &FmaBlockView<usize>, r: usize| {
            polys[f.rounding_significand_key].values[r] - limb * polys[f.rounding_significand_key_high_bit].values[r]
        };
        let mut matched_sides = 0;
        for f in [&m.noised_value_fma_a, &m.noised_value_fma_b] {
            let found = lookups.iter().any(|l| {
                l.table == LutTable::Range16 && (0..rows.len()).all(|r| l.keys[0].eval_table(&polys, r, &[]) == d7_key(f, r))
            });
            matched_sides += usize::from(found);
        }
        assert_eq!(
            matched_sides, 2,
            "expected one D7 ROUNDING_SIGNIFICAND_KEY range proof per FMA side"
        );

        // On every honest row the committed high bit is the key's actual bit 16, leaving a
        // 16-bit low limb for the RC16 lookup.
        for row in &rows {
            for f in [&m.noised_value_fma_a, &m.noised_value_fma_b] {
                let rounding_significand_key = to_u64(row[f.rounding_significand_key]);
                let hi = to_u64(row[f.rounding_significand_key_high_bit]);
                assert!(
                    rounding_significand_key < 1 << 17,
                    "honest ROUNDING_SIGNIFICAND_KEY {rounding_significand_key} out of range"
                );
                assert_eq!(
                    hi,
                    rounding_significand_key >> 16,
                    "honest ROUNDING_SIGNIFICAND_KEY_HIGH_BIT mis-witnessed"
                );
                assert!(
                    rounding_significand_key - (hi << 16) < 1 << 16,
                    "honest D7 RC16 key out of range"
                );
            }
        }

        // The forged pair `(v + 2^17, cut - 1)` leaves the packed RNERND key unchanged. No
        // boolean high bit can make its remaining low limb fit RC16, so the separate range
        // proof rejects it.
        let v: u64 = 0x1_5AAA; // any v < 2^17
        let cut: u64 = 5;
        let honest_packed = v + (1 << 17) * cut;
        let forged_v = v + (1 << 17);
        let forged_packed = forged_v + (1 << 17) * (cut - 1);
        assert_eq!(honest_packed, forged_packed, "the aliasing collision the RC16 must block");
        for hi in [0u64, 1] {
            assert!(
                forged_v.wrapping_sub(hi << 16) >= 1 << 16,
                "forged ROUNDING_SIGNIFICAND_KEY {forged_v} must fail RC16 for high bit {hi}"
            );
        }
    }

    #[test]
    fn degree_is_at_most_three() {
        test_stark_low_degree::<F, S, D>(S::new(test_program())).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        test_stark_circuit_constraints::<F, C, S, D>(S::new(test_program())).unwrap();
    }
}
