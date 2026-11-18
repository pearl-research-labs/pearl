//! Integer certificates for finite binary32 RNE, fused alignment, and the FP8 cast proxy.
//!
//! CoefficientStark uses the rounding gadget on an exact bf16 significand product.
//! InputQuant uses alignment followed by rounding to prove `RN32(beta*n + c*q)`, where
//! `c = RN32(alpha*block_scale)`, `q` is an int8, and `n` is public binary32 noise. The
//! product `beta*n` is not rounded before addition. All witness generation has a matching
//! evaluator plus committed-LUT lookups; witness generation alone proves nothing.
//!
//! # Representations and normalization
//!
//! A finite binary32 magnitude is `M*2^(E*-150)`, with raw exponent `E`, effective
//! exponent `E*=max(E,1)`, and `M = 2^23*[E!=0] + fraction`. A bf16 magnitude instead has
//! `M_b = 128*[E!=0] + fraction` and scale `E*-134`. Thus the unrounded noise product has
//! integer significand `P = M(beta)*M(n) < 2^32` and scale `E*(beta)+E*(n)-284`.
//! The signal has `S = M(c)*|q| < 2^32` and scale `E*(c)-150`.
//!
//! For either nonzero integer `J` of width `w`, FP32TOOLS operation 0 supplies `2^(32-w)`
//! and the zero flag; the normalized integer `N=J*2^(32-w)` lies in `[2^31,2^32)`.
//! The alignment encodings are
//!
//! ```text
//! e_product = E*(beta) + E*(n) + width(P) + 15
//! e_signal  = E*(c) + width(S) + 149,
//! value     = N * 2^(e-331).
//! ```
//!
//! Each nonzero encoding is `floor(log2(value))+300`; zero uses encoding 0. Bounded high
//! limbs certify normalization, and missing negative gap keys certify the exponent order.
//!
//! # Alignment with a signed tail
//!
//! Let `L,H` be the larger- and smaller-exponent normalized integers, with respective
//! signs `s_L,s_H`, and gap `d`. Keep three extra bits and divide
//!
//! ```text
//! D = 2^min(d,35)
//! 8*H = Q*D + R,           0 <= R < D
//! (-1)^sign * V = (-1)^s_L * 8*L + (-1)^s_H * Q.
//! ```
//!
//! For `d <= 35`, the unrepresented tail in units of `2^(max(e)-334)` has magnitude
//! `R/D < 1`; it decreases the result magnitude when the operand signs differ. For larger
//! gaps, `8*H < 2^35` already forces `Q=0`; the cap retains the smaller operand's
//! nonzeroness and sign, which are all that final RNE needs. FP32TOOLS operation 2 also
//! bounds `Q < 2^(35-min(d,35))`. Ranged complement limbs prove both division bounds.
//! `V < 2^36`, its binary scale is `max(e)-334`, and zero uses scale -149. Inverse
//! witnesses bind `has_tail=[R!=0]` and `is_zero=[V=0]`; an exact zero is negative only
//! when both zero addends are negative, as required by binary32 FMA.
//!
//! # One nearest-even rounding
//!
//! For magnitude `V*2^scale`, choose the biased output binade `b >= 1`, reserving `b=1`
//! for both subnormals and the first normal binade. Put `shift=b-150-scale`. FP32TOOLS
//! operation 1 supplies `L=2^max(-shift,0)` and `D=2^min(max(shift,0),37)`. Prove
//!
//! ```text
//! V*L = Q*D + R,    0 <= R < D,    0 <= Q < 2^24
//! Q >= 2^23 when b > 1
//! C = 2*R-D + has_tail*(1-2*tail_negative) + (1-has_tail)*(Q mod 2)
//! round_up = [C >= 1].
//! ```
//!
//! At an exact midpoint with no tail, the parity term implements ties-to-even. A nonzero
//! signed tail resolves the midpoint toward its true side. The cap at 37 is sufficient
//! because `V < 2^36`: a larger right shift always rounds to zero. Nonnegative ranged
//! slacks prove both branches of the `C` comparison. The result recomposes `Q+round_up`,
//! with an optional carry from `2^24` to the next binade; EXPINFO and limb bounds prove
//! the finite output fields and the normal/subnormal classification.
//!
//! The largest rounding recompositions are below `2^61` (`Q < 2^24`, `D <= 2^37`),
//! beneath the Goldilocks modulus. Limb ranges and complement bounds are essential to
//! interpreting the field equalities as these integer divisions and inequalities.
//!
//! # Round-to-odd proxy for QCAST
//!
//! After binary32 rounding, split the 23-bit fraction as `2^16*hi+lo`. The proxy copies
//! sign/exponent/`hi`, then sets `hi`'s low bit when `lo != 0`. This truncation to odd
//! retains eight significant bits for normal inputs and the analogous subnormal grid.
//! QCAST's saturating E4M3 RNE therefore gives the direct binary32-to-E4M3 result: eight
//! retained bits exceed E4M3's four-bit precision by enough to preserve ties. In
//! particular, doing ordinary bf16 RNE here could double-round and is not equivalent.

use crate::v4::api::dtype::bf16_to_f32;
use anyhow::{Result, ensure};
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use starky::lookup::{Column, Filter};

use super::luts::LutTable;
use super::luts::ctl::LutLookup;
use crate::v4::circuit::utils::evaluator::Evaluator;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp32Magnitude<T: Copy> {
    pub exponent: T,
    pub fraction_low: T,
    pub fraction_high: T,
    pub is_exponent_zero: T,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp32Fields<T: Copy> {
    pub sign: T,
    pub exponent: T,
    pub fraction: T,
    pub is_exponent_zero: T,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AlignmentColumns<T: Copy> {
    pub signal_significand: T,
    pub product_width: T,
    pub is_product_zero: T,
    pub product_normalization_power: T,
    pub product_normalized: [T; 2],
    pub product_exponent: T,
    pub signal_width: T,
    pub is_signal_zero: T,
    pub signal_normalization_power: T,
    pub signal_normalized: [T; 2],
    pub signal_exponent: T,
    pub product_is_larger: T,
    pub exponent_gap: T,
    pub alignment_divisor: T,
    pub larger_significand: T,
    pub smaller_significand: T,
    pub larger_sign: T,
    pub smaller_sign: T,
    pub quotient: [T; 3],
    pub remainder: [T; 3],
    pub remainder_complement: [T; 3],
    pub quotient_limit: T,
    pub quotient_complement: [T; 3],
    pub output_sign: T,
    pub magnitude: [T; 3],
    pub is_zero: T,
    pub magnitude_inverse: T,
    pub has_tail: T,
    pub tail_inverse: T,
    pub tail_negative: T,
    pub negative_zero_product: T,
    pub negative_zero_signal: T,
    pub binary_scale: T,
}

const _: () = assert!(size_of::<AlignmentColumns<u8>>() == 45);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OddProxyColumns<T: Copy> {
    pub high_fraction_parity: T,
    pub low_fraction_is_nonzero: T,
    pub low_fraction_inverse: T,
    pub code: T,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DownscaleColumns<T: Copy> {
    pub fraction_bit_15: T,
    pub coin_field: T,
    pub coin: T,
    pub strip_threshold: T,
    pub output_code: T,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Round32Columns<T: Copy> {
    pub binade: T,
    pub is_first_binade: T,
    pub left_shift_power: T,
    pub rounding_divisor: T,
    pub quotient: [T; 2],
    pub remainder: [T; 3],
    pub remainder_complement: [T; 3],
    pub quotient_parity: T,
    pub round_up: T,
    pub comparison_slack: [T; 3],
    pub comparison_complement: [T; 3],
    pub has_exponent_carry: T,
    pub output: Fp32Magnitude<T>,
}

const _: () = assert!(size_of::<Round32Columns<u8>>() == 25);

pub(crate) fn limbs<F: Field, const N: usize>(value: u64) -> [F; N] {
    core::array::from_fn(|i| F::from_canonical_u64((value >> (16 * i)) & 65535))
}

pub(crate) fn limb_value<V: Copy, S: Copy, E: Evaluator<V, S>, const N: usize>(evaluator: &mut E, values: [V; N]) -> V {
    let mut result = evaluator.i32(0);
    for (i, value) in values.into_iter().enumerate() {
        let weight = evaluator.u64(1 << (16 * i));
        result = evaluator.mad(value, weight, result);
    }
    result
}

impl<F: RichField> Fp32Magnitude<F> {
    pub(crate) fn from_word(word: u32) -> Self {
        let exponent = (word >> 23) & 255;
        assert!(exponent < 255, "nonfinite binary32 magnitude");
        Self {
            exponent: F::from_canonical_u32(exponent),
            fraction_low: F::from_canonical_u32(word & 65535),
            fraction_high: F::from_canonical_u32((word >> 16) & 127),
            is_exponent_zero: F::from_bool(exponent == 0),
        }
    }

    pub(crate) fn word(&self) -> u32 {
        ((self.exponent.to_canonical_u64() as u32) << 23)
            | ((self.fraction_high.to_canonical_u64() as u32) << 16)
            | self.fraction_low.to_canonical_u64() as u32
    }
}

impl<F: RichField> Fp32Fields<F> {
    pub(crate) fn from_value(value: f32) -> Self {
        assert!(value.is_finite(), "nonfinite public noise");
        let word = value.to_bits();
        Self {
            sign: F::from_bool(word >> 31 != 0),
            exponent: F::from_canonical_u32((word >> 23) & 255),
            fraction: F::from_canonical_u32(word & 0x7fffff),
            is_exponent_zero: F::from_bool(word & 0x7f800000 == 0),
        }
    }
}

pub(crate) fn generate_fold<F: RichField>(
    coefficient: f32,
    integer: i8,
    beta: u16,
    noise: f32,
) -> Result<(AlignmentColumns<F>, Round32Columns<F>, OddProxyColumns<F>)> {
    ensure!(
        coefficient.is_finite() && noise.is_finite() && beta & 0xff80 < 0x7f80,
        "nonfinite fold operand"
    );
    let coefficient_word = coefficient.to_bits();
    let coefficient_exponent = u64::from((coefficient_word >> 23) & 255);
    let coefficient_exponent_star = coefficient_exponent.max(1);
    let coefficient_significand = u64::from(coefficient_word & 0x7fffff) + (u64::from(coefficient_exponent != 0) << 23);
    let noise_word = noise.to_bits();
    let noise_exponent = u64::from((noise_word >> 23) & 255);
    let noise_significand = u64::from(noise_word & 0x7fffff) + (u64::from(noise_exponent != 0) << 23);
    let beta_exponent = u64::from(beta >> 7);
    let beta_significand = u64::from(beta & 127) + 128 * u64::from(beta_exponent != 0);
    let product = beta_significand * noise_significand;
    let signal = coefficient_significand * u64::from(integer.unsigned_abs());
    let product_width = if product == 0 { 0 } else { product.ilog2() + 1 };
    let signal_width = if signal == 0 { 0 } else { signal.ilog2() + 1 };
    let product_power = if product == 0 { 0 } else { 1u64 << (32 - product_width) };
    let signal_power = if signal == 0 { 0 } else { 1u64 << (32 - signal_width) };
    let normalized_product = product * product_power;
    let normalized_signal = signal * signal_power;
    let product_exponent = if product == 0 {
        0
    } else {
        beta_exponent.max(1) + noise_exponent.max(1) + u64::from(product_width) + 15
    };
    let signal_exponent = if signal == 0 {
        0
    } else {
        coefficient_exponent_star + u64::from(signal_width) + 149
    };
    ensure!(signal_exponent <= 427, "nonfinite signal");
    let product_is_larger = product_exponent >= signal_exponent;
    let gap = product_exponent.abs_diff(signal_exponent);
    let divisor = 1u64 << gap.min(35);
    let quotient_limit = 1u64 << (35 - gap.min(35));
    let product_sign = noise_word >> 31 != 0;
    let signal_sign = (coefficient_word >> 31 != 0) ^ (integer < 0);
    let (larger, smaller, larger_sign, smaller_sign) = if product_is_larger {
        (normalized_product, normalized_signal, product_sign, signal_sign)
    } else {
        (normalized_signal, normalized_product, signal_sign, product_sign)
    };
    let quotient = 8 * smaller / divisor;
    let remainder = 8 * smaller % divisor;
    let signed_sum = (1 - 2 * i64::from(larger_sign)) * (8 * larger) as i64 + (1 - 2 * i64::from(smaller_sign)) * quotient as i64;
    let magnitude = signed_sum.unsigned_abs();
    let negative_zero_product = product == 0 && product_sign;
    let negative_zero_signal = signal == 0 && signal_sign;
    let output_sign = if magnitude == 0 {
        negative_zero_product && negative_zero_signal
    } else {
        signed_sum < 0
    };
    let binary_scale = if magnitude == 0 {
        -149
    } else {
        product_exponent.max(signal_exponent) as i64 - 334
    };
    let magnitude_field = F::from_canonical_u64(magnitude);
    let remainder_field = F::from_canonical_u64(remainder);
    let alignment = AlignmentColumns {
        signal_significand: F::from_canonical_u64(signal),
        product_width: F::from_canonical_u32(product_width),
        is_product_zero: F::from_bool(product == 0),
        product_normalization_power: F::from_canonical_u64(product_power),
        product_normalized: limbs(normalized_product),
        product_exponent: F::from_canonical_u64(product_exponent),
        signal_width: F::from_canonical_u32(signal_width),
        is_signal_zero: F::from_bool(signal == 0),
        signal_normalization_power: F::from_canonical_u64(signal_power),
        signal_normalized: limbs(normalized_signal),
        signal_exponent: F::from_canonical_u64(signal_exponent),
        product_is_larger: F::from_bool(product_is_larger),
        exponent_gap: F::from_canonical_u64(gap),
        alignment_divisor: F::from_canonical_u64(divisor),
        larger_significand: F::from_canonical_u64(larger),
        smaller_significand: F::from_canonical_u64(smaller),
        larger_sign: F::from_bool(larger_sign),
        smaller_sign: F::from_bool(smaller_sign),
        quotient: limbs(quotient),
        remainder: limbs(remainder),
        remainder_complement: limbs(divisor - 1 - remainder),
        quotient_limit: F::from_canonical_u64(quotient_limit),
        quotient_complement: limbs(quotient_limit - 1 - quotient),
        output_sign: F::from_bool(output_sign),
        magnitude: limbs(magnitude),
        is_zero: F::from_bool(magnitude == 0),
        magnitude_inverse: if magnitude == 0 { F::ZERO } else { magnitude_field.inverse() },
        has_tail: F::from_bool(remainder != 0),
        tail_inverse: if remainder == 0 { F::ZERO } else { remainder_field.inverse() },
        tail_negative: F::from_bool(larger_sign ^ smaller_sign),
        negative_zero_product: F::from_bool(negative_zero_product),
        negative_zero_signal: F::from_bool(negative_zero_signal),
        binary_scale: if binary_scale < 0 {
            -F::from_canonical_u64(binary_scale.unsigned_abs())
        } else {
            F::from_canonical_u64(binary_scale as u64)
        },
    };
    let rounding = Round32Columns::generate(magnitude, binary_scale, remainder != 0, larger_sign ^ smaller_sign)?;
    let word = rounding.output.word();
    let fraction_low = word & 65535;
    let fraction_high = word >> 16 & 127;
    let proxy = OddProxyColumns {
        high_fraction_parity: F::from_canonical_u32(fraction_high & 1),
        low_fraction_is_nonzero: F::from_bool(fraction_low != 0),
        low_fraction_inverse: if fraction_low == 0 {
            F::ZERO
        } else {
            F::from_canonical_u32(fraction_low).inverse()
        },
        code: F::from_canonical_u32(
            (u32::from(output_sign) << 15)
                + ((word >> 23) << 7)
                + fraction_high
                + u32::from(fraction_low != 0 && fraction_high & 1 == 0),
        ),
    };
    debug_assert_eq!(
        word | (u32::from(output_sign) << 31),
        crate::v4::api::compute::fp32_fma(bf16_to_f32(beta), noise, coefficient * f32::from(integer))?.to_bits()
    );
    Ok((alignment, rounding, proxy))
}

impl<F: RichField> Round32Columns<F> {
    pub(crate) fn generate(value: u64, scale: i64, has_tail: bool, tail_negative: bool) -> Result<Self> {
        ensure!(value < 1 << 36, "binary32 rounding magnitude exceeds 36 bits");
        let binade = if value == 0 {
            1
        } else {
            (value.ilog2() as i64 + scale + 127).max(1)
        };
        ensure!(binade <= 255, "binary32 rounding overflow");
        let shift = binade - 150 - scale;
        ensure!((-23..=167).contains(&shift), "binary32 rounding shift out of domain");
        let left_power = 1u64 << (-shift).max(0);
        let divisor = 1u64 << shift.clamp(0, 37);
        let quotient = value * left_power / divisor;
        let remainder = value * left_power % divisor;
        ensure!(
            quotient < 1 << 24 && (binade == 1 || quotient >= 1 << 23),
            "noncanonical binary32 binade"
        );
        let parity = quotient & 1;
        let comparison = 2 * remainder as i64 - divisor as i64
            + i64::from(has_tail) * (1 - 2 * i64::from(tail_negative))
            + i64::from(!has_tail) * parity as i64;
        let round_up = comparison >= 1;
        let slack = if round_up { comparison - 1 } else { -comparison } as u64;
        let carry = quotient + u64::from(round_up) == 1 << 24;
        let significand = quotient + u64::from(round_up) - (u64::from(carry) << 23);
        let is_exponent_zero = significand < 1 << 23;
        let exponent = if is_exponent_zero {
            0
        } else {
            binade as u64 + u64::from(carry)
        };
        ensure!(exponent <= 254, "binary32 rounding overflow");
        let fraction = significand - (u64::from(!is_exponent_zero) << 23);
        Ok(Self {
            binade: F::from_canonical_u64(binade as u64),
            is_first_binade: F::from_bool(binade == 1),
            left_shift_power: F::from_canonical_u64(left_power),
            rounding_divisor: F::from_canonical_u64(divisor),
            quotient: limbs(quotient),
            remainder: limbs(remainder),
            remainder_complement: limbs(divisor - 1 - remainder),
            quotient_parity: F::from_canonical_u64(parity),
            round_up: F::from_bool(round_up),
            comparison_slack: limbs(slack),
            comparison_complement: limbs(divisor + 1 - slack),
            has_exponent_carry: F::from_bool(carry),
            output: Fp32Magnitude::from_word(((exponent as u32) << 23) | fraction as u32),
        })
    }
}

pub(crate) fn evaluate_rounding<V: Copy, S: Copy, E: Evaluator<V, S>>(
    evaluator: &mut E,
    columns: &Round32Columns<V>,
    value: V,
    has_tail: V,
    tail_negative: V,
) {
    let one = evaluator.i32(1);
    let two = evaluator.i32(2);
    let quotient = limb_value(evaluator, columns.quotient);
    let remainder = limb_value(evaluator, columns.remainder);
    let remainder_complement = limb_value(evaluator, columns.remainder_complement);
    let comparison_slack = limb_value(evaluator, columns.comparison_slack);
    let comparison_complement = limb_value(evaluator, columns.comparison_complement);
    evaluator.constraint_bool(columns.quotient_parity);
    evaluator.constraint_bool(columns.round_up);
    evaluator.constraint_bool(columns.has_exponent_carry);
    let dividend = evaluator.mul(value, columns.left_shift_power);
    let recomposed = evaluator.mad(quotient, columns.rounding_divisor, remainder);
    evaluator.constraint_eq(dividend, recomposed);
    let remainder_sum = evaluator.add(remainder, remainder_complement);
    let remainder_sum = evaluator.add(remainder_sum, one);
    evaluator.constraint_eq(remainder_sum, columns.rounding_divisor);
    let comparison = evaluator.msub(two, remainder, columns.rounding_divisor);
    let tail_sign = evaluator.msub(two, tail_negative, one);
    let tail_term = evaluator.mul(has_tail, tail_sign);
    let comparison = evaluator.sub(comparison, tail_term);
    let no_tail = evaluator.sub(one, has_tail);
    let comparison = evaluator.mad(no_tail, columns.quotient_parity, comparison);
    let rounding_sign = evaluator.msub(two, columns.round_up, one);
    let expected_slack = evaluator.msub(rounding_sign, comparison, columns.round_up);
    evaluator.constraint_eq(comparison_slack, expected_slack);
    let comparison_sum = evaluator.add(comparison_slack, comparison_complement);
    let divisor_plus_one = evaluator.add(columns.rounding_divisor, one);
    evaluator.constraint_eq(comparison_sum, divisor_plus_one);
    let incremented = evaluator.add(quotient, columns.round_up);
    let significand_limit = evaluator.u64(1 << 24);
    let carry_difference = evaluator.sub(incremented, significand_limit);
    let carry_constraint = evaluator.mul(columns.has_exponent_carry, carry_difference);
    evaluator.constraint(carry_constraint);
    let normal = evaluator.sub(one, columns.output.is_exponent_zero);
    let implicit_bit = evaluator.u64(1 << 23);
    let fraction_weight = evaluator.u64(1 << 16);
    let fraction = evaluator.mad(columns.output.fraction_high, fraction_weight, columns.output.fraction_low);
    let significand = evaluator.mad(normal, implicit_bit, fraction);
    let carry_amount = evaluator.mul(columns.has_exponent_carry, implicit_bit);
    let rounded = evaluator.sub(incremented, carry_amount);
    evaluator.constraint_eq(significand, rounded);
    let carried_binade = evaluator.add(columns.binade, columns.has_exponent_carry);
    let exponent = evaluator.mul(normal, carried_binade);
    evaluator.constraint_eq(columns.output.exponent, exponent);
    let binade_difference = evaluator.sub(columns.binade, one);
    let zero_binade_constraint = evaluator.mul(binade_difference, columns.output.is_exponent_zero);
    evaluator.constraint(zero_binade_constraint);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_alignment<V: Copy, S: Copy, E: Evaluator<V, S>>(
    evaluator: &mut E,
    columns: &AlignmentColumns<V>,
    coefficient: &Fp32Magnitude<V>,
    noise: &Fp32Fields<V>,
    beta: [V; 3],
    absolute_integer: V,
    signal_sign: V,
) -> V {
    let zero = evaluator.i32(0);
    let one = evaluator.i32(1);
    let two = evaluator.i32(2);
    let eight = evaluator.i32(8);
    let fraction_weight = evaluator.u64(1 << 16);
    let implicit_bit = evaluator.u64(1 << 23);
    let coefficient_normal = evaluator.sub(one, coefficient.is_exponent_zero);
    let coefficient_fraction = evaluator.mad(coefficient.fraction_high, fraction_weight, coefficient.fraction_low);
    let coefficient_significand = evaluator.mad(coefficient_normal, implicit_bit, coefficient_fraction);
    let signal = evaluator.mul(coefficient_significand, absolute_integer);
    evaluator.constraint_eq(columns.signal_significand, signal);
    let beta_normal = evaluator.sub(one, beta[2]);
    let beta_implicit = evaluator.i32(128);
    let beta_significand = evaluator.mad(beta_normal, beta_implicit, beta[1]);
    let noise_normal = evaluator.sub(one, noise.is_exponent_zero);
    let noise_significand = evaluator.mad(noise_normal, implicit_bit, noise.fraction);
    let product = evaluator.mul(beta_significand, noise_significand);
    let product_zero_constraint = evaluator.mul(columns.is_product_zero, product);
    evaluator.constraint(product_zero_constraint);
    let normalized_product = limb_value(evaluator, columns.product_normalized);
    let expected_product = evaluator.mul(product, columns.product_normalization_power);
    evaluator.constraint_eq(normalized_product, expected_product);
    let beta_exponent_star = evaluator.add(beta[0], beta[2]);
    let noise_exponent_star = evaluator.add(noise.exponent, noise.is_exponent_zero);
    let product_exponent = evaluator.add(beta_exponent_star, noise_exponent_star);
    let product_exponent = evaluator.add(product_exponent, columns.product_width);
    let product_bias = evaluator.i32(15);
    let product_exponent = evaluator.add(product_exponent, product_bias);
    let product_nonzero = evaluator.sub(one, columns.is_product_zero);
    let product_exponent = evaluator.mul(product_nonzero, product_exponent);
    evaluator.constraint_eq(columns.product_exponent, product_exponent);
    let signal_zero_constraint = evaluator.mul(columns.is_signal_zero, columns.signal_significand);
    evaluator.constraint(signal_zero_constraint);
    let normalized_signal = limb_value(evaluator, columns.signal_normalized);
    let expected_signal = evaluator.mul(columns.signal_significand, columns.signal_normalization_power);
    evaluator.constraint_eq(normalized_signal, expected_signal);
    let coefficient_exponent_star = evaluator.add(coefficient.exponent, coefficient.is_exponent_zero);
    let signal_exponent = evaluator.add(coefficient_exponent_star, columns.signal_width);
    let signal_bias = evaluator.i32(149);
    let signal_exponent = evaluator.add(signal_exponent, signal_bias);
    let signal_nonzero = evaluator.sub(one, columns.is_signal_zero);
    let signal_exponent = evaluator.mul(signal_nonzero, signal_exponent);
    evaluator.constraint_eq(columns.signal_exponent, signal_exponent);
    evaluator.constraint_bool(columns.product_is_larger);
    let order_sign = evaluator.msub(two, columns.product_is_larger, one);
    let exponent_difference = evaluator.sub(columns.product_exponent, columns.signal_exponent);
    let gap = evaluator.mul(order_sign, exponent_difference);
    evaluator.constraint_eq(columns.exponent_gap, gap);
    for (column, first, second) in [
        (columns.larger_significand, normalized_signal, normalized_product),
        (columns.smaller_significand, normalized_product, normalized_signal),
        (columns.larger_sign, signal_sign, noise.sign),
        (columns.smaller_sign, noise.sign, signal_sign),
    ] {
        let selected = evaluator.mux(columns.product_is_larger, first, second);
        evaluator.constraint_eq(column, selected);
    }
    let quotient = limb_value(evaluator, columns.quotient);
    let remainder = limb_value(evaluator, columns.remainder);
    let remainder_complement = limb_value(evaluator, columns.remainder_complement);
    let quotient_complement = limb_value(evaluator, columns.quotient_complement);
    let dividend = evaluator.mul(eight, columns.smaller_significand);
    let recomposed = evaluator.mad(quotient, columns.alignment_divisor, remainder);
    evaluator.constraint_eq(dividend, recomposed);
    let remainder_sum = evaluator.add(remainder, remainder_complement);
    let remainder_sum = evaluator.add(remainder_sum, one);
    evaluator.constraint_eq(remainder_sum, columns.alignment_divisor);
    let quotient_sum = evaluator.add(quotient, quotient_complement);
    let quotient_sum = evaluator.add(quotient_sum, one);
    evaluator.constraint_eq(quotient_sum, columns.quotient_limit);
    evaluator.constraint_bool(columns.output_sign);
    let magnitude = limb_value(evaluator, columns.magnitude);
    let output_factor = evaluator.msub(two, columns.output_sign, one);
    let output_factor = evaluator.sub(zero, output_factor);
    let signed_magnitude = evaluator.mul(output_factor, magnitude);
    let larger_factor = evaluator.msub(two, columns.larger_sign, one);
    let larger_factor = evaluator.sub(zero, larger_factor);
    let larger_magnitude = evaluator.mul(eight, columns.larger_significand);
    let signed_larger = evaluator.mul(larger_factor, larger_magnitude);
    let smaller_factor = evaluator.msub(two, columns.smaller_sign, one);
    let smaller_factor = evaluator.sub(zero, smaller_factor);
    let signed_sum = evaluator.mad(smaller_factor, quotient, signed_larger);
    evaluator.constraint_eq(signed_magnitude, signed_sum);
    let magnitude_inverse_product = evaluator.mul(magnitude, columns.magnitude_inverse);
    let nonzero = evaluator.sub(one, columns.is_zero);
    evaluator.constraint_eq(magnitude_inverse_product, nonzero);
    let zero_constraint = evaluator.mul(magnitude, columns.is_zero);
    evaluator.constraint(zero_constraint);
    let remainder_inverse_product = evaluator.mul(remainder, columns.tail_inverse);
    evaluator.constraint_eq(remainder_inverse_product, columns.has_tail);
    let no_tail = evaluator.sub(one, columns.has_tail);
    let no_tail_constraint = evaluator.mul(remainder, no_tail);
    evaluator.constraint(no_tail_constraint);
    let tail_negative = evaluator.xor_bit(columns.larger_sign, columns.smaller_sign);
    evaluator.constraint_eq(columns.tail_negative, tail_negative);
    let negative_product = evaluator.mul(columns.is_product_zero, noise.sign);
    evaluator.constraint_eq(columns.negative_zero_product, negative_product);
    let negative_signal = evaluator.mul(columns.is_signal_zero, signal_sign);
    evaluator.constraint_eq(columns.negative_zero_signal, negative_signal);
    let negative_zero = evaluator.mul(columns.negative_zero_product, columns.negative_zero_signal);
    evaluator.constraint_eq_if(columns.is_zero, columns.output_sign, negative_zero);
    let larger_exponent = evaluator.mux(columns.product_is_larger, columns.signal_exponent, columns.product_exponent);
    let common_bias = evaluator.i32(334);
    let binary_scale = evaluator.sub(larger_exponent, common_bias);
    let binary_scale = evaluator.mul(nonzero, binary_scale);
    let zero_bias = evaluator.i32(149);
    let zero_scale = evaluator.mul(columns.is_zero, zero_bias);
    let binary_scale = evaluator.sub(binary_scale, zero_scale);
    evaluator.constraint_eq(columns.binary_scale, binary_scale);
    magnitude
}

pub(crate) fn alignment_lookups<F: Field>(columns: &AlignmentColumns<usize>) -> Vec<LutLookup<F>> {
    let mut result = Vec::with_capacity(26);
    for column in columns
        .product_normalized
        .into_iter()
        .chain(columns.signal_normalized)
        .chain(columns.quotient)
        .chain(columns.quotient_complement)
        .chain(columns.remainder)
        .chain(columns.remainder_complement)
        .chain(columns.magnitude)
    {
        result.push(LutLookup::rc16(Column::single(column)));
    }
    result.push(LutLookup::rc16(Column::linear_combination([(
        columns.magnitude[2],
        F::from_canonical_u64(4096),
    )])));
    for (high, is_zero) in [
        (columns.product_normalized[1], columns.is_product_zero),
        (columns.signal_normalized[1], columns.is_signal_zero),
    ] {
        result.push(LutLookup::rc16_filtered(
            Column::linear_combination_with_constant([(high, F::TWO)], -F::from_canonical_u64(65536)),
            Filter::from_column(Column::linear_combination_with_constant([(is_zero, -F::ONE)], F::ONE)),
        ));
    }
    result.push(LutLookup::rc16(Column::linear_combination_with_constant(
        [(columns.signal_exponent, -F::ONE)],
        F::from_canonical_u64(427),
    )));
    for (operation, argument, outputs) in [
        (
            0,
            columns.product_width,
            [columns.product_normalization_power, columns.is_product_zero],
        ),
        (
            0,
            columns.signal_width,
            [columns.signal_normalization_power, columns.is_signal_zero],
        ),
        (2, columns.exponent_gap, [columns.alignment_divisor, columns.quotient_limit]),
    ] {
        result.push(LutLookup {
            table: LutTable::Fp32Tools,
            keys: vec![Column::constant(F::from_canonical_u64(operation)), Column::single(argument)],
            values: Column::singles(outputs).collect(),
            filter: Filter::default(),
        });
    }
    result
}

pub(crate) fn evaluate_proxy<V: Copy, S: Copy, E: Evaluator<V, S>>(
    evaluator: &mut E,
    columns: &OddProxyColumns<V>,
    output: &Fp32Magnitude<V>,
    sign: V,
) {
    let one = evaluator.i32(1);
    evaluator.constraint_bool(columns.high_fraction_parity);
    evaluator.constraint_bool(columns.low_fraction_is_nonzero);
    let inverse_product = evaluator.mul(output.fraction_low, columns.low_fraction_inverse);
    evaluator.constraint_eq(inverse_product, columns.low_fraction_is_nonzero);
    let low_zero = evaluator.sub(one, columns.low_fraction_is_nonzero);
    let zero_constraint = evaluator.mul(output.fraction_low, low_zero);
    evaluator.constraint(zero_constraint);
    let sign_weight = evaluator.i32(32768);
    let exponent_weight = evaluator.i32(128);
    let code = evaluator.mad(output.exponent, exponent_weight, output.fraction_high);
    let code = evaluator.mad(sign, sign_weight, code);
    let even = evaluator.sub(one, columns.high_fraction_parity);
    let code = evaluator.mad(columns.low_fraction_is_nonzero, even, code);
    evaluator.constraint_eq(columns.code, code);
}

pub(crate) fn rounding_lookups<F: Field>(columns: &Round32Columns<usize>, scale: usize) -> Vec<LutLookup<F>> {
    let mut result = Vec::with_capacity(22);
    for column in columns
        .quotient
        .into_iter()
        .chain(columns.remainder)
        .chain(columns.remainder_complement)
        .chain(columns.comparison_slack)
        .chain(columns.comparison_complement)
        .chain([columns.output.fraction_low])
    {
        result.push(LutLookup::rc16(Column::single(column)));
    }
    result.push(LutLookup::rc16(Column::linear_combination([(
        columns.quotient[1],
        F::from_canonical_u64(256),
    )])));
    let half = F::TWO.inverse();
    result.push(LutLookup::rc16(Column::linear_combination([
        (columns.quotient[0], half),
        (columns.quotient_parity, -half),
    ])));
    result.push(LutLookup::rc16_filtered(
        Column::linear_combination_with_constant(
            [(columns.quotient[1], F::from_canonical_u64(512))],
            -F::from_canonical_u64(65536),
        ),
        Filter::from_column(Column::linear_combination_with_constant(
            [(columns.is_first_binade, -F::ONE)],
            F::ONE,
        )),
    ));
    for (key, output) in [
        (
            Column::linear_combination_with_constant([(columns.binade, F::ONE)], -F::ONE),
            columns.is_first_binade,
        ),
        (Column::single(columns.output.exponent), columns.output.is_exponent_zero),
    ] {
        result.push(LutLookup {
            table: LutTable::ExpInfo,
            keys: vec![key],
            values: vec![Column::single(output)],
            filter: Filter::default(),
        });
    }
    result.push(LutLookup {
        table: LutTable::Pair128,
        keys: vec![Column::single(columns.output.fraction_high), Column::constant(F::ZERO)],
        values: vec![],
        filter: Filter::default(),
    });
    let shift =
        Column::linear_combination_with_constant([(columns.binade, F::ONE), (scale, -F::ONE)], -F::from_canonical_u64(150));
    result.push(LutLookup {
        table: LutTable::Fp32Tools,
        keys: vec![Column::constant(F::ONE), shift],
        values: Column::singles([columns.left_shift_power, columns.rounding_divisor]).collect(),
        filter: Filter::default(),
    });
    result
}
