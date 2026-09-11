//! BF16 arithmetic for operand construction. Multiply and divide use f32
//! intermediates; [`bf16_fma`] preserves a single rounding to BF16.

use anyhow::Result;

use super::dtype::{bf16_to_f32, f32_to_bf16};

/// Apply `op` in f32 and round to BF16, ties to even.
/// Returns an error if the result is non-finite or overflows BF16.
fn bf16_op(a: u16, b: u16, op: impl Fn(f32, f32) -> f32) -> Result<u16> {
    f32_to_bf16(op(bf16_to_f32(a), bf16_to_f32(b)))
}

pub fn bf16_mul(a: u16, b: u16) -> Result<u16> {
    bf16_op(a, b, |x, y| x * y)
}

pub fn bf16_div(a: u16, b: u16) -> Result<u16> {
    bf16_op(a, b, |x, y| x / y)
}

/// BF16 minimum; returns one input without rounding.
pub fn bf16_min(a: u16, b: u16) -> u16 {
    if bf16_to_f32(a) <= bf16_to_f32(b) { a } else { b }
}

/// BF16 maximum; returns one input without rounding.
pub fn bf16_max(a: u16, b: u16) -> u16 {
    if bf16_to_f32(a) >= bf16_to_f32(b) { a } else { b }
}

/// Clamp to `[-hi, hi]`, with `hi >= 0`.
pub fn bf16_clamp_sym(x: u16, hi: u16) -> u16 {
    bf16_max(bf16_min(x, hi), hi ^ 0x8000)
}

/// `a*b + c` with one rounding to BF16.
///
/// The BF16 significands have eight bits each, so their 16-bit product is exact
/// in f64. TwoSum recovers the addition error, giving the real-valued identity
/// `a*b + c = rounded_sum + sum_residual`.
///
/// Rounding first to f32 can lose which side of a BF16 midpoint the sum lies
/// on. A sum just above a midpoint may round to that midpoint in f32; BF16
/// then treats it as a tie and may incorrectly round down.
///
/// For an inexact finite f32 result, we force the last significand bit to 1
/// ("round to odd"). If it is currently 0, move one f32 step toward the exact
/// sum. The sign of
/// `rounding_error = (rounded_sum - rounded_f32) + sum_residual`
/// chooses the direction: positive means up, negative means down.
/// BF16 midpoints have that bit 0, so inexact values cannot become false ties.
/// Exact f32 results, including genuine BF16 midpoints, are left unchanged.
///
/// Returns an error on a non-finite result.
pub fn bf16_fma(a: u16, b: u16, c: u16) -> Result<u16> {
    let a64 = bf16_to_f32(a) as f64;
    let b64 = bf16_to_f32(b) as f64;
    let c64 = bf16_to_f32(c) as f64;
    // Two eight-bit BF16 significands have an exact 16-bit product in f64.
    let exact_product = a64 * b64;
    // TwoSum recovers the addition error: a*b + c = rounded_sum + sum_residual.
    let rounded_sum = exact_product + c64;
    let recovered_product = rounded_sum - c64;
    let sum_residual = (exact_product - recovered_product) + (c64 - (rounded_sum - recovered_product));
    let rounded_f32 = rounded_sum as f32;
    // Include both rounding errors to locate the exact sum relative to its f32 approximation.
    let rounding_error = (rounded_sum - rounded_f32 as f64) + sum_residual;
    // An odd low bit cannot represent a BF16 midpoint; move toward the exact sum if needed.
    let needs_odd_rounding = rounding_error != 0.0 && (rounded_f32.to_bits() & 1 == 0) && rounded_f32.is_finite();
    let odd_rounded_f32 = if needs_odd_rounding {
        if rounding_error > 0.0 {
            rounded_f32.next_up()
        } else {
            rounded_f32.next_down()
        }
    } else {
        rounded_f32
    };
    f32_to_bf16(odd_rounded_f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected BF16 results generated with PyTorch using f64 intermediates.
    #[test]
    fn bf16_fma_matches_reference_vectors() {
        // (a, b, c) -> expected, all as BF16 bit patterns.
        let cases: &[(u16, u16, u16, u16)] = &[
            (0x3f80, 0x4000, 0x3f80, 0x4040), // 1*2 + 1 = 3
            (0x4049, 0x3fcc, 0xbf80, 0x4080), // 3.14*1.59 - 1 ~= 4
            (0x3f00, 0x3f00, 0x3f00, 0x3f40), // 0.5*0.5 + 0.5 = 0.75
            (0x4228, 0x3d23, 0x40b0, 0x40e5),
            (0xbf80, 0x4000, 0x4080, 0x4000), // -1*2 + 4 = 2
            (0x4315, 0x3f80, 0x3b95, 0x4315), // alpha*1 + tiny ~= alpha
        ];
        for &(a, b, c, want) in cases {
            assert_eq!(bf16_fma(a, b, c).unwrap(), want, "fma({a:#06x}, {b:#06x}, {c:#06x})");
        }
    }

    #[test]
    fn bf16_max_min_are_exact() {
        assert_eq!(bf16_max(0x3f80, 0x4000), 0x4000); // max(1, 2) = 2
        assert_eq!(bf16_min(0x3f80, 0x4000), 0x3f80); // min(1, 2) = 1
        // A tiny value vs the 2^-32 floor used by the quant scheme.
        let floor = f32_to_bf16(2.0f32.powi(-32)).unwrap();
        assert_eq!(bf16_max(0x0000, floor), floor);
    }
}
