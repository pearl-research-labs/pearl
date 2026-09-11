//! FP8 E4M3, BF16 and f32 conversions used by operand construction and replay.
//!
//! FP8 and BF16 values are stored as raw `u8` and `u16` codes. Conversions to
//! either format round to nearest, ties to even. E4M3 overflow saturates to
//! +/-448; BF16 overflow returns an error. Arithmetic and decoders reject
//! non-finite values.

use super::utils::Dtype;
use anyhow::Result;

pub struct Fp8E4M3;
pub struct Bf16;
pub struct Fp32;

/// Panics if `x` is NaN or ±inf; returns it unchanged otherwise.
fn assert_finite(x: f32) -> f32 {
    assert!(x.is_finite(), "non-finite value {x} is not allowed in Dtype arithmetic");
    x
}

/// Decodes an FP8 E4M3 value (bias 7, no infinities,
/// `S.1111.111` is the only NaN encoding) to f32. Exact.
/// Panics on the NaN encoding.
pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    assert!(bits & 0x7F != 0x7F, "FP8 E4M3 NaN encoding {bits:#04x} is not allowed");
    let sign = ((bits & 0x80) as u32) << 24;
    let biased_exponent = ((bits >> 3) & 0x0F) as u32;
    let fraction_bits = (bits & 0x07) as u32;
    let magnitude = match (biased_exponent, fraction_bits) {
        (0, 0) => 0,
        (0, _) => {
            // Subnormal (`fraction_bits * 2^-9`): normalize into a normal f32.
            let normalization_shift = 3 - (31 - fraction_bits.leading_zeros());
            ((121 - normalization_shift) << 23) | (((fraction_bits << normalization_shift) & 0x07) << 20)
        }
        _ => ((biased_exponent + 120) << 23) | (fraction_bits << 20),
    };
    f32::from_bits(sign | magnitude)
}

/// Round a finite f32 to E4M3, nearest with ties to even, saturating at ±448.
/// Returns an error for NaN or infinity.
pub fn f32_to_fp8_e4m3(x: f32) -> Result<u8> {
    if !x.is_finite() {
        return Err(anyhow::anyhow!("non-finite value {x} unsupported"));
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let abs_value = x.abs();
    let bits = abs_value.to_bits();
    let unbiased_exponent = (bits >> 23) as i32 - 127; // unbiased exponent of |x|

    let magnitude: u8 = if unbiased_exponent < -6 {
        // Subnormal region: every E4M3 value below 2^-6 is a multiple of 2^-9,
        // so the encoding is |x| / 2^-9 rounded to nearest even. |x| * 512 is
        // exact, and a value rounding up to 8 lands on the smallest normal
        // (exp field 1) — the encoding is contiguous across that boundary.
        (abs_value * 512.0).round_ties_even() as u8
    } else {
        // Normal values: retain three fraction bits and round the discarded bits, ties to even.
        let mut biased_exponent = (unbiased_exponent + 7) as u32;
        let fraction_bits = bits & 0x7F_FFFF;
        let mut retained_fraction = fraction_bits >> 20;
        let discarded_fraction = fraction_bits & 0xF_FFFF;
        if discarded_fraction > 0x8_0000 || (discarded_fraction == 0x8_0000 && retained_fraction & 1 == 1) {
            retained_fraction += 1;
            if retained_fraction == 8 {
                retained_fraction = 0;
                biased_exponent += 1;
            }
        }
        if biased_exponent > 15 || (biased_exponent == 15 && retained_fraction >= 7) {
            0x7E // saturate to max finite (448); avoids the S.1111.111 NaN slot
        } else {
            ((biased_exponent << 3) | retained_fraction) as u8
        }
    };
    Ok(sign | magnitude)
}

/// Round f32 to nearest BF16, ties to even. Reject non-finite inputs and BF16 overflow.
pub fn f32_to_bf16(x: f32) -> Result<u16> {
    if !x.is_finite() {
        return Err(anyhow::anyhow!("non-finite value {x} unsupported"));
    }
    let bits = x.to_bits();
    // Round to nearest, ties to even.
    let retained_lsb = (bits >> 16) & 1;
    let bf16 = ((bits + 0x7FFF + retained_lsb) >> 16) as u16;
    // Values above bf16 max round to the infinity encoding.
    if bf16 & 0x7FFF >= 0x7F80 {
        return Err(anyhow::anyhow!("value {x} overflows bf16"));
    }
    Ok(bf16)
}

/// Decodes a bfloat16 value to f32. Exact: bf16 is the top 16 bits of f32.
/// Panics on NaN and infinity encodings.
pub fn bf16_to_f32(bits: u16) -> f32 {
    assert!(bits & 0x7FFF < 0x7F80, "bf16 non-finite encoding {bits:#06x} is not allowed");
    f32::from_bits((bits as u32) << 16)
}

/// Errors unless `bits` encodes a finite BF16 value.
pub fn check_not_nan_or_inf_bf16(bits: u16) -> Result<()> {
    if bits & 0x7F80 == 0x7F80 {
        return Err(anyhow::anyhow!("bf16 value {bits:#06x} is not finite (NaN or infinity)"));
    }
    Ok(())
}

/// Errors unless `x` is a finite f32.
pub fn check_not_nan_or_inf_f32(x: f32) -> Result<()> {
    if !x.is_finite() {
        return Err(anyhow::anyhow!("f32 value {x} is not finite (NaN or infinity)"));
    }
    Ok(())
}

/// Raw f32 exponent field: 0 for zero/subnormals, 0xFF for NaN/infinity.
/// Ignores the sign and fraction.
pub fn f32_biased_exponent(x: f32) -> u32 {
    (x.to_bits() >> 23) & 0xFF
}

/// Convert E4M3 to BF16 exactly, including subnormals.
/// Panics on E4M3 NaN (via `fp8_e4m3_to_f32`).
pub fn fp8_to_bf16(bits: u8) -> u16 {
    f32_to_bf16(fp8_e4m3_to_f32(bits)).expect("every E4M3 value is representable in bf16")
}

pub fn batch_fp8_to_bf16(bits: &[u8]) -> Vec<u16> {
    bits.iter().map(|bits: &u8| fp8_to_bf16(*bits)).collect()
}

impl Dtype<u8, f32> for Fp8E4M3 {
    fn mul(&self, a: u8, b: u8) -> f32 {
        assert_finite(fp8_e4m3_to_f32(a) * fp8_e4m3_to_f32(b))
    }

    fn add(&self, a: u8, b: u8) -> f32 {
        assert_finite(fp8_e4m3_to_f32(a) + fp8_e4m3_to_f32(b))
    }
}

impl Dtype<u16, f32> for Bf16 {
    fn mul(&self, a: u16, b: u16) -> f32 {
        assert_finite(bf16_to_f32(a) * bf16_to_f32(b))
    }

    fn add(&self, a: u16, b: u16) -> f32 {
        assert_finite(bf16_to_f32(a) + bf16_to_f32(b))
    }
}

// Separate FP32 multiply and add round independently; they do not implement FMA.
impl Dtype<u32, f32> for Fp32 {
    fn mul(&self, a: u32, b: u32) -> f32 {
        assert_finite(f32::from_bits(a) * f32::from_bits(b))
    }

    fn add(&self, a: u32, b: u32) -> f32 {
        assert_finite(f32::from_bits(a) + f32::from_bits(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Arithmetic reference, derived independently from the format definition
    // ((1 + man/2^M) * 2^(exp-bias), all steps exact in f32) to cross-check
    // the bit-manipulation decoder.
    fn ref_decode(sign: bool, exp: i32, man: u32, man_width: i32, bias: i32) -> f32 {
        let s = if sign { -1.0f32 } else { 1.0 };
        let m = man as f32;
        let scale = 2f32.powi(man_width);
        if exp == 0 {
            s * m / scale * 2f32.powi(1 - bias)
        } else {
            s * (1.0 + m / scale) * 2f32.powi(exp - bias)
        }
    }

    #[test]
    fn fp8_e4m3_exhaustive() {
        for bits in 0..=u8::MAX {
            // The NaN encodings panic; covered by fp8_e4m3_nan_panics.
            if bits & 0x7F == 0x7F {
                continue;
            }
            let got = fp8_e4m3_to_f32(bits);
            let (exp, man) = (((bits >> 3) & 0x0F) as i32, (bits & 0x07) as u32);
            let want = ref_decode(bits & 0x80 != 0, exp, man, 3, 7);
            assert_eq!(got.to_bits(), want.to_bits(), "{bits:#04x}");
        }
        // Largest normal value: 448; smallest subnormal: 2^-9.
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        assert_eq!(fp8_e4m3_to_f32(0x01), 2f32.powi(-9));
    }

    #[test]
    fn f32_to_fp8_e4m3_round_trips() {
        // Encoding then decoding every non-NaN E4M3 value must recover it: the
        // decoded f32 is exactly representable, so it re-encodes to itself.
        for bits in 0..=u8::MAX {
            if bits & 0x7F == 0x7F {
                continue; // NaN slots
            }
            let f = fp8_e4m3_to_f32(bits);
            assert_eq!(f32_to_fp8_e4m3(f).unwrap(), bits, "{bits:#04x}");
        }
    }

    #[test]
    fn f32_to_fp8_e4m3_rounds_and_saturates() {
        // Rounds to nearest, ties to even.
        assert_eq!(f32_to_fp8_e4m3(1.0).unwrap(), 0x38);
        assert_eq!(f32_to_fp8_e4m3(-1.0).unwrap(), 0xB8);
        // 1.0 and its successor 1.125 (0x39) straddle 1.0625; the tie rounds to
        // even (0x38), just above rounds up (0x39).
        assert_eq!(f32_to_fp8_e4m3(1.0625).unwrap(), 0x38);
        assert_eq!(f32_to_fp8_e4m3(1.0626).unwrap(), 0x39);
        // Half the smallest subnormal (2^-10) ties to even, i.e. down to zero.
        assert_eq!(f32_to_fp8_e4m3(2f32.powi(-10)).unwrap(), 0x00);
        assert_eq!(f32_to_fp8_e4m3(2f32.powi(-9)).unwrap(), 0x01);
        // Overflow saturates to ±448 rather than the NaN/inf region.
        assert_eq!(f32_to_fp8_e4m3(1e30).unwrap(), 0x7E);
        assert_eq!(f32_to_fp8_e4m3(-1e30).unwrap(), 0xFE);
        assert_eq!(f32_to_fp8_e4m3(500.0).unwrap(), 0x7E);
        // Signed zeros.
        assert_eq!(f32_to_fp8_e4m3(0.0).unwrap(), 0x00);
        assert_eq!(f32_to_fp8_e4m3(-0.0).unwrap(), 0x80);
        // Non-finite is rejected.
        assert!(f32_to_fp8_e4m3(f32::NAN).is_err());
        assert!(f32_to_fp8_e4m3(f32::INFINITY).is_err());
    }

    #[test]
    fn check_not_nan_or_inf_bf16_classifies_correctly() {
        // Normal values, zeros, and subnormals are all accepted.
        assert!(check_not_nan_or_inf_bf16(f32_to_bf16(1.0).unwrap()).is_ok());
        assert!(check_not_nan_or_inf_bf16(f32_to_bf16(-2.5).unwrap()).is_ok());
        assert!(check_not_nan_or_inf_bf16(0x0000).is_ok()); // +0
        assert!(check_not_nan_or_inf_bf16(0x8000).is_ok()); // -0
        assert!(check_not_nan_or_inf_bf16(0x0080).is_ok()); // smallest normal
        assert!(check_not_nan_or_inf_bf16(0x0001).is_ok()); // smallest subnormal
        assert!(check_not_nan_or_inf_bf16(0x007F).is_ok()); // largest subnormal
        assert!(check_not_nan_or_inf_bf16(0x8001).is_ok()); // negative subnormal
        // Infinity and NaN (exponent field all ones) are rejected.
        assert!(check_not_nan_or_inf_bf16(0x7F80).is_err()); // +inf
        assert!(check_not_nan_or_inf_bf16(0xFF80).is_err()); // -inf
        assert!(check_not_nan_or_inf_bf16(0x7FC0).is_err()); // NaN
    }

    #[test]
    fn check_not_nan_or_inf_f32_classifies_correctly() {
        assert!(check_not_nan_or_inf_f32(1.0).is_ok());
        assert!(check_not_nan_or_inf_f32(-3.5e10).is_ok());
        assert!(check_not_nan_or_inf_f32(0.0).is_ok());
        assert!(check_not_nan_or_inf_f32(-0.0).is_ok());
        assert!(check_not_nan_or_inf_f32(f32::MIN_POSITIVE).is_ok()); // smallest normal
        assert!(check_not_nan_or_inf_f32(f32::MIN_POSITIVE / 2.0).is_ok()); // subnormal
        assert!(check_not_nan_or_inf_f32(f32::NAN).is_err());
        assert!(check_not_nan_or_inf_f32(f32::INFINITY).is_err());
        assert!(check_not_nan_or_inf_f32(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn f32_biased_exponent_matches_ieee754() {
        assert_eq!(f32_biased_exponent(1.0), 127); // 2^0
        assert_eq!(f32_biased_exponent(2.0), 128); // 2^1
        assert_eq!(f32_biased_exponent(0.5), 126); // 2^-1
        assert_eq!(f32_biased_exponent(-4.0), 129); // sign-agnostic, 2^2
        assert_eq!(f32_biased_exponent(0.0), 0);
        // For powers of two, exponent differences equal log2 of the magnitude ratio.
        assert_eq!(f32_biased_exponent(256.0) - f32_biased_exponent(1.0), 8);
    }

    #[test]
    #[should_panic(expected = "NaN encoding")]
    fn fp8_e4m3_nan_panics() {
        fp8_e4m3_to_f32(0x7F);
    }

    #[test]
    fn fp8_to_bf16_is_lossless() {
        for bits in 0..=u8::MAX {
            // The NaN encodings panic; covered by fp8_to_bf16_nan_panics.
            if bits & 0x7F == 0x7F {
                continue;
            }
            let bf16 = fp8_to_bf16(bits);
            assert_eq!(bf16_to_f32(bf16), fp8_e4m3_to_f32(bits), "{bits:#04x}");
        }
        assert_eq!(fp8_to_bf16(0x00), 0x0000);
        assert_eq!(fp8_to_bf16(0x38), 0x3F80); // 1.0
        assert_eq!(fp8_to_bf16(0x7E), 0x43E0); // 448.0
        assert_eq!(fp8_to_bf16(0x01), 0x3B00); // 2^-9 (E4M3 subnormal -> bf16 normal)
        assert_eq!(fp8_to_bf16(0xB8), 0xBF80); // -1.0
    }

    #[test]
    #[should_panic(expected = "NaN encoding")]
    fn fp8_to_bf16_nan_panics() {
        fp8_to_bf16(0xFF);
    }

    #[test]
    fn bf16_round_trip() {
        for bits in 0..=u16::MAX {
            // Non-finite encodings panic; covered by the tests below.
            if bits & 0x7FFF >= 0x7F80 {
                continue;
            }
            let got = bf16_to_f32(bits);
            assert_eq!(got.to_bits(), (bits as u32) << 16, "{bits:#06x}");
        }
    }

    #[test]
    #[should_panic(expected = "non-finite encoding")]
    fn bf16_nan_panics() {
        bf16_to_f32(0x7FC0);
    }

    #[test]
    #[should_panic(expected = "non-finite encoding")]
    fn bf16_inf_panics() {
        bf16_to_f32(0xFF80);
    }

    #[test]
    fn f32_to_bf16_rejects_non_finite() {
        assert!(f32_to_bf16(f32::NAN).is_err());
        assert!(f32_to_bf16(f32::INFINITY).is_err());
        assert!(f32_to_bf16(f32::NEG_INFINITY).is_err());
        // f32::MAX rounds up past bf16 max (0x7F7F), i.e. to the inf encoding.
        assert!(f32_to_bf16(f32::MAX).is_err());
        assert_eq!(f32_to_bf16(1.0).unwrap(), 0x3F80);
        assert_eq!(f32_to_bf16(bf16_to_f32(0x7F7F)).unwrap(), 0x7F7F);
    }

    #[test]
    fn dtype_ops() {
        // 448 * 448 is exact in f32; 0.875 * 2^-6 is the max E4M3 subnormal.
        assert_eq!(Fp8E4M3.mul(0x7E, 0x7E), 200704.0);
        assert_eq!(Fp8E4M3.add(0x07, 0x08), 0.875 * 2f32.powi(-6) + 2f32.powi(-6));
        // fp32 ops round to nearest-even.
        let (a, b) = (1.1f32, 2.2f32);
        assert_eq!(Fp32.mul(a.to_bits(), b.to_bits()), a * b);
        assert_eq!(Fp32.add(a.to_bits(), b.to_bits()), a + b);
    }

    #[test]
    #[should_panic(expected = "non-finite value")]
    fn overflowing_op_panics() {
        // bf16 max * max overflows f32 to infinity, which is banned.
        Bf16.mul(0x7F7F, 0x7F7F);
    }

    #[test]
    #[should_panic(expected = "non-finite value")]
    fn fp32_nan_input_panics() {
        // NaN operands propagate to a NaN result, caught by the output assert.
        Fp32.mul(f32::NAN.to_bits(), 0);
    }
}
