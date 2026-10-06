//! IEEE-754 half precision (FP16, E5M10) decode/encode and the integer
//! decomposition the A100 accumulation model consumes.
//!
//! FP16 is bias 15, 5 exponent bits, 10 mantissa bits. Every finite FP16 value
//! is exactly representable in f32, so decoding to f32 and multiplying on the
//! f32 datapath reproduces the hardware product bit-for-bit (the A100 tensor
//! core promotes FP16 operands to an f32 product; see [`super::accumulate`]).
//! NaN and infinity are banned, matching the rest of the proof arithmetic.

/// Decodes an FP16 (E5M10) bit pattern to f32. Exact. Panics on NaN/±inf.
pub fn fp16_to_f32(bits: u16) -> f32 {
    let exp = (bits >> 10) & 0x1F;
    let man = bits & 0x03FF;
    assert!(exp != 0x1F, "FP16 NaN/inf encoding {bits:#06x} is not allowed");
    let sign = ((bits & 0x8000) as u32) << 16;
    let magnitude = match (exp, man) {
        (0, 0) => 0,
        (0, _) => {
            // Subnormal: value = man * 2^-24. Renormalize into a normal f32.
            let man = man as u32;
            let hb = 31 - man.leading_zeros(); // index of the top set bit, 0..=9
            let exp_biased = hb + 103; // f32 bias of 2^(hb-24)
            let man_f32 = (man ^ (1 << hb)) << (23 - hb);
            (exp_biased << 23) | man_f32
        }
        _ => ((exp as u32 + 112) << 23) | ((man as u32) << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// Encodes a finite f32 as FP16 (E5M10), round-to-nearest-ties-to-even,
/// saturating to the largest finite magnitude (65504) on overflow. On values
/// FP16 can represent exactly this is the inverse of [`fp16_to_f32`]. Returns
/// `None` on NaN or infinity.
pub fn f32_to_fp16(x: f32) -> Option<u16> {
    if !x.is_finite() {
        return None;
    }
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let m = x.abs();
    if m == 0.0 {
        return Some(sign);
    }
    let bits = m.to_bits();
    let e = (bits >> 23) as i32 - 127; // unbiased exponent of |x|
    let man23 = bits & 0x7F_FFFF;
    // Round the 24-bit significand (implicit 1 + 23 bits) to 10 fractional bits.
    let magnitude: u16 = if e < -14 {
        // Subnormal FP16 (or underflow to zero): value = q * 2^-24.
        let full = 0x80_0000 | man23; // 24-bit significand at scale 2^(e-23)
        let rshift = (-14 - e) as u32 + 13; // align to the 2^-24 grid
        if rshift >= 32 {
            0
        } else {
            round_shift(full as u64, rshift) as u16
        }
    } else if e > 15 {
        0x7BFF // saturate to 65504
    } else {
        let full = 0x80_0000 | man23;
        let q = round_shift(full as u64, 13) as u32; // keep 11 significand bits
        // round_shift may carry into bit 11 (q == 0x800), bumping the exponent.
        let e_adj = e + (q >> 11) as i32;
        if e_adj > 15 {
            0x7BFF
        } else {
            (((e_adj + 15) as u16) << 10) | ((q as u16) & 0x03FF)
        }
    };
    Some(sign | magnitude)
}

/// `x >> shift`, rounded to nearest with ties to even.
fn round_shift(x: u64, shift: u32) -> u64 {
    if shift == 0 {
        return x;
    }
    let dropped = x & ((1u64 << shift) - 1);
    let kept = x >> shift;
    let half = 1u64 << (shift - 1);
    if dropped > half || (dropped == half && (kept & 1) == 1) {
        kept + 1
    } else {
        kept
    }
}

/// Integer decomposition of an FP16 operand: `value = sign * m * 2^(eps - 10)`,
/// where `eps` is the stored (unbiased) exponent clamped to the format minimum
/// `-14` for subnormals, and `m` is the integer significand (11 bits for
/// normals, `1..=1023` for subnormals, `0` for zero). `sign` is `+1`/`-1`, or
/// `+1` for zero. This matches the `decompose` of the validated reference
/// emulator. Panics on NaN/±inf.
pub fn decompose_fp16(bits: u16) -> (i64, i64, i32) {
    let exp = (bits >> 10) & 0x1F;
    let man = (bits & 0x03FF) as i64;
    assert!(exp != 0x1F, "FP16 NaN/inf encoding {bits:#06x} is not allowed");
    let sign: i64 = if bits & 0x8000 != 0 { -1 } else { 1 };
    match (exp, man) {
        (0, 0) => (1, 0, 0),
        (0, _) => (sign, man, -14),          // subnormal: m = man, eps = -14
        _ => (sign, 0x400 | man, exp as i32 - 15), // normal: m = 1024+man
    }
}

/// The FP16DECODE committed-LUT fields of an operand code, in the order the A100 matmul AIR
/// binds them: `(sig, sign_bit, eps_biased, is_zero)` with
///
/// * `sig` the integer significand (`0`, `1..=1023` subnormal, `1024..=2047` normal) — equal
///   to the `m` of [`decompose_fp16`];
/// * `sign_bit` the raw sign bit (bit 15), used only on nonzero lanes (zero products drop it);
/// * `eps_biased = stored_exponent + 15 >= 0`, so a nonzero product's biased stored exponent
///   is `eps_biased(a) + eps_biased(b) + 97 = (ea + eb) + 127`;
/// * `is_zero = [sig == 0]`.
///
/// Matches [`decompose_fp16`] on every finite code. NaN/inf codes (exponent field `0x1F`)
/// decode through the normal formula (`sig = 0x400 | man`, `eps = 16`); they never occur in
/// the scheme (operand finiteness is enforced upstream, as FP8 QCAST saturation does), and the
/// table is total so the lookup argument stays well defined.
pub fn fp16_decode_fields(code: u16) -> (u64, u64, u64, u64) {
    let exp = (code >> 10) & 0x1F;
    let man = u64::from(code & 0x03FF);
    let sign_bit = u64::from(code >> 15);
    let (sig, eps): (u64, i64) = match (exp, man) {
        (0, 0) => (0, 0),
        (0, _) => (man, -14),
        _ => (0x400 | man, exp as i64 - 15),
    };
    (sig, sign_bit, (eps + 15) as u64, u64::from(sig == 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_decode_fields_match_decompose() {
        for bits in 0u16..=0xFFFF {
            if (bits >> 10) & 0x1F == 0x1F {
                continue; // NaN/inf: decompose panics; the table uses the normal formula
            }
            let (sig, sign_bit, eps_biased, is_zero) = fp16_decode_fields(bits);
            let (s, m, eps) = decompose_fp16(bits);
            assert_eq!(sig, m as u64, "sig {bits:#06x}");
            assert_eq!(eps_biased, (eps + 15) as u64, "eps {bits:#06x}");
            assert_eq!(is_zero, u64::from(m == 0), "is_zero {bits:#06x}");
            // sign_bit is the raw bit; decompose forces +1 for zero, so only compare nonzero.
            if m != 0 {
                assert_eq!(sign_bit, u64::from(s < 0), "sign {bits:#06x}");
            }
        }
        // The biased stored exponent of a nonzero product reconstructs ea+eb+127 from the two
        // operands' eps_biased fields: eps_biased(a) + eps_biased(b) + 97.
        for (ca, cb) in [(0x3C00u16, 0x4000u16), (0x0001, 0x7BFF), (0xC000, 0x3800)] {
            let (_, _, ea_b, _) = fp16_decode_fields(ca);
            let (_, _, eb_b, _) = fp16_decode_fields(cb);
            let (_, _, ea) = decompose_fp16(ca);
            let (_, _, eb) = decompose_fp16(cb);
            assert_eq!(ea_b + eb_b + 97, (ea + eb + 127) as u64, "biased exp link {ca:#06x},{cb:#06x}");
        }
    }

    #[test]
    fn fp16_roundtrip_exhaustive() {
        // Every finite FP16 value decodes to f32 and re-encodes to itself.
        for bits in 0u16..=0xFFFF {
            let exp = (bits >> 10) & 0x1F;
            if exp == 0x1F {
                continue; // NaN/inf
            }
            let f = fp16_to_f32(bits);
            // Normalize -0 and +0: both encode from their own bit pattern.
            let re = f32_to_fp16(f).unwrap();
            assert_eq!(re, bits, "roundtrip {bits:#06x} -> {f} -> {re:#06x}");
        }
    }

    #[test]
    fn decompose_reconstructs_value() {
        for bits in 0u16..=0xFFFF {
            let exp = (bits >> 10) & 0x1F;
            if exp == 0x1F {
                continue;
            }
            let (s, m, eps) = decompose_fp16(bits);
            let v = (s as f64) * (m as f64) * 2f64.powi(eps - 10);
            assert_eq!(v as f32, fp16_to_f32(bits), "decompose {bits:#06x}");
        }
    }

    #[test]
    fn known_values() {
        assert_eq!(fp16_to_f32(0x3C00), 1.0); // 1.0
        assert_eq!(fp16_to_f32(0x4000), 2.0); // 2.0
        assert_eq!(fp16_to_f32(0xC000), -2.0);
        assert_eq!(fp16_to_f32(0x7BFF), 65504.0); // max finite
        assert_eq!(fp16_to_f32(0x0001), 2f32.powi(-24)); // min subnormal
        assert_eq!(f32_to_fp16(1.0), Some(0x3C00));
        assert_eq!(f32_to_fp16(65505.0), Some(0x7BFF)); // saturate
        assert_eq!(f32_to_fp16(f32::INFINITY), None);
    }
}
