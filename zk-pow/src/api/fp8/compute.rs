//! Shared `compute_dtype` (BF16) element-wise primitives, mirroring the
//! reference `ComputeOps` (`miner_base.compute_ops`).
//!
//! These ops feed the bit-exact operand-building path (per-row scale derivation
//! and the noise lines, hence the lottery) but have NO accumulation order to pin
//! — they are plain element-wise arithmetic, identical on any device. Each input
//! decodes exactly to f32, the op runs in f32, and the result rounds back to
//! BF16 with ties-to-even: the same semantics as the reference's torch BF16
//! kernels.

use anyhow::Result;

use super::dtype::{bf16_to_f32, f32_to_bf16};

/// Element-wise BF16 op: decode both inputs exactly to f32, apply `op`, round
/// the result back to BF16 (round-to-nearest-even). Errors on a non-finite
/// result (overflow / NaN), matching the FP8 module's no-non-finite invariant.
fn bf16_op(a: u16, b: u16, op: impl Fn(f32, f32) -> f32) -> Result<u16> {
    f32_to_bf16(op(bf16_to_f32(a), bf16_to_f32(b)))
}

pub fn bf16_mul(a: u16, b: u16) -> Result<u16> {
    bf16_op(a, b, |x, y| x * y)
}

pub fn bf16_div(a: u16, b: u16) -> Result<u16> {
    bf16_op(a, b, |x, y| x / y)
}

/// BF16 minimum (`torch.minimum`). The comparison is exact and the result is one
/// of the inputs, so nothing rounds.
pub fn bf16_min(a: u16, b: u16) -> u16 {
    if bf16_to_f32(a) <= bf16_to_f32(b) { a } else { b }
}

/// BF16 maximum (`torch.maximum`). Exact, like [`bf16_min`].
pub fn bf16_max(a: u16, b: u16) -> u16 {
    if bf16_to_f32(a) >= bf16_to_f32(b) { a } else { b }
}

/// Clamp a BF16 value into `[-hi, hi]` (`hi >= 0`), composed of exact min/max
/// comparisons — mirrors the reference's `clamp(x, -hi, hi)`.
pub fn bf16_clamp_sym(x: u16, hi: u16) -> u16 {
    bf16_max(bf16_min(x, hi), hi ^ 0x8000)
}

/// Fused multiply-add `a*b + c` with a SINGLE rounding to BF16 — the hardware
/// FMA contract (CUDA `__hfma` on `__nv_bfloat16`, PTX `fma.rn.bf16`), a bitwise
/// port of the reference's `ComputeOps.fma`.
///
/// BF16 operands make `a*b` exact in f64 (two 8-bit significands need 16 <= 53
/// bits). A TwoSum recovers the exact residual of `a*b + c`, from which the exact
/// real sum is rounded to f32 with round-to-ODD; the final f32 -> BF16 RNE cast
/// then rounds correctly (round-to-odd defuses the double rounding). Errors on a
/// non-finite result.
pub fn bf16_fma(a: u16, b: u16, c: u16) -> Result<u16> {
    let a64 = bf16_to_f32(a) as f64;
    let b64 = bf16_to_f32(b) as f64;
    let c64 = bf16_to_f32(c) as f64;
    let p = a64 * b64; // exact
    let s = p + c64; // f64 RNE approximation of the exact sum x = a*b + c
    let t = s - c64;
    let r = (p - t) + (c64 - (s - t)); // TwoSum: x - s, exact
    let s32 = s as f32; // brackets x: |s32 - x| < 0.51 ulp
    let err = (s - s32 as f64) + r; // sign(x - s32); nonzero iff x != s32
    // Round-to-odd fixup: if x is not representable and s32's significand is
    // even, the bracketing odd neighbour is the next f32 toward x.
    let fix = err != 0.0 && (s32.to_bits() & 1 == 0) && s32.is_finite();
    let rounded = if fix {
        if err > 0.0 { s32.next_up() } else { s32.next_down() }
    } else {
        s32
    };
    f32_to_bf16(rounded)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-checked against the reference `ComputeOps.fma` (torch, f64 path):
    /// each `(a, b, c)` BF16 triple maps to the listed result.
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
