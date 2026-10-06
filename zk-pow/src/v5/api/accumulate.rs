//! Bit-exact model of the NVIDIA A100 (GA100, `sm_80`) `HMMA.16816.F32`
//! accumulation for FP16 operands with FP32 accumulation.
//!
//! This mirrors the reference emulator validated bit-for-bit against A100
//! silicon (see the FP16 scheme whitepaper, Appendix "Empirical validation").
//! The model:
//!
//! * The `k` axis is split into groups of `G = 8` consecutive products,
//!   processed in order. (Instruction boundaries are invisible: `m16n8k8` and
//!   `m16n8k16` chains agree bit-for-bit, so only the group size matters.)
//! * Each product `a_u * b_u` is exact, with stored exponent
//!   `e_u = eps(a_u) + eps(b_u)`.
//! * One group step, with incoming FP32 accumulator `c`:
//!   - `eta = max(e_u over nonzero products, eps(c))`, where `c = 0` does not
//!     participate (a subnormal `c` contributes `-126`); an all-zero group with
//!     unchanged `c` is a no-op.
//!   - every product and `c` are truncated toward zero onto the grid
//!     `2^(eta - W)` with `W = 24`;
//!   - the integers are summed exactly;
//!   - the sum is rounded toward zero to FP32 (24 significant bits, subnormals
//!     on the `2^-149` grid).
//! * A zero result is `+0`; `|result| >= 2^128` is overflow. The verifier
//!   rejects non-finite operands and intermediates, so overflow aborts.

use super::dtype::decompose_fp16;

/// Internal accumulator precision (FP32 significand bits).
const W: i32 = 24;
/// Products per hardware accumulation group.
pub const GROUP: usize = 8;
/// Sentinel "no exponent" (empty group / zero term).
const NEG: i32 = i32::MIN / 2;
/// FP32 minimum nonzero (subnormal) exponent.
const FP32_MIN_EXP: i32 = -149;

/// Per-group policy census, recorded during the verifier's replay. See the
/// whitepaper "unpredictable accumulation steps" check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PolicyStep {
    /// The step did real work (not an empty no-op group).
    pub nonempty: bool,
    /// Breakpoint: the accumulator alignment or the FP32 rounding discarded a
    /// nonzero bit.
    pub breakpoint: bool,
    /// Number of products in the group whose truncation discarded nonzero bits.
    pub products_truncated: u32,
}

/// Decomposes an FP32 accumulator into `(sign, significand, el, ulp)` with
/// `value = sign * significand * 2^ulp` and `el` the stored exponent used for
/// the alignment max (clamped to `-126` for subnormals). Returns `el = NEG`
/// for a zero accumulator (it does not participate in the alignment).
fn acc_parts(c: f32) -> (i64, u64, i32, i32) {
    debug_assert!(c.is_finite());
    if c == 0.0 {
        return (1, 0, NEG, 0);
    }
    let sign: i64 = if c.is_sign_negative() { -1 } else { 1 };
    let bits = c.to_bits();
    let exp_field = ((bits >> 23) & 0xFF) as i32;
    let man = (bits & 0x7F_FFFF) as u64;
    if exp_field > 0 {
        // Normal: 24-bit significand (implicit 1), value = m * 2^(el - 23).
        let el = exp_field - 127;
        (sign, 0x80_0000 | man, el, el - 23)
    } else {
        // Subnormal: value = man * 2^-149, stored exponent clamped to -126.
        (sign, man, -126, FP32_MIN_EXP)
    }
}

/// `x` shifted by `s` (left if positive, truncating-right if negative).
#[inline]
fn shift_i128(x: i128, s: i32) -> i128 {
    if s >= 0 {
        x << s
    } else if -s >= 127 {
        0
    } else {
        x >> (-s)
    }
}

/// Rounds the integer `s * 2^unit` toward zero to an FP32 value (24 significant
/// bits, subnormals floored onto the `2^-149` grid). Mirrors `rz` in the
/// reference emulator.
fn rz_to_f32(s: i128, unit: i32) -> f32 {
    if s == 0 {
        return 0.0;
    }
    let sign = s.signum() as f64;
    let a = s.unsigned_abs();
    let nb = 127 - a.leading_zeros() as i32; // floor(log2|s|)
    let keep = (nb + unit - 23).max(FP32_MIN_EXP);
    let drop = (keep - unit).clamp(0, 127);
    let truncated = (a >> drop) << drop;
    let val = sign * (truncated as f64) * 2f64.powi(unit);
    val as f32
}

/// Computes the A100 device dot product of FP16 rows `a` and `b` (bit patterns),
/// with FP32 carry-in `c`. When `census` is `Some`, appends one [`PolicyStep`]
/// per group. Panics on NaN/inf operands; aborts (via the final finiteness
/// check by the caller) on overflow.
pub fn a100_dot(a: &[u16], b: &[u16], c: f32, mut census: Option<&mut Vec<PolicyStep>>) -> f32 {
    assert_eq!(a.len(), b.len(), "operand length mismatch");
    let mut cur = c;
    let k = a.len();
    let mut g0 = 0;
    while g0 < k {
        let g1 = (g0 + GROUP).min(k);
        let (cs, cm, cel, culp) = acc_parts(cur);

        // Alignment exponent over nonzero products and the accumulator.
        let mut eta = cel;
        for u in g0..g1 {
            let (sa, ma, ea) = decompose_fp16(a[u]);
            let (sb, mb, eb) = decompose_fp16(b[u]);
            if ma != 0 && mb != 0 {
                let _ = (sa, sb);
                eta = eta.max(ea + eb);
            }
        }
        if eta == NEG {
            // Empty group, accumulator zero: no-op.
            if let Some(v) = census.as_deref_mut() {
                v.push(PolicyStep::default());
            }
            g0 = g1;
            continue;
        }
        let unit = eta - W;

        // Sum products and the accumulator, all truncated onto the 2^unit grid.
        let mut sum: i128 = 0;
        let mut products_truncated = 0u32;
        for u in g0..g1 {
            let (sa, ma, ea) = decompose_fp16(a[u]);
            let (sb, mb, eb) = decompose_fp16(b[u]);
            if ma == 0 || mb == 0 {
                continue;
            }
            let prod = (ma * mb) as i128; // < 2^22, exact
            let sh = (ea + eb) - 20 - unit; // product LSB is 2^(ea+eb-20)
            let aligned = shift_i128(prod, sh);
            if sh < 0 && (prod & ((1i128 << (-sh).min(126)) - 1)) != 0 {
                products_truncated += 1;
            }
            sum += (sa * sb) as i128 * aligned;
        }
        // Accumulator term.
        let csh = culp - unit;
        let acc_aligned = shift_i128(cm as i128, csh);
        let acc_truncated = csh < 0 && cm != 0 && (cm & ((1u64 << (-csh).min(63)) - 1)) != 0;
        sum += cs as i128 * acc_aligned;

        let new = rz_to_f32(sum, unit);

        if let Some(v) = census.as_deref_mut() {
            // Breakpoint: accumulator alignment or FP32 rounding dropped a
            // nonzero bit. (`rz` dropped bits iff it differs from the exact sum.)
            let rz_dropped = (new as f64) != (sum as f64) * 2f64.powi(unit);
            v.push(PolicyStep {
                nonempty: true,
                breakpoint: acc_truncated || rz_dropped,
                products_truncated,
            });
        }
        cur = new;
        g0 = g1;
    }
    cur
}

/// Device matmul of an `m x k` operand `a` against an `n x k` operand `b` (both
/// row-major FP16 bit patterns; `b` is the transposed logical operand), with
/// optional `m x n` carry-in. Returns the `m x n` FP32 tile. This is the
/// datapath the verifier replays and the ticket hashes.
pub fn a100_matmul(
    a: &[u16],
    b: &[u16],
    acc: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let c = acc.map_or(0.0, |a| a[i * n + j]);
            out[i * n + j] = a100_dot(&a[i * k..i * k + k], &b[j * k..j * k + k], c, None);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v5::api::dtype::fp16_to_f32;

    // Reference vectors generated from the GPU-validated emulator (emu3.py).
    // Each line: k, then k a-bits, k b-bits, c-bits (f32), d-bits (f32).
    const VECTORS: &str = include_str!("testdata/a100_dot_vectors.txt");

    // Hardware-captured edge-case corpus: run on real sm_80 silicon (CMP 170HX /
    // A100) via `mma.sync.m16n8k16.f32.f16` and cross-checked against this model,
    // by `docs/fp16_scheme/validation/generate_and_capture.py`. Extends the corpus
    // above (which tops out at k=256) with k up to 1024 and explicit
    // subnormal-operand / subnormal-accumulator / subnormal-output / cancellation /
    // grid-boundary / max-magnitude edges. Same line format.
    const VECTORS_K1024: &str = include_str!("testdata/a100_dot_vectors_k1024.txt");

    fn check_vectors(data: &str, min_count: usize) -> usize {
        let mut n = 0;
        for line in data.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let t: Vec<&str> = line.split_whitespace().collect();
            let k: usize = t[0].parse().unwrap();
            let a: Vec<u16> = t[1..1 + k].iter().map(|x| x.parse().unwrap()).collect();
            let b: Vec<u16> = t[1 + k..1 + 2 * k].iter().map(|x| x.parse().unwrap()).collect();
            let c = f32::from_bits(t[1 + 2 * k].parse().unwrap());
            let d = f32::from_bits(t[2 + 2 * k].parse().unwrap());
            let got = a100_dot(&a, &b, c, None);
            assert_eq!(
                got.to_bits(),
                d.to_bits(),
                "vector {n} (k={k}): got {got} ({:#010x}) expected {d} ({:#010x})",
                got.to_bits(),
                d.to_bits()
            );
            n += 1;
        }
        assert!(n >= min_count, "expected >= {min_count} reference vectors, got {n}");
        n
    }

    #[test]
    fn matches_reference_emulator() {
        check_vectors(VECTORS, 100);
    }

    /// The model reproduces the hardware-captured edge corpus bit-for-bit,
    /// including k=1024 and the subnormal/cancellation/grid-boundary edges the
    /// original k<=256 corpus did not cover. This is the model-vs-silicon
    /// cross-check (the device bits were captured on sm_80, not model-ported).
    #[test]
    fn matches_hardware_capture_k1024() {
        let n = check_vectors(VECTORS_K1024, 200);
        // The corpus includes k=1024 vectors (the study's dimension, absent above).
        let has_k1024 = VECTORS_K1024
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .any(|l| l.split_whitespace().next() == Some("1024"));
        assert!(has_k1024, "capture corpus must include k=1024 vectors");
        // And at least one subnormal-FP32 output (exp field 0, nonzero mantissa).
        let has_subnormal_out = VECTORS_K1024
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .any(|l| {
                let t: Vec<&str> = l.split_whitespace().collect();
                let d: u32 = t[t.len() - 1].parse().unwrap();
                (d >> 23) & 0xFF == 0 && (d & 0x7F_FFFF) != 0
            });
        assert!(has_subnormal_out, "capture corpus must exercise the subnormal-output branch");
        assert!(n >= 200, "expected the full captured corpus");
    }

    #[test]
    fn single_product_exact() {
        // 3 * 5 = 15, no accumulation effects.
        let a = [f32_to_bits(3.0)];
        let b = [f32_to_bits(5.0)];
        assert_eq!(a100_dot(&a, &b, 0.0, None), 15.0);
    }

    #[test]
    fn zero_result_is_positive_zero() {
        let a = [f32_to_bits(1.0)];
        let b = [f32_to_bits(-1.0)];
        let d = a100_dot(&a, &b, 1.0, None);
        assert_eq!(d.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn breakpoint_census_detects_rz_drop() {
        // Accumulator 2^24 plus a unit product: the sum 2^24 + 1 needs 25 bits,
        // so the round-toward-zero to FP32 drops the low bit -> a breakpoint.
        let a = [f32_to_bits(1.0)];
        let b = [f32_to_bits(1.0)];
        let mut census = Vec::new();
        let d = a100_dot(&a, &b, 16_777_216.0, Some(&mut census));
        assert_eq!(d, 16_777_216.0); // 2^24 + 1 truncates back to 2^24
        assert_eq!(census.len(), 1);
        assert!(census[0].breakpoint, "RZ dropped a nonzero bit");
    }

    fn f32_to_bits(x: f32) -> u16 {
        let b = super::super::dtype::f32_to_fp16(x).unwrap();
        // sanity: decodes back to a representable neighbor
        let _ = fp16_to_f32(b);
        b
    }
}
