//! Proves the FP16 noise-line **normalization** arithmetic of the plaintext ground truth
//! [`crate::v5::api::noise`]'s `normalize_line`: given the `rank` raw XOF bytes of one line
//! (witness inputs — their binding to the keyed-BLAKE3 XOF is a separate later stage, to be
//! CTL-bound to the Blake3 table through [`super::ctl`]'s byte column), it proves
//!
//! ```text
//! x_i      = (1 - 2*sign(b_i)) * ((b_i & 0x7F) + 1)          // signed decode, |x_i| in [1,128]
//! S        = sum_i x_i^2
//! q        = isqrt(S * INT_SQRT_PREC^2)                       // INT_SQRT_PREC = 32
//! denom    = f32_to_bf16(q)
//! numer    = f32_to_bf16(NOISE_TARGET_NORM * 32) = 8192       // compile-time constant (2^13)
//! scale    = bf16_div(numer, denom)
//! entry_i  = f32_to_fp16(bf16_to_f32(bf16_mul(f32_to_bf16(x_i), scale)))
//! ```
//!
//! # Constraint groups
//!
//! * **B** (decode): `SIGN_BIT` boolean, `BYTE = 128*SIGN_BIT + MAG_MINUS_1`,
//!   `MAGNITUDE = (1 - IS_PAD)*(MAG_MINUS_1 + 1)`.
//! * **S** (sum of squares + isqrt): the prefix sum of `MAGNITUDE^2` chained into
//!   `TOTAL_SUMSQ`, and the Euclidean square-root witness `q^2 + REM = 1024*S`,
//!   `REM + S2 = 2q` (with `REM, S2 >= 0` delegated to the RC16 inventory), which pins
//!   `q = isqrt(1024*S)`.
//! * **D** (`denom`): the boundary products `(4*M_DEN -/+ 2) * 2^gd` of the ties-to-even
//!   round of the integer `q` to a normal BF16. The bracket inequalities themselves
//!   (`B_LO*2^gd + odd <= 4q <= B_HI*2^gd - odd`) are the RC16 range facts on the two slacks,
//!   and `2^gd` is a POW2D value — all declared in [`super::ctl`].
//! * **V** (`scale`): the division bracket. Because `numer = 8192 = 2^13` is a constant power
//!   of two, `scale = RNE(2^13 / denom)` reduces to the single power-of-two comparison
//!   `(4*M_S - 2)*M_DEN + odd <= 2^A <= (4*M_S + 2)*M_DEN - odd` with `A = 283 - DENOM_EXP
//!   - SCALE_EXP` (`283 = 2 + 13 + 2*134`). The AIR materializes the boundary products; the
//!   inequalities are RC16 slacks and `2^A` is a POW2D value.
//! * **E** (per-entry multiply + FP16 cast): the exact product significand
//!   `P = MAGNITUDE * M_S`, the ties-to-even round `(4*M_E -/+ 2)*2^gm` bracketing `4P`, and
//!   the exact BF16 -> FP16 widening `entry = SIGN<<15 | (gm + SCALE_EXP - 112)<<10 |
//!   8*ENTRY_MANT`.
//!
//! The line-level groups S/D/V are replicated identically on every row (`TOTAL_SUMSQ` is held
//! constant by a transition constraint and pinned to the final prefix sum on the last row), so
//! every row re-proves the shared scale; group E is per-row and gated off on the padding rows.
//!
//! # Soundness envelope (documented, not proved here)
//!
//! * `norm_scaled >= 128` (equivalently the line has `rank >= 16`, so `1024*S >= 2^14`): the
//!   `denom` round then has a nonnegative shift `gd = DENOM_EXP - 134 >= 0`, inside POW2D's
//!   `[0, 19]` key domain. Every protocol line satisfies this (the peel rank is at least 16).
//! * Each `entry_i` is a **normal** FP16 value (biased exponent in `[1, 30]`), so the final
//!   `bf16_to_f32 -> f32_to_fp16` cast is an exact widening (FP16's 10 mantissa bits subsume
//!   BF16's 7). This holds for every in-scheme line (entry magnitudes stay in `[2^-7, 2^8]`).
//!   The AIR enforces the exactness and the `[1, 30]` range; a line violating the envelope is
//!   simply unprovable (and never traced, exactly as the sqrt/scale brackets of
//!   [`crate::v4::circuit::scale_stark`] treat their own rejected boundaries).
//! All three RNE brackets carry the full ties-to-even machinery — the mantissa parity split
//! *and* the binade-bottom (`IS_BOTTOM`) correction `+ bottom` on the lower boundary (mirroring
//! `circuit::fp8::scale_stark`'s sqrt bracket) — so each rounded mantissa is pinned to the
//! **unique** RNE result with no residual grinding freedom, including at power-of-two roundings.
//! Because every bracket's effective exponent is structurally `>= 2` (its POW2D / FP16-exp key
//! domain), the `IS_BOTTOM` flag is just `[mantissa == 0]` (an is-zero gadget), with no separate
//! `EXP >= 2` witness.

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

use super::columns::{NUM_NOISE_COLUMNS, NUM_NOISE_PUBLIC_INPUTS, NoiseColumnsView};
use crate::v5::api::dtype::f32_to_fp16;
use crate::v4::api::compute::{bf16_div, bf16_mul};
use crate::v4::api::dtype::{bf16_to_f32, f32_to_bf16};
use crate::v4::api::quantization::NOISE_TARGET_NORM;
use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// The fixed-point precision factor carried through the integer isqrt (`INT_SQRT_PREC = 32`,
/// matching `crate::v5::api::noise`). Its square `1024` scales the sum of squares.
const INT_SQRT_PREC_SQ: u64 = 32 * 32;

/// BF16 unit offset: a finite BF16 is `M * 2^(E* - 134)` with `M = 128 + mantissa` for normals.
const BF16_EXP_UNIT: u64 = 134;

/// The division-bracket key base `A = DIV_KEY_BASE - DENOM_EXP - SCALE_EXP`, where
/// `DIV_KEY_BASE = 2 + log2(numer) + 2*134 = 2 + 13 + 268 = 283`. Valid only because
/// `numer = 8192 = 2^13` is a power of two (asserted in [`NoiseProgram::new`]).
const DIV_KEY_BASE: u64 = 2 + 13 + 2 * BF16_EXP_UNIT;

/// FP16 cast offset: the FP16 biased exponent of a normal entry is `E*(entry_bf16) - 112`
/// (`112 = 134 - 15 - 7`: BF16 unit, FP16 bias, FP16-minus-BF16 mantissa width).
const FP16_EXP_OFFSET: u64 = 112;

/// Field inverse of a small integer, or zero when it is zero (the is-zero gadget's witness).
fn inv_f<F: RichField>(x: u64) -> F {
    if x == 0 {
        F::ZERO
    } else {
        F::from_canonical_u64(x).inverse()
    }
}

/// The compile-time BF16 code of `numer = f32_to_bf16(NOISE_TARGET_NORM * INT_SQRT_PREC)`.
fn numer_code() -> u16 {
    f32_to_bf16((NOISE_TARGET_NORM * 32.0) as f32).expect("8192 is representable in bf16")
}

/// The committed noise geometry: the per-line `rank` (entries per line), the per-line reuse
/// multiplicities (one per line; their length is the line count), and the trace height (a power of
/// two covering `num_lines * rank`). The normalization identity is otherwise program-independent;
/// the multi-line structure (blocks laid end to end, mirroring
/// [`crate::v5::circuit::row_scale_stark`]) lets one NoiseStark instance prove every noise line of
/// a tile — `E_A`, `F_A`, `E_B`, `F_B`.
#[derive(Clone, Debug)]
pub struct NoiseProgram {
    /// Number of entries (raw XOF bytes) in each line (the noise rank `r`).
    pub rank: usize,
    /// Per-line noise-matmul reuse multiplicity (the `OPERAND_MULT` known column), one per line. Its
    /// length is the line count. A standalone single-line program uses `[0]` (no CTL).
    pub mults: Vec<u64>,
    /// Trace height (a power of two `>= num_lines * rank`).
    pub num_rows: usize,
}

impl NoiseProgram {
    /// Builds a single-line program padded to the next power of two (standalone / unit-test path).
    /// The batch uses [`Self::with_lines`] to pin an on-ladder height and the real line layout.
    pub fn new(rank: usize) -> Self {
        Self::with_lines(rank, vec![0], rank.next_power_of_two().max(2))
    }

    /// Builds a multi-line program: `mults.len()` lines of `rank` entries each, at an explicit
    /// `num_rows` height. Asserts the compile-time facts the AIR's constants bake in (`numer` is
    /// `8192 = 2^13`, a power of two; `rank >= 1`).
    pub fn with_lines(rank: usize, mults: Vec<u64>, num_rows: usize) -> Self {
        assert!(rank >= 1, "a noise line has at least one entry");
        assert!(!mults.is_empty(), "at least one line");
        assert!(num_rows.is_power_of_two(), "trace height must be a power of two");
        assert!(num_rows >= mults.len() * rank, "trace height must cover every live block");
        let numer = numer_code();
        assert_eq!(numer, 0x4600, "numer is f32_to_bf16(8192)");
        assert_eq!(numer & 0x7F, 0, "numer must be a power of two for the division bracket");
        assert_eq!(u64::from(numer >> 7), 13 + 127, "numer exponent field is 140 (value 2^13)");
        Self { rank, mults, num_rows }
    }

    /// Number of noise lines this program proves.
    pub fn num_lines(&self) -> usize {
        self.mults.len()
    }

    /// Live rows: one per entry of every line.
    pub fn live_rows(&self) -> usize {
        self.num_lines() * self.rank
    }

    /// Trace height.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The live-row -> `(is_pad, line, is_block_start, is_block_final)` schedule (a pure function of
    /// the geometry; the trailing padding forms one block starting at the first padding row).
    fn block_schedule(&self, i: usize) -> (bool, usize, bool, bool) {
        let live = self.live_rows();
        let is_pad = i >= live;
        let line = if is_pad { self.num_lines() - 1 } else { i / self.rank };
        let is_block_start = (!is_pad && i % self.rank == 0) || (is_pad && i == live);
        let is_block_final = !is_pad && i % self.rank == self.rank - 1;
        (is_pad, line, is_block_start, is_block_final)
    }

    /// The class (a) ("known") columns (`IS_PAD`, `IS_BLOCK_START`, `IS_BLOCK_FINAL`,
    /// `GLOBAL_INDEX`, `OPERAND_MULT`), pure functions of the geometry; bit-exact with
    /// [`Self::generate_trace`]'s fill.
    pub fn known_values<F: RichField>(&self) -> Vec<PolynomialValues<F>> {
        let num_rows = self.num_rows();
        let live = self.live_rows();
        let rank = self.rank;
        let (mut is_pad, mut bstart, mut bfinal, mut gidx, mut omult) = (
            Vec::with_capacity(num_rows),
            Vec::with_capacity(num_rows),
            Vec::with_capacity(num_rows),
            Vec::with_capacity(num_rows),
            Vec::with_capacity(num_rows),
        );
        // ZK noise binding (6e-3c): the egress-pair filter/key. A pair opens on each even live entry
        // whose successor is still in the line (`e+1 < rank`); its key is `GLOBAL_INDEX / 2 =
        // line*16 + e/2`, matching the noise-BLAKE3 line compression's egress base `line*16`.
        let (mut is_egress_pair, mut egress_key) = (Vec::with_capacity(num_rows), Vec::with_capacity(num_rows));
        for i in 0..num_rows {
            let (p, line, bs, bf) = self.block_schedule(i);
            is_pad.push(F::from_bool(p));
            bstart.push(F::from_bool(bs));
            bfinal.push(F::from_bool(bf));
            gidx.push(F::from_canonical_usize(if p { live - 1 } else { i }));
            omult.push(F::from_canonical_u64(self.mults[line]));
            let e = i % rank;
            let pair = !p && e % 2 == 0 && e + 1 < rank;
            is_egress_pair.push(F::from_bool(pair));
            egress_key.push(if pair { F::from_canonical_usize(i / 2) } else { F::ZERO });
        }
        vec![
            PolynomialValues::new(is_pad),
            PolynomialValues::new(bstart),
            PolynomialValues::new(bfinal),
            PolynomialValues::new(gidx),
            PolynomialValues::new(omult),
            PolynomialValues::new(is_egress_pair),
            PolynomialValues::new(egress_key),
        ]
    }

    /// Generates the trace from `num_lines * rank` raw XOF bytes (the lines concatenated in order),
    /// bit-exact with `crate::v5::api::noise`'s `normalize_line` applied to each line.
    pub fn generate_trace<F: RichField>(&self, bytes: &[u8]) -> Vec<[F; NUM_NOISE_COLUMNS]> {
        assert_eq!(bytes.len(), self.live_rows(), "one byte per live line entry");
        let num_rows = self.num_rows();
        let live = self.live_rows();
        let rank = self.rank;

        let mut rows: Vec<[F; NUM_NOISE_COLUMNS]> = Vec::with_capacity(num_rows);
        rows.resize(num_rows, [F::ZERO; NUM_NOISE_COLUMNS]);

        for line in 0..self.num_lines() {
            let line_bytes = &bytes[line * rank..(line + 1) * rank];
            let (scalars, scale_code, scale_exp, scale_mant) = line_scalars::<F>(line_bytes);
            let mut running: u64 = 0;
            for e in 0..rank {
                let i = line * rank + e;
                let mut v = scalars;
                let (_, _, bs, bf) = self.block_schedule(i);
                v.is_pad = F::ZERO;
                v.is_block_start = F::from_bool(bs);
                v.is_block_final = F::from_bool(bf);
                v.global_index = F::from_canonical_usize(i);
                v.operand_mult = F::from_canonical_u64(self.mults[line]);
                // ZK noise binding (6e-3c): open a two-byte XOF limb on even entries (successor in line).
                if e % 2 == 0 && e + 1 < rank {
                    v.is_egress_pair = F::ONE;
                    v.egress_key = F::from_canonical_usize(i / 2);
                    v.byte_pair = F::from_canonical_u64(u64::from(line_bytes[e]) + 256 * u64::from(line_bytes[e + 1]));
                }
                fill_entry::<F>(&mut v, line_bytes[e], scale_code, scale_exp, scale_mant);
                let mag = u64::from(line_bytes[e] & 0x7F) + 1;
                running += mag * mag;
                v.running_sumsq = F::from_canonical_u64(running);
                rows[i] = v.into();
            }
        }

        // ---- Trailing padding: one block of all-zero rows, IS_PAD = 1. ----
        for i in live..num_rows {
            let (_, _, bs, _) = self.block_schedule(i);
            let mut v = NoiseColumnsView::<F>::default();
            v.is_pad = F::ONE;
            v.is_block_start = F::from_bool(bs);
            v.global_index = F::from_canonical_usize(live - 1);
            v.operand_mult = F::from_canonical_u64(*self.mults.last().unwrap());
            rows[i] = v.into();
        }
        rows
    }
}

/// The line-level scalars (sum of squares, integer isqrt, and the BF16 scale derivation) of one
/// `rank`-byte noise line, as a partially-filled view with every scalar column set (the per-entry
/// decode/multiply columns are filled per row by [`fill_entry`]). Bit-exact with `normalize_line`.
fn line_scalars<F: RichField>(bytes: &[u8]) -> (NoiseColumnsView<F>, u16, u64, u64) {
    let mut total: u64 = 0;
    for &b in bytes {
        let mag = u64::from(b & 0x7F) + 1;
        total += mag * mag;
    }
    let x = total * INT_SQRT_PREC_SQ;
    let q = x.isqrt();
    assert!(q >= 128, "envelope: norm_scaled >= 128 (rank >= 16); line out of provable scope");
    let rem = x - q * q;
    let s2 = 2 * q - rem;
    assert!(q < (1 << 21) && total < (1 << 32), "line magnitude out of range");

    let denom_code = f32_to_bf16(q as f32).expect("q < 2^24 is representable in bf16");
    let scale_code = bf16_div(numer_code(), denom_code).expect("noise-line scale is finite");

    let denom_exp = u64::from(denom_code >> 7);
    let denom_mant = u64::from(denom_code & 0x7F);
    let gd = denom_exp as i64 - BF16_EXP_UNIT as i64;
    assert!((0..=19).contains(&gd), "denom shift gd out of POW2D domain");
    let denom_pow = 1u64 << gd;
    let m_den = 128 + denom_mant;
    let denom_bottom = u64::from(denom_mant == 0);
    let denom_blo = (4 * m_den - 2 + denom_bottom) * denom_pow;
    let denom_bhi = (4 * m_den + 2) * denom_pow;
    let denom_parity = denom_mant & 1;
    let denom_half = (denom_mant - denom_parity) / 2;
    let denom_sl = 4 * q - denom_blo - denom_parity;
    let denom_su = denom_bhi - (4 * q) - denom_parity;

    let scale_exp = u64::from(scale_code >> 7);
    let scale_mant = u64::from(scale_code & 0x7F);
    let a = DIV_KEY_BASE as i64 - denom_exp as i64 - scale_exp as i64;
    assert!((0..=19).contains(&a), "division shift A out of POW2D domain");
    let div_pow = 1u64 << a;
    let m_s = 128 + scale_mant;
    let scale_bottom = u64::from(scale_mant == 0);
    let div_blo = (4 * m_s - 2 + scale_bottom) * m_den;
    let div_bhi = (4 * m_s + 2) * m_den;
    let scale_parity = scale_mant & 1;
    let scale_half = (scale_mant - scale_parity) / 2;
    let div_sl = div_pow - div_blo - scale_parity;
    let div_su = div_bhi - div_pow - scale_parity;

    let mut v = NoiseColumnsView::<F>::default();
    v.total_sumsq = F::from_canonical_u64(total);
    v.total_hi = F::from_canonical_u64(total >> 16);
    v.norm_scaled = F::from_canonical_u64(q);
    v.norm_scaled_hi = F::from_canonical_u64(q >> 16);
    v.isqrt_rem = F::from_canonical_u64(rem);
    v.isqrt_rem_hi = F::from_canonical_u64(rem >> 16);
    v.isqrt_s2 = F::from_canonical_u64(s2);
    v.isqrt_s2_hi = F::from_canonical_u64(s2 >> 16);

    v.denom_exp = F::from_canonical_u64(denom_exp);
    v.denom_mant = F::from_canonical_u64(denom_mant);
    v.denom_pow = F::from_canonical_u64(denom_pow);
    v.denom_parity = F::from_canonical_u64(denom_parity);
    v.denom_half = F::from_canonical_u64(denom_half);
    v.denom_bottom = F::from_canonical_u64(denom_bottom);
    v.denom_mant_inv = inv_f::<F>(denom_mant);
    v.denom_blo = F::from_canonical_u64(denom_blo);
    v.denom_bhi = F::from_canonical_u64(denom_bhi);
    v.denom_sl_hi = F::from_canonical_u64(denom_sl >> 16);
    v.denom_su_hi = F::from_canonical_u64(denom_su >> 16);

    v.scale_exp = F::from_canonical_u64(scale_exp);
    v.scale_mant = F::from_canonical_u64(scale_mant);
    v.div_pow = F::from_canonical_u64(div_pow);
    v.scale_parity = F::from_canonical_u64(scale_parity);
    v.scale_half = F::from_canonical_u64(scale_half);
    v.scale_bottom = F::from_canonical_u64(scale_bottom);
    v.scale_mant_inv = inv_f::<F>(scale_mant);
    v.div_blo = F::from_canonical_u64(div_blo);
    v.div_bhi = F::from_canonical_u64(div_bhi);
    v.div_sl_hi = F::from_canonical_u64(div_sl >> 16);
    v.div_su_hi = F::from_canonical_u64(div_su >> 16);
    (v, scale_code, scale_exp, scale_mant)
}

/// Fills the per-entry decode / multiply / FP16-cast columns of one live row from its raw byte and
/// the (already-derived) line scale. Bit-exact with `normalize_line`'s per-entry arithmetic.
fn fill_entry<F: RichField>(v: &mut NoiseColumnsView<F>, b: u8, scale_code: u16, scale_exp: u64, scale_mant: u64) {
    let sign_bit = u64::from(b >> 7);
    let mag_minus_1 = u64::from(b & 0x7F);
    let mag = mag_minus_1 + 1;
    v.byte = F::from_canonical_u64(u64::from(b));
    v.sign_bit = F::from_canonical_u64(sign_bit);
    v.mag_minus_1 = F::from_canonical_u64(mag_minus_1);
    v.magnitude = F::from_canonical_u64(mag);

    let m_s = 128 + scale_mant;
    let x_i = (1 - 2 * sign_bit as i64) * mag as i64;
    let xb = f32_to_bf16(x_i as f32).expect("|x_i| <= 128 is representable");
    let entry_bf16 = bf16_mul(xb, scale_code).expect("noise entry is finite");
    let entry_fp16 = f32_to_fp16(bf16_to_f32(entry_bf16)).expect("entry representable in FP16");

    let entry_mant = u64::from(entry_bf16 & 0x7F);
    let entry_exp_field = u64::from((entry_bf16 >> 7) & 0xFF);
    assert!((1..=254).contains(&entry_exp_field), "entry_bf16 must be normal");
    let m_e = 128 + entry_mant;
    let gm = entry_exp_field as i64 - scale_exp as i64;
    assert!((0..=19).contains(&gm), "entry shift gm out of POW2D domain");
    let entry_pow = 1u64 << gm;
    let prod = mag * m_s;
    let entry_bottom = u64::from(entry_mant == 0);
    let entry_blo = (4 * m_e - 2 + entry_bottom) * entry_pow;
    let entry_bhi = (4 * m_e + 2) * entry_pow;
    let entry_parity = entry_mant & 1;
    let entry_half = (entry_mant - entry_parity) / 2;
    let entry_sl = 4 * prod - entry_blo - entry_parity;
    let entry_su = entry_bhi - 4 * prod - entry_parity;

    let fp16_exp = u64::from((entry_fp16 >> 10) & 0x1F);
    assert_eq!(fp16_exp, entry_exp_field - FP16_EXP_OFFSET, "FP16 cast exponent relation (entry is FP16-normal)");
    assert_eq!(
        u64::from(entry_fp16),
        (sign_bit << 15) | (fp16_exp << 10) | (8 * entry_mant),
        "FP16 cast is an exact widening"
    );

    v.entry_mant = F::from_canonical_u64(entry_mant);
    v.entry_gm = F::from_canonical_u64(gm as u64);
    v.entry_pow = F::from_canonical_u64(entry_pow);
    v.entry_parity = F::from_canonical_u64(entry_parity);
    v.entry_half = F::from_canonical_u64(entry_half);
    v.entry_bottom = F::from_canonical_u64(entry_bottom);
    v.entry_mant_inv = inv_f::<F>(entry_mant);
    v.entry_prod = F::from_canonical_u64(prod);
    v.entry_blo = F::from_canonical_u64(entry_blo);
    v.entry_bhi = F::from_canonical_u64(entry_bhi);
    v.entry_sl_hi = F::from_canonical_u64(entry_sl >> 16);
    v.entry_su_hi = F::from_canonical_u64(entry_su >> 16);
    v.entry_fp16_exp = F::from_canonical_u64(fp16_exp);
    v.entry_fp16 = F::from_canonical_u64(u64::from(entry_fp16));
}

/// Evaluates the B/S/D/V/E constraint groups (module docs). All range facts (the RC16 limb
/// inventory, the bracket slacks, the POW2D shifts, the PAIR128 mantissa checks) are declared in
/// [`super::ctl`], not here.
pub(crate) fn eval_noise_constraints<V, S, E>(
    vars: &StarkFrame<V, S, NUM_NOISE_COLUMNS, NUM_NOISE_PUBLIC_INPUTS>,
    eval: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let lv_arr: &[V; NUM_NOISE_COLUMNS] = vars.get_local_values().try_into().unwrap();
    let lv: &NoiseColumnsView<V> = lv_arr.borrow();
    let nv_arr: &[V; NUM_NOISE_COLUMNS] = vars.get_next_values().try_into().unwrap();
    let nv: &NoiseColumnsView<V> = nv_arr.borrow();

    let one = eval.u64(1);
    let two = eval.u64(2);
    let c128 = eval.u64(128);
    let live = eval.sub(one, lv.is_pad);

    // ---- B: signed byte decode. ----
    eval.constraint_bool(lv.sign_bit);
    let s128 = eval.mul(c128, lv.sign_bit);
    let byte_expr = eval.add(s128, lv.mag_minus_1);
    eval.constraint_eq(lv.byte, byte_expr);
    let mag_p1 = eval.add(lv.mag_minus_1, one);
    let mag_expr = eval.mul(live, mag_p1);
    eval.constraint_eq(lv.magnitude, mag_expr);

    // ---- Multi-block structure (mirrors row_scale_stark): every noise line is a `rank`-row block
    // laid end to end, the final block being trailing all-zero padding. ----
    eval.constraint_bool(lv.is_block_start);
    eval.constraint_bool(lv.is_block_final);
    let not_bs_next = eval.sub(one, nv.is_block_start);

    // ---- S: prefix sum of MAGNITUDE^2 and the Euclidean isqrt, PER BLOCK. ----
    // Reset at each block start: running = x_0^2.
    let sq0 = eval.mul(lv.magnitude, lv.magnitude);
    let reset = eval.sub(lv.running_sumsq, sq0);
    let c = eval.mul(lv.is_block_start, reset);
    eval.constraint(c);
    // Within-block step (nv NOT a block start): running_{i+1} = running_i + x_{i+1}^2.
    let sqn = eval.mul(nv.magnitude, nv.magnitude);
    let step = eval.sub(nv.running_sumsq, lv.running_sumsq);
    let step = eval.sub(step, sqn);
    let c = eval.mul(not_bs_next, step);
    eval.constraint_transition(c);
    // On each LIVE block's final row, running = that block's TOTAL_SUMSQ.
    let last = eval.sub(lv.running_sumsq, lv.total_sumsq);
    let c = eval.mul(lv.is_block_final, last);
    eval.constraint(c);
    // Line scalars (TOTAL_SUMSQ .. DIV_SU_HI) are constant within a block (may jump at a block start).
    {
        let scalar_start = super::columns::NOISE_COL_MAP.total_sumsq;
        let scalar_end = super::columns::NOISE_COL_MAP.div_su_hi;
        for i in scalar_start..=scalar_end {
            let d = eval.sub(nv_arr[i], lv_arr[i]);
            let c = eval.mul(not_bs_next, d);
            eval.constraint_transition(c);
        }
    }
    // q^2 + REM = 1024 * S.
    let q2 = eval.mul(lv.norm_scaled, lv.norm_scaled);
    let lhs = eval.add(q2, lv.isqrt_rem);
    let sqprec = eval.u64(INT_SQRT_PREC_SQ);
    let x = eval.mul(sqprec, lv.total_sumsq);
    eval.constraint_eq(lhs, x);
    // REM + S2 = 2q.
    let rem_s2 = eval.add(lv.isqrt_rem, lv.isqrt_s2);
    let two_q = eval.mul(two, lv.norm_scaled);
    eval.constraint_eq(rem_s2, two_q);

    // ---- D: denom boundary products, parity split, and binade-bottom flag. The bracket uses
    // DENOM_POW as its factor (0 on pad rows -> products vanish), so it needs no live gate; the
    // is-zero gadget is gated off on pad rows (zero mantissa + zero inverse would otherwise fail). ----
    bracket_boundary_affine(eval, lv.denom_mant, lv.denom_bottom, lv.denom_pow, lv.denom_blo, lv.denom_bhi);
    parity_split(eval, lv.denom_mant, lv.denom_half, lv.denom_parity);
    is_zero_flag(eval, lv.denom_mant, lv.denom_mant_inv, lv.denom_bottom, live);

    // ---- V: division boundary products ((4*M_S -/+ 2 + bottom) * M_DEN), parity, bottom. The
    // factor M_DEN = 128 + DENOM_MANT is >= 128 even on pad rows, so the bracket equalities are live-
    // gated (DIV_BLO/DIV_BHI are 0 on pad rows). ----
    let m_den = eval.add(c128, lv.denom_mant);
    bracket_boundary_affine_gated(eval, lv.scale_mant, lv.scale_bottom, m_den, lv.div_blo, lv.div_bhi, live);
    parity_split(eval, lv.scale_mant, lv.scale_half, lv.scale_parity);
    is_zero_flag(eval, lv.scale_mant, lv.scale_mant_inv, lv.scale_bottom, live);

    // ---- E: per-entry exact product, multiply bracket, FP16 cast. ----
    // Exact product significand P = MAGNITUDE * M_S (0 on pad rows: MAGNITUDE = 0).
    let m_s = eval.add(c128, lv.scale_mant);
    let prod = eval.mul(lv.magnitude, m_s);
    eval.constraint_eq(lv.entry_prod, prod);
    bracket_boundary_affine(eval, lv.entry_mant, lv.entry_bottom, lv.entry_pow, lv.entry_blo, lv.entry_bhi);
    parity_split(eval, lv.entry_mant, lv.entry_half, lv.entry_parity);
    // The entry binade-bottom is-zero gadget is gated off on pad rows (zero mantissa there).
    is_zero_flag(eval, lv.entry_mant, lv.entry_mant_inv, lv.entry_bottom, live);
    // FP16 biased exponent = gm + SCALE_EXP - 112 (gated: pad rows carry 0).
    let fp_exp_sum = eval.add(lv.entry_gm, lv.scale_exp);
    let offset = eval.u64(FP16_EXP_OFFSET);
    let fp_exp = eval.sub(fp_exp_sum, offset);
    let fp_exp_diff = eval.sub(lv.entry_fp16_exp, fp_exp);
    let fp_exp_gated = eval.mul(live, fp_exp_diff);
    eval.constraint(fp_exp_gated);
    // FP16 code = SIGN<<15 | FP16_EXP<<10 | 8*ENTRY_MANT (pad rows: all zero -> 0).
    let c_sign = eval.u64(1 << 15);
    let c_exp = eval.u64(1 << 10);
    let c_eight = eval.u64(8);
    let sign_hi = eval.mul(c_sign, lv.sign_bit);
    let exp_hi = eval.mul(c_exp, lv.entry_fp16_exp);
    let mant8 = eval.mul(c_eight, lv.entry_mant);
    let fp16 = eval.add(sign_hi, exp_hi);
    let fp16 = eval.add(fp16, mant8);
    eval.constraint_eq(lv.entry_fp16, fp16);

    // ---- ZK noise binding (6e-3c): the egress-pair limb. On IS_EGRESS_PAIR rows (even live entries),
    // BYTE_PAIR = BYTE + 2^8 * BYTE' (this row's byte and the next row's byte), so the egress-pair
    // channel can export the little-endian 16-bit XOF limb against the noise-BLAKE3 cv_out limbs. Degree
    // 2 (flag * linear); off egress-pair rows the flag is 0 and BYTE_PAIR is free (unconstrained, and
    // unread by the channel). Transition form: IS_EGRESS_PAIR is 0 on the trailing padding, so the
    // last-row wrap never activates it. ----
    let c256 = eval.u64(256);
    let hi_byte = eval.mul(c256, nv.byte);
    let pair_expr = eval.add(lv.byte, hi_byte);
    let pair_diff = eval.sub(lv.byte_pair, pair_expr);
    let pair_c = eval.mul(lv.is_egress_pair, pair_diff);
    eval.constraint_transition(pair_c);
}

/// Materializes the ties-to-even quarter-ulp boundary products for a round whose scale factor is
/// a committed power of two `pow`: `blo = (4*(128 + mant) - 2 + bottom)*pow`, `bhi = (4*(128 +
/// mant) + 2)*pow`. (`4*(128 + mant) = 512 + 4*mant`, so `blo` uses `510 + 4*mant + bottom` and
/// `bhi` `514 + 4*mant`.) The `+ bottom` term is the binade-bottom (`IS_BOTTOM`) correction:
/// at a power of two (`mant = 0`) the predecessor sits one quarter ulp below, so the lower
/// acceptance boundary is `-ulp/4`, not `-ulp/2` — this is what pins each rounded mantissa to
/// the UNIQUE ties-to-even result (mirrors `circuit::fp8::scale_stark`'s sqrt bracket).
fn bracket_boundary_affine<V, S, E>(eval: &mut E, mant: V, bottom: V, pow: V, blo: V, bhi: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let c4 = eval.u64(4);
    let c510 = eval.u64(510);
    let c514 = eval.u64(514);
    let four_mant = eval.mul(c4, mant);
    let coef_lo_base = eval.add(c510, four_mant);
    let coef_lo = eval.add(coef_lo_base, bottom);
    let coef_hi = eval.add(c514, four_mant);
    let blo_e = eval.mul(coef_lo, pow);
    let bhi_e = eval.mul(coef_hi, pow);
    eval.constraint_eq(blo, blo_e);
    eval.constraint_eq(bhi, bhi_e);
}

/// Like [`bracket_boundary_affine`] but multiplies both boundary equalities by `gate` (used by the
/// division bracket, whose factor `M_DEN >= 128` stays nonzero on pad rows, so the boundary products
/// must be switched off there rather than vanishing through a zero factor).
fn bracket_boundary_affine_gated<V, S, E>(eval: &mut E, mant: V, bottom: V, factor: V, blo: V, bhi: V, gate: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let c4 = eval.u64(4);
    let c510 = eval.u64(510);
    let c514 = eval.u64(514);
    let four_mant = eval.mul(c4, mant);
    let coef_lo_base = eval.add(c510, four_mant);
    let coef_lo = eval.add(coef_lo_base, bottom);
    let coef_hi = eval.add(c514, four_mant);
    let blo_e = eval.mul(coef_lo, factor);
    let bhi_e = eval.mul(coef_hi, factor);
    let d_lo = eval.sub(blo, blo_e);
    let c = eval.mul(gate, d_lo);
    eval.constraint(c);
    let d_hi = eval.sub(bhi, bhi_e);
    let c = eval.mul(gate, d_hi);
    eval.constraint(c);
}

/// Pins `flag = [mant == 0]` (the binade-bottom flag): `flag` boolean, `flag*mant = 0` (so
/// `flag = 1 => mant = 0`), and `mant*mant_inv = 1 - flag` (so `mant != 0 => flag = 0`, and
/// `mant = 0 => flag = 1`). The effective exponent is `>= 2` throughout (POW2D/FP16-exp
/// domains), so no separate `EXP >= 2` witness is needed — unlike `scale_stark`, whose sqrt
/// claim may be subnormal. `gate` multiplies every constraint (pass `one` to leave it on, or the
/// live filter to switch it off on pad rows).
fn is_zero_flag<V, S, E>(eval: &mut E, mant: V, mant_inv: V, flag: V, gate: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let one = eval.u64(1);
    let f2 = eval.mul(flag, flag);
    let fb = eval.sub(f2, flag);
    let c = eval.mul(gate, fb);
    eval.constraint(c);
    let fm = eval.mul(flag, mant);
    let c = eval.mul(gate, fm);
    eval.constraint(c);
    let mi = eval.mul(mant, mant_inv);
    let omf = eval.sub(one, flag);
    let d = eval.sub(mi, omf);
    let c = eval.mul(gate, d);
    eval.constraint(c);
}

/// `mant = 2*half + parity` with `parity` boolean (the ties-to-even parity split; `half` is
/// RC16'd in `ctl.rs` so the split is two-sided).
fn parity_split<V, S, E>(eval: &mut E, mant: V, half: V, parity: V)
where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    eval.constraint_bool(parity);
    let c2 = eval.u64(2);
    let two_half = eval.mul(c2, half);
    let recomposed = eval.add(two_half, parity);
    eval.constraint_eq(mant, recomposed);
}

/// FP16 NoiseStark. A CTL party of the FP16 batch (`requires_ctls()`): the byte column is
/// CTL-bound to the keyed-BLAKE3 XOF table and the range/shift facts to the committed LUTs
/// (declared in [`super::ctl`]), so the batch driver is the only supported proving path.
#[derive(Clone, Debug)]
pub struct NoiseStark<F: RichField + Extendable<D>, const D: usize> {
    pub program: NoiseProgram,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> NoiseStark<F, D> {
    pub fn new(program: NoiseProgram) -> Self {
        Self {
            program,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for NoiseStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, NUM_NOISE_COLUMNS, NUM_NOISE_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, NUM_NOISE_COLUMNS, NUM_NOISE_PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_noise_constraints(vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_noise_constraints(vars, &mut evaluator);
    }

    fn constraint_degree(&self) -> usize {
        3
    }

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
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};

    use super::super::columns::NOISE_COL_MAP;
    use super::*;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;
    type Stk = NoiseStark<F, D>;

    fn to_u64(x: F) -> u64 {
        x.to_canonical_u64()
    }

    /// A deterministic pseudo-random byte line (never all-tiny, so `norm_scaled >= 128`).
    fn sample_bytes(rank: usize, seed: u64) -> Vec<u8> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..rank)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    fn trace(rank: usize, seed: u64) -> (NoiseProgram, Vec<[F; NUM_NOISE_COLUMNS]>, Vec<u8>) {
        let program = NoiseProgram::new(rank);
        let bytes = sample_bytes(rank, seed);
        let rows = program.generate_trace::<F>(&bytes);
        (program, rows, bytes)
    }

    fn constraints_violated(stark: &Stk, rows: &[[F; NUM_NOISE_COLUMNS]]) -> bool {
        let n = rows.len();
        (0..n).any(|i| {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], &[]);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::from_bool(i != n - 1),
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            consumer.accumulators().iter().any(|&acc| acc != F::ZERO)
        })
    }

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        for (rank, seed) in [(16usize, 1u64), (32, 7), (48, 99), (64, 123)] {
            let (program, rows, _) = trace(rank, seed);
            assert_eq!(rows.len(), rank.next_power_of_two().max(2));
            // Known columns are bit-exact with the trace fill.
            let known = program.known_values::<F>();
            assert_eq!(known.len(), 7);
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(known[0].values[r], row[NOISE_COL_MAP.is_pad], "is_pad row {r}");
                assert_eq!(known[1].values[r], row[NOISE_COL_MAP.is_block_start], "is_block_start row {r}");
                assert_eq!(known[2].values[r], row[NOISE_COL_MAP.is_block_final], "is_block_final row {r}");
                assert_eq!(known[3].values[r], row[NOISE_COL_MAP.global_index], "global_index row {r}");
                assert_eq!(known[4].values[r], row[NOISE_COL_MAP.operand_mult], "operand_mult row {r}");
                assert_eq!(known[5].values[r], row[NOISE_COL_MAP.is_egress_pair], "is_egress_pair row {r}");
                assert_eq!(known[6].values[r], row[NOISE_COL_MAP.egress_key], "egress_key row {r}");
            }
            assert!(
                !constraints_violated(&Stk::new(program), &rows),
                "rank {rank} seed {seed} honest trace violated a constraint"
            );
        }
    }

    /// Bit-exact against the real `normalize_line` (reached through the pub(crate)
    /// `sample_line`): reconstruct the same keyed-BLAKE3 XOF bytes the reference consumes, feed
    /// them to `generate_trace`, and compare the per-entry FP16 output to the reference line.
    #[test]
    fn trace_is_bit_exact_vs_normalize_line() {
        use crate::v5::api::noise::{NoiseFactor, Side, sample_line, subkey};

        const LABEL_NOISE_LINE: &[u8] = b"pearl/v4/FP16/noise-line";
        for (rank, seed_byte) in [(16u16, 4u8), (32, 3), (48, 17), (64, 200)] {
            let seed = [seed_byte; 32];
            let line = 5u32;
            let reference = sample_line(&seed, Side::A, NoiseFactor::E, line, rank);

            // Reconstruct the raw XOF bytes `normalize_line` was applied to.
            let key = subkey(LABEL_NOISE_LINE, Some(&seed));
            let mut material = vec![Side::A as u8, NoiseFactor::E as u8];
            material.extend_from_slice(&line.to_le_bytes());
            material.resize(64, 0);
            let mut bytes = vec![0u8; usize::from(rank)];
            let mut hasher = blake3::Hasher::new_keyed(&key);
            hasher.update(&material);
            hasher.finalize_xof().fill(&mut bytes);

            let program = NoiseProgram::new(usize::from(rank));
            let rows = program.generate_trace::<F>(&bytes);
            let mine: Vec<u16> = rows
                .iter()
                .take(usize::from(rank))
                .map(|row| {
                    let v: &NoiseColumnsView<F> = row.borrow();
                    to_u64(v.entry_fp16) as u16
                })
                .collect();
            assert_eq!(mine, reference, "FP16 line mismatch at rank {rank}");
        }
    }

    /// Multi-line: a NoiseStark instance proving several lines end to end is bit-exact with
    /// `sample_line` per line, its known columns match the fill, and the whole trace satisfies every
    /// constraint (per-block prefix-sum reset, scalar constancy, trailing padding).
    #[test]
    fn multi_line_trace_is_bit_exact_and_satisfies_constraints() {
        use crate::v5::api::noise::{NoiseFactor, Side, sample_line, subkey};
        const LABEL_NOISE_LINE: &[u8] = b"pearl/v4/FP16/noise-line";
        let rank = 32u16;
        let seed = [0x5au8; 32];

        // Reconstruct the raw XOF bytes of one line (the free-witness bytes NoiseStark consumes).
        let xof = |side: Side, factor: NoiseFactor, line: u32| -> Vec<u8> {
            let key = subkey(LABEL_NOISE_LINE, Some(&seed));
            let mut material = vec![side as u8, factor as u8];
            material.extend_from_slice(&line.to_le_bytes());
            material.resize(64, 0);
            let mut bytes = vec![0u8; usize::from(rank)];
            let mut hasher = blake3::Hasher::new_keyed(&key);
            hasher.update(&material);
            hasher.finalize_xof().fill(&mut bytes);
            bytes
        };

        // Four lines with distinct addresses, each with its own multiplicity. `Side`/`NoiseFactor`
        // are not `Copy`, so a fresh pair is built per line from the byte discriminants.
        let addr = |which: u8| -> (Side, NoiseFactor) {
            match which {
                0 => (Side::A, NoiseFactor::E),
                1 => (Side::A, NoiseFactor::F),
                2 => (Side::B, NoiseFactor::E),
                _ => (Side::B, NoiseFactor::F),
            }
        };
        let specs = [(0u8, 0u32, 3u64), (1, 1, 2), (2, 2, 3), (3, 0, 4)];
        let mut bytes: Vec<u8> = Vec::new();
        let mut refs: Vec<Vec<u16>> = Vec::new();
        let mut mults: Vec<u64> = Vec::new();
        for &(which, line, mult) in &specs {
            let (s, f) = addr(which);
            bytes.extend_from_slice(&xof(s, f, line));
            let (s, f) = addr(which);
            refs.push(sample_line(&seed, s, f, line, rank));
            mults.push(mult);
        }
        // Height padded off the live rows (4 lines * 32 = 128 -> 256 here for a padding block).
        let num_rows = 256usize;
        let program = NoiseProgram::with_lines(usize::from(rank), mults.clone(), num_rows);
        let rows = program.generate_trace::<F>(&bytes);
        assert_eq!(rows.len(), num_rows);

        // Per-line bit-exactness.
        for (l, reference) in refs.iter().enumerate() {
            let mine: Vec<u16> = (0..rank as usize)
                .map(|e| {
                    let v: &NoiseColumnsView<F> = rows[l * rank as usize + e].borrow();
                    to_u64(v.entry_fp16) as u16
                })
                .collect();
            assert_eq!(&mine, reference, "line {l} mismatch");
        }

        // Known columns match the fill (incl. GLOBAL_INDEX and OPERAND_MULT).
        let known = program.known_values::<F>();
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(known[3].values[r], row[NOISE_COL_MAP.global_index], "global_index row {r}");
            assert_eq!(known[4].values[r], row[NOISE_COL_MAP.operand_mult], "operand_mult row {r}");
        }
        // GLOBAL_INDEX is the live row index; OPERAND_MULT is the line's multiplicity.
        for (l, &mult) in mults.iter().enumerate() {
            let v: &NoiseColumnsView<F> = rows[l * rank as usize].borrow();
            assert_eq!(to_u64(v.global_index), (l * rank as usize) as u64);
            assert_eq!(to_u64(v.operand_mult), mult);
        }

        assert!(!constraints_violated(&Stk::new(program), &rows), "multi-line honest trace violated a constraint");
    }

    #[test]
    fn padded_trace_known_column_matches_fill() {
        // A non-power-of-two rank (48 -> 64): pad rows carry IS_PAD = 1 and zero entry columns.
        let (program, rows, _) = trace(48, 321);
        assert_eq!(rows.len(), 64);
        for (r, row) in rows.iter().enumerate() {
            let v: &NoiseColumnsView<F> = row.borrow();
            if r >= 48 {
                assert_eq!(v.is_pad, F::ONE);
                assert_eq!(v.entry_fp16, F::ZERO, "pad entry {r}");
                assert_eq!(v.magnitude, F::ZERO, "pad magnitude {r}");
            } else {
                assert_eq!(v.is_pad, F::ZERO);
            }
        }
        assert!(!constraints_violated(&Stk::new(program), &rows));
    }

    #[test]
    fn tampered_traces_fail() {
        let (program, rows, _) = trace(32, 55);
        let stark = Stk::new(program);
        assert!(!constraints_violated(&stark, &rows), "baseline honest trace must pass");
        let cases = [
            ("byte", NOISE_COL_MAP.byte),
            ("sign_bit", NOISE_COL_MAP.sign_bit),
            ("magnitude", NOISE_COL_MAP.magnitude),
            ("running_sumsq", NOISE_COL_MAP.running_sumsq),
            ("total_sumsq", NOISE_COL_MAP.total_sumsq),
            ("norm_scaled", NOISE_COL_MAP.norm_scaled),
            ("isqrt_rem", NOISE_COL_MAP.isqrt_rem),
            ("denom_blo", NOISE_COL_MAP.denom_blo),
            ("div_blo", NOISE_COL_MAP.div_blo),
            ("entry_prod", NOISE_COL_MAP.entry_prod),
            ("entry_blo", NOISE_COL_MAP.entry_blo),
            ("entry_fp16_exp", NOISE_COL_MAP.entry_fp16_exp),
            ("entry_fp16", NOISE_COL_MAP.entry_fp16),
        ];
        for (name, col) in cases {
            let mut forged = rows.clone();
            // Row 0 is a live row with nonzero entry; tamper there.
            forged[0][col] += F::ONE;
            assert!(constraints_violated(&stark, &forged), "{name} tamper undetected");
        }
    }

    /// Binade-bottom uniqueness: at a power-of-two rounding the former (relaxed) witness — the one
    /// using the half-ulp lower boundary `(4*M - 2)*factor` without the `+IS_BOTTOM` correction,
    /// i.e. differing from the true boundary by exactly the former quarter-ulp slack — must now be
    /// REJECTED by the AIR, while the honest (corrected) trace still passes.
    ///
    /// The line `byte = 31` (magnitude 32, `rank = 1`) makes `q = 1024`, so `denom = 1024`,
    /// `scale = 8`, and every `entry = 256` are all exact powers of two: all three brackets hit
    /// their binade bottom at once (`*_BOTTOM = 1`, `*_MANT = 0`).
    #[test]
    fn binade_bottom_correction_pins_the_rounding() {
        let program = NoiseProgram::new(1);
        let bytes = vec![31u8];
        let rows = program.generate_trace::<F>(&bytes);
        let stark = Stk::new(program);
        assert!(!constraints_violated(&stark, &rows), "honest power-of-two trace must pass");

        // Test premise: row 0 is at the binade bottom of all three brackets.
        let v: &NoiseColumnsView<F> = rows[0].borrow();
        assert_eq!(v.denom_bottom, F::ONE, "denom is a power of two");
        assert_eq!(v.scale_bottom, F::ONE, "scale is a power of two");
        assert_eq!(v.entry_bottom, F::ONE, "entry is a power of two");

        // (a) Reverting each lower boundary product to the uncorrected `(4*M - 2)*factor` value
        // (subtract one `factor`, since corrected = uncorrected + 1*factor at the bottom) is the
        // "former quarter-ulp" witness; the AIR's `blo = (... + bottom)*factor` now rejects it.
        let denom_factor = v.denom_pow; // denom bracket factor = 2^gd
        let div_factor = F::from_canonical_u64(128) + v.denom_mant; // division factor = M_DEN
        let entry_factor = v.entry_pow; // multiply bracket factor = 2^gm
        for (name, blo_col, factor) in [
            ("denom_blo", NOISE_COL_MAP.denom_blo, denom_factor),
            ("div_blo", NOISE_COL_MAP.div_blo, div_factor),
            ("entry_blo", NOISE_COL_MAP.entry_blo, entry_factor),
        ] {
            let mut forged = rows.clone();
            forged[0][blo_col] -= factor;
            assert!(
                constraints_violated(&stark, &forged),
                "{name}: uncorrected (half-ulp) lower boundary must be rejected"
            );
        }

        // (b) The correction flag itself is pinned: clearing any `*_BOTTOM` at a power of two
        // violates the is-zero gadget (mant = 0 forces the flag to 1).
        for (name, flag_col) in [
            ("denom_bottom", NOISE_COL_MAP.denom_bottom),
            ("scale_bottom", NOISE_COL_MAP.scale_bottom),
            ("entry_bottom", NOISE_COL_MAP.entry_bottom),
        ] {
            let mut forged = rows.clone();
            forged[0][flag_col] = F::ZERO;
            assert!(constraints_violated(&stark, &forged), "{name}: flag must stay pinned to 1");
        }
    }

    #[test]
    fn degree_is_at_most_three() {
        let (program, _, _) = trace(32, 1);
        test_stark_low_degree::<F, Stk, D>(Stk::new(program)).unwrap();
    }

    #[test]
    fn circuit_constraints_match_native() {
        let (program, _, _) = trace(32, 1);
        test_stark_circuit_constraints::<F, C, Stk, D>(Stk::new(program)).unwrap();
    }
}
