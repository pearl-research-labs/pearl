//! AIR and trace generation for authenticated prequant decode and FP8 input quantization.
//!
//! # Statement and division of work
//!
//! One trace row handles one A element and one B element. A matrix row is a group of `k`
//! trace rows; each group contains `k/8` scale blocks. For raw signed byte `q`, committed
//! bf16 block scale `s`, public-seed binary32 noise `n`, and bf16 row scales `alpha, beta`,
//! the live-side computation is
//!
//! ```text
//! X          = RN_bf16(q*s)                         // decode for norms/policy
//! c          = RN32(alpha*s)                        // one certificate per block
//! signal     = c*q                                  // exact binary32 product
//! mixed      = RN32(beta*n + signal)                // one fused rounding
//! code_noised = Q8(mixed)                           // saturating RNE E4M3 cast
//! output     = code_noised                          // H100
//!              rewrite(code_noised, row data, coin) // B200
//! ```
//!
//! `RN_bf16` and `RN32` mean round-to-nearest, ties-to-even into the named finite format.
//! The coefficient has at most 16 significant bits; multiplying by an int8 adds at most
//! eight, so `c*q` is exact in binary32 on the finite domain enforced by the AIR. The fold
//! uses the original block scale and byte, while the policy uses decoded `X`; substituting
//! `alpha*X` for `c*q`, or rounding `beta*n` before addition, changes the proved operation.
//!
//! Blake3 authenticates the bytes/scales. This AIR accumulates the canonical row statistics;
//! Scale verifies their norm/scale chain and exports the exact noise-scale fields.
//! CoefficientStark certifies `c` once per block, via the CTL in [`super::ctl`]. Matmul
//! consumes the final device-specific output codes, together with the clean-data scores
//! described below. All arithmetic and lookup constraints are required: this is a batch AIR.
//!
//! # Bf16 representation and decoded values (groups P/U/S)
//!
//! For finite fields `(sign, exp, mantissa, exp_is_zero)`, with `exp_is_zero = [exp = 0]`,
//!
//! ```text
//! M(V)  = 128*(1 - exp_is_zero) + mantissa
//! E*(V) = exp + exp_is_zero
//! V     = (-1)^sign * M(V) * 2^(E*(V) - 134).
//! ```
//!
//! Here `134 = 127 + 7` is exponent bias plus fraction width. Taking `E* = 1` at raw
//! exponent zero covers subnormals and signed zero. An exact product has integer significand
//! `M(a)*M(b) < 2^16` and binary scale `E*(a)+E*(b)-268`. INT8DEC binds the raw byte to its
//! exact bf16 fields; block transitions copy the scale, and CLAMP22/RNERND prove the decoded
//! bf16 product, including subnormal fade and signed zero. Scale later certifies the
//! group-constant `alpha`, `beta`, frame exponent, and sigma fields.
//!
//! # Canonical framed row statistics (groups B/F)
//!
//! L2 uses the pre-rounding block data, whereas `linf` uses the decoded `X`. For block `b`,
//! define
//!
//! ```text
//! n_b     = sum_{j in b} q_j^2
//! p_b     = M(s_b)^2 * n_b
//! F       = max_{b: p_b != 0} 2*E*(s_b), or 0 for an all-zero row
//! Wl2     = 27 - ceil(log2(k))                     // 11..17 on the wire domain
//! shift_b = F - 2*E*(s_b)
//! TERM_b  = floor(p_b * 2^Wl2 / 2^shift_b)         // 0 when p_b = 0
//! S       = sum_b TERM_b
//! max_abs = max_j code_bf16(|X_j|).
//! ```
//!
//! The canonical represented mean-square is `S*2^(F-268-Wl2)/k`, deliberately computed
//! before the elementwise bf16 decode rounding. A common exponent frame avoids adding
//! values hundreds of binary positions apart in one field element. Each block loses less
//! than one frame unit; these floors define the protocol norm rather than an approximate
//! witness for a different norm.
//!
//! Honest blocks satisfy `n_b <= 2^17` and `p_b < 2^33`; the ranged gadget admits the wider
//! `p_b < 2^37`. Its four 16-bit dividend limbs, with six top bits, enforce
//! `p_b*2^Wl2 < 2^54`. The frame choice gives `S < 2^61`, below Scale's `2^62` cap. For
//! `shift_b >= 54`, the quotient is identically zero. Otherwise one selected 16-bit limb
//! is split at `shift_b mod 16`, with POW2D powers and bounded quotient/remainder proving
//! the exact floor. Running maxima are selected from an attained input and range-checked
//! against both candidates; a prover cannot lower the frame or the decoded maximum.
//!
//! Group-final rows export `(S, F, max_abs, alpha/beta, dead bound/count, sigma fields)`
//! to Scale, which proves the square root, grid snap, scale chain, and policy gates.
//!
//! # Binary32 fold and FP8 cast
//!
//! `crate::v4::circuit::fp32` gives the full alignment/rounding equations. In outline,
//! the exact significand products for `beta*n` and `c*q` are normalized to 32 bits and
//! aligned with three extra low bits. Euclidean division retains the smaller operand's
//! quotient and a signed nonzero-tail certificate. The signed integer sum and this tail
//! determine the single RN32 result, including cancellation, subnormal outputs, and ties.
//! The sign of an exact zero follows binary32 FMA rules.
//!
//! To use the 16-bit QCAST table without an intermediate bf16 RNE, the rounded binary32
//! result is compressed to an eight-significant-bit *round-to-odd proxy*: discard the low
//! 16 fraction bits and set the retained low bit if any discarded bit is nonzero. E4M3 has
//! four significant bits, so the proxy retains enough guard information for QCAST's RNE
//! to equal the direct binary32-to-E4M3 cast. The proxy is an encoding for the lookup;
//! it is not an additional bf16 rounding in the computation.
//!
//! # Clean-data policy certificates (groups C/V)
//!
//! The dead-entry flag is `IS_DEAD = [|X| >= DEAD_BOUND]` on live rows. Boolean flags and
//! complementary RC16 inequalities prove both comparison branches. The bound is
//! group-constant and CTL-bound to Scale's `8*delta*l2f`; the group count resets at the
//! boundary and is explicitly anchored on row zero. This lets Scale enforce the per-side
//! dead-entry budget, without assuming equality of A and B strip lengths.
//!
//! For the unpredictable-summand check (check 3), let `a = |alpha*X|` and
//! `sigma = delta*alpha*l2f`. These are exact products of bf16 values, independent of the
//! sampled noise and of B200's output rewrite. Write nonzero addends as
//!
//! ```text
//! a     = X_NORM     * 2^(X_ENC     - 268 - 15)
//! sigma = SIGMA_NORM * 2^(SIGMA_ENC - 268 - 15),
//! ```
//!
//! with 16-bit normalized significands in `[2^15, 2^16)`; zero `a` has both fields zero.
//! Order the addends by exponent, call their significands `big, small`, and let
//! `gap = |X_ENC-SIGMA_ENC|`. The exact integer score chain is
//!
//! ```text
//! V      = big^2 + floor(small^2 / 2^(2*gap))
//! kappa  = floor(V / 2^17)
//! lambda = 128*max(X_ENC, SIGMA_ENC) - 2624 + floor(64*log2(kappa)).
//! ```
//!
//! Two divisions by `2^gap` avoid one oversized divisor. For `gap >= 16`, the smaller
//! quotient is exactly zero because `small^2 < 2^32`; the far branch loses no information.
//! POW2D, WIDTH16, and LOG16 region 0 bind the shift powers, normalizations, and logarithm.
//! The score is a downward approximation to `64*(log2(a^2+sigma^2)+508)` with error less
//! than `65/64` score units; [`crate::v4::circuit::unpredictability`] derives this bound
//! and the Matmul skip predicate. Phantom elements use the all-zero score sentinel.
//!
//! # B200 quantize-first downscaling and Gaussian row gate (check 4)
//!
//! Put `s = alpha*l2f` (signal RMS), so `sigma = s/2` on B200. Scale binds the normalized
//! pair `(SIGMA_NORM, SIGMA_ENC)` and the strip byte
//! `t = Q8(RN32(g*s))`, where `g` is binary32 0.1 for `s < 8` and 0.08 otherwise.
//! It also restricts `s` to `[4, 256)`. Consequently
//! `e = SIGMA_ENC-269 = floor(log2(s))-2` is an E4M3 exponent field in `0..=5`.
//!
//! Each four-element block shares a deterministic coin: extract bits 8..14 from the
//! first element's rounded binary32 `mixed` word and test whether that 7-bit integer is
//! below 112. BYTES2/PAIR128 and two RC16 comparison branches bind the extraction and
//! decision; transitions copy the coin within the block. For `c = code_noised`,
//!
//! ```text
//! inside      = coin && ((c & 127) < t)
//! output_code = (c & 128) | (8*e) | (c & 7) | 1    if inside
//!               c                                otherwise.
//! ```
//!
//! DOWNSCALEREWRITE proves this byte map; the operand-code CTL exports its output. Keeping
//! the sign/fraction bits and forcing an odd fraction describes the rewrite exactly,
//! including the subnormal exponent-zero slot; it is not a uniform division of `mixed`.
//!
//! The Gaussian gate reuses the clean-data score. Let `base = lambda(a=0, sigma)`,
//! `d = lambda-base`, and `region = 1+class`, with `base` and the cap class bound by Scale.
//! LOG16 serves the integer kernel
//!
//! ```text
//! T(d) = floor(2^16 * exp(-(2^((50*d+51)/3200)-1)/2))
//! K(d) = min(T(d), cap_class),    0 <= d < 8192.
//! ```
//!
//! `T(d)=0` for `d >= 290`. The `51/50 = 1.02` score slack makes this a conservative
//! Gaussian proxy; its integer table is certified in `unpredictability::gaussian_tests`.
//! The two-component key `(region, d)` prevents an invalid difference from aliasing
//! another region. A running sum adds `K(d)` per element, resets at group boundaries,
//! and exports its group-final value to Scale, which enforces the minimum row proxy.
//! The Gaussian lookup and rewrite use the same authenticated sigma/threshold data.
//!
//! # Trace boundaries, lookups, and integer soundness
//!
//! The height is `next_power_of_two(max(h,w)*k)`. Each side has its own live prefix;
//! phantom data/noise are zero, with a valid positive-normal alpha filler. The final row
//! is group-final even when the trailing phantom group is truncated. Group/block resets
//! therefore close cyclic next-row constraints without joining unrelated live groups.
//! The verifier recomputes only the two packed binary32 noise words. Group G pins everything
//! else about the schedule in the AIR: the noise decodes recompose to those words (with
//! EXPINFO, RC16 and PAIR128 bounding their fields, so the decode is unique), the element
//! index is the row counter, the parity, block-start and group-final flags follow cyclic
//! counters whose inverse witnesses force each flag exactly at its boundary, the block index
//! increments at block starts, and each side's liveness starts at one and is forced to zero
//! exactly at its public bound `h*k` resp. `w*k`. B200 ties the coin-start flag to the block
//! counter the same way (G13).
//!
//! [`super::ctl`] declares the complementary range, decode, rounding, and CTL relations.
//! Free witness flags are constrained boolean here; lookup outputs inherit their table
//! domains. Bounded limbs and quotient/remainder complements make the relevant equations
//! integer-exact below the Goldilocks modulus `2^64-2^32+1`; missing negative lookup keys
//! certify inequalities. Both the native and recursive AIR use the same evaluator.

use core::borrow::Borrow;
use std::marker::PhantomData;
use std::sync::Arc;

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
    B_KEY_OFFSET_PUBLIC_INPUT, B_LIVE_BOUND_PUBLIC_INPUT, B200InputQuantColumnsView, Bf16FieldsView, BlockL2Columns,
    InputQuantColumnsView, InputQuantKnownColumns, InputQuantScheduleColumns, InputQuantSideColumns, K_PUBLIC_INPUT,
    MulDecodeBlockView, NUM_B200_INPUT_QUANT_COLUMNS, NUM_INPUT_QUANT_COLUMNS, NUM_INPUT_QUANT_KNOWN_COLUMNS,
    NUM_INPUT_QUANT_PUBLIC_INPUTS, WL2_POW_PUBLIC_INPUT, input_quant_column_map,
};
use crate::v4::api::compute::{bf16_clamp_sym, bf16_max, bf16_mul, fp32_fma};
use crate::v4::api::dtype::{bf16_to_f32, f32_to_fp8_e4m3};
use crate::v4::api::prequant::{BLOCK_SIZE, l2_frame_width};
use crate::v4::api::public_params::Device;
use crate::v4::api::quantization::Fp8E4M3Quant;
use crate::v4::circuit::fp32::{
    self, AlignmentColumns, DownscaleColumns, Fp32Fields, Fp32Magnitude, OddProxyColumns, Round32Columns,
};
use crate::v4::circuit::scale_stark::stark::{
    CUT_BIAS, NORM_FLOOR_CODE, SLOT_MAX, ScaleRowTuple, rne_sqrt_hat, rnernd_reference,
};
use crate::v4::circuit::unpredictability::{LambdaWitness, lambda_witness, security_lookup_value, security_row_parameters};
use crate::v4::circuit::utils::evaluator::Evaluator;
use crate::v4::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v4::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

/// The bf16 M-form bias: `value = M * 2^(E* - 134)`, so a product's LSB scale is
/// `E*(a) + E*(b) - 268`.
const MUL_LSB_OFFSET: i64 = 268;
/// bf16 subnormal grid LSB scale (`2^-133`); the signed fade cut is
/// `clamp(-133 - LSB_SCALE, -7, 18)` and the RNERND slot adds [`CUT_BIAS`].
const SUBNORMAL_LSB: i64 = -133;
/// bf16 code of 448 (the fp8 E4M3 clamp bound; `fp8_to_bf16(0x7E)`).
const BF16_CODE_448: u16 = 0x43E0;
/// Far-shift threshold: a block with `sigma >= 54` floors to `TERM = 0` exactly,
/// because `block_l2_product * 2^Wl2 < 2^54`.
const FAR_SIGMA: u64 = 54;

// ==================================================================================================
// Program and trace generation
// ==================================================================================================

/// The public geometry of one FP8 input-quantization run.
///
/// The packed noise words are verifier-known from the public seeds; the schedule follows from
/// this geometry through group G's constraints and public inputs. The strip bytes and scales
/// remain private witness data.
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
    /// Committed arithmetic profile.
    pub device: Device,
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

    /// `Wl2 = 27 - ceil(log2 k)`: the job's block-L2 frame width ([`l2_frame_width`].
    /// In `[11, 17]` over the envelope `1024 <= k <= 2^16`.
    pub fn wl2(&self) -> u32 {
        l2_frame_width(self.k).expect("sanctioned k")
    }

    /// Generates the InputQuantStark trace and public inputs.
    ///
    /// - `a_int8` / `b_int8`: the int8 values planes, row-major (`h*k` resp. `w*k` values; the
    ///   B strip's w opened columns are its "rows" here, matching `B200::matmul_fp8`'s `b`).
    /// - `a_scales` / `b_scales`: the bf16 block-scale planes (`h*k/8` resp. `w*k/8` codes).
    /// - `a_noise` / `b_noise`: binary32 noise `(E @ F)[element]`, derived from the public
    ///   seeds and independently recomputed by the verifier.
    ///
    /// Panics if the geometry is unsupported or a bf16 operation overflows.
    /// Builds the per-group side witnesses natively, together with the compact per-side
    /// outputs the downstream tables consume: the final device codes (the downscale rewrite's
    /// output on B200) and summand scores for Matmul, the per-matrix-row tuples and B200's
    /// Gaussian security sums for Scale, and every block's `(alpha, block scale)` codes for
    /// Coefficient. One parallel job per (group, side); every dead side shares one canonical
    /// all-zero witness by `Arc` (the trailing group may be truncated and reads only its first
    /// `len` elements, whose values do not depend on the witness length).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_native(
        &self,
        a_int8: &[i8],
        a_scales: &[u16],
        a_noise: &[f32],
        b_int8: &[i8],
        b_scales: &[u16],
        b_noise: &[f32],
    ) -> InputQuantNative {
        use plonky2_maybe_rayon::{MaybeIntoParIter, ParallelIterator};

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
        assert_eq!(a_noise.len(), h * k, "a_noise must be h*k binary32 values");
        assert_eq!(b_noise.len(), w * k, "b_noise must be w*k binary32 values");

        let n_groups = num_rows.div_ceil(k);
        let blocks = k / BLOCK_SIZE;
        // Dead-side phantom inputs use zero bytes, scales, and noise. The trailing dead group
        // may be truncated (`k` need not divide the power-of-two height); its length is still a
        // multiple of BLOCK_SIZE, and a prefix of the full-length canonical witness is
        // value-identical to a witness built at that length.
        let zero_int8 = vec![0i8; k];
        let zero_scales = vec![0u16; blocks];
        let zero_noise = vec![0f32; k];
        let dead_canonical: Arc<SideWitness> = Arc::new(SideWitness::build(self, &zero_int8, &zero_scales, &zero_noise, false));

        type SideOutput = (SideNative, Vec<u8>, Vec<u64>, Option<ScaleRowTuple>, Vec<[u16; 2]>);
        let built: Vec<SideOutput> = (0..2 * n_groups)
            .into_par_iter()
            .map(|job| {
                let (g, is_a) = (job / 2, job % 2 == 0);
                let live = if is_a { g < h } else { g < w };
                if !live {
                    let side = SideNative {
                        witness: Arc::clone(&dead_canonical),
                        live: false,
                        security: None,
                    };
                    return (side, Vec::new(), Vec::new(), None, Vec::new());
                }
                let (int8, scales, noise) = if is_a {
                    (
                        &a_int8[g * k..(g + 1) * k],
                        &a_scales[g * blocks..(g + 1) * blocks],
                        &a_noise[g * k..(g + 1) * k],
                    )
                } else {
                    (
                        &b_int8[g * k..(g + 1) * k],
                        &b_scales[g * blocks..(g + 1) * blocks],
                        &b_noise[g * k..(g + 1) * k],
                    )
                };
                let witness = SideWitness::build(self, int8, scales, noise, true);
                let codes: Vec<u8> = (0..k).map(|j| witness.output_code(self.device, j, true)).collect();
                let lambdas: Vec<u64> = witness.elements.iter().map(|e| e.score.lambda).collect();
                // The group-final row's block-L2 state is the side's Scale tuple.
                let last = &witness.elements[k - 1];
                let tuple = ScaleRowTuple {
                    l2_frame_sum: last.blk.running_l2_frame_sum,
                    frame_doubled_scale_exponent: witness.scales.frame_doubled_scale_exponent as u32,
                    max_abs: last.max_abs as u32,
                    dead_count: last.dead_count as u32,
                };
                // Coefficient certifies `alpha * block_scale` from the scale's magnitude fields.
                let coefficient_blocks: Vec<[u16; 2]> = scales
                    .iter()
                    .map(|&scale| [witness.scales.alpha_code, scale & 0x7FFF])
                    .collect();
                // B200: the side's LOG16 region and key base, and each element's capped Gaussian
                // value with the running sum Scale's row gate consumes.
                let security = (self.device == Device::B200).then(|| {
                    let (_, _, region, base) = security_row_parameters(witness.scales.sigma_norm, witness.scales.sigma_enc);
                    let mut sum = 0u64;
                    let terms = witness
                        .elements
                        .iter()
                        .map(|e| {
                            let index = e.score.lambda.checked_sub(base).expect("Gaussian index is nonnegative");
                            let value = security_lookup_value(region, index);
                            sum += value;
                            (value, sum)
                        })
                        .collect();
                    (region, base, terms)
                });
                let side = SideNative {
                    witness: Arc::new(witness),
                    live: true,
                    security,
                };
                (side, codes, lambdas, Some(tuple), coefficient_blocks)
            })
            .collect();

        // Reassemble in (group, A then B) order. Security sums and Coefficient blocks list all
        // of A's rows/blocks, then all of B's.
        let mut native = InputQuantNative {
            groups: Vec::with_capacity(n_groups),
            a_codes: Vec::with_capacity(h * k),
            b_codes: Vec::with_capacity(w * k),
            a_lambdas: Vec::with_capacity(h * k),
            b_lambdas: Vec::with_capacity(w * k),
            a_tuples: Vec::with_capacity(h),
            b_tuples: Vec::with_capacity(w),
            security_sums: Vec::new(),
            coefficient_blocks: Vec::with_capacity((h + w) * blocks),
        };
        let mut b_security_sums = Vec::new();
        let mut b_blocks = Vec::with_capacity(w * blocks);
        let mut built = built.into_iter();
        for _ in 0..n_groups {
            let (side_a, a_codes, a_lambdas, a_tuple, a_blocks) = built.next().expect("A side output");
            let (side_b, b_codes, b_lambdas, b_tuple, blocks_b) = built.next().expect("B side output");
            if let Some((_, _, terms)) = &side_a.security {
                native.security_sums.push(terms.last().expect("live side has k elements").1);
            }
            if let Some((_, _, terms)) = &side_b.security {
                b_security_sums.push(terms.last().expect("live side has k elements").1);
            }
            native.a_codes.extend(a_codes);
            native.b_codes.extend(b_codes);
            native.a_lambdas.extend(a_lambdas);
            native.b_lambdas.extend(b_lambdas);
            native.a_tuples.extend(a_tuple);
            native.b_tuples.extend(b_tuple);
            native.coefficient_blocks.extend(a_blocks);
            b_blocks.extend(blocks_b);
            native.groups.push([side_a, side_b]);
        }
        native.security_sums.extend(b_security_sums);
        native.coefficient_blocks.extend(b_blocks);
        native
    }

    /// Per-fill tables of the schedule's inverse witnesses that depend only on a row's
    /// in-group or in-block position (G4, G9, G13); the liveness flips (G12) are per row.
    fn schedule_tables<F: RichField>(&self) -> ScheduleTables<F> {
        ScheduleTables {
            group_final_inv: (0..self.k).map(|j| inverse_or_zero::<F>(j, self.k - 1)).collect(),
            block8_inv: core::array::from_fn(|position| inverse_or_zero::<F>(position, BLOCK_SIZE - 1)),
            coin_inv: core::array::from_fn(|position| inverse_or_zero::<F>(position, 4)),
        }
    }

    /// One trace row from the native witnesses: the verifier-known noise words, group G's
    /// decode and schedule columns, then both sides' columns (plus the B200 coin and
    /// downscale/Gaussian columns). Shared by the row-major generator and the direct column
    /// fill, so the two are value-identical.
    #[allow(clippy::field_reassign_with_default)]
    fn fill_row<F: RichField, const WIDTH: usize>(
        &self,
        native: &InputQuantNative,
        tables: &ScheduleTables<F>,
        r: usize,
    ) -> [F; WIDTH] {
        let (h, w, k, num_rows) = (self.h, self.w, self.k, self.num_rows());
        let (g, j) = (r / k, r % k);
        let len = k.min(num_rows - g * k);
        let (a_live, b_live) = (g < h, g < w);
        let [side_a, side_b] = &native.groups[g];
        debug_assert!(side_a.live == a_live && side_b.live == b_live, "native liveness differs");
        let (noise_word_a, noise_word_b) = (side_a.witness.elements[j].noise_word, side_b.witness.elements[j].noise_word);
        let known = InputQuantKnownColumns {
            noise_word_a: F::from_canonical_u32(noise_word_a),
            noise_word_b: F::from_canonical_u32(noise_word_b),
        };
        let block_position = r % BLOCK_SIZE;
        let schedule = InputQuantScheduleColumns {
            noise_a: Fp32Fields::from_value(f32::from_bits(noise_word_a)),
            noise_b: Fp32Fields::from_value(f32::from_bits(noise_word_b)),
            noise_fraction_high_a: F::from_canonical_u32((noise_word_a & 0x7f_ffff) >> 16),
            noise_fraction_high_b: F::from_canonical_u32((noise_word_b & 0x7f_ffff) >> 16),
            is_group_final: F::from_bool(j == len - 1),
            element_index: F::from_canonical_usize(r),
            is_even_row: F::from_bool(r.is_multiple_of(2)),
            is_block_start: F::from_bool(block_position == 0),
            block_index: F::from_canonical_usize(r / BLOCK_SIZE),
            a_live: F::from_bool(a_live),
            b_live: F::from_bool(b_live),
            group_counter: F::from_canonical_usize(j),
            group_final_inv: tables.group_final_inv[j],
            block8_counter: F::from_canonical_usize(block_position),
            block8_counter_inv: tables.block8_inv[block_position],
            a_flip_inv: inverse_or_zero::<F>(r, h * k),
            b_flip_inv: inverse_or_zero::<F>(r, w * k),
        };
        match (self.device, WIDTH) {
            (Device::H100, NUM_INPUT_QUANT_COLUMNS) => {
                let mut row = InputQuantColumnsView::<F>::default();
                row.known = known;
                row.schedule = schedule;
                side_a.witness.fill(&mut row.a, j);
                side_b.witness.fill(&mut row.b, j);
                let row: [F; NUM_INPUT_QUANT_COLUMNS] = row.into();
                row.as_slice().try_into().unwrap()
            }
            (Device::B200, NUM_B200_INPUT_QUANT_COLUMNS) => {
                let mut row = B200InputQuantColumnsView::<F>::default();
                row.known = known;
                row.schedule = schedule;
                row.is_coin_start = F::from_bool(block_position.is_multiple_of(4));
                row.coin_inv = tables.coin_inv[block_position];
                side_a.witness.fill(&mut row.a.common, j);
                side_b.witness.fill(&mut row.b.common, j);
                side_a.witness.fill_downscale(&mut row.a.downscale, j, a_live);
                side_b.witness.fill_downscale(&mut row.b.downscale, j, b_live);
                // Each live side's LOG16 region and key base, its element's capped Gaussian
                // value and the running sum (phantom sides keep the zero columns).
                for (output, side) in [(&mut row.a, side_a), (&mut row.b, side_b)] {
                    if let Some((region, base, terms)) = &side.security {
                        let (value, sum) = terms[j];
                        output.security_region = F::from_canonical_u64(*region);
                        output.security_key_base = F::from_canonical_u64(*base);
                        output.gaussian_value = F::from_canonical_u64(value);
                        output.security_sum = F::from_canonical_u64(sum);
                    }
                }
                let row: [F; NUM_B200_INPUT_QUANT_COLUMNS] = row.into();
                row.as_slice().try_into().unwrap()
            }
            _ => panic!("input width differs from committed device"),
        }
    }

    /// The row-major trace (padding rows included) from the native witnesses, one parallel
    /// row at a time, without the row-0 blinding.
    pub(crate) fn rows_from_native<F: RichField, const WIDTH: usize>(&self, native: &InputQuantNative) -> Vec<[F; WIDTH]> {
        use plonky2_maybe_rayon::{MaybeIntoParIter, ParallelIterator};

        let num_rows = self.num_rows();
        assert_eq!(
            native.groups.len(),
            num_rows.div_ceil(self.k),
            "native groups must cover the trace"
        );
        let tables = self.schedule_tables::<F>();
        (0..num_rows)
            .into_par_iter()
            .map(|r| self.fill_row(native, &tables, r))
            .collect()
    }

    /// Generates the trace from the opened strips and the public binary32 noise: the native
    /// build, the parallel row fill, then the row-0 blinding.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_trace<F: RichField, const WIDTH: usize>(
        &self,
        a_int8: &[i8],
        a_scales: &[u16],
        a_noise: &[f32],
        b_int8: &[i8],
        b_scales: &[u16],
        b_noise: &[f32],
    ) -> (Vec<[F; WIDTH]>, [F; NUM_INPUT_QUANT_PUBLIC_INPUTS]) {
        let native = self.build_native(a_int8, a_scales, a_noise, b_int8, b_scales, b_noise);
        let mut rows = self.rows_from_native::<F, WIDTH>(&native);
        blind_trace(&mut rows, self.device);
        (rows, self.public_inputs())
    }

    /// The direct column fill: the trace written column-major straight into the caller's
    /// zeroed column buffers (tile workers own disjoint rows), then the same row-0 blinding
    /// as [`Self::generate_trace`].
    pub(crate) fn fill_columns<F: RichField, const WIDTH: usize>(&self, native: &InputQuantNative, cols: &mut [&mut [F]]) {
        use plonky2_maybe_rayon::{MaybeIntoParIter, ParallelIterator};

        let num_rows = self.num_rows();
        assert_eq!(
            native.groups.len(),
            num_rows.div_ceil(self.k),
            "native groups must cover the trace"
        );
        assert_eq!(cols.len(), WIDTH, "direct fill destination width");
        for col in cols.iter() {
            assert_eq!(col.len(), num_rows, "direct fill column length");
        }
        const TILE: usize = 4096;
        let tables = self.schedule_tables::<F>();
        // Tile workers share raw destination pointers and write disjoint rows.
        let writer = super::super::utils::witness::DisjointColumnWriter::<F, WIDTH>::new(cols);
        (0..num_rows.div_ceil(TILE)).into_par_iter().for_each(|tile| {
            let r0 = tile * TILE;
            let r1 = (r0 + TILE).min(num_rows);
            for r in r0..r1 {
                let row: [F; WIDTH] = self.fill_row(native, &tables, r);
                // SAFETY: tiles partition 0..num_rows. Only this worker writes row r; the
                // destination columns are inaccessible until the join.
                unsafe { writer.write_row(r, &row) };
            }
        });
        // Blinding: the same cells, with the same semantics, as `blind_trace`.
        for side in input_quant_column_map(self.device).sides {
            cols[side.block_l2.shift_remainder_power][0] = F::rand();
            cols[side.block_l2.shift_complement_power][0] = F::rand();
        }
    }

    /// The table's public inputs (`columns.rs` slot order): `2^Wl2`, the geometry scalars
    /// `[h*k, h*k/8, w, h]` the CTL column expressions read, and group G's `k` and `w*k`.
    pub fn public_inputs<F: RichField>(&self) -> [F; NUM_INPUT_QUANT_PUBLIC_INPUTS] {
        [
            F::from_canonical_u64(1u64 << self.wl2()),
            F::from_canonical_usize(self.h * self.k),
            F::from_canonical_usize(self.h * self.k / BLOCK_SIZE),
            F::from_canonical_usize(self.w),
            F::from_canonical_usize(self.h),
            F::from_canonical_usize(self.k),
            F::from_canonical_usize(self.w * self.k),
        ]
    }

    /// The two verifier-known columns: each side's packed binary32 noise word
    /// (`SIGN*2^31 + EXPONENT*2^23 + FRACTION`), zero past the side's live rows. The batch
    /// verifier recomputes them from the public seeds and checks the trace openings; group G
    /// then pins the decodes and the whole schedule to them.
    pub fn known_values<F: RichField>(&self, a_noise: &[f32], b_noise: &[f32]) -> Vec<PolynomialValues<F>> {
        use plonky2_maybe_rayon::*;

        let (h, w, k, num_rows) = (self.h, self.w, self.k, self.num_rows());
        assert_eq!(a_noise.len(), h * k, "a_noise must be h*k binary32 values");
        assert_eq!(b_noise.len(), w * k, "b_noise must be w*k binary32 values");

        let columns: Vec<PolynomialValues<F>> = [a_noise, b_noise]
            .into_iter()
            .map(|noise| {
                let mut column = vec![F::ZERO; num_rows];
                column[..noise.len()]
                    .par_chunks_mut(1024)
                    .zip(noise.par_chunks(1024))
                    .for_each(|(outputs, values)| {
                        for (output, &value) in outputs.iter_mut().zip(values) {
                            assert!(value.is_finite(), "nonfinite public noise");
                            *output = F::from_canonical_u32(value.to_bits());
                        }
                    });
                PolynomialValues::new(column)
            })
            .collect();
        debug_assert_eq!(columns.len(), NUM_INPUT_QUANT_KNOWN_COLUMNS);
        columns
    }
}

/// `1 / (value - target)`, or zero when they coincide: group G's inverse witnesses.
fn inverse_or_zero<F: RichField>(value: usize, target: usize) -> F {
    (F::from_canonical_usize(value) - F::from_canonical_usize(target))
        .try_inverse()
        .unwrap_or(F::ZERO)
}

/// Position-indexed inverse witnesses shared by every row of one fill.
struct ScheduleTables<F> {
    group_final_inv: Vec<F>,
    block8_inv: [F; BLOCK_SIZE],
    coin_inv: [F; BLOCK_SIZE],
}

/// Adds about 256 bits of blinding entropy to the trace commitment by randomizing four
/// otherwise-unused cells on row 0.
///
/// **Why.** The recursive wrapper publishes `zeta`, a Fiat-Shamir challenge derived from the
/// trace commitment. Without committed randomness, that public value could fingerprint a
/// low-entropy witness. Four random field elements provide about 256 bits of entropy, giving
/// statistical distance `2^-64` under the wrapper's blinding bound.
///
/// **Why these cells are safe.** Row 0 is the first row of a block, not its final row, so it
/// contributes no block-L2 term. The shift-power lookups are filtered off there, and every AIR
/// expression that could read these values is multiplied by a zero selector. Randomizing them
/// therefore changes the commitment but not the proved computation.
fn blind_trace<F: RichField, const WIDTH: usize>(rows: &mut [[F; WIDTH]], device: Device) {
    for side in input_quant_column_map(device).sides {
        rows[0][side.block_l2.shift_remainder_power] = F::rand();
        rows[0][side.block_l2.shift_complement_power] = F::rand();
    }
}

/// `bit_length(x)` for `x > 0` (0 for 0): the position of the leading 1.
fn bit_len(x: u64) -> u64 {
    (64 - x.leading_zeros()) as u64
}

// --------------------------------------------------------------------------------------------------
// bf16 field decode and the shared LUT mirrors (trace-generation side)
// --------------------------------------------------------------------------------------------------

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
    let x = p << wl2; // p < 2^37 envelope, Wl2 <= 17: < 2^54
    if sigma >= 64 { 0 } else { x >> sigma }
}

/// A rounded bf16 output block (the RNERND value tuple plus the exponent the consuming
/// constraint reconstructs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RoundedBf16 {
    /// Output exponent field (0 on the zero and subnormal branches, per U5).
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

/// The final bf16 rounding step: the committed RNERND row at `(v, slot)` plus the consuming
/// constraint's exponent reconstruction (U4), so witness generation is the table row by
/// construction.
///
/// `v` is an exact integer in units of `2^lsb_scale` and `slot` must be the CLAMP22 of that
/// scale's signed fade cut, which [`rnernd_reference`] converts back to the rounding position.
/// Subnormal outputs land on bf16's `2^-133` grid. It panics on overflow, matching the native
/// non-finite error path.
pub(crate) fn rnernd_scaled(v: u64, slot: u64, lsb_scale: i64) -> RoundedBf16 {
    debug_assert!(v < 1 << 17, "RNERND significands are pre-compressed to 17 bits");
    debug_assert_eq!(
        slot,
        clamp_slot(SUBNORMAL_LSB - lsb_scale),
        "slot must be the CLAMP22 of the LSB scale"
    );
    let (mantissa, width_adjust, is_zero, exp_is_zero) = rnernd_reference(v, slot);
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

pub(crate) fn qcast(code: u16) -> u8 {
    f32_to_fp8_e4m3(bf16_to_f32(bf16_clamp_sym(code, BF16_CODE_448))).expect("clamped bf16 is finite")
}

// --------------------------------------------------------------------------------------------------
// The protocol L2 (integer frame sum -> RNE sqrt -> grid snap) and the group scale witness
// --------------------------------------------------------------------------------------------------

/// The native `round_l2_to_grid` (private in `api::fp8::quantization`, mirrored here): round a
/// nonnegative bf16 to the nearest multiple of 4 ulps, ties up, by code arithmetic.
fn round_l2_to_grid(l2: u16) -> u16 {
    l2.wrapping_add(2) & !3
}

/// One group's (matrix row's) scale witness: the block frame, the alpha/beta the noising
/// uses, and the jackpot liveness bound (all CTL-bound to ScaleStark's derivation).
#[derive(Clone, Debug)]
struct GroupScales {
    strip_threshold: u8,
    downscale_exponent: u8,
    /// The row's group-B doubled scale exponent frame; define
    /// `F = frame_doubled_scale_exponent = max_live 2*E*(scale_b)`.
    frame_doubled_scale_exponent: u64,
    alpha: Bf16Fields,
    beta: Bf16Fields,
    alpha_code: u16,
    beta_code: u16,
    /// The BF16 code of the device threshold `tau_idle * delta(device) * l2f`.
    dead_bound: u64,
    /// The group's noise-std encoding `enc(sigma) = e(sigma) + 268` for
    /// `sigma = delta(device) * alpha * l2f` (positive on live groups; 0 on dead groups).
    sigma_enc: u64,
    /// The noise-std normalized significand: `sigma = sigma_norm * 2^(sigma_enc - 268 - 15)`
    /// with `sigma_norm in [2^15, 2^16)` on live groups; 0 on dead groups.
    sigma_norm: u64,
}

/// One row's group-B (block-integer L2) witness values — the per-row faces of the running
/// block gadget.
#[derive(Clone, Copy, Debug, Default)]
struct BlockL2Witness {
    /// Running sum of integer squares in the current eight-element block.
    block_int_squared_sum: u64,
    /// Scale-significand square times `block_int_squared_sum`.
    block_l2_product: u64,
    /// Doubled scale exponent candidate on live block-final rows.
    block_doubled_scale_exponent: u64,
    /// Running maximum of the doubled scale exponent candidates.
    running_max_doubled_scale_exponent: u64,
    /// Four limbs of `block_l2_product * 2^Wl2 < 2^54`.
    scaled_block_product_limbs: [u64; 4],
    /// Power of the shift remainder; define `2^r = shift_remainder_power`.
    shift_remainder_power: u64,
    /// Complement power `2^(16-r)`.
    shift_complement_power: u64,
    /// One-hot selector for the limb containing the shift boundary.
    shift_limb_selector: [bool; 4],
    /// Quotient of the selected limb by `2^r`.
    selected_limb_quotient: u64,
    /// Remainder of the selected-limb split.
    selected_limb_remainder: u64,
    /// Live block-final row with a shift of at least 54.
    is_far_shift: bool,
    /// Running framed L2 sum; define its group-final value as `S`.
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
    noise_word: u32,
    coefficient_word: u32,
    mixed_word: u32,
    code_noised: u8,
    is_dead: bool,
    dead_count: u64,
    /// Bit length of the scaled significand product `M(alpha)*M(X)`.
    x_width: u64,
    /// The scaled element's exponent encoding `enc(alpha*x)`, or 0 when `X = 0`.
    x_enc: u64,
    /// The scaled element's normalized significand `M(alpha)*M(X) * 2^(16 - x_width)`,
    /// in `[2^15, 2^16)`, or 0 when `X = 0`.
    x_norm: u64,
    /// The normalizing power `2^(16 - x_width)`.
    x_pow: u64,
    /// The summand's score witness (check 3, reused by B200 check 4), ending in lambda.
    score: LambdaWitness,
}

/// One side's full group witness (k elements + the group scales).
struct SideWitness {
    elements: Vec<ElementWitness>,
    scales: GroupScales,
}

/// One side of one group in the native build: its witness (dead sides share one canonical
/// witness) and, on B200 live sides, `(LOG16 region, key base, per-element (Gaussian value,
/// running sum))`.
pub(crate) struct SideNative {
    witness: Arc<SideWitness>,
    live: bool,
    security: Option<SideSecurity>,
}

/// A B200 live side's Gaussian gate inputs: the LOG16 region, the key base, and each element's
/// `(capped Gaussian value, running sum)`.
type SideSecurity = (u64, u64, Vec<(u64, u64)>);

/// The native InputQuant advice: per-group side witnesses plus the compact per-side outputs
/// the downstream tables consume (live rows only, A then B).
pub(crate) struct InputQuantNative {
    groups: Vec<[SideNative; 2]>,
    /// Matmul's operand codes: the downscale rewrite's output on B200, the noised code else.
    pub(crate) a_codes: Vec<u8>,
    pub(crate) b_codes: Vec<u8>,
    /// Matmul's summand scores.
    pub(crate) a_lambdas: Vec<u64>,
    pub(crate) b_lambdas: Vec<u64>,
    /// Scale's per-matrix-row tuples.
    pub(crate) a_tuples: Vec<ScaleRowTuple>,
    pub(crate) b_tuples: Vec<ScaleRowTuple>,
    /// B200: every live row's final Gaussian security sum, A rows then B rows (empty on H100).
    pub(crate) security_sums: Vec<u64>,
    /// Coefficient's `[alpha code, block scale magnitude code]` per block, A blocks then B.
    pub(crate) coefficient_blocks: Vec<[u16; 2]>,
}

impl SideWitness {
    /// Decodes one matrix row of one side and computes every witness column. `live = false`
    /// builds a dead group's phantom fill: the inputs must be all-zero and the group scales are
    /// the fixed phantom pair `alpha = 2^-126`, `beta = +0` (no native derivation — a dead
    /// group opens no prequant row). The group length is `int8.len()` — `k` for live groups;
    /// the trailing dead group may be shorter (still a multiple of `BLOCK_SIZE`).
    fn build(program: &InputQuantProgram, int8: &[i8], scales: &[u16], noise: &[f32], live: bool) -> Self {
        let k = int8.len();
        debug_assert!(k.is_multiple_of(BLOCK_SIZE), "group length must be whole blocks");
        debug_assert!(!live || k == program.k, "live groups are always full-length");
        debug_assert!(live || int8.iter().all(|&v| v == 0), "dead groups take zero inputs");
        let wl2 = program.wl2();

        // ---- Pass 1: decode X = RNE_bf16(int8 * scale) per element (groups P/U). ----
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

        // ---- Pass 2: compute the canonical prequant L2 frame, decoded max |X|, and row scales.
        // The L2 uses raw int8 squares times exact scale squares; it does not square the
        // already-rounded decoded X values. ----
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
            Fp8E4M3Quant::new(program.device)
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
        // The liveness bound in bf16-code space (`liveness_code_shift` multiplies the
        // floored-L2 code by `tau_idle * delta`, device-dependent). A code past the
        // largest finite abs code
        // (possible when `E*(l2f) >= 253`) marks every entry alive, matching the plaintext:
        // every finite `|X|` is below `4*l2f` there. Dead groups take the floored-zero bound;
        // their flags vanish anyway (X = 0) and C1 pins them to zero.
        let dead_bound = u64::from(bf16_max(l2_code, NORM_FLOOR_CODE)) + program.device.liveness_code_shift();
        // The group's noise-std encoding and normalized significand (checks 3/4). With
        // `l2f` the floored L2 norm, `sigma = delta(device) * alpha * l2f` is exact, and the
        // significand product spans
        // [2^14, 2^16) (both factors are normal), so with `wide = [product >= 2^15]`:
        // `enc(sigma) = e(sigma) + 268 = E(alpha) + E(l2f) + wide + offset(device)` and
        // `sigma_norm = product * 2^(1 - wide)` lands in [2^15, 2^16), where
        // `offset(device)` is `sigma_encoding_offset` (13 on B200, 14 on H100).
        let (sigma_enc, sigma_norm) = if live {
            let l2f = Bf16Fields::from_code(bf16_max(l2_code, NORM_FLOOR_CODE));
            let product = (128 + alpha.mantissa) * l2f.m();
            let wide = u64::from(product >= 1 << 15);
            let enc = alpha.exp + l2f.exp + wide + program.device.sigma_encoding_offset();
            debug_assert_eq!(
                enc as i64 - 268,
                {
                    let sigma = program.device.delta() * f64::from(bf16_to_f32(alpha_code)) * f64::from(bf16_to_f32(l2f.code()));
                    ((sigma.to_bits() >> 52) & 0x7FF) as i64 - 1023
                },
                "sigma encoding diverged from the exact exponent"
            );
            (enc, product << (1 - wide))
        } else {
            (0, 0)
        };
        let (strip_threshold, downscale_exponent) = if live && program.device == Device::B200 {
            let scaled_rms = bf16_to_f32(alpha_code) * bf16_to_f32(bf16_max(l2_code, NORM_FLOOR_CODE));
            assert!(
                scaled_rms >= 4.0 && (269..=274).contains(&sigma_enc),
                "downscale RMS out of domain"
            );
            let factor = if scaled_rms < 8.0 {
                crate::v4::api::utils::DOWNSCALE_G_SMALL_RMS
            } else {
                crate::v4::api::utils::DOWNSCALE_G
            };
            (
                f32_to_fp8_e4m3(factor * scaled_rms).expect("finite strip threshold") & 127,
                (sigma_enc - 269) as u8,
            )
        } else {
            (0, 0)
        };
        let scales_w = GroupScales {
            strip_threshold,
            downscale_exponent,
            frame_doubled_scale_exponent,
            alpha,
            beta,
            alpha_code,
            beta_code,
            dead_bound,
            sigma_enc,
            sigma_norm,
        };

        let coefficients: Vec<f32> = scales
            .iter()
            .map(|&scale| bf16_to_f32(alpha_code) * bf16_to_f32(scale))
            .collect();
        // Coefficients are shared by each eight-element block.
        let mut elements = Vec::with_capacity(k);
        let (mut max_abs, mut dead_count) = (0u64, 0u64);
        for (j, &(byte, int_f, scale_f, dec, x_sign)) in xs.iter().enumerate() {
            let x = dec.out.fields(x_sign);
            max_abs = if j == 0 { x.abs_code() } else { max_abs.max(x.abs_code()) };

            let coefficient = coefficients[j / BLOCK_SIZE];
            assert!(coefficient.is_finite(), "nonfinite block coefficient");
            let signal = coefficient * f32::from(byte as i8);
            let mixed = fp32_fma(bf16_to_f32(scales_w.beta_code), noise[j], signal).expect("finite FP32 fold");
            let code_noised = f32_to_fp8_e4m3(mixed).expect("finite FP8 input");
            let m_p = (128 + scales_w.alpha.mantissa) * x.m();
            let is_dead = x.abs_code() >= scales_w.dead_bound;
            debug_assert_eq!(
                is_dead,
                f64::from(bf16_to_f32(x.code()).abs())
                    >= 8.0 * program.device.delta() * f64::from(bf16_to_f32(bf16_max(l2_code, NORM_FLOOR_CODE))),
                "liveness flag diverged from the plaintext dead predicate"
            );
            dead_count += u64::from(is_dead);

            // The scaled element's exponent encoding and the summand encoding lambda
            // (checks 3/4): `|alpha*x| = M(alpha)*M(X) * 2^(E(alpha) + E*(X) - 268)`, so a
            // nonzero X has `enc(alpha*x) = E(alpha) + E*(X) - 1 + bit_length(M(alpha)*M(X))`.
            let x_width = u64::from(64 - m_p.leading_zeros());
            let x_enc = if x.m() == 0 {
                0
            } else {
                let enc = (scales_w.alpha.exp as i64 + x.e_star() - 1) as u64 + x_width;
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
            let x_pow = 1u64 << (16 - x_width);
            let x_norm = m_p * x_pow;
            // The summand's score witness (checks 3/4). A dead group's phantom rows carry
            // the all-zero witness with the shift power pinned to 2^0; its lambda is 0.
            let score = if live {
                lambda_witness(x_norm, x_enc, scales_w.sigma_norm, scales_w.sigma_enc)
            } else {
                LambdaWitness {
                    x_dominates: false,
                    exponent_gap: 0,
                    gap_is_far: false,
                    near_gap: 0,
                    near_gap_pow: 1,
                    norm_big: 0,
                    norm_small: 0,
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
                noise_word: noise[j].to_bits(),
                coefficient_word: coefficient.to_bits(),
                mixed_word: mixed.to_bits(),
                code_noised,
                is_dead,
                dead_count,
                x_width,
                x_enc,
                x_norm,
                x_pow,
                score,
            });
        }

        SideWitness {
            elements,
            scales: scales_w,
        }
    }

    fn fill<F: RichField>(&self, row: &mut InputQuantSideColumns<F>, j: usize) {
        let e = &self.elements[j];
        row.int8_byte = F::from_canonical_u8(e.byte);
        fill_bf16(&mut row.int_bf16, e.int_f);
        fill_bf16(&mut row.scale, e.scale_f);
        fill_mul_decode(&mut row.decode_multiply, e.dec);
        row.x_sign = F::from_bool(e.x_sign);
        row.x_exp = F::from_canonical_u64(e.dec.out.exp);
        row.x_mantissa = F::from_canonical_u64(e.dec.out.mantissa);
        row.x_exp_is_zero = F::from_bool(e.dec.out.exp_is_zero);
        row.x_is_zero = F::from_bool(e.dec.out.is_zero);
        fill_blk_l2(&mut row.block_l2, e.blk, self.scales.frame_doubled_scale_exponent);
        row.max_abs = F::from_canonical_u64(e.max_abs);
        row.alpha_exp = F::from_canonical_u64(self.scales.alpha.exp);
        row.alpha_mantissa = F::from_canonical_u64(self.scales.alpha.mantissa);
        row.beta_exp = F::from_canonical_u64(self.scales.beta.exp);
        row.beta_mantissa = F::from_canonical_u64(self.scales.beta.mantissa);
        row.beta_exp_is_zero = F::from_bool(self.scales.beta.exp_is_zero());
        row.scaled_decoded_significand =
            F::from_canonical_u64((128 + self.scales.alpha.mantissa) * e.dec.out.fields(e.x_sign).m());
        row.coefficient = Fp32Magnitude::from_word(e.coefficient_word);
        let (alignment, rounding, proxy) = fp32::generate_fold(
            f32::from_bits(e.coefficient_word),
            e.byte as i8,
            self.scales.beta_code,
            f32::from_bits(e.noise_word),
        )
        .expect("finite FP32 certificate");
        row.alignment = alignment;
        row.rounding = rounding;
        row.proxy = proxy;
        row.code_noised = F::from_canonical_u8(e.code_noised);
        row.dead_bound = F::from_canonical_u64(self.scales.dead_bound);
        row.is_dead = F::from_bool(e.is_dead);
        row.dead_count = F::from_canonical_u64(e.dead_count);
        row.x_width = F::from_canonical_u64(e.x_width);
        row.x_enc = F::from_canonical_u64(e.x_enc);
        row.sigma_enc = F::from_canonical_u64(self.scales.sigma_enc);
        row.sigma_norm = F::from_canonical_u64(self.scales.sigma_norm);
        row.x_pow = F::from_canonical_u64(e.x_pow);
        row.x_norm = F::from_canonical_u64(e.x_norm);
        row.x_dominates = F::from_bool(e.score.x_dominates);
        row.exponent_gap = F::from_canonical_u64(e.score.exponent_gap);
        row.gap_is_far = F::from_bool(e.score.gap_is_far);
        row.near_gap = F::from_canonical_u64(e.score.near_gap);
        row.near_gap_pow = F::from_canonical_u64(e.score.near_gap_pow);
        row.norm_big = F::from_canonical_u64(e.score.norm_big);
        row.norm_small = F::from_canonical_u64(e.score.norm_small);
        row.half_quotient = F::from_canonical_u64(e.score.half_quotient);
        row.half_remainder = F::from_canonical_u64(e.score.half_remainder);
        row.quotient = F::from_canonical_u64(e.score.quotient);
        row.remainder = F::from_canonical_u64(e.score.remainder);
        row.sum_of_squares_top = F::from_canonical_u64(e.score.sum_of_squares_top);
        row.sum_of_squares_rest_lo = F::from_canonical_u64(e.score.sum_of_squares_rest & 0xFFFF);
        row.sum_of_squares_rest_hi = F::from_canonical_u64(e.score.sum_of_squares_rest >> 16);
        row.log_fraction = F::from_canonical_u64(e.score.log_fraction);
        row.lambda = F::from_canonical_u64(e.score.lambda);
    }

    /// The B200 downscale decision at `index`: the four-element block's coin and the code
    /// Matmul consumes (the strip rewrite when the coin and threshold select it).
    fn downscale_decision(&self, index: usize, live: bool) -> (bool, u8) {
        let start_word = self.elements[index / 4 * 4].mixed_word;
        let coin = (start_word >> 8 & 127) < 112;
        let code = self.elements[index].code_noised;
        let inside = live && coin && (code & 127) < self.scales.strip_threshold;
        let output = if inside {
            (code & 128) | (self.scales.downscale_exponent * 8) | (code & 7) | 1
        } else {
            code
        };
        (coin, output)
    }

    /// The operand code Matmul consumes at `index`: the downscale rewrite's output on B200,
    /// the noised code on every other device.
    fn output_code(&self, device: Device, index: usize, live: bool) -> u8 {
        match device {
            Device::B200 => self.downscale_decision(index, live).1,
            Device::H100 => self.elements[index].code_noised,
        }
    }

    fn fill_downscale<F: RichField>(&self, output: &mut DownscaleColumns<F>, index: usize, live: bool) {
        let word = self.elements[index].mixed_word;
        let first = index.is_multiple_of(4);
        let (coin, output_code) = self.downscale_decision(index, live);
        *output = DownscaleColumns {
            fraction_bit_15: F::from_canonical_u32(if first { word >> 15 & 1 } else { 0 }),
            coin_field: F::from_canonical_u32(if first { word >> 8 & 127 } else { 0 }),
            coin: F::from_bool(coin),
            strip_threshold: F::from_canonical_u8(self.scales.strip_threshold),
            output_code: F::from_canonical_u8(output_code),
        };
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

// ==================================================================================================
// Constraints, written once against the generic `Evaluator`
// ==================================================================================================

/// One side's column values, copied out of the row view so the A/B constraint logic is written
/// once.
struct SideVals<V: Copy> {
    noise: Fp32Fields<V>,
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
    scaled_decoded_significand: V,
    coefficient: Fp32Magnitude<V>,
    alignment: AlignmentColumns<V>,
    rounding: Round32Columns<V>,
    proxy: OddProxyColumns<V>,
    live: V,
    dead_bound: V,
    is_dead: V,
    dead_count: V,
    x_width: V,
    x_enc: V,
    sigma_enc: V,
    sigma_norm: V,
    x_pow: V,
    x_norm: V,
    x_dominates: V,
    exponent_gap: V,
    gap_is_far: V,
    near_gap: V,
    near_gap_pow: V,
    norm_big: V,
    norm_small: V,
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

fn side_values<V: Copy>(v: &InputQuantSideColumns<V>, noise: Fp32Fields<V>, live: V) -> SideVals<V> {
    SideVals {
        noise,
        int_f: v.int_bf16,
        scale: v.scale,
        mul_decode: v.decode_multiply,
        x_sign: v.x_sign,
        x_exp: v.x_exp,
        x_mantissa: v.x_mantissa,
        x_exp_is_zero: v.x_exp_is_zero,
        x_is_zero: v.x_is_zero,
        int8_byte: v.int8_byte,
        blk: v.block_l2,
        max_abs: v.max_abs,
        alpha_exp: v.alpha_exp,
        alpha_mantissa: v.alpha_mantissa,
        beta_exp: v.beta_exp,
        beta_mantissa: v.beta_mantissa,
        beta_exp_is_zero: v.beta_exp_is_zero,
        scaled_decoded_significand: v.scaled_decoded_significand,
        coefficient: v.coefficient,
        alignment: v.alignment,
        rounding: v.rounding,
        proxy: v.proxy,
        live,
        dead_bound: v.dead_bound,
        is_dead: v.is_dead,
        dead_count: v.dead_count,
        x_width: v.x_width,
        x_enc: v.x_enc,
        sigma_enc: v.sigma_enc,
        sigma_norm: v.sigma_norm,
        x_pow: v.x_pow,
        x_norm: v.x_norm,
        x_dominates: v.x_dominates,
        exponent_gap: v.exponent_gap,
        gap_is_far: v.gap_is_far,
        near_gap: v.near_gap,
        near_gap_pow: v.near_gap_pow,
        norm_big: v.norm_big,
        norm_small: v.norm_small,
        half_quotient: v.half_quotient,
        half_remainder: v.half_remainder,
        quotient: v.quotient,
        remainder: v.remainder,
        sum_of_squares_top: v.sum_of_squares_top,
        sum_of_squares_rest_lo: v.sum_of_squares_rest_lo,
        sum_of_squares_rest_hi: v.sum_of_squares_rest_hi,
        log_fraction: v.log_fraction,
        lambda: v.lambda,
    }
}

/// `M(V) = (1 - EXP_IS_ZERO)*128 + MANTISSA` as an expression (degree 1).
fn m_expr<V: Copy, S: Copy, E: Evaluator<V, S>>(eval: &mut E, exp_is_zero: V, mantissa: V) -> V {
    let c128 = eval.i32(128);
    let one = eval.i32(1);
    let not_eiz = eval.sub(one, exp_is_zero);
    eval.mad(not_eiz, c128, mantissa)
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

    // ---- Group P — bind the raw int8 value and its eight-element block scale. ----
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

    // ---- Group U — prove X = RNE_bf16(int8 * block_scale). ----
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

    // ---- Group S — validate alpha/beta and keep all row-level values group-constant. ----
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

    // ---- Group B — compute the canonical L2 sum from eight-element prequant blocks. ----
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

    // ---- Group F — compute linf's source, the largest decoded |X| in the group. ----
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

    let m_x = m_expr(eval, lv.x_exp_is_zero, lv.x_mantissa);
    let m_alpha = eval.add(c128, lv.alpha_mantissa);
    let policy_product = eval.mul(m_alpha, m_x);
    eval.constraint_eq(lv.scaled_decoded_significand, policy_product);
    let e_star_x = eval.add(lv.x_exp, lv.x_exp_is_zero);
    let twice_sign = eval.add(lv.int_f.sign, lv.int_f.sign);
    let absolute_factor = eval.sub(one, twice_sign);
    let byte_weight = eval.i32(256);
    let negative_offset = eval.mul(byte_weight, lv.int_f.sign);
    let absolute_integer = eval.mad(absolute_factor, lv.int8_byte, negative_offset);
    let magnitude = fp32::evaluate_alignment(
        eval,
        &lv.alignment,
        &lv.coefficient,
        &lv.noise,
        [lv.beta_exp, lv.beta_mantissa, lv.beta_exp_is_zero],
        absolute_integer,
        lv.x_sign,
    );
    fp32::evaluate_rounding(
        eval,
        &lv.rounding,
        magnitude,
        lv.alignment.has_tail,
        lv.alignment.tail_negative,
    );
    fp32::evaluate_proxy(eval, &lv.proxy, &lv.rounding.output, lv.alignment.output_sign);
    let not_block_start = eval.sub(one, next_block_start);
    let not_live = eval.sub(one, lv.live);
    for (current, next, dummy) in [
        (lv.coefficient.exponent, nv.coefficient.exponent, 0),
        (lv.coefficient.fraction_low, nv.coefficient.fraction_low, 0),
        (lv.coefficient.fraction_high, nv.coefficient.fraction_high, 0),
        (lv.coefficient.is_exponent_zero, nv.coefficient.is_exponent_zero, 1),
    ] {
        eval.constraint_eq_if(not_block_start, next, current);
        let dummy = eval.i32(dummy);
        eval.constraint_eq_if(not_live, current, dummy);
    }

    // ---- Group C — the jackpot liveness flag `IS_DEAD = [ABS(X) >= DEAD_BOUND]` and its
    // in-group count (the group-final value rides ScaleStark's group-tuple CTL channel). ----
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

    // ---- Group V — the element's summand score LAMBDA (check 3, reused by B200 check 4), exported to
    // Matmul's per-lane skip certificates by the operand-code channel. Encodings write
    // `enc(v) = e(v) + 268` with `e(v) = floor(log2 v)`, and 0 encodes a zero value. Both
    // addends are written over 16-bit normalized significands,
    // `|alpha*X| = X_NORM * 2^(X_ENC - 268 - 15)` and `sigma = SIGMA_NORM * 2^(SIGMA_ENC - 268 - 15)`,
    // and the score of `S = (alpha*X)^2 + sigma^2` is
    //
    //   gap    = |X_ENC - SIGMA_ENC|
    //   V      = big^2 + floor(floor(small^2 / 2^gap) / 2^gap)   if gap <= 15, else big^2
    //   LAMBDA = 128*max(X_ENC, SIGMA_ENC) - 2624 + floor(64 * log2(floor(V / 2^17)))
    //
    // with (big, small) the two significands ordered by exponent. Every step is exact
    // integer arithmetic. ----
    // V1: bind ENC(alpha*X) from the product width. The WIDTH16 lookup (ctl.rs) returns
    // X_WIDTH = bit_length(M(alpha)*M(X)) and pins the product-nonzero flag to 1 - X_IS_ZERO.
    // For nonzero X, `|alpha*X| = M(alpha)*M(X) * 2^(ALPHA_EXP + E*(X) - 268)`, so
    // `enc(alpha*X) = ALPHA_EXP + E*(X) - 1 + X_WIDTH`; a zero X encodes to 0 (X_WIDTH = 0).
    let not_xz = eval.sub(one, lv.x_is_zero);
    let enc_base = eval.add(lv.alpha_exp, e_star_x);
    let enc_base = eval.sub(enc_base, one);
    let enc_gated = eval.mul(not_xz, enc_base);
    let x_enc_target = eval.add(enc_gated, lv.x_width);
    let c = eval.sub(lv.x_enc, x_enc_target);
    eval.constraint(c);
    // V2 and V3: SIGMA_ENC and SIGMA_NORM are group-constant; a live group's group-final
    // values are bound to ScaleStark's exact sigma encoding and significand by the
    // group-tuple channel. (Dead groups' scores never leave the table: the operand-code
    // channel is liveness-filtered.)
    let sig_diff = eval.sub(nv.sigma_enc, lv.sigma_enc);
    let c = eval.mul(not_gf, sig_diff);
    eval.constraint(c);
    let sig_norm_diff = eval.sub(nv.sigma_norm, lv.sigma_norm);
    let c = eval.mul(not_gf, sig_norm_diff);
    eval.constraint(c);
    // V4: X_NORM = M(alpha)*M(X) * 2^(16 - X_WIDTH) — in [2^15, 2^16) for nonzero X, else 0.
    // The POW2D lookup at the affine key 16 - X_WIDTH (ctl.rs) pins X_POW = 2^(16 - X_WIDTH).
    let norm_target = eval.mul(lv.scaled_decoded_significand, lv.x_pow);
    let c = eval.sub(lv.x_norm, norm_target);
    eval.constraint(c);
    // V5: X_DOMINATES picks the larger addend. The gap identity plus RC16(EXPONENT_GAP)
    // (ctl.rs) force the bit whenever the encodings differ: the wrong bit makes the gap
    // negative, wrapping its range-check key. A zero X has X_ENC = 0 < SIGMA_ENC, so its bit
    // is forced to 0 and the gap equals SIGMA_ENC (far). On a tie either bit gives the same
    // score, so the freedom is inert.
    eval.constraint_bool(lv.x_dominates);
    let two_b = eval.add(lv.x_dominates, lv.x_dominates);
    let sign = eval.sub(two_b, one);
    let enc_diff = eval.sub(lv.x_enc, lv.sigma_enc);
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
    // V8: order the two normalized significands by exponent.
    //   NORM_BIG   = X_DOMINATES * X_NORM + (1 - X_DOMINATES) * SIGMA_NORM
    //   NORM_SMALL = X_DOMINATES * SIGMA_NORM + (1 - X_DOMINATES) * X_NORM
    let big_target = eval.mux(lv.x_dominates, lv.sigma_norm, lv.x_norm);
    let c = eval.sub(lv.norm_big, big_target);
    eval.constraint(c);
    let small_target = eval.mux(lv.x_dominates, lv.x_norm, lv.sigma_norm);
    let c = eval.sub(lv.norm_small, small_target);
    eval.constraint(c);
    // V9: the two exact floor stages, dividing NORM_SMALL^2 by 2^NEAR_GAP twice:
    //   NORM_SMALL^2  = HALF_QUOTIENT * NEAR_GAP_POW + HALF_REMAINDER
    //   HALF_QUOTIENT = QUOTIENT * NEAR_GAP_POW + REMAINDER
    // Each remainder is pinned into [0, NEAR_GAP_POW) by its RC16 pair (ctl.rs), making both
    // quotients exact floors. No identity can wrap the field: V10 keeps QUOTIENT's signed
    // value inside (-2^32, 2^33 + 2^17), so HALF_QUOTIENT stays inside (-2^47, 2^49), each
    // product inside (-2^62, 2^63 + 2^48), and both identities equate values whose
    // difference lies inside (-p, p) — integer equality follows.
    let small_sq = eval.mul(lv.norm_small, lv.norm_small);
    let stage1 = eval.mad(lv.half_quotient, lv.near_gap_pow, lv.half_remainder);
    let c = eval.sub(small_sq, stage1);
    eval.constraint(c);
    let stage2 = eval.mad(lv.quotient, lv.near_gap_pow, lv.remainder);
    let c = eval.sub(lv.half_quotient, stage2);
    eval.constraint(c);
    // V10: split the sum of squares V into its top 16 bits and the 17-bit rest:
    //   V = NORM_BIG^2 + (1 - GAP_IS_FAR) * QUOTIENT
    //   V = SUM_OF_SQUARES_TOP * 2^17 + SUM_OF_SQUARES_REST_LO + 2^16 * SUM_OF_SQUARES_REST_HI
    // The rest is below 2^17 (one RC16 plus one boolean bit), so the split is the exact
    // division of V by 2^17. The LOG16 lookup at SUM_OF_SQUARES_TOP (ctl.rs) bounds the top
    // below 2^16 and serves LOG_FRACTION = floor(64 * log2 SUM_OF_SQUARES_TOP).
    eval.constraint_bool(lv.sum_of_squares_rest_hi);
    let big_sq = eval.mul(lv.norm_big, lv.norm_big);
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
    // V11: the summand score, gated by row liveness (phantom rows score 0):
    //   LAMBDA = LIVE * (128*ENC_BIG - 2624 + LOG_FRACTION)
    // where ENC_BIG = X_DOMINATES*X_ENC + (1 - X_DOMINATES)*SIGMA_ENC = max(X_ENC, SIGMA_ENC)
    // (V5 orders the dominance bit by the encodings; a tie makes both mux branches equal).
    // The score is 64*(log2 S + 508) rounded down, for S = (alpha*x)^2 + sigma^2:
    //   S ~= V * 2^(2*(ENC_BIG - 268) - 30)   (each addend is norm * 2^(enc - 268 - 15))
    //   LOG_FRACTION ~= 64*(log2 V - 17)      (SUM_OF_SQUARES_TOP = V >> 17)
    //   -2624 = 64*(508 - 13) - 128*268       (-13 = -(30 - 17): V is built 30 bits up but
    //                                          read back only 17 bits down; -128*268 strips
    //                                          the two encoding biases from 128*ENC_BIG)
    // Each ~= is a floor, so LAMBDA is never above 64*(log2 S + 508) and sits less than
    // 1.1 steps of 1/64 below it.
    let enc_big = eval.mux(lv.x_dominates, lv.sigma_enc, lv.x_enc);
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
fn evaluate_sides<V: Copy + Default, S: Copy + Default, E: Evaluator<V, S>>(
    evaluator: &mut E,
    schedule: &InputQuantScheduleColumns<V>,
    next_schedule: &InputQuantScheduleColumns<V>,
    sides: [&InputQuantSideColumns<V>; 2],
    next_sides: [&InputQuantSideColumns<V>; 2],
    precision: V,
) {
    for (index, (current, next)) in sides.into_iter().zip(next_sides).enumerate() {
        let (noise, live, next_noise, next_live) = if index == 0 {
            (schedule.noise_a, schedule.a_live, next_schedule.noise_a, next_schedule.a_live)
        } else {
            (schedule.noise_b, schedule.b_live, next_schedule.noise_b, next_schedule.b_live)
        };
        eval_side(
            evaluator,
            &side_values(current, noise, live),
            &side_values(next, next_noise, next_live),
            schedule.is_group_final,
            next_schedule.is_block_start,
            precision,
        );
    }
}

/// The geometry scalars group G reads from the public inputs.
#[derive(Clone, Copy)]
struct ScheduleGeometry<V> {
    /// `k`, the group length.
    k: V,
    /// `h*k`, the A-liveness bound.
    a_bound: V,
    /// `w*k`, the B-liveness bound.
    b_bound: V,
}

/// Group G: pins the noise decodes to the verifier-known words and the schedule to the row
/// counter and the geometry public inputs, so that only the two noise words need verifier
/// recomputation. Every constraint is at most cubic.
fn evaluate_schedule<V: Copy, S: Copy, E: Evaluator<V, S>>(
    eval: &mut E,
    known: &InputQuantKnownColumns<V>,
    lv: &InputQuantScheduleColumns<V>,
    nv: &InputQuantScheduleColumns<V>,
    geometry: ScheduleGeometry<V>,
) {
    let one = eval.i32(1);

    // G1/G2: WORD = 2^31*SIGN + 2^23*EXPONENT + FRACTION with a boolean SIGN. The unfiltered
    // lookups in `super::ctl` give EXPONENT <= 254 with IS_EXPONENT_ZERO = [EXPONENT = 0]
    // (EXPINFO) and FRACTION = 2^16*HIGH + LOW with HIGH < 128, LOW < 2^16 (PAIR128, RC16).
    // The sum is below 2^32 < p, so the decode of the known word is unique on live and
    // phantom rows alike.
    let sign_weight = eval.u64(1 << 31);
    let exponent_weight = eval.u64(1 << 23);
    for (word, fields) in [(known.noise_word_a, &lv.noise_a), (known.noise_word_b, &lv.noise_b)] {
        let signed = eval.mul(fields.sign, sign_weight);
        let biased = eval.mad(fields.exponent, exponent_weight, signed);
        let recomposed = eval.add(biased, fields.fraction);
        eval.constraint_eq(word, recomposed);
        eval.constraint_bool(fields.sign);
    }

    // G3/G4: the flag/gap product and inverse enforce IS_GROUP_FINAL = [GROUP_COUNTER = k-1]
    // off the last row; the last row is always group-final, closing any truncated tail. The
    // cyclic counter resets after each final row, including the last -> first wrap.
    eval.constraint_first_row(lv.group_counter);
    let k_minus_one = eval.sub(geometry.k, one);
    let counter_gap = eval.sub(lv.group_counter, k_minus_one);
    let final_off_boundary = eval.mul(lv.is_group_final, counter_gap);
    eval.constraint_transition(final_off_boundary);
    let not_final = eval.sub(one, lv.is_group_final);
    let gap_inverse_hit = eval.mul(counter_gap, lv.group_final_inv);
    let gap_inverse_miss = eval.sub(gap_inverse_hit, one);
    let forced_final = eval.mul(not_final, gap_inverse_miss);
    eval.constraint(forced_final);
    eval.constraint_last_row(not_final);
    let counter_next = eval.add(lv.group_counter, one);
    let counter_step = eval.mul(counter_next, not_final);
    eval.constraint_eq(nv.group_counter, counter_step);

    // G6: ELEMENT_INDEX is the row counter.
    eval.constraint_first_row(lv.element_index);
    let index_step = eval.sub(nv.element_index, lv.element_index);
    let index_walk = eval.sub(index_step, one);
    eval.constraint_transition(index_walk);

    // G7: IS_EVEN_ROW alternates from 1, including the wrap pair (the trace height is even).
    let even_minus_one = eval.sub(lv.is_even_row, one);
    eval.constraint_first_row(even_minus_one);
    let parity_sum = eval.add(nv.is_even_row, lv.is_even_row);
    let parity_alternates = eval.sub(parity_sum, one);
    eval.constraint(parity_alternates);

    // G8-G10: the product and inverse enforce next.IS_BLOCK_START = [BLOCK8_COUNTER = 7], so
    // the counter walks modulo 8 from 0 without a separate range check; the cyclic constraints
    // close at the wrap because 8 divides the trace height.
    eval.constraint_first_row(lv.block8_counter);
    let block_start_minus_one = eval.sub(lv.is_block_start, one);
    eval.constraint_first_row(block_start_minus_one);
    let block_final_position = eval.i32(BLOCK_SIZE as i32 - 1);
    let block_gap = eval.sub(lv.block8_counter, block_final_position);
    let start_off_boundary = eval.mul(nv.is_block_start, block_gap);
    eval.constraint(start_off_boundary);
    let not_next_start = eval.sub(one, nv.is_block_start);
    let block_inverse_hit = eval.mul(block_gap, lv.block8_counter_inv);
    let block_inverse_miss = eval.sub(block_inverse_hit, one);
    let forced_start = eval.mul(not_next_start, block_inverse_miss);
    eval.constraint(forced_start);
    let block_step = eval.sub(nv.block8_counter, lv.block8_counter);
    let block_step_minus_one = eval.sub(block_step, one);
    let block_modulus = eval.i32(BLOCK_SIZE as i32);
    let block_reset = eval.mul(nv.is_block_start, block_modulus);
    let block_walk = eval.add(block_step_minus_one, block_reset);
    eval.constraint(block_walk);

    // G11: BLOCK_INDEX increments exactly at block starts.
    eval.constraint_first_row(lv.block_index);
    let block_index_step = eval.sub(nv.block_index, lv.block_index);
    let block_index_walk = eval.sub(block_index_step, nv.is_block_start);
    eval.constraint_transition(block_index_walk);

    // G12: boolean liveness starts at 1, can change only on entering its bound, and is forced
    // to 0 there by the inverse witness, so LIVE = [ELEMENT_INDEX < bound]. When the bound
    // equals the trace height no flip row exists and the side stays live throughout.
    for (live, next_live, bound, flip_inv) in [
        (lv.a_live, nv.a_live, geometry.a_bound, lv.a_flip_inv),
        (lv.b_live, nv.b_live, geometry.b_bound, lv.b_flip_inv),
    ] {
        let live_minus_one = eval.sub(live, one);
        eval.constraint_first_row(live_minus_one);
        eval.constraint_bool(live);
        let live_step = eval.sub(live, next_live);
        let next_gap = eval.sub(nv.element_index, bound);
        let flip_only_at_bound = eval.mul(live_step, next_gap);
        eval.constraint_transition(flip_only_at_bound);
        let gap = eval.sub(lv.element_index, bound);
        let flip_inverse_hit = eval.mul(gap, flip_inv);
        let flip_inverse_miss = eval.sub(flip_inverse_hit, one);
        let forced_drop = eval.mul(live, flip_inverse_miss);
        eval.constraint(forced_drop);
    }
}

/// G13 (B200): IS_COIN_START = IS_BLOCK_START + [BLOCK8_COUNTER = 4], the first row of each
/// four-element coin block; the inverse witness certifies the half-block test.
fn evaluate_coin_start<V: Copy, S: Copy, E: Evaluator<V, S>>(
    eval: &mut E,
    schedule: &InputQuantScheduleColumns<V>,
    is_coin_start: V,
    coin_inv: V,
) {
    let one = eval.i32(1);
    let half_block = eval.i32(4);
    let gap = eval.sub(schedule.block8_counter, half_block);
    let inverse_hit = eval.mul(gap, coin_inv);
    let inverse_miss = eval.sub(inverse_hit, one);
    // Either the counter is 4 or the inverse exists (cubic).
    let certified = eval.mul(gap, inverse_miss);
    eval.constraint(certified);
    // `1 - gap*inv` is then exactly [BLOCK8_COUNTER = 4].
    let expected = eval.sub(schedule.is_block_start, inverse_miss);
    eval.constraint_eq(is_coin_start, expected);
}

fn evaluate_downscale<V: Copy, S: Copy, E: Evaluator<V, S>>(
    evaluator: &mut E,
    current: &DownscaleColumns<V>,
    next: &DownscaleColumns<V>,
    first: V,
    next_first: V,
    group_final: V,
) {
    let one = evaluator.i32(1);
    let zero = evaluator.i32(0);
    evaluator.constraint_bool(current.fraction_bit_15);
    let not_first = evaluator.sub(one, first);
    evaluator.constraint_eq_if(not_first, current.fraction_bit_15, zero);
    evaluator.constraint_eq_if(not_first, current.coin_field, zero);
    evaluator.constraint_bool(current.coin);
    let not_next_first = evaluator.sub(one, next_first);
    evaluator.constraint_eq_if(not_next_first, current.coin, next.coin);
    let not_final = evaluator.sub(one, group_final);
    evaluator.constraint_eq_if(not_final, current.strip_threshold, next.strip_threshold);
}

pub(crate) fn eval_input_quant_constraints<V, S, E, const WIDTH: usize>(
    frame: &StarkFrame<V, S, WIDTH, NUM_INPUT_QUANT_PUBLIC_INPUTS>,
    evaluator: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    let public_inputs = frame.get_public_inputs();
    let precision = evaluator.scalar(public_inputs[WL2_POW_PUBLIC_INPUT]);
    let geometry = ScheduleGeometry {
        k: evaluator.scalar(public_inputs[K_PUBLIC_INPUT]),
        a_bound: evaluator.scalar(public_inputs[B_KEY_OFFSET_PUBLIC_INPUT]),
        b_bound: evaluator.scalar(public_inputs[B_LIVE_BOUND_PUBLIC_INPUT]),
    };
    match WIDTH {
        NUM_INPUT_QUANT_COLUMNS => {
            let current: &[V; NUM_INPUT_QUANT_COLUMNS] = frame.get_local_values().try_into().unwrap();
            let next: &[V; NUM_INPUT_QUANT_COLUMNS] = frame.get_next_values().try_into().unwrap();
            let current: &InputQuantColumnsView<V> = current.borrow();
            let next: &InputQuantColumnsView<V> = next.borrow();
            evaluate_schedule(evaluator, &current.known, &current.schedule, &next.schedule, geometry);
            evaluate_sides(
                evaluator,
                &current.schedule,
                &next.schedule,
                [&current.a, &current.b],
                [&next.a, &next.b],
                precision,
            );
        }
        NUM_B200_INPUT_QUANT_COLUMNS => {
            let current: &[V; NUM_B200_INPUT_QUANT_COLUMNS] = frame.get_local_values().try_into().unwrap();
            let next: &[V; NUM_B200_INPUT_QUANT_COLUMNS] = frame.get_next_values().try_into().unwrap();
            let current: &B200InputQuantColumnsView<V> = current.borrow();
            let next: &B200InputQuantColumnsView<V> = next.borrow();
            evaluate_schedule(evaluator, &current.known, &current.schedule, &next.schedule, geometry);
            evaluate_coin_start(evaluator, &current.schedule, current.is_coin_start, current.coin_inv);
            evaluate_sides(
                evaluator,
                &current.schedule,
                &next.schedule,
                [&current.a.common, &current.b.common],
                [&next.a.common, &next.b.common],
                precision,
            );
            for ((side, following), live) in [(&current.a, &next.a), (&current.b, &next.b)]
                .into_iter()
                .zip([current.schedule.a_live, current.schedule.b_live])
            {
                let one = evaluator.i32(1);
                let zero = evaluator.i32(0);
                let phantom = evaluator.sub(one, live);
                for value in [
                    side.downscale.strip_threshold,
                    side.security_region,
                    side.security_key_base,
                    side.gaussian_value,
                    side.security_sum,
                ] {
                    evaluator.constraint_eq_if(phantom, value, zero);
                }
                let not_final = evaluator.sub(one, current.schedule.is_group_final);
                evaluator.constraint_eq_if(not_final, side.security_region, following.security_region);
                evaluator.constraint_eq_if(not_final, side.security_key_base, following.security_key_base);
                let initial = evaluator.sub(side.security_sum, side.gaussian_value);
                evaluator.constraint_first_row(initial);
                let carried = evaluator.mul(not_final, side.security_sum);
                let next_sum = evaluator.add(carried, following.gaussian_value);
                evaluator.constraint_eq(following.security_sum, next_sum);
                evaluate_downscale(
                    evaluator,
                    &side.downscale,
                    &following.downscale,
                    current.is_coin_start,
                    next.is_coin_start,
                    current.schedule.is_group_final,
                );
            }
        }
        _ => panic!("unsupported input width"),
    }
}

// ==================================================================================================
// Stark impl
// ==================================================================================================

/// InputQuantStark. A CTL party of the fp8 batch (`requires_ctls()`): its proofs carry the
/// cross-table openings of the channels declared in `super::ctl`, so the batch driver is the
/// only supported proving path. There are no in-trace lookups (`lookups()` stays empty): every
/// lookup instance targets the committed LUT oracle.
#[derive(Clone, Debug)]
pub struct InputQuantStark<F: RichField + Extendable<D>, const D: usize, const WIDTH: usize = NUM_INPUT_QUANT_COLUMNS> {
    pub program: InputQuantProgram,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize, const WIDTH: usize> InputQuantStark<F, D, WIDTH> {
    pub fn new(program: InputQuantProgram) -> Self {
        assert_eq!(program.r, 32, "noise rank must be exactly 32");
        assert!(
            matches!(
                (program.device, WIDTH),
                (Device::H100, NUM_INPUT_QUANT_COLUMNS) | (Device::B200, NUM_B200_INPUT_QUANT_COLUMNS)
            ),
            "input AIR width differs from device"
        );
        Self {
            program,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize, const WIDTH: usize> Stark<F, D> for InputQuantStark<F, D, WIDTH> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, WIDTH, NUM_INPUT_QUANT_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget = StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, WIDTH, NUM_INPUT_QUANT_PUBLIC_INPUTS>;

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

// ==================================================================================================
// Tests
// ==================================================================================================

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::plonk::config::PoseidonGoldilocksConfig;
    use starky::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};
    use starky::util::trace_rows_to_poly_values;

    use super::super::columns::{INPUT_QUANT_COL_MAP, NUM_INPUT_QUANT_KNOWN_COLUMNS};
    use super::super::ctl::input_quant_lut_lookups;
    use super::*;
    use crate::v4::api::prequant::open_prequant;
    use crate::v4::api::public_params::PublicParams;
    use crate::v4::circuit::ctl::{LutLookup, LutTable};
    use crate::v4::circuit::unpredictability::log2_fixed;

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
            r: 32,
            device: Device::H100,
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
    fn test_noise(len: usize, salt: u64) -> Vec<f32> {
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
            .map(bf16_to_f32)
            .collect()
    }

    struct TestInputs {
        a_int8: Vec<i8>,
        a_scales: Vec<u16>,
        a_noise: Vec<f32>,
        b_int8: Vec<i8>,
        b_scales: Vec<u16>,
        b_noise: Vec<f32>,
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
        let (rows, pis) = program.generate_trace::<F, NUM_INPUT_QUANT_COLUMNS>(
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
    fn constraints_vanish<const WIDTH: usize>(
        stark: &InputQuantStark<F, D, WIDTH>,
        rows: &[[F; WIDTH]],
        pis: &[F; NUM_INPUT_QUANT_PUBLIC_INPUTS],
    ) -> bool {
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

    /// The native build's outputs equal the trace read-backs the driver used to do, and the
    /// direct column fill reproduces every cell of `generate_trace` except the four row-0
    /// blinding cells (fresh randomness on both sides of the comparison).
    #[test]
    fn native_build_and_column_fill_match_generate_trace() {
        fn check<const WIDTH: usize>(program: &InputQuantProgram) {
            let inp = test_inputs(program);
            let (rows, _) = program.generate_trace::<F, WIDTH>(
                &inp.a_int8,
                &inp.a_scales,
                &inp.a_noise,
                &inp.b_int8,
                &inp.b_scales,
                &inp.b_noise,
            );
            let native = program.build_native(
                &inp.a_int8,
                &inp.a_scales,
                &inp.a_noise,
                &inp.b_int8,
                &inp.b_scales,
                &inp.b_noise,
            );
            let (h, w, k) = (program.h, program.w, program.k);
            let map = input_quant_column_map(program.device);
            let read = |column: usize, row: usize| to_u64(rows[row][column]);
            let code_column = |side: usize| {
                map.downscale
                    .map_or(map.sides[side].code_noised, |(_, sides)| sides[side].output_code)
            };
            assert_eq!(
                native.a_codes,
                (0..h * k).map(|r| read(code_column(0), r) as u8).collect::<Vec<_>>()
            );
            assert_eq!(
                native.b_codes,
                (0..w * k).map(|r| read(code_column(1), r) as u8).collect::<Vec<_>>()
            );
            assert_eq!(
                native.a_lambdas,
                (0..h * k).map(|r| read(map.sides[0].lambda, r)).collect::<Vec<_>>()
            );
            assert_eq!(
                native.b_lambdas,
                (0..w * k).map(|r| read(map.sides[1].lambda, r)).collect::<Vec<_>>()
            );
            for (side, count, tuples) in [(0, h, &native.a_tuples), (1, w, &native.b_tuples)] {
                let columns = &map.sides[side];
                assert_eq!(tuples.len(), count);
                for (g, tuple) in tuples.iter().enumerate() {
                    let row = g * k + k - 1;
                    assert_eq!(tuple.l2_frame_sum, read(columns.block_l2.running_l2_frame_sum, row));
                    assert_eq!(
                        u64::from(tuple.frame_doubled_scale_exponent),
                        read(columns.block_l2.frame_doubled_scale_exponent, row)
                    );
                    assert_eq!(u64::from(tuple.max_abs), read(columns.max_abs, row));
                    assert_eq!(u64::from(tuple.dead_count), read(columns.dead_count, row));
                }
            }
            let mut blocks = Vec::new();
            for (side, count) in [(0, h), (1, w)] {
                let columns = &map.sides[side];
                for row in (0..count * k).step_by(BLOCK_SIZE) {
                    blocks.push([
                        (128 * read(columns.alpha_exp, row) + read(columns.alpha_mantissa, row)) as u16,
                        (128 * read(columns.scale.exp, row) + read(columns.scale.mantissa, row)) as u16,
                    ]);
                }
            }
            assert_eq!(native.coefficient_blocks, blocks);
            if program.device == Device::B200 {
                let b200 = &super::super::columns::B200_INPUT_QUANT_COL_MAP;
                let sums: Vec<u64> = [(&b200.a, h), (&b200.b, w)]
                    .into_iter()
                    .flat_map(|(side, count)| (0..count).map(move |g| read(side.security_sum, g * k + k - 1)))
                    .collect();
                assert_eq!(native.security_sums, sums);
            } else {
                assert!(native.security_sums.is_empty());
            }
            let num_rows = program.num_rows();
            let mut cols: Vec<Vec<F>> = (0..WIDTH).map(|_| vec![F::ZERO; num_rows]).collect();
            {
                let mut slices: Vec<&mut [F]> = cols.iter_mut().map(|c| c.as_mut_slice()).collect();
                program.fill_columns::<F, WIDTH>(&native, &mut slices);
            }
            let blinded: Vec<usize> = map
                .sides
                .iter()
                .flat_map(|side| [side.block_l2.shift_remainder_power, side.block_l2.shift_complement_power])
                .collect();
            for (c, col) in cols.iter().enumerate() {
                for r in 0..num_rows {
                    if r == 0 && blinded.contains(&c) {
                        continue;
                    }
                    assert_eq!(col[r], rows[r][c], "column {c} row {r} differs from generate_trace");
                }
            }
        }
        check::<NUM_INPUT_QUANT_COLUMNS>(&test_program());
        check::<NUM_INPUT_QUANT_COLUMNS>(&InputQuantProgram {
            h: 3,
            w: 5,
            k: 1056,
            ..test_program()
        });
        check::<NUM_B200_INPUT_QUANT_COLUMNS>(&InputQuantProgram {
            h: 3,
            w: 5,
            k: 1056,
            device: Device::B200,
            ..test_program()
        });
    }

    #[test]
    fn honest_trace_satisfies_all_constraints() {
        let (program, rows, pis) = test_trace();
        let stark = S::new(program);
        let n = rows.len();
        for i in 0..n {
            // Ordinary constraints also hold on the last-row-to-first-row wrap. G4 marks the
            // last row group-final, so this pair correctly resets the group-0 accumulators and
            // running maximum.
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

        // The lower protocol boundary reaches Wl2 = 17, one bit wider than the old
        // envelope. Exercise it in this existing honest-trace test.
        {
            let device = Device::H100;
            let program = InputQuantProgram {
                h: 1,
                w: 1,
                k: PublicParams::MIN_K,
                block_size: BLOCK_SIZE,
                r: 32,
                device,
            };
            assert_eq!(program.wl2(), 17);
            let inp = test_inputs(&program);
            let (rows, pis) = program.generate_trace::<F, NUM_INPUT_QUANT_COLUMNS>(
                &inp.a_int8,
                &inp.a_scales,
                &inp.a_noise,
                &inp.b_int8,
                &inp.b_scales,
                &inp.b_noise,
            );
            assert_eq!(pis[WL2_POW_PUBLIC_INPUT].to_canonical_u64(), 1 << 17);
            assert!(constraints_vanish(&S::new(program), &rows, &pis));
        }
    }

    #[test]
    fn padded_asymmetric_trace_satisfies_constraints_and_lut_domains() {
        // The padded geometry in one program: h != w (dead A rows [288, 512), dead B rows
        // [480, 512)) and a truncated tail group (k = 96 is not a power of two: 512 = 5*96 +
        // 32, and the 32-row tail still closes with IS_GROUP_FINAL on the last row). The
        // phantom fill must satisfy every constraint—including cyclic wraps through the dead
        // tail—keep active lookup keys in range, and match the verifier-known columns.
        let program = InputQuantProgram {
            h: 3,
            w: 5,
            k: 96,
            block_size: BLOCK_SIZE,
            r: 32,
            device: Device::H100,
        };
        let inp = test_inputs(&program);
        let (rows, pis) = program.generate_trace::<F, NUM_INPUT_QUANT_COLUMNS>(
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

        // Both verifier-known noise-word columns match independent recomputation.
        let known = program.known_values::<F>(&inp.a_noise, &inp.b_noise);
        assert_eq!(known.len(), NUM_INPUT_QUANT_KNOWN_COLUMNS);
        for (c, poly) in known.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(row[c], poly.values[r], "known column {c} differs at row {r}");
            }
        }
        let m = &INPUT_QUANT_COL_MAP;
        for (side, bound, live_column) in [
            ("A", program.h * program.k, m.schedule.a_live),
            ("B", program.w * program.k, m.schedule.b_live),
        ] {
            assert!(bound > 0 && bound < rows.len(), "{side}: fixture must cross the live bound");
            assert_eq!(rows[bound - 1][live_column], F::ONE);
            assert_eq!(rows[bound][live_column], F::ZERO);
            // G12 forces LIVE = 0 at its bound, where the inverse gap is zero.
            let mut extended = rows.clone();
            extended[bound][live_column] = F::ONE;
            assert!(
                !constraints_vanish(&stark, &extended, &pis),
                "{side}: liveness must end at its public bound"
            );
            // ...and cannot be revived once dropped.
            let mut revived = rows.clone();
            revived[bound + 1][live_column] = F::ONE;
            assert!(
                !constraints_vanish(&stark, &revived, &pis),
                "{side}: liveness must stay zero past its bound"
            );
        }
        assert_eq!(rows[511][m.schedule.is_group_final], F::ONE, "the truncated tail must close");

        // Every declared LUT instance stays in its table's domain on the padded trace (the
        // phantom fill's derived fields and keys must be as legal as live ones — unfiltered
        // instances run on dead rows too).
        let polys = trace_rows_to_poly_values(rows.clone());
        let lookups = input_quant_lut_lookups::<F>(Device::H100);
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
        // X vs open_prequant, then the noised codes and the liveness flags vs the native
        // functions, given the trace's alpha/beta. The decode follows the canonical
        // `open_prequant` semantics (the plain verifier has not wired prequant openings yet).
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
                Fp8E4M3Quant::new(program.device)
                    .row_norms(&opened[g * k..(g + 1) * k])
                    .unwrap()
                    .0,
                NORM_FLOOR_CODE,
            )
        };
        // The plaintext dead bound and predicate, exactly as `JackpotPolicy::liveness_ok`
        // (`tau_idle = 8`, `dead iff |X| >= tau_idle * DELTA * l2f`, all in f64).
        let native_dead = |opened: &[u16], g: usize, x_code: u16| {
            f64::from(bf16_to_f32(x_code).abs()) >= 8.0 * program.device.delta() * f64::from(bf16_to_f32(l2f(opened, g)))
        };

        for (r, row) in rows.iter().enumerate() {
            let v: &InputQuantColumnsView<F> = row.borrow();
            let g = r / k;
            for (side, opened, noise) in [
                (
                    &side_values(&v.a, v.schedule.noise_a, v.schedule.a_live),
                    &opened_a,
                    &inp.a_noise,
                ),
                (
                    &side_values(&v.b, v.schedule.noise_b, v.schedule.b_live),
                    &opened_b,
                    &inp.b_noise,
                ),
            ] {
                let x_code = (to_u64(side.x_sign) << 15 | to_u64(side.x_exp) << 7 | to_u64(side.x_mantissa)) as u16;
                assert_eq!(x_code, opened[r], "row {r}: X differs from open_prequant");
                let alpha_code = (to_u64(side.alpha_exp) << 7 | to_u64(side.alpha_mantissa)) as u16;
                let beta_code = (to_u64(side.beta_exp) << 7 | to_u64(side.beta_mantissa)) as u16;
                let is_a = std::ptr::eq(opened, &opened_a);
                let integer = if is_a { inp.a_int8[r] } else { inp.b_int8[r] };
                let scale = if is_a { inp.a_scales[r / 8] } else { inp.b_scales[r / 8] };
                let coefficient = bf16_to_f32(alpha_code) * bf16_to_f32(scale);
                let expected_noised =
                    f32_to_fp8_e4m3(bf16_to_f32(beta_code).mul_add(noise[r], coefficient * f32::from(integer))).unwrap();
                let got_noised = if std::ptr::eq(opened, &opened_a) {
                    to_u64(v.a.code_noised)
                } else {
                    to_u64(v.b.code_noised)
                } as u8;
                assert_eq!(got_noised, expected_noised, "row {r}: noised fp8 code differs from native");
                // Liveness: the committed bound uses the device code shift and the flag
                // matches the plaintext f64 predicate.
                assert_eq!(
                    to_u64(side.dead_bound),
                    u64::from(l2f(opened, g)) + program.device.liveness_code_shift(),
                    "row {r}: dead bound differs from the device liveness threshold"
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
            (&opened_a, INPUT_QUANT_COL_MAP.a.dead_count, h),
            (&opened_b, INPUT_QUANT_COL_MAP.b.dead_count, program.w),
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
        // The honest alpha/beta witness must be exactly the native scheme: floor the group's
        // (l2, linf) at 2^-32 (`bf16_max`, the reference `row_norms`), then derive_row_scales —
        // the same flow as `noisy_quantize`. l2 is the exact-integer path (module docs,
        // native-semantics notes); on this test data it coincides with the native
        // f32-summation row_norms, asserted here.
        let (program, rows, _) = test_trace();
        let inp = test_inputs(&program);
        let (h, k) = (program.h, program.k);
        let opened_a = open_prequant(&inp.a_int8, &inp.a_scales, h, k, BLOCK_SIZE).unwrap();
        let opened_b = open_prequant(&inp.b_int8, &inp.b_scales, program.w, k, BLOCK_SIZE).unwrap();
        for g in 0..h {
            for (opened, alpha_exp_col, alpha_man_col, beta_exp_col, beta_man_col, bound_col) in [
                (
                    &opened_a,
                    INPUT_QUANT_COL_MAP.a.alpha_exp,
                    INPUT_QUANT_COL_MAP.a.alpha_mantissa,
                    INPUT_QUANT_COL_MAP.a.beta_exp,
                    INPUT_QUANT_COL_MAP.a.beta_mantissa,
                    INPUT_QUANT_COL_MAP.a.dead_bound,
                ),
                (
                    &opened_b,
                    INPUT_QUANT_COL_MAP.b.alpha_exp,
                    INPUT_QUANT_COL_MAP.b.alpha_mantissa,
                    INPUT_QUANT_COL_MAP.b.beta_exp,
                    INPUT_QUANT_COL_MAP.b.beta_mantissa,
                    INPUT_QUANT_COL_MAP.b.dead_bound,
                ),
            ] {
                let quant = Fp8E4M3Quant::new(program.device);
                let (l2, linf) = quant.row_norms(&opened[g * k..(g + 1) * k]).unwrap();
                let (alpha, beta) = quant
                    .derive_row_scales(bf16_max(l2, NORM_FLOOR_CODE), bf16_max(linf, NORM_FLOOR_CODE), program.r)
                    .unwrap();
                let row = &rows[g * k];
                let got_alpha = (to_u64(row[alpha_exp_col]) << 7 | to_u64(row[alpha_man_col])) as u16;
                let got_beta = (to_u64(row[beta_exp_col]) << 7 | to_u64(row[beta_man_col])) as u16;
                assert_eq!(got_alpha, alpha, "group {g}: alpha differs from the native derivation");
                assert_eq!(got_beta, beta, "group {g}: beta differs from the native derivation");
                assert_eq!(
                    to_u64(row[bound_col]),
                    u64::from(bf16_max(l2, NORM_FLOOR_CODE)) + program.device.liveness_code_shift(),
                    "group {g}: dead bound differs from the native floored L2"
                );
            }
        }
    }

    #[test]
    fn all_zero_group_is_provable_with_reference_scales() {
        // An all-zero opened row is protocol-legal: the scheme floors both norms at 2^-32, so
        // alpha is finite and beta strictly positive (real noise IS injected — the reference
        // semantics). Regression: witness generation used to divide 448 by an unfloored zero
        // noised-bound and panic. Group 0 of the A side is forced all-zero (ints AND noise on
        // that side stay as generated — only the ints are zeroed, so X = 0 while the noise
        // codes still exercise the beta*n path); every constraint and LUT lookup must hold,
        // and the committed scales must equal the native derivation on the floored norms.
        let program = test_program();
        let mut inp = test_inputs(&program);
        let k = program.k;
        inp.a_int8[..k].fill(0);
        let (rows, pis) = program.generate_trace::<F, NUM_INPUT_QUANT_COLUMNS>(
            &inp.a_int8,
            &inp.a_scales,
            &inp.a_noise,
            &inp.b_int8,
            &inp.b_scales,
            &inp.b_noise,
        );

        let v: &InputQuantColumnsView<F> = rows[k - 1].borrow();
        assert_eq!(v.a.max_abs, F::ZERO, "test premise: the A group decodes to all zeros");
        let (alpha, beta) = Fp8E4M3Quant::new(program.device)
            .derive_row_scales(NORM_FLOOR_CODE, NORM_FLOOR_CODE, program.r)
            .expect("native scales on the floor");
        let got = |exp: F, man: F| (to_u64(exp) << 7 | to_u64(man)) as u16;
        assert_eq!(got(v.a.alpha_exp, v.a.alpha_mantissa), alpha, "alpha on the floored norms");
        assert_eq!(got(v.a.beta_exp, v.a.beta_mantissa), beta, "beta on the floored norms");
        assert_ne!(beta, 0, "the floored l2 keeps the noise scale strictly positive");

        let stark = S::new(program.clone());
        assert!(
            constraints_vanish(&stark, &rows, &pis),
            "all-zero A group must satisfy the AIR"
        );

        // Every declared LUT instance must stay in-domain on this trace too.
        let polys = trace_rows_to_poly_values(rows.clone());
        let lookups = input_quant_lut_lookups::<F>(Device::H100);
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
    fn fp32_fold_matches_fused_reference_at_boundaries() {
        for alpha in [0x0080, 0x3700, 0x3780, 0x3f80, 0x3fff, 0x7f7f] {
            for scale in INTERESTING_CODES {
                let coefficient = bf16_to_f32(alpha) * bf16_to_f32(scale);
                if !coefficient.is_finite() {
                    continue;
                }
                for integer in [-128, -1, 0, 1, 127] {
                    for beta in [0, 1, 0x3c80, 0x3f80, 0x7f7f] {
                        for word in [0, 0x80000000, 1, 0x80000001, 0x00004200, 0x3f800000, 0xbf800000, 0x7f7fffff] {
                            let noise = f32::from_bits(word);
                            let expected = bf16_to_f32(beta).mul_add(noise, coefficient * f32::from(integer));
                            let certificate = fp32::generate_fold::<F>(coefficient, integer, beta, noise);
                            if expected.is_finite() {
                                let (alignment, rounding, _) = certificate.unwrap();
                                assert_eq!(
                                    rounding.output.word() | ((to_u64(alignment.output_sign) as u32) << 31),
                                    expected.to_bits()
                                );
                            } else {
                                assert!(certificate.is_err());
                            }
                        }
                    }
                }
            }
        }
        let coefficient = bf16_to_f32(0x5d80) * bf16_to_f32(0x5dff);
        let beta = 0x5f80;
        let noise = 2.0f32.powi(64);
        assert!((bf16_to_f32(beta) * noise).is_infinite());
        let expected = bf16_to_f32(beta).mul_add(noise, -128.0 * coefficient);
        assert_eq!(expected, 2.0f32.powi(120));
        let (alignment, rounding, _) = fp32::generate_fold::<F>(coefficient, -128, beta, noise).unwrap();
        assert_eq!(
            rounding.output.word() | ((to_u64(alignment.output_sign) as u32) << 31),
            expected.to_bits()
        );
    }

    #[test]
    fn lut_instance_inventory_holds_on_honest_trace() {
        // Evaluates every declared LUT instance (key/value/filter algebra) over the honest trace
        // and checks it against the table semantics (mirror functions). This ties the ctl
        // descriptors to the trace the way the oracle wiring will.
        let (_, rows, _) = test_trace();
        let n = rows.len();
        let polys = trace_rows_to_poly_values(rows.clone());
        let lookups = input_quant_lut_lookups::<F>(Device::H100);
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
            LutTable::Fp32Tools | LutTable::Bytes2 => {
                let (slot, row) = crate::v4::circuit::luts::LutMultiplicities::resolve(lookup.table, keys, values).unwrap();
                if lookup.table == LutTable::Fp32Tools {
                    let expected = crate::v4::circuit::luts::stark::generate::<F>(lookup.table, slot);
                    assert_eq!(values, expected.iter().map(|column| to_u64(column[row])).collect::<Vec<_>>());
                }
            }

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
                assert_eq!(keys[0], 0, "lookup {li} row {r}: LOG16 logarithm region");
                assert!(keys[1] < 1 << 16, "lookup {li} row {r}: LOG16 key out of domain");
                assert_eq!(values[0], log2_fixed(keys[1]), "lookup {li} row {r}: LOG16 value");
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
        t[5][m.a.decode_multiply.sig_product] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "U1 tamper undetected");

        // F2 (group F): a corrupted running max is neither the previous max nor the new element.
        let mut t = rows.clone();
        t[3][m.a.max_abs] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "F2 tamper undetected");

        // W6/W8 (group W): a corrupted fold magnitude breaks the signed-fold equation.
        let vi = (0..rows.len())
            .find(|&i| rows[i][m.a.alignment.magnitude[0]] != F::ZERO)
            .unwrap();
        let mut t = rows.clone();
        t[vi][m.a.alignment.magnitude[0]] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "W6 tamper undetected");

        // W6/W7 (group W): flipping the committed output sign breaks the fold or the ±0 rule.
        let mut t = rows.clone();
        t[vi][m.a.alignment.output_sign] = F::ONE - t[vi][m.a.alignment.output_sign];
        assert!(!constraints_vanish(&stark, &t, &pis), "OUT_SIGN tamper undetected");

        // C3 (group C): a corrupted dead counter breaks the running-count transition.
        let mut t = rows.clone();
        t[7][m.b.dead_count] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "C3 tamper undetected");

        // C1 (group C): a non-boolean dead flag breaks the booleanness bind (and would
        // corrupt the RC16 certificate filters).
        let mut t = rows.clone();
        t[7][m.a.is_dead] += F::TWO;
        assert!(!constraints_vanish(&stark, &t, &pis), "C1 tamper undetected");

        // C2 (group C): the dead bound must be group-constant (only its group-final value is
        // CTL-bound to ScaleStark).
        let mut t = rows.clone();
        t[2][m.a.dead_bound] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "C2 tamper undetected");

        // S4 (group S): alpha must be group-constant.
        let mut t = rows.clone();
        t[2][m.a.alpha_exp] += F::ONE;
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
        t[0][m.a.rounding.has_exponent_carry] += F::TWO;
        assert!(
            !constraints_vanish(&stark, &t, &pis),
            "D7 ROUNDING_SIGNIFICAND_KEY_HIGH_BIT booleanness tamper undetected"
        );

        // Move a block boundary into the middle of a real block and adjust later block indices.
        // Group G pins the schedule to the mod-8 counter, so the relocation itself is rejected
        // (G8: a raised IS_BLOCK_START off `BLOCK8_COUNTER = 7`); it also resets the square
        // accumulator on real mid-block data.
        let mut t = rows.clone();
        t[9][m.schedule.is_block_start] = F::ONE;
        for row in t.iter_mut().skip(9) {
            row[m.schedule.block_index] += F::ONE;
        }
        assert!(
            !constraints_vanish(&stark, &t, &pis),
            "naive block-boundary relocation undetected"
        );
        let program2 = test_program();

        // ---- Group G (noise-decode and schedule pinning) tampers. ----

        // G1: the committed noise fields must recompose to the known packed word.
        let mut t = rows.clone();
        t[3][m.schedule.noise_a.exponent] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "G1 recomposition tamper undetected");

        // G2: a non-boolean noise sign is rejected even when the recomposition is patched
        // (sign 2 contributes exactly 2^32 to the word).
        let mut t = rows.clone();
        t[3][m.schedule.noise_b.sign] += F::TWO;
        t[3][m.known.noise_word_b] += F::from_canonical_u64(1 << 32);
        assert!(!constraints_vanish(&stark, &t, &pis), "G2 sign-booleanness tamper undetected");

        // G3: IS_GROUP_FINAL raised mid-group (counter != k-1) is rejected.
        let mut t = rows.clone();
        t[5][m.schedule.is_group_final] = F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "G3 group-final relocation undetected");

        // G4: suppressing a true group boundary is rejected — the inverse witness cannot
        // exist at `GROUP_COUNTER = k-1`.
        let mut t = rows.clone();
        t[program2.k - 1][m.schedule.is_group_final] = F::ZERO;
        assert!(!constraints_vanish(&stark, &t, &pis), "G4 group-final suppression undetected");

        // G6: shifting the element index breaks the row counter at the splice.
        let mut t = rows.clone();
        for row in t.iter_mut().skip(5) {
            row[m.schedule.element_index] += F::ONE;
        }
        assert!(!constraints_vanish(&stark, &t, &pis), "G6 element-index shift undetected");

        // G7: a parity flag out of alternation is rejected.
        let mut t = rows.clone();
        t[4][m.schedule.is_even_row] = F::ZERO;
        assert!(!constraints_vanish(&stark, &t, &pis), "G7 parity tamper undetected");

        // G11: a block index that does not follow the block starts is rejected.
        let mut t = rows.clone();
        t[12][m.schedule.block_index] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "G11 block-index tamper undetected");

        // G12: dropping A-liveness on a live row (not the h*k flip row) is rejected.
        let mut t = rows.clone();
        t[2][m.schedule.a_live] = F::ZERO;
        assert!(!constraints_vanish(&stark, &t, &pis), "G12 early liveness drop undetected");

        // Group G witness columns: a forged counter, or a forged inverse where the inverse
        // must exist, breaks the walk or the forcing identity (row 6 is mid-group, mid-block
        // and live).
        for (column, label) in [
            (m.schedule.group_counter, "G3 group counter"),
            (m.schedule.group_final_inv, "G4 group-final inverse"),
            (m.schedule.block8_counter, "G10 block counter"),
            (m.schedule.block8_counter_inv, "G9 block-counter inverse"),
            (m.schedule.a_flip_inv, "G12 flip inverse"),
            (m.known.noise_word_a, "G1 known word"),
        ] {
            let mut t = rows.clone();
            t[6][column] += F::ONE;
            assert!(!constraints_vanish(&stark, &t, &pis), "{label} tamper undetected");
        }

        // ---- Group B (block-integer L2) tampers, one per constraint family. ----
        let n = rows.len();
        let is_block_final = |r: usize| rows[(r + 1) % n][m.schedule.is_block_start] == F::ONE;

        // B2: a corrupted in-block integer-square sum breaks the block-sum transition.
        let mut t = rows.clone();
        t[4][m.a.block_l2.block_int_squared_sum] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B2 tamper undetected");

        // B3: `block_l2_product` must equal M(scale)^2 times `block_int_squared_sum`.
        let mut t = rows.clone();
        t[6][m.a.block_l2.block_l2_product] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B3 tamper undetected");

        // B4: claiming a live block dead breaks the two-sided inverse pair (a dead block's
        // candidate would drop out of the frame max and its TERM out of S).
        let live_bf = (0..n)
            .find(|&r| is_block_final(r) && rows[r][m.a.block_l2.block_l2_product_nonzero] == F::ONE)
            .expect("test data has live blocks");
        let mut t = rows.clone();
        t[live_bf][m.a.block_l2.block_l2_product_nonzero] = F::ZERO;
        assert!(!constraints_vanish(&stark, &t, &pis), "B4 tamper undetected");

        // B5: a nonzero frame candidate on a mid-block row.
        let mut t = rows.clone();
        t[2][m.a.block_l2.block_doubled_scale_exponent] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B5 tamper undetected");

        // B6: a running max that is neither the previous max nor the new candidate.
        let mut t = rows.clone();
        t[10][m.a.block_l2.running_max_doubled_scale_exponent] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B6 tamper undetected");

        // B7: a forged (group-constant) frame no longer matches the attained max at the
        // group-final row — inflating the frame to shrink every TERM is rejected.
        let mut t = rows.clone();
        for r in t.iter_mut().take(program2.k) {
            r[m.a.block_l2.frame_doubled_scale_exponent] += F::TWO;
        }
        assert!(!constraints_vanish(&stark, &t, &pis), "B7 frame tamper undetected");

        // B8: `is_far_shift` is block-final-only.
        let mut t = rows.clone();
        t[3][m.a.block_l2.is_far_shift] = F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B8 tamper undetected");

        // B10: a corrupted scaled-block-product limb breaks dividend recomposition.
        let mut t = rows.clone();
        t[5][m.a.block_l2.scaled_block_product_limbs[1]] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B10 tamper undetected");

        // B13: a selector bit on a mid-block row breaks the near-live gate sum.
        let mut t = rows.clone();
        t[1][m.a.block_l2.shift_limb_selector[0]] = F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B13 tamper undetected");

        // B14: a corrupted selected-limb quotient on a near-live block-final row breaks the
        // Euclidean split (and would shift the block's TERM).
        let near = (0..n)
            .find(|&r| {
                m.a.block_l2
                    .shift_limb_selector
                    .iter()
                    .fold(F::ZERO, |acc, &q| acc + rows[r][q])
                    == F::ONE
            })
            .expect("test data has near-live blocks");
        let mut t = rows.clone();
        t[near][m.a.block_l2.selected_limb_quotient] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B14 tamper undetected");

        // B16: forging the group-final frame sum S — the value the Scale tuple carries.
        let gi = (0..n)
            .find(|&r| rows[r][m.schedule.is_group_final] == F::ONE)
            .expect("test data has a group-final row");
        let mut t = rows.clone();
        t[gi][m.a.block_l2.running_l2_frame_sum] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B16 S tamper undetected");
        // ...and the B side's accumulator chain is enforced independently.
        let mut t = rows.clone();
        t[gi][m.b.block_l2.running_l2_frame_sum] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "B16 B-side tamper undetected");

        // ---- Group V (shared score witness) tampers, one per constraint family. ----

        // V3: SIGMA_NORM must be group-constant.
        let mut t = rows.clone();
        t[2][m.a.sigma_norm] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V3 tamper undetected");

        // V5: flipping the dominance bit negates the gap identity on a split-encoding row.
        let vr = (0..n)
            .find(|&r| rows[r][m.a.exponent_gap] != F::ZERO)
            .expect("test data has rows with distinct addend encodings");
        let mut t = rows.clone();
        t[vr][m.a.x_dominates] = F::ONE - t[vr][m.a.x_dominates];
        assert!(!constraints_vanish(&stark, &t, &pis), "V5 tamper undetected");

        // V7: claiming a near row far zeroes the expected NEAR_GAP.
        let nr = (0..n)
            .find(|&r| rows[r][m.a.near_gap] != F::ZERO)
            .expect("test data has near rows");
        let mut t = rows.clone();
        t[nr][m.a.gap_is_far] = F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V7 tamper undetected");

        // V8: a corrupted dominant significand breaks the ordering mux.
        let mut t = rows.clone();
        t[vr][m.a.norm_big] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V8 tamper undetected");

        // V9: a corrupted second-stage quotient breaks the Euclidean split.
        let mut t = rows.clone();
        t[vr][m.a.quotient] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V9 tamper undetected");

        // V10: a corrupted top slice breaks the 2^17 decomposition.
        let mut t = rows.clone();
        t[vr][m.b.sum_of_squares_top] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V10 tamper undetected");

        // V11: a forged score no longer matches its committed sum of squares.
        let mut t = rows.clone();
        t[vr][m.a.lambda] += F::ONE;
        assert!(!constraints_vanish(&stark, &t, &pis), "V11 tamper undetected");
    }

    /// A scaled check accepts `256^-1`; the unscaled check excludes this field alias.
    #[test]
    fn binary32_high_limb_has_both_scaled_and_unscaled_range_checks() {
        let (_, rows, _) = test_trace();
        let map = &INPUT_QUANT_COL_MAP;
        let mut polynomials = trace_rows_to_poly_values(rows);
        let column = map.a.rounding.quotient[1];
        let alias = F::from_canonical_u64(256).inverse();
        polynomials[column].values[0] = alias;
        let checks: Vec<_> = input_quant_lut_lookups::<F>(Device::H100)
            .into_iter()
            .filter(|lookup| lookup.table == LutTable::Range16 && lookup.keys[0].relevant_columns().contains(&column))
            .collect();
        assert!(
            checks
                .iter()
                .any(|lookup| lookup.keys[0].eval_table(&polynomials, 0, &[]) == F::ONE)
        );
        assert!(
            checks
                .iter()
                .any(|lookup| lookup.keys[0].eval_table(&polynomials, 0, &[]) == alias)
        );
        assert!(to_u64(alias) >= 65536);
    }

    #[test]
    fn degree_is_at_most_three() {
        test_stark_low_degree::<F, S, D>(S::new(test_program())).unwrap();
    }

    #[test]
    fn b200_security_sums_cover_minimum_ragged_and_maximum_k() {
        use super::super::columns::B200_INPUT_QUANT_COL_MAP;
        for k in [1024, 1056, 65536] {
            let program = InputQuantProgram {
                h: 1,
                w: 1,
                k,
                block_size: 8,
                r: 32,
                device: Device::B200,
            };
            let input = test_inputs(&program);
            let (rows, public_inputs) = program.generate_trace::<F, NUM_B200_INPUT_QUANT_COLUMNS>(
                &input.a_int8,
                &input.a_scales,
                &input.a_noise,
                &input.b_int8,
                &input.b_scales,
                &input.b_noise,
            );
            let stark = InputQuantStark::<F, D, NUM_B200_INPUT_QUANT_COLUMNS>::new(program);
            assert!(constraints_vanish(&stark, &rows, &public_inputs));
            let map = &B200_INPUT_QUANT_COL_MAP;
            for side in [&map.a, &map.b] {
                let expected: u64 = rows[..k].iter().map(|row| to_u64(row[side.gaussian_value])).sum();
                assert_eq!(to_u64(rows[k - 1][side.security_sum]), expected);
                assert!(expected < 1 << 32);
                for row in &rows[..k] {
                    assert!(to_u64(row[side.common.x_enc]) <= 276);
                    assert!(to_u64(row[side.common.sum_of_squares_top]) >= 8192);
                }
            }
        }
    }

    #[test]
    fn b200_padding_coin_ranges_and_recursive_constraints() {
        use super::super::columns::B200_INPUT_QUANT_COL_MAP;
        use crate::v4::circuit::luts::LutChecker;
        type B200 = InputQuantStark<F, D, NUM_B200_INPUT_QUANT_COLUMNS>;
        let program = InputQuantProgram {
            h: 3,
            w: 5,
            k: 1056,
            block_size: 8,
            r: 32,
            device: Device::B200,
        };
        let inputs = test_inputs(&program);
        let (rows, public_inputs) = program.generate_trace::<F, NUM_B200_INPUT_QUANT_COLUMNS>(
            &inputs.a_int8,
            &inputs.a_scales,
            &inputs.a_noise,
            &inputs.b_int8,
            &inputs.b_scales,
            &inputs.b_noise,
        );
        let stark = B200::new(program.clone());
        for (index, row) in rows.iter().enumerate() {
            let frame = StarkFrame::from_values(row, &rows[(index + 1) % rows.len()], &public_inputs);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(65537)],
                F::from_bool(index + 1 != rows.len()),
                F::from_bool(index == 0),
                F::from_bool(index + 1 == rows.len()),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            assert!(
                consumer.accumulators().into_iter().all(|value| value == F::ZERO),
                "row {index}"
            );
        }
        let known = program.known_values::<F>(&inputs.a_noise, &inputs.b_noise);
        assert_eq!(known.len(), NUM_INPUT_QUANT_KNOWN_COLUMNS);
        for (column, known) in known.iter().enumerate() {
            assert!(rows.iter().zip(&known.values).all(|(row, value)| row[column] == *value));
        }
        let tables = crate::v4::circuit::ctl::lut_tables(Device::B200);
        let lookups = input_quant_lut_lookups::<F>(Device::B200);
        let polynomials = trace_rows_to_poly_values(rows.clone());
        LutChecker::<F>::new(&tables)
            .check_trace(&lookups, &polynomials, &public_inputs, "B200")
            .unwrap();
        let map = &B200_INPUT_QUANT_COL_MAP;
        for row in &rows {
            for (side, live) in [(&map.a, map.schedule.a_live), (&map.b, map.schedule.b_live)] {
                if row[live] == F::ONE {
                    use crate::v4::circuit::unpredictability::{GAUSSIAN_CLASS_HEIGHT, NUM_GAUSSIAN_CLASSES};
                    assert!(to_u64(row[side.common.x_enc]) <= 276);
                    assert!(to_u64(row[side.common.sum_of_squares_top]) >= 8192);
                    assert!((1..=NUM_GAUSSIAN_CLASSES).contains(&to_u64(row[side.security_region])));
                    let index = to_u64(row[side.common.lambda] - row[side.security_key_base]);
                    assert!(index < GAUSSIAN_CLASS_HEIGHT);
                }
            }
        }
        let valid = |trace: &[[F; NUM_B200_INPUT_QUANT_COLUMNS]]| {
            trace.iter().enumerate().all(|(index, row)| {
                let frame = StarkFrame::from_values(row, &trace[(index + 1) % trace.len()], &public_inputs);
                let mut consumer = ConstraintConsumer::new(
                    vec![F::from_canonical_u64(7), F::from_canonical_u64(65537)],
                    F::from_bool(index + 1 != trace.len()),
                    F::from_bool(index == 0),
                    F::from_bool(index + 1 == trace.len()),
                );
                stark.eval_packed_generic(&frame, &mut consumer);
                consumer.accumulators().into_iter().all(|value| value == F::ZERO)
            })
        };
        for index in [0, 1, program.k - 1, program.k, program.h * program.k, rows.len() - 1] {
            for side in [&map.a, &map.b] {
                for column in [
                    side.security_region,
                    side.security_key_base,
                    side.gaussian_value,
                    side.security_sum,
                    side.downscale.strip_threshold,
                ] {
                    let mut forged = rows.clone();
                    forged[index][column] += F::ONE;
                    assert!(!valid(&forged), "row {index}, column {column}");
                }
            }
        }
        let mut forged = rows.clone();
        forged[0][map.a.gaussian_value] = -F::ONE;
        assert!(!valid(&forged));
        // G13: the coin-start flag is pinned to the block counter.
        for (index, value) in [(2, F::ONE), (4, F::ZERO), (8, F::ZERO)] {
            let mut forged = rows.clone();
            forged[index][map.is_coin_start] = value;
            assert!(!valid(&forged), "coin-start tamper at row {index}");
        }
        // G13 witness: a forged half-block inverse off the half-block rows.
        let mut forged = rows.clone();
        forged[1][map.coin_inv] += F::ONE;
        assert!(!valid(&forged), "coin inverse tamper undetected");
        for row in &rows[program.h * program.k..] {
            assert_eq!(row[map.a.downscale.coin], F::ONE);
        }
        let start = (0..program.h * program.k)
            .step_by(4)
            .find(|&index| to_u64(rows[index][map.a.downscale.coin_field]) >= 112)
            .expect("false-coin fixture");
        let mut forged = rows;
        forged[start][map.a.downscale.coin_field] = F::ZERO;
        for row in &mut forged[start..start + 4] {
            row[map.a.downscale.coin] = F::ONE;
        }
        let error = LutChecker::<F>::new(&tables)
            .check_trace(&lookups, &trace_rows_to_poly_values(forged), &public_inputs, "forged coin")
            .unwrap_err();
        assert!(error.contains("oversized byte"), "{error}");
        test_stark_low_degree::<F, B200, D>(stark.clone()).unwrap();
        test_stark_circuit_constraints::<F, C, B200, D>(stark).unwrap();
    }

    /// G1's unique decode needs the exponent/zero-flag and fraction-split bindings on padding
    /// too: these lookups name the schedule columns and are unfiltered on both devices.
    #[test]
    fn noise_decode_range_lookups_are_unconditional() {
        use starky::lookup::Column;
        fn padded_polys<const WIDTH: usize>(device: Device, k: usize) -> Vec<PolynomialValues<F>> {
            let program = InputQuantProgram {
                h: 3,
                w: 5,
                k,
                block_size: BLOCK_SIZE,
                r: 32,
                device,
            };
            let inp = test_inputs(&program);
            let (rows, _) = program.generate_trace::<F, WIDTH>(
                &inp.a_int8,
                &inp.a_scales,
                &inp.a_noise,
                &inp.b_int8,
                &inp.b_scales,
                &inp.b_noise,
            );
            trace_rows_to_poly_values(rows)
        }
        for device in [Device::H100, Device::B200] {
            let polys = match device {
                Device::H100 => padded_polys::<NUM_INPUT_QUANT_COLUMNS>(device, 96),
                Device::B200 => padded_polys::<NUM_B200_INPUT_QUANT_COLUMNS>(device, 1056),
            };
            let m = input_quant_column_map(device);
            let lookups = input_quant_lut_lookups::<F>(device);
            let mut expected = Vec::new();
            for (noise, high) in [
                (&m.schedule.noise_a, m.schedule.noise_fraction_high_a),
                (&m.schedule.noise_b, m.schedule.noise_fraction_high_b),
            ] {
                expected.push((LutTable::ExpInfo, vec![noise.exponent], vec![noise.is_exponent_zero]));
                expected.push((LutTable::Range16, vec![noise.fraction, high], vec![]));
            }
            expected.push((
                LutTable::Pair128,
                vec![m.schedule.noise_fraction_high_a, m.schedule.noise_fraction_high_b],
                vec![],
            ));
            let key_columns = |lookup: &LutLookup<F>| {
                let mut columns: Vec<usize> = lookup.keys.iter().flat_map(Column::relevant_columns).collect();
                columns.sort_unstable();
                columns
            };
            for (table, mut keys, values) in expected {
                keys.sort_unstable();
                let matches: Vec<&LutLookup<F>> = lookups
                    .iter()
                    .filter(|lookup| lookup.table == table && key_columns(lookup) == keys)
                    .collect();
                assert_eq!(
                    matches.len(),
                    1,
                    "{device:?} {table:?} {keys:?}: expected one noise decode lookup"
                );
                let lookup = matches[0];
                assert_eq!(
                    lookup
                        .values
                        .iter()
                        .map(|value| value.get_single_column_idx())
                        .collect::<Vec<_>>(),
                    values.iter().map(|&column| Some(column)).collect::<Vec<_>>(),
                    "{device:?} {table:?}: value columns"
                );
                // Unfiltered: the filter is one on every row, dead and phantom rows included.
                let height = polys[0].values.len();
                assert!(
                    (0..height).all(|r| lookup.filter.eval_table(&polys, r, &[]) == F::ONE),
                    "{device:?} {table:?}: filter must be enabled on every row"
                );
            }
        }
    }

    /// An oversized fraction high part, or a fraction of 2^23 (an alias of the next exponent),
    /// passes G1's recomposition only to fail the unfiltered range lookups.
    #[test]
    fn forged_noise_fraction_split_fails_the_range_lookups() {
        use crate::v4::circuit::luts::LutChecker;
        let (_, rows, pis) = test_trace();
        let m = &INPUT_QUANT_COL_MAP;
        let tables = crate::v4::circuit::ctl::lut_tables(Device::H100);
        let lookups = input_quant_lut_lookups::<F>(Device::H100);
        let check = |trace: Vec<[F; NUM_INPUT_QUANT_COLUMNS]>, label: &str| {
            LutChecker::<F>::new(&tables).check_trace(&lookups, &trace_rows_to_poly_values(trace), &pis, label)
        };
        check(rows.clone(), "honest").unwrap();
        // One more 2^16 in the high part sends the RC16 remainder negative.
        let mut forged = rows.clone();
        forged[3][m.schedule.noise_fraction_high_a] += F::ONE;
        assert!(check(forged, "high part").is_err());
        // FRACTION = 2^23 with HIGH = 128 keeps the remainder in range but fails PAIR128.
        let mut forged = rows.clone();
        forged[3][m.schedule.noise_a.fraction] = F::from_canonical_u64(1 << 23);
        forged[3][m.schedule.noise_fraction_high_a] = F::from_canonical_u64(128);
        assert!(check(forged, "fraction alias").is_err());
        // A zero flag that disagrees with the exponent fails EXPINFO.
        let normal = (0..rows.len())
            .find(|&r| rows[r][m.schedule.noise_a.exponent] != F::ZERO)
            .expect("the fixture has normal noise values");
        let mut forged = rows.clone();
        forged[normal][m.schedule.noise_a.is_exponent_zero] = F::ONE;
        assert!(check(forged, "zero flag").is_err());
    }

    #[test]
    fn circuit_constraints_match_native() {
        test_stark_circuit_constraints::<F, C, S, D>(S::new(test_program())).unwrap();
    }

    // No standalone prove/verify smoke test: this table is a CTL party (`requires_ctls`), so a
    // proof without the cross-table argument is not a supported object. The end-to-end proving
    // path is covered by `fp8::driver::tests::batch_proof_roundtrips_and_rejects_tampering`.
}
