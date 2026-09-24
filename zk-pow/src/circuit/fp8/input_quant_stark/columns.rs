//! Trace-column layout for InputQuantStark's per-element computation:
//! decode `int8 * block_scale` to bf16 `X`, accumulate the row's sum-of-squares and maximum,
//! compute `noise_term = RNE_bf16(beta*n)`, then prove the fused
//! `RNE_bf16(alpha*X + noise_term)` and cast it to fp8. The jackpot liveness flag
//! `IS_DEAD = [ABS(X) >= DEAD_BOUND]` and its in-group count are accumulated alongside this
//! path. Here `n` is public-seed-derived bf16 noise and `alpha`/`beta` are ScaleStark's row
//! constants. Each trace row carries the same flat element index from the A and B strips; a
//! group is `k` rows and a scale block is eight rows.
//!
//! The trace is padded to `next_power_of_two(max(h, w) * k)`. Per-side liveness excludes
//! phantom rows from CTLs; their data and beta are zero while alpha uses its required
//! positive-normal filler. The final row remains group-final so cyclic AIR transitions close
//! at the group-0 boundary.
//!
//! [`InputQuantColumnsView`] is `#[repr(C)]`, so its field order is the committed trace order.
//! [`INPUT_QUANT_COL_MAP`] exposes the same layout as flat column indices for LUT and CTL
//! declarations.
//!
//! A bf16 field tuple `(sign, exp, mantissa, exp_is_zero)` represents
//! `(-1)^sign * M(V) * 2^(E*(V) - 134)`, where `134 = 127 + 7` is the exponent bias plus the
//! seven fraction bits,
//! `M(V) = 128*(1 - exp_is_zero) + mantissa` and `E*(V) = exp + exp_is_zero`.
//! MUL/FMA outputs are bound through CLAMP22 and RNERND; on normal outputs,
//! `out_exp = lsb_scale + width_adjust`. Constraint labels below refer to `stark.rs`.

use crate::circuit::fp8::columns_view::columns_view;

/// One bf16 value's decode fields: sign bit, raw biased exponent field, 7-bit mantissa field,
/// and the exponent-field-is-zero flag.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Bf16FieldsView<T: Copy> {
    /// IEEE sign bit as a field element (0 = positive, 1 = negative).
    pub sign: T,
    /// Raw biased 8-bit exponent field, in `[0, 254]` (255 = inf/NaN is banned everywhere).
    pub exp: T,
    /// 7-bit fraction field without the implicit leading 1, in `[0, 127]`.
    pub mantissa: T,
    /// 1 iff `exp = 0` (value is subnormal or ±0).
    pub exp_is_zero: T,
}

/// The decode multiply `X_s = RNE_bf16(int * scale)` (group U): the M1-M6 MUL pattern whose
/// RNERND-bound outputs *are* the `X_s` fields, so only the product/cut/width columns are here.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct MulDecodeBlockView<T: Copy> {
    /// Exact significand product `M(INT) * M(SCALE)` (<= 255^2 < 2^16); RNERND key component.
    pub sig_product: T,
    /// RNERND slot `clamp(-133 - LSB_SCALE, -7, 18) + 7`: `-133 = 1 - 127 - 7` is bf16's
    /// smallest-subnormal exponent and the clamped signed fade cut is biased by +7 into the
    /// slot index. CLAMP22 embeds its signed fade argument by adding 400, giving key
    /// `535 - E*(INT) - E*(SCALE)`.
    pub cut_depth: T,
    /// RNERND-supplied exponent adjustment (see the module docs); enters the U4 exponent
    /// binding `OUT_EXP = LSB_SCALE + WIDTH_ADJUST`.
    pub width_adjust: T,
}

/// One bf16 multiply gadget instance (group M): the M1-M6 pattern used by
/// `noise_scale_multiply_{a,b}`. Output sign is not a column; it equals the relevant operand's
/// sign.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct MulBlockView<T: Copy> {
    /// Exact significand product `M(a) * M(b)` (M1; <= 255^2 < 2^16).
    pub sig_product: T,
    /// RNERND slot `clamp(-133 - LSB_SCALE, -7, 18) + 7`, bound by CLAMP22 at its embedded
    /// key `535 - E*(a) - E*(b)` (M2).
    pub cut_depth: T,
    /// Output exponent field; bound by M4 on the normal branch, zeroed by M5 otherwise.
    pub out_exp: T,
    /// Output mantissa field, RNERND-bound (M3).
    pub out_mantissa: T,
    /// RNERND-supplied exponent adjustment (M3/M4).
    pub width_adjust: T,
    /// "Output is ±0" flag, RNERND-bound (M3).
    pub out_is_zero: T,
    /// "Output exponent field is 0" flag (subnormal or zero output), RNERND-bound (M3).
    pub out_exp_is_zero: T,
}

/// The block-integer L2-input gadget (group B) computes the group's framed sum of squares:
///
/// ```text
/// block_int_squared_sum = Σ_block int8²
/// block_l2_product = M(scale)² · block_int_squared_sum
/// frame_doubled_scale_exponent = max_live 2·E*(scale)
/// shift_b = frame_doubled_scale_exponent − 2·E*(scale_b)
/// running_l2_frame_sum = Σ_b floor(block_l2_product_b · 2^Wl2 / 2^shift_b)
/// ```
///
/// On a group-final row, the five values above are commonly abbreviated `n_b`, `p_b`, `F`,
/// `sigma_b`, and `S`. The represented sum is `S * 2^(F - 268 - Wl2)`, with less than one
/// frame unit discarded per block. This is the canonical prequant sum from raw int8 values and
/// scales, not a sum of the already-rounded `X` values. `Wl2 = 27 - ceil(log2 k)` controls how
/// many low bits are retained.
///
/// Only live block-final rows contribute. A shift of at least 54 is marked
/// `is_far_shift` and contributes zero because the dividend is below `2^54`. Otherwise the
/// four 16-bit dividend limbs and a one-hot selector prove the Euclidean split exactly; all
/// split columns are zeroed or gated away off near-live rows.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct BlockL2Columns<T: Copy> {
    /// Running in-block sum of integer squares; define `n_b = block_int_squared_sum` on the
    /// block-final row (B1/B2).
    pub block_int_squared_sum: T,
    /// `block_l2_product = M(scale)² * block_int_squared_sum`; define `p_b` as its block-final
    /// value, which is below `2^33` (B3).
    pub block_l2_product: T,
    /// Inverse witness for `block_l2_product`; paired with `block_l2_product_nonzero` (B4).
    pub block_l2_product_inverse: T,
    /// One iff `block_l2_product` is nonzero (B4).
    pub block_l2_product_nonzero: T,
    /// Candidate doubled scale exponent `2*E*(scale)` on live block-final rows, else zero (B5).
    pub block_doubled_scale_exponent: T,
    /// Running maximum of `block_doubled_scale_exponent` in the group (B6).
    pub running_max_doubled_scale_exponent: T,
    /// Group-constant common exponent frame; its group-final value is
    /// `F = max_b 2*E*(scale_b)` over nonzero blocks (S4/B7).
    pub frame_doubled_scale_exponent: T,
    /// Four 16-bit limbs of `block_l2_product * 2^Wl2`, below `2^54` (B10).
    pub scaled_block_product_limbs: [T; 4],
    /// `2^r`, where `r = sigma_b mod 16`; used only on nonzero, block-final, non-far rows
    /// (B11).
    pub shift_remainder_power: T,
    /// Complement power `2^(16-r)` used to recompose the shifted high limbs (B11/B16).
    pub shift_complement_power: T,
    /// One-hot selector for the limb containing the shift boundary (B13).
    pub shift_limb_selector: [T; 4],
    /// Quotient of the selected limb by `2^r̂` (B14/B15).
    pub selected_limb_quotient: T,
    /// Remainder of the selected-limb split, below `2^r` (B14/B15).
    pub selected_limb_remainder: T,
    /// One on live block-final rows whose frame shift is at least 54 (B8).
    pub is_far_shift: T,
    /// Running framed block-term sum; define `S = running_l2_frame_sum` at group-final (B16).
    pub running_l2_frame_sum: T,
}

/// Columns for the fused operation `NOISED = RNE_bf16(p + c)` (constraint group W).
///
/// - `p = alpha·x`, the **exact, unrounded product**: significand `M_P = M(ALPHA)·M(X) < 2^16`,
///   sign `S_P`, in units of `2^EP` with `EP = E*(ALPHA) + E*(X) − 268`;
/// - `c = RNE_bf16(beta·n)`, the **already-rounded addend**: significand `M_C = M(c) < 2^8`,
///   sign `S_C`, in units of `2^EC` with `EC = E*(c) − 134`;
///
/// The gadget aligns both integer significands to a common binary scale, adds or subtracts them
/// exactly, and rounds only the final result. Very large alignment gaps use a sticky-preserving
/// capped shift. A fold wider than RNERND's 17-bit input is first compressed with round-to-odd,
/// which preserves the later eight-bit bf16 rounding. The module's “Single-rounding FMA”
/// section gives the full argument.
///
/// `M_P` is stored outside this block, in `noised_fma_product_{a,b}`.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct FmaBlockView<T: Copy> {
    /// One iff `EP >= EC`, meaning the product has the same or a coarser binary unit than the
    /// addend. This compares units, not numeric magnitudes (W2).
    pub product_scale_ge_addend: T,
    /// Exact scale-gap witness: `EP - EC` when `EP >= EC`, otherwise `EC - EP - 1`.
    /// The minus one makes the two order branches disjoint at equality (W2).
    pub scale_gap_slack: T,
    /// Selects the sticky-only far-gap path. Its threshold is 10 when the smaller operand has
    /// at most eight bits, or 20 when a subtracted 16-bit product can borrow across a binade
    /// boundary (W3).
    pub is_far_gap: T,
    /// `d - threshold` on far rows, range-checked to prove that the threshold was reached.
    /// Ignored on near rows (W3).
    pub far_gap_slack: T,
    /// Alignment shift used by the integer fold: the true gap `d` on near rows, or the last
    /// pre-threshold gap (9 or 19) on far rows (W4).
    pub exp_gap_capped: T,
    /// `2^exp_gap_capped`, supplied by POW2D; its key domain also proves the shift is at most
    /// 19 (W4/W5).
    pub exp_gap_pow2: T,
    /// Product significand after conversion to the fold's common binary unit (W5).
    pub aligned_product: T,
    /// Addend significand after conversion to the fold's common binary unit (W5).
    pub aligned_addend: T,
    /// Magnitude of the exact signed integer sum
    /// `|(-1)^S_P*aligned_product + (-1)^S_C*aligned_addend|` (W6).
    pub folded_magnitude: T,
    /// One iff `folded_magnitude` is zero (W7).
    pub folded_is_zero: T,
    /// Inverse witness for `folded_magnitude` (W7).
    pub folded_inverse: T,
    /// One iff `folded_magnitude >= 2^17`; such a value must be compressed before RNERND (W8).
    pub is_wide: T,
    /// Number of low bits removed on wide rows:
    /// `bit_length(folded_magnitude) - 17`; zero on narrow rows (W8).
    pub compression_shift: T,
    /// `2^compression_shift`, supplied by POW2D; one on narrow rows (W8).
    pub shift_pow2: T,
    /// Leading 17 bits of a wide folded magnitude (W8).
    pub compression_quotient: T,
    /// Least-significant bit of `compression_quotient`, used to detect whether its retained
    /// head is already odd (W9).
    pub compression_quotient_lsb: T,
    /// Low bits discarded by the wide split (W8).
    pub compression_remainder: T,
    /// Inverse witness for `compression_remainder` (W9).
    pub compression_remainder_inverse: T,
    /// One iff `compression_remainder` is nonzero; round-to-odd consumes this in W10.
    pub sticky: T,
    /// Significand handed to RNERND: `folded_magnitude` on narrow rows; on wide rows, the
    /// 17-bit quotient increased by one exactly when it is even and discarded bits were
    /// nonzero (round-to-odd, W10).
    pub rounding_significand_key: T,
    /// High bit of `rounding_significand_key`. Together with an RC16 check it proves that key is
    /// below `2^17`, preventing an alias into a neighboring RNERND cut slot.
    pub rounding_significand_key_high_bit: T,
    /// Binary scale attached to `rounding_significand_key`:
    /// `min(EP, EC) + is_far_gap*(d - exp_gap_capped)
    /// + is_wide*compression_shift`. The two corrections account for the far-gap cap and the
    ///   wide split (W10).
    pub key_scale: T,
    /// RNERND slot of the final rounding, `clamp(-133 - KEY_SCALE, -7, 18) + 7`, bound by
    /// CLAMP22 under its `+400` key embedding: `key = 267 - KEY_SCALE` (W11).
    pub cut_used: T,
    /// Result sign: the exact fold's sign when `folded_magnitude != 0`; for exact zero, the IEEE
    /// rule that only two negative zeros add to negative zero (W6/W7).
    pub out_sign: T,
    /// Output exponent field (W12).
    pub out_exp: T,
    /// Output mantissa field, RNERND-bound (W11).
    pub out_mantissa: T,
    /// RNERND-supplied exponent adjustment (W11/W12).
    pub width_adjust: T,
    /// "Output is ±0" flag, RNERND-bound (W11).
    pub out_is_zero: T,
    /// "Output exponent field is 0" flag, RNERND-bound (W11).
    pub out_exp_is_zero: T,
}

/// View of one InputQuantStark trace row: verifier-known columns first, followed by the A side
/// and then the B side. The two sides carry identical column blocks.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct InputQuantColumnsView<T: Copy> {
    // ------------------------------------------------------------------------------------------
    // Verifier-known columns. They are committed with the trace, but the verifier independently
    // recomputes the public-seed noise and geometry-derived schedule, then checks the opened
    // values for equality. They need no duplicate AIR equations. These are the leading
    // NUM_INPUT_QUANT_KNOWN_COLUMNS columns.
    // ------------------------------------------------------------------------------------------
    /// A-side bf16 noise `n = (E @ F)[element_index]`, derived from the public noise seeds.
    /// The verifier recomputes these fields; finite subnormals are allowed.
    pub noise_a: Bf16FieldsView<T>,
    /// B-side noise decode fields (see `noise_a`).
    pub noise_b: Bf16FieldsView<T>,
    /// 1 on the last row of each k-element group; CTL filter of the Scale channel; gates the
    /// accumulator resets. Period k (closed-form zeta evaluation).
    pub is_group_final: T,
    /// Flat element index (= trace row index); define `ELEM_IDX = element_index` — the A CTL
    /// key. The B key is `ELEM_IDX + h*k` (the public input [`B_KEY_OFFSET_PUBLIC_INPUT`]),
    /// keeping both sides' key spaces disjoint.
    pub element_index: T,
    /// Period-2 flag; filter of the pair-packed Blake3/Matmul channels (each even row's tuple
    /// carries this row's and the next row's element).
    pub is_even_row: T,
    /// Period-8 flag (`element_index ≡ 0 mod 8`); filter of the scales channel and anchor of the
    /// scale-field copy constraints (P2).
    pub is_block_start: T,
    /// Scales-plane block index `floor(element_index / 8)`; define `BLK_IDX = block_index` —
    /// the A scales CTL key. The B key is `BLK_IDX + h*k/8` (the public input
    /// [`B_BLOCK_KEY_OFFSET_PUBLIC_INPUT`]).
    pub block_index: T,
    /// A-side liveness: one on rows `[0, h*k)`, zero beyond. It filters A-side CTLs; dead rows
    /// carry the phantom fill.
    pub a_live: T,
    /// B-side liveness: 1 on rows `[0, w*k)`, 0 beyond (see `a_live`).
    pub b_live: T,

    // ------------------------------------------------------------------------------------------
    // A side (89 columns).
    // ------------------------------------------------------------------------------------------
    /// The A element's committed int8-plane byte (raw two's-complement, in [0, 255]), CTL-bound
    /// to Blake3's int8 channel; INT8DEC key — the key domain is the byte range proof (P1).
    pub int8_byte_a: T,
    /// Exact bf16 decode fields of the A int8 value, bound as the INT8DEC value tuple (byte 0 ->
    /// +0 fields; byte 0x80 = -128 -> `(1, 134, 0, 0)`).
    pub int_bf16_a: Bf16FieldsView<T>,
    /// The A 8-element block scale's decode fields, CTL-bound (as the affine code expression) to
    /// Blake3's scales channel on `IS_BLOCK_START` rows and copied across the block (P2).
    /// Finiteness via the EXPINFO key domain; subnormal and zero scales are legal (P3).
    pub scale_a: Bf16FieldsView<T>,
    /// The A decode multiply `X_A = RNE_bf16(int * scale)` (group U).
    pub decode_multiply_a: MulDecodeBlockView<T>,
    /// Decoded A element sign, bound to `INT_A_SIGN xor SCALE_A_SIGN` (P4).
    pub x_a_sign: T,
    /// Decoded A element exponent field (an output of `decode_multiply_a`; U4/U5).
    pub x_a_exp: T,
    /// Decoded A element mantissa field (RNERND-bound, U3).
    pub x_a_mantissa: T,
    /// Decoded A element exponent-is-zero flag (RNERND-bound).
    pub x_a_exp_is_zero: T,
    /// Decoded A element is-±0 flag (RNERND-bound).
    pub x_a_is_zero: T,
    /// The A-side block-integer L2 gadget (group B): block sums, block product, frame
    /// candidates/max/pin, dividend limbs, shift powers, limb selector, Euclidean split, far
    /// flag, and the group accumulator `S`.
    pub block_l2_a: BlockL2Columns<T>,
    /// Running maximum of `ABS(X_A)` over the group (group F); group-final value = the matrix
    /// row's max-abs code — linf's source only (no L2 role).
    pub max_abs_a: T,
    /// Alpha's exponent field (alpha is positive-normal by ScaleStark's derivation + the S1 RCs,
    /// so no sign / exp-zero columns); CTL-bound to ScaleStark; group-constant (S4).
    pub alpha_a_exp: T,
    /// Alpha's mantissa field (S3 PAIR128).
    pub alpha_a_mantissa: T,
    /// Beta's exponent field (beta is nonnegative; zero and subnormal legal) (S2).
    pub beta_a_exp: T,
    /// Beta's mantissa field (S3 PAIR128).
    pub beta_a_mantissa: T,
    /// Beta's exponent-is-zero flag (EXPINFO-bound, S2).
    pub beta_a_exp_is_zero: T,
    /// The bf16 multiply `NOISE_TERM_A = beta*n` (group M). Output sign = `NOISE_A_SIGN`.
    pub noise_scale_multiply_a: MulBlockView<T>,
    /// A's FMA significand product `(128 + ALPHA_A_MANTISSA) * M(X_A)` (W1).
    pub noised_fma_product_a: T,
    /// The single-rounding `NOISED_A = fma(alpha, x, beta*n)` (group W).
    pub noised_value_fma_a: FmaBlockView<T>,
    /// fp8 E4M3 code of the quantized noised A value, pinned (byte range included) by the
    /// tuple-valued QCAST lookup on the FMA output code (Q1); the value Matmul consumes.
    pub code_noised_a: T,
    /// The liveness threshold in bf16-code space: `code(l2f) + 256`, the code of the plaintext
    /// dead bound `tau_idle * DELTA * l2f = 4 * l2f`, where `l2f = max(l2, NORM_FLOOR)` is the
    /// group's floored L2 norm. Group-constant (C2); the group-final value rides the Scale
    /// tuple, where T1 binds it to the floored-sqrt claim. When `E*(l2f) >= 253` the bound
    /// exceeds every finite abs code and all entries are alive, matching the plaintext.
    pub dead_bound_a: T,
    /// Jackpot liveness flag: 1 iff `ABS(X_A) >= DEAD_BOUND_A`, i.e. `|X| >= 4*l2f`
    /// (bf16 code order equals value order on nonnegative finite codes). Boolean, zero on
    /// phantom rows (C1); certified by the dead/alive RC16 pair (ctl.rs).
    pub is_dead_a: T,
    /// In-group running count of `IS_DEAD_A` (C3); group-final value rides the Scale tuple.
    pub dead_count_a: T,
    /// V1 (jackpot check 4): bit length of the scaled significand product
    /// `NOISED_FMA_PRODUCT_A = M(alpha)*M(X_A)`. Bound by WIDTH16, whose second value column
    /// binds the product-nonzero flag to `1 - X_A_IS_ZERO` (M(alpha) >= 128, so the product is
    /// zero exactly when `X_A` is).
    pub x_width_a: T,
    /// V1 (jackpot check 4): the scaled element's exponent encoding
    /// `enc(alpha*x) = e(alpha*x) + 268`, or 0 when `X_A = 0` — pinned by
    /// `X_ENC = (1 - X_A_IS_ZERO)*(ALPHA_A_EXP + E*(X_A) - 1) + X_WIDTH_A`.
    pub x_enc_a: T,
    /// V2 (jackpot check 4): the group's noise-std encoding `enc(sigma) = e(sigma) + 268`,
    /// group-constant; the group-final value is bound to ScaleStark's affine sigma encoding
    /// by the group tuple's `SIGMA_ENC` slot. Positive on live groups (sigma is never zero
    /// in-scheme); 0 on phantom rows.
    pub sigma_enc_a: T,
    /// V3 (jackpot check 4): the group's normalized noise-std significand
    /// `sigma = SIGMA_NORM * 2^(SIGMA_ENC - 268 - 15)` with `SIGMA_NORM in [2^15, 2^16)`,
    /// group-constant; the group-final value is bound to ScaleStark's committed column by
    /// the group tuple's `SIGMA_NORM` slot. 0 on phantom rows.
    pub sigma_norm_a: T,
    /// V4 (jackpot check 4): `2^(16 - X_WIDTH_A)` — the POW2D value at the affine key
    /// `16 - X_WIDTH_A`.
    pub x_pow_a: T,
    /// V4 (jackpot check 4): the scaled element's normalized significand
    /// `X_NORM = NOISED_FMA_PRODUCT * X_POW in [2^15, 2^16)`, so
    /// `|alpha*x| = X_NORM * 2^(X_ENC - 268 - 15)`; 0 when `X_A = 0`.
    pub x_norm_a: T,
    /// V5 (jackpot check 4): 1 iff `X_ENC_A >= SIGMA_ENC_A` on a nonzero element — which of
    /// the two summand addends `(alpha*x)^2`, `sigma^2` dominates. Boolean; forced by the
    /// EXPONENT_GAP range check (a wrong bit makes the gap negative and its key wrap).
    pub x_dominates_a: T,
    /// V5 (jackpot check 4): the addend exponent gap
    /// `(2*X_DOMINATES - 1) * (X_ENC - SIGMA_ENC) = |e(alpha*x) - e(sigma)|` in binades
    /// (`SIGMA_ENC` when `X_A = 0`, forcing the far cut). Range-checked.
    pub exponent_gap_a: T,
    /// V6 (jackpot check 4): 1 iff `EXPONENT_GAP >= 16` — the far cut: the small addend is
    /// dropped from the sum of squares (exact: its floored term is zero at any such gap). Boolean,
    /// two-sided by a filtered RC16 per branch (`EXPONENT_GAP - 16` under the bit,
    /// `15 - EXPONENT_GAP` under its complement).
    pub gap_is_far_a: T,
    /// V7 (jackpot check 4): `NEAR_GAP = (1 - GAP_IS_FAR) * EXPONENT_GAP in [0, 15]` — the
    /// POW2D shift key of both floor stages.
    pub near_gap_a: T,
    /// V7 (jackpot check 4): `2^NEAR_GAP` — the POW2D value at NEAR_GAP, the shared divisor
    /// of both floor stages.
    pub near_gap_pow_a: T,
    /// V8 (jackpot check 4): the dominant addend's normalized significand
    /// `NORM_BIG = X_DOMINATES * X_NORM + (1 - X_DOMINATES) * SIGMA_NORM in [2^15, 2^16)`.
    pub norm_big_a: T,
    /// V8 (jackpot check 4): the other addend's normalized significand
    /// `NORM_SMALL = X_DOMINATES * SIGMA_NORM + (1 - X_DOMINATES) * X_NORM` (0 when `X_A = 0`).
    pub norm_small_a: T,
    /// V9 (jackpot check 4): `floor(NORM_SMALL^2 / 2^NEAR_GAP)` — the first floor stage,
    /// pinned by `NORM_SMALL^2 = HALF_QUOTIENT * NEAR_GAP_POW + HALF_REMAINDER`.
    pub half_quotient_a: T,
    /// V9 (jackpot check 4): the first stage's remainder, in `[0, NEAR_GAP_POW)` —
    /// range-checked on both `HALF_REMAINDER` and `NEAR_GAP_POW - 1 - HALF_REMAINDER`.
    pub half_remainder_a: T,
    /// V9 (jackpot check 4): `floor(NORM_SMALL^2 / 2^(2*NEAR_GAP))` — the second floor
    /// stage, pinned by `HALF_QUOTIENT = QUOTIENT * NEAR_GAP_POW + REMAINDER`.
    pub quotient_a: T,
    /// V9 (jackpot check 4): the second stage's remainder, in `[0, NEAR_GAP_POW)`.
    pub remainder_a: T,
    /// V10 (jackpot check 4): the top slice `KAPPA = floor(V / 2^17)` of the sum of squares
    /// `V = NORM_BIG^2 + (1 - GAP_IS_FAR) * QUOTIENT`; in `[2^13, 2^16)` on live rows. The
    /// LOG16 lookup serves its fixed-point log and doubles as its range proof.
    pub sum_of_squares_top_a: T,
    /// V10 (jackpot check 4): low 16 bits of the rest `V - 2^17 * KAPPA`. Range-checked.
    pub sum_of_squares_rest_lo_a: T,
    /// V10 (jackpot check 4): high bit of the rest `V - 2^17 * KAPPA`. Boolean.
    pub sum_of_squares_rest_hi_a: T,
    /// V10 (jackpot check 4): `floor(64 * log2 KAPPA)` — the LOG16 value at the top slice.
    pub log_fraction_a: T,
    /// V11 (jackpot check 4): the element's summand score
    /// `LAMBDA = 128*max(X_ENC, SIGMA_ENC) - 2624 + LOG_FRACTION`
    /// `~= 64*(log2((alpha*x)^2 + sigma^2) + 508)`, never above, within 1.1 steps below
    /// (0 on phantom rows); rides the operand-code channel to MatmulB200Stark's per-lane skip
    /// certificates.
    pub lambda_a: T,

    // ------------------------------------------------------------------------------------------
    // B side (89 columns): mirrors the A side.
    // ------------------------------------------------------------------------------------------
    /// The B element's committed int8-plane byte (see `int8_byte_a`).
    pub int8_byte_b: T,
    /// Exact bf16 decode fields of the B int8 value (see `int_bf16_a`).
    pub int_bf16_b: Bf16FieldsView<T>,
    /// The B block scale's decode fields (see `scale_a`).
    pub scale_b: Bf16FieldsView<T>,
    /// The B decode multiply (see `decode_multiply_a`).
    pub decode_multiply_b: MulDecodeBlockView<T>,
    /// Decoded B element sign (see `x_a_sign`).
    pub x_b_sign: T,
    /// Decoded B element exponent field.
    pub x_b_exp: T,
    /// Decoded B element mantissa field.
    pub x_b_mantissa: T,
    /// Decoded B element exponent-is-zero flag.
    pub x_b_exp_is_zero: T,
    /// Decoded B element is-±0 flag.
    pub x_b_is_zero: T,
    /// The B-side block-integer L2 gadget (see `block_l2_a`).
    pub block_l2_b: BlockL2Columns<T>,
    /// Running max of `ABS(X_B)` over the group.
    pub max_abs_b: T,
    /// B alpha exponent field.
    pub alpha_b_exp: T,
    /// B alpha mantissa field.
    pub alpha_b_mantissa: T,
    /// B beta exponent field.
    pub beta_b_exp: T,
    /// B beta mantissa field.
    pub beta_b_mantissa: T,
    /// B beta exponent-is-zero flag.
    pub beta_b_exp_is_zero: T,
    /// The bf16 multiply `NOISE_TERM_B = beta*n`, B side (group M).
    pub noise_scale_multiply_b: MulBlockView<T>,
    /// B's FMA significand product `(128 + ALPHA_B_MANTISSA) * M(X_B)` (W1).
    pub noised_fma_product_b: T,
    /// The single-rounding `NOISED_B = fma(alpha, x, beta*n)`, B side (group W).
    pub noised_value_fma_b: FmaBlockView<T>,
    /// fp8 code of the quantized noised B value (Q2).
    pub code_noised_b: T,
    /// B liveness threshold (see `dead_bound_a`).
    pub dead_bound_b: T,
    /// B liveness flag (see `is_dead_a`).
    pub is_dead_b: T,
    /// In-group running count of `IS_DEAD_B` (C3).
    pub dead_count_b: T,
    /// B scaled-product width (see `x_width_a`).
    pub x_width_b: T,
    /// B scaled-element encoding (see `x_enc_a`).
    pub x_enc_b: T,
    /// B group noise-std encoding (see `sigma_enc_a`).
    pub sigma_enc_b: T,
    /// B group normalized noise-std significand (see `sigma_norm_a`).
    pub sigma_norm_b: T,
    /// B normalization power (see `x_pow_a`).
    pub x_pow_b: T,
    /// B normalized scaled significand (see `x_norm_a`).
    pub x_norm_b: T,
    /// B dominance bit (see `x_dominates_a`).
    pub x_dominates_b: T,
    /// B addend exponent gap (see `exponent_gap_a`).
    pub exponent_gap_b: T,
    /// B far-cut bit (see `gap_is_far_a`).
    pub gap_is_far_b: T,
    /// B near gap (see `near_gap_a`).
    pub near_gap_b: T,
    /// B near-gap power (see `near_gap_pow_a`).
    pub near_gap_pow_b: T,
    /// B dominant significand (see `norm_big_a`).
    pub norm_big_b: T,
    /// B other significand (see `norm_small_a`).
    pub norm_small_b: T,
    /// B first floor stage (see `half_quotient_a`).
    pub half_quotient_b: T,
    /// B first-stage remainder (see `half_remainder_a`).
    pub half_remainder_b: T,
    /// B second floor stage (see `quotient_a`).
    pub quotient_b: T,
    /// B second-stage remainder (see `remainder_a`).
    pub remainder_b: T,
    /// B sum-of-squares top slice (see `sum_of_squares_top_a`).
    pub sum_of_squares_top_b: T,
    /// B sum-of-squares rest, low 16 bits (see `sum_of_squares_rest_lo_a`).
    pub sum_of_squares_rest_lo_b: T,
    /// B sum-of-squares rest, high bit (see `sum_of_squares_rest_hi_a`).
    pub sum_of_squares_rest_hi_b: T,
    /// B fixed-point log (see `log_fraction_a`).
    pub log_fraction_b: T,
    /// B summand score (see `lambda_a`).
    pub lambda_b: T,
}

/// Total number of committed InputQuantStark columns.
pub const NUM_INPUT_QUANT_COLUMNS: usize = size_of::<InputQuantColumnsView<u8>>();

// Layout total: 15 verifier-known columns plus two identical 107-column sides. Each side
// includes one FMA rounding-key high bit, 21 block-L2 columns, the three jackpot-liveness
// columns, and the 22-column summand score (jackpot check 4).
const _: () = assert!(NUM_INPUT_QUANT_COLUMNS == 229);

/// `2^Wl2`, the job's block-L2 precision scale
/// (`Wl2 = 27 - ceil(log2 k)`, in `[11, 16]` over the envelope `2048 <= k <= 2^16`).
pub const WL2_POW_PUBLIC_INPUT: usize = 0;
/// `h*k`: the B slots' element-key offset. The B key spaces stay disjoint from A's
/// (`ELEM_IDX` in `[0, h*k)`) as `ELEM_IDX + h*k`.
pub const B_KEY_OFFSET_PUBLIC_INPUT: usize = 1;
/// `h*k/8`: the B slot's block-key offset (`BLK_IDX + h*k/8`, one scales block per 8 elements).
pub const B_BLOCK_KEY_OFFSET_PUBLIC_INPUT: usize = 2;
/// `w`: the A operand-code multiplicity — Matmul reads each A element in `w` output cells.
pub const OPERAND_MULT_A_PUBLIC_INPUT: usize = 3;
/// `h`: the B operand-code multiplicity — Matmul reads each B element in `h` output cells.
pub const OPERAND_MULT_B_PUBLIC_INPUT: usize = 4;
/// Number of InputQuantStark public inputs. The geometry slots (1..=4) feed the CTL column
/// expressions (`ctl.rs`), keeping the CTL structure — hence the compiled verifier circuit —
/// free of geometry constants; the native gateway pins every slot to its statement-derived
/// value.
pub const NUM_INPUT_QUANT_PUBLIC_INPUTS: usize = 5;

columns_view!(InputQuantColumnsView, NUM_INPUT_QUANT_COLUMNS, INPUT_QUANT_COL_MAP);

/// Number of leading verifier-known columns: both noise decodes, schedule flags and indices,
/// and per-side liveness.
/// `InputQuantProgram::known_values` recomputes them from public seeds and geometry, and the
/// batch verifier checks their trace openings.
pub const NUM_INPUT_QUANT_KNOWN_COLUMNS: usize = INPUT_QUANT_COL_MAP.b_live + 1;

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_INPUT_QUANT_COLUMNS] = INPUT_QUANT_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        // Verifier-known columns occupy the leading contiguous range.
        assert_eq!(INPUT_QUANT_COL_MAP.noise_a.sign, 0);
        assert_eq!(INPUT_QUANT_COL_MAP.noise_b.exp_is_zero, 7);
        assert_eq!(INPUT_QUANT_COL_MAP.is_group_final, 8);
        assert_eq!(INPUT_QUANT_COL_MAP.block_index, 12);
        assert_eq!(INPUT_QUANT_COL_MAP.a_live, 13);
        assert_eq!(INPUT_QUANT_COL_MAP.b_live, 14);
        assert_eq!(NUM_INPUT_QUANT_KNOWN_COLUMNS, 15);
        // A side spans [15, 121], B side [122, 228].
        assert_eq!(INPUT_QUANT_COL_MAP.int8_byte_a, 15);
        assert_eq!(INPUT_QUANT_COL_MAP.dead_count_a, 99);
        assert_eq!(INPUT_QUANT_COL_MAP.x_width_a, 100);
        assert_eq!(INPUT_QUANT_COL_MAP.sigma_norm_a, 103);
        assert_eq!(INPUT_QUANT_COL_MAP.sum_of_squares_top_a, 117);
        assert_eq!(INPUT_QUANT_COL_MAP.lambda_a, 121);
        assert_eq!(INPUT_QUANT_COL_MAP.int8_byte_b, 122);
        assert_eq!(INPUT_QUANT_COL_MAP.lambda_b, NUM_INPUT_QUANT_COLUMNS - 1);
        // Spot-check the nested blocks.
        assert_eq!(INPUT_QUANT_COL_MAP.block_l2_a.block_int_squared_sum, 32);
        assert_eq!(INPUT_QUANT_COL_MAP.block_l2_a.frame_doubled_scale_exponent, 38);
        assert_eq!(INPUT_QUANT_COL_MAP.block_l2_a.running_l2_frame_sum, 52);
        assert_eq!(INPUT_QUANT_COL_MAP.max_abs_a, 53);
        assert_eq!(INPUT_QUANT_COL_MAP.noise_scale_multiply_a.sig_product, 59);
        assert_eq!(INPUT_QUANT_COL_MAP.noised_fma_product_a, 66);
        assert_eq!(INPUT_QUANT_COL_MAP.noised_value_fma_a.product_scale_ge_addend, 67);
        assert_eq!(INPUT_QUANT_COL_MAP.noised_value_fma_a.out_exp_is_zero, 95);
        assert_eq!(INPUT_QUANT_COL_MAP.code_noised_a, 96);
        assert_eq!(INPUT_QUANT_COL_MAP.dead_bound_a, 97);
        assert_eq!(INPUT_QUANT_COL_MAP.block_l2_b.scaled_block_product_limbs[0], 146);
        assert_eq!(INPUT_QUANT_COL_MAP.noised_fma_product_b, 173);
        assert_eq!(INPUT_QUANT_COL_MAP.code_noised_b, 203);
    }

    #[test]
    fn view_array_roundtrip() {
        let mut arr = [0u64; NUM_INPUT_QUANT_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: InputQuantColumnsView<u64> = arr.into();
        assert_eq!(view.noise_a.sign, 1);
        assert_eq!(view.int8_byte_a, INPUT_QUANT_COL_MAP.int8_byte_a as u64 * 3 + 1);
        assert_eq!(
            view.noised_value_fma_b.rounding_significand_key,
            INPUT_QUANT_COL_MAP.noised_value_fma_b.rounding_significand_key as u64 * 3 + 1
        );
        assert_eq!(view.lambda_b, (NUM_INPUT_QUANT_COLUMNS as u64 - 1) * 3 + 1);
        let back: [u64; NUM_INPUT_QUANT_COLUMNS] = view.into();
        assert_eq!(back, arr);

        let borrowed: &InputQuantColumnsView<u64> = arr.borrow();
        assert_eq!(borrowed.max_abs_a, arr[INPUT_QUANT_COL_MAP.max_abs_a]);
    }
}
