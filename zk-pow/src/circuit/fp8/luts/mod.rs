//! The committed LUT oracle: one AIR per logical table.
//!
//! [`columns`] defines the layout, [`stark`] generates and commits the tables,
//! [`ctl`] declares their CTL halves, and [`witness`] checks lookups and counts multiplicities.
//! Complete channels are assembled in [`super::ctl`].
//!
//! Each of the sixteen [`LutTable`]s ([`LUT_TABLES`](super::ctl::LUT_TABLES)) is its own table of
//! the batch STARK — a [`LutStark`] at its natural committed height [`lut_height`] (the
//! array orders the batch by descending height, so nothing pays for another table's
//! padding). A LUT AIR has no constraints of its own; its columns split into two classes:
//!
//! - **Precommitted columns** (class (c)): the key column(s) — a plain ramp for tables whose
//!   key domain fills the height, the enumerated key tuple for BYTES2/PAIR128, or a saturated
//!   key `min(i, live - 1)` for the sub-height tables — followed by every stored value column
//!   in slot order ([`lut_precommitted_values`]). Generated once system-wide and committed at
//!   setup by [`lut_preprocessed_data`] into a `BatchStarkPreprocessedData` whose Merkle cap
//!   is a consensus constant. Proofs never recommit these columns — the batch prover copies
//!   the setup commitment and FRI opens it alongside the trace oracles, while constraints and
//!   lookups keep addressing them through the table's ordinary column indices.
//! - **Multiplicity columns** (per-proof): one trace column per slot, counting how often each
//!   row is looked up — [`LutMultiplicities`], job-dependent, committed with the trace like
//!   any online column.
//!
//! The lookup argument is one [`starky::cross_table_lookup::CrossTableLookup`] per table ([`lut_cross_table_lookups`]):
//! the looking side collects every instance of every AIR's inventory (`keys ++ values` over
//! the consumer's own trace), the looked side is the table's slots — a multi-slot looked side
//! for the folded tables — each slot's `(ramp + key offset, stored values...)` tuple filtered
//! by its multiplicity column, whose value *is* the row's multiplicity in the channel (the
//! logup numerator semantics).
//!
//! Values are produced by exhaustively evaluating the canonical Rust mirror each table binds —
//! reusing the starks' own mirror functions wherever one exists, so the tables and the traces
//! cannot drift apart. [`generate`] returns the stored value columns (`columns[j][row]`); a
//! row's key is its index plus the slot's key offset (`2^17 * slot` for RNERND, `2^16 * rel`
//! for B200ALIGN). `RANGE16` stores nothing: its ramp
//! key column *is* the table, and `BYTES2`/`PAIR128` store the enumerated key tuple itself.
//!
//! Padding (sub-height tables only) is always a *repeated valid entry*, never an
//! out-of-domain key: the saturated key column repeats the last live key and the value columns
//! are padded alike, so a malicious nonzero multiplicity on a padding row only re-proves a
//! fact the table already serves.
//!
//! RNERND's slot index is a *signed* subnormal fade cut biased by +7: slot `s` serves
//! `cut = s - 7 in [-7, 18]`, i.e. `KEY_SCALE = -133 - cut`. The negative cuts (slots 0-6)
//! distinguish normal from subnormal cancellation results with `V < 128`, which share a
//! significand but not a scale; slot 0 also serves every `KEY_SCALE >= -126` (all normal)
//! and slot 25 every `KEY_SCALE <= -151` (all zero), so 26 slots cover the full scale range
//! bit-exactly.

pub mod columns;
pub mod ctl;
pub mod stark;
pub mod witness;

pub use super::ctl::lut_cross_table_lookups;
pub use columns::{
    LutSlotLayout, XFPOW2_CAP, XFPOW2_LIMBS, XFPOW2_ZERO_POINT, lut_height, lut_num_columns, lut_slot_layout,
    num_precommitted_columns, num_slots, slot_height,
};
pub use ctl::ctl_looked_lut_slot;
pub use stark::{
    B200AlignStark, Bytes2Stark, Clamp22Stark, Div448Stark, ExpInfoStark, Int8DecStark, Log16Stark, LutStark, Pair128Stark,
    Pow2DStark, Pow2GbStark, QcastStark, Range16Stark, RneRndStark, Width16Stark, Width32Stark, XfPow2Stark, generate,
    lut_precommitted_values, lut_preprocessed_data, lut_preprocessed_inputs, lut_trace,
};
pub use witness::{LutChecker, LutMultiplicities};

/// The consensus lookup tables (every AIR lookup, range checks included,
/// targets one of these; there are no in-trace tables). Each is
/// generated once by exhaustively evaluating the canonical Rust function it mirrors,
/// precommitted at setup time (`stark::lut_preprocessed_data`) and served as its own
/// AIR of the batch (`stark::LutStark`) through one [`starky::cross_table_lookup::CrossTableLookup`] channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LutTable {
    /// 16-bit range check (the ramp itself is the key) — every `RC16` instance.
    Range16,
    /// Paired 8-bit range check; a lone byte checks `(d, 0)` — Blake message bytes.
    Bytes2,
    /// Paired 7-bit range check (2^14 keys, padded into the 2^16 height group) — mantissa
    /// fields, ScaleStark's sqrt parity/l2-linf pairs.
    Pair128,
    /// Keyed bf16 code, value fp8 E4M3 code (one precommitted column; the binding pins the
    /// committed code into `[0, 256)`): `f32_to_fp8_e4m3(bf16_to_f32(clamp_±448(x)))` exactly
    /// (RNE, saturation, subnormal region, signed zero) — InputQuant Q1-Q3.
    Qcast,
    /// Keyed int8 byte (raw two's complement, `[0, 255]`), tuple value
    /// `(SIGN, EXP, MANTISSA, EXP_IS_ZERO)`: the exact bf16 decode fields of the int8 value
    /// (byte 0 -> +0; byte 0x80 -> `(1, 134, 0, 0)`); the key domain doubles as the byte range
    /// proof — InputQuant P1.
    Int8Dec,
    /// Keyed denominator (noised-bound) bf16 code, value bf16 code `RNE_bf16(448 / denom)` —
    /// the alpha definition; native error paths map to sentinel codes (exponent 255) rejected
    /// by the alpha validity RCs — Scale H3.
    Div448,
    /// Keyed exponent field `[0, 254]` (the inf/NaN field 255 has **no row**), value the
    /// `EXP_IS_ZERO` flag: subnormal-flag derivation *and* the exponent range/finiteness proof
    /// via the out-of-domain pattern — every decode carrying the flag.
    ExpInfo,
    /// Keyed `x + 400` for `x in [-400, 200]`, value `clamp(x, -7, 18) + 7`: the RNERND slot
    /// (the signed subnormal fade cut, biased by +7) for MUL and FMA; the domain doubles as the
    /// key's range proof.
    Clamp22,
    /// Keyed `d in [0, 19]`, value `2^d`: the shared power-of-two table for every capped
    /// shift — the FMA's gap (capped at 19, the largest key anywhere) and round-to-odd
    /// shift, Scale's sqrt mod-16 shift, InputQuant's block split and summand-score
    /// stages (keys at most 16). The domain doubles as each consumer's range proof.
    Pow2D,
    /// Keyed `V + 2^17*CUT_DEPTH` (26 slots, one per slot = signed fade cut + 7 with cuts in
    /// `[-7, 18]`; `V in [0, 2^17)`), tuple value
    /// `(OUT_MANTISSA, WIDTH_ADJUST, OUT_IS_ZERO, OUT_EXP_IS_ZERO)`: the shared bf16 RNE
    /// back-end of every MUL and FMA — width detection, guard/round/sticky, ties-to-even,
    /// mantissa-overflow renormalization, subnormal fade and the normal/subnormal
    /// classification all in-table; cut 18 provably rounds every `V < 2^17` to zero and cut -7
    /// covers every scale with only normal small-significand results.
    RneRnd,
    /// The B200 product decode + window truncation (`matmul_b200_stark`), keyed
    /// `OPERAND_CODES_A + 2^8*OPERAND_CODES_B + 2^16*REL` with `REL in [0, 71]` slot-folded; values
    /// `(ALIGNED_LANE_TERMS, PRODUCT_BIASED_EXPONENT, OPERAND_CODES_A, OPERAND_CODES_B, BINADE)`:
    /// `ALIGNED_LANE_TERMS = ±floor(P*2^19 / 2^REL)` (P = mag_a*mag_b <= 225, sign folded, < 2^27),
    /// `PRODUCT_BIASED_EXPONENT = sh_a + sh_b + 26 in [26, 54]`
    /// the biased stored-exponent sum (sentinel 0 for zero products, with zero lane terms in
    /// every slot), `BINADE` the product's check-3 binade `floor(log2 |prod|) + 139` (0 for
    /// zero products, same in every slot) — MB13's per-lane bound. Slots `REL >= 27` hold
    /// zero lane terms; negative rel-shifts have no slot
    /// (this proves `GROUP_MAX_BIASED_EXPONENT >= PRODUCT_BIASED_EXPONENT_i`) — MB1.
    B200Align,
    /// Keyed `d in [0, 63]`, value `2^min(d, 26)` — Matmul's carry alignment (MB5): the
    /// divisor for `floor(4*GROUP_OUTPUT_SIGNIFICAND / 2^d)`; the 26-cap makes the floor total
    /// (`4*GROUP_OUTPUT_SIGNIFICAND < 2^26`), and missing negative keys prove
    /// `GROUP_MAX_BIASED_EXPONENT >= GROUP_OUTPUT_BIASED_EXPONENT`.
    Pow2Gb,
    /// Keyed `GROUP_SUM_WIDTH in [1, 32]` (a shifted ramp: 32 live rows, no key 0), tuple value
    /// `(TRUNCATION_POWER, LIFTING_POWER) = (2^max(W-24, 0), 2^max(24-W, 0))`: the
    /// truncate-to-24-bits divisor/multiplier pair. Its key domain also proves
    /// `GROUP_SUM_WIDTH <= 32`, hence `GROUP_SUM_ABS < 2^32` through MB7.
    Width32,
    /// TamedStark's squared-comparison power (J4), keyed `t = d + 1024`, decoding `D = 2*d` —
    /// the tau-folded doubled frame gap of the certificate `2^D <= Y` with
    /// `Y = k*pp^2 < 2^80` (`tau_tame^2 = 2^16` rides the key's offset). Value:
    /// `2^min(max(D,0),80)` as a base-2^16 limb vector (6 columns, exactly one limb
    /// nonzero); the cap exceeds `Y`'s width, so the saturated comparison equals the
    /// unsaturated one. The key domain doubles as the frame-gap window proof.
    XfPow2,
    /// Jackpot check 4's significand-product width, keyed `P in [0, 2^16)`, values
    /// `(width(P), [P != 0])` — the bit length of a 16-bit product and its nonzero flag.
    /// InputQuant binds it on the summand's significand product to build the lambda scores;
    /// the nonzero flag pins the zero sentinel exactly, and the key domain doubles as the
    /// product's range proof.
    Width16,
    /// Jackpot check 4's fixed-point log, keyed `n in [0, 2^16)`, value `floor(64 * log2 n)`
    /// (0 at the key-0 sentinel). InputQuant binds it on the summand's sum of squares to
    /// build the lambda scores.
    Log16,
}

#[cfg(test)]
mod tests;
