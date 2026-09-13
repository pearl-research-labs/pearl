//! Cross-table lookups of the ZK FP8 V2 multi-STARK system.
//!
//! This module pins down (a) the table indices, (b) the shared [`LutTable`]/[`LutLookup`]
//! descriptor types every AIR's LUT inventory uses, and (c) [`all_cross_table_lookups`], the
//! assembly of every channel. Each AIR's halves live next to it — `blake3_stark::ctl`,
//! `input_quant_stark::ctl`, `matmul_b200_stark::ctl`, `scale_stark::ctl` and
//! `xor_fold_stark::ctl`.
//!
//! The full channel set is eight main channels (below) plus one channel per committed LUT
//! ([`super::luts::lut_cross_table_lookups`]): each LUT is its own AIR of the batch, its
//! looking side collects every instance of the six main tables' inventories, and its looked
//! side is the LUT's slots filtered by their per-proof multiplicity columns (folded tables are
//! multi-slot looked sides). 24 [`CrossTableLookup`]s in total.
//!
//! The eight main channels are:
//!
//! - two Blake3 -> InputQuant channels: packed int8 pairs and BF16 block-scale codes;
//! - InputQuant -> Matmul: packed fp8 operand-code pairs, each element paired with its
//!   summand score `lambda` (jackpot check 4);
//! - InputQuant -> Scale: group key, L2 frame sum/exponent, max absolute value, and the
//!   liveness dead bound/count;
//! - Matmul -> XorFold: cell id and the two limbs of the final f32 result;
//! - XorFold -> Blake3: lane id and final folded word;
//! - Scale -> Tamed: per-row noise-sigma tuples (jackpot check 3), Scale's side weighted by
//!   the `w`/`h` public-input multiplicities;
//! - Matmul -> Tamed: per-cell replay-magnitude binades and skip censuses
//!   `(CELL_ID, E_CELL, CELL_SKIPS)` (jackpot checks 3 + 4).
//!
//! The first three channels contain disjoint A/B key spaces, so each uses one CTL with two
//! side-specific slots; so does the sigma channel (disjoint A/B group keys).

use plonky2::field::types::Field;
use starky::cross_table_lookup::{CrossTableLookup, TableIdx};
use starky::lookup::{Column, Filter};

use super::blake3_stark::ctl::{
    blake3_lut_lookups, ctl_block_scales_looking_blake3, ctl_int8_bytes_looking_blake3, ctl_lottery_words_looking_blake3,
};
use super::input_quant_stark::ctl::{
    ctl_block_scales_looked_input_quant, ctl_group_tuples_looking_input_quant, ctl_int8_bytes_looked_input_quant,
    ctl_operand_codes_looked_input_quant, input_quant_lut_lookups,
};
use super::luts::lut_cross_table_lookups;
use super::matmul_b200_stark::ctl::{
    ctl_cell_results_looked_matmul_b200, ctl_e_cell_looked_matmul_b200, ctl_operand_codes_looking_matmul_b200,
    matmul_b200_lut_lookups,
};
use super::scale_stark::ctl::{ctl_looked_scale_group_tuple, ctl_sigma_looked_scale, scale_lut_lookups};
use super::scale_stark::stark::ScaleProgram;
use super::tamed_stark::ctl::{ctl_e_cell_looking_tamed, ctl_sigma_looking_tamed, tamed_lut_lookups};
use super::xor_fold_stark::ctl::{ctl_cell_results_looking_xor_fold, ctl_lottery_words_looked_xor_fold, xor_fold_lut_lookups};

/// The tables of the fp8 multi-STARK system, in their fixed batch order. `Matmul` is
/// [`super::matmul_b200_stark`]'s slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    Blake3 = 0,
    InputQuant = 1,
    Scale = 2,
    Matmul = 3,
    XorFold = 4,
    Tamed = 5,
}

/// Number of fp8 main STARK tables.
pub const NUM_TABLES: usize = 6;

impl From<Table> for TableIdx {
    fn from(table: Table) -> Self {
        table as Self
    }
}

/// Number of committed LUT tables — each its own AIR of the batch (`super::luts`).
pub const NUM_LUT_TABLES: usize = 16;

/// The committed LUT tables in their canonical batch order: descending committed height
/// (`super::luts::lut_height` — the shared preprocessed tree stacks its leaves by height),
/// ties in a fixed order. Everything about a LUT's storage keys off its position here.
pub const LUT_TABLES: [LutTable; NUM_LUT_TABLES] = [
    LutTable::RneRnd,    // 2^17
    LutTable::Range16,   // 2^16
    LutTable::Bytes2,    // 2^16
    LutTable::Qcast,     // 2^16
    LutTable::Div448,    // 2^16
    LutTable::B200Align, // 2^16
    LutTable::Width16,   // 2^16
    LutTable::Log16,     // 2^16
    LutTable::Pair128,   // 2^14
    LutTable::XfPow2,    // 2^11
    LutTable::Clamp22,   // 2^10
    LutTable::Int8Dec,   // 2^8
    LutTable::ExpInfo,   // 2^8
    LutTable::Pow2Gb,    // 2^6
    LutTable::Pow2D,     // 2^5
    LutTable::Width32,   // 2^5
];

/// Total tables of the batch: the six main tables, then the family's LUTs.
pub const NUM_ALL_TABLES: usize = NUM_TABLES + NUM_LUT_TABLES;

/// The batch table index of the family's `i`-th LUT (LUTs follow the six main tables).
pub const fn lut_table_idx(i: usize) -> TableIdx {
    NUM_TABLES + i
}

/// All fp8 cross-table lookups: the eight main channels in the module-docs order (the
/// InputQuant-looked channels carry the A and B slots of one channel each), then one channel
/// per committed LUT in [`LUT_TABLES`] order. No half bakes a geometry constant
/// (InputQuant's key offsets and multiplicities are public-input terms of its CTL
/// expressions; the other tables' keys ride their class (a) schedule columns), so the CTL
/// structure is a pure function of `scale.r` — and `r` is the wire constant 32, making the
/// set a consensus constant. `scale` supplies the noise rank behind the H5 `E*(dos)` lookup
/// offset.
///
/// Balance preconditions (each documented on its halves): every table's live rows fill its
/// power-of-two height exactly — Blake3 is the exception (its padding rows are all-zero, every
/// filter off), and InputQuant's per-side liveness filters exclude each side's dead rows
/// (`h` and `w` are independent — v19). Validated end-to-end by `super::consistency` via
/// `starky::cross_table_lookup::debug_utils::check_ctls` (which reads non-binary filter values
/// as multiplicities, the prover/constraint semantics).
pub fn all_cross_table_lookups<F: Field>(scale: &ScaleProgram) -> Vec<CrossTableLookup<F>> {
    let mut ctls = vec![
        CrossTableLookup::new(ctl_int8_bytes_looking_blake3(), ctl_int8_bytes_looked_input_quant()),
        CrossTableLookup::new(ctl_block_scales_looking_blake3(), ctl_block_scales_looked_input_quant()),
        CrossTableLookup::new(
            ctl_operand_codes_looking_matmul_b200(),
            ctl_operand_codes_looked_input_quant(),
        ),
        CrossTableLookup::new(ctl_group_tuples_looking_input_quant(), vec![ctl_looked_scale_group_tuple()]),
        CrossTableLookup::new(
            vec![ctl_cell_results_looking_xor_fold()],
            vec![ctl_cell_results_looked_matmul_b200()],
        ),
        CrossTableLookup::new(ctl_lottery_words_looking_blake3(), vec![ctl_lottery_words_looked_xor_fold()]),
        CrossTableLookup::new(ctl_sigma_looking_tamed(), vec![ctl_sigma_looked_scale()]),
        CrossTableLookup::new(vec![ctl_e_cell_looking_tamed()], vec![ctl_e_cell_looked_matmul_b200()]),
    ];
    ctls.extend(lut_cross_table_lookups(
        &LUT_TABLES,
        &[
            (Table::Blake3.into(), blake3_lut_lookups()),
            (Table::InputQuant.into(), input_quant_lut_lookups()),
            (Table::Scale.into(), scale_lut_lookups(scale)),
            (Table::Matmul.into(), matmul_b200_lut_lookups()),
            (Table::XorFold.into(), xor_fold_lut_lookups()),
            (Table::Tamed.into(), tamed_lut_lookups()),
        ],
    ));
    ctls
}

// ==================================================================================================
// Committed LUT oracle instances
// ==================================================================================================

/// The consensus lookup tables (every AIR lookup, range checks included,
/// targets one of these; there are no in-trace tables). Each is
/// generated once by exhaustively evaluating the canonical Rust function it mirrors,
/// precommitted at setup time (`super::luts::lut_preprocessed_data`) and served as its own
/// AIR of the batch (`super::luts::LutStark`) through one [`CrossTableLookup`] channel.
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

/// One lookup of a STARK trace into a committed LUT: the in-trace key/value column algebra and
/// row filter. `super::luts::lut_cross_table_lookups` turns the inventories into real lookup
/// arguments — one [`CrossTableLookup`] per table, this instance on its looking side.
#[derive(Clone, Debug)]
pub struct LutLookup<F: Field> {
    pub table: LutTable,
    /// Key component expressions (one `Column` per key tuple component).
    pub keys: Vec<Column<F>>,
    /// Value-binding expressions, in the table's value-column order (empty for pure range
    /// checks, whose statement is the key's domain membership).
    pub values: Vec<Column<F>>,
    /// Row filter (degree <= 2), default = every row.
    pub filter: Filter<F>,
}

impl<F: Field> LutLookup<F> {
    /// An unfiltered 16-bit range check of a column expression.
    pub fn rc16(key: Column<F>) -> Self {
        Self {
            table: LutTable::Range16,
            keys: vec![key],
            values: vec![],
            filter: Filter::default(),
        }
    }

    /// A filtered 16-bit range check of a column expression.
    pub fn rc16_filtered(key: Column<F>, filter: Filter<F>) -> Self {
        Self {
            table: LutTable::Range16,
            keys: vec![key],
            values: vec![],
            filter,
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn all_ctls_assemble() {
        let scale = ScaleProgram::new(4, 4, 2048, 32);
        let ctls = all_cross_table_lookups::<F>(&scale);
        assert_eq!(ctls.len(), 8 + NUM_LUT_TABLES, "eight main channels + one per LUT");
    }
}
