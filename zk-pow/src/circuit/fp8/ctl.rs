//! Cross-table channels and fixed table order for the FP8 batch.
//!
//! A cross-table lookup (CTL) equates multisets of tuples. The looking side
//! requests tuples; the looked side supplies them. Filters determine which rows
//! participate and how many times each tuple is counted. These directions need
//! not match the direction in which witness generation passes data.
//!
//! The main channels bind the following data:
//!
//! | Producer | Consumer | Bound data |
//! | --- | --- | --- |
//! | Blake3 | InputQuant | Packed int8 pairs and BF16 block scales |
//! | InputQuant | Matmul | FP8 operand pairs and summand scores |
//! | InputQuant | Scale | Group norms and liveness counts |
//! | Matmul | XorFold | Final f32 cell results |
//! | XorFold | Blake3 | Folded lottery words |
//! | Scale | Tamed | Row and column noise scales |
//! | Matmul | Tamed | Cell magnitude bounds and skip counts |
//!
//! Each committed lookup table (LUT) adds one channel. Its looked side carries
//! per-proof multiplicities: the number of times consumers request each entry.
//! A and B slots share channels but use disjoint keys, so one side cannot satisfy
//! the other side's requests. [`Table`] fixes the batch indices used by all channels.

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

/// Main channels in the order above, followed by LUT channels in [`LUT_TABLES`] order.
/// Geometry enters through public inputs and verifier-known columns; it does not change
/// the channel structure. `scale.r` supplies the noise-rank exponent for Scale's lookups.
/// Each channel's filters exclude padding and inactive A/B rows.
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

// Committed LUT oracle instances

/// Fixed lookup tables, committed at setup by [`super::luts::lut_preprocessed_data`].
/// Each table has one AIR and one cross-table channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LutTable {
    /// 16-bit range check (the ramp itself is the key) — every `RC16` instance.
    Range16,
    /// Paired 8-bit range check; a lone byte checks `(d, 0)` — Blake message bytes.
    Bytes2,
    /// Paired 7-bit range check: all 2^14 pairs.
    Pair128,
    /// BF16 code -> FP8 E4M3 code after clamping to ±448 and round-to-nearest-even.
    /// Includes subnormals and signed zero.
    Qcast,
    /// Raw int8 byte -> `(sign, exponent, mantissa, exponent_is_zero)` of its BF16 value.
    /// The key domain also checks that the byte is in `[0, 255]`.
    Int8Dec,
    /// BF16 denominator -> `RNE_bf16(448 / denominator)`. Invalid divisions produce
    /// exponent-255 sentinels, rejected by Scale's alpha range checks.
    Div448,
    /// Exponent in `[0, 254]` -> zero-exponent flag. No row for 255: membership
    /// also excludes infinities and NaNs.
    ExpInfo,
    /// Key `x + 400`, `x in [-400, 200]` -> `clamp(x, -7, 18) + 7`, the RneRnd slot.
    Clamp22,
    /// Shift `d in [0, 19]` -> `2^d`. The domain bounds the shift as well as its power.
    Pow2D,
    /// Rounding key `V + 2^17*cut_slot` -> `(mantissa, width_adjust, is_zero, exp_is_zero)`.
    /// `V < 2^17` prevents aliasing between slots; `cut_slot = cut + 7`, `cut in [-7, 18]`.
    /// Implements BF16 ties-to-even, overflow renormalization and subnormal rounding.
    /// Cut 18 rounds every allowed V to zero; cut -7 covers normal small significands.
    RneRnd,
    /// Key `code_a + 2^8*code_b + 2^16*shift`, `shift in [0, 71]`, binds both codes,
    /// the aligned signed product, its biased exponent and its magnitude binade.
    /// The aligned term is `±floor(P*2^19 / 2^shift)`, `P <= 225`; zero products
    /// use exponent/binade sentinel 0. Shifts >= 27 yield zero terms.
    /// Negative shifts have no slot, proving the anchor is at least the product exponent.
    B200Align,
    /// Carry shift `d in [0, 63]` -> `2^min(d, 26)`.
    /// Capping is exact for floor division because the dividend is below `2^26`.
    /// No negative keys: the group anchor must be at least the carry exponent.
    Pow2Gb,
    /// Width `W in [1, 32]` -> `(2^max(W-24, 0), 2^max(24-W, 0))`.
    /// These powers truncate or lift to 24 bits. Together with MB7's remainder bounds,
    /// the key domain proves the group sum is below `2^32`.
    Width32,
    /// Key `d + 1024` -> six base-2^16 limbs of `2^min(max(2*d, 0), 80)`.
    /// The key includes the fixed tau offset. Tamed compares this power against
    /// `Y = k*(sigma_sig_a*sigma_sig_b)^2 < 2^80`, so saturation preserves the result.
    XfPow2,
    /// 16-bit significand product -> `(bit_width, is_nonzero)`. The key domain
    /// bounds the product; the flag fixes the zero sentinel in the summand score.
    Width16,
    /// `n in [0, 2^16)` -> `floor(64*log2(n))`, with zero sentinel at n = 0.
    /// Used for the summand scores.
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
