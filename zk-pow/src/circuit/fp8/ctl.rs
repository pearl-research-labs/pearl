//! Cross-table lookups of the ZK FP8 V2 multi-STARK system.
//!
//! This module fixes the batch table indices and assembles every CTL channel.
//! Each AIR declares its halves in its own `ctl` module, including [`super::luts::ctl`].
//! LUT descriptors live with those tables and are re-exported here.
//!
//! The full channel set is eight main channels (below) plus one channel per committed LUT
//! ([`lut_cross_table_lookups`]): each LUT is its own AIR of the batch, its
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
use starky::cross_table_lookup::{CrossTableLookup, TableIdx, TableWithColumns};

use super::blake3_stark::ctl::{
    blake3_lut_lookups, ctl_block_scales_looking_blake3, ctl_int8_bytes_looking_blake3, ctl_lottery_words_looking_blake3,
};
use super::input_quant_stark::ctl::{
    ctl_block_scales_looked_input_quant, ctl_group_tuples_looking_input_quant, ctl_int8_bytes_looked_input_quant,
    ctl_operand_codes_looked_input_quant, input_quant_lut_lookups,
};
pub use super::luts::LutTable;
use super::luts::columns::num_slots;
pub use super::luts::ctl::LutLookup;
use super::luts::ctl::ctl_looked_lut_slot;
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
        &lut_inventories(scale).map(|(table, lookups)| (table.into(), lookups)),
    ));
    ctls
}

/// The same per-table LUT inventories feed CTL assembly and prover multiplicity counting.
pub(crate) fn lut_inventories<F: Field>(scale: &ScaleProgram) -> [(Table, Vec<LutLookup<F>>); NUM_TABLES] {
    [
        (Table::Blake3, blake3_lut_lookups()),
        (Table::InputQuant, input_quant_lut_lookups()),
        (Table::Scale, scale_lut_lookups(scale)),
        (Table::Matmul, matmul_b200_lut_lookups()),
        (Table::XorFold, xor_fold_lut_lookups()),
        (Table::Tamed, tamed_lut_lookups()),
    ]
}

/// The committed LUT channels: one [`CrossTableLookup`] per table of `tables`
/// ([`LUT_TABLES`] — position `i` is batch table [`lut_table_idx`]`(i)`).
/// `inventories` are the main tables' LUT instance inventories at their batch indices; each
/// channel's looking side collects every instance of its table across all of them (tuple
/// `keys ++ values` over the consumer's trace, the instance's filter), and its looked side is
/// the table's slots — a multi-slot looked side for the folded tables, with looked-side helper
/// columns handled by starky.
///
/// Every table must have at least one looking instance (a consumerless LUT signals a wiring
/// bug); width consistency of every half is asserted by `CrossTableLookup::new`.
pub fn lut_cross_table_lookups<F: Field>(
    tables: &[LutTable; NUM_LUT_TABLES],
    inventories: &[(TableIdx, Vec<LutLookup<F>>)],
) -> Vec<CrossTableLookup<F>> {
    tables
        .iter()
        .enumerate()
        .map(|(i, &table)| {
            let looking: Vec<TableWithColumns<F>> = inventories
                .iter()
                .flat_map(|(table_idx, lookups)| {
                    let table_idx = *table_idx;
                    lookups.iter().filter(|l| l.table == table).map(move |l| {
                        TableWithColumns::new(table_idx, l.keys.iter().chain(&l.values).cloned().collect(), l.filter.clone())
                    })
                })
                .collect();
            assert!(!looking.is_empty(), "{table:?} has no looking instances");
            let looked = (0..num_slots(table))
                .map(|slot| ctl_looked_lut_slot(lut_table_idx(i), table, slot))
                .collect();
            CrossTableLookup::new(looking, looked)
        })
        .collect()
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
