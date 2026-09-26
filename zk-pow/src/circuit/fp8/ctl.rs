//! Cross-table lookups of the ZK FP8 V2 multi-STARK system.
//!
//! This module fixes the batch table indices and assembles every CTL channel.
//! Each AIR declares its halves in its own `ctl` module, including [`super::luts::ctl`].
//! LUT descriptors live with those tables and are re-exported here.
//!
//! The full channel set is six main channels (below) plus one channel per committed LUT
//! ([`lut_cross_table_lookups`]): each LUT is its own AIR of the batch, its
//! looking side collects every instance of the five main tables' inventories, and its looked
//! side is the LUT's slots filtered by their per-proof multiplicity columns (folded tables are
//! multi-slot looked sides). 21 [`CrossTableLookup`]s in total.
//!
//! The eight main channels are:
//!
//! - two Blake3 -> InputQuant channels: packed int8 pairs and BF16 block-scale codes;
//! - InputQuant -> Matmul: packed fp8 operand-code pairs, each element paired with its
//!   summand score `lambda` (jackpot check 4);
//! - InputQuant -> Scale: group key, L2 frame sum/exponent, max absolute value, and the
//!   liveness dead bound/count;
//! - Matmul -> XorFold: cell id, the final f32 result, and the cell skip census;
//! - XorFold -> Blake3: lane id and final folded word;
//!
//! The first three channels contain disjoint A/B key spaces, so each uses one CTL with two
//! side-specific slots; so does the sigma channel (disjoint A/B group keys).

use plonky2::field::types::Field;
use starky::cross_table_lookup::{CrossTableLookup, TableIdx, TableWithColumns};

use super::blake3_stark::columns::NUM_BLAKE3_KNOWN_COLUMNS;
use super::blake3_stark::ctl::{
    blake3_lut_lookups, ctl_block_scales_looking_blake3, ctl_int8_bytes_looking_blake3, ctl_lottery_words_looking_blake3,
};
use super::input_quant_stark::columns::NUM_INPUT_QUANT_KNOWN_COLUMNS;
use super::input_quant_stark::ctl::{
    ctl_block_scales_looked_input_quant, ctl_group_tuples_looking_input_quant, ctl_int8_bytes_looked_input_quant,
    ctl_operand_codes_looked_input_quant, input_quant_lut_lookups,
};
pub use super::luts::LutTable;
use super::luts::columns::num_slots;
pub use super::luts::ctl::LutLookup;
use super::luts::ctl::ctl_looked_lut_slot;
use super::matmul_b200_stark::columns::NUM_MATMUL_B200_KNOWN_COLUMNS;
use super::matmul_b200_stark::ctl::{
    ctl_cell_results_looked_matmul_b200, ctl_operand_codes_looking_matmul_b200, matmul_b200_lut_lookups,
};
use super::matmul_stark::columns::NUM_MATMUL_H100_KNOWN_COLUMNS;
use super::matmul_stark::ctl::{ctl_cell_results_looked_matmul, ctl_operand_codes_looking_matmul, matmul_lut_lookups};
use super::scale_stark::columns::NUM_SCALE_KNOWN_COLUMNS;
use super::scale_stark::ctl::{ctl_looked_scale_group_tuple, scale_lut_lookups};
use super::scale_stark::stark::ScaleProgram;
use super::xor_fold_stark::columns::NUM_XOR_FOLD_KNOWN_COLUMNS;
use super::xor_fold_stark::ctl::{ctl_cell_results_looking_xor_fold, ctl_lottery_words_looked_xor_fold, xor_fold_lut_lookups};
use crate::api::fp8::public_params::Device;

/// The tables of the fp8 multi-STARK system, in their fixed batch order. `Matmul` is
/// [`super::matmul_b200_stark`]'s slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    Blake3 = 0,
    InputQuant = 1,
    Scale = 2,
    Matmul = 3,
    XorFold = 4,
}

/// Number of fp8 main STARK tables.
pub const NUM_TABLES: usize = 5;

impl Table {
    /// The main tables in their fixed batch order.
    pub(crate) const ALL: [Self; NUM_TABLES] = [Self::Blake3, Self::InputQuant, Self::Scale, Self::Matmul, Self::XorFold];

    /// Number of verifier-recomputable leading columns for this table.
    pub(crate) const fn known_column_count(self, device: Device) -> usize {
        match self {
            Self::Blake3 => NUM_BLAKE3_KNOWN_COLUMNS,
            Self::InputQuant => NUM_INPUT_QUANT_KNOWN_COLUMNS,
            Self::Scale => NUM_SCALE_KNOWN_COLUMNS,
            Self::Matmul => match device {
                Device::H100 => NUM_MATMUL_H100_KNOWN_COLUMNS,
                Device::B200 => NUM_MATMUL_B200_KNOWN_COLUMNS,
            },
            Self::XorFold => NUM_XOR_FOLD_KNOWN_COLUMNS,
        }
    }
}

impl From<Table> for TableIdx {
    fn from(table: Table) -> Self {
        table as Self
    }
}

/// Number of committed LUT tables — each its own AIR of the batch (`super::luts`).
pub const NUM_LUT_TABLES: usize = 15;

/// Selects the LUT occupying a canonical batch position for `device`.
const fn lut_for_device(device: Device, h100: LutTable, b200: LutTable) -> LutTable {
    match device {
        Device::H100 => h100,
        Device::B200 => b200,
    }
}

/// The device's LUT tables in canonical batch order.
///
/// The shared prefix is hardware-independent. Every later position explicitly records
/// its H100/B200 choice, making the distinct committed orders visible without duplicating
/// two complete arrays.
pub const fn lut_tables(device: Device) -> [LutTable; NUM_LUT_TABLES] {
    [
        LutTable::RneRnd,
        LutTable::Range16,
        LutTable::Bytes2,
        LutTable::Qcast,
        LutTable::Div448,
        lut_for_device(device, LutTable::ProdAlign15, LutTable::B200Align),
        lut_for_device(device, LutTable::WidthNorm, LutTable::Width16),
        lut_for_device(device, LutTable::Width16, LutTable::Log16),
        lut_for_device(device, LutTable::Log16, LutTable::Pair128),
        lut_for_device(device, LutTable::Pair128, LutTable::Clamp22),
        lut_for_device(device, LutTable::Clamp22, LutTable::Int8Dec),
        lut_for_device(device, LutTable::Int8Dec, LutTable::ExpInfo),
        lut_for_device(device, LutTable::ExpInfo, LutTable::Pow2Gb),
        lut_for_device(device, LutTable::Pow2G, LutTable::Pow2D),
        lut_for_device(device, LutTable::Pow2D, LutTable::Width32),
    ]
}

/// Total tables of the batch: the five main tables, then the family's LUTs.
pub const NUM_ALL_TABLES: usize = NUM_TABLES + NUM_LUT_TABLES;
/// Six main-table channels plus one channel for each committed LUT.
pub const NUM_CTL_CHANNELS: usize = 6 + NUM_LUT_TABLES;

/// The batch table index of the family's `i`-th LUT (LUTs follow the five main tables).
pub const fn lut_table_idx(i: usize) -> TableIdx {
    NUM_TABLES + i
}

/// All fp8 cross-table lookups: the six main channels in the module-docs order (the
/// InputQuant-looked channels carry the A and B slots of one channel each), then one channel
/// per committed LUT in the selected device's committed order. No half bakes a geometry constant
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
pub fn all_cross_table_lookups<F: Field>(device: Device, scale: &ScaleProgram) -> Vec<CrossTableLookup<F>> {
    let (operand_codes, cell_results) = match device {
        Device::H100 => (ctl_operand_codes_looking_matmul(), ctl_cell_results_looked_matmul()),
        Device::B200 => (ctl_operand_codes_looking_matmul_b200(), ctl_cell_results_looked_matmul_b200()),
    };
    let mut ctls = vec![
        CrossTableLookup::new(ctl_int8_bytes_looking_blake3(), ctl_int8_bytes_looked_input_quant()),
        CrossTableLookup::new(ctl_block_scales_looking_blake3(), ctl_block_scales_looked_input_quant()),
        CrossTableLookup::new(operand_codes, ctl_operand_codes_looked_input_quant()),
        CrossTableLookup::new(
            ctl_group_tuples_looking_input_quant(),
            vec![ctl_looked_scale_group_tuple(scale)],
        ),
        CrossTableLookup::new(vec![ctl_cell_results_looking_xor_fold()], vec![cell_results]),
        CrossTableLookup::new(ctl_lottery_words_looking_blake3(), vec![ctl_lottery_words_looked_xor_fold()]),
    ];
    ctls.extend(lut_cross_table_lookups(
        &lut_tables(device),
        &lut_inventories(device, scale).map(|(table, lookups)| (table.into(), lookups)),
    ));
    ctls
}

/// The same per-table LUT inventories feed CTL assembly and prover multiplicity counting.
pub(crate) fn lut_inventories<F: Field>(device: Device, scale: &ScaleProgram) -> [(Table, Vec<LutLookup<F>>); NUM_TABLES] {
    let matmul_luts = match device {
        Device::H100 => matmul_lut_lookups(),
        Device::B200 => matmul_b200_lut_lookups(),
    };
    [
        (Table::Blake3, blake3_lut_lookups()),
        (Table::InputQuant, input_quant_lut_lookups()),
        (Table::Scale, scale_lut_lookups(scale)),
        (Table::Matmul, matmul_luts),
        (Table::XorFold, xor_fold_lut_lookups()),
    ]
}

/// The committed LUT channels: one [`CrossTableLookup`] per table of `tables`
/// (position `i` is batch table [`lut_table_idx`]`(i)`).
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
    use crate::api::fp8::public_params::Device;

    type F = GoldilocksField;

    #[test]
    fn all_ctls_assemble() {
        for device in [Device::H100, Device::B200] {
            let scale = ScaleProgram::new_for_device(4, 4, 2048, 32, device);
            let ctls = all_cross_table_lookups::<F>(device, &scale);
            assert_eq!(ctls.len(), NUM_CTL_CHANNELS, "six main channels + one per LUT");
        }
    }
}
