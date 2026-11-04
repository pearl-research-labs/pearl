//! Column positions, key domains, and folded-slot layouts of the committed LUTs.

use plonky2::field::types::Field;
use starky::lookup::Column;

use super::LutTable;

/// Number of slots of a folded table (1 = unfolded).
pub const fn num_slots(table: LutTable) -> usize {
    match table {
        LutTable::RneRnd => 26,      // one per slot = signed cut + 7, cut in [-7, 18]
        LutTable::ProdAlign15 => 59, // one per SHIFT in [0, 58]
        LutTable::B200Align => 72,   // one per REL in [0, 71]
        LutTable::WidthNorm => 16,   // GROUP_SUM_ABS in [0, 2^20)
        LutTable::Pow2G => 4,        // carry, two gap ranges, normalization
        LutTable::Xor8 => 3,         // plain, rotate-by-12 and rotate-by-7 fragments
        _ => 1,
    }
}

/// Live rows per slot (the key-domain size; [`lut_height`] rounds it up to the AIR height).
pub const fn slot_height(table: LutTable) -> usize {
    match table {
        LutTable::Range16
        | LutTable::Bytes2
        | LutTable::Xor8
        | LutTable::Qcast
        | LutTable::Div448
        | LutTable::Width16
        | LutTable::Log16 => 1 << 16,
        LutTable::Pair128 => 1 << 14,
        LutTable::Int8Dec => 256,
        LutTable::ExpInfo => 255, // keys [0, 254]; the inf/NaN field 255 has no row
        LutTable::Clamp22 => 601, // keys x + 400, x in [-400, 200]
        LutTable::Pow2D => 20,    // keys [0, 19]; the FMA's gap cap of 19 sets the domain
        LutTable::Pow2G | LutTable::Pow2Gb => 64,
        LutTable::Width32 => 32, // keys [1, 32] — a shifted ramp, no key 0 (see `generate`)
        LutTable::RneRnd => 1 << 17,
        LutTable::ProdAlign15 | LutTable::B200Align | LutTable::WidthNorm => 1 << 16,
    }
}

/// Committed height of the table's AIR: the slot key domain rounded up to a power of two.
/// Full-height for most tables; EXPINFO (255 -> 256), CLAMP22 (601 -> 1024) and POW2D
/// (20 -> 32) pad by key saturation (module docs).
pub const fn lut_height(table: LutTable) -> usize {
    slot_height(table).next_power_of_two()
}

/// Stored key columns: the enumerated tuple for BYTES2/PAIR128/XOR8, one ramp or saturated
/// key column otherwise.
const fn num_key_columns(table: LutTable) -> usize {
    match table {
        LutTable::Bytes2 | LutTable::Pair128 | LutTable::Xor8 => 2,
        _ => 1,
    }
}

/// Additional fixed columns, including Pow2G's two saturated keys.
const fn num_value_columns(table: LutTable) -> usize {
    match table {
        LutTable::Range16 | LutTable::Bytes2 | LutTable::Pair128 => 0,
        LutTable::Qcast
        | LutTable::Div448
        | LutTable::ExpInfo
        | LutTable::Clamp22
        | LutTable::Pow2D
        | LutTable::Pow2Gb
        | LutTable::Log16 => 1,
        LutTable::Width32 | LutTable::Width16 => 2,
        LutTable::Xor8 => 3, // x ^ y and its two high fragments
        LutTable::Int8Dec => 4,
        LutTable::WidthNorm => 6, // slot-zero outputs and four right-shifted ramps
        LutTable::Pow2G => 7,     // two saturated keys and five values
        LutTable::ProdAlign15 | LutTable::B200Align => 11, // eight truncations and three decode fields
        LutTable::RneRnd => 75,   // independent rounding outputs
    }
}

/// Number of setup-time committed columns, shared by all folded slots.
pub const fn num_precommitted_columns(table: LutTable) -> usize {
    num_key_columns(table) + num_value_columns(table)
}

/// Total column count of the table's AIR: the precommitted block, then one per-proof
/// multiplicity column per slot.
pub const fn lut_num_columns(table: LutTable) -> usize {
    num_precommitted_columns(table) + num_slots(table)
}

/// One folded slot's keys, value expressions, and per-proof multiplicity.
#[derive(Clone, Debug)]
pub struct LutSlotLayout<F: Field> {
    /// Constant added to each key column to distinguish folded slots.
    pub key_offset: u64,
    /// Stored key columns; small tables may use a saturated ramp.
    pub key_columns: Vec<usize>,
    /// Value expressions in the order consumers bind them.
    pub value_columns: Vec<Column<F>>,
    /// The slot's per-proof multiplicity column.
    pub multiplicity_column: usize,
}

impl<F: Field> LutSlotLayout<F> {
    /// The slot's looked tuple, with its key offset included.
    pub fn looked_columns(&self) -> Vec<Column<F>> {
        let offset = F::from_canonical_u64(self.key_offset);
        let mut cols: Vec<Column<F>> = self
            .key_columns
            .iter()
            .map(|&c| Column::linear_combination_with_constant([(c, F::ONE)], offset))
            .collect();
        cols.extend(self.value_columns.iter().cloned());
        cols
    }
}

/// Independent RNERND outputs, in stored-column order after the key.
pub(crate) fn rnernd_stored_values() -> Vec<(usize, usize)> {
    (0..26)
        .flat_map(|slot| (0..4).map(move |value| (slot, value)))
        .filter(|&(slot, value)| {
            !matches!(
                (slot, value),
                (0, 3) | (1..=8, 2) | (17, 3) | (18..=24, 1 | 3) | (24, 2) | (25, _)
            )
        })
        .collect()
}

fn rnernd_value_columns<F: Field>(slot: usize) -> Vec<Column<F>> {
    let stored = rnernd_stored_values();
    let column = |slot, value| 1 + stored.iter().position(|&source| source == (slot, value)).unwrap();
    (0..4)
        .map(|value| match (slot, value) {
            (0, 3) | (1..=7, 2) => Column::single(column(0, 2)),
            (8, 2) => Column::single(column(1, 3)),
            // At cut 10, WIDTH_ADJUST is 144 for normal results and 0 for zero/subnormal results.
            (17, 3) => Column::linear_combination_with_constant([(column(17, 1), -F::from_canonical_u64(144).inverse())], F::ONE),
            (18..=24, 1) | (25, 0 | 1) => Column::constant(F::ZERO),
            (18..=25, 3) | (25, 2) => Column::constant(F::ONE),
            (24, 2) => Column::linear_combination_with_constant([(column(24, 0), -F::ONE)], F::ONE),
            _ => Column::single(column(slot, value)),
        })
        .collect()
}

/// Resolves a slot to affine expressions over its table's fixed columns.
pub fn lut_slot_layout<F: Field>(table: LutTable, slot: usize) -> LutSlotLayout<F> {
    assert!(slot < num_slots(table), "slot {slot} out of range for {table:?}");
    let nk = num_key_columns(table);
    let mut key_columns = (0..nk).collect();
    let (key_offset, value_columns) = match table {
        LutTable::Range16 | LutTable::Bytes2 | LutTable::Pair128 => (0, vec![]),
        LutTable::Qcast
        | LutTable::Div448
        | LutTable::ExpInfo
        | LutTable::Clamp22
        | LutTable::Pow2D
        | LutTable::Pow2Gb
        | LutTable::Log16 => (0, vec![Column::single(nk)]),
        LutTable::Width32 | LutTable::Width16 => (0, Column::singles([nk, nk + 1]).collect()),
        LutTable::Int8Dec => (0, Column::singles(nk..nk + 4).collect()),
        // Columns 2, 3, 4 hold x ^ y, (x ^ y) >> 4, (x ^ y) >> 7; the constant tag pins the slot.
        LutTable::Xor8 => {
            let high = if slot == 0 {
                Column::constant(F::ZERO)
            } else {
                Column::single(nk + slot)
            };
            (
                0,
                vec![Column::single(nk), high, Column::constant(F::from_canonical_usize(slot))],
            )
        }
        LutTable::ProdAlign15 | LutTable::B200Align => {
            let shift = if table == LutTable::ProdAlign15 { 7 } else { 19 };
            let aligned = if slot <= shift {
                Column::linear_combination([(1, F::from_canonical_u64(1 << (shift - slot)))])
            } else if slot <= shift + 7 {
                Column::single(1 + slot - shift)
            } else {
                Column::constant(F::ZERO)
            };
            // key = a + 256*b. Columns 9, 10, 11 hold exponent, a, and product binade.
            let byte_inverse = F::from_canonical_u64(256).inverse();
            (
                (slot as u64) << 16,
                vec![
                    aligned,
                    Column::single(9),
                    Column::single(10),
                    Column::linear_combination([(0, byte_inverse), (10, -byte_inverse)]),
                    Column::single(11),
                ],
            )
        }
        LutTable::WidthNorm => {
            let values = if slot == 0 {
                Column::singles([1, 2]).collect()
            } else {
                let width = slot.ilog2() as usize + 1;
                let shift = width + 2;
                vec![
                    Column::constant(F::from_canonical_usize(16 + width)),
                    Column::linear_combination_with_constant([(shift, F::ONE)], F::from_canonical_usize(slot << (16 - shift))),
                ]
            };
            ((slot as u64) << 16, values)
        }
        // Flag ranges 0, 2..=3, and 4..=5 separate operations even for forged keys.
        LutTable::Pow2G => match slot {
            0 => (0, vec![Column::single(3), Column::constant(F::ZERO)]),
            1 => (
                128,
                vec![
                    Column::single(4),
                    Column::linear_combination_with_constant([(5, F::ONE)], F::TWO),
                ],
            ),
            2 => {
                key_columns = vec![1];
                (
                    192,
                    vec![Column::constant(F::ONE), Column::constant(F::from_canonical_u64(3))],
                )
            }
            3 => {
                key_columns = vec![2];
                (
                    256,
                    vec![
                        Column::single(6),
                        Column::linear_combination_with_constant([(7, F::ONE)], F::from_canonical_u64(4)),
                    ],
                )
            }
            _ => unreachable!(),
        },
        LutTable::RneRnd => ((slot as u64) << 17, rnernd_value_columns(slot)),
    };
    LutSlotLayout {
        key_offset,
        key_columns,
        value_columns,
        multiplicity_column: num_precommitted_columns(table) + slot,
    }
}
