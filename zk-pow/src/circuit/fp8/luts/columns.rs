//! Column positions, key domains, and folded-slot layouts of the committed LUTs.

use plonky2::field::types::Field;
use starky::lookup::Column;

use super::LutTable;

/// Number of slots of a folded table (1 = unfolded).
pub const fn num_slots(table: LutTable) -> usize {
    match table {
        LutTable::RneRnd => 26,    // one per slot = signed cut + 7, cut in [-7, 18]
        LutTable::B200Align => 72, // one per REL in [0, 71]
        _ => 1,
    }
}

/// Live rows per slot (the key-domain size; [`lut_height`] rounds it up to the AIR height).
pub const fn slot_height(table: LutTable) -> usize {
    match table {
        LutTable::Range16 | LutTable::Bytes2 | LutTable::Qcast | LutTable::Div448 | LutTable::Width16 | LutTable::Log16 => {
            1 << 16
        }
        LutTable::Pair128 => 1 << 14,
        LutTable::XfPow2 => 1 << 11,
        LutTable::Int8Dec => 256,
        LutTable::ExpInfo => 255, // keys [0, 254]; the inf/NaN field 255 has no row
        LutTable::Clamp22 => 601, // keys x + 400, x in [-400, 200]
        LutTable::Pow2D => 20,    // keys [0, 19]; the FMA's gap cap of 19 sets the domain
        LutTable::Pow2Gb => 64,
        LutTable::Width32 => 32, // keys [1, 32] — a shifted ramp, no key 0 (see `generate`)
        LutTable::RneRnd => 1 << 17,
        LutTable::B200Align => 1 << 16,
    }
}

/// XFPOW2's saturating exponent cap (jackpot check 3, TamedStark): the table serves
/// `2^A`, `A = min(max(D, 0), 80)`, `D = 2*(E_CELL - exp(A) - exp(B) + 3949)` — the doubled
/// frame gap with `tau_tame^2 = 2^16` folded in. The compared bound `Y = k * PP^2` is below
/// `2^80`, so the cap never changes the verdict of `2^A <= Y`.
pub const XFPOW2_CAP: u64 = 80;
/// XFPOW2's key zero point: `D = 2*(key - 1024)`. The 2^11 key domain covers every reachable
/// gap: `E_CELL in [121, 172]` on keying rows and sigma exponents in `[1781, 2287]` keep the
/// key inside `[521, 1584]`.
pub const XFPOW2_ZERO_POINT: u64 = 1024;
/// XFPOW2's `2^A` value as base-2^16 limbs (`A <= 80`: 6 limbs, exactly one nonzero).
pub const XFPOW2_LIMBS: usize = 6;

/// Committed height of the table's AIR: the slot key domain rounded up to a power of two.
/// Full-height for most tables; EXPINFO (255 -> 256), CLAMP22 (601 -> 1024) and POW2D
/// (20 -> 32) pad by key saturation (module docs).
pub const fn lut_height(table: LutTable) -> usize {
    slot_height(table).next_power_of_two()
}

/// Stored key columns: the enumerated tuple for BYTES2/PAIR128, one ramp or saturated key
/// column otherwise.
const fn num_key_columns(table: LutTable) -> usize {
    match table {
        LutTable::Bytes2 | LutTable::Pair128 => 2,
        _ => 1,
    }
}

/// Stored value columns over all slots (RANGE16 and the key-tuple tables store none).
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
        LutTable::Int8Dec => 4,
        LutTable::XfPow2 => XFPOW2_LIMBS,
        LutTable::B200Align => 76, // 72 per-slot aligned + 4 shared
        LutTable::RneRnd => 104,   // 4 per slot
    }
}

/// Number of setup-time committed columns: the keys followed by the stored values.
pub const fn num_precommitted_columns(table: LutTable) -> usize {
    num_key_columns(table) + num_value_columns(table)
}

/// Total column count of the table's AIR: the precommitted block, then one per-proof
/// multiplicity column per slot.
pub const fn lut_num_columns(table: LutTable) -> usize {
    num_precommitted_columns(table) + num_slots(table)
}

/// One slot's position in its AIR: everything the CTL wiring (and the debug checker) needs to
/// serve lookups into `(table, slot)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LutSlotLayout {
    /// The additive constant of the slot's key formula (`2^17 * cut`, `2^16 * shift/slot`;
    /// 0 for unfolded tables). Looking sides fold it into their key expressions.
    pub key_offset: u64,
    /// The key column(s): the leading ramp/saturated key, or the stored key tuple.
    pub key_columns: Vec<usize>,
    /// The slot's stored value columns, in the order consumers' `LutLookup::values` bind them.
    pub value_columns: Vec<usize>,
    /// The slot's per-proof multiplicity column.
    pub multiplicity_column: usize,
}

impl LutSlotLayout {
    /// The slot's looked tuple `keys ++ values` as column expressions over the AIR's trace
    /// (the key column carries the slot's additive offset), ready for the CTL channel.
    pub fn looked_columns<F: Field>(&self) -> Vec<Column<F>> {
        let offset = F::from_canonical_u64(self.key_offset);
        let mut cols: Vec<Column<F>> = self
            .key_columns
            .iter()
            .map(|&c| Column::linear_combination_with_constant([(c, F::ONE)], offset))
            .collect();
        cols.extend(self.value_columns.iter().map(|&c| Column::single(c)));
        cols
    }
}

/// Resolves `(table, slot)` to its columns (a pure function of `num_key_columns` and the
/// slot-major value layout).
pub fn lut_slot_layout(table: LutTable, slot: usize) -> LutSlotLayout {
    assert!(slot < num_slots(table), "slot {slot} out of range for {table:?}");
    let nk = num_key_columns(table);
    let (key_offset, value_columns) = match table {
        LutTable::Range16 | LutTable::Bytes2 | LutTable::Pair128 => (0, vec![]),
        LutTable::Qcast
        | LutTable::Div448
        | LutTable::ExpInfo
        | LutTable::Clamp22
        | LutTable::Pow2D
        | LutTable::Pow2Gb
        | LutTable::Log16 => (0, vec![nk]),
        LutTable::Width32 | LutTable::Width16 => (0, vec![nk, nk + 1]),
        LutTable::Int8Dec => (0, (nk..nk + 4).collect()),
        LutTable::XfPow2 => (0, (nk..nk + XFPOW2_LIMBS).collect()),
        LutTable::B200Align => {
            let shared = nk + num_slots(table);
            (
                (slot as u64) << 16,
                // (ALIGNED_LANE_TERMS[slot], PRODUCT_BIASED_EXPONENT, OPERAND_CODES_A/B, BINADE).
                vec![nk + slot, shared, shared + 1, shared + 2, shared + 3],
            )
        }
        LutTable::RneRnd => ((slot as u64) << 17, (0..4).map(|v| nk + 4 * slot + v).collect()),
    };
    LutSlotLayout {
        key_offset,
        key_columns: (0..nk).collect(),
        value_columns,
        multiplicity_column: num_precommitted_columns(table) + slot,
    }
}
