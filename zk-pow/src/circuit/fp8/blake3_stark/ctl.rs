//! CTL and LUT wiring for Blake3Stark.
//!
//! Int8 and scale message rows emit four keyed byte pairs from `ctl_key_base` and
//! `uint8_data`. The jackpot load row emits 16 `blake3_msg` words. Every row checks four byte
//! pairs, and the MoE index limbs receive unfiltered RC16 bounds.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::{LutLookup, LutTable, Table};
use super::columns::BLAKE3_COL_MAP;

/// The shared shape of both message channels: 4 byte-pair instances per message row,
/// `(CTL_KEY_BASE + key_stride*j, UINT8_DATA[2j] + 2^8*UINT8_DATA[2j+1])` for `j in 0..4`, under a
/// message-kind filter column.
fn msg_pair_tables<F: Field>(filter_col: usize, key_stride: usize) -> Vec<TableWithColumns<F>> {
    let m = &BLAKE3_COL_MAP;
    let byte_shift = F::from_canonical_u64(1 << 8);
    (0..4)
        .map(|j| {
            TableWithColumns::new(
                Table::Blake3.into(),
                vec![
                    Column::linear_combination_with_constant([(m.ctl_key_base, F::ONE)], F::from_canonical_usize(key_stride * j)),
                    Column::linear_combination([(m.uint8_data[2 * j], F::ONE), (m.uint8_data[2 * j + 1], byte_shift)]),
                ],
                Filter::from_column(Column::single(filter_col)),
            )
        })
        .collect()
}

/// Exports four int8 byte pairs per live message row, keyed by the first element.
/// A/B keys are disjoint through the B offset h*k. Auxiliary and padding bytes do
/// not cross. BYTES2 bounds each byte, making the pair packing unique.
pub fn ctl_int8_bytes_looking_blake3<F: Field>() -> Vec<TableWithColumns<F>> {
    msg_pair_tables(BLAKE3_COL_MAP.is_int8_message, 2)
}

/// Exports four little-endian BF16 scales per live scale-message row.
/// Keys count blocks; B starts at h*k/8. Each opened scale crosses once.
pub fn ctl_block_scales_looking_blake3<F: Field>() -> Vec<TableWithColumns<F>> {
    msg_pair_tables(BLAKE3_COL_MAP.is_scale_message, 1)
}

/// Binds the 16 lottery message words to XorFold's lane outputs on the load row.
/// The filter requires both the lottery flag and compression start. The lottery
/// flag spans all eight rows so it also enables the final hash binding.
pub fn ctl_lottery_words_looking_blake3<F: Field>() -> Vec<TableWithColumns<F>> {
    let m = &BLAKE3_COL_MAP;
    (0..16)
        .map(|j| {
            TableWithColumns::new(
                Table::Blake3.into(),
                vec![Column::constant(F::from_canonical_usize(j)), Column::single(m.blake3_msg[j])],
                Filter::new(
                    vec![(Column::single(m.is_bind_jackpot_hash), Column::single(m.is_new_blake))],
                    vec![],
                ),
            )
        })
        .collect()
}

/// Byte ranges, routing-index limb bounds and MoE order-chain limb bounds.
/// CV routing is an in-trace lookup defined by `Blake3Stark::lookups`.
///
/// Routing limbs need unfiltered `RC16(limb)` and `RC16(8*limb)`. The first prevents
/// field wrap; the second narrows the value to 13 bits. A scaled check alone admits
/// large field aliases. Disabling checks on unsampled slots would also let their
/// limbs absorb changes to a sampled slot in the packed equality.
pub fn blake3_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &BLAKE3_COL_MAP;
    let eight = F::from_canonical_u64(8);

    let mut lookups = Vec::new();
    // BYTES2(UINT8_DATA[2j], UINT8_DATA[2j+1]) x4, every row (padding rows carry
    // zero bytes, which are in-domain). These byte ranges are also what make the byte-pair
    // CTL packings and the buffer-word packings (constraint 5) sound.
    for j in 0..4 {
        lookups.push(LutLookup {
            table: LutTable::Bytes2,
            keys: vec![Column::single(m.uint8_data[2 * j]), Column::single(m.uint8_data[2 * j + 1])],
            values: vec![],
            filter: Filter::default(),
        });
    }
    // Both checks are required on every routing-index limb, including unsampled slots.
    for limb in [
        m.outer_index_first[0],
        m.outer_index_first[1],
        m.outer_index_second[0],
        m.outer_index_second[1],
    ] {
        lookups.push(LutLookup::rc16(Column::single(limb)));
        lookups.push(LutLookup::rc16(Column::linear_combination([(limb, eight)])));
    }
    // RC16 x4, unfiltered — the MoE order-chain limbs (constraint 7). The bounds force
    // each gated difference into [0, 2^32), so a wrapped negative (~2^64) can never
    // satisfy the chain equality. Rows with every gate off hold zeros, which are
    // in-domain.
    for limbs in [m.chain_intra_limbs, m.chain_inter_limbs] {
        lookups.push(LutLookup::rc16(Column::single(limbs[0])));
        lookups.push(LutLookup::rc16(Column::single(limbs[1])));
    }
    lookups
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn blake3_ctl_halves_are_well_formed() {
        assert_eq!(ctl_int8_bytes_looking_blake3::<F>().len(), 4);
        assert_eq!(ctl_block_scales_looking_blake3::<F>().len(), 4);
        assert_eq!(ctl_lottery_words_looking_blake3::<F>().len(), 16);
    }

    #[test]
    fn blake3_lut_inventory_matches_documented_counts() {
        let lookups = blake3_lut_lookups::<F>();
        let count = |t: LutTable| lookups.iter().filter(|l| l.table == t).count();
        // Documented inventory: BYTES2 x4 (byte pairs), RC16 x8 (outer-index limb bounds,
        // an unshifted + scaled pair per limb — see `blake3_lut_lookups` on aliasing),
        // RC16 x4 (order-chain limbs).
        assert_eq!(count(LutTable::Bytes2), 4);
        assert_eq!(count(LutTable::Range16), 12);
        assert_eq!(lookups.len(), 16);
    }
}
