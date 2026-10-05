//! CTL and LUT wiring for Blake3Stark.
//!
//! Int8 and scale message rows emit four keyed byte pairs from `ctl_key_base` and
//! `uint8_data`. The jackpot load row emits 16 `blake3_msg` words. Every row checks four byte
//! pairs, and the MoE index limbs receive unfiltered RC16 bounds.

use plonky2::field::types::Field;
use starky::cross_table_lookup::TableWithColumns;
use starky::lookup::{Column, Filter};

use super::super::ctl::Table;
use super::super::luts::LutTable;
use super::super::luts::ctl::LutLookup;
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

/// Blake3's looking side of the **strip int8 bytes** channel: 4 byte-pair
/// instances per values-message row, `(CTL_KEY_BASE + 2j, UINT8_DATA[2j] + 2^8*UINT8_DATA[2j+1])`
/// for `j in 0..4` — two consecutive int8 elements per tuple (raw bytes; InputQuant's INT8DEC
/// does the two's-complement decode). Filtered by `IS_INT8_MESSAGE`, set only on opened-strip
/// values rows of either side — auxiliary blocks and parents don't cross; every strip element
/// crosses exactly once. One channel serves both sides: the looked side holds InputQuant's A
/// and B column groups as two slots, and `CTL_KEY_BASE` carries the `h*k` B-plane key offset, so
/// the two sides' key spaces are disjoint.
///
/// The pair packing is sound because each byte is individually BYTES2-checked
/// ([`blake3_lut_lookups`]).
///
/// Looked side: `input_quant_stark::ctl::ctl_int8_bytes_looked_input_quant` (slots A, B).
pub fn ctl_int8_bytes_looking_blake3<F: Field>() -> Vec<TableWithColumns<F>> {
    msg_pair_tables(BLAKE3_COL_MAP.is_int8_message, 2)
}

/// Blake3's looking side of the **block scales** channel: 4 instances per
/// scales-message row, `(CTL_KEY_BASE + j, UINT8_DATA[2j] + 2^8*UINT8_DATA[2j+1])` for `j in 0..4`
/// — one LE bf16 block-scale code per tuple, keyed by block index. Filtered by
/// `IS_SCALE_MESSAGE` (opened-strip scales rows of either side; B keys carry the `h*k/8` offset
/// in `CTL_KEY_BASE`); every block scale crosses exactly once.
///
/// Looked side: `input_quant_stark::ctl::ctl_block_scales_looked_input_quant` (slots A, B).
pub fn ctl_block_scales_looking_blake3<F: Field>() -> Vec<TableWithColumns<F>> {
    msg_pair_tables(BLAKE3_COL_MAP.is_scale_message, 1)
}

/// Blake3's looking side of the **lottery words** channel: 16
/// instances `(word_pos, FOLD_OUT)` on the lottery message-load row — the 16 LE u32 words of
/// the 64-byte lottery block, read from `BLAKE3_MSG` (row 0 of a compression holds the message
/// itself). Filter `IS_BIND_JACKPOT_HASH * IS_NEW_BLAKE`, a degree-2 product of committed flags
/// (`IS_BIND_JACKPOT_HASH` spans all 8 lottery rows precisely so this product fires exactly once).
///
/// Looked side: `xor_fold_stark::ctl::ctl_lottery_words_looked_xor_fold` (filter
/// `IS_LANE_FINAL`). The FP8 V2 batch assembler pairs both sides, proving that the 16 jackpot
/// message words are exactly XorFold's 16 lane outputs; the public `HASH_JACKPOT` then binds
/// their keyed BLAKE3 digest.
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

/// Blake3Stark's per-row LUT instance inventory: BYTES2 x4 (every message
/// byte pair), RC16 x8 (the four 13-bit outer-index limb bounds, each as an unshifted +
/// shifted pair, **unfiltered** — every row), RC16 x4 (the MoE order-chain limbs, unfiltered).
/// The CV-routing lookup is *not* a LUT instance — it is in-trace with both sides in this
/// table and lives in `Blake3Stark::lookups` (already wired and proven).
///
/// The limb bounds must be unfiltered, exactly the deployed chip's unfiltered `URANGE13`
/// lookups: constraint 6 decodes the known packed word through the limbs *unconditionally*,
/// and an unpinned slot (selector down — an unsampled neighbor, or any non-routing row) is
/// sound only because its limbs are still range-checked and therefore forced to zero by the
/// unique decomposition. Filtering by the selectors would let a malicious prover put
/// arbitrary field elements in the dark slots' limbs and satisfy the packed equality with a
/// wrap, breaking the pinned slot's binding. Off-routing rows all four limbs are zero, which
/// is in-domain, so the unfiltered instances cost nothing.
///
/// Each 13-bit bound needs **both** RC16s. `RC16(8 * LIMB)` alone does not bound `LIMB` in
/// Goldilocks: `p = 1 (mod 8)`, so every `v < 2^16` has aliases `LIMB = (v + k*p) / 8`
/// (`k in 1..8`, huge canonical values) whose scaled key still lands in `[0, 2^16)` — with
/// only the scaled check, `2^16 * LIMB_1` can contribute e.g. `2^23 * t (mod p)` to the
/// packed word and shift a *pinned* slot's decoded index by `-2^23 * t` while an unpinned
/// neighbor slot absorbs the difference. `RC16(LIMB)` first pins `LIMB < 2^16` (so `8 * LIMB`
/// cannot wrap), and `RC16(8 * LIMB)` then gives the true 13-bit bound.
pub fn blake3_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &BLAKE3_COL_MAP;
    let eight = F::from_canonical_u64(8);

    let mut lookups = Vec::new();
    // ---- BYTES2(UINT8_DATA[2j], UINT8_DATA[2j+1]) x4, every row (padding rows carry
    // ---- zero bytes, which are in-domain). These byte ranges are also what make the byte-pair
    // ---- CTL packings and the buffer-word packings (constraint 5) sound.
    for j in 0..4 {
        lookups.push(LutLookup {
            table: LutTable::Bytes2,
            keys: vec![Column::single(m.uint8_data[2 * j]), Column::single(m.uint8_data[2 * j + 1])],
            values: vec![],
            filter: Filter::default(),
        });
    }
    // ---- RC16(limb) + RC16(8 * limb) x4, unfiltered — the alias-free 13-bit bounds that make
    // ---- the packed outer-index decomposition (constraint 6) unique on every row (module
    // ---- docs above: neither check bounds the limb alone).
    for limb in [
        m.outer_index_first[0],
        m.outer_index_first[1],
        m.outer_index_second[0],
        m.outer_index_second[1],
    ] {
        lookups.push(LutLookup::rc16(Column::single(limb)));
        lookups.push(LutLookup::rc16(Column::linear_combination([(limb, eight)])));
    }
    // ---- RC16 x4, unfiltered — the MoE order-chain limbs (constraint 7). The bounds force
    // ---- each gated difference into [0, 2^32), so a wrapped negative (~2^64) can never
    // ---- satisfy the chain equality. Rows with every gate off hold zeros, which are
    // ---- in-domain.
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
