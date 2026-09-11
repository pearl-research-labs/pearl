//! Trace columns for eight-row BLAKE3 compressions.
//!
//! Each compression streams eight bytes per row, runs seven rounds, and finalizes
//! its chaining value on row 7. The message buffer binds the byte stream to the
//! round messages; a lookup routes chaining values between compressions.
//! Leading verifier-known columns fix the schedule, lookup keys and MoE word pins.

use crate::circuit::fp8::columns_view::columns_view;

/// One BLAKE3 state: words 0..4 and 8..12 are packed u32 values; words 4..8 and
/// 12..16 are bits for the G-function's XOR/rotate steps.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Blake3StateCols<T: Copy> {
    /// State words `v[0..4]`, packed u32.
    pub a_words: [T; 4],
    /// State words `v[4..8]`, 32 LE bits each.
    pub b_bits: [[T; 32]; 4],
    /// State words `v[8..12]`, packed u32.
    pub c_words: [T; 4],
    /// State words `v[12..16]`, 32 LE bits each.
    pub d_bits: [[T; 32]; 4],
}

/// Columns per tracked state.
pub const BLAKE3_STATE_WIDTH: usize = size_of::<Blake3StateCols<u8>>();

/// Number of tracked states per row (input + 3 intermediates; the 4th intermediate is the next
/// row's input state).
pub const NUM_TRACKED_STATES: usize = 4;

/// Number of message bytes ingested per row.
pub const NUM_UINT8: usize = 8;

/// Number of packed flags; [`unpack_flag_cols`] gives their bit order.
pub const NUM_UNPACK_FLAGS: usize = 25;

/// View of one Blake3Stark trace row: control unpacking, routing-index limbs, row counter,
/// streamed bytes, message buffer/schedule, CV selection/routing, four 264-column tracked
/// states, verifier-known schedule columns, and public-root selectors. Constraint numbers
/// 1..7 refer to `super::stark`'s constraint groups.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Blake3ColumnsView<T: Copy> {
    // Verifier-known schedule columns; see `known_values`.
    /// Packed schedule flags, with bit order given by [`unpack_flag_cols`].
    pub row_flags_packed: T,
    /// Flat element index (int8-plane rows) or block index (scales-plane rows) of the first item
    /// ingested on this row; base of the element/scale CTL keys (B-plane bases carry the
    /// `h*k` / `h*k/8` key offsets). 0 off message rows.
    pub ctl_key_base: T,
    /// 1 on live int8-values message rows of either side; filter of the int8-bytes CTL channel.
    /// (One flag serves both sides: the channel's looked side holds the A and B column groups
    /// of InputQuant as two slots of one [`starky::cross_table_lookup::CrossTableLookup`], and
    /// the `CTL_KEY_BASE` keys carry the `h*k` B-plane offset, so the key spaces are disjoint.)
    pub is_int8_message: T,
    /// 1 on live bf16-scales message rows of either side; filter of the block-scales CTL
    /// channel (B keys carry the `h*k/8` offset).
    pub is_scale_message: T,
    /// CV-routing source pointer / packed tweak, by row position within the compression:
    /// on CV-fetch rows (`IS_CV_IN`) the `TRACE_ROW_INDEX` of the earlier row whose `CV_OUT` feeds
    /// this row's `CV_IN`; on row 1 of each compression the packed tweak
    /// `counter(48 bits) + 2^48*flags + 2^56*block_len` consumed by the row-0 init-state check;
    /// 0 elsewhere.
    pub cv_route_key_or_tweak: T,
    /// Packed MoE outer indices, `OUTER_INDEX_FIRST + 2^26 * OUTER_INDEX_SECOND` with each
    /// index < 2^26. Selective pinning: the word carries a
    /// nonzero index only in the *sampled* slots of routing rows
    /// ([`Blake3Program::routing_pins`](super::stark::Blake3Program)); unsampled neighbor
    /// slots and non-routing rows contribute zero, which the unconditional decomposition
    /// (constraint 6) plus the unfiltered limb bounds force onto the limbs.
    pub moe_outer_indices_packed: T,
    /// MoE pinned value for the row's first ingested word (constraint 7). On offsets rows
    /// it holds one of the public offsets or a padding zero, and `IS_WORD_PIN_FIRST` checks
    /// equality; on the routing bound row it holds `m - 1`, and `IS_BOUND_FIRST` checks an
    /// upper bound. 0 when neither flag is set.
    pub word_pin_first: T,
    /// As `word_pin_first` for the row's second ingested word.
    pub word_pin_second: T,

    // Main: committed unpack of ROW_FLAGS_PACKED (constraint 2). Individually usable in constraints
    // and lookup/CTL filters. Bit order = field order (see `NUM_UNPACK_FLAGS`).
    /// CV-source selector: keyed compression under `KEY_A` (the A-side/routing plane and its
    /// parents). Bit 0.
    pub is_use_key_a: T,
    /// CV-source selector: keyed compression under `KEY_B` (the B-side plane and its parents).
    /// Bit 1.
    pub is_use_key_b: T,
    /// CV-source selector: `JACKPOT_KEY` (the lottery compression). Bit 2.
    pub is_use_jackpot_key: T,
    /// CV-source selector: BLAKE3 IV for unkeyed compressions. Bit 3.
    pub is_use_iv: T,
    /// 1 on the A-side commit-fold wrapper's finalization row: binds `CV_OUT` to the `HASH_A`
    /// public limbs. Bit 4.
    pub is_bind_hash_a: T,
    /// As `is_bind_hash_a` for the B side / `HASH_B`. Bit 5.
    pub is_bind_hash_b: T,
    /// 1 on the routing tree root's finalization row: binds `CV_OUT` to `HASH_ROUTING`. Bit 6.
    pub is_bind_routing_hash: T,
    /// 1 on **all 8 rows** of the lottery compression (not just the finalization row): the
    /// lottery-words CTL filter is `IS_BIND_JACKPOT_HASH * IS_NEW_BLAKE` on the message-load row
    /// (`super::ctl`), so the flag must span the compression; the `HASH_JACKPOT` binding is
    /// gated by `IS_BIND_JACKPOT_HASH * IS_LAST_ROUND`. Bit 7.
    pub is_bind_jackpot_hash: T,
    /// 1 on CV-fetch rows: filter of the CV-routing lookup (this row's `CV_IN` is fetched at key
    /// `CV_ROUTE_KEY_OR_TWEAK`), and the `CV_IN` selector of the row-0 CV mux (block 2+ of a
    /// multi-block chunk fetches on its row 0). Bit 8.
    pub is_cv_in: T,
    /// 1 on row 0 of every compression (and on padding rows): anchors the init state and gates
    /// the round/finalization handoff. Bit 9.
    pub is_new_blake: T,
    /// 1 on row 7 of every live compression: pins `BLAKE3_MSG_BUFFER = permute(BLAKE3_MSG)`
    /// (the message the compression actually consumed). Bit 10.
    pub is_last_round: T,
    /// Message-source encoding: plane bytes `100`, auxiliary/routing/lottery bytes `011`,
    /// CV window `001`, no load `000`. Mode `010` is forbidden; lottery words instead
    /// arrive as auxiliary bytes bound by the XorFold channel.
    pub is_msg_bits: [T; 3],
    /// 1 on rows whose first ingested u32 word is pinned to `OUTER_INDEX_FIRST`: routing rows
    /// where that slot holds a sampled entry. Unsampled neighbor slots keep
    /// the selector down and their words free. Bit 14.
    pub is_first_outer: T,
    /// As `is_first_outer` for the second ingested u32 word / `OUTER_INDEX_SECOND`. Bit 15.
    pub is_second_outer: T,
    /// 1 on the offsets tree root's finalization row: binds `CV_OUT` to `HASH_OFFSETS`.
    /// Bit 16.
    pub is_bind_offsets_hash: T,
    /// 1 when the row's first ingested word is pinned to `WORD_PIN_FIRST` (offsets rows:
    /// the public offsets and the zero padding after `O_{e-1}`). Bit 17.
    pub is_word_pin_first: T,
    /// As `is_word_pin_first` for the second ingested word / `WORD_PIN_SECOND`. Bit 18.
    pub is_word_pin_second: T,
    /// 1 when the row's two ingested words are consecutive entries of a checked range (the
    /// offsets list `O`, or the winner slice `R[w]`): checks `word0 <= word1` (strict when
    /// `IS_CHAIN_STRICT` is set) through `CHAIN_INTRA_LIMBS`. Bit 19.
    pub is_chain_intra: T,
    /// 1 when the row's first word continues the chain from the previous stream word, held
    /// by `CHAIN_CARRY`: checks `CHAIN_CARRY <= word0` (strict when `IS_CHAIN_STRICT` is
    /// set) through `CHAIN_INTER_LIMBS`. Bit 20.
    pub is_chain_inter: T,
    /// 1 on routing rows: the chain gates subtract it from their differences, making the
    /// routing order strict (`<`) where the offsets order is non-strict (`<=`). Bit 21.
    pub is_chain_strict: T,
    /// 1 on the routing row whose first word is the slice's last entry: checks
    /// `word0 <= WORD_PIN_FIRST = m - 1` through this row's `CHAIN_INTRA_LIMBS`, which are
    /// idle here (the intra gate needs a second in-slice word). With the strict chain this
    /// bounds the whole slice into `[0, m)`. Bit 22.
    pub is_bound_first: T,
    /// As `IS_BOUND_FIRST` when the slice's last entry is the row's second word: checks
    /// `word1 <= WORD_PIN_SECOND = m - 1` through the next row's `CHAIN_INTER_LIMBS`, which
    /// are idle there (the next pair starts past the slice). Bit 23.
    pub is_bound_second: T,
    /// 1 on every routing/offsets block row. `CHAIN_CARRY` updates to the row's second word
    /// on these rows and copies from the previous row everywhere else, so the chain
    /// survives the Merkle-parent compressions interleaved between blocks. Bit 24.
    pub is_chain_data: T,

    // Main: witness data.
    /// First/second MoE outer index as 13-bit limb pairs (`value = limb0 + 2^13*limb1`),
    /// constrained against `MOE_OUTER_INDICES_PACKED` (constraint 6); each limb bounded
    /// < 2^13 by the **unfiltered** pair `RC16(limb)` + `RC16(8*limb)` (`super::ctl` — the
    /// first pins `limb < 2^16` so the second's `8*limb` cannot alias mod p) — the bounds
    /// must hold on every row so the packed decomposition is unique and dark slots are
    /// pinned to zero.
    pub outer_index_first: [T; 2],
    /// See `outer_index_first`.
    pub outer_index_second: [T; 2],
    /// 16-bit limb pair of a gated order-chain difference: `diff = limbs[0] + 2^16 *
    /// limbs[1]`, both limbs RC16-bounded on every row (`super::ctl`), so the difference
    /// lies in `[0, 2^32)` and a wrapped negative can never satisfy the gate. Holds
    /// `word1 - word0 - IS_CHAIN_STRICT` under `IS_CHAIN_INTRA`, and `WORD_PIN_FIRST -
    /// word0` under `IS_BOUND_FIRST` (that row's intra gate is idle). Free witness when
    /// both gates are off.
    pub chain_intra_limbs: [T; 2],
    /// As `chain_intra_limbs` for the across-row chain: holds `word0 - CHAIN_CARRY -
    /// IS_CHAIN_STRICT` under `IS_CHAIN_INTER`, and the previous row's `WORD_PIN_SECOND -
    /// word1` on the row after `IS_BOUND_SECOND` (whose inter gate is idle).
    pub chain_inter_limbs: [T; 2],
    /// The last routing/offsets word seen before this row: updated to the row's second word
    /// on `IS_CHAIN_DATA` rows, copied from the previous row elsewhere. The inter chain
    /// compares a data row's first word against this carry, so the comparison reaches the
    /// stream-previous word even when Merkle parents sit between two blocks.
    pub chain_carry: T,
    /// Row counter (constraint 8); key of the CV-routing lookup (`CV_OUT` published here).
    pub trace_row_index: T,
    /// The 8 message bytes ingested this row (int8 elements, bf16 scale halves, routing-word
    /// bytes, or auxiliary Merkle/lottery bytes); each byte in [0, 255] via BYTES2
    /// (`super::ctl`); source of the InputQuant CTL channels and of the message-buffer tail
    /// load.
    pub uint8_data: [T; NUM_UINT8],
    /// 16-word sliding buffer accumulating the message (shift-by-2 per row): at each
    /// compression's row 7 it equals the 16 u32 message words.
    pub blake3_msg_buffer: [T; 16],
    /// The 16 u32 message words of the running compression, in this round's schedule order
    /// (row 0 holds the message itself; each next row is the BLAKE3 permutation of the previous;
    /// `permute^8 = id` closes the loop against the buffer at row 7).
    pub blake3_msg: [T; 16],
    /// The 8 chaining-value words routed from an earlier row's `CV_OUT` (fetched by the
    /// CV-routing lookup at key `CV_ROUTE_KEY_OR_TWEAK`); 0 off fetch rows.
    pub cv_in: [T; 8],
    /// The CV entering the compression: one-hot mux of `KEY_A` / `KEY_B` / `JACKPOT_KEY`
    /// / `CV_IN` (constraint 3).
    pub blake3_cv: [T; 8],
    /// Round states: 4 tracked states of the G-function cascade (constraint 1);
    /// `round[0]` is the row's input state (= the init state on `IS_NEW_BLAKE` rows).
    pub round: [Blake3StateCols<T>; NUM_TRACKED_STATES],
    /// The 8 output CV words: on finalization rows the compression output
    /// (`v[i] ^ v[8+i]`), on other rows the same XOR expression over that row's tracked states
    /// (published into the routing lookup with multiplicity 0).
    pub cv_out: [T; 8],
    /// Witness multiplicity of this row's `CV_OUT` in the CV-routing lookup (how many later
    /// rows fetch it); nonzero only on finalization rows of consumed compressions.
    pub cv_out_freq: T,
}

pub const NUM_BLAKE3_COLUMNS: usize = size_of::<Blake3ColumnsView<u8>>();

const _: () = assert!(NUM_BLAKE3_COLUMNS == 1164);
const _: () = assert!(BLAKE3_STATE_WIDTH == 264);

/// Public inputs of Blake3Stark, 8 u32 limbs each, in LE word order of the native 32-byte
/// digests. `HASH_A`/`HASH_B` are the per-side aggregate commitment digests: an in-AIR
/// wrapper per side folds the two plane tree roots (`api/proof_utils.rs::
/// operand_digest_fp10`), so the roots themselves stay internal.
pub const PI_KEY_A: usize = 0;
/// KEY_B limbs (the B-side plane's tree key).
pub const PI_KEY_B: usize = 8;
/// JACKPOT_KEY limbs (the lottery compression's key).
pub const PI_JACKPOT_KEY: usize = 16;
/// HASH_A limbs (the A-side commitment digest, bound at the A combine wrapper).
pub const PI_HASH_A: usize = 24;
/// HASH_B limbs (the B-side commitment digest).
pub const PI_HASH_B: usize = 32;
/// HASH_ROUTING limbs (the routing tree root; all-zero when the job has no routing plane).
pub const PI_HASH_ROUTING: usize = 40;
/// HASH_JACKPOT limbs (the lottery compression output).
pub const PI_HASH_JACKPOT: usize = 48;
/// HASH_OFFSETS limbs (the MoE offsets tree root `HO`; all-zero for dense jobs).
pub const PI_HASH_OFFSETS: usize = 56;
/// Number of Blake3Stark public inputs.
pub const NUM_BLAKE3_PUBLIC_INPUTS: usize = 64;

columns_view!(Blake3ColumnsView, NUM_BLAKE3_COLUMNS, BLAKE3_COL_MAP);

/// Number of leading verifier-known schedule columns.
pub const NUM_BLAKE3_KNOWN_COLUMNS: usize = BLAKE3_COL_MAP.word_pin_second + 1;

/// The 25 unpack-flag column indices in `ROW_FLAGS_PACKED` bit order (constraint 2's weights are
/// `2^position` in this array).
pub const fn unpack_flag_cols() -> [usize; NUM_UNPACK_FLAGS] {
    let m = &BLAKE3_COL_MAP;
    [
        m.is_use_key_a,
        m.is_use_key_b,
        m.is_use_jackpot_key,
        m.is_use_iv,
        m.is_bind_hash_a,
        m.is_bind_hash_b,
        m.is_bind_routing_hash,
        m.is_bind_jackpot_hash,
        m.is_cv_in,
        m.is_new_blake,
        m.is_last_round,
        m.is_msg_bits[0],
        m.is_msg_bits[1],
        m.is_msg_bits[2],
        m.is_first_outer,
        m.is_second_outer,
        m.is_bind_offsets_hash,
        m.is_word_pin_first,
        m.is_word_pin_second,
        m.is_chain_intra,
        m.is_chain_inter,
        m.is_chain_strict,
        m.is_bound_first,
        m.is_bound_second,
        m.is_chain_data,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_BLAKE3_COLUMNS] = BLAKE3_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        assert_eq!(BLAKE3_COL_MAP.row_flags_packed, 0);
        assert_eq!(BLAKE3_COL_MAP.ctl_key_base, 1);
        assert_eq!(BLAKE3_COL_MAP.is_int8_message, 2);
        assert_eq!(BLAKE3_COL_MAP.is_scale_message, 3);
        assert_eq!(BLAKE3_COL_MAP.cv_route_key_or_tweak, 4);
        assert_eq!(BLAKE3_COL_MAP.moe_outer_indices_packed, 5);
        assert_eq!(BLAKE3_COL_MAP.word_pin_first, 6);
        assert_eq!(BLAKE3_COL_MAP.word_pin_second, 7);
        // The unpack flags are consecutive after the verifier-known block.
        let flags = unpack_flag_cols();
        for (j, &c) in flags.iter().enumerate() {
            assert_eq!(c, NUM_BLAKE3_KNOWN_COLUMNS + j, "unpack flag {j} not at its packing position");
        }
        assert_eq!(BLAKE3_COL_MAP.cv_out_freq, NUM_BLAKE3_COLUMNS - 1);
        // The round block is 4 * 264 = 1056 contiguous columns.
        assert_eq!(
            BLAKE3_COL_MAP.round[0].a_words[0] + 4 * BLAKE3_STATE_WIDTH,
            BLAKE3_COL_MAP.cv_out[0]
        );
    }

    #[test]
    fn view_array_roundtrip() {
        let mut arr = [0u64; NUM_BLAKE3_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: Blake3ColumnsView<u64> = arr.into();
        assert_eq!(view.row_flags_packed, 1);
        assert_eq!(view.uint8_data[0], BLAKE3_COL_MAP.uint8_data[0] as u64 * 3 + 1);
        assert_eq!(
            view.round[2].b_bits[1][7],
            BLAKE3_COL_MAP.round[2].b_bits[1][7] as u64 * 3 + 1
        );
        assert_eq!(view.cv_out_freq, (NUM_BLAKE3_COLUMNS as u64 - 1) * 3 + 1);
        let back: [u64; NUM_BLAKE3_COLUMNS] = view.into();
        assert_eq!(back, arr);

        use core::borrow::Borrow;
        let borrowed: &Blake3ColumnsView<u64> = arr.borrow();
        assert_eq!(borrowed.trace_row_index, arr[BLAKE3_COL_MAP.trace_row_index]);
    }
}
