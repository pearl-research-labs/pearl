//! Columns for one XorFold step:
//! `fold_out = rotl32(low32(fold_state_in * 0x9E3779B1 + cell_word), 13)`.
//!
//! Each live row folds one Matmul cell. The leading verifier-known columns `cell_id`, `lane_id`,
//! `is_lane_final`, and `is_pad` bind the committed lane layout; the remaining columns hold
//! the raw f32 cell word, input state, exact 64-bit multiply-add limbs, and rotation split.
//! `0x9E3779B1` and rotation distance 13 are fixed protocol mixing constants.

use crate::circuit::fp8::columns_view::columns_view;

/// View of one XorFoldStark trace row. The folded output (the next fold state) is not a
/// column — it is the affine expression
/// `FOLD_OUT = (ROTATION_INPUT_BOTTOM19_LIMB_0 + 2^16*ROTATION_INPUT_BOTTOM19_LIMB_1)*2^13
/// + ROTATION_INPUT_TOP13`.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct XorFoldColumnsView<T: Copy> {
    /// The Matmul output cell this row folds, per the committed `lane_assignment`; key of the
    /// cell-results CTL. Verifier-known, verifier-recomputed from the committed lane layout.
    pub cell_id: T,
    /// Which of the 16 lottery words this lane produces; key of the Blake3 channel. Verifier-known.
    pub lane_id: T,
    /// 1 on each lane's last row: filter of the Blake3 channel, resets the fold chain. Verifier-known.
    pub is_lane_final: T,
    /// 1 on the all-zero trailing padding rows (`h*w` live rows padded to a power of two);
    /// excludes them from the cell-results channel. Verifier-known.
    pub is_pad: T,
    /// The cell's f32 word as two 16-bit limbs (range-checked), CTL-received from Matmul.
    pub cell_result_f32_lo: T,
    pub cell_result_f32_hi: T,
    /// The lane's running fold state entering this row (0 at lane start).
    pub fold_state_in: T,
    /// 16-bit limbs of the 64-bit multiply-add `FOLD_STATE_IN*0x9E3779B1 + W`, where
    /// `W = CELL_RESULT_F32_LO + 2^16*CELL_RESULT_F32_HI` is the folded cell word: low half
    /// (the pre-rotation u32) ...
    pub muladd_low_limb_0: T,
    pub muladd_low_limb_1: T,
    /// ... and high half (the discarded overflow). The cap `MULADD_HIGH_LIMB_1 <= 0xFFFE`
    /// (RC16 of `MULADD_HIGH_LIMB_1 + 1`) keeps the recomposition below the Goldilocks modulus
    /// `p = 2^64 - 2^32 + 1`, preventing a `+p` limb alias (the honest top limb is at most
    /// `0x9E38`).
    pub muladd_high_limb_0: T,
    pub muladd_high_limb_1: T,
    /// Split of the pre-rotation low u32 as `ROTATION_INPUT_TOP13*2^19
    /// + (ROTATION_INPUT_BOTTOM19_LIMB_1*2^16 + ROTATION_INPUT_BOTTOM19_LIMB_0)`: rotating
    ///   left by 13 moves the bottom 19 bits up and the top 13 bits down.
    pub rotation_input_top13: T,
    pub rotation_input_bottom19_limb_0: T,
    pub rotation_input_bottom19_limb_1: T,
}

pub const NUM_XOR_FOLD_COLUMNS: usize = size_of::<XorFoldColumnsView<u8>>();

const _: () = assert!(NUM_XOR_FOLD_COLUMNS == 14);

/// XorFoldStark has no public inputs.
pub const NUM_XOR_FOLD_PUBLIC_INPUTS: usize = 0;

columns_view!(XorFoldColumnsView, NUM_XOR_FOLD_COLUMNS, XOR_FOLD_COL_MAP);

/// Number of leading verifier-known schedule columns.
pub const NUM_XOR_FOLD_KNOWN_COLUMNS: usize = XOR_FOLD_COL_MAP.is_pad + 1;

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_XOR_FOLD_COLUMNS] = XOR_FOLD_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        assert_eq!(XOR_FOLD_COL_MAP.cell_id, 0);
        assert_eq!(XOR_FOLD_COL_MAP.lane_id, 1);
        assert_eq!(XOR_FOLD_COL_MAP.is_lane_final, 2);
        assert_eq!(XOR_FOLD_COL_MAP.is_pad, 3);
        assert_eq!(XOR_FOLD_COL_MAP.rotation_input_bottom19_limb_1, NUM_XOR_FOLD_COLUMNS - 1);
    }

    #[test]
    fn view_array_roundtrip() {
        let mut arr = [0u64; NUM_XOR_FOLD_COLUMNS];
        for (i, v) in arr.iter_mut().enumerate() {
            *v = i as u64 * 3 + 1;
        }
        let view: XorFoldColumnsView<u64> = arr.into();
        assert_eq!(view.cell_id, 1);
        assert_eq!(view.fold_state_in, XOR_FOLD_COL_MAP.fold_state_in as u64 * 3 + 1);
        let back: [u64; NUM_XOR_FOLD_COLUMNS] = view.into();
        assert_eq!(back, arr);

        let borrowed: &XorFoldColumnsView<u64> = arr.borrow();
        assert_eq!(borrowed.rotation_input_top13, arr[XOR_FOLD_COL_MAP.rotation_input_top13]);
    }
}
