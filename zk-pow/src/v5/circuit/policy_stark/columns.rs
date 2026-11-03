//! Trace columns for the FP16 "unpredictable accumulation steps" policy AIR.
//!
//! One row per `G = 8` group step (the same row grid as the A100 matmul AIR), carrying the
//! per-step census and the tile-global running accumulators that realise the exact integer gate
//! of [`crate::v5::api::policy`]. See [`super::stark`] for the constraint derivation (`PP1..`).

use crate::v4::circuit::columns_view::columns_view;

/// View of one `PolicyStarkA100` trace row.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct PolicyA100ColumnsView<T: Copy> {
    // ---- Structural, class (a): verifier-recomputable from the AIR geometry. ----
    /// Output cell index, constant across the cell's `k/8` rows.
    pub cell_id: T,
    /// 1 on each cell's last row (a cell boundary — resets the run detector).
    pub is_cell_final: T,
    /// 1 on trailing power-of-two padding rows (excluded from every accumulator).
    pub is_padding: T,
    /// Operand flat-index base for this row's 8 A lanes (`r*k + j*GROUP`), mirroring the matmul
    /// AIR's column of the same name. Class (a): verifier-recomputable. Together with
    /// [`Self::operand_index_base_b`] it is the unique per-live-row key the census-import CTL
    /// (`matmul -> policy`) binds on, so the policy reads the matmul's tightly-pinned census
    /// instead of regenerating it.
    pub operand_index_base_a: T,
    /// Operand flat-index base for this row's 8 B lanes (`h*k + c*k + j*GROUP`). Class (a).
    pub operand_index_base_b: T,

    // ---- Per-step census (CTL-imported from the matmul AIR in the full system; see docs). ----
    /// 1 iff this group step is a breakpoint (`a100_dot` breakpoint; MA11 of the matmul AIR).
    pub group_breakpoint: T,
    /// Number of products truncated on this step, in `[0, 8]` (the matmul AIR's per-step
    /// `products_truncated_flag` sum).
    pub products_truncated: T,

    // ---- Derived per-step + tile-global running accumulators. ----
    /// 1 iff this step starts a maximal run of non-breakpoint steps: non-breakpoint AND
    /// (cell-start OR the previous step was a breakpoint). PP2.
    pub run_start: T,
    /// Inclusive running tile count of breakpoint steps (over non-padding rows). PP3.
    pub tile_breakpoints: T,
    /// Inclusive running tile numerator `sum[ 8*N_bp + 32*N_runs + N_pt ]` (over non-padding
    /// rows): each step adds `8*breakpoint + 32*run_start + (1-breakpoint)*products_truncated`. PP4.
    pub tile_numerator: T,

    // ---- Gate slack (meaningful on the last row; RANGE16-checked on every row). ----
    /// Low/high 16-bit limbs of `rho_slack = 5*tile_numerator - 6*cells*k >= 0`. PP5.
    pub rho_slack_lo: T,
    pub rho_slack_hi: T,
    /// Low/high 16-bit limbs of `fbp_slack = 10*tile_breakpoints - 3*total_steps >= 0`. PP6.
    pub fbp_slack_lo: T,
    pub fbp_slack_hi: T,
}

/// Total number of committed `PolicyStarkA100` columns.
pub const NUM_POLICY_A100_COLUMNS: usize = size_of::<PolicyA100ColumnsView<u8>>();

const _: () = assert!(NUM_POLICY_A100_COLUMNS == 14);

columns_view!(PolicyA100ColumnsView, NUM_POLICY_A100_COLUMNS, POLICY_A100_COL_MAP);

/// Number of leading class (a) ("known") columns (pure functions of AIR geometry):
/// `cell_id`, `is_cell_final`, `is_padding`, and the two operand-index bases the census-import
/// CTL keys on.
pub const NUM_POLICY_A100_KNOWN_COLUMNS: usize = POLICY_A100_COL_MAP.operand_index_base_b + 1;

/// No public inputs — the gate thresholds are AIR constants of the program geometry.
pub const NUM_POLICY_A100_PUBLIC_INPUTS: usize = 0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn col_map_is_the_identity_layout() {
        let as_array: [usize; NUM_POLICY_A100_COLUMNS] = POLICY_A100_COL_MAP.into();
        for (i, &c) in as_array.iter().enumerate() {
            assert_eq!(c, i);
        }
        assert_eq!(POLICY_A100_COL_MAP.cell_id, 0);
        assert_eq!(POLICY_A100_COL_MAP.is_padding, 2);
        assert_eq!(POLICY_A100_COL_MAP.operand_index_base_a, 3);
        assert_eq!(POLICY_A100_COL_MAP.operand_index_base_b, 4);
        assert_eq!(NUM_POLICY_A100_KNOWN_COLUMNS, 5);
    }
}
