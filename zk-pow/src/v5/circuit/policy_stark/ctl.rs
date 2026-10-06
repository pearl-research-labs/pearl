//! Committed-LUT and cross-table inventory for the FP16 policy AIR.
//!
//! Two kinds of channel:
//!
//! * **RANGE16** (wired here): the gate slack limbs `rho_slack_{lo,hi}`, `fbp_slack_{lo,hi}`
//!   and the per-step `products_truncated` count. The slack limb checks are the soundness of the
//!   policy inequalities — a tile below either threshold has a negative slack, whose field
//!   representation has no two 16-bit limbs, so no RANGE16-valid witness exists.
//! * **Census import** (wired): `group_breakpoint` and `products_truncated` are bound to equal the
//!   matmul AIR's tightly-pinned per-step census (MA11 / summed MA3 flags) by a cross-table lookup
//!   from the matmul table into the policy table, keyed on the unique per-live-row
//!   `(operand_index_base_a, operand_index_base_b)` pair — see
//!   [`crate::v5::circuit::ctl::census_import_ctl`] and [`census_import_note`]. The trace
//!   generator fills the census from the ground-truth `a100_dot`, and the batch verification
//!   enforces it matches the matmul's.

use plonky2::field::types::Field;
use starky::lookup::Column;

use super::columns::POLICY_A100_COL_MAP;
use crate::v4::circuit::luts::ctl::LutLookup;

/// The policy AIR's committed-LUT instances: RANGE16 on the four gate-slack limbs and the
/// per-step products-truncated count — five instances. All reuse the shared `RANGE16` table.
pub fn policy_a100_lut_lookups<F: Field>() -> Vec<LutLookup<F>> {
    let m = &POLICY_A100_COL_MAP;
    vec![
        LutLookup::rc16(Column::single(m.rho_slack_lo)),
        LutLookup::rc16(Column::single(m.rho_slack_hi)),
        LutLookup::rc16(Column::single(m.fbp_slack_lo)),
        LutLookup::rc16(Column::single(m.fbp_slack_hi)),
        LutLookup::rc16(Column::single(m.products_truncated)),
    ]
}

/// The census-import channel is implemented in
/// [`crate::v5::circuit::ctl::census_import_ctl`]: the matmul AIR's looking side exports, per
/// group step, `(operand_index_base_a, operand_index_base_b, group_breakpoint,
/// Σ products_truncated_flag)`, and the policy AIR's looked side imports the same tuple on its
/// matching row (keyed by the unique `(base_a, base_b)` pair) — binding the policy census to the
/// matmul's tightly-pinned one bit-for-bit.
pub const fn census_import_note() -> &'static str {
    "census import (matmul -> policy) is wired in circuit::fp16::ctl::census_import_ctl"
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;

    use super::*;

    type F = GoldilocksField;

    #[test]
    fn inventory_is_five_range16_instances() {
        let lookups = policy_a100_lut_lookups::<F>();
        assert_eq!(lookups.len(), 5);
        assert!(lookups.iter().all(|l| l.table == crate::v4::circuit::luts::LutTable::Range16));
    }
}
