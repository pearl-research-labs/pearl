//! FP16 jackpot-policy AIR: proves an opened `h x w x k` tile clears the "unpredictable
//! accumulation steps" gate of [`crate::v5::api::policy`] (`f_bp >= 0.30` and `rho >= 1.2`),
//! as exact integer inequalities over the per-group census. See [`stark`].

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{NUM_POLICY_A100_COLUMNS, NUM_POLICY_A100_KNOWN_COLUMNS, POLICY_A100_COL_MAP, PolicyA100ColumnsView};
pub use stark::PolicyStarkA100;
