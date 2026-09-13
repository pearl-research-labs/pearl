//! Computes and proves one scale row per matrix row.
//!
//! It receives InputQuant's framed sum of prequant squares and maximum absolute decoded value,
//! proves the correctly rounded bf16 square root, and snaps its code to the nearest multiple of
//! four (ties upward) to obtain `l2`. It then decodes `linf`, floors both norms at `2^-32`
//! (the reference scheme's `row_norms` floor), derives `noised_bound -> alpha -> beta`,
//! enforces the jackpot liveness totals (check 1) and the per-row noise floor (check 2), and
//! exports each row's exact noise std `sigma` to TamedStark (check 3's sigma channel).
//! [`columns`] fixes trace order, [`stark`] fills and constrains each row, and [`ctl`]
//! declares the group-tuple, sigma-channel, and committed-LUT lookups.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{NUM_SCALE_COLUMNS, NUM_SCALE_PUBLIC_INPUTS, SCALE_COL_MAP, ScaleColumnsView};
pub use ctl::{ctl_looked_scale_group_tuple, scale_lut_lookups};
pub use stark::{ScaleProgram, ScaleRowTuple, ScaleStark};
