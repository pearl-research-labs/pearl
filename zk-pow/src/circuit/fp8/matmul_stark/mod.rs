//! Exact H100 fp8 matmul accumulation, one 32-lane group step per row.
//!
//! Each row aligns and sums its products with the incoming carry; cell-final rows emit f32
//! results.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{MATMUL_COL_MAP, MatmulColumnsView, NUM_MATMUL_COLUMNS, NUM_MATMUL_PUBLIC_INPUTS};
pub use stark::MatmulStarkH100;
