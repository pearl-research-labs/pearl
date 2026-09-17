//! B200 fp8 matrix-multiplication AIR.
//!
//! Each row computes one 32-lane, 25-fractional-bit-window accumulation step and truncates its
//! carry to a 24-bit f32 significand. Final rows emit an f32 cell result.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{MATMUL_B200_COL_MAP, MatmulB200ColumnsView, NUM_MATMUL_B200_COLUMNS, NUM_MATMUL_B200_KNOWN_COLUMNS};
pub use stark::{MatmulB200Stark, MatmulProgram};
