//! A100 (`sm_80`) FP16 `HMMA.16816.F32` matrix-multiplication AIR.
//!
//! Each row computes one `G = 8`-lane accumulation step on the `2^(eta-24)` window
//! and truncates its carry toward zero to a 24-bit FP32 significand; cell-final rows
//! emit an FP32 cell result. This is the ZK analogue of
//! `crate::v5::api::accumulate::a100_dot`. See `docs/fp16_scheme/stark_feasibility.md`.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{
    MATMUL_A100_COL_MAP, MatmulA100ColumnsView, NUM_MATMUL_A100_COLUMNS, NUM_MATMUL_A100_KNOWN_COLUMNS,
};
pub use stark::MatmulStarkA100;
