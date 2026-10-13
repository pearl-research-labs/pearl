//! InputQuantStark handles one A and one B element per trace row. It decodes
//! `X = RNE_bf16(int8*block_scale)`, accumulates the row's prequant sum-of-squares and maximum,
//! computes `noise_term = RNE_bf16(beta*n)`, and proves the fused
//! `RNE_bf16(alpha*X + noise_term)` before casting to fp8. Here `n` is public-seed-derived
//! noise; ScaleStark verifies the row statistics and `alpha`/`beta`. The table also proves
//! the nonzero and flip counters it transports to ScaleStark. [`stark`] defines the AIR and
//! witness, [`columns`] fixes the committed layout, and [`ctl`] declares cross-table and
//! committed-LUT relations.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{
    INPUT_QUANT_COL_MAP, InputQuantColumnsView, NUM_INPUT_QUANT_COLUMNS, NUM_INPUT_QUANT_PUBLIC_INPUTS, WL2_POW_PUBLIC_INPUT,
};
pub use stark::{InputQuantProgram, InputQuantStark};
