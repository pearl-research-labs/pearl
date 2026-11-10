//! Authenticated strip decode, row statistics, and device-specific FP8 input quantization.
//!
//! The fold proves `c=RN32(alpha*scale)`, exact `u=c*int8`, and fused
//! `mixed=RN32(beta*noise+u)`. H100 exports `Q8(mixed)`; B200 additionally proves the
//! quantize-first byte rewrite and accumulates the Gaussian proxy for the row policy gate.
//! Norms and clean-data policy scores use the separate bf16 decode of `int8*scale`.
//!
//! [`stark`] gives the complete arithmetic statement, integer bounds, and boundary rules.
//! [`columns`] defines the two physical layouts; [`ctl`] authenticates the inputs, imports
//! Scale/Coefficient certificates, exports operands, and declares committed-LUT relations.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{
    INPUT_QUANT_COL_MAP, InputQuantColumnsView, NUM_INPUT_QUANT_COLUMNS, NUM_INPUT_QUANT_PUBLIC_INPUTS, WL2_POW_PUBLIC_INPUT,
};
pub use stark::{InputQuantProgram, InputQuantStark};
