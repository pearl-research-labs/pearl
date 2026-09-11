//! FP8 proof-of-work: six main STARKs and committed lookup tables.
//!
//! Each STARK proves an execution trace using an algebraic intermediate
//! representation (AIR): polynomial constraints on its rows. Cross-table
//! lookups (CTLs) bind values shared by different traces. Fixed lookup tables
//! (LUTs) supply range checks and small arithmetic operations.
//!
//! - [`blake3_stark`]: operand commitments, routing and jackpot hashes.
//! - [`input_quant_stark`]: prequant decoding, noise addition and FP8 conversion.
//! - [`scale_stark`]: row norms, quantization scales and noise-floor checks.
//! - [`matmul_b200_stark`]: B200 accumulation and per-cell magnitude/skip counts.
//! - [`xor_fold_stark`]: fold matmul results into the lottery message.
//! - [`tamed_stark`]: tile-wide tamed-product and skip budgets.
//!
//! [`ctl`] connects the tables; [`luts`] supplies preprocessed lookup data.
//! [`known_values`] assembles columns recomputed from the public statement.
//! [`driver`] proves the batch in canonical table order, grouping the LUT
//! commitments. [`wrapper`] recursively verifies it and adds zero knowledge.
//! [`unpredictability`] defines the integer summand scores and skip rule.
//! The test-only `consistency` module checks the complete trace pipeline.
//!
//! Inputs use int8 values, BF16 block scales and FP8 E4M3 matmul.

pub mod blake3_stark;
pub mod circuit_utils;
pub(crate) mod columns_view;
#[cfg(test)]
pub(crate) mod consistency;
pub mod ctl;
pub mod driver;
pub mod input_quant_stark;
pub mod known_values;
pub mod luts;
pub mod matmul_b200_stark;
pub mod scale_stark;
pub mod tamed_stark;
pub mod unpredictability;
pub mod wrapper;
pub mod xor_fold_stark;
