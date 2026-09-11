//! Per-cell magnitude/noise comparisons and tile-wide untamed/skip budgets.
//! Matmul supplies magnitude bounds and skip counts; Scale supplies noise scales.
//! See [`stark`] for the certificates and their one-sided guarantees.

pub mod columns;
pub mod ctl;
pub mod stark;
