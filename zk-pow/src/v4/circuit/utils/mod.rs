//! Reuse the frozen v2 evaluation interface and implementations as one unit.
//! V4 constraints remain local; only their evaluation machinery is inherited.

pub(crate) use crate::v2::circuit::utils::{evaluator, native_evaluator, symbolic_evaluator};
