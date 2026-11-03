//! FP16-owned fork of the eight-row BLAKE3 compression AIR.
//!
//! This is a verbatim fork of [`crate::v4::circuit::blake3_stark`] (the shared, scheme-neutral
//! BLAKE3 engine) with the public types renamed (`Fp16Raw*`) so the FP16 scheme can extend the
//! AIR without touching the live FP8 consensus engine. The only functional addition over the
//! shared engine is an **output-egress** CTL channel (`IS_EGRESS_CV` + `cv_egress_limbs`, see
//! [`ctl::ctl_cv_egress_looking_blake3`]) that exports a compression's `cv_out` as 16-bit limbs
//! on a program-chosen key, which the FP16 noise binding consumes. All existing engine behavior
//! is byte-identical when the egress flag is 0.
//!
//! It still reuses the shared infrastructure it depends on: the `columns_view!` macro
//! ([`crate::v4::circuit::columns_view`]), the generic constraint evaluators
//! ([`crate::v4::circuit::utils`]), and the committed-LUT machinery ([`crate::v4::circuit::luts`]).
//! Only the AIR itself is forked.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{
    FP16_RAW_BLAKE3_COL_MAP, Fp16RawBlake3ColumnsView, NUM_FP16_RAW_BLAKE3_COLUMNS, NUM_FP16_RAW_BLAKE3_PUBLIC_INPUTS,
};
pub use stark::{
    Fp16RawBlake3Instruction, Fp16RawBlake3Program, Fp16RawBlake3Stark, Fp16RawBlake3TraceInputs, Fp16RawCvRef,
    Fp16RawCvSource, Fp16RawMessageSource, Fp16RawPlaneId, Fp16RawPublicBinding,
};
