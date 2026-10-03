//! Eight-row BLAKE3 compression AIR for fp8.
//!
//! Each compression streams one 64-byte message, constrains seven rounds, and finalizes an
//! eight-word chaining value (CV). The module routes those CVs through keyed Merkle trees for
//! matrix values, scales, and optional mixture-of-experts routing, then binds their public roots
//! and the XorFold jackpot hash. The native verifier separately binds matrix dimensions and
//! other job-shape metadata.

pub mod columns;
pub mod ctl;
pub mod stark;

pub use columns::{BLAKE3_COL_MAP, Blake3ColumnsView, NUM_BLAKE3_COLUMNS, NUM_BLAKE3_PUBLIC_INPUTS};
pub use stark::{
    Blake3Instruction, Blake3Program, Blake3Stark, Blake3TraceInputs, CvRef, CvSource, MessageSource, PlaneId, PublicBinding,
};
