//! Blake3 membership program helpers (compression, round logic, and the
//! [`program`] used by the plain FP8 verifier to evaluate Merkle membership).
//! The Blake3 STARK chip itself lives in the `v1`/`v2` clones.

pub mod blake3_compress;
pub mod logic;
pub mod program;
