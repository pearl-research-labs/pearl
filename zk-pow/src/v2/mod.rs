//! V2-format proofs (the pre-FP8 Int7 dense + MoE STARK stack).
//!
//! A self-contained copy of the v2 circuit and API code, used to prove and
//! verify cert-v2 (`ZkMoe`) proofs. Mirrors the `v1` clone's structure.

pub mod api;
pub mod circuit;
pub mod mine;

// Re-export the ensure_eq macro from the main crate for use within this module
pub use crate::ensure_eq;
