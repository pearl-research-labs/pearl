#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
pub extern crate alloc;

/// Re-export of `plonky2_field`.
#[doc(inline)]
pub use plonky2_field as field;

pub mod fri;
pub mod gadgets;
pub mod gates;
#[cfg(any(feature = "gpu_commit", feature = "gpu_quotient"))]
pub(crate) mod gpu;
// [M3 cold-start trim] minimal public surface for the GPU `cs_leaves` marshal cache so a fresh process
// that LOADS a pre-warmed recursion circuit cache can prime the (host-only, consensus-neutral) marshal
// off the per-prove path (`prime_rec_cs_leaves`) — and so the cold-process harness can simulate a fresh
// process by dropping the in-memory marshal warmth (`clear_cs_leaves_caches`). Prover-side only.
#[cfg(feature = "gpu_quotient")]
pub use gpu::{clear_cs_leaves_caches, prime_rec_cs_leaves};
pub mod hash;
pub mod iop;
pub mod plonk;
pub mod recursion;
pub mod util;

#[cfg(test)]
mod lookup_test;
