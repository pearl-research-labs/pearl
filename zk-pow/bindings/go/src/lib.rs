//! Go FFI bindings for ZK-POW.
//!
//! This crate provides C-compatible FFI functions for ZK proof generation and verification,
//! primarily used by the Go pearld node.

#[cfg(unix)]
use tikv_jemallocator::Jemalloc;

#[cfg(unix)]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

mod common;
mod mine;
mod verify;

pub use common::{CZKProof, PUBLICDATA_MAX_SIZE, PUBLICDATA_SIZE};
pub use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration};

pub use mine::mine;
pub use verify::verify_zk_proof_v1;
pub use verify::verify_zk_proof_v2;
pub use verify::verify_zk_proof_v2_with_nbits;
