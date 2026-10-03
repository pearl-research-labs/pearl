//! Go FFI bindings for ZK-POW.
//!
//! This crate provides C-compatible FFI functions for ZK proof generation and verification,
//! primarily used by the Go pearld node.
//!
//! Each `vN` module holds one proof version's entry points (`v2` also serves cert v3: same
//! circuits, salted noise seed). Version modules never import each other; they share only
//! [`common`]. Every symbol exported to C is re-exported below.

#[cfg(unix)]
use tikv_jemallocator::Jemalloc;

#[cfg(unix)]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

mod common;
mod v1;
mod v2;
mod v4;

pub use common::{CZKProof, ERROR_MSG_MAX_SIZE, MAX_ZK_PROOF_SIZE, PUBLICDATA_MAX_SIZE};
pub use zk_pow::v2::api::proof::{IncompleteBlockHeader, MiningConfiguration};

pub use v1::verify_zk_proof_v1;

pub use v2::mine::{default_mining_config, mine, mine_moe};
pub use v2::plain::{prove_plain_proof_ffi, verify_plain_proof_ffi};
pub use v2::verify::{
    check_rank_penalty, verify_zk_proof_v2, verify_zk_proof_v2_with_nbits, verify_zk_proof_v3, verify_zk_proof_v3_with_nbits,
};
pub use v2::{MINING_CONFIG_RESERVED_SIZE, MINING_CONFIG_SERIALIZED_SIZE, MIN_NOISE_RANK, PUBLICDATA_SIZE};

pub use v4::plain::verify_plain_proof_v4_ffi;
pub use v4::verify::verify_zk_proof_v4;
pub use v4::{
    V4_AXIS_PATTERN_NUM_DIMS, V4_BLOCK_HEADER_SERIALIZED_SIZE, V4_MOE_PARAMS_MAX_NUM_EXPERTS, V4_PUBLIC_PARAMS_WIRE_SIZE,
};
