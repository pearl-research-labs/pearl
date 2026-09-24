//! Unified BLAKE3 Merkle tree library with SIMD-optimized hashing.
//!
//! This crate provides:
//! - `Blake3Hasher`: Low-level BLAKE3 primitives with SIMD and optional parallelism
//! - `MerkleTree`: Tree construction and multi-leaf proof generation
//! - `MerkleProof`: Proof verification and byte extraction utilities

pub mod hasher;
pub mod merkle;

/// Python FFI bindings for [`MerkleTree`] and [`MerkleProof`].
#[cfg(feature = "pyo3")]
mod merkle_py;

pub use blake3::{
    BLOCK_LEN as BLAKE3_MSG_LEN, CHUNK_LEN as BLAKE3_CHUNK_LEN, OUT_LEN as BLAKE3_DIGEST_SIZE,
};
pub use hasher::{
    blake3_digest, Blake3Hasher, B3F_CHUNK_END, B3F_CHUNK_START, B3F_KEYED_HASH, B3F_PARENT,
    B3F_ROOT,
};
pub use merkle::{
    is_allowed_chunk_len, pad_to_chunk_boundary, padded_chunk_len, MerkleProof, MerkleTree,
    ALLOWED_CHUNK_LENS,
};

#[cfg(feature = "extension-module")]
#[pyo3::pymodule]
fn pearl_blake3(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;

    module.add("BLAKE3_CHUNK_LEN", BLAKE3_CHUNK_LEN)?;
    module.add("ALLOWED_CHUNK_LENS", ALLOWED_CHUNK_LENS)?;
    module.add_class::<MerkleTree>()?;
    module.add_class::<MerkleProof>()?;
    Ok(())
}
