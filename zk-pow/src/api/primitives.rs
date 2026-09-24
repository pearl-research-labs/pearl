use serde::{Deserialize, Serialize};

pub type Hash256 = [u8; 32];

/// An operand pair (`a`, `b`) with a shared element type. Generic across the
/// protocol, not FP8-specific: it names the two-sided grouping wherever one
/// type is stored once per operand (noise seeds, opened/noised operands, tile
/// bases, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sides<T> {
    pub a: T,
    pub b: T,
}

impl<T> Sides<T> {
    /// The `b` side when `is_b`, otherwise `a`.
    pub fn side(&self, is_b: bool) -> &T {
        if is_b { &self.b } else { &self.a }
    }
}

/// The block header that is set by the verifier and for which the proof should apply.
/// Serialized by miner/node field by field in little endian and with hash bytes reversed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "IncompleteBlockHeader", get_all, set_all))]
pub struct IncompleteBlockHeader {
    pub version: u32,         // Version of the blockchain protocol
    pub prev_block: Hash256,  // commitment hash of previous block header
    pub merkle_root: Hash256, // of transactions
    pub timestamp: u32,       // Unix timestamp. Seconds since epoch.
    pub nbits: u32,           // Difficulty target (U256) encoded as u32
}
