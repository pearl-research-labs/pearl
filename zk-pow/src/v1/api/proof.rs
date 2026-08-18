pub type Hash256 = [u8; 32];

/// The block header that is set by the verifier and for which the proof should apply.
/// Serialized by miner/node field by field in little endian and with hash bytes reversed.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "IncompleteBlockHeader", get_all, set_all))]
pub struct IncompleteBlockHeader {
    pub version: u32,         // Version of the blockchain protocol
    pub prev_block: Hash256,  // commitment hash of previous block header
    pub merkle_root: Hash256, // of transactions
    pub timestamp: u32,       // Unix timestamp. Seconds since epoch.
    pub nbits: u32,           // Difficulty target (U256) encoded as u32
}

/// Matrix multiply-accumulate type.
/// Initial blockchain version only support 0 denoting Int7xInt7ToInt32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(eq, eq_int))]
pub enum MMAType {
    Int7xInt7ToInt32 = 0,
}

/// A periodic pattern of indices, represented as a generalized arithmetic progression.
/// Shape is a fixed-size array of (stride, length) tuples that define the 3D arithmetic progression.
/// a * stride[0] + b * stride[1] + c * stride[2] for a < length[0], b < length[1], c < length[2].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "PeriodicPattern"))]
pub struct PeriodicPattern {
    pub shape: [(u32, u32); 3],
}

/// Size of the reserved field in MiningConfiguration.
pub const MINING_CONFIG_RESERVED_SIZE: usize = 32;

/// The parameters a miner must commit to before starting to mine.
/// Serialized by miner/node field by field in little endian (52 bytes total).
/// rows_pattern and cols_pattern define periodic index patterns that partition
/// the A rows and B columns respectively.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MiningConfiguration", get_all, set_all))]
pub struct MiningConfiguration {
    pub common_dim: u32,                             // common dimension of the matmul, k. (4 bytes)
    pub rank: u16,                                   // Denotes length of inner product per inner hash invocation. (2 bytes)
    pub mma_type: MMAType,                           // (2 bytes)
    pub rows_pattern: PeriodicPattern,               // The periodic partition of A rows. (6 bytes)
    pub cols_pattern: PeriodicPattern,               // The periodic partition of B cols. (6 bytes)
    pub reserved: [u8; MINING_CONFIG_RESERVED_SIZE], // Reserved for future use. (32 bytes)
}

/// Parameters associated with a proof and are required for proof verification.
/// Serialized as the following fields in little endian:
///   ZK_CERTIFICATE_VERSION=1 (4) |  block_header hash (32) | mining_config(52) |
///   hash_a (32) | hash_b (32) | m(4) | n(4) | t_rows(4) | t_cols(4)
/// For deserialization see msgheaders.go::MsgHeader.PrlDecode.
#[derive(Debug, Clone)]
pub struct PublicProofParams {
    pub block_header: IncompleteBlockHeader,
    pub mining_config: MiningConfiguration,
    // job_key = blake3(block_header || mining_config)
    pub hash_a: Hash256, // blake3(A, key=job_key)
    pub hash_b: Hash256, // blake3(B^t, key=job_key)
    // commitment_hash = blake3(blake3(job_key || hash_b) || hash_a)
    /// Interpreted as a little-endian 256-bit integer for difficulty comparisons.
    pub hash_jackpot: Hash256, // blake3(jackpot, key=commitment_hash).
    pub m: u32,      // number of rows of A
    pub n: u32,      // number of columns of B
    pub t_rows: u32, // Describes the jackpot rows in A as the minimum element of the pattern.
    pub t_cols: u32, // Describes the jackpot columns in B as the minimum element of the pattern.
}

/// The ZK-proof. Contains public fields, as well as a ZK proof witnessing the existence of PrivateProofParams.
#[derive(Debug, Clone)]
pub struct ZKProof {
    pub pow_bits: [u8; 3],
    pub rate_bits: [u8; 3],
    pub zeta: [u8; 16],
    pub plonky2_proof: Vec<u8>,
}

/// Prover's private witness. See PlainProof for different representation of this data.
#[derive(Debug, Clone)]
pub struct PrivateProofParams {
    pub s_a: Vec<Vec<i8>>,            // rows_pattern.size() rows of A, each of length common_dim
    pub s_b: Vec<Vec<i8>>,            // cols_pattern.size() rows of B^t, each of length common_dim
    pub external_msgs: Vec<[u8; 64]>, // Additional leaf data, consumed to generate blake3 trace.
    pub external_cvs: Vec<Hash256>,   // Merkle siblings in a merkle proof.
}
