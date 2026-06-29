use crate::ffi::plain_proof::OuterIndices;

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

/// Size of the trailing region in MiningConfiguration that carries the optional
/// MoE config and is otherwise reserved (zero-padded) for future use.
pub const MINING_CONFIG_RESERVED_SIZE: usize = 32;

/// Mining configuration for a Mixture-of-Experts (GROUPED_GEMM) job.
///
/// Its presence on [`MiningConfiguration`] is what selects GROUPED_GEMM mode: a
/// standard job has `moe == None`, a GROUPED_GEMM job has `moe == Some(..)`. The
/// fields are committed in the `job_key` because they live inside the serialized
/// [`MiningConfiguration`] trailer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MoEConfig", get_all, set_all))]
pub struct MoEConfig {
    pub e: u16,     // number of experts
    pub top_k: u16, // number of experts each token is routed to
}

/// The parameters a miner must commit to before starting to mine.
/// Serialized by miner/node field by field in little endian (52 bytes total).
/// rows_pattern and cols_pattern define periodic index patterns that partition
/// the A rows and B columns respectively.
///
/// The trailing 32-byte region encodes the MoE config: `e(2) | top_k(2) | zero-padding(28)`.
/// `e` doubles as the mode discriminant — `e == 0` is a standard job (`moe == None`),
/// `e > 0` is GROUPED_GEMM. Both `e` and `top_k` are committed in the `job_key`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MiningConfiguration", get_all, set_all))]
pub struct MiningConfiguration {
    pub common_dim: u32,               // common dimension of the matmul, k. (4 bytes)
    pub rank: u16,                     // Denotes length of inner product per inner hash invocation. (2 bytes)
    pub mma_type: MMAType,             // (2 bytes)
    pub rows_pattern: PeriodicPattern, // The periodic partition of A rows. (6 bytes)
    pub cols_pattern: PeriodicPattern, // The periodic partition of B cols. (6 bytes)
    /// MoE config. `None` for a standard job, `Some` for GROUPED_GEMM. Serialized
    /// into the 32-byte trailer. (32 bytes)
    pub moe: Option<MoEConfig>,
}

/// Plaintext public parameters associated with a proof and are required for verification.
///
/// On the wire (`to_wire_bytes` / `from_wire_bytes`) serialized in little-endian:
/// `mining_config(52)` | `hash_a(32)` | `hash_b(32)` | `hash_jackpot(32)` |
/// `m(4)` | `n(4)` | `t_rows(4)` | `t_cols(4)`,
/// then if [`MoEParams`] is present: MoE fields | `outer_indices`.
///
///  For deserialization see msgheaders.go::MsgHeader.PrlDecode.
#[derive(Debug, Clone)]
pub struct PublicProofParams {
    pub block_header: IncompleteBlockHeader,
    pub mining_config: MiningConfiguration,
    // job_key = blake3(block_header || mining_config)
    pub hash_a: Hash256, // blake3(A, key=job_key)
    pub hash_b: Hash256, // blake3(B^t, key=job_key)
    // hash_activations = blake3(hash_a || hash_router), where
    //   hash_router = H(H(Routing, key=job_key) || H(pad_1024(routing_offsets_le), key=job_key)),
    // in the MoE setting, otherwise hash_a.
    // commitment_hash = blake3(blake3(job_key || hash_b) || hash_activations)
    /// Interpreted as a little-endian 256-bit integer for difficulty comparisons.
    pub hash_jackpot: Hash256, // blake3(jackpot, key=commitment_hash).
    pub m: u32,      // number of rows of A
    pub n: u32,      // number of columns of B
    pub t_rows: u32, // Describes the jackpot rows in A as the minimum element of the pattern.
    pub t_cols: u32, // Describes the jackpot columns in B as the minimum element of the pattern.
    /// MoE-specific parameters. None for standard (non-MoE) proofs.
    pub moe: Option<MoEParams>,
}

/// Extra per-proof parameters required for MoE (Mixture of Experts) proofs.
///
/// The number of experts `e` and `top_k` are  part of [`MiningConfiguration::moe`].
#[derive(Debug, Clone)]
pub struct MoEParams {
    pub expert_idx: u16,           // expert of the current expert
    pub routing_offsets: Vec<u32>, // exclusive ends (cumulative token counts) of each expert within the flattened routing; the last entry equals m*top_k. Bounded by 2^32 (enforced: m * top_k < 2^32).
    pub hash_routing: Hash256,     // blake3(routing, key=job_key)
    /// The indices that appear in `hash_routing` data at `a_rows_indices()`.
    /// They correspond to the intended position of the tokens in the matrix `A`.
    pub outer_indices: OuterIndices,
}

impl MoEParams {
    /// Start offset (in u32 entries) of `expert_idx` within the flattened routing.
    ///
    /// `routing_offsets` stores exclusive ends (cumulative counts), so an expert's
    /// start is the previous expert's end, and expert 0 starts at 0.
    pub fn expert_start_offset(&self) -> u32 {
        match self.expert_idx {
            0 => 0,
            idx => self.routing_offsets[idx as usize - 1],
        }
    }
}

/// The ZK-proof. Contains public fields, as well as a ZK proof witnessing the existence of PrivateProofParams.
#[derive(Debug, Clone)]
pub struct ZKProof {
    pub pow_bits: [u8; 3],
    pub rate_bits: [u8; 3],
    pub zeta: [u8; 16],
    pub plonky2_proof: Vec<u8>,
}

/// Prover's private witness. See PlainProof for a different representation of this data.
///
/// - *s_a*: `rows_pattern.size()` rows of A, each of length `common_dim`.
/// - *s_b*: `cols_pattern.size()` rows of B^t, each of length `common_dim`.
/// - *s_routing*: 64-byte strips of raw routing data (see below).
/// - *external_msgs*: additional leaf data, consumed to generate the Blake3 trace.
/// - *external_cvs*: Merkle siblings in a Merkle proof.
///
/// Routing strips differ from matrix rows: routing is a list of `u32` token indices rather
/// than a matrix whose width is a multiple of 64 bytes. When laid out as bytes, the current
/// expert's routing does not necessarily start at a 64-byte Blake3 block boundary. We work
/// around this by "virtually" treating routing as a matrix of 64-byte rows. As a result, each
/// strip is 64 bytes long but may also contain indices belonging to other experts that share
/// the same Blake3 block. Under this view, routing membership proofs behave like matrix
/// membership proofs in the program.
#[derive(Debug, Clone)]
pub struct PrivateProofParams {
    pub s_a: Vec<Vec<i8>>,            // rows_pattern.size() rows of A, each of length common_dim
    pub s_b: Vec<Vec<i8>>,            // cols_pattern.size() rows of B^t, each of length common_dim
    pub s_routing: Vec<Vec<u8>>,      // strips of routing data
    pub external_msgs: Vec<[u8; 64]>, // Additional leaf data, consumed to generate blake3 trace.
    pub external_cvs: Vec<Hash256>,   // Merkle siblings in a merkle proof.
}
