//! FP16 (A100) plaintext certificate: the wire witness and its parse gate.
//!
//! The FP16 analogue of [`crate::v4::api::plain_proof::PlainProofV4`], over raw
//! FP16 operands. The public statement (device, `h`/`w`/`k`/`r`, the periodic
//! tile patterns `P_A`/`P_B`, the hash ids, the proof-carried ancestor header)
//! rides in [`Fp16JobParams`]; the private witness is the two committed FP16
//! operand trees opened at the selected tile rows ([`Fp16MatrixProof`]).
//!
//! The codec is canonical fixint bincode and rejects trailing bytes.
//! [`Fp16PlainProof::parse_proof`] is the security gate: it checks witness shape,
//! derives the opening key and the B-then-A noise-seed chain, authenticates and
//! opens both operand trees (the real Merkle open+rebuild), and derives the
//! deterministic FP16 noise — returning everything
//! [`crate::v5::api::verify::verify_fp16_plain_proof`] needs to replay the tile.

use anyhow::{Result, ensure};
use pearl_blake3::MerkleProof;
use serde::{Deserialize, Serialize};

use super::commitment::verify_and_open_rows;
use super::noise::{Noise16, commitment_keys, noise_seeds, sample_noise};
use super::params::{Fp16Device, Fp16Params};
use crate::v4::api::public_params::HashId;
use crate::v4::api::layout::AxisPattern;
use crate::v4::api::primitives::{Hash256, IncompleteBlockHeader, Sides};
use crate::ensure_eq;

/// Per-operand public parameters: total committed rows, Merkle chunking, and the
/// periodic tile pattern selecting the opened rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "Fp16OperandParams", get_all))]
pub struct Fp16OperandParams {
    /// Total rows committed in this operand's tree (`m` for A, `n` for B).
    pub num_rows: u32,
    /// Merkle leaf chunk length.
    pub hash_id: HashId,
    /// The periodic tile partition (`P_A` or `P_B`).
    pub pattern: AxisPattern,
}

/// The public statement tuple carried by the witness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "Fp16JobParams"))]
pub struct Fp16JobParams {
    /// The proof-carried ancestor header `σ_Δ` (folded into the seed chain).
    pub ancestor_header: IncompleteBlockHeader,
    /// The committed device (A100).
    pub device: Fp16Device,
    /// Inner dimension `k`.
    pub k: u32,
    /// Noise rank `r`.
    pub r: u32,
    /// A-side (rows) and B-side (columns) operand params.
    pub operands: Sides<Fp16OperandParams>,
}

impl Fp16JobParams {
    /// `pB` for the seed chain: `σ_Δ ‖ k ‖ r ‖ device ‖ n ‖ hash_id_B ‖ P_B`. Exposed in-crate so the
    /// FP16 ZK driver can derive the noise seeds from the same public-parameter encoding
    /// ([`crate::v5::circuit::driver`]).
    pub(crate) fn encode_p_b(&self) -> Vec<u8> {
        let mut v = self.ancestor_header.to_bytes().to_vec();
        v.extend_from_slice(&self.k.to_le_bytes());
        v.extend_from_slice(&self.r.to_le_bytes());
        v.push(self.device.wire_tag());
        v.extend_from_slice(&self.operands.b.num_rows.to_le_bytes());
        v.push(self.operands.b.hash_id as u8);
        v.extend_from_slice(&self.operands.b.pattern.to_bytes());
        v
    }

    /// `pA` for the seed chain: `k ‖ r ‖ device ‖ m ‖ hash_id_A ‖ P_A`. Exposed in-crate (see
    /// [`Self::encode_p_b`]).
    pub(crate) fn encode_p_a(&self) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&self.k.to_le_bytes());
        v.extend_from_slice(&self.r.to_le_bytes());
        v.push(self.device.wire_tag());
        v.extend_from_slice(&self.operands.a.num_rows.to_le_bytes());
        v.push(self.operands.a.hash_id as u8);
        v.extend_from_slice(&self.operands.a.pattern.to_bytes());
        v
    }
}

/// One operand's committed tree opened at the selected tile rows.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "Fp16MatrixProof"))]
pub struct Fp16MatrixProof {
    #[serde(
        serialize_with = "MerkleProof::serialize_variable_chunk",
        deserialize_with = "MerkleProof::deserialize_variable_chunk"
    )]
    pub proof: MerkleProof,
    /// The opened global row indices (must equal the pattern's tile offsets).
    pub row_indices: Vec<usize>,
}

/// The FP16 plaintext certificate.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "Fp16PlainProof"))]
pub struct Fp16PlainProof {
    pub job: Fp16JobParams,
    pub values: Sides<Fp16MatrixProof>,
}

/// Validates `claimed` opened row indices as exactly one committed lottery tile
/// of `pattern`: a valid periodic base plus the pattern's tile offsets
/// (`base + P.tile_offsets()`), every index inside `num_rows`. Returns the
/// indices as `usize` (GLOBAL addresses).
///
/// `tile_offsets()` always starts at `0`, so the base is the first opened index
/// (`t_r * h` for the contiguous A100 layout; `base == 0` is the dense origin
/// tile). This mirrors the FP8 `_validate_indices` discipline and keeps the
/// opened indices global, so the `E` noise lines — which key on the selected
/// global row/col index — match the miner's full-matrix search.
fn tile_row_indices(pattern: &AxisPattern, claimed: &[usize], num_rows: usize, name: &str) -> Result<Vec<usize>> {
    let offsets = pattern.tile_offsets();
    ensure_eq!(claimed.len(), offsets.len(), "{name}: opened rows must select exactly one lottery tile");
    let base = *claimed.first().expect("non-empty: length checked against tile_offsets above");
    let base_u32 = u32::try_from(base).map_err(|_| anyhow::anyhow!("{name}: tile base {base} overflows u32"))?;
    ensure!(
        pattern.offset_is_valid(base_u32),
        "{name}: opened rows are not based at a valid periodic tile offset (base={base})"
    );
    let expected: Vec<usize> = offsets.iter().map(|&o| base + o as usize).collect();
    ensure_eq!(claimed, expected.as_slice(), "{name}: opened rows must equal base + tile offsets");
    ensure!(claimed.iter().all(|&i| i < num_rows), "{name}: opened tile row out of range for {num_rows} rows");
    Ok(expected)
}

/// The authenticated, opened witness: what the tile replay needs.
pub struct Fp16Opened {
    pub params: Fp16Params,
    pub rows_pattern: AxisPattern,
    pub cols_pattern: AxisPattern,
    /// `h x k` opened FP16 A rows, row-major.
    pub a_rows: Vec<u16>,
    /// `w x k` opened FP16 B rows (columns of `B^T`), row-major.
    pub b_rows: Vec<u16>,
    /// The deterministic FP16 noise factors.
    pub noise: Noise16,
    /// The A-side noise seed that keys the jackpot ticket.
    pub seed_a: Hash256,
}

impl Fp16PlainProof {
    /// Strict fixint bincode.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        use bincode::Options;
        bincode::options()
            .with_fixint_encoding()
            .serialize(self)
            .map_err(|e| anyhow::anyhow!("serialize Fp16PlainProof: {e}"))
    }

    /// Inverse of [`Self::to_bytes`]. No compat ladder; rejects trailing bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        use bincode::Options;
        bincode::options()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize(bytes)
            .map_err(|e| anyhow::anyhow!("deserialize Fp16PlainProof: {e}"))
    }

    /// Authenticate the witness and derive everything the tile replay needs.
    ///
    /// This is the security gate: do not treat a deserialized [`Fp16PlainProof`]
    /// as validated. `proposed_header` (`σ̂`) is the caller's header: it keys the
    /// A-side tree and the A branch of the seed chain. The B-side tree and the B
    /// branch key on the proof-carried ancestor header (`σ_Δ`, the
    /// [`Fp16JobParams::ancestor_header`]). NOTE: this plaintext path is retired as
    /// a consensus path (the wired V5 consensus certificate is the header-bound ZK
    /// cert, `verify_fp16_zk_cert_ffi`); it performs NO state-window authentication
    /// of the ancestor itself — only the consensus verifier's
    /// `check_fp16_certificate_ancestors` (`zk-pow/bindings/go/src/fp16.rs`) does
    /// that. Do not wire this as a standalone acceptance gate (see the FP16 keying
    /// docs in [`crate::v5::api::noise`]).
    pub fn parse_proof(&self, proposed_header: &IncompleteBlockHeader) -> Result<Fp16Opened> {
        ensure!(self.job.device == Fp16Device::A100, "FP16 device must be A100");

        let rows_pattern = self.job.operands.a.pattern.clone();
        let cols_pattern = self.job.operands.b.pattern.clone();
        let k = self.job.k as usize;
        let r = self.job.r as usize;
        let h = rows_pattern.tile_size() as usize;
        let w = cols_pattern.tile_size() as usize;

        let params = Fp16Params { device: self.job.device, h, w, k, r };
        params.validate()?;

        let m = self.job.operands.a.num_rows as usize;
        let n = self.job.operands.b.num_rows as usize;

        // The opened rows must be one committed lottery tile: a valid periodic
        // base plus the pattern's tile offsets (`base + P.tile_offsets()`). The
        // base is the tile's global row origin (`t_r * h` for the contiguous
        // A100 layout); `base == 0` is the dense origin tile.
        let a_idx = tile_row_indices(&rows_pattern, &self.values.a.row_indices, m, "P_A")?;
        let b_idx = tile_row_indices(&cols_pattern, &self.values.b.row_indices, n, "P_B")?;

        // The per-side opening keys: keyA = H_"key-A"(proposed), keyB =
        // H_"key-B"(ancestor). B's tree only rebuilds under the ancestor it was
        // committed against, so an out-of-window / unauthenticated ancestor fails
        // the B-side Merkle rebuild below.
        let keys = commitment_keys(proposed_header, &self.job.ancestor_header);
        let root_a = self.values.a.proof.root;
        let root_b = self.values.b.proof.root;

        // Authenticate and open both operand trees (real Merkle open + rebuild).
        let a_rows =
            verify_and_open_rows(&self.values.a.proof, &a_idx, m, k, self.job.operands.a.hash_id, keys.a, &root_a)?;
        let b_rows =
            verify_and_open_rows(&self.values.b.proof, &b_idx, n, k, self.job.operands.b.hash_id, keys.b, &root_b)?;

        // The B-then-A noise-seed chain over the authenticated roots.
        let roots = Sides { a: root_a, b: root_b };
        let p = Sides { a: self.job.encode_p_a(), b: self.job.encode_p_b() };
        let seeds = noise_seeds(&keys, &roots, &p);

        let a_idx_u32: Vec<u32> = a_idx.iter().map(|&i| i as u32).collect();
        let b_idx_u32: Vec<u32> = b_idx.iter().map(|&i| i as u32).collect();
        let noise = sample_noise(k, r as u16, seeds, &a_idx_u32, &b_idx_u32);

        Ok(Fp16Opened { params, rows_pattern, cols_pattern, a_rows, b_rows, noise, seed_a: seeds.a })
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl Fp16OperandParams {
    #[new]
    fn py_new(num_rows: u32, hash_id: HashId, pattern: AxisPattern) -> Self {
        Self {
            num_rows,
            hash_id,
            pattern,
        }
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl Fp16MatrixProof {
    #[new]
    fn py_new(proof: &MerkleProof, row_indices: Vec<usize>) -> Self {
        Self {
            proof: proof.clone(),
            row_indices,
        }
    }

    #[getter]
    fn row_indices(&self) -> Vec<usize> {
        self.row_indices.clone()
    }

    #[getter]
    fn root<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.proof.root)
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl Fp16JobParams {
    #[new]
    fn py_new(
        ancestor_header: IncompleteBlockHeader,
        device: Fp16Device,
        k: u32,
        r: u32,
        a: Fp16OperandParams,
        b: Fp16OperandParams,
    ) -> Self {
        Self {
            ancestor_header,
            device,
            k,
            r,
            operands: Sides { a, b },
        }
    }

    #[getter]
    fn ancestor_header(&self) -> IncompleteBlockHeader {
        self.ancestor_header
    }

    #[getter]
    fn device(&self) -> Fp16Device {
        self.device
    }

    #[getter]
    fn k(&self) -> u32 {
        self.k
    }

    #[getter]
    fn r(&self) -> u32 {
        self.r
    }

    #[getter]
    fn a(&self) -> Fp16OperandParams {
        self.operands.a.clone()
    }

    #[getter]
    fn b(&self) -> Fp16OperandParams {
        self.operands.b.clone()
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl Fp16PlainProof {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        ancestor_header: IncompleteBlockHeader,
        device: Fp16Device,
        k: u32,
        r: u32,
        a: Fp16OperandParams,
        b: Fp16OperandParams,
        values_a: Fp16MatrixProof,
        values_b: Fp16MatrixProof,
    ) -> Self {
        Self {
            job: Fp16JobParams {
                ancestor_header,
                device,
                k,
                r,
                operands: Sides { a, b },
            },
            values: Sides {
                a: values_a,
                b: values_b,
            },
        }
    }

    #[getter]
    fn ancestor_header(&self) -> IncompleteBlockHeader {
        self.job.ancestor_header
    }

    #[getter]
    fn device(&self) -> Fp16Device {
        self.job.device
    }

    #[getter]
    fn k(&self) -> u32 {
        self.job.k
    }

    #[getter]
    fn r(&self) -> u32 {
        self.job.r
    }

    #[getter]
    fn a(&self) -> Fp16OperandParams {
        self.job.operands.a.clone()
    }

    #[getter]
    fn b(&self) -> Fp16OperandParams {
        self.job.operands.b.clone()
    }

    #[getter]
    fn values_a(&self) -> Fp16MatrixProof {
        self.values.a.clone()
    }

    #[getter]
    fn values_b(&self) -> Fp16MatrixProof {
        self.values.b.clone()
    }

    #[getter]
    fn min_cert_version(&self) -> u32 {
        crate::ffi::CertificateVersion::PlainFp16 as u32
    }

    #[pyo3(name = "to_bytes")]
    fn py_to_bytes(&self) -> pyo3::PyResult<Vec<u8>> {
        Self::to_bytes(self).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(name = "from_bytes")]
    fn py_from_bytes(data: Vec<u8>) -> pyo3::PyResult<Self> {
        Self::from_bytes(&data).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn to_base64(&self) -> pyo3::PyResult<String> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        Ok(STANDARD.encode(self.py_to_bytes()?))
    }

    #[staticmethod]
    fn from_base64(data: &str) -> pyo3::PyResult<Self> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        let bytes = STANDARD
            .decode(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Base64 decode failed: {e}")))?;
        Self::from_bytes(&bytes).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[cfg(test)]
mod seed_chain_dump {
    //! Additive, test-only oracle dump for the host-side FP16 miner seed chain
    //! (`pearl_gemm.fp16_miner._seed_chain`). For a fixed (header, operands,
    //! params) it writes the per-side opening keys, the committed Merkle roots,
    //! the `encode_p_a`/`encode_p_b` public-parameter encodings, the B-then-A
    //! noise seeds, and the jackpot `pow_key` -- so the Python host reproduction
    //! can be asserted bit-for-bit. Ignored by default; run with:
    //! ```text
    //! PEARL_FP16_SEEDCHAIN_OUT=/path/seedchain.txt cargo test -p zk-pow --lib -- \
    //!   api::fp16::plain_proof::seed_chain_dump::dump_seed_chain --ignored --exact --nocapture
    //! ```
    use super::*;
    use crate::v5::api::commitment::commit_operand;
    use crate::v5::api::noise::{commitment_keys, key_a, key_b, noise_seeds, subkey};
    use crate::v5::api::params::Fp16Device;
    use crate::v4::api::public_params::HashId;
    use crate::v4::api::layout::AxisPattern;
    use crate::v4::api::layout::DimType::{Blake, Fold};
    use crate::v4::api::primitives::{IncompleteBlockHeader, Sides};

    /// Deterministic u16 fill, byte-identical to the Python harness formula:
    /// `((i * 2654435761 + 1013904223) (mod 2^64)) >> 13 & 0xFFFF`.
    fn fill(n: usize, salt: u64) -> Vec<u16> {
        (0..n)
            .map(|i| {
                let x = (i as u64)
                    .wrapping_add(salt)
                    .wrapping_mul(2654435761)
                    .wrapping_add(1013904223);
                ((x >> 13) & 0xFFFF) as u16
            })
            .collect()
    }

    #[test]
    #[ignore = "dumps the FP16 miner seed-chain oracle vectors; run explicitly"]
    fn dump_seed_chain() {
        use std::fmt::Write as _;

        fn hx(b: &[u8]) -> String {
            b.iter().map(|x| format!("{x:02x}")).collect()
        }

        const M: usize = 8;
        const N: usize = 128;
        const K: usize = 256;
        const R: u32 = 32;
        const HASH: HashId = HashId::Blake3Chunk1024;
        const NBITS: u32 = 0x207f_ffff;

        let header = IncompleteBlockHeader::new_for_test(NBITS);
        // Depth-0 ancestor coincidence: the miner proposes at depth 0.
        let ancestor = header;

        let a_full = fill(M * K, 0x1111);
        let b_full = fill(N * K, 0x2222);

        let rp = AxisPattern::new(&[(4, Blake)]).unwrap(); // h = 4
        let cp = AxisPattern::new(&[(4, Blake), (16, Fold)]).unwrap(); // w = 64

        let keys = commitment_keys(&header, &ancestor);
        assert_eq!(keys.a, key_a(&header));
        assert_eq!(keys.b, key_b(&ancestor));

        let tree_a = commit_operand(&a_full, M, K, HASH, keys.a).unwrap();
        let tree_b = commit_operand(&b_full, N, K, HASH, keys.b).unwrap();
        let root_a = tree_a.root();
        let root_b = tree_b.root();

        let job = Fp16JobParams {
            ancestor_header: ancestor,
            device: Fp16Device::A100,
            k: K as u32,
            r: R,
            operands: Sides {
                a: Fp16OperandParams { num_rows: M as u32, hash_id: HASH, pattern: rp },
                b: Fp16OperandParams { num_rows: N as u32, hash_id: HASH, pattern: cp },
            },
        };
        let p_a = job.encode_p_a();
        let p_b = job.encode_p_b();

        let roots = Sides { a: root_a, b: root_b };
        let p = Sides { a: p_a.clone(), b: p_b.clone() };
        let seeds = noise_seeds(&keys, &roots, &p);
        let pow_key = subkey(b"pearl/v4/FP8/jackpot", Some(&seeds.a));

        let mut out = String::new();
        writeln!(out, "m {M}").unwrap();
        writeln!(out, "n {N}").unwrap();
        writeln!(out, "k {K}").unwrap();
        writeln!(out, "r {R}").unwrap();
        writeln!(out, "nbits {NBITS:#010x}").unwrap();
        writeln!(out, "a_salt 0x1111").unwrap();
        writeln!(out, "b_salt 0x2222").unwrap();
        writeln!(out, "chunk_len {}", HASH.chunk_len()).unwrap();
        writeln!(out, "header {}", hx(&header.to_bytes())).unwrap();
        writeln!(out, "ancestor {}", hx(&ancestor.to_bytes())).unwrap();
        writeln!(out, "p_a_bytes {}", a_full.len()).unwrap();
        writeln!(out, "key_a {}", hx(&keys.a)).unwrap();
        writeln!(out, "key_b {}", hx(&keys.b)).unwrap();
        writeln!(out, "root_a {}", hx(&root_a)).unwrap();
        writeln!(out, "root_b {}", hx(&root_b)).unwrap();
        writeln!(out, "p_a {}", hx(&p_a)).unwrap();
        writeln!(out, "p_b {}", hx(&p_b)).unwrap();
        writeln!(out, "seed_a {}", hx(&seeds.a)).unwrap();
        writeln!(out, "seed_b {}", hx(&seeds.b)).unwrap();
        writeln!(out, "pow_key {}", hx(&pow_key)).unwrap();
        writeln!(out, "pattern_a {}", hx(&job.operands.a.pattern.to_bytes())).unwrap();
        writeln!(out, "pattern_b {}", hx(&job.operands.b.pattern.to_bytes())).unwrap();

        let path = std::env::var("PEARL_FP16_SEEDCHAIN_OUT")
            .unwrap_or_else(|_| "/tmp/fp16_seedchain.txt".to_string());
        std::fs::write(&path, &out).unwrap();
        println!("wrote FP16 seed-chain oracle to {path}");
    }
}
