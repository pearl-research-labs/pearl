//! Miner witness for an FP8 certificate: job parameters and Merkle openings
//! of the selected values and scales. MoE jobs also carry a [`MoeWitness`].
//!
//! Constructing a [`PlainProofV4`] does not validate the statement. Call
//! [`PlainProofV4::parse_proof`].

use anyhow::{Context, Result, ensure};
use pearl_blake3::{MerkleProof, MerkleTree};
use serde::{Deserialize, Serialize};

use crate::api::fp8::openings::PrivateProofParams;
use crate::api::fp8::prequant::BLOCK_SIZE;
#[cfg(feature = "pyo3")]
use crate::api::fp8::public_params::{CommonParams, MoeParams, OperandParams};
use crate::api::fp8::public_params::{HashId, JackpotStatement, JobParams, MoEStatement, PublicParams};
use crate::api::fp8::transcript::{key_a, key_b};
use crate::api::primitives::{Hash256, IncompleteBlockHeader, Sides};
use crate::api::proof_utils::operand_digest_fp10;
use crate::circuit::utils::macros::ensure_eq;
use crate::ffi::plain_proof::{MatrixMerkleProof, parse_axis};

/// Winning expert, the complete offsets list, and an opening of the winner's routing entries.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MoeWitness", get_all))]
pub struct MoeWitness {
    /// Winning expert's index.
    pub w: u16,
    /// Complete offsets list `O`, whose entries delimit each expert's routing slice.
    pub offsets: Vec<u32>,
    /// Claimed root `HO` of the offsets list; recomputed during verification.
    pub offsets_root: Hash256,
    #[serde(
        serialize_with = "MerkleProof::serialize_variable_chunk",
        deserialize_with = "MerkleProof::deserialize_variable_chunk"
    )]
    pub routing: MerkleProof,
}

/// Job parameters, including the ancestor header, and openings of the selected inputs.
///
/// Does not validate the statement. Verification and proving must go through
/// [`Self::parse_proof`], which runs witness-only checks then
/// [`PublicParams::try_new`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "PlainProofV4"))]
pub struct PlainProofV4 {
    #[serde(with = "job_wire")]
    pub job: JobParams,
    #[serde(with = "variable_chunk_sides")]
    pub values: Sides<MatrixMerkleProof>,
    #[serde(with = "variable_chunk_sides")]
    pub scales: Sides<MatrixMerkleProof>,
    pub moe_witness: Option<MoeWitness>,
}

/// Use [`JobParams::to_wire_bytes`] for `ancestor ‖ pB ‖ pA`.
/// `public_data` uses the same parts with commitment digests between them.
mod job_wire {
    use super::JobParams;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(job: &JobParams, serializer: S) -> Result<S::Ok, S::Error> {
        job.to_wire_bytes().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<JobParams, D::Error> {
        let wire = Vec::<u8>::deserialize(deserializer)?;
        JobParams::from_wire_bytes(&wire).map_err(serde::de::Error::custom)
    }
}

/// Use [`MerkleProof::serialize_variable_chunk`] for each opening to preserve its chunk size.
mod variable_chunk_sides {
    use super::{MatrixMerkleProof, MerkleProof, Sides};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// `serialize_with` is called with `&T`; here `T = &MerkleProof`.
    fn serialize_variable_ref<S: Serializer>(proof: &&MerkleProof, serializer: S) -> Result<S::Ok, S::Error> {
        proof.serialize_variable_chunk(serializer)
    }

    pub fn serialize<S: Serializer>(sides: &Sides<MatrixMerkleProof>, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Matrix<'a> {
            #[serde(serialize_with = "serialize_variable_ref")]
            proof: &'a MerkleProof,
            row_indices: &'a [usize],
        }
        // Use the same `Sides<Matrix>` field order for encoding and decoding.
        Sides {
            a: Matrix {
                proof: &sides.a.proof,
                row_indices: &sides.a.row_indices,
            },
            b: Matrix {
                proof: &sides.b.proof,
                row_indices: &sides.b.row_indices,
            },
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Sides<MatrixMerkleProof>, D::Error> {
        #[derive(Deserialize)]
        struct Matrix {
            #[serde(deserialize_with = "MerkleProof::deserialize_variable_chunk")]
            proof: MerkleProof,
            row_indices: Vec<usize>,
        }
        let Sides { a, b } = Sides::<Matrix>::deserialize(deserializer)?;
        Ok(Sides {
            a: MatrixMerkleProof {
                proof: a.proof,
                row_indices: a.row_indices,
            },
            b: MatrixMerkleProof {
                proof: b.proof,
                row_indices: b.row_indices,
            },
        })
    }
}

impl PlainProofV4 {
    /// Serialize with fixed-width integer encoding.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        use bincode::Options;
        bincode::options()
            .with_fixint_encoding()
            .serialize(self)
            .context("serialize PlainProofV4")
    }

    /// Inverse of [`Self::to_bytes`]; rejects trailing bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        use bincode::Options;
        bincode::options()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize(bytes)
            .context("deserialize PlainProofV4")
    }

    /// Validate the witness and return its authenticated private inputs and public statement.
    /// Deserialization alone does not perform these checks.
    ///
    /// The caller supplies `proposed_header` and must authenticate the proof's
    /// `job.ancestor_header`, which determines the B-side keys.
    pub fn parse_proof(&self, proposed_header: &IncompleteBlockHeader) -> Result<(PrivateProofParams, PublicParams)> {
        self.check_shape()?;

        let (inner_a, inner_b, moe_statement) = self.moe_projection()?;
        let t_rows = parse_axis(&inner_a, &self.job.operands.a.pattern)?;
        let t_cols = parse_axis(&inner_b, &self.job.operands.b.pattern)?;

        // Keys must be derived before the statement is built: they key the per-side folds below.
        let keys = Sides {
            a: key_a(proposed_header),
            b: key_b(&self.job.ancestor_header),
        };

        let public = PublicParams::try_new(
            self.job.clone(),
            JackpotStatement {
                tile_bases: Sides { a: t_rows, b: t_cols },
                hash_jackpot: [0u8; 32],
                // The per-side folds of the claimed tree roots; `verify_openings`
                // below authenticates the roots against the openings.
                hash_a: operand_digest_fp10(&self.values.a.proof.root, &self.scales.a.proof.root, &keys.a),
                hash_b: operand_digest_fp10(&self.values.b.proof.root, &self.scales.b.proof.root, &keys.b),
            },
            moe_statement,
        )?;

        let private = public.verify_openings(
            proposed_header,
            &self.values.a,
            &self.scales.a,
            &self.values.b,
            &self.scales.b,
            self.moe_witness.as_ref(),
        )?;

        Ok((private, public))
    }

    /// Witness-only shape: tree sizes, values/scales index agreement, full `O`.
    fn check_shape(&self) -> Result<()> {
        ensure!(
            self.moe_witness.is_some() == self.job.moe.is_some(),
            "moe_witness must be present iff MoeParams is present"
        );
        ensure_eq!(
            self.values.a.row_indices,
            self.scales.a.row_indices,
            "A values and scales must open the same rows"
        );
        ensure_eq!(
            self.values.b.row_indices,
            self.scales.b.row_indices,
            "B values and scales must open the same rows"
        );

        let m = self.job.operands.a.num_rows as usize;
        let n = self.job.operands.b.num_rows as usize;
        let k = self.job.common.k as usize;

        check_tree_leaves(&self.values.a.proof, &[m, k], self.job.operands.a.hash_id, "A values")?;
        check_tree_leaves(
            &self.scales.a.proof,
            &[m, scale_row_bytes(k)],
            self.job.operands.a.hash_id,
            "A scales",
        )?;
        check_tree_leaves(&self.values.b.proof, &[n, k], self.job.operands.b.hash_id, "B values")?;
        check_tree_leaves(
            &self.scales.b.proof,
            &[n, scale_row_bytes(k)],
            self.job.operands.b.hash_id,
            "B scales",
        )?;

        if let (Some(moe), Some(witness)) = (self.job.moe, &self.moe_witness) {
            ensure!(moe.experts >= 1, "MoE proofs must declare at least one expert");
            let e = moe.experts as usize;
            ensure_eq!(
                witness.offsets.len(),
                e,
                "offset list O must hold one cumulative count per expert"
            );
            // `0 <= O_0 <= ... <= O_{e-1}` (the lower bound holds by the u32 encoding).
            ensure!(
                witness.offsets.windows(2).all(|w| w[0] <= w[1]),
                "routing offsets O must be monotonically non-decreasing"
            );

            let o_last = *witness.offsets.last().expect("|O| = e >= 1");
            check_tree_leaves(
                &witness.routing,
                &[o_last as usize, std::mem::size_of::<u32>()],
                moe.hash_id_r,
                "routing",
            )?;
        }
        Ok(())
    }

    fn moe_projection(&self) -> Result<(Vec<u32>, Vec<u32>, Option<MoEStatement>)> {
        let a_outer: Vec<u32> = self.values.a.row_indices.iter().map(|&x| x as u32).collect();
        let b_rows: Vec<u32> = self.values.b.row_indices.iter().map(|&x| x as u32).collect();

        let Some(moe) = self.job.moe else {
            return Ok((a_outer, b_rows, None));
        };
        let witness = self
            .moe_witness
            .as_ref()
            .expect("check_shape requires moe_witness with MoeParams");

        let w = witness.w;
        let e = u32::from(moe.experts);
        ensure!(u32::from(w) < e, "winner expert out of range for e={e} || w={w}");
        let offsets = &witness.offsets;
        ensure_eq!(offsets.len(), e as usize, "offset list O must hold one count per expert");
        let o_w_prev = if w == 0 { 0 } else { offsets[(w - 1) as usize] };
        let o_w = offsets[w as usize];
        let o_last = *offsets.last().expect("|O| = e >= 1");

        let winner_routing = extract_u32_span(&witness.routing, o_w_prev, o_w)?;
        ensure!(
            winner_routing.len() == (o_w - o_w_prev) as usize,
            "R[w] length must equal O_w - O_{{w-1}} || |R[w]|={} s_w={}",
            winner_routing.len(),
            o_w - o_w_prev
        );
        ensure!(
            winner_routing.iter().all(|&tok| tok < self.job.operands.a.num_rows),
            "R[w] must be a subset of [0, m) || m={}",
            self.job.operands.a.num_rows
        );
        // Map each selected global A row to its position in the winner's routing list.
        let inner_a = find_subset_in_sorted_array(&winner_routing, &a_outer)?;

        ensure!(
            self.job.operands.b.num_rows.is_multiple_of(u32::from(moe.experts)),
            "n must be divisible by e || n={} e={}",
            self.job.operands.b.num_rows,
            moe.experts
        );
        let rows_per_expert = self.job.operands.b.num_rows / u32::from(moe.experts);
        let weight_col_offset = u32::from(w)
            .checked_mul(rows_per_expert)
            .ok_or_else(|| anyhow::anyhow!("w * η overflows u32"))?;
        for &idx in &self.values.b.row_indices {
            let idx = idx as u32;
            ensure!(
                idx >= weight_col_offset && idx < weight_col_offset + rows_per_expert,
                "B column index {idx} out of range for expert {w} (expected [{weight_col_offset}, {}))",
                weight_col_offset + rows_per_expert
            );
        }
        let inner_b: Vec<u32> = b_rows.iter().map(|&idx| idx - weight_col_offset).collect();

        Ok((
            inner_a,
            inner_b,
            Some(MoEStatement {
                w,
                o_w_prev,
                o_w,
                o_last,
                hash_routing: witness.routing.root,
                hash_offsets: witness.offsets_root,
                i_a: a_outer,
            }),
        ))
    }
}

fn scale_row_bytes(k: usize) -> usize {
    2 * (k / BLOCK_SIZE)
}

/// Commit cumulative offsets `O` as little-endian u32s, zero-padded to
/// `hash_id`'s chunk size and hashed under `keyA`. This is a chunk-tree root,
/// which can differ from a flat keyed BLAKE3 digest for smaller chunk sizes.
pub(crate) fn offsets_root(offsets: &[u32], hash_id: HashId, key: Hash256) -> Result<Hash256> {
    let bytes: Vec<u8> = offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
    Ok(MerkleTree::with_chunk_len(&hash_id.pad(&bytes), key, hash_id.chunk_len())?.root())
}

fn check_tree_leaves(proof: &MerkleProof, dims: &[usize], hash_id: HashId, label: &str) -> Result<()> {
    let bytes = dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| anyhow::anyhow!("{label}: declared dimensions {dims:?} overflow usize"))?;
    let chunk = hash_id.chunk_len();
    if let Some(leaf) = proof.leaf_data.first() {
        ensure_eq!(leaf.len(), chunk, "{label}: leaf length must match hash_id chunk_len");
    }
    let expected = hash_id.padded_len(bytes) / chunk;
    ensure_eq!(
        proof.total_leaves,
        expected,
        "{label}: Merkle tree declares {} leaves but {dims:?} imply {expected}",
        proof.total_leaves
    );
    Ok(())
}

fn extract_u32_span(proof: &MerkleProof, start: u32, end: u32) -> Result<Vec<u32>> {
    ensure!(end >= start, "routing slice end precedes start");
    let width = std::mem::size_of::<u32>();
    let byte_start = start as usize * width;
    let byte_len = (end - start) as usize * width;
    let bytes = proof
        .extract_bytes(byte_start, byte_len)
        .with_context(|| format!("extract routing entries [{start}, {end})"))?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

/// Checks `arr[i] < arr[i + 1]` for every adjacent pair, and each
/// `subset[i]` appears in `arr`.
/// Returns the indices in `arr` at which the `subset` entries appear, in `subset` order.
fn find_subset_in_sorted_array(sorted_values: &[u32], subset: &[u32]) -> Result<Vec<u32>> {
    ensure!(
        sorted_values.windows(2).all(|w| w[0] < w[1]),
        "winner routing slice R[w] must be strictly increasing"
    );
    subset
        .iter()
        .map(|&s| {
            let i = sorted_values.partition_point(|&a| a < s);
            ensure!(
                i < sorted_values.len() && sorted_values[i] == s,
                "outer index {s} missing from the winner routing slice"
            );
            u32::try_from(i).context("routing slice longer than u32::MAX")
        })
        .collect()
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl MoeWitness {
    #[new]
    fn py_new(w: u16, offsets: Vec<u32>, offsets_root: Hash256, routing: MerkleProof) -> Self {
        Self {
            w,
            offsets,
            offsets_root,
            routing,
        }
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl PlainProofV4 {
    #[new]
    #[pyo3(signature = (ancestor_header, common, a, b, values_a, values_b, scales_a, scales_b, moe=None, moe_witness=None))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        ancestor_header: IncompleteBlockHeader,
        common: CommonParams,
        a: OperandParams,
        b: OperandParams,
        values_a: MatrixMerkleProof,
        values_b: MatrixMerkleProof,
        scales_a: MatrixMerkleProof,
        scales_b: MatrixMerkleProof,
        moe: Option<MoeParams>,
        moe_witness: Option<MoeWitness>,
    ) -> Self {
        Self {
            job: JobParams {
                ancestor_header,
                common,
                operands: Sides { a, b },
                moe,
            },
            values: Sides {
                a: values_a,
                b: values_b,
            },
            scales: Sides {
                a: scales_a,
                b: scales_b,
            },
            moe_witness,
        }
    }

    #[getter]
    fn ancestor_header(&self) -> IncompleteBlockHeader {
        self.job.ancestor_header
    }

    #[getter]
    fn common(&self) -> CommonParams {
        self.job.common
    }

    #[getter]
    fn a(&self) -> OperandParams {
        self.job.operands.a.clone()
    }

    #[getter]
    fn b(&self) -> OperandParams {
        self.job.operands.b.clone()
    }

    #[getter]
    fn moe(&self) -> Option<MoeParams> {
        self.job.moe
    }

    #[getter]
    fn values_a(&self) -> MatrixMerkleProof {
        self.values.a.clone()
    }

    #[getter]
    fn values_b(&self) -> MatrixMerkleProof {
        self.values.b.clone()
    }

    #[getter]
    fn scales_a(&self) -> MatrixMerkleProof {
        self.scales.a.clone()
    }

    #[getter]
    fn scales_b(&self) -> MatrixMerkleProof {
        self.scales.b.clone()
    }

    #[getter]
    fn moe_witness(&self) -> Option<MoeWitness> {
        self.moe_witness.clone()
    }

    #[getter]
    fn min_cert_version(&self) -> u32 {
        crate::ffi::plain_proof::CertificateVersion::PlainFp8 as u32
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
mod tests {
    use super::*;
    use crate::api::fp8::public_params::{CommonParams, Device, MoeParams, OperandParams, Quant};
    use crate::api::layout::AxisPattern;
    use crate::api::layout::DimType::Blake;
    use crate::api::primitives::{Hash256, IncompleteBlockHeader};
    use pearl_blake3::{MerkleTree, blake3_digest};

    const KEY: Hash256 = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];

    fn le_bytes(offsets: &[u32]) -> Vec<u8> {
        offsets.iter().flat_map(|o| o.to_le_bytes()).collect()
    }

    fn hex(digest: &Hash256) -> String {
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn ho_single_chunk_equals_flat_keyed_digest() {
        // 3 offsets = 12 bytes pad to one 1024 chunk: the chunk-tree root IS
        // the flat keyed BLAKE3 digest of the padded bytes.
        let offsets = [3u32, 7, 9];
        let hash_id = HashId::Blake3Chunk1024;
        let root = offsets_root(&offsets, hash_id, KEY).unwrap();
        assert_eq!(root, blake3_digest(&hash_id.pad(&le_bytes(&offsets)), Some(KEY)));
        assert_eq!(hex(&root), "53aa5cb0b531c648f26f806b927438ad8790b843cbb07a4f367f73d91d0129dd");
    }

    #[test]
    fn ho_multi_leaf_small_chunks_diverge_from_flat_digest() {
        // 40 offsets = 160 bytes pad to two 128-byte leaves: chunked semantics
        // must kick in, so the root differs from the flat keyed digest.
        let offsets: Vec<u32> = (1..=40).collect();
        let hash_id = HashId::Blake3Chunk128;
        let root = offsets_root(&offsets, hash_id, KEY).unwrap();
        assert_ne!(root, blake3_digest(&hash_id.pad(&le_bytes(&offsets)), Some(KEY)));
        assert_eq!(hex(&root), "57b274de6735d0657ffbc0f3cf81016d56e45e6440c017d5a52315c5f676ddcc");
    }

    #[test]
    fn ho_multi_chunk_1024_matches_native_blake3() {
        // 300 offsets = 1200 bytes pad to 2048 = two native chunks: with the
        // native 1024 chunk length the tree root equals plain keyed BLAKE3.
        let offsets: Vec<u32> = (1..=300).collect();
        let hash_id = HashId::Blake3Chunk1024;
        let root = offsets_root(&offsets, hash_id, KEY).unwrap();
        assert_eq!(root, blake3_digest(&hash_id.pad(&le_bytes(&offsets)), Some(KEY)));
        assert_eq!(hex(&root), "068bd1fb990aad33da5f9637e547474a61750d0ce655cfae94013f6034b5d0bd");
    }

    #[test]
    fn ho_depends_on_key_and_hash_id() {
        let offsets = [3u32, 7, 9];
        let base = offsets_root(&offsets, HashId::Blake3Chunk1024, KEY).unwrap();
        assert_ne!(base, offsets_root(&offsets, HashId::Blake3Chunk1024, [9u8; 32]).unwrap());
        // Padding length follows hash_id, so every chunk size hashes differently.
        assert_ne!(base, offsets_root(&offsets, HashId::Blake3Chunk128, KEY).unwrap());
    }

    fn tiny_tree(data_len: usize, hash_id: HashId) -> MerkleProof {
        let data: Vec<u8> = (0..data_len).map(|i| (i as u8).wrapping_mul(31)).collect();
        MerkleTree::with_chunk_len(&hash_id.pad(&data), KEY, hash_id.chunk_len())
            .unwrap()
            .get_multileaf_proof(&[0])
    }

    /// Valid non-MoE shapes let tests isolate the MoE shape checks.
    /// The openings are dummy data and need not pass `parse_proof`.
    fn tiny_moe_proof(experts: u16, offsets: Vec<u32>) -> PlainProofV4 {
        let hash_id = HashId::Blake3Chunk1024;
        let (m, n, k) = (4usize, 4usize, 256usize);
        let pattern = AxisPattern::new(&[(2, Blake)]).unwrap();
        let matrix = |bytes: usize| MatrixMerkleProof {
            proof: tiny_tree(bytes, hash_id),
            row_indices: vec![0],
        };
        let routing_entries = offsets.last().copied().unwrap_or(0).max(1) as usize;
        PlainProofV4 {
            job: JobParams {
                ancestor_header: IncompleteBlockHeader::new_for_test(0x207FFFFF),
                common: CommonParams {
                    k: k as u32,
                    r: 32,
                    quant: Quant::Fp8E4M3Prequant,
                    device: Device::B200,
                },
                operands: Sides {
                    a: OperandParams {
                        num_rows: m as u32,
                        hash_id,
                        pattern: pattern.clone(),
                    },
                    b: OperandParams {
                        num_rows: n as u32,
                        hash_id,
                        pattern,
                    },
                },
                moe: Some(MoeParams {
                    experts,
                    hash_id_r: hash_id,
                    hash_id_o: hash_id,
                }),
            },
            values: Sides {
                a: matrix(m * k),
                b: matrix(n * k),
            },
            scales: Sides {
                a: matrix(m * scale_row_bytes(k)),
                b: matrix(n * scale_row_bytes(k)),
            },
            moe_witness: Some(MoeWitness {
                w: 0,
                // The tree builder cannot hash an empty list (`experts = 0` probes).
                offsets_root: match offsets.is_empty() {
                    true => [0u8; 32],
                    false => offsets_root(&offsets, hash_id, KEY).unwrap(),
                },
                offsets,
                routing: tiny_tree(routing_entries * 4, hash_id),
            }),
        }
    }

    #[test]
    fn check_shape_accepts_a_well_formed_moe_witness() {
        tiny_moe_proof(2, vec![2, 4]).check_shape().unwrap();
    }

    #[test]
    fn check_shape_rejects_zero_experts_without_panicking() {
        // Reject before calling `offsets.last().expect(..)`.
        let err = tiny_moe_proof(0, vec![]).check_shape().unwrap_err();
        assert!(err.to_string().contains("at least one expert"), "{err}");
    }

    #[test]
    fn check_shape_rejects_wrong_offsets_count() {
        // |O| must equal e: 300 declared experts, two disclosed counts.
        let mut proof = tiny_moe_proof(2, vec![2, 4]);
        proof.job.moe.as_mut().unwrap().experts = 300;
        let err = proof.check_shape().unwrap_err();
        assert!(err.to_string().contains("one cumulative count per expert"), "{err}");
    }

    #[test]
    fn check_shape_rejects_non_monotone_offsets() {
        let err = tiny_moe_proof(2, vec![4, 2]).check_shape().unwrap_err();
        assert!(err.to_string().contains("non-decreasing"), "{err}");
    }

    #[test]
    fn check_shape_rejects_moe_params_without_witness() {
        let mut proof = tiny_moe_proof(2, vec![2, 4]);
        proof.moe_witness = None;
        let err = proof.check_shape().unwrap_err();
        assert!(err.to_string().contains("iff MoeParams"), "{err}");

        let mut proof = tiny_moe_proof(2, vec![2, 4]);
        proof.job.moe = None;
        assert!(proof.check_shape().is_err(), "witness without MoeParams must fail");
    }

    #[test]
    fn check_shape_rejects_routing_tree_smaller_than_o_last() {
        // Grow O_last to 300 after the fact: 1200 routing bytes imply two
        // 1024-byte leaves, but the witness tree still carries one.
        let mut proof = tiny_moe_proof(2, vec![2, 4]);
        proof.moe_witness.as_mut().unwrap().offsets = vec![2, 300];
        let err = proof.check_shape().unwrap_err();
        assert!(err.to_string().contains("routing"), "{err}");
    }

    #[test]
    fn parse_proof_rejects_winner_out_of_range() {
        let header = IncompleteBlockHeader::new_for_test(0x207FFFFF);
        let mut proof = tiny_moe_proof(2, vec![2, 4]);
        proof.moe_witness.as_mut().unwrap().w = 2;
        let err = proof.parse_proof(&header).unwrap_err();
        assert!(err.to_string().contains("winner expert out of range"), "{err}");
    }

    #[test]
    fn find_subset_checks_order_and_membership() {
        assert_eq!(find_subset_in_sorted_array(&[2, 5, 9], &[5, 2]).unwrap(), vec![1, 0]);
        assert!(
            find_subset_in_sorted_array(&[2, 2, 9], &[2]).is_err(),
            "R[w] must be strictly increasing"
        );
        assert!(
            find_subset_in_sorted_array(&[2, 5, 9], &[7]).is_err(),
            "outer indices outside R[w] must fail"
        );
    }

    #[test]
    fn moe_witness_survives_the_wire_roundtrip() {
        let proof = tiny_moe_proof(2, vec![2, 4]);
        let parsed = PlainProofV4::from_bytes(&proof.to_bytes().unwrap()).unwrap();
        let (orig, back) = (proof.moe_witness.unwrap(), parsed.moe_witness.unwrap());
        assert_eq!(back.w, orig.w);
        assert_eq!(back.offsets, orig.offsets);
        assert_eq!(back.offsets_root, orig.offsets_root);
        assert_eq!(back.routing.root, orig.routing.root);
        assert_eq!(back.routing.chunk_len(), orig.routing.chunk_len());
        assert_eq!(parsed.job.moe, proof.job.moe);
    }
}
