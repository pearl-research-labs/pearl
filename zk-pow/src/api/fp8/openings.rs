//! Native Merkle opening verification (`VerifyOpen_X`).
//!
//! Authenticate the selected int8 values and BF16 scales against the operand
//! commitments, then extract [`PrequantOperand`] rows. MoE jobs also authenticate
//! the winner's routing entries and the complete offsets list.
//!
//! The compiled Blake3 program determines exactly which leaves and sibling
//! hashes each opening must contain, and their order in the ZK trace.

use anyhow::{Context, Result, ensure};
use blake3::BLOCK_LEN;
use pearl_blake3::{BLAKE3_DIGEST_SIZE, MerkleProof};

use crate::api::fp8::dtype::check_not_nan_or_inf_bf16;
use crate::api::fp8::plain_proof::{MoeWitness, offsets_root};
use crate::api::fp8::prequant::{BLOCK_SIZE, PrequantOperand, PrequantSlice, open_prequant};
use crate::api::fp8::public_params::{HashId, PublicParams};
use crate::api::primitives::{Hash256, IncompleteBlockHeader, Sides};
use crate::circuit::chip::blake3::program::{AuxiliaryCvLocation, AuxiliaryMsgLocation, BlakeProgram, DWORD_SIZE, ProofSource};
use crate::ensure_eq;
use crate::ffi::plain_proof::MatrixMerkleProof;

/// Private inputs from [`PublicParams::verify_openings`].
#[derive(Debug, Clone)]
pub struct PrivateProofParams {
    /// Selected A and B rows, each containing `common_dim` values.
    pub operands: Sides<PrequantOperand>,
    /// Opened 64-byte blocks of u32 token indices. A block may also contain
    /// entries from experts adjacent to the winner.
    pub s_routing: Vec<Vec<u8>>,
    /// Complete, zero-padded offsets list in 64-byte blocks.
    pub s_offsets: Vec<Vec<u8>>,
    /// Additional leaf messages needed by the Blake3 trace.
    pub external_msgs: Vec<[u8; 64]>,
    /// Merkle sibling hashes needed by the Blake3 trace.
    pub external_cvs: Vec<Hash256>,
}

/// Decode routing or offsets blocks into little-endian u32 words, preserving stream order.
pub(crate) fn stream_words(strips: &[Vec<u8>]) -> Vec<u32> {
    strips
        .iter()
        .flat_map(|strip| strip.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)))
        .collect()
}

impl PublicParams {
    /// Verify operand and MoE commitments, then extract the private inputs.
    /// Used by plaintext verification and ZK proving; the ZK verifier checks
    /// these commitments through the recursive proof.
    ///
    /// [`Self::commitment_keys`] derives A's key from the proposed header and
    /// B's from the ancestor. MoE's `HR` is checked through routing openings;
    /// `HO` is recomputed from the full offsets list.
    pub(crate) fn verify_openings(
        &self,
        proposed_header: &IncompleteBlockHeader,
        a: &MatrixMerkleProof,
        a_scales: &MatrixMerkleProof,
        bt: &MatrixMerkleProof,
        bt_scales: &MatrixMerkleProof,
        moe_witness: Option<&MoeWitness>,
    ) -> Result<PrivateProofParams> {
        let keys = self.commitment_keys(proposed_header);
        ensure!(
            moe_witness.is_some() == self.moe_statement().is_some(),
            "MoE witness must be present iff the statement is MoE"
        );
        let routing = moe_witness.map(|w| &w.routing);

        let a_rows: Vec<usize> = self.a_rows_indices().iter().map(|&x| x as usize).collect();
        let b_rows: Vec<usize> = self.b_rows_indices().iter().map(|&x| x as usize).collect();
        ensure_eq!(a.row_indices, a_rows, "A values row indices must match the statement tile");
        ensure_eq!(bt.row_indices, b_rows, "B values row indices must match the statement tile");
        ensure_eq!(
            a.row_indices,
            a_scales.row_indices,
            "int8 values and BF16 scales trees must open the same rows"
        );
        ensure_eq!(
            bt.row_indices,
            bt_scales.row_indices,
            "int8 values and BF16 scales trees must open the same rows"
        );

        let k = self.common_dim() as usize;
        ensure!(
            k.is_multiple_of(BLOCK_SIZE),
            "common_dim {k} must be a multiple of block size {BLOCK_SIZE}"
        );
        let value_row_bytes = self.value_row_bytes();
        let scale_row_bytes = self.scale_row_bytes();
        ensure!(
            value_row_bytes.is_multiple_of(DWORD_SIZE) && scale_row_bytes.is_multiple_of(DWORD_SIZE),
            "prequant rows must align to {DWORD_SIZE}-byte dwords: common_dim {k} must be a multiple of {}",
            DWORD_SIZE * BLOCK_SIZE / 2
        );

        // Use the same opening schedule and auxiliary locations as the ZK trace.
        let (program, message_locations, cv_locations) = BlakeProgram::compile(self);
        let schedules = TreeSchedules::new(self, &cv_locations)?;

        check_minimal_opening(&a.proof, &schedules.values.a, keys.a, "int8 values A")?;
        check_minimal_opening(&a_scales.proof, &schedules.scales.a, keys.a, "BF16 scales A")?;
        check_minimal_opening(&bt.proof, &schedules.values.b, keys.b, "int8 values B")?;
        check_minimal_opening(&bt_scales.proof, &schedules.scales.b, keys.b, "BF16 scales B")?;
        if let Some(witness) = moe_witness {
            let routing_schedule = schedules
                .routing
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("MoE witness present but the program compiled no routing tree"))?;
            check_minimal_opening(&witness.routing, routing_schedule, keys.a, "routing")?;
            check_full_offsets(self, &witness.offsets, keys.a)?;
        }

        let operands = Sides {
            a: PrequantOperand {
                values: extract_prequant_slice(&a.row_indices, value_row_bytes, &a.proof)?,
                scales: extract_prequant_slice(&a.row_indices, scale_row_bytes, &a_scales.proof)?,
            },
            b: PrequantOperand {
                values: extract_prequant_slice(&bt.row_indices, value_row_bytes, &bt.proof)?,
                scales: extract_prequant_slice(&bt.row_indices, scale_row_bytes, &bt_scales.proof)?,
            },
        };
        reject_nonfinite_prequant(&operands.a, k)?;
        reject_nonfinite_prequant(&operands.b, k)?;

        let s_routing = if let Some(routing) = routing {
            let strips = extract_routing_strips(routing, self)?;
            ensure_eq!(
                strips.len(),
                program.num_routing_strips,
                "MoE s_routing strips must match num_routing_strips"
            );
            strips
        } else {
            vec![]
        };
        let s_offsets: Vec<Vec<u8>> = if let Some(witness) = moe_witness {
            // The program ingests the zero-padded u32-LE stream of `O` in
            // 64-byte strips — the same padded bytes the chunk tree hashes.
            let mut bytes: Vec<u8> = witness.offsets.iter().flat_map(|w| w.to_le_bytes()).collect();
            let padded_len = program.num_offsets_strips * BLOCK_LEN;
            ensure!(bytes.len() <= padded_len, "offset list exceeds the compiled strip budget");
            bytes.resize(padded_len, 0);
            bytes.as_chunks::<BLOCK_LEN>().0.iter().map(|v| v.to_vec()).collect()
        } else {
            vec![]
        };

        let external_msgs = extract_aux_msgs(&message_locations, a, a_scales, bt, bt_scales, routing)?;
        let external_cvs = compute_external_cvs(&cv_locations, &schedules, a, a_scales, bt, bt_scales, routing)?;
        ensure_eq!(
            external_msgs.len(),
            program.num_auxiliary_msgs,
            "auxiliary messages must match the compiled schedule"
        );
        ensure_eq!(
            external_cvs.len(),
            program.num_auxiliary_cvs,
            "auxiliary CVs must match the compiled schedule"
        );

        let private = PrivateProofParams {
            operands,
            s_routing,
            s_offsets,
            external_msgs,
            external_cvs,
        };

        let moe_roots = self.moe_statement().map(|s| (s.hash_routing, s.hash_offsets));
        let roots = program.evaluate_blake(keys, &private, moe_roots)?;
        ensure_eq!(roots.hash_a, a.proof.root, "int8 values: A-side Merkle root mismatch");
        ensure_eq!(roots.hash_b, bt.proof.root, "int8 values: B-side Merkle root mismatch");
        let a_scales_root = roots
            .hash_a_scales
            .ok_or_else(|| anyhow::anyhow!("prequant program produced no A scales root"))?;
        let b_scales_root = roots
            .hash_b_scales
            .ok_or_else(|| anyhow::anyhow!("prequant program produced no B scales root"))?;
        ensure_eq!(a_scales_root, a_scales.proof.root, "BF16 scales: A-side Merkle root mismatch");
        ensure_eq!(
            b_scales_root,
            bt_scales.proof.root,
            "BF16 scales: B-side Merkle root mismatch"
        );

        Ok(private)
    }
}

/// Leaves and sibling byte ranges required by [`BlakeProgram::compile`]
/// for one committed tree.
pub(crate) struct TreeSchedule {
    pub(crate) hash_id: HashId,
    pub(crate) padded_len: usize,
    /// Exactly the leaf chunks required by the compiled schedule.
    pub(crate) leaf_indices: Vec<usize>,
    /// Byte ranges of the required unopened sibling subtrees, sorted by position.
    pub(crate) sibling_ranges: Vec<(usize, usize)>,
}

impl TreeSchedule {
    /// Opened leaves are the complement of the unopened sibling subtree ranges.
    pub(crate) fn new(cv_locs: &[AuxiliaryCvLocation], source: ProofSource, tree_bytes: usize, hash_id: HashId) -> Self {
        let chunk_len = hash_id.chunk_len();
        let padded_len = hash_id.padded_len(tree_bytes);
        let mut sibling_ranges: Vec<(usize, usize)> = cv_locs
            .iter()
            .filter(|loc| loc.source == source)
            .map(|loc| (loc.global_start, loc.global_end))
            .collect();
        sibling_ranges.sort_unstable();
        debug_assert!(
            sibling_ranges
                .iter()
                .all(|&(s, e)| s < e && e <= padded_len && s % chunk_len == 0 && e % chunk_len == 0),
            "compiled aux-CV ranges must be chunk-aligned subtrees"
        );
        let mut leaf_indices = Vec::new();
        let mut cursor = 0;
        for &(start, end) in &sibling_ranges {
            leaf_indices.extend(cursor / chunk_len..start / chunk_len);
            cursor = end;
        }
        leaf_indices.extend(cursor / chunk_len..padded_len / chunk_len);
        Self {
            hash_id,
            padded_len,
            leaf_indices,
            sibling_ranges,
        }
    }
}

/// Schedules for trees authenticated through Merkle openings.
/// The offsets list is fully disclosed and hashed directly, so it needs no schedule.
struct TreeSchedules {
    values: Sides<TreeSchedule>,
    scales: Sides<TreeSchedule>,
    routing: Option<TreeSchedule>,
}

impl TreeSchedules {
    fn new(params: &PublicParams, cv_locs: &[AuxiliaryCvLocation]) -> Result<Self> {
        let schedule = |source, rows: usize, row_bytes: usize, hash_id, label: &str| -> Result<TreeSchedule> {
            let tree_bytes = rows
                .checked_mul(row_bytes)
                .ok_or_else(|| anyhow::anyhow!("{label}: matrix byte length overflow"))?;
            Ok(TreeSchedule::new(cv_locs, source, tree_bytes, hash_id))
        };
        let m = params.m() as usize;
        let n = params.n() as usize;
        let (value_bytes, scale_bytes) = (params.value_row_bytes(), params.scale_row_bytes());
        let (a_hash_id, b_hash_id) = (params.a().hash_id, params.b().hash_id);
        let routing = params
            .moe()
            .map(|moe| {
                let entries = params
                    .num_padded_routing_entries()
                    .ok_or_else(|| anyhow::anyhow!("MoE params present but padded routing length missing"))?;
                let bytes = entries * std::mem::size_of::<u32>();
                Ok::<_, anyhow::Error>(TreeSchedule::new(cv_locs, ProofSource::Routing, bytes, moe.hash_id_r))
            })
            .transpose()?;
        Ok(Self {
            values: Sides {
                a: schedule(ProofSource::A, m, value_bytes, a_hash_id, "A values")?,
                b: schedule(ProofSource::B, n, value_bytes, b_hash_id, "B values")?,
            },
            scales: Sides {
                a: schedule(ProofSource::AScales, m, scale_bytes, a_hash_id, "A scales")?,
                b: schedule(ProofSource::BScales, n, scale_bytes, b_hash_id, "B scales")?,
            },
            routing,
        })
    }
}

/// Require exactly the scheduled leaves and sibling count, then reconstruct
/// the claimed root under `key` (`VerifyOpen_X`).
fn check_minimal_opening(proof: &MerkleProof, schedule: &TreeSchedule, key: Hash256, label: &str) -> Result<()> {
    let chunk_len = schedule.hash_id.chunk_len();
    if let Some(leaf) = proof.leaf_data.first() {
        ensure_eq!(leaf.len(), chunk_len, "{label}: leaf length must match hash_id");
    }
    // Bound proof-controlled lengths before any further allocation/work.
    ensure!(
        proof.leaf_indices.len() <= schedule.leaf_indices.len(),
        "{label}: extra leaves (got {}, unique-minimal set has {})",
        proof.leaf_indices.len(),
        schedule.leaf_indices.len()
    );
    ensure!(
        proof.leaf_data.len() <= schedule.leaf_indices.len(),
        "{label}: extra leaf data (got {}, unique-minimal set has {})",
        proof.leaf_data.len(),
        schedule.leaf_indices.len()
    );
    ensure_eq!(
        proof.siblings.len(),
        schedule.sibling_ranges.len(),
        "{label}: sibling count must match the compiled schedule"
    );
    proof
        .sanity_check()
        .with_context(|| format!("{label}: merkle proof structure"))?;
    ensure_eq!(
        proof.leaf_indices,
        schedule.leaf_indices,
        "{label}: leaf set is not the unique minimal set for the opened rows"
    );
    ensure_eq!(
        proof.leaf_data.len(),
        schedule.leaf_indices.len(),
        "{label}: leaf_data count must match the unique leaf set"
    );
    ensure_eq!(
        proof.total_leaves,
        schedule.padded_len / chunk_len,
        "{label}: total_leaves mismatch"
    );
    let root = proof
        .compute_root(key)
        .ok_or_else(|| anyhow::anyhow!("{label}: Merkle reconstruction failed (missing/extra/duplicate siblings)"))?;
    ensure_eq!(root, proof.root, "{label}: reconstructed Merkle root mismatch");
    Ok(())
}

/// Recompute the full offsets commitment and compare it with the statement's `HO`.
fn check_full_offsets(params: &PublicParams, offsets: &[u32], key: Hash256) -> Result<()> {
    let (moe, moe_public) = params
        .moe()
        .zip(params.moe_statement())
        .ok_or_else(|| anyhow::anyhow!("PublicParams has no MoE data; cannot check the offsets commitment"))?;
    let root = offsets_root(offsets, moe.hash_id_o, key)?;
    ensure_eq!(
        root,
        moe_public.hash_offsets,
        "offsets: root of the disclosed O does not match the statement HO"
    );
    Ok(())
}

fn extract_prequant_slice(row_indices: &[usize], row_bytes: usize, proof: &MerkleProof) -> Result<PrequantSlice> {
    let rows = row_indices
        .iter()
        .map(|&idx| {
            let start = idx
                .checked_mul(row_bytes)
                .ok_or_else(|| anyhow::anyhow!("committed-row start overflow"))?;
            proof.extract_bytes(start, row_bytes).context("Failed to extract strip")
        })
        .collect::<Result<Vec<_>>>()?;
    PrequantSlice::try_from_rows(rows, row_bytes)
}

fn extract_routing_strips(routing: &MerkleProof, params: &PublicParams) -> Result<Vec<Vec<u8>>> {
    let moe_public = params
        .moe_statement()
        .ok_or_else(|| anyhow::anyhow!("PublicParams has no MoE data; cannot extract routing strips"))?;
    let selected_routing_indices = params.a_inner_indices();
    ensure_eq!(
        selected_routing_indices.len(),
        params.a_rows_indices().len(),
        "inner A indices and outer row count must match"
    );
    moe_public
        .opened_routing_blocks()
        .iter()
        .map(|&block_index| {
            let block_start = block_index as usize * BLOCK_LEN;
            routing
                .extract_bytes(block_start, BLOCK_LEN)
                .with_context(|| format!("routing strip: extract 64 bytes at row_start={block_start}"))
        })
        .collect()
}

fn extract_aux_msgs(
    locations: &[AuxiliaryMsgLocation],
    a: &MatrixMerkleProof,
    a_scales: &MatrixMerkleProof,
    bt: &MatrixMerkleProof,
    bt_scales: &MatrixMerkleProof,
    routing: Option<&MerkleProof>,
) -> Result<Vec<[u8; 64]>> {
    locations
        .iter()
        .map(|loc| {
            let proof = match loc.source {
                ProofSource::A => &a.proof,
                ProofSource::B => &bt.proof,
                ProofSource::AScales => &a_scales.proof,
                ProofSource::BScales => &bt_scales.proof,
                ProofSource::Routing => routing.ok_or_else(|| anyhow::anyhow!("Routing source on a non-MoE proof"))?,
                // Fully opened: the compiler never emits auxiliary offsets messages.
                ProofSource::Offsets => anyhow::bail!("offsets tree cannot source auxiliary messages"),
            };
            proof.extract_bytes(loc.global_start, 64).map(|b| b.try_into().unwrap())
        })
        .collect()
}

/// Extract sibling hashes in compiled trace order. Requires
/// [`check_minimal_opening`] to have validated the proof's schedule.
#[allow(clippy::too_many_arguments)]
fn compute_external_cvs(
    locations: &[AuxiliaryCvLocation],
    schedules: &TreeSchedules,
    a: &MatrixMerkleProof,
    a_scales: &MatrixMerkleProof,
    bt: &MatrixMerkleProof,
    bt_scales: &MatrixMerkleProof,
    routing: Option<&MerkleProof>,
) -> Result<Vec<[u8; BLAKE3_DIGEST_SIZE]>> {
    let ranges = |proof: &MerkleProof, schedule: &TreeSchedule| proof.compute_sibling_ranges(schedule.padded_len);
    let a_ranges = ranges(&a.proof, &schedules.values.a);
    let b_ranges = ranges(&bt.proof, &schedules.values.b);
    let a_scale_ranges = ranges(&a_scales.proof, &schedules.scales.a);
    let b_scale_ranges = ranges(&bt_scales.proof, &schedules.scales.b);
    let routing_ranges = match (routing, &schedules.routing) {
        (Some(proof), Some(schedule)) => Some(ranges(proof, schedule)),
        _ => None,
    };

    locations
        .iter()
        .map(|loc| {
            let ranges = match loc.source {
                ProofSource::A => &a_ranges,
                ProofSource::B => &b_ranges,
                ProofSource::AScales => &a_scale_ranges,
                ProofSource::BScales => &b_scale_ranges,
                ProofSource::Routing => routing_ranges
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Routing CV source has no Merkle proof on this proof"))?,
                ProofSource::Offsets => anyhow::bail!("offsets tree cannot source auxiliary CVs"),
            };
            ranges
                .iter()
                .find(|(s, e, _)| *s == loc.global_start && *e == loc.global_end)
                .map(|&(_, _, h)| h)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "{:?}: compiled sibling {}..{} is not in the opening proof (schedule drift)",
                        loc.source,
                        loc.global_start,
                        loc.global_end
                    )
                })
        })
        .collect()
}

fn reject_nonfinite_prequant(operand: &PrequantOperand, k: usize) -> Result<()> {
    ensure!(
        operand.values.row_bytes() == k,
        "int8 value strip must hold exactly k={k} bytes"
    );
    ensure!(
        operand.scales.row_bytes() == 2 * (k / BLOCK_SIZE),
        "scales strip must hold 2 bytes per {BLOCK_SIZE}-element block of k={k}"
    );
    let ints: Vec<i8> = operand.values.as_bytes().iter().map(|&b| b as i8).collect();
    let scale_bits: Vec<u16> = operand
        .scales
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    scale_bits.iter().try_for_each(|&s| check_not_nan_or_inf_bf16(s))?;
    let codes = open_prequant(&ints, &scale_bits, operand.num_rows()?, k, BLOCK_SIZE)?;
    codes.iter().try_for_each(|&v| check_not_nan_or_inf_bf16(v))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fp8::plain_proof::PlainProofV4;
    use crate::api::fp8::public_params::{
        CommonParams, Device, HashId, JackpotStatement, JobParams, OperandParams, PublicParams, Quant,
    };
    use crate::api::fp8::transcript::{key_a, key_b};
    use crate::api::layout::AxisPattern;
    use crate::api::layout::DimType::{Blake, Fold};
    use crate::api::primitives::IncompleteBlockHeader;
    use pearl_blake3::BLAKE3_CHUNK_LEN;

    fn matrix_proof_keyed(rows_bytes: &[Vec<u8>], row_indices: &[usize], key: [u8; 32], hash_id: HashId) -> MatrixMerkleProof {
        let row_nbytes = rows_bytes[0].len();
        let flat: Vec<u8> = rows_bytes.iter().flatten().copied().collect();
        let chunk_len = hash_id.chunk_len();
        let tree = pearl_blake3::MerkleTree::with_chunk_len(&hash_id.pad(&flat), key, chunk_len).unwrap();
        let leaf_indices =
            pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(row_indices, (rows_bytes.len(), row_nbytes), chunk_len)
                .unwrap();
        MatrixMerkleProof {
            proof: tree.get_multileaf_proof(&leaf_indices),
            row_indices: row_indices.to_vec(),
        }
    }

    struct Fixture {
        proposed: IncompleteBlockHeader,
        proof: PlainProofV4,
        public: PublicParams,
    }

    fn build_dense_proof(scale_at: impl Fn(usize, usize) -> u16) -> (IncompleteBlockHeader, PlainProofV4) {
        build_dense_proof_with(2048, scale_at, HashId::Blake3Chunk1024, HashId::Blake3Chunk1024, |keys| keys)
    }

    fn build_dense_proof_with(
        k: usize,
        scale_at: impl Fn(usize, usize) -> u16,
        a_hash: HashId,
        b_hash: HashId,
        commit_keys: impl FnOnce(Sides<Hash256>) -> Sides<Hash256>,
    ) -> (IncompleteBlockHeader, PlainProofV4) {
        // k ≥ 2048, 16 Blake lanes, 4×64 tile (256 elements).
        // m > h and n > w so some rows stay unopened (auxiliary msgs/CVs).
        let (m, n) = (8usize, 128usize);
        let n_blocks = k / BLOCK_SIZE;
        let rows_pattern = AxisPattern::new(&[(4, Blake)]).unwrap();
        let cols_pattern = AxisPattern::new(&[(4, Blake), (16, Fold)]).unwrap();
        let a_rows: Vec<usize> = rows_pattern.tile_offsets().iter().map(|&o| o as usize).collect();
        let b_rows: Vec<usize> = cols_pattern.tile_offsets().iter().map(|&o| o as usize).collect();
        // The ancestor and proposed headers coincide in this fixture.
        let proposed = IncompleteBlockHeader::new_for_test(0x207FFFFF);
        let keys = commit_keys(Sides {
            a: key_a(&proposed),
            b: key_b(&proposed),
        });

        let value_tree = |rows: usize, seed: usize| -> Vec<Vec<u8>> {
            (0..rows)
                .map(|i| (0..k).map(|j| ((seed + i * k + j) % 251) as u8).collect())
                .collect()
        };
        let scale_tree = |rows: usize, seed: usize| -> Vec<Vec<u8>> {
            (0..rows)
                .map(|i| (0..n_blocks).flat_map(|b| scale_at(seed + i, b).to_le_bytes()).collect())
                .collect()
        };

        let proof = PlainProofV4 {
            job: JobParams {
                ancestor_header: proposed,
                common: CommonParams {
                    k: k as u32,
                    r: 32,
                    quant: Quant::Fp8E4M3Prequant,
                    device: Device::B200,
                },
                operands: Sides {
                    a: OperandParams {
                        num_rows: m as u32,
                        hash_id: a_hash,
                        pattern: rows_pattern,
                    },
                    b: OperandParams {
                        num_rows: n as u32,
                        hash_id: b_hash,
                        pattern: cols_pattern,
                    },
                },
                moe: None,
            },
            values: Sides {
                a: matrix_proof_keyed(&value_tree(m, 0), &a_rows, keys.a, a_hash),
                b: matrix_proof_keyed(&value_tree(n, 7), &b_rows, keys.b, b_hash),
            },
            scales: Sides {
                a: matrix_proof_keyed(&scale_tree(m, 1), &a_rows, keys.a, a_hash),
                b: matrix_proof_keyed(&scale_tree(n, 5), &b_rows, keys.b, b_hash),
            },
            moe_witness: None,
        };
        (proposed, proof)
    }

    fn honest_scale(i: usize, b: usize) -> u16 {
        0x3f80 + ((1 + i + b) % 16) as u16
    }

    fn dense_fixture() -> Fixture {
        let (proposed, proof) = build_dense_proof(honest_scale);
        let (_, public) = proof.parse_proof(&proposed).expect("honest fixture must parse");
        Fixture { proposed, proof, public }
    }

    fn openings_of_public(public: &PublicParams, f: &Fixture) -> Result<PrivateProofParams> {
        public.verify_openings(
            &f.proposed,
            &f.proof.values.a,
            &f.proof.scales.a,
            &f.proof.values.b,
            &f.proof.scales.b,
            None,
        )
    }

    fn openings_of(f: &Fixture) -> Result<PrivateProofParams> {
        openings_of_public(&f.public, f)
    }

    #[test]
    fn swapped_keys_fail() {
        let (proposed, proof) =
            build_dense_proof_with(2048, honest_scale, HashId::Blake3Chunk1024, HashId::Blake3Chunk1024, |keys| {
                Sides { a: keys.b, b: keys.a }
            });
        let err = proof.parse_proof(&proposed).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Merkle") || msg.contains("root"),
            "swapped keys must fail membership, got: {msg}"
        );
    }

    #[test]
    fn wrong_hash_id_fails() {
        let f = dense_fixture();
        let public = PublicParams::try_new(
            JobParams {
                ancestor_header: *f.public.ancestor_header(),
                common: CommonParams {
                    k: f.public.common().k,
                    r: f.public.common().r,
                    quant: Quant::Fp8E4M3Prequant,
                    device: f.public.common().device,
                },
                operands: Sides {
                    a: OperandParams {
                        num_rows: f.public.m(),
                        hash_id: HashId::Blake3Chunk128,
                        pattern: f.public.a().pattern.clone(),
                    },
                    b: OperandParams {
                        num_rows: f.public.n(),
                        hash_id: HashId::Blake3Chunk1024,
                        pattern: f.public.b().pattern.clone(),
                    },
                },
                moe: None,
            },
            JackpotStatement {
                tile_bases: f.public.tile_bases(),
                hash_jackpot: [0u8; 32],
                hash_a: f.public.hash_a(),
                hash_b: f.public.hash_b(),
            },
            None,
        )
        .unwrap();
        let err = openings_of_public(&public, &f).unwrap_err();
        assert!(err.to_string().contains("hash_id"), "got: {err}");
    }

    #[test]
    fn all_hash_ids_parse() {
        for hash_id in HashId::ALL {
            let (proposed, proof) = build_dense_proof_with(2048, honest_scale, hash_id, hash_id, |keys| keys);
            proof
                .parse_proof(&proposed)
                .unwrap_or_else(|e| panic!("{hash_id:?} must parse: {e}"));
        }
    }

    #[test]
    fn mixed_a_b_hash_ids_parse() {
        let (proposed, proof) =
            build_dense_proof_with(2048, honest_scale, HashId::Blake3Chunk128, HashId::Blake3Chunk512, |keys| {
                keys
            });
        proof.parse_proof(&proposed).expect("mixed A/B hash_ids must parse");
    }

    /// Compare compiled leaves and sibling ranges against pearl-blake3's
    /// minimal opening. `k = 2080` exercises blocks crossing row boundaries.
    #[test]
    fn compiled_schedule_is_the_unique_minimal_opening() {
        for (a_hash, b_hash) in [
            (HashId::Blake3Chunk1024, HashId::Blake3Chunk1024),
            (HashId::Blake3Chunk128, HashId::Blake3Chunk512),
            (HashId::Blake3Chunk256, HashId::Blake3Chunk128),
        ] {
            for k in [2048usize, 2080] {
                let (proposed, proof) = build_dense_proof_with(k, honest_scale, a_hash, b_hash, |keys| keys);
                let (_, public) = proof.parse_proof(&proposed).expect("honest fixture must parse");
                let (_, _, cv_locs) = BlakeProgram::compile(&public);
                let a_rows: Vec<usize> = public.a_rows_indices().iter().map(|&x| x as usize).collect();
                let b_rows: Vec<usize> = public.b_rows_indices().iter().map(|&x| x as usize).collect();
                let (m, n) = (public.m() as usize, public.n() as usize);
                let (vb, sb) = (public.value_row_bytes(), public.scale_row_bytes());
                for (source, proof, rows, num_rows, row_bytes, hash_id) in [
                    (ProofSource::A, &proof.values.a, &a_rows, m, vb, a_hash),
                    (ProofSource::AScales, &proof.scales.a, &a_rows, m, sb, a_hash),
                    (ProofSource::B, &proof.values.b, &b_rows, n, vb, b_hash),
                    (ProofSource::BScales, &proof.scales.b, &b_rows, n, sb, b_hash),
                ] {
                    assert_schedule_is_unique_minimal(&cv_locs, source, &proof.proof, rows, num_rows, row_bytes, hash_id);
                }
            }
        }
    }

    fn assert_schedule_is_unique_minimal(
        cv_locs: &[AuxiliaryCvLocation],
        source: ProofSource,
        proof: &MerkleProof,
        row_indices: &[usize],
        num_rows: usize,
        row_bytes: usize,
        hash_id: HashId,
    ) {
        let schedule = TreeSchedule::new(cv_locs, source, num_rows * row_bytes, hash_id);
        let minimal =
            pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(row_indices, (num_rows, row_bytes), hash_id.chunk_len())
                .unwrap();
        assert_eq!(
            schedule.leaf_indices, minimal,
            "{source:?}: leaf schedule != unique-minimal leaf set"
        );
        let mut proof_ranges: Vec<(usize, usize)> = proof
            .compute_sibling_ranges(schedule.padded_len)
            .into_iter()
            .map(|(s, e, _)| (s, e))
            .collect();
        proof_ranges.sort_unstable();
        assert_eq!(
            schedule.sibling_ranges, proof_ranges,
            "{source:?}: compiled aux-CV ranges != proof sibling ranges"
        );
    }

    #[test]
    fn missing_leaf_fails() {
        let mut f = dense_fixture();
        f.proof.values.a.proof.leaf_indices.pop();
        f.proof.values.a.proof.leaf_data.pop();
        assert!(openings_of(&f).is_err());
    }

    #[test]
    fn extra_leaf_fails() {
        let mut f = dense_fixture();
        let extra = f.proof.values.a.proof.leaf_indices.last().copied().unwrap() + 1;
        f.proof.values.a.proof.leaf_indices.push(extra);
        f.proof.values.a.proof.leaf_data.push(vec![0u8; BLAKE3_CHUNK_LEN]);
        assert!(openings_of(&f).is_err());
    }

    #[test]
    fn duplicate_leaf_fails() {
        let mut f = dense_fixture();
        let dup_idx = f.proof.values.a.proof.leaf_indices[0];
        let dup_data = f.proof.values.a.proof.leaf_data[0].clone();
        f.proof.values.a.proof.leaf_indices.insert(1, dup_idx);
        f.proof.values.a.proof.leaf_data.insert(1, dup_data);
        assert!(openings_of(&f).is_err());
    }

    #[test]
    fn extra_sibling_fails() {
        let mut f = dense_fixture();
        f.proof.values.a.proof.siblings.push([0x11u8; 32]);
        assert!(openings_of(&f).is_err());
    }

    #[test]
    fn missing_sibling_fails() {
        let mut f = dense_fixture();
        assert!(!f.proof.values.a.proof.siblings.is_empty(), "fixture should have siblings");
        f.proof.values.a.proof.siblings.pop();
        assert!(openings_of(&f).is_err());
    }

    #[test]
    fn value_scale_row_mismatch_fails() {
        let mut f = dense_fixture();
        f.proof.scales.a.row_indices.pop();
        let err = openings_of(&f).unwrap_err();
        assert!(err.to_string().contains("same rows"), "got: {err}");
    }

    #[test]
    fn nan_scale_fails() {
        let (proposed, proof) = build_dense_proof(|_, _| 0x7FC0); // BF16 NaN
        let err = proof.parse_proof(&proposed).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("nan") || msg.contains("finite") || msg.contains("inf"),
            "got: {err}"
        );
    }

    #[test]
    fn subnormal_scales_pass() {
        let (proposed, proof) = build_dense_proof(|_, _| 0x0001); // smallest +subnormal
        proof.parse_proof(&proposed).expect("finite subnormal scales must parse");
    }

    /// B-side openings must use the key derived from the proof's `job.ancestor_header`.
    #[test]
    fn ancestor_header_keys_b_side() {
        let ancestor = IncompleteBlockHeader {
            prev_block: [7; 32],
            ..IncompleteBlockHeader::new_for_test(0x207FFFFF)
        };
        // Key B's trees under the ancestor, then declare that ancestor in the job.
        // The derived key must match the one used for the tree.
        let (proposed, mut proof) =
            build_dense_proof_with(2048, honest_scale, HashId::Blake3Chunk1024, HashId::Blake3Chunk1024, |keys| {
                Sides {
                    b: key_b(&ancestor),
                    ..keys
                }
            });
        proof.job.ancestor_header = ancestor;
        let (_, public) = proof.parse_proof(&proposed).expect("ancestor-keyed proof must parse");
        assert_eq!(public.ancestor_header(), &ancestor);
    }
}
