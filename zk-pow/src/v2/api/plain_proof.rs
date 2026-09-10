//! Parses the shared FFI [`PlainProof`] wire type into v2 proof parameters.
//!
//! Self-contained v2/v3 twin of `v1::api::plain_proof`: the Int7 dense + MoE
//! parse path, verifying Merkle membership against the committed roots.
//! FP8 proofs are v4-only and are rejected here.

use anyhow::{Context, Result, bail, ensure};
use blake3::BLOCK_LEN;
use pearl_blake3::MerkleProof;

use crate::ensure_eq;
use crate::ffi::plain_proof::{MoEProofParams, PlainProof};
use crate::v2::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, MoEConfig, MoEParams, PeriodicPattern, PrivateProofParams,
    PublicProofParams,
};
use crate::v2::circuit::chip::blake3::program::{
    AuxiliaryCvLocation, AuxiliaryMsgLocation, ProofSource, routing_blake_hotspot_rows,
};
use pearl_blake3::{BLAKE3_CHUNK_LEN, BLAKE3_DIGEST_SIZE};

fn extract_routing_strips(p: &PlainProof, params: &PublicProofParams) -> Result<Vec<Vec<u8>>> {
    let moe_proof = p
        .moe
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("PlainProof has no MoE data; cannot extract routing strips"))?;
    let moe_public = params
        .moe
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("PublicProofParams has no MoE data; cannot extract routing strips"))?;

    let inner = params.a_inner_indices();
    ensure_eq!(
        inner.len(),
        p.a.row_indices.len(),
        "inner A indices and outer row count must match"
    );

    routing_blake_hotspot_rows(moe_public.expert_start_offset(), &inner)
        .iter()
        .map(|&hotspot_idx| {
            let block_start = hotspot_idx as usize * BLOCK_LEN;
            moe_proof
                .routing_proof
                .extract_bytes(block_start, BLOCK_LEN)
                .with_context(|| format!("routing strip: extract 64 bytes at row_start={}", block_start))
        })
        .collect()
}

fn extract_strips(indices: &[usize], k: usize, strip_len: usize, proof: &MerkleProof) -> Result<Vec<Vec<i8>>> {
    indices
        .iter()
        .map(|&idx| {
            proof
                .extract_bytes(idx * k, strip_len)
                .map(|b| b.into_iter().map(|x| x as i8).collect())
                .context("Failed to extract strip")
        })
        .collect()
}

fn compute_external_cvs(
    locs: &[AuxiliaryCvLocation],
    p: &PlainProof,
    k: usize,
    key: [u8; BLAKE3_DIGEST_SIZE],
) -> Result<Vec<[u8; BLAKE3_DIGEST_SIZE]>> {
    let total_b_cols = total_b_cols(p);
    let a_ranges = p.a.proof.compute_sibling_ranges(pearl_blake3::padded_chunk_len(p.m * k));
    let b_ranges =
        p.bt.proof
            .compute_sibling_ranges(pearl_blake3::padded_chunk_len(total_b_cols * k));

    let routing = p.moe.as_ref().map(|moe| {
        let total_routing_entries = p.m * moe.top_k;
        let raw_len = total_routing_entries * std::mem::size_of::<u32>();
        let ranges = moe
            .routing_proof
            .compute_sibling_ranges(pearl_blake3::padded_chunk_len(raw_len));
        (&moe.routing_proof, ranges)
    });

    locs.iter()
        .map(|loc| {
            let (proof, ranges) = match loc.source {
                ProofSource::A => (&p.a.proof, &a_ranges),
                ProofSource::B => (&p.bt.proof, &b_ranges),
                ProofSource::Routing => {
                    let (proof, ranges) = routing
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("ProofSource::Routing requested on a non-MoE PlainProof"))?;
                    (*proof, ranges)
                }
            };
            if let Some(&(_, _, h)) = ranges.iter().find(|(s, e, _)| *s == loc.global_start && *e == loc.global_end) {
                return Ok(h);
            }
            log::warn!("CV not in proof, computing recursively");
            proof
                .compute_cv(loc.global_start, loc.global_end, ranges, key)
                .context("Failed to compute CV")
        })
        .collect()
}

/// Converts indices to a PeriodicPattern and base offset.
pub fn list_to_pattern(indices: &[u32]) -> Result<(PeriodicPattern, u32)> {
    if indices.is_empty() {
        bail!("pattern parsing error: Empty indices");
    }
    if !indices.windows(2).all(|w| w[0] < w[1]) {
        bail!("pattern parsing error: Indices not strictly increasing");
    }

    let offset = indices[0];
    let normalized: Vec<u32> = indices.iter().map(|&i| i - offset).collect();
    let pattern = PeriodicPattern::from_list(&normalized).context("pattern parsing error")?;

    if !pattern.offset_is_valid(offset) {
        bail!("pattern parsing error: offset {} is not valid for pattern", offset);
    }

    Ok((pattern, offset))
}

/// Returns the merkle proof for a given proof source. Errors if `Routing`
/// is requested on a non-MoE proof.
fn proof_for(p: &PlainProof, source: ProofSource) -> Result<&MerkleProof> {
    match source {
        ProofSource::A => Ok(&p.a.proof),
        ProofSource::B => Ok(&p.bt.proof),
        ProofSource::Routing => p
            .moe
            .as_ref()
            .map(|moe| &moe.routing_proof)
            .ok_or_else(|| anyhow::anyhow!("ProofSource::Routing requested on a non-MoE PlainProof")),
    }
}

/// Extracts external messages from the proof based on the provided locations.
fn extract_external_messages(p: &PlainProof, locs: &[AuxiliaryMsgLocation]) -> Result<Vec<[u8; 64]>> {
    locs.iter()
        .map(|loc| {
            let proof = proof_for(p, loc.source)?;
            proof.extract_bytes(loc.global_start, 64).map(|b| b.try_into().unwrap())
        })
        .collect()
}

fn total_b_cols(p: &PlainProof) -> usize {
    if let Some(moe) = &p.moe { p.n * moe.e } else { p.n }
}

/// Leaf count the Merkle tree of a row-major byte buffer whose length is the
/// product of `dims` must declare. Errors if the product overflows `usize`.
fn expected_merkle_leaves(dims: &[usize]) -> Result<usize> {
    let bytes = dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| anyhow::anyhow!("declared dimensions {dims:?} overflow usize"))?;
    Ok(pearl_blake3::padded_chunk_len(bytes) / BLAKE3_CHUNK_LEN)
}

/// Rejects a proof whose committed Merkle trees don't have the leaf count
/// implied by the declared dimensions.
///
/// The V3 noise seed is salted with the declared `m`/`n`, making them
/// consensus-critical: without pinning each tree's `total_leaves` to those
/// dimensions a miner could open one committed tree under several dimension
/// interpretations. The A/B trees are row-major `m*k` / `total_b_cols*k`
/// int8 bytes; the MoE routing tree is `m*top_k` little-endian `u32`s.
fn check_declared_tree_sizes(p: &PlainProof) -> Result<()> {
    let a_expected = expected_merkle_leaves(&[p.m, p.k])?;
    ensure_eq!(
        p.a.proof.total_leaves,
        a_expected,
        "A Merkle tree declares {} leaves but m={} k={} imply {}",
        p.a.proof.total_leaves,
        p.m,
        p.k,
        a_expected
    );

    let b_expected = expected_merkle_leaves(&[total_b_cols(p), p.k])?;
    ensure_eq!(
        p.bt.proof.total_leaves,
        b_expected,
        "B^T Merkle tree declares {} leaves but n={} k={} (total columns {}) imply {}",
        p.bt.proof.total_leaves,
        p.n,
        p.k,
        total_b_cols(p),
        b_expected
    );

    if let Some(moe) = &p.moe {
        let routing_expected = expected_merkle_leaves(&[p.m, moe.top_k, std::mem::size_of::<u32>()])?;
        ensure_eq!(
            moe.routing_proof.total_leaves,
            routing_expected,
            "routing Merkle tree declares {} leaves but m={} top_k={} imply {}",
            moe.routing_proof.total_leaves,
            p.m,
            moe.top_k,
            routing_expected
        );
    }
    Ok(())
}

/// Derives the inner A/B index lists used to build the periodic patterns,
/// plus the public `MoEParams` (when this is an MoE proof).
fn moe_inner_indices(p: &PlainProof) -> Result<(Vec<u32>, Vec<u32>, Option<MoEParams>)> {
    let a_indices: Vec<u32> =
        p.a.row_indices
            .iter()
            .map(|&x| x.try_into().context("A row index exceeds u32"))
            .collect::<Result<_>>()?;
    let bt_indices: Vec<u32> =
        p.bt.row_indices
            .iter()
            .map(|&x| x.try_into().context("B row index exceeds u32"))
            .collect::<Result<_>>()?;

    let Some(moe) = &p.moe else {
        return Ok((a_indices, bt_indices, None));
    };

    ensure!((moe.expert_idx as usize) < moe.e);
    ensure!(moe.e == moe.routing_end_offsets.len());

    let weight_col_offset = (moe.expert_idx as usize) * p.n;
    // p.bt.row_indices are already global indices (offset by expert_idx * n_e in mining).
    for &idx in &p.bt.row_indices {
        ensure!(
            idx >= weight_col_offset && idx < weight_col_offset + p.n,
            "B column index {} out of range for expert {} (expected [{}, {}))",
            idx,
            moe.expert_idx,
            weight_col_offset,
            weight_col_offset + p.n
        );
    }
    // In the MoE case, n represents the per-expert intermediate dimension n_e.
    ensure!(
        p.bt.row_indices.len() < p.n,
        "B^T row indices length {} is not less than n_e {}",
        p.bt.row_indices.len(),
        p.n
    );

    ensure!(moe.routing_end_offsets.len() == moe.e);

    let inner_a: Vec<u32> = moe
        .inner_a_rows
        .iter()
        .map(|&x| x.try_into().context("MoE inner A row index exceeds u32"))
        .collect::<Result<_>>()?;
    let weight_col_offset_u32: u32 = weight_col_offset
        .try_into()
        .context("expert weight column offset exceeds u32")?;
    let inner_b: Vec<u32> = bt_indices
        .iter()
        .map(|&idx| {
            idx.checked_sub(weight_col_offset_u32)
                .context("B row index below expert offset")
        })
        .collect::<Result<_>>()?;

    ensure!(
        moe.e <= PublicProofParams::MAX_NUM_EXPERTS,
        "number of experts {} exceeds maximum {}",
        moe.e,
        PublicProofParams::MAX_NUM_EXPERTS
    );
    let moe_params = MoEParams {
        routing_offsets: moe.routing_end_offsets.clone(),
        expert_idx: moe.expert_idx,
        hash_routing: moe.routing_proof.root,
        outer_indices: a_indices,
    };
    Ok((inner_a, inner_b, Some(moe_params)))
}

/// Converts a plain proof to v2/v3 proof types, checking the a/bt Merkle roots
/// match the committed hashes. Rejects v4 (FP8) proofs.
pub fn parse_plain_proof(
    header: IncompleteBlockHeader,
    p: &PlainProof,
    seed_derivation: crate::api::seed::SeedDerivation,
) -> Result<(PrivateProofParams, PublicProofParams)> {
    // Leaf-count binds usize; public dims are u32/u16 — wrap would unbind them.
    let m: u32 = p.m.try_into().context("m exceeds u32")?;
    let n: u32 = p.n.try_into().context("n exceeds u32")?;
    let k: u32 = p.k.try_into().context("k exceeds u32")?;
    let noise_rank: u16 = p.noise_rank.try_into().context("noise_rank exceeds u16")?;

    let moe_config = match &p.moe {
        Some(mp) => Some(MoEConfig {
            e: mp.e.try_into().context("e exceeds u16")?,
            top_k: mp.top_k.try_into().context("top_k exceeds u16")?,
        }),
        None => None,
    };

    // Pin the committed trees to the declared dimensions before those
    // dimensions flow into the (salted) noise-seed derivation.
    check_declared_tree_sizes(p)?;

    for &tok in &p.a.row_indices {
        ensure!(tok < m as usize, "routing entry {} out of range for t={}", tok, m);
    }

    let (inner_a_indices, inner_b_indices, moe_params) = moe_inner_indices(p)?;
    let (rows_pattern, t_rows) = list_to_pattern(&inner_a_indices)?;
    let (cols_pattern, t_cols) = list_to_pattern(&inner_b_indices)?;

    let public = PublicProofParams {
        block_header: header,
        seed_derivation,
        mining_config: MiningConfiguration {
            common_dim: k,
            rank: noise_rank,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern,
            cols_pattern,
            moe: moe_config,
        },
        hash_a: p.a.proof.root,
        hash_b: p.bt.proof.root,
        hash_jackpot: [0xFFu8; 32], // Consumed only by ZK verifier
        m,
        n,
        t_rows,
        t_cols,
        moe: moe_params,
    };

    let (compiled, msg_locs, cv_locs) = public.compile();
    let strip_len = public.dot_product_length();

    let s_routing = if p.moe.is_some() {
        let strips = extract_routing_strips(p, &public)?;
        ensure_eq!(
            strips.len(),
            compiled.blake_proof.num_routing_strips,
            "MoE s_routing strips must match num_routing_strips"
        );
        strips
    } else {
        vec![]
    };

    let k = k as usize;
    let private = PrivateProofParams {
        s_a: extract_strips(&p.a.row_indices, k, strip_len, &p.a.proof)?,
        s_b: extract_strips(&p.bt.row_indices, k, strip_len, &p.bt.proof)?,
        s_routing,
        external_msgs: extract_external_messages(p, &msg_locs)?,
        external_cvs: compute_external_cvs(&cv_locs, p, k, public.job_key())?,
    };

    let opt_hash_routing = p.moe.as_ref().map(|moe| moe.routing_proof.root);
    let (hash_a, hash_b) = compiled
        .blake_proof
        .evaluate_blake(compiled.job_key, &private, opt_hash_routing)?;
    ensure_eq!(hash_a, p.a.proof.root, "Hash A mismatch, job_key={:?}", compiled.job_key);
    ensure_eq!(hash_b, p.bt.proof.root, "Hash B mismatch, job_key={:?}", compiled.job_key);

    if let Some(moe) = &p.moe {
        verify_moe_routing(moe, &p.a.row_indices, compiled.job_key)?;
    }

    Ok((private, public))
}

/// Verifies the routing Merkle membership proof (recomputed root matches the committed root)
/// and that, for each sampled outer index, `routing[expert_idx][inner_idx]` equals it.
fn verify_moe_routing(moe: &MoEProofParams, outer_indices: &[usize], key: [u8; BLAKE3_DIGEST_SIZE]) -> Result<()> {
    let computed_root = moe
        .routing_proof
        .compute_root(key)
        .ok_or_else(|| anyhow::anyhow!("routing Merkle proof has no leaves"))?;
    ensure_eq!(
        computed_root,
        moe.routing_proof.root,
        "routing Merkle membership proof failed: computed root does not match hash_routing"
    );

    // The routing is serialized as a flat array of little-endian u32 values.
    // `routing_end_offsets` stores exclusive ends (cumulative counts), so the
    // start of `expert_idx` is the previous expert's end (0 for expert 0).
    let routing_start_offset = match moe.expert_idx {
        0 => 0u32,
        idx => moe.routing_end_offsets[idx as usize - 1],
    };
    for (i, &inner_idx) in moe.inner_a_rows.iter().enumerate() {
        let byte_offset = (routing_start_offset as usize + inner_idx) * std::mem::size_of::<u32>();
        let bytes = moe
            .routing_proof
            .extract_bytes(byte_offset, std::mem::size_of::<u32>())
            .with_context(|| {
                format!(
                    "failed to extract routing entry at inner_idx={} (byte_offset={})",
                    inner_idx, byte_offset
                )
            })?;
        let routing_value = u32::from_le_bytes(bytes.try_into().unwrap());
        ensure_eq!(
            routing_value as usize,
            outer_indices[i],
            "routing mismatch: routing[{}][{}] = {} but outer_indices[{}] = {}",
            moe.expert_idx,
            inner_idx,
            routing_value,
            i,
            outer_indices[i]
        );
    }
    Ok(())
}
