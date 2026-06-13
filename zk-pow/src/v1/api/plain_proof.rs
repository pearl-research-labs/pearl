use anyhow::{Context, Result, bail};
use pearl_blake3::MerkleProof;

use crate::v1::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PrivateProofParams, PublicProofParams,
};
use crate::v1::circuit::chip::blake3::program::{AuxiliaryCvLocation, AuxiliaryMsgLocation};
use crate::v1::ensure_eq;
use pearl_blake3::BLAKE3_DIGEST_SIZE;

use crate::ffi::plain_proof::PlainProof;

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

fn extract_external_messages(locs: &[AuxiliaryMsgLocation], p: &PlainProof) -> Result<Vec<[u8; 64]>> {
    locs.iter()
        .map(|loc| {
            let proof = if loc.is_b { &p.bt.proof } else { &p.a.proof };
            proof.extract_bytes(loc.global_start, 64).map(|b| b.try_into().unwrap())
        })
        .collect()
}

fn compute_external_cvs(
    locs: &[AuxiliaryCvLocation],
    p: &PlainProof,
    m: usize,
    n: usize,
    k: usize,
    key: [u8; BLAKE3_DIGEST_SIZE],
) -> Result<Vec<[u8; BLAKE3_DIGEST_SIZE]>> {
    let a_ranges = p.a.proof.compute_sibling_ranges(pearl_blake3::padded_chunk_len(m * k));
    let b_ranges = p.bt.proof.compute_sibling_ranges(pearl_blake3::padded_chunk_len(n * k));

    locs.iter()
        .map(|loc| {
            let (ranges, proof) = if loc.is_b {
                (&b_ranges, &p.bt.proof)
            } else {
                (&a_ranges, &p.a.proof)
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

pub fn parse_plain_proof(header: IncompleteBlockHeader, p: &PlainProof) -> Result<(PrivateProofParams, PublicProofParams)> {
    let (m, n, k) = (p.m, p.n, p.k);

    let a_indices: Vec<u32> = p.a.row_indices.iter().map(|&x| x as u32).collect();
    let bt_indices: Vec<u32> = p.bt.row_indices.iter().map(|&x| x as u32).collect();
    let (rows_pattern, t_rows) = list_to_pattern(&a_indices)?;
    let (cols_pattern, t_cols) = list_to_pattern(&bt_indices)?;

    let public = PublicProofParams {
        block_header: header,
        mining_config: MiningConfiguration {
            common_dim: k as u32,
            rank: p.noise_rank as u16,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern,
            cols_pattern,
            reserved: MiningConfiguration::RESERVED_VALUE,
        },
        hash_a: p.a.proof.root,
        hash_b: p.bt.proof.root,
        hash_jackpot: [0xFFu8; 32],
        m: m as u32,
        n: n as u32,
        t_rows,
        t_cols,
    };

    let (compiled, msg_locs, cv_locs) = public.compile();

    let strip_len = public.dot_product_length();

    let private = PrivateProofParams {
        s_a: extract_strips(&p.a.row_indices, k, strip_len, &p.a.proof)?,
        s_b: extract_strips(&p.bt.row_indices, k, strip_len, &p.bt.proof)?,
        external_msgs: extract_external_messages(&msg_locs, p)?,
        external_cvs: compute_external_cvs(&cv_locs, p, m, n, k, public.job_key())?,
    };

    let (hash_a, hash_b) = compiled.blake_proof.evaluate_blake(compiled.job_key, &private)?;
    ensure_eq!(hash_a, p.a.proof.root, "Hash A mismatch, job_key={:?}", compiled.job_key);
    ensure_eq!(hash_b, p.bt.proof.root, "Hash B mismatch, job_key={:?}", compiled.job_key);

    Ok((private, public))
}
