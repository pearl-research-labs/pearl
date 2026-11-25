//! Minimal CPU mining for regtest and integration tests.

use anyhow::{Result, ensure};
use pearl_blake3::MerkleTree;
use rand::RngCore;

use crate::v2::ffi::plain_proof::MatrixMerkleProof;
use crate::v4::api::{
    plain_proof::PlainProofV4,
    prequant::BLOCK_SIZE,
    primitives::{BlockHeader, Hash256, IncompleteBlockHeader, Sides},
    proof_utils::nbits_to_difficulty,
    public_params::{CommonParams, Device, HashId, JobParams, OperandParams, Quant},
    transcript::{key_a, key_b},
    verify::verify_plain_proof,
};

/// Retries random dense 16x16x1024 jobs for `device` until one wins at
/// `header.nbits`, using the immediate parent as ancestor. CPU-only; use easy
/// targets, since the search has no attempt limit or cancellation.
pub fn mine(header: &IncompleteBlockHeader, parent: &BlockHeader, device: Device) -> Result<PlainProofV4> {
    const ROWS: usize = 16;
    const K: usize = 1024;
    ensure!(header.prev_block == parent.block_hash(), "V4 mining parent does not connect");
    ensure!(!nbits_to_difficulty(header.nbits).is_zero(), "V4 mining target is zero");

    let hash_id = HashId::Blake3Chunk1024;
    let operand = OperandParams {
        num_rows: ROWS as u32,
        hash_id,
        tile_size: ROWS as u32,
    };
    let job = JobParams {
        ancestor_header: *parent,
        common: CommonParams {
            k: K as u32,
            r: 32,
            quant: Quant::Fp8E4M3Prequant,
            device,
        },
        operands: Sides {
            a: operand.clone(),
            b: operand,
        },
        moe: None,
    };
    let plane = |bytes: &[u8], key: Hash256| -> Result<MatrixMerkleProof> {
        let tree = MerkleTree::with_chunk_len(&hash_id.pad(bytes), key, hash_id.chunk_len())?;
        let leaves: Vec<usize> = (0..hash_id.padded_len(bytes.len()) / hash_id.chunk_len()).collect();
        Ok(MatrixMerkleProof {
            proof: tree.get_multileaf_proof(&leaves),
            row_indices: (0..ROWS).collect(),
        })
    };
    let keys = Sides {
        a: key_a(header),
        b: key_b(parent),
    };
    // Unit BF16 scales, one per prequant block.
    let scales: Vec<u8> = (0..ROWS * K / BLOCK_SIZE).flat_map(|_| 0x3f80u16.to_le_bytes()).collect();
    let scales = Sides {
        a: plane(&scales, keys.a)?,
        b: plane(&scales, keys.b)?,
    };
    let mut rng = rand::rng();
    loop {
        let mut values = |key| {
            let mut bytes = vec![0; ROWS * K];
            // Match the fixtures' centered triangular distribution for B200 noise coverage.
            for byte in &mut bytes {
                let x = rng.next_u32();
                *byte = ((x & 127) as i16 + ((x >> 16) & 127) as i16 - 127) as i8 as u8;
            }
            plane(&bytes, key)
        };
        let proof = PlainProofV4 {
            job: job.clone(),
            ancestor_chain: vec![],
            values: Sides {
                a: values(keys.a)?,
                b: values(keys.b)?,
            },
            scales: scales.clone(),
            moe_witness: None,
        };
        if verify_plain_proof(header, &proof, None).is_ok() {
            return Ok(proof);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mines_and_plain_verifies_at_nontrivial_difficulty() {
        let parent = BlockHeader::default();
        let header = IncompleteBlockHeader {
            prev_block: parent.block_hash(),
            nbits: 0x1e0fffff, // About one quarter of 16x16x1024 tickets win.
            ..Default::default()
        };
        for device in Device::ALL {
            let proof = mine(&header, &parent, device).unwrap();
            assert_eq!(proof.job.common.device, device);
            verify_plain_proof(&header, &proof, None).unwrap();
        }
    }
}
