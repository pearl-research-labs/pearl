//! Minimal CPU mining for regtest and integration tests.

use anyhow::{Result, ensure};
use pearl_blake3::MerkleTree;
use rand::RngCore;

use crate::v2::ffi::plain_proof::MatrixMerkleProof;
use crate::v4::api::plain_proof::PlainProofV4;
use crate::v4::api::prequant::BLOCK_SIZE;
use crate::v4::api::primitives::{BlockHeader, Hash256, IncompleteBlockHeader, Sides};
use crate::v4::api::proof_utils::nbits_to_difficulty;
use crate::v4::api::public_params::{CommonParams, Device, HashId, JobParams, OperandParams, Quant};
use crate::v4::api::transcript::{key_a, key_b};
use crate::v4::api::verify::verify_plain_proof;

const ROWS: usize = 16;
const K: usize = 1024;

/// Generates independent random input commitments until the V4 plain verifier
/// accepts one at `header.nbits`. Uses a fixed dense 16x16x1024 B200-emulated
/// job and the immediate parent as its ancestor. No GPU is required.
///
/// This deliberately rebuilds and checks a whole candidate per attempt. Use easy
/// targets for tests: there is no attempt limit or cancellation, and realistic
/// mining difficulty is impractical. Returns an error for a disconnected parent
/// or a zero target. ZK proving is left to the caller.
pub fn mine(header: &IncompleteBlockHeader, parent: &BlockHeader) -> Result<PlainProofV4> {
    mine_with_rng(header, parent, &mut rand::rng())
}

fn mine_with_rng(header: &IncompleteBlockHeader, parent: &BlockHeader, rng: &mut impl RngCore) -> Result<PlainProofV4> {
    ensure!(header.prev_block == parent.block_hash(), "V4 mining parent does not connect");
    ensure!(!nbits_to_difficulty(header.nbits).is_zero(), "V4 mining target is zero");
    loop {
        let proof = candidate(header, parent, rng)?;
        // Propagate malformed witnesses rather than retrying an invalid job forever.
        proof.parse_proof(header)?;
        if verify_plain_proof(header, &proof, None).is_ok() {
            return Ok(proof);
        }
    }
}

fn candidate(header: &IncompleteBlockHeader, parent: &BlockHeader, rng: &mut impl RngCore) -> Result<PlainProofV4> {
    let hash_id = HashId::Blake3Chunk1024;
    let operand = OperandParams {
        num_rows: ROWS as u32,
        hash_id,
        tile_size: ROWS as u32,
    };
    let keys = Sides {
        a: key_a(header),
        b: key_b(parent),
    };
    let plane = |bytes: &[u8], key: Hash256| -> Result<MatrixMerkleProof> {
        let tree = MerkleTree::with_chunk_len(&hash_id.pad(bytes), key, hash_id.chunk_len())?;
        // All rows are opened, so every leaf belongs to the proof.
        let leaves: Vec<usize> = (0..hash_id.padded_len(bytes.len()) / hash_id.chunk_len()).collect();
        Ok(MatrixMerkleProof {
            proof: tree.get_multileaf_proof(&leaves),
            row_indices: (0..ROWS).collect(),
        })
    };
    let mut values = vec![0; ROWS * K];
    rng.fill_bytes(&mut values);
    let a = plane(&values, keys.a)?;
    rng.fill_bytes(&mut values);
    let b = plane(&values, keys.b)?;
    // Unit BF16 scales, one per prequant block.
    let scales: Vec<u8> = (0..ROWS * K / BLOCK_SIZE).flat_map(|_| 0x3f80u16.to_le_bytes()).collect();
    Ok(PlainProofV4 {
        job: JobParams {
            ancestor_header: *parent,
            common: CommonParams {
                k: K as u32,
                r: 32,
                quant: Quant::Fp8E4M3Prequant,
                device: Device::B200,
            },
            operands: Sides {
                a: operand.clone(),
                b: operand,
            },
            moe: None,
        },
        ancestor_chain: vec![],
        values: Sides { a, b },
        scales: Sides {
            a: plane(&scales, keys.a)?,
            b: plane(&scales, keys.b)?,
        },
        moe_witness: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn mines_a_nontrivial_target_and_roundtrips() {
        let parent = BlockHeader::default();
        let header = IncompleteBlockHeader {
            prev_block: parent.block_hash(),
            nbits: 0x1e0fffff,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // This target accepts about one quarter of the 16x16x1024 tickets.
        let mut losing = false;
        for _ in 0..32 {
            let proof = candidate(&header, &parent, &mut rng).unwrap();
            proof.parse_proof(&header).unwrap();
            losing |= verify_plain_proof(&header, &proof, None).is_err();
        }
        assert!(losing, "the test must exercise rejected candidates");
        let proof = mine_with_rng(&header, &parent, &mut rng).unwrap();
        let restored = PlainProofV4::from_bytes(&proof.to_bytes().unwrap()).unwrap();
        verify_plain_proof(&header, &restored, None).unwrap();
        assert!(verify_plain_proof(&header, &restored, Some(0)).is_err());
    }

    #[test]
    fn rejects_disconnected_parent_and_zero_target() {
        let parent = BlockHeader::default();
        let mut header = IncompleteBlockHeader {
            nbits: 0x207fffff,
            ..Default::default()
        };
        assert!(mine(&header, &parent).unwrap_err().to_string().contains("does not connect"));
        header.prev_block = parent.block_hash();
        header.nbits = 0;
        assert!(mine(&header, &parent).unwrap_err().to_string().contains("target is zero"));
    }
}
