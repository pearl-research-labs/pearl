use anyhow::{Result, ensure};
use primitive_types::U256;

use crate::api::fp8::public_params::PublicParams;
use crate::api::primitives::{Hash256, IncompleteBlockHeader};
use crate::circuit::chip::blake3::program::{AuxiliaryCvLocation, AuxiliaryMsgLocation, BlakeProgram};

use pearl_blake3::blake3_digest;

fn ensure_roundtrip(original: &[u8], reserialized: &[u8], label: &str) -> Result<()> {
    ensure!(
        reserialized == original,
        "{label} round-trip mismatch: deserialized form does not re-serialize to the original bytes"
    );
    Ok(())
}

/// Convert Bitcoin's compact nbits format to an absolute difficulty target as U256
///
/// Bitcoin's nbits is a compact representation where:
/// - First byte is the exponent (number of bytes in the full target)
/// - Last 3 bytes are the mantissa (the significant digits)
///
/// The formula is: target = mantissa * 256^(exponent - 3)
pub fn nbits_to_difficulty(nbits: u32) -> U256 {
    // Extract exponent (first byte) and mantissa (last 3 bytes)
    let exponent = (nbits >> 24) as usize;
    let mantissa = nbits & 0x00ffffff;

    // Handle edge case where mantissa is 0
    if mantissa == 0 || exponent == 0 {
        return U256::zero();
    }

    // Check for negative bit (0x00800000) - Bitcoin treats this as invalid/negative
    if mantissa & 0x00800000 != 0 {
        return U256::zero(); // Invalid/negative target
    }

    // Convert mantissa to U256
    let mut target = U256::from(mantissa);

    // Apply the exponent
    if exponent <= 3 {
        // Shift right
        target >>= 8 * (3 - exponent);
    } else {
        // Shift left
        target <<= 8 * (exponent - 3);
    }

    target
}

impl IncompleteBlockHeader {
    /// Size of serialized IncompleteBlockHeader in bytes.
    /// 4 (version) + 32 (prev_block) + 32 (merkle_root) + 4 (timestamp) + 4 (nbits) = 76
    pub const SERIALIZED_SIZE: usize = 76;

    /// The all-zero (except for a valid difficulty `nbits`) ancestor header used to
    /// anchor canonical, job-independent derivations such as the LUT setup caps.
    /// The header never reaches family-only preprocessed data, so both this and
    /// [`PublicParams::dense_params`] anchors share it.
    pub(crate) fn zero() -> IncompleteBlockHeader {
        Self {
            version: 0,
            prev_block: [0; 32],
            merkle_root: [0; 32],
            timestamp: 0,
            nbits: 0,
        }
    }

    #[cfg(test)]
    pub fn new_for_test(nbits: u32) -> IncompleteBlockHeader {
        Self {
            version: 0,
            prev_block: [1; 32],
            merkle_root: [2; 32],
            timestamp: 0x66666666,
            nbits,
        }
    }

    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut bytes = Vec::with_capacity(Self::SERIALIZED_SIZE);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend(self.prev_block.iter().rev().copied());
        bytes.extend(self.merkle_root.iter().rev().copied());
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.extend_from_slice(&self.nbits.to_le_bytes());
        bytes.try_into().unwrap()
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        ensure!(
            data.len() == Self::SERIALIZED_SIZE,
            "Expected {} bytes, got {}",
            Self::SERIALIZED_SIZE,
            data.len()
        );
        let version = u32::from_le_bytes(data[0..4].try_into().unwrap());
        // prev_block and merkle_root are stored reversed in serialized form
        let mut prev_block: [u8; 32] = data[4..36].try_into().unwrap();
        prev_block.reverse();
        let mut merkle_root: [u8; 32] = data[36..68].try_into().unwrap();
        merkle_root.reverse();
        let timestamp = u32::from_le_bytes(data[68..72].try_into().unwrap());
        let nbits = u32::from_le_bytes(data[72..76].try_into().unwrap());

        let result = Self {
            version,
            prev_block,
            merkle_root,
            timestamp,
            nbits,
        };
        ensure_roundtrip(data, &result.to_bytes(), "IncompleteBlockHeader")?;
        Ok(result)
    }
}

/// The plain v4 winning condition: `hash ≤ target · h · w · k`,
/// clamped to `U256::MAX` on overflow. `Ok(())` accepts, `Err` rejects.
///
/// Both the ZK epilogue and the plain verifier check the same condition (on the
/// proven `hash_jackpot` claim), against the block's `nbits` (or a pool-share override).
pub fn check_jackpot_difficulty(hash: &Hash256, nbits: u32, h: u32, w: u32, k: u32) -> Result<()> {
    let target = nbits_to_difficulty(nbits);
    // The work-bearing tile is the full lottery tile, scaled by the prequant strip
    // width k (rank divides k, so k - k mod rank is k).
    let adjustment = h.checked_mul(w).and_then(|x| x.checked_mul(k)).unwrap_or(u32::MAX);
    let bound = if target > U256::MAX / adjustment {
        U256::MAX
    } else {
        target * adjustment
    };
    ensure!(
        U256::from_little_endian(hash) <= bound,
        "Jackpot condition not satisfied: hash does not meet difficulty target"
    );
    Ok(())
}

#[cfg(test)]
mod difficulty_tests {
    use super::*;

    #[test]
    fn test_difficulty_conversion() {
        // Test case 1: Genesis block difficulty (0x1d00ffff)
        // This represents: 0x00000000ffff0000000000000000000000000000000000000000000000000000
        let header = IncompleteBlockHeader {
            version: 1,
            prev_block: [0; 32],
            merkle_root: [0; 32],
            timestamp: 0,
            nbits: 0x1d00ffff,
        };

        let target = nbits_to_difficulty(header.nbits);
        let expected = U256::from_str_radix("00000000ffff0000000000000000000000000000000000000000000000000000", 16).unwrap();
        assert_eq!(target, expected);

        // Test case 2: A more typical difficulty (0x1b0404cb)
        // This represents: 0x00000000000404cb000000000000000000000000000000000000000000000000
        let header2 = IncompleteBlockHeader {
            version: 1,
            prev_block: [0; 32],
            merkle_root: [0; 32],
            timestamp: 0,
            nbits: 0x1b0404cb,
        };

        let target2 = nbits_to_difficulty(header2.nbits);
        let expected2 = U256::from_str_radix("00000000000404cb000000000000000000000000000000000000000000000000", 16).unwrap();
        assert_eq!(target2, expected2);

        // Test case 3: Edge case with small exponent
        let header3 = IncompleteBlockHeader {
            version: 1,
            prev_block: [0; 32],
            merkle_root: [0; 32],
            timestamp: 0,
            nbits: 0x03123456, // exponent = 3, mantissa = 0x123456
        };

        let target3 = nbits_to_difficulty(header3.nbits);
        // mantissa stays as is when exponent = 3
        assert_eq!(target3, U256::from(0x123456));

        // Test case 4: Zero mantissa should return zero
        let header4 = IncompleteBlockHeader {
            version: 1,
            prev_block: [0; 32],
            merkle_root: [0; 32],
            timestamp: 0,
            nbits: 0x1d000000,
        };

        assert_eq!(nbits_to_difficulty(header4.nbits), U256::zero());

        // Test case 5: Maximum difficulty (0x2077ffff)
        // This is close to the maximum valid nbits value
        let header5 = IncompleteBlockHeader {
            version: 1,
            prev_block: [0; 32],
            merkle_root: [0; 32],
            timestamp: 0,
            nbits: 0x2077ffff,
        };

        let target5 = nbits_to_difficulty(header5.nbits);
        // With exponent 0x20 (32) and mantissa 0x77ffff
        // Result should be 0x77ffff shifted left by (32-3)*8 = 232 bits
        let expected5 = U256::from(0x77ffff) << (29 * 8);
        assert_eq!(target5, expected5);

        // Test case 6: Negative bit set (should return zero)
        let header6 = IncompleteBlockHeader {
            version: 1,
            prev_block: [0; 32],
            merkle_root: [0; 32],
            timestamp: 0,
            nbits: 0x1d800000, // Negative bit (0x800000) is set
        };

        assert_eq!(nbits_to_difficulty(header6.nbits), U256::zero());
    }
}

/// The PoW parameters in the verifier's circuit view.
#[derive(Clone, Debug)]
pub struct CompiledPublicParams {
    pub k: usize, // common dimension of the matmul
    pub h: usize, // h × w is the size of the tiles we compute inner hash about
    pub w: usize,
    pub r: usize, // Common dimension denoting how often an inner hash is computed. Also the rank of the additive noise matrices
    pub blake_proof: BlakeProgram,
    pub moe: Option<CompiledMoE>, // Some iff `params.moe.is_some()`.
}

#[derive(Clone, Debug)]
pub struct CompiledMoE {
    pub routing_start_offset: u32,
    pub inner_indices: Vec<u32>,
    pub expert_idx: u16,
    /// Global column index where this expert's weight rows start
    /// (`expert_idx * η`);
    pub first_expert_col: u32,
}

impl PublicParams {
    /// Compile the Blake program and derive the noise seeds from the proposed
    /// header (and the proof-carried ancestor for the B side).
    pub fn compile(
        &self,
        _proposed_header: &IncompleteBlockHeader,
    ) -> Result<(CompiledPublicParams, Vec<AuxiliaryMsgLocation>, Vec<AuxiliaryCvLocation>)> {
        let (blake_program, msgs, cvs) = BlakeProgram::compile(self);

        let moe = self.moe_statement().map(|moe| CompiledMoE {
            routing_start_offset: moe.o_w_prev,
            inner_indices: self.a_inner_indices(),
            expert_idx: moe.w,
            first_expert_col: self.expert_rows() * moe.w as u32,
        });

        Ok((
            CompiledPublicParams {
                k: self.common_dim() as usize,
                h: self.h() as usize,
                w: self.w() as usize,
                r: self.rank() as usize,
                blake_proof: blake_program,
                moe,
            },
            msgs,
            cvs,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incomplete_block_header_serialized_size() {
        let header = IncompleteBlockHeader {
            version: 1,
            prev_block: [0xab; 32],
            merkle_root: [0xcd; 32],
            timestamp: 1234567890,
            nbits: 0x1d00ffff,
        };
        assert_eq!(header.to_bytes().len(), IncompleteBlockHeader::SERIALIZED_SIZE);
    }

    #[test]
    fn test_incomplete_block_header_roundtrip() {
        let original = IncompleteBlockHeader {
            version: 0x20000000,
            prev_block: [0xab; 32],
            merkle_root: [0xcd; 32],
            timestamp: 1715748000,
            nbits: 0x1d00ffff,
        };
        let serialized = original.to_bytes();
        let restored = IncompleteBlockHeader::from_bytes(&serialized).unwrap();
        assert_eq!(restored.version, original.version);
        assert_eq!(restored.prev_block, original.prev_block);
        assert_eq!(restored.merkle_root, original.merkle_root);
        assert_eq!(restored.timestamp, original.timestamp);
        assert_eq!(restored.nbits, original.nbits);
    }
}

/// `blake3(int8 plane root || scales plane root, key=key)`: the commitment
/// digest a prequantized (`int8 blk8 bf16s`) operand uses for `hash_a` /
/// `hash_b`, where each root is the plane's keyed chunk-Merkle root (the
/// reference miner's `commit_planes`) and `key` is the side's opening key
/// (`keyA` for A, `keyB` for B — already a derived subkey). Shapes are bound
/// by the statement, not the digest.
pub fn operand_digest_fp10(int_root: &Hash256, scales_root: &Hash256, key: &Hash256) -> Hash256 {
    blake3_digest(&[int_root.as_slice(), scales_root.as_slice()].concat(), Some(*key))
}

#[cfg(test)]
mod digest_tests {
    use super::*;

    /// Cross-checked against a direct keyed BLAKE3 computation of the reference's
    /// `commit_planes` digest (`blake3(commit(int8).digest || commit(scales).digest, key=key)`)
    /// with `key = bytes(range(32))` (the `PARENT` pin used across `ds_hash` tests).
    #[test]
    fn operand_digest_fp10_matches_reference() {
        let key: Hash256 = core::array::from_fn(|i| i as u8);
        let int_root: Hash256 = core::array::from_fn(|i| i as u8);
        let scales_root: Hash256 = core::array::from_fn(|i| (i as u8).wrapping_mul(7));
        let got = operand_digest_fp10(&int_root, &scales_root, &key);
        let want = hex_literal("64eaeed16fa66e94a72bd3833ef80f083dac14894c2311ba93fe7d0a8139a748");
        assert_eq!(got, want);
    }

    fn hex_literal(s: &str) -> Hash256 {
        core::array::from_fn(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
    }
}
