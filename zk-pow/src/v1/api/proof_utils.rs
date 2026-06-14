use anyhow::{Result, bail, ensure};
use plonky2::field::extension::quadratic::QuadraticExtension;
use plonky2::field::types::{Field, Field64};
use plonky2::hash::hash_types::HashOut;
use plonky2::util::serialization::{Buffer, Read, Write};
use plonky2_field::goldilocks_field::GoldilocksField;
use primitive_types::U256;

use crate::ensure_eq;
use crate::v1::api::proof::{
    Hash256, IncompleteBlockHeader, MINING_CONFIG_RESERVED_SIZE, MMAType, MiningConfiguration, PeriodicPattern,
    PrivateProofParams, PublicProofParams, ZKProof,
};
use crate::v1::circuit::chip::blake3::program::{AuxiliaryCvLocation, AuxiliaryMsgLocation, BlakeProgram};
use crate::v1::circuit::pearl_circuit::PearlCircuitParams;
use pearl_blake3::blake3_digest;

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

/// Reduce a 32-byte hash into 4 Goldilocks field elements with minimal collision.
fn hash256_to_goldilocks_quartet(hash: &[u8; 32]) -> [GoldilocksField; 4] {
    let p = U256::from(GoldilocksField::ORDER);
    let p4 = p * p * p * p;
    let mut v = U256::from_little_endian(hash);
    if v >= p4 {
        v -= p4;
    }
    let mut elements = [GoldilocksField::ZERO; 4];
    for e in elements.iter_mut() {
        *e = GoldilocksField::from_canonical_u64((v % p).as_u64());
        v /= p;
    }
    elements
}

impl PeriodicPattern {
    /// Maximum number of dimensions in a PeriodicPattern.
    pub const NUM_DIMS: usize = 3;

    /// Create a PeriodicPattern from a byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        ensure!(
            data.len() == 2 * PeriodicPattern::NUM_DIMS,
            "Expected {} bytes, got {}",
            2 * PeriodicPattern::NUM_DIMS,
            data.len()
        );

        let mut shape = [(0u32, 0u32); PeriodicPattern::NUM_DIMS];
        let mut min_stride = 1u32;
        let mut is_done = false;

        for (i, chunk) in data.chunks(2).enumerate() {
            let factor = 1 + (chunk[0] as u32);
            let length = 1 + (chunk[1] as u32);

            if length == 1 || is_done {
                ensure!(factor == 1 && length == 1, "Using non-canonical representation");
                is_done = true;
            } else if factor <= 1 && min_stride != 1 {
                bail!("A single stride must not be broken");
            }

            ensure!(
                min_stride <= (1 << 24) / (factor * length),
                "Pattern must have period <= 2^24"
            );
            let stride = factor * min_stride;
            shape[i] = (stride, length);
            min_stride = stride * length;
        }

        Ok(Self { shape })
    }

    /// Serialize to exactly 2 * PeriodicPattern::NUM_DIMS bytes (6 bytes).
    pub fn to_bytes(&self) -> [u8; 2 * PeriodicPattern::NUM_DIMS] {
        let mut data = [0u8; 2 * PeriodicPattern::NUM_DIMS];
        let mut min_stride = 1u32;

        for (i, &(stride, length)) in self.shape.iter().enumerate() {
            let factor = stride / min_stride;
            data[2 * i] = (factor - 1) as u8;
            data[2 * i + 1] = (length - 1) as u8;
            min_stride = stride * length;
        }

        data
    }

    /// Convert pattern to a list of indices.
    pub fn to_list(&self) -> Vec<u32> {
        let mut res = vec![0u32];
        for &(stride, length) in &self.shape {
            let mut new_res = Vec::with_capacity(res.len() * length as usize);
            for i in 0..length {
                for &r in &res {
                    new_res.push(r + i * stride);
                }
            }
            res = new_res;
        }
        res
    }

    pub fn max(&self) -> u32 {
        self.to_list().into_iter().max().unwrap()
    }

    /// Create a PeriodicPattern from a list of indices.
    pub fn from_list(pattern: &[u32]) -> Result<Self> {
        ensure!(!pattern.is_empty(), "Pattern cannot be empty");

        ensure!(
            pattern.windows(2).all(|w| w[0] < w[1]),
            "Pattern must be sorted and have no duplicates"
        );
        ensure!(pattern[0] == 0, "Pattern must start at 0");

        let mut p: Vec<u32> = pattern.to_vec();

        let mut shape_vec = Vec::new();

        while p.len() > 1 {
            let mut found = false;
            for period in 1..p.len() {
                if p.len().is_multiple_of(period) {
                    let s = p[period];
                    let is_periodic = (0..p.len() - period).all(|i| p[i] + s == p[i + period]);
                    if is_periodic {
                        shape_vec.push((s, (p.len() / period) as u32));
                        p.truncate(period);
                        found = true;
                        break;
                    }
                }
            }
            ensure!(found, "Pattern is not periodic");
        }

        // Reverse and pad to NUM_DIMS with (period, 1) tuples
        shape_vec.reverse();
        let period = shape_vec.last().map_or(1, |&(s, l)| s * l);

        while shape_vec.len() < PeriodicPattern::NUM_DIMS {
            shape_vec.push((period, 1));
        }

        let result = Self {
            shape: shape_vec.try_into().unwrap(),
        };
        ensure!(result.is_valid(), "Constructed pattern is not valid");
        Ok(result)
    }

    /// Check if an offset is valid for this pattern.
    pub fn offset_is_valid(&self, mut offset: u32) -> bool {
        for &(stride, length) in self.shape.iter().rev() {
            offset %= stride * length;
            if offset >= stride {
                return false;
            }
        }
        true
    }

    /// Check if this pattern is valid (roundtrips through serialization).
    pub fn is_valid(&self) -> bool {
        match Self::from_bytes(&self.to_bytes()) {
            Ok(restored) => restored.shape == self.shape,
            Err(_) => false,
        }
    }

    /// Get the period of this pattern.
    pub fn period(&self) -> u32 {
        let &(stride, length) = self.shape.last().unwrap();
        stride * length
    }

    /// Get the size (number of elements) of this pattern.
    pub fn size(&self) -> u32 {
        self.shape.iter().map(|&(_, length)| length).product()
    }

    /// Get the list of indices with an offset applied.
    pub fn indices_with_offset(&self, offset: u32) -> Vec<u32> {
        self.to_list().into_iter().map(|i| i + offset).collect()
    }
}

impl PublicProofParams {
    pub fn a_rows_indices(&self) -> Vec<u32> {
        self.mining_config.rows_pattern.indices_with_offset(self.t_rows)
    }
    pub fn b_cols_indices(&self) -> Vec<u32> {
        self.mining_config.cols_pattern.indices_with_offset(self.t_cols)
    }
    pub fn h(&self) -> usize {
        self.mining_config.rows_pattern.size() as usize
    }
    pub fn w(&self) -> usize {
        self.mining_config.cols_pattern.size() as usize
    }
    pub fn rank(&self) -> usize {
        self.mining_config.rank as usize
    }

    pub fn common_dim(&self) -> usize {
        self.mining_config.common_dim as usize
    }

    pub fn job_key(&self) -> Hash256 {
        blake3_digest(
            &[&self.block_header.to_bytes()[..], &self.mining_config.to_bytes()[..]].concat(),
            None,
        )
    }

    /// Compute commitment hash (B's noise seed, A's noise seed).
    pub fn commitment_hash(&self, job_key: Hash256) -> (Hash256, Hash256) {
        let b_noise_seed = blake3_digest(&[&job_key[..], &self.hash_b[..]].concat(), None);
        let a_noise_seed = blake3_digest(&[&b_noise_seed[..], &self.hash_a[..]].concat(), None);
        (b_noise_seed, a_noise_seed)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_header: IncompleteBlockHeader,
        mining_config: MiningConfiguration,
        hash_a: Hash256,
        hash_b: Hash256,
        hash_jackpot: Hash256,
        m: u32,
        n: u32,
        t_rows: u32,
        t_cols: u32,
    ) -> Self {
        Self {
            block_header,
            mining_config,
            hash_a,
            hash_b,
            hash_jackpot,
            m,
            n,
            t_rows,
            t_cols,
        }
    }

    // Without hash_a, hash_b and jackpot
    #[allow(clippy::too_many_arguments)]
    pub fn new_dummy(
        block_header: IncompleteBlockHeader,
        mining_configuration: MiningConfiguration,
        m: u32,
        n: u32,
        t_rows: u32,
        t_cols: u32,
    ) -> Self {
        Self::new(
            block_header,
            mining_configuration,
            [1; 32], // dummy hash_a
            [2; 32], // dummy hash_b
            [3; 32], // dummy hash_jackpot
            m,
            n,
            t_rows,
            t_cols,
        )
    }

    #[cfg(test)]
    pub fn new_for_tests(m: u32, n: u32, k: u32) -> Self {
        Self::new_dummy(
            IncompleteBlockHeader::new_for_test(0x207FFFFF),
            MiningConfiguration {
                common_dim: k,
                rank: 128,
                mma_type: MMAType::Int7xInt7ToInt32,
                rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
                cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105])
                    .unwrap(),
                reserved: MiningConfiguration::RESERVED_VALUE,
            },
            m,
            n,
            0, // t_rows - offset aligned to rows_pattern
            0, // t_cols - offset aligned to cols_pattern
        )
    }
}

impl MiningConfiguration {
    /// Size of reserved field in bytes.
    pub const RESERVED_SIZE: usize = MINING_CONFIG_RESERVED_SIZE;

    /// Default reserved bytes (all zeros).
    pub const RESERVED_VALUE: [u8; Self::RESERVED_SIZE] = [0u8; Self::RESERVED_SIZE];

    /// Size of serialized MiningConfiguration in bytes.
    /// 4 (common_dim) + 2 (rank) + 2 (mma_type) + 6 (rows_pattern) + 6 (cols_pattern) + 32 (reserved) = 52
    /// Note: IncompleteBlockHeader (76) + MiningConfiguration (52) = 128 bytes = 2 blake3 blocks.
    pub const SERIALIZED_SIZE: usize = 52;

    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut bytes = Vec::with_capacity(Self::SERIALIZED_SIZE);
        bytes.extend_from_slice(&self.common_dim.to_le_bytes()); // 4 bytes
        bytes.extend_from_slice(&self.rank.to_le_bytes()); // 2 bytes
        bytes.extend_from_slice(&(self.mma_type as u16).to_le_bytes()); // 2 bytes
        bytes.extend_from_slice(&self.rows_pattern.to_bytes()); // 6 bytes
        bytes.extend_from_slice(&self.cols_pattern.to_bytes()); // 6 bytes
        bytes.extend_from_slice(&self.reserved); // 24 bytes
        bytes.try_into().unwrap()
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        ensure!(
            data.len() == Self::SERIALIZED_SIZE,
            "Expected {} bytes, got {}",
            Self::SERIALIZED_SIZE,
            data.len()
        );

        let common_dim = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let rank = u16::from_le_bytes(data[4..6].try_into().unwrap());
        let mma_type_raw = u16::from_le_bytes(data[6..8].try_into().unwrap());
        let mma_type = match mma_type_raw {
            0 => MMAType::Int7xInt7ToInt32,
            _ => anyhow::bail!("Invalid MMAType: {}", mma_type_raw),
        };
        let rows_pattern = PeriodicPattern::from_bytes(&data[8..14])?;
        let cols_pattern = PeriodicPattern::from_bytes(&data[14..20])?;
        let reserved: [u8; Self::RESERVED_SIZE] = data[20..52].try_into().unwrap();
        ensure!(reserved == Self::RESERVED_VALUE, "Reserved field must be all zeros");
        Ok(Self {
            common_dim,
            rank,
            mma_type,
            rows_pattern,
            cols_pattern,
            reserved,
        })
    }

    pub fn dot_product_length(&self) -> usize {
        let common_dim = self.common_dim as usize;
        let rank = self.rank as usize;
        common_dim - common_dim % rank
    }
}

impl IncompleteBlockHeader {
    /// Size of serialized IncompleteBlockHeader in bytes.
    /// 4 (version) + 32 (prev_block) + 32 (merkle_root) + 4 (timestamp) + 4 (nbits) = 76
    pub const SERIALIZED_SIZE: usize = 76;

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

        Ok(Self {
            version,
            prev_block,
            merkle_root,
            timestamp,
            nbits,
        })
    }
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
    pub job_key: Hash256, // Hash of IncompleteBlockHeader || MiningConfiguration; used for deriving the commitment hash.
    pub k: usize,         // common dimension of the matmul
    pub h: usize,         // h × w is the size of the tiles we compute inner hash about
    pub w: usize,
    pub r: usize, // Common dimension denoting how often an inner hash is computed. Also the rank of the additive noise matrices
    // pub s: u8, // =128. Number of possible noise values. Even number
    // A specification how the hashes are combined to create the commitment hash.
    // Logically, it specifies what rows of A and cols of B generate the jackpot tile,
    // and number of rows A has and columns B has.
    pub blake_proof: BlakeProgram,
    pub a_rows_indices: Vec<usize>,
    pub b_cols_indices: Vec<usize>,
    pub commitment_hash: (Hash256, Hash256), // (b_noise_seed, a_noise_seed)
}

impl From<&PublicProofParams> for CompiledPublicParams {
    fn from(params: &PublicProofParams) -> Self {
        let (res, _, _) = params.compile();
        res
    }
}

impl PublicProofParams {
    pub fn dot_product_length(&self) -> usize {
        self.mining_config.dot_product_length()
    }

    // Returns the compiled program and a list giving expected order of external data.
    pub fn compile(&self) -> (CompiledPublicParams, Vec<AuxiliaryMsgLocation>, Vec<AuxiliaryCvLocation>) {
        debug_assert!(
            self.a_rows_indices().windows(2).all(|w| w[0] < w[1]),
            "a_rows_indices must be strictly ascending"
        );
        debug_assert!(
            self.b_cols_indices().windows(2).all(|w| w[0] < w[1]),
            "b_cols_indices must be strictly ascending"
        );

        let (blake_program, msgs, cvs) = BlakeProgram::new(self);
        let job_key = self.job_key();

        (
            CompiledPublicParams {
                job_key,
                k: self.common_dim(),
                h: self.h(),
                w: self.w(),
                r: self.rank(),
                blake_proof: blake_program,
                a_rows_indices: self.a_rows_indices().iter().map(|&x| x as usize).collect(),
                b_cols_indices: self.b_cols_indices().iter().map(|&x| x as usize).collect(),
                commitment_hash: self.commitment_hash(job_key),
            },
            msgs,
            cvs,
        )
    }

    /// Verifies that the hashes in the witness match the computed hashes from private_params.
    /// This is for production use where the prover has real witness data.
    pub fn sanity_check_private_params(&self, private_params: &PrivateProofParams) -> Result<()> {
        let (compiled, external_msgs, external_cvs) = self.compile();
        ensure_eq!(
            external_msgs.len(),
            private_params.external_msgs.len(),
            "external_msgs length mismatch expected={}, found={}",
            external_msgs.len(),
            private_params.external_msgs.len()
        );
        ensure_eq!(
            external_cvs.len(),
            private_params.external_cvs.len(),
            "external_cvs length mismatch expected={}, found={}",
            external_cvs.len(),
            private_params.external_cvs.len()
        );

        let (hash_a, hash_b) = compiled.blake_proof.evaluate_blake(compiled.job_key, private_params)?;
        ensure_eq!(self.hash_a, hash_a, "hash_a mismatch");
        ensure_eq!(self.hash_b, hash_b, "hash_b mismatch");

        Ok(())
    }

    /// Computes hashes using dummy witness data. For tests and warmup only.
    /// This fills external_msgs and external_cvs with zeros of the correct size,
    /// then computes and sets the hashes in the witness. hash_jackpot is set to a dummy value.
    /// Returns the PrivateProofParams with correctly-sized external data.
    pub fn fill_dummy_merkle_proof(&mut self, mut private_params: PrivateProofParams) -> Result<PrivateProofParams> {
        let (compiled, external_msgs, external_cvs) = self.compile();

        // Fill external_msgs and external_cvs with zeros of correct size
        private_params.external_msgs = vec![[0u8; 64]; external_msgs.len()];
        private_params.external_cvs = vec![[0u8; 32]; external_cvs.len()];

        let hashes = compiled.blake_proof.evaluate_blake(self.job_key(), &private_params)?;
        (self.hash_a, self.hash_b) = hashes;

        Ok(private_params)
    }
}

#[cfg(test)]
mod tests {
    use pearl_blake3::BLAKE3_CHUNK_LEN;

    use super::*;

    #[test]
    fn test_from_public_proof_params() {
        let public_params = PublicProofParams::new_dummy(
            IncompleteBlockHeader {
                version: 0,
                prev_block: [0; 32],
                merkle_root: [0; 32],
                timestamp: 0,
                nbits: 0,
            },
            MiningConfiguration {
                common_dim: 4096 - 64,
                rank: 128,
                mma_type: MMAType::Int7xInt7ToInt32,
                rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
                cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105])
                    .unwrap(),
                reserved: MiningConfiguration::RESERVED_VALUE,
            },
            256, // m: rows of A
            128, // n: columns of B
            7,
            6,
        );

        let compiled = CompiledPublicParams::from(&public_params);

        // Verify the conversion maintains all values correctly
        assert_eq!(compiled.job_key, public_params.job_key());
        assert_eq!(compiled.k, public_params.common_dim());
        assert_eq!(compiled.h, public_params.h());
        assert_eq!(compiled.w, public_params.w());
        assert_eq!(compiled.r, public_params.rank());

        assert_eq!(compiled.blake_proof.strip_length, public_params.dot_product_length());
        assert_eq!(compiled.blake_proof.num_auxiliary_msgs, 136);
        assert_eq!(compiled.blake_proof.num_auxiliary_cvs, 65);
        assert_eq!(compiled.blake_proof.instructions.len(), 1525);
    }

    #[test]
    fn test_mining_configuration_serialized_size() {
        // Verify the constant matches the actual serialized size
        let config = MiningConfiguration {
            common_dim: 4096,
            rank: 128,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9]).unwrap(),
            reserved: MiningConfiguration::RESERVED_VALUE,
        };
        assert_eq!(config.to_bytes().len(), MiningConfiguration::SERIALIZED_SIZE);
    }

    #[test]
    fn test_mining_configuration_roundtrip() {
        let original = MiningConfiguration {
            common_dim: 2048,
            rank: 64,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
            reserved: MiningConfiguration::RESERVED_VALUE,
        };
        let serialized = original.to_bytes();
        let restored = MiningConfiguration::from_bytes(&serialized).unwrap();
        assert_eq!(restored.common_dim, original.common_dim);
        assert_eq!(restored.rank, original.rank);
        assert_eq!(restored.rows_pattern.to_list(), original.rows_pattern.to_list());
        assert_eq!(restored.cols_pattern.to_list(), original.cols_pattern.to_list());
        assert_eq!(restored.reserved, original.reserved);
    }

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

    // =========================================================================
    // PeriodicPattern tests
    // =========================================================================

    #[test]
    fn test_periodic_pattern_from_list_to_list_roundtrip() {
        let test_cases: &[&[u32]] = &[
            &[0, 1, 2, 3],                                                   // simple consecutive
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],         // 16 consecutive
            &[0, 8, 64, 72],                                                 // sparse pattern
            &[0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105], // pearl cols
            &[0, 8],                                                         // simple pair
            &[0, 2, 4, 6],                                                   // even numbers
            &[0, 1, 4, 5, 8, 9, 12, 13],                                     // pairs with stride 4
        ];

        for pattern_list in test_cases {
            let pattern = PeriodicPattern::from_list(pattern_list).unwrap();
            let mut result = pattern.to_list();
            result.sort();
            assert_eq!(&result, pattern_list, "Roundtrip failed for {:?}", pattern_list);
        }
    }

    #[test]
    fn test_periodic_pattern_from_bytes_to_bytes_roundtrip() {
        let test_cases: &[&[u32]] = &[
            &[0, 1, 2, 3],
            &[0, 8, 64, 72],
            &[0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105],
            &[0, 8],
        ];

        for pattern_list in test_cases {
            let pattern = PeriodicPattern::from_list(pattern_list).unwrap();
            let serialized = pattern.to_bytes();
            let restored = PeriodicPattern::from_bytes(&serialized).unwrap();
            assert_eq!(restored.shape, pattern.shape, "Bytes roundtrip failed for {:?}", pattern_list);
            assert_eq!(restored.to_list(), pattern.to_list());
        }
    }

    #[test]
    fn test_periodic_pattern_is_valid() {
        let test_cases: &[&[u32]] = &[&[0, 1, 2, 3], &[0, 8, 64, 72], &[0, 1, 8, 9], &[0]];

        for pattern_list in test_cases {
            let pattern = PeriodicPattern::from_list(pattern_list).unwrap();
            assert!(pattern.is_valid(), "Pattern {:?} should be valid", pattern_list);
        }
    }

    #[test]
    fn test_periodic_pattern_to_bytes_length() {
        let expected_len = 2 * PeriodicPattern::NUM_DIMS;

        let pattern1 = PeriodicPattern::from_list(&[0, 1, 2, 3]).unwrap();
        assert_eq!(pattern1.to_bytes().len(), expected_len);

        let pattern2 = PeriodicPattern::from_list(&[0, 1, 8, 9]).unwrap();
        assert_eq!(pattern2.to_bytes().len(), expected_len);
    }

    #[test]
    fn test_periodic_pattern_period() {
        assert_eq!(PeriodicPattern::from_list(&[0, 1, 2, 3]).unwrap().period(), 4);
        assert_eq!(PeriodicPattern::from_list(&[0, 1, 8, 9]).unwrap().period(), 16);
    }

    #[test]
    fn test_periodic_pattern_size() {
        assert_eq!(PeriodicPattern::from_list(&[0, 1, 2, 3]).unwrap().size(), 4);
        assert_eq!(PeriodicPattern::from_list(&[0, 1, 8, 9]).unwrap().size(), 4);
        assert_eq!(
            PeriodicPattern::from_list(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
                .unwrap()
                .size(),
            16
        );
    }

    #[test]
    fn test_periodic_pattern_non_periodic_rejected() {
        // gap of 1, then gap of 2 - not periodic
        let result = PeriodicPattern::from_list(&[0, 1, 3]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not periodic"));
    }

    #[test]
    fn test_periodic_pattern_not_starting_at_zero_rejected() {
        let result = PeriodicPattern::from_list(&[1, 2, 3, 4]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("start at 0"));
    }

    #[test]
    fn test_periodic_pattern_empty_rejected() {
        let result = PeriodicPattern::from_list(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_periodic_pattern_from_bytes_wrong_length_rejected() {
        let expected_len = 2 * PeriodicPattern::NUM_DIMS;

        let result = PeriodicPattern::from_bytes(&[0, 1, 2]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains(&format!("Expected {}", expected_len))
        );

        let result = PeriodicPattern::from_bytes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(result.is_err());
    }

    /// Confirms that a ZKProof with zeta bytes = [0xFF; 16] returns Err from
    /// ZKProof::zeta(). Each 8-byte LE chunk decodes to 0xFFFFFFFFFFFFFFFF which
    /// exceeds GoldilocksField::ORDER, causing read_field() to fail deserialization.
    #[test]
    fn test_invalid_zeta_returns_error() {
        let proof = ZKProof {
            pow_bits: [0; 3],
            rate_bits: [0; 3],
            zeta: [0xFF; 16],
            plonky2_proof: vec![],
        };

        assert!(proof.zeta().is_err());
    }

    #[test]
    fn test_compile_non_chunk_aligned_dimensions() {
        // m=255, n=127 with k=4032 causes m*k and n*k to not be multiples of
        // BLAKE3_CHUNK_LEN (1024), exercising the padded_chunk_len path in
        // compile() and recursive_compilation().  k remains a multiple of 64 so
        // that row boundaries still align with BLAKE3 message blocks.
        let mut params = PublicProofParams::new_for_tests(255, 127, 4032);
        let (m, n, k) = (params.m as usize, params.n as usize, params.common_dim());
        assert_ne!(m * k % BLAKE3_CHUNK_LEN, 0);
        assert_ne!(n * k % BLAKE3_CHUNK_LEN, 0);

        let (compiled, _, _) = params.compile();
        assert!(!compiled.blake_proof.instructions.is_empty());

        let strip_len = params.dot_product_length();
        let private_params = PrivateProofParams {
            s_a: vec![vec![0i8; strip_len]; params.h()],
            s_b: vec![vec![0i8; strip_len]; params.w()],
            external_msgs: vec![],
            external_cvs: vec![],
        };
        params
            .fill_dummy_merkle_proof(private_params)
            .expect("hash round-trip should succeed for non-aligned dimensions");
    }
}

impl PublicProofParams {
    /// Fixed-size public_data: config(52) | hash_a(32) | hash_b(32) | hash_jackpot(32) | m(4) | n(4) | t_rows(4) | t_cols(4)
    pub const PUBLICDATA_SIZE: usize = 164;

    /// Serialize as wire format.
    pub fn to_bytes(&self) -> [u8; Self::PUBLICDATA_SIZE] {
        let mut buf = [0u8; Self::PUBLICDATA_SIZE];
        buf[0..52].copy_from_slice(&self.mining_config.to_bytes());
        buf[52..84].copy_from_slice(&self.hash_a);
        buf[84..116].copy_from_slice(&self.hash_b);
        buf[116..148].copy_from_slice(&self.hash_jackpot);
        buf[148..152].copy_from_slice(&self.m.to_le_bytes());
        buf[152..156].copy_from_slice(&self.n.to_le_bytes());
        buf[156..160].copy_from_slice(&self.t_rows.to_le_bytes());
        buf[160..164].copy_from_slice(&self.t_cols.to_le_bytes());
        buf
    }

    /// Deserialize from a 164-byte wire `public_data` blob plus the block header.
    pub fn from_bytes(block_header: IncompleteBlockHeader, bytes: &[u8; Self::PUBLICDATA_SIZE]) -> Result<Self> {
        let mining_config = MiningConfiguration::from_bytes(&bytes[0..52])?;
        let hash_a: [u8; 32] = bytes[52..84].try_into().unwrap();
        let hash_b: [u8; 32] = bytes[84..116].try_into().unwrap();
        let hash_jackpot: [u8; 32] = bytes[116..148].try_into().unwrap();
        let m = u32::from_le_bytes(bytes[148..152].try_into().unwrap());
        let n = u32::from_le_bytes(bytes[152..156].try_into().unwrap());
        let t_rows = u32::from_le_bytes(bytes[156..160].try_into().unwrap());
        let t_cols = u32::from_le_bytes(bytes[160..164].try_into().unwrap());

        ensure!(
            mining_config.rows_pattern.offset_is_valid(t_rows),
            "t_rows must be a valid offset for rows_pattern"
        );
        ensure!(
            mining_config.cols_pattern.offset_is_valid(t_cols),
            "t_cols must be a valid offset for cols_pattern"
        );

        Ok(Self {
            block_header,
            mining_config,
            hash_a,
            hash_b,
            hash_jackpot,
            m,
            n,
            t_rows,
            t_cols,
        })
    }

    /// Compute the HASH_PUBLIC_DATA identifier for the preprocessed columns.
    ///
    /// This is `blake3("V1" || block_header_bytes || public_data_bytes || pow_bits || rate_bits)`,
    /// interpreted as 4 Goldilocks field elements.  The result is fed into the STARK Fiat-Shamir
    /// challenger before the trace commitment, binding zeta to the preprocessed data (grinding
    /// resistance).  The `"V1"` prefix is a domain separator that disambiguates this hash from
    /// other blake3 uses in the protocol.
    ///
    /// `stark_degree_bits` is fully determined by the public params (via
    /// `CompiledPublicParams::degree_bits`) and therefore does not need to appear explicitly in
    /// the preimage.
    ///
    /// The 32-byte hash is reduced into 4 Goldilocks field elements via
    /// [`bytes32_to_goldilocks_quartet`].
    pub fn public_data_commitment(&self, circuit_params: &PearlCircuitParams) -> HashOut<GoldilocksField> {
        let block_header_bytes = self.block_header.to_bytes();
        let public_data = self.to_bytes();
        let pow_bytes: [u8; 3] = circuit_params.pow_bits.map(|b| b as u8);
        let rate_bytes: [u8; 3] = circuit_params.rate_bits.map(|b| b as u8);
        let input = [
            b"V1",
            &block_header_bytes[..],
            &public_data[..],
            &pow_bytes[..],
            &rate_bytes[..],
        ]
        .concat();
        let hash = blake3_digest(&input, None);

        HashOut {
            elements: hash256_to_goldilocks_quartet(&hash),
        }
    }
}

impl ZKProof {
    /// Fixed-size preamble: pow_bits(3) | rate_bits(3) | zeta(16)
    pub const PROOFDATA_PREAMBLE: usize = 22;

    /// Create a new ZKProof from circuit params and raw proof outputs.
    /// `zeta` is the STARK challenge point serialized as 2 Goldilocks field elements (8 bytes each, LE).
    pub fn new(pow_bits: [u8; 3], rate_bits: [u8; 3], zeta: QuadraticExtension<GoldilocksField>, plonky2_proof: Vec<u8>) -> Self {
        let mut buf = Vec::with_capacity(16);
        buf.write_field_vec(&[zeta.0[0], zeta.0[1]])
            .expect("field vec write cannot fail");
        Self {
            pow_bits,
            rate_bits,
            zeta: buf.try_into().expect("zeta must be 16 bytes"),
            plonky2_proof,
        }
    }

    /// Get the rate_bits for a specific stage (0, 1, or 2)
    /// Panics if stage >= 3
    pub fn get_rate_bits(&self, stage: usize) -> usize {
        assert!(stage < 3, "Stage must be 0, 1, or 2, got {}", stage);
        self.rate_bits[stage] as usize
    }

    /// Get the pow_bits for a specific stage (0, 1, or 2)
    /// Panics if stage >= 3
    pub fn get_pow_bits(&self, stage: usize) -> usize {
        assert!(stage < 3, "Stage must be 0, 1, or 2, got {}", stage);
        self.pow_bits[stage] as usize
    }

    /// Deserialize proof_data bytes into a `ZKProof`.
    /// `proof_data` layout: `pow_bits(3) | rate_bits(3) | zeta(16) | plonky2_proof`
    pub fn from_bytes(proof_data: &[u8]) -> Result<Self> {
        ensure!(
            proof_data.len() >= Self::PROOFDATA_PREAMBLE,
            "proof_data too short: need at least {} bytes for pow_bits/rate_bits/zeta header",
            Self::PROOFDATA_PREAMBLE
        );
        Ok(Self {
            pow_bits: proof_data[0..3].try_into().unwrap(),
            rate_bits: proof_data[3..6].try_into().unwrap(),
            zeta: proof_data[6..Self::PROOFDATA_PREAMBLE].try_into().unwrap(),
            plonky2_proof: proof_data[Self::PROOFDATA_PREAMBLE..].to_vec(),
        })
    }

    /// Serialize the proof preamble and body: `pow_bits(3) | rate_bits(3) | zeta(16) | plonky2_proof`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::PROOFDATA_PREAMBLE + self.plonky2_proof.len());
        buf.extend_from_slice(&self.pow_bits);
        buf.extend_from_slice(&self.rate_bits);
        buf.extend_from_slice(&self.zeta);
        buf.extend_from_slice(&self.plonky2_proof);
        buf
    }

    /// Deserialize both halves: `public_data(164) | proof_data`.
    ///
    /// `public_data` layout: `config(52) | hash_a(32) | hash_b(32) | hash_jackpot(32) | m(4) | n(4) | t_rows(4) | t_cols(4)`
    /// `proof_data` layout: `pow_bits(3) | rate_bits(3) | zeta(16) | plonky2_proof`
    pub fn deserialize(
        block_header: IncompleteBlockHeader,
        public_data: &[u8; PublicProofParams::PUBLICDATA_SIZE],
        proof_data: &[u8],
    ) -> Result<(PublicProofParams, Self)> {
        let params = PublicProofParams::from_bytes(block_header, public_data)?;
        let zk_proof = Self::from_bytes(proof_data)?;
        Ok((params, zk_proof))
    }

    /// Serialize both halves. Mirror of [`ZKProof::deserialize`].
    pub fn serialize(&self, params: &PublicProofParams) -> ([u8; PublicProofParams::PUBLICDATA_SIZE], Vec<u8>) {
        (params.to_bytes(), self.to_bytes())
    }

    /// Get the STARK challenge point zeta as a QuadraticExtension (2 field elements, 8 bytes each, little-endian).
    pub fn zeta(&self) -> Result<QuadraticExtension<GoldilocksField>> {
        let elements = Buffer::new(&self.zeta)
            .read_field_vec(2)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(QuadraticExtension([elements[0], elements[1]]))
    }
}

/// Compute blake3(jackpot_msg, key=commitment_hash) where jackpot_msg is 16 u32 words in little-endian.
pub fn compute_jackpot_hash(jackpot: &[u32; 16], commitment_hash: [u8; 32]) -> Hash256 {
    let msg: [u8; 64] = std::array::from_fn(|i| jackpot[i / 4].to_le_bytes()[i % 4]);
    blake3_digest(&msg, Some(commitment_hash))
}

/// Convert a 32-byte hash to an array of 8 field elements (4 bytes each, little-endian).
pub fn hash_to_u32_field_array<F: plonky2_field::types::Field>(hash: &[u8; 32]) -> [F; 8] {
    core::array::from_fn(|i| F::from_canonical_u32(u32::from_le_bytes(hash[i * 4..(i + 1) * 4].try_into().unwrap())))
}

/// Convert an array of 8 field elements (4 bytes each, little-endian) to a 32-byte hash.
pub fn u32_field_array_to_hash<F: plonky2_field::types::PrimeField64>(array: &[F; 8]) -> [u8; 32] {
    core::array::from_fn(|i| {
        let word_bytes = (array[i / 4].to_canonical_u64() as u32).to_le_bytes();
        word_bytes[i % 4]
    })
}

/// Convert a U256 to an array of 8 field elements (4 bytes each, little-endian).
pub fn u256_to_u32_field_array<F: plonky2_field::types::Field>(value: U256) -> [F; 8] {
    let mut bytes = [0u8; 32];
    value.to_little_endian(&mut bytes);
    hash_to_u32_field_array(&bytes)
}
