//! Binding layer shared by the Go bindings (zk_pow_ffi) and the Python bindings (py-pearl-mining).
//!
//! Holds what spans proof versions: [`CertificateVersion`] and the rules tying a block's
//! certificate version to a proof scheme. [`py_v2`] and [`py_v4`] carry the Python methods
//! for each version's types. May import from every version; no version imports it.

use anyhow::{Result, bail, ensure};

use crate::v2::api::seed::SeedDerivation;
use crate::v2::ffi::plain_proof::PlainProof;

pub mod py_v2;
pub mod py_v4;
pub mod py_v5;

/// Block certificate version (the wire format a block's certificate uses).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CertificateVersion {
    /// V1: dense (non-MoE) proofs only.
    ZkDense = 1,
    /// V2: MoE and dense proofs.
    ZkMoe = 2,
    /// V3: same wire layout as V2, salted noise-seed derivation.
    ZkV3 = 3,
    /// V4: FP8 proofs (dense or MoE).
    PlainFp8 = 4,
    /// V5: FP16 (A100) header-bound ZK certificate. NOTE the `PlainFp16` name is
    /// historical — the V5 consensus artifact is the header-bound ZK certificate
    /// (`verify_fp16_zk_cert_ffi`), not a plaintext proof.
    PlainFp16 = 5,
}

const _: () = assert!(CertificateVersion::PlainFp8 as u32 == crate::v4::CERT_VERSION);
const _: () = assert!(CertificateVersion::PlainFp16 as u32 == crate::v5::CERT_VERSION);

impl CertificateVersion {
    /// The noise-seed derivation this certificate version mandates. This is the
    /// single version→derivation mapping; the `api` layer only sees [`SeedDerivation`].
    pub fn seed_derivation(self) -> SeedDerivation {
        match self {
            Self::ZkDense | Self::ZkMoe | Self::PlainFp8 | Self::PlainFp16 => SeedDerivation::Legacy,
            Self::ZkV3 => SeedDerivation::Salted,
        }
    }
}

impl TryFrom<u32> for CertificateVersion {
    type Error = anyhow::Error;

    fn try_from(version: u32) -> Result<Self> {
        match version {
            v if v == Self::ZkDense as u32 => Ok(Self::ZkDense),
            v if v == Self::ZkMoe as u32 => Ok(Self::ZkMoe),
            v if v == Self::ZkV3 as u32 => Ok(Self::ZkV3),
            v if v == Self::PlainFp8 as u32 => Ok(Self::PlainFp8),
            v if v == Self::PlainFp16 as u32 => Ok(Self::PlainFp16),
            v => bail!("unknown certificate version: {v}"),
        }
    }
}

impl PlainProof {
    /// The lowest block certificate version that can certify this proof.
    pub fn min_cert_version(&self) -> CertificateVersion {
        if self.moe.is_some() {
            CertificateVersion::ZkMoe
        } else {
            CertificateVersion::ZkDense
        }
    }

    /// Checks that this Int7 proof can be certified at `cert_version` and
    /// returns the parsed version. Certificate version 4 (PlainFp8) requires
    /// [`crate::v4::api::plain_proof::PlainProofV4`].
    pub fn check_cert_version_eligible(&self, cert_version: u32) -> Result<CertificateVersion> {
        let version = CertificateVersion::try_from(cert_version)?;
        ensure!(
            version != CertificateVersion::PlainFp8,
            "Int7 PlainProof is not eligible at certificate version 4 (PlainFp8); use PlainProofV4"
        );
        ensure!(
            version != CertificateVersion::PlainFp16,
            "Int7 PlainProof is not eligible at certificate version 5 (PlainFp16); use the FP16 ZK certificate"
        );
        let min_version = self.min_cert_version() as u32;
        ensure!(
            min_version <= cert_version,
            "proof requires certificate version >= {min_version}, but the block requires version {cert_version} \
             (MoE proofs are only valid at or after the V2 crossover)"
        );
        Ok(version)
    }
}

#[cfg(test)]
mod tests {
    use pearl_blake3::{BLAKE3_DIGEST_SIZE, MerkleProof};

    use super::*;
    use crate::v2::ffi::plain_proof::{MatrixMerkleProof, MoEProofParams};

    fn dummy_merkle_proof() -> MerkleProof {
        MerkleProof {
            leaf_data: vec![],
            leaf_indices: vec![],
            total_leaves: 0,
            root: [0u8; BLAKE3_DIGEST_SIZE],
            siblings: vec![],
        }
    }

    fn dummy_matrix_proof() -> MatrixMerkleProof {
        MatrixMerkleProof {
            proof: dummy_merkle_proof(),
            row_indices: vec![1, 2, 3],
        }
    }

    fn dense_proof() -> PlainProof {
        PlainProof {
            m: 8,
            n: 4,
            k: 16,
            noise_rank: 2,
            a: dummy_matrix_proof(),
            bt: dummy_matrix_proof(),
            moe: None,
        }
    }

    fn moe_proof() -> PlainProof {
        PlainProof {
            moe: Some(MoEProofParams {
                e: 4,
                top_k: 2,
                expert_idx: 1,
                routing_end_offsets: vec![2, 4, 6, 8],
                inner_a_rows: vec![0, 1],
                routing_proof: dummy_merkle_proof(),
            }),
            ..dense_proof()
        }
    }

    #[test]
    fn min_cert_version_dense_is_v1_moe_is_v2() {
        assert_eq!(dense_proof().min_cert_version(), CertificateVersion::ZkDense);
        assert_eq!(moe_proof().min_cert_version(), CertificateVersion::ZkMoe);
    }

    #[test]
    fn dense_proof_eligible_under_both_versions() {
        assert_eq!(
            dense_proof()
                .check_cert_version_eligible(CertificateVersion::ZkDense as u32)
                .unwrap(),
            CertificateVersion::ZkDense
        );
        assert_eq!(
            dense_proof()
                .check_cert_version_eligible(CertificateVersion::ZkMoe as u32)
                .unwrap(),
            CertificateVersion::ZkMoe
        );
    }

    #[test]
    fn moe_proof_eligible_only_under_v2() {
        assert_eq!(
            moe_proof()
                .check_cert_version_eligible(CertificateVersion::ZkMoe as u32)
                .unwrap(),
            CertificateVersion::ZkMoe
        );
        let err = moe_proof()
            .check_cert_version_eligible(CertificateVersion::ZkDense as u32)
            .unwrap_err();
        assert!(err.to_string().contains("crossover"), "unexpected error: {err}");
    }

    #[test]
    fn both_proof_kinds_eligible_under_v3() {
        for proof in [dense_proof(), moe_proof()] {
            assert_eq!(
                proof.check_cert_version_eligible(CertificateVersion::ZkV3 as u32).unwrap(),
                CertificateVersion::ZkV3
            );
        }
    }

    #[test]
    fn seed_derivation_mapping() {
        assert_eq!(CertificateVersion::ZkDense.seed_derivation(), SeedDerivation::Legacy);
        assert_eq!(CertificateVersion::ZkMoe.seed_derivation(), SeedDerivation::Legacy);
        assert_eq!(CertificateVersion::ZkV3.seed_derivation(), SeedDerivation::Salted);
        assert_eq!(CertificateVersion::PlainFp8.seed_derivation(), SeedDerivation::Legacy);
    }

    #[test]
    fn unknown_cert_versions_rejected() {
        for version in [0u32, 5, u32::MAX] {
            assert!(dense_proof().check_cert_version_eligible(version).is_err());
        }
    }

    #[test]
    fn int7_proof_not_eligible_at_cert_v4() {
        let err = dense_proof()
            .check_cert_version_eligible(CertificateVersion::PlainFp8 as u32)
            .unwrap_err();
        assert!(err.to_string().contains("PlainProofV4"), "unexpected error: {err}");
    }
}
