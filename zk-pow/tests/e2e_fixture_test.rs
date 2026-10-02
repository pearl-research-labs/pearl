//! End-to-end fixture tests for the frozen `v1`/`v2` proof stacks.
//!
//! Each test drives a production byte-level interface from committed fixture
//! bytes, pinning both the wire formats and prover/verifier behavior:
//!
//! - `plain_verify`: serialized `PlainProof` bytes -> accept/reject
//! - `zk_prove`:     serialized `PlainProof` bytes -> proof bytes
//! - `zk_verify`:    proof bytes -> accept/reject
//!
//! Fixtures (all committed):
//! - `fixures/v2_plain_proof.bin`: bincode-serialized `PlainProof`, mined with
//!   the same seeded parameters as the v2 compat tests. v1 and v2 share the
//!   non-MoE mining algorithm, so the same plain proof exercises both stacks.
//! - `fixures/v2_plain_proof_moe.bin`: bincode-serialized MoE `PlainProof`,
//!   mined with the same seeded parameters as the committed MoE ZK fixture.
//! - `fixures/v2_stark_proof.bin`: `public_data | proof_data` of the v2 ZK
//!   proof for that same plain proof (see `test_generate_v2_fixture`).
//! - `fixures/v2_stark_proof_moe.bin`: master-generated MoE ZK proof,
//!   `public_data_len(4 LE) | public_data | proof_data` (MoE public data is
//!   variable-length, hence the prefix).
//! - `src/v1/fixures/stark_proof.bin`: master-generated v1 ZK proof.
//!
//! Regenerate the plain fixtures (only if the freeze is deliberately broken):
//!   cargo test --release --test e2e_fixture_test -- --ignored generate_plain_fixture
//!   cargo test --release --test e2e_fixture_test -- --ignored generate_moe_plain_fixture

use rand_chacha::rand_core::SeedableRng;
use zk_pow::api::seed::SeedDerivation;
use zk_pow::ffi::plain_proof::PlainProof;

/// Mining/proving parameters. Must stay in sync with `v1_params()` /
/// `params()` in the v1/v2 compat tests: the committed ZK fixtures were
/// produced from exactly this seeded mining run.
const MINING_SEED: u64 = 0xdeadbeef;
const NBITS: u32 = 0x1D2FFFFF;
const M: usize = 6144;
const N: usize = 4096;
const RANK: u16 = 64;
const K: usize = 16 * RANK as usize + 192;

fn fixture_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn read_fixture(rel: &str) -> Vec<u8> {
    std::fs::read(fixture_path(rel))
        .unwrap_or_else(|e| panic!("missing fixture {rel}: {e} (regenerate with the ignored generator test)"))
}

/// Mirror of the `cfg(test)`-gated `IncompleteBlockHeader::new_for_test(NBITS)`
/// used when the committed fixtures were generated.
fn v2_header() -> zk_pow::v2::api::proof::IncompleteBlockHeader {
    zk_pow::v2::api::proof::IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x66666666,
        nbits: NBITS,
    }
}

/// Same header through the frozen v1 types.
fn v1_header() -> zk_pow::v1::api::proof::IncompleteBlockHeader {
    zk_pow::v1::api::proof::IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x66666666,
        nbits: NBITS,
    }
}

fn v2_mining_config() -> zk_pow::v2::api::proof::MiningConfiguration {
    use zk_pow::v2::api::proof::{MMAType, MiningConfiguration, PeriodicPattern};
    MiningConfiguration {
        common_dim: K as u32,
        rank: RANK,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73]).unwrap(),
        moe: None,
    }
}

/// Mines the canonical fixture plain proof (deterministic: seeded rng).
fn mine_fixture_plain_proof() -> PlainProof {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(MINING_SEED);
    zk_pow::v2::mine::try_mine_one(
        &mut rng,
        M,
        N,
        K,
        v2_header(),
        v2_mining_config(),
        None,
        false,
        SeedDerivation::Legacy,
    )
    .expect("mining failed")
    .expect("seeded mining must find a solution on the first attempt")
}

/// Writes `fixures/v2_plain_proof.bin`. Run once and commit the output.
#[test]
#[ignore] // Run with: cargo test --release --test e2e_fixture_test -- --ignored generate_plain_fixture
fn generate_plain_fixture() {
    let proof = mine_fixture_plain_proof();
    let bytes = bincode::serialize(&proof).expect("serialize");
    let path = fixture_path("fixures/v2_plain_proof.bin");
    std::fs::write(&path, &bytes).expect("write fixture");
    println!("plain proof fixture written to {path:?} ({} bytes)", bytes.len());
}

/// Mining + serialization determinism: regenerating the plain proof must
/// reproduce the committed fixture bytes exactly.
#[test]
fn plain_fixture_is_reproducible() {
    let committed = read_fixture("fixures/v2_plain_proof.bin");
    let regenerated = bincode::serialize(&mine_fixture_plain_proof()).expect("serialize");
    assert!(
        regenerated == committed,
        "seeded mining no longer reproduces fixures/v2_plain_proof.bin \
         (lengths: regenerated {} vs committed {})",
        regenerated.len(),
        committed.len()
    );
}

/// v2 plain_verify: committed bytes -> accept; corrupted bytes -> reject.
#[test]
fn v2_plain_verify_bytes_to_bool() {
    let bytes = read_fixture("fixures/v2_plain_proof.bin");

    let proof = PlainProof::deserialize_compat(&bytes).expect("fixture must deserialize");
    zk_pow::v2::api::verify::verify_plain_proof(&v2_header(), &proof, None, SeedDerivation::Legacy)
        .expect("committed v2 plain proof must be accepted");

    // Reject: flip one byte of committed matrix data near the end of the blob.
    let mut bad = bytes.clone();
    let idx = bad.len() - 100;
    bad[idx] ^= 0x01;
    let rejected = match PlainProof::deserialize_compat(&bad) {
        Err(_) => true,
        Ok(p) => zk_pow::v2::api::verify::verify_plain_proof(&v2_header(), &p, None, SeedDerivation::Legacy).is_err(),
    };
    assert!(rejected, "corrupted plain proof must be rejected");
}

/// v1 plain_verify: the same non-MoE plain proof through the frozen v1 stack.
#[test]
fn v1_plain_verify_bytes_to_bool() {
    let bytes = read_fixture("fixures/v2_plain_proof.bin");

    let proof = PlainProof::deserialize_compat(&bytes).expect("fixture must deserialize");
    zk_pow::v1::api::verify::verify_plain_proof(&v1_header(), &proof, None)
        .expect("committed plain proof must be accepted by the v1 verifier");

    let mut bad = bytes.clone();
    let idx = bad.len() - 100;
    bad[idx] ^= 0x01;
    let rejected = match PlainProof::deserialize_compat(&bad) {
        Err(_) => true,
        Ok(p) => zk_pow::v1::api::verify::verify_plain_proof(&v1_header(), &p, None).is_err(),
    };
    assert!(rejected, "corrupted plain proof must be rejected by the v1 verifier");
}

/// v2 zk_prove, bytes -> bytes: prove the committed plain proof and check the
/// output against the committed ZK fixture.
///
/// Full byte-equality is impossible, deliberately so: the final recursion
/// layer is built with `zero_knowledge: true`, so on every prove plonky2
/// blinds its wire/partial-product/quotient commitments with fresh random
/// salts (that hiding of the miner's private witness is the "zk" in zk-pow).
/// Every transcript-derived byte after the first blinded commitment — zeta
/// included — legitimately differs between runs (empirically the divergence
/// boundary is byte 170). FRI proof-of-work grinding (a parallel
/// `find_map_any`) adds further scheduling nondeterminism on top. Making the
/// bytes deterministic would mean deriving the blinding from a seeded PRF
/// inside the plonky2 prover — a prover-internals change out of scope here.
///
/// What IS deterministic — and asserted — is the
/// `public_data | pow_bits | rate_bits` prefix and the total proof length.
/// The produced proof must also round-trip through the byte-level verifier.
#[test]
fn v2_zk_prove_bytes_to_bytes() {
    use zk_pow::v2::api::proof::{PublicProofParams, ZKProof};
    use zk_pow::v2::circuit::circuit_utils::CircuitCache;

    let plain_bytes = read_fixture("fixures/v2_plain_proof.bin");
    let plain = PlainProof::deserialize_compat(&plain_bytes).expect("fixture must deserialize");

    let mut cache = CircuitCache::default();
    let result = zk_pow::v2::api::prove::zk_prove_plain_proof(v2_header(), &plain, &mut cache, false, SeedDerivation::Legacy)
        .expect("proving the fixture plain proof must succeed");

    let mut produced = result.public_data.clone();
    produced.extend_from_slice(&result.proof_data);

    let expected = read_fixture("fixures/v2_stark_proof.bin");
    assert_eq!(produced.len(), expected.len(), "proof byte length changed");

    // Deterministic prefix: public_data (WIRE_SIZE) + pow_bits(3) + rate_bits(3).
    let prefix_len = PublicProofParams::WIRE_SIZE + 3 + 3;
    assert!(
        produced[..prefix_len] == expected[..prefix_len],
        "deterministic proof prefix (public data / FRI config) no longer matches the fixture"
    );

    // The produced bytes must verify through the byte-level verifier.
    let (public_data, proof_data) = produced.split_at(PublicProofParams::WIRE_SIZE);
    let (params, proof) =
        ZKProof::deserialize(v2_header(), SeedDerivation::Legacy, public_data, proof_data).expect("deserialize");
    zk_pow::v2::api::verify::verify_block(&params, &proof, &mut cache).expect("freshly produced v2 ZK proof must verify");
}

/// v2 zk_verify: committed proof bytes -> accept; corrupted bytes -> reject.
#[test]
fn v2_zk_verify_bytes_to_bool() {
    use zk_pow::v2::api::proof::{PublicProofParams, ZKProof};
    use zk_pow::v2::circuit::circuit_utils::CircuitCache;

    let buffer = read_fixture("fixures/v2_stark_proof.bin");
    let (public_data, proof_data) = buffer.split_at(PublicProofParams::WIRE_SIZE);

    let (params, proof) =
        ZKProof::deserialize(v2_header(), SeedDerivation::Legacy, public_data, proof_data).expect("deserialize");
    let mut cache = CircuitCache::default();
    zk_pow::v2::api::verify::verify_block(&params, &proof, &mut cache).expect("committed v2 ZK proof must verify");

    // Reject: corrupt one byte of hash_a inside public_data (offset 52..84).
    let mut bad = buffer.clone();
    bad[60] ^= 0x01;
    let (bad_public, bad_proof) = bad.split_at(PublicProofParams::WIRE_SIZE);
    let (bad_params, bad_zk) =
        ZKProof::deserialize(v2_header(), SeedDerivation::Legacy, bad_public, bad_proof).expect("parse still fine");
    assert!(
        zk_pow::v2::api::verify::verify_block(&bad_params, &bad_zk, &mut cache).is_err(),
        "v2 ZK proof with corrupted public data must be rejected"
    );
}

/// v1 zk_verify through `verify_v1`, the production bytes-level entry point:
/// committed proof bytes -> accept; corrupted bytes -> reject.
#[test]
fn v1_zk_verify_bytes_to_bool() {
    use zk_pow::v1::api::proof::{PublicProofParams, ZKProof};
    use zk_pow::v1::circuit::circuit_utils::CircuitCache;

    let buffer = read_fixture("src/v1/fixures/stark_proof.bin");
    let (public_data, proof_data) = buffer.split_at(PublicProofParams::PUBLICDATA_SIZE);

    let header = v1_header();
    let header_bytes = header.to_bytes();

    // Populate the circuit cache (verify_v1 requires pre-compiled circuits,
    // as in production where it runs against the embedded cache).
    let (params, proof) = ZKProof::deserialize(header, public_data.try_into().unwrap(), proof_data).expect("deserialize");
    let mut cache = CircuitCache::default();
    zk_pow::v1::api::verify::verify_block(&params, &proof, &mut cache).expect("committed v1 ZK proof must verify");

    // Bytes -> accept through the production entry point.
    zk_pow::v1::verify_v1(&header_bytes, public_data, proof_data, &cache, None)
        .expect("verify_v1 must accept the committed fixture bytes");

    // Reject: corrupt one byte of hash_a inside public_data (offset 52..84).
    let mut bad_public = public_data.to_vec();
    bad_public[60] ^= 0x01;
    assert!(
        zk_pow::v1::verify_v1(&header_bytes, &bad_public, proof_data, &cache, None).is_err(),
        "v1 ZK proof with corrupted public data must be rejected"
    );
}

/// v1 zk_prove, bytes -> bytes: prove the committed plain proof through the
/// frozen v1 prover and check the output against the committed v1 ZK fixture.
/// `src/v1/fixures/stark_proof.bin` was master-generated from this same seeded
/// plain proof (its hash_a/hash_b bytes match the v2 fixture's), so the
/// deterministic `public_data | pow_bits | rate_bits` prefix and the total
/// length must reproduce exactly; the remaining bytes are ZK-blinded on every
/// run (see `v2_zk_prove_bytes_to_bytes` for the full explanation).
#[test]
fn v1_zk_prove_bytes_to_bytes() {
    use zk_pow::v1::api::proof::{PublicProofParams, ZKProof};
    use zk_pow::v1::circuit::circuit_utils::CircuitCache;

    let plain_bytes = read_fixture("fixures/v2_plain_proof.bin");
    let plain = PlainProof::deserialize_compat(&plain_bytes).expect("fixture must deserialize");

    let mut cache = CircuitCache::default();
    let result = zk_pow::v1::api::prove::zk_prove_plain_proof(v1_header(), &plain, &mut cache, false)
        .expect("proving the fixture plain proof through the frozen v1 prover must succeed");

    let mut produced = result.public_data.to_vec();
    produced.extend_from_slice(&result.proof_data);

    let expected = read_fixture("src/v1/fixures/stark_proof.bin");
    assert_eq!(produced.len(), expected.len(), "v1 proof byte length changed");

    // Deterministic prefix: public_data (PUBLICDATA_SIZE) + pow_bits(3) + rate_bits(3).
    let prefix_len = PublicProofParams::PUBLICDATA_SIZE + 3 + 3;
    assert!(
        produced[..prefix_len] == expected[..prefix_len],
        "deterministic v1 proof prefix (public data / FRI config) no longer matches the fixture"
    );

    // The produced bytes must verify through the frozen v1 verifier.
    let (public_data, proof_data) = produced.split_at(PublicProofParams::PUBLICDATA_SIZE);
    let (params, proof) = ZKProof::deserialize(v1_header(), public_data.try_into().unwrap(), proof_data).expect("deserialize");
    zk_pow::v1::api::verify::verify_block(&params, &proof, &mut cache).expect("freshly produced v1 ZK proof must verify");
}

// =============================================================================
// MoE (cert-v2 `ZkMoe`) fixtures
// =============================================================================

/// MoE mining/proving parameters. Must stay in sync with `moe_params()` in the
/// v2 compat tests: the committed `fixures/v2_stark_proof_moe.bin` was produced
/// from exactly this seeded mining run.
const MOE_MINING_SEED: u64 = 0xcafe_babe;
const MOE_NBITS: u32 = 0x207FFFFF;
const MOE_M: usize = 1024;
const MOE_N: usize = 128;
const MOE_K: usize = 1024;
const MOE_RANK: u16 = 32;

fn moe_v2_header() -> zk_pow::v2::api::proof::IncompleteBlockHeader {
    zk_pow::v2::api::proof::IncompleteBlockHeader {
        version: 0,
        prev_block: [0; 32],
        merkle_root: *b"0123456789abcdef0123456789abcdef",
        timestamp: 0x66666666,
        nbits: MOE_NBITS,
    }
}

/// Same header through the frozen v1 types.
fn moe_v1_header() -> zk_pow::v1::api::proof::IncompleteBlockHeader {
    zk_pow::v1::api::proof::IncompleteBlockHeader {
        version: 0,
        prev_block: [0; 32],
        merkle_root: *b"0123456789abcdef0123456789abcdef",
        timestamp: 0x66666666,
        nbits: MOE_NBITS,
    }
}

fn moe_mining_config() -> zk_pow::v2::api::proof::MiningConfiguration {
    use zk_pow::v2::api::proof::{MMAType, MiningConfiguration, MoEConfig, PeriodicPattern};
    MiningConfiguration {
        common_dim: MOE_K as u32,
        rank: MOE_RANK,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
        moe: Some(MoEConfig { e: 4, top_k: 1 }),
    }
}

/// Mines the canonical MoE fixture plain proof (deterministic: seeded rng,
/// looping until the first solution like the compat-test fixture generator).
fn mine_moe_fixture_plain_proof() -> PlainProof {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(MOE_MINING_SEED);
    loop {
        let attempt = zk_pow::v2::mine::try_mine_one_moe(
            &mut rng,
            MOE_M,
            MOE_N,
            MOE_K,
            moe_v2_header(),
            moe_mining_config(),
            None,
            false,
            SeedDerivation::Legacy,
        )
        .expect("MoE mining failed");
        if let Some(p) = attempt {
            return p;
        }
    }
}

/// Writes `fixures/v2_plain_proof_moe.bin`. Run once and commit the output.
#[test]
#[ignore] // Run with: cargo test --release --test e2e_fixture_test -- --ignored generate_moe_plain_fixture
fn generate_moe_plain_fixture() {
    let proof = mine_moe_fixture_plain_proof();
    let bytes = bincode::serialize(&proof).expect("serialize");
    let path = fixture_path("fixures/v2_plain_proof_moe.bin");
    std::fs::write(&path, &bytes).expect("write fixture");
    println!("MoE plain proof fixture written to {path:?} ({} bytes)", bytes.len());
}

/// MoE mining + serialization determinism: regenerating must reproduce the
/// committed fixture bytes exactly.
#[test]
fn moe_plain_fixture_is_reproducible() {
    let committed = read_fixture("fixures/v2_plain_proof_moe.bin");
    let regenerated = bincode::serialize(&mine_moe_fixture_plain_proof()).expect("serialize");
    assert!(
        regenerated == committed,
        "seeded MoE mining no longer reproduces fixures/v2_plain_proof_moe.bin \
         (lengths: regenerated {} vs committed {})",
        regenerated.len(),
        committed.len()
    );
}

/// v2 plain_verify on an MoE proof: committed bytes -> accept; corrupted -> reject.
#[test]
fn v2_plain_verify_moe_bytes_to_bool() {
    let bytes = read_fixture("fixures/v2_plain_proof_moe.bin");

    let proof = PlainProof::deserialize_compat(&bytes).expect("fixture must deserialize");
    assert!(proof.moe.is_some(), "fixture must be an MoE proof");
    zk_pow::v2::api::verify::verify_plain_proof(&moe_v2_header(), &proof, None, SeedDerivation::Legacy)
        .expect("committed MoE plain proof must be accepted by the v2 verifier");

    let mut bad = bytes.clone();
    let idx = bad.len() - 100;
    bad[idx] ^= 0x01;
    let rejected = match PlainProof::deserialize_compat(&bad) {
        Err(_) => true,
        Ok(p) => zk_pow::v2::api::verify::verify_plain_proof(&moe_v2_header(), &p, None, SeedDerivation::Legacy).is_err(),
    };
    assert!(rejected, "corrupted MoE plain proof must be rejected");
}

/// The frozen v1 stack predates MoE and must reject MoE proofs outright: its
/// parser ignores the routing commitment, so the expert-routed `A` matrix can
/// never match what v1 regenerates from the header seed. (In production the
/// cert-version gate in `ffi::plain_proof` already rejects the v1/MoE
/// crossover before verification; this pins the defense-in-depth behavior of
/// the verifier itself.)
#[test]
fn v1_plain_verify_rejects_moe() {
    let bytes = read_fixture("fixures/v2_plain_proof_moe.bin");
    let proof = PlainProof::deserialize_compat(&bytes).expect("fixture must deserialize");
    assert!(
        zk_pow::v1::api::verify::verify_plain_proof(&moe_v1_header(), &proof, None).is_err(),
        "the v1 verifier must reject MoE proofs"
    );
}

/// v2 zk_verify on the committed (master-generated) MoE ZK proof: bytes ->
/// accept; corrupted public data -> reject. Layout per the compat-test
/// generator: `public_data_len(4 LE) | public_data | proof_data`.
#[test]
fn v2_zk_verify_moe_bytes_to_bool() {
    use zk_pow::v2::api::proof::ZKProof;
    use zk_pow::v2::circuit::circuit_utils::CircuitCache;

    let buffer = read_fixture("fixures/v2_stark_proof_moe.bin");
    let public_data_len = u32::from_le_bytes(buffer[..4].try_into().unwrap()) as usize;
    let public_data = &buffer[4..4 + public_data_len];
    let proof_data = &buffer[4 + public_data_len..];

    let (params, proof) =
        ZKProof::deserialize(moe_v2_header(), SeedDerivation::Legacy, public_data, proof_data).expect("deserialize");
    assert!(params.moe.is_some(), "MoE fixture must carry moe params");
    let mut cache = CircuitCache::default();
    zk_pow::v2::api::verify::verify_block(&params, &proof, &mut cache).expect("committed MoE ZK proof must verify");

    // Reject: corrupt one byte of hash_a inside public_data (offset 52..84,
    // +4 for the length prefix).
    let mut bad = buffer.clone();
    bad[4 + 60] ^= 0x01;
    let bad_public = &bad[4..4 + public_data_len];
    let bad_proof = &bad[4 + public_data_len..];
    let (bad_params, bad_zk) =
        ZKProof::deserialize(moe_v2_header(), SeedDerivation::Legacy, bad_public, bad_proof).expect("parse still fine");
    assert!(
        zk_pow::v2::api::verify::verify_block(&bad_params, &bad_zk, &mut cache).is_err(),
        "MoE ZK proof with corrupted public data must be rejected"
    );
}

/// v2 zk_prove on the MoE plain proof, bytes -> bytes, against the
/// master-generated MoE ZK fixture. Same determinism boundary as
/// `v2_zk_prove_bytes_to_bytes`; MoE public data is variable-length, so the
/// produced bytes are compared in the fixture's
/// `public_data_len(4 LE) | public_data | proof_data` layout.
#[test]
fn v2_zk_prove_moe_bytes_to_bytes() {
    use zk_pow::v2::api::proof::ZKProof;
    use zk_pow::v2::circuit::circuit_utils::CircuitCache;

    let plain_bytes = read_fixture("fixures/v2_plain_proof_moe.bin");
    let plain = PlainProof::deserialize_compat(&plain_bytes).expect("fixture must deserialize");
    assert!(plain.moe.is_some(), "fixture must be an MoE proof");

    let mut cache = CircuitCache::default();
    let result = zk_pow::v2::api::prove::zk_prove_plain_proof(moe_v2_header(), &plain, &mut cache, false, SeedDerivation::Legacy)
        .expect("proving the MoE fixture plain proof must succeed");

    let mut produced = (result.public_data.len() as u32).to_le_bytes().to_vec();
    produced.extend_from_slice(&result.public_data);
    produced.extend_from_slice(&result.proof_data);

    let expected = read_fixture("fixures/v2_stark_proof_moe.bin");
    let expected_public_len = u32::from_le_bytes(expected[..4].try_into().unwrap()) as usize;
    assert_eq!(
        result.public_data.len(),
        expected_public_len,
        "MoE public data length changed"
    );
    assert_eq!(produced.len(), expected.len(), "MoE proof byte length changed");

    // Deterministic prefix: length prefix (4) + public_data + pow_bits(3) + rate_bits(3).
    let prefix_len = 4 + expected_public_len + 3 + 3;
    assert!(
        produced[..prefix_len] == expected[..prefix_len],
        "deterministic MoE proof prefix (public data / FRI config) no longer matches the fixture"
    );

    // The produced bytes must verify through the byte-level verifier.
    let (params, proof) = ZKProof::deserialize(
        moe_v2_header(),
        SeedDerivation::Legacy,
        &produced[4..4 + expected_public_len],
        &produced[4 + expected_public_len..],
    )
    .expect("deserialize");
    zk_pow::v2::api::verify::verify_block(&params, &proof, &mut cache).expect("freshly produced MoE ZK proof must verify");
}
