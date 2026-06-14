//! MoE (Mixture of Experts) proving and verification tests.
//!
//! Tests cover:
//! - Correct end-to-end prove + verify for MoE
//! - Failure: public outer_indices mismatch (CTL / verifier rejects)
//! - Failure: corrupted routing data (STARK constraint failure)
//! - Edge cases: field tamper, weight_col_offset, roundtrip

use rand_chacha::rand_core::SeedableRng;

use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, MoEConfig, PeriodicPattern, ZKProof};
use zk_pow::api::{prove, verify};
use zk_pow::circuit::chip::blake3::program::{BLOCK_LEN, routing_blake_hotspot_rows};
use zk_pow::circuit::circuit_utils::CircuitCache;
use zk_pow::ffi::mine::try_mine_one_moe;
use zk_pow::ffi::plain_proof::PlainProof;

struct MoETestParams {
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    m: usize,
    n: usize,
    k: usize,
}

/// Parameters aligned with test_python_api baseline (rank=32, 4x8 tile).
/// Dimensions kept small for CI speed; difficulty is permissive.
fn moe_params() -> MoETestParams {
    let rank = 32u16;
    let k = 1024;
    MoETestParams {
        header: IncompleteBlockHeader {
            version: 0,
            prev_block: [0; 32],
            merkle_root: *b"0123456789abcdef0123456789abcdef",
            timestamp: 0x66666666,
            nbits: 0x207FFFFF,
        },
        config: MiningConfiguration {
            common_dim: k as u32,
            rank,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
            moe: Some(MoEConfig { e: 4, top_k: 1 }),
        },
        m: 1024,
        n: 128,
        k,
    }
}

fn mine_moe_proof(p: &MoETestParams, seed: u64) -> PlainProof {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    loop {
        let proof = try_mine_one_moe(&mut rng, p.m, p.n, p.k, p.header, p.config, None, false).unwrap();
        if let Some(proof) = proof {
            return proof;
        }
    }
}

/// Prove an MoE proof end-to-end and return the serialized result.
fn prove_moe(p: &MoETestParams, moe_proof: &PlainProof) -> prove::ProveResult {
    let mut cache = CircuitCache::default();
    prove::zk_prove_plain_proof(p.header, moe_proof, &mut cache, true).expect("MoE proving failed")
}

// =============================================================================
// 1. Correct end-to-end MoE prove + verify
// =============================================================================

#[test]
fn test_moe_prove_verify() {
    let p = moe_params();
    let moe_proof = mine_moe_proof(&p, 0xdeadbeef);
    let result = prove_moe(&p, &moe_proof);

    let (public_params, zk_proof) = ZKProof::deserialize(p.header, &result.public_data, &result.proof_data).unwrap();

    let mut cache = CircuitCache::default();
    verify::verify_block(&public_params, &zk_proof, &mut cache).expect("MoE proof must verify");
}

// =============================================================================
// 1b. Routing whose entry count is not a multiple of 16 (byte length not a
//     multiple of 64). `m * top_k = 1000` real entries get padded up to 1008
//     (16-aligned) zero entries before the 1024-chunk padding. This exercises
//     the routing-entry padding deduced from `m`/`top_k`, the Blake hash
//     reconstruction over the padded layout, and the padding-bounds checks in
//     both the plaintext parser and the zk sanity check.
// =============================================================================

/// Routing params with `m * top_k = 1000` real entries (not 16-aligned).
fn moe_params_routing_not_multiple_of_64() -> MoETestParams {
    let mut p = moe_params();
    p.m = 1000; // 1000 * top_k(1) = 1000 entries → 4000 bytes, not a multiple of 64
    assert_ne!(
        (p.m * p.config.moe.unwrap().top_k as usize) % 16,
        0,
        "test must use a routing entry count that is not 16-aligned"
    );
    p
}

/// Fast (parse-only) coverage: exercises evaluate_blake's hash reconstruction over
/// the padded routing layout, the plaintext padding-bounds check, and the zk sanity
/// check — all without the expensive prove step.
#[test]
fn test_moe_routing_not_multiple_of_64_parse() {
    let p = moe_params_routing_not_multiple_of_64();
    let moe_proof = mine_moe_proof(&p, 0x51515151);

    let (private, public) = moe_proof.parse_proof(p.header).unwrap();
    public.sanity_check().unwrap();
    public.sanity_check_private_params(&private).unwrap();
}

/// Full end-to-end prove + verify with non-64-aligned routing. Slow, so `#[ignore]`d like the other e2e variants below.
#[test]
#[ignore]
fn test_moe_routing_not_multiple_of_64_prove_verify() {
    let p = moe_params_routing_not_multiple_of_64();
    let moe_proof = mine_moe_proof(&p, 0x51515151);
    let result = prove_moe(&p, &moe_proof);

    let (public_params, zk_proof) = ZKProof::deserialize(p.header, &result.public_data, &result.proof_data).unwrap();
    let mut cache = CircuitCache::default();
    verify::verify_block(&public_params, &zk_proof, &mut cache).expect("non-64-aligned routing must verify");
}

// =============================================================================
// 2. Failure: public outer_indices do not match the routing
//    The public params carry outer_indices that disagree with the values baked
//    into the STARK proof via the outer-indices CTL.
// =============================================================================

#[test]
fn test_moe_wrong_public_outer_indices_fails_verification() {
    let p = moe_params();
    let moe_proof = mine_moe_proof(&p, 0xcafebabe);
    let result = prove_moe(&p, &moe_proof);

    let (mut public_params, _) = ZKProof::deserialize(p.header, &result.public_data, &result.proof_data).unwrap();

    let moe = public_params.moe.as_mut().unwrap();
    for idx in moe.outer_indices.iter_mut() {
        *idx += 1;
    }

    let tampered_public_data = public_params.to_wire_bytes().unwrap();
    let (tampered_params, zk_proof) = ZKProof::deserialize(p.header, &tampered_public_data, &result.proof_data).unwrap();

    let mut cache = CircuitCache::default();
    let err = verify::verify_block(&tampered_params, &zk_proof, &mut cache).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Preprocessed digest mismatch") || msg.contains("Proof Invalid"),
        "Expected preprocessed digest or proof mismatch, got: {msg}"
    );
}

// =============================================================================
// 3. Failure: corrupted routing Merkle root
//    parse_moe_proof recomputes hash_routing from the Merkle proof and compares
//    it to the committed root — a corrupt root is caught immediately.
// =============================================================================

#[test]
#[should_panic(expected = "hash_routing mismatch between Blake evaluation and public commitment")]
fn test_moe_corrupted_routing_root_fails_parse() {
    let p = moe_params();
    let mut moe_proof = mine_moe_proof(&p, 0xfeedface);

    let mut moe = moe_proof.moe.unwrap();
    moe.routing_proof.root[0] ^= 0xFF;

    moe_proof.moe = Some(moe);

    moe_proof.parse_proof(p.header).unwrap();
}

// =============================================================================
// 4. Failure: routing entries don't match outer_indices
//    Perturb a single a.row_index so the routing entry check fails, while
//    keeping the indices sorted to avoid tripping the sort assertion earlier.
// =============================================================================

#[test]
#[should_panic(expected = "Failed to extract strip")]
fn test_moe_routing_outer_index_mismatch_fails_parse() {
    let p = moe_params();
    let mut moe_proof = mine_moe_proof(&p, 0xbaadf00d);

    let len = moe_proof.a.row_indices.len();
    assert!(len >= 1, "Need at least 1 row index");
    moe_proof.a.row_indices[len - 1] += 1;

    moe_proof.parse_proof(p.header).unwrap();
}

// =============================================================================
// 5. MoE parse roundtrip: verify all fields survive mine → parse correctly
// =============================================================================

#[test]
fn test_moe_roundtrip_parse_only() {
    let p = moe_params();
    let moe_proof = mine_moe_proof(&p, 0xdeadbeef);

    let (private, public) = moe_proof.parse_proof(p.header).unwrap();

    let moe = public.moe.as_ref().unwrap();
    assert_eq!(moe.outer_indices.len(), moe_proof.a.row_indices.len());
    assert_eq!(
        moe.outer_indices,
        moe_proof.a.row_indices.iter().map(|&x| x as u32).collect::<Vec<u32>>()
    );
    let moe_config = public.mining_config.moe.as_ref().unwrap();
    assert_eq!(public.total_b_cols(), p.n * moe_config.e as usize);

    assert_eq!(private.s_a.len(), public.h());
    assert_eq!(private.s_b.len(), public.w());
    for strip in &private.s_a {
        assert_eq!(strip.len(), public.dot_product_length());
    }
    assert!(!private.s_routing.is_empty(), "MoE proofs must have routing strips");

    public.sanity_check().unwrap();
    public.sanity_check_private_params(&private).unwrap();
}

// =============================================================================
// 6. Edge case: tampered hash_routing in public params after proving
//    commitment_hash depends on hash_routing, so the verifier rejects.
// =============================================================================

#[test]
fn test_moe_tampered_hash_routing_fails() {
    let p = moe_params();
    let moe_proof = mine_moe_proof(&p, 0xabcdef01);
    let result = prove_moe(&p, &moe_proof);

    let (mut public_params, _) = ZKProof::deserialize(p.header, &result.public_data, &result.proof_data).unwrap();

    let moe = public_params.moe.as_mut().unwrap();
    moe.hash_routing[0] ^= 0xFF;

    let tampered_data = public_params.to_wire_bytes().unwrap();
    let (tampered_params, zk_proof) = ZKProof::deserialize(p.header, &tampered_data, &result.proof_data).unwrap();

    let mut cache = CircuitCache::default();
    let err = verify::verify_block(&tampered_params, &zk_proof, &mut cache).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Preprocessed digest mismatch") || msg.contains("Proof Invalid"),
        "Expected preprocessed digest or proof mismatch, got: {msg}"
    );
}

// =============================================================================
// 7. Edge case: wrong routing_start_offset
//    Shifting the offset means the routing entries are read from the wrong
//    position, so the routing[expert][inner] != outer_indices check fails.
// =============================================================================

#[test]
#[should_panic(expected = "routing mismatch")]
fn test_moe_wrong_routing_start_offset_fails_parse() {
    let p = moe_params();
    let mut moe_proof = mine_moe_proof(&p, 0x12345678);

    while moe_proof.moe.clone().unwrap().expert_idx == 0 {
        // We need an offset to shift, so if the mined proof is for expert 0, keep mining until we get a non-zero expert.
        moe_proof = mine_moe_proof(&p, 0x12345678);
    }
    let mut moe = moe_proof.moe.unwrap();
    moe.routing_end_offsets[moe.expert_idx as usize - 1] += 1;
    moe_proof.moe = Some(moe);

    moe_proof.parse_proof(p.header).unwrap();
}

// =============================================================================
// 8. Unit tests for routing_blake_hotspot_rows and the dword-boundary logic
//    that maps (hotspot, idx_in_strip) back to (is_first, is_second, outer_index).
//
//    These are cheap, pure-logic tests that exercise the exact edge cases
//    around routing_start_offset alignment without running a full prove+verify.
// =============================================================================

/// Replicate the preprocessing index logic from pearl_preprocess.rs so we can
/// test it in isolation against various offsets and inner_indices.
fn compute_outer_flags(hotspot: u32, idx_in_strip: usize, routing_start_offset: u32, inner_indices: &[u32]) -> (bool, bool, u64) {
    let abs_u32_idx = ((hotspot as usize * BLOCK_LEN + idx_in_strip) / std::mem::size_of::<u32>()) as u32;
    let (mut outer_index, mut is_first, mut is_second) = (0u64, false, false);

    if let Some(first_idx) = abs_u32_idx.checked_sub(routing_start_offset)
        && inner_indices.contains(&first_idx)
    {
        outer_index = inner_indices.iter().position(|&v| v == first_idx).unwrap() as u64;
        is_first = true;
    }
    if let Some(second_idx) = (abs_u32_idx + 1).checked_sub(routing_start_offset)
        && inner_indices.contains(&second_idx)
    {
        if !is_first {
            outer_index = inner_indices.iter().position(|&v| v == second_idx).unwrap() as u64;
        }
        is_second = true;
    }
    (is_first, is_second, outer_index)
}

#[test]
fn test_hotspot_dedup_and_sort() {
    // inner_indices [0, 1] with offset 0 → both map to hotspot 0. Dedup to [0].
    assert_eq!(routing_blake_hotspot_rows(0, &[0, 1]), vec![0]);

    // inner_indices [15, 16] with offset 0 → hotspot(45)=2, hotspot(48)=3 → [2,3]
    assert_eq!(routing_blake_hotspot_rows(0, &[45, 48]), vec![2, 3]);

    // With offset 4, indices [0,1] → absolute [4,5] → hotspot(4)=0, hotspot(5)=0 → [0]
    assert_eq!(routing_blake_hotspot_rows(4, &[0, 1]), vec![0]);
}

#[test]
fn test_offset_aligned_both_u32_in_range() {
    // offset = 0 (perfectly aligned), hotspot = 0, idx_in_strip = 0
    // abs_u32_idx = 0, first_idx = 0, second_idx = 1
    let inner = vec![0, 1, 10, 11];
    let (f, s, idx) = compute_outer_flags(0, 0, 0, &inner);
    assert!(f, "first u32 (index 0) is in inner_indices");
    assert!(s, "second u32 (index 1) is in inner_indices");
    assert_eq!(idx, 0, "outer_index should be position of first match");
}

#[test]
fn test_offset_aligned_neither_u32_matches() {
    // offset = 0, hotspot = 0, idx_in_strip = 8 → abs = 2
    // first_idx = 2, second_idx = 3 — neither in inner_indices
    let inner = vec![0, 1, 10, 11];
    let (f, s, _) = compute_outer_flags(0, 8, 0, &inner);
    assert!(!f);
    assert!(!s);
}

#[test]
fn test_offset_misaligned_first_below_offset() {
    // offset = 17, hotspot = hotspot for inner_index 0 = (17+0)*4/64 = 68/64 = 1
    // idx_in_strip = 0 → abs = 1*64/4 = 16 → first_idx = 16 - 17 → underflow!
    // second_idx = 17 - 17 = 0 → matches inner[0]
    let inner = vec![0, 5, 10, 15];
    let hotspot = routing_blake_hotspot_rows(17, &inner)[0];
    let (f, s, idx) = compute_outer_flags(hotspot, 0, 17, &inner);
    assert!(!f, "first u32 is before routing_start_offset");
    assert!(s, "second u32 at abs 17 = offset → routing index 0 is in inner");
    assert_eq!(idx, 0);
}

#[test]
fn test_offset_misaligned_both_below_offset() {
    // offset = 18, hotspot for inner 0 = (18+0)*4/64 = 72/64 = 1
    // idx_in_strip = 0 → abs = 16 → 16 < 18 → first underflow
    // abs+1 = 17 < 18 → second underflow
    let inner = vec![0, 5];
    let hotspot = routing_blake_hotspot_rows(18, &inner)[0];
    let (f, s, _) = compute_outer_flags(hotspot, 0, 18, &inner);
    assert!(!f, "first u32 below offset");
    assert!(!s, "second u32 also below offset");
}

#[test]
fn test_offset_misaligned_only_first_matches() {
    // offset = 16, hotspot for inner 0 = (16+0)*4/64 = 1
    // idx_in_strip = 0 → abs = 16 → first_idx = 0 ✓, second_idx = 1
    // inner = [0, 10] → 1 not in inner → only first
    let inner = vec![0, 10];
    let hotspot = routing_blake_hotspot_rows(16, &inner)[0];
    let (f, s, idx) = compute_outer_flags(hotspot, 0, 16, &inner);
    assert!(f, "first_idx = 0 matches inner[0]");
    assert!(!s, "second_idx = 1 not in inner");
    assert_eq!(idx, 0);
}

#[test]
fn test_offset_misaligned_only_second_matches() {
    // offset = 15, hotspot for inner 0 = (15+0)*4/64 = 0
    // idx_in_strip = 56 → abs = 0*16 + 56/4 = 14
    // first_idx = 14 - 15 → underflow, second_idx = 15 - 15 = 0 ✓
    let inner = vec![0, 8];
    let (f, s, idx) = compute_outer_flags(0, 56, 15, &inner);
    assert!(!f, "first u32 below offset");
    assert!(s, "second u32 = offset → routing index 0");
    assert_eq!(idx, 0);
}

#[test]
fn test_offset_boundary_exactly_at_offset() {
    // offset = 16, idx_in_strip = 0, hotspot = 1 → abs = 16 = offset → first_idx = 0
    let inner = vec![0];
    let (f, s, idx) = compute_outer_flags(1, 0, 16, &inner);
    assert!(f, "abs == offset → first_idx = 0 matches");
    assert!(!s, "second_idx = 1, not in inner");
    assert_eq!(idx, 0);
}

#[test]
fn test_consecutive_inner_indices_same_dword() {
    // Two consecutive inner indices that land in the same dword:
    // offset = 0, inner = [4, 5], hotspot for idx 4 = (0+4)*4/64 = 1
    // At hotspot 1, idx_in_strip = (4*4) % 64 = 16 → abs = 1*16 + 16/4 = 20
    // Hmm, let's just pick abs = 4: hotspot = 0, idx_in_strip = 16
    let inner = vec![4, 5, 20, 21];
    let (f, s, idx) = compute_outer_flags(0, 16, 0, &inner);
    assert!(f, "first_idx = 4 in inner");
    assert!(s, "second_idx = 5 in inner");
    assert_eq!(idx, 0, "outer_index = position of 4 = 0");
}

#[test]
fn test_large_offset_no_overflow() {
    // Large offset that still produces valid results.
    let offset: u32 = 50_000;
    let inner = vec![0, 100, 200, 300];
    let hotspots = routing_blake_hotspot_rows(offset, &inner);
    // Verify hotspots are sorted and deduplicated.
    for w in hotspots.windows(2) {
        assert!(w[0] <= w[1], "hotspots must be sorted");
    }
    // The first hotspot block may start before the offset.
    let first_hs = hotspots[0];
    let abs_start = first_hs as usize * 16; // u32 indices covered by first block
    if (abs_start as u32) < offset {
        // First u32 in block is below offset → first should underflow
        let (f, _, _) = compute_outer_flags(first_hs, 0, offset, &inner);
        assert!(!f, "first u32 of first block must be below offset");
    }
}

#[test]
fn test_sweep_offsets_1_through_32() {
    // Sweep routing_start_offset from 1..=32 with inner_indices = [0].
    // Ensures the boundary logic is correct for every possible alignment mod 16.
    for offset in 1u32..=32 {
        let inner = vec![0u32];
        let hotspots = routing_blake_hotspot_rows(offset, &inner);
        let hs = hotspots[0]; // hotspot containing inner index 0

        // The target abs u32 index for inner 0 is exactly `offset`.
        // Walk every dword (8-byte) position in the 64-byte block and check
        // that exactly one position produces the right flags.
        let mut found_first = false;
        let mut found_second = false;
        for byte_off in (0..BLOCK_LEN).step_by(8) {
            let (f, s, idx) = compute_outer_flags(hs, byte_off, offset, &inner);
            if f {
                assert!(!found_first, "offset {offset}: duplicate is_first");
                assert_eq!(idx, 0);
                found_first = true;
            }
            if s {
                assert!(!found_second, "offset {offset}: duplicate is_second");
                found_second = true;
            }
        }
        // inner index 0 must appear exactly once, either as first or second.
        assert!(
            found_first || found_second,
            "offset {offset}: inner index 0 was never flagged"
        );
    }
}

// =============================================================================
// 10. End-to-end MoE prove+verify with various expert counts and top_k values
//     that produce different routing_start_offset alignments.
//
//     m and k are kept at 1024 so the miner has enough search space to find a
//     valid jackpot. Only e (experts) and top_k are varied — the random routing
//     assignment gives each non-hot expert ~m*(1-HOT_BIAS)/(e-1) tokens, and
//     the prefix-sum offset for expert 1 = routing[0].len(), which varies with
//     e and the RNG seed, cycling through different mod-16 alignments.
//
//     These are slow (mine + prove + verify), so they are #[ignore]d.
// =============================================================================

fn moe_params_custom(e: usize, top_k: usize) -> MoETestParams {
    let rank = 32u16;
    let k = 1024;
    MoETestParams {
        header: IncompleteBlockHeader {
            version: 0,
            prev_block: [0; 32],
            merkle_root: *b"0123456789abcdef0123456789abcdef",
            timestamp: 0x66666666,
            nbits: 0x207FFFFF,
        },
        config: MiningConfiguration {
            common_dim: k as u32,
            rank,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
            moe: Some(MoEConfig {
                e: e as u16,
                top_k: top_k as u16,
            }),
        },
        m: 1024,
        n: 128,
        k,
    }
}

fn run_moe_prove_verify(p: &MoETestParams, seed: u64) {
    let moe_proof = mine_moe_proof(p, seed);

    let result = prove_moe(p, &moe_proof);
    let moe = moe_proof.moe.unwrap();
    let expert_idx = moe.expert_idx;
    let offset = moe.routing_end_offsets[moe.expert_idx as usize];

    eprintln!(
        "  expert_idx={expert_idx}, routing_start_offset={offset} (mod 16 = {})",
        offset % 16
    );

    let (public_params, zk_proof) = ZKProof::deserialize(p.header, &result.public_data, &result.proof_data).unwrap();
    let mut cache = CircuitCache::default();
    verify::verify_block(&public_params, &zk_proof, &mut cache).expect("MoE proof must verify");
}

/// 3 experts: expert 0 gets ~1024*0.3/2 ≈ 154 tokens → offset ≈ 154 (mod 16 ≈ 10).
/// Multiple seeds to hit different actual offsets.
#[test]
#[ignore]
fn test_moe_prove_verify_e3() {
    let p = moe_params_custom(3, 1);
    for seed in [0xA001, 0xA002, 0xA003, 0xA004] {
        eprintln!("seed=0x{seed:X}");
        run_moe_prove_verify(&p, seed);
    }
}

/// 5 experts: expert 0 gets ~1024*0.3/4 ≈ 77 tokens → offset ≈ 77 (mod 16 ≈ 13).
#[test]
#[ignore]
fn test_moe_prove_verify_e5() {
    let p = moe_params_custom(5, 1);
    for seed in [0xB001, 0xB002, 0xB003, 0xB004] {
        eprintln!("seed=0x{seed:X}");
        run_moe_prove_verify(&p, seed);
    }
}

/// 7 experts: expert 0 gets ~1024*0.3/6 ≈ 51 tokens → offset ≈ 51 (mod 16 ≈ 3).
#[test]
#[ignore]
fn test_moe_prove_verify_e7() {
    let p = moe_params_custom(7, 1);
    for seed in [0xC001, 0xC002, 0xC003, 0xC004] {
        eprintln!("seed=0x{seed:X}");
        run_moe_prove_verify(&p, seed);
    }
}

/// top_k=2 doubles routing entries per token → different alignment patterns.
#[test]
#[ignore]
fn test_moe_prove_verify_top_k2() {
    let p = moe_params_custom(4, 2);
    for seed in [0xD001, 0xD002, 0xD003, 0xD004] {
        eprintln!("seed=0x{seed:X}");
        run_moe_prove_verify(&p, seed);
    }
}

/// 8 experts with top_k=1: many possible offsets, good coverage.
#[test]
#[ignore]
fn test_moe_prove_verify_e8() {
    let p = moe_params_custom(8, 1);
    for seed in [0xE001, 0xE002, 0xE003, 0xE004] {
        eprintln!("seed=0x{seed:X}");
        run_moe_prove_verify(&p, seed);
    }
}

/// top_k=3 with 5 experts: each token is routed to 3 experts, making routing
/// much denser and more likely to produce odd-aligned offsets.
#[test]
#[ignore]
fn test_moe_prove_verify_e5_top_k3() {
    let p = moe_params_custom(5, 3);
    for seed in [0xF001, 0xF002, 0xF003, 0xF004] {
        eprintln!("seed=0x{seed:X}");
        run_moe_prove_verify(&p, seed);
    }
}

// =============================================================================
// 11. Failure: outer_index exceeds 26 bits → URANGE13 range check fails
//     An outer index ≥ 2^26 cannot be decomposed into two 13-bit limbs that
//     satisfy both the packing constraint and the URANGE13 lookup.  The STARK
//     prover's debug constraint check catches this.
// =============================================================================

#[test]
#[should_panic(expected = "Constraint failed")]
fn test_moe_outer_index_exceeding_26_bits_fails() {
    let p = moe_params();
    let moe_proof = mine_moe_proof(&p, 0xdeadbeef);
    let (private_params, mut public_params) = moe_proof.parse_proof(p.header).unwrap();

    // Set the last outer_index to 2^26, which exceeds the 26-bit range.
    // Each outer index is split into two 13-bit limbs that are range-checked
    // against the URANGE13 table. A 27-bit value cannot be faithfully
    // decomposed: either a limb exceeds 8191 (failing the lookup) or the
    // limbs are truncated (failing the packing constraint).
    let moe = public_params.moe.as_mut().unwrap();
    let last = moe.outer_indices.len() - 1;
    moe.outer_indices[last] = 1u32 << 26;

    let mut cache = CircuitCache::default();
    let proof = prove::prove_block(&mut public_params, private_params, &mut cache).unwrap();
    let verif_res = verify::verify_block(&public_params, &proof, &mut cache);
    if let Err(e) = verif_res {
        if e.to_string().contains("outer indices must be <= m") {
            panic!("Constraint failed")
        }
        panic!("Proof should fail with outer indices error");
    }
}
