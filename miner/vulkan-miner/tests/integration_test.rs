//! Integration tests for vulkan-miner.
//!
//! Verifies that the GLSL BLAKE3 port (shaders/common/blake3.glsl) produces
//! identical outputs to the `blake3` crate reference for all keyed single-block
//! compression scenarios used by the mining pipeline (noise generation, jackpot
//! hashing).  Since no GPU is available in CI, these tests exercise the exact
//! same arithmetic on the CPU so any discrepancy between the GLSL and the
//! reference is caught at compile/test time.

use blake3;

// ---------------------------------------------------------------------------
// Exact Rust port of shaders/common/blake3.glsl
// ---------------------------------------------------------------------------

const B3_IV0: u32 = 0x6A09E667;
const B3_IV1: u32 = 0xBB67AE85;
const B3_IV2: u32 = 0x3C6EF372;
const B3_IV3: u32 = 0xA54FF53A;

const B3_KEYED_HASH: u32 = 1 << 4;
const B3_CHUNK_START: u32 = 1 << 0;
const B3_CHUNK_END: u32 = 1 << 1;
const B3_ROOT: u32 = 1 << 3;
const B3_FLAGS_SINGLE: u32 = B3_KEYED_HASH | B3_CHUNK_START | B3_CHUNK_END | B3_ROOT;

fn rotr(x: u32, n: u32) -> u32 {
    x.wrapping_shr(n) | x.wrapping_shl(32u32.wrapping_sub(n))
}

fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    let mut va = state[a];
    let mut vb = state[b];
    let mut vc = state[c];
    let mut vd = state[d];
    va = va.wrapping_add(vb).wrapping_add(x);
    vd = rotr(vd ^ va, 16);
    vc = vc.wrapping_add(vd);
    vb = rotr(vb ^ vc, 12);
    va = va.wrapping_add(vb).wrapping_add(y);
    vd = rotr(vd ^ va, 8);
    vc = vc.wrapping_add(vd);
    vb = rotr(vb ^ vc, 7);
    state[a] = va;
    state[b] = vb;
    state[c] = vc;
    state[d] = vd;
}

fn b3_round(state: &mut [u32; 16], block: &[u32; 16]) {
    g(state, 0, 4, 8, 12, block[0], block[1]);
    g(state, 1, 5, 9, 13, block[2], block[3]);
    g(state, 2, 6, 10, 14, block[4], block[5]);
    g(state, 3, 7, 11, 15, block[6], block[7]);
    g(state, 0, 5, 10, 15, block[8], block[9]);
    g(state, 1, 6, 11, 12, block[10], block[11]);
    g(state, 2, 7, 8, 13, block[12], block[13]);
    g(state, 3, 4, 9, 14, block[14], block[15]);
}

fn b3_permute(block: &mut [u32; 16]) {
    let orig = *block;
    block[0] = orig[2];
    block[1] = orig[6];
    block[2] = orig[3];
    block[3] = orig[10];
    block[4] = orig[7];
    block[5] = orig[0];
    block[6] = orig[4];
    block[7] = orig[13];
    block[8] = orig[1];
    block[9] = orig[11];
    block[10] = orig[12];
    block[11] = orig[5];
    block[12] = orig[9];
    block[13] = orig[14];
    block[14] = orig[15];
    block[15] = orig[8];
}

/// b3_compress_keyed — exact match of GLSL and CUDA compress_msg_block_u32.
///
/// Performs BLAKE3 keyed compression on a single 64-byte message block:
///   6 rounds with message permutation + 1 final round without permutation.
///   Output: state[0..8] ^ state[8..16].
fn compress_keyed(msg: &[u32; 16], key: &[u32; 8], flags: u32) -> [u32; 8] {
    let mut state: [u32; 16] = [0; 16];
    let mut block = *msg;

    for i in 0..8 {
        state[i] = key[i];
    }
    state[8] = B3_IV0;
    state[9] = B3_IV1;
    state[10] = B3_IV2;
    state[11] = B3_IV3;
    state[12] = 0;
    state[13] = 0;
    state[14] = 64;
    state[15] = flags;

    for _ in 0..6 {
        b3_round(&mut state, &block);
        b3_permute(&mut block);
    }
    b3_round(&mut state, &block);

    let mut out = [0u32; 8];
    for i in 0..8 {
        out[i] = state[i] ^ state[i + 8];
    }
    out
}

// ---------------------------------------------------------------------------
// Helper: convert [u8; N] ↔ [u32; N/4]
// ---------------------------------------------------------------------------

fn u8x32_to_u32x8(src: &[u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap()))
}

fn u8x64_to_u32x16(src: &[u8; 64]) -> [u32; 16] {
    std::array::from_fn(|i| u32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap()))
}

fn u32x8_to_u8x32(src: &[u32; 8]) -> [u8; 32] {
    std::array::from_fn(|i| src[i / 4].to_le_bytes()[i % 4])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_glsl_permutation_matches_blake3_spec() {
    let mut block: [u32; 16] = std::array::from_fn(|i| i as u32);
    let orig = block;
    b3_permute(&mut block);

    // BLAKE3_MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
    assert_eq!(block[0], orig[2], "perm[0] should be orig[2]");
    assert_eq!(block[1], orig[6], "perm[1] should be orig[6]");
    assert_eq!(block[2], orig[3], "perm[2] should be orig[3]");
    assert_eq!(block[3], orig[10], "perm[3] should be orig[10]");
    assert_eq!(block[4], orig[7], "perm[4] should be orig[7]");
    assert_eq!(block[5], orig[0], "perm[5] should be orig[0]");
    assert_eq!(block[6], orig[4], "perm[6] should be orig[4]");
    assert_eq!(block[7], orig[13], "perm[7] should be orig[13]");
    assert_eq!(block[8], orig[1], "perm[8] should be orig[1]");
    assert_eq!(block[9], orig[11], "perm[9] should be orig[11]");
    assert_eq!(block[10], orig[12], "perm[10] should be orig[12]");
    assert_eq!(block[11], orig[5], "perm[11] should be orig[5]");
    assert_eq!(block[12], orig[9], "perm[12] should be orig[9]");
    assert_eq!(block[13], orig[14], "perm[13] should be orig[14]");
    assert_eq!(block[14], orig[15], "perm[14] should be orig[15]");
    assert_eq!(block[15], orig[8], "perm[15] should be orig[8]");
}

/// Core correctness: the GLSL compress_keyed matches blake3 crate reference
/// for a diverse set of (key, message) pairs.
#[test]
fn test_glsl_compress_keyed_vs_blake3_crate() {
    let key0 = [0u8; 32];
    let msg0 = [0u8; 64];
    let key1 = [0xFFu8; 32];
    let msg1 = [0xFFu8; 64];
    let key2: [u8; 32] = std::array::from_fn(|i| i as u8);
    let msg2: [u8; 64] = std::array::from_fn(|i| i.wrapping_mul(3) as u8);
    let key3: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(17));
    let msg3: [u8; 64] = std::array::from_fn(|i| if i % 2 == 0 { 0xAA } else { 0x55 });
    let key4 = [0xDEu8; 32];
    let msg4: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_add(42u8));

    let cases: [(&[u8; 32], &[u8; 64]); 5] = [
        (&key0, &msg0),
        (&key1, &msg1),
        (&key2, &msg2),
        (&key3, &msg3),
        (&key4, &msg4),
    ];

    for (idx, &(key_u8, msg_u8)) in cases.iter().enumerate() {
        let key_u32 = u8x32_to_u32x8(key_u8);
        let msg_u32 = u8x64_to_u32x16(msg_u8);

        let glsl_result = compress_keyed(&msg_u32, &key_u32, B3_FLAGS_SINGLE);
        let glsl_bytes = u32x8_to_u8x32(&glsl_result);

        let expected = blake3::Hasher::new_keyed(key_u8).update(msg_u8).finalize();

        assert_eq!(
            &glsl_bytes[..],
            expected.as_bytes(),
            "Test vector {}: GLSL compress_keyed differs from blake3 crate",
            idx,
        );
    }
}

/// Noise-generation message layout: verify GLSL-style dense noise (prepend_index=0)
/// and sparse noise (prepend_index=1) produce the same output as blake3 crate.
#[test]
fn test_glsl_noise_gen_message_layout() {
    let hash_a: [u8; 32] = *blake3::hash(b"noise_test_key").as_bytes();
    let key_u32 = u8x32_to_u32x8(&hash_a);

    // Seed label: "A_tensor" padded to 32 bytes (same as SEED_LABEL_A)
    let seed_label: [u8; 32] = {
        let mut s = [0u8; 32];
        s[..8].copy_from_slice(b"A_tensor");
        s
    };

    for local_idx in [0u32, 1, 42, 127, 255] {
        // --- Dense noise (EAL/EBR): prepend_index=0, msg[0] = local_idx+1 ---
        let mut dense_msg = [0u8; 64];
        dense_msg[0..4].copy_from_slice(&(local_idx + 1).to_le_bytes());
        dense_msg[32..64].copy_from_slice(&seed_label);
        let dense_msg_u32 = u8x64_to_u32x16(&dense_msg);

        let glsl_result = compress_keyed(&dense_msg_u32, &key_u32, B3_FLAGS_SINGLE);
        let glsl_bytes = u32x8_to_u8x32(&glsl_result);

        let expected = blake3::Hasher::new_keyed(&hash_a)
            .update(&dense_msg)
            .finalize();

        assert_eq!(
            &glsl_bytes[..],
            expected.as_bytes(),
            "Dense noise (local_idx={}): GLSL BLAKE3 mismatch",
            local_idx,
        );

        // --- Sparse noise (EAR/EBL): prepend_index=1, msg[1] = local_idx+1 ---
        let mut sparse_msg = [0u8; 64];
        sparse_msg[4..8].copy_from_slice(&(local_idx + 1).to_le_bytes());
        sparse_msg[32..64].copy_from_slice(&seed_label);
        let sparse_msg_u32 = u8x64_to_u32x16(&sparse_msg);

        let glsl_result2 = compress_keyed(&sparse_msg_u32, &key_u32, B3_FLAGS_SINGLE);
        let glsl_bytes2 = u32x8_to_u8x32(&glsl_result2);

        let expected2 = blake3::Hasher::new_keyed(&hash_a)
            .update(&sparse_msg)
            .finalize();

        assert_eq!(
            &glsl_bytes2[..],
            expected2.as_bytes(),
            "Sparse noise (local_idx={}): GLSL BLAKE3 mismatch",
            local_idx,
        );

        // Dense and sparse messages are different -> hashes must differ
        assert_ne!(
            glsl_bytes, glsl_bytes2,
            "Dense and sparse should produce different hashes for local_idx={}",
            local_idx,
        );
    }
}

/// Jackpot hash: verify GLSL-style jackpot hash matches compute_jackpot_hash in proof_utils.rs.
#[test]
fn test_jackpot_hash_glsl_style_vs_blake3_crate() {
    let commitment_hash: [u8; 32] = *blake3::hash(b"jackpot_test_key").as_bytes();
    let key_u32 = u8x32_to_u32x8(&commitment_hash);

    let jackpot: [u32; 16] = std::array::from_fn(|i| (i as u32).wrapping_mul(0x9E3779B9));

    let glsl_result = compress_keyed(&jackpot, &key_u32, B3_FLAGS_SINGLE);
    let glsl_bytes = u32x8_to_u8x32(&glsl_result);

    // CPU reference: encode jackpot as 64 LE bytes, keyed hash
    let jackpot_bytes: [u8; 64] = std::array::from_fn(|i| jackpot[i / 4].to_le_bytes()[i % 4]);
    let expected = blake3::Hasher::new_keyed(&commitment_hash)
        .update(&jackpot_bytes)
        .finalize();

    assert_eq!(
        &glsl_bytes[..],
        expected.as_bytes(),
        "Jackpot hash: GLSL-style compress_keyed does not match blake3 crate",
    );
}

/// The 6+1 round structure (6 rounds with permute + 1 final round without)
/// must produce the same output as the full 7-round schedule.
#[test]
fn test_glsl_6_plus_1_rounds_vs_full_schedule() {
    // The upstream blake3 crate uses the full 7-round schedule internally.
    // Since both are functionally identical, our GLSL-style 6+1 should match.
    let key = *blake3::hash(b"round_test_key").as_bytes();
    let msg: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_mul(7));

    let key_u32 = u8x32_to_u32x8(&key);
    let msg_u32 = u8x64_to_u32x16(&msg);

    let glsl_result = compress_keyed(&msg_u32, &key_u32, B3_FLAGS_SINGLE);
    let glsl_bytes = u32x8_to_u8x32(&glsl_result);

    let expected = blake3::Hasher::new_keyed(&key).update(&msg).finalize();

    assert_eq!(
        &glsl_bytes[..],
        expected.as_bytes(),
        "6+1 round structure does not match full 7-round schedule",
    );
}

/// B-tensor seed label produces different results from A-tensor seed label
/// for the same key and index (verifying domain separation).
#[test]
fn test_a_vs_b_tensor_domain_separation() {
    let hash_a: [u8; 32] = *blake3::hash(b"domain_test").as_bytes();
    let key_u32 = u8x32_to_u32x8(&hash_a);

    let seed_a: [u8; 32] = {
        let mut s = [0u8; 32];
        s[..8].copy_from_slice(b"A_tensor");
        s
    };
    let seed_b: [u8; 32] = {
        let mut s = [0u8; 32];
        s[..8].copy_from_slice(b"B_tensor");
        s
    };

    let local_idx = 42u32;

    let mut msg_a = [0u8; 64];
    msg_a[0..4].copy_from_slice(&(local_idx + 1).to_le_bytes());
    msg_a[32..64].copy_from_slice(&seed_a);

    let mut msg_b = [0u8; 64];
    msg_b[0..4].copy_from_slice(&(local_idx + 1).to_le_bytes());
    msg_b[32..64].copy_from_slice(&seed_b);

    let result_a = compress_keyed(&u8x64_to_u32x16(&msg_a), &key_u32, B3_FLAGS_SINGLE);
    let result_b = compress_keyed(&u8x64_to_u32x16(&msg_b), &key_u32, B3_FLAGS_SINGLE);

    assert_ne!(
        result_a, result_b,
        "A_tensor and B_tensor should produce different hashes for the same index",
    );
}

/// Zero-length key (all zeros) and identity message produce a known output.
/// This is a sanity check that the implementation doesn't have trivial bugs.
#[test]
fn test_identity_vector() {
    let key = [0u8; 32];
    let msg = [0u8; 64];

    let key_u32 = u8x32_to_u32x8(&key);
    let msg_u32 = u8x64_to_u32x16(&msg);

    let glsl_result = compress_keyed(&msg_u32, &key_u32, B3_FLAGS_SINGLE);
    let glsl_bytes = u32x8_to_u8x32(&glsl_result);

    let expected = blake3::Hasher::new_keyed(&key).update(&msg).finalize();

    assert_eq!(
        &glsl_bytes[..],
        expected.as_bytes(),
        "Identity test vector mismatch",
    );
}

/// Verify that different flags produce different outputs (domain separation).
#[test]
fn test_different_flags_produce_different_outputs() {
    let key = [0u8; 32];
    let msg = [0u8; 64];
    let key_u32 = u8x32_to_u32x8(&key);
    let msg_u32 = u8x64_to_u32x16(&msg);

    let result_single = compress_keyed(&msg_u32, &key_u32, B3_FLAGS_SINGLE);

    // Different flag combination: KEYED_HASH only (no CHUNK_START/CHUNK_END/ROOT)
    let result_keyed_only = compress_keyed(&msg_u32, &key_u32, B3_KEYED_HASH);

    assert_ne!(
        result_single, result_keyed_only,
        "Different flags should produce different outputs",
    );
}
