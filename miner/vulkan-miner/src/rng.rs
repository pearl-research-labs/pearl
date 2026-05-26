use blake3::Hash;

/// Derive a u64 seed for the GPU RNG from the job key and iteration counter.
pub fn compute_seed(job_key: &Hash, iteration: u64) -> u32 {
    let mut input = [0u8; 40];
    input[..32].copy_from_slice(job_key.as_bytes());
    input[32..40].copy_from_slice(&iteration.to_le_bytes());
    let h = blake3::hash(&input);
    u32::from_le_bytes([h.as_bytes()[0], h.as_bytes()[1], h.as_bytes()[2], h.as_bytes()[3]])
}
