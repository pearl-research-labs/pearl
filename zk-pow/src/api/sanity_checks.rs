use crate::circuit::chip::blake3::program::DWORD_SIZE;
use crate::{
    api::proof::{MiningConfiguration, PublicProofParams},
    api::proof_utils::nbits_to_difficulty,
    circuit::pearl_program::{TILE_D, TILE_H},
    ensure_eq,
};
use anyhow::{Result, ensure};
use log::info;
use primitive_types::U256;

pub fn public_params_sanity_check(public_params: &PublicProofParams) -> Result<()> {
    let k = public_params.common_dim();
    let r = public_params.rank();
    let h = public_params.h();
    let w = public_params.w();
    let m = public_params.m;
    let n = public_params.n;
    let t_rows = public_params.t_rows;
    let t_cols = public_params.t_cols;
    let dot_product_len = public_params.dot_product_length();
    let worker_input_size = h.saturating_add(w).saturating_mul(dot_product_len);

    // Currently, noise generation supports only these ranks: powers of 2 from 2^5 to 2^10.
    ensure!(
        r.is_power_of_two() && (32..=1024).contains(&r),
        "Rank must be 2^5..2^10 || r={r}"
    );
    ensure!(r.is_multiple_of(TILE_D), "r must be divisible by {TILE_D} || r={r}");
    ensure!(k <= (1 << 16), "k must be <= 2^16 || k={k}");
    ensure!(k.is_multiple_of(64), "k must be divisible by 64 || k={k}");
    ensure!(k <= 4 * r * r, "k must be no more than 4r^2 || k={k} r={r}");
    ensure!(k >= 16 * r, "k must be >= 16r || k={k} r={r}");
    ensure!(k >= 1024, "k must be >= 1024 || k={k}"); // required for collision resistance since matrices are padded to a multiple 1024
    ensure!(
        h.is_multiple_of(TILE_H) && w.is_multiple_of(TILE_H),
        "Inner hash tile dimensions must be divisible by {TILE_H} || h={h} w={w}"
    );
    ensure!(h * w >= 32, "Inner hash size must be >= 32 || h={h} w={w}");
    ensure!(h * w <= 256, "Inner hash size must be <= 256 || h={h} w={w}");
    ensure!(
        dot_product_len.is_multiple_of(DWORD_SIZE),
        "dot_product_length must be divisible by DWORD_SIZE={DWORD_SIZE} || got {dot_product_len}"
    );
    ensure!(m <= (1 << 24), "m must be <= 2^24 || m={m}");
    ensure!(n <= (1 << 24), "n must be <= 2^24 || n={n}");
    ensure!(
        worker_input_size <= (1 << 22),
        "Worker input supported up to 4 MiB, got {} bytes",
        worker_input_size
    );
    let rmax = public_params.mining_config.rows_pattern.max();
    let cmax = public_params.mining_config.cols_pattern.max();
    ensure!(t_rows + rmax < m, "t_rows={t_rows} + pattern max={rmax} must be < m={m}");
    ensure!(t_cols + cmax < n, "t_cols={t_cols} + pattern max={cmax} must be < n={n}");
    ensure_eq!(
        public_params.mining_config.moe.is_some(),
        public_params.moe.is_some(),
        "mining_config.moe and public params moe must both be present or both absent"
    );

    if let Some(moe) = &public_params.moe {
        let moe_config = public_params
            .mining_config
            .moe
            .expect("checked above that both moe options are present");
        let (e, top_k) = (moe_config.e, moe_config.top_k);
        ensure!(e > 0, "number of experts e must be > 0 in the MoE setting");
        ensure!(top_k > 0, "top_k must be > 0 in the MoE setting");
        ensure!(
            (e as usize) <= PublicProofParams::MAX_NUM_EXPERTS,
            "number of experts {} exceeds maximum {}",
            e,
            PublicProofParams::MAX_NUM_EXPERTS
        );
        ensure!(
            e as usize == moe.routing_offsets.len(),
            "weight and routing offsets should have the same length, equal to the number of experts"
        );
        ensure!(top_k < e, "top_k should be < number of experts");
        ensure!(moe.expert_idx < e, "expert_idx should be < number of experts");

        // m * top_k must fit in u32 (routing_offsets are Vec<u32>).
        let num_routing_entries = public_params
            .m
            .checked_mul(top_k as u32)
            .ok_or_else(|| anyhow::anyhow!("m * top_k overflows u32"))? as usize;

        ensure_eq!(
            moe.routing_offsets
                .last()
                .copied()
                .expect("we already checked that we have at least one routing offset") as usize,
            num_routing_entries,
            "The last routing offset should be equal to the total number of routing entries (m*top_k={num_routing_entries})"
        );

        ensure!(
            moe.routing_offsets.windows(2).all(|w| w[0] <= w[1]),
            "The routing offsets must be monotonically non-decreasing"
        );
        ensure!(
            moe.routing_offsets.windows(2).all(|w| w[1] - w[0] <= m),
            "One expert holds at most all m tokens."
        );
        ensure!(
            moe.routing_offsets[0] <= m,
            "The first routing offset should be <= m since expert 0 can hold at most m tokens"
        );
        ensure!(
            public_params.n as u64 * moe_config.e as u64 <= (1 << 24),
            "Total number of columns across all experts must fit in 24 bits"
        );
        let start = moe.expert_start_offset() as usize;
        let next_start = moe.routing_offsets[moe.expert_idx as usize] as usize;
        for inner in public_params.a_inner_indices() {
            let abs_idx = start + inner as usize;
            ensure!(
                abs_idx < num_routing_entries,
                "routing index {abs_idx} (expert offset {start} + inner {inner}) falls in the padding region (real routing entries {num_routing_entries})"
            );
            ensure!(
                abs_idx < next_start,
                "routing index {abs_idx} (expert offset {start} + inner {inner}) falls in the next expert's region (next expert offset {next_start})"
            )
        }
        ensure!(
            moe.outer_indices.len() == public_params.a_inner_indices().len(),
            "outer_indices length must match a_inner_indices length"
        );
        ensure!(
            moe.outer_indices.windows(2).all(|w| w[0] < w[1]),
            "outer_indices must be sorted with no duplicates"
        );
        for outer_index in &moe.outer_indices {
            ensure!(*outer_index < m, "outer indices must be <= m");
        }
    }

    Ok(())
}

/// Checks that `hash_jackpot` meets the difficulty requirement derived from the block header.
/// `nbits_override` overrides the difficulty target when provided.
pub fn check_jackpot_against_nbits(public_params: &PublicProofParams, nbits_override: Option<u32>) -> Result<()> {
    let nbits = nbits_override.unwrap_or(public_params.block_header.nbits);
    let jackpot_hash_bound = extract_difficulty_bound(nbits, &public_params.mining_config);
    // hash_jackpot is interpreted as a little-endian 256-bit integer for the difficulty check
    ensure!(
        U256::from_little_endian(&public_params.hash_jackpot()) <= jackpot_hash_bound,
        "Jackpot condition not satisfied: hash does not meet difficulty target"
    );
    Ok(())
}

/// Computes difficulty bound for proof verification.
///
/// # Arguments
/// * `nbits` - Bitcoin compact difficulty target
/// * `config` - Mining configuration containing rank and pattern dimensions
///
/// # Returns
/// * Adjusted difficulty bound as U256
pub fn extract_difficulty_bound(nbits: u32, config: &MiningConfiguration) -> U256 {
    let target_difficulty = nbits_to_difficulty(nbits);
    let h = config.rows_pattern.size() as usize;
    let w = config.cols_pattern.size() as usize;
    let tile_size = h * w;
    let difficulty_adjustment_factor = tile_size * config.dot_product_length();
    if target_difficulty > U256::MAX / difficulty_adjustment_factor {
        info!(
            "Difficulty is too easy: hardness={} h*w*k={}",
            U256::MAX / target_difficulty,
            difficulty_adjustment_factor
        );
        U256::MAX
    } else {
        target_difficulty * difficulty_adjustment_factor
    }
}
