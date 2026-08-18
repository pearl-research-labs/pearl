use crate::circuit::chip::blake3::program::DWORD_SIZE;
use crate::{
    api::proof::{Hash256, MiningConfiguration, PublicProofParams},
    api::proof_utils::nbits_to_difficulty,
    circuit::pearl_program::{TILE_D, TILE_H},
    ensure_eq,
};
use anyhow::{Result, ensure};
use log::info;
use primitive_types::U256;

/// Minimum accepted noise rank and normalization point for the rank penalty.
pub const PENALTY_BASE_RANK: usize = 128;

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

/// Checks a mining configuration and jackpot hash against the rank-penalty rule.
///
/// Deciding *when* the rule applies is the caller's job: this crate has no notion of
/// block height or fork activation. `nbits` is the difficulty target to measure the
/// jackpot against — a block header's nbits for consensus, or a pool's share target.
pub fn check_rank_penalty(config: &MiningConfiguration, hash_jackpot: &Hash256, nbits: u32) -> Result<()> {
    let rank = config.rank as usize;
    ensure!(rank >= PENALTY_BASE_RANK, "Rank must be >= {PENALTY_BASE_RANK} || r={rank}");
    // Reject zero rather than passing it to scale_target.
    let adjustment_factor = penalized_adjustment_factor(config);
    ensure!(
        adjustment_factor > 0,
        "Degenerate mining configuration: h*w={} dot_product_length={}",
        tile_size(config),
        config.dot_product_length()
    );
    ensure!(
        U256::from_little_endian(hash_jackpot) <= scale_target(nbits, adjustment_factor),
        "Jackpot condition not satisfied: hash does not meet the rank-penalized difficulty target"
    );
    Ok(())
}

/// Number of hashed output elements per jackpot attempt.
fn tile_size(config: &MiningConfiguration) -> usize {
    config.rows_pattern.size() as usize * config.cols_pattern.size() as usize
}

/// How much easier the jackpot bound is made for a given configuration, in
/// proportion to the work one attempt costs: the hashed tile size times the
/// dot product length.
fn difficulty_adjustment_factor(config: &MiningConfiguration) -> usize {
    tile_size(config) * config.dot_product_length()
}

fn penalized_adjustment_factor(config: &MiningConfiguration) -> usize {
    tile_size(config) * (config.dot_product_length() / config.rank as usize) * PENALTY_BASE_RANK
}

fn scale_target(nbits: u32, adjustment_factor: usize) -> U256 {
    scale(nbits_to_difficulty(nbits), adjustment_factor)
}

/// Scales `base` by `adjustment_factor`, saturating at [`U256::MAX`] on overflow.
///
/// Caller must ensure `adjustment_factor > 0`: [`check_rank_penalty`] checks it
/// outright, and [`public_params_sanity_check`] rules it out for a verified proof
/// by requiring `k >= 16r`.
fn scale(base: U256, adjustment_factor: usize) -> U256 {
    debug_assert!(adjustment_factor > 0, "scale: zero adjustment factor");
    if base > U256::MAX / adjustment_factor {
        info!(
            "Difficulty is too easy: hardness={} adjustment_factor={}",
            U256::MAX / base,
            adjustment_factor
        );
        U256::MAX
    } else {
        base * adjustment_factor
    }
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
    scale_target(nbits, difficulty_adjustment_factor(config))
}

/// [`extract_difficulty_bound`] with the rank penalty applied, or `None` when the
/// configuration is degenerate or the bound does not fit in 256 bits.
///
/// Unlike the consensus path this does not saturate: a miner scaling a share
/// target must be told the target is unusable rather than handed [`U256::MAX`],
/// which every hash satisfies.
pub fn penalized_target_bound(target: U256, config: &MiningConfiguration) -> Option<U256> {
    let factor = penalized_adjustment_factor(config);
    if factor == 0 || target > U256::MAX / factor {
        return None;
    }
    Some(target * factor)
}

#[cfg(test)]
mod rank_penalty_tests {
    use super::*;
    use crate::api::proof::{MMAType, PeriodicPattern};

    const TEST_NBITS: u32 = 0x1d00ffff;

    fn test_config(rank: usize, common_dim: u32) -> MiningConfiguration {
        MiningConfiguration {
            common_dim,
            rank: rank as u16,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
            moe: None,
        }
    }

    fn jackpot_of(value: U256) -> Hash256 {
        let mut bytes = [0u8; 32];
        value.to_little_endian(&mut bytes);
        bytes
    }

    /// The penalized counterpart of [`extract_difficulty_bound`], for the nbits
    /// these tests compare against.
    fn penalized_bound(nbits: u32, config: &MiningConfiguration) -> U256 {
        penalized_target_bound(nbits_to_difficulty(nbits), config).unwrap()
    }

    #[test]
    fn penalty_is_neutral_at_base_rank() {
        let config = test_config(PENALTY_BASE_RANK, 4096);
        assert_eq!(
            penalized_bound(TEST_NBITS, &config),
            extract_difficulty_bound(TEST_NBITS, &config)
        );
    }

    #[test]
    fn penalty_divides_bound_in_proportion_to_rank() {
        for multiple in [2usize, 4, 8] {
            let config = test_config(PENALTY_BASE_RANK * multiple, 65536);
            assert_eq!(
                penalized_bound(TEST_NBITS, &config),
                extract_difficulty_bound(TEST_NBITS, &config) / multiple,
                "rank {} should be penalized {}x",
                config.rank,
                multiple
            );
        }
    }

    #[test]
    fn rejects_rank_below_base() {
        let config = test_config(PENALTY_BASE_RANK / 2, 4096);
        let err = check_rank_penalty(&config, &[0u8; 32], TEST_NBITS).unwrap_err();
        assert!(err.to_string().contains("Rank must be"), "unexpected error: {err}");
    }

    /// The bound is inclusive, and the comparison uses the penalized bound: above
    /// the base rank, an unpenalized comparison would accept `bound + 1`.
    #[test]
    fn accepts_jackpot_at_the_bound_and_rejects_above_it() {
        for rank in [PENALTY_BASE_RANK, PENALTY_BASE_RANK * 2] {
            let config = test_config(rank, 65536);
            let bound = penalized_bound(TEST_NBITS, &config);
            check_rank_penalty(&config, &jackpot_of(bound), TEST_NBITS).unwrap();
            check_rank_penalty(&config, &jackpot_of(bound + 1), TEST_NBITS).unwrap_err();
        }
    }

    #[test]
    fn rejects_degenerate_config_without_panicking() {
        let config = test_config(PENALTY_BASE_RANK, 64);
        check_rank_penalty(&config, &[0u8; 32], TEST_NBITS).unwrap_err();
    }

    #[test]
    fn penalized_target_bound_scales_the_target() {
        // At the base rank the penalized factor equals the unpenalized one, so
        // penalized_target_bound(target) == target * difficulty_adjustment_factor.
        let config = test_config(PENALTY_BASE_RANK, 4096);
        let target = U256::from(1u64) << 200;
        let bound = penalized_target_bound(target, &config).unwrap();
        assert_eq!(bound, target * difficulty_adjustment_factor(&config));

        // Doubling the rank halves the bound when common_dim is held fixed (the
        // penalty's purpose): dot/rank halves, and BASE*rank doubles, net 1/2.
        let bigger = test_config(PENALTY_BASE_RANK * 2, 4096);
        let smaller_bound = penalized_target_bound(target, &bigger).unwrap();
        assert_eq!(smaller_bound, bound / 2);
    }

    /// Neither input a miner can be handed may come back as a usable bound: a
    /// degenerate configuration has no bound, and a target easy enough to overflow
    /// must not be clamped to U256::MAX, which every hash satisfies.
    #[test]
    fn penalized_target_bound_rejects_unusable_inputs() {
        let degenerate = test_config(PENALTY_BASE_RANK, 64);
        assert_eq!(penalized_adjustment_factor(&degenerate), 0);
        assert_eq!(penalized_target_bound(U256::from(1u64) << 200, &degenerate), None);

        let config = test_config(PENALTY_BASE_RANK, 4096);
        let largest = U256::MAX / penalized_adjustment_factor(&config);
        assert!(penalized_target_bound(largest, &config).is_some());
        assert_eq!(penalized_target_bound(largest + 1, &config), None);
        assert_eq!(penalized_target_bound(U256::MAX, &config), None);
    }
}
