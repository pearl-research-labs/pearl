//! V4 STARK and recursive PLONK configurations.

use plonky2::fri::{FriConfig, reduction_strategies::FriReductionStrategy};
use plonky2::plonk::circuit_data::CircuitConfig;
use starky::config::StarkConfig;

use super::trace_padding::DEGREE_BITS as V4_REACHABLE_DEGREE_BITS;

/// V4 FRI security target.
const SECURITY_BITS: usize = 128;

/// Builds the V4 inner (stage 1) or ZK outer (stage 2) configuration.
/// Panics for any other stage.
pub fn v4_recursion_config(stage: usize) -> CircuitConfig {
    // Query counts stay fixed when retuning PoW.
    let (rate_bits, pow_bits, num_query_rounds, num_routed_wires, is_zk) = match stage {
        1 => (3, 17, 37, 36, false),
        2 => (8, 25, 14, 37, true),
        _ => panic!("V4 recursion stage must be 1 or 2"),
    };
    debug_assert!(rate_bits * num_query_rounds + pow_bits >= SECURITY_BITS);
    CircuitConfig {
        num_wires: 135,
        num_routed_wires,
        num_constants: 2,
        use_base_arithmetic_gate: true,
        security_bits: SECURITY_BITS,
        num_challenges: 3,
        zero_knowledge: is_zk,
        max_quotient_degree_factor: 8,
        fri_config: FriConfig {
            rate_bits,
            cap_height: 4,
            proof_of_work_bits: pow_bits as u32,
            batching_proof_of_work_bits: pow_bits as u32,
            reduction_strategy: FriReductionStrategy::ConstantArityBits(3, 7),
            num_query_rounds,
        },
    }
}

/// Builds the V4 STARK config with the fixed T3 ladder used by trace padding.
/// Every job absorbs the same config into its Fiat–Shamir transcript.
pub fn v4_stark_config() -> StarkConfig {
    let (rate_bits, pow_bits, num_query_rounds) = (1, 17, 111);
    debug_assert!(rate_bits * num_query_rounds + pow_bits >= SECURITY_BITS);
    let fri_config = FriConfig {
        rate_bits,
        cap_height: 4,
        proof_of_work_bits: pow_bits as u32,
        batching_proof_of_work_bits: pow_bits as u32,
        reduction_strategy: FriReductionStrategy::Ladder(V4_REACHABLE_DEGREE_BITS.to_vec()),
        num_query_rounds,
    };
    // Three CTL/Logup repetitions keep batched-lookup error below the FRI error.
    StarkConfig::new(SECURITY_BITS, 3, fri_config)
}
