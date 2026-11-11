//! V4 STARK and recursive PLONK configurations.

use plonky2::fri::{FriConfig, reduction_strategies::FriReductionStrategy};
use plonky2::plonk::circuit_data::CircuitConfig;
use starky::config::StarkConfig;

use super::trace_padding::DEGREE_BITS as V4_REACHABLE_DEGREE_BITS;

/// Builds the V4 inner (stage 1) or ZK outer (stage 2) configuration.
/// Panics for any other stage.
pub fn v4_recursion_config(stage: usize) -> CircuitConfig {
    let (pow_bits, rate_bits, num_query_rounds, num_routed_wires, is_zk, arity_bits) = match stage {
        1 => (18, 3, 39, 37, false, 4),
        2 => (23, 8, 14, 38, true, 3),
        _ => panic!("V4 recursion stage must be 1 or 2"),
    };
    CircuitConfig {
        num_wires: 135,
        num_routed_wires,
        num_constants: 2,
        use_base_arithmetic_gate: true,
        security_bits: 108,
        num_challenges: 3,
        zero_knowledge: is_zk,
        max_quotient_degree_factor: 8,
        fri_config: FriConfig {
            rate_bits,
            cap_height: 4,
            proof_of_work_bits: pow_bits as u32,
            batching_proof_of_work_bits: pow_bits as u32,
            reduction_strategy: FriReductionStrategy::ConstantArityBits(arity_bits, 7),
            num_query_rounds,
        },
    }
}

/// Builds the V4 STARK config with the fixed T3 ladder used by trace padding.
/// Every job absorbs the same config into its Fiat–Shamir transcript.
pub fn v4_stark_config() -> StarkConfig {
    let (pow_bits, rate_bits) = (18, 1);
    let fri_config = FriConfig {
        rate_bits,
        cap_height: 4,
        proof_of_work_bits: pow_bits as u32,
        batching_proof_of_work_bits: pow_bits as u32,
        reduction_strategy: FriReductionStrategy::Ladder(V4_REACHABLE_DEGREE_BITS.to_vec()),
        num_query_rounds: 114,
    };
    // Three CTL/Logup repetitions keep batched-lookup error below the FRI error.
    StarkConfig::new(108, 3, fri_config)
}
