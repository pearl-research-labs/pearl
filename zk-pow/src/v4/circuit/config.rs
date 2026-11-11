//! V4 STARK and recursive PLONK configurations.

use plonky2::fri::{FriConfig, reduction_strategies::FriReductionStrategy};
use plonky2::plonk::circuit_data::CircuitConfig;
use starky::config::StarkConfig;

use super::trace_padding::DEGREE_BITS as V4_REACHABLE_DEGREE_BITS;

/// V4 query-plus-PoW target; generic checks remain capped by the extension field.
const SECURITY_BITS: usize = 129;

/// Query count meeting the FRI target for the supplied rate and grinding bits.
const fn num_query_rounds(security_bits: usize, pow_bits: usize, rate_bits: usize) -> usize {
    security_bits.saturating_sub(pow_bits).div_ceil(rate_bits)
}

/// Builds the V4 inner (stage 1) or ZK outer (stage 2) configuration.
/// Panics for any other stage.
pub fn v4_recursion_config(stage: usize) -> CircuitConfig {
    let (pow_bits, rate_bits, num_routed_wires, is_zk) = match stage {
        1 => (18, 3, 36, false),
        2 => (23, 8, 37, true),
        _ => panic!("V4 recursion stage must be 1 or 2"),
    };
    CircuitConfig {
        num_wires: 135,
        num_routed_wires,
        num_constants: 2,
        use_base_arithmetic_gate: true,
        security_bits: 128,
        num_challenges: 3,
        zero_knowledge: is_zk,
        max_quotient_degree_factor: 8,
        fri_config: FriConfig {
            rate_bits,
            cap_height: 4,
            proof_of_work_bits: pow_bits as u32,
            batching_proof_of_work_bits: pow_bits as u32,
            reduction_strategy: FriReductionStrategy::ConstantArityBits(3, 7),
            num_query_rounds: num_query_rounds(SECURITY_BITS, pow_bits, rate_bits),
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
        num_query_rounds: num_query_rounds(SECURITY_BITS, pow_bits, rate_bits),
    };
    // Three CTL/Logup repetitions keep batched-lookup error below the FRI error.
    StarkConfig::new(128, 3, fri_config)
}
