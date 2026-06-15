use anyhow::Result;
use plonky2_field::goldilocks_field::GoldilocksField;

use crate::api::proof::{IncompleteBlockHeader, MiningConfiguration, PublicProofParams};
use crate::api::proof::{PrivateProofParams, ZKProof};
use crate::api::proof_utils::u32_field_array_to_hash;
use crate::circuit::circuit_utils::CircuitCache;
use crate::circuit::pearl_circuit::{PearlCircuitParams, PearlRecursion, RecursionCircuit};
use crate::circuit::pearl_layout::pearl_public;
use crate::circuit::pearl_stark::PearlStark;
use crate::ffi::plain_proof::PlainProof;

pub struct ProveResult {
    /// [`PublicProofParams::WIRE_SIZE`] bytes for non-MoE; longer for MoE proof.
    pub public_data: Vec<u8>,
    pub proof_data: Vec<u8>,
}

/// Parse a proof (plain or MoE), generate a ZK proof, and return the serialized result.
pub fn zk_prove_plain_proof(
    block_header: IncompleteBlockHeader,
    proof: &PlainProof,
    cache: &mut CircuitCache,
    sanity_check: bool,
) -> Result<ProveResult> {
    // Convert PlainProof to proof parameters
    let (private, public) = proof.parse_proof(block_header)?;
    if sanity_check {
        public.sanity_check_private_params(&private)?;
    }

    // Generate ZK proof
    let mut public = public;
    let proof = prove_block(&mut public, private, cache)?;

    let (public_data, proof_data) = proof.serialize(&public)?;

    Ok(ProveResult { public_data, proof_data })
}

pub fn prove_block(
    public_params: &mut PublicProofParams,
    private_params: PrivateProofParams,
    cache: &mut CircuitCache,
) -> Result<ZKProof> {
    let stark = PearlStark::<GoldilocksField, 2>::new_with_params(public_params);
    let compiled_params = &stark.config.as_ref().unwrap().compiled_public_params;

    let (trace_rows, stark_pis) = stark.generate_trace(public_params, private_params);

    let default_pow_bits = [18, 18, 22];
    let default_rate_bits = if compiled_params.degree_bits() >= 15 {
        [1, 3, 7]
    } else {
        [2, 3, 7]
    };

    public_params.hash_jackpot = u32_field_array_to_hash(&stark_pis[pearl_public::HASH_JACKPOT_RANGE].try_into().unwrap());

    let circuit_params = PearlCircuitParams {
        stark_degree_bits: compiled_params.degree_bits(),
        pow_bits: default_pow_bits.map(|b| b as usize),
        rate_bits: default_rate_bits.map(|b| b as usize),
    };
    PearlRecursion::compile_circuits(circuit_params, cache, true)?;

    let hash_public_data = public_params.public_data_commitment(&circuit_params)?;

    let proof = PearlRecursion::prove(circuit_params, cache, (trace_rows, stark_pis, hash_public_data))?;
    Ok(proof)
}

/// Warms up the circuit cache by running a proof with the given parameters.
///
/// **Note**: This is an optimization, not a guarantee. For borderline proof sizes,
/// the cached circuit may not match the actual proof's requirements. In such cases,
/// the circuit will be rebuilt during the actual prove call.
///
/// # Arguments
/// * `mining_configuration` - The mining configuration to use
/// * `cache` - Circuit cache to warm up
pub fn warmup_prove(mining_configuration: MiningConfiguration, cache: &mut CircuitCache) -> Result<()> {
    let tile_h = mining_configuration.rows_pattern.size() as usize;
    let tile_w = mining_configuration.cols_pattern.size() as usize;
    let common_dim = mining_configuration.common_dim as usize;

    let private_params = PrivateProofParams {
        s_a: vec![vec![0i8; common_dim]; tile_h],
        s_b: vec![vec![0i8; common_dim]; tile_w],
        s_routing: vec![],
        external_msgs: vec![],
        external_cvs: vec![],
    };

    let block_header = IncompleteBlockHeader {
        version: 0,
        prev_block: [0; 32],
        merkle_root: [0; 32],
        timestamp: 0,
        nbits: 0x207FFFFF, // Most permissive difficulty
    };

    let m = mining_configuration.rows_pattern.max() + 1;
    let n = mining_configuration.cols_pattern.max() + 1;
    let mut public_params = PublicProofParams::new_dummy(block_header, mining_configuration, m, n, 0, 0);
    let private_params = public_params.fill_dummy_merkle_proof(private_params)?;

    let _ = prove_block(&mut public_params, private_params, cache)?;
    Ok(())
}
