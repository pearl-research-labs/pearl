use std::time::Instant;

use anyhow::Result;
use plonky2_field::goldilocks_field::GoldilocksField;

use crate::api::proof::{IncompleteBlockHeader, MiningConfiguration, PublicProofParams};
use crate::api::proof::{PrivateProofParams, ZKProof};
use crate::api::proof_utils::u32_field_array_to_hash;
use crate::circuit::circuit_utils::CircuitCache;
use crate::circuit::pearl_circuit::{PearlCircuitParams, PearlRecursion, RecursionCircuit};
use crate::circuit::pearl_layout::pearl_public;
use crate::circuit::pearl_stark::PearlStark;
use crate::ffi::plain_proof::{PlainProof, parse_plain_proof};

/// Active Rayon thread count visible to the prove call graph. Exported
/// so callers (icemining stratum, A4 benchmark) can report it
/// alongside the prove duration. The thread pool is owned by Rayon's
/// global, which initialises on first use from either an explicit
/// `ThreadPoolBuilder` or the `RAYON_NUM_THREADS` env var, falling
/// back to logical CPU count.
pub fn current_prove_thread_count() -> usize {
    plonky2_maybe_rayon::rayon::current_num_threads()
}

pub struct ProveResult {
    pub public_data: [u8; PublicProofParams::PUBLICDATA_SIZE],
    pub proof_data: Vec<u8>,
}

pub fn zk_prove_plain_proof(
    block_header: IncompleteBlockHeader,
    plain_proof: &PlainProof,
    cache: &mut CircuitCache,
    sanity_check: bool,
) -> Result<ProveResult> {
    // Convert PlainProof to proof parameters
    let (private, public) = parse_plain_proof(block_header, plain_proof)?;
    if sanity_check {
        public.sanity_check_private_params(&private)?;
    }

    // Generate ZK proof
    let mut public = public;
    let started = Instant::now();
    let rayon_threads = current_prove_thread_count();
    let proof = prove_block(&mut public, private, cache)?;
    let elapsed = started.elapsed();
    // §A4 of icemining/spec/PROOF_SPEEDUP.md: every prove call must
    // record the active Rayon thread count alongside the wall-clock
    // duration so dashboards can split prove latency by configured
    // parallelism. Structured as `key=value` so the icemining
    // stratum reporter can scrape this line without parsing prose.
    log::info!(
        "pearl_zk_prove_complete duration_ms={} rayon_threads={}",
        elapsed.as_millis(),
        rayon_threads,
    );

    let (public_data, proof_data) = proof.serialize(&public);

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
    // §A5 measured 2026-05-27 on coin-devnet-a (3970X, 8 vCPU): forcing
    // stage-0 rate_bits[0]=2 for all degrees is a net 10-13% prove-time
    // *regression* across thread counts 1/2/4/8. Higher rate_bits =
    // larger FRI blowup = more commit work, not less. The spec §3's
    // "low-risk lever" suggestion was directionally wrong for prove
    // latency. Stick with the original branch.
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

    let hash_public_data = public_params.public_data_commitment(&circuit_params);

    let proof = PearlRecursion::prove(circuit_params, cache, (trace_rows, stark_pis, hash_public_data))?;
    Ok(proof)
}

/// Architecture-C demonstration: split the prove across the miner/pool boundary.
/// The MINER produces STARK#0 (the heavy ~76%; commitment GPU-accelerated under
/// `PEARL_GPU_COMMIT`) and "ships" the `StarkProofWithPublicInputs`; the POOL runs
/// only Recursion#1+#2. Returns (cert, miner_stark_time, pool_recursion_time,
/// shipped_stark_proof_bytes). Demonstrates the pool shedding STARK#0.
pub fn prove_block_split(
    public_params: &mut PublicProofParams,
    private_params: PrivateProofParams,
    cache: &mut CircuitCache,
) -> Result<(ZKProof, std::time::Duration, std::time::Duration, usize)> {
    use crate::circuit::pearl_circuit::{pearl_prove_recursion_from_stark, pearl_prove_stark};

    let stark = PearlStark::<GoldilocksField, 2>::new_with_params(public_params);
    let compiled_params = &stark.config.as_ref().unwrap().compiled_public_params;
    let (trace_rows, stark_pis) = stark.generate_trace(public_params, private_params);

    let default_pow_bits = [18, 18, 22];
    let default_rate_bits = if compiled_params.degree_bits() >= 15 { [1, 3, 7] } else { [2, 3, 7] };
    public_params.hash_jackpot =
        u32_field_array_to_hash(&stark_pis[pearl_public::HASH_JACKPOT_RANGE].try_into().unwrap());
    let circuit_params = PearlCircuitParams {
        stark_degree_bits: compiled_params.degree_bits(),
        pow_bits: default_pow_bits.map(|b| b as usize),
        rate_bits: default_rate_bits.map(|b| b as usize),
    };
    PearlRecursion::compile_circuits(circuit_params, cache, true)?;
    let hash_public_data = public_params.public_data_commitment(&circuit_params);

    // === MINER: STARK#0 only ===
    let t_miner = std::time::Instant::now();
    let (stark_proof, zeta, stark_pis2, hpd) =
        pearl_prove_stark(circuit_params, (trace_rows, stark_pis, hash_public_data))?;
    let miner = t_miner.elapsed();

    // === ship across the wire: serialize the STARK proof (size demonstrates the ~58 KB payload) ===
    let ship_bytes = bincode::serialize(&stark_proof).map(|b| b.len()).unwrap_or(0);

    // === POOL: recursion only (no STARK#0) ===
    let t_pool = std::time::Instant::now();
    let proof = pearl_prove_recursion_from_stark(circuit_params, cache, stark_proof, zeta, stark_pis2, hpd)?;
    let pool = t_pool.elapsed();

    Ok((proof, miner, pool, ship_bytes))
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
