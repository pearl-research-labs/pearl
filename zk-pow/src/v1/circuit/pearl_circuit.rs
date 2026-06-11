// This file describe the high-level recursive structure of the Pearl proof.
// There are 3 layers to the proof:
//
// Layer 0: STARK computing hashes and matmul (uses Starky)
//
//     inputs:
//          Matrix slices,
//          Blake3 intermediate nodes.
//     Preprocessed data:
//          Noise,
//          Logic (exact layout of Blake3, Matmul chips).
//     public inputs:
//          JOB_KEY,
//          COMMITMENT_HASH,
//          HASH_A,
//          HASH_B,
//          HASH_JACKPOT.
//    Constraints:
//          a. matrix slices are int7
//          b. matrix slices plugged into blake3 merkle tree give HASH_A, HASH_B, using key=JOB_KEY.
//          c. matmul + reduction of noised slices equal JACKPOT_MSG.
//          d. Blake3(JACKPOT_MSG, key=COMMITMENT_HASH) = HASH_JACKPOT.
//
//    Each row of the STARK trace has the following three interleaved chips (see pearl_program.rs):
//      - Blake3 chip: one round per row (8 rounds per blake3 compression).
//      - Matmul chip: loads noised tiles (A+noise, B+noise) via RAM lookups, accumulates
//        inner products into CUMSUM_TILE.
//      - Jackpot chip: reduces CUMSUM_TILE into JACKPOT_MSG via XOR folding and bit rotations.
//
// Layer 1: Verifying STARK's proof and jackpot's inner hash (uses Plonky2 circuit builder)
//
//     inputs:
//          Stark's proof.
//          All stark's public inputs. (JOB_KEY, COMMITMENT_HASH, HASH_A, HASH_B, HASH_JACKPOT)
//     Preprocessed data:
//          Plonky2's circuit constants, defining logic.
//     public inputs:
//          JOB_KEY, // 8 elems, each uint32, encoding uint256
//          COMMITMENT_HASH, // 8 elems, each uint32, encoding uint256
//          HASH_A, // 8 elems, each uint32, encoding uint256
//          HASH_B, // 8 elems, each uint32, encoding uint256
//          HASH_JACKPOT // 8 elems, each uint32, encoding uint256, little endian
//          STARK_PREPROCESSED_DIGEST // 4 elems, each Goldilocks
//     Constraints:
//          a. Stark's proof is correct, with the specific pearl STARK's constraints.
//          b. Stark's preprocessed data hash matches STARK_PREPROCESSED_DIGEST. (checked using FRI)
//          c. JOB_KEY, COMMITMENT_HASH, HASH_A, HASH_B, HASH_JACKPOT match Stark's.
//
// Layer 2: Verifying Layer 1's proof (uses Plonky2 circuit builder)
//
//     inputs:
//          Layer 1's proof.
//          All layer 1's public inputs.
//     Preprocessed data:
//          Plonky2's circuit constants, defining logic.
//     public inputs:
//          Exactly same names as layer 1's public inputs.
//          CIRCUIT_1_DIGEST.
//    Constraints:
//          a. Layer 1's proof is correct.
//          b. Layer 1's public inputs match Layer 2's public inputs.
//          c. Layer 1's preprocessed data hash equals CIRCUIT_1_DIGEST. (checked during FRI)
//    ZK:
//          ZK is turned on for this proof, according to https://eprint.iacr.org/2024/1037.pdf.
//          This includes:
//             a. 128-bit Salting all merkle tree leaves.
//             b. OOD evaluation queries.
//             c. Blinding additive noise for trace polynomials.
//             d. Blinding additive noise for DEEP polynomial (the one checked by FRI to be of correct degree).
//          Does not include:
//             e. Restricting evaluation challenges to be disjoint of lookup values. Starky / Plonky2 has
//                |TRACE| / |GoldilocksField| probability to fail proving a valid trace. Success thus possesses a negligible leak.
//
//     The final proof is composed of two parts: STARK_PREPROCESSED_DIGEST (32 bytes), compact plonky2_proof (~60 KB).
//     This enables fail-fast verification: verifier first checks the proof using the claimed digest,
//     then computes the actual digest from public params and compares.

use crate::v1::{
    api::{proof::ZKProof, proof_utils::CompiledPublicParams},
    circuit::{
        circuit_utils::{
            CircuitCache, FirstCircuitData, SecondCircuitData, VerifierCircuitWithPolynomials, build_recursion_config,
            num_query_rounds,
        },
        pearl_layout::{BITS_PER_LIMB, pearl_columns, pearl_public},
        pearl_stark::PearlStark,
    },
    ensure_eq,
};
use anyhow::ensure;
use anyhow::{Result, bail};

use itertools::{Itertools, iproduct};
use log::{debug, info};
use plonky2::{
    field::{extension::quadratic::QuadraticExtension, goldilocks_field::GoldilocksField, polynomial::PolynomialValues},
    fri::{FriConfig, reduction_strategies::FriReductionStrategy},
    hash::hash_types::{HashOut, MerkleCapTarget},
    iop::witness::{PartialWitness, WitnessWrite},
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::VerifierCircuitTarget,
        config::{Blake3GoldilocksConfig, PoseidonGoldilocksConfig},
        proof::CompactProofWithPublicInputs,
    },
    util::{serialization::Write, timing::TimingTree},
};
use starky::{
    config::StarkConfig,
    prover::prove_and_get_zeta,
    recursive_verifier::{
        PreprocessedDataTarget, add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target, verify_stark_proof_circuit,
    },
    util::trace_rows_to_poly_values,
    verifier::eval_columns_at_zeta_and_next,
};

// Required security bits for IOPs in all proofs.
// We aim at ~110 bits of security,
// but raise it a little bit in light of the recent attacks
// https://eprint.iacr.org/2025/2010
// https://eprint.iacr.org/2025/2046
// https://eccc.weizmann.ac.il/report/2025/169/
pub const SECURITY_BITS: usize = 120;

// Allowed (rate_bits, pow_bits) combinations for each stage.
// Stage 0: STARK proof
pub const STAGE_0_PARAMS: &[(usize, usize)] = &[(1, 18), (2, 18)];
// Stage 1: First recursion circuit
pub const STAGE_1_PARAMS: &[(usize, usize)] = &[(3, 18)];
// Stage 2: Second recursion circuit (final proof, ZK enabled)
pub const STAGE_2_PARAMS: &[(usize, usize)] = &[(7, 22)];

/// Key for caching circuits based on STARK parameters
#[derive(Hash, Eq, PartialEq, Clone, Debug, Copy)]
pub struct PearlCircuitParams {
    pub stark_degree_bits: usize, // 13, 14, .., 18
    pub pow_bits: [usize; 3],
    pub rate_bits: [usize; 3],
}

impl PearlCircuitParams {
    /// Validate proof and circuit parameters for security requirements.
    pub fn sanity_check(&self, compiled_params: &CompiledPublicParams) -> Result<()> {
        for stage in 0..3 {
            validate_fri_params(stage, self.pow_bits[stage], self.rate_bits[stage])?;
        }
        // Check that RAM lookups used in the STARK fit 2*BITS_PER_LIMB bits
        ensure!(
            compiled_params.expected_num_rows() < (2 << (2 * BITS_PER_LIMB)),
            "Too many rows for RAM lookup"
        );
        ensure!(
            self.stark_degree_bits <= 19,
            "Verifier supports stark degree bits <= 19, got {}",
            self.stark_degree_bits
        );
        ensure!(
            self.stark_degree_bits + self.rate_bits[0] <= 20,
            "Verifier supports rate-expanded stark degree bits <= 20, got {}",
            self.stark_degree_bits + self.rate_bits[0]
        );
        Ok(())
    }
}

/// Validate pow_bits and rate_bits for a given FRI stage.
fn validate_fri_params(stage: usize, pow_bits: usize, rate_bits: usize) -> Result<()> {
    let allowed = match stage {
        0 => STAGE_0_PARAMS,
        1 => STAGE_1_PARAMS,
        2 => STAGE_2_PARAMS,
        _ => bail!("Invalid stage {}: must be 0, 1, or 2", stage),
    };
    ensure!(
        allowed.contains(&(rate_bits, pow_bits)),
        "Stage {}: (rate_bits, pow_bits) must be one of {:?}, got ({}, {})",
        stage,
        allowed,
        rate_bits,
        pow_bits
    );
    Ok(())
}

#[derive(Clone)]
pub struct PearlVerifierPIs {
    pub job_key: [GoldilocksField; 8],
    pub commitment_hash: [GoldilocksField; 8],
    pub hash_a: [GoldilocksField; 8],
    pub hash_b: [GoldilocksField; 8],
    pub hash_jackpot: [GoldilocksField; 8],
    pub public_data_commitment: HashOut<GoldilocksField>,
    /// STARK challenge point zeta extracted from the proof.
    pub zeta: QuadraticExtension<GoldilocksField>,
    /// Plaintext preprocessed columns in `StarkConfig::preprocessed_columns` order.
    pub preprocessed_columns: Vec<Vec<GoldilocksField>>,
}

/// API for Pearl Recursion Circuit. Not thread-safe.
pub trait RecursionCircuit: Clone + Default {
    const EXT_D: usize;
    type F;
    type InnerC;
    type OuterC;
    type StarkTrace;
    type CircuitParams;
    type CircuitCache;
    type VerifierPIs;

    fn stark_config(params: Self::CircuitParams) -> StarkConfig;

    fn compile_circuits(params: Self::CircuitParams, cache: &mut Self::CircuitCache, store_prover_data: bool) -> Result<()>;

    /// Pre-compile all verifier circuits for common parameter combinations.
    /// Used by build_cache example to generate the embedded cache binary.
    fn fill_verifier_cache(cache: &mut Self::CircuitCache);

    fn prove(
        circuit_params: Self::CircuitParams,
        cache: &mut Self::CircuitCache,
        stark_trace: Self::StarkTrace,
    ) -> Result<ZKProof>;

    fn verify(
        circuit_params: Self::CircuitParams,
        cache: &Self::CircuitCache,
        pis: Self::VerifierPIs,
        proof_bytes: &[u8],
    ) -> Result<bool>;
}

/// Pearl Recursion implementation for three-layered circuit architecture
#[derive(Clone, Copy, Debug, Default)]
pub struct PearlRecursion;

// Type definitions for PearlRecursion
impl RecursionCircuit for PearlRecursion {
    const EXT_D: usize = 2;
    type F = GoldilocksField;
    type InnerC = PoseidonGoldilocksConfig;
    type OuterC = Blake3GoldilocksConfig;
    /// (trace_rows, stark_public_inputs, hash_public_data)
    type StarkTrace = (
        Vec<[Self::F; pearl_columns::TOTAL]>,
        [Self::F; pearl_public::TOTAL],
        HashOut<Self::F>,
    );
    type CircuitParams = PearlCircuitParams;
    type VerifierPIs = PearlVerifierPIs;
    type CircuitCache = CircuitCache;

    fn stark_config(params: Self::CircuitParams) -> StarkConfig {
        StarkConfig {
            security_bits: SECURITY_BITS,
            num_challenges: 3,
            fri_config: FriConfig {
                rate_bits: params.rate_bits[0],
                cap_height: 5,
                proof_of_work_bits: params.pow_bits[0] as u32,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 7),
                num_query_rounds: num_query_rounds(SECURITY_BITS, params.pow_bits[0], params.rate_bits[0]),
            },
            preprocessed_columns: PearlStark::<Self::F, { Self::EXT_D }>::preprocessed_indices(),
        }
    }

    fn compile_circuits(params: Self::CircuitParams, cache: &mut Self::CircuitCache, store_prover_data: bool) -> Result<()> {
        let stark_config = Self::stark_config(params);
        let recursion_config_1 = build_recursion_config(params.rate_bits[1], params.pow_bits[1], 1, false);
        let recursion_config_2 = build_recursion_config(params.rate_bits[2], params.pow_bits[2], 2, true);

        let stark = PearlStark::<Self::F, { Self::EXT_D }>::default();

        /////////////////////////////////////////
        ///// build first recursion circuit /////
        /////////////////////////////////////////

        let first_key = CircuitCache::make_first_circuit_key(params);

        // Check if first circuit already in cache
        let first_in_cache = if store_prover_data {
            cache.prover_circuits_1.contains_key(&first_key)
        } else {
            cache.verifier_circuits_1.contains_key(&first_key)
        };

        // Build and store first circuit if not in cache
        if !first_in_cache {
            let mut builder_1 = CircuitBuilder::<GoldilocksField, 2>::new(recursion_config_1.clone());

            let mut proof_0_target =
                add_virtual_stark_proof_with_pis(&mut builder_1, &stark, &stark_config, params.stark_degree_bits, 0, 0);
            ensure_eq!(proof_0_target.public_inputs.len(), pearl_public::TOTAL);

            // Layer-1 PI layout: pearl_public | hash_public_data (4) | zeta (2) |
            //                    evals_at_zeta (2*N) | evals_at_g_zeta (2*N)
            // pearl_public (JOB_KEY, COMMITMENT_HASH, HASH_A, HASH_B, HASH_JACKPOT) of stark must PIs given outside
            for inner_pi in proof_0_target.public_inputs.iter_mut() {
                let outer_pi = builder_1.add_virtual_public_input();
                builder_1.connect(*inner_pi, outer_pi);
            }

            assert!(!stark_config.preprocessed_columns.is_empty());
            let num_prep_cols = stark_config.preprocessed_columns.len();

            // HASH_PUBLIC_DATA: opaque identifier for preprocessed columns.
            let hash_public_data_targets = builder_1.add_virtual_hash_public_input();

            let mut add_ext_public_input = || {
                let et = builder_1.add_virtual_extension_target();
                builder_1.register_public_inputs(&et.0);
                et
            };

            let zeta_pi = add_ext_public_input();

            // Preprocessed evaluations sinking from topmost PIs
            let evals_at_zeta = (0..num_prep_cols).map(|_| add_ext_public_input()).collect_vec();
            let evals_at_g_zeta = (0..num_prep_cols).map(|_| add_ext_public_input()).collect_vec();

            let preprocessed_circuit_data = PreprocessedDataTarget {
                digest: Some(hash_public_data_targets),
                evals_at_zeta: evals_at_zeta.clone(),
                evals_at_g_zeta: evals_at_g_zeta.clone(),
            };
            let stark_zeta =
                verify_stark_proof_circuit::<Self::F, Self::InnerC, PearlStark<Self::F, { Self::EXT_D }>, { Self::EXT_D }>(
                    &mut builder_1,
                    stark,
                    &proof_0_target,
                    &stark_config,
                    None,
                    Some(&preprocessed_circuit_data),
                );

            // Connect the exposed zeta PI to the zeta derived inside the STARK verifier.
            builder_1.connect_extension(zeta_pi, stark_zeta);

            let first_circuit = builder_1.build::<Self::InnerC>();

            // Always store verifier data (lightweight)
            cache.verifier_circuits_1.insert(first_key, first_circuit.verifier_data());

            // Conditionally store prover data
            if store_prover_data {
                cache.prover_circuits_1.insert(
                    first_key,
                    FirstCircuitData {
                        circuit: first_circuit.prover_data(),
                        proof_0_target,
                    },
                );
            }
        }

        /////////////////////////////////////////
        ///// build second recursion circuit ////
        /////////////////////////////////////////

        // Read first circuit common data from cache
        let first_common = &cache
            .verifier_circuits_1
            .get(&first_key)
            .ok_or_else(|| anyhow::anyhow!("First circuit verifier data not found in cache"))?
            .common;

        let second_key = CircuitCache::make_second_circuit_key(first_common.clone(), params);

        // Check if second circuit already in cache
        let second_in_cache = if store_prover_data {
            cache.prover_circuits_2.contains_key(&second_key)
        } else {
            cache.verifier_circuits_2.contains_key(&second_key)
        };

        // Build and store second circuit if not in cache
        if !second_in_cache {
            let mut builder_2 = CircuitBuilder::<Self::F, { Self::EXT_D }>::new(recursion_config_2.clone());

            let mut proof_1_target = builder_2.add_virtual_proof_with_pis(first_common);

            // Connect inner public inputs to outer public inputs
            for proof_1_pi in proof_1_target.public_inputs.iter_mut() {
                let proof_2_pi = builder_2.add_virtual_public_input();
                builder_2.connect(*proof_1_pi, proof_2_pi);
            }

            // Make expected_circuit_1_digest public inputs
            let cap_height = first_common.config.fri_config.cap_height;

            // Note that in current form, it is important to consume both circuit digest and sigmas_cap as public inputs, because
            // builder_2.verify_proof it is not verified that constants_sigmas_cap is correctly related to circuit_digest.
            let circuit_1_digest_pis = VerifierCircuitTarget {
                constants_sigmas_cap: MerkleCapTarget(builder_2.add_virtual_hashes_public_input(1 << cap_height)),
                circuit_digest: builder_2.add_virtual_hash_public_input(),
            };

            builder_2.verify_proof::<Self::InnerC>(&proof_1_target, &circuit_1_digest_pis, first_common);

            let second_circuit = builder_2.build::<Self::OuterC>();

            // Always store verifier data with polynomials
            cache.verifier_circuits_2.insert(
                second_key.clone(),
                VerifierCircuitWithPolynomials {
                    verifier_data: second_circuit.verifier_data(),
                    constants_sigmas_polynomials: second_circuit.prover_only.constants_sigmas_commitment.polynomials.clone(),
                },
            );

            // Conditionally store prover data
            if store_prover_data {
                cache.prover_circuits_2.insert(
                    second_key,
                    SecondCircuitData {
                        circuit: second_circuit.prover_data(),
                        proof_1_target,
                    },
                );
            }
        }

        Ok(())
    }

    fn fill_verifier_cache(cache: &mut Self::CircuitCache) {
        info!("Filling verifier cache with all parameter combinations...");

        let params: Vec<_> = iproduct!(
            [13, 14, 15, 16, 17, 18, 19], // stark_degree_bits
            STAGE_0_PARAMS.iter().copied(),
            STAGE_1_PARAMS.iter().copied(),
            STAGE_2_PARAMS.iter().copied()
        )
        .filter(|(degree_bits, (rate_bits_0, _), ..)| degree_bits + rate_bits_0 <= 20)
        .map(
            |(stark_degree_bits, (rate_bits_0, pow_bits_0), (rate_bits_1, pow_bits_1), (rate_bits_2, pow_bits_2))| {
                PearlCircuitParams {
                    stark_degree_bits,
                    pow_bits: [pow_bits_0, pow_bits_1, pow_bits_2],
                    rate_bits: [rate_bits_0, rate_bits_1, rate_bits_2],
                }
            },
        )
        .collect();

        info!("Total parameter combinations to compile: {}", params.len());

        for p in params {
            if let Err(e) = Self::compile_circuits(p, cache, false) {
                log::error!("Failed to compile circuits for params {:?}: {:?}", p, e);
            }
        }
    }

    fn prove(
        circuit_params: Self::CircuitParams,
        cache: &mut Self::CircuitCache,
        stark_trace: Self::StarkTrace,
    ) -> Result<ZKProof> {
        let (trace_rows, stark_public_inputs, hash_public_data) = stark_trace;
        let num_rows = trace_rows.len();

        let stark_timer = std::time::Instant::now();

        let stark_config = Self::stark_config(circuit_params);

        let stark = PearlStark::<Self::F, { Self::EXT_D }>::default();
        let (stark_proof, zeta) = prove_and_get_zeta::<Self::F, Self::InnerC, _, { Self::EXT_D }>(
            stark,
            &stark_config,
            trace_rows_to_poly_values(trace_rows),
            &stark_public_inputs,
            None,
            &mut TimingTree::default(),
            &hash_public_data.elements,
        )?;

        info!("Stark #0 proof time: {:?} || num_rows: {}", stark_timer.elapsed(), num_rows);

        let circuit_1_timer = std::time::Instant::now();

        // Ensure prover circuits are compiled
        let first_key = CircuitCache::make_first_circuit_key(circuit_params);

        // Get first prover circuit from cache
        let first_circuit_data = cache
            .prover_circuits_1
            .get_mut(&first_key)
            .ok_or_else(|| anyhow::anyhow!("First prover circuit not found in cache. Params: {:?}", first_key))?;

        // Create witness for the first recursion
        let mut pw_1 = PartialWitness::new();

        set_stark_proof_with_pis_target(
            &mut pw_1,
            &first_circuit_data.proof_0_target,
            &stark_proof,
            circuit_params.stark_degree_bits,
        )?;

        // Layer-1 PI Layout matches `compile_circuits` and `verify`.
        let openings = &stark_proof.proof.openings;
        let prep_evals = |vals: &[QuadraticExtension<GoldilocksField>]| -> Vec<GoldilocksField> {
            stark_config.preprocessed_columns.iter().flat_map(|&i| vals[i].0).collect()
        };
        let proof_1_pi_values = stark_public_inputs
            .iter()
            .copied()
            .chain(hash_public_data.elements)
            .chain(zeta.0)
            .chain(prep_evals(&openings.local_values))
            .chain(prep_evals(&openings.next_values))
            .collect_vec();

        // Set public inputs of verifier circuit
        let proof_1_pis = &first_circuit_data.circuit.prover_only.public_inputs;
        for (pi_t, pi) in proof_1_pis.iter().zip_eq(proof_1_pi_values.iter()) {
            pw_1.set_target(*pi_t, *pi)?;
        }

        // Compile proof for verifier circuit #1
        let proof_1 = plonky2::plonk::prover::prove_maybe_warmup::<Self::F, Self::InnerC, { Self::EXT_D }>(
            &mut first_circuit_data.circuit.prover_only,
            &first_circuit_data.circuit.common,
            pw_1,
            &mut TimingTree::default(),
        )?;

        {
            let mut proof_1_bytes = Vec::new();
            proof_1_bytes
                .write_proof(&proof_1.proof, &[])
                .map_err(|e| anyhow::anyhow!("Failed to serialize proof: {:?}", e))?;
            info!(
                "Recursion Circuit #1 prove time: {:?} || proof size: {:?}",
                circuit_1_timer.elapsed(),
                proof_1_bytes.len()
            );
        }

        // Create witness for the second recursion
        let circuit_2_timer = std::time::Instant::now();

        // Get second prover circuit from cache
        let second_key = CircuitCache::make_second_circuit_key(first_circuit_data.circuit.common.clone(), circuit_params);
        let second_circuit_data = cache
            .prover_circuits_2
            .get_mut(&second_key)
            .ok_or_else(|| anyhow::anyhow!("Second prover circuit not found in cache. Params: {:?}", second_key))?;

        let mut pw_2 = PartialWitness::new();

        pw_2.set_proof_with_pis_target(&second_circuit_data.proof_1_target, &proof_1)?;

        // Set circuit_2 public inputs: circuit_1 PIs + circuit_1_digest (constants_sigmas_cap + circuit_digest)
        // Get first circuit verifier data for the digest
        let first_verifier_data = cache
            .verifier_circuits_1
            .get(&first_key)
            .ok_or_else(|| anyhow::anyhow!("First verifier circuit not found in cache. Params: {:?}", first_key))?;

        let proof_2_pis = &second_circuit_data.circuit.prover_only.public_inputs;
        let circuit_2_pi_values: Vec<GoldilocksField> = proof_1_pi_values
            .iter()
            .copied()
            .chain(
                first_verifier_data
                    .verifier_only
                    .constants_sigmas_cap
                    .0
                    .iter()
                    .flat_map(|h| h.elements),
            )
            .chain(first_verifier_data.verifier_only.circuit_digest.elements)
            .collect();

        for (pi_t, pi) in proof_2_pis.iter().zip_eq(circuit_2_pi_values.iter()) {
            pw_2.set_target(*pi_t, *pi)?;
        }

        let proof = plonky2::plonk::prover::prove_maybe_warmup::<Self::F, Self::OuterC, { Self::EXT_D }>(
            &mut second_circuit_data.circuit.prover_only,
            &second_circuit_data.circuit.common,
            pw_2,
            &mut TimingTree::default(),
        )?;

        let compact: CompactProofWithPublicInputs<Self::F, Self::OuterC, { Self::EXT_D }> = proof.into();
        let plonky2_proof = compact.to_proof_bytes();

        info!(
            "Second recursion prove time: {:?} || proof size: {:?}",
            circuit_2_timer.elapsed(),
            plonky2_proof.len()
        );

        Ok(ZKProof::new(
            circuit_params.pow_bits.map(|b| b as u8),
            circuit_params.rate_bits.map(|b| b as u8),
            zeta,
            plonky2_proof,
        ))
    }

    fn verify(
        circuit_params: Self::CircuitParams,
        cache: &Self::CircuitCache,
        pis: Self::VerifierPIs,
        proof_bytes: &[u8],
    ) -> Result<bool> {
        let first_key = CircuitCache::make_first_circuit_key(circuit_params);
        // Get verifier circuits from cache
        let first_verifier_data = cache
            .verifier_circuits_1
            .get(&first_key)
            .ok_or_else(|| anyhow::anyhow!("First verifier circuit not found in cache. Params: {:?}", first_key))?;

        let second_key = CircuitCache::make_second_circuit_key(first_verifier_data.common.clone(), circuit_params);
        let second_verifier_data = cache
            .verifier_circuits_2
            .get(&second_key)
            .ok_or_else(|| anyhow::anyhow!("Second verifier circuit not found in cache. Params: {:?}", second_key))?;

        // Compute evaluations of preprocessed columns at zeta and g*zeta (unlike the prove path,
        // we don't have a STARK proof here — we only have the plaintext preprocessed columns, so
        // we must evaluate them explicitly).
        let poly_cols = pis.preprocessed_columns.into_iter().map(PolynomialValues::new).collect_vec();
        let poly_refs = poly_cols.iter().collect_vec();
        let (evals_at_zeta, evals_at_g_zeta) =
            eval_columns_at_zeta_and_next::<GoldilocksField, 2>(&poly_refs, pis.zeta, circuit_params.stark_degree_bits);

        // Layer-1 PI Layout matches `compile_circuits` and `prove`.
        let public_inputs = pis
            .job_key
            .into_iter()
            .chain(pis.commitment_hash)
            .chain(pis.hash_a)
            .chain(pis.hash_b)
            .chain(pis.hash_jackpot)
            .chain(pis.public_data_commitment.elements)
            .chain(pis.zeta.0)
            .chain(evals_at_zeta.iter().flat_map(|e| e.0))
            .chain(evals_at_g_zeta.iter().flat_map(|e| e.0))
            .chain(
                first_verifier_data
                    .verifier_only
                    .constants_sigmas_cap
                    .0
                    .iter()
                    .flat_map(|h| h.elements),
            )
            .chain(first_verifier_data.verifier_only.circuit_digest.elements)
            .collect_vec();

        let compact = CompactProofWithPublicInputs::<Self::F, Self::OuterC, { Self::EXT_D }>::from_bytes(
            proof_bytes,
            public_inputs,
            &second_verifier_data.verifier_data.common,
        )
        .map_err(|e| anyhow::anyhow!("deserializing compact Proof failed: {:?}", e))?;

        match compact.verify(
            &second_verifier_data.verifier_data.verifier_only,
            &second_verifier_data.verifier_data.common,
            &second_verifier_data.constants_sigmas_polynomials,
        ) {
            Ok(_) => Ok(true),
            Err(e) => {
                debug!("Verification failed: {:?}", e);
                Ok(false)
            }
        }
    }
}
