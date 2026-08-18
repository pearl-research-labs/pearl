use anyhow::{Result, bail, ensure};
use plonky2_field::extension::FieldExtension;
use plonky2_field::goldilocks_field::GoldilocksField;

use crate::ffi::plain_proof::PlainProof;
use crate::v1::{
    api::{
        plain_proof::parse_plain_proof,
        proof::{IncompleteBlockHeader, PublicProofParams, ZKProof},
        proof_utils::{CompiledPublicParams, compute_jackpot_hash, hash_to_u32_field_array},
        sanity_checks::{check_jackpot_difficulty_with_nbits, public_params_sanity_check},
    },
    circuit::{
        chip::compute_jackpot,
        circuit_utils::CircuitCache,
        pearl_circuit::{PearlCircuitParams, PearlRecursion, PearlVerifierPIs, RecursionCircuit},
        pearl_noise::compute_noise,
        pearl_stark::PearlStark,
    },
};

/// Verifies a block proof, compiling circuits into cache if needed.
pub fn verify_block(public_params: &PublicProofParams, proof: &ZKProof, cache: &mut CircuitCache) -> Result<()> {
    let (params, pis) = prepare_verification(public_params, proof, None)?;
    PearlRecursion::compile_circuits(params, cache, false)?;
    verify_with_cache(params, cache, &pis, proof)
}

/// Verifies a block proof using pre-compiled circuits. Fails if verifying circuit not in cache.
pub fn verify_block_cached_circuits_only(
    public_params: &PublicProofParams,
    proof: &ZKProof,
    cache: &CircuitCache,
    nbits_override: Option<u32>,
) -> Result<()> {
    let (params, pis) = prepare_verification(public_params, proof, nbits_override)?;
    verify_with_cache(params, cache, &pis, proof)
}

fn verify_with_cache(params: PearlCircuitParams, cache: &CircuitCache, pis: &PearlVerifierPIs, proof: &ZKProof) -> Result<()> {
    if PearlRecursion::verify(params, cache, pis.clone(), &proof.plonky2_proof)? {
        Ok(())
    } else {
        bail!("Proof Invalid")
    }
}

fn prepare_verification(
    public_params: &PublicProofParams,
    proof: &ZKProof,
    nbits_override: Option<u32>,
) -> Result<(PearlCircuitParams, PearlVerifierPIs)> {
    public_params_sanity_check(public_params)?;
    check_jackpot_difficulty_with_nbits(public_params, nbits_override)?;

    let compiled_params = CompiledPublicParams::from(public_params);
    let circuit_params = PearlCircuitParams {
        stark_degree_bits: compiled_params.degree_bits(),
        pow_bits: proof.pow_bits.map(|b| b as usize),
        rate_bits: proof.rate_bits.map(|b| b as usize),
    };
    circuit_params.sanity_check(&compiled_params)?;

    let pis = build_verifier_pis(public_params, &compiled_params, &circuit_params, proof)?;
    Ok((circuit_params, pis))
}

fn build_verifier_pis(
    public_params: &PublicProofParams,
    compiled_params: &CompiledPublicParams,
    circuit_params: &PearlCircuitParams,
    proof: &ZKProof,
) -> Result<PearlVerifierPIs> {
    let (_, commitment_hash) = compiled_params.commitment_hash;
    let zeta = proof.zeta()?;
    ensure!(
        !FieldExtension::<2>::is_in_basefield(&zeta),
        "zeta must lie strictly in the extension field (b component is zero)"
    );
    Ok(PearlVerifierPIs {
        job_key: hash_to_u32_field_array(&compiled_params.job_key),
        commitment_hash: hash_to_u32_field_array(&commitment_hash),
        hash_a: hash_to_u32_field_array(&public_params.hash_a),
        hash_b: hash_to_u32_field_array(&public_params.hash_b),
        hash_jackpot: hash_to_u32_field_array(&public_params.hash_jackpot),
        public_data_commitment: public_params.public_data_commitment(circuit_params),
        zeta,
        preprocessed_columns: PearlStark::<GoldilocksField, 2>::preprocessed_columns(compiled_params)?,
    })
}

/// Verifies a V1 plain proof (mining solution) without generating a ZK proof.
///
/// When `nbits_override` is provided it replaces the header's nbits as the
/// difficulty target (e.g. for pool share-difficulty checks).
pub fn verify_plain_proof(
    block_header: &IncompleteBlockHeader,
    plain_proof: &PlainProof,
    nbits_override: Option<u32>,
) -> Result<()> {
    let (private_params, mut public_params) = parse_plain_proof(*block_header, plain_proof)?;

    public_params_sanity_check(&public_params)?;

    for strip in private_params.s_a.iter().chain(private_params.s_b.iter()) {
        for &val in strip {
            ensure!((-64..=64).contains(&val), "Matrix value {} out of range [-64, 64]", val);
        }
    }

    let compiled = CompiledPublicParams::from(&public_params);
    let noise = compute_noise(&compiled);
    let jackpot = compute_jackpot(&compiled, &private_params.s_a, &private_params.s_b, &noise);

    public_params.hash_jackpot = compute_jackpot_hash(&jackpot, compiled.a_noise_seed());
    check_jackpot_difficulty_with_nbits(&public_params, nbits_override)?;

    Ok(())
}
