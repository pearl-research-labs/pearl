use anyhow::{Result, bail, ensure};

use crate::api::{
    fp8::{
        jackpot_policy::{JackpotPolicy, OperandStrip},
        noise::{OperandNoise, compute_fp8_noise},
        plain_proof::PlainProofV4,
        prequant::{BLOCK_SIZE, PrequantOperand, exact_norms, open_prequant},
        quantization::{Fp8E4M3Quant, Quant},
        transcript::compute_jackpot_ticket,
    },
    primitives::IncompleteBlockHeader,
    proof_utils::check_jackpot_difficulty,
};

/// Open one operand's committed strips, compute (l2, linf) norms and noisy-quantize it:
/// the full "prepare one operand for the jackpot policy" step.
///
/// Return an [`OperandStrip`] whose `clean` field is the opened BF16 codes (the
/// operand as the miner committed it) and whose `built` field contains the noised
/// FP8 rows `A' = Q(alpha·A + beta·E@F)`.
fn open_and_noisy_quantize(
    operand: &PrequantOperand,
    k: usize,
    noise: &OperandNoise,
    quantization: &impl Quant<u16, u8>,
) -> Result<OperandStrip> {
    ensure!(
        operand.values.row_bytes() == k,
        "int8 value strip must hold exactly k={k} bytes"
    );
    ensure!(
        operand.scales.row_bytes() == 2 * (k / BLOCK_SIZE),
        "scales strip must hold 2 bytes per {BLOCK_SIZE}-element block of k={k}"
    );
    let ints: Vec<i8> = operand.values.as_bytes().iter().map(|&b| b as i8).collect();
    let scale_bits: Vec<u16> = operand
        .scales
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let n = operand.num_rows()?;
    let codes = open_prequant(&ints, &scale_bits, n, k, BLOCK_SIZE)?;
    // Exact norms from the int8 blocks + scales, bypassing the codes' BF16
    // rounding so miner and verifier agree despite a 1-ulp sum discrepancy.
    let norms = exact_norms(&ints, &scale_bits, n, k, BLOCK_SIZE)?;
    let built = quantization.noisy_quantize(&codes, noise, &norms)?;
    Ok(OperandStrip { clean: codes, built })
}

pub fn verify_plain_proof_with_policy(
    proposed_header: &IncompleteBlockHeader,
    plain_proof: &PlainProofV4,
    nbits_override: Option<u32>,
    jackpot_policy: JackpotPolicy,
) -> Result<()> {
    let (private_params, public_params) = plain_proof.parse_proof(proposed_header)?;
    let nbits = nbits_override.unwrap_or(proposed_header.nbits);
    let quantization = Fp8E4M3Quant;
    let k = public_params.common_dim() as usize;

    // Open each operand's committed strips, inject the deterministic noise,
    // and quantize: A' = Q(alpha_a·A + beta_a·E1@F1), likewise for B'.
    let noise = compute_fp8_noise(&public_params, proposed_header);
    let tile_a = open_and_noisy_quantize(&private_params.operands.a, k, &noise.a, &quantization)?;
    let tile_b = open_and_noisy_quantize(&private_params.operands.b, k, &noise.b, &quantization)?;

    let Some(message) = jackpot_policy.evaluate(&tile_a, &tile_b, k, &public_params.a().pattern, &public_params.b().pattern)?
    else {
        bail!("The jackpot is not admissible");
    };

    let ticket = compute_jackpot_ticket(&public_params.noise_seeds(proposed_header).a, &message);

    // The plain difficulty condition on the proven ticket digest.
    check_jackpot_difficulty(&ticket.jackpot, nbits, public_params.h(), public_params.w(), k as u32)
}

/// Verifies a v4 (FP8) plain proof, supplying the consensus-default
/// [`JackpotPolicy`].
pub fn verify_plain_proof(
    proposed_header: &IncompleteBlockHeader,
    plain_proof: &PlainProofV4,
    nbits_override: Option<u32>,
) -> Result<()> {
    verify_plain_proof_with_policy(proposed_header, plain_proof, nbits_override, JackpotPolicy::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::fp8::consistency::{fixture_job, fixture_job_untamed};

    /// The full plain-proof acceptance path: wire parse (Merkle openings),
    /// deterministic noise replay, noisy quantization, the four jackpot checks, and the
    /// winning condition (the fixture header's easy nbits saturates the difficulty bound).
    #[test]
    fn plain_verifier_accepts_the_honest_fixture() {
        let (header, plain) = fixture_job();
        verify_plain_proof(&header, &plain, None).unwrap_or_else(|e| panic!("the honest fixture must verify: {e:#}"));
    }

    /// A tile committing the same value plane on both sides replays coherent diagonal
    /// cells that blow the tamed-products allowance (jackpot check 3): the plain
    /// verifier rejects it, and lifting `eps_tame` alone accepts it — pinning the
    /// rejection to check 3. The ZK pipeline refuses to even trace the same job
    /// (`crate::circuit::fp8::consistency` tests).
    #[test]
    fn plain_verifier_rejects_an_untamed_tile() {
        let (header, plain) = fixture_job_untamed();
        let err = verify_plain_proof(&header, &plain, None).expect_err("the untamed tile must be rejected");
        assert!(
            format!("{err:#}").contains("not admissible"),
            "the rejection must be the policy's, got: {err:#}"
        );
        let lifted = JackpotPolicy {
            eps_tame: 1.0,
            ..JackpotPolicy::default()
        };
        verify_plain_proof_with_policy(&header, &plain, None, lifted)
            .unwrap_or_else(|e| panic!("only the tamed allowance may reject this tile: {e:#}"));
    }

    /// The v4 witness codec round-trips the dense consistency fixture: the
    /// fixed-width job blob, both variable-chunk opening pairs, and the
    /// optional MoE witness presence all survive `to_bytes`/`from_bytes`.
    #[test]
    fn plain_proof_v4_bincode_roundtrip() {
        let (_, plain) = fixture_job();
        let bytes = plain.to_bytes().expect("serialize");
        let restored = PlainProofV4::from_bytes(&bytes).expect("deserialize");
        assert_eq!(restored.job, plain.job);
        assert_eq!(restored.values.a.row_indices, plain.values.a.row_indices);
        assert_eq!(restored.values.a.proof.root, plain.values.a.proof.root);
        assert_eq!(restored.values.b.row_indices, plain.values.b.row_indices);
        assert_eq!(restored.values.b.proof.root, plain.values.b.proof.root);
        assert_eq!(restored.scales.a.row_indices, plain.scales.a.row_indices);
        assert_eq!(restored.scales.a.proof.root, plain.scales.a.proof.root);
        assert_eq!(restored.scales.b.row_indices, plain.scales.b.row_indices);
        assert_eq!(restored.scales.b.proof.root, plain.scales.b.proof.root);
        assert_eq!(restored.moe_witness.is_some(), plain.moe_witness.is_some());
        // Non-canonical re-encodings are rejected (no compat ladder).
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(PlainProofV4::from_bytes(&trailing).is_err());
    }
}
