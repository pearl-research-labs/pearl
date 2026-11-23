//! Optional grinding before column batching; zero is a strict transcript no-op.
use crate::field::extension::Extendable;
use crate::fri::FriConfig;
use crate::hash::hash_types::RichField;
use crate::iop::challenger::{Challenger, RecursiveChallenger};
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, GenericConfig, Hasher};
use anyhow::{ensure, Result};
// Domain separator: the ASCII label "FRIBPOW", encoded as a big-endian integer.
pub(crate) const CONFIG_TAG: u64 = u64::from_be_bytes(*b"\0FRIBPOW");
/// Grind on the current challenger state; return its batching nonce, or None when disabled.
pub fn prove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    challenger: &mut Challenger<F, C::Hasher>,
    config: &FriConfig,
) -> Option<F> {
    (config.batching_proof_of_work_bits != 0).then(|| {
        crate::fri::prover::proof_of_work::<F, C, D>(challenger, config.batching_proof_of_work_bits)
    })
}
/// Absorb the configured batching nonce and return its PoW response; None omits the step.
pub(crate) fn response<F: RichField, H: Hasher<F>>(
    challenger: &mut Challenger<F, H>,
    witness: Option<F>,
    config: &FriConfig,
) -> Option<F> {
    if config.batching_proof_of_work_bits == 0 {
        return None;
    }
    // Missing nonces are rejected by proof-shape validation, without choosing a
    // different transcript. Keep challenge derivation total for malformed proofs.
    challenger.observe_element(witness.unwrap_or(F::ZERO));
    Some(challenger.get_challenge())
}
pub fn verify_batching_pow<F: RichField + Extendable<D>, const D: usize>(
    response: Option<F>,
    config: &FriConfig,
) -> Result<()> {
    if config.batching_proof_of_work_bits == 0 {
        ensure!(response.is_none());
        return Ok(());
    }
    let response = response.ok_or_else(|| anyhow::anyhow!("missing batching PoW response"))?;
    crate::fri::verifier::verify_proof_of_work::<F, D>(response, config.batching_proof_of_work_bits)
}
pub fn response_target<F: RichField + Extendable<D>, H: AlgebraicHasher<F>, const D: usize>(
    b: &mut CircuitBuilder<F, D>,
    challenger: &mut RecursiveChallenger<F, H, D>,
    witness: Option<Target>,
    config: &FriConfig,
) -> Option<Target> {
    if config.batching_proof_of_work_bits == 0 {
        assert!(witness.is_none());
        return None;
    }
    challenger.observe_element(witness.expect("configured batching PoW target"));
    Some(challenger.get_challenge(b))
}
pub fn verify_batching_pow_target<F: RichField + Extendable<D>, const D: usize>(
    b: &mut CircuitBuilder<F, D>,
    response: Option<Target>,
    config: &FriConfig,
) {
    if config.batching_proof_of_work_bits == 0 {
        assert!(response.is_none());
        return;
    }
    b.verify_proof_of_work(
        response.expect("configured batching PoW response"),
        config.batching_proof_of_work_bits,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::extension::FieldExtension;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::Field;
    use crate::fri::reduction_strategies::FriReductionStrategy;
    use crate::hash::poseidon::PoseidonHash;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig as C;
    use crate::util::serialization::{Buffer, Read, Write};
    fn config(bits: u32) -> FriConfig {
        FriConfig {
            rate_bits: 3,
            cap_height: 1,
            proof_of_work_bits: 4,
            batching_proof_of_work_bits: bits,
            reduction_strategy: FriReductionStrategy::Fixed(vec![1]),
            num_query_rounds: 2,
        }
    }
    #[test]
    fn native_and_recursive_pow_transcripts() {
        for bits in [0, 4] {
            let cfg = config(bits);
            let mut p = Challenger::<F, PoseidonHash>::new();
            p.observe_elements(&[F::ONE; 2]);
            let initial = p.clone();
            let mut v = initial.clone();
            let nonce = prove::<F, C, 2>(&mut p, &cfg);
            verify_batching_pow::<F, 2>(response(&mut v, nonce, &cfg), &cfg).unwrap();
            let expected = p.get_extension_challenge::<2>();
            assert_eq!(v.get_extension_challenge::<2>(), expected);
            if bits == 0 {
                assert!(nonce.is_none());
                assert_eq!(initial.clone().get_extension_challenge::<2>(), expected);
            }
            let mut b = CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());
            let mut c = RecursiveChallenger::<F, PoseidonHash, 2>::new(&mut b);
            c.observe_elements(&b.constants(&[F::ONE; 2]));
            let target = nonce.map(|_| b.add_virtual_target());
            let gates = b.num_gates();
            let pow_response = response_target(&mut b, &mut c, target, &cfg);
            verify_batching_pow_target(&mut b, pow_response, &cfg);
            if bits == 0 {
                assert_eq!(b.num_gates(), gates);
            }
            let actual = c.get_extension_challenge(&mut b);
            b.register_public_inputs(&actual.0);
            let mut pw = PartialWitness::new();
            if let Some(t) = target {
                pw.set_target(t, nonce.unwrap()).unwrap();
            }
            let data = b.build::<C>();
            let proof = data.prove(pw).unwrap();
            let expected: [F; 2] = expected.to_basefield_array();
            assert_eq!(proof.public_inputs, expected);
            data.verify(proof).unwrap();
            if let Some(t) = target {
                let invalid = (0u64..)
                    .map(F::from_canonical_u64)
                    .find(|&n| {
                        let mut v = initial.clone();
                        let response = response(&mut v, Some(n), &cfg);
                        verify_batching_pow::<F, 2>(response, &cfg).is_err()
                    })
                    .unwrap();
                let mut pw = PartialWitness::new();
                pw.set_target(t, invalid).unwrap();
                if cfg!(debug_assertions) {
                    // Witness generation can reject the bad nonce before verification.
                    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        data.prove(pw).and_then(|proof| data.verify(proof))
                    }));
                    assert!(rejected.is_err() || rejected.unwrap().is_err());
                } else {
                    let proof = data.prove(pw).unwrap();
                    assert!(data.verify(proof).is_err());
                }
            }
        }
    }
    #[test]
    fn trusted_config_and_target_codecs_preserve_legacy_bytes() {
        let cfg = config(0);
        let mut bytes = Vec::new();
        bytes.write_fri_config(&cfg).unwrap();
        let mut old = Vec::new();
        old.write_usize(3).unwrap();
        old.write_usize(1).unwrap();
        old.write_usize(2).unwrap();
        old.write_u32(4).unwrap();
        old.write_fri_reduction_strategy(&cfg.reduction_strategy)
            .unwrap();
        assert_eq!(bytes, old);
        for bits in [0, 4] {
            let cfg = config(bits);
            let mut bytes = Vec::new();
            bytes.write_fri_config(&cfg).unwrap();
            assert_eq!(Buffer::new(&bytes).read_fri_config().unwrap(), cfg);
            let mut b = CircuitBuilder::<F, 2>::new(CircuitConfig::standard_recursion_config());
            let target = b.add_virtual_fri_proof(&[2], &cfg.fri_params(3, false));
            let mut bytes = Vec::new();
            bytes.write_target_fri_proof(&target).unwrap();
            assert_eq!(
                Buffer::new(&bytes).read_target_fri_proof::<2>().unwrap(),
                target
            );
        }
    }
}
