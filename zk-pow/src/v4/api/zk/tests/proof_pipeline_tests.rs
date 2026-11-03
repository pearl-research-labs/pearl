use plonky2::field::types::Field;

use super::super::*;
use crate::v4::circuit::consistency::{
    fixture_job, fixture_job_asym, fixture_job_k32, fixture_job_medium, fixture_job_min_k, fixture_job_moe,
};
use crate::v4::circuit::ctl::Table;
use crate::v4::circuit::input_quant_stark::columns::{
    B_KEY_OFFSET_PUBLIC_INPUT, OPERAND_MULT_A_PUBLIC_INPUT, WL2_POW_PUBLIC_INPUT,
};
use crate::v4::circuit::scale_stark::columns::{K_PUBLIC_INPUT, WL2_PUBLIC_INPUT};

/// The full wire-level round trip: parse the plain proof, prove the batch statement,
/// serialize, verify against independently derived expectations, and reject tampering
/// with the block's claims, the proof's public inputs, and the openings.
#[test]
fn api_roundtrip_and_tamper_rejection() {
    let (header, plain) = fixture_job();
    let mut timing = TimingTree::default();

    let (_, params) = plain.parse_proof(&header).expect("fixture job must parse");
    assert_eq!(params.common().device, Device::B200, "the wire byte must parse back");
    let setup = Fp8ProverSetup::build(params.common().device, &mut timing).expect("prover setup");

    let (public, proof) = zk_prove_plain_proof_fp8(&header, &plain, &setup, &mut timing).expect("proving must succeed");
    assert_ne!(public.hash_jackpot(), [0u8; 32], "prove must ship the proven lottery digest");

    let proof = Fp8Proof::from_bytes(&proof.to_bytes().expect("serialize")).expect("deserialize");
    verify_fp8_block(&public, &header, &proof, &setup, None).expect("honest proof must verify");

    let mut bad_public = public.clone();
    bad_public.set_hash_jackpot({
        let mut h = bad_public.hash_jackpot();
        h[0] ^= 1;
        h
    });
    assert!(verify_fp8_block(&bad_public, &header, &proof, &setup, None).is_err());

    let job = Fp8Job::derive(&public, &header).unwrap();
    let mut tampered = proof.clone();
    tampered.0.public_inputs[job.system.main_table_positions()[Table::Scale as usize]][K_PUBLIC_INPUT] += F::ONE;
    assert!(verify_fp8_block(&public, &header, &tampered, &setup, None).is_err());

    let mut tampered = proof.clone();
    tampered.0.proof.openings[0].local_values[0] += F::ONE.into();
    assert!(verify_fp8_block(&public, &header, &tampered, &setup, None).is_err());
}

/// The padded-geometry wire-level round trip ([`fixture_job_asym`]: `h = 16 != w = 20`,
/// `k = 2048` — a reference-legal job the pre-padding AIRs could not prove, since
/// `h + w = 36` and `h*w = 320` are not powers of two). Exercises `derive` without the
/// old envelope checks and every AIR's padding path through parse -> derive -> prove ->
/// verify.
#[test]
fn asymmetric_api_roundtrip() {
    let (header, plain) = fixture_job_asym();
    let mut timing = TimingTree::default();

    let (_, params) = plain.parse_proof(&header).expect("asymmetric fixture job must parse");
    assert_ne!(params.h(), params.w(), "the fixture must be asymmetric");
    let setup = Fp8ProverSetup::build(params.common().device, &mut timing).expect("prover setup");

    let (public, proof) = zk_prove_plain_proof_fp8(&header, &plain, &setup, &mut timing).expect("proving must succeed");
    let proof = Fp8Proof::from_bytes(&proof.to_bytes().expect("serialize")).expect("deserialize");

    verify_fp8_block(&public, &header, &proof, &setup, None).expect("honest proof must verify");
}

/// The `k % 32` wire-level round trip ([`fixture_job_k32`]: `k = 2080`, committed rows
/// that never tile into whole 64-byte Blake3 blocks). Exercises the straddling-block
/// schedule and the non-power-of-two row-mean scale derivation through parse -> derive ->
/// prove -> verify.
#[test]
fn k_mod_32_api_roundtrip() {
    let (header, plain) = fixture_job_k32();
    let mut timing = TimingTree::default();

    let (_, params) = plain.parse_proof(&header).expect("k % 32 fixture job must parse");
    assert_eq!(params.common_dim() % 64, 32, "the fixture must exercise the k % 32 envelope");
    let setup = Fp8ProverSetup::build(params.common().device, &mut timing).expect("prover setup");

    let (public, proof) = zk_prove_plain_proof_fp8(&header, &plain, &setup, &mut timing).expect("proving must succeed");
    let proof = Fp8Proof::from_bytes(&proof.to_bytes().expect("serialize")).expect("deserialize");

    verify_fp8_block(&public, &header, &proof, &setup, None).expect("honest proof must verify");
}

#[test]
fn hopper_api_roundtrips_minimum_k_and_partial_window() {
    let mut jobs = [fixture_job_min_k(), fixture_job_k32()];
    for (_, plain) in &mut jobs {
        plain.job.common.device = Device::H100;
    }

    let mut timing = TimingTree::default();
    let setup = Fp8ProverSetup::build(Device::H100, &mut timing).expect("Hopper prover setup");
    assert_eq!(setup.device, Device::H100);

    for (header, plain) in jobs {
        let (_, params) = plain.parse_proof(&header).expect("Hopper fixture must parse");
        assert_eq!(params.common().device, Device::H100);
        let (public, proof) =
            zk_prove_plain_proof_fp8(&header, &plain, &setup, &mut timing).expect("Hopper proving must succeed");
        let proof = Fp8Proof::from_bytes(&proof.to_bytes().expect("serialize")).expect("deserialize");
        verify_fp8_block(&public, &header, &proof, &setup, None).expect("Hopper proof must verify");
    }
}

/// Writes `σ̂(76) ‖ u32le public_data_len ‖ public_data ‖ u8 chain_len ‖ chain(108 each,
/// parent first) ‖ proof_data` per device; σ_d itself rides inside `public_data`.
pub(super) fn regenerate_go_fixture() {
    let mut prover = Fp8Prover::setup(Device::H100).expect("fp8 setup");
    let mut fixtures = Vec::new();
    for (device, suffix) in [(Device::H100, "h100"), (Device::B200, "b200")] {
        let (header, mut plain) = match device {
            Device::H100 => fixture_job_k32(),
            Device::B200 => fixture_job(),
        };
        plain.job.common.device = device;
        let (public_data, proof_data) = prover.prove(&header, &plain).expect("wrapped proving must succeed");

        let public_data_len = u32::try_from(public_data.len()).expect("public data length must fit u32");
        let chain_len = u8::try_from(plain.ancestor_chain.len()).expect("the chain fits the state window");
        let mut fixture = Vec::new();
        fixture.extend_from_slice(&header.to_bytes());
        fixture.extend_from_slice(&public_data_len.to_le_bytes());
        fixture.extend_from_slice(&public_data);
        fixture.push(chain_len);
        for ancestor in &plain.ancestor_chain {
            fixture.extend_from_slice(&ancestor.to_bytes());
        }
        fixture.extend_from_slice(&proof_data);
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../node/zkpow/testdata/fp8_zk_proof_{suffix}.bin"));
        fixtures.push((device, path, fixture));
    }
    assert_eq!(prover.setups.len(), 2, "one prover must retain both device setups");

    for (device, path, fixture) in fixtures {
        std::fs::write(&path, fixture).expect("write Go fp8 fixture");
        println!("{device:?} Go fixture written to {}", path.display());
    }
}

#[test]
#[cfg(feature = "embedded_cache")]
#[ignore = "wrapped proving needs more memory than the default CI runner; invoked explicitly on a larger machine"]
fn wrapped_devices_roundtrip_through_embedded_cache() {
    let cache = Fp8VerifierCache::from_bytes(crate::v4::api::embedded_cache::CACHE_DATA).expect("embedded cache must decode");
    assert!(
        cache.contains_all_devices(),
        "the embedded cache must contain exactly both device setups"
    );

    let mut prover = Fp8Prover::setup(Device::H100).expect("device prover setup");
    for device in [Device::H100, Device::B200] {
        let (header, mut plain) = fixture_job();
        plain.job.common.device = device;
        let (public_data, proof_data) = prover.prove(&header, &plain).expect("wrapped proving must succeed");
        let statement = decode_statement(&public_data).expect("published statement must decode");
        assert_eq!(statement.common().device, device);
        let chain = &plain.ancestor_chain;
        cache
            .verify_block(&header, chain, &public_data, &proof_data)
            .expect("embedded device setup must verify the wrapped proof");

        let mut relabeled = statement.clone();
        relabeled.set_device_for_test(match device {
            Device::H100 => Device::B200,
            Device::B200 => Device::H100,
        });
        assert!(
            cache
                .verify_block(&header, chain, &relabeled.to_bytes(), &proof_data)
                .is_err(),
            "a proof must not verify through the other device's setup"
        );
    }
    assert_eq!(prover.setups.len(), 2, "one prover must retain both device setups");
}

/// The wrapped (published) round trip and rejection surface.
#[test]
#[ignore = "exceeds the safe memory budget of an 8 GiB CI runner; run explicitly on a larger machine"]
fn wrapped_api_roundtrip_and_tamper_rejection() {
    for device in [Device::H100, Device::B200] {
        check_wrapped_api_roundtrip_and_tamper_rejection(device);
    }
}

fn check_wrapped_api_roundtrip_and_tamper_rejection(device: Device) {
    use crate::v4::circuit::wrapper::{COMPACT_ZETA_PREAMBLE, degree_bits_offset, table_pis_offset, zeta_offset};

    let _ = env_logger::builder().format_timestamp(None).try_init();

    let mut timing = TimingTree::default();
    let (header, mut plain) = fixture_job();
    plain.job.common.device = device;
    let mut prover = Fp8Prover::setup_with_timing(plain.job.common.device, &mut timing).expect("fp8 setup");
    let setup = prover.setups.get(&plain.job.common.device).expect("setup was prebuilt");
    let (public, job, wrapped) =
        prove_wrapped_statement(setup, &header, &plain, &mut timing).expect("wrapped proving must succeed");
    let public_data = public.to_bytes();
    let proof_data = compact_proof_data(&job.system, &wrapped).expect("compact encode");

    let statement = decode_statement(&public_data).expect("public data must decode");
    let verifier = Fp8Verifier::generate(&statement, &header, &mut timing).expect("verifier-side setup");
    assert!(
        verifier.circuit.common.config.zero_knowledge,
        "the published stage must carry plonky2's ZK blinding"
    );

    let verifier_bytes = verifier.to_bytes().expect("serialize verifier setup");
    let verifier = Fp8Verifier::from_bytes(&verifier_bytes).expect("deserialize verifier setup");

    let chain = &plain.ancestor_chain;
    verifier
        .verify_block(&header, chain, &public_data, &proof_data)
        .expect("honest wrapped proof must verify");

    let mut cache = Fp8VerifierCache::default();
    cache.insert(statement.common().device, verifier.clone());
    let cache_bytes = cache.to_bytes().expect("serialize cache");
    let cache = Fp8VerifierCache::from_bytes(&cache_bytes).expect("deserialize cache");
    assert_eq!(cache.len(), 1);
    cache
        .verify_block(&header, chain, &public_data, &proof_data)
        .expect("the cached setup must verify the honest proof");
    assert!(
        Fp8VerifierCache::from_bytes(&[])
            .expect("an empty blob is a valid cache")
            .is_empty(),
        "the no-embedded-cache default must load as an empty cache"
    );

    let err = Fp8VerifierCache::default()
        .verify_block(&header, chain, &public_data, &proof_data)
        .expect_err("an uncached setup must reject, not compile");
    assert!(
        err.to_string().contains("no cached fp8 verifier setup"),
        "unexpected error: {err:#}"
    );

    let (header_k32, mut plain_k32) = fixture_job_k32();
    plain_k32.job.common.device = device;
    let (public_k32, proof_k32) = prover
        .prove_with_timing(&header_k32, &plain_k32, &mut timing)
        .expect("the prover must prove a different geometry");
    assert_eq!(prover.setups.len(), 1, "one universal setup");
    verifier
        .verify_block(&header_k32, &plain_k32.ancestor_chain, &public_k32, &proof_k32)
        .expect("one universal setup must verify every geometry");
    cache
        .verify_block(&header_k32, &plain_k32.ancestor_chain, &public_k32, &proof_k32)
        .expect("the cached setup must verify the second geometry");
    assert_eq!(cache.len(), 1, "no per-shape setups: the one entry serves both");

    assert_eq!(statement.to_bytes(), public_data);

    let mut tampered = statement.clone();
    tampered.set_hash_jackpot({
        let mut h = tampered.hash_jackpot();
        h[0] ^= 1;
        h
    });
    let bad_public = tampered.to_bytes();
    assert!(verifier.verify_block(&header, chain, &bad_public, &proof_data).is_err());

    let scale_pis = table_pis_offset(&job.system, job.system.main_table_positions()[Table::Scale as usize]);
    let iq_pis = table_pis_offset(&job.system, job.system.main_table_positions()[Table::InputQuant as usize]);
    for slot in [
        scale_pis + K_PUBLIC_INPUT,
        scale_pis + WL2_PUBLIC_INPUT,
        iq_pis + WL2_POW_PUBLIC_INPUT,
        iq_pis + B_KEY_OFFSET_PUBLIC_INPUT,
        iq_pis + OPERAND_MULT_A_PUBLIC_INPUT,
        degree_bits_offset(&job.system) + Table::InputQuant as usize,
        zeta_offset(&job.system),
        zeta_offset(&job.system) + D,
        wrapped.public_inputs.len() - 1,
    ] {
        let mut tampered = wrapped.clone();
        tampered.public_inputs[slot] += F::ONE;
        assert!(
            verify_fp8_block_wrapped(&public, &header, &tampered, &verifier, None).is_err(),
            "tampered public input slot {slot} must be rejected"
        );
    }

    let mut mutated = wrapped.clone();
    mutated.public_inputs[scale_pis + K_PUBLIC_INPUT] += F::ONE;
    assert_eq!(
        compact_proof_data(&job.system, &mutated).expect("compact encode"),
        proof_data,
        "imposed slots must not ride the compact wire"
    );
    for limb in 0..D {
        let mut tampered = wrapped.clone();
        tampered.public_inputs[zeta_offset(&job.system) + limb] += F::ONE;
        let bytes = compact_proof_data(&job.system, &tampered).expect("compact encode");
        assert!(
            verifier.verify_block(&header, chain, &public_data, &bytes).is_err(),
            "a preamble zeta shifted in limb {limb} must be rejected"
        );
    }

    let mut corrupt = proof_data.clone();
    let mid = COMPACT_ZETA_PREAMBLE + (proof_data.len() - COMPACT_ZETA_PREAMBLE) / 2;
    corrupt[mid] ^= 1;
    assert!(verifier.verify_block(&header, chain, &public_data, &corrupt).is_err());

    let mut noncanonical = proof_data.clone();
    noncanonical[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(verifier.verify_block(&header, chain, &public_data, &noncanonical).is_err());

    assert!(
        verifier
            .verify_block(&header, chain, &public_data, &proof_data[..COMPACT_ZETA_PREAMBLE])
            .is_err(),
        "a bare preamble with no proof body must be rejected"
    );

    let mut other_header = header;
    other_header.timestamp ^= 1;
    assert!(
        verifier
            .verify_block(&other_header, chain, &public_data, &proof_data)
            .is_err()
    );

    // σ_d is reached only through the carried intermediates.
    assert!(verifier.verify_block(&header, &[], &public_data, &proof_data).is_err());

    let mut trailing = public_data.clone();
    trailing.push(0);
    assert!(verifier.verify_block(&header, chain, &trailing, &proof_data).is_err());

    let mut trailing = proof_data.clone();
    trailing.push(0);
    assert!(verifier.verify_block(&header, chain, &public_data, &trailing).is_err());

    let mut trailing = verifier_bytes;
    trailing.push(0);
    assert!(Fp8Verifier::from_bytes(&trailing).is_err());
}

#[test]
#[ignore = "exceeds the safe memory budget of an 8 GiB CI runner; run explicitly on a larger machine"]
fn medium_wrapped_roundtrip() {
    let (header, plain) = fixture_job_medium();
    let mut timing = TimingTree::default();

    let prover = Fp8Prover::setup_with_timing(plain.job.common.device, &mut timing).expect("prover setup");
    let setup = prover.setups.get(&plain.job.common.device).expect("setup was prebuilt");
    let (public, _job, proof) =
        prove_wrapped_statement(setup, &header, &plain, &mut timing).expect("medium wrapped proving must succeed");
    let verifier = Fp8Verifier {
        circuit: setup.circuits.verifier_data(),
        constants_sigmas_polynomials: setup.circuits.constants_sigmas_polynomials(),
    };
    verify_fp8_block_wrapped(&public, &header, &proof, &verifier, None).expect("honest medium wrapped proof must verify");
}

/// The MoE wire-level round trip and rejection surface.
#[test]
fn moe_api_roundtrip_and_tamper_rejection() {
    let (header, plain) = fixture_job_moe();
    let mut timing = TimingTree::default();

    let (_, params) = plain.parse_proof(&header).expect("MoE fixture job must parse");
    let moe = params.moe_statement().expect("the fixture is MoE").clone();
    let job = Fp8Job::derive(&params, &header).expect("MoE jobs must derive");
    let a_noise_seed = params.noise_seeds(&header).a;
    assert_eq!(
        job.jackpot_key(),
        crate::v4::api::transcript::subkey(crate::v4::api::transcript::LABEL_JACKPOT, Some(&a_noise_seed)),
        "the MoE lottery key must be Subkey(noise_seedA, jackpot)"
    );
    assert_ne!(
        job.jackpot_key(),
        a_noise_seed,
        "lottery key is the jackpot subkey, not the raw A seed"
    );
    assert_eq!(job.hash_routing(), Some(moe.hash_routing));

    let setup = Fp8ProverSetup::build(params.common().device, &mut timing).expect("prover setup");
    let (public, proof) = zk_prove_plain_proof_fp8(&header, &plain, &setup, &mut timing).expect("MoE proving must succeed");
    let proof = Fp8Proof::from_bytes(&proof.to_bytes().expect("serialize")).expect("deserialize");

    verify_fp8_block(&public, &header, &proof, &setup, None).expect("honest MoE proof must verify");

    let blake3_pis = &proof.0.public_inputs[job.system.main_table_positions()[0]];
    assert_eq!(
        &blake3_pis[PI_HASH_ROUTING..PI_HASH_ROUTING + 8],
        &hash_to_u32_field_array(&moe.hash_routing),
        "the proven routing root must be the job's routing commitment"
    );

    let mut bad_outer = public.clone();
    bad_outer.moe_statement_mut().unwrap().i_a[0] += 1;
    assert!(
        verify_fp8_block(&bad_outer, &header, &proof, &setup, None).is_err(),
        "a forged sampled outer index must be rejected"
    );

    let mut bad_routing = public.clone();
    bad_routing.moe_statement_mut().unwrap().hash_routing[0] ^= 1;
    assert!(
        verify_fp8_block(&bad_routing, &header, &proof, &setup, None).is_err(),
        "a forged routing commitment must be rejected"
    );

    let mut bad_expert = public.clone();
    bad_expert.moe_statement_mut().unwrap().w = 0;
    assert!(
        verify_fp8_block(&bad_expert, &header, &proof, &setup, None).is_err(),
        "a proof must not verify under a different expert's statement"
    );
}

#[test]
fn hopper_moe_api_roundtrip() {
    let (header, mut plain) = fixture_job_moe();
    plain.job.common.device = Device::H100;
    let mut timing = TimingTree::default();
    let (_, params) = plain.parse_proof(&header).expect("Hopper MoE fixture must parse");
    assert_eq!(params.common().device, Device::H100);
    let setup = Fp8ProverSetup::build(Device::H100, &mut timing).expect("Hopper MoE prover setup");
    let (public, proof) =
        zk_prove_plain_proof_fp8(&header, &plain, &setup, &mut timing).expect("Hopper MoE proving must succeed");
    let proof = Fp8Proof::from_bytes(&proof.to_bytes().expect("serialize")).expect("deserialize");
    verify_fp8_block(&public, &header, &proof, &setup, None).expect("honest Hopper MoE proof must verify");
}
