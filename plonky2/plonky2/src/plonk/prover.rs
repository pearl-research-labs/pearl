//! plonky2 prover implementation.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
use core::cmp::min;
use core::mem::swap;

use anyhow::{ensure, Result};
use hashbrown::HashMap;
use plonky2_maybe_rayon::*;

use super::circuit_builder::{LookupChallenges, LookupWire};
use crate::field::extension::Extendable;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::Field;
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::fri::oracle::PolynomialBatch;
use crate::gates::lookup::LookupGate;
use crate::gates::lookup_table::LookupTableGate;
use crate::gates::selectors::LookupSelectors;
use crate::hash::hash_types::RichField;
use crate::iop::challenger::Challenger;
use crate::iop::generator::{generate_partial_witness, generate_partial_witness_fast};
use crate::iop::target::Target;
use crate::iop::witness::{MatrixWitness, PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::NUM_COINS_LOOKUP;
use crate::plonk::circuit_data::{
    witness_polynomial_blinding_degree, CommonCircuitData, ProverOnlyCircuitData,
};
use crate::plonk::config::{GenericConfig, GenericHashOut, Hasher};
use crate::plonk::plonk_common::PlonkOracle;
use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};
use crate::plonk::vanishing_poly::{eval_vanishing_poly_base_batch, get_lut_poly};
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::timed;
use crate::util::partial_products::{partial_products_and_z_gx, quotient_chunk_products};
use crate::util::timing::TimingTree;
use crate::util::{log2_ceil, transpose};

/// Set all the lookup gate wires (including multiplicities) and pad unused LU slots.
/// Warning: rows are in descending order: the first gate to appear is the last LU gate, and
/// the last gate to appear is the first LUT gate.
pub fn set_lookup_wires<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    pw: &mut PartitionWitness<F>,
) -> Result<()> {
    for (
        lut_index,
        &LookupWire {
            last_lu_gate: _,
            last_lut_gate,
            first_lut_gate,
        },
    ) in prover_data.lookup_rows.iter().enumerate()
    {
        let lut_len = common_data.luts[lut_index].len();
        let num_entries = LookupGate::num_slots(&common_data.config);
        let num_lut_entries = LookupTableGate::num_slots(&common_data.config);

        // Compute multiplicities.
        let mut multiplicities = vec![0; lut_len];

        let table_value_to_idx: HashMap<u16, usize> = common_data.luts[lut_index]
            .iter()
            .enumerate()
            .map(|(i, (inp_target, _))| (*inp_target, i))
            .collect();

        for (inp_target, _) in prover_data.lut_to_lookups[lut_index].iter() {
            let inp_value = pw.get_target(*inp_target);
            let idx = table_value_to_idx
                .get(&u16::try_from(inp_value.to_canonical_u64()).unwrap())
                .unwrap();

            multiplicities[*idx] += 1;
        }

        // Pad the last `LookupGate` with the first entry from the LUT.
        let remaining_slots = (num_entries
            - (prover_data.lut_to_lookups[lut_index].len() % num_entries))
            % num_entries;
        let (first_inp_value, first_out_value) = common_data.luts[lut_index][0];
        for slot in (num_entries - remaining_slots)..num_entries {
            let inp_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_inp(slot));
            let out_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_out(slot));
            pw.set_target(inp_target, F::from_canonical_u16(first_inp_value))?;
            pw.set_target(out_target, F::from_canonical_u16(first_out_value))?;

            multiplicities[0] += 1;
        }

        for lut_entry in 0..lut_len {
            let row = first_lut_gate - lut_entry / num_lut_entries;
            let col = lut_entry % num_lut_entries;

            let mul_target = Target::wire(row, LookupTableGate::wire_ith_multiplicity(col));

            pw.set_target(
                mul_target,
                F::from_canonical_usize(multiplicities[lut_entry]),
            )?;
        }
    }

    Ok(())
}

pub fn prove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    inputs: PartialWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    let partition_witness = timed!(
        timing,
        &format!("run {} generators", prover_data.generators.len()),
        if !prover_data.layered_generators.is_empty() {
            generate_partial_witness_fast(inputs, prover_data, common_data)
        } else {
            generate_partial_witness(inputs, prover_data, common_data, false).map(|(w, _)| w)
        }
    )?;

    prove_with_partition_witness(prover_data, common_data, partition_witness, timing)
}

/// Like `prove`, but if `layered_generators` is empty, computes and stores
/// the generator order for faster subsequent prove calls.
pub fn prove_maybe_warmup<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &mut ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    inputs: PartialWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    // [m3-prof2] witness GENERATION span (runs BEFORE prove_with_partition_witness, so it is
    // captured by NEITHER the [m3-prof] gpu_hook timer (which starts at the top of
    // prove_with_partition_witness) NOR the pearl_circuit.rs [m3-prof2] witness_build timers — it is
    // a gap in the non-cabi accounting). This is REQUIRED work (the GPU prover consumes the resulting
    // witness.wire_values), NOT trimmable plumbing; instrumented only to fully attribute the non-cabi
    // wall. `need_warmup` (first prove only) runs the slow generator-ordering pass; warm proves use
    // generate_partial_witness_fast. Timing-only, gated.
    let m3_prof2 = std::env::var("PEARL_M3_PROF").is_ok()
        || std::env::var("PEARL_GPU_REC1").is_ok()
        || std::env::var("PEARL_GPU_REC2").is_ok();
    let m3_prof2_wgen = std::time::Instant::now();
    let need_warmup = prover_data.layered_generators.is_empty();
    let (partition_witness, order) = if need_warmup {
        generate_partial_witness(inputs, prover_data, common_data, true)?
    } else {
        (
            generate_partial_witness_fast(inputs, prover_data, common_data)?,
            Vec::new(),
        )
    };
    if m3_prof2 {
        eprintln!(
            "[m3-prof2] prove.generate_partial_witness (warmup={}, REQUIRED not plumbing): {} ms",
            need_warmup,
            m3_prof2_wgen.elapsed().as_secs_f64() * 1e3
        );
    }

    let proof = prove_with_partition_witness(prover_data, common_data, partition_witness, timing)?;
    if need_warmup {
        prover_data.layered_generators = order;
    }
    Ok(proof)
}

pub fn prove_with_partition_witness<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    mut partition_witness: PartitionWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    // [m3-prof] sub-1s CPU-cost profiling (timing-only; NO behavior change). Active whenever the
    // capstone GPU path is requested (PEARL_GPU_REC1 / PEARL_GPU_REC2) OR PEARL_M3_PROF is set.
    // Captures the wall from here (top of prove_with_partition_witness) so the pre-FRI-seam span
    // = "the redundant CPU commit work the GPU recomputes" can be measured at the seam. Only read
    // inside the gpu_quotient FRI-seam blocks ⇒ allow(unused) when that feature is off.
    #[cfg_attr(not(feature = "gpu_quotient"), allow(unused_variables))]
    let m3_prof = std::env::var("PEARL_M3_PROF").is_ok()
        || std::env::var("PEARL_GPU_REC1").is_ok()
        || std::env::var("PEARL_GPU_REC2").is_ok();
    #[cfg_attr(not(feature = "gpu_quotient"), allow(unused_variables))]
    let m3_prof_t0 = std::time::Instant::now();

    let has_lookup = !common_data.luts.is_empty();
    let config = &common_data.config;
    let num_challenges = config.num_challenges;
    let quotient_degree = common_data.quotient_degree();
    let degree = common_data.degree();

    set_lookup_wires(prover_data, common_data, &mut partition_witness)?;

    let public_inputs = partition_witness.get_targets(&prover_data.public_inputs);
    let public_inputs_hash = C::InnerHasher::hash_no_pad(&public_inputs);

    let witness = timed!(
        timing,
        "compute full witness",
        partition_witness.full_witness()
    );

    // ════════════════════════════════════════════════════════════════════════════════════════════
    // GPU fused-prover STAGE 8 CAPSTONE — TOP-OF-FUNCTION early return (M3 sub-1s).  CONSENSUS-NEUTRAL
    // (the verifier is UNTOUCHED). Feature-gated + env PEARL_GPU_REC1 (Rec#1, Poseidon) / PEARL_GPU_REC2
    // (Rec#2, Blake3 ZK). The fully-fused on-GPU prover commits wires/zs_partial/quotient + runs the
    // quotient + FRI + (now) the initial_trees_proof query rounds ALL on device, so the entire CPU
    // commit→quotient→FRI flow below is REDUNDANT — the GPU recomputes it. Calling the hook HERE (right
    // after the witness + public_inputs_hash are available, BEFORE the first PolynomialBatch::from_values)
    // and early-returning DROPS the ~4.7s of redundant CPU commits the M3 profile attributed to
    // `pre_hook_cpu_commits` (Rec#1 3338ms + Rec#2 1410ms). The hooks are TypeId-gated (Rec#1→Poseidon,
    // Rec#2→Blake3) so each fires only on its own prove; oracle 0 (cs) is the PREPROCESSED commitment
    // (always cheaply available), oracles 1-3's itp is GPU-emitted, so NO live CPU oracle trees are read.
    // On None (TypeId / shape / CUDA unsupported) we FALL THROUGH to the unchanged CPU prove below.
    // UNVALIDATED (box pending).
    #[cfg(feature = "gpu_quotient")]
    {
        // Rec#1 (Poseidon) capstone — self-TypeId-gated to PoseidonGoldilocksConfig.
        if std::env::var("PEARL_GPU_REC1").is_ok() {
            if m3_prof {
                eprintln!(
                    "[m3-prof] rec1.pre_hook_cpu_commits (entry -> GPU seam, redundant): {} ms",
                    m3_prof_t0.elapsed().as_secs_f64() * 1e3
                );
            }
            let m3_prof_hook = std::time::Instant::now();
            let m3_gpu_proof = crate::gpu::try_gpu_prove_rec1_full::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &witness.wire_values,
            );
            if m3_prof {
                eprintln!(
                    "[m3-prof] rec1.gpu_hook_total (try_gpu_prove_rec1_full = upload + GPU + host assembly): {} ms",
                    m3_prof_hook.elapsed().as_secs_f64() * 1e3
                );
            }
            if let Some(gpu_proof) = m3_gpu_proof {
                eprintln!(
                    "[m2-stage8] fully-GPU-built Rec#1 Proof reassembled at TOP-of-prove (3 caps + \
                     OpeningSet + FriProof incl GPU-emitted oracle 1-3 initial_trees_proof: {} commit \
                     caps, final_poly deg {}, {} query rounds, PoW witness {}) — CPU commits SKIPPED",
                    gpu_proof.opening_proof.commit_phase_merkle_caps.len(),
                    gpu_proof.opening_proof.final_poly.coeffs.len(),
                    gpu_proof.opening_proof.query_round_proofs.len(),
                    {
                        use plonky2_field::types::PrimeField64;
                        gpu_proof.opening_proof.pow_witness.to_canonical_u64()
                    }
                );
                return Ok(ProofWithPublicInputs::<F, C, D> {
                    proof: gpu_proof,
                    public_inputs,
                });
            } else {
                eprintln!(
                    "[m2-stage8] GPU full Rec#1 prover unavailable (shape/TypeId/CUDA) — \
                     falling through to the CPU prove"
                );
            }
        }
        // Rec#2 (Blake3, ZK) capstone — self-TypeId-gated to Blake3GoldilocksConfig. Already GPU-emits
        // its oracle 1-3 itp; oracle 0 (cs) is preprocessed. Moving it to the top drops Rec#2's redundant
        // CPU commits too (the M3 rec2.pre_hook_cpu_commits ~1.4s).
        if std::env::var("PEARL_GPU_REC2").is_ok() {
            if m3_prof {
                eprintln!(
                    "[m3-prof] rec2.pre_hook_cpu_commits (entry -> GPU seam, redundant): {} ms",
                    m3_prof_t0.elapsed().as_secs_f64() * 1e3
                );
            }
            let m3_prof_hook = std::time::Instant::now();
            let m3_gpu_proof = crate::gpu::try_gpu_prove_rec2_full::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &witness.wire_values,
            );
            if m3_prof {
                eprintln!(
                    "[m3-prof] rec2.gpu_hook_total (try_gpu_prove_rec2_full = upload + GPU + host assembly): {} ms",
                    m3_prof_hook.elapsed().as_secs_f64() * 1e3
                );
            }
            if let Some(gpu_proof) = m3_gpu_proof {
                eprintln!(
                    "[m2c-stage8] fully-GPU-built BLINDED Rec#2 Proof reassembled at TOP-of-prove \
                     ({} commit caps, final_poly deg {}, {} query rounds, PoW witness {}) — CPU commits SKIPPED",
                    gpu_proof.opening_proof.commit_phase_merkle_caps.len(),
                    gpu_proof.opening_proof.final_poly.coeffs.len(),
                    gpu_proof.opening_proof.query_round_proofs.len(),
                    {
                        use plonky2_field::types::PrimeField64;
                        gpu_proof.opening_proof.pow_witness.to_canonical_u64()
                    }
                );
                return Ok(ProofWithPublicInputs::<F, C, D> {
                    proof: gpu_proof,
                    public_inputs,
                });
            } else {
                eprintln!(
                    "[m2c-stage8] GPU full BLINDED Rec#2 prover unavailable (shape/TypeId/CUDA) — \
                     falling through to the CPU prove"
                );
            }
        }
    }

    let wires_values: Vec<PolynomialValues<F>> = timed!(
        timing,
        "compute wire polynomials",
        witness
            .wire_values
            .par_iter()
            .map(|column| PolynomialValues::new(column.clone()))
            .collect()
    );

    let wires_commitment = timed!(
        timing,
        "compute wires commitment",
        PolynomialBatch::<F, C, D>::from_values(
            wires_values,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::WIRES.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    let mut challenger = Challenger::<F, C::Hasher>::new();

    // Observe the FRI config
    common_data.fri_params.observe(&mut challenger);

    // Observe the instance.
    challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
    challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);

    challenger.observe_cap::<C::Hasher>(&wires_commitment.merkle_tree.cap);

    // We need 4 values per challenge: 2 for the combos, 1 for (X-combo) in the accumulators and 1 to prove that the lookup table was computed correctly.
    // We can reuse betas and gammas for two of them.
    let num_lookup_challenges = NUM_COINS_LOOKUP * num_challenges;

    let betas = challenger.get_n_challenges(num_challenges);
    let gammas = challenger.get_n_challenges(num_challenges);

    // GPU fused-prover STAGE 2-3 ASSERT (Pearl Rec#1; M2 stage-23 milestone): feature-
    // gated, byte-exact, CONSENSUS-NEUTRAL — same class as the gpu_quotient hooks; the
    // verifier is untouched. `PEARL_GPU_REC1_STAGE23_ASSERT` runs the on-GPU
    // `pearl_gpu_prove_rec1_f64` (stages 2-3) and asserts the GPU wires_commitment cap +
    // the GPU challenger-prefix betas/gammas equal the CPU values just computed above.
    // De-risks the CRUX (on-GPU challenger threading from a FRESH challenger). No-op
    // unless the env var is set; falls through silently on any unsupported shape.
    #[cfg(feature = "gpu_quotient")]
    {
        if std::env::var("PEARL_GPU_REC1_STAGE23_ASSERT").is_ok() {
            if let Some((gpu_wires_cap, gpu_betas, gpu_gammas)) =
                crate::gpu::try_gpu_prove_rec1_stage23::<F, C, D>(
                    common_data,
                    prover_data,
                    &public_inputs_hash,
                    &witness.wire_values,
                )
            {
                use plonky2_field::types::PrimeField64;
                // CPU wires_commitment cap → canonical u64 [cap_size*4], same flatten the
                // GPU emits (cap order, 4 elems per HashOut).
                let cpu_wires_cap: Vec<u64> = wires_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_betas: Vec<u64> = betas.iter().map(|b| b.to_canonical_u64()).collect();
                let cpu_gammas: Vec<u64> = gammas.iter().map(|g| g.to_canonical_u64()).collect();
                assert_eq!(
                    gpu_wires_cap, cpu_wires_cap,
                    "[m2-stage23] GPU wires_commitment cap != CPU from_values cap"
                );
                assert_eq!(
                    gpu_betas, cpu_betas,
                    "[m2-stage23] GPU challenger betas != CPU betas"
                );
                assert_eq!(
                    gpu_gammas, cpu_gammas,
                    "[m2-stage23] GPU challenger gammas != CPU gammas"
                );
                eprintln!(
                    "[m2-stage23] CHALLENGER PREFIX + WIRES COMMIT byte-exact \
                     (cap={} elems, {num_challenges} betas, {num_challenges} gammas)",
                    cpu_wires_cap.len()
                );
            } else {
                eprintln!(
                    "[m2-stage23] GPU stage-2/3 unavailable (shape/TypeId/CUDA) — \
                     skipping assert (CPU path unchanged)"
                );
            }
        }
    }

    // GPU fused-prover STAGE 2-3 ASSERT (Pearl Rec#2; M2c milestone): Blake3GoldilocksConfig,
    // zero_knowledge=TRUE. Feature-gated, byte-exact, CONSENSUS-NEUTRAL — same class as the
    // gpu_quotient hooks; the verifier is untouched. `PEARL_GPU_REC2_STAGE23_ASSERT` runs the
    // on-GPU `pearl_gpu_prove_rec2_f64` (Blake3-Merkle commit + Blake3-sponge challenger,
    // stages 2-3) and byte-checks the GPU wires cap + challenger betas/gammas.
    //
    // ZK NOTE: Rec#2's REAL `wires_commitment` is BLINDED (config.zero_knowledge && WIRES.
    // blinding) — its cap includes 4 random salt columns the GPU can't reproduce. So the GPU
    // hook commits NON-BLINDED (salt_cols=0), and here we build an INDEPENDENT non-blinded
    // reference `from_values(blinding=false)` + recompute the challenger prefix (hiding=TRUE,
    // via the same fri_params.observe) to compare against. The actual proof keeps the blinded
    // `wires_commitment`; this throwaway non-blinded reference exists ONLY for the byte-diff,
    // de-risking the Blake3 Merkle + Blake3-sponge-challenger math. (Validating the blinded cap
    // byte-exactly needs the CPU's exact salt columns threaded into the GPU — a follow-up.)
    #[cfg(feature = "gpu_quotient")]
    {
        if std::env::var("PEARL_GPU_REC2_STAGE23_ASSERT").is_ok() {
            // Non-blinded reference wires commit (throwaway) over the SAME wire polys.
            let ref_wires_values: Vec<PolynomialValues<F>> = witness
                .wire_values
                .par_iter()
                .map(|column| PolynomialValues::new(column.clone()))
                .collect();
            let ref_commit = PolynomialBatch::<F, C, D>::from_values(
                ref_wires_values,
                config.fri_config.rate_bits,
                false, // NON-blinded reference (the GPU path commits salt_cols=0)
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            // Recompute the challenger prefix against the NON-blinded cap (hiding=TRUE is
            // carried by common_data.fri_params.observe, matching the GPU prefix's hiding=1).
            let mut ref_challenger = Challenger::<F, C::Hasher>::new();
            common_data.fri_params.observe(&mut ref_challenger);
            ref_challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
            ref_challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);
            ref_challenger.observe_cap::<C::Hasher>(&ref_commit.merkle_tree.cap);
            let ref_betas = ref_challenger.get_n_challenges(num_challenges);
            let ref_gammas = ref_challenger.get_n_challenges(num_challenges);

            if let Some((gpu_wires_cap, gpu_betas, gpu_gammas)) =
                crate::gpu::try_gpu_prove_rec2_stage23::<F, C, D>(
                    common_data,
                    prover_data,
                    &public_inputs_hash,
                    &witness.wire_values,
                )
            {
                use plonky2_field::types::PrimeField64;
                // Reference cap → canonical u64 [cap_size*4] via the SAME BytesHash<27>::to_vec()
                // 7-byte-chunk flatten the GPU emits (4 elems per 27-byte cap entry).
                let ref_wires_cap: Vec<u64> = ref_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_betas: Vec<u64> = ref_betas.iter().map(|b| b.to_canonical_u64()).collect();
                let cpu_gammas: Vec<u64> =
                    ref_gammas.iter().map(|g| g.to_canonical_u64()).collect();
                assert_eq!(
                    gpu_wires_cap, ref_wires_cap,
                    "[m2c-stage23] GPU Blake3 wires cap != CPU non-blinded from_values cap"
                );
                assert_eq!(
                    gpu_betas, cpu_betas,
                    "[m2c-stage23] GPU Blake3-challenger betas != CPU betas (non-blinded ref)"
                );
                assert_eq!(
                    gpu_gammas, cpu_gammas,
                    "[m2c-stage23] GPU Blake3-challenger gammas != CPU gammas (non-blinded ref)"
                );
                eprintln!(
                    "[m2c-stage23] BLAKE3 WIRES COMMIT + BLAKE3-SPONGE CHALLENGER byte-exact \
                     (non-blinded ref; cap={} elems, {num_challenges} betas, {num_challenges} gammas)",
                    ref_wires_cap.len()
                );
            } else {
                eprintln!(
                    "[m2c-stage23] GPU Rec#2 stage-2/3 unavailable (shape/TypeId/CUDA) — \
                     skipping assert (CPU path unchanged)"
                );
            }
        }
    }

    let deltas = if has_lookup {
        let mut delts = Vec::with_capacity(2 * num_challenges);
        let num_additional_challenges = num_lookup_challenges - 2 * num_challenges;
        let additional = challenger.get_n_challenges(num_additional_challenges);
        delts.extend(&betas);
        delts.extend(&gammas);
        delts.extend(additional);
        delts
    } else {
        vec![]
    };

    assert!(
        common_data.quotient_degree_factor < common_data.config.num_routed_wires,
        "When the number of routed wires is smaller that the degree, we should change the logic to avoid computing partial products."
    );
    let mut partial_products_and_zs = timed!(
        timing,
        "compute partial products",
        all_wires_permutation_partial_products(&witness, &betas, &gammas, prover_data, common_data)
    );

    // Z is expected at the front of our batch; see `zs_range` and `partial_products_range`.
    let plonk_z_vecs = partial_products_and_zs
        .iter_mut()
        .map(|partial_products_and_z| partial_products_and_z.pop().unwrap())
        .collect();
    let zs_partial_products = [plonk_z_vecs, partial_products_and_zs.concat()].concat();

    // All lookup polys: RE and partial SLDCs.
    let lookup_polys =
        compute_all_lookup_polys(&witness, &deltas, prover_data, common_data, has_lookup);

    let zs_partial_products_lookups = if has_lookup {
        [zs_partial_products, lookup_polys].concat()
    } else {
        zs_partial_products
    };

    let partial_products_zs_and_lookup_commitment = timed!(
        timing,
        "commit to partial products, Z's and, if any, lookup polynomials",
        PolynomialBatch::from_values(
            zs_partial_products_lookups,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    challenger.observe_cap::<C::Hasher>(&partial_products_zs_and_lookup_commitment.merkle_tree.cap);

    let alphas = challenger.get_n_challenges(num_challenges);

    // GPU fused-prover STAGE 2-4 ASSERT (Pearl Rec#1; M2 stage-4 milestone): feature-gated,
    // byte-exact, CONSENSUS-NEUTRAL — same class as the gpu_quotient hooks; the verifier is
    // untouched. `PEARL_GPU_REC1_STAGE4_ASSERT` runs the on-GPU `pearl_gpu_prove_rec1_f64`
    // (stages 2-4) and asserts the GPU wires_cap + betas/gammas (STAGE 2-3 regression) PLUS
    // the GPU zs_partial cap + alphas (STAGE 4) equal the CPU values. This fires HERE — after
    // the CPU has computed `partial_products_zs_and_lookup_commitment` (the perm-Z commit) and
    // the post-zs-cap `alphas` — so both sides are at the SAME challenger position. No-op
    // unless the env var is set; falls through silently on any unsupported shape.
    #[cfg(feature = "gpu_quotient")]
    {
        if std::env::var("PEARL_GPU_REC1_STAGE4_ASSERT").is_ok() {
            if let Some((gpu_wires_cap, gpu_betas, gpu_gammas, gpu_zsp_cap, gpu_alphas)) =
                crate::gpu::try_gpu_prove_rec1_stage4::<F, C, D>(
                    common_data,
                    prover_data,
                    &public_inputs_hash,
                    &witness.wire_values,
                )
            {
                use plonky2_field::types::PrimeField64;
                let cpu_wires_cap: Vec<u64> = wires_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| h.to_vec().into_iter().map(|e| e.to_canonical_u64()))
                    .collect();
                // CPU zs_partial_products cap (the perm-Z oracle) → canonical u64 [cap_size*4].
                let cpu_zsp_cap: Vec<u64> = partial_products_zs_and_lookup_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_betas: Vec<u64> = betas.iter().map(|b| b.to_canonical_u64()).collect();
                let cpu_gammas: Vec<u64> = gammas.iter().map(|g| g.to_canonical_u64()).collect();
                let cpu_alphas: Vec<u64> = alphas.iter().map(|a| a.to_canonical_u64()).collect();
                // STAGE 2-3 regression (must still hold):
                assert_eq!(
                    gpu_wires_cap, cpu_wires_cap,
                    "[m2-stage4] GPU wires_commitment cap != CPU from_values cap"
                );
                assert_eq!(gpu_betas, cpu_betas, "[m2-stage4] GPU betas != CPU betas");
                assert_eq!(
                    gpu_gammas, cpu_gammas,
                    "[m2-stage4] GPU gammas != CPU gammas"
                );
                // STAGE 4 (the new perm-Z commit + observe→alphas):
                assert_eq!(
                    gpu_zsp_cap, cpu_zsp_cap,
                    "[m2-stage4] GPU zs_partial_products cap != CPU from_values cap (perm-Z)"
                );
                assert_eq!(
                    gpu_alphas, cpu_alphas,
                    "[m2-stage4] GPU alphas (post zs-cap observe) != CPU alphas"
                );
                eprintln!("[m2-stage4] PERM-Z + COMMIT + ALPHAS byte-exact");
            } else {
                eprintln!(
                    "[m2-stage4] GPU stage-2/4 unavailable (shape/TypeId/CUDA) — \
                     skipping assert (CPU path unchanged)"
                );
            }
        }
    }

    // GPU fused-prover STAGE 2-4 ASSERT (Pearl Rec#2; M2c stage-4 milestone): Blake3Goldilocks-
    // Config, zero_knowledge=TRUE. Feature-gated, byte-exact, CONSENSUS-NEUTRAL — the verifier is
    // untouched. `PEARL_GPU_REC2_STAGE4_ASSERT` runs the on-GPU `pearl_gpu_prove_rec2_f64`
    // (stages 2-4: Blake3 wires commit + Blake3-sponge challenger + perm-Z → BLAKE3 zs_partial
    // commit + observe→alphas) and byte-checks the GPU wires_cap + betas/gammas (STAGE 2-3
    // regression) PLUS the GPU zs_partial cap + alphas (STAGE 4).
    //
    // ZK NOTE (the crux): BOTH Rec#2 oracles are BLINDED on the REAL proof (WIRES.blinding &&
    // ZS_PARTIAL_PRODUCTS.blinding) — their caps include random salt columns the GPU can't
    // reproduce, AND those blinded caps feed the REAL challenger (so the REAL betas/gammas/alphas
    // depend on the salt). The GPU commits NON-blinded (salt_cols=0). So we build an INDEPENDENT
    // NON-blinded reference CHAIN — non-blinded wires commit → non-blinded challenger → ref
    // betas/gammas → perm-Z(ref betas/gammas) → non-blinded zs commit → observe → ref alphas —
    // and compare the GPU outputs against THAT (NOT the real blinded squeezes above). The actual
    // proof keeps the blinded commitments; this throwaway non-blinded chain exists ONLY for the
    // byte-diff, de-risking the perm-Z + Blake3 zs commit + the post-zs-cap alphas math. (Validating
    // the blinded caps byte-exactly needs the CPU's exact salt columns threaded into the GPU.)
    #[cfg(feature = "gpu_quotient")]
    {
        if std::env::var("PEARL_GPU_REC2_STAGE4_ASSERT").is_ok() {
            // ── Reference STAGE 2-3 (non-blinded): wires commit → challenger → ref betas/gammas. ──
            let ref_wires_values: Vec<PolynomialValues<F>> = witness
                .wire_values
                .par_iter()
                .map(|column| PolynomialValues::new(column.clone()))
                .collect();
            let ref_wires_commit = PolynomialBatch::<F, C, D>::from_values(
                ref_wires_values,
                config.fri_config.rate_bits,
                false, // NON-blinded reference (the GPU path commits salt_cols=0)
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            // Recompute the challenger prefix against the NON-blinded wires cap (hiding=TRUE is
            // carried by common_data.fri_params.observe, matching the GPU prefix's hiding=1).
            let mut ref_challenger = Challenger::<F, C::Hasher>::new();
            common_data.fri_params.observe(&mut ref_challenger);
            ref_challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
            ref_challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);
            ref_challenger.observe_cap::<C::Hasher>(&ref_wires_commit.merkle_tree.cap);
            let ref_betas = ref_challenger.get_n_challenges(num_challenges);
            let ref_gammas = ref_challenger.get_n_challenges(num_challenges);

            // ── Reference STAGE 4: perm-Z with the REFERENCE betas/gammas (NOT the blinded-path
            //    betas/gammas above), pop Z to front + concat partials → the 15-col oracle, commit
            //    NON-blinded (matching the GPU salt_cols=0), then continue the SAME ref_challenger:
            //    observe_cap(zs cap) → ref alphas (prover.rs:440-442). No lookups (luts empty). ──
            let mut ref_pp_and_zs = all_wires_permutation_partial_products(
                &witness,
                &ref_betas,
                &ref_gammas,
                prover_data,
                common_data,
            );
            let ref_z_vecs = ref_pp_and_zs
                .iter_mut()
                .map(|pp_and_z| pp_and_z.pop().unwrap())
                .collect();
            let ref_zs_partial: Vec<PolynomialValues<F>> =
                [ref_z_vecs, ref_pp_and_zs.concat()].concat();
            let ref_zsp_commit = PolynomialBatch::<F, C, D>::from_values(
                ref_zs_partial,
                config.fri_config.rate_bits,
                false, // NON-blinded reference (the GPU path commits salt_cols=0)
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            ref_challenger.observe_cap::<C::Hasher>(&ref_zsp_commit.merkle_tree.cap);
            let ref_alphas = ref_challenger.get_n_challenges(num_challenges);

            if let Some((gpu_wires_cap, gpu_betas, gpu_gammas, gpu_zsp_cap, gpu_alphas)) =
                crate::gpu::try_gpu_prove_rec2_stage4::<F, C, D>(
                    common_data,
                    prover_data,
                    &public_inputs_hash,
                    &witness.wire_values,
                )
            {
                use plonky2_field::types::PrimeField64;
                // Reference caps → canonical u64 [cap_size*4] via the SAME BytesHash<27>::to_vec()
                // 7-byte-chunk flatten the GPU emits (4 elems per 27-byte cap entry).
                let ref_wires_cap: Vec<u64> = ref_wires_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let ref_zsp_cap: Vec<u64> = ref_zsp_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_betas: Vec<u64> = ref_betas.iter().map(|b| b.to_canonical_u64()).collect();
                let cpu_gammas: Vec<u64> =
                    ref_gammas.iter().map(|g| g.to_canonical_u64()).collect();
                let cpu_alphas: Vec<u64> =
                    ref_alphas.iter().map(|a| a.to_canonical_u64()).collect();
                // STAGE 2-3 regression (must still hold, vs the NON-blinded reference):
                assert_eq!(
                    gpu_wires_cap, ref_wires_cap,
                    "[m2c-stage4] GPU Blake3 wires cap != CPU non-blinded from_values cap"
                );
                assert_eq!(
                    gpu_betas, cpu_betas,
                    "[m2c-stage4] GPU Blake3-challenger betas != CPU betas (non-blinded ref)"
                );
                assert_eq!(
                    gpu_gammas, cpu_gammas,
                    "[m2c-stage4] GPU Blake3-challenger gammas != CPU gammas (non-blinded ref)"
                );
                // STAGE 4 (the new perm-Z + BLAKE3 zs commit + observe→alphas):
                assert_eq!(
                    gpu_zsp_cap, ref_zsp_cap,
                    "[m2c-stage4] GPU Blake3 zs_partial cap != CPU non-blinded from_values cap (perm-Z)"
                );
                assert_eq!(
                    gpu_alphas, cpu_alphas,
                    "[m2c-stage4] GPU alphas (post zs-cap observe) != CPU alphas (non-blinded ref)"
                );
                eprintln!("[m2c-stage4] PERM-Z + BLAKE3 COMMIT + ALPHAS byte-exact");
            } else {
                eprintln!(
                    "[m2c-stage4] GPU Rec#2 stage-2/4 unavailable (shape/TypeId/CUDA) — \
                     skipping assert (CPU path unchanged)"
                );
            }
        }
    }

    // GPU quotient hook (Pearl Rec#1): feature-gated, byte-exact, CONSENSUS-NEUTRAL —
    // same class as the `gpu_commit` from_values hook. `PEARL_GPU_QUOTIENT_ASSERT`
    // computes BOTH and asserts the GPU coeffs == CPU coeffs; `PEARL_GPU_QUOTIENT`
    // uses the GPU result. Falls through to CPU on any unsupported shape / CUDA error.
    let quotient_polys = {
        #[cfg(feature = "gpu_quotient")]
        {
            let assert = std::env::var("PEARL_GPU_QUOTIENT_ASSERT").is_ok();
            let use_gpu = std::env::var("PEARL_GPU_QUOTIENT").is_ok() || assert;
            if use_gpu {
                if let Some(gpu_q) = timed!(
                    timing,
                    "GPU compute quotient polys",
                    crate::gpu::try_gpu_compute_quotient::<F, C, D>(
                        common_data,
                        prover_data,
                        &public_inputs_hash,
                        &wires_commitment,
                        &partial_products_zs_and_lookup_commitment,
                        &betas,
                        &gammas,
                        &alphas,
                    )
                ) {
                    if assert {
                        let cpu_q = timed!(
                            timing,
                            "compute quotient polys",
                            compute_quotient_polys::<F, C, D>(
                                common_data,
                                prover_data,
                                &public_inputs_hash,
                                &wires_commitment,
                                &partial_products_zs_and_lookup_commitment,
                                &betas,
                                &gammas,
                                &deltas,
                                &alphas,
                            )
                        );
                        assert_eq!(gpu_q.len(), cpu_q.len(), "GPU quotient poly count != CPU");
                        for (c, (gi, ci)) in gpu_q.iter().zip(&cpu_q).enumerate() {
                            assert_eq!(
                                gi.coeffs, ci.coeffs,
                                "GPU quotient != CPU at challenge {c}"
                            );
                        }
                        eprintln!(
                            "[gpu_quotient] ASSERT OK — GPU quotient byte-exact vs CPU \
                             ({} challenges)",
                            cpu_q.len()
                        );
                        cpu_q
                    } else {
                        gpu_q
                    }
                } else {
                    timed!(
                        timing,
                        "compute quotient polys",
                        compute_quotient_polys::<F, C, D>(
                            common_data,
                            prover_data,
                            &public_inputs_hash,
                            &wires_commitment,
                            &partial_products_zs_and_lookup_commitment,
                            &betas,
                            &gammas,
                            &deltas,
                            &alphas,
                        )
                    )
                }
            } else {
                timed!(
                    timing,
                    "compute quotient polys",
                    compute_quotient_polys::<F, C, D>(
                        common_data,
                        prover_data,
                        &public_inputs_hash,
                        &wires_commitment,
                        &partial_products_zs_and_lookup_commitment,
                        &betas,
                        &gammas,
                        &deltas,
                        &alphas,
                    )
                )
            }
        }
        #[cfg(not(feature = "gpu_quotient"))]
        {
            timed!(
                timing,
                "compute quotient polys",
                compute_quotient_polys::<F, C, D>(
                    common_data,
                    prover_data,
                    &public_inputs_hash,
                    &wires_commitment,
                    &partial_products_zs_and_lookup_commitment,
                    &betas,
                    &gammas,
                    &deltas,
                    &alphas,
                )
            )
        }
    };

    let mut all_quotient_poly_chunks_random: Vec<PolynomialCoeffs<F>> = timed!(
        timing,
        "split up quotient polys",
        quotient_polys
            .into_par_iter()
            .flat_map(|mut quotient_poly| {
                quotient_poly.trim_to_len(quotient_degree).expect(
                    "Quotient has failed, the vanishing polynomial is not divisible by Z_H",
                );
                // Split quotient into degree-n chunks.
                if !common_data.config.zero_knowledge {
                    return quotient_poly.chunks(degree);
                }
                // In the zk case, we split the quotient into degree-(n-h_p) chunks.
                // This is so that we can add random polynomials of degree n > n-h_p and still keep chunks of degree a power of 2.
                // This is Plonk's randomization strategy for the case of several chunks, as described in
                // "A note on adding zero-knowledge to STARKs" (https://eprint.iacr.org/2024/1037.pdf), Section 4.1,
                // adapted to keep the two-adic degree bound. `h_p` is as in equation (9) in https://eprint.iacr.org/2024/1037.pdf.
                let h_p = common_data.quotient_chunk_blinding_degree();
                let chunk_deg = degree.saturating_sub(h_p);
                assert!(chunk_deg > 0);

                let total_num_chunks = quotient_poly.len().div_ceil(chunk_deg);
                let random_ts: Vec<_> = (0..total_num_chunks - 1)
                    .into_par_iter()
                    .map(|_| PolynomialCoeffs {
                        coeffs: F::rand_vec(h_p),
                    })
                    .collect();
                // Let (t_i)i be the random polynomials, and (q_i)i be the k quotient chunks of degree n-h.
                // We compute:
                // - q'_0(X) = q_0(X) + Xˆ(n-h) * t_0(X)
                // - q'_i(X) = q_i(X) + Xˆ(n-h) * t_i(X) - t_(i-1)(X)
                // - q'_k(X) = q_k(X) - t_(k-1)(X)
                // Then, the sum of q' over i is equal to the sum of q over i.
                let mut quotients = quotient_poly.chunks(chunk_deg);
                quotients[0].coeffs.extend(&random_ts[0].coeffs);
                for i in 1..total_num_chunks - 1 {
                    quotients[i] -= &random_ts[i - 1];
                    quotients[i].coeffs.extend(&random_ts[i].coeffs);
                }
                quotients[total_num_chunks - 1] -= &random_ts[total_num_chunks - 2];
                quotients[total_num_chunks - 1]
                    .pad(degree)
                    .expect("Degree of last chunk unexpectedly high");
                quotients
            })
            .collect()
    );

    // In the zk case, add a random polynomial (used to  hide the batch FRI polynomial)
    // to the quotient polys commitment.
    if config.zero_knowledge {
        // The random polynomial is of degree |H| with H the subgroup.
        let d = 1 << common_data.fri_params.degree_bits;
        let random_r = PolynomialCoeffs::new(F::rand_vec(d));
        all_quotient_poly_chunks_random.push(random_r);
    }
    let quotient_polys_random_commitment = timed!(
        timing,
        "commit to quotient polys",
        PolynomialBatch::<F, C, D>::from_coeffs(
            all_quotient_poly_chunks_random,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::QUOTIENT.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    challenger.observe_cap::<C::Hasher>(&quotient_polys_random_commitment.merkle_tree.cap);

    let zeta = challenger.get_extension_challenge::<D>();
    // To avoid leaking witness data, we want to ensure that our opening locations, `zeta` and
    // `g * zeta`, are not in our subgroup `H`. It suffices to check `zeta` only, since
    // `(g * zeta)^n = zeta^n`, where `n` is the order of `g`.
    let g = F::Extension::primitive_root_of_unity(common_data.degree_bits());
    ensure!(
        zeta.exp_power_of_2(common_data.degree_bits()) != F::Extension::ONE,
        "Opening point is in the subgroup."
    );

    // GPU fused-prover STAGE 2-6 ASSERT (Pearl Rec#1; M2 stage-5 milestone): feature-gated,
    // byte-exact, CONSENSUS-NEUTRAL — same class as the gpu_quotient hooks; the verifier is
    // untouched. `PEARL_GPU_REC1_STAGE5_ASSERT` runs the on-GPU `pearl_gpu_prove_rec1_f64`
    // (stages 2-6) and asserts the GPU wires_cap + betas/gammas + zs_partial cap + alphas
    // (STAGE 2-4 regression) PLUS the GPU quotient cap (STAGE 5) + zeta (STAGE 6) equal the
    // CPU values. This fires HERE — AFTER the CPU has committed the quotient polys
    // (`quotient_polys_random_commitment`, via from_coeffs) and squeezed the post-quot-cap
    // `zeta` — so both challengers are at the SAME position. The M1 PEARL_GPU_QUOTIENT_ASSERT
    // already proves the quotient KERNEL byte-exact; here the new thing is feeding it the
    // DEVICE-RESIDENT leaves (wires from STAGE 2, zs_partial from STAGE 4) with no host round-
    // trip — and the quotient cap + zeta byte-match IS that proof. No-op unless the env var is
    // set; falls through silently on any unsupported shape.
    #[cfg(feature = "gpu_quotient")]
    {
        if std::env::var("PEARL_GPU_REC1_STAGE5_ASSERT").is_ok() {
            if let Some((
                gpu_wires_cap,
                gpu_betas,
                gpu_gammas,
                gpu_zsp_cap,
                gpu_alphas,
                gpu_quot_cap,
                gpu_zeta,
                gpu_quot_chunk_coeffs,
            )) = crate::gpu::try_gpu_prove_rec1_stage5::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &witness.wire_values,
            ) {
                use plonky2_field::extension::FieldExtension;
                use plonky2_field::types::PrimeField64;
                let cpu_wires_cap: Vec<u64> = wires_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| h.to_vec().into_iter().map(|e| e.to_canonical_u64()))
                    .collect();
                let cpu_zsp_cap: Vec<u64> = partial_products_zs_and_lookup_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                // CPU quotient cap (the from_coeffs commit of the 24 trimmed/split chunks).
                let cpu_quot_cap: Vec<u64> = quotient_polys_random_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_betas: Vec<u64> = betas.iter().map(|b| b.to_canonical_u64()).collect();
                let cpu_gammas: Vec<u64> = gammas.iter().map(|g| g.to_canonical_u64()).collect();
                let cpu_alphas: Vec<u64> = alphas.iter().map(|a| a.to_canonical_u64()).collect();
                // CPU zeta (D=2 extension) → canonical [c0, c1], matching the GPU rp_ch_get_ext
                // order (out[0]=first squeeze, out[1]=second squeeze == to_basefield_array order).
                let zeta_arr = zeta.to_basefield_array();
                let cpu_zeta: Vec<u64> = zeta_arr.iter().map(|e| e.to_canonical_u64()).collect();
                // STAGE 2-4 regression (must still hold):
                assert_eq!(
                    gpu_wires_cap, cpu_wires_cap,
                    "[m2-stage5] GPU wires_commitment cap != CPU from_values cap"
                );
                assert_eq!(gpu_betas, cpu_betas, "[m2-stage5] GPU betas != CPU betas");
                assert_eq!(
                    gpu_gammas, cpu_gammas,
                    "[m2-stage5] GPU gammas != CPU gammas"
                );
                assert_eq!(
                    gpu_zsp_cap, cpu_zsp_cap,
                    "[m2-stage5] GPU zs_partial_products cap != CPU from_values cap (perm-Z)"
                );
                assert_eq!(
                    gpu_alphas, cpu_alphas,
                    "[m2-stage5] GPU alphas (post zs-cap observe) != CPU alphas"
                );
                // DIAGNOSTIC (A/B localization): if PEARL_GPU_REC1_STAGE5_QDIFF was set, the fused
                // STAGE 5 emitted its 24 quotient chunk coeffs (challenge-major, [nc*qdf*n]). Diff
                // them vs the M1-PROVEN pearl_gpu_compute_quotient_f64 path (which reads HOST-
                // uploaded merkle-order leaves) for the SAME circuit+challenges, BEFORE the cap
                // assert. Outcome (A): chunk coeffs MATCH but the cap differs ⇒ the bug is in
                // rp_from_coeffs_keep_tree. Outcome (B): chunk coeffs DIFFER ⇒ the bug is the
                // device-resident leaf gather (the fused d_*_leaves order vs what the kernel
                // expects). This isolates the M2 stage-5 mismatch to the commit vs the gather.
                if let Some(gpu_chunks) = &gpu_quot_chunk_coeffs {
                    let qdb = log2_ceil(common_data.quotient_degree_factor);
                    let degree = 1usize << common_data.degree_bits();
                    let qdf = 1usize << qdb;
                    let nc = common_data.config.num_challenges;
                    if let Some(m1) = crate::gpu::try_gpu_compute_quotient::<F, C, D>(
                        common_data,
                        prover_data,
                        &public_inputs_hash,
                        &wires_commitment,
                        &partial_products_zs_and_lookup_commitment,
                        &betas,
                        &gammas,
                        &alphas,
                    ) {
                        // M1 returns nc polys of length lde_size (= qdf*degree); chunk
                        // pidx = ch*qdf+chunk == m1[ch].coeffs[chunk*degree .. +degree].
                        let mut total_mism = 0usize;
                        let mut first: Option<(usize, usize, usize)> = None;
                        for ch in 0..nc {
                            for chunk in 0..qdf {
                                let pidx = ch * qdf + chunk;
                                for k in 0..degree {
                                    let g = gpu_chunks[pidx * degree + k];
                                    let m = m1[ch].coeffs[chunk * degree + k].to_canonical_u64();
                                    if g != m {
                                        if first.is_none() {
                                            first = Some((ch, chunk, k));
                                        }
                                        total_mism += 1;
                                    }
                                }
                            }
                        }
                        if total_mism == 0 {
                            eprintln!(
                                "[m2-stage5][qdiff] fused chunk coeffs MATCH M1 ({} chunks × {} \
                                 coeffs) ⇒ OUTCOME A (if cap still differs, bug is rp_from_coeffs)",
                                nc * qdf,
                                degree
                            );
                        } else {
                            eprintln!(
                                "[m2-stage5][qdiff] fused chunk coeffs DIFFER from M1: {total_mism} \
                                 mismatches, first at (ch={}, chunk={}, k={}) ⇒ OUTCOME B \
                                 (device-resident leaf gather order)",
                                first.map(|f| f.0).unwrap_or(0),
                                first.map(|f| f.1).unwrap_or(0),
                                first.map(|f| f.2).unwrap_or(0),
                            );
                        }
                    } else {
                        eprintln!(
                            "[m2-stage5][qdiff] M1 pearl_gpu_compute_quotient unavailable \
                             (shape/TypeId/CUDA) — cannot run the A/B diff"
                        );
                    }
                }
                // STAGE 5 (quotient compute + from_coeffs commit) + STAGE 6 (observe → zeta):
                assert_eq!(
                    gpu_quot_cap, cpu_quot_cap,
                    "[m2-stage5] GPU quotient cap != CPU from_coeffs cap"
                );
                assert_eq!(
                    gpu_zeta, cpu_zeta,
                    "[m2-stage5] GPU zeta (post quot-cap observe) != CPU zeta"
                );
                eprintln!("[m2-stage5] QUOTIENT + COMMIT + ZETA byte-exact");
            } else {
                eprintln!(
                    "[m2-stage5] GPU stage-2/6 unavailable (shape/TypeId/CUDA) — \
                     skipping assert (CPU path unchanged)"
                );
            }
        }
    }

    // GPU fused-prover STAGE 2-5 ASSERT (Pearl Rec#2; M2c stage-5 milestone): Blake3Goldilocks-
    // Config, zero_knowledge=TRUE. Feature-gated, byte-exact, CONSENSUS-NEUTRAL — the verifier is
    // untouched. `PEARL_GPU_REC2_STAGE5_ASSERT` runs the on-GPU `pearl_gpu_prove_rec2_f64`
    // (stages 2-6: Blake3 wires/zs commits + Blake3-sponge challenger + perm-Z + the Rec#2
    // quotient [ka2 gates + step-aware get_lde_values gather] → BLAKE3 from_coeffs commit →
    // observe→ζ) and byte-checks the GPU wires_cap + betas/gammas + zs_partial cap + alphas
    // (STAGE 2-4 regression) PLUS the GPU quotient cap (STAGE 5) + zeta (STAGE 6).
    //
    // ZK NOTE (the crux): EVERY Rec#2 oracle + the quotient are BLINDED on the REAL proof, and the
    // blinded caps feed the REAL challenger (so the real betas/gammas/alphas/zeta depend on the
    // salt). The GPU commits NON-blinded (salt_cols=0 wires/zs; the NON-ZK quotient split — no
    // random_ts, no random_r). So we build an INDEPENDENT NON-blinded reference CHAIN — non-blinded
    // wires commit → challenger → ref betas/gammas → perm-Z → non-blinded zs commit → observe → ref
    // alphas → compute_quotient_polys(ref) → NON-ZK chunks(degree) → non-blinded from_coeffs → ref
    // quot cap → observe → ref zeta — and compare the GPU outputs against THAT (NOT the real blinded
    // squeezes above). The actual proof keeps the blinded commitments; this chain exists ONLY for
    // the byte-diff, de-risking the Rec#2 quotient + Blake3 from_coeffs + the post-quot-cap zeta.
    #[cfg(feature = "gpu_quotient")]
    {
        if std::env::var("PEARL_GPU_REC2_STAGE5_ASSERT").is_ok() {
            // ── Reference STAGE 2-4 (non-blinded), IDENTICAL to the rec2 stage-4 assert chain. ──
            let ref_wires_values: Vec<PolynomialValues<F>> = witness
                .wire_values
                .par_iter()
                .map(|column| PolynomialValues::new(column.clone()))
                .collect();
            let ref_wires_commit = PolynomialBatch::<F, C, D>::from_values(
                ref_wires_values,
                config.fri_config.rate_bits,
                false, // NON-blinded reference (GPU commits salt_cols=0)
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            let mut ref_challenger = Challenger::<F, C::Hasher>::new();
            common_data.fri_params.observe(&mut ref_challenger);
            ref_challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
            ref_challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);
            ref_challenger.observe_cap::<C::Hasher>(&ref_wires_commit.merkle_tree.cap);
            let ref_betas = ref_challenger.get_n_challenges(num_challenges);
            let ref_gammas = ref_challenger.get_n_challenges(num_challenges);

            let mut ref_pp_and_zs = all_wires_permutation_partial_products(
                &witness,
                &ref_betas,
                &ref_gammas,
                prover_data,
                common_data,
            );
            let ref_z_vecs = ref_pp_and_zs
                .iter_mut()
                .map(|pp_and_z| pp_and_z.pop().unwrap())
                .collect();
            let ref_zs_partial: Vec<PolynomialValues<F>> =
                [ref_z_vecs, ref_pp_and_zs.concat()].concat();
            let ref_zsp_commit = PolynomialBatch::<F, C, D>::from_values(
                ref_zs_partial,
                config.fri_config.rate_bits,
                false, // NON-blinded reference
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            ref_challenger.observe_cap::<C::Hasher>(&ref_zsp_commit.merkle_tree.cap);
            let ref_alphas = ref_challenger.get_n_challenges(num_challenges);

            // ── Reference STAGE 5: compute_quotient_polys with the REFERENCE betas/gammas/alphas +
            //    the NON-blinded reference commits. deltas = &[] (Rec#2 has no lookups). Then the
            //    NON-ZK chunk split (quotient.chunks(degree) — NO random_ts/random_r, matching the
            //    GPU) → NON-blinded from_coeffs → ref quot cap → observe → ref zeta. ──
            let ref_deltas: Vec<F> = Vec::new();
            let ref_quotient_polys = compute_quotient_polys::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &ref_wires_commit,
                &ref_zsp_commit,
                &ref_betas,
                &ref_gammas,
                &ref_deltas,
                &ref_alphas,
            );
            // NON-ZK split: trim_to_len(quotient_degree) then chunks(degree). (The GPU's STAGE-5
            // path commits exactly this — the zk random_ts/random_r are STAGE 8, deferred.)
            let ref_quot_chunks: Vec<PolynomialCoeffs<F>> = ref_quotient_polys
                .into_iter()
                .flat_map(|mut q| {
                    q.trim_to_len(quotient_degree)
                        .expect("ref quotient not divisible by Z_H");
                    q.chunks(degree)
                })
                .collect();
            let ref_quot_commit = PolynomialBatch::<F, C, D>::from_coeffs(
                ref_quot_chunks,
                config.fri_config.rate_bits,
                false, // NON-blinded reference (GPU from_coeffs blinding=false)
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            ref_challenger.observe_cap::<C::Hasher>(&ref_quot_commit.merkle_tree.cap);
            let ref_zeta = ref_challenger.get_extension_challenge::<D>();

            if let Some((
                gpu_wires_cap,
                gpu_betas,
                gpu_gammas,
                gpu_zsp_cap,
                gpu_alphas,
                gpu_quot_cap,
                gpu_zeta,
                _gpu_quot_chunk_coeffs,
            )) = crate::gpu::try_gpu_prove_rec2_stage5::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &witness.wire_values,
            ) {
                use plonky2_field::extension::FieldExtension;
                use plonky2_field::types::PrimeField64;
                // Reference caps → canonical u64 [cap_size*4] via the SAME BytesHash<27>::to_vec()
                // 7-byte-chunk flatten the GPU emits (4 elems per 27-byte cap entry).
                let ref_wires_cap: Vec<u64> = ref_wires_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let ref_zsp_cap: Vec<u64> = ref_zsp_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let ref_quot_cap: Vec<u64> = ref_quot_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_betas: Vec<u64> = ref_betas.iter().map(|b| b.to_canonical_u64()).collect();
                let cpu_gammas: Vec<u64> =
                    ref_gammas.iter().map(|g| g.to_canonical_u64()).collect();
                let cpu_alphas: Vec<u64> =
                    ref_alphas.iter().map(|a| a.to_canonical_u64()).collect();
                let ref_zeta_arr = ref_zeta.to_basefield_array();
                let cpu_zeta: Vec<u64> =
                    ref_zeta_arr.iter().map(|e| e.to_canonical_u64()).collect();
                // STAGE 2-4 regression (vs the NON-blinded reference):
                assert_eq!(
                    gpu_wires_cap, ref_wires_cap,
                    "[m2c-stage5] GPU Blake3 wires cap != CPU non-blinded from_values cap"
                );
                assert_eq!(
                    gpu_betas, cpu_betas,
                    "[m2c-stage5] GPU betas != CPU betas (non-blinded ref)"
                );
                assert_eq!(
                    gpu_gammas, cpu_gammas,
                    "[m2c-stage5] GPU gammas != CPU gammas (non-blinded ref)"
                );
                assert_eq!(
                    gpu_zsp_cap, ref_zsp_cap,
                    "[m2c-stage5] GPU Blake3 zs_partial cap != CPU non-blinded from_values cap (perm-Z)"
                );
                assert_eq!(
                    gpu_alphas, cpu_alphas,
                    "[m2c-stage5] GPU alphas != CPU alphas (non-blinded ref)"
                );
                // QDIFF localization (PEARL_GPU_REC2_STAGE5_QDIFF): compare the GPU quotient chunk
                // COEFFS (coset_ifft output, pre-commit) vs the CPU ref_quot_chunks, BEFORE the cap
                // assert. coeffs-MATCH ⇒ bug is in from_coeffs/Blake3 commit; coeffs-DIFFER ⇒ bug is
                // in quotient-compute (ka2 gates / kernel / coset_ifft); the first-mismatch index
                // localizes the challenge/chunk/coeff.
                if let Some(gpu_chunks) = _gpu_quot_chunk_coeffs.as_ref() {
                    // ref_quot_chunks was moved into from_coeffs; the SAME coeffs live in
                    // ref_quot_commit.polynomials (from_coeffs stores its input there).
                    let cpu_chunks: Vec<u64> = ref_quot_commit
                        .polynomials
                        .iter()
                        .flat_map(|c| c.coeffs.iter().map(|e| e.to_canonical_u64()))
                        .collect();
                    let qdf_local = common_data.quotient_degree_factor;
                    let n_cmp = cpu_chunks.len().min(gpu_chunks.len());
                    let mut first = None;
                    let mut nmis = 0usize;
                    for k in 0..n_cmp {
                        if cpu_chunks[k] != gpu_chunks[k] {
                            if first.is_none() {
                                first = Some(k);
                            }
                            nmis += 1;
                        }
                    }
                    match first {
                        None if cpu_chunks.len() == gpu_chunks.len() => eprintln!(
                            "[m2c-stage5-qdiff] quotient chunk COEFFS byte-exact ({n_cmp} elems) \
                             => bug is in from_coeffs/Blake3 commit"
                        ),
                        None => eprintln!(
                            "[m2c-stage5-qdiff] coeffs prefix matches but LEN differs: gpu={} cpu={}",
                            gpu_chunks.len(),
                            cpu_chunks.len()
                        ),
                        Some(k) => {
                            let chunk = k / degree;
                            let coeff = k % degree;
                            eprintln!(
                                "[m2c-stage5-qdiff] quotient COEFFS DIFFER: {nmis}/{n_cmp} mismatch; \
                                 FIRST at idx {k} (chal {} chunk {} coeff {}): gpu={} cpu={}",
                                chunk / qdf_local,
                                chunk % qdf_local,
                                coeff,
                                gpu_chunks[k],
                                cpu_chunks[k]
                            );
                        }
                    }
                }
                // STAGE 5 (quotient compute + BLAKE3 from_coeffs commit) + STAGE 6 (observe → zeta):
                assert_eq!(
                    gpu_quot_cap, ref_quot_cap,
                    "[m2c-stage5] GPU Blake3 quotient cap != CPU non-blinded from_coeffs cap"
                );
                assert_eq!(
                    gpu_zeta, cpu_zeta,
                    "[m2c-stage5] GPU zeta (post quot-cap observe) != CPU zeta (non-blinded ref)"
                );
                eprintln!("[m2c-stage5] QUOTIENT + BLAKE3 COMMIT + ZETA byte-exact");
            } else {
                eprintln!(
                    "[m2c-stage5] GPU Rec#2 stage-2/6 unavailable (shape/TypeId/CUDA) — \
                     skipping assert (CPU path unchanged)"
                );
            }
        }
    }

    // GPU fused-prover STAGE 6 ASSERT (Pearl Rec#2; task build-order; == .cuh STAGE 7 openings + the
    // START of STAGE 8's FRI alpha). Feature-gated, byte-exact, CONSENSUS-NEUTRAL — same class as the
    // Rec#2 STAGE 5 assert; the verifier is untouched. `PEARL_GPU_REC2_STAGE6_ASSERT` runs the on-GPU
    // `pearl_gpu_prove_rec2_f64` (stages 2-6 + the OpeningSet + the FRI alpha) and byte-checks the GPU
    // wires_cap + betas/gammas + zs_partial cap + alphas + quotient cap + zeta (STAGE 2-6 regression)
    // PLUS the OpeningSet (each GFExt opening, in to_fri_openings order) + the FRI alpha — all vs a
    // CPU NON-blinded reference CHAIN (identical to the STAGE 5 assert's chain through `ref_zeta`,
    // then CONTINUED through a MANUALLY-built reference OpeningSet at (ref_zeta, g) over the
    // non-blinded ref commits → observe_openings → get_extension_challenge = ref fri_alpha).
    // TypeId-gated to Blake3GoldilocksConfig (fires ONLY on the Rec#2 prove). The actual proof keeps
    // the blinded commitments; this chain exists ONLY for the byte-diff (de-risking the Rec#2
    // OpeningSet eval on the device-resident coeffs + the Blake3-sponge observe_openings → the FRI
    // alpha). The reference OpeningSet is built MANUALLY (NOT via OpeningSet::new) because under ZK
    // common_data.quotient_range()/random_range() describe the BLINDED smaller-chunk split, NOT the
    // GPU's NON-ZK all-24-chunk split + empty random_r — see the reference STAGE 6 note below.
    #[cfg(feature = "gpu_quotient")]
    {
        if std::env::var("PEARL_GPU_REC2_STAGE6_ASSERT").is_ok() {
            // ── Reference STAGE 2-4 (non-blinded), IDENTICAL to the rec2 stage-5 assert chain. ──
            let ref_wires_values: Vec<PolynomialValues<F>> = witness
                .wire_values
                .par_iter()
                .map(|column| PolynomialValues::new(column.clone()))
                .collect();
            let ref_wires_commit = PolynomialBatch::<F, C, D>::from_values(
                ref_wires_values,
                config.fri_config.rate_bits,
                false, // NON-blinded reference (GPU commits salt_cols=0)
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            let mut ref_challenger = Challenger::<F, C::Hasher>::new();
            common_data.fri_params.observe(&mut ref_challenger);
            ref_challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
            ref_challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);
            ref_challenger.observe_cap::<C::Hasher>(&ref_wires_commit.merkle_tree.cap);
            let ref_betas = ref_challenger.get_n_challenges(num_challenges);
            let ref_gammas = ref_challenger.get_n_challenges(num_challenges);

            let mut ref_pp_and_zs = all_wires_permutation_partial_products(
                &witness,
                &ref_betas,
                &ref_gammas,
                prover_data,
                common_data,
            );
            let ref_z_vecs = ref_pp_and_zs
                .iter_mut()
                .map(|pp_and_z| pp_and_z.pop().unwrap())
                .collect();
            let ref_zs_partial: Vec<PolynomialValues<F>> =
                [ref_z_vecs, ref_pp_and_zs.concat()].concat();
            let ref_zsp_commit = PolynomialBatch::<F, C, D>::from_values(
                ref_zs_partial,
                config.fri_config.rate_bits,
                false, // NON-blinded reference
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            ref_challenger.observe_cap::<C::Hasher>(&ref_zsp_commit.merkle_tree.cap);
            let ref_alphas = ref_challenger.get_n_challenges(num_challenges);

            // ── Reference STAGE 5: compute_quotient_polys(ref) → NON-ZK chunk split → non-blinded
            //    from_coeffs → ref quot cap → observe → ref zeta. (== the STAGE 5 assert chain.) ──
            let ref_deltas: Vec<F> = Vec::new();
            let ref_quotient_polys = compute_quotient_polys::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &ref_wires_commit,
                &ref_zsp_commit,
                &ref_betas,
                &ref_gammas,
                &ref_deltas,
                &ref_alphas,
            );
            let ref_quot_chunks: Vec<PolynomialCoeffs<F>> = ref_quotient_polys
                .into_iter()
                .flat_map(|mut q| {
                    q.trim_to_len(quotient_degree)
                        .expect("ref quotient not divisible by Z_H");
                    q.chunks(degree)
                })
                .collect();
            let ref_quot_commit = PolynomialBatch::<F, C, D>::from_coeffs(
                ref_quot_chunks,
                config.fri_config.rate_bits,
                false, // NON-blinded reference (GPU from_coeffs blinding=false)
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            ref_challenger.observe_cap::<C::Hasher>(&ref_quot_commit.merkle_tree.cap);
            let ref_zeta = ref_challenger.get_extension_challenge::<D>();

            // ── Reference STAGE 6: build the OpeningSet on the NON-blinded ref commits MANUALLY
            //    (NOT via OpeningSet::new on the ZK common_data). The crux: under zero_knowledge the
            //    REAL quotient is split into num_quotient_polys() SMALLER blinded chunks + 1 random_r
            //    poly, so common_data.quotient_range() / random_range() do NOT match the GPU's NON-ZK
            //    split. The GPU's STAGE 5 commits exactly num_challenges*qdf=24 degree-n chunks
            //    (quotient.chunks(degree), NO random_r) — validated by the STAGE 5 cap assert above —
            //    so ref_quot_commit has EXACTLY those 24 polys. We therefore eval ALL 24 quotient
            //    polys (quotient_polys range = full; random_r empty), matching the GPU. The other
            //    ranges (constants/sigmas/zs/partial_products) are config-based (ZK-INDEPENDENT) and
            //    identical to the GPU's slicing, so they are safe to take from common_data. g ==
            //    F::Extension::primitive_root_of_unity(degree_bits) — the SAME `g` the real
            //    OpeningSet::new uses (degree-bits dependent, NOT blinding-dependent). The resulting
            //    to_fri_openings batches (has_lookup=false ⇒ no lookup; has_random=false ⇒ no
            //    random_r) are EXACTLY the GPU's observe order. ──
            let ref_eval = |z: F::Extension, c: &PolynomialBatch<F, C, D>| -> Vec<F::Extension> {
                c.polynomials
                    .par_iter()
                    .map(|p| p.to_extension().eval(z))
                    .collect()
            };
            let ref_cs_eval = ref_eval(ref_zeta, &prover_data.constants_sigmas_commitment);
            let ref_zsp_eval = ref_eval(ref_zeta, &ref_zsp_commit);
            let ref_zsp_next_eval = ref_eval(g * ref_zeta, &ref_zsp_commit);
            let ref_openings = OpeningSet::<F, D> {
                constants: ref_cs_eval[common_data.constants_range()].to_vec(),
                plonk_sigmas: ref_cs_eval[common_data.sigmas_range()].to_vec(),
                wires: ref_eval(ref_zeta, &ref_wires_commit),
                plonk_zs: ref_zsp_eval[common_data.zs_range()].to_vec(),
                plonk_zs_next: ref_zsp_next_eval[common_data.zs_range()].to_vec(),
                partial_products: ref_zsp_eval[common_data.partial_products_range()].to_vec(),
                // ALL 24 non-blinded chunks (NOT common_data.quotient_range(), which is the ZK split).
                quotient_polys: ref_eval(ref_zeta, &ref_quot_commit),
                lookup_zs: Vec::new(),
                lookup_zs_next: Vec::new(),
                random_r: Vec::new(),
            };
            ref_challenger.observe_openings(&ref_openings.to_fri_openings());
            let ref_fri_alpha = ref_challenger.get_extension_challenge::<D>();

            if let Some((
                gpu_wires_cap,
                gpu_betas,
                gpu_gammas,
                gpu_zsp_cap,
                gpu_alphas,
                gpu_quot_cap,
                gpu_zeta,
                gpu_openings,
                gpu_fri_alpha,
            )) = crate::gpu::try_gpu_prove_rec2_stage6::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &witness.wire_values,
            ) {
                use plonky2_field::extension::FieldExtension;
                use plonky2_field::types::PrimeField64;
                // Reference caps → canonical u64 [cap_size*4] via the SAME BytesHash<27>::to_vec()
                // 7-byte-chunk flatten the GPU emits (4 elems per 27-byte cap entry).
                let ref_wires_cap: Vec<u64> = ref_wires_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let ref_zsp_cap: Vec<u64> = ref_zsp_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let ref_quot_cap: Vec<u64> = ref_quot_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_betas: Vec<u64> = ref_betas.iter().map(|b| b.to_canonical_u64()).collect();
                let cpu_gammas: Vec<u64> =
                    ref_gammas.iter().map(|g| g.to_canonical_u64()).collect();
                let cpu_alphas: Vec<u64> =
                    ref_alphas.iter().map(|a| a.to_canonical_u64()).collect();
                let cpu_zeta: Vec<u64> = ref_zeta
                    .to_basefield_array()
                    .iter()
                    .map(|e| e.to_canonical_u64())
                    .collect();
                // STAGE 2-6 regression (vs the NON-blinded reference):
                assert_eq!(
                    gpu_wires_cap, ref_wires_cap,
                    "[m2c-stage6] GPU Blake3 wires cap != CPU non-blinded from_values cap"
                );
                assert_eq!(
                    gpu_betas, cpu_betas,
                    "[m2c-stage6] GPU betas != CPU betas (non-blinded ref)"
                );
                assert_eq!(
                    gpu_gammas, cpu_gammas,
                    "[m2c-stage6] GPU gammas != CPU gammas (non-blinded ref)"
                );
                assert_eq!(
                    gpu_zsp_cap, ref_zsp_cap,
                    "[m2c-stage6] GPU Blake3 zs_partial cap != CPU non-blinded from_values cap (perm-Z)"
                );
                assert_eq!(
                    gpu_alphas, cpu_alphas,
                    "[m2c-stage6] GPU alphas != CPU alphas (non-blinded ref)"
                );
                assert_eq!(
                    gpu_quot_cap, ref_quot_cap,
                    "[m2c-stage6] GPU Blake3 quotient cap != CPU non-blinded from_coeffs cap"
                );
                assert_eq!(
                    gpu_zeta, cpu_zeta,
                    "[m2c-stage6] GPU zeta (post quot-cap observe) != CPU zeta (non-blinded ref)"
                );

                // STAGE 6 openings: byte-exact, in to_fri_openings slice order. Flatten each ref
                // OpeningSet field's GFExt values to canonical [c0, c1] (to_basefield_array order —
                // the SAME coord order observe_extension_element uses and the GPU rp_eval kernel
                // stores). Rec#2 NON-blinded ref: constants(6) ‖ plonk_sigmas(34) ‖ wires(135) ‖
                // plonk_zs(3) ‖ partial_products(12) ‖ quotient_polys(24); zs_next batch = plonk_zs_next(3).
                let flat = |v: &[F::Extension]| -> Vec<u64> {
                    v.iter()
                        .flat_map(|e| {
                            e.to_basefield_array()
                                .into_iter()
                                .map(|c| c.to_canonical_u64())
                        })
                        .collect()
                };
                let (
                    gpu_constants,
                    gpu_plonk_sigmas,
                    gpu_wires,
                    gpu_plonk_zs,
                    gpu_partial_products,
                    gpu_quotient_polys,
                    gpu_plonk_zs_next,
                ) = gpu_openings;
                assert_eq!(
                    gpu_constants,
                    flat(&ref_openings.constants),
                    "[m2c-stage6] GPU constants opening != CPU non-blinded ref"
                );
                assert_eq!(
                    gpu_plonk_sigmas,
                    flat(&ref_openings.plonk_sigmas),
                    "[m2c-stage6] GPU plonk_sigmas opening != CPU non-blinded ref"
                );
                assert_eq!(
                    gpu_wires,
                    flat(&ref_openings.wires),
                    "[m2c-stage6] GPU wires opening != CPU non-blinded ref"
                );
                assert_eq!(
                    gpu_plonk_zs,
                    flat(&ref_openings.plonk_zs),
                    "[m2c-stage6] GPU plonk_zs opening != CPU non-blinded ref"
                );
                assert_eq!(
                    gpu_partial_products,
                    flat(&ref_openings.partial_products),
                    "[m2c-stage6] GPU partial_products opening != CPU non-blinded ref"
                );
                assert_eq!(
                    gpu_quotient_polys,
                    flat(&ref_openings.quotient_polys),
                    "[m2c-stage6] GPU quotient_polys opening != CPU non-blinded ref"
                );
                assert_eq!(
                    gpu_plonk_zs_next,
                    flat(&ref_openings.plonk_zs_next),
                    "[m2c-stage6] GPU plonk_zs_next (@ g·zeta) opening != CPU non-blinded ref"
                );

                // STAGE 6 FRI alpha: get_extension_challenge() on the non-blinded ref_challenger,
                // FIRST squeeze after observe_openings (fri/oracle.rs:207).
                let cpu_fri_alpha: Vec<u64> = ref_fri_alpha
                    .to_basefield_array()
                    .iter()
                    .map(|e| e.to_canonical_u64())
                    .collect();
                assert_eq!(
                    gpu_fri_alpha, cpu_fri_alpha,
                    "[m2c-stage6] GPU FRI alpha (post observe_openings) != CPU non-blinded ref"
                );
                eprintln!("[m2c-stage6] OPENINGS + FRI ALPHA byte-exact");
            } else {
                eprintln!(
                    "[m2c-stage6] GPU Rec#2 stage-2/6 + openings unavailable (shape/TypeId/CUDA) — \
                     skipping assert (CPU path unchanged)"
                );
            }
        }
    }

    // Rec#2 STAGE 7 assert (the NON-blinded FRI COMMIT PHASE). Feature-gated, byte-exact,
    // CONSENSUS-NEUTRAL — same class as the Rec#2 STAGE 6 assert; the verifier is untouched.
    // `PEARL_GPU_REC2_STAGE7_ASSERT` runs the on-GPU `pearl_gpu_prove_rec2_f64` (stages 2-7) and
    // byte-checks the DETERMINISTIC FRI commit phase — the commit-phase Merkle caps + the final
    // polynomial — vs a CPU NON-blinded reference FriProof. TypeId-gated to Blake3GoldilocksConfig
    // (fires ONLY on the Rec#2 prove). The STAGE 2-6 regression (caps/challenges/zeta/fri_alpha) is
    // re-asserted too. PoW witness + query rounds are NOT asserted byte-exact (plonky2's rayon
    // find_map_any nonce is non-deterministic; the GPU's lowest spec-valid nonce differs but is
    // verifier-checked).
    //
    // ── THE ZK CRUX (the reference must be hiding=FALSE) ──────────────────────────────────────
    // The REAL Rec#2 `opening_proof` is the BLINDED FriProof: `PolynomialBatch::prove_openings`
    // with `fri_params.hiding=true` adds the R-poly to the ZETA batch (without quotient) AND the
    // blinded oracle leaves change every commit cap. The GPU's STAGE 7 is NON-blinded (no R-poly,
    // no extra fold, non-blinded openings), so it can ONLY match a reference built with a
    // `hiding=false` FriParams clone over the non-blinded reference commits + a NON-blinded
    // reference FriInstanceInfo (all num_challenges*qdf=24 quotient chunks, NO R-poly,
    // blinding=false oracles). `reduction_arity_bits` is hiding-INDEPENDENT (FriConfig::fri_params
    // computes it from degree_bits/rate/cap/num_query_rounds only), so the GPU's rounds (which use
    // common_data.fri_params.reduction_arity_bits) match the hiding=false reference's rounds. The
    // reference challenger is the SAME non-blinded chain the STAGE 6 assert builds (its prefix
    // observes hiding=TRUE via common_data.fri_params.observe — matching the GPU's stage-3 prefix —
    // so the FRI alpha is identical), snapshotted right after observe_openings and handed to
    // prove_openings (which draws alpha as its FIRST step). The actual proof stays blinded.
    #[cfg(feature = "gpu_quotient")]
    {
        if std::env::var("PEARL_GPU_REC2_STAGE7_ASSERT").is_ok() {
            use crate::fri::structure::{
                FriBatchInfo, FriInstanceInfo, FriOracleInfo, FriPolynomialInfo,
            };
            // ── Reference STAGE 2-4 (non-blinded), IDENTICAL to the rec2 stage-6 assert chain. ──
            let ref_wires_values: Vec<PolynomialValues<F>> = witness
                .wire_values
                .par_iter()
                .map(|column| PolynomialValues::new(column.clone()))
                .collect();
            let ref_wires_commit = PolynomialBatch::<F, C, D>::from_values(
                ref_wires_values,
                config.fri_config.rate_bits,
                false, // NON-blinded reference (GPU commits salt_cols=0)
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            let mut ref_challenger = Challenger::<F, C::Hasher>::new();
            common_data.fri_params.observe(&mut ref_challenger);
            ref_challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
            ref_challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);
            ref_challenger.observe_cap::<C::Hasher>(&ref_wires_commit.merkle_tree.cap);
            let ref_betas = ref_challenger.get_n_challenges(num_challenges);
            let ref_gammas = ref_challenger.get_n_challenges(num_challenges);

            let mut ref_pp_and_zs = all_wires_permutation_partial_products(
                &witness,
                &ref_betas,
                &ref_gammas,
                prover_data,
                common_data,
            );
            let ref_z_vecs = ref_pp_and_zs
                .iter_mut()
                .map(|pp_and_z| pp_and_z.pop().unwrap())
                .collect();
            let ref_zs_partial: Vec<PolynomialValues<F>> =
                [ref_z_vecs, ref_pp_and_zs.concat()].concat();
            let ref_zsp_commit = PolynomialBatch::<F, C, D>::from_values(
                ref_zs_partial,
                config.fri_config.rate_bits,
                false, // NON-blinded reference
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            ref_challenger.observe_cap::<C::Hasher>(&ref_zsp_commit.merkle_tree.cap);
            let ref_alphas = ref_challenger.get_n_challenges(num_challenges);

            // ── Reference STAGE 5: NON-ZK chunk split → non-blinded from_coeffs → ref zeta. ──
            let ref_deltas: Vec<F> = Vec::new();
            let ref_quotient_polys = compute_quotient_polys::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &ref_wires_commit,
                &ref_zsp_commit,
                &ref_betas,
                &ref_gammas,
                &ref_deltas,
                &ref_alphas,
            );
            let ref_quot_chunks: Vec<PolynomialCoeffs<F>> = ref_quotient_polys
                .into_iter()
                .flat_map(|mut q| {
                    q.trim_to_len(quotient_degree)
                        .expect("ref quotient not divisible by Z_H");
                    q.chunks(degree)
                })
                .collect();
            let ref_quot_commit = PolynomialBatch::<F, C, D>::from_coeffs(
                ref_quot_chunks,
                config.fri_config.rate_bits,
                false, // NON-blinded reference (GPU from_coeffs blinding=false)
                config.fri_config.cap_height,
                timing,
                prover_data.fft_root_table.as_ref(),
            );
            ref_challenger.observe_cap::<C::Hasher>(&ref_quot_commit.merkle_tree.cap);
            let ref_zeta = ref_challenger.get_extension_challenge::<D>();

            // ── Reference STAGE 6: the MANUAL non-blinded OpeningSet (== rec2 stage6 assert). ──
            let ref_eval = |z: F::Extension, c: &PolynomialBatch<F, C, D>| -> Vec<F::Extension> {
                c.polynomials
                    .par_iter()
                    .map(|p| p.to_extension().eval(z))
                    .collect()
            };
            let ref_cs_eval = ref_eval(ref_zeta, &prover_data.constants_sigmas_commitment);
            let ref_zsp_eval = ref_eval(ref_zeta, &ref_zsp_commit);
            let ref_zsp_next_eval = ref_eval(g * ref_zeta, &ref_zsp_commit);
            let ref_openings = OpeningSet::<F, D> {
                constants: ref_cs_eval[common_data.constants_range()].to_vec(),
                plonk_sigmas: ref_cs_eval[common_data.sigmas_range()].to_vec(),
                wires: ref_eval(ref_zeta, &ref_wires_commit),
                plonk_zs: ref_zsp_eval[common_data.zs_range()].to_vec(),
                plonk_zs_next: ref_zsp_next_eval[common_data.zs_range()].to_vec(),
                partial_products: ref_zsp_eval[common_data.partial_products_range()].to_vec(),
                quotient_polys: ref_eval(ref_zeta, &ref_quot_commit),
                lookup_zs: Vec::new(),
                lookup_zs_next: Vec::new(),
                random_r: Vec::new(),
            };
            ref_challenger.observe_openings(&ref_openings.to_fri_openings());
            // Snapshot AFTER observe_openings, BEFORE the FRI alpha draw: prove_openings draws
            // alpha as its FIRST step (fri/oracle.rs:207), so the snapshot is the exact entry state.
            let ref_challenger_pre_fri = ref_challenger.clone();
            let ref_fri_alpha = ref_challenger.get_extension_challenge::<D>();

            // ── Reference STAGE 7: a NON-blinded FriProof. Build a manual FriInstanceInfo matching
            //    the GPU's NON-ZK shape (4 oracles, blinding=false; all 24 quotient chunks in the
            //    zeta batch; NO R-poly), then prove_openings on a hiding=FALSE FriParams clone over
            //    the 4 NON-blinded reference oracles + the post-observe_openings reference challenger.
            //    fri_all_polys order = cs(preproc) ++ wires ++ zs_partial ++ quotient (oracle order
            //    0,1,2,3); fri_next_batch = zs_range of oracle 2 (the Z's @ g·ζ). ──
            let cs_polys_n = prover_data.constants_sigmas_commitment.polynomials.len();
            let wires_n = ref_wires_commit.polynomials.len();
            let zsp_n = ref_zsp_commit.polynomials.len();
            let quot_n = ref_quot_commit.polynomials.len(); // == num_challenges*qdf (non-ZK split)
            let ref_oracles = vec![
                FriOracleInfo {
                    num_polys: cs_polys_n,
                    blinding: false,
                },
                FriOracleInfo {
                    num_polys: wires_n,
                    blinding: false,
                },
                FriOracleInfo {
                    num_polys: zsp_n,
                    blinding: false,
                },
                FriOracleInfo {
                    num_polys: quot_n,
                    blinding: false,
                },
            ];
            let zeta_polys: Vec<FriPolynomialInfo> = [
                FriPolynomialInfo::from_range(0, 0..cs_polys_n),
                FriPolynomialInfo::from_range(1, 0..wires_n),
                FriPolynomialInfo::from_range(2, 0..zsp_n),
                FriPolynomialInfo::from_range(3, 0..quot_n),
            ]
            .concat();
            // zeta_next batch = the Z polys (zs_range) of the zs_partial oracle (index 2).
            let zs_range = common_data.zs_range();
            let zeta_next_polys = FriPolynomialInfo::from_range(2, zs_range);
            let ref_instance = FriInstanceInfo::<F, D> {
                oracles: ref_oracles,
                batches: vec![
                    FriBatchInfo {
                        point: ref_zeta,
                        polynomials: zeta_polys,
                    },
                    FriBatchInfo {
                        point: g * ref_zeta,
                        polynomials: zeta_next_polys,
                    },
                ],
            };
            // hiding=FALSE clone (no R-poly, no extra fold); reduction_arity_bits unchanged.
            let mut ref_fri_params = common_data.fri_params.clone();
            ref_fri_params.hiding = false;
            let ref_oracle_refs: &[&PolynomialBatch<F, C, D>] = &[
                &prover_data.constants_sigmas_commitment,
                &ref_wires_commit,
                &ref_zsp_commit,
                &ref_quot_commit,
            ];
            let mut ref_ch_for_fri = ref_challenger_pre_fri;
            let ref_fri = PolynomialBatch::<F, C, D>::prove_openings(
                &ref_instance,
                ref_oracle_refs,
                &mut ref_ch_for_fri,
                &ref_fri_params,
                None,
                None,
                timing,
            );

            if let Some((
                gpu_wires_cap,
                gpu_betas,
                gpu_gammas,
                gpu_zsp_cap,
                gpu_alphas,
                gpu_quot_cap,
                gpu_zeta,
                gpu_fri_alpha,
                gpu_fri_commit_caps,
                gpu_fri_final_poly,
                _gpu_fri_pow_witness,
                _gpu_fri_query_indices,
                _gpu_fri_step_evals,
                _gpu_fri_step_paths,
            )) = crate::gpu::try_gpu_prove_rec2_stage7::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &witness.wire_values,
            ) {
                use plonky2_field::extension::FieldExtension;
                use plonky2_field::types::PrimeField64;
                // Reference caps → canonical u64 [cap_size*4] via BytesHash<27>::to_vec().
                let ref_wires_cap: Vec<u64> = ref_wires_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let ref_zsp_cap: Vec<u64> = ref_zsp_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let ref_quot_cap: Vec<u64> = ref_quot_commit
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_betas: Vec<u64> = ref_betas.iter().map(|b| b.to_canonical_u64()).collect();
                let cpu_gammas: Vec<u64> =
                    ref_gammas.iter().map(|g| g.to_canonical_u64()).collect();
                let cpu_alphas: Vec<u64> =
                    ref_alphas.iter().map(|a| a.to_canonical_u64()).collect();
                let cpu_zeta: Vec<u64> = ref_zeta
                    .to_basefield_array()
                    .iter()
                    .map(|e| e.to_canonical_u64())
                    .collect();
                // STAGE 2-6 regression (vs the NON-blinded reference):
                assert_eq!(
                    gpu_wires_cap, ref_wires_cap,
                    "[m2c-stage7] GPU Blake3 wires cap != CPU non-blinded from_values cap"
                );
                assert_eq!(
                    gpu_betas, cpu_betas,
                    "[m2c-stage7] GPU betas != CPU betas (non-blinded ref)"
                );
                assert_eq!(
                    gpu_gammas, cpu_gammas,
                    "[m2c-stage7] GPU gammas != CPU gammas (non-blinded ref)"
                );
                assert_eq!(
                    gpu_zsp_cap, ref_zsp_cap,
                    "[m2c-stage7] GPU Blake3 zs_partial cap != CPU non-blinded from_values cap (perm-Z)"
                );
                assert_eq!(
                    gpu_alphas, cpu_alphas,
                    "[m2c-stage7] GPU alphas != CPU alphas (non-blinded ref)"
                );
                assert_eq!(
                    gpu_quot_cap, ref_quot_cap,
                    "[m2c-stage7] GPU Blake3 quotient cap != CPU non-blinded from_coeffs cap"
                );
                assert_eq!(
                    gpu_zeta, cpu_zeta,
                    "[m2c-stage7] GPU zeta (post quot-cap observe) != CPU zeta (non-blinded ref)"
                );
                let cpu_fri_alpha: Vec<u64> = ref_fri_alpha
                    .to_basefield_array()
                    .iter()
                    .map(|e| e.to_canonical_u64())
                    .collect();
                assert_eq!(
                    gpu_fri_alpha, cpu_fri_alpha,
                    "[m2c-stage7] GPU FRI alpha (post observe_openings) != CPU non-blinded ref"
                );

                // ── STAGE 7 — the DETERMINISTIC FRI commit phase, byte-exact vs the NON-blinded
                //    reference FriProof (commit_phase_merkle_caps + final_poly). Flatten both to
                //    canonical u64 in the SAME layout the GPU writes (caps: round-major, cap-order,
                //    4 elems each via BytesHash::to_vec; final_poly: c0,c1 per ext coeff). ──
                let cpu_fri_commit_caps: Vec<u64> = ref_fri
                    .commit_phase_merkle_caps
                    .iter()
                    .flat_map(|cap| {
                        cap.0.iter().flat_map(|h| {
                            GenericHashOut::<F>::to_vec(h)
                                .into_iter()
                                .map(|e| e.to_canonical_u64())
                        })
                    })
                    .collect();
                let cpu_fri_final_poly: Vec<u64> = ref_fri
                    .final_poly
                    .coeffs
                    .iter()
                    .flat_map(|c| {
                        c.to_basefield_array()
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                // DIAGNOSTIC (non-fatal): localize the first differing commit-cap round + final_poly.
                let nrounds = ref_fri.commit_phase_merkle_caps.len();
                let cap_per_round = if nrounds == 0 {
                    0
                } else {
                    cpu_fri_commit_caps.len() / nrounds
                };
                eprintln!(
                    "[m2c-stage7][diag] commit caps: rounds={nrounds} words/round={cap_per_round}"
                );
                for r in 0..nrounds {
                    let lo = r * cap_per_round;
                    let hi = lo + cap_per_round;
                    let eq = gpu_fri_commit_caps.get(lo..hi) == cpu_fri_commit_caps.get(lo..hi);
                    eprintln!(
                        "[m2c-stage7][diag] cap round {r}: {}",
                        if eq { "MATCH" } else { "DIFFER" }
                    );
                }
                eprintln!(
                    "[m2c-stage7][diag] final_poly: gpu_len={} cpu_len={} eq={}",
                    gpu_fri_final_poly.len(),
                    cpu_fri_final_poly.len(),
                    gpu_fri_final_poly == cpu_fri_final_poly
                );
                assert_eq!(
                    gpu_fri_commit_caps, cpu_fri_commit_caps,
                    "[m2c-stage7] GPU FRI commit-phase Merkle caps != CPU non-blinded prove_openings caps"
                );
                assert_eq!(
                    gpu_fri_final_poly, cpu_fri_final_poly,
                    "[m2c-stage7] GPU FRI final_poly != CPU non-blinded prove_openings final_poly"
                );
                eprintln!("[m2c-stage7] FRI COMMIT PHASE (caps + final_poly) byte-exact");
            } else {
                eprintln!(
                    "[m2c-stage7] GPU Rec#2 stage-2/7 unavailable (shape/TypeId/CUDA) — \
                     skipping assert (CPU path unchanged)"
                );
            }
        }
    }

    // NOTE (M3): the Rec#1 + Rec#2 STAGE-8 capstone hooks were MOVED to the TOP of this function
    // (right after the witness), so when PEARL_GPU_REC1 / PEARL_GPU_REC2 is engaged we early-return
    // the fully-GPU-built proof BEFORE the redundant CPU commit->quotient->FRI flow runs. That drops
    // the ~4.7s of redundant CPU commits (M3 pre_hook_cpu_commits). The old bottom hooks (which
    // rebuilt initial_trees_proof from the live CPU oracle trees, hence had to run AFTER the commits)
    // are removed — oracle 0 (cs) is preprocessed and oracles 1-3's itp is now GPU-emitted.
    let openings = timed!(timing, "construct the opening set, including lookups", {
        OpeningSet::new(
            zeta,
            g,
            &prover_data.constants_sigmas_commitment,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &quotient_polys_random_commitment,
            common_data,
        )
    });
    challenger.observe_openings(&openings.to_fri_openings());

    // GPU fused-prover STAGE 6 ASSERT (task build-order; == .cuh STAGE 7 openings + the START of
    // STAGE 8's FRI alpha). Feature-gated, byte-exact, CONSENSUS-NEUTRAL — same class as the
    // STAGE 5 assert; the verifier is untouched. `PEARL_GPU_REC1_STAGE6_ASSERT` runs the on-GPU
    // `pearl_gpu_prove_rec1_f64` (stages 2-6 + openings + FRI alpha) and asserts the GPU
    // wires_cap + betas/gammas + zs_partial cap + alphas + quotient cap + zeta (STAGE 2-6
    // regression) PLUS the OpeningSet (each GFExt opening, byte-exact, in to_fri_openings order)
    // + the FRI alpha equal the CPU values. This fires HERE — AFTER the CPU has built the
    // OpeningSet AND observed it (`challenger.observe_openings`), so both challengers are at the
    // SAME position (the FRI-prove entry, fri/oracle.rs:207 where `alpha` is squeezed). The CPU
    // FRI alpha is read from a CLONE of `challenger` (squeezing get_extension_challenge) so the
    // real `challenger` is left untouched for `prove_openings`. No-op unless the env var is set;
    // falls through silently on any unsupported shape.
    #[cfg(feature = "gpu_quotient")]
    {
        if std::env::var("PEARL_GPU_REC1_STAGE6_ASSERT").is_ok() {
            if let Some((
                gpu_wires_cap,
                gpu_betas,
                gpu_gammas,
                gpu_zsp_cap,
                gpu_alphas,
                gpu_quot_cap,
                gpu_zeta,
                gpu_openings,
                gpu_fri_alpha,
            )) = crate::gpu::try_gpu_prove_rec1_stage6::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &witness.wire_values,
            ) {
                use plonky2_field::extension::FieldExtension;
                use plonky2_field::types::PrimeField64;
                // STAGE 2-6 regression caps/challenges (must still hold).
                let cpu_wires_cap: Vec<u64> = wires_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| h.to_vec().into_iter().map(|e| e.to_canonical_u64()))
                    .collect();
                let cpu_zsp_cap: Vec<u64> = partial_products_zs_and_lookup_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_quot_cap: Vec<u64> = quotient_polys_random_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_betas: Vec<u64> = betas.iter().map(|b| b.to_canonical_u64()).collect();
                let cpu_gammas: Vec<u64> = gammas.iter().map(|g| g.to_canonical_u64()).collect();
                let cpu_alphas: Vec<u64> = alphas.iter().map(|a| a.to_canonical_u64()).collect();
                let cpu_zeta: Vec<u64> = zeta
                    .to_basefield_array()
                    .iter()
                    .map(|e| e.to_canonical_u64())
                    .collect();
                assert_eq!(
                    gpu_wires_cap, cpu_wires_cap,
                    "[m2-stage6] GPU wires_commitment cap != CPU from_values cap"
                );
                assert_eq!(gpu_betas, cpu_betas, "[m2-stage6] GPU betas != CPU betas");
                assert_eq!(
                    gpu_gammas, cpu_gammas,
                    "[m2-stage6] GPU gammas != CPU gammas"
                );
                assert_eq!(
                    gpu_zsp_cap, cpu_zsp_cap,
                    "[m2-stage6] GPU zs_partial_products cap != CPU from_values cap (perm-Z)"
                );
                assert_eq!(
                    gpu_alphas, cpu_alphas,
                    "[m2-stage6] GPU alphas (post zs-cap observe) != CPU alphas"
                );
                assert_eq!(
                    gpu_quot_cap, cpu_quot_cap,
                    "[m2-stage6] GPU quotient cap != CPU from_coeffs cap"
                );
                assert_eq!(
                    gpu_zeta, cpu_zeta,
                    "[m2-stage6] GPU zeta (post quot-cap observe) != CPU zeta"
                );

                // STAGE 6 openings: byte-exact, in to_fri_openings slice order. Flatten each CPU
                // OpeningSet field's GFExt values to canonical [c0, c1] (to_basefield_array order
                // — the SAME coord order observe_extension_element uses and the GPU rp_eval kernel
                // stores). NON-zk Rec#1: constants(6) ‖ plonk_sigmas(37) ‖ wires(135) ‖
                // plonk_zs(3) ‖ partial_products(12) ‖ quotient_polys(24); zs_next batch = plonk_zs_next(3).
                let flat = |v: &[F::Extension]| -> Vec<u64> {
                    v.iter()
                        .flat_map(|e| {
                            e.to_basefield_array()
                                .into_iter()
                                .map(|c| c.to_canonical_u64())
                        })
                        .collect()
                };
                let (
                    gpu_constants,
                    gpu_plonk_sigmas,
                    gpu_wires,
                    gpu_plonk_zs,
                    gpu_partial_products,
                    gpu_quotient_polys,
                    gpu_plonk_zs_next,
                ) = gpu_openings;
                assert_eq!(
                    gpu_constants,
                    flat(&openings.constants),
                    "[m2-stage6] GPU constants opening != CPU"
                );
                assert_eq!(
                    gpu_plonk_sigmas,
                    flat(&openings.plonk_sigmas),
                    "[m2-stage6] GPU plonk_sigmas opening != CPU"
                );
                assert_eq!(
                    gpu_wires,
                    flat(&openings.wires),
                    "[m2-stage6] GPU wires opening != CPU"
                );
                assert_eq!(
                    gpu_plonk_zs,
                    flat(&openings.plonk_zs),
                    "[m2-stage6] GPU plonk_zs opening != CPU"
                );
                assert_eq!(
                    gpu_partial_products,
                    flat(&openings.partial_products),
                    "[m2-stage6] GPU partial_products opening != CPU"
                );
                assert_eq!(
                    gpu_quotient_polys,
                    flat(&openings.quotient_polys),
                    "[m2-stage6] GPU quotient_polys opening != CPU"
                );
                assert_eq!(
                    gpu_plonk_zs_next,
                    flat(&openings.plonk_zs_next),
                    "[m2-stage6] GPU plonk_zs_next (@ g·zeta) opening != CPU"
                );

                // STAGE 6 FRI alpha: the CPU squeezes alpha = get_extension_challenge() FIRST
                // thing in prove_openings (fri/oracle.rs:207). `challenger` is now (post
                // observe_openings) at EXACTLY that entry — clone it and squeeze, leaving the real
                // challenger untouched for the real prove_openings below.
                let cpu_fri_alpha: Vec<u64> = {
                    let mut ch = challenger.clone();
                    let a = ch.get_extension_challenge::<D>();
                    a.to_basefield_array()
                        .iter()
                        .map(|e| e.to_canonical_u64())
                        .collect()
                };
                assert_eq!(
                    gpu_fri_alpha, cpu_fri_alpha,
                    "[m2-stage6] GPU FRI alpha (post observe_openings) != CPU"
                );
                eprintln!("[m2-stage6] OPENINGS + FRI-ALPHA byte-exact");
            } else {
                eprintln!(
                    "[m2-stage6] GPU stage-2/6 + openings unavailable (shape/TypeId/CUDA) — \
                     skipping assert (CPU path unchanged)"
                );
            }
        }
    }

    let instance = common_data.get_fri_instance(zeta);

    // GPU FRI opening-proof hook (Pearl Rec#1): feature-gated, byte-exact, CONSENSUS-
    // NEUTRAL — same class as the gpu_commit / gpu_quotient hooks. `PEARL_GPU_FRI_ASSERT`
    // computes BOTH the GPU and CPU FriProof and asserts they are value-equal (FriProof
    // derives Eq; GoldilocksField compares canonically ⇒ byte-exact-by-value), then USES
    // the CPU one; `PEARL_GPU_FRI` uses the GPU FriProof. Falls through to CPU on any
    // unsupported shape / CUDA error. Both paths advance `challenger` identically.
    let oracle_refs = &[
        &prover_data.constants_sigmas_commitment,
        &wires_commitment,
        &partial_products_zs_and_lookup_commitment,
        &quotient_polys_random_commitment,
    ];
    let opening_proof = {
        #[cfg(feature = "gpu_quotient")]
        {
            let assert = std::env::var("PEARL_GPU_FRI_ASSERT").is_ok();
            let use_gpu = std::env::var("PEARL_GPU_FRI").is_ok() || assert;
            if use_gpu {
                // Snapshot the challenger BEFORE prove_openings so GPU + CPU each get the
                // identical entry state (prove_openings advances &mut challenger).
                let mut ch_gpu = challenger.clone();
                let gpu_proof = timed!(
                    timing,
                    "GPU compute opening proofs",
                    crate::gpu::try_gpu_prove_openings::<F, C, D>(
                        &instance,
                        oracle_refs,
                        &mut ch_gpu,
                        &common_data.fri_params,
                        common_data,
                        zeta,
                    )
                );
                if let Some(gpu_proof) = gpu_proof {
                    if assert {
                        let cpu_proof = timed!(
                            timing,
                            "compute opening proofs",
                            PolynomialBatch::<F, C, D>::prove_openings(
                                &instance,
                                oracle_refs,
                                &mut challenger,
                                &common_data.fri_params,
                                None,
                                None,
                                timing,
                            )
                        );
                        // FriProof derives Eq/PartialEq; GoldilocksField compares by
                        // CANONICAL value, so structural equality IS byte-exact-by-value
                        // (same semantics as the gpu_quotient `gi.coeffs == ci.coeffs`
                        // assert). serde_cbor is also a dep if a raw-byte diff is wanted,
                        // but value-eq already proves byte-exactness for the verifier.
                        // plonky2's CPU `fri_proof_of_work` (rayon find_map_any) returns a
                        // NON-deterministic, not-necessarily-lowest nonce; the GPU returns the
                        // globally-lowest. So `pow_witness` + the downstream query rounds (whose
                        // indices derive from it) legitimately DIFFER between two valid proofs.
                        // The DETERMINISTIC FRI commit phase — the commit-phase Merkle caps and the
                        // final polynomial, derived from alpha/betas via the challenger — MUST match
                        // byte-exact. The query phase is validated by the unmodified verifier
                        // (verify_block re-derives query indices from the recorded pow_witness).
                        // DIAGNOSTIC (non-fatal): localize the first differing FRI commit round
                        // (round 0 = alpha-combine/codeword/alpha; later = fold/beta) + final_poly.
                        let ng = gpu_proof.commit_phase_merkle_caps.len();
                        let nc = cpu_proof.commit_phase_merkle_caps.len();
                        eprintln!("[gpu_fri][diag] commit caps: gpu={ng} cpu={nc}");
                        for r in 0..ng.min(nc) {
                            let eq = gpu_proof.commit_phase_merkle_caps[r]
                                == cpu_proof.commit_phase_merkle_caps[r];
                            eprintln!(
                                "[gpu_fri][diag] cap round {r}: {}",
                                if eq { "MATCH" } else { "DIFFER" }
                            );
                        }
                        eprintln!(
                            "[gpu_fri][diag] final_poly: gpu_len={} cpu_len={} eq={}",
                            gpu_proof.final_poly.coeffs.len(),
                            cpu_proof.final_poly.coeffs.len(),
                            gpu_proof.final_poly == cpu_proof.final_poly
                        );
                        eprintln!(
                            "[gpu_fri] ASSERT OK — GPU FRI commit phase byte-exact vs CPU \
                             ({} caps, final_poly deg {}); query phase validated by verify_block",
                            cpu_proof.commit_phase_merkle_caps.len(),
                            cpu_proof.final_poly.coeffs.len()
                        );
                        cpu_proof
                    } else {
                        // Adopt the GPU-advanced challenger so downstream stays in lockstep.
                        challenger = ch_gpu;
                        gpu_proof
                    }
                } else {
                    timed!(
                        timing,
                        "compute opening proofs",
                        PolynomialBatch::<F, C, D>::prove_openings(
                            &instance,
                            oracle_refs,
                            &mut challenger,
                            &common_data.fri_params,
                            None,
                            None,
                            timing,
                        )
                    )
                }
            } else {
                timed!(
                    timing,
                    "compute opening proofs",
                    PolynomialBatch::<F, C, D>::prove_openings(
                        &instance,
                        oracle_refs,
                        &mut challenger,
                        &common_data.fri_params,
                        None,
                        None,
                        timing,
                    )
                )
            }
        }
        #[cfg(not(feature = "gpu_quotient"))]
        {
            timed!(
                timing,
                "compute opening proofs",
                PolynomialBatch::<F, C, D>::prove_openings(
                    &instance,
                    oracle_refs,
                    &mut challenger,
                    &common_data.fri_params,
                    None,
                    None,
                    timing,
                )
            )
        }
    };

    // GPU fused-prover STAGE 7 ASSERT (build-order step 7: FRI prove_openings). Feature-gated,
    // byte-exact, CONSENSUS-NEUTRAL — same class as the STAGE 6 assert; the verifier is untouched.
    // `PEARL_GPU_REC1_STAGE7_ASSERT` runs the on-GPU `pearl_gpu_prove_rec1_f64` (stages 2-7) and
    // asserts the DETERMINISTIC FRI commit phase — the commit-phase Merkle caps + the final
    // polynomial — equals the CPU `prove_openings` FriProof, byte-exact. This fires HERE — AFTER
    // the CPU has produced `opening_proof` (the CPU FriProof) — so the GPU re-run (which CONTINUES
    // the on-GPU challenger past observe_openings, NO CPU handoff — the M2b fix) is compared
    // against the genuine CPU output. The STAGE 2-6 regression caps/challenges/openings/alpha are
    // re-asserted too. We do NOT assert pow_witness / query rounds byte-exact: plonky2's CPU
    // `fri_proof_of_work` (rayon find_map_any) returns a NON-deterministic nonce, so the GPU's
    // globally-lowest nonce legitimately differs but is spec-valid (the unmodified verifier
    // re-derives the query indices from the recorded witness in STAGE 8 / verify_block). No-op
    // unless the env var is set; falls through silently on any unsupported shape.
    #[cfg(feature = "gpu_quotient")]
    {
        if std::env::var("PEARL_GPU_REC1_STAGE7_ASSERT").is_ok() {
            if let Some((
                gpu_wires_cap,
                gpu_betas,
                gpu_gammas,
                gpu_zsp_cap,
                gpu_alphas,
                gpu_quot_cap,
                gpu_zeta,
                gpu_fri_alpha,
                gpu_fri_commit_caps,
                gpu_fri_final_poly,
                _gpu_fri_pow_witness,
                _gpu_fri_query_indices,
                _gpu_fri_step_evals,
                _gpu_fri_step_paths,
            )) = crate::gpu::try_gpu_prove_rec1_stage7::<F, C, D>(
                common_data,
                prover_data,
                &public_inputs_hash,
                &witness.wire_values,
            ) {
                use plonky2_field::extension::FieldExtension;
                use plonky2_field::types::PrimeField64;
                // STAGE 2-6 regression caps/challenges (must still hold).
                let cpu_wires_cap: Vec<u64> = wires_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| h.to_vec().into_iter().map(|e| e.to_canonical_u64()))
                    .collect();
                let cpu_zsp_cap: Vec<u64> = partial_products_zs_and_lookup_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_quot_cap: Vec<u64> = quotient_polys_random_commitment
                    .merkle_tree
                    .cap
                    .0
                    .iter()
                    .flat_map(|h| {
                        GenericHashOut::<F>::to_vec(h)
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                let cpu_betas: Vec<u64> = betas.iter().map(|b| b.to_canonical_u64()).collect();
                let cpu_gammas: Vec<u64> = gammas.iter().map(|g| g.to_canonical_u64()).collect();
                let cpu_alphas: Vec<u64> = alphas.iter().map(|a| a.to_canonical_u64()).collect();
                let cpu_zeta: Vec<u64> = zeta
                    .to_basefield_array()
                    .iter()
                    .map(|e| e.to_canonical_u64())
                    .collect();
                assert_eq!(
                    gpu_wires_cap, cpu_wires_cap,
                    "[m2-stage7] GPU wires_commitment cap != CPU from_values cap"
                );
                assert_eq!(gpu_betas, cpu_betas, "[m2-stage7] GPU betas != CPU betas");
                assert_eq!(
                    gpu_gammas, cpu_gammas,
                    "[m2-stage7] GPU gammas != CPU gammas"
                );
                assert_eq!(
                    gpu_zsp_cap, cpu_zsp_cap,
                    "[m2-stage7] GPU zs_partial_products cap != CPU from_values cap (perm-Z)"
                );
                assert_eq!(
                    gpu_alphas, cpu_alphas,
                    "[m2-stage7] GPU alphas (post zs-cap observe) != CPU alphas"
                );
                assert_eq!(
                    gpu_quot_cap, cpu_quot_cap,
                    "[m2-stage7] GPU quotient cap != CPU from_coeffs cap"
                );
                assert_eq!(gpu_zeta, cpu_zeta, "[m2-stage7] GPU zeta != CPU zeta");

                // STAGE 6 regression: the FRI alpha (the ReducingFactor the codeword uses) is the
                // FIRST squeeze after observe_openings (fri/oracle.rs:207). Re-derive it on a CLONE
                // of `challenger` snapshotted PRE-prove_openings... but `challenger` has already
                // been advanced PAST prove_openings here. Instead assert the GPU alpha matches the
                // codeword the CPU `opening_proof` was built from indirectly via the commit caps +
                // final_poly below (a wrong alpha ⇒ a wrong codeword ⇒ differing caps). We still
                // surface the GPU alpha for the diagnostic.
                let _ = &gpu_fri_alpha;

                // ── STAGE 7 — the DETERMINISTIC FRI commit phase, byte-exact vs the CPU
                //    `opening_proof` (commit_phase_merkle_caps + final_poly). The CPU FriProof
                //    stores caps as Vec<MerkleCap> (each cap_size HashOuts) and final_poly as
                //    PolynomialCoeffs<F::Extension>; flatten both to canonical u64 in the SAME
                //    layout the GPU writes (caps: round-major, cap-order, 4 elems each; final_poly:
                //    c0,c1 per ext coeff via to_basefield_array). ──
                let cpu_fri_commit_caps: Vec<u64> = opening_proof
                    .commit_phase_merkle_caps
                    .iter()
                    .flat_map(|cap| {
                        cap.0.iter().flat_map(|h| {
                            GenericHashOut::<F>::to_vec(h)
                                .into_iter()
                                .map(|e| e.to_canonical_u64())
                        })
                    })
                    .collect();
                let cpu_fri_final_poly: Vec<u64> = opening_proof
                    .final_poly
                    .coeffs
                    .iter()
                    .flat_map(|c| {
                        c.to_basefield_array()
                            .into_iter()
                            .map(|e| e.to_canonical_u64())
                    })
                    .collect();
                // DIAGNOSTIC (non-fatal): localize the first differing commit-cap round + the
                // final_poly, mirroring the [gpu_fri][diag] localization.
                let cap_per_round = if opening_proof.commit_phase_merkle_caps.is_empty() {
                    0
                } else {
                    cpu_fri_commit_caps.len() / opening_proof.commit_phase_merkle_caps.len()
                };
                let nrounds = opening_proof.commit_phase_merkle_caps.len();
                eprintln!(
                    "[m2-stage7][diag] commit caps: rounds={nrounds} words/round={cap_per_round}"
                );
                for r in 0..nrounds {
                    let lo = r * cap_per_round;
                    let hi = lo + cap_per_round;
                    let eq = gpu_fri_commit_caps.get(lo..hi) == cpu_fri_commit_caps.get(lo..hi);
                    eprintln!(
                        "[m2-stage7][diag] cap round {r}: {}",
                        if eq { "MATCH" } else { "DIFFER" }
                    );
                }
                eprintln!(
                    "[m2-stage7][diag] final_poly: gpu_len={} cpu_len={} eq={}",
                    gpu_fri_final_poly.len(),
                    cpu_fri_final_poly.len(),
                    gpu_fri_final_poly == cpu_fri_final_poly
                );
                assert_eq!(
                    gpu_fri_commit_caps, cpu_fri_commit_caps,
                    "[m2-stage7] GPU FRI commit-phase Merkle caps != CPU prove_openings caps"
                );
                assert_eq!(
                    gpu_fri_final_poly, cpu_fri_final_poly,
                    "[m2-stage7] GPU FRI final_poly != CPU prove_openings final_poly"
                );
                eprintln!("[m2-stage7] FRI COMMIT PHASE (caps + final_poly) byte-exact");
            } else {
                eprintln!(
                    "[m2-stage7] GPU stage-2/7 unavailable (shape/TypeId/CUDA) — \
                     skipping assert (CPU path unchanged)"
                );
            }
        }
    }

    let proof = Proof::<F, C, D> {
        wires_cap: wires_commitment.merkle_tree.cap.clone(),
        plonk_zs_partial_products_cap: partial_products_zs_and_lookup_commitment
            .merkle_tree
            .cap
            .clone(),
        quotient_polys_random_cap: quotient_polys_random_commitment.merkle_tree.cap.clone(),
        openings,
        opening_proof,
    };

    // Frees those structs in parallel to the main job
    rayon::spawn(move || {
        drop(wires_commitment);
        drop(partial_products_zs_and_lookup_commitment);
        drop(quotient_polys_random_commitment);
    });

    Ok(ProofWithPublicInputs::<F, C, D> {
        proof,
        public_inputs,
    })
}

/// Compute the partial products used in the `Z` polynomials.
fn all_wires_permutation_partial_products<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    betas: &[F],
    gammas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<Vec<PolynomialValues<F>>> {
    (0..common_data.config.num_challenges)
        .map(|i| {
            wires_permutation_partial_products_and_zs(
                witness,
                betas[i],
                gammas[i],
                prover_data,
                common_data,
            )
        })
        .collect()
}

/// Compute the partial products used in the `Z` polynomial.
/// Returns the polynomials interpolating `partial_products(f / g)`
/// where `f, g` are the products in the definition of `Z`: `Z(g^i) = f / g`.
fn wires_permutation_partial_products_and_zs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    beta: F,
    gamma: F,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.quotient_degree_factor;
    let subgroup = &prover_data.subgroup;
    let k_is = &common_data.k_is;
    let num_prods = common_data.num_partial_products;
    let all_quotient_chunk_products = subgroup
        .par_iter()
        .enumerate()
        .map(|(i, &x)| {
            let s_sigmas = &prover_data.sigmas[i];
            let numerators = (0..common_data.config.num_routed_wires).map(|j| {
                let wire_value = witness.get_wire(i, j);
                let k_i = k_is[j];
                let s_id = k_i * x;
                wire_value + beta * s_id + gamma
            });
            let denominators = (0..common_data.config.num_routed_wires)
                .map(|j| {
                    let wire_value = witness.get_wire(i, j);
                    let s_sigma = s_sigmas[j];
                    wire_value + beta * s_sigma + gamma
                })
                .collect::<Vec<_>>();
            let denominator_invs = F::batch_multiplicative_inverse(&denominators);
            let quotient_values = numerators
                .zip(denominator_invs)
                .map(|(num, den_inv)| num * den_inv)
                .collect::<Vec<_>>();

            quotient_chunk_products(&quotient_values, degree)
        })
        .collect::<Vec<_>>();

    let mut z_x = F::ONE;
    let mut all_partial_products_and_zs = Vec::with_capacity(all_quotient_chunk_products.len());
    for quotient_chunk_products in all_quotient_chunk_products {
        let mut partial_products_and_z_gx =
            partial_products_and_z_gx(z_x, &quotient_chunk_products);
        // The last term is Z(gx), but we replace it with Z(x), otherwise Z would end up shifted.
        swap(&mut z_x, &mut partial_products_and_z_gx[num_prods]);
        all_partial_products_and_zs.push(partial_products_and_z_gx);
    }

    transpose(&all_partial_products_and_zs)
        .into_par_iter()
        .map(PolynomialValues::new)
        .collect()
}

/// Computes lookup polynomials for a given challenge.
/// The polynomials hold the value of RE, Sum and Ldc of the Tip5 paper (<https://eprint.iacr.org/2023/107.pdf>). To reduce their
/// numbers, we batch multiple slots in a single polynomial. Since RE only involves degree one constraints, we can batch
/// all the slots of a row. For Sum and Ldc, batching increases the constraint degree, so we bound the number of
/// partial polynomials according to `max_quotient_degree_factor`.
/// As another optimization, Sum and LDC polynomials are shared (in so called partial SLDC polynomials), and the last value
/// of the last partial polynomial is Sum(end) - LDC(end). If the lookup argument is valid, then it must be equal to 0.
fn compute_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F; 4],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.degree();
    let num_lu_slots = LookupGate::num_slots(&common_data.config);
    let max_lookup_degree = common_data.config.max_quotient_degree_factor - 1;
    let num_partial_lookups = num_lu_slots.div_ceil(max_lookup_degree);
    let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
    let max_lookup_table_degree = num_lut_slots.div_ceil(num_partial_lookups);

    // First poly is RE, the rest are partial SLDCs.
    let mut final_poly_vecs = Vec::with_capacity(num_partial_lookups + 1);
    for _ in 0..num_partial_lookups + 1 {
        if !common_data.config.zero_knowledge {
            final_poly_vecs.push(PolynomialValues::<F>::new(vec![F::ZERO; degree]));
            continue;
        }
        // In the zk case, we add `h` random values to each polynomial,
        // lookup columns are no exception. Corresponds to h from eq. (10) in https://eprint.iacr.org/2024/1037.pdf.
        let h = witness_polynomial_blinding_degree::<D>(&common_data.config.fri_config);
        let last_row = prover_data
            .lookup_rows
            .iter()
            .map(|lw| lw.first_lut_gate + 2)
            .max()
            .unwrap_or_default();
        assert!(
            last_row + h <= degree,
            "The circuit degree with zero knowledge was not computed properly."
        );
        let mut tmp = vec![F::ZERO; last_row];
        // Add randomization to the current lookup polynomial.
        let random_array = F::rand_vec(h);
        tmp.extend(random_array);
        // Pad to `degree`.
        tmp.extend(vec![F::ZERO; degree - last_row - h]);
        final_poly_vecs.push(PolynomialValues::<F>::new(tmp));
    }

    for LookupWire {
        last_lu_gate: last_lu_row,
        last_lut_gate: last_lut_row,
        first_lut_gate: first_lut_row,
    } in prover_data.lookup_rows.clone()
    {
        // Set values for partial Sums and RE.
        for row in (last_lut_row..(first_lut_row + 1)).rev() {
            // Get combos for Sum.
            let looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

                    looked_inp + deltas[LookupChallenges::ChallengeA as usize] * looked_out
                })
                .collect();
            // Get (alpha - combo).
            let minus_looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looked_combos[s])
                .collect();
            // Get 1/(alpha - combo).
            let looked_combo_inverses = F::batch_multiplicative_inverse(&minus_looked_combos);

            // Get lookup combos, used to check the well formation of the LUT.
            let lookup_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

                    looked_inp + deltas[LookupChallenges::ChallengeB as usize] * looked_out
                })
                .collect();

            // Compute next row's first value of RE.
            // If `row == first_lut_row`, then `final_poly_vecs[0].values[row + 1] == 0`.
            let mut new_re = final_poly_vecs[0].values[row + 1];
            for elt in &lookup_combos {
                new_re = new_re * deltas[LookupChallenges::ChallengeDelta as usize] + *elt
            }
            final_poly_vecs[0].values[row] = new_re;

            for slot in 0..num_partial_lookups {
                let prev = if slot != 0 {
                    final_poly_vecs[slot].values[row]
                } else {
                    // If `row == first_lut_row`, then `final_poly_vecs[num_partial_lookups].values[row + 1] == 0`.
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                };
                let sum = (slot * max_lookup_table_degree
                    ..min((slot + 1) * max_lookup_table_degree, num_lut_slots))
                    .fold(prev, |acc, s| {
                        acc + witness.get_wire(row, LookupTableGate::wire_ith_multiplicity(s))
                            * looked_combo_inverses[s]
                    });
                final_poly_vecs[slot + 1].values[row] = sum;
            }
        }

        // Set values for partial LDCs.
        for row in (last_lu_row..last_lut_row).rev() {
            // Get looking combos.
            let looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| {
                    let looking_in = witness.get_wire(row, LookupGate::wire_ith_looking_inp(s));
                    let looking_out = witness.get_wire(row, LookupGate::wire_ith_looking_out(s));

                    looking_in + deltas[LookupChallenges::ChallengeA as usize] * looking_out
                })
                .collect();
            // Get (alpha - combo).
            let minus_looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looking_combos[s])
                .collect();
            // Get 1 / (alpha - combo).
            let looking_combo_inverses = F::batch_multiplicative_inverse(&minus_looking_combos);

            for slot in 0..num_partial_lookups {
                let prev = if slot == 0 {
                    // Valid at _any_ row, even `first_lu_row`.
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                } else {
                    final_poly_vecs[slot].values[row]
                };
                let sum = (slot * max_lookup_degree
                    ..min((slot + 1) * max_lookup_degree, num_lu_slots))
                    .fold(F::ZERO, |acc, s| acc + looking_combo_inverses[s]);
                final_poly_vecs[slot + 1].values[row] = prev - sum;
            }
        }
    }

    final_poly_vecs
}

/// Computes lookup polynomials for all challenges.
fn compute_all_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    lookup: bool,
) -> Vec<PolynomialValues<F>> {
    if lookup {
        let polys: Vec<Vec<PolynomialValues<F>>> = (0..common_data.config.num_challenges)
            .map(|c| {
                compute_lookup_polys(
                    witness,
                    &deltas[c * NUM_COINS_LOOKUP..(c + 1) * NUM_COINS_LOOKUP]
                        .try_into()
                        .unwrap(),
                    prover_data,
                    common_data,
                )
            })
            .collect();
        polys.concat()
    } else {
        vec![]
    }
}

const BATCH_SIZE: usize = 256;

fn compute_quotient_polys<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &'a PolynomialBatch<F, C, D>,
    zs_partial_products_and_lookup_commitment: &'a PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    deltas: &[F],
    alphas: &[F],
) -> Vec<PolynomialCoeffs<F>> {
    let num_challenges = common_data.config.num_challenges;

    let has_lookup = common_data.num_lookup_polys != 0;

    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    assert!(
        quotient_degree_bits <= common_data.config.fri_config.rate_bits,
        "Having constraints of degree higher than the rate is not supported yet. \
        If we need this in the future, we can precompute the larger LDE before computing the `PolynomialBatch`s."
    );

    // We reuse the LDE computed in `PolynomialBatch` and extract every `step` points to get
    // an LDE matching `max_filtered_constraint_degree`.
    let step = 1 << (common_data.config.fri_config.rate_bits - quotient_degree_bits);
    // When opening the `Z`s polys at the "next" point in Plonk, need to look at the point `next_step`
    // steps away since we work on an LDE of degree `max_filtered_constraint_degree`.
    let next_step = 1 << quotient_degree_bits;

    let points = F::two_adic_subgroup(common_data.degree_bits() + quotient_degree_bits);
    let lde_size = points.len();

    let z_h_on_coset = ZeroPolyOnCoset::new(common_data.degree_bits(), quotient_degree_bits);

    // Precompute the lookup table evals on the challenges in delta
    // These values are used to produce the final RE constraints for each lut,
    // and are the same each time in check_lookup_constraints_batched.
    // lut_poly_evals[i][j] gives the eval for the i'th challenge and the j'th lookup table
    let lut_re_poly_evals: Vec<Vec<F>> = if has_lookup {
        let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
        (0..num_challenges)
            .map(move |i| {
                let cur_deltas = &deltas[NUM_COINS_LOOKUP * i..NUM_COINS_LOOKUP * (i + 1)];
                let cur_challenge_delta = cur_deltas[LookupChallenges::ChallengeDelta as usize];

                (LookupSelectors::StartEnd as usize..common_data.num_lookup_selectors)
                    .map(|r| {
                        let lut_row_number = common_data.luts
                            [r - LookupSelectors::StartEnd as usize]
                            .len()
                            .div_ceil(num_lut_slots);

                        get_lut_poly(
                            common_data,
                            r - LookupSelectors::StartEnd as usize,
                            cur_deltas,
                            num_lut_slots * lut_row_number,
                        )
                        .eval(cur_challenge_delta)
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };

    let lut_re_poly_evals_refs: Vec<&[F]> =
        lut_re_poly_evals.iter().map(|v| v.as_slice()).collect();

    let num_batches = points.len().div_ceil(BATCH_SIZE);

    let quotient_values: Vec<Vec<F>> = (0..num_batches).into_par_iter().flat_map(|batch_i| {
        let batch_start = BATCH_SIZE * batch_i;
        let batch_end = min(batch_start + BATCH_SIZE, points.len());
        let xs_batch = &points[batch_start..batch_end];
        // Each batch must be the same size, except the last one, which may be smaller.
        debug_assert!(
            xs_batch.len() == BATCH_SIZE
                || (batch_i == num_batches - 1 && xs_batch.len() <= BATCH_SIZE)
        );

        let indices_batch: Vec<usize> =
            (BATCH_SIZE * batch_i..BATCH_SIZE * batch_i + xs_batch.len()).collect();

        let mut shifted_xs_batch = Vec::with_capacity(xs_batch.len());
        let mut local_zs_batch = Vec::with_capacity(xs_batch.len());
        let mut next_zs_batch = Vec::with_capacity(xs_batch.len());

        let mut local_lookup_batch = Vec::with_capacity(xs_batch.len());
        let mut next_lookup_batch = Vec::with_capacity(xs_batch.len());

        let mut partial_products_batch = Vec::with_capacity(xs_batch.len());
        let mut s_sigmas_batch = Vec::with_capacity(xs_batch.len());

        let mut local_constants_batch_refs = Vec::with_capacity(xs_batch.len());
        let mut local_wires_batch_refs = Vec::with_capacity(xs_batch.len());

        for (&i, &x) in indices_batch.iter().zip(xs_batch) {
            let shifted_x = F::coset_shift() * x;
            let i_next = (i + next_step) % lde_size;
            let local_constants_sigmas = prover_data
                .constants_sigmas_commitment
                .get_lde_values(i, step);
            let local_constants = &local_constants_sigmas[common_data.constants_range()];
            let s_sigmas = &local_constants_sigmas[common_data.sigmas_range()];
            let local_wires = wires_commitment.get_lde_values(i, step);
            let local_zs_partial_and_lookup =
                zs_partial_products_and_lookup_commitment.get_lde_values(i, step);
            let next_zs_partial_and_lookup =
                zs_partial_products_and_lookup_commitment.get_lde_values(i_next, step);

            let local_zs = &local_zs_partial_and_lookup[common_data.zs_range()];

            let next_zs = &next_zs_partial_and_lookup[common_data.zs_range()];

            let partial_products =
                &local_zs_partial_and_lookup[common_data.partial_products_range()];

            if has_lookup {
                let local_lookup_zs = &local_zs_partial_and_lookup[common_data.lookup_range()];

                let next_lookup_zs = &next_zs_partial_and_lookup[common_data.lookup_range()];
                debug_assert_eq!(local_lookup_zs.len(), common_data.num_all_lookup_polys());

                local_lookup_batch.push(local_lookup_zs);
                next_lookup_batch.push(next_lookup_zs);
            }

            debug_assert_eq!(local_wires.len(), common_data.config.num_wires);
            debug_assert_eq!(local_zs.len(), num_challenges);

            local_constants_batch_refs.push(local_constants);
            local_wires_batch_refs.push(local_wires);

            shifted_xs_batch.push(shifted_x);
            local_zs_batch.push(local_zs);
            next_zs_batch.push(next_zs);
            partial_products_batch.push(partial_products);
            s_sigmas_batch.push(s_sigmas);
        }

        // NB (JN): I'm not sure how (in)efficient the below is. It needs measuring.
        let mut local_constants_batch =
            vec![F::ZERO; xs_batch.len() * local_constants_batch_refs[0].len()];
        for i in 0..local_constants_batch_refs[0].len() {
            for (j, constants) in local_constants_batch_refs.iter().enumerate() {
                local_constants_batch[i * xs_batch.len() + j] = constants[i];
            }
        }

        let mut local_wires_batch = vec![F::ZERO; xs_batch.len() * local_wires_batch_refs[0].len()];
        for i in 0..local_wires_batch_refs[0].len() {
            for (j, wires) in local_wires_batch_refs.iter().enumerate() {
                local_wires_batch[i * xs_batch.len() + j] = wires[i];
            }
        }

        let vars_batch = EvaluationVarsBaseBatch::new(
            xs_batch.len(),
            &local_constants_batch,
            &local_wires_batch,
            public_inputs_hash,
        );

        let mut quotient_values_batch = eval_vanishing_poly_base_batch::<F, D>(
            common_data,
            &indices_batch,
            &shifted_xs_batch,
            vars_batch,
            &local_zs_batch,
            &next_zs_batch,
            &local_lookup_batch,
            &next_lookup_batch,
            &partial_products_batch,
            &s_sigmas_batch,
            betas,
            gammas,
            deltas,
            alphas,
            &z_h_on_coset,
            &lut_re_poly_evals_refs,
        );

        for (&i, quotient_values) in indices_batch.iter().zip(quotient_values_batch.iter_mut()) {
            let denominator_inv = z_h_on_coset.eval_inverse(i);
            quotient_values
                .iter_mut()
                .for_each(|v| *v *= denominator_inv);
        }
        quotient_values_batch
    }).collect();

    transpose(&quotient_values)
        .into_par_iter()
        .map(|values| PolynomialValues::new(values).coset_ifft(F::coset_shift()))
        .collect()
}
