//! Recursive verifier for batched multi-STARK proofs, i.e. where the
//! verification of a [`BatchStarkProof`][crate::batch_proof::BatchStarkProof]
//! is encoded in a plonky2 circuit.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use anyhow::{bail, ensure, Result};
use hashbrown::HashMap;
use itertools::Itertools;
use plonky2::field::extension::Extendable;
use plonky2::fri::witness_util::set_fri_proof_target;
use plonky2::hash::hash_types::{HashOutTarget, MerkleCapTarget, RichField};
use plonky2::iop::challenger::RecursiveChallenger;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::iop::witness::WitnessWrite;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};
use plonky2::util::reducing::ReducingFactorTarget;
use plonky2::with_context;

use crate::batch_proof::{
    BatchStarkProof, BatchStarkProofTarget, BatchStarkProofWithPublicInputs,
    BatchStarkProofWithPublicInputsTarget,
};
use crate::batch_prover::BatchStarkPreprocessedVerifierData;
use crate::batch_stark::{BatchOracle, BatchStark, BatchStarkLayout};
use crate::config::StarkConfig;
use crate::cross_table_lookup::{
    verify_cross_table_lookups_circuit, CrossTableLookup, CtlCheckVarsTarget,
};
use crate::get_challenges::get_dummy_polys_circuit;
use crate::lookup::get_grand_product_challenge_set_target;
use crate::proof::StarkOpeningSetTarget;

/// Circuit version of
/// [`BatchStarkPreprocessedVerifierData`]: the Merkle cap of the setup-time
/// preprocessed commitment, as circuit constants, and the per-table
/// preprocessed column indices.
#[derive(Debug, Clone)]
pub struct BatchStarkPreprocessedVerifierDataTarget {
    /// Constant targets for the Merkle cap of the setup-time commitment.
    pub cap: MerkleCapTarget,
    /// For each table, the sorted indices of its preprocessed columns.
    pub columns_per_table: Vec<Vec<usize>>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    BatchStarkPreprocessedVerifierData<F, C, D>
{
    /// Encodes this verifier data as constants in the circuit. The cap is
    /// baked into the circuit, binding it to this exact preprocessed data.
    pub fn constant_target(
        &self,
        builder: &mut CircuitBuilder<F, D>,
    ) -> BatchStarkPreprocessedVerifierDataTarget
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        BatchStarkPreprocessedVerifierDataTarget {
            cap: builder.constant_merkle_cap(&self.cap),
            columns_per_table: self.columns_per_table.clone(),
        }
    }
}

/// Circuit version of
/// [`BatchKnownColumns`][crate::batch_verifier::BatchKnownColumns]: an opaque
/// digest plus caller-supplied expected evaluations of each known column at
/// `zeta` and `g_t * zeta`, where `g_t` generates table `t`'s trace subgroup.
///
/// The evaluations are *inputs*:
/// [`verify_batch_stark_proof_circuit`] connects them to the proof openings
/// and returns `zeta`, and the caller is responsible for constraining them to
/// be the actual evaluations of the known columns at the returned `zeta`
/// (e.g. by evaluating a closed form, or the columns' constant coefficient
/// form, in-circuit).
#[derive(Debug, Clone)]
pub struct BatchKnownColumnsTarget<const D: usize> {
    /// Opaque digest targets absorbed by the Fiat-Shamir challenger
    /// (`None` = nothing absorbed). Must match the prover.
    pub digest: Option<HashOutTarget>,
    /// For each table, the sorted indices of its known columns within the
    /// table's trace.
    pub columns_per_table: Vec<Vec<usize>>,
    /// For each table, the expected evaluations at `zeta` of its known
    /// columns, in `columns_per_table` order.
    pub evals_at_zeta: Vec<Vec<ExtensionTarget<D>>>,
    /// For each table, the expected evaluations at `g_t * zeta` of its known
    /// columns, in `columns_per_table` order.
    pub evals_at_g_zeta: Vec<Vec<ExtensionTarget<D>>>,
}

impl<const D: usize> BatchKnownColumnsTarget<D> {
    /// Circuit analogue of `BatchKnownColumns::observe`: binds the
    /// known-column indices (as constants) and the digest into the
    /// Fiat-Shamir transcript.
    pub(crate) fn observe<F, H>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        challenger: &mut RecursiveChallenger<F, H, D>,
    ) where
        F: RichField + Extendable<D>,
        H: AlgebraicHasher<F>,
    {
        for columns in &self.columns_per_table {
            // Length-prefix each table's indices for domain separation.
            let len = builder.constant(F::from_canonical_usize(columns.len()));
            challenger.observe_element(len);
            for &c in columns {
                let c = builder.constant(F::from_canonical_usize(c));
                challenger.observe_element(c);
            }
        }
        if let Some(digest) = &self.digest {
            challenger.observe_elements(&digest.elements);
        }
    }

    /// Checks this data against the batch shape: per-table lengths, sorted
    /// in-range column indices, and disjointness from the setup-time
    /// preprocessed columns.
    pub(crate) fn validate(
        &self,
        num_columns: &[usize],
        prep_columns: &[Vec<usize>],
    ) -> Result<()> {
        let n = num_columns.len();
        ensure!(
            self.columns_per_table.len() == n,
            "Known columns: table count mismatch"
        );
        ensure!(self.evals_at_zeta.len() == n && self.evals_at_g_zeta.len() == n);
        for t in 0..n {
            let columns = &self.columns_per_table[t];
            ensure!(
                self.evals_at_zeta[t].len() == columns.len()
                    && self.evals_at_g_zeta[t].len() == columns.len(),
                "Table {t}: known column/eval count mismatch"
            );
            ensure!(
                columns.windows(2).all(|w| w[0] < w[1])
                    && columns.last().is_none_or(|&c| c < num_columns[t]),
                "Table {t}: known column indices must be sorted, distinct and in range"
            );
            ensure!(
                columns.iter().all(|c| !prep_columns[t].contains(c)),
                "Table {t}: a column cannot be both preprocessed and known"
            );
        }
        Ok(())
    }
}

/// Adds a new [`BatchStarkProofWithPublicInputsTarget`] to this circuit.
/// `degree_bits` are the per-table trace degrees (table-index order),
/// `prep_columns` the per-table preprocessed column indices (pass all-empty
/// when the batch has no preprocessed data) and `grouped_tables` the tables
/// sharing one oracle per role (pass `&[]` for all-solo).
pub fn add_virtual_batch_stark_proof_with_pis<F, const D: usize, const N: usize>(
    builder: &mut CircuitBuilder<F, D>,
    starks: &[&dyn BatchStark<F, D>; N],
    config: &StarkConfig,
    degree_bits: &[usize],
    prep_columns: &[Vec<usize>],
    cross_table_lookups: &[CrossTableLookup<F>],
    grouped_tables: &[usize],
) -> Result<BatchStarkProofWithPublicInputsTarget<D>>
where
    F: RichField + Extendable<D>,
{
    let proof = add_virtual_batch_stark_proof(
        builder,
        starks,
        config,
        degree_bits,
        prep_columns,
        cross_table_lookups,
        grouped_tables,
    )?;
    let public_inputs = starks
        .iter()
        .map(|stark| builder.add_virtual_targets(stark.num_public_inputs()))
        .collect();
    Ok(BatchStarkProofWithPublicInputsTarget {
        proof,
        public_inputs,
    })
}

/// Adds a new [`BatchStarkProofTarget`] to this circuit.
pub fn add_virtual_batch_stark_proof<F, const D: usize, const N: usize>(
    builder: &mut CircuitBuilder<F, D>,
    starks: &[&dyn BatchStark<F, D>; N],
    config: &StarkConfig,
    degree_bits: &[usize],
    prep_columns: &[Vec<usize>],
    cross_table_lookups: &[CrossTableLookup<F>],
    grouped_tables: &[usize],
) -> Result<BatchStarkProofTarget<D>>
where
    F: RichField + Extendable<D>,
{
    let layout = BatchStarkLayout::new(
        starks.as_slice(),
        config,
        degree_bits,
        prep_columns,
        cross_table_lookups,
        grouped_tables,
    )?;
    let fri_params = config.fri_params(layout.distinct_degree_bits[0]);
    let cap_height = fri_params.config.cap_height;

    let mut trace_caps = Vec::new();
    let mut auxiliary_polys_caps: Option<Vec<MerkleCapTarget>> = None;
    let mut quotient_polys_caps: Option<Vec<MerkleCapTarget>> = None;
    for &oracle in &layout.oracles {
        match oracle {
            BatchOracle::Trace(_) | BatchOracle::GroupedTrace => {
                trace_caps.push(builder.add_virtual_cap(cap_height))
            }
            // The preprocessed cap is a setup-time constant, not part of the proof.
            BatchOracle::Preprocessed => {}
            BatchOracle::Auxiliary(_) | BatchOracle::GroupedAuxiliary => {
                auxiliary_polys_caps
                    .get_or_insert_with(Vec::new)
                    .push(builder.add_virtual_cap(cap_height));
            }
            BatchOracle::Quotient(_) | BatchOracle::GroupedQuotient => {
                quotient_polys_caps
                    .get_or_insert_with(Vec::new)
                    .push(builder.add_virtual_cap(cap_height));
            }
        }
    }

    let openings = (0..N)
        .map(|t| {
            let num_aux = layout.num_aux_polys(t);
            StarkOpeningSetTarget {
                local_values: builder.add_virtual_extension_targets(layout.num_columns[t]),
                next_values: builder.add_virtual_extension_targets(layout.num_columns[t]),
                auxiliary_polys: (num_aux > 0)
                    .then(|| builder.add_virtual_extension_targets(num_aux)),
                auxiliary_polys_next: (num_aux > 0)
                    .then(|| builder.add_virtual_extension_targets(num_aux)),
                ctl_zs_first: (layout.num_ctl_zs[t] > 0)
                    .then(|| builder.add_virtual_targets(layout.num_ctl_zs[t])),
                quotient_polys: (layout.num_quotient_polys[t] > 0)
                    .then(|| builder.add_virtual_extension_targets(layout.num_quotient_polys[t])),
            }
        })
        .collect();

    let opening_proof = builder.add_virtual_batch_fri_proof(
        &layout.num_leaves_per_oracle(),
        &layout.oracle_degree_bits(),
        &fri_params,
    );

    Ok(BatchStarkProofTarget {
        trace_caps,
        auxiliary_polys_caps,
        quotient_polys_caps,
        openings,
        opening_proof,
        degree_bits: degree_bits.to_vec(),
    })
}

/// Encodes the verification of a [`BatchStarkProofWithPublicInputsTarget`] in
/// a circuit. Mirrors [`batch_verify`][crate::batch_verifier::batch_verify].
///
/// If `known_columns` is provided, its digest is absorbed by the Fiat-Shamir
/// challenger and its expected evaluations are connected to the corresponding
/// proof openings. It must be `Some` iff the prover was given the same data.
///
/// The keys of `ctl_extra_looking_sums` are positions within
/// `cross_table_lookups` of CTLs whose looking sums also include values not
/// associated with any table.
///
/// `grouped_tables` must equal the list given to the prover (see
/// [`batch_prove`][crate::batch_prover::batch_prove]).
///
/// Returns `zeta` so the caller can bind the known-column evaluations to it
/// (and/or connect it to a public input).
pub fn verify_batch_stark_proof_circuit<F, C, const D: usize, const N: usize>(
    builder: &mut CircuitBuilder<F, D>,
    starks: &[&dyn BatchStark<F, D>; N],
    config: &StarkConfig,
    proof_with_pis: &BatchStarkProofWithPublicInputsTarget<D>,
    cross_table_lookups: &[CrossTableLookup<F>],
    grouped_tables: &[usize],
    preprocessed: Option<&BatchStarkPreprocessedVerifierDataTarget>,
    known_columns: Option<&BatchKnownColumnsTarget<D>>,
    ctl_extra_looking_sums: &HashMap<usize, Vec<Target>>,
) -> Result<ExtensionTarget<D>>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    C::Hasher: AlgebraicHasher<F>,
{
    let proof = &proof_with_pis.proof;
    let public_inputs = &proof_with_pis.public_inputs;
    ensure!(proof.openings.len() == N);
    ensure!(proof.degree_bits.len() == N);
    ensure!(public_inputs.len() == N);
    for t in 0..N {
        ensure!(public_inputs[t].len() == starks[t].num_public_inputs());
    }
    ensure!(
        config.preprocessed_columns.is_empty(),
        "In batch mode, preprocessed columns are described by \
         `BatchStarkPreprocessedVerifierDataTarget`, not by `StarkConfig::preprocessed_columns`."
    );

    let empty_prep = vec![vec![]; N];
    let prep_columns = preprocessed.map_or(&empty_prep, |p| &p.columns_per_table);
    let layout = BatchStarkLayout::new(
        starks.as_slice(),
        config,
        &proof.degree_bits,
        prep_columns,
        cross_table_lookups,
        grouped_tables,
    )?;

    let has_aux = (0..N).any(|t| layout.num_aux_polys(t) > 0);
    let has_quotient = (0..N).any(|t| layout.num_quotient_polys[t] > 0);
    ensure!(proof.auxiliary_polys_caps.is_some() == has_aux);
    ensure!(proof.quotient_polys_caps.is_some() == has_quotient);
    let mut n_trace = 0;
    let mut has_prep = false;
    let mut n_aux = 0;
    let mut n_quot = 0;
    for &oracle in &layout.oracles {
        match oracle {
            BatchOracle::Trace(_) | BatchOracle::GroupedTrace => n_trace += 1,
            BatchOracle::Preprocessed => has_prep = true,
            BatchOracle::Auxiliary(_) | BatchOracle::GroupedAuxiliary => n_aux += 1,
            BatchOracle::Quotient(_) | BatchOracle::GroupedQuotient => n_quot += 1,
        }
    }
    ensure!(proof.trace_caps.len() == n_trace);
    if let Some(caps) = &proof.auxiliary_polys_caps {
        ensure!(caps.len() == n_aux);
    }
    if let Some(caps) = &proof.quotient_polys_caps {
        ensure!(caps.len() == n_quot);
    }
    match (preprocessed, has_prep) {
        (Some(_), true) | (None, false) => {}
        (Some(_), false) => bail!("unexpected preprocessed data"),
        (None, true) => bail!("missing preprocessed cap"),
    }
    if let Some(known) = known_columns {
        known.validate(&layout.num_columns, prep_columns)?;
    }

    // Rebuild the Fiat-Shamir transcript: public inputs, preprocessed cap,
    // known columns, config, trace caps.
    let mut challenger = RecursiveChallenger::<F, C::Hasher, D>::new(builder);
    for pis in public_inputs {
        challenger.observe_elements(pis);
    }
    if let Some(prep) = preprocessed {
        challenger.observe_cap(&prep.cap);
    }
    if let Some(known) = known_columns {
        known.observe(builder, &mut challenger);
    }
    config.observe_target(builder, &mut challenger);
    for cap in &proof.trace_caps {
        challenger.observe_cap(cap);
    }

    let challenge_set = has_aux.then(|| {
        get_grand_product_challenge_set_target(builder, &mut challenger, config.num_challenges)
    });
    if let Some(caps) = &proof.auxiliary_polys_caps {
        for cap in caps {
            challenger.observe_cap(cap);
        }
    }

    let lookup_challenges: Option<Vec<Target>> = challenge_set
        .as_ref()
        .map(|set| set.challenges.iter().map(|ch| ch.beta).collect());
    let lookup_challenges_of = |t: usize| -> Option<&Vec<Target>> {
        starks[t]
            .uses_lookups()
            .then(|| lookup_challenges.as_ref().unwrap())
    };

    // The real CTL check vars of every table, extracted from the openings.
    let ctl_vars_per_table: Vec<Option<Vec<CtlCheckVarsTarget<F, D>>>> = (0..N)
        .map(|t| {
            starks[t].requires_ctls().then(|| {
                CtlCheckVarsTarget::from_openings(
                    t,
                    &proof.openings[t],
                    cross_table_lookups,
                    challenge_set.as_ref().unwrap(),
                    layout.num_lookup_columns[t],
                    layout.num_ctl_helpers[t],
                    &layout.num_ctl_helpers_by_ctl[t],
                )
            })
        })
        .collect();

    // Constraint-binding grind: mirror the prover's transcript.
    let alphas_prime = challenger.get_n_challenges(builder, config.num_challenges);
    for t in 0..N {
        let pow_degree = core::cmp::max(2, starks[t].constraint_degree() + 1);
        let dummy_openings = get_dummy_polys_circuit::<F, C, D>(
            builder,
            &mut challenger,
            starks[t].num_columns(),
            layout.num_aux_polys(t),
            pow_degree,
        );

        let num_lookup_columns = layout.num_lookup_columns[t];
        let total_num_ctl_helpers = layout.num_ctl_helpers[t];
        let dummy_ctl_vars = ctl_vars_per_table[t].as_ref().map(|ctl_vars| {
            let mut start_index = 0;
            ctl_vars
                .iter()
                .enumerate()
                .map(|(i, ctl_check_vars)| {
                    let num_ctl_helper_cols = ctl_check_vars.helper_columns.len();
                    let helper_columns = dummy_openings.auxiliary_polys.as_ref().unwrap()
                        [num_lookup_columns + start_index
                            ..num_lookup_columns + start_index + num_ctl_helper_cols]
                        .to_vec();
                    let ctl_vars = CtlCheckVarsTarget::<F, D> {
                        helper_columns,
                        local_z: dummy_openings.auxiliary_polys.as_ref().unwrap()
                            [num_lookup_columns + total_num_ctl_helpers + i],
                        next_z: dummy_openings.auxiliary_polys_next.as_ref().unwrap()
                            [num_lookup_columns + total_num_ctl_helpers + i],
                        challenges: ctl_check_vars.challenges,
                        columns: ctl_check_vars.columns.clone(),
                        filter: ctl_check_vars.filter.clone(),
                    };
                    start_index += num_ctl_helper_cols;
                    ctl_vars
                })
                .collect::<Vec<_>>()
        });

        let zeta_prime = challenger.get_extension_challenge(builder);
        let degree_bits_target = builder.constant(F::from_canonical_usize(proof.degree_bits[t]));
        let constraint_evals = with_context!(
            builder,
            &format!("bind constraints of table {t}"),
            starks[t].eval_vanishing_poly_circuit(
                builder,
                &dummy_openings,
                dummy_ctl_vars.as_deref(),
                lookup_challenges_of(t),
                &public_inputs[t],
                alphas_prime.clone(),
                zeta_prime,
                proof.degree_bits[t],
                degree_bits_target,
                num_lookup_columns,
            )
        );
        challenger.observe_extension_elements(&constraint_evals);
    }

    let alphas = challenger.get_n_challenges(builder, config.num_challenges);
    if let Some(caps) = &proof.quotient_polys_caps {
        for cap in caps {
            challenger.observe_cap(cap);
        }
    }
    let zeta = challenger.get_extension_challenge(builder);

    // Observe the openings table by table (canonical, profile-independent order).
    for os in &proof.openings {
        os.observe(&mut challenger);
    }
    let zero = builder.zero();
    let fri_openings = layout.fri_openings_target(zero, &proof.openings);

    let fri_challenges = challenger.fri_challenges(
        builder,
        &proof.opening_proof.commit_phase_merkle_caps,
        &proof.opening_proof.final_poly,
        proof.opening_proof.pow_witness,
        &config.fri_config,
    );

    // Check each table's polynomial identities `vanishing(x) = Z_H(x) quotient(x)` at zeta.
    let one = builder.one_extension();
    for t in 0..N {
        if starks[t].quotient_degree_factor() == 0 {
            continue;
        }
        let degree_bits_target = builder.constant(F::from_canonical_usize(proof.degree_bits[t]));
        let vanishing_polys_zeta = with_context!(
            builder,
            &format!("evaluate the vanishing polynomial of table {t} at zeta"),
            starks[t].eval_vanishing_poly_circuit(
                builder,
                &proof.openings[t],
                ctl_vars_per_table[t].as_deref(),
                lookup_challenges_of(t),
                &public_inputs[t],
                alphas.clone(),
                zeta,
                proof.degree_bits[t],
                degree_bits_target,
                layout.num_lookup_columns[t],
            )
        );

        let zeta_pow_deg = builder.exp_power_of_2_extension(zeta, proof.degree_bits[t]);
        let z_h_zeta = builder.sub_extension(zeta_pow_deg, one);
        let quotient_polys = proof.openings[t]
            .quotient_polys
            .as_ref()
            .expect("Quotient polys should be provided");
        ensure!(
            vanishing_polys_zeta.len() * starks[t].quotient_degree_factor() == quotient_polys.len(),
            "Table {t}: vanishing/quotient polynomial count mismatch"
        );
        let mut scale = ReducingFactorTarget::new(zeta_pow_deg);
        for (i, chunk) in quotient_polys
            .chunks(starks[t].quotient_degree_factor())
            .enumerate()
        {
            let recombined_quotient = scale.reduce(chunk, builder);
            let computed_vanishing_poly = builder.mul_extension(z_h_zeta, recombined_quotient);
            builder.connect_extension(vanishing_polys_zeta[i], computed_vanishing_poly);
        }
    }

    // Known columns: bind the claimed openings of each known column to the
    // caller-supplied expected evaluations at zeta and `g_t * zeta`.
    if let Some(known) = known_columns {
        for t in 0..N {
            for (j, &c) in known.columns_per_table[t].iter().enumerate() {
                builder.connect_extension(
                    known.evals_at_zeta[t][j],
                    proof.openings[t].local_values[c],
                );
                builder.connect_extension(
                    known.evals_at_g_zeta[t][j],
                    proof.openings[t].next_values[c],
                );
            }
        }
    }

    // Check that the looking and looked CTL sums match across tables.
    let ctl_zs_first: [Vec<Target>; N] =
        core::array::from_fn(|t| proof.openings[t].ctl_zs_first.clone().unwrap_or_default());
    verify_cross_table_lookups_circuit::<F, D, N>(
        builder,
        cross_table_lookups.to_vec(),
        ctl_zs_first,
        ctl_extra_looking_sums,
        config,
    );

    // Check all the openings with a single batched FRI argument.
    let instances = layout.fri_instances_target(builder, zeta);
    let mut trace_i = 0;
    let mut aux_i = 0;
    let mut quot_i = 0;
    let mut caps = Vec::with_capacity(layout.oracles.len());
    for &oracle in &layout.oracles {
        match oracle {
            BatchOracle::Trace(_) | BatchOracle::GroupedTrace => {
                caps.push(proof.trace_caps[trace_i].clone());
                trace_i += 1;
            }
            BatchOracle::Preprocessed => {
                caps.push(preprocessed.expect("missing preprocessed data").cap.clone());
            }
            BatchOracle::Auxiliary(_) | BatchOracle::GroupedAuxiliary => {
                caps.push(
                    proof
                        .auxiliary_polys_caps
                        .as_ref()
                        .expect("missing auxiliary caps")[aux_i]
                        .clone(),
                );
                aux_i += 1;
            }
            BatchOracle::Quotient(_) | BatchOracle::GroupedQuotient => {
                caps.push(
                    proof
                        .quotient_polys_caps
                        .as_ref()
                        .expect("missing quotient caps")[quot_i]
                        .clone(),
                );
                quot_i += 1;
            }
        }
    }
    with_context!(
        builder,
        "verify batch FRI proof",
        builder.verify_batch_fri_proof::<C>(
            &layout.distinct_degree_bits,
            &instances,
            &fri_openings,
            &fri_challenges,
            &caps,
            &proof.opening_proof,
            &config.fri_params(layout.distinct_degree_bits[0]),
        )
    );

    Ok(zeta)
}

/// Sets the targets in a [`BatchStarkProofWithPublicInputsTarget`] to their
/// corresponding values in a [`BatchStarkProofWithPublicInputs`].
pub fn set_batch_stark_proof_with_pis_target<F, C: GenericConfig<D, F = F>, W, const D: usize>(
    witness: &mut W,
    proof_with_pis_target: &BatchStarkProofWithPublicInputsTarget<D>,
    proof_with_pis: &BatchStarkProofWithPublicInputs<F, C, D>,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    C::Hasher: AlgebraicHasher<F>,
    W: WitnessWrite<F>,
{
    ensure!(proof_with_pis_target.public_inputs.len() == proof_with_pis.public_inputs.len());
    for (pi_targets, pis) in proof_with_pis_target
        .public_inputs
        .iter()
        .zip(&proof_with_pis.public_inputs)
    {
        for (&pi_t, &pi) in pi_targets.iter().zip_eq(pis) {
            witness.set_target(pi_t, pi)?;
        }
    }
    set_batch_stark_proof_target(witness, &proof_with_pis_target.proof, &proof_with_pis.proof)
}

/// Sets the targets in a [`BatchStarkProofTarget`] to their corresponding
/// values in a [`BatchStarkProof`].
pub fn set_batch_stark_proof_target<F, C: GenericConfig<D, F = F>, W, const D: usize>(
    witness: &mut W,
    proof_target: &BatchStarkProofTarget<D>,
    proof: &BatchStarkProof<F, C, D>,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    C::Hasher: AlgebraicHasher<F>,
    W: WitnessWrite<F>,
{
    ensure!(proof_target.degree_bits == proof.degree_bits);
    ensure!(proof_target.trace_caps.len() == proof.trace_caps.len());
    for (cap_target, cap) in proof_target.trace_caps.iter().zip(&proof.trace_caps) {
        witness.set_cap_target(cap_target, cap)?;
    }
    debug_assert_eq!(
        proof_target.auxiliary_polys_caps.is_some(),
        proof.auxiliary_polys_caps.is_some()
    );
    if let (Some(cap_targets), Some(caps)) = (
        &proof_target.auxiliary_polys_caps,
        &proof.auxiliary_polys_caps,
    ) {
        ensure!(cap_targets.len() == caps.len());
        for (cap_target, cap) in cap_targets.iter().zip(caps) {
            witness.set_cap_target(cap_target, cap)?;
        }
    }
    debug_assert_eq!(
        proof_target.quotient_polys_caps.is_some(),
        proof.quotient_polys_caps.is_some()
    );
    if let (Some(cap_targets), Some(caps)) = (
        &proof_target.quotient_polys_caps,
        &proof.quotient_polys_caps,
    ) {
        ensure!(cap_targets.len() == caps.len());
        for (cap_target, cap) in cap_targets.iter().zip(caps) {
            witness.set_cap_target(cap_target, cap)?;
        }
    }

    ensure!(proof_target.openings.len() == proof.openings.len());
    for (ot, os) in proof_target.openings.iter().zip(&proof.openings) {
        for (&t, &v) in ot.local_values.iter().zip_eq(&os.local_values) {
            witness.set_extension_target(t, v)?;
        }
        for (&t, &v) in ot.next_values.iter().zip_eq(&os.next_values) {
            witness.set_extension_target(t, v)?;
        }
        debug_assert_eq!(ot.auxiliary_polys.is_some(), os.auxiliary_polys.is_some());
        for (&t, &v) in ot
            .auxiliary_polys
            .iter()
            .flatten()
            .zip_eq(os.auxiliary_polys.iter().flatten())
        {
            witness.set_extension_target(t, v)?;
        }
        debug_assert_eq!(
            ot.auxiliary_polys_next.is_some(),
            os.auxiliary_polys_next.is_some()
        );
        for (&t, &v) in ot
            .auxiliary_polys_next
            .iter()
            .flatten()
            .zip_eq(os.auxiliary_polys_next.iter().flatten())
        {
            witness.set_extension_target(t, v)?;
        }
        debug_assert_eq!(ot.ctl_zs_first.is_some(), os.ctl_zs_first.is_some());
        for (&t, &v) in ot
            .ctl_zs_first
            .iter()
            .flatten()
            .zip_eq(os.ctl_zs_first.iter().flatten())
        {
            witness.set_target(t, v)?;
        }
        debug_assert_eq!(ot.quotient_polys.is_some(), os.quotient_polys.is_some());
        for (&t, &v) in ot
            .quotient_polys
            .iter()
            .flatten()
            .zip_eq(os.quotient_polys.iter().flatten())
        {
            witness.set_extension_target(t, v)?;
        }
    }

    set_fri_proof_target(witness, &proof_target.opening_proof, &proof.opening_proof)
}
