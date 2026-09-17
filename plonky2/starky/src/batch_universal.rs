//! Universal (profile-independent) recursive verifier for batched multi-STARK
//! proofs: one circuit per family verifies any per-table degree profile inside
//! a build-time envelope.
//!
//! The monolithic [`crate::batch_recursive_verifier::verify_batch_stark_proof_circuit`]
//! bakes the tables' trace degrees into the circuit, so every degree profile
//! needs its own compiled verifier. This module instead treats each table's
//! degree bits `b_t` as a circuit *input* (a one-hot selection over
//! `candidates(t) = ladder ∩ [lo_t, hi_t]`) and computes every
//! height-dependent quantity in-circuit:
//!
//! - proof targets are allocated at the envelope maximum (`hi_t` per table);
//!   variable-depth Merkle walks are conditional per level and idle below
//!   their tree's real height
//!   ([`CircuitBuilder::conditional_verify_merkle_proof_to_cap_with_cap_index`]),
//!   while fixed-height oracles — including the shared preprocessed and
//!   grouped-role trees — use exact walks;
//! - the FRI fold runs over the fixed consensus `ladder` of degree boundaries;
//!   commit-phase layers above the batch's tallest oracle are carried through
//!   disabled: their caps are skipped in Fiat-Shamir by sponge-state
//!   selection, their folds are discarded by selection, and their query steps
//!   are Merkle-checked against prover-supplied all-zeros trees (position
//!   independent, hence satisfiable at any index — and binding nothing, since
//!   their caps never enter the transcript);
//! - each query index is materialized as a fixed, *top-aligned* bit array
//!   `A = index · 2^{lde_max − lde_job}` (bit `A[i] = Σ_s jm_s · bit_{i − shift_s}`
//!   over the one-hot job-max indicator `jm`), so every consumer — cap index,
//!   Merkle walk levels, fold coset bits, boundary subgroup points — reads
//!   fixed slots regardless of the profile;
//! - opened claims are folded with globally-canonical alpha powers
//!   ([`plonky2::batch_fri::AlphaRun`]): a claim's exponent depends only on
//!   `(oracle, polynomial, point)`, not on how degrees group tables into FRI
//!   instances, which is what makes the combine profile-independent.
//!
//! Fiat-Shamir equals the native batch verifier's transcript exactly: public
//! inputs, preprocessed cap, known columns, config, caps, per-table openings
//! (canonical order), then per-layer `observe(cap_i); β_i` where a skipped
//! layer selects the sponge state back. Every layer starts squeeze-aligned
//! with empty input buffer, and leftover sponge outputs are never consumed
//! across an observe, so the active branch is transcript-exact.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use anyhow::{bail, ensure, Result};
use hashbrown::HashMap;
use itertools::Itertools;
use plonky2::batch_fri::{batch_alpha_runs_target, AlphaRun};
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, RichField};
use plonky2::hash::hashing::PlonkyPermutation;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::iop::challenger::RecursiveChallenger;
use plonky2::iop::ext_target::{flatten_target, ExtensionTarget};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::WitnessWrite;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};
use plonky2::util::reducing::ReducingFactorTarget;
use plonky2::with_context;

use crate::batch_proof::{BatchStarkProofWithPublicInputs, BatchStarkProofWithPublicInputsTarget};
use crate::batch_recursive_verifier::{
    add_virtual_batch_stark_proof_with_pis, BatchKnownColumnsTarget,
    BatchStarkPreprocessedVerifierDataTarget,
};
use crate::batch_stark::{BatchOracle, BatchRole, BatchStark, BatchStarkLayout};
use crate::config::StarkConfig;
use crate::cross_table_lookup::{
    verify_cross_table_lookups_circuit, CrossTableLookup, CtlCheckVarsTarget,
};
use crate::get_challenges::get_dummy_polys_circuit;
use crate::lookup::get_grand_product_challenge_set_target;

/// The build-time envelope of a universal batch verifier.
#[derive(Debug, Clone)]
pub struct UniversalVerifierEnvelope {
    /// Per-table inclusive trace degree-bits range `[lo, hi]` (table-index
    /// order). Tables with preprocessed columns must be fixed (`lo == hi`).
    pub degree_ranges: Vec<(usize, usize)>,
    /// The consensus fold boundaries (degree bits), strictly descending.
    /// `ladder[0]` must equal the tallest table's `hi`; every reachable table
    /// degree must be a member; every proof folds down to `ladder.last()`.
    pub ladder: Vec<usize>,
    /// The tables sharing one multi-height Merkle tree per role (sorted;
    /// see [`BatchOracle`]). Grouped tables must be fixed-height, so the
    /// shared trees' shapes — and their exact Merkle walks — are
    /// profile-independent. Part of the transcript shape: must equal the
    /// list given to the prover.
    pub grouped_tables: Vec<usize>,
}

impl UniversalVerifierEnvelope {
    fn is_fixed(&self, t: usize) -> bool {
        self.degree_ranges[t].0 == self.degree_ranges[t].1
    }

    /// The degrees table `t` can take: `ladder ∩ [lo_t, hi_t]`, descending.
    fn candidates(&self, t: usize) -> Vec<usize> {
        let (lo, hi) = self.degree_ranges[t];
        self.ladder
            .iter()
            .copied()
            .filter(|&d| (lo..=hi).contains(&d))
            .collect()
    }
}

/// The targets a universal verifier exposes to its caller.
#[derive(Debug)]
pub struct UniversalBatchStarkVerifierTarget<const D: usize> {
    /// The proof, allocated at the envelope-maximum shape. Shorter commit
    /// phases and Merkle walks are padded at witness-setting time.
    pub proof_with_pis: BatchStarkProofWithPublicInputsTarget<D>,
    /// Per-table trace degree bits, `b_t = Σ_j d_j·flag_j`: derived from the
    /// one-hot flags for variable tables, constants for fixed ones.
    pub degree_bits: Vec<Target>,
    /// Per-table one-hot degree flags over `envelope.candidates(t)` (fixed
    /// tables hold a constant-true singleton). These are the verifier's
    /// degree *inputs*: callers must bind them (or `degree_bits`, from which
    /// they are derived) to the statement, e.g. as public inputs. Left
    /// unbound, the profile is a free witness and the proof attests only
    /// "*some* in-envelope profile verifies" — for proof-of-work that severs
    /// the claimed job size from the verified trace.
    pub degree_flags: Vec<Vec<BoolTarget>>,
    /// The FRI opening point, as in the monolithic verifier.
    pub zeta: ExtensionTarget<D>,
}

/// Which fixed degree, or which variable table, a claim's denominator and
/// injection boundary come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DegreeSource {
    /// A fixed trace degree: fixed-height tables' oracles, and the (fixed)
    /// degree classes of the shared preprocessed oracle.
    Fixed(usize),
    /// A variable table's trace degree, `degree_bits[t]` at runtime.
    Table(usize),
}

/// One merged combine term: the runs sharing `(source, kind)` add into one
/// numerator over one denominator `x_source − z_kind`.
#[derive(Debug)]
struct CombineTerm {
    source: DegreeSource,
    /// Opening-point kind: 0 = `zeta`, 1 = `g_source·zeta`, 2 = `1`.
    kind: usize,
    /// Indices into the canonical run list.
    run_indices: Vec<usize>,
}

/// Builds a universal verifier for the given batch family. Mirrors
/// [`verify_batch_stark_proof_circuit`][crate::batch_recursive_verifier::verify_batch_stark_proof_circuit],
/// with every degree-dependent computation lifted to circuit logic over the
/// per-table degree inputs. `config`'s fold schedule from `ladder[0]` must
/// walk exactly the envelope's ladder.
pub fn verify_universal_batch_stark_proof_circuit<F, C, const D: usize, const N: usize>(
    builder: &mut CircuitBuilder<F, D>,
    starks: &[&dyn BatchStark<F, D>; N],
    config: &StarkConfig,
    envelope: &UniversalVerifierEnvelope,
    cross_table_lookups: &[CrossTableLookup<F>],
    preprocessed: Option<&BatchStarkPreprocessedVerifierDataTarget>,
    known_columns: Option<&BatchKnownColumnsTarget<D>>,
    ctl_extra_looking_sums: &HashMap<usize, Vec<Target>>,
) -> Result<UniversalBatchStarkVerifierTarget<D>>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    C::Hasher: AlgebraicHasher<F>,
{
    let rate_bits = config.fri_config.rate_bits;
    let cap_height = config.fri_config.cap_height;
    let ladder = &envelope.ladder;
    ensure!(envelope.degree_ranges.len() == N);
    ensure!(
        ladder.windows(2).all(|w| w[0] > w[1]),
        "ladder must be strictly descending"
    );
    ensure!(
        config.preprocessed_columns.is_empty(),
        "In batch mode, preprocessed columns are described by \
         `BatchStarkPreprocessedVerifierDataTarget`, not by `StarkConfig::preprocessed_columns`."
    );
    for t in 0..N {
        let (lo, hi) = envelope.degree_ranges[t];
        ensure!(lo <= hi);
        let cands = envelope.candidates(t);
        ensure!(
            cands.first() == Some(&hi) && cands.last() == Some(&lo),
            "table {t}: envelope bounds [{lo}, {hi}] must lie on the ladder"
        );
    }
    let max_degree_bits: Vec<usize> = envelope.degree_ranges.iter().map(|&(_, hi)| hi).collect();
    ensure!(
        max_degree_bits.iter().max() == Some(&ladder[0]),
        "ladder[0] must be the tallest table's maximum degree"
    );

    // The fold schedule must walk exactly the ladder: layer `i` folds
    // `ladder[i] -> ladder[i+1]`.
    let fri_params = config.fri_params(ladder[0]);
    let arities = fri_params.reduction_arity_bits.clone();
    {
        let mut reached = vec![ladder[0]];
        let mut d = ladder[0];
        for &a in &arities {
            d -= a;
            reached.push(d);
        }
        ensure!(
            reached == *ladder,
            "config fold schedule {reached:?} must equal the envelope ladder {ladder:?}"
        );
    }
    let num_layers = arities.len();
    let lde_max = ladder[0] + rate_bits;

    // Allocate the proof at the envelope maximum; its layout supplies every
    // profile-independent count (columns, aux polys, quotients, oracle order).
    let empty_prep = vec![vec![]; N];
    let prep_columns = preprocessed.map_or(&empty_prep, |p| &p.columns_per_table);
    for (t, prep) in prep_columns.iter().enumerate() {
        ensure!(
            prep.is_empty() || envelope.is_fixed(t),
            "table {t} has preprocessed columns, so its height must be fixed"
        );
    }
    for &t in &envelope.grouped_tables {
        ensure!(
            envelope.is_fixed(t),
            "table {t} is grouped, so its height must be fixed"
        );
    }
    let layout = BatchStarkLayout::new(
        starks.as_slice(),
        config,
        &max_degree_bits,
        prep_columns,
        cross_table_lookups,
        &envelope.grouped_tables,
    )?;
    let proof_with_pis = add_virtual_batch_stark_proof_with_pis(
        builder,
        starks,
        config,
        &max_degree_bits,
        prep_columns,
        cross_table_lookups,
        &envelope.grouped_tables,
    )?;
    let proof = &proof_with_pis.proof;
    let public_inputs = &proof_with_pis.public_inputs;
    let has_prep = layout
        .oracles
        .iter()
        .any(|o| matches!(o, BatchOracle::Preprocessed));
    match (preprocessed, has_prep) {
        (Some(_), true) | (None, false) => {}
        (Some(_), false) => bail!("unexpected preprocessed data"),
        (None, true) => bail!("missing preprocessed cap"),
    }
    if let Some(known) = known_columns {
        known.validate(&layout.num_columns, prep_columns)?;
    }

    // ==========================================================================
    // Degree inputs: per-table one-hot flags over the candidate degrees.
    //   Σ_j flag_j = 1,   b_t = Σ_j d_j·flag_j.
    // ==========================================================================
    let one = builder.one();
    let zero = builder.zero();
    let candidates: Vec<Vec<usize>> = (0..N).map(|t| envelope.candidates(t)).collect();
    let mut flags: Vec<Vec<BoolTarget>> = Vec::with_capacity(N);
    let mut degree_bits_targets: Vec<Target> = Vec::with_capacity(N);
    for t in 0..N {
        if envelope.is_fixed(t) {
            flags.push(vec![BoolTarget::new_unsafe(one)]);
            degree_bits_targets
                .push(builder.constant(F::from_canonical_usize(envelope.degree_ranges[t].0)));
            continue;
        }
        let table_flags: Vec<BoolTarget> = candidates[t]
            .iter()
            .map(|_| builder.add_virtual_bool_target_safe())
            .collect();
        let sum = table_flags
            .iter()
            .fold(zero, |acc, &f| builder.add(acc, f.target));
        builder.connect(sum, one);
        let b_t = table_flags
            .iter()
            .zip(&candidates[t])
            .fold(zero, |acc, (&f, &d)| {
                builder.mul_const_add(F::from_canonical_usize(d), f.target, acc)
            });
        flags.push(table_flags);
        degree_bits_targets.push(b_t);
    }

    // `geq(t, d) = [b_t >= d]`, a partial sum of table `t`'s one-hot flags.
    let geq = |builder: &mut CircuitBuilder<F, D>,
               flags: &[BoolTarget],
               cands: &[usize],
               d: usize|
     -> Target {
        cands
            .iter()
            .zip(flags)
            .filter(|(&c, _)| c >= d)
            .fold(zero, |acc, (_, &f)| builder.add(acc, f.target))
    };

    // ==========================================================================
    // Fiat-Shamir replay (statement part) — as in the monolithic verifier,
    // with per-table degree targets instead of constants.
    // ==========================================================================
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

    let has_aux = (0..N).any(|t| layout.num_aux_polys(t) > 0);
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

    // Constraint-binding grind: identical to the monolithic verifier; the
    // usize degree parameter is only a bit-width bound, so the envelope
    // maximum stands in for every profile.
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
                max_degree_bits[t],
                degree_bits_targets[t],
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

    // ==========================================================================
    // Layer activity. Layer `i` folds `ladder[i] -> ladder[i+1]` and runs iff
    // the batch's tallest oracle reaches it:
    //   active_i = max_fixed >= ladder[i]  ∨  ∨_t (b_t >= ladder[i]).
    // Layers at or below the tallest fixed degree are compile-time active.
    // ==========================================================================
    let max_fixed_degree = (0..N)
        .filter(|&t| envelope.is_fixed(t))
        .map(|t| envelope.degree_ranges[t].0)
        .max();
    let num_conditional = (0..num_layers)
        .take_while(|&i| max_fixed_degree.is_none_or(|mf| ladder[i] > mf))
        .count();
    let layer_active: Vec<Option<BoolTarget>> = (0..num_layers)
        .map(|i| {
            if i >= num_conditional {
                return None; // compile-time active
            }
            let d = ladder[i];
            // 1 − Π_t (1 − geq(t, d)), over variable tables that can reach d.
            let mut not_any = one;
            for t in (0..N).filter(|&t| !envelope.is_fixed(t) && envelope.degree_ranges[t].1 >= d) {
                let g = geq(builder, &flags[t], &candidates[t], d);
                let not_g = builder.sub(one, g);
                not_any = builder.mul(not_any, not_g);
            }
            Some(BoolTarget::new_unsafe(builder.sub(one, not_any)))
        })
        .collect();

    // Job-max one-hot over the possible top boundaries `ladder[0..=num_conditional]`:
    //   jm_0 = active_0,   jm_s = active_s − active_{s−1},   jm_last = 1 − active_{last−1}.
    // (`active` is monotone along the ladder, so `jm` is one-hot by construction.)
    let jobmax_onehot: Vec<Target> = (0..=num_conditional)
        .map(|s| {
            let hi = if s < num_conditional {
                layer_active[s].unwrap().target
            } else {
                one
            };
            let lo = if s > 0 {
                layer_active[s - 1].unwrap().target
            } else {
                zero
            };
            builder.sub(hi, lo)
        })
        .collect();

    // ==========================================================================
    // FRI challenges, with a conditional Fiat-Shamir prefix: layer `i`'s
    // `observe(cap_i); β_i` reaches the sponge iff the layer is active.
    // ==========================================================================
    let fri_alpha = challenger.get_extension_challenge(builder);
    let mut fri_betas: Vec<ExtensionTarget<D>> = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        match layer_active[i] {
            None => {
                challenger.observe_cap(&proof.opening_proof.commit_phase_merkle_caps[i]);
                fri_betas.push(challenger.get_extension_challenge(builder));
            }
            Some(active) => {
                let s0 = challenger.compact(builder);
                challenger.observe_cap(&proof.opening_proof.commit_phase_merkle_caps[i]);
                fri_betas.push(challenger.get_extension_challenge(builder));
                let s1 = challenger.compact(builder);
                let selected = <C::Hasher as AlgebraicHasher<F>>::AlgebraicPermutation::new(
                    s0.as_ref()
                        .iter()
                        .zip(s1.as_ref())
                        .map(|(&a, &b)| builder.select(active, b, a)),
                );
                challenger = RecursiveChallenger::from_state(selected);
            }
        }
    }
    // An inactive layer's β is squeezed off-transcript, but it only ever
    // multiplies a zero accumulator: whenever the running value is nonzero,
    // every layer above was active and its β is transcript-bound.
    challenger.observe_extension_elements(&proof.opening_proof.final_poly.0);
    challenger.observe_element(proof.opening_proof.pow_witness);
    let fri_pow_response = challenger.get_challenge(builder);
    let query_challenges = challenger.get_n_challenges(builder, config.fri_config.num_query_rounds);
    with_context!(
        builder,
        "check FRI proof-of-work",
        builder.fri_verify_proof_of_work(fri_pow_response, &config.fri_config)
    );
    CircuitBuilder::<F, D>::assert_noncanonical_indices_ok(&config.fri_config);

    // ==========================================================================
    // Per-table geometry: `vanishing(ζ) = Z_H(ζ)·quotient(ζ)`, with
    //   Z_H(ζ) = ζ^{2^{b_t}} − 1
    // selected from a squaring chain by the degree flags.
    // ==========================================================================
    let one_ext = builder.one_extension();
    let zero_ext = builder.zero_extension();
    for t in 0..N {
        if starks[t].quotient_degree_factor() == 0 {
            continue;
        }
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
                max_degree_bits[t],
                degree_bits_targets[t],
                layout.num_lookup_columns[t],
            )
        );

        // ζ^{2^{b_t}} = Σ_j flag_j · ζ^{2^{d_j}}, from the chain ζ^{2^{lo}},
        // ζ^{2^{lo+1}}, ..., ζ^{2^{hi}}.
        let (lo, hi) = envelope.degree_ranges[t];
        let zeta_pow_deg = if envelope.is_fixed(t) {
            builder.exp_power_of_2_extension(zeta, lo)
        } else {
            let mut chain = Vec::with_capacity(hi - lo + 1);
            let mut cur = builder.exp_power_of_2_extension(zeta, lo);
            chain.push(cur);
            for _ in lo..hi {
                cur = builder.mul_extension(cur, cur);
                chain.push(cur);
            }
            candidates[t]
                .iter()
                .zip(&flags[t])
                .fold(zero_ext, |acc, (&d, &f)| {
                    let gated = builder.scalar_mul_ext(f.target, chain[d - lo]);
                    builder.add_extension(acc, gated)
                })
        };
        let z_h_zeta = builder.sub_extension(zeta_pow_deg, one_ext);
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

    // Known columns: bind the claimed openings to the caller's evaluations.
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

    // ==========================================================================
    // Universal batched FRI. Claims are enumerated as canonical alpha runs
    // (computed on the envelope-maximum layout; the run set and exponents are
    // profile-independent because per-table oracles always contribute whole
    // runs and preprocessed degrees are fixed) and merged per
    // `(degree source, point kind)`:
    //
    //   E_source = Σ_kind [Σ_runs α^{offset}·(Σ_j α^j ev_j − Σ_j α^j op_j)] / (x_source − z_kind)
    //
    // The fold injects `old ← β·old + Σ_{sources at d} E_source` at each
    // boundary `d`, exactly like the native verifier.
    // ==========================================================================
    let instances = layout.fri_instances_target(builder, zeta);
    let runs = batch_alpha_runs_target(&instances);
    let fri_openings = layout.fri_openings_target(zero, &proof.openings);

    // The degree source of a run. Multi-height oracles' runs never span
    // degree classes (a run is contained in one instance), so the class of
    // the first polynomial is the run's; their tables are fixed-height.
    let prep_flat = prep_flat_columns(&layout);
    let grouped_flat: [Vec<usize>; 3] = [
        layout.grouped_flat_tables(BatchRole::Trace),
        layout.grouped_flat_tables(BatchRole::Auxiliary),
        layout.grouped_flat_tables(BatchRole::Quotient),
    ];
    let source_of = |run: &AlphaRun| -> DegreeSource {
        match layout.oracles[run.oracle_index] {
            BatchOracle::Trace(t) | BatchOracle::Auxiliary(t) | BatchOracle::Quotient(t) => {
                if envelope.is_fixed(t) {
                    DegreeSource::Fixed(envelope.degree_ranges[t].0)
                } else {
                    DegreeSource::Table(t)
                }
            }
            BatchOracle::Preprocessed => {
                let t = prep_flat[run.poly_start].0;
                DegreeSource::Fixed(envelope.degree_ranges[t].0)
            }
            BatchOracle::GroupedTrace => {
                let t = grouped_flat[0][run.poly_start];
                DegreeSource::Fixed(envelope.degree_ranges[t].0)
            }
            BatchOracle::GroupedAuxiliary => {
                let t = grouped_flat[1][run.poly_start];
                DegreeSource::Fixed(envelope.degree_ranges[t].0)
            }
            BatchOracle::GroupedQuotient => {
                let t = grouped_flat[2][run.poly_start];
                DegreeSource::Fixed(envelope.degree_ranges[t].0)
            }
        }
    };
    let mut terms: Vec<CombineTerm> = Vec::new();
    for (r, run) in runs.iter().enumerate() {
        let source = source_of(run);
        let kind = run.batch;
        match terms
            .iter_mut()
            .find(|term| term.source == source && term.kind == kind)
        {
            Some(term) => term.run_indices.push(r),
            None => terms.push(CombineTerm {
                source,
                kind,
                run_indices: vec![r],
            }),
        }
    }

    // Query-independent precomputations.
    let alpha_pows: Vec<ExtensionTarget<D>> = runs
        .iter()
        .map(|run| builder.exp_u64_extension(fri_alpha, run.alpha_offset as u64))
        .collect();
    let reduced_openings: Vec<ExtensionTarget<D>> = runs
        .iter()
        .map(|run| {
            let values = &fri_openings[run.instance].batches[run.batch].values
                [run.flat_start..run.flat_start + run.len];
            ReducingFactorTarget::new(fri_alpha).reduce(values, builder)
        })
        .collect();

    // Kind-1 opening points, `g_source·ζ`: the subgroup generator is a
    // constant for fixed sources and a one-hot-selected constant for tables.
    let mut g_zeta: HashMap<DegreeSource, ExtensionTarget<D>> = HashMap::new();
    for term in &terms {
        if term.kind != 1 || g_zeta.contains_key(&term.source) {
            continue;
        }
        let g = match term.source {
            DegreeSource::Fixed(d) => builder.constant(F::primitive_root_of_unity(d)),
            DegreeSource::Table(t) => candidates[t]
                .iter()
                .zip(&flags[t])
                .fold(zero, |acc, (&d, &f)| {
                    builder.mul_const_add(F::primitive_root_of_unity(d), f.target, acc)
                }),
        };
        let g_ext = builder.convert_to_ext(g);
        g_zeta.insert(term.source, builder.mul_extension(g_ext, zeta));
    }

    // Injection boundaries. A variable table injects at `d` iff its flag for
    // `d` is set; fixed sources sit at their boundary unconditionally.
    let boundary_tables: Vec<Vec<(usize, BoolTarget)>> = ladder
        .iter()
        .map(|&d| {
            (0..N)
                .filter(|&t| !envelope.is_fixed(t))
                .filter_map(|t| {
                    let j = candidates[t].iter().position(|&c| c == d)?;
                    Some((t, flags[t][j]))
                })
                .collect()
        })
        .collect();
    let boundary_has_fixed: Vec<bool> = ladder
        .iter()
        .map(|&d| {
            terms
                .iter()
                .any(|term| term.source == DegreeSource::Fixed(d))
        })
        .collect();
    // The injection at `ladder[i]` scales the accumulator by β_{i−1} exactly
    // when something injects there (the native fold multiplies by β only at
    // injection boundaries), else passes it through unscaled.
    let injection_scales: Vec<ExtensionTarget<D>> = (1..ladder.len())
        .map(|i| {
            let beta = fri_betas[i - 1];
            if boundary_has_fixed[i] {
                return beta;
            }
            let tables = &boundary_tables[i];
            if tables.is_empty() {
                return one_ext;
            }
            let mut not_any = one;
            for &(_, flag) in tables {
                let not_f = builder.sub(one, flag.target);
                not_any = builder.mul(not_any, not_f);
            }
            let has_inj = BoolTarget::new_unsafe(builder.sub(one, not_any));
            builder.select_ext(has_inj, beta, one_ext)
        })
        .collect();

    // Initial-walk level activity per variable table: an oracle allocated at
    // depth `hi + rate` idles below its real start, so level `p` (leaf side
    // first) is active iff `b_t >= hi − p`. Only levels `p < hi − lo` can
    // idle; the conditional prefix stops there and deeper levels are
    // unconditionally active (no selection gates).
    let walk_active: Vec<Vec<BoolTarget>> = (0..N)
        .map(|t| {
            if envelope.is_fixed(t) {
                return vec![];
            }
            let (lo, hi) = envelope.degree_ranges[t];
            (0..hi - lo)
                .map(|p| BoolTarget::new_unsafe(geq(builder, &flags[t], &candidates[t], hi - p)))
                .collect()
        })
        .collect();

    // The multi-height oracles' leaf groups, `(lde height, group length)`.
    // Grouped tables are fixed-height, so these — like the preprocessed
    // groups — are build-time constants and their walks are exact.
    let prep_groups = prep_leaf_groups(&layout, rate_bits);
    let grouped_groups: [Vec<(usize, usize)>; 3] = [
        layout.grouped_leaf_groups(BatchRole::Trace, rate_bits),
        layout.grouped_leaf_groups(BatchRole::Auxiliary, rate_bits),
        layout.grouped_leaf_groups(BatchRole::Quotient, rate_bits),
    ];

    // Initial-tree caps, in oracle order.
    let caps = {
        let (mut trace_i, mut aux_i, mut quot_i) = (0, 0, 0);
        layout
            .oracles
            .iter()
            .map(|&oracle| match oracle {
                BatchOracle::Trace(_) | BatchOracle::GroupedTrace => {
                    trace_i += 1;
                    proof.trace_caps[trace_i - 1].clone()
                }
                BatchOracle::Preprocessed => {
                    preprocessed.expect("missing preprocessed data").cap.clone()
                }
                BatchOracle::Auxiliary(_) | BatchOracle::GroupedAuxiliary => {
                    aux_i += 1;
                    proof
                        .auxiliary_polys_caps
                        .as_ref()
                        .expect("missing auxiliary caps")[aux_i - 1]
                        .clone()
                }
                BatchOracle::Quotient(_) | BatchOracle::GroupedQuotient => {
                    quot_i += 1;
                    proof
                        .quotient_polys_caps
                        .as_ref()
                        .expect("missing quotient caps")[quot_i - 1]
                        .clone()
                }
            })
            .collect::<Vec<_>>()
    };

    // ==========================================================================
    // Query rounds.
    // ==========================================================================
    let num_queries = config.fri_config.num_query_rounds;
    for (q, &challenge) in query_challenges.iter().enumerate() {
        let level = if q == 1 {
            log::Level::Debug
        } else {
            log::Level::Trace
        };
        with_context!(
            builder,
            level,
            &format!("verify one (of {num_queries}) universal query rounds"),
            {
                // Top-aligned index bits: A = (challenge mod 2^{lde_job}) · 2^{shift},
                // shift = lde_max − lde_job, materialized as
                //   A[i] = Σ_s jm_s·bit_{i − shift_s}   (bit_{<0} = 0),
                // so every consumer reads fixed slots. Bits of the challenge
                // above lde_job are unused, which is exactly the native
                // `mod 2^{lde_job}` index reduction.
                let bits = builder.low_bits(challenge, lde_max, F::BITS);
                let a_bits: Vec<BoolTarget> = (0..lde_max)
                    .map(|i| {
                        let mut acc = zero;
                        for (s, &jm) in jobmax_onehot.iter().enumerate() {
                            let shift = ladder[0] - ladder[s];
                            if i >= shift {
                                acc = builder.mul_add(jm, bits[i - shift].target, acc);
                            }
                        }
                        BoolTarget::new_unsafe(acc)
                    })
                    .collect();

                // Every tree's cap sits at the top of the index; all oracles
                // and commit layers share the cap index.
                let cap_index = builder.le_sum(a_bits[lde_max - cap_height..].iter());

                // Initial Merkle walks: conditional for variable oracles,
                // exact for fixed ones, multi-height (and exact) for the
                // preprocessed and grouped oracles.
                let round = &proof.opening_proof.query_round_proofs[q];
                for (o, &oracle) in layout.oracles.iter().enumerate() {
                    let (evals, merkle_proof) = &round.initial_trees_proof.evals_proofs[o];
                    match oracle {
                        BatchOracle::Trace(t)
                        | BatchOracle::Auxiliary(t)
                        | BatchOracle::Quotient(t) => {
                            let depth = envelope.degree_ranges[t].1 + rate_bits;
                            let bits_window = &a_bits[lde_max - depth..lde_max - cap_height];
                            if envelope.is_fixed(t) {
                                builder.verify_merkle_proof_to_cap_with_cap_index::<C::Hasher>(
                                    evals.clone(),
                                    bits_window,
                                    cap_index,
                                    &caps[o],
                                    merkle_proof,
                                );
                            } else {
                                builder
                                    .conditional_verify_merkle_proof_to_cap_with_cap_index::<C::Hasher>(
                                        evals.clone(),
                                        bits_window,
                                        &walk_active[t],
                                        cap_index,
                                        &caps[o],
                                        merkle_proof,
                                    );
                            }
                        }
                        BatchOracle::Preprocessed
                        | BatchOracle::GroupedTrace
                        | BatchOracle::GroupedAuxiliary
                        | BatchOracle::GroupedQuotient => {
                            let groups: &[(usize, usize)] = match oracle {
                                BatchOracle::Preprocessed => &prep_groups,
                                BatchOracle::GroupedTrace => &grouped_groups[0],
                                BatchOracle::GroupedAuxiliary => &grouped_groups[1],
                                BatchOracle::GroupedQuotient => &grouped_groups[2],
                                _ => unreachable!(),
                            };
                            let mut leaf_groups = Vec::with_capacity(groups.len());
                            let mut heights = Vec::with_capacity(groups.len());
                            let mut start = 0;
                            for &(height, group_len) in groups {
                                leaf_groups.push(evals[start..start + group_len].to_vec());
                                heights.push(height);
                                start += group_len;
                            }
                            assert_eq!(start, evals.len());
                            builder.verify_batch_merkle_proof_to_cap_with_cap_index::<C::Hasher>(
                                &leaf_groups,
                                &heights,
                                &a_bits[lde_max - heights[0]..],
                                cap_index,
                                &caps[o],
                                merkle_proof,
                            );
                        }
                    }
                }

                // Boundary subgroup points, from fixed leading slots:
                //   x_d = shift·φ_m^{Σ_j A[lde_max−1−j]·2^j},  m = d + rate.
                let coset_shift = builder.constant(F::coset_shift());
                let boundary_x: Vec<Target> = ladder
                    .iter()
                    .map(|&d| {
                        let m = d + rate_bits;
                        let phi = F::primitive_root_of_unity(m);
                        let leading = (0..m).map(|j| a_bits[lde_max - 1 - j]);
                        let phi = builder.exp_from_bits_const_base(phi, leading);
                        builder.mul(coset_shift, phi)
                    })
                    .collect();
                // The fold starts at the job's top boundary.
                let mut subgroup_x = jobmax_onehot
                    .iter()
                    .enumerate()
                    .fold(zero, |acc, (s, &jm)| {
                        builder.mul_add(jm, boundary_x[s], acc)
                    });

                // Merged combine numerators:
                //   num(term) = Σ_runs α^{offset}·(Σ_j α^j ev_j − Σ_j α^j op_j).
                let term_numerators: Vec<ExtensionTarget<D>> = terms
                    .iter()
                    .map(|term| {
                        let mut acc = zero_ext;
                        for &r in &term.run_indices {
                            let run = &runs[r];
                            let evals: Vec<Target> = (run.poly_start..run.poly_start + run.len)
                                .map(|p| {
                                    round.initial_trees_proof.evals_proofs[run.oracle_index].0[p]
                                })
                                .collect();
                            let reduced_evals =
                                ReducingFactorTarget::new(fri_alpha).reduce_base(&evals, builder);
                            let diff = builder.sub_extension(reduced_evals, reduced_openings[r]);
                            acc = builder.mul_add_extension(alpha_pows[r], diff, acc);
                        }
                        acc
                    })
                    .collect();

                // Per-source boundary point: a ladder slot for fixed sources,
                // one-hot-selected for variable tables.
                let mut source_x: HashMap<DegreeSource, ExtensionTarget<D>> = HashMap::new();
                for term in &terms {
                    if source_x.contains_key(&term.source) {
                        continue;
                    }
                    let x = match term.source {
                        DegreeSource::Fixed(d) => {
                            boundary_x[ladder.iter().position(|&b| b == d).unwrap()]
                        }
                        DegreeSource::Table(t) => {
                            candidates[t]
                                .iter()
                                .zip(&flags[t])
                                .fold(zero, |acc, (&d, &f)| {
                                    let i = ladder.iter().position(|&b| b == d).unwrap();
                                    builder.mul_add(f.target, boundary_x[i], acc)
                                })
                        }
                    };
                    let x_ext = builder.convert_to_ext(x);
                    source_x.insert(term.source, x_ext);
                }

                // Per-source combine value: Σ_kind num/(x_source − z_kind).
                let mut source_sums: HashMap<DegreeSource, ExtensionTarget<D>> = HashMap::new();
                for (term, &num) in terms.iter().zip(&term_numerators) {
                    let point = match term.kind {
                        0 => zeta,
                        1 => g_zeta[&term.source],
                        2 => one_ext,
                        _ => unreachable!("opening points are zeta, g·zeta and 1"),
                    };
                    let den = builder.sub_extension(source_x[&term.source], point);
                    let entry = source_sums.entry(term.source).or_insert(zero_ext);
                    *entry = builder.div_add_extension(num, den, *entry);
                }

                // The fold: inject at each boundary, then fold the layer below.
                //   inject:  old ← scale·old + E_d
                //   fold:    evals[within] ≟ old (if active);  old ← fold_β(evals)
                let mut old_eval = zero_ext;
                for i in 0..ladder.len() {
                    let mut injected = zero_ext;
                    if boundary_has_fixed[i] {
                        let sum = source_sums[&DegreeSource::Fixed(ladder[i])];
                        injected = builder.add_extension(injected, sum);
                    }
                    for &(t, flag) in &boundary_tables[i] {
                        let sum = source_sums[&DegreeSource::Table(t)];
                        let gated = builder.scalar_mul_ext(flag.target, sum);
                        injected = builder.add_extension(injected, gated);
                    }
                    if i == 0 {
                        old_eval = injected;
                    } else {
                        old_eval =
                            builder.mul_add_extension(old_eval, injection_scales[i - 1], injected);
                    }

                    if i == num_layers {
                        break;
                    }
                    let arity_bits = arities[i];
                    let m = ladder[i] + rate_bits;
                    let within_bits = &a_bits[lde_max - m..lde_max - m + arity_bits];
                    let coset_bits = &a_bits[lde_max - m + arity_bits..lde_max - cap_height];
                    let step = &round.steps[i];
                    let evals = &step.evals;

                    // Consistency of the committed row with the running value.
                    let within_index = builder.le_sum(within_bits.iter());
                    let new_eval = builder.random_access_extension(within_index, evals.clone());
                    match layer_active[i] {
                        None => builder.connect_extension(new_eval, old_eval),
                        Some(active) => {
                            for (&n_limb, &o_limb) in new_eval.0.iter().zip(&old_eval.0) {
                                builder.conditional_assert_eq(active.target, n_limb, o_limb);
                            }
                        }
                    }

                    let folded = builder.compute_evaluation(
                        subgroup_x,
                        within_bits,
                        arity_bits,
                        evals,
                        fri_betas[i],
                    );
                    builder.verify_merkle_proof_to_cap_with_cap_index::<C::Hasher>(
                        flatten_target(evals),
                        coset_bits,
                        cap_index,
                        &proof.opening_proof.commit_phase_merkle_caps[i],
                        &step.merkle_proof,
                    );
                    let x_next = builder.exp_power_of_2(subgroup_x, arity_bits);
                    match layer_active[i] {
                        None => {
                            old_eval = folded;
                            subgroup_x = x_next;
                        }
                        Some(active) => {
                            old_eval = builder.select_ext(active, folded, old_eval);
                            subgroup_x = builder.select(active, x_next, subgroup_x);
                        }
                    }
                }

                // Final polynomial check (every fold bottoms out at the
                // ladder's last boundary, so the length is fixed).
                let eval = proof
                    .opening_proof
                    .final_poly
                    .eval_scalar(builder, subgroup_x);
                builder.connect_extension(eval, old_eval);
            }
        );
    }

    Ok(UniversalBatchStarkVerifierTarget {
        proof_with_pis,
        degree_bits: degree_bits_targets,
        degree_flags: flags,
        zeta,
    })
}

/// The preprocessed oracle's polynomials as `(table, column)` pairs, in leaf
/// order: descending trace degree, ties by table index, columns sorted
/// (see [`BatchOracle::Preprocessed`]).
fn prep_flat_columns(layout: &BatchStarkLayout) -> Vec<(usize, usize)> {
    let mut tables: Vec<usize> = (0..layout.num_tables())
        .filter(|&t| !layout.prep_columns[t].is_empty())
        .collect();
    tables.sort_by_key(|&t| (core::cmp::Reverse(layout.degree_bits[t]), t));
    tables
        .into_iter()
        .flat_map(|t| layout.prep_columns[t].iter().map(move |&c| (t, c)))
        .collect()
}

/// The preprocessed oracle's leaf groups as `(lde height, group length)`, in
/// tree order (descending height).
fn prep_leaf_groups(layout: &BatchStarkLayout, rate_bits: usize) -> Vec<(usize, usize)> {
    let mut tables: Vec<usize> = (0..layout.num_tables())
        .filter(|&t| !layout.prep_columns[t].is_empty())
        .collect();
    tables.sort_by_key(|&t| (core::cmp::Reverse(layout.degree_bits[t]), t));
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for t in tables {
        let height = layout.degree_bits[t] + rate_bits;
        match groups.last_mut() {
            Some((h, len)) if *h == height => *len += layout.prep_columns[t].len(),
            _ => groups.push((height, layout.prep_columns[t].len())),
        }
    }
    groups
}

/// Sets a universal verifier's targets to a runtime batch proof, padding the
/// envelope-maximum shapes:
///
/// - the degree flags are set one-hot at each table's actual degree;
/// - a variable-depth initial Merkle walk idles below its tree's real height,
///   so its unused leading sibling slots get zeros;
/// - commit-phase layers above the batch's tallest oracle get all-zeros
///   trees: per layer, the constant digest chain `d_0 = H(0…0)`,
///   `d_{l+1} = H(d_l ‖ d_l)` serves as the siblings at any index, and
///   `d_last` replicated as the cap.
pub fn set_universal_batch_stark_proof_with_pis_target<F, C, W, const D: usize>(
    witness: &mut W,
    target: &UniversalBatchStarkVerifierTarget<D>,
    proof_with_pis: &BatchStarkProofWithPublicInputs<F, C, D>,
    envelope: &UniversalVerifierEnvelope,
    config: &StarkConfig,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    C::Hasher: AlgebraicHasher<F>,
    W: WitnessWrite<F>,
{
    let target_proof = &target.proof_with_pis.proof;
    let proof = &proof_with_pis.proof;
    let n = envelope.degree_ranges.len();
    ensure!(proof.degree_bits.len() == n);
    ensure!(target.degree_flags.len() == n);

    ensure!(target.proof_with_pis.public_inputs.len() == proof_with_pis.public_inputs.len());
    for (pi_targets, pis) in target
        .proof_with_pis
        .public_inputs
        .iter()
        .zip(&proof_with_pis.public_inputs)
    {
        for (&pi_t, &pi) in pi_targets.iter().zip_eq(pis) {
            witness.set_target(pi_t, pi)?;
        }
    }

    // Degree flags: one-hot at the actual degree.
    for t in 0..n {
        let actual = proof.degree_bits[t];
        if envelope.is_fixed(t) {
            ensure!(
                actual == envelope.degree_ranges[t].0,
                "table {t}: fixed height mismatch"
            );
            continue;
        }
        let cands = envelope.candidates(t);
        ensure!(
            cands.contains(&actual),
            "table {t}: degree {actual} outside the envelope"
        );
        for (&d, &flag) in cands.iter().zip_eq(&target.degree_flags[t]) {
            witness.set_bool_target(flag, d == actual)?;
        }
    }

    // Caps and openings: profile-independent shapes, set directly.
    ensure!(target_proof.trace_caps.len() == proof.trace_caps.len());
    for (cap_target, cap) in target_proof.trace_caps.iter().zip(&proof.trace_caps) {
        witness.set_cap_target(cap_target, cap)?;
    }
    match (
        &target_proof.auxiliary_polys_caps,
        &proof.auxiliary_polys_caps,
    ) {
        (Some(cap_targets), Some(caps)) => {
            for (cap_target, cap) in cap_targets.iter().zip_eq(caps) {
                witness.set_cap_target(cap_target, cap)?;
            }
        }
        (None, None) => {}
        _ => bail!("auxiliary caps presence mismatch"),
    }
    match (
        &target_proof.quotient_polys_caps,
        &proof.quotient_polys_caps,
    ) {
        (Some(cap_targets), Some(caps)) => {
            for (cap_target, cap) in cap_targets.iter().zip_eq(caps) {
                witness.set_cap_target(cap_target, cap)?;
            }
        }
        (None, None) => {}
        _ => bail!("quotient caps presence mismatch"),
    }

    ensure!(target_proof.openings.len() == proof.openings.len());
    for (ot, os) in target_proof.openings.iter().zip(&proof.openings) {
        for (&t, &v) in ot.local_values.iter().zip_eq(&os.local_values) {
            witness.set_extension_target(t, v)?;
        }
        for (&t, &v) in ot.next_values.iter().zip_eq(&os.next_values) {
            witness.set_extension_target(t, v)?;
        }
        ensure!(ot.auxiliary_polys.is_some() == os.auxiliary_polys.is_some());
        for (&t, &v) in ot
            .auxiliary_polys
            .iter()
            .flatten()
            .zip_eq(os.auxiliary_polys.iter().flatten())
        {
            witness.set_extension_target(t, v)?;
        }
        ensure!(ot.auxiliary_polys_next.is_some() == os.auxiliary_polys_next.is_some());
        for (&t, &v) in ot
            .auxiliary_polys_next
            .iter()
            .flatten()
            .zip_eq(os.auxiliary_polys_next.iter().flatten())
        {
            witness.set_extension_target(t, v)?;
        }
        ensure!(ot.ctl_zs_first.is_some() == os.ctl_zs_first.is_some());
        for (&t, &v) in ot
            .ctl_zs_first
            .iter()
            .flatten()
            .zip_eq(os.ctl_zs_first.iter().flatten())
        {
            witness.set_target(t, v)?;
        }
        ensure!(ot.quotient_polys.is_some() == os.quotient_polys.is_some());
        for (&t, &v) in ot
            .quotient_polys
            .iter()
            .flatten()
            .zip_eq(os.quotient_polys.iter().flatten())
        {
            witness.set_extension_target(t, v)?;
        }
    }

    // The FRI proof, padded to the envelope's fold schedule.
    let ladder = &envelope.ladder;
    let arities = config.fri_params(ladder[0]).reduction_arity_bits;
    let rate_bits = config.fri_config.rate_bits;
    let cap_height = config.fri_config.cap_height;
    let rt = &proof.opening_proof;
    let tt = &target_proof.opening_proof;
    ensure!(rt.commit_phase_merkle_caps.len() <= arities.len());
    let pad = arities.len() - rt.commit_phase_merkle_caps.len();
    let job_max = proof.degree_bits.iter().copied().max().unwrap();
    ensure!(
        ladder[pad] == job_max,
        "the proof's commit phase must start at the batch's tallest oracle"
    );

    // Zero-tree digest chains of the skipped layers. Layer `i`'s tree commits
    // the folded codeword of size `2^{ladder[i+1] + rate}` in leaves of
    // `2^{arity_i}` extension values.
    let dummy_chains: Vec<Vec<HashOut<F>>> = (0..pad)
        .map(|i| {
            zero_tree_digests::<F, C::Hasher>(
                (1 << arities[i]) * D,
                ladder[i + 1] + rate_bits - cap_height,
            )
        })
        .collect();
    for (i, chain) in dummy_chains.iter().enumerate() {
        let cap: MerkleCap<F, C::Hasher> = MerkleCap(vec![*chain.last().unwrap(); 1 << cap_height]);
        witness.set_cap_target(&tt.commit_phase_merkle_caps[i], &cap)?;
    }
    for (cap_target, cap) in tt.commit_phase_merkle_caps[pad..]
        .iter()
        .zip_eq(&rt.commit_phase_merkle_caps)
    {
        witness.set_cap_target(cap_target, cap)?;
    }

    // Pre-validate the lengths the padding cannot absorb, so a proof made with
    // an off-envelope config fails with context instead of a `zip_eq` panic.
    ensure!(
        rt.query_round_proofs.len() == tt.query_round_proofs.len(),
        "query round count differs from the envelope's config"
    );
    ensure!(
        rt.final_poly.coeffs.len() == tt.final_poly.0.len(),
        "final polynomial length differs from the envelope's config"
    );

    for (q_t, q) in tt.query_round_proofs.iter().zip_eq(&rt.query_round_proofs) {
        ensure!(
            q.steps.len() == rt.commit_phase_merkle_caps.len(),
            "commit-phase step count differs from the proof's cap count"
        );
        ensure!(
            q.initial_trees_proof.evals_proofs.len() == q_t.initial_trees_proof.evals_proofs.len(),
            "initial-tree oracle count differs from the envelope's layout"
        );
        // Initial trees: equal leaf counts; a deeper target walk idles below
        // the real tree, its leading sibling slots unused.
        for ((leaves_t, proof_t), (leaves, proof)) in q_t
            .initial_trees_proof
            .evals_proofs
            .iter()
            .zip_eq(&q.initial_trees_proof.evals_proofs)
        {
            ensure!(
                leaves_t.len() == leaves.len(),
                "initial-tree leaf count differs from the envelope's layout"
            );
            for (&l_t, &l) in leaves_t.iter().zip_eq(leaves) {
                witness.set_target(l_t, l)?;
            }
            ensure!(proof_t.siblings.len() >= proof.siblings.len());
            let pad_o = proof_t.siblings.len() - proof.siblings.len();
            for &s_t in &proof_t.siblings[..pad_o] {
                witness.set_hash_target(s_t, HashOut::ZERO)?;
            }
            for (&s_t, &s) in proof_t.siblings[pad_o..].iter().zip_eq(&proof.siblings) {
                witness.set_hash_target(s_t, s)?;
            }
        }

        // Commit-phase steps: skipped layers get zero rows and the constant
        // sibling chain; real layers shift down by `pad`.
        for (i, chain) in dummy_chains.iter().enumerate() {
            let step_t = &q_t.steps[i];
            for &e_t in &step_t.evals {
                witness.set_extension_target(e_t, F::Extension::ZERO)?;
            }
            for (&s_t, &s) in step_t
                .merkle_proof
                .siblings
                .iter()
                .zip_eq(&chain[..chain.len() - 1])
            {
                witness.set_hash_target(s_t, s)?;
            }
        }
        for (step_t, step) in q_t.steps[pad..].iter().zip_eq(&q.steps) {
            ensure!(
                step_t.evals.len() == step.evals.len()
                    && step_t.merkle_proof.siblings.len() == step.merkle_proof.siblings.len(),
                "commit-phase step shape differs from the envelope's fold schedule"
            );
            for (&e_t, &e) in step_t.evals.iter().zip_eq(&step.evals) {
                witness.set_extension_target(e_t, e)?;
            }
            for (&s_t, &s) in step_t
                .merkle_proof
                .siblings
                .iter()
                .zip_eq(&step.merkle_proof.siblings)
            {
                witness.set_hash_target(s_t, s)?;
            }
        }
    }

    witness.set_target(tt.pow_witness, rt.pow_witness)?;
    for (&c_t, &c) in tt.final_poly.0.iter().zip_eq(&rt.final_poly.coeffs) {
        witness.set_extension_target(c_t, c)?;
    }
    Ok(())
}

/// The digest chain of an all-zeros Merkle tree with `levels` proof levels:
/// `d_0 = H(0…0)` over one leaf, then `d_{l+1} = H(d_l ‖ d_l)`. Since every
/// node at level `l` equals `d_l`, `&chain[..levels]` is a valid sibling list
/// at any index and `chain[levels]` is every cap entry.
fn zero_tree_digests<F: RichField, H: AlgebraicHasher<F>>(
    leaf_len: usize,
    levels: usize,
) -> Vec<HashOut<F>> {
    let mut digest = H::hash_or_noop(&vec![F::ZERO; leaf_len]);
    let mut chain = Vec::with_capacity(levels + 1);
    chain.push(digest);
    for _ in 0..levels {
        digest = H::two_to_one(digest, digest);
        chain.push(digest);
    }
    chain
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use hashbrown::HashMap;
    use plonky2::field::polynomial::PolynomialValues;
    use plonky2::field::types::Field;
    use plonky2::fri::reduction_strategies::FriReductionStrategy;
    use plonky2::fri::FriConfig;
    use plonky2::hash::poseidon::PoseidonHash;
    use plonky2::iop::ext_target::ExtensionTarget;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, Hasher, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use super::*;
    use crate::batch_prover::{batch_prove, BatchStarkPreprocessedData};
    use crate::batch_stark_testing::{
        BatchLookedStark, BatchLookingStark, SquareStark, NUM_LOOKED_SLOTS,
    };
    use crate::batch_verifier::{batch_verify, BatchKnownColumns};
    use crate::cross_table_lookup::TableWithColumns;
    use crate::lookup::{Column, Filter};

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    const LOOKING_ROWS: usize = 1 << 7;
    const LOOKED_ROWS: usize = 1 << 5;
    const BASE: u64 = 100;

    /// Arity-1 folds from any degree down to 4, so every job's fold schedule
    /// is a suffix of the envelope ladder's.
    fn test_config() -> StarkConfig {
        StarkConfig::new(
            40,
            2,
            FriConfig {
                rate_bits: 1,
                cap_height: 2,
                proof_of_work_bits: 8,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(1, 4),
                num_query_rounds: 40,
            },
        )
    }

    /// The CTL of [`crate::batch_stark_testing`]: filtered looking values
    /// match the union of the looked table's three slots.
    fn ctls() -> Vec<CrossTableLookup<F>> {
        vec![CrossTableLookup::new(
            vec![TableWithColumns::new(
                0,
                vec![Column::single(0)],
                Filter::from_column(Column::single(1)),
            )],
            [0, 2, 3]
                .map(|c| TableWithColumns::new(1, vec![Column::single(c)], Filter::default()))
                .to_vec(),
        )]
    }

    /// Evaluates a polynomial given by constant coefficients at an extension
    /// target, via Horner's rule.
    fn eval_constant_poly_circuit(
        builder: &mut CircuitBuilder<F, D>,
        coeffs: &[F],
        point: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let mut acc = builder.zero_extension();
        for &c in coeffs.iter().rev() {
            let c = builder.constant_extension(c.into());
            acc = builder.mul_add_extension(acc, point, c);
        }
        acc
    }

    /// The full batch feature set — a CTL, a shared preprocessed oracle and a
    /// known column on two fixed tables — plus one variable-height table,
    /// proven at four different height profiles (merging with either fixed
    /// table's degree group, above both, below both) and all verified by one
    /// universal circuit whose public inputs are the claimed heights. The two
    /// fixed tables are *grouped*: each role commits them in one multi-height
    /// tree (heights 7 and 5), exercising the grouped exact walks, the CTL
    /// batch order and the grouped alpha-run sourcing.
    #[test]
    fn test_universal_batch_stark_full_system() -> Result<()> {
        let config = test_config();
        let stark_a = BatchLookingStark::<F, D>::new();
        let stark_b = BatchLookedStark::<F, D>::new(true);
        let stark_c = SquareStark::<F, D>::new();
        let starks: [&dyn BatchStark<F, D>; 3] = [&stark_a, &stark_b, &stark_c];
        let cross_table_lookups = ctls();
        let envelope = UniversalVerifierEnvelope {
            degree_ranges: vec![(7, 7), (5, 5), (4, 8)],
            ladder: vec![8, 7, 6, 5, 4],
            grouped_tables: vec![0, 1],
        };

        let trace_a = BatchLookingStark::<F, D>::generate_trace(LOOKING_ROWS, LOOKED_ROWS, BASE);
        let trace_b = BatchLookedStark::<F, D>::generate_trace(LOOKED_ROWS, BASE);
        let public_inputs = [vec![F::from_canonical_u64(BASE)], vec![], vec![]];

        // The setup-time preprocessed commitment and the known column are
        // independent of the variable table's height: built once.
        let prep_columns = vec![vec![2], vec![0], vec![]];
        let preprocessed = BatchStarkPreprocessedData::<F, C, D>::new(
            vec![
                vec![BatchLookingStark::<F, D>::prep_column(
                    LOOKING_ROWS,
                    LOOKED_ROWS,
                    BASE,
                )],
                vec![BatchLookedStark::<F, D>::prep_column(LOOKED_ROWS, BASE)],
                vec![],
            ],
            prep_columns.clone(),
            &config,
            &mut TimingTree::default(),
        );
        let known_values = vec![
            vec![PolynomialValues::new(
                (0..LOOKING_ROWS)
                    .map(|i| F::from_bool(i < NUM_LOOKED_SLOTS * LOOKED_ROWS))
                    .collect(),
            )],
            vec![],
            vec![],
        ];
        let flat_known: Vec<F> = known_values
            .iter()
            .flatten()
            .flat_map(|v| v.values.iter().copied())
            .collect();
        let known_columns = BatchKnownColumns {
            digest: Some(PoseidonHash::hash_no_pad(&flat_known)),
            columns_per_table: vec![vec![1], vec![], vec![]],
            values_per_table: known_values,
        };

        // Build the universal verifier once.
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let known_target = BatchKnownColumnsTarget::<D> {
            digest: known_columns.digest.map(|d| builder.constant_hash(d)),
            columns_per_table: known_columns.columns_per_table.clone(),
            evals_at_zeta: vec![vec![builder.add_virtual_extension_target()], vec![], vec![]],
            evals_at_g_zeta: vec![vec![builder.add_virtual_extension_target()], vec![], vec![]],
        };
        let prep_target = preprocessed.verifier_data().constant_target(&mut builder);
        let universal = verify_universal_batch_stark_proof_circuit::<F, C, D, 3>(
            &mut builder,
            &starks,
            &config,
            &envelope,
            &cross_table_lookups,
            Some(&prep_target),
            Some(&known_target),
            &HashMap::new(),
        )?;
        // Bind the known-column openings to in-circuit evaluations at zeta
        // and g·zeta (the known column's table is fixed, so g is constant).
        let coeffs = known_columns.values_per_table[0][0].clone().ifft();
        let at_zeta = eval_constant_poly_circuit(&mut builder, &coeffs.coeffs, universal.zeta);
        builder.connect_extension(at_zeta, known_target.evals_at_zeta[0][0]);
        let g = builder.constant_extension(F::primitive_root_of_unity(7).into());
        let g_zeta = builder.mul_extension(g, universal.zeta);
        let at_g_zeta = eval_constant_poly_circuit(&mut builder, &coeffs.coeffs, g_zeta);
        builder.connect_extension(at_g_zeta, known_target.evals_at_g_zeta[0][0]);
        // The claimed heights are part of the statement.
        builder.register_public_inputs(&universal.degree_bits);
        builder.print_gate_counts(0);
        let data = builder.build::<C>();

        let mut last_proof = None;
        for square_bits in [8usize, 7, 5, 4] {
            let trace_c = SquareStark::<F, D>::generate_trace(1 << square_bits);
            let proof = batch_prove::<F, C, D, 3>(
                &starks,
                &config,
                [trace_a.clone(), trace_b.clone(), trace_c],
                &public_inputs,
                &cross_table_lookups,
                &envelope.grouped_tables,
                Some(&preprocessed),
                Some(&known_columns),
                &mut TimingTree::default(),
            )?;
            batch_verify::<F, C, D, 3>(
                &starks,
                &config,
                &proof,
                &cross_table_lookups,
                &envelope.grouped_tables,
                Some(&preprocessed.verifier_data()),
                Some(&known_columns),
                &HashMap::new(),
            )?;

            let mut pw = PartialWitness::new();
            set_universal_batch_stark_proof_with_pis_target(
                &mut pw, &universal, &proof, &envelope, &config,
            )?;
            let rec_proof = data.prove(pw)?;
            assert_eq!(
                rec_proof.public_inputs,
                [7, 5, square_bits].map(F::from_canonical_usize).to_vec(),
            );
            data.verify(rec_proof)?;
            last_proof = Some(proof);
        }

        // A witness claiming a height other than the proof's is contradictory
        // (the one-hot degree flags are bound to the proof).
        let proof = last_proof.unwrap();
        assert_eq!(proof.proof.degree_bits[2], 4);
        let mut pw = PartialWitness::new();
        let six = envelope.candidates(2).iter().position(|&d| d == 6).unwrap();
        pw.set_bool_target(universal.degree_flags[2][six], true)?;
        assert!(set_universal_batch_stark_proof_with_pis_target(
            &mut pw, &universal, &proof, &envelope, &config,
        )
        .is_err());

        Ok(())
    }

    /// Two quotient-only tables with the fixed one at the ladder *bottom*:
    /// exercises deep conditional prefixes (up to four skipped commit
    /// layers), large top-aligned index shifts and padded Merkle walks.
    #[test]
    fn test_universal_batch_stark_deep_conditional() -> Result<()> {
        let config = test_config();
        let stark_a = SquareStark::<F, D>::new();
        let stark_b = SquareStark::<F, D>::new();
        let starks: [&dyn BatchStark<F, D>; 2] = [&stark_a, &stark_b];
        let envelope = UniversalVerifierEnvelope {
            degree_ranges: vec![(4, 4), (4, 8)],
            ladder: vec![8, 7, 6, 5, 4],
            grouped_tables: vec![],
        };

        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let universal = verify_universal_batch_stark_proof_circuit::<F, C, D, 2>(
            &mut builder,
            &starks,
            &config,
            &envelope,
            &[],
            None,
            None,
            &HashMap::new(),
        )?;
        builder.register_public_inputs(&universal.degree_bits);
        let data = builder.build::<C>();

        for var_bits in [8usize, 6, 4] {
            let proof = batch_prove::<F, C, D, 2>(
                &starks,
                &config,
                [
                    SquareStark::<F, D>::generate_trace(1 << 4),
                    SquareStark::<F, D>::generate_trace(1 << var_bits),
                ],
                &[vec![], vec![]],
                &[],
                &[],
                None,
                None,
                &mut TimingTree::default(),
            )?;
            batch_verify::<F, C, D, 2>(
                &starks,
                &config,
                &proof,
                &[],
                &[],
                None,
                None,
                &HashMap::new(),
            )?;

            let mut pw = PartialWitness::new();
            set_universal_batch_stark_proof_with_pis_target(
                &mut pw, &universal, &proof, &envelope, &config,
            )?;
            let rec_proof = data.prove(pw)?;
            assert_eq!(
                rec_proof.public_inputs,
                [4, var_bits].map(F::from_canonical_usize).to_vec(),
            );
            data.verify(rec_proof)?;
        }
        Ok(())
    }
}
