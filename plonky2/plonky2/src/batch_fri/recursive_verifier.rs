#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use itertools::Itertools;

use crate::batch_fri::{batch_alpha_runs_target, AlphaRun};
use crate::field::extension::Extendable;
use crate::fri::proof::{
    FriChallengesTarget, FriInitialTreeProofTarget, FriProofTarget, FriQueryRoundTarget,
};
use crate::fri::structure::{FriInstanceInfoTarget, FriOpeningsTarget};
use crate::fri::FriParams;
use crate::hash::hash_types::{MerkleCapTarget, RichField};
use crate::iop::ext_target::{flatten_target, ExtensionTarget};
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, GenericConfig};
use crate::util::reducing::ReducingFactorTarget;
use crate::with_context;

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    /// Circuit version of `verify_batch_fri_proof`.
    ///
    /// `degree_bits` contains the degree (in bits, without the rate) of each FRI instance,
    /// sorted in descending order with no duplicates.
    pub fn verify_batch_fri_proof<C: GenericConfig<D, F = F>>(
        &mut self,
        degree_bits: &[usize],
        instance: &[FriInstanceInfoTarget<D>],
        openings: &[FriOpeningsTarget<D>],
        challenges: &FriChallengesTarget<D>,
        initial_merkle_caps: &[MerkleCapTarget],
        proof: &FriProofTarget<D>,
        params: &FriParams,
    ) where
        C::Hasher: AlgebraicHasher<F>,
    {
        assert!(!params.hiding, "Batch FRI does not support hiding mode.");
        if let Some(max_arity_bits) = params.max_arity_bits() {
            self.check_recursion_config(max_arity_bits);
        }

        assert_eq!(
            params.final_poly_len(),
            proof.final_poly.len(),
            "Final polynomial has wrong degree."
        );

        with_context!(
            self,
            "check PoW",
            self.fri_verify_proof_of_work(challenges.fri_pow_response, &params.config)
        );

        // Check that parameters are coherent.
        assert_eq!(
            params.config.num_query_rounds,
            proof.query_round_proofs.len(),
            "Number of query rounds does not match config."
        );

        // Canonical alpha runs, with each run's claimed openings reduced by its
        // local alpha powers and its global alpha power, all runs-aligned.
        let runs = batch_alpha_runs_target(instance);
        let reduced_openings = with_context!(
            self,
            "precompute reduced evaluations",
            runs.iter()
                .map(|run| {
                    let values = &openings[run.instance].batches[run.batch].values
                        [run.flat_start..run.flat_start + run.len];
                    ReducingFactorTarget::new(challenges.fri_alpha).reduce(values, self)
                })
                .collect_vec()
        );
        let alpha_pows = runs
            .iter()
            .map(|run| self.exp_u64_extension(challenges.fri_alpha, run.alpha_offset as u64))
            .collect_vec();

        // From here on, work in the LDE domain: oracle heights are `degree + rate` bits.
        let lde_bits = degree_bits
            .iter()
            .map(|d| d + params.config.rate_bits)
            .collect_vec();

        for (i, round_proof) in proof.query_round_proofs.iter().enumerate() {
            // To minimize noise in our logs, we will only record a context for a single FRI query.
            // The very first query will have some extra gates due to constants being registered, so
            // the second query is a better representative.
            let level = if i == 1 {
                log::Level::Debug
            } else {
                log::Level::Trace
            };

            let num_queries = proof.query_round_proofs.len();
            with_context!(
                self,
                level,
                &format!("verify one (of {num_queries}) query rounds"),
                self.batch_fri_verifier_query_round::<C>(
                    &lde_bits,
                    instance,
                    challenges,
                    &runs,
                    &reduced_openings,
                    &alpha_pows,
                    initial_merkle_caps,
                    proof,
                    challenges.fri_query_indices[i],
                    round_proof,
                    params,
                )
            );
        }
    }

    fn batch_fri_verify_initial_proof<H: AlgebraicHasher<F>>(
        &mut self,
        lde_bits: &[usize],
        instances: &[FriInstanceInfoTarget<D>],
        x_index_bits: &[BoolTarget],
        proof: &FriInitialTreeProofTarget,
        initial_merkle_caps: &[MerkleCapTarget],
        cap_index: Target,
    ) {
        for (i, ((evals, merkle_proof), cap)) in proof
            .evals_proofs
            .iter()
            .zip(initial_merkle_caps)
            .enumerate()
        {
            // Reconstruct this oracle's leaf groups, skipping instances where the oracle holds
            // no polynomials: such heights are absent from the oracle's batch Merkle tree.
            let mut leaves = Vec::new();
            let mut heights = Vec::new();
            let mut leaf_index = 0;
            for (inst, &height) in instances.iter().zip(lde_bits) {
                let num_polys = inst.oracles[i].num_polys;
                if num_polys == 0 {
                    continue;
                }
                leaves.push(evals[leaf_index..leaf_index + num_polys].to_vec());
                heights.push(height);
                leaf_index += num_polys;
            }
            assert_eq!(
                leaf_index,
                evals.len(),
                "Initial evals length does not match the instances."
            );
            assert!(!leaves.is_empty(), "Oracle {i} holds no polynomials.");
            // Oracles whose tallest group is smaller than the tallest instance are opened at
            // the correspondingly shifted index; dropping the lowest bits performs the shift.
            let shift = lde_bits[0] - heights[0];

            with_context!(
                self,
                &format!("verify {i}'th initial Merkle proof"),
                self.verify_batch_merkle_proof_to_cap_with_cap_index::<H>(
                    &leaves,
                    &heights,
                    &x_index_bits[shift..],
                    cap_index,
                    cap,
                    merkle_proof
                )
            );
        }
    }

    fn batch_fri_combine_initial(
        &mut self,
        instance: &[FriInstanceInfoTarget<D>],
        index: usize,
        proof: &FriInitialTreeProofTarget,
        alpha: ExtensionTarget<D>,
        runs: &[AlphaRun],
        reduced_openings: &[ExtensionTarget<D>],
        alpha_pows: &[ExtensionTarget<D>],
        subgroup_x: Target,
        params: &FriParams,
    ) -> ExtensionTarget<D> {
        assert!(D > 1, "Not implemented for D=1.");
        // Every oracle's Merkle depth is its own tallest group's; the deepest oracle —
        // at whichever index it sits — must match the FRI params' degree.
        let max_siblings = proof
            .evals_proofs
            .iter()
            .map(|(_, merkle_proof)| merkle_proof.siblings.len())
            .max()
            .expect("batch FRI requires at least one oracle");
        assert_eq!(
            params.degree_bits,
            params.config.cap_height + max_siblings - params.config.rate_bits,
            "Proof shape does not match the FRI params' degree."
        );
        let subgroup_x = self.convert_to_ext(subgroup_x);
        let mut sum = self.zero_extension();

        // `Σ_batches Σ_runs alpha^offset(run) · (Σ_j alpha^j (ev_j − op_j)) / (x − z_batch)`,
        // per [`AlphaRun`]. The denominator is shared within a batch.
        for (k, batch) in instance[index].batches.iter().enumerate() {
            let mut batch_sum = self.zero_extension();
            for (r, run) in runs
                .iter()
                .enumerate()
                .filter(|(_, run)| run.instance == index && run.batch == k)
            {
                let evals = batch.polynomials[run.flat_start..run.flat_start + run.len]
                    .iter()
                    .map(|p| {
                        let poly_blinding = instance[index].oracles[p.oracle_index].blinding;
                        let salted = params.hiding && poly_blinding;
                        proof.unsalted_eval(p.oracle_index, p.polynomial_index, salted)
                    })
                    .collect_vec();
                let reduced_evals = ReducingFactorTarget::new(alpha).reduce_base(&evals, self);
                let numerator = self.sub_extension(reduced_evals, reduced_openings[r]);
                batch_sum = self.mul_add_extension(alpha_pows[r], numerator, batch_sum);
            }
            let denominator = self.sub_extension(subgroup_x, batch.point);
            sum = self.div_add_extension(batch_sum, denominator, sum);
        }

        sum
    }

    fn batch_fri_verifier_query_round<C: GenericConfig<D, F = F>>(
        &mut self,
        lde_bits: &[usize],
        instance: &[FriInstanceInfoTarget<D>],
        challenges: &FriChallengesTarget<D>,
        runs: &[AlphaRun],
        reduced_openings: &[ExtensionTarget<D>],
        alpha_pows: &[ExtensionTarget<D>],
        initial_merkle_caps: &[MerkleCapTarget],
        proof: &FriProofTarget<D>,
        x_index: Target,
        round_proof: &FriQueryRoundTarget<D>,
        params: &FriParams,
    ) where
        C::Hasher: AlgebraicHasher<F>,
    {
        let mut n = lde_bits[0];

        // Note that this `low_bits` decomposition permits non-canonical binary encodings. Here we
        // verify that this has a negligible impact on soundness error.
        Self::assert_noncanonical_indices_ok(&params.config);
        let mut x_index_bits = self.low_bits(x_index, n, F::BITS);

        let cap_index =
            self.le_sum(x_index_bits[x_index_bits.len() - params.config.cap_height..].iter());
        with_context!(
            self,
            "check FRI initial proof",
            self.batch_fri_verify_initial_proof::<C::Hasher>(
                lde_bits,
                instance,
                &x_index_bits,
                &round_proof.initial_trees_proof,
                initial_merkle_caps,
                cap_index
            )
        );

        // `subgroup_x` is `subgroup[x_index]`, i.e., the actual field element in the domain.
        let mut subgroup_x = with_context!(self, "compute x from its index", {
            let g = self.constant(F::coset_shift());
            let phi = F::primitive_root_of_unity(n);
            let phi = self.exp_from_bits_const_base(phi, x_index_bits.iter().rev());
            self.mul(g, phi)
        });

        let mut batch_index = 0;

        // old_eval is the last derived evaluation; it will be checked for consistency with its
        // committed "parent" value in the next iteration.
        let mut old_eval = with_context!(
            self,
            "combine initial oracles",
            self.batch_fri_combine_initial(
                instance,
                batch_index,
                &round_proof.initial_trees_proof,
                challenges.fri_alpha,
                runs,
                reduced_openings,
                alpha_pows,
                subgroup_x,
                params,
            )
        );
        batch_index += 1;

        for (i, &arity_bits) in params.reduction_arity_bits.iter().enumerate() {
            let evals = &round_proof.steps[i].evals;

            // Split x_index into the index of the coset x is in, and the index of x within that coset.
            let coset_index_bits = x_index_bits[arity_bits..].to_vec();
            let x_index_within_coset_bits = &x_index_bits[..arity_bits];
            let x_index_within_coset = self.le_sum(x_index_within_coset_bits.iter());

            // Check consistency with our old evaluation from the previous round.
            let new_eval = self.random_access_extension(x_index_within_coset, evals.clone());
            self.connect_extension(new_eval, old_eval);

            // Infer P(y) from {P(x)}_{x^arity=y}.
            old_eval = with_context!(
                self,
                "infer evaluation using interpolation",
                self.compute_evaluation(
                    subgroup_x,
                    x_index_within_coset_bits,
                    arity_bits,
                    evals,
                    challenges.fri_betas[i],
                )
            );

            with_context!(
                self,
                "verify FRI round Merkle proof.",
                self.verify_merkle_proof_to_cap_with_cap_index::<C::Hasher>(
                    flatten_target(evals),
                    &coset_index_bits,
                    cap_index,
                    &proof.commit_phase_merkle_caps[i],
                    &round_proof.steps[i].merkle_proof,
                )
            );

            // Update the point x to x^arity.
            subgroup_x = self.exp_power_of_2(subgroup_x, arity_bits);

            x_index_bits = coset_index_bits;
            n -= arity_bits;

            if batch_index < lde_bits.len() && n == lde_bits[batch_index] {
                let subgroup_x_init = with_context!(self, "compute init x from its index", {
                    let g = self.constant(F::coset_shift());
                    let phi = F::primitive_root_of_unity(n);
                    let phi = self.exp_from_bits_const_base(phi, x_index_bits.iter().rev());
                    self.mul(g, phi)
                });
                let eval = self.batch_fri_combine_initial(
                    instance,
                    batch_index,
                    &round_proof.initial_trees_proof,
                    challenges.fri_alpha,
                    runs,
                    reduced_openings,
                    alpha_pows,
                    subgroup_x_init,
                    params,
                );
                old_eval = self.mul_extension(old_eval, challenges.fri_betas[i]);
                old_eval = self.add_extension(old_eval, eval);
                batch_index += 1;
            }
        }
        assert_eq!(
            batch_index,
            instance.len(),
            "Wrong number of folded instances."
        );

        // Final check of FRI. After all the reductions, we check that the final polynomial is equal
        // to the one sent by the prover.
        let eval = with_context!(
            self,
            &format!(
                "evaluate final polynomial of length {}",
                proof.final_poly.len()
            ),
            proof.final_poly.eval_scalar(self, subgroup_x)
        );
        self.connect_extension(eval, old_eval);
    }

    /// Like `add_virtual_fri_proof`, but for batch FRI proofs where each initial oracle may
    /// have a different tallest degree.
    ///
    /// `num_leaves_per_oracle[o]` is the total number of polynomials (over all instances) held
    /// by oracle `o`, and `oracle_degree_bits[o]` is the degree (in bits, without the rate) of
    /// its tallest polynomial group.
    pub fn add_virtual_batch_fri_proof(
        &mut self,
        num_leaves_per_oracle: &[usize],
        oracle_degree_bits: &[usize],
        params: &FriParams,
    ) -> FriProofTarget<D> {
        assert_eq!(num_leaves_per_oracle.len(), oracle_degree_bits.len());
        let cap_height = params.config.cap_height;
        let num_queries = params.config.num_query_rounds;
        let commit_phase_merkle_caps = (0..params.reduction_arity_bits.len())
            .map(|_| self.add_virtual_cap(cap_height))
            .collect();
        let query_round_proofs = (0..num_queries)
            .map(|_| {
                self.add_virtual_batch_fri_query(num_leaves_per_oracle, oracle_degree_bits, params)
            })
            .collect();
        let final_poly = self.add_virtual_poly_coeff_ext(params.final_poly_len());
        let pow_witness = self.add_virtual_target();
        FriProofTarget {
            commit_phase_merkle_caps,
            query_round_proofs,
            final_poly,
            pow_witness,
        }
    }

    fn add_virtual_batch_fri_query(
        &mut self,
        num_leaves_per_oracle: &[usize],
        oracle_degree_bits: &[usize],
        params: &FriParams,
    ) -> FriQueryRoundTarget<D> {
        let cap_height = params.config.cap_height;
        assert!(params.lde_bits() >= cap_height);

        let evals_proofs = num_leaves_per_oracle
            .iter()
            .zip(oracle_degree_bits)
            .map(|(&num_oracle_leaves, &degree_bits)| {
                let leaves = self.add_virtual_targets(num_oracle_leaves);
                let oracle_lde_bits = degree_bits + params.config.rate_bits;
                assert!(oracle_lde_bits >= cap_height);
                let merkle_proof = self.add_virtual_merkle_proof(oracle_lde_bits - cap_height);
                (leaves, merkle_proof)
            })
            .collect();
        let initial_trees_proof = FriInitialTreeProofTarget { evals_proofs };

        let mut merkle_proof_len = params.lde_bits() - cap_height;
        let mut steps = Vec::with_capacity(params.reduction_arity_bits.len());
        for &arity_bits in &params.reduction_arity_bits {
            assert!(merkle_proof_len >= arity_bits);
            merkle_proof_len -= arity_bits;
            steps.push(self.add_virtual_fri_query_step(arity_bits, merkle_proof_len));
        }

        FriQueryRoundTarget {
            initial_trees_proof,
            steps,
        }
    }
}
