//! Compatibility tests for the V1 (master) verifier.
//! Verifies that the V1 STARK circuit fingerprint matches master and that
//! master-generated proof fixtures still verify.

#[cfg(test)]
mod test {
    use crate::v1::api::proof::{PublicProofParams, ZKProof};
    use crate::v1::api::verify;
    use crate::v1::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
    use crate::v1::circuit::pearl_stark::PearlStark;

    use crate::v1::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern};
    use crate::v1::circuit::pearl_layout::{pearl_columns, pearl_public};
    use crate::v1::circuit::utils::evaluator::Evaluator;

    use plonky2_field::goldilocks_field::GoldilocksField;
    use rand_chacha::rand_core::SeedableRng;
    use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
    use starky::stark::Stark;

    struct StringEvaluator {
        arena: Vec<String>,
        constraints: Vec<String>,
    }

    impl StringEvaluator {
        fn new(num_local: usize, num_next: usize, num_pis: usize) -> (Self, Vec<usize>, Vec<usize>, Vec<usize>) {
            let mut arena = Vec::with_capacity(num_local + num_next + num_pis);
            let local_ids: Vec<usize> = (0..num_local)
                .map(|i| {
                    let id = arena.len();
                    arena.push(format!("L{i}"));
                    id
                })
                .collect();
            let next_ids: Vec<usize> = (0..num_next)
                .map(|i| {
                    let id = arena.len();
                    arena.push(format!("N{i}"));
                    id
                })
                .collect();
            let pi_ids: Vec<usize> = (0..num_pis)
                .map(|i| {
                    let id = arena.len();
                    arena.push(format!("PI{i}"));
                    id
                })
                .collect();
            (
                Self {
                    arena,
                    constraints: Vec::new(),
                },
                local_ids,
                next_ids,
                pi_ids,
            )
        }
        fn push(&mut self, expr: String) -> usize {
            let id = self.arena.len();
            self.arena.push(expr);
            id
        }
        fn expr(&self, id: usize) -> &str {
            &self.arena[id]
        }
        fn all_constraints_string(&self) -> String {
            let mut c = self.constraints.clone();
            c.sort();
            c.join("\n")
        }
    }

    impl Evaluator<usize, usize> for StringEvaluator {
        fn add(&mut self, a: usize, b: usize) -> usize {
            let s = format!("({} + {})", self.expr(a), self.expr(b));
            self.push(s)
        }
        fn sub(&mut self, a: usize, b: usize) -> usize {
            let s = format!("({} - {})", self.expr(a), self.expr(b));
            self.push(s)
        }
        fn mul(&mut self, a: usize, b: usize) -> usize {
            let s = format!("({} * {})", self.expr(a), self.expr(b));
            self.push(s)
        }
        fn mad(&mut self, a: usize, b: usize, c: usize) -> usize {
            let s = format!("(({} * {}) + {})", self.expr(a), self.expr(b), self.expr(c));
            self.push(s)
        }
        fn msub(&mut self, a: usize, b: usize, c: usize) -> usize {
            let s = format!("(({} * {}) - {})", self.expr(a), self.expr(b), self.expr(c));
            self.push(s)
        }
        fn i32(&mut self, s: i32) -> usize {
            self.push(s.to_string())
        }
        fn u64(&mut self, s: u64) -> usize {
            self.push(s.to_string())
        }
        fn scalar(&mut self, s: usize) -> usize {
            s
        }
        fn constraint(&mut self, c: usize) {
            self.constraints.push(format!("ALL: {}", self.expr(c)));
        }
        fn constraint_transition(&mut self, c: usize) {
            self.constraints.push(format!("TRANSITION: {}", self.expr(c)));
        }
        fn constraint_first_row(&mut self, c: usize) {
            self.constraints.push(format!("FIRST: {}", self.expr(c)));
        }
        fn constraint_last_row(&mut self, c: usize) {
            self.constraints.push(format!("LAST: {}", self.expr(c)));
        }
    }

    fn v1_params() -> (IncompleteBlockHeader, MiningConfiguration, usize, usize, usize) {
        let rank = 64u16;
        let k = 16 * rank as usize + 192;
        let header = IncompleteBlockHeader::new_for_test(0x1D2FFFFF);
        let config = MiningConfiguration {
            common_dim: k as u32,
            rank,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73]).unwrap(),
            reserved: MiningConfiguration::RESERVED_VALUE,
        };
        (header, config, 6144, 4096, k)
    }

    fn v1_starky_fingerprint() -> String {
        let (header, config, m, n, k) = v1_params();

        // Mine using the main module (same algorithm for non-MoE)
        let main_config = crate::api::proof::MiningConfiguration {
            common_dim: config.common_dim,
            rank: config.rank,
            mma_type: crate::api::proof::MMAType::Int7xInt7ToInt32,
            rows_pattern: crate::api::proof::PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73]).unwrap(),
            cols_pattern: crate::api::proof::PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73]).unwrap(),
            moe: None,
        };
        let main_header = crate::api::proof::IncompleteBlockHeader::new_for_test(0x1D2FFFFF);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xdeadbeef);
        let plain_proof = crate::ffi::mine::try_mine_one(&mut rng, m, n, k, main_header, main_config, None, false)
            .unwrap()
            .unwrap();

        // Parse with main module to get public/private params for hashing
        let (private_params, public_params) = plain_proof.parse_proof(main_header).unwrap();

        let mut hasher = blake3::Hasher::new();
        hasher.update(&public_params.block_header.to_bytes());
        hasher.update(&public_params.mining_config.to_bytes());
        hasher.update(&public_params.hash_a());
        hasher.update(&public_params.hash_b());
        hasher.update(&public_params.m.to_le_bytes());
        hasher.update(&public_params.n.to_le_bytes());
        hasher.update(&public_params.t_rows.to_le_bytes());
        hasher.update(&public_params.t_cols.to_le_bytes());

        for row in &private_params.s_a {
            for &val in row {
                hasher.update(&[val as u8]);
            }
        }
        for row in &private_params.s_b {
            for &val in row {
                hasher.update(&[val as u8]);
            }
        }
        for msg in &private_params.external_msgs {
            hasher.update(msg);
        }
        for cv in &private_params.external_cvs {
            hasher.update(cv);
        }

        // Construct V1 PublicProofParams from wire bytes to build V1 STARK
        let wire_bytes = public_params.to_wire_bytes().unwrap();
        let v1_public_params = PublicProofParams::from_bytes(header, wire_bytes[..164].try_into().unwrap()).unwrap();

        let stark = PearlStark::<GoldilocksField, 2>::new_with_params(&v1_public_params);

        // AIR constraints string
        let constraints_str = {
            use crate::v1::circuit::pearl_air::eval_constraints;
            let num_cols = pearl_columns::TOTAL;
            let num_pis = pearl_public::TOTAL;
            let (mut evaluator, local_ids, next_ids, pi_ids) = StringEvaluator::new(num_cols, num_cols, num_pis);
            type Frame = StarkFrame<usize, usize, { pearl_columns::TOTAL }, { pearl_public::TOTAL }>;
            let frame = Frame::from_values(&local_ids, &next_ids, &pi_ids);
            eval_constraints::<_, _, StringEvaluator>(&stark.chips, &frame, &mut evaluator);
            evaluator.all_constraints_string()
        };
        hasher.update(constraints_str.as_bytes());

        let lookups = stark.lookups();
        for lookup in &lookups {
            hasher.update(format!("{:?}", lookup).as_bytes());
        }

        hasher.finalize().to_hex().to_string()
    }

    #[test]
    fn test_v1_starky_fingerprint() {
        assert_eq!(
            v1_starky_fingerprint(),
            "7be24c836fc8e11aee531814e722ba70b2701fb1fbf41a6bd0824db8bef38419",
            "V1 starky_fingerprint mismatch — must match master"
        );
    }

    #[test]
    fn test_v1_proof_fixture() {
        let (header, _config, _m, _n, _k) = v1_params();

        let buffer = include_bytes!("fixures/stark_proof.bin");
        let public_data: &[u8; PublicProofParams::PUBLICDATA_SIZE] =
            buffer[..PublicProofParams::PUBLICDATA_SIZE].try_into().unwrap();
        let proof_data = &buffer[PublicProofParams::PUBLICDATA_SIZE..];

        let (public_params, proof) = ZKProof::deserialize(header, public_data, proof_data).unwrap();

        let mut cache = <PearlRecursion as RecursionCircuit>::CircuitCache::default();
        verify::verify_block(&public_params, &proof, &mut cache).expect("V1 proof must verify");
    }
}
