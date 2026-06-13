// Compatibility test: fingerprints the STARK circuit (AIR constraints and lookups)
// against a known hash to detect unintended changes to the proving pipeline.

#[cfg(test)]
mod test {
    use std::io::Write;
    use std::path::Path;

    use crate::api::proof::{PublicProofParams, ZKProof};
    use crate::api::{prove, verify};
    use crate::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
    use crate::circuit::pearl_stark::PearlStark;
    use crate::ffi::mine::try_mine_one;
    use crate::{
        api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern},
        ffi::plain_proof::PlainProof,
    };

    use plonky2_field::goldilocks_field::GoldilocksField;
    use rand_chacha::rand_core::SeedableRng;

    use crate::circuit::utils::evaluator::Evaluator;

    /// An `Evaluator` that builds symbolic string expressions for AIR constraints.
    ///
    /// Each value is a `usize` index into an internal expression arena. Operations
    /// produce new arena entries with the string representation of the expression.
    /// Constraints are collected as tagged strings.
    struct StringEvaluator {
        arena: Vec<String>,
        pub(crate) constraints: Vec<String>,
    }

    impl StringEvaluator {
        pub(crate) fn new(num_local: usize, num_next: usize, num_pis: usize) -> (Self, Vec<usize>, Vec<usize>, Vec<usize>) {
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

        pub(crate) fn all_constraints_string(&self) -> String {
            // Maybe some constrains order has been moved across the code, the important thing is that are eventually the same.
            let mut constraints = self.constraints.clone();
            constraints.sort();
            constraints.join("\n")
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

    /// Create a hash a full stark parameters + configuration set of the starky air
    fn starky_hash(block_header: IncompleteBlockHeader, plain_proof: &PlainProof) -> String {
        use starky::stark::Stark;

        let (private_params, public_params) = plain_proof.parse_proof(block_header).expect("Failed to parse plain proof");

        let mut hasher = blake3::Hasher::new();

        // --- public params ---
        hasher.update(&public_params.block_header.to_bytes());
        hasher.update(&public_params.mining_config.to_bytes());
        hasher.update(&public_params.hash_a);
        hasher.update(&public_params.hash_b);
        hasher.update(&public_params.m.to_le_bytes());
        hasher.update(&public_params.n.to_le_bytes());
        hasher.update(&public_params.t_rows.to_le_bytes());
        hasher.update(&public_params.t_cols.to_le_bytes());

        // --- private params ---
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

        let stark = PearlStark::<GoldilocksField, 2>::new_with_params(&public_params);

        // --- AIR constraints as symbolic strings ---
        let constraints_str = air_constraints_string(&stark);
        hasher.update(constraints_str.as_bytes());

        // --- lookups structure (full Debug repr *is expected* to capture column indices, coefficients, filters, etc.) ---
        let lookups = stark.lookups();
        for lookup in &lookups {
            hasher.update(format!("{:?}", lookup).as_bytes());
        }

        hasher.finalize().to_hex().to_string()
    }

    /// Symbolically evaluate the Pearl AIR constraints, returning all constraint
    /// polynomials as human-readable string expressions.
    fn air_constraints_string(pearl: &PearlStark<GoldilocksField, 2>) -> String {
        use crate::circuit::pearl_air::eval_constraints;
        use crate::circuit::pearl_layout::{pearl_columns, pearl_public};
        use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};

        let num_cols = pearl_columns::TOTAL;
        let num_pis = pearl_public::TOTAL;

        let (mut evaluator, local_ids, next_ids, pi_ids) = StringEvaluator::new(num_cols, num_cols, num_pis);

        type Frame = StarkFrame<usize, usize, { pearl_columns::TOTAL }, { pearl_public::TOTAL }>;
        let frame = Frame::from_values(&local_ids, &next_ids, &pi_ids);

        eval_constraints::<_, _, StringEvaluator>(&pearl.chips, &frame, &mut evaluator);

        evaluator.all_constraints_string()
    }

    struct Params {
        block_header: IncompleteBlockHeader,
        m: usize,
        n: usize,
        k: usize,
        mining_config: MiningConfiguration,
    }

    fn params() -> Params {
        let rank = 64;
        let k = 16 * rank as usize + 192;
        Params {
            block_header: IncompleteBlockHeader::new_for_test(0x1D2FFFFF), // nontrivial difficulty for testing
            m: 6144,
            n: 4096,
            k,
            mining_config: MiningConfiguration {
                common_dim: k as u32,
                rank,
                mma_type: MMAType::Int7xInt7ToInt32,
                rows_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73]).unwrap(),
                cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 64, 65, 72, 73]).unwrap(),
                moe: None,
            },
        }
    }

    fn generate_plain_proof(p: &Params) -> PlainProof {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xdeadbeef);
        try_mine_one(&mut rng, p.m, p.n, p.k, p.block_header, p.mining_config, None, false)
            .unwrap()
            .unwrap()
    }

    fn starky_fingerprint() -> String {
        let p = params();
        let plain_proof = generate_plain_proof(&p);
        starky_hash(p.block_header, &plain_proof)
    }

    #[allow(unused)]
    fn generate_and_write_stark_proof(proof_path: &Path) {
        let mut cache = <PearlRecursion as RecursionCircuit>::CircuitCache::default();

        let p = params();
        let plain_proof = generate_plain_proof(&p);
        let result = prove::zk_prove_plain_proof(p.block_header, &plain_proof, &mut cache, false).expect("Proving failed");

        let mut f = std::fs::File::create(proof_path).unwrap();
        f.write_all(&result.public_data).unwrap();
        f.write_all(&result.proof_data).unwrap()
    }

    #[test]
    #[ignore] // Run with: cargo test -- --ignored test_generate_v2_fixture
    fn test_generate_v2_fixture() {
        let path = std::path::Path::new("fixures/v2_stark_proof.bin");
        generate_and_write_stark_proof(path);
        println!("V2 non-MoE fixture written to {:?}", path);
    }

    #[test]
    fn test_proof_fixture() {
        let params = params();

        let buffer = include_bytes!("../../fixures/v2_stark_proof.bin");
        let public_data = &buffer[..PublicProofParams::WIRE_SIZE];
        let proof_data = &buffer[PublicProofParams::WIRE_SIZE..];

        let (public_params, proof) = ZKProof::deserialize(params.block_header, public_data, proof_data).unwrap();

        let mut cache = <PearlRecursion as RecursionCircuit>::CircuitCache::default();
        verify::verify_block(&public_params, &proof, &mut cache).expect("Proof must verify");
    }

    #[test]
    fn test_starky_fingerprint() {
        assert_eq!(
            starky_fingerprint(),
            "39f5f852674850c07159fec9be3c1f3899fe4c02679a13a3d4e848137eea4b25",
            "starky_fingerprint mismatch check"
        );
    }

    // =========================================================================
    // MoE fixture test
    // =========================================================================

    fn moe_params() -> (IncompleteBlockHeader, MiningConfiguration, usize, usize, usize) {
        use crate::api::proof::MoEConfig;
        let rank = 32u16;
        let k = 1024usize;
        let header = IncompleteBlockHeader {
            version: 0,
            prev_block: [0; 32],
            merkle_root: *b"0123456789abcdef0123456789abcdef",
            timestamp: 0x66666666,
            nbits: 0x207FFFFF,
        };
        let config = MiningConfiguration {
            common_dim: k as u32,
            rank,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
            moe: Some(MoEConfig { e: 4, top_k: 1 }),
        };
        (header, config, 1024, 128, k) // m, n, k
    }

    /// Generate an MoE proof fixture. Format: public_data_len(4 LE) || public_data || proof_data.
    #[allow(unused)]
    fn generate_and_write_moe_proof(proof_path: &Path) {
        use crate::ffi::mine::try_mine_one_moe;

        let (header, config, m, n, k) = moe_params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xcafe_babe);
        let plain_proof = loop {
            let proof = try_mine_one_moe(&mut rng, m, n, k, header, config, None, false).unwrap();
            if let Some(p) = proof {
                break p;
            }
        };

        let mut cache = <PearlRecursion as RecursionCircuit>::CircuitCache::default();
        let result = prove::zk_prove_plain_proof(header, &plain_proof, &mut cache, false).expect("MoE proving failed");

        let mut f = std::fs::File::create(proof_path).unwrap();
        f.write_all(&(result.public_data.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&result.public_data).unwrap();
        f.write_all(&result.proof_data).unwrap();
    }

    #[test]
    #[ignore] // Run with: cargo test -- --ignored test_generate_moe_fixture
    fn test_generate_moe_fixture() {
        let path = std::path::Path::new("fixures/v2_stark_proof_moe.bin");
        generate_and_write_moe_proof(path);
        println!("MoE fixture written to {:?}", path);
    }

    #[test]
    fn test_moe_proof_fixture() {
        let (header, _config, _m, _n, _k) = moe_params();

        let buffer = include_bytes!("../../fixures/v2_stark_proof_moe.bin");
        let public_data_len = u32::from_le_bytes(buffer[..4].try_into().unwrap()) as usize;
        let public_data = &buffer[4..4 + public_data_len];
        let proof_data = &buffer[4 + public_data_len..];

        let (public_params, proof) = ZKProof::deserialize(header, public_data, proof_data).unwrap();
        assert!(public_params.moe.is_some(), "MoE fixture must have moe params");

        let mut cache = <PearlRecursion as RecursionCircuit>::CircuitCache::default();
        verify::verify_block(&public_params, &proof, &mut cache).expect("MoE proof must verify");
    }
}
