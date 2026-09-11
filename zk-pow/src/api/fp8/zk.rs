//! FP8 proving and verification, connecting
//! [`crate::api::fp8::plain_proof::PlainProofV4`] and [`PublicParams`] to [`Fp8System`].
//!
//! - [`Fp8Prover`] holds setup; [`Fp8Prover::prove`] publishes `(public_data, proof_data)` bytes.
//! - [`Fp8Verifier`] verifies that pair against the caller's expected block header.
//! - [`Fp8VerifierCache`] selects trusted setup by device. A missing entry is an
//!   error; verification never compiles circuits on demand.
//!
//! [`Fp8Verifier::generate`] builds setup offline using [`crate::api::fp8::lut_caps`].
//! Deployments load [`embedded_cache::CACHE_DATA`](crate::api::fp8::embedded_cache::CACHE_DATA).
//! [`Fp8ProverData`] retains the LUT polynomials and Merkle tree;
//! [`Fp8VerifierData`] holds the cap for native [`verify_fp8_block`] calls.
//!
//! [`Fp8Job::derive`] reconstructs programs, known columns and expected public
//! inputs from the statement and header. In MoE, [`MoEStatement::routing_pins`]
//! constrains sampled routing entries; neighboring entries remain private and
//! are authenticated by the routing hash. Operand digests are bound by the
//! in-AIR commitment folds.
//!
//! [`crate::circuit::fp8::wrapper`] wraps the batch proof recursively and adds
//! zero knowledge in the outer stage. [`zk_prove_plain_proof_fp8_wrapped`] reuses
//! compiled circuits through [`Fp8CircuitCache`].
//!
//! The published encoding is `zeta (16 bytes) || compact outer proof`
//! ([`compact_proof_data`]). [`verify_compact_wrapped_proof`] reconstructs the
//! public inputs from the statement and the omitted constants/sigmas data from
//! trusted setup polynomials. [`verify_fp8_block_wrapped`] accepts a full
//! [`ProofWithPublicInputs`] for in-process callers.

use anyhow::{Context, Result, anyhow, ensure};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::plonk::circuit_data::VerifierCircuitData;
use plonky2::plonk::config::PoseidonGoldilocksConfig;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::util::timing::TimingTree;
use starky::batch_proof::BatchStarkProofWithPublicInputs;
use starky::batch_prover::BatchStarkPreprocessedData;

use crate::api::fp8::lut_caps::{LutCap, committed_lut_cap};
use crate::api::fp8::noise::compute_fp8_noise;
use crate::api::fp8::openings::stream_words;
use crate::api::fp8::plain_proof::PlainProofV4;
use crate::api::fp8::prequant::BLOCK_SIZE;
use crate::api::fp8::public_params::{
    CommonParams, Device, HashId, JackpotStatement, JobParams, MoEStatement, OperandParams, PublicParams, Quant,
};
use crate::api::fp8::utils::{B200, fp32_to_bf16_rne};
use crate::api::layout::{AxisPattern, DimType, lane_assignment};
use crate::api::primitives::{Hash256, IncompleteBlockHeader, Sides};
use crate::api::proof_utils::check_jackpot_difficulty;
use crate::circuit::fp8::blake3_stark::columns::{
    NUM_BLAKE3_PUBLIC_INPUTS, PI_HASH_A, PI_HASH_B, PI_HASH_JACKPOT, PI_HASH_OFFSETS, PI_HASH_ROUTING, PI_JACKPOT_KEY, PI_KEY_A,
    PI_KEY_B,
};
use crate::circuit::fp8::blake3_stark::stark::{Blake3Program, MoeSchedule};
pub use crate::circuit::fp8::circuit_utils::{Fp8Verifier, Fp8VerifierCache};
use crate::circuit::fp8::ctl::NUM_TABLES;
use crate::circuit::fp8::driver::{FP8_REACHABLE_DEGREE_BITS, Fp8PublicData, Fp8System, Fp8Witness};
use crate::circuit::fp8::input_quant_stark::stark::InputQuantProgram;
use crate::circuit::fp8::matmul_b200_stark::stark::MatmulProgram;
use crate::circuit::fp8::scale_stark::stark::ScaleProgram;
use crate::circuit::fp8::tamed_stark::stark::TamedProgram;
use crate::circuit::fp8::wrapper::{
    Fp8CircuitCache, Fp8WrapperCircuits, OuterC, compact_proof_data, verify_compact_wrapped_proof, verify_wrapped_proof,
};
use crate::circuit::fp8::xor_fold_stark::stark::XorFoldProgram;
use crate::v2::api::proof_utils::hash_to_u32_field_array;

/// The fp8 proof system's internal field, extension degree and hasher.
pub type F = GoldilocksField;
pub const D: usize = 2;
pub type C = PoseidonGoldilocksConfig;

impl MoEStatement {
    /// Return `(stream_word_index, expected_value)` for each sampled routing entry.
    /// `o_w_prev + inner_indices[i]` selects the slot that must equal `i_a[i]`.
    /// Word indices refer to concatenated [`MoEStatement::opened_routing_blocks`];
    /// other words are private, authenticated by `HR`.
    pub(crate) fn routing_pins(&self, inner_indices: &[u32]) -> Vec<(usize, u32)> {
        assert_eq!(inner_indices.len(), self.i_a.len(), "inner/outer index lists must align");
        let words_per_block = pearl_blake3::BLAKE3_MSG_LEN / std::mem::size_of::<u32>();
        let blocks = self.opened_routing_blocks();
        inner_indices
            .iter()
            .zip(&self.i_a)
            .map(|(&inner, &outer)| {
                let slot = self.o_w_prev as usize + inner as usize;
                let opened_block_index = blocks
                    .binary_search(&((slot / words_per_block) as u32))
                    .expect("sampled slot's block is opened by construction");
                (words_per_block * opened_block_index + slot % words_per_block, outer)
            })
            .collect()
    }
}

/// Prover data and compiled recursive circuits. Verifiers load separate
/// trusted setup from [`Fp8VerifierCache`], built by [`Fp8Verifier::generate`].
pub struct Fp8Prover {
    data: Fp8ProverData,
    circuits: Fp8CircuitCache,
}

impl Fp8Prover {
    /// Build the LUT precommitment and universal wrapper circuits.
    pub fn setup(proposed_header: &IncompleteBlockHeader, input: &PlainProofV4) -> Result<Self> {
        let mut timing = TimingTree::default();
        Self::setup_with_timing(proposed_header, input, &mut timing)
    }

    fn setup_with_timing(proposed_header: &IncompleteBlockHeader, input: &PlainProofV4, timing: &mut TimingTree) -> Result<Self> {
        let (_, params) = input
            .parse_proof(proposed_header)
            .context("parsing the fp8 input during setup")?;
        let data = Fp8ProverData::generate(&params, proposed_header, timing)?;
        let job = Fp8Job::derive(&params, proposed_header)?;
        let mut circuits = Fp8CircuitCache::default();
        // Compile once here so subsequent proofs reuse the circuits.
        Fp8WrapperCircuits::with_cache(&job.system, &data.lut_cap, &mut circuits, timing)?;
        Ok(Self { data, circuits })
    }

    /// Return `(public_data, proof_data)`: the statement and compact recursive
    /// proof ([`compact_proof_data`]). The unwrapped STARK proof is not zero
    /// knowledge and stays inside the prover.
    pub fn prove(&mut self, proposed_header: &IncompleteBlockHeader, input: &PlainProofV4) -> Result<(Vec<u8>, Vec<u8>)> {
        let mut timing = TimingTree::default();
        self.prove_with_timing(proposed_header, input, &mut timing)
    }

    fn prove_with_timing(
        &mut self,
        proposed_header: &IncompleteBlockHeader,
        input: &PlainProofV4,
        timing: &mut TimingTree,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let (statement, wrapped) =
            zk_prove_plain_proof_fp8_wrapped(proposed_header, input, &self.data, &mut self.circuits, timing)?;
        // Reconstruct the layout to locate zeta for the compact preamble.
        let job = Fp8Job::derive(&statement, proposed_header)?;
        let proof_data = compact_proof_data(&job.system, &wrapped)?;
        Ok((statement.to_bytes(), proof_data))
    }
}

impl Fp8Verifier {
    /// Compile the universal wrappers using the committed cap in
    /// [`crate::api::fp8::lut_caps`]. Run offline (`build_cache`) and distribute
    /// through [`Fp8VerifierCache`].
    pub fn generate(params: &PublicParams, proposed_header: &IncompleteBlockHeader, timing: &mut TimingTree) -> Result<Self> {
        let job = Fp8Job::derive(params, proposed_header)?;
        Self::build_for_job(&job, committed_lut_cap(), timing)
    }

    /// Like [`Fp8Verifier::generate`], using an explicit cap. `build_cache`
    /// passes the fresh cap because the running binary still embeds the old file.
    pub fn generate_with_lut_cap(
        params: &PublicParams,
        proposed_header: &IncompleteBlockHeader,
        lut_cap: LutCap,
        timing: &mut TimingTree,
    ) -> Result<Self> {
        let job = Fp8Job::derive(params, proposed_header)?;
        Self::build_for_job(&job, lut_cap, timing)
    }

    fn build_for_job(job: &Fp8Job, lut_cap: LutCap, timing: &mut TimingTree) -> Result<Self> {
        let circuits = Fp8WrapperCircuits::build(&job.system, &lut_cap, timing)?;
        Ok(Self {
            lut_cap,
            circuit: circuits.verifier_data(),
            constants_sigmas_polynomials: circuits.constants_sigmas_polynomials(),
        })
    }

    /// Verifies a published `(public_data, proof_data)` pair against the
    /// caller's expected block header: `Ok(())` accepts, `Err` rejects.
    pub fn verify_block(&self, proposed_header: &IncompleteBlockHeader, public_data: &[u8], proof_data: &[u8]) -> Result<()> {
        self.verify_with_nbits(proposed_header, public_data, proof_data, None)
    }

    /// Verifies a pool share under an explicit share target.
    pub fn verify_share(
        &self,
        proposed_header: &IncompleteBlockHeader,
        public_data: &[u8],
        proof_data: &[u8],
        share_nbits: u32,
    ) -> Result<()> {
        self.verify_with_nbits(proposed_header, public_data, proof_data, Some(share_nbits))
    }

    fn verify_with_nbits(
        &self,
        proposed_header: &IncompleteBlockHeader,
        public_data: &[u8],
        proof_data: &[u8],
        nbits_override: Option<u32>,
    ) -> Result<()> {
        let statement = decode_statement(public_data)?;
        let job = Fp8Job::derive(&statement, proposed_header)?;
        let nbits = nbits_override.unwrap_or(proposed_header.nbits);
        self.verify_derived(&job, proof_data, nbits)
    }

    /// Verify a compact outer proof for an already-derived job.
    /// Shared with [`Fp8VerifierCache`] to avoid deriving the job twice.
    fn verify_derived(&self, job: &Fp8Job, proof_data: &[u8], nbits: u32) -> Result<()> {
        let expected = job.expected_public_inputs();
        verify_compact_wrapped_proof(
            &job.system,
            &self.circuit,
            &self.constants_sigmas_polynomials,
            proof_data,
            &expected,
            job.params.digest(&job.proposed_header),
        )?;
        job.native_epilogue(nbits)
    }
}

impl Fp8VerifierCache {
    /// Look up the device setup from the cache; missing entries are errors.
    fn verifier_for_job(&self, job: &Fp8Job) -> Result<&Fp8Verifier> {
        let device = job.params.common().device;
        self.get(device)
            .ok_or_else(|| anyhow!("no cached fp8 verifier setup for {device:?}; regenerate fp8_cache.bin with build_cache"))
    }

    /// Verify the published pair against the expected header using cached
    /// setup. Reject if the device has no cached setup.
    pub fn verify_block(&self, proposed_header: &IncompleteBlockHeader, public_data: &[u8], proof_data: &[u8]) -> Result<()> {
        self.verify_with_nbits(proposed_header, public_data, proof_data, None)
    }

    /// Verifies a pool share under an explicit share target.
    pub fn verify_share(
        &self,
        proposed_header: &IncompleteBlockHeader,
        public_data: &[u8],
        proof_data: &[u8],
        share_nbits: u32,
    ) -> Result<()> {
        self.verify_with_nbits(proposed_header, public_data, proof_data, Some(share_nbits))
    }

    fn verify_with_nbits(
        &self,
        proposed_header: &IncompleteBlockHeader,
        public_data: &[u8],
        proof_data: &[u8],
        nbits_override: Option<u32>,
    ) -> Result<()> {
        let statement = decode_statement(public_data)?;
        let job = Fp8Job::derive(&statement, proposed_header)?;
        let verifier = self.verifier_for_job(&job)?;
        let nbits = nbits_override.unwrap_or(proposed_header.nbits);
        verifier.verify_derived(&job, proof_data, nbits)
    }
}

/// Dense setup fixture: `m = n = 32`, `k = 2048`, rank 32,
/// pattern `[(4, Fold), (4, Blake)]` on each axis. Hashes, tile bases and
/// ancestor header are zero. `build_cache` uses this envelope-legal job
/// to build the universal circuits.
pub fn sample_dense_statement() -> Result<PublicParams> {
    let pattern = AxisPattern::new(&[(4, DimType::Fold), (4, DimType::Blake)])?;
    let params = PublicParams::try_new(
        JobParams {
            ancestor_header: IncompleteBlockHeader::zero(),
            common: CommonParams {
                k: 2048,
                r: 32,
                quant: Quant::Fp8E4M3Prequant,
                device: Device::B200,
            },
            operands: Sides {
                a: OperandParams {
                    num_rows: 32,
                    hash_id: HashId::Blake3Chunk1024,
                    pattern: pattern.clone(),
                },
                b: OperandParams {
                    num_rows: 32,
                    hash_id: HashId::Blake3Chunk1024,
                    pattern,
                },
            },
            moe: None,
        },
        JackpotStatement {
            tile_bases: Sides { a: 0, b: 0 },
            hash_jackpot: [0; 32],
            hash_a: [0; 32],
            hash_b: [0; 32],
        },
        None,
    )?;
    Ok(params)
}

/// Decode and validate proof-controlled statement bytes via [`PublicParams`].
pub(crate) fn decode_statement(public_data: &[u8]) -> Result<PublicParams> {
    PublicParams::from_bytes(public_data)
}

/// Setup-time Merkle cap for the LUT tables, fixed by their contents,
/// batch layout and consensus STARK configuration.
#[derive(Clone, Debug)]
pub struct Fp8VerifierData {
    pub lut_cap: LutCap,
}

/// Job-independent LUT cap and precommitment: polynomials, LDEs and
/// batched Merkle tree.
pub struct Fp8ProverData {
    pub lut_cap: LutCap,
    pub preprocessed: BatchStarkPreprocessedData<F, C, D>,
}

impl Fp8ProverData {
    /// Derive the batch layout and commit to its LUT tables for reuse across jobs.
    pub fn generate(params: &PublicParams, proposed_header: &IncompleteBlockHeader, timing: &mut TimingTree) -> Result<Self> {
        let job = Fp8Job::derive(params, proposed_header)?;
        let preprocessed = job.system.preprocessed_data::<C>(timing);
        let lut_cap = preprocessed.cap();
        Ok(Self { lut_cap, preprocessed })
    }

    /// The verifier's view of this cached data.
    pub fn verifier_data(&self) -> Fp8VerifierData {
        Fp8VerifierData {
            lut_cap: self.lut_cap.clone(),
        }
    }
}

impl Fp8VerifierData {
    /// Builds the verifier's cached data from the consensus cap.
    pub fn new(lut_cap: LutCap) -> Self {
        Self { lut_cap }
    }
}

/// One batched FP8 STARK proof, including all tables' public inputs.
#[derive(Clone, Debug)]
pub struct Fp8Proof(pub BatchStarkProofWithPublicInputs<F, C, D>);

impl Fp8Proof {
    /// Serializes the proof (and all its public inputs) for the wire.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(&self.0).context("serializing an fp8 proof")
    }

    /// Deserialize only; use [`verify_fp8_block`] to check the proof.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self(bincode::deserialize(bytes).context("deserializing an fp8 proof")?))
    }
}

/// Public proving/verifying context: the [`Fp8System`] and the statement
/// used to derive expected public inputs and check difficulty.
pub struct Fp8Job {
    /// Statement used for public inputs, transcript digest and difficulty checks.
    params: PublicParams,
    /// The caller's expected block header.
    proposed_header: IncompleteBlockHeader,
    pub system: Fp8System<F, D>,
}

impl Fp8Job {
    /// Derive the dense or MoE job; return an error for unsupported parameters.
    pub fn derive(params: &PublicParams, proposed_header: &IncompleteBlockHeader) -> Result<Self> {
        params.recheck_moe()?;
        let (compiled, _, _) = params.compile(proposed_header)?;
        let (h, w, k, r) = (compiled.h, compiled.w, compiled.k, compiled.r);

        // Known BF16 noise values, using the same B200 matmul as `noisy_quantize`.
        let noise = compute_fp8_noise(params, proposed_header);
        let noise_codes = |e: &[u8], f: &[u8], rows: usize| -> Result<Vec<u16>> {
            Ok(B200 {}
                .matmul_fp8(e, f, None, rows, k, r)?
                .into_iter()
                .map(fp32_to_bf16_rne)
                .collect())
        };
        let a_noise = noise_codes(&noise.a.e, &noise.a.f, h).context("A-side noise codes")?;
        let b_noise = noise_codes(&noise.b.e, &noise.b.f, w).context("B-side noise codes")?;

        // Match public routing indices to the compiler's opened blocks.
        let (routing_pins, moe_schedule) = match (params.moe_statement(), &compiled.moe) {
            (Some(moe), Some(compiled_moe)) => (
                moe.routing_pins(&compiled_moe.inner_indices),
                Some(MoeSchedule::new(
                    moe,
                    params.moe().expect("MoE statement implies MoE params").experts,
                    params.m(),
                )?),
            ),
            (None, None) => (vec![], None),
            _ => unreachable!("compilation mirrors params.moe"),
        };
        let blake3 = Blake3Program::from_blake_program(&compiled.blake_proof, k, routing_pins, moe_schedule);
        let input_quant = InputQuantProgram {
            h,
            w,
            k,
            block_size: BLOCK_SIZE,
            r,
        };
        let scale = ScaleProgram::new(h, w, k, r);
        let matmul = MatmulProgram { h, w, k };
        let xor_fold = XorFoldProgram {
            lanes: lane_assignment(&params.a().pattern, &params.b().pattern),
        };

        let system = Fp8System::new(Fp8PublicData {
            blake3,
            input_quant: input_quant.clone(),
            scale: scale.clone(),
            matmul,
            xor_fold,
            a_values_len: h * k,
            a_scales_len: h * (k / BLOCK_SIZE) * 2,
            b_values_len: w * k,
            b_scales_len: w * (k / BLOCK_SIZE) * 2,
            a_noise,
            b_noise,
        });
        // Require a degree profile supported by the fixed FRI schedule.
        ensure!(
            system
                .degree_bits()
                .iter()
                .all(|bits| FP8_REACHABLE_DEGREE_BITS.contains(bits)),
            "degree profile {:?} is off the consensus FRI ladder",
            system.degree_bits()
        );

        Ok(Self {
            params: params.clone(),
            proposed_header: *proposed_header,
            system,
        })
    }

    /// The A-side and routing tree key, derived from the proposed header.
    fn key_a(&self) -> Hash256 {
        self.params.key_a(&self.proposed_header)
    }

    /// The B-side tree key, derived from the ancestor header.
    fn key_b(&self) -> Hash256 {
        self.params.key_b()
    }

    /// Lottery key derived by [`PublicParams::jackpot_key`].
    fn jackpot_key(&self) -> Hash256 {
        self.params.jackpot_key(&self.proposed_header)
    }

    /// The public routing commitment a MoE trace's routing tree root must bind to
    /// (`HASH_ROUTING`); `None` for a dense job (the trace slots stay zero).
    fn hash_routing(&self) -> Option<Hash256> {
        self.params.moe_statement().map(|moe| moe.hash_routing)
    }

    /// The public offsets commitment (`HASH_OFFSETS`); `None` for a dense job.
    fn hash_offsets(&self) -> Option<Hash256> {
        self.params.moe_statement().map(|moe| moe.hash_offsets)
    }

    /// The Scale program (expected-public-input assembly and jackpot allowances).
    fn scale(&self) -> ScaleProgram {
        let params = &self.params;
        let as_usize = |v: u32| usize::try_from(v).expect("geometry fits usize");
        ScaleProgram::new(
            as_usize(params.h()),
            as_usize(params.w()),
            as_usize(params.common_dim()),
            as_usize(params.rank()),
        )
    }

    /// The InputQuant program (expected-public-input assembly and CTL geometry).
    fn input_quant(&self) -> InputQuantProgram {
        let params = &self.params;
        let as_usize = |v: u32| usize::try_from(v).expect("geometry fits usize");
        InputQuantProgram {
            h: as_usize(params.h()),
            w: as_usize(params.w()),
            k: as_usize(params.common_dim()),
            block_size: BLOCK_SIZE,
            r: as_usize(params.rank()),
        }
    }

    /// Expected public inputs for every main table, in canonical `Table` order.
    fn expected_public_inputs(&self) -> [Vec<F>; NUM_TABLES] {
        let params = &self.params;
        let mut blake3 = vec![F::ZERO; NUM_BLAKE3_PUBLIC_INPUTS];
        let mut set_hash = |base: usize, hash: &Hash256| {
            blake3[base..base + 8].copy_from_slice(&hash_to_u32_field_array(hash));
        };
        set_hash(PI_KEY_A, &self.key_a());
        set_hash(PI_KEY_B, &self.key_b());
        set_hash(PI_JACKPOT_KEY, &self.jackpot_key());
        set_hash(PI_HASH_A, &params.hash_a());
        set_hash(PI_HASH_B, &params.hash_b());
        // Dense jobs have no routing or offsets trees, so those slots are zero.
        set_hash(PI_HASH_ROUTING, &self.hash_routing().unwrap_or([0u8; 32]));
        set_hash(PI_HASH_OFFSETS, &self.hash_offsets().unwrap_or([0u8; 32]));
        set_hash(PI_HASH_JACKPOT, &params.hash_jackpot());

        let scale = self.scale().public_inputs::<F>().to_vec();
        // InputQuant exposes `2^Wl2`; Scale exposes `Wl2`. Pin both from the geometry.
        let input_quant_program = self.input_quant();
        let input_quant = input_quant_program.public_inputs::<F>().to_vec();
        let tamed = TamedProgram {
            h: input_quant_program.h,
            w: input_quant_program.w,
            k: input_quant_program.k,
        }
        .public_inputs::<F>()
        .to_vec();
        [blake3, input_quant, scale, vec![], vec![], tamed]
    }

    /// Check difficulty against `nbits` after proof verification has bound the
    /// jackpot digest to the statement. Shared by native and recursive paths.
    fn native_epilogue(&self, nbits: u32) -> Result<()> {
        let params = &self.params;
        check_jackpot_difficulty(&params.hash_jackpot(), nbits, params.h(), params.w(), params.common_dim())
    }
}

/// Verify the plain proof's openings and prove the FP8 batch.
/// Return the public statement, updated with the proven jackpot digest,
/// and the batch proof. Reuse data from [`Fp8ProverData::generate`].
pub fn zk_prove_plain_proof_fp8(
    proposed_header: &IncompleteBlockHeader,
    plain_proof: &PlainProofV4,
    prover_data: &Fp8ProverData,
    timing: &mut TimingTree,
) -> Result<(PublicParams, Fp8Proof)> {
    let (public, _, proof) = prove_batch_statement(proposed_header, plain_proof, prover_data, timing)?;
    Ok((public, Fp8Proof(proof)))
}

/// Run [`zk_prove_plain_proof_fp8`] and wrap it into an outer zero-knowledge
/// Plonky2 proof. `prover_data` comes from [`Fp8ProverData::generate`];
/// `cache` reuses the compiled wrappers across job geometries.
pub fn zk_prove_plain_proof_fp8_wrapped(
    proposed_header: &IncompleteBlockHeader,
    plain_proof: &PlainProofV4,
    prover_data: &Fp8ProverData,
    cache: &mut Fp8CircuitCache,
    timing: &mut TimingTree,
) -> Result<(PublicParams, ProofWithPublicInputs<F, OuterC, D>)> {
    let (public, job, batch_proof) = prove_batch_statement(proposed_header, plain_proof, prover_data, timing)?;
    let circuits = Fp8WrapperCircuits::with_cache(&job.system, &prover_data.lut_cap, cache, timing)?;
    // The proving callback updated `public` with the jackpot digest.
    // Use it here: `job.params` still contains the pre-proving statement.
    let wrapped = circuits.prove(&job.system, &batch_proof, public.digest(proposed_header), timing)?;
    Ok((public, wrapped))
}

/// Return the updated statement, derived job and batch proof.
/// The returned job retains the pre-proving jackpot claim.
fn prove_batch_statement(
    proposed_header: &IncompleteBlockHeader,
    plain_proof: &PlainProofV4,
    prover_data: &Fp8ProverData,
    timing: &mut TimingTree,
) -> Result<(PublicParams, Fp8Job, BatchStarkProofWithPublicInputs<F, C, D>)> {
    let (private, mut public) = plain_proof.parse_proof(proposed_header)?;
    let mut job = Fp8Job::derive(&public, proposed_header)?;

    // The cached precommitment must match this job's batch layout and the consensus cap.
    let expected_preprocessed = job.system.preprocessed_verifier_data::<C>(&prover_data.lut_cap);
    ensure!(
        prover_data.preprocessed.columns_per_table == expected_preprocessed.columns_per_table,
        "the cached LUT precommitment does not match this job's batch layout; \
         regenerate the prover data"
    );
    ensure!(
        prover_data.preprocessed.cap() == prover_data.lut_cap,
        "the cached LUT precommitment disagrees with the consensus cap"
    );

    // `parse_proof` authenticated these routing blocks and offsets.
    // Convert them to the word order consumed by the trace.
    let routing_words = stream_words(&private.s_routing);
    let offsets_words = stream_words(&private.s_offsets);
    let witness = Fp8Witness {
        a_values: private.operands.a.values.as_bytes(),
        a_scales: private.operands.a.scales.as_bytes(),
        b_values: private.operands.b.values.as_bytes(),
        b_scales: private.operands.b.scales.as_bytes(),
        routing_words: &routing_words,
        offsets_words: &offsets_words,
        aux_msgs: &private.external_msgs,
        aux_cvs: &private.external_cvs,
        key_a: hash_words(&job.key_a()),
        key_b: hash_words(&job.key_b()),
        jackpot_key: hash_words(&job.jackpot_key()),
        a_hash_id: public.a().hash_id,
        b_hash_id: public.b().hash_id,
        routing_hash_id: public.moe().map(|m| m.hash_id_r).unwrap_or(HashId::Blake3Chunk1024),
        offsets_hash_id: public.moe().map(|m| m.hash_id_o).unwrap_or(HashId::Blake3Chunk1024),
    };
    // Set the jackpot before Fiat-Shamir: `PublicParams::to_bytes()` includes
    // it, so the statement digest must use the generated value.
    let proof = job.system.prove::<C>(
        &witness,
        |jackpot| {
            public.set_hash_jackpot(jackpot);
            public.digest(proposed_header)
        },
        &prover_data.preprocessed,
        timing,
    )?;

    Ok((public, job, proof))
}

/// Verify a native batch proof against the expected statement and trusted LUT cap.
/// Also check the proven jackpot against the header's difficulty, or `nbits_override`.
pub fn verify_fp8_block(
    public_params: &PublicParams,
    proposed_header: &IncompleteBlockHeader,
    proof: &Fp8Proof,
    verifier_data: &Fp8VerifierData,
    nbits_override: Option<u32>,
) -> Result<()> {
    let mut job = Fp8Job::derive(public_params, proposed_header)?;
    let nbits = nbits_override.unwrap_or(proposed_header.nbits);
    verify_block_with_job(&mut job, proof, verifier_data, nbits)
}

/// The [`verify_fp8_block`] core on an already-derived statement.
fn verify_block_with_job(job: &mut Fp8Job, proof: &Fp8Proof, verifier_data: &Fp8VerifierData, nbits: u32) -> Result<()> {
    let expected = job.expected_public_inputs();
    let digest = job.params.digest(&job.proposed_header);
    job.system.bind_statement_digest(digest);
    job.system.verify::<C>(&proof.0, &expected, &verifier_data.lut_cap)?;
    job.native_epilogue(nbits)
}

/// Verify the recursive proof against the expected statement and trusted outer `circuit`.
///
/// [`verify_wrapped_proof`] binds the public inputs, statement digest and known-column
/// evaluations. The native check then compares the jackpot against the header's
/// difficulty, or `nbits_override`.
pub fn verify_fp8_block_wrapped(
    public_params: &PublicParams,
    proposed_header: &IncompleteBlockHeader,
    proof: &ProofWithPublicInputs<F, OuterC, D>,
    circuit: &VerifierCircuitData<F, OuterC, D>,
    nbits_override: Option<u32>,
) -> Result<()> {
    let job = Fp8Job::derive(public_params, proposed_header)?;
    let nbits = nbits_override.unwrap_or(proposed_header.nbits);
    verify_wrapped_with_job(&job, proof, circuit, nbits)
}

/// [`verify_fp8_block_wrapped`] with an already-derived job, also used by
/// [`Fp8Verifier`] and [`Fp8VerifierCache`].
fn verify_wrapped_with_job(
    job: &Fp8Job,
    proof: &ProofWithPublicInputs<F, OuterC, D>,
    circuit: &VerifierCircuitData<F, OuterC, D>,
    nbits: u32,
) -> Result<()> {
    let expected = job.expected_public_inputs();
    verify_wrapped_proof(
        &job.system,
        circuit,
        proof,
        &expected,
        job.params.digest(&job.proposed_header),
    )?;
    job.native_epilogue(nbits)
}

/// A 32-byte hash as 8 little-endian u32 words (the STARK witness key encoding).
fn hash_words(hash: &Hash256) -> [u32; 8] {
    core::array::from_fn(|i| u32::from_le_bytes(hash[4 * i..4 * i + 4].try_into().unwrap()))
}

#[cfg(test)]
mod tests {

    use plonky2::field::types::Field;

    use super::*;
    use crate::circuit::fp8::circuit_utils::Fp8VerifierKey;
    use crate::circuit::fp8::consistency::{fixture_job, fixture_job_asym, fixture_job_k32, fixture_job_medium, fixture_job_moe};
    use crate::circuit::fp8::ctl::Table;
    use crate::circuit::fp8::input_quant_stark::columns::{
        B_KEY_OFFSET_PUBLIC_INPUT, OPERAND_MULT_A_PUBLIC_INPUT, WL2_POW_PUBLIC_INPUT,
    };
    use crate::circuit::fp8::scale_stark::columns::{K_PUBLIC_INPUT, WL2_PUBLIC_INPUT};

    // The fixtures use an easy `nbits` that saturates the difficulty bound, so the plain
    // winning condition `hash_jackpot <= bound` is deterministic on pseudo-random data.

    #[test]
    fn compile_jackpot_key_is_labelled_subkey_for_dense_and_moe() {
        use crate::api::fp8::transcript::{LABEL_JACKPOT, subkey};

        for (name, (header, plain)) in [("dense", fixture_job()), ("moe", fixture_job_moe())] {
            let (_, params) = plain.parse_proof(&header).expect("must parse");
            let seed_a = params.noise_seeds(&header).a;
            assert_eq!(params.jackpot_key(&header), subkey(LABEL_JACKPOT, Some(&seed_a)), "{name}");
            assert_ne!(params.jackpot_key(&header), seed_a, "{name}");
        }
    }

    /// Round-trip the batch proof and reject modified jackpot claims,
    /// public inputs and openings.
    #[test]
    fn api_roundtrip_and_tamper_rejection() {
        let (header, plain) = fixture_job();
        let mut timing = TimingTree::default();

        let (_, params) = plain.parse_proof(&header).expect("fixture job must parse");
        assert_eq!(params.common().device, Device::B200, "the wire byte must parse back");
        let prover_data = Fp8ProverData::generate(&params, &header, &mut timing).expect("prover setup");

        let (public, proof) = zk_prove_plain_proof_fp8(&header, &plain, &prover_data, &mut timing).expect("proving must succeed");
        assert_ne!(public.hash_jackpot(), [0u8; 32], "prove must ship the proven lottery digest");

        // Wire round trip.
        let proof = Fp8Proof::from_bytes(&proof.to_bytes().expect("serialize")).expect("deserialize");

        let verifier_data = prover_data.verifier_data();
        verify_fp8_block(&public, &header, &proof, &verifier_data, None).expect("honest proof must verify");

        // Tamper 1: a different shipped lottery digest is a public-input mismatch.
        let mut bad_public = public.clone();
        bad_public.set_hash_jackpot({
            let mut h = bad_public.hash_jackpot();
            h[0] ^= 1;
            h
        });
        assert!(verify_fp8_block(&bad_public, &header, &proof, &verifier_data, None).is_err());

        // Changing the Scale K public input must invalidate the proof.
        let job = Fp8Job::derive(&public, &header).unwrap();
        let mut tampered = proof.clone();
        tampered.0.public_inputs[job.system.main_table_positions()[Table::Scale as usize]][K_PUBLIC_INPUT] += F::ONE;
        assert!(verify_fp8_block(&public, &header, &tampered, &verifier_data, None).is_err());

        // Tamper 3: any opening perturbation breaks the batched FRI argument.
        let mut tampered = proof.clone();
        tampered.0.proof.openings[0].local_values[0] += F::ONE.into();
        assert!(verify_fp8_block(&public, &header, &tampered, &verifier_data, None).is_err());
    }

    /// Exercise padding with [`fixture_job_asym`]: `h = 16`, `w = 20`, `k = 2048`.
    /// Neither `h + w` nor `h * w` is a power of two.
    #[test]
    fn asymmetric_api_roundtrip() {
        let (header, plain) = fixture_job_asym();
        let mut timing = TimingTree::default();

        let (_, params) = plain.parse_proof(&header).expect("asymmetric fixture job must parse");
        assert_ne!(params.h(), params.w(), "the fixture must be asymmetric");
        let prover_data = Fp8ProverData::generate(&params, &header, &mut timing).expect("prover setup");

        let (public, proof) = zk_prove_plain_proof_fp8(&header, &plain, &prover_data, &mut timing).expect("proving must succeed");
        let proof = Fp8Proof::from_bytes(&proof.to_bytes().expect("serialize")).expect("deserialize");

        let verifier_data = prover_data.verifier_data();
        verify_fp8_block(&public, &header, &proof, &verifier_data, None).expect("honest proof must verify");
    }

    /// [`fixture_job_k32`] uses `k = 2080`: rows cross 64-byte BLAKE3 boundaries
    /// and scale derivation uses a non-power-of-two number of blocks.
    #[test]
    fn k_mod_32_api_roundtrip() {
        let (header, plain) = fixture_job_k32();
        let mut timing = TimingTree::default();

        let (_, params) = plain.parse_proof(&header).expect("k % 32 fixture job must parse");
        assert_eq!(params.common_dim() % 64, 32, "the fixture must exercise the k % 32 envelope");
        let prover_data = Fp8ProverData::generate(&params, &header, &mut timing).expect("prover setup");

        let (public, proof) = zk_prove_plain_proof_fp8(&header, &plain, &prover_data, &mut timing).expect("proving must succeed");
        let proof = Fp8Proof::from_bytes(&proof.to_bytes().expect("serialize")).expect("deserialize");

        let verifier_data = prover_data.verifier_data();
        verify_fp8_block(&public, &header, &proof, &verifier_data, None).expect("honest proof must verify");
    }

    /// The preamble's extension-field zeta must occupy the expected public-input slots.
    #[test]
    fn compact_preamble_zeta_binding() {
        use plonky2::field::extension::quadratic::QuadraticExtension;

        use crate::circuit::fp8::wrapper::{COMPACT_ZETA_PREAMBLE, expected_wrapper_public_inputs, zeta_offset};

        let (header, plain) = fixture_job();
        let (_, params) = plain.parse_proof(&header).expect("fixture job must parse");
        let job = Fp8Job::derive(&params, &header).expect("fixture job must derive");
        let expected_pis = job.expected_public_inputs();
        let digest = params.digest(&header);

        // The preamble is exactly zeta's D basefield limbs, 8 bytes each.
        assert_eq!(COMPACT_ZETA_PREAMBLE, 8 * D);

        // A basefield zeta is rejected before any vector is built.
        let basefield = QuadraticExtension([F::from_canonical_u64(7), F::ZERO]);
        assert!(
            expected_wrapper_public_inputs(&job.system, &expected_pis, digest, basefield)
                .expect_err("basefield zeta must be rejected")
                .to_string()
                .contains("extension field"),
            "the rejection must name the basefield rule"
        );

        // Expected public inputs are determined by the statement and zeta.
        let zeta = QuadraticExtension([F::from_canonical_u64(3), F::from_canonical_u64(41)]);
        let expected = expected_wrapper_public_inputs(&job.system, &expected_pis, digest, zeta).expect("legal zeta must build");
        let z = zeta_offset(&job.system);
        assert_eq!(&expected[z..z + D], &zeta.0, "zeta must land at the preamble's slots");
        let again = expected_wrapper_public_inputs(&job.system, &expected_pis, digest, zeta).expect("legal zeta must build");
        assert_eq!(expected, again, "the imposed vector must be deterministic");
    }

    /// Different geometries and degree profiles must select the same device setup.
    #[test]
    fn verifier_key_is_universal_across_geometries() {
        let (header, plain) = fixture_job();
        let (_, params) = plain.parse_proof(&header).expect("fixture job must parse");
        let job = Fp8Job::derive(&params, &header).expect("fixture job must derive");
        let key = Fp8VerifierKey::new(job.params.common().device);

        // A different geometry (k = 2080 vs 2048, hence a different degree profile too)
        // resolves to the same universal setup.
        let (header_k32, plain_k32) = fixture_job_k32();
        let (_, params_k32) = plain_k32.parse_proof(&header_k32).expect("k32 fixture must parse");
        let job_k32 = Fp8Job::derive(&params_k32, &header_k32).expect("k32 fixture must derive");
        assert_ne!(
            job.system.geometry(),
            job_k32.system.geometry(),
            "the fixtures must differ in geometry for this test to bite"
        );
        assert_eq!(
            key,
            Fp8VerifierKey::new(job_k32.params.common().device),
            "different geometry: one universal setup"
        );
    }

    #[test]
    fn missing_verifier_cache_rejects_before_proof_decoding() {
        let statement = sample_dense_statement().unwrap();
        let error = Fp8VerifierCache::default()
            .verify_block(&IncompleteBlockHeader::zero(), &statement.to_bytes(), &[])
            .unwrap_err();
        assert!(error.to_string().contains("no cached fp8 verifier setup"), "{error:#}");
    }

    /// Regenerates the wrapped proof consumed by the Go V4 certificate tests.
    #[test]
    #[ignore = "fixture generator: expensive wrapped proof; run with task generate:fp8-fixture"]
    fn regenerate_go_fixture() {
        let (header, plain) = fixture_job();
        let mut prover = Fp8Prover::setup(&header, &plain).expect("fp8 setup");
        let (public_data, proof_data) = prover.prove(&header, &plain).expect("wrapped proving must succeed");

        let public_data_len = u32::try_from(public_data.len()).expect("public data length must fit u32");
        let mut fixture = Vec::with_capacity(IncompleteBlockHeader::SERIALIZED_SIZE + 4 + public_data.len() + proof_data.len());
        fixture.extend_from_slice(&header.to_bytes());
        fixture.extend_from_slice(&public_data_len.to_le_bytes());
        fixture.extend_from_slice(&public_data);
        fixture.extend_from_slice(&proof_data);

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../node/zkpow/testdata/fp8_zk_proof_b200.bin");
        std::fs::write(&path, fixture).expect("write Go fp8 fixture");
        println!("Go fixture written to {}", path.display());
    }

    /// Round-trip a compact recursive proof, reuse setup for another geometry,
    /// and reject modified statements, public inputs, zeta and proof bytes.
    #[test]
    #[ignore = "exceeds the safe memory budget of an 8 GiB CI runner; run explicitly on a larger machine"]
    fn wrapped_api_roundtrip_and_tamper_rejection() {
        use crate::circuit::fp8::wrapper::{COMPACT_ZETA_PREAMBLE, degree_bits_offset, table_pis_offset, zeta_offset};

        // Surface the wrapper's gate/size logs under `RUST_LOG=info`.
        let _ = env_logger::builder().format_timestamp(None).try_init();

        let mut timing = TimingTree::default();
        let (header, plain) = fixture_job();
        let mut prover = Fp8Prover::setup_with_timing(&header, &plain, &mut timing).expect("fp8 setup");
        // Keep a full proof for public-input tampering tests, then encode it
        // as [`Fp8Prover::prove`] does.
        let (public, wrapped) =
            zk_prove_plain_proof_fp8_wrapped(&header, &plain, &prover.data, &mut prover.circuits, &mut timing)
                .expect("wrapped proving must succeed");
        let job = Fp8Job::derive(&public, &header).unwrap();
        let public_data = public.to_bytes();
        let proof_data = compact_proof_data(&job.system, &wrapped).expect("compact encode");

        // Build trusted setup independently from the public statement.
        let statement = decode_statement(&public_data).expect("public data must decode");
        let verifier = Fp8Verifier::generate(&statement, &header, &mut timing).expect("verifier-side setup");
        assert!(
            verifier.circuit.common.config.zero_knowledge,
            "the published stage must carry plonky2's ZK blinding"
        );

        // Round-trip setup bytes before verifying the published byte pair.
        let verifier_bytes = verifier.to_bytes().expect("serialize verifier setup");
        let verifier = Fp8Verifier::from_bytes(&verifier_bytes).expect("deserialize verifier setup");

        verifier
            .verify_block(&header, &public_data, &proof_data)
            .expect("honest wrapped proof must verify");

        // Exercise verification through a serialized and reloaded cache.
        let mut cache = Fp8VerifierCache::default();
        cache.insert(&statement, verifier.clone());
        let cache_bytes = cache.to_bytes().expect("serialize cache");
        let cache = Fp8VerifierCache::from_bytes(&cache_bytes).expect("deserialize cache");
        assert_eq!(cache.len(), 1);
        cache
            .verify_block(&header, &public_data, &proof_data)
            .expect("the cached setup must verify the honest proof");
        assert!(
            Fp8VerifierCache::from_bytes(&[])
                .expect("an empty blob is a valid cache")
                .is_empty(),
            "the no-embedded-cache default must load as an empty cache"
        );

        // An empty cache must reject without compiling circuits.
        let err = Fp8VerifierCache::default()
            .verify_block(&header, &public_data, &proof_data)
            .expect_err("an uncached setup must reject, not compile");
        assert!(
            err.to_string().contains("no cached fp8 verifier setup"),
            "unexpected error: {err:#}"
        );

        // A different geometry and degree profile must reuse the same cached circuits.
        let (header_k32, plain_k32) = fixture_job_k32();
        let (public_k32, proof_k32) = prover
            .prove_with_timing(&header_k32, &plain_k32, &mut timing)
            .expect("the prover must prove a different geometry");
        assert_eq!(prover.circuits.len(), 1, "one universal circuit set");
        verifier
            .verify_block(&header_k32, &public_k32, &proof_k32)
            .expect("one universal setup must verify every geometry");
        cache
            .verify_block(&header_k32, &public_k32, &proof_k32)
            .expect("the cached setup must verify the second geometry");
        assert_eq!(cache.len(), 1, "no per-shape setups: the one entry serves both");

        // Re-encoding the decoded statement reproduces the published bytes.
        assert_eq!(statement.to_bytes(), public_data);

        // Tamper 1: a different shipped lottery digest is a public-input mismatch.
        let mut tampered = statement.clone();
        tampered.set_hash_jackpot({
            let mut h = tampered.hash_jackpot();
            h[0] ^= 1;
            h
        });
        let bad_public = tampered.to_bytes();
        assert!(verifier.verify_block(&header, &bad_public, &proof_data).is_err());

        // Modify geometry, degree, zeta and known-column evaluation slots.
        // Verification must bind each to the statement-derived expectations.
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
        ] {
            let mut tampered = wrapped.clone();
            tampered.public_inputs[slot] += F::ONE;
            assert!(
                verify_fp8_block_wrapped(&public, &header, &tampered, &verifier.circuit, None).is_err(),
                "tampered public input slot {slot} must be rejected"
            );
        }

        // Public inputs are reconstructed, so editing the vector does not change compact bytes.
        let mut mutated = wrapped.clone();
        mutated.public_inputs[scale_pis + K_PUBLIC_INPUT] += F::ONE;
        assert_eq!(
            compact_proof_data(&job.system, &mutated).expect("compact encode"),
            proof_data,
            "imposed slots must not ride the compact wire"
        );
        // Zeta is serialized in the preamble; modifying it must invalidate the proof.
        for limb in 0..D {
            let mut tampered = wrapped.clone();
            tampered.public_inputs[zeta_offset(&job.system) + limb] += F::ONE;
            let bytes = compact_proof_data(&job.system, &tampered).expect("compact encode");
            assert!(
                verifier.verify_block(&header, &public_data, &bytes).is_err(),
                "a preamble zeta shifted in limb {limb} must be rejected"
            );
        }

        // Reject corrupted proof bytes, non-canonical zeta limbs and truncated proofs.
        let mut corrupt = proof_data.clone();
        let mid = COMPACT_ZETA_PREAMBLE + (proof_data.len() - COMPACT_ZETA_PREAMBLE) / 2;
        corrupt[mid] ^= 1;
        assert!(verifier.verify_block(&header, &public_data, &corrupt).is_err());

        let mut noncanonical = proof_data.clone();
        noncanonical[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(verifier.verify_block(&header, &public_data, &noncanonical).is_err());

        assert!(
            verifier
                .verify_block(&header, &public_data, &proof_data[..COMPACT_ZETA_PREAMBLE])
                .is_err(),
            "a bare preamble with no proof body must be rejected"
        );

        // Verification is tied to the caller's expected block, not merely the
        // header embedded in the published statement.
        let mut other_header = header;
        other_header.timestamp ^= 1;
        assert!(verifier.verify_block(&other_header, &public_data, &proof_data).is_err());

        // Trailing bytes are rejected in both encodings.
        let mut trailing = public_data.clone();
        trailing.push(0);
        assert!(verifier.verify_block(&header, &trailing, &proof_data).is_err());

        let mut trailing = proof_data.clone();
        trailing.push(0);
        assert!(verifier.verify_block(&header, &public_data, &trailing).is_err());

        let mut trailing = verifier_bytes;
        trailing.push(0);
        assert!(Fp8Verifier::from_bytes(&trailing).is_err());
    }

    /// Full published pipeline at `h = w = 16`, `k = 4096`, for measuring
    /// proving and verification beyond the smallest fixture.
    #[test]
    #[ignore = "exceeds the safe memory budget of an 8 GiB CI runner; run explicitly on a larger machine"]
    fn medium_wrapped_roundtrip() {
        let (header, plain) = fixture_job_medium();
        let mut timing = TimingTree::default();

        let (_, params) = plain.parse_proof(&header).expect("medium fixture job must parse");
        let prover_data = Fp8ProverData::generate(&params, &header, &mut timing).expect("prover setup");
        let mut cache = Fp8CircuitCache::default();

        let (public, proof) = zk_prove_plain_proof_fp8_wrapped(&header, &plain, &prover_data, &mut cache, &mut timing)
            .expect("medium wrapped proving must succeed");

        let job = Fp8Job::derive(&public, &header).unwrap();
        let circuit = Fp8WrapperCircuits::with_cache(&job.system, &prover_data.lut_cap, &mut cache, &mut timing)
            .expect("cache hit")
            .verifier_data();
        verify_fp8_block_wrapped(&public, &header, &proof, &circuit, None).expect("honest medium wrapped proof must verify");
    }

    /// Round-trip a MoE batch proof; reject modified selected rows,
    /// routing commitments and expert indices.
    #[test]
    fn moe_api_roundtrip_and_tamper_rejection() {
        let (header, plain) = fixture_job_moe();
        let mut timing = TimingTree::default();

        let (_, params) = plain.parse_proof(&header).expect("MoE fixture job must parse");
        let moe = params.moe_statement().expect("the fixture is MoE").clone();

        // Check the MoE jackpot key against the labelled subkey.
        let job = Fp8Job::derive(&params, &header).expect("MoE jobs must derive");
        let a_noise_seed = params.noise_seeds(&header).a;
        assert_eq!(
            job.jackpot_key(),
            crate::api::fp8::transcript::subkey(crate::api::fp8::transcript::LABEL_JACKPOT, Some(&a_noise_seed)),
            "the MoE lottery key must be Subkey(noise_seedA, jackpot)"
        );
        assert_ne!(
            job.jackpot_key(),
            a_noise_seed,
            "lottery key is the jackpot subkey, not the raw A seed"
        );
        assert_eq!(job.hash_routing(), Some(moe.hash_routing));

        let prover_data = Fp8ProverData::generate(&params, &header, &mut timing).expect("prover setup");
        let (public, proof) =
            zk_prove_plain_proof_fp8(&header, &plain, &prover_data, &mut timing).expect("MoE proving must succeed");
        let proof = Fp8Proof::from_bytes(&proof.to_bytes().expect("serialize")).expect("deserialize");

        let verifier_data = prover_data.verifier_data();
        verify_fp8_block(&public, &header, &proof, &verifier_data, None).expect("honest MoE proof must verify");

        // The proof's HASH_ROUTING must match the authenticated public commitment.
        let blake3_pis = &proof.0.public_inputs[job.system.main_table_positions()[0]];
        assert_eq!(
            &blake3_pis[PI_HASH_ROUTING..PI_HASH_ROUTING + 8],
            &hash_to_u32_field_array(&moe.hash_routing),
            "the proven routing root must be the job's routing commitment"
        );

        // Changing a selected row changes the expected known columns.
        let mut bad_outer = public.clone();
        bad_outer.moe_statement_mut().unwrap().i_a[0] += 1;
        assert!(
            verify_fp8_block(&bad_outer, &header, &proof, &verifier_data, None).is_err(),
            "a forged sampled outer index must be rejected"
        );

        // Tamper 2: a forged routing commitment moves the expected HASH_ROUTING.
        let mut bad_routing = public.clone();
        bad_routing.moe_statement_mut().unwrap().hash_routing[0] ^= 1;
        assert!(
            verify_fp8_block(&bad_routing, &header, &proof, &verifier_data, None).is_err(),
            "a forged routing commitment must be rejected"
        );

        // Changing the expert must invalidate the proof.
        let mut bad_expert = public.clone();
        bad_expert.moe_statement_mut().unwrap().w = 0;
        assert!(
            verify_fp8_block(&bad_expert, &header, &proof, &verifier_data, None).is_err(),
            "a proof must not verify under a different expert's statement"
        );
    }

    /// Check [`PublicParams::WIRE_SIZE`], [`PublicParams::is_valid_wire_size`]
    /// and statement round-trips against actual encoded bytes.
    #[test]
    fn wire_size_constants_match_encoding() {
        let params = sample_dense_statement().expect("canonical dense");

        let encoded = params.to_bytes();
        assert_eq!(encoded.len(), PublicParams::WIRE_SIZE);
        assert!(PublicParams::is_valid_wire_size(encoded.len()));
        assert!(PublicParams::is_valid_wire_size(PublicParams::MAX_WIRE_SIZE));
        assert!(!PublicParams::is_valid_wire_size(PublicParams::WIRE_SIZE - 1));
        assert!(!PublicParams::is_valid_wire_size(PublicParams::MAX_WIRE_SIZE + 1));
        // `from_bytes ∘ to_bytes` is fixed-point on the canonical dense statement.
        let decoded = PublicParams::from_bytes(&encoded).expect("dense re-decode");
        assert_eq!(decoded.to_bytes(), encoded);
    }

    #[test]
    fn routing_pins_maps_sampled_slots_to_stream_words() {
        let stmt = |o_w_prev: u32, o_w: u32, i_a: Vec<u32>| MoEStatement {
            w: 0,
            o_w_prev,
            o_w,
            o_last: o_w,
            hash_routing: [0u8; 32],
            hash_offsets: [0u8; 32],
            i_a,
        };
        assert_eq!(
            stmt(4, 8, vec![9, 11, 20, 33]).routing_pins(&[0, 1, 2, 3]),
            vec![(4, 9), (5, 11), (6, 20), (7, 33)]
        );
        assert_eq!(stmt(30, 36, vec![7, 2]).routing_pins(&[0, 5]), vec![(14, 7), (19, 2)]);
    }
}
