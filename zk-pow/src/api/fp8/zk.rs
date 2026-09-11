//! ZK verification API for the prequant fp8 scheme — glue between
//! [`crate::api::fp8::plain_proof::PlainProofV4`] / [`PublicParams`] and the
//! `fp8` batch multi-STARK driver ([`Fp8System`]).
//!
//! **Miner/node surface.** Four types: [`PlainProofV4`], [`Fp8Prover`], [`Fp8Verifier`]
//! and [`Fp8VerifierCache`].
//! [`Fp8Prover::prove`] returns the published artifact as the same `(public_data,
//! proof_data)` byte pair the v1/v2 certificates carry; verification accepts (`Ok(())`)
//! or rejects (`Err`) — the jackpot policy is a binary gate and no credit is reported.
//! The trusted verifier setup is generated on the verifier side from the published
//! statement alone ([`Fp8Verifier::generate`]) and round-trips through bytes.
//!
//! **Verifier setup resolution.** A verifier needs one [`Fp8Verifier`]: the stage-1
//! wrapper is the *universal* batch verifier (D1) — the degree profile and geometry are
//! public inputs, not circuit shape — so a single setup covers every envelope-legal job.
//! [`Fp8VerifierCache`] holds it keyed by the statement's device byte, preloaded
//! from [`embedded_cache::CACHE_DATA`](crate::api::fp8::embedded_cache::CACHE_DATA)
//! (built offline by `build_cache` against the committed caps file
//! [`crate::api::fp8::lut_caps`] — the verifier side never builds a LUT oracle).
//! Verification is lookup-only: a setup missing from the cache rejects the proof
//! rather than compiling it on demand, so no input can force an expensive
//! circuit build through the verify path.
//!
//! **Cached data.** [`Fp8ProverData`] is the prover's cache: the LUT cap (the committed
//! consensus value of [`crate::api::fp8::lut_caps`]) plus
//! the full LUT precommitment oracle (LDEs + batched Merkle tree — prover-only). Both are
//! job-independent: the batch order, the LUT positions and the consensus
//! config are canonical constants. [`Fp8VerifierData`] is the verifier's cached view of
//! the cap and an argument of [`verify_fp8_block`], exactly the "cached commitment" shape.
//!
//! **The statement and its public inputs.** [`Fp8Job::derive`] rebuilds the full
//! statement from public data alone: the five programs (the Blake3 schedule from the shared
//! job compiler, the geometry-parameterized InputQuant/Scale/Matmul programs, the extractor
//! lanes), the class (a) recompute inputs (plane byte lengths and the bf16 noise codes
//! `(E @ F)[elem]` from the public seeds), and the key material. The proof ships every
//! table's public inputs; the verifier pins every slot to its own recomputation — `KEY_A` /
//! `KEY_B`, `JACKPOT_KEY` (`Subkey(noise_seedA, "pearl/v4/FP8/jackpot")`),
//! `HASH_A` / `HASH_B` (the job's side digests, bound in-AIR by the commit-fold wrappers —
//! AIR spec Section 1.0), `HASH_ROUTING`, `HASH_JACKPOT`
//! (the block's shipped claim, subsequently difficulty-checked), the geometry parameters
//! `K` / `WL2` / `2^WL2` (Scale's and InputQuant's program-independence slots, all filled
//! from the statement's `k`), the `dr`/`dos` scale constants (from the job's rank) and
//! the jackpot liveness allowances `DEAD_LIMIT_A/B` (from the statement's geometry).
//!
//! **Scheme boundary.** Everything here rejects `mma_type != Bf16ToFp8Fp32` up front.
//!
//! **MoE.** A MoE job adds the routing statement, handled exactly like the deployed (v2)
//! chip: the Blake3 forest gains the routing tree (a sparse opening of the flat routing
//! array's keyed chunk tree, scheduled by the shared job compiler from public data), whose
//! root binds to `HASH_ROUTING = moe.hash_routing`; the opened hotspot blocks are **witness**
//! bytes of which only the sampled entries are publicly pinned — [`MoEStatement::routing_pins`] maps
//! each public `(inner, outer)` index pair to its stream word, the trace's
//! `IS_FIRST/SECOND_OUTER` selectors fire exactly there, and the unsampled neighbor words
//! stay free witness bound only by the hash chain (deployed selective-pinning semantics).
//! Key material is MoE-aware end to end: the a-side noise seed folds the routing commitment
//! (`compute_hash_activations`, inside v2 `PublicProofParams::commitment_hash`) and the
//! lottery runs under `H_"jackpot"(z; noise_seedA)` — the same subkey as dense, with no
//! expert index — bit-identical to `api::verify`.
//!
//! **Wrapped (published) mode.** The batch proof is what the network *verifies*; what it
//! *publishes* is the two-stage recursive wrap of [`crate::circuit::fp8::wrapper`]: the
//! batch verification encoded in a plonky2 circuit (stage 1), re-wrapped with
//! `zero_knowledge: true` (stage 2) — mirroring the deployed v2 `pearl_circuit`
//! architecture, including its circuit-cache pattern: [`zk_prove_plain_proof_fp8_wrapped`]
//! takes a [`Fp8CircuitCache`] and compiles the universal wrapper circuits on the first
//! job, reusing them afterwards. [`verify_fp8_block_wrapped`] verifies the
//! published proof: the same statement derivation and native epilogues as the unwrapped
//! path, with the batch-level checks replaced by the wrapper's public-input pinning (every
//! slot recomputed from the statement, the class (a) columns natively re-evaluated at the
//! proof's `zeta`) plus one plonky2 verification against the consensus stage-2 verifier
//! data.
//!
//! **The published wire format is compact**, as in the deployed v1/v2 certificates:
//! [`Fp8Prover::prove`] emits `zeta (16 bytes) || compact stage-2 proof`
//! ([`compact_proof_data`]), omitting the public-input vector (imposed by the verifier
//! from the statement) and the constants/sigmas oracle data (recomputed from the setup's
//! trusted polynomials). [`Fp8Verifier`] / [`Fp8VerifierCache`] accept *only* this format
//! ([`verify_compact_wrapped_proof`]) — one wire encoding, no dual-format ambiguity;
//! [`verify_fp8_block_wrapped`] stays as the typed in-process gateway for callers that
//! hold a full [`ProofWithPublicInputs`].

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
// The u32-limb hash packing helpers live in the frozen v2 clone since the mainline
// API no longer carries the legacy STARK plumbing.
use crate::v2::api::proof_utils::hash_to_u32_field_array;

/// The fp8 proof system's internal field, extension degree and hasher.
pub type F = GoldilocksField;
pub const D: usize = 2;
pub type C = PoseidonGoldilocksConfig;

impl MoEStatement {
    /// Maps each sampled `(inner, I_A)` onto a word index in the concatenated opened
    /// routing 64-byte blocks ([`MoEStatement::opened_routing_blocks`]).
    ///
    /// Sampled entry `i` lives at routing slot `o_w_prev + inner_indices[i]` and must
    /// equal public `i_a[i]`. Neighbor words in the same opened blocks stay prover
    /// witness, bound only by the hash to `HR`.
    pub(crate) fn routing_pins(&self, inner_indices: &[u32]) -> Vec<(usize, u32)> {
        assert_eq!(inner_indices.len(), self.i_a.len(), "inner/outer index lists must align");
        let words_per_block = pearl_blake3::BLAKE3_MSG_LEN / std::mem::size_of::<u32>();
        let blocks = self.opened_routing_blocks();
        inner_indices
            .iter()
            .zip(&self.i_a)
            .map(|(&inner, &outer)| {
                let slot = self.o_w_prev as usize + inner as usize;
                let strip = blocks
                    .binary_search(&((slot / words_per_block) as u32))
                    .expect("sampled slot's block is opened by construction");
                (words_per_block * strip + slot % words_per_block, outer)
            })
            .collect()
    }
}

/// Per-family prover setup and recursive-circuit cache.
///
/// Strictly prover-side. Verifiers use their own trusted setup — compiled offline
/// by `build_cache` ([`Fp8Verifier::generate`]) and shipped in [`Fp8VerifierCache`];
/// both sides derive the same circuits deterministically from public data, so
/// nothing needs to be handed over.
pub struct Fp8Prover {
    data: Fp8ProverData,
    circuits: Fp8CircuitCache,
}

impl Fp8Prover {
    /// Builds the prover's per-family data (the LUT precommitment and the
    /// compiled universal wrapper circuits).
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
        // Pay the wrapper-circuit compilation cost up front so `prove` runs hot.
        Fp8WrapperCircuits::with_cache(&job.system, &data.lut_cap, &mut circuits, timing)?;
        Ok(Self { data, circuits })
    }

    /// Proves one block and returns the published artifact as the same
    /// `(public_data, proof_data)` byte pair the v1/v2 certificates carry:
    /// the proven statement and the constant-size recursive proof in the
    /// *compact* encoding ([`compact_proof_data`]) — `zeta` preamble plus the
    /// stage-2 proof with the verifier-recomputable parts omitted.
    /// The unwrapped multi-STARK proof stays internal (it is not zero knowledge).
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
        // Re-derive the job for its batch layout (the zeta slot position); cheap next to proving.
        let job = Fp8Job::derive(&statement, proposed_header)?;
        let proof_data = compact_proof_data(&job.system, &wrapped)?;
        Ok((statement.to_bytes(), proof_data))
    }
}

impl Fp8Verifier {
    /// Generates the trusted setup: compiles the two-stage universal wrapper
    /// circuits against the committed LUT cap
    /// ([`crate::api::fp8::lut_caps`] — read from the committed caps file, so no
    /// LUT oracle is ever built). Expensive — run once, offline
    /// (`build_cache`); verification only ever reads the result from [`Fp8VerifierCache`].
    pub fn generate(params: &PublicParams, proposed_header: &IncompleteBlockHeader, timing: &mut TimingTree) -> Result<Self> {
        let job = Fp8Job::derive(params, proposed_header)?;
        Self::build_for_job(&job, committed_lut_cap(), timing)
    }

    /// Build tooling: [`Fp8Verifier::generate`] against an explicit LUT cap instead
    /// of the committed caps file. `build_cache` uses this to regenerate the caps
    /// file and the embeddable cache in one pass — the running binary still embeds
    /// the previous file, so the freshly derived caps must be passed in.
    pub fn generate_with_lut_cap(
        params: &PublicParams,
        proposed_header: &IncompleteBlockHeader,
        lut_cap: LutCap,
        timing: &mut TimingTree,
    ) -> Result<Self> {
        let job = Fp8Job::derive(params, proposed_header)?;
        Self::build_for_job(&job, lut_cap, timing)
    }

    /// The shared constructor core: compile the universal wrapper circuits
    /// against the given cap, keeping the cap alongside the circuit.
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

    /// The verification core on an already-derived statement (shared with
    /// [`Fp8VerifierCache`], which derives the job once for both the setup
    /// lookup and the verification).
    ///
    /// `proof_data` is the compact wire encoding of the stage-2 proof.
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
    /// The trusted setup for `job`'s device, from the cache alone. A missing
    /// setup is an error: this cache never compiles circuits (see the type docs).
    fn verifier_for_job(&self, job: &Fp8Job) -> Result<&Fp8Verifier> {
        let device = job.params.common().device;
        self.get(device)
            .ok_or_else(|| anyhow!("no cached fp8 verifier setup for {device:?}; regenerate fp8_cache.bin with build_cache"))
    }

    /// Verifies a published `(public_data, proof_data)` pair against the caller's
    /// expected block header, resolving the setup by the statement's device byte
    /// from the cache alone — a missing setup rejects the proof (never compiles; see
    /// the type docs). A proof of a different shape than its statement fails closed.
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

/// The canonical dense statement of the launch geometry — `m = n = 32`, `k = 2048`,
/// rank 32, per-axis pattern `[(4, Fold), (4, Blake)]`:
/// the statement `build_cache` compiles the universal setup from (any
/// envelope-legal statement yields the same circuits) and the
/// committed-LUT-cap tests pin. Its consumers only need an envelope-legal
/// statement, so hashes, header words and tile bases stay zero. Returns the
/// statement with a zeroed `ancestor_header` (so the statement binds the zero
/// header on both sides).
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

/// Decodes and validates a `public_data` blob (proof-controlled input) via the
/// [`PublicParams`] codec.
pub(crate) fn decode_statement(public_data: &[u8]) -> Result<PublicParams> {
    PublicParams::from_bytes(public_data)
}

/// The verifier's cached data: the setup-time Merkle cap of the committed LUT tables. The
/// cap is a network constant — a pure function of the LUT contents and
/// the consensus STARK config, independent of job geometry. A cap mismatch fails
/// closed (the proof cannot open a different cap).
#[derive(Clone, Debug)]
pub struct Fp8VerifierData {
    pub lut_cap: LutCap,
}

/// The prover's cached data: the consensus LUT cap plus the full LUT precommitment
/// (polynomials, LDEs, batched Merkle tree). Job-independent —
/// the batch order, the LUT positions and the consensus config are canonical constants —
/// so any job reuses it.
pub struct Fp8ProverData {
    pub lut_cap: LutCap,
    pub preprocessed: BatchStarkPreprocessedData<F, C, D>,
}

impl Fp8ProverData {
    /// Generates the prover's cached data (any job reuses it): derives the
    /// statement once to fix the batch layout, then commits to the LUT tables.
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

/// One fp8 batch proof (the twenty-two-table batched STARK argument with every table's
/// public inputs), plus its wire encoding.
#[derive(Clone, Debug)]
pub struct Fp8Proof(pub BatchStarkProofWithPublicInputs<F, C, D>);

impl Fp8Proof {
    /// Serializes the proof (and all its public inputs) for the wire.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(&self.0).context("serializing an fp8 proof")
    }

    /// Deserializes a wire proof. Structural validity only — [`verify_fp8_block`] is the
    /// judge of everything else.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self(bincode::deserialize(bytes).context("deserializing an fp8 proof")?))
    }
}

// ==================================================================================================
// The statement derivation
// ==================================================================================================

/// One job's derived proving/verifying context: the batch [`Fp8System`] plus the
/// key material the expected public inputs and the native epilogue need. Both sides
/// derive it from public data alone.
pub struct Fp8Job {
    /// The statement the job was derived from, and the proposed header it is
    /// verified against: everything downstream (expected public inputs,
    /// statement digests, the native epilogue) reads from this pair.
    params: PublicParams,
    /// `σ̂`: the caller's expected block header (`params.ancestor_header()` is `σ_Δ`).
    proposed_header: IncompleteBlockHeader,
    pub system: Fp8System<F, D>,
}

impl Fp8Job {
    /// Derives the statement of `params` (MoE or dense) from the proposed header. Fails
    /// (rather than panics) on any job outside the fp8 envelope.
    pub fn derive(params: &PublicParams, proposed_header: &IncompleteBlockHeader) -> Result<Self> {
        params.recheck_moe()?;
        let (compiled, _, _) = params.compile(proposed_header)?;
        let (h, w, k, r) = (compiled.h, compiled.w, compiled.k, compiled.r);

        // ---- Class (a) noise codes: bf16((E @ F)[elem]) on the bit-exact hardware MMA,
        // exactly the intermediate `noisy_quantize` computes. ----
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

        // ---- The five programs. ----
        // The MoE sampled-entry pins (empty for a dense job): the deployed compiler already
        // scheduled the routing tree's hotspot blocks from the same public data, so the pin
        // positions land exactly on the scheduled routing leaves
        // (`Blake3Program::validate` cross-checks).
        let (routing_pins, moe_schedule) = match (params.moe_statement(), &compiled.moe) {
            (Some(moe), Some(cm)) => (
                moe.routing_pins(&cm.inner_indices),
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
        // Consensus gate: every envelope-legal profile lands on the fixed FRI ladder
        // (`ladder_covers_the_envelope`); anything else is a bug upstream, not a job.
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

    /// `KEY_A` (`H_"key-A"(σ̂)`): the A-side/routing plane's tree key.
    fn key_a(&self) -> Hash256 {
        self.params.key_a(&self.proposed_header)
    }

    /// `KEY_B` (`H_"key-B"(σ_Δ)`): the B-side plane's tree key.
    fn key_b(&self) -> Hash256 {
        self.params.key_b()
    }

    /// The lottery key: `Subkey(noise_seedA, "pearl/v4/FP8/jackpot")` — the single
    /// derivation source lives on [`PublicParams::jackpot_key`].
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
        let p = &self.params;
        let u = |v: u32| usize::try_from(v).expect("geometry fits usize");
        ScaleProgram::new(u(p.h()), u(p.w()), u(p.common_dim()), u(p.rank()))
    }

    /// The InputQuant program (expected-public-input assembly and CTL geometry).
    fn input_quant(&self) -> InputQuantProgram {
        let p = &self.params;
        let u = |v: u32| usize::try_from(v).expect("geometry fits usize");
        InputQuantProgram {
            h: u(p.h()),
            w: u(p.w()),
            k: u(p.common_dim()),
            block_size: BLOCK_SIZE,
            r: u(p.rank()),
        }
    }

    /// The expected public inputs of every main table (canonical `Table` order): every slot
    /// is derived from the statement and the block's claims.
    fn expected_public_inputs(&self) -> [Vec<F>; NUM_TABLES] {
        let params = &self.params;
        let mut blake3 = vec![F::ZERO; NUM_BLAKE3_PUBLIC_INPUTS];
        let mut set = |base: usize, hash: &Hash256| {
            blake3[base..base + 8].copy_from_slice(&hash_to_u32_field_array(hash));
        };
        set(PI_KEY_A, &self.key_a());
        set(PI_KEY_B, &self.key_b());
        set(PI_JACKPOT_KEY, &self.jackpot_key());
        // The in-AIR commit folds must land on the job's public side digests.
        set(PI_HASH_A, &params.hash_a());
        set(PI_HASH_B, &params.hash_b());
        // MoE: the routing/offsets tree roots must equal the job's public commitments.
        // Dense: no such trees — the trace's slots stay zero.
        set(PI_HASH_ROUTING, &self.hash_routing().unwrap_or([0u8; 32]));
        set(PI_HASH_OFFSETS, &self.hash_offsets().unwrap_or([0u8; 32]));
        set(PI_HASH_JACKPOT, &params.hash_jackpot());

        let scale = self.scale().public_inputs::<F>().to_vec();
        // InputQuant: `2^Wl2` (the plaintext power of Scale's `Wl2` slot — the verifier pins
        // both; the wrapper only exposes the wires) and the geometry slots the CTL
        // expressions read.
        let iq = self.input_quant();
        let input_quant = iq.public_inputs::<F>().to_vec();
        // Tamed: the untamed threshold and allowance and the skip budget, pure functions of
        // the geometry.
        let tamed = TamedProgram {
            h: iq.h,
            w: iq.w,
            k: iq.k,
        }
        .public_inputs::<F>()
        .to_vec();
        [blake3, input_quant, scale, vec![], vec![], tamed]
    }

    /// The native epilogue shared by the batch and wrapped verification paths, run after
    /// the proof itself (and its public-input pinning) has been checked: the plain
    /// difficulty condition on the proven lottery digest, against `nbits`.
    /// (`HASH_A`/`HASH_B` need no epilogue — the in-AIR folds bind them directly.)
    ///
    /// `Ok(())` accepts; a failed difficulty check rejects.
    fn native_epilogue(&self, nbits: u32) -> Result<()> {
        let params = &self.params;
        // The winning condition on the proven lottery digest (HASH_JACKPOT was pinned to
        // params.hash_jackpot by the public-input check).
        check_jackpot_difficulty(&params.hash_jackpot(), nbits, params.h(), params.w(), params.common_dim())
    }
}

// ==================================================================================================
// Prove / verify entry points
// ==================================================================================================

/// Parses an fp8 plain proof (verifying every Merkle membership), proves the full fp8
/// batch statement, and returns the public params — with `hash_jackpot` set from the proven
/// lottery digest, like the v1 `prove_block` — alongside the proof.
///
/// `prover_data` must have been generated for this job's geometry
/// ([`Fp8ProverData::generate`]).
pub fn zk_prove_plain_proof_fp8(
    proposed_header: &IncompleteBlockHeader,
    plain_proof: &PlainProofV4,
    prover_data: &Fp8ProverData,
    timing: &mut TimingTree,
) -> Result<(PublicParams, Fp8Proof)> {
    let (public, _, proof) = prove_batch_statement(proposed_header, plain_proof, prover_data, timing)?;
    Ok((public, Fp8Proof(proof)))
}

/// [`zk_prove_plain_proof_fp8`], then the two-stage recursive wrap: the returned proof
/// is the constant-size stage-2 zero-knowledge plonky2 proof — the batch proof and the
/// stage-1 proof never leave the prover.
///
/// `prover_data` must come from [`Fp8ProverData::generate`]. `cache` collects the
/// compiled wrapper circuits: the first job compiles them, every later job reuses
/// them whatever its geometry.
pub fn zk_prove_plain_proof_fp8_wrapped(
    proposed_header: &IncompleteBlockHeader,
    plain_proof: &PlainProofV4,
    prover_data: &Fp8ProverData,
    cache: &mut Fp8CircuitCache,
    timing: &mut TimingTree,
) -> Result<(PublicParams, ProofWithPublicInputs<F, OuterC, D>)> {
    let (public, job, batch_proof) = prove_batch_statement(proposed_header, plain_proof, prover_data, timing)?;
    let circuits = Fp8WrapperCircuits::with_cache(&job.system, &prover_data.lut_cap, cache, timing)?;
    // Digest the live `public`, not the stale `job.params`: the Fiat-Shamir callback in
    // `prove_batch_statement` set the jackpot on `public`, so it (not the pre-jackpot
    // copy the job derived from) is what the batch proof baked into its known columns.
    let wrapped = circuits.prove(&job.system, &batch_proof, public.digest(proposed_header), timing)?;
    Ok((public, wrapped))
}

/// The shared proving core: parse the plain proof, derive the statement, prove the batch.
/// Returns the public params (with `hash_jackpot` set from the proven lottery digest), the
/// derived job (so wrapping callers need not re-derive it) and the batch proof.
fn prove_batch_statement(
    proposed_header: &IncompleteBlockHeader,
    plain_proof: &PlainProofV4,
    prover_data: &Fp8ProverData,
    timing: &mut TimingTree,
) -> Result<(PublicParams, Fp8Job, BatchStarkProofWithPublicInputs<F, C, D>)> {
    let (private, mut public) = plain_proof.parse_proof(proposed_header)?;
    let mut job = Fp8Job::derive(&public, proposed_header)?;

    // The cached precommitment must match this job's batch layout and the consensus cap.
    let expected_prep = job.system.preprocessed_verifier_data::<C>(&prover_data.lut_cap);
    ensure!(
        prover_data.preprocessed.columns_per_table == expected_prep.columns_per_table,
        "the cached LUT precommitment does not match this job's batch layout; \
         regenerate the prover data"
    );
    ensure!(
        prover_data.preprocessed.cap() == prover_data.lut_cap,
        "the cached LUT precommitment disagrees with the consensus cap"
    );

    // The opened routing hotspot blocks and the full offsets list (MoE; empty for dense
    // jobs) as u32 words, stream order — `parse_proof` already Merkle-verified them against
    // `moe.hash_routing`/`moe.hash_offsets` and checked the sampled entries against the
    // public outer indices.
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
    // Ship the lottery digest as the block's claim *before* Fiat-Shamir: `J` is in
    // `PublicParams::to_bytes()`, so `digest(proposed_header)` must see the generated jackpot.
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

/// Verifies one fp8 block proof end to end:
///
/// 1. derives the statement from `public_params` (rejecting non-fp8 or out-of-envelope
///    jobs; MoE and dense jobs both derive),
/// 2. pins **every** public input slot to the verifier's own recomputation —
///    `HASH_A`/`HASH_B` to the job's side digests, `HASH_JACKPOT` to the block's
///    shipped claim,
/// 3. runs the batched multi-STARK verification (constraints, class (a) openings against
///    the recomputed known columns, the consensus LUT cap, CTLs, one FRI argument),
/// 4. checks the plain difficulty condition on the proven lottery digest against `nbits`
///    (or `nbits_override`, e.g. a pool share target).
///
/// `Ok(())` accepts, `Err` rejects.
pub fn verify_fp8_block(
    public_params: &PublicParams,
    proposed_header: &IncompleteBlockHeader,
    proof: &Fp8Proof,
    verifier_data: &Fp8VerifierData,
    nbits_override: Option<u32>,
) -> Result<()> {
    let mut job = Fp8Job::derive(public_params, proposed_header)?;
    let nbits = nbits_override.unwrap_or(proposed_header.nbits);
    let expected = job.expected_public_inputs();
    let digest = job.params.digest(&job.proposed_header);
    job.system.bind_statement_digest(digest);
    job.system.verify::<C>(&proof.0, &expected, &verifier_data.lut_cap)?;
    job.native_epilogue(nbits)
}

/// Verifies one *wrapped* fp8 block proof end to end — the same statement derivation,
/// public-input pinning and native epilogue as [`verify_fp8_block`], with the batch
/// verification replaced by its in-circuit encoding:
///
/// 1. derives the statement from `public_params`,
/// 2. pins **every** slot — the per-table public inputs to the verifier's own
///    expectations, the known-column digest to the statement's, and the known-column
///    evaluations to the verifier's native recomputation at the proof's `zeta`
///    ([`verify_wrapped_proof`]),
/// 3. verifies the stage-2 zero-knowledge plonky2 proof against `circuit` — the consensus
///    stage-2 verifier data ([`Fp8WrapperCircuits::verifier_data`]), which transitively
///    pins the whole baked statement shape (batch layout, AIR constraint set, CTLs, the LUT
///    family cap),
/// 4. runs the native epilogue (the plain difficulty condition).
pub fn verify_fp8_block_wrapped(
    public_params: &PublicParams,
    proposed_header: &IncompleteBlockHeader,
    proof: &ProofWithPublicInputs<F, OuterC, D>,
    circuit: &VerifierCircuitData<F, OuterC, D>,
    nbits_override: Option<u32>,
) -> Result<()> {
    let job = Fp8Job::derive(public_params, proposed_header)?;
    let nbits = nbits_override.unwrap_or(proposed_header.nbits);
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

    /// The full wire-level round trip: parse the plain proof, prove the batch statement,
    /// serialize, verify against independently derived expectations, and reject tampering
    /// with the block's claims, the proof's public inputs, and the openings.
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

        // Tamper 2: bump the Scale geometry slot (`K`) — every public input is pinned to the
        // verifier's own statement-derived expectation, so a shifted slot must be rejected.
        let job = Fp8Job::derive(&public, &header).unwrap();
        let mut tampered = proof.clone();
        tampered.0.public_inputs[job.system.main_table_positions()[Table::Scale as usize]][K_PUBLIC_INPUT] += F::ONE;
        assert!(verify_fp8_block(&public, &header, &tampered, &verifier_data, None).is_err());

        // Tamper 3: any opening perturbation breaks the batched FRI argument.
        let mut tampered = proof.clone();
        tampered.0.proof.openings[0].local_values[0] += F::ONE.into();
        assert!(verify_fp8_block(&public, &header, &tampered, &verifier_data, None).is_err());
    }

    /// The padded-geometry wire-level round trip ([`fixture_job_asym`]: `h = 16 != w = 20`,
    /// `k = 2048` — a reference-legal job the pre-padding AIRs could not prove, since
    /// `h + w = 36` and `h*w = 320` are not powers of two). Exercises `derive` without the
    /// old envelope checks and every AIR's
    /// padding path (InputQuant per-side liveness, Scale and XorFold pad rows, Matmul
    /// phantom cells) through parse -> derive -> prove -> verify.
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

    /// The `k % 32` wire-level round trip ([`fixture_job_k32`]: `k = 2080`, committed rows
    /// that never tile into whole 64-byte Blake3 blocks — a job the wire layer rejected
    /// while it required block-aligned rows). Exercises the straddling-block schedule
    /// (cross-strip messages and `SplitLeaf`s of both orders) and the non-power-of-two
    /// row-mean scale derivation through parse -> derive -> prove -> verify.
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

    /// The compact wire's one prover-supplied slot: the expected-PI builder embeds a legal
    /// `zeta` at exactly the preamble-encoded slots and fails closed on a basefield `zeta`
    /// (the deployed v1/v2 rejection rule) — no proving required, pure layout.
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

        // A legal zeta lands at the layout's zeta slots, and the vector is a pure
        // function of (statement, zeta): same inputs, same imposed public inputs.
        let zeta = QuadraticExtension([F::from_canonical_u64(3), F::from_canonical_u64(41)]);
        let expected = expected_wrapper_public_inputs(&job.system, &expected_pis, digest, zeta).expect("legal zeta must build");
        let z = zeta_offset(&job.system);
        assert_eq!(&expected[z..z + D], &zeta.0, "zeta must land at the preamble's slots");
        let again = expected_wrapper_public_inputs(&job.system, &expected_pis, digest, zeta).expect("legal zeta must build");
        assert_eq!(expected, again, "the imposed vector must be deterministic");
    }

    /// Setup keys are universal (the D1 design): two jobs with different geometries
    /// and degree profiles share one key — the profile and geometry ride the
    /// wrapper's public inputs, pinned natively.
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

    /// The wrapped (published) round trip: prove the batch statement, wrap it in the
    /// two-stage universal recursive circuit (compiled into the cache), serialize, verify
    /// (public-input pinning + one plonky2 verification + the native epilogues); then the
    /// universal payoff — a second geometry of the same family through the same circuits
    /// and setup — and the wrapped rejection surface: a tampered block claim, a tampered
    /// claim slot, a tampered degree slot, a shifted zeta and a forged known-column
    /// evaluation must all fail closed.
    #[test]
    #[ignore = "exceeds the safe memory budget of an 8 GiB CI runner; run explicitly on a larger machine"]
    fn wrapped_api_roundtrip_and_tamper_rejection() {
        use crate::circuit::fp8::wrapper::{COMPACT_ZETA_PREAMBLE, degree_bits_offset, table_pis_offset, zeta_offset};

        // Surface the wrapper's gate/size logs under `RUST_LOG=info`.
        let _ = env_logger::builder().format_timestamp(None).try_init();

        let mut timing = TimingTree::default();
        let (header, plain) = fixture_job();
        let mut prover = Fp8Prover::setup_with_timing(&header, &plain, &mut timing).expect("fp8 setup");
        // Prove through the typed wrapped API (so the tamper section below can mutate
        // slots of the full proof) and compact it with the same encoder
        // [`Fp8Prover::prove`] publishes through.
        let (public, wrapped) =
            zk_prove_plain_proof_fp8_wrapped(&header, &plain, &prover.data, &mut prover.circuits, &mut timing)
                .expect("wrapped proving must succeed");
        let job = Fp8Job::derive(&public, &header).unwrap();
        let public_data = public.to_bytes();
        let proof_data = compact_proof_data(&job.system, &wrapped).expect("compact encode");

        // The verifier side generates its own trusted setup from the published statement
        // alone — nothing crosses over from the prover.
        let statement = decode_statement(&public_data).expect("public data must decode");
        let verifier = Fp8Verifier::generate(&statement, &header, &mut timing).expect("verifier-side setup");
        assert!(
            verifier.circuit.common.config.zero_knowledge,
            "the published stage must carry plonky2's ZK blinding"
        );

        // Independent-process wire round trip of the trusted setup; the published
        // artifact is already the `(public_data, proof_data)` byte pair.
        let verifier_bytes = verifier.to_bytes().expect("serialize verifier setup");
        let verifier = Fp8Verifier::from_bytes(&verifier_bytes).expect("deserialize verifier setup");

        verifier
            .verify_block(&header, &public_data, &proof_data)
            .expect("honest wrapped proof must verify");

        // The cache surface: register the setup, round-trip the cache bytes, and verify
        // through the cache (the embedded-`CACHE_DATA` path).
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

        // A setup missing from the cache fails closed: verification never compiles
        // circuits, so no proof can force the expensive setup build (DoS hardening).
        let err = Fp8VerifierCache::default()
            .verify_block(&header, &public_data, &proof_data)
            .expect_err("an uncached setup must reject, not compile");
        assert!(
            err.to_string().contains("no cached fp8 verifier setup"),
            "unexpected error: {err:#}"
        );

        // The universal (D1) payoff: a different geometry *and* degree profile
        // (`k = 2080`: non-power-of-two blocks, 65-row Matmul cells)
        // proves through the same prover without recompiling — the circuit cache still
        // holds exactly the one entry — and verifies against the same setup.
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

        // Tampering with the recursive proof's public inputs: the Scale geometry slots
        // (`K`, `WL2`), InputQuant's `2^WL2` and two of its CTL geometry slots (the B key
        // offset `h*k` and the A operand multiplicity `w`), a degree slot (the universal
        // circuit's height input, pinned to the statement's profile), a shifted zeta
        // (moves the native class (a) recompute), and a forged known-column evaluation
        // (the first slot after zeta).
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

        // The compact wire carries no public-input vector so changing a public input
        // should not change the encoding.
        let mut mutated = wrapped.clone();
        mutated.public_inputs[scale_pis + K_PUBLIC_INPUT] += F::ONE;
        assert_eq!(
            compact_proof_data(&job.system, &mutated).expect("compact encode"),
            proof_data,
            "imposed slots must not ride the compact wire"
        );
        // But zeta does ride the wire (the 16-byte preamble) and a shifted value is
        // rejected.
        for limb in 0..D {
            let mut tampered = wrapped.clone();
            tampered.public_inputs[zeta_offset(&job.system) + limb] += F::ONE;
            let bytes = compact_proof_data(&job.system, &tampered).expect("compact encode");
            assert!(
                verifier.verify_block(&header, &public_data, &bytes).is_err(),
                "a preamble zeta shifted in limb {limb} must be rejected"
            );
        }

        // Compact-wire malleability surface: a corrupted proof byte, a non-canonical
        // zeta limb (>= the field order) and a truncated body must all be rejected.
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

    /// A mid-size job through the full published pipeline — batch proof plus both recursion
    /// stages — at a geometry comparable to the existing recursion test jobs
    /// (`h = w = 16`, `k = 4096`: InputQuant at 2^16 rows, Blake3 and Matmul at 2^15),
    /// large enough that per-proof overheads stop dominating the measurement.
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

    /// The MoE wire-level round trip: parse the MoE plain proof (routing membership +
    /// sampled-entry checks), derive the MoE statement (routing pins, jackpot subkey,
    /// `HASH_ROUTING` binding), prove, verify — then the MoE-specific rejection surface:
    /// a different sampled outer index, a forged routing commitment, and a different
    /// expert index must all fail closed.
    #[test]
    fn moe_api_roundtrip_and_tamper_rejection() {
        let (header, plain) = fixture_job_moe();
        let mut timing = TimingTree::default();

        let (_, params) = plain.parse_proof(&header).expect("MoE fixture job must parse");
        let moe = params.moe_statement().expect("the fixture is MoE").clone();

        // The statement's MoE key material, bit-exact vs the labelled jackpot subkey.
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

        // The proof ships the routing tree root at HASH_ROUTING, bound to the public
        // commitment (the same root the plain proof's Merkle membership verified).
        let blake3_pis = &proof.0.public_inputs[job.system.main_table_positions()[0]];
        assert_eq!(
            &blake3_pis[PI_HASH_ROUTING..PI_HASH_ROUTING + 8],
            &hash_to_u32_field_array(&moe.hash_routing),
            "the proven routing root must be the job's routing commitment"
        );

        // Tamper 1: a different sampled outer index is a different pin schedule — the
        // verifier's class (a) recompute (known columns) diverges from the committed trace.
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

        // Tamper 3: a different expert index is a different statement (pin schedule, POW
        // key, hotspot schedule) — the proof must not transfer.
        let mut bad_expert = public.clone();
        bad_expert.moe_statement_mut().unwrap().w = 0;
        assert!(
            verify_fp8_block(&bad_expert, &header, &proof, &verifier_data, None).is_err(),
            "a proof must not verify under a different expert's statement"
        );
    }

    /// The size constants must match the actual encoding: a dense statement encodes to exactly
    /// [`PublicParams::WIRE_SIZE`] bytes, every statement length passes the
    /// [`PublicParams::is_valid_wire_size`] pre-filter, and the codec round-trips.
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
