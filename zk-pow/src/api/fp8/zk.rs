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
//! **Verifier setup resolution.** A verifier needs one [`Fp8Verifier`] per device: the stage-1
//! wrapper is the *universal* batch verifier (D1) — the degree profile and geometry are
//! public inputs, not circuit shape — so a single setup covers every envelope-legal job.
//! [`Fp8VerifierCache`] holds it keyed by the statement's device byte, preloaded
//! from [`embedded_cache::CACHE_DATA`](crate::api::fp8::embedded_cache::CACHE_DATA)
//! (built offline by `build_cache`; each device's LUT cap is derived and baked into
//! its cached circuit, so the verifier side never builds a LUT oracle).
//! Verification is lookup-only: a setup missing from the cache rejects the proof
//! rather than compiling it on demand, so no input can force an expensive
//! circuit build through the verify path.
//!
//! **Cached data.** [`Fp8Prover`] owns one setup per device: the full LUT precommitment
//! oracle (LDEs + batched Merkle tree — prover-only) and the compiled wrapper circuits.
//! Each setup is job-independent: the batch order, LUT positions and consensus config are
//! canonical constants. The first requested device is compiled by [`Fp8Prover::setup`];
//! another device is compiled lazily on its first proof.
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
//! (the block's shipped claim, difficulty-checked before proof verification), the geometry parameters
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
//! architecture. [`Fp8Prover`] compiles each device's universal wrapper circuits once and
//! reuses them afterwards. Wrapped verification applies the same statement derivation and
//! public jackpot check as the unwrapped path, with the batch-level checks replaced by the
//! wrapper's public-input pinning (every slot recomputed from the statement, the class (a)
//! columns natively re-evaluated at the
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
use hashbrown::HashMap;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::plonk::config::PoseidonGoldilocksConfig;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::util::timing::TimingTree;
use starky::batch_proof::BatchStarkProofWithPublicInputs;
use starky::batch_prover::BatchStarkPreprocessedData;

use crate::api::fp8::noise::compute_fp8_noise;
use crate::api::fp8::openings::stream_words;
use crate::api::fp8::plain_proof::PlainProofV4;
use crate::api::fp8::prequant::BLOCK_SIZE;
use crate::api::fp8::public_params::{
    CommonParams, Device, HashId, JackpotStatement, JobParams, MoEStatement, OperandParams, PublicParams, Quant,
};
use crate::api::fp8::utils::{MMA_GROUP_PRODUCTS, fp32_to_bf16_rne};
use crate::api::layout::{AxisPattern, DimType, lane_assignment};
use crate::api::primitives::{BlockHeader, Hash256, IncompleteBlockHeader, Sides};
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
use crate::circuit::fp8::scale_stark::stark::ScaleProgram;
use crate::circuit::fp8::unpredictability::budget;
#[cfg(test)]
use crate::circuit::fp8::wrapper::verify_wrapped_proof;
use crate::circuit::fp8::wrapper::{Fp8WrapperCircuits, OuterC, compact_proof_data, verify_compact_wrapped_proof};
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

/// Reusable prover setup for one device.
struct Fp8ProverSetup {
    device: Device,
    preprocessed: BatchStarkPreprocessedData<F, C, D>,
    circuits: Fp8WrapperCircuits,
}

impl Fp8ProverSetup {
    fn build(device: Device, timing: &mut TimingTree) -> Result<Self> {
        let params = sample_dense_statement_for_device(device)?;
        let job = Fp8Job::derive(&params, &IncompleteBlockHeader::zero())?;
        let preprocessed = job.system.preprocessed_data::<C>(timing);
        let circuits = Fp8WrapperCircuits::build(&job.system, &preprocessed.cap(), timing)?;
        Ok(Self {
            device,
            preprocessed,
            circuits,
        })
    }
}

/// Per-family prover. Strictly prover-side: it retains one setup per device and
/// reuses it across every supported geometry.
///
/// Verifiers use their own trusted setup — compiled offline by `build_cache`
/// ([`Fp8Verifier::generate`]) and shipped in [`Fp8VerifierCache`].
pub struct Fp8Prover {
    setups: HashMap<Device, Fp8ProverSetup>,
}

impl Fp8Prover {
    /// Builds and retains the first device's LUT precommitment and universal
    /// wrapper circuits. Other devices are built lazily by [`Self::prove`].
    pub fn setup(device: Device) -> Result<Self> {
        let mut timing = TimingTree::default();
        Self::setup_with_timing(device, &mut timing)
    }

    fn setup_with_timing(device: Device, timing: &mut TimingTree) -> Result<Self> {
        // Pay the LUT-preprocessing and wrapper-circuit compilation costs up front so
        // `prove` does not pay them.
        // TODO: Add `Fp8Prover::warmup_prove(&mut self, input: &PlainProofV4)` if
        // shape-specific first-proof warm-up is needed. It can prove once against
        // `IncompleteBlockHeader::zero()` and discard the result.
        let setup = Fp8ProverSetup::build(device, timing)?;
        Ok(Self {
            setups: HashMap::from([(device, setup)]),
        })
    }

    fn setup_for_device(&mut self, device: Device, timing: &mut TimingTree) -> Result<&mut Fp8ProverSetup> {
        if !self.setups.contains_key(&device) {
            self.setups.insert(device, Fp8ProverSetup::build(device, timing)?);
        }
        Ok(self.setups.get_mut(&device).expect("the device setup was just inserted"))
    }

    /// Builds and retains this device's LUT precommitment and universal wrapper
    /// circuits now, so a later [`Self::prove`] does not pay that setup cost.
    /// Calling this again for an already-cached device is a no-op.
    pub fn setup_device(&mut self, device: Device) -> Result<()> {
        let mut timing = TimingTree::default();
        self.setup_for_device(device, &mut timing).map(|_| ())
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
        let setup = self.setup_for_device(input.job.common.device, timing)?;
        let (statement, job, wrapped) = prove_wrapped_statement(setup, proposed_header, input, timing)?;
        let proof_data = compact_proof_data(&job.system, &wrapped)?;
        Ok((statement.to_bytes(), proof_data))
    }
}

impl Fp8Verifier {
    /// Generates the trusted setup: compiles the two-stage universal wrapper
    /// circuits against a freshly derived LUT cap. Expensive — run once, offline
    /// (`build_cache`); verification only ever reads the result from [`Fp8VerifierCache`].
    pub fn generate(params: &PublicParams, proposed_header: &IncompleteBlockHeader, timing: &mut TimingTree) -> Result<Self> {
        let job = Fp8Job::derive(params, proposed_header)?;
        let preprocessed = job.system.preprocessed_data::<C>(timing);
        let circuits = Fp8WrapperCircuits::build(&job.system, &preprocessed.cap(), timing)?;
        Ok(Self {
            circuit: circuits.verifier_data(),
            constants_sigmas_polynomials: circuits.constants_sigmas_polynomials(),
        })
    }

    /// Verifies a published `(public_data, proof_data)` pair against the
    /// caller's expected block header: `Ok(())` accepts, `Err` rejects.
    /// `ancestor_chain` links the statement's `σ_d` to `proposed_header`
    /// ([`JobParams::check_ancestry`]).
    pub fn verify_block(
        &self,
        proposed_header: &IncompleteBlockHeader,
        ancestor_chain: &[BlockHeader],
        public_data: &[u8],
        proof_data: &[u8],
    ) -> Result<()> {
        self.verify_share(
            proposed_header,
            ancestor_chain,
            public_data,
            proof_data,
            proposed_header.nbits,
        )
    }

    /// Verifies a pool share under an explicit share target.
    pub fn verify_share(
        &self,
        proposed_header: &IncompleteBlockHeader,
        ancestor_chain: &[BlockHeader],
        public_data: &[u8],
        proof_data: &[u8],
        share_nbits: u32,
    ) -> Result<()> {
        let statement = validate_public_statement(proposed_header, ancestor_chain, public_data, share_nbits)?;
        let job = Fp8Job::derive(&statement, proposed_header)?;
        self.verify_derived(&job, proof_data)
    }

    /// The verification core on an already-derived statement (shared with
    /// [`Fp8VerifierCache`], which resolves the setup before deriving the job).
    ///
    /// `proof_data` is the compact wire encoding of the stage-2 proof.
    fn verify_derived(&self, job: &Fp8Job, proof_data: &[u8]) -> Result<()> {
        let expected = job.expected_public_inputs();
        verify_compact_wrapped_proof(
            &job.system,
            &self.circuit,
            &self.constants_sigmas_polynomials,
            proof_data,
            &expected,
            job.params.digest(&job.proposed_header),
        )
    }
}

impl Fp8VerifierCache {
    /// The trusted setup for the decoded statement's device, from the cache alone. A missing
    /// setup is an error: this cache never compiles circuits (see the type docs).
    fn verifier_for_statement(&self, statement: &PublicParams) -> Result<&Fp8Verifier> {
        let device = statement.common().device;
        self.get(device)
            .ok_or_else(|| anyhow!("no cached fp8 verifier setup for {device:?}; regenerate fp8_cache.bin with build_cache"))
    }

    /// Verifies a published `(public_data, proof_data)` pair against the caller's
    /// expected block header, resolving the setup by the statement's device byte
    /// from the cache alone — a missing setup rejects the proof (never compiles; see
    /// the type docs). A proof of a different shape than its statement fails closed.
    /// `ancestor_chain` links the statement's `σ_d` to `proposed_header`
    /// ([`JobParams::check_ancestry`]).
    pub fn verify_block(
        &self,
        proposed_header: &IncompleteBlockHeader,
        ancestor_chain: &[BlockHeader],
        public_data: &[u8],
        proof_data: &[u8],
    ) -> Result<()> {
        self.verify_share(
            proposed_header,
            ancestor_chain,
            public_data,
            proof_data,
            proposed_header.nbits,
        )
    }

    /// Verifies a pool share under an explicit share target.
    pub fn verify_share(
        &self,
        proposed_header: &IncompleteBlockHeader,
        ancestor_chain: &[BlockHeader],
        public_data: &[u8],
        proof_data: &[u8],
        share_nbits: u32,
    ) -> Result<()> {
        let statement = validate_public_statement(proposed_header, ancestor_chain, public_data, share_nbits)?;
        let verifier = self.verifier_for_statement(&statement)?;
        let job = Fp8Job::derive(&statement, proposed_header)?;
        verifier.verify_derived(&job, proof_data)
    }
}

/// The canonical minimum-envelope dense statement — `m = n = 32`, `k = 1024`,
/// rank 32, per-axis pattern `[(4, Fold), (4, Blake)]`:
/// the statement `build_cache` compiles the universal setup from (any
/// envelope-legal statement yields the same circuits) and the
/// committed-LUT-cap tests pin. Its consumers only need an envelope-legal
/// statement, so hashes, header words and tile bases stay zero. Returns the
/// statement with a zeroed `ancestor_header` for setup derivation.
pub fn sample_dense_statement() -> Result<PublicParams> {
    sample_dense_statement_for_device(Device::B200)
}

/// [`sample_dense_statement`] with an explicit committed device, used to derive
/// each device's job-independent prover setup, LUT commitment and universal
/// verifier setup.
pub fn sample_dense_statement_for_device(device: Device) -> Result<PublicParams> {
    let pattern = AxisPattern::new(&[(4, DimType::Fold), (4, DimType::Blake)])?;
    let params = PublicParams::try_new(
        JobParams {
            ancestor_header: BlockHeader::zero(),
            common: CommonParams {
                k: PublicParams::MIN_K as u32,
                r: 32,
                quant: Quant::Fp8E4M3Prequant,
                device,
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

fn validate_public_statement(
    proposed_header: &IncompleteBlockHeader,
    ancestor_chain: &[BlockHeader],
    public_data: &[u8],
    nbits: u32,
) -> Result<PublicParams> {
    let statement = decode_statement(public_data)?;
    statement.job().check_ancestry(proposed_header, ancestor_chain)?;
    check_public_jackpot(&statement, nbits)?;
    Ok(statement)
}

/// One fp8 batch proof (the twenty-table batched STARK argument with every table's
/// public inputs), plus its wire encoding.
#[derive(Clone, Debug)]
#[cfg(test)]
struct Fp8Proof(BatchStarkProofWithPublicInputs<F, C, D>);

#[cfg(test)]
impl Fp8Proof {
    /// Serializes the proof (and all its public inputs) for the wire.
    fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(&self.0).context("serializing an fp8 proof")
    }

    /// Deserializes a wire proof. Structural validity only — [`verify_fp8_block`] is the
    /// judge of everything else.
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self(bincode::deserialize(bytes).context("deserializing an fp8 proof")?))
    }
}

// ==================================================================================================
// The statement derivation
// ==================================================================================================

/// One job's derived proving/verifying context: the batch [`Fp8System`] plus the
/// key material the expected public inputs need. Both sides
/// derive it from public data alone.
pub struct Fp8Job {
    /// The statement the job was derived from, and the proposed header it is
    /// verified against: everything downstream (expected public inputs,
    /// statement digests) reads from this pair.
    params: PublicParams,
    /// `σ̂`: the caller's expected block header (`params.ancestor_header()` is `σ_d`).
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
        let device = params.common().device;
        // Each output cell is an independent length-`r` dot product, so the output
        // columns split into `k / MMA_GROUP_PRODUCTS` blocks without touching any
        // cell's value: every block `e @ (32 x r)` runs the identical per-cell
        // kernel on both devices. One job set spans both sides.
        let noise_codes = |e: &[u8], f: &[u8], rows: usize| -> Result<Vec<u16>> {
            use plonky2_maybe_rayon::{MaybeIntoParIter, ParallelIterator};

            let block = MMA_GROUP_PRODUCTS;
            let blocks: Vec<Vec<f32>> = (0..k / block)
                .into_par_iter()
                .map(|b| device.matmul_fp8(e, &f[b * block * r..(b + 1) * block * r], None, rows, block, r))
                .collect::<Result<_>>()?;
            let mut codes = Vec::with_capacity(rows * k);
            for i in 0..rows {
                for b in 0..k / block {
                    let block_row = &blocks[b][i * block..(i + 1) * block];
                    codes.extend(block_row.iter().map(|&v| fp32_to_bf16_rne(v)));
                }
            }
            Ok(codes)
        };
        let (a_noise, b_noise) = plonky2_maybe_rayon::join(
            || noise_codes(&noise.a.e, &noise.a.f, h),
            || noise_codes(&noise.b.e, &noise.b.f, w),
        );
        let a_noise = a_noise.context("A-side noise codes")?;
        let b_noise = b_noise.context("B-side noise codes")?;

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
            device: params.common().device,
        };
        let scale = ScaleProgram::new_for_device(h, w, k, r, params.common().device);
        let xor_fold = XorFoldProgram {
            lanes: lane_assignment(&params.a().pattern, &params.b().pattern),
            skip_limit: budget(k, h, w),
        };

        let system = Fp8System::new(Fp8PublicData {
            blake3,
            input_quant: input_quant.clone(),
            scale: scale.clone(),
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

    /// `KEY_B` (`H_"key-B"(σ_d)`): the B-side plane's tree key.
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
        ScaleProgram::new_for_device(u(p.h()), u(p.w()), u(p.common_dim()), u(p.rank()), p.common().device)
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
            device: p.common().device,
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
        let xor_fold = vec![F::from_canonical_u64(budget(iq.k, iq.h, iq.w))];
        [blake3, input_quant, scale, vec![], xor_fold]
    }
}

/// Reject a losing public jackpot claim before deriving known columns or verifying
/// the proof. The proof must still bind `HASH_JACKPOT` to this exact statement
/// value; this check alone never authenticates the claim.
fn check_public_jackpot(params: &PublicParams, nbits: u32) -> Result<()> {
    check_jackpot_difficulty(&params.hash_jackpot(), nbits, params.h(), params.w(), params.common_dim())
}

// ==================================================================================================
// Prove / verify entry points
// ==================================================================================================

/// Parses an fp8 plain proof (verifying every Merkle membership), proves the full fp8
/// batch statement, and returns the public params — with `hash_jackpot` set from the proven
/// lottery digest, like the v1 `prove_block` — alongside the proof.
///
/// `setup` must match the statement's device.
///
/// Test-only thin wrapper over the same [`prove_batch_statement`] core production proves
/// through: the unwrapped proof is observable here for negative testing, while production
/// publishes only the wrapped compact form.
#[cfg(test)]
fn zk_prove_plain_proof_fp8(
    proposed_header: &IncompleteBlockHeader,
    plain_proof: &PlainProofV4,
    setup: &Fp8ProverSetup,
    timing: &mut TimingTree,
) -> Result<(PublicParams, Fp8Proof)> {
    let (public, _, proof) = prove_batch_statement(proposed_header, plain_proof, setup, timing)?;
    Ok((public, Fp8Proof(proof)))
}

/// The shared proving core: parse the plain proof, derive the statement, prove the batch.
/// Returns the public params (with `hash_jackpot` set from the proven lottery digest), the
/// derived job (so wrapping callers need not re-derive it) and the batch proof.
fn prove_batch_statement(
    proposed_header: &IncompleteBlockHeader,
    plain_proof: &PlainProofV4,
    setup: &Fp8ProverSetup,
    timing: &mut TimingTree,
) -> Result<(PublicParams, Fp8Job, BatchStarkProofWithPublicInputs<F, C, D>)> {
    let (private, mut public) = plain_proof.parse_proof(proposed_header)?;
    let mut job = Fp8Job::derive(&public, proposed_header)?;
    ensure!(
        setup.device == public.common().device,
        "the FP8 prover setup is for {:?}, but the statement commits {:?}",
        setup.device,
        public.common().device
    );

    // The cached precommitment must match this job's batch layout and the consensus cap.
    let lut_cap = setup.preprocessed.cap();
    let expected_prep = job.system.preprocessed_verifier_data::<C>(&lut_cap);
    ensure!(
        setup.preprocessed.columns_per_table == expected_prep.columns_per_table,
        "the cached LUT precommitment does not match this job's batch layout; \
         regenerate the prover data"
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
        &setup.preprocessed,
        timing,
    )?;

    Ok((public, job, proof))
}

/// The shared batch-prove-and-wrap core: parse the plain proof, prove the batch
/// statement, and wrap it in the two-stage recursive circuit. Both
/// [`Fp8Prover::prove`] and the wrapped round-trip tests prove through here so the
/// sequence cannot drift between production and test paths.
fn prove_wrapped_statement(
    setup: &Fp8ProverSetup,
    proposed_header: &IncompleteBlockHeader,
    input: &PlainProofV4,
    timing: &mut TimingTree,
) -> Result<(PublicParams, Fp8Job, ProofWithPublicInputs<F, OuterC, D>)> {
    let (statement, job, batch_proof) = prove_batch_statement(proposed_header, input, setup, timing)?;
    // Digest the live `statement`, not `job.params`: proving sets the jackpot on the
    // returned statement after the job was derived.
    let wrapped = setup
        .circuits
        .prove(&job.system, &batch_proof, statement.digest(proposed_header), timing)?;
    Ok((statement, job, wrapped))
}

/// Verifies one fp8 block proof end to end:
///
/// 1. checks the public jackpot claim against `nbits`, then derives the statement
///    from `public_params` (rejecting non-fp8 or out-of-envelope jobs; MoE and
///    dense jobs both derive),
/// 2. pins **every** public input slot to the verifier's own recomputation —
///    `HASH_A`/`HASH_B` to the job's side digests, `HASH_JACKPOT` to the block's
///    shipped claim,
/// 3. runs the batched multi-STARK verification (constraints, class (a) openings against
///    the recomputed known columns, the consensus LUT cap, CTLs, one FRI argument).
///
/// The public check uses `nbits_override` for pool shares; the proof then binds its
/// lottery digest to the already-checked statement claim.
///
/// `Ok(())` accepts, `Err` rejects.
#[cfg(test)]
fn verify_fp8_block(
    public_params: &PublicParams,
    proposed_header: &IncompleteBlockHeader,
    proof: &Fp8Proof,
    setup: &Fp8ProverSetup,
    nbits_override: Option<u32>,
) -> Result<()> {
    let nbits = nbits_override.unwrap_or(proposed_header.nbits);
    check_public_jackpot(public_params, nbits)?;
    let mut job = Fp8Job::derive(public_params, proposed_header)?;
    ensure!(
        setup.device == public_params.common().device,
        "the FP8 prover setup is for {:?}, but the statement commits {:?}",
        setup.device,
        public_params.common().device
    );
    let expected = job.expected_public_inputs();
    let digest = job.params.digest(&job.proposed_header);
    job.system.bind_statement_digest(digest);
    job.system.verify::<C>(&proof.0, &expected, &setup.preprocessed.cap())
}

/// Verifies one *wrapped* fp8 block proof end to end — the same statement derivation,
/// public-input pinning and early jackpot check as [`verify_fp8_block`], with the batch
/// verification replaced by its in-circuit encoding:
///
/// 1. checks the public jackpot claim, then derives the statement from `public_params`,
/// 2. pins **every** slot — the per-table public inputs to the verifier's own
///    expectations, the known-column digest to the statement's, and the known-column
///    evaluations to the verifier's native recomputation at the proof's `zeta`
///    ([`verify_wrapped_proof`]),
/// 3. verifies the stage-2 zero-knowledge plonky2 proof against `circuit` — the consensus
///    stage-2 verifier data ([`Fp8WrapperCircuits::verifier_data`]), which transitively
///    pins the whole baked statement shape (batch layout, AIR constraint set, CTLs, the LUT
///    family cap).
#[cfg(test)]
fn verify_fp8_block_wrapped(
    public_params: &PublicParams,
    proposed_header: &IncompleteBlockHeader,
    proof: &ProofWithPublicInputs<F, OuterC, D>,
    verifier: &Fp8Verifier,
    nbits_override: Option<u32>,
) -> Result<()> {
    let nbits = nbits_override.unwrap_or(proposed_header.nbits);
    check_public_jackpot(public_params, nbits)?;
    let job = Fp8Job::derive(public_params, proposed_header)?;
    let expected = job.expected_public_inputs();
    verify_wrapped_proof(
        &job.system,
        &verifier.circuit,
        proof,
        &expected,
        job.params.digest(&job.proposed_header),
    )
}

/// A 32-byte hash as 8 little-endian u32 words (the STARK witness key encoding).
fn hash_words(hash: &Hash256) -> [u32; 8] {
    core::array::from_fn(|i| u32::from_le_bytes(hash[4 * i..4 * i + 4].try_into().unwrap()))
}

#[cfg(test)]
mod tests {

    use plonky2::field::types::Field;

    use super::*;
    use crate::circuit::fp8::consistency::{fixture_job, fixture_job_k32, fixture_job_moe};

    // The fixtures use an easy `nbits` that saturates the difficulty bound, so the plain
    // winning condition `hash_jackpot <= bound` is deterministic on pseudo-random data.

    mod proof_pipeline_tests;

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

    #[test]
    fn published_statement_rejects_an_unknown_device_byte() {
        let mut bytes = sample_dense_statement().unwrap().to_bytes();
        const DEVICE_OFFSET: usize = BlockHeader::SERIALIZED_SIZE + 4 + 4 + 2 + 1;
        bytes[DEVICE_OFFSET] = 2;
        let error = decode_statement(&bytes).expect_err("unknown device byte must reject");
        assert!(error.to_string().contains("Device discriminant"), "{error}");
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
        let key = params.common().device;

        // A different geometry (k = 2080 vs 2048, hence a different degree profile too)
        // resolves to the same universal setup.
        let (header_k32, plain_k32) = fixture_job_k32();
        let (_, params_k32) = plain_k32.parse_proof(&header_k32).expect("k32 fixture must parse");
        assert_ne!(
            [params.h(), params.w(), params.common_dim()],
            [params_k32.h(), params_k32.w(), params_k32.common_dim()],
            "the fixtures must differ in geometry for this test to bite"
        );
        assert_eq!(key, params_k32.common().device, "different geometry: one universal setup");

        let params_min = sample_dense_statement().expect("minimum-envelope statement");
        let job_min =
            Fp8Job::derive(&params_min, &IncompleteBlockHeader::zero()).expect("minimum-envelope statement must derive");
        assert!(
            job_min.system.degree_bits().contains(&13),
            "the minimum-envelope Matmul table must exercise the new 2^13 rung"
        );
        assert_eq!(key, job_min.params.common().device, "minimum geometry: one universal setup");
    }

    #[test]
    fn missing_verifier_cache_rejects_before_proof_decoding() {
        let statement = sample_dense_statement().unwrap();
        let proposed = IncompleteBlockHeader {
            prev_block: statement.ancestor_header().block_hash(),
            ..IncompleteBlockHeader::zero()
        };
        let error = Fp8VerifierCache::default()
            .verify_block(&proposed, &[], &statement.to_bytes(), &[])
            .unwrap_err();
        assert!(error.to_string().contains("no cached fp8 verifier setup"), "{error:#}");
    }

    #[test]
    fn verifier_cache_authenticates_the_ancestor_before_the_setup_lookup() {
        let statement = sample_dense_statement().unwrap();
        let error = Fp8VerifierCache::default()
            .verify_block(&IncompleteBlockHeader::zero(), &[], &statement.to_bytes(), &[])
            .unwrap_err();
        assert!(error.to_string().contains("depth 1 does not connect"), "{error:#}");
    }

    #[test]
    fn losing_public_jackpot_rejects_before_job_derivation_or_proof_decoding() {
        let mut statement = sample_dense_statement().unwrap();
        statement.set_hash_jackpot([u8::MAX; 32]);
        let proposed = IncompleteBlockHeader {
            prev_block: statement.ancestor_header().block_hash(),
            nbits: 0x1d00ffff,
            ..IncompleteBlockHeader::zero()
        };
        // No setup and a malformed proof body: the public lottery claim must
        // reject before either the expensive job derivation or proof decoding.
        let error = Fp8VerifierCache::default()
            .verify_block(&proposed, &[], &statement.to_bytes(), &[0])
            .unwrap_err();
        assert!(error.to_string().contains("Jackpot condition not satisfied"), "{error:#}");

        // A pool share must use its stricter override even when the block header
        // itself has an easy, saturating target.
        let easy_header = IncompleteBlockHeader {
            nbits: 0x207fffff,
            ..proposed
        };
        let error = Fp8VerifierCache::default()
            .verify_share(&easy_header, &[], &statement.to_bytes(), &[0], 0x1d00ffff)
            .unwrap_err();
        assert!(error.to_string().contains("Jackpot condition not satisfied"), "{error:#}");
    }

    #[test]
    fn committed_device_selects_the_matmul_air_and_lut_inventory() {
        use crate::circuit::fp8::luts::LutTable;
        use crate::circuit::fp8::matmul_b200_stark::columns::NUM_MATMUL_B200_COLUMNS;
        use crate::circuit::fp8::matmul_h100::columns::NUM_MATMUL_COLUMNS;

        let h100 = sample_dense_statement_for_device(Device::H100).unwrap();
        let b200 = sample_dense_statement_for_device(Device::B200).unwrap();
        let h100 = Fp8Job::derive(&h100, &IncompleteBlockHeader::zero()).unwrap();
        let b200 = Fp8Job::derive(&b200, &IncompleteBlockHeader::zero()).unwrap();
        assert_eq!(h100.system.matmul_num_columns(), NUM_MATMUL_COLUMNS);
        assert_eq!(b200.system.matmul_num_columns(), NUM_MATMUL_B200_COLUMNS);
        assert!(h100.system.committed_lut_tables().contains(&LutTable::ProdAlign15));
        assert!(b200.system.committed_lut_tables().contains(&LutTable::B200Align));
    }

    #[test]
    #[ignore = "fixture generator: expensive wrapped proof; run with task generate:fp8-fixture"]
    fn regenerate_go_fixture() {
        proof_pipeline_tests::regenerate_go_fixture();
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
