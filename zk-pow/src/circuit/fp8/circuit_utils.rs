//! FP8 recursion configuration, trusted verifier setup, and cache serialization.
//!
//! The API layer implements job derivation and verification on these types. The cache
//! retains FP8's device keys, canonical bincode format and read-only lookup semantics;
//! it does not use the legacy v2 per-degree cache format.

use anyhow::{Context, Result, ensure};
use bincode::Options;
use hashbrown::HashMap;
use plonky2::field::cosets::get_unique_coset_shifts;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::fri::{FriConfig, reduction_strategies::FriReductionStrategy};
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData, VerifierCircuitData};
use plonky2::util::serialization::{Buffer, DefaultGateSerializer, Read, Remaining};
use serde::{Deserialize, Serialize};

use super::wrapper::{D, F, OuterC};
use crate::api::fp8::lut_caps::LutCap;
use crate::api::fp8::public_params::{Device, PublicParams};
use crate::ensure_eq;

// Preserve the recursion security target previously imported from the frozen v2 stack.
const SECURITY_BITS: usize = 120;

/// Calculate number of query rounds for FRI
pub fn num_query_rounds(security_bits: usize, pow_bits: usize, rate_bits: usize) -> usize {
    security_bits.saturating_sub(pow_bits).div_ceil(rate_bits)
}

/// Build a recursion circuit config with the given parameters
pub fn build_recursion_config(rate_bits: usize, pow_bits: usize, stage: usize, is_zk: bool) -> CircuitConfig {
    debug_assert!(rate_bits >= 3);
    CircuitConfig {
        num_wires: 135,
        num_routed_wires: if stage == 2 { 40 } else { 37 },
        num_constants: 2,
        use_base_arithmetic_gate: true,
        security_bits: SECURITY_BITS,
        num_challenges: 3,
        zero_knowledge: is_zk,
        max_quotient_degree_factor: 8,
        fri_config: FriConfig {
            rate_bits,
            cap_height: 5,
            proof_of_work_bits: pow_bits as u32,
            reduction_strategy: FriReductionStrategy::ConstantArityBits(3, 7),
            num_query_rounds: num_query_rounds(SECURITY_BITS, pow_bits, rate_bits),
        },
    }
}

/// Trusted verifier setup: the universal wrapper's stage-2 verifier data
/// plus the committed LUT cap, plus the stage-2 constants/sigmas polynomial
/// coefficients the compact proof encoding omits
/// ([`super::wrapper::verify_compact_wrapped_proof`] recomputes the omitted oracle data from
/// them — same trust class as the verifier data itself, exactly the deployed
/// v2 `VerifierCircuitWithPolynomials` bundle). Covers every envelope-legal
/// job; obtain one from [`Fp8Verifier::generate`] or [`Fp8Verifier::from_bytes`].
#[derive(Clone, Debug)]
pub struct Fp8Verifier {
    pub(crate) lut_cap: LutCap,
    pub(crate) circuit: VerifierCircuitData<F, OuterC, D>,
    pub(crate) constants_sigmas_polynomials: Vec<PolynomialCoeffs<F>>,
}

impl Fp8Verifier {
    /// Serializes trusted verifier setup for distribution to independent
    /// verifier processes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let circuit = self
            .circuit
            .to_bytes(&DefaultGateSerializer)
            .map_err(|error| anyhow::anyhow!("serializing the recursive fp8 verifier circuit: {error:?}"))?;
        let wire = Fp8VerifierWire {
            lut_cap: self
                .lut_cap
                .0
                .iter()
                .map(|hash| hash.elements.map(|element| element.to_canonical_u64()))
                .collect(),
            circuit,
            constants_sigmas_polynomials: serialize_polynomials(&self.constants_sigmas_polynomials, &self.circuit.common),
        };
        fp8_wire_options().serialize(&wire).context("serializing fp8 verifier setup")
    }

    /// Loads verifier setup previously produced by [`Fp8Verifier::to_bytes`].
    /// The bytes are consensus/trusted-setup data, not proof-controlled input.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let wire: Fp8VerifierWire = fp8_wire_options()
            .deserialize(bytes)
            .context("deserializing fp8 verifier setup")?;
        ensure!(!wire.lut_cap.is_empty(), "fp8 verifier LUT cap cannot be empty");
        ensure!(
            wire.lut_cap.iter().flatten().all(|&element| element < F::ORDER),
            "fp8 verifier LUT cap contains a non-canonical field element"
        );
        let lut_cap = MerkleCap(
            wire.lut_cap
                .into_iter()
                .map(|elements| HashOut {
                    elements: elements.map(F::from_canonical_u64),
                })
                .collect(),
        );
        let circuit = VerifierCircuitData::from_bytes(wire.circuit, &DefaultGateSerializer)
            .map_err(|error| anyhow::anyhow!("deserializing the recursive fp8 verifier circuit: {error}"))?;
        ensure!(
            circuit.common.config.zero_knowledge,
            "the fp8 verifier setup must contain the ZK wrapper stage"
        );
        // The codec pins the polynomial count and degree to `circuit.common` (and rejects
        // trailing bytes), so a shape mismatch with the circuit fails closed here.
        let constants_sigmas_polynomials = deserialize_polynomials(&wire.constants_sigmas_polynomials, &circuit.common)
            .context("deserializing the fp8 verifier constants/sigmas polynomials")?;
        Ok(Self {
            lut_cap,
            circuit,
            constants_sigmas_polynomials,
        })
    }
}

/// A read-only store of verifier setups keyed by the statement's device byte. The
/// stage-1 wrapper is the *universal* batch verifier: one compiled circuit covers
/// every envelope-legal degree profile and geometry, so the whole cache holds a single
/// setup. Deployments preload
/// [`embedded_cache::CACHE_DATA`](crate::api::fp8::embedded_cache::CACHE_DATA), built
/// offline by `build_cache` ([`Fp8Verifier::generate`] + [`Fp8VerifierCache::insert`]).
///
/// Verification never compiles circuits: a setup missing from the cache rejects the
/// proof. An on-demand fallback would let anyone force the expensive setup
/// build through the verify path (denial of service); a stale or incomplete cache is a
/// deployment error instead, surfaced by the returned message.
#[derive(Default)]
pub struct Fp8VerifierCache {
    verifiers: HashMap<Fp8VerifierKey, Fp8Verifier>,
}

/// Everything that selects one trusted setup: the statement's device byte.
///
/// Nothing else is key material (the D1 universal design): the degree profile rides the
/// wrapper's degree public inputs and the geometry rides the `K`/`WL2`/`2^WL2` public
/// inputs and the known columns, all pinned natively by the gateway; the AIR identities,
/// the CTL set and the FRI ladder are consensus constants.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Fp8VerifierKey {
    hardware: u8,
}

impl Fp8VerifierKey {
    pub(crate) fn new(hardware: Device) -> Self {
        Self {
            hardware: hardware as u8,
        }
    }
}

impl Fp8VerifierCache {
    /// Loads a cache previously produced by [`Fp8VerifierCache::to_bytes`] (consensus /
    /// trusted-setup data, not proof-controlled input). An empty blob — the embedded
    /// default when no cache has been built — loads as an empty cache.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }
        let wire: Fp8VerifierCacheWire = fp8_wire_options()
            .deserialize(bytes)
            .context("deserializing the fp8 verifier cache")?;
        let mut verifiers = HashMap::new();
        for entry in wire.entries {
            let key = Fp8VerifierKey {
                hardware: entry.hardware,
            };
            ensure!(
                verifiers.insert(key, Fp8Verifier::from_bytes(&entry.verifier)?).is_none(),
                "duplicate fp8 verifier cache entry"
            );
        }
        Ok(Self { verifiers })
    }

    /// Serializes the cache (canonical entry order, so equal caches share bytes).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut entries = self
            .verifiers
            .iter()
            .map(|(key, verifier)| {
                Ok(Fp8VerifierCacheEntryWire {
                    hardware: key.hardware,
                    verifier: verifier.to_bytes()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.hardware);
        fp8_wire_options()
            .serialize(&Fp8VerifierCacheWire { entries })
            .context("serializing the fp8 verifier cache")
    }

    /// Number of cached setups.
    pub fn len(&self) -> usize {
        self.verifiers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.verifiers.is_empty()
    }

    /// Registers a pre-built verifier setup under `params`' device byte
    /// (build tooling: assembles an embeddable cache without regenerating setups
    /// already compiled elsewhere). Replaces any previous setup of the same device.
    pub fn insert(&mut self, params: &PublicParams, verifier: Fp8Verifier) {
        self.verifiers.insert(Fp8VerifierKey::new(params.common().device), verifier);
    }

    /// Looks up a pre-built setup; never compiles circuits on a cache miss.
    pub(crate) fn get(&self, hardware: Device) -> Option<&Fp8Verifier> {
        self.verifiers.get(&Fp8VerifierKey::new(hardware))
    }
}

fn fp8_wire_options() -> impl Options {
    bincode::options().with_fixint_encoding().reject_trailing_bytes()
}

/// Serialized verifier-circuit bytes plus the LUT cap they were compiled against, plus
/// the circuit's constants/sigmas polynomials.
#[derive(Serialize, Deserialize)]
struct Fp8VerifierWire {
    lut_cap: Vec<[u64; 4]>,
    circuit: Vec<u8>,
    constants_sigmas_polynomials: Vec<u8>,
}

/// Wire form of [`Fp8VerifierCache`]: the setups with their device-byte keys, in
/// canonical order. The blob is embedded in the binary that reads it (`fp8_cache.bin`),
/// so there is no cross-version exchange to tag: a stale file fails deserialization or
/// rejects proofs, both closed.
#[derive(Serialize, Deserialize)]
struct Fp8VerifierCacheWire {
    entries: Vec<Fp8VerifierCacheEntryWire>,
}

#[derive(Serialize, Deserialize)]
struct Fp8VerifierCacheEntryWire {
    hardware: u8,
    verifier: Vec<u8>,
}

// Polynomial encoding is part of the existing FP8 verifier/cache wire format.

fn bytes_for_max_value(max_val: usize) -> usize {
    if max_val == 0 {
        return 1;
    }
    let bits = usize::BITS - max_val.leading_zeros();
    bits.div_ceil(8) as usize
}

fn write_tight_le(buf: &mut Vec<u8>, val: usize, num_bytes: usize) {
    assert!(val < (1 << (num_bytes * 8)), "value overflows tight encoding");
    buf.extend_from_slice(&val.to_le_bytes()[..num_bytes]);
}

/// Precomputed coset geometry used by both the polynomial serializer and deserializer.
struct CosetLayout {
    bytes_per_index: usize,
    k_is: Vec<GoldilocksField>,
    subgroup: Vec<GoldilocksField>,
}

impl CosetLayout {
    fn new(common_data: &CommonCircuitData<GoldilocksField, 2>) -> Self {
        let num_routed_wires = common_data.config.num_routed_wires;
        let degree = common_data.degree();
        Self {
            bytes_per_index: bytes_for_max_value(num_routed_wires * degree - 1),
            k_is: get_unique_coset_shifts(degree, num_routed_wires),
            subgroup: GoldilocksField::two_adic_subgroup(common_data.degree_bits()),
        }
    }
}

/// Serialize constants_sigmas polynomials in compact form.
/// Constants are stored as u64 evaluations; sigmas as tightly packed permutation indices.
fn serialize_polynomials(
    polys: &[PolynomialCoeffs<GoldilocksField>],
    common_data: &CommonCircuitData<GoldilocksField, 2>,
) -> Vec<u8> {
    let num_constants = common_data.num_constants;
    let num_routed_wires = common_data.config.num_routed_wires;
    let degree = common_data.degree();
    let layout = CosetLayout::new(common_data);

    // Build reverse lookup: field element -> flat index
    let mut reverse_map: HashMap<GoldilocksField, usize> = HashMap::with_capacity(num_routed_wires * degree);
    for col in 0..num_routed_wires {
        for row in 0..degree {
            let val = layout.k_is[col] * layout.subgroup[row];
            reverse_map.insert(val, col * degree + row);
        }
    }

    let mut buf = Vec::new();

    // Constants: FFT to get evaluations, store as u64
    for poly in &polys[..num_constants] {
        let evals = poly.clone().fft();
        for &v in &evals.values {
            buf.extend_from_slice(&v.to_canonical_u64().to_le_bytes());
        }
    }

    // Sigmas: FFT to get evaluations, map to tight indices
    for poly in &polys[num_constants..num_constants + num_routed_wires] {
        let evals = poly.clone().fft();
        for &v in &evals.values {
            let idx = reverse_map[&v];
            write_tight_le(&mut buf, idx, layout.bytes_per_index);
        }
    }

    buf
}

/// Deserialize constants_sigmas polynomials from compact form.
/// Reconstructs field elements from indices and runs iFFT to get coefficients.
fn deserialize_polynomials(
    data: &[u8],
    common_data: &CommonCircuitData<GoldilocksField, 2>,
) -> Result<Vec<PolynomialCoeffs<GoldilocksField>>> {
    let num_constants = common_data.num_constants;
    let num_routed_wires = common_data.config.num_routed_wires;
    let degree = common_data.degree();
    let layout = CosetLayout::new(common_data);

    let mut reader = Buffer::new(data);
    let mut polys = Vec::with_capacity(num_constants + num_routed_wires);

    for _ in 0..num_constants {
        let mut values = Vec::with_capacity(degree);
        for _ in 0..degree {
            values.push(
                reader
                    .read_field::<GoldilocksField>()
                    .map_err(|_| anyhow::anyhow!("invalid field element in polynomial data"))?,
            );
        }
        polys.push(PolynomialValues::new(values).ifft());
    }

    for _ in 0..num_routed_wires {
        let mut values = Vec::with_capacity(degree);
        for _ in 0..degree {
            let idx = reader
                .read_uint_le(layout.bytes_per_index)
                .map_err(|_| anyhow::anyhow!("unexpected end of polynomial data"))?;
            ensure!(idx / degree < layout.k_is.len(), "sigma permutation index out of bounds");
            values.push(layout.k_is[idx / degree] * layout.subgroup[idx % degree]);
        }
        polys.push(PolynomialValues::new(values).ifft());
    }

    ensure_eq!(reader.remaining(), 0, "unexpected trailing bytes in polynomial data");

    Ok(polys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fp8::lut_caps::committed_lut_cap;
    use crate::api::fp8::zk::sample_dense_statement;

    // A small real circuit exercises setup serialization and compact verification without
    // the full FP8 prover's memory requirements. The reduced FRI parameters are test-only.
    fn cache_test_circuit() -> (
        plonky2::plonk::circuit_data::CircuitData<F, OuterC, D>,
        plonky2::iop::target::Target,
    ) {
        use plonky2::fri::reduction_strategies::FriReductionStrategy;
        use plonky2::plonk::circuit_builder::CircuitBuilder;

        // Keep the golden codec fixture independent of later production parameter tuning.
        let config = CircuitConfig {
            num_wires: 135,
            num_routed_wires: 40,
            num_constants: 2,
            use_base_arithmetic_gate: true,
            security_bits: 6,
            num_challenges: 3,
            zero_knowledge: true,
            max_quotient_degree_factor: 8,
            fri_config: FriConfig {
                rate_bits: 3,
                cap_height: 2,
                proof_of_work_bits: 0,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(2, 2),
                num_query_rounds: 2,
            },
        };
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let x = builder.add_virtual_target();
        let copied_x = builder.add_virtual_target();
        builder.connect(x, copied_x);
        let square = builder.mul(x, copied_x);
        builder.register_public_input(square);
        (builder.build::<OuterC>(), x)
    }

    #[test]
    fn verifier_cache_wire_compatibility() -> Result<()> {
        use plonky2::iop::witness::{PartialWitness, WitnessWrite};
        use plonky2::plonk::proof::CompactProofWithPublicInputs;

        let (data, x) = cache_test_circuit();
        let verifier = Fp8Verifier {
            lut_cap: committed_lut_cap(),
            circuit: data.verifier_data(),
            constants_sigmas_polynomials: data.prover_only.constants_sigmas_commitment.polynomials.clone(),
        };
        // Fingerprints captured with the original codecs in api::fp8::zk, before extraction.
        let verifier_bytes = verifier.to_bytes()?;
        let mut cache = Fp8VerifierCache::default();
        let statement = sample_dense_statement()?;
        cache.insert(&statement, verifier);
        let cache_bytes = cache.to_bytes()?;
        assert_eq!(
            blake3::hash(&verifier_bytes).to_hex().as_str(),
            "59e35386018bcd923128056fdaf3c0453f39a362eab1bec8bed08f24a4adbea3"
        );
        assert_eq!(
            blake3::hash(&cache_bytes).to_hex().as_str(),
            "c8cb12508c528741112265987ff9d0942fa1b3340efaea633f917610dbf22674"
        );

        let loaded = Fp8VerifierCache::from_bytes(&cache_bytes)?;
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.to_bytes()?, cache_bytes);
        let loaded_verifier = &loaded.verifiers[&Fp8VerifierKey::new(Device::B200)];
        assert_eq!(loaded_verifier.to_bytes()?, verifier_bytes);

        let mut witness = PartialWitness::new();
        witness.set_target(x, F::from_canonical_u64(3))?;
        let proof = data.prove(witness)?;
        let public_inputs = proof.public_inputs.clone();
        let compact: CompactProofWithPublicInputs<F, OuterC, D> = proof.into();
        let bytes = compact.to_proof_bytes();
        CompactProofWithPublicInputs::<F, OuterC, D>::from_bytes(&bytes, public_inputs, &loaded_verifier.circuit.common)
            .map_err(|e| anyhow::anyhow!("compact decode: {e:?}"))?
            .verify(
                &loaded_verifier.circuit.verifier_only,
                &loaded_verifier.circuit.common,
                &loaded_verifier.constants_sigmas_polynomials,
            )?;
        Ok(())
    }

    #[test]
    fn verifier_cache_order_and_rejection_rules() -> Result<()> {
        assert!(Fp8VerifierCache::from_bytes(&[])?.is_empty());
        assert_eq!(Fp8VerifierCache::default().to_bytes()?, vec![0; 8]);
        assert!(Fp8VerifierCache::from_bytes(&[0; 7]).is_err());
        assert!(Fp8VerifierCache::from_bytes(&[0; 9]).is_err());

        let (data, _) = cache_test_circuit();
        let verifier = Fp8Verifier {
            lut_cap: committed_lut_cap(),
            circuit: data.verifier_data(),
            constants_sigmas_polynomials: data.prover_only.constants_sigmas_commitment.polynomials.clone(),
        };
        let verifier_bytes = verifier.to_bytes()?;
        let statement = sample_dense_statement()?;
        let mut cache = Fp8VerifierCache::default();
        cache.insert(&statement, verifier.clone());
        cache.insert(&statement, verifier.clone());
        assert_eq!(cache.len(), 1, "insertion replaces the existing device setup");

        // Loading preserves arbitrary device-byte keys, as the previous codec did.
        // Ordering is canonical regardless of map insertion order.
        let mut reverse_cache = Fp8VerifierCache::default();
        for hardware in [254, 253] {
            reverse_cache.verifiers.insert(Fp8VerifierKey { hardware }, verifier.clone());
        }
        let mut forward_cache = Fp8VerifierCache::default();
        for hardware in [253, 254] {
            forward_cache.verifiers.insert(Fp8VerifierKey { hardware }, verifier.clone());
        }
        assert_eq!(forward_cache.to_bytes()?, reverse_cache.to_bytes()?);
        let bytes = reverse_cache.to_bytes()?;
        assert_eq!(Fp8VerifierCache::from_bytes(&bytes)?.to_bytes()?, bytes);

        let duplicate = fp8_wire_options().serialize(&Fp8VerifierCacheWire {
            entries: (0..2)
                .map(|_| Fp8VerifierCacheEntryWire {
                    hardware: Device::B200 as u8,
                    verifier: verifier_bytes.clone(),
                })
                .collect(),
        })?;
        assert!(Fp8VerifierCache::from_bytes(&duplicate).is_err());
        let mut trailing_cache = cache.to_bytes()?;
        trailing_cache.push(0);
        assert!(Fp8VerifierCache::from_bytes(&trailing_cache).is_err());
        let mut trailing_verifier = verifier_bytes.clone();
        trailing_verifier.push(0);
        assert!(Fp8Verifier::from_bytes(&trailing_verifier).is_err());

        let mut wire: Fp8VerifierWire = fp8_wire_options().deserialize(&verifier_bytes)?;
        wire.lut_cap.clear();
        assert!(Fp8Verifier::from_bytes(&fp8_wire_options().serialize(&wire)?).is_err());
        let mut wire: Fp8VerifierWire = fp8_wire_options().deserialize(&verifier_bytes)?;
        wire.lut_cap[0][0] = F::ORDER;
        assert!(Fp8Verifier::from_bytes(&fp8_wire_options().serialize(&wire)?).is_err());
        let mut wire: Fp8VerifierWire = fp8_wire_options().deserialize(&verifier_bytes)?;
        let mut non_zk = data.verifier_data();
        non_zk.common.config.zero_knowledge = false;
        wire.circuit = non_zk.to_bytes(&DefaultGateSerializer).unwrap();
        assert!(Fp8Verifier::from_bytes(&fp8_wire_options().serialize(&wire)?).is_err());
        Ok(())
    }

    #[test]
    fn polynomial_codec_rejects_malformed_values() -> Result<()> {
        let (data, _) = cache_test_circuit();
        let polys = &data.prover_only.constants_sigmas_commitment.polynomials;
        let common = &data.common;
        let bytes = serialize_polynomials(polys, common);
        assert_eq!(deserialize_polynomials(&bytes, common)?, *polys);
        assert!(deserialize_polynomials(&bytes[..bytes.len() - 1], common).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(deserialize_polynomials(&trailing, common).is_err());

        let mut noncanonical = bytes.clone();
        noncanonical[..8].copy_from_slice(&F::ORDER.to_le_bytes());
        assert!(deserialize_polynomials(&noncanonical, common).is_err());

        let mut bad_sigma = bytes;
        let sigma_start = common.num_constants * common.degree() * 8;
        let width = CosetLayout::new(common).bytes_per_index;
        let invalid_index = common.config.num_routed_wires * common.degree();
        bad_sigma[sigma_start..sigma_start + width].copy_from_slice(&invalid_index.to_le_bytes()[..width]);
        assert!(
            deserialize_polynomials(&bad_sigma, common)
                .unwrap_err()
                .to_string()
                .contains("sigma permutation index out of bounds")
        );
        Ok(())
    }
}
