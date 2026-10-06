//! FP8 recursion configuration, trusted verifier setup, and cache serialization.
//!
//! [`crate::api::fp8::zk`] builds setups and uses them to verify proofs. This module
//! caches them by device and encodes their circuit data and polynomials.
//!
//! The polynomial codec stores constant evaluations as little-endian u64s.
//! Sigma polynomials describe the wire permutation. Each sigma evaluation
//! identifies a wire position and is encoded as a smaller integer:
//!
//! ```text
//! value = k_is[column] * subgroup[row]
//! index = column * degree + row
//! ```
//!
//! Here `degree` is the number of circuit rows, `k_is` holds one coset shift per
//! routed-wire column, and `subgroup` holds the evaluation points for the rows.

use anyhow::{Context, Result, ensure};
use bincode::Options;
use hashbrown::HashMap;
use plonky2::field::cosets::get_unique_coset_shifts;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::field::types::{Field, PrimeField64};
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData, VerifierCircuitData};
use plonky2::util::serialization::{Buffer, DefaultGateSerializer, Read, Remaining};
use serde::{Deserialize, Serialize};

use super::wrapper::{D, F, OuterC};
use crate::ensure_eq;
use crate::v2::circuit::circuit_utils::build_recursion_config as v2_build_recursion_config;
pub use crate::v2::circuit::circuit_utils::num_query_rounds;
use crate::v4::api::public_params::Device;

/// Build recursion settings for an FP8 wrapper stage.
/// Stage 2 uses more routed wires to reduce the outer proof size.
pub fn build_recursion_config(rate_bits: usize, pow_bits: usize, stage: usize, is_zk: bool) -> CircuitConfig {
    let mut config = v2_build_recursion_config(rate_bits, pow_bits, stage, is_zk);
    if stage == 2 {
        config.num_routed_wires = 40;
    }
    config
}

/// Trusted verifier setup: the universal wrapper's stage-2 verifier data plus the
/// stage-2 constants/sigmas polynomial coefficients the compact proof encoding omits
/// ([`super::wrapper::verify_compact_wrapped_proof`] recomputes the omitted oracle data from
/// them — same trust class as the verifier data itself, exactly the deployed
/// v2 `VerifierCircuitWithPolynomials` bundle). Covers every envelope-legal
/// job; obtain one from [`Fp8Verifier::generate`] or [`Fp8Verifier::from_bytes`].
#[derive(Clone, Debug)]
pub struct Fp8Verifier {
    pub(crate) circuit: VerifierCircuitData<F, OuterC, D>,
    pub(crate) constants_sigmas_polynomials: Vec<PolynomialCoeffs<F>>,
}

impl Fp8Verifier {
    /// Serialize setup for storage or distribution to verifiers.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let circuit = self
            .circuit
            .to_bytes(&DefaultGateSerializer)
            .map_err(|error| anyhow::anyhow!("serializing the recursive fp8 verifier circuit: {error:?}"))?;
        let wire = Fp8VerifierWire {
            circuit,
            constants_sigmas_polynomials: serialize_polynomials(&self.constants_sigmas_polynomials, &self.circuit.common),
        };
        verifier_setup_codec_options()
            .serialize(&wire)
            .context("serializing fp8 verifier setup")
    }

    /// Load setup serialized by [`Self::to_bytes`].
    /// The bytes must come from a trusted source; format checks do not authenticate the setup.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let wire: Fp8VerifierWire = verifier_setup_codec_options()
            .deserialize(bytes)
            .context("deserializing fp8 verifier setup")?;
        let circuit = VerifierCircuitData::from_bytes(wire.circuit, &DefaultGateSerializer)
            .map_err(|error| anyhow::anyhow!("deserializing the recursive fp8 verifier circuit: {error}"))?;
        ensure!(
            circuit.common.config.zero_knowledge,
            "the fp8 verifier setup must contain the ZK wrapper stage"
        );
        // The circuit determines the polynomial count and degree; the decoder rejects missing or extra data.
        let constants_sigmas_polynomials = deserialize_polynomials(&wire.constants_sigmas_polynomials, &circuit.common)
            .context("deserializing the fp8 verifier constants/sigmas polynomials")?;
        Ok(Self {
            circuit,
            constants_sigmas_polynomials,
        })
    }
}

/// A read-only store of verifier setups keyed by the statement's device byte. The
/// stage-1 wrapper is universal within one device family: one compiled circuit covers
/// every envelope-legal degree profile and geometry for that device, while Hopper and
/// Blackwell occupy distinct cache entries. Deployments preload
/// [`embedded_cache::CACHE_DATA`](crate::v4::api::embedded_cache::CACHE_DATA), built
/// offline by `build_cache` ([`Fp8Verifier::generate`] + [`Fp8VerifierCache::insert`]).
///
/// Verification fails if the device has no cached setup. Compiling on a cache miss
/// would let submitted proofs trigger expensive circuit builds.
#[derive(Default)]
pub struct Fp8VerifierCache {
    verifiers: HashMap<Fp8VerifierKey, Fp8Verifier>,
}

/// Everything that selects one trusted setup: the statement's device.
///
/// Nothing else is key material (the D1 universal design): the degree profile rides the
/// wrapper's degree public inputs and the geometry rides the `K`/`WL2`/`2^WL2` public
/// inputs and the known columns, all pinned natively by the gateway; the AIR identities,
/// the CTL set and the FRI ladder are consensus constants.
type Fp8VerifierKey = Device;

impl Fp8VerifierCache {
    /// Load trusted cache bytes produced by [`Self::to_bytes`].
    /// Empty bytes, used when no cache is embedded, produce an empty cache.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }
        let entries: Vec<(u8, Vec<u8>)> = verifier_setup_codec_options()
            .deserialize(bytes)
            .context("deserializing the fp8 verifier cache")?;
        let mut verifiers = HashMap::new();
        for (hardware, serialized_verifier) in entries {
            let key = Device::try_from(hardware).context("invalid device in fp8 verifier cache")?;
            let verifier = Fp8Verifier::from_bytes(&serialized_verifier)?;
            ensure!(
                verifiers.insert(key, verifier).is_none(),
                "duplicate fp8 verifier cache entry"
            );
        }
        Ok(Self { verifiers })
    }

    /// Serialize entries in device order, giving identical bytes regardless of insertion order.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut entries = self
            .verifiers
            .iter()
            .map(|(key, verifier)| Ok((*key as u8, verifier.to_bytes()?)))
            .collect::<Result<Vec<_>>>()?;
        entries.sort_by_key(|(hardware, _)| *hardware);
        verifier_setup_codec_options()
            .serialize(&entries)
            .context("serializing the fp8 verifier cache")
    }

    /// Number of cached setups.
    pub fn len(&self) -> usize {
        self.verifiers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.verifiers.is_empty()
    }

    /// Whether the cache contains a setup for `device`.
    pub fn contains_device(&self, device: Device) -> bool {
        self.verifiers.contains_key(&device)
    }

    /// Production completeness: exactly one setup for every committed device.
    pub fn contains_all_devices(&self) -> bool {
        self.len() == Device::ALL.len() && Device::ALL.into_iter().all(|device| self.contains_device(device))
    }

    /// Registers a pre-built verifier setup under `device`
    /// (build tooling: assembles an embeddable cache without regenerating setups
    /// already compiled elsewhere). Replaces any previous setup of the same device.
    pub fn insert(&mut self, device: Device, verifier: Fp8Verifier) {
        self.verifiers.insert(device, verifier);
    }

    /// Look up the device's setup; return `None` if it has not been loaded.
    pub(crate) fn get(&self, hardware: Device) -> Option<&Fp8Verifier> {
        self.verifiers.get(&hardware)
    }
}

fn verifier_setup_codec_options() -> impl Options {
    bincode::options().with_fixint_encoding().reject_trailing_bytes()
}

/// Serialized verifier-circuit bytes plus the circuit's constants/sigmas polynomials.
#[derive(Serialize, Deserialize)]
struct Fp8VerifierWire {
    circuit: Vec<u8>,
    constants_sigmas_polynomials: Vec<u8>,
}

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

/// Evaluation points and index width shared by the polynomial encoder and decoder.
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

/// Encode setup polynomials as field evaluations and wire-permutation indices.
fn serialize_polynomials(
    polys: &[PolynomialCoeffs<GoldilocksField>],
    common_data: &CommonCircuitData<GoldilocksField, 2>,
) -> Vec<u8> {
    let num_constants = common_data.num_constants;
    let num_routed_wires = common_data.config.num_routed_wires;
    let degree = common_data.degree();
    let layout = CosetLayout::new(common_data);

    // Map each allowed sigma evaluation to its wire position.
    let mut reverse_map: HashMap<GoldilocksField, usize> = HashMap::with_capacity(num_routed_wires * degree);
    for col in 0..num_routed_wires {
        for row in 0..degree {
            let val = layout.k_is[col] * layout.subgroup[row];
            reverse_map.insert(val, col * degree + row);
        }
    }

    let mut buf = Vec::new();

    // Store constant evaluations as canonical little-endian field elements.
    for poly in &polys[..num_constants] {
        let evals = poly.clone().fft();
        for &v in &evals.values {
            buf.extend_from_slice(&v.to_canonical_u64().to_le_bytes());
        }
    }

    // Sigma evaluations use only enough bytes to encode a wire position.
    for poly in &polys[num_constants..num_constants + num_routed_wires] {
        let evals = poly.clone().fft();
        for &v in &evals.values {
            let idx = reverse_map[&v];
            write_tight_le(&mut buf, idx, layout.bytes_per_index);
        }
    }

    buf
}

/// Decode the compact evaluations and recover polynomial coefficients.
/// The circuit fixes the required polynomial count and degree.
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
            // Split the wire index into column and row to recover its sigma evaluation.
            values.push(layout.k_is[idx / degree] * layout.subgroup[idx % degree]);
        }
        polys.push(PolynomialValues::new(values).ifft());
    }

    ensure_eq!(reader.remaining(), 0, "unexpected trailing bytes in polynomial data");

    Ok(polys)
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::Field64;
    use plonky2::fri::FriConfig;

    use super::*;

    // Small circuit for testing setup serialization and compact verification.
    // The reduced FRI security is suitable only for these tests.
    fn cache_test_circuit() -> (
        plonky2::plonk::circuit_data::CircuitData<F, OuterC, D>,
        plonky2::iop::target::Target,
    ) {
        use plonky2::fri::reduction_strategies::FriReductionStrategy;
        use plonky2::plonk::circuit_builder::CircuitBuilder;

        // Fix the parameters here so production tuning cannot change the expected serialized bytes.
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
            circuit: data.verifier_data(),
            constants_sigmas_polynomials: data.prover_only.constants_sigmas_commitment.polynomials.clone(),
        };
        // Pin the canonical verifier and cache encodings.
        let verifier_bytes = verifier.to_bytes()?;
        let mut cache = Fp8VerifierCache::default();
        cache.insert(Device::B200, verifier);
        let cache_bytes = cache.to_bytes()?;
        assert_eq!(
            blake3::hash(&verifier_bytes).to_hex().as_str(),
            "b10810a6550f57a7ede20d07df8c6acb03901611b6f7065017d77e730d1281bc"
        );
        assert_eq!(
            blake3::hash(&cache_bytes).to_hex().as_str(),
            "e6e1429f4d0a1c9f1ca1661539aeb769628d46d56109e9d016c92ad0d4d1d2c6"
        );

        let loaded = Fp8VerifierCache::from_bytes(&cache_bytes)?;
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.to_bytes()?, cache_bytes);
        let loaded_verifier = &loaded.verifiers[&Device::B200];
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
            circuit: data.verifier_data(),
            constants_sigmas_polynomials: data.prover_only.constants_sigmas_commitment.polynomials.clone(),
        };
        let verifier_bytes = verifier.to_bytes()?;
        let mut cache = Fp8VerifierCache::default();
        cache.insert(Device::B200, verifier.clone());
        cache.insert(Device::B200, verifier.clone());
        assert_eq!(cache.len(), 1, "insertion replaces the existing device setup");

        // Ordering is canonical regardless of map insertion order.
        let h100_verifier = verifier.clone();
        let mut reverse_cache = Fp8VerifierCache::default();
        for (device, setup) in [(Device::B200, verifier.clone()), (Device::H100, h100_verifier.clone())] {
            reverse_cache.verifiers.insert(device, setup);
        }
        let mut forward_cache = Fp8VerifierCache::default();
        for (device, setup) in [(Device::H100, h100_verifier), (Device::B200, verifier.clone())] {
            forward_cache.verifiers.insert(device, setup);
        }
        assert_eq!(forward_cache.to_bytes()?, reverse_cache.to_bytes()?);
        let bytes = reverse_cache.to_bytes()?;
        assert_eq!(Fp8VerifierCache::from_bytes(&bytes)?.to_bytes()?, bytes);

        let duplicate = verifier_setup_codec_options().serialize(&vec![(Device::B200 as u8, verifier_bytes.clone()); 2])?;
        assert!(Fp8VerifierCache::from_bytes(&duplicate).is_err());
        let unknown = verifier_setup_codec_options().serialize(&vec![(254, verifier_bytes.clone())])?;
        assert!(Fp8VerifierCache::from_bytes(&unknown).is_err());
        let mut trailing_cache = cache.to_bytes()?;
        trailing_cache.push(0);
        assert!(Fp8VerifierCache::from_bytes(&trailing_cache).is_err());
        let mut trailing_verifier = verifier_bytes.clone();
        trailing_verifier.push(0);
        assert!(Fp8Verifier::from_bytes(&trailing_verifier).is_err());

        let mut wire: Fp8VerifierWire = verifier_setup_codec_options().deserialize(&verifier_bytes)?;
        let mut non_zk = data.verifier_data();
        non_zk.common.config.zero_knowledge = false;
        wire.circuit = non_zk.to_bytes(&DefaultGateSerializer).unwrap();
        assert!(Fp8Verifier::from_bytes(&verifier_setup_codec_options().serialize(&wire)?).is_err());
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
