// Utility functions and types for circuit configuration and validation

use std::hash::{Hash, Hasher};
use std::mem::size_of;

use anyhow::{Result, bail};
use hashbrown::HashMap;
use plonky2::{
    field::{
        cosets::get_unique_coset_shifts,
        goldilocks_field::GoldilocksField,
        polynomial::{PolynomialCoeffs, PolynomialValues},
        types::Field,
    },
    fri::{FriConfig, reduction_strategies::FriReductionStrategy},
    plonk::{
        circuit_data::{CircuitConfig, CommonCircuitData, ProverCircuitData, VerifierCircuitData},
        config::{Blake3GoldilocksConfig, PoseidonGoldilocksConfig},
        proof::ProofWithPublicInputsTarget,
    },
    util::serialization::{Buffer, DefaultGateSerializer, Read, Remaining},
};
use plonky2_field::types::PrimeField64;
use starky::proof::StarkProofWithPublicInputsTarget;

use crate::ensure_eq;
use crate::v1::circuit::pearl_circuit::{PearlCircuitParams, SECURITY_BITS};

//==============================================================================
// Cache key and data structures
//==============================================================================

/// Key for first circuit cache - minimal entropy based on first layer parameters
#[derive(Hash, Eq, PartialEq, Clone, Debug, Copy, serde::Serialize, serde::Deserialize)]
pub struct FirstCircuitKey {
    pub stark_degree_bits: usize,
    pub pow_bits_0: usize,
    pub rate_bits_0: usize,
    pub pow_bits_1: usize,
    pub rate_bits_1: usize,
}

/// Key for second circuit cache - based on first circuit's structure and second layer parameters
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecondCircuitKey {
    pub first_circuit: CommonCircuitData<GoldilocksField, 2>,
    pub pow_bits_2: usize,
    pub rate_bits_2: usize,
}

impl Hash for SecondCircuitKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the identifying fields from CommonCircuitData
        self.first_circuit.degree_bits().hash(state);
        self.first_circuit.num_public_inputs.hash(state);
        self.first_circuit.num_constants.hash(state);
        self.first_circuit.num_gate_constraints.hash(state);
        self.pow_bits_2.hash(state);
        self.rate_bits_2.hash(state);
    }
}

impl SecondCircuitKey {
    /// Serialize to bytes: pow_bits_2(1) | rate_bits_2(1) | first_circuit_len(4) | first_circuit
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let first_circuit_bytes = self
            .first_circuit
            .to_bytes(&DefaultGateSerializer)
            .map_err(|_| anyhow::anyhow!("Failed to serialize first_circuit"))?;
        Ok([
            &[self.pow_bits_2 as u8, self.rate_bits_2 as u8][..],
            &(first_circuit_bytes.len() as u32).to_le_bytes(),
            &first_circuit_bytes,
        ]
        .concat())
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let total_len = Self::serialized_len(data)?;
        let first_circuit = CommonCircuitData::from_bytes(data[6..total_len].to_vec(), &DefaultGateSerializer)
            .map_err(|_| anyhow::anyhow!("Failed to deserialize first_circuit"))?;
        Ok(Self {
            first_circuit,
            pow_bits_2: data[0] as usize,
            rate_bits_2: data[1] as usize,
        })
    }

    /// Returns the byte length of the serialized key (for parsing)
    pub fn serialized_len(data: &[u8]) -> Result<usize> {
        let len_bytes: [u8; 4] = data
            .get(2..6)
            .ok_or_else(|| anyhow::anyhow!("truncated SecondCircuitKey"))?
            .try_into()?;
        Ok(6 + u32::from_le_bytes(len_bytes) as usize)
    }
}

/// First circuit data stored in cache (prover-only)
pub struct FirstCircuitData {
    pub circuit: ProverCircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>,
    pub proof_0_target: StarkProofWithPublicInputsTarget<2>,
}

/// Second circuit data stored in cache (prover-only)
pub struct SecondCircuitData {
    pub circuit: ProverCircuitData<GoldilocksField, Blake3GoldilocksConfig, 2>,
    pub proof_1_target: ProofWithPublicInputsTarget<2>,
}

//==============================================================================
// Circuit configuration utilities
//==============================================================================

/// Calculate number of query rounds for FRI
pub fn num_query_rounds(security_bits: usize, pow_bits: usize, rate_bits: usize) -> usize {
    security_bits.saturating_sub(pow_bits).div_ceil(rate_bits)
}

/// Build a recursion circuit config with the given parameters
pub fn build_recursion_config(rate_bits: usize, pow_bits: usize, stage: usize, is_zk: bool) -> CircuitConfig {
    debug_assert!(rate_bits >= 3);
    CircuitConfig {
        num_wires: 135,
        num_routed_wires: if stage == 2 { 34 } else { 37 },
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

//==============================================================================
// CircuitCache and polynomial serialization
//==============================================================================

/// Verifier data bundled with the constants_sigmas polynomial coefficients
/// needed for compact proof verification.
pub struct VerifierCircuitWithPolynomials {
    pub verifier_data: VerifierCircuitData<GoldilocksField, Blake3GoldilocksConfig, 2>,
    pub constants_sigmas_polynomials: Vec<PolynomialCoeffs<GoldilocksField>>,
}

/// Circuit cache with separate storage for verifier-only and full prover data
#[derive(Default)]
pub struct CircuitCache {
    // Verifier-only data (lightweight)
    pub verifier_circuits_1: HashMap<FirstCircuitKey, VerifierCircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>>,
    pub verifier_circuits_2: HashMap<SecondCircuitKey, VerifierCircuitWithPolynomials>,

    // Full prover data (includes everything needed for proving)
    pub prover_circuits_1: HashMap<FirstCircuitKey, FirstCircuitData>,
    pub prover_circuits_2: HashMap<SecondCircuitKey, SecondCircuitData>,
}

const CACHE_MAGIC: [u8; 4] = *b"PRL1"; // Pearl Cache Version 1

impl CircuitCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn make_first_circuit_key(params: PearlCircuitParams) -> FirstCircuitKey {
        FirstCircuitKey {
            stark_degree_bits: params.stark_degree_bits,
            pow_bits_0: params.pow_bits[0],
            rate_bits_0: params.rate_bits[0],
            pow_bits_1: params.pow_bits[1],
            rate_bits_1: params.rate_bits[1],
        }
    }

    pub(crate) fn make_second_circuit_key(
        first_circuit: CommonCircuitData<GoldilocksField, 2>,
        params: PearlCircuitParams,
    ) -> SecondCircuitKey {
        SecondCircuitKey {
            first_circuit,
            pow_bits_2: params.pow_bits[2],
            rate_bits_2: params.rate_bits[2],
        }
    }

    pub fn clear(&mut self) {
        self.verifier_circuits_1.clear();
        self.verifier_circuits_2.clear();
        self.prover_circuits_1.clear();
        self.prover_circuits_2.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.verifier_circuits_1.is_empty()
            && self.verifier_circuits_2.is_empty()
            && self.prover_circuits_1.is_empty()
            && self.prover_circuits_2.is_empty()
    }

    /// Serialize verifier circuits to binary format for cache generation
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut binary_data = Vec::new();

        // Write header: magic bytes + counts
        binary_data.extend_from_slice(&CACHE_MAGIC);
        binary_data.extend_from_slice(&(self.verifier_circuits_1.len() as u32).to_le_bytes());
        binary_data.extend_from_slice(&(self.verifier_circuits_2.len() as u32).to_le_bytes());

        // Helper to write length-prefixed metadata + circuit bytes
        let mut write_entry = |key_bytes: &[u8], circuit_bytes: &[u8]| {
            let total_len = key_bytes.len() + circuit_bytes.len();
            binary_data.extend_from_slice(&(total_len as u32).to_le_bytes());
            binary_data.extend_from_slice(key_bytes);
            binary_data.extend_from_slice(circuit_bytes);
        };

        // Serialize first circuits (key + circuit data) - sorted for deterministic output
        let mut first_keys: Vec<_> = self.verifier_circuits_1.keys().collect();
        first_keys.sort_by_key(|k| (k.stark_degree_bits, k.rate_bits_0, k.pow_bits_0));
        for key in first_keys {
            let circuit = &self.verifier_circuits_1[key];
            let key_bytes = bincode::serialize(key)?;
            let circuit_bytes = circuit
                .to_bytes(&DefaultGateSerializer)
                .map_err(|_| anyhow::anyhow!("Failed to serialize first circuit"))?;
            write_entry(&key_bytes, &circuit_bytes);
        }

        // Serialize second circuits (key + circuit data + polynomials) - sorted for deterministic output
        let mut second_keys: Vec<_> = self.verifier_circuits_2.keys().collect();
        second_keys.sort_by_key(|k| (k.first_circuit.degree_bits(), k.pow_bits_2, k.rate_bits_2));
        for key in second_keys {
            let entry = &self.verifier_circuits_2[key];
            let key_bytes = key.to_bytes()?;
            let circuit_bytes = entry
                .verifier_data
                .to_bytes(&DefaultGateSerializer)
                .map_err(|_| anyhow::anyhow!("Failed to serialize second circuit"))?;
            let poly_bytes = serialize_polynomials(&entry.constants_sigmas_polynomials, &entry.verifier_data.common);

            let total_len = key_bytes.len() + size_of::<u32>() + circuit_bytes.len() + poly_bytes.len();
            binary_data.extend_from_slice(&(total_len as u32).to_le_bytes());
            binary_data.extend_from_slice(&key_bytes);
            binary_data.extend_from_slice(&(circuit_bytes.len() as u32).to_le_bytes());
            binary_data.extend_from_slice(&circuit_bytes);
            binary_data.extend_from_slice(&poly_bytes);
        }

        Ok(binary_data)
    }

    /// Load verifier circuits from binary format.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cache = Self::default();
        if data.len() <= CACHE_MAGIC.len() + 2 * size_of::<u32>() {
            return Ok(cache);
        }

        let mut reader = Buffer::new(data);
        let magic = reader
            .read_bytes(CACHE_MAGIC.len())
            .map_err(|_| anyhow::anyhow!("unexpected end of cache data"))?;
        if magic != CACHE_MAGIC {
            bail!("Invalid cache magic bytes or version mismatch. Please regenerate the cache.");
        }

        let first_count = reader
            .read_u32()
            .map_err(|_| anyhow::anyhow!("unexpected end of cache data"))? as usize;
        let second_count = reader
            .read_u32()
            .map_err(|_| anyhow::anyhow!("unexpected end of cache data"))? as usize;

        for _ in 0..first_count {
            let chunk_len = reader
                .read_u32()
                .map_err(|_| anyhow::anyhow!("unexpected end of cache data"))? as usize;
            let chunk = reader
                .read_bytes(chunk_len)
                .map_err(|_| anyhow::anyhow!("unexpected end of cache data"))?;

            let key_size = size_of::<FirstCircuitKey>();
            let key: FirstCircuitKey = bincode::deserialize(
                chunk
                    .get(..key_size)
                    .ok_or_else(|| anyhow::anyhow!("truncated first circuit key"))?,
            )?;
            let circuit_bytes = chunk
                .get(key_size..)
                .ok_or_else(|| anyhow::anyhow!("truncated first circuit data"))?;
            let circuit = VerifierCircuitData::from_bytes(circuit_bytes.to_vec(), &DefaultGateSerializer)
                .map_err(|_| anyhow::anyhow!("Failed to deserialize first circuit"))?;
            cache.verifier_circuits_1.insert(key, circuit);
        }

        for _ in 0..second_count {
            let chunk_len = reader
                .read_u32()
                .map_err(|_| anyhow::anyhow!("unexpected end of cache data"))? as usize;
            let chunk = reader
                .read_bytes(chunk_len)
                .map_err(|_| anyhow::anyhow!("unexpected end of cache data"))?;

            let key_len = SecondCircuitKey::serialized_len(chunk)?;
            let key = SecondCircuitKey::from_bytes(chunk)?;
            let mut chunk_reader = Buffer::new(
                chunk
                    .get(key_len..)
                    .ok_or_else(|| anyhow::anyhow!("truncated second circuit chunk"))?,
            );

            let circuit_len = chunk_reader
                .read_u32()
                .map_err(|_| anyhow::anyhow!("unexpected end of cache data"))? as usize;
            let circuit_data = chunk_reader
                .read_bytes(circuit_len)
                .map_err(|_| anyhow::anyhow!("unexpected end of cache data"))?;
            let verifier_data = VerifierCircuitData::from_bytes(circuit_data.to_vec(), &DefaultGateSerializer)
                .map_err(|_| anyhow::anyhow!("Failed to deserialize second circuit"))?;

            let constants_sigmas_polynomials = deserialize_polynomials(chunk_reader.unread_bytes(), &verifier_data.common)?;

            cache.verifier_circuits_2.insert(
                key,
                VerifierCircuitWithPolynomials {
                    verifier_data,
                    constants_sigmas_polynomials,
                },
            );
        }

        Ok(cache)
    }
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
            values.push(layout.k_is[idx / degree] * layout.subgroup[idx % degree]);
        }
        polys.push(PolynomialValues::new(values).ifft());
    }

    ensure_eq!(reader.remaining(), 0, "unexpected trailing bytes in polynomial data");

    Ok(polys)
}
