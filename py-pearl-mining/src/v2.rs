//! V2 ZK functions, which also serve cert v3 (same circuits, salted noise seed), plus Int7
//! mining and the rank-penalty bound.

use lazy_static::lazy_static;
use pyo3::prelude::*;
use std::sync::Mutex;

use primitive_types::U256;
use zk_pow::ffi::CertificateVersion;
use zk_pow::v2::api::sanity_checks;
use zk_pow::v2::api::seed::SeedDerivation;
use zk_pow::v2::ffi::plain_proof::{MatrixMerkleProof, MoEProofParams, PlainProof};

use zk_pow::v2::api::proof as v2_proof;
use zk_pow::v2::api::proof::{MMAType, MiningConfiguration, MoEConfig, PeriodicPattern, PublicProofParams, ZKProof};
use zk_pow::v2::api::{prove, verify};
use zk_pow::v2::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
use zk_pow::v2::mine::{mine as ffi_mine, mine_moe as ffi_mine_moe};

use crate::common::{py_err, value_err, IncompleteBlockHeader, PyProof};

/// The Python-facing `IncompleteBlockHeader` is the mainline (v3) type; the
/// v1/v2 clones each carry an identical struct that we convert into at the
/// dispatch boundary.
fn header_to_v2(h: &IncompleteBlockHeader) -> v2_proof::IncompleteBlockHeader {
    v2_proof::IncompleteBlockHeader {
        version: h.version,
        prev_block: h.prev_block,
        merkle_root: h.merkle_root,
        timestamp: h.timestamp,
        nbits: h.nbits,
    }
}

// ============================================================================
// ZK Functions
// ============================================================================

type CircuitCache = <PearlRecursion as RecursionCircuit>::CircuitCache;

lazy_static! {
    static ref CIRCUIT_CACHE: Mutex<CircuitCache> = Mutex::new(CircuitCache::default());
}

fn acquire_cache() -> PyResult<std::sync::MutexGuard<'static, CircuitCache>> {
    CIRCUIT_CACHE
        .lock()
        .map_err(|_| py_err("Cache poisoned by prior panic", "restart required"))
}

fn generate_proof_impl(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
    seed_derivation: SeedDerivation,
) -> PyResult<PyProof> {
    let mut cache = acquire_cache()?;
    let result = prove::zk_prove_plain_proof(
        header_to_v2(&block_header),
        &plain_proof,
        &mut cache,
        true,
        seed_derivation,
    )
    .map_err(|e| py_err("Prove failed", e))?;

    Ok(PyProof {
        public_data: result.public_data,
        proof_data: result.proof_data,
    })
}

#[pyfunction]
pub fn generate_proof_v2(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
) -> PyResult<PyProof> {
    generate_proof_impl(block_header, plain_proof, SeedDerivation::Legacy)
}

/// V3 proving: same circuits as V2, salted noise-seed derivation.
#[pyfunction]
pub fn generate_proof_v3(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
) -> PyResult<PyProof> {
    generate_proof_impl(block_header, plain_proof, SeedDerivation::Salted)
}

fn verify_proof_impl(
    block_header: IncompleteBlockHeader,
    proof: &PyProof,
    seed_derivation: SeedDerivation,
) -> PyResult<(bool, String)> {
    if !PublicProofParams::is_valid_wire_size(proof.public_data.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "public_data length must be {} bytes (non-MoE) or {}..={} (MoE)",
            PublicProofParams::WIRE_SIZE,
            PublicProofParams::MIN_MOE_WIRE_SIZE,
            PublicProofParams::MAX_WIRE_SIZE,
        )));
    }

    let (params, zk_proof) = ZKProof::deserialize(
        header_to_v2(&block_header),
        seed_derivation,
        &proof.public_data,
        &proof.proof_data,
    )
    .map_err(|e| py_err("Deserialize failed", e))?;

    let mut cache = acquire_cache()?;
    match verify::verify_block(&params, &zk_proof, &mut cache) {
        Ok(_) => Ok((true, "Verified".into())),
        Err(e) => Ok((false, format!("Rejected: {}", e))),
    }
}

#[pyfunction]
pub fn verify_proof_v2(
    block_header: IncompleteBlockHeader,
    proof: &PyProof,
) -> PyResult<(bool, String)> {
    verify_proof_impl(block_header, proof, SeedDerivation::Legacy)
}

/// V3 (salted noise-seed) verification: same circuits and cache as V2.
#[pyfunction]
pub fn verify_proof_v3(
    block_header: IncompleteBlockHeader,
    proof: &PyProof,
) -> PyResult<(bool, String)> {
    verify_proof_impl(block_header, proof, SeedDerivation::Salted)
}

#[pyfunction]
pub fn clear_circuit_cache_v2() -> PyResult<()> {
    acquire_cache()?.clear();
    Ok(())
}

#[pyfunction]
pub fn warmup_prove_v2(mining_config: MiningConfiguration) -> PyResult<()> {
    let mut cache = acquire_cache()?;
    prove::warmup_prove(mining_config, &mut cache).map_err(|e| py_err("Warmup prove failed", e))
}

fn verify_plain_proof_impl(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
    nbits_override: Option<u32>,
    seed_derivation: SeedDerivation,
) -> PyResult<(bool, String)> {
    match verify::verify_plain_proof(
        &header_to_v2(&block_header),
        &plain_proof,
        nbits_override,
        seed_derivation,
    ) {
        Ok(()) => Ok((true, "Mining solution verified successfully".into())),
        Err(e) => Ok((false, e.to_string())),
    }
}

#[pyfunction]
#[pyo3(signature = (block_header, plain_proof, nbits_override=None))]
pub fn verify_plain_proof_v2(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
    nbits_override: Option<u32>,
) -> PyResult<(bool, String)> {
    verify_plain_proof_impl(
        block_header,
        plain_proof,
        nbits_override,
        SeedDerivation::Legacy,
    )
}

/// V3 (salted noise-seed) Int7 plain-proof verification.
#[pyfunction]
#[pyo3(signature = (block_header, plain_proof, nbits_override=None))]
pub fn verify_plain_proof_v3(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
    nbits_override: Option<u32>,
) -> PyResult<(bool, String)> {
    verify_plain_proof_impl(
        block_header,
        plain_proof,
        nbits_override,
        SeedDerivation::Salted,
    )
}

#[pyfunction]
pub fn penalized_target_bound<'py>(
    py: Python<'py>,
    target: &Bound<'_, PyAny>,
    mining_config: &MiningConfiguration,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    // target (int) -> 32 little-endian bytes
    let target_bytes = target.call_method1("to_bytes", (32usize, "little"))?;
    let target_bytes: &[u8] = target_bytes.extract()?;
    let bound = match sanity_checks::penalized_target_bound(
        U256::from_little_endian(target_bytes),
        mining_config,
    ) {
        Some(b) => b,
        None => return Ok(None),
    };
    let mut out = [0u8; 32];
    bound.to_little_endian(&mut out);
    // 32 little-endian bytes -> int
    let py_bytes = pyo3::types::PyBytes::new(py, &out);
    let bound_int = py
        .get_type::<pyo3::types::PyInt>()
        .call_method1("from_bytes", (py_bytes, "little"))?;
    Ok(Some(bound_int))
}

/// Map a certificate version to its noise-seed derivation (errors on unknown versions).
fn seed_derivation_for(cert_version: u32) -> PyResult<SeedDerivation> {
    Ok(CertificateVersion::try_from(cert_version)
        .map_err(value_err)?
        .seed_derivation())
}

#[pyfunction]
#[pyo3(signature = (m, n, k, block_header, mining_config, signal_range=None, wrong_jackpot_hash=false, *, cert_version))]
#[allow(clippy::too_many_arguments)]
pub fn mine(
    m: usize,
    n: usize,
    k: usize,
    block_header: IncompleteBlockHeader,
    mining_config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
    cert_version: u32,
) -> PyResult<PlainProof> {
    ffi_mine(
        m,
        n,
        k,
        header_to_v2(&block_header),
        mining_config,
        signal_range,
        wrong_jackpot_hash,
        seed_derivation_for(cert_version)?,
    )
    .map_err(|e| py_err("Mining failed", e))
}

#[pyfunction]
#[pyo3(signature = (m, n, k, block_header, mining_config, signal_range=None, wrong_jackpot_hash=false, *, cert_version))]
#[allow(clippy::too_many_arguments)]
pub fn mine_moe(
    m: usize,
    n: usize,
    k: usize,
    block_header: IncompleteBlockHeader,
    mining_config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
    cert_version: u32,
) -> PyResult<PlainProof> {
    // Both `e` and `top_k` are committed in `mining_config` (via its `moe` field), so
    // the caller selects GROUPED_GEMM by passing a config with `moe` set.
    ffi_mine_moe(
        m,
        n,
        k,
        header_to_v2(&block_header),
        mining_config,
        signal_range,
        wrong_jackpot_hash,
        seed_derivation_for(cert_version)?,
    )
    .map_err(|e| py_err("MoE mining failed", e))
}

pub fn register_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PUBLICDATA_SIZE", PublicProofParams::WIRE_SIZE)?;
    m.add(
        "MIN_MOE_PUBLICDATA_SIZE",
        PublicProofParams::MIN_MOE_WIRE_SIZE,
    )?;
    m.add("PUBLICDATA_MAX_SIZE", PublicProofParams::MAX_WIRE_SIZE)?;
    m.add(
        "PENALTY_BASE_RANK",
        zk_pow::v2::api::sanity_checks::PENALTY_BASE_RANK,
    )?;
    Ok(())
}

pub fn register_pattern(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PeriodicPattern>()?;
    Ok(())
}

pub fn register_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MiningConfiguration>()?;
    m.add_class::<MoEConfig>()?;
    m.add_class::<MMAType>()?;
    m.add_class::<MatrixMerkleProof>()?;
    m.add_class::<PlainProof>()?;
    Ok(())
}

pub fn register_moe_proof_params(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MoEProofParams>()?;
    Ok(())
}

pub fn register_mining(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mine, m)?)?;
    m.add_function(wrap_pyfunction!(mine_moe, m)?)?;
    m.add_function(wrap_pyfunction!(penalized_target_bound, m)?)?;
    Ok(())
}

pub fn register_proofs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_proof_v2, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof_v2, m)?)?;
    m.add_function(wrap_pyfunction!(verify_plain_proof_v2, m)?)?;
    m.add_function(wrap_pyfunction!(clear_circuit_cache_v2, m)?)?;
    m.add_function(wrap_pyfunction!(warmup_prove_v2, m)?)?;
    m.add_function(wrap_pyfunction!(generate_proof_v3, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof_v3, m)?)?;
    m.add_function(wrap_pyfunction!(verify_plain_proof_v3, m)?)?;
    Ok(())
}
