//! Unified Python module for Pearl mining.
//!
//! Registers types from pearl-blake3 and zk-pow into a single Python module.
//! No wrapper types -- all #[pyclass] types are defined in their respective core crates.

#[cfg(unix)]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use lazy_static::lazy_static;
use pyo3::prelude::*;
use std::sync::Mutex;

use blake3::CHUNK_LEN;
use pearl_blake3::{pad_to_chunk_boundary, MerkleProof, MerkleTree};
use zk_pow::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PublicProofParams,
    ZKProof,
};
use zk_pow::api::{prove, verify};
use zk_pow::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
use zk_pow::ffi::mine::mine as ffi_mine;
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof};

fn py_err(msg: &str, e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}: {}", msg, e))
}

// ============================================================================
// ZK Proof (only type defined in the binding crate)
// ============================================================================

#[pyclass(name = "ZKProof", get_all)]
#[derive(Clone)]
struct PyProof {
    public_data: Vec<u8>,
    proof_data: Vec<u8>,
}

#[pymethods]
impl PyProof {
    #[new]
    fn new(public_data: Vec<u8>, proof_data: Vec<u8>) -> PyResult<Self> {
        if public_data.len() != PublicProofParams::PUBLICDATA_SIZE {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "public_data must be exactly {} bytes",
                PublicProofParams::PUBLICDATA_SIZE
            )));
        }
        Ok(Self {
            public_data,
            proof_data,
        })
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

#[pyfunction]
fn generate_proof(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
) -> PyResult<PyProof> {
    let mut cache = acquire_cache()?;
    let result = prove::zk_prove_plain_proof(block_header, &plain_proof, &mut cache, true)
        .map_err(|e| py_err("Prove failed", e))?;

    Ok(PyProof {
        public_data: result.public_data.to_vec(),
        proof_data: result.proof_data,
    })
}

#[pyfunction]
fn verify_proof(block_header: IncompleteBlockHeader, proof: &PyProof) -> PyResult<(bool, String)> {
    if proof.public_data.len() != PublicProofParams::PUBLICDATA_SIZE {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "public_data must be exactly {} bytes",
            PublicProofParams::PUBLICDATA_SIZE
        )));
    }

    let public_data: &[u8; PublicProofParams::PUBLICDATA_SIZE] =
        proof.public_data.as_slice().try_into().unwrap();
    // (length already validated above)
    let (params, zk_proof) = ZKProof::deserialize(block_header, public_data, &proof.proof_data)
        .map_err(|e| py_err("Deserialize failed", e))?;

    let mut cache = acquire_cache()?;
    match verify::verify_block(&params, &zk_proof, &mut cache) {
        Ok(_) => Ok((true, "Verified".into())),
        Err(e) => Ok((false, format!("Rejected: {}", e))),
    }
}

#[pyfunction]
#[pyo3(name = "pad_to_chunk_boundary")]
fn py_pad_to_chunk_boundary(data: &[u8]) -> Vec<u8> {
    pad_to_chunk_boundary(data)
}

#[pyfunction]
fn clear_circuit_cache() -> PyResult<()> {
    acquire_cache()?.clear();
    Ok(())
}

#[pyfunction]
fn warmup_prove(mining_config: MiningConfiguration) -> PyResult<()> {
    let mut cache = acquire_cache()?;
    prove::warmup_prove(mining_config, &mut cache).map_err(|e| py_err("Warmup prove failed", e))
}

#[pyfunction]
#[pyo3(signature = (block_header, plain_proof, nbits_override=None))]
fn verify_plain_proof(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
    nbits_override: Option<u32>,
) -> PyResult<(bool, String)> {
    match verify::verify_plain_proof(&block_header, &plain_proof, nbits_override) {
        Ok(()) => Ok((true, "Mining solution verified successfully".into())),
        Err(e) => Ok((false, e.to_string())),
    }
}

#[pyfunction]
#[pyo3(signature = (m, n, k, block_header, mining_config, signal_range=None, wrong_jackpot_hash=false))]
fn mine(
    m: usize,
    n: usize,
    k: usize,
    block_header: IncompleteBlockHeader,
    mining_config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> PyResult<PlainProof> {
    ffi_mine(
        m,
        n,
        k,
        block_header,
        mining_config,
        signal_range,
        wrong_jackpot_hash,
    )
    .map_err(|e| py_err("Mining failed", e))
}

// ============================================================================
// Module
// ============================================================================

const DEFAULT_RAYON_THREADS: usize = 6;

fn rayon_thread_count() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_RAYON_THREADS)
}

#[pymodule]
fn pearl_mining(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init();
    rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_thread_count())
        .build_global()
        .expect("Failed to initialize rayon global thread pool");
    m.add("MERKLE_LEAF_SIZE", CHUNK_LEN)?;
    m.add("PUBLICDATA_SIZE", PublicProofParams::PUBLICDATA_SIZE)?;
    m.add_class::<MerkleTree>()?;
    m.add_class::<MerkleProof>()?;
    m.add_class::<PeriodicPattern>()?;
    m.add_class::<IncompleteBlockHeader>()?;
    m.add_class::<MiningConfiguration>()?;
    m.add_class::<MMAType>()?;
    m.add_class::<MatrixMerkleProof>()?;
    m.add_class::<PlainProof>()?;
    m.add_class::<PyProof>()?;
    m.add_function(wrap_pyfunction!(mine, m)?)?;
    m.add_function(wrap_pyfunction!(generate_proof, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof, m)?)?;
    m.add_function(wrap_pyfunction!(verify_plain_proof, m)?)?;
    m.add_function(wrap_pyfunction!(clear_circuit_cache, m)?)?;
    m.add_function(wrap_pyfunction!(warmup_prove, m)?)?;
    m.add_function(wrap_pyfunction!(py_pad_to_chunk_boundary, m)?)?;
    Ok(())
}
