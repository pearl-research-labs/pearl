//! V1 ZK functions: the legacy circuit (dense proofs only) and its cache.

use lazy_static::lazy_static;
use pyo3::prelude::*;
use std::sync::Mutex;

use zk_pow::v1::api::proof as v1_proof;
use zk_pow::v1::api::{prove as v1_prove, verify as v1_verify};
use zk_pow::v2::api::proof::MiningConfiguration;
use zk_pow::v2::ffi::plain_proof::PlainProof;

use crate::common::{py_err, IncompleteBlockHeader, PyProof};

// ============================================================================
// V1 ZK Functions
// ============================================================================

type V1CircuitCache = zk_pow::v1::circuit::circuit_utils::CircuitCache;

lazy_static! {
    static ref V1_CIRCUIT_CACHE: Mutex<V1CircuitCache> = Mutex::new(V1CircuitCache::default());
}

fn acquire_v1_cache() -> PyResult<std::sync::MutexGuard<'static, V1CircuitCache>> {
    V1_CIRCUIT_CACHE
        .lock()
        .map_err(|_| py_err("V1 cache poisoned by prior panic", "restart required"))
}

fn v2_header_to_v1(h: &IncompleteBlockHeader) -> v1_proof::IncompleteBlockHeader {
    v1_proof::IncompleteBlockHeader {
        version: h.version,
        prev_block: h.prev_block,
        merkle_root: h.merkle_root,
        timestamp: h.timestamp,
        nbits: h.nbits,
    }
}

fn v2_config_to_v1(cfg: &MiningConfiguration) -> v1_proof::MiningConfiguration {
    v1_proof::MiningConfiguration {
        common_dim: cfg.common_dim,
        rank: cfg.rank,
        mma_type: v1_proof::MMAType::Int7xInt7ToInt32,
        rows_pattern: v1_proof::PeriodicPattern {
            shape: cfg.rows_pattern.shape,
        },
        cols_pattern: v1_proof::PeriodicPattern {
            shape: cfg.cols_pattern.shape,
        },
        reserved: v1_proof::MiningConfiguration::RESERVED_VALUE,
    }
}

#[pyfunction]
pub fn generate_proof_v1(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
) -> PyResult<PyProof> {
    let v1_header = v2_header_to_v1(&block_header);
    let mut cache = acquire_v1_cache()?;
    let result = v1_prove::zk_prove_plain_proof(v1_header, &plain_proof, &mut cache, true)
        .map_err(|e| py_err("V1 prove failed", e))?;

    Ok(PyProof {
        public_data: result.public_data.to_vec(),
        proof_data: result.proof_data,
    })
}

#[pyfunction]
pub fn verify_proof_v1(
    block_header: IncompleteBlockHeader,
    proof: &PyProof,
) -> PyResult<(bool, String)> {
    if proof.public_data.len() != v1_proof::PublicProofParams::PUBLICDATA_SIZE {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "V1 public_data must be exactly {} bytes",
            v1_proof::PublicProofParams::PUBLICDATA_SIZE
        )));
    }

    let v1_header = v2_header_to_v1(&block_header);
    let public_data: &[u8; v1_proof::PublicProofParams::PUBLICDATA_SIZE] =
        proof.public_data.as_slice().try_into().unwrap();
    let (params, zk_proof) =
        v1_proof::ZKProof::deserialize(v1_header, public_data, &proof.proof_data)
            .map_err(|e| py_err("V1 deserialize failed", e))?;

    let mut cache = acquire_v1_cache()?;
    match v1_verify::verify_block(&params, &zk_proof, &mut cache) {
        Ok(_) => Ok((true, "Verified".into())),
        Err(e) => Ok((false, format!("Rejected: {}", e))),
    }
}

#[pyfunction]
#[pyo3(signature = (block_header, plain_proof, nbits_override=None))]
pub fn verify_plain_proof_v1(
    block_header: IncompleteBlockHeader,
    plain_proof: PlainProof,
    nbits_override: Option<u32>,
) -> PyResult<(bool, String)> {
    let v1_header = v2_header_to_v1(&block_header);
    match v1_verify::verify_plain_proof(&v1_header, &plain_proof, nbits_override) {
        Ok(()) => Ok((true, "Mining solution verified successfully".into())),
        Err(e) => Ok((false, e.to_string())),
    }
}

#[pyfunction]
pub fn warmup_prove_v1(mining_config: MiningConfiguration) -> PyResult<()> {
    let v1_config = v2_config_to_v1(&mining_config);
    let mut cache = acquire_v1_cache()?;
    v1_prove::warmup_prove(v1_config, &mut cache).map_err(|e| py_err("V1 warmup prove failed", e))
}

#[pyfunction]
pub fn clear_v1_circuit_cache() -> PyResult<()> {
    acquire_v1_cache()?.clear();
    Ok(())
}
