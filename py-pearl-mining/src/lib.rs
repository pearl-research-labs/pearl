//! Unified Python module for Pearl mining.
//!
//! Registers types from pearl-blake3 and zk-pow into a single Python module.
//! No wrapper types -- all #[pyclass] types are defined in their respective core crates.
//!
//! Each `vN` module holds one proof version's functions (`v2` also serves cert v3: same
//! circuits, salted noise seed). Version modules never import each other; they share only
//! [`common`]. [`cert_version`] dispatches across versions. Everything Python sees is
//! imported below and registered in [`pearl_mining`].

#[cfg(unix)]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod cert_version;
mod common;
mod v1;
mod v2;
mod v4;

use pyo3::prelude::*;

use blake3::CHUNK_LEN;
use pearl_blake3::{MerkleProof, MerkleTree};
use zk_pow::ffi::py_v4::{PyFp8Prover, PyFp8Verifier};
use zk_pow::ffi::CertificateVersion;
use zk_pow::v2::ffi::plain_proof::{MatrixMerkleProof, MoEProofParams, PlainProof};
use zk_pow::v4::api::layout::{AxisPattern, DimType};
use zk_pow::v4::api::plain_proof::{MoeWitness, PlainProofV4};
use zk_pow::v4::api::primitives::BlockHeader;
use zk_pow::v4::api::public_params::{
    CommonParams, Device, HashId, MoeParams, OperandParams, Quant,
};

use zk_pow::v1::api::proof as v1_proof;

use zk_pow::v2::api::proof::{
    MMAType, MiningConfiguration, MoEConfig, PeriodicPattern, PublicProofParams,
};

use cert_version::{
    generate_proof_for_cert_version, py_check_cert_version_eligible,
    verify_plain_proof_for_cert_version, verify_proof_for_cert_version,
};
use common::{py_pad_to_chunk_boundary, IncompleteBlockHeader, PyProof};
use v1::{
    clear_v1_circuit_cache, generate_proof_v1, verify_plain_proof_v1, verify_proof_v1,
    warmup_prove_v1,
};
use v2::{
    clear_circuit_cache_v2, generate_proof_v2, generate_proof_v3, mine, mine_moe,
    penalized_target_bound, verify_plain_proof_v2, verify_plain_proof_v3, verify_proof_v2,
    verify_proof_v3, warmup_prove_v2,
};
use v4::verify_plain_proof_v4;

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
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_thread_count())
        .build_global()
        .expect("Failed to initialize rayon global thread pool");
    m.add("MERKLE_LEAF_SIZE", CHUNK_LEN)?;
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
    m.add_class::<MerkleTree>()?;
    m.add_class::<MerkleProof>()?;
    m.add_class::<PeriodicPattern>()?;
    m.add_class::<IncompleteBlockHeader>()?;
    m.add_class::<BlockHeader>()?;
    m.add_class::<MiningConfiguration>()?;
    m.add_class::<MoEConfig>()?;
    m.add_class::<MMAType>()?;
    m.add_class::<MatrixMerkleProof>()?;
    m.add_class::<PlainProof>()?;
    m.add_class::<PlainProofV4>()?;
    m.add_class::<MoeWitness>()?;
    m.add_class::<CommonParams>()?;
    m.add_class::<OperandParams>()?;
    m.add_class::<MoeParams>()?;
    m.add_class::<HashId>()?;
    m.add_class::<Quant>()?;
    m.add_class::<DimType>()?;
    m.add_class::<AxisPattern>()?;
    m.add_class::<Device>()?;
    m.add_class::<MoEProofParams>()?;
    m.add_class::<PyProof>()?;
    m.add_function(wrap_pyfunction!(mine, m)?)?;
    m.add_function(wrap_pyfunction!(mine_moe, m)?)?;
    m.add_function(wrap_pyfunction!(penalized_target_bound, m)?)?;
    m.add_function(wrap_pyfunction!(py_pad_to_chunk_boundary, m)?)?;
    // V2 functions (current circuit; MoE and dense proofs)
    m.add_function(wrap_pyfunction!(generate_proof_v2, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof_v2, m)?)?;
    m.add_function(wrap_pyfunction!(verify_plain_proof_v2, m)?)?;
    m.add_function(wrap_pyfunction!(clear_circuit_cache_v2, m)?)?;
    m.add_function(wrap_pyfunction!(warmup_prove_v2, m)?)?;
    // V3 functions (same circuits as V2; salted noise-seed derivation)
    m.add_function(wrap_pyfunction!(generate_proof_v3, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof_v3, m)?)?;
    m.add_function(wrap_pyfunction!(verify_plain_proof_v3, m)?)?;
    m.add_function(wrap_pyfunction!(verify_plain_proof_v4, m)?)?;
    m.add_class::<PyFp8Prover>()?;
    m.add_class::<PyFp8Verifier>()?;
    // V1 functions (legacy circuit; dense proofs only)
    m.add(
        "V1_PUBLICDATA_SIZE",
        v1_proof::PublicProofParams::PUBLICDATA_SIZE,
    )?;
    m.add_function(wrap_pyfunction!(generate_proof_v1, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof_v1, m)?)?;
    m.add_function(wrap_pyfunction!(verify_plain_proof_v1, m)?)?;
    m.add_function(wrap_pyfunction!(warmup_prove_v1, m)?)?;
    m.add_function(wrap_pyfunction!(clear_v1_circuit_cache, m)?)?;
    // Certificate-version dispatchers (recommended entry points)
    m.add("CERT_VERSION_ZK_DENSE", CertificateVersion::ZkDense as u32)?;
    m.add("CERT_VERSION_ZK_MOE", CertificateVersion::ZkMoe as u32)?;
    m.add("CERT_VERSION_ZK_V3", CertificateVersion::ZkV3 as u32)?;
    m.add(
        "CERT_VERSION_PLAIN_FP8",
        CertificateVersion::PlainFp8 as u32,
    )?;
    m.add_function(wrap_pyfunction!(py_check_cert_version_eligible, m)?)?;
    m.add_function(wrap_pyfunction!(generate_proof_for_cert_version, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof_for_cert_version, m)?)?;
    m.add_function(wrap_pyfunction!(verify_plain_proof_for_cert_version, m)?)?;
    Ok(())
}
