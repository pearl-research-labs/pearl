//! Unified Python module for Pearl mining.
//!
//! Registers types from pearl-blake3 and zk-pow into a single Python module.
//! No wrapper types -- all #[pyclass] types are defined in their respective core crates.
//!
//! Each `vN` module holds one proof version's functions (`v2` also serves cert v3: same
//! circuits, salted noise seed). Version modules never import each other; they share only
//! [`common`]. [`cert_version`] dispatches across versions. Everything Python sees is
//! registered by those modules through [`pearl_mining`].

#[cfg(unix)]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod cert_version;
mod common;
mod v1;
mod v2;
mod v4;
mod v5;

use pyo3::prelude::*;

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
    // Keep the original registration order, including shared types and helpers.
    common::register_constants(m)?;
    v2::register_constants(m)?;
    common::register_merkle_types(m)?;
    v2::register_pattern(m)?;
    common::register_header(m)?;
    v4::register_header(m)?;
    v2::register_types(m)?;
    v4::register_types(m)?;
    v2::register_moe_proof_params(m)?;
    common::register_proof_type(m)?;
    v2::register_mining(m)?;
    common::register_functions(m)?;
    v2::register_proofs(m)?;
    v4::register_proofs(m)?;
    v5::register_types(m)?;
    v5::register_proofs(m)?;
    v1::register(m)?;
    cert_version::register(m)?;
    Ok(())
}
