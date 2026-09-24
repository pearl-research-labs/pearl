//! FFI utilities shared between Go bindings (zk_pow_ffi) and Python bindings (zk-pow-python).
//!
//! This module contains the core data structures and utilities that are used by both
//! the Go FFI layer and the Python bindings, eliminating code duplication.

pub mod plain_proof;
pub mod pybind;
