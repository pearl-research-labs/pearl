//! V4 (FP8) Go FFI: ZK certificate verification ([`verify`]), plain-proof share validation
//! ([`plain`]), and the read-only verifier cache the former uses.

use zk_pow::v4::api::zk::Fp8VerifierCache;

pub mod plain;
pub mod verify;

// Keep the previously generated C constant names stable. Literal values let
// cbindgen emit them without depending on which core types it discovers.
pub const V4_PUBLIC_PARAMS_WIRE_SIZE: usize = 244;
pub const V4_BLOCK_HEADER_SERIALIZED_SIZE: usize = 108;
pub const V4_MOE_PARAMS_MAX_NUM_EXPERTS: u16 = 1024;
pub const V4_AXIS_PATTERN_NUM_DIMS: usize = 6;

const _: () = assert!(V4_PUBLIC_PARAMS_WIRE_SIZE == zk_pow::v4::api::public_params::PublicParams::WIRE_SIZE);
const _: () = assert!(V4_BLOCK_HEADER_SERIALIZED_SIZE == zk_pow::v4::api::primitives::BlockHeader::SERIALIZED_SIZE);
const _: () = assert!(V4_MOE_PARAMS_MAX_NUM_EXPERTS == zk_pow::v4::api::public_params::MoeParams::MAX_NUM_EXPERTS);
const _: () = assert!(V4_AXIS_PATTERN_NUM_DIMS == zk_pow::v4::api::layout::AxisPattern::NUM_DIMS);

lazy_static::lazy_static! {
    /// FP8 verifier cache: the trusted setup (LUT cap + universal wrapper circuits),
    /// keyed by device byte and preloaded from the embedded `fp8_cache.bin`. Read-only —
    /// a device missing from the cache rejects the proof rather than compiling its setup
    /// on demand, so no proof can force an expensive circuit build (denial of service).
    pub static ref FP8_VERIFIER_CACHE: Fp8VerifierCache = {
        use zk_pow::v4::api::embedded_cache;
        let cache = Fp8VerifierCache::from_bytes(embedded_cache::CACHE_DATA)
            .expect("fp8 verifier cache is missing or corrupt; cannot verify fp8 proofs");
        assert!(
            cache.contains_all_devices(),
            "fp8 verifier cache must contain exactly the H100 and B200 setups"
        );
        cache
    };
}

/// The fp8 verifier cache: read-only, so no lock — concurrent verifications share it.
pub(crate) fn fp8_cache() -> &'static Fp8VerifierCache {
    &FP8_VERIFIER_CACHE
}
