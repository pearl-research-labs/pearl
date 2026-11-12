#![cfg(feature = "embedded_cache")]

use zk_pow::v4::api::embedded_cache::CACHE_DATA;

#[test]
fn embedded_v4_cache_matches_pinned_digest() {
    assert!(!CACHE_DATA.is_empty(), "embedded V4 verifier cache is empty");
}
