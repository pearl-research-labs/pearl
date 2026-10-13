use zk_pow::v2::circuit::circuit_utils::CircuitCache;

/// Test cache serialization roundtrip using embedded CACHE_DATA.
/// Only runs when embedded_cache feature is enabled (requires cache.bin to exist).
#[test]
#[cfg(feature = "embedded_cache")]
fn test_cache_serialization_roundtrip() {
    use zk_pow::v2::circuit::embedded_cache::CACHE_DATA;

    // Load cache from embedded CACHE_DATA
    let cache = CircuitCache::from_bytes(CACHE_DATA).expect("Failed to load embedded cache");

    let original_first_count = cache.verifier_circuits_1.len();
    let original_second_count = cache.verifier_circuits_2.len();

    assert!(original_first_count > 0, "Should have generated first circuits");
    assert!(original_second_count > 0, "Should have generated second circuits");
    // Serialize
    let bytes = cache.to_bytes().expect("Serialization failed");
    assert!(bytes.len() > 8, "Serialized data should be more than just header");
    // Deserialize
    let restored_cache = CircuitCache::from_bytes(&bytes).expect("Deserialization failed");
    let bytes2 = restored_cache.to_bytes().expect("Serialization failed");
    assert_eq!(bytes, bytes2, "Serialized data should be the same");
}

#[test]
fn test_empty_cache_serialization() {
    let cache = CircuitCache::default();

    // Serialize empty cache
    let bytes = cache.to_bytes().expect("Failed to serialize empty cache");
    assert_eq!(bytes.len(), 12, "Empty cache should be 12 bytes (magic + two u32 zeros)");

    // Deserialize
    let restored = CircuitCache::from_bytes(&bytes).expect("Failed to deserialize empty cache");
    assert_eq!(restored.verifier_circuits_1.len(), 0);
    assert_eq!(restored.verifier_circuits_2.len(), 0);
}
