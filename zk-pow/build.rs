fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "embedded_cache")]
    {
        println!("cargo:rerun-if-changed=src/v4/api/v4_cache.bin");
        let cache = std::fs::read("src/v4/api/v4_cache.bin").expect("missing V4 verifier cache; run build_cache");
        assert_eq!(
            &blake3::hash(&cache).to_hex().as_str()[..32],
            "2bc22ffd6e9fccff0638e6839035e1c1",
            "V4 verifier cache digest mismatch"
        );
    }
}
