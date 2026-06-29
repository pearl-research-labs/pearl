// Build verifier cache binary files (v2 + v1).
// Usage: cargo run --release --bin build_cache <output_dir>
// Must be run before building the package with the embedded_cache feature.

use std::env;
use std::fs;
use std::path::PathBuf;
use zk_pow::circuit::circuit_utils::CircuitCache;
use zk_pow::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --release --bin build_cache <v2_output> [v1_output]");
        eprintln!("Example: cargo run --release --bin build_cache src/circuit/v2_cache.bin src/v1/v1_cache.bin");
        std::process::exit(1);
    }

    let main_output = PathBuf::from(&args[1]);

    // Build main (MoE) cache
    println!("Generating main verifier cache...");
    let mut main_cache = CircuitCache::default();
    PearlRecursion::fill_verifier_cache(&mut main_cache);

    println!(
        "Main cache: {} first circuits, {} second circuits",
        main_cache.verifier_circuits_1.len(),
        main_cache.verifier_circuits_2.len()
    );

    let main_data = main_cache.to_bytes().expect("Failed to serialize main cache");
    println!("Writing {} bytes to {:?}", main_data.len(), main_output);
    fs::write(&main_output, &main_data).expect("Failed to write main cache file");

    // Build V1 cache if output path provided
    if args.len() >= 3 {
        let v1_output = PathBuf::from(&args[2]);

        println!("\nGenerating V1 verifier cache...");
        use zk_pow::v1::circuit::circuit_utils::CircuitCache as V1Cache;
        use zk_pow::v1::circuit::pearl_circuit::{PearlRecursion as V1Recursion, RecursionCircuit as _};

        let mut v1_cache = V1Cache::default();
        V1Recursion::fill_verifier_cache(&mut v1_cache);

        println!(
            "V1 cache: {} first circuits, {} second circuits",
            v1_cache.verifier_circuits_1.len(),
            v1_cache.verifier_circuits_2.len()
        );

        let v1_data = v1_cache.to_bytes().expect("Failed to serialize V1 cache");
        println!("Writing {} bytes to {:?}", v1_data.len(), v1_output);
        fs::write(&v1_output, &v1_data).expect("Failed to write V1 cache file");
    }

    println!("\nCache generation complete!");
}
