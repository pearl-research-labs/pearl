// Compiles the GPU `from_values` FFI when the `gpu_commit` feature is on, and the
// GPU `compute_quotient_polys` FFI when the `gpu_quotient` feature is on.
// Requires nvcc (CUDA). On hosts without CUDA, leave the features off.
fn main() {
    println!("cargo:rerun-if-changed=src/gpu/pearl_commit.cu");
    println!("cargo:rerun-if-changed=src/gpu/pearl_commit_kernels.cuh");
    println!("cargo:rerun-if-changed=src/gpu/pearl_quotient.cu");
    println!("cargo:rerun-if-changed=src/gpu/pearl_fri.cu");
    println!("cargo:rerun-if-changed=src/gpu/recursion_prover.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_LIB");
    println!("cargo:rerun-if-env-changed=PEARL_QUOTIENT_ICE_ZK");
    println!("cargo:rerun-if-env-changed=PEARL_QUOTIENT_ICE_PROVER");

    let gpu_commit = std::env::var("CARGO_FEATURE_GPU_COMMIT").is_ok();
    let gpu_quotient = std::env::var("CARGO_FEATURE_GPU_QUOTIENT").is_ok();
    if !gpu_commit && !gpu_quotient {
        return;
    }

    let out = std::env::var("OUT_DIR").unwrap();
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".to_string());

    // ── GPU commit FFI (libpearlcommit.a) ───────────────────────────────────────
    // gpu_quotient depends on gpu_commit (Cargo feature), so this always runs when
    // gpu_quotient is on; pearl_commit.cu provides poseidon_upload_constants() etc.
    if gpu_commit {
        let obj = format!("{out}/pearl_commit.o");
        let lib = format!("{out}/libpearlcommit.a");
        let st = std::process::Command::new(&nvcc)
            .args([
                "-O3",
                "-arch=sm_86",
                "-std=c++17",
                "-Xcompiler",
                "-fPIC",
                "-c",
                "src/gpu/pearl_commit.cu",
                "-o",
                &obj,
            ])
            .status()
            .expect("run nvcc (pearl_commit)");
        assert!(st.success(), "nvcc failed (pearl_commit)");
        let st = std::process::Command::new("ar")
            .args(["crus", &lib, &obj])
            .status()
            .expect("ar (pearl_commit)");
        assert!(st.success(), "ar failed (pearl_commit)");
        println!("cargo:rustc-link-search=native={out}");
        println!("cargo:rustc-link-lib=static=pearlcommit");
    }

    // ── GPU quotient FFI (libpearlquotient.a) ───────────────────────────────────
    // Mirrors the commit block. The two -I dirs come from env so they are not
    // hardcoded; zk-dir FIRST (see pearl_quotient.cu / recursion_quotient.cuh note).
    if gpu_quotient {
        let ice_zk = std::env::var("PEARL_QUOTIENT_ICE_ZK").expect(
            "gpu_quotient: set PEARL_QUOTIENT_ICE_ZK to the worktree \
             gpu-miner/src/algos/pearl_pouw/zk dir (holds the gate_*.cuh + goldilocks.cuh)",
        );
        let ice_prover = std::env::var("PEARL_QUOTIENT_ICE_PROVER").expect(
            "gpu_quotient: set PEARL_QUOTIENT_ICE_PROVER to the worktree \
             gpu-miner/pearl-gpu-prover/cuda dir (holds recursion_quotient.cuh)",
        );
        let inc_zk = format!("-I{ice_zk}");
        let inc_prover = format!("-I{ice_prover}");
        let lib = format!("{out}/libpearlquotient.a");

        // Always start from a FRESH archive: `ar crus` ADDS/updates members but never removes
        // previously-archived objects, so a stale pearl_quotient.o / pearl_fri.o from an earlier
        // build would linger and re-introduce the __constant__ aliasing (the M2 stage-5 OUTCOME-B
        // bug). Removing the .a guarantees only the current TU(s) are archived.
        let _ = std::fs::remove_file(&lib);
        // The recursion C-ABIs live in ICE-worktree .cuh (pulled via -I); cargo's rerun-if-changed
        // does NOT track included headers, so editing a .cuh alone leaves a STALE .o. Watch the two
        // ICE dirs so any .cuh edit forces a re-nvcc.
        println!("cargo:rerun-if-changed={ice_zk}");
        println!("cargo:rerun-if-changed={ice_prover}");

        // Compile each FFI TU separately, then archive all into libpearlquotient.a.
        //   pearl_quotient.cu    → pearl_gpu_compute_quotient_f64 (recursion_quotient.cuh)
        //   pearl_fri.cu         → pearl_gpu_prove_openings_f64    (recursion_fri.cuh)
        //   recursion_prover.cu  → pearl_gpu_prove_rec1_f64        (recursion_prover.cuh)
        // The exported extern "C" symbols are disjoint, so no C-ABI collision. recursion_-
        // prover.cu pulls recursion_quotient.cuh + recursion_fri.cuh + perm_z.cuh, so it
        // carries VERBATIM copies of the rq_*/rf_*/k_permz_* file-scope __global__ kernels
        // that also live in pearl_quotient.o / pearl_fri.o. Those are byte-identical dups;
        // the final link tolerates them via the parent's RUSTFLAGS
        // `-C link-arg=-Wl,--allow-multiple-definition` (documented build invariant — the
        // fused-prover work requires this flag, same as the rest of gpu_quotient). The
        // __device__ __forceinline__ helpers (goldilocks/poseidon) are internal-linkage so
        // they never collide regardless.
        // SINGLE recursion TU: recursion_prover.cu #includes recursion_quotient.cuh +
        // recursion_fri.cuh, so it ALONE defines all three extern-C C-ABIs
        // (pearl_gpu_compute_quotient_f64 / pearl_gpu_prove_openings_f64 /
        // pearl_gpu_prove_rec1_f64) PLUS one copy each of the rq_*/rf_* kernels, the gate
        // __device__/__global__/__constant__, and the poseidon/coset upload fns. Compiling
        // pearl_quotient.cu / pearl_fri.cu too created DUPLICATE copies; under
        // --allow-multiple-definition the linker then bound the poseidon/coset upload fn and
        // the k_recursion_acc_lde launcher from DIFFERENT object modules, so the fused upload
        // wrote a different device module's __constant__ than the kernel read (M2 stage-5
        // OUTCOME-B: identical kernel args, garbage quotient). Building the single TU keeps the
        // upload + kernel + __constant__ in ONE module — no aliasing.
        for src in ["src/gpu/recursion_prover.cu"] {
            let stem = std::path::Path::new(src)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap();
            let obj = format!("{out}/{stem}.o");
            let st = std::process::Command::new(&nvcc)
                .args([
                    "-O3",
                    "-arch=sm_86",
                    "-std=c++17",
                    "-Xcompiler",
                    "-fPIC",
                    &inc_zk,     // zk-dir FIRST so its goldilocks.cuh wins the #pragma once
                    &inc_prover, // prover-dir for recursion_*.cuh + siblings
                    "-c",
                    src,
                    "-o",
                    &obj,
                ])
                .status()
                .unwrap_or_else(|e| panic!("run nvcc ({src}): {e}"));
            assert!(st.success(), "nvcc failed ({src})");
            let st = std::process::Command::new("ar")
                .args(["crus", &lib, &obj])
                .status()
                .unwrap_or_else(|e| panic!("ar ({src}): {e}"));
            assert!(st.success(), "ar failed ({src})");
        }
        println!("cargo:rustc-link-search=native={out}");
        println!("cargo:rustc-link-lib=static=pearlquotient");
    }

    // ── shared CUDA runtime link (cudart + stdc++) ──────────────────────────────
    let cuda_lib =
        std::env::var("CUDA_LIB").unwrap_or_else(|_| "/usr/local/cuda/lib64".to_string());
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
