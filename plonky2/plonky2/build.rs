// Compiles the GPU `from_values` FFI when the `gpu_commit` feature is on.
// Requires nvcc (CUDA). On hosts without CUDA, leave the feature off.
fn main() {
    println!("cargo:rerun-if-changed=src/gpu/pearl_commit.cu");
    println!("cargo:rerun-if-changed=src/gpu/pearl_commit_kernels.cuh");
    if std::env::var("CARGO_FEATURE_GPU_COMMIT").is_err() {
        return;
    }
    let out = std::env::var("OUT_DIR").unwrap();
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".to_string());
    let obj = format!("{out}/pearl_commit.o");
    let lib = format!("{out}/libpearlcommit.a");
    let st = std::process::Command::new(&nvcc)
        .args([
            "-O3", "-arch=sm_86", "-std=c++17", "-Xcompiler", "-fPIC",
            "-c", "src/gpu/pearl_commit.cu", "-o", &obj,
        ])
        .status()
        .expect("run nvcc");
    assert!(st.success(), "nvcc failed");
    let st = std::process::Command::new("ar").args(["crus", &lib, &obj]).status().expect("ar");
    assert!(st.success(), "ar failed");

    let cuda_lib = std::env::var("CUDA_LIB").unwrap_or_else(|_| "/usr/local/cuda/lib64".to_string());
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=pearlcommit");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
