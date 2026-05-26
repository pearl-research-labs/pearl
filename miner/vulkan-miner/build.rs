use std::fs;
use std::path::Path;

fn main() {
    let shader_dir = Path::new("shaders");
    let out_dir_str = std::env::var("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir_str);

    if !shader_dir.exists() {
        return;
    }

    let glslc_path = std::env::var("GLSLC_PATH").unwrap_or_else(|_| "glslc".to_string());

    collect_shaders(shader_dir, shader_dir, out_dir, &glslc_path);
}

fn collect_shaders(base: &Path, dir: &Path, out_dir: &Path, glslc: &str) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_dir() {
            collect_shaders(base, &path, out_dir, glslc);
        } else if path.extension().map_or(false, |e| e == "comp")
        {
            let rel = path.strip_prefix(base).unwrap();
            let out_path = out_dir.join(rel).with_extension("spv");
            fs::create_dir_all(out_path.parent().unwrap()).ok();

            let status = std::process::Command::new(glslc)
                .arg("-fshader-stage=compute")
                .arg("--target-env=vulkan1.3")
                .arg("-I")
                .arg(base)
                .arg("-Werror")
                .arg(&path)
                .arg("-o")
                .arg(&out_path)
                .status()
                .unwrap_or_else(|_| panic!("glslc not found (PATH={}); install Vulkan SDK or set GLSLC_PATH", glslc));

            assert!(status.success(), "Failed to compile {:?}", path);

            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
