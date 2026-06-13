import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import torch
from packaging.version import Version, parse
from pearl_gemm_build_utils.generate_instantiations import generate_instantiations
from pearl_gemm_build_utils.write_static_switches import (
    write_matmul_switch,
    write_noising_a_switch,
    write_noising_b_switch,
)
from setuptools import setup

# For arch-dependent _write_ninja_file hack below
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CUDAExtension,
    _is_cuda_file,
    _join_cuda_home,
    _maybe_write,
    get_cxx_compiler,
)
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# ninja build does not work unless include_dirs are abs path
ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / "csrc"
GEMM_DIR = CSRC_DIR / "gemm"

PACKAGE_NAME = "pearl_gemm"

with open(ROOT_DIR / "pyproject.toml", "rb") as _f:
    PACKAGE_VERSION = tomllib.load(_f)["project"]["version"]

BASE_WHEEL_URL = ""


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).casefold() in ("t", "true", "1", "y", "yes")


# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist`
# to copy over raw files, without any cuda compilation
FORCE_BUILD = _env_flag("PEARL_GEMM_FORCE_BUILD", "TRUE")
SKIP_CUDA_BUILD = _env_flag("PEARL_GEMM_SKIP_CUDA_BUILD", "FALSE")
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = _env_flag("PEARL_GEMM_FORCE_CXX11_ABI", "FALSE")

DISABLE_SKIP_REDUCTION = _env_flag("PEARL_GEMM_DISABLE_SKIP_REDUCTION", "FALSE")
DISABLE_SKIP_DENOISING = _env_flag("PEARL_GEMM_DISABLE_SKIP_DENOISING", "FALSE")
DISABLE_DEBUG_MODE = _env_flag("PEARL_GEMM_DISABLE_DEBUG_MODE", "FALSE")
DISABLE_R32 = _env_flag("PEARL_GEMM_DISABLE_R32", "TRUE")
DISABLE_R64 = _env_flag("PEARL_GEMM_DISABLE_R64", "FALSE")
DISABLE_R128 = _env_flag("PEARL_GEMM_DISABLE_R128", "FALSE")

SKIP_CPP_GENERATION = _env_flag("PEARL_GEMM_SKIP_CPP_GENERATION", "FALSE")

FEATURE_FLAGS = {
    "DISABLE_SKIP_REDUCTION": DISABLE_SKIP_REDUCTION,
    "DISABLE_SKIP_DENOISING": DISABLE_SKIP_DENOISING,
    "DISABLE_DEBUG_MODE": DISABLE_DEBUG_MODE,
    "DISABLE_R32": DISABLE_R32,
    "DISABLE_R64": DISABLE_R64,
    "DISABLE_R128": DISABLE_R128,
}

R_VALUE_TOGGLES = {32: DISABLE_R32, 64: DISABLE_R64, 128: DISABLE_R128}
ENABLED_R_VALUES = [r for r, disabled in R_VALUE_TOGGLES.items() if not disabled]

OUTPUT_TYPES = ["bf16"]

RAM_PER_JOB_GB = 6  # Conservative estimate
CORES_PER_JOB = 1
FALLBACK_MAX_JOBS = 4
KB_PER_GB = 1024 * 1024
NVCC_THREAD_COUNT = "4"
COMPUTE_CAPABILITY = "arch=compute_90a,code=sm_90a"


def linux_total_ram_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if not line.startswith("MemAvailable:"):
                continue

            match = re.search(r"MemAvailable:\s+(\d+)\s+kB", line)
            if not match:
                raise RuntimeError(f"Could not parse MemAvailable line: {line.strip()}")
            return int(match.group(1))
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def available_cpu_count() -> int:
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count()
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    if not (count := os.cpu_count()):
        raise RuntimeError("Could not determine CPU core count")
    return count


def smart_max_jobs(cores: int = available_cpu_count(), ram_kb: int = linux_total_ram_kb()) -> int:
    try:
        return int(min(cores // CORES_PER_JOB, ram_kb // (RAM_PER_JOB_GB * KB_PER_GB)))
    except Exception as e:
        warnings.warn(
            f"smart_max_jobs: falling back to {FALLBACK_MAX_JOBS} due to error: {e!r}",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return FALLBACK_MAX_JOBS


os.environ["MAX_JOBS"] = os.environ.get("MAX_JOBS", str(smart_max_jobs()))

config_module = importlib.import_module(
    "pearl_gemm_build_utils.kernel_configs.default_compiled_kernels"
)

# Config file and env variables jointly control which kernels are compiled
kernel_configs = config_module.KERNEL_CONFIGS

MATMUL_KERNELS = [k for k in kernel_configs.matmul_kernels if k.R in ENABLED_R_VALUES]
NOISING_A_KERNELS = [k for k in kernel_configs.noising_a_kernels if k.R in ENABLED_R_VALUES]
NOISING_B_KERNELS = [k for k in kernel_configs.noising_b_kernels if k.R in ENABLED_R_VALUES]


def get_platform() -> str:
    """Returns the platform name as used in wheel filenames."""
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    raise ValueError(f"Unsupported platform: {sys.platform}")


def get_cuda_bare_metal_version(cuda_dir: str) -> tuple[str, Version]:
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version


def warn_if_cuda_home_missing(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # Warn instead of error: user could be downloading prebuilt wheels where nvcc isn't necessary
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc.",
        stacklevel=2,
    )


def append_nvcc_threads(nvcc_extra_args: list[str]) -> list[str]:
    return nvcc_extra_args + ["--threads", NVCC_THREAD_COUNT]


# HACK from FA3: monkey-patch pytorch's _write_ninja_file


def _ninja_escape(path: str) -> str:
    """Escape spaces for ninja syntax."""
    return path.replace(" ", "$ ")


def _build_ninja_config(
    compiler: str, with_cuda: bool, cuda_dlink_post_cflags: list[str]
) -> list[str]:
    """Build the ninja config block (version, compilers)."""
    # Version 1.3 is required for the `deps` directive
    config = [
        "ninja_required_version = 1.3",
        f"cxx = {compiler}",
    ]
    if with_cuda or cuda_dlink_post_cflags:
        nvcc = _join_cuda_home("bin", "nvcc")
        nvcc_from_env = os.getenv("PYTORCH_NVCC", nvcc)
        config += [f"nvcc_from_env = {nvcc_from_env}", f"nvcc = {nvcc}"]
    return config


def _build_ninja_flags(
    cflags: list[str],
    post_cflags: list[str],
    cuda_cflags: list[str],
    cuda_post_cflags: list[str],
    cuda_dlink_post_cflags: list[str],
    ldflags: list[str],
    with_cuda: bool,
) -> list[str]:
    """Build the ninja variable-assignment block for all flag sets."""
    flags = [
        f"cflags = {' '.join(cflags)}",
        f"post_cflags = {' '.join(post_cflags)}",
    ]
    if with_cuda:
        flags += [
            f"cuda_cflags = {' '.join(cuda_cflags)}",
            f"cuda_post_cflags = {' '.join(cuda_post_cflags)}",
        ]
    flags += [
        f"cuda_dlink_post_cflags = {' '.join(cuda_dlink_post_cflags)}",
        f"ldflags = {' '.join(ldflags)}",
    ]
    return flags


def _build_cuda_compile_rule() -> list[str]:
    """Build the ninja cuda_compile rule, with optional dependency generation."""
    rule = ["rule cuda_compile"]
    nvcc_gendeps = ""
    if (
        torch.version.cuda is not None
        and os.getenv("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "0") != "1"
    ):
        rule += ["  depfile = $out.d", "  deps = gcc"]
        nvcc_gendeps = "--generate-dependencies-with-compile --dependency-output $out.d"
    rule.append(
        f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags"
    )
    return rule


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    cuda_dlink_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_cuda,
    **kwargs,
) -> None:
    """Write a ninja build file for compiling and linking CUDA extensions.

    See https://ninja-build.org/build.ninja.html for ninja syntax reference.
    """

    def _normalize_flags(flags):
        return [f.strip() for f in flags] if flags else []

    cflags = _normalize_flags(cflags)
    post_cflags = _normalize_flags(post_cflags)
    cuda_cflags = _normalize_flags(cuda_cflags)
    cuda_post_cflags = _normalize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = _normalize_flags(cuda_dlink_post_cflags)
    ldflags = _normalize_flags(ldflags)

    assert len(sources) == len(objects) > 0

    sources = [os.path.abspath(f) for f in sources]

    config = _build_ninja_config(get_cxx_compiler(), with_cuda, cuda_dlink_post_cflags)
    flags = _build_ninja_flags(
        cflags,
        post_cflags,
        cuda_cflags,
        cuda_post_cflags,
        cuda_dlink_post_cflags,
        ldflags,
        with_cuda,
    )

    compile_rule = [
        "rule compile",
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
    ]

    cuda_compile_rule = _build_cuda_compile_rule() if with_cuda else []

    build = []
    for src, obj in zip(sources, objects, strict=True):
        rule = "cuda_compile" if _is_cuda_file(src) and with_cuda else "compile"
        build.append(f"build {_ninja_escape(obj)}: {rule} {_ninja_escape(src)}")

    devlink_rule, devlink = [], []
    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), "dlink.o")
        devlink_rule = [
            "rule cuda_devlink",
            "  command = $nvcc $in -o $out $cuda_dlink_post_cflags",
        ]
        devlink = [f"build {devlink_out}: cuda_devlink {' '.join(objects)}"]
        objects += [devlink_out]

    link_rule, link, default = [], [], []
    if library_target is not None:
        link_rule = [
            "rule link",
            "  command = $cxx $in $ldflags -o $out",
        ]
        link = [f"build {library_target}: link {' '.join(objects)}"]
        default = [f"default {library_target}"]

    blocks = [
        config,
        flags,
        compile_rule,
        cuda_compile_rule,
        devlink_rule,
        link_rule,
        build,
        devlink,
        link,
        default,
    ]
    content = "\n\n".join("\n".join(b) for b in blocks) + "\n"
    _maybe_write(path, content)


torch.utils.cpp_extension._write_ninja_file = _write_ninja_file

cmdclass = {}
ext_modules = []

# Needed even when SKIP_CUDA_BUILD so that sdist includes .hpp files for source compilation
cutlass_dir = ROOT_DIR / "third_party" / "cutlass"
try:
    subprocess.run(["git", "submodule", "update", "--init", str(cutlass_dir)], check=True)
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    print(f"Warning: Could not initialize git submodules: {e}")
    print("This may be expected in containerized environments or when git is not available.")

assert os.path.exists(cutlass_dir), f"cutlass_dir {cutlass_dir} does not exist"

if not SKIP_CUDA_BUILD:
    print(f"\n\ntorch.__version__  = {torch.__version__}\n\n")

    warn_if_cuda_home_missing("pearl_gemm")
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    print(f"cuda version = {bare_metal_version}\n\n")
    arch_flags = ["-gencode", COMPUTE_CAPABILITY]

    if not SKIP_CPP_GENERATION:
        # Generate template instantiations for all possible kernels
        instantiations_dir = GEMM_DIR / "instantiations"
        print(f"Writing template instantiations to {instantiations_dir}")
        generate_instantiations(
            MATMUL_KERNELS, NOISING_A_KERNELS, NOISING_B_KERNELS, instantiations_dir
        )

        print(f"Writing static switches to {GEMM_DIR}")
        write_matmul_switch(GEMM_DIR / "static_switch_matmul.h", MATMUL_KERNELS)
        write_noising_a_switch(GEMM_DIR / "static_switch_noisingA.h", NOISING_A_KERNELS)
        write_noising_b_switch(GEMM_DIR / "static_switch_noisingB.h", NOISING_B_KERNELS)

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    sources = [
        "csrc/gemm/pearl_gemm_api.cpp",
        "csrc/gemm/noise_generation.cu",
        "csrc/gemm/denoise_converter.cu",
        "csrc/gemm/inner_hash_kernel.cu",
        "csrc/gemm/quantize_kernel.cu",
        "csrc/blake3/blake3.cu",
        "csrc/tensor_hash/tensor_hash.cu",
        "csrc/moe/build_routing_data.cu",
    ]
    sources.extend(
        f"csrc/gemm/instantiations/gemm_R{cfg.R}_{out_type}_{cfg.tile_size_m}x{cfg.tile_size_n}x{cfg.tile_size_k}_{cfg.pipeline_stages}stages_cluster{cfg.cM}x{cfg.cN}.cu"
        for cfg in MATMUL_KERNELS
        for out_type in OUTPUT_TYPES
    )
    sources.extend(
        f"csrc/gemm/instantiations/noisingA_R{cfg.R}_{cfg.AxEBL_type}_{cfg.tile_size_m}x{cfg.tile_size_k}_{cfg.pipeline_stages}stages.cu"
        for cfg in NOISING_A_KERNELS
    )
    sources.extend(
        f"csrc/gemm/instantiations/noisingB_R{cfg.R}_{cfg.EARxBpEB_type}_{cfg.tile_size_n}x{cfg.tile_size_k}_{cfg.pipeline_stages}stages.cu"
        for cfg in NOISING_B_KERNELS
    )

    feature_args = [f"-D{name}" for name, enabled in FEATURE_FLAGS.items() if enabled]

    gcc_flags = [
        "-O3",
        "-std=c++20",
        "-fvisibility=hidden",  # silence some pybind11 warnings
    ]

    nvcc_flags = [
        "-O3",
        "-std=c++20",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
    ]
    include_dirs = [
        CSRC_DIR,
        cutlass_dir / "include",
        cutlass_dir / "examples" / "common",
        cutlass_dir / "tools" / "util" / "include",
    ]
    if sys.platform.startswith("linux"):
        cuda_cccl_include_dir = (
            Path(CUDA_HOME) / "targets" / f"{platform.machine()}-linux" / "include" / "cccl"
        )
        if cuda_cccl_include_dir.exists():
            include_dirs.append(cuda_cccl_include_dir)

    # Get PyTorch library path for rpath
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

    ext_modules.append(
        CUDAExtension(
            name="pearl_gemm_cuda",
            sources=sources,
            extra_compile_args={
                "cxx": gcc_flags + feature_args,
                "nvcc": append_nvcc_threads(nvcc_flags + arch_flags + feature_args),
            },
            extra_link_args=[f"-Wl,-rpath,{torch_lib_path}", "-Wl,-rpath,$ORIGIN"],
            include_dirs=include_dirs,
            # Without this we get an error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"],
        )
    )


def get_wheel_url() -> tuple[str, str]:
    torch_cuda_version = parse(torch.version.cuda)
    torch_version_raw = parse(torch.__version__)
    MIN_CUDA_VERSION = parse("13.0")
    if torch_cuda_version < MIN_CUDA_VERSION:
        raise RuntimeError(
            f"CUDA >= {MIN_CUDA_VERSION} is required, but torch was built with CUDA {torch_cuda_version}"
        )
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    package_version = PACKAGE_VERSION
    cuda_version = f"{torch_cuda_version.major}{torch_cuda_version.minor}"
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    wheel_filename = f"{PACKAGE_NAME}-{package_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{package_version}", wheel_name=wheel_filename)
    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if self.dist_dir and not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            if self.dist_dir:
                wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
                print("Raw wheel path", wheel_path)
                shutil.move(wheel_filename, wheel_path)
        except urllib.error.HTTPError:
            print("Precompiled wheel not found. Building from source...")
            super().run()


setup(
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
)
