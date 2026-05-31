# pearl-c-driver SKELETON Makefile
#
# Builds against the pre-built pearl_gemm_w19_cuda.so .so file
# (the same .so that the Python driver loads). All hot-path symbols we call
# are exposed as C linkage; CUTLASS / libtorch are NOT linked here.
#
# Discoverable extern "C" symbols (verified with `nm -gD ...`):
#   pearl_gemm_sm89_w19r_64x64x128_R128_prod      <-- this skeleton uses this
#   pearl_gemm_sm89_pow_64x64x128_R128
#   pearl_gemm_sm89_w19r_64x64x128_R128_prod_cohort
#   pearl_gemm_sm89_w19r_64x64x128_R128_prod_regres
#
# Layout assumptions (CPU01 docker build container `pearl-build-hostnet`):
#   PEARL_GEMM_DIR  = /host_home/pearl-deploy/pearl-gemm
#   CUTLASS_DIR     = $(PEARL_GEMM_DIR)/third_party/cutlass
#   .so location    = $(PEARL_GEMM_DIR)/src/pearl_gemm_w19_cuda.cpython-312-x86_64-linux-gnu.so

PEARL_GEMM_DIR ?= /host_home/pearl-deploy/pearl-gemm
CUTLASS_DIR    ?= $(PEARL_GEMM_DIR)/third_party/cutlass
PEARL_GEMM_SO  ?= $(PEARL_GEMM_DIR)/src/pearl_gemm_w19_cuda.cpython-312-x86_64-linux-gnu.so
CUDA_HOME      ?= /usr/local/cuda

NVCC   ?= $(CUDA_HOME)/bin/nvcc
NVCC_ARCH ?= -gencode=arch=compute_89,code=sm_89

# We only need CUTLASS headers (cutlass/numeric_types.h, etc.) to satisfy
# the C++ types in the extern "C" signatures. No libcutlass linkage needed.
INCLUDES = -I$(CUTLASS_DIR)/include \
           -I$(CUTLASS_DIR)/tools/util/include \
           -I$(CUDA_HOME)/include

NVCCFLAGS = -O3 -std=c++17 -Xcompiler -O3,-march=native,-fPIC \
            $(NVCC_ARCH) $(INCLUDES) --expt-relaxed-constexpr \
            -Xcudafe --diag_suppress=20012

# Link directly against the prebuilt .so (provides the extern "C" trampolines).
# We use a wildcard so the toolchain finds whatever .so version is present.
LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lcuda $(PEARL_GEMM_SO) \
          -Xlinker -rpath -Xlinker $(dir $(PEARL_GEMM_SO)) \
          -Xlinker --unresolved-symbols=ignore-in-shared-libs

SRCS = src/main.cu
BIN  = pearl_c_driver

all: $(BIN)

$(BIN): $(SRCS) Makefile
	$(NVCC) $(NVCCFLAGS) $(SRCS) $(LDFLAGS) -o $@

clean:
	rm -f $(BIN)

.PHONY: all clean
