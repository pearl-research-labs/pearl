#pragma once
// Torch-free stub for standalone (no-libtorch) sm_89/sm_120 builds.
// error_check.hpp includes this only for TORCH_CHECK, which the standalone
// translation unit defines itself before including any tensor_hash header.
