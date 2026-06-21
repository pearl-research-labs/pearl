import pearl_gemm_cuda
import torch

BLAKE3_DIGEST_SIZE_U32 = 8


def make_pow_target_tensor(value: int, device="cuda") -> torch.Tensor:
    """Create a pow_target tensor from a uint256 integer value."""

    result = torch.empty((BLAKE3_DIGEST_SIZE_U32,), dtype=torch.uint32, device=device)
    for i in range(BLAKE3_DIGEST_SIZE_U32):
        result[i] = value & 0xFFFFFFFF
        value >>= 32
    return result


def denoise_converter(
    EARxBpEB_in=None,  # (n, R), int32
    AxEBL_in=None,  # (m, R), int32
    EARxBpEB_out=None,  # (n, R), fp16
    AxEBL_out=None,  # (m, R), fp16
):
    """
    Provide either EARxBpEB_in/out or AxEBL_in/out (or both) to convert data from the
    "in" tensor(s) from int32 to fp16 and store results in the corresponding "out" tensor(s).
    """
    pearl_gemm_cuda.denoise_converter(
        EARxBpEB_in,
        AxEBL_in,
        EARxBpEB_out,
        AxEBL_out,
    )


# Fake Tensor function for torch.compile support
@torch.library.register_fake("pearl_gemm::denoise_converter")
def _abstract_denoise_converter(EARxBpEB_in=None, AxEBL_in=None, EARxBpEB_out=None, AxEBL_out=None):
    return None


# Fake Tensor function for torch.compile support
@torch.library.register_fake("pearl_gemm::noise_gen")
def _abstract_noise_gen(
    R,
    num_threads=64,
    EAL=None,  # (m, R), int8
    EAL_fp16=None,  # (m, R), float16
    EAR_R_major=None,  # (k, R), int8
    EAR_K_major=None,  # (R, k), int8
    EBL_R_major=None,  # (k, R), int8
    EBL_K_major=None,  # (R, k), int8
    EBR=None,  # (n, R), int8
    EBR_fp16=None,  # (n, R), float16
    key_A=None,  #   (32), uint8
    key_B=None,  #   (32), uint8
    aux_buffer=None,  # (S), int32 or uint32
):
    return None


def noise_gen(
    R=128,
    num_threads=64,
    EAL=None,  # (m, R), int8
    EAL_fp16=None,  # (m, R), float16
    EAR_R_major=None,  # (k, R), int8
    EAR_K_major=None,  # (R, k), int8
    EBL_R_major=None,  # (k, R), int8
    EBL_K_major=None,  # (R, k), int8
    EBR=None,  # (n, R), int8
    EBR_fp16=None,  # (n, R), float16
    key_A=None,  #   (32), uint8
    key_B=None,  #   (32), uint8
    aux_buffer=None,  # (S), int32 or uint32
):
    """
    Generate noise matrices E{A,B}{L,R} based on blake3 hash of seed and key from commitment kernel.

    Empty buffers for E{A,B}{L,R} must be provided with the indicated shapes.

    To *not* involve noise matrices, just skip that argument.
    For example, if EAR is already generated and you want to use seed/key for EAL only, skip EAR.

    For EAR and EBL (which are structured sparse matrices with one 1 and one -1 for each value of k, and
    0s elsewhere), this function can generate k x R matrices EAR_R_major, EBL_R_major; and their R x k
    transposes EAR_K_major, EBL_K_major. We generate transposes here to save some work in the noising
    kernels.

    EAL and EBR also have an option to write out in fp16 for use in denoising.

    Kernel always uses seed_X and key_X for E_X{L,R}.
    Generation of EAR from seed/key_B could be added, but we need to stipulate how to set hash parameters.

    Can pass in an optional aux_buffer to zero initialize (e.g., AxEBL when doing noise_A in inference mode).
    Size of aux_buffer must be divisible by 4 for 16 byte alignment.

    NOTE: Might want to support multiple aux_buffer in future.
    For now, arrange for fixed block in memory holding multiple tensors to zero init (e.g., AxEBL and EARxBpEB).
    """
    pearl_gemm_cuda.noise_gen(
        R,
        num_threads,
        EAL,
        EAL_fp16,
        EAR_R_major,
        EAR_K_major,
        EBL_R_major,
        EBL_K_major,
        EBR,
        EBR_fp16,
        key_A,
        key_B,
        aux_buffer,
    )


def noise_A(
    A,  # m x k
    EAL,  # m x r
    AxEBL,  # m x r
    ApEA,  # m x n
    EAR=None,  # k x r
    EBL=None,  # r x k
    tile_size_m: int | None = None,
    tile_size_k: int | None = None,
    pipeline_stages: int = 2,
    k_blocks_per_split: int | None = None,
):
    """Add noise to A and generate denoising component AxEBL.

    This computes the following three matrices:
    ApEA = (1/h)A + (EAL * EAR)
    AxEBL = s * A * EBL, used for denoising
    The scale factor s is 2^(-14) if AxEBL is fp16 and 1 if AxEBL is int32.

    Args:
        A (m x k, int8): Operand matrix, with entries in [-64, 63].
        EAL (m x r, int8): Noise factor created by noise_gen, with entries in [-64, 63].
        EAR (k x r, int8): Sparse matrix created by noise_gen. Each row is
            all 0s except for one 1 and one -1.
        EBL (r x k, int8): Sparse matrix created by noise_gen. Each column is
            all 0s except for one 1 and one -1.
        AxEBL (m x r, fp16 or int32): Empty tensor to store A * EBL. Should be
            zeroed out first if int32 (this can be done in noise gen).
        ApEA (m x k, int32): Zero tensor to store noised version of A.
        tile_size_m: Default 64. Must be a multiple of 64.
        tile_size_k: Default 64.
        pipeline_stages: Number of stages in the load and store pipelines.
            Default 2.
        k_blocks_per_split: If nonzero, each CTA computes this number of k-blocks of
            AxEBL and atomicAdds. This requires AxEBL in int32. Default value of None uses
            a heuristic to try to pick the best-performing value.
    """

    pearl_gemm_cuda.noise_A(
        A,
        EAL,
        AxEBL,
        ApEA,
        EAR,
        EBL,
        tile_size_m,
        tile_size_k,
        pipeline_stages,
        k_blocks_per_split,
    )


# Fake Tensor function for torch.compile support
@torch.library.register_fake("pearl_gemm::noise_A")
def _abstract_noise_a(
    A,
    EAL,
    AxEBL,
    ApEA,
    EAR=None,
    EBL=None,
    tile_size_m=None,
    tile_size_k=None,
    pipeline_stages=2,
    k_blocks_per_split=None,
):
    return None


def noise_B(
    B,  # n x k
    EBR,  # n x r
    EARxBpEB,  # n x r
    BpEB,  # n x k
    EAR=None,  # r x k
    EBL=None,  # k x r
    tile_size_n: int | None = None,
    tile_size_k: int | None = None,
    pipeline_stages: int = 2,
    k_blocks_per_split: int | None = None,
):
    """Add noise to B and generate denoising component EARxBpEB.

    This computes the following three matrices:
    BpEB = B + (EBL * EBR)
    EARxBpEB = EAR * (B + EBL * EBR) * t, used for denoising
    The scale factor t is 2^(-12) if EARxBpEB is fp16 and 1 if EARxBpEB is int32.

    Args:
        B (n x k, int8): Operand matrix, with entries in [-64, 63].
        EBR (n x r, int8): Noise factor created by noise_gen, with entries in [-64, 63].
        EAR (r x k, int8): Sparse matrix created by noise_gen. Each row is
            all 0s except for one 1 and one -1.
        EBL (k x r, int8): Sparse matrix created by noise_gen. Each column is
            all 0s except for one 1 and one -1.
        EARxBpEB (n x r, fp16 or int32): Empty tensor to store denoising factor.
            Should be zeroed out first if int32 (this can be done in noise gen).
        BpEB (n x k, int32): Zero tensor to store noised version of B.
        tile_size_n: Default 64.
        tile_size_k: Default 64.
        pipeline_stages: Number of stages in the load and store pipelines.
            Default 2.
        k_blocks_per_split: If nonzero, each CTA computes this number
            of k-blocks of EARxBpEB and atomicAdds. This requires EARxBpEB in int32. Default
            value of None uses a heuristic to try to pick the best-performing value.
    """
    pearl_gemm_cuda.noise_B(
        B,
        EBR,
        EARxBpEB,
        BpEB,
        EAR,
        EBL,
        tile_size_n,
        tile_size_k,
        pipeline_stages,
        k_blocks_per_split,
    )


# Fake Tensor function for torch.compile support
@torch.library.register_fake("pearl_gemm::noise_B")
def _abstract_noise_b(
    B,
    EBR,
    EARxBpEB,
    BpEB,
    EAR,
    EBL,
    tile_size_n=None,
    tile_size_k=None,
    pipeline_stages=2,
    k_blocks_per_split=None,
):
    return None


def gemm(
    A,  # m x k
    B,  # n x k
    A_scales,  # m
    B_scales,  # n
    C,  # m x n
    tile_size_m: int = 128,
    tile_size_n: int = 256,
    tile_size_k: int = 128,
    cluster_size_m: int = 1,
    cluster_size_n: int = 1,
    pipeline_stages: int | None = None,
    swizzle: int | None = None,
    swizzle_n_maj: bool = True,
):
    """Perform gemm without noising or denoising.

    Args:
        A (m x k, int8), B (n x k, int8): operands.
        A_scales (m, fp32), B_scales (n, fp32): scaling factors.
        C (m x n, bf16): Empty tensor for output.
        tile_size_m: Allowed values are [32, 64, 128, 192, 256].
        tile_size_n: Allowed values are [64, 128, 192, 256].
        cluster_size_m: Number of CTAs in cluster in M direction.
        cluster_size_n: Number of CTAs in cluster in N direction.
        pipeline_stages: The number of stages in the mainloop pipeline.
            If None, pick the largest number that fits in SMEM, up to 5.
        swizzle: We assign (tile_size_m x tile_size_n) output tiles to
            CTAs in strips of this size in the N direction. For example, if swizzle = 64,
            then all output tiles in the first (tile_size_n * 64) columns are assigned
            before moving to the next set of columns. The point is to continue reading
            tiles of B out of L2 cache.             If None, attempt to set swizzle size to maximize
            L2 hit rate, based on a heuristic.
    """
    pearl_gemm_cuda.gemm(
        A,
        B,
        A_scales,
        B_scales,
        C,
        tile_size_m,
        tile_size_n,
        tile_size_k,
        cluster_size_m,
        cluster_size_n,
        pipeline_stages,
        swizzle,
        swizzle_n_maj,
    )


# Fake Tensor function for torch.compile support
@torch.library.register_fake("pearl_gemm::gemm")
def _abstract_gemm(
    A,
    B,
    A_scales,
    B_scales,
    C,
    tile_size_m=128,
    tile_size_n=256,
    tile_size_k=128,
    cluster_size_m=1,
    cluster_size_n=1,
    pipeline_stages=None,
    swizzle=None,
    swizzle_n_maj=True,
):
    return None


def noisy_gemm(
    A,  # m x k
    B,  # n x k
    EAL,  # m x r
    EAL_fp16,  # m x r
    EBR,  # n x r
    EBR_fp16,
    EAR_R_major,  # k x r
    EBL_R_major,  # k x r
    EAR_K_major,  # r x k
    EBL_K_major,  # r x k
    AxEBL_fp16,  # m x r
    EARxBpEB_fp16,  # n x r
    ApEA,  # m x k
    BpEB,  # n x k
    A_scales,  # m
    B_scales,  # n
    C,  # m x n
    host_signal_header_pinned,  # host_signal_header_size
    host_signal_sync,  # host_signal_sync_size
    pow_target: torch.Tensor,  # (8,) uint32, PoW target (uint256, LE word order)
    pow_key: torch.Tensor,  # (8,) uint32, PoW key for keyed BLAKE3 hash
    AxEBL_int32=None,  # m x r
    EARxBpEB_int32=None,  # n x r
    tile_size_m: int = 128,
    tile_size_n: int = 256,
    tile_size_k: int = 128,
    cluster_size_m: int = 1,
    cluster_size_n: int = 1,
    pipeline_stages: int | None = None,
    swizzle: int | None = None,
    swizzle_n_maj: bool = True,
    tile_size_m_noising_A: int | None = None,
    tile_size_n_noising_B: int | None = None,
    tile_size_k_noising_A: int | None = None,
    tile_size_k_noising_B: int | None = None,
    pipeline_stages_noising_A: int = 2,
    pipeline_stages_noising_B: int = 2,
    k_blocks_per_split_noising_A: int | None = None,
    k_blocks_per_split_noising_B: int | None = None,
    run_noising_A: bool = True,
    run_noising_B: bool = True,
    skip_reduction: bool = False,
    skip_denoising: bool = False,
    inner_hash_counter: torch.Tensor | None = None,
    enable_debug: bool = False,
):
    """Perform noising, matmul, and denoising.

    This does calculations equivalent to the following:
    ApEA = A + (EAL * EAR)
    BpEB = B + (EBL * EBR)
    AxEBL = s * A * EBL
    EARxBpEB = EAR * (B + (EBL * EBR)) * t
    AxEB = s^(-1) * (AxEBL * EBR)
    EAxBpEB = (EAL * EARxBpEB) * t^(-1)
    ApEAxBpEB = ApEA * BpEB
    C = A_scales * (ApEAxBpEB - AxEB - EAxBpEB) * B_scales
    The scale factor s is 2^(-14) if AxEBL is fp16 and 1 if AxEBL is int32.
    The scale factor t is 2^(-12) if EARxBpEB is fp16 and 1 if EARxBpEB is int32.

    Args:
        A (m x k, int8): Operand matrix, with entries in [-64, 63].
        B (n x k, int8): Operand matrix, with entries in [-64, 63].
        EAL (m x r), EBR (n x r): int8 noise factors defined by noise_gen, with entries
            in [-64, 63]. Used only by noising kernels. The _fp16 versions of these are
            used by the main kernel for denoising.
        EAR_R_major, EBL_R_major (k x r, int8): Sparse matrices created by
            noise_gen. Each row is all 0s except for one 1 and one -1.
        EAR_K_major, EBL_K_major (r x k, int8): Sparse matrices created by
            noise_gen. Each column is all 0s except for one 1 and one -1. These should be
            the transposes of EAR_R_major and EBL_R_major.
        AxEBL_fp16 (m x r), EARxBpEB_fp16 (n x r): fp16 tensors for the main kernel's denoising step.
            They are either produced by the noising kernel or converted and stored here from the
            corresponding int32 tensors by the denoising converter kernel.
            denoising factors produced by noising kernels.
        AxEBL_int32 (m x r), EARxBpEB_int32 (n x r): optional int32 tensors, either of which
            can be produced by noising kernels.  The denoise converter kernel will convert either
            or both to fp16 for the main kernel's denoising step. Should be zeroed out first
            (this can be done in noise_gen).
        ApEA (m x k, int32), BpEB (n x k, int32): Zero tensors to store noised operands.
        A_scales (m, fp32), B_scales (n, fp32): scaling factors.
        C (m x n, bf16): Empty tensor for output.
        host_signal_header_pinned (128 * 2, uint8): Pinned memory tensor written to upon
            successful extraction. Currently a 2x128B header (see host_signal_header.hpp)
            with space to write A and B.
        host_signal_sync (8, uint8): Sync tensor for host signal.
        pow_target (8, uint32): PoW target as uint256 (8 x uint32). Little-endian word order
            (index 0 = LSW, index 7 = MSW). Hash must be <= target for extraction.
        pow_key (8, uint32): Key for keyed BLAKE3 hash in PoW check. Should be the same as
            key_A used in noise generation.
        tile_size_m: Allowed values are [32, 64, 128, 192, 256].
        tile_size_n: Allowed values are [64, 128, 192, 256].
        pipeline_stages: The number of stages in the mainloop pipeline.
            If None, pick the largest number that fits in SMEM, up to 5.
        cluster_size_m: Number of CTAs in cluster in M direction.
        cluster_size_n: Number of CTAs in cluster in N direction.
        swizzle: We assign (tile_size_m x tile_size_n) output tiles to
            CTAs in strips of this size in the N direction. For example, if swizzle = 64,
            then all output tiles in the first (tile_size_n * 64) columns are assigned
            before moving to the next set of columns. The point is to continue reading
            tiles of B out of L2 cache. If None, attempt to set swizzle size to maximize
            L2 hit rate, based on a heuristic.
        tile_size_m_noising_A: Default 64.
        tile_size_n_noising_B: Default 64.
        tile_size_k_noising_A: Default 64.
        tile_size_k_noising_B: Default 64.
        pipeline_stages_noising_A: Number of stages in the load and
            store pipelines for noisingA. Default 2.
        pipeline_stages_noising_B: Number of stages in the load and
            store pipelines for noisingB. Default 2.
        k_blocks_per_split_noising_A: If nonzero, use split-K for noisingA kernel with
            this many k-blocks per CTA. Requires AxEBL to be int32. Default value of None
            uses a heuristic to try to pick the best-performing value.
        k_blocks_per_split_noising_B: If nonzero, use split-K for noisingB kernel with
            this many k-blocks per CTA. Requires EARxBpEB to be int32. Default value of
            None uses a heuristic to try to pick the best-performing value.
        run_noising_A: If False, skip noise_A (default True).
        run_noising_B: If False, skip noise_B (default True).
        skip_reduction: Whether to disable the extraction step.
        skip_denoising: Whether to disable the denoising epilogue.
        inner_hash_counter: Optional tensor to count inner hashes (for testing/debugging).
        enable_debug: If True, enables debug mode for inner hash counting validation.
    """
    pearl_gemm_cuda.noisy_gemm(
        A,
        B,
        EAL,
        EAL_fp16,
        EBR,
        EBR_fp16,
        EAR_R_major,
        EBL_R_major,
        EAR_K_major,
        EBL_K_major,
        AxEBL_fp16,
        EARxBpEB_fp16,
        ApEA,
        BpEB,
        A_scales,
        B_scales,
        C,
        host_signal_header_pinned,
        host_signal_sync,
        pow_target,
        pow_key,
        AxEBL_int32,
        EARxBpEB_int32,
        tile_size_m,
        tile_size_n,
        tile_size_k,
        cluster_size_m,
        cluster_size_n,
        pipeline_stages,
        swizzle,
        swizzle_n_maj,
        tile_size_m_noising_A,
        tile_size_n_noising_B,
        tile_size_k_noising_A,
        tile_size_k_noising_B,
        pipeline_stages_noising_A,
        pipeline_stages_noising_B,
        k_blocks_per_split_noising_A,
        k_blocks_per_split_noising_B,
        run_noising_A,
        run_noising_B,
        skip_reduction,
        skip_denoising,
        inner_hash_counter,
        enable_debug,
    )


# Fake Tensor function for torch.compile support
@torch.library.register_fake("pearl_gemm::noisy_gemm")
def _abstract_noisy_gemm(
    A,
    B,
    EAL,
    EAL_fp16,
    EBR,
    EBR_fp16,
    EAR_R_major,
    EBL_R_major,
    EAR_K_major,
    EBL_K_major,
    AxEBL_fp16,
    EARxBpEB_fp16,
    ApEA,
    BpEB,
    A_scales,
    B_scales,
    C,
    host_signal_header_pinned,
    host_signal_sync,
    pow_target,
    pow_key,
    AxEBL_int32=None,
    EARxBpEB_int32=None,
    tile_size_m=128,
    tile_size_n=256,
    tile_size_k=128,
    cluster_size_m=1,
    cluster_size_n=1,
    pipeline_stages=None,
    swizzle=None,
    swizzle_n_maj=True,
    tile_size_m_noising_A=None,
    tile_size_n_noising_B=None,
    tile_size_k_noising_A=None,
    tile_size_k_noising_B=None,
    pipeline_stages_noising_A=2,
    pipeline_stages_noising_B=2,
    k_blocks_per_split_noising_A=None,
    k_blocks_per_split_noising_B=None,
    run_noising_A=True,
    run_noising_B=True,
    skip_reduction=False,
    skip_denoising=False,
    inner_hash_counter=None,
    enable_debug=False,
):
    return None


def tensor_hash(
    data,
    key,
    out,
    roots,
    threads_per_block=128,
    num_stages=2,
    leaves_per_mt_block=512,
):
    """Hash a tensor using a key with configurable kernel parameters.

    Args:
        data (Tensor): 2D uint8 or int8 tensor to hash as raw storage bytes.
            int8 and uint8 inputs with the same bit pattern produce the same root.
        key (Tensor): Key tensor used for hashing
        out (Tensor): Output tensor for hash result
        roots (Tensor): Scratchpad tensor for intermediate results
        threads_per_block (int): Number of threads per block for merkle_tree_roots_kernel.
                                Supported values: 128, 256, 512. Default: 128
        num_stages (int): Number of pipeline stages for merkle_tree_roots_kernel.
                         Supported values: 2, 3, 4. Default: 2
        leaves_per_mt_block (int): Number of threads for compute_blake_mt_kernel.
                                  Supported values: 256, 512, 1024. Default: 512

    Returns:
        Tensor: Hash output tensor
    """
    return pearl_gemm_cuda.tensor_hash(
        data, key, out, roots, threads_per_block, num_stages, leaves_per_mt_block
    )


def commitment_hash_from_merkle_roots(
    A_merkle_root,
    B_merkle_root,
    key,
    A_commitment_hash,
    B_commitment_hash,
    routing_root=None,
    offsets_hash=None,
):
    """
    Compute the commitment hash from merkle roots of a 2D tensor.
    """
    return pearl_gemm_cuda.commitment_hash_from_merkle_roots(
        A_merkle_root,
        B_merkle_root,
        key,
        A_commitment_hash,
        B_commitment_hash,
        routing_root,
        offsets_hash,
    )


def commitment_hash_from_b_commitment(
    A_merkle_root,
    B_commitment_hash,
    A_commitment_hash,
    routing_root=None,
    offsets_hash=None,
):
    return pearl_gemm_cuda.commitment_hash_from_b_commitment(
        A_merkle_root,
        B_commitment_hash,
        A_commitment_hash,
        routing_root,
        offsets_hash,
    )


@torch.library.register_fake("pearl_gemm::commitment_hash_from_merkle_roots")
def _abstract_commitment_hash_from_merkle_roots(
    A_merkle_root,
    B_merkle_root,
    key,
    A_commitment_hash,
    B_commitment_hash,
    routing_root=None,
    offsets_hash=None,
):
    return None


@torch.library.register_fake("pearl_gemm::commitment_hash_from_b_commitment")
def _abstract_commitment_hash_from_b_commitment(
    A_merkle_root,
    B_commitment_hash,
    A_commitment_hash,
    routing_root=None,
    offsets_hash=None,
):
    return None


def quantize(input_tensor, output, scales, max_val=63, smooth_scale=None, fast_math=False):
    """
    Perform dynamic per-token quantization on input tensor.

    This function quantizes each row of the input tensor independently using
    dynamic scaling. Supports 7-bit ([-63, 63]) and 8-bit ([-127, 127]) ranges.

    Args:
        input_tensor (Tensor): Input tensor to quantize, shape (num_tokens, hidden_size).
                              Must be float16 or bfloat16 dtype.
        output (Tensor): Output tensor for quantized values, shape (num_tokens, hidden_size).
                        Must be int8 dtype.
        scales (Tensor): Output tensor for per-row scales, shape (num_tokens, 1).
                        Must be float32 dtype.
        max_val (int): Maximum quantization value. Must be 63 (7-bit) or 127 (8-bit).
                      Default: 63.
        smooth_scale (Tensor, optional): Per-channel scale tensor, shape (hidden_size,).
                                        Must be float32, float16, or bfloat16 dtype.
                                        When provided, input is divided by smooth_scale
                                        before quantization. Default: None.
        fast_math (bool): If True, use div.approx.ftz (standard division) for division.
                         If False, use IEEE754 compatible div.rn.ftz.
                         Default: False.

    The quantization formula:
        If smooth_scale is None:
            scale[i] = max(abs(input[i, :])) / max_val
            output[i, j] = round(input[i, j] / scale[i])
        If smooth_scale is provided:
            scaled_input[i, j] = input[i, j] / smooth_scale[j]
            scale[i] = max(abs(scaled_input[i, :])) / max_val
            output[i, j] = round(scaled_input[i, j] / scale[i])
    """
    return pearl_gemm_cuda.quantize(input_tensor, output, scales, max_val, smooth_scale, fast_math)


@torch.library.register_fake("pearl_gemm::quantize")
def _abstract_quantize(
    input_tensor, output, scales, max_val=63, smooth_scale=None, fast_math=False
):
    return None


# Fake Tensor function for torch.compile support
@torch.library.register_fake("pearl_gemm::tensor_hash")
def _abstract_tensor_hash(
    data,
    key,
    out,
    roots,
    threads_per_block=128,
    num_stages=2,
    leaves_per_mt_block=512,
):
    return None
