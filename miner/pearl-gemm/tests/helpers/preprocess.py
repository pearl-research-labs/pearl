"""Offline B-side preprocessing for the FP8 miner (cert-v4 chain).

Consumer code (not part of the ``pearl_gemm`` package); drives the pipeline
tests' B side. Everything that depends only on ``B``, the block header and
the mining configuration is computed here, once, off the timed path, through
the same GPU chain the production miner runs:

* ``keyA``/``keyB`` (from the header),
* ``pre_quant`` of ``B`` into its codes/scales planes, their commitment
  (``tensor_hash_plus_stats_b``), ``HB`` and ``noise seedB`` (from ``HB``,
  ``keyB`` and the committed ``pB``),
* the F bases keyed by ``seedB`` (``noise_lines``): ``F_A`` at the ``Side.A``
  address and ``F_B``,
* ``noisy_quant_b``: the noised quantized operand ``B'`` (n x k, FP8) with its
  per-row scales and the complete peel
  ``[ (beta_B (.) E_B@F_B - B') @ F_A^T | -(beta_B (.) E_B) ]``.

Nothing on the B side depends on A: ``E_A`` is the only factor keyed by
``noise seedA``.
"""

from dataclasses import dataclass

import torch
from miner_base.commitment import (
    BlockHeader,
    Device,
    MiningConfiguration,
    bits_to_target,
    commitment_keys,
)
from miner_base.commitment_hash import noise_line_key, noise_seed_b, operand_digest
from miner_base.mining_config import default_mining_config, tall_tile_mining_config
from miner_base.policy import effective_work

from pearl_gemm import (
    LABEL_F1,
    LABEL_F2,
    NoisyQuantBConfig,
    TensorHashConfig,
    noise_lines,
    noisy_quant_b,
    pack_noise_factor,
    pre_quant,
    pre_quant_output_shapes,
    tensor_hash_plus_stats_b,
    tensor_hash_workspace_bytes,
)
from pearl_gemm.protocol_constants import R

# Grid/subtile lottery layout: each subtile is exactly one thread's
# natively-held TMEM accumulator fragment at fixed h within one
# tile_cols-wide block, so the GPU fold is thread-local (no shuffles). The
# merged tile is 4 contiguous rows x tile_cols contiguous cols, full
# coverage. tile_cols is a per-job choice committed into ``pB``
# (128 default; 64/192/256 legal -- subtile elems <= 256 for all four,
# see mixed_gemm/).
TILE_ROWS = 4
TILE_COLS = 128
_MAX_256 = (1 << 256) - 1


def default_config(
    k: int, r: int = R, device: Device = Device.BLACKWELL, tile_cols: int = TILE_COLS
) -> MiningConfiguration:
    assert tile_cols in (64, 128, 192, 256)
    return default_mining_config(k, r, ltile_cols=tile_cols, device=device)


def tall_tile_config(k: int, r: int = R, device: Device = Device.BLACKWELL) -> MiningConfiguration:
    """16x32 merged tile (mixed_gemm's ltile_rows=16, ltile_cols=32 variant).

    Same 512-element area as 4x128 (difficulty-invariant) but rows+cols = 48,
    so the peel proof fits the verifier's 4 MiB worker input up to k ~ 43520
    (vs 15872 for 4x128) -- what makes k=16384 (GLM o_proj) mineable. Grid is
    16x1 with a 1x32 subtile: one lane per tile row, each lane folding its
    row's 32 contiguous columns in ascending order -- exactly the kernel's
    thread-local single-row fold. Verifier bounds: blake product = 16*1 =
    LANES, subtile 1*32 = 32 <= 256 elems.
    """
    return tall_tile_mining_config(k, r, device)


def default_header(nbits: int = 0x1D3FFFFF, timestamp: int = 1_700_000_000) -> BlockHeader:
    return BlockHeader(
        version=1,
        prev_block=b"\x11" * 32,
        merkle_root=b"\x22" * 32,
        timestamp=timestamp,
        nbits=nbits,
    )


@dataclass
class MinerContext:
    """Everything the runtime path needs, preprocessed from (B, header, config)."""

    config: MiningConfiguration
    header: BlockHeader
    k: int
    n: int

    key_a: bytes  # keyA: A's opening key (header-derived)
    key_b: bytes  # keyB: B's opening key (header-derived)
    seed_b: bytes  # noise seedB (both F bases / B's E lines; the B-side stamp)

    f1: torch.Tensor  # (r x k) FP8, on GPU: F_A (keyed by seedB, Side.A address)
    f2: torch.Tensor  # (r x k) FP8, on GPU: F_B
    b_prime: torch.Tensor  # (n x k) FP8, on GPU
    b_peel: torch.Tensor  # (n x 2r) BF16, on GPU: the complete job-constant peel
    alpha_b: torch.Tensor  # (n,) BF16, on GPU

    target: int  # 256-bit lottery target (from header.nbits)

    @property
    def noise_key_b(self) -> bytes:
        return noise_line_key(self.seed_b)

    def threshold_for(self) -> int:
        """Lottery threshold ``min(target * work, 2^256-1)`` for the merged tile."""
        rows = self.config.rows_pattern.tile_size
        cols = self.config.cols_pattern.tile_size
        work = effective_work(rows, cols, self.k, self.config.rank)
        return min(self.target * work, _MAX_256)

    def threshold_bytes(self) -> bytes:
        return self.threshold_for().to_bytes(32, "little")


def _device_bytes(data: bytes, device: torch.device) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)


def preprocess(
    B: torch.Tensor,
    header: BlockHeader | None = None,
    config: MiningConfiguration | None = None,
) -> MinerContext:
    """Preprocess the B side. ``B``: (n x k) BF16, on GPU (or CPU; moved as needed)."""
    assert B.dim() == 2 and B.dtype == torch.bfloat16
    n, k = B.shape
    if header is None:
        header = default_header()
    if config is None:
        config = default_config(k)
    assert config.common_dim == k
    assert config.rank == R, "the kernels are specialized for the protocol rank"
    device = B.device if B.is_cuda else torch.device("cuda")
    B_dev = B.to(device)

    key_a, key_b = commitment_keys(bytes(header.to_bytes()))

    codes_shape, scales_shape = pre_quant_output_shapes(n, k)
    codes = torch.zeros(codes_shape, dtype=torch.int8, device=device)
    scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device=device)
    pre_quant(B_dev, codes, scales)

    hash_config = TensorHashConfig()
    root_codes = torch.zeros(32, dtype=torch.uint8, device=device)
    root_scales = torch.zeros(32, dtype=torch.uint8, device=device)
    roots = torch.zeros(
        tensor_hash_workspace_bytes(n, k, hash_config), dtype=torch.uint8, device=device
    )
    commit_stats = torch.zeros(2 * (n * k // 512), dtype=torch.float32, device=device)
    tensor_hash_plus_stats_b(
        codes,
        scales,
        _device_bytes(key_b, device),
        root_codes,
        root_scales,
        roots,
        commit_stats,
        config=hash_config,
    )
    hash_b = operand_digest(
        [bytes(root_codes.cpu().numpy()), bytes(root_scales.cpu().numpy())], key_b
    )
    seed_b = noise_seed_b(hash_b, key_b, config.p_b(n))
    noise_key_b = _device_bytes(noise_line_key(seed_b), device)

    lines = torch.zeros(k, R, dtype=torch.float8_e4m3fn, device=device)
    noise_lines(noise_key_b, LABEL_F1, lines)
    f_a = lines.t().contiguous()
    noise_lines(noise_key_b, LABEL_F2, lines)
    f_b = lines.t().contiguous()

    alpha_b = torch.zeros(n, dtype=torch.bfloat16, device=device)
    beta_b = torch.zeros_like(alpha_b)
    e_b = torch.zeros(n, R, dtype=torch.float8_e4m3fn, device=device)
    b_prime = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device=device)
    b_peel = torch.zeros(n, 2 * R, dtype=torch.bfloat16, device=device)
    noisy_quant_b(
        codes,
        scales,
        noise_key_b,
        commit_stats,
        pack_noise_factor(f_b),
        f_a,
        alpha_b,
        beta_b,
        e_b,
        b_prime,
        b_peel,
        torch.zeros(R, R, dtype=torch.float32, device=device),
        config=NoisyQuantBConfig(),
    )
    torch.cuda.synchronize()

    return MinerContext(
        config=config,
        header=header,
        k=k,
        n=n,
        key_a=key_a,
        key_b=key_b,
        seed_b=seed_b,
        f1=f_a,
        f2=f_b,
        b_prime=b_prime,
        b_peel=b_peel,
        alpha_b=alpha_b,
        target=bits_to_target(header.nbits),
    )
