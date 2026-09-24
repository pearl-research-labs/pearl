"""Offline B-side preprocessing for the FP8 miner (cert-v4 chain).

Consumer code (not part of the ``pearl_gemm`` package); also drives the
mixed_gemm test's B side. Everything that depends only on ``B``, the block
header and the mining configuration is computed here, once, off the timed
path:

* ``keyA``/``keyB`` (from the header), ``HB``, ``noise seedB`` (from ``HB``,
  ``keyB`` and the committed ``pB``),
* the noise factors keyed by ``seedB``: ``E_B`` (n x r), ``F_B`` (r x k) and
  ``F_A`` (r x k; the ``Side.A`` address under the same key),
* the noised quantized operand ``B'`` (n x k, FP8) with its per-row scales,
* the complete peel ``[ (beta_B (.) E_B@F_B - B') @ F_A^T | -(beta_B (.) E_B) ]``.

Nothing on the B side depends on A: ``E_A`` is the only factor keyed by
``noise seedA``, and ``build_b_rows`` never reads it, so the reference
``Noiser`` here is built from ``seedB`` alone.

Fidelity: the operand construction runs the *reference miner code itself*
(``PearlScheme.build_b_rows`` / ``Fp8QuantScheme``) -- the reference ops are plain
device-agnostic torch, so we execute them on GPU tensors via a small adapter
that moves the ``Noiser``'s CPU-generated factor lines onto the device.
"""

from dataclasses import dataclass

import torch
from miner_base.commitment import (
    BlockHeader,
    Device,
    MiningConfiguration,
    bits_to_target,
    commit_planes,
    commitment_keys,
)
from miner_base.commitment_hash import noise_line_key, noise_seed_b
from miner_base.hardware import Hardware, hardware_for
from miner_base.mining_config import default_mining_config, tall_tile_mining_config
from miner_base.noise import Noiser
from miner_base.policy import effective_work
from miner_base.prequant import RowNorms, round_l2_to_grid
from miner_base.quantization import Fp8QuantScheme
from miner_base.scheme import PearlScheme, StackedRows

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


def _row_norms(opened: torch.Tensor) -> RowNorms:
    """Per-row norms of a raw BF16 operand (no int8/scale structure to bypass)."""
    sumsq = opened.to(torch.float32).pow(2).sum(dim=1, keepdim=True)
    linf = opened.abs().amax(dim=1, keepdim=True)
    l2 = round_l2_to_grid((sumsq / opened.shape[1]).sqrt().to(torch.bfloat16))
    return RowNorms(l2=l2, linf=linf)


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


class _DeviceNoiser:
    """Adapter: reference ``Noiser`` whose factor tensors land on ``device``."""

    def __init__(self, noiser: Noiser, device: torch.device):
        self._noiser = noiser
        self._device = device

    def E_A(self, row_indices):
        return self._noiser.E_A(row_indices).to(self._device)

    def F_A(self):
        return self._noiser.F_A().to(self._device)

    def E_B(self, row_indices):
        return self._noiser.E_B(row_indices).to(self._device)

    def F_B(self):
        return self._noiser.F_B().to(self._device)


@dataclass
class MinerContext:
    """Everything the runtime path needs, preprocessed from (B, header, config)."""

    config: MiningConfiguration
    header: BlockHeader
    hardware: Hardware
    k: int
    r: int
    n: int

    key_a: bytes  # keyA: A's opening key (header-derived)
    key_b: bytes  # keyB: B's opening key (header-derived)
    seed_b: bytes  # noise seedB (both F bases / B's E lines; the B-side stamp)

    e2: torch.Tensor  # (n x r) FP8, on GPU: E_B
    f1: torch.Tensor  # (r x k) FP8, on GPU: F_A (keyed by seedB, Side.A address)
    f2: torch.Tensor  # (r x k) FP8, on GPU: F_B
    b_prime: torch.Tensor  # (n x k) FP8, on GPU
    b_peel: torch.Tensor  # (n x 2r) BF16, on GPU: the complete job-constant peel
    alpha_b: torch.Tensor  # (n x 1) BF16, on GPU
    beta_b: torch.Tensor  # (n x 1) BF16, on GPU
    l2_b: torch.Tensor  # (n x 1) BF16, on GPU: floored row rms the scales came from

    target: int  # 256-bit lottery target (from header.nbits)

    @property
    def noise_key_b(self) -> bytes:
        return noise_line_key(self.seed_b)

    def noise(self, seed_a: bytes | None = None) -> Noiser:
        """The job's reference noise factors; ``seed_a`` unlocks ``E_A``."""
        return Noiser(self.seed_b, self.r, self.k, self.hardware.compute, seed_a=seed_a)

    def reference_b_rows(self, b: torch.Tensor) -> StackedRows:
        """``build_b_rows`` (the reference ops, on ``b``'s device)."""
        return _reference_b_rows(b, self.seed_b, self.r, self.hardware)

    def threshold_for(self) -> int:
        """Lottery threshold ``min(target * work, 2^256-1)`` for the merged tile."""
        rows = self.config.rows_pattern.tile_size
        cols = self.config.cols_pattern.tile_size
        work = effective_work(rows, cols, self.k, self.r)
        return min(self.target * work, _MAX_256)

    def threshold_bytes(self) -> bytes:
        return self.threshold_for().to_bytes(32, "little")


def _reference_b_rows(b: torch.Tensor, seed_b: bytes, r: int, hardware: Hardware) -> StackedRows:
    """``build_b_rows`` over ``b`` on its own device, from ``seedB`` alone."""
    n, k = b.shape
    scheme = PearlScheme(hardware, Fp8QuantScheme(), k, r)
    noise = _DeviceNoiser(Noiser(seed_b, r, k, hardware.compute), b.device)
    return scheme.build_b_rows(b, noise, list(range(n)), _row_norms(b))


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
    r = config.rank
    device = B.device if B.is_cuda else torch.device("cuda")

    key_a, key_b = commitment_keys(bytes(header.to_bytes()))
    # HB / seedB: B is a raw BF16 operand, so it commits as a single plane
    # over its CPU bytes (still wrapped by commit_planes' keyed digest combine).
    digest_b = commit_planes([B.cpu()], key_b, config.b_hash_id).digest
    seed_b = noise_seed_b(digest_b, key_b, config.p_b(n))

    hardware = hardware_for(config.device)
    noise = _DeviceNoiser(Noiser(seed_b, r, k, hardware.compute), device)
    f_a = noise.F_A()

    B_dev = B.to(device)
    # B is a raw BF16 research operand, so its norms reduce over the rows
    # themselves (no int8 blocks to take them off). The whole B side is
    # job-constant, so build_b_rows runs to completion here; beta_b is not
    # carried on the built rows and comes back from the quant step.
    e_b = noise.E_B(list(range(n)))
    f_b = noise.F_B()
    stacked = _reference_b_rows(B_dev, seed_b, r, hardware)
    _, _, beta_b, _ = Fp8QuantScheme().noisy_quantize(B_dev, e_b, f_b, hardware, _row_norms(B_dev))

    return MinerContext(
        config=config,
        header=header,
        hardware=hardware,
        k=k,
        r=r,
        n=n,
        key_a=key_a,
        key_b=key_b,
        seed_b=seed_b,
        e2=e_b.contiguous(),
        f1=f_a.contiguous(),
        f2=f_b.contiguous(),
        b_prime=stacked.quant_part.contiguous(),
        b_peel=stacked.peel_part.contiguous(),
        alpha_b=stacked.alpha.contiguous(),
        beta_b=beta_b.contiguous(),
        l2_b=stacked.l2.contiguous(),
        target=bits_to_target(header.nbits),
    )
