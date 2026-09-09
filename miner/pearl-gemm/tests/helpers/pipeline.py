"""Full runtime mining pipeline: A (BF16, on GPU) -> (C'', first winner).

Example consumer of the 4-op kernel API (not part of the ``pearl_gemm``
package). The B side is preprocessed offline (``preprocess.py``), the timed path receives only ``A``
and runs four GPU stages, all launch-only (no D2H sync, CUDA-graph capturable):

    1. prequant -- block-scaled INT8 quantization of A into the two
                   committed blobs (codes and scales)
    2. commit   -- one keyed BLAKE3 Merkle root per blob + fused per-block
                   stats partials + on-GPU finalize of A's keys (seedA,
                   noise-line key, jackpot key)
    3. noise    -- F_A from A's noise-line key (``noise_lines``), then
                   combine partials -> alpha/beta, E_A in-kernel, and
                   A' = Q(alpha (.) A + beta (.) E_A@F_A) + peel columns
    4. gemm     -- fused FP8 UMMA + in-register lottery + BF16 peel + unscale
                   (B's peel mid half is re-derived per launch from F_A)

Only stage 1 reads the BF16 activation: stages 2 and 3 both consume the two
committed blobs, so the noise stage is protocol-canonical by construction
rather than by assuming A is a block-scaled fixpoint.

The persistent hit signal and the ``C''`` buffer are owned by the pipeline.
"""

from dataclasses import dataclass

import torch

from pearl_gemm import (
    LABEL_F1,
    HitSignal,
    HitSignalConfig,
    MixedGemmConfig,
    NoisyQuantConfig,
    PreQuantConfig,
    TensorHashConfig,
    mixed_gemm,
    noise_lines,
    noisy_quant,
    pre_quant,
    pre_quant_output_shapes,
    tensor_hash_plus_stats,
    tensor_hash_plus_stats_record_is_legal,
    tensor_hash_workspace_bytes,
)
from pearl_gemm.autotune import get_tuned
from pearl_gemm.protocol_constants import R

from .preprocess import MinerContext


@dataclass(frozen=True)
class LotteryWinner:
    """Coordinates of the winning lottery tile read from the hit signal."""

    tile_row: int
    tile_column: int


@dataclass
class MinerResult:
    """Views over the pipeline's stable output buffers (valid after sync)."""

    c: torch.Tensor  # (m, n) BF16: C'' ~= A @ B.T
    hit_signal: HitSignal  # the pipeline-owned persistent hit signal

    @property
    def jackpot(self) -> bool:
        return self.winner is not None

    @property
    def winner(self) -> LotteryWinner | None:
        hit = self.hit_signal.read_hit()
        if hit is None or not hit.valid:
            return None
        return LotteryWinner(tile_row=hit.tile_row, tile_column=hit.tile_column)


class MinerPipeline:
    """Compiled per-(m, ctx) mining pipeline over a preprocessed B side.

    Tune dicts default (None) to the device's autotune config via
    ``get_tuned``; pass an explicit ``{}`` to force the library defaults.
    """

    def __init__(
        self,
        ctx: MinerContext,
        m: int,
        device="cuda",
        *,
        prequant_tune: dict | None = None,
        commit_tune: dict | None = None,
        prepare_tune: dict | None = None,
        gemm_tune: dict | None = None,
    ):
        self.ctx = ctx
        self.m = m
        self.device = torch.device(device)

        if prequant_tune is None:
            prequant_tune = get_tuned("pre_quant", m=m, k=ctx.k)
        if commit_tune is None:
            commit_tune = get_tuned(
                "tensor_hash_plus_stats",
                legal=tensor_hash_plus_stats_record_is_legal,
                m=m,
                k=ctx.k,
            )
        if prepare_tune is None:
            prepare_tune = get_tuned("noisy_quant", m=m, k=ctx.k)
        if gemm_tune is None:
            gemm_tune = get_tuned("mixed_gemm", m=m, n=ctx.n, k=ctx.k)

        self.prequant_config = PreQuantConfig(**prequant_tune)
        self.commit_config = TensorHashConfig(**commit_tune)
        self.prepare_config = NoisyQuantConfig(**prepare_tune)
        self.gemm_config = MixedGemmConfig(**gemm_tune)

        self.key_a = torch.frombuffer(bytearray(ctx.key_a), dtype=torch.uint8).to(device)
        self.seed_b = torch.frombuffer(bytearray(ctx.seed_b), dtype=torch.uint8).to(device)
        self.p_a = ctx.config.p_a(m)
        codes_shape, scales_shape = pre_quant_output_shapes(m, ctx.k)
        self.codes = torch.zeros(codes_shape, dtype=torch.int8, device=device)
        self.scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device=device)
        self.root_codes = torch.zeros(32, dtype=torch.uint8, device=device)
        self.root_scales = torch.zeros(32, dtype=torch.uint8, device=device)
        self.roots = torch.zeros(
            tensor_hash_workspace_bytes(m, ctx.k, self.commit_config),
            dtype=torch.uint8,
            device=device,
        )
        # seedA || noise-line key A || jackpot key, finalized on device.
        self.a_keys = torch.zeros(96, dtype=torch.uint8, device=device)
        self.stats = torch.zeros(2 * (m * ctx.k // 512), dtype=torch.float32, device=device)

        self.alpha_a = torch.zeros(m, dtype=torch.bfloat16, device=device)
        self.beta_a = torch.zeros_like(self.alpha_a)
        self.e1 = torch.zeros(m, R, dtype=torch.float8_e4m3fn, device=device)
        self.a_prime = torch.zeros(m, ctx.k, dtype=torch.float8_e4m3fn, device=device)
        self.a_peel = torch.zeros(m, 2 * R, dtype=torch.bfloat16, device=device)
        self.f1_lines = torch.zeros(ctx.k, R, dtype=torch.float8_e4m3fn, device=device)
        self.f2 = ctx.f2.contiguous()
        self.b_peel = torch.zeros_like(ctx.b_peel)

        self.inv_alpha_b = torch.reciprocal(ctx.alpha_b.reshape(-1).float()).contiguous()
        self.c = torch.zeros(m, ctx.n, dtype=torch.bfloat16, device=device)
        self.hit_signal = HitSignal(HitSignalConfig(max_m=m, max_k=ctx.k), device=self.device)
        # The kernel implements one committed lottery pattern per instance
        # (merged tile = ltile_rows rows x ltile_cols cols, thread-local
        # subtiles; see mixed_gemm/) -- reject configs that
        # committed anything else.
        assert (
            ctx.config.rows_pattern.tile_size == self.gemm_config.ltile_rows
            and ctx.config.cols_pattern.tile_size == self.gemm_config.ltile_cols
        ), "committed pattern must match the gemm's lottery tile (default_config)"

        self.threshold = torch.frombuffer(bytearray(ctx.threshold_bytes()), dtype=torch.uint8).to(
            device
        )

    # ---- individual stages (used by the per-part benchmark) ----

    def stage_prequant(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_quant(A, self.codes, self.scales, config=self.prequant_config)
        return self.codes, self.scales

    def stage_commit(self) -> torch.Tensor:
        tensor_hash_plus_stats(
            self.codes,
            self.scales,
            self.key_a,
            self.seed_b,
            self.root_codes,
            self.root_scales,
            self.roots,
            self.a_keys,
            self.stats,
            p_a=self.p_a,
            config=self.commit_config,
        )
        return self.a_keys

    @property
    def seed_a(self) -> torch.Tensor:
        return self.a_keys[0:32]

    @property
    def noise_key_a(self) -> torch.Tensor:
        return self.a_keys[32:64]

    @property
    def pow_key(self) -> torch.Tensor:
        return self.a_keys[64:96]

    def stage_prepare(self) -> None:
        noise_lines(self.noise_key_a, LABEL_F1, self.f1_lines)
        noisy_quant(
            self.codes,
            self.scales,
            self.noise_key_a,
            self.stats,
            self.f1_lines.view(torch.int8),  # == pack_noise_factor(F_A) at R == PACKED_NOISE_K
            self.f2,
            self.alpha_a,
            self.beta_a,
            self.e1,
            self.a_prime,
            self.a_peel,
            config=self.prepare_config,
        )

    def stage_gemm(self) -> torch.Tensor:
        # Consume any pending hit so this launch's result is its own (the
        # persistent latch would otherwise drop it). reset_hit() synchronizes
        # the signal's dedicated stream, which is illegal during CUDA graph
        # capture -- fail closed there instead of corrupting the capture.
        if self.hit_signal.doorbell():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "pending hit at graph capture: consume/reset the hit signal "
                    "(hit_signal.reset_hit()) before capturing the pipeline"
                )
            self.hit_signal.reset_hit()
        self.ctx.b_peel_for(self.f1_lines, out=self.b_peel)
        mixed_gemm(
            self.a_prime,
            self.ctx.b_prime,
            self.a_peel,
            self.b_peel,
            self.alpha_a,
            self.inv_alpha_b,
            self.pow_key,
            self.threshold,
            self.c,
            self.hit_signal,
            self.codes,
            self.scales,
            self.seed_b,
            config=self.gemm_config,
        )
        return self.c

    def run(self, A: torch.Tensor) -> MinerResult:
        """A: (m, k) BF16 CUDA tensor. Launch-only; sync before reading results."""
        self.stage_prequant(A)
        self.stage_commit()
        self.stage_prepare()
        self.stage_gemm()
        return MinerResult(self.c, self.hit_signal)
