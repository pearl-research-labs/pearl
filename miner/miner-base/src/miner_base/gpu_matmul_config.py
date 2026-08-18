"""GPU-specific MatmulConfig factory.

This module requires torch.cuda, so it should only be imported
in GPU environments.
"""

from pearl_gateway.comm.mining_configuration import (
    MiningConfiguration,
    MMAType,
    MoEConfig,
    PeriodicPattern,
)

from .matmul_config import MatmulConfig
from .settings import MinerSettings


class GPUMatmulConfigFactory:
    """Factory for creating MatmulConfig for GPU-based mining."""

    @staticmethod
    def create(k: int, noise_rank: int, moe: MoEConfig | None = None) -> MatmulConfig:
        settings = MinerSettings()
        rows_pattern = PeriodicPattern.from_list(settings.rows_pattern)
        cols_pattern = PeriodicPattern.from_list(settings.cols_pattern)

        mining_config = MiningConfiguration(
            common_dim=k,
            rank=noise_rank,
            mma_type=MMAType.Int7xInt7ToInt32,
            rows_pattern=rows_pattern,
            cols_pattern=cols_pattern,
            moe=moe,
        )

        return MatmulConfig(
            matmul_tile_h=settings.tile_size_m,
            matmul_tile_w=settings.tile_size_n,
            matmul_tile_k=settings.tile_size_k,
            mining_config=mining_config,
        )
