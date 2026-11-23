"""Skip tensor-hash launches whose CTA shared memory the local architecture cannot hold.

SM100 parts have 228 KB per SM and SM120 parts 99 KB, so a TMA staging ring
that fits one family can exceed the other's opt-in maximum. Geometry tests
that pin a wide ring use this instead of failing on the smaller part; the
estimator itself is covered by ``test_smem_estimator_brackets_the_launch_boundary``.
"""

import pytest

from pearl_gemm import TensorHashConfig
from pearl_gemm._utils._arch import arch_of
from pearl_gemm.tensor_hash_plus_stats._merkle_host import tensor_hash_smem_fits


def tensor_hash_config_fits(config: TensorHashConfig, stats_chunk: int | None = None) -> bool:
    return tensor_hash_smem_fits(
        config.threads_per_block,
        config.num_stages,
        config.thread_load_size,
        stats_chunk,
        sync_loads=config.sync_loads,
    )


def skip_unless_tensor_hash_fits(config: TensorHashConfig, stats_chunk: int | None = None) -> None:
    if not tensor_hash_config_fits(config, stats_chunk):
        pytest.skip(
            f"threads={config.threads_per_block} stages={config.num_stages} "
            f"load={config.thread_load_size} exceeds {arch_of().name}'s shared-memory budget"
        )
