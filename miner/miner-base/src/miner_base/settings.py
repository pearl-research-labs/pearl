from pydantic_settings import BaseSettings, SettingsConfigDict


class MinerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="miner_")

    noise_range: int = 128
    noise_rank: int = 128
    idxs_per_col: int = 2

    # GEMM tile sizes
    tile_size_m: int = 128
    tile_size_n: int = 256
    tile_size_k: int = 128

    # fmt: off
    # Hash tile pattern for the 128x256 tile
    rows_pattern: list[int] = [0, 8]
    cols_pattern: list[int] = [
    0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57,
    64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 104, 105, 112, 113, 120, 121,
    128, 129, 136, 137, 144, 145, 152, 153, 160, 161, 168, 169, 176, 177, 184, 185,
    192, 193, 200, 201, 208, 209, 216, 217, 224, 225, 232, 233, 240, 241, 248, 249,
    ]
    # fmt: on

    pinned_pool_size: int = 128

    debug: bool = False
    print_header_hash: bool = False
    no_gateway: bool = False
    no_mining: bool = False
    skip_block_submission: bool = False
    no_vllm_plugin: bool = False

    enable_async_cuda_event_processing: bool = True
