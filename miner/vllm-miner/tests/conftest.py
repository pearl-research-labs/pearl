import pytest
import torch
from miner_base.settings import MinerSettings


@pytest.fixture
def miner_settings():
    return MinerSettings(no_gateway=True)


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture
def gpu_async_manager(miner_settings):
    from vllm_miner.runtime import delete_state, get_async_manager, init_async_manager
    from vllm_miner.settings import set_miner_settings

    init_async_manager(miner_settings)
    am = get_async_manager()
    am._conf = miner_settings
    set_miner_settings(miner_settings)
    yield am
    delete_state()


@pytest.fixture(autouse=True)
def async_manager(gpu_async_manager):
    yield gpu_async_manager
