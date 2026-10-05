import pytest
from miner_base.settings import MinerSettings


@pytest.fixture
def miner_settings():
    return MinerSettings(no_gateway=True)


@pytest.fixture(autouse=True)
def async_manager(gpu_async_manager):
    yield gpu_async_manager
