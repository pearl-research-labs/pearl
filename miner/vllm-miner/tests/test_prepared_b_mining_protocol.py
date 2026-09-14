from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("salted", [False, True])
def test_salted_jobs_preserve_full_commitment_preparation(monkeypatch, salted):
    import vllm_miner.gemm_operators as operators
    from vllm_miner.prepared_b_mining_state import PreparedBMiningState

    class CpuTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *args, **kwargs):
            return torch.empty(*args, **{**kwargs, "device": "cpu"})

        def zeros(self, *args, **kwargs):
            return torch.zeros(*args, **{**kwargs, "device": "cpu"})

        def frombuffer(self, *args, **kwargs):
            return SimpleNamespace(to=lambda device: torch.zeros(32, dtype=torch.uint8))

    state = PreparedBMiningState(*(torch.zeros(32, dtype=torch.uint8) for _ in range(6)))
    full_factors = tuple(torch.zeros(1, dtype=torch.int8) for _ in range(8))
    cached_a_factors = tuple(torch.zeros(1, dtype=torch.int8) for _ in range(4))
    cache = Mock(return_value=state)
    full_commitment = Mock()
    cached_commitment = Mock()
    full_noise = Mock(return_value=full_factors)
    cached_a_noise = Mock(return_value=cached_a_factors)
    hash_tensor = Mock()
    gemm = Mock()
    pool = SimpleNamespace(acquire=lambda: torch.empty(1), release=Mock())
    job = SimpleNamespace(
        cert_version=SimpleNamespace(uses_salted_seeds=salted),
        incomplete_header_bytes=b"header",
        adjust_target=lambda **kwargs: 1,
    )
    monkeypatch.setattr(operators, "torch", CpuTorch())
    monkeypatch.setattr(
        operators, "get_async_manager", lambda: SimpleNamespace(get_mining_job=lambda: job)
    )
    monkeypatch.setattr(operators, "get_pinned_pool", lambda: pool)
    monkeypatch.setattr(
        operators,
        "GPUMatmulConfigFactory",
        SimpleNamespace(create=lambda **kwargs: SimpleNamespace(mining_config=None)),
    )
    monkeypatch.setattr(
        operators, "CommitmentHasher", SimpleNamespace(get_key=lambda *args: bytes(32))
    )
    monkeypatch.setattr(operators, "get_required_scratchpad_bytes", lambda size: 1)
    monkeypatch.setattr(operators, "get_host_signal_sync_size", lambda: 1)
    monkeypatch.setattr(
        operators, "make_pow_target_tensor", lambda target: torch.zeros(8, dtype=torch.uint32)
    )
    monkeypatch.setattr(operators, "tensor_hash", hash_tensor)
    monkeypatch.setattr(operators, "get_or_prepare_b_mining_state", cache)
    monkeypatch.setattr(operators, "commitment_hash_from_merkle_roots", full_commitment)
    monkeypatch.setattr(operators, "commitment_hash_from_b_commitment", cached_commitment)
    monkeypatch.setattr(operators, "generate_noise_factors", full_noise)
    monkeypatch.setattr(operators, "generate_a_noise_factors", cached_a_noise)
    monkeypatch.setattr(operators, "noisy_gemm", gemm)

    activations = torch.zeros((2, 4), dtype=torch.int8)
    weights = torch.zeros((3, 4), dtype=torch.int8)
    result = operators.pearl_gemm_noisy(
        activations,
        weights,
        torch.ones((2, 1)),
        torch.ones((3, 1)),
        torch.bfloat16,
        submit_block=False,
    )
    assert result.shape == (2, 3)
    assert hash_tensor.call_args_list[0].args[0] is activations
    gemm.assert_called_once()
    pool.release.assert_called_once()
    if salted:
        cache.assert_not_called()
        cached_commitment.assert_not_called()
        cached_a_noise.assert_not_called()
        assert hash_tensor.call_count == 2
        assert hash_tensor.call_args_list[1].args[0] is weights
        full_commitment.assert_called_once()
        assert full_commitment.call_args.kwargs == {"salted_dims": (2, 3)}
        full_noise.assert_called_once()
        assert gemm.call_args.kwargs["EBL_R_major"] is full_factors[2]
    else:
        full_commitment.assert_not_called()
        full_noise.assert_not_called()
        assert hash_tensor.call_count == 1
        cache.assert_called_once()
        assert cache.call_args.args[0] is weights
        cached_commitment.assert_called_once()
        assert cached_commitment.call_args.args[1] is state.commitment_b
        cached_a_noise.assert_called_once()
        assert gemm.call_args.kwargs["EBL_R_major"] is state.ebl_r_major
