"""CPU contract for the compressed-tensors -> BF16 reconstruction used by the
upcast load path: pushing an identity through a delegate's ``apply`` recovers
the delegate's weight, including across chunk boundaries."""

import os
import subprocess
import sys

import pytest
import torch
from vllm_miner import upcast


class _FakeLinearMethod:
    """Minimal delegate whose ``apply`` computes ``x @ W.T`` (bias unused)."""

    def __init__(self, weight: torch.Tensor, out_dtype: torch.dtype | None = None):
        self._weight = weight  # (n, k)
        self._out_dtype = out_dtype
        self.seen_shapes: list[tuple[int, ...]] = []

    def apply(self, layer, x, bias=None):  # noqa: ARG002 - layer/bias unused
        self.seen_shapes.append(tuple(x.shape))
        out = x @ self._weight.t()
        return out if self._out_dtype is None else out.to(self._out_dtype)


# Shape grid for the round-trip recovery tests. Small shapes keep the CPU tests
# fast; the GLM-5.2 entries are the real (n, k) of three representative mineable
# linears so the path is exercised at production scale (not just toy sizes).
_SHAPES = [
    pytest.param(5, 12, id="tiny-5x12"),
    pytest.param(128, 512, id="med-128x512"),
    # GLM-5.2 mineable dense linears (TP=1, pre-shard shapes).
    pytest.param(16384, 2048, id="glm-q_b_proj"),
    pytest.param(24576, 6144, id="glm-gate_up_proj-dense"),
    pytest.param(6144, 12288, id="glm-down_proj-dense"),
]


@pytest.mark.parametrize(("n", "k"), _SHAPES)
def test_reconstruct_bf16_recovers_weight(n, k):
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    source = _FakeLinearMethod(weight)

    out = upcast.reconstruct_bf16(torch.nn.Identity(), source, n, k, torch.device("cpu"))

    assert out.shape == (n, k)
    assert out.dtype == torch.bfloat16
    assert out.is_contiguous()
    torch.testing.assert_close(out, weight)


@pytest.mark.parametrize(("n", "k"), _SHAPES)
def test_reconstruct_bf16_spans_chunk_boundary(n, k):
    # A chunk size that does not divide k exercises the trailing partial chunk
    # on every shape in the grid. Passed explicitly rather than monkeypatched.
    chunk = max(2, k // 3)
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    source = _FakeLinearMethod(weight)

    out = upcast.reconstruct_bf16(
        torch.nn.Identity(), source, n, k, torch.device("cpu"), chunk_rows=chunk
    )

    torch.testing.assert_close(out, weight)
    # The final chunk is narrowed to the remainder, never padded past k.
    expected_rows = [chunk] * (k // chunk)
    if k % chunk:
        expected_rows.append(k % chunk)
    assert source.seen_shapes == [(rows, k) for rows in expected_rows]


@pytest.mark.parametrize(("n", "k"), _SHAPES)
def test_reconstruct_bf16_with_exact_chunk_multiple(n, k):
    # k-aligned chunk size: no trailing partial chunk.
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    source = _FakeLinearMethod(weight)

    out = upcast.reconstruct_bf16(
        torch.nn.Identity(), source, n, k, torch.device("cpu"), chunk_rows=k
    )

    torch.testing.assert_close(out, weight)
    assert source.seen_shapes == [(k, k)]


def test_reconstruct_bf16_uses_the_configured_chunk_size_by_default(monkeypatch):
    # With no explicit chunk_rows the reconstruction reads PEARL_RECON_CHUNK_ROWS
    # via runtime_settings(); a small configured value forces multiple chunks.
    from vllm_miner import settings

    monkeypatch.setattr(settings, "_settings", None)
    monkeypatch.setenv("PEARL_RECON_CHUNK_ROWS", "4")
    n, k = 3, 10
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    source = _FakeLinearMethod(weight)

    out = upcast.reconstruct_bf16(torch.nn.Identity(), source, n, k, torch.device("cpu"))

    torch.testing.assert_close(out, weight)
    assert source.seen_shapes == [(4, k), (4, k), (2, k)]


@pytest.mark.parametrize(("n", "k"), _SHAPES)
def test_reconstruct_bf16_downcasts_higher_precision_delegate_output(n, k):
    # Real fp8/fp4 kernels return the layer's compute dtype, which can be wider
    # than bfloat16; the reconstruction must still land in bfloat16.
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    source = _FakeLinearMethod(weight, out_dtype=torch.float32)

    out = upcast.reconstruct_bf16(torch.nn.Identity(), source, n, k, torch.device("cpu"))

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, weight)


def test_reconstruct_bf16_rejects_a_delegate_output_of_the_wrong_shape():
    # A delegate that folds padding or a reshape into ``apply`` must not be
    # broadcast into the destination columns as if it were the real weight.
    n, k = 4, 6
    source = _FakeLinearMethod(torch.randn(n + 3, k, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="upcast delegate returned shape"):
        upcast.reconstruct_bf16(torch.nn.Identity(), source, n, k, torch.device("cpu"))


def test_upcast_imports_without_vllm():
    # vllm_miner must not import vLLM at module import time, or this
    # module (and these CPU tests) becomes unimportable off a serving host. Checked in a fresh interpreter because this session imports vLLM.
    script = (
        "import sys, vllm_miner.upcast as u\n"
        "assert callable(u.reconstruct_bf16)\n"
        "assert 'vllm' not in sys.modules, 'upcast pulled in vLLM at import time'\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("n", "k"), [(0, 8), (8, 0), (-1, 8)])
def test_reconstruct_bf16_rejects_degenerate_shapes(n, k):
    # A zero-width loop would silently return uninitialised memory.
    source = _FakeLinearMethod(torch.zeros(1, 1, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="cannot reconstruct"):
        upcast.reconstruct_bf16(torch.nn.Identity(), source, n, k, torch.device("cpu"))


@pytest.mark.parametrize("bad", [0, -1, -4096, True, 4.0, "4096"])
def test_reconstruct_bf16_rejects_an_invalid_chunk_rows_override(bad):
    # An explicit chunk_rows bypasses RuntimeSettings validation: 0 would raise
    # an opaque range() error after allocating, and a negative value would skip
    # the loop and return uninitialised torch.empty data as a "reconstruction".
    # bool is rejected too (True == 1 would otherwise slip through the int check).
    n, k = 4, 6
    source = _FakeLinearMethod(torch.randn(n, k, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="chunk_rows must be an integer"):
        upcast.reconstruct_bf16(
            torch.nn.Identity(), source, n, k, torch.device("cpu"), chunk_rows=bad
        )


def test_static_activation_scale_warning_returns_bool_and_fires_once():
    # The helper returns True when it warns and stays silent on dynamic schemes.
    dynamic = torch.nn.Identity()
    dynamic.input_scale = None  # vLLM's marker for dynamic activation quant
    assert upcast._warn_if_static_activation_scale(dynamic) is False

    static = torch.nn.Identity()
    static.input_scale = torch.tensor(0.05)
    assert upcast._warn_if_static_activation_scale(static) is True
    # The warning is informational; a second call still reports the static scale.
    assert upcast._warn_if_static_activation_scale(static) is True


def test_reconstruct_bf16_skips_budget_check_on_cpu():
    # The CPU path is never budget-gated (host RAM is not the concern); a shape
    # that would be refused on a CUDA device must reconstruct fine on CPU.
    n, k = 4, 6
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    source = _FakeLinearMethod(weight)

    out = upcast.reconstruct_bf16(torch.nn.Identity(), source, n, k, torch.device("cpu"))

    torch.testing.assert_close(out, weight)


def test_reconstruct_bf16_raises_budget_error_when_cuda_free_memory_is_low(monkeypatch):
    # A quantized checkpoint with a mineable dense linear can pass the shape gate
    # yet OOM the worker during reconstruction (packed source weights are already
    # resident). reconstruct_bf16 must refuse *before* allocating the workspace
    # so the caller can fall back to serving via the source scheme.
    n, k = 8192, 8192  # destination alone is ~128 MB BF16
    source = _FakeLinearMethod(torch.zeros(n, k, dtype=torch.bfloat16))

    class _FakeCuda:
        @staticmethod
        def mem_get_info(_device):
            # 100 MB free -> budget = 50 MB, but peak workspace is ~384 MB.
            return 100_000_000, 10_000_000_000

        @staticmethod
        def synchronize(_device=None):
            pass

    monkeypatch.setattr(upcast.torch, "cuda", _FakeCuda)
    # Track whether the destination tensor was allocated; the budget check must
    # raise before that happens. Use the real torch.empty under the hood.
    real_empty = torch.empty
    allocated: list[bool] = []

    def _tracking_empty(*args, **kwargs):
        allocated.append(True)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(upcast.torch, "empty", _tracking_empty)

    with pytest.raises(upcast.UpcastBudgetError, match="upcast budget"):
        upcast.reconstruct_bf16(torch.nn.Identity(), source, n, k, torch.device("cuda:0"))
    assert allocated == [], "workspace was allocated before the budget check ran"


def test_estimate_reconstruct_peak_bytes_counts_destination_plus_one_chunk():
    # Peak = destination (n*k*2 BF16) + identity chunk (chunk*k*2 BF16) + delegate
    # output (chunk*n*4, charged at FP32 since fp8/CT kernels can return FP32) +
    # its BF16 conversion copy (chunk*n*2). k smaller than chunk_rows shrinks it.
    assert upcast._estimate_reconstruct_peak_bytes(8, 4, 4096) == (
        8 * 4 * 2 + 4 * 4 * 2 + 4 * 8 * 4 + 4 * 8 * 2
    )
    # k larger than chunk_rows: the chunk is capped at chunk_rows.
    n, k, chunk = 8192, 100_000, 4096
    assert upcast._estimate_reconstruct_peak_bytes(n, k, chunk) == (
        n * k * 2 + chunk * k * 2 + chunk * n * 4 + chunk * n * 2
    )
