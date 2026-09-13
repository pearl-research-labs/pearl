"""CUDA-graph capture for the caller-owned pipeline (no-winner + mining)."""

import pytest
import torch

from tests.helpers.pipeline import MinerPipeline
from tests.helpers.preprocess import default_config, preprocess

pytestmark = pytest.mark.gpu


@pytest.fixture
def pipeline_inputs():
    torch.manual_seed(7)
    m, n, k = 256, 128, 512
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    context = preprocess(b, config=default_config(k))
    pipeline = MinerPipeline(
        context,
        m,
        prequant_tune={},
        commit_tune={},
        prepare_tune={},
        gemm_tune={"cluster_m": 1, "cluster_n": 1},
    )
    return a, pipeline


def test_pipeline_no_winner_path_is_cuda_graph_capturable(pipeline_inputs):
    a, pipeline = pipeline_inputs
    pipeline.threshold.zero_()
    pipeline.run(a)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline.run(a)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(pipeline.c).all()
    assert pipeline.hit_signal.read_hit() is None


def test_pipeline_winning_path_is_cuda_graph_capturable(pipeline_inputs):
    a, pipeline = pipeline_inputs
    # Max threshold: every tile wins, so each launch must publish a winner.
    pipeline.threshold.fill_(0xFF)
    pipeline.run(a)
    torch.cuda.synchronize()
    assert pipeline.hit_signal.read_hit() is not None

    # ``reset_hit()`` synchronizes the consumer stream and is not
    # graph-capturable. Clear before capture so ``stage_gemm`` does not
    # reset inside the capture region; clear again between replays.
    pipeline.hit_signal.reset_hit()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline.run(a)

    for _ in range(3):
        pipeline.hit_signal.reset_hit()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(pipeline.c).all()
        assert pipeline.hit_signal.read_hit() is not None
