"""Tests for graph setup mining guards."""

from vllm_miner.capture import graph_setup_no_mining, in_graph_setup_no_mining


def test_graph_setup_no_mining_context_is_reentrant():
    assert not in_graph_setup_no_mining()
    with graph_setup_no_mining():
        assert in_graph_setup_no_mining()
        with graph_setup_no_mining():
            assert in_graph_setup_no_mining()
        assert in_graph_setup_no_mining()
    assert not in_graph_setup_no_mining()
