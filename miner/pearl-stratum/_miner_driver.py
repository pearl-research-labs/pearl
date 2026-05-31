"""Minimal Pearl miner driver: drives `pearl_gemm_noisy` in a loop against
random A/B inputs while letting `AsyncLoopManager` route found-shares to
alphapool via our `pearl-stratum` shim.

Run inside the pearl-ab container with the Python 3.12 venv active.

This skips vLLM entirely. The kernel doesn't care that A/B come from random
init vs LLM activations — the mining work and pool-credited hashrate are the
same. Pearl mining IS the matmul work; vLLM is just a particular workload that
happens to do many matmuls.

Goal: 30-min run vs alpha-miner's `cpu01-ab-baseline` so we can compare
pool-credited TH/s on the SAME decoy wallet.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import threading

# vllm-miner src on path so we can import its modules without needing the full
# `pip install vllm-miner` (which would pull a multi-GB vLLM wheel we don't need).
sys.path.insert(0, "/host_home/pearl-deploy/vllm-miner/src")

# Install the gateway shim BEFORE any miner_base imports so AsyncLoopManager
# picks up our StratumGatewayClient instead of the real MiningClient.
import pearl_stratum.gateway_shim as _shim
import miner_base.gateway_client as _gc
_gc.MiningClient = _shim.StratumGatewayClient  # monkey-patch

import torch
from pearl_stratum.stratum_client import StratumClient
from pearl_stratum.gateway_shim import init_shared_state
from miner_base.settings import MinerSettings

# Now we can safely import vllm_miner code paths that read mining_state
from vllm_miner.mining_state import (
    get_async_manager,
    get_pinned_pool,
    init_async_manager,
    init_pinned_pool,
)
from vllm_miner.gemm_operators import pearl_gemm_noisy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("pearl_miner")

DECOY = "prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg"
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16


def main():
    # 1) bring up stratum
    client = StratumClient(
        host="us2.alphapool.tech", port=5566,
        address=DECOY,
        worker="cpu02-ours",
        password="x;d=1048576",
        user_agent="pearl-stratum/0.1",
    )
    state = init_shared_state(client)
    if not state.wait_for_first_job(timeout=60.0):
        log.error("no mining.notify in 60s; bailing")
        return 1

    # 2) bring up pinned pool + async loop manager (these read MinerSettings)
    # MinerSettings.no_gateway=False (default) → AsyncLoopManager will call
    # _make_client(...) which now returns our StratumGatewayClient thanks to
    # the monkey-patch above.
    init_pinned_pool(128)
    init_async_manager()  # starts thread + creates (shimmed) MiningClient
    mgr = get_async_manager()

    # 3) mining params come from `pearl.set_mining_params` push:
    mp = state._client.mining_params
    if mp is None:
        log.error("pool never sent pearl.set_mining_params; bailing")
        return 2
    M, N, K = int(mp["m"]), int(mp["n"]), int(mp["k"])
    log.info("mining params: M=%d N=%d K=%d rank=%d", M, N, K, mp.get("rank"))

    # Pool emits M=N=131072 K=4096. We chunk to a small mining tile per attempt.
    # Per the Tier 1a memo, 2048³ beats alpha-miner at the kernel level.
    CHUNK_M, CHUNK_N, CHUNK_K = 2048, 2048, K  # K=4096 from pool params

    # 4) main mining loop
    n_attempts = 0
    n_jackpots = 0
    t_start = time.time()
    last_log = t_start

    while True:
        # mint a fresh random (A, B) per attempt. The PoW commitment is over
        # these matrices via tensor_hash inside pearl_gemm_noisy.
        A = torch.randint(-127, 127, (CHUNK_M, CHUNK_K),
                          dtype=torch.int8, device=DEVICE)
        B = torch.randint(-127, 127, (CHUNK_N, CHUNK_K),
                          dtype=torch.int8, device=DEVICE)
        scale_a = torch.rand(CHUNK_M, dtype=torch.float32, device=DEVICE) * 0.02 + 0.005
        scale_b = torch.rand(CHUNK_N, dtype=torch.float32, device=DEVICE) * 0.02 + 0.005

        try:
            _C = pearl_gemm_noisy(A, B, scale_a, scale_b, DTYPE)
        except Exception as e:
            log.exception("pearl_gemm_noisy failed: %s", e)
            break

        n_attempts += 1
        now = time.time()
        if now - last_log >= 30.0:
            elapsed = now - t_start
            rate = n_attempts / elapsed
            ops_per_attempt = 2.0 * CHUNK_M * CHUNK_N * CHUNK_K  # main_gemm equivalent
            tops = ops_per_attempt * rate * 1e-12
            log.info(
                "stats: attempts=%d rate=%.2f/s main_TOPS=%.2f blocks_submitted=%d",
                n_attempts, rate, tops, mgr.blocks_submitted,
            )
            last_log = now


if __name__ == "__main__":
    sys.exit(main() or 0)
