"""sm_89-native pearl miner driver.

Bypasses vllm-miner's `pearl_gemm_noisy` wrapper (which always uses fp16
noising, the path our sm_89 build doesn't have an inst for). Calls
`pearl_gemm_cuda.noisy_gemm()` directly with int32 noising tensors so
the dispatch resolves to the sm_89 inst files:
  - noisingA 64x64 R=64 int32 stages=2
  - noisingB 64x64 R=64 int32 stages=2
  - gemm 128x128x64 R=64 stages=2 (PoW path, SkipReduction=False)

Connects to alphapool via the pearl-stratum shim. Logs main_TOPS, attempts/s,
and submitted-shares count every 30s.

Run inside container pearl-ab on CPU02. Decoy wallet — safe to run alongside
production rigs (they keep using the main wallet).
"""
from __future__ import annotations

import logging
import os
import sys
import time

# pearl-stratum shim path
sys.path.insert(0, "/host_home/pearl-deploy/vllm-miner/src")

import pearl_stratum.gateway_shim as _shim
import miner_base.gateway_client as _gc
_gc.MiningClient = _shim.StratumGatewayClient  # monkey-patch BEFORE miner_base imports

import torch
import pearl_gemm_cuda as pg
from pearl_stratum.stratum_client import StratumClient
from pearl_stratum.gateway_shim import init_shared_state
from vllm_miner.mining_state import (
    get_async_manager,
    init_async_manager,
    init_pinned_pool,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("pearl_miner_sm89")

DECOY = "prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg"
HOST = os.environ.get("PEARL_POOL_HOST", "us2.alphapool.tech")
PORT = int(os.environ.get("PEARL_POOL_PORT", "5566"))
WORKER = os.environ.get("PEARL_WORKER", "cpu02-sm89")
DEVICE = torch.device("cuda:0")

# Mining tile — sm_89 inst constraints:
#   matmul tile 128x128x64 R=64 stages=2 (PoW) or 3 (Noiseless)
#   noising 64x64 R=64 int32 stages=2
# Chunk M=N=2048 K=4096 R=64 (mining params from pool come at M=N=131072 K=4096
# — we chunk to 2048).
CHUNK_M, CHUNK_N, CHUNK_K, R = 2048, 2048, 4096, 64


def main() -> int:
    cap = torch.cuda.get_device_capability(0)
    log.info("GPU: %s cap=%s", torch.cuda.get_device_name(0), cap)
    log.info("pearl_gemm_cuda _min_compute_capability=%s",
             getattr(pg, "_min_compute_capability", "?"))
    assert cap == (8, 9), f"Expected sm_89, got cap={cap}"

    # 1) stratum + shared state
    client = StratumClient(
        host=HOST, port=PORT,
        address=DECOY, worker=WORKER,
        password="x;d=1048576",
        user_agent="pearl-stratum-sm89/0.1",
    )
    state = init_shared_state(client)
    if not state.wait_for_first_job(timeout=60.0):
        log.error("no mining.notify in 60s; bailing")
        return 1
    mp = state._client.mining_params
    if mp is None:
        log.error("pool never sent pearl.set_mining_params; bailing")
        return 2
    pool_M, pool_N, pool_K = int(mp["m"]), int(mp["n"]), int(mp["k"])
    log.info("pool params: M=%d N=%d K=%d rank=%s",
             pool_M, pool_N, pool_K, mp.get("rank"))

    # 2) pinned pool + async manager (uses our shim)
    init_pinned_pool(128)
    init_async_manager()
    mgr = get_async_manager()

    # 3) pre-allocate sm_89 buffers (re-used per attempt)
    M, N, K = CHUNK_M, CHUNK_N, CHUNK_K
    EAL = torch.zeros(M, R, dtype=torch.int8, device=DEVICE)
    EBR = torch.zeros(N, R, dtype=torch.int8, device=DEVICE)
    EAL_fp16 = torch.zeros(M, R, dtype=torch.float16, device=DEVICE)
    EBR_fp16 = torch.zeros(N, R, dtype=torch.float16, device=DEVICE)
    EAR_R_major = torch.zeros(K, R, dtype=torch.int8, device=DEVICE)
    EBL_R_major = torch.zeros(K, R, dtype=torch.int8, device=DEVICE)
    EAR_K_major = torch.zeros(R, K, dtype=torch.int8, device=DEVICE)
    EBL_K_major = torch.zeros(R, K, dtype=torch.int8, device=DEVICE)
    AxEBL_fp16 = torch.zeros(M, R, dtype=torch.float16, device=DEVICE)
    EARxBpEB_fp16 = torch.zeros(N, R, dtype=torch.float16, device=DEVICE)
    AxEBL_int32 = torch.zeros(M, R, dtype=torch.int32, device=DEVICE)
    EARxBpEB_int32 = torch.zeros(N, R, dtype=torch.int32, device=DEVICE)
    ApEA = torch.zeros(M, K, dtype=torch.int8, device=DEVICE)
    BpEB = torch.zeros(N, K, dtype=torch.int8, device=DEVICE)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device=DEVICE)

    hh_size = pg.get_host_signal_header_size()
    hs_size = pg.get_host_signal_sync_size()
    host_signal_header = torch.zeros(hh_size, dtype=torch.int8, pin_memory=True)
    host_signal_sync = torch.zeros(hs_size, dtype=torch.int8, device=DEVICE)

    # Trivial PoW target (any hash passes) for warm-up; in production the pool
    # sends the real target via mining.set_target which the kernel applies via
    # host_signal_header.
    pow_target = torch.full((8,), 0xFFFFFFFF, dtype=torch.uint32, device=DEVICE)

    # 4) main loop
    n_attempts = 0
    t_start = time.time()
    last_log = t_start
    log.info("entering mining loop: chunk M=%d N=%d K=%d", M, N, K)

    while True:
        # Mint fresh random (A, B) per attempt — the PoW commitment is over
        # these matrices via tensor_hash inside noisy_gemm.
        A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=DEVICE)
        B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=DEVICE)
        A_scales = torch.rand(M, dtype=torch.float32, device=DEVICE) * 0.02 + 0.005
        B_scales = torch.rand(N, dtype=torch.float32, device=DEVICE) * 0.02 + 0.005
        pow_key = torch.zeros(8, dtype=torch.uint32, device=DEVICE)

        try:
            pg.noisy_gemm(
                A, B, EAL, EAL_fp16, EBR, EBR_fp16,
                EAR_R_major, EBL_R_major, EAR_K_major, EBL_K_major,
                AxEBL_fp16, EARxBpEB_fp16, ApEA, BpEB,
                A_scales, B_scales, C,
                host_signal_header, host_signal_sync,
                pow_target, pow_key,
                AxEBL_int32, EARxBpEB_int32,
                128, 128, 64, 1, 1, 2,         # bM, bN, bK, cM, cN, pipeline_stages (PoW=2)
                None, True,                      # swizzle, swizzle_n_maj
                64, 64, 64, 64,                  # noisingA/B tile sizes (sm_89 inst)
                2, 2,                            # noisingA/B pipeline_stages
                None, None,                      # k_blocks_per_split
                True, True,                      # run_noising_a, run_noising_b
                False, False,                    # skip_reduction, skip_denoising (PoW)
                None, False,                     # inner_hash_counter, enable_debug
            )
        except Exception as e:
            log.exception("noisy_gemm failed: %s", e)
            return 3

        n_attempts += 1
        now = time.time()
        if now - last_log >= 30.0:
            elapsed = now - t_start
            rate = n_attempts / elapsed
            ops_per_attempt = 2.0 * M * N * K
            tops = ops_per_attempt * rate * 1e-12
            log.info(
                "stats: attempts=%d rate=%.2f/s main_TOPS=%.2f blocks_submitted=%d",
                n_attempts, rate, tops, mgr.blocks_submitted,
            )
            last_log = now


if __name__ == "__main__":
    sys.exit(main() or 0)
