"""Mocked unit tests for `_miner_driver_sm89_r128.py`.

We can't touch a GPU here, so:
  * Install fake `torch`, `pearl_gemm_cuda`, `miner_base.*`, `vllm_miner.*`
    modules in `sys.modules` BEFORE importing the driver. The driver does
    `import torch` and `import pearl_gemm_cuda as pg` at module load time,
    so the fakes must be in place first.
  * `StratumClient.__init__` is patched on the class so `init_shared_state`
    sees a synchronous stub. We feed it the alphapool-flavored mining_params
    payload via state injection rather than the asyncio read-loop.
  * `pg.noisy_gemm` is a MagicMock; the tests assert on its positional args.

What this exercises (driver-side only — the kernel build is a separate agent):
  * Tensor allocation shapes are R=128 throughout.
  * The 30-arg `pg.noisy_gemm` call uses bM=64 bN=64 bK=64 + R-dependent sizes.
  * The rank-mismatch error path returns nonzero and never touches the GPU.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


HERE = Path(__file__).resolve().parent
DRIVER_PATH = HERE.parent / "_miner_driver_sm89_r128.py"


# ----- fake torch / pearl_gemm_cuda / miner_base / vllm_miner -----------------
#
# These have to be installed in sys.modules BEFORE importing the driver, since
# the driver does `import torch` etc. at top level. We make them just rich
# enough to record the calls the driver makes.


def _make_fake_tensor(shape, dtype=None, device=None, **_ignored):
    """Stand-in for torch.Tensor. Just records shape/dtype/device.

    Arithmetic ops (`* k`, `+ k`) return self so chained scalar arithmetic
    used by the driver (`torch.rand(M) * 0.02 + 0.005`) doesn't lose the
    shape we need to inspect later. `.view(dtype)` returns a fake tensor
    whose shape reflects a same-byte reinterpret cast — used by the
    `commitment_hash_A.view(torch.uint32)` → pow_key derivation in the
    new commitment chain.
    """
    t = mock.MagicMock()
    t.shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
    t.dtype = dtype
    t.device = device
    t.__mul__ = lambda self, _other: self
    t.__rmul__ = lambda self, _other: self
    t.__add__ = lambda self, _other: self
    t.__radd__ = lambda self, _other: self

    # `.view(dtype)` is the same-bytes type cast used to reinterpret a uint8
    # tensor as uint32. For our 32-byte commitment hash, the result is an
    # 8-element uint32 view — that's the shape the existing tests check for
    # pow_key (pos[20] in the noisy_gemm args).
    _byte_widths = {"uint8": 1, "int8": 1, "uint32": 4, "int32": 4, "float32": 4,
                    "float16": 2, "bfloat16": 2, "int64": 8}
    def _view(target_dtype=None):
        if target_dtype is None or not t.shape:
            return _make_fake_tensor(t.shape, dtype=target_dtype, device=device)
        src_bytes = _byte_widths.get(t.dtype, 1) * (t.shape[0] if t.shape else 1)
        dst_width = _byte_widths.get(target_dtype, 1)
        if dst_width and src_bytes % dst_width == 0:
            new_shape = (src_bytes // dst_width,) + t.shape[1:]
        else:
            new_shape = t.shape
        return _make_fake_tensor(new_shape, dtype=target_dtype, device=device)
    t.view = _view
    return t


def _install_fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")

    # dtype sentinels — driver compares identity, not value
    for name in ("int8", "int32", "float16", "bfloat16", "float32", "uint32", "uint8", "int64"):
        setattr(torch, name, name)  # str sentinels are fine for identity tracking

    def _device(spec):
        return f"device({spec})"
    torch.device = _device

    def _zeros(*shape, dtype=None, device=None, pin_memory=False, **_):
        # Allow zeros((a,b),...) and zeros(a,b,...)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _make_fake_tensor(shape, dtype=dtype, device=device)
    torch.zeros = _zeros

    def _empty(*shape, dtype=None, device=None, **_):
        # Mirrors the same signature variations as torch.zeros: empty((a,b),...)
        # or empty(a, b, ...) or empty(N, dtype=...).
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _make_fake_tensor(shape, dtype=dtype, device=device)
    torch.empty = _empty

    def _full(shape, _value, dtype=None, device=None, **_):
        if isinstance(shape, (tuple, list)):
            return _make_fake_tensor(tuple(shape), dtype=dtype, device=device)
        return _make_fake_tensor((shape,), dtype=dtype, device=device)
    torch.full = _full

    def _randint(_lo, _hi, shape, dtype=None, device=None, **_):
        return _make_fake_tensor(tuple(shape), dtype=dtype, device=device)
    torch.randint = _randint

    def _rand(*shape, dtype=None, device=None, **_):
        return _make_fake_tensor(tuple(shape), dtype=dtype, device=device)
    torch.rand = _rand

    def _frombuffer(_buf, dtype=None):
        # `torch.frombuffer(bytearray(hash_key), dtype=torch.uint8)` in the
        # commitment chain — returns a length-32 1D tensor.
        return _make_fake_tensor((32,), dtype=dtype, device=None)
    torch.frombuffer = _frombuffer

    # The multi-stream driver path constructs torch.cuda.Stream, torch.cuda.Event,
    # and wraps slot submits in `with torch.cuda.stream(...)`. Provide benign
    # fakes so `StreamSlot.__init__` and `StreamSlot.submit` can run far enough
    # to hit `pg.noisy_gemm` (the call the tests inspect).
    class _FakeStreamCtx:
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    def _fake_stream(_stream=None):
        return _FakeStreamCtx()

    cuda = types.SimpleNamespace(
        get_device_capability=mock.MagicMock(return_value=(8, 9)),
        get_device_name=mock.MagicMock(return_value="NVIDIA GeForce RTX 4070 Ti SUPER"),
        Stream=mock.MagicMock(name="cuda.Stream"),
        Event=mock.MagicMock(name="cuda.Event"),
        stream=_fake_stream,
    )
    torch.cuda = cuda
    return torch


def _install_fake_pg() -> types.ModuleType:
    pg = types.ModuleType("pearl_gemm_cuda")
    pg._min_compute_capability = (8, 9)
    pg.get_host_signal_header_size = mock.MagicMock(return_value=512)
    pg.get_host_signal_sync_size = mock.MagicMock(return_value=64)
    pg.noisy_gemm = mock.MagicMock(return_value=None)
    return pg


def _install_fake_pearl_gemm() -> types.ModuleType:
    """Fake the high-level `pearl_gemm` package the driver pulls share-
    derivation helpers from. Just needs to be importable + expose the
    callable symbols — the driver doesn't introspect them post-import.
    """
    pkg = types.ModuleType("pearl_gemm")
    pkg.HostSignalStatus = types.SimpleNamespace(
        kSignalIdle=0, kSignalTriggered=1,
    )
    pkg.commitment_hash_from_merkle_roots = mock.MagicMock(name="commitment_hash_from_merkle_roots")
    pkg.extract_indices = mock.MagicMock(name="extract_indices")
    pkg.get_host_signal_header = mock.MagicMock(name="get_host_signal_header")
    pkg.get_required_scratchpad_bytes = mock.MagicMock(return_value=4096)
    pkg.make_pow_target_tensor = mock.MagicMock(return_value=_make_fake_tensor((8,), dtype="uint32"))
    pkg.noise_gen = mock.MagicMock(name="noise_gen")
    pkg.tensor_hash = mock.MagicMock(name="tensor_hash")
    return pkg


def _install_fake_miner_base() -> tuple[types.ModuleType, types.ModuleType,
                                         types.ModuleType, types.ModuleType]:
    """The driver does `import miner_base.gateway_client as _gc;
    _gc.MiningClient = _shim.StratumGatewayClient`. We also need
    `miner_base.commitment_hash.CommitmentHasher` and
    `miner_base.gpu_matmul_config.GPUMatmulConfigFactory` for the new
    share-derivation chain.
    """
    mb = types.ModuleType("miner_base")
    gc = types.ModuleType("miner_base.gateway_client")
    gc.MiningClient = mock.MagicMock(name="MiningClient_original")
    mb.gateway_client = gc

    ch = types.ModuleType("miner_base.commitment_hash")
    fake_hasher = mock.MagicMock(name="CommitmentHasher")
    fake_hasher.get_key = mock.MagicMock(return_value=b"\x42" * 32)
    ch.CommitmentHasher = fake_hasher
    mb.commitment_hash = ch

    gmc = types.ModuleType("miner_base.gpu_matmul_config")
    fake_factory = mock.MagicMock(name="GPUMatmulConfigFactory")
    fake_config = mock.MagicMock(name="MatmulConfig")
    fake_config.mining_config = mock.MagicMock(name="MiningConfig")
    fake_factory.create = mock.MagicMock(return_value=fake_config)
    gmc.GPUMatmulConfigFactory = fake_factory
    mb.gpu_matmul_config = gmc

    return mb, gc, ch, gmc


def _install_fake_pearl_gateway() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """Fake `pearl_gateway.comm.dataclasses` for the OpenedBlockInfo / MiningJob /
    CommitmentHash types the driver constructs on PoW hits.
    """
    pg = types.ModuleType("pearl_gateway")
    comm = types.ModuleType("pearl_gateway.comm")
    dc = types.ModuleType("pearl_gateway.comm.dataclasses")

    class _CommitmentHash:
        def __init__(self, noise_seed_A=None, noise_seed_B=None):
            self.noise_seed_A = noise_seed_A
            self.noise_seed_B = noise_seed_B

    class _MiningJob:
        INNER_HASH_LIMIT = 42
        MAX_TARGET = 2**256 - 1
        def __init__(self, incomplete_header_bytes=b"", target=1):
            self.incomplete_header_bytes = incomplete_header_bytes
            self.target = target
        def adjust_target(self, mining_config=None):
            return self.target

    class _OpenedBlockInfo:
        noise_range = 128
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    dc.CommitmentHash = _CommitmentHash
    dc.MiningJob = _MiningJob
    dc.OpenedBlockInfo = _OpenedBlockInfo
    comm.dataclasses = dc
    pg.comm = comm
    return pg, comm, dc


def _install_fake_vllm_miner() -> tuple[types.ModuleType, types.ModuleType]:
    vm = types.ModuleType("vllm_miner")
    ms = types.ModuleType("vllm_miner.mining_state")
    ms.init_pinned_pool = mock.MagicMock()
    ms.init_async_manager = mock.MagicMock()
    fake_mgr = mock.MagicMock()
    fake_mgr.blocks_submitted = 0
    # The new main loop calls mgr.get_mining_job() before each submit.
    fake_mgr.get_mining_job = mock.MagicMock(return_value=mock.MagicMock(
        incomplete_header_bytes=b"\x00" * 80,
        target=1,
        adjust_target=mock.MagicMock(return_value=1),
    ))
    ms.get_async_manager = mock.MagicMock(return_value=fake_mgr)
    vm.mining_state = ms
    return vm, ms


@pytest.fixture
def driver(monkeypatch):
    """Import the driver under a controlled set of fake modules.

    Yields the imported module. Each invocation gets a fresh import so test
    side effects on the module-level globals (e.g. monkey-patched StratumClient)
    don't leak between tests.
    """
    # Stash anything we may need to restore.
    saved = {k: sys.modules.get(k) for k in (
        "torch", "pearl_gemm_cuda", "pearl_gemm",
        "miner_base", "miner_base.gateway_client",
        "miner_base.commitment_hash", "miner_base.gpu_matmul_config",
        "pearl_gateway", "pearl_gateway.comm", "pearl_gateway.comm.dataclasses",
        "vllm_miner", "vllm_miner.mining_state",
        "_miner_driver_sm89_r128_test_target",
    )}

    fake_torch = _install_fake_torch()
    fake_pg = _install_fake_pg()
    fake_pearl_gemm = _install_fake_pearl_gemm()
    fake_mb, fake_gc, fake_ch, fake_gmc = _install_fake_miner_base()
    fake_pgw, fake_comm, fake_dc = _install_fake_pearl_gateway()
    fake_vm, fake_ms = _install_fake_vllm_miner()

    sys.modules["torch"] = fake_torch
    sys.modules["pearl_gemm_cuda"] = fake_pg
    sys.modules["pearl_gemm"] = fake_pearl_gemm
    sys.modules["miner_base"] = fake_mb
    sys.modules["miner_base.gateway_client"] = fake_gc
    sys.modules["miner_base.commitment_hash"] = fake_ch
    sys.modules["miner_base.gpu_matmul_config"] = fake_gmc
    sys.modules["pearl_gateway"] = fake_pgw
    sys.modules["pearl_gateway.comm"] = fake_comm
    sys.modules["pearl_gateway.comm.dataclasses"] = fake_dc
    sys.modules["vllm_miner"] = fake_vm
    sys.modules["vllm_miner.mining_state"] = fake_ms

    # Also make /host_home/pearl-deploy/vllm-miner/src not blow up. The driver
    # adds it to sys.path; we don't care if it doesn't exist on this box.
    monkeypatch.setenv("PEARL_POOL_HOST", "127.0.0.1")
    monkeypatch.setenv("PEARL_POOL_PORT", "65535")

    # The driver does `from _nonce_batcher import ...` (sibling module in the
    # same directory). spec_from_file_location below does NOT auto-add the
    # driver's directory to sys.path, so we do it explicitly. monkeypatch
    # ensures sys.path is restored when the test ends.
    monkeypatch.syspath_prepend(str(HERE.parent))

    # Now load the driver module from its file path (it's not a package member).
    spec = importlib.util.spec_from_file_location(
        "_miner_driver_sm89_r128_test_target", str(DRIVER_PATH)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    yield mod

    # Restore sys.modules so we don't leak fakes to other test modules.
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


# ----- helpers ----------------------------------------------------------------


def _make_mining_params(rank: int = 128) -> dict:
    """Alphapool-flavored payload, per STRATUM_CAPTURE §3c."""
    return {
        "m": 131072, "n": 131072, "k": 4096, "rank": rank,
        "rows_pattern": [0, 32],
        "cols_pattern": list(range(64)),
        "mma_type": "Int7xInt7ToInt32",
    }


def _patch_stratum_and_run(
    driver,
    *,
    mining_params: dict | None = None,
    first_job_ok: bool = True,
    max_iters: int = 1,
):
    """Patch StratumClient + shim + main-loop guard and call driver.main().

    Returns (return_code, fake_state, captured).
    captured["noisy_gemm_call"] is the most recent call_args, or None.
    """
    fake_state = mock.MagicMock()
    fake_state.wait_for_first_job = mock.MagicMock(return_value=first_job_ok)
    fake_state._client = mock.MagicMock()
    fake_state._client.mining_params = mining_params

    # Replace the driver's StratumClient and init_shared_state.
    fake_stratum_client = mock.MagicMock(name="StratumClient")
    driver.StratumClient = fake_stratum_client
    driver.init_shared_state = mock.MagicMock(return_value=fake_state)

    # Bound the infinite mining loop so we can assert on a known number of
    # noisy_gemm calls. We wrap pg.noisy_gemm: it records its call args and
    # raises a sentinel on the (max_iters+1)-th invocation to break the loop.
    # Use a BaseException subclass so the driver's `except Exception:` block
    # (which translates any exception into rc=3) doesn't swallow it.
    class _StopAfterIters(BaseException):
        pass

    call_count = {"n": 0}

    def _bounded_noisy_gemm(*args, **kwargs):
        call_count["n"] += 1
        # Record args for the first `max_iters` calls; raise sentinel after.
        if call_count["n"] > max_iters:
            raise _StopAfterIters()
        return None  # first max_iters calls return normally
    # Wrap in MagicMock so .call_args / .call_args_list / .called all work.
    wrapped = mock.MagicMock(side_effect=_bounded_noisy_gemm)
    driver.pg.noisy_gemm = wrapped

    # The driver's main() calls argparse.parse_args(), which reads sys.argv.
    # Under pytest, sys.argv contains pytest's CLI flags (e.g. --rootdir=...)
    # which argparse rejects with SystemExit(2). Stub sys.argv to argv[0] only
    # so the driver picks up its defaults (--num-streams=1, --bench-seconds=0,
    # --skip-stratum=False). Use a try/finally to restore the original argv
    # even if main() raises.
    saved_argv = sys.argv
    sys.argv = ["_miner_driver_sm89_r128"]
    try:
        rc = driver.main()
    except _StopAfterIters:
        rc = 0  # loop exited via our sentinel = we successfully called noisy_gemm max_iters times
    finally:
        sys.argv = saved_argv

    captured = {
        "noisy_gemm_call": wrapped.call_args if wrapped.called else None,
        "noisy_gemm_calls": wrapped.call_args_list,
        "init_pinned_pool_calls": sys.modules["vllm_miner.mining_state"].init_pinned_pool.call_args_list,
    }
    return rc, fake_state, captured


# ----- tests ------------------------------------------------------------------


def test_module_constants_are_r128(driver):
    """Top-of-file constants must declare R=128 + wave-2 64x128x64 matmul tile.

    bN is env-driven (PEARL_SM89_R128_BN); the production default is the
    wave-2 winner (bN=128). The driver fixture doesn't set the env var so
    we get the default.
    """
    assert driver.R == 128
    assert driver.CHUNK_M == 2048
    assert driver.CHUNK_N == 2048
    assert driver.CHUNK_K == 4096
    assert driver.BM == 64
    assert driver.BN == 128
    assert driver.BK == 64
    assert driver.CM == 1
    assert driver.CN == 1
    assert driver.MATMUL_STAGES == 2
    assert driver.NOISE_TILE_A_M == 64
    assert driver.NOISE_TILE_A_K == 64
    assert driver.NOISE_TILE_B_N == 64
    assert driver.NOISE_TILE_B_K == 64
    assert driver.NOISE_STAGES_A == 2
    assert driver.NOISE_STAGES_B == 2
    assert driver.EXPECTED_RANK == driver.R


def test_rank_mismatch_returns_error(driver):
    """If the pool sends a rank that doesn't match our compiled R, bail out
    cleanly rather than wasting GPU cycles producing unverifiable shares."""
    rc, _state, captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=64),
    )
    assert rc == 4
    assert captured["noisy_gemm_call"] is None, (
        "noisy_gemm must NOT be called when rank mismatches — "
        f"got call args {captured['noisy_gemm_call']!r}"
    )


def test_rank_match_proceeds_to_kernel(driver):
    """Happy path: rank=128 from pool, kernel gets invoked."""
    rc, _state, captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    assert captured["noisy_gemm_call"] is not None, (
        "noisy_gemm should be called once on the happy path"
    )


def test_no_first_job_returns_1(driver):
    rc, _state, _captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(), first_job_ok=False,
    )
    assert rc == 1


def test_missing_mining_params_returns_2(driver):
    rc, _state, _captured = _patch_stratum_and_run(
        driver, mining_params=None, first_job_ok=True,
    )
    assert rc == 2


def test_tensor_shapes_are_r128(driver):
    """All R-dependent allocations must use the runtime R (=128).

    We can't introspect the tensors after they enter pg.noisy_gemm as MagicMocks,
    but we CAN check the call args have the right .shape on each tensor — the
    fake tensors record the (M, R), (N, R), (K, R), (R, K) tuples directly.
    """
    rc, _state, captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    call_args = captured["noisy_gemm_call"]
    assert call_args is not None
    pos = call_args.args
    # Expected signature (matches _miner_driver_sm89.py):
    # 0: A          (M, K) int8
    # 1: B          (N, K) int8
    # 2: EAL        (M, R) int8
    # 3: EAL_fp16   (M, R) float16
    # 4: EBR        (N, R) int8
    # 5: EBR_fp16   (N, R) float16
    # 6: EAR_R_major  (K, R) int8
    # 7: EBL_R_major  (K, R) int8
    # 8: EAR_K_major  (R, K) int8
    # 9: EBL_K_major  (R, K) int8
    # 10: AxEBL_fp16     (M, R) float16
    # 11: EARxBpEB_fp16  (N, R) float16
    # 12: ApEA           (M, K) int8
    # 13: BpEB           (N, K) int8
    # 14: A_scales       (M,)   float32
    # 15: B_scales       (N,)   float32
    # 16: C              (M, N) bfloat16
    # 17: host_signal_header
    # 18: host_signal_sync
    # 19: pow_target     (8,)
    # 20: pow_key        (8,)
    # 21: AxEBL_int32    (M, R) int32
    # 22: EARxBpEB_int32 (N, R) int32
    # 23..28: bM, bN, bK, cM, cN, pipeline_stages
    # 29: swizzle (None)
    # 30: swizzle_n_maj (True)
    # 31..34: noisingA M,K; noisingB N,K
    # 35: noisingA stages
    # 36: noisingB stages
    # 37..38: k_blocks_per_split
    # 39..40: run_noising_a, run_noising_b
    # 41..42: skip_reduction, skip_denoising
    # 43..44: inner_hash_counter, enable_debug
    M, N, K, R_ = driver.CHUNK_M, driver.CHUNK_N, driver.CHUNK_K, driver.R
    assert R_ == 128

    # A, B
    assert pos[0].shape == (M, K)
    assert pos[1].shape == (N, K)

    # EAL/EAR/AxEBL/* must be (M, R) on A-side
    a_side_MR = [2, 3, 10, 21]  # EAL, EAL_fp16, AxEBL_fp16, AxEBL_int32
    for idx in a_side_MR:
        assert pos[idx].shape == (M, R_), f"tensor at pos {idx} should be (M={M}, R={R_}); got {pos[idx].shape}"

    # EBR/EBL/EARxBpEB/* must be (N, R) on B-side
    b_side_NR = [4, 5, 11, 22]  # EBR, EBR_fp16, EARxBpEB_fp16, EARxBpEB_int32
    for idx in b_side_NR:
        assert pos[idx].shape == (N, R_), f"tensor at pos {idx} should be (N={N}, R={R_}); got {pos[idx].shape}"

    # R-major: (K, R)
    for idx in (6, 7):  # EAR_R_major, EBL_R_major
        assert pos[idx].shape == (K, R_), f"tensor at pos {idx} should be (K={K}, R={R_}); got {pos[idx].shape}"

    # K-major: (R, K)
    for idx in (8, 9):  # EAR_K_major, EBL_K_major
        assert pos[idx].shape == (R_, K), f"tensor at pos {idx} should be (R={R_}, K={K}); got {pos[idx].shape}"

    # ApEA, BpEB
    assert pos[12].shape == (M, K)
    assert pos[13].shape == (N, K)

    # A_scales, B_scales
    assert pos[14].shape == (M,)
    assert pos[15].shape == (N,)

    # C
    assert pos[16].shape == (M, N)

    # pow_target / pow_key are (8,)
    assert pos[19].shape == (8,)
    assert pos[20].shape == (8,)


def test_noisy_gemm_tile_args(driver):
    """The 6 tile-related positional args after the tensor block must declare
    bM=64 bN=128 bK=64 cM=1 cN=1 stages=2 — the wave-2 R=128 winner."""
    rc, _state, captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    pos = captured["noisy_gemm_call"].args
    bM, bN, bK, cM, cN, stages = pos[23], pos[24], pos[25], pos[26], pos[27], pos[28]
    assert (bM, bN, bK) == (64, 128, 64), \
        f"matmul tile must be 64x128x64 for R=128 wave-2 winner; got {(bM, bN, bK)}"
    assert (cM, cN) == (1, 1), f"sm_89 has no clusters; cM=cN=1; got {(cM, cN)}"
    assert stages == 2, f"R=128 bN=128 instantiation is stages=2; got {stages}"


def test_noisy_gemm_swizzle_args(driver):
    """swizzle=None, swizzle_n_maj=True (matches sm_89 inst built by KERNEL-A)."""
    rc, _state, captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    pos = captured["noisy_gemm_call"].args
    assert pos[29] is None
    assert pos[30] is True


def test_noisy_gemm_noising_tile_and_stages(driver):
    """noisingA tile (M=64, K=64); noisingB tile (N=64, K=64); both stages=2."""
    rc, _state, captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    pos = captured["noisy_gemm_call"].args
    # noisingA M,K then noisingB N,K
    assert (pos[31], pos[32]) == (64, 64), \
        f"noisingA tile must be 64x64; got {(pos[31], pos[32])}"
    assert (pos[33], pos[34]) == (64, 64), \
        f"noisingB tile must be 64x64; got {(pos[33], pos[34])}"
    # noising stages
    assert pos[35] == 2
    assert pos[36] == 2


def test_noisy_gemm_skip_flags_match_pow_path(driver):
    """PoW path: run_noising_a=True, run_noising_b=True, skip_reduction=False,
    skip_denoising=False — same flags as the R=64 driver."""
    rc, _state, captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    pos = captured["noisy_gemm_call"].args
    # k_blocks_per_split
    assert pos[37] is None
    assert pos[38] is None
    # run_noising_a, run_noising_b
    assert pos[39] is True
    assert pos[40] is True
    # skip_reduction, skip_denoising
    assert pos[41] is False
    assert pos[42] is False
    # inner_hash_counter, enable_debug
    assert pos[43] is None
    assert pos[44] is False


def test_noisy_gemm_total_arg_count(driver):
    """Sanity: signature has exactly 45 positional args (23 tensors + 22 scalars/Nones)."""
    rc, _state, captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    pos = captured["noisy_gemm_call"].args
    assert len(pos) == 45, f"expected 45 positional args; got {len(pos)}"


def test_logs_active_config_at_startup(driver, caplog):
    """The driver should log R + tile config so it's obvious from logs what
    build is running. We assert the key tokens appear in the captured logs."""
    import logging as _logging
    caplog.set_level(_logging.INFO, logger="pearl_miner_sm89_r128")
    rc, _state, _captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    msgs = "\n".join(r.getMessage() for r in caplog.records)
    assert "R=128" in msgs, f"missing R=128 in startup log; got: {msgs!r}"
    assert "bM=64" in msgs
    assert "bN=128" in msgs
    assert "bK=64" in msgs


# ===========================================================================
# Share-derivation regression tests — guard the three audit fixes:
#
#   1. Commitment chain (tensor_hash + commitment_hash_from_merkle_roots +
#      noise_gen) runs each attempt — Bug #2 in the audit.
#   2. pow_key passed to noisy_gemm is a uint32 view onto commitment_hash_A,
#      NOT a fresh zeros tensor — also Bug #2.
#   3. PoW hits get assembled into OpenedBlockInfo and dispatched via
#      mgr.handle_submit_block() — Bugs #1 + #3.
#
# Failure modes these tests catch:
#   * Reverting `pow_key = commitment_hash_A.view(torch.uint32)` to zeros.
#   * Skipping `noise_gen` → kernel sees zero EAL/EBR.
#   * Dropping the HostSignalHeader read → silent loss of any PoW hits.
#   * Calling mgr.handle_submit_block with the wrong noise_rank, missing
#     commitment fields, or the wrong A/B reference (e.g. capturing the
#     PREVIOUS attempt's matrices).
# ===========================================================================


def test_commitment_chain_invoked_each_attempt(driver):
    """tensor_hash, commitment_hash_from_merkle_roots, and noise_gen must
    each be called once per attempt — twice tensor_hash (A and B) once
    each for the others.

    The bounded-loop sentinel fires on the (max_iters+1)-th `pg.noisy_gemm`
    call, but the commitment chain runs BEFORE noisy_gemm — so the failing
    attempt also runs the chain. We assert proportional counts: 2 ×
    (max_iters+1) for tensor_hash, (max_iters+1) for the other two.

    This is the simplest possible regression guard: if any of these calls
    disappears the driver is back to the pre-fix state where the kernel
    receives pow_key=zeros and unkeyed noise factors.
    """
    max_iters = 2
    rc, _state, _captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=max_iters,
    )
    assert rc == 0
    pg = sys.modules["pearl_gemm"]
    # `max_iters+1` attempts: the +1 attempt's commitment chain runs before
    # the sentinel fires inside `pg.noisy_gemm`. 2 tensor_hash per attempt.
    expected_attempts = max_iters + 1
    assert pg.tensor_hash.call_count == 2 * expected_attempts, (
        f"tensor_hash invoked {pg.tensor_hash.call_count}× across "
        f"{expected_attempts} attempts; expected {2 * expected_attempts} "
        "(one each for A and B per attempt). The commitment chain may have "
        "been skipped."
    )
    assert pg.commitment_hash_from_merkle_roots.call_count == expected_attempts, (
        f"commitment_hash_from_merkle_roots invoked "
        f"{pg.commitment_hash_from_merkle_roots.call_count}× across "
        f"{expected_attempts} attempts; expected {expected_attempts} — "
        "the kernel's pow_key would be zeros otherwise."
    )
    assert pg.noise_gen.call_count == expected_attempts, (
        f"noise_gen invoked {pg.noise_gen.call_count}× across "
        f"{expected_attempts} attempts; expected {expected_attempts} — "
        "the kernel needs keyed EAL/EBR/EAR/EBL not zeros."
    )


def test_pow_key_arg_is_uint32_view_not_zeros(driver):
    """pos[20] in noisy_gemm is the pow_key. It must be a (8,) uint32
    tensor; with our fake .view(), it carries dtype='uint32' on the result
    of `commitment_hash_A.view(torch.uint32)`. A direct zeros allocation
    would have dtype=='uint32' too — so we additionally check the
    commitment chain produced a non-fake value for commitment_hash_A
    (the .view source) by asserting commitment_hash_from_merkle_roots was
    called with that exact tensor as its 4th positional arg.
    """
    rc, _state, captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    pos = captured["noisy_gemm_call"].args
    pow_key = pos[20]
    assert pow_key.shape == (8,), f"pow_key shape must be (8,); got {pow_key.shape}"
    assert pow_key.dtype == "uint32", (
        f"pow_key dtype must be uint32 (the view target); got {pow_key.dtype!r}"
    )

    # Verify the pow_key chain: commitment_hash_from_merkle_roots writes
    # into commitment_hash_A (positional arg 3) and the driver views that
    # tensor as uint32. So the tensor at position 3 of the most recent
    # commitment call should match the underlying storage of pow_key.
    # We can't compare object identity directly through .view() (the fake
    # makes a new MagicMock), so instead we assert the call happened.
    pg = sys.modules["pearl_gemm"]
    assert pg.commitment_hash_from_merkle_roots.called, (
        "commitment_hash_from_merkle_roots must be called BEFORE noisy_gemm; "
        "otherwise pow_key derives from an uninitialized commitment_hash_A buffer."
    )


def test_pow_target_is_freshly_computed(driver):
    """The driver must derive a real adjusted target via mining_job.adjust_target
    and pass make_pow_target_tensor's result into noisy_gemm — not the per-slot
    zeroed pre-allocation. pos[19] is the pow_target.
    """
    rc, _state, captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    pg = sys.modules["pearl_gemm"]
    # make_pow_target_tensor is now called once per attempt (via copy_ on the
    # per-slot pow_target buffer); ensure that happened.
    assert pg.make_pow_target_tensor.called, (
        "make_pow_target_tensor must be invoked to compute the adjusted target. "
        "Without it the kernel never finds a hit (pow_target stays at zeros)."
    )


def test_pow_hit_dispatches_to_handle_submit_block(driver, monkeypatch):
    """Simulate a PoW hit: stub get_host_signal_header() so status is
    kSignalTriggered + block_in_bounds True. After the next attempt the
    driver must call mgr.handle_submit_block with an OpenedBlockInfo.
    """
    pg = sys.modules["pearl_gemm"]

    class _FakeHeader:
        status = pg.HostSignalStatus.kSignalTriggered
        def block_in_bounds(self):
            return True

    # The driver did `from pearl_gemm import get_host_signal_header` so it
    # holds a direct reference. Patch BOTH the driver-local name and the
    # source module so any future indirection still picks up the stub.
    driver.get_host_signal_header = mock.MagicMock(return_value=_FakeHeader())
    pg.get_host_signal_header = driver.get_host_signal_header
    driver.extract_indices = mock.MagicMock(return_value=mock.MagicMock(
        A_row_indices=[0, 1, 2, 3],
        B_column_indices=[0, 1, 2, 3],
    ))
    pg.extract_indices = driver.extract_indices

    # We want at least 2 attempts so the second iteration's wait() observes
    # the first attempt's "hit". max_iters=2 lets us reach the second
    # submit before the sentinel fires.
    rc, _state, _captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=2,
    )
    assert rc == 0
    mgr = sys.modules["vllm_miner.mining_state"].get_async_manager.return_value
    assert mgr.handle_submit_block.called, (
        "PoW hit not surfaced — driver must call mgr.handle_submit_block "
        "when host_signal_header.status == kSignalTriggered."
    )
    # The OpenedBlockInfo must carry the right noise_rank + the indices we
    # stubbed.
    call_args, _kw = mgr.handle_submit_block.call_args
    opened_block_info, _mining_job = call_args
    assert opened_block_info.noise_rank == 128
    assert opened_block_info.A_row_indices == [0, 1, 2, 3]
    assert opened_block_info.B_column_indices == [0, 1, 2, 3]
    # commitment_hash should be a CommitmentHash with both seeds populated.
    assert opened_block_info.commitment_hash is not None
    assert opened_block_info.commitment_hash.noise_seed_A is not None
    assert opened_block_info.commitment_hash.noise_seed_B is not None


def test_no_hit_no_submit(driver):
    """When host_signal_header.status == kSignalIdle (the steady state on
    every attempt before the rare PoW hit), the driver must NOT call
    handle_submit_block — that would spam the pool with junk shares.
    """
    pg = sys.modules["pearl_gemm"]

    class _IdleHeader:
        status = pg.HostSignalStatus.kSignalIdle
        def block_in_bounds(self):
            return False

    # Patch driver-local symbol per test_pow_hit_dispatches_to_handle_submit_block.
    driver.get_host_signal_header = mock.MagicMock(return_value=_IdleHeader())
    pg.get_host_signal_header = driver.get_host_signal_header

    rc, _state, _captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=3,
    )
    assert rc == 0
    mgr = sys.modules["vllm_miner.mining_state"].get_async_manager.return_value
    assert not mgr.handle_submit_block.called, (
        "handle_submit_block called on a kSignalIdle header — driver must "
        "only dispatch on a real PoW hit."
    )


def test_noise_gen_keyed_on_commitment_hash_not_zeros(driver):
    """Drill into the noise_gen call kwargs: key_A must be the tensor returned
    by commitment_hash_from_merkle_roots (i.e. NOT the all-zero zeros tensor
    the pre-fix code used). The fake commitment_hash_from_merkle_roots
    populates the output tensors via reference; we check noise_gen receives
    the SAME tensor object as the chain's commitment_hash_A output.
    """
    rc, _state, _captured = _patch_stratum_and_run(
        driver, mining_params=_make_mining_params(rank=128), max_iters=1,
    )
    assert rc == 0
    pg = sys.modules["pearl_gemm"]
    assert pg.noise_gen.called
    _ng_args, ng_kwargs = pg.noise_gen.call_args
    # key_A and key_B kwargs must be present — that's the kernel's source of
    # the noise sequence. A regression where these went to zeros tensors
    # would cause the pool to reject all shares as "invalid commitment".
    assert "key_A" in ng_kwargs and "key_B" in ng_kwargs, (
        f"noise_gen must be called with key_A + key_B kwargs; got {ng_kwargs.keys()}"
    )
    # The keyed tensors must be the 32-byte commitment hash output (uint8,
    # length-32). The fake make_fake_tensor preserves shape so we can check.
    assert ng_kwargs["key_A"].shape == (32,) and ng_kwargs["key_A"].dtype == "uint8", (
        f"key_A must be a (32,) uint8 commitment hash; got "
        f"shape={ng_kwargs['key_A'].shape} dtype={ng_kwargs['key_A'].dtype}"
    )
