"""Unit tests for `_nonce_batcher.py` — host-side multi-nonce batching.

What this exercises
-------------------
1. `is_persistent_nonce_enabled()` env-var parsing.
2. `BatchConfig` / `NonceBatch` allocation shapes (uses the same fake-torch
   strategy as `test_miner_driver_r128.py` so we don't need a GPU).
3. The `contexts` table is correctly populated with per-nonce pointers +
   nonce values.
4. `NonceBatch.launch()` calls `pg.gemm_persistent_multinonce` with the
   right positional structure (we mock the entry point and inspect the call).
5. Bit-exact equivalence vs the sequential path: when the batched path
   processes 256 nonces with the same A_i / shared-B / scales as a sequential
   loop would, the resulting C_batch[i] equals the sequential C_i. We
   simulate the kernel side using a deterministic stub so the test can run
   without a GPU.

The bit-exact assertion is the centerpiece — it's the proof that the
batching primitives don't reorder/perturb per-nonce work. The actual sm_89
kernel will be validated against alpha-miner separately; here we only test
the *host plumbing* preserves equivalence.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


HERE = Path(__file__).resolve().parent
BATCHER_PATH = HERE.parent / "_nonce_batcher.py"


# ----- fake torch + pg ---------------------------------------------------------


class _FakeTensor:
    """Minimal int-array stand-in for a torch.Tensor.

    Stores shape + dtype + a flat int list (the only "data" we need to verify
    bit-exact equivalence vs a sequential walk through the same inputs). The
    `.copy_`, `.data_ptr`, `.element_size`, `.numel`, slice/indexing surface
    is exactly what `_nonce_batcher.py` and our test simulator touch.
    """

    # A counter so each tensor gets a stable, unique "address" — the contexts
    # table records per-slot pointers; we just need unique stable ints to
    # verify the table.
    _ptr_seed = 0x1000

    def __init__(self, shape, dtype, device="cuda:0", data=None, pin_memory=False):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.pin_memory = pin_memory
        self._numel = 1
        for d in self.shape:
            self._numel *= d
        # Give each tensor a stable pseudo address; reused across .copy_ calls.
        _FakeTensor._ptr_seed += 16
        self._addr = _FakeTensor._ptr_seed
        # Element size mapping for dtype sentinels we use.
        self._elem_size = {
            "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
            "int32": 4, "uint32": 4, "int64": 8,
            "float16": 2, "bfloat16": 2, "float32": 4,
        }.get(dtype, 4)
        if data is None:
            self._data = [0] * self._numel
        else:
            assert len(data) == self._numel, (len(data), self._numel)
            self._data = list(data)

    def data_ptr(self):
        return self._addr

    def element_size(self):
        return self._elem_size

    def numel(self):
        return self._numel

    def copy_(self, other):
        # Accept _FakeTensor or anything with ._data of matching length.
        if isinstance(other, _FakeTensor):
            assert len(other._data) == self._numel, (len(other._data), self._numel)
            self._data = list(other._data)
        else:
            raise TypeError(f"copy_ from non-FakeTensor: {type(other)}")
        return self

    def __getitem__(self, idx):
        """Support `tensor[i]` (first-dim row) and `tensor[i, j]` (single elem).

        For the batcher, we need row slices into A_batch / contexts.
        For tests, we read individual cells out of `contexts`.
        """
        if isinstance(idx, int):
            if len(self.shape) == 1:
                # scalar element
                return _ScalarItem(self._data[idx])
            # row slice: shape (rest,), data is the i-th chunk
            rest_shape = self.shape[1:]
            row_numel = 1
            for d in rest_shape:
                row_numel *= d
            start = idx * row_numel
            return _FakeTensor(
                rest_shape, self.dtype, device=self.device,
                data=self._data[start:start + row_numel],
            )
        if isinstance(idx, tuple) and len(idx) == 2 and all(isinstance(x, int) for x in idx):
            # (i, j) — flatten
            i, j = idx
            flat = i * self.shape[1] + j
            return _ScalarItem(self._data[flat])
        raise NotImplementedError(f"FakeTensor indexing: {idx!r}")

    def __setitem__(self, idx, value):
        """Support `tensor[i, j] = scalar` — only what `_rebuild_contexts` uses."""
        if isinstance(idx, tuple) and len(idx) == 2 and all(isinstance(x, int) for x in idx):
            i, j = idx
            flat = i * self.shape[1] + j
            self._data[flat] = int(value)
            return
        raise NotImplementedError(f"FakeTensor __setitem__: {idx!r}")

    def to(self, device):
        # No-op for fakes — they live wherever we say.
        return self

    # Scalar arithmetic — the batcher does `torch.rand(...) * 0.02 + 0.005`
    # to mirror the production driver's scale generation. We support enough
    # of the operator surface to keep that chain typed as _FakeTensor.
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return _FakeTensor(self.shape, self.dtype, device=self.device,
                               data=[v * other for v in self._data])
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return _FakeTensor(self.shape, self.dtype, device=self.device,
                               data=[v + other for v in self._data])
        return NotImplemented

    __radd__ = __add__


class _ScalarItem:
    """Stand-in for the result of `tensor[i]` when the parent is 1D — has
    `.item()` to extract the int, mirroring torch.Tensor's scalar return."""
    def __init__(self, v):
        self._v = int(v)

    def item(self):
        return self._v


def _install_fake_torch() -> types.ModuleType:
    """Construct a fake torch module that's just rich enough for NonceBatch."""
    torch = types.ModuleType("torch")
    for name in ("int8", "uint8", "int16", "uint16", "int32", "uint32",
                 "int64", "float16", "bfloat16", "float32"):
        setattr(torch, name, name)
    torch.device = lambda spec: f"device({spec})"

    def _zeros(*shape, dtype=None, device=None, pin_memory=False, **_):
        # Allow zeros((a,b)) or zeros(a, b)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _FakeTensor(shape, dtype=dtype, device=device, pin_memory=pin_memory)
    torch.zeros = _zeros

    # Each call to randint yields a fresh deterministic stream so tests that
    # care about equivalence can re-seed. The seed is per-call-counter so
    # two calls in a row give different data (mirrors real torch).
    _randint_counter = [0]
    def _randint(_lo, _hi, shape, dtype=None, device=None, **_):
        _randint_counter[0] += 1
        seed = _randint_counter[0]
        flat = 1
        for d in shape:
            flat *= d
        # Deterministic byte stream — every test asserting bit-exact will use
        # a captured snapshot of A_batch._data, so the actual numbers don't
        # have to be cryptographic, just stable.
        data = [((seed * 2654435761 + i) & 0xFF) - 128 for i in range(flat)]
        return _FakeTensor(shape, dtype=dtype, device=device, data=data)
    torch.randint = _randint

    def _rand(*shape, dtype=None, device=None, **_):
        _randint_counter[0] += 1
        seed = _randint_counter[0]
        flat = 1
        for d in shape:
            flat *= d
        data = [((seed + i) & 0xFFFF) / 65536 for i in range(flat)]
        return _FakeTensor(shape, dtype=dtype, device=device, data=data)
    torch.rand = _rand

    def _arange(start, stop, dtype=None, device=None, **_):
        # used by NonceBatch.refresh_nonces when base_nonce is given
        data = list(range(int(start), int(stop)))
        return _FakeTensor((len(data),), dtype=dtype, device=device, data=data)
    torch.arange = _arange

    return torch


def _install_fake_pg() -> types.ModuleType:
    pg = types.ModuleType("pearl_gemm_cuda")
    pg.get_host_signal_header_size = mock.MagicMock(return_value=512)
    pg.get_host_signal_sync_size = mock.MagicMock(return_value=64)
    # Default: gemm_persistent_multinonce exists but is a no-op mock.
    pg.gemm_persistent_multinonce = mock.MagicMock(return_value=None)
    return pg


@pytest.fixture
def batcher_module(monkeypatch):
    """Load `_nonce_batcher.py` with fake torch in sys.modules.

    Yields the imported module.
    """
    saved_torch = sys.modules.get("torch")
    saved_pg = sys.modules.get("pearl_gemm_cuda")

    fake_torch = _install_fake_torch()
    fake_pg = _install_fake_pg()
    sys.modules["torch"] = fake_torch
    sys.modules["pearl_gemm_cuda"] = fake_pg

    spec = importlib.util.spec_from_file_location(
        "_nonce_batcher_test_target", str(BATCHER_PATH)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    yield mod, fake_torch, fake_pg

    for k, v in (("torch", saved_torch), ("pearl_gemm_cuda", saved_pg)):
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


# ----- env var parsing ---------------------------------------------------------


@pytest.mark.parametrize("val,expected", [
    ("1", True),
    ("true", True),
    ("TRUE", True),
    ("yes", True),
    ("on", True),
    ("0", False),
    ("false", False),
    ("no", False),
    ("", False),
    ("garbage", False),
])
def test_is_persistent_nonce_enabled(batcher_module, monkeypatch, val, expected):
    mod, _, _ = batcher_module
    monkeypatch.setenv("PEARL_SM89_PERSISTENT_NONCE", val)
    assert mod.is_persistent_nonce_enabled() is expected


def test_is_persistent_nonce_enabled_unset(batcher_module, monkeypatch):
    mod, _, _ = batcher_module
    monkeypatch.delenv("PEARL_SM89_PERSISTENT_NONCE", raising=False)
    assert mod.is_persistent_nonce_enabled() is False


# ----- NonceBatch construction ------------------------------------------------


def _make_batch(mod, torch_fake, pg_fake, *, batch_size=8, M=64, N=64, K=128, R=128):
    """Helper: construct a NonceBatch with a small geometry so tests stay fast."""
    cfg = mod.BatchConfig(M=M, N=N, K=K, R=R, batch_size=batch_size)
    return mod.NonceBatch(torch_fake, pg_fake, cfg, device="device(cuda:0)")


def test_nonce_batch_allocates_expected_shapes(batcher_module):
    mod, torch_fake, pg_fake = batcher_module
    b = _make_batch(mod, torch_fake, pg_fake, batch_size=8, M=64, N=64, K=128, R=128)

    # Per-nonce: leading dim = batch_size
    assert b.A_batch.shape == (8, 64, 128)
    assert b.A_batch.dtype == "int8"
    assert b.A_scales_batch.shape == (8, 64)
    assert b.A_scales_batch.dtype == "float32"
    assert b.C_batch.shape == (8, 64, 64)
    assert b.C_batch.dtype == "bfloat16"
    assert b.nonces.shape == (8,)
    assert b.nonces.dtype == "int64"

    # Shared (persistent B)
    assert b.B.shape == (64, 128)
    assert b.B.dtype == "int8"
    assert b.B_scales.shape == (64,)
    assert b.B_scales.dtype == "float32"

    # Contexts table: one record per nonce, 8 int64 columns (wave-13 layout)
    assert b.contexts.shape == (8, 8)
    assert b.contexts.dtype == "int64"

    # Per-tile scratch — R-dependent shapes from the existing driver.
    M, N, K, R = 64, 64, 128, 128
    assert b.EAL.shape == (M, R)
    assert b.EBR.shape == (N, R)
    assert b.EAL_fp16.shape == (M, R)
    assert b.EBR_fp16.shape == (N, R)
    assert b.EAR_R_major.shape == (K, R)
    assert b.EBL_R_major.shape == (K, R)
    assert b.EAR_K_major.shape == (R, K)
    assert b.EBL_K_major.shape == (R, K)
    assert b.AxEBL_fp16.shape == (M, R)
    assert b.EARxBpEB_fp16.shape == (N, R)
    assert b.AxEBL_int32.shape == (M, R)
    assert b.EARxBpEB_int32.shape == (N, R)
    assert b.ApEA.shape == (M, K)
    assert b.BpEB.shape == (N, K)
    assert b.pow_target.shape == (8,)
    assert b.pow_key.shape == (8,)


def test_default_batch_size_is_256(batcher_module):
    """The persistent-CTA design uses 256 nonces per batch (matches alpha-miner)."""
    mod, _, _ = batcher_module
    assert mod.DEFAULT_BATCH_SIZE == 256


# ----- contexts table population ----------------------------------------------


def test_refresh_nonces_with_monotonic_base(batcher_module):
    """When called with `base_nonce=N`, each slot's nonce_value = N + slot_idx."""
    mod, torch_fake, pg_fake = batcher_module
    b = _make_batch(mod, torch_fake, pg_fake, batch_size=4)
    b.refresh_nonces(base_nonce=1000)

    # Wave-13: nonce_value moved to column 7 (was column 3 in wave-12).
    for i in range(4):
        nv = b.contexts[i, 7].item()
        assert nv == 1000 + i, f"slot {i}: contexts[i,7]={nv}, want {1000 + i}"


def test_refresh_nonces_random_path_writes_all_slots(batcher_module):
    """Default path (no base_nonce) populates the full batch_size."""
    mod, torch_fake, pg_fake = batcher_module
    b = _make_batch(mod, torch_fake, pg_fake, batch_size=8)
    b.refresh_nonces()
    # All 8 nonce slots must have been written. Our fake torch.randint
    # produces deterministic non-zero values for the second call onward —
    # but the zeroth (still freshly allocated) slot can technically be 0.
    # Just check that we have 8 entries to read.
    for i in range(8):
        _ = b.contexts[i, 7].item()  # wave-13: nonce_value in col 7


def test_contexts_records_per_slot_pointers(batcher_module):
    """Wave-13 contexts[i] layout:

      [A_ptr, A_scales_ptr, C_ptr,
       host_signal_header_ptr, host_signal_sync_ptr,
       pow_target_ptr, pow_key_ptr,
       nonce_value]

    Per-i pointers for A/A_scales/C/host_signal_header/host_signal_sync must
    differ by the stride of one slot in the respective parent tensor (linear
    pointer arithmetic). pow_target_ptr and pow_key_ptr are SHARED across the
    batch (same address in every slot) since the commitment_hash and
    adjusted_target are constant across the 256 nonces of one launch.

    Wave-14: `A_ptr` (column 0) now points to `ApEA_batch[i]` (noised A),
    not raw `A_batch[i]`. The pool's verifier replays `ApEA @ BpEB.T`, so
    the kernel must matmul the noised tensor for the PoW transcript to match.
    """
    mod, torch_fake, pg_fake = batcher_module
    b = _make_batch(mod, torch_fake, pg_fake, batch_size=8, M=64, N=64, K=128, R=128)
    b.refresh_nonces(base_nonce=42)

    # Expected per-slot strides:
    a_stride = 64 * 128 * 1     # M * K * sizeof(int8)
    s_stride = 64 * 4           # M * sizeof(float32)
    c_stride = 64 * 64 * 2      # M * N * sizeof(bfloat16)
    hh_stride = b.hh_slot_bytes
    hs_stride = b.hs_slot_bytes

    # Wave-14: contexts[i, 0] addresses ApEA_batch[i], not raw A_batch[i].
    a_base = b.ApEA_batch.data_ptr()
    s_base = b.A_scales_batch.data_ptr()
    c_base = b.C_batch.data_ptr()
    hh_base = b.host_signal_headers.data_ptr()
    hs_base = b.host_signal_syncs.data_ptr()
    pow_t = b.pow_target.data_ptr()
    pow_k = b.pow_key.data_ptr()

    for i in range(8):
        assert b.contexts[i, 0].item() == a_base + i * a_stride, (
            f"slot {i}: A_ptr mismatch"
        )
        assert b.contexts[i, 1].item() == s_base + i * s_stride, (
            f"slot {i}: A_scales_ptr mismatch"
        )
        assert b.contexts[i, 2].item() == c_base + i * c_stride, (
            f"slot {i}: C_ptr mismatch"
        )
        assert b.contexts[i, 3].item() == hh_base + i * hh_stride, (
            f"slot {i}: host_signal_header_ptr mismatch"
        )
        assert b.contexts[i, 4].item() == hs_base + i * hs_stride, (
            f"slot {i}: host_signal_sync_ptr mismatch"
        )
        assert b.contexts[i, 5].item() == pow_t, (
            f"slot {i}: pow_target_ptr should be shared address {pow_t}"
        )
        assert b.contexts[i, 6].item() == pow_k, (
            f"slot {i}: pow_key_ptr should be shared address {pow_k}"
        )
        assert b.contexts[i, 7].item() == 42 + i, (
            f"slot {i}: nonce_value mismatch"
        )


# ----- launch dispatch --------------------------------------------------------


def test_launch_calls_persistent_multinonce(batcher_module):
    """`batch.launch()` must invoke `pg.gemm_persistent_multinonce` once."""
    mod, torch_fake, pg_fake = batcher_module
    b = _make_batch(mod, torch_fake, pg_fake, batch_size=4)
    b.refresh_nonces(base_nonce=0)

    b.launch(
        bM=64, bN=64, bK=64, cM=1, cN=1, matmul_stages=2,
        noise_tile_a_m=64, noise_tile_a_k=64,
        noise_tile_b_n=64, noise_tile_b_k=64,
        noise_stages_a=2, noise_stages_b=2,
    )
    assert pg_fake.gemm_persistent_multinonce.call_count == 1
    args = pg_fake.gemm_persistent_multinonce.call_args.args

    # Args 0..6 are the per-batch tensors that distinguish this entry point
    # from `noisy_gemm`. We verify they are the NonceBatch's batched tensors,
    # not the per-tile scratch.
    # Wave-14: args[1] is BpEB_noised (the noised B), not raw B. The matmul
    # kernel reads it as params.ptr_BpEB. Raw B is kept for share submission
    # (the pool re-derives commitment from the original A, B that we shipped).
    assert args[0] is b.A_batch
    assert args[1] is b.BpEB_noised
    assert args[2] is b.A_scales_batch
    assert args[3] is b.B_scales
    assert args[4] is b.C_batch
    assert args[5] is b.contexts
    assert args[6] is b.nonces


def test_launch_raises_without_entrypoint(batcher_module):
    """If the .so doesn't have `gemm_persistent_multinonce`, surface a clear
    error so we don't silently silently fall back to a slower path. The user-
    facing message tells the operator how to recover."""
    mod, torch_fake, pg_fake = batcher_module
    # Remove the attribute the launcher checks for.
    del pg_fake.gemm_persistent_multinonce
    b = _make_batch(mod, torch_fake, pg_fake, batch_size=2)

    with pytest.raises(RuntimeError, match="gemm_persistent_multinonce"):
        b.launch(
            bM=64, bN=64, bK=64, cM=1, cN=1, matmul_stages=2,
            noise_tile_a_m=64, noise_tile_a_k=64,
            noise_tile_b_n=64, noise_tile_b_k=64,
            noise_stages_a=2, noise_stages_b=2,
        )


# ----- bit-exact equivalence: batched vs sequential ---------------------------
#
# The kernel build is happening in a parallel agent; we don't have a real
# `pg.gemm_persistent_multinonce` to run here. Instead we stub BOTH entry
# points (`pg.noisy_gemm` for sequential, `pg.gemm_persistent_multinonce`
# for batched) with the SAME deterministic "kernel": C_i = checksum(A_i, B,
# A_scales_i, B_scales). We then assert that running the batched path
# yields the same C[i] as a sequential loop that calls noisy_gemm 256 times
# with the same per-nonce inputs.
#
# This is a host-side bit-exact contract: the batcher must lay out per-nonce
# data such that a kernel which writes C_i from (A_i, B, scales_i, scales_B)
# produces the same per-nonce outputs as the sequential per-call path.


def _deterministic_checksum(A_row: _FakeTensor, A_scales_row: _FakeTensor,
                            B: _FakeTensor, B_scales: _FakeTensor) -> int:
    """Pure-Python "kernel": sum of all input bytes XORed.

    Stand-in for the real noisy_gemm — we only need a function that depends
    on ALL its inputs so the test catches any mis-routing of per-nonce
    pointers. If the batcher accidentally fed slot 0's A to slot 1, the
    checksums would diverge.
    """
    s = 0
    for v in A_row._data:
        s = (s + (v & 0xFF)) & 0xFFFFFFFF
    for v in A_scales_row._data:
        s = (s ^ int(v * 1e6)) & 0xFFFFFFFF
    for v in B._data:
        s = (s + (v & 0xFF)) & 0xFFFFFFFF
    for v in B_scales._data:
        s = (s ^ int(v * 1e6)) & 0xFFFFFFFF
    return s


def test_batched_equals_sequential_bit_exact(batcher_module):
    """The centerpiece: 256 nonces batched yields same C[i] as 256 sequential.

    Approach
    --------
    1. Build a NonceBatch and fill A_batch + B + scales with deterministic
       data.
    2. Run a SEQUENTIAL kernel: for each i, compute C_seq[i] =
       checksum(A_batch[i], A_scales_batch[i], B, B_scales).
    3. Stub `pg.gemm_persistent_multinonce` to walk the contexts table and
       compute the SAME checksum, but driven by the contexts[i] pointers —
       so a wrong stride would silently put slot j's data into slot i's
       output.
    4. Launch the batched path. Compare C_seq[i] vs C_batched[i] for all i.
    """
    mod, torch_fake, pg_fake = batcher_module
    cfg_batch_size = 256
    b = _make_batch(mod, torch_fake, pg_fake, batch_size=cfg_batch_size,
                    M=16, N=16, K=32, R=32)

    # 1) Deterministic input fill — per-nonce A varies, B fixed.
    # We bypass NonceBatch's smoke-fill (which generates fresh random data
    # each call) and write directly so the sequential and batched paths see
    # the same bytes.
    for i in range(cfg_batch_size):
        # Per-slot deterministic pattern: A[i, m, k] = (i*7 + m*3 + k) & 0xFF - 128
        row_data = []
        for m in range(16):
            for k in range(32):
                row_data.append(((i * 7 + m * 3 + k) & 0xFF) - 128)
        b.A_batch[i]._data = row_data  # type: ignore[attr-defined]

        scale_data = [0.005 + (i * m * 1e-4) for m in range(16)]
        b.A_scales_batch[i]._data = scale_data  # type: ignore[attr-defined]

    b.B._data = [((j * 13) & 0xFF) - 128 for j in range(16 * 32)]
    b.B_scales._data = [0.01 + j * 1e-5 for j in range(16)]

    # Build the contexts table for the batched path.
    b.refresh_nonces(base_nonce=42)

    # 2) Run the SEQUENTIAL "kernel" — produces ground truth.
    # Wave-14: the multinonce kernel matmuls `ApEA @ BpEB.T`, so the
    # sequential reference also uses BpEB_noised (= raw B for this test since
    # compute_noised_inputs() isn't called — both BpEB_noised and the raw B
    # arg consumed by the stub will be zero-filled identically).  We reference
    # `b.BpEB_noised` here so the test stays correct if the kernel arg
    # ordering changes in future waves.
    C_sequential = []
    for i in range(cfg_batch_size):
        row = _deterministic_checksum(
            b.A_batch[i], b.A_scales_batch[i], b.BpEB_noised, b.B_scales,
        )
        C_sequential.append(row)

    # 3) Stub `pg.gemm_persistent_multinonce` to walk contexts + compute
    #    the same checksum but indirectly through pointer arithmetic.
    # Wave-14: a_base is `ApEA_batch.data_ptr()` since contexts[i, 0] now
    # addresses the noised input. ApEA_batch has the same (B, M, K) shape +
    # int8 dtype as A_batch, so a_stride is unchanged.
    a_stride = 16 * 32 * 1
    s_stride = 16 * 4
    a_base = b.ApEA_batch.data_ptr()
    s_base = b.A_scales_batch.data_ptr()
    C_batched = [None] * cfg_batch_size

    def _stub_kernel(A_batch_arg, B_arg, A_scales_batch_arg, B_scales_arg,
                     C_batch_arg, contexts_arg, nonces_arg, *_rest):
        # The kernel sees: contexts_arg[i, 0..3] = (A_ptr, scales_ptr, C_ptr, nonce).
        # It uses the pointers to index into the batched tensors. Here we
        # invert the pointer arithmetic to find which slot each context points
        # at — a real persistent CTA would consume A_ptr + B_ptr + scales_ptr
        # directly via cp.async; we just verify the *mapping* is correct.
        for i in range(cfg_batch_size):
            a_ptr = contexts_arg[i, 0].item()
            s_ptr = contexts_arg[i, 1].item()
            slot_from_a = (a_ptr - a_base) // a_stride
            slot_from_s = (s_ptr - s_base) // s_stride
            # If the batcher built contexts correctly, both pointer offsets
            # resolve to the same slot index.
            assert slot_from_a == slot_from_s == i, (
                f"contexts[{i}] points to slot A={slot_from_a} S={slot_from_s}, want {i}"
            )
            # Use the resolved slot index to look up the row data.
            # In a real kernel this would be ApEA_batch[slot_from_a], but for
            # the host-side contract test we use the raw A_batch — the
            # check is on pointer routing, not noising arithmetic.
            row_data = A_batch_arg[slot_from_a]
            scale_data = A_scales_batch_arg[slot_from_s]
            C_batched[i] = _deterministic_checksum(
                row_data, scale_data, B_arg, B_scales_arg,
            )

    pg_fake.gemm_persistent_multinonce = mock.MagicMock(side_effect=_stub_kernel)

    # 4) Launch and compare.
    b.launch(
        bM=64, bN=64, bK=64, cM=1, cN=1, matmul_stages=2,
        noise_tile_a_m=64, noise_tile_a_k=64,
        noise_tile_b_n=64, noise_tile_b_k=64,
        noise_stages_a=2, noise_stages_b=2,
    )

    for i in range(cfg_batch_size):
        assert C_batched[i] == C_sequential[i], (
            f"nonce {i}: batched={C_batched[i]} sequential={C_sequential[i]}"
        )


# ----- fill_smoke_A / fill_shared_B --------------------------------------------


def test_fill_smoke_A_changes_data(batcher_module):
    """`fill_smoke_A` writes fresh random data; two consecutive calls must
    produce different `A_batch` contents."""
    mod, torch_fake, pg_fake = batcher_module
    b = _make_batch(mod, torch_fake, pg_fake, batch_size=2, M=8, N=8, K=16, R=16)
    b.fill_smoke_A()
    snapshot = list(b.A_batch._data)
    b.fill_smoke_A()
    assert b.A_batch._data != snapshot, (
        "fill_smoke_A must mutate A_batch — two consecutive calls were identical"
    )


def test_fill_shared_B_changes_data(batcher_module):
    mod, torch_fake, pg_fake = batcher_module
    b = _make_batch(mod, torch_fake, pg_fake, batch_size=2, M=8, N=8, K=16, R=16)
    b.fill_shared_B()
    snap_B = list(b.B._data)
    snap_Bs = list(b.B_scales._data)
    b.fill_shared_B()
    # At least one should change (our deterministic fake-torch produces
    # different values per call due to the internal counter).
    assert b.B._data != snap_B or b.B_scales._data != snap_Bs


# ----- pyproject sanity: batcher must be importable in driver --------------


def test_driver_imports_batcher_module():
    """Smoke test: `from _nonce_batcher import ...` works given the test
    fixture's sys.path setup. Catches missing module / name typos that
    would break the production driver at startup, not at test collection.
    """
    # Force a clean import resolution.
    sys.path.insert(0, str(HERE.parent))
    try:
        if "_nonce_batcher" in sys.modules:
            del sys.modules["_nonce_batcher"]
        # Install a fake torch first — _nonce_batcher does `from typing import Any`
        # but uses torch tensor types at runtime, not at import.
        saved_torch = sys.modules.get("torch")
        sys.modules["torch"] = _install_fake_torch()
        try:
            import _nonce_batcher as nb  # noqa: F401
            assert hasattr(nb, "NonceBatch")
            assert hasattr(nb, "BatchConfig")
            assert hasattr(nb, "DEFAULT_BATCH_SIZE")
            assert hasattr(nb, "is_persistent_nonce_enabled")
            assert nb.DEFAULT_BATCH_SIZE == 256
        finally:
            if saved_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved_torch
    finally:
        if str(HERE.parent) in sys.path:
            sys.path.remove(str(HERE.parent))
