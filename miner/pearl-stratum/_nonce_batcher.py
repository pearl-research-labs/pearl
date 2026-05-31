"""Host-side multi-nonce batching primitives for the sm_89 persistent-CTA path.

Motivation
----------
The current `_miner_driver_sm89_r128.py` does ONE nonce per `pg.noisy_gemm()`
call. Alpha-miner amortizes per-attempt overhead across 256 nonces in a single
persistent-CTA kernel (B operand resident, A varies per nonce; see
`project_pearl_miner_dominate_2026_05_17.md` section "What alpha-miner's sm_89
GEMM is actually doing"). Closing this is ~3-5× of the 13× gap.

This module owns the host-side staging so the new kernel entry point
`pg.gemm_persistent_multinonce` (being built in parallel) can be called once
per attempt-batch instead of once per nonce.

Layout
------
For a batch of N nonces (default 256):

  Per-nonce, in one contiguous tensor each:
    A_batch        (N, M, K) int8
    A_scales_batch (N, M)    float32
    C_batch        (N, M, N) bfloat16
    nonces         (N,)      uint64

  Shared across the batch (B is the persistent operand):
    B              (N_cols, K) int8
    B_scales       (N_cols,)   float32

  Device-side `contexts` array — what the persistent CTA scheduler reads on
  each iteration to pick up the next nonce's pointers without going back to
  the host. Wave-13 layout (8 int64 columns per slot):
    contexts       (N, 8)      int64
      contexts[i] = [
        A_ptr_i, A_scales_ptr_i, C_ptr_i,
        host_signal_header_ptr_i,    # pinned host mem, per-nonce slot
        host_signal_sync_ptr_i,      # device mem, per-nonce slot
        pow_target_ptr,              # shared across batch (commitment is shared)
        pow_key_ptr,                 # shared across batch
        nonce_i_as_int64,            # kernel-side bookkeeping
      ]

  Per-tile scratch (NOT per-nonce — reused across the batch by the kernel):
    EAL, EBR, EAL_fp16, EBR_fp16,
    EAR_R_major, EBL_R_major, EAR_K_major, EBL_K_major,
    AxEBL_fp16, EARxBpEB_fp16, AxEBL_int32, EARxBpEB_int32,
    ApEA, BpEB,
    pow_target, pow_key,            # shared across the batch (one buffer each)
    host_signal_headers, host_signal_syncs  # PER-NONCE arrays (256 slots each)

  Wave-14 per-batch noised buffers (Python-computed):
    ApEA_batch     (B, M, K) int8  — A_batch[i] + E_A (per-nonce)
    BpEB_noised    (N, K)    int8  — B + E_B (shared across the batch)
  E_A = EAL @ EAR_R_major.T and E_B = EBR @ EBL_R_major.T are constant per
  mining_job (commitment-derived) so they're computed once and broadcast.
  The contexts table's `ptr_A` slot is repointed to ApEA_batch[i] so the
  multinonce kernel sees noised inputs and the PoW transcript matches what
  the pool's verifier will replay.

All buffers are allocated once at construction and reused across attempts.
The `refresh_nonces()` method draws fresh nonces + writes them and re-points
the contexts array; per-nonce A/A_scales are filled by the caller before the
kernel launch.

Env-gated activation
--------------------
`PEARL_SM89_PERSISTENT_NONCE=1` activates the batched path. When unset (or 0),
the driver uses the existing per-nonce loop unchanged. This lets us A/B the
two paths against the same pool without rebuilding the .so.

`PEARL_ONE_A_PER_ATTEMPT=1` (wave-16) further changes the batched path so
that ONE A is generated per attempt and all 256 nonce-slot pointers in the
`contexts` table point at slot 0. This matches the protocol-authoritative
Rust reference (one A per attempt; the 256 "nonces" become 256 distinct
tile coords the kernel scheduler walks over a single A). Drops a 155 ms
torch.randint per attempt at B=256 production geometry, and also fixes a
latent share-validity bug where the driver hashed A_batch[0] for the
commitment but submitted per-nonce A_batch[i] (i>0), causing the pool's
commitment re-derivation to silently mismatch and reject the share.

NOTE: bit-exact equivalence vs the sequential path is asserted by
`tests/test_nonce_batcher.py` — see that file for the contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


# Default batch size — alpha-miner uses 256 (one block of 256 per persistent CTA).
DEFAULT_BATCH_SIZE = 256

# Env var controlling whether the driver uses the batched path. Defaults off
# so the production R=64 path is untouched until the kernel agent has the
# matching entry point built.
ENV_PERSISTENT_NONCE = "PEARL_SM89_PERSISTENT_NONCE"

# Wave-16: one A per attempt mode. When set, the batched path stops
# generating per-nonce A matrices (saves ~155 ms torch.randint at B=256
# production geometry) and instead generates ONE A per attempt, with all
# 256 contexts pointing at slot 0. This matches the protocol-authoritative
# Rust reference (`zk-pow/src/ffi/mine.rs:33-39`): one A per attempt, the
# 256 "nonces" become 256 tile-pattern offsets the persistent-CTA
# scheduler walks. See `pearl-investigation/wave16-domination/04_alpha_prf.md`
# for the protocol analysis. This ALSO fixes a latent share-validity bug
# where commitment_hash was derived from A_batch[0] but per-nonce A_batch[i]
# (i>0) was submitted -- the pool's re-derivation would mismatch.
ENV_ONE_A_PER_ATTEMPT = "PEARL_ONE_A_PER_ATTEMPT"


def is_persistent_nonce_enabled() -> bool:
    """True when caller has explicitly opted into the batched path.

    Accepts: "1", "true", "yes", "on" (case-insensitive). Anything else
    including unset → False, so this is safe-by-default.
    """
    val = os.environ.get(ENV_PERSISTENT_NONCE, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def is_one_A_per_attempt_enabled() -> bool:
    """True when caller has opted into one-A-per-attempt mode.

    See ENV_ONE_A_PER_ATTEMPT docstring. Safe-by-default off so the legacy
    per-nonce path is untouched unless the env explicitly opts in.
    """
    val = os.environ.get(ENV_ONE_A_PER_ATTEMPT, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


@dataclass
class BatchConfig:
    """Geometry shared by all tensors in the batch.

    M, N, K, R follow the existing sm_89 driver chunk parameters. batch_size
    is the persistent-CTA nonce count. Stored as a dataclass to keep call
    sites readable.
    """
    M: int
    N: int
    K: int
    R: int
    batch_size: int = DEFAULT_BATCH_SIZE


class NonceBatch:
    """Container for one persistent-CTA attempt batch.

    The fields below are the device tensors the kernel reads/writes. The
    `contexts` tensor is the array of per-nonce pointer records the kernel
    scheduler iterates over.

    Lifecycle
    ---------
    1. Construct once at driver startup with `NonceBatch(torch_mod, pg_mod, cfg, device)`.
    2. Per attempt-batch:
       a. Call `refresh_nonces()` to draw a fresh `batch_size` nonces and
          repoint `contexts` (the underlying buffers don't reallocate).
       b. Fill `A_batch[i]` / `A_scales_batch[i]` for each i — by hand
          (random fill, mirrors current driver) or via a future fill kernel
          driven by the nonce seed.
       c. Call `launch(pg)` to invoke `pg.gemm_persistent_multinonce(...)`.
       d. Read `C_batch[i]` per nonce.

    The PoW target / key tensors are device-side state the caller may rewrite
    between batches (e.g. when the pool sends a new target). We expose them
    as attributes rather than baking them into refresh_nonces() — the driver
    already does that.

    All `torch` access goes through `_torch` so the unit tests can swap in
    a fake module without importing real torch.
    """

    def __init__(
        self,
        torch_mod: Any,
        pg_mod: Any,
        cfg: BatchConfig,
        device: Any,
    ) -> None:
        self._torch = torch_mod
        self._pg = pg_mod
        self.cfg = cfg
        self.device = device

        torch = torch_mod
        M, N, K, R, B = cfg.M, cfg.N, cfg.K, cfg.R, cfg.batch_size

        # -- per-nonce buffers (one per attempt, batched) ----------------------
        # A_batch[i] is the int8 (M, K) operand for nonce i; the persistent CTA
        # kernel pulls slice i out of this on iteration i.
        self.A_batch = torch.zeros(B, M, K, dtype=torch.int8, device=device)
        self.A_scales_batch = torch.zeros(B, M, dtype=torch.float32, device=device)
        self.C_batch = torch.zeros(B, M, N, dtype=torch.bfloat16, device=device)
        # uint64 nonce values, one per batch slot.
        self.nonces = torch.zeros(B, dtype=torch.int64, device=device)

        # -- shared (persistent) operand ---------------------------------------
        # B operand and its scales are reused across the whole batch — that's
        # the whole point of "persistent CTA": B sits resident in smem/regs
        # while A varies per nonce.
        self.B = torch.zeros(N, K, dtype=torch.int8, device=device)
        self.B_scales = torch.zeros(N, dtype=torch.float32, device=device)

        # -- Wave-14 noised buffers (Python-computed before each launch) ------
        # ApEA_batch[i] = (A_batch[i].int() + E_A_int32).to(int8)  modular wrap
        # BpEB_noised   = (B.int() + E_B_int32).to(int8)           modular wrap
        # where:
        #   E_A_int32 = EAL.int() @ EAR_R_major.t().int()    (M, K) int32
        #   E_B_int32 = EBR.int() @ EBL_R_major.t().int()    (N, K) int32
        # Both E_A and E_B depend only on EAL/EAR/EBR/EBL (commitment-derived),
        # so they're constant across the 256 nonces of one launch and computed
        # once per submit. The persistent-CTA kernel reads ApEA_batch[i] via
        # contexts[i, 0] (per-nonce override) and BpEB_noised via the standard
        # B argument (shared across the batch). PoW transcript matches the
        # pool's verifier replay because both operands are now noised in the
        # same way as wave-11's `pg.noisy_gemm` would have produced.
        self.ApEA_batch = torch.zeros(B, M, K, dtype=torch.int8, device=device)
        self.BpEB_noised = torch.zeros(N, K, dtype=torch.int8, device=device)

        # -- contexts: pointer record per nonce, read by the scheduler ----------
        # int64 columns: see module docstring for wave-13 layout. Length=8.
        self.contexts = torch.zeros(B, 8, dtype=torch.int64, device=device)

        # -- per-tile scratch (reused across the batch by the kernel) ----------
        # These match the existing R=128 driver allocations; the persistent
        # CTA kernel re-uses them in-place across all nonces.
        self.EAL            = torch.zeros(M, R, dtype=torch.int8,    device=device)
        self.EBR            = torch.zeros(N, R, dtype=torch.int8,    device=device)
        self.EAL_fp16       = torch.zeros(M, R, dtype=torch.float16, device=device)
        self.EBR_fp16       = torch.zeros(N, R, dtype=torch.float16, device=device)
        self.EAR_R_major    = torch.zeros(K, R, dtype=torch.int8,    device=device)
        self.EBL_R_major    = torch.zeros(K, R, dtype=torch.int8,    device=device)
        self.EAR_K_major    = torch.zeros(R, K, dtype=torch.int8,    device=device)
        self.EBL_K_major    = torch.zeros(R, K, dtype=torch.int8,    device=device)
        self.AxEBL_fp16     = torch.zeros(M, R, dtype=torch.float16, device=device)
        self.EARxBpEB_fp16  = torch.zeros(N, R, dtype=torch.float16, device=device)
        self.AxEBL_int32    = torch.zeros(M, R, dtype=torch.int32,   device=device)
        self.EARxBpEB_int32 = torch.zeros(N, R, dtype=torch.int32,   device=device)
        self.ApEA           = torch.zeros(M, K, dtype=torch.int8,    device=device)
        self.BpEB           = torch.zeros(N, K, dtype=torch.int8,    device=device)

        # -- PoW target / key (SHARED across the batch — commitment_hash and
        # adjusted_target depend only on (mining_job, matmul_config), which are
        # constant across the 256 nonces in one launch). ------------------------
        self.pow_target = torch.zeros(8, dtype=torch.uint32, device=device)
        self.pow_key = torch.zeros(8, dtype=torch.uint32, device=device)

        # -- Per-nonce HostSignalHeader / HostSignalSync slots ----------------
        # Wave-13: each nonce needs its own (header, sync) pair so 256 PoW
        # hits in one launch don't all serialize on a single global_lock CAS.
        # We allocate two contiguous arrays of size (B * slot_bytes) and let
        # the contexts table point at slot i's base address. The kernel's
        # write_host_signal_header dereferences each pointer field-by-field
        # so the inner stride is the size of HostSignalHeader (640) /
        # HostSignalSync (8) respectively.
        self.hh_slot_bytes = pg_mod.get_host_signal_header_size()
        self.hs_slot_bytes = pg_mod.get_host_signal_sync_size()
        self.host_signal_headers = torch.zeros(
            B * self.hh_slot_bytes, dtype=torch.int8, pin_memory=True
        )
        self.host_signal_syncs = torch.zeros(
            B * self.hs_slot_bytes, dtype=torch.int8, device=device
        )

        # Back-compat aliases (NonceBatch.launch references these on the
        # signature side; they're slot-0 views via narrow). When the tensor
        # type supports slicing (real torch) we use it; otherwise (test fakes)
        # we just point at the full buffer — slot 0 is its first bytes.
        try:
            self.host_signal_header = self.host_signal_headers[: self.hh_slot_bytes]
            self.host_signal_sync = self.host_signal_syncs[: self.hs_slot_bytes]
        except (NotImplementedError, TypeError):
            self.host_signal_header = self.host_signal_headers
            self.host_signal_sync = self.host_signal_syncs

    # ------------------------------------------------------------------
    # Nonce / context plumbing
    # ------------------------------------------------------------------

    def refresh_nonces(self, base_nonce: Optional[int] = None) -> None:
        """Draw fresh nonces and repoint the contexts array.

        If `base_nonce` is None we draw uniformly random int64s — fine for
        the kernel-side smoke path. In production the driver passes a
        monotonically incrementing base; the kernel only cares that nonces
        are distinct within the batch (Pearl share-uniqueness).

        Does NOT touch A_batch / A_scales_batch / C_batch — those are the
        caller's responsibility per attempt (random fill in the smoke path;
        nonce-seeded noise gen in the production path once it's wired).
        """
        torch = self._torch
        B = self.cfg.batch_size
        if base_nonce is None:
            # uniform draw across int64 space; the bias toward 0/1 from torch
            # default `low=0` is irrelevant for a smoke buffer
            self.nonces.copy_(
                torch.randint(0, 2**62, (B,), dtype=torch.int64, device=self.device)
            )
        else:
            # Monotonic stripe — easier to reason about in tests/replay
            self.nonces.copy_(
                torch.arange(base_nonce, base_nonce + B, dtype=torch.int64, device=self.device)
            )

        # Recompute the contexts table. Pointer offsets are stable for the
        # life of the batch (we re-use the same underlying tensors), so this
        # is technically only required after construction — but redoing it
        # is cheap and lets us defensively re-sync if a subclass swaps out
        # one of the per-nonce tensors.
        self._rebuild_contexts()

    def _rebuild_contexts(self) -> None:
        """Compute per-nonce pointers into the wave-13 (B, 8) contexts table.

        Records, for each batch index i:
          [A_ptr, A_scales_ptr, C_ptr, host_signal_header_ptr,
           host_signal_sync_ptr, pow_target_ptr, pow_key_ptr, nonce_value]

        Wave-14: A_ptr points to `ApEA_batch[i]` (noised A), not raw `A_batch[i]`.
        The kernel matmul is `ApEA @ BpEB.T` -- the pool's verifier replays the
        same expression, so transcript bytes match and submitted shares credit.
        `A_batch` is still kept so the share submission can ship the raw A_i
        the pool needs to re-derive commitment + replay (OpenedBlockInfo).

        Wave-16 (PEARL_ONE_A_PER_ATTEMPT): when enabled, A_ptr and A_scales_ptr
        in EVERY slot point at slot 0 -- all 256 nonces share one A. This
        matches the protocol-authoritative Rust ref (one A per attempt) and
        eliminates per-nonce A regen cost.

        host_signal_header and host_signal_sync pointers are per-nonce slot
        addresses into the contiguous arrays allocated by __init__. The
        pow_target / pow_key pointers are SHARED (one address each, written
        identically into all 256 slots) since the commitment_hash and the
        adjusted PoW target are constant across the 256 nonces of one launch.
        """
        torch = self._torch
        B = self.cfg.batch_size

        # Wave-14: contexts[i, 0] points to ApEA_batch[i] (noised), not A_batch[i].
        # The kernel reads this as `ptr_A` and matmuls it against `BpEB_noised`.
        a_base = int(self.ApEA_batch.data_ptr())
        a_stride = self.ApEA_batch[0].numel() * self.ApEA_batch.element_size()
        s_base = int(self.A_scales_batch.data_ptr())
        s_stride = self.A_scales_batch[0].numel() * self.A_scales_batch.element_size()
        c_base = int(self.C_batch.data_ptr())
        c_stride = self.C_batch[0].numel() * self.C_batch.element_size()

        hh_base = int(self.host_signal_headers.data_ptr())
        hs_base = int(self.host_signal_syncs.data_ptr())
        hh_stride = self.hh_slot_bytes
        hs_stride = self.hs_slot_bytes

        pow_t_ptr = int(self.pow_target.data_ptr())
        pow_k_ptr = int(self.pow_key.data_ptr())

        # Wave-16: stride=0 for A pointers when one-A-per-attempt mode is on,
        # so every slot resolves to slot 0. C stride is left per-nonce because
        # the miner doesn't read C (skip_denoising=True in the launch) but each
        # kernel CTA writes its own per-tile result and overlapping slots would
        # produce write races.
        one_A = is_one_A_per_attempt_enabled()
        a_eff_stride = 0 if one_A else a_stride
        s_eff_stride = 0 if one_A else s_stride

        # 256 * 8 * 8 B = 16 KB host buffer — one H2D copy per refresh.
        ctx_host = torch.zeros(B, 8, dtype=torch.int64)
        for i in range(B):
            ctx_host[i, 0] = a_base + i * a_eff_stride
            ctx_host[i, 1] = s_base + i * s_eff_stride
            ctx_host[i, 2] = c_base + i * c_stride
            ctx_host[i, 3] = hh_base + i * hh_stride
            ctx_host[i, 4] = hs_base + i * hs_stride
            ctx_host[i, 5] = pow_t_ptr
            ctx_host[i, 6] = pow_k_ptr
            ctx_host[i, 7] = int(self.nonces[i].item())
        self.contexts.copy_(ctx_host.to(self.device))

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------

    def compute_noised_inputs(self) -> None:
        """Wave-14: compute ApEA_batch and BpEB_noised from the noise factors.

        Mirrors `pearl_noisingA_kernel_sm89.h` / `pearl_noisingB_kernel_sm89.h`
        but lifts the shared `E_A = EAL @ EAR^T` and `E_B = EBR @ EBL^T` matmuls
        out of the inner per-nonce loop (both are commitment-constant across
        the 256 nonces in one launch). The kernels are bit-exact equivalents:

          ApEA[m, k] = int8_wrap(A[m, k] + int8_wrap(E_A_int32[m, k]))
                    == int8_wrap(A[m, k] + E_A_int32[m, k])   (mod-256 assoc.)
          BpEB[n, k] = int8_wrap(B[n, k] + E_B_int32[n, k])

        Implementation:
          1. E_A_int8 (M, K) and E_B_int8 (N, K) via float32 matmul + int8
             cast (uses tensor cores, ~1 ms each at production shape).
          2. ApEA_batch = A_batch + E_A_int8.unsqueeze(0)   (broadcasted)
             BpEB_noised = B + E_B_int8
             torch int8 + int8 wraps mod-256 by default (verified empirically
             on CUDA: 100 + 50 = -106), matching the kernel's
             `int8_t(int(a) + int(b))` semantics.

        Cost on production geometry (M=N=2048, K=4096, R=128, B=256):
          E_A matmul:   ~0.4 ms (2.15 GMACs via tensor cores).
          E_B matmul:   ~0.2 ms (1.07 GMACs).
          ApEA add:     ~2-3 ms (2 GB write).
          BpEB add:     ~0.1 ms (8 MB write).
          Total: ~3 ms vs the ~12 ms kernel launch.
        """
        torch = self._torch

        # int8 matmul via float intermediate. Exact because |EAL @ EAR.T| max
        # = 128 * 128 * 128 = 2,097,152, well below float32 mantissa precision
        # (2^24 = 16,777,216). float32 matmul uses tensor cores.
        EAL_f = self.EAL.to(torch.float32)
        EBR_f = self.EBR.to(torch.float32)
        EAR_R_f = self.EAR_R_major.to(torch.float32)    # (K, R)
        EBL_R_f = self.EBL_R_major.to(torch.float32)    # (K, R)

        # (M, R) @ (R, K) = (M, K) int8 (modular truncation of int32 result)
        E_A_int8 = (EAL_f @ EAR_R_f.t()).to(torch.int32).to(torch.int8)
        # (N, R) @ (R, K) = (N, K) int8
        E_B_int8 = (EBR_f @ EBL_R_f.t()).to(torch.int32).to(torch.int8)

        # int8 + int8 in torch wraps mod-256 (matches kernel arithmetic).
        # BpEB_noised: (N, K) = (N, K) + (N, K)
        torch.add(self.B, E_B_int8, out=self.BpEB_noised)

        if is_one_A_per_attempt_enabled():
            # Wave-16: only slot 0 is consumed by the kernel (all 256 contexts
            # point at ApEA_batch[0]). Skip the (B, M, K) broadcast — saves
            # ~2-3 ms / 2 GB write on production geometry.
            torch.add(self.A_batch[0], E_A_int8, out=self.ApEA_batch[0])
        else:
            # Legacy per-nonce path: ApEA_batch: (B, M, K) = (B, M, K) + (1, M, K)
            # Use torch.add with out= to avoid allocating a 2 GB intermediate.
            torch.add(self.A_batch, E_A_int8.unsqueeze(0), out=self.ApEA_batch)

    def launch(
        self,
        *,
        bM: int, bN: int, bK: int, cM: int, cN: int, matmul_stages: int,
        noise_tile_a_m: int, noise_tile_a_k: int,
        noise_tile_b_n: int, noise_tile_b_k: int,
        noise_stages_a: int, noise_stages_b: int,
        skip_reduction: bool = False,
        skip_denoising: bool = False,
    ) -> None:
        """Invoke the persistent-CTA multi-nonce kernel entry point.

        Wave-13: defaults to skip_reduction=False so the kernel runs
        `check_pow_target` + `write_host_signal_header` per (m_block, n_block,
        nonce_idx) tile and emits hits into the per-nonce host_signal_header
        slot the contexts table assigns. Set skip_reduction=True for a
        noiseless benchmark.

        Wave-14: passes `BpEB_noised` (noised B) as the B argument and uses
        `ApEA_batch` (via contexts[i, 0]) for the per-nonce A. Caller must
        invoke `compute_noised_inputs()` after refreshing A/B/noise factors
        and before this method.
        """
        pg = self._pg
        if not hasattr(pg, "gemm_persistent_multinonce"):
            raise RuntimeError(
                "pg.gemm_persistent_multinonce is not present in this .so build. "
                "Either disable the batched path (unset "
                f"{ENV_PERSISTENT_NONCE}) or rebuild against a .so that has "
                "the persistent-CTA entry."
            )
        # Wave-14: pass BpEB_noised (noised B) as the B operand; the matmul
        # kernel reads it as params.ptr_BpEB and uses it as the second factor.
        # Note ApEA_batch is wired via contexts[i, 0] (per-nonce override),
        # not via the A_batch positional arg here -- but we still pass A_batch
        # because the C++ shape check uses A_batch.size() to derive M, K.
        pg.gemm_persistent_multinonce(
            self.A_batch, self.BpEB_noised,
            self.A_scales_batch, self.B_scales,
            self.C_batch,
            self.contexts,
            self.nonces,
            # Shared scratch — same as the noisy_gemm signature. The host_signal
            # buffers below are the SHARED (slot-0) views; the kernel will read
            # the per-nonce slot pointers from the contexts table instead.
            self.EAL, self.EAL_fp16,
            self.EBR, self.EBR_fp16,
            self.EAR_R_major, self.EBL_R_major,
            self.EAR_K_major, self.EBL_K_major,
            self.AxEBL_fp16, self.EARxBpEB_fp16,
            self.ApEA, self.BpEB,
            self.host_signal_header, self.host_signal_sync,
            self.pow_target, self.pow_key,
            self.AxEBL_int32, self.EARxBpEB_int32,
            bM, bN, bK, cM, cN, matmul_stages,
            None, True,
            noise_tile_a_m, noise_tile_a_k,
            noise_tile_b_n, noise_tile_b_k,
            noise_stages_a, noise_stages_b,
            None, None,
            True, True,
            skip_reduction, skip_denoising,
            None, False,
        )

    # ------------------------------------------------------------------
    # Convenience: fill per-nonce A from the random source the
    # current driver uses. The "smoke" fill — production code will
    # replace this with a nonce-seeded noise-gen pass once that's
    # wired into the kernel agent's path.
    # ------------------------------------------------------------------

    def fill_smoke_A(self) -> None:
        """Refresh A_batch / A_scales_batch with random data (driver-parity).

        Mirrors the current `_miner_driver_sm89_r128.py` per-attempt random
        fill, but does it for all batch_size slots at once. Useful both for
        the bit-exact test (sequential vs batched compare on identical
        synthetic inputs) and as the bench fallback when the nonce-seeded
        fill kernel isn't yet integrated.
        """
        torch = self._torch
        B, M, K = self.cfg.batch_size, self.cfg.M, self.cfg.K
        self.A_batch.copy_(
            torch.randint(-127, 127, (B, M, K), dtype=torch.int8, device=self.device)
        )
        # Match the current driver's scale formula (rand * 0.02 + 0.005)
        self.A_scales_batch.copy_(
            torch.rand(B, M, dtype=torch.float32, device=self.device) * 0.02 + 0.005
        )

    def fill_smoke_A_single(self) -> None:
        """Wave-16 one-A-per-attempt fill: random A_batch[0] only.

        Generates ONE M×K matrix (the entire attempt's A) instead of B×M×K.
        At production B=256 M=N=2048 K=4096 this drops a 2 GiB torch.randint
        to an 8 MiB call — ~155 ms savings per attempt. The other 255 slots
        of A_batch / A_scales_batch are NOT written; they're already zero
        from __init__ and the kernel never reads them because the contexts
        table redirects every slot's A_ptr at slot 0 (see `_rebuild_contexts`).

        Use this when `is_one_A_per_attempt_enabled()` is True. Calling
        `fill_smoke_A()` instead in that mode is benign (it just wastes the
        time we wanted to save).
        """
        torch = self._torch
        M, K = self.cfg.M, self.cfg.K
        self.A_batch[0].copy_(
            torch.randint(-127, 127, (M, K), dtype=torch.int8, device=self.device)
        )
        # Match the current driver's scale formula (rand * 0.02 + 0.005)
        self.A_scales_batch[0].copy_(
            torch.rand(M, dtype=torch.float32, device=self.device) * 0.02 + 0.005
        )

    def fill_shared_B(self) -> None:
        """Refresh the shared B operand (random fill — smoke path).

        Called less often than fill_smoke_A — once per "epoch" of batches —
        since B is meant to persist across all 256 nonces of an attempt.
        Caller chooses when to rotate B.
        """
        torch = self._torch
        N, K = self.cfg.N, self.cfg.K
        self.B.copy_(
            torch.randint(-127, 127, (N, K), dtype=torch.int8, device=self.device)
        )
        self.B_scales.copy_(
            torch.rand(N, dtype=torch.float32, device=self.device) * 0.02 + 0.005
        )
